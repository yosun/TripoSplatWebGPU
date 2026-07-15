import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildTripoSplatEncoderTensors,
  compositeRgbaOnBlack,
  erodeAlpha,
  resizeRgbaLanczos,
  type RgbaImage,
} from '../src/models/triposplat/preprocess.ts'
import {
  createShiftedFlowSchedule,
  sampleFlow4Steps,
  type FlowModelInvocation,
} from '../src/models/triposplat/flowSampler.ts'
import {
  buildOctreeParentCenters,
  expandOctreeFrontier,
  expandOctreeLeavesToPoints,
  octreeSoftmax,
  sampleOctree,
  systematicResample,
  TRIPOSPLAT_OCTREE_LEVELS,
  type OctreeFrontier,
} from '../src/models/triposplat/octree.ts'

function assertFloatArrayClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance = 1e-6,
): void {
  assert.equal(actual.length, expected.length)
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `index ${index}: expected ${actual[index]} to be within ${tolerance} of ${expected[index]}`,
    )
  }
}

test('RGBA Lanczos resize follows Pillow premultiplied-alpha behavior', () => {
  const source: RgbaImage = {
    width: 2,
    height: 1,
    data: Uint8ClampedArray.from([255, 0, 0, 0, 0, 0, 255, 255]),
  }
  const resized = resizeRgbaLanczos(source, 8, 1)
  assert.deepEqual([...resized.data], [
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 255, 28,
    0, 0, 255, 93,
    0, 0, 255, 162,
    0, 0, 255, 227,
    0, 0, 255, 255,
    0, 0, 255, 255,
  ])
})

test('alpha erosion and black composition match Pillow integer semantics', () => {
  const image: RgbaImage = { width: 5, height: 5, data: new Uint8ClampedArray(5 * 5 * 4) }
  for (let pixel = 0; pixel < 25; pixel += 1) image.data[pixel * 4 + 3] = pixel
  const eroded = erodeAlpha(image, 1)
  const alpha = [...eroded.data].filter((_value, index) => index % 4 === 3)
  assert.deepEqual(alpha, [
    0, 0, 1, 2, 3,
    0, 0, 1, 2, 3,
    5, 5, 6, 7, 8,
    10, 10, 11, 12, 13,
    15, 15, 16, 17, 18,
  ])
  assert.equal(
    compositeRgbaOnBlack({
      width: 1,
      height: 1,
      data: Uint8ClampedArray.from([127, 0, 0, 127]),
    }).data[0],
    63,
  )
})

test('encoder tensor builders produce NCHW RGB, DINO, and VAE domains', () => {
  const tensors = buildTripoSplatEncoderTensors({
    width: 1,
    height: 1,
    data: Uint8ClampedArray.from([0, 127, 255]),
  })
  assert.deepEqual(tensors.rgb.dims, [1, 3, 1, 1])
  assert.equal(tensors.rgb.data[0], 0)
  assert.ok(Math.abs(tensors.rgb.data[1] - 127 / 255) < 1e-7)
  assert.equal(tensors.rgb.data[2], 1)
  assert.ok(Math.abs(tensors.dinov3.data[0] - (0 - 0.485) / 0.229) < 1e-6)
  assert.ok(Math.abs(tensors.vae.data[1] - (2 * 127) / 255 + 1) < 1e-6)
  assert.deepEqual([...tensors.vae.data], [-1, tensors.vae.data[1], 1])
})

test('shifted schedule and four-step CFG Euler loop match official equations', async () => {
  assert.deepEqual(
    createShiftedFlowSchedule(4, 3).map(({ timestep }) => timestep).concat(0),
    [1, 0.9, 0.75, 0.5, 0],
  )
  const calls: FlowModelInvocation<string>[] = []
  const noise = { latent: Float32Array.of(10) }
  const result = await sampleFlow4Steps(
    (invocation) => {
      calls.push(invocation)
      return { latent: Float32Array.of(invocation.pass === 'conditional' ? 2 : 1) }
    },
    noise,
    { condition: 'image', negativeCondition: 'zeros', guidanceScale: 3, shift: 3 },
  )
  assert.equal(calls.length, 8)
  assert.equal(calls[0].timestepTensor[0], 1000)
  assert.ok(Math.abs(result.latent[0] - 6) < 1e-6)
  assert.equal(noise.latent[0], 10)
})

test('systematic resampling preserves exact counts', () => {
  assert.deepEqual([...systematicResample([0.1, 0.2, 0.7], 10, () => 0.5)], [1, 2, 7])
  assert.deepEqual([...systematicResample([0, 0, 0, 0, 0, 0, 0, 0], 8, () => 0.5)], [
    1, 1, 1, 1, 1, 1, 1, 1,
  ])
})

test('systematic resampling matches official underfull, zero-row, and CDF-boundary behavior', () => {
  // Official sample_probs divides by row_sums.clamp_min_(1), so a positive row
  // below unit mass is not scaled up. Any targets past its CDF clamp to the last child.
  assert.deepEqual(
    [...systematicResample([0.2, 0.2, 0, 0, 0, 0, 0, 0], 4, () => 0.5)],
    [1, 1, 0, 0, 0, 0, 0, 2],
  )
  assert.deepEqual([...systematicResample([2, 2], 4, () => 0.5)], [2, 2])
  // Negative entries clamp to zero; a resulting all-zero row is replaced by uniform mass.
  assert.deepEqual([...systematicResample([-4, -1, 0], 3, () => 0.5)], [1, 1, 1])
  // torch.searchsorted(..., right=False) assigns an exact CDF boundary to the left bin.
  assert.deepEqual([...systematicResample([0.5, 0.5], 1, () => 0.5)], [1, 0])
  assert.deepEqual([...systematicResample([0, 1], 1, () => 0)], [1, 0])

  let rngCalls = 0
  assert.deepEqual([...systematicResample([0.5, 0.5], 0, () => {
    rngCalls += 1
    return 0.5
  })], [0, 0])
  assert.equal(rngCalls, 0)
})

test('octree systematic offsets follow official ascending count-group RNG order', () => {
  const random = [0.9, 0.1]
  const result = expandOctreeFrontier(
    {
      coordinates: Int32Array.of(0, 0, 0, 1, 0, 0),
      counts: Uint32Array.of(2, 1),
      logProbabilities: Float32Array.of(0, 0),
    },
    {
      probabilities: Float32Array.of(
        0.5, 0.5, 0, 0, 0, 0, 0, 0,
        0.5, 0.5, 0, 0, 0, 0, 0, 0,
      ),
      logProbabilities: new Float32Array(16),
    },
    () => random.shift() ?? 0,
  )
  assert.deepEqual([...result.coordinates], [0, 0, 0, 1, 0, 0, 3, 0, 0])
  assert.deepEqual([...result.counts], [1, 1, 1])
})

test('octree expansion preserves parent-major compaction and x-fastest child order', () => {
  const frontier: OctreeFrontier = {
    coordinates: Int32Array.of(0, 0, 0, 1, 0, 1),
    counts: Uint32Array.of(4, 3),
    logProbabilities: Float32Array.of(-0.25, -1),
  }
  const probabilities = Float32Array.of(
    0.25, 0.25, 0, 0, 0, 0, 0, 0.5,
    0, 0, 1, 0, 0, 0, 0, 0,
  )
  const logProbabilities = Float32Array.from(probabilities, (probability) =>
    probability === 0 ? -100 : Math.log(probability),
  )
  const expanded = expandOctreeFrontier(
    frontier,
    { probabilities, logProbabilities },
    () => 0.5,
  )

  // Official child_offset order is [x, y, z] with x changing fastest.
  assert.deepEqual([...expanded.coordinates], [
    0, 0, 0,
    1, 0, 0,
    1, 1, 1,
    2, 1, 2,
  ])
  assert.deepEqual([...expanded.counts], [1, 1, 2, 3])
  assertFloatArrayClose(expanded.logProbabilities, [
    -0.25 + Math.log(0.25),
    -0.25 + Math.log(0.25),
    -0.25 + Math.log(0.5),
    -1,
  ])
})

test('octree leaf expansion repeats compact leaves and jitters xyz in official order', () => {
  const leaves: OctreeFrontier = {
    coordinates: Int32Array.of(1, 2, 3, 0, 0, 0),
    counts: Uint32Array.of(2, 1),
    logProbabilities: Float32Array.of(-0.5, -1),
  }
  assert.deepEqual([...buildOctreeParentCenters(leaves, 4)], [
    0.375, 0.625, 0.875,
    0.125, 0.125, 0.125,
  ])
  const randomValues = [0, 0.25, 0.5, 0.75, 0.125, 0.875, 0.5, 0.5, 0.5]
  let randomIndex = 0
  const expanded = expandOctreeLeavesToPoints(leaves, 4, () => randomValues[randomIndex++])

  assert.deepEqual([...expanded.points], [
    0.25, 0.5625, 0.875,
    0.4375, 0.53125, 0.96875,
    0.125, 0.125, 0.125,
  ])
  assert.deepEqual([...expanded.logProbabilities], [-0.5, -0.5, -1])
  assert.equal(randomIndex, 9)
})

test('octree softmax is row-local, temperature-scaled, and numerically stable', () => {
  const distribution = octreeSoftmax(Float32Array.of(
    1000, 999, 998, 997, 996, 995, 994, 993,
    -4, -4, -4, -4, -4, -4, -4, -4,
  ), 2, 2)

  const firstSum = distribution.probabilities.slice(0, 8).reduce((sum, value) => sum + value, 0)
  const secondSum = distribution.probabilities.slice(8).reduce((sum, value) => sum + value, 0)
  assert.ok(Math.abs(firstSum - 1) < 1e-6)
  assert.ok(Math.abs(secondSum - 1) < 1e-6)
  assert.deepEqual([...distribution.probabilities.slice(8)], new Array(8).fill(0.125))
  for (let index = 0; index < distribution.probabilities.length; index += 1) {
    assert.ok(Math.abs(
      Math.log(distribution.probabilities[index]) - distribution.logProbabilities[index]
    ) < 1e-6)
  }
})

test('eight-level octree forced path matches the official compaction fixture exactly', async () => {
  // Recorded from OctreeProbabilityFixedlenDecoder.sample at official commit
  // a78fa12d06dbf1381ca548bfac32bb68cb8c451d with level N forced to child N-1
  // and both torch.rand/systematic offsets and final torch.rand_like jitter fixed at 0.5.
  const expectedCenters = [
    [0.5, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.375, 0.125, 0.125],
    [0.3125, 0.1875, 0.0625],
    [0.34375, 0.21875, 0.03125],
    [0.328125, 0.203125, 0.046875],
    [0.3359375, 0.1953125, 0.0546875],
    [0.33203125, 0.19921875, 0.05859375],
  ]
  const progress: Array<[number, number, number]> = []
  let rngCalls = 0
  const result = await sampleOctree(
    ({ parentCenters, parentCount, level, resolution, targetPointCount }) => {
      assert.equal(parentCount, 1)
      assert.equal(resolution, 2 ** level)
      assert.equal(targetPointCount, 3)
      assert.deepEqual([...parentCenters], expectedCenters[level - 1])
      const logits = new Float32Array(8).fill(-100)
      logits[level - 1] = 0
      return { logits }
    },
    {
      condition: 'fixture',
      numPoints: 3,
      levels: TRIPOSPLAT_OCTREE_LEVELS,
      rng: () => {
        rngCalls += 1
        return 0.5
      },
      onLevel: ({ level, resolution, occupiedVoxels }) => {
        progress.push([level, resolution, occupiedVoxels])
      },
    },
  )

  assert.deepEqual([...result.leaves.coordinates], [85, 51, 15])
  assert.deepEqual([...result.leaves.counts], [3])
  assert.deepEqual([...result.leaves.logProbabilities], [0])
  assert.deepEqual([...result.points], [
    0.333984375, 0.201171875, 0.060546875,
    0.333984375, 0.201171875, 0.060546875,
    0.333984375, 0.201171875, 0.060546875,
  ])
  assert.deepEqual([...result.logProbabilities], [0, 0, 0])
  assert.deepEqual(progress, Array.from({ length: 8 }, (_unused, index) => [
    index + 1,
    2 ** (index + 1),
    1,
  ]))
  // One systematic offset per level, then one xyz jitter triplet per final point.
  assert.equal(rngCalls, 8 + 3 * 3)
})

test('eight-level nontrivial octree fixture matches official sampling and path log probabilities', async () => {
  // Independent oracle from the same official commit: seven samples, logits
  // (((parent*5 + child*3 + level*2) % 11) - 5) / 2, systematic U=0.5,
  // and final xyz jitter 0.25. The official active-parent counts are 1,4,7,7,7,7,7,7.
  const parentCounts: number[] = []
  let rngCalls = 0
  const result = await sampleOctree(
    ({ parentCount, level }) => {
      parentCounts.push(parentCount)
      const logits = new Float32Array(parentCount * 8)
      for (let parent = 0; parent < parentCount; parent += 1) {
        for (let child = 0; child < 8; child += 1) {
          logits[parent * 8 + child] = (
            ((parent * 5 + child * 3 + level * 2) % 11) - 5
          ) / 2
        }
      }
      return { logits }
    },
    {
      condition: null,
      numPoints: 7,
      levels: 8,
      // The eight frontiers consume 1+4+6*7 = 47 systematic offsets.
      rng: () => rngCalls++ < 47 ? 0.5 : 0.25,
    },
  )

  assert.deepEqual(parentCounts, [1, 4, 7, 7, 7, 7, 7, 7])
  assert.equal(rngCalls, 47 + 7 * 3)
  assert.deepEqual([...result.leaves.counts], new Array(7).fill(1))
  assert.deepEqual([...result.leaves.coordinates], [
    175, 76, 51,
    21, 169, 54,
    50, 133, 110,
    254, 48, 205,
    23, 166, 153,
    10, 148, 251,
    57, 194, 247,
  ])
  assert.deepEqual([...result.points], [
    0.6845703125, 0.2978515625, 0.2001953125,
    0.0830078125, 0.6611328125, 0.2119140625,
    0.1962890625, 0.5205078125, 0.4306640625,
    0.9931640625, 0.1884765625, 0.8017578125,
    0.0908203125, 0.6494140625, 0.5986328125,
    0.0400390625, 0.5791015625, 0.9814453125,
    0.2236328125, 0.7587890625, 0.9658203125,
  ])
  assertFloatArrayClose(result.logProbabilities, [
    -10.276244163513184,
    -11.039340019226074,
    -9.35226058959961,
    -11.298528671264648,
    -8.880032539367676,
    -9.087876319885254,
    -13.00637149810791,
  ], 2e-6)
})

test('eight-level octree loop keeps numPoints and jitters inside level-8 voxels', async () => {
  const levelsSeen: number[] = []
  const result = await sampleOctree(
    ({ parentCount, level }) => {
      levelsSeen.push(level)
      return { logits: new Float32Array(parentCount * 8) }
    },
    {
      condition: null,
      numPoints: 8,
      levels: TRIPOSPLAT_OCTREE_LEVELS,
      rng: () => 0.5,
    },
  )
  assert.deepEqual(levelsSeen, [1, 2, 3, 4, 5, 6, 7, 8])
  assert.equal(result.resolution, 256)
  assert.equal(result.points.length, 8 * 3)
  assert.equal(result.logProbabilities.length, 8)
  assert.equal(
    [...result.leaves.counts].reduce((sum, count) => sum + count, 0),
    8,
  )
  for (const coordinate of result.points) {
    assert.ok(coordinate >= 0 && coordinate < 1)
  }
})
