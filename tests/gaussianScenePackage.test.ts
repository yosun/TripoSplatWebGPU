import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createGaussianScene,
  GaussianSceneDisposedError,
  type CreateGaussianSceneInput,
  type GaussianSceneMetadata,
} from '../packages/gaussian-scene/dist/index.js'

const TRIPOSPLAT_EXPORT_TRANSFORM = [
  1, 0, 0,
  0, 0, -1,
  0, 1, 0,
] as const

function metadata(overrides: Partial<GaussianSceneMetadata> = {}): GaussianSceneMetadata {
  return {
    coordinateSystem: 'triposplat-object',
    units: 'normalized object units',
    rotationOrder: 'wxyz',
    scaleEncoding: 'linear',
    opacityEncoding: 'linear',
    colorSemantics: null,
    sphericalHarmonicsSemantics: 'degree-0-rgb',
    modelRevision: 'a78fa12d06dbf1381ca548bfac32bb68cb8c451d',
    generationSettings: { steps: 20, guidanceScale: 3, nested: { enabled: true } },
    seed: 42,
    runtimeVersion: '0.0.0-test',
    plyExportTransform: TRIPOSPLAT_EXPORT_TRANSFORM,
    ...overrides,
  }
}

function sceneInput(overrides: Partial<CreateGaussianSceneInput> = {}): CreateGaussianSceneInput {
  return {
    count: 1,
    positions: Float32Array.of(1, 2, 3),
    scales: Float32Array.of(1, Math.E, 0.5),
    rotations: Float32Array.of(1, 0, 0, 0),
    opacities: Float32Array.of(0.25),
    sphericalHarmonics: Float32Array.of(0.1, 0.2, 0.3),
    metadata: metadata(),
    ...overrides,
  }
}

function assertClose(actual: number, expected: number, tolerance = 1e-6): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  )
}

const PLY_PROPERTIES = [
  'x', 'y', 'z',
  'nx', 'ny', 'nz',
  'f_dc_0', 'f_dc_1', 'f_dc_2',
  'opacity',
  'scale_0', 'scale_1', 'scale_2',
  'rot_0', 'rot_1', 'rot_2', 'rot_3',
]

async function parseSingleVertex(blob: Blob): Promise<{ header: string; values: number[] }> {
  const bytes = new Uint8Array(await blob.arrayBuffer())
  const decoded = new TextDecoder().decode(bytes)
  const marker = 'end_header\n'
  const headerLength = decoded.indexOf(marker) + marker.length
  assert.ok(headerLength >= marker.length, 'PLY end_header marker is missing')
  const header = decoded.slice(0, headerLength)
  const view = new DataView(bytes.buffer, bytes.byteOffset + headerLength)
  const values = Array.from({ length: PLY_PROPERTIES.length }, (_unused, index) =>
    view.getFloat32(index * 4, true),
  )
  return { header, values }
}

test('factory owns defensive array and metadata copies', () => {
  const input = sceneInput()
  const originalSettings = input.metadata.generationSettings as {
    steps: number
    nested: { enabled: boolean }
  }
  const scene = createGaussianScene(input)

  input.positions[0] = 99
  input.sphericalHarmonics![0] = 99
  originalSettings.steps = 4
  originalSettings.nested.enabled = false

  assert.equal(scene.positions[0], 1)
  assertClose(scene.sphericalHarmonics![0], 0.1)
  assert.equal(scene.metadata.generationSettings.steps, 20)
  assert.deepEqual(scene.metadata.generationSettings.nested, { enabled: true })
  assert.ok(Object.isFrozen(scene.metadata))
  assert.ok(Object.isFrozen(scene.metadata.generationSettings))
  assert.ok(Object.isFrozen(scene.metadata.generationSettings.nested))
})

test('PLY Blob matches the existing 17-float TripoSplat export semantics', async () => {
  const scene = createGaussianScene(sceneInput())
  const blob = await scene.exportPLY()
  const { header, values } = await parseSingleVertex(blob)

  assert.equal(blob.type, 'application/octet-stream')
  assert.deepEqual(header.trimEnd().split('\n'), [
    'ply',
    'format binary_little_endian 1.0',
    'element vertex 1',
    ...PLY_PROPERTIES.map((name) => `property float ${name}`),
    'end_header',
  ])
  assert.deepEqual(values.slice(0, 6), [1, -3, 2, 0, 0, 0])
  assertClose(values[6], 0.1)
  assertClose(values[7], 0.2)
  assertClose(values[8], 0.3)
  assertClose(values[9], Math.log(0.25 / 0.75))
  assertClose(values[10], 0)
  assertClose(values[11], 1)
  assertClose(values[12], Math.log(0.5))
  assertClose(values[13], Math.SQRT1_2)
  assertClose(values[14], Math.SQRT1_2)
  assertClose(values[15], 0)
  assertClose(values[16], 0)
})

test('PLY export honors xyzw, pre-encoded values, and sRGB color semantics', async () => {
  const shC0 = 0.28209479177387814
  const scene = createGaussianScene(sceneInput({
    scales: Float32Array.of(Math.log(0.25), 0, Math.log(4)),
    rotations: Float32Array.of(0, 0, 0, 2),
    opacities: Float32Array.of(-2),
    sphericalHarmonics: undefined,
    colors: Float32Array.of(0.5, 0.75, 0.25),
    metadata: metadata({
      coordinateSystem: 'viewer-world',
      rotationOrder: 'xyzw',
      scaleEncoding: 'log',
      opacityEncoding: 'logit',
      colorSemantics: 'srgb',
      sphericalHarmonicsSemantics: null,
      plyExportTransform: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    }),
  }))
  const { values } = await parseSingleVertex(await scene.exportPLY())

  assert.deepEqual(values.slice(0, 3), [1, 2, 3])
  assertClose(values[6], 0)
  assertClose(values[7], 0.25 / shC0)
  assertClose(values[8], -0.25 / shC0)
  assertClose(values[9], -2)
  assertClose(values[10], Math.log(0.25))
  assertClose(values[11], 0)
  assertClose(values[12], Math.log(4))
  assert.deepEqual(values.slice(13), [1, 0, 0, 0])
})

test('splat export emits the compatible 32-byte browser layout', async () => {
  const scene = createGaussianScene(sceneInput())
  const blob = await scene.exportSplat()
  const bytes = new Uint8Array(await blob.arrayBuffer())
  const view = new DataView(bytes.buffer)

  assert.equal(blob.type, 'application/octet-stream')
  assert.equal(bytes.byteLength, 32)
  assert.deepEqual(
    [view.getFloat32(0, true), view.getFloat32(4, true), view.getFloat32(8, true)],
    [1, -3, 2],
  )
  assertClose(view.getFloat32(12, true), 1)
  assertClose(view.getFloat32(16, true), Math.E)
  assertClose(view.getFloat32(20, true), 0.5)
  assert.deepEqual(Array.from(bytes.slice(24, 28)), [134, 141, 149, 63])
  assert.deepEqual(Array.from(bytes.slice(28, 32)), [218, 218, 128, 128])
})

test('splat export is byte-identical to the official TripoSplat exporter fixture', async () => {
  const scene = createGaussianScene(sceneInput({
    count: 4,
    positions: Float32Array.of(
      1.25, -2.5, 3.75,
      -4, 5.5, -6.25,
      0.125, -0.25, 0.5,
      7, -8, 9,
    ),
    scales: Float32Array.of(
      0.5, 0.25, 2,
      1.5, 2, 0.5,
      0.2, 0.3, 0.4,
      3, 0.25, 0.5,
    ),
    rotations: Float32Array.of(
      1, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5,
      2, -1, 0.5, 0.25,
      0.3, 0.2, -0.8, 0.1,
    ),
    opacities: Float32Array.of(0.8, 0.4, 0.95, 0.6),
    sphericalHarmonics: Float32Array.of(
      0, 0.1, -0.1,
      2, -2, 0.25,
      -3, 3, 0,
      0.333, -0.777, 1.234,
    ),
  }))

  const actual = new Uint8Array(await (await scene.exportSplat()).arrayBuffer())
  // Generated once by Gaussian.to_splat_bytes() at official TripoSplat commit
  // a78fa12d06dbf1381ca548bfac32bb68cb8c451d. Keeping the oracle static makes
  // this normal Node test independent of Python, NumPy, Torch, and host endianness.
  const expected = Uint8Array.from(Buffer.from(
    '000080c00000c8400000b0400000c03f000000400000003fff00916680da80da'
      + '0000e040000010c1000000c1000040400000803e0000003f9747d8998ab32338'
      + '0000a03f000070c0000020c00000003f0000803e000000407f8678ccdada8080'
      + '0000003e000000bf000080becdcc4c3e9a99993ecdcccc3e00ff7ff2f5a7899d',
    'hex',
  ))

  assert.deepEqual(actual, expected)
})

test('splat export follows official opacity-volume ordering', async () => {
  const scene = createGaussianScene(sceneInput({
    count: 2,
    positions: Float32Array.of(1, 0, 0, 2, 0, 0),
    scales: Float32Array.of(1, 1, 1, 2, 2, 2),
    rotations: Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0),
    opacities: Float32Array.of(0.9, 0.2),
    sphericalHarmonics: Float32Array.of(0, 0, 0, 0, 0, 0),
  }))
  const bytes = new Uint8Array(await (await scene.exportSplat()).arrayBuffer())
  const view = new DataView(bytes.buffer)

  // Scores are .9 * 1 and .2 * 8, so the second source splat is written first.
  assert.equal(view.getFloat32(0, true), 2)
  assert.equal(view.getFloat32(32, true), 1)
})

test('splat export preserves official float32 quaternion and SH byte boundaries', async () => {
  const scene = createGaussianScene(sceneInput({
    positions: Float32Array.of(0, 0, 0),
    scales: Float32Array.of(1, 1, 1),
    rotations: Float32Array.of(
      -1.05434835,
      -1.05440962,
      -1.08684886,
      0.67140311,
    ),
    opacities: Float32Array.of(0.5),
    sphericalHarmonics: Float32Array.of(-1.0773738623, 0, 0),
  }))
  const bytes = new Uint8Array(await (await scene.exportSplat()).arrayBuffer())

  // Oracle bytes from official NumPy float32 transform/quantization. The first
  // color is a threshold case; the quaternion is a trace-rounding sign case.
  assert.deepEqual(Array.from(bytes.slice(24, 32)), [49, 127, 127, 127, 127, 225, 209, 147])
})

test('splat sort score uses official float32 products', async () => {
  const scene = createGaussianScene(sceneInput({
    count: 2,
    positions: Float32Array.of(1, 0, 0, 2, 0, 0),
    scales: Float32Array.of(
      1.0779575109481812, 1.0959299802780151, 1.0772638320922852,
      1.0525925159454346, 1.0850036144256592, 1.0478612184524536,
    ),
    rotations: Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0),
    opacities: Float32Array.of(0.7857662439346313, 0.8356119990348816),
    sphericalHarmonics: Float32Array.of(0, 0, 0, 0, 0, 0),
  }))
  const bytes = new Uint8Array(await (await scene.exportSplat()).arrayBuffer())
  const view = new DataView(bytes.buffer)

  assert.equal(view.getFloat32(0, true), 1)
  assert.equal(view.getFloat32(32, true), 2)
})

test('factory rejects inconsistent arrays, encodings, appearance metadata, and transforms', () => {
  assert.throws(
    () => createGaussianScene(sceneInput({ positions: new Float32Array(2) })),
    /positions has 2 values; expected 3/,
  )
  assert.throws(
    () => createGaussianScene(sceneInput({ scales: Float32Array.of(1, 0, 1) })),
    /scales\[1\] must be greater than zero/,
  )
  assert.throws(
    () => createGaussianScene(sceneInput({ opacities: Float32Array.of(1.1) })),
    /opacities\[0\] must be in \[0, 1\]/,
  )
  assert.throws(
    () => createGaussianScene(sceneInput({ rotations: new Float32Array(4) })),
    /must have non-zero norm/,
  )
  assert.throws(
    () => createGaussianScene(sceneInput({
      sphericalHarmonics: undefined,
      metadata: metadata(),
    })),
    /sphericalHarmonics and metadata.sphericalHarmonicsSemantics/,
  )
  assert.throws(
    () => createGaussianScene(sceneInput({
      metadata: metadata({ plyExportTransform: [1, 0, 0, 0, 1, 0, 0, 0, -1] }),
    })),
    /must preserve orientation/,
  )
})

test('dispose is idempotent, releases owned arrays, and rejects later exports', async () => {
  const scene = createGaussianScene(sceneInput())
  scene.dispose()
  scene.dispose()

  assert.equal(scene.isDisposed, true)
  assert.equal(scene.count, 0)
  assert.equal(scene.positions.length, 0)
  assert.equal(scene.scales.length, 0)
  assert.equal(scene.rotations.length, 0)
  assert.equal(scene.opacities.length, 0)
  assert.equal(scene.sphericalHarmonics, undefined)
  assert.equal(scene.colors, undefined)
  await assert.rejects(
    scene.exportPLY(),
    (error: unknown) =>
      error instanceof GaussianSceneDisposedError &&
      error.code === 'GAUSSIAN_SCENE_DISPOSED',
  )
  await assert.rejects(scene.exportSplat(), GaussianSceneDisposedError)
})
