import assert from 'node:assert/strict'
import test from 'node:test'

import { buildGaussianPly } from '../src/lib/gaussianPly.ts'
import type { GaussianAttributes } from '../src/models/ImageToGaussianModel.ts'
import {
  decodeTripoSplatGaussianFeatures,
  TRIPOSPLAT_GAUSSIANS_PER_POINT,
  TRIPOSPLAT_GS_FEATURE_WIDTH,
} from '../src/models/triposplat/gaussianDecoder.ts'

const XYZ_START = 0
const DC_START = 96
const SCALING_START = 192
const ROTATION_START = 288
const OPACITY_START = 416
const OFFSET_SCALE_START = 448

function assertClose(actual: number, expected: number, tolerance = 1e-6): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  )
}

function inverseSoftplus(value: number): number {
  return Math.log(Math.expm1(value))
}

function softplus(value: number): number {
  return Math.log1p(Math.exp(value))
}

function radicalInverse(base: number, value: number): number {
  let result = 0
  let inversePower = 1 / base
  let remaining = value
  while (remaining > 0) {
    result += (remaining % base) * inversePower
    remaining = Math.floor(remaining / base)
    inversePower /= base
  }
  return result
}

test('Gaussian feature decoder honors the 480-wide layout and all output activations', () => {
  const point = Float32Array.of(1, 2, 3)
  const features = new Float32Array(TRIPOSPLAT_GS_FEATURE_WIDTH)
  const gaussian = 7
  const component = gaussian * 3

  features.set([0.2, -0.3, 0.4], XYZ_START + component)
  features.set([0.11, 0.22, 0.33], DC_START + component)

  const targetScales = [0.01, 0.02, 0.03]
  for (let axis = 0; axis < 3; axis += 1) {
    features[SCALING_START + component + axis] =
      inverseSoftplus(targetScales[axis]) - inverseSoftplus(0.004)
  }
  features.set([1, 2, 3, 4], ROTATION_START + gaussian * 4)
  features[OPACITY_START + gaussian] = Math.log(9)
  features[OFFSET_SCALE_START + gaussian] = inverseSoftplus(0.1) - inverseSoftplus(0.05)

  const decoded = decodeTripoSplatGaussianFeatures(point, features)
  assert.equal(decoded.opacities.length, TRIPOSPLAT_GAUSSIANS_PER_POINT)
  assert.equal(decoded.positions.length, TRIPOSPLAT_GAUSSIANS_PER_POINT * 3)
  assert.equal(decoded.rotations.length, TRIPOSPLAT_GAUSSIANS_PER_POINT * 4)

  const output3 = gaussian * 3
  const output4 = gaussian * 4
  const learnedOffsetScale = softplus(
    features[OFFSET_SCALE_START + gaussian] + inverseSoftplus(0.05),
  )
  const samples = [
    gaussian / TRIPOSPLAT_GAUSSIANS_PER_POINT,
    radicalInverse(2, gaussian),
    radicalInverse(3, gaussian),
  ]
  for (let axis = 0; axis < 3; axis += 1) {
    const baseOffset = Math.atanh((samples[axis] * 2 - 1) / 1.5)
    const expectedPosition =
      point[axis] +
      Math.tanh(features[XYZ_START + component + axis] + baseOffset) *
        0.75 *
        learnedOffsetScale -
      0.5
    assertClose(decoded.positions[output3 + axis], expectedPosition)
    assertClose(decoded.sh0[output3 + axis], features[DC_START + component + axis])

    const positiveScale = softplus(
      features[SCALING_START + component + axis] + inverseSoftplus(0.004),
    )
    assertClose(
      decoded.scales[output3 + axis],
      Math.hypot(positiveScale, 0.0009),
      1e-7,
    )
  }
  assertClose(decoded.rotations[output4], 1.1)
  assertClose(decoded.rotations[output4 + 1], 0.2)
  assertClose(decoded.rotations[output4 + 2], 0.3)
  assertClose(decoded.rotations[output4 + 3], 0.4)
  assertClose(decoded.opacities[gaussian], 0.5)

  // Untouched feature channels recover the decoder's official biases.
  assertClose(decoded.opacities[0], 0.1)
  assertClose(decoded.scales[0], Math.hypot(0.004, 0.0009), 1e-7)
  assert.deepEqual([...decoded.rotations.subarray(0, 4)], [1, 0, 0, 0])
})

test('Gaussian decoder validates point triplets and feature width', () => {
  assert.throws(
    () => decodeTripoSplatGaussianFeatures(Float32Array.of(1, 2), new Float32Array(0)),
    /xyz triplets/,
  )
  assert.throws(
    () => decodeTripoSplatGaussianFeatures(Float32Array.of(1, 2, 3), new Float32Array(479)),
    /does not match 1x480/,
  )
})

const PLY_PROPERTIES = [
  'x', 'y', 'z',
  'nx', 'ny', 'nz',
  'f_dc_0', 'f_dc_1', 'f_dc_2',
  'opacity',
  'scale_0', 'scale_1', 'scale_2',
  'rot_0', 'rot_1', 'rot_2', 'rot_3',
]

function parseSingleVertexPly(bytes: Uint8Array): { header: string; values: number[] } {
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

test('binary Gaussian PLY has the standard header and official TripoSplat transform', () => {
  const attributes: GaussianAttributes = {
    positions: Float32Array.of(1, 2, 3),
    scales: Float32Array.of(1, Math.E, 0.5),
    rotations: Float32Array.of(1, 0, 0, 0),
    sh0: Float32Array.of(0.1, 0.2, 0.3),
    opacities: Float32Array.of(0.25),
  }
  const bytes = buildGaussianPly(attributes)
  const { header, values } = parseSingleVertexPly(bytes)

  assert.deepEqual(header.trimEnd().split('\n'), [
    'ply',
    'format binary_little_endian 1.0',
    'element vertex 1',
    ...PLY_PROPERTIES.map((name) => `property float ${name}`),
    'end_header',
  ])
  assert.equal(bytes.length, new TextEncoder().encode(header).length + 17 * 4)
  assert.deepEqual(values.slice(0, 6), [1, -3, 2, 0, 0, 0])
  assertClose(values[6], 0.1)
  assertClose(values[7], 0.2)
  assertClose(values[8], 0.3)
  assertClose(values[9], Math.log(0.25 / 0.75))
  assertClose(values[10], 0)
  assertClose(values[11], 1)
  assertClose(values[12], Math.log(0.5))

  // The export basis maps identity to a +90 degree rotation about X.
  assertClose(values[13], Math.SQRT1_2)
  assertClose(values[14], Math.SQRT1_2)
  assertClose(values[15], 0)
  assertClose(values[16], 0)
})

test('PLY export clamps probability and scale singularities and validates array lengths', () => {
  const attributes: GaussianAttributes = {
    positions: Float32Array.of(0, 0, 0),
    scales: Float32Array.of(0, -1, 1e-30),
    rotations: Float32Array.of(0, 0, 0, 0),
    sh0: Float32Array.of(0, 0, 0),
    opacities: Float32Array.of(0),
  }
  const { values } = parseSingleVertexPly(buildGaussianPly(attributes))
  assertClose(values[9], Math.log(1e-6 / (1 - 1e-6)), 1e-5)
  assertClose(values[10], Math.log(1e-12), 1e-5)
  assertClose(values[11], Math.log(1e-12), 1e-5)
  assertClose(values[12], Math.log(1e-12), 1e-5)
  assert.ok(values.slice(13).every(Number.isFinite))

  assert.throws(
    () => buildGaussianPly({ ...attributes, positions: new Float32Array(2) }),
    /positions length/,
  )
})
