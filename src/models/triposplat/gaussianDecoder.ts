import type { GaussianAttributes } from '../ImageToGaussianModel'

export const TRIPOSPLAT_GAUSSIANS_PER_POINT = 32
export const TRIPOSPLAT_GS_FEATURE_WIDTH = 480

const XYZ_START = 0
const DC_START = 96
const SCALING_START = 192
const ROTATION_START = 288
const OPACITY_START = 416
const OFFSET_SCALE_START = 448
const OFFSET_AMPLITUDE = 0.75
const KERNEL_SIZE = 0.0009
const OPACITY_BIAS = Math.log(0.1 / 0.9)
const SCALE_BIAS = inverseSoftplus(0.004)
const OFFSET_SCALE_BIAS = inverseSoftplus(0.05)

const BASE_OFFSETS = buildBaseOffsets()

function inverseSoftplus(value: number): number {
  return Math.log(Math.expm1(value))
}

function softplus(value: number): number {
  if (value > 20) return value
  if (value < -20) return Math.exp(value)
  return Math.log1p(Math.exp(value))
}

function sigmoid(value: number): number {
  if (value >= 0) {
    const z = Math.exp(-value)
    return 1 / (1 + z)
  }
  const z = Math.exp(value)
  return z / (1 + z)
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

function buildBaseOffsets(): Float32Array {
  const values = new Float32Array(TRIPOSPLAT_GAUSSIANS_PER_POINT * 3)
  for (let gaussian = 0; gaussian < TRIPOSPLAT_GAUSSIANS_PER_POINT; gaussian += 1) {
    const samples = [
      gaussian / TRIPOSPLAT_GAUSSIANS_PER_POINT,
      radicalInverse(2, gaussian),
      radicalInverse(3, gaussian),
    ]
    for (let axis = 0; axis < 3; axis += 1) {
      values[gaussian * 3 + axis] = Math.atanh((samples[axis] * 2 - 1) / 1.5)
    }
  }
  return values
}

/** Apply the official ElasticGaussianFixedlenDecoder representation semantics. */
export function decodeTripoSplatGaussianFeatures(
  points: Float32Array,
  features: Float32Array,
): GaussianAttributes {
  if (points.length % 3 !== 0) throw new Error('TripoSplat points must contain xyz triplets.')
  const pointCount = points.length / 3
  if (features.length !== pointCount * TRIPOSPLAT_GS_FEATURE_WIDTH) {
    throw new Error(
      `TripoSplat GS feature length ${features.length} does not match ${pointCount}x${TRIPOSPLAT_GS_FEATURE_WIDTH}.`,
    )
  }

  const count = pointCount * TRIPOSPLAT_GAUSSIANS_PER_POINT
  const positions = new Float32Array(count * 3)
  const scales = new Float32Array(count * 3)
  const rotations = new Float32Array(count * 4)
  const sh0 = new Float32Array(count * 3)
  const opacities = new Float32Array(count)

  for (let point = 0; point < pointCount; point += 1) {
    const featureBase = point * TRIPOSPLAT_GS_FEATURE_WIDTH
    for (let gaussian = 0; gaussian < TRIPOSPLAT_GAUSSIANS_PER_POINT; gaussian += 1) {
      const outputIndex = point * TRIPOSPLAT_GAUSSIANS_PER_POINT + gaussian
      const output3 = outputIndex * 3
      const output4 = outputIndex * 4
      const learnedOffsetScale = softplus(features[featureBase + OFFSET_SCALE_START + gaussian] + OFFSET_SCALE_BIAS)

      for (let axis = 0; axis < 3; axis += 1) {
        const component = gaussian * 3 + axis
        const offset =
          Math.tanh(features[featureBase + XYZ_START + component] + BASE_OFFSETS[component]) *
          OFFSET_AMPLITUDE *
          learnedOffsetScale
        positions[output3 + axis] = points[point * 3 + axis] + offset - 0.5
        sh0[output3 + axis] = features[featureBase + DC_START + component]

        const positiveScale = softplus(features[featureBase + SCALING_START + component] + SCALE_BIAS)
        scales[output3 + axis] = Math.sqrt(positiveScale * positiveScale + KERNEL_SIZE * KERNEL_SIZE)
      }

      for (let component = 0; component < 4; component += 1) {
        rotations[output4 + component] =
          features[featureBase + ROTATION_START + gaussian * 4 + component] * 0.1 + (component === 0 ? 1 : 0)
      }
      opacities[outputIndex] = sigmoid(features[featureBase + OPACITY_START + gaussian] + OPACITY_BIAS)
    }
  }

  return { positions, scales, rotations, sh0, opacities }
}
