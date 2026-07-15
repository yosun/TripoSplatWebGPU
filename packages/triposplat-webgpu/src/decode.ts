import { createGaussianScene, type GaussianScene, type GaussianSceneMetadata } from '@ai3d/gaussian-scene'

export const GAUSSIANS_PER_POINT = 32
export const GAUSSIAN_FEATURE_WIDTH = 480

const OPACITY_BIAS = Math.log(0.1 / 0.9)
const SCALE_BIAS = Math.log(Math.expm1(0.004))
const OFFSET_SCALE_BIAS = Math.log(Math.expm1(0.05))
const BASE_OFFSETS = buildBaseOffsets()

function softplus(value: number): number {
  if (value > 20) return value
  if (value < -20) return Math.exp(value)
  return Math.log1p(Math.exp(value))
}

function sigmoid(value: number): number {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value))
}

function radicalInverse(base: number, value: number): number {
  let output = 0
  let inversePower = 1 / base
  while (value > 0) {
    output += (value % base) * inversePower
    value = Math.floor(value / base)
    inversePower /= base
  }
  return output
}

function buildBaseOffsets(): Float32Array {
  const offsets = new Float32Array(GAUSSIANS_PER_POINT * 3)
  for (let index = 0; index < GAUSSIANS_PER_POINT; index += 1) {
    const values = [index / GAUSSIANS_PER_POINT, radicalInverse(2, index), radicalInverse(3, index)]
    for (let axis = 0; axis < 3; axis += 1) {
      offsets[index * 3 + axis] = Math.atanh((values[axis] * 2 - 1) / 1.5)
    }
  }
  return offsets
}

export interface DecodeGaussiansOptions {
  metadata: GaussianSceneMetadata
}

/** Decode the official fixed-length ElasticGaussian 480-feature representation. */
export function decodeGaussians(
  points: Float32Array,
  features: Float32Array,
  options: DecodeGaussiansOptions,
): GaussianScene {
  if (points.length % 3 !== 0) throw new RangeError('points must contain xyz triplets.')
  const pointCount = points.length / 3
  if (features.length !== pointCount * GAUSSIAN_FEATURE_WIDTH) {
    throw new RangeError(`features must contain ${pointCount * GAUSSIAN_FEATURE_WIDTH} values.`)
  }
  const count = pointCount * GAUSSIANS_PER_POINT
  const positions = new Float32Array(count * 3)
  const scales = new Float32Array(count * 3)
  const rotations = new Float32Array(count * 4)
  const sphericalHarmonics = new Float32Array(count * 3)
  const opacities = new Float32Array(count)
  for (let point = 0; point < pointCount; point += 1) {
    const featureBase = point * GAUSSIAN_FEATURE_WIDTH
    for (let gaussian = 0; gaussian < GAUSSIANS_PER_POINT; gaussian += 1) {
      const output = point * GAUSSIANS_PER_POINT + gaussian
      const output3 = output * 3
      const output4 = output * 4
      const learnedOffsetScale = softplus(features[featureBase + 448 + gaussian] + OFFSET_SCALE_BIAS)
      for (let axis = 0; axis < 3; axis += 1) {
        const component = gaussian * 3 + axis
        positions[output3 + axis] = points[point * 3 + axis]
          + Math.tanh(features[featureBase + component] + BASE_OFFSETS[component]) * 0.75 * learnedOffsetScale
          - 0.5
        sphericalHarmonics[output3 + axis] = features[featureBase + 96 + component]
        const scale = softplus(features[featureBase + 192 + component] + SCALE_BIAS)
        scales[output3 + axis] = Math.sqrt(scale * scale + 0.0009 ** 2)
      }
      for (let component = 0; component < 4; component += 1) {
        rotations[output4 + component] = features[featureBase + 288 + gaussian * 4 + component] * 0.1
          + (component === 0 ? 1 : 0)
      }
      opacities[output] = sigmoid(features[featureBase + 416 + gaussian] + OPACITY_BIAS)
    }
  }
  return createGaussianScene({
    count,
    positions,
    scales,
    rotations,
    opacities,
    sphericalHarmonics,
    metadata: options.metadata,
  })
}
