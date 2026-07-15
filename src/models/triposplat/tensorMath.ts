export function layerNormRows(
  input: Float32Array,
  width: number,
  epsilon = 1e-5,
): Float32Array {
  if (!Number.isInteger(width) || width <= 0 || input.length % width !== 0) {
    throw new Error(`layerNormRows width ${width} does not divide input length ${input.length}.`)
  }
  const output = new Float32Array(input.length)
  for (let offset = 0; offset < input.length; offset += width) {
    let mean = 0
    for (let index = 0; index < width; index += 1) mean += input[offset + index]
    mean /= width
    let variance = 0
    for (let index = 0; index < width; index += 1) {
      const centered = input[offset + index] - mean
      variance += centered * centered
    }
    const inverseStandardDeviation = 1 / Math.sqrt(variance / width + epsilon)
    for (let index = 0; index < width; index += 1) {
      output[offset + index] = (input[offset + index] - mean) * inverseStandardDeviation
    }
  }
  return output
}

export interface TensorComparison {
  count: number
  maxAbsoluteError: number
  meanAbsoluteError: number
  rmse: number
  cosineSimilarity: number
  finite: boolean
}

export function compareFloat32(reference: Float32Array, candidate: Float32Array): TensorComparison {
  if (reference.length !== candidate.length) {
    throw new Error(`Tensor lengths differ: reference=${reference.length}, candidate=${candidate.length}.`)
  }
  let maxAbsoluteError = 0
  let absoluteErrorSum = 0
  let squaredErrorSum = 0
  let dot = 0
  let referenceSquared = 0
  let candidateSquared = 0
  let finite = true
  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index]
    const actual = candidate[index]
    if (!Number.isFinite(expected) || !Number.isFinite(actual)) finite = false
    const delta = actual - expected
    const absolute = Math.abs(delta)
    maxAbsoluteError = Math.max(maxAbsoluteError, absolute)
    absoluteErrorSum += absolute
    squaredErrorSum += delta * delta
    dot += expected * actual
    referenceSquared += expected * expected
    candidateSquared += actual * actual
  }
  const denominator = Math.sqrt(referenceSquared * candidateSquared)
  return {
    count: reference.length,
    maxAbsoluteError,
    meanAbsoluteError: reference.length > 0 ? absoluteErrorSum / reference.length : 0,
    rmse: reference.length > 0 ? Math.sqrt(squaredErrorSum / reference.length) : 0,
    cosineSimilarity: denominator > 0 ? dot / denominator : 1,
    finite,
  }
}
