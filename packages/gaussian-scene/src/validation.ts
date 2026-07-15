import type {
  CreateGaussianSceneInput,
  GaussianPlyExportTransform,
  GaussianSceneMetadata,
  JsonValue,
} from './types.js'

const IDENTITY_TRANSFORM: GaussianPlyExportTransform = [
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
]

function assertNonEmptyString(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
}

function cloneJson(value: unknown, path: string): JsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError(`${path} must contain only finite numbers.`)
    return value
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map((item, index) => cloneJson(item, `${path}[${index}]`)))
  }
  if (typeof value === 'object') {
    const prototype = Object.getPrototypeOf(value)
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(`${path} must contain only JSON-compatible objects.`)
    }
    const result: Record<string, JsonValue> = {}
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      result[key] = cloneJson(item, `${path}.${key}`)
    }
    return Object.freeze(result)
  }
  throw new TypeError(`${path} must contain only JSON-compatible values.`)
}

function validateTransform(value: readonly number[]): GaussianPlyExportTransform {
  if (value.length !== 9 || value.some((component) => !Number.isFinite(component))) {
    throw new TypeError('metadata.plyExportTransform must contain nine finite numbers.')
  }
  const matrix = Array.from(value)
  const tolerance = 1e-5
  for (let row = 0; row < 3; row += 1) {
    for (let other = row; other < 3; other += 1) {
      let dot = 0
      for (let column = 0; column < 3; column += 1) {
        dot += matrix[row * 3 + column] * matrix[other * 3 + column]
      }
      const expected = row === other ? 1 : 0
      if (Math.abs(dot - expected) > tolerance) {
        throw new TypeError('metadata.plyExportTransform must be orthonormal.')
      }
    }
  }
  const determinant =
    matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
    matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
    matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
  if (Math.abs(determinant - 1) > tolerance) {
    throw new TypeError('metadata.plyExportTransform must preserve orientation.')
  }
  return Object.freeze(matrix) as unknown as GaussianPlyExportTransform
}

export function cloneAndValidateMetadata(metadata: GaussianSceneMetadata): GaussianSceneMetadata {
  if (typeof metadata !== 'object' || metadata === null) {
    throw new TypeError('metadata must be an object.')
  }
  assertNonEmptyString(metadata.coordinateSystem, 'metadata.coordinateSystem')
  assertNonEmptyString(metadata.units, 'metadata.units')
  assertNonEmptyString(metadata.modelRevision, 'metadata.modelRevision')
  assertNonEmptyString(metadata.runtimeVersion, 'metadata.runtimeVersion')
  if (metadata.rotationOrder !== 'wxyz' && metadata.rotationOrder !== 'xyzw') {
    throw new TypeError("metadata.rotationOrder must be 'wxyz' or 'xyzw'.")
  }
  if (metadata.scaleEncoding !== 'linear' && metadata.scaleEncoding !== 'log') {
    throw new TypeError("metadata.scaleEncoding must be 'linear' or 'log'.")
  }
  if (metadata.opacityEncoding !== 'linear' && metadata.opacityEncoding !== 'logit') {
    throw new TypeError("metadata.opacityEncoding must be 'linear' or 'logit'.")
  }
  if (
    metadata.colorSemantics !== null &&
    metadata.colorSemantics !== 'linear-rgb' &&
    metadata.colorSemantics !== 'srgb'
  ) {
    throw new TypeError("metadata.colorSemantics must be 'linear-rgb', 'srgb', or null.")
  }
  if (
    metadata.sphericalHarmonicsSemantics !== null &&
    metadata.sphericalHarmonicsSemantics !== 'degree-0-rgb'
  ) {
    throw new TypeError("metadata.sphericalHarmonicsSemantics must be 'degree-0-rgb' or null.")
  }
  if (!Number.isSafeInteger(metadata.seed)) {
    throw new TypeError('metadata.seed must be a safe integer.')
  }
  if (
    typeof metadata.generationSettings !== 'object' ||
    metadata.generationSettings === null ||
    Array.isArray(metadata.generationSettings)
  ) {
    throw new TypeError('metadata.generationSettings must be an object.')
  }
  const generationSettings = cloneJson(
    metadata.generationSettings,
    'metadata.generationSettings',
  ) as Readonly<Record<string, JsonValue>>
  const plyExportTransform = metadata.plyExportTransform === undefined
    ? undefined
    : validateTransform(metadata.plyExportTransform)

  return Object.freeze({
    coordinateSystem: metadata.coordinateSystem,
    units: metadata.units,
    rotationOrder: metadata.rotationOrder,
    scaleEncoding: metadata.scaleEncoding,
    opacityEncoding: metadata.opacityEncoding,
    colorSemantics: metadata.colorSemantics,
    sphericalHarmonicsSemantics: metadata.sphericalHarmonicsSemantics,
    modelRevision: metadata.modelRevision,
    generationSettings,
    seed: metadata.seed,
    runtimeVersion: metadata.runtimeVersion,
    ...(plyExportTransform === undefined ? {} : { plyExportTransform }),
  })
}

function cloneFloat32Array(value: unknown, expectedLength: number, label: string): Float32Array {
  if (!(value instanceof Float32Array)) throw new TypeError(`${label} must be a Float32Array.`)
  if (value.length !== expectedLength) {
    throw new RangeError(`${label} has ${value.length} values; expected ${expectedLength}.`)
  }
  const result = new Float32Array(value)
  for (let index = 0; index < result.length; index += 1) {
    if (!Number.isFinite(result[index])) {
      throw new RangeError(`${label}[${index}] must be finite.`)
    }
  }
  return result
}

export interface ValidatedGaussianSceneData {
  count: number
  positions: Float32Array
  scales: Float32Array
  rotations: Float32Array
  opacities: Float32Array
  sphericalHarmonics: Float32Array | undefined
  colors: Float32Array | undefined
  metadata: GaussianSceneMetadata
  plyExportTransform: GaussianPlyExportTransform
}

export function cloneAndValidateSceneInput(input: CreateGaussianSceneInput): ValidatedGaussianSceneData {
  if (typeof input !== 'object' || input === null) throw new TypeError('scene input must be an object.')
  if (!Number.isSafeInteger(input.count) || input.count < 0) {
    throw new RangeError('count must be a non-negative safe integer.')
  }
  const count = input.count
  const metadata = cloneAndValidateMetadata(input.metadata)
  const positions = cloneFloat32Array(input.positions, count * 3, 'positions')
  const scales = cloneFloat32Array(input.scales, count * 3, 'scales')
  const rotations = cloneFloat32Array(input.rotations, count * 4, 'rotations')
  const opacities = cloneFloat32Array(input.opacities, count, 'opacities')
  const sphericalHarmonics = input.sphericalHarmonics === undefined
    ? undefined
    : cloneFloat32Array(input.sphericalHarmonics, count * 3, 'sphericalHarmonics')
  const colors = input.colors === undefined
    ? undefined
    : cloneFloat32Array(input.colors, count * 3, 'colors')

  if ((colors === undefined) !== (metadata.colorSemantics === null)) {
    throw new TypeError('colors and metadata.colorSemantics must either both be present or both be absent.')
  }
  if ((sphericalHarmonics === undefined) !== (metadata.sphericalHarmonicsSemantics === null)) {
    throw new TypeError(
      'sphericalHarmonics and metadata.sphericalHarmonicsSemantics must either both be present or both be absent.',
    )
  }
  if (metadata.scaleEncoding === 'linear') {
    for (let index = 0; index < scales.length; index += 1) {
      if (scales[index] <= 0) throw new RangeError(`scales[${index}] must be greater than zero.`)
    }
  }
  if (metadata.opacityEncoding === 'linear') {
    for (let index = 0; index < opacities.length; index += 1) {
      if (opacities[index] < 0 || opacities[index] > 1) {
        throw new RangeError(`opacities[${index}] must be in [0, 1].`)
      }
    }
  }
  if (colors) {
    for (let index = 0; index < colors.length; index += 1) {
      if (colors[index] < 0 || colors[index] > 1) {
        throw new RangeError(`colors[${index}] must be in [0, 1].`)
      }
    }
  }
  for (let gaussian = 0; gaussian < count; gaussian += 1) {
    const base = gaussian * 4
    if (Math.hypot(
      rotations[base],
      rotations[base + 1],
      rotations[base + 2],
      rotations[base + 3],
    ) === 0) {
      throw new RangeError(`rotations for Gaussian ${gaussian} must have non-zero norm.`)
    }
  }

  return {
    count,
    positions,
    scales,
    rotations,
    opacities,
    sphericalHarmonics,
    colors,
    metadata,
    plyExportTransform: metadata.plyExportTransform ?? IDENTITY_TRANSFORM,
  }
}
