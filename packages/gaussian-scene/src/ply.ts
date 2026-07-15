import type {
  GaussianPlyExportTransform,
  GaussianRotationOrder,
  GaussianSceneMetadata,
} from './types.js'

const SH_C0 = 0.28209479177387814

const PROPERTY_NAMES = [
  'x', 'y', 'z',
  'nx', 'ny', 'nz',
  'f_dc_0', 'f_dc_1', 'f_dc_2',
  'opacity',
  'scale_0', 'scale_1', 'scale_2',
  'rot_0', 'rot_1', 'rot_2', 'rot_3',
] as const

export interface GaussianPlyData {
  count: number
  positions: Float32Array
  scales: Float32Array
  rotations: Float32Array
  opacities: Float32Array
  sphericalHarmonics: Float32Array | undefined
  colors: Float32Array | undefined
  metadata: GaussianSceneMetadata
  transform: GaussianPlyExportTransform
}

function linearToSrgb(value: number): number {
  return value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055
}

function rgbToSh0(value: number): number {
  return (value - 0.5) / SH_C0
}

function clampProbability(value: number): number {
  return Math.min(1 - 1e-6, Math.max(1e-6, value))
}

function readQuaternion(
  values: Float32Array,
  offset: number,
  order: GaussianRotationOrder,
): [number, number, number, number] {
  return order === 'wxyz'
    ? [values[offset], values[offset + 1], values[offset + 2], values[offset + 3]]
    : [values[offset + 3], values[offset], values[offset + 1], values[offset + 2]]
}

function quaternionToMatrix(w: number, x: number, y: number, z: number): Float64Array {
  const inverseNorm = 1 / Math.hypot(w, x, y, z)
  w *= inverseNorm
  x *= inverseNorm
  y *= inverseNorm
  z *= inverseNorm
  return new Float64Array([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ])
}

function leftMultiply3x3(
  left: GaussianPlyExportTransform,
  right: Float64Array,
): Float64Array {
  const result = new Float64Array(9)
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      result[row * 3 + column] =
        left[row * 3] * right[column] +
        left[row * 3 + 1] * right[3 + column] +
        left[row * 3 + 2] * right[6 + column]
    }
  }
  return result
}

function matrixToQuaternion(matrix: Float64Array): [number, number, number, number] {
  const m00 = matrix[0]
  const m11 = matrix[4]
  const m22 = matrix[8]
  const trace = m00 + m11 + m22
  let w: number
  let x: number
  let y: number
  let z: number
  // Match official TripoSplat `_matrix_to_quat`: use the trace formula for
  // every rotation except the exact trace=-1 (180 degree) case.
  const traceScale = Math.sqrt(Math.max(trace + 1, 0)) * 2
  if (traceScale !== 0) {
    const s = traceScale
    w = 0.25 * s
    x = (matrix[7] - matrix[5]) / s
    y = (matrix[2] - matrix[6]) / s
    z = (matrix[3] - matrix[1]) / s
  } else if (m00 >= m11 && m00 >= m22) {
    const s = Math.sqrt(Math.max(0, 1 + m00 - m11 - m22)) * 2 || 1
    w = (matrix[7] - matrix[5]) / s
    x = 0.25 * s
    y = (matrix[1] + matrix[3]) / s
    z = (matrix[2] + matrix[6]) / s
  } else if (m11 >= m22) {
    const s = Math.sqrt(Math.max(0, 1 + m11 - m00 - m22)) * 2 || 1
    w = (matrix[2] - matrix[6]) / s
    x = (matrix[1] + matrix[3]) / s
    y = 0.25 * s
    z = (matrix[5] + matrix[7]) / s
  } else {
    const s = Math.sqrt(Math.max(0, 1 + m22 - m00 - m11)) * 2 || 1
    w = (matrix[3] - matrix[1]) / s
    x = (matrix[2] + matrix[6]) / s
    y = (matrix[5] + matrix[7]) / s
    z = 0.25 * s
  }
  const inverseNorm = 1 / Math.hypot(w, x, y, z)
  return [w * inverseNorm, x * inverseNorm, y * inverseNorm, z * inverseNorm]
}

function transformPosition(
  transform: GaussianPlyExportTransform,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  return [
    transform[0] * x + transform[1] * y + transform[2] * z,
    transform[3] * x + transform[4] * y + transform[5] * z,
    transform[6] * x + transform[7] * y + transform[8] * z,
  ]
}

function appearance(data: GaussianPlyData, gaussian: number): [number, number, number] {
  const base = gaussian * 3
  if (data.sphericalHarmonics) {
    return [
      data.sphericalHarmonics[base],
      data.sphericalHarmonics[base + 1],
      data.sphericalHarmonics[base + 2],
    ]
  }
  if (!data.colors) return [0, 0, 0]
  const convert = data.metadata.colorSemantics === 'linear-rgb'
    ? (value: number) => rgbToSh0(linearToSrgb(value))
    : rgbToSh0
  return [convert(data.colors[base]), convert(data.colors[base + 1]), convert(data.colors[base + 2])]
}

/** Encode the standard 17-float binary little-endian 3DGS PLY vertex layout. */
export function encodeGaussianPly(data: GaussianPlyData): Uint8Array {
  const header = [
    'ply',
    'format binary_little_endian 1.0',
    `element vertex ${data.count}`,
    ...PROPERTY_NAMES.map((name) => `property float ${name}`),
    'end_header',
    '',
  ].join('\n')
  const headerBytes = new TextEncoder().encode(header)
  const stride = PROPERTY_NAMES.length * 4
  const bytes = new Uint8Array(headerBytes.length + data.count * stride)
  bytes.set(headerBytes)
  const view = new DataView(bytes.buffer)
  let offset = headerBytes.length

  for (let index = 0; index < data.count; index += 1) {
    const index3 = index * 3
    const index4 = index * 4
    const position = transformPosition(
      data.transform,
      data.positions[index3],
      data.positions[index3 + 1],
      data.positions[index3 + 2],
    )
    const sourceRotation = readQuaternion(data.rotations, index4, data.metadata.rotationOrder)
    const rotation = matrixToQuaternion(
      leftMultiply3x3(data.transform, quaternionToMatrix(...sourceRotation)),
    )
    const color = appearance(data, index)
    const opacity = data.metadata.opacityEncoding === 'linear'
      ? Math.log(clampProbability(data.opacities[index]) / (1 - clampProbability(data.opacities[index])))
      : data.opacities[index]
    const scales = data.metadata.scaleEncoding === 'linear'
      ? [
          Math.log(data.scales[index3]),
          Math.log(data.scales[index3 + 1]),
          Math.log(data.scales[index3 + 2]),
        ]
      : [data.scales[index3], data.scales[index3 + 1], data.scales[index3 + 2]]
    const values = [
      ...position,
      0, 0, 0,
      ...color,
      opacity,
      ...scales,
      ...rotation,
    ]
    for (const value of values) {
      view.setFloat32(offset, value, true)
      offset += 4
    }
  }
  return bytes
}
