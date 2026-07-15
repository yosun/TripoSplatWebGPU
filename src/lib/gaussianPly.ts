import type { GaussianAttributes } from '../models/ImageToGaussianModel'

const TRIPOSPLAT_EXPORT_TRANSFORM = new Float32Array([
  1, 0, 0,
  0, 0, -1,
  0, 1, 0,
])

const PROPERTY_NAMES = [
  'x', 'y', 'z',
  'nx', 'ny', 'nz',
  'f_dc_0', 'f_dc_1', 'f_dc_2',
  'opacity',
  'scale_0', 'scale_1', 'scale_2',
  'rot_0', 'rot_1', 'rot_2', 'rot_3',
] as const

function clampProbability(value: number): number {
  return Math.min(1 - 1e-6, Math.max(1e-6, value))
}

function validate(attributes: GaussianAttributes): number {
  const count = attributes.opacities.length
  if (attributes.positions.length !== count * 3) throw new Error('positions length must equal count * 3.')
  if (attributes.scales.length !== count * 3) throw new Error('scales length must equal count * 3.')
  if (attributes.rotations.length !== count * 4) throw new Error('rotations length must equal count * 4.')
  if (attributes.sh0.length !== count * 3) throw new Error('sh0 length must equal count * 3.')
  return count
}

function quaternionToMatrix(w: number, x: number, y: number, z: number): Float64Array {
  const inverseNorm = 1 / (Math.hypot(w, x, y, z) || 1)
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

function leftMultiply3x3(left: Float32Array, right: Float64Array): Float64Array {
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
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2
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
  const inverseNorm = 1 / (Math.hypot(w, x, y, z) || 1)
  return [w * inverseNorm, x * inverseNorm, y * inverseNorm, z * inverseNorm]
}

/** Standard binary 3DGS PLY matching TripoSplat's official field semantics. */
export function buildGaussianPly(attributes: GaussianAttributes): Uint8Array {
  const count = validate(attributes)
  const header = [
    'ply',
    'format binary_little_endian 1.0',
    `element vertex ${count}`,
    ...PROPERTY_NAMES.map((name) => `property float ${name}`),
    'end_header',
    '',
  ].join('\n')
  const headerBytes = new TextEncoder().encode(header)
  const stride = PROPERTY_NAMES.length * 4
  const bytes = new Uint8Array(headerBytes.length + count * stride)
  bytes.set(headerBytes)
  const view = new DataView(bytes.buffer)
  let offset = headerBytes.length

  for (let index = 0; index < count; index += 1) {
    const index3 = index * 3
    const index4 = index * 4
    const px = attributes.positions[index3]
    const py = attributes.positions[index3 + 1]
    const pz = attributes.positions[index3 + 2]
    const rotation = matrixToQuaternion(
      leftMultiply3x3(
        TRIPOSPLAT_EXPORT_TRANSFORM,
        quaternionToMatrix(
          attributes.rotations[index4],
          attributes.rotations[index4 + 1],
          attributes.rotations[index4 + 2],
          attributes.rotations[index4 + 3],
        ),
      ),
    )
    const opacity = clampProbability(attributes.opacities[index])
    const values = [
      px, -pz, py,
      0, 0, 0,
      attributes.sh0[index3], attributes.sh0[index3 + 1], attributes.sh0[index3 + 2],
      Math.log(opacity / (1 - opacity)),
      Math.log(Math.max(attributes.scales[index3], 1e-12)),
      Math.log(Math.max(attributes.scales[index3 + 1], 1e-12)),
      Math.log(Math.max(attributes.scales[index3 + 2], 1e-12)),
      rotation[0], rotation[1], rotation[2], rotation[3],
    ]
    for (const value of values) {
      view.setFloat32(offset, value, true)
      offset += 4
    }
  }
  return bytes
}
