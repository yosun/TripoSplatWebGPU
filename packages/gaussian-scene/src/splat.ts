import type {
  GaussianPlyExportTransform,
  GaussianRotationOrder,
  GaussianSceneMetadata,
} from './types.js'

const SH_C0 = 0.28209479177387814
const SPLAT_STRIDE = 32
const f32 = Math.fround

function add32(left: number, right: number): number {
  return f32(f32(left) + f32(right))
}

function subtract32(left: number, right: number): number {
  return f32(f32(left) - f32(right))
}

function multiply32(left: number, right: number): number {
  return f32(f32(left) * f32(right))
}

function divide32(left: number, right: number): number {
  return f32(f32(left) / f32(right))
}

function norm4(w: number, x: number, y: number, z: number): number {
  const sum = add32(
    add32(multiply32(w, w), multiply32(x, x)),
    add32(multiply32(y, y), multiply32(z, z)),
  )
  return f32(Math.sqrt(sum))
}

export interface GaussianSplatData {
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

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function linearToSrgb(value: number): number {
  const clamped = clamp01(value)
  return clamped <= 0.0031308
    ? 12.92 * clamped
    : 1.055 * clamped ** (1 / 2.4) - 0.055
}

function sigmoid(value: number): number {
  return f32(value >= 0
    ? 1 / (1 + Math.exp(-value))
    : Math.exp(value) / (1 + Math.exp(value)))
}

function byte(value: number): number {
  // Match NumPy's clip(...).astype(uint8) in the official TripoSplat exporter.
  return Math.trunc(Math.min(255, Math.max(0, value)))
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

function quaternionToMatrix(w: number, x: number, y: number, z: number): Float32Array {
  const norm = norm4(w, x, y, z)
  w = divide32(w, norm)
  x = divide32(x, norm)
  y = divide32(y, norm)
  z = divide32(z, norm)
  const twice = (value: number) => multiply32(2, value)
  return new Float32Array([
    subtract32(1, twice(add32(multiply32(y, y), multiply32(z, z)))),
    twice(subtract32(multiply32(x, y), multiply32(w, z))),
    twice(add32(multiply32(x, z), multiply32(w, y))),
    twice(add32(multiply32(x, y), multiply32(w, z))),
    subtract32(1, twice(add32(multiply32(x, x), multiply32(z, z)))),
    twice(subtract32(multiply32(y, z), multiply32(w, x))),
    twice(subtract32(multiply32(x, z), multiply32(w, y))),
    twice(add32(multiply32(y, z), multiply32(w, x))),
    subtract32(1, twice(add32(multiply32(x, x), multiply32(y, y)))),
  ])
}

function leftMultiply3x3(
  left: GaussianPlyExportTransform,
  right: Float32Array,
): Float32Array {
  const result = new Float32Array(9)
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      result[row * 3 + column] = add32(
        add32(
          multiply32(left[row * 3], right[column]),
          multiply32(left[row * 3 + 1], right[3 + column]),
        ),
        multiply32(left[row * 3 + 2], right[6 + column]),
      )
    }
  }
  return result
}

function matrixToQuaternion(matrix: Float32Array): [number, number, number, number] {
  const m00 = matrix[0]
  const m11 = matrix[4]
  const m22 = matrix[8]
  const trace = add32(add32(m00, m11), m22)
  let w: number
  let x: number
  let y: number
  let z: number
  // Match official TripoSplat `_matrix_to_quat`: use the trace formula for
  // every rotation except the exact trace=-1 (180 degree) case.
  const traceScale = multiply32(f32(Math.sqrt(Math.max(add32(trace, 1), 0))), 2)
  if (traceScale !== 0) {
    const s = traceScale
    w = multiply32(0.25, s)
    x = divide32(subtract32(matrix[7], matrix[5]), s)
    y = divide32(subtract32(matrix[2], matrix[6]), s)
    z = divide32(subtract32(matrix[3], matrix[1]), s)
  } else if (m00 >= m11 && m00 >= m22) {
    const s = multiply32(f32(Math.sqrt(Math.max(0, subtract32(subtract32(add32(1, m00), m11), m22)))), 2) || 1
    w = divide32(subtract32(matrix[7], matrix[5]), s)
    x = multiply32(0.25, s)
    y = divide32(add32(matrix[1], matrix[3]), s)
    z = divide32(add32(matrix[2], matrix[6]), s)
  } else if (m11 >= m22) {
    const s = multiply32(f32(Math.sqrt(Math.max(0, subtract32(subtract32(add32(1, m11), m00), m22)))), 2) || 1
    w = divide32(subtract32(matrix[2], matrix[6]), s)
    x = divide32(add32(matrix[1], matrix[3]), s)
    y = multiply32(0.25, s)
    z = divide32(add32(matrix[5], matrix[7]), s)
  } else {
    const s = multiply32(f32(Math.sqrt(Math.max(0, subtract32(subtract32(add32(1, m22), m00), m11)))), 2) || 1
    w = divide32(subtract32(matrix[3], matrix[1]), s)
    x = divide32(add32(matrix[2], matrix[6]), s)
    y = divide32(add32(matrix[5], matrix[7]), s)
    z = multiply32(0.25, s)
  }
  const norm = norm4(w, x, y, z)
  return [divide32(w, norm), divide32(x, norm), divide32(y, norm), divide32(z, norm)]
}

function transformedQuaternion(
  data: GaussianSplatData,
  offset: number,
): [number, number, number, number] {
  return matrixToQuaternion(leftMultiply3x3(
    data.transform,
    quaternionToMatrix(...readQuaternion(data.rotations, offset, data.metadata.rotationOrder)),
  ))
}

function color(data: GaussianSplatData, offset: number): [number, number, number] {
  if (data.sphericalHarmonics) {
    return [
      clamp01(add32(0.5, multiply32(f32(SH_C0), data.sphericalHarmonics[offset]))),
      clamp01(add32(0.5, multiply32(f32(SH_C0), data.sphericalHarmonics[offset + 1]))),
      clamp01(add32(0.5, multiply32(f32(SH_C0), data.sphericalHarmonics[offset + 2]))),
    ]
  }
  if (!data.colors) return [0.5, 0.5, 0.5]
  const convert = data.metadata.colorSemantics === 'linear-rgb' ? linearToSrgb : clamp01
  return [
    convert(data.colors[offset]),
    convert(data.colors[offset + 1]),
    convert(data.colors[offset + 2]),
  ]
}

/**
 * Encode the de-facto 32-byte browser `.splat` layout:
 * xyz/scale float32, RGBA uint8, then a quantized wxyz quaternion.
 */
export function encodeGaussianSplat(data: GaussianSplatData): Uint8Array {
  const bytes = new Uint8Array(data.count * SPLAT_STRIDE)
  const view = new DataView(bytes.buffer)
  // Official TripoSplat writes high-opacity, large-volume splats first.
  const order = Array.from({ length: data.count }, (_unused, index) => index)
  order.sort((left, right) => {
    const opacity = (index: number) => data.metadata.opacityEncoding === 'logit'
      ? sigmoid(data.opacities[index])
      : clamp01(data.opacities[index])
    const volume = (index: number) => {
      const offset = index * 3
      const scale = (axis: number) => data.metadata.scaleEncoding === 'log'
        ? Math.exp(data.scales[offset + axis])
        : data.scales[offset + axis]
      return multiply32(multiply32(scale(0), scale(1)), scale(2))
    }
    return multiply32(opacity(right), volume(right)) - multiply32(opacity(left), volume(left))
  })
  for (let outputIndex = 0; outputIndex < order.length; outputIndex += 1) {
    const index = order[outputIndex]
    const source3 = index * 3
    const source4 = index * 4
    const output = outputIndex * SPLAT_STRIDE
    const x = data.positions[source3]
    const y = data.positions[source3 + 1]
    const z = data.positions[source3 + 2]
    const transformed = (row: number) => add32(
      add32(multiply32(data.transform[row], x), multiply32(data.transform[row + 1], y)),
      multiply32(data.transform[row + 2], z),
    )
    view.setFloat32(output, transformed(0), true)
    view.setFloat32(output + 4, transformed(3), true)
    view.setFloat32(output + 8, transformed(6), true)
    for (let axis = 0; axis < 3; axis += 1) {
      const scale = data.scales[source3 + axis]
      view.setFloat32(output + 12 + axis * 4, data.metadata.scaleEncoding === 'log' ? f32(Math.exp(scale)) : scale, true)
    }
    const rgb = color(data, source3)
    bytes[output + 24] = byte(multiply32(rgb[0], 255))
    bytes[output + 25] = byte(multiply32(rgb[1], 255))
    bytes[output + 26] = byte(multiply32(rgb[2], 255))
    const opacity = data.metadata.opacityEncoding === 'logit'
      ? sigmoid(data.opacities[index])
      : clamp01(data.opacities[index])
    bytes[output + 27] = byte(multiply32(opacity, 255))
    const quaternion = transformedQuaternion(data, source4)
    for (let component = 0; component < 4; component += 1) {
      bytes[output + 28 + component] = byte(add32(multiply32(quaternion[component], 128), 128))
    }
  }
  return bytes
}
