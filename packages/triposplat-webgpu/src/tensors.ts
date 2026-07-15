export type TensorType = 'float32' | 'float16' | 'int32' | 'int64'

export interface TensorDataMap {
  float32: Float32Array
  /** Raw IEEE-754 binary16 bit patterns. */
  float16: Uint16Array
  int32: Int32Array
  int64: BigInt64Array
}

export type TensorPayloadOfType<Type extends TensorType> = {
  type: Type
  dims: number[]
  data: TensorDataMap[Type]
}

export type TensorPayload = {
  [Type in TensorType]: TensorPayloadOfType<Type>
}[TensorType]

export type TensorMap = Record<string, TensorPayload>

export function tensorElementCount(dims: readonly number[]): number {
  let count = 1
  for (const [index, dimension] of dims.entries()) {
    if (!Number.isSafeInteger(dimension) || dimension < 0) {
      throw new RangeError(`dims[${index}] must be a non-negative safe integer.`)
    }
    count *= dimension
    if (!Number.isSafeInteger(count)) throw new RangeError('Tensor element count exceeds Number.MAX_SAFE_INTEGER.')
  }
  return count
}

function storageMatches(type: TensorType, data: unknown): boolean {
  switch (type) {
    case 'float32': return data instanceof Float32Array
    case 'float16': return data instanceof Uint16Array
    case 'int32': return data instanceof Int32Array
    case 'int64': return data instanceof BigInt64Array
  }
}

export function assertTensorPayload(value: unknown, label = 'tensor'): asserts value is TensorPayload {
  if (typeof value !== 'object' || value === null) throw new TypeError(`${label} must be an object.`)
  const candidate = value as Partial<TensorPayload>
  if (!candidate.type || !['float32', 'float16', 'int32', 'int64'].includes(candidate.type)) {
    throw new TypeError(`${label}.type is not a supported transferable tensor type.`)
  }
  if (!Array.isArray(candidate.dims)) throw new TypeError(`${label}.dims must be an array.`)
  if (!storageMatches(candidate.type, candidate.data)) {
    throw new TypeError(`${label}.data does not match dtype '${candidate.type}'.`)
  }
  if (!(candidate.data?.buffer instanceof ArrayBuffer)) {
    throw new TypeError(`${label}.data must be backed by an ArrayBuffer.`)
  }
  const expected = tensorElementCount(candidate.dims)
  if (candidate.data.length !== expected) {
    throw new RangeError(`${label} has ${candidate.data.length} values; shape requires ${expected}.`)
  }
}

export function createTensor<Type extends TensorType>(
  type: Type,
  data: TensorDataMap[Type],
  dims: readonly number[],
): TensorPayloadOfType<Type> {
  const value = { type, data, dims: Array.from(dims) } as TensorPayloadOfType<Type>
  assertTensorPayload(value)
  return value
}

export function assertTensorMap(value: unknown, label = 'tensors'): asserts value is TensorMap {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`)
  }
  for (const [name, tensor] of Object.entries(value)) {
    if (name.trim().length === 0 || name.includes('\0')) throw new TypeError(`${label} has an invalid tensor name.`)
    assertTensorPayload(tensor, `${label}.${name}`)
  }
}

export function tensorTransferables(tensors: Readonly<TensorMap>): Transferable[] {
  assertTensorMap(tensors)
  const buffers = new Set<ArrayBuffer>()
  for (const tensor of Object.values(tensors)) buffers.add(tensor.data.buffer as ArrayBuffer)
  return [...buffers]
}

const scalarFloat = new Float32Array(1)
const scalarBits = new Uint32Array(scalarFloat.buffer)

export function numberToFloat16Bits(value: number): number {
  scalarFloat[0] = value
  const bits = scalarBits[0]
  const sign = (bits >>> 16) & 0x8000
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15
  let mantissa = bits & 0x7fffff
  if (exponent <= 0) {
    if (exponent < -10) return sign
    mantissa = (mantissa | 0x800000) >>> (1 - exponent)
    return sign | ((mantissa + 0xfff + ((mantissa >>> 13) & 1)) >>> 13)
  }
  if (exponent >= 31) {
    return ((bits >>> 23) & 0xff) === 0xff && mantissa !== 0
      ? sign | 0x7c00 | Math.max(1, mantissa >>> 13)
      : sign | 0x7c00
  }
  mantissa += 0xfff + ((mantissa >>> 13) & 1)
  if ((mantissa & 0x800000) !== 0) {
    mantissa = 0
    exponent += 1
    if (exponent >= 31) return sign | 0x7c00
  }
  return sign | (exponent << 10) | (mantissa >>> 13)
}

export function float32ToFloat16(values: Float32Array): Uint16Array {
  const result = new Uint16Array(values.length)
  for (let index = 0; index < values.length; index += 1) result[index] = numberToFloat16Bits(values[index])
  return result
}

export function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1
  const exponent = (bits >>> 10) & 0x1f
  const mantissa = bits & 0x3ff
  if (exponent === 0) return mantissa === 0 ? sign * 0 : sign * 2 ** -14 * (mantissa / 1024)
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024)
}

export function float16ToFloat32(values: Uint16Array): Float32Array {
  const result = new Float32Array(values.length)
  for (let index = 0; index < values.length; index += 1) result[index] = float16BitsToNumber(values[index])
  return result
}
