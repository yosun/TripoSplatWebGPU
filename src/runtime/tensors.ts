export const TRANSFERABLE_TENSOR_TYPES = ['float32', 'float16', 'int32', 'int64'] as const

export type TransferableTensorType = (typeof TRANSFERABLE_TENSOR_TYPES)[number]

export interface TensorDataTypeMap {
  float32: Float32Array
  /** Raw IEEE-754 binary16 bit patterns, as expected by ONNX Runtime Web. */
  float16: Uint16Array
  int32: Int32Array
  int64: BigInt64Array
}

export type TransferableTensorData = TensorDataTypeMap[TransferableTensorType]

export type TensorPayloadOfType<T extends TransferableTensorType> = {
  type: T
  dims: number[]
  data: TensorDataTypeMap[T]
}

export type TensorPayload = {
  [T in TransferableTensorType]: TensorPayloadOfType<T>
}[TransferableTensorType]

export type TensorPayloadMap = Record<string, TensorPayload>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function dataMatchesType(type: TransferableTensorType, data: unknown): data is TransferableTensorData {
  switch (type) {
    case 'float32':
      return data instanceof Float32Array
    case 'float16':
      return data instanceof Uint16Array
    case 'int32':
      return data instanceof Int32Array
    case 'int64':
      return data instanceof BigInt64Array
  }
}

export function isTransferableTensorType(value: unknown): value is TransferableTensorType {
  return typeof value === 'string' && TRANSFERABLE_TENSOR_TYPES.some((type) => type === value)
}

/** Calculate a shape's element count while rejecting malformed or overflowing dimensions. */
export function tensorElementCount(dims: readonly number[], label = 'tensor.dims'): number {
  if (!Array.isArray(dims)) {
    throw new TypeError(`${label} must be an array.`)
  }

  for (let index = 0; index < dims.length; index += 1) {
    const dimension = dims[index]
    if (!Number.isSafeInteger(dimension) || dimension < 0) {
      throw new RangeError(`${label}[${index}] must be a non-negative safe integer.`)
    }
  }

  if (dims.some((dimension) => dimension === 0)) {
    return 0
  }

  let count = 1
  for (const dimension of dims) {
    count *= dimension
    if (!Number.isSafeInteger(count)) {
      throw new RangeError(`${label} has an element count larger than Number.MAX_SAFE_INTEGER.`)
    }
  }
  return count
}

/**
 * Validate dtype, shape, backing buffer, and byte count.
 *
 * SharedArrayBuffer-backed views are rejected because they cannot be included
 * in a `postMessage` transfer list. Use `cloneTensorPayload` to make an owned,
 * exactly-sized ArrayBuffer copy.
 */
function assertTensorPayloadContents(
  value: unknown,
  label: string,
  requireTransferableBuffer: boolean,
): asserts value is TensorPayload {
  if (!isRecord(value)) {
    throw new TypeError(`${label} must be an object.`)
  }
  if (!isTransferableTensorType(value.type)) {
    throw new TypeError(
      `${label}.type must be one of ${TRANSFERABLE_TENSOR_TYPES.map((type) => `'${type}'`).join(', ')}.`,
    )
  }
  if (!Array.isArray(value.dims)) {
    throw new TypeError(`${label}.dims must be an array.`)
  }
  if (!dataMatchesType(value.type, value.data)) {
    throw new TypeError(`${label}.data is not the typed array required by dtype '${value.type}'.`)
  }
  if (requireTransferableBuffer && !(value.data.buffer instanceof ArrayBuffer)) {
    throw new TypeError(`${label}.data must be backed by a transferable ArrayBuffer.`)
  }

  const expectedLength = tensorElementCount(value.dims, `${label}.dims`)
  if (value.data.length !== expectedLength) {
    throw new RangeError(
      `${label}.data contains ${value.data.length} elements; shape [${value.dims.join(', ')}] requires ${expectedLength}.`,
    )
  }
}

export function assertTensorPayload(value: unknown, label = 'tensor'): asserts value is TensorPayload {
  assertTensorPayloadContents(value, label, true)
}

export function createTensorPayload<T extends TransferableTensorType>(
  type: T,
  data: TensorDataTypeMap[T],
  dims: readonly number[],
): TensorPayloadOfType<T> {
  const payload = { type, data, dims: Array.from(dims) } as TensorPayloadOfType<T>
  assertTensorPayload(payload)
  return payload
}

function copyTensorData(payload: TensorPayload): TransferableTensorData {
  switch (payload.type) {
    case 'float32': {
      const copy = new Float32Array(payload.data.length)
      copy.set(payload.data)
      return copy
    }
    case 'float16': {
      const copy = new Uint16Array(payload.data.length)
      copy.set(payload.data)
      return copy
    }
    case 'int32': {
      const copy = new Int32Array(payload.data.length)
      copy.set(payload.data)
      return copy
    }
    case 'int64': {
      const copy = new BigInt64Array(payload.data.length)
      copy.set(payload.data)
      return copy
    }
  }
}

/** Make a transferable, exactly-sized copy (also useful for ORT/WASM-backed views). */
export function cloneTensorPayload(payload: TensorPayload): TensorPayload {
  // Unlike the transport validator, copying intentionally accepts a
  // SharedArrayBuffer-backed view and converts it to an owned ArrayBuffer.
  assertTensorPayloadContents(payload, 'tensor', false)
  const data = copyTensorData(payload)
  switch (payload.type) {
    case 'float32':
      return createTensorPayload('float32', data as Float32Array, payload.dims)
    case 'float16':
      return createTensorPayload('float16', data as Uint16Array, payload.dims)
    case 'int32':
      return createTensorPayload('int32', data as Int32Array, payload.dims)
    case 'int64':
      return createTensorPayload('int64', data as BigInt64Array, payload.dims)
  }
}

export function assertTensorPayloadMap(value: unknown, label = 'tensors'): asserts value is TensorPayloadMap {
  if (!isRecord(value)) {
    throw new TypeError(`${label} must be an object keyed by tensor name.`)
  }

  for (const [name, payload] of Object.entries(value)) {
    if (name.trim().length === 0 || name.includes('\0')) {
      throw new TypeError(`${label} contains an invalid empty tensor name.`)
    }
    assertTensorPayload(payload, `${label}.${name}`)
  }
}

/** Validate a tensor map and return each distinct backing ArrayBuffer once. */
export function tensorPayloadTransferables(tensors: Readonly<TensorPayloadMap>): Transferable[] {
  assertTensorPayloadMap(tensors)
  const buffers = new Set<ArrayBuffer>()
  for (const tensor of Object.values(tensors)) {
    const buffer = tensor.data.buffer
    if (!(buffer instanceof ArrayBuffer)) {
      // assertTensorPayload already checks this; retain the guard for type narrowing.
      throw new TypeError('Tensor data is not backed by a transferable ArrayBuffer.')
    }
    buffers.add(buffer)
  }
  return [...buffers]
}
