const scalarFloat = new Float32Array(1)
const scalarBits = new Uint32Array(scalarFloat.buffer)

/** Convert one JavaScript number to an IEEE-754 binary16 bit pattern. */
export function numberToFloat16Bits(value: number): number {
  scalarFloat[0] = value
  const bits = scalarBits[0]
  const sign = (bits >>> 16) & 0x8000
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15
  let mantissa = bits & 0x7fffff

  if (exponent <= 0) {
    if (exponent < -10) return sign
    mantissa = (mantissa | 0x800000) >>> (1 - exponent)
    // Round to nearest, ties to even.
    const rounded = (mantissa + 0xfff + ((mantissa >>> 13) & 1)) >>> 13
    return sign | rounded
  }
  if (exponent >= 31) {
    if (((bits >>> 23) & 0xff) === 0xff && mantissa !== 0) {
      return sign | 0x7c00 | Math.max(1, mantissa >>> 13)
    }
    return sign | 0x7c00
  }

  mantissa += 0xfff + ((mantissa >>> 13) & 1)
  if ((mantissa & 0x800000) !== 0) {
    mantissa = 0
    exponent += 1
    if (exponent >= 31) return sign | 0x7c00
  }
  return sign | (exponent << 10) | (mantissa >>> 13)
}

export function float32ToFloat16(source: Float32Array): Uint16Array {
  const result = new Uint16Array(source.length)
  for (let index = 0; index < source.length; index += 1) {
    result[index] = numberToFloat16Bits(source[index])
  }
  return result
}

export function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1
  const exponent = (bits >>> 10) & 0x1f
  const mantissa = bits & 0x3ff
  if (exponent === 0) {
    return mantissa === 0 ? sign * 0 : sign * 2 ** -14 * (mantissa / 1024)
  }
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024)
}

export function float16ToFloat32(source: Uint16Array): Float32Array {
  const result = new Float32Array(source.length)
  for (let index = 0; index < source.length; index += 1) {
    result[index] = float16BitsToNumber(source[index])
  }
  return result
}
