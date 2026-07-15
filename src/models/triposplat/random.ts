export interface RandomSource {
  next(): number
}

/** Small deterministic PRNG for repeatable browser runs. It is not PyTorch's RNG. */
export class Mulberry32 implements RandomSource {
  private state: number

  constructor(seed: number) {
    this.state = seed >>> 0
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0
    let value = this.state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000
  }
}

export function fillUniform(target: Float32Array, random: RandomSource): Float32Array {
  for (let i = 0; i < target.length; i += 1) target[i] = random.next()
  return target
}

export function fillNormal(target: Float32Array, random: RandomSource): Float32Array {
  let index = 0
  while (index < target.length) {
    const u1 = Math.max(random.next(), Number.EPSILON)
    const u2 = random.next()
    const radius = Math.sqrt(-2 * Math.log(u1))
    const angle = 2 * Math.PI * u2
    target[index] = radius * Math.cos(angle)
    index += 1
    if (index < target.length) {
      target[index] = radius * Math.sin(angle)
      index += 1
    }
  }
  return target
}
