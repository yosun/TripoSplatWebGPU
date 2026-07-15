export const TRIPOSPLAT_IMAGE_SHAPE = [1, 3, 1024, 1024] as const
export const TRIPOSPLAT_VAE_NOISE_SHAPE = [1, 32, 128, 128] as const
export const TRIPOSPLAT_FEATURE1_SHAPE = [1, 4101, 1280] as const
export const TRIPOSPLAT_FEATURE2_SHAPE = [1, 4101, 128] as const
export const TRIPOSPLAT_LATENT_SHAPE = [1, 8192, 16] as const
export const TRIPOSPLAT_CAMERA_SHAPE = [1, 1, 5] as const
export const TRIPOSPLAT_MAX_DECODER_POINTS = 8192
export const TRIPOSPLAT_MAX_GAUSSIANS = 262_144

export function elementCount(shape: readonly number[]): number {
  return shape.reduce((product, dimension) => product * dimension, 1)
}

export function assertLength(label: string, value: ArrayLike<unknown>, expected: number): void {
  if (value.length !== expected) {
    throw new RangeError(`${label} contains ${value.length} values; expected ${expected}.`)
  }
}
