export const TRIPOSPLAT_IMAGE_SHAPE = [1, 3, 1024, 1024] as const
export const TRIPOSPLAT_VAE_NOISE_SHAPE = [1, 32, 128, 128] as const
export const TRIPOSPLAT_FEATURE1_SHAPE = [1, 4101, 1280] as const
export const TRIPOSPLAT_FEATURE2_SHAPE = [1, 4101, 128] as const
export const TRIPOSPLAT_LATENT_SHAPE = [1, 8192, 16] as const
export const TRIPOSPLAT_CAMERA_SHAPE = [1, 1, 5] as const
export const TRIPOSPLAT_OCTREE_QUERY_SHAPE = [1, 8192, 3] as const
export const TRIPOSPLAT_OCTREE_LOGITS_SHAPE = [1, 8192, 8] as const
export const TRIPOSPLAT_MAX_DECODER_POINTS = 8192
export const TRIPOSPLAT_MIN_GAUSSIANS = 32768
export const TRIPOSPLAT_MAX_GAUSSIANS = 262144

export function elementCount(shape: readonly number[]): number {
  return shape.reduce((product, dimension) => product * dimension, 1)
}

export function assertShape(
  label: string,
  actual: readonly number[],
  expected: readonly number[],
): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw new Error(`${label} shape [${actual.join(',')}] does not match [${expected.join(',')}].`)
  }
}
