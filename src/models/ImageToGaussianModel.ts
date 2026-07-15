export type ModelKind = 'sharp' | 'triposplat'

export type GenerationStage =
  | 'loading-model'
  | 'preprocessing'
  | 'inference'
  | 'encoding-dinov3'
  | 'encoding-vae'
  | 'sampling'
  | 'decoding-octree'
  | 'decoding-gaussians'
  | 'filtering'
  | 'building-ply'

export interface GenerationProgress {
  stage: GenerationStage
  message: string
  /** Normalized progress for the current stage, when it can be measured. */
  progress?: number
  /** One-based iterative step, used by TripoSplat's flow sampler. */
  step?: number
  totalSteps?: number
}

export interface GenerationOptions {
  signal?: AbortSignal
  onProgress?: (progress: GenerationProgress) => void

  /** SHARP camera input. The UI supplies EXIF-derived pixels when available. */
  focalPx?: number
  /** SHARP output filtering. */
  opacityThreshold?: number
  maxGaussians?: number

  /** TripoSplat flow-matching controls. */
  steps?: 4 | 20 | number
  seed?: number
  guidanceScale?: number
  shift?: number
  numGaussians?: number
  erodeRadius?: number
  /** Skip BiRefNet/crop only for an official 1024x1024 RGB-on-black prepared image. */
  inputIsPrepared?: boolean
  /** Optional recorded tensors for cross-runtime parity; otherwise browser noise is seeded. */
  vaeNoise?: Float32Array
  latentNoise?: Float32Array
  cameraNoise?: Float32Array
}

export interface GaussianAttributes {
  /** Object-space center, xyz-interleaved. */
  positions: Float32Array
  /** Positive linear scale, xyz-interleaved. */
  scales: Float32Array
  /** Unit or pre-normalized quaternion in wxyz order. */
  rotations: Float32Array
  /** Degree-zero spherical-harmonic coefficients, rgb-interleaved. */
  sh0: Float32Array
  /** Activated opacity in [0, 1]. */
  opacities: Float32Array
}

export interface GaussianScene {
  model: ModelKind
  count: number
  totalCount: number
  /** Canonical 3DGS binary PLY, ready for the existing viewers/export path. */
  ply: Uint8Array
  gaussians?: GaussianAttributes
  coordinateSystem: 'opencv-camera' | 'triposplat-object'
  colorSpace: 'linear-rgb' | 'sh0'
  metadata?: Readonly<Record<string, string | number | boolean>>
}

export interface ImageToGaussianModel {
  load(): Promise<void>
  generate(image: ImageBitmap, options?: GenerationOptions): Promise<GaussianScene>
  dispose(): Promise<void>
}

export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new DOMException('Operation aborted', 'AbortError')
  }
}
