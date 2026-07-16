import {
  TripoSplatWebGPU,
  type CompatibilityOptions,
  type CompatibilityReport,
  type GenerateOptions,
  type GaussianSceneMetadata,
  type LoadOptions,
  type TripoSplatImageSource,
  type TripoSplatWebGPUOptions,
} from '@ai3d/triposplat-webgpu'

export interface ImageToSplatResult {
  count: number
  metadata: GaussianSceneMetadata
  ply: Blob
  splat: Blob
  elapsedMs: number
}

/**
 * Smallest reusable browser boundary: image-like input in, portable files out.
 * The GaussianScene is always released after both files have been encoded.
 */
export class ImageToSplatGenerator {
  readonly model: TripoSplatWebGPU

  constructor(options: TripoSplatWebGPUOptions) {
    this.model = new TripoSplatWebGPU(options)
  }

  static checkCompatibility(options?: CompatibilityOptions): Promise<CompatibilityReport> {
    return TripoSplatWebGPU.checkCompatibility(options)
  }

  initialize(options: LoadOptions = {}): Promise<void> {
    return this.model.load(options)
  }

  async generate(
    input: TripoSplatImageSource,
    options: GenerateOptions = {},
  ): Promise<ImageToSplatResult> {
    const startedAt = performance.now()
    const scene = await this.model.generate(input, options)
    try {
      // Encode sequentially to avoid holding both encoder workspaces at once.
      const ply = await scene.exportPLY()
      const splat = await scene.exportSplat()
      return {
        count: scene.count,
        metadata: scene.metadata,
        ply,
        splat,
        elapsedMs: performance.now() - startedAt,
      }
    } finally {
      scene.dispose()
    }
  }

  dispose(): Promise<void> {
    return this.model.dispose()
  }
}
