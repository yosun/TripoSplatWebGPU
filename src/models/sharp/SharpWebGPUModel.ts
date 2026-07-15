import { convertFocalMmToPx } from '../../lib/focal'
import { imageBitmapToSharpTensor } from '../../lib/image'
import {
  DEFAULT_FOCAL_MM,
  DEFAULT_MAX_GAUSSIANS,
  DEFAULT_OPACITY_THRESHOLD,
  DEFAULT_WEB_MODEL_URL,
  SHARP_INTERNAL_RESOLUTION,
} from '../../lib/sharpConstants'
import { SharpWorkerClient } from '../../lib/sharpWorkerClient'
import type { WorkerStatusMessage } from '../../workers/messages'
import type {
  GaussianScene,
  GenerationOptions,
  GenerationProgress,
  ImageToGaussianModel,
} from '../ImageToGaussianModel'
import { throwIfAborted } from '../ImageToGaussianModel'

export interface SharpWebGPUModelOptions {
  modelUrl?: string
  onStatus?: (status: WorkerStatusMessage) => void
}

/**
 * The known-good ml-sharp-web pipeline behind the model-independent lifecycle.
 *
 * SHARP's graph, NDC unprojection, opacity pruning and camera-rich PLY writer
 * intentionally remain in its existing worker. TripoSplat does not share those
 * assumptions; only the public model contract is shared.
 */
export class SharpWebGPUModel implements ImageToGaussianModel {
  readonly modelUrl: string

  private readonly worker: SharpWorkerClient
  private loaded = false
  private disposed = false

  constructor(options: SharpWebGPUModelOptions = {}) {
    this.modelUrl = options.modelUrl ?? DEFAULT_WEB_MODEL_URL
    this.worker = new SharpWorkerClient(options.onStatus)
  }

  async load(): Promise<void> {
    this.assertUsable()
    if (this.loaded) return
    await this.worker.loadModel({ modelUrl: this.modelUrl })
    this.loaded = true
  }

  async generate(image: ImageBitmap, options: GenerationOptions = {}): Promise<GaussianScene> {
    this.assertUsable()
    throwIfAborted(options.signal)
    await this.load()

    const progress = this.createProgressReporter(options)
    progress({ stage: 'preprocessing', message: 'Preparing SHARP image tensor…' })
    const imageTensor = imageBitmapToSharpTensor(image, SHARP_INTERNAL_RESOLUTION)
    throwIfAborted(options.signal)

    const focalPx = options.focalPx ?? convertFocalMmToPx(image.width, image.height, DEFAULT_FOCAL_MM)
    if (!Number.isFinite(focalPx) || focalPx <= 0) {
      throw new Error('SHARP focalPx must be a positive finite number.')
    }

    progress({ stage: 'inference', message: 'Running SHARP with ONNX Runtime WebGPU…' })
    const result = await this.worker.runInference({
      modelUrl: this.modelUrl,
      imageTensor: imageTensor.buffer,
      imageWidth: image.width,
      imageHeight: image.height,
      focalPx,
      disparityFactor: focalPx / image.width,
      opacityThreshold: options.opacityThreshold ?? DEFAULT_OPACITY_THRESHOLD,
      maxGaussians: options.maxGaussians ?? DEFAULT_MAX_GAUSSIANS,
    })
    throwIfAborted(options.signal)

    return {
      model: 'sharp',
      count: result.selectedGaussians,
      totalCount: result.totalGaussians,
      ply: new Uint8Array(result.plyBuffer as ArrayBuffer),
      coordinateSystem: 'opencv-camera',
      colorSpace: 'linear-rgb',
      metadata: {
        focalPx,
        imageWidth: image.width,
        imageHeight: image.height,
      },
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return
    this.disposed = true
    this.loaded = false
    this.worker.dispose()
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error('SharpWebGPUModel has been disposed.')
    }
  }

  private createProgressReporter(options: GenerationOptions): (progress: GenerationProgress) => void {
    return (progress) => options.onProgress?.(progress)
  }
}
