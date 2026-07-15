import { encodeGaussianPly } from './ply.js'
import { encodeGaussianSplat } from './splat.js'
import type {
  CreateGaussianSceneInput,
  GaussianScene,
  GaussianSceneMetadata,
} from './types.js'
import { cloneAndValidateSceneInput } from './validation.js'

export class GaussianSceneDisposedError extends Error {
  readonly code = 'GAUSSIAN_SCENE_DISPOSED'

  constructor() {
    super('GaussianScene has been disposed.')
    this.name = 'GaussianSceneDisposedError'
  }
}

class OwnedGaussianScene implements GaussianScene {
  readonly metadata: GaussianSceneMetadata

  private countValue: number
  private positionsValue: Float32Array
  private scalesValue: Float32Array
  private rotationsValue: Float32Array
  private opacitiesValue: Float32Array
  private sphericalHarmonicsValue: Float32Array | undefined
  private colorsValue: Float32Array | undefined
  private readonly transform
  private disposedValue = false

  constructor(input: CreateGaussianSceneInput) {
    const validated = cloneAndValidateSceneInput(input)
    this.countValue = validated.count
    this.positionsValue = validated.positions
    this.scalesValue = validated.scales
    this.rotationsValue = validated.rotations
    this.opacitiesValue = validated.opacities
    this.sphericalHarmonicsValue = validated.sphericalHarmonics
    this.colorsValue = validated.colors
    this.metadata = validated.metadata
    this.transform = validated.plyExportTransform
  }

  get count(): number {
    return this.countValue
  }

  get positions(): Float32Array {
    return this.positionsValue
  }

  get scales(): Float32Array {
    return this.scalesValue
  }

  get rotations(): Float32Array {
    return this.rotationsValue
  }

  get opacities(): Float32Array {
    return this.opacitiesValue
  }

  get sphericalHarmonics(): Float32Array | undefined {
    return this.sphericalHarmonicsValue
  }

  get colors(): Float32Array | undefined {
    return this.colorsValue
  }

  get isDisposed(): boolean {
    return this.disposedValue
  }

  async exportPLY(): Promise<Blob> {
    this.assertUsable()
    const bytes = encodeGaussianPly({
      count: this.countValue,
      positions: this.positionsValue,
      scales: this.scalesValue,
      rotations: this.rotationsValue,
      opacities: this.opacitiesValue,
      sphericalHarmonics: this.sphericalHarmonicsValue,
      colors: this.colorsValue,
      metadata: this.metadata,
      transform: this.transform,
    })
    return new Blob([new Uint8Array(bytes)], { type: 'application/octet-stream' })
  }

  async exportSplat(): Promise<Blob> {
    this.assertUsable()
    const bytes = encodeGaussianSplat({
      count: this.countValue,
      positions: this.positionsValue,
      scales: this.scalesValue,
      rotations: this.rotationsValue,
      opacities: this.opacitiesValue,
      sphericalHarmonics: this.sphericalHarmonicsValue,
      colors: this.colorsValue,
      metadata: this.metadata,
      transform: this.transform,
    })
    return new Blob([new Uint8Array(bytes)], { type: 'application/octet-stream' })
  }

  dispose(): void {
    if (this.disposedValue) return
    this.disposedValue = true
    this.countValue = 0
    this.positionsValue = new Float32Array(0)
    this.scalesValue = new Float32Array(0)
    this.rotationsValue = new Float32Array(0)
    this.opacitiesValue = new Float32Array(0)
    this.sphericalHarmonicsValue = undefined
    this.colorsValue = undefined
  }

  private assertUsable(): void {
    if (this.disposedValue) throw new GaussianSceneDisposedError()
  }
}

/** Validate and defensively copy canonical Gaussian data into an owned scene. */
export function createGaussianScene(input: CreateGaussianSceneInput): GaussianScene {
  return new OwnedGaussianScene(input)
}
