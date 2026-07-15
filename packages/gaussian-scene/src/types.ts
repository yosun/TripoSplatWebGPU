export type JsonPrimitive = string | number | boolean | null
export type JsonValue = JsonPrimitive | readonly JsonValue[] | Readonly<{ [key: string]: JsonValue }>

export type GaussianRotationOrder = 'wxyz' | 'xyzw'
export type GaussianScaleEncoding = 'linear' | 'log'
export type GaussianOpacityEncoding = 'linear' | 'logit'
export type GaussianColorSemantics = 'linear-rgb' | 'srgb' | null
export type GaussianSphericalHarmonicsSemantics = 'degree-0-rgb' | null

/** Row-major proper 3D rotation applied when exporting PLY coordinates. */
export type GaussianPlyExportTransform = readonly [
  number, number, number,
  number, number, number,
  number, number, number,
]

/** Explicit interpretation and provenance for a canonical Gaussian scene. */
export interface GaussianSceneMetadata {
  coordinateSystem: string
  units: string
  rotationOrder: GaussianRotationOrder
  scaleEncoding: GaussianScaleEncoding
  opacityEncoding: GaussianOpacityEncoding
  /** `null` records that no direct-color array is present. */
  colorSemantics: GaussianColorSemantics
  /** `null` records that no spherical-harmonic array is present. */
  sphericalHarmonicsSemantics: GaussianSphericalHarmonicsSemantics
  modelRevision: string
  generationSettings: Readonly<Record<string, JsonValue>>
  seed: number
  runtimeVersion: string
  /** Defaults to identity. Must be an orthonormal, orientation-preserving matrix. */
  plyExportTransform?: GaussianPlyExportTransform
}

export interface GaussianScene {
  readonly count: number
  readonly positions: Float32Array
  readonly scales: Float32Array
  readonly rotations: Float32Array
  readonly opacities: Float32Array
  /** Degree-zero RGB coefficients, interleaved per Gaussian. */
  readonly sphericalHarmonics?: Float32Array | undefined
  /** RGB values, interleaved per Gaussian, interpreted by `metadata.colorSemantics`. */
  readonly colors?: Float32Array | undefined
  readonly metadata: GaussianSceneMetadata
  readonly isDisposed: boolean
  exportPLY(): Promise<Blob>
  exportSplat(): Promise<Blob>
  dispose(): void
}

export interface CreateGaussianSceneInput {
  count: number
  positions: Float32Array
  scales: Float32Array
  rotations: Float32Array
  opacities: Float32Array
  sphericalHarmonics?: Float32Array
  colors?: Float32Array
  metadata: GaussianSceneMetadata
}
