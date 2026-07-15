export type ExecutionProvider = 'webgpu' | 'wasm'
export type CacheBackend = 'opfs' | 'cache-api' | 'none'
export type LogLevel = 'silent' | 'error' | 'info' | 'debug'
export type Precision = 'fp16' | 'fp32'

export type TripoSplatGraphName =
  | 'dino'
  | 'vae'
  | 'dit'
  | 'octree'
  | 'gaussianDecoder'

export type GenerationStage =
  | 'preprocessing'
  | 'dino'
  | 'vae'
  | 'sampling'
  | 'octree'
  | 'gaussian-decoder'
  | 'packing'
  | 'complete'

export type LoadStage =
  | 'manifest'
  | 'compatibility'
  | 'runtime'
  | 'graphs'
  | 'complete'

export interface StageProgress<Stage extends string> {
  stage: Stage
  message: string
  /** Only present when this stage has a measurable denominator. */
  progress?: number
  loadedBytes?: number
  totalBytes?: number
}

export interface LoadProgress extends StageProgress<LoadStage> {
  graph?: TripoSplatGraphName
}

export interface GenerationProgress extends StageProgress<GenerationStage> {
  /** One-based official flow step, when sampling. */
  step?: number
  totalSteps?: number
  /** One-based DiT call including conditional and unconditional passes. */
  invocation?: number
  totalInvocations?: number
}

export interface TripoSplatOptions {
  modelBaseUrl: string
  manifestUrl?: string
  executionProviders?: ExecutionProvider[]
  cache?: CacheBackend
  workerUrl?: string | URL
  workerFactory?: () => Worker
  logLevel?: LogLevel
  /** Used for manifest and model-artifact requests. */
  fetch?: typeof globalThis.fetch
  manifestRequestInit?: Omit<RequestInit, 'signal'>
  /** Headers/credentials for graph and external-data requests; method and body stay package-controlled. */
  artifactRequestInit?: Omit<RequestInit, 'signal' | 'body' | 'method'>
  /** Explicit ONNX Runtime WASM asset prefix or file mapping. */
  wasmPaths?: string | { mjs?: string; wasm?: string }
}

export interface LoadOptions {
  signal?: AbortSignal
  onProgress?: (progress: LoadProgress) => void
}

export interface GenerateOptions {
  /** Official fast (4) or quality (20) shifted-flow schedule. Defaults to 20. */
  steps?: 4 | 20
  guidanceScale?: number
  shift?: number
  gaussianCount?: number
  seed?: number
  precision?: Precision
  /** Explicit fixtures bypass the package PRNG and enable cross-runtime parity tests. */
  vaeNoise?: Float32Array
  latentNoise?: Float32Array
  cameraNoise?: Float32Array
  /** Dynamic octree softmax temperature. Defaults to 1. */
  octreeTemperature?: number
  /** Alpha minimum-filter radius used by official preprocessing. Defaults to 1. */
  erodeRadius?: number
  signal?: AbortSignal
  onProgress?: (progress: GenerationProgress) => void
  /** Required for raw opaque inputs until a browser BiRefNet graph is configured. */
  inputIsPrepared?: boolean
}

export type TripoSplatInput =
  | Blob
  | File
  | ImageBitmap
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas

export type {
  CreateGaussianSceneInput as GaussianSceneData,
  GaussianOpacityEncoding,
  GaussianRotationOrder,
  GaussianScaleEncoding,
  GaussianScene,
  GaussianSceneMetadata,
  JsonPrimitive,
  JsonValue,
} from '@ai3d/gaussian-scene'

export interface CompatibilityOptions {
  requestAdapterOptions?: {
    powerPreference?: 'low-power' | 'high-performance'
    forceFallbackAdapter?: boolean
  }
  /** Optional exported-graph requirement checked against the adapter limit. */
  minimumStorageBufferBindingSize?: number
  /** Declared download size from a resolved model manifest. Not a GPU-memory estimate. */
  estimatedModelBytes?: number
  /** Include only when measured for this exact browser/device/model configuration. */
  estimatedPeakBytes?: number
}

export interface CompatibilityReport {
  supported: boolean
  level: 'supported' | 'experimental' | 'unsupported'
  browser: string
  webgpu: boolean
  adapterName?: string
  limits: Record<string, number>
  estimatedModelBytes: number
  estimatedPeakBytes?: number
  warnings: string[]
  blockers: string[]
}

export interface PipelineCapabilities {
  manifestLoaded: boolean
  configuredGraphs: readonly TripoSplatGraphName[]
  missingGraphs: readonly TripoSplatGraphName[]
  encoderSlice: boolean
  fullGeneration: boolean
  reasons: readonly string[]
}
