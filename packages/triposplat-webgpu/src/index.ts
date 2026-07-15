export { TripoSplatWebGPU } from './triposplat.js'
export type {
  TripoSplatPipelineContext,
  TripoSplatPipelineExecutor,
  TripoSplatPreprocessor,
  TripoSplatWebGPUOptions,
} from './triposplat.js'
export { runBuiltInTripoSplatPipeline } from './pipeline.js'
export { clearModelCache, getModelCacheStatus } from './modelCache.js'
export type {
  ModelCacheBackendStatus,
  ModelCacheEntry,
  ModelCacheStatus,
  ModelCacheStatusOptions,
} from './modelCache.js'

export { checkCompatibility } from './compatibility.js'

export {
  BackgroundRemovalRequiredError,
  CancelledError,
  ExportError,
  GraphCapabilityError,
  GraphLoadError,
  InferenceError,
  ManifestError,
  ModelDownloadError,
  ModelIntegrityError,
  OutOfMemoryError,
  TripoSplatError,
  UnsupportedAdapterError,
  WebGPUUnavailableError,
} from './errors.js'
export type {
  TripoSplatDiagnostics,
  TripoSplatErrorCode,
  TripoSplatErrorOptions,
  TripoSplatErrorStage,
} from './errors.js'

export {
  buildTripoSplatEncoderTensors,
  decodeTripoSplatImageSource,
  DINOV3_IMAGE_MEAN,
  DINOV3_IMAGE_STD,
  normalizeTripoSplatImageInput,
  preprocessTripoSplatRgba,
  rgbImageToCanvas,
  rgbImageToImageBitmap,
  TRIPOSPLAT_CANVAS_SIZE,
} from './preprocess.js'
export type {
  NchwImageTensor,
  NormalizedTripoSplatImage,
  NormalizeTripoSplatImageOptions,
  RgbaImage,
  RgbImage,
  TripoSplatBackgroundRemover,
  TripoSplatCanvas,
  TripoSplatEncoderTensors,
  TripoSplatImageSource,
  TripoSplatPreprocessOptions,
  TripoSplatPreprocessResult,
} from './preprocess.js'

export type {
  CacheBackend,
  CompatibilityOptions,
  CompatibilityReport,
  ExecutionProvider,
  GenerateOptions,
  GenerationProgress,
  GenerationStage,
  GaussianScene,
  GaussianSceneMetadata,
  LoadOptions,
  LoadProgress,
  LoadStage,
  LogLevel,
  PipelineCapabilities,
  Precision,
  TripoSplatGraphName,
  TripoSplatInput,
  TripoSplatOptions,
} from './types.js'

export type { ResolvedTripoSplatModelManifest, TripoSplatModelManifest } from './manifest.js'
