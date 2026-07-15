export type TripoSplatErrorCode =
  | 'WEBGPU_UNAVAILABLE'
  | 'UNSUPPORTED_ADAPTER'
  | 'MODEL_DOWNLOAD_FAILED'
  | 'MODEL_INTEGRITY_FAILED'
  | 'MANIFEST_INVALID'
  | 'GRAPH_LOAD_FAILED'
  | 'GRAPH_CAPABILITY_UNAVAILABLE'
  | 'BACKGROUND_REMOVAL_REQUIRED'
  | 'INFERENCE_FAILED'
  | 'OUT_OF_MEMORY'
  | 'CANCELLED'
  | 'EXPORT_FAILED'
  | 'DISPOSED'

export type TripoSplatErrorStage =
  | 'initialization'
  | 'compatibility'
  | 'manifest'
  | 'download'
  | 'graph-load'
  | 'preprocessing'
  | 'inference'
  | 'export'
  | 'dispose'

export interface TripoSplatDiagnostics {
  [key: string]: unknown
}

export interface TripoSplatErrorOptions {
  code: TripoSplatErrorCode
  stage: TripoSplatErrorStage
  recoverable: boolean
  cause?: unknown
  diagnostics?: TripoSplatDiagnostics
}

export class TripoSplatError extends Error {
  readonly code: TripoSplatErrorCode
  readonly stage: TripoSplatErrorStage
  readonly recoverable: boolean
  readonly diagnostics: Readonly<TripoSplatDiagnostics>
  override readonly cause: unknown

  constructor(message: string, options: TripoSplatErrorOptions) {
    super(message)
    this.name = 'TripoSplatError'
    this.code = options.code
    this.stage = options.stage
    this.recoverable = options.recoverable
    this.cause = options.cause
    this.diagnostics = Object.freeze({ ...(options.diagnostics ?? {}) })
  }
}

type SpecializedOptions = Omit<TripoSplatErrorOptions, 'code' | 'stage' | 'recoverable'>

export class WebGPUUnavailableError extends TripoSplatError {
  constructor(message = 'WebGPU is not available in this browser context.', options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'WEBGPU_UNAVAILABLE', stage: 'compatibility', recoverable: false })
    this.name = 'WebGPUUnavailableError'
  }
}

export class UnsupportedAdapterError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'UNSUPPORTED_ADAPTER', stage: 'compatibility', recoverable: false })
    this.name = 'UnsupportedAdapterError'
  }
}

export class ModelDownloadError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'MODEL_DOWNLOAD_FAILED', stage: 'download', recoverable: true })
    this.name = 'ModelDownloadError'
  }
}

export class ModelIntegrityError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'MODEL_INTEGRITY_FAILED', stage: 'download', recoverable: true })
    this.name = 'ModelIntegrityError'
  }
}

export class ManifestError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'MANIFEST_INVALID', stage: 'manifest', recoverable: true })
    this.name = 'ManifestError'
  }
}

export class GraphLoadError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'GRAPH_LOAD_FAILED', stage: 'graph-load', recoverable: true })
    this.name = 'GraphLoadError'
  }
}

export class GraphCapabilityError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'GRAPH_CAPABILITY_UNAVAILABLE', stage: 'initialization', recoverable: true })
    this.name = 'GraphCapabilityError'
  }
}

export class BackgroundRemovalRequiredError extends TripoSplatError {
  constructor(
    message = 'Opaque TripoSplat input requires external foreground segmentation.',
    options: SpecializedOptions = {},
  ) {
    super(message, {
      ...options,
      code: 'BACKGROUND_REMOVAL_REQUIRED',
      stage: 'preprocessing',
      recoverable: true,
    })
    this.name = 'BackgroundRemovalRequiredError'
  }
}

export class InferenceError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'INFERENCE_FAILED', stage: 'inference', recoverable: true })
    this.name = 'InferenceError'
  }
}

export class OutOfMemoryError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'OUT_OF_MEMORY', stage: 'inference', recoverable: true })
    this.name = 'OutOfMemoryError'
  }
}

export class CancelledError extends TripoSplatError {
  constructor(message = 'The TripoSplat operation was cancelled.', options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'CANCELLED', stage: 'inference', recoverable: true })
    this.name = 'CancelledError'
  }
}

export class ExportError extends TripoSplatError {
  constructor(message: string, options: SpecializedOptions = {}) {
    super(message, { ...options, code: 'EXPORT_FAILED', stage: 'export', recoverable: false })
    this.name = 'ExportError'
  }
}

export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new CancelledError('The TripoSplat operation was cancelled.', { cause: signal.reason })
  }
}
