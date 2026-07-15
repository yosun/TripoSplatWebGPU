import type { OnnxModelManifest } from './modelManifest'
import { copyModelManifest, resolveModelManifest } from './modelManifest'
import type { TensorPayloadMap } from './tensors'
import { assertTensorPayloadMap, tensorPayloadTransferables } from './tensors'

export type OrtExecutionProvider = 'webgpu' | 'wasm'

export interface OrtRuntimeConfiguration {
  /** Absolute prefix or explicit files for the ONNX Runtime WASM artifacts. */
  wasmPaths?: string | { mjs?: string; wasm?: string }
  wasmThreads?: number
  wasmSimd?: boolean | 'fixed' | 'relaxed'
}

export interface OrtWebGpuSessionOptions {
  preferredLayout?: 'NCHW' | 'NHWC'
  forceCpuNodeNames?: readonly string[]
  validationMode?: 'disabled' | 'wgpuOnly' | 'basic' | 'full'
}

export interface OrtSessionLoadOptions {
  /** If WebGPU session creation fails, retry the whole graph with WASM. Defaults to false. */
  allowWasmFallback?: boolean
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'layout' | 'all'
  freeDimensionOverrides?: Readonly<Record<string, number>>
  enableGraphCapture?: boolean
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4
  webgpu?: OrtWebGpuSessionOptions
}

export interface OrtLoadSessionRequest {
  sessionId: string
  manifest: OnnxModelManifest
  options?: OrtSessionLoadOptions
}

export interface OrtRunSessionRequest {
  sessionId: string
  inputs: TensorPayloadMap
  /** Omit to fetch all graph outputs. */
  outputs?: readonly string[]
  tag?: string
}

export interface OrtRunClientOptions {
  /**
   * Move input buffers into the worker instead of cloning them. Defaults to
   * true. The caller's typed arrays are detached after posting when enabled.
   */
  transferInputs?: boolean
}

export interface OrtValueMetadata {
  name: string
  isTensor: boolean
  type?: string
  shape?: Array<number | string>
}

export interface OrtConfigureRuntimeResult {
  wasmThreads: number
  wasmSimd: boolean | 'fixed' | 'relaxed'
  wasmPaths: string | { mjs?: string; wasm?: string }
}

export interface OrtLoadSessionResult {
  sessionId: string
  executionProvider: OrtExecutionProvider
  inputNames: string[]
  outputNames: string[]
  inputMetadata: OrtValueMetadata[]
  outputMetadata: OrtValueMetadata[]
  loadMs: number
}

export interface OrtRunTimings {
  /** Time spent inside `InferenceSession.run`. */
  inferenceMs: number
  /** Time spent materializing and copying outputs to transferable CPU buffers. */
  readbackMs: number
  totalMs: number
}

export interface OrtRunSessionResult {
  sessionId: string
  outputs: TensorPayloadMap
  timings: OrtRunTimings
}

export interface OrtDisposeSessionResult {
  sessionId: string
  disposed: boolean
}

export interface OrtDisposeAllResult {
  disposedSessionIds: string[]
}

export interface OrtWorkerRequestPayloadMap {
  'configure-runtime': OrtRuntimeConfiguration
  'load-session': OrtLoadSessionRequest
  'run-session': OrtRunSessionRequest
  'dispose-session': { sessionId: string }
  'dispose-all': Record<string, never>
}

export interface OrtWorkerResultMap {
  'configure-runtime': OrtConfigureRuntimeResult
  'load-session': OrtLoadSessionResult
  'run-session': OrtRunSessionResult
  'dispose-session': OrtDisposeSessionResult
  'dispose-all': OrtDisposeAllResult
}

export type OrtWorkerOperation = keyof OrtWorkerRequestPayloadMap

export type OrtWorkerRequest = {
  [Operation in OrtWorkerOperation]: {
    type: Operation
    requestId: string
    payload: OrtWorkerRequestPayloadMap[Operation]
  }
}[OrtWorkerOperation]

export interface SerializedWorkerError {
  name: string
  message: string
  stack?: string
}

export type OrtWorkerReply = {
  [Operation in OrtWorkerOperation]:
    | {
        type: 'reply'
        operation: Operation
        requestId: string
        ok: true
        result: OrtWorkerResultMap[Operation]
      }
    | {
        type: 'reply'
        operation: Operation
        requestId: string
        ok: false
        error: SerializedWorkerError
      }
}[OrtWorkerOperation]

export type OrtWorkerStage =
  | 'runtime-configuring'
  | 'runtime-ready'
  | 'session-loading'
  | 'session-fallback'
  | 'session-ready'
  | 'inference-queued'
  | 'inference-running'
  | 'outputs-reading'
  | 'inference-complete'
  | 'session-disposing'
  | 'session-disposed'
  | 'worker-disposing'
  | 'worker-disposed'

export interface OrtWorkerStatus {
  type: 'status'
  stage: OrtWorkerStage
  message: string
  timestampMs: number
  requestId?: string
  sessionId?: string
  executionProvider?: OrtExecutionProvider
  /** Normalized progress within this stage, when meaningful. */
  progress?: number
}

export type OrtWorkerMessage = OrtWorkerReply | OrtWorkerStatus

export interface OrtWorkerClientOptions {
  onStatus?: (status: OrtWorkerStatus) => void
  /** Configure ORT before the first model is loaded. Worker defaults are used when omitted. */
  runtime?: OrtRuntimeConfiguration
  /** Base used to resolve relative graph and external-data URLs before posting. */
  baseUrl?: string | URL
  /** Test/embed hook. The default creates the generic module worker. */
  workerFactory?: () => Worker
}

interface PendingRequest {
  operation: OrtWorkerOperation
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

function requestId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function defaultBaseUrl(): string {
  if (typeof document !== 'undefined' && document.baseURI) {
    return document.baseURI
  }
  if (typeof location !== 'undefined') {
    return location.href
  }
  throw new Error('OrtWorkerClient needs an explicit baseUrl outside a browser document.')
}

function assertSessionId(sessionId: unknown): asserts sessionId is string {
  if (typeof sessionId !== 'string' || sessionId.trim().length === 0 || sessionId.includes('\0')) {
    throw new TypeError('sessionId must be a non-empty string without null characters.')
  }
}

function deserializeError(error: SerializedWorkerError): Error {
  const result = new Error(error.message)
  result.name = error.name
  if (error.stack) {
    result.stack = error.stack
  }
  return result
}

function isWorkerMessage(value: unknown): value is OrtWorkerMessage {
  if (typeof value !== 'object' || value === null || !('type' in value)) {
    return false
  }
  const type = (value as { type?: unknown }).type
  return type === 'status' || type === 'reply'
}

/** Correlated, typed facade over the generic multi-session ONNX worker. */
export class OrtWorkerClient {
  private readonly worker: Worker
  private readonly baseUrl: string
  private readonly pending = new Map<string, PendingRequest>()
  private onStatus?: (status: OrtWorkerStatus) => void
  private ready: Promise<unknown> = Promise.resolve()
  private fatalError?: Error
  private disposePromise?: Promise<void>

  constructor(options: OrtWorkerClientOptions = {}) {
    this.baseUrl = new URL(options.baseUrl ?? defaultBaseUrl()).href
    this.onStatus = options.onStatus
    this.worker = options.workerFactory?.() ?? new Worker(new URL('../workers/onnxWorker.ts', import.meta.url), {
      type: 'module',
      name: 'onnx-runtime-webgpu',
    })

    this.worker.onmessage = (event: MessageEvent<unknown>) => {
      this.handleMessage(event.data)
    }
    this.worker.onerror = (event: ErrorEvent) => {
      const cause = event.error instanceof Error ? event.error.message : undefined
      const location = event.filename
        ? `${event.filename}${event.lineno ? `:${event.lineno}${event.colno ? `:${event.colno}` : ''}` : ''}`
        : undefined
      const details = [event.message, cause, location].filter((value): value is string => Boolean(value))
      this.fail(new Error(details.length > 0 ? `ONNX worker error: ${details.join(' · ')}` : 'ONNX worker error.'))
    }
    this.worker.onmessageerror = () => {
      this.fail(new Error('Could not deserialize a message from the ONNX worker.'))
    }

    if (options.runtime) {
      this.ready = this.sendRequest('configure-runtime', options.runtime)
      // Avoid an unhandled-rejection report before the first public call awaits readiness.
      void this.ready.catch(() => undefined)
    }
  }

  setStatusHandler(handler?: (status: OrtWorkerStatus) => void): void {
    this.onStatus = handler
  }

  async configureRuntime(configuration: OrtRuntimeConfiguration): Promise<OrtConfigureRuntimeResult> {
    await this.ready
    const configuring = this.sendRequest('configure-runtime', configuration)
    this.ready = configuring
    return configuring
  }

  async loadSession(request: OrtLoadSessionRequest): Promise<OrtLoadSessionResult> {
    assertSessionId(request.sessionId)
    const manifest = resolveModelManifest(copyModelManifest(request.manifest), this.baseUrl)
    await this.ready
    return this.sendRequest('load-session', {
      sessionId: request.sessionId,
      manifest,
      options: request.options,
    })
  }

  async runSession(
    request: OrtRunSessionRequest,
    options: OrtRunClientOptions = {},
  ): Promise<OrtRunSessionResult> {
    assertSessionId(request.sessionId)
    assertTensorPayloadMap(request.inputs, 'request.inputs')
    if (request.outputs !== undefined) {
      const seen = new Set<string>()
      for (const output of request.outputs) {
        if (typeof output !== 'string' || output.trim().length === 0 || seen.has(output)) {
          throw new TypeError('Requested output names must be unique, non-empty strings.')
        }
        seen.add(output)
      }
    }

    await this.ready
    const transfer = options.transferInputs === false ? [] : tensorPayloadTransferables(request.inputs)
    return this.sendRequest('run-session', request, transfer)
  }

  async disposeSession(sessionId: string): Promise<OrtDisposeSessionResult> {
    assertSessionId(sessionId)
    await this.ready
    return this.sendRequest('dispose-session', { sessionId })
  }

  async disposeAllSessions(): Promise<OrtDisposeAllResult> {
    await this.ready
    return this.sendRequest('dispose-all', {})
  }

  /** Release every ORT session, acknowledge disposal, and then terminate the worker. */
  dispose(): Promise<void> {
    if (this.disposePromise) {
      return this.disposePromise
    }

    this.disposePromise = (async () => {
      try {
        await this.ready
        if (!this.fatalError) {
          await this.sendRequest('dispose-all', {}, [], true)
        }
      } finally {
        this.worker.terminate()
        this.rejectAll(new Error('ONNX worker client disposed.'))
      }
    })()
    return this.disposePromise
  }

  private sendRequest<Operation extends OrtWorkerOperation>(
    operation: Operation,
    payload: OrtWorkerRequestPayloadMap[Operation],
    transfer: Transferable[] = [],
    allowDuringDispose = false,
  ): Promise<OrtWorkerResultMap[Operation]> {
    if (this.fatalError) {
      return Promise.reject(this.fatalError)
    }
    if (this.disposePromise && !allowDuringDispose) {
      return Promise.reject(new Error('ONNX worker client is disposing or disposed.'))
    }

    const id = requestId()
    const promise = new Promise<OrtWorkerResultMap[Operation]>((resolve, reject) => {
      this.pending.set(id, {
        operation,
        resolve: resolve as (value: unknown) => void,
        reject,
      })
    })

    const message: OrtWorkerRequest = { type: operation, requestId: id, payload } as OrtWorkerRequest
    try {
      this.worker.postMessage(message, transfer)
    } catch (error) {
      this.pending.delete(id)
      const typedError = error instanceof Error ? error : new Error(String(error))
      return Promise.reject(typedError)
    }
    return promise
  }

  private handleMessage(value: unknown): void {
    if (!isWorkerMessage(value)) {
      this.fail(new Error('Received a malformed message from the ONNX worker.'))
      return
    }
    if (value.type === 'status') {
      this.onStatus?.(value)
      return
    }

    const pending = this.pending.get(value.requestId)
    if (!pending) {
      return
    }
    this.pending.delete(value.requestId)
    if (pending.operation !== value.operation) {
      pending.reject(
        new Error(`ONNX worker replied to '${pending.operation}' with mismatched operation '${value.operation}'.`),
      )
      return
    }
    if (value.ok) {
      pending.resolve(value.result)
    } else {
      pending.reject(deserializeError(value.error))
    }
  }

  private fail(error: Error): void {
    if (this.fatalError) {
      return
    }
    this.fatalError = error
    this.worker.terminate()
    this.rejectAll(error)
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.reject(error)
    }
    this.pending.clear()
  }
}
