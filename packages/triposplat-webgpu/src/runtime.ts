import { CancelledError, GraphLoadError, InferenceError, TripoSplatError } from './errors.js'
import type { ResolvedGraphManifestEntry } from './manifest.js'
import { assertTensorMap, tensorTransferables, type TensorMap } from './tensors.js'
import type { ExecutionProvider } from './types.js'

export interface RuntimeConfiguration {
  wasmPaths?: string | { mjs?: string; wasm?: string }
  wasmThreads?: number
  wasmSimd?: boolean | 'fixed' | 'relaxed'
}

export interface RuntimeStatus {
  stage:
    | 'runtime-configuring'
    | 'runtime-ready'
    | 'graph-loading'
    | 'graph-ready'
    | 'inference-queued'
    | 'inference-running'
    | 'outputs-reading'
    | 'inference-complete'
    | 'graph-disposing'
    | 'graph-disposed'
    | 'runtime-disposed'
  message: string
  timestampMs: number
  sessionId?: string
  provider?: ExecutionProvider
  progress?: number
}

export interface CreateRuntimeOptions {
  workerUrl?: string | URL
  workerFactory?: () => Worker
  baseUrl?: string | URL
  executionProviders?: ExecutionProvider[]
  configuration?: RuntimeConfiguration
  onStatus?: (status: RuntimeStatus) => void
}

export interface LoadGraphOptions {
  executionProviders?: ExecutionProvider[]
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'layout' | 'all'
  freeDimensionOverrides?: Readonly<Record<string, number>>
  enableGraphCapture?: boolean
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4
  /** Used by verified artifact preparation wrappers; not cloned into the worker. */
  signal?: AbortSignal
}

export interface GraphInfo {
  sessionId: string
  executionProvider: ExecutionProvider
  inputNames: string[]
  outputNames: string[]
  loadMs: number
}

export interface RunGraphOptions {
  outputs?: readonly string[]
  tag?: string
  signal?: AbortSignal
  /** Defaults to true; transferred input arrays are detached. */
  transferInputs?: boolean
}

export interface GraphRunResult {
  outputs: TensorMap
  timings: {
    inferenceMs: number
    readbackMs: number
    totalMs: number
  }
}

export interface TripoSplatRuntime {
  readonly disposed: boolean
  loadGraph(sessionId: string, graph: ResolvedGraphManifestEntry, options?: LoadGraphOptions): Promise<GraphInfo>
  runGraph(sessionId: string, inputs: TensorMap, options?: RunGraphOptions): Promise<GraphRunResult>
  disposeGraph(sessionId: string): Promise<boolean>
  dispose(): Promise<void>
}

interface WorkerConfigureRequest {
  type: 'configure'
  requestId: string
  configuration: RuntimeConfiguration
}

interface WorkerLoadRequest {
  type: 'load'
  requestId: string
  sessionId: string
  graph: ResolvedGraphManifestEntry
  options: LoadGraphOptions
}

interface WorkerRunRequest {
  type: 'run'
  requestId: string
  sessionId: string
  inputs: TensorMap
  reusableInputsId?: string
  outputs?: readonly string[]
  tag?: string
}

interface WorkerRetainInputsRequest {
  type: 'retain-inputs'
  requestId: string
  sessionId: string
  reusableInputsId: string
  inputs: TensorMap
}

interface WorkerDisposeGraphRequest {
  type: 'dispose-graph'
  requestId: string
  sessionId: string
}

interface WorkerDisposeRequest {
  type: 'dispose'
  requestId: string
}

export type RuntimeWorkerRequest =
  | WorkerConfigureRequest
  | WorkerLoadRequest
  | WorkerRetainInputsRequest
  | WorkerRunRequest
  | WorkerDisposeGraphRequest
  | WorkerDisposeRequest

export type RuntimeWorkerResult =
  | { operation: 'configure'; configured: true }
  | { operation: 'load'; graph: GraphInfo }
  | { operation: 'retain-inputs'; retainedInputNames: string[] }
  | { operation: 'run'; result: GraphRunResult }
  | { operation: 'dispose-graph'; disposed: boolean }
  | { operation: 'dispose'; disposedSessionIds: string[] }

export type RuntimeWorkerMessage =
  | { type: 'status'; status: RuntimeStatus }
  | { type: 'reply'; requestId: string; ok: true; result: RuntimeWorkerResult }
  | { type: 'reply'; requestId: string; ok: false; error: { name: string; message: string; stack?: string } }

interface Pending {
  operation: RuntimeWorkerRequest['type']
  resolve: (value: RuntimeWorkerResult) => void
  reject: (error: Error) => void
}

function id(): string {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function assertSessionId(sessionId: string): void {
  if (sessionId.trim().length === 0 || sessionId.includes('\0')) throw new TypeError('sessionId is invalid.')
}

function assertReusableInputsId(reusableInputsId: string): void {
  if (reusableInputsId.trim().length === 0 || reusableInputsId.includes('\0')) {
    throw new TypeError('reusableInputsId is invalid.')
  }
}

function cloneTensorMap(inputs: Readonly<TensorMap>): TensorMap {
  const result: TensorMap = {}
  for (const [name, tensor] of Object.entries(inputs)) {
    switch (tensor.type) {
      case 'float32':
        result[name] = { type: tensor.type, dims: [...tensor.dims], data: new Float32Array(tensor.data) }
        break
      case 'float16':
        result[name] = { type: tensor.type, dims: [...tensor.dims], data: new Uint16Array(tensor.data) }
        break
      case 'int32':
        result[name] = { type: tensor.type, dims: [...tensor.dims], data: new Int32Array(tensor.data) }
        break
      case 'int64':
        result[name] = { type: tensor.type, dims: [...tensor.dims], data: new BigInt64Array(tensor.data) }
        break
    }
  }
  return result
}

function assertDisjointInputs(reusable: Readonly<TensorMap>, dynamic: Readonly<TensorMap>): void {
  for (const name of Object.keys(dynamic)) {
    if (Object.prototype.hasOwnProperty.call(reusable, name)) {
      throw new TypeError(`Input '${name}' cannot be both reusable and dynamic.`)
    }
  }
}

interface ReusableGraphInputCapability {
  retain(
    sessionId: string,
    reusableInputsId: string,
    inputs: TensorMap,
    signal?: AbortSignal,
  ): Promise<void>
  run(
    sessionId: string,
    reusableInputsId: string,
    inputs: TensorMap,
    options?: RunGraphOptions,
  ): Promise<GraphRunResult>
}

const reusableGraphInputCapabilities = new WeakMap<TripoSplatRuntime, ReusableGraphInputCapability>()

/** Internal capability forwarding used by runtime decorators. Not part of the package entrypoints. */
export function forwardReusableGraphInputCapability(
  source: TripoSplatRuntime,
  target: TripoSplatRuntime,
): void {
  const capability = reusableGraphInputCapabilities.get(source)
  if (capability) reusableGraphInputCapabilities.set(target, capability)
}

export interface PreparedReusableGraphInputs {
  run(inputs: TensorMap, options?: RunGraphOptions): Promise<GraphRunResult>
}

/**
 * Retain immutable graph inputs in the built-in worker for a session's lifetime.
 * Custom runtimes fall back to a fresh clone per call, preserving the public
 * TripoSplatRuntime contract and its default transfer/detach semantics.
 */
export async function prepareReusableGraphInputs(
  runtime: TripoSplatRuntime,
  sessionId: string,
  reusableInputsId: string,
  inputs: TensorMap,
  signal?: AbortSignal,
): Promise<PreparedReusableGraphInputs> {
  assertSessionId(sessionId)
  assertReusableInputsId(reusableInputsId)
  assertTensorMap(inputs)
  if (signal?.aborted) throw new CancelledError(undefined, { cause: signal.reason })
  const capability = reusableGraphInputCapabilities.get(runtime)
  if (capability) {
    await capability.retain(sessionId, reusableInputsId, inputs, signal)
    return {
      run(dynamicInputs, options) {
        assertTensorMap(dynamicInputs)
        assertDisjointInputs(inputs, dynamicInputs)
        return capability.run(sessionId, reusableInputsId, dynamicInputs, options)
      },
    }
  }

  const template = cloneTensorMap(inputs)
  return {
    run(dynamicInputs, options) {
      assertTensorMap(dynamicInputs)
      assertDisjointInputs(template, dynamicInputs)
      return runtime.runGraph(sessionId, { ...cloneTensorMap(template), ...dynamicInputs }, options)
    },
  }
}

function resolveBaseUrl(value?: string | URL): string {
  if (value !== undefined) return new URL(value).href
  if (typeof document !== 'undefined') return document.baseURI
  if (typeof location !== 'undefined') return location.href
  return import.meta.url
}

function packagedWasmPaths(): { mjs: string; wasm: string } {
  // Static URL expressions are intentional: Vite, Rollup and webpack discover
  // and emit these package-owned assets when the runtime is installed from npm.
  return {
    mjs: new URL('./ort/ort-wasm-simd-threaded.asyncify.mjs', import.meta.url).href,
    wasm: new URL('./ort/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url).href,
  }
}

class WorkerRuntime implements TripoSplatRuntime {
  private readonly worker: Worker
  private readonly pending = new Map<string, Pending>()
  private readonly providers: ExecutionProvider[]
  private readonly ready: Promise<void>
  private onStatus: ((status: RuntimeStatus) => void) | undefined
  private disposedValue = false
  private disposePromise: Promise<void> | undefined
  private workerTerminatedValue = false
  private fatalError?: Error

  constructor(options: CreateRuntimeOptions) {
    const baseUrl = resolveBaseUrl(options.baseUrl)
    this.providers = options.executionProviders ?? ['webgpu']
    if (this.providers.length === 0) throw new TypeError('executionProviders must not be empty.')
    const workerUrl = options.workerUrl === undefined
      ? new URL('./worker.js', import.meta.url)
      : new URL(options.workerUrl, baseUrl)
    this.worker = options.workerFactory?.() ?? new Worker(workerUrl, {
      type: 'module',
      name: 'triposplat-onnx-webgpu',
    })
    this.onStatus = options.onStatus
    this.worker.onmessage = (event: MessageEvent<RuntimeWorkerMessage>) => this.handleMessage(event.data)
    this.worker.onerror = (event: ErrorEvent) => {
      this.fail(new Error(event.message || 'TripoSplat runtime worker failed.'))
    }
    this.worker.onmessageerror = () => this.fail(new Error('Could not deserialize a runtime worker message.'))
    const configuration = { ...(options.configuration ?? {}) }
    configuration.wasmPaths ??= packagedWasmPaths()
    this.ready = this.send({
      type: 'configure',
      requestId: id(),
      configuration,
    }).then(() => undefined)
    void this.ready.catch(() => undefined)
    reusableGraphInputCapabilities.set(this, {
      retain: (sessionId, reusableInputsId, inputs, signal) => (
        this.retainGraphInputs(sessionId, reusableInputsId, inputs, signal)
      ),
      run: (sessionId, reusableInputsId, inputs, runOptions) => (
        this.runGraphWithReusableInputs(sessionId, reusableInputsId, inputs, runOptions)
      ),
    })
  }

  get disposed(): boolean {
    return this.disposedValue
  }

  async loadGraph(
    sessionId: string,
    graph: ResolvedGraphManifestEntry,
    options: LoadGraphOptions = {},
  ): Promise<GraphInfo> {
    assertSessionId(sessionId)
    if (options.signal?.aborted) throw new CancelledError(undefined, { cause: options.signal.reason })
    await this.waitUntilReady(options.signal)
    const workerOptions = { ...options }
    delete workerOptions.signal
    try {
      const result = await this.sendWithSignal(
        {
          type: 'load',
          requestId: id(),
          sessionId,
          graph,
          options: {
            ...workerOptions,
            executionProviders: options.executionProviders ?? this.providers,
          },
        },
        [],
        options.signal,
      )
      if (result.operation !== 'load') throw new Error('Runtime worker returned a mismatched load result.')
      return result.graph
    } catch (cause) {
      if (cause instanceof CancelledError) throw cause
      throw new GraphLoadError(`Could not load ONNX graph '${sessionId}'.`, {
        cause,
        diagnostics: { sessionId, graphUrl: graph.url },
      })
    }
  }

  async runGraph(
    sessionId: string,
    inputs: TensorMap,
    options: RunGraphOptions = {},
  ): Promise<GraphRunResult> {
    assertSessionId(sessionId)
    assertTensorMap(inputs)
    const request: WorkerRunRequest = { type: 'run', requestId: id(), sessionId, inputs }
    return this.executeGraphRequest(request, options)
  }

  private async retainGraphInputs(
    sessionId: string,
    reusableInputsId: string,
    inputs: TensorMap,
    signal?: AbortSignal,
  ): Promise<void> {
    assertSessionId(sessionId)
    assertReusableInputsId(reusableInputsId)
    assertTensorMap(inputs)
    if (signal?.aborted) throw new CancelledError(undefined, { cause: signal.reason })
    await this.waitUntilReady(signal)
    if (signal?.aborted) throw new CancelledError(undefined, { cause: signal.reason })
    try {
      const result = await this.sendWithSignal(
        {
          type: 'retain-inputs',
          requestId: id(),
          sessionId,
          reusableInputsId,
          inputs,
        },
        tensorTransferables(inputs),
        signal,
      )
      if (result.operation !== 'retain-inputs') {
        throw new Error('Runtime worker returned a mismatched retain-inputs result.')
      }
    } catch (cause) {
      if (cause instanceof CancelledError) throw cause
      throw new InferenceError(`Could not retain reusable ONNX inputs for '${sessionId}'.`, {
        cause,
        diagnostics: { sessionId, reusableInputsId },
      })
    }
  }

  private runGraphWithReusableInputs(
    sessionId: string,
    reusableInputsId: string,
    inputs: TensorMap,
    options: RunGraphOptions = {},
  ): Promise<GraphRunResult> {
    assertSessionId(sessionId)
    assertReusableInputsId(reusableInputsId)
    assertTensorMap(inputs)
    const request: WorkerRunRequest = {
      type: 'run',
      requestId: id(),
      sessionId,
      reusableInputsId,
      inputs,
    }
    return this.executeGraphRequest(request, options)
  }

  private async executeGraphRequest(
    request: WorkerRunRequest,
    options: RunGraphOptions,
  ): Promise<GraphRunResult> {
    if (options.signal?.aborted) throw new CancelledError(undefined, { cause: options.signal.reason })
    await this.waitUntilReady(options.signal)
    if (options.signal?.aborted) throw new CancelledError(undefined, { cause: options.signal.reason })
    if (options.outputs !== undefined) request.outputs = options.outputs
    if (options.tag !== undefined) request.tag = options.tag
    const transfer = options.transferInputs === false ? [] : tensorTransferables(request.inputs)
    try {
      const result = await this.sendWithSignal(request, transfer, options.signal)
      if (result.operation !== 'run') throw new Error('Runtime worker returned a mismatched run result.')
      return result.result
    } catch (cause) {
      if (cause instanceof CancelledError) throw cause
      throw new InferenceError(`ONNX inference failed for '${request.sessionId}'.`, {
        cause,
        diagnostics: { sessionId: request.sessionId, tag: options.tag },
      })
    }
  }

  async disposeGraph(sessionId: string): Promise<boolean> {
    assertSessionId(sessionId)
    if (this.disposedValue) return false
    await this.ready
    const result = await this.send({ type: 'dispose-graph', requestId: id(), sessionId })
    if (result.operation !== 'dispose-graph') throw new Error('Runtime worker returned a mismatched dispose result.')
    return result.disposed
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise
    this.disposedValue = true
    this.disposePromise = this.performDispose()
    return this.disposePromise
  }

  private async performDispose(): Promise<void> {
    const disposedError = new TripoSplatError('TripoSplat runtime has been disposed.', {
      code: 'DISPOSED', stage: 'dispose', recoverable: false,
    })
    // Waiting for a submitted ORT graph (or even worker initialization) can
    // make dispose hang indefinitely. Worker termination is the only safe
    // cancellation boundary while a request is outstanding.
    if (this.fatalError || this.pending.size > 0) {
      this.terminateWorker()
      this.rejectAll(disposedError)
      return
    }
    try {
      await this.ready
      if (!this.fatalError) await this.send({ type: 'dispose', requestId: id() }, [], true)
    } finally {
      this.terminateWorker()
      this.rejectAll(disposedError)
    }
  }

  private async waitUntilReady(signal?: AbortSignal): Promise<void> {
    if (signal === undefined) {
      await this.ready
      return
    }
    if (signal.aborted) throw new CancelledError(undefined, { cause: signal.reason })
    let abortListener: (() => void) | undefined
    const cancellation = new Promise<never>((_resolve, reject) => {
      abortListener = () => {
        const error = new CancelledError(undefined, { cause: signal.reason })
        // The configure request is already in flight. Terminate it just like a
        // submitted graph request so model-level retry cannot hang in dispose().
        this.fail(error)
        reject(error)
      }
      signal.addEventListener('abort', abortListener, { once: true })
    })
    try {
      await Promise.race([this.ready, cancellation])
    } finally {
      if (abortListener) signal.removeEventListener('abort', abortListener)
    }
  }

  private async sendWithSignal(
    request: RuntimeWorkerRequest,
    transfer: Transferable[],
    signal?: AbortSignal,
  ): Promise<RuntimeWorkerResult> {
    if (signal === undefined) return this.send(request, transfer)
    if (signal.aborted) throw new CancelledError(undefined, { cause: signal.reason })
    let abortListener: (() => void) | undefined
    const cancellation = new Promise<never>((_resolve, reject) => {
      abortListener = () => {
        const error = new CancelledError(undefined, { cause: signal.reason })
        // ORT cannot cancel submitted graph creation or execution safely. The
        // worker is therefore single-use after an in-flight request is aborted.
        this.fail(error)
        reject(error)
      }
      signal.addEventListener('abort', abortListener, { once: true })
    })
    try {
      return await Promise.race([this.send(request, transfer), cancellation])
    } finally {
      if (abortListener) signal.removeEventListener('abort', abortListener)
    }
  }

  private send(
    request: RuntimeWorkerRequest,
    transfer: Transferable[] = [],
    allowDisposed = false,
  ): Promise<RuntimeWorkerResult> {
    if (this.fatalError) return Promise.reject(this.fatalError)
    if (this.disposedValue && !allowDisposed) {
      return Promise.reject(new TripoSplatError('TripoSplat runtime has been disposed.', {
        code: 'DISPOSED', stage: 'dispose', recoverable: false,
      }))
    }
    const promise = new Promise<RuntimeWorkerResult>((resolve, reject) => {
      this.pending.set(request.requestId, { operation: request.type, resolve, reject })
    })
    try {
      this.worker.postMessage(request, transfer)
    } catch (cause) {
      this.pending.delete(request.requestId)
      return Promise.reject(cause)
    }
    return promise
  }

  private handleMessage(message: RuntimeWorkerMessage): void {
    if (message.type === 'status') {
      this.onStatus?.(message.status)
      return
    }
    const pending = this.pending.get(message.requestId)
    if (!pending) return
    this.pending.delete(message.requestId)
    if (message.ok) pending.resolve(message.result)
    else {
      const error = new Error(message.error.message)
      error.name = message.error.name
      if (message.error.stack) error.stack = message.error.stack
      pending.reject(error)
    }
  }

  private fail(error: Error): void {
    if (this.fatalError) return
    this.fatalError = error
    this.terminateWorker()
    this.rejectAll(error)
  }

  private terminateWorker(): void {
    if (this.workerTerminatedValue) return
    this.workerTerminatedValue = true
    this.worker.terminate()
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
  }
}

export function createRuntime(options: CreateRuntimeOptions = {}): TripoSplatRuntime {
  return new WorkerRuntime(options)
}

export function loadGraph(
  runtime: TripoSplatRuntime,
  sessionId: string,
  graph: ResolvedGraphManifestEntry,
  options?: LoadGraphOptions,
): Promise<GraphInfo> {
  return runtime.loadGraph(sessionId, graph, options)
}

export function runGraph(
  runtime: TripoSplatRuntime,
  sessionId: string,
  inputs: TensorMap,
  options?: RunGraphOptions,
): Promise<GraphRunResult> {
  return runtime.runGraph(sessionId, inputs, options)
}
