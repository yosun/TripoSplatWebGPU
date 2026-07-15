/// <reference lib="WebWorker" />

import * as ort from 'onnxruntime-web/webgpu'

import type {
  GraphInfo,
  GraphRunResult,
  RuntimeConfiguration,
  RuntimeStatus,
  RuntimeWorkerMessage,
  RuntimeWorkerRequest,
  RuntimeWorkerResult,
} from './runtime.js'
import { assertTensorMap, createTensor, tensorTransferables, type TensorPayload } from './tensors.js'
import type { ExecutionProvider } from './types.js'

const scope = self as unknown as DedicatedWorkerGlobalScope

interface SessionRecord {
  session: ort.InferenceSession
  provider: ExecutionProvider
  graphInfo: GraphInfo
  runTail: Promise<void>
  reusableInputs: Map<string, Record<string, ort.Tensor>>
}

const sessions = new Map<string, SessionRecord>()
let configured = false

function status(value: Omit<RuntimeStatus, 'timestampMs'>): void {
  const message: RuntimeWorkerMessage = {
    type: 'status',
    status: { ...value, timestampMs: Date.now() },
  }
  scope.postMessage(message)
}

function success(requestId: string, result: RuntimeWorkerResult, transfer: Transferable[] = []): void {
  const message: RuntimeWorkerMessage = { type: 'reply', requestId, ok: true, result }
  scope.postMessage(message, transfer)
}

function failure(requestId: string, value: unknown): void {
  const error = value instanceof Error ? value : new Error(String(value))
  const serialized: { name: string; message: string; stack?: string } = {
    name: error.name,
    message: error.message,
  }
  if (error.stack) serialized.stack = error.stack
  const message: RuntimeWorkerMessage = { type: 'reply', requestId, ok: false, error: serialized }
  scope.postMessage(message)
}

function defaultWasmPaths(): { mjs: string; wasm: string } {
  const base = new URL('./ort/', scope.location.href)
  return {
    mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs', base).href,
    wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', base).href,
  }
}

function configure(configuration: RuntimeConfiguration): void {
  if (configured) return
  status({ stage: 'runtime-configuring', message: 'Configuring ONNX Runtime.' })
  const threads = configuration.wasmThreads
    ?? (scope.crossOriginIsolated ? Math.max(1, Math.min(4, scope.navigator.hardwareConcurrency || 2)) : 1)
  if (!Number.isInteger(threads) || threads < 1) throw new RangeError('wasmThreads must be positive.')
  ort.env.wasm.numThreads = threads
  ort.env.wasm.simd = configuration.wasmSimd ?? true
  if (typeof configuration.wasmPaths === 'string') {
    ort.env.wasm.wasmPaths = new URL(configuration.wasmPaths, scope.location.href).href
  } else if (configuration.wasmPaths) {
    const paths: { mjs?: string; wasm?: string } = {}
    if (configuration.wasmPaths.mjs !== undefined) {
      paths.mjs = new URL(configuration.wasmPaths.mjs, scope.location.href).href
    }
    if (configuration.wasmPaths.wasm !== undefined) {
      paths.wasm = new URL(configuration.wasmPaths.wasm, scope.location.href).href
    }
    if (paths.mjs === undefined && paths.wasm === undefined) {
      throw new TypeError('wasmPaths must include mjs or wasm.')
    }
    ort.env.wasm.wasmPaths = paths
  } else {
    ort.env.wasm.wasmPaths = defaultWasmPaths()
  }
  configured = true
  status({ stage: 'runtime-ready', message: 'ONNX Runtime is configured.', progress: 1 })
}

function webGpuOptions(
  graph: RuntimeWorkerRequest & { type: 'load' },
): ort.InferenceSession.SessionOptions {
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }],
    graphOptimizationLevel: graph.options.graphOptimizationLevel ?? 'all',
    preferredOutputLocation: 'cpu',
  }
  if (graph.graph.externalData !== undefined) {
    options.externalData = graph.graph.externalData.map(({ path, url }) => ({ path, data: url }))
  }
  if (graph.options.freeDimensionOverrides) options.freeDimensionOverrides = graph.options.freeDimensionOverrides
  if (graph.options.enableGraphCapture !== undefined) options.enableGraphCapture = graph.options.enableGraphCapture
  if (graph.options.logSeverityLevel !== undefined) options.logSeverityLevel = graph.options.logSeverityLevel
  return options
}

function wasmOptions(graph: RuntimeWorkerRequest & { type: 'load' }): ort.InferenceSession.SessionOptions {
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: graph.options.graphOptimizationLevel ?? 'all',
    preferredOutputLocation: 'cpu',
  }
  if (graph.graph.externalData !== undefined) {
    options.externalData = graph.graph.externalData.map(({ path, url }) => ({ path, data: url }))
  }
  if (graph.options.freeDimensionOverrides) options.freeDimensionOverrides = graph.options.freeDimensionOverrides
  if (graph.options.logSeverityLevel !== undefined) options.logSeverityLevel = graph.options.logSeverityLevel
  return options
}

function providerOptions(
  request: RuntimeWorkerRequest & { type: 'load' },
  provider: ExecutionProvider,
): ort.InferenceSession.SessionOptions {
  return provider === 'webgpu' ? webGpuOptions(request) : wasmOptions(request)
}

async function load(request: RuntimeWorkerRequest & { type: 'load' }): Promise<GraphInfo> {
  configure({})
  const existing = sessions.get(request.sessionId)
  if (existing) return existing.graphInfo
  const providers = request.options.executionProviders ?? ['webgpu']
  if (providers.length === 0) throw new Error('No execution provider was configured.')
  let lastError: unknown
  for (const provider of providers) {
    status({
      stage: 'graph-loading',
      message: `Loading '${request.sessionId}' with ${provider}.`,
      sessionId: request.sessionId,
      provider,
    })
    const startedAt = performance.now()
    try {
      const session = await ort.InferenceSession.create(request.graph.url, providerOptions(request, provider))
      const graphInfo: GraphInfo = {
        sessionId: request.sessionId,
        executionProvider: provider,
        inputNames: Array.from(session.inputNames),
        outputNames: Array.from(session.outputNames),
        loadMs: performance.now() - startedAt,
      }
      sessions.set(request.sessionId, {
        session,
        provider,
        graphInfo,
        runTail: Promise.resolve(),
        reusableInputs: new Map(),
      })
      status({
        stage: 'graph-ready',
        message: `Graph '${request.sessionId}' is ready.`,
        sessionId: request.sessionId,
        provider,
        progress: 1,
      })
      return graphInfo
    } catch (error) {
      lastError = error
    }
  }
  throw new Error(`No execution provider could load '${request.sessionId}': ${String(lastError)}`)
}

function toOrtTensor(payload: TensorPayload): ort.Tensor {
  switch (payload.type) {
    case 'float32': return new ort.Tensor('float32', payload.data, payload.dims)
    case 'float16': return new ort.Tensor('float16', payload.data, payload.dims)
    case 'int32': return new ort.Tensor('int32', payload.data, payload.dims)
    case 'int64': return new ort.Tensor('int64', payload.data, payload.dims)
  }
}

async function toPayload(name: string, tensor: ort.Tensor): Promise<TensorPayload> {
  const data = await tensor.getData(true)
  switch (tensor.type) {
    case 'float32':
      if (data instanceof Float32Array) return createTensor('float32', new Float32Array(data), tensor.dims)
      break
    case 'float16':
      if (data instanceof Uint16Array) return createTensor('float16', new Uint16Array(data), tensor.dims)
      break
    case 'int32':
      if (data instanceof Int32Array) return createTensor('int32', new Int32Array(data), tensor.dims)
      break
    case 'int64':
      if (data instanceof BigInt64Array) return createTensor('int64', new BigInt64Array(data), tensor.dims)
      break
  }
  throw new TypeError(`Output '${name}' has unsupported dtype or storage '${tensor.type}'.`)
}

async function executeRun(
  request: RuntimeWorkerRequest & { type: 'run' },
  record: SessionRecord,
): Promise<GraphRunResult> {
  status({
    stage: 'inference-running',
    message: `Running '${request.sessionId}'.`,
    sessionId: request.sessionId,
    provider: record.provider,
  })
  const reusable = request.reusableInputsId === undefined
    ? undefined
    : record.reusableInputs.get(request.reusableInputsId)
  if (request.reusableInputsId !== undefined && reusable === undefined) {
    throw new Error(
      `Reusable inputs '${request.reusableInputsId}' are not retained for '${request.sessionId}'.`,
    )
  }
  const feeds: Record<string, ort.Tensor> = { ...reusable }
  const transientFeeds: ort.Tensor[] = []
  try {
    for (const [name, tensor] of Object.entries(request.inputs)) {
      if (Object.prototype.hasOwnProperty.call(feeds, name)) {
        throw new TypeError(`Input '${name}' cannot be both reusable and dynamic.`)
      }
      const feed = toOrtTensor(tensor)
      feeds[name] = feed
      transientFeeds.push(feed)
    }
  } catch (error) {
    for (const tensor of transientFeeds) tensor.dispose()
    throw error
  }
  const totalStart = performance.now()
  const inferenceStart = performance.now()
  let outputs: ort.InferenceSession.ReturnType
  try {
    const runOptions: ort.InferenceSession.RunOptions = request.tag ? { tag: request.tag } : {}
    outputs = request.outputs === undefined
      ? await record.session.run(feeds, runOptions)
      : await record.session.run(feeds, Array.from(request.outputs), runOptions)
  } finally {
    for (const tensor of transientFeeds) tensor.dispose()
  }
  const inferenceMs = performance.now() - inferenceStart
  const readbackStart = performance.now()
  const entries = Object.entries(outputs)
  status({
    stage: 'outputs-reading',
    message: `Reading ${entries.length} output tensor(s) from '${request.sessionId}'.`,
    sessionId: request.sessionId,
    provider: record.provider,
    progress: entries.length === 0 ? 1 : 0,
  })
  const payloadEntries = await Promise.all(entries.map(async ([name, tensor], index) => {
    try {
      const payload = await toPayload(name, tensor)
      status({
        stage: 'outputs-reading',
        message: `Read '${name}' from '${request.sessionId}'.`,
        sessionId: request.sessionId,
        provider: record.provider,
        progress: (index + 1) / entries.length,
      })
      return [name, payload] as const
    } finally {
      tensor.dispose()
    }
  }))
  const result: GraphRunResult = {
    outputs: Object.fromEntries(payloadEntries),
    timings: {
      inferenceMs,
      readbackMs: performance.now() - readbackStart,
      totalMs: performance.now() - totalStart,
    },
  }
  status({
    stage: 'inference-complete',
    message: `Inference for '${request.sessionId}' completed.`,
    sessionId: request.sessionId,
    provider: record.provider,
    progress: 1,
  })
  return result
}

async function retainInputs(
  request: RuntimeWorkerRequest & { type: 'retain-inputs' },
): Promise<string[]> {
  assertTensorMap(request.inputs)
  const record = sessions.get(request.sessionId)
  if (!record) throw new Error(`Graph '${request.sessionId}' is not loaded.`)
  await record.runTail
  const retained: Record<string, ort.Tensor> = {}
  try {
    for (const [name, tensor] of Object.entries(request.inputs)) retained[name] = toOrtTensor(tensor)
  } catch (error) {
    for (const tensor of Object.values(retained)) tensor.dispose()
    throw error
  }
  const previous = record.reusableInputs.get(request.reusableInputsId)
  record.reusableInputs.set(request.reusableInputsId, retained)
  if (previous) {
    for (const tensor of Object.values(previous)) tensor.dispose()
  }
  return Object.keys(retained)
}

async function run(request: RuntimeWorkerRequest & { type: 'run' }): Promise<GraphRunResult> {
  assertTensorMap(request.inputs)
  const record = sessions.get(request.sessionId)
  if (!record) throw new Error(`Graph '${request.sessionId}' is not loaded.`)
  status({ stage: 'inference-queued', message: `Queued '${request.sessionId}'.`, sessionId: request.sessionId })
  const result = record.runTail.then(() => executeRun(request, record))
  record.runTail = result.then(() => undefined, () => undefined)
  return result
}

async function disposeGraph(sessionId: string): Promise<boolean> {
  const record = sessions.get(sessionId)
  if (!record) return false
  sessions.delete(sessionId)
  status({ stage: 'graph-disposing', message: `Disposing '${sessionId}'.`, sessionId })
  await record.runTail
  for (const inputs of record.reusableInputs.values()) {
    for (const tensor of Object.values(inputs)) tensor.dispose()
  }
  record.reusableInputs.clear()
  await record.session.release()
  status({ stage: 'graph-disposed', message: `Disposed '${sessionId}'.`, sessionId, progress: 1 })
  return true
}

async function disposeAll(): Promise<string[]> {
  const ids = Array.from(sessions.keys())
  await Promise.all(ids.map(disposeGraph))
  status({ stage: 'runtime-disposed', message: 'TripoSplat runtime is disposed.', progress: 1 })
  return ids
}

async function dispatch(request: RuntimeWorkerRequest): Promise<void> {
  try {
    switch (request.type) {
      case 'configure':
        configure(request.configuration)
        success(request.requestId, { operation: 'configure', configured: true })
        return
      case 'load': {
        const graph = await load(request)
        success(request.requestId, { operation: 'load', graph })
        return
      }
      case 'retain-inputs': {
        const retainedInputNames = await retainInputs(request)
        success(request.requestId, { operation: 'retain-inputs', retainedInputNames })
        return
      }
      case 'run': {
        const result = await run(request)
        success(request.requestId, { operation: 'run', result }, tensorTransferables(result.outputs))
        return
      }
      case 'dispose-graph': {
        const disposed = await disposeGraph(request.sessionId)
        success(request.requestId, { operation: 'dispose-graph', disposed })
        return
      }
      case 'dispose': {
        const disposedSessionIds = await disposeAll()
        success(request.requestId, { operation: 'dispose', disposedSessionIds })
        return
      }
    }
  } catch (error) {
    failure(request.requestId, error)
  }
}

scope.onmessage = (event: MessageEvent<RuntimeWorkerRequest>) => {
  void dispatch(event.data)
}
