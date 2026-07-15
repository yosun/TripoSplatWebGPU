/// <reference lib="WebWorker" />

import * as ort from 'onnxruntime-web/webgpu'

import { assertModelManifest } from '../runtime/modelManifest'
import type { OnnxModelManifest } from '../runtime/modelManifest'
import type {
  OrtConfigureRuntimeResult,
  OrtExecutionProvider,
  OrtLoadSessionRequest,
  OrtLoadSessionResult,
  OrtRunSessionRequest,
  OrtRunSessionResult,
  OrtRuntimeConfiguration,
  OrtSessionLoadOptions,
  OrtValueMetadata,
  OrtWorkerOperation,
  OrtWorkerReply,
  OrtWorkerRequest,
  OrtWorkerResultMap,
  OrtWorkerStage,
  OrtWorkerStatus,
  SerializedWorkerError,
} from '../runtime/OrtWorkerClient'
import type { TensorPayload, TensorPayloadMap } from '../runtime/tensors'
import {
  assertTensorPayloadMap,
  createTensorPayload,
  tensorPayloadTransferables,
} from '../runtime/tensors'

const workerScope = self as DedicatedWorkerGlobalScope

interface LoadedSession {
  session: ort.InferenceSession
  executionProvider: OrtExecutionProvider
  metadata: Omit<OrtLoadSessionResult, 'loadMs'>
  loadMs: number
}

interface SessionRecord {
  fingerprint: string
  loading: Promise<LoadedSession>
  runTail: Promise<void>
  disposed: boolean
}

const sessions = new Map<string, SessionRecord>()
let runtimeConfiguration: OrtConfigureRuntimeResult | undefined

function serializeError(error: unknown): SerializedWorkerError {
  if (error instanceof Error) {
    return { name: error.name, message: error.message, stack: error.stack }
  }
  return { name: 'Error', message: String(error) }
}

function postMessageSafe(message: OrtWorkerReply | OrtWorkerStatus, transfer: Transferable[] = []): void {
  workerScope.postMessage(message, transfer)
}

function postStatus(
  stage: OrtWorkerStage,
  message: string,
  requestId?: string,
  sessionId?: string,
  executionProvider?: OrtExecutionProvider,
  progress?: number,
): void {
  postMessageSafe({
    type: 'status',
    stage,
    message,
    timestampMs: Date.now(),
    requestId,
    sessionId,
    executionProvider,
    progress,
  })
}

function postSuccess<Operation extends OrtWorkerOperation>(
  operation: Operation,
  requestId: string,
  result: OrtWorkerResultMap[Operation],
  transfer: Transferable[] = [],
): void {
  const reply = {
    type: 'reply',
    operation,
    requestId,
    ok: true,
    result,
  } as OrtWorkerReply
  postMessageSafe(reply, transfer)
}

function postError(operation: OrtWorkerOperation, requestId: string, error: unknown): void {
  const reply = {
    type: 'reply',
    operation,
    requestId,
    ok: false,
    error: serializeError(error),
  } as OrtWorkerReply
  postMessageSafe(reply)
}

function assertSessionId(sessionId: unknown): asserts sessionId is string {
  if (typeof sessionId !== 'string' || sessionId.trim().length === 0 || sessionId.includes('\0')) {
    throw new TypeError('sessionId must be a non-empty string without null characters.')
  }
}

function defaultRuntimeConfiguration(): OrtConfigureRuntimeResult {
  const baseUrl = new URL(`${import.meta.env.BASE_URL}ort/`, workerScope.location.origin).href
  return {
    wasmThreads: workerScope.crossOriginIsolated
      ? Math.max(1, Math.min(4, workerScope.navigator.hardwareConcurrency || 2))
      : 1,
    wasmSimd: true,
    wasmPaths: {
      mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs', baseUrl).href,
      wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', baseUrl).href,
    },
  }
}

function normalizeRuntimeConfiguration(configuration: OrtRuntimeConfiguration): OrtConfigureRuntimeResult {
  const defaults = defaultRuntimeConfiguration()
  const wasmThreads = configuration.wasmThreads ?? defaults.wasmThreads
  if (!Number.isInteger(wasmThreads) || wasmThreads < 1) {
    throw new RangeError('wasmThreads must be a positive integer.')
  }

  const wasmSimd = configuration.wasmSimd ?? defaults.wasmSimd
  if (
    typeof wasmSimd !== 'boolean'
    && wasmSimd !== 'fixed'
    && wasmSimd !== 'relaxed'
  ) {
    throw new TypeError("wasmSimd must be boolean, 'fixed', or 'relaxed'.")
  }

  let wasmPaths: OrtConfigureRuntimeResult['wasmPaths']
  if (configuration.wasmPaths === undefined) {
    wasmPaths = defaults.wasmPaths
  } else if (typeof configuration.wasmPaths === 'string') {
    if (configuration.wasmPaths.trim().length === 0) {
      throw new TypeError('wasmPaths prefix must be non-empty.')
    }
    wasmPaths = new URL(configuration.wasmPaths, workerScope.location.href).href
  } else {
    if (configuration.wasmPaths.mjs === undefined && configuration.wasmPaths.wasm === undefined) {
      throw new TypeError('wasmPaths must include at least one of mjs or wasm.')
    }
    wasmPaths = {
      mjs: configuration.wasmPaths.mjs === undefined
        ? undefined
        : new URL(configuration.wasmPaths.mjs, workerScope.location.href).href,
      wasm: configuration.wasmPaths.wasm === undefined
        ? undefined
        : new URL(configuration.wasmPaths.wasm, workerScope.location.href).href,
    }
  }

  return { wasmThreads, wasmSimd, wasmPaths }
}

function configureRuntime(
  configuration: OrtRuntimeConfiguration,
  requestId?: string,
): OrtConfigureRuntimeResult {
  const normalized = normalizeRuntimeConfiguration(configuration)
  if (runtimeConfiguration) {
    if (JSON.stringify(runtimeConfiguration) !== JSON.stringify(normalized)) {
      throw new Error('ONNX Runtime is already configured; start a new worker to use different WASM settings.')
    }
    return runtimeConfiguration
  }

  postStatus('runtime-configuring', 'Configuring ONNX Runtime.', requestId)
  ort.env.wasm.numThreads = normalized.wasmThreads
  ort.env.wasm.simd = normalized.wasmSimd
  ort.env.wasm.wasmPaths = normalized.wasmPaths
  runtimeConfiguration = normalized
  postStatus('runtime-ready', 'ONNX Runtime is configured.', requestId)
  return normalized
}

function ensureRuntimeConfigured(requestId: string): OrtConfigureRuntimeResult {
  return runtimeConfiguration ?? configureRuntime({}, requestId)
}

function toMetadata(metadata: readonly ort.InferenceSession.ValueMetadata[]): OrtValueMetadata[] {
  return metadata.map((value) => value.isTensor
    ? {
        name: value.name,
        isTensor: true,
        type: value.type,
        shape: Array.from(value.shape),
      }
    : { name: value.name, isTensor: false })
}

function sessionFingerprint(request: OrtLoadSessionRequest): string {
  return JSON.stringify({ manifest: request.manifest, options: request.options ?? {} })
}

function createSessionOptions(
  manifest: OnnxModelManifest,
  options: OrtSessionLoadOptions | undefined,
  provider: OrtExecutionProvider,
): ort.InferenceSession.SessionOptions {
  const common: ort.InferenceSession.SessionOptions = {
    graphOptimizationLevel: options?.graphOptimizationLevel ?? 'all',
    preferredOutputLocation: 'cpu',
    externalData: manifest.externalData?.map(({ path, url }) => ({ path, data: url })),
  }

  if (options?.freeDimensionOverrides) {
    common.freeDimensionOverrides = options.freeDimensionOverrides
  }
  if (options?.logSeverityLevel !== undefined) {
    common.logSeverityLevel = options.logSeverityLevel
  }

  if (provider === 'wasm') {
    common.executionProviders = ['wasm']
    return common
  }

  const webgpu: ort.InferenceSession.WebGpuExecutionProviderOption = {
    name: 'webgpu',
    preferredLayout: options?.webgpu?.preferredLayout,
    forceCpuNodeNames: options?.webgpu?.forceCpuNodeNames,
    validationMode: options?.webgpu?.validationMode,
  }
  common.executionProviders = [webgpu]
  if (options?.enableGraphCapture !== undefined) {
    common.enableGraphCapture = options.enableGraphCapture
  }
  return common
}

async function loadOrtSession(request: OrtLoadSessionRequest, requestId: string): Promise<LoadedSession> {
  ensureRuntimeConfigured(requestId)
  const { sessionId, manifest, options } = request
  const startedAt = performance.now()
  postStatus('session-loading', `Loading ONNX session '${sessionId}'.`, requestId, sessionId, 'webgpu')

  const heartbeatStartedAt = performance.now()
  const heartbeat = setInterval(() => {
    const seconds = Math.floor((performance.now() - heartbeatStartedAt) / 1000)
    postStatus(
      'session-loading',
      `Loading ONNX session '${sessionId}' (${seconds}s elapsed).`,
      requestId,
      sessionId,
      'webgpu',
    )
  }, 1000)

  let session: ort.InferenceSession
  let executionProvider: OrtExecutionProvider = 'webgpu'
  try {
    try {
      session = await ort.InferenceSession.create(
        manifest.graphUrl,
        createSessionOptions(manifest, options, 'webgpu'),
      )
    } catch (webgpuError) {
      if (!options?.allowWasmFallback) {
        throw webgpuError
      }
      executionProvider = 'wasm'
      postStatus(
        'session-fallback',
        `WebGPU could not load '${sessionId}'; retrying with WASM.`,
        requestId,
        sessionId,
        'wasm',
      )
      try {
        session = await ort.InferenceSession.create(
          manifest.graphUrl,
          createSessionOptions(manifest, options, 'wasm'),
        )
      } catch (wasmError) {
        throw new Error(
          `Could not create '${sessionId}' with WebGPU (${String(webgpuError)}) or WASM (${String(wasmError)}).`,
        )
      }
    }
  } finally {
    clearInterval(heartbeat)
  }

  const loadMs = performance.now() - startedAt
  const metadata: Omit<OrtLoadSessionResult, 'loadMs'> = {
    sessionId,
    executionProvider,
    inputNames: Array.from(session.inputNames),
    outputNames: Array.from(session.outputNames),
    inputMetadata: toMetadata(session.inputMetadata),
    outputMetadata: toMetadata(session.outputMetadata),
  }
  postStatus(
    'session-ready',
    `ONNX session '${sessionId}' is ready.`,
    requestId,
    sessionId,
    executionProvider,
    1,
  )
  return { session, executionProvider, metadata, loadMs }
}

async function loadSession(request: OrtLoadSessionRequest, requestId: string): Promise<OrtLoadSessionResult> {
  assertSessionId(request.sessionId)
  assertModelManifest(request.manifest)
  const fingerprint = sessionFingerprint(request)
  const existing = sessions.get(request.sessionId)
  if (existing) {
    if (existing.fingerprint !== fingerprint) {
      throw new Error(`Session '${request.sessionId}' is already loaded with a different manifest or options.`)
    }
    const loaded = await existing.loading
    postStatus(
      'session-ready',
      `ONNX session '${request.sessionId}' was already loaded.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
      1,
    )
    return { ...loaded.metadata, loadMs: loaded.loadMs }
  }

  const record: SessionRecord = {
    fingerprint,
    loading: Promise.resolve(undefined as never),
    runTail: Promise.resolve(),
    disposed: false,
  }
  record.loading = loadOrtSession(request, requestId).catch((error: unknown) => {
    if (sessions.get(request.sessionId) === record) {
      sessions.delete(request.sessionId)
    }
    throw error
  })
  sessions.set(request.sessionId, record)

  const loaded = await record.loading
  return { ...loaded.metadata, loadMs: loaded.loadMs }
}

function toOrtTensor(payload: TensorPayload): ort.Tensor {
  switch (payload.type) {
    case 'float32':
      return new ort.Tensor('float32', payload.data, payload.dims)
    case 'float16':
      return new ort.Tensor('float16', payload.data, payload.dims)
    case 'int32':
      return new ort.Tensor('int32', payload.data, payload.dims)
    case 'int64':
      return new ort.Tensor('int64', payload.data, payload.dims)
  }
}

async function outputTensorPayload(name: string, tensor: ort.Tensor): Promise<TensorPayload> {
  const data = await tensor.getData(true)
  switch (tensor.type) {
    case 'float32': {
      if (!(data instanceof Float32Array)) {
        throw new TypeError(`Output '${name}' declared float32 but returned a different storage type.`)
      }
      const copy = new Float32Array(data.length)
      copy.set(data)
      return createTensorPayload('float32', copy, tensor.dims)
    }
    case 'float16': {
      if (!(data instanceof Uint16Array)) {
        throw new TypeError(`Output '${name}' declared float16 but returned a different storage type.`)
      }
      const copy = new Uint16Array(data.length)
      copy.set(data)
      return createTensorPayload('float16', copy, tensor.dims)
    }
    case 'int32': {
      if (!(data instanceof Int32Array)) {
        throw new TypeError(`Output '${name}' declared int32 but returned a different storage type.`)
      }
      const copy = new Int32Array(data.length)
      copy.set(data)
      return createTensorPayload('int32', copy, tensor.dims)
    }
    case 'int64': {
      if (!(data instanceof BigInt64Array)) {
        throw new TypeError(`Output '${name}' declared int64 but returned a different storage type.`)
      }
      const copy = new BigInt64Array(data.length)
      copy.set(data)
      return createTensorPayload('int64', copy, tensor.dims)
    }
    default:
      throw new TypeError(
        `Output '${name}' uses unsupported dtype '${tensor.type}'. Supported worker payloads are float32, float16, int32, and int64.`,
      )
  }
}

function enqueueSessionRun<T>(record: SessionRecord, task: () => Promise<T>): Promise<T> {
  const result = record.runTail.then(task)
  record.runTail = result.then(() => undefined, () => undefined)
  return result
}

async function runSession(request: OrtRunSessionRequest, requestId: string): Promise<OrtRunSessionResult> {
  assertSessionId(request.sessionId)
  assertTensorPayloadMap(request.inputs, 'request.inputs')
  const record = sessions.get(request.sessionId)
  if (!record || record.disposed) {
    throw new Error(`ONNX session '${request.sessionId}' is not loaded.`)
  }

  postStatus('inference-queued', `Queued inference for '${request.sessionId}'.`, requestId, request.sessionId)
  return enqueueSessionRun(record, async () => {
    if (record.disposed) {
      throw new Error(`ONNX session '${request.sessionId}' was disposed before inference started.`)
    }
    const loaded = await record.loading
    const requestedOutputs = request.outputs === undefined ? undefined : Array.from(request.outputs)
    if (requestedOutputs) {
      const available = new Set(loaded.session.outputNames)
      const seen = new Set<string>()
      for (const output of requestedOutputs) {
        if (typeof output !== 'string' || output.trim().length === 0 || seen.has(output)) {
          throw new TypeError('Requested output names must be unique, non-empty strings.')
        }
        if (!available.has(output)) {
          throw new Error(
            `Session '${request.sessionId}' has no output '${output}'. Available outputs: ${loaded.session.outputNames.join(', ')}.`,
          )
        }
        seen.add(output)
      }
    }

    const feeds: Record<string, ort.Tensor> = {}
    for (const [name, payload] of Object.entries(request.inputs)) {
      feeds[name] = toOrtTensor(payload)
    }

    const totalStartedAt = performance.now()
    const inferenceStartedAt = performance.now()
    postStatus(
      'inference-running',
      `Running inference for '${request.sessionId}'.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
    )

    let ortOutputs: ort.InferenceSession.ReturnType
    try {
      const runOptions: ort.InferenceSession.RunOptions = request.tag ? { tag: request.tag } : {}
      ortOutputs = requestedOutputs === undefined
        ? await loaded.session.run(feeds, runOptions)
        : await loaded.session.run(feeds, requestedOutputs, runOptions)
    } finally {
      for (const tensor of Object.values(feeds)) {
        tensor.dispose()
      }
    }
    const inferenceMs = performance.now() - inferenceStartedAt

    const readbackStartedAt = performance.now()
    const entries = Object.entries(ortOutputs)
    postStatus(
      'outputs-reading',
      `Reading ${entries.length} output tensor${entries.length === 1 ? '' : 's'} from '${request.sessionId}'.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
      entries.length === 0 ? 1 : 0,
    )

    const outputEntries = await Promise.all(entries.map(async ([name, tensor], index) => {
      try {
        const payload = await outputTensorPayload(name, tensor)
        postStatus(
          'outputs-reading',
          `Read output '${name}' from '${request.sessionId}'.`,
          requestId,
          request.sessionId,
          loaded.executionProvider,
          entries.length === 0 ? 1 : (index + 1) / entries.length,
        )
        return [name, payload] as const
      } finally {
        tensor.dispose()
      }
    }))
    const outputs = Object.fromEntries(outputEntries) as TensorPayloadMap
    const readbackMs = performance.now() - readbackStartedAt
    const totalMs = performance.now() - totalStartedAt
    postStatus(
      'inference-complete',
      `Inference for '${request.sessionId}' completed.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
      1,
    )
    return {
      sessionId: request.sessionId,
      outputs,
      timings: { inferenceMs, readbackMs, totalMs },
    }
  })
}

async function disposeSession(sessionId: string, requestId?: string): Promise<boolean> {
  assertSessionId(sessionId)
  const record = sessions.get(sessionId)
  if (!record) {
    return false
  }

  record.disposed = true
  sessions.delete(sessionId)
  postStatus('session-disposing', `Disposing ONNX session '${sessionId}'.`, requestId, sessionId)
  await record.runTail
  let loaded: LoadedSession | undefined
  try {
    loaded = await record.loading
  } catch {
    // A failed load has no session resources left to release. Preserve disposal semantics.
  }
  if (loaded) {
    await loaded.session.release()
  }
  postStatus('session-disposed', `Disposed ONNX session '${sessionId}'.`, requestId, sessionId, undefined, 1)
  return true
}

async function disposeAll(requestId: string): Promise<string[]> {
  const sessionIds = Array.from(sessions.keys())
  postStatus('worker-disposing', `Disposing ${sessionIds.length} ONNX session(s).`, requestId)
  await Promise.all(sessionIds.map((sessionId) => disposeSession(sessionId, requestId)))
  postStatus('worker-disposed', 'All ONNX sessions are disposed.', requestId, undefined, undefined, 1)
  return sessionIds
}

async function dispatch(request: OrtWorkerRequest): Promise<void> {
  try {
    switch (request.type) {
      case 'configure-runtime': {
        const result = configureRuntime(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result)
        return
      }
      case 'load-session': {
        const result = await loadSession(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result)
        return
      }
      case 'run-session': {
        const result = await runSession(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result, tensorPayloadTransferables(result.outputs))
        return
      }
      case 'dispose-session': {
        const disposed = await disposeSession(request.payload.sessionId, request.requestId)
        postSuccess(request.type, request.requestId, { sessionId: request.payload.sessionId, disposed })
        return
      }
      case 'dispose-all': {
        const disposedSessionIds = await disposeAll(request.requestId)
        postSuccess(request.type, request.requestId, { disposedSessionIds })
        return
      }
    }
  } catch (error) {
    postError(request.type, request.requestId, error)
  }
}

workerScope.onmessage = (event: MessageEvent<OrtWorkerRequest>) => {
  void dispatch(event.data)
}
