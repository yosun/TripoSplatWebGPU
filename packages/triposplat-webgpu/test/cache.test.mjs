import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import test from 'node:test'

import {
  CancelledError,
  ModelDownloadError,
  ModelIntegrityError,
  TripoSplatWebGPU,
} from '../dist/index.js'
import {
  clearModelCache,
  createRuntime,
  getModelCacheStatus,
  MemoryModelArtifactStorage,
  ModelArtifactManager,
  Sha256,
  sha256Hex,
  withVerifiedModelArtifacts,
} from '../dist/low-level.js'
import { prepareReusableGraphInputs } from '../dist/runtime.js'

function bytes(value) {
  return new TextEncoder().encode(value)
}

function digest(value) {
  return createHash('sha256').update(value).digest('hex')
}

function graphDescriptor(graphBytes, shardBytes, suffix = '') {
  return {
    url: `https://cdn.example.test/model.onnx?signature=${suffix}`,
    byteLength: graphBytes.byteLength,
    integrity: { algorithm: 'sha256', digest: digest(graphBytes) },
    externalData: [{
      path: 'model.onnx.data',
      url: `https://cdn.example.test/model.onnx.data?signature=${suffix}`,
      byteLength: shardBytes.byteLength,
      integrity: { algorithm: 'sha256', digest: digest(shardBytes) },
    }],
  }
}

test('incremental SHA-256 matches the standard digest', () => {
  assert.equal(
    sha256Hex(bytes('abc')),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  )
  for (const length of [0, 55, 56, 63, 64, 65, 1000]) {
    const value = new Uint8Array(length)
    for (let index = 0; index < length; index += 1) value[index] = index % 251
    const hash = new Sha256()
    for (let offset = 0; offset < value.length; offset += 17) {
      hash.update(value.subarray(offset, Math.min(value.length, offset + 17)))
    }
    assert.equal(hash.hex(), digest(value), `SHA-256 mismatch at ${length} bytes`)
  }
})

test('graph and external-data bytes are verified, cached, and reused without CDN access', async () => {
  const graphBytes = bytes('small onnx graph')
  const shardBytes = bytes('external tensor bytes')
  const graph = graphDescriptor(graphBytes, shardBytes, 'first')
  const storage = new MemoryModelArtifactStorage()
  const requests = []
  const progress = []
  const manager = new ModelArtifactManager({
    backend: 'cache-api',
    namespace: 'triposplat/1/revision/fp16',
    storage,
    requestInit: { headers: { authorization: 'Bearer local-test' } },
    fetch: async (url, init) => {
      requests.push(String(url))
      assert.equal(new Headers(init?.headers).get('authorization'), 'Bearer local-test')
      return new Response(String(url).includes('.data') ? shardBytes : graphBytes)
    },
    onProgress: (event) => progress.push(event),
  })

  const prepared = await manager.prepareGraph('dit', graph)
  try {
    assert.equal(prepared.graph.externalData[0].path, 'model.onnx.data')
    assert.deepEqual(new Uint8Array(await (await fetch(prepared.graph.url)).arrayBuffer()), graphBytes)
    assert.deepEqual(
      new Uint8Array(await (await fetch(prepared.graph.externalData[0].url)).arrayBuffer()),
      shardBytes,
    )
  } finally {
    prepared.release()
  }
  assert.equal(requests.length, 2)
  assert.ok(progress.every(({ source }) => source === 'network'))

  const cacheProgress = []
  const cachedManager = new ModelArtifactManager({
    backend: 'cache-api',
    namespace: 'triposplat/1/revision/fp16',
    storage,
    fetch: async () => { throw new Error('CDN should not be called for a verified cache hit.') },
    onProgress: (event) => cacheProgress.push(event),
  })
  const cached = await cachedManager.prepareGraph('dit', graphDescriptor(graphBytes, shardBytes, 'refreshed'))
  cached.release()
  assert.equal(cacheProgress.length, 2)
  assert.ok(cacheProgress.every(({ source }) => source === 'cache'))
  assert.equal((await cachedManager.status()).length, 2)
})

test('persistent prefetch memoizes verified blobs for prepareGraph within one manager lifecycle', async () => {
  const artifact = bytes('prefetched persistent graph')
  const graph = {
    url: 'https://cdn.example.test/prefetched.onnx?signature=initial',
    byteLength: artifact.byteLength,
    integrity: { algorithm: 'sha256', digest: digest(artifact) },
  }
  const storage = new MemoryModelArtifactStorage()
  const seed = new ModelArtifactManager({
    backend: 'cache-api',
    namespace: 'triposplat/1/prefetch/fp32',
    storage,
    fetch: async () => new Response(artifact),
  })
  ;(await seed.prepareGraph('dino', graph)).release()

  const [entry] = await storage.entries()
  const record = await storage.get(entry.cacheKey)
  let hashReads = 0
  record.blob = new class extends Blob {
    stream() {
      hashReads += 1
      return super.stream()
    }
  }([record.blob])
  const originalGet = storage.get.bind(storage)
  let storageReads = 0
  storage.get = async (cacheKey) => {
    storageReads += 1
    return originalGet(cacheKey)
  }

  const manager = new ModelArtifactManager({
    backend: 'cache-api',
    namespace: 'triposplat/1/prefetch/fp32',
    storage,
    fetch: async () => { throw new Error('verified persistent bytes should not be downloaded again') },
  })
  await manager.prefetchManifest({
    name: 'triposplat-webgpu',
    version: '1',
    modelRevision: 'prefetch',
    precision: 'fp32',
    graphs: { dino: graph },
  })
  const prepared = await manager.prepareGraph('dino', {
    ...graph,
    url: 'https://cdn.example.test/prefetched.onnx?signature=refreshed',
  })
  prepared.release()

  assert.equal(storageReads, 1)
  assert.equal(hashReads, 1)
})

test("cache 'none' never retains verified artifacts between prepareGraph calls", async () => {
  const artifact = bytes('non-persistent graph')
  const storage = new MemoryModelArtifactStorage()
  let downloads = 0
  const manager = new ModelArtifactManager({
    backend: 'none',
    namespace: 'triposplat/1/no-retention/fp32',
    storage,
    fetch: async () => {
      downloads += 1
      return new Response(artifact)
    },
  })
  const graph = {
    url: 'https://cdn.example.test/non-persistent.onnx',
    byteLength: artifact.byteLength,
    integrity: { algorithm: 'sha256', digest: digest(artifact) },
  }

  for (let run = 0; run < 2; run += 1) {
    const prepared = await manager.prepareGraph('dino', graph)
    prepared.release()
    await new Promise((resolve) => setTimeout(resolve, 0))
    assert.deepEqual(await storage.entries(), [])
  }
  assert.equal(downloads, 2)
})

test('artifact fetch implementations are invoked without a ModelArtifactManager receiver', async () => {
  const artifact = bytes('unbound fetch graph')
  let receiver
  const manager = new ModelArtifactManager({
    backend: 'none',
    namespace: 'triposplat/1/unbound-fetch/fp32',
    fetch: async function () {
      receiver = this
      return new Response(artifact)
    },
  })
  const prepared = await manager.prepareGraph('dino', {
    url: 'https://cdn.example.test/unbound-fetch.onnx',
    byteLength: artifact.byteLength,
    integrity: { algorithm: 'sha256', digest: digest(artifact) },
  })
  prepared.release()
  assert.equal(receiver, undefined)
})

test('integrity or byte-length mismatch is rejected before an artifact is committed', async () => {
  const storage = new MemoryModelArtifactStorage()
  const manager = new ModelArtifactManager({
    backend: 'cache-api',
    namespace: 'triposplat/1/bad/fp16',
    storage,
    fetch: async () => new Response(bytes('corrupted')),
  })
  const graph = {
    url: 'https://cdn.example.test/bad.onnx',
    byteLength: 999,
    integrity: { algorithm: 'sha256', digest: '0'.repeat(64) },
  }
  await assert.rejects(
    manager.prepareGraph('dino', graph),
    (error) => error instanceof ModelIntegrityError
      && error.code === 'MODEL_INTEGRITY_FAILED'
      && error.diagnostics.expectedByteLength === 999,
  )
  assert.deepEqual(await storage.entries(), [])
})

test('verified runtime wrapper gives ORT local Blob URLs and revokes them after session creation', async () => {
  const graphBytes = bytes('worker graph bytes')
  const shardBytes = bytes('worker shard bytes')
  const graph = graphDescriptor(graphBytes, shardBytes, 'runtime')
  let capturedGraphUrl
  let capturedShardUrl
  const controller = new AbortController()
  const delegate = {
    disposed: false,
    async loadGraph(sessionId, prepared, options) {
      capturedGraphUrl = prepared.url
      capturedShardUrl = prepared.externalData[0].url
      assert.deepEqual(new Uint8Array(await (await fetch(prepared.url)).arrayBuffer()), graphBytes)
      assert.deepEqual(
        new Uint8Array(await (await fetch(prepared.externalData[0].url)).arrayBuffer()),
        shardBytes,
      )
      assert.equal(options.signal, controller.signal)
      return {
        sessionId,
        executionProvider: 'webgpu',
        inputNames: [],
        outputNames: [],
        loadMs: 1,
      }
    },
    async runGraph() { throw new Error('not used') },
    async disposeGraph() { return true },
    async dispose() { this.disposed = true },
  }
  const manager = new ModelArtifactManager({
    backend: 'none',
    namespace: 'triposplat/1/runtime/fp16',
    fetch: async (url) => new Response(String(url).includes('.data') ? shardBytes : graphBytes),
  })
  const runtime = withVerifiedModelArtifacts(delegate, manager, { 'triposplat/dit': 'dit' })
  await runtime.loadGraph('triposplat/dit', graph, { signal: controller.signal })
  await assert.rejects(fetch(capturedGraphUrl))
  await assert.rejects(fetch(capturedShardUrl))
  await runtime.dispose()
  assert.equal(delegate.disposed, true)
})

test('verified runtime wrapper preserves built-in reusable-input worker capability', async () => {
  const messages = []
  const worker = {
    onmessage: null,
    onerror: null,
    onmessageerror: null,
    postMessage(request, transfer = []) {
      const delivered = structuredClone(request, { transfer })
      messages.push(delivered)
      queueMicrotask(() => {
        const result = delivered.type === 'configure'
          ? { operation: 'configure', configured: true }
          : delivered.type === 'retain-inputs'
            ? { operation: 'retain-inputs', retainedInputNames: Object.keys(delivered.inputs) }
            : { operation: 'dispose', disposedSessionIds: [] }
        this.onmessage?.({
          data: { type: 'reply', requestId: delivered.requestId, ok: true, result },
        })
      })
    },
    terminate() {},
  }
  const base = createRuntime({ workerFactory: () => worker })
  const manager = new ModelArtifactManager({
    backend: 'none',
    namespace: 'triposplat/1/reusable-wrapper/fp32',
  })
  const runtime = withVerifiedModelArtifacts(base, manager, {})
  const feature = Float32Array.of(1, 2, 3, 4)
  await prepareReusableGraphInputs(
    runtime,
    'triposplat/dit',
    'condition',
    { feature: { type: 'float32', data: feature, dims: [1, 4] } },
  )
  assert.equal(feature.byteLength, 0)
  assert.equal(messages.filter(({ type }) => type === 'retain-inputs').length, 1)
  await runtime.dispose()
})

test('corrupt cached bytes are evicted and replaced from the authoritative object', async () => {
  const original = bytes('verified bytes')
  const storage = new MemoryModelArtifactStorage()
  const graph = {
    url: 'https://cdn.example.test/verified.onnx',
    byteLength: original.byteLength,
    integrity: { algorithm: 'sha256', digest: digest(original) },
  }
  const first = new ModelArtifactManager({
    backend: 'opfs',
    namespace: 'triposplat/1/revision/fp16',
    storage,
    fetch: async () => new Response(original),
  })
  ;(await first.prepareGraph('vae', graph)).release()
  const [entry] = await storage.entries()
  const record = await storage.get(entry.cacheKey)
  record.blob = new Blob([bytes('tampered local bytes')])

  let downloads = 0
  const second = new ModelArtifactManager({
    backend: 'opfs',
    namespace: 'triposplat/1/revision/fp16',
    storage,
    fetch: async () => {
      downloads += 1
      return new Response(original)
    },
  })
  ;(await second.prepareGraph('vae', graph)).release()
  assert.equal(downloads, 1)
  assert.equal((await storage.entries()).length, 1)
})

test('cache namespaces isolate revisions and targeted clear preserves other revisions', async () => {
  const data = bytes('same immutable artifact')
  const storage = new MemoryModelArtifactStorage()
  const graph = {
    url: 'https://cdn.example.test/model.onnx',
    byteLength: data.byteLength,
    integrity: { algorithm: 'sha256', digest: digest(data) },
  }
  for (const namespace of ['triposplat/1/rev-a/fp16', 'triposplat/2/rev-b/fp16']) {
    const manager = new ModelArtifactManager({
      backend: 'cache-api',
      namespace,
      storage,
      fetch: async () => new Response(data),
    })
    ;(await manager.prepareGraph('octree', graph)).release()
  }
  assert.equal((await storage.entries()).length, 2)
  await storage.clear('triposplat/1/rev-a/fp16')
  const remaining = await storage.entries()
  assert.equal(remaining.length, 1)
  assert.equal(remaining[0].namespace, 'triposplat/2/rev-b/fp16')
})

test('aborted artifact streams leave no cache entry', async () => {
  const storage = new MemoryModelArtifactStorage()
  const controller = new AbortController()
  const manager = new ModelArtifactManager({
    backend: 'cache-api',
    namespace: 'triposplat/1/cancelled/fp16',
    storage,
    fetch: async () => new Response(new ReadableStream({
      start(stream) {
        stream.enqueue(bytes('first chunk'))
        stream.enqueue(bytes('second chunk'))
        stream.close()
      },
    })),
    onProgress() { controller.abort('test cancellation') },
  })
  await assert.rejects(
    manager.prepareGraph('gaussianDecoder', { url: 'https://cdn.example.test/large.onnx' }, controller.signal),
    (error) => error instanceof CancelledError,
  )
  assert.deepEqual(await storage.entries(), [])
})

test('Cache API status and clear report verified persistent entries', async () => {
  const originalCaches = globalThis.caches
  const stores = new Map()
  const cache = {
    async match(request) { return stores.get(request.url)?.clone() },
    async put(request, response) {
      const body = await response.blob()
      stores.set(request.url, new Response(body, { headers: response.headers }))
    },
    async delete(request) { return stores.delete(request.url) },
    async keys() { return Array.from(stores.keys(), (url) => new Request(url)) },
  }
  globalThis.caches = {
    async open() { return cache },
    async delete() { stores.clear(); return true },
  }
  try {
    const data = bytes('cache api bytes')
    const manager = new ModelArtifactManager({
      backend: 'cache-api',
      namespace: 'triposplat/1/cache-api/fp16',
      fetch: async () => new Response(data),
    })
    ;(await manager.prepareGraph('dino', {
      url: 'https://cdn.example.test/cache-api.onnx',
      byteLength: data.byteLength,
      integrity: { algorithm: 'sha256', digest: digest(data) },
    })).release()
    const status = await getModelCacheStatus({ backend: 'cache-api' })
    assert.equal(status.entryCount, 1)
    assert.equal(status.totalBytes, data.byteLength)
    assert.equal(status.backends[0].available, true)
    await clearModelCache({ backend: 'cache-api' })
    assert.equal((await getModelCacheStatus({ backend: 'cache-api' })).entryCount, 0)
  } finally {
    if (originalCaches === undefined) delete globalThis.caches
    else globalThis.caches = originalCaches
  }
})

test('high-level load prefetches persistent artifacts and reuses them across model instances', async () => {
  const originalCaches = globalThis.caches
  const stores = new Map()
  const cache = {
    async match(request) { return stores.get(request.url)?.clone() },
    async put(request, response) {
      const body = await response.blob()
      stores.set(request.url, new Response(body, { headers: response.headers }))
    },
    async delete(request) { return stores.delete(request.url) },
    async keys() { return Array.from(stores.keys(), (url) => new Request(url)) },
  }
  globalThis.caches = {
    async open() { return cache },
    async delete() { stores.clear(); return true },
  }
  const artifact = bytes('high-level graph')
  const workerFactory = () => ({
    onmessage: null,
    onerror: null,
    onmessageerror: null,
    postMessage(request) {
      queueMicrotask(() => {
        const result = request.type === 'configure'
          ? { operation: 'configure', configured: true }
          : { operation: 'dispose', disposedSessionIds: [] }
        this.onmessage?.({ data: { type: 'reply', requestId: request.requestId, ok: true, result } })
      })
    },
    terminate() {},
  })
  let artifactDownloads = 0
  let manifestRevision = 0
  const customFetch = async (url, init) => {
    if (String(url).endsWith('manifest.json')) {
      manifestRevision += 1
      assert.equal(new Headers(init?.headers).get('x-manifest'), 'manifest-header')
      return Response.json({
        name: 'triposplat-webgpu',
        version: '1.0.0',
        modelRevision: 'immutable-revision',
        precision: 'fp16',
        graphs: {
          dino: {
            url: `dino.onnx?signature=${manifestRevision}`,
            byteLength: artifact.byteLength,
            integrity: { algorithm: 'sha256', digest: digest(artifact) },
          },
        },
      })
    }
    artifactDownloads += 1
    assert.equal(new Headers(init?.headers).get('authorization'), 'Bearer artifact-header')
    return new Response(artifact)
  }
  try {
    for (let run = 0; run < 2; run += 1) {
      const progress = []
      const model = new TripoSplatWebGPU({
        modelBaseUrl: 'https://cdn.example.test/model/',
        executionProviders: ['wasm'],
        cache: 'cache-api',
        fetch: customFetch,
        manifestRequestInit: { headers: { 'x-manifest': 'manifest-header' } },
        artifactRequestInit: { headers: { authorization: 'Bearer artifact-header' } },
        workerFactory,
      })
      await model.load({ onProgress: (event) => progress.push(event) })
      assert.ok(progress.some((event) => event.stage === 'graphs'))
      await model.dispose()
    }
    assert.equal(artifactDownloads, 1)
  } finally {
    if (originalCaches === undefined) delete globalThis.caches
    else globalThis.caches = originalCaches
  }
})

test("cache 'none' streams through temporary OPFS and deletes the staged file on release", async () => {
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  const files = new Map()
  const directory = {
    async getFileHandle(name, options = {}) {
      if (!files.has(name)) {
        if (!options.create) throw new DOMException('Not found', 'NotFoundError')
        files.set(name, { parts: [] })
      }
      const record = files.get(name)
      return {
        async createWritable() {
          record.parts = []
          return {
            async write(value) {
              const bytes = value instanceof ArrayBuffer
                ? new Uint8Array(value)
                : new Uint8Array(value.buffer, value.byteOffset, value.byteLength)
              record.parts.push(bytes.slice().buffer)
            },
            async close() {},
            async abort() { record.parts = [] },
          }
        },
        async getFile() { return new Blob(record.parts) },
      }
    },
    async removeEntry(name) { files.delete(name) },
    async *entries() {
      for (const name of files.keys()) yield [name, {}]
    },
  }
  const root = { async getDirectoryHandle() { return directory } }
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { storage: { async getDirectory() { return root } } },
  })
  try {
    const artifact = bytes('disk-backed ephemeral graph')
    const manager = new ModelArtifactManager({
      backend: 'none',
      namespace: 'triposplat/1/ephemeral/fp32',
      fetch: async () => new Response(artifact),
    })
    const prepared = await manager.prepareGraph('dino', {
      url: 'https://cdn.example.test/ephemeral.onnx',
      byteLength: artifact.byteLength,
      integrity: { algorithm: 'sha256', digest: digest(artifact) },
    })
    assert.equal(files.size, 1)
    assert.deepEqual(
      new Uint8Array(await (await fetch(prepared.graph.url)).arrayBuffer()),
      artifact,
    )
    prepared.release()
    await new Promise((resolve) => setTimeout(resolve, 0))
    assert.equal(files.size, 0)
    manager.dispose()
  } finally {
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator)
    else delete globalThis.navigator
  }
})

test('default OPFS load fails before downloading when declared artifacts exceed origin quota', async () => {
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  const directory = {
    async *entries() {},
    async removeEntry() {},
  }
  const root = { async getDirectoryHandle() { return directory } }
  let persisted = false
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      storage: {
        async getDirectory() { return root },
        async persist() { persisted = true; return true },
        async estimate() { return { quota: 100, usage: 10 } },
      },
    },
  })
  let workerCreated = false
  const model = new TripoSplatWebGPU({
    modelBaseUrl: 'https://cdn.example.test/quota/',
    executionProviders: ['wasm'],
    fetch: async (url) => {
      assert.match(String(url), /manifest\.json$/)
      return Response.json({
        name: 'triposplat-webgpu',
        version: 'quota-test',
        modelRevision: 'quota-test',
        precision: 'fp32',
        estimatedModelBytes: 1000,
        graphs: {},
      })
    },
    workerFactory: () => {
      workerCreated = true
      throw new Error('quota preflight must run before worker creation')
    },
  })
  try {
    await assert.rejects(
      model.load(),
      (error) => error instanceof ModelDownloadError
        && error.diagnostics.requiredBytes === 1000
        && error.diagnostics.availableBytes === 90,
    )
    assert.equal(persisted, true)
    assert.equal(workerCreated, false)
  } finally {
    await model.dispose()
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator)
    else delete globalThis.navigator
  }
})
