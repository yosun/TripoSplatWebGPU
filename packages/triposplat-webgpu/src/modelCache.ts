import {
  CancelledError,
  ModelDownloadError,
  ModelIntegrityError,
  TripoSplatError,
  throwIfAborted,
} from './errors.js'
import type {
  GraphManifestEntry,
  IntegrityDescriptor,
  ResolvedExternalDataDescriptor,
  ResolvedGraphManifestEntry,
  ResolvedTripoSplatModelManifest,
} from './manifest.js'
import type { CacheBackend, TripoSplatGraphName } from './types.js'
import { forwardReusableGraphInputCapability } from './runtime.js'
import type {
  GraphInfo,
  GraphRunResult,
  LoadGraphOptions,
  RunGraphOptions,
  TripoSplatRuntime,
} from './runtime.js'

const CACHE_NAME = 'ai3d-triposplat-webgpu-v1'
const OPFS_DIRECTORY = 'ai3d-triposplat-webgpu-v1'
const OPFS_EPHEMERAL_DIRECTORY = 'ai3d-triposplat-webgpu-ephemeral-v1'
const CACHE_URL_PREFIX = '/.ai3d-triposplat-webgpu-cache/v1/'

const SHA256_INITIAL = new Uint32Array([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
])

const SHA256_CONSTANTS = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
])

function rotateRight(value: number, bits: number): number {
  return (value >>> bits) | (value << (32 - bits))
}

/** Incremental SHA-256 so multi-hundred-megabyte model files need no ArrayBuffer copy. */
export class Sha256 {
  private readonly state = new Uint32Array(SHA256_INITIAL)
  private readonly block = new Uint8Array(64)
  private readonly words = new Uint32Array(64)
  private blockLength = 0
  private bytesHashed = 0
  private finished = false

  update(input: Uint8Array): this {
    if (this.finished) throw new Error('SHA-256 digest is already finalized.')
    this.bytesHashed += input.byteLength
    let offset = 0
    while (offset < input.byteLength) {
      const available = Math.min(64 - this.blockLength, input.byteLength - offset)
      this.block.set(input.subarray(offset, offset + available), this.blockLength)
      this.blockLength += available
      offset += available
      if (this.blockLength === 64) {
        this.compress(this.block)
        this.blockLength = 0
      }
    }
    return this
  }

  hex(): string {
    if (!this.finished) {
      const bitLength = this.bytesHashed * 8
      this.block[this.blockLength] = 0x80
      this.blockLength += 1
      if (this.blockLength > 56) {
        this.block.fill(0, this.blockLength)
        this.compress(this.block)
        this.blockLength = 0
      }
      this.block.fill(0, this.blockLength, 56)
      const view = new DataView(this.block.buffer)
      view.setUint32(56, Math.floor(bitLength / 0x1_0000_0000), false)
      view.setUint32(60, bitLength >>> 0, false)
      this.compress(this.block)
      this.finished = true
    }
    return Array.from(this.state, (value) => value.toString(16).padStart(8, '0')).join('')
  }

  private compress(block: Uint8Array): void {
    const view = new DataView(block.buffer, block.byteOffset, block.byteLength)
    for (let index = 0; index < 16; index += 1) this.words[index] = view.getUint32(index * 4, false)
    for (let index = 16; index < 64; index += 1) {
      const first = this.words[index - 15]
      const second = this.words[index - 2]
      const sigma0 = rotateRight(first, 7) ^ rotateRight(first, 18) ^ (first >>> 3)
      const sigma1 = rotateRight(second, 17) ^ rotateRight(second, 19) ^ (second >>> 10)
      this.words[index] = (this.words[index - 16] + sigma0 + this.words[index - 7] + sigma1) >>> 0
    }

    let a = this.state[0]
    let b = this.state[1]
    let c = this.state[2]
    let d = this.state[3]
    let e = this.state[4]
    let f = this.state[5]
    let g = this.state[6]
    let h = this.state[7]
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25)
      const choice = (e & f) ^ (~e & g)
      const temporary1 = (h + sum1 + choice + SHA256_CONSTANTS[index] + this.words[index]) >>> 0
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22)
      const majority = (a & b) ^ (a & c) ^ (b & c)
      const temporary2 = (sum0 + majority) >>> 0
      h = g
      g = f
      f = e
      e = (d + temporary1) >>> 0
      d = c
      c = b
      b = a
      a = (temporary1 + temporary2) >>> 0
    }
    this.state[0] = (this.state[0] + a) >>> 0
    this.state[1] = (this.state[1] + b) >>> 0
    this.state[2] = (this.state[2] + c) >>> 0
    this.state[3] = (this.state[3] + d) >>> 0
    this.state[4] = (this.state[4] + e) >>> 0
    this.state[5] = (this.state[5] + f) >>> 0
    this.state[6] = (this.state[6] + g) >>> 0
    this.state[7] = (this.state[7] + h) >>> 0
  }
}

export function sha256Hex(bytes: Uint8Array): string {
  return new Sha256().update(bytes).hex()
}

export interface ModelCacheEntry {
  cacheKey: string
  namespace: string
  label: string
  byteLength: number
  sha256: string
  integrityVerified: boolean
  createdAt: number
}

export interface StoredModelArtifact {
  blob: Blob
  metadata: ModelCacheEntry
}

export interface ModelArtifactWriter {
  write(chunk: Uint8Array): Promise<void>
  commit(metadata: ModelCacheEntry): Promise<StoredModelArtifact>
  abort(): Promise<void>
}

export interface ModelArtifactStorage {
  readonly backend: CacheBackend
  get(cacheKey: string): Promise<StoredModelArtifact | undefined>
  createWriter(cacheKey: string): Promise<ModelArtifactWriter>
  delete(cacheKey: string): Promise<void>
  entries(): Promise<ModelCacheEntry[]>
  clear(namespace?: string): Promise<void>
}

class BlobWriter implements ModelArtifactWriter {
  private readonly parts: ArrayBuffer[] = []
  private closed = false

  constructor(
    private readonly commitBlob: (blob: Blob, metadata: ModelCacheEntry) => Promise<StoredModelArtifact>,
  ) {}

  async write(chunk: Uint8Array): Promise<void> {
    if (this.closed) throw new Error('Artifact writer is closed.')
    const copy = new Uint8Array(chunk.byteLength)
    copy.set(chunk)
    this.parts.push(copy.buffer)
  }

  async commit(metadata: ModelCacheEntry): Promise<StoredModelArtifact> {
    if (this.closed) throw new Error('Artifact writer is closed.')
    this.closed = true
    return this.commitBlob(new Blob(this.parts, { type: 'application/octet-stream' }), metadata)
  }

  async abort(): Promise<void> {
    this.closed = true
    this.parts.length = 0
  }
}

/** In-memory adapter used by deterministic tests and advanced ephemeral integrations. */
export class MemoryModelArtifactStorage implements ModelArtifactStorage {
  readonly backend: CacheBackend = 'none'
  private readonly records = new Map<string, StoredModelArtifact>()

  async get(cacheKey: string): Promise<StoredModelArtifact | undefined> {
    return this.records.get(cacheKey)
  }

  async createWriter(cacheKey: string): Promise<ModelArtifactWriter> {
    return new BlobWriter(async (blob, metadata) => {
      const record = { blob, metadata }
      this.records.set(cacheKey, record)
      return record
    })
  }

  async delete(cacheKey: string): Promise<void> {
    this.records.delete(cacheKey)
  }

  async entries(): Promise<ModelCacheEntry[]> {
    return Array.from(this.records.values(), ({ metadata }) => metadata)
  }

  async clear(namespace?: string): Promise<void> {
    if (namespace === undefined) this.records.clear()
    else {
      for (const [key, record] of this.records) {
        if (record.metadata.namespace === namespace) this.records.delete(key)
      }
    }
  }
}

class EphemeralArtifactStorage implements ModelArtifactStorage {
  readonly backend: CacheBackend = 'none'
  async get(): Promise<undefined> { return undefined }
  async createWriter(): Promise<ModelArtifactWriter> {
    return new BlobWriter(async (blob, metadata) => ({ blob, metadata }))
  }
  async delete(): Promise<void> {}
  async entries(): Promise<ModelCacheEntry[]> { return [] }
  async clear(): Promise<void> {}
}

function cacheOrigin(): string {
  if (typeof location !== 'undefined' && /^https?:$/.test(location.protocol)) return location.origin
  return 'https://cache.triposplat.invalid'
}

function temporaryId(): string {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function metadataHeader(metadata: ModelCacheEntry): string {
  return encodeURIComponent(JSON.stringify(metadata))
}

function parseMetadataHeader(value: string | null): ModelCacheEntry | undefined {
  if (!value) return undefined
  try {
    const parsed: unknown = JSON.parse(decodeURIComponent(value))
    if (typeof parsed !== 'object' || parsed === null) return undefined
    const candidate = parsed as Partial<ModelCacheEntry>
    if (
      typeof candidate.cacheKey !== 'string'
      || typeof candidate.namespace !== 'string'
      || typeof candidate.label !== 'string'
      || typeof candidate.byteLength !== 'number'
      || typeof candidate.sha256 !== 'string'
      || typeof candidate.integrityVerified !== 'boolean'
      || typeof candidate.createdAt !== 'number'
    ) return undefined
    return candidate as ModelCacheEntry
  } catch {
    return undefined
  }
}

class CacheApiArtifactStorage implements ModelArtifactStorage {
  readonly backend: CacheBackend = 'cache-api'

  private async cache(): Promise<Cache> {
    if (typeof caches === 'undefined') {
      throw new ModelDownloadError('Cache API is unavailable in this browser context.', {
        diagnostics: { backend: this.backend },
      })
    }
    return caches.open(CACHE_NAME)
  }

  private async request(cacheKey: string): Promise<Request> {
    const id = sha256Hex(new TextEncoder().encode(cacheKey))
    return new Request(`${cacheOrigin()}${CACHE_URL_PREFIX}${id}`)
  }

  async get(cacheKey: string): Promise<StoredModelArtifact | undefined> {
    const cache = await this.cache()
    const response = await cache.match(await this.request(cacheKey))
    if (!response) return undefined
    const metadata = parseMetadataHeader(response.headers.get('x-ai3d-metadata'))
    if (!metadata || metadata.cacheKey !== cacheKey) return undefined
    return { blob: await response.blob(), metadata }
  }

  async createWriter(cacheKey: string): Promise<ModelArtifactWriter> {
    const cache = await this.cache()
    const request = await this.request(cacheKey)
    const temporaryRequest = new Request(`${request.url}.partial-${temporaryId()}`)
    const stream = new TransformStream<Uint8Array, Uint8Array>()
    const streamWriter = stream.writable.getWriter()
    const cacheWrite = cache.put(temporaryRequest, new Response(stream.readable, {
      headers: { 'content-type': 'application/octet-stream' },
    }))
    let closed = false
    return {
      async write(chunk) {
        if (closed) throw new Error('Artifact writer is closed.')
        await streamWriter.write(chunk)
      },
      async commit(metadata) {
        if (closed) throw new Error('Artifact writer is closed.')
        closed = true
        await streamWriter.close()
        await cacheWrite
        try {
          const temporary = await cache.match(temporaryRequest)
          if (!temporary) throw new Error('Temporary Cache API artifact was not committed.')
          await cache.put(request, new Response(temporary.body, {
            headers: {
              'content-type': 'application/octet-stream',
              'content-length': String(metadata.byteLength),
              'x-ai3d-metadata': metadataHeader(metadata),
            },
          }))
          const finalResponse = await cache.match(request)
          if (!finalResponse) throw new Error('Verified Cache API artifact was not committed.')
          return { blob: await finalResponse.blob(), metadata }
        } finally {
          await cache.delete(temporaryRequest)
        }
      },
      async abort() {
        if (!closed) {
          closed = true
          await streamWriter.abort().catch(() => undefined)
        }
        await cacheWrite.catch(() => undefined)
        await cache.delete(temporaryRequest)
      },
    }
  }

  async delete(cacheKey: string): Promise<void> {
    const cache = await this.cache()
    await cache.delete(await this.request(cacheKey))
  }

  async entries(): Promise<ModelCacheEntry[]> {
    const cache = await this.cache()
    const result: ModelCacheEntry[] = []
    for (const request of await cache.keys()) {
      if (!new URL(request.url).pathname.startsWith(CACHE_URL_PREFIX)) continue
      if (request.url.includes('.partial-')) {
        await cache.delete(request)
        continue
      }
      const response = await cache.match(request)
      const metadata = parseMetadataHeader(response?.headers.get('x-ai3d-metadata') ?? null)
      if (metadata) result.push(metadata)
    }
    return result
  }

  async clear(namespace?: string): Promise<void> {
    const cache = await this.cache()
    if (namespace === undefined) {
      await caches.delete(CACHE_NAME)
      return
    }
    for (const entry of await this.entries()) {
      if (entry.namespace === namespace) await cache.delete(await this.request(entry.cacheKey))
    }
  }
}

interface OpfsMetadataFile extends ModelCacheEntry {
  id: string
}

interface IterableDirectoryHandle extends FileSystemDirectoryHandle {
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>
}

class OpfsArtifactStorage implements ModelArtifactStorage {
  readonly backend: CacheBackend = 'opfs'

  private async directory(): Promise<IterableDirectoryHandle> {
    const storage = typeof navigator === 'undefined'
      ? undefined
      : (navigator.storage as StorageManager & { getDirectory?: () => Promise<FileSystemDirectoryHandle> })
    if (!storage?.getDirectory) {
      throw new ModelDownloadError('Origin Private File System is unavailable in this browser context.', {
        diagnostics: { backend: this.backend },
      })
    }
    const root = await storage.getDirectory()
    return root.getDirectoryHandle(OPFS_DIRECTORY, { create: true }) as Promise<IterableDirectoryHandle>
  }

  private id(cacheKey: string): string {
    return sha256Hex(new TextEncoder().encode(cacheKey))
  }

  private async metadata(directory: FileSystemDirectoryHandle, id: string): Promise<OpfsMetadataFile | undefined> {
    try {
      const handle = await directory.getFileHandle(`${id}.json`)
      const parsed: unknown = JSON.parse(await (await handle.getFile()).text())
      if (typeof parsed !== 'object' || parsed === null || (parsed as { id?: unknown }).id !== id) return undefined
      return parsed as OpfsMetadataFile
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotFoundError') return undefined
      return undefined
    }
  }

  async get(cacheKey: string): Promise<StoredModelArtifact | undefined> {
    const directory = await this.directory()
    const id = this.id(cacheKey)
    const metadata = await this.metadata(directory, id)
    if (!metadata || metadata.cacheKey !== cacheKey) return undefined
    try {
      const file = await (await directory.getFileHandle(`${id}.bin`)).getFile()
      return { blob: file, metadata }
    } catch {
      return undefined
    }
  }

  async createWriter(cacheKey: string): Promise<ModelArtifactWriter> {
    const directory = await this.directory()
    const id = this.id(cacheKey)
    const dataHandle = await directory.getFileHandle(`${id}.bin`, { create: true })
    const writable = await dataHandle.createWritable()
    let closed = false
    return {
      async write(chunk) {
        if (closed) throw new Error('Artifact writer is closed.')
        const copy = new Uint8Array(chunk.byteLength)
        copy.set(chunk)
        await writable.write(copy.buffer)
      },
      async commit(metadata) {
        if (closed) throw new Error('Artifact writer is closed.')
        closed = true
        await writable.close()
        const metadataHandle = await directory.getFileHandle(`${id}.json`, { create: true })
        const metadataWriter = await metadataHandle.createWritable()
        await metadataWriter.write(JSON.stringify({ ...metadata, id }))
        await metadataWriter.close()
        return { blob: await dataHandle.getFile(), metadata }
      },
      async abort() {
        if (!closed) {
          closed = true
          await writable.abort().catch(() => undefined)
        }
        await directory.removeEntry(`${id}.bin`).catch(() => undefined)
        await directory.removeEntry(`${id}.json`).catch(() => undefined)
      },
    }
  }

  async delete(cacheKey: string): Promise<void> {
    const directory = await this.directory()
    const id = this.id(cacheKey)
    await directory.removeEntry(`${id}.bin`).catch(() => undefined)
    await directory.removeEntry(`${id}.json`).catch(() => undefined)
  }

  async entries(): Promise<ModelCacheEntry[]> {
    const directory = await this.directory()
    const result: ModelCacheEntry[] = []
    for await (const [name] of directory.entries()) {
      if (!name.endsWith('.json')) continue
      const metadata = await this.metadata(directory, name.slice(0, -5))
      if (metadata) result.push(metadata)
    }
    return result
  }

  async clear(namespace?: string): Promise<void> {
    const directory = await this.directory()
    if (namespace === undefined) {
      for await (const [name] of directory.entries()) await directory.removeEntry(name).catch(() => undefined)
      return
    }
    for (const entry of await this.entries()) {
      if (entry.namespace === namespace) await this.delete(entry.cacheKey)
    }
  }
}

/**
 * Disk-backed, non-persistent staging for `cache: 'none'`.
 *
 * Multi-gigabyte sidecars must not be accumulated as JS ArrayBuffer chunks.
 * When OPFS is available this writer streams each artifact into a temporary
 * origin-private file, returns a file-backed Blob for ORT session creation, and
 * removes the directory entry as soon as the prepared graph is released.
 */
class EphemeralOpfsArtifactStorage implements ModelArtifactStorage {
  readonly backend: CacheBackend = 'none'
  private readonly ids = new Map<string, string>()

  private async directory(): Promise<IterableDirectoryHandle> {
    const storage = navigator.storage as StorageManager & {
      getDirectory: () => Promise<FileSystemDirectoryHandle>
    }
    const root = await storage.getDirectory()
    return root.getDirectoryHandle(OPFS_EPHEMERAL_DIRECTORY, { create: true }) as Promise<IterableDirectoryHandle>
  }

  async get(): Promise<undefined> { return undefined }

  async createWriter(cacheKey: string): Promise<ModelArtifactWriter> {
    const directory = await this.directory()
    const id = `${sha256Hex(new TextEncoder().encode(cacheKey))}-${temporaryId()}.bin`
    const handle = await directory.getFileHandle(id, { create: true })
    const writable = await handle.createWritable()
    let closed = false
    return {
      async write(chunk) {
        if (closed) throw new Error('Artifact writer is closed.')
        const copy = new Uint8Array(chunk.byteLength)
        copy.set(chunk)
        await writable.write(copy.buffer)
      },
      commit: async (metadata) => {
        if (closed) throw new Error('Artifact writer is closed.')
        closed = true
        await writable.close()
        const previous = this.ids.get(cacheKey)
        this.ids.set(cacheKey, id)
        if (previous) await directory.removeEntry(previous).catch(() => undefined)
        return { blob: await handle.getFile(), metadata }
      },
      abort: async () => {
        if (!closed) {
          closed = true
          await writable.abort().catch(() => undefined)
        }
        await directory.removeEntry(id).catch(() => undefined)
      },
    }
  }

  async delete(cacheKey: string): Promise<void> {
    const id = this.ids.get(cacheKey)
    if (!id) return
    this.ids.delete(cacheKey)
    const directory = await this.directory()
    await directory.removeEntry(id).catch(() => undefined)
  }

  async entries(): Promise<ModelCacheEntry[]> { return [] }

  async clear(): Promise<void> {
    this.ids.clear()
    const directory = await this.directory()
    for await (const [name] of directory.entries()) {
      await directory.removeEntry(name).catch(() => undefined)
    }
  }
}

export interface ModelArtifactProgress {
  graph: TripoSplatGraphName
  label: string
  source: 'network' | 'cache'
  loadedBytes: number
  totalBytes?: number
}

export interface ModelArtifactManagerOptions {
  backend: CacheBackend
  namespace: string
  fetch?: typeof globalThis.fetch
  requestInit?: Omit<RequestInit, 'signal' | 'body' | 'method'>
  storage?: ModelArtifactStorage
  onProgress?: (progress: ModelArtifactProgress) => void
}

export interface PreparedGraphArtifacts {
  graph: ResolvedGraphManifestEntry
  release(): void
}

interface ArtifactDescriptor {
  graph: TripoSplatGraphName
  label: string
  url: string
  byteLength?: number
  integrity?: IntegrityDescriptor
}

interface VerifiedArtifact extends StoredModelArtifact {
  source: 'network' | 'cache'
}

function persistentStorage(backend: CacheBackend): ModelArtifactStorage {
  if (backend === 'opfs') return new OpfsArtifactStorage()
  if (backend === 'cache-api') return new CacheApiArtifactStorage()
  const storage = typeof navigator === 'undefined'
    ? undefined
    : navigator.storage as StorageManager & { getDirectory?: () => Promise<FileSystemDirectoryHandle> }
  if (typeof storage?.getDirectory === 'function') return new EphemeralOpfsArtifactStorage()
  return new EphemeralArtifactStorage()
}

async function hashBlob(blob: Blob, signal?: AbortSignal): Promise<{ sha256: string; byteLength: number }> {
  const hash = new Sha256()
  let byteLength = 0
  const reader = blob.stream().getReader()
  try {
    while (true) {
      throwIfAborted(signal)
      const { done, value } = await reader.read()
      if (done) break
      hash.update(value)
      byteLength += value.byteLength
    }
  } catch (error) {
    await reader.cancel(error).catch(() => undefined)
    throw error
  } finally {
    reader.releaseLock()
  }
  return { sha256: hash.hex(), byteLength }
}

function assertArtifactIntegrity(
  descriptor: ArtifactDescriptor,
  actual: { sha256: string; byteLength: number },
): void {
  if (descriptor.byteLength !== undefined && actual.byteLength !== descriptor.byteLength) {
    throw new ModelIntegrityError(
      `Model artifact '${descriptor.label}' has ${actual.byteLength} bytes; expected ${descriptor.byteLength}.`,
      {
        diagnostics: {
          graph: descriptor.graph,
          label: descriptor.label,
          expectedByteLength: descriptor.byteLength,
          actualByteLength: actual.byteLength,
        },
      },
    )
  }
  if (descriptor.integrity && actual.sha256 !== descriptor.integrity.digest) {
    throw new ModelIntegrityError(`SHA-256 mismatch for model artifact '${descriptor.label}'.`, {
      diagnostics: {
        graph: descriptor.graph,
        label: descriptor.label,
        expectedSha256: descriptor.integrity.digest,
        actualSha256: actual.sha256,
      },
    })
  }
}

export function modelCacheNamespace(
  manifest: Pick<ResolvedTripoSplatModelManifest, 'name' | 'version' | 'modelRevision' | 'precision'>,
): string {
  return `${manifest.name}/${manifest.version}/${manifest.modelRevision}/${manifest.precision}`
}

export class ModelArtifactManager {
  readonly backend: CacheBackend
  readonly namespace: string

  private readonly fetchImplementation: typeof globalThis.fetch
  private readonly requestInit: Omit<RequestInit, 'signal' | 'body' | 'method'> | undefined
  private readonly storage: ModelArtifactStorage
  private readonly onProgress: ((progress: ModelArtifactProgress) => void) | undefined
  private readonly pending = new Map<string, Promise<VerifiedArtifact>>()
  private readonly verifiedArtifacts = new Map<string, VerifiedArtifact>()
  private readonly objectUrls = new Set<string>()

  constructor(options: ModelArtifactManagerOptions) {
    this.backend = options.backend
    this.namespace = options.namespace
    const fetchImplementation = options.fetch ?? globalThis.fetch
    if (typeof fetchImplementation !== 'function') {
      throw new ModelDownloadError('No Fetch implementation is available for model artifacts.')
    }
    this.fetchImplementation = fetchImplementation
    this.requestInit = options.requestInit
    this.storage = options.storage ?? persistentStorage(options.backend)
    this.onProgress = options.onProgress
  }

  async prefetchManifest(
    manifest: ResolvedTripoSplatModelManifest,
    signal?: AbortSignal,
  ): Promise<void> {
    for (const graph of Object.keys(manifest.graphs) as TripoSplatGraphName[]) {
      const descriptor = manifest.graphs[graph]
      if (!descriptor) continue
      for (const asset of this.descriptors(graph, descriptor)) {
        throwIfAborted(signal)
        await this.getArtifact(asset, signal)
      }
    }
  }

  async prepareGraph(
    name: TripoSplatGraphName,
    graph: ResolvedGraphManifestEntry,
    signal?: AbortSignal,
  ): Promise<PreparedGraphArtifacts> {
    if (typeof URL.createObjectURL !== 'function') {
      throw new ModelDownloadError('Blob object URLs are unavailable for verified ONNX artifacts.')
    }
    const urls: string[] = []
    let ephemeralCacheKeys: string[] = []
    try {
      const descriptors = this.descriptors(name, graph)
      if (this.backend === 'none') {
        ephemeralCacheKeys = descriptors.map((descriptor) => this.cacheKey(descriptor))
      }
      const graphArtifact = await this.getArtifact(descriptors[0], signal)
      const graphUrl = URL.createObjectURL(graphArtifact.blob)
      urls.push(graphUrl)
      this.objectUrls.add(graphUrl)
      const externalData: ResolvedExternalDataDescriptor[] = []
      for (let index = 1; index < descriptors.length; index += 1) {
        const descriptor = descriptors[index]
        const artifact = await this.getArtifact(descriptor, signal)
        const url = URL.createObjectURL(artifact.blob)
        urls.push(url)
        this.objectUrls.add(url)
        const source = graph.externalData?.[index - 1]
        if (!source) throw new ModelDownloadError(`Missing external-data descriptor ${index - 1} for '${name}'.`)
        externalData.push({ ...source, url })
      }
      const prepared: ResolvedGraphManifestEntry = { ...graph, url: graphUrl }
      if (externalData.length > 0) prepared.externalData = externalData
      else delete prepared.externalData
      let released = false
      return {
        graph: prepared,
        release: () => {
          if (released) return
          released = true
          for (const url of urls) {
            URL.revokeObjectURL(url)
            this.objectUrls.delete(url)
          }
          for (const cacheKey of ephemeralCacheKeys) {
            void this.storage.delete(cacheKey).catch(() => undefined)
          }
        },
      }
    } catch (error) {
      for (const url of urls) {
        URL.revokeObjectURL(url)
        this.objectUrls.delete(url)
      }
      await Promise.all(ephemeralCacheKeys.map((cacheKey) => this.storage.delete(cacheKey).catch(() => undefined)))
      throw error
    }
  }

  async status(): Promise<ModelCacheEntry[]> {
    return this.storage.entries()
  }

  async clear(namespace = this.namespace): Promise<void> {
    await this.storage.clear(namespace)
    if (namespace === this.namespace) this.verifiedArtifacts.clear()
  }

  dispose(): void {
    for (const url of this.objectUrls) URL.revokeObjectURL(url)
    this.objectUrls.clear()
    this.verifiedArtifacts.clear()
    if (this.backend === 'none') void this.storage.clear().catch(() => undefined)
  }

  private descriptors(name: TripoSplatGraphName, graph: ResolvedGraphManifestEntry): ArtifactDescriptor[] {
    const graphDescriptor: ArtifactDescriptor = { graph: name, label: `${name}:graph`, url: graph.url }
    if (graph.byteLength !== undefined) graphDescriptor.byteLength = graph.byteLength
    if (graph.integrity !== undefined) graphDescriptor.integrity = graph.integrity
    const result = [graphDescriptor]
    for (const external of graph.externalData ?? []) {
      const descriptor: ArtifactDescriptor = {
        graph: name,
        label: `${name}:external:${external.path}`,
        url: external.url,
      }
      if (external.byteLength !== undefined) descriptor.byteLength = external.byteLength
      if (external.integrity !== undefined) descriptor.integrity = external.integrity
      result.push(descriptor)
    }
    return result
  }

  private cacheKey(descriptor: ArtifactDescriptor): string {
    // Never persist signed URLs or credentials in OPFS metadata/Cache headers.
    const identity = descriptor.integrity?.digest
      ?? sha256Hex(new TextEncoder().encode(descriptor.url))
    return `${this.namespace}|${descriptor.label}|${identity}`
  }

  private getArtifact(descriptor: ArtifactDescriptor, signal?: AbortSignal): Promise<VerifiedArtifact> {
    const cacheKey = this.cacheKey(descriptor)
    if (this.backend !== 'none') {
      const verified = this.verifiedArtifacts.get(cacheKey)
      if (verified) {
        try {
          throwIfAborted(signal)
          assertArtifactIntegrity(descriptor, {
            sha256: verified.metadata.sha256,
            byteLength: verified.metadata.byteLength,
          })
          return Promise.resolve(verified)
        } catch (error) {
          return Promise.reject(error)
        }
      }
    }
    const existing = this.pending.get(cacheKey)
    if (existing) return existing
    const pending = this.loadArtifact(cacheKey, descriptor, signal)
      .then((artifact) => {
        if (this.backend !== 'none') this.verifiedArtifacts.set(cacheKey, artifact)
        return artifact
      })
      .finally(() => this.pending.delete(cacheKey))
    this.pending.set(cacheKey, pending)
    return pending
  }

  private async loadArtifact(
    cacheKey: string,
    descriptor: ArtifactDescriptor,
    signal?: AbortSignal,
  ): Promise<VerifiedArtifact> {
    throwIfAborted(signal)
    let cached: StoredModelArtifact | undefined
    try {
      cached = await this.storage.get(cacheKey)
    } catch (cause) {
      throw new ModelDownloadError(`Could not read cached model artifact '${descriptor.label}'.`, {
        cause,
        diagnostics: { backend: this.backend, graph: descriptor.graph, label: descriptor.label },
      })
    }
    if (cached) {
      try {
        const actual = await hashBlob(cached.blob, signal)
        assertArtifactIntegrity(descriptor, actual)
        if (
          actual.sha256 !== cached.metadata.sha256
          || actual.byteLength !== cached.metadata.byteLength
          || cached.metadata.namespace !== this.namespace
        ) throw new ModelIntegrityError(`Cached model artifact '${descriptor.label}' is corrupted.`)
        this.onProgress?.({
          graph: descriptor.graph,
          label: descriptor.label,
          source: 'cache',
          loadedBytes: actual.byteLength,
          totalBytes: descriptor.byteLength ?? actual.byteLength,
        })
        return { ...cached, source: 'cache' }
      } catch (error) {
        if (error instanceof CancelledError) throw error
        try {
          await this.storage.delete(cacheKey)
        } catch (cause) {
          throw new ModelDownloadError(`Could not evict cached model artifact '${descriptor.label}'.`, {
            cause,
            diagnostics: { backend: this.backend, graph: descriptor.graph, label: descriptor.label },
          })
        }
        // A corrupt local record is discarded; the authoritative CDN object is
        // downloaded and independently verified before the session can load.
      }
    }

    let response: Response
    try {
      const init: RequestInit = { ...this.requestInit, method: 'GET' }
      if (signal !== undefined) init.signal = signal
      const fetchImplementation = this.fetchImplementation
      response = await fetchImplementation(descriptor.url, init)
    } catch (cause) {
      if (signal?.aborted) throw new CancelledError(undefined, { cause: signal.reason })
      throw new ModelDownloadError(`Could not download model artifact '${descriptor.label}'.`, {
        cause,
        diagnostics: { graph: descriptor.graph, label: descriptor.label },
      })
    }
    if (!response.ok) {
      throw new ModelDownloadError(
        `Model artifact '${descriptor.label}' failed with HTTP ${response.status} ${response.statusText}.`,
        { diagnostics: { graph: descriptor.graph, label: descriptor.label, status: response.status } },
      )
    }

    let writer: ModelArtifactWriter
    try {
      writer = await this.storage.createWriter(cacheKey)
    } catch (cause) {
      await response.body?.cancel().catch(() => undefined)
      throw new ModelDownloadError(`Could not open ${this.backend} storage for '${descriptor.label}'.`, {
        cause,
        diagnostics: { backend: this.backend, graph: descriptor.graph, label: descriptor.label },
      })
    }
    const hash = new Sha256()
    let byteLength = 0
    const contentLength = Number(response.headers.get('content-length'))
    const totalBytes = descriptor.byteLength
      ?? (Number.isSafeInteger(contentLength) && contentLength >= 0 ? contentLength : undefined)
    try {
      const body = response.body
      if (body) {
        const reader = body.getReader()
        try {
          while (true) {
            throwIfAborted(signal)
            const { done, value } = await reader.read()
            if (done) break
            hash.update(value)
            byteLength += value.byteLength
            await writer.write(value)
            const progress: ModelArtifactProgress = {
              graph: descriptor.graph,
              label: descriptor.label,
              source: 'network',
              loadedBytes: byteLength,
            }
            if (totalBytes !== undefined) progress.totalBytes = totalBytes
            this.onProgress?.(progress)
          }
        } catch (error) {
          await reader.cancel(error).catch(() => undefined)
          throw error
        } finally {
          reader.releaseLock()
        }
      }
      const actual = { sha256: hash.hex(), byteLength }
      assertArtifactIntegrity(descriptor, actual)
      const metadata: ModelCacheEntry = {
        cacheKey,
        namespace: this.namespace,
        label: descriptor.label,
        byteLength,
        sha256: actual.sha256,
        integrityVerified: descriptor.integrity !== undefined,
        createdAt: Date.now(),
      }
      const stored = await writer.commit(metadata)
      return { ...stored, source: 'network' }
    } catch (error) {
      await writer.abort().catch(() => undefined)
      if (signal?.aborted && !(error instanceof CancelledError)) {
        throw new CancelledError(undefined, { cause: signal.reason })
      }
      if (error instanceof TripoSplatError) throw error
      throw new ModelDownloadError(`Could not persist model artifact '${descriptor.label}'.`, {
        cause: error,
        diagnostics: { backend: this.backend, graph: descriptor.graph, label: descriptor.label },
      })
    }
  }
}

/**
 * Decorate a runtime so every ONNX graph and external-data shard is fetched,
 * verified, and materialized locally before the worker creates a session.
 */
export function withVerifiedModelArtifacts(
  runtime: TripoSplatRuntime,
  manager: ModelArtifactManager,
  graphForSession: Readonly<Record<string, TripoSplatGraphName>>,
): TripoSplatRuntime {
  const decorated: TripoSplatRuntime = {
    get disposed() { return runtime.disposed },
    async loadGraph(
      sessionId: string,
      graph: ResolvedGraphManifestEntry,
      options: LoadGraphOptions = {},
    ): Promise<GraphInfo> {
      const name = graphForSession[sessionId]
      if (!name) {
        throw new ModelDownloadError(`No model-artifact identity is configured for session '${sessionId}'.`, {
          diagnostics: { sessionId },
        })
      }
      const prepared = await manager.prepareGraph(name, graph, options.signal)
      try {
        // The built-in runtime removes the signal before posting to the worker,
        // but still needs it locally to terminate an in-flight ORT graph load.
        return await runtime.loadGraph(sessionId, prepared.graph, options)
      } finally {
        prepared.release()
      }
    },
    runGraph(
      sessionId: string,
      inputs: Parameters<TripoSplatRuntime['runGraph']>[1],
      options?: RunGraphOptions,
    ): Promise<GraphRunResult> {
      return runtime.runGraph(sessionId, inputs, options)
    },
    disposeGraph(sessionId: string): Promise<boolean> {
      return runtime.disposeGraph(sessionId)
    },
    async dispose(): Promise<void> {
      manager.dispose()
      await runtime.dispose()
    },
  }
  forwardReusableGraphInputCapability(runtime, decorated)
  return decorated
}

export interface ModelCacheStatusOptions {
  backend?: Exclude<CacheBackend, 'none'>
  namespace?: string
}

export interface ModelCacheBackendStatus {
  backend: Exclude<CacheBackend, 'none'>
  available: boolean
  entryCount: number
  totalBytes: number
  entries: ModelCacheEntry[]
  error?: string
}

export interface ModelCacheStatus {
  entryCount: number
  totalBytes: number
  backends: ModelCacheBackendStatus[]
}

function persistentBackends(
  requested?: Exclude<CacheBackend, 'none'>,
): Array<Exclude<CacheBackend, 'none'>> {
  return requested ? [requested] : ['opfs', 'cache-api']
}

export async function getModelCacheStatus(
  options: ModelCacheStatusOptions = {},
): Promise<ModelCacheStatus> {
  const backends: ModelCacheBackendStatus[] = []
  for (const backend of persistentBackends(options.backend)) {
    try {
      const storage = persistentStorage(backend)
      const allEntries = await storage.entries()
      const entries = options.namespace === undefined
        ? allEntries
        : allEntries.filter((entry) => entry.namespace === options.namespace)
      backends.push({
        backend,
        available: true,
        entryCount: entries.length,
        totalBytes: entries.reduce((total, entry) => total + entry.byteLength, 0),
        entries,
      })
    } catch (error) {
      backends.push({
        backend,
        available: false,
        entryCount: 0,
        totalBytes: 0,
        entries: [],
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
  return {
    entryCount: backends.reduce((total, status) => total + status.entryCount, 0),
    totalBytes: backends.reduce((total, status) => total + status.totalBytes, 0),
    backends,
  }
}

export async function clearModelCache(options: ModelCacheStatusOptions = {}): Promise<void> {
  for (const backend of persistentBackends(options.backend)) {
    try {
      await persistentStorage(backend).clear(options.namespace)
    } catch {
      // Clearing all backends is best-effort on browsers that expose only one.
      if (options.backend !== undefined) throw new ModelDownloadError(`Could not clear ${backend} model cache.`)
    }
  }
}

export function graphArtifactByteLength(graph: GraphManifestEntry): number | undefined {
  const lengths = [graph.byteLength, ...(graph.externalData ?? []).map((entry) => entry.byteLength)]
  return lengths.every((length) => length !== undefined)
    ? lengths.reduce((total, length) => total + (length ?? 0), 0)
    : undefined
}
