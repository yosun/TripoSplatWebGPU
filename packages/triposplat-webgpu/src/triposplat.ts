import type { GaussianScene } from '@ai3d/gaussian-scene'

import { checkCompatibility } from './compatibility.js'
import {
  CancelledError,
  GraphCapabilityError,
  ManifestError,
  ModelDownloadError,
  TripoSplatError,
  throwIfAborted,
  WebGPUUnavailableError,
} from './errors.js'
import {
  configuredGraphNames,
  fetchModelManifest,
  REQUIRED_GENERATION_GRAPHS,
  type ResolvedTripoSplatModelManifest,
} from './manifest.js'
import {
  ModelArtifactManager,
  modelCacheNamespace,
  withVerifiedModelArtifacts,
} from './modelCache.js'
import { runBuiltInTripoSplatPipeline } from './pipeline.js'
import {
  normalizeTripoSplatImageInput,
  type NormalizedTripoSplatImage,
  type NormalizeTripoSplatImageOptions,
  type TripoSplatBackgroundRemover,
} from './preprocess.js'
import { createRuntime, type RuntimeStatus, type TripoSplatRuntime } from './runtime.js'
import type {
  CompatibilityOptions,
  CompatibilityReport,
  GenerateOptions,
  LoadOptions,
  PipelineCapabilities,
  TripoSplatGraphName,
  TripoSplatInput,
  TripoSplatOptions,
} from './types.js'

const SESSION_IDS: Readonly<Record<TripoSplatGraphName, string>> = {
  dino: 'triposplat/dino',
  vae: 'triposplat/vae',
  dit: 'triposplat/dit',
  octree: 'triposplat/octree',
  gaussianDecoder: 'triposplat/gaussian-decoder',
}

const GRAPH_FOR_SESSION: Readonly<Record<string, TripoSplatGraphName>> = Object.freeze(
  Object.fromEntries(
    Object.entries(SESSION_IDS).map(([name, sessionId]) => [sessionId, name as TripoSplatGraphName]),
  ),
)

export interface TripoSplatPipelineContext {
  input: TripoSplatInput
  options: GenerateOptions
  manifest: ResolvedTripoSplatModelManifest
  runtime: TripoSplatRuntime
  sessionIds: Readonly<Record<TripoSplatGraphName, string>>
  preprocess: TripoSplatPreprocessor
  removeBackground?: TripoSplatBackgroundRemover
}

export type TripoSplatPreprocessor = (
  input: TripoSplatInput,
  options: NormalizeTripoSplatImageOptions,
) => Promise<NormalizedTripoSplatImage>

/**
 * Advanced integration boundary used while graph exports are being staged.
 * A pipeline must preserve official TripoSplat numerical behavior and return
 * the canonical Gaussian scene contract.
 */
export type TripoSplatPipelineExecutor = (
  context: TripoSplatPipelineContext,
) => Promise<GaussianScene>

export interface TripoSplatWebGPUOptions extends TripoSplatOptions {
  pipeline?: TripoSplatPipelineExecutor
  /** Advanced override for a validated browser-local preprocessing/segmentation stage. */
  preprocess?: TripoSplatPreprocessor
  /** Browser-local opaque-image remover, normally backed by a separately validated BiRefNet graph. */
  removeBackground?: TripoSplatBackgroundRemover
}

function browserBaseUrl(): string {
  if (typeof document !== 'undefined') return document.baseURI
  if (typeof location !== 'undefined') return location.href
  throw new ManifestError('Relative modelBaseUrl needs a browser document or an absolute URL.')
}

function directoryUrl(value: string): URL {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    url = new URL(value, browserBaseUrl())
  }
  if (!url.pathname.endsWith('/')) url.pathname += '/'
  return url
}

function manifestUrl(options: TripoSplatOptions): URL {
  const base = directoryUrl(options.modelBaseUrl)
  return new URL(options.manifestUrl ?? 'manifest.json', base)
}

function declaredArtifactBytes(manifest: ResolvedTripoSplatModelManifest): number {
  if (manifest.estimatedModelBytes !== undefined) return manifest.estimatedModelBytes
  let total = 0
  for (const graph of Object.values(manifest.graphs)) {
    if (!graph) continue
    if (graph.byteLength === undefined) return 0
    total += graph.byteLength
    for (const external of graph.externalData ?? []) {
      if (external.byteLength === undefined) return 0
      total += external.byteLength
    }
  }
  return total
}

function largestDeclaredArtifact(manifest: ResolvedTripoSplatModelManifest): number {
  let largest = 0
  for (const graph of Object.values(manifest.graphs)) {
    if (!graph) continue
    largest = Math.max(largest, graph.byteLength ?? 0)
    for (const external of graph.externalData ?? []) {
      largest = Math.max(largest, external.byteLength ?? 0)
    }
  }
  return largest
}

async function preflightPersistentStorage(
  manifest: ResolvedTripoSplatModelManifest,
  backend: Exclude<TripoSplatOptions['cache'], 'none' | undefined>,
  cachedBytes: number,
  onProgress: LoadOptions['onProgress'],
): Promise<void> {
  if (typeof navigator === 'undefined') return
  const storage = navigator.storage as StorageManager & {
    estimate?: () => Promise<StorageEstimate>
    persist?: () => Promise<boolean>
  } | undefined
  if (!storage) return
  if (typeof storage.persist === 'function') {
    const persistent = await storage.persist().catch(() => false)
    if (!persistent) {
      onProgress?.({
        stage: 'graphs',
        message: 'Browser storage is not persistent; verified model files may be evicted under storage pressure.',
      })
    }
  }
  if (typeof storage.estimate !== 'function') return
  const estimate = await storage.estimate().catch(() => undefined)
  const quota = estimate?.quota
  const usage = estimate?.usage
  if (!Number.isFinite(quota) || !Number.isFinite(usage)) return
  const declaredBytes = declaredArtifactBytes(manifest)
  if (declaredBytes === 0) return
  const missingBytes = Math.max(0, declaredBytes - cachedBytes)
  // Cache API promotion briefly keeps the current temporary and final object;
  // OPFS commits in place and needs only the missing verified bytes.
  const requiredBytes = missingBytes + (backend === 'cache-api' ? largestDeclaredArtifact(manifest) : 0)
  const availableBytes = Math.max(0, (quota as number) - (usage as number))
  if (requiredBytes > availableBytes) {
    throw new ModelDownloadError(
      `Insufficient browser storage for TripoSplat: ${requiredBytes} bytes are required but `
        + `${availableBytes} bytes are available. Clear model caches or free disk space.`,
      {
        diagnostics: {
          backend,
          quotaBytes: quota,
          usageBytes: usage,
          availableBytes,
          declaredModelBytes: declaredBytes,
          cachedBytes,
          requiredBytes,
        },
      },
    )
  }
}

function capabilityReport(
  manifest: ResolvedTripoSplatModelManifest | undefined,
): PipelineCapabilities {
  const configured = manifest ? configuredGraphNames(manifest) : []
  const missing = REQUIRED_GENERATION_GRAPHS.filter((name) => !configured.includes(name))
  const reasons: string[] = []
  if (!manifest) reasons.push('The versioned model manifest has not been loaded.')
  if (missing.length > 0) reasons.push(`Missing graph descriptors: ${missing.join(', ')}.`)
  return {
    manifestLoaded: manifest !== undefined,
    configuredGraphs: configured,
    missingGraphs: missing,
    encoderSlice: configured.includes('dino') || configured.includes('vae'),
    fullGeneration: missing.length === 0,
    reasons,
  }
}

/** Framework-neutral lifecycle facade. It never silently substitutes missing model stages. */
export class TripoSplatWebGPU {
  private readonly options: TripoSplatWebGPUOptions
  private runtimeValue: TripoSplatRuntime | undefined
  private artifactManagerValue: ModelArtifactManager | undefined
  private manifestValue: ResolvedTripoSplatModelManifest | undefined
  private loadPromise: Promise<void> | undefined
  private generationTail: Promise<void> = Promise.resolve()
  private disposePromise: Promise<void> | undefined
  private disposedValue = false

  constructor(options: TripoSplatWebGPUOptions) {
    if (!options || typeof options.modelBaseUrl !== 'string' || options.modelBaseUrl.trim().length === 0) {
      throw new TypeError('modelBaseUrl must be a non-empty URL or browser-relative path.')
    }
    if (options.executionProviders?.length === 0) throw new TypeError('executionProviders must not be empty.')
    this.options = { ...options }
  }

  static checkCompatibility(options?: CompatibilityOptions): Promise<CompatibilityReport> {
    return checkCompatibility(options)
  }

  get manifest(): ResolvedTripoSplatModelManifest | undefined {
    return this.manifestValue
  }

  get capabilities(): PipelineCapabilities {
    return capabilityReport(this.manifestValue)
  }

  async load(options: LoadOptions = {}): Promise<void> {
    this.assertUsable()
    throwIfAborted(options.signal)
    if (this.loadPromise) return this.loadPromise
    this.loadPromise = this.performLoad(options).catch(async (error: unknown) => {
      const runtime = this.runtimeValue
      const artifactManager = this.artifactManagerValue
      this.runtimeValue = undefined
      this.artifactManagerValue = undefined
      this.manifestValue = undefined
      artifactManager?.dispose()
      await runtime?.dispose().catch(() => undefined)
      // Keep the rejected promise installed until cleanup is complete. A retry
      // can then never publish a new manager that this failed load disposes.
      this.loadPromise = undefined
      throw error
    })
    return this.loadPromise
  }

  async generate(input: TripoSplatInput, options: GenerateOptions = {}): Promise<GaussianScene> {
    this.assertUsable()
    throwIfAborted(options.signal)
    // Every built-in stage uses a stable session id. Serialize generations so
    // repeated calls cannot load/dispose the same worker session concurrently.
    const previousGeneration = this.generationTail
    const generation = this.waitForGenerationTurn(previousGeneration, options.signal)
      .then(() => this.performGenerate(input, options))
    // If this queued call is cancelled early, later work must still remain
    // behind the previous active generation rather than bypassing the queue.
    this.generationTail = Promise.allSettled([previousGeneration, generation]).then(() => undefined)
    return generation
  }

  private async waitForGenerationTurn(
    previousGeneration: Promise<void>,
    signal?: AbortSignal,
  ): Promise<void> {
    if (signal === undefined) {
      await previousGeneration
      return
    }
    throwIfAborted(signal)
    let abortListener: (() => void) | undefined
    const cancellation = new Promise<never>((_resolve, reject) => {
      abortListener = () => {
        reject(new CancelledError('The queued TripoSplat generation was cancelled.', {
          cause: signal.reason,
        }))
      }
      signal.addEventListener('abort', abortListener, { once: true })
    })
    try {
      await Promise.race([previousGeneration, cancellation])
    } finally {
      if (abortListener) signal.removeEventListener('abort', abortListener)
    }
  }

  private async performGenerate(
    input: TripoSplatInput,
    options: GenerateOptions,
  ): Promise<GaussianScene> {
    this.assertUsable()
    throwIfAborted(options.signal)
    await this.load(options.signal === undefined ? {} : { signal: options.signal })
    this.assertUsable()
    const capabilities = this.capabilities
    if (!capabilities.fullGeneration || !this.manifestValue || !this.runtimeValue) {
      throw new GraphCapabilityError('End-to-end TripoSplat generation is unavailable for this configuration.', {
        diagnostics: { capabilities },
      })
    }
    const pipeline = this.options.pipeline ?? runBuiltInTripoSplatPipeline
    try {
      const context: TripoSplatPipelineContext = {
        input,
        options,
        manifest: this.manifestValue,
        runtime: this.runtimeValue,
        sessionIds: SESSION_IDS,
        preprocess: this.options.preprocess ?? normalizeTripoSplatImageInput,
      }
      if (this.options.removeBackground !== undefined) {
        context.removeBackground = this.options.removeBackground
      }
      const scene = await pipeline(context)
      if (this.disposedValue) {
        scene.dispose()
        this.assertUsable()
      }
      return scene
    } catch (error) {
      // Aborting an in-flight ORT WebGPU run terminates the worker because ORT
      // cannot cancel a submitted GPU graph safely. Recreate the runtime lazily
      // on the next load/generate call so cancellation is recoverable without a
      // page reload or a new TripoSplatWebGPU instance.
      if (error instanceof CancelledError) await this.resetAfterCancellation()
      throw error
    }
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise
    this.disposedValue = true
    this.disposePromise = this.performDispose()
    return this.disposePromise
  }

  private async performDispose(): Promise<void> {
    await this.loadPromise?.catch(() => undefined)
    this.manifestValue = undefined
    const runtime = this.runtimeValue
    this.runtimeValue = undefined
    const artifactManager = this.artifactManagerValue
    this.artifactManagerValue = undefined
    this.loadPromise = undefined
    try {
      await runtime?.dispose()
    } finally {
      artifactManager?.dispose()
      // Active work observes the disposed runtime and queued work observes
      // assertUsable(). Do not resolve dispose() while either is still unwinding.
      await this.generationTail.catch(() => undefined)
    }
  }

  private async performLoad(options: LoadOptions): Promise<void> {
    options.onProgress?.({ stage: 'manifest', message: 'Loading the versioned TripoSplat manifest.' })
    const resolvedManifestUrl = manifestUrl(this.options)
    const fetchOptions: Parameters<typeof fetchModelManifest>[1] = {}
    if (this.options.fetch !== undefined) fetchOptions.fetch = this.options.fetch
    if (this.options.manifestRequestInit !== undefined) {
      fetchOptions.requestInit = this.options.manifestRequestInit
    }
    if (options.signal !== undefined) fetchOptions.signal = options.signal
    this.manifestValue = await fetchModelManifest(resolvedManifestUrl, fetchOptions)
    throwIfAborted(options.signal)
    this.assertUsable()

    const providers = this.options.executionProviders ?? ['webgpu']
    if (providers.includes('webgpu')) {
      options.onProgress?.({ stage: 'compatibility', message: 'Checking WebGPU adapter availability.' })
      const compatibilityOptions: CompatibilityOptions = {
        estimatedModelBytes: declaredArtifactBytes(this.manifestValue),
      }
      if (this.manifestValue.estimatedPeakBytes !== undefined) {
        compatibilityOptions.estimatedPeakBytes = this.manifestValue.estimatedPeakBytes
      }
      const compatibility = await checkCompatibility(compatibilityOptions)
      if (!compatibility.supported && !providers.includes('wasm')) {
        throw new WebGPUUnavailableError(compatibility.blockers.join(' '), {
          diagnostics: { compatibility },
        })
      }
    }
    throwIfAborted(options.signal)
    this.assertUsable()

    // OPFS avoids retaining multi-gigabyte external-data chunks in renderer JS
    // memory and is therefore the production default for the canonical model.
    const backend = this.options.cache ?? 'opfs'
    const managerOptions: ConstructorParameters<typeof ModelArtifactManager>[0] = {
      backend,
      namespace: modelCacheNamespace(this.manifestValue),
      onProgress: (progress) => {
        const update: Parameters<NonNullable<LoadOptions['onProgress']>>[0] = {
          stage: 'graphs',
          graph: progress.graph,
          message: progress.source === 'cache'
            ? `Verified cached model artifact '${progress.label}'.`
            : `Downloading and verifying model artifact '${progress.label}'.`,
          loadedBytes: progress.loadedBytes,
        }
        if (progress.totalBytes !== undefined) {
          update.totalBytes = progress.totalBytes
          update.progress = progress.totalBytes === 0
            ? 1
            : Math.min(1, progress.loadedBytes / progress.totalBytes)
        }
        options.onProgress?.(update)
      },
    }
    if (this.options.fetch !== undefined) managerOptions.fetch = this.options.fetch
    if (this.options.artifactRequestInit !== undefined) {
      managerOptions.requestInit = this.options.artifactRequestInit
    }
    const artifactManager = new ModelArtifactManager(managerOptions)
    this.artifactManagerValue = artifactManager
    if (backend !== 'none') {
      // Validate backend availability even for an encoder-only/empty manifest.
      const cacheEntries = await artifactManager.status()
      const cachedBytes = cacheEntries
        .filter((entry) => entry.namespace === modelCacheNamespace(this.manifestValue!))
        .reduce((total, entry) => total + entry.byteLength, 0)
      await preflightPersistentStorage(this.manifestValue, backend, cachedBytes, options.onProgress)
      await artifactManager.prefetchManifest(this.manifestValue, options.signal)
      throwIfAborted(options.signal)
      this.assertUsable()
    }

    options.onProgress?.({ stage: 'runtime', message: 'Starting the browser-local ONNX worker.' })
    const runtimeOptions: Parameters<typeof createRuntime>[0] = {
      executionProviders: providers,
      onStatus: (status: RuntimeStatus) => this.logRuntimeStatus(status),
      configuration: this.options.wasmPaths === undefined ? {} : { wasmPaths: this.options.wasmPaths },
    }
    if (this.options.workerUrl !== undefined) runtimeOptions.workerUrl = this.options.workerUrl
    if (this.options.workerFactory !== undefined) runtimeOptions.workerFactory = this.options.workerFactory
    runtimeOptions.baseUrl = directoryUrl(this.options.modelBaseUrl)
    this.runtimeValue = withVerifiedModelArtifacts(
      createRuntime(runtimeOptions),
      artifactManager,
      GRAPH_FOR_SESSION,
    )

    // Graphs are intentionally loaded during generate(), one stage at a time,
    // so encoder, DiT, and decoder weights are not resident together.
    options.onProgress?.({
      stage: 'complete',
      message: 'TripoSplat manifest and browser-local runtime are ready.',
      progress: 1,
    })
  }

  private logRuntimeStatus(status: RuntimeStatus): void {
    if (this.options.logLevel === 'debug') console.debug('[TripoSplatWebGPU]', status)
    else if (this.options.logLevel === 'info') console.info('[TripoSplatWebGPU]', status.message)
  }

  private async resetAfterCancellation(): Promise<void> {
    const runtime = this.runtimeValue
    const artifactManager = this.artifactManagerValue
    this.runtimeValue = undefined
    this.artifactManagerValue = undefined
    this.manifestValue = undefined
    this.loadPromise = undefined
    artifactManager?.dispose()
    await runtime?.dispose().catch(() => undefined)
  }

  private assertUsable(): void {
    if (this.disposedValue) {
      throw new TripoSplatError('TripoSplatWebGPU has been disposed.', {
        code: 'DISPOSED', stage: 'dispose', recoverable: false,
      })
    }
  }
}
