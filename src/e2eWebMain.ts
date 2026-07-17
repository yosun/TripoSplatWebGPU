import { createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import {
  createImagePreparer,
  type ImagePreparer,
  type PreparationProgress,
} from 'reconstruction-image-prep'

import {
  BackgroundRemovalRequiredError,
  CancelledError,
  TripoSplatError,
  TripoSplatWebGPU,
  clearModelCache,
  getModelCacheStatus,
  type CacheBackend,
  type CompatibilityReport,
  type GenerationProgress,
  type LoadProgress,
} from '../packages/triposplat-webgpu/dist/index.js'
import { SplatPreview, type SplatPreviewStatus } from './components/SplatPreview'

const MODEL_BYTES = 6_465_182_402
const MODEL_LARGEST_ARTIFACT_BYTES = 3_362_042_880
const DEFAULT_MODEL_BASE = 'https://huggingface.co/Yosun/TripoSplat-WebGPU/resolve/main/triposplat-webgpu/0.1.0-fp32.20260715/'
const DEFAULT_STEPS = 20
const ACTIVE_RUN_STORAGE_KEY = 'triposplat.active-run.v1'

type FlowStage = 'source' | 'model' | 'conditioning' | 'sampling' | 'decode' | 'preview'
type ProgressDetailMode = 'guided' | 'technical'

interface RunStatusSnapshot {
  stage: string
  message: string
  progress?: number
}

interface SelectedImage {
  blob: Blob
  sourceBlob: Blob
  name: string
  previewUrl: string
  width: number
  height: number
  hasAlpha: boolean
  inputIsPrepared: boolean
}

interface ImageSelectionOptions {
  sourceBlob?: Blob
  previewBlob?: Blob
  inputIsPrepared?: boolean
  summary?: string
}

interface BrowserMemorySnapshot {
  source: 'performance.memory' | 'unavailable'
  usedJsHeapBytes?: number
  totalJsHeapBytes?: number
  jsHeapLimitBytes?: number
}

interface RunTelemetry {
  startedAt: string
  startedAtMs: number
  activeFlowStage?: FlowStage
  flowStageStartedAtMs: number
  flowDurationsMs: Partial<Record<FlowStage, number>>
  memoryBefore: BrowserMemorySnapshot
  memoryAfter?: BrowserMemorySnapshot
  peakUsedJsHeapBytes?: number
  cachedBytesBefore: number
}

interface PreviewFrame {
  position: [number, number, number]
  target: [number, number, number]
}

interface ModelManifestSummary {
  namespace: string
  declaredBytes: number
  largestArtifactBytes: number
}

interface StorageQualification {
  supported: boolean
  state: 'ready' | 'warning' | 'problem'
  message: string
  quotaBytes?: number
  usageBytes?: number
  availableBytes?: number
  cachedBytes?: number
  requiredBytes?: number
}

interface ActiveRunMarker {
  version: 1
  runId: string
  startedAt: number
  updatedAt: number
  phase: string
  cacheBackend: CacheBackend
  declaredModelBytes: number
}

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`Required public runner element '${selector}' was not found.`)
  return element
}

const fileInput = requiredElement<HTMLInputElement>('#image-file')
const chooseFileButton = requiredElement<HTMLButtonElement>('#choose-file')
const imageUrlInput = requiredElement<HTMLInputElement>('#image-url')
const loadImageUrlButton = requiredElement<HTMLButtonElement>('#load-image-url')
const imageUrlError = requiredElement<HTMLElement>('#image-url-error')
const imageSummary = requiredElement<HTMLElement>('#image-summary')
const imagePreview = requiredElement<HTMLElement>('#image-preview')
const imagePreviewImage = requiredElement<HTMLImageElement>('#image-preview-image')
const prepareImageButton = requiredElement<HTMLButtonElement>('#prepare-image')
const imagePreparationStatus = requiredElement<HTMLElement>('#image-preparation-status')
const dropZone = requiredElement<HTMLElement>('#drop-zone')
const sourcePanel = requiredElement<HTMLElement>('.web-controls')
const modelBaseInput = requiredElement<HTMLInputElement>('#model-base')
const modelStatus = requiredElement<HTMLElement>('#model-status')
const cacheMode = requiredElement<HTMLElement>('#cache-mode')
const generateButton = requiredElement<HTMLButtonElement>('#generate')
const cancelButton = requiredElement<HTMLButtonElement>('#cancel')
const clearCacheButton = requiredElement<HTMLButtonElement>('#clear-cache')
const runStage = requiredElement<HTMLElement>('#run-stage')
const runStatus = requiredElement<HTMLElement>('#run-status')
const runAnnouncement = requiredElement<HTMLElement>('#run-announcement')
const progressTrack = requiredElement<HTMLElement>('.progress-track')
const progressFill = requiredElement<HTMLElement>('#progress-fill')
const progressDetailButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-progress-detail]'))
const progressDetailDescription = requiredElement<HTMLElement>('#progress-detail-description')
const generationFlow = requiredElement<HTMLOListElement>('#generation-flow')
const benchmarkReport = requiredElement<HTMLElement>('#benchmark-report')
const benchmarkSummary = requiredElement<HTMLElement>('#benchmark-summary')
const benchmarkDetails = requiredElement<HTMLElement>('#benchmark-details')
const copyBenchmarkButton = requiredElement<HTMLButtonElement>('#copy-benchmark')
const diagnostics = requiredElement<HTMLElement>('#diagnostics')
const diagnosticMessage = requiredElement<HTMLElement>('#diagnostic-message')
const diagnosticDetails = requiredElement<HTMLElement>('#diagnostic-details')
const platformBadge = requiredElement<HTMLElement>('#platform-badge')
const compatibilityList = requiredElement<HTMLUListElement>('#compatibility-list')
const previewMount = requiredElement<HTMLElement>('#web-preview-root')
const viewerState = requiredElement<HTMLElement>('#viewer-state')
const previewRunState = requiredElement<HTMLElement>('#preview-run-state')
const downloadPlyButton = requiredElement<HTMLButtonElement>('#download-ply')
const downloadSplatButton = requiredElement<HTMLButtonElement>('#download-splat')

const previewRoot: Root = createRoot(previewMount)
const flowOrder: readonly FlowStage[] = ['source', 'model', 'conditioning', 'sampling', 'decode', 'preview']
const flowItems = Array.from(generationFlow.querySelectorAll<HTMLElement>('[data-flow-stage]'))
let compatibility: CompatibilityReport | undefined
let cacheBackend: CacheBackend = 'none'
let activeManifestSummary: ModelManifestSummary = {
  namespace: 'triposplat-webgpu/0.1.0-fp32.20260715/a78fa12d06dbf1381ca548bfac32bb68cb8c451d/fp32',
  declaredBytes: MODEL_BYTES,
  largestArtifactBytes: MODEL_LARGEST_ARTIFACT_BYTES,
}
let storageQualification: StorageQualification | undefined
let platformRunBlocker: string | undefined
let activeRunMarker: ActiveRunMarker | undefined
let lastRunMarkerPhase = ''
let pageIsHiding = false
const interruptedRunNotice = consumeInterruptedRunMarker()
let selectedImage: SelectedImage | undefined
let imagePreparer: ImagePreparer | undefined
let model: TripoSplatWebGPU | undefined
let loadedModelBase: string | undefined
let controller: AbortController | undefined
let busy = false
let activePlyUrl: string | undefined
let downloadablePly: Blob | undefined
let downloadableSplat: Blob | undefined
let previewFrame: PreviewFrame | undefined
let completionChimeContext: AudioContext | undefined
let completionChimeArmed = false
let previewGenerationKey = 0
let activeRunTelemetry: RunTelemetry | undefined
let benchmarkReportText = ''
let lastAnnouncedRunStage = ''
let progressDetailMode: ProgressDetailMode = readProgressDetailMode()
let latestRunStatus: RunStatusSnapshot = {
  stage: 'STATUS',
  message: 'Checking browser compatibility…',
}
const retiredPlyUrls = new Set<string>()

function formatBytes(bytes: number): string {
  if (bytes < 1_024) return `${bytes} B`
  if (bytes < 1_024 ** 2) return `${(bytes / 1_024).toFixed(1)} KiB`
  if (bytes < 1_024 ** 3) return `${(bytes / 1_024 ** 2).toFixed(1)} MiB`
  return `${(bytes / 1_024 ** 3).toFixed(2)} GiB`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isMobilePlatform(): boolean {
  const userAgent = navigator.userAgent
  const iPadDesktopMode = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1
  return /iPhone|iPad|iPod|Android.*Mobile|Mobile.*Android/i.test(userAgent) || iPadDesktopMode
}

function summarizeModelManifest(value: unknown): ModelManifestSummary {
  if (!isRecord(value) || !isRecord(value.graphs)) throw new Error('The model manifest is missing its graph declarations.')
  const identity = [value.name, value.version, value.modelRevision, value.precision]
  if (!identity.every((part) => typeof part === 'string' && part.length > 0)) {
    throw new Error('The model manifest is missing its cache identity fields.')
  }
  const requiredGraphs = ['dino', 'vae', 'dit', 'octree', 'gaussianDecoder'] as const
  let summedBytes = 0
  let largestArtifactBytes = 0
  for (const graphName of requiredGraphs) {
    const graph = value.graphs[graphName]
    if (!isRecord(graph)) throw new Error(`The model manifest is missing the required '${graphName}' graph.`)
    const lengths: unknown[] = [graph.byteLength]
    if (graph.externalData !== undefined && !Array.isArray(graph.externalData)) {
      throw new Error(`The '${graphName}' graph has an invalid external-data declaration.`)
    }
    for (const external of graph.externalData ?? []) {
      if (!isRecord(external)) throw new Error(`The '${graphName}' graph contains an invalid external-data artifact.`)
      lengths.push(external.byteLength)
    }
    for (const length of lengths) {
      if (!Number.isSafeInteger(length) || (length as number) <= 0) {
        throw new Error(`Every '${graphName}' artifact must declare a positive byte size for browser storage checks.`)
      }
      summedBytes += length as number
      largestArtifactBytes = Math.max(largestArtifactBytes, length as number)
    }
  }
  const estimatedBytes = value.estimatedModelBytes
  if (estimatedBytes !== undefined && (!Number.isSafeInteger(estimatedBytes) || (estimatedBytes as number) <= 0)) {
    throw new Error('The model manifest contains an invalid estimated model size.')
  }
  const declaredBytes = Math.max(summedBytes, typeof estimatedBytes === 'number' ? estimatedBytes : 0)
  return {
    namespace: identity.join('/'),
    declaredBytes,
    largestArtifactBytes,
  }
}

function consumeInterruptedRunMarker(): string | undefined {
  try {
    const raw = sessionStorage.getItem(ACTIVE_RUN_STORAGE_KEY)
    sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY)
    if (!raw) return undefined
    const marker = JSON.parse(raw) as Partial<ActiveRunMarker>
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined
    if (
      marker.version !== 1
      || typeof marker.updatedAt !== 'number'
      || typeof marker.phase !== 'string'
      || Date.now() - marker.updatedAt > 6 * 60 * 60 * 1_000
      || navigation?.type !== 'reload'
    ) return undefined
    const likelyMemory = isMobilePlatform() && /graph|generation|export/i.test(marker.phase)
    return likelyMemory
      ? `A previous run ended during ${marker.phase} when this tab reloaded. If you did not refresh it, the mobile browser probably terminated the page under memory pressure.`
      : `A previous run ended during ${marker.phase} when this tab reloaded. If you did not refresh it, browser memory or storage pressure may have interrupted the run.`
  } catch {
    return undefined
  }
}

function updateActiveRunMarker(phase: string): void {
  if (!activeRunMarker || phase === lastRunMarkerPhase) return
  activeRunMarker.phase = phase
  activeRunMarker.updatedAt = Date.now()
  lastRunMarkerPhase = phase
  try {
    sessionStorage.setItem(ACTIVE_RUN_STORAGE_KEY, JSON.stringify(activeRunMarker))
  } catch {
    // Interruption diagnostics are best-effort only.
  }
}

function startActiveRunMarker(): void {
  const now = Date.now()
  activeRunMarker = {
    version: 1,
    runId: typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `${now}-${Math.random()}`,
    startedAt: now,
    updatedAt: now,
    phase: 'preflight',
    cacheBackend,
    declaredModelBytes: activeManifestSummary.declaredBytes,
  }
  lastRunMarkerPhase = ''
  updateActiveRunMarker('preflight')
}

function clearActiveRunMarker(): void {
  activeRunMarker = undefined
  lastRunMarkerPhase = ''
  if (pageIsHiding) return
  try {
    sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY)
  } catch {
    // Interruption diagnostics are best-effort only.
  }
}

async function qualifyStorage(summary: ModelManifestSummary): Promise<StorageQualification> {
  if (cacheBackend === 'none') {
    return {
      supported: false,
      state: 'problem',
      message: 'No disk-backed model cache is available. This 6.5 GB fp32 build cannot safely stage its largest graph in page memory.',
    }
  }
  const status = await getModelCacheStatus({ backend: cacheBackend, namespace: summary.namespace })
  const backendStatus = status.backends[0]
  if (!backendStatus?.available) {
    return {
      supported: false,
      state: 'problem',
      message: `${cacheBackend.toUpperCase()} storage is unavailable${backendStatus?.error ? `: ${backendStatus.error}` : '.'}`,
    }
  }
  const estimate = await navigator.storage?.estimate?.().catch(() => undefined)
  if (!Number.isFinite(estimate?.quota) || !Number.isFinite(estimate?.usage)) {
    return {
      supported: true,
      state: 'warning',
      message: `Browser storage capacity could not be measured. The ${formatBytes(summary.declaredBytes)} model may still fail if this origin has a restricted quota.`,
      cachedBytes: backendStatus.totalBytes,
    }
  }
  const quotaBytes = estimate!.quota as number
  const usageBytes = estimate!.usage as number
  const availableBytes = Math.max(0, quotaBytes - usageBytes)
  const missingBytes = Math.max(0, summary.declaredBytes - backendStatus.totalBytes)
  const requiredBytes = missingBytes + (cacheBackend === 'cache-api' ? summary.largestArtifactBytes : 0)
  if (requiredBytes > availableBytes) {
    return {
      supported: false,
      state: 'problem',
      message: `Not enough browser storage: ${formatBytes(requiredBytes)} is required, but this browser allows ${formatBytes(availableBytes)} free for this site (${formatBytes(quotaBytes)} total quota).`,
      quotaBytes,
      usageBytes,
      availableBytes,
      cachedBytes: backendStatus.totalBytes,
      requiredBytes,
    }
  }
  const margin = Math.max(512 * 1_024 ** 2, requiredBytes * 0.1)
  return {
    supported: true,
    state: availableBytes - requiredBytes < margin ? 'warning' : 'ready',
    message: `${formatBytes(availableBytes)} browser storage available; ${formatBytes(requiredBytes)} still required for the verified ${cacheBackend.toUpperCase()} model cache.`,
    quotaBytes,
    usageBytes,
    availableBytes,
    cachedBytes: backendStatus.totalBytes,
    requiredBytes,
  }
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1_000) return `${Math.round(milliseconds)} ms`
  const seconds = milliseconds / 1_000
  if (seconds < 60) return `${seconds.toFixed(2)} s`
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(1)}s`
}

function armCompletionChime(): void {
  completionChimeArmed = true
  if (!completionChimeContext) {
    try {
      completionChimeContext = new AudioContext()
    } catch {
      return
    }
  }
  void completionChimeContext.resume().catch(() => undefined)
}

function playCompletionChime(): void {
  if (!completionChimeArmed) return
  completionChimeArmed = false
  const context = completionChimeContext
  if (!context || context.state !== 'running') return

  // A gentle, original three-note appliance-completion chime synthesized in
  // the browser: no audio asset is downloaded or imitated from a specific device.
  const startedAt = context.currentTime + 0.03
  for (const [index, frequency] of [783.99, 1046.5, 1318.51].entries()) {
    const oscillator = context.createOscillator()
    const gain = context.createGain()
    const noteAt = startedAt + index * 0.16
    oscillator.type = 'sine'
    oscillator.frequency.setValueAtTime(frequency, noteAt)
    gain.gain.setValueAtTime(0.0001, noteAt)
    gain.gain.exponentialRampToValueAtTime(0.09, noteAt + 0.015)
    gain.gain.exponentialRampToValueAtTime(0.0001, noteAt + 0.25)
    oscillator.connect(gain).connect(context.destination)
    oscillator.start(noteAt)
    oscillator.stop(noteAt + 0.27)
  }
}

function fitPreviewFrame(positions: Float32Array): PreviewFrame | undefined {
  if (positions.length < 3) return undefined
  let minX = Infinity
  let minY = Infinity
  let minZ = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  let maxZ = -Infinity
  for (let index = 0; index + 2 < positions.length; index += 3) {
    const x = positions[index]
    const y = positions[index + 1]
    const z = positions[index + 2]
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    minZ = Math.min(minZ, z)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
    maxZ = Math.max(maxZ, z)
  }
  if (![minX, minY, minZ, maxX, maxY, maxZ].every(Number.isFinite)) return undefined

  const center: [number, number, number] = [
    (minX + maxX) / 2,
    -((minY + maxY) / 2),
    -((minZ + maxZ) / 2),
  ]
  const longestSide = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.1)
  const distance = Math.max(longestSide / (2 * Math.tan(Math.PI / 6)) * 1.35, 0.35)
  return {
    position: [center[0], center[1], center[2] + distance],
    target: center,
  }
}

function compactStatus(message: string): string {
  return message.length > 72 ? `${message.slice(0, 69)}…` : message
}

type PerformanceWithMemory = Performance & {
  memory?: {
    usedJSHeapSize: number
    totalJSHeapSize: number
    jsHeapSizeLimit: number
  }
}

// Chromium exposes `performance.memory` unprefixed and without requiring
// cross-origin isolation. It only reports the main-realm JS heap: it excludes
// the ONNX Runtime worker heap, WebGPU buffers, and native/Metal driver
// residency, so it is a floor on actual usage, not a total. Non-Chromium
// browsers do not expose any comparable API, so the report says so plainly
// instead of guessing.
function measureBrowserMemory(): BrowserMemorySnapshot {
  const memory = (performance as PerformanceWithMemory).memory
  if (!memory) return { source: 'unavailable' }
  return {
    source: 'performance.memory',
    usedJsHeapBytes: memory.usedJSHeapSize,
    totalJsHeapBytes: memory.totalJSHeapSize,
    jsHeapLimitBytes: memory.jsHeapSizeLimit,
  }
}

function sampleMemoryPeak(): void {
  if (!activeRunTelemetry) return
  const sample = measureBrowserMemory()
  if (sample.usedJsHeapBytes === undefined) return
  activeRunTelemetry.peakUsedJsHeapBytes = Math.max(
    activeRunTelemetry.peakUsedJsHeapBytes ?? 0,
    sample.usedJsHeapBytes,
  )
}

function startRunTelemetry(cachedBytesBefore: number): void {
  const startedAtMs = performance.now()
  const memoryBefore = measureBrowserMemory()
  activeRunTelemetry = {
    startedAt: new Date().toISOString(),
    startedAtMs,
    flowStageStartedAtMs: startedAtMs,
    flowDurationsMs: {},
    memoryBefore,
    peakUsedJsHeapBytes: memoryBefore.usedJsHeapBytes,
    cachedBytesBefore,
  }
}

function recordFlowStage(flowStage: FlowStage | undefined): void {
  sampleMemoryPeak()
  if (!activeRunTelemetry || !flowStage || activeRunTelemetry.activeFlowStage === flowStage) return
  const now = performance.now()
  if (activeRunTelemetry.activeFlowStage) {
    const previous = activeRunTelemetry.activeFlowStage
    activeRunTelemetry.flowDurationsMs[previous] = (activeRunTelemetry.flowDurationsMs[previous] ?? 0)
      + now - activeRunTelemetry.flowStageStartedAtMs
  }
  activeRunTelemetry.activeFlowStage = flowStage
  activeRunTelemetry.flowStageStartedAtMs = now
}

function finishRunTelemetry(): RunTelemetry | undefined {
  if (!activeRunTelemetry) return undefined
  const completed = activeRunTelemetry
  if (completed.activeFlowStage) {
    completed.flowDurationsMs[completed.activeFlowStage] = (completed.flowDurationsMs[completed.activeFlowStage] ?? 0)
      + performance.now() - completed.flowStageStartedAtMs
  }
  completed.memoryAfter = measureBrowserMemory()
  if (completed.memoryAfter.usedJsHeapBytes !== undefined) {
    completed.peakUsedJsHeapBytes = Math.max(completed.peakUsedJsHeapBytes ?? 0, completed.memoryAfter.usedJsHeapBytes)
  }
  activeRunTelemetry = undefined
  return completed
}

function numericTimingEntries(value: unknown): Array<[string, number]> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return []
  return Object.entries(value).flatMap(([name, timing]) => typeof timing === 'number' ? [[name, timing]] : [])
}

function buildBenchmarkReport(
  telemetry: RunTelemetry,
  scene: { count: number; metadata: { generationSettings: Readonly<Record<string, unknown>>; seed: number; modelRevision: string } },
  base: string,
  cachedBytesAfter: number,
): string {
  const totalDuration = performance.now() - telemetry.startedAtMs
  const settings = scene.metadata.generationSettings
  const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory
  const lines = [
    'TripoSplat WebGPU benchmark',
    `Completed (UTC): ${new Date().toISOString()}`,
    `Started (UTC): ${telemetry.startedAt}`,
    `End-to-end duration: ${formatDuration(totalDuration)}`,
    '',
    'Environment',
    `Browser: ${compatibility?.browser ?? navigator.userAgent}`,
    `WebGPU adapter: ${compatibility?.adapterName ?? 'not exposed by this browser'}`,
    `Logical CPU cores: ${navigator.hardwareConcurrency || 'not exposed'}`,
    `Device memory hint: ${deviceMemory ? `${deviceMemory} GiB` : 'not exposed'}`,
    `Cache backend: ${cacheBackend}`,
    `Model revision: ${scene.metadata.modelRevision}`,
    `Model base: ${base}`,
    '',
    'Input and output',
    `Image: ${selectedImage?.name ?? 'unknown'} · ${selectedImage ? `${selectedImage.width}×${selectedImage.height}` : 'unknown size'} · ${selectedImage?.hasAlpha ? 'alpha' : 'opaque'}`,
    `Gaussians: ${scene.count.toLocaleString()}`,
    `Steps: ${String(settings.steps ?? DEFAULT_STEPS)} · seed ${scene.metadata.seed} · precision ${String(settings.precision ?? 'unknown')}`,
    '',
    'Pipeline wall time',
    ...flowOrder.flatMap((stage) => {
      const duration = telemetry.flowDurationsMs[stage]
      return duration === undefined ? [] : [`${stage}: ${formatDuration(duration)}`]
    }),
    '',
    'Memory used',
    ...formatMemorySection(telemetry),
    '',
    'Browser storage used',
    `Model cache before run: ${formatBytes(telemetry.cachedBytesBefore)}`,
    `Model cache after run: ${formatBytes(cachedBytesAfter)}`,
    `Downloaded and verified this run: ${formatBytes(Math.max(0, cachedBytesAfter - telemetry.cachedBytesBefore))}`,
  ]
  const runtimeTimings = numericTimingEntries(settings.measuredTimingsMs)
  if (runtimeTimings.length) {
    lines.push('', 'Runtime timings reported by the model')
    lines.push(...runtimeTimings.map(([name, milliseconds]) => `${name}: ${formatDuration(milliseconds)}`))
  }
  const limits = compatibility ? Object.entries(compatibility.limits) : []
  if (limits.length) {
    lines.push('', 'Selected WebGPU limits')
    lines.push(...limits.map(([name, value]) => `${name}: ${value.toLocaleString()}`))
  }
  return lines.join('\n')
}

function formatMemorySection(telemetry: RunTelemetry): string[] {
  const before = telemetry.memoryBefore
  const after = telemetry.memoryAfter
  if (before.source === 'unavailable' || !after || after.source === 'unavailable') {
    return [
      'This browser does not expose performance.memory (Chromium-only, main-realm JS heap).',
      'Peak memory could not be measured. This does not mean memory usage was low.',
    ]
  }
  const usedBefore = before.usedJsHeapBytes ?? 0
  const usedAfter = after.usedJsHeapBytes ?? 0
  const peak = telemetry.peakUsedJsHeapBytes ?? usedAfter
  const lines = [
    `Main-realm JS heap before run: ${formatBytes(usedBefore)}`,
    `Main-realm JS heap after run: ${formatBytes(usedAfter)}`,
    `Peak main-realm JS heap sampled during run: ${formatBytes(peak)}`,
    `Net main-realm JS heap growth: ${formatBytes(Math.max(0, usedAfter - usedBefore))}`,
  ]
  if (after.jsHeapLimitBytes !== undefined) lines.push(`Main-realm JS heap limit: ${formatBytes(after.jsHeapLimitBytes)}`)
  lines.push(
    'This is the main-realm JS heap only. It excludes the ONNX Runtime worker heap, WebGPU buffers, WASM linear memory, and native/Metal driver residency, so actual peak memory is higher than this number.',
  )
  return lines
}

function showBenchmarkReport(report: string, totalDuration: number): void {
  benchmarkReportText = report
  benchmarkReport.hidden = false
  benchmarkSummary.textContent = `Completed in ${formatDuration(totalDuration)}. Copy this report when sharing a benchmark result.`
  benchmarkDetails.textContent = report
  copyBenchmarkButton.disabled = false
  copyBenchmarkButton.textContent = 'Copy report'
}

async function copyBenchmarkReport(): Promise<void> {
  if (!benchmarkReportText) return
  try {
    await navigator.clipboard.writeText(benchmarkReportText)
  } catch {
    const textarea = document.createElement('textarea')
    textarea.value = benchmarkReportText
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    const copied = document.execCommand('copy')
    textarea.remove()
    if (!copied) throw new Error('The browser denied clipboard access.')
  }
  copyBenchmarkButton.textContent = 'Copied'
  benchmarkSummary.textContent = 'Benchmark report copied to your clipboard.'
}

function chooseCacheBackend(): CacheBackend {
  const storage = navigator.storage as StorageManager & { getDirectory?: unknown }
  if (typeof storage.getDirectory === 'function') return 'opfs'
  if ('caches' in window) return 'cache-api'
  return 'none'
}

function normalizedModelBase(value: string): string {
  const url = new URL(value.trim())
  if (url.protocol !== 'https:' && !['localhost', '127.0.0.1'].includes(url.hostname)) {
    throw new Error('Use an HTTPS model-server URL. Local HTTP is only supported during development.')
  }
  if (url.pathname.endsWith('/manifest.json')) return new URL('.', url).href
  url.pathname = url.pathname.endsWith('/') ? url.pathname : `${url.pathname}/`
  return url.href
}

function stageToFlowStage(stage: string): FlowStage | undefined {
  const normalized = stage.toLowerCase()
  if (normalized.includes('image') || normalized.includes('source')) return 'source'
  if (normalized.includes('model') || normalized.includes('manifest') || normalized.includes('runtime') || normalized.includes('graph')) return 'model'
  if (normalized.includes('preprocess') || normalized.includes('dino') || normalized.includes('vae')) return 'conditioning'
  if (normalized.includes('sampling')) return 'sampling'
  if (normalized.includes('octree') || normalized.includes('gaussian')) return 'decode'
  if (normalized.includes('packing') || normalized.includes('export') || normalized.includes('complete')) return 'preview'
  return undefined
}

const guidedStageCopy: Record<FlowStage, { stage: string; message: string; detail: string }> = {
  source: {
    stage: 'SOURCE IMAGE',
    message: 'Reading your image and preparing it for local generation.',
    detail: 'Preparing the image',
  },
  model: {
    stage: 'PREPARING THE ENGINE',
    message: 'Checking, downloading, and loading the model components this browser needs.',
    detail: 'Getting the AI model ready',
  },
  conditioning: {
    stage: 'UNDERSTANDING THE IMAGE',
    message: 'Turning the image into visual features the 3D generator can work with.',
    detail: 'Learning the image structure',
  },
  sampling: {
    stage: 'SHAPING THE 3D SCENE',
    message: 'Iteratively refining a hidden spatial representation of the object.',
    detail: 'Refining the spatial structure',
  },
  decode: {
    stage: 'BUILDING THE GAUSSIANS',
    message: 'Converting the spatial representation into visible 3D Gaussian points.',
    detail: 'Creating visible 3D points',
  },
  preview: {
    stage: 'PREPARING YOUR RESULT',
    message: 'Packaging the completed scene for the viewer and downloads.',
    detail: 'Preparing the interactive result',
  },
}

function readProgressDetailMode(): ProgressDetailMode {
  try {
    return localStorage.getItem('triposplat-progress-detail') === 'guided' ? 'guided' : 'technical'
  } catch {
    return 'technical'
  }
}

function statusForDetailMode(snapshot: RunStatusSnapshot): { stage: string; message: string } {
  if (progressDetailMode === 'technical') return snapshot
  const flowStage = stageToFlowStage(snapshot.stage)
  if (!flowStage) return snapshot
  const guided = guidedStageCopy[flowStage]
  const percentage = snapshot.progress === undefined ? '' : ` ${Math.round(snapshot.progress * 100)}% through this stage.`
  return { stage: guided.stage, message: `${guided.message}${percentage}` }
}

function setFlowStage(activeStage: FlowStage, detail?: string): void {
  const activeIndex = flowOrder.indexOf(activeStage)
  for (const item of flowItems) {
    const itemStage = item.dataset.flowStage as FlowStage | undefined
    const itemIndex = itemStage ? flowOrder.indexOf(itemStage) : -1
    const state = itemIndex < activeIndex ? 'complete' : itemIndex === activeIndex ? 'active' : 'waiting'
    item.dataset.state = state
    const status = item.querySelector('small')
    const visibleDetail = progressDetailMode === 'technical'
      ? compactStatus(detail ?? 'Working locally')
      : guidedStageCopy[activeStage].detail
    if (status) status.textContent = state === 'active' ? `RUNNING · ${visibleDetail}` : state === 'complete' ? 'COMPLETE' : 'WAITING'
    if (state === 'active') item.setAttribute('aria-current', 'step')
    else item.removeAttribute('aria-current')
  }
}

function renderRunStatus(announceStageChange = true): void {
  const visible = statusForDetailMode(latestRunStatus)
  runStage.textContent = visible.stage
  runStatus.textContent = visible.message
  if (announceStageChange && visible.stage !== lastAnnouncedRunStage) {
    runAnnouncement.textContent = `${visible.stage}. ${visible.message}`
    lastAnnouncedRunStage = visible.stage
  }
  const percent = latestRunStatus.progress === undefined ? 0 : Math.max(0, Math.min(100, latestRunStatus.progress * 100))
  progressFill.style.width = `${percent.toFixed(1)}%`
  progressTrack.setAttribute('aria-valuenow', String(Math.round(percent)))
  progressTrack.setAttribute('aria-valuetext', `${visible.stage}: ${visible.message}`)
}

function applyProgressDetailMode(mode: ProgressDetailMode, announce = true): void {
  progressDetailMode = mode
  for (const button of progressDetailButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.progressDetail === mode))
  }
  progressDetailDescription.textContent = mode === 'technical'
    ? 'Technical mode shows exact graph stages, sampler steps, CFG invocations, and decode boundaries as they happen.'
    : 'Guided mode translates the same pipeline into plain-language milestones while keeping the six-stage process visible.'
  try {
    localStorage.setItem('triposplat-progress-detail', mode)
  } catch {
    // A blocked storage preference should not affect generation.
  }
  lastAnnouncedRunStage = ''
  renderRunStatus(false)
  const activeStage = stageToFlowStage(latestRunStatus.stage)
  if (activeStage) setFlowStage(activeStage, latestRunStatus.message)
  if (announce) runAnnouncement.textContent = `${mode === 'technical' ? 'Technical' : 'Guided'} progress mode selected.`
}

function setPreviewRunState(state: 'waiting' | 'working' | 'ready' | 'retained' | 'failed', message: string): void {
  previewRunState.dataset.state = state
  previewRunState.textContent = message
}

function setRunStatus(stage: string, message: string, progress?: number): void {
  latestRunStatus = progress === undefined ? { stage, message } : { stage, message, progress }
  const flowStage = stageToFlowStage(stage)
  if (flowStage) {
    recordFlowStage(flowStage)
    setFlowStage(flowStage, message)
  }
  renderRunStatus()
}

function setBusy(next: boolean): void {
  busy = next
  cancelButton.disabled = !next
  chooseFileButton.disabled = next
  loadImageUrlButton.disabled = next
  modelBaseInput.disabled = next
  imageUrlInput.disabled = next
  clearCacheButton.disabled = next
  updateGenerateButton()
  updatePrepareImageButton()
}

function updateGenerateButton(): void {
  const compatible = compatibility?.supported === true
  const hasModelBase = modelBaseInput.value.trim().length > 0
  const storageReady = storageQualification?.supported === true
  generateButton.disabled = busy || !compatible || !storageReady || platformRunBlocker !== undefined || !selectedImage || !hasModelBase
  if (busy) {
    generateButton.classList.add('is-working')
    generateButton.firstElementChild!.textContent = 'Working in your browser…'
  } else {
    generateButton.classList.remove('is-working')
    generateButton.firstElementChild!.textContent = platformRunBlocker
      ? 'Desktop browser required'
      : storageQualification?.supported === false
        ? 'Browser storage is insufficient'
        : selectedImage && hasModelBase
          ? 'Generate spatial scene'
          : 'Choose an image to begin'
  }
}

function updatePrepareImageButton(): void {
  prepareImageButton.disabled = busy || !selectedImage
}

function hideDiagnostics(): void {
  diagnostics.hidden = true
  diagnosticMessage.textContent = ''
  diagnosticDetails.textContent = ''
}

function showDiagnostics(message: string, detail?: unknown): void {
  diagnosticMessage.textContent = message
  diagnosticDetails.textContent = typeof detail === 'string'
    ? detail
    : JSON.stringify(detail ?? {}, null, 2)
  diagnostics.hidden = false
}

function friendlyError(error: unknown): { message: string; details: unknown } {
  if (error instanceof BackgroundRemovalRequiredError) {
    return {
      message: 'This image has no transparency. Click “Process image locally” to remove its background and create the required TripoSplat input.',
      details: { code: error.code, stage: error.stage, diagnostics: error.diagnostics },
    }
  }
  if (error instanceof CancelledError || (error instanceof DOMException && error.name === 'AbortError')) {
    return { message: 'Cancelled. A future run starts with a clean browser worker.', details: error.message }
  }
  if (error instanceof TripoSplatError) {
    if (
      error.code === 'MODEL_DOWNLOAD_FAILED'
      && typeof error.diagnostics.requiredBytes === 'number'
      && typeof error.diagnostics.availableBytes === 'number'
    ) {
      const backend = typeof error.diagnostics.backend === 'string' ? error.diagnostics.backend.toUpperCase() : 'browser'
      const quota = typeof error.diagnostics.quotaBytes === 'number'
        ? ` Its total quota for this site is ${formatBytes(error.diagnostics.quotaBytes)}.`
        : ''
      return {
        message: `Not enough ${backend} storage for the model. ${formatBytes(error.diagnostics.requiredBytes)} is required, but only ${formatBytes(error.diagnostics.availableBytes)} is available.${quota} Free browser storage, clear cached model files, or use a browser profile with a larger site quota.`,
        details: { code: error.code, stage: error.stage, diagnostics: error.diagnostics },
      }
    }
    const help: Record<string, string> = {
      WEBGPU_UNAVAILABLE: 'WebGPU is unavailable. Use a current desktop Chrome or Edge browser with hardware acceleration enabled.',
      UNSUPPORTED_ADAPTER: 'Your GPU/browser combination does not meet this model’s current WebGPU requirements.',
      MODEL_DOWNLOAD_FAILED: 'The model server could not be read. Check the URL, CORS response headers, redirects, and network connection.',
      MODEL_INTEGRITY_FAILED: 'A downloaded model file did not match its manifest. Clear the cache and ask the model host to verify the immutable artifacts.',
      MANIFEST_INVALID: 'The model server manifest is missing or is not a valid TripoSplat manifest.',
      GRAPH_LOAD_FAILED: 'The downloaded model could not be initialized by ONNX Runtime WebGPU. Try a supported Chrome or Edge version and verify available device memory.',
      GRAPH_CAPABILITY_UNAVAILABLE: 'This manifest does not contain all five graphs needed for generation.',
      OUT_OF_MEMORY: 'The browser or GPU ran out of available memory. Close GPU-heavy tabs and retry on a higher-memory device.',
      INFERENCE_FAILED: 'The browser GPU could not complete the generation. The technical details may help diagnose the graph or driver.',
    }
    return {
      message: help[error.code] ?? error.message,
      details: { code: error.code, stage: error.stage, recoverable: error.recoverable, diagnostics: error.diagnostics, cause: String(error.cause ?? '') },
    }
  }
  if (error instanceof TypeError && /fetch|network/i.test(error.message)) {
    return { message: 'The browser could not fetch that URL. Confirm it is reachable over HTTPS and explicitly permits CORS from this site.', details: error.message }
  }
  return { message: error instanceof Error ? error.message : String(error), details: error }
}

function setViewerStatus(status: SplatPreviewStatus): void {
  viewerState.dataset.state = status.state
  viewerState.textContent = status.state === 'ready' ? 'Interactive' : status.state
  if (status.state === 'ready') {
    playCompletionChime()
    setPreviewRunState('ready', 'Completed scene is framed to fit. Drag to orbit and scroll to zoom.')
  } else if (status.state === 'loading') setPreviewRunState('working', 'Loading the completed scene into the interactive viewer…')
  else if (status.state === 'failed') {
    completionChimeArmed = false
    setPreviewRunState('failed', 'The preview could not be loaded. Downloads remain available.')
  } else if (!activePlyUrl) setPreviewRunState('waiting', 'A completed scene will appear here without leaving this page.')
}

function releaseRetiredPlyUrl(plyUrl: string): void {
  if (!retiredPlyUrls.delete(plyUrl)) return
  URL.revokeObjectURL(plyUrl)
}

function renderPreview(): void {
  previewRoot.render(createElement(SplatPreview, {
    plyUrl: activePlyUrl ?? null,
    generationKey: previewGenerationKey,
    bgColor: '#030509',
    fov: 60,
    autoRotate: false,
    maxScreenSize: 2048,
    dynamicScene: false,
    initialCameraPosition: previewFrame?.position,
    initialCameraTarget: previewFrame?.target,
    splatPosition: [0, 0, 0],
    // The PLY already has TripoSplat's official +90° export mapping. This is
    // an additional proper presentation rotation for the viewer convention.
    splatRotation: [180, 0, 0],
    splatFlip: [false, false, false],
    onViewerStateChange: setViewerStatus,
    onViewerDisposed: releaseRetiredPlyUrl,
  }))
}

function replaceOutput(ply: Blob, splat: Blob, frame: PreviewFrame | undefined): void {
  const previousPlyUrl = activePlyUrl
  const nextPlyUrl = URL.createObjectURL(ply)
  downloadablePly = ply
  downloadableSplat = splat
  previewFrame = frame
  activePlyUrl = nextPlyUrl
  previewGenerationKey += 1
  if (previousPlyUrl) retiredPlyUrls.add(previousPlyUrl)
  renderPreview()
  downloadPlyButton.disabled = false
  downloadSplatButton.disabled = false
}

function download(blob: Blob | undefined, name: string): void {
  if (!blob) return
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

async function inspectImage(blob: Blob): Promise<Pick<SelectedImage, 'width' | 'height' | 'hasAlpha'>> {
  const bitmap = await createImageBitmap(blob)
  try {
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) throw new Error('Could not inspect the selected image in this browser.')
    context.drawImage(bitmap, 0, 0)
    const alpha = context.getImageData(0, 0, bitmap.width, bitmap.height).data
    let hasAlpha = false
    for (let index = 3; index < alpha.length; index += 4) {
      if (alpha[index] !== 255) {
        hasAlpha = true
        break
      }
    }
    return { width: bitmap.width, height: bitmap.height, hasAlpha }
  } finally {
    bitmap.close()
  }
}

async function setSelectedImage(
  blob: Blob,
  name: string,
  options: ImageSelectionOptions = {},
): Promise<void> {
  if (!blob.type.startsWith('image/')) throw new Error('Choose an image file, or a URL that returns an image content type.')
  const previewBlob = options.previewBlob ?? blob
  const image = await inspectImage(previewBlob)
  const previousPreviewUrl = selectedImage?.previewUrl
  const previewUrl = URL.createObjectURL(previewBlob)
  const inputIsPrepared = options.inputIsPrepared ?? false
  selectedImage = {
    blob,
    sourceBlob: options.sourceBlob ?? blob,
    name,
    previewUrl,
    inputIsPrepared,
    ...image,
  }
  imagePreviewImage.src = previewUrl
  imagePreview.hidden = false
  if (previousPreviewUrl) URL.revokeObjectURL(previousPreviewUrl)
  const alphaDescription = options.summary ?? (image.hasAlpha
    ? 'Transparency detected — ready for generation.'
    : 'No transparency detected — process this image locally before generation.')
  imageSummary.dataset.state = image.hasAlpha || inputIsPrepared ? 'ready' : 'warning'
  imageSummary.textContent = `${name} · ${image.width}×${image.height} · ${alphaDescription}`
  imagePreparationStatus.dataset.state = inputIsPrepared ? 'ready' : 'idle'
  imagePreparationStatus.textContent = inputIsPrepared
    ? 'Prepared locally — the 1024px opaque-black model input will be used for generation.'
    : image.hasAlpha
      ? 'Transparency is already present. Local processing is available if you want a new model-ready crop.'
      : 'Opaque image detected — starting local background removal now.'
  updateGenerateButton()
  updatePrepareImageButton()
  if (!inputIsPrepared && !image.hasAlpha) void prepareSelectedImage()
}

function reportImagePreparationProgress(progress: PreparationProgress): void {
  const percentage = progress.fraction === undefined ? undefined : Math.round(progress.fraction * 100)
  imagePreparationStatus.dataset.state = 'working'
  imagePreparationStatus.textContent = percentage === undefined
    ? progress.message
    : `${progress.message} (${percentage}%)`
  setRunStatus('IMAGE PROCESSING', progress.message, percentage)
}

async function prepareSelectedImage(): Promise<void> {
  const image = selectedImage
  if (!image || busy) return
  hideDiagnostics()
  controller?.abort()
  const preparationController = new AbortController()
  controller = preparationController
  setBusy(true)
  setRunStatus('IMAGE PROCESSING', 'Starting local background removal and TripoSplat framing…', 0)
  imagePreparationStatus.dataset.state = 'working'
  imagePreparationStatus.textContent = 'Starting local image processing…'
  try {
    imagePreparer ??= createImagePreparer({
      defaults: { profile: 'triposplat', strictCompatibility: true },
    })
    const prepared = await imagePreparer.prepare(image.sourceBlob, {
      profile: 'triposplat',
      strictCompatibility: true,
      signal: preparationController.signal,
      onProgress: reportImagePreparationProgress,
    })
    if (preparationController.signal.aborted) return
    await setSelectedImage(prepared.modelInput, image.name, {
      sourceBlob: image.sourceBlob,
      previewBlob: prepared.transparentCutout,
      inputIsPrepared: true,
      summary: prepared.warnings.length > 0
        ? `Prepared locally with note: ${prepared.warnings[0]}`
        : 'Prepared locally — background removed and subject framed for TripoSplat.',
    })
    setRunStatus('IMAGE READY', 'Local image processing finished. The model-ready image is selected.', 100)
  } catch (error) {
    if (preparationController.signal.aborted) return
    const friendly = friendlyError(error)
    imagePreparationStatus.dataset.state = 'error'
    imagePreparationStatus.textContent = `Image processing failed: ${friendly.message}`
    setRunStatus('IMAGE PROCESSING FAILED', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  } finally {
    if (controller === preparationController) controller = undefined
    setBusy(false)
  }
}

function clearImageUrlError(): void {
  imageUrlError.hidden = true
  imageUrlError.textContent = ''
  imageUrlInput.removeAttribute('aria-invalid')
}

function showImageUrlError(message: string): void {
  imageUrlError.textContent = message
  imageUrlError.hidden = false
  imageUrlInput.setAttribute('aria-invalid', 'true')
}

function imageUrlErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return 'The image could not be loaded. Check the URL and try another image host.'
}

async function loadImageFromUrl(): Promise<void> {
  const value = imageUrlInput.value.trim()
  if (!value) throw new Error('Enter an image URL first.')
  let url: URL
  try {
    url = new URL(value)
  } catch {
    throw new Error('Enter a complete image URL, including https://.')
  }
  clearImageUrlError()
  setRunStatus('IMAGE URL', 'Downloading the image directly into this browser…')
  let response: Response
  try {
    response = await fetch(url, { mode: 'cors' })
  } catch {
    throw new Error('The browser could not read this image. The host may be blocking cross-origin access (CORS), or the URL may be unavailable. Use an image host that sends Access-Control-Allow-Origin for this site, or choose a local file instead.')
  }
  if (!response.ok) throw new Error(`The image host returned HTTP ${response.status} ${response.statusText}. Check the URL or choose another image host.`)
  const blob = await response.blob()
  const name = decodeURIComponent(url.pathname.split('/').pop() || 'remote-image')
  await setSelectedImage(blob, name)
  clearImageUrlError()
  setRunStatus('IMAGE READY', 'Image loaded locally. The default browser model package is ready when you are.')
}

type CompatibilityItem = { text: string; state: 'ready' | 'warning' | 'problem' }

let modelServerProblem: string | undefined
let modelServerCheckSequence = 0
let modelServerCheckTimer: number | undefined

function updateCompatibilityList(items: CompatibilityItem[]): void {
  compatibilityList.replaceChildren(...items.map(({ text, state }) => {
    const item = document.createElement('li')
    item.className = `is-${state}`
    item.textContent = text
    return item
  }))
}

function renderPlatformQualification(): void {
  const items: CompatibilityItem[] = []
  if (compatibility) {
    items.push({
      text: compatibility.webgpu ? 'WebGPU adapter detected' : 'WebGPU unavailable',
      state: compatibility.webgpu ? 'ready' : 'problem',
    })
  } else {
    items.push({ text: 'WebGPU compatibility could not be inspected.', state: 'problem' })
  }
  items.push(platformRunBlocker
    ? { text: platformRunBlocker, state: 'problem' }
    : { text: 'Desktop-class browser detected.', state: 'ready' })
  if (modelServerProblem) {
    items.push({ text: modelServerProblem, state: 'problem' })
  } else if (storageQualification) {
    items.push({ text: storageQualification.message, state: storageQualification.state })
  } else {
    items.push({ text: 'Checking the model manifest and browser storage quota…', state: 'warning' })
  }
  if (interruptedRunNotice) items.push({ text: interruptedRunNotice, state: 'warning' })
  for (const warning of compatibility?.warnings ?? []) items.push({ text: warning, state: 'warning' })
  for (const blocker of compatibility?.blockers ?? []) items.push({ text: blocker, state: 'problem' })
  updateCompatibilityList(items)

  const ready = compatibility?.supported === true
    && storageQualification?.supported === true
    && platformRunBlocker === undefined
    && modelServerProblem === undefined
  platformBadge.classList.toggle('is-ready', ready)
  platformBadge.classList.toggle('is-missing', !ready)
  platformBadge.lastElementChild!.textContent = ready
    ? storageQualification?.state === 'warning' ? 'Ready with warning' : 'Browser ready'
    : platformRunBlocker
      ? 'Desktop required'
      : compatibility?.supported === false
        ? 'WebGPU blocked'
        : storageQualification?.supported === false
          ? 'Storage blocked'
          : modelServerProblem
            ? 'Model check failed'
            : 'Checking storage'
  updateGenerateButton()
}

function renderStorageStatus(summary: ModelManifestSummary, qualification: StorageQualification): void {
  cacheMode.textContent = cacheBackend === 'none' ? 'NO PERSISTENT CACHE' : `${cacheBackend.toUpperCase()} CACHE`
  const cached = qualification.cachedBytes ?? 0
  const details = [
    `Verified model size: ${formatBytes(summary.declaredBytes)}.`,
    `${formatBytes(cached)} cached; ${formatBytes(Math.max(0, summary.declaredBytes - cached))} not yet cached.`,
  ]
  if (
    qualification.quotaBytes !== undefined
    && qualification.usageBytes !== undefined
    && qualification.availableBytes !== undefined
  ) {
    details.push(
      `This site is using ${formatBytes(qualification.usageBytes)} of a ${formatBytes(qualification.quotaBytes)} browser quota; ${formatBytes(qualification.availableBytes)} is available.`,
    )
  }
  if (qualification.requiredBytes !== undefined) {
    details.push(`${formatBytes(qualification.requiredBytes)} of browser-storage capacity is required before download.`)
  }
  details.push(qualification.message)
  modelStatus.textContent = details.join(' ')
}

async function refreshCacheStatus(summary = activeManifestSummary): Promise<StorageQualification> {
  const qualification = await qualifyStorage(summary)
  storageQualification = qualification
  renderStorageStatus(summary, qualification)
  renderPlatformQualification()
  return qualification
}

async function requestPersistentStorage(): Promise<void> {
  if (cacheBackend === 'none') return
  const storage = navigator.storage
  if (!storage?.persisted || !storage.persist) return
  const persisted = await storage.persisted().catch(() => false)
  if (!persisted) await storage.persist().catch(() => false)
}

async function verifyModelServer(
  base: string,
  signal?: AbortSignal,
): Promise<{ summary: ModelManifestSummary; qualification: StorageQualification }> {
  setRunStatus('MODEL SERVER', 'Checking manifest access and declared artifact sizes before the large download…')
  const manifestUrl = new URL('manifest.json', base)
  let response: Response
  try {
    response = await fetch(manifestUrl, { mode: 'cors', cache: 'no-cache', signal })
  } catch (error) {
    if (signal?.aborted) throw signal.reason
    throw new Error(`The model manifest could not be fetched. The server must permit CORS from this site. ${error instanceof Error ? error.message : String(error)}`)
  }
  if (!response.ok) {
    throw new Error(`The model manifest returned HTTP ${response.status} ${response.statusText}. Use the CDN directory containing manifest.json.`)
  }
  const manifest: unknown = await response.json().catch(() => undefined)
  if (manifest === undefined) throw new Error('The model manifest was not valid JSON.')
  const summary = summarizeModelManifest(manifest)
  const qualification = await qualifyStorage(summary)
  return { summary, qualification }
}

function applyVerifiedModel(result: { summary: ModelManifestSummary; qualification: StorageQualification }): void {
  activeManifestSummary = result.summary
  storageQualification = result.qualification
  modelServerProblem = undefined
  renderStorageStatus(result.summary, result.qualification)
  renderPlatformQualification()
}

function requireQualifiedStorage(qualification: StorageQualification | undefined): asserts qualification is StorageQualification {
  if (!qualification?.supported) {
    throw new Error(qualification?.message ?? 'Browser storage has not passed the model-capacity check yet.')
  }
}

async function checkPlatform(): Promise<void> {
  cacheBackend = chooseCacheBackend()
  platformRunBlocker = isMobilePlatform()
    ? 'This 6.5 GB fp32 engineering runtime is desktop-only for now. Mobile browsers expose no reliable unified-memory limit and may terminate or reload this page under memory pressure.'
    : undefined

  try {
    compatibility = await TripoSplatWebGPU.checkCompatibility({ estimatedModelBytes: activeManifestSummary.declaredBytes })
  } catch (error) {
    compatibility = undefined
    setRunStatus('CHECK FAILED', 'Could not inspect WebGPU compatibility.')
    showDiagnostics('Could not complete the browser compatibility check.', friendlyError(error).details)
  }

  try {
    const base = normalizedModelBase(modelBaseInput.value)
    applyVerifiedModel(await verifyModelServer(base))
  } catch (error) {
    storageQualification = undefined
    modelServerProblem = friendlyError(error).message
    cacheMode.textContent = cacheBackend === 'none' ? 'NO PERSISTENT CACHE' : `${cacheBackend.toUpperCase()} CACHE`
    modelStatus.textContent = `The model manifest and storage requirement could not be verified. ${modelServerProblem}`
    showDiagnostics(modelServerProblem, friendlyError(error).details)
  }
  renderPlatformQualification()

  if (platformRunBlocker) {
    setRunStatus('MOBILE NOT QUALIFIED', platformRunBlocker)
    showDiagnostics(platformRunBlocker, {
      probableCauseOfUnexpectedReload: 'Browser or OS process termination under memory pressure; this cannot be proven from browser APIs.',
      reliableUnifiedMemoryCapacityAvailable: false,
      modelBytes: activeManifestSummary.declaredBytes,
    })
  } else if (!compatibility?.supported) {
    setRunStatus('UNSUPPORTED', compatibility?.blockers.join(' ') || 'WebGPU is unavailable in this browser.')
  } else if (modelServerProblem) {
    setRunStatus('MODEL CHECK FAILED', modelServerProblem)
  } else if (!storageQualification?.supported) {
    setRunStatus('STORAGE BLOCKED', storageQualification?.message ?? 'Browser storage could not be qualified.')
    showDiagnostics(storageQualification?.message ?? 'Browser storage could not be qualified.', storageQualification)
  } else if (interruptedRunNotice) {
    setRunStatus('PREVIOUS RUN INTERRUPTED', interruptedRunNotice)
    showDiagnostics(interruptedRunNotice, {
      note: 'Browsers do not expose a definitive reason for a tab reload or process termination.',
    })
  } else {
    setRunStatus('SYSTEM READY', 'WebGPU, the model manifest, and browser-storage capacity passed the preflight checks.')
  }
}

async function prepareModel(base: string, signal: AbortSignal): Promise<TripoSplatWebGPU> {
  if (model && loadedModelBase === base) return model
  if (model) await model.dispose()
  model = new TripoSplatWebGPU({
    modelBaseUrl: base,
    manifestUrl: 'manifest.json',
    executionProviders: ['webgpu'],
    cache: cacheBackend,
    wasmPaths: {
      mjs: '/ort/ort-wasm-simd-threaded.asyncify.mjs',
      wasm: '/ort/ort-wasm-simd-threaded.asyncify.wasm',
    },
  })
  loadedModelBase = undefined
  await model.load({ signal, onProgress: reportLoadProgress })
  loadedModelBase = base
  return model
}

function reportLoadProgress(progress: LoadProgress): void {
  sampleMemoryPeak()
  const fraction = progress.progress ?? (progress.totalBytes ? (progress.loadedBytes ?? 0) / progress.totalBytes : undefined)
  const phase = /download/i.test(progress.message)
    ? 'model download'
    : progress.stage === 'graphs'
      ? 'graph load'
      : 'runtime startup'
  updateActiveRunMarker(phase)
  setRunStatus(`MODEL · ${progress.stage.toUpperCase()}`, progress.message, fraction)
}

function reportGenerationProgress(progress: GenerationProgress): void {
  sampleMemoryPeak()
  const samplingFraction = progress.stage === 'sampling' && progress.totalInvocations
    ? (progress.invocation ?? 0) / progress.totalInvocations
    : undefined
  const fraction = samplingFraction ?? progress.progress ?? (progress.totalSteps ? (progress.step ?? 0) / progress.totalSteps : undefined)
  const samplingDetail = progress.stage === 'sampling' && progress.step && progress.totalSteps && progress.invocation && progress.totalInvocations
    ? ` Step ${progress.step}/${progress.totalSteps} · CFG invocation ${progress.invocation}/${progress.totalInvocations}.`
    : ''
  updateActiveRunMarker(`generation: ${progress.stage}`)
  setRunStatus(`GENERATING · ${progress.stage.toUpperCase()}`, `${progress.message}${samplingDetail}`, fraction)
}

async function run(): Promise<void> {
  if (!selectedImage) throw new Error('Choose an image before generating.')
  if (!compatibility?.supported) throw new Error('This browser does not meet the current WebGPU requirements.')
  if (platformRunBlocker) throw new Error(platformRunBlocker)
  requireQualifiedStorage(storageQualification)
  const base = normalizedModelBase(modelBaseInput.value)
  hideDiagnostics()
  armCompletionChime()
  controller?.abort()
  controller = new AbortController()
  startRunTelemetry(storageQualification?.cachedBytes ?? 0)
  setBusy(true)
  startActiveRunMarker()
  setRunStatus('SOURCE IMAGE', 'Image accepted. Starting a local, browser-only generation…')
  setPreviewRunState(
    activePlyUrl ? 'retained' : 'working',
    activePlyUrl
      ? 'Generating a replacement. The last completed scene remains interactive.'
      : 'Generating your first scene. This page and preview stay in place.',
  )
  let completed = false
  try {
    updateActiveRunMarker('storage and manifest preflight')
    await requestPersistentStorage()
    const verified = await verifyModelServer(base, controller.signal)
    applyVerifiedModel(verified)
    requireQualifiedStorage(verified.qualification)
    const activeModel = await prepareModel(base, controller.signal)
    setRunStatus('PREPROCESSING', 'Preparing the image locally…')
    updateActiveRunMarker('generation: preprocessing')
    const scene = await activeModel.generate(selectedImage.blob, {
      steps: DEFAULT_STEPS,
      gaussianCount: 262_144,
      seed: 42,
      inputIsPrepared: selectedImage.inputIsPrepared,
      signal: controller.signal,
      onProgress: reportGenerationProgress,
    })
    try {
      updateActiveRunMarker('export')
      setRunStatus('EXPORTING', 'Encoding portable PLY and .splat files…')
      const ply = await scene.exportPLY()
      const splat = await scene.exportSplat()
      const frame = fitPreviewFrame(scene.positions)
      setPreviewRunState('working', 'Framing and swapping in the completed scene without resetting the page…')
      replaceOutput(ply, splat, frame)
      setRunStatus('COMPLETE', `Generated ${scene.count.toLocaleString()} Gaussians. Preview and downloads are ready.`, 1)
      const telemetry = finishRunTelemetry()
      const qualificationAfter = await refreshCacheStatus()
      if (telemetry) {
        const cachedBytesAfter = qualificationAfter.cachedBytes ?? telemetry.cachedBytesBefore
        showBenchmarkReport(buildBenchmarkReport(telemetry, scene, base, cachedBytesAfter), performance.now() - telemetry.startedAtMs)
      }
      completed = true
    } finally {
      scene.dispose()
    }
  } catch (error) {
    const friendly = friendlyError(error)
    setRunStatus('NEEDS ATTENTION', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  } finally {
    if (controller?.signal.aborted) setRunStatus('CANCELLED', 'Cancelled. The next run will start a clean worker.')
    if (!completed) {
      completionChimeArmed = false
      finishRunTelemetry()
      setPreviewRunState(
        activePlyUrl ? 'retained' : 'failed',
        activePlyUrl ? 'The last completed scene is still available.' : 'No completed scene was produced. Review the guidance below and try again.',
      )
    }
    clearActiveRunMarker()
    controller = undefined
    setBusy(false)
  }
}

async function recheckModelServer(sequence: number): Promise<void> {
  try {
    const base = normalizedModelBase(modelBaseInput.value)
    const verified = await verifyModelServer(base)
    if (sequence !== modelServerCheckSequence) return
    applyVerifiedModel(verified)
    setRunStatus(
      verified.qualification.supported ? 'MODEL SERVER READY' : 'STORAGE BLOCKED',
      verified.qualification.message,
    )
    if (!verified.qualification.supported) showDiagnostics(verified.qualification.message, verified.qualification)
  } catch (error) {
    if (sequence !== modelServerCheckSequence) return
    storageQualification = undefined
    modelServerProblem = friendlyError(error).message
    modelStatus.textContent = `The model server could not be qualified. ${modelServerProblem}`
    renderPlatformQualification()
    setRunStatus('MODEL CHECK FAILED', modelServerProblem)
    showDiagnostics(modelServerProblem, friendlyError(error).details)
  }
}

function modelBaseFromLocation(): string {
  const supplied = new URLSearchParams(location.search).get('modelBaseUrl')
  return supplied ?? DEFAULT_MODEL_BASE
}

modelBaseInput.value = modelBaseFromLocation()
applyProgressDetailMode(progressDetailMode, false)
renderPreview()

for (const button of progressDetailButtons) {
  button.addEventListener('click', () => {
    const mode = button.dataset.progressDetail
    if (mode === 'guided' || mode === 'technical') applyProgressDetailMode(mode)
  })
}

chooseFileButton.addEventListener('click', () => fileInput.click())
prepareImageButton.addEventListener('click', () => { void prepareSelectedImage() })
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0]
  fileInput.value = ''
  if (!file) return
  void setSelectedImage(file, file.name).catch((error) => {
    const friendly = friendlyError(error)
    setRunStatus('IMAGE ERROR', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  })
})

loadImageUrlButton.addEventListener('click', () => {
  void loadImageFromUrl().catch((error) => {
    const message = imageUrlErrorMessage(error)
    showImageUrlError(message)
    setRunStatus('IMAGE URL ERROR', message)
    showDiagnostics(message, friendlyError(error).details)
  })
})
imageUrlInput.addEventListener('input', clearImageUrlError)

for (const eventName of ['dragenter', 'dragover']) {
  sourcePanel.addEventListener(eventName, (event) => {
    event.preventDefault()
    sourcePanel.classList.add('is-dragging')
  })
}
for (const eventName of ['dragleave', 'drop']) {
  sourcePanel.addEventListener(eventName, (event) => {
    event.preventDefault()
    sourcePanel.classList.remove('is-dragging')
  })
}
sourcePanel.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files[0]
  if (!file) return
  void setSelectedImage(file, file.name).catch((error) => {
    const friendly = friendlyError(error)
    setRunStatus('IMAGE ERROR', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  })
})

dropZone.addEventListener('click', () => fileInput.click())
modelBaseInput.addEventListener('input', () => {
  modelServerCheckSequence += 1
  const sequence = modelServerCheckSequence
  storageQualification = undefined
  modelServerProblem = undefined
  modelStatus.textContent = 'Model server changed. Waiting briefly, then checking its manifest and storage requirement…'
  renderPlatformQualification()
  if (modelServerCheckTimer !== undefined) window.clearTimeout(modelServerCheckTimer)
  modelServerCheckTimer = window.setTimeout(() => { void recheckModelServer(sequence) }, 600)
})
generateButton.addEventListener('click', () => { void run().catch((error) => {
  const friendly = friendlyError(error)
  setRunStatus('NEEDS ATTENTION', friendly.message)
  showDiagnostics(friendly.message, friendly.details)
}) })
cancelButton.addEventListener('click', () => {
  imagePreparer?.cancel()
  controller?.abort(new DOMException('Cancelled by user.', 'AbortError'))
})
clearCacheButton.addEventListener('click', () => {
  void clearModelCache().then(async () => {
    const qualification = await refreshCacheStatus()
    const message = qualification.supported
      ? 'Verified model files were removed from browser storage. A future run will download them again.'
      : `Verified files were removed. ${qualification.message}`
    setRunStatus(qualification.supported ? 'CACHE CLEARED' : 'STORAGE BLOCKED', message)
    if (!qualification.supported) showDiagnostics(qualification.message, qualification)
  }).catch((error) => {
    const friendly = friendlyError(error)
    setRunStatus('CACHE ERROR', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  })
})
downloadPlyButton.addEventListener('click', () => download(downloadablePly, 'triposplat-scene.ply'))
downloadSplatButton.addEventListener('click', () => download(downloadableSplat, 'triposplat-scene.splat'))
copyBenchmarkButton.addEventListener('click', () => {
  void copyBenchmarkReport().catch((error) => {
    const friendly = friendlyError(error)
    benchmarkSummary.textContent = `Could not copy the report: ${friendly.message}`
  })
})

void checkPlatform().catch((error) => {
  const friendly = friendlyError(error)
  setRunStatus('CHECK FAILED', friendly.message)
  showDiagnostics(friendly.message, friendly.details)
})
window.addEventListener('pagehide', () => {
  pageIsHiding = true
  imagePreparer?.cancel()
  imagePreparer?.dispose()
  controller?.abort()
  void completionChimeContext?.close()
  void model?.dispose()
  previewRoot.unmount()
  if (selectedImage) URL.revokeObjectURL(selectedImage.previewUrl)
  if (activePlyUrl) URL.revokeObjectURL(activePlyUrl)
  for (const plyUrl of retiredPlyUrls) URL.revokeObjectURL(plyUrl)
}, { once: true })
