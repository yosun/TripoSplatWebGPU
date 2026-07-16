import { createElement } from 'react'
import { flushSync } from 'react-dom'
import { createRoot, type Root } from 'react-dom/client'

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
const DEFAULT_MODEL_BASE = 'https://huggingface.co/Yosun/TripoSplat-WebGPU/resolve/main/triposplat-webgpu/0.1.0-fp32.20260715/'
const DEFAULT_STEPS = 4

interface SelectedImage {
  blob: Blob
  name: string
  previewUrl: string
  width: number
  height: number
  hasAlpha: boolean
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
const imageSummary = requiredElement<HTMLElement>('#image-summary')
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
const progressTrack = requiredElement<HTMLElement>('.progress-track')
const progressFill = requiredElement<HTMLElement>('#progress-fill')
const diagnostics = requiredElement<HTMLElement>('#diagnostics')
const diagnosticMessage = requiredElement<HTMLElement>('#diagnostic-message')
const diagnosticDetails = requiredElement<HTMLElement>('#diagnostic-details')
const platformBadge = requiredElement<HTMLElement>('#platform-badge')
const compatibilityList = requiredElement<HTMLUListElement>('#compatibility-list')
const previewMount = requiredElement<HTMLElement>('#web-preview-root')
const viewerState = requiredElement<HTMLElement>('#viewer-state')
const downloadPlyButton = requiredElement<HTMLButtonElement>('#download-ply')
const downloadSplatButton = requiredElement<HTMLButtonElement>('#download-splat')

const previewRoot: Root = createRoot(previewMount)
let compatibility: CompatibilityReport | undefined
let cacheBackend: CacheBackend = 'none'
let selectedImage: SelectedImage | undefined
let model: TripoSplatWebGPU | undefined
let loadedModelBase: string | undefined
let controller: AbortController | undefined
let busy = false
let activePlyUrl: string | undefined
let downloadablePly: Blob | undefined
let downloadableSplat: Blob | undefined
let previewGenerationKey = 0

function formatBytes(bytes: number): string {
  if (bytes < 1_024) return `${bytes} B`
  if (bytes < 1_024 ** 2) return `${(bytes / 1_024).toFixed(1)} KiB`
  if (bytes < 1_024 ** 3) return `${(bytes / 1_024 ** 2).toFixed(1)} MiB`
  return `${(bytes / 1_024 ** 3).toFixed(2)} GiB`
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

function setRunStatus(stage: string, message: string, progress?: number): void {
  runStage.textContent = stage
  runStatus.textContent = message
  const percent = progress === undefined ? 0 : Math.max(0, Math.min(100, progress * 100))
  progressFill.style.width = `${percent.toFixed(1)}%`
  progressTrack.setAttribute('aria-valuenow', String(Math.round(percent)))
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
}

function updateGenerateButton(): void {
  const compatible = compatibility?.supported === true
  const hasModelBase = modelBaseInput.value.trim().length > 0
  generateButton.disabled = busy || !compatible || !selectedImage || !hasModelBase
  if (busy) {
    generateButton.classList.add('is-working')
    generateButton.firstElementChild!.textContent = 'Working in your browser…'
  } else {
    generateButton.classList.remove('is-working')
    generateButton.firstElementChild!.textContent = selectedImage && hasModelBase
      ? 'Generate spatial scene'
      : 'Choose an image to begin'
  }
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
      message: 'This image has no transparency. Use a PNG or WebP with a transparent background; automatic background removal is not bundled into this browser-only preview.',
      details: { code: error.code, stage: error.stage, diagnostics: error.diagnostics },
    }
  }
  if (error instanceof CancelledError || (error instanceof DOMException && error.name === 'AbortError')) {
    return { message: 'Cancelled. A future run starts with a clean browser worker.', details: error.message }
  }
  if (error instanceof TripoSplatError) {
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
}

function renderPreview(): void {
  flushSync(() => {
    previewRoot.render(createElement(SplatPreview, {
      plyUrl: activePlyUrl ?? null,
      generationKey: previewGenerationKey,
      bgColor: '#030509',
      fov: 60,
      autoRotate: false,
      maxScreenSize: 2048,
      splatPosition: [0, 0, 0],
      splatRotation: [0, 0, 0],
      splatFlip: [false, false, false],
      onViewerStateChange: setViewerStatus,
    }))
  })
}

function clearOutput(): void {
  if (activePlyUrl) URL.revokeObjectURL(activePlyUrl)
  activePlyUrl = undefined
  downloadablePly = undefined
  downloadableSplat = undefined
  previewGenerationKey += 1
  downloadPlyButton.disabled = true
  downloadSplatButton.disabled = true
  renderPreview()
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

async function setSelectedImage(blob: Blob, name: string): Promise<void> {
  if (!blob.type.startsWith('image/')) throw new Error('Choose an image file, or a URL that returns an image content type.')
  const image = await inspectImage(blob)
  if (selectedImage) URL.revokeObjectURL(selectedImage.previewUrl)
  clearOutput()
  selectedImage = { blob, name, previewUrl: URL.createObjectURL(blob), ...image }
  const alphaDescription = image.hasAlpha
    ? 'Transparency detected — ready for generation.'
    : 'No transparency detected — this will need a browser-local background remover and cannot run in this preview.'
  imageSummary.dataset.state = image.hasAlpha ? 'ready' : 'warning'
  imageSummary.textContent = `${name} · ${image.width}×${image.height} · ${alphaDescription}`
  updateGenerateButton()
}

async function loadImageFromUrl(): Promise<void> {
  const value = imageUrlInput.value.trim()
  if (!value) throw new Error('Enter an image URL first.')
  const url = new URL(value)
  setRunStatus('IMAGE URL', 'Downloading the image directly into this browser…')
  const response = await fetch(url, { mode: 'cors' })
  if (!response.ok) throw new Error(`The image URL returned HTTP ${response.status} ${response.statusText}.`)
  const blob = await response.blob()
  const name = decodeURIComponent(url.pathname.split('/').pop() || 'remote-image')
  await setSelectedImage(blob, name)
  setRunStatus('IMAGE READY', 'Image loaded locally. Add your model server URL to continue.')
}

function updateCompatibilityList(items: Array<{ text: string; state: 'ready' | 'warning' | 'problem' }>): void {
  compatibilityList.replaceChildren(...items.map(({ text, state }) => {
    const item = document.createElement('li')
    item.className = `is-${state}`
    item.textContent = text
    return item
  }))
}

async function refreshCacheStatus(): Promise<void> {
  const status = await getModelCacheStatus()
  const persistent = status.backends.find((entry) => entry.backend === cacheBackend)
  const cacheLabel = cacheBackend === 'none' ? 'NO PERSISTENT CACHE' : `${cacheBackend.toUpperCase()} CACHE`
  cacheMode.textContent = cacheLabel
  const storage = navigator.storage
  const estimate = await storage?.estimate?.().catch(() => undefined)
  const cached = status.entryCount > 0 ? `${formatBytes(status.totalBytes)} verified files cached.` : 'No verified model files cached yet.'
  const quota = estimate?.quota ? ` Browser quota: ${formatBytes(estimate.quota)}.` : ''
  const availability = persistent && !persistent.available ? ` Cache unavailable: ${persistent.error ?? 'unknown error'}` : ''
  modelStatus.textContent = `${cached}${quota}${availability}`
}

async function checkPlatform(): Promise<void> {
  cacheBackend = chooseCacheBackend()
  try {
    compatibility = await TripoSplatWebGPU.checkCompatibility({ estimatedModelBytes: MODEL_BYTES })
    const items: Array<{ text: string; state: 'ready' | 'warning' | 'problem' }> = []
    items.push({ text: compatibility.webgpu ? 'WebGPU detected' : 'WebGPU unavailable', state: compatibility.webgpu ? 'ready' : 'problem' })
    items.push({ text: `${cacheBackend === 'opfs' ? 'Persistent browser storage' : cacheBackend === 'cache-api' ? 'Cache API storage' : 'No persistent cache'} selected`, state: cacheBackend === 'none' ? 'warning' : 'ready' })
    for (const warning of compatibility.warnings) items.push({ text: warning, state: 'warning' })
    for (const blocker of compatibility.blockers) items.push({ text: blocker, state: 'problem' })
    updateCompatibilityList(items)
    platformBadge.classList.toggle('is-ready', compatibility.supported)
    platformBadge.classList.toggle('is-missing', !compatibility.supported)
    platformBadge.lastElementChild!.textContent = compatibility.supported ? 'WebGPU ready' : 'WebGPU blocked'
    if (!compatibility.supported) setRunStatus('UNSUPPORTED', compatibility.blockers.join(' ') || 'WebGPU is unavailable in this browser.')
  } catch (error) {
    compatibility = undefined
    updateCompatibilityList([{ text: 'Could not inspect WebGPU compatibility.', state: 'problem' }])
    platformBadge.classList.add('is-missing')
    platformBadge.lastElementChild!.textContent = 'Platform check failed'
    setRunStatus('CHECK FAILED', 'Could not inspect WebGPU compatibility.')
    showDiagnostics('Could not complete the browser compatibility check.', friendlyError(error).details)
  }
  await refreshCacheStatus()
  updateGenerateButton()
}

async function requestPersistentStorage(): Promise<void> {
  if (cacheBackend === 'none') return
  const storage = navigator.storage
  if (!storage?.persisted || !storage.persist) return
  const persisted = await storage.persisted()
  if (!persisted) await storage.persist().catch(() => false)
}

async function verifyModelServer(base: string): Promise<void> {
  setRunStatus('MODEL SERVER', 'Checking manifest access before the large download…')
  const manifestUrl = new URL('manifest.json', base)
  let response: Response
  try {
    response = await fetch(manifestUrl, { mode: 'cors', cache: 'no-cache' })
  } catch (error) {
    throw new Error(`The model manifest could not be fetched. The server must permit CORS from this site. ${error instanceof Error ? error.message : String(error)}`)
  }
  if (!response.ok) throw new Error(`The model manifest returned HTTP ${response.status} ${response.statusText}. Use the CDN directory containing manifest.json.`)
  const manifest: unknown = await response.json().catch(() => undefined)
  if (!manifest || typeof manifest !== 'object') throw new Error('The model manifest was not valid JSON.')
  modelStatus.textContent = `Model server verified. First download: about ${formatBytes(MODEL_BYTES)}; verified cache: ${cacheBackend}.`
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
  const fraction = progress.progress ?? (progress.totalBytes ? (progress.loadedBytes ?? 0) / progress.totalBytes : undefined)
  setRunStatus(`MODEL · ${progress.stage.toUpperCase()}`, progress.message, fraction)
}

function reportGenerationProgress(progress: GenerationProgress): void {
  const fraction = progress.progress ?? (progress.totalSteps ? (progress.step ?? 0) / progress.totalSteps : undefined)
  setRunStatus(`GENERATING · ${progress.stage.toUpperCase()}`, progress.message, fraction)
}

async function run(): Promise<void> {
  if (!selectedImage) throw new Error('Choose an image before generating.')
  if (!compatibility?.supported) throw new Error('This browser does not meet the current WebGPU requirements.')
  const base = normalizedModelBase(modelBaseInput.value)
  hideDiagnostics()
  controller?.abort()
  controller = new AbortController()
  setBusy(true)
  clearOutput()
  try {
    await requestPersistentStorage()
    await verifyModelServer(base)
    const activeModel = await prepareModel(base, controller.signal)
    setRunStatus('PREPROCESSING', 'Preparing the image locally…')
    const scene = await activeModel.generate(selectedImage.blob, {
      steps: DEFAULT_STEPS,
      gaussianCount: 262_144,
      seed: 42,
      signal: controller.signal,
      onProgress: reportGenerationProgress,
    })
    try {
      setRunStatus('EXPORTING', 'Encoding portable PLY and .splat files…')
      downloadablePly = await scene.exportPLY()
      downloadableSplat = await scene.exportSplat()
      activePlyUrl = URL.createObjectURL(downloadablePly)
      previewGenerationKey += 1
      renderPreview()
      downloadPlyButton.disabled = false
      downloadSplatButton.disabled = false
      setRunStatus('COMPLETE', `Generated ${scene.count.toLocaleString()} Gaussians. Preview and downloads are ready.`, 1)
      await refreshCacheStatus()
    } finally {
      scene.dispose()
    }
  } catch (error) {
    const friendly = friendlyError(error)
    setRunStatus('NEEDS ATTENTION', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  } finally {
    if (controller?.signal.aborted) setRunStatus('CANCELLED', 'Cancelled. The next run will start a clean worker.')
    controller = undefined
    setBusy(false)
  }
}

function modelBaseFromLocation(): string {
  const supplied = new URLSearchParams(location.search).get('modelBaseUrl')
  return supplied ?? DEFAULT_MODEL_BASE
}

modelBaseInput.value = modelBaseFromLocation()
renderPreview()

chooseFileButton.addEventListener('click', () => fileInput.click())
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
    const friendly = friendlyError(error)
    setRunStatus('IMAGE URL ERROR', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  })
})

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
modelBaseInput.addEventListener('input', updateGenerateButton)
generateButton.addEventListener('click', () => { void run().catch((error) => {
  const friendly = friendlyError(error)
  setRunStatus('NEEDS ATTENTION', friendly.message)
  showDiagnostics(friendly.message, friendly.details)
}) })
cancelButton.addEventListener('click', () => controller?.abort(new DOMException('Cancelled by user.', 'AbortError')))
clearCacheButton.addEventListener('click', () => {
  void clearModelCache().then(async () => {
    await refreshCacheStatus()
    setRunStatus('CACHE CLEARED', 'Verified model files were removed from browser storage. A future run will download them again.')
  }).catch((error) => {
    const friendly = friendlyError(error)
    setRunStatus('CACHE ERROR', friendly.message)
    showDiagnostics(friendly.message, friendly.details)
  })
})
downloadPlyButton.addEventListener('click', () => download(downloadablePly, 'triposplat-scene.ply'))
downloadSplatButton.addEventListener('click', () => download(downloadableSplat, 'triposplat-scene.splat'))

void checkPlatform()
window.addEventListener('pagehide', () => {
  controller?.abort()
  void model?.dispose()
  previewRoot.unmount()
  if (selectedImage) URL.revokeObjectURL(selectedImage.previewUrl)
  if (activePlyUrl) URL.revokeObjectURL(activePlyUrl)
}, { once: true })
