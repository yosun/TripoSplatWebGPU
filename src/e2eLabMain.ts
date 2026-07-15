/*
This version adds:

* Editable steps, guidance scale, shift, and seed
* Step presets: 4, 8, 12, 20, 32, 50
* Deterministic comparisons using the same fixture tensors
* Run history showing parameters, generation time, hashes, Gaussian count, and pass/fail
* Download PLY and Download .splat
* Persistent OPFS preference where supported
* Storage usage and persistence reporting
* Image dimension and alpha-channel reporting
* Package-owned production worker rather than the workspace source worker
* Current run cancellation
* No separate HTML changes because the additional controls are created dynamically
*/import { createElement } from 'react'
import { flushSync } from 'react-dom'
import { createRoot, type Root } from 'react-dom/client'

import { TripoSplatWebGPU } from '../packages/triposplat-webgpu/dist/index.js'
import type {
  CacheBackend,
  GenerationProgress,
  GaussianScene,
  LoadProgress,
} from '../packages/triposplat-webgpu/dist/index.js'
import {
  qualifyGaussianSceneStructure,
  qualifyGaussianViewer,
  type GaussianSceneStructuralQualification,
  type GaussianViewerObservation,
  type GaussianViewerQualification,
} from './e2eLabQualification'
import {
  SplatPreview,
  type SplatPreviewStatus,
} from './components/SplatPreview'

const FLOAT_COUNTS = {
  vaeNoise: 1 * 32 * 128 * 128,
  latentNoise: 1 * 8192 * 16,
  cameraNoise: 1 * 1 * 5,
} as const

const DEFAULT_GENERATION_SETTINGS = {
  steps: 20,
  guidanceScale: 3,
  shift: 3,
  seed: 42,
} as const

const STEP_PRESETS = [4, 8, 12, 20, 32, 50] as const

const PLY_FLOATS_PER_GAUSSIAN = 17
const SPLAT_BYTES_PER_GAUSSIAN = 32
const VIEWER_LOAD_TIMEOUT_MS = 90_000
const MAX_PROGRESS_RECORDS = 5_000
const MAX_RUN_HISTORY = 20

interface GenerationParameters {
  steps: number
  guidanceScale: number
  shift: number
  seed: number
}

interface PreparedImageMetadata {
  mimeType: string
  byteLength: number
  width: number
  height: number
  hasAlpha: boolean
}

interface ProgressRecord {
  phase: 'load' | 'generate'
  elapsedMs: number
  stage: string
  message: string
  progress?: number
  loadedBytes?: number
  totalBytes?: number
  graph?: string
  step?: number
  totalSteps?: number
  invocation?: number
  totalInvocations?: number
}

interface ExportQualification {
  passed: boolean
  ply: {
    byteLength: number
    sha256: string
    headerBytes: number
    expectedByteLength: number
    binaryLittleEndian: boolean
    vertexCountMatches: boolean
    byteLengthMatches: boolean
  }
  splat: {
    byteLength: number
    sha256: string
    expectedByteLength: number
    byteLengthMatches: boolean
  }
}

interface StorageQualification {
  persistenceRequested: boolean
  persisted?: boolean
  usageBytes?: number
  quotaBytes?: number
}

interface E2ELabSuccess {
  passed: boolean
  completed: true
  qualificationScope: 'end-to-end-execution-structure-export-and-viewer-sanity'
  numericalParityClaimed: false
  model: {
    baseUrl: string
    manifestVersion?: string
    modelRevision?: string
    declaredModelBytes?: number
    configuredGraphs: readonly string[]
    cache: CacheBackend
  }
  fixtures: {
    preparedImage: string
    preparedImageMetadata: PreparedImageMetadata
    vaeNoise: string
    latentNoise: string
    cameraNoise: string
    explicitDeterministicNoise: true
  }
  generation: GenerationParameters
  scene: GaussianSceneStructuralQualification
  gaussianCount: number
  exports: ExportQualification
  viewer: GaussianViewerQualification
  timingsMs: {
    fixtureFetch: number
    load: number
    generate: number
    plyExportAndHash: number
    splatExportAndHash: number
    viewerLoad: number
    total: number
  }
  pipelineMeasuredTimingsMs?: unknown
  progress: ProgressRecord[]
  lifecycle: {
    sceneDisposed: boolean
    modelDisposed: boolean
  }
  storage?: StorageQualification
  environment: {
    userAgent: string
    crossOriginIsolated: boolean
    webgpu: boolean
  }
}

interface E2ELabFailure {
  passed: false
  completed: false
  qualificationScope: 'end-to-end-execution-structure-export-and-viewer-sanity'
  numericalParityClaimed: false
  error: SerializedError
  progress: ProgressRecord[]
  timingsMs: {
    total: number
  }
  environment: {
    userAgent: string
    crossOriginIsolated: boolean
    webgpu: boolean
  }
}

interface SerializedError {
  name: string
  message: string
  stack?: string
  code?: string
  stage?: string
  diagnostics?: unknown
  cause?: SerializedError
}

interface RunHistoryRecord {
  runNumber: number
  timestamp: string
  passed: boolean
  steps: number
  guidanceScale: number
  shift: number
  seed: number
  gaussianCount: number
  fixtureFetchMs: number
  loadMs: number
  generateMs: number
  totalMs: number
  plyBytes: number
  splatBytes: number
  plySha256: string
  splatSha256: string
}

interface InjectedControls {
  stepsInput: HTMLInputElement
  guidanceInput: HTMLInputElement
  shiftInput: HTMLInputElement
  seedInput: HTMLInputElement
  downloadPlyButton: HTMLButtonElement
  downloadSplatButton: HTMLButtonElement
  clearHistoryButton: HTMLButtonElement
  storageStatus: HTMLElement
  imageStatus: HTMLElement
  historyBody: HTMLTableSectionElement
}

type E2ELabResult = E2ELabSuccess | E2ELabFailure

declare global {
  interface Window {
    __TRIPOSPLAT_E2E_RESULT__?: E2ELabResult
    __TRIPOSPLAT_E2E_HISTORY__?: RunHistoryRecord[]
  }
}

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector)

  if (element === null) {
    throw new Error(`Required E2E lab element '${selector}' was not found.`)
  }

  return element
}

const modelBaseInput = requiredElement<HTMLInputElement>('#model-base')
const cacheInput = requiredElement<HTMLSelectElement>('#cache')
const preparedImageInput = requiredElement<HTMLInputElement>('#prepared-image')
const vaeNoiseInput = requiredElement<HTMLInputElement>('#vae-noise')
const flowFixtureInput = requiredElement<HTMLInputElement>('#flow-fixture')
const runButton = requiredElement<HTMLButtonElement>('#run')
const cancelButton = requiredElement<HTMLButtonElement>('#cancel')
const statusElement = requiredElement<HTMLElement>('[data-testid="e2e-status"]')
const errorElement = requiredElement<HTMLPreElement>('[data-testid="e2e-error"]')
const resultElement = requiredElement<HTMLPreElement>('[data-testid="e2e-result"]')
const previewMount = requiredElement<HTMLElement>('#e2e-preview-root')
const previewStatusElement = requiredElement<HTMLElement>(
  '[data-testid="e2e-viewer-status"]',
)

const previewRoot: Root = createRoot(previewMount)
const injectedControls = createInjectedControls()

let activeController: AbortController | undefined
let activeModel: TripoSplatWebGPU | undefined
let activePlyUrl: string | undefined
let downloadablePly: Blob | undefined
let downloadableSplat: Blob | undefined
let previewGenerationKey = 0
let runCounter = 0
let busy = false
let pageClosing = false

const runHistory: RunHistoryRecord[] = []
window.__TRIPOSPLAT_E2E_HISTORY__ = runHistory

function createInjectedControls(): InjectedControls {
  const panel = document.createElement('section')
  panel.id = 'e2e-generation-controls'
  panel.setAttribute('aria-label', 'Generation comparison controls')
  panel.style.display = 'grid'
  panel.style.gap = '12px'
  panel.style.margin = '16px 0'
  panel.style.padding = '16px'
  panel.style.border = '1px solid currentColor'
  panel.style.borderRadius = '8px'

  const heading = document.createElement('h2')
  heading.textContent = 'Generation comparison'
  heading.style.margin = '0'
  heading.style.fontSize = '1rem'
  panel.appendChild(heading)

  const fields = document.createElement('div')
  fields.style.display = 'grid'
  fields.style.gridTemplateColumns = 'repeat(auto-fit, minmax(140px, 1fr))'
  fields.style.gap = '12px'

  const stepsInput = createNumberInput(
    fields,
    'Sampling steps',
    'e2e-steps',
    DEFAULT_GENERATION_SETTINGS.steps,
    1,
    100,
    1,
  )

  const guidanceInput = createNumberInput(
    fields,
    'Guidance scale',
    'e2e-guidance-scale',
    DEFAULT_GENERATION_SETTINGS.guidanceScale,
    0,
    30,
    0.1,
  )

  const shiftInput = createNumberInput(
    fields,
    'Shift',
    'e2e-shift',
    DEFAULT_GENERATION_SETTINGS.shift,
    0,
    30,
    0.1,
  )

  const seedInput = createNumberInput(
    fields,
    'Seed',
    'e2e-seed',
    DEFAULT_GENERATION_SETTINGS.seed,
    0,
    2_147_483_647,
    1,
  )

  panel.appendChild(fields)

  const presets = document.createElement('div')
  presets.style.display = 'flex'
  presets.style.flexWrap = 'wrap'
  presets.style.gap = '8px'

  const presetLabel = document.createElement('span')
  presetLabel.textContent = 'Step presets:'
  presets.appendChild(presetLabel)

  for (const steps of STEP_PRESETS) {
    const button = document.createElement('button')
    button.type = 'button'
    button.textContent = String(steps)
    button.dataset.stepsPreset = String(steps)

    button.addEventListener('click', () => {
      stepsInput.value = String(steps)
    })

    presets.appendChild(button)
  }

  panel.appendChild(presets)

  const actions = document.createElement('div')
  actions.style.display = 'flex'
  actions.style.flexWrap = 'wrap'
  actions.style.gap = '8px'

  const downloadPlyButton = document.createElement('button')
  downloadPlyButton.type = 'button'
  downloadPlyButton.textContent = 'Download current PLY'
  downloadPlyButton.disabled = true

  const downloadSplatButton = document.createElement('button')
  downloadSplatButton.type = 'button'
  downloadSplatButton.textContent = 'Download current .splat'
  downloadSplatButton.disabled = true

  const clearHistoryButton = document.createElement('button')
  clearHistoryButton.type = 'button'
  clearHistoryButton.textContent = 'Clear comparison history'

  actions.append(
    downloadPlyButton,
    downloadSplatButton,
    clearHistoryButton,
  )

  panel.appendChild(actions)

  const storageStatus = document.createElement('div')
  storageStatus.dataset.testid = 'e2e-storage-status'
  storageStatus.textContent = 'Storage status has not been measured.'
  panel.appendChild(storageStatus)

  const imageStatus = document.createElement('div')
  imageStatus.dataset.testid = 'e2e-image-status'
  imageStatus.textContent = 'Prepared image has not been inspected.'
  panel.appendChild(imageStatus)

  const historyContainer = document.createElement('div')
  historyContainer.style.overflowX = 'auto'

  const historyTable = document.createElement('table')
  historyTable.style.width = '100%'
  historyTable.style.borderCollapse = 'collapse'
  historyTable.dataset.testid = 'e2e-run-history'

  const historyHead = document.createElement('thead')
  historyHead.innerHTML = `
    <tr>
      <th>Run</th>
      <th>Result</th>
      <th>Steps</th>
      <th>Guidance</th>
      <th>Shift</th>
      <th>Seed</th>
      <th>Gaussians</th>
      <th>Generate</th>
      <th>Total</th>
      <th>PLY hash</th>
      <th>.splat hash</th>
    </tr>
  `

  const historyBody = document.createElement('tbody')
  historyTable.append(historyHead, historyBody)
  historyContainer.appendChild(historyTable)
  panel.appendChild(historyContainer)

  const insertionParent = runButton.parentElement ?? runButton
  insertionParent.insertAdjacentElement('afterend', panel)

  downloadPlyButton.addEventListener('click', () => {
    if (downloadablePly !== undefined) {
      downloadBlob(
        downloadablePly,
        buildExportFilename('ply'),
      )
    }
  })

  downloadSplatButton.addEventListener('click', () => {
    if (downloadableSplat !== undefined) {
      downloadBlob(
        downloadableSplat,
        buildExportFilename('splat'),
      )
    }
  })

  clearHistoryButton.addEventListener('click', () => {
    runHistory.splice(0, runHistory.length)
    window.__TRIPOSPLAT_E2E_HISTORY__ = runHistory
    renderRunHistory(historyBody)
  })

  return {
    stepsInput,
    guidanceInput,
    shiftInput,
    seedInput,
    downloadPlyButton,
    downloadSplatButton,
    clearHistoryButton,
    storageStatus,
    imageStatus,
    historyBody,
  }
}

function createNumberInput(
  parent: HTMLElement,
  labelText: string,
  id: string,
  value: number,
  min: number,
  max: number,
  step: number,
): HTMLInputElement {
  const label = document.createElement('label')
  label.htmlFor = id
  label.style.display = 'grid'
  label.style.gap = '4px'

  const labelTextElement = document.createElement('span')
  labelTextElement.textContent = labelText

  const input = document.createElement('input')
  input.id = id
  input.type = 'number'
  input.value = String(value)
  input.min = String(min)
  input.max = String(max)
  input.step = String(step)

  label.append(labelTextElement, input)
  parent.appendChild(label)

  return input
}

function environment(): E2ELabResult['environment'] {
  return {
    userAgent: navigator.userAgent,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: 'gpu' in navigator,
  }
}

function normalizeDirectory(value: string): string {
  return value.replace(/\/+$/, '')
}

function requiredInputValue(
  input: HTMLInputElement | HTMLSelectElement,
  label: string,
): string {
  const value = input.value.trim()

  if (value.length === 0) {
    throw new Error(`${label} must not be empty.`)
  }

  return value
}

function parseFiniteNumber(
  input: HTMLInputElement,
  label: string,
  minimum: number,
  maximum: number,
): number {
  const value = Number(input.value)

  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number.`)
  }

  if (value < minimum || value > maximum) {
    throw new Error(
      `${label} must be between ${minimum} and ${maximum}.`,
    )
  }

  return value
}

function parseInteger(
  input: HTMLInputElement,
  label: string,
  minimum: number,
  maximum: number,
): number {
  const value = parseFiniteNumber(
    input,
    label,
    minimum,
    maximum,
  )

  if (!Number.isInteger(value)) {
    throw new Error(`${label} must be an integer.`)
  }

  return value
}

function selectedGenerationParameters(): GenerationParameters {
  return {
    steps: parseInteger(
      injectedControls.stepsInput,
      'Sampling steps',
      1,
      100,
    ),
    guidanceScale: parseFiniteNumber(
      injectedControls.guidanceInput,
      'Guidance scale',
      0,
      30,
    ),
    shift: parseFiniteNumber(
      injectedControls.shiftInput,
      'Shift',
      0,
      30,
    ),
    seed: parseInteger(
      injectedControls.seedInput,
      'Seed',
      0,
      2_147_483_647,
    ),
  }
}

function selectedCache(): CacheBackend {
  const selected = cacheInput.value

  if (
    selected === 'none'
    || selected === 'opfs'
    || selected === 'cache-api'
  ) {
    return selected
  }

  throw new Error(`Unsupported cache selection '${selected}'.`)
}

function configureDefaultCache(): void {
  const hasOpfs =
    navigator.storage !== undefined
    && typeof navigator.storage.getDirectory === 'function'

  if (hasOpfs) {
    cacheInput.value = 'opfs'
    return
  }

  if ('caches' in window) {
    cacheInput.value = 'cache-api'
    return
  }

  cacheInput.value = 'none'
}

async function prepareStorage(
  cache: CacheBackend,
): Promise<StorageQualification | undefined> {
  if (cache !== 'opfs' || navigator.storage === undefined) {
    injectedControls.storageStatus.textContent =
      `Cache backend: ${cache}. OPFS persistence was not requested.`

    return undefined
  }

  const result: StorageQualification = {
    persistenceRequested: false,
  }

  if (typeof navigator.storage.persisted === 'function') {
    result.persisted = await navigator.storage.persisted()
      .catch(() => false)
  }

  if (
    result.persisted !== true
    && typeof navigator.storage.persist === 'function'
  ) {
    result.persistenceRequested = true
    result.persisted = await navigator.storage.persist()
      .catch(() => false)
  }

  await updateStorageEstimate(result)
  renderStorageStatus(result)

  return result
}

async function updateStorageEstimate(
  storage: StorageQualification,
): Promise<void> {
  if (
    navigator.storage === undefined
    || typeof navigator.storage.estimate !== 'function'
  ) {
    return
  }

  const estimate = await navigator.storage.estimate()
    .catch(() => undefined)

  if (estimate?.usage !== undefined) {
    storage.usageBytes = estimate.usage
  }

  if (estimate?.quota !== undefined) {
    storage.quotaBytes = estimate.quota
  }
}

function renderStorageStatus(
  storage: StorageQualification,
): void {
  const persistence =
    storage.persisted === true
      ? 'persistent'
      : storage.persisted === false
        ? 'not guaranteed persistent'
        : 'persistence unknown'

  const usage =
    storage.usageBytes === undefined
      ? 'usage unknown'
      : `${formatBytes(storage.usageBytes)} used`

  const quota =
    storage.quotaBytes === undefined
      ? 'quota unknown'
      : `${formatBytes(storage.quotaBytes)} quota`

  injectedControls.storageStatus.textContent =
    `OPFS cache: ${persistence}; ${usage}; ${quota}.`
}

async function fetchRequired(
  url: string,
  signal: AbortSignal,
): Promise<Response> {
  if (url.length === 0) {
    throw new Error('Attempted to fetch an empty URL.')
  }

  const response = await fetch(url, {
    signal,
    cache: 'no-cache',
  })

  if (!response.ok) {
    throw new Error(
      `Could not fetch ${url}: HTTP ${response.status} ${response.statusText}.`,
    )
  }

  return response
}

async function fetchFloat32(
  url: string,
  expectedElements: number,
  signal: AbortSignal,
): Promise<Float32Array> {
  const response = await fetchRequired(url, signal)
  const bytes = await response.arrayBuffer()
  const expectedBytes =
    expectedElements * Float32Array.BYTES_PER_ELEMENT

  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `${url} has ${bytes.byteLength.toLocaleString()} bytes; `
      + `expected ${expectedBytes.toLocaleString()} bytes `
      + `(${expectedElements.toLocaleString()} Float32 values).`,
    )
  }

  return new Float32Array(bytes)
}

async function inspectPreparedImage(
  blob: Blob,
  url: string,
): Promise<PreparedImageMetadata> {
  if (blob.size === 0) {
    throw new Error(`Prepared image fixture '${url}' is empty.`)
  }

  if (blob.type.length > 0 && !blob.type.startsWith('image/')) {
    throw new Error(
      `Prepared image fixture '${url}' has unexpected MIME type '${blob.type}'.`,
    )
  }

  const bitmap = await createImageBitmap(blob)

  try {
    if (bitmap.width === 0 || bitmap.height === 0) {
      throw new Error(
        `Prepared image fixture '${url}' decoded with zero dimensions.`,
      )
    }

    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height

    const context = canvas.getContext('2d', {
      willReadFrequently: true,
    })

    if (context === null) {
      throw new Error(
        'Could not create a 2D canvas context to inspect the prepared image.',
      )
    }

    context.drawImage(bitmap, 0, 0)

    const pixels = context.getImageData(
      0,
      0,
      bitmap.width,
      bitmap.height,
    ).data

    let hasAlpha = false

    for (let index = 3; index < pixels.length; index += 4) {
      if (pixels[index] !== 255) {
        hasAlpha = true
        break
      }
    }

    const metadata: PreparedImageMetadata = {
      mimeType: blob.type || 'unknown',
      byteLength: blob.size,
      width: bitmap.width,
      height: bitmap.height,
      hasAlpha,
    }

    injectedControls.imageStatus.textContent =
      `Prepared image: ${metadata.width}×${metadata.height}, `
      + `${metadata.mimeType}, ${formatBytes(metadata.byteLength)}, `
      + `${metadata.hasAlpha ? 'contains transparency' : 'fully opaque'}.`

    return metadata
  } finally {
    bitmap.close()
  }
}

async function sha256(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    await blob.arrayBuffer(),
  )

  return Array.from(
    new Uint8Array(digest),
    (value) => value.toString(16).padStart(2, '0'),
  ).join('')
}

function formatBytes(bytes: number): string {
  if (bytes < 1_024) {
    return `${bytes} B`
  }

  if (bytes < 1_024 * 1_024) {
    return `${(bytes / 1_024).toFixed(1)} KiB`
  }

  if (bytes < 1_024 * 1_024 * 1_024) {
    return `${(bytes / (1_024 * 1_024)).toFixed(1)} MiB`
  }

  return `${(bytes / (1_024 * 1_024 * 1_024)).toFixed(2)} GiB`
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1_000) {
    return `${milliseconds.toFixed(0)} ms`
  }

  return `${(milliseconds / 1_000).toFixed(2)} s`
}

function shortHash(hash: string): string {
  return hash.slice(0, 12)
}

function buildExportFilename(
  extension: 'ply' | 'splat',
): string {
  const parameters = selectedGenerationParameters()

  return [
    'triposplat',
    `${parameters.steps}steps`,
    `g${parameters.guidanceScale}`,
    `shift${parameters.shift}`,
    `seed${parameters.seed}`,
  ].join('-') + `.${extension}`
}

function downloadBlob(
  blob: Blob,
  filename: string,
): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')

  anchor.href = url
  anchor.download = filename
  anchor.style.display = 'none'

  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()

  window.setTimeout(() => {
    URL.revokeObjectURL(url)
  }, 0)
}

function setDownloadableExports(
  ply: Blob | undefined,
  splat: Blob | undefined,
): void {
  downloadablePly = ply
  downloadableSplat = splat

  injectedControls.downloadPlyButton.disabled =
    ply === undefined

  injectedControls.downloadSplatButton.disabled =
    splat === undefined
}

function setControlsDisabled(disabled: boolean): void {
  injectedControls.stepsInput.disabled = disabled
  injectedControls.guidanceInput.disabled = disabled
  injectedControls.shiftInput.disabled = disabled
  injectedControls.seedInput.disabled = disabled

  for (const button of document.querySelectorAll<HTMLButtonElement>(
    '[data-steps-preset]',
  )) {
    button.disabled = disabled
  }

  injectedControls.clearHistoryButton.disabled = disabled
}

function setPreviewStatus(
  state: 'waiting' | 'loading' | 'ready' | 'failed',
  message: string,
): void {
  previewStatusElement.dataset.viewerState = state
  previewStatusElement.textContent = message
}

function releaseActivePreview(): void {
  if (!pageClosing) {
    flushSync(() => {
      previewRoot.render(null)
    })
  }

  if (activePlyUrl !== undefined) {
    URL.revokeObjectURL(activePlyUrl)
    activePlyUrl = undefined
  }
}

function resetPreview(): void {
  releaseActivePreview()

  setPreviewStatus(
    'waiting',
    'Viewer is waiting for an exported PLY.',
  )
}

function signalError(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new DOMException(
      'The operation was aborted.',
      'AbortError',
    )
}

function observeViewerCanvas(): GaussianViewerObservation['canvas'] {
  const canvas = previewMount.querySelector<HTMLCanvasElement>('canvas')

  return {
    present: canvas !== null,
    width: canvas?.width ?? 0,
    height: canvas?.height ?? 0,
    clientWidth: canvas?.clientWidth ?? 0,
    clientHeight: canvas?.clientHeight ?? 0,
  }
}

async function loadPlyInViewer(
  ply: Blob,
  signal: AbortSignal,
): Promise<GaussianViewerQualification> {
  if (signal.aborted) {
    throw signalError(signal)
  }

  resetPreview()

  activePlyUrl = URL.createObjectURL(ply)
  previewGenerationKey += 1

  const plyUrl = activePlyUrl
  const generationKey = previewGenerationKey
  const startedAt = performance.now()

  const observation = await new Promise<GaussianViewerObservation>(
    (resolve, reject) => {
      let settled = false
      let layoutFrame: number | undefined

      const cleanup = () => {
        window.clearTimeout(timeout)
        signal.removeEventListener('abort', onAbort)

        if (layoutFrame !== undefined) {
          cancelAnimationFrame(layoutFrame)
          layoutFrame = undefined
        }
      }

      const complete = (
        status: GaussianViewerObservation['status'],
        message: string,
        afterLayout = false,
      ) => {
        if (settled) {
          return
        }

        settled = true
        cleanup()
        setPreviewStatus(status, message)

        const finish = () => {
          layoutFrame = undefined

          if (signal.aborted) {
            reject(signalError(signal))
            return
          }

          resolve({
            status,
            message,
            loadMs: performance.now() - startedAt,
            timeoutMs: VIEWER_LOAD_TIMEOUT_MS,
            canvas: observeViewerCanvas(),
          })
        }

        if (afterLayout) {
          layoutFrame = requestAnimationFrame(finish)
        } else {
          finish()
        }
      }

      const onStatus = (status: SplatPreviewStatus) => {
        if (settled) {
          return
        }

        setPreviewStatus(status.state, status.message)

        if (status.state === 'ready') {
          complete('ready', status.message, true)
        } else if (status.state === 'failed') {
          complete('failed', status.message)
        }
      }

      const onAbort = () => {
        if (settled) {
          return
        }

        settled = true
        cleanup()
        setPreviewStatus('failed', 'Viewer load aborted.')
        reject(signalError(signal))
      }

      const timeout = window.setTimeout(() => {
        complete(
          'failed',
          'Preview failed: viewer did not reach ready or failed within '
          + `${VIEWER_LOAD_TIMEOUT_MS} ms.`,
        )
      }, VIEWER_LOAD_TIMEOUT_MS)

      signal.addEventListener('abort', onAbort, {
        once: true,
      })

      try {
        previewRoot.render(
          createElement(SplatPreview, {
            plyUrl,
            generationKey,
            bgColor: '#000000',
            fov: 45,
            autoRotate: false,
            maxScreenSize: 1024,
            splatPosition: [0, 0, 0],
            splatRotation: [0, 0, 0],
            splatFlip: [false, false, false],
            onViewerStateChange: onStatus,
          }),
        )
      } catch (error) {
        complete(
          'failed',
          `Preview failed: ${
            error instanceof Error
              ? error.message
              : String(error)
          }`,
        )
      }
    },
  )

  return qualifyGaussianViewer(observation)
}

async function qualifyExports(
  ply: Blob,
  splat: Blob,
  count: number,
  plyHash: string,
  splatHash: string,
): Promise<ExportQualification> {
  const headerPrefix = await ply.slice(0, 4096).text()
  const marker = 'end_header\n'
  const markerIndex = headerPrefix.indexOf(marker)

  const markerEnd =
    markerIndex >= 0
      ? markerIndex + marker.length
      : -1

  const headerBytes =
    markerEnd >= 0
      ? new TextEncoder()
        .encode(headerPrefix.slice(0, markerEnd))
        .byteLength
      : 0

  const expectedPlyBytes =
    headerBytes
    + count
    * PLY_FLOATS_PER_GAUSSIAN
    * Float32Array.BYTES_PER_ELEMENT

  const expectedSplatBytes =
    count * SPLAT_BYTES_PER_GAUSSIAN

  const plySummary = {
    byteLength: ply.size,
    sha256: plyHash,
    headerBytes,
    expectedByteLength: expectedPlyBytes,
    binaryLittleEndian: headerPrefix.startsWith(
      'ply\nformat binary_little_endian 1.0\n',
    ),
    vertexCountMatches: headerPrefix.includes(
      `element vertex ${count}\n`,
    ),
    byteLengthMatches:
      headerBytes > 0
      && ply.size === expectedPlyBytes,
  }

  const splatSummary = {
    byteLength: splat.size,
    sha256: splatHash,
    expectedByteLength: expectedSplatBytes,
    byteLengthMatches:
      splat.size === expectedSplatBytes,
  }

  return {
    passed:
      plySummary.binaryLittleEndian
      && plySummary.vertexCountMatches
      && plySummary.byteLengthMatches
      && splatSummary.byteLengthMatches,
    ply: plySummary,
    splat: splatSummary,
  }
}

function createProgressRecorder(startedAt: number): {
  records: ProgressRecord[]
  load: (progress: LoadProgress) => void
  generate: (progress: GenerationProgress) => void
} {
  const records: ProgressRecord[] = []
  let previousKey = ''

  const record = (
    phase: ProgressRecord['phase'],
    update: LoadProgress | GenerationProgress,
  ) => {
    const bucket =
      update.progress === undefined
        ? ''
        : Math.floor(update.progress * 20)

    const generation = update as GenerationProgress
    const load = update as LoadProgress

    const key = [
      phase,
      update.stage,
      load.graph ?? '',
      generation.invocation ?? '',
      generation.step ?? '',
      bucket,
    ].join(':')

    const next: ProgressRecord = {
      phase,
      elapsedMs: performance.now() - startedAt,
      stage: update.stage,
      message: update.message,
    }

    if (update.progress !== undefined) {
      next.progress = update.progress
    }

    if (update.loadedBytes !== undefined) {
      next.loadedBytes = update.loadedBytes
    }

    if (update.totalBytes !== undefined) {
      next.totalBytes = update.totalBytes
    }

    if (load.graph !== undefined) {
      next.graph = load.graph
    }

    if (generation.step !== undefined) {
      next.step = generation.step
    }

    if (generation.totalSteps !== undefined) {
      next.totalSteps = generation.totalSteps
    }

    if (generation.invocation !== undefined) {
      next.invocation = generation.invocation
    }

    if (generation.totalInvocations !== undefined) {
      next.totalInvocations = generation.totalInvocations
    }

    if (key === previousKey && records.length > 0) {
      records[records.length - 1] = next
    } else if (records.length < MAX_PROGRESS_RECORDS) {
      records.push(next)
    }

    previousKey = key
    statusElement.textContent = update.message
  }

  return {
    records,
    load: (progress) => record('load', progress),
    generate: (progress) => record('generate', progress),
  }
}

function publish(result: E2ELabResult): void {
  window.__TRIPOSPLAT_E2E_RESULT__ = result
  resultElement.textContent = JSON.stringify(result, null, 2)
  resultElement.hidden = false

  if (result.completed) {
    if (result.passed) {
      statusElement.textContent =
        'PASS: the browser-local pipeline produced '
        + `${result.gaussianCount.toLocaleString()} finite Gaussians `
        + `using ${result.generation.steps} steps, valid-size exports, `
        + 'and a ready Gaussian viewer canvas.'
    } else {
      statusElement.textContent =
        'FAIL: end-to-end execution completed, but a scene, export, '
        + 'or viewer sanity check failed.'
    }

    return
  }

  statusElement.textContent =
    `E2E run did not complete: ${result.error.message}`
}

function serializeError(
  value: unknown,
  depth = 0,
): SerializedError {
  const baseError =
    value instanceof Error
      ? value
      : new Error(String(value))

  const error = errorWithDetails(baseError)

  const serialized: SerializedError = {
    name: error.name,
    message: error.message,
  }

  if (error.stack !== undefined) {
    serialized.stack = error.stack
  }

  if (typeof error.code === 'string') {
    serialized.code = error.code
  }

  if (typeof error.stage === 'string') {
    serialized.stage = error.stage
  }

  if (error.diagnostics !== undefined) {
    serialized.diagnostics = error.diagnostics
  }

  if (error.cause !== undefined && depth < 6) {
    serialized.cause = serializeError(
      error.cause,
      depth + 1,
    )
  }

  return serialized
}

function errorWithDetails(value: Error): Error & {
  code?: unknown
  stage?: unknown
  diagnostics?: unknown
  cause?: unknown
} {
  return value
}

async function disposeModel(
  model: TripoSplatWebGPU | undefined,
): Promise<boolean> {
  if (model === undefined) {
    return true
  }

  await model.dispose()
  return true
}

function addRunHistoryRecord(
  result: E2ELabSuccess,
): void {
  runCounter += 1

  runHistory.unshift({
    runNumber: runCounter,
    timestamp: new Date().toISOString(),
    passed: result.passed,
    steps: result.generation.steps,
    guidanceScale: result.generation.guidanceScale,
    shift: result.generation.shift,
    seed: result.generation.seed,
    gaussianCount: result.gaussianCount,
    fixtureFetchMs: result.timingsMs.fixtureFetch,
    loadMs: result.timingsMs.load,
    generateMs: result.timingsMs.generate,
    totalMs: result.timingsMs.total,
    plyBytes: result.exports.ply.byteLength,
    splatBytes: result.exports.splat.byteLength,
    plySha256: result.exports.ply.sha256,
    splatSha256: result.exports.splat.sha256,
  })

  if (runHistory.length > MAX_RUN_HISTORY) {
    runHistory.length = MAX_RUN_HISTORY
  }

  window.__TRIPOSPLAT_E2E_HISTORY__ = runHistory
  renderRunHistory(injectedControls.historyBody)
}

function renderRunHistory(
  body: HTMLTableSectionElement,
): void {
  body.replaceChildren()

  for (const record of runHistory) {
    const row = document.createElement('tr')

    appendHistoryCell(row, String(record.runNumber))
    appendHistoryCell(row, record.passed ? 'PASS' : 'FAIL')
    appendHistoryCell(row, String(record.steps))
    appendHistoryCell(row, String(record.guidanceScale))
    appendHistoryCell(row, String(record.shift))
    appendHistoryCell(row, String(record.seed))
    appendHistoryCell(
      row,
      record.gaussianCount.toLocaleString(),
    )
    appendHistoryCell(
      row,
      formatDuration(record.generateMs),
    )
    appendHistoryCell(
      row,
      formatDuration(record.totalMs),
    )
    appendHistoryCell(
      row,
      shortHash(record.plySha256),
      record.plySha256,
    )
    appendHistoryCell(
      row,
      shortHash(record.splatSha256),
      record.splatSha256,
    )

    body.appendChild(row)
  }
}

function appendHistoryCell(
  row: HTMLTableRowElement,
  text: string,
  title?: string,
): void {
  const cell = document.createElement('td')
  cell.textContent = text
  cell.style.padding = '6px'
  cell.style.borderTop = '1px solid currentColor'

  if (title !== undefined) {
    cell.title = title
  }

  row.appendChild(cell)
}

async function run(): Promise<void> {
  if (busy) {
    return
  }

  busy = true
  runButton.disabled = true
  cancelButton.disabled = false
  errorElement.hidden = true
  resultElement.hidden = true
  delete window.__TRIPOSPLAT_E2E_RESULT__

  setControlsDisabled(true)
  setDownloadableExports(undefined, undefined)
  resetPreview()

  const startedAt = performance.now()
  const controller = new AbortController()
  activeController = controller
  const progress = createProgressRecorder(startedAt)

  let model: TripoSplatWebGPU | undefined
  let scene: GaussianScene | undefined
  let sceneDisposed = false
  let modelDisposed = false

  try {
    if (!('gpu' in navigator)) {
      throw new Error(
        'WebGPU is unavailable in this browser.',
      )
    }

    const generation = selectedGenerationParameters()

    const modelBaseUrl = normalizeDirectory(
      requiredInputValue(
        modelBaseInput,
        'Model base URL',
      ),
    )

    const preparedImageUrl = requiredInputValue(
      preparedImageInput,
      'Prepared image URL',
    )

    const vaeNoiseUrl = requiredInputValue(
      vaeNoiseInput,
      'VAE noise URL',
    )

    const flowBase = normalizeDirectory(
      requiredInputValue(
        flowFixtureInput,
        'Flow fixture directory',
      ),
    )

    const latentNoiseUrl = `${flowBase}/latent.f32`
    const cameraNoiseUrl = `${flowBase}/camera.f32`
    const cache = selectedCache()

    statusElement.textContent =
      cache === 'opfs'
        ? 'Preparing persistent OPFS model cache…'
        : `Preparing ${cache} model cache…`

    const storage = await prepareStorage(cache)
    const fixtureStartedAt = performance.now()

    statusElement.textContent =
      'Fetching prepared image and deterministic noise fixtures…'

    const [
      imageResponse,
      vaeNoise,
      latentNoise,
      cameraNoise,
    ] = await Promise.all([
      fetchRequired(
        preparedImageUrl,
        controller.signal,
      ),
      fetchFloat32(
        vaeNoiseUrl,
        FLOAT_COUNTS.vaeNoise,
        controller.signal,
      ),
      fetchFloat32(
        latentNoiseUrl,
        FLOAT_COUNTS.latentNoise,
        controller.signal,
      ),
      fetchFloat32(
        cameraNoiseUrl,
        FLOAT_COUNTS.cameraNoise,
        controller.signal,
      ),
    ])

    const preparedImage = await imageResponse.blob()

    const preparedImageMetadata =
      await inspectPreparedImage(
        preparedImage,
        preparedImageUrl,
      )

    const fixtureFetchMs =
      performance.now() - fixtureStartedAt

    /*
     * The package dist entry owns production worker creation.
     *
     * Do not point this lab at:
     * ../packages/triposplat-webgpu/src/worker.ts
     *
     * Doing that would test the workspace source worker instead of the
     * packaged worker shipped by the installed distribution.
     */
    model = new TripoSplatWebGPU({
      modelBaseUrl,
      executionProviders: ['webgpu'],
      cache,
      logLevel: 'info',
      wasmPaths: {
        mjs: '/ort/ort-wasm-simd-threaded.asyncify.mjs',
        wasm: '/ort/ort-wasm-simd-threaded.asyncify.wasm',
      },
    })

    activeModel = model

    const loadStartedAt = performance.now()

    await model.load({
      signal: controller.signal,
      onProgress: progress.load,
    })

    const loadMs =
      performance.now() - loadStartedAt

    const generationStartedAt = performance.now()

    scene = await model.generate(preparedImage, {
      steps: generation.steps,
      guidanceScale: generation.guidanceScale,
      shift: generation.shift,
      seed: generation.seed,
      inputIsPrepared: true,
      vaeNoise,
      latentNoise,
      cameraNoise,
      signal: controller.signal,
      onProgress: progress.generate,
    })

    const generateMs =
      performance.now() - generationStartedAt

    const gaussianCount = scene.count

    const sceneQualification =
      qualifyGaussianSceneStructure(scene)

    statusElement.textContent =
      'Encoding and hashing binary PLY…'

    const plyStartedAt = performance.now()
    const ply = await scene.exportPLY()
    const plyHash = await sha256(ply)

    const plyExportAndHashMs =
      performance.now() - plyStartedAt

    statusElement.textContent =
      'Encoding and hashing browser .splat…'

    const splatStartedAt = performance.now()
    const splat = await scene.exportSplat()
    const splatHash = await sha256(splat)

    const splatExportAndHashMs =
      performance.now() - splatStartedAt

    const exportsQualification = await qualifyExports(
      ply,
      splat,
      gaussianCount,
      plyHash,
      splatHash,
    )

    setDownloadableExports(ply, splat)

    statusElement.textContent =
      'Loading the exported PLY in the browser Gaussian viewer…'

    const viewerQualification = await loadPlyInViewer(
      ply,
      controller.signal,
    )

    const measuredGenerationSettings =
      scene.metadata.generationSettings

    const manifest = model.manifest

    const configuredGraphs =
      model.capabilities.configuredGraphs

    scene.dispose()
    sceneDisposed = scene.isDisposed

    modelDisposed = await disposeModel(model)

    activeModel = undefined
    model = undefined

    if (storage !== undefined) {
      await updateStorageEstimate(storage)
      renderStorageStatus(storage)
    }

    const result: E2ELabSuccess = {
      passed:
        sceneQualification.passed
        && exportsQualification.passed
        && viewerQualification.passed,
      completed: true,
      qualificationScope:
        'end-to-end-execution-structure-export-and-viewer-sanity',
      numericalParityClaimed: false,
      model: {
        baseUrl: modelBaseUrl,
        configuredGraphs,
        cache,
      },
      fixtures: {
        preparedImage: preparedImageUrl,
        preparedImageMetadata,
        vaeNoise: vaeNoiseUrl,
        latentNoise: latentNoiseUrl,
        cameraNoise: cameraNoiseUrl,
        explicitDeterministicNoise: true,
      },
      generation,
      scene: sceneQualification,
      gaussianCount,
      exports: exportsQualification,
      viewer: viewerQualification,
      timingsMs: {
        fixtureFetch: fixtureFetchMs,
        load: loadMs,
        generate: generateMs,
        plyExportAndHash:
          plyExportAndHashMs,
        splatExportAndHash:
          splatExportAndHashMs,
        viewerLoad:
          viewerQualification.loadMs,
        total:
          performance.now() - startedAt,
      },
      pipelineMeasuredTimingsMs:
        measuredGenerationSettings.measuredTimingsMs,
      progress: progress.records,
      lifecycle: {
        sceneDisposed,
        modelDisposed,
      },
      storage,
      environment: environment(),
    }

    if (manifest !== undefined) {
      result.model.manifestVersion =
        manifest.version

      result.model.modelRevision =
        manifest.modelRevision

      if (
        manifest.estimatedModelBytes !== undefined
      ) {
        result.model.declaredModelBytes =
          manifest.estimatedModelBytes
      }
    }

    addRunHistoryRecord(result)
    publish(result)
  } catch (caught) {
    if (scene !== undefined && !scene.isDisposed) {
      scene.dispose()
      sceneDisposed = scene.isDisposed
    }

    if (model !== undefined) {
      modelDisposed = await disposeModel(model)
        .catch(() => false)
    }

    activeModel = undefined

    const error =
      caught instanceof Error
        ? caught
        : new Error(String(caught))

    releaseActivePreview()

    setPreviewStatus(
      'failed',
      'Viewer unavailable because the E2E run stopped: '
      + error.message,
    )

    const serialized = serializeError(error)

    const result: E2ELabFailure = {
      passed: false,
      completed: false,
      qualificationScope:
        'end-to-end-execution-structure-export-and-viewer-sanity',
      numericalParityClaimed: false,
      error: serialized,
      progress: progress.records,
      timingsMs: {
        total:
          performance.now() - startedAt,
      },
      environment: environment(),
    }

    errorElement.textContent =
      `${error.name}: ${error.message}`

    errorElement.hidden = false
    publish(result)

    void sceneDisposed
    void modelDisposed
  } finally {
    if (activeController === controller) {
      activeController = undefined
    }

    busy = false
    runButton.disabled = false
    cancelButton.disabled = true
    setControlsDisabled(false)
  }
}

configureDefaultCache()
renderRunHistory(injectedControls.historyBody)

runButton.addEventListener('click', () => {
  void run()
})

cancelButton.addEventListener('click', () => {
  cancelButton.disabled = true

  statusElement.textContent =
    'Cancelling after the active ONNX invocation returns…'

  activeController?.abort(
    new DOMException(
      'Cancelled from the E2E lab.',
      'AbortError',
    ),
  )
})

document.documentElement.dataset.e2eBootstrap = 'ready'

window.addEventListener('beforeunload', () => {
  pageClosing = true

  activeController?.abort(
    new DOMException(
      'E2E lab page is closing.',
      'AbortError',
    ),
  )

  void activeModel?.dispose()

  previewRoot.unmount()

  if (activePlyUrl !== undefined) {
    URL.revokeObjectURL(activePlyUrl)
    activePlyUrl = undefined
  }

  downloadablePly = undefined
  downloadableSplat = undefined
})