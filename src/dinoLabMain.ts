import {
  buildTripoSplatEncoderTensors,
  compositeRgbaOnBlack,
  imageBitmapToRgba,
} from './models/triposplat/preprocess'
import { compareFloat32, type TensorComparison } from './models/triposplat/tensorMath'
import { OrtWorkerClient } from './runtime/OrtWorkerClient'
import { createTensorPayload, type TensorPayload } from './runtime/tensors'

const SESSION_ID = 'triposplat/dinov3-parity'
const SHAPES = {
  pixelValues: [1, 3, 1024, 1024],
  feature1: [1, 4101, 1280],
} as const
interface NumericTolerance {
  absolute: number
  relative: number
  minimumCosineSimilarity: number
}

const DEFAULT_TOLERANCES: { strict: NumericTolerance; qualification: NumericTolerance } = {
  strict: { absolute: 0.03, relative: 0.03, minimumCosineSimilarity: 0.99999 },
  qualification: { absolute: 0.15, relative: 0.03, minimumCosineSimilarity: 0.99999 },
}

interface ToleranceGate extends TensorComparison {
  fractionWithinTolerance: number
  maxErrorIndex: number
  referenceAtMaxError: number
  candidateAtMaxError: number
  passed: boolean
}

interface DinoLabResult {
  passed: boolean
  strictPassed: boolean
  executionProvider: string
  modelLoadMs: number
  modelTransferBytes?: number
  inferenceMs: number
  readbackMs: number
  preprocessing: TensorComparison
  strictComparison: ToleranceGate
  qualificationComparison: ToleranceGate
  tolerances: {
    strict: NumericTolerance
    qualification: NumericTolerance
  }
  environment: { userAgent: string; crossOriginIsolated: boolean; webgpu: boolean }
}

declare global {
  interface Window {
    __TRIPOSPLAT_DINO_RESULT__?: DinoLabResult
  }
}

const modelInput = document.querySelector<HTMLInputElement>('#model')!
const fixtureInput = document.querySelector<HTMLInputElement>('#fixture')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const statusElement = document.querySelector<HTMLElement>('[data-testid="dino-status"]')!
const errorElement = document.querySelector<HTMLPreElement>('[data-testid="dino-error"]')!
const resultElement = document.querySelector<HTMLPreElement>('[data-testid="dino-result"]')!
let activeClient: OrtWorkerClient | undefined
let busy = false

function elementCount(shape: readonly number[]): number {
  return shape.reduce((product, value) => product * value, 1)
}

async function fetchFloat32(url: string, expectedElements: number): Promise<Float32Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`)
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength !== expectedElements * 4) {
    throw new Error(`${url} has ${buffer.byteLength} bytes; expected ${expectedElements * 4}.`)
  }
  return new Float32Array(buffer)
}

async function contentLength(url: string): Promise<number | undefined> {
  try {
    const response = await fetch(url, { method: 'HEAD' })
    if (!response.ok) return undefined
    const value = Number(response.headers.get('content-length'))
    return Number.isFinite(value) && value >= 0 ? value : undefined
  } catch {
    return undefined
  }
}

function payloadFloat32(payload: TensorPayload | undefined): Float32Array {
  if (!payload || payload.type !== 'float32') {
    throw new Error("Output 'feature1' is missing or is not float32.")
  }
  return new Float32Array(payload.data)
}

function gate(
  reference: Float32Array,
  candidate: Float32Array,
  tolerance: NumericTolerance,
): ToleranceGate {
  const comparison = compareFloat32(reference, candidate)
  let within = 0
  let maxError = -1
  let maxErrorIndex = 0
  for (let index = 0; index < reference.length; index += 1) {
    const error = Math.abs(reference[index] - candidate[index])
    if (error <= tolerance.absolute + tolerance.relative * Math.abs(reference[index])) within += 1
    if (error > maxError) {
      maxError = error
      maxErrorIndex = index
    }
  }
  const fractionWithinTolerance = within / reference.length
  return {
    ...comparison,
    fractionWithinTolerance,
    maxErrorIndex,
    referenceAtMaxError: reference[maxErrorIndex],
    candidateAtMaxError: candidate[maxErrorIndex],
    passed: comparison.finite
      && fractionWithinTolerance === 1
      && comparison.cosineSimilarity >= tolerance.minimumCosineSimilarity,
  }
}

function validTolerance(value: unknown): value is Record<string, number> {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return ['absolute', 'relative', 'minimum_cosine_similarity'].every((key) => (
    typeof candidate[key] === 'number' && Number.isFinite(candidate[key]) && candidate[key] >= 0
  ))
}

async function fixtureTolerances(fixtureUrl: string): Promise<DinoLabResult['tolerances']> {
  const response = await fetch(`${fixtureUrl}/manifest.json`)
  if (!response.ok) return DEFAULT_TOLERANCES
  const manifest = await response.json() as {
    tolerances?: { strict?: unknown; qualification?: unknown }
  }
  const strict = manifest.tolerances?.strict
  const qualification = manifest.tolerances?.qualification
  if (!validTolerance(strict) || !validTolerance(qualification)) return DEFAULT_TOLERANCES
  const normalize = (value: Record<string, number>): NumericTolerance => ({
    absolute: value.absolute,
    relative: value.relative,
    minimumCosineSimilarity: value.minimum_cosine_similarity,
  })
  return { strict: normalize(strict), qualification: normalize(qualification) }
}

async function run(): Promise<void> {
  if (busy) return
  busy = true
  runButton.disabled = true
  runButton.textContent = 'Running…'
  errorElement.hidden = true
  resultElement.hidden = true
  delete window.__TRIPOSPLAT_DINO_RESULT__
  try {
    if (activeClient) await activeClient.dispose()
    const modelUrl = modelInput.value
    const fixtureUrl = fixtureInput.value
    const client = new OrtWorkerClient({ onStatus: ({ message }) => { statusElement.textContent = message } })
    activeClient = client
    statusElement.textContent = 'Fetching prepared image and official PyTorch tensors…'
    const [imageResponse, recordedInput, reference, tolerances] = await Promise.all([
      fetch(`${fixtureUrl}/prepared.png`),
      fetchFloat32(`${fixtureUrl}/pixel_values.f32`, elementCount(SHAPES.pixelValues)),
      fetchFloat32(`${fixtureUrl}/feature1.f32`, elementCount(SHAPES.feature1)),
      fixtureTolerances(fixtureUrl),
    ])
    if (!imageResponse.ok) throw new Error(`Could not fetch prepared image: HTTP ${imageResponse.status}`)
    const bitmap = await createImageBitmap(await imageResponse.blob())
    const browserInput = (() => {
      try {
        const rgb = compositeRgbaOnBlack(imageBitmapToRgba(bitmap))
        return buildTripoSplatEncoderTensors(rgb).dinov3.data
      } finally {
        bitmap.close()
      }
    })()
    const preprocessing = compareFloat32(recordedInput, browserInput)

    const sidecarUrl = `${modelUrl}.data`
    const graphName = new URL(modelUrl, document.baseURI).pathname.split('/').at(-1)
    if (!graphName) throw new Error(`Could not derive graph name from ${modelUrl}.`)
    const transferParts = await Promise.all([contentLength(modelUrl), contentLength(sidecarUrl)])
    const modelTransferBytes = transferParts.every((value) => value !== undefined)
      ? transferParts.reduce<number>((sum, value) => sum + (value ?? 0), 0)
      : undefined
    const loaded = await client.loadSession({
      sessionId: SESSION_ID,
      manifest: {
        graphUrl: modelUrl,
        externalData: [{ path: `${decodeURIComponent(graphName)}.data`, url: sidecarUrl }],
      },
      options: { allowWasmFallback: false, graphOptimizationLevel: 'disabled' },
    })
    if (loaded.executionProvider !== 'webgpu') throw new Error(`Expected WebGPU, loaded ${loaded.executionProvider}.`)
    const response = await client.runSession({
      sessionId: SESSION_ID,
      inputs: {
        pixel_values: createTensorPayload('float32', browserInput, SHAPES.pixelValues),
      },
      outputs: ['feature1'],
      tag: 'dinov3-parity',
    })
    const candidate = payloadFloat32(response.outputs.feature1)
    const strictComparison = gate(reference, candidate, tolerances.strict)
    const qualificationComparison = gate(reference, candidate, tolerances.qualification)
    const result: DinoLabResult = {
      passed: preprocessing.finite
        && preprocessing.maxAbsoluteError <= 1e-6
        && qualificationComparison.passed,
      strictPassed: strictComparison.passed,
      executionProvider: loaded.executionProvider,
      modelLoadMs: loaded.loadMs,
      modelTransferBytes,
      inferenceMs: response.timings.inferenceMs,
      readbackMs: response.timings.readbackMs,
      preprocessing,
      strictComparison,
      qualificationComparison,
      tolerances,
      environment: {
        userAgent: navigator.userAgent,
        crossOriginIsolated: self.crossOriginIsolated,
        webgpu: 'gpu' in navigator,
      },
    }
    await client.dispose()
    if (activeClient === client) activeClient = undefined
    window.__TRIPOSPLAT_DINO_RESULT__ = result
    resultElement.textContent = JSON.stringify(result, null, 2)
    resultElement.hidden = false
    statusElement.textContent = result.passed
      ? result.strictPassed
        ? 'PASS: WebGPU DINOv3 matches the strict official gate.'
        : 'PASS: WebGPU DINOv3 matches the recorded qualification envelope; strict gate remains failed.'
      : 'FAIL: WebGPU DINOv3 exceeds the recorded qualification envelope.'
  } catch (caught) {
    const message = caught instanceof Error ? caught.message : String(caught)
    errorElement.textContent = message
    errorElement.hidden = false
    statusElement.textContent = 'DINOv3 validation failed.'
    if (activeClient) await activeClient.dispose().catch(() => undefined)
    activeClient = undefined
  } finally {
    busy = false
    runButton.disabled = false
    runButton.textContent = 'Run DINOv3 parity gate'
  }
}

runButton.addEventListener('click', () => { void run() })
addEventListener('beforeunload', () => { if (activeClient) void activeClient.dispose() })
if (new URLSearchParams(location.search).get('autorun') === '1') queueMicrotask(() => { void run() })

const style = document.createElement('style')
style.textContent = `
  :root { color: #ececf3; background: #101014; font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  body { margin: 0; } main { max-width: 920px; margin: 0 auto; padding: 48px 24px; }
  h1 { font: 600 28px/1.2 system-ui, sans-serif; } label { display: grid; gap: 6px; margin: 18px 0; }
  input { box-sizing: border-box; width: 100%; padding: 10px; color: inherit; background: #1b1b22; border: 1px solid #3a3a48; border-radius: 6px; }
  button { padding: 10px 16px; color: #08080a; background: #f8cf00; border: 0; border-radius: 6px; font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .55; cursor: wait; } pre { overflow: auto; padding: 16px; background: #18181f; border-radius: 8px; }
  .error { color: #ff9b9b; }
`
document.head.append(style)
