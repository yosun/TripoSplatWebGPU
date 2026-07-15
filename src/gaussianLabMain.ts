import { compareFloat32, type TensorComparison } from './models/triposplat/tensorMath'
import { OrtWorkerClient } from './runtime/OrtWorkerClient'
import { createTensorPayload, type TensorPayload } from './runtime/tensors'

const SESSION_ID = 'triposplat/gaussian-decoder-parity'
const ATOL = 0.02
const RTOL = 0.01
const MINIMUM_COSINE = 0.9999
const SHAPES = {
  points: [1, 8192, 3],
  cond: [1, 8192, 16],
  features: [1, 8192, 480],
} as const

interface OutputGate extends TensorComparison {
  fractionWithinTolerance: number
  maxErrorIndex: number
  referenceAtMaxError: number
  candidateAtMaxError: number
  passed: boolean
}

interface GaussianLabResult {
  passed: boolean
  executionProvider: string
  modelLoadMs: number
  modelTransferBytes?: number
  inferenceMs: number
  readbackMs: number
  comparison: OutputGate
  tolerance: { absolute: number; relative: number; minimumCosineSimilarity: number }
  environment: { userAgent: string; crossOriginIsolated: boolean; webgpu: boolean }
}

declare global {
  interface Window {
    __TRIPOSPLAT_GAUSSIAN_RESULT__?: GaussianLabResult
  }
}

const modelInput = document.querySelector<HTMLInputElement>('#model')!
const fixtureInput = document.querySelector<HTMLInputElement>('#fixture')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const statusElement = document.querySelector<HTMLElement>('[data-testid="gaussian-status"]')!
const errorElement = document.querySelector<HTMLPreElement>('[data-testid="gaussian-error"]')!
const resultElement = document.querySelector<HTMLPreElement>('[data-testid="gaussian-result"]')!
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
    throw new Error("Output 'features' is missing or is not float32.")
  }
  return new Float32Array(payload.data)
}

function gate(reference: Float32Array, candidate: Float32Array): OutputGate {
  const comparison = compareFloat32(reference, candidate)
  let within = 0
  let maxError = -1
  let maxErrorIndex = 0
  for (let index = 0; index < reference.length; index += 1) {
    const error = Math.abs(reference[index] - candidate[index])
    if (error <= ATOL + RTOL * Math.abs(reference[index])) within += 1
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
      && comparison.cosineSimilarity >= MINIMUM_COSINE,
  }
}

async function run(): Promise<void> {
  if (busy) return
  busy = true
  runButton.disabled = true
  runButton.textContent = 'Running…'
  errorElement.hidden = true
  resultElement.hidden = true
  delete window.__TRIPOSPLAT_GAUSSIAN_RESULT__
  try {
    if (activeClient) await activeClient.dispose()
    const modelUrl = modelInput.value
    const fixtureUrl = fixtureInput.value
    const client = new OrtWorkerClient({ onStatus: ({ message }) => { statusElement.textContent = message } })
    activeClient = client
    statusElement.textContent = 'Fetching the official fp32 Gaussian fixture…'
    const [points, cond, reference] = await Promise.all([
      fetchFloat32(`${fixtureUrl}/points.f32`, elementCount(SHAPES.points)),
      fetchFloat32(`${fixtureUrl}/cond.f32`, elementCount(SHAPES.cond)),
      fetchFloat32(`${fixtureUrl}/features.f32`, elementCount(SHAPES.features)),
    ])
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
    if (loaded.executionProvider !== 'webgpu') {
      throw new Error(`Expected WebGPU, loaded ${loaded.executionProvider}.`)
    }
    const response = await client.runSession({
      sessionId: SESSION_ID,
      inputs: {
        points: createTensorPayload('float32', points, SHAPES.points),
        cond: createTensorPayload('float32', cond, SHAPES.cond),
      },
      outputs: ['features'],
      tag: 'gaussian-decoder-parity',
    })
    const comparison = gate(reference, payloadFloat32(response.outputs.features))
    const result: GaussianLabResult = {
      passed: comparison.passed,
      executionProvider: loaded.executionProvider,
      modelLoadMs: loaded.loadMs,
      modelTransferBytes,
      inferenceMs: response.timings.inferenceMs,
      readbackMs: response.timings.readbackMs,
      comparison,
      tolerance: { absolute: ATOL, relative: RTOL, minimumCosineSimilarity: MINIMUM_COSINE },
      environment: {
        userAgent: navigator.userAgent,
        crossOriginIsolated: self.crossOriginIsolated,
        webgpu: 'gpu' in navigator,
      },
    }
    await client.dispose()
    if (activeClient === client) activeClient = undefined
    window.__TRIPOSPLAT_GAUSSIAN_RESULT__ = result
    resultElement.textContent = JSON.stringify(result, null, 2)
    resultElement.hidden = false
    statusElement.textContent = result.passed
      ? 'PASS: WebGPU Gaussian features match official fp32 PyTorch.'
      : 'FAIL: WebGPU Gaussian features exceed tolerance.'
  } catch (caught) {
    const message = caught instanceof Error ? caught.message : String(caught)
    errorElement.textContent = message
    errorElement.hidden = false
    statusElement.textContent = 'Gaussian decoder validation failed.'
    if (activeClient) await activeClient.dispose().catch(() => undefined)
    activeClient = undefined
  } finally {
    busy = false
    runButton.disabled = false
    runButton.textContent = 'Run Gaussian decoder parity gate'
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
