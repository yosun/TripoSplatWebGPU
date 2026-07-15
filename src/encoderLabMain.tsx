import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { createVaeEncoderSliceManifest } from './models/triposplat/manifests'
import { compositeRgbaOnBlack, imageBitmapToRgba } from './models/triposplat/preprocess'
import { compareFloat32, type TensorComparison } from './models/triposplat/tensorMath'
import { TripoSplatWebGPUModel } from './models/triposplat/TripoSplatWebGPUModel'
import type { OrtRunTimings, OrtWorkerStatus } from './runtime/OrtWorkerClient'

const DEFAULT_MODEL = '/models/triposplat/flux2_vae_encoder.onnx'
const DEFAULT_FIXTURE = '/fixtures/generated/flux2-vae-fp32'

interface EncoderLabResult {
  passed: boolean
  executionProvider: string
  modelLoadMs: number
  endToEndEncodeMs: number
  runtime: OrtRunTimings
  comparison: TensorComparison
  preprocessing: ByteComparison
  runs: EncoderLabRun[]
  medianInferenceMs: number
  medianEndToEndEncodeMs: number
  modelTransferBytes?: number
  memory: {
    beforeLoad: BrowserMemorySnapshot
    afterLoad: BrowserMemorySnapshot
    afterRuns: BrowserMemorySnapshot
  }
  environment: BrowserEnvironment
  tolerance: { maxAbsoluteError: number; minimumCosineSimilarity: number }
}

interface EncoderLabRun {
  run: number
  endToEndEncodeMs: number
  runtime: OrtRunTimings
  comparison: TensorComparison
}

interface ByteComparison {
  count: number
  mismatched: number
  maxAbsoluteError: number
  meanAbsoluteError: number
  passed: boolean
}

interface BrowserMemorySnapshot {
  source: 'measureUserAgentSpecificMemory' | 'performance.memory' | 'unavailable'
  bytes?: number
  usedJsHeapBytes?: number
  totalJsHeapBytes?: number
  jsHeapLimitBytes?: number
}

interface BrowserEnvironment {
  userAgent: string
  platform: string
  crossOriginIsolated: boolean
  webgpu: boolean
  adapter?: {
    vendor?: string
    architecture?: string
    device?: string
    description?: string
  }
}

type PerformanceWithMemory = Performance & {
  memory?: {
    usedJSHeapSize: number
    totalJSHeapSize: number
    jsHeapSizeLimit: number
  }
}

type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<{
      info?: {
        vendor?: string
        architecture?: string
        device?: string
        description?: string
      }
    } | null>
  }
}

const BENCHMARK_RUNS = 3

declare global {
  interface Window {
    __TRIPOSPLAT_ENCODER_RESULT__?: EncoderLabResult
  }
}

async function fetchFloat32(url: string): Promise<Float32Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`)
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength % 4 !== 0) throw new Error(`${url} is not aligned float32 data.`)
  return new Float32Array(buffer)
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

function compareBytes(reference: Uint8ClampedArray, candidate: Uint8ClampedArray): ByteComparison {
  if (reference.length !== candidate.length) {
    throw new Error(`Prepared RGB length ${candidate.length} does not match Python ${reference.length}.`)
  }
  let mismatched = 0
  let maxAbsoluteError = 0
  let absoluteErrorSum = 0
  for (let index = 0; index < reference.length; index += 1) {
    const error = Math.abs(reference[index] - candidate[index])
    if (error !== 0) mismatched += 1
    maxAbsoluteError = Math.max(maxAbsoluteError, error)
    absoluteErrorSum += error
  }
  const meanAbsoluteError = absoluteErrorSum / reference.length
  return {
    count: reference.length,
    mismatched,
    maxAbsoluteError,
    meanAbsoluteError,
    passed: mismatched === 0,
  }
}

async function measureBrowserMemory(): Promise<BrowserMemorySnapshot> {
  const browserPerformance = performance as PerformanceWithMemory
  // `measureUserAgentSpecificMemory()` can interrupt a live ORT worker in
  // Chromium. Keep the parity gate non-invasive and report the narrower main
  // realm heap counter with its limitation instead.
  if (browserPerformance.memory) {
    return {
      source: 'performance.memory',
      usedJsHeapBytes: browserPerformance.memory.usedJSHeapSize,
      totalJsHeapBytes: browserPerformance.memory.totalJSHeapSize,
      jsHeapLimitBytes: browserPerformance.memory.jsHeapSizeLimit,
    }
  }
  return { source: 'unavailable' }
}

async function inspectBrowserEnvironment(): Promise<BrowserEnvironment> {
  const gpu = (navigator as NavigatorWithGpu).gpu
  const adapter = gpu ? await gpu.requestAdapter() : null
  const info = adapter?.info
  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: Boolean(gpu),
    adapter: info ? {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    } : undefined,
  }
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

export function EncoderLab() {
  const modelRef = useRef<TripoSplatWebGPUModel | null>(null)
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL)
  const [fixtureUrl, setFixtureUrl] = useState(DEFAULT_FIXTURE)
  const [status, setStatus] = useState('Ready to run the recorded PyTorch fixture.')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<EncoderLabResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => () => {
    const model = modelRef.current
    modelRef.current = null
    if (model) void model.dispose()
  }, [])

  const run = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    delete window.__TRIPOSPLAT_ENCODER_RESULT__
    let executionProvider = 'unknown'
    const onRuntimeStatus = (event: OrtWorkerStatus) => {
      setStatus(event.message)
      if (event.executionProvider) executionProvider = event.executionProvider
    }

    try {
      const previous = modelRef.current
      modelRef.current = null
      if (previous) await previous.dispose()
      const model = new TripoSplatWebGPUModel({
        graphs: createVaeEncoderSliceManifest(modelUrl, 'float32'),
        allowWasmFallback: false,
        onRuntimeStatus,
      })
      modelRef.current = model

      setStatus('Fetching the prepared image, explicit epsilon and PyTorch reference…')
      const [imageResponse, preparedResponse, epsilon, reference, environment] = await Promise.all([
        fetch(`${fixtureUrl}/source.png`),
        fetch(`${fixtureUrl}/prepared.png`),
        fetchFloat32(`${fixtureUrl}/epsilon.f32`),
        fetchFloat32(`${fixtureUrl}/feature2.f32`),
        inspectBrowserEnvironment(),
      ])
      if (!imageResponse.ok) throw new Error(`Could not fetch fixture image: HTTP ${imageResponse.status}`)
      if (!preparedResponse.ok) throw new Error(`Could not fetch prepared reference: HTTP ${preparedResponse.status}`)
      const imageBlob = await imageResponse.blob()
      const preparedBitmap = await createImageBitmap(await preparedResponse.blob())
      const preparedReference = (() => {
        try {
          return compositeRgbaOnBlack(imageBitmapToRgba(preparedBitmap)).data
        } finally {
          preparedBitmap.close()
        }
      })()

      const manifest = model.graphs.vaeEncoder?.manifest
      const transferLengths = manifest
        ? await Promise.all([
            contentLength(manifest.graphUrl),
            ...(manifest.externalData ?? []).map(({ url }) => contentLength(url)),
          ])
        : []
      const modelTransferBytes = transferLengths.length > 0 && transferLengths.every((value) => value !== undefined)
        ? transferLengths.reduce<number>((total, value) => total + (value ?? 0), 0)
        : undefined

      const beforeLoad = await measureBrowserMemory()
      const loadStarted = performance.now()
      await model.load()
      const modelLoadMs = performance.now() - loadStarted
      const afterLoad = await measureBrowserMemory()
      const runs: EncoderLabRun[] = []
      let preparedCandidate: Uint8ClampedArray | undefined
      for (let run = 1; run <= BENCHMARK_RUNS; run += 1) {
        const bitmap = await createImageBitmap(imageBlob)
        const encodeStarted = performance.now()
        const encoded = await (async () => {
          try {
            return await model.encode(bitmap, {
              vaeNoise: epsilon,
              onProgress: (progress) => setStatus(`Run ${run}/${BENCHMARK_RUNS}: ${progress.message}`),
            })
          } finally {
            bitmap.close()
          }
        })()
        const endToEndEncodeMs = performance.now() - encodeStarted
        if (!encoded.feature2 || !encoded.timings.vaeEncoder) {
          throw new Error('The VAE slice returned no feature2 tensor or runtime timings.')
        }
        if (!preparedCandidate) preparedCandidate = new Uint8ClampedArray(encoded.preparedImage.data)
        runs.push({
          run,
          endToEndEncodeMs,
          runtime: encoded.timings.vaeEncoder,
          comparison: compareFloat32(reference, encoded.feature2),
        })
      }
      const afterRuns = await measureBrowserMemory()
      const finalRun = runs[runs.length - 1]
      if (!preparedCandidate) throw new Error('Browser preprocessing returned no prepared RGB image.')
      const preprocessing = compareBytes(preparedReference, preparedCandidate)
      const tolerance = { maxAbsoluteError: 0.006, minimumCosineSimilarity: 0.99999 }
      const nextResult: EncoderLabResult = {
        passed: preprocessing.passed && runs.every(({ comparison }) =>
          comparison.finite &&
          comparison.maxAbsoluteError <= tolerance.maxAbsoluteError &&
          comparison.cosineSimilarity >= tolerance.minimumCosineSimilarity),
        executionProvider,
        modelLoadMs,
        endToEndEncodeMs: finalRun.endToEndEncodeMs,
        runtime: finalRun.runtime,
        comparison: finalRun.comparison,
        preprocessing,
        runs,
        medianInferenceMs: median(runs.map(({ runtime }) => runtime.inferenceMs)),
        medianEndToEndEncodeMs: median(runs.map(({ endToEndEncodeMs }) => endToEndEncodeMs)),
        modelTransferBytes,
        memory: { beforeLoad, afterLoad, afterRuns },
        environment,
        tolerance,
      }
      window.__TRIPOSPLAT_ENCODER_RESULT__ = nextResult
      setResult(nextResult)
      setStatus(nextResult.passed ? 'PASS: WebGPU output matches the recorded PyTorch tensor.' : 'FAIL: tensor drift exceeds tolerance.')
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught)
      setError(message)
      setStatus('Encoder validation failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main>
      <h1>TripoSplat Flux VAE · WebGPU parity</h1>
      <p>This page runs the recorded alpha-present source image through browser preprocessing, then sends the resulting 1024² tensor and explicit epsilon through ONNX Runtime WebGPU.</p>
      <label>ONNX graph <input value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} /></label>
      <label>Fixture directory <input value={fixtureUrl} onChange={(event) => setFixtureUrl(event.target.value)} /></label>
      <button type="button" disabled={busy} onClick={() => void run()}>{busy ? 'Running…' : 'Run parity gate'}</button>
      <p role="status" data-testid="encoder-status">{status}</p>
      {error ? <pre className="error" data-testid="encoder-error">{error}</pre> : null}
      {result ? <pre data-testid="encoder-result">{JSON.stringify(result, null, 2)}</pre> : null}
    </main>
  )
}

const style = document.createElement('style')
style.textContent = `
  :root { color: #ececf3; background: #101014; font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  body { margin: 0; }
  main { max-width: 920px; margin: 0 auto; padding: 48px 24px; }
  h1 { font: 600 28px/1.2 system-ui, sans-serif; }
  label { display: grid; gap: 6px; margin: 18px 0; }
  input { box-sizing: border-box; width: 100%; padding: 10px; color: inherit; background: #1b1b22; border: 1px solid #3a3a48; border-radius: 6px; }
  button { padding: 10px 16px; color: #08080a; background: #f8cf00; border: 0; border-radius: 6px; font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .55; cursor: wait; }
  pre { overflow: auto; padding: 16px; background: #18181f; border-radius: 8px; }
  .error { color: #ff9b9b; }
`
document.head.append(style)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <EncoderLab />
  </StrictMode>,
)
