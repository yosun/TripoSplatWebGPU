import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { compareFloat32, type TensorComparison } from './models/triposplat/tensorMath'
import { OrtWorkerClient, type OrtWorkerStatus } from './runtime/OrtWorkerClient'
import { createTensorPayload } from './runtime/tensors'

const DEFAULT_MODEL = '/models/triposplat/dit_step.onnx'
const DEFAULT_FIXTURE = '/fixtures/generated/dit-step-fp32'
const SESSION_ID = 'triposplat/dit-parity'
const FP16_TOLERANCE = { absolute: 0.04, relative: 0.03, minimumCosineSimilarity: 0.9999 }
const FP32_TOLERANCE = { absolute: 0.00002, relative: 0.001, minimumCosineSimilarity: 0.999999999 }

const SHAPES = {
  latent: [1, 8192, 16],
  camera: [1, 1, 5],
  t: [1],
  feature1: [1, 4101, 1280],
  feature2: [1, 4101, 128],
  pred_latent: [1, 8192, 16],
  pred_camera: [1, 1, 5],
} as const

interface OutputGate extends TensorComparison {
  fractionWithinTolerance: number
  maxErrorIndex: number
  referenceAtMaxError: number
  candidateAtMaxError: number
  firstMismatches: Array<{ index: number; reference: number; candidate: number }>
  passed: boolean
}

interface DitLabResult {
  passed: boolean
  executionProvider: string
  modelLoadMs: number
  modelTransferBytes?: number
  inferenceMs: number
  readbackMs: number
  outputs?: {
    pred_latent: OutputGate
    pred_camera: OutputGate
  }
  probes?: Record<string, OutputGate>
  tolerance: { absolute: number; relative: number; minimumCosineSimilarity: number }
  environment: {
    userAgent: string
    crossOriginIsolated: boolean
    webgpu: boolean
  }
}

interface ProbeManifest {
  includeFinalOutputs?: boolean
  outputs: Array<{ name: string; path: string; shape: number[]; elements: number }>
}

declare global {
  interface Window {
    __TRIPOSPLAT_DIT_RESULT__?: DitLabResult
  }
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

function elementCount(shape: readonly number[]): number {
  return shape.reduce((product, value) => product * value, 1)
}

type DitTolerance = typeof FP16_TOLERANCE

async function fixtureTolerance(fixtureUrl: string): Promise<DitTolerance> {
  const response = await fetch(`${fixtureUrl}/manifest.json`)
  if (response.status === 404) return FP16_TOLERANCE
  if (!response.ok) throw new Error(`Could not fetch fixture manifest: HTTP ${response.status}`)
  const manifest = await response.json() as { metadata?: { internal_precision?: unknown } }
  return manifest.metadata?.internal_precision === 'fp32' ? FP32_TOLERANCE : FP16_TOLERANCE
}

function gateOutput(
  reference: Float32Array,
  candidate: Float32Array,
  tolerance: DitTolerance,
): OutputGate {
  const comparison = compareFloat32(reference, candidate)
  let within = 0
  let maxErrorIndex = 0
  let maxError = -1
  const firstMismatches: OutputGate['firstMismatches'] = []
  for (let index = 0; index < reference.length; index += 1) {
    const error = Math.abs(reference[index] - candidate[index])
    const isWithin = error <= tolerance.absolute + tolerance.relative * Math.abs(reference[index])
    if (error > maxError) {
      maxError = error
      maxErrorIndex = index
    }
    if (isWithin) {
      within += 1
    } else if (firstMismatches.length < 8) {
      firstMismatches.push({ index, reference: reference[index], candidate: candidate[index] })
    }
  }
  const fractionWithinTolerance = within / reference.length
  return {
    ...comparison,
    fractionWithinTolerance,
    maxErrorIndex,
    referenceAtMaxError: reference[maxErrorIndex],
    candidateAtMaxError: candidate[maxErrorIndex],
    firstMismatches,
    passed: comparison.finite
      && fractionWithinTolerance === 1
      && comparison.cosineSimilarity >= tolerance.minimumCosineSimilarity,
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

export function DitLab() {
  const clientRef = useRef<OrtWorkerClient | null>(null)
  const autoRunStartedRef = useRef(false)
  const runRef = useRef<() => Promise<void>>(async () => undefined)
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL)
  const [fixtureUrl, setFixtureUrl] = useState(DEFAULT_FIXTURE)
  const [status, setStatus] = useState('Ready to validate one official DiT invocation.')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DitLabResult | null>(null)

  useEffect(() => () => {
    const client = clientRef.current
    clientRef.current = null
    if (client) void client.dispose()
  }, [])

  const run = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    delete window.__TRIPOSPLAT_DIT_RESULT__
    let client: OrtWorkerClient | undefined
    try {
      if (clientRef.current) await clientRef.current.dispose()
      const onStatus = (event: OrtWorkerStatus) => setStatus(event.message)
      client = new OrtWorkerClient({ onStatus })
      clientRef.current = client
      setStatus('Fetching deterministic official one-step tensors…')
      const tolerance = await fixtureTolerance(fixtureUrl)
      const names = Object.keys(SHAPES) as Array<keyof typeof SHAPES>
      const tensors = Object.fromEntries(await Promise.all(names.map(async (name) => [
        name,
        await fetchFloat32(`${fixtureUrl}/${name}.f32`, elementCount(SHAPES[name])),
      ]))) as Record<keyof typeof SHAPES, Float32Array>
      let probeManifest: ProbeManifest | undefined
      const probeManifestResponse = await fetch(`${fixtureUrl}/probes.json`)
      if (
        probeManifestResponse.ok
        && probeManifestResponse.headers.get('content-type')?.includes('application/json')
      ) {
        probeManifest = await probeManifestResponse.json() as ProbeManifest
      } else if (!probeManifestResponse.ok && probeManifestResponse.status !== 404) {
        throw new Error(`Could not fetch probe manifest: HTTP ${probeManifestResponse.status}`)
      }
      const probeReferences = probeManifest
        ? Object.fromEntries(await Promise.all(probeManifest.outputs.map(async ({ name, path, elements }) => [
            name,
            await fetchFloat32(`${fixtureUrl}/${path}`, elements),
          ]))) as Record<string, Float32Array>
        : undefined

      const sidecarUrl = `${modelUrl}.data`
      const externalDataPath = new URL(modelUrl, document.baseURI).pathname.split('/').at(-1)
      if (!externalDataPath) throw new Error(`Could not derive external-data path from ${modelUrl}.`)
      const transferParts = await Promise.all([contentLength(modelUrl), contentLength(sidecarUrl)])
      const modelTransferBytes = transferParts.every((value) => value !== undefined)
        ? transferParts.reduce<number>((sum, value) => sum + (value ?? 0), 0)
        : undefined
      const loaded = await client.loadSession({
        sessionId: SESSION_ID,
        manifest: {
          graphUrl: modelUrl,
          externalData: [{ path: `${decodeURIComponent(externalDataPath)}.data`, url: sidecarUrl }],
        },
        options: {
          allowWasmFallback: false,
          graphOptimizationLevel: 'disabled',
        },
      })
      if (loaded.executionProvider !== 'webgpu') {
        throw new Error(`Expected WebGPU, loaded ${loaded.executionProvider}.`)
      }

      const response = await client.runSession({
        sessionId: SESSION_ID,
        inputs: {
          latent: createTensorPayload('float32', tensors.latent, SHAPES.latent),
          camera: createTensorPayload('float32', tensors.camera, SHAPES.camera),
          t: createTensorPayload('float32', tensors.t, SHAPES.t),
          feature1: createTensorPayload('float32', tensors.feature1, SHAPES.feature1),
          feature2: createTensorPayload('float32', tensors.feature2, SHAPES.feature2),
        },
        outputs: [
          ...(probeManifest?.includeFinalOutputs === false ? [] : ['pred_latent', 'pred_camera']),
          ...(probeManifest?.outputs.map(({ name }) => name) ?? []),
        ],
        tag: 'official-one-step-parity',
      })
      const outputs = probeManifest?.includeFinalOutputs === false
        ? undefined
        : (() => {
            const predLatent = response.outputs.pred_latent
            const predCamera = response.outputs.pred_camera
            if (!predLatent || predLatent.type !== 'float32' || !predCamera || predCamera.type !== 'float32') {
              throw new Error('DiT did not return both public float32 outputs.')
            }
            return {
              pred_latent: gateOutput(tensors.pred_latent, predLatent.data, tolerance),
              pred_camera: gateOutput(tensors.pred_camera, predCamera.data, tolerance),
            }
          })()
      const probes = probeReferences
        ? Object.fromEntries(Object.entries(probeReferences).map(([name, reference]) => {
            const payload = response.outputs[name]
            if (!payload || payload.type !== 'float32') {
              throw new Error(`DiT probe '${name}' did not return float32 data.`)
            }
            return [name, gateOutput(reference, payload.data, tolerance)]
          }))
        : undefined
      const next: DitLabResult = {
        passed: (outputs === undefined || (outputs.pred_latent.passed && outputs.pred_camera.passed))
          && (probes === undefined || Object.values(probes).every(({ passed }) => passed)),
        executionProvider: loaded.executionProvider,
        modelLoadMs: loaded.loadMs,
        modelTransferBytes,
        inferenceMs: response.timings.inferenceMs,
        readbackMs: response.timings.readbackMs,
        outputs,
        probes,
        tolerance,
        environment: {
          userAgent: navigator.userAgent,
          crossOriginIsolated: self.crossOriginIsolated,
          webgpu: 'gpu' in navigator,
        },
      }
      window.__TRIPOSPLAT_DIT_RESULT__ = next
      setResult(next)
      setStatus(next.passed ? 'PASS: one WebGPU DiT invocation matches official PyTorch.' : 'FAIL: WebGPU DiT output drift exceeds tolerance.')
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught)
      setError(message)
      setStatus('DiT validation failed.')
    } finally {
      setBusy(false)
    }
  }

  runRef.current = run
  useEffect(() => {
    const timeout = window.setTimeout(() => {
      if (
        new URLSearchParams(window.location.search).get('autorun') === '1'
        && !autoRunStartedRef.current
      ) {
        autoRunStartedRef.current = true
        void runRef.current()
      }
    }, 0)
    return () => window.clearTimeout(timeout)
  }, [])

  return (
    <main>
      <h1>TripoSplat DiT · WebGPU parity</h1>
      <p>Loads the query-chunked official one-step graph and compares one browser invocation with an untouched PyTorch fixture.</p>
      <label>ONNX graph <input value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} /></label>
      <label>Fixture directory <input value={fixtureUrl} onChange={(event) => setFixtureUrl(event.target.value)} /></label>
      <button type="button" disabled={busy} onClick={() => void run()}>{busy ? 'Running…' : 'Run DiT parity gate'}</button>
      <p role="status" data-testid="dit-status">{status}</p>
      {error ? <pre className="error" data-testid="dit-error">{error}</pre> : null}
      {result ? <pre data-testid="dit-result">{JSON.stringify(result, null, 2)}</pre> : null}
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
    <DitLab />
  </StrictMode>,
)
