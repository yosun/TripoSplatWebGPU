import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import {
  sampleFlowEulerCfg,
  type FlowModelInvocation,
  type FlowTensorState,
} from './models/triposplat/flowSampler'
import { compareFloat32, type TensorComparison } from './models/triposplat/tensorMath'
import { OrtWorkerClient, type OrtWorkerStatus } from './runtime/OrtWorkerClient'
import { createTensorPayload, type TensorPayload } from './runtime/tensors'

const DEFAULT_MODEL = '/models/triposplat/dit_step_webgpu_fp32.onnx'
const DEFAULT_FIXTURE = '/fixtures/generated/flow4-fp32-compute'
const SESSION_ID = 'triposplat/flow-parity'
const FP16_TOLERANCE = { absolute: 0.2, relative: 0.05, minimumCosineSimilarity: 0.9995 }
const FP32_STRICT_TOLERANCE = { absolute: 0.0001, relative: 0.001, minimumCosineSimilarity: 0.99999999 }
const FP32_QUALIFICATION_TOLERANCE = { absolute: 0.005, relative: 0.003, minimumCosineSimilarity: 0.99999998 }

const SHAPES = {
  latent: [1, 8192, 16],
  camera: [1, 1, 5],
  feature1: [1, 4101, 1280],
  feature2: [1, 4101, 128],
} as const

interface FlowCondition {
  feature1: Float32Array
  feature2: Float32Array
}

interface OutputGate extends TensorComparison {
  fractionWithinTolerance: number
  maxErrorIndex: number
  referenceAtMaxError: number
  candidateAtMaxError: number
  firstMismatches: Array<{ index: number; reference: number; candidate: number }>
  passed: boolean
}

interface FlowLabResult {
  passed: boolean
  strictPassed: boolean
  executionProvider: string
  modelLoadMs: number
  modelTransferBytes?: number
  invocations: number
  inferenceMs: number
  readbackMs: number
  samplingWallMs: number
  outputs: { latent: OutputGate; camera: OutputGate }
  strictOutputs: { latent: OutputGate; camera: OutputGate }
  settings: { steps: number; guidanceScale: 3; shift: 3; arithmetic: 'float16' | 'float32' }
  tolerance: { absolute: number; relative: number; minimumCosineSimilarity: number }
  strictTolerance: { absolute: number; relative: number; minimumCosineSimilarity: number }
  environment: {
    userAgent: string
    crossOriginIsolated: boolean
    webgpu: boolean
  }
}

interface FlowTrajectoryInvocation {
  invocation: number
  step: number
  pass: 'conditional' | 'unconditional'
  tensors: Record<'sample_latent' | 'sample_camera' | 't' | 'pred_latent' | 'pred_camera', {
    path: string
  }>
}

interface FlowTrajectoryResult {
  passed: boolean
  executionProvider: string
  modelLoadMs: number
  modelTransferBytes?: number
  invocations: number
  inferenceMs: number
  readbackMs: number
  wallMs: number
  tolerance: { absolute: number; relative: number; minimumCosineSimilarity: number }
  records: Array<{
    invocation: number
    step: number
    pass: 'conditional' | 'unconditional'
    latent: OutputGate
    camera: OutputGate
  }>
  environment: FlowLabResult['environment']
}

declare global {
  interface Window {
    __TRIPOSPLAT_FLOW_RESULT__?: FlowLabResult
    __TRIPOSPLAT_FLOW_TRAJECTORY_RESULT__?: FlowTrajectoryResult
    /** @deprecated Use __TRIPOSPLAT_FLOW_RESULT__. */
    __TRIPOSPLAT_FLOW4_RESULT__?: FlowLabResult
  }
}

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

function payloadFloat32(name: string, payload: TensorPayload | undefined): Float32Array {
  if (!payload || payload.type !== 'float32') {
    throw new Error(`DiT output '${name}' is missing or is not float32.`)
  }
  return new Float32Array(payload.data)
}

type FlowTolerance = typeof FP16_TOLERANCE

async function fixtureConfiguration(fixtureUrl: string): Promise<{
  steps: number
  arithmetic: 'float16' | 'float32'
}> {
  const response = await fetch(`${fixtureUrl}/flow.json`)
  if (!response.ok) throw new Error(`Could not fetch ${fixtureUrl}/flow.json: HTTP ${response.status}`)
  const manifest = await response.json() as {
    settings?: { internal_precision?: unknown; steps?: unknown }
  }
  const precision = manifest.settings?.internal_precision
  const arithmetic = precision === 'fp32'
    ? 'float32'
    : precision === 'fp16' || precision === undefined
      ? 'float16'
      : undefined
  const steps = manifest.settings?.steps
  if (!arithmetic) throw new Error(`Unsupported flow fixture precision ${String(precision)}.`)
  if (!Number.isInteger(steps) || (steps as number) <= 0) {
    throw new Error(`Flow fixture has invalid step count ${String(steps)}.`)
  }
  return { steps: steps as number, arithmetic }
}

async function trajectoryConfiguration(fixtureUrl: string): Promise<{
  arithmetic: 'float16' | 'float32'
  trajectory: FlowTrajectoryInvocation[]
}> {
  const response = await fetch(`${fixtureUrl}/flow.json`)
  if (!response.ok) throw new Error(`Could not fetch ${fixtureUrl}/flow.json: HTTP ${response.status}`)
  const manifest = await response.json() as {
    settings?: { internal_precision?: unknown }
    trajectory?: FlowTrajectoryInvocation[]
  }
  const arithmetic = manifest.settings?.internal_precision === 'fp32'
    ? 'float32'
    : manifest.settings?.internal_precision === 'fp16'
      ? 'float16'
      : undefined
  if (!arithmetic) throw new Error('Trajectory fixture has no supported internal precision.')
  if (!Array.isArray(manifest.trajectory) || manifest.trajectory.length === 0) {
    throw new Error('Fixture has no recorded official invocation trajectory.')
  }
  return { arithmetic, trajectory: manifest.trajectory }
}

function gateOutput(
  reference: Float32Array,
  candidate: Float32Array,
  tolerance: FlowTolerance,
): OutputGate {
  const comparison = compareFloat32(reference, candidate)
  let within = 0
  let maxError = -1
  let maxErrorIndex = 0
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
    passed:
      comparison.finite
      && fractionWithinTolerance === 1
      && comparison.cosineSimilarity >= tolerance.minimumCosineSimilarity,
  }
}

export function FlowLab() {
  const clientRef = useRef<OrtWorkerClient | null>(null)
  const autoRunStartedRef = useRef(false)
  const runRef = useRef<() => Promise<void>>(async () => undefined)
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL)
  const [fixtureUrl, setFixtureUrl] = useState(DEFAULT_FIXTURE)
  const [status, setStatus] = useState('Ready to validate the 4-step browser flow loop.')
  const [progress, setProgress] = useState('No DiT invocations yet.')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FlowLabResult | null>(null)
  const [trajectoryResult, setTrajectoryResult] = useState<FlowTrajectoryResult | null>(null)

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
    setProgress('Loading one reusable DiT session…')
    delete window.__TRIPOSPLAT_FLOW4_RESULT__
    delete window.__TRIPOSPLAT_FLOW_RESULT__
    let client: OrtWorkerClient | undefined
    try {
      if (clientRef.current) await clientRef.current.dispose()
      const onStatus = (event: OrtWorkerStatus) => setStatus(event.message)
      client = new OrtWorkerClient({ onStatus })
      clientRef.current = client
      const { steps, arithmetic: predictionArithmetic } = await fixtureConfiguration(fixtureUrl)
      const expectedInvocations = steps * 2
      setStatus(`Fetching official ${steps}-step tensors…`)
      const [
        latent,
        camera,
        feature1,
        feature2,
        referenceLatent,
        referenceCamera,
      ] = await Promise.all([
        fetchFloat32(`${fixtureUrl}/latent.f32`, elementCount(SHAPES.latent)),
        fetchFloat32(`${fixtureUrl}/camera.f32`, elementCount(SHAPES.camera)),
        fetchFloat32(`${fixtureUrl}/feature1.f32`, elementCount(SHAPES.feature1)),
        fetchFloat32(`${fixtureUrl}/feature2.f32`, elementCount(SHAPES.feature2)),
        fetchFloat32(`${fixtureUrl}/flow${steps}_latent.f32`, elementCount(SHAPES.latent)),
        fetchFloat32(`${fixtureUrl}/flow${steps}_camera.f32`, elementCount(SHAPES.camera)),
      ])
      const strictTolerance = predictionArithmetic === 'float32'
        ? FP32_STRICT_TOLERANCE
        : FP16_TOLERANCE
      const tolerance = predictionArithmetic === 'float32'
        ? FP32_QUALIFICATION_TOLERANCE
        : FP16_TOLERANCE
      const condition: FlowCondition = { feature1, feature2 }
      const negativeCondition: FlowCondition = {
        feature1: new Float32Array(feature1.length),
        feature2: new Float32Array(feature2.length),
      }

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
        options: { allowWasmFallback: false, graphOptimizationLevel: 'disabled' },
      })
      if (loaded.executionProvider !== 'webgpu') {
        throw new Error(`Expected WebGPU, loaded ${loaded.executionProvider}.`)
      }

      let invocations = 0
      let inferenceMs = 0
      let readbackMs = 0
      const samplingStarted = performance.now()
      const sampled = await sampleFlowEulerCfg(
        async (invocation: FlowModelInvocation<FlowCondition>): Promise<FlowTensorState> => {
          invocations += 1
          setProgress(
            `Flow step ${invocation.step}/${steps} · ${invocation.pass} `
            + `(DiT invocation ${invocations}/${expectedInvocations})`,
          )
          setStatus(
            `Flow step ${invocation.step}/${steps} · ${invocation.pass} `
            + `(DiT invocation ${invocations}/${expectedInvocations})…`,
          )
          const response = await client!.runSession({
            sessionId: SESSION_ID,
            inputs: {
              latent: createTensorPayload(
                'float32',
                new Float32Array(invocation.sample.latent),
                SHAPES.latent,
              ),
              camera: createTensorPayload(
                'float32',
                new Float32Array(invocation.sample.camera),
                SHAPES.camera,
              ),
              t: createTensorPayload(
                'float32',
                new Float32Array(invocation.timestepTensor),
                [1],
              ),
              feature1: createTensorPayload(
                'float32',
                new Float32Array(invocation.condition.feature1),
                SHAPES.feature1,
              ),
              feature2: createTensorPayload(
                'float32',
                new Float32Array(invocation.condition.feature2),
                SHAPES.feature2,
              ),
            },
            outputs: ['pred_latent', 'pred_camera'],
            tag: `flow${steps}-${invocation.pass}-${invocation.step}`,
          })
          inferenceMs += response.timings.inferenceMs
          readbackMs += response.timings.readbackMs
          return {
            latent: payloadFloat32('pred_latent', response.outputs.pred_latent),
            camera: payloadFloat32('pred_camera', response.outputs.pred_camera),
          }
        },
        { latent, camera },
        {
          condition,
          negativeCondition,
          steps,
          guidanceScale: 3,
          shift: 3,
          predictionArithmetic,
        },
      )
      const samplingWallMs = performance.now() - samplingStarted
      const outputs = {
        latent: gateOutput(referenceLatent, sampled.latent, tolerance),
        camera: gateOutput(referenceCamera, sampled.camera, tolerance),
      }
      const strictOutputs = {
        latent: gateOutput(referenceLatent, sampled.latent, strictTolerance),
        camera: gateOutput(referenceCamera, sampled.camera, strictTolerance),
      }
      const strictPassed = invocations === expectedInvocations
        && strictOutputs.latent.passed
        && strictOutputs.camera.passed
      const next: FlowLabResult = {
        passed: invocations === expectedInvocations && outputs.latent.passed && outputs.camera.passed,
        strictPassed,
        executionProvider: loaded.executionProvider,
        modelLoadMs: loaded.loadMs,
        modelTransferBytes,
        invocations,
        inferenceMs,
        readbackMs,
        samplingWallMs,
        outputs,
        strictOutputs,
        settings: { steps, guidanceScale: 3, shift: 3, arithmetic: predictionArithmetic },
        tolerance,
        strictTolerance,
        environment: {
          userAgent: navigator.userAgent,
          crossOriginIsolated: self.crossOriginIsolated,
          webgpu: 'gpu' in navigator,
        },
      }
      await client.dispose()
      if (clientRef.current === client) clientRef.current = null
      window.__TRIPOSPLAT_FLOW4_RESULT__ = next
      window.__TRIPOSPLAT_FLOW_RESULT__ = next
      setProgress(`Completed ${invocations}/${expectedInvocations} DiT invocations.`)
      setResult(next)
      setStatus(
        next.passed
          ? next.strictPassed
            ? `PASS: ${steps}-step WebGPU flow matches the strict official gate.`
            : `PASS: ${steps}-step WebGPU flow matches the fp32 qualification envelope; strict gate remains failed.`
          : `FAIL: ${steps}-step WebGPU flow drift exceeds tolerance.`,
      )
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught)
      setError(message)
      setStatus('Flow validation failed.')
      if (client) await client.dispose().catch(() => undefined)
      if (clientRef.current === client) clientRef.current = null
    } finally {
      setBusy(false)
    }
  }

  const runTeacherForced = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    setTrajectoryResult(null)
    delete window.__TRIPOSPLAT_FLOW_TRAJECTORY_RESULT__
    let client: OrtWorkerClient | undefined
    try {
      if (clientRef.current) await clientRef.current.dispose()
      client = new OrtWorkerClient({ onStatus: (event) => setStatus(event.message) })
      clientRef.current = client
      setStatus('Fetching the official per-invocation trajectory…')
      const configuration = await trajectoryConfiguration(fixtureUrl)
      const tolerance = configuration.arithmetic === 'float32'
        ? FP32_STRICT_TOLERANCE
        : FP16_TOLERANCE
      const [feature1, feature2] = await Promise.all([
        fetchFloat32(`${fixtureUrl}/feature1.f32`, elementCount(SHAPES.feature1)),
        fetchFloat32(`${fixtureUrl}/feature2.f32`, elementCount(SHAPES.feature2)),
      ])
      const zeroFeature1 = new Float32Array(feature1.length)
      const zeroFeature2 = new Float32Array(feature2.length)
      const sidecarUrl = `${modelUrl}.data`
      const externalDataPath = new URL(modelUrl, document.baseURI).pathname.split('/').at(-1)
      if (!externalDataPath) throw new Error(`Could not derive external-data path from ${modelUrl}.`)
      const transferParts = await Promise.all([contentLength(modelUrl), contentLength(sidecarUrl)])
      const modelTransferBytes = transferParts.every((value) => value !== undefined)
        ? transferParts.reduce<number>((sum, value) => sum + (value ?? 0), 0)
        : undefined
      const loaded = await client.loadSession({
        sessionId: `${SESSION_ID}/trajectory`,
        manifest: {
          graphUrl: modelUrl,
          externalData: [{ path: `${decodeURIComponent(externalDataPath)}.data`, url: sidecarUrl }],
        },
        options: { allowWasmFallback: false, graphOptimizationLevel: 'disabled' },
      })
      if (loaded.executionProvider !== 'webgpu') {
        throw new Error(`Expected WebGPU, loaded ${loaded.executionProvider}.`)
      }
      let inferenceMs = 0
      let readbackMs = 0
      const records: FlowTrajectoryResult['records'] = []
      const started = performance.now()
      for (const invocation of configuration.trajectory) {
        setProgress(
          `Teacher-forced step ${invocation.step} · ${invocation.pass} `
          + `(DiT invocation ${invocation.invocation}/${configuration.trajectory.length})`,
        )
        const path = (name: keyof FlowTrajectoryInvocation['tensors']) =>
          `${fixtureUrl}/${invocation.tensors[name].path}`
        const [latent, camera, timestep, referenceLatent, referenceCamera] = await Promise.all([
          fetchFloat32(path('sample_latent'), elementCount(SHAPES.latent)),
          fetchFloat32(path('sample_camera'), elementCount(SHAPES.camera)),
          fetchFloat32(path('t'), 1),
          fetchFloat32(path('pred_latent'), elementCount(SHAPES.latent)),
          fetchFloat32(path('pred_camera'), elementCount(SHAPES.camera)),
        ])
        const conditional = invocation.pass === 'conditional'
        const response = await client.runSession({
          sessionId: `${SESSION_ID}/trajectory`,
          inputs: {
            latent: createTensorPayload('float32', latent, SHAPES.latent),
            camera: createTensorPayload('float32', camera, SHAPES.camera),
            t: createTensorPayload('float32', timestep, [1]),
            feature1: createTensorPayload(
              'float32',
              new Float32Array(conditional ? feature1 : zeroFeature1),
              SHAPES.feature1,
            ),
            feature2: createTensorPayload(
              'float32',
              new Float32Array(conditional ? feature2 : zeroFeature2),
              SHAPES.feature2,
            ),
          },
          outputs: ['pred_latent', 'pred_camera'],
          tag: `teacher-${invocation.pass}-${invocation.step}`,
        })
        inferenceMs += response.timings.inferenceMs
        readbackMs += response.timings.readbackMs
        records.push({
          invocation: invocation.invocation,
          step: invocation.step,
          pass: invocation.pass,
          latent: gateOutput(
            referenceLatent,
            payloadFloat32('pred_latent', response.outputs.pred_latent),
            tolerance,
          ),
          camera: gateOutput(
            referenceCamera,
            payloadFloat32('pred_camera', response.outputs.pred_camera),
            tolerance,
          ),
        })
      }
      const next: FlowTrajectoryResult = {
        passed: records.every((record) => record.latent.passed && record.camera.passed),
        executionProvider: loaded.executionProvider,
        modelLoadMs: loaded.loadMs,
        ...(modelTransferBytes === undefined ? {} : { modelTransferBytes }),
        invocations: records.length,
        inferenceMs,
        readbackMs,
        wallMs: performance.now() - started,
        tolerance,
        records,
        environment: {
          userAgent: navigator.userAgent,
          crossOriginIsolated: self.crossOriginIsolated,
          webgpu: 'gpu' in navigator,
        },
      }
      await client.dispose()
      if (clientRef.current === client) clientRef.current = null
      window.__TRIPOSPLAT_FLOW_TRAJECTORY_RESULT__ = next
      setTrajectoryResult(next)
      setProgress(`Completed ${records.length}/${configuration.trajectory.length} teacher-forced calls.`)
      setStatus(next.passed
        ? 'PASS: every teacher-forced DiT invocation matches its official state.'
        : 'FAIL: at least one teacher-forced DiT invocation exceeds the strict gate.')
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught)
      setError(message)
      setStatus('Teacher-forced flow validation failed.')
      if (client) await client.dispose().catch(() => undefined)
      if (clientRef.current === client) clientRef.current = null
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
      <h1>TripoSplat · WebGPU flow parity</h1>
      <p>Runs the TypeScript CFG/Euler loop using the official fixture's 4- or 20-step schedule.</p>
      <label>ONNX graph <input value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} /></label>
      <label>Fixture directory <input value={fixtureUrl} onChange={(event) => setFixtureUrl(event.target.value)} /></label>
      <button type="button" disabled={busy} onClick={() => void run()}>
        {busy ? 'Running…' : 'Run flow parity gate'}
      </button>
      <button type="button" disabled={busy} onClick={() => void runTeacherForced()}>
        {busy ? 'Running…' : 'Run teacher-forced invocation gates'}
      </button>
      <p role="status" data-testid="flow-status">{status}</p>
      <p data-testid="flow-progress">{progress}</p>
      {error ? <pre className="error" data-testid="flow-error">{error}</pre> : null}
      {result ? <pre data-testid="flow-result">{JSON.stringify(result, null, 2)}</pre> : null}
      {trajectoryResult
        ? <pre data-testid="flow-trajectory-result">{JSON.stringify(trajectoryResult, null, 2)}</pre>
        : null}
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
  button { margin-right: 10px; padding: 10px 16px; color: #08080a; background: #f8cf00; border: 0; border-radius: 6px; font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .55; cursor: wait; }
  pre { overflow: auto; padding: 16px; background: #18181f; border-radius: 8px; }
  .error { color: #ff9b9b; }
`
document.head.append(style)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <FlowLab />
  </StrictMode>,
)
