import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { octreeSoftmax, sampleOctree, systematicResample } from './models/triposplat/octree'
import { compareFloat32, type TensorComparison } from './models/triposplat/tensorMath'
import { OrtWorkerClient, type OrtWorkerStatus } from './runtime/OrtWorkerClient'
import { createTensorPayload, type TensorPayload } from './runtime/tensors'

const DEFAULT_MODEL = '/models/triposplat/octree_occupancy_decoder_fp32.onnx'
const DEFAULT_FIXTURE = '/fixtures/generated/octree-occupancy-fp32'
const DEFAULT_TRAJECTORY_FIXTURE = '/fixtures/generated/octree-trajectory-flow4-fp32'
const SESSION_ID = 'triposplat/octree-occupancy-parity'
const ATOL = 0.005
const RTOL = 0.01
const MINIMUM_COSINE = 0.99999

const SHAPES = {
  x: [1, 8192, 3],
  l: [1],
  cond: [1, 8192, 16],
  logits: [1, 8192, 8],
} as const

interface OutputGate extends TensorComparison {
  fractionWithinTolerance: number
  maxErrorIndex: number
  referenceAtMaxError: number
  candidateAtMaxError: number
  passed: boolean
}

interface OctreeLabResult {
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

interface FixtureFile {
  path: string
  shape: number[]
}

interface OctreeTrajectoryLevel {
  level: number
  resolution: number
  parent_count: number
  count_group_order: number[]
  files: Record<
    'parent_centers' | 'parent_counts' | 'logits' | 'sampled_child_counts' | 'parent_uniforms',
    FixtureFile
  >
}

interface OctreeTrajectoryManifest {
  settings: {
    levels: number
    num_points: number
    temperature: number
    internal_precision: string
  }
  condition: FixtureFile
  levels: OctreeTrajectoryLevel[]
  outputs: Record<'points' | 'log_probabilities' | 'voxel_jitter', FixtureFile>
}

interface OctreeTrajectoryResult {
  passed: boolean
  executionProvider: string
  modelLoadMs: number
  modelTransferBytes?: number
  inferenceMs: number
  readbackMs: number
  wallMs: number
  hostReplay: {
    passed: boolean
    points: OutputGate
    logProbabilities: OutputGate
    consumedRandomValues: number
    expectedRandomValues: number
  }
  levels: Array<{
    level: number
    resolution: number
    parentCount: number
    hostCenters: OutputGate
    logits: OutputGate
    paddingInvariant: OutputGate
    sampledChildCountsPassed: boolean
    sampledChildCountMismatches: number
    inferenceMs: number
    paddingProbeInferenceMs: number
  }>
  tolerance: { absolute: number; relative: number; minimumCosineSimilarity: number }
  environment: OctreeLabResult['environment']
}

declare global {
  interface Window {
    __TRIPOSPLAT_OCTREE_RESULT__?: OctreeLabResult
    __TRIPOSPLAT_OCTREE_TRAJECTORY_RESULT__?: OctreeTrajectoryResult
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

async function fetchUint32(url: string, expectedElements: number): Promise<Uint32Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`)
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength !== expectedElements * 4) {
    throw new Error(`${url} has ${buffer.byteLength} bytes; expected ${expectedElements * 4}.`)
  }
  return new Uint32Array(buffer)
}

async function fetchTrajectoryManifest(fixtureUrl: string): Promise<OctreeTrajectoryManifest> {
  const response = await fetch(`${fixtureUrl}/octree.json`)
  if (!response.ok) throw new Error(`Could not fetch ${fixtureUrl}/octree.json: HTTP ${response.status}`)
  const manifest = await response.json() as OctreeTrajectoryManifest
  if (
    !Array.isArray(manifest.levels)
    || manifest.levels.length !== manifest.settings?.levels
    || manifest.settings.internal_precision !== 'fp32'
  ) {
    throw new Error('Octree trajectory fixture does not implement the fp32 level contract.')
  }
  return manifest
}

function fixturePath(fixtureUrl: string, file: FixtureFile): string {
  return `${fixtureUrl}/${file.path}`
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
    throw new Error(`Output '${name}' is missing or is not float32.`)
  }
  return new Float32Array(payload.data)
}

function gate(
  reference: Float32Array,
  candidate: Float32Array,
  tolerance = { absolute: ATOL, relative: RTOL, minimumCosineSimilarity: MINIMUM_COSINE },
): OutputGate {
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

export function OctreeLab() {
  const clientRef = useRef<OrtWorkerClient | null>(null)
  const autoRunStartedRef = useRef(false)
  const runRef = useRef<() => Promise<void>>(async () => undefined)
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL)
  const [fixtureUrl, setFixtureUrl] = useState(DEFAULT_FIXTURE)
  const [trajectoryFixtureUrl, setTrajectoryFixtureUrl] = useState(DEFAULT_TRAJECTORY_FIXTURE)
  const [status, setStatus] = useState('Ready to validate occupancy logits.')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<OctreeLabResult | null>(null)
  const [trajectoryResult, setTrajectoryResult] = useState<OctreeTrajectoryResult | null>(null)

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
    setTrajectoryResult(null)
    delete window.__TRIPOSPLAT_OCTREE_RESULT__
    delete window.__TRIPOSPLAT_OCTREE_TRAJECTORY_RESULT__
    let client: OrtWorkerClient | undefined
    try {
      if (clientRef.current) await clientRef.current.dispose()
      const onStatus = (event: OrtWorkerStatus) => setStatus(event.message)
      client = new OrtWorkerClient({ onStatus })
      clientRef.current = client
      setStatus('Fetching the official fp32 occupancy fixture…')
      const [x, l, cond, reference] = await Promise.all([
        fetchFloat32(`${fixtureUrl}/x.f32`, elementCount(SHAPES.x)),
        fetchFloat32(`${fixtureUrl}/l.f32`, elementCount(SHAPES.l)),
        fetchFloat32(`${fixtureUrl}/cond.f32`, elementCount(SHAPES.cond)),
        fetchFloat32(`${fixtureUrl}/logits.f32`, elementCount(SHAPES.logits)),
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
          x: createTensorPayload('float32', x, SHAPES.x),
          l: createTensorPayload('float32', l, SHAPES.l),
          cond: createTensorPayload('float32', cond, SHAPES.cond),
        },
        outputs: ['logits'],
        tag: 'octree-occupancy-parity',
      })
      const comparison = gate(reference, payloadFloat32('logits', response.outputs.logits))
      const next: OctreeLabResult = {
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
      if (clientRef.current === client) clientRef.current = null
      window.__TRIPOSPLAT_OCTREE_RESULT__ = next
      setResult(next)
      setStatus(next.passed
        ? 'PASS: WebGPU occupancy logits match official fp32 PyTorch.'
        : 'FAIL: WebGPU occupancy logits exceed tolerance.')
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught)
      setError(message)
      setStatus('Occupancy validation failed.')
      if (client) await client.dispose().catch(() => undefined)
      if (clientRef.current === client) clientRef.current = null
    } finally {
      setBusy(false)
    }
  }

  const runTrajectory = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    setTrajectoryResult(null)
    delete window.__TRIPOSPLAT_OCTREE_RESULT__
    delete window.__TRIPOSPLAT_OCTREE_TRAJECTORY_RESULT__
    let client: OrtWorkerClient | undefined
    try {
      if (clientRef.current) await clientRef.current.dispose()
      client = new OrtWorkerClient({ onStatus: (event) => setStatus(event.message) })
      clientRef.current = client
      setStatus('Fetching the official eight-level octree trajectory…')
      const manifest = await fetchTrajectoryManifest(trajectoryFixtureUrl)
      const [condition, referencePoints, referenceLogProbabilities, voxelJitter, levelData] =
        await Promise.all([
          fetchFloat32(
            fixturePath(trajectoryFixtureUrl, manifest.condition),
            elementCount(manifest.condition.shape),
          ),
          fetchFloat32(
            fixturePath(trajectoryFixtureUrl, manifest.outputs.points),
            elementCount(manifest.outputs.points.shape),
          ),
          fetchFloat32(
            fixturePath(trajectoryFixtureUrl, manifest.outputs.log_probabilities),
            elementCount(manifest.outputs.log_probabilities.shape),
          ),
          fetchFloat32(
            fixturePath(trajectoryFixtureUrl, manifest.outputs.voxel_jitter),
            elementCount(manifest.outputs.voxel_jitter.shape),
          ),
          Promise.all(manifest.levels.map(async (level) => {
            const [parentCenters, parentCounts, logits, sampledChildCounts, parentUniforms] =
              await Promise.all([
                fetchFloat32(
                  fixturePath(trajectoryFixtureUrl, level.files.parent_centers),
                  elementCount(level.files.parent_centers.shape),
                ),
                fetchUint32(
                  fixturePath(trajectoryFixtureUrl, level.files.parent_counts),
                  elementCount(level.files.parent_counts.shape),
                ),
                fetchFloat32(
                  fixturePath(trajectoryFixtureUrl, level.files.logits),
                  elementCount(level.files.logits.shape),
                ),
                fetchUint32(
                  fixturePath(trajectoryFixtureUrl, level.files.sampled_child_counts),
                  elementCount(level.files.sampled_child_counts.shape),
                ),
                fetchFloat32(
                  fixturePath(trajectoryFixtureUrl, level.files.parent_uniforms),
                  elementCount(level.files.parent_uniforms.shape),
                ),
              ])
            return { manifest: level, parentCenters, parentCounts, logits, sampledChildCounts, parentUniforms }
          })),
        ])
      if (condition.length !== elementCount(SHAPES.cond)) {
        throw new Error(`Trajectory condition has ${condition.length} values; expected ${elementCount(SHAPES.cond)}.`)
      }

      const systematicRandomValues: number[] = []
      const sampledCountChecks = levelData.map((level) => {
        const observedGroupOrder = [...new Set(level.parentCounts)].sort((left, right) => left - right)
        if (observedGroupOrder.join(',') !== level.manifest.count_group_order.join(',')) {
          throw new Error(`Level ${level.manifest.level} count-group order differs from official PyTorch.`)
        }
        for (const count of observedGroupOrder) {
          for (let parent = 0; parent < level.parentCounts.length; parent += 1) {
            if (level.parentCounts[parent] === count) {
              systematicRandomValues.push(level.parentUniforms[parent])
            }
          }
        }
        const distribution = octreeSoftmax(
          level.logits,
          level.manifest.parent_count,
          manifest.settings.temperature,
        )
        let mismatches = 0
        for (let parent = 0; parent < level.parentCounts.length; parent += 1) {
          const offset = parent * 8
          const sampled = systematicResample(
            distribution.probabilities.subarray(offset, offset + 8),
            level.parentCounts[parent],
            () => level.parentUniforms[parent],
          )
          for (let child = 0; child < 8; child += 1) {
            if (sampled[child] !== level.sampledChildCounts[offset + child]) mismatches += 1
          }
        }
        return mismatches
      })
      const randomValues = [...systematicRandomValues, ...voxelJitter]
      let randomIndex = 0

      const sidecarUrl = `${modelUrl}.data`
      const graphName = new URL(modelUrl, document.baseURI).pathname.split('/').at(-1)
      if (!graphName) throw new Error(`Could not derive graph name from ${modelUrl}.`)
      const transferParts = await Promise.all([contentLength(modelUrl), contentLength(sidecarUrl)])
      const modelTransferBytes = transferParts.every((value) => value !== undefined)
        ? transferParts.reduce<number>((sum, value) => sum + (value ?? 0), 0)
        : undefined
      const loaded = await client.loadSession({
        sessionId: `${SESSION_ID}/trajectory`,
        manifest: {
          graphUrl: modelUrl,
          externalData: [{ path: `${decodeURIComponent(graphName)}.data`, url: sidecarUrl }],
        },
        options: { allowWasmFallback: false, graphOptimizationLevel: 'disabled' },
      })
      if (loaded.executionProvider !== 'webgpu') {
        throw new Error(`Expected WebGPU, loaded ${loaded.executionProvider}.`)
      }

      let inferenceMs = 0
      let readbackMs = 0
      const records: OctreeTrajectoryResult['levels'] = []
      const started = performance.now()
      const replay = await sampleOctree(
        async (invocation) => {
          const fixture = levelData[invocation.level - 1]
          if (!fixture || fixture.manifest.resolution !== invocation.resolution) {
            throw new Error(`Missing official fixture for octree level ${invocation.level}.`)
          }
          setStatus(
            `Octree level ${invocation.level}/${manifest.settings.levels}: `
            + `running official-input WebGPU and padding probes…`,
          )
          const hostCenters = gate(
            fixture.parentCenters,
            invocation.parentCenters,
            { absolute: 0, relative: 0, minimumCosineSimilarity: 0.999999999999 },
          )
          const paddedCenters = new Float32Array(elementCount(SHAPES.x))
          paddedCenters.set(fixture.parentCenters)
          const alternatePadding = new Float32Array(paddedCenters)
          for (let parent = fixture.manifest.parent_count; parent < SHAPES.x[1]; parent += 1) {
            const offset = parent * 3
            alternatePadding[offset] = ((parent % 251) + 0.5) / 256
            alternatePadding[offset + 1] = (((parent * 3) % 251) + 0.5) / 256
            alternatePadding[offset + 2] = (((parent * 7) % 251) + 0.5) / 256
          }
          const inputs = (x: Float32Array) => ({
            x: createTensorPayload('float32', x, SHAPES.x),
            l: createTensorPayload('float32', Float32Array.of(invocation.resolution), SHAPES.l),
            cond: createTensorPayload('float32', new Float32Array(condition), SHAPES.cond),
          })
          const response = await client!.runSession({
            sessionId: `${SESSION_ID}/trajectory`,
            inputs: inputs(paddedCenters),
            outputs: ['logits'],
            tag: `octree-teacher-level-${invocation.level}`,
          })
          const paddingResponse = await client!.runSession({
            sessionId: `${SESSION_ID}/trajectory`,
            inputs: inputs(alternatePadding),
            outputs: ['logits'],
            tag: `octree-padding-level-${invocation.level}`,
          })
          inferenceMs += response.timings.inferenceMs + paddingResponse.timings.inferenceMs
          readbackMs += response.timings.readbackMs + paddingResponse.timings.readbackMs
          const activeLength = fixture.manifest.parent_count * 8
          const candidate = payloadFloat32('logits', response.outputs.logits).slice(0, activeLength)
          const paddingCandidate = payloadFloat32(
            'logits',
            paddingResponse.outputs.logits,
          ).slice(0, activeLength)
          records.push({
            level: invocation.level,
            resolution: invocation.resolution,
            parentCount: invocation.parentCount,
            hostCenters,
            logits: gate(fixture.logits, candidate),
            paddingInvariant: gate(
              candidate,
              paddingCandidate,
              { absolute: 0.000001, relative: 0.000001, minimumCosineSimilarity: 0.999999999999 },
            ),
            sampledChildCountsPassed: sampledCountChecks[invocation.level - 1] === 0,
            sampledChildCountMismatches: sampledCountChecks[invocation.level - 1] ?? -1,
            inferenceMs: response.timings.inferenceMs,
            paddingProbeInferenceMs: paddingResponse.timings.inferenceMs,
          })
          // Return official logits so the host replay isolates TypeScript softmax,
          // systematic resampling, compaction, expansion, and final jitter.
          return { logits: new Float32Array(fixture.logits) }
        },
        {
          condition,
          numPoints: manifest.settings.num_points,
          levels: manifest.settings.levels,
          temperature: manifest.settings.temperature,
          rng: () => {
            const value = randomValues[randomIndex]
            if (value === undefined) throw new Error('Octree replay exhausted recorded random values.')
            randomIndex += 1
            return value
          },
        },
      )
      const hostReplay = {
        passed: false,
        points: gate(
          referencePoints,
          replay.points,
          { absolute: 0.0000001, relative: 0, minimumCosineSimilarity: 0.999999999999 },
        ),
        logProbabilities: gate(
          referenceLogProbabilities,
          replay.logProbabilities,
          { absolute: 0.000002, relative: 0.000001, minimumCosineSimilarity: 0.999999999999 },
        ),
        consumedRandomValues: randomIndex,
        expectedRandomValues: randomValues.length,
      }
      hostReplay.passed = hostReplay.points.passed
        && hostReplay.logProbabilities.passed
        && randomIndex === randomValues.length
      const passed = records.length === manifest.settings.levels
        && records.every((record) => record.hostCenters.passed
          && record.logits.passed
          && record.paddingInvariant.passed
          && record.sampledChildCountsPassed)
        && hostReplay.passed
      const next: OctreeTrajectoryResult = {
        passed,
        executionProvider: loaded.executionProvider,
        modelLoadMs: loaded.loadMs,
        ...(modelTransferBytes === undefined ? {} : { modelTransferBytes }),
        inferenceMs,
        readbackMs,
        wallMs: performance.now() - started,
        hostReplay,
        levels: records,
        tolerance: { absolute: ATOL, relative: RTOL, minimumCosineSimilarity: MINIMUM_COSINE },
        environment: {
          userAgent: navigator.userAgent,
          crossOriginIsolated: self.crossOriginIsolated,
          webgpu: 'gpu' in navigator,
        },
      }
      await client.dispose()
      if (clientRef.current === client) clientRef.current = null
      window.__TRIPOSPLAT_OCTREE_TRAJECTORY_RESULT__ = next
      setTrajectoryResult(next)
      setStatus(next.passed
        ? 'PASS: all eight WebGPU occupancy calls and the TypeScript octree replay match official PyTorch.'
        : 'FAIL: at least one neural, padding, resampling, or host-replay octree gate failed.')
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught)
      setError(message)
      setStatus('Eight-level octree validation failed.')
      if (client) await client.dispose().catch(() => undefined)
      if (clientRef.current === client) clientRef.current = null
    } finally {
      setBusy(false)
    }
  }

  runRef.current = run
  useEffect(() => {
    const timeout = window.setTimeout(() => {
      if (new URLSearchParams(location.search).get('autorun') === '1' && !autoRunStartedRef.current) {
        autoRunStartedRef.current = true
        void runRef.current()
      }
    }, 0)
    return () => clearTimeout(timeout)
  }, [])

  return <main>
    <h1>TripoSplat · occupancy WebGPU parity</h1>
    <p>Validates one occupancy call or replays the complete official eight-level dynamic octree.</p>
    <label>ONNX graph <input value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} /></label>
    <label>Fixture directory <input value={fixtureUrl} onChange={(event) => setFixtureUrl(event.target.value)} /></label>
    <button type="button" disabled={busy} onClick={() => void run()}>{busy ? 'Running…' : 'Run occupancy parity gate'}</button>
    <label>Trajectory fixture <input value={trajectoryFixtureUrl} onChange={(event) => setTrajectoryFixtureUrl(event.target.value)} /></label>
    <button type="button" disabled={busy} onClick={() => void runTrajectory()}>{busy ? 'Running…' : 'Run eight-level trajectory gate'}</button>
    <p role="status" data-testid="octree-status">{status}</p>
    {error ? <pre className="error" data-testid="octree-error">{error}</pre> : null}
    {result ? <pre data-testid="octree-result">{JSON.stringify(result, null, 2)}</pre> : null}
    {trajectoryResult
      ? <pre data-testid="octree-trajectory-result">{JSON.stringify(trajectoryResult, null, 2)}</pre>
      : null}
  </main>
}

const style = document.createElement('style')
style.textContent = `
  :root { color: #ececf3; background: #101014; font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  body { margin: 0; } main { max-width: 920px; margin: 0 auto; padding: 48px 24px; }
  h1 { font: 600 28px/1.2 system-ui, sans-serif; }
  label { display: grid; gap: 6px; margin: 18px 0; }
  input { box-sizing: border-box; width: 100%; padding: 10px; color: inherit; background: #1b1b22; border: 1px solid #3a3a48; border-radius: 6px; }
  button { margin-right: 10px; padding: 10px 16px; color: #08080a; background: #f8cf00; border: 0; border-radius: 6px; font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .55; cursor: wait; } pre { overflow: auto; padding: 16px; background: #18181f; border-radius: 8px; }
  .error { color: #ff9b9b; }
`
document.head.append(style)

createRoot(document.getElementById('root')!).render(<StrictMode><OctreeLab /></StrictMode>)
