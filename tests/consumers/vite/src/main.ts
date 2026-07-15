import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'
import {
  createFlowSchedule,
  createRuntime,
  createTensor,
} from '@ai3d/triposplat-webgpu/low-level'
import { exportPLY, exportSplat } from '@ai3d/triposplat-webgpu/export'

declare global {
  interface Window {
    __TRIPOSPLAT_PACKED_FULL_RESULT__?: PackedFullResult
  }
}

interface PackedFullResult {
  passed: boolean
  packagePath: 'packed-vite-consumer'
  numericalParityClaimed: false
  modelBaseUrl: string
  count?: number
  plyBytes?: number
  splatBytes?: number
  allArraysFinite?: boolean
  timingsMs: Record<string, number>
  pipelineMeasuredTimingsMs?: unknown
  error?: { name: string; message: string; stack?: string }
}

const model = new TripoSplatWebGPU({
  modelBaseUrl: 'https://models.example.invalid/triposplat/v1/',
  cache: 'none',
})

Object.assign(globalThis, {
  triposplatPackageSmoke: {
    model,
    createRuntime,
    schedule: createFlowSchedule(4, 3),
    exportPLY,
    exportSplat,
  },
})

const IDENTITY_ONNX_BASE64 =
  'CAk6RgoQCgF4EgF5IghJZGVudGl0eRIIaWRlbnRpdHlaEwoBeBIOCgwIARIICgIIAQoCCARiEwoBeRIOCgwIARIICgIIAQoCCARCBAoAEBQ='

async function runInstalledPackageSmoke(): Promise<void> {
  document.body.dataset.smoke = 'running'
  const bytes = Uint8Array.from(atob(IDENTITY_ONNX_BASE64), (character) => character.charCodeAt(0))
  const graphUrl = URL.createObjectURL(new Blob([bytes], { type: 'application/octet-stream' }))
  const runtime = createRuntime({ executionProviders: ['webgpu'] })
  try {
    const graph = await runtime.loadGraph('identity', { url: graphUrl })
    const result = await runtime.runGraph('identity', {
      x: createTensor('float32', Float32Array.of(1, 2, 3, 4), [1, 4]),
    })
    const output = result.outputs.y
    if (graph.executionProvider !== 'webgpu' || output?.type !== 'float32') {
      throw new Error('The installed package did not execute the identity graph with WebGPU.')
    }
    const values = Array.from(output.data)
    if (values.join(',') !== '1,2,3,4') throw new Error(`Unexpected identity output: ${values}`)
    document.body.dataset.smoke = 'pass'
    document.body.textContent = JSON.stringify({ provider: graph.executionProvider, values })
  } finally {
    await runtime.dispose()
    URL.revokeObjectURL(graphUrl)
  }
}

async function fetchFloat32(url: string, expectedElements: number): Promise<Float32Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}.`)
  const bytes = await response.arrayBuffer()
  if (bytes.byteLength !== expectedElements * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(`${url} has ${bytes.byteLength} bytes; expected ${expectedElements * 4}.`)
  }
  return new Float32Array(bytes)
}

function allFinite(values: Float32Array | undefined): boolean {
  if (values === undefined) return false
  for (const value of values) if (!Number.isFinite(value)) return false
  return true
}

function queryUrl(parameters: URLSearchParams, name: string, fallback: string): string {
  return parameters.get(name) ?? fallback
}

async function runInstalledPackageFull(): Promise<void> {
  document.body.dataset.full = 'running'
  const parameters = new URLSearchParams(location.search)
  const fixtureOrigin = queryUrl(parameters, 'fixtureOrigin', 'http://127.0.0.1:5173')
    .replace(/\/$/, '')
  const modelBaseUrl = queryUrl(
    parameters,
    'modelBaseUrl',
    `${fixtureOrigin}/models/triposplat/`,
  )
  const startedAt = performance.now()
  const timingsMs: Record<string, number> = {}
  let fullModel: TripoSplatWebGPU | undefined
  let scene: Awaited<ReturnType<TripoSplatWebGPU['generate']>> | undefined
  try {
    document.body.textContent = 'Fetching deterministic full-pipeline fixtures…'
    const fixtureStartedAt = performance.now()
    const flowBase = `${fixtureOrigin}/fixtures/generated/flow4-fp32-compute`
    const [imageResponse, vaeNoise, latentNoise, cameraNoise] = await Promise.all([
      fetch(`${fixtureOrigin}/fixtures/generated/dinov3-fp32/prepared.png`),
      fetchFloat32(
        `${fixtureOrigin}/fixtures/generated/flux2-vae-fp32/epsilon.f32`,
        1 * 32 * 128 * 128,
      ),
      fetchFloat32(`${flowBase}/latent.f32`, 1 * 8192 * 16),
      fetchFloat32(`${flowBase}/camera.f32`, 1 * 1 * 5),
    ])
    if (!imageResponse.ok) {
      throw new Error(`Could not fetch prepared image: HTTP ${imageResponse.status}.`)
    }
    const preparedImage = await imageResponse.blob()
    timingsMs.fixtureFetch = performance.now() - fixtureStartedAt

    fullModel = new TripoSplatWebGPU({
      modelBaseUrl,
      cache: 'none',
      executionProviders: ['webgpu'],
      logLevel: 'info',
    })
    const loadStartedAt = performance.now()
    await fullModel.load({
      onProgress(progress) {
        document.body.textContent = `Packed load: ${progress.message}`
      },
    })
    timingsMs.load = performance.now() - loadStartedAt

    const generateStartedAt = performance.now()
    scene = await fullModel.generate(preparedImage, {
      steps: 4,
      guidanceScale: 3,
      shift: 3,
      seed: 42,
      inputIsPrepared: true,
      vaeNoise,
      latentNoise,
      cameraNoise,
      onProgress(progress) {
        document.body.textContent = `Packed generate: ${progress.message}`
      },
    })
    timingsMs.generate = performance.now() - generateStartedAt

    const allArraysFinite = allFinite(scene.positions)
      && allFinite(scene.scales)
      && allFinite(scene.rotations)
      && allFinite(scene.opacities)
      && allFinite(scene.sphericalHarmonics)
    const exportStartedAt = performance.now()
    const [ply, splat] = await Promise.all([scene.exportPLY(), scene.exportSplat()])
    timingsMs.exports = performance.now() - exportStartedAt
    const passed = scene.count === 262_144
      && allArraysFinite
      && ply.size === 17_826_208
      && splat.size === 8_388_608
    const result: PackedFullResult = {
      passed,
      packagePath: 'packed-vite-consumer',
      numericalParityClaimed: false,
      modelBaseUrl,
      count: scene.count,
      plyBytes: ply.size,
      splatBytes: splat.size,
      allArraysFinite,
      timingsMs: { ...timingsMs, total: performance.now() - startedAt },
      pipelineMeasuredTimingsMs: scene.metadata.generationSettings.measuredTimingsMs,
    }
    window.__TRIPOSPLAT_PACKED_FULL_RESULT__ = result
    document.body.dataset.full = passed ? 'pass' : 'fail'
    document.body.textContent = JSON.stringify(result, null, 2)
  } catch (caught) {
    const error = caught instanceof Error ? caught : new Error(String(caught))
    const result: PackedFullResult = {
      passed: false,
      packagePath: 'packed-vite-consumer',
      numericalParityClaimed: false,
      modelBaseUrl,
      timingsMs: { ...timingsMs, total: performance.now() - startedAt },
      error: { name: error.name, message: error.message, stack: error.stack },
    }
    window.__TRIPOSPLAT_PACKED_FULL_RESULT__ = result
    document.body.dataset.full = 'fail'
    document.body.textContent = JSON.stringify(result, null, 2)
  } finally {
    scene?.dispose()
    await fullModel?.dispose().catch(() => undefined)
  }
}

const parameters = new URLSearchParams(location.search)
if (parameters.get('full') === '1') {
  void runInstalledPackageFull()
} else if (parameters.get('run') === '1') {
  runInstalledPackageSmoke().catch((error: unknown) => {
    document.body.dataset.smoke = 'fail'
    document.body.textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  })
}
