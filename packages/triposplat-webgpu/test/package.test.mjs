import assert from 'node:assert/strict'
import test from 'node:test'

import { TripoSplatWebGPU } from '../dist/index.js'
import { exportPLY, exportSplat } from '../dist/export.js'
import { prepareReusableGraphInputs } from '../dist/runtime.js'
import {
  blendGuidance,
  createFlowSchedule,
  createRuntime,
  createSampler,
  decodeGaussians,
  expandOctreeFrontier,
  parseModelManifest,
  resolveModelManifest,
  sampleOctree,
  systematicResample,
} from '../dist/low-level.js'

const metadata = {
  coordinateSystem: 'triposplat-object',
  units: 'model-unit',
  rotationOrder: 'wxyz',
  scaleEncoding: 'linear',
  opacityEncoding: 'linear',
  colorSemantics: null,
  sphericalHarmonicsSemantics: 'degree-0-rgb',
  modelRevision: 'test-revision',
  generationSettings: { steps: 4 },
  seed: 42,
  runtimeVersion: 'test',
}

test('manifest parser resolves multiple external-data shards without changing virtual paths', () => {
  const manifest = parseModelManifest({
    name: 'triposplat-webgpu',
    version: '1.0.0',
    modelRevision: 'abc',
    precision: 'fp16',
    graphs: {
      dit: {
        url: 'graphs/dit.onnx',
        externalData: [
          { path: 'weights.0', url: 'weights/dit.0' },
          { path: 'weights.1', url: 'weights/dit.1' },
        ],
      },
    },
  })
  const resolved = resolveModelManifest(manifest, 'https://models.example/v1/manifest.json')
  assert.equal(resolved.graphs.dit.url, 'https://models.example/v1/graphs/dit.onnx')
  assert.deepEqual(resolved.graphs.dit.externalData.map(({ path }) => path), ['weights.0', 'weights.1'])
  assert.deepEqual(
    resolved.graphs.dit.externalData.map(({ url }) => url),
    ['https://models.example/v1/weights/dit.0', 'https://models.example/v1/weights/dit.1'],
  )
})

test('flow sampler uses the official shifted schedule and conditional/unconditional CFG calls', async () => {
  assert.deepEqual(
    createFlowSchedule(4, 3).map(({ timestep }) => timestep),
    [1, 0.9, 0.75, 0.5],
  )
  const calls = []
  const sampler = createSampler((invocation) => {
    calls.push(`${invocation.step}:${invocation.pass}`)
    return { latent: Float32Array.of(invocation.pass === 'conditional' ? 2 : 1) }
  })
  const result = await sampler.sample(
    { latent: Float32Array.of(5) },
    {
      condition: 'positive',
      negativeCondition: 'negative',
      steps: 1,
      shift: 1,
      guidanceScale: 3,
      arithmetic: 'fp32',
    },
  )
  assert.deepEqual(calls, ['1:conditional', '1:unconditional'])
  assert.deepEqual(Array.from(result.latent), [1])
})

test('fp32 CFG preserves PyTorch per-operation rounding', () => {
  const conditional = { latent: Float32Array.of(0.42336001992225647) }
  const unconditional = { latent: Float32Array.of(0.7559554576873779) }
  const result = blendGuidance(conditional, unconditional, 3, 'fp32')
  assert.equal(result.latent[0], -0.24183082580566406)
})

test('Gaussian decoder returns canonical scenes with both PLY and splat export', async () => {
  const scene = decodeGaussians(
    Float32Array.of(0.5, 0.5, 0.5),
    new Float32Array(480),
    { metadata },
  )
  assert.equal(scene.count, 32)
  const ply = await exportPLY(scene)
  const splat = await exportSplat(scene)
  assert.equal(ply.type, 'application/octet-stream')
  assert.equal(splat.size, 32 * 32)
  scene.dispose()
  assert.equal(scene.isDisposed, true)
})

test('high-level facade does not claim generation before a manifest is loaded', () => {
  const model = new TripoSplatWebGPU({ modelBaseUrl: 'https://models.example/v1/' })
  assert.equal(model.capabilities.manifestLoaded, false)
  assert.equal(model.capabilities.fullGeneration, false)
  assert.deepEqual(model.capabilities.missingGraphs, ['dino', 'vae', 'dit', 'octree', 'gaussianDecoder'])
})

test('systematic octree resampling preserves official underfull-row behavior', () => {
  assert.deepEqual(
    [...systematicResample([0.2, 0.2, 0, 0, 0, 0, 0, 0], 4, () => 0.5)],
    [1, 1, 0, 0, 0, 0, 0, 2],
  )
  assert.deepEqual([...systematicResample([2, 2], 4, () => 0.5)], [2, 2])
  assert.deepEqual([...systematicResample([-3, -1, 0], 3, () => 0.5)], [1, 1, 1])
  assert.deepEqual([...systematicResample([0.5, 0.5], 1, () => 0.5)], [1, 0])
})

test('octree compaction preserves count, x-fastest child order, and accumulated log mass', () => {
  const result = expandOctreeFrontier(
    {
      coordinates: Int32Array.of(2, 3, 4),
      counts: Uint32Array.of(4),
      logProbabilities: Float32Array.of(-2),
    },
    {
      probabilities: Float32Array.of(0.5, 0, 0, 0, 0, 0, 0, 0.5),
      logProbabilities: Float32Array.of(Math.log(0.5), -99, -99, -99, -99, -99, -99, Math.log(0.5)),
    },
    () => 0.5,
  )
  assert.deepEqual([...result.coordinates], [4, 6, 8, 5, 7, 9])
  assert.deepEqual([...result.counts], [2, 2])
  assert.ok(Math.abs(result.logProbabilities[0] - (-2 + Math.log(0.5))) < 1e-6)
  assert.ok(Math.abs(result.logProbabilities[1] - (-2 + Math.log(0.5))) < 1e-6)
})

test('octree systematic offsets follow official ascending count-group RNG order', () => {
  const random = [0.9, 0.1]
  const result = expandOctreeFrontier(
    {
      coordinates: Int32Array.of(0, 0, 0, 1, 0, 0),
      counts: Uint32Array.of(2, 1),
      logProbabilities: Float32Array.of(0, 0),
    },
    {
      probabilities: Float32Array.of(
        0.5, 0.5, 0, 0, 0, 0, 0, 0,
        0.5, 0.5, 0, 0, 0, 0, 0, 0,
      ),
      logProbabilities: new Float32Array(16),
    },
    () => random.shift(),
  )
  assert.deepEqual([...result.coordinates], [0, 0, 0, 1, 0, 0, 3, 0, 0])
  assert.deepEqual([...result.counts], [1, 1, 1])
})

test('eight-level octree host loop follows the official forced-child oracle', async () => {
  let rngCalls = 0
  const result = await sampleOctree(
    ({ parentCount }) => {
      const logits = new Float32Array(parentCount * 8)
      for (let parent = 0; parent < parentCount; parent += 1) {
        logits.fill(-80, parent * 8, parent * 8 + 8)
        logits[parent * 8 + 5] = 80
      }
      return { logits }
    },
    {
      condition: null,
      numPoints: 1,
      levels: 8,
      rng: () => {
        rngCalls += 1
        return 0.5
      },
    },
  )
  assert.deepEqual([...result.leaves.coordinates], [255, 0, 255])
  assert.deepEqual([...result.points], [255.5 / 256, 0.5 / 256, 255.5 / 256])
  assert.equal(result.resolution, 256)
  assert.equal(rngCalls, 11)
})

test('runtime accepts a custom worker factory and disposes it cleanly', async () => {
  let terminated = false
  const worker = {
    onmessage: null,
    onerror: null,
    onmessageerror: null,
    postMessage(request) {
      queueMicrotask(() => {
        const result = request.type === 'configure'
          ? { operation: 'configure', configured: true }
          : { operation: 'dispose', disposedSessionIds: [] }
        this.onmessage?.({ data: { type: 'reply', requestId: request.requestId, ok: true, result } })
      })
    },
    terminate() { terminated = true },
  }
  const runtime = createRuntime({
    baseUrl: 'https://app.example/',
    workerFactory: () => worker,
  })
  await runtime.dispose()
  assert.equal(runtime.disposed, true)
  assert.equal(terminated, true)
})

test('built-in runtime transfers reusable inputs once and references them on later runs', async () => {
  const messages = []
  let terminated = false
  const worker = {
    onmessage: null,
    onerror: null,
    onmessageerror: null,
    postMessage(request, transfer = []) {
      const transferBytes = transfer.reduce((total, buffer) => total + buffer.byteLength, 0)
      const delivered = structuredClone(request, { transfer })
      messages.push({
        request: delivered,
        transferBytes,
      })
      queueMicrotask(() => {
        let result
        if (delivered.type === 'configure') {
          result = { operation: 'configure', configured: true }
        } else if (delivered.type === 'retain-inputs') {
          result = { operation: 'retain-inputs', retainedInputNames: Object.keys(delivered.inputs) }
        } else if (delivered.type === 'run') {
          result = {
            operation: 'run',
            result: { outputs: {}, timings: { inferenceMs: 1, readbackMs: 0, totalMs: 1 } },
          }
        } else {
          result = { operation: 'dispose', disposedSessionIds: [] }
        }
        this.onmessage?.({
          data: { type: 'reply', requestId: delivered.requestId, ok: true, result },
        })
      })
    },
    terminate() { terminated = true },
  }
  const runtime = createRuntime({ workerFactory: () => worker })
  const conditioning = new Float32Array(32)
  conditioning.fill(7)
  const retained = await prepareReusableGraphInputs(
    runtime,
    'test/dit',
    'positive-conditioning',
    { feature: { type: 'float32', data: conditioning, dims: [1, 32] } },
  )
  assert.equal(conditioning.byteLength, 0)

  await retained.run({ latent: { type: 'float32', data: Float32Array.of(1), dims: [1] } })
  await retained.run({ latent: { type: 'float32', data: Float32Array.of(2), dims: [1] } })
  await assert.rejects(
    async () => retained.run({ feature: { type: 'float32', data: Float32Array.of(3), dims: [1] } }),
    /both reusable and dynamic/,
  )
  await runtime.dispose()

  const retainMessages = messages.filter(({ request }) => request.type === 'retain-inputs')
  const runMessages = messages.filter(({ request }) => request.type === 'run')
  assert.equal(retainMessages.length, 1)
  assert.equal(retainMessages[0].transferBytes, 32 * Float32Array.BYTES_PER_ELEMENT)
  assert.deepEqual(Object.keys(retainMessages[0].request.inputs), ['feature'])
  assert.equal(runMessages.length, 2)
  assert.ok(runMessages.every(({ request }) => request.reusableInputsId === 'positive-conditioning'))
  assert.ok(runMessages.every(({ request }) => Object.keys(request.inputs).join(',') === 'latent'))
  assert.ok(runMessages.every(({ transferBytes }) => transferBytes === Float32Array.BYTES_PER_ELEMENT))
  assert.equal(terminated, true)
})

test('reusable input fallback clones immutable tensors for custom runtimes', async () => {
  const received = []
  const receivedValues = []
  const runtime = {
    disposed: false,
    async loadGraph() { throw new Error('not used') },
    async runGraph(_sessionId, inputs) {
      received.push(inputs.feature.data)
      receivedValues.push(Array.from(inputs.feature.data))
      structuredClone(inputs, {
        transfer: Object.values(inputs).map(({ data }) => data.buffer),
      })
      return { outputs: {}, timings: { inferenceMs: 1, readbackMs: 0, totalMs: 1 } }
    },
    async disposeGraph() { return true },
    async dispose() {},
  }
  const retained = await prepareReusableGraphInputs(
    runtime,
    'custom/dit',
    'conditioning',
    { feature: { type: 'float32', data: Float32Array.of(4, 5), dims: [1, 2] } },
  )
  await retained.run({ latent: { type: 'float32', data: Float32Array.of(1), dims: [1] } })
  await retained.run({ latent: { type: 'float32', data: Float32Array.of(2), dims: [1] } })
  assert.equal(received.length, 2)
  assert.notEqual(received[0], received[1])
  assert.deepEqual(receivedValues, [[4, 5], [4, 5]])
  assert.equal(received[0].byteLength, 0)
  assert.equal(received[1].byteLength, 0)
})
