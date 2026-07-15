import assert from 'node:assert/strict'
import test from 'node:test'

import { runBuiltInTripoSplatPipeline } from '../dist/index.js'
import {
  TRIPOSPLAT_CAMERA_SHAPE,
  TRIPOSPLAT_FEATURE1_SHAPE,
  TRIPOSPLAT_FEATURE2_SHAPE,
  TRIPOSPLAT_IMAGE_SHAPE,
  TRIPOSPLAT_LATENT_SHAPE,
  TRIPOSPLAT_MAX_DECODER_POINTS,
  TRIPOSPLAT_VAE_NOISE_SHAPE,
} from '../dist/low-level.js'

const count = (shape) => shape.reduce((product, dimension) => product * dimension, 1)
const tensor = (data, dims) => ({ type: 'float32', data, dims: [...dims] })

const contracts = {
  dino: { inputs: ['pixel_values'], outputs: ['feature1'] },
  vae: { inputs: ['image_rgb', 'epsilon'], outputs: ['feature2'] },
  dit: {
    inputs: ['latent', 'camera', 't', 'feature1', 'feature2'],
    outputs: ['pred_latent', 'pred_camera'],
  },
  octree: { inputs: ['x', 'l', 'cond'], outputs: ['logits'] },
  gaussianDecoder: { inputs: ['points', 'cond'], outputs: ['features'] },
}

test('reusable DiT conditioning has a fixed two-upload host-transfer budget', () => {
  const conditioningBytes = (
    count(TRIPOSPLAT_FEATURE1_SHAPE) + count(TRIPOSPLAT_FEATURE2_SHAPE)
  ) * Float32Array.BYTES_PER_ELEMENT
  const retainedBytes = conditioningBytes * 2
  assert.equal(conditioningBytes, 23_096_832)
  assert.deepEqual({
    fast: {
      previousBytes: conditioningBytes * 8,
      retainedBytes,
      avoidedBytes: conditioningBytes * 8 - retainedBytes,
    },
    quality: {
      previousBytes: conditioningBytes * 40,
      retainedBytes,
      avoidedBytes: conditioningBytes * 40 - retainedBytes,
    },
  }, {
    fast: { previousBytes: 184_774_656, retainedBytes: 46_193_664, avoidedBytes: 138_580_992 },
    quality: { previousBytes: 923_873_280, retainedBytes: 46_193_664, avoidedBytes: 877_679_616 },
  })
})

test('built-in pipeline stages every graph, preserves public dtypes, and returns a canonical scene', async () => {
  const sessionIds = {
    dino: 'test/dino',
    vae: 'test/vae',
    dit: 'test/dit',
    octree: 'test/octree',
    gaussianDecoder: 'test/gaussian',
  }
  const graphBySession = Object.fromEntries(
    Object.entries(sessionIds).map(([graph, session]) => [session, graph]),
  )
  const loaded = []
  const disposed = []
  const runs = []
  const runtime = {
    disposed: false,
    async loadGraph(sessionId) {
      const graph = graphBySession[sessionId]
      loaded.push(graph)
      return {
        sessionId,
        executionProvider: 'webgpu',
        inputNames: [...contracts[graph].inputs],
        outputNames: [...contracts[graph].outputs],
        loadMs: 1,
      }
    },
    async runGraph(sessionId, inputs, options) {
      const graph = graphBySession[sessionId]
      runs.push({ graph, types: Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, value.type])) })
      let outputs
      if (graph === 'dino') {
        outputs = { feature1: tensor(new Float32Array(count(TRIPOSPLAT_FEATURE1_SHAPE)), TRIPOSPLAT_FEATURE1_SHAPE) }
      } else if (graph === 'vae') {
        outputs = { feature2: tensor(new Float32Array(count(TRIPOSPLAT_FEATURE2_SHAPE)), TRIPOSPLAT_FEATURE2_SHAPE) }
      } else if (graph === 'dit') {
        outputs = {
          pred_latent: tensor(new Float32Array(count(TRIPOSPLAT_LATENT_SHAPE)), TRIPOSPLAT_LATENT_SHAPE),
          pred_camera: tensor(new Float32Array(count(TRIPOSPLAT_CAMERA_SHAPE)), TRIPOSPLAT_CAMERA_SHAPE),
        }
      } else if (graph === 'octree') {
        const logits = new Float32Array(TRIPOSPLAT_MAX_DECODER_POINTS * 8)
        for (let parent = 0; parent < TRIPOSPLAT_MAX_DECODER_POINTS; parent += 1) {
          logits.fill(-80, parent * 8, parent * 8 + 8)
          logits[parent * 8 + 5] = 80
        }
        outputs = { logits: tensor(logits, [1, TRIPOSPLAT_MAX_DECODER_POINTS, 8]) }
      } else {
        outputs = {
          features: tensor(
            new Float32Array(TRIPOSPLAT_MAX_DECODER_POINTS * 480),
            [1, TRIPOSPLAT_MAX_DECODER_POINTS, 480],
          ),
        }
      }
      return { outputs, timings: { inferenceMs: 1, readbackMs: 0.1, totalMs: 1.1 }, options }
    },
    async disposeGraph(sessionId) {
      disposed.push(graphBySession[sessionId])
      return true
    },
    async dispose() {},
  }
  const graph = (name, extra = {}) => ({
    url: `https://models.example/${name}.onnx`,
    precision: 'fp16',
    ...extra,
  })
  const manifest = {
    name: 'triposplat-webgpu',
    version: 'test',
    modelRevision: 'a78fa12d',
    precision: 'fp16',
    sourceUrl: 'https://models.example/manifest.json',
    graphs: {
      dino: graph('dino'),
      vae: graph('vae', { inputPrecision: 'fp16' }),
      dit: graph('dit'),
      octree: graph('octree'),
      gaussianDecoder: graph('gaussian'),
    },
  }
  let preparedDisposed = false
  const preprocessor = async () => ({
    image: { width: 1024, height: 1024, data: new Uint8ClampedArray(1024 * 1024 * 3) },
    foreground: { width: 1, height: 1, data: new Uint8ClampedArray(4) },
    usedBackgroundRemoval: false,
    tensors: {
      rgb: { data: new Float32Array(count(TRIPOSPLAT_IMAGE_SHAPE)), dims: TRIPOSPLAT_IMAGE_SHAPE },
      dinov3: { data: new Float32Array(count(TRIPOSPLAT_IMAGE_SHAPE)), dims: TRIPOSPLAT_IMAGE_SHAPE },
      vae: { data: new Float32Array(0), dims: TRIPOSPLAT_IMAGE_SHAPE },
    },
    canvas: {},
    dispose() { preparedDisposed = true },
  })
  const progress = []
  const scene = await runBuiltInTripoSplatPipeline({
    input: new Blob(),
    options: {
      steps: 4,
      inputIsPrepared: true,
      vaeNoise: new Float32Array(count(TRIPOSPLAT_VAE_NOISE_SHAPE)),
      latentNoise: new Float32Array(count(TRIPOSPLAT_LATENT_SHAPE)),
      cameraNoise: new Float32Array(count(TRIPOSPLAT_CAMERA_SHAPE)),
      onProgress: ({ stage }) => progress.push(stage),
    },
    manifest,
    runtime,
    sessionIds,
    preprocess: preprocessor,
  })

  assert.equal(scene.count, 262_144)
  assert.deepEqual(loaded, ['dino', 'vae', 'dit', 'octree', 'gaussianDecoder'])
  assert.deepEqual(disposed, loaded)
  assert.equal(runs.filter(({ graph: name }) => name === 'dit').length, 8)
  assert.equal(runs.filter(({ graph: name }) => name === 'octree').length, 8)
  assert.deepEqual(runs.find(({ graph: name }) => name === 'dino').types, { pixel_values: 'float32' })
  assert.deepEqual(runs.find(({ graph: name }) => name === 'vae').types, {
    image_rgb: 'float16',
    epsilon: 'float16',
  })
  assert.deepEqual(runs.find(({ graph: name }) => name === 'dit').types, {
    latent: 'float32',
    camera: 'float32',
    t: 'float32',
    feature1: 'float32',
    feature2: 'float32',
  })
  assert.equal(progress.at(-1), 'complete')
  assert.equal(preparedDisposed, true)
  assert.equal(scene.metadata.modelRevision, 'a78fa12d')
  assert.equal(scene.metadata.generationSettings.steps, 4)
  scene.dispose()
})
