import assert from 'node:assert/strict'
import test from 'node:test'

import { createGaussianScene } from '@ai3d/gaussian-scene'
import { CancelledError, TripoSplatWebGPU } from '../dist/index.js'

const metadata = {
  coordinateSystem: 'triposplat-object',
  units: 'model-unit',
  rotationOrder: 'wxyz',
  scaleEncoding: 'linear',
  opacityEncoding: 'linear',
  colorSemantics: 'linear-rgb',
  sphericalHarmonicsSemantics: null,
  modelRevision: 'retry-test',
  generationSettings: {},
  seed: 42,
  runtimeVersion: 'test',
}

const graphNames = ['dino', 'vae', 'dit', 'octree', 'gaussianDecoder']

function manifest() {
  return {
    name: 'triposplat-webgpu',
    version: 'retry-test',
    modelRevision: 'retry-test',
    precision: 'fp32',
    graphs: Object.fromEntries(graphNames.map((name) => [
      name,
      { url: `${name}.onnx`, precision: 'fp32', inputPrecision: 'fp32' },
    ])),
  }
}

function makeScene(position = 0) {
  return createGaussianScene({
    count: 1,
    positions: Float32Array.of(position, 0, 0),
    scales: Float32Array.of(1, 1, 1),
    rotations: Float32Array.of(1, 0, 0, 0),
    opacities: Float32Array.of(1),
    colors: new Float32Array(3),
    metadata,
  })
}

test('cancelled generation rebuilds its worker and can retry on the same model instance', async () => {
  let workerCount = 0
  let terminatedWorkers = 0
  let manifestRequests = 0
  let pipelineCalls = 0
  const workerFactory = () => {
    workerCount += 1
    const generation = workerCount
    return {
      onmessage: null,
      onerror: null,
      onmessageerror: null,
      postMessage(request) {
        // The first cancellation deliberately occurs before worker
        // initialization replies; retry must not wait forever in dispose().
        if (generation === 1 && request.type === 'configure') return
        queueMicrotask(() => {
          const result = request.type === 'configure'
            ? { operation: 'configure', configured: true }
            : { operation: 'dispose', disposedSessionIds: [] }
          this.onmessage?.({ data: { type: 'reply', requestId: request.requestId, ok: true, result } })
        })
      },
      terminate() { terminatedWorkers += 1 },
    }
  }
  const fetch = async (url) => {
    assert.match(String(url), /manifest\.json$/)
    manifestRequests += 1
    return Response.json(manifest())
  }
  const scene = makeScene()
  const model = new TripoSplatWebGPU({
    modelBaseUrl: 'https://cdn.example.test/retry/',
    executionProviders: ['wasm'],
    cache: 'none',
    fetch,
    workerFactory,
    pipeline: async () => {
      pipelineCalls += 1
      if (pipelineCalls === 1) throw new CancelledError('cancelled by lifecycle test')
      return scene
    },
  })
  try {
    await assert.rejects(
      model.generate(new Blob()),
      (error) => error instanceof CancelledError && error.recoverable,
    )
    assert.equal(workerCount, 1)
    assert.equal(terminatedWorkers, 1)
    assert.equal(model.capabilities.manifestLoaded, false)

    assert.equal(await model.generate(new Blob()), scene)
    assert.equal(workerCount, 2)
    assert.equal(manifestRequests, 2)
  } finally {
    await model.dispose()
    scene.dispose()
  }
  assert.equal(terminatedWorkers, 2)
})

test('repeated calls serialize, while queued cancellation rejects promptly without bypassing the queue', async () => {
  let workerCount = 0
  let terminatedWorkers = 0
  let manifestRequests = 0
  let pipelineCalls = 0
  let activePipelines = 0
  let maximumActivePipelines = 0
  let enterFirst
  let releaseFirst
  const firstEntered = new Promise((resolve) => { enterFirst = resolve })
  const firstRelease = new Promise((resolve) => { releaseFirst = resolve })
  const workerFactory = () => {
    workerCount += 1
    return {
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
      terminate() { terminatedWorkers += 1 },
    }
  }
  const model = new TripoSplatWebGPU({
    modelBaseUrl: 'https://cdn.example.test/repeated/',
    executionProviders: ['wasm'],
    cache: 'none',
    fetch: async () => {
      manifestRequests += 1
      return Response.json(manifest())
    },
    workerFactory,
    pipeline: async () => {
      pipelineCalls += 1
      const invocation = pipelineCalls
      activePipelines += 1
      maximumActivePipelines = Math.max(maximumActivePipelines, activePipelines)
      try {
        if (invocation === 1) {
          enterFirst()
          await firstRelease
        }
        return makeScene(invocation)
      } finally {
        activePipelines -= 1
      }
    },
  })
  try {
    const first = model.generate(new Blob())
    await firstEntered
    const queuedController = new AbortController()
    const cancelled = model.generate(new Blob(), { signal: queuedController.signal })
    queuedController.abort('cancel queued lifecycle test')
    let cancellationTimeout
    try {
      await assert.rejects(
        Promise.race([
          cancelled,
          new Promise((_resolve, reject) => {
            cancellationTimeout = setTimeout(
              () => reject(new Error('Queued cancellation did not settle promptly.')),
              250,
            )
          }),
        ]),
        (error) => error instanceof CancelledError && error.cause === 'cancel queued lifecycle test',
      )
    } finally {
      clearTimeout(cancellationTimeout)
    }
    const second = model.generate(new Blob())
    await Promise.resolve()
    assert.equal(pipelineCalls, 1)
    releaseFirst()
    const scenes = await Promise.all([first, second])
    assert.deepEqual(scenes.map((scene) => scene.positions[0]), [1, 2])
    for (const scene of scenes) scene.dispose()
    assert.equal(maximumActivePipelines, 1)
    assert.equal(workerCount, 1)
    assert.equal(manifestRequests, 1)
  } finally {
    releaseFirst()
    await model.dispose()
  }
  assert.equal(terminatedWorkers, 1)
})

test('dispose waits for active generation unwinding and disposes a late scene', async () => {
  let enterPipeline
  let releasePipeline
  let terminatedWorkers = 0
  const pipelineEntered = new Promise((resolve) => { enterPipeline = resolve })
  const pipelineRelease = new Promise((resolve) => { releasePipeline = resolve })
  const lateScene = makeScene()
  const model = new TripoSplatWebGPU({
    modelBaseUrl: 'https://cdn.example.test/dispose/',
    executionProviders: ['wasm'],
    cache: 'none',
    fetch: async () => Response.json(manifest()),
    workerFactory: () => ({
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
      terminate() { terminatedWorkers += 1 },
    }),
    pipeline: async () => {
      enterPipeline()
      await pipelineRelease
      return lateScene
    },
  })
  const generation = model.generate(new Blob())
  await pipelineEntered
  let disposeSettled = false
  const disposal = model.dispose().then(() => { disposeSettled = true })
  await Promise.resolve()
  await Promise.resolve()
  assert.equal(disposeSettled, false)
  releasePipeline()
  await assert.rejects(generation, (error) => error?.code === 'DISPOSED')
  await disposal
  assert.equal(lateScene.isDisposed, true)
  assert.equal(terminatedWorkers, 1)
  await model.dispose()
  assert.equal(terminatedWorkers, 1)
})

test('aborting an in-flight graph load rebuilds the worker and permits retry', async () => {
  let workerCount = 0
  let terminatedWorkers = 0
  let manifestRequests = 0
  let artifactRequests = 0
  let announceLoadStarted
  const loadStarted = new Promise((resolve) => { announceLoadStarted = resolve })
  const workerFactory = () => {
    workerCount += 1
    const generation = workerCount
    return {
      onmessage: null,
      onerror: null,
      onmessageerror: null,
      postMessage(request) {
        if (request.type === 'load' && generation === 1) {
          announceLoadStarted()
          return
        }
        queueMicrotask(() => {
          let result
          if (request.type === 'configure') {
            result = { operation: 'configure', configured: true }
          } else if (request.type === 'load') {
            result = {
              operation: 'load',
              graph: {
                sessionId: request.sessionId,
                executionProvider: 'wasm',
                inputNames: ['pixel_values'],
                outputNames: ['feature1'],
                loadMs: 1,
              },
            }
          } else {
            result = { operation: 'dispose', disposedSessionIds: [] }
          }
          this.onmessage?.({ data: { type: 'reply', requestId: request.requestId, ok: true, result } })
        })
      },
      terminate() { terminatedWorkers += 1 },
    }
  }
  const model = new TripoSplatWebGPU({
    modelBaseUrl: 'https://cdn.example.test/load-abort/',
    executionProviders: ['wasm'],
    cache: 'none',
    fetch: async (url) => {
      if (String(url).endsWith('manifest.json')) {
        manifestRequests += 1
        return Response.json(manifest())
      }
      artifactRequests += 1
      return new Response(Uint8Array.of(1, 2, 3), {
        headers: { 'content-length': '3' },
      })
    },
    workerFactory,
    pipeline: async (context) => {
      await context.runtime.loadGraph(
        context.sessionIds.dino,
        context.manifest.graphs.dino,
        { signal: context.options.signal },
      )
      return makeScene(workerCount)
    },
  })
  const controller = new AbortController()
  try {
    const cancelled = model.generate(new Blob(), { signal: controller.signal })
    await loadStarted
    controller.abort('lifecycle graph-load cancellation')
    await assert.rejects(
      cancelled,
      (error) => error instanceof CancelledError && error.cause === 'lifecycle graph-load cancellation',
    )
    assert.equal(workerCount, 1)
    assert.equal(terminatedWorkers, 1)
    assert.equal(model.capabilities.manifestLoaded, false)

    const scene = await model.generate(new Blob())
    assert.equal(scene.positions[0], 2)
    scene.dispose()
    assert.equal(workerCount, 2)
    assert.equal(manifestRequests, 2)
    assert.equal(artifactRequests, 2)
  } finally {
    await model.dispose()
  }
  assert.equal(terminatedWorkers, 2)
})
