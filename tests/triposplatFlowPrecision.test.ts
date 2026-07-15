import assert from 'node:assert/strict'
import test from 'node:test'

import {
  blendClassifierFreeGuidance,
  sampleFlowEulerCfg,
} from '../src/models/triposplat/flowSampler.ts'
import {
  float16BitsToNumber,
  numberToFloat16Bits,
} from '../src/runtime/float16.ts'
import type { GenerationOptions } from '../src/models/ImageToGaussianModel.ts'
import type { OrtWorkerRequest } from '../src/runtime/OrtWorkerClient.ts'

const TRIPOSPLAT_FEATURE1_SHAPE = [1, 4101, 1280] as const
const TRIPOSPLAT_FEATURE2_SHAPE = [1, 4101, 128] as const
const TRIPOSPLAT_LATENT_SHAPE = [1, 8192, 16] as const
const TRIPOSPLAT_CAMERA_SHAPE = [1, 1, 5] as const

function elementCount(shape: readonly number[]): number {
  return shape.reduce((product, dimension) => product * dimension, 1)
}

function roundFloat16(value: number): number {
  return float16BitsToNumber(numberToFloat16Bits(value))
}

test('CFG matches PyTorch per-operation rounding for fp16 and fp32', () => {
  const conditionalValue = Math.fround(0.42336001992225647)
  const unconditionalValue = Math.fround(0.7559554576873779)
  const conditional = { latent: Float32Array.of(conditionalValue) }
  const unconditional = { latent: Float32Array.of(unconditionalValue) }

  const result = blendClassifierFreeGuidance(
    conditional,
    unconditional,
    3,
    'float16',
  )
  const expected = roundFloat16(
    roundFloat16(3 * conditionalValue) -
      roundFloat16(2 * unconditionalValue),
  )
  const singleRound = roundFloat16(3 * conditionalValue - 2 * unconditionalValue)
  const float32 = blendClassifierFreeGuidance(conditional, unconditional, 3, 'float32')

  assert.equal(expected, -0.2412109375)
  assert.equal(result.latent[0], expected)
  assert.equal(singleRound, -0.2418212890625)
  assert.notEqual(result.latent[0], singleRound)
  // PyTorch float32 evaluates both scalar products and the subtraction at the
  // tensor dtype; JavaScript must not fuse the expression in float64.
  assert.equal(float32.latent[0], -0.24183082580566406)
})

test('fp16 sampler rounds CFG and velocity scaling but accumulates the sample in float32', async () => {
  const conditionalValue = Math.fround(0.42336001992225647)
  const unconditionalValue = Math.fround(0.7559554576873779)
  const initial = Math.fround(1.234567)

  const result = await sampleFlowEulerCfg(
    ({ pass }) => ({
      latent: Float32Array.of(
        pass === 'conditional' ? conditionalValue : unconditionalValue,
      ),
    }),
    { latent: Float32Array.of(initial) },
    {
      condition: 'image',
      negativeCondition: 'zeros',
      steps: 2,
      shift: 3,
      guidanceScale: 3,
      predictionArithmetic: 'float16',
    },
  )

  const cfg = roundFloat16(
    roundFloat16(3 * conditionalValue) -
      roundFloat16(2 * unconditionalValue),
  )
  // Shift=3 with two steps yields t=[1, 0.75, 0], hence dt=[0.25, 0.75].
  const afterFirstStep = Math.fround(initial - roundFloat16(cfg * 0.25))
  const expected = Math.fround(afterFirstStep - roundFloat16(cfg * 0.75))

  assert.equal(cfg, -0.2412109375)
  assert.equal(roundFloat16(cfg * 0.25), -0.060302734375)
  assert.equal(roundFloat16(cfg * 0.75), -0.180908203125)
  assert.equal(afterFirstStep, 1.2948697805404663)
  assert.equal(expected, 1.4757779836654663)
  assert.equal(result.latent[0], expected)
})

type WorkerMessageHandler = ((event: { data: unknown }) => void) | null

class DetachingOrtWorker {
  onmessage: WorkerMessageHandler = null
  onerror: ((event: unknown) => void) | null = null
  onmessageerror: ((event: unknown) => void) | null = null
  readonly receivedDitTimesteps: number[] = []
  readonly detachedTransportLengths: number[] = []
  terminated = false

  postMessage(message: OrtWorkerRequest, transfer: Transferable[] = []): void {
    // Browser Worker.postMessage performs this transfer internally. Reproducing
    // it here detaches every sender-side tensor buffer before the reply arrives.
    const cloned = structuredClone(message, { transfer })

    if (cloned.type === 'run-session' && cloned.payload.sessionId === 'triposplat/dit') {
      this.receivedDitTimesteps.push(cloned.payload.inputs.t.data[0])
      if (message.type !== 'run-session') throw new Error('Structured-clone type drifted.')
      this.detachedTransportLengths.push(message.payload.inputs.t.data.length)
    }

    const reply = this.replyFor(cloned)
    queueMicrotask(() => this.onmessage?.({ data: reply }))
  }

  terminate(): void {
    this.terminated = true
  }

  private replyFor(message: OrtWorkerRequest): unknown {
    if (message.type === 'load-session') {
      if (message.payload.sessionId === 'triposplat/octree') {
        return {
          type: 'reply',
          operation: message.type,
          requestId: message.requestId,
          ok: false,
          error: {
            name: 'Error',
            message: 'intentional stop after flow sampling',
          },
        }
      }
      return {
        type: 'reply',
        operation: message.type,
        requestId: message.requestId,
        ok: true,
        result: {
          sessionId: message.payload.sessionId,
          executionProvider: 'webgpu',
          inputNames: [],
          outputNames: [],
          inputMetadata: [],
          outputMetadata: [],
          loadMs: 0,
        },
      }
    }
    if (message.type === 'run-session') {
      return {
        type: 'reply',
        operation: message.type,
        requestId: message.requestId,
        ok: true,
        result: {
          sessionId: message.payload.sessionId,
          outputs: {
            pred_latent: {
              type: 'float32',
              dims: Array.from(TRIPOSPLAT_LATENT_SHAPE),
              data: new Float32Array(elementCount(TRIPOSPLAT_LATENT_SHAPE)),
            },
            pred_camera: {
              type: 'float32',
              dims: Array.from(TRIPOSPLAT_CAMERA_SHAPE),
              data: new Float32Array(elementCount(TRIPOSPLAT_CAMERA_SHAPE)),
            },
          },
          timings: { inferenceMs: 0, readbackMs: 0, totalMs: 0 },
        },
      }
    }
    if (message.type === 'dispose-session') {
      return {
        type: 'reply',
        operation: message.type,
        requestId: message.requestId,
        ok: true,
        result: { sessionId: message.payload.sessionId, disposed: true },
      }
    }
    if (message.type === 'dispose-all') {
      return {
        type: 'reply',
        operation: message.type,
        requestId: message.requestId,
        ok: true,
        result: { disposedSessionIds: [] },
      }
    }
    throw new Error(`Unexpected fake-worker operation: ${String(message.type)}`)
  }
}

test('TripoSplat makes a fresh timestep transport tensor before each transferred CFG call', async () => {
  const { createServer } = await import('vite')
  const vite = await createServer({
    root: new URL('..', import.meta.url).pathname,
    configFile: false,
    appType: 'custom',
    logLevel: 'silent',
    server: { middlewareMode: true },
  })
  const worker = new DetachingOrtWorker()
  const graph = (name: string) => ({
    precision: 'float32' as const,
    manifest: {
      graphUrl: `https://models.example/${name}.onnx`,
      externalData: [],
    },
  })
  interface TestTripoSplatModel {
    encode: (...args: unknown[]) => Promise<unknown>
    generate(image: ImageBitmap, options?: GenerationOptions): Promise<unknown>
    dispose(): Promise<void>
  }
  let model: TestTripoSplatModel | undefined
  try {
    const module = await vite.ssrLoadModule(
      '/src/models/triposplat/TripoSplatWebGPUModel.ts',
    ) as { TripoSplatWebGPUModel: new (options: unknown) => TestTripoSplatModel }
    model = new module.TripoSplatWebGPUModel({
      graphs: {
        dinov3: graph('dino'),
        vaeEncoder: graph('vae'),
        dit: graph('dit'),
        octree: graph('octree'),
        gaussianDecoder: graph('gaussian'),
      },
      worker: {
        baseUrl: 'https://app.example/',
        workerFactory: () => worker,
      },
    })

    // Bypass image/canvas work so this Node test reaches the real generate-time
    // DiT adapter and worker transport without requiring browser globals.
    model.encode = async () => ({
      preparedImage: { width: 1, height: 1, data: new Uint8ClampedArray(3) },
      feature1: new Float32Array(elementCount(TRIPOSPLAT_FEATURE1_SHAPE)),
      feature2: new Float32Array(elementCount(TRIPOSPLAT_FEATURE2_SHAPE)),
      timings: {},
    })

    await assert.rejects(
      model.generate({} as ImageBitmap, {
        steps: 1,
        guidanceScale: 3,
        latentNoise: new Float32Array(elementCount(TRIPOSPLAT_LATENT_SHAPE)),
        cameraNoise: new Float32Array(elementCount(TRIPOSPLAT_CAMERA_SHAPE)),
      }),
      /intentional stop after flow sampling/,
    )

    assert.deepEqual(worker.receivedDitTimesteps, [1000, 1000])
    assert.deepEqual(
      worker.detachedTransportLengths,
      [0, 0],
      'each posted transport copy should detach in the sender realm',
    )
  } finally {
    if (model) await model.dispose()
    await vite.close()
  }
  assert.equal(worker.terminated, true)
})
