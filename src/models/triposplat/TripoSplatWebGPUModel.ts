import { buildGaussianPly } from '../../lib/gaussianPly'
import { float16ToFloat32, float32ToFloat16 } from '../../runtime/float16'
import type {
  OrtLoadSessionResult,
  OrtRunTimings,
  OrtWorkerClientOptions,
  OrtWorkerStatus,
} from '../../runtime/OrtWorkerClient'
import { OrtWorkerClient } from '../../runtime/OrtWorkerClient'
import { createTensorPayload, type TensorPayload } from '../../runtime/tensors'
import type {
  GaussianScene,
  GenerationOptions,
  ImageToGaussianModel,
} from '../ImageToGaussianModel'
import { throwIfAborted } from '../ImageToGaussianModel'
import {
  elementCount,
  TRIPOSPLAT_FEATURE1_SHAPE,
  TRIPOSPLAT_FEATURE2_SHAPE,
  TRIPOSPLAT_IMAGE_SHAPE,
  TRIPOSPLAT_LATENT_SHAPE,
  TRIPOSPLAT_CAMERA_SHAPE,
  TRIPOSPLAT_MAX_DECODER_POINTS,
  TRIPOSPLAT_MAX_GAUSSIANS,
  TRIPOSPLAT_MIN_GAUSSIANS,
  TRIPOSPLAT_VAE_NOISE_SHAPE,
} from './contracts'
import {
  decodeTripoSplatGaussianFeatures,
  TRIPOSPLAT_GAUSSIANS_PER_POINT,
  TRIPOSPLAT_GS_FEATURE_WIDTH,
} from './gaussianDecoder'
import {
  sampleFlowEulerCfg,
  type FlowModelInvocation,
  type FlowTensorState,
} from './flowSampler'
import type {
  TripoSplatGraphDescriptor,
  TripoSplatGraphName,
  TripoSplatModelGraphs,
} from './manifests'
import { createTripoSplatModelManifest } from './manifests'
import {
  buildTripoSplatEncoderTensors,
  preprocessTripoSplatImage,
  type RgbImage,
  type TripoSplatBackgroundRemover,
} from './preprocess'
import { fillNormal, Mulberry32 } from './random'
import {
  sampleOctree,
  type OctreeOccupancyInvocation,
  type OctreeSampleResult,
} from './octree'

const SESSION_IDS: Record<TripoSplatGraphName, string> = {
  dinov3: 'triposplat/dinov3',
  vaeEncoder: 'triposplat/vae-encoder',
  dit: 'triposplat/dit',
  octree: 'triposplat/octree',
  gaussianDecoder: 'triposplat/gaussian-decoder',
}

export interface TripoSplatWebGPUModelOptions {
  graphs?: TripoSplatModelGraphs
  removeBackground?: TripoSplatBackgroundRemover
  allowWasmFallback?: boolean
  onRuntimeStatus?: (status: OrtWorkerStatus) => void
  worker?: Omit<OrtWorkerClientOptions, 'onStatus'>
}

export interface TripoSplatEncoderResult {
  preparedImage: RgbImage
  feature1?: Float32Array
  feature2?: Float32Array
  timings: Partial<Record<'dinov3' | 'vaeEncoder', OrtRunTimings>>
}

interface TripoSplatCondition {
  feature1: Float32Array
  feature2: Float32Array
}

function payloadToFloat32(label: string, payload: TensorPayload): Float32Array {
  if (payload.type === 'float32') return payload.data
  if (payload.type === 'float16') return float16ToFloat32(payload.data)
  throw new Error(`${label} must be float32 or float16, got ${payload.type}.`)
}

function inputPayload(
  descriptor: TripoSplatGraphDescriptor,
  data: Float32Array,
  dims: readonly number[],
): TensorPayload {
  return descriptor.precision === 'float16'
    ? createTensorPayload('float16', float32ToFloat16(data), dims)
    : createTensorPayload('float32', data, dims)
}

/** Browser TripoSplat adapter. Encoder-only manifests are supported for parity bring-up. */
export class TripoSplatWebGPUModel implements ImageToGaussianModel {
  readonly graphs: TripoSplatModelGraphs

  private readonly client: OrtWorkerClient
  private readonly removeBackground?: TripoSplatBackgroundRemover
  private readonly allowWasmFallback: boolean
  private readonly sessions = new Map<TripoSplatGraphName, OrtLoadSessionResult>()
  private disposed = false

  constructor(options: TripoSplatWebGPUModelOptions = {}) {
    this.graphs = options.graphs ?? createTripoSplatModelManifest()
    this.removeBackground = options.removeBackground
    this.allowWasmFallback = options.allowWasmFallback ?? false
    this.client = new OrtWorkerClient({ ...options.worker, onStatus: options.onRuntimeStatus })
  }

  async load(): Promise<void> {
    this.assertUsable()
    // Full-pipeline graphs are deliberately staged to fit the 16 GB target.
    // `load()` warms only the encoder slice; generate() releases it before DiT.
    for (const name of ['dinov3', 'vaeEncoder'] as const) {
      if (this.graphs[name]) await this.loadGraph(name)
    }
  }

  async encode(
    image: ImageBitmap,
    options: GenerationOptions = {},
  ): Promise<TripoSplatEncoderResult> {
    this.assertUsable()
    throwIfAborted(options.signal)
    const hasDino = this.graphs.dinov3 !== undefined
    const hasVae = this.graphs.vaeEncoder !== undefined
    if (!hasDino && !hasVae) throw new Error('TripoSplat encoder manifest contains neither DINOv3 nor Flux VAE.')

    options.onProgress?.({ stage: 'preprocessing', message: 'Preparing the TripoSplat 1024px RGB composite…' })
    const prepared = await preprocessTripoSplatImage(image, {
      erodeRadius: options.erodeRadius,
      removeBackground: this.removeBackground,
      opaqueImageIsAlreadyPrepared: options.inputIsPrepared,
    })
    const tensors = buildTripoSplatEncoderTensors(prepared.image)
    throwIfAborted(options.signal)

    const result: TripoSplatEncoderResult = { preparedImage: prepared.image, timings: {} }
    if (hasDino) {
      options.onProgress?.({ stage: 'encoding-dinov3', message: 'Running DINOv3 on WebGPU…' })
      const descriptor = this.requireGraph('dinov3')
      await this.loadGraph('dinov3')
      const response = await this.client.runSession({
        sessionId: SESSION_IDS.dinov3,
        inputs: {
          pixel_values: inputPayload(descriptor, tensors.dinov3.data, TRIPOSPLAT_IMAGE_SHAPE),
        },
        outputs: ['feature1'],
      })
      const feature1 = response.outputs.feature1
      if (!feature1) throw new Error('DINOv3 graph did not return feature1.')
      result.feature1 = payloadToFloat32('feature1', feature1)
      if (result.feature1.length !== elementCount(TRIPOSPLAT_FEATURE1_SHAPE)) {
        throw new Error(`DINOv3 returned ${result.feature1.length} values; expected ${elementCount(TRIPOSPLAT_FEATURE1_SHAPE)}.`)
      }
      result.timings.dinov3 = response.timings
    }

    if (hasVae) {
      options.onProgress?.({ stage: 'encoding-vae', message: 'Running the Flux VAE encoder on WebGPU…' })
      const descriptor = this.requireGraph('vaeEncoder')
      await this.loadGraph('vaeEncoder')
      const epsilon = options.vaeNoise
        ? new Float32Array(options.vaeNoise)
        : fillNormal(
            new Float32Array(elementCount(TRIPOSPLAT_VAE_NOISE_SHAPE)),
            new Mulberry32(options.seed ?? 42),
          )
      if (epsilon.length !== elementCount(TRIPOSPLAT_VAE_NOISE_SHAPE)) {
        throw new Error(`VAE epsilon contains ${epsilon.length} values; expected ${elementCount(TRIPOSPLAT_VAE_NOISE_SHAPE)}.`)
      }
      const response = await this.client.runSession({
        sessionId: SESSION_IDS.vaeEncoder,
        inputs: {
          image_rgb: inputPayload(descriptor, tensors.rgb.data, TRIPOSPLAT_IMAGE_SHAPE),
          epsilon: inputPayload(descriptor, epsilon, TRIPOSPLAT_VAE_NOISE_SHAPE),
        },
        outputs: ['feature2'],
      })
      const feature2Payload = response.outputs.feature2
      if (!feature2Payload) throw new Error('Flux VAE graph did not return feature2.')
      result.feature2 = payloadToFloat32('feature2', feature2Payload)
      if (result.feature2.length !== elementCount(TRIPOSPLAT_FEATURE2_SHAPE)) {
        throw new Error(`Flux VAE returned ${result.feature2.length} values; expected ${elementCount(TRIPOSPLAT_FEATURE2_SHAPE)}.`)
      }
      result.timings.vaeEncoder = response.timings
    }
    throwIfAborted(options.signal)
    return result
  }

  async generate(image: ImageBitmap, options: GenerationOptions = {}): Promise<GaussianScene> {
    for (const graph of ['dinov3', 'vaeEncoder', 'dit', 'octree', 'gaussianDecoder'] as const) {
      this.requireGraph(graph)
    }
    const random = new Mulberry32(options.seed ?? 42)
    const vaeNoise = options.vaeNoise
      ? new Float32Array(options.vaeNoise)
      : fillNormal(new Float32Array(elementCount(TRIPOSPLAT_VAE_NOISE_SHAPE)), random)
    let encoded: TripoSplatEncoderResult
    try {
      encoded = await this.encode(image, { ...options, vaeNoise })
    } finally {
      await Promise.all([this.disposeGraph('dinov3'), this.disposeGraph('vaeEncoder')])
    }
    if (!encoded.feature1 || !encoded.feature2) {
      throw new Error('TripoSplat encoding did not produce both conditioning tensors.')
    }
    throwIfAborted(options.signal)

    const condition: TripoSplatCondition = {
      feature1: encoded.feature1,
      feature2: encoded.feature2,
    }
    const negativeCondition: TripoSplatCondition = {
      feature1: new Float32Array(encoded.feature1.length),
      feature2: new Float32Array(encoded.feature2.length),
    }
    const latent = options.latentNoise
      ? new Float32Array(options.latentNoise)
      : fillNormal(new Float32Array(elementCount(TRIPOSPLAT_LATENT_SHAPE)), random)
    const camera = options.cameraNoise
      ? new Float32Array(options.cameraNoise)
      : fillNormal(new Float32Array(elementCount(TRIPOSPLAT_CAMERA_SHAPE)), random)
    this.assertLength('latent noise', latent, elementCount(TRIPOSPLAT_LATENT_SHAPE))
    this.assertLength('camera noise', camera, elementCount(TRIPOSPLAT_CAMERA_SHAPE))

    const ditDescriptor = this.requireGraph('dit')
    await this.loadGraph('dit')
    let ditInferenceMs = 0
    let ditReadbackMs = 0
    const steps = options.steps ?? 20
    let flowState: FlowTensorState
    try {
      flowState = await sampleFlowEulerCfg(
        async (invocation: FlowModelInvocation<TripoSplatCondition>) => {
        const response = await this.client.runSession({
          sessionId: SESSION_IDS.dit,
          inputs: {
            latent: inputPayload(ditDescriptor, invocation.sample.latent, TRIPOSPLAT_LATENT_SHAPE),
            camera: inputPayload(ditDescriptor, invocation.sample.camera, TRIPOSPLAT_CAMERA_SHAPE),
            t: inputPayload(
              ditDescriptor,
              new Float32Array(invocation.timestepTensor),
              [1],
            ),
            feature1: inputPayload(
              ditDescriptor,
              new Float32Array(invocation.condition.feature1),
              TRIPOSPLAT_FEATURE1_SHAPE,
            ),
            feature2: inputPayload(
              ditDescriptor,
              new Float32Array(invocation.condition.feature2),
              TRIPOSPLAT_FEATURE2_SHAPE,
            ),
          },
          outputs: ['pred_latent', 'pred_camera'],
          tag: `flow-${invocation.pass}-${invocation.step}-of-${invocation.totalSteps}`,
        })
        ditInferenceMs += response.timings.inferenceMs
        ditReadbackMs += response.timings.readbackMs
        const predictedLatent = response.outputs.pred_latent
        const predictedCamera = response.outputs.pred_camera
        if (!predictedLatent || !predictedCamera) {
          throw new Error('DiT graph must return pred_latent and pred_camera.')
        }
        return {
          latent: payloadToFloat32('pred_latent', predictedLatent),
          camera: payloadToFloat32('pred_camera', predictedCamera),
        }
        },
        { latent, camera },
        {
          condition,
          negativeCondition,
          steps,
          guidanceScale: options.guidanceScale ?? 3,
          shift: options.shift ?? 3,
          predictionArithmetic: ditDescriptor.internalPrecision === 'float32'
            ? 'float32'
            : 'float16',
          signal: options.signal,
          onStep: ({ step, totalSteps }) => {
            options.onProgress?.({
              stage: 'sampling',
              message: `TripoSplat flow step ${step}/${totalSteps}…`,
              progress: step / totalSteps,
              step,
              totalSteps,
            })
          },
        },
      )
    } finally {
      await this.disposeGraph('dit')
    }

    const numGaussians = this.normalizeGaussianCount(options.numGaussians ?? TRIPOSPLAT_MAX_GAUSSIANS)
    const numPoints = numGaussians / TRIPOSPLAT_GAUSSIANS_PER_POINT
    const octreeDescriptor = this.requireGraph('octree')
    await this.loadGraph('octree')
    let octreeInferenceMs = 0
    options.onProgress?.({ stage: 'decoding-octree', message: 'Sampling the dynamic occupancy octree…' })
    let points: OctreeSampleResult
    try {
      points = await sampleOctree(
        async (invocation: OctreeOccupancyInvocation<Float32Array>) => {
        const paddedCenters = new Float32Array(TRIPOSPLAT_MAX_DECODER_POINTS * 3)
        paddedCenters.set(invocation.parentCenters)
        const response = await this.client.runSession({
          sessionId: SESSION_IDS.octree,
          inputs: {
            x: inputPayload(octreeDescriptor, paddedCenters, [1, TRIPOSPLAT_MAX_DECODER_POINTS, 3]),
            l: inputPayload(octreeDescriptor, Float32Array.of(invocation.resolution), [1]),
            cond: inputPayload(
              octreeDescriptor,
              new Float32Array(invocation.condition),
              TRIPOSPLAT_LATENT_SHAPE,
            ),
          },
          outputs: ['logits'],
          tag: `octree-level-${invocation.level}`,
        })
        octreeInferenceMs += response.timings.inferenceMs
        const logitsPayload = response.outputs.logits
        if (!logitsPayload) throw new Error('Octree graph did not return logits.')
        const paddedLogits = payloadToFloat32('logits', logitsPayload)
        const required = invocation.parentCount * 8
        if (paddedLogits.length < required) {
          throw new Error(`Octree returned ${paddedLogits.length} logits; active frontier needs ${required}.`)
        }
        return { logits: paddedLogits.slice(0, required) }
        },
        {
          condition: flowState.latent,
          numPoints,
          rng: () => random.next(),
          signal: options.signal,
          onLevel: ({ level, totalLevels, occupiedVoxels }) => {
            options.onProgress?.({
              stage: 'decoding-octree',
              message: `Octree level ${level}/${totalLevels}: ${occupiedVoxels.toLocaleString()} occupied voxels…`,
              progress: level / totalLevels,
            })
          },
        },
      )
    } finally {
      await this.disposeGraph('octree')
    }

    const gaussianDescriptor = this.requireGraph('gaussianDecoder')
    await this.loadGraph('gaussianDecoder')
    options.onProgress?.({ stage: 'decoding-gaussians', message: 'Decoding Gaussian attributes…' })
    let features: Float32Array
    let gaussianInferenceMs = 0
    try {
      const gaussianResponse = await this.client.runSession({
        sessionId: SESSION_IDS.gaussianDecoder,
        inputs: {
          points: inputPayload(gaussianDescriptor, new Float32Array(points.points), [1, numPoints, 3]),
          cond: inputPayload(
            gaussianDescriptor,
            new Float32Array(flowState.latent),
            TRIPOSPLAT_LATENT_SHAPE,
          ),
        },
        outputs: ['features'],
      })
      gaussianInferenceMs = gaussianResponse.timings.inferenceMs
      const featurePayload = gaussianResponse.outputs.features
      if (!featurePayload) throw new Error('Gaussian decoder graph did not return features.')
      features = payloadToFloat32('features', featurePayload)
    } finally {
      await this.disposeGraph('gaussianDecoder')
    }
    this.assertLength('Gaussian decoder features', features, numPoints * TRIPOSPLAT_GS_FEATURE_WIDTH)
    const gaussians = decodeTripoSplatGaussianFeatures(points.points, features)
    options.onProgress?.({ stage: 'building-ply', message: 'Building browser Gaussian PLY…' })
    const ply = buildGaussianPly(gaussians)

    return {
      model: 'triposplat',
      count: numGaussians,
      totalCount: numGaussians,
      ply,
      gaussians,
      coordinateSystem: 'triposplat-object',
      colorSpace: 'sh0',
      metadata: {
        steps,
        guidanceScale: options.guidanceScale ?? 3,
        shift: options.shift ?? 3,
        seed: options.seed ?? 42,
        ditInferenceMs,
        ditReadbackMs,
        octreeInferenceMs,
        gaussianInferenceMs,
      },
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return
    this.disposed = true
    this.sessions.clear()
    await this.client.dispose()
  }

  private async loadGraph(name: TripoSplatGraphName): Promise<OrtLoadSessionResult> {
    const existing = this.sessions.get(name)
    if (existing) return existing
    const descriptor = this.requireGraph(name)
    const loaded = await this.client.loadSession({
      sessionId: SESSION_IDS[name],
      manifest: descriptor.manifest,
      options: {
        allowWasmFallback: this.allowWasmFallback,
        // Match the validated browser labs for every published graph. DiT also
        // relies on exported Add(0) dense-layout barriers that ORT must retain.
        graphOptimizationLevel: 'disabled',
      },
    })
    this.sessions.set(name, loaded)
    return loaded
  }

  private async disposeGraph(name: TripoSplatGraphName): Promise<void> {
    if (!this.sessions.delete(name)) return
    await this.client.disposeSession(SESSION_IDS[name])
  }

  private requireGraph(name: TripoSplatGraphName): TripoSplatGraphDescriptor {
    const graph = this.graphs[name]
    if (!graph) throw new Error(`TripoSplat graph '${name}' is not configured.`)
    return graph
  }

  private assertLength(label: string, value: Float32Array, expected: number): void {
    if (value.length !== expected) throw new Error(`${label} contains ${value.length} values; expected ${expected}.`)
  }

  private normalizeGaussianCount(requested: number): number {
    if (!Number.isFinite(requested) || requested < TRIPOSPLAT_MIN_GAUSSIANS || requested > TRIPOSPLAT_MAX_GAUSSIANS) {
      throw new Error(
        `numGaussians must be in [${TRIPOSPLAT_MIN_GAUSSIANS}, ${TRIPOSPLAT_MAX_GAUSSIANS}].`,
      )
    }
    const rounded = Math.round(requested / TRIPOSPLAT_GAUSSIANS_PER_POINT) * TRIPOSPLAT_GAUSSIANS_PER_POINT
    if (rounded !== TRIPOSPLAT_MAX_GAUSSIANS) {
      throw new Error(
        `The checked-in Gaussian decoder contract is fixed at ${TRIPOSPLAT_MAX_GAUSSIANS} ` +
          `Gaussians (8192 decoder points); requested ${rounded}. Export a separate fixed-shape ` +
          `decoder to support another count without changing full self-attention semantics.`,
      )
    }
    return rounded
  }

  private assertUsable(): void {
    if (this.disposed) throw new Error('TripoSplatWebGPUModel has been disposed.')
  }
}
