import type { GaussianScene, GaussianSceneMetadata } from '@ai3d/gaussian-scene'

import {
  assertLength,
  elementCount,
  TRIPOSPLAT_CAMERA_SHAPE,
  TRIPOSPLAT_FEATURE1_SHAPE,
  TRIPOSPLAT_FEATURE2_SHAPE,
  TRIPOSPLAT_IMAGE_SHAPE,
  TRIPOSPLAT_LATENT_SHAPE,
  TRIPOSPLAT_MAX_DECODER_POINTS,
  TRIPOSPLAT_MAX_GAUSSIANS,
  TRIPOSPLAT_VAE_NOISE_SHAPE,
} from './contracts.js'
import { decodeGaussians, GAUSSIAN_FEATURE_WIDTH } from './decode.js'
import { GraphCapabilityError, throwIfAborted } from './errors.js'
import type { ResolvedGraphManifestEntry } from './manifest.js'
import { sampleOctree } from './octree.js'
import { fillNormal, Mulberry32 } from './random.js'
import {
  prepareReusableGraphInputs,
  type GraphInfo,
  type PreparedReusableGraphInputs,
  type RunGraphOptions,
} from './runtime.js'
import { createSampler, type FlowInvocation, type FlowState } from './sampler.js'
import {
  createTensor,
  float16ToFloat32,
  float32ToFloat16,
  type TensorMap,
  type TensorPayload,
} from './tensors.js'
import type { GenerateOptions, Precision, TripoSplatGraphName } from './types.js'
import type { TripoSplatPipelineContext } from './triposplat.js'

const RUNTIME_VERSION = '@ai3d/triposplat-webgpu/0.1.0-alpha.0; onnxruntime-web/1.27.0'
const TRIPOSPLAT_EXPORT_TRANSFORM = [
  1, 0, 0,
  0, 0, -1,
  0, 1, 0,
] as const

interface TripoSplatCondition {
  inputs: PreparedReusableGraphInputs
}

interface StageTimings {
  dinoInferenceMs: number
  vaeInferenceMs: number
  ditInferenceMs: number
  ditReadbackMs: number
  octreeInferenceMs: number
  gaussianInferenceMs: number
  /** Wall time through Gaussian neural inference; excludes host activation and scene ownership copies. */
  throughGaussianInferenceMs: number
}

const GRAPH_CONTRACTS: Readonly<Record<
  TripoSplatGraphName,
  { inputs: readonly string[]; outputs: readonly string[] }
>> = {
  dino: { inputs: ['pixel_values'], outputs: ['feature1'] },
  vae: { inputs: ['image_rgb', 'epsilon'], outputs: ['feature2'] },
  dit: {
    inputs: ['latent', 'camera', 't', 'feature1', 'feature2'],
    outputs: ['pred_latent', 'pred_camera'],
  },
  octree: { inputs: ['x', 'l', 'cond'], outputs: ['logits'] },
  gaussianDecoder: { inputs: ['points', 'cond'], outputs: ['features'] },
}

function requireGraph(
  context: TripoSplatPipelineContext,
  name: TripoSplatGraphName,
): ResolvedGraphManifestEntry {
  const graph = context.manifest.graphs[name]
  if (!graph) {
    throw new GraphCapabilityError(`TripoSplat graph '${name}' is not configured.`, {
      diagnostics: { graph: name },
    })
  }
  return graph
}

function assertGraphContract(name: TripoSplatGraphName, info: GraphInfo): void {
  const contract = GRAPH_CONTRACTS[name]
  const missingInputs = contract.inputs.filter((input) => !info.inputNames.includes(input))
  const missingOutputs = contract.outputs.filter((output) => !info.outputNames.includes(output))
  if (missingInputs.length > 0 || missingOutputs.length > 0) {
    throw new GraphCapabilityError(`Graph '${name}' does not implement the published browser contract.`, {
      diagnostics: {
        graph: name,
        expectedInputs: [...contract.inputs],
        expectedOutputs: [...contract.outputs],
        actualInputs: info.inputNames,
        actualOutputs: info.outputNames,
        missingInputs,
        missingOutputs,
      },
    })
  }
}

async function loadStage(
  context: TripoSplatPipelineContext,
  name: TripoSplatGraphName,
): Promise<ResolvedGraphManifestEntry> {
  const graph = requireGraph(context, name)
  const info = await context.runtime.loadGraph(context.sessionIds[name], graph, {
    // All published Chrome parity gates use the export graph exactly as written.
    // DiT in particular contains Add(0) layout barriers that ORT must preserve;
    // keep one conservative session policy until optimized variants are gated
    // independently for every stage.
    graphOptimizationLevel: 'disabled',
    ...(context.options.signal === undefined ? {} : { signal: context.options.signal }),
  })
  assertGraphContract(name, info)
  return graph
}

function publicInputPrecision(graph: ResolvedGraphManifestEntry): Precision {
  // Internal graph precision and public I/O precision are deliberately distinct.
  // Every published export defaults to float32 I/O; only the Flux VAE exporter
  // has an explicit optional fp16 input contract.
  return graph.inputPrecision ?? 'fp32'
}

function inputTensor(
  graph: ResolvedGraphManifestEntry,
  values: Float32Array,
  dims: readonly number[],
): TensorPayload {
  return publicInputPrecision(graph) === 'fp16'
    ? createTensor('float16', float32ToFloat16(values), dims)
    : createTensor('float32', values, dims)
}

function outputFloat32(outputs: TensorMap, name: string): Float32Array {
  const output = outputs[name]
  if (!output) throw new GraphCapabilityError(`Graph did not return required output '${name}'.`)
  if (output.type === 'float32') return output.data
  if (output.type === 'float16') return float16ToFloat32(output.data)
  throw new GraphCapabilityError(`Output '${name}' must be float32 or float16, got ${output.type}.`)
}

function explicitOrRandomNoise(
  provided: Float32Array | undefined,
  expected: number,
  label: string,
  random: Mulberry32,
): Float32Array {
  const values = provided === undefined
    ? fillNormal(new Float32Array(expected), random)
    : new Float32Array(provided)
  assertLength(label, values, expected)
  return values
}

function runOptions(
  outputs: readonly string[],
  signal: AbortSignal | undefined,
  tag?: string,
): RunGraphOptions {
  return {
    outputs,
    ...(tag === undefined ? {} : { tag }),
    ...(signal === undefined ? {} : { signal }),
  }
}

function generationConfiguration(options: GenerateOptions): {
  steps: 4 | 20
  guidanceScale: number
  shift: number
  gaussianCount: number
  seed: number
} {
  const steps = options.steps ?? 20
  if (steps !== 4 && steps !== 20) {
    throw new RangeError(`steps must be the official fast (4) or quality (20) schedule, got ${steps}.`)
  }
  const guidanceScale = options.guidanceScale ?? 3
  if (!Number.isFinite(guidanceScale) || guidanceScale < 1) {
    throw new RangeError('guidanceScale must be finite and at least 1.')
  }
  const shift = options.shift ?? 3
  if (!Number.isFinite(shift) || shift <= 0) throw new RangeError('shift must be positive and finite.')
  const gaussianCount = options.gaussianCount ?? TRIPOSPLAT_MAX_GAUSSIANS
  if (gaussianCount !== TRIPOSPLAT_MAX_GAUSSIANS) {
    throw new GraphCapabilityError(
      `The published decoder graph emits exactly ${TRIPOSPLAT_MAX_GAUSSIANS} Gaussians; ` +
        `a different count requires a separately exported decoder contract.`,
      { diagnostics: { gaussianCount } },
    )
  }
  const seed = options.seed ?? 42
  if (!Number.isFinite(seed)) throw new RangeError('seed must be finite.')
  return { steps, guidanceScale, shift, gaussianCount, seed }
}

/**
 * Default browser-local TripoSplat executor.
 *
 * Graphs are loaded and released by stage so encoder, DiT, and decoder weights
 * are never intentionally resident together. All model invocations stay in the
 * package worker; only preprocessing, sampler state, and dynamic octree logic
 * run on the browser host.
 */
export async function runBuiltInTripoSplatPipeline(
  context: TripoSplatPipelineContext,
): Promise<GaussianScene> {
  const { options } = context
  const configuration = generationConfiguration(options)
  throwIfAborted(options.signal)
  const startedAt = performance.now()
  const random = new Mulberry32(configuration.seed)
  const timings: StageTimings = {
    dinoInferenceMs: 0,
    vaeInferenceMs: 0,
    ditInferenceMs: 0,
    ditReadbackMs: 0,
    octreeInferenceMs: 0,
    gaussianInferenceMs: 0,
    throughGaussianInferenceMs: 0,
  }

  options.onProgress?.({ stage: 'preprocessing', message: 'Preparing the 1024px RGB composite.' })
  const prepared = await context.preprocess(context.input, {
    ...(options.signal === undefined ? {} : { signal: options.signal }),
    inputIsPrepared: options.inputIsPrepared ?? false,
    ...(context.removeBackground === undefined ? {} : { removeBackground: context.removeBackground }),
    ...(options.erodeRadius === undefined ? {} : { erodeRadius: options.erodeRadius }),
  })
  try {
    const vaeNoise = explicitOrRandomNoise(
      options.vaeNoise,
      elementCount(TRIPOSPLAT_VAE_NOISE_SHAPE),
      'VAE noise',
      random,
    )

    let feature1: Float32Array
    options.onProgress?.({ stage: 'dino', message: 'Running DINOv3 on WebGPU.' })
    const dinoGraph = await loadStage(context, 'dino')
    try {
      const response = await context.runtime.runGraph(
        context.sessionIds.dino,
        {
          pixel_values: inputTensor(
            dinoGraph,
            prepared.tensors.dinov3.data,
            TRIPOSPLAT_IMAGE_SHAPE,
          ),
        },
        runOptions(['feature1'], options.signal, 'dino-encoder'),
      )
      timings.dinoInferenceMs = response.timings.inferenceMs
      feature1 = outputFloat32(response.outputs, 'feature1')
      assertLength('DINOv3 feature1', feature1, elementCount(TRIPOSPLAT_FEATURE1_SHAPE))
    } finally {
      await context.runtime.disposeGraph(context.sessionIds.dino)
    }

    let feature2: Float32Array
    options.onProgress?.({ stage: 'vae', message: 'Running the Flux VAE encoder on WebGPU.' })
    const vaeGraph = await loadStage(context, 'vae')
    try {
      const response = await context.runtime.runGraph(
        context.sessionIds.vae,
        {
          image_rgb: inputTensor(vaeGraph, prepared.tensors.rgb.data, TRIPOSPLAT_IMAGE_SHAPE),
          epsilon: inputTensor(vaeGraph, vaeNoise, TRIPOSPLAT_VAE_NOISE_SHAPE),
        },
        runOptions(['feature2'], options.signal, 'flux-vae-encoder'),
      )
      timings.vaeInferenceMs = response.timings.inferenceMs
      feature2 = outputFloat32(response.outputs, 'feature2')
      assertLength('Flux VAE feature2', feature2, elementCount(TRIPOSPLAT_FEATURE2_SHAPE))
    } finally {
      await context.runtime.disposeGraph(context.sessionIds.vae)
    }
    throwIfAborted(options.signal)

    const latent = explicitOrRandomNoise(
      options.latentNoise,
      elementCount(TRIPOSPLAT_LATENT_SHAPE),
      'latent noise',
      random,
    )
    const camera = explicitOrRandomNoise(
      options.cameraNoise,
      elementCount(TRIPOSPLAT_CAMERA_SHAPE),
      'camera noise',
      random,
    )

    const ditGraph = await loadStage(context, 'dit')
    let flowState: FlowState
    try {
      const positiveInputs: TensorMap = {
        feature1: inputTensor(ditGraph, feature1, TRIPOSPLAT_FEATURE1_SHAPE),
        feature2: inputTensor(ditGraph, feature2, TRIPOSPLAT_FEATURE2_SHAPE),
      }
      const negativeInputs: TensorMap = {
        feature1: inputTensor(
          ditGraph,
          new Float32Array(elementCount(TRIPOSPLAT_FEATURE1_SHAPE)),
          TRIPOSPLAT_FEATURE1_SHAPE,
        ),
        feature2: inputTensor(
          ditGraph,
          new Float32Array(elementCount(TRIPOSPLAT_FEATURE2_SHAPE)),
          TRIPOSPLAT_FEATURE2_SHAPE,
        ),
      }
      const condition: TripoSplatCondition = {
        inputs: await prepareReusableGraphInputs(
          context.runtime,
          context.sessionIds.dit,
          'dit-positive-conditioning',
          positiveInputs,
          options.signal,
        ),
      }
      const negativeCondition: TripoSplatCondition = {
        inputs: await prepareReusableGraphInputs(
          context.runtime,
          context.sessionIds.dit,
          'dit-negative-conditioning',
          negativeInputs,
          options.signal,
        ),
      }
      const sampler = createSampler<TripoSplatCondition>(async (
        invocation: FlowInvocation<TripoSplatCondition>,
      ) => {
        options.onProgress?.({
          stage: 'sampling',
          message: `Running ${invocation.pass} DiT invocation ${invocation.invocation}/${invocation.totalInvocations}.`,
          progress: invocation.invocation / invocation.totalInvocations,
          step: invocation.step,
          totalSteps: invocation.totalSteps,
          invocation: invocation.invocation,
          totalInvocations: invocation.totalInvocations,
        })
        const response = await invocation.condition.inputs.run(
          {
            latent: inputTensor(
              ditGraph,
              new Float32Array(invocation.sample.latent),
              TRIPOSPLAT_LATENT_SHAPE,
            ),
            camera: inputTensor(
              ditGraph,
              new Float32Array(invocation.sample.camera),
              TRIPOSPLAT_CAMERA_SHAPE,
            ),
            t: inputTensor(ditGraph, new Float32Array(invocation.timestepTensor), [1]),
          },
          runOptions(
            ['pred_latent', 'pred_camera'],
            options.signal,
            `flow-${invocation.pass}-${invocation.step}-of-${invocation.totalSteps}`,
          ),
        )
        timings.ditInferenceMs += response.timings.inferenceMs
        timings.ditReadbackMs += response.timings.readbackMs
        const predictedLatent = outputFloat32(response.outputs, 'pred_latent')
        const predictedCamera = outputFloat32(response.outputs, 'pred_camera')
        assertLength('DiT pred_latent', predictedLatent, elementCount(TRIPOSPLAT_LATENT_SHAPE))
        assertLength('DiT pred_camera', predictedCamera, elementCount(TRIPOSPLAT_CAMERA_SHAPE))
        return { latent: predictedLatent, camera: predictedCamera }
      })
      flowState = await sampler.sample(
        { latent, camera },
        {
          condition,
          negativeCondition,
          steps: configuration.steps,
          guidanceScale: configuration.guidanceScale,
          shift: configuration.shift,
          arithmetic: (ditGraph.precision ?? context.manifest.precision) === 'fp32'
            ? 'fp32'
            : 'fp16',
          ...(options.signal === undefined ? {} : { signal: options.signal }),
        },
      )
    } finally {
      await context.runtime.disposeGraph(context.sessionIds.dit)
    }
    const sampledLatent = flowState.latent
    assertLength('sampled latent', sampledLatent, elementCount(TRIPOSPLAT_LATENT_SHAPE))

    const octreeGraph = await loadStage(context, 'octree')
    options.onProgress?.({ stage: 'octree', message: 'Sampling the eight-level occupancy octree.' })
    let points: Float32Array
    try {
      const result = await sampleOctree(
        async (invocation) => {
          const paddedCenters = new Float32Array(TRIPOSPLAT_MAX_DECODER_POINTS * 3)
          paddedCenters.set(invocation.parentCenters)
          const response = await context.runtime.runGraph(
            context.sessionIds.octree,
            {
              x: inputTensor(octreeGraph, paddedCenters, [1, TRIPOSPLAT_MAX_DECODER_POINTS, 3]),
              l: inputTensor(octreeGraph, Float32Array.of(invocation.resolution), [1]),
              cond: inputTensor(
                octreeGraph,
                new Float32Array(invocation.condition),
                TRIPOSPLAT_LATENT_SHAPE,
              ),
            },
            runOptions(['logits'], options.signal, `octree-level-${invocation.level}`),
          )
          timings.octreeInferenceMs += response.timings.inferenceMs
          const paddedLogits = outputFloat32(response.outputs, 'logits')
          const activeLength = invocation.parentCount * 8
          if (paddedLogits.length < activeLength) {
            throw new GraphCapabilityError(
              `Octree graph returned ${paddedLogits.length} logits for ${invocation.parentCount} parents.`,
            )
          }
          return { logits: paddedLogits.slice(0, activeLength) }
        },
        {
          condition: sampledLatent,
          numPoints: TRIPOSPLAT_MAX_DECODER_POINTS,
          temperature: options.octreeTemperature ?? 1,
          rng: () => random.next(),
          ...(options.signal === undefined ? {} : { signal: options.signal }),
          onLevel: ({ level, totalLevels, occupiedVoxels }) => {
            options.onProgress?.({
              stage: 'octree',
              message: `Octree level ${level}/${totalLevels}: ${occupiedVoxels} occupied voxels.`,
              progress: level / totalLevels,
            })
          },
        },
      )
      points = result.points
    } finally {
      await context.runtime.disposeGraph(context.sessionIds.octree)
    }
    assertLength('octree points', points, TRIPOSPLAT_MAX_DECODER_POINTS * 3)

    const gaussianGraph = await loadStage(context, 'gaussianDecoder')
    options.onProgress?.({ stage: 'gaussian-decoder', message: 'Decoding Gaussian features on WebGPU.' })
    let features: Float32Array
    try {
      const response = await context.runtime.runGraph(
        context.sessionIds.gaussianDecoder,
        {
          points: inputTensor(
            gaussianGraph,
            new Float32Array(points),
            [1, TRIPOSPLAT_MAX_DECODER_POINTS, 3],
          ),
          cond: inputTensor(
            gaussianGraph,
            new Float32Array(sampledLatent),
            TRIPOSPLAT_LATENT_SHAPE,
          ),
        },
        runOptions(['features'], options.signal, 'gaussian-decoder'),
      )
      timings.gaussianInferenceMs = response.timings.inferenceMs
      features = outputFloat32(response.outputs, 'features')
    } finally {
      await context.runtime.disposeGraph(context.sessionIds.gaussianDecoder)
    }
    assertLength(
      'Gaussian features',
      features,
      TRIPOSPLAT_MAX_DECODER_POINTS * GAUSSIAN_FEATURE_WIDTH,
    )

    timings.throughGaussianInferenceMs = performance.now() - startedAt
    options.onProgress?.({ stage: 'packing', message: 'Packing the canonical Gaussian scene.' })
    const metadata: GaussianSceneMetadata = {
      coordinateSystem: 'triposplat-object',
      units: 'model-unit',
      rotationOrder: 'wxyz',
      scaleEncoding: 'linear',
      opacityEncoding: 'linear',
      colorSemantics: null,
      sphericalHarmonicsSemantics: 'degree-0-rgb',
      modelRevision: context.manifest.modelRevision,
      generationSettings: {
        steps: configuration.steps,
        guidanceScale: configuration.guidanceScale,
        shift: configuration.shift,
        gaussianCount: configuration.gaussianCount,
        precision: context.manifest.precision,
        inputIsPrepared: options.inputIsPrepared ?? false,
        usedBackgroundRemoval: prepared.usedBackgroundRemoval,
        measuredTimingsMs: { ...timings },
      },
      seed: configuration.seed,
      runtimeVersion: RUNTIME_VERSION,
      plyExportTransform: TRIPOSPLAT_EXPORT_TRANSFORM,
    }
    const scene = decodeGaussians(points, features, { metadata })
    options.onProgress?.({ stage: 'complete', message: 'TripoSplat generation is complete.', progress: 1 })
    return scene
  } finally {
    prepared.dispose()
  }
}
