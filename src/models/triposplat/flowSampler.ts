/** Host-controlled Euler/CFG loop translated from TripoSplat's FlowEulerCfgSampler. */

import { float16BitsToNumber, numberToFloat16Bits } from '../../runtime/float16.ts'

export const TRIPOSPLAT_FAST_FLOW_STEPS = 4
export const TRIPOSPLAT_QUALITY_FLOW_STEPS = 20
export const TRIPOSPLAT_DEFAULT_FLOW_SHIFT = 3
export const TRIPOSPLAT_DEFAULT_GUIDANCE_SCALE = 3

export type FlowTensorState = Record<string, Float32Array>
export type GuidanceScale = number | Readonly<Record<string, number>> | null
export type FlowArithmeticPrecision = 'float32' | 'float16'

export interface ShiftedFlowStep {
  /** Normalized source timestep in [0, 1]. */
  timestep: number
  /** Normalized destination timestep in [0, 1]. */
  previousTimestep: number
  /** Positive Euler interval: timestep - previousTimestep. */
  delta: number
}

export interface FlowModelInvocation<Condition> {
  /** A defensive copy of the current host state. */
  sample: Readonly<FlowTensorState>
  /** Normalized flow timestep before the official x1000 model scaling. */
  timestep: number
  /** `[batch]` float32 tensor containing `1000 * timestep`. */
  timestepTensor: Float32Array
  condition: Condition
  pass: 'conditional' | 'unconditional'
  step: number
  totalSteps: number
}

export type FlowModelPredictor<Condition> = (
  invocation: FlowModelInvocation<Condition>,
) => FlowTensorState | Promise<FlowTensorState>

export interface FlowStepProgress {
  step: number
  totalSteps: number
  timestep: number
  previousTimestep: number
  sample: Readonly<FlowTensorState>
}

export interface FlowSamplerOptions<Condition> {
  condition: Condition
  /** Required only when at least one effective guidance scale is greater than 1. */
  negativeCondition?: Condition
  steps?: number
  shift?: number
  guidanceScale?: GuidanceScale
  /** Precision used by official CFG and velocity scaling; the fp16 DiT requires `float16`. */
  predictionArithmetic?: FlowArithmeticPrecision
  /** TripoSplat image-to-3D inference currently uses one image per invocation. */
  batchSize?: number
  signal?: AbortSignal
  onStep?: (progress: FlowStepProgress) => void
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}.`)
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new DOMException('Operation aborted', 'AbortError')
  }
}

function ownEntries(state: Readonly<FlowTensorState>): [string, Float32Array][] {
  return Object.entries(state)
}

export function cloneFlowTensorState(state: Readonly<FlowTensorState>): FlowTensorState {
  const clone: FlowTensorState = {}
  for (const [key, value] of ownEntries(state)) {
    if (!(value instanceof Float32Array)) {
      throw new Error(`Flow tensor '${key}' must be a Float32Array.`)
    }
    clone[key] = new Float32Array(value)
  }
  if (Object.keys(clone).length === 0) {
    throw new Error('Flow state must contain at least one tensor.')
  }
  return clone
}

/** The official schedule transform: shift*t / (1 + (shift - 1)*t). */
export function shiftFlowTimestep(timestep: number, shift: number): number {
  if (!Number.isFinite(timestep) || timestep < 0 || timestep > 1) {
    throw new Error(`timestep must be finite and in [0, 1], got ${timestep}.`)
  }
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new Error(`shift must be a positive finite number, got ${shift}.`)
  }
  return (shift * timestep) / (1 + (shift - 1) * timestep)
}

/**
 * Builds the exact descending schedule used by the official NumPy sampler.
 * A shift of 1 is uniform; values above 1 retain more high-noise timesteps.
 */
export function createShiftedFlowSchedule(
  steps: number,
  shift = TRIPOSPLAT_DEFAULT_FLOW_SHIFT,
): readonly ShiftedFlowStep[] {
  assertPositiveInteger(steps, 'steps')
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new Error(`shift must be a positive finite number, got ${shift}.`)
  }

  const timesteps = new Float64Array(steps + 1)
  // NumPy's descending `linspace(1, 0, steps + 1)` is formed from a negative
  // delta. Keeping that operation order preserves its last-bit schedule values.
  const linearDelta = -1 / steps
  for (let index = 0; index <= steps; index += 1) {
    const linearTimestep = index === steps ? 0 : 1 + index * linearDelta
    timesteps[index] = shiftFlowTimestep(linearTimestep, shift)
  }
  // Avoid a possible signed zero at the terminal endpoint.
  timesteps[steps] = 0

  const schedule: ShiftedFlowStep[] = new Array(steps)
  for (let index = 0; index < steps; index += 1) {
    const timestep = timesteps[index]
    const previousTimestep = timesteps[index + 1]
    schedule[index] = {
      timestep,
      previousTimestep,
      delta: timestep - previousTimestep,
    }
  }
  return schedule
}

function guidanceForKey(guidanceScale: GuidanceScale | undefined, key: string): number {
  if (guidanceScale === undefined || guidanceScale === null) return 1
  if (typeof guidanceScale === 'number') return guidanceScale
  return guidanceScale[key] ?? 1
}

export function usesClassifierFreeGuidance(guidanceScale: GuidanceScale | undefined): boolean {
  if (guidanceScale === undefined || guidanceScale === null) return false
  if (typeof guidanceScale === 'number') {
    if (!Number.isFinite(guidanceScale)) {
      throw new Error(`guidanceScale must be finite, got ${guidanceScale}.`)
    }
    return guidanceScale > 1
  }
  for (const [key, scale] of Object.entries(guidanceScale)) {
    if (!Number.isFinite(scale)) {
      throw new Error(`guidanceScale['${key}'] must be finite, got ${scale}.`)
    }
    if (scale > 1) return true
  }
  return false
}

function assertPredictionMatchesSample(
  prediction: Readonly<FlowTensorState>,
  sample: Readonly<FlowTensorState>,
  label: string,
): void {
  for (const [key, sampleTensor] of ownEntries(sample)) {
    const predictionTensor = prediction[key]
    if (!(predictionTensor instanceof Float32Array)) {
      throw new Error(`${label} is missing Float32Array tensor '${key}'.`)
    }
    if (predictionTensor.length !== sampleTensor.length) {
      throw new Error(
        `${label} tensor '${key}' has ${predictionTensor.length} values; ` +
          `expected ${sampleTensor.length}.`,
      )
    }
  }
}

/**
 * Diffusers-style CFG from the official implementation:
 * `scale * conditional - (scale - 1) * unconditional`.
 */
export function blendClassifierFreeGuidance(
  conditional: Readonly<FlowTensorState>,
  unconditional: Readonly<FlowTensorState>,
  guidanceScale: Exclude<GuidanceScale, null>,
  arithmetic: FlowArithmeticPrecision = 'float32',
): FlowTensorState {
  const blended: FlowTensorState = {}
  for (const [key, conditionalTensor] of ownEntries(conditional)) {
    const scale = guidanceForKey(guidanceScale, key)
    if (!Number.isFinite(scale)) {
      throw new Error(`guidanceScale['${key}'] must be finite, got ${scale}.`)
    }
    if (scale <= 1) {
      blended[key] = new Float32Array(conditionalTensor)
      continue
    }
    const unconditionalTensor = unconditional[key]
    if (!(unconditionalTensor instanceof Float32Array)) {
      throw new Error(`Unconditional prediction is missing Float32Array tensor '${key}'.`)
    }
    if (unconditionalTensor.length !== conditionalTensor.length) {
      throw new Error(
        `Unconditional tensor '${key}' has ${unconditionalTensor.length} values; ` +
          `expected ${conditionalTensor.length}.`,
      )
    }
    const output = new Float32Array(conditionalTensor.length)
    for (let index = 0; index < output.length; index += 1) {
      if (arithmetic === 'float16') {
        const conditionalScaled = roundFloat16(scale * conditionalTensor[index])
        const unconditionalScaled = roundFloat16(
          (scale - 1) * unconditionalTensor[index],
        )
        output[index] = roundFloat16(conditionalScaled - unconditionalScaled)
      } else {
        const conditionalScaled = Math.fround(Math.fround(scale) * conditionalTensor[index])
        const unconditionalScaled = Math.fround(
          Math.fround(scale - 1) * unconditionalTensor[index],
        )
        output[index] = Math.fround(conditionalScaled - unconditionalScaled)
      }
    }
    blended[key] = output
  }
  return blended
}

function roundFloat16(value: number): number {
  return float16BitsToNumber(numberToFloat16Bits(value))
}

function scaleVelocity(
  value: number,
  delta: number,
  arithmetic: FlowArithmeticPrecision,
): number {
  return arithmetic === 'float16'
    ? roundFloat16(value * delta)
    : Math.fround(value * Math.fround(delta))
}

function makeModelTimestep(timestep: number, batchSize: number): Float32Array {
  const modelTimestep = new Float32Array(batchSize)
  modelTimestep.fill(1000 * timestep)
  return modelTimestep
}

export class FlowEulerCfgSampler<Condition> {
  private readonly predictor: FlowModelPredictor<Condition>

  constructor(predictor: FlowModelPredictor<Condition>) {
    this.predictor = predictor
  }

  async sample(
    noise: Readonly<FlowTensorState>,
    options: FlowSamplerOptions<Condition>,
  ): Promise<FlowTensorState> {
    const steps = options.steps ?? TRIPOSPLAT_QUALITY_FLOW_STEPS
    const shift = options.shift ?? TRIPOSPLAT_DEFAULT_FLOW_SHIFT
    const guidanceScale = options.guidanceScale ?? TRIPOSPLAT_DEFAULT_GUIDANCE_SCALE
    const predictionArithmetic = options.predictionArithmetic ?? 'float32'
    const batchSize = options.batchSize ?? 1
    assertPositiveInteger(batchSize, 'batchSize')
    const schedule = createShiftedFlowSchedule(steps, shift)
    const needsUnconditional = usesClassifierFreeGuidance(guidanceScale)
    if (needsUnconditional && !Object.prototype.hasOwnProperty.call(options, 'negativeCondition')) {
      throw new Error('negativeCondition is required when guidanceScale is greater than 1.')
    }

    // Numerically equivalent to official `sample = noise`, without mutating the caller's buffers.
    const sample = cloneFlowTensorState(noise)
    for (let index = 0; index < schedule.length; index += 1) {
      throwIfAborted(options.signal)
      const interval = schedule[index]
      const invocationBase = {
        timestep: interval.timestep,
        timestepTensor: makeModelTimestep(interval.timestep, batchSize),
        step: index + 1,
        totalSteps: schedule.length,
      }
      const conditional = await this.predictor({
        ...invocationBase,
        sample: cloneFlowTensorState(sample),
        condition: options.condition,
        pass: 'conditional',
      })
      throwIfAborted(options.signal)
      assertPredictionMatchesSample(conditional, sample, 'Conditional prediction')

      let prediction = conditional
      if (needsUnconditional) {
        const unconditional = await this.predictor({
          ...invocationBase,
          sample: cloneFlowTensorState(sample),
          condition: options.negativeCondition as Condition,
          pass: 'unconditional',
        })
        throwIfAborted(options.signal)
        assertPredictionMatchesSample(unconditional, sample, 'Unconditional prediction')
        prediction = blendClassifierFreeGuidance(
          conditional,
          unconditional,
          guidanceScale,
          predictionArithmetic,
        )
      }

      // Official Euler update: sample = sample - velocity * (t - t_previous).
      for (const [key, sampleTensor] of ownEntries(sample)) {
        const velocity = prediction[key]
        for (let element = 0; element < sampleTensor.length; element += 1) {
          sampleTensor[element] = Math.fround(
            sampleTensor[element] - scaleVelocity(
              velocity[element],
              interval.delta,
              predictionArithmetic,
            ),
          )
        }
      }
      throwIfAborted(options.signal)
      options.onStep?.({
        step: index + 1,
        totalSteps: schedule.length,
        timestep: interval.timestep,
        previousTimestep: interval.previousTimestep,
        sample,
      })
    }
    return sample
  }
}

export function sampleFlowEulerCfg<Condition>(
  predictor: FlowModelPredictor<Condition>,
  noise: Readonly<FlowTensorState>,
  options: FlowSamplerOptions<Condition>,
): Promise<FlowTensorState> {
  return new FlowEulerCfgSampler(predictor).sample(noise, options)
}

export function sampleFlow4Steps<Condition>(
  predictor: FlowModelPredictor<Condition>,
  noise: Readonly<FlowTensorState>,
  options: Omit<FlowSamplerOptions<Condition>, 'steps'>,
): Promise<FlowTensorState> {
  return sampleFlowEulerCfg(predictor, noise, { ...options, steps: TRIPOSPLAT_FAST_FLOW_STEPS })
}

export function sampleFlow20Steps<Condition>(
  predictor: FlowModelPredictor<Condition>,
  noise: Readonly<FlowTensorState>,
  options: Omit<FlowSamplerOptions<Condition>, 'steps'>,
): Promise<FlowTensorState> {
  return sampleFlowEulerCfg(predictor, noise, { ...options, steps: TRIPOSPLAT_QUALITY_FLOW_STEPS })
}
