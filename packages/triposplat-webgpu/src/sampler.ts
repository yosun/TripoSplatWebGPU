import { throwIfAborted } from './errors.js'
import { float16BitsToNumber, numberToFloat16Bits } from './tensors.js'

export const FAST_FLOW_STEPS = 4
export const QUALITY_FLOW_STEPS = 20
export const DEFAULT_FLOW_SHIFT = 3
export const DEFAULT_GUIDANCE_SCALE = 3

export type FlowState = Record<string, Float32Array>
export type FlowCondition = unknown
export type FlowArithmetic = 'fp16' | 'fp32'

export interface FlowInvocation<Condition = FlowCondition> {
  sample: Readonly<FlowState>
  timestep: number
  /** Official model input: timestep multiplied by 1000. */
  timestepTensor: Float32Array
  condition: Condition
  pass: 'conditional' | 'unconditional'
  step: number
  totalSteps: number
  invocation: number
  totalInvocations: number
}

export type FlowPredictor<Condition = FlowCondition> = (
  invocation: FlowInvocation<Condition>,
) => FlowState | Promise<FlowState>

export interface SamplerOptions<Condition = FlowCondition> {
  condition: Condition
  negativeCondition?: Condition
  steps?: number
  shift?: number
  guidanceScale?: number | Readonly<Record<string, number>> | null
  arithmetic?: FlowArithmetic
  signal?: AbortSignal
  onStep?: (progress: {
    step: number
    totalSteps: number
    timestep: number
    previousTimestep: number
    sample: Readonly<FlowState>
  }) => void
}

export interface FlowSampler<Condition = FlowCondition> {
  sample(noise: Readonly<FlowState>, options: SamplerOptions<Condition>): Promise<FlowState>
}

export interface ShiftedFlowStep {
  timestep: number
  previousTimestep: number
  delta: number
}

function positiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer.`)
}

export function shiftedFlowTimestep(timestep: number, shift: number): number {
  if (!Number.isFinite(timestep) || timestep < 0 || timestep > 1) {
    throw new RangeError('timestep must be in [0, 1].')
  }
  if (!Number.isFinite(shift) || shift <= 0) throw new RangeError('shift must be positive.')
  return shift * timestep / (1 + (shift - 1) * timestep)
}

export function createFlowSchedule(steps: number, shift = DEFAULT_FLOW_SHIFT): ShiftedFlowStep[] {
  positiveInteger(steps, 'steps')
  const values = new Float64Array(steps + 1)
  for (let index = 0; index <= steps; index += 1) {
    const linear = index === steps ? 0 : 1 + index * (-1 / steps)
    values[index] = shiftedFlowTimestep(linear, shift)
  }
  values[steps] = 0
  return Array.from({ length: steps }, (_, index) => ({
    timestep: values[index],
    previousTimestep: values[index + 1],
    delta: values[index] - values[index + 1],
  }))
}

export function cloneFlowState(state: Readonly<FlowState>): FlowState {
  const clone: FlowState = {}
  for (const [name, values] of Object.entries(state)) {
    if (!(values instanceof Float32Array)) throw new TypeError(`Flow state '${name}' must be Float32Array.`)
    clone[name] = new Float32Array(values)
  }
  if (Object.keys(clone).length === 0) throw new TypeError('Flow state must not be empty.')
  return clone
}

function roundFp16(value: number): number {
  return float16BitsToNumber(numberToFloat16Bits(value))
}

function guidanceFor(scale: SamplerOptions['guidanceScale'], name: string): number {
  if (scale === undefined || scale === null) return 1
  return typeof scale === 'number' ? scale : scale[name] ?? 1
}

function usesGuidance(scale: SamplerOptions['guidanceScale']): boolean {
  if (scale === undefined || scale === null) return false
  if (typeof scale === 'number') return scale > 1
  return Object.values(scale).some((value) => value > 1)
}

function validatePrediction(prediction: Readonly<FlowState>, sample: Readonly<FlowState>, label: string): void {
  for (const [name, values] of Object.entries(sample)) {
    if (!(prediction[name] instanceof Float32Array) || prediction[name].length !== values.length) {
      throw new TypeError(`${label} tensor '${name}' does not match the sample.`)
    }
  }
}

export function blendGuidance(
  conditional: Readonly<FlowState>,
  unconditional: Readonly<FlowState>,
  scale: NonNullable<SamplerOptions['guidanceScale']>,
  arithmetic: FlowArithmetic = 'fp32',
): FlowState {
  const result: FlowState = {}
  for (const [name, values] of Object.entries(conditional)) {
    const strength = guidanceFor(scale, name)
    if (!Number.isFinite(strength)) throw new TypeError(`Guidance for '${name}' is not finite.`)
    const negative = unconditional[name]
    if (!(negative instanceof Float32Array) || negative.length !== values.length) {
      throw new TypeError(`Unconditional tensor '${name}' does not match the conditional tensor.`)
    }
    const blended = new Float32Array(values.length)
    for (let index = 0; index < values.length; index += 1) {
      if (strength <= 1) blended[index] = values[index]
      else if (arithmetic === 'fp16') {
        blended[index] = roundFp16(
          roundFp16(strength * values[index]) - roundFp16((strength - 1) * negative[index]),
        )
      } else {
        const positive = Math.fround(Math.fround(strength) * values[index])
        const negativeScaled = Math.fround(Math.fround(strength - 1) * negative[index])
        blended[index] = Math.fround(positive - negativeScaled)
      }
    }
    result[name] = blended
  }
  return result
}

class EulerCfgSampler<Condition> implements FlowSampler<Condition> {
  constructor(private readonly predictor: FlowPredictor<Condition>) {}

  async sample(noise: Readonly<FlowState>, options: SamplerOptions<Condition>): Promise<FlowState> {
    const steps = options.steps ?? QUALITY_FLOW_STEPS
    const schedule = createFlowSchedule(steps, options.shift ?? DEFAULT_FLOW_SHIFT)
    const scale = options.guidanceScale ?? DEFAULT_GUIDANCE_SCALE
    const guided = usesGuidance(scale)
    if (guided && !Object.prototype.hasOwnProperty.call(options, 'negativeCondition')) {
      throw new TypeError('negativeCondition is required for classifier-free guidance.')
    }
    const arithmetic = options.arithmetic ?? 'fp16'
    const state = cloneFlowState(noise)
    const totalInvocations = steps * (guided ? 2 : 1)
    let invocation = 0
    for (const [index, interval] of schedule.entries()) {
      throwIfAborted(options.signal)
      const shared = {
        timestep: interval.timestep,
        timestepTensor: Float32Array.of(interval.timestep * 1000),
        step: index + 1,
        totalSteps: steps,
        totalInvocations,
      }
      invocation += 1
      const conditional = await this.predictor({
        ...shared,
        invocation,
        sample: cloneFlowState(state),
        condition: options.condition,
        pass: 'conditional',
      })
      validatePrediction(conditional, state, 'Conditional prediction')
      let velocity = conditional
      if (guided) {
        throwIfAborted(options.signal)
        invocation += 1
        const unconditional = await this.predictor({
          ...shared,
          invocation,
          sample: cloneFlowState(state),
          condition: options.negativeCondition as Condition,
          pass: 'unconditional',
        })
        validatePrediction(unconditional, state, 'Unconditional prediction')
        velocity = blendGuidance(conditional, unconditional, scale, arithmetic)
      }
      for (const [name, values] of Object.entries(state)) {
        const prediction = velocity[name]
        for (let element = 0; element < values.length; element += 1) {
          const delta = arithmetic === 'fp16'
            ? roundFp16(prediction[element] * interval.delta)
            : Math.fround(prediction[element] * Math.fround(interval.delta))
          values[element] = Math.fround(values[element] - delta)
        }
      }
      throwIfAborted(options.signal)
      options.onStep?.({
        step: index + 1,
        totalSteps: steps,
        timestep: interval.timestep,
        previousTimestep: interval.previousTimestep,
        sample: state,
      })
    }
    return state
  }
}

export function createSampler<Condition>(predictor: FlowPredictor<Condition>): FlowSampler<Condition> {
  return new EulerCfgSampler(predictor)
}
