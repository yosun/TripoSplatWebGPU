import assert from 'node:assert/strict'
import test from 'node:test'

import {
  sampleFlow4Steps,
  sampleFlow20Steps,
  type FlowModelInvocation,
} from '../src/models/triposplat/flowSampler.ts'

function assertClose(actual: number, expected: number, tolerance = 1e-5): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  )
}

test('four-step flow performs eight CFG invocations and applies every shifted Euler interval', async () => {
  const calls: FlowModelInvocation<string>[] = []
  const progress: number[] = []
  const originalNoise = Float32Array.of(20)

  const result = await sampleFlow4Steps(
    (invocation) => {
      calls.push(invocation)
      const stepVelocity = invocation.step
      return {
        // With scale 3 this blends to 3*(2*step) - 2*step = 4*step.
        latent: Float32Array.of(
          invocation.pass === 'conditional' ? 2 * stepVelocity : stepVelocity,
        ),
      }
    },
    { latent: originalNoise },
    {
      condition: 'image',
      negativeCondition: 'zeros',
      guidanceScale: 3,
      shift: 3,
      batchSize: 2,
      onStep: ({ sample }) => progress.push(sample.latent[0]),
    },
  )

  assert.equal(calls.length, 8)
  assert.deepEqual(calls.map(({ pass }) => pass), [
    'conditional', 'unconditional',
    'conditional', 'unconditional',
    'conditional', 'unconditional',
    'conditional', 'unconditional',
  ])
  assert.deepEqual(calls.map(({ condition }) => condition), [
    'image', 'zeros', 'image', 'zeros', 'image', 'zeros', 'image', 'zeros',
  ])
  assert.deepEqual(
    calls.filter(({ pass }) => pass === 'conditional').map(({ timestep }) => timestep),
    [1, 0.9, 0.75, 0.5],
  )
  assert.deepEqual(
    calls.filter(({ pass }) => pass === 'conditional').map(({ timestepTensor }) => [
      ...timestepTensor,
    ]),
    [[1000, 1000], [900, 900], [750, 750], [500, 500]],
  )

  // 20 - (0.1*4 + 0.15*8 + 0.25*12 + 0.5*16) = 7.4.
  assertClose(result.latent[0], 7.4)
  assert.equal(progress.length, 4)
  assertClose(progress[0], 19.6)
  assertClose(progress[1], 18.4)
  assertClose(progress[2], 15.4)
  assertClose(progress[3], 7.4)
  assert.equal(originalNoise[0], 20, 'sampling must not mutate caller-owned noise')
})

test('twenty-step flow performs forty CFG invocations and its deltas integrate to one', async () => {
  const calls: FlowModelInvocation<string>[] = []
  const result = await sampleFlow20Steps(
    (invocation) => {
      calls.push(invocation)
      return {
        // With scale 3: 3*5 - 2*1 = 13 for every step.
        latent: Float32Array.of(invocation.pass === 'conditional' ? 5 : 1),
      }
    },
    { latent: Float32Array.of(100) },
    {
      condition: 'image',
      negativeCondition: 'zeros',
      guidanceScale: 3,
      shift: 3,
    },
  )

  assert.equal(calls.length, 40)
  for (let step = 1; step <= 20; step += 1) {
    const conditional = calls[(step - 1) * 2]
    const unconditional = calls[(step - 1) * 2 + 1]
    assert.equal(conditional.step, step)
    assert.equal(unconditional.step, step)
    assert.equal(conditional.totalSteps, 20)
    assert.equal(unconditional.totalSteps, 20)
    assert.equal(conditional.pass, 'conditional')
    assert.equal(unconditional.pass, 'unconditional')
    assert.equal(conditional.timestep, unconditional.timestep)
  }
  assert.equal(calls[0].timestep, 1)
  assert.ok(calls.at(-1)!.timestep > 0)
  assertClose(result.latent[0], 87, 2e-5)
})

test('flow rejects missing CFG conditions and malformed prediction tensors', async () => {
  let calls = 0
  await assert.rejects(
    sampleFlow4Steps(
      () => {
        calls += 1
        return { latent: Float32Array.of(1) }
      },
      { latent: Float32Array.of(1) },
      { condition: 'image', guidanceScale: 3 },
    ),
    /negativeCondition is required/,
  )
  assert.equal(calls, 0)

  await assert.rejects(
    sampleFlow4Steps(
      () => ({ latent: new Float32Array(0) }),
      { latent: Float32Array.of(1) },
      { condition: 'image', guidanceScale: 1 },
    ),
    /has 0 values; expected 1/,
  )
})

test('flow observes a pre-aborted signal before invoking the model', async () => {
  const controller = new AbortController()
  controller.abort(new Error('stop now'))
  let calls = 0

  await assert.rejects(
    sampleFlow20Steps(
      () => {
        calls += 1
        return { latent: Float32Array.of(0) }
      },
      { latent: Float32Array.of(1) },
      { condition: 'image', guidanceScale: 1, signal: controller.signal },
    ),
    /stop now/,
  )
  assert.equal(calls, 0)
})
