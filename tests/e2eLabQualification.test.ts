import assert from 'node:assert/strict'
import test from 'node:test'

import {
  qualifyGaussianSceneStructure,
  qualifyGaussianViewer,
} from '../src/e2eLabQualification.ts'

function validScene() {
  return {
    count: 2,
    positions: Float32Array.of(0, 1, 2, 3, 4, 5),
    scales: Float32Array.of(1, 1, 1, 2, 2, 2),
    rotations: Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0),
    opacities: Float32Array.of(0.5, 0.75),
    sphericalHarmonics: Float32Array.of(0, 0.1, 0.2, 0.3, 0.4, 0.5),
    metadata: {
      scaleEncoding: 'linear',
      opacityEncoding: 'linear',
      sphericalHarmonicsSemantics: 'degree-0-rgb',
      generationSettings: { steps: 4, inputIsPrepared: true },
    },
    isDisposed: false,
  }
}

test('E2E scene qualification accepts the canonical finite array contract', () => {
  const result = qualifyGaussianSceneStructure(validScene(), 2)

  assert.equal(result.passed, true)
  assert.equal(result.countMatches, true)
  assert.equal(result.arrays.rotations.expectedLength, 8)
  assert.equal(result.arrays.sphericalHarmonics.finite, true)
  assert.deepEqual(result.metadata, {
    fourStepSchedule: true,
    preparedInputRecorded: true,
    degreeZeroSphericalHarmonics: true,
    linearScales: true,
    linearOpacities: true,
  })
})

test('E2E scene qualification rejects non-finite, missing, and stale metadata', () => {
  const scene = validScene()
  scene.positions[4] = Number.NaN
  const result = qualifyGaussianSceneStructure({
    ...scene,
    sphericalHarmonics: undefined,
    metadata: {
      ...scene.metadata,
      generationSettings: { steps: 20, inputIsPrepared: false },
    },
  }, 2)

  assert.equal(result.passed, false)
  assert.equal(result.arrays.positions.nonFiniteCount, 1)
  assert.equal(result.arrays.sphericalHarmonics.present, false)
  assert.equal(result.metadata.fourStepSchedule, false)
  assert.equal(result.metadata.preparedInputRecorded, false)
})

test('E2E scene qualification rejects a disposed or wrong-size scene', () => {
  const scene = validScene()
  const result = qualifyGaussianSceneStructure({
    ...scene,
    count: 1,
    isDisposed: true,
  }, 2)

  assert.equal(result.passed, false)
  assert.equal(result.countMatches, false)
  assert.equal(result.sceneWasLive, false)
})

test('E2E viewer qualification accepts ready state with a visible drawing canvas', () => {
  const result = qualifyGaussianViewer({
    status: 'ready',
    message: 'Preview ready.',
    loadMs: 123.5,
    timeoutMs: 90_000,
    canvas: {
      present: true,
      width: 1600,
      height: 900,
      clientWidth: 800,
      clientHeight: 450,
    },
  })

  assert.equal(result.passed, true)
  assert.equal(result.canvas.drawingBufferDimensionsPresent, true)
  assert.equal(result.canvas.displayDimensionsPresent, true)
  assert.equal(result.loadMs, 123.5)
})

test('E2E viewer qualification rejects failed state and zero-size canvases', () => {
  const failed = qualifyGaussianViewer({
    status: 'failed',
    message: 'Preview failed.',
    loadMs: 50,
    timeoutMs: 90_000,
    canvas: {
      present: true,
      width: 1600,
      height: 900,
      clientWidth: 800,
      clientHeight: 450,
    },
  })
  const zeroSize = qualifyGaussianViewer({
    status: 'ready',
    message: 'Preview ready.',
    loadMs: 25,
    timeoutMs: 90_000,
    canvas: {
      present: true,
      width: 0,
      height: 0,
      clientWidth: 0,
      clientHeight: 0,
    },
  })

  assert.equal(failed.passed, false)
  assert.equal(zeroSize.passed, false)
  assert.equal(zeroSize.canvas.drawingBufferDimensionsPresent, false)
  assert.equal(zeroSize.canvas.displayDimensionsPresent, false)
})
