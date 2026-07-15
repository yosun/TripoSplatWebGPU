import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

import { decodeGaussians } from '../dist/low-level.js'

const metadata = {
  coordinateSystem: 'triposplat-object',
  units: 'model-unit',
  rotationOrder: 'wxyz',
  scaleEncoding: 'linear',
  opacityEncoding: 'linear',
  colorSemantics: null,
  sphericalHarmonicsSemantics: 'degree-0-rgb',
  modelRevision: 'official-oracle',
  generationSettings: {},
  seed: 20260715,
  runtimeVersion: 'test',
}

function maximumAbsoluteError(actual, expected) {
  assert.equal(actual.length, expected.length)
  let maximum = 0
  for (let index = 0; index < actual.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(actual[index] - expected[index]))
  }
  return maximum
}

test('Gaussian host activations match official _build_gaussians', async () => {
  const fixtureUrl = new URL('./fixtures/gaussian-activation-official.json', import.meta.url)
  const fixture = JSON.parse(await readFile(fixtureUrl, 'utf8'))
  assert.equal(fixture.source.commit, 'a78fa12d06dbf1381ca548bfac32bb68cb8c451d')

  const scene = decodeGaussians(
    Float32Array.from(fixture.inputs.points),
    Float32Array.from(fixture.inputs.features),
    { metadata },
  )
  try {
    const comparisons = [
      ['positions', scene.positions, fixture.expected.positions],
      ['scales', scene.scales, fixture.expected.scales],
      ['rotations', scene.rotations, fixture.expected.rotations],
      ['opacities', scene.opacities, fixture.expected.opacities],
      ['spherical harmonics', scene.sphericalHarmonics, fixture.expected.spherical_harmonics],
    ]
    for (const [label, actual, expected] of comparisons) {
      assert.ok(actual, `${label} output is missing`)
      const maximum = maximumAbsoluteError(actual, expected)
      assert.ok(maximum <= 2e-6, `${label} maximum absolute error ${maximum} exceeds 2e-6`)
    }
  } finally {
    scene.dispose()
  }
})
