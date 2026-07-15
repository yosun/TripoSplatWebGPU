import assert from 'node:assert/strict'
import test from 'node:test'

import {
  float16BitsToNumber,
  float16ToFloat32,
  float32ToFloat16,
  numberToFloat16Bits,
} from '../src/runtime/float16.ts'
import {
  assertModelManifest,
  copyModelManifest,
  createConventionalExternalDataManifest,
  resolveModelManifest,
  type OnnxModelManifest,
} from '../src/runtime/modelManifest.ts'
import {
  createTripoSplatModelManifest,
  createVaeEncoderSliceManifest,
} from '../src/models/triposplat/manifests.ts'

test('model manifests preserve explicit virtual sidecar paths while resolving fetch URLs', () => {
  const original: OnnxModelManifest = {
    graphUrl: 'graphs/model.onnx',
    externalData: [
      { path: 'weights/block-0.bin', url: 'objects/model-v7/block-0.bin?token=abc' },
      { path: '../shared/block-1.bin', url: 'https://weights.example/block-1.bin' },
    ],
  }
  assert.doesNotThrow(() => assertModelManifest(original))

  const resolved = resolveModelManifest(original, 'https://cdn.example/releases/v7/')
  assert.equal(resolved.graphUrl, 'https://cdn.example/releases/v7/graphs/model.onnx')
  assert.deepEqual(resolved.externalData, [
    {
      path: 'weights/block-0.bin',
      url: 'https://cdn.example/releases/v7/objects/model-v7/block-0.bin?token=abc',
    },
    {
      path: '../shared/block-1.bin',
      url: 'https://weights.example/block-1.bin',
    },
  ])

  const copied = copyModelManifest(original)
  copied.externalData![0].url = 'changed'
  assert.equal(original.externalData![0].url, 'objects/model-v7/block-0.bin?token=abc')
})

test('manifest validation rejects ambiguous duplicate and malformed sidecars', () => {
  assert.throws(
    () => assertModelManifest({
      graphUrl: 'model.onnx',
      externalData: [
        { path: 'model.data', url: 'a' },
        { path: 'model.data', url: 'b' },
      ],
    }),
    /Duplicate ONNX external-data path 'model.data'/,
  )
  assert.throws(
    () => assertModelManifest({ graphUrl: 'model.onnx', externalData: 'model.onnx.data' }),
    /must be an array/,
  )
  assert.throws(
    () => assertModelManifest({ graphUrl: '', externalData: [] }),
    /graphUrl must be a non-empty string/,
  )
  assert.throws(
    () => assertModelManifest({
      graphUrl: 'model.onnx',
      externalData: [{ path: 'bad\0path', url: 'data.bin' }],
    }),
    /must not contain a null character/,
  )
})

test('conventional sidecar helper is opt-in and preserves query/hash placement', () => {
  assert.deepEqual(
    createConventionalExternalDataManifest('https://cdn.example/model%20v2.onnx?sig=123#asset'),
    {
      graphUrl: 'https://cdn.example/model%20v2.onnx?sig=123#asset',
      externalData: [{
        path: 'model v2.onnx.data',
        url: 'https://cdn.example/model%20v2.onnx.data?sig=123#asset',
      }],
    },
  )
})

test('TripoSplat graph manifests explicitly declare each ONNX sidecar', () => {
  const graphs = createTripoSplatModelManifest({
    baseUrl: 'https://models.example/triposplat/v1',
  })
  assert.deepEqual(Object.keys(graphs), [
    'dinov3',
    'vaeEncoder',
    'dit',
    'octree',
    'gaussianDecoder',
  ])
  for (const descriptor of Object.values(graphs)) {
    assert.ok(descriptor)
    const graphFile = descriptor.manifest.graphUrl.split('/').at(-1)!
    assert.equal(descriptor.precision, 'float32')
    assert.deepEqual(descriptor.manifest.externalData, [{
      path: `${graphFile}.data`,
      url: `https://models.example/triposplat/v1/${graphFile}.data`,
    }])
  }

  const slice = createVaeEncoderSliceManifest(
    'https://models.example/graphs/vae-v3.onnx',
    'float16',
  )
  assert.equal(slice.vaeEncoder!.precision, 'float16')
  assert.deepEqual(slice.vaeEncoder!.manifest.externalData, [{
    path: 'flux2_vae_encoder.onnx.data',
    url: 'https://models.example/graphs/vae-v3.onnx.data',
  }])
})

test('float16 conversion matches canonical IEEE-754 binary16 encodings', () => {
  const vectors: Array<[number, number]> = [
    [0, 0x0000],
    [-0, 0x8000],
    [1, 0x3c00],
    [-2, 0xc000],
    [65504, 0x7bff],
    [2 ** -14, 0x0400],
    [2 ** -24, 0x0001],
    [2 ** -25, 0x0000],
    [Infinity, 0x7c00],
    [-Infinity, 0xfc00],
  ]
  for (const [value, bits] of vectors) {
    assert.equal(numberToFloat16Bits(value), bits)
  }
  assert.ok(Number.isNaN(float16BitsToNumber(numberToFloat16Bits(Number.NaN))))
  assert.ok(Object.is(float16BitsToNumber(0x8000), -0))

  // Round-to-nearest, ties-to-even on either side of an odd half code.
  assert.equal(numberToFloat16Bits(1 + 2 ** -11), 0x3c00)
  assert.equal(numberToFloat16Bits(1 + 3 * 2 ** -11), 0x3c02)
})

test('float16 vector helpers round-trip every non-NaN binary16 pattern', () => {
  for (let bits = 0; bits <= 0xffff; bits += 1) {
    const exponent = (bits >>> 10) & 0x1f
    const mantissa = bits & 0x3ff
    if (exponent === 0x1f && mantissa !== 0) continue
    assert.equal(numberToFloat16Bits(float16BitsToNumber(bits)), bits)
  }

  const source = Float32Array.of(0, -0, 1 / 3, 65504, Infinity, Number.NaN)
  const encoded = float32ToFloat16(source)
  const decoded = float16ToFloat32(encoded)
  assert.equal(decoded.length, source.length)
  assert.ok(Object.is(decoded[1], -0))
  assert.equal(decoded[3], 65504)
  assert.equal(decoded[4], Infinity)
  assert.ok(Number.isNaN(decoded[5]))
  assert.ok(Math.abs(decoded[2] - 1 / 3) < 2e-4)
})
