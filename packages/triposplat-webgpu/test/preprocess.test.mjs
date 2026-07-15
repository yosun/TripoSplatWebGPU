import assert from 'node:assert/strict'
import test from 'node:test'

import {
  BackgroundRemovalRequiredError,
  CancelledError,
  preprocessTripoSplatRgba,
} from '../dist/index.js'
import {
  buildTripoSplatEncoderTensors,
  calculateTripoSplatCrop,
  calculateTripoSplatResize,
  compositeRgbaOnBlack,
  cropRgba,
  erodeAlpha,
  resizeRgbaLanczos,
} from '../dist/preprocess.js'

test('resize geometry follows Python ties-to-even rounding', () => {
  assert.deepEqual(calculateTripoSplatResize(5, 4, 2), {
    width: 2,
    height: 2,
    scale: 0.5,
  })
  assert.deepEqual(calculateTripoSplatResize(7, 4, 2), {
    width: 4,
    height: 2,
    scale: 0.5,
  })
})

test('square crop geometry keeps official centering, padding, and truncation', () => {
  assert.deepEqual(
    calculateTripoSplatCrop({ minX: 1, minY: 2, maxX: 5, maxY: 4 }),
    { left: 0, top: 0, right: 5, bottom: 5 },
  )

  const cropped = cropRgba(
    {
      width: 2,
      height: 2,
      data: Uint8ClampedArray.from([
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
      ]),
    },
    { left: -1, top: 0, right: 2, bottom: 3 },
  )
  assert.equal(cropped.width, 3)
  assert.equal(cropped.height, 3)
  assert.deepEqual([...cropped.data.slice(4, 12)], [1, 2, 3, 4, 5, 6, 7, 8])
  assert.deepEqual([...cropped.data.slice(28)], new Array(8).fill(0))
})

test('Pillow-compatible RGBA operations preserve edge and alpha integer semantics', () => {
  const resized = resizeRgbaLanczos({
    width: 2,
    height: 1,
    data: Uint8ClampedArray.from([255, 0, 0, 0, 0, 0, 255, 255]),
  }, 8, 1)
  assert.deepEqual([...resized.data], [
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 255, 28,
    0, 0, 255, 93,
    0, 0, 255, 162,
    0, 0, 255, 227,
    0, 0, 255, 255,
    0, 0, 255, 255,
  ])

  const alphaRamp = {
    width: 3,
    height: 1,
    data: Uint8ClampedArray.from([0, 0, 0, 7, 0, 0, 0, 19, 0, 0, 0, 31]),
  }
  assert.deepEqual(
    [...erodeAlpha(alphaRamp, 1).data].filter((_value, index) => index % 4 === 3),
    [7, 7, 19],
  )
  assert.equal(compositeRgbaOnBlack({
    width: 1,
    height: 1,
    data: Uint8ClampedArray.from([127, 0, 0, 127]),
  }).data[0], 63)
})

test('encoder tensors are planar NCHW with official RGB, DINOv3, and VAE domains', () => {
  const image = {
    width: 2,
    height: 1,
    data: Uint8ClampedArray.from([0, 127, 255, 255, 128, 0]),
  }
  const tensors = buildTripoSplatEncoderTensors(image)
  assert.deepEqual(tensors.rgb.dims, [1, 3, 1, 2])
  assert.deepEqual(tensors.dinov3.dims, [1, 3, 1, 2])
  assert.deepEqual(tensors.vae.dims, [1, 3, 1, 2])

  const rgb = [0, 1, Math.fround(127 / 255), Math.fround(128 / 255), 1, 0]
  assert.deepEqual([...tensors.rgb.data], rgb)
  assert.deepEqual([...tensors.vae.data], rgb.map((value) => Math.fround(Math.fround(value * 2) - 1)))

  const means = [0.485, 0.456, 0.406]
  const standardDeviations = [0.229, 0.224, 0.225]
  const expectedDino = rgb.map((value, index) => {
    const channel = Math.floor(index / 2)
    return Math.fround(
      Math.fround(value - Math.fround(means[channel]))
      / Math.fround(standardDeviations[channel]),
    )
  })
  assert.deepEqual([...tensors.dinov3.data], expectedDino)
})

test('opaque raw input requires real external segmentation', async () => {
  const opaque = {
    width: 2,
    height: 2,
    data: new Uint8ClampedArray(2 * 2 * 4).fill(255),
  }
  await assert.rejects(
    preprocessTripoSplatRgba(opaque, { canvasSize: 2, erodeRadius: 0 }),
    (error) => error instanceof BackgroundRemovalRequiredError
      && error.code === 'BACKGROUND_REMOVAL_REQUIRED'
      && error.stage === 'preprocessing',
  )

  const prepared = await preprocessTripoSplatRgba(opaque, {
    canvasSize: 2,
    erodeRadius: 0,
    inputIsPrepared: true,
  })
  assert.deepEqual([...prepared.image.data], new Array(12).fill(255))
  assert.equal(prepared.usedBackgroundRemoval, false)

  let removerCalls = 0
  const segmented = await preprocessTripoSplatRgba(opaque, {
    canvasSize: 2,
    erodeRadius: 0,
    async removeBackground(resized, { signal } = {}) {
      removerCalls += 1
      assert.equal(signal, undefined)
      return {
        width: resized.width,
        height: resized.height,
        data: Uint8ClampedArray.from([
          255, 255, 255, 0,
          255, 255, 255, 255,
          255, 255, 255, 255,
          255, 255, 255, 0,
        ]),
      }
    },
  })
  assert.equal(removerCalls, 1)
  assert.equal(segmented.usedBackgroundRemoval, true)
  assert.ok(segmented.foreground.data.some((value, index) => index % 4 === 3 && value === 0))

  await assert.rejects(
    preprocessTripoSplatRgba(opaque, {
      canvasSize: 2,
      erodeRadius: 0,
      removeBackground: (resized) => ({
        width: resized.width + 1,
        height: resized.height,
        data: new Uint8ClampedArray((resized.width + 1) * resized.height * 4),
      }),
    }),
    /Background remover changed image size/,
  )
})

test('pure preprocessing honors a pre-aborted AbortSignal', async () => {
  const controller = new AbortController()
  controller.abort('test cancellation')
  await assert.rejects(
    preprocessTripoSplatRgba({
      width: 1,
      height: 1,
      data: Uint8ClampedArray.from([0, 0, 0, 0]),
    }, { signal: controller.signal }),
    (error) => error instanceof CancelledError,
  )
})
