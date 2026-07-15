import { BackgroundRemovalRequiredError, throwIfAborted } from './errors.js'

export const TRIPOSPLAT_CANVAS_SIZE = 1024
export const DINOV3_IMAGE_MEAN = [0.485, 0.456, 0.406] as const
export const DINOV3_IMAGE_STD = [0.229, 0.224, 0.225] as const

export interface RgbaImage {
  width: number
  height: number
  /** Row-major, straight-alpha sRGB bytes. */
  data: Uint8ClampedArray
}

export interface RgbImage {
  width: number
  height: number
  /** Row-major sRGB bytes. */
  data: Uint8ClampedArray
}

export interface NchwImageTensor {
  data: Float32Array
  dims: readonly [1, 3, number, number]
}

export interface TripoSplatEncoderTensors {
  /** Torchvision `ToTensor`: RGB in [0, 1]. */
  rgb: NchwImageTensor
  /** RGB normalized by the official DINOv3/ImageNet mean and standard deviation. */
  dinov3: NchwImageTensor
  /** Flux VAE input: RGB in [-1, 1]. */
  vae: NchwImageTensor
}

export type TripoSplatCanvas = OffscreenCanvas | HTMLCanvasElement

export type TripoSplatImageSource =
  | Blob
  | File
  | ImageBitmap
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas

export type TripoSplatBackgroundRemover = (
  resizedImage: Readonly<RgbaImage>,
  options?: { signal?: AbortSignal },
) => RgbaImage | Promise<RgbaImage>

export interface TripoSplatPreprocessOptions {
  signal?: AbortSignal
  canvasSize?: number
  erodeRadius?: number
  /** Browser-local BiRefNet-compatible stage used only when every source alpha byte is opaque. */
  removeBackground?: TripoSplatBackgroundRemover
  /**
   * Use only for the exact opaque RGB-on-black result of official preprocessing.
   * Raw opaque images require external segmentation and reject without this flag.
   */
  inputIsPrepared?: boolean
}

export interface NormalizeTripoSplatImageOptions extends TripoSplatPreprocessOptions {
  /** Create a final ImageBitmap in addition to the canvas when the API exists. Defaults to true. */
  includeImageBitmap?: boolean
}

export interface TripoSplatPreprocessResult {
  /** Exact RGB-on-black image consumed by both encoders. */
  image: RgbImage
  /** Foreground RGBA image immediately before black compositing. */
  foreground: RgbaImage
  /** True only when the configured browser-local remover was invoked. */
  usedBackgroundRemoval: boolean
}

export interface NormalizedTripoSplatImage extends TripoSplatPreprocessResult {
  tensors: TripoSplatEncoderTensors
  /** Opaque RGB-on-black browser canvas at the graph input resolution. */
  canvas: TripoSplatCanvas
  /** Optional canvas snapshot owned by this result. */
  imageBitmap?: ImageBitmap
  /** Closes the owned ImageBitmap. Pixel arrays and canvas remain caller-owned. */
  dispose(): void
}

export interface TripoSplatResizeGeometry {
  width: number
  height: number
  scale: number
}

export interface AlphaBounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export interface TripoSplatCropGeometry {
  left: number
  top: number
  right: number
  bottom: number
}

type Browser2dContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer, got ${value}.`)
  }
}

function assertRgbaImage(image: Readonly<RgbaImage>, label: string): void {
  assertPositiveInteger(image.width, `${label}.width`)
  assertPositiveInteger(image.height, `${label}.height`)
  const expected = image.width * image.height * 4
  if (!(image.data instanceof Uint8ClampedArray) || image.data.length !== expected) {
    throw new RangeError(`${label}.data must contain ${expected} RGBA bytes.`)
  }
}

function assertRgbImage(image: Readonly<RgbImage>, label: string): void {
  assertPositiveInteger(image.width, `${label}.width`)
  assertPositiveInteger(image.height, `${label}.height`)
  const expected = image.width * image.height * 3
  if (!(image.data instanceof Uint8ClampedArray) || image.data.length !== expected) {
    throw new RangeError(`${label}.data must contain ${expected} RGB bytes.`)
  }
}

function makeCanvas(width: number, height: number): TripoSplatCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height)
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    return canvas
  }
  throw new Error('Image conversion requires OffscreenCanvas or an HTML canvas document.')
}

function get2dContext(canvas: TripoSplatCanvas): Browser2dContext {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context || !('getImageData' in context)) {
    throw new Error('Could not create a readable 2D canvas context.')
  }
  return context
}

function isImageBitmap(value: unknown): value is ImageBitmap {
  return typeof ImageBitmap !== 'undefined' && value instanceof ImageBitmap
}

function isImageData(value: unknown): value is ImageData {
  return typeof ImageData !== 'undefined' && value instanceof ImageData
}

function isHtmlImage(value: unknown): value is HTMLImageElement {
  return typeof HTMLImageElement !== 'undefined' && value instanceof HTMLImageElement
}

function isHtmlCanvas(value: unknown): value is HTMLCanvasElement {
  return typeof HTMLCanvasElement !== 'undefined' && value instanceof HTMLCanvasElement
}

function isOffscreenCanvas(value: unknown): value is OffscreenCanvas {
  return typeof OffscreenCanvas !== 'undefined' && value instanceof OffscreenCanvas
}

async function waitForHtmlImage(image: HTMLImageElement, signal?: AbortSignal): Promise<void> {
  throwIfAborted(signal)
  if (image.complete && image.naturalWidth > 0 && image.naturalHeight > 0) return
  if (typeof image.decode === 'function') {
    await image.decode()
    throwIfAborted(signal)
    return
  }
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      image.removeEventListener('load', onLoad)
      image.removeEventListener('error', onError)
      signal?.removeEventListener('abort', onAbort)
    }
    const onLoad = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error('HTMLImageElement could not be decoded.'))
    }
    const onAbort = () => {
      cleanup()
      try {
        throwIfAborted(signal)
      } catch (error) {
        reject(error)
      }
    }
    image.addEventListener('load', onLoad, { once: true })
    image.addEventListener('error', onError, { once: true })
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

function drawableDimensions(source: CanvasImageSource): { width: number; height: number } {
  if (isHtmlImage(source)) {
    return { width: source.naturalWidth, height: source.naturalHeight }
  }
  if (isImageBitmap(source) || isHtmlCanvas(source) || isOffscreenCanvas(source)) {
    return { width: source.width, height: source.height }
  }
  throw new TypeError('Unsupported drawable image source.')
}

function drawableToRgba(source: CanvasImageSource): RgbaImage {
  const { width, height } = drawableDimensions(source)
  assertPositiveInteger(width, 'image.width')
  assertPositiveInteger(height, 'image.height')
  const canvas = makeCanvas(width, height)
  const context = get2dContext(canvas)
  context.clearRect(0, 0, width, height)
  context.drawImage(source, 0, 0, width, height)
  let pixels: ImageData
  try {
    pixels = context.getImageData(0, 0, width, height)
  } catch (cause) {
    throw new Error(
      'Could not read image pixels. Cross-origin HTML images require CORS permission before drawing.',
      { cause },
    )
  }
  return { width, height, data: new Uint8ClampedArray(pixels.data) }
}

async function decodeBlob(blob: Blob, signal?: AbortSignal): Promise<RgbaImage> {
  throwIfAborted(signal)
  if (typeof createImageBitmap !== 'function') {
    throw new Error('Blob/File image decoding requires createImageBitmap in this browser or worker.')
  }
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' })
  } catch {
    throwIfAborted(signal)
    bitmap = await createImageBitmap(blob)
  }
  try {
    throwIfAborted(signal)
    return drawableToRgba(bitmap)
  } finally {
    bitmap.close()
  }
}

/** Decode a browser image source to owned, straight-alpha RGBA bytes without resizing. */
export async function decodeTripoSplatImageSource(
  source: TripoSplatImageSource,
  options: { signal?: AbortSignal } = {},
): Promise<RgbaImage> {
  throwIfAborted(options.signal)
  if (typeof Blob !== 'undefined' && source instanceof Blob) {
    return decodeBlob(source, options.signal)
  }
  if (isImageData(source)) {
    assertPositiveInteger(source.width, 'image.width')
    assertPositiveInteger(source.height, 'image.height')
    return { width: source.width, height: source.height, data: new Uint8ClampedArray(source.data) }
  }
  if (isHtmlImage(source)) {
    await waitForHtmlImage(source, options.signal)
  }
  if (isImageBitmap(source) || isHtmlImage(source) || isHtmlCanvas(source) || isOffscreenCanvas(source)) {
    throwIfAborted(options.signal)
    return drawableToRgba(source)
  }
  throw new TypeError(
    'Unsupported image input. Expected Blob/File, ImageBitmap, ImageData, HTMLImageElement, HTMLCanvasElement, or OffscreenCanvas.',
  )
}

/** Python 3 `round` for non-negative dimensions (ties to even). */
function pythonRound(value: number): number {
  const floor = Math.floor(value)
  const fraction = value - floor
  if (fraction < 0.5) return floor
  if (fraction > 0.5) return floor + 1
  return floor % 2 === 0 ? floor : floor + 1
}

/** Official first resize: make the shorter side equal to the model canvas. */
export function calculateTripoSplatResize(
  width: number,
  height: number,
  canvasSize = TRIPOSPLAT_CANVAS_SIZE,
): TripoSplatResizeGeometry {
  assertPositiveInteger(width, 'width')
  assertPositiveInteger(height, 'height')
  assertPositiveInteger(canvasSize, 'canvasSize')
  const scale = canvasSize / Math.min(width, height)
  return {
    width: Math.max(1, pythonRound(width * scale)),
    height: Math.max(1, pythonRound(height * scale)),
    scale,
  }
}

function sinc(value: number): number {
  if (value === 0) return 1
  const angle = Math.PI * value
  return Math.sin(angle) / angle
}

function lanczos(value: number): number {
  return value >= -3 && value < 3 ? sinc(value) * sinc(value / 3) : 0
}

interface ResampleContribution {
  first: number
  coefficients: Int32Array
}

const PILLOW_PRECISION_BITS = 22
const PILLOW_COEFFICIENT_SCALE = 2 ** PILLOW_PRECISION_BITS
const PILLOW_ROUNDING_BIAS = 2 ** (PILLOW_PRECISION_BITS - 1)

function multiplyDivide255(left: number, right: number): number {
  const temporary = left * right + 128
  return ((temporary >> 8) + temporary) >> 8
}

function clipResampleAccumulator(value: number): number {
  return Math.max(0, Math.min(255, Math.floor(value / PILLOW_COEFFICIENT_SCALE)))
}

function buildContributions(
  sourceSize: number,
  targetSize: number,
  signal?: AbortSignal,
): ResampleContribution[] {
  const scale = sourceSize / targetSize
  const filterScale = Math.max(scale, 1)
  const support = 3 * filterScale
  const contributions = new Array<ResampleContribution>(targetSize)
  for (let target = 0; target < targetSize; target += 1) {
    if ((target & 63) === 0) throwIfAborted(signal)
    const center = (target + 0.5) * scale
    const first = Math.max(0, Math.trunc(center - support + 0.5))
    const end = Math.min(sourceSize, Math.trunc(center + support + 0.5))
    const weights = new Float64Array(end - first)
    let total = 0
    for (let index = 0; index < weights.length; index += 1) {
      const weight = lanczos((index + first - center + 0.5) / filterScale)
      weights[index] = weight
      total += weight
    }
    const coefficients = new Int32Array(weights.length)
    for (let index = 0; index < weights.length; index += 1) {
      const normalized = total === 0 ? weights[index] : weights[index] / total
      coefficients[index] = normalized < 0
        ? Math.trunc(-0.5 + normalized * PILLOW_COEFFICIENT_SCALE)
        : Math.trunc(0.5 + normalized * PILLOW_COEFFICIENT_SCALE)
    }
    contributions[target] = { first, coefficients }
  }
  return contributions
}

/** Pillow-compatible, premultiplied-alpha three-lobe Lanczos resize. */
export function resizeRgbaLanczos(
  image: Readonly<RgbaImage>,
  targetWidth: number,
  targetHeight: number,
  signal?: AbortSignal,
): RgbaImage {
  assertRgbaImage(image, 'image')
  assertPositiveInteger(targetWidth, 'targetWidth')
  assertPositiveInteger(targetHeight, 'targetHeight')
  throwIfAborted(signal)
  if (targetWidth === image.width && targetHeight === image.height) {
    return { width: image.width, height: image.height, data: new Uint8ClampedArray(image.data) }
  }
  const horizontal = buildContributions(image.width, targetWidth, signal)
  const vertical = buildContributions(image.height, targetHeight, signal)
  const intermediate = new Uint8ClampedArray(targetWidth * image.height * 4)
  for (let y = 0; y < image.height; y += 1) {
    if ((y & 15) === 0) throwIfAborted(signal)
    for (let x = 0; x < targetWidth; x += 1) {
      const contribution = horizontal[x]
      const outputOffset = (y * targetWidth + x) * 4
      let red = PILLOW_ROUNDING_BIAS
      let green = PILLOW_ROUNDING_BIAS
      let blue = PILLOW_ROUNDING_BIAS
      let alpha = PILLOW_ROUNDING_BIAS
      for (let index = 0; index < contribution.coefficients.length; index += 1) {
        const sourceOffset = (y * image.width + contribution.first + index) * 4
        const coefficient = contribution.coefficients[index]
        const sourceAlpha = image.data[sourceOffset + 3]
        red += multiplyDivide255(image.data[sourceOffset], sourceAlpha) * coefficient
        green += multiplyDivide255(image.data[sourceOffset + 1], sourceAlpha) * coefficient
        blue += multiplyDivide255(image.data[sourceOffset + 2], sourceAlpha) * coefficient
        alpha += sourceAlpha * coefficient
      }
      intermediate[outputOffset] = clipResampleAccumulator(red)
      intermediate[outputOffset + 1] = clipResampleAccumulator(green)
      intermediate[outputOffset + 2] = clipResampleAccumulator(blue)
      intermediate[outputOffset + 3] = clipResampleAccumulator(alpha)
    }
  }
  const premultiplied = new Uint8ClampedArray(targetWidth * targetHeight * 4)
  for (let y = 0; y < targetHeight; y += 1) {
    if ((y & 15) === 0) throwIfAborted(signal)
    const contribution = vertical[y]
    for (let x = 0; x < targetWidth; x += 1) {
      let red = PILLOW_ROUNDING_BIAS
      let green = PILLOW_ROUNDING_BIAS
      let blue = PILLOW_ROUNDING_BIAS
      let alpha = PILLOW_ROUNDING_BIAS
      for (let index = 0; index < contribution.coefficients.length; index += 1) {
        const sourceOffset = ((contribution.first + index) * targetWidth + x) * 4
        const coefficient = contribution.coefficients[index]
        red += intermediate[sourceOffset] * coefficient
        green += intermediate[sourceOffset + 1] * coefficient
        blue += intermediate[sourceOffset + 2] * coefficient
        alpha += intermediate[sourceOffset + 3] * coefficient
      }
      const outputOffset = (y * targetWidth + x) * 4
      premultiplied[outputOffset] = clipResampleAccumulator(red)
      premultiplied[outputOffset + 1] = clipResampleAccumulator(green)
      premultiplied[outputOffset + 2] = clipResampleAccumulator(blue)
      premultiplied[outputOffset + 3] = clipResampleAccumulator(alpha)
    }
  }
  const output = new Uint8ClampedArray(premultiplied.length)
  for (let offset = 0; offset < premultiplied.length; offset += 4) {
    const alpha = premultiplied[offset + 3]
    output[offset + 3] = alpha
    if (alpha === 0 || alpha === 255) {
      output[offset] = premultiplied[offset]
      output[offset + 1] = premultiplied[offset + 1]
      output[offset + 2] = premultiplied[offset + 2]
    } else {
      output[offset] = Math.min(255, Math.floor((premultiplied[offset] * 255) / alpha))
      output[offset + 1] = Math.min(255, Math.floor((premultiplied[offset + 1] * 255) / alpha))
      output[offset + 2] = Math.min(255, Math.floor((premultiplied[offset + 2] * 255) / alpha))
    }
  }
  throwIfAborted(signal)
  return { width: targetWidth, height: targetHeight, data: output }
}

export function hasRealAlpha(image: Readonly<RgbaImage>): boolean {
  assertRgbaImage(image, 'image')
  for (let index = 3; index < image.data.length; index += 4) {
    if (image.data[index] < 255) return true
  }
  return false
}

/** Pillow `MinFilter(2 * radius + 1)` semantics with clamped edges. */
export function erodeAlpha(
  image: Readonly<RgbaImage>,
  radius: number,
  signal?: AbortSignal,
): RgbaImage {
  assertRgbaImage(image, 'image')
  if (!Number.isInteger(radius) || radius < 0) {
    throw new RangeError(`erodeRadius must be a non-negative integer, got ${radius}.`)
  }
  const output = new Uint8ClampedArray(image.data)
  if (radius === 0) return { width: image.width, height: image.height, data: output }
  for (let y = 0; y < image.height; y += 1) {
    if ((y & 15) === 0) throwIfAborted(signal)
    const minY = Math.max(0, y - radius)
    const maxY = Math.min(image.height - 1, y + radius)
    for (let x = 0; x < image.width; x += 1) {
      const minX = Math.max(0, x - radius)
      const maxX = Math.min(image.width - 1, x + radius)
      let minimum = 255
      for (let sourceY = minY; sourceY <= maxY && minimum > 0; sourceY += 1) {
        for (let sourceX = minX; sourceX <= maxX; sourceX += 1) {
          minimum = Math.min(minimum, image.data[(sourceY * image.width + sourceX) * 4 + 3])
          if (minimum === 0) break
        }
      }
      output[(y * image.width + x) * 4 + 3] = minimum
    }
  }
  return { width: image.width, height: image.height, data: output }
}

export function findNonZeroAlphaBounds(image: Readonly<RgbaImage>): AlphaBounds | null {
  assertRgbaImage(image, 'image')
  let minX = image.width
  let minY = image.height
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      if (image.data[(y * image.width + x) * 4 + 3] === 0) continue
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
    }
  }
  return maxX < 0 ? null : { minX, minY, maxX, maxY }
}

export function calculateTripoSplatCrop(
  bounds: Readonly<AlphaBounds>,
  padding = 1.2,
): TripoSplatCropGeometry {
  if (![bounds.minX, bounds.minY, bounds.maxX, bounds.maxY, padding].every(Number.isFinite)) {
    throw new TypeError('Crop bounds and padding must be finite.')
  }
  if (bounds.maxX < bounds.minX || bounds.maxY < bounds.minY || padding <= 0) {
    throw new RangeError('Crop bounds must be ordered and padding must be positive.')
  }
  const centerX = (bounds.minX + bounds.maxX) / 2
  const centerY = (bounds.minY + bounds.maxY) / 2
  const half = (Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) / 2) * padding
  const crop = {
    left: Math.trunc(centerX - half),
    top: Math.trunc(centerY - half),
    right: Math.trunc(centerX + half),
    bottom: Math.trunc(centerY + half),
  }
  if (crop.right <= crop.left || crop.bottom <= crop.top) {
    throw new RangeError('Alpha matte is too small to form the official square crop.')
  }
  return crop
}

/** PIL-style crop; right/bottom are exclusive and out-of-image pixels are transparent black. */
export function cropRgba(
  image: Readonly<RgbaImage>,
  crop: Readonly<TripoSplatCropGeometry>,
): RgbaImage {
  assertRgbaImage(image, 'image')
  if (![crop.left, crop.top, crop.right, crop.bottom].every(Number.isInteger)) {
    throw new TypeError('Crop coordinates must be integers.')
  }
  const width = crop.right - crop.left
  const height = crop.bottom - crop.top
  assertPositiveInteger(width, 'crop width')
  assertPositiveInteger(height, 'crop height')
  const output = new Uint8ClampedArray(width * height * 4)
  const sourceLeft = Math.max(0, crop.left)
  const sourceTop = Math.max(0, crop.top)
  const sourceRight = Math.min(image.width, crop.right)
  const sourceBottom = Math.min(image.height, crop.bottom)
  for (let sourceY = sourceTop; sourceY < sourceBottom; sourceY += 1) {
    for (let sourceX = sourceLeft; sourceX < sourceRight; sourceX += 1) {
      const sourceOffset = (sourceY * image.width + sourceX) * 4
      const outputOffset = ((sourceY - crop.top) * width + sourceX - crop.left) * 4
      output.set(image.data.subarray(sourceOffset, sourceOffset + 4), outputOffset)
    }
  }
  return { width, height, data: output }
}

/** Paste straight-alpha RGBA onto the official all-black RGB background. */
export function compositeRgbaOnBlack(image: Readonly<RgbaImage>): RgbImage {
  assertRgbaImage(image, 'image')
  const output = new Uint8ClampedArray(image.width * image.height * 3)
  let outputOffset = 0
  for (let sourceOffset = 0; sourceOffset < image.data.length; sourceOffset += 4) {
    const alpha = image.data[sourceOffset + 3]
    output[outputOffset] = multiplyDivide255(image.data[sourceOffset], alpha)
    output[outputOffset + 1] = multiplyDivide255(image.data[sourceOffset + 1], alpha)
    output[outputOffset + 2] = multiplyDivide255(image.data[sourceOffset + 2], alpha)
    outputOffset += 3
  }
  return { width: image.width, height: image.height, data: output }
}

/** Pure-pixel official preprocessing with an injectable browser-local segmentation boundary. */
export async function preprocessTripoSplatRgba(
  source: Readonly<RgbaImage>,
  options: TripoSplatPreprocessOptions = {},
): Promise<TripoSplatPreprocessResult> {
  assertRgbaImage(source, 'source')
  throwIfAborted(options.signal)
  const canvasSize = options.canvasSize ?? TRIPOSPLAT_CANVAS_SIZE
  const erodeRadius = options.erodeRadius ?? 1
  assertPositiveInteger(canvasSize, 'canvasSize')
  if (!Number.isInteger(erodeRadius) || erodeRadius < 0) {
    throw new RangeError(`erodeRadius must be a non-negative integer, got ${erodeRadius}.`)
  }
  const geometry = calculateTripoSplatResize(source.width, source.height, canvasSize)
  let foreground = resizeRgbaLanczos(source, geometry.width, geometry.height, options.signal)
  let usedBackgroundRemoval = false
  if (!hasRealAlpha(foreground) && !options.inputIsPrepared) {
    if (!options.removeBackground) {
      throw new BackgroundRemovalRequiredError(undefined, {
        diagnostics: { width: source.width, height: source.height },
      })
    }
    const removed = await options.removeBackground(
      foreground,
      options.signal === undefined ? {} : { signal: options.signal },
    )
    throwIfAborted(options.signal)
    assertRgbaImage(removed, 'removeBackground result')
    if (removed.width !== foreground.width || removed.height !== foreground.height) {
      throw new RangeError(
        `Background remover changed image size from ${foreground.width}x${foreground.height} `
        + `to ${removed.width}x${removed.height}.`,
      )
    }
    foreground = {
      width: removed.width,
      height: removed.height,
      data: new Uint8ClampedArray(removed.data),
    }
    usedBackgroundRemoval = true
  }
  if (options.inputIsPrepared && !hasRealAlpha(foreground)) {
    if (foreground.width !== canvasSize || foreground.height !== canvasSize) {
      throw new RangeError(
        `An already-prepared opaque input must be ${canvasSize}x${canvasSize}; got `
        + `${foreground.width}x${foreground.height}.`,
      )
    }
    return { image: compositeRgbaOnBlack(foreground), foreground, usedBackgroundRemoval }
  }
  foreground = erodeAlpha(foreground, erodeRadius, options.signal)
  const bounds = findNonZeroAlphaBounds(foreground)
  if (!bounds) throw new RangeError('TripoSplat alpha matte is empty after erosion.')
  const crop = calculateTripoSplatCrop(bounds)
  foreground = cropRgba(foreground, crop)
  foreground = resizeRgbaLanczos(foreground, canvasSize, canvasSize, options.signal)
  throwIfAborted(options.signal)
  return { image: compositeRgbaOnBlack(foreground), foreground, usedBackgroundRemoval }
}

function buildNchwTensor(
  image: Readonly<RgbImage>,
  transform: (value: number, channel: number) => number,
  signal?: AbortSignal,
): NchwImageTensor {
  assertRgbImage(image, 'image')
  const planeSize = image.width * image.height
  const tensor = new Float32Array(planeSize * 3)
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    if ((pixel & 16383) === 0) throwIfAborted(signal)
    const sourceOffset = pixel * 3
    tensor[pixel] = transform(Math.fround(image.data[sourceOffset] / 255), 0)
    tensor[planeSize + pixel] = transform(Math.fround(image.data[sourceOffset + 1] / 255), 1)
    tensor[planeSize * 2 + pixel] = transform(Math.fround(image.data[sourceOffset + 2] / 255), 2)
  }
  return { data: tensor, dims: [1, 3, image.height, image.width] }
}

export function buildRgb01Tensor(image: Readonly<RgbImage>, signal?: AbortSignal): NchwImageTensor {
  return buildNchwTensor(image, (value) => value, signal)
}

export function buildDinov3Tensor(image: Readonly<RgbImage>, signal?: AbortSignal): NchwImageTensor {
  return buildNchwTensor(
    image,
    (value, channel) => Math.fround(
      Math.fround(value - Math.fround(DINOV3_IMAGE_MEAN[channel]))
      / Math.fround(DINOV3_IMAGE_STD[channel]),
    ),
    signal,
  )
}

export function buildFluxVaeTensor(image: Readonly<RgbImage>, signal?: AbortSignal): NchwImageTensor {
  return buildNchwTensor(
    image,
    (value) => Math.fround(Math.fround(value * 2) - 1),
    signal,
  )
}

export function buildTripoSplatEncoderTensors(
  image: Readonly<RgbImage>,
  signal?: AbortSignal,
): TripoSplatEncoderTensors {
  return {
    rgb: buildRgb01Tensor(image, signal),
    dinov3: buildDinov3Tensor(image, signal),
    vae: buildFluxVaeTensor(image, signal),
  }
}

/** Materialize the final opaque RGB image as an OffscreenCanvas, or HTML canvas fallback. */
export function rgbImageToCanvas(image: Readonly<RgbImage>): TripoSplatCanvas {
  assertRgbImage(image, 'image')
  const canvas = makeCanvas(image.width, image.height)
  const context = get2dContext(canvas)
  const pixels = context.createImageData(image.width, image.height)
  for (let pixel = 0; pixel < image.width * image.height; pixel += 1) {
    const source = pixel * 3
    const target = pixel * 4
    pixels.data[target] = image.data[source]
    pixels.data[target + 1] = image.data[source + 1]
    pixels.data[target + 2] = image.data[source + 2]
    pixels.data[target + 3] = 255
  }
  context.putImageData(pixels, 0, 0)
  return canvas
}

async function canvasToImageBitmap(
  canvas: TripoSplatCanvas,
  signal?: AbortSignal,
): Promise<ImageBitmap | undefined> {
  throwIfAborted(signal)
  if (typeof createImageBitmap !== 'function') return undefined
  const bitmap = await createImageBitmap(canvas)
  if (signal?.aborted) {
    bitmap.close()
    throwIfAborted(signal)
  }
  return bitmap
}

export async function rgbImageToImageBitmap(
  image: Readonly<RgbImage>,
  options: { signal?: AbortSignal } = {},
): Promise<ImageBitmap> {
  const bitmap = await canvasToImageBitmap(rgbImageToCanvas(image), options.signal)
  if (!bitmap) throw new Error('createImageBitmap is unavailable in this browser context.')
  return bitmap
}

/** Decode, preprocess, tensorize, and materialize one TripoSplat browser image input. */
export async function normalizeTripoSplatImageInput(
  source: TripoSplatImageSource,
  options: NormalizeTripoSplatImageOptions = {},
): Promise<NormalizedTripoSplatImage> {
  const decoded = await decodeTripoSplatImageSource(
    source,
    options.signal === undefined ? {} : { signal: options.signal },
  )
  const prepared = await preprocessTripoSplatRgba(decoded, options)
  throwIfAborted(options.signal)
  const tensors = buildTripoSplatEncoderTensors(prepared.image, options.signal)
  const canvas = rgbImageToCanvas(prepared.image)
  const imageBitmap = options.includeImageBitmap === false
    ? undefined
    : await canvasToImageBitmap(canvas, options.signal)
  let disposed = false
  return {
    ...prepared,
    tensors,
    canvas,
    ...(imageBitmap === undefined ? {} : { imageBitmap }),
    dispose() {
      if (disposed) return
      disposed = true
      imageBitmap?.close()
    },
  }
}
