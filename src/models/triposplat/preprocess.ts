/** Browser-side image preparation matching TripoSplat's `preprocess_image`. */

export const TRIPOSPLAT_CANVAS_SIZE = 1024

export const DINOV3_IMAGE_MEAN = [0.485, 0.456, 0.406] as const
export const DINOV3_IMAGE_STD = [0.229, 0.224, 0.225] as const

export interface RgbaImage {
  width: number
  height: number
  /** Row-major, straight-alpha RGBA bytes. */
  data: Uint8ClampedArray
}

export interface RgbImage {
  width: number
  height: number
  /** Row-major RGB bytes. */
  data: Uint8ClampedArray
}

export interface NchwImageTensor {
  data: Float32Array
  dims: readonly [1, 3, number, number]
}

export interface TripoSplatEncoderTensors {
  /** `torchvision.transforms.ToTensor`: RGB in [0, 1]. */
  rgb: NchwImageTensor
  /** RGB normalized by ImageNet/DINOv3 mean and standard deviation. */
  dinov3: NchwImageTensor
  /** Flux VAE image input: RGB in [-1, 1]. */
  vae: NchwImageTensor
}

export type TripoSplatBackgroundRemover = (
  resizedImage: Readonly<RgbaImage>,
) => RgbaImage | Promise<RgbaImage>

export interface TripoSplatPreprocessOptions {
  canvasSize?: number
  erodeRadius?: number
  /**
   * Official TripoSplat invokes BiRefNet when every source alpha byte is 255.
   * Supply the browser BiRefNet adapter here; images with real alpha bypass it.
   */
  removeBackground?: TripoSplatBackgroundRemover
  /**
   * Explicit escape hatch for an image that is already the opaque, black RGB
   * composite returned by official preprocessing. Raw opaque photos should not
   * set this flag because doing so skips the required BiRefNet stage.
   */
  opaqueImageIsAlreadyPrepared?: boolean
}

export interface TripoSplatPreprocessResult {
  /** The RGB-on-black image consumed by both official encoders. */
  image: RgbImage
  /** Resized foreground immediately before its alpha is composited on black. */
  foreground: RgbaImage
  usedBackgroundRemoval: boolean
}

type Browser2dContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}.`)
  }
}

function assertRgbaImage(image: Readonly<RgbaImage>, label: string): void {
  assertPositiveInteger(image.width, `${label}.width`)
  assertPositiveInteger(image.height, `${label}.height`)
  const expected = image.width * image.height * 4
  if (image.data.length !== expected) {
    throw new Error(`${label}.data has ${image.data.length} bytes; expected ${expected}.`)
  }
}

function assertRgbImage(image: Readonly<RgbImage>, label: string): void {
  assertPositiveInteger(image.width, `${label}.width`)
  assertPositiveInteger(image.height, `${label}.height`)
  const expected = image.width * image.height * 3
  if (image.data.length !== expected) {
    throw new Error(`${label}.data has ${image.data.length} bytes; expected ${expected}.`)
  }
}

function makeCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height)
  }
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    return canvas
  }
  throw new Error('ImageBitmap preprocessing requires OffscreenCanvas or an HTML canvas.')
}

function get2dContext(canvas: OffscreenCanvas | HTMLCanvasElement): Browser2dContext {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context || !('getImageData' in context)) {
    throw new Error('Could not create a 2D context for TripoSplat preprocessing.')
  }
  return context
}

/** Extracts unpremultiplied, sRGB RGBA bytes without resizing the bitmap. */
export function imageBitmapToRgba(image: ImageBitmap): RgbaImage {
  assertPositiveInteger(image.width, 'image.width')
  assertPositiveInteger(image.height, 'image.height')
  const canvas = makeCanvas(image.width, image.height)
  const context = get2dContext(canvas)
  context.clearRect(0, 0, image.width, image.height)
  context.drawImage(image, 0, 0)
  const pixels = context.getImageData(0, 0, image.width, image.height)
  return {
    width: image.width,
    height: image.height,
    data: new Uint8ClampedArray(pixels.data),
  }
}

/** Python 3's `round` for non-negative values (ties to even). */
function pythonRound(value: number): number {
  const floor = Math.floor(value)
  const fraction = value - floor
  if (fraction < 0.5) return floor
  if (fraction > 0.5) return floor + 1
  return floor % 2 === 0 ? floor : floor + 1
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

/** Pillow's `MULDIV255` macro, used by its RGBA <-> RGBa conversion. */
function multiplyDivide255(left: number, right: number): number {
  const temporary = left * right + 128
  return ((temporary >> 8) + temporary) >> 8
}

/** Pillow's signed fixed-point `clip8` after a resampling accumulator. */
function clipResampleAccumulator(value: number): number {
  const shifted = Math.floor(value / PILLOW_COEFFICIENT_SCALE)
  return Math.max(0, Math.min(255, shifted))
}

function buildContributions(sourceSize: number, targetSize: number): ResampleContribution[] {
  // This follows Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc`
  // literally. The half-pixel convention and fixed-point rounding boundary
  // are observable at translucent object edges.
  const scale = sourceSize / targetSize
  const filterScale = Math.max(scale, 1)
  const support = 3 * filterScale
  const contributions: ResampleContribution[] = new Array(targetSize)

  for (let target = 0; target < targetSize; target += 1) {
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

/**
 * Deterministic, separable three-lobe Lanczos resize. Filtering occurs in
 * premultiplied-alpha space, as Pillow does for RGBA resampling, then converts
 * back to straight alpha.
 */
export function resizeRgbaLanczos(
  image: Readonly<RgbaImage>,
  targetWidth: number,
  targetHeight: number,
): RgbaImage {
  assertRgbaImage(image, 'image')
  assertPositiveInteger(targetWidth, 'targetWidth')
  assertPositiveInteger(targetHeight, 'targetHeight')
  if (targetWidth === image.width && targetHeight === image.height) {
    return { width: image.width, height: image.height, data: new Uint8ClampedArray(image.data) }
  }

  const horizontal = buildContributions(image.width, targetWidth)
  const vertical = buildContributions(image.height, targetHeight)
  // Pillow converts RGBA to its 8-bit premultiplied `RGBa` mode before the
  // two resampling passes. Keeping the intermediate byte-quantized matters at
  // translucent edges (and is substantially closer than float premultiplying).
  const intermediate = new Uint8ClampedArray(targetWidth * image.height * 4)

  for (let y = 0; y < image.height; y += 1) {
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
      // Pillow leaves RGBa color bytes untouched at both exact alpha limits.
      output[offset] = premultiplied[offset]
      output[offset + 1] = premultiplied[offset + 1]
      output[offset + 2] = premultiplied[offset + 2]
      continue
    }
    // Pillow's RGBa -> RGBA conversion truncates the unpremultiplied quotient.
    output[offset] = Math.min(255, Math.floor((premultiplied[offset] * 255) / alpha))
    output[offset + 1] = Math.min(255, Math.floor((premultiplied[offset + 1] * 255) / alpha))
    output[offset + 2] = Math.min(255, Math.floor((premultiplied[offset + 2] * 255) / alpha))
  }
  return { width: targetWidth, height: targetHeight, data: output }
}

export function hasRealAlpha(image: Readonly<RgbaImage>): boolean {
  assertRgbaImage(image, 'image')
  for (let index = 3; index < image.data.length; index += 4) {
    if (image.data[index] < 255) return true
  }
  return false
}

/** Pillow `MinFilter(2 * radius + 1)` semantics with clamped image edges. */
export function erodeAlpha(image: Readonly<RgbaImage>, radius: number): RgbaImage {
  assertRgbaImage(image, 'image')
  if (!Number.isInteger(radius) || radius < 0) {
    throw new Error(`erodeRadius must be a non-negative integer, got ${radius}.`)
  }
  const output = new Uint8ClampedArray(image.data)
  if (radius === 0) return { width: image.width, height: image.height, data: output }

  for (let y = 0; y < image.height; y += 1) {
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

export interface AlphaBounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
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

/** PIL-style crop: right/bottom are exclusive and out-of-image pixels are transparent black. */
export function cropRgba(
  image: Readonly<RgbaImage>,
  left: number,
  top: number,
  right: number,
  bottom: number,
): RgbaImage {
  assertRgbaImage(image, 'image')
  if (![left, top, right, bottom].every(Number.isInteger)) {
    throw new Error('Crop coordinates must be integers.')
  }
  const width = right - left
  const height = bottom - top
  assertPositiveInteger(width, 'crop width')
  assertPositiveInteger(height, 'crop height')
  const output = new Uint8ClampedArray(width * height * 4)
  const sourceLeft = Math.max(0, left)
  const sourceTop = Math.max(0, top)
  const sourceRight = Math.min(image.width, right)
  const sourceBottom = Math.min(image.height, bottom)
  for (let sourceY = sourceTop; sourceY < sourceBottom; sourceY += 1) {
    for (let sourceX = sourceLeft; sourceX < sourceRight; sourceX += 1) {
      const sourceOffset = (sourceY * image.width + sourceX) * 4
      const outputOffset = ((sourceY - top) * width + sourceX - left) * 4
      output.set(image.data.subarray(sourceOffset, sourceOffset + 4), outputOffset)
    }
  }
  return { width, height, data: output }
}

/** Equivalent to pasting the RGBA foreground onto an all-black Pillow RGB image. */
export function compositeRgbaOnBlack(image: Readonly<RgbaImage>): RgbImage {
  assertRgbaImage(image, 'image')
  const output = new Uint8ClampedArray(image.width * image.height * 3)
  let outputOffset = 0
  for (let sourceOffset = 0; sourceOffset < image.data.length; sourceOffset += 4) {
    const alpha = image.data[sourceOffset + 3]
    // Pillow's RGB paste uses the same fixed-point `DIV255` operation.
    output[outputOffset] = multiplyDivide255(image.data[sourceOffset], alpha)
    output[outputOffset + 1] = multiplyDivide255(image.data[sourceOffset + 1], alpha)
    output[outputOffset + 2] = multiplyDivide255(image.data[sourceOffset + 2], alpha)
    outputOffset += 3
  }
  return { width: image.width, height: image.height, data: output }
}

/** Pure-pixel entry point, useful in a worker and for PyTorch comparison fixtures. */
export async function preprocessTripoSplatRgba(
  source: Readonly<RgbaImage>,
  options: TripoSplatPreprocessOptions = {},
): Promise<TripoSplatPreprocessResult> {
  assertRgbaImage(source, 'source')
  const canvasSize = options.canvasSize ?? TRIPOSPLAT_CANVAS_SIZE
  const erodeRadius = options.erodeRadius ?? 1
  assertPositiveInteger(canvasSize, 'canvasSize')
  if (!Number.isInteger(erodeRadius) || erodeRadius < 0) {
    throw new Error(`erodeRadius must be a non-negative integer, got ${erodeRadius}.`)
  }

  const scale = canvasSize / Math.min(source.width, source.height)
  const resizedWidth = Math.max(1, pythonRound(source.width * scale))
  const resizedHeight = Math.max(1, pythonRound(source.height * scale))
  let foreground = resizeRgbaLanczos(source, resizedWidth, resizedHeight)
  let usedBackgroundRemoval = false

  if (!hasRealAlpha(foreground) && !options.opaqueImageIsAlreadyPrepared) {
    if (!options.removeBackground) {
      throw new Error(
        'Opaque TripoSplat input requires a BiRefNet background remover. ' +
          'Only set opaqueImageIsAlreadyPrepared for an official RGB-on-black preprocessing result.',
      )
    }
    const removed = await options.removeBackground(foreground)
    assertRgbaImage(removed, 'removeBackground result')
    if (removed.width !== foreground.width || removed.height !== foreground.height) {
      throw new Error(
        `Background remover changed image size from ${foreground.width}x${foreground.height} ` +
          `to ${removed.width}x${removed.height}.`,
      )
    }
    foreground = {
      width: removed.width,
      height: removed.height,
      data: new Uint8ClampedArray(removed.data),
    }
    usedBackgroundRemoval = true
  }

  if (options.opaqueImageIsAlreadyPrepared && !hasRealAlpha(foreground)) {
    if (foreground.width !== canvasSize || foreground.height !== canvasSize) {
      throw new Error(
        `An already-prepared opaque input must be ${canvasSize}x${canvasSize}; got ` +
          `${foreground.width}x${foreground.height}.`,
      )
    }
    return {
      image: compositeRgbaOnBlack(foreground),
      foreground,
      usedBackgroundRemoval,
    }
  }

  foreground = erodeAlpha(foreground, erodeRadius)
  const bounds = findNonZeroAlphaBounds(foreground)
  if (!bounds) {
    throw new Error('TripoSplat alpha matte is empty after erosion.')
  }

  const centerX = (bounds.minX + bounds.maxX) / 2
  const centerY = (bounds.minY + bounds.maxY) / 2
  const half = (Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) / 2) * 1.2
  const left = Math.trunc(centerX - half)
  const top = Math.trunc(centerY - half)
  const right = Math.trunc(centerX + half)
  const bottom = Math.trunc(centerY + half)
  if (right <= left || bottom <= top) {
    throw new Error('TripoSplat alpha matte is too small to form the official square crop.')
  }

  foreground = cropRgba(foreground, left, top, right, bottom)
  foreground = resizeRgbaLanczos(foreground, canvasSize, canvasSize)
  return {
    image: compositeRgbaOnBlack(foreground),
    foreground,
    usedBackgroundRemoval,
  }
}

export function preprocessTripoSplatImage(
  image: ImageBitmap,
  options: TripoSplatPreprocessOptions = {},
): Promise<TripoSplatPreprocessResult> {
  return preprocessTripoSplatRgba(imageBitmapToRgba(image), options)
}

function buildNchwTensor(
  image: Readonly<RgbImage>,
  transform: (value: number, channel: number) => number,
): NchwImageTensor {
  assertRgbImage(image, 'image')
  const planeSize = image.width * image.height
  const tensor = new Float32Array(planeSize * 3)
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    const sourceOffset = pixel * 3
    // `ToTensor` performs the uint8 division into a float32 tensor before any
    // subsequent normalization, so preserve that rounding boundary.
    tensor[pixel] = transform(Math.fround(image.data[sourceOffset] / 255), 0)
    tensor[planeSize + pixel] = transform(Math.fround(image.data[sourceOffset + 1] / 255), 1)
    tensor[planeSize * 2 + pixel] = transform(Math.fround(image.data[sourceOffset + 2] / 255), 2)
  }
  return { data: tensor, dims: [1, 3, image.height, image.width] }
}

export function buildRgb01Tensor(image: Readonly<RgbImage>): NchwImageTensor {
  return buildNchwTensor(image, (value) => value)
}

export function buildDinov3Tensor(image: Readonly<RgbImage>): NchwImageTensor {
  return buildNchwTensor(
    image,
    (value, channel) =>
      Math.fround(
        Math.fround(value - Math.fround(DINOV3_IMAGE_MEAN[channel])) /
          Math.fround(DINOV3_IMAGE_STD[channel]),
      ),
  )
}

export function buildFluxVaeTensor(image: Readonly<RgbImage>): NchwImageTensor {
  return buildNchwTensor(image, (value) => Math.fround(Math.fround(value * 2) - 1))
}

export function buildTripoSplatEncoderTensors(
  image: Readonly<RgbImage>,
): TripoSplatEncoderTensors {
  return {
    rgb: buildRgb01Tensor(image),
    dinov3: buildDinov3Tensor(image),
    vae: buildFluxVaeTensor(image),
  }
}
