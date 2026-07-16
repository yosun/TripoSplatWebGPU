import {
  BackgroundRemovalRequiredError,
  CancelledError,
} from '@ai3d/triposplat-webgpu'
import {
  ImageToSplatGenerator,
  type ImageToSplatResult,
} from './image-to-splat'

const imageInput = document.querySelector<HTMLInputElement>('#image')!
const generateButton = document.querySelector<HTMLButtonElement>('#generate')!
const cancelButton = document.querySelector<HTMLButtonElement>('#cancel')!
const plyButton = document.querySelector<HTMLButtonElement>('#download-ply')!
const splatButton = document.querySelector<HTMLButtonElement>('#download-splat')!
const status = document.querySelector<HTMLElement>('#status')!
const modelBaseUrl = document.querySelector<HTMLMetaElement>(
  'meta[name="triposplat-model-base-url"]',
)?.content

if (!modelBaseUrl) throw new Error('Set the triposplat-model-base-url meta tag.')

const generator = new ImageToSplatGenerator({
  modelBaseUrl,
  manifestUrl: 'manifest.json',
  executionProviders: ['webgpu'],
  cache: 'opfs',
})

let controller: AbortController | undefined
let result: ImageToSplatResult | undefined

function setResult(next?: ImageToSplatResult) {
  result = next
  plyButton.disabled = !next
  splatButton.disabled = !next
}

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.click()
  URL.revokeObjectURL(url)
}

const compatibility = await ImageToSplatGenerator.checkCompatibility()
if (!compatibility.supported) {
  status.textContent = compatibility.blockers.join('\n') || 'WebGPU is unavailable.'
  generateButton.disabled = true
} else {
  status.textContent = 'Ready. Select an alpha-bearing PNG or prepared image.'
}

cancelButton.addEventListener('click', () => controller?.abort())
plyButton.addEventListener('click', () => {
  if (result) download(result.ply, 'triposplat.ply')
})
splatButton.addEventListener('click', () => {
  if (result) download(result.splat, 'triposplat.splat')
})

generateButton.addEventListener('click', async () => {
  const file = imageInput.files?.[0]
  if (!file) {
    status.textContent = 'Choose an image first.'
    return
  }

  controller?.abort()
  controller = new AbortController()
  setResult()
  generateButton.disabled = true

  try {
    await generator.initialize({
      signal: controller.signal,
      onProgress: (event) => { status.textContent = event.message },
    })
    const next = await generator.generate(file, {
      steps: 4,
      gaussianCount: 262144,
      seed: 42,
      signal: controller.signal,
      onProgress: (event) => { status.textContent = event.message },
    })
    setResult(next)
    status.textContent = [
      `Generated ${next.count.toLocaleString()} Gaussians.`,
      `PLY: ${(next.ply.size / 1_048_576).toFixed(1)} MiB`,
      `.splat: ${(next.splat.size / 1_048_576).toFixed(1)} MiB`,
      `Elapsed: ${(next.elapsedMs / 1000).toFixed(1)} s`,
    ].join('\n')
  } catch (error) {
    if (error instanceof CancelledError) {
      status.textContent = 'Cancelled. The next run will create a clean worker.'
    } else if (error instanceof BackgroundRemovalRequiredError) {
      status.textContent = 'This opaque image needs a local background remover. Use an alpha-bearing PNG.'
    } else {
      status.textContent = error instanceof Error ? error.message : String(error)
    }
  } finally {
    generateButton.disabled = false
  }
})

window.addEventListener('pagehide', () => {
  controller?.abort()
  void generator.dispose()
}, { once: true })
