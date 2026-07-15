// Map this vendor path to the packed workspace package in the host application.
import { CancelledError, TripoSplatWebGPU } from '/vendor/@ai3d/triposplat-webgpu/index.js'

const input = document.querySelector('#image')
const generateButton = document.querySelector('#generate')
const cancelButton = document.querySelector('#cancel')
const status = document.querySelector('#status')

const model = new TripoSplatWebGPU({
  modelBaseUrl: 'https://models.example.com/triposplat-webgpu/v1/',
})

let activeController
let activeScene

const compatibility = await TripoSplatWebGPU.checkCompatibility()
if (!compatibility.supported) {
  status.textContent = compatibility.blockers.join('\n') || 'WebGPU is unsupported.'
  generateButton.disabled = true
} else {
  await model.load({ onProgress: (event) => { status.textContent = event.message } })
  status.textContent = 'Model loaded'
}

cancelButton.addEventListener('click', () => activeController?.abort())

generateButton.addEventListener('click', async () => {
  const file = input.files?.[0]
  if (!file) return

  activeController?.abort()
  activeScene?.dispose()
  activeController = new AbortController()

  try {
    activeScene = await model.generate(file, {
      steps: 20,
      guidanceScale: 3,
      gaussianCount: 262144,
      seed: 42,
      signal: activeController.signal,
      onProgress: (event) => { status.textContent = event.message },
    })
    status.textContent = `Generated ${activeScene.count.toLocaleString()} Gaussians locally.`
  } catch (error) {
    status.textContent = error instanceof CancelledError ? 'Cancelled' : String(error)
  }
})

window.addEventListener('pagehide', () => {
  activeController?.abort()
  activeScene?.dispose()
  void model.dispose()
})
