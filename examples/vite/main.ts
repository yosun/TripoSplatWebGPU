/** Alpha package example; browser inference still carries the documented release gates. */
import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'

const model = new TripoSplatWebGPU({
  modelBaseUrl: import.meta.env.VITE_TRIPOSPLAT_MODEL_BASE_URL,
  manifestUrl: 'manifest.json',
  cache: 'opfs',
  executionProviders: ['webgpu'],
})

const report = await TripoSplatWebGPU.checkCompatibility()
if (!report.supported) {
  throw new Error(report.blockers.join('\n') || 'This browser cannot run TripoSplat WebGPU.')
}

await model.load({
  onProgress: ({ message, loadedBytes, totalBytes }) => {
    console.info(message, { loadedBytes, totalBytes })
  },
})

const input = document.querySelector<HTMLInputElement>('#image')
const file = input?.files?.[0]
if (file) {
  const scene = await model.generate(file, {
    steps: 20,
    guidanceScale: 3,
    gaussianCount: 262144,
    seed: 42,
    onProgress: ({ stage, invocation, totalInvocations }) => {
      console.info(stage, { invocation, totalInvocations })
    },
  })
  console.info(scene.metadata)
  scene.dispose()
}

await model.dispose()
