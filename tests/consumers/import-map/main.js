import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'
import { exportSplat } from '@ai3d/triposplat-webgpu/export'
import { createFlowSchedule } from '@ai3d/triposplat-webgpu/low-level'
import { preprocessTripoSplatRgba } from '@ai3d/triposplat-webgpu/preprocess'

const model = new TripoSplatWebGPU({
  modelBaseUrl: 'https://models.example.invalid/triposplat/v1/',
  cache: 'none',
})

try {
  const schedule = createFlowSchedule(4, 3)
  if (schedule.length !== 5) throw new Error(`Unexpected flow schedule length: ${schedule.length}`)
  if (typeof exportSplat !== 'function' || typeof preprocessTripoSplatRgba !== 'function') {
    throw new Error('One or more packed package subpath exports are unavailable.')
  }

  globalThis.triposplatImportMapSmoke = {
    package: '@ai3d/triposplat-webgpu',
    steps: schedule.length - 1,
    inferenceStarted: false,
  }
  document.body.dataset.smoke = 'pass'
  document.querySelector('#status').textContent = 'Packed native ESM imports passed (inference not started).'
} catch (error) {
  document.body.dataset.smoke = 'fail'
  document.querySelector('#status').textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  throw error
} finally {
  await model.dispose()
}
