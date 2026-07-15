import type { GaussianScene } from '@ai3d/gaussian-scene'

export {
  createGaussianScene,
  encodeGaussianPly,
  encodeGaussianSplat,
  GaussianSceneDisposedError,
} from '@ai3d/gaussian-scene'
export type {
  CreateGaussianSceneInput,
  GaussianScene,
  GaussianSceneMetadata,
} from '@ai3d/gaussian-scene'

export function exportPLY(scene: GaussianScene): Promise<Blob> {
  return scene.exportPLY()
}

export function exportSplat(scene: GaussianScene): Promise<Blob> {
  return scene.exportSplat()
}
