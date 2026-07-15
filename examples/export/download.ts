/** Export any owned GaussianScene returned by the package or another compatible source. */
import type { GaussianScene } from '@ai3d/triposplat-webgpu'

export async function downloadPly(scene: GaussianScene, name = 'triposplat.ply') {
  const blob = await scene.exportPLY()
  const url = URL.createObjectURL(blob)
  try {
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = name
    anchor.click()
  } finally {
    URL.revokeObjectURL(url)
  }
}

export async function downloadSplat(scene: GaussianScene, name = 'triposplat.splat') {
  const blob = await scene.exportSplat()
  const url = URL.createObjectURL(blob)
  try {
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = name
    anchor.click()
  } finally {
    URL.revokeObjectURL(url)
  }
}

export async function exportAndRelease(scene: GaussianScene) {
  try {
    await downloadPly(scene)
  } finally {
    scene.dispose()
  }
}
