import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { cpSync, mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

/**
 * Model weights and parity fixtures live under public/ for local development,
 * but they must never be copied into an application deployment. Besides being
 * hosted separately in production, the export workspace contains several large
 * diagnostic graphs that intentionally share hard-linked sidecars. Vite's
 * ordinary public-directory copy expands those links into many gigabytes.
 *
 * Only the ORT WASM fallback assets and the social-preview image are
 * application-owned static files.
 */
function copyRuntimeAssets() {
  return {
    name: 'copy-runtime-assets',
    closeBundle() {
      const outputDirectory = resolve('dist')
      mkdirSync(outputDirectory, { recursive: true })
      cpSync(resolve('public/ort'), resolve(outputDirectory, 'ort'), {
        recursive: true,
      })
      cpSync(resolve('public/vite.svg'), resolve(outputDirectory, 'vite.svg'))
      cpSync(
        resolve('public/corgi.ceo_image_header.social.jpg'),
        resolve(outputDirectory, 'corgi.ceo_image_header.social.jpg'),
      )
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), copyRuntimeAssets()],
  build: {
    // Do not deploy local model artifacts or deterministic parity fixtures.
    copyPublicDir: false,
    rollupOptions: {
      input: {
        app: 'index.html',
        e2eWeb: 'e2e-web.html',
        sharpLab: 'sharp-lab.html',
        encoderLab: 'encoder-lab.html',
        dinoLab: 'dino-lab.html',
        ditLab: 'dit-lab.html',
        flowLab: 'flow-lab.html',
        octreeLab: 'octree-lab.html',
        gaussianLab: 'gaussian-lab.html',
        e2eLab: 'e2e-lab.html',
      },
    },
  },
  // dialkit and motion each import react; without deduping, Vite can hand them a
  // separate React instance than the app, triggering "Invalid hook call".
  resolve: {
    dedupe: ['react', 'react-dom'],
  },
})
