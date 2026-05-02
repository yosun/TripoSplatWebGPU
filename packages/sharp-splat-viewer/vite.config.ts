import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'

export default defineConfig(({ mode }) => {
  const isIife = mode === 'iife'
  return {
    plugins: isIife
      ? []
      : [
          dts({
            insertTypesEntry: true,
            rollupTypes: true,
            tsconfigPath: resolve(__dirname, 'tsconfig.json'),
          }),
        ],
    build: {
      emptyOutDir: !isIife,
      sourcemap: true,
      target: 'es2020',
      lib: {
        entry: resolve(__dirname, 'src/index.ts'),
        name: 'SharpSplatViewer',
        formats: isIife ? ['iife'] : ['es'],
        fileName: (format) =>
          format === 'iife' ? 'sharp-splat-viewer.iife.js' : 'sharp-splat-viewer.es.js',
      },
      rollupOptions: isIife
        ? {}
        : {
            external: ['three', '@mkkellogg/gaussian-splats-3d'],
            output: {
              globals: {
                three: 'THREE',
                '@mkkellogg/gaussian-splats-3d': 'GaussianSplats3D',
              },
            },
          },
    },
  }
})
