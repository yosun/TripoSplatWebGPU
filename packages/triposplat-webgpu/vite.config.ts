import { defineConfig } from 'vite'

export default defineConfig({
  resolve: {
    // Select ORT's external-WASM entry. The default bundle embeds a ~46 MiB
    // base64 WASM payload and defeats package caching/worker asset reuse.
    conditions: ['onnxruntime-web-use-extern-wasm'],
  },
  build: {
    emptyOutDir: false,
    minify: false,
    sourcemap: true,
    lib: {
      entry: 'src/worker.ts',
      formats: ['es'],
      fileName: () => 'worker.js',
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
  },
})
