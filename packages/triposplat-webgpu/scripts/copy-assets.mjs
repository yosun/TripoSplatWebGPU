import { copyFile, mkdir } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = resolve(packageRoot, '../..')
const sourceRoot = resolve(repositoryRoot, 'node_modules/onnxruntime-web/dist')
const outputRoot = resolve(packageRoot, 'dist/ort')
const assets = [
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]

await mkdir(outputRoot, { recursive: true })
await Promise.all(assets.map((asset) => copyFile(resolve(sourceRoot, asset), resolve(outputRoot, asset))))
