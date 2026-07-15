import { spawnSync } from 'node:child_process'
import { cp, mkdir, mkdtemp, readdir, readFile, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const viteFixtureRoot = join(repositoryRoot, 'tests/consumers/vite')
const nextClientFixtureRoot = join(repositoryRoot, 'tests/consumers/next-client')
const importMapFixtureRoot = join(repositoryRoot, 'tests/consumers/import-map')
const temporaryRoot = await mkdtemp(join(tmpdir(), 'triposplat-vite-consumer-'))
const packRoot = join(temporaryRoot, 'packs')
const consumerRoot = join(temporaryRoot, 'consumer')
const importMapRoot = join(temporaryRoot, 'import-map')
const keepTemporary = process.env.KEEP_PACKAGE_CONSUMER_TEMP === '1'
let failed = false

function executable(name) {
  return process.platform === 'win32' ? `${name}.cmd` : name
}

function run(command, args, cwd = repositoryRoot) {
  process.stdout.write(`\n> ${command} ${args.join(' ')}\n`)
  const result = spawnSync(executable(command), args, {
    cwd,
    env: process.env,
    stdio: 'inherit',
  })
  if (result.error) throw result.error
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} exited with status ${result.status}.`)
  }
}

async function archiveContaining(fragment) {
  const names = await readdir(packRoot)
  const matches = names.filter((name) => name.includes(fragment) && name.endsWith('.tgz'))
  if (matches.length !== 1) {
    throw new Error(`Expected one '${fragment}' tarball in ${packRoot}; found ${matches.length}.`)
  }
  return join(packRoot, matches[0])
}

async function copyFixture(sourceRoot, destinationRoot) {
  await cp(sourceRoot, destinationRoot, {
    recursive: true,
    filter(source) {
      const name = basename(source)
      return name !== 'node_modules'
        && name !== 'dist'
        && name !== '.next'
        && name !== 'package-lock.json'
    },
  })
}

async function filesBelow(root) {
  const output = []
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name)
    if (entry.isDirectory()) output.push(...await filesBelow(path))
    else output.push(path)
  }
  return output
}

async function validateBundle() {
  const distRoot = join(consumerRoot, 'dist')
  const files = await filesBelow(distRoot)
  if (!files.some((file) => file.endsWith('.wasm'))) {
    throw new Error('The installed-package Vite build omitted the ONNX Runtime WASM asset.')
  }
  let containsWorker = false
  for (const file of files.filter((path) => path.endsWith('.js'))) {
    const details = await stat(file)
    if (details.size > 2_000_000) continue
    const source = await readFile(file, 'utf8')
    if (source.includes('triposplat-onnx-webgpu') || source.includes('TripoSplat runtime worker')) {
      containsWorker = true
      break
    }
  }
  if (!containsWorker) {
    throw new Error('The installed-package Vite build omitted the TripoSplat module worker.')
  }
  const relativeFiles = files.map((file) => file.slice(distRoot.length + 1)).sort()
  process.stdout.write(`\nFresh Vite consumer build passed in ${consumerRoot}.\n`)
  process.stdout.write(`Built assets:\n${relativeFiles.map((file) => `  ${file}`).join('\n')}\n`)
}

function assertInside(root, path, label) {
  const pathFromRoot = relative(root, path)
  if (pathFromRoot === '' || (!pathFromRoot.startsWith('..') && !isAbsolute(pathFromRoot))) return
  throw new Error(`${label} resolves outside the static consumer root: ${path}`)
}

function resolveImportSpecifier(specifier, parentFile, imports) {
  let target
  if (specifier.startsWith('./') || specifier.startsWith('../')) {
    target = resolve(dirname(parentFile), specifier)
  } else {
    const exact = imports[specifier]
    if (typeof exact === 'string') {
      target = resolve(importMapRoot, exact)
    } else {
      const prefix = Object.keys(imports)
        .filter((key) => key.endsWith('/') && specifier.startsWith(key))
        .sort((left, right) => right.length - left.length)[0]
      const mappedPrefix = prefix === undefined ? undefined : imports[prefix]
      if (typeof prefix !== 'string' || typeof mappedPrefix !== 'string') {
        throw new Error(`No import-map entry resolves '${specifier}' imported by ${parentFile}.`)
      }
      target = resolve(importMapRoot, mappedPrefix, specifier.slice(prefix.length))
    }
  }
  assertInside(importMapRoot, target, `Module '${specifier}'`)
  return target
}

function staticModuleSpecifiers(source) {
  const specifiers = []
  const pattern = /\b(?:import|export)\s+(?:(?:[^"'();]*?)\s+from\s+)?["']([^"']+)["']/g
  for (const match of source.matchAll(pattern)) specifiers.push(match[1])
  return specifiers
}

async function validateMappedPackageExports(packageName, packageRoot, imports) {
  const manifest = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8'))
  for (const [subpath, descriptor] of Object.entries(manifest.exports ?? {})) {
    const importTarget = typeof descriptor === 'string' ? descriptor : descriptor?.import
    if (typeof importTarget !== 'string') continue
    const specifier = subpath === '.' ? packageName : `${packageName}${subpath.slice(1)}`
    const mapped = imports[specifier]
    if (typeof mapped !== 'string') {
      throw new Error(`The native ESM fixture does not map public export '${specifier}'.`)
    }
    const expected = resolve(packageRoot, importTarget)
    const actual = resolve(importMapRoot, mapped)
    if (actual !== expected) {
      throw new Error(`Import-map target for '${specifier}' is '${actual}', expected '${expected}'.`)
    }
    await stat(actual)
  }
}

async function prepareImportMapConsumer() {
  await copyFixture(importMapFixtureRoot, importMapRoot)
  const scopeRoot = join(importMapRoot, 'vendor/@ai3d')
  await mkdir(scopeRoot, { recursive: true })
  await cp(
    join(consumerRoot, 'node_modules/@ai3d/gaussian-scene'),
    join(scopeRoot, 'gaussian-scene'),
    { recursive: true },
  )
  await cp(
    join(consumerRoot, 'node_modules/@ai3d/triposplat-webgpu'),
    join(scopeRoot, 'triposplat-webgpu'),
    { recursive: true },
  )
}

async function validateImportMapConsumer() {
  const htmlPath = join(importMapRoot, 'index.html')
  const html = await readFile(htmlPath, 'utf8')
  const importMapMatch = /<script\s+type=["']importmap["'][^>]*>([\s\S]*?)<\/script>/i.exec(html)
  const moduleMatch = /<script\s+type=["']module["']\s+src=["']([^"']+)["'][^>]*><\/script>/i.exec(html)
  if (!importMapMatch || !moduleMatch) {
    throw new Error('Native ESM fixture needs an import map followed by an external module script.')
  }
  if ((importMapMatch.index ?? 0) > (moduleMatch.index ?? 0)) {
    throw new Error('The import map must precede the module script in the HTML document.')
  }

  const parsed = JSON.parse(importMapMatch[1])
  const imports = parsed.imports
  if (!imports || typeof imports !== 'object' || Array.isArray(imports)) {
    throw new Error('Native ESM fixture has no valid import-map imports object.')
  }

  const triposplatRoot = join(importMapRoot, 'vendor/@ai3d/triposplat-webgpu')
  const gaussianSceneRoot = join(importMapRoot, 'vendor/@ai3d/gaussian-scene')
  await validateMappedPackageExports('@ai3d/triposplat-webgpu', triposplatRoot, imports)
  await validateMappedPackageExports('@ai3d/gaussian-scene', gaussianSceneRoot, imports)

  const requiredAssets = [
    join(triposplatRoot, 'dist/worker.js'),
    join(triposplatRoot, 'dist/ort/ort-wasm-simd-threaded.asyncify.mjs'),
    join(triposplatRoot, 'dist/ort/ort-wasm-simd-threaded.asyncify.wasm'),
  ]
  for (const asset of requiredAssets) await stat(asset)

  const entry = resolve(importMapRoot, moduleMatch[1])
  assertInside(importMapRoot, entry, 'Module entry')
  const pending = [entry]
  const visited = new Set()
  while (pending.length > 0) {
    const modulePath = pending.pop()
    if (visited.has(modulePath)) continue
    visited.add(modulePath)
    const source = await readFile(modulePath, 'utf8')
    for (const specifier of staticModuleSpecifiers(source)) {
      const dependency = resolveImportSpecifier(specifier, modulePath, imports)
      await stat(dependency)
      if (dependency.endsWith('.js') || dependency.endsWith('.mjs')) pending.push(dependency)
    }
  }

  process.stdout.write(`\nPacked native ESM/import-map graph passed (${visited.size} JavaScript modules).\n`)
  process.stdout.write('This build-time check does not launch a browser or run WebGPU inference.\n')
}

try {
  await mkdir(packRoot, { recursive: true })
  run('pnpm', ['--filter', '@ai3d/gaussian-scene', 'pack', '--pack-destination', packRoot])
  run('pnpm', ['--filter', '@ai3d/triposplat-webgpu', 'pack', '--pack-destination', packRoot])

  const sceneArchive = await archiveContaining('gaussian-scene')
  const coreArchive = await archiveContaining('triposplat-webgpu')
  await copyFixture(viteFixtureRoot, consumerRoot)
  await copyFixture(nextClientFixtureRoot, join(consumerRoot, 'next-client'))
  run('npm', [
    'install',
    '--ignore-scripts',
    '--no-audit',
    '--no-fund',
    '--no-package-lock',
    sceneArchive,
    coreArchive,
  ], consumerRoot)
  run('npm', ['run', 'build'], consumerRoot)
  await validateBundle()
  run('npm', ['run', 'typecheck:next-client'], consumerRoot)
  process.stdout.write('\nPacked Next-style client component typecheck passed.\n')
  process.stdout.write('This check does not install or invoke Next.js/Turbopack.\n')
  await prepareImportMapConsumer()
  await validateImportMapConsumer()
} catch (error) {
  failed = true
  process.stderr.write(`\nPacked package consumer smoke failed. Temporary files: ${temporaryRoot}\n`)
  throw error
} finally {
  if (!keepTemporary && !failed) await rm(temporaryRoot, { recursive: true, force: true })
  else process.stdout.write(`Temporary files retained at ${temporaryRoot}\n`)
}
