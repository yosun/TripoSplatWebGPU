# Alpha API reference

This page documents the current alpha workspace surface. It is not a production API guarantee. The package is `UNLICENSED` and unpublished. It now includes a built-in full-stage executor, but the complete browser output has not passed the official end-to-end numerical and visual release gates described in [Current status](current-status.md).

## Entry points

```ts
import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'
import {
  createRuntime,
  createSampler,
  sampleOctree,
} from '@ai3d/triposplat-webgpu/low-level'
import { exportPLY, exportSplat } from '@ai3d/triposplat-webgpu/export'
import { buildDinov3Tensor } from '@ai3d/triposplat-webgpu/preprocess'
import {
  clearModelCache,
  getModelCacheStatus,
} from '@ai3d/triposplat-webgpu'
```

The root entry exports the high-level class, built-in pipeline function, cache controls, image normalizer, compatibility helper, manifest/capability types, canonical Gaussian scene types, and typed errors. The low-level entry exports worker runtime, verified-artifact helpers, graph loading/running, tensor conversion, preprocessing math, flow sampler, dynamic octree, Gaussian decoder, and manifest helpers. The preprocessing entry exposes the image boundary and its pure pixel/geometry helpers. The export entry re-exports `@ai3d/gaussian-scene` and convenience export functions.

## `TripoSplatWebGPU`

```ts
const model = new TripoSplatWebGPU({
  modelBaseUrl: 'https://models.example.com/triposplat/v1/',
  manifestUrl: 'manifest.json',
  executionProviders: ['webgpu'],
  cache: 'opfs',
  logLevel: 'info',
})
```

Constructor options:

| Option | Alpha behavior |
| --- | --- |
| `modelBaseUrl` | Required absolute or browser-relative model directory |
| `manifestUrl` | Optional; defaults to `manifest.json` relative to the model base |
| `executionProviders` | Defaults to `['webgpu']`; an empty array rejects |
| `cache` | `'none'`, `'opfs'`, or `'cache-api'`; defaults to `'opfs'` |
| `workerUrl` | Optional custom module-worker URL |
| `workerFactory` | Optional custom worker constructor hook |
| `logLevel` | Runtime status logging policy |
| `fetch` | Custom fetch for the manifest and every graph/external-data artifact |
| `manifestRequestInit` | Headers and other manifest request options except signal |
| `artifactRequestInit` | Headers/credentials for artifact GETs; method, body, and signal remain package-controlled |
| `wasmPaths` | Explicit ONNX Runtime WASM asset prefix or mapping |
| `pipeline` | Optional advanced override; the built-in staged executor is the default |
| `preprocess` | Optional validated browser-local preprocessing/segmentation override |
| `removeBackground` | Optional browser-local opaque-image remover; receives the official short-side-1024 RGBA resize and must return same-size RGBA |

The high-level facade downloads and verifies model objects on the browser host, then gives the worker short-lived Blob URLs. Custom authentication can therefore use the shared `fetch` hook, per-artifact request options, or signed URLs in the manifest. Signed query strings are not included in stable cache keys.

### `TripoSplatWebGPU.checkCompatibility()`

Requests a WebGPU adapter and reports selected numeric limits, warnings, and blockers. A successful result is still `level: 'experimental'`. The static check does not load a manifest, so `estimatedModelBytes` remains zero. It never invents a peak-memory value because browsers do not expose reliable unified/GPU memory totals.

### `load(options?)`

Loads and validates the JSON manifest, checks WebGPU when requested, initializes the selected artifact cache, and creates the worker runtime. With `opfs` or `cache-api`, it prefetches and verifies configured graph and external-data artifacts. With `none`, artifacts are downloaded and verified when each stage loads. Progress stages are `manifest`, `compatibility`, `runtime`, `graphs`, and `complete`.

Graph sessions are deliberately not preloaded: `generate()` loads and disposes them by stage to avoid intentionally retaining the encoders, DiT, octree, and Gaussian decoder at once. Concurrent `load()` calls share one promise; a failed load can be retried. High-level `generate()` calls are serialized because they share stable worker session IDs. An aborted queued call rejects promptly without bypassing an active earlier call.

### `manifest` and `capabilities`

`manifest` exposes the resolved manifest after load. `capabilities` reports configured and missing graph descriptors, whether an encoder slice exists, and whether all five descriptors are present. `capabilities.fullGeneration` is a configuration check, not an end-to-end numerical qualification or hardware guarantee.

### `generate(input, options?)`

Accepted input types are `Blob`/`File`, `ImageBitmap`, `ImageData`, `HTMLImageElement`, `HTMLCanvasElement`, and `OffscreenCanvas`. Options include steps, guidance scale, Gaussian count, seed, precision, `AbortSignal`, and real stage progress fields.

The class calls `load()`, requires `dino`, `vae`, `dit`, `octree`, and `gaussianDecoder` descriptors, and then runs the built-in browser-local executor unless a custom `pipeline` override was supplied. Missing descriptors produce `GraphCapabilityError`; the package does not return a partial or fabricated scene.

The built-in path:

1. normalizes a prepared alpha-bearing or RGB-on-black input;
2. runs DINOv3, disposes it, then runs and disposes the Flux VAE;
3. loads the DiT and performs the official 4-step or 20-step conditional/unconditional loop;
4. runs eight dynamic octree levels with host-side systematic resampling;
5. runs the Gaussian feature decoder and packs a canonical `GaussianScene`.

The graph declared precision controls tensor arithmetic and public graph conversion. `GenerateOptions.precision` does not convert a loaded graph to another precision.

## Image normalization and preprocessing

```ts
const normalized = await normalizeTripoSplatImageInput(source, {
  signal,
  includeImageBitmap: true,
})
```

`normalizeTripoSplatImageInput()` decodes the browser source, applies the official resize/alpha-erosion/square-crop/black-composite geometry, and returns:

- `image`: owned interleaved RGB bytes;
- `foreground`: owned straight-alpha RGBA bytes immediately before compositing;
- `tensors.rgb`: planar NCHW float32 RGB in `[0, 1]`;
- `tensors.dinov3`: the same RGB normalized with the official DINOv3 mean and standard deviation;
- `tensors.vae`: the same RGB mapped to `[-1, 1]`;
- `canvas`: an `OffscreenCanvas` when available, otherwise an `HTMLCanvasElement` main-thread fallback;
- optional `imageBitmap` and an idempotent `dispose()` that closes it.

The default model canvas is 1024×1024. Set `includeImageBitmap: false` when only pixels, tensors, or canvas are needed. The operation checks its `AbortSignal` during CPU loops and closes a bitmap created after cancellation.

The package does not bundle or simulate a foreground-segmentation model. An alpha-bearing source proceeds through preprocessing; an opaque raw source throws `BackgroundRemovalRequiredError` (`code: 'BACKGROUND_REMOVAL_REQUIRED'`) unless the constructor received a browser-local `removeBackground` callback. That callback receives the official Lanczos-resized straight-alpha RGBA image after its short side reaches 1024, must return same-size RGBA, and receives the generation `AbortSignal`. The package copies its result before alpha erosion/cropping and records `usedBackgroundRemoval` in scene metadata. `inputIsPrepared: true` bypasses segmentation only for an opaque input already in the official 1024×1024 RGB-on-black representation.

```ts
const model = new TripoSplatWebGPU({
  modelBaseUrl,
  removeBackground: async (resizedRgba, { signal } = {}) => {
    // Run a separately validated browser-local matte model here.
    return runLocalMatte(resizedRgba, signal)
  },
})
```

For worker-owned bytes and numerical tests, import `preprocessTripoSplatRgba()`, `calculateTripoSplatResize()`, `calculateTripoSplatCrop()`, `resizeRgbaLanczos()`, the alpha helpers, and the tensor builders from `@ai3d/triposplat-webgpu/preprocess`. No function evaluates an unguarded DOM global at module import time.

### `dispose()`

Idempotently clears manifest/runtime references, terminates pending worker work, and waits for high-level generation work to unwind. Aborting an in-flight worker configuration, graph load, reusable-input transfer, or inference terminates that one-shot worker; the same model object lazily creates a clean runtime on its next `load()` or `generate()` call. Calling `load()` or `generate()` after model disposal throws a typed disposed `TripoSplatError`.

## Built-in pipeline and executor override

`runBuiltInTripoSplatPipeline(context)` is exported from the root entry and is the default used by `generate()`. Advanced integrations can replace it with `options.pipeline`; tests can also replace `options.preprocess` with a validated local segmentation/preprocessing stage.

An executor receives:

```ts
interface TripoSplatPipelineContext {
  input: TripoSplatInput
  options: GenerateOptions
  manifest: ResolvedTripoSplatModelManifest
  runtime: TripoSplatRuntime
  sessionIds: Readonly<Record<TripoSplatGraphName, string>>
  preprocess: TripoSplatPreprocessor
  removeBackground?: TripoSplatBackgroundRemover
}
```

It must return an owned canonical `GaussianScene`. Supplying an override transfers numerical correctness, graph residency, cancellation, and resource-disposal responsibility to the integrator.

## Low-level runtime

```ts
const runtime = createRuntime({
  executionProviders: ['webgpu'],
  baseUrl: modelBaseUrl,
})

const info = await loadGraph(runtime, 'triposplat/dit', graph)
const result = await runGraph(runtime, 'triposplat/dit', inputs, {
  outputs: ['pred_latent', 'pred_camera'],
  signal,
})

await runtime.disposeGraph('triposplat/dit')
await runtime.dispose()
```

`runGraph` transfers input arrays by default, detaching their buffers. Set `transferInputs: false` when ownership must remain with the caller, at a copy cost. Output timings separate inference, readback, and total worker time.

The worker supports multiple external-data shards through the manifest's `{path, url}` records. `path` must match ONNX external-data metadata. Bare `createRuntime()` is intentionally a low-level URL loader; use the high-level `TripoSplatWebGPU` facade or wrap a runtime with `withVerifiedModelArtifacts()` and `ModelArtifactManager` when byte-length/integrity enforcement and persistent caching are required.

## Verified model cache

`getModelCacheStatus(options?)` inspects Cache API and OPFS namespaces without loading a model. `clearModelCache(options?)` removes all persistent package entries or only the selected backend/namespace.

```ts
const status = await getModelCacheStatus({ backend: 'opfs' })
await clearModelCache({
  backend: 'opfs',
  namespace: 'triposplat-webgpu/1.0.0/revision/fp32',
})
```

Declared graph and shard `byteLength` and SHA-256 values are checked before session creation. Persistent writes use temporary entries and are committed only after verification. Cache hits are rehashed; corrupt entries are evicted and replaced from the authoritative URL. OPFS is the default and performs a storage-quota preflight. `cache: 'none'` performs the same declared checks without retaining artifacts and uses temporary OPFS files when the API is available.

## Flow sampler

`createSampler(predictor)` returns an Euler classifier-free-guidance sampler. It supports arbitrary positive step counts, the official shifted schedule, per-tensor guidance, fp16/fp32 prediction arithmetic, cancellation between invocations, and actual invocation counters.

With guidance greater than one, each official step performs conditional and unconditional predictor calls. Twenty steps therefore means forty DiT calls.

## Gaussian decoding and export

`decodeGaussians(points, features, {metadata})` decodes the fixed official 480-feature ElasticGaussian representation into 32 Gaussians per point and returns an owned canonical scene.

```ts
const ply = await exportPLY(scene)
const splat = await exportSplat(scene)
scene.dispose()
```

PLY uses the binary 17-float 3DGS layout. `.splat` uses the documented de-facto 32-byte browser layout. See [Gaussian conventions](gaussian-conventions.md).

## Typed errors

All package errors extend `TripoSplatError` and expose `code`, `stage`, `recoverable`, `cause`, and frozen `diagnostics`:

- `WebGPUUnavailableError`;
- `UnsupportedAdapterError`;
- `ModelDownloadError`;
- `ModelIntegrityError`;
- `ManifestError`;
- `GraphLoadError`;
- `GraphCapabilityError`;
- `BackgroundRemovalRequiredError`;
- `InferenceError`;
- `OutOfMemoryError`;
- `CancelledError`;
- `ExportError`.

The high-level artifact manager raises `ModelIntegrityError` for a declared digest or length mismatch before ONNX Runtime sees the object. Low-level callers that use bare `createRuntime()` bypass that manager by design.

## Remaining alpha limitations

- There is no built-in BiRefNet graph for opaque photographs.
- HTTP Range-based cross-reload download resume is not implemented; interrupted partial entries are discarded.
- Built-in orchestration is implemented. A prepared-image four-step run completes through 262,144 finite Gaussians and both exports, but official whole-scene/render parity is not claimed. The measured 20-step browser loop completes all 40 calls but fails its final-state gate.
- Cache behavior has unit coverage; production-origin CDN/OPFS stress behavior still needs real-browser qualification.
- Compatibility reporting cannot provide a trustworthy unified/GPU peak-memory figure.
- Chrome/Edge on a 16 GB Apple Silicon Mac are not yet qualified.
