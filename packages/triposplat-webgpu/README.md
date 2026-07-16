# @ai3d/triposplat-webgpu

Framework-neutral browser primitives for the TripoSplat WebGPU port. This is an alpha package: the worker runtime, versioned multi-shard manifest format, staged built-in pipeline, official CFG flow sampler, dynamic octree host logic, fixed-length Gaussian decoder, canonical scene contract, PLY/`.splat` export, and browser-local `removeBackground` integration hook are implemented. A prepared-image structural end-to-end browser run passes through exported-PLY viewer loading and a live canvas; official whole-scene/render parity and a bundled WebGPU-qualified opaque-photo segmenter remain release gates.

## Install

The package is not published to the npm registry yet. Install the validated repository tarballs during alpha development:

```sh
npm install ./ai3d-gaussian-scene-0.0.0.tgz ./ai3d-triposplat-webgpu-0.1.0-alpha.0.tgz
```

After both packages are released, the intended registry command is `npm install @ai3d/triposplat-webgpu`.

The package is browser-only at runtime and has no React, viewer, Vite, or Next.js dependency. It ships an ESM worker plus its ONNX Runtime WASM asset. Serve package assets with the same origin or correct CORS headers.

For a copy-ready image-in/PLY-and-`.splat`-out adapter and platform recipes, see the repository's [direct and agent integration guide](../../docs/agent-integration.md) and [`examples/direct`](../../examples/direct).

## Image input and preprocessing

`normalizeTripoSplatImageInput()` is the framework-neutral image boundary. It accepts `Blob`/`File`, `ImageBitmap`, `ImageData`, `HTMLImageElement`, `HTMLCanvasElement`, or `OffscreenCanvas`; DOM-only constructors are feature-guarded so the same module can run in a worker with `Blob`, `ImageBitmap`, `ImageData`, and `OffscreenCanvas`.

```ts
import {
  BackgroundRemovalRequiredError,
  normalizeTripoSplatImageInput,
} from '@ai3d/triposplat-webgpu'

const controller = new AbortController()

try {
  const normalized = await normalizeTripoSplatImageInput(file, {
    signal: controller.signal,
    includeImageBitmap: false,
    // Optional browser-local BiRefNet adapter. It receives the official
    // short-side-1024 RGBA resize and must return same-sized straight-alpha RGBA.
    removeBackground: localBackgroundRemover,
  })

  // NCHW Float32Array inputs at [1, 3, 1024, 1024].
  const { rgb, dinov3, vae } = normalized.tensors
  // `normalized.image` is owned RGB bytes; `normalized.canvas` is the same
  // opaque RGB-on-black input materialized for browser APIs.

  normalized.dispose()
} catch (error) {
  if (error instanceof BackgroundRemovalRequiredError) {
    // Segment the image locally, then pass an alpha-bearing source again.
  }
}
```

The package does not pretend that an opaque photo has a usable foreground mask. Configure `removeBackground` with a validated browser-local segmenter, or an opaque raw input throws `BackgroundRemovalRequiredError`. Alpha-bearing inputs bypass the callback. `inputIsPrepared: true` is only for an already-preprocessed, opaque 1024×1024 RGB-on-black input. Pure RGBA geometry, Pillow-compatible alpha/resize operations, and tensor builders are exported from `@ai3d/triposplat-webgpu/preprocess`.

## High-level lifecycle

```ts
import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'

const model = new TripoSplatWebGPU({
  modelBaseUrl: 'https://models.example.com/triposplat/v1/',
  cache: 'opfs',
  removeBackground: localBackgroundRemover,
})

console.log(await TripoSplatWebGPU.checkCompatibility())
await model.load({ onProgress: console.log })
console.log(model.capabilities)

// generate() uses the built-in staged pipeline when all five graphs are
// configured. It rejects with GraphCapabilityError when any stage is missing.
// Omit removeBackground only when all inputs already carry alpha or are the
// exact prepared RGB-on-black representation.
await model.dispose()
```

`workerUrl` and `workerFactory` are supported for restrictive bundlers. By default the package uses `new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })`.

## Verified model cache

`cache: 'opfs'` is the default and streams graph and external-data bytes into Origin Private File System storage. Before prefetching, the package requests persistent origin storage when available and rejects with storage diagnostics when declared missing bytes exceed the reported quota. `cache: 'cache-api'` uses a temporary streaming Cache API entry and promotes it only after verification. `cache: 'none'` is non-persistent; it still verifies every object and uses temporary OPFS staging when available instead of collecting multi-gigabyte sidecars in renderer memory.

Cache keys include manifest name, version, model revision, precision, graph role, and artifact digest. Signed URLs are never stored in cache metadata, and refreshed signed URLs reuse an immutable artifact when its declared digest is unchanged. Cache hits are rehashed on their first read by each model-manager lifecycle; that verified file-backed Blob is then reused for session creation without a duplicate multi-gigabyte read. Corrupted local entries are removed and replaced from the authoritative CDN object.

```ts
import {
  clearModelCache,
  getModelCacheStatus,
} from '@ai3d/triposplat-webgpu'

const status = await getModelCacheStatus()
console.log(status.entryCount, status.totalBytes, status.backends)

await clearModelCache({
  backend: 'opfs',
  namespace: 'triposplat-webgpu/0.1.0-fp32.20260715/a78fa12d06dbf1381ca548bfac32bb68cb8c451d/fp32',
})
```

## Low-level APIs

```ts
import {
  createRuntime,
  loadGraph,
  runGraph,
  createSampler,
  decodeGaussians,
} from '@ai3d/triposplat-webgpu/low-level'
```

These APIs are usable independently. `createSampler` performs the official shifted Euler schedule and conditional/unconditional classifier-free-guidance calls. `decodeGaussians` implements the 480-feature, 32-Gaussians-per-point representation used by the official decoder.

```ts
import { exportPLY, exportSplat } from '@ai3d/triposplat-webgpu/export'

const ply: Blob = await exportPLY(scene)
const splat: Blob = await exportSplat(scene)
```

## Manifest

The default URL is `<modelBaseUrl>/manifest.json`. Graph and shard URLs resolve relative to the manifest URL; each external-data `path` remains exactly as embedded in the ONNX graph.

The repository's current deployable fp32 manifest is [`public/models/triposplat/manifest.json`](../../public/models/triposplat/manifest.json). It declares all five graph roles, a 6,465,182,402-byte transfer set, and verified SHA-256 digests for every graph and sidecar. See [`docs/model-hosting.md`](../../docs/model-hosting.md) for immutable S3/CloudFront and GCS/Cloud CDN layouts.

```json
{
  "name": "triposplat-webgpu",
  "version": "0.1.0-fp32.20260715",
  "modelRevision": "a78fa12d06dbf1381ca548bfac32bb68cb8c451d",
  "precision": "fp32",
  "graphs": {
    "dit": {
      "url": "dit_step_webgpu_fp32.onnx",
      "precision": "fp32",
      "inputPrecision": "fp32",
      "byteLength": 10685614,
      "integrity": {
        "algorithm": "sha256",
        "digest": "558a52ea5eb0a714ba76e1b273159fe16e473424fe39abb6cc0c4ad65b4bc54d"
      },
      "externalData": [
        {
          "path": "dit_step_webgpu_fp32.onnx.data",
          "url": "dit_step_webgpu_fp32.onnx.data",
          "byteLength": 1633210368,
          "integrity": {
            "algorithm": "sha256",
            "digest": "eacadf21577a976ce0a635cf8b1bbdb4b2c0cdb9cb55d4d73af87957c1e72023"
          }
        }
      ]
    }
  }
}
```

Signed absolute graph URLs are supported. A custom `fetch` applies to the manifest and every model artifact; `manifestRequestInit` and `artifactRequestInit` configure their respective headers and credentials. Artifact requests run on the browser host, are verified locally, and are exposed to the ONNX worker only as short-lived Blob URLs.

## Validation status

On the recorded Apple M3 Max/Chrome WebGPU environment, browser preprocessing plus the full fp32 DINOv3 encoder and one fp32 DiT invocation pass their strict official-PyTorch gates. The fp32 four-step CFG/Euler loop completes all eight WebGPU calls and passes its published qualification envelope, while its latent output still fails a separately recorded stricter diagnostic.

Paired step-4 CPU ORT validation reproduces the same conditional/zero-context asymmetry without WebGPU or session reuse. Stable RMS, static Sobol, real RoPE, and the special output projection are not primary causes. A 39-boundary probe localizes the first material split to the self-attention residual of `context_refiner.0`, before its MLP: conditional maximum error is `3.0994e-6`, while unconditional maximum error is `1.1063e-4`. Adapted PyTorch remains within tolerance at this boundary, so the initiating defect is exported/ORT context-attention execution. No experimental graph has been promoted on that diagnosis.

The full eight-level octree trajectory passes: all eight primary occupancy calls pass on active logits, eight varied-padding probes leave active outputs bit-identical, every sampled child count matches, all 41,972 recorded random values are consumed, and final points are bit-exact. Primary neural inference totals 5,713.7 ms after a 1,327.4 ms model load; the final log-probability maximum error is 0.0000028610. The deliberate padding probes are excluded from that primary sum. See the [octree trajectory report](../../docs/validation/2026-07-15-octree-trajectory-webgpu-fp32-chrome.json).

The raw fp32 Gaussian feature boundary also passes. The packaged four-step path returns 262,144 finite activated Gaussians, valid-size PLY/`.splat` exports, and a ready viewer canvas. The current measured run took 247,901.5 ms through `generate()` and 248,951.4 ms through both export hashes plus viewer load, including non-persistent verified artifact staging. This is a structural/export/viewer-load pass, not official whole-scene or rendered-image parity. See the [end-to-end viewer report](../../docs/benchmarks/2026-07-15-e2e-render-structural-webgpu-fp32-chrome.json).

The built-in worker retains the positive and all-zero DiT conditioning tensors once per DiT session. Each CFG invocation then transfers only latent, camera, and timestep inputs. For fp32 tensors this reduces deterministic host-to-worker payload traffic from 184,774,656 to 46,193,664 bytes at four steps and from 923,873,280 to 46,193,664 bytes at twenty steps. These figures do not claim a wall-time improvement or prove that ONNX Runtime avoids an internal CPU-to-WebGPU upload on each `run()`.

High-level generations are serialized because the staged executor deliberately reuses fixed graph session IDs. A queued call with an aborted signal rejects promptly. Cancelling worker configuration, graph creation, reusable-input retention, or inference terminates the one-shot worker and clears the loaded runtime; the same `TripoSplatWebGPU` instance can then load a fresh worker and retry. `dispose()` is idempotent, terminates pending worker work, and waits for queued generation work to unwind. These paths have deterministic worker tests; repeated full-model cancellation/retry remains a real-browser release gate.

The official 20-step browser loop now has a completed 40-invocation Chrome/WebGPU measurement, but accumulated latent drift exceeds both the qualification and strict gates. Microsoft Edge and the 16 GB Apple Silicon target still have no completed result. Timings in the repository are measured single-machine records, not estimates. See [Current status](../../docs/current-status.md) for exact tolerances, metrics, and evidence files.

## Current limitations

- Interrupted downloads are discarded safely, but HTTP Range-based cross-reload resume is not implemented yet.
- There is no built-in BiRefNet background-removal stage yet.
- All five graph artifacts and complete structural execution pass, but official whole-scene/render parity remains an open release gate.
- Compatibility checks cannot report total GPU/unified memory because browsers do not expose it reliably.
- Opaque source photos require a caller-supplied browser-local `removeBackground` implementation; the canonical package does not yet ship a WebGPU-qualified BiRefNet artifact.
- Production-origin persistent-cache stress behavior, repeated-generation peak memory, Edge, and 16 GB hardware qualification remain open.

No input image is uploaded by this package. Model and manifest files are the only network requests.
