# Direct image-to-splat example

This is the shortest reusable integration for turning one browser image into both a 3D Gaussian Splatting PLY file and a browser `.splat` file.

The example is framework-neutral. It keeps the input image in browser memory, reports progress, supports cancellation, exports both formats, and releases the generated scene after encoding.

> **Alpha status:** generation currently runs in a WebGPU-capable browser and the package is still experimental. Use an alpha-bearing image or a prepared RGB-on-black image. A normal opaque JPEG requires a caller-provided local background-removal adapter.

## Files

| File | Purpose |
| --- | --- |
| [`image-to-splat.ts`](image-to-splat.ts) | Reusable image-in/files-out adapter |
| [`browser.ts`](browser.ts) | File picker, compatibility check, progress, cancellation, and downloads |
| [`index.html`](index.html) | Minimal Vite-compatible page |
| [`model-config-server.mjs`](model-config-server.mjs) | Optional Node endpoint for public model configuration |

The Node script serves configuration only. It does not perform inference and does not receive user images.

## Runtime architecture

```text
User image (File/Blob)
        │
        ▼
Browser WebGPU + module worker
        │
        ▼
GaussianScene
        │
        ├── binary PLY Blob
        └── browser .splat Blob
```

The model manifest and ONNX artifacts are downloaded by the browser from `modelBaseUrl`. The image is not uploaded by this package. A backend can render the model URL into a page or return it from an API, but generation must remain behind the browser/client boundary.

## Run in an existing Vite app

The package is not published to npm yet. From this repository, build the two public workspace packages and pack them into a temporary directory:

```sh
pnpm install --frozen-lockfile
pnpm --filter @ai3d/gaussian-scene build
pnpm --filter @ai3d/triposplat-webgpu build
mkdir -p /tmp/triposplat-packs
pnpm --filter @ai3d/gaussian-scene pack --pack-destination /tmp/triposplat-packs
pnpm --filter @ai3d/triposplat-webgpu pack --pack-destination /tmp/triposplat-packs
```

In the consuming Vite application, install the generated tarballs:

```sh
npm install /tmp/triposplat-packs/ai3d-gaussian-scene-*.tgz \
  /tmp/triposplat-packs/ai3d-triposplat-webgpu-*.tgz
```

After publication, use:

```sh
npm install @ai3d/triposplat-webgpu
```

Copy these files into the client-side portion of the application:

```text
examples/direct/image-to-splat.ts
examples/direct/browser.ts
examples/direct/index.html
```

Set the model directory in `index.html`:

```html
<meta
  name="triposplat-model-base-url"
  content="https://cdn.example.com/triposplat/v1/"
/>
```

The URL must point to the directory containing `manifest.json`. The manifest must resolve every graph and external-data artifact. Do not point this value at an application API route that proxies multi-gigabyte model files.

Start the Vite app normally and open the page in a current Chrome or Edge build with WebGPU enabled. Choose an alpha-bearing PNG/WebP and click **Generate**. The first run downloads and verifies the model artifacts; the current fp32 manifest declares approximately 6.465 GB.

The example defaults to the 4-step schedule and produces:

- `triposplat.ply` — binary little-endian 17-float 3DGS PLY;
- `triposplat.splat` — 32-byte browser splat records;
- Gaussian count, output sizes, and elapsed time in the page status.

## Use the adapter directly

For an application that already has its own UI, import only the reusable adapter:

```ts
import { ImageToSplatGenerator } from './image-to-splat'

const generator = new ImageToSplatGenerator({
  modelBaseUrl: 'https://cdn.example.com/triposplat/v1/',
  manifestUrl: 'manifest.json',
  executionProviders: ['webgpu'],
  cache: 'opfs',
})

const compatibility = await ImageToSplatGenerator.checkCompatibility()
if (!compatibility.supported) {
  throw new Error(compatibility.blockers.join('\n') || 'WebGPU is unavailable.')
}

const output = await generator.generate(file, {
  steps: 4,
  guidanceScale: 3,
  gaussianCount: 262144,
  seed: 42,
  onProgress: ({ stage, message, progress }) => {
    console.info(stage, message, progress)
  },
})

await saveBlob(output.ply, 'object.ply')
await saveBlob(output.splat, 'object.splat')
await generator.dispose()
```

`saveBlob` is application code. A browser download helper can use `URL.createObjectURL`; a viewer can use the object URL or `blob.arrayBuffer()`; a deliberate remote-storage integration can upload the blob after generation.

Do not retain the internal `GaussianScene` in UI state. The adapter exports both files and disposes the scene in a `finally` block.

## Input contract

The adapter accepts the browser image sources supported by `TripoSplatWebGPU.generate()`:

- `File` or `Blob`;
- `ImageBitmap`;
- `ImageData`;
- `HTMLImageElement`;
- `HTMLCanvasElement`;
- `OffscreenCanvas`.

Use a transparent subject image when possible. The preprocessing pipeline uses the alpha channel to isolate the subject and composites the result onto the required black background.

A raw opaque photo does not contain a foreground mask. It raises `BackgroundRemovalRequiredError` unless the generator is configured with a validated browser-local `removeBackground` callback. Do not set `inputIsPrepared: true` as a generic bypass; that flag is only for the exact official 1024×1024 RGB-on-black representation.

A remover can be provided at construction time:

```ts
const generator = new ImageToSplatGenerator({
  modelBaseUrl,
  removeBackground: async (resizedRgba, { signal } = {}) => {
    return runLocalBackgroundRemover(resizedRgba, signal)
  },
})
```

The remover must return same-size straight-alpha RGBA pixels and remain local to the browser. This repository does not bundle a qualified background-removal model.

## Output contract

`generate()` returns:

```ts
interface ImageToSplatResult {
  count: number
  metadata: GaussianSceneMetadata
  ply: Blob
  splat: Blob
  elapsedMs: number
}
```

`ply` is the canonical binary little-endian 3DGS PLY export. `splat` uses the de-facto 32-byte browser record layout documented in [`docs/gaussian-conventions.md`](../../docs/gaussian-conventions.md). The `.splat` extension is not a universal format identifier; confirm that the target viewer supports this layout.

The current published decoder contract emits 262,144 Gaussians. The adapter passes that value explicitly so a changed decoder contract cannot silently produce a different output shape.

## Cancellation and lifecycle

Use one `AbortController` per user request:

```ts
const controller = new AbortController()
const pending = generator.generate(file, {
  signal: controller.signal,
  onProgress: ({ message }) => updateStatus(message),
})

cancelButton.onclick = () => controller.abort()
await pending
```

A cancelled in-flight WebGPU operation tears down the one-shot worker runtime. The same generator can be retried and will lazily create a clean runtime. Calls on one generator are serialized; reuse one generator instead of creating one per click.

Dispose the generator when the page, route, or component is destroyed:

```ts
window.addEventListener('pagehide', () => {
  void generator.dispose()
}, { once: true })
```

Revoke any object URLs created for previews or downloads. Keep only the output blobs that the host application actually needs.

## Server-rendered and multi-tenant apps

The optional Node endpoint exposes only a public model URL:

```sh
TRIPOSPLAT_MODEL_BASE_URL=https://cdn.example.com/triposplat/v1/ \
  node examples/direct/model-config-server.mjs
```

It responds at `/api/triposplat-config` with:

```json
{
  "modelBaseUrl": "https://cdn.example.com/triposplat/v1/",
  "manifestUrl": "manifest.json"
}
```

Mount `createTripoSplatConfigHandler()` in an existing Node server, or implement the same response in another backend. A client can consume it before constructing the generator:

```ts
const config = await fetch('/api/triposplat-config').then((response) => {
  if (!response.ok) throw new Error('Unable to load TripoSplat configuration.')
  return response.json() as Promise<{
    modelBaseUrl: string
    manifestUrl: string
  }>
})

const generator = new ImageToSplatGenerator(config)
```

Do not put CDN signing secrets in this response. For private model access, use short-lived browser-compatible signed URLs or the package `fetch`, `manifestRequestInit`, and `artifactRequestInit` hooks. Avoid logging signed URLs.

## Model hosting checklist

Host the complete immutable model release on an object CDN or static origin:

- serve `manifest.json`, every graph, and every external-data sidecar;
- preserve the external-data paths embedded in each ONNX graph;
- enable CORS for the application origin;
- support byte-range requests and correct `Content-Length` values;
- keep versioned artifacts immutable;
- do not route model payloads through serverless functions or an application proxy;
- do not include model payloads in npm tarballs or the JavaScript bundle.

See [`docs/model-hosting.md`](../../docs/model-hosting.md) for the deployment layout and cache behavior.

## Platform boundaries

- **Vite, Vue, Svelte, Angular, plain web:** import the adapter from client code and let the bundler resolve the package worker.
- **React:** keep one generator in a ref/provider and call `dispose()` in effect cleanup. Do not put typed Gaussian arrays in React state.
- **Next.js:** place the adapter in a `'use client'` module or a client component loaded with SSR disabled. Server Components and Route Handlers cannot run generation.
- **Node/serverless/edge:** serve configuration or model metadata only. These runtimes are not an inference fallback.
- **Electron/native shells:** use a renderer/WebView with WebGPU, module workers, Canvas/OffscreenCanvas, and sufficient storage; check compatibility on the target shell.

## Coding-agent handoff

Give an implementation agent this instruction when integrating the example:

```text
Integrate @ai3d/triposplat-webgpu using examples/direct/image-to-splat.ts.
Keep inference client-only. Accept File/Blob input, run compatibility checks,
show load and generation progress, support AbortController cancellation, expose
both PLY and .splat downloads, handle BackgroundRemovalRequiredError clearly,
and dispose the generator on teardown. Reuse one generator instance. Do not
proxy model artifacts through application compute or upload user images by
default. Preserve the package's alpha and browser/WebGPU limitations.
```

Acceptance checks:

- an alpha-bearing image reaches all five generation stages;
- cancellation followed by retry works without a page reload;
- both output blobs are non-empty and use the expected file extensions;
- count and metadata are exposed to the host application;
- opaque input produces a clear background-removal action;
- route/component teardown disposes the generator;
- network inspection shows only configured application/model requests unless remote output storage was deliberately added.

## Validation

For the repository-level checks:

```sh
pnpm typecheck
pnpm lint
pnpm test
pnpm test:package-consumer
```

For a real integration, use a complete immutable manifest in current Chrome or Edge and verify a generated PLY in a compatible viewer plus a `.splat` viewer. Recorded structural passes do not establish final image parity, production memory safety, or support on every WebGPU device.
