# Direct and agent integration

Use this guide when adding local image-to-3D generation to an existing web product. The stable application boundary is intentionally small:

```text
Blob/File/ImageBitmap/ImageData/canvas
  -> WebGPU browser generation
  -> { ply: Blob, splat: Blob, metadata, count }
```

The reusable implementation is [`examples/direct/image-to-splat.ts`](../examples/direct/image-to-splat.ts). It accepts browser image sources, exports both portable formats, and disposes the large in-memory Gaussian scene deterministically.

> **Runtime boundary:** inference runs in a current WebGPU-capable Chromium browser. Node.js, serverless functions, React Server Components, and edge workers may provide configuration or host assets, but they must not call `generate()`. Do not upload a user's image to a backend unless the product explicitly requires and discloses that behavior.

## Five-minute onboarding

1. Install the alpha tarballs as described in the [package README](../packages/triposplat-webgpu/README.md), or install `@ai3d/triposplat-webgpu` after publication.
2. Copy `examples/direct/image-to-splat.ts` into the client-side portion of the app.
3. Host the complete, immutable model manifest and artifacts on an object CDN. This is currently about 6.465 GB; do not proxy it through an application function.
4. Construct one `ImageToSplatGenerator` per browser tab and reuse it for queued requests.
5. Pass an alpha-bearing image, await `generate()`, and save or upload the returned PLY/`.splat` blobs according to the host product's policy.
6. Call `dispose()` when the page/component is torn down.

```ts
import { ImageToSplatGenerator } from './image-to-splat'

const generator = new ImageToSplatGenerator({
  modelBaseUrl: 'https://cdn.example.com/triposplat/v1/',
  cache: 'opfs',
})

const report = await ImageToSplatGenerator.checkCompatibility()
if (!report.supported) throw new Error(report.blockers.join('\n'))

const output = await generator.generate(file, {
  steps: 4,
  seed: 42,
  onProgress: ({ message }) => console.info(message),
})

saveBlob(output.ply, 'object.ply')
saveBlob(output.splat, 'object.splat')
await generator.dispose()
```

## Inputs

The package accepts `Blob`, `File`, `ImageBitmap`, `ImageData`, `HTMLImageElement`, `HTMLCanvasElement`, and `OffscreenCanvas`. Prefer a `File`/`Blob` from a local picker or drag-and-drop surface; no base64 conversion is needed.

An alpha-bearing subject image works directly. A normal opaque JPEG does **not** contain a foreground mask and fails with `BackgroundRemovalRequiredError`. Until a qualified remover is bundled, either:

- request a transparent PNG/WebP from the user;
- provide a validated browser-local `removeBackground` callback in the constructor; or
- pass `inputIsPrepared: true` only for the exact preprocessed 1024×1024 opaque RGB-on-black representation.

Never set `inputIsPrepared` merely to bypass the error for a raw photograph.

## Outputs

`ImageToSplatResult` owns two browser `Blob`s and JSON-compatible metadata:

| Field | Meaning |
| --- | --- |
| `ply` | Binary little-endian 17-float 3DGS PLY |
| `splat` | De-facto 32-byte browser `.splat` records |
| `count` | Number of exported Gaussians; currently 262,144 |
| `metadata` | Coordinate, encoding, model revision, seed, and generation settings |
| `elapsedMs` | Generation plus both encodes |

Use `blob.arrayBuffer()` for APIs that need bytes, `URL.createObjectURL(blob)` for browser downloads/viewers, or send the blob to object storage only when remote persistence is an intentional product feature. The `.splat` extension is ambiguous across tools; consumers must support the layout documented in [Gaussian conventions](gaussian-conventions.md).

## Platform recipes

### Vite, Vue, Svelte, Angular, and plain bundled apps

Import the adapter only from client code. Keep one instance in application scope, generate from event handlers, and dispose it on application teardown. Vite resolves the package worker automatically.

### React

Create the generator in a client component or provider, retain it in a ref, and dispose it in the effect cleanup. Keep output blobs or object URLs in state; never put the canonical scene's typed arrays in React state.

### Next.js

Place the adapter behind a file containing `'use client'`, or dynamically import the client component with SSR disabled. A Route Handler may return the public model URL, but generation cannot run in a Server Component or Route Handler.

### Server-rendered and multi-tenant products

Expose only non-secret public configuration to the client. [`examples/direct/model-config-server.mjs`](../examples/direct/model-config-server.mjs) shows a dependency-free Node endpoint:

```sh
TRIPOSPLAT_MODEL_BASE_URL=https://cdn.example.com/triposplat/v1/ \
  node examples/direct/model-config-server.mjs
```

Mount `createTripoSplatConfigHandler()` in an existing Node server, or implement the same JSON response in another backend. Do not expose signing secrets. Use the package's custom `fetch`, request-init options, or short-lived signed artifact URLs when model access is private.

### Native mobile and desktop shells

Run the client integration inside a Chromium WebView that exposes WebGPU, module workers, Canvas/OffscreenCanvas, and adequate storage. A native Node/Electron main process cannot substitute for the browser renderer. Always execute `checkCompatibility()` on the target shell.

## Model hosting requirements

- Preserve every external-data path exactly as declared by its ONNX graph and manifest.
- Serve immutable versioned files from an object CDN with CORS, byte ranges, and correct content lengths.
- Allow the app origin to `GET` the manifest, graph files, and sidecars.
- Prefer `cache: 'opfs'`; the first run downloads and verifies the declared artifacts.
- Do not bundle model payloads into npm packages, serverless functions, or the application JavaScript bundle.
- Treat the model URL as public unless access is enforced with short-lived browser-compatible authorization.

See [Model hosting](model-hosting.md) for deployment details.

## Error and lifecycle contract

- Call `checkCompatibility()` before a large model download and present its blockers to the user.
- Use one `AbortController` per request. Cancellation terminates unsafe in-flight worker state; the same generator can retry and lazily creates a clean worker.
- `BackgroundRemovalRequiredError` is an input/action error, not a WebGPU failure.
- Show load and generation progress; initial verified artifact loading can be long.
- Generation calls on one model are serialized. Implement a visible queue or cancel/replace policy rather than creating concurrent model instances.
- Keep returned blobs only as long as needed. Revoke object URLs created for previews/downloads.
- Call `dispose()` exactly once during teardown; it is idempotent.

## Instructions for coding agents

When adapting this example to another codebase, preserve these invariants:

1. Put every import and `TripoSplatWebGPU` call behind the platform's browser/client boundary.
2. Accept native browser image objects; do not invent a base64-only API.
3. Run compatibility checking and expose cancellation plus progress.
4. Return or download **both** PLY and `.splat` unless the product explicitly requests one format.
5. Export before disposing the scene. Prefer the supplied adapter, which already guarantees this ordering.
6. Reuse one generator and dispose it on teardown; do not construct a model per render or request.
7. Do not claim opaque JPEG support unless a real local background-removal callback is configured.
8. Do not proxy the 6+ GB model through application compute, and do not upload input images by default.
9. Keep alpha-status and model-quality warnings visible while the package remains pre-release.
10. Validate the final integration in a real target browser against a complete immutable manifest.

A useful implementation request for another agent is:

```text
Integrate @ai3d/triposplat-webgpu using docs/agent-integration.md and
examples/direct/image-to-splat.ts. Keep inference client-only, reuse one
ImageToSplatGenerator, accept File/Blob input, show compatibility/progress/
cancellation, expose both PLY and .splat downloads, handle opaque-input errors,
and dispose resources on teardown. Do not proxy models or upload user images.
```

Acceptance checks:

- an alpha-bearing image reaches all five generation stages;
- cancel followed by retry works without a page reload;
- both non-empty blobs download with the expected extensions;
- the reported count and metadata are available to the host app;
- opaque input gives a clear background-removal action;
- page/component teardown disposes the generator;
- network inspection shows only configured app/model requests unless remote output storage was deliberately added.

This remains an alpha integration. Review [Current status](current-status.md) before making production quality, parity, memory, or cross-browser claims.
