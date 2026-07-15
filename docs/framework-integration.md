# Framework integration

The core package target is browser-only and framework-neutral. It must not import React, Next.js, Vite, or a viewer. Worker resolution uses a module-relative URL and allows an explicit override for restrictive bundlers. The emitted-worker path has completed a full five-graph run from an exact tarball installed into a fresh Vite consumer; Next-style client and native-ESM fixtures currently validate their package/type/import surfaces but have not run the full model in a browser.

See the [alpha API reference](api-reference.md) for current entry points and fail-fast boundaries.

> The examples below use the implemented alpha contract. The package now has a built-in executor, persistent and ephemeral verified cache backends, and a complete five-graph manifest. It remains unpublished, and complete browser execution from a packed consumer is still a release gate.

## Vanilla browser application

Create and dispose one model instance for the lifetime of the feature, decode the selected file locally, and revoke every object URL. See [`examples/vanilla/main.js`](../examples/vanilla/main.js).

Plain browser ESM still needs an import map or an ESM CDN that has a published package. No such CDN artifact exists today.

## Vite

The intended package should work through a normal dependency import; model files should remain on a model CDN rather than under Vite's `public/` directory. See [`examples/vite/main.ts`](../examples/vite/main.ts).

The package validation job installs packed tarballs into a fresh Vite app and verifies the production bundle, emitted module worker, and ONNX Runtime WASM asset. A separate recorded Chrome smoke executes a tiny identity ONNX graph through that installed worker with WebGPU. This proves package asset resolution and provider selection, not full TripoSplat inference; see the [packed Vite browser record](validation/2026-07-15-packed-vite-consumer-webgpu-chrome.json).

## React

Store the model in a ref, not component state. Dispose it in the effect cleanup and abort active work before replacing an input or unmounting. Dispose the previous `GaussianScene` before publishing a new one. See [`examples/react/TripoSplatPanel.tsx`](../examples/react/TripoSplatPanel.tsx).

React Strict Mode intentionally exercises mount/unmount behavior during development. Initialization and disposal must therefore be idempotent.

## Next.js

Inference belongs in a client component. Do not run compatibility checks, create workers, or access `navigator.gpu` in a Server Component or route handler. See [`examples/next/app/triposplat/TripoSplatClient.tsx`](../examples/next/app/triposplat/TripoSplatClient.tsx).

The model CDN is independent of Next.js asset hosting. Do not create an API route that downloads and relays multi-hundred-megabyte model files.

## Signed model access

The package supports a custom fetch function, per-artifact request settings, and absolute signed URLs while keeping the ONNX logical external-data path unchanged. See [`examples/signed-fetch/load.ts`](../examples/signed-fetch/load.ts).

Avoid putting reusable bearer tokens in query strings or logs. If a signed URL expires during a resumed transfer, obtain a new URL for the same content hash and validate the completed bytes.

## Cancellation and retry

Pass an `AbortSignal` to both `load` and `generate`. Cancellation during an ONNX call terminates the package worker and resets runtime state so the same high-level instance can load and retry. The public error is a typed `CancelledError`. See [`examples/cancellation/generate.ts`](../examples/cancellation/generate.ts).

## Export and viewer integration

The canonical output owns typed arrays and can export a binary PLY `Blob`. Create a short-lived object URL for download or pass the blob/arrays to an optional viewer adapter, then revoke the URL and dispose the scene. See [`examples/export/download.ts`](../examples/export/download.ts) and [Gaussian conventions](gaussian-conventions.md).

`exportSplat()` uses the explicitly documented de-facto 32-byte browser layout: float32 position and linear scale, byte RGBA, and quantized `wxyz` quaternion. Tests include a byte-identical fixture from the official TripoSplat exporter. Other files called “`.splat`” may use different conventions; name and test the exact viewer format rather than relying on the extension alone.
