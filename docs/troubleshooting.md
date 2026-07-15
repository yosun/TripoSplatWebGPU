# Troubleshooting

## `@ai3d/triposplat-webgpu` cannot be resolved

That package is not published at the current milestone. Build and consume the workspace package or its packed tarball from this checkout. The built-in executor and complete local manifest exist, while the repository labs remain the authoritative numerical validation surfaces.

## OPFS model loading reports insufficient quota

OPFS is the default and the package checks `navigator.storage.estimate()` before downloading the canonical 6.465 GB artifact set. Clear old model revisions or free disk space when the declared missing bytes exceed available origin quota. `cache: 'none'` still verifies every object and uses temporary OPFS staging when available so multi-gigabyte downloads are not accumulated as renderer `ArrayBuffer[]` chunks.

## `generate()` throws `GraphCapabilityError`

The built-in executor requires all five canonical graph entries. Confirm the manifest uses the exact roles `dino`, `vae`, `dit`, `octree`, and `gaussianDecoder`; a custom pipeline remains optional for advanced integrations.

## A graph returns 404 for `.onnx.data`

Serve the ONNX graph and every external-data file. The manifest `path` must match the graph's embedded external-data `location`; the fetch `url` can be different. Do not derive a sidecar URL by appending `.data` to a signed graph URL.

## External-data fetch is blocked by CORS

Check the final CDN response after redirects. It must allow the app origin and the required `GET`, `HEAD`, `OPTIONS`, `Range`, and authorization headers. Expose `Content-Length`, `Content-Range`, `Accept-Ranges`, and `ETag` if the cache/downloader reads them.

## A WASM request returns HTML

An error such as “expected magic word … found `3c 21 64 6f`” means a `.wasm` URL returned an HTML page. Run through Vite rather than `file://`, verify files under `/ort/`, and inspect the response content type and body.

## The four-step lab says execution completed but parity failed

Check which artifact and precision produced the record. The current fp32 four-step path passes its qualification envelope but fails a stricter diagnostic; the older fp16 record failed its declared gate. Do not erase either result or loosen a threshold without output-sensitivity evidence.

## The 20-step lab completes but reports failure

The measured fp32 run completes all 40 conditional/unconditional WebGPU calls, but its final latent has a 0.0487 worst error, 0.999999659 cosine similarity, and 0.998329 fraction within the qualification tolerance. This is accumulated numerical drift, not an incomplete control loop.

## Generation says a TripoSplat graph is not configured or returns 404

The canonical manifest contains all five fp32 graphs and the built-in executor has completed a prepared-image structural generation pass. A capability error now means the configured manifest is incomplete or malformed. Whole-scene numerical/render parity is still unqualified, so successful execution is not yet a production-quality claim.

## WebGPU is unavailable

Use a current desktop Chrome or Edge build and verify `navigator.gpu` in a secure context (`https://` or localhost). Browser support does not guarantee that a usable adapter/device can be requested. Use `TripoSplatWebGPU.checkCompatibility()` for a structured adapter report; a successful report is not a memory or model-execution qualification.

## The page crashes or the GPU process resets

Close other memory-heavy tabs, reload the worker, and capture the exact graph, browser, adapter limits, and stage. Do not infer “out of memory” solely from a crash. Chromium does not expose reliable free GPU memory. A production package should translate confirmed allocation failures into `OutOfMemoryError` with diagnostics.

## First load is slow

Separate network download, integrity verification, session creation, graph compilation, and inference. The recorded canonical fp32 DiT transfer is 1,643,895,982 bytes and its measured cold session load was 12,584.6 ms on one 128 GB M3 Max machine. That does not predict another device or network.

## A second generation fails with a detached buffer

Worker messages transfer `ArrayBuffer` ownership. Reusing a transferred input without cloning produces zero-length or detached views. Recreate per-invocation tensors, especially the conditional and unconditional DiT inputs, and add a regression test covering both calls.

## PLY orientation or color is wrong

Confirm the model's coordinate convention, quaternion order, scale/opacity encoding, and RGB/SH semantics. TripoSplat uses an object export rotation mapping `(x,y,z)` to `(x,-z,y)`. SHARP camera transforms are not valid for TripoSplat.

## `pnpm test` cannot resolve the Gaussian package output

Build the workspace helper first:

```bash
pnpm --filter @ai3d/gaussian-scene build
pnpm test
```

CI performs this package build before root validation.
