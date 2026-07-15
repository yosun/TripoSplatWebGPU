# Browser architecture audit

Reference snapshots used for the initial port:

- `bring-shrubbery/ml-sharp-web` `01ff783f782a0eab1eb0dbb533d51695dc526df6`
- `VAST-AI-Research/TripoSplat` `a78fa12d06dbf1381ca548bfac32bb68cb8c451d`
- `john-rocky/coreai-model-zoo` `ede54d103c191edda7d93b0e6c3c47ea0a0664c0`
- `apple/ml-sharp` `1eaa046834b81852261262b41b0919f5c1efdd2e`

The official TripoSplat Python implementation is the numerical source of truth. The Core AI
conversion is used only to identify useful static graph boundaries.

## 1. Generic WebGPU and ONNX Runtime infrastructure

The reusable portion of `ml-sharp-web` is smaller than its worker name suggests:

- a Vite module worker;
- request IDs, promise correlation and transferable `ArrayBuffer` messages;
- ONNX Runtime Web's WebGPU entry point;
- explicit threaded-WASM asset paths and a WebGPU-then-WASM fallback;
- an in-worker session-promise cache;
- cross-origin isolation headers in the original deployment.

It does not directly manage a `GPUAdapter`, `GPUDevice`, command buffers or WGSL. Before this
refactor, it also had no multi-graph manifest, tensor-neutral RPC, cancellation, session release,
GPU-resident graph chaining or byte-accurate load progress.

The port extracts those reusable pieces under `src/runtime/` and keeps model orchestration in
model adapters. This prevents the generic layer from acquiring either SHARP's camera assumptions or
TripoSplat's sampling schedule.

## 2. SHARP-specific preprocessing and output assumptions

The SHARP path is deliberately retained as the known-good baseline. Its model-specific behavior is:

- stretch the decoded image to `1536 x 1536`;
- construct planar RGB float32 in `[0,1]`;
- infer or accept focal length and pass `focal_px / original_width`;
- expect two raw graph inputs, or five inputs from an older wrapper;
- expect named mean, scale, quaternion, RGB and opacity outputs;
- prune by opacity and optionally cap the result;
- convert screen-aligned/NDC covariance to metric camera space;
- recover anisotropic scale and quaternion using a per-Gaussian Jacobi eigendecomposition;
- write SHARP camera metadata into its PLY.

Those operations are not valid TripoSplat postprocessing. `SharpWebGPUModel` therefore wraps the
existing worker instead of making that worker the generic inference abstraction.

One known parity limitation is preserved rather than hidden: browser `drawImage` resizing does not
numerically match SHARP's PyTorch bilinear interpolation with `align_corners=True`.

## 3. Model loading and external data

The original loader inferred a single sidecar named `<graph>.onnx.data` for every URL ending in
`.onnx`. That works for its SHARP export, but not for a signed CDN URL, multiple shards, or a graph
whose ONNX external-data `location` uses another name.

TripoSplat uses an explicit manifest entry per graph. Each entry records:

- graph URL and stable graph ID;
- zero or more `{ path, url }` external-data objects;
- preferred execution providers and whether WASM fallback is allowed;
- tensor names and fixed shapes in the checked-in model manifest;
- model revision separately from the application deployment.

The normal SHARP export is a small graph plus a roughly 2.4 GiB sidecar. The TripoSplat reference
weights total about 3.52 GiB before ONNX conversion, so they must not be bundled into Vite or proxied
through an application function.

The original single-file model picker cannot load a normal external-data SHARP export. It is kept as
a self-contained-ONNX convenience, not represented as a complete sidecar upload workflow.

## 4. Worker communication

The original protocol exposed `load-model` and `run-inference`, with a SHARP image tensor, camera
scalars, opacity controls and a final PLY. The reusable protocol now deals in:

- `load-graph` and `run-graph` operations;
- named tensors with explicit type, dimensions and transferable storage;
- explicit graph IDs rather than an implicit singleton;
- graph/stage status messages;
- disposal of individual graph sessions and of the worker;
- errors correlated to the originating request.

TripoSplat's higher-level worker/orchestrator owns the DINO/VAE calls, repeated conditional and
unconditional DiT calls, Euler updates, octree graph calls and Gaussian decode. A final optimization
will keep iterative tensors GPU-resident; the correctness path is allowed to transfer them while the
algorithms and parity gates are being established.

## 5. Gaussian viewer and export

Both existing viewers accept compatible 3DGS PLY data and can be retained. Their default orientation
was SHARP-specific, however, and their current blob-URL path serializes then reparses a scene.

The common `GaussianScene` representation records:

- positions, positive scales, `wxyz` rotations, degree-zero SH and opacity when available;
- coordinate system and color semantics;
- binary PLY and the explicitly documented de-facto 32-byte browser `.splat` layout for viewer/download interoperability.

SHARP continues using its camera-rich PLY writer. TripoSplat uses a standard 17-float 3DGS vertex
layout and applies the official object export transform `[[1,0,0],[0,0,-1],[0,1,0]]`. Treating
SHARP's linear RGB as TripoSplat SH coefficients, or applying SHARP's default flips to TripoSplat,
would be a silent rendering bug.

## 6. Deployment and caching

The original Vercel deployment correctly excludes local `.onnx` and `.onnx.data` files and enabled
cross-origin isolation, but it has no service worker, Cache API/OPFS cache, integrity manifest or
resumable model download. Its default hosted SHARP sidecar was observed with byte ranges and CORS but
without an explicit immutable cache policy.

The tested Chromium 150 runtime terminated the ONNX WebGPU worker when COOP/COEP was enabled and ran
the same graph successfully without it. This port therefore does not require cross-origin isolation
for WebGPU. The optional WASM fallback must remain single-threaded in that mode; any future attempt
to restore threaded WASM must repeat the WebGPU parity gate under the exact production headers.

The deployment contract for this port is:

- Vercel or another static host serves only the application;
- S3 plus CloudFront, or GCS plus Cloud CDN, serves versioned model objects;
- model responses allow the application origin, `GET`, `HEAD` and byte ranges;
- model responses may include `Cross-Origin-Resource-Policy: cross-origin`, but the tested app does
  not send `Cross-Origin-Embedder-Policy`;
- content-addressed/versioned objects use `Cache-Control: public,max-age=31536000,immutable`;
- manifests use short caching and point to a fixed model revision;
- user images remain local `File`/`Blob`/`ImageBitmap` data and are never uploaded.

## TripoSplat graph contracts

| Stage | Inputs | Outputs |
| --- | --- | --- |
| DINOv3 | RGB/ImageNet-normalized FP32 `[1,3,1024,1024]` | `feature1` FP32 `[1,4101,1280]`, including the official final non-affine layer norm |
| Flux VAE encoder | RGB `[0,1]` `[1,3,1024,1024]`, epsilon `[1,32,128,128]` | padded feature2 `[1,4101,128]` |
| one DiT invocation | latent `[1,8192,16]`, camera `[1,1,5]`, scaled time `[1]`, feature1, feature2 | latent and camera velocity |
| octree probability | padded centers `[1,8192,3]`, resolution `[1]`, latent condition | child logits `[1,8192,8]` |
| Gaussian transformer | points `[1,P,3]`, latent condition | features `[1,P,480]` |

The official VAE samples stochastically. The browser graph takes epsilon explicitly; copying the Core
AI port's deterministic-mean wrapper would not be faithful. The official 20-step run with guidance
scale 3 performs 40 DiT invocations, not 20.

## Memory constraint discovered during audit

Naively lowered attention is a deployment blocker for a 16 GiB target. A single unfused FP16 score
matrix is approximately 512 MiB in the Flux VAE mid-block and 4.5 GiB in a full 12,294-token DiT
joint-attention block. Export validation must therefore inspect the generated ONNX and browser trace
for a memory-efficient attention implementation or introduce query chunking/custom WebGPU kernels.
This is a required correctness/memory milestone, not a speculative performance optimization.

## Source/licensing note

TripoSplat source and released weights state MIT licensing. Apple SHARP weights have separate
research-use terms. The `ml-sharp-web` snapshot used as the chassis has no detected root license;
redistribution of inherited source should be clarified before publishing this derived repository.
