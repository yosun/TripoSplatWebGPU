# Developer guide

This guide describes the current repository workflow. It is not a public npm installation guide: the alpha `@ai3d/triposplat-webgpu` workspace package is not published. The package now includes built-in all-stage orchestration, while complete official output parity and target-hardware qualification remain under development.

## Requirements

- Node.js 22 or later;
- Corepack;
- pnpm 11.13.0, as pinned in `package.json`;
- recent desktop Chrome or Edge with WebGPU;
- enough local storage for the selected ONNX graph, its external data, and deterministic fixtures;
- Python 3.10–3.12 plus the pinned exporter dependencies when regenerating graphs.

## Install and validate

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm --filter @ai3d/gaussian-scene build
pnpm typecheck
pnpm test
pnpm test:package-consumer
```

The explicit Gaussian package build is required because root tests consume its generated workspace output. `test:package-consumer` packs both packages, installs the tarballs into clean Vite, strict Next-client, and native-ESM fixtures, and verifies that the Vite production build includes the TripoSplat worker and ONNX Runtime WASM asset. Browser launch is a separate step: the recorded packed Vite smoke executes a tiny identity graph through WebGPU, not the full model.

To run the browser labs:

```bash
pnpm dev
```

Open the exact URL printed by Vite, then choose one of:

- `encoder-lab.html` for preprocessing and VAE comparison;
- `dino-lab.html` for preprocessing and the full fp32 DINOv3 feature tensor;
- `dit-lab.html` for one fixed DiT invocation;
- `flow-lab.html` for a fixture-declared four-step or 20-step loop;
- `octree-lab.html` for one fp32 occupancy-logit invocation or the full eight-level neural/host trajectory;
- `gaussian-lab.html` for one fp32 raw 480-feature decoder invocation;
- `e2e-lab.html` for complete prepared-image package execution plus structural scene/export checks;
- the root page for the preserved SHARP path.

The component labs report narrow numerical gates. The E2E lab reports complete execution, lifecycle, finiteness, lengths, export sanity, and exported-PLY viewer/canvas readiness while explicitly setting `numericalParityClaimed: false`; do not reinterpret it as official whole-scene or rendered-pixel parity.

## Model and fixture setup

Generated weights, ONNX sidecars, and binary fixtures are intentionally not npm package material. Export scripts under `scripts/triposplat/` load weights from a local official TripoSplat checkout; they do not implicitly download checkpoints or upload source images.

The exporter/validator pairs under `scripts/triposplat/` cover DINOv3, Flux VAE, DiT, flow teacher trajectories, octree occupancy/full trajectory replay, Gaussian feature decoding, and the official Gaussian activation oracle. Fixture packers create browser-readable arrays without placing model weights in npm packages. [`scripts/triposplat/README.md`](../scripts/triposplat/README.md) is the complete tool index; use each script's `--help` as its authoritative command surface.

Each export should record:

- official repository URL and commit;
- whether the official checkout was clean;
- checkpoint source and checksum;
- public tensor names, dtypes, and fixed shapes;
- ONNX and external-data checksums;
- exporter/runtime versions;
- adapter-versus-official and ONNX-versus-official comparisons.

The official TripoSplat checkout at commit `a78fa12d06dbf1381ca548bfac32bb68cb8c451d` is the current numerical reference. The Core AI decomposition is useful only for graph boundaries.

The current correctness manifest uses fp32 artifacts for all five neural graphs. The Core AI notes about real-valued RoPE, static positional data, explicit stable normalization, and large-graph optimization are treated as hypotheses and re-gated against official PyTorch; they are not accepted solely because they worked in Core AI. The explicit stable-RMS rewrite was measured and failed the same unconditional teacher calls, so it remains an unpromoted diagnostic graph. Fresh CPU ORT reproduces most of the same zero-context failure, while paired adapted PyTorch remains much closer to official output. The 39-boundary report localizes the first material split to `context_refiner.0`'s attention residual; use `validate_dit_onnx.py --trajectory-fixture-dir ... --trajectory-invocation ...` for exact invocation inputs and `validate_dit_block_probes.py` for the internal boundary gate.

## Graph loading

The generic worker accepts explicit graph and external-data descriptors. External data is not inferred safely from arbitrary signed URLs. The `path` field must match the `location` embedded in the ONNX graph, while the fetch URL may be absolute, relative, or signed.

The high-level package resolves all artifact URLs from the manifest, streams them through declared byte-length and SHA-256 checks, and creates short-lived Blob URLs for the worker. OPFS is the production default and is preflighted against the browser's origin-storage estimate; `cache: 'cache-api'` provides persistent Cache API storage, while `cache: 'none'` uses verified temporary OPFS staging when possible. Cache entries are rehashed on first read per manager lifecycle, then the verified file-backed Blob is reused without a duplicate hash before session creation. Corrupt entries are evicted, and cache identity uses the immutable artifact digest rather than signed query parameters. Unit tests cover these transitions; real production-origin CDN/OPFS stress testing remains a release gate.

The built-in package executor stages graph sessions to reduce residency:

1. load and run encoder sessions;
2. dispose encoders before loading the DiT;
3. dispose the DiT before octree decode;
4. dispose the octree before Gaussian decode;
5. terminate the worker on model disposal.

The correctness path currently transfers iterative tensors between worker and main thread. The eight-level fp32 octree trajectory now passes against official logits, sampling decisions, random consumption, and final points. GPU-resident chaining and WGSL octree kernels remain performance work that must preserve this gate.

## Numerical change policy

Any change to preprocessing, random sampling, timestep shift, classifier-free guidance, Euler updates, Q/K normalization, RoPE, attention chunking, octree resampling, Gaussian activation, or export orientation requires a deterministic comparison with untouched official PyTorch.

Do not loosen a threshold merely to turn a failing run green. Record the old and new errors, explain the numerical cause, and choose a threshold from output sensitivity rather than implementation convenience.

## Current package boundary

`@ai3d/gaussian-scene` is a framework-neutral workspace package with ESM, declarations, source maps, deterministic validation, disposal, binary 3DGS PLY export, and the tested de-facto 32-byte browser `.splat` layout. The alpha browser-only `@ai3d/triposplat-webgpu` package exposes:

- `TripoSplatWebGPU`, the built-in staged pipeline, cache controls, manifest/capability helpers, typed errors, and public types from the root entry;
- `createRuntime`, `loadGraph`, `runGraph`, artifact manager, sampler, dynamic octree, decoder, tensor, and manifest helpers from `./low-level`;
- Gaussian scene helpers plus `exportPLY` and `exportSplat` from `./export`.

The alpha class does not fabricate missing graph stages. `generate()` requires a complete five-graph manifest and uses the built-in DINO/VAE/DiT/octree/Gaussian executor by default; `pipeline` remains an advanced override. A browser-local `removeBackground` hook accepts a validated segmenter for opaque input, but the package does not bundle a WebGPU-qualified BiRefNet artifact. OPFS and Cache API persistence, integrity verification, custom model fetch, cache status, and cache clear are implemented.

The packed tarballs pass clean Vite, Next-client, and native-ESM consumer checks, including worker/WASM emission where applicable. The exact final packed Vite build reaches WebGPU through its installed worker on a tiny identity graph and then completes the full prepared-image five-graph path, yielding 262,144 finite Gaussians and both exports. Separately, the workspace E2E lab adds a ready PLY viewer/canvas gate. These are structural execution results, not whole-scene numerical/render parity. The 20-step browser run still fails its accumulated-drift gate; actual Next/native-ESM full runs, repeated generation, Edge, and 16 GB qualification remain before the public quick start can be called production-ready.

The built-in worker retains each immutable positive/negative DiT conditioning set once for the DiT session, and each invocation transfers only latent, camera, and timestep inputs. Custom runtimes retain the ordinary cloned-input behavior. The measured byte-accounting reduction is documented in the E2E report; do not describe it as a measured latency or GPU-residency improvement.

For target consumer code, see [`examples/`](../examples/README.md). Treat the examples as API fixtures until a deployed immutable model manifest and the release gates in [Current status](current-status.md) are satisfied.
