# Current status and release gates

Recorded 2026-07-15. This document is the source of truth for what the repository can demonstrate today. A code path existing in TypeScript is not treated as numerically complete until its graph artifact exists and its output has been compared with the untouched official TripoSplat PyTorch implementation.

## Milestone summary

Strict fp32 browser slices now pass for image preprocessing plus DINOv3, Flux VAE, and one DiT invocation. The four-step TypeScript CFG/Euler loop executes all eight expected conditional and unconditional DiT calls on WebGPU. Its final fp32 state passes the recorded qualification envelope, but the latent misses a separately recorded stricter diagnostic. Teacher-forced replay now narrows that failure: every conditional call passes the strict gate, while the zero-conditioning call passes at step 1 and fails from step 2 onward. Both final-state outcomes and the per-invocation failure remain visible; the qualification pass does not rewrite the strict failure.

The browser package now contains a built-in DINO → VAE → staged DiT flow → dynamic octree → Gaussian decode executor. It also streams declared model artifacts through byte-length and SHA-256 verification, with `none`, Cache API, and OPFS storage modes and public inspection/clear APIs. These are real package capabilities, not caller-supplied placeholders.

The complete prepared-image package path now executes in Chrome/WebGPU: it stages all five verified graphs, runs the four-step pipeline, returns 262,144 finite Gaussians, produces valid-size PLY and `.splat` exports, and loads the PLY into a ready viewer canvas. This is a structural/export/viewer-load pass, not a whole-scene numerical or rendered-pixel comparison with official PyTorch. A `removeBackground` hook exists, but no bundled WebGPU-qualified BiRefNet artifact does; the measured 20-step browser loop exceeds its qualification envelope; and final whole-scene/render parity remains open. Deterministic lifecycle coverage now includes serialized generation, prompt queued cancellation, abort-and-retry worker recreation, failed-load ownership, and dispose during pending configuration. Production-origin CDN/OPFS behavior, repeated full-model browser generation, Microsoft Edge, and the 16 GB target remain unvalidated.

## Stage ledger

| Stage | Implementation state | Validation state | Release consequence |
| --- | --- | --- | --- |
| Image preprocessing | Browser implementation, deterministic fixture, and browser-local `removeBackground` hook exist | Exact prepared-image bytes pass; DINO and VAE fixtures use the browser result | Prepared RGBA/RGB-on-black input is supported; no WebGPU-qualified BiRefNet artifact is bundled |
| DINOv3 encoder | Full fp32 ViT-H graph, browser preprocessing, worker contract, and local artifact exist | **STRICT PASS** in Chrome/WebGPU against official PyTorch | First conditioning branch is qualified on the recorded machine |
| Flux VAE encoder | Fixed-shape export, explicit epsilon, browser graph, and comparison lab exist | **PASS** against PyTorch fixture | First vertical slice complete |
| One-step DiT | Fixed-shape fp32 graph with real RoPE, static Sobol positional data, chunked attention/output projection, and Q/K padding workaround exists | **STRICT PASS** in Chrome/WebGPU against untouched PyTorch | Single-invocation fp32 gate complete |
| Four-step flow | TypeScript schedule, fp32 PyTorch-order CFG/Euler arithmetic, cancellation checks, final-state lab, teacher-forced replay, and 39 block-boundary probes exist | Eight WebGPU calls complete; **qualification PASS, stricter diagnostic FAIL**. The first material invocation-7/8 split is the `context_refiner.0` attention residual | Fast path executes, but the exported/ORT zero-context self-attention reduction remains a strict-parity defect |
| Twenty-step flow | Official 20-step control path and fp32 fixture exist | All 40 WebGPU calls complete; **qualification FAIL and strict FAIL** from accumulated latent drift | A quality-path execution result exists, but no parity pass is claimed |
| Octree occupancy | Correctness-first TypeScript traversal, systematic resampling, compaction/expansion, jitter replay, plus fp32 ONNX graph exist | **PASS** across all eight official Chrome/WebGPU frontiers: active logits, padding independence, sampled counts, random consumption, and final points | Full octree boundary is qualified on the recorded machine; live downstream Gaussian scene remains open |
| Gaussian decoder | fp32 ONNX graph and host decode/export code exist | Raw `[1,8192,480]` Chrome/WebGPU boundary **passes**; a small official `_build_gaussians` activation oracle passes; the live E2E run produces finite activated arrays | Whole-scene comparison against official PyTorch remains open |
| Canonical Gaussian scene | Framework-neutral workspace package exists and the built-in pipeline returns its contract | Unit tests cover validation/disposal/exports; the E2E run returns 262,144 finite Gaussians, valid-size PLY/`.splat` files, and a ready viewer canvas | Scene execution/export/viewer loading is demonstrated; official whole-scene and rendered-pixel parity is not established |
| Alpha browser package | High/low-level API, built-in staged executor, typed errors, manifest/runtime/sampler/octree/decode/export surfaces exist | Real Chrome/WebGPU prepared-image path **STRUCTURAL PASS**; exact final tarball installed into fresh Vite and completed the full five-graph path; packed Next-client/native-ESM checks pass | Workspace and packed-Vite execution are demonstrated; official numerical/visual parity, full Next/native-ESM browser execution, and target-hardware qualification remain open |
| Model delivery/cache | Canonical 6,465,182,402-byte manifest supports multiple sidecars, SHA-256, custom fetch, Cache API, OPFS, and disk-backed ephemeral mode | Unit coverage includes cache hits, corruption eviction, signed-URL refresh, cancellation, quota failure, namespace clear, and public status; packed Vite includes worker/WASM assets | Real-browser production CDN and OPFS stress qualification remains open |
| Viewer integration | Preserved SHARP viewer consumes the exported TripoSplat PLY | The E2E gate reached viewer-ready state with a 1916×954 drawing buffer | This is a renderer-load/canvas sanity result; TripoSplat visual parity remains unproven |

## Recorded numerical gates

### DINOv3 browser slice — strict pass

- execution provider: WebGPU only, fallback disabled;
- fp32 graph plus sidecar transfer: 3,367,847,196 bytes;
- cold model/session load: 8,702.1 ms;
- one inference: 4,262.4 ms;
- readback: 2.4 ms;
- preprocessing maximum error: 0;
- feature maximum error: 0.0001943950;
- feature mean absolute error: 0.0000014715;
- cosine similarity: 0.9999999999966;
- declared gate: `atol=0.0003`, `rtol=0.001`, minimum cosine 0.999999999.

The full `[1,4101,1280]` feature tensor passed on the recorded Chrome/WebGPU machine. Evidence: [`docs/validation/2026-07-15-dinov3-fp32-webgpu.json`](validation/2026-07-15-dinov3-fp32-webgpu.json) and [`docs/validation/2026-07-15-dinov3-fp32-onnx.json`](validation/2026-07-15-dinov3-fp32-onnx.json).

### Flux VAE browser slice — pass

- execution provider: WebGPU only;
- transfer size: 137,800,544 bytes;
- median inference: 1,304.1 ms over three recorded runs;
- median source preprocessing plus encode: 1,601.7 ms;
- maximum tensor error: 0.0001683235;
- cosine similarity: 0.9999999997;
- declared gate: maximum error at most 0.006 and cosine at least 0.99999.

Evidence: [`docs/benchmarks/2026-07-14-flux2-vae-webgpu.json`](benchmarks/2026-07-14-flux2-vae-webgpu.json).

### One fp32 DiT invocation — strict pass

- execution provider: WebGPU only, fallback disabled;
- graph plus sidecar transfer: 1,643,895,982 bytes;
- cold model/session load: 12,584.6 ms;
- one inference: 11,844.6 ms;
- readback: 0.9 ms;
- latent maximum error: 0.0000057220;
- latent cosine similarity: 0.9999999999996;
- camera maximum error: 0.0000017881;
- camera cosine similarity: 0.9999999999997;
- declared gate: `atol=0.00002`, `rtol=0.001`, minimum cosine 0.999999999.

Evidence: [`docs/validation/2026-07-15-dit-step-webgpu-fp32-chrome.json`](validation/2026-07-15-dit-step-webgpu-fp32-chrome.json) and [`docs/validation/2026-07-15-dit-step-webgpu-fp32-onnx.json`](validation/2026-07-15-dit-step-webgpu-fp32-onnx.json).

The earlier fp16 graph and its looser gate remain useful historical records, but the fp32 WebGPU-safe graph is the current correctness artifact.

### Four-step fp32 CFG/Euler loop — qualification pass, stricter diagnostic fail

- four shifted Euler steps;
- guidance scale 3 and shift 3;
- four conditional and four unconditional WebGPU invocations;
- model load: 11,411.3 ms;
- sampling wall time: 102,268.8 ms;
- latent maximum error: 0.0044521093;
- latent cosine similarity: 0.9999999885;
- camera maximum error: 0.0000410676;
- camera cosine similarity: 0.9999999986.

The recorded qualification envelope is `atol=0.005`, `rtol=0.003`, minimum cosine 0.99999998. Both latent and camera have a fraction within tolerance of 1.0 and pass that envelope.

The separately recorded stricter diagnostic is `atol=0.0001`, `rtol=0.001`, minimum cosine 0.99999999. The camera passes, but only 0.9829483 of latent values fall within its combined tolerance, so the strict diagnostic fails. Evidence: [`docs/benchmarks/2026-07-15-flow4-fp32-webgpu.json`](benchmarks/2026-07-15-flow4-fp32-webgpu.json).

This is one single-machine run. It is not evidence for final visual fidelity or the 16 GB target.

### Four-step teacher-forced DiT trajectory — conditional pass, unconditional failure

The teacher fixture records the official PyTorch input state, scaled timestep, conditioning branch, and raw prediction for each of the four conditional and four unconditional calls. Replaying those states directly through the browser graph removes CFG/Euler state accumulation from the comparison.

- all four conditional latent and camera outputs pass the strict gate;
- the step-1 unconditional call passes;
- unconditional latent calls fail at steps 2, 3, and 4;
- every camera output passes;
- failing unconditional latent maximum errors grow from 0.000330925 at step 2, to 0.000472546 at step 3, to 0.000844243 at step 4;
- their fractions within combined tolerance fall from 0.9994812, to 0.9990692, to 0.9958420;
- strict gate: `atol=0.0001`, `rtol=0.001`, minimum cosine 0.99999999, with every value required inside combined tolerance;
- eight WebGPU invocations completed without fallback in 108,012.6 ms wall time after a 12,231.8 ms load.

This rules out “only autoregressive Euler accumulation” as the explanation. It localizes the current strict-flow defect to invocation-dependent graph behavior, especially the all-zero conditioning branch. Evidence: [`docs/validation/2026-07-15-flow4-teacher-forced-webgpu-fp32-chrome.json`](validation/2026-07-15-flow4-teacher-forced-webgpu-fp32-chrome.json).

An explicit stable-RMS export rewrite was tested as a diagnostic hypothesis inspired by the Core AI conversion notes. It passed the Python three-way export gate but failed the same browser teacher-forced calls: unconditional maximum errors at steps 2–4 were 0.000346661, 0.000476122, and 0.000827583. The canonical values were 0.000330925, 0.000472546, and 0.000844243, so the rewrite is not a material improvement and was not promoted. Evidence: [`docs/validation/2026-07-15-flow4-teacher-forced-stable-rms-webgpu-fp32-chrome.json`](validation/2026-07-15-flow4-teacher-forced-stable-rms-webgpu-fp32-chrome.json).

A paired fresh-process diagnostic uses the exact same step-4 latent, camera, and timestep for invocation 7 (conditional) and invocation 8 (zero-conditioned). CPU ONNX Runtime passes the conditional latent at 0.000043750 maximum error but fails the unconditional latent at 0.000764251; Chrome's corresponding unconditional maximum is 0.000844243. The adapted PyTorch graph's unconditional maximum against untouched official PyTorch is only 0.000071019. This shows that most of the failure reproduces without WebGPU or session reuse and is introduced when the adapted attention graph executes through ONNX reductions. The fixed fp32 Sobol initializer, real-valued RoPE primitive, and special output projection are separately ruled out as primary causes: the RoPE primitive maximum is `2.384e-7`, and an earlier ordinary-linear projection graph produces a nearly identical 0.000745058 unconditional maximum.

A 39-boundary invocation-7/8 probe now finds the first material split at `context_refiner.0`'s self-attention residual, before its MLP or any joint block. At that boundary, CPU ORT versus official PyTorch has `3.0994e-6` conditional maximum error and `1.1063e-4` unconditional maximum error, a 35.69× separation. Adapted PyTorch remains within tolerance at `6.6757e-6` on the unconditional path, while ORT versus adapted PyTorch fails at `1.1730e-4`. Both noise refiners remain matched, and the unconditional error grows to `2.1186e-3` by joint block 23. This proves the initiating drift lies in exported/ORT context self-attention execution rather than the adapter rewrite.

The next bounded experiment is inside `context_refiner.0`: probe Q/K normalization, logits, softmax, value accumulation, and output projection. Only if that isolates the reduction should the port test a mathematically collapsed unconditional context with multiplicity-aware joint attention or a fused WGSL online-softmax path. More RMS or RoPE variants are not justified by these results. Evidence: [`block-boundary report`](validation/2026-07-15-dit-flow4-invocations07-08-block-probes.json), [`conditional invocation 7`](validation/2026-07-15-dit-step-flow4-invocation07-fp32-onnx-strict.json), [`unconditional invocation 8`](validation/2026-07-15-dit-step-flow4-invocation08-fp32-onnx-strict.json), and [`projection A/B`](validation/2026-07-15-dit-invocation08-legacy-projection-cpu.json).

### Twenty-step fp32 CFG/Euler loop — completed, qualification fail

- twenty shifted Euler steps;
- guidance scale 3 and shift 3;
- twenty conditional and twenty unconditional WebGPU invocations;
- model load: 12,510.0 ms;
- sampling wall time: 676,669.1 ms;
- untouched official PyTorch/MPS sampling: 169,155.4 ms;
- latent maximum error: 0.0487092733;
- latent mean absolute error: 0.0003517170;
- latent cosine similarity: 0.9999996593;
- latent fraction within the qualification tolerance: 0.9983292;
- camera maximum error: 0.0001322459;
- camera cosine similarity: 0.9999999807.

All 40 WebGPU invocations completed without fallback. The recorded qualification envelope is `atol=0.005`, `rtol=0.003`, minimum cosine 0.99999998. The camera passes, but the latent misses both the elementwise and cosine requirements, so the overall result fails. The strict diagnostic also fails. Evidence: [`docs/benchmarks/2026-07-15-flow20-fp32-webgpu.json`](benchmarks/2026-07-15-flow20-fp32-webgpu.json).

### Eight-level octree neural and host trajectory — pass

The fp32 graph maps padded parent coordinates `x [1,8192,3]`, resolution `l [1]`, and latent condition `cond [1,8192,16]` to child logits `[1,8192,8]`. The teacher fixture uses the official four-step fp32 flow latent and records all eight official frontiers, systematic-resampling uniforms, sampled counts, and final jitter.

On the recorded Chrome 150/WebGPU environment, fallback disabled:

- graph plus sidecar transfer: 221,419,396 bytes;
- model/session load: 1,327.4 ms;
- eight primary official-frontier calls plus eight varied-padding probes;
- primary eight-call summed inference: 5,713.700000166893 ms;
- padding-probe summed inference: 5,673.1 ms;
- all 16 calls plus host replay wall time: 11,415.3 ms;
- every active logit passes `atol=0.005`, `rtol=0.01`, minimum cosine 0.99999;
- replacing every inactive zero-padded tail with varied valid coordinates changes no active output bit;
- every sampled child count matches official PyTorch at all eight levels;
- 41,972 of 41,972 recorded uniform random values are consumed;
- all 24,576 final point coordinates are bit-identical;
- final log-probability maximum error is 0.0000028610 with cosine similarity 0.9999999999999923.

The primary timing excludes the deliberate second padding probe at each level and is not an end-to-end generation benchmark. This gate validates the fp32 occupancy graph, padding independence, TypeScript systematic resampling, compaction, expansion, random consumption, captured jitter, and final occupied points together. Evidence: [`docs/validation/2026-07-15-octree-trajectory-webgpu-fp32-chrome.json`](validation/2026-07-15-octree-trajectory-webgpu-fp32-chrome.json).

The separate fp16 occupancy export remains failed and has no browser pass. That historical artifact does not change the full fp32 trajectory result.

### Gaussian feature decoder — fp32 raw boundary passed, final scene pending

The fp32 graph maps points `[1,8192,3]` and sampled condition `[1,8192,16]` to 3,932,160 raw values shaped `[1,8192,480]`. Its graph plus sidecar transfer is 1,094,219,284 bytes.

On the recorded Chrome 150/WebGPU environment, fallback disabled, the raw feature boundary passed against the deterministic official fp32 PyTorch reference:

- model/session load: 7,929.6 ms;
- inference: 5,370.6 ms;
- readback: 1.3 ms;
- maximum absolute error: 0.0032768250;
- mean absolute error: 0.0000083869;
- RMSE: 0.0000279985;
- cosine similarity: 0.99999999997;
- fraction within tolerance: 1.0.

The browser gate used `atol=0.02`, `rtol=0.01`, and minimum cosine similarity 0.9999. Evidence: [`docs/benchmarks/2026-07-14-gaussian-decoder-webgpu.json`](benchmarks/2026-07-14-gaussian-decoder-webgpu.json).

The separate Python ONNX Runtime comparison missed its original strict `atol=0.0001`, `rtol=0.001` gate, with maximum error 0.0067958832 and 0.9986687 of values inside tolerance. It passed a separately recorded relaxed `atol=0.008`, `rtol=0.005` gate. The browser pass uses its declared browser tolerance; neither result rewrites the failed Python strict gate. Evidence: [`docs/validation/2026-07-14-gaussian-decoder-fp32-python.json`](validation/2026-07-14-gaussian-decoder-fp32-python.json) and [`docs/validation/2026-07-14-gaussian-decoder-fp32-python-relaxed.json`](validation/2026-07-14-gaussian-decoder-fp32-python-relaxed.json).

The host activation equations now have a deterministic fixture generated by the official `triposplat._build_gaussians` function. The package test covers offsets, 32-Gaussians-per-point expansion, scale kernel and bias, opacity, SH degree zero, and quaternion bias with a maximum allowed error of `2e-6`. Export tests also include a byte-identical official `.splat` oracle. What remains open is running the Gaussian decoder from the now-qualified live octree points and sampled latent, applying those operations, and comparing the complete scene and render with official PyTorch.

### Packaged prepared-image end-to-end path — structural/export/viewer pass

The high-level `TripoSplatWebGPU.generate()` path completed with the canonical five-graph fp32 manifest, `cache: 'none'`, an already-prepared fixture, and explicit VAE/latent/camera noises. Every model object was streamed through temporary OPFS, byte-length and SHA-256 checked, then released by stage.

- `generate()` including verified artifact staging: 247,901.5 ms;
- total through PLY and `.splat` export, SHA-256, and viewer load: 248,951.4 ms;
- DINO inference: 4,339.6 ms;
- VAE inference: 1,745.3 ms;
- eight DiT calls: 120,168.9 ms;
- eight octree calls: 7,405.3 ms;
- Gaussian decoder inference: 7,138.4 ms;
- scene count: 262,144, with all required arrays present, correctly sized, and finite;
- PLY: 17,826,208 bytes, binary little-endian, correct vertex count;
- `.splat`: 8,388,608 bytes, exactly 32 bytes per Gaussian;
- viewer: ready in 569.2 ms, with a 1916×954 drawing buffer and 958×477 displayed canvas;
- model and scene disposal completed; post-disposal origin-storage usage was 216 bytes.

The built-in runtime used worker-retained positive and negative conditioning tensors. At four steps this reduces deterministic host-to-worker conditioning payload traffic from 184,774,656 to 46,193,664 bytes. At twenty steps it reduces the corresponding accounting from 923,873,280 to 46,193,664 bytes. This is not a controlled wall-time comparison and does not establish that ONNX Runtime avoids internal CPU-to-WebGPU uploads.

This gate deliberately records `numerical_parity_claimed: false`. It proves the packaged browser-local orchestration, lifecycle, scene structure, export sizes, and viewer-load/canvas path, but does not compare the complete scene or rendered pixels with an official PyTorch run. Evidence: [`docs/benchmarks/2026-07-15-e2e-render-structural-webgpu-fp32-chrome.json`](benchmarks/2026-07-15-e2e-render-structural-webgpu-fp32-chrome.json).

## Test environment and limits

The browser records above came from:

- Apple M3 Max;
- 128 GB unified memory;
- macOS 26.3;
- Chrome 150;
- ONNX Runtime WebGPU 1.27.0;
- `crossOriginIsolated === false`;
- WASM fallback disabled.

These runs establish Apple Silicon WebGPU execution on that machine only. They do not establish:

- compatibility with a 16 GB Mac;
- compatibility with M1, M2, M4, or later Apple GPUs;
- Microsoft Edge compatibility on macOS;
- Chrome or Edge on Windows;
- Safari or Firefox support;
- a peak GPU-memory figure;
- production inference time for an end-to-end scene.

## Release blockers

The package must not be described as production-usable until all of these gates are complete:

1. integrate and qualify a browser-local BiRefNet graph through the existing `removeBackground` hook, or fully validate the production prepared-input contract;
2. compare the live activated/packed scene with an official PyTorch whole-scene reference, including export orientation and a rendered result;
3. probe Q/K normalization, logits, softmax, value accumulation, and projection inside the now-localized `context_refiner.0` zero-context attention failure; then correct the confirmed reduction with a multiplicity-equivalent collapsed context or fused controlled reduction and re-run the strict/final-scene gates;
4. diagnose or bound the measured 20-step accumulated drift enough to pass an output-sensitivity-backed quality gate;
5. exercise cancellation, retry, repeated generation, worker termination, CDN failures, and persistent OPFS/Cache API behavior in real browsers without reloading the page;
6. extend the now-passing full packed-Vite browser execution to actual Next.js and native-ESM browser runs; their current package checks cover client TypeScript and import-map resolution only;
7. measure end-to-end time and memory behavior on the stated 16 GB Apple Silicon minimum in both Chrome and Edge;
8. resolve inherited `ml-sharp-web` redistribution rights and include complete source/model provenance notices.

Passing an internal lab or building a private Vite app is not a substitute for these release gates.

## Alpha package validation

The current workspace packages are built, tested, packed, and installed together into a fresh temporary Vite project:

| Check | Result |
| --- | --- |
| `@ai3d/gaussian-scene` and `@ai3d/triposplat-webgpu` package builds/tests | Passed |
| Packed tarballs installed through npm into a clean Vite fixture | Passed |
| Vite production bundle from installed tarballs | Passed |
| Packaged TripoSplat module worker present | Passed |
| Packaged ONNX Runtime WASM asset present | Passed |
| Installed Vite bundle executes a tiny identity ONNX graph through its worker with WebGPU | Passed in recorded Chrome 150 smoke |
| Exact final installed Vite tarball completes DINO, VAE, eight DiT calls, eight octree levels, Gaussian decode, PLY, and `.splat` export | **Passed** in Chrome 150; 262,144 finite Gaussians |
| Strict Next.js client TypeScript fixture | Passed |
| Native ESM/import-map module graph | Passed |
| Forbidden model/checkpoint/fixture payload scan | Passed |

The tiny Vite smoke validates installed-tarball worker/ONNX Runtime asset resolution directly. The separate exact-tarball full run used cross-origin model URLs and `cache: 'none'`, completed all five stages, returned 262,144 finite Gaussians, and produced both exact-size exports in 210,158.7 ms. It does not exercise persistent cache behavior or claim whole-scene numerical/render parity. Evidence: [`worker smoke`](validation/2026-07-15-packed-vite-consumer-webgpu-chrome.json) and [`full packed-consumer report`](benchmarks/2026-07-15-packed-vite-full-e2e-webgpu-fp32-chrome.json).
