# Compatibility and measured benchmarks

## Compatibility policy

The target is desktop Chrome and Edge on Apple Silicon Macs with at least 16 GB unified memory. That is a target, not a current support claim.

The only recorded browser environment is Chrome 150 on an Apple M3 Max Mac with 128 GB unified memory. No Microsoft Edge configuration, 16 GB Apple Silicon, Safari, Firefox, integrated Intel GPU, or discrete Windows GPU qualification has been completed.

Until the complete pipeline and package are available, compatibility should be described as experimental even on the measured machine.

## What a compatibility check can know

A browser can report whether `navigator.gpu` exists, whether an adapter/device can be requested, and selected device limits and features. It cannot reliably report total GPU memory, free unified memory, or the ONNX Runtime/driver peak allocation that a future generation will require.

A structured compatibility report should therefore include:

- browser and platform strings;
- WebGPU availability;
- adapter metadata when the browser exposes it;
- relevant adapter/device limits;
- estimated static model transfer bytes from the manifest;
- only measured or carefully derived peak-byte data, clearly labeled;
- warnings for unqualified browsers and fallback paths;
- blockers such as a missing required WebGPU feature or graph larger than a supported buffer limit.

It must not convert JavaScript heap size or `maxBufferSize` into a claim about available GPU memory.

## Measured results

Environment shared by all browser measurements:

| Field | Value |
| --- | --- |
| Hardware | Apple M3 Max Mac, 128 GB unified memory |
| Operating system | macOS 26.3 (25D125) |
| Browser | Chrome 150.0.0.0 |
| Execution provider | ONNX Runtime WebGPU |
| ONNX Runtime Web | 1.27.0 |
| Cross-origin isolated | No |
| WASM fallback | Disabled |

### Flux VAE encoder

| Metric | Value |
| --- | ---: |
| Model transfer | 137,800,544 bytes |
| Model load | 817 ms |
| Median inference, three runs | 1,304.1 ms |
| Median source preprocessing + encode | 1,601.7 ms |
| Maximum absolute tensor error | 0.0001683235 |
| Cosine similarity | 0.9999999997 |

### One fp32 DiT invocation

| Metric | Value |
| --- | ---: |
| Model transfer | 1,643,895,982 bytes |
| Cold model/session load | 12,584.6 ms |
| Inference | 11,844.6 ms |
| Readback | 0.9 ms |
| Latent maximum absolute error | 0.0000057220 |
| Latent cosine similarity | 0.9999999999996 |
| Camera maximum absolute error | 0.0000017881 |
| Camera cosine similarity | 0.9999999999997 |
| Declared strict gate | **Passed** |

### Four-step fp32 CFG/Euler sampling

| Metric | Value |
| --- | ---: |
| WebGPU DiT invocations | 8 |
| Model load | 11,411.3 ms |
| Sampling wall | 102,268.8 ms |
| Summed inference | 102,236.0 ms |
| Latent maximum absolute error | 0.0044521093 |
| Latent cosine similarity | 0.9999999885 |
| Camera cosine similarity | 0.9999999986 |
| Qualification envelope | **Passed** |
| Separate strict diagnostic | **Failed** |

### Four-step teacher-forced DiT trajectory

The browser graph was also run eight times with the exact official PyTorch sample and timestep injected at every conditional and unconditional invocation. This removes autoregressive CFG/Euler state drift from each comparison.

| Invocation class | Strict result |
| --- | --- |
| Conditional, steps 1–4 | **Passed** |
| Unconditional, step 1 | **Passed** |
| Unconditional, steps 2–4 | **Failed** for latent; camera passed |
| All camera outputs | **Passed** |

The failing unconditional latent maximum errors were 0.000330925, 0.000472546, and 0.000844243 at steps 2–4. Their fractions within combined tolerance were 0.9994812, 0.9990692, and 0.9958420. The strict gate required every value within `atol=0.0001` plus `rtol=0.001 * abs(reference)` and cosine similarity of at least 0.99999999.

| Timing/size | Value |
| --- | ---: |
| Model transfer | 1,643,895,982 bytes |
| Model load | 12,231.8 ms |
| WebGPU invocations | 8 |
| Summed inference | 107,931.5 ms |
| Sampling wall | 108,012.6 ms |

This measured result localizes the strict defect to invocation-dependent behavior, especially the all-zero conditioning branch; it is not explained only by autoregressive Euler accumulation. Evidence: [`teacher-forced browser report`](validation/2026-07-15-flow4-teacher-forced-webgpu-fp32-chrome.json).

An explicit stable-RMS graph rewrite was measured as a diagnostic experiment. It failed the same step-2 through step-4 unconditional latent gates, with maximum errors of 0.000346661, 0.000476122, and 0.000827583. This is not a material improvement over the canonical graph, so the manifest remains unchanged. Its 95,085.9 ms wall time is an engineering diagnostic, not a promoted performance result. Evidence: [`stable-RMS teacher-forced report`](validation/2026-07-15-flow4-teacher-forced-stable-rms-webgpu-fp32-chrome.json).

Fresh CPU ONNX Runtime reproduces most of the step-4 asymmetry with no browser or reused session: invocation 7 conditional passes at 0.000043750 maximum latent error, while invocation 8—same sample/timestep, zero context—fails at 0.000764251. Adapted PyTorch remains much closer to untouched official PyTorch on the unconditional input (0.000071019 maximum), and an earlier ordinary output-projection ONNX graph fails almost identically at 0.000745058. Static Sobol and the real-RoPE primitive are already gated. The evidence therefore points to the long repeated-token ONNX attention reductions, not stable RMS, RoPE, the output-projection workaround, WebGPU, or session reuse. This is a numerical diagnosis, not a browser performance result. Evidence: [`conditional CPU report`](validation/2026-07-15-dit-step-flow4-invocation07-fp32-onnx-strict.json), [`unconditional CPU report`](validation/2026-07-15-dit-step-flow4-invocation08-fp32-onnx-strict.json), and [`projection A/B report`](validation/2026-07-15-dit-invocation08-legacy-projection-cpu.json).

The follow-up 39-boundary probe localizes the first material divergence to the attention residual of `context_refiner.0`, before its MLP and before any joint block. CPU ORT versus official PyTorch measures `3.0994e-6` conditional maximum error and `1.1063e-4` unconditional maximum error at this boundary (35.69×). Adapted PyTorch remains within tolerance on the same unconditional boundary (`6.6757e-6`), while ORT versus adapted PyTorch fails (`1.1730e-4`). The next diagnostic must split this one attention operation into Q/K normalization, logits, softmax, value accumulation, and projection. Evidence: [`block-boundary report`](validation/2026-07-15-dit-flow4-invocations07-08-block-probes.json).

### Twenty-step fp32 CFG/Euler sampling

| Metric | Value |
| --- | ---: |
| WebGPU DiT invocations | 40 |
| Model load | 12,510.0 ms |
| Sampling wall | 676,669.1 ms |
| Summed inference | 676,525.5 ms |
| Official PyTorch/MPS sampling | 169,155.4 ms |
| Latent maximum absolute error | 0.0487092733 |
| Latent mean absolute error | 0.0003517170 |
| Latent cosine similarity | 0.9999996593 |
| Latent fraction within qualification tolerance | 0.9983292 |
| Camera cosine similarity | 0.9999999807 |
| Qualification envelope | **Failed** |
| Strict diagnostic | **Failed** |

The control loop completed all 40 WebGPU invocations without fallback. This is a measured failure from accumulated final-state drift, not an extrapolation or an incomplete run.

### Eight-level fp32 octree trajectory

| Metric | Value |
| --- | ---: |
| Model transfer | 221,419,396 bytes |
| Model/session load | 1,327.4 ms |
| Primary official-frontier calls | 8 |
| Varied-padding probes | 8 |
| Primary summed inference | 5,713.700000166893 ms |
| Padding-probe summed inference | 5,673.1 ms |
| All-call trajectory wall | 11,415.3 ms |
| Active-logit levels passing | 8/8 |
| Active padding-probe maximum error | 0 |
| Sampled child-count mismatches | 0 |
| Recorded random values consumed | 41,972/41,972 |
| Final point-coordinate maximum error | 0 |
| Final log-probability maximum error | 0.0000028610 |
| Full trajectory gate | **Passed** |

Every active WebGPU occupancy logit passes the declared `atol=0.005`, `rtol=0.01`, minimum-cosine 0.99999 gate. The second call at each level replaces the inactive zero-padded tail with varied normalized coordinates; active outputs remain bit-identical. TypeScript systematic resampling reproduces every official child count, consumes every captured random value exactly once, and produces bit-identical final points after compaction, expansion, and jitter replay.

The 5,713.700000166893 ms primary sum covers the eight official neural calls. The 11,415.3 ms wall time deliberately includes eight extra padding probes and host comparison work, so neither is an end-to-end generation benchmark. Evidence: [`full octree trajectory report`](validation/2026-07-15-octree-trajectory-webgpu-fp32-chrome.json).

### fp32 Gaussian raw feature decoder

| Metric | Value |
| --- | ---: |
| Model transfer | 1,094,219,284 bytes |
| Model/session load | 7,929.6 ms |
| Inference | 5,370.6 ms |
| Readback | 1.3 ms |
| Values compared | 3,932,160 |
| Maximum absolute error | 0.0032768250 |
| Mean absolute error | 0.0000083869 |
| RMSE | 0.0000279985 |
| Cosine similarity | 0.99999999997 |
| Fraction within tolerance | 1.0 |
| Declared browser gate | Passed |

This validates raw `[1,8192,480]` graph features. A separate small official `_build_gaussians` fixture validates the host activation equations and export tests validate the official `.splat` layout. The upstream octree points are now qualified, but a live Gaussian decode from those points through activated arrays and a rendered final scene remains open.

These numbers are single-machine engineering records. They do not constitute a distribution or service-level objective; the 20-step entry is an actual completed run rather than a forecast.

### Packaged four-step end-to-end structural/viewer run

| Metric | Value |
| --- | ---: |
| Declared verified artifact set | 6,465,182,402 bytes |
| `generate()` including non-persistent staging | 247,901.5 ms |
| DINO inference | 4,339.6 ms |
| Flux VAE inference | 1,745.3 ms |
| DiT inference, eight calls | 120,168.9 ms |
| Octree inference, eight calls | 7,405.3 ms |
| Gaussian decoder inference | 7,138.4 ms |
| PLY viewer load | 569.2 ms |
| Total through both export hashes and viewer load | 248,951.4 ms |
| Gaussian count | 262,144 |
| PLY bytes | 17,826,208 |
| `.splat` bytes | 8,388,608 |
| Viewer drawing buffer | 1916×954 |
| Structural/export/viewer gate | **Passed** |
| Whole-scene numerical/render parity | **Not claimed** |

The run used `cache: 'none'`: each graph was streamed into temporary OPFS, verified, loaded, and released before the next model stage. All scene arrays were present, correctly sized, and finite; both exports matched their structural byte contracts; the PLY reached viewer-ready state with non-zero drawing/display dimensions; the scene and model were disposed. The reported post-disposal origin usage of 216 bytes is not a peak-memory or peak-storage measurement. Evidence: [`end-to-end viewer report`](benchmarks/2026-07-15-e2e-render-structural-webgpu-fp32-chrome.json).

The same run exercised retained worker-side DiT conditioning. Deterministic host-to-worker payload accounting falls from 184,774,656 to 46,193,664 bytes for four steps and from 923,873,280 to 46,193,664 bytes for twenty steps. The measured run is not an A/B benchmark, and the optimization does not prove GPU-resident input reuse inside ONNX Runtime.

### Exact packed Vite consumer — full five-stage run

The final `@ai3d/triposplat-webgpu@0.1.0-alpha.0` tarball, SHA-256 `a2ba5ce623c7b6a9d86225742789ef2e7f93e9b8aeec95bc44bcb0725cb21859`, was installed with npm into a fresh temporary Vite project. No workspace package link or custom workspace worker was used. Vite emitted the package's module worker and ONNX Runtime assets; the installed bundle first passed a WebGPU identity graph and then completed the full prepared-image TripoSplat path.

| Metric | Value |
| --- | ---: |
| Tarball bytes | 6,128,163 |
| Declared model artifacts | 6,465,182,402 bytes |
| `generate()` including non-persistent verified staging | 209,712.2 ms |
| DINO inference | 4,285.2 ms |
| Flux VAE inference | 1,756.5 ms |
| DiT inference, eight calls | 111,538.3 ms |
| Octree inference, eight calls | 6,455.0 ms |
| Gaussian decoder inference | 6,111.2 ms |
| Total through PLY and `.splat` export | 210,158.7 ms |
| Gaussian count | 262,144 |
| All required arrays finite | Yes |
| PLY bytes | 17,826,208 |
| `.splat` bytes | 8,388,608 |
| Clean-install structural/export gate | **Passed** |

The model origin was a loopback cross-origin static server with CORS, so this is not a public-CDN download benchmark and may benefit from operating-system file caching. It used `cache: 'none'`, did not upload the image, and does not claim whole-scene numerical/render parity or peak memory. Evidence: [`packed Vite full-run report`](benchmarks/2026-07-15-packed-vite-full-e2e-webgpu-fp32-chrome.json).

## Memory reporting

The VAE record includes `performance.memory` values for the main JavaScript realm. After three runs, its recorded used heap was 59,219,068 bytes. This excludes the ONNX worker, WebGPU buffers, the Metal driver, and other unified-memory residency. It must not be reported as model or peak memory.

The full attention design avoids materializing the largest naïve score matrices by using query chunking and export adaptations. That addresses a known graph-shape hazard but is not proof that the complete pipeline fits a 16 GB device.

## Benchmark protocol for future claims

Before publishing a performance claim:

1. identify exact hardware, memory capacity, OS, browser, ONNX Runtime version, provider, model revision, graph hashes, settings, and whether fallback occurred;
2. separate download, cache hit, session construction, preprocessing, each graph, readback, packing, and total wall time;
3. include warm-up policy and all raw runs, not only the fastest result;
4. record parity or output-quality gate beside timing;
5. report only memory counters whose coverage is understood;
6. test cancellation and a second generation after disposal;
7. store the machine-readable result under `docs/benchmarks/`.

Do not publish a “20-step estimate” obtained by multiplying a one-step timing. Session staging, cache state, thermal behavior, readbacks, and decoding make that extrapolation unreliable.
