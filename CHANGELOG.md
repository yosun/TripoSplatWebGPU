# Changelog

This project uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) structure. Semantic versioning begins when an installable public package has passed the release gates; repository milestones before that point remain unreleased.

## Unreleased

### Added

- model-neutral ONNX Runtime WebGPU worker/session protocol with explicit external-data descriptors and session disposal;
- preserved `SharpWebGPUModel` baseline behind a model-independent `ImageToGaussianModel` interface;
- TripoSplat preprocessing, explicit stochastic Flux VAE epsilon, fixed-shape encoder contracts, and deterministic browser fixtures;
- TypeScript classifier-free-guided Euler sampling for four and twenty step schedules, including cancellation checks and float16 prediction arithmetic;
- correctness-first dynamic octree resampling/expansion algorithms and Gaussian feature decoding;
- framework-neutral `@ai3d/gaussian-scene` workspace package with canonical metadata, disposal, binary 17-float 3DGS PLY export, and official-fixture-compatible 32-byte browser `.splat` export;
- alpha `@ai3d/triposplat-webgpu` package work, release API contract examples, model-hosting guidance, compatibility policy, and package-validation CI;
- reproducible TripoSplat export/validation scripts and machine-readable benchmark records.
- complete five-graph fp32 manifest, built-in staged executor, verified OPFS/Cache API/ephemeral artifact delivery, browser-local background-remover hook, and E2E structural/export/viewer lab;
- packed Vite, strict Next-client TypeScript, and native ESM/import-map consumer checks.
- worker-retained immutable DiT conditioning inputs, with transparent cloning fallback for custom runtimes and cleanup on graph disposal.
- serialized high-level generation; prompt queued cancellation; recoverable worker recreation after abort during configuration, graph load, retained-input transfer, or inference; and idempotent disposal that terminates pending work and awaits generation unwinding.

### Validated

- Flux VAE WebGPU vertical slice passed its deterministic PyTorch parity gate on the recorded Apple M3 Max/Chrome 150 environment;
- the full fp32 DINOv3 WebGPU encoder and one canonical fp32 DiT invocation passed strict deterministic untouched-PyTorch gates;
- four-step CFG/Euler control flow completed all eight expected WebGPU DiT invocations with fallback disabled and passed its qualification envelope while missing the separately recorded strict diagnostic;
- the measured 20-step loop completed all forty WebGPU calls without fallback, but failed its qualification and strict final-state gates;
- the full eight-level fp32 octree trajectory passed active logits, padding independence, systematic child counts, random consumption, and bit-exact final points;
- the fp32 Gaussian decoder WebGPU graph passed its deterministic raw `[1,8192,480]` feature-boundary gate with fallback disabled;
- the packaged prepared-image four-step browser path completed all five stages, returned 262,144 finite Gaussians, produced valid-size PLY and `.splat` exports, loaded the PLY into a ready 1916×954 viewer canvas, and disposed its scene/model;
- alpha Gaussian/core tarballs built and installed together in fresh consumer fixtures without including model or fixture payloads.
- an installed packed Vite consumer resolved the emitted worker and ONNX Runtime assets and executed a tiny identity ONNX graph with the WebGPU provider;
- the exact final packed Vite tarball completed the full five-graph WebGPU path through 262,144 finite Gaussians and exact-size PLY/`.splat` exports; official whole-scene parity is not claimed.

### Known issues

- four-step output misses the strict diagnostic; teacher-forced replay localizes it to unconditional calls from step 2 onward, and a 39-boundary CPU ORT probe finds the first material split in the `context_refiner.0` attention residual before its MLP;
- twenty-step final state fails both the qualification and strict gates despite completing all calls;
- the structural/viewer E2E scene is not yet compared numerically or pixel-by-pixel with an official PyTorch whole-scene reference;
- the fp16 octree occupancy graph failed its Python numerical gate and has no browser pass;
- a WebGPU-qualified built-in BiRefNet graph, rendered-reference parity, real-browser cancellation/retry stress, and repeated generation remain unvalidated;
- no 16 GB Apple Silicon or Windows WebGPU qualification exists;
- production-origin persistent-cache stress and resumable downloads remain release gates;
- redistribution rights for inherited `ml-sharp-web` source must be clarified or the affected code reimplemented before public release.

### Measured

- Flux VAE: 1,304.1 ms median inference and 1,601.7 ms median source preprocessing plus encode;
- DINOv3: 8,702.1 ms cold model/session load and 4,262.4 ms inference;
- one fp32 DiT invocation: 12,584.6 ms cold model/session load and 11,844.6 ms inference;
- four-step/eight-invocation loop: 11,411.3 ms model load and 102,268.8 ms sampling wall time;
- twenty-step/forty-invocation loop: 12,510.0 ms model load and 676,669.1 ms sampling wall time;
- full eight-level fp32 octree trajectory: 1,327.4 ms load and 5,713.7 ms primary inference sum;
- fp32 Gaussian raw feature decoder: 7,929.6 ms model load and 5,370.6 ms inference.
- packaged structural/viewer E2E: 247,901.5 ms `generate()` including verified non-persistent artifact staging and 248,951.4 ms through PLY/`.splat` export hashes plus viewer load;
- retained fp32 DiT conditioning: four-step host-to-worker payload accounting falls from 184,774,656 to 46,193,664 bytes, and twenty-step accounting from 923,873,280 to 46,193,664 bytes. This is not a wall-time claim.
- final packed Vite full run: 209,712.2 ms through `generate()` and 210,158.7 ms through both exports on the recorded Apple M3 Max/Chrome 150 environment.

All measurements above are from one Apple M3 Max Mac with 128 GB unified memory, macOS 26.3, and Chrome 150. They are not performance estimates for other devices.

Package-validation sizes were 13,362 compressed bytes for `@ai3d/gaussian-scene` and 6,071,865 compressed bytes for `@ai3d/triposplat-webgpu` in this checkout.
