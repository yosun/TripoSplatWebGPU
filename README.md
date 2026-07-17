# TripoSplat WebGPU

[![Package validation](https://github.com/yosun/TripoSplatWebGPU/actions/workflows/package-validation.yml/badge.svg)](https://github.com/yosun/TripoSplatWebGPU/actions/workflows/package-validation.yml)
![WebGPU required](https://img.shields.io/badge/WebGPU-required-4285F4)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-browser--local%20demo-FFD21E)](https://huggingface.co/spaces/Yosun/TripoSplat-WebGPU-Demo)
[![Vercel demo](https://img.shields.io/badge/Vercel-browser--local%20demo-c6ff4a?logo=vercel&logoColor=black)](https://triposplat-webgpu.vercel.app/e2e-web)

A working, browser-local WebGPU engineering preview of [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat), built from the [ml-sharp-web](https://github.com/bring-shrubbery/ml-sharp-web) application chassis. It selects an image, runs the model pipeline locally in the browser, displays the resulting Gaussian scene, and exports binary PLY or `.splat` files. The official TripoSplat PyTorch implementation remains the numerical source of truth; the existing SHARP path remains a known-good browser inference baseline.

## Try the browser-local demos

- **[Hugging Face Space](https://huggingface.co/spaces/Yosun/TripoSplat-WebGPU-Demo)** — static Vite demo, with no Hugging Face inference compute.
- **[Vercel runner](https://triposplat-webgpu.vercel.app/e2e-web)** — public browser runner; [`/e2e-lab.html`](https://triposplat-webgpu.vercel.app/e2e-lab) is the fixture-driven qualification surface.

Both deployments are static front ends: the model package is fetched directly from [Yosun/TripoSplat-WebGPU](https://huggingface.co/Yosun/TripoSplat-WebGPU), while model inference and exports run locally in the browser. The model layer does not upload source pixels, although deployment analytics, CDN logs, browser extensions, and an image URL's host are separate privacy considerations.

### Recorded result videos

These are browser recordings from the tester results:

<table>
  <tr>
    <td><strong>Cartoon House</strong><br><video controls preload="metadata" width="320" src="https://raw.githubusercontent.com/yosun/TripoSplatWebGPU/main/public/_testers/results/Cartoon%20House/cartoon.house.webm"></video></td>
    <td><strong>Women's Shoes</strong><br><video controls preload="metadata" width="320" src="https://raw.githubusercontent.com/yosun/TripoSplatWebGPU/main/public/_testers/results/Womens%20Shoes%20Red/womens.shoes.webm"></video></td>
    <td><strong>Corgi CEO Basket</strong><br><video controls preload="metadata" width="320" src="https://raw.githubusercontent.com/yosun/TripoSplatWebGPU/main/public/_testers/results/corgi.ceo-basket/corgi.ceo-basket.webm"></video></td>
  </tr>
</table>

The original files are available in [`public/_testers/results`](public/_testers/results).

The current runner source accepts PNG, WebP, JPEG, and AVIF. Alpha-bearing images are ready to run; opaque photos automatically start the browser-local preparation path, which removes the background, frames the subject, and produces the model-ready 1024px input. Its preparation model is fetched and cached separately on first use. This is a practical input workflow, not a claim of numerical or visual-quality parity for background removal.

## Before you run

- Use desktop Chrome with WebGPU. The only recorded qualification environment is Chrome 150 on an Apple M3 Max with 128 GB unified memory; Edge, 16 GB Apple Silicon, Safari, Firefox, and Windows GPUs are not yet qualified.
- Allow roughly 6.47 GB for the first verified model download and browser cache. Cache persistence depends on browser storage and quota.
- The public runner requests 20 sampling steps / 40 CFG DiT calls and can take several minutes on the recorded high-end hardware. That path executes but fails its recorded qualification and strict final-state gates.

This is an engineering preview, not a production or broad-hardware support claim. The four-step prepared-image path has a structural/export/viewer pass but misses a stricter diagnostic; official whole-scene and rendered-pixel parity, repeated full-generation memory behavior, and production CDN/OPFS qualification remain open. See [Current status](docs/current-status.md) before integrating or benchmarking.

**[View the source, benchmarks, and issue tracker on GitHub →](https://github.com/yosun/TripoSplatWebGPU)**

## Local development

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:<PORT>/e2e-web.html`. Test inputs and recorded outputs are stored in [`public/_testers`](https://github.com/yosun/TripoSplatWebGPU/tree/main/public/_testers).

## Hugging Face Space deployment

The dedicated [`huggingface-space/README.md`](huggingface-space/README.md) is the Space card and contains the `sdk: static` metadata. Keeping it separate prevents Hugging Face configuration from leaking into the project README. During deployment, its contents must become the Space repository's root `README.md` because nested metadata is not interpreted.

Build locally to avoid depending on Hugging Face's custom build environment. The Vite build intentionally excludes local model artifacts and deterministic fixtures; browsers retrieve the versioned model manifest and files directly from the model repository.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm build
hf upload Yosun/TripoSplat-WebGPU-Demo dist . \
  --repo-type space \
  --commit-message 'Deploy TripoSplat WebGPU demo'
hf repos cp huggingface-space/README.md \
  hf://spaces/Yosun/TripoSplat-WebGPU-Demo/README.md
```

Upload the Space card after the bundle because `dist/` does not contain `README.md`. Do not upload local model artifacts, credentials, caches, or uncommitted experiment files. Hugging Face may require a paid plan or account credits to activate updated Static Space hosting; check current Spaces pricing before deployment.


> **Current milestone, not a production release.** The alpha `@ai3d/triposplat-webgpu` workspace package contains the five-stage browser executor and a complete 6.465 GB fp32 manifest. A measured prepared-image run now completes the entire packaged browser path, exports 262,144 finite Gaussians, and loads the PLY into the retained browser viewer with a live canvas. DINOv3, Flux VAE, one DiT invocation, the full eight-level octree trajectory, and the raw Gaussian decoder boundary also pass their recorded Chrome/WebGPU gates. The four-step loop passes its qualification envelope but misses a stricter diagnostic; the measured 20-step loop fails its final-state gate. Official whole-scene/render parity, bundled BiRefNet, Edge/16 GB qualification, and production memory measurements remain release blockers. See [Current status](docs/current-status.md) before integrating or benchmarking this work.

## Verified status

| Capability | Result | Evidence |
| --- | --- | --- |
| TripoSplat preprocessing + DINOv3 encoder | **STRICT PASS** on one recorded fixture | [DINOv3 WebGPU validation](docs/validation/2026-07-15-dinov3-fp32-webgpu.json) |
| TripoSplat preprocessing + Flux VAE encoder | **PASS** on one recorded fixture | [VAE WebGPU benchmark](docs/benchmarks/2026-07-14-flux2-vae-webgpu.json) |
| One exported fp32 DiT invocation | **STRICT PASS** against untouched official PyTorch | [DiT WebGPU validation](docs/validation/2026-07-15-dit-step-webgpu-fp32-chrome.json) |
| Four-step CFG/Euler browser loop | Eight calls complete; **qualification PASS, strict diagnostic FAIL** | [Four-step benchmark](docs/benchmarks/2026-07-15-flow4-fp32-webgpu.json) |
| Twenty-step guided sampling | Forty calls complete; **qualification and strict FAIL** | [Twenty-step benchmark](docs/benchmarks/2026-07-15-flow20-fp32-webgpu.json) |
| Eight-level fp32 octree trajectory | **PASS** for logits, padding independence, resampling, and final points | [Octree trajectory validation](docs/validation/2026-07-15-octree-trajectory-webgpu-fp32-chrome.json) |
| fp32 Gaussian feature decoder | **PASS** on WebGPU for raw `[1,8192,480]` features | [Gaussian decoder WebGPU benchmark](docs/benchmarks/2026-07-14-gaussian-decoder-webgpu.json); final scene parity remains open |
| Packaged prepared-image end-to-end path | **STRUCTURAL/VIEWER PASS**: all five stages, 262,144 finite Gaussians, PLY and `.splat` export, ready viewer canvas | [End-to-end viewer benchmark](docs/benchmarks/2026-07-15-e2e-render-structural-webgpu-fp32-chrome.json); whole-scene numerical/render parity is not claimed |
| Canonical Gaussian scene + PLY/`.splat` helpers | Implemented as [`@ai3d/gaussian-scene`](packages/gaussian-scene/README.md) workspace package | Package tests cover ownership, disposal, validation, PLY, and official-fixture `.splat` bytes |
| `@ai3d/triposplat-webgpu` package | [Alpha tarball](packages/triposplat-webgpu/README.md), built-in five-stage executor, integrity cache, and clean consumer checks pass; not published | Exact final tarball installed into fresh Vite and completed the five-stage WebGPU path with 262,144 finite Gaussians and both exports; [packed-consumer report](docs/benchmarks/2026-07-15-packed-vite-full-e2e-webgpu-fp32-chrome.json). Strict Next-client TypeScript and native ESM/import-map checks also pass |
| SHARP browser path | Preserved | Existing app path remains the baseline; SHARP weights have separate research-only terms |

“PASS” above means the recorded fixture met its declared numerical thresholds. It does not establish general image quality, end-to-end correctness, leak-free repeated generation, or support on untested devices.

## Current DiT experiment

An opt-in `--collapsed-unconditional-context` DiT export is under investigation to localize the remaining unconditional-context reduction error. It still accepts the normal public conditioning tensors, passes them through both official embedders, retains one representative context token, and applies a `log(4101)` attention bias to preserve the aggregate contribution of the 4,101 identical zero-conditioning keys.

This is an **unconditional-only diagnostic graph**: both `feature1` and `feature2` must be exactly zero, it cannot replace the canonical conditional/unconditional graph, and it is not part of the browser manifest. It has no recorded parity result yet. Export and validate it separately with the [TripoSplat script guide](scripts/triposplat/README.md); only an untouched-official and ONNX Runtime parity pass can promote it beyond this experiment.

## Measured browser results

All browser measurements below came from one Apple M3 Max Mac with 128 GB unified memory, macOS 26.3, and Chrome 150 using ONNX Runtime WebGPU 1.27.0. WASM fallback was disabled. They are not estimates and must not be extrapolated to the 16 GB target.

| Stage | Measured result |
| --- | --- |
| DINOv3 encoder | 8,702.1 ms cold model/session load; 4,262.4 ms inference; feature cosine similarity 0.9999999999966; strict gate passed |
| Flux VAE encoder | 1,304.1 ms median inference; 1,601.7 ms median source preprocessing plus encode; tensor cosine similarity 0.9999999997 |
| One fp32 DiT invocation | 12,584.6 ms cold model/session load; 11,844.6 ms inference; latent cosine similarity 0.9999999999996; strict gate passed |
| Four-step fp32 CFG/Euler loop | 11,411.3 ms model load; 102,268.8 ms sampling wall for eight WebGPU calls; latent cosine 0.9999999885; qualification passed, strict diagnostic failed |
| Twenty-step fp32 CFG/Euler loop | 12,510.0 ms model load; 676,669.1 ms sampling wall for forty WebGPU calls; qualification and strict gates failed |
| Full fp32 octree trajectory | 1,327.4 ms model load; 5,713.7 ms primary eight-call inference; final points bit-exact; full trajectory gate passed |
| fp32 Gaussian raw feature decoder | 7,929.6 ms model load; 5,370.6 ms inference; cosine similarity 0.99999999997; browser gate passed |
| Packaged four-step end-to-end structure/export/viewer | 247,901.5 ms `generate()` including verified temporary artifact staging; 248,951.4 ms through both hashed exports and viewer load; 262,144 finite Gaussians; viewer canvas 1916×954; structural/viewer gate passed |
| Final packed-tarball Vite consumer, full five-stage run | 209,712.2 ms `generate()` including verified temporary artifact staging; 210,158.7 ms through PLY and `.splat` export; exact tarball SHA-256 `a2ba5ce6…`; clean-install structural/export gate passed |
| Reusable fp32 DiT conditioning payloads | 4-step host-to-worker traffic reduced from 184,774,656 to 46,193,664 bytes; 20-step from 923,873,280 to 46,193,664 bytes. This is deterministic payload accounting, not measured wall-time or GPU-upload savings |

Chromium exposes limited main-realm JavaScript heap counters, not worker memory, WebGPU buffer residency, driver allocations, or reliable total GPU/unified memory. No peak-memory or 16 GB compatibility result is claimed. See [Compatibility and benchmarks](docs/compatibility-and-benchmarks.md).

## Repository development

This is currently a contributor workflow, not an npm consumer workflow.

Requirements:

- Node.js 22 or newer;
- Corepack and pnpm 11.13.0;
- current desktop Chrome or Edge with WebGPU enabled;
- local or separately hosted model artifacts and deterministic fixtures for the lab being run.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm dev
```

`pnpm typecheck` first builds the `@ai3d/gaussian-scene` and `@ai3d/triposplat-webgpu` workspace packages, so the root tests resolve their generated package output without a separate manual build step.

The development server exposes a public runner and engineering validation surfaces:

- `/` — redirects to the public TripoSplat image-to-spatial-scene runner;
- `/e2e-web.html` — public browser-local runner: visible file/URL image input, remote model-CDN configuration, compatibility checks, verified browser cache, generation, preview, and exports;
- `/sharp-lab.html` — preserved legacy SHARP application path;
- `/encoder-lab.html` — preprocessing and encoder vertical-slice comparison;
- `/dit-lab.html` — one DiT invocation against a deterministic PyTorch fixture;
- `/flow-lab.html` — four-step CFG/Euler execution and final-state comparison;
- `/octree-lab.html` — one fp32 occupancy-logit invocation or full eight-level trajectory replay;
- `/gaussian-lab.html` — one fp32 raw Gaussian-feature invocation.
- `/e2e-lab.html` — complete prepared-image package execution, export qualification, and exported-PLY viewer/canvas sanity gate.

The labs remain engineering fixtures rather than the package API. The public `/e2e-web.html` runner uses the package’s normal image-input path rather than deterministic fixtures, but it is still an engineering preview and does not make a production-readiness or numerical-parity claim.

## Preserved SHARP baseline

The root application retains the original SHARP browser workflow: local image selection, ONNX Runtime WebGPU inference, SHARP-specific camera/NDC postprocessing, preview, and PLY download. Its normal external-data export needs both `sharp_web_predictor.onnx` and `sharp_web_predictor.onnx.data` with the sidecar path expected by the graph.

SHARP is intentionally separate from TripoSplat. Its image stretch, focal-length input, camera-space covariance conversion, opacity filtering, coordinate flips, and PLY metadata are not reused as TripoSplat numerical logic. Apple SHARP weights are restricted to research purposes under `LICENSE_MODEL`; do not bundle or host them without accepting those terms.

The SHARP path is the known-good browser chassis baseline, not evidence that every TripoSplat release gate has passed.

## Model files

ONNX graphs may use external data. The graph and every sidecar must be served with the exact `location` path embedded in the ONNX graph. The current immutable manifest references ten objects totaling 6,465,182,402 bytes. The canonical fp32 DiT pair alone totals 1,643,895,982 bytes; exact object sizes and SHA-256 digests are declared in [`public/models/triposplat/manifest.json`](public/models/triposplat/manifest.json).

Do not route these objects through Vercel functions or another application proxy. Use S3/CloudFront, GCS/Cloud CDN, or an equivalent static object CDN with CORS, byte-range requests, immutable versioned objects, and correct external-data paths. User images must remain in the browser. See [Model hosting](docs/model-hosting.md).

## Intended package API

The alpha framework-neutral `@ai3d/triposplat-webgpu` workspace package exposes the high-level class, typed errors and contracts, built-in five-stage executor, manifest/capability handling, verified OPFS/Cache API/ephemeral artifact delivery, low-level graph runtime, sampler/octree/decode helpers, and Gaussian export surface. It refuses generation when the manifest lacks any required graph. The built-in worker retains the positive and negative DiT conditioning tensors once per session, so each sampling invocation transfers only latent, camera, and timestep payloads. High-level generations are serialized, queued aborts reject promptly, and an in-flight abort tears down the one-shot worker so the same model instance can retry with a clean runtime. Prepared inputs work directly; opaque photographs require a caller-supplied browser-local `removeBackground` implementation until a qualified BiRefNet artifact is bundled.

Packed tarballs pass clean Vite, strict Next-client TypeScript, and native ESM/import-map consumer checks. The exact final `@ai3d/triposplat-webgpu` tarball (`a2ba5ce6…`) was installed without workspace links into a fresh Vite app; its emitted worker first passed a tiny identity graph and then completed the full five-graph path with 262,144 finite Gaussians and both exports. This closes the packed-Vite full-execution gate, but not official whole-scene parity, full browser runs through Next/native ESM, repeated-generation memory, Edge, or 16 GB qualification.

## Architecture

The initial audit separates:

- model-neutral ONNX Runtime worker/session infrastructure;
- SHARP-specific image, camera, and Gaussian postprocessing;
- TripoSplat preprocessing and stochastic VAE conditioning;
- TypeScript-controlled classifier-free-guided flow sampling;
- dynamic octree algorithms;
- canonical Gaussian ownership and export.

Read the [browser architecture audit](docs/architecture-audit.md) for graph contracts, deployment behavior, attention memory constraints, and inherited-code boundaries.

## Documentation

- [Direct and agent integration](docs/agent-integration.md) — copy-ready image input to PLY/`.splat` onboarding
- [Current status and release gates](docs/current-status.md)
- [Developer guide](docs/developer-guide.md)
- [Alpha API reference](docs/api-reference.md)
- [Model hosting, CORS, and caching](docs/model-hosting.md)
- [Framework integration examples](docs/framework-integration.md)
- [Compatibility and measured benchmarks](docs/compatibility-and-benchmarks.md)
- [Gaussian conventions and export](docs/gaussian-conventions.md)
- [Privacy and security](docs/privacy-and-security.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Contribution and release process](docs/contributing-and-release.md)
- [License and provenance notices](THIRD_PARTY_NOTICES.md)
- [Changelog](CHANGELOG.md)

## Privacy

The inference design is browser-local: source images are decoded and processed in browser memory and are not sent to an inference server. A deployment still controls its own analytics, error reporting, authentication, and CDN logs. Review those systems separately, avoid logging signed model URLs, and enforce a Content Security Policy appropriate for workers, WebAssembly, WebGPU, and the selected model CDN.

## Licensing

Do not publish this repository or model bundle under a new blanket license yet. TripoSplat source and its released repository material are MIT-licensed at the pinned reference revision. Apple SHARP code uses Apple's repository license, while Apple SHARP model weights are restricted to research purposes. The inherited `ml-sharp-web` snapshot has no detected root license, so redistribution rights for copied chassis code must be clarified or that code must be reimplemented before a public package release.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project does not bundle SHARP or TripoSplat weights into an npm package.


## Authorship and provenance

TripoSplat WebGPU was directed, developed, tested, and validated by

[Yosun Chang](https://github.com/yosun), with substantial implementation

assistance from OpenAI Codex using GPT-5.6 Sol Ultra and GPT-5.6 Sol Max in Kiro.

The repository began from the `ml-sharp-web` application chassis, whose

original Git history is retained for provenance. Names shown by GitHub in

the contributor panel therefore include upstream contributors and do not

represent authorship of the TripoSplat WebGPU port.

The official TripoSplat implementation remains the numerical source of truth.
