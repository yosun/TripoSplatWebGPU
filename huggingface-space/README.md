---
title: TripoSplat WebGPU
emoji: 🫧
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
models:
  - Yosun/TripoSplat-WebGPU
---

# TripoSplat WebGPU

Generate a Gaussian splat from an image entirely in your browser with WebGPU. The static app runs model inference and exports locally; it does not use Hugging Face inference compute.

## Before you run

- Use desktop Chrome with WebGPU. The recorded qualification environment is Chrome 150 on an Apple M3 Max with 128 GB unified memory; Edge and 16 GB devices are not yet qualified.
- Allow roughly 6.47 GB for the first verified model download and browser cache; persistence depends on available browser storage and quota.
- Choose PNG, WebP, JPEG, or AVIF. Alpha-bearing images are ready to run; opaque photos automatically start browser-local background removal and TripoSplat framing. The separate preparation model downloads and caches on its first use.
- The public runner requests 20 sampling steps / 40 CFG DiT calls and can take several minutes on the recorded high-end hardware. It is an execution path, not a 20-step parity or quality pass.

The app downloads versioned model artifacts directly from [Yosun/TripoSplat-WebGPU](https://huggingface.co/Yosun/TripoSplat-WebGPU), verifies their declared size and SHA-256, and caches them in browser storage when permitted. It exports binary PLY and `.splat` files locally. The model layer does not upload source pixels, though deployment analytics, CDN logs, browser extensions, and image URL hosts are separate privacy considerations.

> **Engineering preview:** the four-step prepared-image path has a structural/export/viewer pass but misses a stricter diagnostic. Whole-scene numerical/render parity, repeated-run memory behavior, Edge, and 16 GB devices are not yet qualified.

[Source, status, benchmarks, and issue tracker](https://github.com/yosun/TripoSplatWebGPU) · [Vercel mirror](https://triposplat-webgpu.vercel.app/e2e-web)

## Deploy this card and the prebuilt app

Run from the repository root:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm build
hf upload Yosun/TripoSplat-WebGPU-Demo dist . --repo-type space \
  --commit-message 'Deploy TripoSplat WebGPU demo'
hf repos cp huggingface-space/README.md \
  hf://spaces/Yosun/TripoSplat-WebGPU-Demo/README.md
```

The root `dist/index.html` is already built; this card intentionally uses only `sdk: static` and does not request Hugging Face compute or a custom build command. Hugging Face may still require plan credits to activate updated Static Space hosting.