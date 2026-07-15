# Packed consumer checks

These fixtures validate the npm tarballs from outside the workspace:

- `vite` performs a fresh installed-package Vite build and checks that the
  TripoSplat module worker and ONNX Runtime WASM assets are emitted. The built
  fixture accepts `?run=1` for the tiny WebGPU identity smoke and `?full=1`
  for the deterministic prepared-image five-graph run. `fixtureOrigin` and
  `modelBaseUrl` query parameters can point the full run at separate CORS-enabled
  fixture/model origins.
- `next-client` strictly typechecks a minimal App Router-style `'use client'`
  component against the installed tarballs. It intentionally does not install
  or invoke Next.js/Turbopack; that remains a release-candidate integration gate.
- `import-map` copies only installed tarball contents into a static vendor tree
  and validates the native ESM/import-map dependency graph plus worker/WASM
  files. The CI check does not launch a browser or run WebGPU inference.

Run all three build-time checks with `pnpm test:package-consumer`.

Recorded browser evidence:

- [packed Vite worker/WebGPU smoke](../../docs/validation/2026-07-15-packed-vite-consumer-webgpu-chrome.json);
- [exact final-tarball full five-stage run](../../docs/benchmarks/2026-07-15-packed-vite-full-e2e-webgpu-fp32-chrome.json).

The full report is a structural/export gate. It does not claim official
whole-scene numerical or rendered-pixel parity.
