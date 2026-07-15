# Consumer API examples

These examples are contract fixtures for the intended `@ai3d/triposplat-webgpu` API.

> **Milestone warning:** the package now has a built-in five-stage executor, canonical fp32 manifest, persistent verified caches, strict DINO/one-step-DiT gates, a passing full octree trajectory, and packed consumer coverage. These examples are still not end-to-end-qualified: opaque photos need local background removal, the measured 20-step loop fails its accumulated-drift gate, and a live activated/rendered final scene remains open. Do not publish these as a production demo until the release gates in [`docs/current-status.md`](../docs/current-status.md) pass.

| Example | Demonstrates |
| --- | --- |
| [`vanilla/`](vanilla/) | plain browser ESM lifecycle and local file input |
| [`vite/main.ts`](vite/main.ts) | normal package import and Vite worker handling |
| [`react/TripoSplatPanel.tsx`](react/TripoSplatPanel.tsx) | component lifecycle, progress, repeated input, cleanup |
| [`next/app/triposplat/TripoSplatClient.tsx`](next/app/triposplat/TripoSplatClient.tsx) | Next.js client-component boundary |
| [`signed-fetch/load.ts`](signed-fetch/load.ts) | custom authenticated fetch and signed manifest URL |
| [`cancellation/generate.ts`](cancellation/generate.ts) | cancellation, typed error, retry, disposal |
| [`export/download.ts`](export/download.ts) | PLY blob download and scene ownership |

Production examples must install the packed package into a fresh app, point at a complete immutable model manifest, and pass a browser test before this warning is removed.

All inference inputs stay local. Example network requests are limited to package assets and configured model manifests/graphs; a host application may have additional analytics or telemetry that must be audited separately.
