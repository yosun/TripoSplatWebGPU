# Contribution and release process

## Contribution principles

The official TripoSplat PyTorch repository is the numerical source of truth. Keep SHARP working while moving reusable runtime code into model-neutral modules. Treat Core AI graphs only as decomposition guidance.

For every model or numerical change:

1. pin the official source commit and verify the checkout is clean;
2. record checkpoint origin and checksum;
3. define fixed graph names, shapes, dtypes, and timestep semantics;
4. compare any export adapter with untouched official PyTorch;
5. compare ONNX Runtime with untouched official PyTorch;
6. run the same deterministic fixture through browser WebGPU with fallback disabled;
7. record thresholds before evaluating the result;
8. store a machine-readable report and exact environment;
9. update the stage ledger honestly, including failures.

Do not commit credentials, signed URLs, proprietary images, or model weights whose redistribution terms are unclear.

## Pull-request checks

The GitHub Actions workflow performs install, workspace package builds, typecheck, unit tests, lint, application build, package dry-run packing, publish-file safety checks, and documentation hygiene. Browser parity jobs will be added once their model artifacts can be fetched from an authorized CI store without publishing restricted or multi-gigabyte objects as repository artifacts.

A change that affects browser numerics must include its browser report even when generic CI is green.

## Versioning

Use semantic versioning only after an installable public package exists:

- patch: compatible fixes without numerical-contract or public-type changes;
- minor: backward-compatible API or graph-manifest capability;
- major: incompatible API, manifest, cache, Gaussian convention, or required-graph change.

Before `1.0.0`, every release must still explain compatibility and migration impact. Model revisions are independent of npm package versions and must be explicit in scene metadata and manifests.

## Release checklist

1. resolve redistribution rights for inherited `ml-sharp-web` code and choose a valid package license;
2. complete notices for package source, TripoSplat code/weights, SHARP code/weights, exporters, and bundled runtime assets;
3. run all unit, worker, external-data, signed-URL, interruption, cache, cancellation, leak, export round-trip, browser, and bundler tests;
4. export all graphs from a clean pinned official checkout;
5. validate every graph and both four-step and twenty-step final states;
6. generate an immutable manifest with byte counts and SHA-256 for all graph and external-data objects;
7. upload models to an object CDN and verify CORS, ranges, caching, signed URL refresh, integrity failure, and interrupted resume;
8. build ESM, declarations, source maps, worker assets, and any WASM assets;
9. inspect package contents and reject fixtures, model weights, viewers, benchmarks, React, Node-only modules, credentials, and absolute development paths;
10. install the packed tarball into clean Vite, Next.js, Rollup/webpack, and plain ESM consumers;
11. run compatibility, generate, cancel, retry, repeated generation, export, render, and dispose in those consumers;
12. benchmark actual supported devices, including a 16 GB Apple Silicon Mac and a Windows WebGPU target;
13. update the changelog with measured results and known limitations;
14. publish with provenance and verify the installed package from the registry.

## Package inspection

Before publication:

```bash
pnpm -r --if-present build
pnpm --filter @ai3d/triposplat-webgpu pack --pack-destination artifacts/packs
tar -tf artifacts/packs/ai3d-triposplat-webgpu-*.tgz
```

The exact package filter becomes valid only after the package exists. The automated workflow conditionally performs equivalent inspection for every non-private workspace package.

## Release definition

The work is not complete when an export script succeeds, an internal lab passes, or an application build renders. It is complete only when an external developer can install the packed package, point it at hosted complete model files, generate locally, receive real progress, cancel and retry, export/render a canonical scene, and dispose resources without reading repository source.
