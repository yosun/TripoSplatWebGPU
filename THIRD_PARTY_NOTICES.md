# License and provenance notices

This file records known source and model boundaries for the current engineering checkout. It is not legal advice and is not a substitute for shipping the complete license texts required by each dependency.

## Release status

No blanket root license is granted for this combined repository. The alpha `@ai3d/triposplat-webgpu` package is marked `UNLICENSED`, and `@ai3d/gaussian-scene` does not yet declare a package license. Do not publish source, packages, or model bundles until inherited-code rights are resolved and a release review has produced complete notices.

Model weights are not intended to be included in npm tarballs.

## Direct implementation references

### TripoSplat

- repository: [VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat);
- pinned numerical reference: `a78fa12d06dbf1381ca548bfac32bb68cb8c451d`;
- repository license at that revision: [MIT License](https://github.com/VAST-AI-Research/TripoSplat/blob/a78fa12d06dbf1381ca548bfac32bb68cb8c451d/LICENSE).

The official PyTorch code and released repository material are the numerical source of truth. Before hosting or redistributing weights, preserve the upstream license and notices and verify the exact checkpoint source rather than assuming every similarly named artifact has identical terms.

### Apple SHARP

- repository: [apple/ml-sharp](https://github.com/apple/ml-sharp);
- pinned reference: `1eaa046834b81852261262b41b0919f5c1efdd2e`;
- code terms: [LICENSE](https://github.com/apple/ml-sharp/blob/1eaa046834b81852261262b41b0919f5c1efdd2e/LICENSE);
- model terms: [LICENSE_MODEL](https://github.com/apple/ml-sharp/blob/1eaa046834b81852261262b41b0919f5c1efdd2e/LICENSE_MODEL);
- additional upstream notices: [ACKNOWLEDGEMENTS](https://github.com/apple/ml-sharp/blob/1eaa046834b81852261262b41b0919f5c1efdd2e/ACKNOWLEDGEMENTS).

SHARP model weights are limited to research purposes under Apple's model agreement and must not be represented as generally commercial-use weights. The SHARP baseline and TripoSplat model are separate paths with separate provenance.

### ml-sharp-web chassis

- repository: [bring-shrubbery/ml-sharp-web](https://github.com/bring-shrubbery/ml-sharp-web);
- pinned chassis snapshot: `01ff783f782a0eab1eb0dbb533d51695dc526df6`;
- detected root license at that snapshot: none.

The absence of a detected license is not permission to redistribute. Before public release, obtain clarification from the copyright holder or replace inherited implementation with independently written, properly licensed modules while preserving only non-copyrightable interfaces and behavior as appropriate.

### TripoSplat Mac/Core AI decomposition

- repository: [john-rocky/coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/TripoSplatMac);
- pinned reference: `ede54d103c191edda7d93b0e6c3c47ea0a0664c0`.

This implementation uses that project only to understand useful static graph boundaries. It is not a numerical source of truth. Review its repository license and notices before copying any source or exported assets; none should be assumed inherited merely because the graph boundary is similar.

## Runtime and package dependencies

The JavaScript dependency graph includes ONNX Runtime Web, React, Vite, TypeScript, Three.js, Gaussian viewers, and their transitive dependencies. Their licenses are recorded in the lockfile/package metadata, not reproduced here.

Before each release:

1. generate a production dependency license inventory from the exact lockfile;
2. include required license and notice texts in the appropriate npm tarball or distribution;
3. review worker, WASM, and viewer assets separately from JavaScript source;
4. confirm optional viewers are not pulled into the core package;
5. reject packages containing ONNX graphs, external data, checkpoints, fixtures, or license-incompatible assets.

## Exported model provenance

Every hosted model revision should publish a provenance record containing:

- source repository and commit;
- clean/dirty source state;
- source checkpoint URL or controlled origin;
- source checkpoint SHA-256;
- exporter command and environment lock;
- all adapter changes required for WebGPU;
- ONNX graph and external-data hashes/byte counts;
- graph contract and numerical validation reports;
- applicable code, model, dataset, and submodel license notices.

DINOv3 and Flux VAE components may carry upstream obligations distinct from the top-level TripoSplat repository. Their exact checkpoint cards and licenses must be reviewed before a production model manifest is distributed.

## User responsibility

Application developers are responsible for determining whether their use of model weights, generated outputs, viewers, and dependencies complies with applicable licenses and law. This repository's technical validation does not grant rights to third-party materials.
