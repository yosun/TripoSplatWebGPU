# TripoSplat ONNX export and parity tooling

These scripts export fixed-shape browser graphs from a local checkout of the [official TripoSplat repository](https://github.com/VAST-AI-Research/TripoSplat), generate deterministic reference fixtures, and compare ONNX Runtime output with untouched official PyTorch. The official repository is the numerical source of truth; the Core AI port is used only as a guide to graph boundaries and conversion hazards.

No script implicitly downloads model weights, uploads an image, or edits the official checkout. Generated graphs, sidecars, and large fixtures remain local deployment/test artifacts and are excluded from npm packages.

## Environment

Use a Python version supported by the selected PyTorch wheel, normally Python 3.10–3.12:

```bash
python3.12 -m venv .venv-triposplat
source .venv-triposplat/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/triposplat/requirements.txt
```

On Apple Silicon, install native arm64 PyTorch. Export devices are selectable per script. Public graph inputs and outputs are normally fp32 even when a graph uses lower internal precision. The current correctness manifest uses fp32 artifacts for all five neural graphs.

Run any script with `--help` for its complete and authoritative options.

## Tool index

| Stage | Export/reference tool | Validator/browser fixture tool | Purpose |
| --- | --- | --- | --- |
| Image/VAE input | `make_encoder_fixture.py` | `pack_browser_fixture.py` | Build a prepared alpha-bearing 1024px image, explicit VAE noise, and browser-readable VAE assets |
| DINOv3 | `export_dinov3_onnx.py` | `validate_dinov3_onnx.py`, `pack_dinov3_browser_fixture.py` | Export and gate `pixel_values [1,3,1024,1024] → feature1 [1,4101,1280]` |
| Flux VAE | `export_flux2_vae_encoder_onnx.py` | `validate_flux2_vae_encoder_onnx.py`, `pack_browser_fixture.py` | Export and gate `image_rgb` plus explicit `epsilon` to `feature2 [1,4101,128]` |
| One DiT call | `export_dit_onnx.py` | `validate_dit_onnx.py`, `pack_dit_browser_fixture.py` | Export and gate one latent/camera prediction with fixed conditioning tensors |
| DiT diagnostics | `make_dit_probe_fixture.py` | `dit-lab.html` | Add selected block-boundary outputs to a graph and record CPU ORT probe references |
| Flow loop | `make_flow_fixture.py` | `flow-lab.html` | Produce official 4-step/20-step final states and optional teacher-forced per-invocation trajectories |
| Octree neural boundary | `export_octree_occupancy_onnx.py` | `validate_octree_occupancy_onnx.py`, `pack_decoder_browser_fixture.py` | Export and gate one parent-to-eight-child occupancy-logit invocation |
| Full octree trajectory | `make_octree_trajectory_fixture.py` | `octree-lab.html` | Record and replay all data-dependent eight-level neural and host-side sampling inputs |
| Gaussian neural boundary | `export_gaussian_decoder_onnx.py` | `validate_gaussian_decoder_onnx.py`, `pack_decoder_browser_fixture.py` | Export and gate raw `features [1,8192,480]` |
| Gaussian activation oracle | `make_gaussian_activation_fixture.py` | package tests | Gate host `_build_gaussians` activation and packing math against official PyTorch |

Shared implementation helpers live in `dinov3_common.py`, `flux2_vae_common.py`, `dit_common.py`, and `decoder_onnx_common.py`. They load the official model definitions, enforce fixed contracts, consolidate ONNX external data, record provenance, and compute deployment hashes.

## Export and validate a neural stage

Every exporter accepts a local official checkout, local checkpoint, output graph, internal precision, and trace device. For example:

```bash
python scripts/triposplat/export_dinov3_onnx.py --triposplat-repo /path/to/TripoSplat --weights /path/to/dino_v3_vit_h.safetensors --output public/models/triposplat/dinov3_encoder_fp32.onnx --internal-precision fp32 --device mps

python scripts/triposplat/validate_dinov3_onnx.py --triposplat-repo /path/to/TripoSplat --weights /path/to/dino_v3_vit_h.safetensors --onnx public/models/triposplat/dinov3_encoder_fp32.onnx --fixture /path/to/dino-inputs.npz --report /path/to/dino-validation.json
```

The VAE, DiT, octree, and Gaussian exporter/validator pairs follow the same pattern. Validators check public names, dtypes, fixed shapes, finite values, elementwise tolerance, cosine similarity, and provenance. They can emit JSON reports and reusable NPZ fixtures. Browser packers expand validated NPZ arrays into little-endian static files plus a manifest consumed by the corresponding lab.

Exporters normally create one consolidated `<graph>.onnx.data` sidecar. The deployment manifest's external-data `path` must exactly match the location embedded in the ONNX graph.

## Flow fixtures and teacher-forced trajectories

`make_flow_fixture.py` feeds a validated DiT input fixture into the untouched official `FlowEulerCfgSampler` and writes the final latent/camera state for any positive step count. Use the official fast/quality settings with `--steps 4` or `--steps 20`, `--guidance-scale 3`, and `--shift 3`.

Add `--record-trajectory` to write the exact official sample, scaled timestep, and raw latent/camera prediction for every conditional and unconditional DiT call:

```bash
python scripts/triposplat/make_flow_fixture.py --triposplat-repo /path/to/TripoSplat --weights /path/to/triposplat_fp16.safetensors --input-fixture-dir public/fixtures/generated/dit-step-fp32-compute --output-fixture-dir public/fixtures/generated/flow4-fp32-trajectory --device mps --internal-precision fp32 --steps 4 --record-trajectory
```

The teacher-forced flow lab injects each recorded official state directly into the browser graph. This separates invocation-dependent graph error from autoregressive CFG/Euler accumulation.

The DiT validator can consume one recorded call directly, including the implicit all-zero tensors used by the official unconditional branch. This avoids duplicating the large conditioning tensors into another NPZ and makes a failing browser invocation reproducible in untouched PyTorch, adapted PyTorch, and CPU ONNX Runtime:

```bash
python scripts/triposplat/validate_dit_onnx.py --triposplat-repo /path/to/TripoSplat --weights /path/to/triposplat_fp16.safetensors --onnx public/models/triposplat/dit_step_webgpu_fp32.onnx --trajectory-fixture-dir public/fixtures/generated/flow4-fp32-trajectory --trajectory-invocation 8 --device mps --atol 0.0001 --rtol 0.001 --report /path/to/invocation-08.json
```

`export_dit_onnx.py --rms-norm-eps ...` enables an experimental explicit stable-RMS rewrite inspired by the Core AI conversion notes. It is a diagnostic variant, not the canonical graph and not a parity or performance result. A variant must pass the same untouched-official gates before it can replace the default official `F.normalize` behavior.

To localize a value-dependent failure without retaining every 12,294-token activation, add the `blocks` diagnostic outputs and compare a conditional/unconditional trajectory pair. The validator retains only a `[1,4,16]` slice at embeddings, attention residuals, refiner outputs, and joint-block outputs; it does not replace the full-output gate:

```bash
python scripts/triposplat/make_dit_probe_fixture.py --graph public/models/triposplat/dit_step_webgpu_fp32.onnx --sidecar public/models/triposplat/dit_step_webgpu_fp32.onnx.data --input-fixture-dir /path/to/flat-invocation-inputs --output-graph /tmp/dit-block-probes.onnx --output-fixture-dir /tmp/dit-block-probe-fixture --probe-set blocks
python scripts/triposplat/validate_dit_block_probes.py --triposplat-repo /path/to/TripoSplat --weights /path/to/triposplat_fp16.safetensors --probe-onnx /tmp/dit-block-probes.onnx --trajectory-fixture-dir public/fixtures/generated/flow4-fp32-trajectory --invocation 7 --invocation 8 --device mps --report /tmp/dit-block-probe-report.json
```

## Full octree trajectory

The occupancy ONNX graph deliberately excludes softmax, systematic resampling, compaction, child expansion, and final voxel jitter. `make_octree_trajectory_fixture.py` records those data-dependent boundaries from the untouched official eight-level sampler:

- normalized parent centers and resolution for each neural call;
- official raw child logits;
- parent counts and sampled child counts;
- the uniform variate assigned to each parent by official `sample_probs`;
- final points, log probabilities, and per-point voxel jitter.

```bash
python scripts/triposplat/make_octree_trajectory_fixture.py --triposplat-repo /path/to/TripoSplat --weights /path/to/triposplat_vae_decoder_fp16.safetensors --condition public/fixtures/generated/flow4-fp32-compute/flow4_latent.f32 --output-fixture-dir public/fixtures/generated/octree-trajectory-flow4-fp32 --device mps --internal-precision fp32
```

Recording the official uniform values and jitter lets the TypeScript sampler replay the exact path without depending on PyTorch and JavaScript random-number generators matching.

## Gaussian activation oracle

The Gaussian ONNX graph stops at 480 raw features per octree point. `make_gaussian_activation_fixture.py` calls upstream `triposplat._build_gaussians` directly, without decoder weights, and records expected positions, scales, rotations, opacities, and degree-zero spherical harmonics for deterministic probe values:

```bash
python scripts/triposplat/make_gaussian_activation_fixture.py --triposplat-repo /path/to/TripoSplat --output packages/triposplat-webgpu/test/fixtures/gaussian-activation-official.json
```

This oracle covers host activation equations only. A passing small oracle does not qualify a live eight-level browser decoder result, final scene orientation, or rendered output.

## Evidence policy

Keep machine-readable reports under `docs/validation/` or `docs/benchmarks/`. Each record should include the official source revision, checkpoint identity, graph and sidecar hashes, precision, device/provider, thresholds, output errors, and measured timings. Never loosen a threshold solely to relabel a failure as a pass, and never present Python timings as Chrome/Edge WebGPU results.
