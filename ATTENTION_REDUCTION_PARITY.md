# Context attention reduction parity

**Status:** no-go, diagnostic only, 2026-07-16. No graph, sampler, model, tolerance, or product runtime was replaced.

## Decision

The first **material** unconditional divergence inside canonical `context_refiner.0` is the probability-times-V reduction. Q/K normalization and scaled logits have small upstream error, while the captured normalized probabilities are bit-identical between canonical ORT CPU and ORT WebGPU for invocation 8. The weighted-value boundary then jumps to `2.0122528e-4` max error for ORT CPU versus untouched official PyTorch. Chrome WebGPU differs from ORT CPU by another `1.8501282e-4` at the same boundary. Output projection amplifies the ORT CPU error to `8.3923340e-4` max before residual addition.

This is not a WebGPU-only defect: ORT CPU reproduces it. WebGPU weighted-value reduction is confirmed as an additional source of browser/CPU separation, but not as the initiating cross-backend cause.

The implementation decision is **no-go**. None of the measured materialized, sequential, pairwise, blockwise, or fused online-softmax/value variants improved weighted-value RMSE over canonical ORT on both invocation 7 and invocation 8. No WGSL kernel or production feature flag was added. The canonical graph remains the only deployed path.

Machine-readable evidence is in [`attention-reduction-results.json`](attention-reduction-results.json). The browser comparison is in [`docs/validation/2026-07-16-context0-attention-webgpu.json`](docs/validation/2026-07-16-context0-attention-webgpu.json).

## Reproduced localized baseline

The original paired invocation 7/8 command was rerun unchanged except for a new temporary report destination. It reproduced the prior result exactly:

| Boundary | Conditional ORT CPU vs official max | Unconditional ORT CPU vs official max | Ratio |
| --- | ---: | ---: | ---: |
| `context_refiner_00_attention_residual` | `3.0994415e-6` | `1.1062622e-4` | `35.6923077×` |

The official source remained clean at commit `a78fa12d06dbf1381ca548bfac32bb68cb8c451d`; torch was `2.13.0`, ORT was `1.27.0`, PyTorch used MPS, and ORT used `CPUExecutionProvider` with `ORT_DISABLE_ALL` and eight threads. The canonical deployment graph/sidecar hashes remained `558a52ea…` and `eacadf21…`. The reproduced temporary report is `/private/tmp/dit-block-probe-20260715/report-attention-reproduced-20260716.json`; canonical checked-in evidence was not overwritten.
## Exact capture and reproducibility

`make_dit_probe_fixture.py --probe-set context0` adds 17 diagnostic outputs to a copy of the canonical graph. It does not rewrite canonical operators. The diagnostic graph is 10,694,337 bytes, SHA-256 `9a1bbf841833c265e11ed46ca8ee3df23b1a7acb8d82e217bd7e5f338ca41916`, and hard-links the byte-identical canonical 1,633,210,368-byte sidecar. It is not a deployment candidate.

Captured direct boundaries are block input, normalized attention input, QKV, Q/K after RoPE, V, normalized Q/K, four transposed query rows, all K/V rows, scaled logits, normalized probabilities, weighted-value output, pre-projection output, post-projection output, and residual output. Shapes and SHA-256 hashes for untouched official, adapted PyTorch, and ORT CPU tensors are recorded in the JSON report. Raw `.npy` captures are under `/private/tmp/dit-attention-parity-20260716/captures`; they are intentionally not committed.

The MPS SDPA API does not expose row maximum, shifted logits, exponentials, or denominator. Those arrays were deterministically derived from exact hooked official normalized Q/K and are explicitly labeled `derived_not_captured`. Replacing canonical `Softmax` with a decomposed graph merely to expose internals would change the operation under test, so it was not done.

Recreate the diagnostic graph and report with:

```bash
/private/tmp/triposplat-onnx-venv/bin/python scripts/triposplat/make_dit_probe_fixture.py --graph public/models/triposplat/dit_step_webgpu_fp32.onnx --sidecar public/models/triposplat/dit_step_webgpu_fp32.onnx.data --input-fixture-dir /private/tmp/dit-attention-parity-20260716/input-inv08 --output-graph /private/tmp/dit-attention-parity-20260716/dit_context0_attention.onnx --output-fixture-dir /private/tmp/dit-attention-parity-20260716/ort-context0-inv08 --probe-set context0 --session-threads 8
/private/tmp/triposplat-onnx-venv/bin/python scripts/triposplat/validate_context_attention_parity.py --triposplat-repo /private/tmp/TripoSplat --source-commit a78fa12d06dbf1381ca548bfac32bb68cb8c451d --weights /private/tmp/triposplat-weights/diffusion_models/triposplat_fp16.safetensors --weights-sha256 c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20 --probe-onnx /private/tmp/dit-attention-parity-20260716/dit_context0_attention.onnx --trajectory-fixture-dir public/fixtures/generated/flow4-fp32-trajectory --invocation 7 --invocation 8 --artifact-dir /private/tmp/dit-attention-parity-20260716/captures --write-raw-tensors --device mps --session-threads 8 --atol 0.00001 --rtol 0.00001 --report attention-reduction-results.json
```

## Boundary attribution

Invocation-8 ORT CPU versus untouched official PyTorch:

| Boundary | Max abs | Mean abs | RMSE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Block input | `0` | `0` | `0` | Exact input |
| Attention input | `4.7684e-7` | `7.9926e-8` | `1.1047e-7` | Immaterial |
| QKV | `2.2888e-5` | `6.3113e-7` | `1.0538e-6` | Small projection/backend error |
| Q after RoPE | `2.2888e-5` | `9.0005e-7` | `1.5295e-6` | No material amplification |
| K after RoPE | `8.2254e-6` | `7.6898e-7` | `1.1033e-6` | No material amplification |
| Q normalized | `4.5300e-6` | `4.1161e-7` | `6.0513e-7` | Normalization reduces error |
| K normalized | `2.9802e-6` | `4.6825e-7` | `6.5123e-7` | Normalization reduces error |
| Scaled logits, derived official comparison | `1.4305e-6` | `8.0187e-7` | `8.6882e-7` | Small |
| Normalized probabilities | `0` | `0` | `0` | Bit-identical captured rows |
| Probability × V | `2.0123e-4` | `3.0922e-5` | `4.5354e-5` | First material sub-operation |
| Post projection | `8.3923e-4` | `4.4548e-5` | `6.0775e-5` | Amplifies earlier error |
| Residual | `8.3923e-4` | `4.4549e-5` | `6.0775e-5` | Carries projection error |

Adapted PyTorch is exact at invocation-8 weighted-value output and differs by `9.1553e-5` max only after its adapted output projection. This reinforces that the dominant canonical ORT attention error is not initiated by Q/K normalization, logits, or softmax normalization.
## Untouched official backend

Verified facts:

- PyTorch `2.13.0`, MPS device, model/internal dtype fp32.
- Official source calls `torch.nn.functional.scaled_dot_product_attention` with layout `[1,16,4101,64]`, no mask, dropout zero, and no deterministic-algorithm requirement.
- Profiler dispatch is `aten::_scaled_dot_product_attention_math_for_mps` beneath `aten::scaled_dot_product_attention`; FlashAttention, xFormers, CUDA memory-efficient attention, and custom project attention are not active.
- A four-query replay of exact captured Q/K/V dispatches to the same operator but differs from the original 4,101-query call by `1.9788742e-4` max and `4.5482865e-5` RMSE on the retained rows. Therefore the backend's numerical reduction depends on full query geometry/tiling even though rows are mathematically independent.

Not observable and therefore not claimed as fact: MPS accumulator precision, whether probabilities are physically materialized, internal tile sizes, and the exact reduction tree. The query-length experiment strongly indicates geometry-dependent tiling/reduction order, but does not identify its source implementation details.

## Captured-tensor laboratory

All candidates used real canonical invocation tensors. Metrics below are weighted-value RMSE against untouched official output; lower is better.

| Candidate | Invocation 7 conditional | Invocation 8 unconditional | Result |
| --- | ---: | ---: | --- |
| Canonical ORT CPU | `1.6756651e-7` | `4.5354315e-5` | Baseline |
| Adapted PyTorch | `2.8443796e-8` | `0` | Diagnostic, not deployable kernel |
| Float64 materialized | `1.1578065e-7` | `4.5447353e-5` | Worse unconditional |
| Straight f32 materialized | `1.5082024e-7` | `5.9429572e-5` | Worse unconditional |
| Sequential f32 probability×V | `1.5082024e-7` | `5.9429572e-5` | Worse unconditional |
| Pairwise/tree f32 | `1.1672181e-7` | `4.5447641e-5` | Worse unconditional |
| Best fused online, block 64 | `1.1858132e-7` | `4.5439442e-5` | Improves conditional; `8.51e-8` worse unconditional |
| Fused online, blocks 16–1024 | `1.18e-7`–`1.23e-7` | `4.54e-5`–`5.00e-5` | No robust winner |
| Separate blockwise probabilities, blocks 16–1024 | `1.50e-7`–`1.54e-7` | `5.9429572e-5` | Consistently worse |

Per-head, per-row, runtime, repeated-V exactness, and temporary-memory estimates are in the machine report. Candidate timings are diagnostic NumPy timings, not browser kernel benchmarks.

## ORT CPU versus WebGPU

Chrome 150 on Apple M3 Max/Metal ran the identical invocation-8 diagnostic graph through WebGPU with WASM fallback disabled. Against ORT CPU outputs from that graph, normalized-Q max error was `4.5300e-6`, scaled-logit max error was `1.9073e-6`, and probabilities were bit-identical. The first material WebGPU/CPU split was again weighted-value accumulation: max `1.8501282e-4`, mean `2.9788754e-5`, RMSE `4.3699567e-5`. Model load was 32.439 s, inference 28.313 s, and readback 17.1 ms. This is a bounded diagnostic run, not a production performance benchmark.

## Why the prior key-chunked online-softmax experiment failed

The removed implementation is not present in repository history, so its exact WGSL/graph reduction tree cannot be audited. The new measurements establish the relevant explanation without inventing unavailable details:

1. A generic online formulation does not match the target MPS math backend's geometry-dependent reduction order.
2. Separate probability materialization plus V accumulation is consistently worse than ORT on the unconditional call.
3. Fusing online softmax and V accumulation helps relative to the separate form, but all tested block sizes still miss untouched official output; block 64 is `8.51e-8` RMSE worse than ORT on invocation 8.
4. Replaying only four query rows through the same MPS operator changes retained outputs materially, proving that matching formulas and key block size alone is insufficient.

The recorded earlier regression (`0.00105971` to `0.00106657` at invocation 40) is therefore consistent with a reduction-order/tile-geometry mismatch and extra rounding boundaries. Claims about its exact workgroup structure, exp implementation, or buffer writes would be speculative.
## Go/no-go and validation scope

**No-go.** A production intervention was forbidden by the evidence gate because no candidate materially and robustly improved untouched-official parity. Consequently there is no justified WGSL source, feature flag, graph split, custom ORT operation, or hybrid runtime to qualify. The narrow browser change only lets `dit-lab.html` accept `model` and `fixture` query parameters for reproducible diagnostic runs; it does not affect generation.

Completed validation:

- original invocation 7/8 localized baseline reproduction;
- exact official/adapted/ORT CPU boundary captures with tensor hashes;
- real-tensor conditional and unconditional candidate matrix;
- per-head and per-query-row metrics;
- exact MPS SDPA dispatch profiling;
- Chrome WebGPU bounded invocation-8 capture with fallback disabled;
- Python compile checks, TypeScript diagnostics, repository typecheck, tests, and build.

Stopped by the predeclared no-go gate and intentionally not rerun: full `context_refiner.0` replacement, multi-invocation teacher-forced candidate replay, autoregressive 4-step/20-step candidate generation, octree/Gaussian/PLY candidate comparisons, fixed-camera renders, peak production memory, Edge, and target 16 GB hardware. Running those gates for a candidate that already fails captured-tensor parity would not justify deployment and would repeat expensive broad investigation.

Canonical 20-step metrics therefore remain unchanged: browser final-state latent max `0.0487092733`, mean `0.00035171698`, RMSE `0.00081205185`; canonical invocation-40 CPU metric remains `0.00105971`. No before/after production metric exists because no candidate was integrated.

## Qualification statement and next action

The faithful product statement remains: the canonical 20-step WebGPU path completes but is not official-quality/parity-qualified. No exact FAL parity is claimed; FAL provenance remains unavailable.

The single highest-value next action is a controlled **full-query-geometry MPS reduction study**: hold the first captured Q rows and all K/V fixed, sweep total query count/padding and inspect retained outputs to infer the tile transition used by `_scaled_dot_product_attention_math_for_mps`. Only after a reduction tree reproduces the original 4,101-query MPS output should it be implemented as an isolated WGSL kernel and reconsidered for integration.
