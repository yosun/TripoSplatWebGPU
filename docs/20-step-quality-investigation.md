# Twenty-step quality and parity investigation

**Status:** diagnostic conclusion, 2026-07-16. No experimental graph described here is deployed.

## Executive conclusion

The 20-step WebGPU path is structurally correct and often visually acceptable, but it accumulates a value-dependent DiT error that is largest on the all-zero unconditional CFG branch. The first material divergence is in `context_refiner.0` attention, where 4,101 identical zero-condition embeddings create a long repeated-token reduction. Small prediction differences compound across 40 DiT calls and are amplified by the discrete eight-level octree into visible topology and Gaussian-scale changes.

The official shifted schedule, `1000 * timestep`, CFG expression, float32 rounding order, and Euler update are not implicated. Standalone octree occupancy and Gaussian decoder validation pass. Improving the quality path therefore requires a more faithful attention reduction, not a changed sampler or relaxed tolerance.

## Established evidence

- FAL and WebGPU PLYs are valid binary little-endian files with 262,144 vertices and the same 17 Gaussian properties; there is no malformed header, stride error, NaN contamination, or gross decoder corruption.
- The 20-step result tends toward slightly smaller Gaussian scales than the 4-step result. On some subjects this produces thinner geometry or holes.
- Browser final-state max latent error grows from `0.0044521` at four steps to `0.0487093` at twenty steps, approximately 10.9× across 40 DiT calls.
- Teacher-forced replay proves that the defect is not only autoregressive accumulation: later unconditional calls diverge even when each call receives the exact official state and timestep.
- Paired block probes localize the first material conditional/unconditional separation to `context_refiner_00_attention_residual`: ORT-versus-official max error is `3.0994e-6` conditional and `1.1063e-4` unconditional, a 35.69× ratio.
- At that boundary, adapted PyTorch remains close to untouched official PyTorch while ONNX Runtime diverges from both. Real RoPE, static Sobol positions, and host CFG/Euler arithmetic are not the initiating cause.
- FAL PLYs contain no provenance metadata. Exact service parity cannot be claimed without matching image preparation, features, initial noise, model revision, random stream, and export conventions.

Primary evidence: [`flow20 browser benchmark`](benchmarks/2026-07-15-flow20-fp32-webgpu.json) and [`invocation 7/8 block probes`](validation/2026-07-15-dit-flow4-invocations07-08-block-probes.json).

## Experiments and outcomes

| Experiment | Invocation-40 max latent error | Outcome |
| --- | ---: | --- |
| Canonical fp32 ONNX | `0.00105971` | Baseline |
| Context key-chunked online softmax | `0.00106657` | Slightly worse; removed |
| First-token selection for repeated context | `0.00105458` | No material change; removed |
| Stable RMS rewrite | Comparable unconditional failures | Not causal; not deployed |
| One representative plus exact multiplicity bias | `0.00092310` | Modest improvement, but adapter parity failed |
| Sixteen representatives plus `4101/16` bias | `0.00092727` | No further benefit; temporary change reverted |

The one-representative ONNX graph closely matched its adapted PyTorch specialization (`5.97e-5` max latent error), but that specialization differed from untouched official PyTorch by `9.29e-4`. The problem is therefore the changed floating-point reduction order, not an ONNX transcription failure in that candidate. Its 1.633 GB sidecar was byte-identical to the canonical sidecar, proving a future specialized graph need not duplicate weights.

## Decisions and next direction

1. Keep the canonical graph for both CFG passes; do not add runtime routing for the collapsed candidate.
2. Preserve the official sampler and public float32 state. Do not tune guidance, schedule, or tolerances to conceal graph error.
3. Keep collapsed-context support diagnostic-only until it passes untouched-official PyTorch and Chrome/Edge WebGPU gates.
4. Focus the next bounded investigation inside `context_refiner.0`: Q/K normalization, logits, softmax, value accumulation, and output projection for an exact unconditional invocation.
5. Prefer a custom fused WebGPU online-softmax/value-accumulation kernel, or another implementation that preserves the official reduction behavior, over additional graph-level token-collapse variants.
6. Qualify any candidate in order: untouched official vs adapted PyTorch, ORT CPU vs official, several unconditional trajectory calls, browser teacher-forced replay, full autoregressive 20-step state, then fixed-camera PLY renders against current WebGPU, Hugging Face, and FAL.

Until those gates pass, the correct product statement is: 20-step WebGPU generation completes and may look acceptable, but it is not yet quality/parity-qualified against the official implementation.