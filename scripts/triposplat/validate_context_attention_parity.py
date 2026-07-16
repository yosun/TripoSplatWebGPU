#!/usr/bin/env python3
"""Diagnose first context-refiner attention parity on exact flow invocations.

Consumes a graph made by ``make_dit_probe_fixture.py --probe-set context0``.
Untouched official PyTorch remains the sole truth source; softmax internals that
MPS SDPA does not expose are explicitly labelled as derived from hooked Q/K/V.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from dit_common import (
    CONDITION_TOKENS,
    OFFICIAL_REPOSITORY_URL,
    adapt_official_flow_for_onnx,
    choose_torch_device,
    comparison_metrics,
    load_official_flow_model,
    make_browser_flow_step,
    metadata_dict,
    sha256_file,
    source_revision,
    synchronize_torch,
    torch_inputs_from_numpy,
    verify_external_data_files,
)
from validate_dit_block_probes import adapter_configuration
from validate_dit_onnx import load_trajectory_invocation_fixture

HEADS = 16
HEAD_DIM = 64
MODEL_DIM = HEADS * HEAD_DIM
ROWS = 4
BLOCK_SIZES = (16, 32, 64, 128, 256, 512, 1024)
PROBES = (
    "probe_context0_block_input",
    "probe_context0_attention_input",
    "probe_context0_qkv",
    "probe_context0_q_after_rope",
    "probe_context0_k_after_rope",
    "probe_context0_v",
    "probe_context0_q_normalized",
    "probe_context0_k_normalized",
    "probe_context0_q_transposed",
    "probe_context0_k_transposed",
    "probe_context0_v_transposed",
    "probe_context0_scaled_logits_rows",
    "probe_context0_probabilities_rows",
    "probe_context0_weighted_value_rows",
    "probe_context0_pre_projection",
    "probe_context0_post_projection",
    "probe_context0_residual",
)
DIRECT_MAP = {
    "probe_context0_block_input": "block_input",
    "probe_context0_attention_input": "attention_input",
    "probe_context0_qkv": "qkv",
    "probe_context0_q_after_rope": "q_after_rope",
    "probe_context0_k_after_rope": "k_after_rope",
    "probe_context0_v": "v",
    "probe_context0_q_normalized": "q_normalized",
    "probe_context0_k_normalized": "k_normalized",
    "probe_context0_q_transposed": "q_transposed_rows",
    "probe_context0_k_transposed": "k_transposed",
    "probe_context0_v_transposed": "v_transposed",
    "probe_context0_weighted_value_rows": "weighted_value_transposed",
    "probe_context0_pre_projection": "pre_projection",
    "probe_context0_post_projection": "post_projection",
    "probe_context0_residual": "residual",
}
DERIVED_MAP = {
    "probe_context0_scaled_logits_rows": "scaled_logits",
    "probe_context0_probabilities_rows": "probabilities_f32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triposplat-repo", type=Path, required=True)
    parser.add_argument(
        "--source-commit", required=True,
        help="Required 40-hex official checkout pin; must also match graph metadata.",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--weights-sha256", required=True,
        help="Required 64-hex official checkpoint pin.",
    )
    parser.add_argument("--probe-onnx", type=Path, required=True)
    parser.add_argument("--trajectory-fixture-dir", type=Path, required=True)
    parser.add_argument("--invocation", action="append", type=int, dest="invocations")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--write-raw-tensors", action=argparse.BooleanOptionalAction, default=False,
        help="Write captured arrays as .npy below the ignored artifact directory.",
    )
    parser.add_argument(
        "--profile-ort", action=argparse.BooleanOptionalAction, default=True,
        help="Collect an ORT CPU profile in the artifact directory (default: enabled).",
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="mps"
    )
    parser.add_argument("--session-threads", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--low-memory-construction", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.invocations = args.invocations or [7, 8]
    if len(args.invocations) != 2 or len(set(args.invocations)) != 2:
        parser.error("exactly two distinct --invocation values are required")
    if any(value <= 0 for value in args.invocations):
        parser.error("--invocation values must be positive")
    if args.session_threads <= 0:
        parser.error("--session-threads must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    for value, length, label in (
        (args.source_commit, 40, "--source-commit"),
        (args.weights_sha256, 64, "--weights-sha256"),
    ):
        if len(value) != length or any(c not in "0123456789abcdefABCDEF" for c in value):
            parser.error(f"{label} must be exactly {length} hexadecimal characters")
    args.source_commit = args.source_commit.lower()
    args.weights_sha256 = args.weights_sha256.lower()
    return args


def array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def tensor_metadata(value: Any, origin: str, raw_path: str | None = None) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "bytes": int(array.nbytes),
        "sha256": array_sha256(array),
        "origin": origin,
    }
    if raw_path is not None:
        result["raw_npy"] = raw_path
    return result


def ensure_artifact_directory(path: Path, repository: Path) -> Path:
    artifact = path.expanduser().resolve()
    report_parent = artifact.parent
    report_parent.mkdir(parents=True, exist_ok=True)
    try:
        artifact.relative_to(repository)
    except ValueError:
        pass
    else:
        check = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", str(artifact)],
            check=False,
            capture_output=True,
        )
        if check.returncode != 0:
            raise ValueError(
                f"--artifact-dir is inside the repository but is not ignored: {artifact}"
            )
    artifact.mkdir(parents=True, exist_ok=True)
    return artifact


def write_raw_arrays(
    artifact: Path,
    invocation: int,
    source_name: str,
    values: Mapping[str, Any],
    enabled: bool,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    manifest: dict[str, dict[str, Any]] = {}
    target = artifact / f"invocation_{invocation:02d}" / source_name
    if enabled:
        target.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        relative = None
        if enabled:
            output = target / f"{name}.npy"
            np.save(output, np.ascontiguousarray(value), allow_pickle=False)
            relative = str(output.relative_to(artifact))
        if source_name == "onnxruntime_cpu":
            origin = "onnxruntime_cpu_graph_output"
        elif source_name == "derived_official_qk":
            origin = "derived_from_untouched_hooked_normalized_qk"
        else:
            origin = "module_hook"
            if name in {
                "q_transposed_rows", "k_transposed", "v_transposed",
                "weighted_value_rows", "weighted_value_transposed",
            }:
                origin = "lossless_view_of_hook_capture"
        manifest[name] = tensor_metadata(value, origin, relative)
    return manifest


def compact_metrics(
    reference: Any,
    candidate: Any,
    atol: float,
    rtol: float,
    chunk_elements: int = 1_048_576,
) -> dict[str, Any]:
    """comparison_metrics-equivalent core fields without full float64 copies."""
    import numpy as np

    ref_array = np.asarray(reference)
    got_array = np.asarray(candidate)
    if ref_array.shape != got_array.shape:
        raise ValueError(f"Comparison shape mismatch: {ref_array.shape} vs {got_array.shape}")
    ref = ref_array.reshape(-1)
    got = got_array.reshape(-1)
    finite_ref = bool(np.isfinite(ref).all())
    finite_got = bool(np.isfinite(got).all())
    if not finite_ref or not finite_got:
        return {
            "passed": False,
            "count": int(ref.size),
            "reference_finite": finite_ref,
            "candidate_finite": finite_got,
        }
    max_abs = -1.0
    max_rel = -1.0
    worst = 0
    absolute_sum = squared_sum = relative_sum = 0.0
    ref_sq = got_sq = dot = 0.0
    within_count = 0
    for start in range(0, ref.size, chunk_elements):
        end = min(start + chunk_elements, ref.size)
        r = np.asarray(ref[start:end], dtype=np.float64)
        g = np.asarray(got[start:end], dtype=np.float64)
        delta = g - r
        absolute = np.abs(delta)
        relative = absolute / np.maximum(np.abs(r), 1e-6)
        local = int(np.argmax(absolute))
        if float(absolute[local]) > max_abs:
            max_abs = float(absolute[local])
            worst = start + local
        max_rel = max(max_rel, float(relative.max()))
        absolute_sum += float(absolute.sum(dtype=np.float64))
        squared_sum += float(np.dot(delta, delta))
        relative_sum += float(relative.sum(dtype=np.float64))
        ref_sq += float(np.dot(r, r))
        got_sq += float(np.dot(g, g))
        dot += float(np.dot(r, g))
        within_count += int(np.count_nonzero(absolute <= atol + rtol * np.abs(r)))
    count = int(ref.size)
    denominator = math.sqrt(ref_sq * got_sq)
    return {
        "passed": within_count == count,
        "count": count,
        "reference_finite": True,
        "candidate_finite": True,
        "max_absolute_error": max_abs,
        "mean_absolute_error": absolute_sum / count,
        "rmse": math.sqrt(squared_sum / count),
        "max_relative_error_at_1e-6_floor": max_rel,
        "mean_relative_error_at_1e-6_floor": relative_sum / count,
        "cosine_similarity": dot / denominator if denominator else 1.0,
        "fraction_within_tolerance": within_count / count,
        "worst_index": [
            int(index) for index in np.unravel_index(worst, ref_array.shape)
        ],
    }


def detailed_metrics(reference: Any, candidate: Any, atol: float, rtol: float) -> dict[str, Any]:
    """Global, per-head, and (for four-row values) per-query-row metrics."""
    import numpy as np

    ref = np.asarray(reference)
    got = np.asarray(candidate)
    global_metrics = compact_metrics(ref, got, atol, rtol)
    result: dict[str, Any] = {"global": global_metrics}
    head_axis = None
    row_axis = None
    if ref.ndim == 4 and ref.shape[1] == HEADS:
        head_axis = 1
        if ref.shape[2] == ROWS:
            row_axis = 2
    elif ref.ndim >= 4 and ref.shape[-2:] == (HEADS, HEAD_DIM):
        head_axis = ref.ndim - 2
        if ref.shape[1] == ROWS:
            row_axis = 1
    if head_axis is not None:
        result["per_head"] = [
            compact_metrics(
                np.take(ref, index, axis=head_axis),
                np.take(got, index, axis=head_axis),
                atol,
                rtol,
            )
            for index in range(HEADS)
        ]
    if row_axis is not None:
        result["per_query_row"] = [
            compact_metrics(
                np.take(ref, index, axis=row_axis),
                np.take(got, index, axis=row_axis),
                atol,
                rtol,
            )
            for index in range(ROWS)
        ]
    return result


def boundary_metrics(reference: Any, candidate: Any, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np

    # Reuse the repository helper for bounded tensors; stream large activations.
    if np.asarray(reference).size <= 1_000_000:
        base = comparison_metrics(reference, candidate, atol, rtol)
        details = detailed_metrics(reference, candidate, atol, rtol)
        details["global"] = base
        return details
    return detailed_metrics(reference, candidate, atol, rtol)


class ContextAttentionCapture:
    """Capture semantic boundaries solely with public module hooks."""

    def __init__(self, model: Any) -> None:
        self.values: dict[str, Any] = {}
        self.handles: list[Any] = []
        block = model.context_refiner[0]
        attention = block.attn
        required = ("qkv", "q_norm", "k_norm", "out")
        missing = [name for name in required if not hasattr(attention, name)]
        if missing:
            raise AttributeError(f"context_refiner.0.attn lacks modules {missing}")

        def numpy(value: Any) -> Any:
            import numpy as np

            if isinstance(value, (tuple, list)):
                if len(value) != 1:
                    raise ValueError("Expected a single attention hook tensor")
                value = value[0]
            return np.ascontiguousarray(value.detach().cpu().numpy())

        def capture_input(name: str, first_rows: bool = False) -> Callable[..., None]:
            def hook(_module: Any, inputs: Any) -> None:
                if not inputs:
                    raise RuntimeError(f"{name} hook received no positional input")
                value = inputs[0]
                if first_rows:
                    value = value[:, :ROWS]
                self.values[name] = numpy(value)
            return hook

        def capture_output(name: str, first_rows: bool = False) -> Callable[..., None]:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output
                if first_rows:
                    value = value[:, :ROWS]
                self.values[name] = numpy(value)
            return hook

        self.handles.extend(
            [
                block.register_forward_pre_hook(capture_input("block_input", True)),
                block.norm1.register_forward_hook(capture_output("attention_input", True)),
                attention.qkv.register_forward_hook(capture_output("qkv_linear")),
                attention.q_norm.register_forward_pre_hook(capture_input("q_after_rope")),
                attention.k_norm.register_forward_pre_hook(capture_input("k_after_rope")),
                attention.q_norm.register_forward_hook(capture_output("q_normalized")),
                attention.k_norm.register_forward_hook(capture_output("k_normalized")),
                attention.out.register_forward_pre_hook(capture_input("pre_projection", True)),
                attention.out.register_forward_hook(capture_output("post_projection", True)),
                block.norm2.register_forward_pre_hook(capture_input("residual", True)),
            ]
        )

    def begin(self) -> None:
        self.values = {}

    def finish(self) -> dict[str, Any]:
        import numpy as np

        expected = {
            "block_input", "attention_input", "qkv_linear", "q_after_rope",
            "k_after_rope", "q_normalized", "k_normalized", "pre_projection",
            "post_projection", "residual",
        }
        if set(self.values) != expected:
            raise RuntimeError(
                f"Hook capture mismatch: missing={sorted(expected - set(self.values))}, "
                f"extra={sorted(set(self.values) - expected)}"
            )
        values = dict(self.values)
        qkv = values.pop("qkv_linear")
        if qkv.ndim == 3 and qkv.shape[-1] == 3 * MODEL_DIM:
            qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, HEADS, HEAD_DIM)
        if qkv.shape[2:] != (3, HEADS, HEAD_DIM):
            raise ValueError(f"Unexpected hooked QKV shape {qkv.shape}")
        values["qkv"] = np.ascontiguousarray(qkv)
        values["v"] = np.ascontiguousarray(qkv[:, :, 2])
        for name in ("q_after_rope", "k_after_rope", "q_normalized", "k_normalized"):
            if values[name].shape[-2:] != (HEADS, HEAD_DIM):
                raise ValueError(f"Unexpected {name} shape {values[name].shape}")
        values["q_transposed_rows"] = np.ascontiguousarray(
            values["q_normalized"][:, :ROWS].transpose(0, 2, 1, 3)
        )
        values["k_transposed"] = np.ascontiguousarray(
            values["k_normalized"].transpose(0, 2, 1, 3)
        )
        values["v_transposed"] = np.ascontiguousarray(values["v"].transpose(0, 2, 1, 3))
        pre = values["pre_projection"]
        if pre.ndim == 3 and pre.shape[-1] == MODEL_DIM:
            weighted = pre.reshape(pre.shape[0], pre.shape[1], HEADS, HEAD_DIM)
        elif pre.ndim == 4 and pre.shape[-2:] == (HEADS, HEAD_DIM):
            weighted = pre
            values["pre_projection"] = np.ascontiguousarray(pre.reshape(pre.shape[0], pre.shape[1], MODEL_DIM))
        else:
            raise ValueError(f"Unexpected pre-projection shape {pre.shape}")
        values["weighted_value_rows"] = np.ascontiguousarray(weighted)
        values["weighted_value_transposed"] = np.ascontiguousarray(
            weighted.transpose(0, 2, 1, 3)
        )
        return values

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def run_torch_capture(
    torch: Any,
    graph: Any,
    capture: ContextAttentionCapture,
    inputs: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], float]:
    tensors = torch_inputs_from_numpy(torch, inputs, device)
    capture.begin()
    synchronize_torch(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = graph(*tensors)
    synchronize_torch(torch, device)
    duration_ms = (time.perf_counter() - started) * 1000.0
    values = capture.finish()
    del tensors, outputs
    return values, duration_ms


def profile_sdpa_dispatch(
    torch: Any,
    captured: Mapping[str, Any],
    device: Any,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Record the exact PyTorch dispatch selected for captured Q/K/V rows."""
    if device.type != "mps":
        return {
            "available": False,
            "reason": f"Dispatch profiling is currently recorded only for MPS, got {device}.",
        }
    q = torch.from_numpy(captured["q_transposed_rows"]).to(device)
    k = torch.from_numpy(captured["k_transposed"]).to(device)
    v = torch.from_numpy(captured["v_transposed"]).to(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    synchronize_torch(torch, device)
    with torch.inference_mode(), torch.profiler.profile(activities=activities) as profiler:
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        synchronize_torch(torch, device)
    candidate = output.detach().float().cpu().numpy()
    expected = captured["weighted_value_transposed"]
    operators = sorted(
        event.key for event in profiler.key_averages()
        if "scaled_dot_product_attention" in event.key
    )
    return {
        "available": True,
        "profiled_shape": list(q.shape),
        "operators": operators,
        "output_vs_captured_full_call": compact_metrics(
            expected, candidate, atol, rtol
        ),
        "note": (
            "The four-row replay verifies dispatch and row independence; it does not "
            "expose the backend's internal tiling or materialization strategy."
        ),
    }


def derive_attention(q_normalized: Any, k_normalized: Any) -> dict[str, Any]:
    """Materialize four rows in f32; these are not untouched SDPA internals."""
    import numpy as np

    q = np.asarray(q_normalized, dtype=np.float32)[:, :ROWS].transpose(0, 2, 1, 3)
    k = np.asarray(k_normalized, dtype=np.float32).transpose(0, 2, 1, 3)
    logits = np.empty((q.shape[0], HEADS, ROWS, k.shape[2]), dtype=np.float32)
    scale = np.float32(HEAD_DIM ** -0.5)
    for head in range(HEADS):
        for row in range(ROWS):
            logits[:, head, row] = np.sum(
                q[:, head, row, None, :] * k[:, head], axis=-1, dtype=np.float32
            ) * scale
    row_max = np.max(logits, axis=-1, keepdims=True)
    shifted = np.subtract(logits, row_max, dtype=np.float32)
    exponentials = np.exp(shifted, dtype=np.float32)
    denominator = np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32)
    probabilities = np.divide(exponentials, denominator, dtype=np.float32)
    return {
        "scaled_logits": logits,
        "row_max": row_max,
        "shifted_logits": shifted,
        "exponentials": exponentials,
        "denominator": denominator,
        "probabilities_f32": probabilities,
    }


def row_logits(q_row: Any, keys: Any, dtype: Any) -> Any:
    import numpy as np

    q = np.asarray(q_row, dtype=dtype)
    k = np.asarray(keys, dtype=dtype)
    return np.sum(k * q[None, :], axis=-1, dtype=dtype) * dtype(HEAD_DIM ** -0.5)


def candidate_float64(q: Any, k: Any, v: Any) -> Any:
    import numpy as np

    output = np.empty((1, HEADS, ROWS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        keys = k[0, head]
        values = np.asarray(v[0, head], dtype=np.float64)
        for row in range(ROWS):
            logits = row_logits(q[0, head, row], keys, np.float64)
            exp = np.exp(logits - np.max(logits))
            probability = exp / np.sum(exp, dtype=np.float64)
            output[0, head, row] = np.sum(
                probability[:, None] * values, axis=0, dtype=np.float64
            ).astype(np.float32)
    return output


def f32_probabilities(q_row: Any, keys: Any) -> Any:
    import numpy as np

    logits = row_logits(q_row, keys, np.float32)
    shifted = np.subtract(logits, np.max(logits), dtype=np.float32)
    exp = np.exp(shifted, dtype=np.float32)
    return np.divide(exp, np.sum(exp, dtype=np.float32), dtype=np.float32)


def candidate_float32(q: Any, k: Any, v: Any) -> Any:
    import numpy as np

    output = np.empty((1, HEADS, ROWS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        for row in range(ROWS):
            probability = f32_probabilities(q[0, head, row], k[0, head])
            products = np.multiply(
                probability[:, None], v[0, head], dtype=np.float32
            )
            output[0, head, row] = np.sum(products, axis=0, dtype=np.float32)
    return output


def candidate_sequential(q: Any, k: Any, v: Any) -> Any:
    import numpy as np

    output = np.empty((1, HEADS, ROWS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        for row in range(ROWS):
            probability = f32_probabilities(q[0, head, row], k[0, head])
            accumulator = np.zeros(HEAD_DIM, dtype=np.float32)
            for index in range(probability.size):
                accumulator = np.add(
                    accumulator,
                    np.multiply(probability[index], v[0, head, index], dtype=np.float32),
                    dtype=np.float32,
                )
            output[0, head, row] = accumulator
    return output


def pairwise_reduce(products: Any) -> Any:
    import numpy as np

    current = np.asarray(products, dtype=np.float32)
    while current.shape[0] > 1:
        pairs = current.shape[0] // 2
        reduced = np.add(current[: 2 * pairs : 2], current[1 : 2 * pairs : 2], dtype=np.float32)
        if current.shape[0] % 2:
            reduced = np.concatenate((reduced, current[-1:]), axis=0)
        current = reduced
    return current[0]


def candidate_pairwise(q: Any, k: Any, v: Any) -> Any:
    import numpy as np

    output = np.empty((1, HEADS, ROWS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        for row in range(ROWS):
            probability = f32_probabilities(q[0, head, row], k[0, head])
            products = np.multiply(probability[:, None], v[0, head], dtype=np.float32)
            output[0, head, row] = pairwise_reduce(products)
    return output


def online_max_denominator(q_row: Any, keys: Any, block_size: int) -> tuple[Any, Any]:
    import numpy as np

    running_max = np.float32(-np.inf)
    running_denominator = np.float32(0.0)
    for start in range(0, keys.shape[0], block_size):
        logits = row_logits(q_row, keys[start : start + block_size], np.float32)
        block_max = np.max(logits)
        next_max = np.maximum(running_max, block_max).astype(np.float32)
        old_scale = np.exp(np.float32(running_max - next_max), dtype=np.float32)
        block_scale = np.exp(np.float32(block_max - next_max), dtype=np.float32)
        block_sum = np.sum(
            np.exp(np.subtract(logits, block_max, dtype=np.float32), dtype=np.float32),
            dtype=np.float32,
        )
        running_denominator = np.add(
            np.multiply(running_denominator, old_scale, dtype=np.float32),
            np.multiply(block_sum, block_scale, dtype=np.float32),
            dtype=np.float32,
        )
        running_max = next_max
    return running_max, running_denominator


def candidate_blockwise_materialized(q: Any, k: Any, v: Any, block_size: int) -> Any:
    import numpy as np

    output = np.empty((1, HEADS, ROWS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        keys = k[0, head]
        for row in range(ROWS):
            maximum, denominator = online_max_denominator(q[0, head, row], keys, block_size)
            probability = np.empty(keys.shape[0], dtype=np.float32)
            for start in range(0, keys.shape[0], block_size):
                end = min(start + block_size, keys.shape[0])
                logits = row_logits(q[0, head, row], keys[start:end], np.float32)
                probability[start:end] = np.divide(
                    np.exp(np.subtract(logits, maximum, dtype=np.float32), dtype=np.float32),
                    denominator,
                    dtype=np.float32,
                )
            products = np.multiply(probability[:, None], v[0, head], dtype=np.float32)
            output[0, head, row] = np.sum(products, axis=0, dtype=np.float32)
    return output


def candidate_blockwise_fused(q: Any, k: Any, v: Any, block_size: int) -> Any:
    import numpy as np

    output = np.empty((1, HEADS, ROWS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        keys = k[0, head]
        values = v[0, head]
        for row in range(ROWS):
            running_max = np.float32(-np.inf)
            denominator = np.float32(0.0)
            accumulator = np.zeros(HEAD_DIM, dtype=np.float32)
            for start in range(0, keys.shape[0], block_size):
                end = min(start + block_size, keys.shape[0])
                logits = row_logits(q[0, head, row], keys[start:end], np.float32)
                block_max = np.max(logits)
                next_max = np.maximum(running_max, block_max).astype(np.float32)
                old_scale = np.exp(np.float32(running_max - next_max), dtype=np.float32)
                weights = np.exp(
                    np.subtract(logits, next_max, dtype=np.float32), dtype=np.float32
                )
                denominator = np.add(
                    np.multiply(denominator, old_scale, dtype=np.float32),
                    np.sum(weights, dtype=np.float32),
                    dtype=np.float32,
                )
                block_products = np.multiply(
                    weights[:, None], values[start:end], dtype=np.float32
                )
                block_value = np.sum(block_products, axis=0, dtype=np.float32)
                accumulator = np.add(
                    np.multiply(accumulator, old_scale, dtype=np.float32),
                    block_value,
                    dtype=np.float32,
                )
                running_max = next_max
            output[0, head, row] = np.divide(accumulator, denominator, dtype=np.float32)
    return output


def candidate_specs(key_count: int) -> list[tuple[str, Callable[..., Any], int]]:
    f32 = 4
    f64 = 8
    product32 = key_count * HEAD_DIM * f32
    specs: list[tuple[str, Callable[..., Any], int]] = [
        ("float64_materialized_reference", candidate_float64, key_count * (HEAD_DIM + 2) * f64),
        ("float32_materialized_softmax", candidate_float32, product32 + key_count * 2 * f32),
        ("sequential_f32_probability_times_v", candidate_sequential, key_count * 2 * f32 + HEAD_DIM * f32),
        ("pairwise_tree_f32_reduction", candidate_pairwise, product32 * 2 + key_count * 2 * f32),
    ]
    for size in BLOCK_SIZES:
        specs.append(
            (
                f"blockwise_materialized_probabilities_{size}",
                lambda q, k, v, size=size: candidate_blockwise_materialized(q, k, v, size),
                product32 + key_count * f32 + size * (HEAD_DIM + 1) * f32,
            )
        )
        specs.append(
            (
                f"fused_blockwise_online_softmax_v_{size}",
                lambda q, k, v, size=size: candidate_blockwise_fused(q, k, v, size),
                size * (HEAD_DIM + 2) * f32 + 2 * HEAD_DIM * f32,
            )
        )
    return specs


def evaluate_algorithm_candidates(
    official: Mapping[str, Any], atol: float, rtol: float
) -> dict[str, dict[str, Any]]:
    import numpy as np

    q = np.asarray(official["q_transposed_rows"], dtype=np.float32)
    k = np.asarray(official["k_transposed"], dtype=np.float32)
    v = np.asarray(official["v_transposed"], dtype=np.float32)
    reference = np.asarray(official["weighted_value_rows"], dtype=np.float32).transpose(0, 2, 1, 3)
    repeated = np.broadcast_to(v[:, :, :1, :], v.shape)
    repeated_expected = np.broadcast_to(v[:, :, :1, :], reference.shape)
    results: dict[str, dict[str, Any]] = {}
    for name, function, peak_bytes in candidate_specs(k.shape[2]):
        started = time.perf_counter()
        output = function(q, k, v)
        duration_ms = (time.perf_counter() - started) * 1000.0
        repeat_started = time.perf_counter()
        repeated_output = function(q, k, repeated)
        repeat_ms = (time.perf_counter() - repeat_started) * 1000.0
        repeated_metrics = compact_metrics(repeated_expected, repeated_output, 0.0, 0.0)
        results[name] = {
            "runtime_ms": duration_ms,
            "repeated_v_runtime_ms": repeat_ms,
            "peak_temporary_bytes_estimate": int(peak_bytes),
            "metrics_vs_untouched_official": detailed_metrics(reference, output, atol, rtol),
            "output": output,
            "repeated_v_exact": bool(np.array_equal(repeated_expected, repeated_output)),
            "repeated_v_metrics": repeated_metrics,
        }
    return results


EXPECTED_SHAPES = {
    "probe_context0_block_input": (1, ROWS, MODEL_DIM),
    "probe_context0_attention_input": (1, ROWS, MODEL_DIM),
    "probe_context0_qkv": (1, CONDITION_TOKENS, 3, HEADS, HEAD_DIM),
    "probe_context0_q_after_rope": (1, CONDITION_TOKENS, HEADS, HEAD_DIM),
    "probe_context0_k_after_rope": (1, CONDITION_TOKENS, HEADS, HEAD_DIM),
    "probe_context0_v": (1, CONDITION_TOKENS, HEADS, HEAD_DIM),
    "probe_context0_q_normalized": (1, CONDITION_TOKENS, HEADS, HEAD_DIM),
    "probe_context0_k_normalized": (1, CONDITION_TOKENS, HEADS, HEAD_DIM),
    "probe_context0_q_transposed": (1, HEADS, ROWS, HEAD_DIM),
    "probe_context0_k_transposed": (1, HEADS, CONDITION_TOKENS, HEAD_DIM),
    "probe_context0_v_transposed": (1, HEADS, CONDITION_TOKENS, HEAD_DIM),
    "probe_context0_scaled_logits_rows": (1, HEADS, ROWS, CONDITION_TOKENS),
    "probe_context0_probabilities_rows": (1, HEADS, ROWS, CONDITION_TOKENS),
    "probe_context0_weighted_value_rows": (1, HEADS, ROWS, HEAD_DIM),
    "probe_context0_pre_projection": (1, ROWS, MODEL_DIM),
    "probe_context0_post_projection": (1, ROWS, MODEL_DIM),
    "probe_context0_residual": (1, ROWS, MODEL_DIM),
}


def validate_graph_contract(onnx: Any, graph_path: Path) -> tuple[Any, dict[str, str], list[Path]]:
    model = onnx.load_model(str(graph_path), load_external_data=False)
    outputs = tuple(value.name for value in model.graph.output)
    if outputs != PROBES:
        raise ValueError(
            "Graph is not the ordered make_dit_probe_fixture.py --probe-set context0 "
            f"contract: {outputs}"
        )
    count = int(metadata_dict(model).get("triposplat.diagnostic_probes", "0"))
    if count != len(PROBES):
        raise ValueError(f"Graph records {count} probes; expected {len(PROBES)}")
    for value in model.graph.output:
        shape = tuple(int(item.dim_value) for item in value.type.tensor_type.shape.dim)
        if shape != EXPECTED_SHAPES[value.name]:
            raise ValueError(
                f"Probe {value.name} shape is {shape}; expected {EXPECTED_SHAPES[value.name]}"
            )
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Probe {value.name} must be float32")
    metadata = metadata_dict(model)
    if metadata.get("triposplat.source_repository") != OFFICIAL_REPOSITORY_URL:
        raise ValueError("Graph is not pinned to the official TripoSplat repository URL")
    if metadata.get("triposplat.source_loader") != "official triposplat.load_flow_model":
        raise ValueError("Graph was not exported through the official loader")
    if metadata.get("triposplat.source_files_dirty") != "false":
        raise ValueError("Graph metadata does not attest to clean official source files")
    external = verify_external_data_files(graph_path, model)
    if not external:
        raise ValueError("Diagnostic graph has no external weight sidecar")
    return model, metadata, external


def parse_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"available": False, "reason": "profiling disabled or profile unavailable"}
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
        operators = sorted(
            {
                str(event.get("args", {}).get("op_name"))
                for event in events
                if isinstance(event, dict)
                and isinstance(event.get("args"), dict)
                and event["args"].get("op_name")
            }
        )
        attention = sorted(
            {
                str(event.get("name"))
                for event in events
                if isinstance(event, dict)
                and any(
                    marker in str(event.get("name", ""))
                    for marker in ("context_refiner.0/attn", "triposplat_probe")
                )
            }
        )
        return {
            "available": True,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "operator_names": operators,
            "context_attention_profile_event_names": attention[:128],
            "context_attention_event_names_truncated": len(attention) > 128,
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"available": False, "reason": f"profile parse failed: {exc}"}


def compare_direct_boundaries(
    official: Mapping[str, Any],
    adapted: Mapping[str, Any],
    ort_values: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for probe, local in DIRECT_MAP.items():
        official_adapted = boundary_metrics(official[local], adapted[local], atol, rtol)
        official_ort = boundary_metrics(official[local], ort_values[probe], atol, rtol)
        adapted_ort = boundary_metrics(adapted[local], ort_values[probe], atol, rtol)
        comparisons[probe] = {
            "capture_boundary": local,
            "directly_corresponding": True,
            "official_capture_origin": (
                "lossless_view_of_hook_capture"
                if local in {
                    "q_transposed_rows", "k_transposed", "v_transposed",
                    "weighted_value_rows", "weighted_value_transposed",
                }
                else "module_hook"
            ),
            "adapted_vs_untouched_official": official_adapted,
            "ort_cpu_vs_untouched_official": official_ort,
            "ort_cpu_vs_adapted": adapted_ort,
        }
        ort_rmse = float(official_ort["global"]["rmse"])
        adapted_rmse = float(official_adapted["global"]["rmse"])
        ranking.append(
            {
                "boundary": probe,
                "ort_cpu_rmse": ort_rmse,
                "adapted_pytorch_rmse": adapted_rmse,
                "runtime_attributable_rmse_excess": ort_rmse - adapted_rmse,
                "ort_cpu_max_absolute_error": float(
                    official_ort["global"]["max_absolute_error"]
                ),
            }
        )
    ranking.sort(
        key=lambda item: (item["runtime_attributable_rmse_excess"], item["ort_cpu_rmse"]),
        reverse=True,
    )
    return comparisons, ranking


def summarized_candidate(
    output: Any,
    reference: Any,
    runtime_ms: float,
    peak_bytes: int | None,
    atol: float,
    rtol: float,
    role: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "runtime_ms": runtime_ms,
        "peak_temporary_bytes_estimate": peak_bytes,
        "peak_temporary_bytes_estimate_note": (
            None if peak_bytes is not None else
            "Unavailable for captured full-model backend execution."
        ),
        "metrics_vs_untouched_official": detailed_metrics(reference, output, atol, rtol),
        "tensor": tensor_metadata(output, role),
        "repeated_v_exact": None,
        "repeated_v_metrics": None,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch, ONNX, ONNX Runtime, and NumPy are required: {exc}") from exc

    project_root = Path(__file__).resolve().parents[2]
    artifact = ensure_artifact_directory(args.artifact_dir, project_root)
    graph_path = args.probe_onnx.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    fixture_dir = args.trajectory_fixture_dir.expanduser().resolve()
    repo = args.triposplat_repo.expanduser().resolve()
    for path, label in ((graph_path, "probe graph"), (weights, "weights")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    manifest_path = fixture_dir / "flow.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"trajectory manifest does not exist: {manifest_path}")

    graph_proto, metadata, sidecars = validate_graph_contract(onnx, graph_path)
    config = adapter_configuration(metadata)
    if config["collapsed_unconditional_context"]:
        raise ValueError("context0 parity requires canonical, non-collapsed conditioning")
    commit, dirty = source_revision(repo)
    if commit != args.source_commit:
        raise RuntimeError(f"Official checkout is {commit}; required pin is {args.source_commit}")
    if dirty is not False:
        raise RuntimeError("Official model.py/triposplat.py must be verifiably clean")
    graph_commit = metadata.get("triposplat.source_commit")
    if graph_commit != args.source_commit:
        raise RuntimeError(
            f"Probe graph source commit {graph_commit!r} differs from required pin "
            f"{args.source_commit}"
        )
    actual_weights_hash = sha256_file(weights)
    if actual_weights_hash.lower() != args.weights_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 {actual_weights_hash} differs from required pin "
            f"{args.weights_sha256}"
        )

    records: dict[int, dict[str, Any]] = {}
    loaded_inputs: dict[int, Mapping[str, Any]] = {}
    for invocation in args.invocations:
        inputs, _expected, source_text = load_trajectory_invocation_fixture(
            fixture_dir, invocation
        )
        loaded_inputs[invocation] = inputs
        pass_name = source_text.rsplit(" pass=", 1)[-1]
        records[invocation] = {
            "source": source_text,
            "pass": pass_name,
            "inputs": {
                name: tensor_metadata(value, "exact_recorded_trajectory_input")
                for name, value in inputs.items()
            },
            "timings_ms": {},
        }
    first, second = (loaded_inputs[value] for value in args.invocations)
    paired = {
        name: {
            "exactly_equal": bool(np.array_equal(first[name], second[name])),
            "sha256": array_sha256(first[name]),
        }
        for name in ("latent", "camera", "t")
    }
    if not all(value["exactly_equal"] for value in paired.values()):
        raise ValueError("The two trajectory calls must share exact latent, camera, and t")

    device = choose_torch_device(torch, args.device)
    print(f"Loading pinned official {config['precision']} model on {device}", flush=True)
    model, source = load_official_flow_model(
        torch=torch,
        triposplat_repo=repo,
        weights=weights,
        device=device,
        internal_precision=config["precision"],
        low_memory_construction=args.low_memory_construction,
    )
    official_graph = make_browser_flow_step(torch, model, config["precision"])
    official_capture = ContextAttentionCapture(model)
    official_values: dict[int, dict[str, Any]] = {}
    for invocation in args.invocations:
        print(f"Untouched official PyTorch invocation {invocation}", flush=True)
        values, duration = run_torch_capture(
            torch, official_graph, official_capture, loaded_inputs[invocation], device
        )
        official_values[invocation] = values
        records[invocation]["timings_ms"]["untouched_official_pytorch"] = duration
        records[invocation]["tensor_manifests"] = {
            "untouched_official": write_raw_arrays(
                artifact, invocation, "untouched_official", values, args.write_raw_tensors
            )
        }
    official_capture.close()
    backend_probe_invocation = next(
        invocation
        for invocation in args.invocations
        if records[invocation]["pass"] == "unconditional"
    )
    untouched_sdpa_dispatch = profile_sdpa_dispatch(
        torch,
        official_values[backend_probe_invocation],
        device,
        args.atol,
        args.rtol,
    )

    adapter = adapt_official_flow_for_onnx(
        torch,
        model,
        source.model_module,
        attention_query_chunk=config["attention_query_chunk"],
        collapsed_unconditional_context=config["collapsed_unconditional_context"],
        attention_head_chunk=config["attention_head_chunk"],
        attention_head_padding=config["attention_head_padding"],
        qk_norm_padding_tokens=config["qk_norm_padding_tokens"],
        rms_norm_eps=config["rms_norm_eps"],
        attention_output_chunk=config["attention_output_chunk"],
        attention_output_reduction_chunk=config["attention_output_reduction_chunk"],
    )
    adapted_graph = make_browser_flow_step(torch, model, config["precision"])
    adapted_capture = ContextAttentionCapture(model)
    adapted_values: dict[int, dict[str, Any]] = {}
    for invocation in args.invocations:
        print(f"Adapted PyTorch invocation {invocation}", flush=True)
        values, duration = run_torch_capture(
            torch, adapted_graph, adapted_capture, loaded_inputs[invocation], device
        )
        adapted_values[invocation] = values
        records[invocation]["timings_ms"]["adapted_pytorch"] = duration
        records[invocation]["tensor_manifests"]["adapted_pytorch"] = write_raw_arrays(
            artifact, invocation, "adapted_pytorch", values, args.write_raw_tensors
        )
    adapted_capture.close()
    del official_graph, adapted_graph, model, source
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.intra_op_num_threads = args.session_threads
    if args.profile_ort:
        options.enable_profiling = True
        options.profile_file_prefix = str(artifact / "ort_context0_profile")
    print("Loading context0 graph in ONNX Runtime CPU (ORT_DISABLE_ALL)", flush=True)
    load_started = time.perf_counter()
    session = ort.InferenceSession(
        str(graph_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    session_load_ms = (time.perf_counter() - load_started) * 1000.0
    observed_outputs = tuple(output.name for output in session.get_outputs())
    if observed_outputs != PROBES:
        raise ValueError(f"ORT output order differs from graph: {observed_outputs}")

    for invocation in args.invocations:
        print(f"ORT CPU and numerical candidates invocation {invocation}", flush=True)
        started = time.perf_counter()
        output_list = session.run(list(PROBES), dict(loaded_inputs[invocation]))
        ort_duration = (time.perf_counter() - started) * 1000.0
        ort_values = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in zip(PROBES, output_list)
        }
        records[invocation]["timings_ms"]["onnxruntime_cpu"] = ort_duration
        records[invocation]["tensor_manifests"]["onnxruntime_cpu"] = write_raw_arrays(
            artifact, invocation, "onnxruntime_cpu", ort_values, args.write_raw_tensors
        )

        derived_started = time.perf_counter()
        derived = derive_attention(
            official_values[invocation]["q_normalized"],
            official_values[invocation]["k_normalized"],
        )
        records[invocation]["timings_ms"]["derived_official_qk_materialization"] = (
            time.perf_counter() - derived_started
        ) * 1000.0
        records[invocation]["derived_softmax_internals"] = {
            "status": "derived_not_captured",
            "reason": (
                "Untouched MPS/CPU/CUDA functional SDPA exposes no logits, row maxima, "
                "exponentials, denominator, or probabilities to module hooks. These arrays "
                "are deterministically derived from exact untouched hooked normalized Q/K."
            ),
            "tensor_manifests": write_raw_arrays(
                artifact, invocation, "derived_official_qk", derived, args.write_raw_tensors
            ),
            "ort_comparisons": {
                probe: boundary_metrics(derived[local], ort_values[probe], args.atol, args.rtol)
                for probe, local in DERIVED_MAP.items()
            },
        }

        direct, attribution = compare_direct_boundaries(
            official_values[invocation], adapted_values[invocation], ort_values,
            args.atol, args.rtol,
        )
        records[invocation]["direct_boundary_comparisons"] = direct
        records[invocation]["ranked_boundary_attribution"] = attribution

        reference = np.asarray(
            official_values[invocation]["weighted_value_rows"], dtype=np.float32
        ).transpose(0, 2, 1, 3)
        adapted_output = np.asarray(
            adapted_values[invocation]["weighted_value_rows"], dtype=np.float32
        ).transpose(0, 2, 1, 3)
        ort_output = ort_values["probe_context0_weighted_value_rows"]
        candidates: dict[str, dict[str, Any]] = {
            "untouched_official_captured_output": summarized_candidate(
                reference, reference,
                records[invocation]["timings_ms"]["untouched_official_pytorch"],
                None, args.atol, args.rtol, "oracle_captured_model_output",
            ),
            "adapted_pytorch_captured_output": summarized_candidate(
                adapted_output, reference,
                records[invocation]["timings_ms"]["adapted_pytorch"],
                None, args.atol, args.rtol, "diagnostic_captured_model_output",
            ),
            "ort_cpu_captured_output": summarized_candidate(
                ort_output, reference, ort_duration, None,
                args.atol, args.rtol, "measured_baseline_captured_graph_output",
            ),
        }
        algorithm_results = evaluate_algorithm_candidates(
            official_values[invocation], args.atol, args.rtol
        )
        for name, item in algorithm_results.items():
            output = item.pop("output")
            item["role"] = "eligible_numerical_attention_candidate"
            item["tensor"] = tensor_metadata(output, "materialized_from_untouched_official_qkv")
            candidates[name] = item
        records[invocation]["candidates"] = candidates
        del output_list, ort_values, derived, algorithm_results
        gc.collect()

    profile_path = Path(session.end_profiling()).resolve() if args.profile_ort else None
    del session

    eligible_names = [name for name, _, _ in candidate_specs(CONDITION_TOKENS)]
    candidate_ranking: list[dict[str, Any]] = []
    baseline_by_invocation = {
        invocation: float(
            records[invocation]["candidates"]["ort_cpu_captured_output"]
            ["metrics_vs_untouched_official"]["global"]["rmse"]
        )
        for invocation in args.invocations
    }
    for name in eligible_names:
        errors = {
            invocation: float(
                records[invocation]["candidates"][name]
                ["metrics_vs_untouched_official"]["global"]["rmse"]
            )
            for invocation in args.invocations
        }
        improvements = {
            invocation: baseline_by_invocation[invocation] - errors[invocation]
            for invocation in args.invocations
        }
        candidate_ranking.append(
            {
                "candidate": name,
                "mean_rmse": float(np.mean(list(errors.values()))),
                "rmse_by_invocation": {str(k): v for k, v in errors.items()},
                "improvement_over_ort_rmse_by_invocation": {
                    str(k): v for k, v in improvements.items()
                },
                "strictly_improves_every_invocation": all(value > 0.0 for value in improvements.values()),
            }
        )
    candidate_ranking.sort(key=lambda item: (item["mean_rmse"], item["candidate"]))
    improving = [item for item in candidate_ranking if item["strictly_improves_every_invocation"]]
    decision = {
        "decision": "go" if improving else "no-go",
        "criterion": (
            "GO only when an eligible real-tensor numerical candidate has strictly lower "
            "weighted-value RMSE than measured ORT CPU against untouched official output "
            "on each requested invocation; no tolerance or derived oracle decides this gate."
        ),
        "baseline": "ort_cpu_captured_output",
        "eligible_candidates": eligible_names,
        "excluded_from_decision": {
            "untouched_official_captured_output": "truth/oracle",
            "adapted_pytorch_captured_output": "diagnostic backend, not a candidate kernel",
            "float64_note": "float64 is eligible because it is a measured materialized candidate",
        },
        "best_improving_candidate": improving[0] if improving else None,
    }

    return {
        "passed": decision["decision"] == "go",
        "component": "context_refiner_0_attention_parity",
        "scope": "Exact first-context-refiner attention boundaries and four query rows.",
        "go_no_go": decision,
        "candidate_ranking": candidate_ranking,
        "source": {
            "repository": OFFICIAL_REPOSITORY_URL,
            "checkout": str(repo),
            "required_commit": args.source_commit,
            "observed_commit": commit,
            "tracked_source_dirty": dirty,
            "weights": str(weights),
            "required_weights_sha256": args.weights_sha256,
            "observed_weights_sha256": actual_weights_hash,
        },
        "graph": {
            "path": str(graph_path),
            "sha256": sha256_file(graph_path),
            "bytes": graph_path.stat().st_size,
            "metadata": metadata,
            "external_data": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sidecars
            ],
            "probe_count": len(PROBES),
            "probe_shapes": {name: list(shape) for name, shape in EXPECTED_SHAPES.items()},
        },
        "fixture": {
            "directory": str(fixture_dir),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "invocations": args.invocations,
            "paired_dynamic_inputs": paired,
        },
        "artifacts": {
            "directory": str(artifact),
            "raw_tensors_written": args.write_raw_tensors,
            "ort_profile": parse_profile(profile_path),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_device": str(device),
            "torch_mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "onnxruntime_available_providers": ort.get_available_providers(),
        },
        "backend_facts": {
            "untouched_attention": "torch functional scaled_dot_product_attention",
            "untouched_sdpa_dispatch": untouched_sdpa_dispatch,
            "device": str(device),
            "dtype": config["precision"],
            "mask": None,
            "dropout": 0.0,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "probabilities_materialized": "not observable through the public MPS SDPA API",
            "accumulator_precision": "not exposed by the public MPS SDPA API",
            "verified_vs_inferred": {
                "verified": [
                    "operator dispatch", "device", "dtype", "layout", "mask", "dropout"
                ],
                "inference": [
                    "internal tiling", "reduction tree", "probability materialization",
                    "accumulator precision"
                ],
            },
            "onnxruntime_provider": "CPUExecutionProvider",
            "onnxruntime_graph_optimization": "ORT_DISABLE_ALL",
            "onnxruntime_session_threads": args.session_threads,
            "onnxruntime_session_load_ms": session_load_ms,
            "profiling_enabled": args.profile_ort,
            "candidate_timings_are_diagnostic_not_benchmarks": True,
        },
        "adapter": adapter.__dict__,
        "tolerance": {"absolute": args.atol, "relative": args.rtol},
        "invocations": {str(key): value for key, value in records.items()},
        "interpretation": {
            "derived_arrays_are_untouched_sdpa_captures": False,
            "report_contains_full_tensor_arrays": False,
            "raw_arrays_are_optional_and_confined_to_artifact_directory": True,
            "candidate_peak_bytes_are_algorithmic_temporary_estimates_excluding_inputs_and_output": True,
        },
    }


def main() -> None:
    args = parse_args()
    report = validate(args)
    destination = args.report.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "report": str(destination),
        "decision": report["go_no_go"]["decision"],
        "best_improving_candidate": (
            None
            if report["go_no_go"]["best_improving_candidate"] is None
            else report["go_no_go"]["best_improving_candidate"]["candidate"]
        ),
        "top_attribution_by_invocation": {
            invocation: value["ranked_boundary_attribution"][0]
            for invocation, value in report["invocations"].items()
        },
        "raw_tensors_written": report["artifacts"]["raw_tensors_written"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
