#!/usr/bin/env python3
"""Validate DINOv3 ONNX output against official TripoSplat PyTorch.

The validator enforces the fixed FP32 browser I/O contract before inference, runs
the official ``DinoV3ViT`` at the graph's recorded internal precision, includes
the official extra FP32 non-affine layer norm on the reference side, reports
parity metrics, and exits nonzero when the requested tolerance is exceeded.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from dinov3_common import (
    DINOV3_MEAN,
    DINOV3_STD,
    INPUT_NAME,
    INPUT_SHAPE,
    INTERNAL_PRECISION_METADATA_KEY,
    OFFICIAL_REPOSITORY_URL,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    array_summary,
    checkpoint_dtype_counts,
    choose_torch_device,
    comparison_metrics,
    deterministic_pixel_values,
    load_input_fixture,
    load_official_encoder,
    make_browser_encoder,
    release_torch_cache,
    synchronize_torch,
    validate_pixel_values,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--triposplat-repo",
        type=Path,
        required=True,
        help=f"Local clone of {OFFICIAL_REPOSITORY_URL} (must contain model.py).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Local dino_v3_vit_h.safetensors used by the ONNX export.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Exported fixed-shape DINOv3 ONNX graph.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help=(
            "Optional NPZ with normalized pixel_values, or shared image_rgb [0,1]. "
            "A feature1 array is also checked when present."
        ),
    )
    parser.add_argument(
        "--internal-precision",
        "--precision",
        dest="internal_precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help=(
            "Expected internal graph precision. auto requires exporter metadata "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
        help="Device for the official PyTorch reference (default: %(default)s).",
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help=(
            "Python ONNX Runtime provider; repeat for fallback order "
            "(default: CPUExecutionProvider)."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        help="Absolute allclose tolerance (default: 1e-4 fp32, 3e-2 fp16).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        help="Relative allclose tolerance (default: 1e-3 fp32, 3e-2 fp16).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON metrics report destination.",
    )
    parser.add_argument(
        "--save-fixture",
        type=Path,
        help=(
            "Optional NPZ destination containing normalized pixel_values and the "
            "official FP32 feature1 reference for browser parity."
        ),
    )
    parser.add_argument(
        "--session-threads",
        type=int,
        default=0,
        help="ORT intra-op thread count; zero leaves the runtime default (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.atol is not None and args.atol < 0:
        parser.error("--atol must be non-negative")
    if args.rtol is not None and args.rtol < 0:
        parser.error("--rtol must be non-negative")
    if args.session_threads < 0:
        parser.error("--session-threads must be non-negative")
    return args


def make_session(ort: Any, graph_path: Path, providers: list[str], threads: int) -> Any:
    available = set(ort.get_available_providers())
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise RuntimeError(
            f"Requested ORT providers are unavailable: {missing}. Available: {sorted(available)}"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(graph_path), sess_options=options, providers=providers)


def require_fixed_contract(session: Any) -> str:
    """Validate public names/dtypes/shapes and return recorded internal precision."""

    inputs = {value.name: value for value in session.get_inputs()}
    outputs = {value.name: value for value in session.get_outputs()}
    if set(inputs) != {INPUT_NAME}:
        raise ValueError(f"ONNX inputs are {sorted(inputs)}; expected exactly {INPUT_NAME}")
    if set(outputs) != {OUTPUT_NAME}:
        raise ValueError(f"ONNX outputs are {sorted(outputs)}; expected exactly {OUTPUT_NAME}")
    if tuple(inputs[INPUT_NAME].shape) != INPUT_SHAPE:
        raise ValueError(
            f"ONNX {INPUT_NAME} shape is {inputs[INPUT_NAME].shape}; expected {INPUT_SHAPE}"
        )
    if tuple(outputs[OUTPUT_NAME].shape) != OUTPUT_SHAPE:
        raise ValueError(
            f"ONNX {OUTPUT_NAME} shape is {outputs[OUTPUT_NAME].shape}; expected {OUTPUT_SHAPE}"
        )
    if inputs[INPUT_NAME].type != "tensor(float)":
        raise ValueError(
            f"ONNX {INPUT_NAME} type is {inputs[INPUT_NAME].type}; expected tensor(float)"
        )
    if outputs[OUTPUT_NAME].type != "tensor(float)":
        raise ValueError(
            f"ONNX {OUTPUT_NAME} type is {outputs[OUTPUT_NAME].type}; expected tensor(float)"
        )

    metadata = session.get_modelmeta().custom_metadata_map
    component = metadata.get("triposplat.component")
    if component != "dinov3_encoder":
        raise ValueError(
            "ONNX is missing strict TripoSplat DINOv3 component metadata; "
            f"observed {component!r}"
        )
    precision = metadata.get(INTERNAL_PRECISION_METADATA_KEY)
    if precision not in {"fp16", "fp32"}:
        raise ValueError(
            f"ONNX metadata {INTERNAL_PRECISION_METADATA_KEY!r} is {precision!r}; "
            "expected fp16 or fp32"
        )
    post_norm = metadata.get("triposplat.post_normalization", "")
    if "FP32 non-affine layer_norm" not in post_norm:
        raise ValueError("ONNX metadata does not attest the required FP32 post-layer-normalization")
    return precision


def validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing validation dependency. Install a PyTorch-supported Python version "
            "and run `python -m pip install -r scripts/triposplat/requirements.txt`. "
            f"Original error: {exc}"
        ) from exc

    graph_path = args.onnx.expanduser().resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f"ONNX graph does not exist: {graph_path}")
    providers = args.providers or ["CPUExecutionProvider"]
    session = make_session(ort, graph_path, providers, args.session_threads)
    graph_precision = require_fixed_contract(session)
    if args.internal_precision != "auto" and args.internal_precision != graph_precision:
        raise ValueError(
            f"--internal-precision {args.internal_precision} conflicts with ONNX "
            f"metadata ({graph_precision})"
        )
    internal_precision = graph_precision
    atol = args.atol if args.atol is not None else (1e-4 if internal_precision == "fp32" else 3e-2)
    rtol = args.rtol if args.rtol is not None else (1e-3 if internal_precision == "fp32" else 3e-2)

    if args.fixture:
        pixel_values, fixture_expected, fixture_key = load_input_fixture(args.fixture)
        input_source = f"{args.fixture.expanduser().resolve()}:{fixture_key}"
    else:
        pixel_values = deterministic_pixel_values()
        fixture_expected = None
        input_source = "deterministic synthetic normalized image"
    validate_pixel_values(pixel_values)
    pixel_values = np.ascontiguousarray(pixel_values, dtype=np.float32)

    device = choose_torch_device(torch, args.device)
    source_dtypes = checkpoint_dtype_counts(args.weights)
    print(f"Loading official PyTorch DINOv3 ({internal_precision} internals) on {device}")
    encoder = load_official_encoder(
        torch=torch,
        triposplat_repo=args.triposplat_repo,
        weights=args.weights,
        device=device,
        internal_precision=internal_precision,
    )
    reference_model = make_browser_encoder(torch, encoder, internal_precision).eval().to(device)
    torch_input = torch.from_numpy(pixel_values).to(device=device, dtype=torch.float32)

    synchronize_torch(torch, device)
    torch_started = time.perf_counter()
    with torch.inference_mode():
        reference_tensor = reference_model(torch_input)
    synchronize_torch(torch, device)
    torch_duration_ms = (time.perf_counter() - torch_started) * 1000.0
    reference = reference_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if tuple(reference.shape) != OUTPUT_SHAPE:
        raise ValueError(f"PyTorch output shape is {reference.shape}; expected {OUTPUT_SHAPE}")
    if not np.isfinite(reference).all():
        raise ValueError("PyTorch reference contains NaN or infinity")

    # Release the second copy of the very large weights before ORT allocates its
    # attention temporaries.  The already-created ORT session retains only its copy.
    del reference_tensor, torch_input, reference_model, encoder
    gc.collect()
    release_torch_cache(torch, device)

    ort_started = time.perf_counter()
    candidate = session.run([OUTPUT_NAME], {INPUT_NAME: pixel_values})[0]
    ort_duration_ms = (time.perf_counter() - ort_started) * 1000.0
    candidate = np.asarray(candidate, dtype=np.float32)
    if tuple(candidate.shape) != OUTPUT_SHAPE:
        raise ValueError(f"ORT output shape is {candidate.shape}; expected {OUTPUT_SHAPE}")

    metrics = comparison_metrics(reference, candidate, atol=atol, rtol=rtol)
    fixture_metrics = None
    overall_passed = bool(metrics["passed"])
    if fixture_expected is not None:
        fixture_metrics = comparison_metrics(
            fixture_expected,
            reference,
            atol=atol,
            rtol=rtol,
        )
        overall_passed = bool(overall_passed and fixture_metrics["passed"])

    report: dict[str, Any] = {
        "passed": overall_passed,
        "contract": {
            INPUT_NAME: {
                "shape": list(INPUT_SHAPE),
                "precision": "fp32",
                "normalization": {"mean": list(DINOV3_MEAN), "std": list(DINOV3_STD)},
            },
            OUTPUT_NAME: {"shape": list(OUTPUT_SHAPE), "precision": "fp32"},
            "internal_precision": internal_precision,
            "post_normalization": "FP32 non-affine layer_norm",
        },
        "checkpoint_tensor_dtypes": source_dtypes,
        "inputs": {"source": input_source, INPUT_NAME: array_summary(pixel_values)},
        "comparison": metrics,
        "fixture_reference_comparison": fixture_metrics,
        "outputs": {
            "pytorch": array_summary(reference),
            "onnxruntime": array_summary(candidate),
        },
        "tolerance": {"atol": atol, "rtol": rtol},
        "measured_single_run_ms": {
            "pytorch": torch_duration_ms,
            "onnxruntime_python": ort_duration_ms,
            "note": "Measured validation runs; these are not Chrome/Edge WebGPU benchmarks.",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "torch_device": str(device),
            "ort_providers": session.get_providers(),
            "onnx": str(graph_path),
        },
    }

    if args.save_fixture:
        fixture_path = args.save_fixture.expanduser().resolve()
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fixture_path,
            pixel_values=pixel_values,
            feature1=reference,
            metadata=np.asarray(
                json.dumps(
                    {
                        "internal_precision": internal_precision,
                        "source": "official TripoSplat DinoV3ViT PyTorch plus FP32 post-norm",
                        "input_normalization": {
                            "mean": list(DINOV3_MEAN),
                            "std": list(DINOV3_STD),
                        },
                    },
                    sort_keys=True,
                )
            ),
        )
        report["saved_fixture"] = str(fixture_path)
        print(f"Wrote browser parity fixture {fixture_path}")
    if args.report:
        write_json(args.report, report)
        print(f"Wrote report {args.report.expanduser().resolve()}")

    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    print(
        "PASS: official PyTorch and Python ONNX Runtime agree"
        if report["passed"]
        else "FAIL: output is outside the requested tolerances"
    )
    return report


def main() -> None:
    report = validate(parse_args())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
