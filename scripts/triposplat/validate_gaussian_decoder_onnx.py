#!/usr/bin/env python3
"""Validate Gaussian decoder ONNX features against official TripoSplat PyTorch.

The command feeds identical float32 ``points`` and ``cond`` tensors to upstream
``decoder.gs`` and Python ONNX Runtime, reports detailed errors, and exits nonzero
unless every raw feature value satisfies the configured allclose gate.  Gaussian
offset construction, representation scaling, biases, and activations are deliberately
outside this validation boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decoder_onnx_common import (
    COND_SHAPE,
    FEATURES_SHAPE,
    OFFICIAL_REPOSITORY_URL,
    POINTS_SHAPE,
    array_summary,
    choose_torch_device,
    comparison_metrics,
    default_tolerances,
    deterministic_gaussian_inputs,
    load_gaussian_fixture,
    load_graph_contract,
    run_pytorch_reference,
    validate_array,
    validate_gaussian_inputs,
    validate_with_ort,
    validation_runtime_report,
    write_json,
)


COMPONENT = "gaussian_decoder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--triposplat-repo",
        type=Path,
        required=True,
        help=f"Local clone of {OFFICIAL_REPOSITORY_URL}.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Official decoder checkpoint used for the ONNX export.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Exported fixed-shape Gaussian decoder ONNX graph.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Optional NPZ with points, cond and optionally features.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260714,
        help="Synthetic input seed when --fixture is omitted (default: %(default)s).",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help="Expected internal graph precision; auto reads metadata (default: %(default)s).",
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
            "Python ONNX Runtime provider. Repeat for fallback order "
            "(default: CPUExecutionProvider)."
        ),
    )
    parser.add_argument("--atol", type=float, help="Absolute allclose tolerance.")
    parser.add_argument("--rtol", type=float, help="Relative allclose tolerance.")
    parser.add_argument("--report", type=Path, help="Optional JSON report destination.")
    parser.add_argument(
        "--save-fixture",
        type=Path,
        help="Optional NPZ destination containing inputs and official features.",
    )
    parser.add_argument(
        "--session-threads",
        type=int,
        default=0,
        help="ORT intra-op threads; zero leaves the runtime default (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.atol is not None and args.atol < 0:
        parser.error("--atol must be non-negative")
    if args.rtol is not None and args.rtol < 0:
        parser.error("--rtol must be non-negative")
    if args.session_threads < 0:
        parser.error("--session-threads must be non-negative")
    return args


def validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing validation dependency. Install a PyTorch-supported Python "
            "version and run `python -m pip install -r "
            "scripts/triposplat/requirements.txt`. "
            f"Original error: {exc}"
        ) from exc

    graph_path = args.onnx.expanduser().resolve()
    metadata, artifacts, precision = load_graph_contract(
        onnx,
        graph_path,
        COMPONENT,
        args.precision,
    )
    default_atol, default_rtol = default_tolerances(precision)
    atol = args.atol if args.atol is not None else default_atol
    rtol = args.rtol if args.rtol is not None else default_rtol

    if args.fixture:
        points, cond, fixture_expected = load_gaussian_fixture(args.fixture)
        input_source = str(args.fixture.expanduser().resolve())
    else:
        points, cond = deterministic_gaussian_inputs(args.seed)
        fixture_expected = None
        input_source = f"deterministic synthetic inputs (seed={args.seed})"
    validate_gaussian_inputs(points, cond)

    device = choose_torch_device(torch, args.device)
    print(f"Running official PyTorch decoder.gs ({precision} internal) on {device}")
    reference, torch_duration_ms = run_pytorch_reference(
        torch=torch,
        component=COMPONENT,
        triposplat_repo=args.triposplat_repo,
        weights=args.weights,
        device=device,
        internal_precision=precision,
        inputs=(points, cond),
    )
    validate_array(reference, "PyTorch features", FEATURES_SHAPE)

    providers = args.providers or ["CPUExecutionProvider"]
    print(f"Running Python ONNX Runtime with providers {providers}")
    candidate, ort_duration_ms, session = validate_with_ort(
        ort=ort,
        graph_path=graph_path,
        providers=providers,
        threads=args.session_threads,
        output_name="features",
        feeds={"points": points, "cond": cond},
    )
    validate_array(candidate, "ONNX Runtime features", FEATURES_SHAPE)

    metrics = comparison_metrics(reference, candidate, atol=atol, rtol=rtol)
    fixture_metrics = None
    passed = bool(metrics["passed"])
    if fixture_expected is not None:
        fixture_metrics = comparison_metrics(
            fixture_expected,
            reference,
            atol=atol,
            rtol=rtol,
        )
        passed = bool(passed and fixture_metrics["passed"])

    report: dict[str, Any] = {
        "passed": passed,
        "component": COMPONENT,
        "contract": {
            "points": {
                "shape": list(POINTS_SHAPE),
                "dtype": "float32",
                "range": [0.0, 1.0],
            },
            "cond": {"shape": list(COND_SHAPE), "dtype": "float32"},
            "features": {"shape": list(FEATURES_SHAPE), "dtype": "float32"},
            "internal_precision": precision,
            "gaussians_per_point": 32,
        },
        "graph_metadata": metadata,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size} for path in artifacts
        ],
        "inputs": {
            "source": input_source,
            "points": array_summary(points),
            "cond": array_summary(cond),
        },
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
            "note": "Measured validation runs only; these are not browser WebGPU benchmarks.",
        },
        "runtime": validation_runtime_report(
            torch=torch,
            ort=ort,
            device=device,
            session=session,
            graph_path=graph_path,
        ),
    }

    if args.save_fixture:
        fixture_path = args.save_fixture.expanduser().resolve()
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fixture_path,
            points=np.ascontiguousarray(points, dtype=np.float32),
            cond=np.ascontiguousarray(cond, dtype=np.float32),
            features=np.ascontiguousarray(reference, dtype=np.float32),
            metadata=np.asarray(
                json.dumps(
                    {
                        "component": COMPONENT,
                        "internal_precision": precision,
                        "source": "official TripoSplat decoder.gs PyTorch reference",
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
        "PASS: official PyTorch and Python ONNX Runtime features agree"
        if report["passed"]
        else "FAIL: Gaussian features are outside the requested tolerances"
    )
    return report


def main() -> None:
    report = validate(parse_args())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

