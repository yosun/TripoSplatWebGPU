#!/usr/bin/env python3
"""Validate octree occupancy ONNX logits against official TripoSplat PyTorch.

The command feeds identical float32 ``x``, ``l``, and ``cond`` tensors to the
upstream ``decoder.octree`` wrapper and Python ONNX Runtime, prints detailed error
metrics, and exits nonzero unless every value satisfies the configured allclose gate.
It does not run or reimplement the data-dependent octree sampler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decoder_onnx_common import (
    COND_SHAPE,
    LEVEL_SHAPE,
    LOGITS_SHAPE,
    OFFICIAL_REPOSITORY_URL,
    POINTS_SHAPE,
    array_summary,
    choose_torch_device,
    comparison_metrics,
    default_tolerances,
    deterministic_occupancy_inputs,
    load_graph_contract,
    load_occupancy_fixture,
    run_pytorch_reference,
    validate_array,
    validate_occupancy_inputs,
    validate_with_ort,
    validation_runtime_report,
    write_json,
)


COMPONENT = "octree_occupancy_decoder"


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
        help="Exported fixed-shape occupancy ONNX graph.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Optional NPZ with x, l, cond and optionally logits.",
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
        help="Optional NPZ destination containing inputs and official logits.",
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
        x, level, cond, fixture_expected = load_occupancy_fixture(args.fixture)
        input_source = str(args.fixture.expanduser().resolve())
    else:
        x, level, cond = deterministic_occupancy_inputs(args.seed)
        fixture_expected = None
        input_source = f"deterministic synthetic inputs (seed={args.seed})"
    validate_occupancy_inputs(x, level, cond)

    device = choose_torch_device(torch, args.device)
    print(f"Running official PyTorch decoder.octree ({precision} internal) on {device}")
    reference, torch_duration_ms = run_pytorch_reference(
        torch=torch,
        component=COMPONENT,
        triposplat_repo=args.triposplat_repo,
        weights=args.weights,
        device=device,
        internal_precision=precision,
        inputs=(x, level, cond),
    )
    validate_array(reference, "PyTorch logits", LOGITS_SHAPE)

    providers = args.providers or ["CPUExecutionProvider"]
    print(f"Running Python ONNX Runtime with providers {providers}")
    candidate, ort_duration_ms, session = validate_with_ort(
        ort=ort,
        graph_path=graph_path,
        providers=providers,
        threads=args.session_threads,
        output_name="logits",
        feeds={"x": x, "l": level, "cond": cond},
    )
    validate_array(candidate, "ONNX Runtime logits", LOGITS_SHAPE)

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
            "x": {"shape": list(POINTS_SHAPE), "dtype": "float32", "range": [0.0, 1.0]},
            "l": {
                "shape": list(LEVEL_SHAPE),
                "dtype": "float32",
                "semantics": "current octree resolution: 2,4,...,256",
            },
            "cond": {"shape": list(COND_SHAPE), "dtype": "float32"},
            "logits": {"shape": list(LOGITS_SHAPE), "dtype": "float32"},
            "internal_precision": precision,
        },
        "graph_metadata": metadata,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size} for path in artifacts
        ],
        "inputs": {
            "source": input_source,
            "x": array_summary(x),
            "l": array_summary(level),
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
            x=np.ascontiguousarray(x, dtype=np.float32),
            l=np.ascontiguousarray(level, dtype=np.float32),
            cond=np.ascontiguousarray(cond, dtype=np.float32),
            logits=np.ascontiguousarray(reference, dtype=np.float32),
            metadata=np.asarray(
                json.dumps(
                    {
                        "component": COMPONENT,
                        "internal_precision": precision,
                        "source": "official TripoSplat decoder.octree PyTorch reference",
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
        "PASS: official PyTorch and Python ONNX Runtime logits agree"
        if report["passed"]
        else "FAIL: occupancy logits are outside the requested tolerances"
    )
    return report


def main() -> None:
    report = validate(parse_args())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

