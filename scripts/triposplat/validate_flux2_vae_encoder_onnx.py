#!/usr/bin/env python3
"""Compare the TripoSplat Flux2 VAE ONNX slice with official PyTorch output.

The same image and explicit epsilon tensor are fed to both runtimes.  The command
prints absolute/relative error metrics, enforces configurable allclose tolerances,
and can save a JSON report plus an NPZ fixture for browser-side comparison.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from flux2_vae_common import (
    IMAGE_SHAPE,
    LATENT_SHAPE,
    OFFICIAL_REPOSITORY_URL,
    OUTPUT_SHAPE,
    choose_torch_device,
    deterministic_inputs,
    load_input_fixture,
    load_official_encoder,
    make_explicit_noise_encoder,
    synchronize_torch,
    tensor_type_to_precision,
    validate_input_arrays,
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
        help="Local flux2-vae.safetensors checkpoint used for the ONNX export.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Exported fixed-shape Flux2 VAE ONNX graph.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help=(
            "Optional NPZ containing image_rgb [1,3,1024,1024] and epsilon "
            "[1,32,128,128]. If omitted, deterministic synthetic inputs are used."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260516,
        help="Seed for synthetic epsilon when --fixture is omitted (default: %(default)s).",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="Expected graph precision; auto reads the ONNX input type (default: %(default)s).",
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
            "Python ONNX Runtime execution provider. Repeat to set fallback order "
            "(default: CPUExecutionProvider)."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        help="Absolute allclose tolerance (default: 6e-3 fp32, 2e-2 fp16).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        help="Relative allclose tolerance (default: 1e-3 fp32, 2e-2 fp16).",
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
            "Optional NPZ destination containing graph-typed image_rgb/epsilon and "
            "the official float32 feature2 reference for browser parity tests."
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


def require_fixed_contract(session: Any) -> str:
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    if set(inputs) != {"image_rgb", "epsilon"}:
        raise ValueError(
            f"ONNX inputs are {sorted(inputs)}; expected exactly image_rgb and epsilon"
        )
    if set(outputs) != {"feature2"}:
        raise ValueError(f"ONNX outputs are {sorted(outputs)}; expected exactly feature2")
    if tuple(inputs["image_rgb"].shape) != IMAGE_SHAPE:
        raise ValueError(
            f"ONNX image_rgb shape is {inputs['image_rgb'].shape}; expected {IMAGE_SHAPE}"
        )
    if tuple(inputs["epsilon"].shape) != LATENT_SHAPE:
        raise ValueError(
            f"ONNX epsilon shape is {inputs['epsilon'].shape}; expected {LATENT_SHAPE}"
        )
    if tuple(outputs["feature2"].shape) != OUTPUT_SHAPE:
        raise ValueError(
            f"ONNX feature2 shape is {outputs['feature2'].shape}; expected {OUTPUT_SHAPE}"
        )
    image_precision = tensor_type_to_precision(inputs["image_rgb"].type)
    epsilon_precision = tensor_type_to_precision(inputs["epsilon"].type)
    if image_precision != epsilon_precision:
        raise ValueError(
            "image_rgb and epsilon use different ONNX element types: "
            f"{inputs['image_rgb'].type} vs {inputs['epsilon'].type}"
        )
    if outputs["feature2"].type != "tensor(float)":
        raise ValueError(
            f"ONNX feature2 type is {outputs['feature2'].type}; expected tensor(float)"
        )
    return image_precision


def array_summary(array: Any) -> dict[str, float]:
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "l2_norm": float(np.linalg.norm(values.ravel())),
    }


def comparison_metrics(reference: Any, candidate: Any, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np

    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        raise ValueError(
            f"Output shape mismatch: PyTorch {reference64.shape}, ORT {candidate64.shape}"
        )
    reference_finite = bool(np.isfinite(reference64).all())
    candidate_finite = bool(np.isfinite(candidate64).all())
    if not reference_finite or not candidate_finite:
        return {
            "passed": False,
            "reference_finite": reference_finite,
            "candidate_finite": candidate_finite,
            "count": int(reference64.size),
        }

    delta = candidate64 - reference64
    absolute = np.abs(delta)
    # The floor keeps relative-error metrics interpretable around reference zeros.
    relative = absolute / np.maximum(np.abs(reference64), 1e-6)
    allowed = atol + rtol * np.abs(reference64)
    within = absolute <= allowed
    ref_flat = reference64.ravel()
    candidate_flat = candidate64.ravel()
    denominator = float(np.linalg.norm(ref_flat) * np.linalg.norm(candidate_flat))
    cosine = float(np.dot(ref_flat, candidate_flat) / denominator) if denominator else 1.0
    worst_flat_index = int(np.argmax(absolute))
    worst_index = tuple(int(v) for v in np.unravel_index(worst_flat_index, absolute.shape))
    return {
        "passed": bool(within.all()),
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
        "count": int(reference64.size),
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_relative_error_at_1e-6_floor": float(relative.max()),
        "mean_relative_error_at_1e-6_floor": float(relative.mean()),
        "cosine_similarity": cosine,
        "fraction_within_tolerance": float(within.mean()),
        "worst_index": list(worst_index),
        "worst_reference": float(reference64[worst_index]),
        "worst_candidate": float(candidate64[worst_index]),
        "worst_allowed_error": float(allowed[worst_index]),
    }


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
    if args.precision != "auto" and args.precision != graph_precision:
        raise ValueError(
            f"--precision {args.precision} conflicts with ONNX inputs ({graph_precision})"
        )
    precision = graph_precision
    atol = args.atol if args.atol is not None else (6e-3 if precision == "fp32" else 2e-2)
    rtol = args.rtol if args.rtol is not None else (1e-3 if precision == "fp32" else 2e-2)

    if args.fixture:
        image, epsilon, fixture_expected = load_input_fixture(args.fixture)
        input_source = str(args.fixture.expanduser().resolve())
    else:
        image, epsilon = deterministic_inputs(args.seed)
        fixture_expected = None
        input_source = f"deterministic synthetic inputs (seed={args.seed})"
    validate_input_arrays(image, epsilon)

    numpy_dtype = np.float16 if precision == "fp16" else np.float32
    image_feed = np.ascontiguousarray(image.astype(numpy_dtype, copy=False))
    epsilon_feed = np.ascontiguousarray(epsilon.astype(numpy_dtype, copy=False))
    device = choose_torch_device(torch, args.device)
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32

    print(f"Loading official PyTorch encoder ({precision}) on {device}")
    encoder = load_official_encoder(
        torch=torch,
        triposplat_repo=args.triposplat_repo,
        weights=args.weights,
        device=device,
        precision=precision,
    )
    reference_model = make_explicit_noise_encoder(torch, encoder).eval().to(device)
    torch_image = torch.from_numpy(image_feed).to(device=device, dtype=torch_dtype)
    torch_epsilon = torch.from_numpy(epsilon_feed).to(device=device, dtype=torch_dtype)

    synchronize_torch(torch, device)
    torch_started = time.perf_counter()
    with torch.inference_mode():
        reference_tensor = reference_model(torch_image, torch_epsilon)
    synchronize_torch(torch, device)
    torch_duration_ms = (time.perf_counter() - torch_started) * 1000.0
    reference = reference_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    ort_started = time.perf_counter()
    candidate = session.run(
        ["feature2"],
        {"image_rgb": image_feed, "epsilon": epsilon_feed},
    )[0]
    ort_duration_ms = (time.perf_counter() - ort_started) * 1000.0
    candidate = np.asarray(candidate, dtype=np.float32)

    if tuple(reference.shape) != OUTPUT_SHAPE:
        raise ValueError(f"PyTorch output shape is {reference.shape}; expected {OUTPUT_SHAPE}")
    if tuple(candidate.shape) != OUTPUT_SHAPE:
        raise ValueError(f"ORT output shape is {candidate.shape}; expected {OUTPUT_SHAPE}")

    metrics = comparison_metrics(reference, candidate, atol=atol, rtol=rtol)
    fixture_reference_metrics = None
    overall_passed = bool(metrics["passed"])
    if fixture_expected is not None:
        fixture_reference_metrics = comparison_metrics(
            fixture_expected,
            reference,
            atol=atol,
            rtol=rtol,
        )
        # A fixture with a saved reference is part of the contract too.  A stale
        # checkpoint/fixture must not be hidden by fresh PyTorch-vs-ORT agreement.
        overall_passed = bool(overall_passed and fixture_reference_metrics["passed"])

    report: dict[str, Any] = {
        "passed": overall_passed,
        "contract": {
            "image_rgb": {"shape": list(IMAGE_SHAPE), "range": [0.0, 1.0], "precision": precision},
            "epsilon": {"shape": list(LATENT_SHAPE), "precision": precision},
            "feature2": {"shape": list(OUTPUT_SHAPE), "precision": "fp32"},
        },
        "inputs": {
            "source": input_source,
            "image_rgb": array_summary(image_feed),
            "epsilon": array_summary(epsilon_feed),
        },
        "comparison": metrics,
        "fixture_reference_comparison": fixture_reference_metrics,
        "outputs": {
            "pytorch": array_summary(reference),
            "onnxruntime": array_summary(candidate),
        },
        "tolerance": {"atol": atol, "rtol": rtol},
        "measured_single_run_ms": {
            "pytorch": torch_duration_ms,
            "onnxruntime_python": ort_duration_ms,
            "note": "Validation timings are measured single runs, not browser benchmarks.",
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
            image_rgb=image_feed,
            epsilon=epsilon_feed,
            feature2=reference,
            metadata=np.asarray(
                json.dumps(
                    {
                        "precision": precision,
                        "image_range": [0.0, 1.0],
                        "source": "official TripoSplat Flux2VAEEncoder PyTorch reference",
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
