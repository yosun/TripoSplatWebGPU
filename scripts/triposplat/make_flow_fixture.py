#!/usr/bin/env python3
"""Generate an official PyTorch reference for the browser-controlled flow loop."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dit_common import (
    CAMERA_SHAPE,
    FEATURE1_SHAPE,
    FEATURE2_SHAPE,
    LATENT_SHAPE,
    choose_torch_device,
    load_official_flow_model,
    resolved_file,
    sha256_file,
    source_revision,
    synchronize_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triposplat-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input-fixture-dir", type=Path, required=True)
    parser.add_argument("--output-fixture-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--internal-precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Official flow-model precision used for the reference (default: %(default)s).",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument(
        "--record-trajectory",
        action="store_true",
        help="Record official per-invocation sample, timestep, and raw predictions.",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.guidance_scale <= 1:
        parser.error("--guidance-scale must be greater than one to exercise CFG")
    if args.shift <= 0:
        parser.error("--shift must be positive")
    return args


def hard_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        destination.unlink()
    os.link(source, destination)


def main() -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch and NumPy are required: {exc}") from exc

    args = parse_args()
    repository = args.triposplat_repo.expanduser().resolve()
    weights = resolved_file(args.weights, "TripoSplat flow-model weights")
    input_dir = args.input_fixture_dir.expanduser().resolve()
    output_dir = args.output_fixture_dir.expanduser().resolve()
    shapes = {
        "latent": LATENT_SHAPE,
        "camera": CAMERA_SHAPE,
        "feature1": FEATURE1_SHAPE,
        "feature2": FEATURE2_SHAPE,
    }
    arrays = {}
    for name, shape in shapes.items():
        path = resolved_file(input_dir / f"{name}.f32", f"fixture {name}")
        array = np.fromfile(path, dtype="<f4")
        expected = int(np.prod(shape))
        if array.size != expected:
            raise ValueError(f"{path} has {array.size} floats; expected {expected}")
        arrays[name] = array.reshape(shape)

    device = choose_torch_device(torch, args.device)
    commit, dirty = source_revision(repository)
    print(f"Loading untouched official flow model at {commit} on {device}")
    model, source = load_official_flow_model(
        torch=torch,
        triposplat_repo=repository,
        weights=weights,
        device=device,
        internal_precision=args.internal_precision,
        low_memory_construction=True,
    )
    noise = {
        "latent": torch.from_numpy(arrays["latent"]).to(device=device, dtype=torch.float32),
        "camera": torch.from_numpy(arrays["camera"]).to(device=device, dtype=torch.float32),
    }
    condition = {
        "feature1": torch.from_numpy(arrays["feature1"]).to(device=device, dtype=torch.float32),
        "feature2": torch.from_numpy(arrays["feature2"]).to(device=device, dtype=torch.float32),
    }
    negative_condition = {name: torch.zeros_like(value) for name, value in condition.items()}
    sampler = source.pipeline_module.FlowEulerCfgSampler()
    trajectory = []

    class RecordingModel:
        def __call__(self, x_t, timestep, cond):
            prediction = model(x_t, timestep, cond)
            trajectory.append(
                {
                    "sample": {name: value.detach().clone() for name, value in x_t.items()},
                    "timestep": timestep.detach().clone(),
                    "prediction": {
                        name: value.detach().clone() for name, value in prediction.items()
                    },
                }
            )
            return prediction

    sampling_model = RecordingModel() if args.record_trajectory else model

    synchronize_torch(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = sampler.sample(
            sampling_model,
            {name: value.clone() for name, value in noise.items()},
            cond=condition,
            neg_cond=negative_condition,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
        )
    synchronize_torch(torch, device)
    duration_ms = (time.perf_counter() - started) * 1000

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in shapes:
        hard_link(input_dir / f"{name}.f32", output_dir / f"{name}.f32")
    output_files = {}
    for name, tensor in result.items():
        path = output_dir / f"flow{args.steps}_{name}.f32"
        np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype="<f4").tofile(path)
        output_files[name] = {
            "path": path.name,
            "shape": list(tensor.shape),
            "sha256": sha256_file(path),
        }

    trajectory_files = []
    if args.record_trajectory:
        expected_invocations = args.steps * 2
        if len(trajectory) != expected_invocations:
            raise RuntimeError(
                f"Recorded {len(trajectory)} model calls; expected {expected_invocations}"
            )
        trajectory_dir = output_dir / "trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(trajectory):
            invocation = index + 1
            prefix = f"invocation_{invocation:02d}"
            paths = {}
            tensors = {
                "sample_latent": record["sample"]["latent"],
                "sample_camera": record["sample"]["camera"],
                "t": record["timestep"],
                "pred_latent": record["prediction"]["latent"],
                "pred_camera": record["prediction"]["camera"],
            }
            for name, tensor in tensors.items():
                path = trajectory_dir / f"{prefix}_{name}.f32"
                np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype="<f4").tofile(path)
                paths[name] = {
                    "path": str(path.relative_to(output_dir)),
                    "shape": list(tensor.shape),
                    "sha256": sha256_file(path),
                }
            trajectory_files.append(
                {
                    "invocation": invocation,
                    "step": index // 2 + 1,
                    "pass": "conditional" if index % 2 == 0 else "unconditional",
                    "timestep": float(record["timestep"].flatten()[0].detach().cpu()),
                    "tensors": paths,
                }
            )

    manifest = {
        "source": {
            "repository": str(repository),
            "commit": commit,
            "tracked_source_dirty": dirty,
            "loader": "official triposplat.load_flow_model",
            "sampler": "official triposplat.FlowEulerCfgSampler",
        },
        "settings": {
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "internal_precision": args.internal_precision,
            "conditional_invocations": args.steps,
            "unconditional_invocations": args.steps,
        },
        "pytorch": {"device": str(device), "duration_ms": duration_ms},
        "outputs": output_files,
    }
    if trajectory_files:
        manifest["trajectory"] = trajectory_files
    manifest_path = output_dir / "flow.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path} ({duration_ms:.1f} ms official PyTorch sampling)")


if __name__ == "__main__":
    main()
