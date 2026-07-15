#!/usr/bin/env python3
"""Record an official eight-level TripoSplat octree trajectory.

The fixture keeps the neural and host-controlled parts separate.  Every occupancy
logit comes from the untouched upstream decoder.  The script also records the
uniform variate assigned to each parent by upstream ``sample_probs`` and the final
per-point voxel jitter, so the browser TypeScript sampler can replay the exact
data-dependent trajectory without depending on PyTorch's random-number generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from decoder_onnx_common import (
    COND_SHAPE,
    OFFICIAL_REPOSITORY_URL,
    TOKEN_COUNT,
    choose_torch_device,
    load_official_decoder,
    resolved_file,
    sha256,
    source_commit,
    synchronize_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triposplat-repo",
        type=Path,
        required=True,
        help=f"Local clone of {OFFICIAL_REPOSITORY_URL}.",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--condition",
        type=Path,
        required=True,
        help="Raw little-endian float32 latent with shape (1,8192,16).",
    )
    parser.add_argument("--output-fixture-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    parser.add_argument("--internal-precision", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=TOKEN_COUNT)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.levels < 1 or args.levels > 8:
        parser.error("--levels must be in [1, 8]")
    if args.num_points <= 0 or args.num_points > TOKEN_COUNT:
        parser.error(f"--num-points must be in [1, {TOKEN_COUNT}]")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    return args


def write_array(np: Any, path: Path, value: Any, dtype: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=dtype)
    array.tofile(path)
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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
    repo = args.triposplat_repo.expanduser().resolve()
    weights = resolved_file(args.weights, "TripoSplat decoder checkpoint")
    condition_path = resolved_file(args.condition, "octree condition latent")
    output_dir = args.output_fixture_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_array = np.fromfile(condition_path, dtype="<f4")
    if condition_array.size != int(np.prod(COND_SHAPE)):
        raise ValueError(
            f"{condition_path} has {condition_array.size} floats; expected {int(np.prod(COND_SHAPE))}"
        )
    condition_array = condition_array.reshape(COND_SHAPE)

    device = choose_torch_device(torch, args.device)
    print(f"Loading official decoder at {source_commit(repo)} on {device}")
    decoder = load_official_decoder(
        torch,
        repo,
        weights,
        device,
        args.internal_precision,
    )
    condition = torch.from_numpy(condition_array).to(device=device, dtype=torch.float32)

    model_module = sys.modules.get(decoder.octree.__class__.__module__)
    if model_module is None:
        raise RuntimeError("Could not find the imported official model module")
    official_sample_probs = model_module.sample_probs
    original_rand = torch.rand
    original_rand_like = torch.rand_like
    pending_rand_calls: list[Any] = []
    final_jitter_calls: list[Any] = []
    level_records: list[dict[str, Any]] = []
    neural_records: list[dict[str, Any]] = []

    def recording_rand(*rand_args: Any, **rand_kwargs: Any) -> Any:
        value = original_rand(*rand_args, **rand_kwargs)
        pending_rand_calls.append(value.detach().clone())
        return value

    def recording_rand_like(*rand_args: Any, **rand_kwargs: Any) -> Any:
        value = original_rand_like(*rand_args, **rand_kwargs)
        final_jitter_calls.append(value.detach().clone())
        return value

    def recording_sample_probs(probs: Any, counts: Any, algo: str = "systematic") -> Any:
        call_start = len(pending_rand_calls)
        sampled = official_sample_probs(probs, counts, algo=algo)
        calls = pending_rand_calls[call_start:]
        flat_counts = counts.detach().reshape(-1).to(dtype=torch.long)
        unique_counts, inverse = flat_counts.unique(sorted=False, return_inverse=True)
        parent_uniforms = torch.zeros_like(flat_counts, dtype=torch.float32)
        call_index = 0
        group_order: list[int] = []
        for index, count in enumerate(unique_counts.tolist()):
            if count == 0:
                continue
            rows = (inverse == index).nonzero(as_tuple=False).squeeze(1)
            if call_index >= len(calls):
                raise RuntimeError("Official sample_probs made fewer torch.rand calls than expected")
            raw = calls[call_index].reshape(-1).to(parent_uniforms.device, dtype=torch.float32)
            if raw.numel() != rows.numel():
                raise RuntimeError(
                    f"Random call {call_index} has {raw.numel()} values for {rows.numel()} rows"
                )
            parent_uniforms.index_copy_(0, rows, raw)
            group_order.append(int(count))
            call_index += 1
        if call_index != len(calls):
            raise RuntimeError("Official sample_probs made unexpected extra torch.rand calls")
        level_records.append(
            {
                "counts": counts.detach().clone(),
                "sampled": sampled.detach().clone(),
                "parent_uniforms": parent_uniforms.reshape(counts.shape).detach().clone(),
                "count_group_order": group_order,
            }
        )
        return sampled

    class RecordingOccupancy:
        def __call__(self, x: Any, resolution: Any, cond: Any, num_points: Any) -> Any:
            del num_points
            result = decoder.octree(x, resolution, cond)
            neural_records.append(
                {
                    "x": x.detach().clone(),
                    "resolution": resolution.detach().clone(),
                    "logits": result["logits"].detach().clone(),
                }
            )
            return result

    torch.manual_seed(args.seed)
    torch.rand = recording_rand
    torch.rand_like = recording_rand_like
    model_module.sample_probs = recording_sample_probs
    synchronize_torch(torch, device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = decoder.octree.sample(
                RecordingOccupancy(),
                condition,
                num_points=args.num_points,
                level=args.levels,
                temperature=args.temperature,
                algo="systematic",
            )
        synchronize_torch(torch, device)
    finally:
        model_module.sample_probs = official_sample_probs
        torch.rand = original_rand
        torch.rand_like = original_rand_like
    duration_ms = (time.perf_counter() - started) * 1000

    if len(neural_records) != args.levels or len(level_records) != args.levels:
        raise RuntimeError(
            f"Recorded {len(neural_records)} neural calls and {len(level_records)} samples; "
            f"expected {args.levels} each"
        )
    if len(final_jitter_calls) != 1:
        raise RuntimeError(f"Recorded {len(final_jitter_calls)} final rand_like calls; expected one")

    hard_link(condition_path, output_dir / "condition.f32")
    condition_info = {
        "path": "condition.f32",
        "shape": list(COND_SHAPE),
        "dtype": "float32",
        "bytes": (output_dir / "condition.f32").stat().st_size,
        "sha256": sha256(output_dir / "condition.f32"),
    }
    levels_manifest = []
    for index, (neural, sampling) in enumerate(zip(neural_records, level_records), start=1):
        parent_count = int(neural["x"].shape[1])
        prefix = f"level_{index:02d}"
        files = {
            "parent_centers": write_array(
                np, output_dir / f"{prefix}_parent_centers.f32", neural["x"].float().cpu().numpy(), "<f4"
            ),
            "parent_counts": write_array(
                np, output_dir / f"{prefix}_parent_counts.u32", sampling["counts"].cpu().numpy(), "<u4"
            ),
            "logits": write_array(
                np, output_dir / f"{prefix}_logits.f32", neural["logits"].float().cpu().numpy(), "<f4"
            ),
            "sampled_child_counts": write_array(
                np, output_dir / f"{prefix}_sampled_child_counts.u32", sampling["sampled"].cpu().numpy(), "<u4"
            ),
            "parent_uniforms": write_array(
                np, output_dir / f"{prefix}_parent_uniforms.f32", sampling["parent_uniforms"].cpu().numpy(), "<f4"
            ),
        }
        levels_manifest.append(
            {
                "level": index,
                "resolution": int(neural["resolution"].reshape(-1)[0].item()),
                "parent_count": parent_count,
                "count_group_order": sampling["count_group_order"],
                "files": files,
            }
        )

    output_files = {
        "points": write_array(np, output_dir / "points.f32", result["points"].float().cpu().numpy(), "<f4"),
        "log_probabilities": write_array(
            np, output_dir / "log_probabilities.f32", result["log_probs"].float().cpu().numpy(), "<f4"
        ),
        "voxel_jitter": write_array(
            np, output_dir / "voxel_jitter.f32", final_jitter_calls[0].float().cpu().numpy(), "<f4"
        ),
    }
    manifest = {
        "source": {
            "repository": str(repo),
            "commit": source_commit(repo),
            "loader": "official triposplat.load_decoder",
            "sampler": "official model.OctreeProbabilityFixedlenDecoder.sample",
            "sample_probs": "official model.sample_probs",
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "condition": str(condition_path),
            "condition_sha256": hashlib.sha256(condition_path.read_bytes()).hexdigest(),
        },
        "settings": {
            "seed": args.seed,
            "levels": args.levels,
            "num_points": args.num_points,
            "temperature": args.temperature,
            "internal_precision": args.internal_precision,
            "device": str(device),
        },
        "pytorch": {"duration_ms": duration_ms},
        "condition": condition_info,
        "levels": levels_manifest,
        "outputs": output_files,
    }
    manifest_path = output_dir / "octree.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path} ({duration_ms:.1f} ms official PyTorch octree sampling)")


if __name__ == "__main__":
    main()
