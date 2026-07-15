#!/usr/bin/env python3
"""Generate a small official-PyTorch oracle for host Gaussian activation math.

The neural Gaussian decoder emits 480 raw features per octree point.  The browser
then applies the official ElasticGaussianFixedlenDecoder representation semantics
on the host.  This fixture calls the upstream ``_build_gaussians`` implementation
directly, without loading decoder weights, so TypeScript tests are gated against
the numerical source of truth rather than a second handwritten formula.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from decoder_onnx_common import import_official_triposplat, source_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triposplat-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--point-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.point_count <= 0:
        raise ValueError("--point-count must be positive")

    import torch

    upstream = import_official_triposplat(args.triposplat_repo)
    model_module = sys.modules.get("model")
    if model_module is None:
        raise RuntimeError("Official model.py was not loaded")
    decoder_type = model_module.ElasticGaussianFixedlenDecoder

    class ActivationOracle(torch.nn.Module):
        _calc_layout = decoder_type._calc_layout
        _build_perturbation = decoder_type._build_perturbation
        _get_offset = decoder_type._get_offset

        def __init__(self) -> None:
            super().__init__()
            self.rep_config = copy.deepcopy(upstream.GS_DECODER_ARGS["representation_config"])
            self.use_learned_offset_scale = True
            self.use_per_offset = True
            self._calc_layout()
            self._build_perturbation()

    rng = np.random.default_rng(args.seed)
    points = rng.uniform(0.05, 0.95, size=(1, args.point_count, 3)).astype(np.float32)
    features = rng.normal(0.0, 2.0, size=(1, args.point_count, 480)).astype(np.float32)
    # Exercise stable softplus/sigmoid branches and non-trivial quaternion lanes.
    probes = np.asarray([-32.0, -12.0, -2.0, 0.0, 2.0, 12.0, 32.0], dtype=np.float32)
    flat = features.reshape(-1)
    flat[: probes.size] = probes
    flat[192 : 192 + probes.size] = probes
    flat[416 : 416 + probes.size] = probes
    flat[448 : 448 + probes.size] = probes

    oracle = ActivationOracle()
    with torch.no_grad():
        gaussian = upstream._build_gaussians(
            oracle,
            {"points": torch.from_numpy(points)},
            {"features": torch.from_numpy(features)},
        )[0]

    payload = {
        "source": {
            "repository": "https://github.com/VAST-AI-Research/TripoSplat",
            "commit": source_commit(args.triposplat_repo),
            "function": "triposplat._build_gaussians",
        },
        "settings": {
            "seed": args.seed,
            "point_count": args.point_count,
            "gaussians_per_point": 32,
            "feature_width": 480,
        },
        "inputs": {
            "points": points.reshape(-1).tolist(),
            "features": features.reshape(-1).tolist(),
        },
        "expected": {
            "positions": gaussian.get_xyz.detach().cpu().float().reshape(-1).tolist(),
            "scales": gaussian.get_scaling.detach().cpu().float().reshape(-1).tolist(),
            "rotations": (
                gaussian._rotation + gaussian.rots_bias[None, :]
            ).detach().cpu().float().reshape(-1).tolist(),
            "opacities": gaussian.get_opacity.detach().cpu().float().reshape(-1).tolist(),
            "spherical_harmonics": (
                gaussian._features_dc[:, 0, :]
            ).detach().cpu().float().reshape(-1).tolist(),
        },
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
