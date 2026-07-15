#!/usr/bin/env python3
"""Unpack a validated DINOv3 NPZ into browser-readable parity assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Prepared 1024px RGB-on-black PNG.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--internal-precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Precision used by the validated ONNX graph (default: %(default)s).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_float32(path: Path, value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype="<f4")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "file": path.name,
        "dtype": "float32-le",
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    fixture_path = args.fixture.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not fixture_path.is_file() or not image_path.is_file():
        raise FileNotFoundError("Both --fixture and --image must exist.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        required = {"pixel_values", "feature1"}
        if not required.issubset(fixture.files):
            raise KeyError(f"{fixture_path} must contain {sorted(required)}; found {fixture.files}")
        pixel_values = np.asarray(fixture["pixel_values"])
        feature1 = np.asarray(fixture["feature1"])

    static_image = output_dir / "prepared.png"
    shutil.copy2(image_path, static_image)
    tolerances = (
        {
            "strict": {
                "absolute": 3e-4,
                "relative": 1e-3,
                "minimum_cosine_similarity": 0.999999999,
            },
            "qualification": {
                "absolute": 1e-3,
                "relative": 3e-3,
                "minimum_cosine_similarity": 0.99999999,
            },
        }
        if args.internal_precision == "fp32"
        else {
            "strict": {
                "absolute": 3e-2,
                "relative": 3e-2,
                "minimum_cosine_similarity": 0.99999,
            },
            "qualification": {
                "absolute": 1.5e-1,
                "relative": 3e-2,
                "minimum_cosine_similarity": 0.99999,
            },
        }
    )
    manifest = {
        "source": "official TripoSplat DINOv3 PyTorch plus FP32 post-normalization",
        "internal_precision": args.internal_precision,
        "tolerances": tolerances,
        "prepared_image": {
            "file": static_image.name,
            "width": 1024,
            "height": 1024,
            "sha256": sha256(static_image),
        },
        "pixel_values": write_float32(output_dir / "pixel_values.f32", pixel_values),
        "feature1": write_float32(output_dir / "feature1.f32", feature1),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
