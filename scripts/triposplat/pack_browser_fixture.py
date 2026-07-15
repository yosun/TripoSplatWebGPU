#!/usr/bin/env python3
"""Unpack an NPZ parity fixture into static browser-readable little-endian files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True, help="NPZ with epsilon and feature2.")
    parser.add_argument("--image", type=Path, required=True, help="Prepared 1024x1024 RGB PNG.")
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional alpha-present source PNG used to exercise browser preprocessing.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Static output directory.")
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
    source_path = args.source.expanduser().resolve() if args.source else None
    output_dir = args.output_dir.expanduser().resolve()
    if not fixture_path.is_file() or not image_path.is_file():
        raise FileNotFoundError("Both --fixture and --image must be existing files.")
    if source_path is not None and not source_path.is_file():
        raise FileNotFoundError(f"--source does not exist: {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(fixture_path, allow_pickle=False) as fixture:
        if "epsilon" not in fixture.files or "feature2" not in fixture.files:
            raise KeyError(f"{fixture_path} must contain epsilon and feature2; found {fixture.files}")
        epsilon = np.asarray(fixture["epsilon"])
        feature2 = np.asarray(fixture["feature2"])

    static_image = output_dir / "prepared.png"
    shutil.copy2(image_path, static_image)
    manifest = {
        "image": {
            "file": static_image.name,
            "width": 1024,
            "height": 1024,
            "sha256": sha256(static_image),
        },
        "epsilon": write_float32(output_dir / "epsilon.f32", epsilon),
        "feature2": write_float32(output_dir / "feature2.f32", feature2),
    }
    if source_path is not None:
        static_source = output_dir / "source.png"
        shutil.copy2(source_path, static_source)
        manifest["source"] = {
            "file": static_source.name,
            "sha256": sha256(static_source),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
