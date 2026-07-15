#!/usr/bin/env python3
"""Expand a validated DiT NPZ into browser-fetchable float32 tensor files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SHAPES = {
    "latent": (1, 8192, 16),
    "camera": (1, 1, 5),
    "t": (1,),
    "feature1": (1, 4101, 1280),
    "feature2": (1, 4101, 128),
    "pred_latent": (1, 8192, 16),
    "pred_camera": (1, 1, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True, help="Validated DiT NPZ fixture.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Browser fixture directory.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"NumPy is required: {exc}") from exc

    args = parse_args()
    fixture = args.fixture.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    output.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}
    with np.load(fixture, allow_pickle=False) as values:
        missing = [name for name in SHAPES if name not in values.files]
        if missing:
            raise KeyError(f"{fixture} is missing {missing}")
        for name, shape in SHAPES.items():
            array = np.ascontiguousarray(values[name], dtype="<f4")
            if tuple(array.shape) != shape:
                raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinity")
            path = output / f"{name}.f32"
            array.tofile(path)
            files[name] = {
                "path": path.name,
                "dtype": "float32-le",
                "shape": list(shape),
                "elements": int(array.size),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        metadata = str(values["metadata"]) if "metadata" in values.files else None

    manifest = {
        "format": "triposplat-dit-browser-fixture-v1",
        "source": str(fixture),
        "metadata": json.loads(metadata) if metadata else None,
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
