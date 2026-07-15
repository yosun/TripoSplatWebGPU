#!/usr/bin/env python3
"""Expand a validated decoder NPZ into browser-fetchable float32 files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONTRACTS = {
    "octree": {
        "x": (1, 8192, 3),
        "l": (1,),
        "cond": (1, 8192, 16),
        "logits": (1, 8192, 8),
    },
    "gaussian": {
        "points": (1, 8192, 3),
        "cond": (1, 8192, 16),
        "features": (1, 8192, 480),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=tuple(CONTRACTS), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fixture = args.fixture.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    with np.load(fixture, allow_pickle=False) as values:
        for name, shape in CONTRACTS[args.component].items():
            if name not in values.files:
                raise KeyError(f"{fixture} is missing {name!r}; found {values.files}")
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
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "format": "triposplat-decoder-browser-fixture-v1",
        "component": args.component,
        "source": str(fixture),
        "files": files,
    }
    destination = output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
