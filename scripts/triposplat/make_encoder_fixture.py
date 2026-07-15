#!/usr/bin/env python3
"""Create deterministic RGBA/preprocessed RGB inputs for Flux2 VAE parity.

With no --input, the script creates a synthetic transparent RGBA object.  It then
follows the alpha-present branch of official TripoSplat preprocessing: resize the
short side to 1024, erode alpha, crop a square around the alpha bounds with 1.2x
padding, resize to 1024, and composite on black.  It also writes an NPZ containing
``image_rgb`` and a seeded explicit ``epsilon`` tensor for the ONNX validator.

This helper intentionally does not export or emulate BiRefNet.  An opaque input must
first be background-removed elsewhere and supplied with meaningful alpha.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMAGE_SIZE = 1024
IMAGE_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
EPSILON_SHAPE = (1, 32, 128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional transparent RGBA source. Omit to generate a deterministic source.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("fixtures/triposplat/flux2_vae"),
        help=(
            "Output prefix for *_source_rgba.png, *_preprocessed_rgb.png and *_inputs.npz "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--erode-radius",
        type=int,
        default=1,
        help="Alpha MinFilter radius matching official preprocessing (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260516,
        help="NumPy PCG64 seed for explicit epsilon (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.erode_radius < 0:
        parser.error("--erode-radius must be non-negative")
    return args


def deterministic_rgba(np: Any, image_module: Any) -> Any:
    """Generate a stable, nontrivial transparent fixture without external assets."""

    width, height = 1280, 960
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xx - np.float32(width * 0.52)) / np.float32(width * 0.29)
    y = (yy - np.float32(height * 0.48)) / np.float32(height * 0.37)
    # A soft superellipse plus a smaller asymmetric lobe exercises alpha crop and
    # Lanczos resampling without relying on platform font or drawing rasterization.
    body_distance = np.power(np.abs(x), 3.2) + np.power(np.abs(y), 3.2)
    lobe_distance = (
        ((xx - np.float32(width * 0.70)) / np.float32(width * 0.12)) ** 2
        + ((yy - np.float32(height * 0.34)) / np.float32(height * 0.16)) ** 2
    )
    alpha = np.maximum(
        np.clip((np.float32(1.04) - body_distance) * np.float32(12.0), 0.0, 1.0),
        np.clip((np.float32(1.03) - lobe_distance) * np.float32(10.0), 0.0, 1.0),
    )
    stripe = np.float32(0.5) + np.float32(0.5) * np.sin(
        xx * np.float32(0.031) + yy * np.float32(0.017)
    )
    red = np.clip(np.float32(0.15) + np.float32(0.75) * xx / width, 0.0, 1.0)
    green = np.clip(np.float32(0.20) + np.float32(0.65) * yy / height, 0.0, 1.0)
    blue = np.clip(np.float32(0.18) + np.float32(0.70) * stripe, 0.0, 1.0)
    rgba = np.stack((red, green, blue, alpha), axis=-1)
    rgba_u8 = np.rint(rgba * np.float32(255.0)).astype(np.uint8)
    return image_module.fromarray(rgba_u8, mode="RGBA")


def preprocess_alpha_present(image: Any, np: Any, image_module: Any, image_filter: Any, radius: int) -> Any:
    """Mirror official preprocess_image after its alpha/background-removal choice."""

    image = image.convert("RGBA")
    width, height = image.size
    scale = IMAGE_SIZE / min(width, height)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    image = image.resize(resized_size, image_module.Resampling.LANCZOS)
    alpha_array = np.asarray(image.getchannel("A"), dtype=np.uint8)
    if int(alpha_array.min()) == 255:
        raise ValueError(
            "Input alpha is fully opaque. This helper does not run BiRefNet; provide a "
            "background-removed RGBA image with transparent pixels."
        )
    if radius:
        image.putalpha(image.getchannel("A").filter(image_filter.MinFilter(2 * radius + 1)))

    alpha_array = np.asarray(image.getchannel("A"), dtype=np.uint8)
    ys, xs = np.nonzero(alpha_array)
    if xs.size == 0:
        raise ValueError("Alpha erosion removed the entire foreground")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0 * 1.2
    if half <= 0:
        raise ValueError("Foreground alpha bounds are degenerate")
    image = image.crop(
        [
            int(center_x - half),
            int(center_y - half),
            int(center_x + half),
            int(center_y + half),
        ]
    )
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), image_module.Resampling.LANCZOS)
    background = image_module.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    background.paste(image, mask=image.getchannel("A"))
    return background


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def output_paths(prefix: Path) -> tuple[Path, Path, Path]:
    resolved = prefix.expanduser().resolve()
    return (
        resolved.parent / f"{resolved.name}_source_rgba.png",
        resolved.parent / f"{resolved.name}_preprocessed_rgb.png",
        resolved.parent / f"{resolved.name}_inputs.npz",
    )


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise SystemExit(
            "Missing fixture dependency. Run `python -m pip install numpy Pillow`. "
            f"Original error: {exc}"
        ) from exc

    if args.input:
        input_path = args.input.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"RGBA input does not exist: {input_path}")
        with Image.open(input_path) as opened:
            source = opened.convert("RGBA")
        source_description = str(input_path)
    else:
        source = deterministic_rgba(np, Image)
        source_description = "generated deterministic RGBA"

    preprocessed = preprocess_alpha_present(
        source,
        np=np,
        image_module=Image,
        image_filter=ImageFilter,
        radius=args.erode_radius,
    )
    image_rgb = (
        np.asarray(preprocessed, dtype=np.float32).transpose(2, 0, 1)[None, ...]
        / np.float32(255.0)
    )
    rng = np.random.default_rng(args.seed)
    epsilon = rng.standard_normal(EPSILON_SHAPE, dtype=np.float32)
    assert tuple(image_rgb.shape) == IMAGE_SHAPE

    source_path, rgb_path, fixture_path = output_paths(args.output_prefix)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source.save(source_path, format="PNG")
    preprocessed.save(rgb_path, format="PNG")
    metadata = {
        "source": source_description,
        "preprocess": "official alpha-present branch",
        "canvas_size": IMAGE_SIZE,
        "erode_radius": args.erode_radius,
        "epsilon_rng": "numpy.random.Generator(PCG64).standard_normal(float32)",
        "epsilon_seed": args.seed,
        "image_layout": "NCHW RGB float32 [0,1]",
    }
    np.savez_compressed(
        fixture_path,
        image_rgb=np.ascontiguousarray(image_rgb, dtype=np.float32),
        epsilon=np.ascontiguousarray(epsilon, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    for path in (source_path, rgb_path, fixture_path):
        print(f"Wrote {path} ({path.stat().st_size:,} bytes, sha256={checksum(path)})")


if __name__ == "__main__":
    main()

