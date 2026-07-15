#!/usr/bin/env python3
"""Export the official TripoSplat Gaussian feature decoder to fixed-shape ONNX.

Public graph contract (all tensors float32):

* ``points``: systematic-octree samples ``[1, 8192, 3]`` in ``[0, 1]``
* ``cond``: sampled TripoSplat latent ``[1, 8192, 16]``
* ``features``: raw Gaussian features ``[1, 8192, 480]``

The 480 channels are produced directly by the official
``ElasticGaussianFixedlenDecoder``.  This graph does not duplicate ``_get_offset``,
``_build_gaussians``, representation scaling/biases, or activation functions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from decoder_onnx_common import (
    COND_SHAPE,
    FEATURES_SHAPE,
    OFFICIAL_REPOSITORY_URL,
    POINTS_SHAPE,
    adapt_official_decoder_for_onnx,
    choose_torch_device,
    export_fixed_decoder_graph,
    load_official_decoder,
    make_dummy_inputs,
    make_gaussian_features_graph,
    sha256,
    source_commit,
)


COMPONENT = "gaussian_decoder"


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
        help="Official triposplat_vae_decoder_fp16.safetensors checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/models/triposplat/gaussian_decoder.onnx"),
        help="Destination ONNX graph (default: %(default)s).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help=(
            "Internal parameter/compute precision. Public graph I/O stays float32 "
            "for both choices (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
        help="PyTorch device used while tracing (default: %(default)s).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version (default: %(default)s).",
    )
    parser.add_argument(
        "--external-data-threshold",
        type=int,
        default=1024,
        metavar="BYTES",
        help=(
            "Initializers at least this large go into one .onnx.data sidecar "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip path-based onnx.checker validation after consolidation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose torch.onnx tracing output.",
    )
    args = parser.parse_args()
    if args.opset < 18:
        parser.error("--opset must be at least 18 for scaled dot-product attention export")
    if args.external_data_threshold < 0:
        parser.error("--external-data-threshold must be non-negative")
    return args


def export_graph(args: argparse.Namespace) -> list[Path]:
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing export dependency. Install a PyTorch-supported Python version "
            "and run `python -m pip install -r scripts/triposplat/requirements.txt`. "
            f"Original error: {exc}"
        ) from exc

    repo = args.triposplat_repo.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = choose_torch_device(torch, args.device)

    print(f"Loading official TripoSplat decoder via load_decoder from {repo}")
    decoder = load_official_decoder(
        torch=torch,
        triposplat_repo=repo,
        weights=weights,
        device=device,
        internal_precision=args.precision,
    )
    adapter = adapt_official_decoder_for_onnx(torch, decoder.gs)
    graph = make_gaussian_features_graph(torch, decoder).to(device=device).eval()
    # The wrapper owns decoder.gs only; drop the unused octree sibling before tracing.
    del decoder
    dummy_inputs = make_dummy_inputs(torch, COMPONENT, device)

    metadata = {
        "triposplat.component": COMPONENT,
        "triposplat.source_repository": OFFICIAL_REPOSITORY_URL,
        "triposplat.source_commit": source_commit(repo),
        "triposplat.checkpoint_filename": weights.name,
        "triposplat.internal_precision": args.precision,
        "triposplat.public_io_precision": "float32",
        "triposplat.points_input": "points [1,8192,3] float32 normalized coordinates [0,1]",
        "triposplat.cond_input": "cond [1,8192,16] float32 sampled latent",
        "triposplat.output": "features [1,8192,480] float32 raw decoder features",
        "triposplat.feature_layout": (
            "official ElasticGaussianFixedlenDecoder layout; 32 Gaussians per point"
        ),
        "triposplat.excluded_host_logic": (
            "_get_offset, _build_gaussians, representation scales/biases/activations"
        ),
        "triposplat.attention_query_chunk": str(adapter["attention_query_chunk"]),
        "triposplat.attention_modules": str(adapter["attention_modules"]),
        "triposplat.qk_norm_padding_tokens": str(adapter["qk_norm_padding_tokens"]),
        "triposplat.qk_norm_modules": str(adapter["qk_norm_modules"]),
        "triposplat.attention_output_modules": str(adapter["attention_output_modules"]),
    }
    print(
        f"Tracing {args.precision}-internal graph on {device}: "
        f"points={POINTS_SHAPE}, cond={COND_SHAPE} -> features={FEATURES_SHAPE}"
    )
    artifacts = export_fixed_decoder_graph(
        torch=torch,
        onnx=onnx,
        graph=graph,
        dummy_inputs=dummy_inputs,
        input_names=("points", "cond"),
        output_name="features",
        output_path=output,
        component=COMPONENT,
        internal_precision=args.precision,
        opset=args.opset,
        external_data_threshold=args.external_data_threshold,
        metadata=metadata,
        verbose=args.verbose,
        run_checker=not args.skip_check,
    )

    print(f"Published fixed float32 I/O contract: features {FEATURES_SHAPE}")
    for artifact in artifacts:
        print(
            f"Wrote {artifact} ({artifact.stat().st_size:,} bytes, "
            f"sha256={sha256(artifact)})"
        )
    return artifacts


def main() -> None:
    export_graph(parse_args())


if __name__ == "__main__":
    main()
