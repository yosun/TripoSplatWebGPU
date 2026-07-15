#!/usr/bin/env python3
"""Export the official TripoSplat Flux2 VAE encoder as a fixed-shape ONNX graph.

Graph contract:

* ``image_rgb``: ``float32`` or ``float16`` ``[1, 3, 1024, 1024]`` in ``[0, 1]``
* ``epsilon``: matching dtype ``[1, 32, 128, 128]``
* ``feature2``: ``float32 [1, 4101, 128]``

The graph includes image range conversion, stochastic reparameterization with the
explicit epsilon input, 2x2 latent packing, the checkpointed BatchNorm running-stat
normalization, flatten/transpose, and the five-token DINO alignment prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from flux2_vae_common import (
    IMAGE_SHAPE,
    LATENT_SHAPE,
    OFFICIAL_REPOSITORY_URL,
    OUTPUT_SHAPE,
    choose_torch_device,
    load_official_encoder,
    make_explicit_noise_encoder,
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
        help="Local flux2-vae.safetensors checkpoint from VAST-AI/TripoSplat.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/models/triposplat/flux2_vae_encoder.onnx"),
        help="Destination ONNX graph (default: %(default)s).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help=(
            "Graph parameter/input precision. feature2 remains float32 exactly as in "
            "the official pipeline (default: %(default)s)."
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
        "--external-data",
        choices=("always", "auto", "inline"),
        default="always",
        help=(
            "Weight storage: one consolidated .onnx.data file, automatic based on "
            "the initial export, or inline in the graph (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--external-data-threshold",
        type=int,
        default=1024,
        metavar="BYTES",
        help=(
            "In 'always' mode, keep tensors smaller than this inline while putting all "
            "larger tensors in one data file (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip onnx.checker after consolidation (not recommended).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose PyTorch ONNX export logging.",
    )
    args = parser.parse_args()
    if args.opset < 18:
        parser.error("--opset must be at least 18 for scaled dot-product attention export")
    if args.external_data_threshold < 0:
        parser.error("--external-data-threshold must be non-negative")
    return args


def source_commit(repo: Path) -> str:
    """Return the checked-out source commit without making Git a hard dependency."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def set_metadata(model_proto: Any, values: dict[str, str]) -> None:
    """Replace this tool's metadata keys while retaining unrelated metadata."""

    retained = {entry.key: entry.value for entry in model_proto.metadata_props}
    retained.update(values)
    del model_proto.metadata_props[:]
    for key, value in sorted(retained.items()):
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def staged_graph_uses_external_data(onnx: Any, graph_path: Path) -> bool:
    proto = onnx.load_model(str(graph_path), load_external_data=False)
    return any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in proto.graph.initializer
    )


def fixed_value_shape(value_info: Any) -> tuple[int | None, ...]:
    tensor_type = value_info.type.tensor_type
    return tuple(
        int(dimension.dim_value) if dimension.HasField("dim_value") else None
        for dimension in tensor_type.shape.dim
    )


def verify_fixed_contract(onnx: Any, model_proto: Any, precision: str) -> None:
    """Reject an export whose public names, element types, or axes drifted."""

    inputs = {value.name: value for value in model_proto.graph.input}
    outputs = {value.name: value for value in model_proto.graph.output}
    if set(inputs) != {"image_rgb", "epsilon"}:
        raise RuntimeError(f"Exported ONNX inputs are {sorted(inputs)}, not image_rgb/epsilon")
    if set(outputs) != {"feature2"}:
        raise RuntimeError(f"Exported ONNX outputs are {sorted(outputs)}, not feature2")
    expected_input_type = (
        onnx.TensorProto.FLOAT16 if precision == "fp16" else onnx.TensorProto.FLOAT
    )
    contracts = (
        (inputs["image_rgb"], IMAGE_SHAPE, expected_input_type),
        (inputs["epsilon"], LATENT_SHAPE, expected_input_type),
        (outputs["feature2"], OUTPUT_SHAPE, onnx.TensorProto.FLOAT),
    )
    for value_info, expected_shape, expected_type in contracts:
        observed_shape = fixed_value_shape(value_info)
        observed_type = value_info.type.tensor_type.elem_type
        if observed_shape != expected_shape:
            raise RuntimeError(
                f"Exported {value_info.name} shape is {observed_shape}, expected {expected_shape}"
            )
        if observed_type != expected_type:
            raise RuntimeError(
                f"Exported {value_info.name} ONNX element type is {observed_type}, "
                f"expected {expected_type}"
            )


def consolidate_and_publish(
    onnx: Any,
    staged_export: Path,
    output_path: Path,
    external_mode: str,
    external_threshold: int,
    metadata: dict[str, str],
    run_checker: bool,
    expected_precision: str | None = None,
) -> list[Path]:
    """Consolidate exporter shards, validate, then atomically publish artifacts."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_filename = output_path.name + ".data"
    final_data_path = output_path.parent / data_filename
    original_was_external = staged_graph_uses_external_data(onnx, staged_export)

    model_proto = onnx.load_model(str(staged_export), load_external_data=True)
    # onnx.save_model can otherwise preserve the exporter's shard locations.  First
    # materialize every initializer, then choose one explicit publication policy.
    onnx.external_data_helper.convert_model_from_external_data(model_proto)
    set_metadata(model_proto, metadata)
    if expected_precision is not None:
        verify_fixed_contract(onnx, model_proto, expected_precision)

    write_external = external_mode == "always" or (
        external_mode == "auto" and original_was_external
    )
    with tempfile.TemporaryDirectory(
        dir=str(output_path.parent),
        prefix=".flux2-vae-publish-",
    ) as publish_dir_string:
        publish_dir = Path(publish_dir_string)
        publish_graph = publish_dir / output_path.name
        if write_external:
            threshold = external_threshold if external_mode == "always" else 0
            onnx.save_model(
                model_proto,
                str(publish_graph),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_filename,
                size_threshold=threshold,
                convert_attribute=False,
            )
            publish_data = publish_dir / data_filename
            if not publish_data.is_file():
                raise RuntimeError(
                    "ONNX requested external data but did not create the consolidated "
                    f"artifact {publish_data}"
                )
        else:
            onnx.save_model(model_proto, str(publish_graph), save_as_external_data=False)
            publish_data = None

        if run_checker:
            # Checking the path (rather than only the in-memory protobuf) also verifies
            # that every external-data offset resolves in the consolidated file.
            onnx.checker.check_model(str(publish_graph), full_check=True)

        # Publish data first and the graph second, so the final graph never points at a
        # missing new data file. os.replace is atomic on the destination filesystem.
        if publish_data is not None:
            os.replace(publish_data, final_data_path)
        os.replace(publish_graph, output_path)

    if not write_external and final_data_path.exists():
        # This is only the exact sidecar name owned by this graph, never a broad cleanup.
        final_data_path.unlink()
    artifacts = [output_path, final_data_path] if write_external else [output_path]
    # TemporaryDirectory files can inherit mode 0600. Make static model artifacts
    # readable by a CDN/web-server user while preserving all existing permission bits.
    for artifact in artifacts:
        artifact.chmod(artifact.stat().st_mode | stat.S_IRGRP | stat.S_IROTH)
    return artifacts


def export_graph(args: argparse.Namespace) -> list[Path]:
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing export dependency. Install a PyTorch-supported Python version and "
            "run `python -m pip install -r scripts/triposplat/requirements.txt`. "
            f"Original error: {exc}"
        ) from exc

    repo = args.triposplat_repo.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = choose_torch_device(torch, args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    print(f"Loading official Flux2VAEEncoder from {repo}")
    encoder = load_official_encoder(
        torch=torch,
        triposplat_repo=repo,
        weights=weights,
        device=device,
        precision=args.precision,
    )
    graph = make_explicit_noise_encoder(torch, encoder).eval().to(device)
    image = torch.zeros(IMAGE_SHAPE, dtype=dtype, device=device)
    epsilon = torch.zeros(LATENT_SHAPE, dtype=dtype, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output.parent),
        prefix=".flux2-vae-export-",
    ) as export_dir_string:
        staged_export = Path(export_dir_string) / output.name
        kwargs: dict[str, Any] = {
            "export_params": True,
            "do_constant_folding": True,
            "opset_version": args.opset,
            "input_names": ["image_rgb", "epsilon"],
            "output_names": ["feature2"],
            "verbose": args.verbose,
        }
        export_signature = inspect.signature(torch.onnx.export)
        if "dynamo" in export_signature.parameters:
            # The legacy tracer is deliberately selected: it has stable fixed-shape
            # SDPA lowering and does not add an onnxscript dependency.
            kwargs["dynamo"] = False
        if "external_data" in export_signature.parameters:
            # Let PyTorch shard if needed; consolidation below normalizes the result.
            kwargs["external_data"] = True

        print(
            f"Tracing {args.precision} graph on {device} with inputs "
            f"{IMAGE_SHAPE} and {LATENT_SHAPE}"
        )
        with torch.inference_mode():
            torch.onnx.export(
                graph,
                (image, epsilon),
                str(staged_export),
                **kwargs,
            )

        metadata = {
            "triposplat.component": "flux2_vae_encoder",
            "triposplat.source_repository": OFFICIAL_REPOSITORY_URL,
            "triposplat.source_commit": source_commit(repo),
            "triposplat.precision": args.precision,
            "triposplat.image_input": "image_rgb NCHW [1,3,1024,1024] RGB [0,1]",
            "triposplat.epsilon_input": "epsilon NCHW [1,32,128,128]",
            "triposplat.output": "feature2 [1,4101,128] float32",
            "triposplat.stochastic_sampling": "mean + exp(0.5 * logvar) * epsilon",
        }
        artifacts = consolidate_and_publish(
            onnx=onnx,
            staged_export=staged_export,
            output_path=output,
            external_mode=args.external_data,
            external_threshold=args.external_data_threshold,
            metadata=metadata,
            run_checker=not args.skip_check,
            expected_precision=args.precision,
        )

    print(f"Output contract: feature2 float32 {OUTPUT_SHAPE}")
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
