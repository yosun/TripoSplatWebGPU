#!/usr/bin/env python3
"""Export official TripoSplat DINOv3 ViT-H/16+ for ONNX Runtime WebGPU.

Fixed graph contract:

* ``pixel_values``: float32 ``[1, 3, 1024, 1024]``, already normalized with
  ImageNet/DINOv3 mean and standard deviation;
* ``feature1``: float32 ``[1, 4101, 1280]``;
* model weights/attention/MLP activations: FP16 by default (FP32 optional).

The graph includes the extra FP32 non-affine layer norm applied by official
``triposplat.encode_image`` after ``DinoV3ViT.forward``.  BF16 or FP16 source
safetensors are converted to the requested internal precision; BF16 is never
emitted into the WebGPU artifact.  Parameters are consolidated into exactly one
``<output>.data`` external-data file.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dinov3_common import (
    EXTRA_LAYER_NORM_EPS,
    INPUT_NAME,
    INPUT_SHAPE,
    INTERNAL_PRECISION_METADATA_KEY,
    OFFICIAL_REPOSITORY_URL,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    adapt_official_dinov3_for_onnx,
    checkpoint_dtype_counts,
    choose_torch_device,
    import_official_model,
    load_official_encoder,
    make_browser_encoder,
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
        help="Local dino_v3_vit_h.safetensors checkpoint; never downloaded by this tool.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/models/triposplat/dinov3_encoder.onnx"),
        help="Destination ONNX graph (default: %(default)s).",
    )
    parser.add_argument(
        "--internal-precision",
        "--precision",
        dest="internal_precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help=(
            "Weight/activation precision. Public input/output remain FP32. FP16 is "
            "the WebGPU artifact and converts BF16 source weights (default: %(default)s)."
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
            "Keep tensors smaller than this inline; all others go into the single "
            ".onnx.data sidecar (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--attention-query-chunk",
        type=int,
        default=256,
        help="Independent query rows per explicit FP32 attention chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--attention-head-chunk",
        type=int,
        default=5,
        help="Independent heads per explicit FP32 attention chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--linear-token-chunk",
        type=int,
        default=256,
        help="Independent tokens per q/k/v/o and gated-MLP projection (default: %(default)s).",
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
    if (
        args.attention_query_chunk < 1
        or args.attention_head_chunk < 1
        or args.linear_token_chunk < 1
    ):
        parser.error("attention chunk sizes must be positive")
    if args.output.suffix.lower() != ".onnx":
        parser.error("--output must end in .onnx")
    return args


def source_commit(repo: Path) -> str:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def set_metadata(model_proto: Any, values: dict[str, str]) -> None:
    retained = {entry.key: entry.value for entry in model_proto.metadata_props}
    retained.update(values)
    del model_proto.metadata_props[:]
    for key, value in sorted(retained.items()):
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value


def fixed_value_shape(value_info: Any) -> tuple[int | None, ...]:
    return tuple(
        int(dimension.dim_value) if dimension.HasField("dim_value") else None
        for dimension in value_info.type.tensor_type.shape.dim
    )


def iter_graphs(onnx: Any, graph: Any) -> Iterator[Any]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from iter_graphs(onnx, attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from iter_graphs(onnx, child)


def iter_embedded_tensors(onnx: Any, graph: Any) -> Iterator[Any]:
    """Yield initializer and Constant tensors, including nested subgraphs."""

    for child in iter_graphs(onnx, graph):
        yield from child.initializer
        for sparse in child.sparse_initializer:
            yield sparse.values
            yield sparse.indices
        for node in child.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    yield attribute.t
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    yield from attribute.tensors


def verify_fixed_contract(onnx: Any, model_proto: Any, internal_precision: str) -> None:
    """Reject name, shape, dtype, BF16, and post-normalization drift."""

    inputs = {value.name: value for value in model_proto.graph.input}
    outputs = {value.name: value for value in model_proto.graph.output}
    if set(inputs) != {INPUT_NAME}:
        raise RuntimeError(f"Exported inputs are {sorted(inputs)}, expected only {INPUT_NAME}")
    if set(outputs) != {OUTPUT_NAME}:
        raise RuntimeError(f"Exported outputs are {sorted(outputs)}, expected only {OUTPUT_NAME}")

    contracts = (
        (inputs[INPUT_NAME], INPUT_SHAPE),
        (outputs[OUTPUT_NAME], OUTPUT_SHAPE),
    )
    for value, expected_shape in contracts:
        observed_shape = fixed_value_shape(value)
        observed_type = value.type.tensor_type.elem_type
        if observed_shape != expected_shape:
            raise RuntimeError(
                f"Exported {value.name} shape is {observed_shape}, expected {expected_shape}"
            )
        if observed_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(
                f"Exported {value.name} element type is {observed_type}, expected float32"
            )

    tensors = list(iter_embedded_tensors(onnx, model_proto.graph))
    bfloat16 = [
        tensor.name or "<unnamed Constant>"
        for tensor in tensors
        if tensor.data_type == onnx.TensorProto.BFLOAT16
    ]
    if bfloat16:
        raise RuntimeError(
            "Export retained BF16 tensors, which violates the WebGPU artifact contract: "
            + ", ".join(bfloat16[:8])
        )
    required_parameter_type = (
        onnx.TensorProto.FLOAT16 if internal_precision == "fp16" else onnx.TensorProto.FLOAT
    )
    initializer_types = {initializer.data_type for initializer in model_proto.graph.initializer}
    if required_parameter_type not in initializer_types:
        raise RuntimeError(
            f"Export has no {internal_precision} initializers; internal precision was not preserved"
        )

    # Returning the functional layer_norm directly makes it the public output's
    # producer with current supported PyTorch exporters.  Checking this catches the
    # easy-to-miss error of exporting DinoV3ViT alone and normalizing only in JS.
    producer = next(
        (node for node in model_proto.graph.node if OUTPUT_NAME in node.output),
        None,
    )
    if producer is None or producer.op_type != "LayerNormalization":
        observed = producer.op_type if producer is not None else "<none>"
        raise RuntimeError(
            f"{OUTPUT_NAME} producer is {observed}, expected the official extra "
            "FP32 LayerNormalization"
        )


def consolidate_and_publish(
    onnx: Any,
    staged_export: Path,
    output_path: Path,
    external_threshold: int,
    metadata: dict[str, str],
    internal_precision: str,
    run_checker: bool,
) -> list[Path]:
    """Consolidate exporter shards and atomically publish graph plus one sidecar."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_filename = output_path.name + ".data"
    final_data_path = output_path.parent / data_filename

    # Materialize any exporter-created shards, then deliberately choose one stable
    # relative external-data location for CDN hosting and ORT Web's externalData map.
    model_proto = onnx.load_model(str(staged_export), load_external_data=True)
    onnx.external_data_helper.convert_model_from_external_data(model_proto)
    set_metadata(model_proto, metadata)
    verify_fixed_contract(onnx, model_proto, internal_precision)

    with tempfile.TemporaryDirectory(
        dir=str(output_path.parent), prefix=".dinov3-publish-"
    ) as publish_dir_string:
        publish_dir = Path(publish_dir_string)
        publish_graph = publish_dir / output_path.name
        onnx.save_model(
            model_proto,
            str(publish_graph),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=external_threshold,
            convert_attribute=False,
        )
        publish_data = publish_dir / data_filename
        if not publish_data.is_file():
            raise RuntimeError(
                f"ONNX did not create required consolidated external data: {publish_data}"
            )
        published_proto = onnx.load_model(str(publish_graph), load_external_data=False)
        locations = {
            next(
                (
                    entry.value
                    for entry in initializer.external_data
                    if entry.key == "location"
                ),
                "",
            )
            for initializer in published_proto.graph.initializer
            if initializer.data_location == onnx.TensorProto.EXTERNAL
        }
        if locations != {data_filename}:
            raise RuntimeError(
                f"External-data locations are {sorted(locations)}, expected only "
                f"{data_filename!r}"
            )
        if run_checker:
            # A path check also resolves and verifies external-data locations.
            onnx.checker.check_model(str(publish_graph), full_check=True)

        os.replace(publish_data, final_data_path)
        os.replace(publish_graph, output_path)
    return [output_path, final_data_path]


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
    source_dtypes = checkpoint_dtype_counts(weights)
    print(f"Checkpoint tensor dtypes: {json.dumps(source_dtypes, sort_keys=True)}")
    if "BF16" in source_dtypes and args.internal_precision == "fp16":
        print("Converting BF16 source parameters to FP16 for ONNX Runtime WebGPU")

    print(f"Loading official DinoV3ViT from {repo}")
    encoder = load_official_encoder(
        torch=torch,
        triposplat_repo=repo,
        weights=weights,
        device=device,
        internal_precision=args.internal_precision,
    )
    adapter = adapt_official_dinov3_for_onnx(
        torch,
        encoder,
        import_official_model(repo),
        attention_query_chunk=args.attention_query_chunk,
        attention_head_chunk=args.attention_head_chunk,
        linear_token_chunk=args.linear_token_chunk,
    )
    graph = make_browser_encoder(torch, encoder, args.internal_precision).eval().to(device)
    pixel_values = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output.parent), prefix=".dinov3-export-"
    ) as export_dir_string:
        staged_export = Path(export_dir_string) / output.name
        kwargs: dict[str, Any] = {
            "export_params": True,
            "do_constant_folding": True,
            "opset_version": args.opset,
            "input_names": [INPUT_NAME],
            "output_names": [OUTPUT_NAME],
            "verbose": args.verbose,
        }
        export_signature = inspect.signature(torch.onnx.export)
        if "dynamo" in export_signature.parameters:
            # Legacy tracing currently gives stable fixed-shape SDPA lowering and
            # avoids an additional onnxscript dependency.
            kwargs["dynamo"] = False
        if "external_data" in export_signature.parameters:
            kwargs["external_data"] = True
        elif "use_external_data_format" in export_signature.parameters:
            kwargs["use_external_data_format"] = True

        print(
            f"Tracing {args.internal_precision} internals on {device}; public input "
            f"is float32 {INPUT_SHAPE}"
        )
        with torch.inference_mode():
            torch.onnx.export(graph, (pixel_values,), str(staged_export), **kwargs)

        metadata = {
            "triposplat.component": "dinov3_encoder",
            "triposplat.source_repository": OFFICIAL_REPOSITORY_URL,
            "triposplat.source_commit": source_commit(repo),
            INTERNAL_PRECISION_METADATA_KEY: args.internal_precision,
            "triposplat.checkpoint_tensor_dtypes": json.dumps(source_dtypes, sort_keys=True),
            "triposplat.input": (
                "pixel_values float32 NCHW [1,3,1024,1024], ImageNet normalized "
                "mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]"
            ),
            "triposplat.output": "feature1 float32 [1,4101,1280]",
            "triposplat.post_normalization": (
                f"FP32 non-affine layer_norm normalized_shape=[1280] eps={EXTRA_LAYER_NORM_EPS}"
            ),
            "triposplat.attention_export": "exact independent head/query chunks",
            "triposplat.attention_compute": "float32 scores, softmax, and value accumulation",
            "triposplat.attention_modules": str(adapter["attention_modules"]),
            "triposplat.attention_query_chunk": str(adapter["attention_query_chunk"]),
            "triposplat.attention_head_chunk": str(adapter["attention_head_chunk"]),
            "triposplat.linear_token_chunk": str(adapter["linear_token_chunk"]),
            "triposplat.mlp_modules": str(adapter["mlp_modules"]),
        }
        artifacts = consolidate_and_publish(
            onnx=onnx,
            staged_export=staged_export,
            output_path=output,
            external_threshold=args.external_data_threshold,
            metadata=metadata,
            internal_precision=args.internal_precision,
            run_checker=not args.skip_check,
        )

    print(f"Output contract: {OUTPUT_NAME} float32 {OUTPUT_SHAPE}")
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
