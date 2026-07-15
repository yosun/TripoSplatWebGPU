#!/usr/bin/env python3
"""Export one official TripoSplat flow/DiT invocation to fixed-shape ONNX.

Public graph contract (all tensors are float32 and every axis is static):

* latent:       [1, 8192, 16]
* camera:       [1, 1, 5]
* t:            [1] (official sampler's already-scaled ``1000 * timestep``)
* feature1:     [1, 4101, 1280]
* feature2:     [1, 4101, 128]
* pred_latent:  [1, 8192, 16]
* pred_camera:  [1, 1, 5]

Weights and internal compute are fp16 by default.  Explicit graph-entry and graph-exit
casts retain float32 browser/host Euler state and preserve the official float32 timestep.
The artifact always uses one consolidated ``<graph>.data`` ONNX external-data sidecar.

The model is loaded only through the official repository's ``load_flow_model``.  The
source checkout is never edited.  Export-only real RoPE and fixed Sobol-position
adaptations are applied in memory and parity-gated against official operations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from dit_common import (
    FEATURE1_SHAPE,
    FEATURE2_SHAPE,
    INPUT_NAMES,
    INTERNAL_PRECISION_METADATA_KEY,
    LATENT_SHAPE,
    CAMERA_SHAPE,
    OFFICIAL_REPOSITORY_URL,
    OUTPUT_NAMES,
    PUBLIC_IO_METADATA_VALUE,
    TIMESTEP_SHAPE,
    adapt_official_flow_for_onnx,
    choose_torch_device,
    load_official_flow_model,
    make_browser_flow_step,
    sha256_file,
    source_revision,
    validate_real_rope_primitives,
    verify_external_data_files,
    verify_onnx_contract,
)


@dataclass
class ExternalSlice:
    tensor: Any
    source: Path
    offset: int
    length: int | None


TRACE_ATTENTION_TOKEN_PADDING = 256
RUNTIME_ATTENTION_TOKEN_PADDING = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--triposplat-repo",
        type=Path,
        required=True,
        help=f"Clean local clone of {OFFICIAL_REPOSITORY_URL}.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Official triposplat_fp16.safetensors flow-model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/models/triposplat/dit_step.onnx"),
        help="Destination ONNX graph (default: %(default)s).",
    )
    parser.add_argument(
        "--internal-precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help=(
            "Weight/compute precision inside the graph. Public I/O remains float32 "
            "for both choices (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
        help=(
            "PyTorch trace device. CPU is reproducible but expensive; use an "
            "accelerator with sufficient memory when available (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset (default: %(default)s).",
    )
    parser.add_argument(
        "--external-data-threshold",
        type=int,
        default=1024,
        metavar="BYTES",
        help=(
            "Move inline raw-data initializers at least this large into the single "
            "sidecar. Already-external tensors stay external (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--attention-query-chunk",
        type=int,
        default=256,
        metavar="TOKENS",
        help=(
            "Split SDPA along its independent query axis to bound WebGPU score "
            "buffers (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--attention-head-chunk",
        type=int,
        default=16,
        metavar="HEADS",
        help=(
            "Split SDPA along its independent head axis to avoid the corrupt "
            "16-head ORT WebGPU kernel (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--attention-head-padding",
        type=int,
        default=0,
        metavar="HEADS",
        help=(
            "Optionally append live duplicate head(s) to each SDPA group and discard "
            "their outputs (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--qk-norm-padding-tokens",
        type=int,
        default=1,
        metavar="TOKENS",
        help=(
            "Append live duplicate token(s) during Q/K RMS normalization and discard "
            "them, shielding the final reduction lane (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--rms-norm-eps",
        type=float,
        default=None,
        help=(
            "Opt into the Core AI-style x*rsqrt(mean(x^2)+eps) multi-head RMS "
            "norm rewrite. Omit to preserve official F.normalize exactly."
        ),
    )
    parser.add_argument(
        "--attention-output-chunk",
        type=int,
        default=256,
        metavar="CHANNELS",
        help=(
            "Split each independent attention output projection into bounded column "
            "groups to avoid the ORT WebGPU 1024-wide MatMul path (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--attention-output-reduction-chunk",
        type=int,
        default=256,
        metavar="CHANNELS",
        help=(
            "Split the reduction axis of each attention output projection so ORT "
            "WebGPU never selects its inaccurate 1024-wide kernel (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--low-memory-construction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Construct parameters directly in the selected precision before the "
            "official strict checkpoint load (default: enabled)."
        ),
    )
    parser.add_argument(
        "--allow-dirty-official-source",
        action="store_true",
        help=(
            "Allow tracked edits to official model.py/triposplat.py. The default "
            "rejects them so a locally patched Core AI source cannot become truth."
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip onnx.checker after publication staging (not recommended).",
    )
    parser.add_argument(
        "--full-check",
        action="store_true",
        help=(
            "Ask onnx.checker to run full shape inference. This can require substantial "
            "time/memory for the 24-block graph."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose PyTorch ONNX export logging.",
    )
    args = parser.parse_args()
    if args.opset < 18:
        parser.error("--opset must be at least 18 for scaled-dot-product attention export")
    if args.external_data_threshold < 0:
        parser.error("--external-data-threshold must be non-negative")
    if args.attention_query_chunk <= 0:
        parser.error("--attention-query-chunk must be positive")
    if args.attention_head_chunk <= 0:
        parser.error("--attention-head-chunk must be positive")
    if args.attention_head_padding < 0:
        parser.error("--attention-head-padding must be non-negative")
    if args.attention_head_padding > args.attention_head_chunk:
        parser.error("--attention-head-padding cannot exceed --attention-head-chunk")
    if args.qk_norm_padding_tokens <= 0:
        parser.error("--qk-norm-padding-tokens must be positive")
    if args.rms_norm_eps is not None and args.rms_norm_eps <= 0:
        parser.error("--rms-norm-eps must be positive")
    if args.attention_output_chunk <= 0:
        parser.error("--attention-output-chunk must be positive")
    if args.attention_output_reduction_chunk <= 0:
        parser.error("--attention-output-reduction-chunk must be positive")
    return args


def set_metadata(model_proto: Any, values: dict[str, str]) -> None:
    retained = {entry.key: entry.value for entry in model_proto.metadata_props}
    retained.update(values)
    del model_proto.metadata_props[:]
    for key, value in sorted(retained.items()):
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value


def materialize_attention_projection_inputs(
    onnx: Any,
    model_proto: Any,
    internal_precision: str,
    attention_output_chunk: int,
) -> int:
    """Insert an exact Add(0) barrier before each attention output projection.

    ONNX Runtime WebGPU 1.27 otherwise feeds the Reshape(Transpose(...)) value to
    the following large MatMul with an incorrect physical layout.  Add produces a
    densely materialized tensor.  The browser intentionally disables graph
    optimization for this graph so the correctness barrier remains observable.
    """

    element_type = (
        onnx.TensorProto.FLOAT16
        if internal_precision == "fp16"
        else onnx.TensorProto.FLOAT
    )
    original_nodes = list(model_proto.graph.node)
    del model_proto.graph.node[:]
    inserted = 0
    projection_nodes = 0
    dense_by_input: dict[str, str] = {}
    for node in original_nodes:
        if node.op_type in {"MatMul", "Gemm"} and "/attn/out/" in node.name:
            if len(node.input) < 2:
                raise ValueError(f"Attention projection {node.name!r} has too few inputs")
            original_input = node.input[0]
            dense_name = dense_by_input.get(original_input)
            if dense_name is None:
                zero_name = f"triposplat_attention_materialize_zero_{inserted:02d}"
                dense_name = f"{original_input}__triposplat_dense"
                model_proto.graph.initializer.append(
                    onnx.helper.make_tensor(zero_name, element_type, [], [0.0])
                )
                model_proto.graph.node.append(
                    onnx.helper.make_node(
                        "Add",
                        [original_input, zero_name],
                        [dense_name],
                        name=f"/triposplat_attention_materialize_{inserted:02d}/Add",
                    )
                )
                dense_by_input[original_input] = dense_name
                inserted += 1
            node.input[0] = dense_name
            projection_nodes += 1
        model_proto.graph.node.append(node)
    expected_projections = 28 * ((1024 + attention_output_chunk - 1) // attention_output_chunk)
    if projection_nodes != expected_projections or inserted != 28:
        raise RuntimeError(
            f"Found {projection_nodes} attention projection MatMuls and inserted {inserted} "
            f"barriers; expected {expected_projections} and 28"
        )
    return inserted


def rewrite_attention_projection_token_padding(onnx: Any, model_proto: Any) -> tuple[int, int]:
    """Shrink aligned trace-only padding to one WebGPU boundary-guard token.

    Tracing the full model on CPU with an 8193-column MatMul is prohibitively slow,
    so PyTorch traces one aligned 256-column zero tile.  ONNX MatMul is shape-generic:
    replacing those zeros with one column and changing the following Slice starts
    from 256 to 1 yields the desired runtime graph without touching weights or real
    values.  Static public inputs and outputs are unchanged.
    """

    padding_constants = 0
    slice_starts = 0
    for node in model_proto.graph.node:
        if node.op_type != "Constant" or "/attn/out/Constant" not in node.name:
            continue
        value_attribute = next(
            (attribute for attribute in node.attribute if attribute.name == "value"),
            None,
        )
        if value_attribute is None:
            continue
        tensor = value_attribute.t
        if list(tensor.dims) == [1, 1024, TRACE_ATTENTION_TOKEN_PADDING]:
            element_bytes = {
                onnx.TensorProto.FLOAT16: 2,
                onnx.TensorProto.FLOAT: 4,
            }.get(tensor.data_type)
            if element_bytes is None:
                raise TypeError(
                    f"Unexpected attention padding type {tensor.data_type} in {node.name!r}"
                )
            del tensor.dims[:]
            tensor.dims.extend([1, 1024, RUNTIME_ATTENTION_TOKEN_PADDING])
            tensor.ClearField("float_data")
            tensor.ClearField("int32_data")
            tensor.raw_data = bytes(element_bytes * 1024 * RUNTIME_ATTENTION_TOKEN_PADDING)
            padding_constants += 1
            continue
        if list(tensor.dims) != [1] or tensor.data_type != onnx.TensorProto.INT64:
            continue
        if tensor.int64_data:
            scalar = int(tensor.int64_data[0])
        elif len(tensor.raw_data) == 8:
            scalar = int.from_bytes(tensor.raw_data, "little", signed=True)
        else:
            continue
        if scalar != TRACE_ATTENTION_TOKEN_PADDING:
            continue
        tensor.ClearField("int64_data")
        tensor.raw_data = int(RUNTIME_ATTENTION_TOKEN_PADDING).to_bytes(
            8,
            "little",
            signed=True,
        )
        slice_starts += 1

    if padding_constants != 28 or slice_starts != 112:
        raise RuntimeError(
            "Rewrote "
            f"{padding_constants} attention padding constants and {slice_starts} Slice starts; "
            "expected 28 and 112"
        )
    return padding_constants, slice_starts


def replace_attention_projection_token_slices_with_gather(
    onnx: Any,
    model_proto: Any,
) -> int:
    """Replace offset Slice views with explicit dense token selection.

    ORT WebGPU can propagate the padding-removal Slice with a stale physical base
    address into its next operator.  Gather makes every real token address explicit
    and uses one shared 64 KiB int64 index initializer for all 112 projections.
    """

    token_lengths = {
        "noise_refiner": 8192,
        "context_refiner": 4101,
        "blocks": 8192 + 4101 + 1,
    }
    existing = {initializer.name for initializer in model_proto.graph.initializer}
    indices_by_kind: dict[str, str] = {}
    for kind, token_length in token_lengths.items():
        indices_name = f"triposplat_attention_real_token_indices_{token_length}"
        indices_by_kind[kind] = indices_name
        if indices_name not in existing:
            model_proto.graph.initializer.append(
                onnx.helper.make_tensor(
                    indices_name,
                    onnx.TensorProto.INT64,
                    [token_length],
                    list(
                        range(
                            RUNTIME_ATTENTION_TOKEN_PADDING,
                            token_length + RUNTIME_ATTENTION_TOKEN_PADDING,
                        )
                    ),
                )
            )

    constants: dict[str, int] = {}
    for node in model_proto.graph.node:
        if node.op_type != "Constant":
            continue
        attribute = next(
            (candidate for candidate in node.attribute if candidate.name == "value"),
            None,
        )
        if attribute is None:
            continue
        tensor = attribute.t
        if list(tensor.dims) != [1] or tensor.data_type != onnx.TensorProto.INT64:
            continue
        if tensor.int64_data:
            value = int(tensor.int64_data[0])
        elif len(tensor.raw_data) == 8:
            value = int.from_bytes(tensor.raw_data, "little", signed=True)
        else:
            continue
        if node.output:
            constants[node.output[0]] = value

    replaced = 0
    by_kind = {kind: 0 for kind in token_lengths}
    for node in model_proto.graph.node:
        if "/attn/out/Slice_" not in node.name:
            continue
        kind = next((candidate for candidate in token_lengths if f"/{candidate}." in node.name), None)
        if kind is None:
            continue
        if node.op_type == "Gather":
            node.input[1] = indices_by_kind[kind]
        elif node.op_type == "Slice" and len(node.input) >= 5:
            start = constants.get(node.input[1])
            axis = constants.get(node.input[3])
            if start != RUNTIME_ATTENTION_TOKEN_PADDING or axis != 2:
                continue
            data_input = node.input[0]
            del node.input[:]
            node.input.extend([data_input, indices_by_kind[kind]])
            del node.attribute[:]
            node.attribute.append(onnx.helper.make_attribute("axis", 2))
            node.op_type = "Gather"
        else:
            continue
        replaced += 1
        by_kind[kind] += 1

    if replaced != 112 or by_kind != {"noise_refiner": 8, "context_refiner": 8, "blocks": 96}:
        raise RuntimeError(
            f"Configured {replaced} attention padding-removal Gathers {by_kind}; "
            "expected 112 split as 8/8/96"
        )
    return replaced


def replace_zero_attention_padding_with_first_token(onnx: Any, model_proto: Any) -> int:
    """Use a real-valued sacrificial token instead of an all-zero guard.

    The affected ORT WebGPU MatMul computes an all-zero leading column correctly
    and corrupts the first nonzero column.  Duplicating token zero makes the first
    copy sacrificial; the later Gather retains the independently-computed second
    copy and all remaining tokens.
    """

    index_name = "triposplat_attention_first_token_index"
    if index_name not in {value.name for value in model_proto.graph.initializer}:
        model_proto.graph.initializer.append(
            onnx.helper.make_tensor(
                index_name,
                onnx.TensorProto.INT64,
                [1],
                [0],
            )
        )

    original_nodes = list(model_proto.graph.node)
    del model_proto.graph.node[:]
    replaced = 0
    for node in original_nodes:
        if node.op_type == "Concat" and node.name.endswith("/attn/out/Concat"):
            if len(node.input) != 2:
                raise RuntimeError(f"Unexpected attention padding Concat inputs: {node.name}")
            token_source = node.input[1]
            guard_output = f"{token_source}__triposplat_first_token_guard"
            model_proto.graph.node.append(
                onnx.helper.make_node(
                    "Gather",
                    [token_source, index_name],
                    [guard_output],
                    name=f"{node.name}_first_token_guard/Gather",
                    axis=2,
                )
            )
            node.input[0] = guard_output
            replaced += 1
        model_proto.graph.node.append(node)

    if replaced != 28:
        raise RuntimeError(
            f"Replaced {replaced} zero attention guards with token zero; expected 28"
        )
    return replaced


def _external_info(tensor: Any) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _safe_external_source(staged_graph: Path, tensor_name: str, location: str) -> Path:
    source = (staged_graph.parent / location).resolve()
    try:
        source.relative_to(staged_graph.parent.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Exporter external initializer {tensor_name!r} escapes staging directory: "
            f"{location!r}"
        ) from exc
    if not source.is_file():
        raise FileNotFoundError(
            f"Exporter external initializer {tensor_name!r} references missing {source}"
        )
    return source


def collect_external_slices(model_proto: Any, staged_graph: Path) -> list[ExternalSlice]:
    slices: list[ExternalSlice] = []
    by_source: dict[Path, list[ExternalSlice]] = {}
    for tensor in model_proto.graph.initializer:
        if not tensor.external_data:
            continue
        info = _external_info(tensor)
        location = info.get("location")
        if not location:
            raise ValueError(f"External initializer {tensor.name!r} has no location")
        source = _safe_external_source(staged_graph, tensor.name, location)
        offset = int(info.get("offset", "0"))
        length = int(info["length"]) if "length" in info else None
        if offset < 0 or (length is not None and length < 0):
            raise ValueError(
                f"External initializer {tensor.name!r} has invalid offset/length"
            )
        item = ExternalSlice(tensor=tensor, source=source, offset=offset, length=length)
        slices.append(item)
        by_source.setdefault(source, []).append(item)

    # Some exporters omit length when a tensor runs to the next offset or EOF.
    for source, source_slices in by_source.items():
        size = source.stat().st_size
        ordered = sorted(source_slices, key=lambda item: item.offset)
        for index, item in enumerate(ordered):
            if item.offset > size:
                raise ValueError(
                    f"External initializer {item.tensor.name!r} starts beyond {source}"
                )
            inferred_end = ordered[index + 1].offset if index + 1 < len(ordered) else size
            if item.length is None:
                item.length = inferred_end - item.offset
            assert item.length is not None
            if item.offset + item.length > size:
                raise ValueError(
                    f"External initializer {item.tensor.name!r} exceeds {source}"
                )
    return slices


def _set_external_reference(
    onnx: Any,
    tensor: Any,
    location: str,
    offset: int,
    length: int,
) -> None:
    del tensor.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = onnx.TensorProto.EXTERNAL


def _copy_range(source: BinaryIO, destination: BinaryIO, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        block = source.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise EOFError(f"Unexpected EOF while copying {length} external-data bytes")
        destination.write(block)
        remaining -= len(block)


def consolidate_external_data_streaming(
    onnx: Any,
    model_proto: Any,
    staged_graph: Path,
    output_data: Path,
    data_location: str,
    inline_threshold: int,
) -> tuple[int, int]:
    """Build one sidecar without materializing all model weights in memory."""

    external = collect_external_slices(model_proto, staged_graph)
    external_by_name = {item.tensor.name: item for item in external}
    if len(external_by_name) != len(external):
        raise ValueError("Exporter produced duplicate external initializer names")
    source_handles: dict[Path, BinaryIO] = {}
    externalized = 0
    total_bytes = 0
    try:
        with output_data.open("wb") as destination:
            for tensor in model_proto.graph.initializer:
                source_slice = external_by_name.get(tensor.name)
                raw = tensor.raw_data if tensor.HasField("raw_data") else None
                if source_slice is None and (raw is None or len(raw) < inline_threshold):
                    continue

                # A modest alignment is friendly to range readers and ORT mmap paths.
                padding = (-destination.tell()) % 64
                if padding:
                    destination.write(b"\0" * padding)
                offset = destination.tell()
                if source_slice is not None:
                    assert source_slice.length is not None
                    handle = source_handles.get(source_slice.source)
                    if handle is None:
                        handle = source_slice.source.open("rb")
                        source_handles[source_slice.source] = handle
                    _copy_range(
                        handle,
                        destination,
                        source_slice.offset,
                        source_slice.length,
                    )
                    length = source_slice.length
                else:
                    assert raw is not None
                    destination.write(raw)
                    length = len(raw)
                    tensor.ClearField("raw_data")
                _set_external_reference(
                    onnx,
                    tensor,
                    location=data_location,
                    offset=offset,
                    length=length,
                )
                externalized += 1
                total_bytes += length
    finally:
        for handle in source_handles.values():
            handle.close()
    if externalized == 0 or output_data.stat().st_size == 0:
        raise RuntimeError("DiT export produced no external initializers/sidecar payload")
    return externalized, total_bytes


def publish_graph(
    onnx: Any,
    staged_graph: Path,
    output: Path,
    metadata: dict[str, str],
    external_threshold: int,
    run_checker: bool,
    full_check: bool,
) -> tuple[list[Path], dict[str, int]]:
    """Validate, stream-consolidate, and atomically publish graph plus sidecar."""

    output.parent.mkdir(parents=True, exist_ok=True)
    data_name = output.name + ".data"
    final_data = output.parent / data_name
    model_proto = onnx.load_model(str(staged_graph), load_external_data=False)
    if metadata.get("triposplat.attention_output_kernel") == "weight_left_matmul":
        padding_constants, padding_slice_starts = rewrite_attention_projection_token_padding(
            onnx,
            model_proto,
        )
        padding_gathers = replace_attention_projection_token_slices_with_gather(
            onnx,
            model_proto,
        )
        real_token_guards = replace_zero_attention_padding_with_first_token(
            onnx,
            model_proto,
        )
        metadata["triposplat.webgpu_attention_token_padding"] = str(
            RUNTIME_ATTENTION_TOKEN_PADDING
        )
        metadata["triposplat.webgpu_attention_padding_constants"] = str(padding_constants)
        metadata["triposplat.webgpu_attention_padding_slice_starts"] = str(
            padding_slice_starts
        )
        metadata["triposplat.webgpu_attention_padding_gathers"] = str(padding_gathers)
        metadata["triposplat.webgpu_attention_real_token_guards"] = str(real_token_guards)
    set_metadata(model_proto, metadata)
    verify_onnx_contract(onnx, model_proto, require_metadata=True)

    with tempfile.TemporaryDirectory(
        dir=str(output.parent),
        prefix=".triposplat-dit-publish-",
    ) as publish_dir_string:
        publish_dir = Path(publish_dir_string)
        publish_graph_path = publish_dir / output.name
        publish_data_path = publish_dir / data_name
        count, payload_bytes = consolidate_external_data_streaming(
            onnx=onnx,
            model_proto=model_proto,
            staged_graph=staged_graph,
            output_data=publish_data_path,
            data_location=data_name,
            inline_threshold=external_threshold,
        )
        # Tensors are already external references; do not ask ONNX to materialize or
        # reconvert them while serializing the now-small graph protobuf.
        onnx.save_model(model_proto, str(publish_graph_path), save_as_external_data=False)
        published_proto = onnx.load_model(
            str(publish_graph_path),
            load_external_data=False,
        )
        verify_onnx_contract(onnx, published_proto, require_metadata=True)
        external_files = verify_external_data_files(publish_graph_path, published_proto)
        if external_files != [publish_data_path.resolve()]:
            raise RuntimeError(
                f"Published graph references {external_files}, expected only {publish_data_path}"
            )
        if run_checker:
            # Passing a path supports external data and avoids protobuf's 2 GiB API limit.
            onnx.checker.check_model(str(publish_graph_path), full_check=full_check)

        # Publish the sidecar first so the visible graph never points at a missing file.
        os.replace(publish_data_path, final_data)
        os.replace(publish_graph_path, output)

    artifacts = [output, final_data]
    for artifact in artifacts:
        artifact.chmod(artifact.stat().st_mode | stat.S_IRGRP | stat.S_IROTH)
    return artifacts, {"initializers": count, "payload_bytes": payload_bytes}


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
    commit, dirty = source_revision(repo)
    if dirty and not args.allow_dirty_official_source:
        raise RuntimeError(
            f"Official model.py/triposplat.py in {repo} have tracked edits. Use a clean "
            "checkout; --allow-dirty-official-source is available only for deliberate audits."
        )
    device = choose_torch_device(torch, args.device)
    print(
        f"Loading official load_flow_model at {commit} "
        f"({args.internal_precision}, {device})"
    )
    flow_model, source = load_official_flow_model(
        torch=torch,
        triposplat_repo=repo,
        weights=weights,
        device=device,
        internal_precision=args.internal_precision,
        low_memory_construction=args.low_memory_construction,
    )

    primitive_gate = validate_real_rope_primitives(
        torch,
        flow_model,
        source.model_module,
    )
    print("RoPE primitive parity: " + json.dumps(primitive_gate, sort_keys=True))
    if not primitive_gate["passed"]:
        raise RuntimeError(
            "Real RoPE primitive parity failed; refusing to export a graph that drifts "
            "from official complex RoPE"
        )
    adapter = adapt_official_flow_for_onnx(
        torch,
        flow_model,
        source.model_module,
        attention_query_chunk=args.attention_query_chunk,
        attention_head_chunk=args.attention_head_chunk,
        attention_head_padding=args.attention_head_padding,
        qk_norm_padding_tokens=args.qk_norm_padding_tokens,
        rms_norm_eps=args.rms_norm_eps,
        attention_output_chunk=args.attention_output_chunk,
        attention_output_reduction_chunk=args.attention_output_reduction_chunk,
    )
    graph = make_browser_flow_step(torch, flow_model, args.internal_precision)

    latent = torch.zeros(LATENT_SHAPE, dtype=torch.float32, device=device)
    camera = torch.zeros(CAMERA_SHAPE, dtype=torch.float32, device=device)
    timestep = torch.full(TIMESTEP_SHAPE, 1000.0, dtype=torch.float32, device=device)
    feature1 = torch.zeros(FEATURE1_SHAPE, dtype=torch.float32, device=device)
    feature2 = torch.zeros(FEATURE2_SHAPE, dtype=torch.float32, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output.parent),
        prefix=".triposplat-dit-export-",
    ) as export_dir_string:
        staged_graph = Path(export_dir_string) / output.name
        export_kwargs: dict[str, Any] = {
            "export_params": True,
            "do_constant_folding": True,
            "opset_version": args.opset,
            "input_names": list(INPUT_NAMES),
            "output_names": list(OUTPUT_NAMES),
            "verbose": args.verbose,
        }
        signature = inspect.signature(torch.onnx.export)
        if "dynamo" in signature.parameters:
            # Legacy tracing has stable fixed-shape SDPA lowering and avoids adding
            # onnxscript to the reproducible export environment.
            export_kwargs["dynamo"] = False
        if "external_data" in signature.parameters:
            export_kwargs["external_data"] = True

        print(
            "Tracing fixed graph with float32 public I/O and "
            f"{args.internal_precision} internal compute"
        )
        with torch.inference_mode():
            torch.onnx.export(
                graph,
                (latent, camera, timestep, feature1, feature2),
                str(staged_graph),
                **export_kwargs,
            )

        metadata = {
            "triposplat.component": "flow_dit_one_step",
            "triposplat.source_repository": OFFICIAL_REPOSITORY_URL,
            "triposplat.source_commit": commit,
            "triposplat.source_files_dirty": str(bool(dirty)).lower() if dirty is not None else "unknown",
            "triposplat.source_loader": "official triposplat.load_flow_model",
            INTERNAL_PRECISION_METADATA_KEY: args.internal_precision,
            "triposplat.public_io": PUBLIC_IO_METADATA_VALUE,
            "triposplat.timestep": "float32 [1], already scaled as 1000 * normalized_t",
            "triposplat.contract": (
                "latent[1,8192,16],camera[1,1,5],t[1],"
                "feature1[1,4101,1280],feature2[1,4101,128] -> "
                "pred_latent[1,8192,16],pred_camera[1,1,5]; all float32"
            ),
            "triposplat.rope_export": "algebraically exact real pair rotation",
            "triposplat.rope_modules": str(adapter.real_rope_modules),
            "triposplat.attention_export": "exact independent head- and query-axis chunks",
            "triposplat.attention_query_chunk": str(adapter.attention_query_chunk),
            "triposplat.attention_head_chunk": str(adapter.attention_head_chunk),
            "triposplat.attention_head_padding": str(adapter.attention_head_padding),
            "triposplat.qk_norm_padding_tokens": str(adapter.qk_norm_padding_tokens),
            "triposplat.qk_norm_modules": str(adapter.qk_norm_modules),
            "triposplat.rms_norm_export": (
                "official F.normalize"
                if adapter.rms_norm_eps is None
                else "x*rsqrt(mean(x^2)+eps)*gamma"
            ),
            "triposplat.rms_norm_eps": (
                "disabled" if adapter.rms_norm_eps is None else repr(adapter.rms_norm_eps)
            ),
            "triposplat.stable_rms_norm_modules": str(adapter.stable_rms_norm_modules),
            "triposplat.attention_output_chunk": str(adapter.attention_output_chunk),
            "triposplat.attention_output_kernel": "conv2d_1x1",
            "triposplat.attention_output_reduction_chunk": str(
                adapter.attention_output_reduction_chunk
            ),
            "triposplat.attention_output_modules": str(adapter.attention_output_modules),
            "triposplat.attention_compute": "float32 scores, softmax, and value accumulation; cast output to model dtype",
            "triposplat.rope_primitive_gate": json.dumps(primitive_gate, sort_keys=True),
            "triposplat.static_position_shape": json.dumps(adapter.static_position_shape),
            "triposplat.static_position_dtype": adapter.static_position_dtype,
            "triposplat.static_position_sha256": adapter.static_position_sha256,
            "triposplat.validation": (
                "Run validate_dit_onnx.py; it gates untouched official complex PyTorch, "
                "adapted real PyTorch, and ONNX Runtime on identical inputs"
            ),
        }
        artifacts, external_stats = publish_graph(
            onnx=onnx,
            staged_graph=staged_graph,
            output=output,
            metadata=metadata,
            external_threshold=args.external_data_threshold,
            run_checker=not args.skip_check,
            full_check=args.full_check,
        )

    print(
        "Fixed public contract: all float32; "
        "latent[1,8192,16] + camera[1,1,5] + t[1] + "
        "feature1[1,4101,1280] + feature2[1,4101,128]"
    )
    print(
        f"Externalized {external_stats['initializers']} initializers "
        f"({external_stats['payload_bytes']:,} payload bytes)"
    )
    for artifact in artifacts:
        print(
            f"Wrote {artifact} ({artifact.stat().st_size:,} bytes, "
            f"sha256={sha256_file(artifact)})"
        )
    return artifacts


def main() -> None:
    export_graph(parse_args())


if __name__ == "__main__":
    main()
