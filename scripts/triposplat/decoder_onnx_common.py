"""Shared ONNX export and parity helpers for the official TripoSplat decoders.

This module deliberately does not contain an octree sampler or Gaussian activation
logic.  It imports ``load_decoder`` from a caller-provided checkout of the official
TripoSplat repository and only exposes the two neural graph boundaries used by the
browser port:

* occupancy: ``x, l, cond -> logits``
* Gaussian decoder: ``points, cond -> features``

The public ONNX boundary is always float32.  ``internal_precision=fp16`` keeps model
parameters and the expensive transformer math in float16; the thin wrappers cast
their output back to float32.  This makes the browser contract explicit while still
allowing fp16 WebGPU execution internally.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


OFFICIAL_REPOSITORY_URL = "https://github.com/VAST-AI-Research/TripoSplat"

BATCH_SIZE = 1
TOKEN_COUNT = 8192
POINT_CHANNELS = 3
COND_CHANNELS = 16
OCCUPANCY_CHANNELS = 8
GAUSSIAN_FEATURE_CHANNELS = 480

POINTS_SHAPE = (BATCH_SIZE, TOKEN_COUNT, POINT_CHANNELS)
LEVEL_SHAPE = (BATCH_SIZE,)
COND_SHAPE = (BATCH_SIZE, TOKEN_COUNT, COND_CHANNELS)
LOGITS_SHAPE = (BATCH_SIZE, TOKEN_COUNT, OCCUPANCY_CHANNELS)
FEATURES_SHAPE = (BATCH_SIZE, TOKEN_COUNT, GAUSSIAN_FEATURE_CHANNELS)

# The official sampler calls the occupancy model with the current octree
# resolution (2**level), not the zero-based level index.
VALID_OCTREE_RESOLUTIONS = tuple(1 << level for level in range(1, 9))

COMPONENT_CONTRACTS: dict[str, dict[str, tuple[int, ...]]] = {
    "octree_occupancy_decoder": {
        "x": POINTS_SHAPE,
        "l": LEVEL_SHAPE,
        "cond": COND_SHAPE,
        "logits": LOGITS_SHAPE,
    },
    "gaussian_decoder": {
        "points": POINTS_SHAPE,
        "cond": COND_SHAPE,
        "features": FEATURES_SHAPE,
    },
}


def resolved_file(path: Path, description: str) -> Path:
    """Resolve *path* and reject missing or non-file inputs."""

    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {result}")
    return result


def source_commit(repo: Path) -> str:
    """Return the checked-out official source commit when Git is available."""

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
    """Return the SHA-256 digest of one published artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _module_file(module: ModuleType) -> Path | None:
    value = getattr(module, "__file__", None)
    return Path(value).resolve() if value else None


def import_official_triposplat(triposplat_repo: Path) -> ModuleType:
    """Import ``triposplat.py`` and its sibling ``model.py`` from an exact checkout.

    The upstream module uses ``from model import ...`` rather than a package-relative
    import.  We therefore load the sibling under that expected name, but refuse to
    silently reuse an unrelated pre-existing module called ``model``.
    """

    repo = triposplat_repo.expanduser().resolve()
    model_path = repo / "model.py"
    pipeline_path = repo / "triposplat.py"
    if not model_path.is_file() or not pipeline_path.is_file():
        raise FileNotFoundError(
            f"{repo} must contain model.py and triposplat.py from "
            f"{OFFICIAL_REPOSITORY_URL}"
        )

    cache_key = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()[:16]
    pipeline_module_name = f"_triposplat_official_{cache_key}"
    cached = sys.modules.get(pipeline_module_name)
    if cached is not None:
        if _module_file(cached) != pipeline_path:
            raise ImportError(
                f"Cached {pipeline_module_name} points at {_module_file(cached)}, "
                f"not {pipeline_path}"
            )
        return cached

    existing_model = sys.modules.get("model")
    if existing_model is not None and _module_file(existing_model) != model_path:
        raise ImportError(
            "Cannot import the official TripoSplat source because sys.modules['model'] "
            f"already points at {_module_file(existing_model)}. Run the exporter in a "
            "fresh Python process."
        )

    inserted_repo_path = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        inserted_repo_path = True
    try:
        if existing_model is None:
            model_spec = importlib.util.spec_from_file_location("model", model_path)
            if model_spec is None or model_spec.loader is None:
                raise ImportError(f"Unable to create an import specification for {model_path}")
            official_model = importlib.util.module_from_spec(model_spec)
            sys.modules["model"] = official_model
            try:
                model_spec.loader.exec_module(official_model)
            except Exception:
                sys.modules.pop("model", None)
                raise

        pipeline_spec = importlib.util.spec_from_file_location(
            pipeline_module_name,
            pipeline_path,
        )
        if pipeline_spec is None or pipeline_spec.loader is None:
            raise ImportError(
                f"Unable to create an import specification for {pipeline_path}"
            )
        pipeline_module = importlib.util.module_from_spec(pipeline_spec)
        sys.modules[pipeline_module_name] = pipeline_module
        try:
            pipeline_spec.loader.exec_module(pipeline_module)
        except Exception:
            sys.modules.pop(pipeline_module_name, None)
            raise
    finally:
        if inserted_repo_path:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass

    if not callable(getattr(pipeline_module, "load_decoder", None)):
        raise AttributeError(
            f"{pipeline_path} does not expose the official load_decoder function"
        )
    return pipeline_module


def choose_torch_device(torch: Any, requested: str) -> Any:
    """Resolve ``auto`` and reject unavailable explicit accelerator requests."""

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch reports CUDA unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("--device mps was requested, but PyTorch reports MPS unavailable")
    return torch.device(requested)


def synchronize_torch(torch: Any, device: Any) -> None:
    """Synchronize accelerator work before recording a measured duration."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_official_decoder(
    torch: Any,
    triposplat_repo: Path,
    weights: Path,
    device: Any,
    internal_precision: str,
) -> Any:
    """Load the checkpoint only through upstream ``triposplat.load_decoder``."""

    if internal_precision not in {"fp16", "fp32"}:
        raise ValueError(
            f"Unsupported internal precision {internal_precision!r}; expected fp16 or fp32"
        )
    checkpoint = resolved_file(weights, "TripoSplat decoder safetensors checkpoint")
    official = import_official_triposplat(triposplat_repo)
    dtype = torch.float16 if internal_precision == "fp16" else torch.float32
    decoder = official.load_decoder(str(checkpoint), device=device, dtype=dtype)
    if getattr(decoder.gs, "out_channels", None) != GAUSSIAN_FEATURE_CHANNELS:
        raise RuntimeError(
            "Official Gaussian decoder reports out_channels="
            f"{getattr(decoder.gs, 'out_channels', None)!r}; expected "
            f"{GAUSSIAN_FEATURE_CHANNELS}. The source/checkpoint contract changed."
        )
    if getattr(decoder, "gaussians_per_point", None) != 32:
        raise RuntimeError(
            "Official decoder reports gaussians_per_point="
            f"{getattr(decoder, 'gaussians_per_point', None)!r}; expected 32"
        )
    return decoder.eval()


def _query_chunked_attention(
    qkv: Any = None,
    q: Any = None,
    k: Any = None,
    v: Any = None,
    kv: Any = None,
    *,
    query_chunk_size: int,
) -> Any:
    """Exact query-row partition of official attention with explicit fp32 scores.

    Query rows are independent once keys and values are fixed. Splitting only that
    axis avoids materializing an 8192x8192 score matrix while preserving the
    official fused-fp16 SDPA's higher-precision score and softmax arithmetic.
    """

    import torch
    import torch.nn.functional as functional

    if qkv is not None:
        q, k, v = qkv.unbind(dim=2)
    elif kv is not None:
        k, v = kv.unbind(dim=2)
    if q is None or k is None or v is None:
        raise ValueError("Chunked decoder attention requires q, k, and v")
    output_dtype = q.dtype
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3).float()
    v = v.permute(0, 2, 1, 3).float()
    chunks = [
        functional.scaled_dot_product_attention(
            q[:, :, start : start + query_chunk_size, :].float(),
            k,
            v,
        ).to(dtype=output_dtype)
        for start in range(0, q.shape[2], query_chunk_size)
    ]
    return torch.cat(chunks, dim=2).permute(0, 2, 1, 3)


def adapt_official_decoder_for_onnx(
    torch: Any,
    decoder_component: Any,
    *,
    query_chunk_size: int = 256,
    qk_norm_padding_tokens: int = 1,
) -> dict[str, int]:
    """Apply export-only, dependency-preserving decoder attention adaptations."""

    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if qk_norm_padding_tokens <= 0:
        raise ValueError("qk_norm_padding_tokens must be positive")
    model_module = sys.modules.get(decoder_component.__class__.__module__)
    if model_module is None:
        raise RuntimeError("Official decoder model module is not loaded")

    model_module.scaled_dot_product_attention = (
        lambda qkv=None, q=None, k=None, v=None, kv=None: _query_chunked_attention(
            qkv=qkv,
            q=q,
            k=k,
            v=v,
            kv=kv,
            query_chunk_size=query_chunk_size,
        )
    )

    class ChunkedAttentionOutputProjection(torch.nn.Module):
        """Bound the token-wise 1024x1024 reduction and protect its first lane."""

        def __init__(self, projection: Any, output_chunk: int = 256) -> None:
            super().__init__()
            self.projection = projection
            self.output_chunk = output_chunk

        def forward(self, value: Any) -> Any:
            import torch.nn.functional as functional

            outputs = []
            output_channels = int(self.projection.weight.shape[0])
            original_shape = value.shape
            token_count = int(value.shape[-2])
            first_token = value[..., :1, :].float().unsqueeze(-2)
            tail_indices = torch.arange(1, token_count, dtype=torch.int64, device=value.device)
            channels_first = value.transpose(-1, -2).unsqueeze(-1)
            for start in range(0, output_channels, self.output_chunk):
                end = min(start + self.output_chunk, output_channels)
                bias = None if self.projection.bias is None else self.projection.bias[start:end]
                projected = functional.conv2d(
                    channels_first,
                    self.projection.weight[start:end].unsqueeze(-1).unsqueeze(-1),
                    bias,
                )
                bulk_tail = torch.index_select(
                    projected.squeeze(-1).transpose(-1, -2),
                    -2,
                    tail_indices,
                )
                products = (
                    first_token
                    * self.projection.weight[start:end].float().unsqueeze(0).unsqueeze(0)
                ).transpose(-1, -2)
                rows = int(products.shape[-2])
                if rows & (rows - 1):
                    raise ValueError("Attention projection width must be a power of two")
                while rows > 256:
                    half = rows // 2
                    products = products[..., :half, :] + products[..., half:rows, :]
                    rows = half
                while rows > 1:
                    half = rows // 2
                    padding = torch.zeros(256 - half, dtype=torch.int64, device=value.device)
                    left = torch.cat((torch.arange(half, device=value.device), padding))
                    right = torch.cat((torch.arange(half, rows, device=value.device), padding))
                    active = torch.cat((
                        torch.ones(half, dtype=torch.float32, device=value.device),
                        torch.zeros(256 - half, dtype=torch.float32, device=value.device),
                    )).reshape(1, 1, 256, 1)
                    products = (
                        torch.index_select(products, -2, left) * active
                        + torch.index_select(products, -2, right) * active
                    )
                    rows = half
                special = products[..., 0, :]
                if bias is not None:
                    special = special + bias.float().reshape(1, 1, -1)
                outputs.append(torch.cat((special.to(value.dtype), bulk_tail), dim=-2))
            return torch.cat(outputs, dim=-1).reshape(*original_shape[:-1], output_channels)

    class PaddedQkRmsNorm(torch.nn.Module):
        """Keep every real vector away from a runtime's final reduction lane."""

        def __init__(self, norm: Any) -> None:
            super().__init__()
            self.norm = norm

        def forward(self, value: Any) -> Any:
            token_count = value.shape[1]
            padded = torch.cat(
                (value, value[:, :qk_norm_padding_tokens, ...]),
                dim=1,
            )
            return self.norm(padded)[:, :token_count, ...]

    attention_type = model_module.MultiHeadAttention
    attention_modules = 0
    qk_norm_modules = 0
    output_projection_modules = 0
    for submodule in decoder_component.modules():
        if not isinstance(submodule, attention_type):
            continue
        attention_modules += 1
        if getattr(submodule, "qk_rms_norm", False):
            if submodule._type == "self":
                submodule.q_rms_norm = PaddedQkRmsNorm(submodule.q_rms_norm)
                submodule.k_rms_norm = PaddedQkRmsNorm(submodule.k_rms_norm)
            else:
                submodule.q_rms_norm = PaddedQkRmsNorm(submodule.q_rms_norm)
                submodule.k_rms_norm = PaddedQkRmsNorm(submodule.k_rms_norm)
            qk_norm_modules += 2
        submodule.to_out = ChunkedAttentionOutputProjection(submodule.to_out)
        output_projection_modules += 1
    if attention_modules <= 0 or qk_norm_modules != attention_modules * 2:
        raise RuntimeError(
            "Unexpected decoder attention contract: "
            f"attention={attention_modules}, padded_qk={qk_norm_modules}"
        )
    return {
        "attention_query_chunk": query_chunk_size,
        "attention_modules": attention_modules,
        "qk_norm_padding_tokens": qk_norm_padding_tokens,
        "qk_norm_modules": qk_norm_modules,
        "attention_output_modules": output_projection_modules,
    }


def make_octree_logits_graph(torch: Any, decoder: Any) -> Any:
    """Wrap only ``decoder.octree`` and expose its unactivated logits."""

    class OctreeLogitsGraph(torch.nn.Module):
        def __init__(self, octree: Any) -> None:
            super().__init__()
            self.octree = octree

        def forward(self, x: Any, l: Any, cond: Any) -> Any:
            # Upstream immediately converts x/cond to its parameter dtype and l to
            # float32 in LevelEmbedder.  Float32 l avoids an int64 WebGPU boundary.
            return self.octree(x, l, cond)["logits"].to(torch.float32)

    return OctreeLogitsGraph(decoder.octree).eval()


def make_gaussian_features_graph(torch: Any, decoder: Any) -> Any:
    """Wrap only ``decoder.gs`` and expose the official raw feature tensor."""

    class GaussianFeaturesGraph(torch.nn.Module):
        def __init__(self, gaussian_decoder: Any) -> None:
            super().__init__()
            self.gaussian_decoder = gaussian_decoder

        def forward(self, points: Any, cond: Any) -> Any:
            return self.gaussian_decoder(
                x={"points": points},
                cond=cond,
            )["features"].to(torch.float32)

    return GaussianFeaturesGraph(decoder.gs).eval()


def make_dummy_inputs(torch: Any, component: str, device: Any) -> tuple[Any, ...]:
    """Create fixed-shape float32 tracing inputs for one component."""

    points = torch.zeros(POINTS_SHAPE, dtype=torch.float32, device=device)
    cond = torch.zeros(COND_SHAPE, dtype=torch.float32, device=device)
    if component == "octree_occupancy_decoder":
        level = torch.full(LEVEL_SHAPE, 256.0, dtype=torch.float32, device=device)
        return points, level, cond
    if component == "gaussian_decoder":
        return points, cond
    raise ValueError(f"Unknown decoder component {component!r}")


def fixed_value_shape(value_info: Any) -> tuple[int | None, ...]:
    """Read fixed integer axes from an ONNX ValueInfoProto."""

    return tuple(
        int(dimension.dim_value) if dimension.HasField("dim_value") else None
        for dimension in value_info.type.tensor_type.shape.dim
    )


def set_metadata(model_proto: Any, values: dict[str, str]) -> None:
    """Merge stable export metadata without retaining duplicate keys."""

    merged = {entry.key: entry.value for entry in model_proto.metadata_props}
    merged.update(values)
    del model_proto.metadata_props[:]
    for key, value in sorted(merged.items()):
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value


def metadata_dict(model_proto: Any) -> dict[str, str]:
    """Return ONNX metadata as a normal mapping."""

    return {entry.key: entry.value for entry in model_proto.metadata_props}


def require_fixed_onnx_contract(
    onnx: Any,
    model_proto: Any,
    component: str,
    expected_internal_precision: str | None = None,
) -> dict[str, str]:
    """Reject names, dtypes, shapes, or metadata that drift from the browser ABI."""

    try:
        contract = COMPONENT_CONTRACTS[component]
    except KeyError as exc:
        raise ValueError(f"Unknown decoder component {component!r}") from exc

    output_name = "logits" if component == "octree_occupancy_decoder" else "features"
    input_names = set(contract) - {output_name}
    inputs = {value.name: value for value in model_proto.graph.input}
    outputs = {value.name: value for value in model_proto.graph.output}
    if set(inputs) != input_names:
        raise RuntimeError(
            f"Exported {component} inputs are {sorted(inputs)}, expected {sorted(input_names)}"
        )
    if set(outputs) != {output_name}:
        raise RuntimeError(
            f"Exported {component} outputs are {sorted(outputs)}, expected [{output_name!r}]"
        )

    for name, expected_shape in contract.items():
        value = inputs[name] if name in inputs else outputs[name]
        shape = fixed_value_shape(value)
        dtype = value.type.tensor_type.elem_type
        if shape != expected_shape:
            raise RuntimeError(
                f"Exported {name} shape is {shape}; expected fixed shape {expected_shape}"
            )
        if dtype != onnx.TensorProto.FLOAT:
            raise RuntimeError(
                f"Exported {name} element type is {dtype}; public decoder I/O must be float32"
            )

    metadata = metadata_dict(model_proto)
    if metadata.get("triposplat.component") != component:
        raise RuntimeError(
            "ONNX metadata triposplat.component is "
            f"{metadata.get('triposplat.component')!r}; expected {component!r}"
        )
    if metadata.get("triposplat.public_io_precision") != "float32":
        raise RuntimeError(
            "ONNX metadata must contain triposplat.public_io_precision=float32"
        )
    precision = metadata.get("triposplat.internal_precision")
    if precision not in {"fp16", "fp32"}:
        raise RuntimeError(
            "ONNX metadata must contain triposplat.internal_precision=fp16 or fp32"
        )
    if expected_internal_precision is not None and precision != expected_internal_precision:
        raise RuntimeError(
            f"ONNX internal precision is {precision}; expected {expected_internal_precision}"
        )
    initializer_types = {tensor.data_type for tensor in model_proto.graph.initializer}
    expected_initializer_type = (
        onnx.TensorProto.FLOAT16 if precision == "fp16" else onnx.TensorProto.FLOAT
    )
    if expected_initializer_type not in initializer_types:
        raise RuntimeError(
            f"Graph metadata declares {precision} internal precision, but no initializer "
            f"uses ONNX element type {expected_initializer_type}"
        )
    if precision == "fp32" and onnx.TensorProto.FLOAT16 in initializer_types:
        raise RuntimeError(
            "Graph metadata declares fp32 internal precision, but float16 initializers remain"
        )
    return metadata


def _uses_external_data(onnx: Any, model_proto: Any) -> bool:
    return any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in model_proto.graph.initializer
    )


def external_data_locations(onnx: Any, model_proto: Any) -> set[str]:
    """Collect sidecar locations without loading the external tensor bytes."""

    locations: set[str] = set()
    for tensor in model_proto.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        values = {entry.key: entry.value for entry in tensor.external_data}
        location = values.get("location")
        if not location:
            raise RuntimeError(f"External initializer {tensor.name!r} has no location")
        locations.add(location)
    return locations


def require_consolidated_external_data(
    onnx: Any,
    graph_path: Path,
    model_proto: Any,
) -> list[Path]:
    """Ensure external tensors resolve through at most one same-directory sidecar."""

    locations = external_data_locations(onnx, model_proto)
    if len(locations) > 1:
        raise RuntimeError(
            f"ONNX external weights are sharded across {sorted(locations)}; expected one file"
        )
    artifacts = [graph_path]
    for location in locations:
        location_path = Path(location)
        if location_path.is_absolute() or len(location_path.parts) != 1:
            raise RuntimeError(
                f"External-data location {location!r} must be a same-directory basename"
            )
        sidecar = graph_path.parent / location_path
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"ONNX graph references missing external-data sidecar: {sidecar}"
            )
        artifacts.append(sidecar)
    return artifacts


def consolidate_and_publish(
    onnx: Any,
    staged_export: Path,
    output_path: Path,
    component: str,
    internal_precision: str,
    external_data_threshold: int,
    metadata: dict[str, str],
    run_checker: bool,
) -> list[Path]:
    """Normalize exporter shards into one sidecar, check, and atomically publish."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_filename = output_path.name + ".data"
    final_data_path = output_path.parent / data_filename

    # Materialize any shards emitted by torch.onnx before choosing one deterministic
    # output location.  Decoder graphs remain below protobuf's 2 GiB limit, but their
    # weights are intentionally external for independent CDN caching and range fetches.
    model_proto = onnx.load_model(str(staged_export), load_external_data=True)
    if _uses_external_data(onnx, model_proto):
        onnx.external_data_helper.convert_model_from_external_data(model_proto)
    set_metadata(model_proto, metadata)
    require_fixed_onnx_contract(
        onnx,
        model_proto,
        component,
        expected_internal_precision=internal_precision,
    )

    with tempfile.TemporaryDirectory(
        dir=str(output_path.parent),
        prefix=f".{component}-publish-",
    ) as publish_dir_string:
        publish_dir = Path(publish_dir_string)
        publish_graph = publish_dir / output_path.name
        onnx.save_model(
            model_proto,
            str(publish_graph),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=external_data_threshold,
            convert_attribute=False,
        )
        publish_data = publish_dir / data_filename
        if not publish_data.is_file():
            raise RuntimeError(
                f"ONNX did not create the requested consolidated sidecar {publish_data}"
            )

        published_proto = onnx.load_model(str(publish_graph), load_external_data=False)
        require_fixed_onnx_contract(
            onnx,
            published_proto,
            component,
            expected_internal_precision=internal_precision,
        )
        require_consolidated_external_data(onnx, publish_graph, published_proto)
        if run_checker:
            # Path-based checking also verifies external-data offsets and lengths.
            onnx.checker.check_model(str(publish_graph), full_check=True)

        # Publish the sidecar before the graph so a newly visible graph never points
        # at a sidecar that has not been installed yet.
        os.replace(publish_data, final_data_path)
        os.replace(publish_graph, output_path)

    return [output_path, final_data_path]


def export_fixed_decoder_graph(
    *,
    torch: Any,
    onnx: Any,
    graph: Any,
    dummy_inputs: tuple[Any, ...],
    input_names: Sequence[str],
    output_name: str,
    output_path: Path,
    component: str,
    internal_precision: str,
    opset: int,
    external_data_threshold: int,
    metadata: dict[str, str],
    verbose: bool,
    run_checker: bool,
) -> list[Path]:
    """Trace, consolidate, and publish one fixed decoder graph."""

    import inspect

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(output_path.parent),
        prefix=f".{component}-export-",
    ) as export_dir_string:
        staged_export = Path(export_dir_string) / output_path.name
        kwargs: dict[str, Any] = {
            "export_params": True,
            "do_constant_folding": True,
            "opset_version": opset,
            "input_names": list(input_names),
            "output_names": [output_name],
            "verbose": verbose,
        }
        export_signature = inspect.signature(torch.onnx.export)
        if "dynamo" in export_signature.parameters:
            # The fixed-shape legacy lowering has stable SDPA decomposition and does
            # not add onnxscript as another required export dependency.
            kwargs["dynamo"] = False
        if "external_data" in export_signature.parameters:
            kwargs["external_data"] = True

        with torch.inference_mode():
            torch.onnx.export(
                graph,
                dummy_inputs,
                str(staged_export),
                **kwargs,
            )

        return consolidate_and_publish(
            onnx=onnx,
            staged_export=staged_export,
            output_path=output_path,
            component=component,
            internal_precision=internal_precision,
            external_data_threshold=external_data_threshold,
            metadata=metadata,
            run_checker=run_checker,
        )


def load_graph_contract(
    onnx: Any,
    graph_path: Path,
    component: str,
    requested_precision: str,
) -> tuple[dict[str, str], list[Path], str]:
    """Inspect a graph without materializing its external parameter tensors."""

    graph = resolved_file(graph_path, "ONNX graph")
    model_proto = onnx.load_model(str(graph), load_external_data=False)
    metadata = require_fixed_onnx_contract(onnx, model_proto, component)
    artifacts = require_consolidated_external_data(onnx, graph, model_proto)
    precision = metadata["triposplat.internal_precision"]
    if requested_precision != "auto" and requested_precision != precision:
        raise ValueError(
            f"--precision {requested_precision} conflicts with graph metadata ({precision})"
        )
    return metadata, artifacts, precision


def deterministic_occupancy_inputs(seed: int) -> tuple[Any, Any, Any]:
    """Create representative fixed-shape occupancy inputs in float32."""

    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.random(POINTS_SHAPE, dtype=np.float32)
    l = np.asarray([256.0], dtype=np.float32)
    cond = rng.standard_normal(COND_SHAPE, dtype=np.float32)
    return x, l, cond


def deterministic_gaussian_inputs(seed: int) -> tuple[Any, Any]:
    """Create representative fixed-shape Gaussian decoder inputs in float32."""

    import numpy as np

    rng = np.random.default_rng(seed)
    points = rng.random(POINTS_SHAPE, dtype=np.float32)
    cond = rng.standard_normal(COND_SHAPE, dtype=np.float32)
    return points, cond


def load_occupancy_fixture(path: Path) -> tuple[Any, Any, Any, Any | None]:
    """Load strict occupancy inputs and an optional saved PyTorch reference."""

    import numpy as np

    fixture = resolved_file(path, "Occupancy fixture")
    with np.load(fixture, allow_pickle=False) as archive:
        required = {"x", "l", "cond"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(
                f"{fixture} is missing required arrays {sorted(missing)}; found {archive.files}"
            )
        x = np.asarray(archive["x"], dtype=np.float32)
        level = np.asarray(archive["l"], dtype=np.float32)
        cond = np.asarray(archive["cond"], dtype=np.float32)
        expected = (
            np.asarray(archive["logits"], dtype=np.float32)
            if "logits" in archive.files
            else None
        )
    validate_occupancy_inputs(x, level, cond)
    if expected is not None:
        validate_array(expected, "logits", LOGITS_SHAPE)
    return x, level, cond, expected


def load_gaussian_fixture(path: Path) -> tuple[Any, Any, Any | None]:
    """Load strict Gaussian decoder inputs and an optional PyTorch reference."""

    import numpy as np

    fixture = resolved_file(path, "Gaussian decoder fixture")
    with np.load(fixture, allow_pickle=False) as archive:
        required = {"points", "cond"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(
                f"{fixture} is missing required arrays {sorted(missing)}; found {archive.files}"
            )
        points = np.asarray(archive["points"], dtype=np.float32)
        cond = np.asarray(archive["cond"], dtype=np.float32)
        expected = (
            np.asarray(archive["features"], dtype=np.float32)
            if "features" in archive.files
            else None
        )
    validate_gaussian_inputs(points, cond)
    if expected is not None:
        validate_array(expected, "features", FEATURES_SHAPE)
    return points, cond, expected


def validate_array(array: Any, name: str, shape: tuple[int, ...]) -> None:
    """Validate one fixed-shape finite float-compatible array."""

    import numpy as np

    if tuple(array.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(array.shape)}; expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")


def validate_occupancy_inputs(x: Any, level: Any, cond: Any) -> None:
    """Validate occupancy shape, range, and official resolution semantics."""

    validate_array(x, "x", POINTS_SHAPE)
    validate_array(level, "l", LEVEL_SHAPE)
    validate_array(cond, "cond", COND_SHAPE)
    minimum = float(x.min())
    maximum = float(x.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"x must be normalized to [0,1], observed [{minimum}, {maximum}]")
    resolution = float(level[0])
    if resolution not in VALID_OCTREE_RESOLUTIONS:
        raise ValueError(
            f"l must contain an official octree resolution in {VALID_OCTREE_RESOLUTIONS}; "
            f"observed {resolution}"
        )


def validate_gaussian_inputs(points: Any, cond: Any) -> None:
    """Validate Gaussian point/conditioning inputs."""

    validate_array(points, "points", POINTS_SHAPE)
    validate_array(cond, "cond", COND_SHAPE)
    minimum = float(points.min())
    maximum = float(points.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"points must be normalized to [0,1], observed [{minimum}, {maximum}]"
        )


def array_summary(array: Any) -> dict[str, float]:
    """Return stable numerical summary values for JSON reports."""

    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "l2_norm": float(np.linalg.norm(values.ravel())),
    }


def comparison_metrics(
    reference: Any,
    candidate: Any,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compute and enforce allclose plus diagnostic parity metrics."""

    import numpy as np

    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        raise ValueError(
            f"Output shape mismatch: PyTorch {reference64.shape}, ORT {candidate64.shape}"
        )
    reference_finite = bool(np.isfinite(reference64).all())
    candidate_finite = bool(np.isfinite(candidate64).all())
    if not reference_finite or not candidate_finite:
        return {
            "passed": False,
            "reference_finite": reference_finite,
            "candidate_finite": candidate_finite,
            "count": int(reference64.size),
        }

    delta = candidate64 - reference64
    absolute = np.abs(delta)
    relative = absolute / np.maximum(np.abs(reference64), 1e-6)
    allowed = atol + rtol * np.abs(reference64)
    within = absolute <= allowed
    ref_flat = reference64.ravel()
    candidate_flat = candidate64.ravel()
    denominator = float(np.linalg.norm(ref_flat) * np.linalg.norm(candidate_flat))
    cosine = float(np.dot(ref_flat, candidate_flat) / denominator) if denominator else 1.0
    worst_flat_index = int(np.argmax(absolute))
    worst_index = tuple(int(v) for v in np.unravel_index(worst_flat_index, absolute.shape))
    return {
        "passed": bool(within.all()),
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
        "count": int(reference64.size),
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_relative_error_at_1e-6_floor": float(relative.max()),
        "mean_relative_error_at_1e-6_floor": float(relative.mean()),
        "p99_relative_error_at_1e-6_floor": float(np.quantile(relative, 0.99)),
        "cosine_similarity": cosine,
        "fraction_within_tolerance": float(within.mean()),
        "worst_index": list(worst_index),
        "worst_reference": float(reference64[worst_index]),
        "worst_candidate": float(candidate64[worst_index]),
        "worst_allowed_error": float(allowed[worst_index]),
    }


def make_ort_session(
    ort: Any,
    graph_path: Path,
    providers: list[str],
    threads: int,
) -> Any:
    """Construct an optimized ORT session after validating provider availability."""

    available = set(ort.get_available_providers())
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise RuntimeError(
            f"Requested ORT providers are unavailable: {missing}. Available: {sorted(available)}"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(graph_path), sess_options=options, providers=providers)


def run_pytorch_reference(
    *,
    torch: Any,
    component: str,
    triposplat_repo: Path,
    weights: Path,
    device: Any,
    internal_precision: str,
    inputs: tuple[Any, ...],
) -> tuple[Any, float]:
    """Execute one official graph slice and release its model before ORT loads."""

    import numpy as np

    decoder = load_official_decoder(
        torch,
        triposplat_repo,
        weights,
        device,
        internal_precision,
    )
    if component == "octree_occupancy_decoder":
        graph = make_octree_logits_graph(torch, decoder)
    elif component == "gaussian_decoder":
        graph = make_gaussian_features_graph(torch, decoder)
    else:
        raise ValueError(f"Unknown decoder component {component!r}")
    # The official checkpoint contains both large submodels.  The wrapper owns only
    # the selected child, so release the unused sibling before allocating activations.
    del decoder
    gc.collect()
    tensors = tuple(
        torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device=device)
        for value in inputs
    )
    synchronize_torch(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = graph(*tensors)
    synchronize_torch(torch, device)
    duration_ms = (time.perf_counter() - started) * 1000.0
    result = output.detach().to(device="cpu", dtype=torch.float32).numpy()
    del output, tensors, graph
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return result, duration_ms


def validate_with_ort(
    *,
    ort: Any,
    graph_path: Path,
    providers: list[str],
    threads: int,
    output_name: str,
    feeds: dict[str, Any],
) -> tuple[Any, float, Any]:
    """Create a session, run one measured inference, and return the session."""

    import numpy as np

    session = make_ort_session(ort, graph_path, providers, threads)
    contiguous_feeds = {
        name: np.ascontiguousarray(value, dtype=np.float32) for name, value in feeds.items()
    }
    started = time.perf_counter()
    output = session.run([output_name], contiguous_feeds)[0]
    duration_ms = (time.perf_counter() - started) * 1000.0
    return np.asarray(output, dtype=np.float32), duration_ms, session


def default_tolerances(internal_precision: str) -> tuple[float, float]:
    """Return parity gates appropriate to the graph's internal arithmetic."""

    return (1e-4, 1e-3) if internal_precision == "fp32" else (2e-2, 2e-2)


def validation_runtime_report(
    *,
    torch: Any,
    ort: Any,
    device: Any,
    session: Any,
    graph_path: Path,
) -> dict[str, Any]:
    """Describe the measured Python runtimes without implying browser performance."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "torch_device": str(device),
        "ort_providers": session.get_providers(),
        "onnx": str(graph_path),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable human-readable metrics report."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
