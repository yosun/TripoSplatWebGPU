"""Shared fixed-shape helpers for the TripoSplat one-step flow/DiT graph.

The neural network definition and checkpoint loader remain in a caller-provided
checkout of the official TripoSplat repository.  This module owns only the ONNX
boundary, two export-only algebraic adaptations, deterministic fixtures, and strict
contract/parity utilities.

Public graph tensors are always float32.  A browser artifact may use float16 weights
and compute internally; explicit casts at the graph boundary preserve the official
float32 timestep and the host-side float32 Euler sampler state.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


OFFICIAL_REPOSITORY_URL = "https://github.com/VAST-AI-Research/TripoSplat"

BATCH_SIZE = 1
LATENT_TOKENS = 8192
LATENT_CHANNELS = 16
CAMERA_CHANNELS = 5
CONDITION_TOKENS = 4101
FEATURE1_CHANNELS = 1280
FEATURE2_CHANNELS = 128

INPUT_NAMES = ("latent", "camera", "t", "feature1", "feature2")
OUTPUT_NAMES = ("pred_latent", "pred_camera")

LATENT_SHAPE = (BATCH_SIZE, LATENT_TOKENS, LATENT_CHANNELS)
CAMERA_SHAPE = (BATCH_SIZE, 1, CAMERA_CHANNELS)
TIMESTEP_SHAPE = (BATCH_SIZE,)
FEATURE1_SHAPE = (BATCH_SIZE, CONDITION_TOKENS, FEATURE1_CHANNELS)
FEATURE2_SHAPE = (BATCH_SIZE, CONDITION_TOKENS, FEATURE2_CHANNELS)
PRED_LATENT_SHAPE = LATENT_SHAPE
PRED_CAMERA_SHAPE = CAMERA_SHAPE

INPUT_SHAPES: Mapping[str, tuple[int, ...]] = {
    "latent": LATENT_SHAPE,
    "camera": CAMERA_SHAPE,
    "t": TIMESTEP_SHAPE,
    "feature1": FEATURE1_SHAPE,
    "feature2": FEATURE2_SHAPE,
}
OUTPUT_SHAPES: Mapping[str, tuple[int, ...]] = {
    "pred_latent": PRED_LATENT_SHAPE,
    "pred_camera": PRED_CAMERA_SHAPE,
}

INTERNAL_PRECISION_METADATA_KEY = "triposplat.internal_precision"
PUBLIC_IO_METADATA_VALUE = "float32"


@dataclass(frozen=True)
class OfficialSource:
    """Imported official pipeline and model modules from one checkout."""

    repository: Path
    pipeline_module: ModuleType
    model_module: ModuleType


@dataclass(frozen=True)
class AdapterMetadata:
    """Facts recorded after making the official model ONNX-exportable."""

    real_rope_modules: int
    attention_query_chunk: int
    attention_head_chunk: int
    attention_head_padding: int
    qk_norm_padding_tokens: int
    qk_norm_modules: int
    stable_rms_norm_modules: int
    rms_norm_eps: float | None
    attention_output_chunk: int
    attention_output_reduction_chunk: int
    attention_output_modules: int
    static_position_shape: tuple[int, ...]
    static_position_dtype: str
    static_position_sha256: str


def resolved_file(path: Path, description: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {result}")
    return result


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def import_official_source(triposplat_repo: Path) -> OfficialSource:
    """Import official ``model.py`` and ``triposplat.py`` without modifying them.

    The official pipeline uses ``from model import ...``.  A temporary ``model``
    module alias is installed only while executing ``triposplat.py``; the previous
    process module and ``sys.path`` are restored even when an import fails.
    """

    repository = triposplat_repo.expanduser().resolve()
    model_path = repository / "model.py"
    pipeline_path = repository / "triposplat.py"
    if not model_path.is_file() or not pipeline_path.is_file():
        raise FileNotFoundError(
            f"Expected model.py and triposplat.py in {repository}. Pass a clone of "
            f"{OFFICIAL_REPOSITORY_URL}."
        )

    identity = hashlib.sha256(str(repository).encode("utf-8")).hexdigest()[:16]
    model_name = f"_triposplat_official_flow_model_{identity}"
    pipeline_name = f"_triposplat_official_pipeline_{identity}"
    cached_model = sys.modules.get(model_name)
    cached_pipeline = sys.modules.get(pipeline_name)
    if cached_model is not None and cached_pipeline is not None:
        return OfficialSource(repository, cached_pipeline, cached_model)

    model_module = cached_model or _load_module(model_name, model_path)
    previous_model_alias = sys.modules.get("model")
    inserted_path = str(repository)
    sys.modules["model"] = model_module
    sys.path.insert(0, inserted_path)
    try:
        pipeline_module = _load_module(pipeline_name, pipeline_path)
    finally:
        try:
            sys.path.remove(inserted_path)
        except ValueError:
            pass
        if previous_model_alias is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous_model_alias

    if not hasattr(pipeline_module, "load_flow_model"):
        raise AttributeError(
            f"Official source {pipeline_path} does not define load_flow_model"
        )
    required_model_symbols = (
        "LatentSeqMMFlowModel",
        "RePo3DRotaryEmbedding",
        "apply_rotary_emb",
    )
    missing = [name for name in required_model_symbols if not hasattr(model_module, name)]
    if missing:
        raise AttributeError(
            f"Official source {model_path} is missing required symbols: {missing}"
        )
    return OfficialSource(repository, pipeline_module, model_module)


def source_revision(repository: Path) -> tuple[str, bool | None]:
    """Return the checkout commit and whether tracked source files are dirty."""

    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--", "model.py", "triposplat.py"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit or "unknown", bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def torch_dtype_for_precision(torch: Any, precision: str) -> Any:
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported internal precision {precision!r}; expected fp16 or fp32")


def choose_torch_device(torch: Any, requested: str) -> Any:
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def release_torch_model(torch: Any, device: Any, *objects: Any) -> None:
    """Drop large references and ask the selected accelerator to release caches."""

    # Deleting a function's local aliases cannot delete the caller's names, but it
    # does ensure this helper retains no references while collection runs.
    del objects
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def validate_official_flow_configuration(model: Any) -> None:
    expected = {
        "q_token_length": LATENT_TOKENS,
        "in_channels": LATENT_CHANNELS,
        "cam_channels": CAMERA_CHANNELS,
        "cond_channels": FEATURE1_CHANNELS,
        "cond2_channels": FEATURE2_CHANNELS,
        "out_channels": LATENT_CHANNELS,
    }
    mismatches = {
        name: (getattr(model, name, None), value)
        for name, value in expected.items()
        if getattr(model, name, None) != value
    }
    if mismatches:
        formatted = ", ".join(
            f"{name}={observed!r} (expected {wanted!r})"
            for name, (observed, wanted) in mismatches.items()
        )
        raise ValueError(f"Official flow-model configuration does not match the browser contract: {formatted}")


def load_official_flow_model(
    torch: Any,
    triposplat_repo: Path,
    weights: Path,
    device: Any,
    internal_precision: str,
    low_memory_construction: bool = True,
) -> tuple[Any, OfficialSource]:
    """Load the checkpoint through official ``triposplat.load_flow_model``.

    For an fp16 artifact, temporarily using fp16 as PyTorch's default construction
    dtype avoids allocating a second full fp32 parameter set.  The official loader,
    class, strict safetensors state load, and placement logic are still used.
    Explicitly-dtyped official constants (including the Sobol draw) are unaffected.
    """

    checkpoint = resolved_file(weights, "TripoSplat flow-model safetensors checkpoint")
    source = import_official_source(triposplat_repo)
    dtype = torch_dtype_for_precision(torch, internal_precision)
    previous_dtype = torch.get_default_dtype()
    original_sobol_engine = torch.quasirandom.SobolEngine

    class OfficialFloat32SobolEngine:
        """Keep official Sobol construction/draw float32 during fp16 allocation.

        SobolEngine builds a large integer-scaled first point using the current
        default float dtype. Constructing that state in fp16 overflows before
        ``draw(dtype=float32)`` can help, so both phases must retain the untouched
        official process default.
        """

        def __init__(self, *engine_args: Any, **engine_kwargs: Any) -> None:
            construction_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                self.engine = original_sobol_engine(*engine_args, **engine_kwargs)
            finally:
                torch.set_default_dtype(construction_dtype)

        def draw(
            self,
            n: int = 1,
            out: Any = None,
            dtype: Any = None,
        ) -> Any:
            return self.engine.draw(
                n=n,
                out=out,
                dtype=torch.float32 if dtype is None else dtype,
            )

    try:
        if low_memory_construction:
            torch.set_default_dtype(dtype)
            torch.quasirandom.SobolEngine = OfficialFloat32SobolEngine
        model = source.pipeline_module.load_flow_model(
            str(checkpoint),
            device=device,
            dtype=dtype,
        )
    finally:
        torch.quasirandom.SobolEngine = original_sobol_engine
        torch.set_default_dtype(previous_dtype)
    model = model.eval()
    validate_official_flow_configuration(model)

    mismatched = [
        f"{name}={parameter.dtype}"
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    ]
    if mismatched:
        raise RuntimeError(
            f"Official flow-model parameters did not convert to {dtype}: "
            + ", ".join(mismatched[:8])
        )
    return model, source


def _real_repo_frequencies(self: Any, hidden_states: Any) -> Any:
    """Real-valued equivalent of official ``RePo3DRotaryEmbedding.forward``.

    The final axis stores ``(cos(angle), sin(angle))`` instead of a complex64
    number.  Every operation used to produce ``angle`` intentionally keeps the
    ordering of the official source, including ``clamp_mul``'s detached branch.
    """

    import torch

    h = self.norm(hidden_states)
    feat = self.act(self.gate_map(h)) * self.content_map(h)
    out = self.final_map(feat)
    batch, length, _ = out.shape
    delta_pos = out.reshape(batch, length, self.num_heads, 3)

    def clamp_mul_exact(x: Any, frequency: Any) -> Any:
        frequency_tanh = frequency.tanh()
        return x * frequency_tanh + x.detach() * (frequency - frequency_tanh)

    angle_0 = clamp_mul_exact(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
    angle_1 = clamp_mul_exact(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
    angle_2 = clamp_mul_exact(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
    angles = torch.cat((angle_0, angle_1, angle_2), dim=-1).float()
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


def _apply_rotary_emb_real(hidden_states: Any, frequencies: Any) -> Any:
    """Pairwise real multiplication equal to official complex RoPE multiplication."""

    import torch

    pairs = hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2)
    real = pairs[..., 0]
    imaginary = pairs[..., 1]
    cosine = frequencies[..., 0]
    sine = frequencies[..., 1]
    rotated = torch.stack(
        (
            real * cosine - imaginary * sine,
            real * sine + imaginary * cosine,
        ),
        dim=-1,
    ).reshape(*hidden_states.shape)
    return rotated.type_as(hidden_states)


def _scaled_dot_product_attention_query_chunked(
    qkv: Any = None,
    q: Any = None,
    k: Any = None,
    v: Any = None,
    kv: Any = None,
    *,
    query_chunk_size: int,
    head_chunk_size: int,
    head_padding_size: int,
) -> Any:
    """Official SDPA split along independent head and query-token axes.

    Each query row has the same keys, values, scale, and row-wise softmax as the
    untouched operation.  Splitting queries therefore changes no attention
    dependency while preventing ONNX Runtime WebGPU from materializing the full
    ``[B,H,Lq,Lk]`` score tensor. A trailing duplicate head shields an ORT WebGPU
    1.27 correctness defect that corrupts the final head of the final SDPA group;
    the duplicate output is discarded. The fixed-shape exporter unrolls both loops.
    """

    import torch
    import torch.nn.functional as functional

    if not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer")
    if not isinstance(head_chunk_size, int) or head_chunk_size <= 0:
        raise ValueError("head_chunk_size must be a positive integer")
    if not isinstance(head_padding_size, int) or head_padding_size < 0:
        raise ValueError("head_padding_size must be a non-negative integer")
    if qkv is not None:
        q, k, v = qkv.unbind(dim=2)
    elif kv is not None:
        k, v = kv.unbind(dim=2)
    if q is None or k is None or v is None:
        raise ValueError("Chunked attention requires q, k, and v tensors")
    output_dtype = q.dtype
    q, k, v = (
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
    )
    # PyTorch's fused fp16 SDPA uses a higher-precision score/softmax path. Make
    # that boundary explicit so the decomposed ONNX graph does not ask WebGPU to
    # carry 24 residual blocks through fp16 softmax accumulation.
    k_float = k.float()
    v_float = v.float()
    head_groups = []
    for head_start in range(0, q.shape[1], head_chunk_size):
        head_end = min(head_start + head_chunk_size, q.shape[1])
        head_q = q[:, head_start:head_end, :, :]
        head_k = k_float[:, head_start:head_end, :, :]
        head_v = v_float[:, head_start:head_end, :, :]
        real_head_count = head_q.shape[1]
        if head_padding_size:
            if head_padding_size > head_q.shape[1]:
                raise ValueError("head_padding_size cannot exceed the current head group")
            # Duplicate live input rather than appending a constant-zero head. This
            # prevents export/runtime optimization from deleting the shielding work.
            head_q = torch.cat((head_q, head_q[:, :head_padding_size, :, :]), dim=1)
            head_k = torch.cat((head_k, head_k[:, :head_padding_size, :, :]), dim=1)
            head_v = torch.cat((head_v, head_v[:, :head_padding_size, :, :]), dim=1)
        query_chunks = [
            functional.scaled_dot_product_attention(
                head_q[:, :, start : start + query_chunk_size, :].float(),
                head_k,
                head_v,
            ).to(dtype=output_dtype)
            for start in range(0, q.shape[2], query_chunk_size)
        ]
        head_groups.append(torch.cat(query_chunks, dim=2)[:, :real_head_count, :, :])
    return torch.cat(head_groups, dim=1).permute(0, 2, 1, 3)


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def validate_real_rope_primitives(
    torch: Any,
    model: Any,
    model_module: ModuleType,
    seed: int = 20260714,
) -> dict[str, float | bool]:
    """Cheaply gate the real RoPE algebra against untouched official complex ops.

    This does not replace the validator's full one-call gate.  It fails early before
    a multi-gigabyte export if the official RoPE layout or arithmetic has changed.
    """

    layer = model.noise_repo_layers[0]
    length = 7
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(
        (1, length, int(model.model_channels)),
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.dtype)
    q = torch.randn(
        (1, length, int(model.num_heads), int(layer.head_dim)),
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.dtype)
    with torch.inference_mode():
        official_frequencies = layer(hidden)
        official_rotated = model_module.apply_rotary_emb(q, official_frequencies)
        real_frequencies = _real_repo_frequencies(layer, hidden)
        real_rotated = _apply_rotary_emb_real(q, real_frequencies)
    frequency_reference = torch.view_as_real(official_frequencies).float()
    frequency_error = float(
        (frequency_reference - real_frequencies.float()).abs().max().detach().cpu()
    )
    rotation_error = float(
        (official_rotated.float() - real_rotated.float()).abs().max().detach().cpu()
    )
    tolerance = 2e-3 if model.dtype == torch.float16 else 3e-6
    passed = frequency_error <= tolerance and rotation_error <= tolerance
    return {
        "passed": passed,
        "max_frequency_pair_error": frequency_error,
        "max_rotated_tensor_error": rotation_error,
        "atol": tolerance,
    }


def adapt_official_flow_for_onnx(
    torch: Any,
    model: Any,
    model_module: ModuleType,
    attention_query_chunk: int = 256,
    attention_head_chunk: int = 16,
    attention_head_padding: int = 0,
    qk_norm_padding_tokens: int = 1,
    rms_norm_eps: float | None = None,
    attention_output_chunk: int = 256,
    attention_output_reduction_chunk: int = 256,
) -> AdapterMetadata:
    """Apply three one-way, export-only adaptations to an official model instance.

    1. Complex RoPE is represented as real ``cos/sin`` pairs and applied with the
       algebraically identical two-real multiply.
    2. The fixed seeded-Sobol absolute position embedding is evaluated once with
       official code and registered as a buffer.  This prevents exporter/runtime
       constant-folding of large trigonometric arguments from changing its values.
    3. SDPA is split on the independent query and head axes so WebGPU never
       allocates the full multi-gigabyte attention score tensor and never invokes
       the known-corrupt 16-head ORT WebGPU kernel.

    Validators must run the untouched official forward first, then this adapted
    forward on identical tensors.  This function deliberately does not patch files
    in the source checkout.
    """

    existing = getattr(model, "_triposplat_onnx_adapter_metadata", None)
    if existing is not None:
        if existing.attention_query_chunk != attention_query_chunk:
            raise ValueError(
                "Flow model was already adapted with attention query chunk "
                f"{existing.attention_query_chunk}, not {attention_query_chunk}"
            )
        if existing.attention_head_chunk != attention_head_chunk:
            raise ValueError(
                "Flow model was already adapted with attention head chunk "
                f"{existing.attention_head_chunk}, not {attention_head_chunk}"
            )
        if existing.attention_head_padding != attention_head_padding:
            raise ValueError(
                "Flow model was already adapted with attention head padding "
                f"{existing.attention_head_padding}, not {attention_head_padding}"
            )
        if existing.qk_norm_padding_tokens != qk_norm_padding_tokens:
            raise ValueError(
                "Flow model was already adapted with Q/K norm token padding "
                f"{existing.qk_norm_padding_tokens}, not {qk_norm_padding_tokens}"
            )
        if existing.rms_norm_eps != rms_norm_eps:
            raise ValueError(
                "Flow model was already adapted with RMS norm epsilon "
                f"{existing.rms_norm_eps}, not {rms_norm_eps}"
            )
        if existing.attention_output_chunk != attention_output_chunk:
            raise ValueError(
                "Flow model was already adapted with attention output chunk "
                f"{existing.attention_output_chunk}, not {attention_output_chunk}"
            )
        if existing.attention_output_reduction_chunk != attention_output_reduction_chunk:
            raise ValueError(
                "Flow model was already adapted with attention output reduction chunk "
                f"{existing.attention_output_reduction_chunk}, not "
                f"{attention_output_reduction_chunk}"
            )
        return existing

    if not isinstance(attention_query_chunk, int) or attention_query_chunk <= 0:
        raise ValueError("attention_query_chunk must be a positive integer")
    if not isinstance(attention_head_chunk, int) or attention_head_chunk <= 0:
        raise ValueError("attention_head_chunk must be a positive integer")
    if not isinstance(attention_head_padding, int) or attention_head_padding < 0:
        raise ValueError("attention_head_padding must be a non-negative integer")
    if attention_head_padding > attention_head_chunk:
        raise ValueError("attention_head_padding cannot exceed attention_head_chunk")
    if not isinstance(qk_norm_padding_tokens, int) or qk_norm_padding_tokens <= 0:
        raise ValueError("qk_norm_padding_tokens must be a positive integer")
    if rms_norm_eps is not None and (
        not isinstance(rms_norm_eps, (int, float)) or rms_norm_eps <= 0
    ):
        raise ValueError("rms_norm_eps must be None or a positive number")
    if not isinstance(attention_output_chunk, int) or attention_output_chunk <= 0:
        raise ValueError("attention_output_chunk must be a positive integer")
    if (
        not isinstance(attention_output_reduction_chunk, int)
        or attention_output_reduction_chunk <= 0
    ):
        raise ValueError("attention_output_reduction_chunk must be a positive integer")

    if not hasattr(model, "pos_pe") or not hasattr(model, "pos_embedder"):
        raise AttributeError("Official flow model lacks pos_pe/pos_embedder")
    with torch.inference_mode():
        position_input = model.pos_pe.to(device=model.device, dtype=torch.float32)
        fixed_position = model.pos_embedder(position_input).to(dtype=model.dtype)
    expected_position_shape = (BATCH_SIZE, LATENT_TOKENS, int(model.model_channels))
    if tuple(fixed_position.shape) != expected_position_shape:
        raise ValueError(
            f"Official static position embedding has shape {tuple(fixed_position.shape)}, "
            f"expected {expected_position_shape}"
        )

    class StaticOfficialPositionEmbedding(torch.nn.Module):
        def __init__(self, value: Any) -> None:
            super().__init__()
            self.register_buffer("value", value.detach().clone(), persistent=True)

        def forward(self, unused_position_input: Any) -> Any:
            return self.value

    model.pos_embedder = StaticOfficialPositionEmbedding(fixed_position).to(model.device)

    repo_type = model_module.RePo3DRotaryEmbedding
    replaced = 0
    for submodule in model.modules():
        if isinstance(submodule, repo_type):
            submodule.forward = types.MethodType(_real_repo_frequencies, submodule)
            replaced += 1
    expected_replaced = sum(
        len(getattr(model, name))
        for name in ("noise_repo_layers", "context_repo_layers", "repo_layers")
    )
    if replaced != expected_replaced or replaced <= 0:
        raise RuntimeError(
            f"Adapted {replaced} RePo modules, expected {expected_replaced}; "
            "the official architecture may have changed"
        )
    model_module.apply_rotary_emb = _apply_rotary_emb_real
    model_module.scaled_dot_product_attention = lambda qkv=None, q=None, k=None, v=None, kv=None: (
        _scaled_dot_product_attention_query_chunked(
            qkv=qkv,
            q=q,
            k=k,
            v=v,
            kv=kv,
            query_chunk_size=attention_query_chunk,
            head_chunk_size=attention_head_chunk,
            head_padding_size=attention_head_padding,
        )
    )

    # Core AI's TripoSplat conversion identified a true-scale failure in the
    # original F.normalize formulation: some converters/runtimes omit its eps
    # clamp for near-zero unconditional Q/K vectors.  The mean-square form is
    # algebraically identical away from epsilon and exposes the stabilization as
    # primitive ONNX ops that ORT WebGPU preserves.
    stable_rms_norm_modules = 0
    if rms_norm_eps is not None:
        rms_norm_type = model_module.MultiHeadRMSNorm

        def stable_rms_norm_forward(self: Any, value: Any) -> Any:
            original_dtype = value.dtype
            fp32 = value.float()
            inverse_rms = torch.rsqrt(
                (fp32 * fp32).mean(dim=-1, keepdim=True) + rms_norm_eps
            )
            return (fp32 * inverse_rms * self.gamma.float()).to(dtype=original_dtype)

        for submodule in model.modules():
            if isinstance(submodule, rms_norm_type):
                submodule.forward = types.MethodType(stable_rms_norm_forward, submodule)
                stable_rms_norm_modules += 1
        if stable_rms_norm_modules <= 0:
            raise RuntimeError("No MultiHeadRMSNorm modules were adapted")

    class ChunkedAttentionOutputProjection(torch.nn.Module):
        """Express the token-wise linear map as bounded 1x1 convolutions."""

        def __init__(self, projection: Any) -> None:
            super().__init__()
            self.projection = projection

        def forward(self, value: Any) -> Any:
            import torch.nn.functional as functional

            outputs = []
            output_channels = int(self.projection.weight.shape[0])
            original_shape = value.shape
            token_count = int(value.shape[-2])
            first_token = value[..., :1, :].float().unsqueeze(-2)
            tail_indices = torch.arange(
                1,
                token_count,
                dtype=torch.int64,
                device=value.device,
            )
            channels_first = value.transpose(-1, -2).unsqueeze(-1)
            for start in range(0, output_channels, attention_output_chunk):
                end = min(start + attention_output_chunk, output_channels)
                bias = (
                    None
                    if self.projection.bias is None
                    else self.projection.bias[start:end]
                )
                projected = functional.conv2d(
                    channels_first,
                    self.projection.weight[start:end].unsqueeze(-1).unsqueeze(-1),
                    bias,
                )
                bulk = projected.squeeze(-1).transpose(-1, -2)
                bulk_tail = torch.index_select(bulk, -2, tail_indices)
                special_products = (
                    first_token
                    * self.projection.weight[start:end]
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                ).transpose(-1, -2)
                reduction_rows = int(special_products.shape[-2])
                if reduction_rows & (reduction_rows - 1):
                    raise ValueError(
                        "Attention projection input width must be a power of two"
                    )
                while reduction_rows > 256:
                    half = reduction_rows // 2
                    # Keep output channels as the 256-wide innermost dimension.
                    special_products = (
                        special_products[..., :half, :]
                        + special_products[..., half:reduction_rows, :]
                    )
                    reduction_rows = half
                while reduction_rows > 1:
                    half = reduction_rows // 2
                    padding = torch.zeros(
                        256 - half,
                        dtype=torch.int64,
                        device=value.device,
                    )
                    left_indices = torch.cat(
                        (
                            torch.arange(0, half, dtype=torch.int64, device=value.device),
                            padding,
                        )
                    )
                    right_indices = torch.cat(
                        (
                            torch.arange(
                                half,
                                reduction_rows,
                                dtype=torch.int64,
                                device=value.device,
                            ),
                            padding,
                        )
                    )
                    active_mask = torch.cat(
                        (
                            torch.ones(half, dtype=torch.float32, device=value.device),
                            torch.zeros(256 - half, dtype=torch.float32, device=value.device),
                        )
                    ).reshape(1, 1, 256, 1)
                    # ORT WebGPU corrupts an accumulator once this dimension reaches
                    # 128 rows (the tail starts at flat element 16,384). Both gathers,
                    # masks, and the Add therefore keep exactly 256 rows, with inactive
                    # lanes explicitly zeroed.
                    special_products = (
                        torch.index_select(special_products, -2, left_indices) * active_mask
                        + torch.index_select(special_products, -2, right_indices) * active_mask
                    )
                    reduction_rows = half
                special = special_products[..., 0, :]
                if bias is not None:
                    special = special + bias.float().reshape(1, 1, -1)
                outputs.append(
                    torch.cat(
                        (special.to(dtype=value.dtype), bulk_tail),
                        dim=-2,
                    )
                )
            return torch.cat(outputs, dim=-1).reshape(*original_shape[:-1], output_channels)

    class PaddedQkRmsNorm(torch.nn.Module):
        """Keep real vectors away from ORT WebGPU's final reduction lane."""

        def __init__(self, norm: Any) -> None:
            super().__init__()
            self.norm = norm

        def forward(self, value: Any) -> Any:
            token_count = value.shape[1]
            if qk_norm_padding_tokens > token_count:
                raise ValueError("Q/K norm padding cannot exceed the token count")
            padded = torch.cat(
                (value, value[:, :qk_norm_padding_tokens, ...]),
                dim=1,
            )
            return self.norm(padded)[:, :token_count, ...]

    attention_type = model_module.RopeMultiHeadAttention
    output_modules = 0
    qk_norm_modules = 0
    for submodule in model.modules():
        if isinstance(submodule, attention_type):
            if getattr(submodule, "qk_rms_norm", False):
                submodule.q_norm = PaddedQkRmsNorm(submodule.q_norm)
                submodule.k_norm = PaddedQkRmsNorm(submodule.k_norm)
                qk_norm_modules += 2
            submodule.out = ChunkedAttentionOutputProjection(submodule.out)
            output_modules += 1
    if output_modules != 28:
        raise RuntimeError(
            f"Chunked {output_modules} attention output projections; expected 28"
        )
    if qk_norm_modules != 56:
        raise RuntimeError(f"Padded {qk_norm_modules} Q/K norm modules; expected 56")

    metadata = AdapterMetadata(
        real_rope_modules=replaced,
        attention_query_chunk=attention_query_chunk,
        attention_head_chunk=attention_head_chunk,
        attention_head_padding=attention_head_padding,
        qk_norm_padding_tokens=qk_norm_padding_tokens,
        qk_norm_modules=qk_norm_modules,
        stable_rms_norm_modules=stable_rms_norm_modules,
        rms_norm_eps=None if rms_norm_eps is None else float(rms_norm_eps),
        attention_output_chunk=attention_output_chunk,
        attention_output_reduction_chunk=attention_output_reduction_chunk,
        attention_output_modules=output_modules,
        static_position_shape=tuple(int(value) for value in fixed_position.shape),
        static_position_dtype=str(fixed_position.dtype).removeprefix("torch."),
        static_position_sha256=_tensor_sha256(fixed_position),
    )
    model._triposplat_onnx_adapter_metadata = metadata
    return metadata


def make_browser_flow_step(torch: Any, model: Any, internal_precision: str) -> Any:
    """Wrap official dict I/O with fixed float32 browser tensors and explicit casts."""

    internal_dtype = torch_dtype_for_precision(torch, internal_precision)

    class BrowserFlowStep(torch.nn.Module):
        def __init__(self, official_model: Any) -> None:
            super().__init__()
            self.flow_model = official_model

        def forward(
            self,
            latent: Any,
            camera: Any,
            t: Any,
            feature1: Any,
            feature2: Any,
        ) -> tuple[Any, Any]:
            result = self.flow_model(
                {
                    "latent": latent.to(dtype=internal_dtype),
                    "camera": camera.to(dtype=internal_dtype),
                },
                # Official FlowEulerCfgSampler constructs 1000*t in float32.
                t.to(dtype=torch.float32),
                {
                    "feature1": feature1.to(dtype=internal_dtype),
                    "feature2": feature2.to(dtype=internal_dtype),
                },
            )
            return (
                result["latent"].to(dtype=torch.float32),
                result["camera"].to(dtype=torch.float32),
            )

    return BrowserFlowStep(model).eval().to(model.device)


def deterministic_inputs(seed: int = 20260714, timestep: float = 1000.0) -> dict[str, Any]:
    """Create stable float32 tensors with encoder-like unit-normal conditioning."""

    import numpy as np

    if not np.isfinite(timestep) or timestep < 0.0 or timestep > 1000.0:
        raise ValueError(f"timestep must be finite and in [0,1000], got {timestep}")
    rng = np.random.default_rng(seed)

    def normal(shape: tuple[int, ...]) -> Any:
        return np.ascontiguousarray(rng.standard_normal(shape).astype(np.float32))

    values = {
        "latent": normal(LATENT_SHAPE),
        "camera": normal(CAMERA_SHAPE),
        "t": np.asarray([timestep], dtype=np.float32),
        "feature1": normal(FEATURE1_SHAPE),
        "feature2": normal(FEATURE2_SHAPE),
    }
    validate_input_arrays(values)
    return values


def validate_input_arrays(values: Mapping[str, Any]) -> None:
    import numpy as np

    missing = [name for name in INPUT_NAMES if name not in values]
    extra = [name for name in values if name not in INPUT_NAMES]
    if missing or extra:
        raise ValueError(f"DiT inputs mismatch: missing={missing}, extra={extra}")
    for name in INPUT_NAMES:
        array = np.asarray(values[name])
        if tuple(array.shape) != INPUT_SHAPES[name]:
            raise ValueError(
                f"{name} has shape {tuple(array.shape)}; expected {INPUT_SHAPES[name]}"
            )
        if array.dtype != np.float32:
            raise TypeError(f"{name} has dtype {array.dtype}; public contract requires float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
    timestep_min = float(np.asarray(values["t"]).min())
    timestep_max = float(np.asarray(values["t"]).max())
    if timestep_min < 0.0 or timestep_max > 1000.0:
        raise ValueError(
            f"t must contain already-scaled official timesteps in [0,1000], "
            f"observed [{timestep_min:.8g},{timestep_max:.8g}]"
        )


def load_input_fixture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load strict public inputs plus optional official prediction references."""

    import numpy as np

    fixture_path = resolved_file(path, "DiT NPZ fixture")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        missing = [name for name in INPUT_NAMES if name not in fixture.files]
        if missing:
            raise KeyError(
                f"{fixture_path} is missing {missing}; found {fixture.files}"
            )
        inputs = {
            name: np.ascontiguousarray(np.asarray(fixture[name], dtype=np.float32))
            for name in INPUT_NAMES
        }
        expected: dict[str, Any] = {}
        for name in OUTPUT_NAMES:
            if name in fixture.files:
                expected[name] = np.ascontiguousarray(
                    np.asarray(fixture[name], dtype=np.float32)
                )
    if expected and set(expected) != set(OUTPUT_NAMES):
        raise KeyError(
            f"{fixture_path} must contain both {list(OUTPUT_NAMES)} or neither; "
            f"found reference keys {sorted(expected)}"
        )
    validate_input_arrays(inputs)
    for name, array in expected.items():
        if tuple(array.shape) != OUTPUT_SHAPES[name]:
            raise ValueError(
                f"Fixture {name} has shape {tuple(array.shape)}; expected {OUTPUT_SHAPES[name]}"
            )
    return inputs, expected


def fixed_value_shape(value_info: Any) -> tuple[int | None, ...]:
    return tuple(
        int(dimension.dim_value) if dimension.HasField("dim_value") else None
        for dimension in value_info.type.tensor_type.shape.dim
    )


def metadata_dict(model_proto: Any) -> dict[str, str]:
    return {entry.key: entry.value for entry in model_proto.metadata_props}


def verify_onnx_contract(
    onnx: Any,
    model_proto: Any,
    require_metadata: bool = True,
) -> str:
    """Reject names, symbolic axes, dtypes, complex tensors, or metadata drift."""

    inputs = {value.name: value for value in model_proto.graph.input}
    outputs = {value.name: value for value in model_proto.graph.output}
    if set(inputs) != set(INPUT_NAMES):
        raise ValueError(f"ONNX inputs are {sorted(inputs)}; expected {list(INPUT_NAMES)}")
    if set(outputs) != set(OUTPUT_NAMES):
        raise ValueError(f"ONNX outputs are {sorted(outputs)}; expected {list(OUTPUT_NAMES)}")

    for name, expected_shape in (*INPUT_SHAPES.items(), *OUTPUT_SHAPES.items()):
        value = inputs[name] if name in inputs else outputs[name]
        observed_shape = fixed_value_shape(value)
        if observed_shape != expected_shape:
            raise ValueError(
                f"ONNX {name} shape is {observed_shape}; expected fixed {expected_shape}"
            )
        observed_type = value.type.tensor_type.elem_type
        if observed_type != onnx.TensorProto.FLOAT:
            raise ValueError(
                f"ONNX {name} element type is {observed_type}; public I/O must be float32"
            )

    complex_types = {onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128}
    complex_initializers = [
        tensor.name
        for tensor in model_proto.graph.initializer
        if tensor.data_type in complex_types
    ]
    complex_value_info = [
        value.name
        for value in (
            list(model_proto.graph.input)
            + list(model_proto.graph.output)
            + list(model_proto.graph.value_info)
        )
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type in complex_types
    ]
    if complex_initializers or complex_value_info:
        raise ValueError(
            "Exported graph still contains complex tensors: "
            f"initializers={complex_initializers[:8]}, values={complex_value_info[:8]}"
        )

    metadata = metadata_dict(model_proto)
    precision = metadata.get(INTERNAL_PRECISION_METADATA_KEY, "")
    if require_metadata and precision not in {"fp16", "fp32"}:
        raise ValueError(
            f"ONNX metadata {INTERNAL_PRECISION_METADATA_KEY!r} is {precision!r}; "
            "expected fp16 or fp32"
        )
    public_io = metadata.get("triposplat.public_io", "")
    if require_metadata and public_io != PUBLIC_IO_METADATA_VALUE:
        raise ValueError(
            f"ONNX metadata triposplat.public_io is {public_io!r}; expected float32"
        )
    return precision or "unknown"


def verify_external_data_files(graph_path: Path, model_proto: Any) -> list[Path]:
    """Validate every external-data location, offset, and byte range."""

    graph = graph_path.expanduser().resolve()
    locations: dict[str, Path] = {}
    for tensor in model_proto.graph.initializer:
        if not tensor.external_data:
            continue
        info = {entry.key: entry.value for entry in tensor.external_data}
        location = info.get("location")
        if not location:
            raise ValueError(f"External initializer {tensor.name!r} has no location")
        candidate = (graph.parent / location).resolve()
        try:
            candidate.relative_to(graph.parent)
        except ValueError as exc:
            raise ValueError(
                f"External initializer {tensor.name!r} escapes graph directory: {location}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"External initializer {tensor.name!r} references missing {candidate}"
            )
        offset = int(info.get("offset", "0"))
        length_text = info.get("length")
        if offset < 0:
            raise ValueError(f"External initializer {tensor.name!r} has negative offset")
        if length_text is not None:
            length = int(length_text)
            if length < 0 or offset + length > candidate.stat().st_size:
                raise ValueError(
                    f"External initializer {tensor.name!r} range "
                    f"[{offset},{offset + length}) exceeds {candidate}"
                )
        locations[location] = candidate
    return [locations[key] for key in sorted(locations)]


def array_summary(array: Any) -> dict[str, float | int]:
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    return {
        "count": int(values.size),
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
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    if ref.shape != got.shape:
        raise ValueError(f"Comparison shape mismatch: {ref.shape} vs {got.shape}")
    reference_finite = bool(np.isfinite(ref).all())
    candidate_finite = bool(np.isfinite(got).all())
    if not reference_finite or not candidate_finite:
        return {
            "passed": False,
            "count": int(ref.size),
            "reference_finite": reference_finite,
            "candidate_finite": candidate_finite,
        }
    delta = got - ref
    absolute = np.abs(delta)
    relative = absolute / np.maximum(np.abs(ref), 1e-6)
    allowed = atol + rtol * np.abs(ref)
    within = absolute <= allowed
    ref_flat = ref.ravel()
    got_flat = got.ravel()
    norm_product = float(np.linalg.norm(ref_flat) * np.linalg.norm(got_flat))
    cosine = float(np.dot(ref_flat, got_flat) / norm_product) if norm_product else 1.0
    worst_flat = int(np.argmax(absolute))
    worst_index = tuple(int(value) for value in np.unravel_index(worst_flat, ref.shape))
    return {
        "passed": bool(within.all()),
        "count": int(ref.size),
        "reference_finite": True,
        "candidate_finite": True,
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_relative_error_at_1e-6_floor": float(relative.max()),
        "mean_relative_error_at_1e-6_floor": float(relative.mean()),
        "cosine_similarity": cosine,
        "fraction_within_tolerance": float(within.mean()),
        "worst_index": list(worst_index),
        "worst_reference": float(ref[worst_index]),
        "worst_candidate": float(got[worst_index]),
        "worst_allowed_error": float(allowed[worst_index]),
    }


def shifted_flow_schedule(steps: int, shift: float) -> list[tuple[float, float]]:
    """Return official ``FlowEulerCfgSampler`` (t, t_previous) float64 pairs."""

    import numpy as np

    if not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"steps must be a positive integer, got {steps}")
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError(f"shift must be a positive finite number, got {shift}")
    linear = np.linspace(1, 0, steps + 1)
    timesteps = shift * linear / (1 + (shift - 1) * linear)
    return [
        (float(timesteps[index]), float(timesteps[index + 1]))
        for index in range(steps)
    ]


def write_json(path: Path, value: Any) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def coerce_float32_inputs(values: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    result = {
        name: np.ascontiguousarray(np.asarray(values[name], dtype=np.float32))
        for name in INPUT_NAMES
    }
    validate_input_arrays(result)
    return result


def torch_inputs_from_numpy(
    torch: Any,
    values: Mapping[str, Any],
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    return tuple(
        torch.from_numpy(values[name]).to(device=device, dtype=torch.float32)
        for name in INPUT_NAMES
    )  # type: ignore[return-value]


def output_mapping(outputs: Sequence[Any]) -> dict[str, Any]:
    if len(outputs) != len(OUTPUT_NAMES):
        raise ValueError(f"Expected {len(OUTPUT_NAMES)} outputs, got {len(outputs)}")
    return dict(zip(OUTPUT_NAMES, outputs))
