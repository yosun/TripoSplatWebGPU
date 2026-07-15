"""Shared helpers for the fixed-shape TripoSplat DINOv3 encoder export.

The network itself is always imported from a caller-provided checkout of the
official TripoSplat repository.  This module only defines the browser graph
boundary, deterministic parity inputs, and validation/reporting utilities.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from types import MethodType
from typing import Any


IMAGE_SIZE = 1024
HIDDEN_SIZE = 1280
PATCH_SIZE = 16
PREFIX_TOKENS = 5
PATCH_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2
TOKEN_COUNT = PREFIX_TOKENS + PATCH_TOKENS

INPUT_NAME = "pixel_values"
OUTPUT_NAME = "feature1"
INPUT_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
OUTPUT_SHAPE = (1, TOKEN_COUNT, HIDDEN_SIZE)

DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)
EXTRA_LAYER_NORM_EPS = 1e-5

OFFICIAL_REPOSITORY_URL = "https://github.com/VAST-AI-Research/TripoSplat"
INTERNAL_PRECISION_METADATA_KEY = "triposplat.internal_precision"


def resolved_file(path: Path, description: str) -> Path:
    """Resolve *path* and reject a missing/non-file input."""

    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {result}")
    return result


def import_official_model(triposplat_repo: Path) -> ModuleType:
    """Import ``model.py`` directly from an official TripoSplat checkout."""

    repo = triposplat_repo.expanduser().resolve()
    model_path = repo / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Could not find {model_path}. Pass --triposplat-repo pointing at a clone "
            f"of {OFFICIAL_REPOSITORY_URL}."
        )

    module_name = "_triposplat_official_model"
    loaded = sys.modules.get(module_name)
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == model_path:
        module = loaded
    else:
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create an import specification for {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    if not hasattr(module, "DinoV3ViT"):
        raise AttributeError(
            f"{model_path} does not define DinoV3ViT; update the checkout or use "
            "the official TripoSplat repository."
        )
    return module


def torch_dtype_for_precision(torch: Any, precision: str) -> Any:
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported internal precision {precision!r}; expected fp16 or fp32")


def checkpoint_dtype_counts(weights: Path) -> dict[str, int]:
    """Read safetensors dtype headers without materializing checkpoint tensors."""

    from safetensors import safe_open

    checkpoint = resolved_file(weights, "DINOv3 safetensors checkpoint")
    counts: dict[str, int] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            dtype = str(handle.get_slice(key).get_dtype()).upper()
            counts[dtype] = counts.get(dtype, 0) + 1
    if not counts:
        raise ValueError(f"DINOv3 checkpoint contains no tensors: {checkpoint}")
    return dict(sorted(counts.items()))


def load_official_encoder(
    torch: Any,
    triposplat_repo: Path,
    weights: Path,
    device: Any,
    internal_precision: str,
) -> Any:
    """Load the official encoder and convert source FP16/BF16 weights for WebGPU.

    ``DinoV3ViT.load_safetensors`` remains the checkpoint loader/source of truth.
    Constructing directly in the requested dtype avoids first allocating an extra
    3+ GiB FP32 parameter set when producing the normal FP16 browser artifact.
    ``load_state_dict`` performs the source BF16/FP16 -> target dtype conversion.
    """

    checkpoint = resolved_file(weights, "DINOv3 safetensors checkpoint")
    official_model = import_official_model(triposplat_repo)
    dtype = torch_dtype_for_precision(torch, internal_precision)

    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        encoder = official_model.DinoV3ViT()
    finally:
        torch.set_default_dtype(previous_default_dtype)

    encoder.load_safetensors(str(checkpoint))
    encoder = encoder.eval().to(device=device, dtype=dtype)
    floating_state = [
        (f"parameter:{name}", tensor)
        for name, tensor in encoder.named_parameters()
    ] + [
        (f"buffer:{name}", tensor)
        for name, tensor in encoder.named_buffers()
    ]
    mismatched = [
        f"{name}={tensor.dtype}"
        for name, tensor in floating_state
        if tensor.is_floating_point() and tensor.dtype != dtype
    ]
    if mismatched:
        preview = ", ".join(mismatched[:8])
        raise RuntimeError(f"Official DINOv3 state did not convert to {dtype}: {preview}")
    return encoder


def make_browser_encoder(torch: Any, encoder: Any, internal_precision: str) -> Any:
    """Wrap official DINOv3 with stable FP32 browser I/O and pipeline post-norm."""

    internal_dtype = torch_dtype_for_precision(torch, internal_precision)

    class BrowserDinoV3Encoder(torch.nn.Module):
        def __init__(self, official_encoder: Any) -> None:
            super().__init__()
            self.encoder = official_encoder

        def forward(self, pixel_values: Any) -> Any:
            # The public contract is always FP32.  Browser-friendly artifacts cast
            # once to FP16 internally, avoiding BF16 operators unsupported by the
            # WebGPU execution provider.
            raw = self.encoder(pixel_values.to(dtype=internal_dtype))

            # This is intentionally separate from DinoV3ViT.norm.  Official
            # triposplat.encode_image() applies this extra, non-affine normalization
            # in FP32 before feature1 enters the flow model.
            return torch.nn.functional.layer_norm(
                raw.to(dtype=torch.float32),
                (HIDDEN_SIZE,),
                weight=None,
                bias=None,
                eps=EXTRA_LAYER_NORM_EPS,
            )

    return BrowserDinoV3Encoder(encoder)


def adapt_official_dinov3_for_onnx(
    torch: Any,
    encoder: Any,
    official_model: ModuleType,
    *,
    attention_query_chunk: int = 256,
    attention_head_chunk: int = 5,
    linear_token_chunk: int = 256,
) -> dict[str, int]:
    """Replace only DINO attention evaluation with an algebraically equivalent form.

    ONNX Runtime WebGPU's full fp16 20-head, 4101-token attention accumulates
    enough error across 32 blocks to invalidate the final conditioning tensor.
    Query/head chunks are independent, so evaluating them separately does not
    change attention semantics. Scores, softmax, and value accumulation stay
    float32, matching the stable conversion strategy used by the DiT export.
    """

    if attention_query_chunk < 1 or attention_head_chunk < 1 or linear_token_chunk < 1:
        raise ValueError("DINO attention chunk sizes must be positive")
    attention_type = getattr(official_model, "DinoV3Attention", None)
    mlp_type = getattr(official_model, "DinoV3MLP", None)
    if attention_type is None or mlp_type is None:
        raise AttributeError("Official model module has no DinoV3Attention/DinoV3MLP")
    adapted = 0
    adapted_mlps = 0

    def project_tokens(linear: Any, value: Any) -> Any:
        token_count = int(value.shape[1])
        return torch.cat(
            [
                linear(value[:, start : min(start + linear_token_chunk, token_count)])
                for start in range(0, token_count, linear_token_chunk)
            ],
            dim=1,
        )

    def chunked_forward(
        self: Any,
        x: Any,
        cos: Any,
        sin: Any,
        num_prefix_tokens: int = 0,
    ) -> Any:
        batch, tokens, channels = x.shape
        q = project_tokens(self.q_proj, x).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = project_tokens(self.k_proj, x).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = project_tokens(self.v_proj, x).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        def rotate_half(value: Any) -> Any:
            first, second = value.chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)

        if num_prefix_tokens > 0:
            q_prefix, q_patch = q.split(
                (num_prefix_tokens, tokens - num_prefix_tokens), dim=-2
            )
            k_prefix, k_patch = k.split(
                (num_prefix_tokens, tokens - num_prefix_tokens), dim=-2
            )
            q = torch.cat(
                (q_prefix, q_patch * cos + rotate_half(q_patch) * sin), dim=-2
            )
            k = torch.cat(
                (k_prefix, k_patch * cos + rotate_half(k_patch) * sin), dim=-2
            )
        else:
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

        token_count = int(tokens)
        head_count = int(self.num_heads)
        scale = float(self.head_dim) ** -0.5
        head_outputs = []
        for head_start in range(0, head_count, attention_head_chunk):
            head_end = min(head_start + attention_head_chunk, head_count)
            keys = k[:, head_start:head_end].to(dtype=torch.float32)
            values = v[:, head_start:head_end].to(dtype=torch.float32)
            transposed_keys = keys.transpose(-2, -1)
            query_outputs = []
            for query_start in range(0, token_count, attention_query_chunk):
                query_end = min(query_start + attention_query_chunk, token_count)
                queries = q[:, head_start:head_end, query_start:query_end].to(
                    dtype=torch.float32
                )
                scores = torch.matmul(queries, transposed_keys) * scale
                probabilities = torch.softmax(scores, dim=-1)
                query_outputs.append(torch.matmul(probabilities, values))
            head_outputs.append(torch.cat(query_outputs, dim=-2))
        out = torch.cat(head_outputs, dim=1).to(dtype=x.dtype)
        out = out.transpose(1, 2).contiguous().reshape(batch, tokens, channels)
        return project_tokens(self.o_proj, out)

    def chunked_mlp_forward(self: Any, x: Any) -> Any:
        token_count = int(x.shape[1])
        outputs = []
        for start in range(0, token_count, linear_token_chunk):
            chunk = x[:, start : min(start + linear_token_chunk, token_count)]
            outputs.append(
                self.down_proj(
                    torch.nn.functional.silu(self.gate_proj(chunk)) * self.up_proj(chunk)
                )
            )
        return torch.cat(outputs, dim=1)

    for module in encoder.modules():
        if isinstance(module, attention_type):
            module.forward = MethodType(chunked_forward, module)
            adapted += 1
        elif isinstance(module, mlp_type):
            module.forward = MethodType(chunked_mlp_forward, module)
            adapted_mlps += 1
    if adapted == 0:
        raise RuntimeError("No official DinoV3Attention modules were adapted")
    if adapted_mlps == 0:
        raise RuntimeError("No official DinoV3MLP modules were adapted")
    return {
        "attention_modules": adapted,
        "mlp_modules": adapted_mlps,
        "attention_query_chunk": attention_query_chunk,
        "attention_head_chunk": attention_head_chunk,
        "linear_token_chunk": linear_token_chunk,
    }


def choose_torch_device(torch: Any, requested: str) -> Any:
    """Resolve a CLI device name while validating accelerator availability."""

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


def release_torch_cache(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def normalize_rgb01(image_rgb: Any) -> Any:
    """Apply the official torchvision/ImageNet DINOv3 normalization in FP32."""

    import numpy as np

    image = np.asarray(image_rgb, dtype=np.float32)
    if tuple(image.shape) != INPUT_SHAPE:
        raise ValueError(f"image_rgb has shape {tuple(image.shape)}; expected {INPUT_SHAPE}")
    if not np.isfinite(image).all():
        raise ValueError("image_rgb contains NaN or infinity")
    minimum, maximum = float(image.min()), float(image.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"image_rgb must be in [0, 1], observed [{minimum:.8g}, {maximum:.8g}]"
        )
    mean = np.asarray(DINOV3_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(DINOV3_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((image - mean) / std, dtype=np.float32)


def validate_pixel_values(pixel_values: Any) -> None:
    """Enforce the normalized fixed-shape public input contract."""

    import numpy as np

    values = np.asarray(pixel_values)
    if tuple(values.shape) != INPUT_SHAPE:
        raise ValueError(f"pixel_values has shape {tuple(values.shape)}; expected {INPUT_SHAPE}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"pixel_values dtype is {values.dtype}; expected a floating dtype")
    if not np.isfinite(values).all():
        raise ValueError("pixel_values contains NaN or infinity")

    # A normalized image derived from RGB [0,1] has channel-specific bounds.
    tolerance = 5e-5
    for channel, (mean, std) in enumerate(zip(DINOV3_MEAN, DINOV3_STD)):
        channel_min = float(values[:, channel].min())
        channel_max = float(values[:, channel].max())
        allowed_min = (0.0 - mean) / std
        allowed_max = (1.0 - mean) / std
        if channel_min < allowed_min - tolerance or channel_max > allowed_max + tolerance:
            raise ValueError(
                f"pixel_values channel {channel} range [{channel_min:.8g}, "
                f"{channel_max:.8g}] is outside normalized RGB bounds "
                f"[{allowed_min:.8g}, {allowed_max:.8g}]"
            )


def deterministic_pixel_values() -> Any:
    """Create a stable, nontrivial normalized FP32 image for parity validation."""

    import numpy as np

    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float32)
    xx /= np.float32(IMAGE_SIZE - 1)
    yy /= np.float32(IMAGE_SIZE - 1)
    radial = np.sqrt((xx - np.float32(0.52)) ** 2 + (yy - np.float32(0.47)) ** 2)
    image = np.stack(
        (
            np.clip(xx * np.float32(0.8) + (radial < 0.31) * np.float32(0.2), 0.0, 1.0),
            np.clip(yy * np.float32(0.7) + (radial < 0.24) * np.float32(0.3), 0.0, 1.0),
            np.clip(
                (1.0 - xx) * (1.0 - yy) + (radial < 0.16) * np.float32(0.4),
                0.0,
                1.0,
            ),
        ),
        axis=0,
    )[None, ...].astype(np.float32, copy=False)
    return normalize_rgb01(image)


def load_input_fixture(path: Path) -> tuple[Any, Any | None, str]:
    """Load normalized input, or normalize an existing shared RGB fixture."""

    import numpy as np

    fixture_path = resolved_file(path, "DINOv3 input fixture")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        files = set(fixture.files)
        if "pixel_values" in files:
            pixel_values = np.asarray(fixture["pixel_values"], dtype=np.float32)
            source_key = "pixel_values"
        elif "dinov3" in files:
            pixel_values = np.asarray(fixture["dinov3"], dtype=np.float32)
            source_key = "dinov3"
        else:
            image_key = (
                "image_rgb"
                if "image_rgb" in files
                else "image"
                if "image" in files
                else None
            )
            if image_key is None:
                raise KeyError(
                    f"{fixture_path} must contain pixel_values/dinov3 or image_rgb/image; "
                    f"found {fixture.files}"
                )
            pixel_values = normalize_rgb01(fixture[image_key])
            source_key = f"{image_key} (normalized by validator)"
        expected = (
            np.asarray(fixture[OUTPUT_NAME], dtype=np.float32)
            if OUTPUT_NAME in files
            else None
        )

    validate_pixel_values(pixel_values)
    pixel_values = np.ascontiguousarray(pixel_values, dtype=np.float32)
    if expected is not None:
        if tuple(expected.shape) != OUTPUT_SHAPE:
            raise ValueError(
                f"Fixture {OUTPUT_NAME} has shape {tuple(expected.shape)}; expected {OUTPUT_SHAPE}"
            )
        if not np.isfinite(expected).all():
            raise ValueError(f"Fixture {OUTPUT_NAME} contains NaN or infinity")
        expected = np.ascontiguousarray(expected, dtype=np.float32)
    return pixel_values, expected, source_key


def array_summary(array: Any) -> dict[str, float]:
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "l2_norm": float(np.linalg.norm(values.ravel())),
    }


def comparison_metrics(reference: Any, candidate: Any, atol: float, rtol: float) -> dict[str, Any]:
    """Compute strict allclose plus diagnostic parity metrics."""

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
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "p99_absolute_error": float(np.percentile(absolute, 99)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_relative_error_at_1e-6_floor": float(relative.max()),
        "mean_relative_error_at_1e-6_floor": float(relative.mean()),
        "cosine_similarity": cosine,
        "fraction_within_tolerance": float(within.mean()),
        "worst_index": list(worst_index),
        "worst_reference": float(reference64[worst_index]),
        "worst_candidate": float(candidate64[worst_index]),
        "worst_allowed_error": float(allowed[worst_index]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
