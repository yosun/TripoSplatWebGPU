"""Shared helpers for the fixed-shape TripoSplat Flux2 VAE encoder slice.

The implementation intentionally imports ``Flux2VAEEncoder`` from a caller-provided
clone of the official TripoSplat repository.  Keeping the model definition there
prevents this browser port from silently drifting away from the numerical source of
truth.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


IMAGE_SIZE = 1024
IMAGE_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
LATENT_SHAPE = (1, 32, 128, 128)
PACKED_SHAPE = (1, 128, 64, 64)
OUTPUT_SHAPE = (1, 4101, 128)

OFFICIAL_REPOSITORY_URL = "https://github.com/VAST-AI-Research/TripoSplat"


def resolved_file(path: Path, description: str) -> Path:
    """Resolve *path* and fail with a useful message when it is not a file."""

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
    # Reuse one definition if a validator imports it more than once in-process.
    loaded = sys.modules.get(module_name)
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == model_path:
        return loaded

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import specification for {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Do not retain a partially initialized model module after a dependency or
        # source error; a subsequent invocation should be able to retry cleanly.
        sys.modules.pop(module_name, None)
        raise

    if not hasattr(module, "Flux2VAEEncoder"):
        raise AttributeError(
            f"{model_path} does not define Flux2VAEEncoder; update the checkout or "
            "use the official TripoSplat repository."
        )
    return module


def load_official_encoder(
    torch: Any,
    triposplat_repo: Path,
    weights: Path,
    device: Any,
    precision: str,
) -> Any:
    """Instantiate the official encoder, load its safetensors, and place it."""

    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported precision {precision!r}; expected fp32 or fp16")
    weights_path = resolved_file(weights, "Flux2 VAE safetensors checkpoint")
    official_model = import_official_model(triposplat_repo)
    encoder = official_model.Flux2VAEEncoder()
    encoder.load_safetensors(str(weights_path))
    dtype = torch.float16 if precision == "fp16" else torch.float32
    return encoder.eval().to(device=device, dtype=dtype)


def make_explicit_noise_encoder(torch: Any, encoder: Any) -> Any:
    """Wrap the official encoder with explicit sampling and browser I/O semantics.

    Official ``Flux2VAEEncoder.encode(..., deterministic=False)`` samples epsilon
    internally.  An ONNX/browser graph must receive epsilon as an input so PyTorch,
    Python ORT, and ORT Web can execute the exact same stochastic sample.
    """

    class ExplicitNoiseFlux2VAEEncoder(torch.nn.Module):
        def __init__(self, official_encoder: Any) -> None:
            super().__init__()
            self.encoder = official_encoder

        def forward(self, image_rgb: Any, epsilon: Any) -> Any:
            # Browser input is RGB in [0, 1]; the official Flux2 encoder consumes
            # [-1, 1].  Everything after this line mirrors encode() and
            # triposplat.encode_image() from the official repository.
            images = image_rgb * 2.0 - 1.0
            moments = self.encoder.quant_conv(self.encoder.encoder(images))
            mean, logvar = moments.chunk(2, dim=1)
            latents = mean + torch.exp(0.5 * logvar) * epsilon

            # This export slice is deliberately fixed to one 1024x1024 image.  Use
            # literal dimensions here (rather than values read from Tensor.shape) so
            # legacy ONNX tracing cannot turn the public feature2 axes symbolic.
            latents = latents.view(
                1,
                32,
                64,
                2,
                64,
                2,
            ).permute(0, 1, 3, 5, 2, 4)
            latents = latents.reshape(PACKED_SHAPE)

            bn_mean = self.encoder.bn.running_mean.view(1, -1, 1, 1).to(
                device=latents.device,
                dtype=latents.dtype,
            )
            bn_std = torch.sqrt(
                self.encoder.bn.running_var.view(1, -1, 1, 1).to(
                    device=latents.device,
                    dtype=latents.dtype,
                )
                + self.encoder.bn.eps
            )
            tokens = ((latents - bn_mean) / bn_std).to(torch.float32)
            tokens = tokens.flatten(2).transpose(1, 2).contiguous()

            # DINOv3 has one class token and four register tokens.  The VAE patch
            # sequence is aligned by prepending the same five empty positions.
            zero_prefix = torch.zeros((1, 5, 128), dtype=tokens.dtype, device=tokens.device)
            return torch.cat((zero_prefix, tokens), dim=1).reshape(OUTPUT_SHAPE)

    return ExplicitNoiseFlux2VAEEncoder(encoder)


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
    """Synchronize accelerator work before collecting a validation duration."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def tensor_type_to_precision(ort_type: str) -> str:
    """Map an ONNX Runtime input type string to the exporter precision name."""

    if ort_type == "tensor(float)":
        return "fp32"
    if ort_type == "tensor(float16)":
        return "fp16"
    raise ValueError(
        f"Unsupported ONNX input type {ort_type!r}; expected tensor(float) or tensor(float16)"
    )


def deterministic_inputs(seed: int = 20260516) -> tuple[Any, Any]:
    """Return deterministic float32 image/epsilon arrays for parity smoke tests."""

    import numpy as np

    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float32)
    xx /= np.float32(IMAGE_SIZE - 1)
    yy /= np.float32(IMAGE_SIZE - 1)
    radial = np.sqrt((xx - np.float32(0.52)) ** 2 + (yy - np.float32(0.47)) ** 2)
    image = np.stack(
        (
            np.clip(xx * np.float32(0.8) + (radial < 0.31) * np.float32(0.2), 0.0, 1.0),
            np.clip(yy * np.float32(0.7) + (radial < 0.24) * np.float32(0.3), 0.0, 1.0),
            np.clip((1.0 - xx) * (1.0 - yy) + (radial < 0.16) * np.float32(0.4), 0.0, 1.0),
        ),
        axis=0,
    )[None, ...].astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    epsilon = rng.standard_normal(LATENT_SHAPE, dtype=np.float32)
    return image, epsilon


def load_input_fixture(path: Path) -> tuple[Any, Any, Any | None]:
    """Load a strict NPZ fixture used by both Python and browser parity tests."""

    import numpy as np

    fixture_path = resolved_file(path, "Input fixture")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        image_key = "image_rgb" if "image_rgb" in fixture.files else "image"
        if image_key not in fixture.files or "epsilon" not in fixture.files:
            raise KeyError(
                f"{fixture_path} must contain image_rgb (or image) and epsilon arrays; "
                f"found {fixture.files}"
            )
        image = np.asarray(fixture[image_key], dtype=np.float32)
        epsilon = np.asarray(fixture["epsilon"], dtype=np.float32)
        expected = (
            np.asarray(fixture["feature2"], dtype=np.float32)
            if "feature2" in fixture.files
            else None
        )
    validate_input_arrays(image, epsilon)
    if expected is not None and tuple(expected.shape) != OUTPUT_SHAPE:
        raise ValueError(
            f"Fixture feature2 has shape {tuple(expected.shape)}; expected {OUTPUT_SHAPE}"
        )
    return image, epsilon, expected


def validate_input_arrays(image: Any, epsilon: Any) -> None:
    """Validate the fixed graph contract and image numerical range."""

    import numpy as np

    if tuple(image.shape) != IMAGE_SHAPE:
        raise ValueError(f"image_rgb has shape {tuple(image.shape)}; expected {IMAGE_SHAPE}")
    if tuple(epsilon.shape) != LATENT_SHAPE:
        raise ValueError(f"epsilon has shape {tuple(epsilon.shape)}; expected {LATENT_SHAPE}")
    if not np.isfinite(image).all():
        raise ValueError("image_rgb contains NaN or infinity")
    if not np.isfinite(epsilon).all():
        raise ValueError("epsilon contains NaN or infinity")
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"image_rgb must be in [0, 1], observed range [{minimum:.8g}, {maximum:.8g}]"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON report."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
