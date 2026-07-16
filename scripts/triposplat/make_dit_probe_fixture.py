#!/usr/bin/env python3
"""Add tiny block-boundary outputs to a DiT graph and record CPU ORT references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


INPUT_SHAPES = {
    "latent": (1, 8192, 16),
    "camera": (1, 1, 5),
    "t": (1,),
    "feature1": (1, 4101, 1280),
    "feature2": (1, 4101, 128),
}
FIXTURE_FILES = (*INPUT_SHAPES, "pred_latent", "pred_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--input-fixture-dir", type=Path, required=True)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--output-fixture-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-set",
        choices=("blocks", "noise0", "context0"),
        default="blocks",
        help=(
            "Block progression, detailed first noise-refiner probes, or bounded "
            "first context-refiner attention probes (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--inline-node-prefix",
        action="append",
        default=[],
        help=(
            "Inline external initializers consumed by nodes whose names start with this prefix. "
            "May be repeated; intended for diagnosing browser external-data boundaries."
        ),
    )
    parser.add_argument("--session-threads", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def replace_with_hard_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        destination.unlink()
    os.link(source, destination)


def probe_sources(
    probe_set: str,
    available: set[str],
    shapes: dict[str, list[int]],
) -> list[tuple[str, str, list[int] | None]]:
    vector = [1, 4, 16]
    if probe_set == "context0":
        context_vector = [1, 4, 1024]
        attention_rows = [1, 16, 4, 4101]
        return [
            (
                "context0_block_input",
                "/flow_model/context_refiner.0/norm1/Cast_output_0",
                context_vector,
            ),
            (
                "context0_attention_input",
                "/flow_model/context_refiner.0/norm1/Cast_1_output_0",
                context_vector,
            ),
            (
                "context0_qkv",
                "/flow_model/context_refiner.0/attn/Reshape_output_0",
                None,
            ),
            (
                "context0_q_after_rope",
                "/flow_model/context_refiner.0/attn/Reshape_2_output_0",
                None,
            ),
            (
                "context0_k_after_rope",
                "/flow_model/context_refiner.0/attn/Reshape_4_output_0",
                None,
            ),
            (
                "context0_v",
                "/flow_model/context_refiner.0/attn/Squeeze_2_output_0",
                None,
            ),
            (
                "context0_q_normalized",
                "/flow_model/context_refiner.0/attn/q_norm/Slice_1_output_0",
                None,
            ),
            (
                "context0_k_normalized",
                "/flow_model/context_refiner.0/attn/k_norm/Slice_1_output_0",
                None,
            ),
            (
                "context0_q_transposed",
                "/flow_model/context_refiner.0/attn/Transpose_output_0",
                attention_rows[:-1] + [64],
            ),
            (
                "context0_k_transposed",
                "/flow_model/context_refiner.0/attn/Transpose_1_output_0",
                [1, 16, 4101, 64],
            ),
            (
                "context0_v_transposed",
                "/flow_model/context_refiner.0/attn/Transpose_2_output_0",
                [1, 16, 4101, 64],
            ),
            (
                "context0_scaled_logits_rows",
                "/flow_model/context_refiner.0/attn/MatMul_output_0",
                attention_rows,
            ),
            (
                "context0_probabilities_rows",
                "/flow_model/context_refiner.0/attn/Softmax_output_0",
                attention_rows,
            ),
            (
                "context0_weighted_value_rows",
                "/flow_model/context_refiner.0/attn/MatMul_1_output_0",
                [1, 16, 4, 64],
            ),
            (
                "context0_pre_projection",
                "/flow_model/context_refiner.0/attn/Reshape_5_output_0",
                context_vector,
            ),
            (
                "context0_post_projection",
                "/flow_model/context_refiner.0/attn/out/Reshape_output_0",
                context_vector,
            ),
            (
                "context0_residual",
                "/flow_model/context_refiner.0/Add_output_0",
                context_vector,
            ),
        ]
    if probe_set == "noise0":
        attention = [1, 2, 4, 8]
        full_attention_candidates = [
            name
            for name, shape in shapes.items()
            if name.startswith("/flow_model/noise_refiner.0/attn/Concat")
            and len(shape) == 4
            and shape[1:] == [16, 8192, 64]
        ]
        if len(full_attention_candidates) != 1:
            raise ValueError(
                "Expected exactly one full 16-head attention concatenation, found "
                f"{full_attention_candidates}"
            )
        full_attention = full_attention_candidates[0]
        attention_head_groups = [
            name
            for name, shape in shapes.items()
            if name.startswith("/flow_model/noise_refiner.0/attn/Concat")
            and len(shape) == 4
            and shape[1:] == [8, 8192, 64]
        ]
        projected = next(
            candidate
            for candidate in (
                "/flow_model/noise_refiner.0/attn/out/Reshape_1_output_0",
                "/flow_model/noise_refiner.0/attn/out/Reshape_output_0",
                "/flow_model/noise_refiner.0/attn/out/Add_output_0",
            )
            if candidate in available
        )
        result = [
            ("input_positioned", "/flow_model/Add_1_output_0", vector),
            ("noise0_modulated", "/flow_model/noise_refiner.0/Add_3_output_0", vector),
            ("noise0_qkv", "/flow_model/noise_refiner.0/attn/qkv/Add_output_0", vector),
            ("noise0_qkv_full", "/flow_model/noise_refiner.0/attn/qkv/Add_output_0", None),
            ("noise0_q_norm", "/flow_model/noise_refiner.0/attn/q_norm/Cast_1_output_0", attention),
            ("noise0_q_norm_full", "/flow_model/noise_refiner.0/attn/q_norm/Cast_1_output_0", None),
            ("noise0_k_norm", "/flow_model/noise_refiner.0/attn/k_norm/Cast_1_output_0", attention),
            ("noise0_k_norm_full", "/flow_model/noise_refiner.0/attn/k_norm/Cast_1_output_0", None),
            ("noise0_q_transposed", "/flow_model/noise_refiner.0/attn/Transpose_output_0", attention),
            ("noise0_q_transposed_full", "/flow_model/noise_refiner.0/attn/Transpose_output_0", None),
            ("noise0_k_transposed_full", "/flow_model/noise_refiner.0/attn/Transpose_1_output_0", None),
            ("noise0_v_transposed_full", "/flow_model/noise_refiner.0/attn/Transpose_2_output_0", None),
            ("noise0_q_chunk0", "/flow_model/noise_refiner.0/attn/Cast_6_output_0", attention),
            ("noise0_score0", "/flow_model/noise_refiner.0/attn/MatMul_output_0", attention),
            ("noise0_softmax0", "/flow_model/noise_refiner.0/attn/Softmax_output_0", attention),
            ("noise0_value0", "/flow_model/noise_refiner.0/attn/MatMul_1_output_0", attention),
            ("noise0_attention_all", full_attention, attention),
            ("noise0_attention_reshaped", "/flow_model/noise_refiner.0/attn/Reshape_5_output_0", vector),
            ("noise0_attention_projected", projected, vector),
            ("noise0_attention_gated", "/flow_model/noise_refiner.0/Mul_7_output_0", vector),
            ("noise0_attention_residual", "/flow_model/noise_refiner.0/Add_4_output_0", vector),
            ("noise0_mlp_normalized", "/flow_model/noise_refiner.0/norm2/Cast_1_output_0", vector),
            ("noise0_mlp_hidden", "/flow_model/noise_refiner.0/mlp/mlp/mlp.1/Gelu_output_0", vector),
            ("noise0_output", "/flow_model/noise_refiner.0/Add_7_output_0", vector),
        ]
        if (
            "/flow_model/noise_refiner.0/attn/out/MatMul_output_0" in available
            and "/flow_model/noise_refiner.0/attn/out/Reshape_1_output_0" in available
        ):
            projection_steps = [
                (
                    f"noise0_projection_input_tile_{index}",
                    (
                        "/flow_model/noise_refiner.0/attn/out/Slice_output_0"
                        if index == 0
                        else f"/flow_model/noise_refiner.0/attn/out/Slice_{index}_output_0"
                    ),
                    [4, 8],
                )
                for index in range(4)
            ]
            projection_steps.extend(
                (
                    f"noise0_projection_partial_{index}",
                    (
                        "/flow_model/noise_refiner.0/attn/out/MatMul_output_0"
                        if index == 0
                        else f"/flow_model/noise_refiner.0/attn/out/MatMul_{index}_output_0"
                    ),
                    [4, 8],
                )
                for index in range(4)
            )
            projection_steps.extend(
                (
                    name,
                    source,
                    [4, 8],
                )
                for name, source in (
                    ("noise0_projection_sum_01", "/flow_model/noise_refiner.0/attn/out/Add_output_0"),
                    ("noise0_projection_sum_23", "/flow_model/noise_refiner.0/attn/out/Add_1_output_0"),
                    ("noise0_projection_sum", "/flow_model/noise_refiner.0/attn/out/Add_2_output_0"),
                    ("noise0_projection_bias", "/flow_model/noise_refiner.0/attn/out/Add_3_output_0"),
                    ("noise0_projection_cast", "/flow_model/noise_refiner.0/attn/out/Cast_4_output_0"),
                )
            )
            result[12:12] = projection_steps
        elif "/flow_model/noise_refiner.0/attn/out/MatMul_output_0" in available:
            result[12:12] = [
                (
                    "noise0_projection_channels_first",
                    "/flow_model/noise_refiner.0/attn/out/Slice_output_0",
                    [1, 8, 4],
                ),
                (
                    "noise0_projection_weight_left",
                    "/flow_model/noise_refiner.0/attn/out/MatMul_output_0",
                    [1, 8, 4],
                ),
                (
                    "noise0_projection_weight_left_transposed",
                    "/flow_model/noise_refiner.0/attn/out/Transpose_1_output_0",
                    [1, 4, 8],
                ),
            ]
        elif shapes.get("/flow_model/noise_refiner.0/attn/out/Add_9_output_0") == [1, 1, 256, 256]:
            fixed_256_tree = [
                *[
                    (
                        f"noise0_attention_head_group_{index}_full",
                        source,
                        None,
                    )
                    for index, source in enumerate(attention_head_groups)
                ],
                (
                    "noise0_attention_all_full",
                    full_attention,
                    None,
                ),
                (
                    "noise0_attention_reshaped_full",
                    "/flow_model/noise_refiner.0/attn/Reshape_5_output_0",
                    None,
                ),
                (
                    "noise0_projection_first_token_full",
                    "/flow_model/noise_refiner.0/attn/out/Slice_output_0",
                    None,
                ),
                (
                    "noise0_projection_special_products",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    [1, 1, 4, 8],
                ),
                (
                    "noise0_projection_special_products_full",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    None,
                ),
                (
                    "noise0_projection_products_transposed",
                    "/flow_model/noise_refiner.0/attn/out/Transpose_2_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_products_transposed_full",
                    "/flow_model/noise_refiner.0/attn/out/Transpose_2_output_0",
                    None,
                ),
                *[
                    (
                        f"noise0_projection_tree_{index}",
                        (
                            "/flow_model/noise_refiner.0/attn/out/Add_output_0"
                            if index == 0
                            else f"/flow_model/noise_refiner.0/attn/out/Add_{index}_output_0"
                        ),
                        [1, 1, 4, 16],
                    )
                    for index in range(10)
                ],
                (
                    "noise0_projection_tree_0_full",
                    "/flow_model/noise_refiner.0/attn/out/Add_output_0",
                    None,
                ),
                (
                    "noise0_projection_tree_1_full",
                    "/flow_model/noise_refiner.0/attn/out/Add_1_output_0",
                    None,
                ),
                (
                    "noise0_projection_fixed_gather_left",
                    "/flow_model/noise_refiner.0/attn/out/Gather_1_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_fixed_gather_right",
                    "/flow_model/noise_refiner.0/attn/out/Gather_2_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_special_sum",
                    "/flow_model/noise_refiner.0/attn/out/Gather_17_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_special_bias",
                    "/flow_model/noise_refiner.0/attn/out/Add_10_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_bulk_tail",
                    "/flow_model/noise_refiner.0/attn/out/Gather_output_0",
                    [1, 4, 16],
                ),
                (
                    "noise0_projection_corrected_chunk",
                    "/flow_model/noise_refiner.0/attn/out/Concat_output_0",
                    [1, 4, 16],
                ),
            ]
            result[12:12] = fixed_256_tree
        elif shapes.get("/flow_model/noise_refiner.0/attn/out/Add_9_output_0") == [1, 1, 128, 256]:
            fixed_tree = [
                (
                    "noise0_projection_special_products",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    [1, 1, 4, 8],
                ),
                (
                    "noise0_projection_products_transposed",
                    "/flow_model/noise_refiner.0/attn/out/Transpose_2_output_0",
                    [1, 1, 4, 16],
                ),
                *[
                    (
                        f"noise0_projection_tree_{index}",
                        (
                            "/flow_model/noise_refiner.0/attn/out/Add_output_0"
                            if index == 0
                            else f"/flow_model/noise_refiner.0/attn/out/Add_{index}_output_0"
                        ),
                        [1, 1, 4, 16],
                    )
                    for index in range(10)
                ],
                (
                    "noise0_projection_tree_2_full",
                    "/flow_model/noise_refiner.0/attn/out/Add_2_output_0",
                    None,
                ),
                (
                    "noise0_projection_fixed_gather_left",
                    "/flow_model/noise_refiner.0/attn/out/Gather_1_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_fixed_gather_right",
                    "/flow_model/noise_refiner.0/attn/out/Gather_2_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_fixed_masked_left",
                    "/flow_model/noise_refiner.0/attn/out/Mul_1_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_fixed_masked_right",
                    "/flow_model/noise_refiner.0/attn/out/Mul_2_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_special_sum",
                    "/flow_model/noise_refiner.0/attn/out/Gather_15_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_special_bias",
                    "/flow_model/noise_refiner.0/attn/out/Add_10_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_bulk_tail",
                    "/flow_model/noise_refiner.0/attn/out/Gather_output_0",
                    [1, 4, 16],
                ),
                (
                    "noise0_projection_corrected_chunk",
                    "/flow_model/noise_refiner.0/attn/out/Concat_output_0",
                    [1, 4, 16],
                ),
            ]
            result[12:12] = fixed_tree
        elif "/flow_model/noise_refiner.0/attn/out/Gather_15_output_0" in available:
            gathered_tree = [
                (
                    "noise0_projection_special_products",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    [1, 1, 4, 8],
                ),
                *[
                    (
                        f"noise0_projection_tree_{index}",
                        (
                            "/flow_model/noise_refiner.0/attn/out/Add_output_0"
                            if index == 0
                            else f"/flow_model/noise_refiner.0/attn/out/Add_{index}_output_0"
                        ),
                        [1, 1, 4, min(8, 512 >> index)] if index < 3 else [1, 1, 4, 128],
                    )
                    for index in range(10)
                ],
                (
                    "noise0_projection_gathered_left",
                    "/flow_model/noise_refiner.0/attn/out/Gather_1_output_0",
                    [1, 1, 4, 8],
                ),
                (
                    "noise0_projection_gathered_right",
                    "/flow_model/noise_refiner.0/attn/out/Gather_2_output_0",
                    [1, 1, 4, 8],
                ),
                (
                    "noise0_projection_padded_left",
                    "/flow_model/noise_refiner.0/attn/out/Concat_output_0",
                    [1, 1, 4, 128],
                ),
                (
                    "noise0_projection_padded_right",
                    "/flow_model/noise_refiner.0/attn/out/Concat_1_output_0",
                    [1, 1, 4, 128],
                ),
                (
                    "noise0_projection_special_sum",
                    "/flow_model/noise_refiner.0/attn/out/Gather_15_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_special_bias",
                    "/flow_model/noise_refiner.0/attn/out/Add_10_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_bulk_tail",
                    "/flow_model/noise_refiner.0/attn/out/Gather_output_0",
                    [1, 4, 16],
                ),
                (
                    "noise0_projection_corrected_chunk",
                    "/flow_model/noise_refiner.0/attn/out/Concat_14_output_0",
                    [1, 4, 16],
                ),
            ]
            result[12:12] = gathered_tree
        elif "/flow_model/noise_refiner.0/attn/out/Concat_14_output_0" in available:
            padded_tree = [
                (
                    "noise0_projection_special_products",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    [1, 1, 4, 8],
                ),
                *[
                    (
                        f"noise0_projection_tree_{index}",
                        (
                            "/flow_model/noise_refiner.0/attn/out/Add_output_0"
                            if index == 0
                            else f"/flow_model/noise_refiner.0/attn/out/Add_{index}_output_0"
                        ),
                        [1, 1, 4, min(8, 512 >> index)] if index < 3 else [1, 1, 4, 128],
                    )
                    for index in range(10)
                ],
                (
                    "noise0_projection_padded_left",
                    "/flow_model/noise_refiner.0/attn/out/Concat_output_0",
                    [1, 1, 4, 128],
                ),
                (
                    "noise0_projection_padded_right",
                    "/flow_model/noise_refiner.0/attn/out/Concat_1_output_0",
                    [1, 1, 4, 128],
                ),
                (
                    "noise0_projection_special_sum",
                    "/flow_model/noise_refiner.0/attn/out/Gather_1_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_special_bias",
                    "/flow_model/noise_refiner.0/attn/out/Add_10_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_bulk_tail",
                    "/flow_model/noise_refiner.0/attn/out/Gather_output_0",
                    [1, 4, 16],
                ),
                (
                    "noise0_projection_corrected_chunk",
                    "/flow_model/noise_refiner.0/attn/out/Concat_14_output_0",
                    [1, 4, 16],
                ),
            ]
            result[12:12] = padded_tree
        elif shapes.get("/flow_model/noise_refiner.0/attn/out/Add_9_output_0") == [1, 1, 1, 256]:
            transposed_tree = [
                (
                    "noise0_projection_special_products",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    [1, 1, 4, 8],
                ),
                (
                    "noise0_projection_products_transposed",
                    "/flow_model/noise_refiner.0/attn/out/Transpose_2_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_first_left",
                    "/flow_model/noise_refiner.0/attn/out/Slice_1_output_0",
                    [1, 1, 4, 16],
                ),
                (
                    "noise0_projection_first_right",
                    "/flow_model/noise_refiner.0/attn/out/Slice_2_output_0",
                    [1, 1, 4, 16],
                ),
                *[
                    (
                        f"noise0_projection_tree_{index}",
                        (
                            "/flow_model/noise_refiner.0/attn/out/Add_output_0"
                            if index == 0
                            else f"/flow_model/noise_refiner.0/attn/out/Add_{index}_output_0"
                        ),
                        [1, 1, min(4, 512 >> index), 16],
                    )
                    for index in range(10)
                ],
                (
                    "noise0_projection_special_sum",
                    "/flow_model/noise_refiner.0/attn/out/Squeeze_1_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_special_bias",
                    "/flow_model/noise_refiner.0/attn/out/Add_10_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_bulk_tail",
                    "/flow_model/noise_refiner.0/attn/out/Gather_output_0",
                    [1, 4, 16],
                ),
                (
                    "noise0_projection_corrected_chunk",
                    "/flow_model/noise_refiner.0/attn/out/Concat_output_0",
                    [1, 4, 16],
                ),
            ]
            result[12:12] = transposed_tree
        elif "/flow_model/noise_refiner.0/attn/out/Add_9_output_0" in available:
            tree_levels = [
                (
                    f"noise0_projection_tree_{index}",
                    (
                        "/flow_model/noise_refiner.0/attn/out/Add_output_0"
                        if index == 0
                        else f"/flow_model/noise_refiner.0/attn/out/Add_{index}_output_0"
                    ),
                    [1, 1, 4, min(8, 512 >> index)],
                )
                for index in range(10)
            ]
            result[12:12] = [
                (
                    "noise0_projection_special_products",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    [1, 1, 4, 8],
                ),
                *tree_levels,
                (
                    "noise0_projection_special_sum",
                    "/flow_model/noise_refiner.0/attn/out/Squeeze_1_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_special_bias",
                    "/flow_model/noise_refiner.0/attn/out/Add_10_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_bulk_tail",
                    "/flow_model/noise_refiner.0/attn/out/Gather_output_0",
                    [1, 4, 16],
                ),
                (
                    "noise0_projection_corrected_chunk",
                    "/flow_model/noise_refiner.0/attn/out/Concat_output_0",
                    [1, 4, 16],
                ),
            ]
        elif "/flow_model/noise_refiner.0/attn/out/ReduceSum_output_0" in available:
            result[12:12] = [
                (
                    "noise0_projection_special_products",
                    "/flow_model/noise_refiner.0/attn/out/Mul_output_0",
                    [1, 1, 4, 8],
                ),
                (
                    "noise0_projection_special_sum",
                    "/flow_model/noise_refiner.0/attn/out/ReduceSum_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_special_bias",
                    "/flow_model/noise_refiner.0/attn/out/Add_output_0",
                    [1, 1, 16],
                ),
                (
                    "noise0_projection_bulk_tail",
                    "/flow_model/noise_refiner.0/attn/out/Gather_output_0",
                    [1, 4, 16],
                ),
                (
                    "noise0_projection_corrected_chunk",
                    "/flow_model/noise_refiner.0/attn/out/Concat_output_0",
                    [1, 4, 16],
                ),
            ]
        return result

    result = [
        ("input_linear", "/flow_model/input_layer/Add_output_0", vector),
        ("input_positioned", "/flow_model/Add_1_output_0", vector),
        ("condition_dino", "/flow_model/cond_embedder/Add_output_0", vector),
        ("condition_vae", "/flow_model/cond_embedder2/Add_output_0", vector),
        ("condition_combined", "/flow_model/Add_output_0", vector),
        (
            "noise_refiner_00_attention_residual",
            "/flow_model/noise_refiner.0/Add_4_output_0",
            vector,
        ),
        ("noise_refiner_00", "/flow_model/noise_refiner.0/Add_7_output_0", vector),
        (
            "noise_refiner_01_attention_residual",
            "/flow_model/noise_refiner.1/Add_4_output_0",
            vector,
        ),
        ("noise_refiner_01", "/flow_model/noise_refiner.1/Add_7_output_0", vector),
        (
            "context_refiner_00_attention_residual",
            "/flow_model/context_refiner.0/Add_output_0",
            vector,
        ),
        ("context_refiner_00", "/flow_model/context_refiner.0/Add_1_output_0", vector),
        (
            "context_refiner_01_attention_residual",
            "/flow_model/context_refiner.1/Add_output_0",
            vector,
        ),
        ("context_refiner_01", "/flow_model/context_refiner.1/Add_1_output_0", vector),
        ("combined_tokens", "/flow_model/Concat_output_0", vector),
        (
            "block_00_attention_residual",
            "/flow_model/blocks.0/Add_4_output_0",
            vector,
        ),
    ]
    result.extend(
        (f"block_{index:02d}", f"/flow_model/blocks.{index}/Add_7_output_0", vector)
        for index in range(24)
    )
    return result


def external_data_fields(tensor: Any) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def inline_selected_initializers(model: Any, sidecar: Path, node_prefixes: list[str]) -> list[str]:
    if not node_prefixes:
        return []
    selected_inputs = {
        input_name
        for node in model.graph.node
        if any(node.name.startswith(prefix) for prefix in node_prefixes)
        for input_name in node.input
    }
    inlined: list[str] = []
    with sidecar.open("rb") as stream:
        for tensor in model.graph.initializer:
            if tensor.name not in selected_inputs or not tensor.external_data:
                continue
            fields = external_data_fields(tensor)
            location = fields.get("location")
            if location and Path(location).name != sidecar.name:
                raise ValueError(
                    f"Initializer {tensor.name!r} points to {location!r}, not {sidecar.name!r}."
                )
            offset = int(fields.get("offset", "0"))
            length = int(fields["length"])
            stream.seek(offset)
            raw_data = stream.read(length)
            if len(raw_data) != length:
                raise EOFError(
                    f"Initializer {tensor.name!r} read {len(raw_data)} bytes at {offset}; expected {length}."
                )
            tensor.raw_data = raw_data
            tensor.ClearField("external_data")
            tensor.data_location = 0
            inlined.append(tensor.name)
    if not inlined:
        raise ValueError(f"No external initializers matched node prefixes: {node_prefixes}")
    return inlined


def main() -> None:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import helper
    except ImportError as exc:
        raise SystemExit(f"ONNX, ONNX Runtime, and NumPy are required: {exc}") from exc

    args = parse_args()
    graph = args.graph.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    fixture_dir = args.input_fixture_dir.expanduser().resolve()
    output_graph = args.output_graph.expanduser().resolve()
    output_fixture = args.output_fixture_dir.expanduser().resolve()
    for path in (graph, sidecar):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = onnx.load_model(str(graph), load_external_data=False)
    available = {value for node in model.graph.node for value in node.output}
    inferred_model = onnx.shape_inference.infer_shapes(model)
    shapes = {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in (*inferred_model.graph.value_info, *inferred_model.graph.output)
        if value.type.HasField("tensor_type")
    }
    # ONNX shape inference loses the fixed batch value at attention
    # concatenations created by the statically unrolled chunk loops. All public
    # inputs are fixed to batch one, so make that contract explicit on probes.
    for shape in shapes.values():
        if shape and shape[0] == 0:
            shape[0] = 1
    sources = probe_sources(args.probe_set, available, shapes)
    missing = [source for _, source, _ in sources if source not in available]
    if missing:
        raise KeyError(f"Graph is missing probe values: {missing}")

    inlined_initializers = inline_selected_initializers(model, sidecar, args.inline_node_prefix)

    include_final_outputs = args.probe_set == "blocks"
    if not include_final_outputs:
        del model.graph.output[:]

    for index, (name, source, ends) in enumerate(sources):
        prefix = f"triposplat_probe_{index:02d}"
        output_shape = shapes[source] if ends is None else ends
        cast_input = source
        if ends is not None:
            constants = {
                "starts": [0] * len(ends),
                "ends": ends,
                "axes": list(range(len(ends))),
                "steps": [1] * len(ends),
            }
            inputs = []
            for suffix, values in constants.items():
                initializer_name = f"{prefix}_{suffix}"
                model.graph.initializer.append(
                    helper.make_tensor(
                        initializer_name,
                        onnx.TensorProto.INT64,
                        [len(values)],
                        values,
                    )
                )
                inputs.append(initializer_name)
            slice_output = f"{prefix}_slice"
            model.graph.node.append(
                helper.make_node(
                    "Slice",
                    [source, *inputs],
                    [slice_output],
                    name=f"/{prefix}/Slice",
                )
            )
            cast_input = slice_output
        output_name = f"probe_{name}"
        model.graph.node.append(
            helper.make_node(
                "Cast",
                [cast_input],
                [output_name],
                name=f"/{prefix}/Cast",
                to=onnx.TensorProto.FLOAT,
            )
        )
        model.graph.output.append(
            helper.make_tensor_value_info(
                output_name,
                onnx.TensorProto.FLOAT,
                output_shape,
            )
        )

    sidecar_name = output_graph.name + ".data"
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = sidecar_name
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["triposplat.diagnostic_probes"] = str(len(sources))
    if inlined_initializers:
        metadata["triposplat.diagnostic_inlined_initializers"] = ",".join(inlined_initializers)
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    output_graph.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_graph), save_as_external_data=False)
    output_sidecar = output_graph.parent / sidecar_name
    replace_with_hard_link(sidecar, output_sidecar)

    output_fixture.mkdir(parents=True, exist_ok=True)
    for name in FIXTURE_FILES:
        replace_with_hard_link(fixture_dir / f"{name}.f32", output_fixture / f"{name}.f32")

    feeds = {
        name: np.fromfile(fixture_dir / f"{name}.f32", dtype="<f4").reshape(shape)
        for name, shape in INPUT_SHAPES.items()
    }
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.intra_op_num_threads = args.session_threads
    session = ort.InferenceSession(
        str(output_graph),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [f"probe_{name}" for name, _, _ in sources]
    outputs = session.run(output_names, feeds)
    probes: list[dict[str, Any]] = []
    for output_name, value in zip(output_names, outputs):
        array = np.ascontiguousarray(value, dtype="<f4")
        path = output_fixture / f"{output_name}.f32"
        array.tofile(path)
        probes.append(
            {
                "name": output_name,
                "path": path.name,
                "shape": list(array.shape),
                "elements": int(array.size),
                "sha256": sha256(path),
            }
        )
    manifest = {
        "format": "triposplat-dit-probes-v1",
        "graph": output_graph.name,
        "graphSha256": sha256(output_graph),
        "externalData": output_sidecar.name,
        "externalDataSha256": sha256(output_sidecar),
        "includeFinalOutputs": include_final_outputs,
        "outputs": probes,
    }
    manifest_path = output_fixture / "probes.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_graph}, {output_sidecar}, and {manifest_path}")


if __name__ == "__main__":
    main()
