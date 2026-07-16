#!/usr/bin/env python3
"""Gate one TripoSplat DiT ONNX invocation against official PyTorch.

The validator runs three paths on identical float32 public tensors:

1. untouched official complex-RoPE ``load_flow_model`` PyTorch;
2. the in-memory real-RoPE/static-position export adapter in PyTorch;
3. the exported graph in Python ONNX Runtime.

It verifies exact public names/dtypes/static shapes and every external-data range before
loading weights.  The export-adapter comparison is mandatory: the Core AI decomposition
is not treated as numerical truth.  Timings are measured Python validation calls and are
explicitly not browser WebGPU benchmark results.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from dit_common import (
    INPUT_NAMES,
    INPUT_SHAPES,
    INTERNAL_PRECISION_METADATA_KEY,
    OFFICIAL_REPOSITORY_URL,
    OUTPUT_NAMES,
    OUTPUT_SHAPES,
    adapt_official_flow_for_onnx,
    array_summary,
    choose_torch_device,
    comparison_metrics,
    deterministic_inputs,
    load_input_fixture,
    load_official_flow_model,
    make_browser_flow_step,
    metadata_dict,
    output_mapping,
    source_revision,
    synchronize_torch,
    torch_inputs_from_numpy,
    validate_input_arrays,
    validate_real_rope_primitives,
    verify_external_data_files,
    verify_onnx_contract,
    write_json,
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
        help=f"Clean local clone of {OFFICIAL_REPOSITORY_URL}.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Official flow-model checkpoint used to export the ONNX graph.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Exported fixed-shape dit_step.onnx graph.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help=(
            "Optional NPZ with latent, camera, t, feature1, and feature2 float32 "
            "arrays. Optional pred_latent/pred_camera arrays are also checked."
        ),
    )
    parser.add_argument(
        "--trajectory-fixture-dir",
        type=Path,
        help=(
            "Optional flow fixture directory containing flow.json and a recorded "
            "per-invocation trajectory. Use with --trajectory-invocation to validate "
            "the exact conditional or all-zero unconditional call."
        ),
    )
    parser.add_argument(
        "--trajectory-invocation",
        type=int,
        help="One-based invocation from --trajectory-fixture-dir/flow.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260714,
        help="Synthetic-input seed when --fixture is omitted (default: %(default)s).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1000.0,
        help=(
            "Already-scaled synthetic model timestep in [0,1000] when --fixture is "
            "omitted (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--internal-precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help="Expected internal graph precision; auto reads metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
        help="Device for both official PyTorch calls (default: %(default)s).",
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help=(
            "Python ONNX Runtime execution provider. Repeat for fallback order "
            "(default: CPUExecutionProvider)."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        help="PyTorch-vs-ORT absolute tolerance (precision-specific default).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        help="PyTorch-vs-ORT relative tolerance (precision-specific default).",
    )
    parser.add_argument(
        "--adapter-atol",
        type=float,
        help="Official-complex-vs-real-adapter absolute tolerance (strict default).",
    )
    parser.add_argument(
        "--adapter-rtol",
        type=float,
        help="Official-complex-vs-real-adapter relative tolerance (strict default).",
    )
    parser.add_argument(
        "--session-threads",
        type=int,
        default=0,
        help="ORT intra-op threads; zero leaves the runtime default (default: %(default)s).",
    )
    parser.add_argument(
        "--attention-query-chunk",
        type=int,
        metavar="TOKENS",
        help=(
            "Expected query chunk. By default this is required and read from ONNX "
            "metadata."
        ),
    )
    parser.add_argument(
        "--collapsed-unconditional-context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Optionally require collapsed unconditional-only context to be enabled or "
            "disabled. Missing metadata reconstructs the disabled canonical graph."
        ),
    )
    parser.add_argument(
        "--attention-head-chunk",
        type=int,
        metavar="HEADS",
        help=(
            "Expected head chunk. By default this is required and read from ONNX "
            "metadata."
        ),
    )
    parser.add_argument(
        "--attention-head-padding",
        type=int,
        metavar="HEADS",
        help=(
            "Expected optional head padding. By default this is read from ONNX metadata."
        ),
    )
    parser.add_argument(
        "--qk-norm-padding-tokens",
        type=int,
        metavar="TOKENS",
        help=(
            "Expected protective Q/K norm token padding. By default this is required "
            "and read from ONNX metadata."
        ),
    )
    parser.add_argument(
        "--low-memory-construction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Construct official parameters directly in graph precision (default: enabled).",
    )
    parser.add_argument(
        "--allow-dirty-official-source",
        action="store_true",
        help="Allow tracked edits to official model.py/triposplat.py.",
    )
    parser.add_argument(
        "--allow-source-commit-mismatch",
        action="store_true",
        help="Allow validator source commit to differ from graph metadata.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report destination.")
    parser.add_argument(
        "--save-fixture",
        type=Path,
        help=(
            "Optional NPZ output containing public inputs and untouched official "
            "pred_latent/pred_camera references for browser validation."
        ),
    )
    args = parser.parse_args()
    for name in ("atol", "rtol", "adapter_atol", "adapter_rtol"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.session_threads < 0:
        parser.error("--session-threads must be non-negative")
    if args.attention_query_chunk is not None and args.attention_query_chunk <= 0:
        parser.error("--attention-query-chunk must be positive")
    if args.attention_head_chunk is not None and args.attention_head_chunk <= 0:
        parser.error("--attention-head-chunk must be positive")
    if args.attention_head_padding is not None and args.attention_head_padding < 0:
        parser.error("--attention-head-padding must be non-negative")
    if args.qk_norm_padding_tokens is not None and args.qk_norm_padding_tokens <= 0:
        parser.error("--qk-norm-padding-tokens must be positive")
    if not 0.0 <= args.timestep <= 1000.0:
        parser.error("--timestep must be in [0,1000]")
    has_trajectory_dir = args.trajectory_fixture_dir is not None
    has_trajectory_invocation = args.trajectory_invocation is not None
    if has_trajectory_dir != has_trajectory_invocation:
        parser.error(
            "--trajectory-fixture-dir and --trajectory-invocation must be provided together"
        )
    if args.fixture is not None and has_trajectory_dir:
        parser.error("--fixture and --trajectory-fixture-dir are mutually exclusive")
    if has_trajectory_invocation and args.trajectory_invocation <= 0:
        parser.error("--trajectory-invocation must be positive")
    return args


def load_trajectory_invocation_fixture(
    directory: Path,
    invocation_number: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load one exact official flow call, including its implicit CFG condition.

    ``make_flow_fixture.py --record-trajectory`` stores the changing sample, time,
    and prediction per invocation. The positive conditioning tensors are shared at
    the fixture root; the official unconditional branch uses same-shaped zeros.
    Loading this form directly avoids copying hundreds of megabytes into redundant
    NPZ files while making value-dependent browser failures reproducible in the
    three-way Python validator.
    """

    import numpy as np

    fixture_dir = directory.expanduser().resolve()
    manifest_path = fixture_dir / "flow.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Flow trajectory manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectory = manifest.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"{manifest_path} has no recorded trajectory")
    matches = [
        record
        for record in trajectory
        if isinstance(record, dict) and record.get("invocation") == invocation_number
    ]
    if len(matches) != 1:
        available = [
            record.get("invocation")
            for record in trajectory
            if isinstance(record, dict)
        ]
        raise ValueError(
            f"Expected one trajectory invocation {invocation_number}, found {len(matches)}; "
            f"available={available}"
        )
    record = matches[0]
    pass_name = record.get("pass")
    if pass_name not in {"conditional", "unconditional"}:
        raise ValueError(f"Invocation {invocation_number} has invalid pass {pass_name!r}")
    tensors = record.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError(f"Invocation {invocation_number} has no tensor records")

    def load_raw(path_value: Any, shape: tuple[int, ...], label: str) -> Any:
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Invocation {invocation_number} {label} has no path")
        path = (fixture_dir / path_value).resolve()
        try:
            path.relative_to(fixture_dir)
        except ValueError as exc:
            raise ValueError(f"Trajectory tensor path escapes fixture directory: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory tensor does not exist: {path}")
        values = np.fromfile(path, dtype="<f4")
        expected = int(np.prod(shape))
        if values.size != expected:
            raise ValueError(
                f"{label} has {values.size} float32 values; expected {expected} for {shape}"
            )
        return np.ascontiguousarray(values.reshape(shape), dtype=np.float32)

    def trajectory_tensor(name: str, shape: tuple[int, ...]) -> Any:
        descriptor = tensors.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"Invocation {invocation_number} lacks tensor {name!r}")
        recorded_shape = descriptor.get("shape")
        if recorded_shape != list(shape):
            raise ValueError(
                f"Invocation {invocation_number} {name} records shape {recorded_shape}; "
                f"expected {list(shape)}"
            )
        return load_raw(descriptor.get("path"), shape, name)

    inputs = {
        "latent": trajectory_tensor("sample_latent", INPUT_SHAPES["latent"]),
        "camera": trajectory_tensor("sample_camera", INPUT_SHAPES["camera"]),
        "t": trajectory_tensor("t", INPUT_SHAPES["t"]),
    }
    for name in ("feature1", "feature2"):
        shape = INPUT_SHAPES[name]
        if pass_name == "conditional":
            inputs[name] = load_raw(f"{name}.f32", shape, name)
        else:
            inputs[name] = np.zeros(shape, dtype=np.float32)
    validate_input_arrays(inputs)
    expected = {
        "pred_latent": trajectory_tensor("pred_latent", OUTPUT_SHAPES["pred_latent"]),
        "pred_camera": trajectory_tensor("pred_camera", OUTPUT_SHAPES["pred_camera"]),
    }
    step = record.get("step")
    source = (
        f"{manifest_path} invocation={invocation_number} step={step} pass={pass_name}"
    )
    return inputs, expected, source


def require_ort_contract(session: Any) -> None:
    inputs = {value.name: value for value in session.get_inputs()}
    outputs = {value.name: value for value in session.get_outputs()}
    if set(inputs) != set(INPUT_NAMES):
        raise ValueError(f"ORT inputs are {sorted(inputs)}; expected {list(INPUT_NAMES)}")
    if set(outputs) != set(OUTPUT_NAMES):
        raise ValueError(f"ORT outputs are {sorted(outputs)}; expected {list(OUTPUT_NAMES)}")
    for name, expected_shape in (*INPUT_SHAPES.items(), *OUTPUT_SHAPES.items()):
        value = inputs[name] if name in inputs else outputs[name]
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"ORT {name} shape is {value.shape}; expected fixed {expected_shape}"
            )
        if value.type != "tensor(float)":
            raise ValueError(
                f"ORT {name} type is {value.type}; public contract requires tensor(float)"
            )


def make_session(ort: Any, graph: Path, providers: list[str], threads: int) -> Any:
    available = set(ort.get_available_providers())
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise RuntimeError(
            f"Requested ORT providers unavailable: {missing}. Available: {sorted(available)}"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # FP16 constant-fold misses are expected on the CPU optimizer and otherwise
    # produce hundreds of warning lines without affecting execution.
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(graph), sess_options=options, providers=providers)
    require_ort_contract(session)
    return session


def tensor_outputs_to_numpy(outputs: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    import numpy as np

    mapped = output_mapping(outputs)
    result = {
        name: np.ascontiguousarray(
            value.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
        for name, value in mapped.items()
    }
    for name, value in result.items():
        if tuple(value.shape) != OUTPUT_SHAPES[name]:
            raise ValueError(
                f"PyTorch {name} shape is {tuple(value.shape)}; expected {OUTPUT_SHAPES[name]}"
            )
    return result


def compare_output_maps(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], bool]:
    metrics = {
        name: comparison_metrics(reference[name], candidate[name], atol=atol, rtol=rtol)
        for name in OUTPUT_NAMES
    }
    return metrics, all(bool(item["passed"]) for item in metrics.values())


def validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing validation dependency. Install a PyTorch-supported Python version "
            "and run `python -m pip install -r scripts/triposplat/requirements.txt`. "
            f"Original error: {exc}"
        ) from exc

    graph_path = args.onnx.expanduser().resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f"ONNX graph does not exist: {graph_path}")
    graph_proto = onnx.load_model(str(graph_path), load_external_data=False)
    graph_precision = verify_onnx_contract(onnx, graph_proto, require_metadata=True)
    external_files = verify_external_data_files(graph_path, graph_proto)
    graph_metadata = metadata_dict(graph_proto)
    try:
        graph_attention_query_chunk = int(
            graph_metadata.get("triposplat.attention_query_chunk", "0")
        )
    except ValueError as exc:
        raise ValueError("Graph attention query chunk metadata is not an integer") from exc
    if graph_attention_query_chunk <= 0:
        raise ValueError(
            "Graph is missing a positive triposplat.attention_query_chunk; "
            "the unchunked graph is not a production browser artifact"
        )
    if (
        args.attention_query_chunk is not None
        and args.attention_query_chunk != graph_attention_query_chunk
    ):
        raise ValueError(
            f"Expected query chunk {args.attention_query_chunk}, graph records "
            f"{graph_attention_query_chunk}"
        )
    attention_query_chunk = graph_attention_query_chunk
    collapsed_unconditional_context_value = graph_metadata.get(
        "triposplat.collapsed_unconditional_context", "false"
    )
    if collapsed_unconditional_context_value not in {"true", "false"}:
        raise ValueError(
            "Graph collapsed unconditional-context metadata must be true or false"
        )
    collapsed_unconditional_context = (
        collapsed_unconditional_context_value == "true"
    )
    if (
        args.collapsed_unconditional_context is not None
        and args.collapsed_unconditional_context
        != collapsed_unconditional_context
    ):
        raise ValueError(
            "Expected collapsed unconditional context "
            f"{args.collapsed_unconditional_context}, graph records "
            f"{collapsed_unconditional_context}"
        )
    try:
        graph_attention_head_chunk = int(
            graph_metadata.get("triposplat.attention_head_chunk", "0")
        )
    except ValueError as exc:
        raise ValueError("Graph attention head chunk metadata is not an integer") from exc
    if graph_attention_head_chunk <= 0:
        raise ValueError(
            "Graph is missing a positive triposplat.attention_head_chunk; "
            "the 16-head WebGPU kernel is not a production-safe artifact"
        )
    if (
        args.attention_head_chunk is not None
        and args.attention_head_chunk != graph_attention_head_chunk
    ):
        raise ValueError(
            f"Expected head chunk {args.attention_head_chunk}, graph records "
            f"{graph_attention_head_chunk}"
        )
    attention_head_chunk = graph_attention_head_chunk
    if "triposplat.attention_head_padding" not in graph_metadata:
        raise ValueError("Graph is missing triposplat.attention_head_padding metadata")
    try:
        graph_attention_head_padding = int(
            graph_metadata["triposplat.attention_head_padding"]
        )
    except ValueError as exc:
        raise ValueError("Graph attention head padding metadata is not an integer") from exc
    if graph_attention_head_padding < 0:
        raise ValueError("Graph attention head padding must be non-negative")
    if (
        args.attention_head_padding is not None
        and args.attention_head_padding != graph_attention_head_padding
    ):
        raise ValueError(
            f"Expected head padding {args.attention_head_padding}, graph records "
            f"{graph_attention_head_padding}"
        )
    attention_head_padding = graph_attention_head_padding
    try:
        graph_qk_norm_padding_tokens = int(
            graph_metadata.get("triposplat.qk_norm_padding_tokens", "0")
        )
    except ValueError as exc:
        raise ValueError("Graph Q/K norm padding metadata is not an integer") from exc
    if graph_qk_norm_padding_tokens <= 0:
        raise ValueError(
            "Graph is missing positive triposplat.qk_norm_padding_tokens; "
            "the final real K normalization vector is not shielded"
        )
    if (
        args.qk_norm_padding_tokens is not None
        and args.qk_norm_padding_tokens != graph_qk_norm_padding_tokens
    ):
        raise ValueError(
            f"Expected Q/K norm padding {args.qk_norm_padding_tokens}, graph records "
            f"{graph_qk_norm_padding_tokens}"
        )
    qk_norm_padding_tokens = graph_qk_norm_padding_tokens
    try:
        attention_output_chunk = int(
            graph_metadata.get("triposplat.attention_output_chunk", "0")
        )
    except ValueError as exc:
        raise ValueError("Graph attention output chunk metadata is not an integer") from exc
    if attention_output_chunk <= 0:
        raise ValueError("Graph is missing a positive triposplat.attention_output_chunk")
    try:
        attention_output_reduction_chunk = int(
            graph_metadata.get("triposplat.attention_output_reduction_chunk", "0")
        )
    except ValueError as exc:
        raise ValueError(
            "Graph attention output reduction chunk metadata is not an integer"
        ) from exc
    if attention_output_reduction_chunk <= 0:
        raise ValueError(
            "Graph is missing a positive triposplat.attention_output_reduction_chunk"
        )
    graph_rms_norm_eps_value = graph_metadata.get("triposplat.rms_norm_eps", "disabled")
    if graph_rms_norm_eps_value == "disabled":
        rms_norm_eps = None
    else:
        try:
            rms_norm_eps = float(graph_rms_norm_eps_value)
        except ValueError as exc:
            raise ValueError("Graph RMS norm epsilon metadata is not numeric") from exc
        if rms_norm_eps <= 0:
            raise ValueError("Graph RMS norm epsilon must be positive or disabled")
    if args.internal_precision != "auto" and args.internal_precision != graph_precision:
        raise ValueError(
            f"--internal-precision {args.internal_precision} conflicts with graph "
            f"metadata ({graph_precision})"
        )
    precision = graph_precision

    if args.trajectory_fixture_dir:
        inputs, fixture_expected, input_source = load_trajectory_invocation_fixture(
            args.trajectory_fixture_dir,
            args.trajectory_invocation,
        )
    elif args.fixture:
        inputs, fixture_expected = load_input_fixture(args.fixture)
        input_source = str(args.fixture.expanduser().resolve())
    else:
        inputs = deterministic_inputs(args.seed, timestep=args.timestep)
        fixture_expected = {}
        input_source = f"deterministic synthetic inputs (seed={args.seed})"

    if collapsed_unconditional_context:
        nonzero_conditioning = {
            name: int(np.count_nonzero(inputs[name]))
            for name in ("feature1", "feature2")
            if np.count_nonzero(inputs[name]) != 0
        }
        if nonzero_conditioning:
            raise ValueError(
                "Collapsed unconditional-context graphs require feature1 and feature2 "
                "to be exactly zero; nonzero counts are "
                f"{nonzero_conditioning}"
            )

    ort_atol = args.atol if args.atol is not None else (4e-2 if precision == "fp16" else 2e-3)
    ort_rtol = args.rtol if args.rtol is not None else (3e-2 if precision == "fp16" else 2e-3)
    adapter_atol = (
        args.adapter_atol
        if args.adapter_atol is not None
        else (3e-3 if precision == "fp16" else 3e-5)
    )
    adapter_rtol = (
        args.adapter_rtol
        if args.adapter_rtol is not None
        else (3e-3 if precision == "fp16" else 3e-5)
    )

    repo = args.triposplat_repo.expanduser().resolve()
    commit, dirty = source_revision(repo)
    if dirty and not args.allow_dirty_official_source:
        raise RuntimeError(
            "Official model.py/triposplat.py have tracked edits; validation must use "
            "the untouched numerical source of truth"
        )
    exported_commit = graph_metadata.get("triposplat.source_commit", "unknown")
    if (
        commit != "unknown"
        and exported_commit != "unknown"
        and commit != exported_commit
        and not args.allow_source_commit_mismatch
    ):
        raise RuntimeError(
            f"Validator official commit {commit} differs from graph commit "
            f"{exported_commit}. Check out the recorded revision or explicitly allow it."
        )

    device = choose_torch_device(torch, args.device)
    print(f"Loading untouched official flow model ({precision}) on {device}")
    flow_model, source = load_official_flow_model(
        torch=torch,
        triposplat_repo=repo,
        weights=args.weights,
        device=device,
        internal_precision=precision,
        low_memory_construction=args.low_memory_construction,
    )
    torch_inputs = torch_inputs_from_numpy(torch, inputs, device)
    official_graph = make_browser_flow_step(torch, flow_model, precision)
    primitive_gate = validate_real_rope_primitives(
        torch,
        flow_model,
        source.model_module,
    )

    synchronize_torch(torch, device)
    official_started = time.perf_counter()
    with torch.inference_mode():
        official_tensors = official_graph(*torch_inputs)
    synchronize_torch(torch, device)
    official_duration_ms = (time.perf_counter() - official_started) * 1000.0
    official = tensor_outputs_to_numpy(official_tensors)

    adapter_metadata = adapt_official_flow_for_onnx(
        torch,
        flow_model,
        source.model_module,
        attention_query_chunk=attention_query_chunk,
        collapsed_unconditional_context=collapsed_unconditional_context,
        attention_head_chunk=attention_head_chunk,
        attention_head_padding=attention_head_padding,
        qk_norm_padding_tokens=qk_norm_padding_tokens,
        rms_norm_eps=rms_norm_eps,
        attention_output_chunk=attention_output_chunk,
        attention_output_reduction_chunk=attention_output_reduction_chunk,
    )
    adapted_graph = make_browser_flow_step(torch, flow_model, precision)
    synchronize_torch(torch, device)
    adapted_started = time.perf_counter()
    with torch.inference_mode():
        adapted_tensors = adapted_graph(*torch_inputs)
    synchronize_torch(torch, device)
    adapted_duration_ms = (time.perf_counter() - adapted_started) * 1000.0
    adapted = tensor_outputs_to_numpy(adapted_tensors)
    adapter_metrics, adapter_passed = compare_output_maps(
        official,
        adapted,
        atol=adapter_atol,
        rtol=adapter_rtol,
    )

    # ORT loads another multi-gigabyte copy of the weights. Drop every PyTorch model,
    # tensor, and wrapper reference first so validation remains viable on a 16 GB Mac.
    del official_tensors, adapted_tensors, official_graph, adapted_graph, torch_inputs
    del flow_model, source
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    providers = args.providers or ["CPUExecutionProvider"]
    print(f"Loading ONNX Runtime session with {providers}")
    session_started = time.perf_counter()
    session = make_session(ort, graph_path, providers, args.session_threads)
    session_load_ms = (time.perf_counter() - session_started) * 1000.0
    ort_started = time.perf_counter()
    ort_values = session.run(list(OUTPUT_NAMES), dict(inputs))
    ort_duration_ms = (time.perf_counter() - ort_started) * 1000.0
    candidate = {
        name: np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        for name, value in zip(OUTPUT_NAMES, ort_values)
    }
    for name, value in candidate.items():
        if tuple(value.shape) != OUTPUT_SHAPES[name]:
            raise ValueError(
                f"ORT {name} output shape is {tuple(value.shape)}; expected {OUTPUT_SHAPES[name]}"
            )

    ort_metrics, ort_passed = compare_output_maps(
        official,
        candidate,
        atol=ort_atol,
        rtol=ort_rtol,
    )
    adapted_ort_metrics, _ = compare_output_maps(
        adapted,
        candidate,
        atol=ort_atol,
        rtol=ort_rtol,
    )
    fixture_metrics = None
    fixture_passed = True
    if fixture_expected:
        fixture_metrics, fixture_passed = compare_output_maps(
            fixture_expected,
            official,
            atol=adapter_atol,
            rtol=adapter_rtol,
        )

    overall_passed = bool(
        primitive_gate["passed"] and adapter_passed and ort_passed and fixture_passed
    )
    report: dict[str, Any] = {
        "passed": overall_passed,
        "contract": {
            "public_dtype": "float32",
            "internal_precision": precision,
            "inputs": {name: list(INPUT_SHAPES[name]) for name in INPUT_NAMES},
            "outputs": {name: list(OUTPUT_SHAPES[name]) for name in OUTPUT_NAMES},
            "timestep_semantics": "already scaled 1000 * normalized timestep",
            "collapsed_unconditional_context": collapsed_unconditional_context,
            "conditioning_requirement": (
                "feature1 and feature2 must be exactly zero"
                if collapsed_unconditional_context
                else "canonical conditional and unconditional inputs"
            ),
        },
        "source": {
            "repository": OFFICIAL_REPOSITORY_URL,
            "checkout": str(repo),
            "commit": commit,
            "exported_commit": exported_commit,
            "tracked_source_dirty": dirty,
            "loader": "official triposplat.load_flow_model",
        },
        "inputs": {
            "source": input_source,
            "summaries": {name: array_summary(inputs[name]) for name in INPUT_NAMES},
        },
        "rope_primitive_gate": primitive_gate,
        "adapter": {
            "passed": adapter_passed,
            "metadata": {
                "real_rope_modules": adapter_metadata.real_rope_modules,
                "attention_query_chunk": adapter_metadata.attention_query_chunk,
                "collapsed_unconditional_context": (
                    adapter_metadata.collapsed_unconditional_context
                ),
                "attention_head_chunk": adapter_metadata.attention_head_chunk,
                "attention_head_padding": adapter_metadata.attention_head_padding,
                "qk_norm_padding_tokens": adapter_metadata.qk_norm_padding_tokens,
                "qk_norm_modules": adapter_metadata.qk_norm_modules,
                "stable_rms_norm_modules": adapter_metadata.stable_rms_norm_modules,
                "rms_norm_eps": adapter_metadata.rms_norm_eps,
                "attention_output_chunk": adapter_metadata.attention_output_chunk,
                "attention_output_reduction_chunk": (
                    adapter_metadata.attention_output_reduction_chunk
                ),
                "attention_output_modules": adapter_metadata.attention_output_modules,
                "static_position_shape": list(adapter_metadata.static_position_shape),
                "static_position_dtype": adapter_metadata.static_position_dtype,
                "static_position_sha256": adapter_metadata.static_position_sha256,
            },
            "comparison_to_untouched_official": adapter_metrics,
        },
        "onnxruntime_comparison_to_untouched_official": ort_metrics,
        "onnxruntime_comparison_to_adapted_pytorch": adapted_ort_metrics,
        "fixture_reference_comparison": fixture_metrics,
        "output_summaries": {
            name: {
                "official_pytorch": array_summary(official[name]),
                "adapted_pytorch": array_summary(adapted[name]),
                "onnxruntime": array_summary(candidate[name]),
            }
            for name in OUTPUT_NAMES
        },
        "tolerances": {
            "onnxruntime": {"atol": ort_atol, "rtol": ort_rtol},
            "adapter": {"atol": adapter_atol, "rtol": adapter_rtol},
        },
        "measured_single_call_ms": {
            "official_pytorch": official_duration_ms,
            "adapted_pytorch": adapted_duration_ms,
            "onnxruntime_session_load": session_load_ms,
            "onnxruntime_inference": ort_duration_ms,
            "note": "Python validation measurements; not Chrome/Edge WebGPU benchmarks.",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch_device": str(device),
            "ort_providers": session.get_providers(),
            "graph": str(graph_path),
            "external_data": [str(path) for path in external_files],
            "graph_internal_precision_metadata": graph_metadata.get(
                INTERNAL_PRECISION_METADATA_KEY
            ),
        },
    }

    if args.save_fixture:
        fixture_path = args.save_fixture.expanduser().resolve()
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fixture_path,
            **inputs,
            **official,
            metadata=np.asarray(
                json.dumps(
                    {
                        "source": "untouched official TripoSplat load_flow_model",
                        "source_commit": commit,
                        "internal_precision": precision,
                        "public_dtype": "float32",
                    },
                    sort_keys=True,
                )
            ),
        )
        report["saved_fixture"] = str(fixture_path)
        print(f"Wrote browser one-call fixture {fixture_path}")
    if args.report:
        report_path = write_json(args.report, report)
        print(f"Wrote validation report {report_path}")

    print(json.dumps(report["onnxruntime_comparison_to_untouched_official"], indent=2, sort_keys=True))
    print(
        "PASS: official complex PyTorch, real-adapted PyTorch, and ONNX Runtime agree"
        if overall_passed
        else "FAIL: one or more mandatory parity gates are outside tolerance"
    )
    return report


def main() -> None:
    report = validate(parse_args())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
