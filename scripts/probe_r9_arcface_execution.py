from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ASSETS = {
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
SESSION_OPTION_FIELDS = (
    "enable_cpu_mem_arena",
    "enable_mem_pattern",
    "enable_mem_reuse",
    "execution_mode",
    "execution_order",
    "graph_optimization_level",
    "inter_op_num_threads",
    "intra_op_num_threads",
    "log_severity_level",
    "log_verbosity_level",
    "logid",
    "optimized_model_filepath",
    "use_deterministic_compute",
    "use_per_session_threads",
)
EXCLUDED_SESSION_OPTION_FIELDS = ("enable_profiling", "profile_file_prefix")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve_input_shape(
    metadata_shape: Sequence[Any], filename: str, det_size: tuple[int, int]
) -> list[int]:
    if len(metadata_shape) != 4:
        raise RuntimeError(f"{filename}: expected four NCHW dimensions")
    batch, channel, height, width = metadata_shape
    if batch is None or isinstance(batch, str):
        if isinstance(batch, str) and not batch:
            raise RuntimeError(f"{filename}: empty batch symbol")
        resolved_batch = 1
    elif isinstance(batch, bool) or not isinstance(batch, int) or batch != 1:
        raise RuntimeError(f"{filename}: unsupported batch metadata {batch!r}")
    else:
        resolved_batch = batch
    if isinstance(channel, bool) or not isinstance(channel, int) or channel != 3:
        raise RuntimeError(f"{filename}: channel metadata must be integer 3")
    resolved_spatial = []
    for axis, dimension in enumerate((height, width)):
        if filename == "det_10g.onnx":
            if dimension != "?":
                raise RuntimeError(f"{filename}: detector spatial metadata must be '?'")
            resolved_spatial.append(det_size[axis])
        else:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise RuntimeError(
                    f"{filename}: non-detector spatial metadata must be fixed"
                )
            resolved_spatial.append(dimension)
    return [resolved_batch, channel, *resolved_spatial]


def _profile_asset(
    session: Any, filename: str, det_size: tuple[int, int]
) -> dict[str, Any]:
    import numpy as np

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"{filename}: expected exactly one input")
    metadata = inputs[0]
    if metadata.type != "tensor(float)":
        raise RuntimeError(f"{filename}: expected tensor(float), got {metadata.type!r}")
    metadata_shape = []
    for value in metadata.shape:
        if (
            value is None
            or isinstance(value, (str, int))
            and not isinstance(value, bool)
        ):
            metadata_shape.append(value)
        else:
            raise RuntimeError(f"{filename}: non-canonical input metadata {value!r}")
    shape = _resolve_input_shape(metadata_shape, filename, det_size)
    session.run(None, {metadata.name: np.zeros(tuple(shape), dtype=np.float32)})
    profile_path = Path(session.end_profiling()).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    events: list[list[str]] = []
    for event in profile:
        if not isinstance(event, Mapping) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        if not isinstance(args, Mapping):
            continue
        provider = args.get("provider")
        if not isinstance(provider, str) or not provider:
            continue
        if provider not in PROVIDERS:
            raise RuntimeError(f"{filename}: unregistered Node provider {provider!r}")
        if not isinstance(event.get("name"), str) or not isinstance(
            args.get("op_name"), str
        ):
            raise RuntimeError(f"{filename}: malformed registered Node event")
        events.append([event["name"], args["op_name"], provider])
    events.sort()
    counts = {
        provider: sum(event[2] == provider for event in events)
        for provider in PROVIDERS
    }
    if counts["CUDAExecutionProvider"] <= 0:
        raise RuntimeError(f"{filename}: no CUDA Node events")
    if filename == "det_10g.onnx":
        if counts["CPUExecutionProvider"] <= 0:
            raise RuntimeError("det_10g.onnx: no CPU Node events")
    elif counts["CPUExecutionProvider"] != 0:
        raise RuntimeError(f"{filename}: unexpected CPU Node events")
    digest = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "input_name": metadata.name,
        "input_metadata_shape": metadata_shape,
        "input_shape": shape,
        "input_dtype": "float32",
        "node_assignment_counts": counts,
        "ordered_node_events_sha256": digest,
    }


def _session_binding(session: Any, device_id: int, label: str) -> dict[str, Any]:
    providers = session.get_providers()
    if providers != PROVIDERS:
        raise RuntimeError(f"{label}: provider order {providers!r}")
    raw_provider_options = session.get_provider_options()
    if not isinstance(raw_provider_options, Mapping) or set(
        raw_provider_options
    ) != set(PROVIDERS):
        raise RuntimeError(f"{label}: provider options set mismatch")
    provider_options: dict[str, dict[str, str]] = {}
    for provider in PROVIDERS:
        raw_options = raw_provider_options[provider]
        if not isinstance(raw_options, Mapping) or any(
            not isinstance(key, str) for key in raw_options
        ):
            raise RuntimeError(f"{label}: provider options are not canonical")
        provider_options[provider] = {
            key: str(raw_options[key]) for key in sorted(raw_options)
        }
    options = provider_options["CUDAExecutionProvider"]
    expected_options = {
        "device_id": str(device_id),
        "use_tf32": "0",
        "cudnn_conv_algo_search": "DEFAULT",
    }
    for key, expected in expected_options.items():
        actual = str(options.get(key))
        if actual != expected:
            raise RuntimeError(f"{label}: CUDA option mismatch {key}")
    options["device_id"] = "runtime"
    session_options = session.get_session_options()
    projection = {}
    for field in SESSION_OPTION_FIELDS:
        if not hasattr(session_options, field):
            raise RuntimeError(f"{label}: session option missing {field}")
        projection[field] = str(getattr(session_options, field))
    return {
        "providers": list(providers),
        "provider_options": provider_options,
        "session_options_projection": projection,
    }


def run_probe(model_root: Path, device_id: int) -> dict[str, Any]:
    import insightface
    from insightface.app import FaceAnalysis
    import onnxruntime

    if insightface.__version__ != "0.7.3":
        raise RuntimeError(f"unexpected insightface version {insightface.__version__}")
    if onnxruntime.__version__ != "1.26.0":
        raise RuntimeError(f"unexpected onnxruntime version {onnxruntime.__version__}")
    model_dir = model_root / "models" / "buffalo_l"
    for filename, expected in ASSETS.items():
        if _sha256_file(model_dir / filename) != expected:
            raise RuntimeError(f"asset digest mismatch: {filename}")
    cuda_options = {
        "device_id": str(device_id),
        "use_tf32": "0",
        "cudnn_conv_algo_search": "DEFAULT",
    }
    analyzer = FaceAnalysis(
        name="buffalo_l",
        root=str(model_root),
        providers=PROVIDERS,
        provider_options=[cuda_options, {}],
    )
    analyzer.prepare(ctx_id=device_id, det_size=(224, 224))
    with tempfile.TemporaryDirectory(prefix="safa-r9-arcface-bootstrap-") as temp_dir:
        result_assets: dict[str, Any] = {}
        loaded = set()
        for model in analyzer.models.values():
            model_path = Path(model.model_file).resolve()
            filename = model_path.name
            if filename not in ASSETS or filename in loaded:
                raise RuntimeError(f"unexpected or duplicate loaded asset: {filename}")
            expected_path = (model_dir / filename).resolve()
            if (
                model_path != expected_path
                or _sha256_file(model_path) != ASSETS[filename]
            ):
                raise RuntimeError(f"production asset binding mismatch: {filename}")
            loaded.add(filename)
            production_binding = _session_binding(
                model.session, device_id, f"{filename} production session"
            )
            profile_options = onnxruntime.SessionOptions()
            profile_options.enable_profiling = True
            profile_options.profile_file_prefix = str(
                Path(temp_dir) / filename.removesuffix(".onnx")
            )
            profile_session = onnxruntime.InferenceSession(
                str(expected_path),
                sess_options=profile_options,
                providers=PROVIDERS,
                provider_options=[cuda_options, {}],
            )
            profile_binding = _session_binding(
                profile_session, device_id, f"{filename} matched direct session"
            )
            if profile_binding != production_binding:
                raise RuntimeError(f"{filename}: matched direct binding mismatch")
            profile_result = _profile_asset(profile_session, filename, (224, 224))
            profile_result.update(
                {
                    "provider_options": production_binding["provider_options"],
                    "provider_options_sha256": _canonical_sha256(
                        production_binding["provider_options"]
                    ),
                    "session_options_projection": production_binding[
                        "session_options_projection"
                    ],
                    "session_options_projection_sha256": _canonical_sha256(
                        production_binding["session_options_projection"]
                    ),
                }
            )
            result_assets[filename] = profile_result
        if loaded != set(ASSETS):
            raise RuntimeError(f"loaded asset set mismatch: {sorted(loaded)!r}")
    execution = {
        "providers": PROVIDERS,
        "cuda_provider_options": {
            "device_id": "runtime",
            "use_tf32": "0",
            "cudnn_conv_algo_search": "DEFAULT",
        },
        "probe": {
            "definition": "zeros_float32_nchw_from_session_input_metadata",
            "session_construction": "matched_direct_session_probe",
            "production_session_match": {
                "asset_path_and_sha256": "exact",
                "providers": "exact",
                "provider_options": "complete_normalized_exact",
                "session_options_projection": "exact",
                "excluded_session_option_fields": list(EXCLUDED_SESSION_OPTION_FIELDS),
                "session_options_projection_fields": list(SESSION_OPTION_FIELDS),
                "locked_cuda_provider_options": [
                    "device_id",
                    "use_tf32",
                    "cudnn_conv_algo_search",
                ],
            },
            "dynamic_dimension_resolution": {
                "batch_axis": "null_or_symbol_to_1",
                "channel_axis": "fixed_integer_3",
                "detector_spatial_axes": "question_mark_to_locked_det_size",
                "other_spatial_axes": "fixed_positive_integers",
            },
            "event_projection": ["name", "op_name", "provider"],
            "node_provider_policy": "fail_nonempty_unregistered",
            "ordering": "lexicographic_keep_duplicates",
            "assets": {
                filename: result_assets[filename] for filename in sorted(ASSETS)
            },
        },
    }
    return {
        "schema_version": 1,
        "contract_type": "safa_r9_arcface_execution_probe_v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_device_id": device_id,
        "execution": execution,
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _write_exclusive(args.output, run_probe(args.model_root.resolve(), args.device_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
