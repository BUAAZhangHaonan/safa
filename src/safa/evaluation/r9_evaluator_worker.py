from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from safa.evaluation.r9_phase_results import (
        ArcFaceEvaluationRequest,
        HeldoutEvaluationRequest,
        QualityEvaluationRequest,
        SampleEvidence,
    )


QUALITY_METRICS = ("fid", "kid", "niqe", "sharpness")
ARCFACE_MODEL_NAME = "buffalo_l"
ARCFACE_ASSET_NAMES = frozenset(
    {
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    }
)
WORKER_TASKS = ("quality", "arcface", "heldout")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_CUDA_DEVICE = re.compile(r"cuda:[0-9]+")
_ARCFACE_SESSION_OPTION_FIELDS = (
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
_ARCFACE_EXCLUDED_SESSION_OPTION_FIELDS = (
    "enable_profiling",
    "profile_file_prefix",
)


class R9EvaluatorError(ValueError):
    """Raised when production evaluator evidence violates the R9 contract."""


def _load_phase_request_types() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    """Import phase-result request types only for evaluator execution paths."""
    from safa.evaluation.r9_phase_results import (
        ArcFaceEvaluationRequest,
        HeldoutEvaluationRequest,
        QualityEvaluationRequest,
        SampleEvidence,
    )

    return (
        QualityEvaluationRequest,
        ArcFaceEvaluationRequest,
        HeldoutEvaluationRequest,
        SampleEvidence,
    )


QualityBackend = Callable[..., Mapping[str, Any]]
FaceAnalyzerFactory = Callable[[Mapping[str, Any], str], Any]
RepresentationBackend = Callable[..., Mapping[str, Any]]
RecognizerBackend = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class EvaluatorDependencies:
    quality_backend: QualityBackend
    face_analyzer_factory: FaceAnalyzerFactory
    representation_backend: RepresentationBackend
    recognizer_backend: RecognizerBackend


@dataclass(frozen=True)
class ProductionEvaluatorConfig:
    repo_root: Path
    device: str
    work_root: Path
    quality_script: Mapping[str, Any]
    arcface: Mapping[str, Any]
    worker_contract: Mapping[str, Any]
    batch_size: int = 16

    def __post_init__(self) -> None:
        repo_root = self.repo_root.resolve()
        if not repo_root.is_dir():
            raise R9EvaluatorError(f"repository root is not a directory: {repo_root}")
        if _CUDA_DEVICE.fullmatch(self.device) is None:
            raise R9EvaluatorError(
                "production evaluator requires an explicit cuda:N device"
            )
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise R9EvaluatorError("evaluator batch size must be a positive integer")
        work_root = _require_contained(
            repo_root, self.work_root, "evaluator work root", must_exist=False
        )
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "work_root", work_root)
        object.__setattr__(
            self,
            "quality_script",
            _validate_quality_script_binding(
                self.quality_script,
                repo_root=repo_root,
            ),
        )
        object.__setattr__(
            self,
            "arcface",
            _validate_arcface_contract(self.arcface, repo_root=repo_root),
        )
        object.__setattr__(
            self,
            "worker_contract",
            _validate_worker_contract(self.worker_contract),
        )


class R9ProductionEvaluators:
    def __init__(
        self,
        config: ProductionEvaluatorConfig,
        dependencies: EvaluatorDependencies,
    ) -> None:
        self.config = config
        self.dependencies = dependencies

    @classmethod
    def production(cls, config: ProductionEvaluatorConfig) -> R9ProductionEvaluators:
        return cls(config, production_dependencies())

    def quality(self, request: QualityEvaluationRequest) -> Mapping[str, Any]:
        return evaluate_quality_request(
            request,
            config=self.config,
            backend=self.dependencies.quality_backend,
        )

    def arcface(self, request: ArcFaceEvaluationRequest) -> Sequence[Mapping[str, Any]]:
        return evaluate_arcface_request(
            request,
            config=self.config,
            analyzer_factory=self.dependencies.face_analyzer_factory,
        )

    def heldout(self, request: HeldoutEvaluationRequest) -> Mapping[str, Any]:
        return evaluate_heldout_request(
            request,
            config=self.config,
            face_analyzer_factory=self.dependencies.face_analyzer_factory,
            representation_backend=self.dependencies.representation_backend,
            recognizer_backend=self.dependencies.recognizer_backend,
        )


def production_dependencies() -> EvaluatorDependencies:
    return EvaluatorDependencies(
        quality_backend=_production_quality_backend,
        face_analyzer_factory=_production_face_analyzer_factory,
        representation_backend=_production_representation_backend,
        recognizer_backend=_production_recognizer_backend,
    )


def _validate_worker_contract(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "implementation_path",
        "implementation_sha256",
    }:
        raise R9EvaluatorError(
            "worker implementation contract fields are not canonical"
        )
    normalized = {}
    for path_field, digest_field, label in (
        ("path", "sha256", "worker wrapper"),
        ("implementation_path", "implementation_sha256", "worker implementation"),
    ):
        path = Path(str(value[path_field])).resolve()
        expected = _require_sha256(value[digest_field], f"{label} SHA256")
        if _sha256_file(path) != expected:
            raise R9EvaluatorError(f"{label} digest mismatch")
        normalized[path_field] = str(path)
        normalized[digest_field] = expected
    if Path(normalized["implementation_path"]) != Path(__file__).resolve():
        raise R9EvaluatorError(
            "worker implementation contract does not name this module"
        )
    return normalized


def _validate_quality_script_binding(
    value: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise R9EvaluatorError("quality script binding fields are not canonical")
    raw_path = Path(str(value["path"]))
    if repo_root is None:
        if not raw_path.is_absolute():
            raise R9EvaluatorError("normalized quality script path must be absolute")
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"quality script does not exist: {path}")
    else:
        candidate = raw_path if raw_path.is_absolute() else repo_root / raw_path
        path = _require_contained(
            repo_root,
            candidate,
            "quality script",
            must_exist=True,
        )
    expected_sha256 = _require_sha256(value["sha256"], "quality script SHA256")
    if _sha256_file(path) != expected_sha256:
        raise R9EvaluatorError("quality script digest mismatch")
    return {"path": str(path), "sha256": expected_sha256}


def _validate_arcface_contract(
    value: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    expected_fields = {
        "model_name",
        "model_root",
        "det_size",
        "provider",
        "insightface_version",
        "onnxruntime_version",
        "assets",
        "execution_probe",
    }
    if not isinstance(value, Mapping):
        raise R9EvaluatorError("ArcFace runtime contract fields are not canonical")
    field_set = set(value)
    has_inline_execution = "execution" in field_set
    if field_set != expected_fields | ({"execution"} if has_inline_execution else set()):
        raise R9EvaluatorError("ArcFace runtime contract fields are not canonical")
    if value.get("model_name") != ARCFACE_MODEL_NAME:
        raise R9EvaluatorError("R9 ArcFace model must be buffalo_l")
    if value.get("det_size") != [224, 224]:
        raise R9EvaluatorError("R9 ArcFace detection size must be [224,224]")
    if value.get("provider") != "CUDAExecutionProvider":
        raise R9EvaluatorError("R9 ArcFace provider must be CUDAExecutionProvider")
    if value.get("insightface_version") != "0.7.3":
        raise R9EvaluatorError("R9 InsightFace version must be 0.7.3")
    if value.get("onnxruntime_version") != "1.26.0":
        raise R9EvaluatorError("R9 ONNX Runtime version must be 1.26.0")
    model_root = Path(str(value.get("model_root", "")))
    if not model_root.is_absolute() or not model_root.is_dir():
        raise R9EvaluatorError(
            "ArcFace model root must be an existing absolute directory"
        )
    assets = value.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != ARCFACE_ASSET_NAMES:
        raise R9EvaluatorError(
            "ArcFace contract must lock exactly five buffalo_l assets"
        )
    normalized_assets = {}
    for filename in sorted(ARCFACE_ASSET_NAMES):
        expected_sha256 = _require_sha256(
            assets[filename], f"ArcFace asset {filename} SHA256"
        )
        path = model_root / "models" / ARCFACE_MODEL_NAME / filename
        if _sha256_file(path) != expected_sha256:
            raise R9EvaluatorError(f"ArcFace asset digest mismatch: {filename}")
        normalized_assets[filename] = expected_sha256
    if has_inline_execution:
        raw_execution = value.get("execution")
    else:
        raw_execution = _read_arcface_execution_from_probe_binding(
            value.get("execution_probe"), repo_root=repo_root
        )
    execution = _validate_arcface_execution_contract(raw_execution)
    execution_probe = _validate_arcface_execution_provenance(
        value.get("execution_probe"),
        execution=execution,
        repo_root=repo_root,
    )
    return {
        "model_name": ARCFACE_MODEL_NAME,
        "model_root": str(model_root.resolve()),
        "det_size": [224, 224],
        "provider": "CUDAExecutionProvider",
        "insightface_version": "0.7.3",
        "onnxruntime_version": "1.26.0",
        "assets": normalized_assets,
        "execution": execution,
        "execution_probe": execution_probe,
    }


def _read_arcface_execution_from_probe_binding(
    value: Any, *, repo_root: Path | None
) -> Any:
    if not isinstance(value, Mapping):
        raise R9EvaluatorError(
            "ArcFace execution probe provenance fields are not canonical"
        )
    raw_path = Path(str(value.get("path", "")))
    if repo_root is None:
        if not raw_path.is_absolute():
            raise R9EvaluatorError(
                f"normalized ArcFace execution probe path must be absolute: {raw_path}"
            )
        probe_path = raw_path.resolve()
        if not probe_path.is_file():
            raise FileNotFoundError(
                f"ArcFace execution probe does not exist: {probe_path}"
            )
    else:
        candidate = raw_path if raw_path.is_absolute() else repo_root / raw_path
        probe_path = _require_contained(
            repo_root,
            candidate,
            "ArcFace execution probe",
            must_exist=True,
        )
    probe_sha256 = _require_sha256(
        value.get("sha256"), "ArcFace execution probe SHA256"
    )
    if _sha256_file(probe_path) != probe_sha256:
        raise R9EvaluatorError("ArcFace execution probe digest mismatch")
    probe = _read_json_mapping(probe_path, "ArcFace execution probe")
    if set(probe) != {
        "schema_version",
        "contract_type",
        "cuda_visible_devices",
        "runtime_device_id",
        "execution",
    } or (
        probe.get("schema_version") != 1
        or probe.get("contract_type") != "safa_r9_arcface_execution_probe_v1"
    ):
        raise R9EvaluatorError("ArcFace execution probe identity mismatch")
    return probe.get("execution")


def _validate_arcface_execution_provenance(
    value: Any,
    *,
    execution: Mapping[str, Any],
    repo_root: Path | None,
) -> dict[str, str]:
    expected_fields = {
        "path",
        "sha256",
        "bootstrap_claim_path",
        "bootstrap_claim_sha256",
        "bootstrap_claim_file_sha256",
        "bootstrap_result_path",
        "bootstrap_result_sha256",
        "bootstrap_result_file_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise R9EvaluatorError(
            "ArcFace execution probe provenance fields are not canonical"
        )

    def resolve_path(field: str, label: str) -> Path:
        raw = Path(str(value[field]))
        if repo_root is None:
            if not raw.is_absolute():
                raise R9EvaluatorError(
                    f"normalized {label} path must be absolute: {raw}"
                )
            resolved = raw.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"{label} does not exist: {resolved}")
            return resolved
        candidate = raw if raw.is_absolute() else repo_root / raw
        return _require_contained(
            repo_root,
            candidate,
            label,
            must_exist=True,
        )

    probe_path = resolve_path("path", "ArcFace execution probe")
    claim_path = resolve_path("bootstrap_claim_path", "ArcFace bootstrap claim")
    result_path = resolve_path("bootstrap_result_path", "ArcFace bootstrap result")
    probe_sha256 = _require_sha256(value["sha256"], "ArcFace execution probe SHA256")
    if _sha256_file(probe_path) != probe_sha256:
        raise R9EvaluatorError("ArcFace execution probe digest mismatch")
    claim_sha256 = _require_sha256(
        value["bootstrap_claim_sha256"], "ArcFace bootstrap claim SHA256"
    )
    claim_file_sha256 = _require_sha256(
        value["bootstrap_claim_file_sha256"],
        "ArcFace bootstrap claim file SHA256",
    )
    result_sha256 = _require_sha256(
        value["bootstrap_result_sha256"], "ArcFace bootstrap result SHA256"
    )
    result_file_sha256 = _require_sha256(
        value["bootstrap_result_file_sha256"],
        "ArcFace bootstrap result file SHA256",
    )
    if _sha256_file(claim_path) != claim_file_sha256:
        raise R9EvaluatorError("ArcFace bootstrap claim file digest mismatch")
    if _sha256_file(result_path) != result_file_sha256:
        raise R9EvaluatorError("ArcFace bootstrap result file digest mismatch")
    probe = _read_json_mapping(probe_path, "ArcFace execution probe")
    claim = _read_json_mapping(claim_path, "ArcFace bootstrap claim")
    result = _read_json_mapping(result_path, "ArcFace bootstrap result")
    _validate_digest_contract(
        claim,
        digest_field="bootstrap_claim_sha256",
        contract_type="safa_r9_bootstrap_resource_smoke_claim_v1",
    )
    _validate_digest_contract(
        result,
        digest_field="bootstrap_result_sha256",
        contract_type="safa_r9_bootstrap_resource_smoke_result_v1",
    )
    if claim.get("bootstrap_claim_sha256") != claim_sha256:
        raise R9EvaluatorError("ArcFace bootstrap claim provenance mismatch")
    if result.get("bootstrap_result_sha256") != result_sha256:
        raise R9EvaluatorError("ArcFace bootstrap result provenance mismatch")
    if (
        claim.get("kind") != "arcface_profile"
        or claim.get("retry_allowed") is not False
    ):
        raise R9EvaluatorError("ArcFace bootstrap claim policy mismatch")
    claim_probe_output = Path(str(claim.get("probe_output", "")))
    if (
        not claim_probe_output.is_absolute()
        or claim_probe_output.resolve() != probe_path
    ):
        raise R9EvaluatorError("ArcFace bootstrap claim probe path mismatch")
    if (
        result.get("bootstrap_claim_sha256") != claim_sha256
        or result.get("status") != "succeeded"
        or result.get("failure_reason") is not None
        or result.get("returncode") != 0
        or result.get("retry_allowed") is not False
    ):
        raise R9EvaluatorError("ArcFace bootstrap result did not succeed exactly once")
    if result.get("probe_output_sha256") != probe_sha256:
        raise R9EvaluatorError("ArcFace bootstrap result probe digest mismatch")
    if set(probe) != {
        "schema_version",
        "contract_type",
        "cuda_visible_devices",
        "runtime_device_id",
        "execution",
    } or (
        probe.get("schema_version") != 1
        or probe.get("contract_type") != "safa_r9_arcface_execution_probe_v1"
    ):
        raise R9EvaluatorError("ArcFace execution probe identity mismatch")
    if _json_roundtrip(probe.get("execution"), "ArcFace probe execution") != execution:
        raise R9EvaluatorError(
            "ArcFace probe execution disagrees with runtime contract"
        )
    return {
        "path": str(probe_path),
        "sha256": probe_sha256,
        "bootstrap_claim_path": str(claim_path),
        "bootstrap_claim_sha256": claim_sha256,
        "bootstrap_claim_file_sha256": claim_file_sha256,
        "bootstrap_result_path": str(result_path),
        "bootstrap_result_sha256": result_sha256,
        "bootstrap_result_file_sha256": result_file_sha256,
    }


def _validate_arcface_execution_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "providers",
        "cuda_provider_options",
        "probe",
    }:
        raise R9EvaluatorError("ArcFace execution contract fields are not canonical")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if value.get("providers") != providers:
        raise R9EvaluatorError("ArcFace providers must explicitly lock CUDA then CPU")
    cuda_options = value.get("cuda_provider_options")
    expected_cuda_options = {
        "device_id": "runtime",
        "use_tf32": "0",
        "cudnn_conv_algo_search": "DEFAULT",
    }
    if cuda_options != expected_cuda_options:
        raise R9EvaluatorError("ArcFace CUDA provider options are not canonical")
    probe = value.get("probe")
    if not isinstance(probe, Mapping) or set(probe) != {
        "definition",
        "dynamic_dimension_resolution",
        "event_projection",
        "node_provider_policy",
        "ordering",
        "production_session_match",
        "session_construction",
        "assets",
    }:
        raise R9EvaluatorError("ArcFace execution probe fields are not canonical")
    if probe.get("definition") != "zeros_float32_nchw_from_session_input_metadata":
        raise R9EvaluatorError("ArcFace execution probe definition mismatch")
    if probe.get("session_construction") != "matched_direct_session_probe":
        raise R9EvaluatorError("ArcFace probe session construction mismatch")
    production_session_match = {
        "asset_path_and_sha256": "exact",
        "providers": "exact",
        "provider_options": "complete_normalized_exact",
        "session_options_projection": "exact",
        "excluded_session_option_fields": list(_ARCFACE_EXCLUDED_SESSION_OPTION_FIELDS),
        "session_options_projection_fields": list(_ARCFACE_SESSION_OPTION_FIELDS),
        "locked_cuda_provider_options": [
            "device_id",
            "use_tf32",
            "cudnn_conv_algo_search",
        ],
    }
    if probe.get("production_session_match") != production_session_match:
        raise R9EvaluatorError("ArcFace production session match contract mismatch")
    dynamic_dimension_resolution = {
        "batch_axis": "null_or_symbol_to_1",
        "channel_axis": "fixed_integer_3",
        "detector_spatial_axes": "question_mark_to_locked_det_size",
        "other_spatial_axes": "fixed_positive_integers",
    }
    if probe.get("dynamic_dimension_resolution") != dynamic_dimension_resolution:
        raise R9EvaluatorError("ArcFace dynamic dimension resolution contract mismatch")
    if probe.get("event_projection") != ["name", "op_name", "provider"]:
        raise R9EvaluatorError("ArcFace profile event projection mismatch")
    if probe.get("node_provider_policy") != "fail_nonempty_unregistered":
        raise R9EvaluatorError("ArcFace profile Node provider policy mismatch")
    if probe.get("ordering") != "lexicographic_keep_duplicates":
        raise R9EvaluatorError("ArcFace profile event ordering mismatch")
    probe_assets = probe.get("assets")
    if (
        not isinstance(probe_assets, Mapping)
        or set(probe_assets) != ARCFACE_ASSET_NAMES
    ):
        raise R9EvaluatorError("ArcFace execution probe must lock exactly five assets")
    normalized_probe_assets: dict[str, Any] = {}
    for filename in sorted(ARCFACE_ASSET_NAMES):
        asset = probe_assets[filename]
        if not isinstance(asset, Mapping) or set(asset) != {
            "input_name",
            "input_metadata_shape",
            "input_shape",
            "input_dtype",
            "node_assignment_counts",
            "ordered_node_events_sha256",
            "provider_options",
            "provider_options_sha256",
            "session_options_projection",
            "session_options_projection_sha256",
        }:
            raise R9EvaluatorError(
                f"ArcFace execution probe fields are not canonical: {filename}"
            )
        input_name = asset.get("input_name")
        input_metadata_shape = asset.get("input_metadata_shape")
        input_shape = asset.get("input_shape")
        if not isinstance(input_name, str) or not input_name:
            raise R9EvaluatorError(f"ArcFace probe input name is invalid: {filename}")
        if (
            not isinstance(input_shape, list)
            or len(input_shape) != 4
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in input_shape
            )
            or input_shape[:2] != [1, 3]
        ):
            raise R9EvaluatorError(f"ArcFace probe input shape is invalid: {filename}")
        if not isinstance(input_metadata_shape, list) or len(input_metadata_shape) != 4:
            raise R9EvaluatorError(
                f"ArcFace probe input metadata shape is invalid: {filename}"
            )
        resolved_shape = _resolve_arcface_input_shape(
            input_metadata_shape,
            filename=filename,
            detector_size=(224, 224),
        )
        if resolved_shape != input_shape:
            raise R9EvaluatorError(
                f"ArcFace probe resolved input shape mismatch: {filename}"
            )
        if asset.get("input_dtype") != "float32":
            raise R9EvaluatorError(f"ArcFace probe input dtype is invalid: {filename}")
        provider_options = _validate_normalized_arcface_provider_options(
            asset.get("provider_options"),
            providers=providers,
            label=f"ArcFace probe provider options: {filename}",
        )
        if _canonical_value_digest(provider_options) != _require_sha256(
            asset.get("provider_options_sha256"),
            f"ArcFace provider options SHA256: {filename}",
        ):
            raise R9EvaluatorError(
                f"ArcFace provider options digest mismatch: {filename}"
            )
        session_projection = _validate_arcface_session_options_projection(
            asset.get("session_options_projection"),
            label=f"ArcFace session options projection: {filename}",
        )
        if _canonical_value_digest(session_projection) != _require_sha256(
            asset.get("session_options_projection_sha256"),
            f"ArcFace session options projection SHA256: {filename}",
        ):
            raise R9EvaluatorError(
                f"ArcFace session options projection digest mismatch: {filename}"
            )
        counts = asset.get("node_assignment_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(providers):
            raise R9EvaluatorError(
                f"ArcFace node assignment counts are not canonical: {filename}"
            )
        for provider in providers:
            count = counts[provider]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise R9EvaluatorError(
                    f"ArcFace node assignment count is invalid: {filename}/{provider}"
                )
        if counts["CUDAExecutionProvider"] <= 0:
            raise R9EvaluatorError(f"ArcFace probe has no CUDA nodes: {filename}")
        cpu_count = counts["CPUExecutionProvider"]
        if filename == "det_10g.onnx":
            if cpu_count <= 0:
                raise R9EvaluatorError("ArcFace detector probe has no CPU nodes")
        elif cpu_count != 0:
            raise R9EvaluatorError(
                f"ArcFace non-detector probe assigned nodes to CPU: {filename}"
            )
        normalized_probe_assets[filename] = {
            "input_name": input_name,
            "input_metadata_shape": list(input_metadata_shape),
            "input_shape": list(input_shape),
            "input_dtype": "float32",
            "node_assignment_counts": {
                provider: counts[provider] for provider in providers
            },
            "provider_options": provider_options,
            "provider_options_sha256": _canonical_value_digest(provider_options),
            "session_options_projection": session_projection,
            "session_options_projection_sha256": _canonical_value_digest(
                session_projection
            ),
            "ordered_node_events_sha256": _require_sha256(
                asset.get("ordered_node_events_sha256"),
                f"ArcFace ordered node events SHA256: {filename}",
            ),
        }
    return {
        "providers": providers,
        "cuda_provider_options": expected_cuda_options,
        "probe": {
            "definition": "zeros_float32_nchw_from_session_input_metadata",
            "session_construction": "matched_direct_session_probe",
            "production_session_match": production_session_match,
            "dynamic_dimension_resolution": dynamic_dimension_resolution,
            "event_projection": ["name", "op_name", "provider"],
            "node_provider_policy": "fail_nonempty_unregistered",
            "ordering": "lexicographic_keep_duplicates",
            "assets": normalized_probe_assets,
        },
    }


def _validate_normalized_arcface_provider_options(
    value: Any, *, providers: Sequence[str], label: str
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(providers):
        raise R9EvaluatorError(f"{label} provider set mismatch")
    normalized: dict[str, dict[str, str]] = {}
    for provider in providers:
        options = value[provider]
        if not isinstance(options, Mapping):
            raise R9EvaluatorError(f"{label} options are not a mapping: {provider}")
        normalized_options = {}
        for key, option_value in options.items():
            if not isinstance(key, str) or not isinstance(option_value, str):
                raise R9EvaluatorError(
                    f"{label} options are not normalized strings: {provider}"
                )
            normalized_options[key] = option_value
        normalized[provider] = {
            key: normalized_options[key] for key in sorted(normalized_options)
        }
    cuda = normalized["CUDAExecutionProvider"]
    for key, expected in (
        ("device_id", "runtime"),
        ("use_tf32", "0"),
        ("cudnn_conv_algo_search", "DEFAULT"),
    ):
        if cuda.get(key) != expected:
            raise R9EvaluatorError(f"{label} CUDA option {key} mismatch")
    return normalized


def _validate_arcface_session_options_projection(
    value: Any, *, label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(
        _ARCFACE_SESSION_OPTION_FIELDS
    ):
        raise R9EvaluatorError(f"{label} fields mismatch")
    normalized = {}
    for field in _ARCFACE_SESSION_OPTION_FIELDS:
        field_value = value[field]
        if not isinstance(field_value, str):
            raise R9EvaluatorError(f"{label} value is not a string: {field}")
        normalized[field] = field_value
    return normalized


def _resolve_arcface_input_shape(
    metadata_shape: Sequence[Any],
    *,
    filename: str,
    detector_size: tuple[int, int],
) -> list[int]:
    if len(metadata_shape) != 4:
        raise R9EvaluatorError(f"ArcFace model input shape is invalid: {filename}")
    batch, channel, height, width = metadata_shape
    if batch is None or isinstance(batch, str):
        if isinstance(batch, str) and not batch:
            raise R9EvaluatorError(
                f"ArcFace batch dimension symbol is empty: {filename}"
            )
        resolved_batch = 1
    elif isinstance(batch, bool) or not isinstance(batch, int) or batch != 1:
        raise R9EvaluatorError(f"ArcFace batch dimension is invalid: {filename}")
    else:
        resolved_batch = batch
    if isinstance(channel, bool) or not isinstance(channel, int) or channel != 3:
        raise R9EvaluatorError(f"ArcFace channel dimension is invalid: {filename}")
    resolved_spatial = []
    for axis, dimension in enumerate((height, width)):
        if filename == "det_10g.onnx":
            if dimension != "?":
                raise R9EvaluatorError(
                    f"ArcFace detector spatial metadata is invalid: {filename}"
                )
            resolved_spatial.append(detector_size[axis])
        else:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise R9EvaluatorError(
                    f"ArcFace non-detector spatial metadata is dynamic: {filename}"
                )
            resolved_spatial.append(dimension)
    return [resolved_batch, channel, *resolved_spatial]


def evaluate_quality_request(
    request: QualityEvaluationRequest,
    *,
    config: ProductionEvaluatorConfig,
    backend: QualityBackend,
) -> dict[str, Any]:
    samples = _validate_quality_request(request, config)
    role = request.image_role
    generated_paths = [getattr(sample, role).resolve() for sample in samples]
    source_paths = [sample.source.resolve() for sample in samples]
    sample_ids = [sample.sample_id for sample in samples]
    manifest_sha256 = _sha256_file(request.manifest_path)
    ordered_sample_id_sha256 = _sample_id_sha256(sample_ids)
    real_asset_sha256 = _asset_manifest_sha256(samples, "source")
    generated_asset_sha256 = _asset_manifest_sha256(samples, role)
    binding = {
        "schema_version": 1,
        "algorithm_config_sha256": _require_sha256(
            request.algorithm_config_sha256, "algorithm config SHA256"
        ),
        "runner_arm_config_sha256": _require_sha256(
            request.runner_arm_config_sha256, "runner arm config SHA256"
        ),
        "semantic_output_sha256": _require_sha256(
            request.semantic_output_sha256, "semantic output SHA256"
        ),
        "evidence_binding_sha256": _require_sha256(
            request.evidence_binding_sha256, "evidence binding SHA256"
        ),
        "generation_result_set_sha256": _require_sha256(
            request.generation_result_set_sha256,
            "generation result set SHA256",
        ),
        "per_sample_set_sha256": _require_sha256(
            request.per_sample_set_sha256, "per-sample set SHA256"
        ),
        "manifest_sha256": manifest_sha256,
        "source_index_sha256": request.source_index_sha256,
        "ordered_sample_id_sha256": ordered_sample_id_sha256,
        "real_asset_manifest_sha256": real_asset_sha256,
        "generated_asset_manifest_sha256": generated_asset_sha256,
    }
    work_root = _prepare_work_root(config, generated_paths)
    with tempfile.TemporaryDirectory(
        prefix=f"quality-{request.logical_run_id}-{role}-", dir=work_root
    ) as temporary:
        temporary_root = Path(temporary)
        generated_dir = temporary_root / "generated"
        generated_dir.mkdir()
        real_index = temporary_root / "real_index.jsonl"
        per_sample = temporary_root / "per_sample.jsonl"
        output = temporary_root / "quality.json"
        generated_rows = []
        real_rows = []
        for index, (sample_id, source, generated) in enumerate(
            zip(sample_ids, source_paths, generated_paths, strict=True)
        ):
            suffix = generated.suffix.lower()
            if suffix not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
                ".tif",
                ".tiff",
            }:
                raise R9EvaluatorError(
                    f"unsupported generated image extension: {generated}"
                )
            linked = generated_dir / f"{index:06d}{suffix}"
            os.link(generated, linked)
            if _sha256_file(linked) != _sha256_file(generated):
                raise R9EvaluatorError("hard-linked quality view changed image bytes")
            generated_rows.append({"sample_id": sample_id, "generated": str(linked)})
            real_rows.append({"sample_id": sample_id, "image_path": str(source)})
        _write_exclusive_jsonl(real_index, real_rows)
        _write_exclusive_jsonl(per_sample, generated_rows)
        quality_kwargs = {
            "quality_script": config.quality_script,
            "real_index": real_index,
            "generated_dir": generated_dir,
            "output": output,
            "iqa_method": "niqe",
            "metrics": QUALITY_METRICS,
            "max_generated": None,
            "max_real": None,
            "subset_seed": request.seed,
            "device": config.device,
            "sample_id_manifest": request.manifest_path,
            "per_sample_jsonl": per_sample,
            "generation_result": None,
            "reuse_valid_output": False,
        }
        if request.phase == "full_e2e":
            if len(samples) < 2:
                raise R9EvaluatorError("full_e2e quality requires at least two samples for KID")
            quality_kwargs["kid_subset_size"] = len(samples) - 1
        raw = backend(**quality_kwargs)
        if not output.is_file():
            raise R9EvaluatorError(
                "quality backend did not write its registered output"
            )
        disk_raw = _read_json_mapping(output, "quality backend output")
        if _json_roundtrip(raw, "quality backend result") != disk_raw:
            raise R9EvaluatorError(
                "quality backend return value disagrees with output file"
            )
    result = dict(disk_raw)
    result["r9_evidence_binding"] = binding
    _validate_quality_raw(
        result,
        sample_count=len(samples),
        sample_ids=sample_ids,
        manifest_sha256=manifest_sha256,
        ordered_sample_id_sha256=ordered_sample_id_sha256,
        real_asset_sha256=real_asset_sha256,
        generated_asset_sha256=generated_asset_sha256,
    )
    return _json_roundtrip(result, "quality raw evidence")


def evaluate_arcface_request(
    request: ArcFaceEvaluationRequest,
    *,
    config: ProductionEvaluatorConfig,
    analyzer_factory: FaceAnalyzerFactory,
) -> list[dict[str, Any]]:
    _validate_common_request(
        request.phase, request.logical_run_id, request.arm_id, request.seed
    )
    samples = _validate_evidence_assets(
        request.samples,
        config=config,
        source_index_path=request.source_index_path,
        source_index_sha256=request.source_index_sha256,
    )
    analyzer = analyzer_factory(config.arcface, config.device)
    cache: dict[Path, tuple[int, Any]] = {}

    def observe(path: Path) -> tuple[int, Any]:
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = _arcface_observation(analyzer, resolved)
        return cache[resolved]

    rows: list[dict[str, Any]] = []
    for sample in samples:
        source_count, source_embedding = observe(sample.source)
        native_count, native_embedding = observe(sample.native)
        candidate_count, candidate_embedding = observe(sample.candidate)
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "source_face_count": source_count,
            "native_face_count": native_count,
            "candidate_face_count": candidate_count,
        }
        if source_count == native_count == candidate_count == 1:
            row.update(
                {
                    "source_native_cosine": _embedding_cosine(
                        source_embedding, native_embedding, "ArcFace source-native"
                    ),
                    "source_candidate_cosine": _embedding_cosine(
                        source_embedding,
                        candidate_embedding,
                        "ArcFace source-candidate",
                    ),
                }
            )
        rows.append(row)
    _assert_finite_json(rows, "ArcFace rows")
    return rows


def build_worker_request(
    task: str,
    request: QualityEvaluationRequest
    | ArcFaceEvaluationRequest
    | HeldoutEvaluationRequest,
    *,
    config: ProductionEvaluatorConfig,
) -> dict[str, Any]:
    if task not in WORKER_TASKS:
        raise R9EvaluatorError(f"unknown evaluator task: {task!r}")
    (
        quality_request_type,
        arcface_request_type,
        heldout_request_type,
        _,
    ) = _load_phase_request_types()
    expected_type = {
        "quality": quality_request_type,
        "arcface": arcface_request_type,
        "heldout": heldout_request_type,
    }[task]
    if not isinstance(request, expected_type):
        raise R9EvaluatorError(f"{task} worker request type mismatch")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_evaluator_request_v1",
        "task": task,
        "config": {
            "repo_root": str(config.repo_root.resolve()),
            "device": config.device,
            "work_root": str(config.work_root.resolve()),
            "quality_script": _json_roundtrip(
                config.quality_script, "quality script binding"
            ),
            "arcface": _json_roundtrip(config.arcface, "ArcFace contract"),
            "worker_contract": _json_roundtrip(
                config.worker_contract, "worker implementation contract"
            ),
            "batch_size": config.batch_size,
        },
        "payload": _serialize_evaluator_request(task, request),
    }
    payload["evaluator_request_sha256"] = _canonical_digest(
        payload, "evaluator_request_sha256"
    )
    return payload


def execute_worker_request(
    request_path: Path,
    output_path: Path,
    *,
    dependencies: EvaluatorDependencies,
) -> dict[str, Any]:
    envelope = _read_json_mapping(request_path, "evaluator worker request")
    _validate_digest_contract(
        envelope,
        digest_field="evaluator_request_sha256",
        contract_type="safa_r9_phase_evaluator_request_v1",
    )
    if set(envelope) != {
        "schema_version",
        "contract_type",
        "task",
        "config",
        "payload",
        "evaluator_request_sha256",
    }:
        raise R9EvaluatorError("evaluator worker request fields are not canonical")
    task = envelope.get("task")
    if task not in WORKER_TASKS:
        raise R9EvaluatorError("evaluator worker task is not registered")
    config = _decode_config(envelope.get("config"))
    request = _decode_evaluator_request(str(task), envelope.get("payload"))
    evaluators = R9ProductionEvaluators(config, dependencies)
    result = getattr(evaluators, str(task))(request)
    _assert_finite_json(result, "evaluator worker result")
    output = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_evaluator_output_v1",
        "task": task,
        "evaluator_request_sha256": envelope["evaluator_request_sha256"],
        "worker_contract": _json_roundtrip(
            config.worker_contract, "worker implementation contract"
        ),
        "quality_script_sha256": config.quality_script["sha256"],
        "arcface_contract_sha256": _canonical_value_digest(config.arcface),
        "result": _json_roundtrip(result, "evaluator worker result"),
    }
    output["evaluator_output_sha256"] = _canonical_digest(
        output, "evaluator_output_sha256"
    )
    _write_exclusive_json(output_path, output)
    return output


def execute_worker_request_production(
    request_path: Path, output_path: Path
) -> dict[str, Any]:
    return execute_worker_request(
        request_path,
        output_path,
        dependencies=production_dependencies(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one immutable R9 phase evaluator request."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    execute_worker_request_production(args.request, args.output)
    return 0


def _production_quality_backend(
    *, quality_script: Mapping[str, Any], **kwargs: Any
) -> Mapping[str, Any]:
    binding = _validate_quality_script_binding(quality_script)
    path = Path(binding["path"])
    module_name = f"_safa_r9_quality_{binding['sha256']}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise R9EvaluatorError("quality script has no importable module specification")
    module = importlib.util.module_from_spec(spec)
    loader_get_data = getattr(spec.loader, "get_data", None)
    if not callable(loader_get_data):
        raise R9EvaluatorError("quality script loader cannot read locked source bytes")
    source = loader_get_data(str(path))
    if not isinstance(source, bytes):
        raise R9EvaluatorError("quality script loader returned non-byte source")
    if hashlib.sha256(source).hexdigest() != binding["sha256"]:
        raise R9EvaluatorError("quality script digest mismatch during import")
    code = compile(source, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    if Path(str(getattr(module, "__file__", ""))).resolve() != path:
        raise R9EvaluatorError("quality script module identity mismatch")
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if origin is None or Path(str(origin)).resolve() != path:
        raise R9EvaluatorError("quality script module origin mismatch")
    evaluator = getattr(module, "evaluate_generation_quality", None)
    if not callable(evaluator):
        raise R9EvaluatorError(
            "quality script is missing callable evaluate_generation_quality"
        )
    return evaluator(**kwargs)


def _production_face_analyzer_factory(
    arcface_contract: Mapping[str, Any], device: str
) -> Any:
    contract = _validate_arcface_contract(arcface_contract)
    if _CUDA_DEVICE.fullmatch(device) is None:
        raise R9EvaluatorError("R9 ArcFace requires an explicit CUDA device")
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError("insightface is required for R9 ArcFace evaluation") from exc
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for R9 ArcFace evaluation") from exc
    if getattr(insightface, "__version__", None) != contract["insightface_version"]:
        raise R9EvaluatorError(
            "InsightFace runtime version disagrees with locked contract"
        )
    if getattr(onnxruntime, "__version__", None) != contract["onnxruntime_version"]:
        raise R9EvaluatorError("ONNX Runtime version disagrees with locked contract")
    ctx_id = int(device.split(":", maxsplit=1)[1])
    cuda_options = {
        "device_id": str(ctx_id),
        "use_tf32": "0",
        "cudnn_conv_algo_search": "DEFAULT",
    }
    execution = contract["execution"]
    analyzer = FaceAnalysis(
        name=contract["model_name"],
        root=contract["model_root"],
        providers=list(execution["providers"]),
        provider_options=[cuda_options, {}],
    )
    analyzer.prepare(ctx_id=ctx_id, det_size=tuple(contract["det_size"]))
    models = getattr(analyzer, "models", None)
    if not isinstance(models, Mapping) or not models:
        raise R9EvaluatorError("ArcFace analyzer exposes no loaded model sessions")
    loaded_assets = set()
    with tempfile.TemporaryDirectory(prefix="safa-r9-arcface-profile-") as temp_dir:
        for model_key, model in models.items():
            model_file = getattr(model, "model_file", None)
            if not isinstance(model_file, str):
                raise R9EvaluatorError(
                    f"ArcFace model {model_key!r} exposes no ONNX asset path"
                )
            model_path = Path(model_file).resolve()
            filename = model_path.name
            if filename not in contract["assets"]:
                raise R9EvaluatorError(
                    f"ArcFace loaded unregistered ONNX asset: {filename!r}"
                )
            expected_path = (
                Path(contract["model_root"])
                / "models"
                / contract["model_name"]
                / filename
            ).resolve()
            if model_path != expected_path:
                raise R9EvaluatorError(
                    f"ArcFace asset path disagrees with locked model root: {model_path}"
                )
            if _sha256_file(model_path) != contract["assets"][filename]:
                raise R9EvaluatorError(
                    f"ArcFace loaded asset digest mismatch: {filename}"
                )
            if filename in loaded_assets:
                raise R9EvaluatorError(f"ArcFace loaded duplicate asset: {filename}")
            loaded_assets.add(filename)
            session = getattr(model, "session", None)
            if session is None or not hasattr(session, "get_providers"):
                raise R9EvaluatorError(
                    f"ArcFace model {model_key!r} exposes no ONNX Runtime session"
                )
            production_binding = _arcface_session_binding(
                session,
                label=f"ArcFace model {model_key!r}",
                providers=execution["providers"],
                ctx_id=ctx_id,
            )
            profile_options = onnxruntime.SessionOptions()
            profile_options.enable_profiling = True
            profile_options.profile_file_prefix = str(
                Path(temp_dir) / filename.removesuffix(".onnx")
            )
            profile_session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=profile_options,
                providers=list(execution["providers"]),
                provider_options=[cuda_options, {}],
            )
            profile_binding = _arcface_session_binding(
                profile_session,
                label=f"ArcFace matched direct profile {filename!r}",
                providers=execution["providers"],
                ctx_id=ctx_id,
            )
            if profile_binding != production_binding:
                raise R9EvaluatorError(
                    f"ArcFace matched direct session binding mismatch: {filename}"
                )
            expected_probe = execution["probe"]["assets"][filename]
            if (
                production_binding["provider_options"]
                != expected_probe["provider_options"]
                or production_binding["session_options_projection"]
                != expected_probe["session_options_projection"]
            ):
                raise R9EvaluatorError(
                    f"ArcFace session binding disagrees with execution probe: {filename}"
                )
            _verify_arcface_profile_probe(
                profile_session,
                filename=filename,
                expected=expected_probe,
                providers=execution["providers"],
                detector_size=tuple(contract["det_size"]),
            )
        if loaded_assets != set(contract["assets"]):
            raise R9EvaluatorError(
                "ArcFace did not load exactly the five locked ONNX assets"
            )
    return analyzer


def _arcface_session_binding(
    session: Any,
    *,
    label: str,
    providers: Sequence[str],
    ctx_id: int,
) -> dict[str, Any]:
    for method in ("get_providers", "get_provider_options", "get_session_options"):
        if not hasattr(session, method):
            raise R9EvaluatorError(f"{label} exposes no {method}")
    actual_providers = session.get_providers()
    if actual_providers != list(providers):
        raise R9EvaluatorError(f"{label} provider order mismatch")
    raw_provider_options = session.get_provider_options()
    if not isinstance(raw_provider_options, Mapping) or set(
        raw_provider_options
    ) != set(providers):
        raise R9EvaluatorError(f"{label} provider options set mismatch")
    normalized_provider_options = {}
    for provider in providers:
        raw_options = raw_provider_options[provider]
        if not isinstance(raw_options, Mapping) or any(
            not isinstance(key, str) for key in raw_options
        ):
            raise R9EvaluatorError(f"{label} provider options are not a mapping")
        normalized_provider_options[provider] = {
            key: str(raw_options[key]) for key in sorted(raw_options)
        }
    actual_cuda_options = normalized_provider_options["CUDAExecutionProvider"]
    for option, expected in (
        ("device_id", str(ctx_id)),
        ("use_tf32", "0"),
        ("cudnn_conv_algo_search", "DEFAULT"),
    ):
        actual = str(actual_cuda_options.get(option))
        if actual != expected:
            raise R9EvaluatorError(f"{label} CUDA option {option} mismatch")
    actual_cuda_options["device_id"] = "runtime"
    normalized_provider_options = _validate_normalized_arcface_provider_options(
        normalized_provider_options,
        providers=providers,
        label=f"{label} normalized provider options",
    )
    session_options = session.get_session_options()
    projection = {}
    for field in _ARCFACE_SESSION_OPTION_FIELDS:
        if not hasattr(session_options, field):
            raise R9EvaluatorError(f"{label} session option is missing: {field}")
        projection[field] = str(getattr(session_options, field))
    projection = _validate_arcface_session_options_projection(
        projection,
        label=f"{label} session options projection",
    )
    return {
        "providers": list(actual_providers),
        "provider_options": normalized_provider_options,
        "session_options_projection": projection,
    }


def _verify_arcface_profile_probe(
    session: Any,
    *,
    filename: str,
    expected: Mapping[str, Any],
    providers: Sequence[str],
    detector_size: tuple[int, int],
) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for the ArcFace execution probe") from exc
    for method in ("get_inputs", "run", "end_profiling"):
        if not hasattr(session, method):
            raise R9EvaluatorError(
                f"ArcFace model cannot execute locked profile probe: {filename}/{method}"
            )
    inputs = session.get_inputs()
    if not isinstance(inputs, Sequence) or len(inputs) != 1:
        raise R9EvaluatorError(
            f"ArcFace profile probe requires exactly one model input: {filename}"
        )
    metadata = inputs[0]
    input_name = getattr(metadata, "name", None)
    raw_shape = getattr(metadata, "shape", None)
    input_type = getattr(metadata, "type", None)
    if not isinstance(input_name, str) or not input_name:
        raise R9EvaluatorError(f"ArcFace model input name is invalid: {filename}")
    if (
        not isinstance(raw_shape, Sequence)
        or isinstance(raw_shape, (str, bytes))
        or len(raw_shape) != 4
    ):
        raise R9EvaluatorError(f"ArcFace model input shape is invalid: {filename}")
    input_metadata_shape: list[Any] = []
    for dimension in raw_shape:
        if dimension is None or isinstance(dimension, str):
            input_metadata_shape.append(dimension)
        elif isinstance(dimension, bool) or not isinstance(dimension, int):
            raise R9EvaluatorError(
                f"ArcFace model input metadata is not canonical: {filename}"
            )
        else:
            input_metadata_shape.append(dimension)
    input_shape = _resolve_arcface_input_shape(
        input_metadata_shape,
        filename=filename,
        detector_size=detector_size,
    )
    if input_type != "tensor(float)":
        raise R9EvaluatorError(f"ArcFace model input dtype is not float32: {filename}")
    actual_metadata = {
        "input_name": input_name,
        "input_metadata_shape": input_metadata_shape,
        "input_shape": input_shape,
        "input_dtype": "float32",
    }
    for field, actual in actual_metadata.items():
        if expected[field] != actual:
            raise R9EvaluatorError(
                f"ArcFace profile probe {field} mismatch: {filename}"
            )
    probe = np.zeros(tuple(input_shape), dtype=np.float32)
    session.run(None, {input_name: probe})
    profile_path = Path(str(session.end_profiling())).resolve()
    if not profile_path.is_file():
        raise R9EvaluatorError(f"ArcFace profile output is missing: {filename}")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R9EvaluatorError(
            f"ArcFace profile output is invalid: {filename}"
        ) from exc
    if not isinstance(profile, list):
        raise R9EvaluatorError(f"ArcFace profile is not an event list: {filename}")
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
        if provider not in providers:
            raise R9EvaluatorError(
                f"ArcFace profile has an unregistered Node provider: {filename}"
            )
        name = event.get("name")
        op_name = args.get("op_name")
        if not isinstance(name, str) or not isinstance(op_name, str):
            raise R9EvaluatorError(
                f"ArcFace profile Node event is malformed: {filename}"
            )
        events.append([name, op_name, provider])
    events.sort()
    counts = {provider: 0 for provider in providers}
    for _, _, provider in events:
        counts[provider] += 1
    event_digest = hashlib.sha256(
        json.dumps(
            events,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if counts != expected["node_assignment_counts"]:
        raise R9EvaluatorError(
            f"ArcFace profile node assignment counts mismatch: {filename}"
        )
    if event_digest != expected["ordered_node_events_sha256"]:
        raise R9EvaluatorError(
            f"ArcFace ordered node events digest mismatch: {filename}"
        )


def _arcface_observation(analyzer: Any, path: Path) -> tuple[int, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for ArcFace") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise R9EvaluatorError(f"ArcFace could not decode image: {path}")
    faces = analyzer.get(image)
    if not isinstance(faces, Sequence):
        raise R9EvaluatorError("ArcFace analyzer returned a non-sequence")
    count = len(faces)
    if count != 1:
        return count, None
    embedding = np.asarray(faces[0].embedding, dtype=np.float64)
    if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
        raise R9EvaluatorError("ArcFace returned an invalid embedding")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 0.0:
        raise R9EvaluatorError("ArcFace returned a zero or non-finite embedding")
    return 1, embedding / norm


def _embedding_cosine(first: Any, second: Any, label: str) -> float:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for evaluator cosine") from exc
    value = float(np.dot(first, second))
    if not math.isfinite(value) or value < -1.000001 or value > 1.000001:
        raise R9EvaluatorError(f"{label} is outside the finite cosine range")
    return max(-1.0, min(1.0, value))


def _validate_quality_request(
    request: QualityEvaluationRequest, config: ProductionEvaluatorConfig
) -> tuple[SampleEvidence, ...]:
    _validate_common_request(
        request.phase, request.logical_run_id, request.arm_id, request.seed
    )
    if request.image_role not in {"candidate", "native"}:
        raise R9EvaluatorError("quality image role must be candidate or native")
    samples = _validate_evidence_assets(
        request.samples,
        config=config,
        source_index_path=request.source_index_path,
        source_index_sha256=request.source_index_sha256,
    )
    manifest = request.manifest_path.resolve()
    _require_contained(config.repo_root, manifest, "quality manifest", must_exist=True)
    manifest_ids = _read_manifest_ids(manifest)
    if manifest_ids != [sample.sample_id for sample in samples]:
        raise R9EvaluatorError("quality samples do not exactly follow manifest order")
    for field in (
        "algorithm_config_sha256",
        "runner_arm_config_sha256",
        "semantic_output_sha256",
        "evidence_binding_sha256",
        "generation_result_set_sha256",
        "per_sample_set_sha256",
    ):
        _require_sha256(getattr(request, field), field.replace("_", " "))
    return samples


def _validate_quality_raw(
    raw: Mapping[str, Any],
    *,
    sample_count: int,
    sample_ids: Sequence[str],
    manifest_sha256: str,
    ordered_sample_id_sha256: str,
    real_asset_sha256: str,
    generated_asset_sha256: str,
) -> None:
    if raw.get("metrics") != list(QUALITY_METRICS):
        raise R9EvaluatorError("quality backend did not run the registered metric set")
    for field in ("num_generated", "num_real", "sample_id_count"):
        if raw.get(field) != sample_count:
            raise R9EvaluatorError(f"quality {field} does not match manifest coverage")
    if raw.get("sample_id_sha256") != ordered_sample_id_sha256:
        raise R9EvaluatorError("quality sample-ID digest mismatch")
    contract = raw.get("quality_contract")
    if not isinstance(contract, Mapping):
        raise R9EvaluatorError("quality output lacks quality_contract")
    required_contract = {
        "schema_version": 1,
        "metrics": list(QUALITY_METRICS),
        "sample_id_manifest_sha256": manifest_sha256,
        "real_asset_manifest_sha256": real_asset_sha256,
        "generated_asset_manifest_sha256": generated_asset_sha256,
    }
    for field, expected in required_contract.items():
        if contract.get(field) != expected:
            raise R9EvaluatorError(f"quality contract field mismatch: {field}")
    iqa = raw.get("iqa")
    sharpness = raw.get("sharpness")
    if not isinstance(iqa, Mapping) or iqa.get("method") != "niqe":
        raise R9EvaluatorError("quality output lacks registered NIQE")
    if not isinstance(sharpness, Mapping) or sharpness.get("definition") != (
        "grayscale_laplacian_variance"
    ):
        raise R9EvaluatorError("quality output lacks registered Sharpness")
    for value, label in (
        (raw.get("fid"), "FID"),
        (raw.get("kid_mean"), "KID"),
        (iqa.get("mean"), "NIQE"),
        (sharpness.get("mean"), "Sharpness"),
    ):
        _finite_float(value, label)
    _validate_quality_per_sample_metrics(
        raw.get("per_sample_metrics"),
        sample_ids=sample_ids,
        niqe_mean=_finite_float(iqa.get("mean"), "NIQE"),
        sharpness_mean=_finite_float(sharpness.get("mean"), "Sharpness"),
    )
    _assert_finite_json(raw, "quality output")


def _validate_quality_per_sample_metrics(
    value: Any,
    *,
    sample_ids: Sequence[str],
    niqe_mean: float,
    sharpness_mean: float,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R9EvaluatorError("quality output lacks per_sample_metrics")
    required = {
        "schema_version",
        "contract_type",
        "sample_count",
        "ordered_sample_id_sha256",
        "metric_fields",
        "rows",
        "per_sample_metrics_sha256",
    }
    if set(value) != required:
        raise R9EvaluatorError("per_sample_metrics fields are not canonical")
    _assert_finite_json(value, "per_sample_metrics")
    _validate_digest_contract(
        value,
        digest_field="per_sample_metrics_sha256",
        contract_type="safa_r9_quality_per_sample_metrics_v1",
    )
    expected_ids = list(sample_ids)
    if value.get("sample_count") != len(expected_ids):
        raise R9EvaluatorError("per_sample_metrics count mismatch")
    if value.get("ordered_sample_id_sha256") != _sample_id_sha256(expected_ids):
        raise R9EvaluatorError("per_sample_metrics ordered sample digest mismatch")
    if value.get("metric_fields") != ["niqe", "sharpness"]:
        raise R9EvaluatorError("per_sample_metrics metric fields mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise R9EvaluatorError("per_sample_metrics rows do not cover the manifest")
    normalized_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "niqe",
            "sharpness",
        }:
            raise R9EvaluatorError(
                f"per_sample_metrics row {index} fields are not canonical"
            )
        normalized_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "niqe": _finite_float(row.get("niqe"), "per-sample NIQE"),
                "sharpness": _finite_float(
                    row.get("sharpness"), "per-sample Sharpness"
                ),
            }
        )
    if [row["sample_id"] for row in normalized_rows] != expected_ids:
        raise R9EvaluatorError("per_sample_metrics rows violate manifest order")
    if statistics.fmean(row["niqe"] for row in normalized_rows) != niqe_mean:
        raise R9EvaluatorError("per-sample NIQE summary mismatch")
    if (
        statistics.fmean(row["sharpness"] for row in normalized_rows)
        != sharpness_mean
    ):
        raise R9EvaluatorError("per-sample Sharpness summary mismatch")
    return {
        **{key: value[key] for key in required - {"rows"}},
        "rows": normalized_rows,
    }


def _validate_common_request(
    phase: Any, logical_run_id: Any, arm_id: Any, seed: Any
) -> None:
    if phase not in {"diagnose", "calibrate", "confirm512", "full", "full_e2e"}:
        raise R9EvaluatorError("evaluator phase is not registered")
    for value, label in ((logical_run_id, "logical run ID"), (arm_id, "arm ID")):
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise R9EvaluatorError(f"{label} must be filesystem-safe")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise R9EvaluatorError("evaluator seed must be a nonnegative integer")


def _validate_samples(samples: Sequence[SampleEvidence]) -> tuple[SampleEvidence, ...]:
    if (
        not isinstance(samples, Sequence)
        or isinstance(samples, (str, bytes))
        or not samples
    ):
        raise R9EvaluatorError("evaluator samples must be a nonempty sequence")
    normalized = tuple(samples)
    ids = []
    sample_evidence_type = _load_phase_request_types()[3]
    for sample in normalized:
        if not isinstance(sample, sample_evidence_type):
            raise R9EvaluatorError("evaluator sample has the wrong type")
        if not isinstance(sample.sample_id, str) or not sample.sample_id:
            raise R9EvaluatorError("sample ID must be a nonempty string")
        ids.append(sample.sample_id)
        for role in ("source", "native", "candidate"):
            path = getattr(sample, role).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{role} image is missing: {path}")
    if len(ids) != len(set(ids)):
        raise R9EvaluatorError("evaluator sample IDs must be unique")
    return normalized


def _validate_evidence_assets(
    samples: Sequence[SampleEvidence],
    *,
    config: ProductionEvaluatorConfig,
    source_index_path: Path,
    source_index_sha256: str,
) -> tuple[SampleEvidence, ...]:
    normalized = _validate_samples(samples)
    index_path = _require_contained(
        config.repo_root,
        source_index_path,
        "source index",
        must_exist=True,
    )
    expected_index_sha256 = _require_sha256(source_index_sha256, "source index SHA256")
    if _sha256_file(index_path) != expected_index_sha256:
        raise R9EvaluatorError("source index digest mismatch")
    index_by_id: dict[str, Path] = {}
    for line_no, row in enumerate(_read_jsonl(index_path, "source index"), start=1):
        sample_id = row.get("sample_id")
        image_path = row.get("image_path")
        if not isinstance(sample_id, str) or not sample_id or "\x00" in sample_id:
            raise R9EvaluatorError(
                f"source index row {line_no} has an invalid sample ID"
            )
        if sample_id in index_by_id:
            raise R9EvaluatorError(
                f"source index has duplicate sample ID {sample_id!r}"
            )
        if not isinstance(image_path, str) or not image_path:
            raise R9EvaluatorError(
                f"source index row {line_no} has an invalid image path"
            )
        raw_image_path = Path(image_path)
        resolved = (
            raw_image_path
            if raw_image_path.is_absolute()
            else config.repo_root / raw_image_path
        ).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"source index image does not exist for {sample_id!r}: {resolved}"
            )
        index_by_id[sample_id] = resolved
    for sample in normalized:
        indexed_source = index_by_id.get(sample.sample_id)
        if indexed_source is None:
            raise R9EvaluatorError(
                f"source index does not contain sample ID {sample.sample_id!r}"
            )
        source = sample.source.resolve()
        native = _require_contained(
            config.repo_root,
            sample.native,
            "native image",
            must_exist=True,
        )
        candidate = _require_contained(
            config.repo_root,
            sample.candidate,
            "candidate image",
            must_exist=True,
        )
        if source != indexed_source:
            raise R9EvaluatorError(
                f"source path disagrees with locked index for {sample.sample_id!r}"
            )
        for role, path, expected in (
            ("source", source, sample.source_sha256),
            ("native", native, sample.native_sha256),
            ("candidate", candidate, sample.candidate_sha256),
        ):
            expected_sha256 = _require_sha256(expected, f"{role} image SHA256")
            if _sha256_file(path) != expected_sha256:
                raise R9EvaluatorError(
                    f"{role} image digest mismatch for {sample.sample_id!r}"
                )
    return normalized


def _prepare_work_root(
    config: ProductionEvaluatorConfig, generated_paths: Sequence[Path]
) -> Path:
    root = _require_contained(
        config.repo_root, config.work_root, "evaluator work root", must_exist=False
    )
    root.mkdir(parents=True, exist_ok=True)
    root_device = root.stat().st_dev
    for path in generated_paths:
        _require_contained(config.repo_root, path, "generated image", must_exist=True)
        if path.stat().st_dev != root_device:
            raise R9EvaluatorError(
                "quality hard-link work root and generated images must share a filesystem"
            )
    return root


def _read_manifest_ids(path: Path) -> list[str]:
    rows = _read_jsonl(path, "sample manifest")
    ids = []
    for index, row in enumerate(rows):
        if set(row) != {"sample_id"}:
            raise R9EvaluatorError(f"manifest row {index} fields are not canonical")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise R9EvaluatorError(f"manifest row {index} has an invalid sample ID")
        ids.append(sample_id)
    if len(ids) != len(set(ids)):
        raise R9EvaluatorError("manifest sample IDs are not unique")
    return ids


def _serialize_samples(samples: Sequence[SampleEvidence]) -> list[dict[str, str]]:
    return [
        {
            "sample_id": sample.sample_id,
            "source": str(sample.source.resolve()),
            "native": str(sample.native.resolve()),
            "candidate": str(sample.candidate.resolve()),
            "source_sha256": sample.source_sha256,
            "native_sha256": sample.native_sha256,
            "candidate_sha256": sample.candidate_sha256,
        }
        for sample in samples
    ]


def _serialize_evaluator_request(task: str, request: Any) -> dict[str, Any]:
    if task == "quality":
        return {
            "phase": request.phase,
            "logical_run_id": request.logical_run_id,
            "arm_id": request.arm_id,
            "seed": request.seed,
            "image_role": request.image_role,
            "manifest_path": str(request.manifest_path.resolve()),
            "source_index_path": str(request.source_index_path.resolve()),
            "source_index_sha256": request.source_index_sha256,
            "samples": _serialize_samples(request.samples),
            "algorithm_config_sha256": request.algorithm_config_sha256,
            "runner_arm_config_sha256": request.runner_arm_config_sha256,
            "semantic_output_sha256": request.semantic_output_sha256,
            "evidence_binding_sha256": request.evidence_binding_sha256,
            "generation_result_set_sha256": request.generation_result_set_sha256,
            "per_sample_set_sha256": request.per_sample_set_sha256,
        }
    if task == "arcface":
        return {
            "phase": request.phase,
            "logical_run_id": request.logical_run_id,
            "arm_id": request.arm_id,
            "seed": request.seed,
            "source_index_path": str(request.source_index_path.resolve()),
            "source_index_sha256": request.source_index_sha256,
            "samples": _serialize_samples(request.samples),
        }
    return {
        "phase": request.phase,
        "arm_id": request.arm_id,
        "seed": request.seed,
        "source_index_path": str(request.source_index_path.resolve()),
        "source_index_sha256": request.source_index_sha256,
        "samples": _serialize_samples(request.samples),
        "selection": _json_roundtrip(request.selection, "selection"),
        "heldout_seal": _json_roundtrip(request.heldout_seal, "heldout seal"),
    }


def _decode_config(value: Any) -> ProductionEvaluatorConfig:
    if not isinstance(value, Mapping) or set(value) != {
        "repo_root",
        "device",
        "work_root",
        "quality_script",
        "arcface",
        "worker_contract",
        "batch_size",
    }:
        raise R9EvaluatorError("evaluator worker config fields are not canonical")
    return ProductionEvaluatorConfig(
        repo_root=Path(str(value["repo_root"])),
        device=str(value["device"]),
        work_root=Path(str(value["work_root"])),
        quality_script=value["quality_script"],
        arcface=value["arcface"],
        worker_contract=value["worker_contract"],
        batch_size=value["batch_size"],
    )


def _decode_evaluator_request(task: str, value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise R9EvaluatorError("evaluator payload must be a mapping")
    (
        quality_request_type,
        arcface_request_type,
        heldout_request_type,
        _,
    ) = _load_phase_request_types()
    samples = _decode_samples(value.get("samples"))
    if task == "quality":
        expected = {
            "phase",
            "logical_run_id",
            "arm_id",
            "seed",
            "image_role",
            "manifest_path",
            "source_index_path",
            "source_index_sha256",
            "samples",
            "algorithm_config_sha256",
            "runner_arm_config_sha256",
            "semantic_output_sha256",
            "evidence_binding_sha256",
            "generation_result_set_sha256",
            "per_sample_set_sha256",
        }
        if set(value) != expected:
            raise R9EvaluatorError("quality evaluator payload fields are not canonical")
        return quality_request_type(
            phase=value["phase"],
            logical_run_id=value["logical_run_id"],
            arm_id=value["arm_id"],
            seed=value["seed"],
            image_role=value["image_role"],
            manifest_path=Path(str(value["manifest_path"])),
            source_index_path=Path(str(value["source_index_path"])),
            source_index_sha256=value["source_index_sha256"],
            samples=samples,
            algorithm_config_sha256=value["algorithm_config_sha256"],
            runner_arm_config_sha256=value["runner_arm_config_sha256"],
            semantic_output_sha256=value["semantic_output_sha256"],
            evidence_binding_sha256=value["evidence_binding_sha256"],
            generation_result_set_sha256=value["generation_result_set_sha256"],
            per_sample_set_sha256=value["per_sample_set_sha256"],
        )
    if task == "arcface":
        expected = {
            "phase",
            "logical_run_id",
            "arm_id",
            "seed",
            "source_index_path",
            "source_index_sha256",
            "samples",
        }
        if set(value) != expected:
            raise R9EvaluatorError("ArcFace evaluator payload fields are not canonical")
        return arcface_request_type(
            phase=value["phase"],
            logical_run_id=value["logical_run_id"],
            arm_id=value["arm_id"],
            seed=value["seed"],
            source_index_path=Path(str(value["source_index_path"])),
            source_index_sha256=value["source_index_sha256"],
            samples=samples,
        )
    expected = {
        "phase",
        "arm_id",
        "seed",
        "source_index_path",
        "source_index_sha256",
        "samples",
        "selection",
        "heldout_seal",
    }
    if set(value) != expected:
        raise R9EvaluatorError("heldout evaluator payload fields are not canonical")
    if not isinstance(value["selection"], Mapping) or not isinstance(
        value["heldout_seal"], Mapping
    ):
        raise R9EvaluatorError("heldout evaluator contracts must be mappings")
    return heldout_request_type(
        phase=value["phase"],
        arm_id=value["arm_id"],
        seed=value["seed"],
        source_index_path=Path(str(value["source_index_path"])),
        source_index_sha256=value["source_index_sha256"],
        samples=samples,
        selection=dict(value["selection"]),
        heldout_seal=dict(value["heldout_seal"]),
    )


def _decode_samples(value: Any) -> tuple[SampleEvidence, ...]:
    if not isinstance(value, list) or not value:
        raise R9EvaluatorError("worker samples must be a nonempty list")
    sample_evidence_type = _load_phase_request_types()[3]
    samples = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "source",
            "native",
            "candidate",
            "source_sha256",
            "native_sha256",
            "candidate_sha256",
        }:
            raise R9EvaluatorError(f"worker sample {index} fields are not canonical")
        samples.append(
            sample_evidence_type(
                sample_id=str(row["sample_id"]),
                source=Path(str(row["source"])),
                native=Path(str(row["native"])),
                candidate=Path(str(row["candidate"])),
                source_sha256=str(row["source_sha256"]),
                native_sha256=str(row["native_sha256"]),
                candidate_sha256=str(row["candidate_sha256"]),
            )
        )
    return tuple(samples)


def _production_representation_backend(**kwargs: Any) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for heldout representation evaluation"
        ) from exc
    from PIL import Image

    from safa.models.e0 import freeze_e0, load_e0_checkpoint
    from safa.training.transforms import eval_transform

    name = str(kwargs["name"])
    checkpoint = Path(kwargs["checkpoint"])
    device = str(kwargs["device"])
    batch_size = kwargs["batch_size"]
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise R9EvaluatorError("representation batch size must be a positive integer")
    model, _ = load_e0_checkpoint(checkpoint, device="cpu")
    freeze_e0(model)
    model = model.to(device)
    transform = eval_transform(224)
    result: dict[str, Any] = {}
    try:
        for role in ("source", "native", "winner"):
            paths = tuple(Path(path) for path in kwargs[f"{role}_paths"])
            if not paths:
                raise R9EvaluatorError(f"{name} {role} path group is empty")
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(paths), batch_size):
                    images = []
                    for path in paths[start : start + batch_size]:
                        with Image.open(path) as image:
                            images.append(transform(image.convert("RGB")))
                    output = model(torch.stack(images).to(device))
                    if not isinstance(output, Mapping) or "embedding" not in output:
                        raise R9EvaluatorError(
                            f"heldout encoder {name} did not return embeddings"
                        )
                    chunk = output["embedding"].detach().float().cpu()
                    if chunk.ndim != 2 or not torch.isfinite(chunk).all():
                        raise R9EvaluatorError(
                            f"heldout encoder {name} returned invalid {role} embeddings"
                        )
                    chunks.append(chunk)
            result[role] = torch.cat(chunks, dim=0)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _production_recognizer_backend(**kwargs: Any) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for heldout identity evaluation") from exc
    from safa.evaluation.recognizers import TorchScriptRecognizer

    name = str(kwargs["name"])
    checkpoint = Path(kwargs["checkpoint"])
    device = str(kwargs["device"])
    batch_size = kwargs["batch_size"]
    input_size = kwargs["input_size"]
    embedding_dim = kwargs["embedding_dim"]
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise R9EvaluatorError("recognizer batch size must be a positive integer")
    recognizer = TorchScriptRecognizer(
        name=name,
        checkpoint=checkpoint,
        device=device,
        embedding_dim=embedding_dim,
        input_size=input_size,
    )
    result: dict[str, Any] = {}
    try:
        for role in ("source", "native", "winner"):
            paths = tuple(Path(path) for path in kwargs[f"{role}_paths"])
            if not paths:
                raise R9EvaluatorError(f"{name} {role} path group is empty")
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(paths), batch_size):
                    images = torch.stack(
                        [
                            _load_rgb_tensor(path)
                            for path in paths[start : start + batch_size]
                        ]
                    ).to(device)
                    chunk = recognizer.embed(images).detach().float().cpu()
                    if chunk.ndim != 2 or not torch.isfinite(chunk).all():
                        raise R9EvaluatorError(
                            f"heldout recognizer {name} returned invalid {role} embeddings"
                        )
                    chunks.append(chunk)
            result[role] = torch.cat(chunks, dim=0)
    finally:
        del recognizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def evaluate_heldout_request(
    request: HeldoutEvaluationRequest,
    *,
    config: ProductionEvaluatorConfig,
    face_analyzer_factory: FaceAnalyzerFactory,
    representation_backend: RepresentationBackend,
    recognizer_backend: RecognizerBackend,
) -> dict[str, Any]:
    if request.phase != "full":
        raise R9EvaluatorError("heldout evaluator is only valid for Full")
    _validate_common_request(
        request.phase,
        f"heldout_{request.arm_id}",
        request.arm_id,
        request.seed,
    )
    samples = _validate_evidence_assets(
        request.samples,
        config=config,
        source_index_path=request.source_index_path,
        source_index_sha256=request.source_index_sha256,
    )
    selection, assets = _validate_heldout_contracts(
        request.selection,
        request.heldout_seal,
        arm_id=request.arm_id,
        repo_root=config.repo_root,
    )
    del selection
    for sample in samples:
        _require_contained(
            config.repo_root, sample.native, "heldout native image", must_exist=True
        )
        _require_contained(
            config.repo_root,
            sample.candidate,
            "heldout winner image",
            must_exist=True,
        )
    source_paths = tuple(sample.source.resolve() for sample in samples)
    native_paths = tuple(sample.native.resolve() for sample in samples)
    winner_paths = tuple(sample.candidate.resolve() for sample in samples)
    sample_ids = [sample.sample_id for sample in samples]

    representation_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("e1", "e2"):
        groups = representation_backend(
            name=name,
            checkpoint=assets[name],
            source_paths=source_paths,
            native_paths=native_paths,
            winner_paths=winner_paths,
            device=config.device,
            batch_size=config.batch_size,
        )
        matrices = _normalize_embedding_groups(
            groups, expected_count=len(samples), label=name
        )
        representation_rows[name] = _paired_cosine_rows(sample_ids, matrices)

    recognizer_rows: dict[str, dict[str, Any]] = {}
    identity_embeddings: dict[str, Mapping[str, Any]] = {}
    recognizer_specs = {
        "facenet": {"embedding_dim": 512, "input_size": 160},
        "adaface": {"embedding_dim": 512, "input_size": 112},
    }
    for name, recognizer_spec in recognizer_specs.items():
        groups = recognizer_backend(
            name=name,
            checkpoint=assets[name],
            source_paths=source_paths,
            native_paths=native_paths,
            winner_paths=winner_paths,
            device=config.device,
            batch_size=config.batch_size,
            **recognizer_spec,
        )
        matrices = _normalize_embedding_groups(
            groups, expected_count=len(samples), label=name
        )
        rows = _paired_cosine_rows(sample_ids, matrices)
        recognizer_rows[name] = {
            "source_exact_one_count": len(samples),
            "native_exact_one_count": len(samples),
            "winner_exact_one_count": len(samples),
            "paired_exact_one_count": len(samples),
            "failure_sample_ids": [],
            "rows": rows,
        }
        identity_embeddings[name] = matrices

    analyzer = face_analyzer_factory(config.arcface, config.device)
    arcface_groups: dict[str, list[Any]] = {
        role: [] for role in ("source", "native", "winner")
    }
    arcface_failures = []
    for sample_id, source_path, native_path, winner_path in zip(
        sample_ids, source_paths, native_paths, winner_paths, strict=True
    ):
        observations = {
            role: _arcface_observation(analyzer, path)
            for role, path in (
                ("source", source_path),
                ("native", native_path),
                ("winner", winner_path),
            )
        }
        if all(count == 1 for count, _ in observations.values()):
            for role, (_, embedding) in observations.items():
                arcface_groups[role].append(embedding)
        else:
            arcface_failures.append(sample_id)
    if arcface_failures:
        arcface_unavailable = {
            "reason": "incomplete_exact_one_face_coverage",
            "coverage": len(samples) - len(arcface_failures),
        }
    else:
        identity_embeddings["arcface"] = _normalize_embedding_groups(
            arcface_groups,
            expected_count=len(samples),
            label="arcface",
        )
        arcface_unavailable = None
    raw = {
        "representations": representation_rows,
        "recognizers": recognizer_rows,
        "identity_report": _identity_report(
            identity_embeddings,
            sample_count=len(samples),
            arcface_unavailable=arcface_unavailable,
        ),
    }
    _assert_finite_json(raw, "heldout raw evidence")
    return _json_roundtrip(raw, "heldout raw evidence")


def _validate_heldout_contracts(
    selection: Mapping[str, Any],
    heldout_seal: Mapping[str, Any],
    *,
    arm_id: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    normalized_selection = _json_roundtrip(selection, "selection")
    normalized_seal = _json_roundtrip(heldout_seal, "heldout seal")
    if not isinstance(normalized_selection, dict) or not isinstance(
        normalized_seal, dict
    ):
        raise R9EvaluatorError("heldout contracts must be mappings")
    selection_type = normalized_selection.get("contract_type")
    if selection_type not in {
        "safa_r9_selection_v1",
        "safa_r9_full_continuation_selection_v1",
    }:
        raise R9EvaluatorError("heldout selection contract type is not supported")
    _validate_digest_contract(
        normalized_selection,
        digest_field="selection_sha256",
        contract_type=str(selection_type),
    )
    _validate_digest_contract(
        normalized_seal,
        digest_field="heldout_seal_sha256",
        contract_type="safa_r9_heldout_seal_v1",
    )
    if (
        normalized_selection.get("winner_locked") is not True
        or normalized_selection.get("reselection_allowed") is not False
    ):
        raise R9EvaluatorError("heldout selection is not irreversibly winner-locked")
    winner = normalized_selection.get("winner")
    if not isinstance(winner, Mapping) or winner.get("arm_id") != arm_id:
        raise R9EvaluatorError("heldout request arm does not match locked winner")
    if (
        normalized_seal.get("selection_sha256")
        != normalized_selection["selection_sha256"]
        or normalized_seal.get("winner") != winner
        or normalized_seal.get("sealed") is not True
        or normalized_seal.get("execution_count") != 0
    ):
        raise R9EvaluatorError("heldout seal does not bind the unrun locked winner")
    assets = normalized_seal.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != {
        "e1",
        "e2",
        "facenet",
        "adaface",
    }:
        raise R9EvaluatorError("heldout seal assets are not canonical")
    paths: dict[str, Path] = {}
    for name, value in assets.items():
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "state"}:
            raise R9EvaluatorError(f"heldout asset {name} fields are not canonical")
        if value.get("state") != "sealed_unrun":
            raise R9EvaluatorError(f"heldout asset {name} is not sealed-unrun")
        raw_path = Path(str(value.get("path", "")))
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        resolved = _require_contained(
            repo_root, path, f"heldout asset {name}", must_exist=True
        )
        if _sha256_file(resolved) != _require_sha256(
            value.get("sha256"), f"heldout asset {name} SHA256"
        ):
            raise R9EvaluatorError(f"heldout asset {name} digest mismatch")
        paths[str(name)] = resolved
    return normalized_selection, paths


def _load_rgb_tensor(path: Path) -> Any:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "numpy and torch are required for identity evaluation"
        ) from exc
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).copy()
    if array.shape != (224, 224, 3) or not np.isfinite(array).all():
        raise R9EvaluatorError(f"invalid RGB image tensor: {path}")
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0)


def _normalize_embedding_groups(
    value: Mapping[str, Any], *, expected_count: int, label: str
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("torch is required for embedding validation") from exc
    if not isinstance(value, Mapping) or set(value) != {"source", "native", "winner"}:
        raise R9EvaluatorError(f"{label} embedding groups are not canonical")
    groups = {}
    shape = None
    for role in ("source", "native", "winner"):
        raw_matrix = value[role]
        if torch.is_tensor(raw_matrix):
            matrix = raw_matrix.detach().float().cpu()
        else:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "numpy is required for embedding validation"
                ) from exc
            matrix = torch.from_numpy(np.asarray(raw_matrix)).detach().float().cpu()
        if (
            matrix.ndim != 2
            or matrix.shape[0] != expected_count
            or matrix.shape[1] <= 0
            or not torch.isfinite(matrix).all()
        ):
            raise R9EvaluatorError(f"{label} {role} embeddings are invalid")
        if shape is None:
            shape = tuple(matrix.shape)
        elif tuple(matrix.shape) != shape:
            raise R9EvaluatorError(f"{label} role embedding shapes disagree")
        norms = torch.linalg.vector_norm(matrix, dim=1)
        if bool((norms <= 0).any()):
            raise R9EvaluatorError(f"{label} {role} contains a zero embedding")
        groups[role] = functional.normalize(matrix, p=2, dim=1)
    return groups


def _paired_cosine_rows(
    sample_ids: Sequence[str], groups: Mapping[str, Any]
) -> list[dict[str, Any]]:
    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("torch is required for paired cosine evaluation") from exc
    native = functional.cosine_similarity(groups["source"], groups["native"], dim=1)
    winner = functional.cosine_similarity(groups["source"], groups["winner"], dim=1)
    rows = []
    for index, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "native_cosine": _cosine_float(native[index], "native cosine"),
                "winner_cosine": _cosine_float(winner[index], "winner cosine"),
            }
        )
    return rows


def _identity_report(
    recognizers: Mapping[str, Mapping[str, Any]],
    *,
    sample_count: int,
    arcface_unavailable: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("torch is required for identity reporting") from exc
    from safa.evaluation.runner import (
        _privacy_roc_metrics,
        deterministic_impostor_indices,
    )

    report: dict[str, Any] = {}
    for name in ("arcface", "facenet", "adaface"):
        if name == "arcface" and arcface_unavailable is not None:
            reason = arcface_unavailable.get("reason")
            coverage = arcface_unavailable.get("coverage")
            if not isinstance(reason, str) or not reason:
                raise R9EvaluatorError("unavailable ArcFace report requires a reason")
            if (
                isinstance(coverage, bool)
                or not isinstance(coverage, int)
                or not 0 <= coverage < sample_count
            ):
                raise R9EvaluatorError(
                    "unavailable ArcFace report requires incomplete coverage"
                )
            report[name] = {
                "status": "unavailable",
                "reason": reason,
                "coverage": coverage,
                "roles": {
                    role: {"status": "unavailable", "reason": reason}
                    for role in ("native", "winner")
                },
            }
            continue
        groups = recognizers.get(name)
        if not isinstance(groups, Mapping):
            raise R9EvaluatorError(f"identity report lacks {name} embeddings")
        count = int(groups["source"].shape[0])
        if count != sample_count:
            raise R9EvaluatorError(f"identity report {name} coverage mismatch")
        impostor = torch.tensor(deterministic_impostor_indices(count), dtype=torch.long)
        roles = {}
        for role in ("native", "winner"):
            same = functional.cosine_similarity(groups["source"], groups[role], dim=1)
            impostor_scores = functional.cosine_similarity(
                groups["source"], groups[role][impostor], dim=1
            )
            metrics = _privacy_roc_metrics(same.tolist(), impostor_scores.tolist())
            roles[role] = {
                "status": "available",
                "tar_at_far": {
                    "0.001": _unit_interval(metrics["tar_at_far_1e-3"], "TAR@FAR 1e-3"),
                    "0.0001": _unit_interval(
                        metrics["tar_at_far_1e-4"], "TAR@FAR 1e-4"
                    ),
                },
                "eer": _unit_interval(metrics["eer"], "EER"),
                "auc": _unit_interval(metrics["auc"], "AUC"),
            }
        report[name] = {
            "status": "available",
            "reason": None,
            "coverage": count,
            "roles": roles,
        }
    return {"schema_version": 1, "recognizers": report}


def _unit_interval(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise R9EvaluatorError(f"{label} must be in [0,1]")
    return parsed


def _cosine_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed < -1.000001 or parsed > 1.000001:
        raise R9EvaluatorError(f"{label} is outside cosine range")
    return max(-1.0, min(1.0, parsed))


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise R9EvaluatorError(f"invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise R9EvaluatorError(f"{label} must contain a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise R9EvaluatorError(
                    f"{path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise R9EvaluatorError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    if not rows:
        raise R9EvaluatorError(f"{label} contains no rows")
    return rows


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_digest_contract(
    payload: Mapping[str, Any], *, digest_field: str, contract_type: str
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_type") != contract_type
    ):
        raise R9EvaluatorError("evaluator digest contract identity mismatch")
    expected = _require_sha256(payload.get(digest_field), digest_field)
    if expected != _canonical_digest(payload, digest_field):
        raise R9EvaluatorError(f"{digest_field} mismatch")


def _canonical_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_value_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_roundtrip(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise R9EvaluatorError(f"{label} is not canonical finite JSON") from exc


def _assert_finite_json(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise R9EvaluatorError(f"{label} contains a non-string key")
            _assert_finite_json(item, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise R9EvaluatorError(f"{label} contains a non-finite number")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise R9EvaluatorError(f"{label} contains a non-JSON value")


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise R9EvaluatorError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise R9EvaluatorError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise R9EvaluatorError(f"{label} must be finite")
    return parsed


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R9EvaluatorError(f"{label} must be a lowercase SHA256")
    return value


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _asset_manifest_sha256(samples: Sequence[SampleEvidence], role: str) -> str:
    return hashlib.sha256(
        "".join(
            f"{sample.sample_id}\t{_sha256_file(getattr(sample, role))}\n"
            for sample in samples
        ).encode("utf-8")
    ).hexdigest()


def _require_contained(root: Path, path: Path, label: str, *, must_exist: bool) -> Path:
    root_resolved = root.resolve()
    resolved = path.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise R9EvaluatorError(f"{label} escapes repository root: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
