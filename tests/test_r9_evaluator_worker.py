from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

from safa.evaluation.r9_evaluator_worker import (
    EvaluatorDependencies,
    ProductionEvaluatorConfig,
    R9EvaluatorError,
    R9ProductionEvaluators,
    _production_face_analyzer_factory,
    _production_quality_backend,
    build_worker_request,
    execute_worker_request,
)
from safa.evaluation.r9_phase_results import (
    ArcFaceEvaluationRequest,
    HeldoutEvaluationRequest,
    QualityEvaluationRequest,
    SampleEvidence,
)


SHA = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker_contract() -> dict[str, str]:
    implementation = Path(
        sys.modules[R9ProductionEvaluators.__module__].__file__
    ).resolve()
    wrapper = implementation.parents[3] / "scripts" / "run_r9_phase_evaluator.py"
    return {
        "path": str(wrapper),
        "sha256": _sha(wrapper),
        "implementation_path": str(implementation),
        "implementation_sha256": _sha(implementation),
    }


def _quality_script_contract(root: Path, source: str | None = None) -> dict[str, str]:
    script = root / "scripts" / "eval_generation_quality.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        source or ("def evaluate_generation_quality(**kwargs):\n    return kwargs\n"),
        encoding="utf-8",
    )
    return {
        "path": str(script.relative_to(root)),
        "sha256": _sha(script),
    }


def _sample_digest(ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in ids).encode()
    ).hexdigest()


def _contract_digest(payload: dict[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _per_sample_metrics(ids: list[str]) -> dict[str, Any]:
    niqe_values = [3.9 + 0.2 * index for index in range(len(ids))]
    sharpness_values = [349.0 + 2.0 * index for index in range(len(ids))]
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_quality_per_sample_metrics_v1",
        "sample_count": len(ids),
        "ordered_sample_id_sha256": _sample_digest(ids),
        "metric_fields": ["niqe", "sharpness"],
        "rows": [
            {"sample_id": sample_id, "niqe": niqe, "sharpness": sharpness}
            for sample_id, niqe, sharpness in zip(
                ids, niqe_values, sharpness_values, strict=True
            )
        ],
    }
    payload["per_sample_metrics_sha256"] = _contract_digest(
        payload, "per_sample_metrics_sha256"
    )
    return payload


def _asset_digest(ids: list[str], paths: list[Path]) -> str:
    return hashlib.sha256(
        "".join(
            f"{sample_id}\t{_sha(path)}\n"
            for sample_id, path in zip(ids, paths, strict=True)
        ).encode()
    ).hexdigest()


def _contract_digest(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _image(path: Path, value: int) -> None:
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def _fake_arcface_profile_inputs() -> dict[str, tuple[str, list[Any]]]:
    return {
        "1k3d68.onnx": ("data", ["None", 3, 192, 192]),
        "2d106det.onnx": ("input.1", ["None", 3, 192, 192]),
        "det_10g.onnx": ("input.1", [1, 3, "?", "?"]),
        "genderage.onnx": ("data", ["None", 3, 96, 96]),
        "w600k_r50.onnx": ("input.1", ["None", 3, 112, 112]),
    }


def _fake_arcface_resolved_shape(filename: str) -> list[int]:
    _, metadata_shape = _fake_arcface_profile_inputs()[filename]
    if filename == "det_10g.onnx":
        return [1, 3, 224, 224]
    return [1, 3, int(metadata_shape[2]), int(metadata_shape[3])]


def _fake_arcface_profile_events(filename: str) -> list[list[str]]:
    if filename == "det_10g.onnx":
        return [
            ["cuda_conv_kernel_time", "Conv", "CUDAExecutionProvider"],
            ["cpu_shape_kernel_time", "Shape", "CPUExecutionProvider"],
        ]
    return [
        ["cuda_kernel_time", "Conv", "CUDAExecutionProvider"],
        ["cuda_kernel_time", "Conv", "CUDAExecutionProvider"],
    ]


def _profile_event_digest(events: list[list[str]]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(events), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _value_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fake_provider_options() -> dict[str, dict[str, str]]:
    return {
        "CUDAExecutionProvider": {
            "cudnn_conv_algo_search": "DEFAULT",
            "device_id": "runtime",
            "use_tf32": "0",
        },
        "CPUExecutionProvider": {},
    }


def _fake_session_options_projection() -> dict[str, str]:
    return {
        "enable_cpu_mem_arena": "True",
        "enable_mem_pattern": "True",
        "enable_mem_reuse": "True",
        "execution_mode": "ExecutionMode.ORT_SEQUENTIAL",
        "execution_order": "ExecutionOrder.DEFAULT",
        "graph_optimization_level": "GraphOptimizationLevel.ORT_ENABLE_ALL",
        "inter_op_num_threads": "0",
        "intra_op_num_threads": "0",
        "log_severity_level": "-1",
        "log_verbosity_level": "0",
        "logid": "",
        "optimized_model_filepath": "",
        "use_deterministic_compute": "False",
        "use_per_session_threads": "True",
    }


def _add_arcface_execution_provenance(root: Path, contract: dict[str, Any]) -> None:
    artifact_root = root / "artifacts" / "arcface-execution-probe"
    artifact_root.mkdir(parents=True, exist_ok=True)
    probe_path = artifact_root / "probe.json"
    claim_path = artifact_root / "claim.json"
    result_path = artifact_root / "result.json"
    probe = {
        "schema_version": 1,
        "contract_type": "safa_r9_arcface_execution_probe_v1",
        "cuda_visible_devices": "GPU-test",
        "runtime_device_id": 0,
        "execution": contract["execution"],
    }
    probe_path.write_text(json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8")
    probe_sha256 = _sha(probe_path)
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_bootstrap_resource_smoke_claim_v1",
        "kind": "arcface_profile",
        "probe_output": str(probe_path.resolve()),
        "retry_allowed": False,
    }
    claim["bootstrap_claim_sha256"] = _contract_digest(claim, "bootstrap_claim_sha256")
    claim_path.write_text(json.dumps(claim, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "contract_type": "safa_r9_bootstrap_resource_smoke_result_v1",
        "bootstrap_claim_sha256": claim["bootstrap_claim_sha256"],
        "status": "succeeded",
        "failure_reason": None,
        "returncode": 0,
        "retry_allowed": False,
        "probe_output_sha256": probe_sha256,
    }
    result["bootstrap_result_sha256"] = _contract_digest(
        result, "bootstrap_result_sha256"
    )
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    contract["execution_probe"] = {
        "path": str(probe_path.relative_to(root)),
        "sha256": probe_sha256,
        "bootstrap_claim_path": str(claim_path.relative_to(root)),
        "bootstrap_claim_sha256": claim["bootstrap_claim_sha256"],
        "bootstrap_claim_file_sha256": _sha(claim_path),
        "bootstrap_result_path": str(result_path.relative_to(root)),
        "bootstrap_result_sha256": result["bootstrap_result_sha256"],
        "bootstrap_result_file_sha256": _sha(result_path),
    }


def _arcface_contract(root: Path) -> dict[str, Any]:
    model_root = root / "insightface"
    asset_root = model_root / "models" / "buffalo_l"
    asset_root.mkdir(parents=True)
    assets = {}
    for filename in (
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    ):
        path = asset_root / filename
        path.write_bytes(filename.encode())
        assets[filename] = _sha(path)
    provider_options = _fake_provider_options()
    session_projection = _fake_session_options_projection()
    contract = {
        "model_name": "buffalo_l",
        "model_root": str(model_root),
        "det_size": [224, 224],
        "provider": "CUDAExecutionProvider",
        "insightface_version": "0.7.3",
        "onnxruntime_version": "1.26.0",
        "assets": assets,
        "execution": {
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
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
                    "excluded_session_option_fields": [
                        "enable_profiling",
                        "profile_file_prefix",
                    ],
                    "session_options_projection_fields": list(session_projection),
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
                    filename: {
                        "input_name": input_name,
                        "input_metadata_shape": input_metadata_shape,
                        "input_shape": _fake_arcface_resolved_shape(filename),
                        "input_dtype": "float32",
                        "node_assignment_counts": {
                            "CUDAExecutionProvider": len(
                                [
                                    event
                                    for event in _fake_arcface_profile_events(filename)
                                    if event[2] == "CUDAExecutionProvider"
                                ]
                            ),
                            "CPUExecutionProvider": len(
                                [
                                    event
                                    for event in _fake_arcface_profile_events(filename)
                                    if event[2] == "CPUExecutionProvider"
                                ]
                            ),
                        },
                        "provider_options": provider_options,
                        "provider_options_sha256": _value_digest(provider_options),
                        "session_options_projection": session_projection,
                        "session_options_projection_sha256": _value_digest(
                            session_projection
                        ),
                        "ordered_node_events_sha256": _profile_event_digest(
                            _fake_arcface_profile_events(filename)
                        ),
                    }
                    for filename, (input_name, input_metadata_shape) in (
                        _fake_arcface_profile_inputs().items()
                    )
                },
            },
        },
    }
    _add_arcface_execution_provenance(root, contract)
    return contract


def _fixture(
    tmp_path: Path,
) -> tuple[ProductionEvaluatorConfig, tuple[SampleEvidence, ...], Path]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "artifacts").mkdir()
    source_root = tmp_path / "affectnet"
    source_root.mkdir()
    ids = ["sample-a", "sample-b"]
    samples = []
    source_rows = []
    for index, sample_id in enumerate(ids):
        source = source_root / f"source-{index}.png"
        native = root / f"native-{index}.png"
        candidate = root / f"candidate-{index}.png"
        _image(source, 10 + index)
        _image(native, 30 + index)
        _image(candidate, 50 + index)
        samples.append(
            SampleEvidence(
                sample_id,
                source,
                native,
                candidate,
                _sha(source),
                _sha(native),
                _sha(candidate),
            )
        )
        source_rows.append({"sample_id": sample_id, "image_path": str(source)})
    (root / "source_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps({"sample_id": value}) + "\n" for value in ids),
        encoding="utf-8",
    )
    config = ProductionEvaluatorConfig(
        repo_root=root,
        device="cuda:0",
        work_root=root / "artifacts" / "evaluator-work",
        quality_script=_quality_script_contract(root),
        arcface=_arcface_contract(root),
        worker_contract=_worker_contract(),
        batch_size=2,
    )
    return config, tuple(samples), manifest


class FakeQualityBackend:
    def __init__(self, mutation: str | None = None) -> None:
        self.generated_dir: Path | None = None
        self.mutation = mutation

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        generated_dir = Path(kwargs["generated_dir"])
        self.generated_dir = generated_dir
        manifest_rows = [
            json.loads(line)
            for line in Path(kwargs["sample_id_manifest"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        real_rows = [
            json.loads(line)
            for line in Path(kwargs["real_index"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        generated_rows = [
            json.loads(line)
            for line in Path(kwargs["per_sample_jsonl"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        ids = [row["sample_id"] for row in manifest_rows]
        real_paths = [Path(row["image_path"]) for row in real_rows]
        generated_paths = [Path(row["generated"]) for row in generated_rows]
        assert sorted(generated_dir.iterdir()) == sorted(generated_paths)
        payload = {
            "metrics": list(kwargs["metrics"]),
            "num_generated": len(ids),
            "num_real": len(ids),
            "sample_id_manifest": str(kwargs["sample_id_manifest"]),
            "sample_id_count": len(ids),
            "sample_id_sha256": _sample_digest(ids),
            "quality_contract": {
                "schema_version": 1,
                "metrics": list(kwargs["metrics"]),
                "sample_id_manifest_sha256": _sha(kwargs["sample_id_manifest"]),
                "per_sample_jsonl_sha256": _sha(kwargs["per_sample_jsonl"]),
                "real_asset_manifest_sha256": _asset_digest(ids, real_paths),
                "generated_asset_manifest_sha256": _asset_digest(ids, generated_paths),
            },
            "fid": 10.0,
            "kid_mean": 0.01,
            "kid_std": 0.001,
            "iqa": {"method": "niqe", "mean": 4.0, "std": 0.2},
            "sharpness": {
                "definition": "grayscale_laplacian_variance",
                "mean": 350.0,
                "median": 350.0,
                "std": 1.0,
                "p05": 349.0,
                "p10": 349.0,
                "p90": 351.0,
                "p95": 351.0,
            },
            "per_sample_metrics": _per_sample_metrics(ids),
        }
        per_sample_metrics = payload["per_sample_metrics"]
        if self.mutation == "digest":
            per_sample_metrics["per_sample_metrics_sha256"] = "0" * 64
        elif self.mutation == "order":
            per_sample_metrics["rows"].reverse()
            per_sample_metrics["per_sample_metrics_sha256"] = _contract_digest(
                per_sample_metrics, "per_sample_metrics_sha256"
            )
        elif self.mutation == "count":
            per_sample_metrics["sample_count"] -= 1
            per_sample_metrics["per_sample_metrics_sha256"] = _contract_digest(
                per_sample_metrics, "per_sample_metrics_sha256"
            )
        elif self.mutation == "nonfinite":
            per_sample_metrics["rows"][0]["niqe"] = float("inf")
        elif self.mutation == "summary":
            payload["iqa"]["mean"] += 1.0
        Path(kwargs["output"]).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
        return payload


class FakeAnalyzer:
    def __init__(self, counts: list[int]) -> None:
        self.counts = iter(counts)
        self.index = 0

    def get(self, image: Any) -> list[Any]:
        import numpy as np

        count = next(self.counts)
        self.index += 1
        return [
            SimpleNamespace(
                embedding=np.asarray([float(self.index), 1.0], dtype=np.float32)
            )
            for _ in range(count)
        ]


def _dependencies(
    quality: FakeQualityBackend, analyzer: FakeAnalyzer
) -> EvaluatorDependencies:
    def forbidden(**kwargs: Any) -> Any:
        raise AssertionError(f"unexpected heldout backend call: {kwargs}")

    return EvaluatorDependencies(
        quality_backend=quality,
        face_analyzer_factory=lambda contract, device: analyzer,
        representation_backend=forbidden,
        recognizer_backend=forbidden,
    )


def _quality_request(
    samples: tuple[SampleEvidence, ...], manifest: Path
) -> QualityEvaluationRequest:
    source_index = manifest.parent / "source_index.jsonl"
    return QualityEvaluationRequest(
        phase="calibrate",
        logical_run_id="arm_seed_7",
        arm_id="arm",
        seed=7,
        image_role="candidate",
        manifest_path=manifest,
        source_index_path=source_index,
        source_index_sha256=_sha(source_index),
        samples=samples,
        algorithm_config_sha256="1" * 64,
        runner_arm_config_sha256="2" * 64,
        semantic_output_sha256="3" * 64,
        evidence_binding_sha256="4" * 64,
        generation_result_set_sha256="5" * 64,
        per_sample_set_sha256="6" * 64,
    )


def _arcface_request(
    config: ProductionEvaluatorConfig, samples: tuple[SampleEvidence, ...]
) -> ArcFaceEvaluationRequest:
    source_index = config.repo_root / "source_index.jsonl"
    return ArcFaceEvaluationRequest(
        phase="calibrate",
        logical_run_id="arm_seed_7",
        arm_id="arm",
        seed=7,
        source_index_path=source_index,
        source_index_sha256=_sha(source_index),
        samples=samples,
    )


def test_quality_uses_exact_hardlink_view_and_cleans_it(tmp_path: Path) -> None:
    config, samples, manifest = _fixture(tmp_path)
    backend = FakeQualityBackend()
    evaluator = R9ProductionEvaluators(
        config,
        _dependencies(backend, FakeAnalyzer([1] * 6)),
    )
    result = evaluator.quality(_quality_request(samples, manifest))
    binding = result["r9_evidence_binding"]
    assert binding == {
        "schema_version": 1,
        "algorithm_config_sha256": "1" * 64,
        "runner_arm_config_sha256": "2" * 64,
        "semantic_output_sha256": "3" * 64,
        "evidence_binding_sha256": "4" * 64,
        "generation_result_set_sha256": "5" * 64,
        "per_sample_set_sha256": "6" * 64,
        "manifest_sha256": _sha(manifest),
        "source_index_sha256": _sha(config.repo_root / "source_index.jsonl"),
        "ordered_sample_id_sha256": _sample_digest(["sample-a", "sample-b"]),
        "real_asset_manifest_sha256": _asset_digest(
            ["sample-a", "sample-b"], [sample.source for sample in samples]
        ),
        "generated_asset_manifest_sha256": _asset_digest(
            ["sample-a", "sample-b"], [sample.candidate for sample in samples]
        ),
    }
    assert backend.generated_dir is not None
    assert not backend.generated_dir.exists()
    assert all(sample.candidate.is_file() for sample in samples)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("digest", "per_sample_metrics_sha256 mismatch"),
        ("order", "manifest order"),
        ("count", "count mismatch"),
        ("nonfinite", "finite JSON"),
        ("summary", "NIQE summary mismatch"),
    ],
)
def test_quality_rejects_invalid_per_sample_metrics(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config, samples, manifest = _fixture(tmp_path)
    evaluator = R9ProductionEvaluators(
        config,
        _dependencies(FakeQualityBackend(mutation), FakeAnalyzer([1] * 6)),
    )
    with pytest.raises(R9EvaluatorError, match=message):
        evaluator.quality(_quality_request(samples, manifest))


def test_quality_rejects_manifest_order_mismatch(tmp_path: Path) -> None:
    config, samples, manifest = _fixture(tmp_path)
    with pytest.raises(R9EvaluatorError, match="manifest order"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).quality(_quality_request(tuple(reversed(samples)), manifest))


def test_arcface_omits_cosines_for_any_non_exact_one_role(tmp_path: Path) -> None:
    config, samples, _ = _fixture(tmp_path)
    analyzer = FakeAnalyzer([1, 1, 1, 1, 0, 1])
    rows = R9ProductionEvaluators(
        config,
        _dependencies(FakeQualityBackend(), analyzer),
    ).arcface(_arcface_request(config, samples))
    assert set(rows[0]) == {
        "sample_id",
        "source_face_count",
        "native_face_count",
        "candidate_face_count",
        "source_native_cosine",
        "source_candidate_cosine",
    }
    assert set(rows[1]) == {
        "sample_id",
        "source_face_count",
        "native_face_count",
        "candidate_face_count",
    }
    assert rows[1]["native_face_count"] == 0


def test_arcface_rehashes_index_and_all_three_roles(tmp_path: Path) -> None:
    config, samples, _ = _fixture(tmp_path)
    samples[0].native.write_bytes(b"tampered")
    with pytest.raises(R9EvaluatorError, match="native image digest mismatch"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).arcface(_arcface_request(config, samples))

    request = replace(_arcface_request(config, samples), source_index_sha256="9" * 64)
    with pytest.raises(R9EvaluatorError, match="source index digest mismatch"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).arcface(request)


@pytest.mark.parametrize("role", ["source", "candidate"])
def test_arcface_rejects_source_and_candidate_byte_tamper(
    tmp_path: Path, role: str
) -> None:
    config, samples, _ = _fixture(tmp_path)
    getattr(samples[0], role).write_bytes(b"tampered")
    with pytest.raises(R9EvaluatorError, match=rf"{role} image digest mismatch"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).arcface(_arcface_request(config, samples))


def test_arcface_rejects_changed_source_index_bytes(tmp_path: Path) -> None:
    config, samples, _ = _fixture(tmp_path)
    request = _arcface_request(config, samples)
    request.source_index_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(R9EvaluatorError, match="source index digest mismatch"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).arcface(request)


def test_arcface_source_must_match_index_and_outputs_must_stay_in_repo(
    tmp_path: Path,
) -> None:
    config, samples, _ = _fixture(tmp_path)
    outside = tmp_path / "outside.png"
    _image(outside, 77)
    wrong_source = replace(samples[0], source=outside, source_sha256=_sha(outside))
    with pytest.raises(R9EvaluatorError, match="source path disagrees"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).arcface(_arcface_request(config, (wrong_source, samples[1])))

    outside_native = replace(samples[0], native=outside, native_sha256=_sha(outside))
    with pytest.raises(R9EvaluatorError, match="native image escapes"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).arcface(_arcface_request(config, (outside_native, samples[1])))

    outside_candidate = replace(
        samples[0], candidate=outside, candidate_sha256=_sha(outside)
    )
    with pytest.raises(R9EvaluatorError, match="candidate image escapes"):
        R9ProductionEvaluators(
            config,
            _dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        ).arcface(_arcface_request(config, (outside_candidate, samples[1])))


def _install_fake_arcface_modules(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    *,
    first_provider: str = "CUDAExecutionProvider",
    use_tf32: str = "0",
    profile_events: dict[str, list[list[str]]] | None = None,
    metadata_inputs: dict[str, tuple[str, list[Any]]] | None = None,
    direct_provider_option: tuple[str, str] | None = None,
    direct_session_option: tuple[str, Any] | None = None,
) -> type:
    inputs_by_asset = metadata_inputs or _fake_arcface_profile_inputs()

    class SessionOptions:
        def __init__(self) -> None:
            self.enable_profiling = False
            self.profile_file_prefix = ""
            self.enable_cpu_mem_arena = True
            self.enable_mem_pattern = True
            self.enable_mem_reuse = True
            self.execution_mode = "ExecutionMode.ORT_SEQUENTIAL"
            self.execution_order = "ExecutionOrder.DEFAULT"
            self.graph_optimization_level = "GraphOptimizationLevel.ORT_ENABLE_ALL"
            self.inter_op_num_threads = 0
            self.intra_op_num_threads = 0
            self.log_severity_level = -1
            self.log_verbosity_level = 0
            self.logid = ""
            self.optimized_model_filepath = ""
            self.use_deterministic_compute = False
            self.use_per_session_threads = True

    class Session:
        def __init__(
            self,
            filename: str,
            session_options: SessionOptions,
            provider_options: dict[str, str],
        ) -> None:
            self.filename = filename
            self.session_options = session_options
            self.provider_options = dict(provider_options)
            self.provider_options["use_tf32"] = use_tf32
            self.run_count = 0

        def get_providers(self) -> list[str]:
            return [first_provider, "CPUExecutionProvider"]

        def get_provider_options(self) -> dict[str, dict[str, str]]:
            return {
                "CUDAExecutionProvider": self.provider_options,
                "CPUExecutionProvider": {},
            }

        def get_session_options(self) -> SessionOptions:
            return self.session_options

        def get_inputs(self) -> list[Any]:
            name, shape = inputs_by_asset[self.filename]
            return [SimpleNamespace(name=name, shape=shape, type="tensor(float)")]

        def run(self, output_names: Any, feeds: dict[str, Any]) -> list[Any]:
            import numpy as np

            assert output_names is None
            name, _ = inputs_by_asset[self.filename]
            shape = contract["execution"]["probe"]["assets"][self.filename][
                "input_shape"
            ]
            assert set(feeds) == {name}
            assert feeds[name].shape == tuple(shape)
            assert feeds[name].dtype == np.float32
            assert not feeds[name].any()
            self.run_count += 1
            return []

        def end_profiling(self) -> str:
            assert self.session_options.enable_profiling is True
            assert self.run_count == 1
            events_by_asset = profile_events or {
                filename: _fake_arcface_profile_events(filename)
                for filename in _fake_arcface_profile_inputs()
            }
            events = [
                {
                    "cat": "Node",
                    "name": name,
                    "args": {"op_name": op_name, "provider": provider},
                }
                for name, op_name, provider in reversed(events_by_asset[self.filename])
            ]
            events.append({"cat": "Session", "name": "ignored", "args": {}})
            path = Path(
                f"{self.session_options.profile_file_prefix}-{self.filename}.json"
            )
            path.write_text(json.dumps(events), encoding="utf-8")
            return str(path)

    class FaceAnalysis:
        instance: Any = None
        direct_sessions: list[Any] = []

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["name"] == contract["model_name"]
            assert kwargs["root"] == contract["model_root"]
            assert kwargs["providers"] == contract["execution"]["providers"]
            assert kwargs["provider_options"][1] == {}
            assert "sess_options" not in kwargs
            self.model_dir = str(
                Path(contract["model_root"]) / "models" / contract["model_name"]
            )
            task_names = {
                "1k3d68.onnx": "landmark_3d_68",
                "2d106det.onnx": "landmark_2d_106",
                "det_10g.onnx": "detection",
                "genderage.onnx": "genderage",
                "w600k_r50.onnx": "recognition",
            }
            self.models = {
                task_names[filename]: SimpleNamespace(
                    model_file=str(Path(self.model_dir) / filename),
                    taskname=task_names[filename],
                    session=Session(
                        filename,
                        SessionOptions(),
                        kwargs["provider_options"][0],
                    ),
                )
                for filename in sorted(contract["assets"])
            }
            self.det_model = self.models["detection"]

        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            self.prepared = (ctx_id, det_size)
            type(self).instance = self

    insightface = ModuleType("insightface")
    insightface.__version__ = "0.7.3"
    insightface.__path__ = []
    app = ModuleType("insightface.app")
    app.FaceAnalysis = FaceAnalysis
    onnxruntime = ModuleType("onnxruntime")
    onnxruntime.__version__ = "1.26.0"
    onnxruntime.SessionOptions = SessionOptions

    def inference_session(model_path: str, **kwargs: Any) -> Session:
        assert kwargs["providers"] == contract["execution"]["providers"]
        assert kwargs["provider_options"][1] == {}
        session = Session(
            Path(model_path).name,
            kwargs["sess_options"],
            kwargs["provider_options"][0],
        )
        if direct_provider_option is not None:
            key, value = direct_provider_option
            session.provider_options[key] = value
        if direct_session_option is not None:
            key, value = direct_session_option
            setattr(session.session_options, key, value)
        FaceAnalysis.direct_sessions.append(session)
        return session

    onnxruntime.InferenceSession = inference_session
    monkeypatch.setitem(sys.modules, "insightface", insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", app)
    monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
    return FaceAnalysis


def test_production_arcface_factory_locks_assets_versions_and_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _fixture(tmp_path)
    face_analysis = _install_fake_arcface_modules(monkeypatch, dict(config.arcface))
    analyzer = _production_face_analyzer_factory(config.arcface, "cuda:3")
    assert analyzer is face_analysis.instance
    assert analyzer.model_dir == str(
        Path(config.arcface["model_root"]) / "models" / "buffalo_l"
    )
    assert analyzer.prepared == (3, (224, 224))
    for model in analyzer.models.values():
        assert model.session.get_providers() == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert (
            model.session.get_provider_options()["CUDAExecutionProvider"]["use_tf32"]
            == "0"
        )
        assert model.session.run_count == 0
    assert len(face_analysis.direct_sessions) == 5
    assert all(session.run_count == 1 for session in face_analysis.direct_sessions)


def test_production_arcface_factory_rejects_cpu_session_and_asset_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _fixture(tmp_path)
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        first_provider="CPUExecutionProvider",
    )
    with pytest.raises(R9EvaluatorError, match="provider order mismatch"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")

    asset = Path(config.arcface["model_root"]) / "models" / "buffalo_l" / "det_10g.onnx"
    asset.write_bytes(b"tampered")
    with pytest.raises(R9EvaluatorError, match="asset digest mismatch"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


def test_production_arcface_factory_rejects_tf32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        use_tf32="1",
    )
    with pytest.raises(R9EvaluatorError, match="CUDA option use_tf32 mismatch"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


def test_production_arcface_factory_rejects_profile_assignment_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    changed = {
        filename: _fake_arcface_profile_events(filename)
        for filename in _fake_arcface_profile_inputs()
    }
    changed["w600k_r50.onnx"] = changed["w600k_r50.onnx"] + [
        ["unexpected_cpu_kernel_time", "Shape", "CPUExecutionProvider"]
    ]
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        profile_events=changed,
    )
    with pytest.raises(R9EvaluatorError, match="node assignment counts mismatch"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


def test_production_arcface_factory_rejects_unregistered_node_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    changed = {
        filename: _fake_arcface_profile_events(filename)
        for filename in _fake_arcface_profile_inputs()
    }
    changed["genderage.onnx"] = changed["genderage.onnx"] + [
        ["foreign_kernel_time", "Identity", "ForeignExecutionProvider"]
    ]
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        profile_events=changed,
    )
    with pytest.raises(R9EvaluatorError, match="unregistered Node provider"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


@pytest.mark.parametrize(
    ("provider_option", "session_option"),
    [
        (("new_default", "1"), None),
        (None, ("logid", "direct-only")),
    ],
)
def test_production_arcface_factory_rejects_incomplete_session_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_option: tuple[str, str] | None,
    session_option: tuple[str, Any] | None,
) -> None:
    config, _, _ = _fixture(tmp_path)
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        direct_provider_option=provider_option,
        direct_session_option=session_option,
    )
    with pytest.raises(
        R9EvaluatorError, match="matched direct session binding mismatch"
    ):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


def test_production_arcface_factory_rejects_profile_event_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    changed = {
        filename: _fake_arcface_profile_events(filename)
        for filename in _fake_arcface_profile_inputs()
    }
    changed["genderage.onnx"] = [
        ["changed_cuda_kernel_time", "Conv", "CUDAExecutionProvider"],
        ["changed_cuda_kernel_time", "Conv", "CUDAExecutionProvider"],
    ]
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        profile_events=changed,
    )
    with pytest.raises(R9EvaluatorError, match="node events digest mismatch"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


def test_production_arcface_factory_locks_raw_dynamic_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    changed = {
        filename: (name, list(shape))
        for filename, (name, shape) in _fake_arcface_profile_inputs().items()
    }
    changed["1k3d68.onnx"][1][0] = None
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        metadata_inputs=changed,
    )
    with pytest.raises(R9EvaluatorError, match="input_metadata_shape mismatch"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


def test_production_arcface_factory_rejects_unregistered_dynamic_spatial_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    changed = {
        filename: (name, list(shape))
        for filename, (name, shape) in _fake_arcface_profile_inputs().items()
    }
    changed["genderage.onnx"][1][2] = "?"
    _install_fake_arcface_modules(
        monkeypatch,
        dict(config.arcface),
        metadata_inputs=changed,
    )
    with pytest.raises(R9EvaluatorError, match="non-detector spatial metadata"):
        _production_face_analyzer_factory(config.arcface, "cuda:0")


def test_arcface_contract_rejects_missing_partition_evidence(tmp_path: Path) -> None:
    contract = _arcface_contract(tmp_path)
    contract["execution"]["probe"]["assets"]["det_10g.onnx"]["node_assignment_counts"][
        "CPUExecutionProvider"
    ] = 0
    root = tmp_path / "repo"
    root.mkdir()
    wrapper_contract = _worker_contract()
    with pytest.raises(R9EvaluatorError, match="detector probe has no CPU nodes"):
        ProductionEvaluatorConfig(
            repo_root=root,
            device="cuda:0",
            work_root=root / "work",
            quality_script=_quality_script_contract(root),
            arcface=contract,
            worker_contract=wrapper_contract,
        )


def test_arcface_contract_rejects_unregistered_dynamic_shape(tmp_path: Path) -> None:
    contract = _arcface_contract(tmp_path)
    contract["execution"]["probe"]["assets"]["w600k_r50.onnx"]["input_metadata_shape"][
        2
    ] = "?"
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(R9EvaluatorError, match="non-detector spatial metadata"):
        ProductionEvaluatorConfig(
            repo_root=root,
            device="cuda:0",
            work_root=root / "work",
            quality_script=_quality_script_contract(root),
            arcface=contract,
            worker_contract=_worker_contract(),
        )


def _production_config_for_arcface_contract(
    root: Path, contract: dict[str, Any]
) -> ProductionEvaluatorConfig:
    return ProductionEvaluatorConfig(
        repo_root=root,
        device="cuda:0",
        work_root=root / "work",
        quality_script=_quality_script_contract(root),
        arcface=contract,
        worker_contract=_worker_contract(),
    )


def _provenance_path(root: Path, contract: dict[str, Any], field: str) -> Path:
    return root / contract["execution_probe"][field]


def test_arcface_execution_probe_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    contract = _arcface_contract(root)
    probe = _provenance_path(root, contract, "path")
    outside = tmp_path / "outside-probe.json"
    outside.write_bytes(probe.read_bytes())
    contract["execution_probe"]["path"] = str(outside)
    with pytest.raises(R9EvaluatorError, match="escapes repository root"):
        _production_config_for_arcface_contract(root, contract)


def test_arcface_execution_probe_rejects_claim_semantic_tamper(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    contract = _arcface_contract(root)
    claim_path = _provenance_path(root, contract, "bootstrap_claim_path")
    result_path = _provenance_path(root, contract, "bootstrap_result_path")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["kind"] = "quality"
    claim["bootstrap_claim_sha256"] = _contract_digest(claim, "bootstrap_claim_sha256")
    claim_path.write_text(json.dumps(claim, sort_keys=True) + "\n", encoding="utf-8")
    contract["execution_probe"]["bootstrap_claim_sha256"] = claim[
        "bootstrap_claim_sha256"
    ]
    contract["execution_probe"]["bootstrap_claim_file_sha256"] = _sha(claim_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["bootstrap_claim_sha256"] = claim["bootstrap_claim_sha256"]
    result["bootstrap_result_sha256"] = _contract_digest(
        result, "bootstrap_result_sha256"
    )
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    contract["execution_probe"]["bootstrap_result_sha256"] = result[
        "bootstrap_result_sha256"
    ]
    contract["execution_probe"]["bootstrap_result_file_sha256"] = _sha(result_path)
    with pytest.raises(R9EvaluatorError, match="claim policy mismatch"):
        _production_config_for_arcface_contract(root, contract)


def test_arcface_execution_probe_rejects_result_semantic_tamper(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    contract = _arcface_contract(root)
    result_path = _provenance_path(root, contract, "bootstrap_result_path")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "failed"
    result["failure_reason"] = "tampered"
    result["bootstrap_result_sha256"] = _contract_digest(
        result, "bootstrap_result_sha256"
    )
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    contract["execution_probe"]["bootstrap_result_sha256"] = result[
        "bootstrap_result_sha256"
    ]
    contract["execution_probe"]["bootstrap_result_file_sha256"] = _sha(result_path)
    with pytest.raises(R9EvaluatorError, match="did not succeed exactly once"):
        _production_config_for_arcface_contract(root, contract)


def test_arcface_execution_probe_rejects_probe_byte_tamper(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    contract = _arcface_contract(root)
    probe_path = _provenance_path(root, contract, "path")
    probe_path.write_bytes(probe_path.read_bytes() + b" ")
    with pytest.raises(R9EvaluatorError, match="execution probe digest mismatch"):
        _production_config_for_arcface_contract(root, contract)


def test_arcface_execution_probe_rejects_whitespace_only_contract_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    contract = _arcface_contract(root)
    claim_path = _provenance_path(root, contract, "bootstrap_claim_path")
    claim_path.write_bytes(claim_path.read_bytes() + b" ")
    with pytest.raises(R9EvaluatorError, match="claim file digest mismatch"):
        _production_config_for_arcface_contract(root, contract)


def test_arcface_execution_probe_rejects_execution_chain_tamper(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    contract = _arcface_contract(root)
    probe_path = _provenance_path(root, contract, "path")
    result_path = _provenance_path(root, contract, "bootstrap_result_path")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["execution"]["probe"]["assets"]["1k3d68.onnx"]["input_name"] = "changed"
    probe_path.write_text(json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8")
    probe_sha256 = _sha(probe_path)
    contract["execution_probe"]["sha256"] = probe_sha256
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["probe_output_sha256"] = probe_sha256
    result["bootstrap_result_sha256"] = _contract_digest(
        result, "bootstrap_result_sha256"
    )
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    contract["execution_probe"]["bootstrap_result_sha256"] = result[
        "bootstrap_result_sha256"
    ]
    contract["execution_probe"]["bootstrap_result_file_sha256"] = _sha(result_path)
    with pytest.raises(R9EvaluatorError, match="disagrees with runtime contract"):
        _production_config_for_arcface_contract(root, contract)


def test_quality_script_binding_rejects_path_escape(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    outside = tmp_path / "outside_quality.py"
    outside.write_text(
        "def evaluate_generation_quality(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )
    with pytest.raises(R9EvaluatorError, match="escapes repository root"):
        replace(
            config,
            quality_script={"path": str(outside), "sha256": _sha(outside)},
        )


def test_quality_script_binding_rejects_missing_file(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="quality script does not exist"):
        replace(
            config,
            quality_script={
                "path": "scripts/missing_quality.py",
                "sha256": "1" * 64,
            },
        )


def test_production_quality_backend_rehashes_locked_script(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    script = Path(config.quality_script["path"])
    script.write_bytes(script.read_bytes() + b"\n")
    with pytest.raises(R9EvaluatorError, match="quality script digest mismatch"):
        _production_quality_backend(quality_script=config.quality_script)


def test_production_quality_backend_rejects_missing_callable(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    script = config.repo_root / "scripts" / "missing_callable.py"
    script.write_text("VALUE = 7\n", encoding="utf-8")
    locked = replace(
        config,
        quality_script={
            "path": str(script.relative_to(config.repo_root)),
            "sha256": _sha(script),
        },
    )
    with pytest.raises(
        R9EvaluatorError,
        match="missing callable evaluate_generation_quality",
    ):
        _production_quality_backend(quality_script=locked.quality_script)


def test_production_quality_backend_locks_module_identity_and_sha(
    tmp_path: Path,
) -> None:
    config, _, _ = _fixture(tmp_path)
    script = config.repo_root / "scripts" / "identity_quality.py"
    script.write_text(
        "def evaluate_generation_quality(**kwargs):\n"
        "    return {\n"
        "        'module_name': __name__,\n"
        "        'module_file': __file__,\n"
        "        'module_origin': __spec__.origin,\n"
        "        'value': kwargs['value'],\n"
        "    }\n",
        encoding="utf-8",
    )
    locked = replace(
        config,
        quality_script={
            "path": str(script.relative_to(config.repo_root)),
            "sha256": _sha(script),
        },
    )
    original_sys_path = list(sys.path)
    result = _production_quality_backend(
        quality_script=locked.quality_script,
        value=11,
    )
    assert result == {
        "module_name": f"_safa_r9_quality_{_sha(script)}",
        "module_file": str(script.resolve()),
        "module_origin": str(script.resolve()),
        "value": 11,
    }
    assert sys.path == original_sys_path


def test_worker_request_output_digests_and_o_excl(tmp_path: Path) -> None:
    config, samples, manifest = _fixture(tmp_path)
    quality = FakeQualityBackend()
    request = build_worker_request(
        "quality", _quality_request(samples, manifest), config=config
    )
    assert request["config"]["quality_script"] == dict(config.quality_script)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"
    output = execute_worker_request(
        request_path,
        output_path,
        dependencies=_dependencies(quality, FakeAnalyzer([1] * 6)),
    )
    assert output_path.is_file()
    assert len(output["evaluator_output_sha256"]) == 64
    assert (
        output["arcface_contract_sha256"]
        == hashlib.sha256(
            json.dumps(
                dict(config.arcface), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    assert output["worker_contract"] == dict(config.worker_contract)
    assert output["quality_script_sha256"] == config.quality_script["sha256"]
    with pytest.raises(FileExistsError):
        execute_worker_request(
            request_path,
            output_path,
            dependencies=_dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        )


def test_worker_rejects_request_digest_tampering(tmp_path: Path) -> None:
    config, samples, manifest = _fixture(tmp_path)
    request = build_worker_request(
        "quality", _quality_request(samples, manifest), config=config
    )
    request["payload"]["seed"] = 8
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(R9EvaluatorError, match="request_sha256 mismatch"):
        execute_worker_request(
            request_path,
            tmp_path / "output.json",
            dependencies=_dependencies(FakeQualityBackend(), FakeAnalyzer([1] * 6)),
        )


def _heldout_contracts(
    root: Path, arm_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = {
        "schema_version": 1,
        "contract_type": "safa_r9_selection_v1",
        "winner": {"arm_id": arm_id, "config_sha256": "7" * 64},
        "winner_locked": True,
        "reselection_allowed": False,
    }
    selection["selection_sha256"] = _contract_digest(selection, "selection_sha256")
    assets = {}
    for name in ("e1", "e2", "facenet", "adaface"):
        path = root / f"{name}.pt"
        path.write_bytes(name.encode())
        assets[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha(path),
            "state": "sealed_unrun",
        }
    seal = {
        "schema_version": 1,
        "contract_type": "safa_r9_heldout_seal_v1",
        "selection_sha256": selection["selection_sha256"],
        "winner": dict(selection["winner"]),
        "assets": assets,
        "execution_count": 0,
        "sealed": True,
    }
    seal["heldout_seal_sha256"] = _contract_digest(seal, "heldout_seal_sha256")
    return selection, seal


def _embedding_backend(**kwargs: Any) -> dict[str, list[list[float]]]:
    del kwargs
    return {
        "source": [[1.0, 0.0], [0.0, 1.0]],
        "native": [[0.8, 0.2], [0.2, 0.8]],
        "winner": [[1.0, 0.0], [0.0, 1.0]],
    }


def test_heldout_emits_complete_paired_schema_and_identity_report(
    tmp_path: Path,
) -> None:
    config, samples, _ = _fixture(tmp_path)
    selection, seal = _heldout_contracts(config.repo_root, "winner")
    analyzer = FakeAnalyzer([1] * 6)
    evaluator = R9ProductionEvaluators(
        config,
        EvaluatorDependencies(
            quality_backend=FakeQualityBackend(),
            face_analyzer_factory=lambda contract, device: analyzer,
            representation_backend=_embedding_backend,
            recognizer_backend=_embedding_backend,
        ),
    )
    raw = evaluator.heldout(
        HeldoutEvaluationRequest(
            phase="full",
            arm_id="winner",
            seed=7919,
            source_index_path=config.repo_root / "source_index.jsonl",
            source_index_sha256=_sha(config.repo_root / "source_index.jsonl"),
            samples=samples,
            selection=selection,
            heldout_seal=seal,
        )
    )
    assert set(raw) == {"representations", "recognizers", "identity_report"}
    for name in ("e1", "e2"):
        assert [row["sample_id"] for row in raw["representations"][name]] == [
            "sample-a",
            "sample-b",
        ]
        assert set(raw["representations"][name][0]) == {
            "sample_id",
            "native_cosine",
            "winner_cosine",
        }
    for name in ("facenet", "adaface"):
        recognizer = raw["recognizers"][name]
        assert recognizer["paired_exact_one_count"] == 2
        assert recognizer["failure_sample_ids"] == []
        assert len(recognizer["rows"]) == 2
    identity = raw["identity_report"]
    assert identity["schema_version"] == 1
    assert set(identity["recognizers"]) == {
        "arcface",
        "facenet",
        "adaface",
    }
    for recognizer in identity["recognizers"].values():
        assert recognizer["status"] == "available"
        assert recognizer["reason"] is None
        assert recognizer["coverage"] == 2
        assert set(recognizer["roles"]) == {"native", "winner"}
        for role in recognizer["roles"].values():
            assert set(role) == {"status", "tar_at_far", "eer", "auc"}
            assert role["status"] == "available"
            assert set(role["tar_at_far"]) == {"0.001", "0.0001"}
            assert 0.0 <= role["eer"] <= 1.0
            assert 0.0 <= role["auc"] <= 1.0


def test_heldout_arcface_nonexact_is_unavailable_not_worker_failure(
    tmp_path: Path,
) -> None:
    config, samples, _ = _fixture(tmp_path)
    selection, seal = _heldout_contracts(config.repo_root, "winner")
    raw = R9ProductionEvaluators(
        config,
        EvaluatorDependencies(
            quality_backend=FakeQualityBackend(),
            face_analyzer_factory=lambda contract, device: FakeAnalyzer(
                [1, 0, 1, 1, 1, 1]
            ),
            representation_backend=_embedding_backend,
            recognizer_backend=_embedding_backend,
        ),
    ).heldout(
        HeldoutEvaluationRequest(
            phase="full",
            arm_id="winner",
            seed=7919,
            source_index_path=config.repo_root / "source_index.jsonl",
            source_index_sha256=_sha(config.repo_root / "source_index.jsonl"),
            samples=samples,
            selection=selection,
            heldout_seal=seal,
        )
    )
    assert len(raw["representations"]["e1"]) == 2
    assert raw["recognizers"]["facenet"]["paired_exact_one_count"] == 2
    identity = raw["identity_report"]["recognizers"]
    arcface = identity["arcface"]
    assert arcface == {
        "status": "unavailable",
        "reason": "incomplete_exact_one_face_coverage",
        "coverage": 1,
        "roles": {
            "native": {
                "status": "unavailable",
                "reason": "incomplete_exact_one_face_coverage",
            },
            "winner": {
                "status": "unavailable",
                "reason": "incomplete_exact_one_face_coverage",
            },
        },
    }
    assert identity["facenet"]["status"] == "available"
    assert identity["adaface"]["status"] == "available"


def test_heldout_rehashes_every_sealed_asset(tmp_path: Path) -> None:
    config, samples, _ = _fixture(tmp_path)
    selection, seal = _heldout_contracts(config.repo_root, "winner")
    (config.repo_root / "facenet.pt").write_bytes(b"tampered")
    with pytest.raises(R9EvaluatorError, match="facenet digest mismatch"):
        R9ProductionEvaluators(
            config,
            EvaluatorDependencies(
                quality_backend=FakeQualityBackend(),
                face_analyzer_factory=lambda contract, device: FakeAnalyzer([1] * 6),
                representation_backend=_embedding_backend,
                recognizer_backend=_embedding_backend,
            ),
        ).heldout(
            HeldoutEvaluationRequest(
                phase="full",
                arm_id="winner",
                seed=7919,
                source_index_path=config.repo_root / "source_index.jsonl",
                source_index_sha256=_sha(config.repo_root / "source_index.jsonl"),
                samples=samples,
                selection=selection,
                heldout_seal=seal,
            )
        )


def test_all_worker_payloads_bind_source_index_and_three_role_hashes(
    tmp_path: Path,
) -> None:
    config, samples, manifest = _fixture(tmp_path)
    selection, seal = _heldout_contracts(config.repo_root, "winner")
    source_index = config.repo_root / "source_index.jsonl"
    heldout = HeldoutEvaluationRequest(
        phase="full",
        arm_id="winner",
        seed=7919,
        source_index_path=source_index,
        source_index_sha256=_sha(source_index),
        samples=samples,
        selection=selection,
        heldout_seal=seal,
    )
    requests = {
        "quality": _quality_request(samples, manifest),
        "arcface": _arcface_request(config, samples),
        "heldout": heldout,
    }
    expected_sample_fields = {
        "sample_id",
        "source",
        "native",
        "candidate",
        "source_sha256",
        "native_sha256",
        "candidate_sha256",
    }
    for task, request in requests.items():
        payload = build_worker_request(task, request, config=config)["payload"]
        assert payload["source_index_path"] == str(source_index.resolve())
        assert payload["source_index_sha256"] == _sha(source_index)
        assert all(
            set(sample) == expected_sample_fields for sample in payload["samples"]
        )
