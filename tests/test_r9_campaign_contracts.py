from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import pytest

import safa.evaluation.r9_campaign_contracts as campaign_contracts_module
from safa.evaluation.r9_campaign_contracts import (
    CampaignContractError,
    R9_BOOTSTRAP_ITERATIONS,
    aggregate_seed_metrics,
    build_a_gate_contract,
    build_b_gate_contract,
    build_c_gate_contract,
    build_d_gate_contract,
    build_heldout_seal_contract,
    build_resource_smoke_contract,
    build_selection_contract,
    canonical_campaign_runtime_sha256,
    derive_visual_arm_pass,
    paired_metric_cluster_bootstrap,
    privacy_delta_cluster_bootstrap,
    validate_campaign_runtime,
    validate_diagnose_manifest_contract,
    validate_gate_contract,
    validate_identity_report,
    validate_manifest_contracts,
    validate_resource_smoke_contract,
    validate_selection_contract,
    write_immutable_contract,
)
from safa.evaluation.r9_evaluator_resources import (
    materialize_evaluator_resource_profiles,
)
from safa.evaluation.r9_evaluator_worker import _validate_arcface_contract
from safa.evaluation.r9_semigroup_contracts import (
    R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
    R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SEEDS = (1337, 2027, 3407)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_sha(ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in ids).encode()
    ).hexdigest()


def _canonical_sha(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _value_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _arcface_execution_probe(root: Path, assets: dict[str, str]) -> dict[str, str]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_options = {
        "CUDAExecutionProvider": {
            "cudnn_conv_algo_search": "DEFAULT",
            "device_id": "runtime",
            "use_tf32": "0",
        },
        "CPUExecutionProvider": {},
    }
    session_fields = (
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
    session_projection = {field: f"fixture-{field}" for field in session_fields}
    metadata = {
        "1k3d68.onnx": [None, 3, 192, 192],
        "2d106det.onnx": [None, 3, 192, 192],
        "det_10g.onnx": [1, 3, "?", "?"],
        "genderage.onnx": [None, 3, 96, 96],
        "w600k_r50.onnx": [None, 3, 112, 112],
    }
    probe_assets = {}
    for name in sorted(assets):
        shape = metadata[name]
        resolved = (
            [1, 3, 224, 224] if name == "det_10g.onnx" else [1, 3, shape[2], shape[3]]
        )
        probe_assets[name] = {
            "input_name": "input.1",
            "input_metadata_shape": shape,
            "input_shape": resolved,
            "input_dtype": "float32",
            "node_assignment_counts": {
                "CUDAExecutionProvider": 1,
                "CPUExecutionProvider": 1 if name == "det_10g.onnx" else 0,
            },
            "ordered_node_events_sha256": "f" * 64,
            "provider_options": provider_options,
            "provider_options_sha256": _value_sha(provider_options),
            "session_options_projection": session_projection,
            "session_options_projection_sha256": _value_sha(session_projection),
        }
    execution = {
        "providers": providers,
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
                "session_options_projection_fields": list(session_fields),
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
            "assets": probe_assets,
        },
    }
    artifact_root = root / "artifacts" / "arcface-probe"
    artifact_root.mkdir(parents=True)
    probe_path = artifact_root / "probe.json"
    claim_path = artifact_root / "claim.json"
    result_path = artifact_root / "result.json"
    probe = {
        "schema_version": 1,
        "contract_type": "safa_r9_arcface_execution_probe_v1",
        "cuda_visible_devices": "GPU-fixture",
        "runtime_device_id": 0,
        "execution": execution,
    }
    probe_path.write_text(json.dumps(probe) + "\n", encoding="utf-8")
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_bootstrap_resource_smoke_claim_v1",
        "kind": "arcface_profile",
        "probe_output": str(probe_path.resolve()),
        "retry_allowed": False,
    }
    claim["bootstrap_claim_sha256"] = _canonical_sha(claim)
    claim_path.write_text(json.dumps(claim) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "contract_type": "safa_r9_bootstrap_resource_smoke_result_v1",
        "bootstrap_claim_sha256": claim["bootstrap_claim_sha256"],
        "status": "succeeded",
        "failure_reason": None,
        "returncode": 0,
        "retry_allowed": False,
        "probe_output_sha256": _sha(probe_path),
    }
    result["bootstrap_result_sha256"] = _canonical_sha(result)
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return {
        "path": str(probe_path.relative_to(root)),
        "sha256": _sha(probe_path),
        "bootstrap_claim_path": str(claim_path.relative_to(root)),
        "bootstrap_claim_file_sha256": _sha(claim_path),
        "bootstrap_claim_sha256": str(claim["bootstrap_claim_sha256"]),
        "bootstrap_result_path": str(result_path.relative_to(root)),
        "bootstrap_result_file_sha256": _sha(result_path),
        "bootstrap_result_sha256": str(result["bootstrap_result_sha256"]),
    }


def _write_digest_contract(
    path: Path, payload: dict[str, Any], digest_field: str
) -> dict[str, Any]:
    value = dict(payload)
    value[digest_field] = _canonical_sha(value)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return value


def _rewrite_smoke_request_claim(
    path: Path, **changes: object
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    payload.pop("smoke_request_claim_sha256", None)
    payload["smoke_request_claim_sha256"] = _canonical_sha(payload)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def _evaluator_resource_fixture(
    root: Path,
    *,
    worker_contract: dict[str, str],
    arcface_contract: dict[str, Any],
    quality_script: Path,
    runtime_config: Path,
) -> dict[str, Any]:
    normalized_worker = {
        "path": str((root / worker_contract["path"]).resolve()),
        "sha256": worker_contract["sha256"],
        "implementation_path": str(
            (root / worker_contract["implementation_path"]).resolve()
        ),
        "implementation_sha256": worker_contract["implementation_sha256"],
    }
    normalized_arcface = _validate_arcface_contract(arcface_contract, repo_root=root)
    arcface_sha = _value_sha(normalized_arcface)
    quality_binding = {
        "path": str(quality_script.resolve()),
        "sha256": _sha(quality_script),
    }
    raw_profiles: dict[str, Any] = {}
    for kind in ("arcface", "quality"):
        artifact_root = root / "artifacts" / f"{kind}-smoke"
        artifact_root.mkdir(parents=True)
        request = _write_digest_contract(
            artifact_root / "request.json",
            {
                "schema_version": 1,
                "contract_type": "safa_r9_phase_evaluator_request_v1",
                "task": kind,
                "config": {
                    "repo_root": str(root.resolve()),
                    "device": "cuda:0",
                    "work_root": str((artifact_root / "work").resolve()),
                    "arcface": normalized_arcface,
                    "quality_script": quality_binding,
                    "worker_contract": normalized_worker,
                    "batch_size": 16,
                },
                "payload": {},
            },
            "evaluator_request_sha256",
        )
        claim = _write_digest_contract(
            artifact_root / "request_claim.json",
            {
                "schema_version": 1,
                "contract_type": "safa_r9_evaluator_resource_smoke_request_v1",
                "kind": kind,
                "sample_count": 64,
                "worker_contract": normalized_worker,
                "arcface_contract_sha256": arcface_sha,
                "quality_script_sha256": quality_binding["sha256"],
                "evaluator_request_sha256": request["evaluator_request_sha256"],
                "runtime_config": str(runtime_config.resolve()),
                "runtime_config_sha256": _sha(runtime_config),
                "retry_allowed": False,
            },
            "smoke_request_claim_sha256",
        )
        execution_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract_type": (
                "safa_r9_evaluator_resource_smoke_execution_v1"
                if kind == "arcface"
                else "safa_r9_quality_bootstrap_smoke_execution_v1"
            ),
            "evaluator_request_sha256": request["evaluator_request_sha256"],
            "request_claim_sha256": claim["smoke_request_claim_sha256"],
            "retry_allowed": False,
        }
        if kind == "arcface":
            execution_payload["kind"] = "arcface"
        else:
            execution_payload.update(
                {
                    "global_exclusive_slots": 16,
                    "ram": {"admission_percent": 85, "hard_limit_percent": 90},
                }
            )
        execution = _write_digest_contract(
            artifact_root / "execution_claim.json",
            execution_payload,
            "execution_claim_sha256",
        )
        if kind == "arcface":
            worker_raw: Any = [
                {
                    "sample_id": f"sample-{index:03d}",
                    "source_face_count": 1,
                    "native_face_count": 1,
                    "candidate_face_count": 1,
                    "source_native_cosine": 0.2,
                    "source_candidate_cosine": 0.1,
                }
                for index in range(64)
            ]
            peak_rss = 1_000_000_000
        else:
            worker_raw = {
                "metrics": ["fid", "kid", "niqe", "sharpness"],
                "num_generated": 64,
                "num_real": 64,
                "sample_id_count": 64,
                "fid": 100.0,
                "kid_mean": 0.02,
                "kid_std": 0.001,
                "iqa": {"mean": 4.0, "std": 0.5},
                "sharpness": {
                    "mean": 500.0,
                    "std": 20.0,
                    "median": 490.0,
                    "p05": 400.0,
                    "p10": 420.0,
                    "p90": 580.0,
                    "p95": 600.0,
                },
            }
            peak_rss = 2_000_000_000
        worker_result = _write_digest_contract(
            artifact_root / "worker_result.json",
            {
                "schema_version": 1,
                "contract_type": "safa_r9_phase_evaluator_output_v1",
                "task": kind,
                "evaluator_request_sha256": request["evaluator_request_sha256"],
                "worker_contract": normalized_worker,
                "arcface_contract_sha256": arcface_sha,
                "quality_script_sha256": quality_binding["sha256"],
                "result": worker_raw,
            },
            "evaluator_output_sha256",
        )
        (artifact_root / "worker.log").write_text("ok\n", encoding="utf-8")
        (artifact_root / "controller.log").write_text("ok\n", encoding="utf-8")
        _write_digest_contract(
            artifact_root / "resource_result.json",
            {
                "schema_version": 1,
                "contract_type": (
                    "safa_r9_evaluator_resource_smoke_result_v1"
                    if kind == "arcface"
                    else "safa_r9_quality_bootstrap_smoke_result_v1"
                ),
                "execution_claim_sha256": execution["execution_claim_sha256"],
                "status": "succeeded",
                "failure_reason": None,
                "returncode": 0,
                "peak_process_tree_rss_bytes": peak_rss,
                "peak_gpu_memory_bytes": 1_000_000,
                "ram_slot_budget_bytes": (peak_rss * 110 + 99) // 100,
                "worker_output_sha256": _sha(artifact_root / "worker_result.json"),
                "worker_evaluator_output_sha256": worker_result[
                    "evaluator_output_sha256"
                ],
                "worker_log_sha256": _sha(artifact_root / "worker.log"),
                "gpu_uuid": "GPU-fixture",
                "retry_allowed": False,
            },
            "resource_smoke_result_sha256",
        )
        raw_profiles[kind] = {
            "mode": (
                "measured_single_worker"
                if kind == "arcface"
                else "measured_exclusive_bootstrap"
            ),
            "artifact_root": str(artifact_root.relative_to(root)),
        }
    raw_profiles["heldout"] = {
        "mode": "exclusive_single_official_run",
        "smoke_execution": "sealed_until_winner_lock",
        "global_exclusive_slots": 16,
        "ram_admission_percent": 85,
        "ram_hard_limit_percent": 90,
    }
    return materialize_evaluator_resource_profiles(
        raw_profiles,
        repo_root=root,
        worker_contract=worker_contract,
        arcface_contract_sha256=arcface_sha,
        quality_script_sha256=quality_binding["sha256"],
        runtime_config_path=runtime_config,
        runtime_config_sha256=_sha(runtime_config),
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    ids = [str(row["sample_id"]) for row in rows]
    return {
        "path": str(path),
        "sha256": _sha(path),
        "sample_count": len(ids),
        "ordered_sample_id_sha256": _id_sha(ids),
    }


def _write_manifest(path: Path, ids: list[str]) -> dict[str, object]:
    return _write_rows(path, [{"sample_id": sample_id} for sample_id in ids])


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_diagnose(root: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    pairs = []
    for pair_index in range(9):
        pair_rows = [row for row in rows if row["pair_index"] == pair_index]
        difficult = next(row for row in pair_rows if row["role"] == "difficult")
        control = next(row for row in pair_rows if row["role"] == "control")
        pairs.append(
            {
                "pair_index": pair_index,
                "difficult_sample_id": difficult["sample_id"],
                "control_sample_id": control["sample_id"],
                "difficult_native_e0_cosine": difficult["native_e0_cosine"],
                "control_native_e0_cosine": control["native_e0_cosine"],
            }
        )
    result = _write_rows(root / "manifests" / "diagnose_18.jsonl", rows)
    result["path"] = "manifests/diagnose_18.jsonl"
    result.update(
        {
            "difficult_count": 9,
            "control_count": 9,
            "matched_pair_sha256": _canonical_sha(
                {"schema_version": 1, "pairs": pairs}
            ),
        }
    )
    return result


def _manifest_fixture(
    root: Path,
) -> tuple[dict[str, object], dict[str, list[str]], dict[str, object]]:
    ids = {
        "calibration_64": [f"cal-{index}" for index in range(64)],
        "validate_512": [f"val-{index}" for index in range(512)],
        "full_2048": [f"full-{index}" for index in range(2048)],
    }
    ids["full_visual_64"] = ids["full_2048"][:64]
    ids["arcface_clean_pool"] = (
        ids["calibration_64"] + ids["validate_512"] + ids["full_2048"]
    )
    contracts = {}
    for name, values in ids.items():
        path = root / "manifests" / f"{name}.jsonl"
        if name == "full_visual_64":
            contracts[name] = _write_rows(
                path,
                [
                    {"sample_id": sample_id, "full_index": index}
                    for index, sample_id in enumerate(values)
                ],
            )
        else:
            contracts[name] = _write_manifest(path, values)
    for entry in contracts.values():
        entry["path"] = str(Path(str(entry["path"])).relative_to(root))
    r8 = _write_manifest(root / "r8" / "calibration_64.jsonl", ids["calibration_64"])
    r8["path"] = str(Path(str(r8["path"])).relative_to(root))
    diagnose_rows = []
    pairs = []
    for pair_index in range(9):
        difficult = ids["calibration_64"][pair_index]
        control = ids["calibration_64"][pair_index + 9]
        difficult_e0 = -0.4 + pair_index * 0.01
        control_e0 = -0.3 + pair_index * 0.01
        diagnose_rows.extend(
            [
                {
                    "sample_id": difficult,
                    "pair_index": pair_index,
                    "role": "difficult",
                    "matched_control_sample_id": control,
                    "native_e0_cosine": difficult_e0,
                },
                {
                    "sample_id": control,
                    "pair_index": pair_index,
                    "role": "control",
                    "matched_difficult_sample_id": difficult,
                    "native_e0_cosine": control_e0,
                },
            ]
        )
        pairs.append(
            {
                "pair_index": pair_index,
                "difficult_sample_id": difficult,
                "control_sample_id": control,
                "difficult_native_e0_cosine": difficult_e0,
                "control_native_e0_cosine": control_e0,
            }
        )
    diagnose = _write_rows(root / "manifests" / "diagnose_18.jsonl", diagnose_rows)
    diagnose["path"] = str(Path(str(diagnose["path"])).relative_to(root))
    diagnose.update(
        {
            "difficult_count": 9,
            "control_count": 9,
            "matched_pair_sha256": _canonical_sha(
                {"schema_version": 1, "pairs": pairs}
            ),
        }
    )
    clean_source = {**contracts["arcface_clean_pool"], "arcface_exact_one": True}
    return (
        contracts,
        ids,
        {
            "clean_source": clean_source,
            "r8_calibration_64": r8,
            "diagnose_18": diagnose,
        },
    )


def _validate_manifests(
    manifests: dict[str, object], evidence: dict[str, object], root: Path
) -> dict[str, object]:
    return validate_manifest_contracts(
        manifests,
        root,
        clean_source=evidence["clean_source"],
        r8_calibration_binding=evidence["r8_calibration_64"],
        diagnose_manifest=evidence["diagnose_18"],
    )


def _runtime(
    root: Path, manifests: dict[str, object], evidence: dict[str, object]
) -> dict[str, object]:
    template = root / "configs" / "campaign.yaml"
    base = root / "configs" / "base.yaml"
    checkpoint = root / "checkpoint.pt"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("schema_version: 1\n", encoding="utf-8")
    base.write_text("experiment_contract: safa_r9_meanflow_v1\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = _sha(checkpoint)
    smoke_peak = 1_000_000_000
    smoke = build_resource_smoke_contract(
        run_id="resource-smoke-native-calibration",
        arm_id="native",
        manifest="calibration_64",
        manifest_sha256=manifests["calibration_64"]["sha256"],
        checkpoint_sha256=checkpoint_sha,
        peak_rss_bytes=smoke_peak,
    )
    smoke_path = root / "campaigns" / "r9-test-campaign" / "resource_smoke.json"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    schedule_contract_sha = SHA_D
    gate_path = root / "contracts" / "semigroup_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_semigroup_gate_v1",
        "experiment_contract": "safa_r9_meanflow_v1",
        "gate_passed": True,
        "determinism_policy_sha256": (
            "ea6a4e81627a993066d9b1a3ca4ae791a0bcb3e21e399a5d2cb27811aa22147f"
        ),
        "attention_backend_requested": "native",
        "attention_backend_resolved": "native",
        "checkpoint_sha256": checkpoint_sha,
        "sample_id_manifest_sha256": manifests["calibration_64"]["sha256"],
        "schedule_contract_sha256": schedule_contract_sha,
    }
    gate_payload["gate_contract_sha256"] = _canonical_sha(gate_payload)
    gate_path.write_text(json.dumps(gate_payload), encoding="utf-8")
    schedule_path = root / "contracts" / "schedule.json"
    schedule_payload = {
        "schema_version": 3,
        "gate_passed": True,
        "checkpoint_sha256": checkpoint_sha,
        "semigroup_sample_id_manifest_sha256": manifests["calibration_64"]["sha256"],
        "schedule_contract_sha256": schedule_contract_sha,
        "r9_semigroup_gate_contract": str(gate_path.relative_to(root)),
        "r9_semigroup_gate_contract_sha256": _sha(gate_path),
    }
    schedule_path.write_text(json.dumps(schedule_payload), encoding="utf-8")
    worker = root / "scripts" / "run_r9_phase_evaluator.py"
    worker_implementation = (
        root / "src" / "safa" / "evaluation" / "r9_evaluator_worker.py"
    )
    quality_script = root / "scripts" / "eval_generation_quality.py"
    real_index = root / "data" / "index" / "val.jsonl"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker_implementation.parent.mkdir(parents=True, exist_ok=True)
    real_index.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text("raise SystemExit(0)\n", encoding="utf-8")
    worker_implementation.write_text(
        "# fixture worker implementation\n", encoding="utf-8"
    )
    quality_script.write_text(
        "def evaluate_generation_quality(): pass\n", encoding="utf-8"
    )
    real_index.write_text('{"sample_id":"x","image_path":"x.png"}\n', encoding="utf-8")
    model_root = root / "insightface"
    model_dir = model_root / "models" / "buffalo_l"
    model_dir.mkdir(parents=True, exist_ok=True)
    arcface_assets = {}
    for name in (
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    ):
        asset = model_dir / name
        asset.write_bytes(name.encode("utf-8"))
        arcface_assets[name] = _sha(asset)
    execution_probe = _arcface_execution_probe(root, arcface_assets)
    worker_contract = {
        "path": str(worker.relative_to(root)),
        "sha256": _sha(worker),
        "implementation_path": str(worker_implementation.relative_to(root)),
        "implementation_sha256": _sha(worker_implementation),
    }
    probe_payload = json.loads(
        (root / execution_probe["path"]).read_text(encoding="utf-8")
    )
    arcface_contract = {
        "model_name": "buffalo_l",
        "model_root": str(model_root),
        "det_size": [224, 224],
        "provider": "CUDAExecutionProvider",
        "insightface_version": "0.7.3",
        "onnxruntime_version": "1.26.0",
        "assets": arcface_assets,
        "execution": probe_payload["execution"],
        "execution_probe": execution_probe,
    }
    resource_smokes = _evaluator_resource_fixture(
        root,
        worker_contract=worker_contract,
        arcface_contract=arcface_contract,
        quality_script=quality_script,
        runtime_config=template,
    )
    diagnose_arms = [{"arm_id": "native", "family": "native"}]
    diagnose_arms.extend(
        {"arm_id": f"flow-{index}", "family": "flow_map2"} for index in range(3)
    )
    diagnose_arms.extend(
        {"arm_id": f"paper-{index}", "family": "paper_split_constant"}
        for index in range(6)
    )
    diagnose_arms.extend(
        {
            "arm_id": f"ablation-{index}",
            "family": "paper_split_interval_ablation",
        }
        for index in range(3)
    )
    return {
        "schema_version": 1,
        "experiment_contract": "safa_r9_campaign_v1",
        "generation_experiment_contract": "safa_r9_meanflow_v1",
        "campaign_id": "r9-test-campaign",
        "campaign_root": "campaigns/r9-test-campaign",
        "campaign_template": {
            "path": str(template.relative_to(root)),
            "sha256": _sha(template),
        },
        "base_config": {
            "path": str(base.relative_to(root)),
            "sha256": _sha(base),
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(root)),
            "sha256": checkpoint_sha,
        },
        "determinism_policy_sha256": (
            "ea6a4e81627a993066d9b1a3ca4ae791a0bcb3e21e399a5d2cb27811aa22147f"
        ),
        "attention_backend": "native",
        "schedule": {
            "path": str(schedule_path.relative_to(root)),
            "file_sha256": _sha(schedule_path),
            "contract_sha256": schedule_contract_sha,
        },
        "semigroup_gate": {
            "path": str(gate_path.relative_to(root)),
            "file_sha256": _sha(gate_path),
            "contract_sha256": gate_payload["gate_contract_sha256"],
        },
        "seeds": {
            "preflight": [1337],
            "diagnose": [1337],
            "calibrate": [1337, 2027, 3407],
            "confirm512": [4409],
            "full": [5501],
        },
        "manifests": manifests,
        "clean_source": evidence["clean_source"],
        "manifest_construction": {
            "r8_calibration_64": evidence["r8_calibration_64"],
            "diagnose_18": evidence["diagnose_18"],
        },
        "resources": {
            "physical_gpus": [0, 1, 2, 3],
            "global_slot_lock_root": "/tmp/safa-r9-gpu-slots-v1",
            "max_slots_per_gpu": 4,
            "gpu_slot_claim_bytes": 4_938_792_960,
            "gpu_headroom_bytes": 2 * 1024**3,
            "ram_admission_percent": 85,
            "ram_hard_limit_percent": 90,
            "require_tmux": True,
            "retry_count": 0,
            "resource_smoke": {
                "required": True,
                "run_id": "resource-smoke-native-calibration",
                "arm_id": "native",
                "manifest": "calibration_64",
                "output_path": str(smoke_path.relative_to(root)),
                "factor": 1.10,
                "result": {
                    "path": str(smoke_path.relative_to(root)),
                    "file_sha256": _sha(smoke_path),
                    "contract_sha256": smoke["resource_smoke_sha256"],
                },
            },
            "ram_slot_budget_bytes": (smoke_peak * 110 + 99) // 100,
        },
        "bootstrap": {
            "resamples": 10_000,
            "confidence": 0.95,
            "seed": 91637,
            "cluster": "sample_id",
            "identity_delta_direction": "source_candidate_minus_source_native",
        },
        "evaluation": {
            "worker": worker_contract,
            "quality": {
                "script": {
                    "path": str(quality_script.relative_to(root)),
                    "sha256": _sha(quality_script),
                },
                "real_index": {
                    "path": str(real_index.relative_to(root)),
                    "sha256": _sha(real_index),
                },
                "metrics": ["fid", "kid", "niqe", "sharpness"],
                "iqa_method": "niqe",
                "device": "cuda:0",
            },
            "arcface": {
                "model_name": "buffalo_l",
                "model_root": str(model_root),
                "det_size": [224, 224],
                "provider": "CUDAExecutionProvider",
                "insightface_version": "0.7.3",
                "onnxruntime_version": "1.26.0",
                "assets": arcface_assets,
                "execution_probe": execution_probe,
            },
            "heldout": {
                "batch_size": 16,
                "representation_image_size": 224,
                "facenet": {"embedding_dim": 512, "input_size": 160},
                "adaface": {"embedding_dim": 512, "input_size": 112},
            },
            "resource_smokes": resource_smokes,
        },
        "phases": {
            "preflight": {
                "manifest": "calibration_64",
                "sample_count": 64,
                "shards_per_logical_run": 4,
                "seed": 1337,
            },
            "diagnose": {
                "manifest": "diagnose_18",
                "sample_count": 18,
                "shards_per_logical_run": 1,
                "seed": 1337,
                "repeats": 3,
                "determinism_repeats_must_match": 3,
                "arms": diagnose_arms,
                "gate": {
                    "metrics_role": "report_only",
                    "difficult_severe_reference": 3,
                    "control_severe_reference": 1,
                    "e0_mean_reference": 0.75,
                    "edev_vs_matched_native_reference": 0.0,
                    "require_finite_diagnostics": True,
                    "require_bitwise_repeat_identity": True,
                    "max_candidates_per_family": 1,
                    "family_order": [
                        "flow_map2",
                        "paper_split_constant",
                        "paper_split_interval_ablation",
                    ],
                    "rank_order": ["severe", "edev_desc", "e0_desc", "arm_id"],
                },
            },
            "calibrate": {
                "manifest": "calibration_64",
                "sample_count": 64,
                "shards_per_logical_run": 1,
                "seeds": [1337, 2027, 3407],
                "candidate_slots": 3,
                "collect_interval_diagnostics": False,
                "gate": {
                    "severe_max_per_seed": 3,
                    "repeated_severe_same_id_max": 1,
                    "fid_native_margin_max": 3.0,
                    "kid_native_margin_max": 0.005,
                    "niqe_native_margin_max": 0.10,
                    "sharpness_absolute_min": 300.0,
                    "sharpness_native_ratio_min": 0.95,
                    "e0_min": 0.75,
                    "e0_delta_min": 0.30,
                    "edev_delta_min": 0.05,
                    "face_count_exact": 1,
                    "identity_delta_upper_95_max": 0.02,
                    "max_advancing_candidates": 2,
                    "rank_order": [
                        "severe",
                        "kid",
                        "fid",
                        "edev_desc",
                        "e0_desc",
                        "arm_id",
                    ],
                },
            },
            "confirm512": {
                "manifest": "validate_512",
                "sample_count": 512,
                "shards_per_logical_run": 8,
                "seed": 4409,
                "candidate_slots": 2,
                "visual_severe_max": 25,
                "gate_ref": "calibrate.gate",
                "rank_order": [
                    "severe",
                    "kid",
                    "fid",
                    "edev_desc",
                    "e0_desc",
                    "arm_id",
                ],
                "winner_count": 1,
            },
            "full": {
                "manifest": "full_2048",
                "visual_manifest": "full_visual_64",
                "sample_count": 2048,
                "shards_per_logical_run": 16,
                "seed": 5501,
                "visual_severe_max": 3,
                "gate_ref": "calibrate.gate",
                "representation": {
                    "e1_mean_delta_lower_95_min_exclusive": 0.0,
                    "e2_mean_delta_lower_95_min_exclusive": 0.0,
                },
                "privacy": {
                    "recognizers": ["arcface", "facenet", "adaface"],
                    "identity_delta_upper_95_max": 0.02,
                    "require_complete_pairs": 2048,
                },
                "sealed_until_selection_lock": [
                    "heldout_e1",
                    "heldout_e2",
                    "facenet",
                    "adaface",
                ],
                "report_only_metrics": ["tar_at_far", "eer", "auc"],
                "reselect_after_failure": False,
            },
        },
    }


def _convert_runtime_to_recovery(
    root: Path, runtime: dict[str, object]
) -> dict[str, object]:
    gate_binding = runtime["semigroup_gate"]
    schedule_binding = runtime["schedule"]
    assert isinstance(gate_binding, dict)
    assert isinstance(schedule_binding, dict)
    gate_path = root / str(gate_binding["path"])
    schedule_path = root / str(schedule_binding["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    gate.update(
        {
            "schema_version": 2,
            "contract_type": "safa_r9_semigroup_recovery_gate_v2",
            "recovery_policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
            "numerical_metrics_role": "report_only",
            "selection_rule": R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
            "selected_t_cut": 0.25,
        }
    )
    schedule.update(
        {
            "recovery_policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
            "numerical_metrics_role": "report_only",
            "selection_rule": R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
            "t_cut": 0.25,
        }
    )
    gate["gate_contract_sha256"] = _canonical_sha(
        {key: value for key, value in gate.items() if key != "gate_contract_sha256"}
    )
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    gate_binding["file_sha256"] = _sha(gate_path)
    gate_binding["contract_sha256"] = gate["gate_contract_sha256"]
    schedule["r9_semigroup_gate_contract_sha256"] = gate_binding["file_sha256"]
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    schedule_binding["file_sha256"] = _sha(schedule_path)
    return runtime


def _resolved_recovery(runtime: dict[str, object]) -> dict[str, object]:
    return {
        "formal_campaign_id": runtime["campaign_id"],
        "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
        "schedule": deepcopy(runtime["schedule"]),
        "gate": deepcopy(runtime["semigroup_gate"]),
    }


def _context(*, manifest_sha: str = SHA_D) -> dict[str, str]:
    return {
        "campaign_id": "r9-test-campaign",
        "campaign_runtime_sha256": SHA_A,
        "manifest_contracts_sha256": SHA_B,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": SHA_C,
        "phase_results_sha256": "e" * 64,
        "automatic_evidence_sha256": "f" * 64,
        "run_plan_sha256": "1" * 64,
        "evaluator_evidence_sha256": "2" * 64,
    }


def _diagnose_manifest() -> dict[str, object]:
    return {
        "path": "manifests/diagnose_18.jsonl",
        "sha256": SHA_D,
        "sample_count": 18,
        "ordered_sample_id_sha256": SHA_A,
        "difficult_count": 9,
        "control_count": 9,
        "matched_pair_sha256": SHA_B,
    }


def _a_arm(
    arm_id: str,
    family: str,
    *,
    difficult: int = 0,
    control: int = 0,
    e0: float = 0.80,
    edev_delta: float = 0.06,
    digest: str = SHA_A,
    diagnostics_finite: bool = True,
) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "family": family,
        "config_sha256": SHA_B,
        "output_sha256": SHA_C,
        "repeat_results": [
            {
                "repeat_index": index,
                "run_sha256": digest,
                "difficult_severe_count": difficult,
                "control_severe_count": control,
                "e0_mean": e0,
                "edev_delta_vs_matched_native": edev_delta,
                "diagnostics_finite": diagnostics_finite,
            }
            for index in range(3)
        ],
    }


def _seed_result(
    seed: int,
    *,
    severe_ids: tuple[str, ...] = (),
    fid: float = 10.0,
    kid: float = 0.01,
    e0: float = 0.80,
) -> dict[str, object]:
    return {
        "seed": seed,
        "severe_count": len(severe_ids),
        "severe_sample_ids": list(severe_ids),
        "fid": fid,
        "native_fid": 10.0,
        "kid": kid,
        "native_kid": 0.01,
        "niqe": 4.0,
        "native_niqe": 4.0,
        "sharpness": 350.0,
        "native_sharpness": 350.0,
        "e0": e0,
        "delta_e0": e0 - 0.45,
        "delta_edev": 0.58 - 0.5,
        "arcface_exact_one": True,
    }


def _privacy_rows(
    sample_count: int, seeds: tuple[int, ...], *, delta: float = 0.0
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{sample_index}",
            "seed": seed,
            "source_candidate_cosine": 0.5 + delta,
            "source_native_cosine": 0.5,
        }
        for sample_index in range(sample_count)
        for seed in seeds
    ]


def _paired_metric_rows_contract(
    sample_count: int,
    seeds: tuple[int, ...],
    *,
    candidate_e0: float = 0.80,
    native_e0: float = 0.45,
    candidate_edev: float = 0.58,
    native_edev: float = 0.50,
    candidate_niqe: float = 4.0,
    native_niqe: float = 4.0,
    candidate_sharpness: float = 350.0,
    native_sharpness: float = 350.0,
) -> dict[str, object]:
    sample_ids = [f"sample-{index}" for index in range(sample_count)]
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_type": "safa_r9_paired_metric_rows_v1",
        "direction": "candidate_minus_native",
        "seeds": list(seeds),
        "sample_count": sample_count,
        "observation_count": sample_count * len(seeds),
        "ordered_sample_id_sha256": _id_sha(sample_ids),
        "metric_fields": [
            "candidate_e0",
            "native_e0",
            "candidate_edev",
            "native_edev",
            "candidate_niqe",
            "native_niqe",
            "candidate_sharpness",
            "native_sharpness",
        ],
        "rows": [
            {
                "sample_id": sample_id,
                "seed": seed,
                "candidate_e0": candidate_e0,
                "native_e0": native_e0,
                "candidate_edev": candidate_edev,
                "native_edev": native_edev,
                "candidate_niqe": candidate_niqe,
                "native_niqe": native_niqe,
                "candidate_sharpness": candidate_sharpness,
                "native_sharpness": native_sharpness,
            }
            for seed in seeds
            for sample_id in sample_ids
        ],
    }
    payload["paired_metric_rows_sha256"] = _canonical_sha(payload)
    return payload


def _quality_arm(
    arm_id: str,
    seeds: tuple[int, ...],
    sample_count: int,
    *,
    fid: float = 10.0,
    kid: float = 0.01,
    privacy_delta: float = 0.0,
    repeated_severe: bool = False,
    arcface_exact_one: bool = True,
) -> dict[str, object]:
    rows = []
    for index, seed in enumerate(seeds):
        severe = ("repeat-id",) if repeated_severe and index < 2 else ()
        seed_row = _seed_result(seed, severe_ids=severe, fid=fid, kid=kid)
        seed_row["arcface_exact_one"] = arcface_exact_one
        rows.append(seed_row)
    return {
        "arm_id": arm_id,
        "family": "flow_map2",
        "config_sha256": SHA_B,
        "output_sha256": hashlib.sha256(arm_id.encode()).hexdigest(),
        "seed_results": rows,
        "privacy_rows": (
            _privacy_rows(sample_count, seeds, delta=privacy_delta)
            if arcface_exact_one
            else []
        ),
        "paired_metric_rows": _paired_metric_rows_contract(sample_count, seeds),
    }


def _selection_fixture() -> tuple[dict[str, object], dict[str, object]]:
    gate = build_c_gate_contract(
        _context(),
        [_quality_arm("winner", (4409,), 512)],
        confirm_seed=4409,
        bootstrap_seed=17,
    )
    selection = build_selection_contract(
        gate,
        manifest_sha256s={
            name: SHA_A
            for name in (
                "calibration_64",
                "validate_512",
                "full_2048",
                "full_visual_64",
                "arcface_clean_pool",
            )
        },
    )
    return gate, selection


def test_runtime_digest_is_generated_and_seed_contract_is_strict(
    tmp_path: Path,
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)

    validated = validate_campaign_runtime(runtime, tmp_path)

    assert validated["campaign_runtime_sha256"] == canonical_campaign_runtime_sha256(
        validated
    )
    assert (
        validated["manifest_contracts_sha256"]
        == _validate_manifests(manifests, evidence, tmp_path)[
            "manifest_contracts_sha256"
        ]
    )
    hand_filled = {**runtime, "campaign_runtime_sha256": SHA_A}
    with pytest.raises(CampaignContractError, match="hand-filled"):
        validate_campaign_runtime(hand_filled, tmp_path)
    wrong_seed = deepcopy(runtime)
    wrong_seed["seeds"]["calibrate"] = [1337, 2027, 9999]
    with pytest.raises(CampaignContractError, match="calibrate seeds"):
        validate_campaign_runtime(wrong_seed, tmp_path)


def test_runtime_accepts_exact_resolved_recovery_schedule_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _convert_runtime_to_recovery(
        tmp_path, _runtime(tmp_path, manifests, evidence)
    )
    resolved = _resolved_recovery(runtime)
    monkeypatch.setattr(
        campaign_contracts_module,
        "resolve_formal_campaign_semigroup_closure",
        lambda *args, **kwargs: resolved,
    )

    validated = validate_campaign_runtime(runtime, tmp_path)

    assert validated["schedule"] == runtime["schedule"]
    assert validated["semigroup_gate"] == runtime["semigroup_gate"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("policy_sha", "policy semantics"),
        ("metrics_role", "policy semantics"),
        ("selection_rule", "policy semantics"),
        ("t_cut", "policy semantics"),
        ("mixed_schema", "schema/type pairing"),
        ("closure_binding", "exactly match the resolved closure"),
    ),
)
def test_runtime_rejects_recovery_semantics_or_resolved_closure_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _convert_runtime_to_recovery(
        tmp_path, _runtime(tmp_path, manifests, evidence)
    )
    gate_binding = runtime["semigroup_gate"]
    schedule_binding = runtime["schedule"]
    assert isinstance(gate_binding, dict)
    assert isinstance(schedule_binding, dict)
    gate_path = tmp_path / str(gate_binding["path"])
    schedule_path = tmp_path / str(schedule_binding["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if mutation == "policy_sha":
        gate["recovery_policy_sha256"] = SHA_A
        schedule["recovery_policy_sha256"] = SHA_A
    elif mutation == "metrics_role":
        gate["numerical_metrics_role"] = "hard_gate"
        schedule["numerical_metrics_role"] = "hard_gate"
    elif mutation == "selection_rule":
        gate["selection_rule"] = "smallest_numeric_t_cut"
        schedule["selection_rule"] = "smallest_numeric_t_cut"
    elif mutation == "t_cut":
        gate["selected_t_cut"] = 0.5
        schedule["t_cut"] = 0.5
    elif mutation == "mixed_schema":
        gate["schema_version"] = 1
    elif mutation != "closure_binding":
        raise AssertionError(f"unregistered mutation: {mutation}")
    if mutation != "closure_binding":
        gate["gate_contract_sha256"] = _canonical_sha(
            {key: value for key, value in gate.items() if key != "gate_contract_sha256"}
        )
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        gate_binding["file_sha256"] = _sha(gate_path)
        gate_binding["contract_sha256"] = gate["gate_contract_sha256"]
        schedule["r9_semigroup_gate_contract_sha256"] = gate_binding["file_sha256"]
        schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
        schedule_binding["file_sha256"] = _sha(schedule_path)
    resolved = _resolved_recovery(runtime)
    if mutation == "closure_binding":
        resolved_gate = resolved["gate"]
        assert isinstance(resolved_gate, dict)
        resolved_gate["file_sha256"] = SHA_A
    monkeypatch.setattr(
        campaign_contracts_module,
        "resolve_formal_campaign_semigroup_closure",
        lambda *args, **kwargs: resolved,
    )

    with pytest.raises(CampaignContractError, match=message):
        validate_campaign_runtime(runtime, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("checkpoint_missing", "checkpoint must be an existing regular file"),
        ("checkpoint_hash", "checkpoint file SHA256 mismatch"),
        ("extra_field", "campaign runtime fields mismatch"),
        ("generation_contract", "generation_experiment_contract"),
        ("determinism", "determinism policy"),
        ("attention", "attention backend must be native"),
    ),
)
def test_runtime_rejects_unbound_or_noncanonical_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    if mutation == "checkpoint_missing":
        (tmp_path / str(runtime["checkpoint"]["path"])).unlink()
    elif mutation == "checkpoint_hash":
        runtime["checkpoint"]["sha256"] = SHA_A
    elif mutation == "extra_field":
        runtime["manual_override"] = True
    elif mutation == "generation_contract":
        runtime["generation_experiment_contract"] = "safa_r8_meanflow_v1"
    elif mutation == "determinism":
        runtime["determinism_policy_sha256"] = SHA_A
    else:
        runtime["attention_backend"] = "auto"

    with pytest.raises(CampaignContractError, match=message):
        validate_campaign_runtime(runtime, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("gate_canonical", "semigroup gate canonical digest mismatch"),
        ("gate_schedule_binding", "does not bind locked schedule"),
        ("gate_determinism", "semigroup gate determinism policy mismatch"),
    ),
)
def test_runtime_rejects_semigroup_gate_tamper(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    gate_path = tmp_path / str(runtime["semigroup_gate"]["path"])
    schedule_path = tmp_path / str(runtime["schedule"]["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if mutation == "gate_canonical":
        gate["unexpected"] = True
    elif mutation == "gate_schedule_binding":
        gate["schedule_contract_sha256"] = SHA_A
        gate["gate_contract_sha256"] = _canonical_sha(
            {key: value for key, value in gate.items() if key != "gate_contract_sha256"}
        )
        runtime["semigroup_gate"]["contract_sha256"] = gate["gate_contract_sha256"]
    else:
        gate["determinism_policy_sha256"] = SHA_A
        gate["gate_contract_sha256"] = _canonical_sha(
            {key: value for key, value in gate.items() if key != "gate_contract_sha256"}
        )
        runtime["semigroup_gate"]["contract_sha256"] = gate["gate_contract_sha256"]
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    runtime["semigroup_gate"]["file_sha256"] = _sha(gate_path)
    schedule["r9_semigroup_gate_contract_sha256"] = _sha(gate_path)
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    runtime["schedule"]["file_sha256"] = _sha(schedule_path)

    with pytest.raises(CampaignContractError, match=message):
        validate_campaign_runtime(runtime, tmp_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("resources", "max_slots_per_gpu", 5, "runtime resource"),
        ("resources", "retry_count", 1, "runtime resource"),
        ("bootstrap", "resamples", 9_999, "runtime bootstrap"),
        (
            "bootstrap",
            "identity_delta_direction",
            "native-candidate",
            "runtime bootstrap",
        ),
        ("phases", "preflight", {}, "preflight phase exact contract mismatch"),
    ),
)
def test_runtime_rejects_resource_bootstrap_or_phase_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    runtime[section][field] = value

    with pytest.raises(CampaignContractError, match=message):
        validate_campaign_runtime(runtime, tmp_path)


def test_resource_smoke_is_measured_bound_and_exactly_derives_ram_budget(
    tmp_path: Path,
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)

    validated = validate_campaign_runtime(runtime, tmp_path)

    result = validated["resources"]["resource_smoke"]["result"]
    assert result["peak_rss_bytes"] == 1_000_000_000
    assert validated["resources"]["ram_slot_budget_bytes"] == 1_100_000_000
    result_path = tmp_path / result["path"]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert validate_resource_smoke_contract(payload) == payload

    wrong_budget = deepcopy(runtime)
    wrong_budget["resources"]["ram_slot_budget_bytes"] += 1
    with pytest.raises(CampaignContractError, match="ceil"):
        validate_campaign_runtime(wrong_budget, tmp_path)

    tampered = deepcopy(payload)
    tampered["peak_rss_bytes"] += 1
    with pytest.raises(CampaignContractError, match="canonical digest mismatch"):
        validate_resource_smoke_contract(tampered)


def test_evaluator_resource_profiles_lock_kind_budgets_and_heldout_seal(
    tmp_path: Path,
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)

    validated = validate_campaign_runtime(runtime, tmp_path)
    profiles = validated["evaluation"]["resource_smokes"]

    assert profiles["arcface"]["ram_slot_budget_bytes"] == 1_100_000_000
    assert profiles["quality"]["ram_slot_budget_bytes"] == 2_200_000_000
    assert profiles["heldout"] == {
        "mode": "exclusive_single_official_run",
        "smoke_execution": "sealed_until_winner_lock",
        "global_exclusive_slots": 16,
        "ram_admission_percent": 85,
        "ram_hard_limit_percent": 90,
    }


def test_evaluator_resource_profile_budget_or_artifact_tamper_fails(
    tmp_path: Path,
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    budget_tamper = deepcopy(runtime)
    budget_tamper["evaluation"]["resource_smokes"]["arcface"][
        "ram_slot_budget_bytes"
    ] += 1
    with pytest.raises(CampaignContractError, match="canonical digest mismatch"):
        validate_campaign_runtime(budget_tamper, tmp_path)

    result_path = (
        tmp_path
        / runtime["evaluation"]["resource_smokes"]["quality"]["resource_result"]["path"]
    )
    result_path.write_bytes(result_path.read_bytes() + b"\n")
    with pytest.raises(CampaignContractError, match="immutable smoke artifacts"):
        validate_campaign_runtime(runtime, tmp_path)


def test_evaluator_resource_claim_runtime_path_drift_fails(tmp_path: Path) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    expected_path = tmp_path / runtime["campaign_template"]["path"]
    drifted_path = tmp_path / "configs" / "campaign-copy.yaml"
    drifted_path.write_bytes(expected_path.read_bytes())
    claim_path = (
        tmp_path
        / runtime["evaluation"]["resource_smokes"]["arcface"]["request_claim"][
            "path"
        ]
    )
    _rewrite_smoke_request_claim(
        claim_path,
        runtime_config=str(drifted_path.resolve()),
        runtime_config_sha256=_sha(drifted_path),
    )

    with pytest.raises(CampaignContractError, match="runtime config binding mismatch"):
        validate_campaign_runtime(runtime, tmp_path)


def test_evaluator_resource_claim_runtime_sha_drift_fails(tmp_path: Path) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    claim_path = (
        tmp_path
        / runtime["evaluation"]["resource_smokes"]["quality"]["request_claim"][
            "path"
        ]
    )
    _rewrite_smoke_request_claim(claim_path, runtime_config_sha256=SHA_A)

    with pytest.raises(CampaignContractError, match="runtime config SHA256 mismatch"):
        validate_campaign_runtime(runtime, tmp_path)


def test_old_arcface_runtime_claim_is_rejected(tmp_path: Path) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    claim_path = (
        tmp_path
        / runtime["evaluation"]["resource_smokes"]["arcface"]["request_claim"][
            "path"
        ]
    )
    _rewrite_smoke_request_claim(
        claim_path,
        runtime_config_sha256=(
            "4f964c86664b0d84684349febba5337b1ecd24cc887d97b42f32262b3fd636a3"
        ),
    )

    with pytest.raises(CampaignContractError, match="runtime config SHA256 mismatch"):
        validate_campaign_runtime(runtime, tmp_path)


@pytest.mark.parametrize("binding", ("run_id", "arm_id", "manifest"))
def test_resource_smoke_declaration_must_match_measured_result(
    tmp_path: Path, binding: str
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)
    runtime = _runtime(tmp_path, manifests, evidence)
    if binding == "manifest":
        runtime["resources"]["resource_smoke"][binding] = "validate_512"
    else:
        runtime["resources"]["resource_smoke"][binding] = "different"

    with pytest.raises(CampaignContractError, match=f"{binding} binding mismatch"):
        validate_campaign_runtime(runtime, tmp_path)


def test_manifests_lock_hash_count_order_disjoint_subset_and_clean_pool(
    tmp_path: Path,
) -> None:
    manifests, _, evidence = _manifest_fixture(tmp_path)

    contract = _validate_manifests(manifests, evidence, tmp_path)

    assert contract["relationships"] == {
        "calibration_validate_full_disjoint": True,
        "full_visual_strict_subset_of_full": True,
        "all_ids_arcface_clean": True,
    }
    assert len(contract["manifest_contracts_sha256"]) == 64


@pytest.mark.parametrize("mutation", ("hash", "overlap", "not_clean", "not_subset"))
def test_manifest_tamper_or_membership_violation_fails(
    tmp_path: Path, mutation: str
) -> None:
    manifests, ids, evidence = _manifest_fixture(tmp_path)
    if mutation == "hash":
        manifests["calibration_64"]["sha256"] = SHA_A
    elif mutation == "overlap":
        ids["validate_512"][0] = ids["calibration_64"][0]
        manifests["validate_512"] = _write_manifest(
            tmp_path / "manifests" / "validate_512.jsonl", ids["validate_512"]
        )
        manifests["validate_512"]["path"] = "manifests/validate_512.jsonl"
    elif mutation == "not_clean":
        pool = ids["arcface_clean_pool"][:-1]
        manifests["arcface_clean_pool"] = _write_manifest(
            tmp_path / "manifests" / "arcface_clean_pool.jsonl", pool
        )
        manifests["arcface_clean_pool"]["path"] = "manifests/arcface_clean_pool.jsonl"
    else:
        visual = ids["full_visual_64"][:-1] + [ids["validate_512"][0]]
        manifests["full_visual_64"] = _write_manifest(
            tmp_path / "manifests" / "full_visual_64.jsonl", visual
        )
        manifests["full_visual_64"]["path"] = "manifests/full_visual_64.jsonl"

    with pytest.raises(CampaignContractError):
        _validate_manifests(manifests, evidence, tmp_path)


def test_calibration_must_exactly_match_bound_r8_fixed64(
    tmp_path: Path,
) -> None:
    manifests, ids, evidence = _manifest_fixture(tmp_path)
    r8 = _write_manifest(
        tmp_path / "r8" / "calibration_64.jsonl",
        list(reversed(ids["calibration_64"])),
    )
    r8["path"] = "r8/calibration_64.jsonl"
    evidence["r8_calibration_64"] = r8

    with pytest.raises(CampaignContractError, match="fixed R8 64-ID binding"):
        _validate_manifests(manifests, evidence, tmp_path)


def test_calibration_may_rebind_r8_ids_with_r9_row_schema(
    tmp_path: Path,
) -> None:
    manifests, ids, evidence = _manifest_fixture(tmp_path)
    rebound = _write_rows(
        tmp_path / "manifests" / "calibration_64.jsonl",
        [
            {"sample_id": sample_id, "r9_contract": "safa_r9_meanflow_v1"}
            for sample_id in ids["calibration_64"]
        ],
    )
    rebound["path"] = "manifests/calibration_64.jsonl"
    manifests["calibration_64"] = rebound

    contract = _validate_manifests(manifests, evidence, tmp_path)

    assert (
        contract["manifests"]["calibration_64"]["sha256"]
        != evidence["r8_calibration_64"]["sha256"]
    )


def test_diagnose_pairs_must_be_symmetric_digest_bound_and_calibration_clean(
    tmp_path: Path,
) -> None:
    manifests, ids, evidence = _manifest_fixture(tmp_path)
    diagnose_path = tmp_path / str(evidence["diagnose_18"]["path"])
    rows = _read_rows(diagnose_path)

    asymmetric_rows = deepcopy(rows)
    asymmetric_rows[0]["matched_control_sample_id"] = "wrong"
    asymmetric = _rewrite_diagnose(tmp_path, asymmetric_rows)
    with pytest.raises(CampaignContractError, match="not symmetric"):
        _validate_manifests(
            manifests,
            {**evidence, "diagnose_18": asymmetric},
            tmp_path,
        )

    restored = _rewrite_diagnose(tmp_path, rows)
    digest_tamper = {**restored, "matched_pair_sha256": SHA_A}
    with pytest.raises(CampaignContractError, match="matched-pair digest mismatch"):
        _validate_manifests(
            manifests,
            {**evidence, "diagnose_18": digest_tamper},
            tmp_path,
        )

    outside_rows = deepcopy(rows)
    outside_rows[0]["sample_id"] = ids["validate_512"][0]
    outside_rows[0]["matched_control_sample_id"] = ids["validate_512"][1]
    outside_rows[1]["sample_id"] = ids["validate_512"][1]
    outside_rows[1]["matched_difficult_sample_id"] = ids["validate_512"][0]
    outside = _rewrite_diagnose(tmp_path, outside_rows)
    with pytest.raises(CampaignContractError, match="belong to calibration_64"):
        _validate_manifests(
            manifests,
            {**evidence, "diagnose_18": outside},
            tmp_path,
        )


def test_full_visual_rows_must_bind_their_full_2048_indices(
    tmp_path: Path,
) -> None:
    manifests, ids, evidence = _manifest_fixture(tmp_path)
    rows = [
        {"sample_id": sample_id, "full_index": index}
        for index, sample_id in enumerate(ids["full_visual_64"])
    ]
    rows[0]["full_index"] = 1
    visual = _write_rows(tmp_path / "manifests" / "full_visual_64.jsonl", rows)
    visual["path"] = "manifests/full_visual_64.jsonl"
    manifests["full_visual_64"] = visual

    with pytest.raises(CampaignContractError, match="full_index"):
        _validate_manifests(manifests, evidence, tmp_path)


@pytest.mark.parametrize("mutation", ("not_certified", "different_source"))
def test_arcface_clean_pool_requires_bound_exact_one_source_evidence(
    tmp_path: Path, mutation: str
) -> None:
    manifests, ids, evidence = _manifest_fixture(tmp_path)
    if mutation == "not_certified":
        evidence["clean_source"]["arcface_exact_one"] = False
        message = "arcface_exact_one"
    else:
        source = _write_manifest(
            tmp_path / "evidence" / "clean_source.jsonl",
            list(reversed(ids["arcface_clean_pool"])),
        )
        source["path"] = "evidence/clean_source.jsonl"
        evidence["clean_source"] = {**source, "arcface_exact_one": True}
        message = "exactly equal"

    with pytest.raises(CampaignContractError, match=message):
        _validate_manifests(manifests, evidence, tmp_path)


def test_visual_pass_is_derived_from_complete_severe_rows() -> None:
    evidence = {
        "sample_count": 4,
        "samples": [{"sample_id": f"id-{index}"} for index in range(4)],
    }
    review = {
        "samples": [
            {"sample_id": f"id-{index}", "severe": index == 0} for index in range(4)
        ]
    }

    passed = derive_visual_arm_pass(review, evidence, severe_limit=1)
    failed = derive_visual_arm_pass(review, evidence, severe_limit=0)

    assert passed["passed"] is True and passed["severe_count"] == 1
    assert failed["passed"] is False
    with pytest.raises(CampaignContractError, match="hand-filled"):
        derive_visual_arm_pass({**review, "passed": False}, evidence, severe_limit=1)
    with pytest.raises(CampaignContractError, match="cover every"):
        derive_visual_arm_pass(
            {"samples": review["samples"][:-1]}, evidence, severe_limit=1
        )


def test_seed_aggregate_is_sample_id_clustered_and_requires_all_seeds() -> None:
    rows = [
        {"sample_id": sample, "seed": seed, "metric": value}
        for sample, value in (("a", 1.0), ("b", 3.0))
        for seed in (1, 2)
    ]
    aggregate = aggregate_seed_metrics(
        rows, expected_seeds=(1, 2), metric_fields=("metric",)
    )
    assert aggregate["sample_count"] == 2
    assert aggregate["aggregate"]["metric"] == 2.0
    with pytest.raises(CampaignContractError, match="cover every"):
        aggregate_seed_metrics(
            rows[:-1], expected_seeds=(1, 2), metric_fields=("metric",)
        )


def test_privacy_bootstrap_has_exact_10k_cluster_draws_and_fixed_direction() -> None:
    positive = privacy_delta_cluster_bootstrap(
        _privacy_rows(8, (1, 2), delta=0.1),
        expected_seeds=(1, 2),
        bootstrap_seed=9,
    )
    negative = privacy_delta_cluster_bootstrap(
        _privacy_rows(8, (1, 2), delta=-0.1),
        expected_seeds=(1, 2),
        bootstrap_seed=9,
    )

    assert positive["iterations"] == R9_BOOTSTRAP_ITERATIONS == 10_000
    assert positive["cluster_unit"] == "sample_id"
    assert positive["direction"] == "source_candidate_minus_source_native"
    assert positive["mean_delta"] > 0 and negative["mean_delta"] < 0
    with pytest.raises(CampaignContractError, match="exactly 10000"):
        privacy_delta_cluster_bootstrap(
            _privacy_rows(2, (1,), delta=0.0),
            expected_seeds=(1,),
            bootstrap_seed=1,
            iterations=9_999,
        )


def test_paired_metric_bootstrap_aggregates_ids_and_keeps_candidate_direction() -> None:
    raw = _paired_metric_rows_contract(
        8,
        (1, 2),
        candidate_e0=0.8,
        native_e0=0.4,
        candidate_edev=0.7,
        native_edev=0.5,
        candidate_niqe=3.0,
        native_niqe=4.0,
        candidate_sharpness=420.0,
        native_sharpness=360.0,
    )
    summary = paired_metric_cluster_bootstrap(
        raw,
        expected_seeds=(1, 2),
        expected_sample_count=8,
        bootstrap_seed=9,
    )
    repeated = paired_metric_cluster_bootstrap(
        raw,
        expected_seeds=(1, 2),
        expected_sample_count=8,
        bootstrap_seed=9,
    )

    assert summary["iterations"] == R9_BOOTSTRAP_ITERATIONS == 10_000
    assert repeated == summary
    assert summary["bootstrap_rng"] == "numpy_pcg64"
    assert summary["resample_index_policy"] == "shared_across_metrics"
    assert summary["direction"] == "candidate_minus_native"
    assert summary["cluster_unit"] == "sample_id"
    assert summary["metrics"]["e0"]["mean_delta"] == pytest.approx(0.4)
    assert summary["metrics"]["edev"]["mean_delta"] == pytest.approx(0.2)
    assert summary["metrics"]["niqe"]["mean_delta"] == pytest.approx(-1.0)
    assert summary["metrics"]["niqe"]["favorable_direction"] == "lower"
    assert summary["metrics"]["sharpness"]["mean_delta"] == pytest.approx(60.0)
    assert summary["metrics"]["e0"]["lower_95_one_sided"] == pytest.approx(0.4)
    assert summary["seed_summaries"] == [
        {
            "seed": seed,
            "candidate_e0": 0.8,
            "native_e0": 0.4,
            "delta_e0": 0.4,
            "candidate_edev": 0.7,
            "native_edev": 0.5,
            "delta_edev": pytest.approx(0.2),
            "candidate_niqe": 3.0,
            "native_niqe": 4.0,
            "delta_niqe": -1.0,
            "candidate_sharpness": 420.0,
            "native_sharpness": 360.0,
            "delta_sharpness": 60.0,
        }
        for seed in (1, 2)
    ]
    canonical_summary = deepcopy(summary)
    del canonical_summary["paired_metric_bootstrap_sha256"]
    assert summary["paired_metric_bootstrap_sha256"] == _canonical_sha(
        canonical_summary
    )

    tampered = deepcopy(raw)
    tampered["rows"][0]["candidate_e0"] = 0.1
    with pytest.raises(CampaignContractError, match="digest mismatch"):
        paired_metric_cluster_bootstrap(
            tampered,
            expected_seeds=(1, 2),
            expected_sample_count=8,
            bootstrap_seed=9,
        )


def test_a_gate_binds_diagnose_pairs_and_selects_at_most_one_per_family() -> None:
    arms = [
        _a_arm("flow-worse", "flow_map2", difficult=2, e0=0.80),
        _a_arm("flow-best", "flow_map2", difficult=0, e0=0.82),
        _a_arm("paper-fail", "paper_split_constant", difficult=4),
        _a_arm("ablation", "paper_split_interval_ablation", e0=0.78),
    ]

    assert (
        validate_diagnose_manifest_contract(_diagnose_manifest())["sample_count"] == 18
    )
    gate = build_a_gate_contract(
        _context(), arms, diagnose_manifest=_diagnose_manifest()
    )

    assert gate["selected_arm_ids"] == ["flow-best", "paper-fail", "ablation"]
    assert gate["diagnose_manifest"]["difficult_count"] == 9
    assert gate["failures"] == []
    paper = next(row for row in gate["arms"] if row["arm_id"] == "paper-fail")
    assert paper["passed"] is True
    assert paper["observations"]["severe_count_max"] == 4
    assert gate["thresholds"]["numerical_metrics_role"] == "report_only"
    report_only = build_a_gate_contract(
        _context(),
        [_a_arm("only-fail", "flow_map2", difficult=9)],
        diagnose_manifest=_diagnose_manifest(),
    )
    assert report_only["verdict"] == "continue"
    assert report_only["selected_arm_ids"] == ["only-fail"]


def test_a_gate_keeps_repeat_and_finite_contracts_as_hard_failures() -> None:
    nonfinite = _a_arm("nonfinite", "flow_map2", diagnostics_finite=False)
    mismatch = _a_arm("mismatch", "paper_split_constant")
    mismatch["repeat_results"][2]["run_sha256"] = SHA_D

    gate = build_a_gate_contract(
        _context(), [nonfinite, mismatch], diagnose_manifest=_diagnose_manifest()
    )

    assert gate["selected_arm_ids"] == []
    assert gate["verdict"] == "stop_zero_candidates"
    assert gate["failures"] == [
        {
            "arm_id": "nonfinite",
            "reasons": [
                "repeat_0:diagnostics_nonfinite_or_contract_mismatch",
                "repeat_1:diagnostics_nonfinite_or_contract_mismatch",
                "repeat_2:diagnostics_nonfinite_or_contract_mismatch",
            ],
        },
        {
            "arm_id": "mismatch",
            "reasons": ["three_repeats_not_bitwise_identical"],
        },
    ]


def test_b_gate_reports_metric_misses_and_still_limits_selection_to_two() -> None:
    arms = [
        _quality_arm("best", SEEDS, 64, kid=0.009),
        _quality_arm("second", SEEDS, 64, kid=0.011),
        _quality_arm("bad", SEEDS, 64, fid=14.0, repeated_severe=True),
    ]

    gate = build_b_gate_contract(_context(), arms, bootstrap_seed=13)

    assert gate["selected_arm_ids"] == ["best", "second"]
    assert gate["verdict"] == "continue"
    assert gate["arms"][2]["passed"] is True
    assert gate["arms"][2]["failures"] == []
    assert gate["arms"][2]["observations"]["repeated_severe_sample_ids"] == [
        "repeat-id"
    ]
    assert "fid_above_native_plus_3" in gate["arms"][2]["seed_results"][0][
        "observations"
    ]["numerical_reference_misses"]
    report_only = build_b_gate_contract(
        _context(),
        [_quality_arm("bad", SEEDS, 64, fid=14.0, privacy_delta=0.1)],
        bootstrap_seed=13,
    )
    assert report_only["selected_arm_ids"] == ["bad"]
    assert report_only["verdict"] == "continue"
    assert report_only["failures"] == []
    assert report_only["arms"][0]["observations"][
        "privacy_reference_misses"
    ] == ["privacy_delta_upper_gt_0.02"]


@pytest.mark.parametrize("phase", ("calibrate", "confirm512"))
@pytest.mark.parametrize("field", ("e0", "niqe"))
def test_b_and_c_gates_reject_scalar_paired_raw_drift(
    phase: str,
    field: str,
) -> None:
    if phase == "calibrate":
        arm = _quality_arm("drift", SEEDS, 64)
        build = lambda: build_b_gate_contract(  # noqa: E731
            _context(), [arm], bootstrap_seed=13
        )
        expected_seed = SEEDS[0]
    else:
        arm = _quality_arm("drift", (4409,), 512)
        build = lambda: build_c_gate_contract(  # noqa: E731
            _context(), [arm], confirm_seed=4409, bootstrap_seed=17
        )
        expected_seed = 4409
    arm["seed_results"][0][field] = float(arm["seed_results"][0][field]) + 1.0

    with pytest.raises(
        CampaignContractError,
        match=rf"seed {expected_seed} field {field} disagrees",
    ):
        build()


def test_c_gate_locks_one_winner_despite_visual_and_numerical_reference_misses() -> (
    None
):
    poor = _quality_arm("poor", (4409,), 512, fid=40.0, privacy_delta=0.1)
    seed_result = poor["seed_results"][0]
    severe_ids = [f"severe-{index}" for index in range(30)]
    seed_result.update(
        {
            "severe_count": len(severe_ids),
            "severe_sample_ids": severe_ids,
            "kid": 0.5,
            "niqe": 20.0,
            "sharpness": 10.0,
            "e0": 0.1,
            "delta_e0": 0.1 - 0.3,
            "delta_edev": 0.2 - 0.5,
        }
    )
    poor["paired_metric_rows"] = _paired_metric_rows_contract(
        512,
        (4409,),
        candidate_e0=0.1,
        native_e0=0.3,
        candidate_edev=0.2,
        native_edev=0.5,
        candidate_niqe=20.0,
        native_niqe=4.0,
        candidate_sharpness=10.0,
        native_sharpness=350.0,
    )

    gate = build_c_gate_contract(
        _context(), [poor], confirm_seed=4409, bootstrap_seed=17
    )

    assert gate["verdict"] == "winner_locked"
    assert gate["selected_arm_ids"] == ["poor"]
    assert gate["failures"] == []
    observations = gate["arms"][0]["seed_results"][0]["observations"]
    assert observations["visual_reference_misses"] == ["severe_count_gt_25"]
    assert set(observations["numerical_reference_misses"]) == {
        "fid_above_native_plus_3",
        "kid_above_native_plus_0.005",
        "niqe_above_native_plus_0.10",
        "sharpness_below_gate",
        "e0_below_0.75",
        "delta_e0_below_0.30",
        "delta_edev_below_0.05",
    }


def test_quality_gate_keeps_arcface_failure_as_failed_exploratory_arm() -> None:
    no_face = _quality_arm("no-face", SEEDS, 64, arcface_exact_one=False)
    gate = build_b_gate_contract(
        _context(),
        [_quality_arm("passing", SEEDS, 64), no_face],
        bootstrap_seed=13,
    )
    failed = next(row for row in gate["arms"] if row["arm_id"] == "no-face")
    assert gate["selected_arm_ids"] == ["passing"]
    assert failed["passed"] is False
    assert failed["privacy_bootstrap"] is None
    assert all(
        "arcface_not_exactly_one_face_per_image" in reason
        for reason in failed["failures"]
    )

    tampered = deepcopy(no_face)
    tampered["privacy_rows"] = _privacy_rows(64, SEEDS)[:1]
    with pytest.raises(CampaignContractError, match="must be empty"):
        build_b_gate_contract(_context(), [tampered], bootstrap_seed=13)


def test_c_selection_is_winner_only_and_tamper_is_detected() -> None:
    gate, selection = _selection_fixture()

    assert gate["selected_arm_ids"] == ["winner"]
    assert selection["winner"]["arm_id"] == "winner"
    assert "arms" not in selection
    assert validate_selection_contract(selection, gate) == selection


def test_child_gate_and_selection_share_continuation_contract_sha256() -> None:
    context = _context()
    context["continuation_contract_sha256"] = "3" * 64
    gate = build_c_gate_contract(
        context,
        [_quality_arm("winner", (4409,), 512)],
        confirm_seed=4409,
        bootstrap_seed=17,
    )
    selection = build_selection_contract(
        gate,
        manifest_sha256s={
            name: SHA_A for name in campaign_contracts_module.R9_MANIFEST_KEYS
        },
    )

    assert gate["context"]["continuation_contract_sha256"] == "3" * 64
    assert selection["continuation_contract_sha256"] == "3" * 64
    assert validate_selection_contract(selection, gate) == selection

    tampered = deepcopy(selection)
    tampered["winner"]["arm_id"] = "other"
    with pytest.raises(CampaignContractError, match="digest mismatch"):
        validate_selection_contract(tampered, gate)


def test_gate_digest_and_immutable_writer_reject_tamper_or_replacement(
    tmp_path: Path,
) -> None:
    gate = build_a_gate_contract(
        _context(),
        [_a_arm("flow", "flow_map2")],
        diagnose_manifest=_diagnose_manifest(),
    )
    path = tmp_path / "gate_contract.json"
    assert (
        write_immutable_contract(path, gate, digest_field="gate_contract_sha256")
        == path
    )
    write_immutable_contract(path, gate, digest_field="gate_contract_sha256")

    tampered = deepcopy(gate)
    tampered["selected_arm_ids"] = []
    with pytest.raises(CampaignContractError, match="digest mismatch"):
        validate_gate_contract(tampered)
    other = build_a_gate_contract(
        _context(),
        [_a_arm("other", "flow_map2")],
        diagnose_manifest=_diagnose_manifest(),
    )
    with pytest.raises(CampaignContractError, match="other content"):
        write_immutable_contract(path, other, digest_field="gate_contract_sha256")


def _identity_report(*, arcface_coverage: int = 2048) -> dict[str, Any]:
    available_role = {
        "status": "available",
        "tar_at_far": {"0.001": 0.2, "0.0001": 0.1},
        "eer": 0.4,
        "auc": 0.6,
    }
    recognizers = {}
    for name in ("arcface", "facenet", "adaface"):
        coverage = arcface_coverage if name == "arcface" else 2048
        if coverage == 2048:
            recognizers[name] = {
                "status": "available",
                "reason": None,
                "coverage": coverage,
                "roles": {
                    "native": dict(available_role),
                    "winner": dict(available_role),
                },
            }
        else:
            reason = "incomplete_exact_one_coverage"
            recognizers[name] = {
                "status": "unavailable",
                "reason": reason,
                "coverage": coverage,
                "roles": {
                    role: {"status": "unavailable", "reason": reason}
                    for role in ("native", "winner")
                },
            }
    return {"schema_version": 1, "recognizers": recognizers}


@pytest.mark.parametrize(
    "mutation",
    ("empty", "missing_recognizer", "missing_role", "missing_far", "range", "reason"),
)
def test_identity_report_strict_schema_rejects_partial_or_invented_metrics(
    mutation: str,
) -> None:
    report = _identity_report()
    if mutation == "empty":
        report = {}
    elif mutation == "missing_recognizer":
        del report["recognizers"]["arcface"]
    elif mutation == "missing_role":
        del report["recognizers"]["facenet"]["roles"]["winner"]
    elif mutation == "missing_far":
        del report["recognizers"]["adaface"]["roles"]["native"]["tar_at_far"]["0.0001"]
    elif mutation == "range":
        report["recognizers"]["arcface"]["roles"]["winner"]["auc"] = 1.1
    else:
        report = _identity_report(arcface_coverage=2047)
        report["recognizers"]["arcface"]["reason"] = ""
    with pytest.raises(CampaignContractError):
        validate_identity_report(report, expected_count=2048)


def test_heldout_metrics_are_report_only_but_coverage_and_run_once_stay_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fast_paired_metric_bootstrap(
        value: dict[str, object],
        *,
        expected_seeds: tuple[int, ...],
        expected_sample_count: int,
        bootstrap_seed: int,
        iterations: int = R9_BOOTSTRAP_ITERATIONS,
    ) -> dict[str, object]:
        assert (expected_seeds, expected_sample_count) in {
            ((4409,), 512),
            ((5501,), 2048),
        }
        assert bootstrap_seed == 17
        assert iterations == R9_BOOTSTRAP_ITERATIONS
        assert value["sample_count"] == expected_sample_count
        assert value["seeds"] == list(expected_seeds)
        rows = value["rows"]
        seed_summaries = []
        for seed in expected_seeds:
            seed_rows = [row for row in rows if row["seed"] == seed]
            summary: dict[str, object] = {"seed": seed}
            for metric in ("e0", "edev", "niqe", "sharpness"):
                candidate_field = f"candidate_{metric}"
                native_field = f"native_{metric}"
                candidate_mean = statistics.fmean(
                    float(row[candidate_field]) for row in seed_rows
                )
                native_mean = statistics.fmean(
                    float(row[native_field]) for row in seed_rows
                )
                summary[candidate_field] = candidate_mean
                summary[native_field] = native_mean
                summary[f"delta_{metric}"] = candidate_mean - native_mean
            seed_summaries.append(summary)
        return {
            "schema_version": 1,
            "contract_type": "safa_r9_paired_metric_cluster_bootstrap_set_v1",
            "metrics": {
                "e0": {"candidate_mean": 0.80, "mean_delta": 0.35},
                "edev": {"candidate_mean": 0.58, "mean_delta": 0.08},
                "niqe": {"candidate_mean": 4.0, "mean_delta": 0.0},
                "sharpness": {"candidate_mean": 350.0, "mean_delta": 0.0},
            },
            "seed_summaries": seed_summaries,
            "paired_metric_rows_sha256": value["paired_metric_rows_sha256"],
            "paired_metric_bootstrap_sha256": SHA_C,
        }

    monkeypatch.setattr(
        campaign_contracts_module,
        "paired_metric_cluster_bootstrap",
        fast_paired_metric_bootstrap,
    )
    _, selection = _selection_fixture()
    seal = build_heldout_seal_contract(
        selection,
        {
            name: {"path": f"models/{name}.pt", "sha256": SHA_A}
            for name in ("e1", "e2", "facenet", "adaface")
        },
    )
    result = {
        "execution_count": 1,
        "winner_arm_id": "winner",
        "config_sha256": selection["winner"]["config_sha256"],
        "output_sha256": SHA_D,
        "seed": 5501,
        "full_visual_severe_count": 0,
        "representations": {
            name: {
                "winner_mean": 0.7,
                "native_mean": 0.6,
                "paired_bootstrap_lower_95": 0.02,
            }
            for name in ("e1", "e2")
        },
        "recognizers": {
            name: {
                "coverage": 2048,
                "privacy_delta_upper_95": 0.01,
                "bootstrap_sha256": SHA_B,
            }
            for name in ("arcface", "facenet", "adaface")
        },
        "quality": {
            **_seed_result(5501),
            "paired_metric_rows": _paired_metric_rows_contract(2048, (5501,)),
        },
        "identity_report": _identity_report(),
    }

    passed = build_d_gate_contract(
        _context(),
        selection=selection,
        heldout_seal=seal,
        result=result,
        bootstrap_seed=17,
    )
    for field in ("e0", "niqe"):
        drifted = deepcopy(result)
        drifted["quality"][field] = float(drifted["quality"][field]) + 1.0
        with pytest.raises(
            CampaignContractError,
            match=rf"seed 5501 field {field} disagrees",
        ):
            build_d_gate_contract(
                _context(),
                selection=selection,
                heldout_seal=seal,
                result=drifted,
                bootstrap_seed=17,
            )
    report_only_result = deepcopy(result)
    report_only_result["full_visual_severe_count"] = 64
    report_only_result["representations"] = {
        name: {
            "winner_mean": 0.5,
            "native_mean": 0.6,
            "paired_bootstrap_lower_95": -0.02,
        }
        for name in ("e1", "e2")
    }
    for row in report_only_result["recognizers"].values():
        row["privacy_delta_upper_95"] = 0.5
    report_only_result["quality"].update(
        {
            "severe_count": 4,
            "severe_sample_ids": [f"severe-{index}" for index in range(4)],
            "fid": 40.0,
            "kid": 0.5,
            "niqe": 20.0,
            "sharpness": 10.0,
            "e0": 0.1,
            "delta_e0": 0.1 - 0.3,
            "delta_edev": 0.2 - 0.5,
        }
    )
    report_only_result["quality"]["paired_metric_rows"] = (
        _paired_metric_rows_contract(
            2048,
            (5501,),
            candidate_e0=0.1,
            native_e0=0.3,
            candidate_edev=0.2,
            native_edev=0.5,
            candidate_niqe=20.0,
            native_niqe=4.0,
            candidate_sharpness=10.0,
            native_sharpness=350.0,
        )
    )
    report_only = build_d_gate_contract(
        _context(),
        selection=selection,
        heldout_seal=seal,
        result=report_only_result,
        bootstrap_seed=17,
    )

    assert seal["execution_count"] == 0 and seal["sealed"] is True
    assert passed["verdict"] == "passed_locked_winner"
    assert report_only["verdict"] == "passed_locked_winner"
    assert report_only["selected_arm_ids"] == ["winner"]
    assert report_only["reselection_allowed"] is False
    assert report_only["failures"] == []
    arm = report_only["arms"][0]
    assert arm["observations"]["full_visual_reference_misses"] == [
        "full_visual_severe_count_gt_3"
    ]
    assert arm["representations"]["e1"]["observations"]["reference_misses"] == [
        "e1_winner_mean_not_above_native",
        "e1_bootstrap_lower_not_positive",
    ]
    assert arm["recognizers"]["arcface"]["observations"][
        "reference_misses"
    ] == ["arcface_privacy_upper_gt_0.02"]
    assert "fid_above_native_plus_3" in arm["quality"]["observations"][
        "numerical_reference_misses"
    ]
    incomplete_result = deepcopy(result)
    incomplete_result["recognizers"]["arcface"] = {
        "coverage": 2047,
        "privacy_delta_upper_95": None,
        "bootstrap_sha256": None,
    }
    incomplete_result["identity_report"] = _identity_report(arcface_coverage=2047)
    incomplete = build_d_gate_contract(
        _context(),
        selection=selection,
        heldout_seal=seal,
        result=incomplete_result,
        bootstrap_seed=17,
    )
    assert incomplete["verdict"] == "failed_locked_winner"
    assert incomplete["arms"][0]["recognizers"]["arcface"] == {
        "coverage": 2047,
        "privacy_delta_upper_95": None,
        "bootstrap_sha256": None,
        "observations": {
            "privacy_metric_role": "report_only",
            "reference_misses": [],
        },
    }
    assert "arcface_coverage_not_2048" in incomplete["arms"][0]["failures"]

    partial_bootstrap = deepcopy(incomplete_result)
    partial_bootstrap["recognizers"]["arcface"]["privacy_delta_upper_95"] = 0.01
    with pytest.raises(CampaignContractError, match="forbids a partial"):
        build_d_gate_contract(
            _context(),
            selection=selection,
            heldout_seal=seal,
            result=partial_bootstrap,
            bootstrap_seed=17,
        )
    rerun = {**result, "execution_count": 2}
    with pytest.raises(CampaignContractError, match="exactly once"):
        build_d_gate_contract(
            _context(),
            selection=selection,
            heldout_seal=seal,
            result=rerun,
            bootstrap_seed=17,
        )
