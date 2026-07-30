from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from safa.evaluation.r9_phase_results import SampleEvidence
from safa.evaluation.r9_evaluator_worker import _validate_arcface_contract


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_r9_meanflow_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_r9_meanflow_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


def runtime() -> dict:
    return driver.load_runtime_config(
        ROOT / "configs/medium_v2/experiments/r9_meanflow_campaign.yaml"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_digest(value: dict[str, Any], digest_field: str) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return driver._canonical_json_sha256(payload)


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


def _profile_event_digest(events: list[list[str]]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(events), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _arcface_contract(root: Path) -> dict[str, Any]:
    model_root = root / "insightface"
    asset_root = model_root / "models" / "buffalo_l"
    asset_root.mkdir(parents=True)
    assets = {}
    for filename in sorted(_fake_arcface_profile_inputs()):
        path = asset_root / filename
        path.write_bytes(filename.encode())
        assets[filename] = _sha(path)
    provider_options = _fake_provider_options()
    session_projection = _fake_session_options_projection()
    execution = {
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
    }
    contract = {
        "model_name": "buffalo_l",
        "model_root": str(model_root),
        "det_size": [224, 224],
        "provider": "CUDAExecutionProvider",
        "insightface_version": "0.7.3",
        "onnxruntime_version": "1.26.0",
        "assets": assets,
        "execution": execution,
    }
    artifact_root = root / "artifacts" / "arcface-execution-probe"
    artifact_root.mkdir(parents=True)
    probe_path = artifact_root / "probe.json"
    claim_path = artifact_root / "claim.json"
    result_path = artifact_root / "result.json"
    probe = {
        "schema_version": 1,
        "contract_type": "safa_r9_arcface_execution_probe_v1",
        "cuda_visible_devices": "GPU-test",
        "runtime_device_id": 0,
        "execution": execution,
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
    claim["bootstrap_claim_sha256"] = _contract_digest(
        claim, "bootstrap_claim_sha256"
    )
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
    return contract


def test_exact_cli_rejects_algorithm_override() -> None:
    with pytest.raises(SystemExit):
        driver.parse_args(
            ["--phase", "diagnose", "--campaign-id", "r9-test", "--eta", "0.25"]
        )


def test_execute_requires_allow_busy_gpus() -> None:
    with pytest.raises(SystemExit):
        driver.parse_args(
            ["--phase", "diagnose", "--campaign-id", "r9-test", "--execute"]
        )


def test_continuation_cli_rejects_parent_phases_and_source_overrides() -> None:
    for phase in ("preflight", "diagnose"):
        with pytest.raises(SystemExit):
            driver.parse_args(
                [
                    "--phase",
                    phase,
                    "--campaign-id",
                    driver.CONTINUATION_CHILD_CAMPAIGN_ID,
                ]
            )
    with pytest.raises(SystemExit):
        driver.parse_args(
            [
                "--phase",
                "calibrate",
                "--campaign-id",
                driver.CONTINUATION_CHILD_CAMPAIGN_ID,
                "--parent-campaign-id",
                "other",
            ]
        )


def test_confirm_continuation_rejects_all_upstream_phases() -> None:
    for phase in ("preflight", "diagnose", "calibrate"):
        with pytest.raises(SystemExit):
            driver.parse_args(
                [
                    "--phase",
                    phase,
                    "--campaign-id",
                    driver.CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID,
                ]
            )


def test_confirm_continuation_all_is_exactly_c_and_d() -> None:
    selected = ["paper_eta_0p125", "flow_map2_normalized_eta_0p125"]
    plans = driver.build_requested_plans(
        runtime(),
        phase="all",
        campaign_id=driver.CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID,
        continuation_selected_arm_ids=selected,
        continuation_start_phase="confirm512",
    )
    assert [plan.phase for plan in plans] == ["confirm512", "full"]
    confirm = plans[0]
    assert confirm.logical_run_count == 3
    assert confirm.shard_count == 48
    assert confirm.logical_run_count * confirm.runs[0].sample_count == 1536
    assert [
        run.arm_ref for run in confirm.runs if run.shard_index == 0
    ] == ["native", *selected]


def test_full_continuation_cli_accepts_only_full() -> None:
    for phase in ("preflight", "diagnose", "calibrate", "confirm512", "all"):
        with pytest.raises(SystemExit):
            driver.parse_args(
                [
                    "--phase",
                    phase,
                    "--campaign-id",
                    driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
                ]
            )
    args = driver.parse_args(
        [
            "--phase",
            "full",
            "--campaign-id",
            driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        ]
    )
    assert args.phase == "full"


def test_full_continuation_plan_is_only_locked_winner_at_batch2() -> None:
    selected = ["paper_eta_0p125"]
    plans = driver.build_requested_plans(
        runtime(),
        phase="full",
        campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        continuation_selected_arm_ids=selected,
        continuation_start_phase="full",
    )
    assert [plan.phase for plan in plans] == ["full"]
    assert plans[0].logical_run_count == 2
    assert [run.arm_ref for run in plans[0].runs if run.shard_index == 0] == [
        "native",
        "paper_eta_0p125",
    ]
    with pytest.raises(ValueError, match="rejects every upstream"):
        driver.build_requested_plans(
            runtime(),
            phase="confirm512",
            campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
            continuation_selected_arm_ids=selected,
            continuation_start_phase="full",
        )


def test_confirm_request_locks_generation_batch_benchmark() -> None:
    payload, _, _ = driver.load_confirm_continuation_request()
    assert payload["generation_batch_benchmark"] == {
        "contract_path": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v8/generation_batch_benchmark.json"
        ),
        "output_root": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v8/generation_batch_benchmark"
        ),
        "manifest": "calibration_64",
        "sample_count": 8,
        "seed": 4549,
        "required_arms": [
            "native",
            "paper_eta_0p125",
            "flow_map2_normalized_eta_0p125",
        ],
        "batch_sizes": [2, 4],
        "record_final_latent_sha256": True,
    }


def test_failed_v7_request_is_preserved_unchanged() -> None:
    path = ROOT / (
        "configs/medium_v2/experiments/"
        "r9_meanflow_confirm_continuation_campaign.yaml"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "70ca0bdb38a7b180fd9be57e9253f44ab5374d39edfb33c8213befa2929fee21"
    )
    assert driver.CONFIRM_CONTINUATION_RUNTIME_CONFIG.name == (
        "r9_meanflow_confirm_continuation_campaign_v8.yaml"
    )


def test_confirm_execute_barrier_requires_materialized_batch_benchmark() -> None:
    with pytest.raises(RuntimeError, match="generation_batch_benchmark.json"):
        driver._require_generation_batch_benchmark_before_confirm(
            is_confirm_continuation=True,
            campaign_runtime={"campaign_runtime_sha256": "a" * 64},
        )
    driver._require_generation_batch_benchmark_before_confirm(
        is_confirm_continuation=True,
        campaign_runtime={"generation_batch_benchmark": {"contract_sha256": "a" * 64}},
    )


def test_continuation_request_is_source_only_and_all_starts_at_b() -> None:
    payload, request_path, source = driver.load_continuation_request()
    request = driver.yaml.safe_load((driver.REPO_ROOT / request_path).read_text())
    assert set(request) == {
        "schema_version",
        "contract_type",
        "child_campaign_id",
        "base_runtime",
        "source",
        "evaluator_resources",
    }
    assert "continuation_contract_sha256" not in request
    assert set(request["source"]) == {
        "parent_campaign_id",
        "diagnose_gate_contract_sha256",
        "diagnose_phase_results_sha256",
    }
    for kind in ("arcface", "quality"):
        assert request["evaluator_resources"][kind]["artifact_root"].startswith(
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            f"{driver.CONTINUATION_CHILD_CAMPAIGN_ID}/evaluator_smoke/"
        )
    selected = [
        "flow_map2_normalized_eta_0p125",
        "paper_eta_0p125",
        "paper_eta_0p25_disable_i2",
    ]
    plans = driver.build_requested_plans(
        payload,
        phase="all",
        campaign_id=driver.CONTINUATION_CHILD_CAMPAIGN_ID,
        continuation_selected_arm_ids=selected,
    )
    assert [plan.phase for plan in plans] == ["calibrate", "confirm512", "full"]
    assert [run.arm_ref for run in plans[0].runs if run.seed == 1337] == [
        "native",
        *selected,
    ]
    assert source == request["source"]


def test_continuation_calibrate_imports_only_parent_selected_arms(monkeypatch) -> None:
    selected = [
        {"arm_id": "flow_map2_normalized_eta_0p125"},
        {"arm_id": "paper_eta_0p125"},
        {"arm_id": "paper_eta_0p25_disable_i2"},
    ]
    monkeypatch.setattr(
        driver,
        "_continuation_for_runtime",
        lambda *args, **kwargs: {"selected_arms": selected},
    )
    promoted, winner = driver.resolve_phase_promotion(
        {},
        {
            "campaign_root": "artifacts/child",
            "continuation": {"contract_sha256": "a" * 64},
        },
        phase="calibrate",
        campaign_id=driver.CONTINUATION_CHILD_CAMPAIGN_ID,
    )
    assert promoted == [row["arm_id"] for row in selected]
    assert winner is None


def test_continuation_rejects_candidate_config_drift(monkeypatch) -> None:
    payload = runtime()
    arm_id = "flow_map2_normalized_eta_0p125"
    monkeypatch.setattr(driver, "_formal_closure_for_runtime", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_bind_locked_schedule", lambda *a, **k: None)
    monkeypatch.setattr(
        driver,
        "_continuation_for_runtime",
        lambda *a, **k: {
            "continuation_contract_sha256": "1" * 64,
            "selected_arms": [
                {"arm_id": arm_id, "config_sha256": "2" * 64}
            ],
        },
    )
    monkeypatch.setattr(
        driver,
        "resolve_frozen_effective_guidance_config",
        lambda config: {**config, "arm_config_sha256": "3" * 64},
    )
    monkeypatch.setattr(
        driver,
        "canonical_r9_algorithm_config_digest",
        lambda *args, **kwargs: "3" * 64,
    )
    run = driver.RunSpec(
        phase="calibrate",
        logical_run_id=f"{arm_id}__seed_1337",
        arm_ref=arm_id,
        seed=1337,
        repeat_index=None,
        shard_index=0,
        num_shards=1,
        sample_count=64,
        manifest_key="calibration_64",
        runtime_config=Path("runtime.yaml"),
        output_dir=Path("output"),
        command=(),
    )
    calibration = payload["manifests"]["calibration_64"]
    manifest_contract = {
        "manifest_contracts_sha256": "4" * 64,
        "manifests": {"calibration_64": calibration},
    }
    campaign_runtime = {
        "campaign_id": driver.CONTINUATION_CHILD_CAMPAIGN_ID,
        "campaign_runtime_sha256": "5" * 64,
        "checkpoint": payload["checkpoint"],
        "continuation": {"contract_sha256": "1" * 64},
    }

    with pytest.raises(ValueError, match="config drifted"):
        driver.build_run_runtime_config(
            payload, campaign_runtime, manifest_contract, run
        )


@pytest.mark.parametrize(
    "arm_id",
    (
        "flow_map2_normalized_eta_0p125",
        "paper_eta_0p125",
        "paper_eta_0p25_disable_i2",
    ),
)
def test_continuation_binds_algorithm_projection_not_runner_fields(
    monkeypatch, arm_id: str
) -> None:
    payload = runtime()
    monkeypatch.setattr(driver, "_formal_closure_for_runtime", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_bind_locked_schedule", lambda *a, **k: None)
    monkeypatch.setattr(
        driver,
        "_continuation_for_runtime",
        lambda *a, **k: {
            "continuation_contract_sha256": "1" * 64,
            "selected_arms": [
                {"arm_id": arm_id, "config_sha256": "2" * 64}
            ],
        },
    )
    monkeypatch.setattr(
        driver,
        "resolve_frozen_effective_guidance_config",
        lambda config: {**config, "arm_config_sha256": "3" * 64},
    )
    monkeypatch.setattr(
        driver,
        "canonical_r9_algorithm_config_digest",
        lambda *args, **kwargs: "2" * 64,
    )
    run = driver.RunSpec(
        phase="calibrate",
        logical_run_id=f"{arm_id}__seed_2027",
        arm_ref=arm_id,
        seed=2027,
        repeat_index=None,
        shard_index=0,
        num_shards=1,
        sample_count=64,
        manifest_key="calibration_64",
        runtime_config=Path("runtime.yaml"),
        output_dir=Path("output"),
        command=(),
    )
    calibration = payload["manifests"]["calibration_64"]
    manifest_contract = {
        "manifest_contracts_sha256": "4" * 64,
        "manifests": {"calibration_64": calibration},
    }
    campaign_runtime = {
        "campaign_id": driver.CONTINUATION_CHILD_CAMPAIGN_ID,
        "campaign_runtime_sha256": "5" * 64,
        "checkpoint": payload["checkpoint"],
        "continuation": {"contract_sha256": "1" * 64},
    }

    resolved = driver.build_run_runtime_config(
        payload, campaign_runtime, manifest_contract, run
    )

    assert resolved["arm_config_sha256"] == "3" * 64
    assert resolved["sampling_seed"] == 2027


def test_full_gate_forwards_runtime_bootstrap_seed(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_sha = "1" * 64
    runtime_sha = "2" * 64
    manifests_sha = "3" * 64
    captured: dict[str, int] = {}
    results = {
        "phase_results_sha256": "4" * 64,
        "automatic_evidence_sha256": "5" * 64,
        "run_plan_sha256": "6" * 64,
        "campaign_runtime_sha256": runtime_sha,
        "manifest_contracts_sha256": manifests_sha,
        "manifest_sha256": manifest_sha,
        "result": {"arm_id": "winner"},
    }

    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver, "_load_phase_results", lambda *a, **k: results)
    monkeypatch.setattr(driver, "_validate_phase_evidence_chain", lambda *a, **k: {})
    monkeypatch.setattr(
        driver, "_phase_evaluator_evidence_sha256", lambda *a, **k: "7" * 64
    )
    monkeypatch.setattr(driver, "_continuation_for_runtime", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_load_gate", lambda *a, **k: {})
    monkeypatch.setattr(driver, "_read_json_mapping", lambda *a, **k: {})
    monkeypatch.setattr(
        driver,
        "validate_selection_contract",
        lambda *a, **k: {"winner": {"arm_id": "winner"}},
    )
    monkeypatch.setattr(driver, "_require_selection_continuation", lambda *a: None)
    monkeypatch.setattr(driver, "_require_gate_continuation", lambda *a: None)
    monkeypatch.setattr(driver, "write_immutable_contract", lambda *a, **k: None)

    def build_d(*args, bootstrap_seed: int, **kwargs):
        del args, kwargs
        captured["bootstrap_seed"] = bootstrap_seed
        return {"verdict": "pass"}

    monkeypatch.setattr(driver, "build_d_gate_contract", build_d)
    gate = driver.finalize_phase_gate(
        {},
        {
            "campaign_root": "child",
            "campaign_runtime_sha256": runtime_sha,
            "checkpoint": {"sha256": "8" * 64},
            "bootstrap": {"seed": 91637},
        },
        {
            "manifest_contracts_sha256": manifests_sha,
            "manifests": {"full_2048": {"sha256": manifest_sha}},
        },
        {},
        phase="full",
        campaign_id="r9-child-v3",
    )

    assert gate == {"verdict": "pass"}
    assert captured == {"bootstrap_seed": 91637}


def test_full_finalize_gate_revalidates_frozen_selection_without_digest_cycle(
    monkeypatch, tmp_path: Path
) -> None:
    full_request = yaml.safe_load(
        (driver.REPO_ROOT / driver.FULL_CONTINUATION_RUNTIME_CONFIG).read_text(
            encoding="utf-8"
        )
    )
    source = full_request["source"]
    runtime = driver.load_runtime_config(
        driver.REPO_ROOT / full_request["base_runtime"]["path"]
    )
    continuation = driver.build_full_continuation_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    selection = driver.build_full_continuation_selection_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    assert "continuation_contract_sha256" not in selection
    selection_path = tmp_path / "full_continuation_selection.json"
    selection_path.write_text(
        json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    campaign_root = tmp_path / "child"
    campaign_root.mkdir()
    (campaign_root / "heldout_seal.json").write_text("{}\n", encoding="utf-8")
    manifest_sha = "1" * 64
    runtime_sha = "2" * 64
    manifests_sha = "3" * 64
    results = {
        "phase_results_sha256": "4" * 64,
        "automatic_evidence_sha256": "5" * 64,
        "run_plan_sha256": "6" * 64,
        "campaign_runtime_sha256": runtime_sha,
        "manifest_contracts_sha256": manifests_sha,
        "manifest_sha256": manifest_sha,
        "result": {"arm_id": "paper_eta_0p125"},
    }
    real_repo_path = driver._repo_path

    def repo_path(root, value, label):
        if label == "Full continuation selection":
            assert value == continuation["selection"]["path"]
            return selection_path
        return real_repo_path(root, value, label)

    monkeypatch.setattr(driver, "_repo_path", repo_path)
    monkeypatch.setattr(
        driver, "_continuation_for_runtime", lambda *a, **k: continuation
    )
    monkeypatch.setattr(driver, "_load_phase_results", lambda *a, **k: results)
    monkeypatch.setattr(driver, "_validate_phase_evidence_chain", lambda *a, **k: {})
    monkeypatch.setattr(
        driver, "_phase_evaluator_evidence_sha256", lambda *a, **k: "7" * 64
    )
    monkeypatch.setattr(driver, "_require_gate_continuation", lambda *a: None)
    monkeypatch.setattr(driver, "write_immutable_contract", lambda *a, **k: None)
    monkeypatch.setattr(
        driver,
        "build_d_gate_contract",
        lambda *a, **k: {
            "verdict": "pass",
            "selection": k["selection"]["selection_sha256"],
        },
    )
    gate = driver.finalize_phase_gate(
        runtime,
        {
            "campaign_root": str(campaign_root),
            "campaign_runtime_sha256": runtime_sha,
            "checkpoint": {"sha256": "8" * 64},
            "bootstrap": {"seed": 91637},
            "continuation": {
                "path": "unused",
                "file_sha256": "9" * 64,
                "contract_sha256": continuation["full_continuation_sha256"],
            },
        },
        {
            "manifest_contracts_sha256": manifests_sha,
            "manifests": {"full_2048": {"sha256": manifest_sha}},
        },
        {},
        phase="full",
        campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
    )
    assert gate["selection"] == selection["selection_sha256"]


def test_full_execution_is_blocked_without_formal_e2e_gate(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    continuation = {
        "start_phase": "full",
        "full_continuation_sha256": "a" * 64,
        "bindings": {
            "current_evaluation": {
                "classification": "canonical_current_v9_execution_authority",
                "worker": {"sha256": "d" * 64},
                "quality_script": {"sha256": "e" * 64},
            }
        },
    }
    monkeypatch.setattr(
        driver, "_continuation_for_runtime", lambda *a, **k: continuation
    )
    monkeypatch.setattr(
        driver,
        "_require_full_selection_binding",
        lambda *a, **k: {"selection_sha256": "b" * 64},
    )
    with pytest.raises(
        (FileNotFoundError, ValueError), match="Full E2E gate"
    ):
        driver._require_full_e2e_gate(
            {
                "campaign_root": "child",
                "manifests": {
                    "full_2048": {"path": "full.jsonl", "sha256": "c" * 64}
                },
                "evaluation": {
                    "worker": {"sha256": "d" * 64},
                    "quality": {"script": {"sha256": "e" * 64}},
                },
            }
        )


@pytest.mark.parametrize(
    ("phase", "logical_runs", "shards"),
    (
        ("preflight", 1, 4),
        ("diagnose", 39, 39),
        ("calibrate", 12, 12),
        ("confirm512", 3, 48),
        ("full", 2, 32),
    ),
)
def test_locked_phase_counts(phase: str, logical_runs: int, shards: int) -> None:
    plan = driver.build_phase_plan(runtime(), phase=phase, campaign_id="r9-test")
    assert plan.logical_run_count == logical_runs
    assert plan.shard_count == shards


def test_diagnose_is_thirteen_yaml_arms_repeated_three_times() -> None:
    plan = driver.build_phase_plan(runtime(), phase="diagnose", campaign_id="r9-test")
    arm_ids = [arm["arm_id"] for arm in runtime()["phases"]["diagnose"]["arms"]]
    for repeat_index in range(3):
        selected = [run for run in plan.runs if run.repeat_index == repeat_index]
        assert [run.arm_ref for run in selected] == arm_ids
        assert {run.seed for run in selected} == {1337}
        assert {run.sample_count for run in selected} == {18}


def test_calibrate_has_matched_native_for_every_seed() -> None:
    plan = driver.build_phase_plan(runtime(), phase="calibrate", campaign_id="r9-test")
    for seed in (1337, 2027, 3407):
        selected = [run for run in plan.runs if run.seed == seed]
        assert [run.arm_ref for run in selected] == [
            "native",
            "diagnose_candidate_0",
            "diagnose_candidate_1",
            "diagnose_candidate_2",
        ]


def test_confirm_and_full_sharding_contracts() -> None:
    confirm = driver.build_phase_plan(
        runtime(), phase="confirm512", campaign_id="r9-test"
    )
    full = driver.build_phase_plan(runtime(), phase="full", campaign_id="r9-test")
    assert {run.num_shards for run in confirm.runs} == {16}
    assert {run.sample_count for run in confirm.runs} == {512}
    assert {run.manifest_key for run in confirm.runs} == {"validate_512"}
    assert {run.num_shards for run in full.runs} == {16}
    assert {run.sample_count for run in full.runs} == {2048}
    assert {run.manifest_key for run in full.runs} == {"full_2048"}


def test_confirm_launch_schedule_is_latin_rotated_and_gpu_balanced() -> None:
    selected = ["paper_eta_0p125", "flow_map2_normalized_eta_0p125"]
    plan = driver.build_phase_plan(
        runtime(),
        phase="confirm512",
        campaign_id="r9-test",
        promoted_arm_ids=selected,
    )
    schedule = driver._generation_launch_schedule(plan)
    assert len(schedule) == 48
    assert [row["launch_index"] for row in schedule] == list(range(48))
    assert len(
        {(row["logical_run_id"], row["shard_index"]) for row in schedule}
    ) == 48
    first_arm_each_shard = [
        schedule[shard * 3]["arm_ref"] for shard in range(16)
    ]
    assert first_arm_each_shard[:6] == [
        "native",
        "paper_eta_0p125",
        "flow_map2_normalized_eta_0p125",
        "native",
        "paper_eta_0p125",
        "flow_map2_normalized_eta_0p125",
    ]
    for arm_id in ["native", *selected]:
        counts = {
            gpu: sum(
                row["arm_ref"] == arm_id and row["preferred_gpu_index"] == gpu
                for row in schedule
            )
            for gpu in range(4)
        }
        assert counts == {0: 4, 1: 4, 2: 4, 3: 4}


def test_full_launch_schedule_runs_one_arm_before_the_next() -> None:
    plan = driver.build_phase_plan(
        runtime(),
        phase="full",
        campaign_id="r9-test",
        winner_arm_id="paper_eta_0p125",
    )
    schedule = driver._generation_launch_schedule(plan)
    assert len(schedule) == 32
    assert [row["launch_index"] for row in schedule] == list(range(32))
    assert [row["logical_run_id"] for row in schedule[:16]] == ["native"] * 16
    assert [row["logical_run_id"] for row in schedule[16:]] == ["winner"] * 16
    for arm_id in ["native", "paper_eta_0p125"]:
        counts = {
            gpu: sum(
                row["arm_ref"] == arm_id and row["preferred_gpu_index"] == gpu
                for row in schedule
            )
            for gpu in range(4)
        }
        assert counts == {0: 4, 1: 4, 2: 4, 3: 4}


def test_dry_run_performs_no_write(tmp_path: Path) -> None:
    payload = runtime()
    payload["campaign_root"] = str(tmp_path / "campaigns")
    plans = driver.build_requested_plans(payload, phase="all", campaign_id="r9-test")
    rendered = json.loads(driver.render_dry_run(payload, plans))
    assert rendered["runtime_has_cli_algorithm_overrides"] is False
    assert not (tmp_path / "campaigns").exists()


def test_confirm_dry_run_reports_six_benchmark_runs_without_writes(
    tmp_path: Path,
) -> None:
    payload = runtime()
    payload["campaign_root"] = str(tmp_path / "campaigns")
    payload["generation_batch_benchmark"] = {
        "contract_path": "artifacts/benchmark.json",
        "output_root": "artifacts/benchmark",
        "manifest": "calibration_64",
        "sample_count": 8,
        "seed": 4549,
        "required_arms": [
            "native",
            "paper_eta_0p125",
            "flow_map2_normalized_eta_0p125",
        ],
        "batch_sizes": [2, 4],
        "record_final_latent_sha256": True,
    }
    plans = driver.build_requested_plans(
        payload,
        phase="confirm512",
        campaign_id=driver.CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID,
        continuation_selected_arm_ids=[
            "paper_eta_0p125",
            "flow_map2_normalized_eta_0p125",
        ],
        continuation_start_phase="confirm512",
    )
    rendered = json.loads(driver.render_dry_run(payload, plans))
    assert rendered["generation_batch_benchmark"] == {
        "logical_run_count": 6,
        "sample_run_count": 48,
        "arms": payload["generation_batch_benchmark"]["required_arms"],
        "batch_sizes": [2, 4],
        "contract_materialized": False,
    }
    assert not (tmp_path / "campaigns").exists()


def test_generation_batch_launcher_materializes_six_measured_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "calibration.jsonl"
    manifest.write_text(
        "".join(json.dumps({"sample_id": f"sample-{i}"}) + "\n" for i in range(8)),
        encoding="utf-8",
    )
    runtime_payload = {
        "python": sys.executable,
        "generation_script": "scripts/run_meanflow_guidance.py",
        "generation_batch_benchmark": {
            "contract_path": "artifacts/benchmark.json",
            "output_root": "artifacts/benchmark_runs",
            "manifest": "calibration_64",
            "sample_count": 8,
            "seed": 4549,
            "required_arms": [
                "native",
                "paper_eta_0p125",
                "flow_map2_normalized_eta_0p125",
            ],
            "batch_sizes": [2, 4],
            "record_final_latent_sha256": True,
        },
    }
    campaign_runtime = {
        "campaign_id": driver.CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID,
        "campaign_root": "artifacts/campaign",
        "campaign_runtime_sha256": "1" * 64,
        "continuation": {"contract_sha256": "2" * 64},
    }
    manifest_contract = {
        "manifest_contracts_sha256": "3" * 64,
        "manifests": {
            "calibration_64": {
                "path": str(manifest.relative_to(tmp_path)),
                "sha256": driver._sha256_path(manifest),
                "sample_count": 64,
            }
        },
    }
    monkeypatch.setattr(
        driver,
        "_continuation_for_runtime",
        lambda *a, **k: {
            "confirm_continuation_sha256": "2" * 64,
            "selected_arms": [],
        },
    )
    monkeypatch.setattr(
        driver,
        "build_run_runtime_config",
        lambda _runtime, _campaign, _manifests, run: {
            "seed": run.seed,
            "sampling_seed": run.seed,
            "mode": "native" if run.arm_ref == "native" else "paper_algorithm_split",
            "arm_config_sha256": "4" * 64,
        },
    )
    monkeypatch.setattr(
        driver, "resolve_frozen_effective_guidance_config", lambda config: config
    )
    monkeypatch.setattr(driver, "validate_worker_completion", lambda run: {})

    class Probe:
        snapshots = tuple(
            SimpleNamespace(
                index=index,
                uuid=f"GPU-{index}",
                total_bytes=32 * 1024**3,
                free_bytes=30 * 1024**3,
            )
            for index in range(4)
        )

        def gpu_snapshots(self):
            return self.snapshots

        def gpu_snapshot(self, index, *, expected_uuid):
            row = self.snapshots[index]
            assert row.uuid == expected_uuid
            return row

        @staticmethod
        def ram_snapshot():
            return SimpleNamespace(
                used_bytes=10 * 1024**3,
                total_bytes=128 * 1024**3,
            )

    launches = []

    class Process:
        pid = 12345

        def __init__(self):
            self.returncode = None
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                return None
            self.returncode = 0
            return 0

        def wait(self):
            self.returncode = 0
            return 0

    def process_factory(command, *, env, **kwargs):
        del kwargs
        config_path = tmp_path / command[command.index("--config") + 1]
        config = driver.yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output = tmp_path / command[command.index("--output-dir") + 1]
        output.mkdir(parents=True, exist_ok=True)
        logical_id = output.name
        arm_id = logical_id.rsplit("__batch_", 1)[0]
        launches.append((arm_id, config["batch_size"], env["CUDA_VISIBLE_DEVICES"]))
        rows = []
        for ordinal in range(8):
            generated = output / f"candidate-{ordinal}.png"
            native = output / f"native-{ordinal}.png"
            generated.write_bytes(f"{arm_id}:candidate:{ordinal}".encode())
            native.write_bytes(f"{arm_id}:native:{ordinal}".encode())
            rows.append(
                {
                    "sample_id": f"sample-{ordinal}",
                    "ordinal": ordinal,
                    "shard": 0,
                    "source": str(tmp_path / "source.png"),
                    "generated": str(generated),
                    "native": str(native),
                    "candidate_cosine": 0.7,
                    "native_cosine": 0.1,
                    "edev_cosine": 0.6,
                    "native_edev_cosine": 0.2,
                    "candidate_nfe": 7,
                    "native_nfe": 1,
                    "candidate_trace": [{"kind": "candidate"}],
                    "native_trace": [{"kind": "native"}],
                    "route_diagnostics": {"finite": True},
                    "candidate_latent_sha256": driver.hashlib.sha256(
                        f"{arm_id}:candidate:{ordinal}".encode()
                    ).hexdigest(),
                    "native_latent_sha256": driver.hashlib.sha256(
                        f"{arm_id}:native:{ordinal}".encode()
                    ).hexdigest(),
                    "candidate_generation_seconds": float(config["batch_size"]),
                    "native_generation_seconds": float(config["batch_size"]),
                    "io_seconds": float(config["batch_size"]),
                }
            )
        (output / "per_sample.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (output / "generation_result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "sample_count": 8,
                    "config": {
                        "batch_size": config["batch_size"],
                        "seed": 4549,
                        "record_final_latent_sha256": True,
                    },
                    "max_memory": {
                        "allocated_bytes": 4 * 1024**3 - 1,
                        "reserved_bytes": 4 * 1024**3,
                    },
                }
            ),
            encoding="utf-8",
        )
        return Process()

    (tmp_path / "source.png").write_bytes(b"source")
    contract = driver.run_generation_batch_benchmark(
        runtime_payload,
        campaign_runtime,
        manifest_contract,
        probe=Probe(),
        process_factory=process_factory,
        rss_sampler=lambda _pid: 2 * 1024**3,
        sleep=lambda _seconds: None,
    )
    assert launches == [
        ("native", 2, "GPU-0"),
        ("native", 4, "GPU-0"),
        ("paper_eta_0p125", 2, "GPU-1"),
        ("paper_eta_0p125", 4, "GPU-1"),
        ("flow_map2_normalized_eta_0p125", 2, "GPU-2"),
        ("flow_map2_normalized_eta_0p125", 4, "GPU-2"),
    ]
    assert contract["status"] == "ready"
    assert contract["decision"]["selected_batch_size"] == 4
    assert (tmp_path / "artifacts/benchmark_runs/benchmark_request.json").is_file()
    assert (tmp_path / "artifacts/benchmark.json").is_file()


def test_generation_commands_have_zero_semantic_cli_overrides() -> None:
    plans = driver.build_requested_plans(runtime(), phase="all", campaign_id="r9-test")
    allowed_flags = {"--config", "--output-dir", "--shard-index", "--num-shards"}
    for plan in plans:
        for run in plan.runs:
            flags = {value for value in run.command if value.startswith("--")}
            assert flags == allowed_flags


def test_runtime_yaml_locks_two_narrow_fmrg_families() -> None:
    phase = runtime()["phases"]["diagnose"]
    arms = phase["arms"]
    families = {arm["family"] for arm in arms if arm["family"] != "native"}
    assert families == {
        "flow_map2",
        "paper_split_constant",
        "paper_split_interval_ablation",
    }
    assert [
        arm["step_size"] for arm in arms if arm["family"] == "paper_split_constant"
    ] == [
        0.125,
        0.1875,
        0.25,
        0.3125,
        0.375,
        0.5,
    ]
    assert all(
        arm["sample_mode"] == "flow_map2"
        and arm["optimization_mode"] == "paper_normalized_direct_autograd"
        for arm in arms
        if arm["family"] == "flow_map2"
    )


def test_runtime_resource_constants_are_exact() -> None:
    resources = runtime()["resources"]
    assert {
        key: resources[key]
        for key in (
            "physical_gpus",
            "global_slot_lock_root",
            "max_slots_per_gpu",
            "gpu_slot_claim_bytes",
            "gpu_headroom_bytes",
            "ram_smoke_margin_numerator",
            "ram_smoke_margin_denominator",
            "ram_admission_percent",
            "ram_hard_limit_percent",
            "require_tmux",
            "retry_count",
        )
    } == {
        "physical_gpus": [0, 1, 2, 3],
        "global_slot_lock_root": "/tmp/safa-r9-gpu-slots-v1",
        "max_slots_per_gpu": 4,
        "gpu_slot_claim_bytes": 4_938_792_960,
        "gpu_headroom_bytes": 2 * 1024**3,
        "ram_smoke_margin_numerator": 110,
        "ram_smoke_margin_denominator": 100,
        "ram_admission_percent": 85,
        "ram_hard_limit_percent": 90,
        "require_tmux": True,
        "retry_count": 0,
    }
    assert resources["resource_smoke"] == {
        "required": True,
        "run_id": "native_smoke_calibration_64_v4",
        "arm_id": "native",
        "manifest": "calibration_64",
        "output_path": "artifacts/r9_meanflow_flow_map_guidance/shared/resource_smoke_v4.json",
        "factor": 1.10,
    }


def test_sealed_v2_remains_immutable_and_continuation_binds_current_evaluator() -> None:
    payload, request_path, _ = driver.load_continuation_request()
    request = driver.yaml.safe_load((ROOT / request_path).read_text(encoding="utf-8"))
    base_path = ROOT / request["base_runtime"]["path"]
    assert driver._sha256_path(base_path) == request["base_runtime"]["sha256"]
    worker = payload["evaluation"]["worker"]
    quality_script = payload["evaluation"]["quality"]["script"]
    assert worker["sha256"] == driver._sha256_path(ROOT / worker["path"])
    assert worker["implementation_sha256"] == driver._sha256_path(
        ROOT / worker["implementation_path"]
    )
    assert quality_script["sha256"] == driver._sha256_path(
        ROOT / quality_script["path"]
    )
    declared_arcface = payload["evaluation"]["arcface"]
    assert set(declared_arcface) == {
        "model_name",
        "model_root",
        "det_size",
        "provider",
        "insightface_version",
        "onnxruntime_version",
        "assets",
        "execution_probe",
    }
    probe_path = ROOT / declared_arcface["execution_probe"]["path"]
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    expected_full = _validate_arcface_contract(
        {**declared_arcface, "execution": probe["execution"]},
        repo_root=ROOT,
    )
    assert expected_full["execution"] == probe["execution"]


def test_raw_arcface_execution_injection_is_rejected() -> None:
    payload, request_path, _ = driver.load_continuation_request()
    payload = deepcopy(payload)
    probe_path = ROOT / payload["evaluation"]["arcface"]["execution_probe"]["path"]
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    payload["evaluation"]["arcface"]["execution"] = probe["execution"]

    with pytest.raises(ValueError, match="ArcFace evaluation fields are not canonical"):
        driver._build_effective_evaluation(
            payload,
            repo_root=ROOT,
            runtime_config={
                "path": str(request_path),
                "sha256": driver._sha256_path(ROOT / request_path),
            },
        )


def test_yaml_binds_exact_r8_and_diagnose_evidence() -> None:
    payload = runtime()
    assert payload["clean_source"]["arcface_exact_one"] is True
    construction = payload["manifest_construction"]
    assert construction["r8_calibration_64"] == {
        "path": "artifacts/r8_meanflow_flow_map_guidance/campaigns/r8-calibration-v3/calibration/native_unguided_64/sample_id_manifest.jsonl",
        "sha256": "b030b23ab5e688f709213f4671c1b12c2f53905a488909882234f0d5688b1a63",
        "sample_count": 64,
        "ordered_sample_id_sha256": "0e9dc5bd1da3c265efe4d66959cdc6649a6b60b82c29058adf0dab843b7c1df3",
    }
    assert construction["diagnose_18"]["matched_pair_sha256"] == (
        "080f45a7d6f108afa903df4e03d8773a198b83d878e435b8cf128436bcbc5c24"
    )


def _run_runtime_inputs() -> tuple[dict, dict, dict]:
    payload = runtime()
    manifests = {
        name: {
            key: payload["manifests"][name][key]
            for key in ("path", "sha256", "sample_count", "ordered_sample_id_sha256")
        }
        for name in ("calibration_64", "validate_512", "full_2048", "full_visual_64")
    }
    manifests["arcface_clean_pool"] = {
        "path": payload["clean_source"]["index"],
        "sha256": payload["clean_source"]["index_sha256"],
        "sample_count": payload["clean_source"]["sample_count"],
        "ordered_sample_id_sha256": payload["clean_source"]["ordered_sample_id_sha256"],
    }
    manifest_contract = {
        "manifests": manifests,
        "provenance": {"diagnose_18": payload["manifest_construction"]["diagnose_18"]},
        "manifest_contracts_sha256": "1" * 64,
    }
    campaign_runtime = {
        "campaign_id": "r9-test",
        "campaign_runtime_sha256": "2" * 64,
        "checkpoint": payload["checkpoint"],
        "schedule": {
            "path": payload["schedule_manifest"],
            "file_sha256": payload["schedule_manifest_sha256"],
            "contract_sha256": payload["schedule_contract_sha256"],
        },
        "semigroup_gate": {
            "path": payload["gate_contract"],
            "file_sha256": payload["gate_contract_sha256"],
            "contract_sha256": payload["gate_canonical_sha256"],
        },
    }
    return payload, manifest_contract, campaign_runtime


def test_run_runtime_config_binds_registered_arm_and_campaign_hashes() -> None:
    payload, manifest_contract, campaign_runtime = _run_runtime_inputs()
    plan = driver.build_phase_plan(payload, phase="diagnose", campaign_id="r9-test")
    paper_run = next(run for run in plan.runs if run.arm_ref == "paper_eta_0p25")
    native_run = next(run for run in plan.runs if run.arm_ref == "native")
    preflight_run = driver.build_phase_plan(
        payload, phase="preflight", campaign_id="r9-test"
    ).runs[0]
    paper = driver.build_run_runtime_config(
        payload, campaign_runtime, manifest_contract, paper_run
    )
    native = driver.build_run_runtime_config(
        payload, campaign_runtime, manifest_contract, native_run
    )
    preflight = driver.build_run_runtime_config(
        payload, campaign_runtime, manifest_contract, preflight_run
    )
    assert paper["mode"] == "paper_algorithm_split"
    assert paper["step_size"] == 0.25
    assert paper["active_guidance_intervals"] == ["I1", "I2", "I3"]
    assert paper["r9_campaign_runtime_sha256"] == "2" * 64
    assert paper["r9_manifest_contracts_sha256"] == "1" * 64
    assert paper["r9_semigroup_gate_contract_sha256"] == payload["gate_contract_sha256"]
    assert len(paper["arm_config_sha256"]) == 64
    assert driver.resolve_frozen_effective_guidance_config(paper) == paper
    assert native["mode"] == "native"
    assert len(native["arm_config_sha256"]) == 64
    assert "active_guidance_intervals" not in native
    assert "step_size" not in native
    assert preflight["mode"] == "semigroup"
    assert preflight["phase"] == "semigroup"
    assert preflight["r9_phase_contract"]["phase"] == "semigroup"
    assert len(preflight["arm_config_sha256"]) == 64


def test_confirm_runtime_config_binds_measured_batch_resource_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest_contract, campaign_runtime = _run_runtime_inputs()
    campaign_runtime.update(
        {
            "campaign_id": driver.CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID,
            "continuation": {"contract_sha256": "8" * 64},
            "generation_batch_benchmark": {
                "path": "artifacts/benchmark.json",
                "file_sha256": "9" * 64,
                "contract_sha256": "a" * 64,
            },
            "resources": {
                "generation_batch_size": 4,
                "generation_slots_per_gpu": 2,
                "gpu_slot_claim_bytes": 6_000_000_000,
                "ram_slot_budget_bytes": 3_000_000_000,
            },
        }
    )
    monkeypatch.setattr(driver, "_formal_closure_for_runtime", lambda *a, **k: None)
    monkeypatch.setattr(
        driver,
        "_continuation_for_runtime",
        lambda *a, **k: {
            "confirm_continuation_sha256": "8" * 64,
            "selected_arms": [
                {"arm_id": "paper_eta_0p125", "config_sha256": "b" * 64},
                {
                    "arm_id": "flow_map2_normalized_eta_0p125",
                    "config_sha256": "c" * 64,
                },
            ],
        },
    )
    run = driver.build_phase_plan(
        payload,
        phase="confirm512",
        campaign_id=driver.CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID,
        promoted_arm_ids=["paper_eta_0p125", "flow_map2_normalized_eta_0p125"],
    ).runs[0]
    config = driver.build_run_runtime_config(
        payload, campaign_runtime, manifest_contract, run
    )
    assert config["batch_size"] == 4
    assert config["r9_generation_batch_benchmark_sha256"] == "a" * 64
    assert config["r9_generation_gpu_slot_claim_bytes"] == 6_000_000_000
    assert config["r9_generation_ram_slot_budget_bytes"] == 3_000_000_000
    assert config["r9_generation_slots_per_gpu"] == 2


def test_resource_scheduler_caps_generation_slots_to_measured_declaration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    smoke_path = tmp_path / "shared" / "resource_smoke.json"
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        driver,
        "validate_resource_smoke_contract",
        lambda payload: {"peak_rss_bytes": 1_000_000},
    )

    class Probe:
        @staticmethod
        def gpu_snapshots():
            return tuple(
                SimpleNamespace(
                    index=index,
                    uuid=f"GPU-{index}",
                    free_bytes=24 * 1024**3,
                    total_bytes=24 * 1024**3,
                )
                for index in range(4)
            )

        @staticmethod
        def ram_snapshot():
            return SimpleNamespace(
                used_bytes=1_000_000,
                total_bytes=128 * 1024**3,
            )

    scheduler, _, _ = driver.build_resource_scheduler(
        {
            "campaign_id": "r9-test",
            "campaign_root": "campaigns/r9-test",
            "resources": {
                "physical_gpus": [0, 1, 2, 3],
                "global_slot_lock_root": str(tmp_path / "slots"),
                "max_slots_per_gpu": 4,
                "generation_slots_per_gpu": 2,
                "gpu_slot_claim_bytes": 4_987_027_456,
                "gpu_headroom_bytes": 2 * 1024**3,
                "ram_slot_budget_bytes": 2_000_000,
                "resource_smoke": {
                    "result": {"path": "shared/resource_smoke.json"},
                },
            },
        },
        probe=Probe(),
    )

    assert scheduler.max_gpu_slots == 2


def test_formal_preflight_binds_closure_seal_and_its_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, manifest_contract, campaign_runtime = _run_runtime_inputs()
    closure = {
        "bootstrap_campaign_id": "bootstrap-r9-test",
        "formal_campaign_id": "r9-test",
        "closure": {
            "path": "artifacts/r9/closures/bootstrap__for__r9-test/closure_seal.json",
            "file_sha256": "3" * 64,
            "contract_sha256": "4" * 64,
        },
        "schedule": dict(campaign_runtime["schedule"]),
        "gate": dict(campaign_runtime["semigroup_gate"]),
    }
    monkeypatch.setattr(
        driver,
        "resolve_formal_campaign_semigroup_closure",
        lambda *args, **kwargs: closure,
    )
    bound = []

    def bind_schedule(config, runtime_contract):
        bound.append(runtime_contract)
        config["schedule_manifest"] = runtime_contract["schedule"]["path"]
        config["schedule_contract_sha256"] = runtime_contract["schedule"][
            "contract_sha256"
        ]
        config["r9_semigroup_gate_contract"] = runtime_contract["semigroup_gate"][
            "path"
        ]
        config["r9_semigroup_gate_contract_sha256"] = runtime_contract[
            "semigroup_gate"
        ]["file_sha256"]

    monkeypatch.setattr(driver, "_bind_locked_schedule", bind_schedule)
    run = driver.build_phase_plan(
        payload, phase="preflight", campaign_id="r9-test"
    ).runs[0]
    config = driver.build_run_runtime_config(
        payload, campaign_runtime, manifest_contract, run
    )

    assert config["r9_semigroup_closure_seal"] == closure["closure"]["path"]
    assert config["r9_semigroup_closure_seal_sha256"] == "3" * 64
    assert config["r9_semigroup_closure_contract_sha256"] == "4" * 64
    assert config["r9_semigroup_bootstrap_campaign_id"] == "bootstrap-r9-test"
    assert config["schedule_manifest"] == campaign_runtime["schedule"]["path"]
    assert (
        config["schedule_contract_sha256"]
        == campaign_runtime["schedule"]["contract_sha256"]
    )
    assert (
        config["r9_semigroup_gate_contract"]
        == campaign_runtime["semigroup_gate"]["path"]
    )
    assert bound == [campaign_runtime]


def test_formal_campaign_cannot_fall_back_to_static_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, campaign_runtime = _run_runtime_inputs()
    replacement = {
        "bootstrap_campaign_id": "bootstrap-r9-test",
        "formal_campaign_id": "r9-test",
        "closure": {
            "path": "new/closure_seal.json",
            "file_sha256": "3" * 64,
            "contract_sha256": "4" * 64,
        },
        "schedule": {
            **campaign_runtime["schedule"],
            "path": "new/locked_schedule_manifest.json",
        },
        "gate": {
            **campaign_runtime["semigroup_gate"],
            "path": "new/gate_contract.json",
        },
    }
    monkeypatch.setattr(
        driver,
        "resolve_formal_campaign_semigroup_closure",
        lambda *args, **kwargs: replacement,
    )

    with pytest.raises(ValueError, match="closure schedule/gate"):
        driver._formal_closure_for_runtime(campaign_runtime)


def test_only_explicit_bootstrap_preflight_can_run_without_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, campaign_runtime = _run_runtime_inputs()
    monkeypatch.setattr(
        driver,
        "resolve_formal_campaign_semigroup_closure",
        lambda *args, **kwargs: None,
    )

    assert (
        driver._validate_requested_campaign_role(
            campaign_runtime, requested_phase="preflight"
        )
        is None
    )
    with pytest.raises(
        ValueError, match="bootstrap campaign can only execute preflight"
    ):
        driver._validate_requested_campaign_role(
            campaign_runtime, requested_phase="diagnose"
        )


def test_worker_completion_must_preserve_formal_closure_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    plan = _one_run_plan(tmp_path)
    run = plan.runs[0]
    runtime_path = tmp_path / run.runtime_config
    config = driver.yaml.safe_load(runtime_path.read_text())
    config.update(
        {
            "r9_semigroup_closure_seal": "closure/closure_seal.json",
            "r9_semigroup_closure_seal_sha256": "1" * 64,
            "r9_semigroup_closure_contract_sha256": "2" * 64,
            "r9_semigroup_bootstrap_campaign_id": "bootstrap-r9-test",
            "schedule_manifest": "closure/locked_schedule_manifest.json",
            "schedule_contract_sha256": "3" * 64,
            "r9_semigroup_gate_contract": "closure/gate_contract.json",
            "r9_semigroup_gate_contract_sha256": "4" * 64,
        }
    )
    runtime_path.write_text(driver.yaml.safe_dump(config), encoding="utf-8")
    _write_valid_completion(tmp_path, run, config)
    result_path = tmp_path / run.output_dir / "generation_result.json"
    result = json.loads(result_path.read_text())
    result["config"].pop("r9_semigroup_closure_seal")
    content = json.dumps(result, sort_keys=True) + "\n"
    result_path.write_text(content, encoding="utf-8")
    (tmp_path / run.output_dir / "run_manifest.json").write_text(
        content, encoding="utf-8"
    )

    with pytest.raises(ValueError, match="r9_semigroup_closure_seal mismatch"):
        driver.validate_worker_completion(run)


def test_immutable_runtime_write_rejects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    driver._write_immutable_bytes(path, b"mode: native\n")
    driver._write_immutable_bytes(path, b"mode: native\n")
    with pytest.raises(ValueError, match="different content"):
        driver._write_immutable_bytes(path, b"mode: paper_algorithm_split\n")


def test_invalid_campaign_id_is_rejected() -> None:
    with pytest.raises(SystemExit):
        driver.parse_args(["--campaign-id", "R9_bad"])


class _FakeScheduler:
    resource_contract_sha256 = "a" * 64
    ram_slot_budget_bytes = 1_100_000_000

    def __init__(self) -> None:
        self.requests = []
        self.released = []
        self.failures = []
        self.ram_checks = 0

    def admit_worker(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            status=driver.AdmissionStatus.ADMITTED,
            lease=SimpleNamespace(
                gpu_uuid=request.expected_gpu_uuid,
                slot_index=0,
            ),
        )

    def enforce_actual_ram_limit(self):
        self.ram_checks += 1
        return None

    def release_worker(self, worker_id):
        self.released.append(worker_id)

    def fail_worker(self, worker_id, *, kind):
        self.failures.append((worker_id, kind))
        raise driver.ResourceContractError(f"failed:{kind.value}")


class _StatusStore:
    def __init__(self) -> None:
        self.states = {}

    def record_admitted(self, worker_id, *, controller_pid=None):
        assert worker_id not in self.states
        self.states[worker_id] = "admitted"

    def record_running(self, worker_id, *, pid):
        assert pid > 0 and self.states[worker_id] == "admitted"
        self.states[worker_id] = "running"

    def record_terminal(self, worker_id, *, state):
        assert self.states[worker_id] in {"admitted", "running"}
        self.states[worker_id] = state

    def is_terminal(self, campaign_id, worker_id):
        return self.states.get(worker_id) in {"succeeded", "failed", "terminated"}


def _one_run_plan(tmp_path: Path) -> driver.PhasePlan:
    runtime_config = Path("runtime.yaml")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "sample-0", "source": "source.png"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "mode": "native",
        "max_samples": 1,
        "sample_id_manifest": str(manifest),
        "sample_id_manifest_sha256": driver._sha256_path(manifest),
        "checkpoint_sha256": "b" * 64,
        "arm_config_sha256": "c" * 64,
        "r9_campaign_id": "r9-test",
        "r9_campaign_runtime_sha256": "d" * 64,
        "r9_manifest_contracts_sha256": "e" * 64,
        "r9_phase_manifest_sha256": driver._sha256_path(manifest),
    }
    (tmp_path / runtime_config).write_text(
        driver.yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    run = driver.RunSpec(
        phase="diagnose",
        logical_run_id="native__repeat_0",
        arm_ref="native",
        seed=1337,
        repeat_index=0,
        shard_index=0,
        num_shards=1,
        sample_count=18,
        manifest_key="diagnose_18",
        runtime_config=runtime_config,
        output_dir=Path("output"),
        command=("python", "worker.py"),
    )
    plan = driver.PhasePlan(
        phase="diagnose",
        campaign_id="r9-test",
        campaign_root=Path("campaign"),
        logical_run_count=1,
        runs=(run,),
    )
    _write_valid_completion(tmp_path, run, config)
    return plan


def _write_valid_completion(tmp_path: Path, run, config: dict) -> None:
    output = tmp_path / run.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sample_sha = driver.hashlib.sha256(b"sample-0\n").hexdigest()
    result_path = output / "generation_result.json"
    run_manifest_path = output / "run_manifest.json"
    result = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": 1,
        "sample_id_sha256": sample_sha,
        "arm_config_sha256": config["arm_config_sha256"],
        "shard": {"index": 0, "count": 1},
        "checkpoint": {"sha256": config["checkpoint_sha256"]},
        "config": dict(config),
    }
    content = json.dumps(result, sort_keys=True) + "\n"
    result_path.write_text(content, encoding="utf-8")
    run_manifest_path.write_text(content, encoding="utf-8")
    completion = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": 1,
        "sample_id_sha256": sample_sha,
        "arm_config_sha256": config["arm_config_sha256"],
        "generation_result": str(run.output_dir / "generation_result.json"),
        "run_manifest": str(run.output_dir / "run_manifest.json"),
    }
    (output / "completion.json").write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_worker_completion_accepts_repo_relative_producer_paths(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    plan = _one_run_plan(tmp_path)

    verified = driver.validate_worker_completion(plan.runs[0])

    assert verified["worker_id"] == "diagnose:native__repeat_0:shard-0"
    assert (tmp_path / plan.runs[0].output_dir / "verified_completion.json").is_file()


def test_worker_completion_rejects_absolute_alias_for_relative_output(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    plan = _one_run_plan(tmp_path)
    run = plan.runs[0]
    completion_path = tmp_path / run.output_dir / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["generation_result"] = str(
        (tmp_path / run.output_dir / "generation_result.json").resolve()
    )
    completion["run_manifest"] = str(
        (tmp_path / run.output_dir / "run_manifest.json").resolve()
    )
    completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="worker completion disagrees with immutable run contract"
    ):
        driver.validate_worker_completion(run)


def test_worker_completion_rejects_output_dir_traversal(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(driver, "REPO_ROOT", repo_root)
    plan = _one_run_plan(repo_root)
    source = plan.runs[0]
    traversal = driver.RunSpec(
        **{
            **source.__dict__,
            "output_dir": Path("../outside"),
        }
    )

    with pytest.raises(ValueError, match="worker output directory escapes repo root"):
        driver.validate_worker_completion(traversal)


def _remove_completion(tmp_path: Path) -> None:
    for name in ("completion.json", "verified_completion.json"):
        (tmp_path / "output" / name).unlink(missing_ok=True)


class _FakeProcess:
    _next_pid = 10_000

    def __init__(self, returncode: int, *, running_polls: int = 1) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = returncode
        self.running_polls = running_polls
        self.terminated = False
        self.killed = False

    def poll(self):
        if self.running_polls:
            self.running_polls -= 1
            return None
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


def test_execute_admits_uuid_bound_worker_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    scheduler = _FakeScheduler()
    status_store = _StatusStore()
    launches = []

    plan = _one_run_plan(tmp_path)
    _remove_completion(tmp_path)

    def process_factory(command, *, cwd, env):
        launches.append((command, cwd, env))
        config = driver.yaml.safe_load((tmp_path / "runtime.yaml").read_text())
        _write_valid_completion(tmp_path, plan.runs[0], config)
        return _FakeProcess(0)

    result = driver.execute_campaign(
        (plan,),
        scheduler=scheduler,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        peer_status_store=status_store,
        process_factory=process_factory,
        sleep=lambda _: None,
    )
    assert result == 0
    assert len(scheduler.requests) == 1
    assert scheduler.requests[0].expected_gpu_uuid == "GPU-0"
    assert launches[0][2]["CUDA_VISIBLE_DEVICES"] == "GPU-0"
    assert scheduler.ram_checks == 2
    assert scheduler.released == ["diagnose:native__repeat_0:shard-0"]
    assert scheduler.failures == []
    assert status_store.states == {"diagnose:native__repeat_0:shard-0": "succeeded"}


def test_stale_live_peer_fails_admission_without_polling_forever() -> None:
    class StaleScheduler:
        resource_contract_sha256 = "a" * 64

        def admit_worker(self, request):
            return SimpleNamespace(
                status=driver.AdmissionStatus.STALE_PEER,
                lease=None,
                incumbent=SimpleNamespace(worker_id="campaign-a-live-worker"),
            )

    with pytest.raises(driver.StaleSlotLeaseError, match="campaign-a-live-worker"):
        driver._admit_worker(
            StaleScheduler(),
            worker_id="campaign-b-worker",
            launch_ordinal=10_000,
            gpu_bindings={index: f"GPU-{index}" for index in range(4)},
            ram_slot_budget_bytes=1_100_000_000,
        )


def test_execute_nonzero_fails_campaign_without_retry(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    scheduler = _FakeScheduler()
    launch_count = 0
    plan = _one_run_plan(tmp_path)
    _remove_completion(tmp_path)

    def process_factory(command, *, cwd, env):
        nonlocal launch_count
        launch_count += 1
        return _FakeProcess(7, running_polls=0)

    with pytest.raises(driver.ResourceContractError, match="peer_failure"):
        driver.execute_campaign(
            (plan,),
            scheduler=scheduler,
            gpu_bindings={index: f"GPU-{index}" for index in range(4)},
            peer_status_store=_StatusStore(),
            process_factory=process_factory,
            sleep=lambda _: None,
        )
    assert launch_count == 1
    assert scheduler.failures == [
        ("diagnose:native__repeat_0:shard-0", driver.FailureKind.PEER_FAILURE)
    ]
    assert scheduler.released == []


def test_zero_exit_missing_completion_is_contract_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    scheduler = _FakeScheduler()
    plan = _one_run_plan(tmp_path)
    _remove_completion(tmp_path)
    with pytest.raises(driver.ResourceContractError, match="contract_mismatch"):
        driver.execute_campaign(
            (plan,),
            scheduler=scheduler,
            gpu_bindings={index: f"GPU-{index}" for index in range(4)},
            peer_status_store=_StatusStore(),
            process_factory=lambda *args, **kwargs: _FakeProcess(0, running_polls=0),
            sleep=lambda _: None,
        )
    assert scheduler.failures[-1][1] is driver.FailureKind.CONTRACT_MISMATCH
    assert scheduler.released == []


def test_running_worker_is_terminated_on_ram_hard_limit(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)

    class RamFailScheduler(_FakeScheduler):
        def enforce_actual_ram_limit(self):
            self.ram_checks += 1
            failure = SimpleNamespace(
                reason="actual system RAM reached the R9 90% hard limit",
                terminate_worker_id="diagnose:native__repeat_0:shard-0",
            )
            raise driver.CampaignFailedError(failure)

    scheduler = RamFailScheduler()
    process = _FakeProcess(0, running_polls=100)
    plan = _one_run_plan(tmp_path)
    _remove_completion(tmp_path)
    with pytest.raises(
        driver.CampaignFailedError,
        match="actual system RAM reached",
    ):
        driver.execute_campaign(
            (plan,),
            scheduler=scheduler,
            gpu_bindings={index: f"GPU-{index}" for index in range(4)},
            peer_status_store=_StatusStore(),
            process_factory=lambda *args, **kwargs: process,
            sleep=lambda _: None,
        )
    assert process.terminated is True
    assert process.killed is False
    assert scheduler.released == ["diagnose:native__repeat_0:shard-0"]


def test_resource_smoke_measures_rss_and_exclusively_writes_contract(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    plan = _one_run_plan(tmp_path)
    source_run = plan.runs[0]
    smoke_run = driver.RunSpec(
        **{
            **source_run.__dict__,
            "phase": "resource_smoke",
            "logical_run_id": "native_smoke_calibration_64_v4",
        }
    )
    declaration = {
        "run_id": "native_smoke_calibration_64_v4",
        "arm_id": "native",
        "manifest": "calibration_64",
        "output_path": "shared/resource_smoke_v4.json",
    }
    monkeypatch.setattr(
        driver,
        "materialize_resource_smoke_runtime",
        lambda *args, **kwargs: (smoke_run, declaration),
    )
    monkeypatch.setattr(
        driver,
        "validate_worker_completion",
        lambda run: {"verified": True},
    )

    class Probe:
        def ram_snapshot(self):
            return SimpleNamespace(
                total_bytes=1_000_000,
                used_bytes=100_000,
            )

        def gpu_snapshots(self):
            return tuple(
                SimpleNamespace(
                    index=index,
                    uuid=f"GPU-{index}",
                    free_bytes=20 * 1024**3,
                    total_bytes=24 * 1024**3,
                )
                for index in range(4)
            )

    process = None
    rss_samples = 0
    observed_exit_reasons = []

    def process_factory(*args, **kwargs):
        nonlocal process
        process = subprocess.Popen(
            [sys.executable, "-c", "import os; os.read(0, 1)"],
            stdin=subprocess.PIPE,
        )
        return process

    def race_safe_sampler(pid):
        nonlocal rss_samples
        assert process is not None
        assert pid == process.pid
        rss_samples += 1
        if rss_samples == 1:
            return driver._process_tree_rss_bytes(pid)
        assert process.poll() is None
        assert process.stdin is not None
        process.stdin.write(b"x")
        process.stdin.close()
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        assert process.returncode is None
        try:
            return driver._process_tree_rss_bytes(pid)
        except driver._ProcessTreeRootExitObserved as error:
            observed_exit_reasons.append(error.reason)
            raise

    runtime_payload = {
        "campaign_root": "campaigns",
        "checkpoint": {"sha256": "b" * 64},
    }
    claim = {
        "campaign_runtime_sha256": None,
        "campaign_claim_sha256": "a" * 64,
    }
    manifest_contract = {
        "manifests": {
            "calibration_64": {"sha256": "f" * 64},
        }
    }
    try:
        contract = driver.run_resource_smoke(
            runtime_payload,
            claim,
            manifest_contract,
            probe=Probe(),
            process_factory=process_factory,
            rss_sampler=race_safe_sampler,
            sleep=lambda _: None,
        )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    assert contract["peak_rss_bytes"] > 0
    assert rss_samples == 2
    assert observed_exit_reasons == []
    assert process is not None and process.returncode == 0
    result_path = tmp_path / declaration["output_path"]
    assert json.loads(result_path.read_text(encoding="utf-8")) == contract
    with pytest.raises(FileExistsError):
        driver._write_exclusive_bytes(result_path, b"tamper")


def test_process_tree_rss_classifies_reaped_root_as_vanished() -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    assert process.wait(timeout=1.0) == 0
    with pytest.raises(driver._ProcessTreeRootExitObserved) as observed:
        driver._process_tree_rss_bytes(process.pid)
    assert observed.value.pid == process.pid
    assert observed.value.reason == "vanished"


def _proc_stat(
    pid: int,
    *,
    state: str = "S",
    starttime: int = 41,
    comm: bytes = b"test",
) -> bytes:
    suffix = [state.encode("ascii"), *([b"0"] * 18), str(starttime).encode("ascii")]
    return str(pid).encode("ascii") + b" (" + comm + b") " + b" ".join(suffix) + b"\n"


def _write_proc_node(
    proc_root: Path,
    pid: int,
    *,
    state: str = "S",
    starttime: int = 41,
    resident_pages: int = 7,
    children: str = "",
    comm: bytes = b"test",
) -> None:
    root = proc_root / str(pid)
    task = root / "task" / str(pid)
    task.mkdir(parents=True)
    (root / "stat").write_bytes(
        _proc_stat(pid, state=state, starttime=starttime, comm=comm)
    )
    (root / "statm").write_text(
        f"10 {resident_pages} 0 0 0 0 0\n", encoding="ascii"
    )
    (task / "children").write_text(children, encoding="ascii")


def test_process_tree_rss_exec_transition_uses_zero_statm_resident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_pid = 123
    proc_root = tmp_path / "proc"
    _write_proc_node(
        proc_root,
        root_pid,
        state="R",
        resident_pages=0,
        comm=b"exec ) transition \xff",
    )
    monkeypatch.setattr(driver.os, "sysconf", lambda name: 16384)
    assert driver._process_tree_rss_bytes(root_pid, proc_root=proc_root) == 0


@pytest.mark.parametrize("page_size", [4096, 16384])
def test_process_tree_rss_multiplies_statm_resident_by_system_page_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    page_size: int,
) -> None:
    root_pid = 123
    proc_root = tmp_path / "proc"
    _write_proc_node(proc_root, root_pid, resident_pages=7)
    monkeypatch.setattr(driver.os, "sysconf", lambda name: page_size)
    assert driver._process_tree_rss_bytes(root_pid, proc_root=proc_root) == 7 * page_size


def test_process_tree_rss_exit_between_identity_reads_is_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_pid = 123
    proc_root = tmp_path / "proc"
    _write_proc_node(proc_root, root_pid)
    stat_path = proc_root / str(root_pid) / "stat"
    original = Path.read_bytes
    reads = 0

    def vanish_on_second_read(path: Path) -> bytes:
        nonlocal reads
        if path == stat_path:
            reads += 1
            if reads == 2:
                raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", vanish_on_second_read)
    with pytest.raises(driver._ProcessTreeRootExitObserved, match="vanished"):
        driver._process_tree_rss_bytes(root_pid, proc_root=proc_root)


def test_process_tree_rss_pid_reuse_between_identity_reads_is_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_pid = 123
    proc_root = tmp_path / "proc"
    _write_proc_node(proc_root, root_pid, starttime=41)
    stat_path = proc_root / str(root_pid) / "stat"
    original = Path.read_bytes
    reads = 0

    def reuse_on_second_read(path: Path) -> bytes:
        nonlocal reads
        if path == stat_path:
            reads += 1
            if reads == 2:
                return _proc_stat(root_pid, starttime=42)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reuse_on_second_read)
    with pytest.raises(driver._ProcessTreeRootExitObserved, match="vanished"):
        driver._process_tree_rss_bytes(root_pid, proc_root=proc_root)


def test_process_tree_rss_zombie_is_zero_without_statm(tmp_path: Path) -> None:
    root_pid = 123
    proc_root = tmp_path / "proc"
    root = proc_root / str(root_pid)
    root.mkdir(parents=True)
    (root / "stat").write_bytes(_proc_stat(root_pid, state="Z"))
    assert driver._process_tree_rss_bytes(root_pid, proc_root=proc_root) == 0


def test_process_tree_rss_stable_live_malformed_statm_is_hard_failure(
    tmp_path: Path,
) -> None:
    root_pid = 123
    proc_root = tmp_path / "proc"
    _write_proc_node(proc_root, root_pid)
    (proc_root / str(root_pid) / "statm").write_bytes(b"10 not-a-number\n")
    with pytest.raises(
        driver.ResourceContractError,
        match=r"live process 123 statm format is invalid",
    ):
        driver._process_tree_rss_bytes(root_pid, proc_root=proc_root)


def test_process_tree_rss_ignores_descendant_that_vanishes_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_pid = 123
    proc_root = tmp_path / "proc"
    _write_proc_node(
        proc_root,
        root_pid,
        resident_pages=17,
        children="456\n",
    )
    monkeypatch.setattr(driver.os, "sysconf", lambda name: 16384)
    assert (
        driver._process_tree_rss_bytes(root_pid, proc_root=proc_root) == 17 * 16384
    )


def test_root_exit_wait_timeout_while_process_is_live_is_hard_failure(
    tmp_path: Path,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import os; os.read(0, 1)"],
        stdin=subprocess.PIPE,
    )
    empty_proc_root = tmp_path / "proc"
    empty_proc_root.mkdir()
    try:
        with pytest.raises(
            driver.ResourceContractError,
            match=r"did not exit within 1\.0 seconds",
        ):
            driver._sample_or_reap_process_tree(
                process,
                lambda pid: driver._process_tree_rss_bytes(
                    pid, proc_root=empty_proc_root
                ),
            )
        assert process.poll() is None
    finally:
        process.kill()
        process.wait()


def test_dynamic_plans_use_only_gate_promoted_registered_arms() -> None:
    payload = runtime()
    promoted_a = ["paper_eta_0p25", "flow_map2_normalized_eta_0p1875"]
    calibrate = driver.build_phase_plan(
        payload,
        phase="calibrate",
        campaign_id="r9-test",
        promoted_arm_ids=promoted_a,
    )
    for seed in (1337, 2027, 3407):
        assert [run.arm_ref for run in calibrate.runs if run.seed == seed] == [
            "native",
            *promoted_a,
        ]
    confirm = driver.build_phase_plan(
        payload,
        phase="confirm512",
        campaign_id="r9-test",
        promoted_arm_ids=[promoted_a[0]],
    )
    assert {run.arm_ref for run in confirm.runs} == {"native", promoted_a[0]}
    full = driver.build_phase_plan(
        payload,
        phase="full",
        campaign_id="r9-test",
        winner_arm_id=promoted_a[0],
    )
    assert {run.arm_ref for run in full.runs} == {"native", promoted_a[0]}


def test_phase_results_request_locks_exact_diagnose_run_plan() -> None:
    payload = runtime()
    plan = driver.build_phase_plan(payload, phase="diagnose", campaign_id="r9-test")
    diagnose = payload["manifest_construction"]["diagnose_18"]
    request = driver.build_phase_results_request(
        payload,
        {
            "campaign_runtime_sha256": "a" * 64,
            "campaign_root": str(plan.campaign_root),
            "checkpoint": {"sha256": "b" * 64},
            "bootstrap": {"seed": 91637},
            "evaluation": {
                "quality": {
                    "real_index": {
                        "path": "data/index/val_face_mixed_e14.jsonl",
                        "sha256": "d" * 64,
                    }
                }
            },
            "phases": payload["phases"],
        },
        {
            "manifest_contracts_sha256": "c" * 64,
            "manifests": {},
        },
        diagnose,
        plan=plan,
        campaign_id="r9-test",
    )
    assert request.expected_candidate_arm_ids == tuple(
        arm["arm_id"]
        for arm in payload["phases"]["diagnose"]["arms"]
        if arm["arm_id"] != "native"
    )
    assert request.expected_seeds == (1337,)
    assert request.upstream_gate is None
    assert request.source_index_path == (
        driver.REPO_ROOT / "data/index/val_face_mixed_e14.jsonl"
    )
    assert request.source_index_sha256 == "d" * 64
    assert len(request.runs) == 39


@pytest.mark.parametrize(
    "promoted",
    ([], ["unknown"], ["native"], ["paper_eta_0p25", "paper_eta_0p25"]),
)
def test_dynamic_plan_rejects_zero_invalid_or_duplicate_promotions(promoted) -> None:
    with pytest.raises(ValueError, match="promotion"):
        driver.build_phase_plan(
            runtime(),
            phase="calibrate",
            campaign_id="r9-test",
            promoted_arm_ids=promoted,
        )


def test_phase_results_digest_tamper_is_rejected(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_results_v1",
        "phase": "diagnose",
        "campaign_runtime_sha256": "a" * 64,
        "manifest_contracts_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "automatic_evidence_sha256": "d" * 64,
        "run_plan_sha256": "e" * 64,
        "arms": [],
    }
    payload["phase_results_sha256"] = driver._canonical_json_sha256(payload)
    path = tmp_path / "phase_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert driver._load_phase_results(path, "diagnose")["arms"] == []
    payload["manifest_sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        driver._load_phase_results(path, "diagnose")


def test_phase_results_must_bind_current_automatic_evidence(tmp_path: Path) -> None:
    automatic = {
        "schema_version": 1,
        "contract_type": "safa_r9_automatic_phase_evidence_v1",
        "phase": "diagnose",
        "run_plan_sha256": "a" * 64,
    }
    automatic["automatic_evidence_sha256"] = driver._canonical_json_sha256(automatic)
    phase_root = tmp_path / "diagnose"
    phase_root.mkdir()
    (phase_root / "automatic_evidence.json").write_text(
        json.dumps(automatic), encoding="utf-8"
    )
    results = {
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "run_plan_sha256": "a" * 64,
    }
    assert (
        driver._validate_phase_evidence_chain(tmp_path, "diagnose", results)
        == automatic
    )
    results["run_plan_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="current run plan"):
        driver._validate_phase_evidence_chain(tmp_path, "diagnose", results)


def test_all_stops_before_b_when_a_has_zero_candidates(monkeypatch) -> None:
    built = []
    fake_plan = driver.PhasePlan(
        phase="diagnose",
        campaign_id="r9-test",
        campaign_root=Path("campaign"),
        logical_run_count=0,
        runs=(),
    )
    monkeypatch.setattr(driver, "PHASES", ("diagnose", "calibrate"))
    monkeypatch.setattr(
        driver,
        "resolve_phase_promotion",
        lambda *args, phase, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        driver,
        "build_phase_plan",
        lambda *args, phase, **kwargs: built.append(phase) or fake_plan,
    )
    monkeypatch.setattr(driver, "materialize_phase_runtime_configs", lambda *a, **k: {})
    monkeypatch.setattr(driver, "build_phase_results_request", lambda *a, **k: object())
    monkeypatch.setattr(
        driver,
        "resume_phase_results",
        lambda *a, **k: _closure("needs_generation"),
    )
    monkeypatch.setattr(
        driver,
        "materialize_phase_results",
        lambda *a, **k: _closure("complete"),
    )
    monkeypatch.setattr(driver, "execute_campaign", lambda *a, **k: 0)
    monkeypatch.setattr(
        driver,
        "finalize_phase_gate",
        lambda *a, **k: {"verdict": "stop_zero_candidates"},
    )
    assert (
        driver.execute_dynamic_campaign(
            {},
            {},
            {},
            {},
            requested_phase="all",
            campaign_id="r9-test",
            scheduler=SimpleNamespace(),
            gpu_bindings={},
            peer_status_store=_StatusStore(),
        )
        == 0
    )
    assert built == ["diagnose"]


def _closure(status: str):
    return SimpleNamespace(
        status=status,
        phase_results_path=(
            Path("phase_results.json") if status == "complete" else None
        ),
        awaiting_path=(
            Path("awaiting_visual_review.json")
            if status == "awaiting_visual_review"
            else None
        ),
        required_review_count=2,
        completed_review_count=(2 if status == "complete" else 0),
    )


def _patch_dynamic_phase(monkeypatch, *, resume_status: str, materialize_status: str):
    plan = driver.PhasePlan(
        phase="diagnose",
        campaign_id="r9-test",
        campaign_root=Path("campaign"),
        logical_run_count=0,
        runs=(),
    )
    calls = {"execute": 0, "materialize": 0, "gate": 0}
    monkeypatch.setattr(driver, "PHASES", ("diagnose",))
    monkeypatch.setattr(driver, "resolve_phase_promotion", lambda *a, **k: (None, None))
    monkeypatch.setattr(driver, "build_phase_plan", lambda *a, **k: plan)
    monkeypatch.setattr(driver, "materialize_phase_runtime_configs", lambda *a, **k: {})
    monkeypatch.setattr(driver, "build_phase_results_request", lambda *a, **k: object())
    monkeypatch.setattr(
        driver, "resume_phase_results", lambda *a, **k: _closure(resume_status)
    )

    def execute(*args, **kwargs):
        calls["execute"] += 1
        return 0

    def materialize(*args, **kwargs):
        calls["materialize"] += 1
        return _closure(materialize_status)

    def gate(*args, **kwargs):
        calls["gate"] += 1
        return {"verdict": "advance"}

    monkeypatch.setattr(driver, "execute_campaign", execute)
    monkeypatch.setattr(driver, "materialize_phase_results", materialize)
    monkeypatch.setattr(driver, "finalize_phase_gate", gate)
    return calls


def _execute_dynamic() -> int:
    return driver.execute_dynamic_campaign(
        {},
        {},
        {},
        {},
        requested_phase="all",
        campaign_id="r9-test",
        scheduler=SimpleNamespace(),
        gpu_bindings={},
        peer_status_store=_StatusStore(),
    )


def test_dynamic_phase_resumes_awaiting_review_without_generation(monkeypatch) -> None:
    calls = _patch_dynamic_phase(
        monkeypatch,
        resume_status="awaiting_visual_review",
        materialize_status="complete",
    )
    assert _execute_dynamic() == driver.AWAITING_VISUAL_REVIEW_EXIT_CODE
    assert calls == {"execute": 0, "materialize": 0, "gate": 0}


def test_dynamic_phase_bounds_after_materializing_review_evidence(monkeypatch) -> None:
    calls = _patch_dynamic_phase(
        monkeypatch,
        resume_status="needs_generation",
        materialize_status="awaiting_visual_review",
    )
    assert _execute_dynamic() == driver.AWAITING_VISUAL_REVIEW_EXIT_CODE
    assert calls == {"execute": 1, "materialize": 1, "gate": 0}


def test_dynamic_phase_resumes_complete_results_directly_to_gate(monkeypatch) -> None:
    calls = _patch_dynamic_phase(
        monkeypatch,
        resume_status="complete",
        materialize_status="complete",
    )
    assert _execute_dynamic() == 0
    assert calls == {"execute": 0, "materialize": 0, "gate": 1}


def test_dynamic_phase_closes_generation_then_builds_gate(monkeypatch) -> None:
    calls = _patch_dynamic_phase(
        monkeypatch,
        resume_status="needs_generation",
        materialize_status="complete",
    )
    assert _execute_dynamic() == 0
    assert calls == {"execute": 1, "materialize": 1, "gate": 1}


def test_production_evaluator_callback_runs_bound_worker_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    worker = tmp_path / "scripts" / "run_r9_phase_evaluator.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("raise SystemExit(0)\n", encoding="utf-8")
    implementation = tmp_path / "src" / "safa" / "evaluation" / "r9_evaluator_worker.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# test worker implementation\n", encoding="utf-8")
    quality_script = tmp_path / "scripts" / "eval_generation_quality.py"
    quality_script.write_text(
        "def evaluate_generation_quality(**kwargs): return kwargs\n",
        encoding="utf-8",
    )
    campaign_root = tmp_path / "campaign"
    arcface_contract = _arcface_contract(tmp_path)
    arcface_contract.pop("execution")
    evaluation = {
        "worker": {
            "path": "scripts/run_r9_phase_evaluator.py",
            "sha256": driver._sha256_path(worker),
            "implementation_path": "src/safa/evaluation/r9_evaluator_worker.py",
            "implementation_sha256": driver._sha256_path(implementation),
        },
        "quality": {
            "script": {
                "path": "scripts/eval_generation_quality.py",
                "sha256": driver._sha256_path(quality_script),
            }
        },
        "arcface": arcface_contract,
        "heldout": {"batch_size": 16},
        "resource_smokes": {
            "arcface": {"ram_slot_budget_bytes": 1_758_923_162},
            "quality": {"ram_slot_budget_bytes": 2_301_343_335},
            "heldout": {
                "mode": "exclusive_single_official_run",
                "smoke_execution": "sealed_until_winner_lock",
                "global_exclusive_slots": 16,
                "ram_admission_percent": 85,
                "ram_hard_limit_percent": 90,
            },
        },
    }

    class DelayedScheduler(_FakeScheduler):
        def __init__(self) -> None:
            super().__init__()
            self.admission_calls = 0

        def admit_worker(self, request):
            self.admission_calls += 1
            if self.admission_calls <= 4:
                return SimpleNamespace(
                    status=driver.AdmissionStatus.GPU_LIMIT,
                    lease=None,
                )
            return super().admit_worker(request)

    scheduler = DelayedScheduler()
    status = _StatusStore()
    calls = []
    wait_checks = []
    attempt_path = (
        campaign_root
        / "calibrate"
        / "evaluator_runs"
        / "quality"
        / "candidate__seed_1337__candidate"
        / "attempt.json"
    )

    class Process:
        pid = 1234
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def process_factory(command, **kwargs):
        assert attempt_path.is_file()
        calls.append((command, kwargs))
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        assert request_payload["payload"]["source_index_path"] == str(
            source_index.resolve()
        )
        assert request_payload["payload"]["source_index_sha256"] == index_sha256
        assert request_payload["payload"]["samples"][0]["source_sha256"] == image_sha256
        result = {
            "schema_version": 1,
            "contract_type": "safa_r9_phase_evaluator_output_v1",
            "task": "quality",
            "evaluator_request_sha256": request_payload["evaluator_request_sha256"],
            "worker_contract": request_payload["config"]["worker_contract"],
            "arcface_contract_sha256": driver._canonical_json_sha256(
                request_payload["config"]["arcface"]
            ),
            "quality_script_sha256": driver._sha256_path(quality_script),
            "result": {"strict": True},
        }
        result["evaluator_output_sha256"] = driver._canonical_json_sha256(result)
        output_path.write_text(json.dumps(result), encoding="utf-8")
        return Process()

    callbacks = driver.R9ProductionEvaluatorCallbacks(
        runtime={"python": sys.executable},
        campaign_runtime={
            "campaign_root": str(campaign_root),
            "evaluation": evaluation,
        },
        scheduler=scheduler,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        peer_status_store=status,
        process_factory=process_factory,
        sleep=lambda _: wait_checks.append(not attempt_path.exists()),
    )
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    image_sha256 = driver._sha256_path(image)
    source_index = tmp_path / "source_index.jsonl"
    source_index.write_text(
        json.dumps({"sample_id": "sample-0", "image_path": str(image.resolve())})
        + "\n",
        encoding="utf-8",
    )
    index_sha256 = driver._sha256_path(source_index)
    sample = SampleEvidence(
        sample_id="sample-0",
        source=image,
        native=image,
        candidate=image,
        source_sha256=image_sha256,
        native_sha256=image_sha256,
        candidate_sha256=image_sha256,
    )
    request = driver.QualityEvaluationRequest(
        phase="calibrate",
        logical_run_id="candidate__seed_1337",
        arm_id="candidate",
        seed=1337,
        image_role="candidate",
        manifest_path=tmp_path / "manifest.jsonl",
        source_index_path=source_index,
        source_index_sha256=index_sha256,
        samples=(sample,),
        algorithm_config_sha256="a" * 64,
        runner_arm_config_sha256="b" * 64,
        semantic_output_sha256="c" * 64,
        evidence_binding_sha256="d" * 64,
        generation_result_set_sha256="e" * 64,
        per_sample_set_sha256="f" * 64,
    )
    assert callbacks.quality(request) == {"strict": True}
    assert callbacks.quality(request) == {"strict": True}
    assert len(calls) == 1
    assert wait_checks == [True]
    assert scheduler.released == [
        "evaluator:quality:calibrate:candidate__seed_1337__candidate"
    ]
    assert scheduler.requests[-1].ram_slot_budget_bytes == 2_301_343_335
    assert set(status.states.values()) == {"succeeded"}


class _FourGpuSlotScheduler:
    resource_contract_sha256 = "a" * 64
    ram_slot_budget_bytes = 1_100_000_000

    def __init__(self, capacities=None) -> None:
        self.active = {}
        self.max_active = 0
        self.admitted_slots = []
        self.capacities = capacities or {f"GPU-{index}": 4 for index in range(4)}

    @property
    def active_leases(self):
        return tuple(self.active.values())

    def admit_worker(self, request):
        occupied = {
            (lease.gpu_uuid, lease.slot_index) for lease in self.active.values()
        }
        slot = next(
            (
                index
                for index in range(self.capacities[request.expected_gpu_uuid])
                if (request.expected_gpu_uuid, index) not in occupied
            ),
            None,
        )
        if slot is None:
            return SimpleNamespace(
                status=driver.AdmissionStatus.GPU_LIMIT,
                lease=None,
            )
        lease = SimpleNamespace(
            worker_id=request.worker_id,
            gpu_uuid=request.expected_gpu_uuid,
            slot_index=slot,
            launch_ordinal=request.launch_ordinal,
        )
        self.active[request.worker_id] = lease
        self.admitted_slots.append((lease.gpu_uuid, slot))
        self.max_active = max(self.max_active, len(self.active))
        return SimpleNamespace(
            status=driver.AdmissionStatus.ADMITTED,
            lease=lease,
        )

    def enforce_actual_ram_limit(self):
        return None

    def release_worker(self, worker_id):
        if worker_id not in self.active:
            raise driver.ResourceContractError("unknown worker")
        del self.active[worker_id]

    def fail_worker(self, worker_id, *, kind):
        self.active.pop(worker_id, None)
        failure = SimpleNamespace(reason=f"failed:{kind.value}")
        raise driver.CampaignFailedError(failure)


def _many_run_plan(tmp_path: Path, count: int) -> driver.PhasePlan:
    runs = []
    for index in range(count):
        runtime_path = Path(f"runtime-{index}.yaml")
        (tmp_path / runtime_path).write_text("mode: native\n", encoding="utf-8")
        runs.append(
            driver.RunSpec(
                phase="diagnose",
                logical_run_id=f"arm-{index}",
                arm_ref=f"arm-{index}",
                seed=1337,
                repeat_index=0,
                shard_index=0,
                num_shards=1,
                sample_count=1,
                manifest_key="diagnose_18",
                runtime_config=runtime_path,
                output_dir=Path(f"output-{index}"),
                command=("python", f"worker-{index}.py"),
            )
        )
    return driver.PhasePlan(
        phase="diagnose",
        campaign_id="r9-test",
        campaign_root=Path("campaign"),
        logical_run_count=count,
        runs=tuple(runs),
    )


def _full_shard_plan(tmp_path: Path, count: int) -> driver.PhasePlan:
    runs = []
    for index in range(count):
        runtime_path = Path(f"full-runtime-{index}.yaml")
        (tmp_path / runtime_path).write_text(
            "mode: paper_algorithm_split\n", encoding="utf-8"
        )
        runs.append(
            driver.RunSpec(
                phase="full",
                logical_run_id="winner",
                arm_ref="paper_eta_0p125",
                seed=7919,
                repeat_index=None,
                shard_index=index,
                num_shards=count,
                sample_count=128,
                manifest_key="full_2048",
                runtime_config=runtime_path,
                output_dir=Path(f"full/winner/shards/shard_{index}"),
                command=("python", f"full-worker-{index}.py"),
            )
        )
    return driver.PhasePlan(
        phase="full",
        campaign_id="r9-test",
        campaign_root=Path("campaign"),
        logical_run_count=1,
        runs=tuple(runs),
    )


def test_execute_refills_four_by_four_slots_without_exceeding_sixteen(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver, "validate_worker_completion", lambda run: {})
    scheduler = _FourGpuSlotScheduler()
    allow_complete = {"value": False}
    processes = []

    class GateProcess(_FakeProcess):
        def poll(self):
            return 0 if allow_complete["value"] else None

    def process_factory(*args, **kwargs):
        process = GateProcess(0)
        processes.append(process)
        return process

    driver.execute_campaign(
        (_many_run_plan(tmp_path, 20),),
        scheduler=scheduler,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        peer_status_store=_StatusStore(),
        process_factory=process_factory,
        sleep=lambda _: allow_complete.update(value=True),
    )
    assert len(processes) == 20
    assert scheduler.max_active == 16
    assert scheduler.admitted_slots[:16] == [
        (f"GPU-{gpu}", slot) for slot in range(4) for gpu in range(4)
    ]
    assert set(scheduler.admitted_slots[:16]) == {
        (f"GPU-{gpu}", slot) for gpu in range(4) for slot in range(4)
    }
    assert scheduler.active == {}


def test_execute_full_with_runtime_guard_limits_two_workers_total(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver, "validate_worker_completion", lambda run: {})
    scheduler = _FourGpuSlotScheduler()
    allow_complete = {"value": False}
    processes = []

    class GateProcess(_FakeProcess):
        def poll(self):
            return 0 if allow_complete["value"] else None

    class Guard:
        max_seen = 0

        def enforce(self, processes_by_worker):
            self.max_seen = max(self.max_seen, len(processes_by_worker))

    guard = Guard()

    def process_factory(*args, **kwargs):
        process = GateProcess(0)
        processes.append(process)
        return process

    driver.execute_campaign(
        (_full_shard_plan(tmp_path, 8),),
        scheduler=scheduler,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        peer_status_store=_StatusStore(),
        process_factory=process_factory,
        runtime_guard=guard,
        sleep=lambda _: allow_complete.update(value=True),
    )
    assert len(processes) == 8
    assert scheduler.max_active == 2
    assert guard.max_seen == 2
    assert scheduler.active == {}


def test_execute_spreads_twelve_workers_three_per_gpu(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver, "validate_worker_completion", lambda run: {})
    scheduler = _FourGpuSlotScheduler()
    allow_complete = {"value": False}

    class GateProcess(_FakeProcess):
        def poll(self):
            return 0 if allow_complete["value"] else None

    driver.execute_campaign(
        (_many_run_plan(tmp_path, 12),),
        scheduler=scheduler,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        peer_status_store=_StatusStore(),
        process_factory=lambda *args, **kwargs: GateProcess(0),
        sleep=lambda _: allow_complete.update(value=True),
    )
    assert scheduler.max_active == 12
    assert scheduler.admitted_slots == [
        (f"GPU-{gpu}", slot) for slot in range(3) for gpu in range(4)
    ]
    assert scheduler.active == {}


def test_execute_round_robin_respects_heterogeneous_gpu_capacity(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver, "validate_worker_completion", lambda run: {})
    scheduler = _FourGpuSlotScheduler(
        {"GPU-0": 1, "GPU-1": 2, "GPU-2": 3, "GPU-3": 4}
    )
    allow_complete = {"value": False}

    class GateProcess(_FakeProcess):
        def poll(self):
            return 0 if allow_complete["value"] else None

    driver.execute_campaign(
        (_many_run_plan(tmp_path, 10),),
        scheduler=scheduler,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        peer_status_store=_StatusStore(),
        process_factory=lambda *args, **kwargs: GateProcess(0),
        sleep=lambda _: allow_complete.update(value=True),
    )
    assert scheduler.max_active == 10
    assert scheduler.admitted_slots == [
        ("GPU-0", 0),
        ("GPU-1", 0),
        ("GPU-2", 0),
        ("GPU-3", 0),
        ("GPU-1", 1),
        ("GPU-2", 1),
        ("GPU-2", 2),
        ("GPU-3", 1),
        ("GPU-3", 2),
        ("GPU-3", 3),
    ]


def test_admission_skips_busy_and_locked_gpus_in_round_robin_order() -> None:
    class BusyAndLockedScheduler:
        resource_contract_sha256 = "a" * 64

        def __init__(self) -> None:
            self.requests = []

        def admit_worker(self, request):
            self.requests.append(request)
            status = {
                "GPU-1": driver.AdmissionStatus.LOCK_CONTENTION,
                "GPU-2": driver.AdmissionStatus.GPU_LIMIT,
                "GPU-3": driver.AdmissionStatus.ADMITTED,
            }[request.expected_gpu_uuid]
            lease = None
            if status is driver.AdmissionStatus.ADMITTED:
                lease = SimpleNamespace(gpu_uuid=request.expected_gpu_uuid, slot_index=0)
            return SimpleNamespace(status=status, lease=lease, incumbent=None)

    scheduler = BusyAndLockedScheduler()
    lease = driver._admit_worker(
        scheduler,
        worker_id="worker-0",
        launch_ordinal=10_000,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        ram_slot_budget_bytes=1_100_000_000,
        start_gpu_index=1,
    )
    assert lease.gpu_uuid == "GPU-3"
    assert [request.gpu_index for request in scheduler.requests] == [1, 2, 3]
    assert driver._next_gpu_index(
        {index: f"GPU-{index}" for index in range(4)}, lease.gpu_uuid
    ) == 0


@pytest.mark.parametrize(
    "status",
    [driver.AdmissionStatus.RESUMED, driver.AdmissionStatus.RECLAIMED],
)
def test_round_robin_accepts_bound_resume_and_reclaimed_leases(status) -> None:
    class ResumeScheduler:
        resource_contract_sha256 = "a" * 64

        def admit_worker(self, request):
            return SimpleNamespace(
                status=status,
                lease=SimpleNamespace(
                    gpu_uuid=request.expected_gpu_uuid,
                    slot_index=2,
                ),
                incumbent=None,
            )

    lease = driver._admit_worker(
        ResumeScheduler(),
        worker_id="worker-resume",
        launch_ordinal=10_000,
        gpu_bindings={index: f"GPU-{index}" for index in range(4)},
        ram_slot_budget_bytes=1_100_000_000,
        start_gpu_index=2,
    )
    assert lease.gpu_uuid == "GPU-2"
    assert driver._next_gpu_index(
        {index: f"GPU-{index}" for index in range(4)}, lease.gpu_uuid
    ) == 3


def test_round_robin_rejects_success_lease_with_wrong_gpu_uuid() -> None:
    class WrongUuidScheduler:
        resource_contract_sha256 = "a" * 64

        def admit_worker(self, request):
            return SimpleNamespace(
                status=driver.AdmissionStatus.ADMITTED,
                lease=SimpleNamespace(gpu_uuid="GPU-wrong", slot_index=0),
                incumbent=None,
            )

    with pytest.raises(driver.ResourceContractError, match="outside its request"):
        driver._admit_worker(
            WrongUuidScheduler(),
            worker_id="worker-wrong-uuid",
            launch_ordinal=10_000,
            gpu_bindings={index: f"GPU-{index}" for index in range(4)},
            ram_slot_budget_bytes=1_100_000_000,
            start_gpu_index=0,
        )


def test_peer_failure_terminates_and_releases_all_other_live_workers(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver, "validate_worker_completion", lambda run: {})
    scheduler = _FourGpuSlotScheduler()
    processes = []

    def process_factory(*args, **kwargs):
        process = _FakeProcess(
            7 if not processes else 0,
            running_polls=0 if not processes else 100,
        )
        processes.append(process)
        return process

    with pytest.raises(driver.CampaignFailedError, match="peer_failure"):
        driver.execute_campaign(
            (_many_run_plan(tmp_path, 4),),
            scheduler=scheduler,
            gpu_bindings={index: f"GPU-{index}" for index in range(4)},
            peer_status_store=_StatusStore(),
            process_factory=process_factory,
            sleep=lambda _: None,
        )
    assert all(process.terminated for process in processes[1:])
    assert scheduler.admitted_slots == [
        ("GPU-0", 0),
        ("GPU-1", 0),
        ("GPU-2", 0),
        ("GPU-3", 0),
    ]
    assert scheduler.active == {}


def test_launch_ordinals_are_stable_across_phase_invocation_modes() -> None:
    assert [driver._stable_launch_ordinal("diagnose", index) for index in range(3)] == [
        10_000,
        10_001,
        10_002,
    ]
    assert driver._stable_launch_ordinal("calibrate", 0) == 20_000
    assert driver._stable_launch_ordinal("confirm512", 0) == 30_000
    assert driver._stable_launch_ordinal("full", 0) == 40_000


def test_full_admission_ignores_temperatures_for_unselected_gpus() -> None:
    class Probe:
        @staticmethod
        def ram_snapshot():
            return SimpleNamespace(used_bytes=10, total_bytes=100)

        @staticmethod
        def gpu_snapshots():
            return tuple(
                SimpleNamespace(
                    index=index,
                    uuid=f"GPU-{index}",
                    total_bytes=4 * 1024**3,
                    free_bytes=3 * 1024**3,
                )
                for index in range(4)
            )

    admission = driver._full_admission_preflight(
        resource_probe=Probe(),
        compute_apps=(),
        temperatures={
            "GPU-0": 40,
            "GPU-1": 40,
            "GPU-2": 40,
            "GPU-3": 40,
            "GPU-4": 95,
        },
        disk_usage=SimpleNamespace(used=10, total=100),
        swap_io_delta=(0, 0),
    )

    assert admission["temperatures_c"] == {
        "GPU-0": 40,
        "GPU-1": 40,
        "GPU-2": 40,
        "GPU-3": 40,
    }


def test_full_admission_external_pid_baseline_requires_explicit_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Probe:
        @staticmethod
        def ram_snapshot():
            return SimpleNamespace(used_bytes=10, total_bytes=100)

        @staticmethod
        def gpu_snapshots():
            return tuple(
                SimpleNamespace(
                    index=index,
                    uuid=f"GPU-{index}",
                    total_bytes=4 * 1024**3,
                    free_bytes=3 * 1024**3,
                )
                for index in range(4)
            )

    common = {
        "resource_probe": Probe(),
        "compute_apps": (("GPU-3", 4242),),
        "temperatures": {f"GPU-{index}": 40 for index in range(4)},
        "disk_usage": SimpleNamespace(used=10, total=100),
        "swap_io_delta": (0, 0),
    }
    monkeypatch.delenv(driver.FULL_ADMISSION_EXTERNAL_PID_BASELINE_ENV, raising=False)
    with pytest.raises(driver.ResourceContractError, match="unknown GPU compute PIDs"):
        driver._full_admission_preflight(**common)

    monkeypatch.setenv(driver.FULL_ADMISSION_EXTERNAL_PID_BASELINE_ENV, "1")
    monkeypatch.setattr(
        driver,
        "_gpu_pid_baseline_row",
        lambda uuid, pid: {
            "gpu_uuid": uuid,
            "pid": pid,
            "start_time_ticks": 12345,
            "user": "guoxin",
            "command": "python train_model.py",
        },
    )
    admission = driver._full_admission_preflight(**common)
    assert admission["unknown_compute_pid_count"] == 1
    assert admission["external_compute_pid_baseline"] == [
        {
            "gpu_uuid": "GPU-3",
            "pid": 4242,
            "start_time_ticks": 12345,
            "user": "guoxin",
            "command": "python train_model.py",
        }
    ]
    assert admission["external_compute_pid_policy"] == {
        "schema_version": 1,
        "mode": "user_authorized_preexisting_gpu_pid_baseline_v1",
        "authorization_env": driver.FULL_ADMISSION_EXTERNAL_PID_BASELINE_ENV,
        "new_unknown_gpu_pids": "forbidden_after_admission",
        "resource_hard_stops": "unchanged",
    }
    assert driver._validate_admission_external_pid_policy(admission) is True


def test_full_runtime_guard_tracks_external_pid_baseline_and_rejects_new_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)

    class Probe:
        @staticmethod
        def ram_snapshot():
            return SimpleNamespace(used_bytes=10, total_bytes=100)

        @staticmethod
        def gpu_snapshots():
            return tuple(
                SimpleNamespace(
                    index=index,
                    uuid=f"GPU-{index}",
                    total_bytes=100,
                    free_bytes=90,
                )
                for index in range(4)
            )

    policy = {
        "policy_id": "frozen_conservative_e2e_v1",
        "gpu_indices": [0, 1, 2, 3],
        "hard_stop": {
            "gpu_memory_percent_at_or_above": 90,
            "ram_percent_at_or_above": 90,
            "disk_percent_at_or_above": 90,
            "cpu_percent_at_or_above": 90,
            "temperature_c_above": 85,
            "swap_io_positive": True,
            "sustained_sample_count": 2,
        },
    }
    cpu_values = iter(((100, 50), (200, 100), (300, 150)))
    monkeypatch.setattr(
        driver,
        "_pid_start_time_ticks",
        lambda pid: {4242: 12345, 5252: 23456}[pid],
    )
    baseline = [
        {
            "gpu_uuid": "GPU-3",
            "pid": 4242,
            "start_time_ticks": 12345,
            "user": "guoxin",
            "command": "python train_model.py",
        }
    ]
    guard = driver.FullRuntimeGuard(
        policy,
        monitor_path=tmp_path / "guard.jsonl",
        probe=Probe(),
        temperatures=lambda: {f"GPU-{index}": 40 for index in range(4)},
        swap_reader=lambda: (0, 0),
        disk_usage=lambda _: SimpleNamespace(used=10, total=100),
        gpu_process_memory=lambda: {},
        gpu_compute_apps=lambda: (("GPU-3", 4242),),
        cpu_reader=lambda: next(cpu_values),
        allowed_external_gpu_pids=baseline,
    )
    sample = guard.enforce()
    assert sample["external_compute_pid_baseline"] == baseline

    rejecting_cpu_values = iter(((100, 50), (200, 100)))
    rejecting = driver.FullRuntimeGuard(
        policy,
        monitor_path=tmp_path / "rejecting.jsonl",
        probe=Probe(),
        temperatures=lambda: {f"GPU-{index}": 40 for index in range(4)},
        swap_reader=lambda: (0, 0),
        disk_usage=lambda _: SimpleNamespace(used=10, total=100),
        gpu_process_memory=lambda: {},
        gpu_compute_apps=lambda: (("GPU-3", 5252),),
        cpu_reader=lambda: next(rejecting_cpu_values),
        allowed_external_gpu_pids=baseline,
    )
    with pytest.raises(driver.ResourceContractError, match="unknown GPU compute PIDs"):
        rejecting.enforce()


def test_full_runtime_guard_records_authorized_external_pid_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)

    class Probe:
        @staticmethod
        def ram_snapshot():
            return SimpleNamespace(used_bytes=10, total_bytes=100)

        @staticmethod
        def gpu_snapshots():
            return tuple(
                SimpleNamespace(
                    index=index,
                    uuid=f"GPU-{index}",
                    total_bytes=100,
                    free_bytes=90,
                )
                for index in range(4)
            )

    policy = {
        "policy_id": "frozen_conservative_e2e_v1",
        "gpu_indices": [0, 1, 2, 3],
        "hard_stop": {
            "gpu_memory_percent_at_or_above": 90,
            "ram_percent_at_or_above": 90,
            "disk_percent_at_or_above": 90,
            "cpu_percent_at_or_above": 90,
            "temperature_c_above": 85,
            "swap_io_positive": True,
            "sustained_sample_count": 2,
        },
    }
    monkeypatch.setattr(driver, "_pid_start_time_ticks", lambda pid: 777)
    monkeypatch.setattr(
        driver,
        "_gpu_pid_baseline_row",
        lambda uuid, pid: {
            "gpu_uuid": uuid,
            "pid": pid,
            "start_time_ticks": 777,
            "user": "guoxin",
            "command": "python train_model.py",
        },
    )
    monkeypatch.setenv(driver.FULL_RUNTIME_EXTERNAL_PID_DRIFT_ENV, "1")
    cpu_values = iter(((100, 50), (200, 100)))
    guard = driver.FullRuntimeGuard(
        policy,
        monitor_path=tmp_path / "drift.jsonl",
        probe=Probe(),
        temperatures=lambda: {f"GPU-{index}": 40 for index in range(4)},
        swap_reader=lambda: (0, 0),
        disk_usage=lambda _: SimpleNamespace(used=10, total=100),
        gpu_process_memory=lambda: {},
        gpu_compute_apps=lambda: (("GPU-2", 81567),),
        cpu_reader=lambda: next(cpu_values),
        allowed_external_gpu_pids=(),
    )

    sample = guard.enforce()

    assert sample["external_compute_pid_drift"] == [
        {
            "gpu_uuid": "GPU-2",
            "pid": 81567,
            "start_time_ticks": 777,
            "user": "guoxin",
            "command": "python train_model.py",
        }
    ]


def test_full_runtime_guard_hard_stops_on_sustained_host_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)

    class Probe:
        @staticmethod
        def ram_snapshot():
            return SimpleNamespace(used_bytes=10, total_bytes=100)

        @staticmethod
        def gpu_snapshots():
            return tuple(
                SimpleNamespace(
                    index=index,
                    uuid=f"GPU-{index}",
                    total_bytes=100,
                    free_bytes=90,
                )
                for index in range(4)
            )

    cpu_values = iter(((100, 50), (200, 60), (300, 70)))
    policy = {
        "policy_id": "frozen_conservative_e2e_v1",
        "gpu_indices": [0, 1, 2, 3],
        "hard_stop": {
            "gpu_memory_percent_at_or_above": 90,
            "ram_percent_at_or_above": 90,
            "disk_percent_at_or_above": 90,
            "cpu_percent_at_or_above": 90,
            "temperature_c_above": 85,
            "swap_io_positive": True,
            "sustained_sample_count": 2,
        },
    }
    guard = driver.FullRuntimeGuard(
        policy,
        monitor_path=tmp_path / "monitor.jsonl",
        probe=Probe(),
        temperatures=lambda: {f"GPU-{index}": 40 for index in range(4)},
        swap_reader=lambda: (0, 0),
        disk_usage=lambda _: SimpleNamespace(used=10, total=100),
        gpu_process_memory=lambda: {},
        gpu_compute_apps=lambda: (),
        cpu_reader=lambda: next(cpu_values),
    )
    guard.enforce()
    with pytest.raises(driver.ResourceContractError, match="host CPU"):
        guard.enforce()


def test_full_runtime_guard_fails_when_bound_monitor_dies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    policy = {
        "policy_id": "frozen_conservative_e2e_v1",
        "gpu_indices": [0, 1, 2, 3],
        "hard_stop": {
            "gpu_memory_percent_at_or_above": 90,
            "ram_percent_at_or_above": 90,
            "disk_percent_at_or_above": 90,
            "cpu_percent_at_or_above": 90,
            "temperature_c_above": 85,
            "swap_io_positive": True,
            "sustained_sample_count": 2,
        },
    }
    guard = driver.FullRuntimeGuard(
        policy,
        monitor_path=tmp_path / "guard.jsonl",
        swap_reader=lambda: (0, 0),
        gpu_compute_apps=lambda: (),
        cpu_reader=lambda: (100, 50),
    )
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_formal_full_monitor_claim_v1",
    }
    claim["monitor_claim_sha256"] = driver._canonical_json_sha256(claim)
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    guard.bind_monitor(
        session_name="safa-r9-v9-formal-full-monitor",
        claim_path=claim_path,
        claim_sha256=claim["monitor_claim_sha256"],
    )
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(driver.ResourceContractError, match="monitor tmux session died"):
        guard.enforce()


def test_formal_monitor_start_requires_live_tmux_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    outcomes = iter((1, 0, 1))
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=next(outcomes), stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="exited before first sample"):
        driver._start_formal_full_monitor(
            {"campaign_root": "campaign"},
            {
                "session_name": "safa-r9-v9-formal-full-monitor",
                "command": ["python", "monitor.py"],
            },
        )


def test_execute_campaign_runtime_guard_failure_terminates_owned_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    scheduler = _FourGpuSlotScheduler()
    process = _FakeProcess(0, running_polls=100)

    class Guard:
        @staticmethod
        def enforce(processes):
            assert processes
            raise driver.ResourceContractError("guard hard stop")

    with pytest.raises(driver.ResourceContractError, match="guard hard stop"):
        driver.execute_campaign(
            (_many_run_plan(tmp_path, 1),),
            scheduler=scheduler,
            gpu_bindings={index: f"GPU-{index}" for index in range(4)},
            peer_status_store=_StatusStore(),
            process_factory=lambda *args, **kwargs: process,
            runtime_guard=Guard(),
            sleep=lambda _: None,
        )
    assert process.terminated is True
    assert scheduler.active == {}


def test_formal_report_only_resume_executes_no_worker_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    order = []
    chain = {
        "claim": {"formal_execution_claim_sha256": "1" * 64},
        "terminal": {
            "status": "awaiting_visual_review",
            "formal_execution_terminal_sha256": "2" * 64,
        },
    }
    monkeypatch.setattr(
        driver,
        "_validate_formal_full_execution_chain",
        lambda campaign_runtime: order.append("validate") or chain,
    )
    monkeypatch.setattr(
        driver,
        "_require_full_e2e_gate",
        lambda campaign_runtime: order.append("gate") or {},
    )
    monkeypatch.setattr(
        driver,
        "build_phase_results_request",
        lambda *args, **kwargs: order.append("request") or {"request": True},
    )
    monkeypatch.setattr(
        driver,
        "resume_phase_results",
        lambda request: order.append("resume")
        or SimpleNamespace(status="complete"),
    )
    monkeypatch.setattr(
        driver,
        "finalize_phase_gate",
        lambda *args, **kwargs: order.append("finalize")
        or {"gate_contract_sha256": "3" * 64},
    )
    monkeypatch.setattr(
        driver,
        "execute_campaign",
        lambda *args, **kwargs: pytest.fail("generation callback executed"),
    )
    monkeypatch.setattr(
        driver,
        "execute_dynamic_campaign",
        lambda *args, **kwargs: pytest.fail("dynamic callback executed"),
    )
    writes = []
    monkeypatch.setattr(
        driver,
        "_write_immutable_bytes",
        lambda path, content: writes.append((path, content)),
    )
    campaign_runtime = {
        "campaign_root": "campaign",
        "campaign_runtime_sha256": "4" * 64,
    }
    plan = driver.PhasePlan(
        phase="full",
        campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        campaign_root=Path("campaign"),
        logical_run_count=0,
        runs=(),
    )
    assert (
        driver._resume_formal_full_report_only(
            {},
            campaign_runtime,
            {},
            {},
            plan=plan,
            campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        )
        == 0
    )
    assert order == ["validate", "gate", "request", "resume", "finalize"]
    assert len(writes) == 1
    assert writes[0][0] == tmp_path / "campaign/full/report_only_finalize.json"
    report = json.loads(writes[0][1])
    assert report["generation_execution_count"] == 0
    assert report["evaluator_execution_count"] == 0
    assert report["heldout_execution_count"] == 0


def _minimal_evaluator_callbacks(tmp_path: Path):
    callbacks = object.__new__(driver.R9ProductionEvaluatorCallbacks)
    callbacks._validate_current_worker_contract = lambda: None
    callbacks._campaign_root = tmp_path / "campaign"
    callbacks._evaluation = {
        "heldout": {"batch_size": 16},
        "arcface": {},
        "quality": {"script": {}},
    }
    callbacks._worker_contract = {"sha256": "1" * 64}
    callbacks._arcface_contract_sha256 = driver._canonical_json_sha256({})
    callbacks._quality_script_sha256 = "2" * 64
    callbacks._evaluator_ram_slot_budgets = {
        "quality": 1024,
        "arcface": 1024,
    }
    class RecordingScheduler(_FourGpuSlotScheduler):
        def __init__(self):
            super().__init__()
            self.requests = []

        def admit_worker(self, request):
            self.requests.append(request)
            return super().admit_worker(request)

    callbacks._scheduler = RecordingScheduler()
    callbacks._gpu_bindings = {
        index: f"GPU-{index}" for index in range(4)
    }
    callbacks._peer_status_store = _StatusStore()
    callbacks._python = sys.executable
    callbacks._worker_script = tmp_path / "worker.py"
    callbacks._sleep = lambda _: None
    callbacks._poll_interval_seconds = 0.01
    callbacks._launch_counter = 0
    callbacks._scheduler_lock = driver.threading.RLock()
    callbacks._quality_execution_lock = driver.threading.Lock()
    callbacks._active_evaluator_processes = {}
    callbacks._runtime_guard = None
    callbacks._rss_sampler = lambda _: 1
    callbacks._gpu_process_memory = lambda: {"GPU-0": 1}

    class Process:
        pid = 1234
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def process_factory(command, **kwargs):
        del kwargs
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output = {
            "schema_version": 1,
            "contract_type": "safa_r9_phase_evaluator_output_v1",
            "task": request["task"],
            "evaluator_request_sha256": request["evaluator_request_sha256"],
            "worker_contract": callbacks._worker_contract,
            "arcface_contract_sha256": callbacks._arcface_contract_sha256,
            "quality_script_sha256": callbacks._quality_script_sha256,
            "result": {"strict": True},
        }
        output["evaluator_output_sha256"] = driver._canonical_json_sha256(output)
        output_path.write_text(json.dumps(output), encoding="utf-8")
        return Process()

    callbacks._process_factory = process_factory
    return callbacks


def test_evaluator_runtime_guard_receives_all_active_evaluator_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    callbacks = _minimal_evaluator_callbacks(tmp_path)
    peer = SimpleNamespace(pid=1111, poll=lambda: None)
    callbacks._active_evaluator_processes[
        "evaluator:arcface:full:winner"
    ] = peer
    observed_worker_sets = []

    class Guard:
        @staticmethod
        def enforce(processes):
            assert callbacks._scheduler_lock._is_owned()
            observed_worker_sets.append(set(processes))

    class Process:
        pid = 2222
        returncode = 0

        def __init__(self):
            self._poll_count = 0
            self.terminated = False

        def poll(self):
            self._poll_count += 1
            return None if self._poll_count == 1 else 0

    def process_factory(command, **kwargs):
        del kwargs
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output = {
            "schema_version": 1,
            "contract_type": "safa_r9_phase_evaluator_output_v1",
            "task": request["task"],
            "evaluator_request_sha256": request["evaluator_request_sha256"],
            "worker_contract": callbacks._worker_contract,
            "arcface_contract_sha256": callbacks._arcface_contract_sha256,
            "quality_script_sha256": callbacks._quality_script_sha256,
            "result": {"strict": True},
        }
        output["evaluator_output_sha256"] = driver._canonical_json_sha256(output)
        output_path.write_text(json.dumps(output), encoding="utf-8")
        return Process()

    callbacks._runtime_guard = Guard()
    callbacks._process_factory = process_factory

    assert callbacks._run("quality", "full", "winner__candidate", {}) == {
        "strict": True
    }
    assert observed_worker_sets == [
        {
            "evaluator:arcface:full:winner",
            "evaluator:quality:full:winner__candidate",
        }
    ]


def test_heldout_uses_one_lease_and_attempt_is_single_use(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    callbacks = _minimal_evaluator_callbacks(tmp_path)
    payload = {"winner": "paper_eta_0p125"}
    assert callbacks._run("heldout", "full", "winner", payload) == {
        "strict": True
    }
    assert len(callbacks._scheduler.requests) == 1
    assert (
        callbacks._scheduler.requests[0].ram_slot_budget_bytes
        == 16 * 1024**3
    )
    assert callbacks._scheduler.active_leases == ()
    assert callbacks._run("heldout", "full", "winner", payload) == {
        "strict": True
    }
    assert len(callbacks._scheduler.requests) == 1

    result_path = (
        tmp_path
        / "campaign/full/evaluator_runs/heldout/winner/result.json"
    )
    result_path.unlink()
    with pytest.raises(RuntimeError, match="automatic retry is forbidden"):
        callbacks._run("heldout", "full", "winner", payload)


def test_quality_callbacks_are_process_wide_mutually_exclusive(
    tmp_path: Path,
) -> None:
    callbacks = _minimal_evaluator_callbacks(tmp_path)
    active = 0
    maximum = 0
    active_lock = driver.threading.Lock()
    release = driver.threading.Event()
    entered = driver.threading.Event()

    def run(*args):
        nonlocal active, maximum
        del args
        with active_lock:
            active += 1
            maximum = max(maximum, active)
            entered.set()
        release.wait(timeout=2)
        with active_lock:
            active -= 1
        return {"strict": True}

    callbacks._run = run
    request = SimpleNamespace(
        phase="full",
        logical_run_id="winner",
        arm_id="paper_eta_0p125",
        seed=7919,
        image_role="candidate",
        manifest_path=tmp_path / "manifest.jsonl",
        source_index_path=tmp_path / "source.jsonl",
        source_index_sha256="1" * 64,
        samples=(),
        algorithm_config_sha256="2" * 64,
        runner_arm_config_sha256="3" * 64,
        semantic_output_sha256="4" * 64,
        evidence_binding_sha256="5" * 64,
        generation_result_set_sha256="6" * 64,
        per_sample_set_sha256="7" * 64,
    )
    threads = [
        driver.threading.Thread(target=callbacks.quality, args=(request,))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=1)
    release.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert maximum == 1


def test_full_e2e_measured_resource_profile_rebuild_and_tamper_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    worker_contract = {
        "path": "scripts/worker.py",
        "sha256": "1" * 64,
        "implementation_path": "src/worker.py",
        "implementation_sha256": "2" * 64,
    }
    (tmp_path / worker_contract["path"]).parent.mkdir(parents=True)
    (tmp_path / worker_contract["path"]).write_text("worker wrapper", encoding="utf-8")
    (tmp_path / worker_contract["implementation_path"]).parent.mkdir(parents=True)
    (tmp_path / worker_contract["implementation_path"]).write_text(
        "worker implementation", encoding="utf-8"
    )
    arcface = {"weights_sha256": "3" * 64}
    quality_sha256 = "4" * 64
    campaign_runtime = {
        "campaign_root": "campaign",
        "evaluation": {
            "worker": worker_contract,
            "arcface": arcface,
            "quality": {"script": {"sha256": quality_sha256}},
        },
    }
    expected_worker_contract = {
        "path": str((tmp_path / worker_contract["path"]).resolve()),
        "sha256": worker_contract["sha256"],
        "implementation_path": str(
            (tmp_path / worker_contract["implementation_path"]).resolve()
        ),
        "implementation_sha256": worker_contract["implementation_sha256"],
    }
    expected_arcface_sha256 = driver._canonical_json_sha256(arcface)
    monkeypatch.setattr(
        driver,
        "_arcface_evaluation_contract_sha256",
        lambda evaluation: expected_arcface_sha256,
    )
    units = {
        "arcface": ("arcface", "formal_e2e_arcface_8", 100, 10),
        "quality_native": (
            "quality",
            "formal_e2e_quality_8__native",
            200,
            20,
        ),
        "quality_candidate": (
            "quality",
            "formal_e2e_quality_8__candidate",
            300,
            30,
        ),
    }
    for task, unit, peak_rss, peak_gpu in units.values():
        unit_root = tmp_path / "campaign/full_e2e/evaluator_runs" / task / unit
        unit_root.mkdir(parents=True)
        request = {
            "evaluator_request_sha256": hashlib.sha256(unit.encode()).hexdigest()
        }
        result = {
            "evaluator_output_sha256": hashlib.sha256(task.encode()).hexdigest()
        }
        observation = {
            "schema_version": 1,
            "contract_type": (
                "safa_r9_full_e2e_evaluator_resource_observation_v1"
            ),
            "task": task,
            "unit_id": unit,
            "evaluator_request_sha256": request["evaluator_request_sha256"],
            "evaluator_output_sha256": result["evaluator_output_sha256"],
            "worker_contract": expected_worker_contract,
            "arcface_contract_sha256": expected_arcface_sha256,
            "quality_script_sha256": quality_sha256,
            "resource_policy_id": "frozen_conservative_e2e_v1",
            "peak_process_tree_rss_bytes": peak_rss,
            "peak_gpu_memory_bytes": peak_gpu,
            "gpu_uuid": "GPU-0",
        }
        observation["resource_observation_sha256"] = (
            driver._canonical_json_sha256(observation)
        )
        for name, payload in (
            ("request.json", request),
            ("result.json", result),
            ("resource_observation.json", observation),
        ):
            (unit_root / name).write_text(json.dumps(payload), encoding="utf-8")

    profiles = driver.build_full_e2e_resource_profiles(campaign_runtime)
    assert profiles["arcface"]["ram_slot_budget_bytes"] == 110
    assert profiles["quality"]["peak_process_tree_rss_bytes"] == 300
    assert profiles["quality"]["ram_slot_budget_bytes"] == 330
    profile_path = tmp_path / "campaign/full_e2e/resource_profiles.json"
    profile_path.write_text(json.dumps(profiles), encoding="utf-8")
    binding = {
        "path": str(profile_path.relative_to(tmp_path)),
        "file_sha256": driver._sha256_path(profile_path),
        "contract_sha256": profiles["resource_profiles_sha256"],
    }
    runtime_profiles = {
        key: profiles[key] for key in ("arcface", "quality", "heldout")
    }
    runtime_profiles.update(
        {
            "resource_profiles_sha256": profiles["resource_profiles_sha256"],
            "resource_profile_binding": binding,
        }
    )
    assert (
        driver._validate_full_e2e_runtime_resource_profiles(
            runtime_profiles,
            repo_root=tmp_path,
            worker_contract=expected_worker_contract,
            arcface_contract_sha256=expected_arcface_sha256,
            quality_script_sha256=quality_sha256,
        )
        == runtime_profiles
    )

    profile_path.write_text(json.dumps({**profiles, "source": "tampered"}))
    with pytest.raises(ValueError, match="does not bind"):
        driver._validate_full_e2e_runtime_resource_profiles(
            runtime_profiles,
            repo_root=tmp_path,
            worker_contract=expected_worker_contract,
            arcface_contract_sha256=expected_arcface_sha256,
            quality_script_sha256=quality_sha256,
        )
    profile_path.unlink()
    with pytest.raises(FileNotFoundError, match="resource profile"):
        driver._validate_full_e2e_runtime_resource_profiles(
            runtime_profiles,
            repo_root=tmp_path,
            worker_contract=worker_contract,
            arcface_contract_sha256=expected_arcface_sha256,
            quality_script_sha256=quality_sha256,
        )
