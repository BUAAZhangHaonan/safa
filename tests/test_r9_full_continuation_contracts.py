from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from safa.evaluation import r9_full_continuation_contracts as full
from safa.evaluation import r9_full_smoke_supersession as smoke
from safa.evaluation.r9_campaign_contracts import build_heldout_seal_contract
from safa.evaluation.r9_evaluator_worker import (
    R9EvaluatorError,
    _validate_heldout_contracts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/medium_v2/experiments/"
    "r9_meanflow_full_continuation_campaign_v9.yaml"
)


def _canonical(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source() -> dict:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["source"])


def test_full_contract_binds_frozen_v3_winner_and_batch2() -> None:
    contract = full.build_full_continuation_contract(
        repo_root=REPO_ROOT, expected_source=_source()
    )
    assert contract["contract_type"] == "safa_r9_full_continuation_v1"
    assert contract["start_phase"] == "full"
    assert contract["selected_arms"] == [
        {
            "arm_id": "paper_eta_0p125",
            "config_sha256": (
                "a087eaf6767ba51bddcc9012c309b30d6c6e86bf76d30672a5d202412bc0a77f"
            ),
            "output_sha256": (
                "2112eb04068403d7c4372ee7706339d8805b651c6abc6f1f12115a0807786729"
            ),
        }
    ]
    assert contract["policy"]["allowed_phases"] == ["full"]
    assert contract["policy"]["reselection_allowed"] is False
    assert contract["bindings"]["generation_batch_policy"] == {
        "batch_size": 2,
        "workers_per_gpu": 2,
        "physical_gpus": [0, 1, 2, 3],
        "batch4_equivalent": False,
        "source_campaign_id": "r9-report-only-formal-v8",
        "source_continuation_contract_sha256": (
            "2c95c4cbaa917742cf1a3694b0162ca5fb8d71ce0ca6b9952aded7adeac466ce"
        ),
    }
    source_evaluation = contract["bindings"]["source_evaluation_provenance"]
    current_evaluation = contract["bindings"]["current_evaluation"]
    assert source_evaluation["classification"] == (
        "historical_v8_runtime_provenance_not_execution_authority"
    )
    assert current_evaluation["classification"] == (
        "canonical_current_v9_execution_authority"
    )
    assert (
        source_evaluation["evaluation"]["worker"]["implementation_sha256"]
        != current_evaluation["worker"]["implementation_sha256"]
    )
    assert current_evaluation["current_evaluation_sha256"] == _canonical(
        current_evaluation, "current_evaluation_sha256"
    )
    historical = contract["bindings"]["evaluator_smoke_requests"]
    assert historical["classification"] == (
        "historical_invalid_smoke_provenance_only_not_a_full_resource_profile"
    )
    declared_historical_digest = historical["request_set_sha256"]
    assert declared_historical_digest == _canonical(
        historical, "request_set_sha256"
    )
    supersession = historical[
        "smoke_supersession"
    ]
    historical_smoke = json.loads(
        (REPO_ROOT / supersession["path"]).read_text(encoding="utf-8")
    )
    assert supersession["contract_sha256"] == (
        "ff8152dd8529bae94c0f81668477299fa9f03303fb22dae73c9d479217485df1"
    )
    assert historical_smoke["smoke_supersession_sha256"] == supersession[
        "contract_sha256"
    ]
    assert historical_smoke["worker"]["implementation_sha256"] == (
        "57e083fa6c910d268e2b90fc1aae4bd0d3a8143a67146822c2626bce7612cfd3"
    )
    assert historical_smoke["execution"]["v2_execution_count"] == 0
    current_smoke = smoke.build_full_smoke_supersession_contract(
        repo_root=REPO_ROOT
    )
    assert current_smoke["smoke_supersession_sha256"] == (
        "0b4108daef264addd84145a86e79bd5ea9aeaeba88f359edf4d00dd302763ccf"
    )
    assert current_smoke["worker"]["implementation_sha256"] == (
        "dd734b51c630a4d11e9d5b6dea7953d25471cdb71f682d77e791bab47b50c0dd"
    )
    assert (
        current_smoke["smoke_supersession_sha256"]
        != historical_smoke["smoke_supersession_sha256"]
    )
    failed_v1 = current_smoke["failed_v1"]
    assert "terminal_observation" in failed_v1
    assert failed_v1["expected_absent_outputs"] == {
        "arcface_worker_result": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v9/evaluator_smoke/arcface-full-v1/"
            "worker_result.json"
        ),
        "arcface_resource_result": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v9/evaluator_smoke/arcface-full-v1/"
            "resource_result.json"
        ),
        "quality_worker_result": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v9/evaluator_smoke/quality-full-v1/"
            "worker_result.json"
        ),
        "quality_resource_result": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v9/evaluator_smoke/quality-full-v1/"
            "resource_result.json"
        ),
    }
    assert contract["bindings"]["full_e2e_requirement"]["policy"] == {
        "phase": "full",
        "seed": 7919,
        "batch_size": 2,
        "arms": ["native", "paper_eta_0p125"],
        "evaluator_tasks": [
            "arcface",
            "quality_native",
            "quality_candidate",
        ],
        "resource_policy": {
            "policy_id": "frozen_conservative_e2e_v1",
            "claim_type": (
                "preregistered_exclusive_upper_bound_not_measured_profile"
            ),
            "source": "pre_execution_protocol_registration",
            "rationale": (
                "e2e_bootstrap_must_not_depend_on_or_reuse_a_prior_campaign_"
                "evaluator_profile"
            ),
            "gpu_indices": [0, 1, 2, 3],
            "generation": {
                "gpu_slot_claim_bytes": 17179869184,
                "ram_slot_budget_bytes": 17179869184,
                "max_slots_per_gpu": 2,
                "concurrent_workers": 2,
            },
            "evaluator": {
                "gpu_slot_claim_bytes": 17179869184,
                "ram_slot_budget_bytes": 17179869184,
                "global_exclusive": True,
                "concurrent_workers": 1,
            },
            "admission": {
                "minimum_free_gpu_bytes": 2147483648,
                "ram_percent_below": 85,
                "disk_percent_below": 85,
                "unknown_gpu_pid_count": 0,
                "initial_swap_io_pages": 0,
            },
            "hard_stop": {
                "gpu_memory_percent_at_or_above": 90,
                "ram_percent_at_or_above": 90,
                "disk_percent_at_or_above": 90,
                "cpu_percent_at_or_above": 90,
                "temperature_c_above": 85,
                "swap_io_positive": True,
                "sustained_sample_count": 2,
            },
        },
        "retry_count": 0,
    }
    assert "resource_profiles_sha256" not in historical


def test_full_contract_rejects_bad_frozen_sha_before_execution() -> None:
    source = _source()
    source["supersession_contract_sha256"] = "0" * 64
    with pytest.raises(full.FullContinuationContractError, match="supersession"):
        full.build_full_continuation_contract(
            repo_root=REPO_ROOT, expected_source=source
        )


def test_full_selection_reselection_is_rejected_by_heldout_worker(
    tmp_path: Path,
) -> None:
    winner = {
        "arm_id": "paper_eta_0p125",
        "config_sha256": "1" * 64,
        "output_sha256": "2" * 64,
    }
    selection = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_continuation_selection_v1",
        "winner": winner,
        "winner_locked": True,
        "reselection_allowed": False,
    }
    selection["selection_sha256"] = _canonical(selection, "selection_sha256")
    assets = {}
    for index, name in enumerate(("e1", "e2", "facenet", "adaface"), 1):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(bytes([index]))
        assets[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    seal = build_heldout_seal_contract(selection, assets)
    locked, _ = _validate_heldout_contracts(
        selection, seal, arm_id=winner["arm_id"], repo_root=tmp_path
    )
    assert locked["reselection_allowed"] is False
    selection["reselection_allowed"] = True
    selection["selection_sha256"] = _canonical(selection, "selection_sha256")
    with pytest.raises(R9EvaluatorError, match="winner-locked"):
        _validate_heldout_contracts(
            selection, seal, arm_id=winner["arm_id"], repo_root=tmp_path
        )


def test_missing_smoke_v2_supersession_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(full.FullContinuationContractError, match="not materialized"):
        full._load_evaluator_smoke_requests(tmp_path)


def test_smoke_v2_rejects_legacy_calibration_payload() -> None:
    _, requests = smoke.build_full_smoke_requests(repo_root=REPO_ROOT)
    request, _ = requests["arcface"]
    legacy = json.loads(json.dumps(request))
    legacy["payload"]["phase"] = "calibrate"
    legacy["payload"]["arm_id"] = "paper_split_eta0.25"
    sample_ids = [
        row["sample_id"]
        for row in json.loads(json.dumps(request))["payload"]["samples"]
    ]
    with pytest.raises(
        smoke.FullSmokeSupersessionError, match="rejects legacy"
    ):
        smoke._validate_v2_request(
            legacy, task="arcface", sample_ids=sample_ids
        )
