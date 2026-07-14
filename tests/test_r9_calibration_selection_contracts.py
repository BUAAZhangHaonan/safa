from __future__ import annotations

import copy
from pathlib import Path

import pytest

from safa.evaluation import r9_calibration_selection_contracts as contracts


def _arm(arm_id: str, family: str, marker: str) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "family": family,
        "config_sha256": marker * 64,
        "output_sha256": chr(ord(marker) + 1) * 64,
        "evaluator_evidence_sha256": chr(ord(marker) + 2) * 64,
    }


def _fixture_payloads() -> dict[str, dict[str, object]]:
    arms = [
        _arm("flow_map2_normalized_eta_0p125", "flow_map2", "1"),
        _arm("paper_eta_0p125", "paper_split_constant", "4"),
        _arm("paper_eta_0p25_disable_i2", "paper_split_interval_ablation", "7"),
    ]
    automatic_arms = []
    phase_arms = []
    gate_arms = []
    for arm in arms:
        automatic_arms.append(
            {
                key: arm[key] for key in ("arm_id", "family", "config_sha256")
            }
            | {
                "seed_results": [
                    {"seed": seed, "arcface_summary": {}} for seed in contracts.SEEDS
                ]
            }
        )
        phase_arms.append(dict(arm))
        gate_arms.append(
            {
                key: arm[key]
                for key in ("arm_id", "family", "config_sha256", "output_sha256")
            }
            | {
                "failures": [
                    "seed_2027:arcface_not_exactly_one_face_per_image"
                ],
                "seed_results": [
                    {"seed": seed, "severe_count": 0} for seed in contracts.SEEDS
                ],
            }
        )
    runtime = {
        "campaign_id": contracts.SOURCE_CAMPAIGN_ID,
        "campaign_runtime_sha256": contracts.SOURCE_RUNTIME_SHA256,
    }
    phase = {
        "phase": contracts.SOURCE_PHASE,
        "phase_results_sha256": contracts.SOURCE_PHASE_RESULTS_SHA256,
        "automatic_evidence_sha256": contracts.SOURCE_AUTOMATIC_EVIDENCE_SHA256,
        "run_plan_sha256": "a" * 64,
        "arms": phase_arms,
    }
    automatic = {
        "phase": contracts.SOURCE_PHASE,
        "automatic_evidence_sha256": contracts.SOURCE_AUTOMATIC_EVIDENCE_SHA256,
        "run_plan_sha256": "a" * 64,
        "arms": automatic_arms,
    }
    gate = {
        "phase": contracts.SOURCE_PHASE,
        "gate_contract_sha256": contracts.SOURCE_GATE_SHA256,
        "verdict": "stop_zero_candidates",
        "selected_arm_ids": [],
        "context": {
            "campaign_runtime_sha256": contracts.SOURCE_RUNTIME_SHA256,
            "phase_results_sha256": contracts.SOURCE_PHASE_RESULTS_SHA256,
            "automatic_evidence_sha256": (
                contracts.SOURCE_AUTOMATIC_EVIDENCE_SHA256
            ),
        },
        "arms": gate_arms,
    }
    repair = {
        "phase": contracts.SOURCE_PHASE,
        "repair_contract_sha256": contracts.SOURCE_REPAIR_SHA256,
        "generation_evidence": {
            "inventory_sha256": contracts.SOURCE_GENERATION_INVENTORY_SHA256,
            "logical_run_count": 12,
            "shard_count": 12,
            "completion_count": 12,
            "generation_result_count": 12,
            "file_count": 1440,
            "png_count": 1344,
        },
    }
    return {
        "campaign_runtime.json": runtime,
        "phase_results.json": phase,
        "automatic_evidence.json": automatic,
        "evaluation_repair_contract_v3.json": repair,
        "gate_contract.json": gate,
    }


@pytest.fixture()
def isolated_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, dict[str, object]]:
    payloads = _fixture_payloads()

    def read_contract(path: Path, field: str) -> dict[str, object]:
        del field
        return copy.deepcopy(payloads[path.name])

    monkeypatch.setattr(contracts, "_read_contract", read_contract)
    monkeypatch.setattr(
        contracts, "_read_json", lambda path: copy.deepcopy(payloads[path.name])
    )
    monkeypatch.setattr(contracts, "validate_gate_contract", lambda value: value)
    monkeypatch.setattr(
        contracts,
        "_binding",
        lambda root, path, digest: {
            "path": str(path.relative_to(root)),
            "file_sha256": "f" * 64,
            "contract_sha256": digest,
        },
    )

    def complete_case_rows(
        *, root: Path, arm_id: str, automatic_arm: object
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        del root, automatic_arm
        marker = {
            "flow_map2_normalized_eta_0p125": 1.0,
            "paper_eta_0p125": 2.0,
            "paper_eta_0p25_disable_i2": 3.0,
        }[arm_id]
        return ([{"marker": marker}] * 189, [{"arm_id": arm_id}])

    monkeypatch.setattr(contracts, "_complete_case_rows", complete_case_rows)

    def bootstrap(rows: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
        del kwargs
        marker = rows[0]["marker"]
        arm_id = {
            1.0: "flow_map2_normalized_eta_0p125",
            2.0: "paper_eta_0p125",
            3.0: "paper_eta_0p25_disable_i2",
        }[marker]
        return {
            "bootstrap_sha256": contracts.EXPECTED_BOOTSTRAP_SHA256.get(
                arm_id, "e" * 64
            ),
            "sample_count": 63,
            "observation_count": 189,
        }

    monkeypatch.setattr(contracts, "privacy_delta_cluster_bootstrap", bootstrap)
    return payloads


def test_build_selects_only_the_two_fixed_mainlines(
    isolated_builder: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    del isolated_builder
    result = contracts.build_calibration_report_only_selection_contract(
        repo_root=tmp_path
    )
    assert [row["arm_id"] for row in result["selected_arms"]] == list(
        contracts.SELECTED_ARM_IDS
    )
    assert result["verdict"] == "continue_to_confirm512"
    assert result["supersedes"] == {
        "scope": "promotion_decision_only",
        "original_verdict": "stop_zero_candidates",
        "original_selected_arm_ids": [],
        "original_gate_contract_sha256": contracts.SOURCE_GATE_SHA256,
    }
    assert result["coverage_report"]["sample_count"] == 63
    assert result["coverage_report"]["observation_count"] == 189


def test_noncoverage_failure_is_rejected(
    isolated_builder: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    gate = isolated_builder["gate_contract.json"]
    gate["arms"][0]["failures"] = ["seed_1337:nonfinite_metric"]
    with pytest.raises(
        contracts.CalibrationSelectionContractError,
        match="non-coverage Phase-B failure",
    ):
        contracts.build_calibration_report_only_selection_contract(
            repo_root=tmp_path
        )


def test_generation_inventory_tamper_is_rejected(
    isolated_builder: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    repair = isolated_builder["evaluation_repair_contract_v3.json"]
    repair["generation_evidence"]["png_count"] = 1343
    with pytest.raises(
        contracts.CalibrationSelectionContractError,
        match="generation inventory changed",
    ):
        contracts.build_calibration_report_only_selection_contract(
            repo_root=tmp_path
        )


def test_selection_digest_tamper_is_rejected(
    isolated_builder: dict[str, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_builder
    result = contracts.build_calibration_report_only_selection_contract(
        repo_root=tmp_path
    )
    monkeypatch.setattr(
        contracts,
        "build_calibration_report_only_selection_contract",
        lambda **kwargs: result,
    )
    tampered = copy.deepcopy(result)
    tampered["selected_arms"].reverse()
    with pytest.raises(
        contracts.CalibrationSelectionContractError,
        match="canonical digest mismatch",
    ):
        contracts.validate_calibration_report_only_selection_contract(
            tampered, repo_root=tmp_path
        )


def test_materialization_is_idempotent_but_never_overwrites(
    isolated_builder: dict[str, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_builder
    result = contracts.build_calibration_report_only_selection_contract(
        repo_root=tmp_path
    )
    monkeypatch.setattr(
        contracts,
        "build_calibration_report_only_selection_contract",
        lambda **kwargs: result,
    )
    first, binding = contracts.materialize_calibration_report_only_selection_contract(
        repo_root=tmp_path
    )
    second, second_binding = (
        contracts.materialize_calibration_report_only_selection_contract(
            repo_root=tmp_path
        )
    )
    assert first == second
    assert binding == second_binding
    path = tmp_path / binding["path"]
    path.write_text("different", encoding="utf-8")
    with pytest.raises(
        contracts.CalibrationSelectionContractError,
        match="already differs",
    ):
        contracts.materialize_calibration_report_only_selection_contract(
            repo_root=tmp_path
        )
