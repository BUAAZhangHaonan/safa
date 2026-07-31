from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
import yaml

from safa.evaluation.triangle32_evaluation import load_arm_set
from safa.evaluation.triangle_screening import ArmResult


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(arm_id: str, *, passed: bool = True, failed=(), score=0.2) -> ArmResult:
    return ArmResult(
        arm_id=arm_id,
        sample_count=32,
        candidate_exact_one_count=32,
        e0=0.8,
        delta_e0=0.4,
        delta_edev=0.1,
        niqe=4.0,
        native_niqe=4.0,
        fid=None,
        native_fid=None,
        kid=None,
        native_kid=None,
        sharpness=400.0,
        native_sharpness=400.0,
        arcface_delta=0.0,
        arcface_delta_u95=None,
        hard_gate_pass=passed,
        failed_gates=tuple(failed),
        r_margin=score,
        q_margin=score,
        p_margin=score,
    )


def matrix(u12_pass: bool, u16_pass: bool):
    return {
        "regular32": {
            "u12_regular32": result(
                "u12_regular32", passed=u12_pass, failed=() if u12_pass else ("niqe",)
            ),
            "u16_regular32": result(
                "u16_regular32", passed=u16_pass, failed=() if u16_pass else ("niqe",), score=0.3
            ),
        },
        "sharpness_tail32": {
            "u12_tail32": result(
                "u12_tail32", passed=u12_pass, failed=() if u12_pass else ("sharpness",)
            ),
            "u16_tail32": result(
                "u16_tail32", passed=u16_pass, failed=() if u16_pass else ("sharpness",), score=0.3
            ),
        },
    }


def test_r12_configs_lock_strict_seed_aligned_matrix() -> None:
    expected = {
        "u12_regular32": (12, "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"),
        "u16_regular32": (16, "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"),
        "u12_tail32": (12, "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl"),
        "u16_tail32": (16, "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl"),
    }
    for arm_id, (updates, manifest) in expected.items():
        config = yaml.safe_load(
            (
                ROOT
                / "configs/medium_v2/experiments"
                / f"r12_initial_noise_fixed_eta05_{arm_id}.yaml"
            ).read_text(encoding="utf-8")
        )
        assert config["experiment_contract"] == "safa_r9_meanflow_v1"
        assert config["attention_backend"] == "native"
        assert config["phase"] == "diagnose"
        assert config["seed"] == config["sampling_seed"] == 7919
        assert config["mode"] == "initial_noise"
        assert config["projection"] == "fixed_radius"
        assert config["eta"] == 0.5
        assert config["num_updates"] == updates
        assert config["sample_id_manifest"] == manifest
        assert config["batch_size"] == 2
        assert config["max_samples"] == 32
        assert config["quality_metrics"] == ["niqe", "sharpness"]


def test_prepare_writes_exact_four_card_ledgers_and_typed_contracts(tmp_path: Path) -> None:
    module = load_script("prepare_r12_seed_aligned_trajectory.py")
    output = tmp_path / "preparation"
    manifest = module.prepare(output)
    generation = json.loads(
        (output / "generation_ledger.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "prepared_not_launched"
    assert [job["physical_gpu"]["physical_index"] for job in generation["jobs"]] == [0, 1, 2, 3]
    assert [job["num_updates"] for job in generation["jobs"]] == [12, 16, 12, 16]
    assert [job["expected_candidate_nfe"] for job in generation["jobs"]] == [13, 17, 13, 17]
    assert all(job["retry_count"] == 0 for job in generation["jobs"])
    assert load_arm_set(output / "evaluation_contracts/regular32.json").arm_ids == (
        "u12_regular32",
        "u16_regular32",
    )
    assert load_arm_set(output / "evaluation_contracts/sharpness_tail32.json").arm_ids == (
        "u12_tail32",
        "u16_tail32",
    )
    evaluation = json.loads(
        (output / "evaluation_ledger.json").read_text(encoding="utf-8")
    )
    assert len(evaluation["quality_jobs"]) == 6
    assert len(evaluation["arcface_jobs"]) == 4
    assert all(job["metrics"] == ["niqe", "sharpness"] for job in evaluation["quality_jobs"])
    assert evaluation["fid_kid_interpretation"] == "forbidden"
    selection = json.loads((output / "selection_contract.json").read_text(encoding="utf-8"))
    assert [row["selection_eligible"] for row in selection["paired_horizons"]] == [True, True]
    assert selection["paired_horizons"][0]["update_budget_reduction_vs_u16"] == 0.25
    assert selection["paired_horizons"][0]["candidate_nfe_reduction_vs_u16"] == pytest.approx(
        4.0 / 17.0
    )
    assert selection["legacy_tail_scope"]["selection_basis"] == "full_image_laplacian_sharpness"
    assert selection["legacy_tail_scope"]["post_hoc_roi_tail_reselection"] == "forbidden"
    validator = load_script("validate_r12_seed_aligned_preparation.py")
    validation = validator.validate(output)
    assert validation["status"] == "validated_prepared_not_launched"
    assert validation["binding_counts"] == {"regular32": 32, "sharpness_tail32": 32}


def test_trajectory_prefix_binding_is_exact() -> None:
    module = load_script("classify_r12_seed_aligned_trajectory.py")
    u16 = {"loss_history": [float(index) for index in range(17)], "initial_norm": 64.0}
    u12 = {"loss_history": u16["loss_history"][:13], "initial_norm": 64.0}
    module.require_trajectory_prefix(u12, u16, "sample")
    bad = {**u12, "loss_history": [*u12["loss_history"][:-1], -1.0]}
    with pytest.raises(module.R12ClassificationError, match="protocol_binding_failure"):
        module.require_trajectory_prefix(bad, u16, "sample")


def test_paired_outcomes_and_horizon_selection_are_locked() -> None:
    module = load_script("classify_r12_seed_aligned_trajectory.py")
    early = module.outcome(matrix(True, False))
    assert early["outcome"] == "early_stop_quality_recovery"
    assert early["selected_horizon"] == "u12"
    full = module.outcome(matrix(False, True))
    assert full["outcome"] == "full_horizon_required"
    assert full["selected_horizon"] == "u16"
    both = module.outcome(matrix(True, True))
    assert both["outcome"] == "early_stop_not_needed_at32"
    assert both["selected_horizon"] == "u16"
    neither = module.outcome(matrix(False, False))
    assert neither["outcome"] == "initial_noise_quality_limited"
    assert neither["stop_required"] is True


def test_quality_recovery_label_requires_u16_quality_only_failure() -> None:
    module = load_script("classify_r12_seed_aligned_trajectory.py")
    values = matrix(True, False)
    values["regular32"]["u16_regular32"] = result(
        "u16_regular32", passed=False, failed=("candidate_exact_one",), score=0.3
    )
    values["sharpness_tail32"]["u16_tail32"] = result(
        "u16_tail32", passed=False, failed=("arcface_delta",), score=0.3
    )
    decision = module.outcome(values)
    assert decision["outcome"] == "early_stop_gate_recovery"
    assert decision["selected_horizon"] == "u12"


def test_tail_only_failure_never_promotes_partial_recovery() -> None:
    module = load_script("classify_r12_seed_aligned_trajectory.py")
    values = matrix(False, False)
    values["regular32"]["u12_regular32"] = result("u12_regular32")
    values["regular32"]["u16_regular32"] = result("u16_regular32")
    decision = module.outcome(values)
    assert decision["outcome"] == "tail_fragility"
    assert decision["advance_arm_ids"] == []
