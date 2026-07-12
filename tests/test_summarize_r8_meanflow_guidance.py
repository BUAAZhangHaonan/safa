from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d"
SAMPLE_DIGEST = "a" * 64
SCHEDULE_FILE_SHA256 = "d" * 64
SCHEDULE_CONTRACT_SHA256 = "e" * 64


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _generation(
    *,
    e0: float = 0.55,
    native_e0: float = 0.50,
    edev: float = 0.62,
    native_edev: float = 0.60,
) -> dict:
    return {
        "status": "complete",
        "mode": "official_head_current_xt",
        "checkpoint": {"sha256": CHECKPOINT_SHA256},
        "sample_count": 64,
        "sample_id_sha256": SAMPLE_DIGEST,
        "cosine": {
            "candidate_e0_target": {"mean": e0},
            "native_e0_target": {"mean": native_e0},
            "candidate_edev_source": {"mean": edev},
            "native_edev_source": {"mean": native_edev},
        },
        "native_sharpness": {"mean": 320.0},
        "nfe": {"candidate": 5, "matched_native": 1},
        "timing": {"images_per_second": 1.25, "wall_seconds": 51.2},
        "max_memory": {"allocated_bytes": 10_000, "reserved_bytes": 20_000},
        "schedule": {
            "manifest": "artifacts/r8_meanflow_flow_map_guidance/semigroup/locked_schedule_manifest.json",
            "manifest_sha256": SCHEDULE_FILE_SHA256,
            "schedule_contract_sha256": SCHEDULE_CONTRACT_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "t_cut": 0.25,
            "guided_times": [1.0, 0.75, 0.5, 0.25],
            "unguided_times": [0.25, 0.125, 0.0],
            "gate_passed": True,
        },
        "config": {
            "experiment_name": "arm",
            "sampling_seed": 1337,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "t_cut": 0.25,
            "schedule_manifest": "artifacts/r8_meanflow_flow_map_guidance/semigroup/locked_schedule_manifest.json",
            "heldout_e1_checkpoint": "metadata-only-e1.pt",
            "heldout_e2_checkpoint": "metadata-only-e2.pt",
        },
    }


def _quality(*, fid: float = 50.0, sharpness: float = 310.0) -> dict:
    return {
        "metrics": ["fid", "kid", "niqe", "sharpness"],
        "num_real": 64,
        "num_generated": 64,
        "sample_id_count": 64,
        "sample_id_sha256": SAMPLE_DIGEST,
        "fid": fid,
        "kid_mean": 0.03,
        "kid_std": 0.001,
        "iqa": {"method": "niqe", "mean": 5.5, "std": 0.2},
        "sharpness": {"mean": sharpness, "median": sharpness - 5.0},
    }


def _review(*, severe: int = 1) -> dict:
    return {
        "severe_failure_count": severe,
        "failures": [
            {"sample_id": f"sample-{index}", "category": "broken_global_structure"}
            for index in range(severe)
        ],
    }


def test_calibration_join_requires_exact_arm_and_contract() -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    row = module.validate_calibration_arm("arm-a", _generation(), _quality(), _review())

    assert row["arm_id"] == "arm-a"
    assert row["sample_count"] == 64
    assert row["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert row["e0_cosine"] == pytest.approx(0.55)
    assert row["native_e0_cosine"] == pytest.approx(0.50)
    assert row["edev_cosine"] == pytest.approx(0.62)
    assert row["native_edev_cosine"] == pytest.approx(0.60)
    assert row["sharpness_retention"] == pytest.approx(310 / 320)
    assert row["eligible"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("generation", "sample_count", 63), "sample count"),
        (("generation", "sample_id_sha256", "b" * 64), "sample-ID"),
        (("generation_checkpoint", "sha256", "b" * 64), "checkpoint"),
        (("generation_config", "sampling_seed", 99), "seed"),
        (("quality", "fid", float("nan")), "finite"),
    ],
)
def test_calibration_join_rejects_mismatched_or_nonfinite_contracts(mutation, message: str) -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    generation = _generation()
    quality = _quality()
    location, key, value = mutation
    if location == "generation":
        generation[key] = value
    elif location == "generation_checkpoint":
        generation["checkpoint"][key] = value
    elif location == "generation_config":
        generation["config"][key] = value
    else:
        quality[key] = value

    with pytest.raises(ValueError, match=message):
        module.validate_calibration_arm("arm-a", generation, quality, _review())


def test_calibration_rejects_heldout_outputs_but_allows_locked_metadata() -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    generation = _generation()
    assert module.validate_calibration_arm("arm", generation, _quality(), _review())["eligible"]

    generation["heldout_results"] = {"e1_cosine": 0.9}
    with pytest.raises(ValueError, match="held-out"):
        module.validate_calibration_arm("arm", generation, _quality(), _review())


def test_calibration_ignores_face_detection_and_treats_64_fid_as_diagnostic() -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    low_fid = module.validate_calibration_arm(
        "low-fid",
        _generation(e0=0.53, edev=0.61),
        {**_quality(fid=1.0), "face_detection_rate": 1.0},
        _review(),
    )
    high_e0 = module.validate_calibration_arm(
        "high-e0",
        _generation(e0=0.58, edev=0.62),
        {**_quality(fid=200.0), "face_detection_rate": 0.0},
        _review(),
    )

    selection = module.select_calibration_winner([low_fid, high_e0])

    assert selection["winner"]["arm_id"] == "high-e0"
    assert selection["fid_policy"] == "64-sample FID is diagnostic only"


@pytest.mark.parametrize(
    ("generation", "quality", "review", "reason"),
    [
        (_generation(e0=0.51), _quality(), _review(), "e0_delta"),
        (_generation(edev=0.59), _quality(), _review(), "edev_direction"),
        (_generation(), _quality(sharpness=250.0), _review(), "sharpness_retention"),
        (_generation(), _quality(), _review(severe=7), "visual_failure_rate"),
    ],
)
def test_calibration_eligibility_uses_fixed_quality_representation_gates(
    generation, quality, review, reason: str
) -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    row = module.validate_calibration_arm("arm", generation, quality, review)

    assert row["eligible"] is False
    assert reason in row["ineligible_reasons"]


def test_calibration_winner_order_is_deterministic() -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    rows = [
        module.validate_calibration_arm(
            "z-arm", _generation(e0=0.58, edev=0.63), _quality(), _review(severe=2)
        ),
        module.validate_calibration_arm(
            "a-arm", _generation(e0=0.58, edev=0.63), _quality(), _review(severe=1)
        ),
    ]

    assert module.select_calibration_winner(rows)["winner"]["arm_id"] == "a-arm"
    with pytest.raises(ValueError, match="no eligible"):
        module.select_calibration_winner(
            [module.validate_calibration_arm("failed", _generation(e0=0.50), _quality(), _review())]
        )


def test_calibration_selection_locks_fmrg_tcut_and_schedule_digests() -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    row = module.validate_calibration_arm("official", _generation(), _quality(), _review())

    selection = module.select_calibration_winner([row])

    assert selection["winner"]["t_cut"] == 0.25
    assert selection["winner"]["schedule_manifest_sha256"] == SCHEDULE_FILE_SHA256
    assert selection["winner"]["schedule_contract_sha256"] == SCHEDULE_CONTRACT_SHA256

    missing = _generation()
    missing.pop("schedule")
    with pytest.raises(ValueError, match="schedule"):
        module.validate_calibration_arm("official", missing, _quality(), _review())


def test_calibration_selection_rejects_cross_arm_sample_membership_mismatch() -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    first = module.validate_calibration_arm("first", _generation(), _quality(), _review())
    second_generation = _generation(e0=0.56)
    second_quality = _quality()
    second_generation["sample_id_sha256"] = "b" * 64
    second_quality["sample_id_sha256"] = "b" * 64
    second = module.validate_calibration_arm(
        "second", second_generation, second_quality, _review()
    )

    with pytest.raises(ValueError, match="same sample-ID"):
        module.select_calibration_winner([first, second])


def _write_arm(root: Path, arm_id: str, generation: dict, quality: dict) -> None:
    arm = root / "calibration" / arm_id
    arm.mkdir(parents=True)
    (arm / "generation_result.json").write_text(json.dumps(generation), encoding="utf-8")
    (arm / "quality.json").write_text(json.dumps(quality), encoding="utf-8")


def test_calibration_summary_requires_visual_review_and_writes_json_csv_markdown(
    tmp_path: Path,
) -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    _write_arm(tmp_path, "arm-a", _generation(e0=0.57), _quality(fid=55.0))
    _write_arm(tmp_path, "arm-b", _generation(e0=0.56), _quality(fid=40.0))

    with pytest.raises(FileNotFoundError, match="visual_review"):
        module.summarize_calibration(tmp_path)

    visual = {
        "reviewed_sample_count": 64,
        "arms": {"arm-a": _review(severe=1), "arm-b": _review(severe=2)},
    }
    (tmp_path / "visual_review.json").write_text(json.dumps(visual), encoding="utf-8")
    selection = module.summarize_calibration(tmp_path)

    assert selection["winner"]["arm_id"] == "arm-a"
    assert (tmp_path / "selection.json").is_file()
    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    for column in ("FID", "KID", "NIQE", "Sharpness", "E0 cosine", "NFE", "images/s", "peak VRAM", "severe"):
        assert column in markdown
    assert "fid" in csv_text and "peak_vram_bytes" in csv_text


def _full_arm(*, fid: float, kid: float, sharpness: float, e0: float) -> dict:
    return {
        "fid": fid,
        "kid_mean": kid,
        "sharpness_mean": sharpness,
        "e0_cosine": e0,
        "all_finite": True,
    }


def _heldout(native: float, winner: float) -> dict:
    return {
        "encoders": {
            "e1_dinov2_large_v2": {
                "native": {"paired_source_generated_cosine": {"mean": native}},
                "winner": {"paired_source_generated_cosine": {"mean": winner}},
            },
            "e2_convnext_tiny": {
                "native": {"paired_source_generated_cosine": {"mean": native - 0.01}},
                "winner": {"paired_source_generated_cosine": {"mean": winner - 0.01}},
            },
        }
    }


def test_full_decision_labels_solved_directional_and_failed_exactly() -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    native = _full_arm(fid=40.0, kid=0.02, sharpness=320.0, e0=0.45)

    solved = module.classify_full_result(
        native,
        _full_arm(fid=42.0, kid=0.024, sharpness=310.0, e0=0.52),
        _heldout(0.4, 0.45),
        severe_failure_count=3,
    )
    directional = module.classify_full_result(
        native,
        _full_arm(fid=50.0, kid=0.03, sharpness=290.0, e0=0.51),
        _heldout(0.4, 0.42),
        severe_failure_count=4,
    )
    failed = module.classify_full_result(
        native,
        _full_arm(fid=50.0, kid=0.03, sharpness=200.0, e0=0.46),
        _heldout(0.4, 0.35),
        severe_failure_count=8,
    )

    assert solved["label"] == "solved"
    assert directional["label"] == "directional_evidence"
    assert failed["label"] == "failed"


def test_full_summary_reads_merged_per_sample_cosine_without_generation_manifest(
    tmp_path: Path,
) -> None:
    module = _load_script("summarize_r8_meanflow_guidance")
    quality = _quality(fid=42.0, sharpness=310.0)
    (tmp_path / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    rows = [
        {"sample_id": "a", "candidate_cosine": 0.5, "native_cosine": 0.4},
        {"sample_id": "b", "candidate_cosine": 0.7, "native_cosine": 0.6},
    ]
    (tmp_path / "per_sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    winner = module._full_quality_row(tmp_path, e0_key="candidate")
    native = module._full_quality_row(tmp_path, e0_key="native")

    assert winner["e0_cosine"] == pytest.approx(0.6)
    assert native["e0_cosine"] == pytest.approx(0.5)


def test_heldout_contract_requires_locked_winner_exact_manifests_and_one_shot_marker(
    tmp_path: Path,
) -> None:
    module = _load_script("eval_r8_heldout_encoders")
    selection = {
        "winner": {"arm_id": "winner", "config_sha256": "c" * 64},
        "winner_locked_before_heldout": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "full_sample_count": 2048,
        "full_sample_id_sha256": SAMPLE_DIGEST,
    }
    native = {"sample_count": 2048, "sample_id_sha256": SAMPLE_DIGEST}
    winner = {"sample_count": 2048, "sample_id_sha256": SAMPLE_DIGEST}

    contract = module.validate_heldout_contract(selection, native, winner)
    assert contract["sample_count"] == 2048

    marker = tmp_path / "heldout_protocol_marker.json"
    module.claim_protocol_marker(marker, contract)
    with pytest.raises(FileExistsError, match="second"):
        module.claim_protocol_marker(marker, contract)

    with pytest.raises(ValueError, match="2048"):
        module.validate_heldout_contract(
            selection, {**native, "sample_count": 2047}, winner
        )
