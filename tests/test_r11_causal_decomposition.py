from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from safa.evaluation.causal_decomposition import (
    CausalDecompositionError,
    classify_causal_decomposition,
)
from safa.evaluation import meanflow_guidance_runner as runner


REPO = Path(__file__).resolve().parents[1]
PREPARATION = REPO / "artifacts/r11_causal_decomposition/preparation_v1"


def _evidence(*, transport_pass: bool, paper_fail: bool) -> dict:
    result = {}
    for dataset_id, count in (("prefix128", 128), ("sharpness_tail32", 32)):
        native = {"niqe": 4.0, "sharpness": 100.0}
        transport = (
            {"niqe": 4.1, "sharpness": 95.0}
            if transport_pass
            else {"niqe": 4.1001, "sharpness": 95.0}
        )
        paper = (
            {"niqe": 4.11, "sharpness": 100.0}
            if paper_fail
            else {"niqe": 4.0, "sharpness": 100.0}
        )
        result[dataset_id] = {
            "sample_count": count,
            "quality": {
                "native": native,
                "transport_only_nfe5": transport,
                "paper_eta_0p125": paper,
            },
            "representation": {
                role: {"e0_cosine": 0.5, "edev_cosine": 0.6}
                for role in (
                    "native",
                    "transport_only_nfe5",
                    "paper_eta_0p125",
                )
            },
        }
    return result


def test_quality_only_classifier_applies_locked_branch_rule() -> None:
    correction = classify_causal_decomposition(
        _evidence(transport_pass=True, paper_fail=True)
    )
    assert correction["classification"] == "correction_limited"
    assert correction["geometry"]["status"] == "not_evaluated"
    assert correction["privacy"]["status"] == "not_evaluated"
    assert correction["candidate_promotion"]["status"] == "forbidden"

    schedule = classify_causal_decomposition(
        _evidence(transport_pass=True, paper_fail=False)
    )
    assert schedule["classification"] == "schedule_branch"
    schedule = classify_causal_decomposition(
        _evidence(transport_pass=False, paper_fail=True)
    )
    assert schedule["classification"] == "schedule_branch"


def test_classifier_rejects_fid_kid_or_u95_as_gate_evidence() -> None:
    evidence = _evidence(transport_pass=True, paper_fail=True)
    evidence["prefix128"]["fid"] = 1.0
    with pytest.raises(CausalDecompositionError, match="non-classifier metric"):
        classify_causal_decomposition(evidence)


def test_external_native_contract_is_strict_and_default_bindings_are_unchanged(
    tmp_path: Path,
) -> None:
    contract = {
        "schema_version": 1,
        "contract_type": "safa_r11_causal_external_native_v1",
        "manifest": str(tmp_path / "native.jsonl"),
        "manifest_sha256": "1" * 64,
        "sample_count": 2,
        "ordered_sample_id_sha256": "2" * 64,
    }
    config = {
        "experiment_contract": "safa_r9_meanflow_v1",
        "arm_name": "transport_only_nfe5",
        "causal_contract_type": "safa_r11_transport_only_nfe5_v1",
        "mode": "paper_algorithm_split",
        "phase": "calibrate",
        "sampling_seed": 7919,
        "step_size": 0.125,
        "active_guidance_intervals": [],
        "collect_interval_diagnostics": False,
        "external_native_contract": contract,
    }
    config[runner.R9_PHASE_CONTRACT_FIELD] = runner.validate_r9_phase_contract(config)
    assert runner._validate_r11_causal_config(config) == contract
    invalid = dict(config, active_guidance_intervals=["I1"])
    with pytest.raises(ValueError, match="active_guidance_intervals"):
        runner._validate_r11_causal_config(invalid)

    selected = [
        {"sample_id": "a", "source": "/source/a.png"},
        {"sample_id": "b", "source": "/source/b.png"},
    ]
    generated = tmp_path / "generated"
    native = tmp_path / "native"
    default = runner._expected_row_bindings(
        selected, generated, native, "paper_algorithm_split"
    )
    assert [row["native"] for row in default] == [
        str(native / "00000000__a.png"),
        str(native / "00000001__b.png"),
    ]
    external = runner._expected_row_bindings(
        selected,
        generated,
        native,
        "paper_algorithm_split",
        external_native_rows=[
            {"native": "/formal/native-a.png"},
            {"native": "/formal/native-b.png"},
        ],
    )
    assert [row["native"] for row in external] == [
        "/formal/native-a.png",
        "/formal/native-b.png",
    ]


def test_preparation_artifacts_bind_nfe5_reuse_and_exact_retry0_ledger() -> None:
    manifest = json.loads(
        (PREPARATION / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (PREPARATION / "launch_ledger.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "go_not_launched"
    assert manifest["formal_reuse"]["regeneration"] == {
        "native": False,
        "paper": False,
    }
    assert manifest["job_counts"] == {"generation": 2, "quality": 6}
    jobs = ledger["jobs"]
    assert len(jobs) == 8
    assert sum(job["job_type"] == "generation" for job in jobs) == 2
    assert sum(job["job_type"] == "quality" for job in jobs) == 6
    assert all(job["retry_limit"] == 0 for job in jobs)
    assert max(
        sum(job["wave"] == wave for job in jobs)
        for wave in {job["wave"] for job in jobs}
    ) <= 4

    for dataset_id, count in (("prefix128", 128), ("sharpness_tail32", 32)):
        config = yaml.safe_load(
            (
                REPO
                / "configs/medium_v2/experiments"
                / f"r11_transport_only_nfe5_{dataset_id}.yaml"
            ).read_text(encoding="utf-8")
        )
        assert config["arm_name"] == "transport_only_nfe5"
        assert config["sampling_seed"] == 7919
        assert config["active_guidance_intervals"] == []
        assert config["r9_guidance_interval_contract"]["expected_algorithm_nfe"] == 5
        assert config["r9_phase_contract"]["edev_required"] is True
        assert config["device"] == "cuda:0"
        rows = [
            json.loads(line)
            for line in (
                PREPARATION / f"reuse/{dataset_id}/native_external.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == count
        assert [row["ordinal"] for row in rows] == list(range(count))
        selection_rows = [
            json.loads(line)
            for line in Path(config["sample_id_manifest"]).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert [row["sample_id"] for row in rows] == [
            row["sample_id"] for row in selection_rows
        ]
        external_contract, bound = runner._load_external_native_bindings(
            config,
            [
                {"sample_id": row["sample_id"], "source": row["source"]}
                for row in rows
            ],
        )
        assert external_contract == config["external_native_contract"]
        assert bound is not None and len(bound) == count
