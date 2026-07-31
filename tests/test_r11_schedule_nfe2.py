from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts import classify_r11_schedule_nfe2 as classifier_cli
from scripts import prepare_r11_schedule_nfe2 as preparation_cli
from scripts import run_r11_schedule_nfe2_job as job_runner
from safa.evaluation import meanflow_guidance_runner as guidance_runner
from safa.evaluation.r9_determinism import canonical_guidance_arm_config_digest
from safa.evaluation.schedule_nfe2 import (
    ScheduleNFE2Error,
    classify_schedule_nfe2,
)


REPO = Path(__file__).resolve().parents[1]
PREPARATION = REPO / "artifacts/r11_schedule_nfe2/preparation_v1"
PARENT = REPO / "artifacts/r11_causal_decomposition/preparation_v1"
PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
FORMAL_SHARDS = (
    REPO
    / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
    / "r9-report-only-formal-v9/full/winner/shards"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(*, prefix_pass: bool, tail_pass: bool) -> dict:
    evidence = {}
    for dataset_id, count, candidate_pass in (
        ("prefix128", 128, prefix_pass),
        ("sharpness_tail32", 32, tail_pass),
    ):
        native = {"niqe": 4.0, "sharpness": 100.0}
        candidate = (
            {"niqe": 4.1, "sharpness": 95.0}
            if candidate_pass
            else {"niqe": 4.1001, "sharpness": 95.0}
        )
        evidence[dataset_id] = {
            "sample_count": count,
            "quality": {
                "native": native,
                "paper_eta_0p125": {"niqe": 3.9, "sharpness": 90.0},
                "schedule_nfe2": candidate,
            },
            "representation": {
                role: {"e0_cosine": 0.5, "edev_cosine": 0.6}
                for role in ("native", "paper_eta_0p125", "schedule_nfe2")
            },
        }
    return evidence


def test_nfe2_classifier_applies_locked_two_branch_rule() -> None:
    schedule = classify_schedule_nfe2(
        _evidence(prefix_pass=True, tail_pass=True)
    )
    assert schedule["classification"] == "schedule_limited"
    assert schedule["nfe2_quality_passes_both_datasets"] is True
    assert schedule["split_route"] == {
        "status": "diagnostic_complete",
        "stop_required": False,
    }
    assert schedule["candidate_promotion"]["status"] == "forbidden"
    assert schedule["geometry"]["status"] == "not_evaluated"
    assert schedule["privacy"]["status"] == "not_evaluated"

    for prefix_pass, tail_pass in ((False, True), (True, False), (False, False)):
        failure = classify_schedule_nfe2(
            _evidence(prefix_pass=prefix_pass, tail_pass=tail_pass)
        )
        assert failure["classification"] == "mixed_guidance_failure"
        assert failure["nfe2_quality_passes_both_datasets"] is False
        assert failure["split_route"] == {
            "status": "stop",
            "stop_required": True,
        }


def test_nfe2_classifier_fails_closed_on_nonfinite_or_forbidden_metrics() -> None:
    evidence = _evidence(prefix_pass=True, tail_pass=True)
    evidence["prefix128"]["quality"]["schedule_nfe2"]["niqe"] = math.nan
    with pytest.raises(ScheduleNFE2Error, match="finite"):
        classify_schedule_nfe2(evidence)

    evidence = _evidence(prefix_pass=True, tail_pass=True)
    evidence["prefix128"]["fid"] = 1.0
    with pytest.raises(ScheduleNFE2Error, match="non-classifier metric"):
        classify_schedule_nfe2(evidence)

    evidence = _evidence(prefix_pass=True, tail_pass=True)
    evidence["prefix128"]["quality"]["schedule_nfe2"]["arcface"] = 1.0
    with pytest.raises(ScheduleNFE2Error, match="non-classifier metric"):
        classify_schedule_nfe2(evidence)


def test_classifier_requires_exact_materialized_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = {"asset_digest_cache": "relative/cache.json"}
    execution_contract = {
        "experiment_contract": "safa_r9_meanflow_v1",
        "attention_backend": "native",
    }
    runtime = {
        "asset_digest_cache": "/repo/relative/cache.json",
        "r9_execution_contract": execution_contract,
    }
    monkeypatch.setattr(
        classifier_cli,
        "materialize_runtime_guidance_config",
        lambda config, *, shard_index, num_shards: dict(runtime),
    )

    assert (
        classifier_cli._require_exact_runtime_config(
            prepared,
            runtime,
            dataset_id="prefix128",
        )
        == runtime
    )
    mutations = {
        "unexpected field": {
            **runtime,
            "unexpected": True,
        },
        "asset path": {
            **runtime,
            "asset_digest_cache": "/wrong/cache.json",
        },
        "execution contract": {
            **runtime,
            "r9_execution_contract": {
                **execution_contract,
                "attention_backend": "substituted",
            },
        },
    }
    for label, executed in mutations.items():
        with pytest.raises(
            ScheduleNFE2Error,
            match="executed config differs from prepared config",
        ):
            classifier_cli._require_exact_runtime_config(
                prepared,
                executed,
                dataset_id=f"prefix128 {label}",
            )


def test_one_shot_job_runner_writes_receipt_and_refuses_second_attempt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.txt"
    log = tmp_path / "job.log"
    receipt = tmp_path / "receipt.json"
    ledger_path = tmp_path / "ledger.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("arm: test\n", encoding="utf-8")
    job_id = "test__one_shot"
    ledger = {
        "schema_version": 1,
        "contract_type": "safa_r11_schedule_nfe2_launch_ledger_v1",
        "status": "prepared_not_launched",
        "retry_limit": 0,
        "jobs": [
            {
                "job_id": job_id,
                "retry_limit": 0,
                "attempt_limit": 1,
                "gpu": {"index": 0, "uuid": "GPU-test"},
                "env": {"CUDA_VISIBLE_DEVICES": "GPU-test"},
                "argv": [
                    PYTHON,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(output)!r}).write_text('ok')"
                    ),
                ],
                "config_binding": {
                    "path": str(config_path),
                    "sha256": "0" * 64,
                },
                "job_type": "quality",
                "log_path": str(log),
                "attempt_receipt_path": str(receipt),
                "fresh_output_paths": [
                    str(output),
                    str(log),
                    str(receipt),
                ],
                "prerequisite_paths": [],
                "success_output_paths": [str(output)],
            }
        ],
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="config SHA256 differs"):
        job_runner.run_job(ledger_path, job_id)
    assert not any(path.exists() for path in (output, log, receipt))
    ledger["jobs"][0]["config_binding"]["sha256"] = _sha256(config_path)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    assert job_runner.run_job(ledger_path, job_id) == 0

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["attempt_count"] == 1
    assert payload["retry_count"] == 0
    assert payload["exit_status"] == 0
    assert payload["ledger_sha256"] == _sha256(ledger_path)
    assert payload["config_binding"] == ledger["jobs"][0]["config_binding"]
    with pytest.raises(FileExistsError, match="repeated attempt"):
        job_runner.run_job(ledger_path, job_id)


def test_preparer_materializes_fresh_retry0_contract_without_launch(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "preparation"
    config_dir = tmp_path / "configs"

    manifest = preparation_cli.prepare(output_root, config_dir)

    assert manifest["status"] == "prepared_not_launched"
    assert manifest["job_counts"] == {"generation": 2, "quality": 2}
    ledger = json.loads(
        (output_root / "launch_ledger.json").read_text(encoding="utf-8")
    )
    assert len(ledger["jobs"]) == 4
    assert all(job["attempt_limit"] == 1 for job in ledger["jobs"])
    assert all(job["retry_limit"] == 0 for job in ledger["jobs"])
    assert all(
        not Path(path).exists()
        for job in ledger["jobs"]
        for path in job["fresh_output_paths"]
    )
    for dataset_id in ("prefix128", "sharpness_tail32"):
        config = yaml.safe_load(
            (config_dir / f"r11_schedule_nfe2_{dataset_id}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["r11_nfe2_schedule_contract"][
            "times"
        ] == [1.0, 0.25, 0.0]
        assert config["r9_guidance_interval_contract"][
            "expected_algorithm_nfe"
        ] == 2
        assert all(
            job["config_binding"]["sha256"]
            == _sha256(
                config_dir / f"r11_schedule_nfe2_{dataset_id}.yaml"
            )
            for job in ledger["jobs"]
            if dataset_id in job["job_id"]
        )


def test_preparation_binds_exact_schedule_parent_assets_and_retry0_ledger() -> None:
    manifest = json.loads(
        (PREPARATION / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (PREPARATION / "launch_ledger.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (PREPARATION / "diagnostic_contract.json").read_text(encoding="utf-8")
    )
    parent_classification = json.loads(
        (PARENT / "causal_classification.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == ledger["status"] == "prepared_not_launched"
    assert manifest["job_counts"] == {"generation": 2, "quality": 2}
    assert manifest["parent_evidence"]["classification"] == "schedule_branch"
    assert parent_classification["classification"] == "schedule_branch"
    assert (
        manifest["parent_evidence"]["causal_classification"]["sha256"]
        == _sha256(PARENT / "causal_classification.json")
    )
    assert manifest["invariants"] == {
        "active_guidance_intervals": [],
        "attempt_limit": 1,
        "batch_size": 2,
        "candidate_promotion": "forbidden",
        "checkpoint": "E15",
        "effective_edev": True,
        "expected_algorithm_nfe": 2,
        "expected_diagnostic_nfe": 0,
        "expected_matched_native_nfe": 0,
        "retry_limit": 0,
        "sampling_seed": 7919,
        "schedule_times": [1.0, 0.25, 0.0],
    }
    assert contract["classifier"]["metrics"] == ["niqe", "sharpness"]
    assert contract["classifier"]["prefix_fid_kid"] == "descriptive_only"
    assert contract["classifier"]["tail_fid_kid"] == "forbidden"
    assert contract["classifier"]["candidate_promotion"] == "forbidden"

    jobs = ledger["jobs"]
    assert len(jobs) == 4
    assert sum(job["job_type"] == "generation" for job in jobs) == 2
    assert sum(job["job_type"] == "quality" for job in jobs) == 2
    assert [job["wave"] for job in jobs] == [0, 1, 0, 1]
    assert all(job["retry_limit"] == 0 for job in jobs)
    assert all(job["attempt_limit"] == 1 for job in jobs)
    assert all(job["argv"][0] == PYTHON for job in jobs)
    assert all(Path(job["argv"][0]).is_file() for job in jobs)
    assert all(job["env"]["CUDA_VISIBLE_DEVICES"] == job["gpu"]["uuid"] for job in jobs)
    assert len({job["tmux"]["session"] for job in jobs}) == 4
    assert all(
        job["tmux"]["argv"][1] == "scripts/run_r11_schedule_nfe2_job.py"
        for job in jobs
    )
    assert all(job["attempt_receipt_path"].endswith(".json") for job in jobs)
    fresh = [path for job in jobs for path in job["fresh_output_paths"]]
    assert len(fresh) == len(set(fresh)) == 14
    assert not any(
        forbidden in token.lower()
        for job in jobs
        for token in job["argv"]
        for forbidden in ("arcface", "privacy", "promotion")
    )
    generation_jobs = [job for job in jobs if job["job_type"] == "generation"]
    assert all(
        job["argv"][1] == "scripts/run_meanflow_flow_map_guidance.py"
        and "--output-dir" in job["argv"]
        and job["asset_digest_path"] in job["fresh_output_paths"]
        and job["log_path"] in job["fresh_output_paths"]
        and job["attempt_receipt_path"] in job["fresh_output_paths"]
        and job["expected_algorithm_nfe"] == 2
        and job["expected_diagnostic_nfe"] == 0
        and job["expected_matched_native_nfe"] == 0
        for job in generation_jobs
    )
    quality_jobs = [job for job in jobs if job["job_type"] == "quality"]
    assert all(
        "--generation-result" in job["argv"]
        and job["log_path"] in job["fresh_output_paths"]
        and job["attempt_receipt_path"] in job["fresh_output_paths"]
        for job in quality_jobs
    )
    assert next(
        job for job in quality_jobs if job["dataset_id"] == "prefix128"
    )["metrics"] == ["fid", "kid", "niqe", "sharpness"]
    assert next(
        job for job in quality_jobs if job["dataset_id"] == "sharpness_tail32"
    )["metrics"] == ["niqe", "sharpness"]

    schedule_contract = guidance_runner.locked_r11_nfe2_schedule_contract()
    for dataset in contract["datasets"]:
        dataset_id = dataset["dataset_id"]
        parent_config_path = (
            REPO
            / "configs/medium_v2/experiments"
            / f"r11_transport_only_nfe5_{dataset_id}.yaml"
        )
        nfe2_config_path = (
            REPO
            / "configs/medium_v2/experiments"
            / f"r11_schedule_nfe2_{dataset_id}.yaml"
        )
        parent_config = yaml.safe_load(parent_config_path.read_text(encoding="utf-8"))
        nfe2_config = yaml.safe_load(nfe2_config_path.read_text(encoding="utf-8"))
        assert (
            nfe2_config["arm_config_sha256"]
            == canonical_guidance_arm_config_digest(nfe2_config)
        )
        assert (
            nfe2_config["arm_config_sha256"]
            != canonical_guidance_arm_config_digest(parent_config)
        )
        drifted_config = deepcopy(nfe2_config)
        drifted_config["r11_nfe2_schedule_contract"]["times"][1] = 0.5
        assert (
            canonical_guidance_arm_config_digest(drifted_config)
            != nfe2_config["arm_config_sha256"]
        )
        assert dataset["nfe2_config"]["sha256"] == _sha256(nfe2_config_path)
        assert nfe2_config["sampling_seed"] == nfe2_config["seed"] == 7919
        assert nfe2_config["batch_size"] == 2
        assert nfe2_config["arm_name"] == "schedule_nfe2"
        assert nfe2_config["active_guidance_intervals"] == []
        assert nfe2_config["r11_nfe2_schedule_contract"] == schedule_contract
        assert nfe2_config["r9_guidance_interval_contract"][
            "expected_algorithm_trace"
        ] == schedule_contract["expected_algorithm_trace"]
        assert nfe2_config["r9_guidance_interval_contract"][
            "expected_algorithm_nfe"
        ] == 2
        assert nfe2_config["r9_guidance_interval_contract"][
            "expected_diagnostic_nfe"
        ] == 0
        assert (
            nfe2_config["external_native_contract"]
            == parent_config["external_native_contract"]
        )
        assert "seed1337" not in json.dumps(dataset).lower()
        roles = dataset["roles"]
        for role in ("native", "paper_eta_0p125"):
            for binding in roles[role].values():
                path = REPO / binding["path"]
                assert path.is_file()
                assert binding["sha256"] == _sha256(path)
        for binding in roles["schedule_nfe2"].values():
            assert binding["expected_absent_at_preparation"] is True

        selection = [
            json.loads(line)
            for line in (REPO / dataset["selection_manifest"]["path"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        reuse_path = REPO / dataset["reuse_asset_bindings"]["path"]
        reuse = [
            json.loads(line)
            for line in reuse_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(reuse) == dataset["sample_count"]
        assert [row["sample_id"] for row in reuse] == [
            row["sample_id"] for row in selection
        ]
        for ordinal, row in enumerate(reuse):
            assert row["ordinal"] == ordinal
            assert "seed1337" not in json.dumps(row).lower()
            for role in ("source", "native", "paper"):
                path = Path(row[role])
                assert path.is_file()
                assert row[f"{role}_sha256"] == _sha256(path)
            assert FORMAL_SHARDS.resolve() in Path(row["native"]).resolve().parents
            assert FORMAL_SHARDS.resolve() in Path(row["paper"]).resolve().parents

    tampered_contract = deepcopy(contract)
    tampered_contract["datasets"][0]["nfe2_config"]["sha256"] = "0" * 64
    with pytest.raises(ScheduleNFE2Error, match="prepared NFE2 config SHA256 differs"):
        classifier_cli.materialize_evidence(tampered_contract)
