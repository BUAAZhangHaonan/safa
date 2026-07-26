from __future__ import annotations

import json
from pathlib import Path

import pytest

from safa.closeout.ledger import (
    CloseoutError,
    build_closeout_snapshot,
    write_closeout_snapshot,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(
        repo / "configs/closeout/historical_evidence_policy.json",
        json.dumps(
            {
                "schema_version": 1,
                "contract_type": "safa_historical_evidence_policy_v1",
                "legacy_invalid_series": {
                    "logical_experiment_ids": ["R1", "R2", "R3", "R4", "R5"],
                    "evidence_level": "legacy_untrusted_candidate_discovery",
                    "status_when_execution_evidence_exists": "invalid_evaluation",
                    "reason": "frozen reason",
                    "sources": ["legacy.md"],
                },
                "trusted_historical_baselines": {
                    "logical_experiment_ids": ["R6", "R7", "R9"],
                    "evidence_level": "strong_provenance_historical_baseline",
                },
                "protocol_families": {
                    "R1-R5": "legacy_lora_loader_bug",
                    "R6-R7": "lora_pu_historical",
                    "R8": "meanflow_guidance_r8",
                    "R9": "meanflow_guidance_r9",
                    "E1-E15": "medium_v2_training_historical",
                    "E16-E23": "medium_v2_generator_matrix",
                },
            }
        ),
    )
    _write(repo / "legacy.md", "legacy evidence\n")
    _write(
        repo / "configs/medium_v2/experiments/e16_model.yaml",
        "seed: 1337\nquality_eval:\n  output_dir: artifacts/unrelated/quality\n",
    )
    _write(
        repo / "configs/medium_v2/experiments/r1_legacy.yaml",
        "experiment_name: r1_legacy\nseed: 7\nout_dir: artifacts/checkpoints/r1_legacy\n",
    )
    _write(
        repo / "configs/medium_v2/experiments/r9_campaign.yaml",
        "campaign_id: r9-campaign\nseed: 4549\n",
    )
    _write(
        repo / "configs/medium_v2/experiments/r8_gate.yaml",
        "campaign_id: r8-campaign\nseed: 4549\n",
    )
    _write(
        repo / "configs/medium_v2/experiments/r7_closed.yaml",
        "experiment_name: r7-closed\nseed: 7919\n",
    )
    _write(
        repo / "configs/medium_v2/experiments/r9_superseded.yaml",
        "campaign_id: r9-superseded\nseed: 4549\nsource:\n  campaign_id: r9-upstream\n",
    )
    _write(
        repo / "configs/medium_v2/experiments/r8_clean.yaml",
        "campaign_id: r8-clean\nseed: 4549\n",
    )
    existing = {"E16", "R1", "R7", "R8", "R9"}
    for series, maximum in (("E", 23), ("R", 9)):
        for number in range(1, maximum + 1):
            logical_id = f"{series}{number}"
            if logical_id in existing:
                continue
            _write(
                repo
                / "configs/medium_v2/experiments"
                / f"{logical_id.lower()}_placeholder.yaml",
                f"experiment_name: {logical_id.lower()}_placeholder\nseed: 1\n",
            )
    _write(repo / "artifacts/checkpoints/r1_legacy/last.pt", "checkpoint\n")
    _write(repo / "artifacts/r1_legacy/metrics.json", '{"metric": 1}\n')
    _write(
        repo / "artifacts/r9-campaign/awaiting_visual_review.json",
        '{"status": "awaiting_visual_review"}\n',
    )
    _write(repo / "artifacts/r8-campaign/gate_contract.json", '{"passed": false}\n')
    _write(
        repo / "artifacts/r7-closed/gate_contract.json",
        '{"contract_type": "safa_r9_full_continuation_v1", "phase": "full", "passed": true}\n',
    )
    chain = "a" * 64
    gate_sha = "b" * 64
    selection_sha = "c" * 64
    _write(
        repo / "artifacts/r9-superseded/awaiting_visual_review.json",
        json.dumps(
            {
                "status": "awaiting_visual_review",
                "supersession_contract_sha256": chain,
            }
        ),
    )
    _write(
        repo / "artifacts/r9-superseded/gate_contract_v3.json",
        json.dumps(
            {
                "contract_type": "safa_r9_confirm512_report_only_gate_v3",
                "supersession_contract_sha256": chain,
                "gate_contract_sha256": gate_sha,
                "verdict": "winner_locked_report_only",
            }
        ),
    )
    _write(
        repo / "artifacts/r9-superseded/selection.json",
        json.dumps(
            {
                "contract_type": "safa_r9_confirm512_report_only_selection_v3",
                "supersession_contract_sha256": chain,
                "gate_contract_sha256": gate_sha,
                "selection_sha256": selection_sha,
                "next_stage": "new_v9_full_continuation_required",
                "reselection_allowed": False,
            }
        ),
    )
    _write(
        repo / "artifacts/r9-superseded/supersession_result.json",
        json.dumps(
            {
                "supersession_contract_sha256": chain,
                "gate_contract_sha256": gate_sha,
                "selection_sha256": selection_sha,
                "verdict": "winner_locked_report_only",
                "generation_execution_count": 0,
                "evaluator_execution_count": 0,
            }
        ),
    )
    _write(
        repo / "artifacts/r9-upstream/awaiting_visual_review.json",
        json.dumps(
            {
                "status": "awaiting_visual_review",
                "supersession_contract_sha256": "d" * 64,
            }
        ),
    )
    _write(
        repo / "artifacts/r8-clean-collision/gate_contract.json",
        '{"passed": false}\n',
    )
    _write(repo / "artifacts/unrelated/quality/metrics.json", '{"fid": 1}\n')
    _write(repo / "src/safa/evaluation/fixture_eval.py", "VALUE = 1\n")
    import subprocess

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    _write(repo / "src/safa/evaluation/fixture_eval.py", "VALUE = 2\n")
    return repo


def test_classifies_config_only_legacy_invalid_and_pending_visual(tmp_path: Path) -> None:
    snapshot = build_closeout_snapshot(_repo(tmp_path))
    rows = {row["run_id"]: row for row in snapshot["rows"]}
    assert rows["e16_model"]["status"] == "config_only_never_started"
    assert rows["e16_model"]["evidence_level"] == "config_only"
    assert rows["r1_legacy"]["status"] == "invalid_evaluation"
    assert (
        rows["r1_legacy"]["evidence_level"]
        == "legacy_untrusted_candidate_discovery"
    )
    assert rows["r9_campaign"]["status"] == "pending_visual_finalize"
    assert rows["r9_campaign"]["evidence"]["review_paths"]
    assert rows["r8_gate"]["status"] == "completed_gate_fail"
    assert rows["r7_closed"]["status"] == "started_incomplete"
    assert rows["r9_superseded"]["status"] == "started_incomplete"
    assert rows["r8_clean"]["status"] == "config_only_never_started"
    evaluator = snapshot["provenance"]["evaluator_bundle"]
    assert evaluator["source_count"] == 1
    assert evaluator["sources"][0]["dirty"] is True
    assert len(evaluator["sha256"]) == 64


def test_writes_required_outputs_and_refuses_overwrite(tmp_path: Path) -> None:
    snapshot = build_closeout_snapshot(_repo(tmp_path))
    target = tmp_path / "closeout"
    write_closeout_snapshot(snapshot, target)
    assert {path.name for path in target.iterdir()} == {
        "artifact_sha_manifest.jsonl",
        "closeout_binding.json",
        "documentation_conflicts.json",
        "experiment_ledger.csv",
        "experiment_ledger.jsonl",
        "missing_evidence.json",
        "protocol_registry.json",
        "provenance_snapshot.json",
    }
    with pytest.raises(CloseoutError, match="Refusing to overwrite"):
        write_closeout_snapshot(snapshot, target)


def test_missing_conflict_source_fails_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "legacy.md").unlink()
    with pytest.raises(CloseoutError, match="sources are missing"):
        build_closeout_snapshot(repo)


def test_bound_malformed_json_fails_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo / "artifacts/r9-campaign/broken_result.json", "{")
    with pytest.raises(CloseoutError, match="Cannot parse bound JSON evidence"):
        build_closeout_snapshot(repo)
