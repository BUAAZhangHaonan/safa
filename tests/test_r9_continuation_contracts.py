from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from safa.evaluation import r9_continuation_contracts as continuation
from safa.evaluation.r9_campaign_contracts import build_a_gate_contract


SHA = "a" * 64
PARENT = continuation.R9_CONTINUATION_PARENT_CAMPAIGN_ID


def _digest(payload: dict[str, object], field: str) -> str:
    canonical = dict(payload)
    canonical.pop(field, None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fixture(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    files = {
        "checkpoint": "models/meanflow.pt",
        "schedule": "contracts/schedule.json",
        "semigroup": "contracts/semigroup.json",
        "driver": "scripts/run_r9_meanflow_campaign.py",
        "entrypoint": "scripts/run_r9_phase_evaluator.py",
        "worker": "src/safa/evaluation/r9_evaluator_worker.py",
        "quality": "scripts/eval_generation_quality.py",
    }
    hashes = {name: _write(repo / path, name.encode()) for name, path in files.items()}
    _write(repo / continuation.R9_CONTINUATION_REQUEST_PATH, b"request")
    _write(repo / continuation.R9_CONTINUATION_BASE_RUNTIME_PATH, b"base")
    manifests = {}
    for name in sorted(continuation.R9_CONTINUATION_MANIFESTS):
        path = f"manifests/{name}.jsonl"
        manifests[name] = {"path": path, "sha256": _write(repo / path, name.encode())}
    runtime = {
        "campaign_id": PARENT,
        "manifest_contracts_sha256": "b" * 64,
        "manifests": manifests,
        "checkpoint": {"path": files["checkpoint"], "sha256": hashes["checkpoint"]},
        "determinism_policy_sha256": "c" * 64,
        "attention_backend": "native",
        "schedule": {
            "path": files["schedule"],
            "file_sha256": hashes["schedule"],
            "contract_sha256": "d" * 64,
        },
        "semigroup_gate": {
            "path": files["semigroup"],
            "file_sha256": hashes["semigroup"],
            "contract_sha256": "e" * 64,
        },
        "evaluation": {
            "worker": {
                "path": files["entrypoint"],
                "implementation_path": files["worker"],
            },
            "quality": {"script": {"path": files["quality"]}},
        },
    }
    runtime["campaign_runtime_sha256"] = _digest(
        runtime, "campaign_runtime_sha256"
    )
    parent = (
        repo
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / PARENT
    )
    _write(
        parent / "campaign_runtime.json",
        (json.dumps(runtime, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    phase = {
        "phase": "diagnose",
        "campaign_runtime_sha256": runtime["campaign_runtime_sha256"],
    }
    phase["phase_results_sha256"] = _digest(phase, "phase_results_sha256")
    _write(
        parent / "diagnose/phase_results.json",
        (json.dumps(phase, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    context = {
        "campaign_id": PARENT,
        "campaign_runtime_sha256": runtime["campaign_runtime_sha256"],
        "manifest_contracts_sha256": runtime["manifest_contracts_sha256"],
        "manifest_sha256": "9" * 64,
        "checkpoint_sha256": hashes["checkpoint"],
        "phase_results_sha256": phase["phase_results_sha256"],
        "automatic_evidence_sha256": "1" * 64,
        "run_plan_sha256": "2" * 64,
        "evaluator_evidence_sha256": "3" * 64,
    }
    arms = []
    for arm_id, family, digest in (
        ("flow", "flow_map2", "4" * 64),
        ("paper", "paper_split_constant", "5" * 64),
        ("ablation", "paper_split_interval_ablation", "6" * 64),
    ):
        arms.append(
            {
                "arm_id": arm_id,
                "family": family,
                "config_sha256": "7" * 64,
                "output_sha256": "8" * 64,
                "repeat_results": [
                    {
                        "repeat_index": index,
                        "run_sha256": digest,
                        "difficult_severe_count": 0,
                        "control_severe_count": 0,
                        "e0_mean": 0.8,
                        "edev_delta_vs_matched_native": 0.1,
                        "diagnostics_finite": True,
                    }
                    for index in range(3)
                ],
            }
        )
    gate = build_a_gate_contract(
        context,
        arms,
        diagnose_manifest={
            "path": "manifests/diagnose.jsonl",
            "sha256": "9" * 64,
            "sample_count": 18,
            "ordered_sample_id_sha256": "a" * 64,
            "difficult_count": 9,
            "control_count": 9,
            "matched_pair_sha256": "b" * 64,
        },
    )
    _write(
        parent / "diagnose/gate_contract.json",
        (json.dumps(gate, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    monkeypatch.setattr(
        continuation, "R9_CONTINUATION_PARENT_GATE_SHA256", gate["gate_contract_sha256"]
    )
    monkeypatch.setattr(
        continuation,
        "R9_CONTINUATION_PARENT_PHASE_RESULTS_SHA256",
        phase["phase_results_sha256"],
    )
    return {
        "parent_campaign_id": PARENT,
        "diagnose_gate_contract_sha256": gate["gate_contract_sha256"],
        "diagnose_phase_results_sha256": phase["phase_results_sha256"],
    }


def test_continuation_materializes_idempotently_and_validates_live_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture(tmp_path, monkeypatch)
    first, binding = continuation.materialize_continuation_contract(
        repo_root=tmp_path, child_campaign_id="r9-child-v3", source=source
    )
    second, repeated_binding = continuation.materialize_continuation_contract(
        repo_root=tmp_path, child_campaign_id="r9-child-v3", source=source
    )

    assert first == second
    assert binding == repeated_binding
    assert len(first["selected_arms"]) == 3
    assert "campaign_runtime_sha256" not in first
    assert continuation.validate_continuation_contract(
        first, repo_root=tmp_path
    ) == first


def test_continuation_rejects_wrong_parent_and_parent_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture(tmp_path, monkeypatch)
    with pytest.raises(continuation.ContinuationContractError, match="sealed R9 v2"):
        continuation.build_continuation_contract(
            repo_root=tmp_path,
            child_campaign_id="r9-child-v3",
            source={**source, "parent_campaign_id": "wrong-parent"},
        )
    contract = continuation.build_continuation_contract(
        repo_root=tmp_path, child_campaign_id="r9-child-v3", source=source
    )
    gate_path = (
        tmp_path
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / PARENT
        / "diagnose/gate_contract.json"
    )
    gate_path.write_bytes(gate_path.read_bytes() + b" ")
    with pytest.raises(continuation.ContinuationContractError):
        continuation.validate_continuation_contract(contract, repo_root=tmp_path)


def test_continuation_rejects_selected_candidate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture(tmp_path, monkeypatch)
    contract = continuation.build_continuation_contract(
        repo_root=tmp_path, child_campaign_id="r9-child-v3", source=source
    )
    changed = json.loads(json.dumps(contract))
    changed["selected_arms"][0]["output_sha256"] = SHA
    changed["continuation_contract_sha256"] = _digest(
        changed, "continuation_contract_sha256"
    )
    with pytest.raises(continuation.ContinuationContractError, match="live bindings"):
        continuation.validate_continuation_contract(changed, repo_root=tmp_path)
