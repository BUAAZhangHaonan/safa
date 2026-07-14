from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_r9_evaluation_repair.py"
SPEC = importlib.util.spec_from_file_location("run_r9_evaluation_repair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repair
SPEC.loader.exec_module(repair)


def _inventory() -> dict[str, object]:
    return {
        **repair.EXPECTED_V6_GENERATION_COUNTS,
        "schema_version": 1,
        "contract_type": "safa_r9_generation_evidence_inventory_v1",
        "inventory_sha256": "1" * 64,
        "files": [],
    }


def test_cli_requires_explicit_execution_and_busy_gpu_permission() -> None:
    base = [
        "--campaign-id",
        "r9-report-only-formal-v6",
        "--phase",
        "calibrate",
        "--failed-unit-id",
        "unit",
        "--source-commit",
        "a" * 40,
        "--prior-phase-results-sha256",
        "b" * 64,
    ]
    with pytest.raises(SystemExit):
        repair.parse_args(base)
    with pytest.raises(SystemExit):
        repair.parse_args([*base, "--execute"])


def test_contract_binding_accepts_effective_runtime_without_contract_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repair, "REPO_ROOT", tmp_path)
    payload = {"schema_version": 1, "campaign_id": "v6"}
    payload["campaign_runtime_sha256"] = repair.driver._canonical_json_sha256(
        payload
    )
    path = tmp_path / "campaign_runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    binding = repair._contract_binding(
        path, digest_field="campaign_runtime_sha256", contract_type=None
    )
    assert binding["contract_sha256"] == payload["campaign_runtime_sha256"]


def test_contract_binding_rejects_digest_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repair, "REPO_ROOT", tmp_path)
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "contract_type": "safa_r9_phase_evaluator_request_v1",
                "evaluator_request_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(repair.EvaluationRepairError, match="canonical digest"):
        repair._contract_binding(
            path,
            digest_field="evaluator_request_sha256",
            contract_type="safa_r9_phase_evaluator_request_v1",
        )


def test_v6_inventory_requires_exact_frozen_counts() -> None:
    repair._assert_v6_generation_inventory(_inventory())
    changed = _inventory()
    changed["png_count"] = 1343
    with pytest.raises(repair.EvaluationRepairError, match="counts changed"):
        repair._assert_v6_generation_inventory(changed)


@dataclass(frozen=True)
class _Evaluation:
    logical_run_id: str


def _run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventories: list[dict[str, object]],
) -> tuple[int, list[str], dict[str, object]]:
    phase_root = tmp_path / "campaign" / "calibrate"
    request = SimpleNamespace(phase_root=phase_root)
    plan = object()
    effective = {"campaign_root": str(tmp_path / "campaign")}
    monkeypatch.setenv("TMUX", "test")
    monkeypatch.setattr(
        repair,
        "_load_campaign",
        lambda *_: ({"runtime": True}, effective, {}, {}, plan, request),
    )
    terminal_checks: list[object] = []
    monkeypatch.setattr(
        repair,
        "_assert_generation_terminal",
        lambda bound_request, bound_plan: terminal_checks.append(
            (bound_request, bound_plan)
        ),
    )
    inventory_iter = iter(inventories)
    monkeypatch.setattr(
        repair, "generation_evidence_inventory", lambda _: next(inventory_iter)
    )
    contract = {
        "schema_version": 1,
        "contract_type": "safa_r9_evaluation_repair_contract_v1",
        "policy": {"generation_execution": "forbidden"},
    }
    monkeypatch.setattr(repair, "_build_repair_contract", lambda *_, **__: contract)
    monkeypatch.setattr(
        repair,
        "evaluation_repair_binding",
        lambda _: {"contract_sha256": contract["repair_contract_sha256"]},
    )
    monkeypatch.setattr(
        repair.driver,
        "build_resource_scheduler",
        lambda _: ("scheduler", "gpu-bindings", "status"),
    )
    generation_calls: list[object] = []
    monkeypatch.setattr(
        repair.driver,
        "execute_campaign",
        lambda *args, **kwargs: generation_calls.append((args, kwargs)),
    )
    captured: dict[str, object] = {}
    evaluated_ids: list[str] = []

    class _Callbacks:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def quality(self, evaluation):
            evaluated_ids.append(evaluation.logical_run_id)
            return {"quality": True}

        def arcface(self, evaluation):
            evaluated_ids.append(evaluation.logical_run_id)
            return {"arcface": True}

    monkeypatch.setattr(repair.driver, "R9ProductionEvaluatorCallbacks", _Callbacks)

    def _materialize(_, *, quality_evaluator, arcface_evaluator):
        quality_evaluator(_Evaluation("quality-unit"))
        arcface_evaluator(_Evaluation("arcface-unit"))
        return SimpleNamespace(
            status="complete", required_review_count=0, completed_review_count=0
        )

    monkeypatch.setattr(repair, "materialize_phase_results", _materialize)
    result = repair.main(
        [
            "--campaign-id",
            "r9-report-only-formal-v6",
            "--phase",
            "calibrate",
            "--failed-unit-id",
            "unit",
            "--source-commit",
            "a" * 40,
            "--prior-phase-results-sha256",
            "b" * 64,
            "--execute",
            "--allow-busy-gpus",
        ]
    )
    assert generation_calls == []
    assert len(terminal_checks) == 2
    return result, evaluated_ids, captured


def test_main_binds_full_repair_sha_and_never_executes_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, evaluated_ids, captured = _run_main(
        tmp_path, monkeypatch, [_inventory(), _inventory()]
    )
    repair_contract = json.loads(
        (tmp_path / "campaign" / "calibrate" / repair.EVALUATION_REPAIR_FILENAME)
        .read_text(encoding="utf-8")
    )
    digest = repair_contract["repair_contract_sha256"]
    assert result == 0
    assert evaluated_ids == [
        f"repair_{digest}__quality-unit",
        f"repair_{digest}__arcface-unit",
    ]
    assert captured["campaign_runtime"]["campaign_root"] == str(
        tmp_path / "campaign" / "evaluation_repairs" / digest
    )


def test_main_rejects_any_generation_inventory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = _inventory()
    changed["inventory_sha256"] = "2" * 64
    with pytest.raises(repair.PhaseResultsError, match="changed frozen generation"):
        _run_main(tmp_path, monkeypatch, [_inventory(), changed])
