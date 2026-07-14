from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from safa.evaluation import r9_confirm512_report_only_supersession as v2
from safa.evaluation.r9_campaign_contracts import write_immutable_contract


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_r9_confirm512_report_only_supersession.py"
SPEC = importlib.util.spec_from_file_location(
    "run_r9_confirm512_report_only_supersession", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _arcface_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(v2.EXPECTED_SAMPLE_COUNT):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "source_face_count": 1,
                "native_face_count": 1,
                "candidate_face_count": 1,
                "source_candidate_cosine": 0.1 + index / 100000.0,
                "source_native_cosine": 0.09,
            }
        )
    rows[-1] = {
        "sample_id": v2.UNIQUE_ARCFACE_MISS,
        "source_face_count": 1,
        "native_face_count": 0,
        "candidate_face_count": 1,
    }
    return rows


def test_complete_case_privacy_is_511_report_only_bootstrap() -> None:
    result = v2._complete_case_privacy(_arcface_rows(), bootstrap_seed=91637)
    assert result["role"] == "report_only_complete_case"
    assert result["observation_count"] == 511
    assert result["excluded_sample_ids"] == [v2.UNIQUE_ARCFACE_MISS]
    bootstrap = result["bootstrap"]
    assert bootstrap["iterations"] == 10000
    assert bootstrap["observation_count"] == 511
    assert bootstrap["upper_95_one_sided"] > bootstrap["mean_delta"]


def test_complete_case_rejects_any_other_missing_face() -> None:
    rows = _arcface_rows()
    rows[-1]["sample_id"] = "different"
    with pytest.raises(v2.Confirm512SupersessionError, match="unique-miss"):
        v2._complete_case_privacy(rows, bootstrap_seed=91637)


@dataclass(frozen=True)
class _Phase:
    phase_root: Path


def test_visual_builder_materializes_exact_two_full_512_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SimpleNamespace(
        phase_request=_Phase(tmp_path / "old"),
        validated_phase_request={"request": _Phase(tmp_path / "old")},
        canonical_runs={
            arm_id: {
                "logical_run_id": arm_id,
                "arm_id": arm_id,
                "seed": v2.EXPECTED_SEED,
                "repeat_index": None,
                "rows": [{"sample_id": f"s-{index}"} for index in range(512)],
            }
            for arm_id in v2.EXPECTED_ARMS[1:]
        },
    )
    request = SimpleNamespace(source=source)
    prepared = SimpleNamespace(
        namespace_root=tmp_path / "new",
        source=source,
        request=request,
    )
    seen = []

    def fake_materialize(validated, run):
        seen.append((validated["request"].phase_root, run))
        return {
            "unit_id": run["logical_run_id"],
            "arm_id": run["arm_id"],
            "evidence_path": str(tmp_path / f"{run['arm_id']}.json"),
            "evidence_contract_sha256": "1" * 64,
            "review_path": str(tmp_path / f"{run['arm_id']}.review.json"),
        }

    monkeypatch.setattr(v2, "_materialize_visual_unit", fake_materialize)
    units = v2._materialize_visual_evidence(prepared)
    assert len(units) == 2
    assert {unit["arm_id"] for unit in units} == set(v2.EXPECTED_ARMS[1:])
    assert all(root == tmp_path / "new" / "confirm512" for root, _ in seen)
    assert all(len(run["rows"]) == 512 for _, run in seen)


def test_rank_uses_severe_before_kid_fid_edev_e0_arm_id() -> None:
    flow = {
        "arm_id": "flow",
        "quality": {"kid": 0.01, "fid": 70.0},
        "representation": {"delta_edev": 0.2, "e0": 0.8},
    }
    paper = {
        "arm_id": "paper",
        "quality": {"kid": 0.02, "fid": 72.0},
        "representation": {"delta_edev": 0.3, "e0": 0.9},
    }
    assert v2._post_hoc_rank(paper, 0) < v2._post_hoc_rank(flow, 1)
    assert v2._post_hoc_rank(flow, 0) < v2._post_hoc_rank(paper, 0)


def test_finalize_stays_awaiting_until_both_reviews_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "v2"
    units = [
        {
            "unit_id": arm,
            "arm_id": arm,
            "evidence_path": str(namespace / f"{arm}.evidence.json"),
            "evidence_contract_sha256": "1" * 64,
            "review_path": str(namespace / f"{arm}.review.json"),
        }
        for arm in v2.EXPECTED_ARMS[1:]
    ]
    automatic = {
        "schema_version": 2,
        "contract_type": "safa_r9_confirm512_report_only_automatic_v2",
        "visual_units": units,
    }
    automatic["automatic_evidence_sha256"] = v2._canonical_digest(
        automatic, "automatic_evidence_sha256"
    )
    write_immutable_contract(
        namespace / "confirm512" / "automatic_evidence_v2.json",
        automatic,
        digest_field="automatic_evidence_sha256",
    )
    monkeypatch.setattr(
        v2,
        "materialize_visual_stage",
        lambda _: {
            "status": "awaiting_visual_review",
            "awaiting_visual_review_sha256": "2" * 64,
        },
    )
    prepared = SimpleNamespace(namespace_root=namespace)
    result = v2.finalize_report_only_selection(prepared)
    assert result["status"] == "awaiting_visual_review"
    assert result["bounded_exit_code"] == 20
    assert len(result["missing_review_paths"]) == 2
    assert not (namespace / "selection.json").exists()


def test_runner_dry_run_executes_no_materializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = SimpleNamespace(
        contract_sha256="3" * 64,
        namespace_root=tmp_path / "v2",
        contract={
            "generation_inventory_sha256": "4" * 64,
            "evaluator_results": [{}, {}, {}, {}, {}],
        },
    )
    monkeypatch.setattr(runner, "_build", lambda _: prepared)
    monkeypatch.setattr(
        runner,
        "materialize_visual_stage",
        lambda _: pytest.fail("dry run materialized visual evidence"),
    )
    assert (
        runner.main(
            [
                "--campaign-id",
                "r9-report-only-formal-v8",
                "--source-repair-id",
                "canonical-native-v1",
                "--source-repair-sha256",
                v2.SOURCE_REPAIR_SHA256,
                "--supersession-id",
                "report-only-v2",
                "--phase",
                "prepare",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run_validated"
    assert payload["bound_evaluator_result_count"] == 5
    assert payload["generation_execution_count"] == 0
    assert payload["evaluator_execution_count"] == 0


def test_finalize_locks_winner_only_after_both_complete_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "v2"
    units = []
    review_by_evidence = {}
    for index, arm in enumerate(v2.EXPECTED_ARMS[1:]):
        evidence_path = namespace / f"{arm}.evidence.json"
        review_path = namespace / f"{arm}.review.json"
        evidence = {"test_severe_count": 1 - index}
        evidence["evidence_contract_sha256"] = v2._canonical_digest(
            evidence, "evidence_contract_sha256"
        )
        write_immutable_contract(
            evidence_path, evidence, digest_field="evidence_contract_sha256"
        )
        review_path.write_text("{}", encoding="utf-8")
        review_by_evidence[str(evidence_path)] = "a" * 64
        units.append(
            {
                "unit_id": arm,
                "arm_id": arm,
                "evidence_path": str(evidence_path),
                "evidence_contract_sha256": evidence["evidence_contract_sha256"],
                "review_path": str(review_path),
            }
        )
    automatic = {
        "schema_version": 2,
        "contract_type": "safa_r9_confirm512_report_only_automatic_v2",
        "visual_units": units,
    }
    automatic["automatic_evidence_sha256"] = v2._canonical_digest(
        automatic, "automatic_evidence_sha256"
    )
    write_immutable_contract(
        namespace / "confirm512" / "automatic_evidence_v2.json",
        automatic,
        digest_field="automatic_evidence_sha256",
    )
    monkeypatch.setattr(
        v2,
        "materialize_visual_stage",
        lambda _: {
            "status": "awaiting_visual_review",
            "awaiting_visual_review_sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(
        v2,
        "validate_visual_review",
        lambda review_path, evidence_path: {
            "review_sha256": review_by_evidence[str(evidence_path)]
        },
    )
    monkeypatch.setattr(
        v2,
        "derive_visual_arm_pass",
        lambda review, evidence, severe_limit: {
            "reviewed_sample_count": v2.EXPECTED_SAMPLE_COUNT,
            "severe_count": evidence["test_severe_count"],
        },
    )
    arms = (
        {
            "arm_id": "flow_map2_normalized_eta_0p125",
            "quality": {"kid": 0.01, "fid": 70.0},
            "representation": {"delta_edev": 0.3, "e0": 0.8},
            "complete_case_privacy": {"bootstrap": {"bootstrap_sha256": "b" * 64}},
            "config_sha256": "c" * 64,
            "source_generation_output_sha256": "d" * 64,
            "canonical_evidence_binding_sha256": "e" * 64,
            "evaluator_evidence_sha256": "f" * 64,
        },
        {
            "arm_id": "paper_eta_0p125",
            "quality": {"kid": 0.02, "fid": 72.0},
            "representation": {"delta_edev": 0.2, "e0": 0.9},
            "complete_case_privacy": {"bootstrap": {"bootstrap_sha256": "1" * 64}},
            "config_sha256": "2" * 64,
            "source_generation_output_sha256": "3" * 64,
            "canonical_evidence_binding_sha256": "4" * 64,
            "evaluator_evidence_sha256": "5" * 64,
        },
    )
    prepared = SimpleNamespace(
        namespace_root=namespace,
        request=SimpleNamespace(campaign_id="campaign"),
        contract_sha256="6" * 64,
        contract={
            "policy": {
                "coverage_role": "report_only",
                "numerical_metrics_role": "report_only",
                "privacy_metrics_role": "report_only_complete_case",
                "visual_metrics_role": "report_only_full_512_required_before_selection",
                "ranking": [
                    "severe_count",
                    "kid",
                    "fid",
                    "-delta_edev",
                    "-e0",
                    "arm_id",
                ],
                "reselection_allowed": False,
            },
            "generation_inventory_sha256": "7" * 64,
        },
        source=SimpleNamespace(phase_request=SimpleNamespace(manifest_sha256="8" * 64)),
        arms=arms,
    )
    result = v2.finalize_report_only_selection(prepared)
    assert result["winner_arm_id"] == "paper_eta_0p125"
    assert result["generation_execution_count"] == 0
    assert result["evaluator_execution_count"] == 0
    selection = json.loads((namespace / "selection.json").read_text())
    assert selection["winner"]["arm_id"] == "paper_eta_0p125"
    assert selection["next_stage"] == "new_v9_full_continuation_required"
