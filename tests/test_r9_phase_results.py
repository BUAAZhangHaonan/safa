from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
import pytest

from safa.evaluation.r8_visual_evidence import (
    build_visual_evidence_contract,
    write_contact_sheets,
)
from safa.evaluation.r9_campaign_contracts import (
    build_a_gate_contract,
    write_immutable_contract,
)
from safa.evaluation.r9_phase_results import (
    PhaseResultsError,
    PhaseResultsRequest,
    RunEvidenceSpec,
    SampleEvidence,
    _algorithm_config_digest,
    _asset_manifest_digest,
    _canonical_digest,
    _evaluate_arcface,
    _evaluate_quality,
    _load_run_evidence,
    _materialize_heldout,
    _normalize_heldout_raw,
    _request_context,
    _sample_evidence,
    _validate_automatic_context,
    _validate_request,
    _write_exclusive_json,
    submit_visual_review,
    validate_interval_diagnostics,
    validate_visual_review,
)


SHA = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _write_source_index(
    root: Path,
    sample_ids: list[str] | tuple[str, ...],
    *,
    paths: Mapping[str, Path] | None = None,
) -> Path:
    rows = []
    for index, sample_id in enumerate(sample_ids):
        if paths is None:
            source = root / "source_images" / f"source_{index}.png"
            _write_png(source, (index % 255, 2, 3))
        else:
            source = paths[sample_id]
        rows.append({"sample_id": sample_id, "image_path": str(source.resolve())})
    index_path = root / "source_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return index_path


def _visual_evidence(tmp_path: Path, count: int = 3) -> tuple[Path, list[str]]:
    sample_ids = [f"sample_{index}" for index in range(count)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"sample_id": sample_id}) + "\n" for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    rows = []
    for index, sample_id in enumerate(sample_ids):
        paths = {}
        for role, offset in (("source", 0), ("native", 20), ("candidate", 40)):
            path = tmp_path / "images" / f"{sample_id}_{role}.png"
            _write_png(path, (index * 10 + offset, 30, 60))
            paths[role] = str(path)
        rows.append({"sample_id": sample_id, **paths})
    pages = write_contact_sheets(
        tmp_path / "pages",
        rows,
        columns=("source", "native", "candidate"),
    )
    evidence = build_visual_evidence_contract(
        manifest_path=manifest,
        rows=rows,
        pages=pages,
        columns=("source", "native", "candidate"),
        expected_count=count,
    )
    evidence_path = tmp_path / "visual_evidence.json"
    write_immutable_contract(
        evidence_path, evidence, digest_field="evidence_contract_sha256"
    )
    return evidence_path, sample_ids


def test_visual_review_is_severe_only_exact_coverage_and_o_excl(
    tmp_path: Path,
) -> None:
    evidence_path, sample_ids = _visual_evidence(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    decisions_path = tmp_path / "decisions.json"
    _write_json(
        decisions_path,
        {
            "evidence_contract_sha256": evidence["evidence_contract_sha256"],
            "samples": [
                {"sample_id": sample_id, "severe": index == 1}
                for index, sample_id in enumerate(sample_ids)
            ],
        },
    )
    output_path = tmp_path / "review.json"
    review = submit_visual_review(
        evidence_path=evidence_path,
        decisions_path=decisions_path,
        output_path=output_path,
    )
    assert validate_visual_review(output_path, evidence_path) == review
    assert not ({"passed", "severe_count", "failures", "verdict"} & set(review))
    with pytest.raises(FileExistsError):
        submit_visual_review(
            evidence_path=evidence_path,
            decisions_path=decisions_path,
            output_path=output_path,
        )


@pytest.mark.parametrize("mutation", ["reordered", "missing", "derived"])
def test_visual_review_rejects_noncanonical_input(
    tmp_path: Path, mutation: str
) -> None:
    evidence_path, sample_ids = _visual_evidence(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = [
        {"sample_id": sample_id, "severe": False} for sample_id in sample_ids
    ]
    if mutation == "reordered":
        rows.reverse()
    elif mutation == "missing":
        rows.pop()
    else:
        rows[0]["passed"] = True
    decisions = tmp_path / "decisions.json"
    _write_json(
        decisions,
        {
            "evidence_contract_sha256": evidence["evidence_contract_sha256"],
            "samples": rows,
        },
    )
    with pytest.raises((PhaseResultsError, ValueError)):
        submit_visual_review(
            evidence_path=evidence_path,
            decisions_path=decisions,
            output_path=tmp_path / "review.json",
        )


def test_visual_review_resume_rehashes_assets(tmp_path: Path) -> None:
    evidence_path, sample_ids = _visual_evidence(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    decisions = tmp_path / "decisions.json"
    _write_json(
        decisions,
        {
            "evidence_contract_sha256": evidence["evidence_contract_sha256"],
            "samples": [
                {"sample_id": sample_id, "severe": False} for sample_id in sample_ids
            ],
        },
    )
    review_path = tmp_path / "review.json"
    submit_visual_review(
        evidence_path=evidence_path,
        decisions_path=decisions,
        output_path=review_path,
    )
    Path(evidence["samples"][0]["assets"]["source"]["path"]).write_bytes(b"tamper")
    with pytest.raises(PhaseResultsError, match="replaced"):
        validate_visual_review(review_path, evidence_path)


def _config(*, collect: bool, intervals: list[str]) -> dict[str, Any]:
    return {
        "mode": "paper_split",
        "sample_mode": "paper_split",
        "optimization_mode": "gradient_descent",
        "num_optim_iters": 1,
        "step_size": 0.25,
        "active_guidance_intervals": intervals,
        "collect_interval_diagnostics": collect,
        "determinism_policy_sha256": "b" * 64,
        "attention_backend": "native",
        "locked_schedule": {"schedule_contract_sha256": "c" * 64},
        "r9_semigroup_gate_contract_sha256": "d" * 64,
        "r9_campaign_runtime_sha256": "e" * 64,
        "r9_manifest_contracts_sha256": "f" * 64,
        "r9_phase_manifest_sha256": "1" * 64,
        "sampling_seed": 1337,
    }


def _make_run(
    root: Path,
    name: str,
    *,
    collect: bool,
    operational_tag: str,
    intervals: list[str] | None = None,
    source_path: Path | None = None,
) -> tuple[dict[str, Any], RunEvidenceSpec]:
    output = root / name
    output.mkdir(parents=True)
    source = source_path if source_path is not None else root / "source.png"
    if not source.exists():
        _write_png(source, (1, 2, 3))
    native = output / "native.png"
    candidate = output / "candidate.png"
    _write_png(native, (4, 5, 6))
    _write_png(candidate, (7, 8, 9))
    config = _config(
        collect=collect,
        intervals=intervals or ["I1", "I2", "I3"],
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint": {"sha256": SHA},
        "config": config,
        "arm_config_sha256": "2" * 64,
        "shard": {"index": 0, "count": 1},
        "operational": {"worker": operational_tag, "rss": len(name)},
    }
    _write_json(output / "generation_result.json", result)
    _write_json(output / "run_manifest.json", result)
    _write_json(output / "completion.json", {"status": "complete", "rss": len(name)})
    metrics: dict[str, Any] = {
        "sample_id": "sample_0",
        "source": str(source),
        "native": str(native),
        "generated": str(candidate),
        "candidate_cosine": 0.8,
        "native_cosine": 0.4,
        "edev_cosine": 0.7,
        "native_edev_cosine": 0.6,
        "candidate_nfe": 3,
        "native_nfe": 1,
        "candidate_algorithm_nfe": 3,
        "candidate_trace": [0.8, 0.7, 0.6],
        "native_trace": [0.4],
    }
    if collect:
        metrics.update(
            {
                "candidate_diagnostic_nfe": 2,
                "candidate_diagnostic_trace": [0.11, 0.12],
                "route_diagnostics": {
                    "interval_diagnostics": {
                        interval: {"correction_norm": 0.1}
                        for interval in ("I1", "I2", "I3")
                    }
                },
            }
        )
    (output / "per_sample.jsonl").write_text(
        json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    source_index = _write_source_index(root, ("sample_0",), paths={"sample_0": source})
    request = PhaseResultsRequest(
        repo_root=root,
        phase_root=root / "phase",
        phase="diagnose",
        campaign_id="campaign",
        campaign_runtime_sha256="e" * 64,
        manifest_contracts_sha256="f" * 64,
        manifest_path=root / "unused.jsonl",
        manifest_sha256="1" * 64,
        source_index_path=source_index,
        source_index_sha256=_sha256(source_index),
        checkpoint_sha256=SHA,
        bootstrap_seed=5,
        runs=(
            RunEvidenceSpec(
                logical_run_id=name,
                arm_id="arm",
                family="paper_split",
                seed=1337,
                repeat_index=0,
                shard_output_dirs=(output,),
            ),
        ),
        expected_candidate_arm_ids=("arm",),
        expected_seeds=(1337,),
    )
    spec = request.runs[0]
    validated = {
        "request": request,
        "repo_root": root,
        "manifest_ids": ["sample_0"],
        "source_paths": {"sample_0": source.resolve()},
    }
    return _load_run_evidence(validated, spec), spec


def test_semantic_digests_ignore_paths_and_operational_metadata(tmp_path: Path) -> None:
    first, _ = _make_run(tmp_path, "run_a", collect=True, operational_tag="worker-a")
    second, _ = _make_run(tmp_path, "run_b", collect=True, operational_tag="worker-b")
    assert first["algorithm_config_sha256"] == second["algorithm_config_sha256"]
    assert first["semantic_run_sha256"] == second["semantic_run_sha256"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["evidence_binding_sha256"] != second["evidence_binding_sha256"]


def test_external_source_is_accepted_only_through_locked_index(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_source = tmp_path / "AffectNet" / "external.png"
    _write_png(external_source, (11, 12, 13))
    run, _ = _make_run(
        repo_root,
        "external_source_run",
        collect=False,
        operational_tag="worker",
        source_path=external_source,
    )
    assert Path(str(run["rows"][0]["source"])) == external_source.resolve()
    assert run["rows"][0]["source_sha256"] == _sha256(external_source)


def test_sample_evidence_accepts_real_r8_id_and_rejects_invalid_id(
    tmp_path: Path,
) -> None:
    paths = []
    for role, color in (("source", 1), ("native", 2), ("candidate", 3)):
        path = tmp_path / f"{role}.png"
        _write_png(path, (color, 4, 5))
        paths.append(path)
    evidence = SampleEvidence(
        sample_id="val:Manually_Annotated_Images/0000001.jpg",
        source=paths[0],
        native=paths[1],
        candidate=paths[2],
        source_sha256=_sha256(paths[0]),
        native_sha256=_sha256(paths[1]),
        candidate_sha256=_sha256(paths[2]),
    )
    assert evidence.sample_id.startswith("val:")
    with pytest.raises(PhaseResultsError, match="non-empty"):
        replace(evidence, sample_id="")
    with pytest.raises(PhaseResultsError, match="without NUL"):
        replace(evidence, sample_id="val:\0broken")


def test_diagnostics_toggle_is_not_algorithmic_but_changes_output(
    tmp_path: Path,
) -> None:
    without, _ = _make_run(
        tmp_path, "without_diagnostics", collect=False, operational_tag="worker"
    )
    with_diagnostics, _ = _make_run(
        tmp_path, "with_diagnostics", collect=True, operational_tag="worker"
    )
    assert (
        without["algorithm_config_sha256"]
        == with_diagnostics["algorithm_config_sha256"]
    )
    assert without["semantic_run_sha256"] != with_diagnostics["semantic_run_sha256"]
    assert without["output_sha256"] != with_diagnostics["output_sha256"]


def test_active_interval_mask_changes_algorithm_digest() -> None:
    full = _algorithm_config_digest(
        _config(collect=False, intervals=["I1", "I2", "I3"]), SHA
    )
    ablated = _algorithm_config_digest(
        _config(collect=True, intervals=["I2", "I3"]), SHA
    )
    assert full != ablated


def test_algorithm_digest_excludes_seed_but_binds_fixed_asset_digests() -> None:
    first = _config(collect=False, intervals=["I1", "I2", "I3"])
    second = dict(first)
    second["sampling_seed"] = 2027
    assert _algorithm_config_digest(first, SHA) == _algorithm_config_digest(second, SHA)
    second["e0_sha256"] = "9" * 64
    assert _algorithm_config_digest(first, SHA) != _algorithm_config_digest(second, SHA)


def _diagnostic_fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm_trace = [{"t": 1.0, "r": 0.75, "kind": "paper_algorithm_split"}]
    diagnostic_trace = [{"t": 1.0, "r": 0.0, "kind": "interval_diagnostic"}]
    contract = {
        "schema_version": 1,
        "mode": "paper_algorithm_split",
        "active_guidance_intervals": ["I1", "I3"],
        "collect_interval_diagnostics": True,
        "expected_algorithm_nfe": 1,
        "expected_diagnostic_nfe": 1,
        "expected_algorithm_trace": algorithm_trace,
        "expected_diagnostic_trace": diagnostic_trace,
    }
    intervals = {}
    for interval_id, t, s in (
        ("I1", 1.0, 0.75),
        ("I2", 0.75, 0.5),
        ("I3", 0.5, 0.25),
    ):
        active = interval_id in {"I1", "I3"}
        intervals[interval_id] = {
            "interval_id": interval_id,
            "active": active,
            "t": t,
            "s": s,
            "loss_before_correction": 0.4,
            "loss_after_correction": 0.3 if active else 0.4,
            "gradient_norm": 0.2 if active else 0.0,
            "velocity_norm": 0.5,
            "transport_norm": 0.4,
            "correction_norm": 0.1 if active else 0.0,
            "correction_transport_ratio": 0.25 if active else 0.0,
            "gradient_velocity_cosine": 0.1 if active else 0.0,
            "local_semigroup_residual": 0.05,
        }
    route = {
        "active_guidance_intervals": ["I1", "I3"],
        "interval_diagnostics_enabled": True,
        "interval_diagnostics": intervals,
        "algorithm_nfe": 1,
        "diagnostic_nfe": 1,
        "diagnostic_flow_map_trace": diagnostic_trace,
        "mode": "paper_algorithm_split",
        "flow_map_trace": algorithm_trace,
    }
    rows = [
        {
            "sample_id": "sample_0",
            "metrics": {
                "candidate_nfe": 1,
                "candidate_algorithm_nfe": 1,
                "candidate_diagnostic_nfe": 1,
                "candidate_trace": algorithm_trace,
                "candidate_diagnostic_trace": diagnostic_trace,
                "route_diagnostics": route,
            },
        }
    ]
    return rows, contract


def test_interval_diagnostics_strict_contract_and_nonfinite_rejection() -> None:
    rows, contract = _diagnostic_fixture()
    assert validate_interval_diagnostics(rows, contract)["diagnostics_contract_sha256"]
    rows[0]["metrics"]["route_diagnostics"]["interval_diagnostics"]["I1"][
        "gradient_norm"
    ] = float("nan")
    with pytest.raises(PhaseResultsError, match="finite"):
        validate_interval_diagnostics(rows, contract)


def _diagnose_request(tmp_path: Path) -> PhaseResultsRequest:
    manifest = tmp_path / "diagnose.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"sample_{index}",
                    "role": "difficult" if index < 9 else "control",
                }
            )
            + "\n"
            for index in range(18)
        ),
        encoding="utf-8",
    )
    arm_ids = tuple(f"arm_{index:02d}" for index in range(12))
    source_index = _write_source_index(
        tmp_path, tuple(f"sample_{index}" for index in range(18))
    )
    runs = [
        RunEvidenceSpec(
            logical_run_id=f"native_r{repeat}",
            arm_id="native",
            family="native",
            seed=1337,
            repeat_index=repeat,
            shard_output_dirs=(tmp_path / "outputs" / f"native_r{repeat}",),
        )
        for repeat in range(3)
    ]
    runs.extend(
        RunEvidenceSpec(
            logical_run_id=f"{arm_id}_r{repeat}",
            arm_id=arm_id,
            family="flow_map2",
            seed=1337,
            repeat_index=repeat,
            shard_output_dirs=(tmp_path / "outputs" / f"{arm_id}_r{repeat}",),
        )
        for arm_id in arm_ids
        for repeat in range(3)
    )
    return PhaseResultsRequest(
        repo_root=tmp_path,
        phase_root=tmp_path / "phase",
        phase="diagnose",
        campaign_id="campaign",
        campaign_runtime_sha256="a" * 64,
        manifest_contracts_sha256="b" * 64,
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        source_index_path=source_index,
        source_index_sha256=_sha256(source_index),
        checkpoint_sha256="c" * 64,
        bootstrap_seed=9,
        runs=tuple(runs),
        expected_candidate_arm_ids=arm_ids,
        expected_seeds=(1337,),
    )


def test_request_run_plan_rejects_missing_and_extra_runs(tmp_path: Path) -> None:
    request = _diagnose_request(tmp_path)
    validated = _validate_request(request)
    assert validated["run_plan_sha256"]
    with pytest.raises(PhaseResultsError, match="missing, stale, or extra"):
        _validate_request(replace(request, runs=request.runs[:-1]))
    duplicate = replace(
        request.runs[-1],
        logical_run_id="extra_logical_run",
    )
    with pytest.raises(PhaseResultsError, match="repeats a candidate"):
        _validate_request(replace(request, runs=(*request.runs, duplicate)))


def test_automatic_resume_rejects_stale_run_plan(tmp_path: Path) -> None:
    request = _diagnose_request(tmp_path)
    validated = _validate_request(request)
    automatic = {
        "phase": "diagnose",
        "campaign_id": request.campaign_id,
        "context": _request_context(request),
        "run_plan": {"stale": True},
        "run_plan_sha256": "0" * 64,
    }
    with pytest.raises(PhaseResultsError, match="run plan mismatch"):
        _validate_automatic_context(automatic, validated)


def _a_gate(selected_arm_id: str) -> dict[str, Any]:
    context = {
        "campaign_id": "campaign",
        "campaign_runtime_sha256": "a" * 64,
        "manifest_contracts_sha256": "b" * 64,
        "manifest_sha256": "d" * 64,
        "checkpoint_sha256": "c" * 64,
        "phase_results_sha256": "e" * 64,
        "automatic_evidence_sha256": "f" * 64,
        "run_plan_sha256": "1" * 64,
        "evaluator_evidence_sha256": "2" * 64,
    }
    arm = {
        "arm_id": selected_arm_id,
        "family": "flow_map2",
        "config_sha256": "3" * 64,
        "output_sha256": "4" * 64,
        "repeat_results": [
            {
                "repeat_index": repeat,
                "run_sha256": "5" * 64,
                "difficult_severe_count": 0,
                "control_severe_count": 0,
                "e0_mean": 0.8,
                "edev_delta_vs_matched_native": 0.1,
                "diagnostics_finite": True,
            }
            for repeat in range(3)
        ],
    }
    diagnose = {
        "path": "diagnose.jsonl",
        "sha256": "d" * 64,
        "sample_count": 18,
        "ordered_sample_id_sha256": "6" * 64,
        "difficult_count": 9,
        "control_count": 9,
        "matched_pair_sha256": "7" * 64,
    }
    return build_a_gate_contract(context, [arm], diagnose_manifest=diagnose)


def test_upstream_selected_arms_must_equal_run_plan(tmp_path: Path) -> None:
    manifest = tmp_path / "calibration.jsonl"
    manifest.write_text(
        "".join(json.dumps({"sample_id": f"s{index}"}) + "\n" for index in range(64)),
        encoding="utf-8",
    )
    source_index = _write_source_index(
        tmp_path, tuple(f"s{index}" for index in range(64))
    )
    runs = []
    for seed in (1337, 2027, 3407):
        runs.extend(
            [
                RunEvidenceSpec(
                    logical_run_id=f"native_{seed}",
                    arm_id="native",
                    family="native",
                    seed=seed,
                    repeat_index=None,
                    shard_output_dirs=(tmp_path / f"native_{seed}",),
                ),
                RunEvidenceSpec(
                    logical_run_id=f"planned_{seed}",
                    arm_id="planned",
                    family="flow_map2",
                    seed=seed,
                    repeat_index=None,
                    shard_output_dirs=(tmp_path / f"planned_{seed}",),
                ),
            ]
        )
    request = PhaseResultsRequest(
        repo_root=tmp_path,
        phase_root=tmp_path / "phase",
        phase="calibrate",
        campaign_id="campaign",
        campaign_runtime_sha256="a" * 64,
        manifest_contracts_sha256="b" * 64,
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        source_index_path=source_index,
        source_index_sha256=_sha256(source_index),
        checkpoint_sha256="c" * 64,
        bootstrap_seed=9,
        runs=tuple(runs),
        expected_candidate_arm_ids=("planned",),
        expected_seeds=(1337, 2027, 3407),
        upstream_gate=_a_gate("selected_other"),
    )
    with pytest.raises(PhaseResultsError, match="upstream selected arms"):
        _validate_request(request)


def _heldout_raw(sample_ids: list[str], *, coverage: int) -> dict[str, Any]:
    paired = [
        {"sample_id": sample_id, "native_cosine": 0.2, "winner_cosine": 0.1}
        for sample_id in sample_ids
    ]
    failures = sample_ids[coverage:]
    recognizer = {
        "source_exact_one_count": coverage,
        "native_exact_one_count": coverage,
        "winner_exact_one_count": coverage,
        "paired_exact_one_count": coverage,
        "failure_sample_ids": failures,
        "rows": paired if coverage == len(sample_ids) else [],
    }
    available_role = {
        "status": "available",
        "tar_at_far": {"0.001": 0.2, "0.0001": 0.1},
        "eer": 0.3,
        "auc": 0.7,
    }
    unavailable_reason = "incomplete_exact_one_coverage"
    identity_recognizer = (
        {
            "status": "available",
            "reason": None,
            "coverage": len(sample_ids),
            "roles": {
                "native": dict(available_role),
                "winner": dict(available_role),
            },
        }
        if coverage == len(sample_ids)
        else {
            "status": "unavailable",
            "reason": unavailable_reason,
            "coverage": coverage,
            "roles": {
                role: {"status": "unavailable", "reason": unavailable_reason}
                for role in ("native", "winner")
            },
        }
    )
    return {
        "representations": {"e1": paired, "e2": paired},
        "recognizers": {"facenet": recognizer, "adaface": recognizer},
        "identity_report": {
            "schema_version": 1,
            "recognizers": {
                name: identity_recognizer for name in ("arcface", "facenet", "adaface")
            },
        },
    }


def test_full_incomplete_recognizer_coverage_is_failure_evidence_not_error() -> None:
    sample_ids = ["a", "b", "c"]
    arcface_summary = {
        "source_exact_one_count": 2,
        "native_exact_one_count": 2,
        "candidate_exact_one_count": 2,
        "paired_exact_one_count": 2,
        "failure_sample_ids": ["c"],
    }
    heldout = _normalize_heldout_raw(
        _heldout_raw(sample_ids, coverage=2),
        manifest_ids=sample_ids,
        seed=7,
        bootstrap_seed=11,
        raw_sha256=SHA,
        arcface_summary=arcface_summary,
        arcface_bootstrap=None,
    )
    for recognizer in heldout["recognizers"].values():
        assert recognizer["coverage"] == 2
        assert recognizer["failure_sample_ids"] == ["c"]
        assert recognizer["privacy_delta_upper_95"] is None
        assert recognizer["bootstrap_sha256"] is None


def test_full_arcface_unavailable_keeps_other_identity_reports() -> None:
    sample_ids = ["a", "b", "c"]
    raw = _heldout_raw(sample_ids, coverage=3)
    reason = "incomplete_exact_one_coverage"
    raw["identity_report"]["recognizers"]["arcface"] = {
        "status": "unavailable",
        "reason": reason,
        "coverage": 2,
        "roles": {
            role: {"status": "unavailable", "reason": reason}
            for role in ("native", "winner")
        },
    }
    arcface_summary = {
        "source_exact_one_count": 2,
        "native_exact_one_count": 2,
        "candidate_exact_one_count": 2,
        "paired_exact_one_count": 2,
        "failure_sample_ids": ["c"],
    }
    heldout = _normalize_heldout_raw(
        raw,
        manifest_ids=sample_ids,
        seed=7,
        bootstrap_seed=11,
        raw_sha256=SHA,
        arcface_summary=arcface_summary,
        arcface_bootstrap=None,
    )
    report = heldout["identity_report"]["recognizers"]
    assert report["arcface"]["status"] == "unavailable"
    assert report["facenet"]["status"] == "available"
    assert report["adaface"]["status"] == "available"


def test_full_identity_report_rejects_empty_or_noncanonical_metrics() -> None:
    sample_ids = ["a", "b", "c"]
    raw = _heldout_raw(sample_ids, coverage=3)
    arcface_summary = {
        "source_exact_one_count": 3,
        "native_exact_one_count": 3,
        "candidate_exact_one_count": 3,
        "paired_exact_one_count": 3,
        "failure_sample_ids": [],
    }
    for report in ({}, {"schema_version": 1, "recognizers": {}}):
        mutated = {**raw, "identity_report": report}
        with pytest.raises(PhaseResultsError, match="identity report contract"):
            _normalize_heldout_raw(
                mutated,
                manifest_ids=sample_ids,
                seed=7,
                bootstrap_seed=11,
                raw_sha256=SHA,
                arcface_summary=arcface_summary,
                arcface_bootstrap={
                    "upper_95_one_sided": 0.01,
                    "bootstrap_sha256": "6" * 64,
                },
            )


def test_full_incomplete_recognizer_forbids_partial_cosines() -> None:
    sample_ids = ["a", "b", "c"]
    raw = _heldout_raw(sample_ids, coverage=2)
    raw["recognizers"]["facenet"]["rows"] = [
        {"sample_id": "a", "native_cosine": 0.2, "winner_cosine": 0.1}
    ]
    arcface_summary = {
        "source_exact_one_count": 2,
        "native_exact_one_count": 2,
        "candidate_exact_one_count": 2,
        "paired_exact_one_count": 2,
        "failure_sample_ids": ["c"],
    }
    with pytest.raises(PhaseResultsError, match="forbids partial cosine"):
        _normalize_heldout_raw(
            raw,
            manifest_ids=sample_ids,
            seed=7,
            bootstrap_seed=11,
            raw_sha256=SHA,
            arcface_summary=arcface_summary,
            arcface_bootstrap=None,
        )


def _evaluation_request(root: Path, run: Mapping[str, Any]) -> PhaseResultsRequest:
    source = Path(str(run["rows"][0]["source"]))
    source_index = _write_source_index(root, ("sample_0",), paths={"sample_0": source})
    return PhaseResultsRequest(
        repo_root=root,
        phase_root=root / "phase",
        phase="diagnose",
        campaign_id="campaign",
        campaign_runtime_sha256="e" * 64,
        manifest_contracts_sha256="f" * 64,
        manifest_path=root / "unused.jsonl",
        manifest_sha256="1" * 64,
        source_index_path=source_index,
        source_index_sha256=_sha256(source_index),
        checkpoint_sha256=SHA,
        bootstrap_seed=5,
        runs=(
            RunEvidenceSpec(
                logical_run_id=str(run["logical_run_id"]),
                arm_id="arm",
                family="flow_map2",
                seed=1337,
                repeat_index=0,
                shard_output_dirs=(root / "unused",),
            ),
        ),
        expected_candidate_arm_ids=("arm",),
        expected_seeds=(1337,),
    )


def test_arcface_rejects_negative_face_count(tmp_path: Path) -> None:
    run, _ = _make_run(tmp_path, "arcface_run", collect=False, operational_tag="worker")
    request = _evaluation_request(tmp_path, run)
    with pytest.raises(PhaseResultsError, match="non-negative"):
        _evaluate_arcface(
            request,
            run,
            _sample_evidence(run),
            lambda evaluation: [
                {
                    "sample_id": evaluation.samples[0].sample_id,
                    "source_face_count": -1,
                    "native_face_count": 1,
                    "candidate_face_count": 1,
                }
            ],
        )


@pytest.mark.parametrize(
    "mutation", ["asset", "sample", "generation_set", "per_sample_set"]
)
def test_quality_rejects_wrong_provenance(tmp_path: Path, mutation: str) -> None:
    run, _ = _make_run(
        tmp_path, f"quality_{mutation}", collect=False, operational_tag="worker"
    )
    request = _evaluation_request(tmp_path, run)

    def evaluator(evaluation) -> dict[str, Any]:
        sample_ids = [sample.sample_id for sample in evaluation.samples]
        binding = {
            "schema_version": 1,
            "algorithm_config_sha256": evaluation.algorithm_config_sha256,
            "runner_arm_config_sha256": evaluation.runner_arm_config_sha256,
            "semantic_output_sha256": evaluation.semantic_output_sha256,
            "evidence_binding_sha256": evaluation.evidence_binding_sha256,
            "generation_result_set_sha256": evaluation.generation_result_set_sha256,
            "per_sample_set_sha256": evaluation.per_sample_set_sha256,
            "manifest_sha256": request.manifest_sha256,
            "source_index_path": str(request.source_index_path.resolve()),
            "source_index_sha256": request.source_index_sha256,
            "ordered_sample_id_sha256": hashlib.sha256(
                "".join(f"{sample_id}\n" for sample_id in sample_ids).encode()
            ).hexdigest(),
            "real_asset_manifest_sha256": _asset_manifest_digest(
                evaluation.samples, "source"
            ),
            "generated_asset_manifest_sha256": _asset_manifest_digest(
                evaluation.samples, "candidate"
            ),
        }
        if mutation == "generation_set":
            binding["generation_result_set_sha256"] = "0" * 64
        elif mutation == "per_sample_set":
            binding["per_sample_set_sha256"] = "0" * 64
        result = {
            "metrics": ["fid", "kid", "niqe", "sharpness"],
            "num_generated": len(sample_ids),
            "num_real": len(sample_ids),
            "sample_id_manifest": str(request.manifest_path),
            "sample_id_count": len(sample_ids),
            "sample_id_sha256": binding["ordered_sample_id_sha256"],
            "r9_evidence_binding": binding,
            "quality_contract": {
                "schema_version": 1,
                "metrics": ["fid", "kid", "niqe", "sharpness"],
                "sample_id_manifest_sha256": request.manifest_sha256,
                "per_sample_jsonl_sha256": "8" * 64,
                "real_asset_manifest_sha256": binding["real_asset_manifest_sha256"],
                "generated_asset_manifest_sha256": binding[
                    "generated_asset_manifest_sha256"
                ],
            },
            "fid": 10.0,
            "kid_mean": 0.01,
            "iqa": {"method": "niqe", "mean": 4.0},
            "sharpness": {
                "definition": "grayscale_laplacian_variance",
                "mean": 350.0,
            },
        }
        if mutation == "asset":
            result["quality_contract"]["generated_asset_manifest_sha256"] = "0" * 64
        elif mutation == "sample":
            result["sample_id_count"] = 0
        return result

    with pytest.raises(PhaseResultsError):
        _evaluate_quality(
            request,
            run,
            _sample_evidence(run),
            "candidate",
            evaluator,
        )


def _heldout_fixture(
    tmp_path: Path,
) -> tuple[PhaseResultsRequest, dict[str, Any], dict[str, Any], dict[str, Any]]:
    asset_rows = {}
    for name in ("e1", "e2", "facenet", "adaface"):
        path = tmp_path / "models" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        asset_rows[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "state": "sealed_unrun",
        }
    selection = {
        "winner": {"arm_id": "winner", "config_sha256": "3" * 64},
        "gate_contract_sha256": "4" * 64,
    }
    selection["selection_sha256"] = _canonical_digest(selection, "selection_sha256")
    seal = {
        "selection_sha256": selection["selection_sha256"],
        "assets": asset_rows,
        "execution_count": 0,
        "sealed": True,
    }
    seal["heldout_seal_sha256"] = _canonical_digest(seal, "heldout_seal_sha256")
    rows = []
    for index, sample_id in enumerate(("a", "b", "c")):
        source = tmp_path / "images" / f"{sample_id}_source.png"
        native = tmp_path / "images" / f"{sample_id}_native.png"
        candidate = tmp_path / "images" / f"{sample_id}_candidate.png"
        _write_png(source, (index, 1, 2))
        _write_png(native, (index, 3, 4))
        _write_png(candidate, (index, 5, 6))
        rows.append(
            {
                "sample_id": sample_id,
                "source": str(source),
                "native": str(native),
                "candidate": str(candidate),
                "source_sha256": _sha256(source),
                "native_sha256": _sha256(native),
                "candidate_sha256": _sha256(candidate),
            }
        )
    source_index = _write_source_index(
        tmp_path,
        ("a", "b", "c"),
        paths={str(row["sample_id"]): Path(str(row["source"])) for row in rows},
    )
    request = PhaseResultsRequest(
        repo_root=tmp_path,
        phase_root=tmp_path / "phase",
        phase="full",
        campaign_id="campaign",
        campaign_runtime_sha256="a" * 64,
        manifest_contracts_sha256="b" * 64,
        manifest_path=tmp_path / "full.jsonl",
        manifest_sha256="c" * 64,
        source_index_path=source_index,
        source_index_sha256=_sha256(source_index),
        checkpoint_sha256=SHA,
        bootstrap_seed=5,
        runs=(
            RunEvidenceSpec(
                logical_run_id="winner",
                arm_id="winner",
                family="flow_map2",
                seed=5501,
                repeat_index=None,
                shard_output_dirs=(tmp_path / "winner",),
            ),
        ),
        expected_candidate_arm_ids=("winner",),
        expected_seeds=(5501,),
        upstream_gate={"gate_contract_sha256": "4" * 64},
        visual_manifest_path=tmp_path / "visual.jsonl",
        visual_manifest_sha256="d" * 64,
        selection=selection,
        heldout_seal=seal,
    )
    winner_run = {
        "arm_id": "winner",
        "algorithm_config_sha256": "3" * 64,
        "output_sha256": "5" * 64,
        "seed": 5501,
        "rows": rows,
    }
    automatic_arm = {
        "privacy_bootstrap": {
            "upper_95_one_sided": 0.01,
            "bootstrap_sha256": "6" * 64,
        },
        "seed_results": [
            {
                "arcface_summary": {
                    "source_exact_one_count": 3,
                    "native_exact_one_count": 3,
                    "candidate_exact_one_count": 3,
                    "paired_exact_one_count": 3,
                    "failure_sample_ids": [],
                }
            }
        ],
    }
    return request, winner_run, automatic_arm, seal


def _heldout_claim(
    request: PhaseResultsRequest, winner_run: Mapping[str, Any]
) -> dict[str, Any]:
    assert request.selection is not None and request.heldout_seal is not None
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_heldout_execution_claim_v1",
        "selection_sha256": request.selection["selection_sha256"],
        "heldout_seal_sha256": request.heldout_seal["heldout_seal_sha256"],
        "winner_output_sha256": winner_run["output_sha256"],
        "source_index_path": str(request.source_index_path.resolve()),
        "source_index_sha256": request.source_index_sha256,
        "source_asset_manifest_sha256": _asset_manifest_digest(
            _sample_evidence(winner_run), "source"
        ),
        "native_asset_manifest_sha256": _asset_manifest_digest(
            _sample_evidence(winner_run), "native"
        ),
        "winner_asset_manifest_sha256": _asset_manifest_digest(
            _sample_evidence(winner_run), "candidate"
        ),
    }
    claim["heldout_execution_claim_sha256"] = _canonical_digest(
        claim, "heldout_execution_claim_sha256"
    )
    return claim


def test_heldout_claim_only_resume_executes_once(tmp_path: Path) -> None:
    request, winner_run, automatic_arm, _ = _heldout_fixture(tmp_path)
    claim = _heldout_claim(request, winner_run)
    _write_exclusive_json(request.phase_root / "heldout_execution_claim.json", claim)
    calls = 0

    def evaluator(evaluation) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _heldout_raw(
            [sample.sample_id for sample in evaluation.samples], coverage=3
        )

    result = _materialize_heldout(
        {"request": request, "manifest_ids": ["a", "b", "c"]},
        winner_run,
        evaluator,
        automatic_arm,
    )
    assert result["execution_count"] == 1
    assert calls == 1
    assert (request.phase_root / "heldout_execution_started.json").is_file()
    assert (request.phase_root / "heldout_raw_evidence.json").is_file()


def test_heldout_existing_raw_reuses_strict_chain_without_evaluator(
    tmp_path: Path,
) -> None:
    request, winner_run, automatic_arm, _ = _heldout_fixture(tmp_path)
    calls = 0

    def evaluator(evaluation) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _heldout_raw(
            [sample.sample_id for sample in evaluation.samples], coverage=3
        )

    validated = {"request": request, "manifest_ids": ["a", "b", "c"]}
    first = _materialize_heldout(validated, winner_run, evaluator, automatic_arm)

    def forbidden(_evaluation) -> dict[str, Any]:
        raise AssertionError("existing heldout raw must not rerun the evaluator")

    second = _materialize_heldout(validated, winner_run, forbidden, automatic_arm)
    assert first == second
    assert calls == 1


@pytest.mark.parametrize("mutation", ["missing", "digest", "claim"])
def test_heldout_existing_raw_rejects_invalid_started_chain_without_evaluator(
    tmp_path: Path, mutation: str
) -> None:
    request, winner_run, automatic_arm, _ = _heldout_fixture(tmp_path)
    validated = {"request": request, "manifest_ids": ["a", "b", "c"]}
    _materialize_heldout(
        validated,
        winner_run,
        lambda evaluation: _heldout_raw(
            [sample.sample_id for sample in evaluation.samples], coverage=3
        ),
        automatic_arm,
    )
    started_path = request.phase_root / "heldout_execution_started.json"
    if mutation == "missing":
        started_path.unlink()
    else:
        started = json.loads(started_path.read_text(encoding="utf-8"))
        if mutation == "digest":
            started["heldout_execution_started_sha256"] = "0" * 64
        else:
            started["heldout_execution_claim_sha256"] = "9" * 64
            started["heldout_execution_started_sha256"] = _canonical_digest(
                started, "heldout_execution_started_sha256"
            )
        _write_json(started_path, started)
    with pytest.raises(PhaseResultsError, match="heldout"):
        _materialize_heldout(
            validated,
            winner_run,
            lambda _evaluation: (_ for _ in ()).throw(
                AssertionError("invalid heldout chain must not rerun")
            ),
            automatic_arm,
        )


@pytest.mark.parametrize("role", ["source", "native", "candidate"])
def test_heldout_existing_raw_rehashes_each_input_before_reuse(
    tmp_path: Path, role: str
) -> None:
    request, winner_run, automatic_arm, _ = _heldout_fixture(tmp_path)
    validated = {"request": request, "manifest_ids": ["a", "b", "c"]}
    _materialize_heldout(
        validated,
        winner_run,
        lambda evaluation: _heldout_raw(
            [sample.sample_id for sample in evaluation.samples], coverage=3
        ),
        automatic_arm,
    )
    Path(str(winner_run["rows"][0][role])).write_bytes(b"tampered")
    with pytest.raises(PhaseResultsError, match=f"heldout {role} image SHA256"):
        _materialize_heldout(
            validated,
            winner_run,
            lambda _evaluation: (_ for _ in ()).throw(
                AssertionError("tampered heldout input must not rerun")
            ),
            automatic_arm,
        )


def test_heldout_existing_raw_rehashes_source_index_before_reuse(
    tmp_path: Path,
) -> None:
    request, winner_run, automatic_arm, _ = _heldout_fixture(tmp_path)
    validated = {"request": request, "manifest_ids": ["a", "b", "c"]}
    _materialize_heldout(
        validated,
        winner_run,
        lambda evaluation: _heldout_raw(
            [sample.sample_id for sample in evaluation.samples], coverage=3
        ),
        automatic_arm,
    )
    request.source_index_path.write_bytes(b"tampered")
    with pytest.raises(PhaseResultsError, match="source index SHA256"):
        _materialize_heldout(
            validated,
            winner_run,
            lambda _evaluation: (_ for _ in ()).throw(
                AssertionError("tampered source index must not rerun")
            ),
            automatic_arm,
        )


@pytest.mark.parametrize("mutation", ["digest", "claim"])
def test_heldout_existing_raw_rejects_tampered_raw_contract_without_evaluator(
    tmp_path: Path, mutation: str
) -> None:
    request, winner_run, automatic_arm, _ = _heldout_fixture(tmp_path)
    validated = {"request": request, "manifest_ids": ["a", "b", "c"]}
    _materialize_heldout(
        validated,
        winner_run,
        lambda evaluation: _heldout_raw(
            [sample.sample_id for sample in evaluation.samples], coverage=3
        ),
        automatic_arm,
    )
    raw_path = request.phase_root / "heldout_raw_evidence.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if mutation == "digest":
        raw["heldout_raw_evidence_sha256"] = "0" * 64
    else:
        raw["claim_sha256"] = "9" * 64
        raw["heldout_raw_evidence_sha256"] = _canonical_digest(
            raw, "heldout_raw_evidence_sha256"
        )
    _write_json(raw_path, raw)
    with pytest.raises(PhaseResultsError, match="heldout"):
        _materialize_heldout(
            validated,
            winner_run,
            lambda _evaluation: (_ for _ in ()).throw(
                AssertionError("tampered heldout raw must not rerun")
            ),
            automatic_arm,
        )


def test_heldout_started_without_result_forbids_rerun(tmp_path: Path) -> None:
    request, winner_run, automatic_arm, _ = _heldout_fixture(tmp_path)
    claim = _heldout_claim(request, winner_run)
    _write_exclusive_json(request.phase_root / "heldout_execution_claim.json", claim)
    started = {
        "schema_version": 1,
        "contract_type": "safa_r9_heldout_execution_started_v1",
        "heldout_execution_claim_sha256": claim["heldout_execution_claim_sha256"],
    }
    started["heldout_execution_started_sha256"] = _canonical_digest(
        started, "heldout_execution_started_sha256"
    )
    _write_exclusive_json(
        request.phase_root / "heldout_execution_started.json", started
    )
    with pytest.raises(PhaseResultsError, match="rerun is forbidden"):
        _materialize_heldout(
            {"request": request, "manifest_ids": ["a", "b", "c"]},
            winner_run,
            lambda evaluation: _heldout_raw(
                [sample.sample_id for sample in evaluation.samples], coverage=3
            ),
            automatic_arm,
        )


def test_heldout_sealed_asset_tamper_fails_before_claim(tmp_path: Path) -> None:
    request, winner_run, automatic_arm, seal = _heldout_fixture(tmp_path)
    Path(seal["assets"]["e1"]["path"]).write_bytes(b"tampered")
    with pytest.raises(PhaseResultsError, match="asset e1 digest mismatch"):
        _materialize_heldout(
            {"request": request, "manifest_ids": ["a", "b", "c"]},
            winner_run,
            lambda evaluation: _heldout_raw(
                [sample.sample_id for sample in evaluation.samples], coverage=3
            ),
            automatic_arm,
        )
    assert not (request.phase_root / "heldout_execution_claim.json").exists()
