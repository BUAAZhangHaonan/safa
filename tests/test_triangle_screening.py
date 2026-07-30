from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from safa.evaluation.triangle_screening import (
    ArmResult,
    TriangleScreeningError,
    apply_stage_cap,
    evaluate_arms,
    join_eligibility_evidence,
    load_historical_primary_artifacts,
    paired_bootstrap_upper,
    pareto_frontier,
    select_eligible512,
    select_historical24,
)


def eligibility_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(2045):
        rows.append(
            {
                "sample_id": f"sample-{index:04d}",
                "label": index % 8,
                "native_sharpness": 100.0 + index,
                "source_detector": "buffalo_l",
                "native_detector": "buffalo_l",
                "source_face_count": 1,
                "native_face_count": 1,
            }
        )
    return rows


def arm_rows(
    count: int,
    *,
    candidate_count: int = 1,
    candidate_e0: float = 0.80,
    native_e0: float = 0.45,
    candidate_edev: float = 0.55,
    native_edev: float = 0.45,
    candidate_niqe: float = 5.05,
    native_niqe: float = 5.00,
    candidate_sharpness: float = 310.0,
    native_sharpness: float = 300.0,
    source_candidate_cosine: float = 0.40,
    source_native_cosine: float = 0.40,
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"s{index:04d}",
            "native_e0": native_e0,
            "candidate_e0": candidate_e0,
            "native_edev": native_edev,
            "candidate_edev": candidate_edev,
            "native_niqe": native_niqe,
            "candidate_niqe": candidate_niqe,
            "native_sharpness": native_sharpness,
            "candidate_sharpness": candidate_sharpness,
            "source_face_count": 1,
            "native_face_count": 1,
            "candidate_face_count": (
                candidate_count if index == count - 1 else 1
            ),
            "source_native_cosine": source_native_cosine,
            "source_candidate_cosine": source_candidate_cosine,
        }
        for index in range(count)
    ]


def arm_result(
    arm_id: str,
    r: float,
    q: float,
    p: float,
    *,
    passed: bool = True,
) -> ArmResult:
    return ArmResult(
        arm_id=arm_id,
        sample_count=32,
        candidate_exact_one_count=32 if passed else 31,
        e0=0.8,
        delta_e0=0.35,
        delta_edev=0.1,
        niqe=5.0,
        native_niqe=5.0,
        fid=10.0,
        native_fid=10.0,
        kid=0.01,
        native_kid=0.01,
        sharpness=310.0,
        native_sharpness=300.0,
        arcface_delta=0.0,
        arcface_delta_u95=0.0,
        hard_gate_pass=passed,
        failed_gates=() if passed else ("candidate_exact_one",),
        r_margin=r,
        q_margin=q,
        p_margin=p,
    )


def test_selector_is_repeatable_has_16_per_cell_and_nested_prefixes() -> None:
    rows = eligibility_rows()
    first = select_eligible512(rows)
    second = select_eligible512(list(reversed(rows)))
    assert first == second
    assert len(first) == 512
    counts: dict[tuple[int, int], int] = {}
    for row in first:
        key = (row["label"], row["sharpness_quartile"])
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {16}
    assert {row["cell_rank"] for row in first[:32]} == {0}
    assert {row["cell_rank"] for row in first[:128]} == {0, 1, 2, 3}
    assert len({row["sample_id"] for row in first[:32]}) == 32
    assert len({row["sample_id"] for row in first[:128]}) == 128


def test_selector_eligibility_requires_both_buffalo_l_exact_one() -> None:
    rows = eligibility_rows()
    rows[0]["source_face_count"] = 0
    with pytest.raises(TriangleScreeningError, match="eligible count"):
        select_eligible512(rows)
    assert len(select_eligible512(rows, expected_eligible_count=2044)) == 512


def test_eligibility_join_is_exact_and_preserves_three_sources() -> None:
    rows = eligibility_rows()[:8]
    full = [
        {
            key: row[key]
            for key in (
                "sample_id",
                "source_detector",
                "native_detector",
                "source_face_count",
                "native_face_count",
            )
        }
        for row in rows
    ]
    labels = [
        {"sample_id": row["sample_id"], "label": row["label"]} for row in rows
    ]
    sharpness = [
        {
            "sample_id": row["sample_id"],
            "native_sharpness": row["native_sharpness"],
        }
        for row in rows
    ]
    assert join_eligibility_evidence(full, labels, sharpness) == rows
    with pytest.raises(TriangleScreeningError, match="same sample IDs"):
        join_eligibility_evidence(full, labels[:-1], sharpness)


def test_candidate_exact_one_equality_passes_and_511_fails() -> None:
    pass_rows = arm_rows(512)
    fail_rows = arm_rows(512, candidate_count=0)
    results = evaluate_arms(
        [
            {
                "arm_id": "paper_eta_0p125",
                "rows": pass_rows,
                "fid": 12.0,
                "kid": 0.012,
            },
            {
                "arm_id": "bad",
                "rows": fail_rows,
                "fid": 12.0,
                "kid": 0.012,
            },
        ],
        stage=512,
        native_fid=10.0,
        native_kid=0.01,
    )
    baseline, bad = results
    assert baseline.candidate_exact_one_count == 512
    assert baseline.hard_gate_pass
    assert bad.candidate_exact_one_count == 511
    assert not bad.hard_gate_pass
    assert "candidate_exact_one" in bad.failed_gates


def test_normalized_axes_are_monotonic() -> None:
    baseline_rows = arm_rows(32)
    better_rows = arm_rows(
        32,
        candidate_e0=0.85,
        candidate_edev=0.60,
        candidate_niqe=5.0,
        candidate_sharpness=330.0,
        source_candidate_cosine=0.39,
    )
    baseline, better = evaluate_arms(
        [
            {
                "arm_id": "paper_eta_0p125",
                "rows": baseline_rows,
                "fid": 12.0,
                "kid": 0.012,
            },
            {
                "arm_id": "better",
                "rows": better_rows,
                "fid": 11.0,
                "kid": 0.011,
            },
        ],
        stage=32,
        native_fid=10.0,
        native_kid=0.01,
    )
    assert better.r_margin > baseline.r_margin
    assert better.q_margin > baseline.q_margin
    assert better.p_margin > baseline.p_margin
    assert better.status == "privacy_positive_breakthrough"


def test_pareto_keeps_ties_and_removes_strictly_dominated() -> None:
    a = arm_result("a", 1.0, 1.0, 1.0)
    tied = arm_result("tied", 1.0, 1.0, 1.0)
    dominated = arm_result("dominated", 0.9, 1.0, 1.0)
    assert [item.arm_id for item in pareto_frontier([a, tied, dominated])] == [
        "a",
        "tied",
    ]


def test_stage_caps_use_locked_axis_priority_and_ties() -> None:
    frontier = [
        arm_result("p", 0.1, 0.2, 0.9),
        arm_result("q", 0.2, 0.9, 0.1),
        arm_result("r", 0.9, 0.1, 0.2),
        arm_result("balanced", 0.5, 0.5, 0.5),
        arm_result("extra", 0.4, 0.4, 0.4),
    ]
    assert {item.arm_id for item in apply_stage_cap(frontier, 128)} == {
        "p",
        "q",
        "r",
        "balanced",
    }
    assert [item.arm_id for item in apply_stage_cap(frontier, 512)] == ["p", "q"]


def test_paired_bootstrap_is_repeatable_and_tracks_constant_delta() -> None:
    values = np.full(128, 0.0125)
    first = paired_bootstrap_upper(values, iterations=2_000)
    second = paired_bootstrap_upper(values, iterations=2_000)
    assert first == second == pytest.approx(0.0125)


def test_eight_sample_cli_writes_locked_schema(tmp_path: Path) -> None:
    rows = arm_rows(8)
    rows_path = tmp_path / "paper.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    request = {
        "stage": 8,
        "native_fid": 10.0,
        "native_kid": 0.01,
        "baseline_arm_id": "paper_eta_0p125",
        "arms": [
            {
                "arm_id": "paper_eta_0p125",
                "rows_path": rows_path.name,
                "fid": 12.0,
                "kid": 0.012,
            }
        ],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output_dir = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_triangle_screening.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in output_dir.iterdir()} == {
        "summary.json",
        "arms.csv",
        "arcface_failures.json",
        "conclusion.md",
    }
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["contract_type"] == "safa_triangle_screening_v1"
    assert summary["stage"] == 8
    assert summary["bootstrap"] == {
        "bit_generator": "PCG64",
        "iterations": 2000,
        "paired": True,
        "seed": 91637,
        "shared_across_arms": True,
    }
    assert summary["selected_arm_ids"] == ["paper_eta_0p125"]


def test_missing_nonfinite_and_overwrite_are_rejected(tmp_path: Path) -> None:
    rows = arm_rows(8)
    rows[0]["candidate_e0"] = float("nan")
    with pytest.raises(TriangleScreeningError, match="finite"):
        evaluate_arms(
            [
                {
                    "arm_id": "paper_eta_0p125",
                    "rows": rows,
                    "fid": 12.0,
                    "kid": 0.012,
                }
            ],
            stage=8,
            native_fid=10.0,
            native_kid=0.01,
        )


def test_historical24_average_ties_directions_and_unique_manifest() -> None:
    family_ids = [f"family-{index:02d}" for index in range(12)]
    candidates = []
    manifest = []
    for index in range(193):
        candidate_id = f"candidate-{index:03d}"
        # The tied pairs exercise average-tie percentile ranks.
        tied_value = float(index // 2)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "family_id": family_ids[index % 12],
                "smoke8_primary": True,
                "e0": tied_value,
                "edev": tied_value,
                "fid": -tied_value,
                "kid": -tied_value,
                "niqe": -tied_value,
                "sharpness": tied_value,
                "arcface_source_candidate_cosine": -tied_value,
            }
        )
        manifest.append(
            {
                "candidate_id": candidate_id,
                "checkpoint_path": f"/checkpoints/{candidate_id}.pt",
                "checkpoint_sha256": f"{index:064x}",
            }
        )
    selected = select_historical24(
        candidates, manifest, family_ids=family_ids
    )
    assert len(selected) == len({row["candidate_id"] for row in selected}) == 24
    assert [row["selection_group"] for row in selected[:12]] == [
        "family_balance"
    ] * 12
    assert [row["selection_group"] for row in selected[12:16]] == ["top_R"] * 4
    assert [row["selection_group"] for row in selected[16:20]] == ["top_Q"] * 4
    assert [row["selection_group"] for row in selected[20:24]] == ["top_P"] * 4
    assert all(row["R"] == row["Q"] == row["P"] for row in selected)
    assert all("checkpoint_path" in row for row in selected)

    with pytest.raises(TriangleScreeningError, match="mapping is not exact"):
        select_historical24(
            candidates, manifest[:-1], family_ids=family_ids
        )


def test_historical_artifact_adapter_reads_directory_and_manifest_json(
    tmp_path: Path,
) -> None:
    family_ids = [
        "E11", "E12", "E13", "E15", "E2", "R1",
        "R2", "R3", "R4", "R5", "R6", "R7",
    ]
    primary_root = tmp_path / "smoke8_primary"
    manifest = {"candidates": []}
    for index in range(193):
        candidate_id = f"g_{index:016x}_raw"
        manifest["candidates"].append(
            {
                "candidate_id": candidate_id,
                "source_logical_experiment_ids": [family_ids[index % 12]],
                "checkpoint_path": f"artifacts/checkpoints/{candidate_id}.pt",
                "checkpoint_sha256": f"{index:064x}",
            }
        )
        result_dir = primary_root / candidate_id
        result_dir.mkdir(parents=True)
        value = float(index)
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "failure": None,
                    "evidence": {
                        "mode": "smoke8",
                        "replicate": "primary",
                        "sample_count": 8,
                        "e0_mean": value,
                        "edev_mean": value + 1,
                        "quality": {
                            "fid": value + 2,
                            "canonical_kid": {"kid_mean": value + 3},
                            "iqa": {"mean": value + 4},
                            "sharpness": {"mean": value + 5},
                        },
                        "arcface": {
                            "mean_source_candidate_cosine": value + 6
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    candidates, manifest_rows = load_historical_primary_artifacts(
        primary_root, manifest
    )
    assert len(candidates) == len(manifest_rows) == 193
    first = min(candidates, key=lambda row: row["candidate_id"])
    assert first == {
        "candidate_id": "g_0000000000000000_raw",
        "family_id": "E11",
        "smoke8_primary": True,
        "e0": 0.0,
        "edev": 1.0,
        "fid": 2.0,
        "kid": 3.0,
        "niqe": 4.0,
        "sharpness": 5.0,
        "arcface_source_candidate_cosine": 6.0,
    }
    selected = select_historical24(
        candidates, manifest_rows, family_ids=family_ids
    )
    assert len(selected) == 24


def test_incomplete_arm_row_is_rejected() -> None:
    incomplete = arm_rows(8)
    del incomplete[0]["candidate_edev"]
    with pytest.raises(TriangleScreeningError, match="missing fields"):
        evaluate_arms(
            [
                {
                    "arm_id": "paper_eta_0p125",
                    "rows": incomplete,
                    "fid": 12.0,
                    "kid": 0.012,
                }
            ],
            stage=8,
            native_fid=10.0,
            native_kid=0.01,
        )


def test_real_artifact_adapter_script_builds_nested_prefixes(
    tmp_path: Path,
) -> None:
    full_root = tmp_path / "full"
    (full_root / "evaluator_evidence" / "arcface").mkdir(parents=True)
    request_dir = full_root / "evaluator_runs" / "arcface" / "winner"
    request_dir.mkdir(parents=True)
    source_index_path = tmp_path / "source_index.jsonl"
    manifest_path = tmp_path / "full_2048.jsonl"
    manifest_rows = []
    source_rows = []
    face_rows = []
    paired_rows = []
    for index in range(2048):
        sample_id = f"sample-{index:04d}"
        manifest_rows.append({"sample_id": sample_id})
        source_rows.append({"sample_id": sample_id, "label": index % 8})
        face_rows.append(
            {
                "sample_id": sample_id,
                "source_face_count": 1,
                "native_face_count": 0 if index < 3 else 1,
                "candidate_face_count": 1,
                "source_native_cosine": 0.0,
                "source_candidate_cosine": 0.0,
            }
        )
        paired_rows.append(
            {"sample_id": sample_id, "native_sharpness": 100.0 + index}
        )
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    source_index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (full_root / "automatic_evidence.json").write_text(
        json.dumps(
            {
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": digest(manifest_path),
                },
                "source_index": {
                    "path": str(source_index_path),
                    "sha256": digest(source_index_path),
                },
                "arms": [
                    {
                        "arm_id": "paper_eta_0p125",
                        "paired_metric_rows": {"rows": paired_rows},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (full_root / "evaluator_evidence" / "arcface" / "winner.json").write_text(
        json.dumps({"arcface": {"rows": face_rows}}), encoding="utf-8"
    )
    (request_dir / "request.json").write_text(
        json.dumps({"config": {"arcface": {"model_name": "buffalo_l"}}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_triangle_eligible512.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--full-root",
            str(full_root),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rows512 = [
        json.loads(line)
        for line in (output_dir / "eligible512.jsonl").read_text().splitlines()
    ]
    rows32 = [
        json.loads(line)
        for line in (output_dir / "prefix32.jsonl").read_text().splitlines()
    ]
    rows128 = [
        json.loads(line)
        for line in (output_dir / "prefix128.jsonl").read_text().splitlines()
    ]
    assert len(rows512) == 512
    assert rows32 == rows512[:32]
    assert rows128 == rows512[:128]
