from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from safa.evaluation.triangle32_evaluation import load_arm_set
from safa.evaluation.triangle_screening import (
    evaluate_arms,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "scripts/prepare_r11_official_evaluation.py"
MATERIALIZER = ROOT / "scripts/materialize_fixed32_triangle_rows.py"
QUALITY = ROOT / "scripts/run_r11_quality_evaluation.py"
RUN_CONTRACTS = (
    ROOT
    / "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/run_contracts.json"
)


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(
    count: int,
    *,
    native_face_count: int = 1,
    candidate_face_count: int = 1,
) -> list[dict[str, object]]:
    privacy_available = native_face_count == candidate_face_count == 1
    return [
        {
            "sample_id": f"s{index:03d}",
            "native_e0": 0.40,
            "candidate_e0": 0.80,
            "native_edev": 0.40,
            "candidate_edev": 0.50,
            "native_niqe": 5.00,
            "candidate_niqe": 5.01,
            "native_sharpness": 400.0,
            "candidate_sharpness": 400.0,
            "source_face_count": 1,
            "native_face_count": native_face_count,
            "candidate_face_count": candidate_face_count,
            "source_native_cosine": 0.40 if privacy_available else None,
            "source_candidate_cosine": 0.39 if privacy_available else None,
        }
        for index in range(count)
    ]


def test_real_r11_dry_preparation_binds_both_locked_datasets(
    tmp_path: Path,
) -> None:
    module = _module(PREPARER, "prepare_r11_official_evaluation")
    output = tmp_path / "evaluation-preparation"
    plan = module.prepare(
        run_contracts_path=RUN_CONTRACTS,
        output_dir=output,
        write=True,
    )
    assert plan["expected_arcface_request_count"] == 4
    datasets = {row["dataset_id"]: row for row in plan["datasets"]}
    assert set(datasets) == {"prefix128", "sharpness_tail32"}
    prefix = load_arm_set(Path(datasets["prefix128"]["dataset_contract"]))
    tail = load_arm_set(Path(datasets["sharpness_tail32"]["dataset_contract"]))
    assert (
        prefix.selection_role,
        prefix.sample_count,
        prefix.stage,
        prefix.quality_metrics,
        prefix.baseline_arm_id,
    ) == (
        "prefix128",
        128,
        128,
        ("fid", "kid", "niqe", "sharpness"),
        "eta025_prefix128",
    )
    assert (
        tail.selection_role,
        tail.sample_count,
        tail.stage,
        tail.quality_metrics,
        tail.baseline_arm_id,
    ) == (
        "sharpness_tail32",
        32,
        32,
        ("niqe", "sharpness"),
        "eta025_tail32",
    )
    assert prefix.selection_manifest_sha256 == (
        "9e1bfa90057c9b72f02a29340d91f0d87d4d4f35119733824b9659bd7ee8db89"
    )
    assert prefix.selection_sample_id_sha256 == (
        "3af3ecd3f844f61b7e7e5a25519ed97fa30dc56663e52fc2ee84cdd9cabac2d5"
    )
    assert tail.selection_manifest_sha256 == (
        "f38fb6f6542c267b6c7d9cbec9ce57abdf9b0657edfe8d060fe533178a9f5b29"
    )
    assert tail.selection_sample_id_sha256 == (
        "0df6a923d37a45c67910d5f9f118700392403e12bdf6e8a6dbb04edfde86987f"
    )
    for dataset in datasets.values():
        quality_argv = dataset["commands"]["quality_prepare"]
        assert "--real-index" in quality_argv
        assert quality_argv[quality_argv.index("--real-index") + 1] == str(
            module.REAL_INDEX
        )


def test_prefix128_uses_shared_pcg64_2000_and_tail32_forbids_inference_metrics(
    tmp_path: Path,
) -> None:
    prefix_rows = _rows(128)
    prefix = evaluate_arms(
        [
            {
                "arm_id": "eta025_prefix128",
                "rows": prefix_rows,
                "fid": 10.0,
                "kid": 0.01,
            },
            {
                "arm_id": "eta05_prefix128",
                "rows": prefix_rows,
                "fid": 10.0,
                "kid": 0.01,
            },
        ],
        stage=128,
        native_fid=10.0,
        native_kid=0.01,
        baseline_arm_id="eta025_prefix128",
        expected_sample_ids=[row["sample_id"] for row in prefix_rows],
    )
    assert all(result.arcface_delta_u95 == pytest.approx(-0.01) for result in prefix)
    prefix_output = tmp_path / "prefix"
    write_outputs(
        prefix_output,
        prefix,
        stage=128,
        baseline_arm_id="eta025_prefix128",
    )
    prefix_summary = json.loads(
        (prefix_output / "summary.json").read_text(encoding="utf-8")
    )
    assert prefix_summary["bootstrap"] == {
        "bit_generator": "PCG64",
        "seed": 91637,
        "iterations": 2000,
        "paired": True,
        "shared_across_arms": True,
    }

    tail_rows = _rows(32)
    tail = evaluate_arms(
        [
            {"arm_id": "eta025_tail32", "rows": tail_rows},
            {"arm_id": "eta05_tail32", "rows": tail_rows},
        ],
        stage=32,
        native_fid=None,
        native_kid=None,
        baseline_arm_id="eta025_tail32",
        expected_sample_ids=[row["sample_id"] for row in tail_rows],
    )
    assert all(
        result.arcface_delta_u95 is None
        and result.fid is None
        and result.kid is None
        for result in tail
    )
    tail_output = tmp_path / "tail"
    write_outputs(
        tail_output,
        tail,
        stage=32,
        baseline_arm_id="eta025_tail32",
        selection_manifest=tail_rows,
        selection_metadata={
            "selection_role": "sharpness_tail32",
            "manifest_sha256": "f" * 64,
        },
    )
    tail_summary = json.loads(
        (tail_output / "summary.json").read_text(encoding="utf-8")
    )
    assert tail_summary["bootstrap"] is None
    assert tail_summary["selection"]["selection_role"] == "sharpness_tail32"
    assert "selector" not in tail_summary["selection"]


@pytest.mark.parametrize(
    ("native_count", "candidate_count"),
    [(1, 2), (2, 1)],
)
def test_official_arcface_role_failure_is_explicit_null_and_hard_fails(
    native_count: int,
    candidate_count: int,
) -> None:
    materializer = _module(MATERIALIZER, "r11_materializer")
    official_row = {
        "source_face_count": 1,
        "native_face_count": native_count,
        "candidate_face_count": candidate_count,
    }
    assert materializer._identity_cosines(
        official_row, label="ArcFace"
    ) == (None, None)
    result = evaluate_arms(
        [
            {
                "arm_id": "eta025_tail32",
                "rows": _rows(
                    32,
                    native_face_count=native_count,
                    candidate_face_count=candidate_count,
                ),
            }
        ],
        stage=32,
        native_fid=None,
        native_kid=None,
        baseline_arm_id="eta025_tail32",
    )[0]
    assert not result.hard_gate_pass
    assert result.arcface_delta is None
    expected_gate = (
        "native_exact_one" if native_count != 1 else "candidate_exact_one"
    )
    assert expected_gate in result.failed_gates


def test_direct_meanflow_representation_has_no_aliases_and_kid_cli_is_exact() -> None:
    materializer = _module(MATERIALIZER, "r11_direct_materializer")
    row = {
        "native_cosine": 0.4,
        "candidate_cosine": 0.8,
        "native_edev_cosine": 0.5,
        "edev_cosine": 0.7,
        "cosine": 0.8,
    }
    assert materializer._direct_meanflow_representation_cosines(
        row, candidate_label="arm"
    ) == (0.4, 0.8, 0.5, 0.7)
    assert row["cosine"] == row["candidate_cosine"]
    alias = dict(row, e0_cosine=0.8)
    with pytest.raises(Exception, match="forbidden representation aliases"):
        materializer._direct_meanflow_representation_cosines(
            alias, candidate_label="arm"
        )
    quality = _module(QUALITY, "r11_quality")
    parsed = quality.parse_args(
        [
            "--real-index",
            "/real.jsonl",
            "--generated-dir",
            "/generated",
            "--output",
            "/quality.json",
            "--sample-id-manifest",
            "/selection.jsonl",
            "--per-sample-jsonl",
            "/rows.jsonl",
            "--device",
            "cuda:0",
            "--metrics",
            "fid",
            "kid",
            "niqe",
            "sharpness",
            "--kid-subset-size",
            "127",
        ]
    )
    assert parsed.kid_subset_size == 127
