from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from safa.evaluation.triangle32_evaluation import (
    canonical_digest,
    load_arm_set,
    sha256_file,
    validate_generation_result,
)
from safa.evaluation.triangle_screening import TriangleScreeningError


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = (
    ROOT
    / "artifacts/r10_triangle_exploration/checkpoint_fixed32_pilot/"
    "preparation_v1/preparation_manifest.json"
)
MATERIALIZER = ROOT / "scripts/materialize_fixed32_triangle_rows.py"
QUALITY_PREPARER = ROOT / "scripts/prepare_fixed32_quality_inputs.py"
REAL_INDEX = ROOT / "data/index/val_face_mixed_e14.jsonl"
RUNS_ROOT = (
    ROOT
    / "artifacts/r10_triangle_exploration/checkpoint_fixed32_pilot/runs_v1"
)
SELECTION = (
    ROOT
    / "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"
)
NATIVE_PER_SAMPLE = (
    ROOT
    / "artifacts/r10_triangle_exploration/fixed32_evaluation/"
    "inputs/native_per_sample.jsonl"
)


def _materializer():
    spec = importlib.util.spec_from_file_location("triangle_materializer", MATERIALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality_preparer():
    spec = importlib.util.spec_from_file_location(
        "triangle_quality_preparer", QUALITY_PREPARER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_triangle32_preparation_loads_mixed_selected24_in_order() -> None:
    arm_set = load_arm_set(PREPARATION)
    assert len(arm_set.arm_ids) == 24
    assert arm_set.arm_ids[:2] == (
        "g_a398b1ab6b504799_ema",
        "g_d75236bccc581e59_ema",
    )
    assert {arm_id.rsplit("_", 1)[-1] for arm_id in arm_set.arm_ids} == {
        "raw",
        "ema",
    }
    assert arm_set.result_filename == "result.json"
    assert validate_generation_result(
        arm_set,
        ROOT
        / "artifacts/r10_triangle_exploration/checkpoint_fixed32_pilot/runs_v1",
        arm_set.arm_ids[0],
    ).name == "per_sample.jsonl"


def test_triangle32_preparation_rejects_duplicate_arm_ids(tmp_path: Path) -> None:
    preparation = json.loads(PREPARATION.read_text(encoding="utf-8"))
    selected_path = Path(preparation["selected24"]["path"])
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected["selected"][1]["candidate_id"] = selected["selected"][0]["candidate_id"]
    changed_selected = tmp_path / "selected24.json"
    changed_selected.write_text(json.dumps(selected) + "\n", encoding="utf-8")
    preparation["selected24"] = {
        "path": str(changed_selected),
        "sha256": sha256_file(changed_selected),
    }
    changed_preparation = tmp_path / "preparation.json"
    changed_preparation.write_text(json.dumps(preparation) + "\n", encoding="utf-8")
    with pytest.raises(TriangleScreeningError, match="duplicate arm ID"):
        load_arm_set(changed_preparation)


def test_quality_preparer_requires_explicit_real_index() -> None:
    module = _quality_preparer()
    with pytest.raises(SystemExit):
        module.parse_args([
            "--runs-root",
            str(RUNS_ROOT),
            "--selection-manifest",
            str(SELECTION),
            "--output-dir",
            "/unused",
        ])


def test_quality_preparer_binds_registered_real_index(tmp_path: Path) -> None:
    module = _quality_preparer()
    output_dir = tmp_path / "quality-inputs"
    assert module.main([
        "--runs-root",
        str(RUNS_ROOT),
        "--selection-manifest",
        str(SELECTION),
        "--real-index",
        str(REAL_INDEX),
        "--output-dir",
        str(output_dir),
        "--arm-set-manifest",
        str(PREPARATION),
        "--native-per-sample",
        str(NATIVE_PER_SAMPLE),
    ]) == 0
    manifest = json.loads(
        (output_dir / "quality_input_manifest.json").read_text(encoding="utf-8")
    )
    binding = {"path": str(REAL_INDEX), "sha256": sha256_file(REAL_INDEX)}
    assert manifest["schema_version"] == 2
    assert manifest["contract_type"] == "safa_triangle_fixed32_quality_inputs_v2"
    assert manifest["real_index"] == binding
    assert manifest["native"]["real_index"] == binding
    assert len(manifest["candidates"]) == 24
    assert all(row["real_index"] == binding for row in manifest["candidates"])


def test_quality_preparer_rejects_substitute_real_index(tmp_path: Path) -> None:
    module = _quality_preparer()
    with pytest.raises(TriangleScreeningError, match="registered E14 index"):
        module.main([
            "--runs-root",
            str(RUNS_ROOT),
            "--selection-manifest",
            str(SELECTION),
            "--real-index",
            str(SELECTION),
            "--output-dir",
            str(tmp_path / "quality-inputs"),
            "--arm-set-manifest",
            str(PREPARATION),
            "--native-per-sample",
            str(NATIVE_PER_SAMPLE),
        ])


def test_official_arcface_requires_three_roles_and_bound_output() -> None:
    module = _materializer()
    arm_id = "g_mixed.raw-1"
    sample = {
        "sample_id": "s",
        "source": "/source.png",
        "native": "/native.png",
        "candidate": "/candidate.png",
        "source_sha256": "1" * 64,
        "native_sha256": "2" * 64,
        "candidate_sha256": "3" * 64,
    }
    request = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_evaluator_request_v1",
        "task": "arcface",
        "config": {"arcface": {"model_name": "buffalo_l"}},
        "payload": {"arm_id": arm_id, "samples": [sample]},
    }
    request["evaluator_request_sha256"] = canonical_digest(
        request, "evaluator_request_sha256"
    )
    result = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_evaluator_output_v1",
        "task": "arcface",
        "evaluator_request_sha256": request["evaluator_request_sha256"],
        "result": [{
            "sample_id": "s",
            "source_face_count": 0,
            "native_face_count": 1,
            "candidate_face_count": 1,
        }],
    }
    result["evaluator_output_sha256"] = canonical_digest(
        result, "evaluator_output_sha256"
    )
    assert module._official_arcface(
        request, result, arm_id=arm_id, sample_ids=["s"]
    ) == result["result"]
    del request["payload"]["samples"][0]["native"]
    request["evaluator_request_sha256"] = canonical_digest(
        request, "evaluator_request_sha256"
    )
    with pytest.raises(TriangleScreeningError, match="three-role"):
        module._official_arcface(
            request, result, arm_id=arm_id, sample_ids=["s"]
        )


def test_materializer_rejects_duplicate_nonfinite_and_preserves_null() -> None:
    module = _materializer()
    with pytest.raises(TriangleScreeningError, match="duplicate sample_id"):
        module._indexed([{"sample_id": "s"}, {"sample_id": "s"}], "ArcFace")
    exact = {
        "source_face_count": 1,
        "native_face_count": 1,
        "candidate_face_count": 1,
        "source_native_cosine": float("nan"),
        "source_candidate_cosine": 0.1,
    }
    with pytest.raises(TriangleScreeningError, match="finite"):
        module._identity_cosines(exact, label="ArcFace")
    unavailable = {
        "source_face_count": 0,
        "native_face_count": 1,
        "candidate_face_count": 1,
    }
    assert module._identity_cosines(unavailable, label="ArcFace") == (None, None)
    unavailable["source_native_cosine"] = 0.1
    with pytest.raises(TriangleScreeningError, match="must omit"):
        module._identity_cosines(unavailable, label="ArcFace")
