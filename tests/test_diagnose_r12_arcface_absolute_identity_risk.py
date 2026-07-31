from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "diagnose_r12_arcface_absolute_identity_risk.py"
spec = importlib.util.spec_from_file_location(
    "diagnose_r12_arcface_absolute_identity_risk", SCRIPT
)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


class FakeFace:
    def __init__(self, embedding: np.ndarray) -> None:
        self.bbox = np.array([10.0, 10.0, 90.0, 90.0])
        self.kps = np.array(
            [[30.0, 35.0], [70.0, 35.0], [50.0, 52.0], [35.0, 70.0], [65.0, 70.0]]
        )
        self.det_score = 0.99
        self.normed_embedding = embedding


def _unit_embedding() -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float64)
    embedding[0] = 1.0
    return embedding


def test_exact_one_embedding_serializes_and_requires_unit_finite_vector() -> None:
    detection, embedding = cli.extract_exact_one_embedding(
        [FakeFace(_unit_embedding())], (100, 100, 3)
    )
    assert detection["face_count"] == 1
    assert detection["bbox_area_ratio"] == pytest.approx(0.64)
    assert embedding.shape == (512,)
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="exact-one"):
        cli.extract_exact_one_embedding([], (100, 100, 3))
    bad = _unit_embedding()
    bad[2] = np.nan
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="finite"):
        cli.extract_exact_one_embedding([FakeFace(bad)], (100, 100, 3))
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="differs from one"):
        cli.extract_exact_one_embedding(
            [FakeFace(_unit_embedding() * 0.9)], (100, 100, 3)
        )


def test_retrieval_uses_strict_rank_and_positive_margin() -> None:
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    result = cli.retrieval_metrics(np.array([1.0, 0.0]), gallery, 0)
    assert result["true_source_rank"] == 1
    assert result["retrieval_percentile"] == pytest.approx(1.0)
    assert result["max_impostor_margin"] == pytest.approx(0.2)
    assert result["recall_at_1"] == 1
    assert result["positive_margin"] is True


def test_retrieval_tie_is_not_strict_rank_one() -> None:
    gallery = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = cli.retrieval_metrics(np.array([1.0, 0.0]), gallery, 0)
    assert result["true_source_rank"] == 2
    assert result["retrieval_percentile"] == pytest.approx(0.5)
    assert result["max_impostor_margin"] == pytest.approx(0.0)
    assert result["recall_at_1"] == 0
    assert result["positive_margin"] is False


def test_retrieval_rejects_nonfinite_and_shape_mismatch() -> None:
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="shapes"):
        cli.retrieval_metrics(np.zeros(3), np.zeros((2, 2)), 0)
    gallery = np.eye(2)
    query = np.array([np.nan, 0.0])
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="non-finite"):
        cli.retrieval_metrics(query, gallery, 0)


def test_paired_bootstrap_is_deterministic_and_candidate_minus_native() -> None:
    indices = np.random.default_rng(91637).integers(0, 4, size=(1000, 4))
    native = np.array([0.0, 1.0, 2.0, 3.0])
    candidate = native + 0.25
    first = cli.paired_bootstrap_mean_delta(native, candidate, indices)
    second = cli.paired_bootstrap_mean_delta(native, candidate, indices)
    assert first == second
    assert first["direction"] == "candidate_minus_native"
    assert first["mean_delta"] == pytest.approx(0.25)
    assert first["ci95"] == pytest.approx([0.25, 0.25])


def test_top_k_enrichment_and_zero_event_case() -> None:
    result = cli.top_k_enrichment(
        [True, True, False, False, False, False, False, False], [0, 1]
    )
    assert result["observed_in_top_k"] == 2
    assert result["random_expected_overlap"] == pytest.approx(0.5)
    assert result["enrichment_factor"] == pytest.approx(4.0)
    zero = cli.top_k_enrichment([False] * 8, [0, 1])
    assert zero["event_count_all"] == 0
    assert zero["observed_in_top_k"] == 0
    assert zero["enrichment_factor"] == pytest.approx(0.0)


def _decision_cell(
    *,
    percentile_lower: float,
    recall5_lower: float,
    recall5_enrichment: float,
    margin_enrichment: float,
    relative_delta: float = 0.03,
    spearman: float = -0.6,
    cosine_median: float = 0.05,
) -> dict:
    return {
        "mean_locked_relative_delta": relative_delta,
        "relative_delta_vs_source_native_spearman": spearman,
        "paired_candidate_minus_native": {
            "retrieval_percentile": {"ci95": [percentile_lower, 0.2]},
            "recall_at_5": {"ci95": [recall5_lower, 0.2]},
        },
        "top8_relative_delta_outliers": {
            "candidate_true_source_cosine_median": cosine_median,
            "candidate_recall_at_5": {"enrichment_factor": recall5_enrichment},
            "candidate_positive_margin": {"enrichment_factor": margin_enrichment},
        },
    }


def test_decision_supports_actual_retrieval_leakage_only_for_both_arms() -> None:
    cell = _decision_cell(
        percentile_lower=0.01,
        recall5_lower=0.01,
        recall5_enrichment=2.0,
        margin_enrichment=2.5,
    )
    decision = cli.evaluate_decision({"u12": cell, "u16": cell})
    assert decision["classification"] == "actual_retrieval_leakage_supported"
    assert decision["actual_retrieval_leakage_supported"] is True
    one_unsupported = _decision_cell(
        percentile_lower=-0.01,
        recall5_lower=0.01,
        recall5_enrichment=2.0,
        margin_enrichment=2.5,
    )
    decision = cli.evaluate_decision({"u12": cell, "u16": one_unsupported})
    assert decision["actual_retrieval_leakage_supported"] is False


def test_decision_identifies_baseline_conditioned_geometry() -> None:
    cell = _decision_cell(
        percentile_lower=-0.01,
        recall5_lower=0.0,
        recall5_enrichment=1.2,
        margin_enrichment=1.0,
    )
    decision = cli.evaluate_decision({"u12": cell, "u16": cell})
    assert decision["classification"] == "baseline_conditioned_metric_geometry"
    assert decision["baseline_conditioned_metric_geometry"] is True
    assert decision["locked_relative_delta_gate_changed"] is False


def test_decision_is_inconclusive_when_baseline_rule_is_not_met() -> None:
    cell = _decision_cell(
        percentile_lower=-0.01,
        recall5_lower=0.0,
        recall5_enrichment=1.2,
        margin_enrichment=1.0,
        relative_delta=0.01,
    )
    decision = cli.evaluate_decision({"u12": cell, "u16": cell})
    assert decision["classification"] == "identity_risk_inconclusive"


def test_arcface_rows_fail_closed_on_count_and_nonfinite(tmp_path: Path) -> None:
    row = {
        "sample_id": "a",
        "source_face_count": 1,
        "native_face_count": 1,
        "candidate_face_count": 2,
        "source_native_cosine": 0.0,
        "source_candidate_cosine": 0.1,
    }
    path = tmp_path / "arcface.json"
    path.write_text(json.dumps({"result": [row]}), encoding="utf-8")
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="exact-one"):
        cli.arcface_rows(path, ["a"], "arcface")
    row["candidate_face_count"] = 1
    row["source_candidate_cosine"] = float("inf")
    path.write_text(json.dumps({"result": [row]}), encoding="utf-8")
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="finite"):
        cli.arcface_rows(path, ["a"], "arcface")


def test_predeclared_config_locks_gallery_statistics_decision_and_inputs() -> None:
    path = (
        REPO_ROOT
        / "configs"
        / "medium_v2"
        / "experiments"
        / "r12_arcface_absolute_identity_risk.json"
    )
    request = json.loads(path.read_text(encoding="utf-8"))
    cli.validate_request(request)
    assert request["gallery"]["expected_source_count"] == 64
    assert request["statistics"] == {
        "bootstrap_seed": 91637,
        "bootstrap_iterations": 10000,
        "paired_resampling_unit": "sample_id",
        "top_k": 8,
    }
    assert [dataset["dataset_id"] for dataset in request["datasets"]] == [
        "regular32",
        "sharpness_tail32",
    ]
    for dataset in request["datasets"]:
        assert set(dataset["arms"]) == {"u12", "u16"}
    mutated = json.loads(json.dumps(request))
    mutated["decision_rule"]["top8_enrichment_threshold"] = 1.5
    with pytest.raises(cli.IdentityRiskDiagnosticError, match="predeclared"):
        cli.validate_request(mutated)
