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
SCRIPT = SCRIPT_DIR / "diagnose_r12_lpips_conflict.py"
spec = importlib.util.spec_from_file_location("diagnose_r12_lpips_conflict", SCRIPT)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def _bootstrap() -> np.ndarray:
    return np.random.default_rng(91637).integers(0, 32, size=(200, 32))


def test_association_is_deterministic_and_reports_enrichment() -> None:
    x = np.arange(32, dtype=float)
    y = x + np.sin(x) * 0.01
    first = cli.association(x, y, _bootstrap(), 8)
    second = cli.association(x, y, _bootstrap(), 8)
    assert first == second
    assert first["spearman"] > 0.99
    assert first["top_outlier_overlap"] == 8
    assert first["top_outlier_enrichment_factor"] == pytest.approx(4.0)
    assert first["top_failure_group_predictor_mean_gap"] > 0.0


def _cell(rho: float, enrichment: float, lower: float) -> dict:
    return {
        "spearman": rho,
        "spearman_ci95": [lower, 0.8],
        "top_failure_group_predictor_mean_gap": enrichment,
    }


def test_license_requires_consistent_direction_and_one_supported_update() -> None:
    result = cli.evaluate_target_license(
        {"u12": _cell(0.3, 1.5, -0.1), "u16": _cell(0.4, 2.0, 0.01)}
    )
    assert result["licensed"] is True
    assert result["positive_spearman_and_failure_enrichment_both_updates"] is True


def test_license_rejects_one_negative_or_unenriched_update() -> None:
    negative = cli.evaluate_target_license(
        {"u12": _cell(-0.1, 2.0, 0.01), "u16": _cell(0.4, 2.0, 0.01)}
    )
    assert negative["licensed"] is False
    unenriched = cli.evaluate_target_license(
        {"u12": _cell(0.2, 0.0, 0.01), "u16": _cell(0.4, 2.0, 0.01)}
    )
    assert unenriched["licensed"] is False


def test_license_rejects_direction_without_ci_support() -> None:
    result = cli.evaluate_target_license(
        {"u12": _cell(0.2, 1.5, -0.2), "u16": _cell(0.3, 2.0, -0.1)}
    )
    assert result["positive_spearman_and_failure_enrichment_both_updates"] is True
    assert result["at_least_one_update_spearman_ci_lower_gt_zero"] is False
    assert result["licensed"] is False


def test_shared_license_requires_both_targets() -> None:
    decisions = {
        "regular_privacy": {"licensed": True},
        "tail_sharpness": {"licensed": False},
    }
    assert cli.evaluate_shared_license(decisions) is False
    decisions["tail_sharpness"]["licensed"] = True
    assert cli.evaluate_shared_license(decisions) is True


def test_positive_ratio_fails_closed() -> None:
    assert cli.positive_ratio(2.0, 4.0, "metric") == pytest.approx(0.5)
    with pytest.raises(cli.LpipsConflictError, match="positive denominator"):
        cli.positive_ratio(1.0, 0.0, "metric")
    with pytest.raises(cli.LpipsConflictError, match="non-negative numerator"):
        cli.positive_ratio(-1.0, 2.0, "metric")


def test_paired_mean_delta_uses_u16_minus_u12() -> None:
    indices = np.tile(np.arange(32), (100, 1))
    result = cli.paired_mean_delta(np.zeros(32), np.ones(32), indices)
    assert result["mean_delta"] == pytest.approx(1.0)
    assert result["ci95"] == pytest.approx([1.0, 1.0])


def test_quality_rows_fail_on_order_and_nonfinite(tmp_path: Path) -> None:
    rows = [
        {"sample_id": "b", "niqe": 1.0, "sharpness": 2.0},
        {"sample_id": "a", "niqe": 1.0, "sharpness": 2.0},
    ]
    path = tmp_path / "quality.json"
    path.write_text(json.dumps({"per_sample_metrics": {"rows": rows}}), encoding="utf-8")
    with pytest.raises(cli.LpipsConflictError, match="order"):
        cli.quality_rows(path, ["a", "b"], "quality")
    rows.reverse()
    rows[0]["sharpness"] = float("inf")
    path.write_text(json.dumps({"per_sample_metrics": {"rows": rows}}), encoding="utf-8")
    with pytest.raises(cli.LpipsConflictError, match="finite"):
        cli.quality_rows(path, ["a", "b"], "quality")


def test_arcface_rows_fail_closed_on_non_exact_one(tmp_path: Path) -> None:
    row = {
        "sample_id": "a",
        "source_face_count": 1,
        "native_face_count": 1,
        "candidate_face_count": 2,
        "source_candidate_cosine": 0.1,
        "source_native_cosine": 0.0,
    }
    path = tmp_path / "arcface.json"
    path.write_text(json.dumps({"result": [row]}), encoding="utf-8")
    with pytest.raises(cli.LpipsConflictError, match="exact-one"):
        cli.arcface_rows(path, ["a"], "arcface")


class FakeLpips:
    def __call__(self, first, second):
        import torch

        return torch.mean(torch.square(first - second)).reshape(1, 1, 1, 1)


def test_lpips_distance_uses_only_the_supplied_pair() -> None:
    native = np.zeros((16, 16, 3), dtype=np.uint8)
    candidate = np.full((16, 16, 3), 255, dtype=np.uint8)
    distance = cli.lpips_distance(FakeLpips(), native, candidate, "cpu")
    assert distance == pytest.approx(4.0)


def test_lpips_rejects_shape_mismatch_and_nonfinite() -> None:
    with pytest.raises(cli.LpipsConflictError, match="identical shapes"):
        cli.lpips_distance(
            FakeLpips(),
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
            "cpu",
        )
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[0, 0, 0] = np.nan
    with pytest.raises(cli.LpipsConflictError, match="non-finite"):
        cli.lpips_tensor(image, "cpu")
