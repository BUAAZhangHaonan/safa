from __future__ import annotations

import importlib
import json

import pytest


torch = pytest.importorskip("torch")


def test_toy_config_requires_all_keys(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(json.dumps({"run_name": "missing_fields"}), encoding="utf-8")

    with pytest.raises(KeyError, match="Missing required config keys"):
        toy.load_config(config_path)


def test_soft_margin_projection_allows_explicit_fm_budget() -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    g_repr = [torch.tensor([1.0, 0.0])]
    g_fm = [torch.tensor([-1.0, 0.0])]

    hard = toy.project_gradient_with_soft_margin(g_repr, g_fm, epsilon=0.0, eps=1e-12)
    soft = toy.project_gradient_with_soft_margin(g_repr, g_fm, epsilon=0.25, eps=1e-12)

    assert hard.projection_applied is True
    assert torch.allclose(hard.dot_after, torch.tensor(0.0), atol=1e-6)
    assert soft.projection_applied is True
    assert torch.allclose(soft.dot_after, torch.tensor(-0.25), atol=1e-6)
    assert soft.fm_first_order_effect > 0


def test_delta_zero_projected_smoke_improves_fm_and_repr(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_delta0",
        output_dir=str(tmp_path),
        device="cpu",
        seed=7,
        deltas_deg=[0.0],
        methods=["projected_two_step"],
        lambdas=[0.1],
        soft_margins=[0.0],
        steps=80,
        batch_size=64,
        eval_batch_size=128,
        hidden_dim=16,
        layers=2,
        sigma=0.02,
        k_classes=8,
        sample_steps=4,
        learning_rate=0.003,
        fm_learning_rate=0.003,
        repr_learning_rate=0.003,
        weight_decay=0.0,
        repr_relation_weight=0.0,
        normalize_losses=True,
        calibration_batches=2,
        eval_interval=40,
        projection_eps=1e-12,
    )

    result = toy.run_single_experiment(config, delta_deg=0.0, method="projected_two_step", lambda_repr=0.1, soft_margin=0.0)

    assert result["final"]["valid_fm_loss"] < result["initial"]["valid_fm_loss"]
    assert result["final"]["repr_point_loss"] < result["initial"]["repr_point_loss"]
    assert result["final"]["repr_cosine_mean"] > result["initial"]["repr_cosine_mean"]
    assert result["final"]["conflict_fraction"] >= 0.0
    assert "projected_repr_norm_ratio" in result["final"]


def test_toy_grid_writes_required_outputs(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_grid",
        output_dir=str(tmp_path),
        device="cpu",
        seed=11,
        deltas_deg=[0.0],
        methods=["fm_only", "repr_only"],
        lambdas=[0.1],
        soft_margins=[0.0],
        steps=4,
        batch_size=16,
        eval_batch_size=32,
        hidden_dim=8,
        layers=1,
        sigma=0.02,
        k_classes=4,
        sample_steps=2,
        learning_rate=0.001,
        fm_learning_rate=0.001,
        repr_learning_rate=0.001,
        weight_decay=0.0,
        repr_relation_weight=0.0,
        normalize_losses=True,
        calibration_batches=1,
        eval_interval=2,
        projection_eps=1e-12,
    )

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_grid"

    assert summary["run_name"] == "pytest_grid"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "curves.png").is_file()
    assert (run_dir / "trajectory.png").is_file()

    first_metric = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for key in [
        "valid_fm_loss",
        "generation_mse_to_x1",
        "repr_point_loss",
        "repr_cosine_mean",
        "conflict_fraction",
    ]:
        assert key in first_metric


def test_toy_grid_runs_weighted_pcgrad_and_soft_margin(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_methods",
        output_dir=str(tmp_path),
        device="cpu",
        seed=19,
        deltas_deg=[15.0],
        methods=["weighted_sum", "pcgrad", "soft_margin_projected"],
        lambdas=[0.1],
        soft_margins=[0.05],
        steps=4,
        batch_size=16,
        eval_batch_size=32,
        hidden_dim=8,
        layers=1,
        sigma=0.02,
        k_classes=4,
        sample_steps=2,
        learning_rate=0.001,
        fm_learning_rate=0.001,
        repr_learning_rate=0.001,
        weight_decay=0.0,
        repr_relation_weight=0.0,
        normalize_losses=True,
        calibration_batches=1,
        eval_interval=2,
        projection_eps=1e-12,
    )

    summary = toy.run_experiment_grid(config)
    observed = {item["method"] for item in summary["experiments"]}

    assert observed == {"weighted_sum", "pcgrad", "soft_margin_projected"}
