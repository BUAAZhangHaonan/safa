from __future__ import annotations

import importlib
import json

import pytest


torch = pytest.importorskip("torch")


def _toy_config(toy, tmp_path, **overrides):
    payload = {
        "run_name": "pytest_primal_dual",
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seed": 23,
        "deltas_deg": [45.0],
        "methods": ["primal_dual_projected"],
        "lambdas": [1.0],
        "soft_margins": [0.0],
        "steps": 4,
        "batch_size": 16,
        "eval_batch_size": 32,
        "hidden_dim": 8,
        "layers": 1,
        "sigma": 0.02,
        "k_classes": 4,
        "sample_steps": 2,
        "learning_rate": 0.001,
        "fm_learning_rate": 0.001,
        "repr_learning_rate": 0.001,
        "weight_decay": 0.0,
        "repr_relation_weight": 0.0,
        "normalize_losses": True,
        "calibration_batches": 1,
        "eval_interval": 2,
        "projection_eps": 1e-12,
        "fm_delta_target": 0.01,
        "dual_lr": 10.0,
        "dual_max": 100.0,
        "primal_dual_warmup_steps": 1,
        "cagrad_c": None,
        "fm_descent_floor_fraction": None,
        "fm_budget_fraction": None,
        "uncertainty_log_var_lr": None,
        "uncertainty_log_var_init_fm": None,
        "uncertainty_log_var_init_cl": None,
    }
    payload.update(overrides)
    return toy.ToyConfig(**payload)


def test_toy_config_requires_all_keys(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(json.dumps({"run_name": "missing_fields"}), encoding="utf-8")

    with pytest.raises(KeyError, match="Missing required config keys"):
        toy.load_config(config_path)


def test_primal_dual_config_requires_explicit_dual_fields(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _toy_config(toy, tmp_path).__dict__.copy()
    del payload["fm_delta_target"]
    config_path = tmp_path / "missing_dual_field.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError, match="fm_delta_target"):
        toy.load_config(config_path)


def test_primal_dual_projected_must_run_standalone(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(toy, tmp_path, methods=["primal_dual_projected", "weighted_sum"])

    with pytest.raises(ValueError, match="standalone"):
        toy.validate_config(config)


def test_primal_dual_projected_requires_unit_lambda(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(toy, tmp_path, lambdas=[0.3])

    with pytest.raises(ValueError, match=r"lambdas must be \[1\.0\]"):
        toy.validate_config(config)


@pytest.mark.parametrize("field", ["dual_lr", "dual_max"])
def test_primal_dual_projected_requires_positive_dual_parameters(tmp_path, field: str) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(toy, tmp_path, **{field: 0.0})

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        toy.validate_config(config)


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


def test_cagrad_two_task_aggregation_matches_documented_example() -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    g_fm = [torch.tensor([-4.0, 1.0, 1.0])]
    g_cl = [torch.tensor([6.0, 1.0, 1.0])]

    result = toy.aggregate_two_task_cagrad(g_fm, g_cl, c=0.5, eps=1e-12)

    assert result.fm_weight == pytest.approx(1.0)
    assert result.cl_weight == pytest.approx(0.0)
    assert torch.allclose(result.combined_gradients[0], torch.tensor([0.1835, 1.2041, 1.2041]), atol=1e-4)
    assert result.combined_norm > 0.0
    assert result.gradient_cosine == pytest.approx(torch.nn.functional.cosine_similarity(g_fm[0], g_cl[0], dim=0).item())


def test_cagrad_config_requires_explicit_c_value(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(toy, tmp_path, methods=["cagrad"], lambdas=[1.0])

    with pytest.raises(ValueError, match="cagrad requires cagrad_c"):
        toy.validate_config(config)


def test_fm_anchored_cagrad_raises_fm_weight_when_cagrad_degenerates() -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    g_fm = [torch.tensor([6.0, 1.0, 1.0])]
    g_cl = [torch.tensor([-4.0, 1.0, 1.0])]

    raw = toy.aggregate_two_task_cagrad(g_fm, g_cl, c=0.5, eps=1e-12)
    anchored = toy.aggregate_two_task_fm_anchored_cagrad(
        g_fm,
        g_cl,
        c=0.5,
        fm_descent_floor_fraction=0.5,
        eps=1e-12,
    )

    assert raw.fm_weight == pytest.approx(0.0)
    assert anchored.anchor_active is True
    assert anchored.fm_weight > raw.fm_weight
    assert anchored.cl_weight < raw.cl_weight
    assert anchored.fm_descent_after_anchor >= anchored.fm_descent_floor - 1e-6


def test_fm_anchored_cagrad_keeps_direction_when_floor_is_already_met() -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    g_fm = [torch.tensor([1.0, 0.0])]
    g_cl = [torch.tensor([1.0, 0.0])]

    raw = toy.aggregate_two_task_cagrad(g_fm, g_cl, c=0.5, eps=1e-12)
    anchored = toy.aggregate_two_task_fm_anchored_cagrad(
        g_fm,
        g_cl,
        c=0.5,
        fm_descent_floor_fraction=0.5,
        eps=1e-12,
    )

    assert anchored.anchor_active is False
    assert anchored.fm_weight == pytest.approx(raw.fm_weight)
    assert anchored.cl_weight == pytest.approx(raw.cl_weight)
    assert torch.allclose(anchored.combined_gradients[0], raw.combined_gradients[0])


def test_uncertainty_weighted_config_requires_explicit_log_var_fields(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(toy, tmp_path, methods=["uncertainty_weighted"], lambdas=[1.0])

    with pytest.raises(ValueError, match="uncertainty_weighted requires"):
        toy.validate_config(config)


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
        fm_delta_target=0.01,
        dual_lr=10.0,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
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
        fm_delta_target=0.01,
        dual_lr=10.0,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
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
        fm_delta_target=0.01,
        dual_lr=10.0,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
    )

    summary = toy.run_experiment_grid(config)
    observed = {item["method"] for item in summary["experiments"]}

    assert observed == {"weighted_sum", "pcgrad", "soft_margin_projected"}


def test_toy_grid_runs_cagrad_and_logs_required_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(
        toy,
        tmp_path,
        run_name="pytest_cagrad",
        methods=["cagrad"],
        cagrad_c=0.5,
        primal_dual_warmup_steps=0,
    )

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_cagrad"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_initial = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "cagrad"
    for key in [
        "cagrad_fm_weight",
        "cagrad_cl_weight",
        "gradient_cosine_mean",
        "combined_grad_norm",
        "valid_fm_loss",
        "repr_cosine_mean",
    ]:
        assert key in non_initial
    assert non_initial["combined_grad_norm"] > 0.0
    assert non_initial["cagrad_fm_weight"] + non_initial["cagrad_cl_weight"] == pytest.approx(1.0)


def test_toy_grid_runs_fm_anchored_cagrad_and_logs_floor_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(
        toy,
        tmp_path,
        run_name="pytest_fm_anchored_cagrad",
        methods=["fm_anchored_cagrad"],
        cagrad_c=0.5,
        fm_descent_floor_fraction=0.25,
        primal_dual_warmup_steps=0,
    )

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_fm_anchored_cagrad"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_initial = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "fm_anchored_cagrad"
    for key in [
        "cagrad_fm_weight",
        "cagrad_cl_weight",
        "cagrad_raw_fm_weight",
        "cagrad_raw_cl_weight",
        "fm_descent_floor",
        "fm_descent_after_cagrad",
        "fm_descent_after_anchor",
        "fm_anchor_active",
        "gradient_cosine_mean",
        "combined_grad_norm",
    ]:
        assert key in non_initial
    assert non_initial["cagrad_fm_weight"] + non_initial["cagrad_cl_weight"] == pytest.approx(1.0)
    assert non_initial["fm_descent_after_anchor"] >= non_initial["fm_descent_floor"] - 1e-6


def test_toy_grid_runs_uncertainty_weighted_and_logs_formula_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(
        toy,
        tmp_path,
        run_name="pytest_uncertainty",
        methods=["uncertainty_weighted"],
        uncertainty_log_var_lr=0.01,
        uncertainty_log_var_init_fm=0.0,
        uncertainty_log_var_init_cl=0.0,
        primal_dual_warmup_steps=0,
    )

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_uncertainty"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_initial = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "uncertainty_weighted"
    for key in [
        "uncertainty_fm_log_var",
        "uncertainty_cl_log_var",
        "uncertainty_fm_weight",
        "uncertainty_cl_weight",
        "uncertainty_formula_total",
    ]:
        assert key in non_initial
    assert non_initial["uncertainty_fm_weight"] == pytest.approx(0.5, abs=0.1)
    assert non_initial["uncertainty_cl_weight"] == pytest.approx(0.5, abs=0.1)


def test_primal_dual_projected_smoke_writes_non_initial_dual_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = _toy_config(toy, tmp_path)

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_primal_dual"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_initial = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "primal_dual_projected"
    for key in [
        "dual_value_used",
        "dual_value_next",
        "effective_repr_lr",
        "primal_dual_violation",
        "primal_dual_warmup_active",
        "actual_fm_delta_after_repr_step",
        "projected_repr_norm_ratio",
    ]:
        assert key in non_initial
    assert non_initial["effective_repr_lr"] > 0.0
