from __future__ import annotations

import importlib
import inspect
import json

import pytest


torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# PGC helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Master payload helpers
# ---------------------------------------------------------------------------


def _adaptive_trust_payload(tmp_path) -> dict:
    return {
        "run_name": "adaptive_trust_ok",
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seed": 1,
        "deltas_deg": [0.0],
        "methods": ["adaptive_trust_projected"],
        "lambdas": [1.0],
        "soft_margins": [0.02],
        "steps": 2,
        "batch_size": 8,
        "eval_batch_size": 8,
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
        "eval_interval": 1,
        "projection_eps": 1e-12,
        "adaptive_margin_mode": "target",
        "adaptive_margin_target": 1.0,
        "adaptive_margin_ema_beta": None,
        "adaptive_margin_step": 0.01,
        "adaptive_margin_min": 0.0,
        "adaptive_margin_max": 0.05,
        "adaptive_margin_initial": 0.02,
        "fm_delta_target": 0.01,
        "dual_lr": 0.5,
        "dual_max": 100.0,
        "primal_dual_warmup_steps": 0,
        "trust_radius_initial": 1.0,
        "trust_radius_min": 0.25,
        "trust_radius_max": 2.0,
    }


def _line_search_payload(tmp_path) -> dict:
    return {
        "run_name": "line_search_ok",
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seed": 1,
        "deltas_deg": [0.0],
        "methods": ["line_search_projected"],
        "lambdas": [1.0],
        "soft_margins": [0.0],
        "steps": 2,
        "batch_size": 8,
        "eval_batch_size": 8,
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
        "eval_interval": 1,
        "projection_eps": 1e-12,
        "fm_delta_target": 1.0,
        "dual_lr": 10.0,
        "dual_max": 100.0,
        "primal_dual_warmup_steps": 0,
        "line_search_max_backtracks": 4,
        "line_search_contraction": 0.5,
    }


def _dual_step_payload(tmp_path) -> dict:
    return {
        "run_name": "dual_step_ok",
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seed": 1,
        "deltas_deg": [0.0],
        "methods": ["dual_step_projected"],
        "lambdas": [1.0],
        "soft_margins": [0.0],
        "steps": 2,
        "batch_size": 8,
        "eval_batch_size": 8,
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
        "eval_interval": 1,
        "projection_eps": 1e-12,
        "fm_delta_target": 0.01,
        "dual_lr": 10.0,
        "dual_max": 100.0,
        "primal_dual_warmup_steps": 1,
    }


def _budgeted_cl_payload(tmp_path) -> dict:
    return {
        "run_name": "budgeted_cl_ok",
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seed": 1,
        "deltas_deg": [0.0],
        "methods": ["budgeted_cl_line_search"],
        "lambdas": [1.0],
        "soft_margins": [0.0],
        "steps": 2,
        "batch_size": 8,
        "eval_batch_size": 8,
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
        "eval_interval": 1,
        "projection_eps": 1e-12,
        "fm_delta_target": 1.0,
        "dual_lr": 10.0,
        "dual_max": 100.0,
        "primal_dual_warmup_steps": 0,
        "line_search_max_backtracks": 4,
        "line_search_contraction": 0.5,
    }


def _descent_credit_payload(tmp_path) -> dict:
    return {
        "run_name": "descent_credit_ok",
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seed": 1,
        "deltas_deg": [0.0],
        "methods": ["descent_credit_projected"],
        "lambdas": [1.0],
        "soft_margins": [0.0],
        "steps": 2,
        "batch_size": 8,
        "eval_batch_size": 8,
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
        "eval_interval": 1,
        "projection_eps": 1e-12,
        "fm_delta_target": 0.01,
        "dual_lr": 10.0,
        "dual_max": 100.0,
        "primal_dual_warmup_steps": 0,
    }


def _descent_scaled_payload(tmp_path) -> dict:
    payload = _descent_credit_payload(tmp_path)
    payload["run_name"] = "descent_scaled_ok"
    payload["methods"] = ["descent_credit_scaled"]
    return payload


# ---------------------------------------------------------------------------
# PGC tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Master tests
# ---------------------------------------------------------------------------


def test_adaptive_method_requires_explicit_margin_fields(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "adaptive_missing_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "adaptive_missing",
                "output_dir": str(tmp_path),
                "device": "cpu",
                "seed": 1,
                "deltas_deg": [0.0],
                "methods": ["adaptive_margin_projected"],
                "lambdas": [1.0],
                "soft_margins": [0.02],
                "steps": 2,
                "batch_size": 8,
                "eval_batch_size": 8,
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
                "eval_interval": 1,
                "projection_eps": 1e-12,
                "fm_delta_target": 0.01,
                "dual_lr": 10.0,
                "dual_max": 100.0,
                "primal_dual_warmup_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="adaptive_margin_projected requires"):
        toy.load_config(config_path)


def test_adaptive_method_load_config_accepts_explicit_margin_fields(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "adaptive_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "adaptive_ok",
                "output_dir": str(tmp_path),
                "device": "cpu",
                "seed": 1,
                "deltas_deg": [0.0],
                "methods": ["adaptive_margin_projected"],
                "lambdas": [1.0],
                "soft_margins": [0.02],
                "steps": 2,
                "batch_size": 8,
                "eval_batch_size": 8,
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
                "eval_interval": 1,
                "projection_eps": 1e-12,
                "adaptive_margin_mode": "target",
                "adaptive_margin_target": 1.0,
                "adaptive_margin_ema_beta": None,
                "adaptive_margin_step": 0.01,
                "adaptive_margin_min": 0.0,
                "adaptive_margin_max": 0.05,
                "adaptive_margin_initial": 0.02,
                "fm_delta_target": 0.01,
                "dual_lr": 10.0,
                "dual_max": 100.0,
                "primal_dual_warmup_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    config = toy.load_config(config_path)

    assert config.methods == ["adaptive_margin_projected"]
    assert config.adaptive_margin_initial == pytest.approx(0.02)


def test_adaptive_method_rejects_lambda_sweep(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "adaptive_bad_lambda.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "adaptive_bad_lambda",
                "output_dir": str(tmp_path),
                "device": "cpu",
                "seed": 1,
                "deltas_deg": [0.0],
                "methods": ["adaptive_margin_projected"],
                "lambdas": [0.1],
                "soft_margins": [0.02],
                "steps": 2,
                "batch_size": 8,
                "eval_batch_size": 8,
                "hidden_dim": 8,
                "layers": 1,
                "sigma": 0.1,
                "k_classes": 4,
                "sample_steps": 2,
                "learning_rate": 0.001,
                "fm_learning_rate": 0.001,
                "repr_learning_rate": 0.001,
                "weight_decay": 0.0,
                "repr_relation_weight": 0.0,
                "normalize_losses": True,
                "calibration_batches": 1,
                "eval_interval": 1,
                "projection_eps": 1.0e-12,
                "adaptive_margin_mode": "target",
                "adaptive_margin_target": 1.0,
                "adaptive_margin_ema_beta": None,
                "adaptive_margin_step": 0.01,
                "adaptive_margin_min": 0.0,
                "adaptive_margin_max": 0.05,
                "adaptive_margin_initial": 0.02,
                "fm_delta_target": 0.01,
                "dual_lr": 10.0,
                "dual_max": 100.0,
                "primal_dual_warmup_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed lambda_repr=1.0"):
        toy.load_config(config_path)


def test_adaptive_method_rejects_soft_margin_mismatch(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "adaptive_bad_margin.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "adaptive_bad_margin",
                "output_dir": str(tmp_path),
                "device": "cpu",
                "seed": 1,
                "deltas_deg": [0.0],
                "methods": ["adaptive_margin_projected"],
                "lambdas": [1.0],
                "soft_margins": [0.03],
                "steps": 2,
                "batch_size": 8,
                "eval_batch_size": 8,
                "hidden_dim": 8,
                "layers": 1,
                "sigma": 0.1,
                "k_classes": 4,
                "sample_steps": 2,
                "learning_rate": 0.001,
                "fm_learning_rate": 0.001,
                "repr_learning_rate": 0.001,
                "weight_decay": 0.0,
                "repr_relation_weight": 0.0,
                "normalize_losses": True,
                "calibration_batches": 1,
                "eval_interval": 1,
                "projection_eps": 1.0e-12,
                "adaptive_margin_mode": "target",
                "adaptive_margin_target": 1.0,
                "adaptive_margin_ema_beta": None,
                "adaptive_margin_step": 0.01,
                "adaptive_margin_min": 0.0,
                "adaptive_margin_max": 0.05,
                "adaptive_margin_initial": 0.02,
                "fm_delta_target": 0.01,
                "dual_lr": 10.0,
                "dual_max": 100.0,
                "primal_dual_warmup_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="soft_margins\\[0\\] must equal adaptive_margin_initial"):
        toy.load_config(config_path)


def test_adaptive_method_rejects_mixed_method_grid(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "adaptive_mixed_grid.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "adaptive_mixed_grid",
                "output_dir": str(tmp_path),
                "device": "cpu",
                "seed": 1,
                "deltas_deg": [0.0],
                "methods": ["weighted_sum", "adaptive_margin_projected"],
                "lambdas": [1.0],
                "soft_margins": [0.02],
                "steps": 2,
                "batch_size": 8,
                "eval_batch_size": 8,
                "hidden_dim": 8,
                "layers": 1,
                "sigma": 0.1,
                "k_classes": 4,
                "sample_steps": 2,
                "learning_rate": 0.001,
                "fm_learning_rate": 0.001,
                "repr_learning_rate": 0.001,
                "weight_decay": 0.0,
                "repr_relation_weight": 0.0,
                "normalize_losses": True,
                "calibration_batches": 1,
                "eval_interval": 1,
                "projection_eps": 1.0e-12,
                "adaptive_margin_mode": "target",
                "adaptive_margin_target": 1.0,
                "adaptive_margin_ema_beta": None,
                "adaptive_margin_step": 0.01,
                "adaptive_margin_min": 0.0,
                "adaptive_margin_max": 0.05,
                "adaptive_margin_initial": 0.02,
                "fm_delta_target": 0.01,
                "dual_lr": 10.0,
                "dual_max": 100.0,
                "primal_dual_warmup_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dedicated config"):
        toy.load_config(config_path)


def test_adaptive_trust_projected_requires_explicit_trust_fields_and_standalone_config(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    missing_path = tmp_path / "adaptive_trust_missing_config.json"
    payload = _adaptive_trust_payload(tmp_path)
    for key in ["trust_radius_initial", "trust_radius_min", "trust_radius_max"]:
        payload.pop(key)
    missing_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptive_trust_projected requires"):
        toy.load_config(missing_path)

    mixed_path = tmp_path / "adaptive_trust_mixed_config.json"
    mixed_payload = _adaptive_trust_payload(tmp_path)
    mixed_payload["methods"] = ["weighted_sum", "adaptive_trust_projected"]
    mixed_path.write_text(json.dumps(mixed_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptive_trust_projected requires a standalone config"):
        toy.load_config(mixed_path)


def test_adaptive_trust_projected_load_config_accepts_explicit_fields(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "adaptive_trust_config.json"
    config_path.write_text(json.dumps(_adaptive_trust_payload(tmp_path)), encoding="utf-8")

    config = toy.load_config(config_path)

    assert config.methods == ["adaptive_trust_projected"]
    assert config.lambdas == [1.0]
    assert config.soft_margins[0] == pytest.approx(config.adaptive_margin_initial)
    assert config.fm_delta_target == pytest.approx(0.01)
    assert config.dual_lr == pytest.approx(0.5)
    assert config.trust_radius_initial == pytest.approx(1.0)


def test_line_search_projected_requires_explicit_line_search_fields(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "line_search_missing_config.json"
    payload = _line_search_payload(tmp_path)
    for key in ["line_search_max_backtracks", "line_search_contraction"]:
        payload.pop(key)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="line_search_projected requires explicit config fields"):
        toy.load_config(config_path)


def test_line_search_projected_rejects_mixed_method_grid(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "line_search_mixed_methods.json"
    payload = _line_search_payload(tmp_path)
    payload["methods"] = ["weighted_sum", "line_search_projected"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="line_search_projected requires a standalone config"):
        toy.load_config(config_path)


def test_line_search_projected_rejects_lambda_sweep(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "line_search_bad_lambda.json"
    payload = _line_search_payload(tmp_path)
    payload["lambdas"] = [0.1, 1.0]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"line_search_projected requires lambdas to equal \[1\.0\]"):
        toy.load_config(config_path)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_line_search_projected_rejects_invalid_contraction(tmp_path, value: float) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "line_search_bad_contraction.json"
    payload = _line_search_payload(tmp_path)
    payload["line_search_contraction"] = value
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="line_search_contraction must be in \\(0, 1\\)"):
        toy.load_config(config_path)


def test_line_search_helper_shrinks_alpha_after_first_candidate_fails() -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    observed_alphas: list[float] = []

    def evaluate(alpha: float) -> tuple[float, float]:
        observed_alphas.append(alpha)
        if alpha == pytest.approx(1.0):
            return 1.2, 0.8
        return 1.01, 0.8

    result = toy._find_line_search_acceptance(
        normalized_flow_before=1.0,
        normalized_repr_before=1.0,
        fm_delta_target=0.05,
        max_backtracks=3,
        contraction=0.5,
        evaluate_candidate=evaluate,
    )

    assert observed_alphas == [1.0, 0.5]
    assert result.accepted is True
    assert result.attempts == 2
    assert result.alpha == pytest.approx(0.5)
    assert result.flow_delta == pytest.approx(0.01)
    assert result.repr_delta == pytest.approx(-0.2)


def test_dual_step_projected_rejects_mixed_method_grid(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "dual_step_mixed_methods.json"
    payload = _dual_step_payload(tmp_path)
    payload["methods"] = ["weighted_sum", "dual_step_projected"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dual_step_projected requires a standalone config"):
        toy.load_config(config_path)


def test_dual_step_projected_rejects_lambda_sweep(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "dual_step_bad_lambda.json"
    payload = _dual_step_payload(tmp_path)
    payload["lambdas"] = [0.1, 1.0]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"dual_step_projected requires lambdas to equal \[1\.0\]"):
        toy.load_config(config_path)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("dual_lr", 0.0, "dual_lr must be positive"),
        ("dual_lr", -1.0, "dual_lr must be positive"),
        ("dual_max", 0.0, "dual_max must be positive"),
        ("dual_max", -1.0, "dual_max must be positive"),
        ("primal_dual_warmup_steps", -1, "primal_dual_warmup_steps must be a non-negative integer"),
    ],
)
def test_dual_step_projected_rejects_invalid_dual_parameters(tmp_path, key: str, value, match: str) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / f"dual_step_bad_{key}.json"
    payload = _dual_step_payload(tmp_path)
    payload[key] = value
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        toy.load_config(config_path)


def test_budgeted_cl_line_search_requires_explicit_fields(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "budgeted_cl_missing_config.json"
    payload = _budgeted_cl_payload(tmp_path)
    for key in ["line_search_max_backtracks", "line_search_contraction"]:
        payload.pop(key)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="budgeted_cl_line_search requires explicit config fields"):
        toy.load_config(config_path)


def test_budgeted_cl_line_search_rejects_mixed_method_grid(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "budgeted_cl_mixed_methods.json"
    payload = _budgeted_cl_payload(tmp_path)
    payload["methods"] = ["weighted_sum", "budgeted_cl_line_search"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="budgeted_cl_line_search requires a standalone config"):
        toy.load_config(config_path)


def test_budgeted_cl_line_search_rejects_lambda_sweep(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "budgeted_cl_bad_lambda.json"
    payload = _budgeted_cl_payload(tmp_path)
    payload["lambdas"] = [0.1, 1.0]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"budgeted_cl_line_search requires lambdas to equal \[1\.0\]"):
        toy.load_config(config_path)


def test_budgeted_cl_line_search_direct_call_rejects_non_unit_lambda(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(**_budgeted_cl_payload(tmp_path))

    with pytest.raises(ValueError, match="budgeted_cl_line_search uses fixed lambda_repr=1.0"):
        toy.run_single_experiment(
            config,
            delta_deg=15.0,
            method="budgeted_cl_line_search",
            lambda_repr=0.1,
            soft_margin=0.0,
        )


def test_budgeted_cl_line_search_uses_full_repr_gradient_source() -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    source = inspect.getsource(toy._step_budgeted_cl_line_search)

    assert "_manual_parameter_step(model, g_repr," in source
    assert "projection.projected_gradients" not in source
    assert "project_gradient_onto_fm_feasible_cone" not in source


def test_descent_credit_projected_rejects_mixed_method_grid(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "descent_credit_mixed_methods.json"
    payload = _descent_credit_payload(tmp_path)
    payload["methods"] = ["weighted_sum", "descent_credit_projected"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="descent_credit_projected requires a standalone config"):
        toy.load_config(config_path)


def test_descent_credit_projected_rejects_lambda_sweep(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "descent_credit_bad_lambda.json"
    payload = _descent_credit_payload(tmp_path)
    payload["lambdas"] = [0.1, 1.0]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"descent_credit_projected requires lambdas to equal \[1\.0\]"):
        toy.load_config(config_path)


def test_descent_credit_projected_direct_call_rejects_non_unit_lambda(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(**_descent_credit_payload(tmp_path))

    with pytest.raises(ValueError, match="descent_credit_projected uses fixed lambda_repr=1.0"):
        toy.run_single_experiment(
            config,
            delta_deg=15.0,
            method="descent_credit_projected",
            lambda_repr=0.1,
            soft_margin=0.0,
        )


def test_descent_credit_scaled_rejects_mixed_method_grid(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "descent_scaled_mixed_methods.json"
    payload = _descent_scaled_payload(tmp_path)
    payload["methods"] = ["weighted_sum", "descent_credit_scaled"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="descent_credit_scaled requires a standalone config"):
        toy.load_config(config_path)


def test_descent_credit_scaled_rejects_lambda_sweep(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config_path = tmp_path / "descent_scaled_bad_lambda.json"
    payload = _descent_scaled_payload(tmp_path)
    payload["lambdas"] = [0.1, 1.0]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"descent_credit_scaled requires lambdas to equal \[1\.0\]"):
        toy.load_config(config_path)


def test_descent_credit_scaled_direct_call_rejects_non_unit_lambda(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(**_descent_scaled_payload(tmp_path))

    with pytest.raises(ValueError, match="descent_credit_scaled uses fixed lambda_repr=1.0"):
        toy.run_single_experiment(
            config,
            delta_deg=15.0,
            method="descent_credit_scaled",
            lambda_repr=0.1,
            soft_margin=0.0,
        )


def test_project_gradient_to_dot_lower_bound_projects_conflict() -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    g_repr = [torch.tensor([-1.0, 0.0])]
    g_fm = [torch.tensor([1.0, 0.0])]

    result = toy.project_gradient_to_dot_lower_bound(g_repr, g_fm, lower_bound=-0.25, eps=1e-12)

    assert result.projection_applied
    assert result.dot_before.item() == pytest.approx(-1.0)
    assert result.dot_after.item() == pytest.approx(-0.25)
    assert result.projected_gradients[0].tolist() == pytest.approx([-0.25, 0.0])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("lambdas", [0.1], "lambdas to equal \\[1.0\\]"),
        ("soft_margins", [0.03], "soft_margins\\[0\\] to equal adaptive_margin_initial"),
    ],
)
def test_adaptive_trust_projected_fails_fast_on_lambda_and_margin_mismatch(
    tmp_path,
    field: str,
    value,
    match: str,
) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _adaptive_trust_payload(tmp_path)
    payload[field] = value
    config_path = tmp_path / f"adaptive_trust_bad_{field}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        toy.load_config(config_path)


def test_adaptive_margin_projected_runs_and_records_margin_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_adaptive",
        output_dir=str(tmp_path),
        device="cpu",
        seed=17,
        deltas_deg=[15.0],
        methods=["adaptive_margin_projected"],
        lambdas=[1.0],
        soft_margins=[0.02],
        steps=6,
        batch_size=24,
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
        eval_interval=3,
        projection_eps=1e-12,
        adaptive_margin_mode="target",
        adaptive_margin_target=1.0,
        adaptive_margin_ema_beta=None,
        adaptive_margin_step=0.01,
        adaptive_margin_min=0.0,
        adaptive_margin_max=0.05,
        adaptive_margin_initial=0.02,
        fm_delta_target=0.01,
        dual_lr=10.0,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
    )

    result = toy.run_single_experiment(
        config,
        delta_deg=15.0,
        method="adaptive_margin_projected",
        lambda_repr=1.0,
        soft_margin=0.02,
    )

    assert result["final"]["adaptive_margin"] >= 0.0
    assert result["final"]["adaptive_margin"] <= 0.05
    assert "actual_fm_delta_after_repr_step" in result["final"]
    assert "repr_cosine_mean" in result["final"]
    assert "valid_fm_loss" in result["final"]
    assert "adaptive_normalized_fm_loss" in result["final"]
    assert "adaptive_margin_baseline" in result["final"]
    assert "adaptive_margin_direction" in result["final"]


def test_adaptive_margin_projected_rejects_direct_call_margin_mismatch(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_adaptive_mismatch",
        output_dir=str(tmp_path),
        device="cpu",
        seed=17,
        deltas_deg=[15.0],
        methods=["adaptive_margin_projected"],
        lambdas=[1.0],
        soft_margins=[0.02],
        steps=2,
        batch_size=8,
        eval_batch_size=8,
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
        eval_interval=1,
        projection_eps=1e-12,
        adaptive_margin_mode="target",
        adaptive_margin_target=1.0,
        adaptive_margin_ema_beta=None,
        adaptive_margin_step=0.01,
        adaptive_margin_min=0.0,
        adaptive_margin_max=0.05,
        adaptive_margin_initial=0.02,
        fm_delta_target=0.01,
        dual_lr=10.0,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
    )

    with pytest.raises(ValueError, match="soft_margin must equal adaptive_margin_initial"):
        toy.run_single_experiment(
            config,
            delta_deg=15.0,
            method="adaptive_margin_projected",
            lambda_repr=1.0,
            soft_margin=0.03,
        )


def test_adaptive_margin_projected_rejects_direct_call_lambda_mismatch(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_adaptive_bad_lambda_call",
        output_dir=str(tmp_path),
        device="cpu",
        seed=17,
        deltas_deg=[15.0],
        methods=["adaptive_margin_projected"],
        lambdas=[1.0],
        soft_margins=[0.02],
        steps=2,
        batch_size=8,
        eval_batch_size=8,
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
        eval_interval=1,
        projection_eps=1e-12,
        adaptive_margin_mode="target",
        adaptive_margin_target=1.0,
        adaptive_margin_ema_beta=None,
        adaptive_margin_step=0.01,
        adaptive_margin_min=0.0,
        adaptive_margin_max=0.05,
        adaptive_margin_initial=0.02,
        fm_delta_target=0.01,
        dual_lr=10.0,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
    )

    with pytest.raises(ValueError, match="fixed lambda_repr=1.0"):
        toy.run_single_experiment(
            config,
            delta_deg=15.0,
            method="adaptive_margin_projected",
            lambda_repr=0.1,
            soft_margin=0.02,
        )


def test_adaptive_trust_projected_smoke_run_persists_trust_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_adaptive_trust",
        output_dir=str(tmp_path),
        device="cpu",
        seed=23,
        deltas_deg=[15.0],
        methods=["adaptive_trust_projected"],
        lambdas=[1.0],
        soft_margins=[0.02],
        steps=4,
        batch_size=16,
        eval_batch_size=16,
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
        eval_interval=1,
        projection_eps=1e-12,
        adaptive_margin_mode="target",
        adaptive_margin_target=1.0,
        adaptive_margin_ema_beta=None,
        adaptive_margin_step=0.01,
        adaptive_margin_min=0.0,
        adaptive_margin_max=0.05,
        adaptive_margin_initial=0.02,
        fm_delta_target=0.01,
        dual_lr=0.5,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
        trust_radius_initial=1.0,
        trust_radius_min=0.25,
        trust_radius_max=2.0,
    )

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_adaptive_trust"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trained_metric = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "adaptive_trust_projected"
    for key in [
        "trust_radius_used",
        "trust_radius_next",
        "trust_scale",
        "trust_region_active",
        "dual_value_used",
        "dual_value_next",
        "fm_budget_violation",
        "fm_delta_target",
        "scaled_dot_after_mean",
        "repr_step_fm_first_order_effect",
    ]:
        assert key in summary["experiments"][0]["final"]
        assert key in trained_metric
    assert "trust_radius" not in trained_metric
    assert "dual_value" not in trained_metric
    assert trained_metric["fm_delta_target"] == pytest.approx(0.01)
    assert trained_metric["trust_radius_used"] == pytest.approx(1.0)
    assert trained_metric["trust_radius_next"] >= 0.25
    assert trained_metric["trust_radius_next"] <= 2.0
    assert trained_metric["scaled_dot_after_mean"] == pytest.approx(
        -trained_metric["repr_step_fm_first_order_effect"] / config.repr_learning_rate
    )


def test_adaptive_trust_projected_direct_call_rejects_parameter_mismatch(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_adaptive_trust_direct",
        output_dir=str(tmp_path),
        device="cpu",
        seed=29,
        deltas_deg=[15.0],
        methods=["adaptive_trust_projected"],
        lambdas=[1.0],
        soft_margins=[0.02],
        steps=2,
        batch_size=8,
        eval_batch_size=8,
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
        eval_interval=1,
        projection_eps=1e-12,
        adaptive_margin_mode="target",
        adaptive_margin_target=1.0,
        adaptive_margin_ema_beta=None,
        adaptive_margin_step=0.01,
        adaptive_margin_min=0.0,
        adaptive_margin_max=0.05,
        adaptive_margin_initial=0.02,
        fm_delta_target=0.01,
        dual_lr=0.5,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
        trust_radius_initial=1.0,
        trust_radius_min=0.25,
        trust_radius_max=2.0,
    )

    with pytest.raises(ValueError, match="adaptive_trust_projected uses fixed lambda_repr=1.0"):
        toy.run_single_experiment(
            config,
            delta_deg=15.0,
            method="adaptive_trust_projected",
            lambda_repr=0.1,
            soft_margin=0.02,
        )


def test_line_search_projected_smoke_run_persists_accepted_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _line_search_payload(tmp_path)
    payload.update(
        {
            "run_name": "pytest_line_search",
            "seed": 31,
            "deltas_deg": [15.0],
            "steps": 3,
            "batch_size": 16,
            "eval_batch_size": 16,
            "fm_delta_target": 1.0,
            "repr_learning_rate": 0.001,
            "eval_interval": 1,
        }
    )
    config = toy.ToyConfig(**payload)

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_line_search"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trained_metric = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "line_search_projected"
    for key in [
        "line_search_attempts",
        "line_search_alpha",
        "line_search_accepted",
        "line_search_flow_delta",
        "line_search_repr_delta",
        "dot_before_mean",
        "dot_after_mean",
        "projected_repr_norm_ratio",
    ]:
        assert key in summary["experiments"][0]["final"]
        assert key in trained_metric
    assert trained_metric["line_search_accepted"] == pytest.approx(1.0)
    assert trained_metric["line_search_alpha"] > 0.0
    assert trained_metric["line_search_attempts"] >= 1.0
    assert trained_metric["line_search_flow_delta"] <= config.fm_delta_target
    assert trained_metric["line_search_repr_delta"] < 0.0


def test_dual_step_projected_smoke_run_persists_dual_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _dual_step_payload(tmp_path)
    payload.update(
        {
            "run_name": "pytest_dual_step",
            "seed": 41,
            "deltas_deg": [15.0],
            "steps": 3,
            "batch_size": 16,
            "eval_batch_size": 16,
            "eval_interval": 1,
        }
    )
    config = toy.ToyConfig(**payload)

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_dual_step"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trained_metric = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "dual_step_projected"
    for key in [
        "dual_value_used",
        "dual_value_next",
        "effective_repr_lr",
        "primal_dual_violation",
        "primal_dual_warmup_active",
        "actual_fm_delta_after_repr_step",
        "fm_budget_violation",
    ]:
        assert key in summary["experiments"][0]["final"]
        assert key in trained_metric
    assert trained_metric["effective_repr_lr"] > 0.0
    assert trained_metric["primal_dual_warmup_active"] >= 0.0


def test_budgeted_cl_line_search_smoke_run_persists_full_direction_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _budgeted_cl_payload(tmp_path)
    payload.update(
        {
            "run_name": "pytest_budgeted_cl",
            "seed": 51,
            "deltas_deg": [15.0],
            "steps": 3,
            "batch_size": 16,
            "eval_batch_size": 16,
            "fm_delta_target": 1.0,
            "repr_learning_rate": 0.001,
            "eval_interval": 1,
        }
    )
    config = toy.ToyConfig(**payload)

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_budgeted_cl"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trained_metric = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "budgeted_cl_line_search"
    for key in [
        "line_search_attempts",
        "line_search_alpha",
        "line_search_accepted",
        "line_search_flow_delta",
        "line_search_repr_delta",
        "budgeted_direction_norm_ratio",
    ]:
        assert key in summary["experiments"][0]["final"]
        assert key in trained_metric
    assert trained_metric["line_search_accepted"] == pytest.approx(1.0)
    assert trained_metric["line_search_flow_delta"] <= config.fm_delta_target
    assert trained_metric["line_search_repr_delta"] < 0.0
    assert trained_metric["budgeted_direction_norm_ratio"] == pytest.approx(1.0)


def test_descent_credit_projected_smoke_run_persists_credit_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _descent_credit_payload(tmp_path)
    payload.update(
        {
            "run_name": "pytest_descent_credit",
            "seed": 61,
            "deltas_deg": [15.0],
            "steps": 3,
            "batch_size": 16,
            "eval_batch_size": 16,
            "repr_learning_rate": 0.001,
            "eval_interval": 1,
        }
    )
    config = toy.ToyConfig(**payload)

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_descent_credit"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trained_metric = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "descent_credit_projected"
    for key in [
        "fm_descent_credit",
        "credit_dot_lower_bound",
        "credit_budget_used_fraction",
        "net_fm_delta_after_two_step",
        "actual_fm_delta_after_repr_step",
    ]:
        assert key in summary["experiments"][0]["final"]
        assert key in trained_metric
    assert trained_metric["fm_descent_credit"] >= 0.0
    assert trained_metric["credit_dot_lower_bound"] <= 0.0
    assert trained_metric["credit_budget_used_fraction"] >= 0.0
    expected_lower_bound = -trained_metric["fm_descent_credit"] / config.repr_learning_rate
    assert trained_metric["credit_dot_lower_bound"] == pytest.approx(expected_lower_bound, abs=1.0e-6)
    assert trained_metric["dot_after_mean"] + 1.0e-6 >= trained_metric["credit_dot_lower_bound"]
    if trained_metric["fm_descent_credit"] > 0.0:
        assert trained_metric["credit_budget_used_fraction"] <= 1.0 + 1.0e-5


def test_descent_credit_scaled_smoke_run_persists_scale_metrics(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _descent_scaled_payload(tmp_path)
    payload.update(
        {
            "run_name": "pytest_descent_scaled",
            "seed": 71,
            "deltas_deg": [15.0],
            "steps": 3,
            "batch_size": 16,
            "eval_batch_size": 16,
            "repr_learning_rate": 0.001,
            "eval_interval": 1,
        }
    )
    config = toy.ToyConfig(**payload)

    summary = toy.run_experiment_grid(config)
    run_dir = tmp_path / "pytest_descent_scaled"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trained_metric = next(row for row in rows if row["step"] > 0)

    assert summary["experiments"][0]["method"] == "descent_credit_scaled"
    for key in [
        "fm_descent_credit",
        "credit_dot_lower_bound",
        "credit_budget_used_fraction",
        "net_fm_delta_after_two_step",
        "actual_fm_delta_after_repr_step",
        "credit_scale",
        "effective_repr_lr",
    ]:
        assert key in summary["experiments"][0]["final"]
        assert key in trained_metric
    assert 0.0 <= trained_metric["credit_scale"] <= 1.0
    assert trained_metric["effective_repr_lr"] == pytest.approx(config.repr_learning_rate * trained_metric["credit_scale"])
    expected_lower_bound = -trained_metric["fm_descent_credit"] / config.repr_learning_rate
    assert trained_metric["credit_dot_lower_bound"] == pytest.approx(expected_lower_bound, abs=1.0e-6)
    assert trained_metric["dot_after_mean"] + 1.0e-6 >= trained_metric["credit_dot_lower_bound"]
    if trained_metric["fm_descent_credit"] > 0.0:
        assert trained_metric["credit_budget_used_fraction"] <= 1.0 + 1.0e-5


def test_run_single_experiment_invokes_metrics_callback(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    payload = _descent_credit_payload(tmp_path)
    payload.update({"steps": 2, "eval_interval": 1, "batch_size": 8, "eval_batch_size": 8})
    config = toy.ToyConfig(**payload)
    emitted: list[dict] = []

    result = toy.run_single_experiment(
        config,
        delta_deg=15.0,
        method="descent_credit_projected",
        lambda_repr=1.0,
        soft_margin=0.0,
        metrics_callback=emitted.append,
    )

    assert [row["step"] for row in emitted] == [0, 1, 2]
    assert emitted == result["metrics"]


def test_adaptive_margin_projected_ema_mode_records_baseline(tmp_path) -> None:
    toy = importlib.import_module("scripts.run_toy_fm_cl_projected")
    config = toy.ToyConfig(
        run_name="pytest_adaptive_ema",
        output_dir=str(tmp_path),
        device="cpu",
        seed=21,
        deltas_deg=[15.0],
        methods=["adaptive_margin_projected"],
        lambdas=[1.0],
        soft_margins=[0.02],
        steps=4,
        batch_size=16,
        eval_batch_size=16,
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
        adaptive_margin_mode="ema",
        adaptive_margin_target=None,
        adaptive_margin_ema_beta=0.9,
        adaptive_margin_step=0.01,
        adaptive_margin_min=0.0,
        adaptive_margin_max=0.05,
        adaptive_margin_initial=0.02,
        fm_delta_target=0.01,
        dual_lr=10.0,
        dual_max=100.0,
        primal_dual_warmup_steps=0,
    )

    result = toy.run_single_experiment(
        config,
        delta_deg=15.0,
        method="adaptive_margin_projected",
        lambda_repr=1.0,
        soft_margin=0.02,
    )

    assert result["final"]["adaptive_margin_baseline"] > 0.0
    assert result["final"]["adaptive_normalized_fm_loss"] > 0.0
