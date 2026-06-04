from __future__ import annotations

from dataclasses import is_dataclass
import math

import pytest


torch = pytest.importorskip("torch")

from safa.training.projected_update import (
    AdaptiveMarginAdjustment,
    DualBudgetControlResult,
    ProjectionResult,
    TrustRegionScaleResult,
    apply_fm_anchor_trust_region_scaling,
    compute_adaptive_margin_adjustment,
    project_gradient_onto_fm_feasible_cone,
    update_dual_budget_controller,
)


def _dot(left: list["torch.Tensor"], right: list["torch.Tensor"]):
    return sum((a * b).sum() for a, b in zip(left, right))


def test_negative_repr_fm_dot_projects_gradient_to_fm_boundary() -> None:
    g_repr = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, -2.0])]
    g_fm = [torch.tensor([-1.0, 0.0]), torch.tensor([0.0, 0.0])]

    result = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=1e-12)

    assert is_dataclass(ProjectionResult)
    assert isinstance(result, ProjectionResult)
    assert result.projection_applied is True
    assert torch.allclose(result.dot_before, torch.tensor(-1.0))
    assert torch.allclose(result.dot_after, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(result.fm_norm, torch.tensor(1.0))
    assert torch.allclose(result.repr_norm, torch.tensor(math.sqrt(5.0)))
    assert torch.allclose(result.projected_repr_norm, torch.tensor(2.0))
    assert torch.allclose(result.projection_removed_norm, torch.tensor(1.0))
    assert torch.allclose(result.repr_descent_inner_product, torch.tensor(4.0))
    assert torch.allclose(result.fm_first_order_effect, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(_dot(result.projected_gradients, g_fm), torch.tensor(0.0), atol=1e-6)
    assert [tuple(item.shape) for item in result.projected_gradients] == [(2,), (2,)]
    assert torch.allclose(result.projected_gradients[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(result.projected_gradients[1], torch.tensor([0.0, -2.0]))
    assert torch.allclose(g_repr[0], torch.tensor([1.0, 0.0]))


def test_nonnegative_dot_keeps_repr_gradient_unprojected() -> None:
    g_repr = [torch.tensor([1.0, 2.0])]
    g_fm = [torch.tensor([3.0, 0.0])]

    result = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=1e-12)

    assert result.projection_applied is False
    assert torch.allclose(result.dot_before, torch.tensor(3.0))
    assert torch.allclose(result.dot_after, torch.tensor(3.0))
    assert torch.allclose(result.projection_removed_norm, torch.tensor(0.0))
    assert torch.allclose(result.projected_gradients[0], g_repr[0])
    assert torch.allclose(result.fm_first_order_effect, torch.tensor(-3.0))


def test_tiny_fm_gradient_keeps_repr_gradient_unprojected_even_when_conflicting() -> None:
    g_repr = [torch.tensor([1.0, 0.0])]
    g_fm = [torch.tensor([-1e-8, 0.0])]

    result = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=1e-6)

    assert result.projection_applied is False
    assert result.fm_norm <= torch.tensor(1e-6)
    assert result.dot_before < 0
    assert torch.allclose(result.dot_after, result.dot_before)
    assert torch.allclose(result.projected_gradients[0], g_repr[0])


def test_fm_anchor_trust_region_clips_projected_gradient_norm() -> None:
    projected_gradients = [torch.tensor([3.0, 4.0])]
    g_fm = [torch.tensor([0.0, 4.0])]

    result = apply_fm_anchor_trust_region_scaling(
        projected_gradients=projected_gradients,
        g_fm=g_fm,
        trust_radius=0.5,
        eps=1e-12,
    )

    assert is_dataclass(TrustRegionScaleResult)
    assert isinstance(result, TrustRegionScaleResult)
    assert result.trust_region_active is True
    assert result.trust_scale == pytest.approx(0.4)
    assert result.scaled_norm == pytest.approx(2.0)
    assert result.projected_norm == pytest.approx(5.0)
    assert torch.allclose(result.scaled_gradients[0], torch.tensor([1.2, 1.6]))


def test_fm_anchor_trust_region_allows_float32_boundary_roundoff() -> None:
    generator = torch.Generator().manual_seed(10)
    projected_gradients = [torch.randn(10, generator=generator)]
    g_fm = [torch.randn(10, generator=generator)]

    result = apply_fm_anchor_trust_region_scaling(
        projected_gradients=projected_gradients,
        g_fm=g_fm,
        trust_radius=0.7,
        eps=1e-12,
    )

    max_norm = 0.7 * torch.linalg.vector_norm(g_fm[0])
    scaled_norm = torch.linalg.vector_norm(result.scaled_gradients[0])
    assert result.trust_region_active is True
    assert torch.allclose(scaled_norm, max_norm, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("current_dual_value", "current_trust_radius", "actual_fm_delta", "fm_delta_target", "expected_direction"),
    [
        (0.0, 1.0, 0.3, 0.1, "tighten"),
        (0.5, 1.0, 0.05, 0.1, "loosen"),
    ],
)
def test_dual_budget_controller_tightens_and_releases_trust_radius(
    current_dual_value: float,
    current_trust_radius: float,
    actual_fm_delta: float,
    fm_delta_target: float,
    expected_direction: str,
) -> None:
    result = update_dual_budget_controller(
        current_dual_value=current_dual_value,
        current_trust_radius=current_trust_radius,
        actual_fm_delta=actual_fm_delta,
        fm_delta_target=fm_delta_target,
        dual_lr=0.5,
        trust_radius_min=0.25,
        trust_radius_max=2.0,
    )

    assert is_dataclass(DualBudgetControlResult)
    assert isinstance(result, DualBudgetControlResult)
    assert result.direction == expected_direction
    assert result.fm_budget_violation == pytest.approx(actual_fm_delta - fm_delta_target)
    if expected_direction == "tighten":
        assert result.next_dual_value > current_dual_value
        assert result.next_trust_radius < current_trust_radius
    else:
        assert result.next_dual_value < current_dual_value
        assert result.next_trust_radius > current_trust_radius


@pytest.mark.parametrize(
    ("normalized_fm_loss", "baseline", "expected_margin", "expected_direction"),
    [
        (1.2, 1.0, 0.05, "tighten"),
        (0.8, 1.0, 0.11, "loosen"),
    ],
)
def test_adaptive_margin_adjustment_moves_in_expected_direction(
    normalized_fm_loss: float,
    baseline: float,
    expected_margin: float,
    expected_direction: str,
) -> None:
    result = compute_adaptive_margin_adjustment(
        current_margin=0.08,
        normalized_fm_loss=normalized_fm_loss,
        baseline=baseline,
        step=0.03,
        min_margin=0.0,
        max_margin=0.11,
    )

    assert is_dataclass(AdaptiveMarginAdjustment)
    assert isinstance(result, AdaptiveMarginAdjustment)
    assert result.direction == expected_direction
    assert result.next_margin == pytest.approx(expected_margin)
    assert result.baseline == pytest.approx(baseline)
    assert result.normalized_fm_loss == pytest.approx(normalized_fm_loss)


@pytest.mark.parametrize(
    ("g_repr", "g_fm", "eps", "error", "match"),
    [
        ((torch.ones(1),), [torch.ones(1)], 1e-12, TypeError, "list"),
        ([], [], 1e-12, ValueError, "non-empty"),
        ([torch.ones(1)], [torch.ones(1), torch.ones(1)], 1e-12, ValueError, "same length"),
        ([torch.ones(2)], [torch.ones(1)], 1e-12, ValueError, "same shape"),
        ([torch.tensor([float("inf")])], [torch.ones(1)], 1e-12, FloatingPointError, "finite"),
        ([torch.ones(1)], [torch.ones(1)], -1.0, ValueError, "eps"),
    ],
)
def test_project_gradient_rejects_invalid_inputs(g_repr, g_fm, eps: float, error, match: str) -> None:
    with pytest.raises(error, match=match):
        project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=eps)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {
                "projected_gradients": [torch.ones(1)],
                "g_fm": [torch.ones(1)],
                "trust_radius": -0.1,
                "eps": 1e-12,
            },
            ValueError,
            "trust_radius",
        ),
        (
            {
                "current_dual_value": -0.1,
                "current_trust_radius": 1.0,
                "actual_fm_delta": 0.2,
                "fm_delta_target": 0.1,
                "dual_lr": 0.5,
                "trust_radius_min": 0.25,
                "trust_radius_max": 2.0,
            },
            ValueError,
            "current_dual_value",
        ),
    ],
)
def test_trust_and_dual_helpers_reject_invalid_inputs(kwargs, error, match: str) -> None:
    if "projected_gradients" in kwargs:
        with pytest.raises(error, match=match):
            apply_fm_anchor_trust_region_scaling(**kwargs)
    else:
        with pytest.raises(error, match=match):
            update_dual_budget_controller(**kwargs)
