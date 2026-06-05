from __future__ import annotations

from dataclasses import is_dataclass
import math

import pytest


torch = pytest.importorskip("torch")

from safa.training.projected_update import (
    FMAnchoredCAGradResult,
    ProjectionResult,
    aggregate_two_task_cagrad,
    aggregate_two_task_fm_anchored_cagrad,
    project_gradient_onto_fm_feasible_cone,
    project_gradient_to_dot_lower_bound,
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


def test_lower_bound_projection_uses_requested_fm_credit_boundary() -> None:
    g_repr = [torch.tensor([-1.0, 0.0])]
    g_fm = [torch.tensor([1.0, 0.0])]

    result = project_gradient_to_dot_lower_bound(g_repr, g_fm, lower_bound=-0.25, eps=1e-12)

    assert isinstance(result, ProjectionResult)
    assert result.projection_applied is True
    assert torch.allclose(result.dot_before, torch.tensor(-1.0))
    assert torch.allclose(result.dot_after, torch.tensor(-0.25), atol=1e-6)
    assert torch.allclose(result.projected_gradients[0], torch.tensor([-0.25, 0.0]))
    assert torch.allclose(_dot(result.projected_gradients, g_fm), torch.tensor(-0.25), atol=1e-6)
    assert torch.allclose(g_repr[0], torch.tensor([-1.0, 0.0]))


def test_lower_bound_projection_keeps_feasible_repr_gradient() -> None:
    g_repr = [torch.tensor([0.5, 0.0])]
    g_fm = [torch.tensor([1.0, 0.0])]

    result = project_gradient_to_dot_lower_bound(g_repr, g_fm, lower_bound=-0.25, eps=1e-12)

    assert result.projection_applied is False
    assert torch.allclose(result.dot_before, torch.tensor(0.5))
    assert torch.allclose(result.dot_after, torch.tensor(0.5))
    assert torch.allclose(result.projected_gradients[0], g_repr[0])


def test_tiny_fm_gradient_keeps_repr_gradient_unprojected_even_when_conflicting() -> None:
    g_repr = [torch.tensor([1.0, 0.0])]
    g_fm = [torch.tensor([-1e-8, 0.0])]

    result = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=1e-6)

    assert result.projection_applied is False
    assert result.fm_norm <= torch.tensor(1e-6)
    assert result.dot_before < 0
    assert torch.allclose(result.dot_after, result.dot_before)
    assert torch.allclose(result.projected_gradients[0], g_repr[0])


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
    ("lower_bound", "eps", "error", "match"),
    [
        (float("nan"), 1e-12, ValueError, "lower_bound"),
        (-0.25, -1.0, ValueError, "eps"),
        ("-0.25", 1e-12, TypeError, "lower_bound"),
    ],
)
def test_lower_bound_projection_rejects_invalid_controls(lower_bound, eps: float, error, match: str) -> None:
    with pytest.raises(error, match=match):
        project_gradient_to_dot_lower_bound([torch.ones(1)], [torch.ones(1)], lower_bound=lower_bound, eps=eps)


def test_cagrad_two_task_aggregation_matches_toy_reference_example() -> None:
    g_fm = [torch.tensor([1.0, 0.0])]
    g_repr = [torch.tensor([0.0, 1.0])]

    result = aggregate_two_task_cagrad(g_fm, g_repr, c=0.5, eps=1e-12)

    assert result.fm_weight == pytest.approx(0.5)
    assert result.cl_weight == pytest.approx(0.5)
    assert result.gradient_cosine == pytest.approx(0.0)
    assert torch.allclose(result.combined_gradients[0], torch.tensor([0.75, 0.75]), atol=1e-6)
    assert result.combined_norm == pytest.approx(float(torch.sqrt(torch.tensor(1.125))))


def test_fm_anchored_cagrad_raises_fm_weight_to_descent_floor() -> None:
    g_fm = [torch.tensor([6.0, 1.0, 1.0])]
    g_repr = [torch.tensor([-4.0, 1.0, 1.0])]

    raw = aggregate_two_task_cagrad(g_fm, g_repr, c=0.5, eps=1e-12)
    result = aggregate_two_task_fm_anchored_cagrad(
        g_fm,
        g_repr,
        c=0.5,
        fm_descent_floor_fraction=0.5,
        eps=1e-12,
    )

    assert isinstance(result, FMAnchoredCAGradResult)
    assert result.raw_fm_weight == pytest.approx(raw.fm_weight)
    assert result.raw_cl_weight == pytest.approx(raw.cl_weight)
    assert result.fm_descent_floor == pytest.approx(19.0)
    assert result.fm_descent_after_cagrad < result.fm_descent_floor
    assert result.fm_descent_after_anchor + 1e-6 >= result.fm_descent_floor
    assert result.fm_weight > raw.fm_weight
    assert result.cl_weight < raw.cl_weight
    assert result.anchor_active is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"c": None, "eps": 1e-12}, "c"),
        ({"c": 1.0, "eps": 1e-12}, "c"),
        ({"c": 0.5, "eps": 0.0}, "eps"),
    ],
)
def test_cagrad_rejects_invalid_controls(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        aggregate_two_task_cagrad([torch.ones(1)], [torch.ones(1)], **kwargs)
