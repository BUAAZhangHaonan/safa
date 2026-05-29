from __future__ import annotations

from dataclasses import is_dataclass
import math

import pytest


torch = pytest.importorskip("torch")

from safa.training.projected_update import ProjectionResult, project_gradient_onto_fm_feasible_cone


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
