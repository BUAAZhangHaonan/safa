from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from safa.training.representation_losses import hyperspherical_gram_loss


def _manual_relation_loss(pred_embedding, target_embedding, offdiag_only: bool):
    pred_gram = pred_embedding @ pred_embedding.T
    target_gram = target_embedding @ target_embedding.T
    squared_error = (pred_gram - target_gram).pow(2)
    if not offdiag_only:
        return squared_error.mean()
    mask = ~torch.eye(pred_embedding.shape[0], dtype=torch.bool, device=pred_embedding.device)
    return squared_error[mask].mean()


def test_hyperspherical_gram_loss_returns_weighted_point_and_offdiag_relation_terms() -> None:
    pred = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [math.sqrt(0.5), math.sqrt(0.5)],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [math.sqrt(0.5), math.sqrt(0.5)],
        ],
        dtype=torch.float64,
    )

    losses = hyperspherical_gram_loss(pred, target, point_weight=2.0, relation_weight=3.0)

    expected_point = (1.0 - (pred * target).sum(dim=1)).mean()
    expected_relation = _manual_relation_loss(pred, target, offdiag_only=True)
    expected_total = 2.0 * expected_point + 3.0 * expected_relation
    assert losses.keys() >= {"repr", "point", "relation"}
    assert torch.allclose(losses["point"], expected_point)
    assert torch.allclose(losses["relation"], expected_relation)
    assert torch.allclose(losses["repr"], expected_total)
    assert torch.allclose(losses["point_loss"], losses["point"])
    assert torch.allclose(losses["relation_loss"], losses["relation"])
    assert torch.allclose(losses["total_loss"], losses["repr"])


def test_hyperspherical_gram_loss_can_include_diagonal_in_relation_loss() -> None:
    pred = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [math.sqrt(0.5), math.sqrt(0.5)]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[1.0, 0.0], [math.sqrt(0.5), math.sqrt(0.5)], [0.0, 1.0]],
        dtype=torch.float64,
    )

    offdiag = hyperspherical_gram_loss(pred, target, 1.0, 1.0, offdiag_only=True)
    full = hyperspherical_gram_loss(pred, target, 1.0, 1.0, offdiag_only=False)

    assert torch.allclose(offdiag["relation"], _manual_relation_loss(pred, target, True))
    assert torch.allclose(full["relation"], _manual_relation_loss(pred, target, False))
    assert full["relation"] < offdiag["relation"]


@pytest.mark.parametrize(
    ("pred", "target", "match"),
    [
        (torch.ones(1, 2), torch.ones(1, 2), "B > 1"),
        (torch.ones(2), torch.ones(2), "2D"),
        (torch.ones(2, 3), torch.ones(2, 2), "same shape"),
        (torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]), torch.eye(2), "finite"),
        (torch.tensor([[2.0, 0.0], [0.0, 1.0]]), torch.eye(2), "unit-norm"),
    ],
)
def test_hyperspherical_gram_loss_rejects_invalid_inputs(pred, target, match: str) -> None:
    with pytest.raises((ValueError, FloatingPointError), match=match):
        hyperspherical_gram_loss(pred, target, point_weight=1.0, relation_weight=1.0)
