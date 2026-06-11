from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_learned_null_condition_returns_batch_tensor_matching_request() -> None:
    from safa.models.conditioning import LearnedNullCondition

    null_condition = LearnedNullCondition(embedding_dim=3).to(dtype=torch.float64)

    output = null_condition(batch_size=4, device=torch.device("cpu"), dtype=torch.float64)

    assert output.shape == (4, 3)
    assert output.device == torch.device("cpu")
    assert output.dtype == torch.float64
    assert output.requires_grad

    output.sum().backward()

    assert torch.equal(null_condition.embedding.grad, torch.full((3,), 4.0, dtype=torch.float64))


def test_apply_condition_dropout_with_probability_one_replaces_all_rows() -> None:
    from safa.models.conditioning import apply_condition_dropout

    condition = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    null_condition = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]])

    dropped = apply_condition_dropout(condition, null_condition, dropout_prob=1.0, training=True)

    assert torch.equal(dropped, null_condition)


def test_apply_condition_dropout_with_probability_zero_keeps_real_condition() -> None:
    from safa.models.conditioning import apply_condition_dropout

    condition = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    null_condition = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]])

    dropped = apply_condition_dropout(condition, null_condition, dropout_prob=0.0, training=True)

    assert torch.equal(dropped, condition)


def test_apply_condition_dropout_is_disabled_outside_training() -> None:
    from safa.models.conditioning import apply_condition_dropout

    condition = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    null_condition = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]])

    dropped = apply_condition_dropout(condition, null_condition, dropout_prob=1.0, training=False)

    assert torch.equal(dropped, condition)
