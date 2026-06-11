from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LearnedNullCondition(nn.Module):
    """Trainable null condition vector expanded to a requested batch."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        embedding_dim = int(embedding_dim)
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, *, batch_size: int, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        embedding = self.embedding
        if device is not None or dtype is not None:
            embedding = embedding.to(
                device=device if device is not None else embedding.device,
                dtype=dtype if dtype is not None else embedding.dtype,
            )
        return embedding.unsqueeze(0).expand(batch_size, -1)


def fixed_null_condition_like(condition: torch.Tensor) -> torch.Tensor:
    if not isinstance(condition, torch.Tensor):
        raise TypeError(f"condition must be a torch.Tensor, got {type(condition).__name__}")
    return condition.new_zeros(condition.shape)


def learned_null_condition_like(provider: Any, condition: torch.Tensor) -> torch.Tensor:
    if not isinstance(condition, torch.Tensor):
        raise TypeError(f"condition must be a torch.Tensor, got {type(condition).__name__}")
    make_null_condition = getattr(provider, "make_null_condition", None)
    if not callable(make_null_condition):
        raise RuntimeError("learned null condition requires a generator with make_null_condition")
    null_condition = make_null_condition(batch_size=condition.shape[0], device=condition.device, dtype=condition.dtype)
    if not isinstance(null_condition, torch.Tensor):
        raise TypeError(f"make_null_condition must return a torch.Tensor, got {type(null_condition).__name__}")
    if tuple(null_condition.shape) != tuple(condition.shape):
        raise ValueError(f"null condition shape must match condition shape {tuple(condition.shape)}, got {tuple(null_condition.shape)}")
    if null_condition.device != condition.device:
        raise ValueError(f"null condition device must match condition device {condition.device}, got {null_condition.device}")
    if null_condition.dtype != condition.dtype:
        raise TypeError(f"null condition dtype must match condition dtype {condition.dtype}, got {null_condition.dtype}")
    return null_condition


def apply_condition_dropout(
    condition: torch.Tensor,
    null_condition: torch.Tensor,
    *,
    dropout_prob: float,
    training: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not isinstance(condition, torch.Tensor):
        raise TypeError(f"condition must be a torch.Tensor, got {type(condition).__name__}")
    if not isinstance(null_condition, torch.Tensor):
        raise TypeError(f"null_condition must be a torch.Tensor, got {type(null_condition).__name__}")
    if tuple(condition.shape) != tuple(null_condition.shape):
        raise ValueError(f"null_condition shape must match condition shape {tuple(condition.shape)}, got {tuple(null_condition.shape)}")
    if condition.device != null_condition.device:
        raise ValueError(f"null_condition device must match condition device {condition.device}, got {null_condition.device}")
    if condition.dtype != null_condition.dtype:
        raise TypeError(f"null_condition dtype must match condition dtype {condition.dtype}, got {null_condition.dtype}")

    dropout_prob = float(dropout_prob)
    if dropout_prob < 0.0 or dropout_prob > 1.0:
        raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
    if not training or dropout_prob == 0.0:
        return condition
    if dropout_prob == 1.0:
        return null_condition

    mask_shape = (condition.shape[0],) + (1,) * (condition.ndim - 1)
    mask = torch.rand(mask_shape, device=condition.device, generator=generator) < dropout_prob
    return torch.where(mask, null_condition, condition)
