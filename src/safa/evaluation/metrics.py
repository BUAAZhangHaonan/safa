from __future__ import annotations

import math
from typing import Iterable


_UNIT_NORM_ATOL = 1e-4
_UNIT_NORM_RTOL = 1e-4
DEFAULT_DENSE_GRAM_MAX_SAMPLES = 2048


def validate_dense_gram_cap(value, *, context: str = "dense Gram cap") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    return int(value)


def validate_dense_gram_sample_limit(
    max_samples,
    *,
    dense_gram_cap: int = DEFAULT_DENSE_GRAM_MAX_SAMPLES,
    context: str = "validation relation metrics",
) -> int:
    cap = validate_dense_gram_cap(dense_gram_cap, context=f"{context} dense Gram cap")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
        raise ValueError(f"{context} max_samples must be a positive integer before dense Gram computation, got {max_samples!r}")
    sample_limit = int(max_samples)
    if sample_limit > cap:
        raise ValueError(
            f"{context} max_samples={sample_limit} exceeds dense Gram cap {cap}; "
            "pass an explicit higher dense Gram cap to run this many samples"
        )
    return sample_limit


def validate_dense_gram_sample_count(
    sample_count,
    *,
    dense_gram_cap: int = DEFAULT_DENSE_GRAM_MAX_SAMPLES,
    context: str = "validation relation metrics",
) -> int:
    cap = validate_dense_gram_cap(dense_gram_cap, context=f"{context} dense Gram cap")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError(f"{context} sample_count must be a non-negative integer, got {sample_count!r}")
    count = int(sample_count)
    if count > cap:
        raise ValueError(
            f"{context} sample_count={count} exceeds dense Gram cap {cap}; "
            "reduce max_samples or pass an explicit higher dense Gram cap"
        )
    return count


def face_count_rates(counts: Iterable[int]) -> dict[str, float]:
    import numpy as np

    count_list = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise ValueError(f"Face count must be an integer count, got {type(count).__name__} {count!r}")
        parsed = int(count)
        if parsed < 0:
            raise ValueError(f"Face count must be non-negative, got {count!r}")
        count_list.append(parsed)
    if not count_list:
        raise ValueError("Cannot compute face count rates from an empty count list")
    total = float(len(count_list))
    return {
        "face_detect_ge1_rate": sum(1 for count in count_list if count >= 1) / total,
        "single_face_eq1_rate": sum(1 for count in count_list if count == 1) / total,
        "zero_face_rate": sum(1 for count in count_list if count == 0) / total,
        "multi_face_rate": sum(1 for count in count_list if count > 1) / total,
    }


def summarize(values: Iterable[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric list")
    if not np.isfinite(array).all():
        raise ValueError("Metric list contains non-finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def flatten_finite_numbers(payload) -> list[float]:
    values: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            values.extend(flatten_finite_numbers(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(flatten_finite_numbers(value))
    elif isinstance(payload, (int, float)):
        value = float(payload)
        if not math.isfinite(value):
            raise ValueError("Non-finite number in metrics payload")
        values.append(value)
    return values


def compute_validation_relation_metrics(pred_embedding, target_embedding) -> dict[str, float]:
    """Compute validation representation relation metrics for normalized embeddings."""
    import torch

    _validate_relation_embedding_pair(pred_embedding, target_embedding)
    pred = pred_embedding.detach().to(dtype=torch.float64)
    target = target_embedding.detach().to(dtype=torch.float64)

    point_loss = (1.0 - (pred * target).sum(dim=1)).mean()
    pred_gram = pred @ pred.T
    target_gram = target @ target.T
    pred_pairs = _offdiag_values(pred_gram)
    target_pairs = _offdiag_values(target_gram)
    gram_error = pred_pairs - target_pairs
    offdiag_mse = gram_error.pow(2).mean()
    offdiag_mae = gram_error.abs().mean()

    return {
        "repr_point_loss": _to_float(point_loss),
        "repr_relation_loss": _to_float(offdiag_mse),
        "offdiag_gram_mse": _to_float(offdiag_mse),
        "offdiag_gram_mae": _to_float(offdiag_mae),
        "pairwise_pearson": _to_float(_pearson("pairwise_pearson", pred_pairs, target_pairs)),
        "pairwise_spearman": _to_float(
            _pearson("pairwise_spearman", _average_ranks(pred_pairs), _average_ranks(target_pairs))
        ),
    }


def _validate_relation_embedding_pair(pred_embedding, target_embedding) -> None:
    import torch

    if not isinstance(pred_embedding, torch.Tensor):
        raise TypeError(f"pred_embedding must be a torch.Tensor, got {type(pred_embedding).__name__}")
    if not isinstance(target_embedding, torch.Tensor):
        raise TypeError(f"target_embedding must be a torch.Tensor, got {type(target_embedding).__name__}")
    if pred_embedding.ndim != 2 or target_embedding.ndim != 2:
        raise ValueError("pred_embedding and target_embedding must be 2D tensors")
    if pred_embedding.shape != target_embedding.shape:
        raise ValueError("pred_embedding and target_embedding must have the same shape")
    if pred_embedding.shape[0] <= 1:
        raise ValueError("Batch dimension B > 1 is required for validation relation metrics")
    if not torch.isfinite(pred_embedding).all() or not torch.isfinite(target_embedding).all():
        raise FloatingPointError("pred_embedding and target_embedding must be finite")
    _validate_unit_norm_rows("pred_embedding", pred_embedding)
    _validate_unit_norm_rows("target_embedding", target_embedding)


def _validate_unit_norm_rows(name: str, tensor) -> None:
    import torch

    norms = tensor.detach().to(dtype=torch.float64).norm(dim=1)
    expected = torch.ones_like(norms)
    if not torch.allclose(norms, expected, rtol=_UNIT_NORM_RTOL, atol=_UNIT_NORM_ATOL):
        raise ValueError(f"{name} rows must be unit-norm within tolerance")


def _offdiag_values(matrix):
    import torch

    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _pearson(name: str, x, y):
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    x_norm = centered_x.norm()
    y_norm = centered_y.norm()
    if float(x_norm.detach().cpu()) == 0.0:
        raise ValueError(f"{name} is undefined because pred pairwise similarities have zero variance")
    if float(y_norm.detach().cpu()) == 0.0:
        raise ValueError(f"{name} is undefined because target pairwise similarities have zero variance")
    return (centered_x * centered_y).sum() / (x_norm * y_norm)


def _average_ranks(values):
    import torch

    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    sorted_values = values[order]
    start = 0
    count = int(values.numel())
    while start < count:
        end = start + 1
        while end < count and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _to_float(value) -> float:
    numeric = float(value.detach().cpu())
    if not math.isfinite(numeric):
        raise FloatingPointError("validation relation metric is non-finite")
    return numeric
