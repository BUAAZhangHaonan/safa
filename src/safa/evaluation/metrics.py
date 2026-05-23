from __future__ import annotations

import math
from typing import Iterable


def face_count_rates(counts: Iterable[int]) -> dict[str, float]:
    count_list = []
    for count in counts:
        if isinstance(count, bool):
            raise ValueError(f"Face count must be an integer count, got bool {count!r}")
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
