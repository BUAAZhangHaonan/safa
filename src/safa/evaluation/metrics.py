from __future__ import annotations

import math
from typing import Iterable


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

