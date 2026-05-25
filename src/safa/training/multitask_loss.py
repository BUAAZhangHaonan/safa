from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - training imports validate torch at runtime.
    torch = None

    class _MissingTorchModule:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for UncertaintyWeightedLoss")

    nn = type("_MissingNN", (), {"Module": _MissingTorchModule, "ParameterDict": dict, "Parameter": object})


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty weighting for the flow and cycle tasks.

    For each task loss L_i and trainable log variance s_i, the weighted
    component is:

        0.5 * exp(-s_i) * L_i + 0.5 * s_i

    This implementation intentionally only accepts the current G-training
    tasks: ``flow`` and ``cycle``.
    """

    _ALLOWED_TASKS = ("flow", "cycle")

    def __init__(self, task_names: list[str]):
        super().__init__()
        if torch is None:
            raise RuntimeError("torch is required for UncertaintyWeightedLoss")
        if not isinstance(task_names, list) or not task_names:
            raise ValueError("task_names must be a non-empty list")
        seen = set()
        for name in task_names:
            if name not in self._ALLOWED_TASKS:
                raise ValueError(f"Unsupported task {name!r}; allowed tasks are {self._ALLOWED_TASKS}")
            if name in seen:
                raise ValueError(f"task_names contains duplicate task {name!r}")
            seen.add(name)
        self.task_names = tuple(task_names)
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in self.task_names})

    def forward(self, losses: dict):
        if not isinstance(losses, dict):
            raise TypeError("losses must be a dict")
        expected = set(self.task_names)
        actual = set(losses)
        missing = sorted(expected - actual)
        if missing:
            raise KeyError(f"losses missing required task(s): {missing}")
        unexpected = sorted(actual - expected)
        if unexpected:
            raise KeyError(f"losses contains unexpected task(s): {unexpected}")

        total = None
        metrics = {}
        for name in self.task_names:
            loss = losses[name]
            if not hasattr(loss, "detach"):
                loss = torch.as_tensor(loss, dtype=torch.float32)
            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            weighted = 0.5 * precision * loss + 0.5 * log_var
            total = weighted if total is None else total + weighted
            metrics[f"loss_weighting_uw_{name}_log_var"] = float(log_var.detach().cpu())
            metrics[f"loss_weighting_uw_{name}_precision"] = float(precision.detach().cpu())
            metrics[f"loss_weighting_uw_{name}_normalized"] = float(loss.detach().cpu())
            metrics[f"loss_weighting_uw_{name}_weighted"] = float(weighted.detach().cpu())
        if total is None:
            raise RuntimeError("No task losses were provided")
        return total, metrics
