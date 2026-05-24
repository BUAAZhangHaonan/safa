from __future__ import annotations

from collections.abc import Mapping


class ExponentialMovingAverage:
    def __init__(self, model, decay: float):
        import torch

        if isinstance(decay, bool) or not isinstance(decay, (float, int)):
            raise ValueError(f"EMA decay must be numeric in (0, 1), got {decay!r}")
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay!r}")
        self._state = {}
        for name, value in model.state_dict().items():
            if not torch.is_tensor(value):
                raise ValueError(f"EMA state entry {name} is not a tensor")
            self._state[name] = value.detach().clone()

    def update(self, model) -> None:
        import torch

        live_state = model.state_dict()
        self._assert_same_keys(live_state)
        one_minus_decay = 1.0 - self.decay
        with torch.no_grad():
            for name, live_value in live_state.items():
                ema_value = self._state[name]
                if ema_value.shape != live_value.shape:
                    raise ValueError(f"EMA state shape mismatch for {name}: ema={tuple(ema_value.shape)} live={tuple(live_value.shape)}")
                if torch.is_floating_point(ema_value):
                    ema_value.mul_(self.decay).add_(live_value.detach().to(device=ema_value.device, dtype=ema_value.dtype), alpha=one_minus_decay)
                else:
                    ema_value.copy_(live_value.detach().to(device=ema_value.device, dtype=ema_value.dtype))

    def copy_to(self, model) -> None:
        model.load_state_dict(self.state_dict())

    def state_dict(self) -> dict:
        return {name: value.detach().clone() for name, value in self._state.items()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._assert_same_keys(state_dict)
        for name, value in state_dict.items():
            if not hasattr(value, "detach"):
                raise ValueError(f"EMA state entry {name} is not a tensor")
            self._state[name] = value.detach().clone()

    def _assert_same_keys(self, state_dict: Mapping[str, object]) -> None:
        expected = set(self._state)
        actual = set(state_dict)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"EMA state keys mismatch: missing={missing} extra={extra}")
