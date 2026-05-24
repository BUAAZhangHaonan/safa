from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for EMA tests")
class ExponentialMovingAverageTests(unittest.TestCase):
    def test_update_uses_decay_formula_after_optimizer_step(self) -> None:
        import torch

        from safa.utils.ema import ExponentialMovingAverage

        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 4.0]]))

        ema = ExponentialMovingAverage(model, decay=0.5)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[10.0, 20.0]]))

        ema.update(model)

        state = ema.state_dict()
        self.assertTrue(torch.equal(state["weight"], torch.tensor([[6.0, 12.0]])))

    def test_decay_must_be_between_zero_and_one(self) -> None:
        import torch

        from safa.utils.ema import ExponentialMovingAverage

        model = torch.nn.Linear(2, 1, bias=False)

        for decay in (0.0, 1.0, True):
            with self.subTest(decay=decay):
                with self.assertRaisesRegex(ValueError, "decay"):
                    ExponentialMovingAverage(model, decay=decay)
