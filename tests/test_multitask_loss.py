from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for multitask loss tests")
class UncertaintyWeightedLossTests(unittest.TestCase):
    def test_uncertainty_weighted_loss_uses_homoscedastic_formula(self) -> None:
        import torch

        from safa.training.multitask_loss import UncertaintyWeightedLoss

        loss_module = UncertaintyWeightedLoss(["flow", "cycle"])
        total, metrics = loss_module({"flow": torch.tensor(2.0), "cycle": torch.tensor(8.0)})

        self.assertTrue(torch.allclose(total, torch.tensor(5.0)))
        self.assertAlmostEqual(metrics["loss_weighting_uw_flow_log_var"], 0.0)
        self.assertAlmostEqual(metrics["loss_weighting_uw_cycle_log_var"], 0.0)
        self.assertAlmostEqual(metrics["loss_weighting_uw_flow_weighted"], 1.0)
        self.assertAlmostEqual(metrics["loss_weighting_uw_cycle_weighted"], 4.0)

    def test_uncertainty_weighted_loss_rejects_missing_extra_and_unknown_tasks(self) -> None:
        import torch

        from safa.training.multitask_loss import UncertaintyWeightedLoss

        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            UncertaintyWeightedLoss(["flow", "identity"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            UncertaintyWeightedLoss(["flow", "flow"])

        loss_module = UncertaintyWeightedLoss(["flow", "cycle"])
        with self.assertRaisesRegex(KeyError, "missing"):
            loss_module({"flow": torch.tensor(1.0)})
        with self.assertRaisesRegex(KeyError, "unexpected"):
            loss_module({"flow": torch.tensor(1.0), "cycle": torch.tensor(1.0), "extra": torch.tensor(1.0)})


if __name__ == "__main__":
    unittest.main()
