from __future__ import annotations

import importlib.util
import inspect
import tempfile
from pathlib import Path
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for generator contract tests")
class GeneratorContractTests(unittest.TestCase):
    def _small_config(self) -> dict:
        return {
            "embedding_dim": 128,
            "image_size": 64,
            "base_channels": 4,
            "channel_multipliers": [1],
            "time_embedding_dim": 16,
            "condition_dim": 32,
            "sample_steps": 1,
            "train_cycle_steps": 1,
            "sampler": "euler",
        }

    def test_generator_forward_accepts_only_z(self) -> None:
        import torch

        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        parameters = [name for name in inspect.signature(generator.forward).parameters]
        self.assertEqual(parameters, ["z"])
        output = generator(torch.randn(2, 128))
        self.assertEqual(tuple(output.shape), (2, 3, 64, 64))
        self.assertTrue(torch.isfinite(output).all())
        detached = output.detach()
        self.assertGreaterEqual(float(detached.min()), 0.0)
        self.assertLessEqual(float(detached.max()), 1.0)
        with self.assertRaises(TypeError):
            generator(torch.randn(2, 128), image=torch.randn(2, 3, 64, 64))

    def test_generator_rejects_wrong_z_shape(self) -> None:
        import torch

        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        with self.assertRaises(ValueError):
            generator(torch.randn(2, 512))

    def test_flow_matching_loss_contract(self) -> None:
        import torch

        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        loss, metrics = generator.flow_matching_loss(torch.rand(2, 3, 64, 64), torch.randn(2, 128))
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("flow_matching_mse", metrics)
        self.assertTrue(torch.isfinite(metrics["flow_matching_mse"]))

    def test_checkpoint_model_config_reconstructs_generator(self) -> None:
        import torch

        from safa.evaluation.runner import _load_generator
        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.pt"
            torch.save(
                {
                    "model_state_dict": generator.state_dict(),
                    "model_config": generator.config.to_dict(),
                    "metrics": {},
                },
                path,
            )
            loaded = _load_generator(str(path), {"image_size": 64}, "cpu")
        output = loaded(torch.randn(1, 128))
        self.assertEqual(tuple(output.shape), (1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
