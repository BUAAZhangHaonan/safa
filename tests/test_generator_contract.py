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

    def test_flow_generator_config_from_dict_requires_core_fields(self) -> None:
        from safa.models.generator import FlowGeneratorConfig

        required_fields = (
            "embedding_dim",
            "image_size",
            "base_channels",
            "channel_multipliers",
            "condition_dim",
            "sample_steps",
            "train_cycle_steps",
            "sampler",
        )

        for field in required_fields:
            with self.subTest(field=field):
                payload = self._small_config()
                payload.pop(field)

                with self.assertRaisesRegex(ValueError, field):
                    FlowGeneratorConfig.from_dict(payload)

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

    def test_generator_sample_signature_accepts_keyword_only_controls(self) -> None:
        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        signature = inspect.signature(generator.sample)

        self.assertEqual([name for name in signature.parameters], ["z", "steps", "checkpoint_steps", "x_init", "clamp_output"])
        self.assertEqual(signature.parameters["x_init"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters["x_init"].default, None)
        self.assertEqual(signature.parameters["clamp_output"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters["clamp_output"].default, True)

    def test_generator_sample_with_x_init_is_deterministic(self) -> None:
        import torch

        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        z = torch.randn(2, 128)
        x_init = torch.randn(2, 3, 64, 64)

        first = generator.sample(z, x_init=x_init)
        second = generator.sample(z, x_init=x_init)

        self.assertTrue(torch.equal(first, second))

    def test_generator_sample_uses_x_init(self) -> None:
        import torch
        from torch import nn

        from safa.models.generator import ConditionalFlowGenerator

        class ZeroVectorField(nn.Module):
            def forward(self, x_t, t, z):
                return torch.zeros_like(x_t)

        generator = ConditionalFlowGenerator(self._small_config())
        generator.vector_field = ZeroVectorField()
        z = torch.randn(2, 128)

        first = generator.sample(z, x_init=torch.full((2, 3, 64, 64), -0.5))
        second = generator.sample(z, x_init=torch.full((2, 3, 64, 64), 0.5))

        self.assertFalse(torch.equal(first, second))

    def test_generator_sample_rejects_invalid_x_init(self) -> None:
        import torch

        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        z = torch.randn(2, 128)

        with self.assertRaises(ValueError):
            generator.sample(z, x_init=torch.randn(2, 3, 32, 32))
        with self.assertRaises(TypeError):
            generator.sample(z, x_init=torch.randn(2, 3, 64, 64, dtype=torch.float64))
        with self.assertRaises(ValueError):
            generator.sample(z, x_init=torch.empty((2, 3, 64, 64), device="meta"))

    def test_generator_sample_clamps_output_by_default(self) -> None:
        import torch

        from safa.models.generator import ConditionalFlowGenerator

        generator = ConditionalFlowGenerator(self._small_config())
        output = generator.sample(torch.randn(2, 128))

        self.assertGreaterEqual(float(output.detach().min()), 0.0)
        self.assertLessEqual(float(output.detach().max()), 1.0)

    def test_generator_sample_can_return_unclamped_image_space_values(self) -> None:
        import torch
        from torch import nn

        from safa.models.generator import ConditionalFlowGenerator

        class ZeroVectorField(nn.Module):
            def forward(self, x_t, t, z):
                return torch.zeros_like(x_t)

        generator = ConditionalFlowGenerator(self._small_config())
        generator.vector_field = ZeroVectorField()
        z = torch.randn(2, 128)
        x_init = torch.full((2, 3, 64, 64), 3.0)

        output = generator.sample(z, x_init=x_init, clamp_output=False)

        self.assertGreater(float(output.detach().max()), 1.0)
        self.assertTrue(torch.equal(output, (x_init + 1.0) * 0.5))

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
