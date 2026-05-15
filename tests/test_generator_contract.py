from __future__ import annotations

import importlib.util
import inspect
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for generator contract tests")
class GeneratorContractTests(unittest.TestCase):
    def test_generator_forward_accepts_only_z(self) -> None:
        import torch

        from safa.models.generator import ZOnlyGenerator

        generator = ZOnlyGenerator()
        parameters = [name for name in inspect.signature(generator.forward).parameters]
        self.assertEqual(parameters, ["z"])
        output = generator(torch.randn(2, 512))
        self.assertEqual(tuple(output.shape), (2, 3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(TypeError):
            generator(torch.randn(2, 512), image=torch.randn(2, 3, 224, 224))

    def test_generator_rejects_wrong_z_shape(self) -> None:
        import torch

        from safa.models.generator import ZOnlyGenerator

        generator = ZOnlyGenerator()
        with self.assertRaises(ValueError):
            generator(torch.randn(2, 128))


if __name__ == "__main__":
    unittest.main()

