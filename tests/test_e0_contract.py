from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch and torchvision are required for E0 contract tests")
class E0ContractTests(unittest.TestCase):
    def test_e0_embedding_contract_with_random_init_for_unit_test_only(self) -> None:
        import torch

        from safa.models.e0 import E0Config, build_e0

        model = build_e0(E0Config(imagenet_weights=""), allow_random_init=True)
        output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output["embedding"].shape), (2, 512))
        norms = output["embedding"].float().norm(dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_freeze_audit_rejects_optimizer_with_e0_parameters(self) -> None:
        import torch

        from safa.models.e0 import E0Config, assert_e0_frozen, build_e0, freeze_e0

        model = build_e0(E0Config(imagenet_weights=""), allow_random_init=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        freeze_e0(model)
        with self.assertRaises(RuntimeError):
            assert_e0_frozen(model, optimizer)
        assert_e0_frozen(model)


if __name__ == "__main__":
    unittest.main()

