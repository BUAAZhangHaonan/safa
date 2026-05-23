from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
TORCH_ONLY_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch and torchvision are required for E0 contract tests")
class E0ContractTests(unittest.TestCase):
    def test_e0_embedding_contract_uses_configured_embedding_dim(self) -> None:
        import torch

        from safa.models.e0 import E0Config, build_e0

        model = build_e0(E0Config(embedding_dim=128, imagenet_weights=""), allow_random_init=True)
        output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output["embedding"].shape), (2, 128))
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


@unittest.skipUnless(TORCH_ONLY_AVAILABLE, "torch is required for E0 training loop tests")
class E0TrainingLoopTests(unittest.TestCase):
    def test_train_e0_hard_fails_on_non_finite_loss(self) -> None:
        import torch

        from safa.training import e0_loop
        from safa.utils.distributed import DistributedContext

        class FakeE0(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, images):
                batch_size = images.shape[0]
                embedding = torch.ones(batch_size, 2, dtype=torch.float32) * self.weight
                embedding = torch.nn.functional.normalize(embedding, dim=1)
                logits = torch.zeros(batch_size, 2, dtype=torch.float32) + self.weight * 0.0
                return {"embedding": embedding, "logits": logits}

        class NonFiniteCriterion:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __call__(self, logits, labels):
                return logits.sum() * torch.tensor(float("nan"), dtype=logits.dtype)

        train_set = SimpleNamespace(records=[SimpleNamespace(label=0), SimpleNamespace(label=1)])
        val_set = SimpleNamespace(records=[SimpleNamespace(label=0), SimpleNamespace(label=1)])

        def fake_data_loader(dataset, **kwargs):
            if dataset is train_set:
                return [{"image": torch.zeros(2, 3, 4, 4), "label": torch.tensor([0, 1])}]
            return []

        out_dir = tempfile.TemporaryDirectory()
        self.addCleanup(out_dir.cleanup)
        config = {
            "seed": 1,
            "device": "cpu",
            "num_workers": 1,
            "batch_size": 2,
            "image_size": 4,
            "train_index": "train.jsonl",
            "val_index": "val.jsonl",
            "out_dir": out_dir.name,
            "num_classes": 2,
            "embedding_dim": 2,
            "imagenet_weights": "",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 1,
        }
        distributed = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            device=torch.device("cpu"),
            backend="single",
        )

        with patch.object(e0_loop, "set_seed"), patch.object(e0_loop, "init_distributed", return_value=distributed), patch.object(
            e0_loop, "barrier"
        ), patch.object(e0_loop, "cleanup_distributed"), patch.object(
            e0_loop, "train_transform", return_value=object()
        ), patch.object(
            e0_loop, "eval_transform", return_value=object()
        ), patch.object(
            e0_loop, "AffectNetRecords", side_effect=[train_set, val_set]
        ), patch.object(
            e0_loop, "build_e0", return_value=FakeE0()
        ), patch.object(
            e0_loop, "evaluate_e0", return_value={"accuracy": 0.5, "num_samples": 2, "mean_abs_logit": 0.0}
        ), patch(
            "torch.utils.data.DataLoader", side_effect=fake_data_loader
        ), patch(
            "torch.nn.CrossEntropyLoss", NonFiniteCriterion
        ), patch(
            "tqdm.tqdm", side_effect=lambda iterable, **kwargs: iterable
        ):
            with self.assertRaisesRegex(RuntimeError, "non-finite E0 loss"):
                e0_loop.train_e0_from_config(config)


if __name__ == "__main__":
    unittest.main()
