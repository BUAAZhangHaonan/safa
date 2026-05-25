from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from safa.utils.config import load_yaml


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
TORCH_ONLY_AVAILABLE = importlib.util.find_spec("torch") is not None


class E0MediumConfigTests(unittest.TestCase):
    def test_medium_v1_train_config_explicitly_sets_required_plan_fields(self) -> None:
        config = load_yaml("configs/medium_v1/train_e0_medium_v1.yaml")

        expected = {
            "seed": 1337,
            "device": "cuda:0",
            "num_workers": 8,
            "image_size": 224,
            "num_classes": 8,
            "embedding_dim": 512,
            "train_index": "data/index/train_balanced_medium.jsonl",
            "val_index": "data/index/val_single_face.jsonl",
            "out_dir": "artifacts/checkpoints/e0_medium_v1",
            "epochs": 60,
            "batch_size": 64,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "warmup_epochs": 5,
            "early_stopping_patience": 15,
            "augmentation": "strong",
            "class_weight": False,
            "label_smoothing": 0.1,
            "imagenet_weights": "IMAGENET1K_V2",
        }

        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertIn(key, config)
                self.assertEqual(config[key], value)

    def test_e0_train_config_rejects_missing_medium_required_fields(self) -> None:
        from safa.training.e0_loop import require_e0_train_config

        config = {
            "seed": 1337,
            "device": "cuda:0",
            "num_workers": 8,
            "image_size": 224,
            "num_classes": 8,
            "embedding_dim": 512,
            "train_index": "data/index/train_balanced_medium.jsonl",
            "val_index": "data/index/val_single_face.jsonl",
            "out_dir": "artifacts/checkpoints/e0_medium_v1",
            "epochs": 60,
            "batch_size": 64,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "warmup_epochs": 5,
            "early_stopping_patience": 15,
            "augmentation": "strong",
            "class_weight": False,
            "label_smoothing": 0.1,
            "imagenet_weights": "IMAGENET1K_V2",
        }

        for missing in ("warmup_epochs", "early_stopping_patience", "augmentation", "class_weight", "label_smoothing"):
            incomplete = dict(config)
            incomplete.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(KeyError, missing):
                    require_e0_train_config(incomplete)


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
    def test_evaluate_e0_reports_macro_metrics_confusion_matrix_and_norm_payload(self) -> None:
        import torch
        import torch.nn.functional as F

        from safa.training.e0_loop import evaluate_e0

        class FakeE0(torch.nn.Module):
            num_classes = 4

            def forward(self, images):
                embeddings = F.normalize(
                    torch.tensor(
                        [
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                            [-1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    p=2,
                    dim=1,
                )
                logits = torch.tensor(
                    [
                        [4.0, 0.0, 0.0, 0.0],
                        [3.0, 1.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 6.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
                return {"embedding": embeddings, "logits": logits}

        loader = [
            {
                "image": torch.zeros(4, 3, 4, 4),
                "label": torch.tensor([0, 1, 1, 2]),
            }
        ]

        metrics = evaluate_e0(FakeE0(), loader, torch.device("cpu"), num_classes=4)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (1.0 + 0.5 + 1.0) / 3.0)
        self.assertAlmostEqual(metrics["macro_f1"], ((2.0 / 3.0) + (2.0 / 3.0) + 1.0) / 3.0)
        self.assertEqual(
            metrics["confusion_matrix"],
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
        )
        self.assertEqual(set(metrics["per_class_accuracy"]), {"class_0", "class_1", "class_2", "class_3"})
        self.assertEqual(metrics["per_class_accuracy"]["class_3"], None)
        self.assertEqual(metrics["per_class_support"]["class_3"], 0)
        self.assertEqual(metrics["per_class_accuracy_note"], "null means the class has zero validation samples")
        self.assertTrue(metrics["embedding_norm_check"]["passed"])
        self.assertLessEqual(metrics["embedding_norm_check"]["max_abs_deviation"], 1e-4)

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
            "warmup_epochs": 0,
            "early_stopping_patience": 0,
            "augmentation": "default",
            "class_weight": False,
            "label_smoothing": 0.0,
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
