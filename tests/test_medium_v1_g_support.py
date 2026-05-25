from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class MediumV1GSupportTests(unittest.TestCase):
    def _base_stage2_metrics(self) -> dict:
        return {
            "stage": "stage2",
            "loss": 1.0,
            "validation_raw_latent_cosine_mean": 0.90,
            "validation_raw_single_face_eq1_rate": 0.80,
            "validation_ema_latent_cosine_mean": 0.90,
            "validation_ema_single_face_eq1_rate": 0.90,
        }

    def test_stage2_checkpoint_filenames_cover_utility_and_real_quality_metrics(self) -> None:
        from safa.training.g_loop import _stage2_checkpoint_filenames_to_save

        metrics = self._base_stage2_metrics()
        metrics.update(
            {
                "quality_raw_fid": 12.0,
                "quality_raw_kid_mean": 0.02,
                "quality_raw_niqe": 4.5,
            }
        )
        previous = [
            {
                "stage": "stage2",
                "loss": 1.2,
                "validation_raw_latent_cosine_mean": 0.80,
                "validation_raw_single_face_eq1_rate": 0.70,
                "validation_ema_latent_cosine_mean": 0.80,
                "validation_ema_single_face_eq1_rate": 0.80,
                "quality_raw_fid": 20.0,
                "quality_raw_kid_mean": 0.03,
                "quality_raw_niqe": 5.0,
            }
        ]

        names = _stage2_checkpoint_filenames_to_save(metrics, previous)

        self.assertIn("best_raw_utility.pt", names)
        self.assertIn("best_ema_utility.pt", names)
        self.assertIn("best_raw_quality.pt", names)
        self.assertNotIn("best_ema_quality.pt", names)

    def test_quality_checkpoint_is_not_selected_without_complete_quality_metrics(self) -> None:
        from safa.training.g_loop import _stage2_checkpoint_filenames_to_save

        metrics = self._base_stage2_metrics()
        metrics["quality_raw_fid"] = 12.0

        names = _stage2_checkpoint_filenames_to_save(metrics, [])

        self.assertIn("best_raw_utility.pt", names)
        self.assertNotIn("best_raw_quality.pt", names)

    def test_uncertainty_calibration_rejects_non_positive_or_non_finite_scales(self) -> None:
        from safa.training.g_loop import _finalize_uncertainty_calibration

        with self.assertRaisesRegex(RuntimeError, "flow_loss_initial"):
            _finalize_uncertainty_calibration(flow_sum=0.0, cycle_sum=1.0, batches=1)
        with self.assertRaisesRegex(RuntimeError, "cycle_loss_initial"):
            _finalize_uncertainty_calibration(flow_sum=1.0, cycle_sum=float("inf"), batches=1)
        with self.assertRaisesRegex(RuntimeError, "calibration_batches"):
            _finalize_uncertainty_calibration(flow_sum=1.0, cycle_sum=1.0, batches=0)

    def test_quality_eval_hook_can_be_monkeypatched_and_writes_epoch_metrics(self) -> None:
        from safa.training import g_loop

        config = {
            "stages": {
                "stage1": {
                    "quality_eval": {
                        "enabled": True,
                        "interval": 2,
                        "real_index": "data/index/val_single_face.jsonl",
                        "generated_dir": "artifacts/eval/medium_v1/raw/generated",
                        "output_dir": "artifacts/eval/medium_v1/quality",
                        "model": "raw",
                    }
                }
            }
        }

        calls: list[dict] = []

        def fake_quality_eval(**kwargs):
            calls.append(kwargs)
            return {
                "fid": 11.0,
                "kid_mean": 0.01,
                "kid_std": 0.002,
                "iqa": {"method": "niqe", "mean": 4.25, "std": 0.5},
            }

        with patch.object(g_loop, "_evaluate_generation_quality", side_effect=fake_quality_eval):
            metrics = g_loop._run_quality_eval_hook(config, "stage1", 1)

        self.assertEqual(len(calls), 1)
        self.assertEqual(metrics["quality_raw_fid"], 11.0)
        self.assertEqual(metrics["quality_raw_kid_mean"], 0.01)
        self.assertEqual(metrics["quality_raw_niqe"], 4.25)
        self.assertTrue(str(calls[0]["output"]).endswith("stage1_epoch_0002_raw.json"))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for loss weighting integration tests")
    def test_generator_training_step_fixed_loss_weighting_uses_config_weights(self) -> None:
        import torch
        from torch import nn

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _GeneratorTrainingStep, _loss_weighting_runtime_from_config

        class DummyGenerator(nn.Module):
            def flow_matching_loss(self, images, z):
                loss = images.sum() * 0.0 + z.sum() * 0.0 + 2.0
                return loss, {"flow_matching_mse": loss.detach()}

            def sample(self, z, **kwargs):
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        class DummyE0(nn.Module):
            def forward(self, images):
                return {
                    "embedding": torch.tensor([[0.0, 1.0]], device=images.device).repeat(images.shape[0], 1),
                    "logits": torch.zeros(images.shape[0], 2, device=images.device),
                }

        runtime = _loss_weighting_runtime_from_config({"loss_weighting": {"type": "fixed", "flow_weight": 2.0, "cycle_weight": 0.01}})
        module = _GeneratorTrainingStep(DummyGenerator(), DummyE0(), FlowGeneratorConfig(embedding_dim=2, image_size=4), 1337, runtime)

        loss, _, cycle, _, _ = module(torch.zeros(1, 3, 4, 4), torch.tensor([[1.0, 0.0]]), ["sample"], True, 99.0)

        self.assertAlmostEqual(float(cycle), 1.0, places=6)
        self.assertAlmostEqual(float(loss.detach()), 4.01, places=6)
        self.assertEqual(module.last_loss_metrics["loss_weighting_type"], "fixed")

    def test_plot_script_missing_required_input_fails_fast(self) -> None:
        from scripts.plot_medium_v1_curves import load_medium_v1_history

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_medium_v1_history(Path(tmp) / "missing.json")

    def test_medium_v1_g_configs_use_e0_medium_cache_and_checkpoint(self) -> None:
        expected_train_features = "artifacts/e0_features/train_balanced_medium_e0_medium_v1"
        expected_validation_features = "artifacts/e0_features/val_single_face_e0_medium_v1"
        expected_e0_checkpoint = "artifacts/checkpoints/e0_medium_v1/best.pt"
        config_paths = (
            Path("configs/medium_v1/train_g_medium_v1_stage1.yaml"),
            Path("configs/medium_v1/train_g_medium_v1_stage2_m0.yaml"),
            Path("configs/medium_v1/train_g_medium_v1_stage2_m1_uw.yaml"),
        )

        for path in config_paths:
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file())
                config = yaml.safe_load(path.read_text(encoding="utf-8"))

                self.assertEqual(config["train_features"], expected_train_features)
                self.assertEqual(config["validation"]["features"], expected_validation_features)
                self.assertEqual(config["e0_checkpoint"], expected_e0_checkpoint)

    def test_medium_v1_stage2_m0_and_m1_configs_only_differ_in_loss_weighting_and_out_dir(self) -> None:
        from safa.training import g_loop

        m0_path = Path("configs/medium_v1/train_g_medium_v1_stage2_m0.yaml")
        m1_path = Path("configs/medium_v1/train_g_medium_v1_stage2_m1_uw.yaml")
        stage1_path = Path("configs/medium_v1/train_g_medium_v1_stage1.yaml")
        self.assertTrue(stage1_path.is_file())
        self.assertTrue(m0_path.is_file())
        self.assertTrue(m1_path.is_file())

        stage1 = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))
        m0 = yaml.safe_load(m0_path.read_text(encoding="utf-8"))
        m1 = yaml.safe_load(m1_path.read_text(encoding="utf-8"))

        self.assertEqual(stage1["out_dir"], "artifacts/checkpoints/g_medium_v1_stage1_m0")
        self.assertEqual(m0["resume_from"], "artifacts/checkpoints/g_medium_v1_stage1_m0/best_single_face.pt")
        self.assertEqual(m1["resume_from"], "artifacts/checkpoints/g_medium_v1_stage1_m0/best_single_face.pt")
        self.assertEqual(m0["loss_weighting"], {"type": "fixed", "flow_weight": 1.0, "cycle_weight": 0.01})
        self.assertEqual(m1["loss_weighting"]["type"], "uncertainty")

        comparable_m0 = copy.deepcopy(m0)
        comparable_m1 = copy.deepcopy(m1)
        comparable_m0.pop("out_dir")
        comparable_m1.pop("out_dir")
        comparable_m0.pop("loss_weighting")
        comparable_m1.pop("loss_weighting")
        self.assertEqual(comparable_m0, comparable_m1)

        g_loop._validate_train_g_config(stage1)
        g_loop._validate_train_g_config(m0)
        g_loop._validate_train_g_config(m1)


if __name__ == "__main__":
    unittest.main()
