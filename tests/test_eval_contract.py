from __future__ import annotations

import inspect
import importlib.util
import json
import math
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from safa.evaluation.metrics import face_count_rates, flatten_finite_numbers, summarize
from safa.evaluation import perturbations
from safa.evaluation.runner import (
    _attach_face_detection_rows,
    _guard_result,
    _run_privacy_pass,
    _summarize_rows,
    deterministic_impostor_indices,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class EvalContractTests(unittest.TestCase):
    def test_summarize_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            summarize([1.0, math.nan])

    def test_flatten_rejects_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            flatten_finite_numbers({"x": [float("inf")]})

    def test_anti_steg_functions_do_not_accept_metadata(self) -> None:
        forbidden = {"path", "label", "sample_id", "filename", "metadata"}
        for name in [
            "apply_jpeg",
            "apply_blur",
            "apply_downsample",
            "apply_center_crop_resize",
            "apply_noise",
        ]:
            params = set(inspect.signature(getattr(perturbations, name)).parameters)
            self.assertFalse(forbidden.intersection(params), name)

    def test_impostor_indices_are_dataset_level_derangement(self) -> None:
        indices = deterministic_impostor_indices(5)
        self.assertEqual(indices, [2, 3, 4, 0, 1])
        self.assertTrue(all(index != impostor for index, impostor in enumerate(indices)))
        with self.assertRaises(ValueError):
            deterministic_impostor_indices(1)

    def test_face_detection_guard_requires_both_thresholds(self) -> None:
        metrics = {
            "face_detection": {
                "detected": {"mean": 1.0},
                "face_detect_ge1_rate": {"mean": 1.0},
                "single_face_eq1_rate": {"mean": 0.97},
                "zero_face_rate": {"mean": 0.0},
                "multi_face_rate": {"mean": 0.03},
            },
            "latent_cosine": {"mean": 0.94},
        }
        guard = _guard_result(
            metrics,
            {
                "enabled": True,
                "model_name": "buffalo_l",
                "threshold": 0.95,
                "single_face_eq1_threshold": 0.98,
                "latent_cosine_threshold": 0.95,
            },
        )
        self.assertFalse(guard["passed"])
        self.assertEqual(guard["face_detection_rate"], metrics["face_detection"]["detected"]["mean"])
        self.assertEqual(guard["face_detect_ge1_rate"], metrics["face_detection"]["face_detect_ge1_rate"]["mean"])
        self.assertEqual(guard["single_face_eq1_rate"], metrics["face_detection"]["single_face_eq1_rate"]["mean"])
        self.assertEqual(guard["zero_face_rate"], metrics["face_detection"]["zero_face_rate"]["mean"])
        self.assertEqual(guard["multi_face_rate"], metrics["face_detection"]["multi_face_rate"]["mean"])
        self.assertEqual(guard["single_face_eq1_threshold"], 0.98)
        metrics["latent_cosine"]["mean"] = 0.96
        guard = _guard_result(
            metrics,
            {
                "enabled": True,
                "model_name": "buffalo_l",
                "threshold": 0.95,
                "single_face_eq1_threshold": 0.98,
                "latent_cosine_threshold": 0.95,
            },
        )
        self.assertFalse(guard["passed"])
        metrics["face_detection"]["single_face_eq1_rate"]["mean"] = 0.99
        guard = _guard_result(
            metrics,
            {
                "enabled": True,
                "model_name": "buffalo_l",
                "threshold": 0.95,
                "single_face_eq1_threshold": 0.98,
                "latent_cosine_threshold": 0.95,
            },
        )
        self.assertTrue(guard["passed"])

    def test_face_detection_guard_rejects_missing_detection_metrics(self) -> None:
        with self.assertRaises(RuntimeError):
            _guard_result(
                {"latent_cosine": {"mean": 0.99}},
                {
                    "enabled": True,
                    "model_name": "buffalo_l",
                    "threshold": 0.95,
                    "single_face_eq1_threshold": 0.98,
                    "latent_cosine_threshold": 0.95,
                },
            )

    def test_face_detection_guard_requires_explicit_threshold_fields(self) -> None:
        metrics = {
            "face_detection": {
                "detected": {"mean": 1.0},
                "face_detect_ge1_rate": {"mean": 1.0},
                "single_face_eq1_rate": {"mean": 1.0},
                "zero_face_rate": {"mean": 0.0},
                "multi_face_rate": {"mean": 0.0},
            },
            "latent_cosine": {"mean": 1.0},
        }
        with self.assertRaisesRegex(ValueError, "threshold"):
            _guard_result(metrics, {"enabled": True, "model_name": "buffalo_l", "latent_cosine_threshold": 0.95})
        with self.assertRaisesRegex(ValueError, "latent_cosine_threshold"):
            _guard_result(
                metrics,
                {
                    "enabled": True,
                    "model_name": "buffalo_l",
                    "threshold": 0.95,
                    "single_face_eq1_threshold": 0.98,
                },
            )
        with self.assertRaisesRegex(ValueError, "single_face_eq1_threshold"):
            _guard_result(
                metrics,
                {
                    "enabled": True,
                    "model_name": "buffalo_l",
                    "threshold": 0.95,
                    "latent_cosine_threshold": 0.95,
                },
            )

    def test_face_detection_guard_requires_explicit_enabled_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "face_detection.enabled"):
            _guard_result({}, {})

    def test_face_detection_guard_requires_rate_summary_fields(self) -> None:
        metrics = {
            "face_detection": {"detected": {"mean": 1.0}},
            "latent_cosine": {"mean": 1.0},
        }

        with self.assertRaisesRegex(RuntimeError, "face_detection.face_detect_ge1_rate.mean"):
            _guard_result(
                metrics,
                {
                    "enabled": True,
                    "model_name": "buffalo_l",
                    "threshold": 0.95,
                    "single_face_eq1_threshold": 0.98,
                    "latent_cosine_threshold": 0.95,
                },
            )

    def test_privacy_enabled_requires_single_face_guard_threshold(self) -> None:
        from safa.evaluation import runner

        config = {
            "privacy": {
                "enabled": True,
                "recognizers": [{"name": "arcface", "type": "insightface", "model_name": "buffalo_l"}],
            },
            "face_detection": {
                "enabled": True,
                "model_name": "buffalo_l",
                "threshold": 0.95,
                "latent_cosine_threshold": 0.95,
            },
            "anti_steg": {"enabled": False},
        }

        with self.assertRaisesRegex(ValueError, "face_detection.single_face_eq1_threshold"):
            runner._eval_monitor_configs(config)

    def test_privacy_guard_uses_single_face_rate_not_legacy_ge1_rate(self) -> None:
        metrics = {
            "face_detection": {
                "detected": {"mean": 1.0},
                "face_detect_ge1_rate": {"mean": 1.0},
                "single_face_eq1_rate": {"mean": 0.5},
                "zero_face_rate": {"mean": 0.0},
                "multi_face_rate": {"mean": 0.5},
            },
            "latent_cosine": {"mean": 0.99},
        }

        guard = _guard_result(
            metrics,
            {
                "enabled": True,
                "model_name": "buffalo_l",
                "threshold": 0.95,
                "single_face_eq1_threshold": 0.98,
                "latent_cosine_threshold": 0.95,
            },
        )

        self.assertFalse(guard["passed"])

    def test_eval_monitor_config_requires_explicit_blocks(self) -> None:
        from safa.evaluation import runner

        base = {
            "privacy": {"enabled": False},
            "face_detection": {"enabled": False},
            "anti_steg": {"enabled": False},
        }
        privacy_cfg, face_detection_cfg, anti_cfg = runner._eval_monitor_configs(base)
        self.assertFalse(privacy_cfg["enabled"])
        self.assertFalse(face_detection_cfg["enabled"])
        self.assertFalse(anti_cfg["enabled"])

        for missing in ("privacy", "face_detection", "anti_steg"):
            config = {key: dict(value) for key, value in base.items()}
            config.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, missing):
                    runner._eval_monitor_configs(config)

    def test_eval_monitor_config_requires_enabled_flags_and_enabled_fields(self) -> None:
        from safa.evaluation import runner

        cases = [
            (
                "privacy.enabled",
                {
                    "privacy": {},
                    "face_detection": {"enabled": False},
                    "anti_steg": {"enabled": False},
                },
            ),
            (
                "privacy.recognizers",
                {
                    "privacy": {"enabled": True},
                    "face_detection": {"enabled": False},
                    "anti_steg": {"enabled": False},
                },
            ),
            (
                "face_detection.model_name",
                {
                    "privacy": {"enabled": False},
                    "face_detection": {"enabled": True, "threshold": 0.95, "latent_cosine_threshold": 0.95},
                    "anti_steg": {"enabled": False},
                },
            ),
            (
                "face_detection.threshold",
                {
                    "privacy": {"enabled": False},
                    "face_detection": {"enabled": True, "model_name": "buffalo_l", "latent_cosine_threshold": 0.95},
                    "anti_steg": {"enabled": False},
                },
            ),
            (
                "face_detection.latent_cosine_threshold",
                {
                    "privacy": {"enabled": False},
                    "face_detection": {"enabled": True, "model_name": "buffalo_l", "threshold": 0.95},
                    "anti_steg": {"enabled": False},
                },
            ),
            (
                "face_detection.single_face_eq1_threshold",
                {
                    "privacy": {
                        "enabled": True,
                        "recognizers": [{"name": "arcface", "type": "insightface", "model_name": "buffalo_l"}],
                    },
                    "face_detection": {
                        "enabled": True,
                        "model_name": "buffalo_l",
                        "threshold": 0.95,
                        "latent_cosine_threshold": 0.95,
                    },
                    "anti_steg": {"enabled": False},
                },
            ),
            (
                "anti_steg.jpeg_quality",
                {
                    "privacy": {"enabled": False},
                    "face_detection": {"enabled": False},
                    "anti_steg": {
                        "enabled": True,
                        "blur_radius": 1.5,
                        "downsample_scale": 0.5,
                        "crop_fraction": 0.9,
                        "noise_std": 0.01,
                    },
                },
            ),
        ]

        for field, config in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    runner._eval_monitor_configs(config)

    def test_eval_face_count_rows_and_summary_expose_new_rates_with_legacy_ge1(self) -> None:
        rows = [
            {"affective": {"latent_cosine": 0.9}, "face_detection": {}, "anti_steg": {}, "privacy": {}},
            {"affective": {"latent_cosine": 0.8}, "face_detection": {}, "anti_steg": {}, "privacy": {}},
            {"affective": {"latent_cosine": 0.7}, "face_detection": {}, "anti_steg": {}, "privacy": {}},
        ]

        _attach_face_detection_rows(rows, [0, 1, 2])
        summary = _summarize_rows(rows)["face_detection"]

        self.assertEqual(rows[0]["face_detection"]["zero_face_rate"], 1.0)
        self.assertEqual(rows[1]["face_detection"]["single_face_eq1_rate"], 1.0)
        self.assertEqual(rows[2]["face_detection"]["multi_face_rate"], 1.0)
        self.assertAlmostEqual(summary["face_detect_ge1_rate"]["mean"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["single_face_eq1_rate"]["mean"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["zero_face_rate"]["mean"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["multi_face_rate"]["mean"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["detected"]["mean"], summary["face_detect_ge1_rate"]["mean"])

    def test_face_count_rates_rejects_non_integer_counts_without_truncation(self) -> None:
        import numpy as np

        rates = face_count_rates([np.int64(1), 0])
        self.assertAlmostEqual(rates["face_detect_ge1_rate"], 0.5)
        for bad in (True, 1.0, "1"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "integer"):
                    face_count_rates([bad])

    def test_torchscript_recognizer_config_requires_embedding_dim_before_loading(self) -> None:
        from safa.evaluation import recognizers

        with patch.object(recognizers, "TorchScriptRecognizer", side_effect=AssertionError("must validate before loading")):
            with self.assertRaisesRegex(ValueError, "embedding_dim"):
                recognizers.build_recognizers(
                    [{"name": "ts", "type": "torchscript", "checkpoint": "unused.pt", "input_size": 112}],
                    "cpu",
                )

    def test_torchscript_recognizer_asset_description_requires_input_size(self) -> None:
        from safa.evaluation import recognizers

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "recognizer.pt"
            checkpoint.write_bytes(b"not a real torchscript checkpoint")
            with self.assertRaisesRegex(ValueError, "input_size"):
                recognizers.describe_recognizer_assets(
                    [{"name": "ts", "type": "torchscript", "checkpoint": str(checkpoint), "embedding_dim": 512}]
                )

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for eval checkpoint tests")
    def test_eval_generator_loader_rejects_checkpoint_missing_model_config(self) -> None:
        import torch

        from safa.evaluation.runner import _load_generator

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.pt"
            torch.save({"model_state_dict": {}}, path)

            with patch("safa.evaluation.runner.build_generator", side_effect=AssertionError("must not build without model_config")):
                with self.assertRaisesRegex(ValueError, "model_config"):
                    _load_generator(str(path), {}, "cpu")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for eval checkpoint tests")
    def test_eval_generator_loader_rejects_requested_ema_without_state_dict(self) -> None:
        import torch

        from safa.evaluation.runner import _load_generator

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.pt"
            torch.save(
                {
                    "model_config": {
                        "model_type": "conditional_flow_matching",
                        "embedding_dim": 2,
                        "image_size": 4,
                        "base_channels": 4,
                        "channel_multipliers": [1],
                        "time_embedding_dim": 4,
                        "condition_dim": 4,
                        "sample_steps": 1,
                        "train_cycle_steps": 1,
                        "sampler": "euler",
                    },
                    "model_state_dict": {},
                    "ema_config": {"enabled": True},
                    "training_config": {"best_model": "ema"},
                },
                path,
            )

            with patch("safa.evaluation.runner.build_generator", side_effect=AssertionError("must not build without ema state")):
                with self.assertRaisesRegex(ValueError, "ema_model_state_dict"):
                    _load_generator(str(path), {}, "cpu")

    def test_eval_feature_metadata_uses_cache_dim_and_checks_model_dims(self) -> None:
        from safa.evaluation.runner import _feature_metadata_for_eval

        dataset = SimpleNamespace(manifest=SimpleNamespace(feature_dim=128))
        generator = SimpleNamespace(config=SimpleNamespace(embedding_dim=128))
        e0_checkpoint = {"model_config": {"embedding_dim": 128}}

        metadata = _feature_metadata_for_eval(dataset, generator, e0_checkpoint, "features/cache")

        self.assertEqual(metadata, {"dim": 128, "l2_normalized": True, "cache": "features/cache"})

    def test_eval_feature_metadata_rejects_generator_dim_mismatch(self) -> None:
        from safa.evaluation.runner import _feature_metadata_for_eval

        dataset = SimpleNamespace(manifest=SimpleNamespace(feature_dim=128))
        generator = SimpleNamespace(config=SimpleNamespace(embedding_dim=64))
        e0_checkpoint = {"model_config": {"embedding_dim": 128}}

        with self.assertRaisesRegex(RuntimeError, "feature_dim"):
            _feature_metadata_for_eval(dataset, generator, e0_checkpoint, "features/cache")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for eval sampling tests")
    def test_eval_generation_uses_sample_with_stable_x_init_not_forward(self) -> None:
        import torch

        from safa.evaluation.runner import _sample_generated_for_eval
        from safa.utils.sampling import make_x_init_for_sample_ids

        class DummyGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_kwargs = None

            def forward(self, z):
                raise AssertionError("eval must not call generator(z)")

            def sample(self, z, **kwargs):
                self.sample_kwargs = kwargs
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        generator = DummyGenerator()
        z = torch.ones(2, 2)
        generated = _sample_generated_for_eval(generator, z, ["sample-a", "sample-b"], 1337, 4)

        self.assertEqual(tuple(generated.shape), (2, 3, 4, 4))
        self.assertIsNotNone(generator.sample_kwargs)
        expected = make_x_init_for_sample_ids(["sample-a", "sample-b"], 1337, 4, z.device, z.dtype)
        self.assertTrue(torch.equal(generator.sample_kwargs["x_init"], expected))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for eval image export tests")
    def test_eval_single_image_export_uses_default_or_explicit_dir_and_rejects_overwrite(self) -> None:
        import torch

        from safa.evaluation.runner import _generated_image_output_dir, _save_generated_image_for_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples"
            explicit_dir = root / "explicit"

            self.assertIsNone(_generated_image_output_dir({"sample_dir": str(sample_dir)}))
            self.assertEqual(
                _generated_image_output_dir({"sample_dir": str(sample_dir), "save_generated_images": True}),
                sample_dir / "generated_images",
            )
            self.assertEqual(
                _generated_image_output_dir(
                    {
                        "sample_dir": str(sample_dir),
                        "save_generated_images": True,
                        "generated_image_dir": str(explicit_dir),
                    }
                ),
                explicit_dir,
            )

            row = {"sample_id": "../subject 01:abc", "artifacts": {}}
            path = _save_generated_image_for_eval(
                torch.zeros(3, 4, 4),
                explicit_dir,
                global_index=7,
                sample_id=row["sample_id"],
                row=row,
            )

            self.assertEqual(path, explicit_dir / "00000007__subject_01_abc.png")
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, explicit_dir)
            self.assertNotIn("/", path.name)
            self.assertNotIn("\\", path.name)
            self.assertEqual(row["artifacts"]["generated_image_path"], str(path))
            with self.assertRaises(FileExistsError):
                _save_generated_image_for_eval(
                    torch.zeros(3, 4, 4),
                    explicit_dir,
                    global_index=7,
                    sample_id=row["sample_id"],
                    row={"artifacts": {}},
                )

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for eval runner tests")
    def test_run_eval_skips_privacy_without_recognizers_when_single_face_guard_fails(self) -> None:
        import torch

        from safa.evaluation import runner

        class DummyDataset(torch.utils.data.Dataset):
            manifest = SimpleNamespace(feature_dim=2, l2_normalized=True)

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {
                    "image": torch.zeros(3, 4, 4),
                    "z": torch.tensor([1.0, 0.0]),
                    "label": torch.tensor(0),
                    "sample_id": f"sample-{index}",
                }

        class DummyE0(torch.nn.Module):
            def forward(self, images):
                batch = images.shape[0]
                return {
                    "embedding": torch.tensor([[1.0, 0.0]], device=images.device).repeat(batch, 1),
                    "logits": torch.tensor([[1.0, 0.0]], device=images.device).repeat(batch, 1),
                }

        class DummyGenerator(torch.nn.Module):
            config = SimpleNamespace(embedding_dim=2)

            def sample(self, z, **kwargs):
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device)

        class DummyDetector:
            def detect_counts(self, images):
                return [1, 2]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e0_path = root / "e0.pt"
            g_path = root / "g.pt"
            e0_path.write_bytes(b"e0")
            g_path.write_bytes(b"g")
            config = {
                "seed": 1337,
                "sampling_seed": 1337,
                "device": "cuda:0",
                "num_workers": 0,
                "batch_size": 2,
                "image_size": 4,
                "index": "dummy-index",
                "features": "dummy-features",
                "e0_checkpoint": str(e0_path),
                "g_checkpoint": str(g_path),
                "out_json": str(root / "result.json"),
                "per_sample_jsonl": str(root / "per_sample.jsonl"),
                "sample_dir": str(root / "samples"),
                "face_detection": {
                    "enabled": True,
                    "model_name": "buffalo_l",
                    "threshold": 0.95,
                    "single_face_eq1_threshold": 0.98,
                    "latent_cosine_threshold": 0.95,
                },
                "privacy": {
                    "enabled": True,
                    "recognizers": [{"name": "arcface", "type": "insightface", "model_name": "buffalo_l"}],
                },
                "anti_steg": {"enabled": False},
            }

            with (
                patch.object(runner, "require_cuda_device", return_value=torch.device("cpu")),
                patch.object(runner, "FeatureAlignedAffectNet", return_value=DummyDataset()),
                patch.object(runner, "load_e0_checkpoint", return_value=(DummyE0(), {"model_config": {"embedding_dim": 2}})),
                patch.object(runner, "_load_generator", return_value=DummyGenerator()),
                patch.object(runner, "_build_face_detector", return_value=DummyDetector()),
                patch.object(runner, "build_recognizers", side_effect=AssertionError("recognizers must not load")),
                patch.object(runner, "describe_recognizer_assets", side_effect=AssertionError("recognizers must not describe")),
            ):
                result = runner.run_eval_from_config(config)

            self.assertTrue(result["privacy_skipped"])
            self.assertEqual(result["skip_reason"], "privacy_guard_failed")
            self.assertFalse(result["privacy_guard_pass"])
            self.assertEqual(result["metrics"]["privacy"], {})
            serialized_metrics = json.dumps(result["metrics"], sort_keys=True)
            for forbidden in ("tar_at_far", "eer", "auc"):
                self.assertNotIn(forbidden, serialized_metrics)
            self.assertTrue(Path(config["out_json"]).is_file())
            self.assertTrue(Path(config["per_sample_jsonl"]).is_file())
            persisted = json.loads(Path(config["out_json"]).read_text(encoding="utf-8"))
            self.assertTrue(persisted["privacy_skipped"])
            self.assertEqual(persisted["metrics"]["privacy"], {})
            persisted_metrics = json.dumps(persisted["metrics"], sort_keys=True)
            for forbidden in ("tar_at_far", "eer", "auc"):
                self.assertNotIn(forbidden, persisted_metrics)

    def test_privacy_summary_adds_roc_metrics_from_clean_same_and_impostor_scores(self) -> None:
        rows = []
        for same, impostor in [(0.9, 0.1), (0.8, 0.4), (0.4, 0.4)]:
            rows.append(
                {
                    "affective": {"latent_cosine": 1.0},
                    "face_detection": {},
                    "anti_steg": {},
                    "privacy": {"dummy": {"same_similarity": same, "impostor_similarity": impostor}},
                }
            )

        summary = _summarize_rows(rows)["privacy"]["dummy"]

        self.assertAlmostEqual(summary["same_identity_similarity_mean"], 0.7)
        self.assertAlmostEqual(summary["tar_at_far_1e-3"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["tar_at_far_1e-4"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["auc"], 8.0 / 9.0)
        self.assertGreaterEqual(summary["eer"], 0.0)
        self.assertLess(summary["eer"], 0.35)

    def test_privacy_summary_rejects_missing_clean_same_or_impostor_scores(self) -> None:
        rows = [
            {
                "affective": {"latent_cosine": 1.0},
                "face_detection": {},
                "anti_steg": {},
                "privacy": {"dummy": {"same_similarity": 0.9}},
            }
        ]

        with self.assertRaisesRegex(ValueError, "same_similarity.*impostor_similarity"):
            _summarize_rows(rows)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for validation sampling tests")
    def test_validation_reuses_stable_x_init_for_same_sample_id(self) -> None:
        import torch

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _evaluate_validation

        class DummyGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x_inits = []

            def sample(self, z, **kwargs):
                self.x_inits.append(kwargs["x_init"].detach().clone())
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        class DummyE0(torch.nn.Module):
            def forward(self, images):
                batch = images.shape[0]
                return {"embedding": torch.ones(batch, 2, device=images.device), "logits": torch.zeros(batch, 2, device=images.device)}

        generator = DummyGenerator()
        loader = [
            {"image": torch.zeros(1, 3, 4, 4), "z": torch.ones(1, 2), "sample_id": ["same-sample"]},
            {"image": torch.zeros(1, 3, 4, 4), "z": torch.ones(1, 2), "sample_id": ["same-sample"]},
        ]

        _evaluate_validation(
            generator,
            DummyE0(),
            loader,
            detector=None,
            device=torch.device("cpu"),
            generator_config=FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1),
            sampling_seed=1337,
        )

        self.assertEqual(len(generator.x_inits), 2)
        self.assertTrue(torch.equal(generator.x_inits[0], generator.x_inits[1]))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for validation metric tests")
    def test_validation_face_count_rates_aggregate_counts_with_legacy_ge1(self) -> None:
        import torch

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _evaluate_validation

        class DummyGenerator(torch.nn.Module):
            def sample(self, z, **kwargs):
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        class DummyE0(torch.nn.Module):
            def forward(self, images):
                batch = images.shape[0]
                return {"embedding": torch.ones(batch, 2, device=images.device), "logits": torch.zeros(batch, 2, device=images.device)}

        class DummyDetector:
            def detect_counts(self, images):
                return [0, 1, 2]

        loader = [
            {
                "image": torch.zeros(3, 3, 4, 4),
                "z": torch.ones(3, 2),
                "sample_id": ["zero", "single", "multi"],
            }
        ]

        metrics = _evaluate_validation(
            DummyGenerator(),
            DummyE0(),
            loader,
            detector=DummyDetector(),
            device=torch.device("cpu"),
            generator_config=FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1),
            sampling_seed=1337,
        )

        self.assertAlmostEqual(metrics["face_detect_ge1_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["single_face_eq1_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["zero_face_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["multi_face_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["face_detection_rate"], metrics["face_detect_ge1_rate"])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for privacy cache tests")
    def test_privacy_pass_uses_cached_generated_images(self) -> None:
        import torch

        class DummyRecognizer:
            name = "dummy"

            def embed(self, images):
                return torch.nn.functional.normalize(images.flatten(1)[:, :4].float() + 1.0, p=2, dim=1)

        loader = [{"image": torch.zeros(2, 3, 4, 4)}]
        generated = [torch.ones(2, 3, 4, 4)]
        store = {"dummy": {"source": [], "generated": {"clean": []}}}
        _run_privacy_pass({}, loader, generated, [DummyRecognizer()], {}, store, torch.device("cpu"))
        self.assertEqual(len(store["dummy"]["source"]), 1)
        self.assertEqual(len(store["dummy"]["generated"]["clean"]), 1)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for privacy cache tests")
    def test_privacy_pass_reports_non_single_face_as_protocol_blocker(self) -> None:
        import torch

        from safa.evaluation.runner import PrivacyProtocolError

        class BadRecognizer:
            name = "arcface"

            def embed(self, images):
                raise RuntimeError("Recognizer arcface expected exactly one face, detected 2")

        loader = [{"image": torch.zeros(1, 3, 4, 4)}]
        generated = [torch.ones(1, 3, 4, 4)]
        store = {"arcface": {"source": [], "generated": {"clean": []}}}

        with self.assertRaisesRegex(PrivacyProtocolError, "Privacy protocol blocker.*source.*expected exactly one face"):
            _run_privacy_pass({}, loader, generated, [BadRecognizer()], {}, store, torch.device("cpu"))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for privacy cache tests")
    def test_privacy_pass_rejects_generated_cache_mismatch(self) -> None:
        import torch

        class DummyRecognizer:
            name = "dummy"

            def embed(self, images):
                return torch.ones(images.shape[0], 4)

        loader = [{"image": torch.zeros(2, 3, 4, 4)}]
        store = {"dummy": {"source": [], "generated": {"clean": []}}}
        with self.assertRaises(RuntimeError):
            _run_privacy_pass({}, loader, [], [DummyRecognizer()], {}, store, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
