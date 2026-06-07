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
    def _run_candidate_rerank_eval(
        self,
        *,
        candidate_rerank: dict | None,
        cosines: list[float] | list[list[float]],
        face_counts: list[int] | list[list[int]] | None,
        face_detection_enabled: bool,
        dataset_len: int = 1,
        batch_size: int = 1,
        save_generated_images: bool = False,
    ):
        import torch

        from safa.evaluation import runner

        def candidate_value(values, sample_index: int, candidate_index: int):
            first = values[0]
            if isinstance(first, list):
                return values[sample_index][candidate_index]
            return values[candidate_index]

        dataset_len = int(dataset_len)
        batch_size = int(batch_size)
        num_candidates = 1
        if candidate_rerank is not None and candidate_rerank.get("enabled", False):
            num_candidates = int(candidate_rerank.get("num_candidates", 1))

        class DummyDataset(torch.utils.data.Dataset):
            manifest = SimpleNamespace(feature_dim=2, l2_normalized=True)

            def __len__(self):
                return dataset_len

            def __getitem__(self, index):
                return {
                    "image": torch.full((3, 4, 4), -1.0),
                    "z": torch.tensor([1.0, float(index)]),
                    "label": torch.tensor(index % 2),
                    "sample_id": f"sample-{index}",
                }

        class DummyE0(torch.nn.Module):
            def forward(self, images):
                embeddings = []
                for image in images.detach().cpu():
                    marker = float(image[0, 0, 0])
                    if marker < 0:
                        embeddings.append([1.0, 0.0])
                        continue
                    candidate_index = int(round(marker))
                    sample_index = int(round(float(image[0, 0, 1])))
                    cosine = float(candidate_value(cosines, sample_index, candidate_index))
                    z_unit = torch.tensor([1.0, float(sample_index)], dtype=torch.float64)
                    z_unit = z_unit / z_unit.norm()
                    perpendicular = torch.tensor([-z_unit[1], z_unit[0]], dtype=torch.float64)
                    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
                    embedding = cosine * z_unit + sine * perpendicular
                    embeddings.append([float(embedding[0]), float(embedding[1])])
                embedding = torch.tensor(embeddings, device=images.device, dtype=images.dtype)
                logits = torch.tensor([[1.0, 0.0]], device=images.device, dtype=images.dtype).repeat(images.shape[0], 1)
                return {"embedding": embedding, "logits": logits}

        class DummyGenerator(torch.nn.Module):
            config = SimpleNamespace(embedding_dim=2)

            def __init__(self):
                super().__init__()
                self.x_inits = []
                self.saved_images = []

            def sample(self, z, **kwargs):
                call_index = len(self.x_inits)
                self.x_inits.append(kwargs["x_init"].detach().clone())
                candidate_index = call_index % num_candidates
                generated = torch.zeros((z.shape[0], 3, 4, 4), device=z.device, dtype=z.dtype)
                generated[:, 0, 0, 0] = float(candidate_index)
                generated[:, 0, 0, 1] = z[:, 1]
                return generated

        class DummyDetector:
            def detect_counts(self, images):
                counts = []
                for image in images.detach().cpu():
                    candidate_index = int(round(float(image[0, 0, 0])))
                    sample_index = int(round(float(image[0, 0, 1])))
                    counts.append(int(candidate_value(face_counts, sample_index, candidate_index)))
                return counts

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
                "batch_size": batch_size,
                "image_size": 4,
                "index": "dummy-index",
                "features": "dummy-features",
                "e0_checkpoint": str(e0_path),
                "g_checkpoint": str(g_path),
                "out_json": str(root / "result.json"),
                "per_sample_jsonl": str(root / "per_sample.jsonl"),
                "sample_dir": str(root / "samples"),
                "face_detection": {"enabled": False},
                "privacy": {"enabled": False},
                "anti_steg": {"enabled": False},
            }
            if face_detection_enabled:
                config["face_detection"] = {
                    "enabled": True,
                    "model_name": "buffalo_l",
                    "threshold": 0.95,
                    "single_face_eq1_threshold": 0.0,
                    "latent_cosine_threshold": 0.0,
                }
            if candidate_rerank is not None:
                config["candidate_rerank"] = candidate_rerank
            if save_generated_images:
                config["save_generated_images"] = True

            generator = DummyGenerator()
            detector = DummyDetector() if face_counts is not None else None

            def fake_save_generated_image_for_eval(image, output_dir, *, global_index: int, sample_id, row: dict):
                generator.saved_images.append(
                    {
                        "image": image.detach().clone(),
                        "global_index": int(global_index),
                        "sample_id": str(sample_id),
                    }
                )
                path = Path(output_dir) / f"{int(global_index):08d}.png"
                row.setdefault("artifacts", {})["generated_image_path"] = str(path)
                return path

            with (
                patch.object(runner, "require_cuda_device", return_value=torch.device("cpu")),
                patch.object(runner, "FeatureAlignedAffectNet", return_value=DummyDataset()),
                patch.object(runner, "load_e0_checkpoint", return_value=(DummyE0(), {"model_config": {"embedding_dim": 2}})),
                patch.object(runner, "_load_generator", return_value=generator),
                patch.object(runner, "_build_face_detector", return_value=detector),
                patch.object(runner, "normalize_for_e0", side_effect=lambda images: images),
                patch.object(runner, "_save_generated_image_for_eval", side_effect=fake_save_generated_image_for_eval),
            ):
                result = runner.run_eval_from_config(config)

            rows = [
                json.loads(line)
                for line in Path(config["per_sample_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return result, rows, generator

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
                    _load_generator(str(path), {"checkpoint_model": "ema"}, "cpu")

    def test_eval_checkpoint_model_source_requires_explicit_config(self) -> None:
        from safa.evaluation import runner

        with self.assertRaisesRegex(ValueError, "checkpoint_model.*required"):
            runner._eval_checkpoint_model_source({"training_config": {"best_model": "ema"}}, {})

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

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_disabled_preserves_single_candidate_sampling(self) -> None:
        import torch

        from safa.utils.sampling import make_x_init_for_sample_ids

        _, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank=None,
            cosines=[0.8],
            face_counts=None,
            face_detection_enabled=False,
        )

        self.assertEqual(len(generator.x_inits), 1)
        expected = make_x_init_for_sample_ids(["sample-0"], 1337, 4, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(generator.x_inits[0], expected))
        self.assertNotIn("candidate_rerank", rows[0])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_prefers_single_face_before_cosine(self) -> None:
        _, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 3,
                "single_face_priority": True,
                "selection_metric": "latent_cosine",
            },
            cosines=[0.95, 0.8, 0.7],
            face_counts=[2, 1, 1],
            face_detection_enabled=True,
        )

        self.assertEqual(len(generator.x_inits), 3)
        self.assertEqual(rows[0]["candidate_rerank"]["selected_candidate_index"], 1)
        self.assertAlmostEqual(rows[0]["affective"]["latent_cosine"], 0.8)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_uses_latent_cosine_after_single_face_gate(self) -> None:
        _, rows, _ = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 3,
                "single_face_priority": True,
                "selection_metric": "latent_cosine",
            },
            cosines=[0.6, 0.9, 0.99],
            face_counts=[1, 1, 2],
            face_detection_enabled=True,
        )

        self.assertEqual(rows[0]["candidate_rerank"]["selected_candidate_index"], 1)
        self.assertAlmostEqual(rows[0]["candidate_rerank"]["selected_latent_cosine"], 0.9)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_falls_back_to_latent_cosine_without_face_detector(self) -> None:
        result, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 3,
                "single_face_priority": True,
                "selection_metric": "latent_cosine",
            },
            cosines=[0.55, 0.92, 0.8],
            face_counts=None,
            face_detection_enabled=False,
        )

        self.assertEqual(len(generator.x_inits), 3)
        self.assertEqual(result["sampling"]["candidate_rerank"]["enabled"], True)
        self.assertEqual(rows[0]["face_detection"], {})
        metadata = rows[0]["candidate_rerank"]
        self.assertEqual(metadata["selected_candidate_index"], 1)
        self.assertAlmostEqual(metadata["selected_latent_cosine"], 0.92)
        self.assertIsNone(metadata["selected_face_count"])
        self.assertNotIn("face_count", metadata["candidates"][0])
        self.assertAlmostEqual(rows[0]["affective"]["latent_cosine"], 0.92)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_selects_independently_per_sample_in_batch(self) -> None:
        import torch

        from safa.utils.sampling import make_x_init_for_sample_ids

        cosines = [
            [0.95, 0.20, 0.10],
            [0.15, 0.91, 0.25],
            [0.30, 0.35, 0.93],
        ]

        result, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 3,
                "single_face_priority": False,
                "selection_metric": "latent_cosine",
            },
            cosines=cosines,
            face_counts=None,
            face_detection_enabled=False,
            dataset_len=3,
            batch_size=3,
            save_generated_images=True,
        )

        sample_ids = ["sample-0", "sample-1", "sample-2"]
        selected_indices = [0, 1, 2]
        self.assertEqual(result["dataset"]["num_samples"], 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(generator.x_inits), 3)
        self.assertEqual(len(generator.saved_images), 3)
        self.assertEqual([row["sample_id"] for row in rows], sample_ids)

        for candidate_index, x_init in enumerate(generator.x_inits):
            self.assertEqual(tuple(x_init.shape), (3, 3, 4, 4))
            expected = make_x_init_for_sample_ids(
                [f"{sample_id}::candidate::{candidate_index}" for sample_id in sample_ids],
                1337,
                4,
                x_init.device,
                x_init.dtype,
            )
            self.assertTrue(torch.equal(x_init, expected))

        for sample_index, row in enumerate(rows):
            selected_index = selected_indices[sample_index]
            metadata = row["candidate_rerank"]
            self.assertEqual(metadata["enabled"], True)
            self.assertEqual(metadata["num_candidates"], 3)
            self.assertEqual(metadata["single_face_priority"], False)
            self.assertEqual(metadata["selection_metric"], "latent_cosine")
            self.assertEqual(metadata["selected_candidate_index"], selected_index)
            self.assertAlmostEqual(
                metadata["selected_latent_cosine"],
                cosines[sample_index][selected_index],
                places=6,
            )
            self.assertIsNone(metadata["selected_face_count"])
            self.assertEqual([candidate["index"] for candidate in metadata["candidates"]], [0, 1, 2])
            self.assertFalse(any("face_count" in candidate for candidate in metadata["candidates"]))
            for candidate_index, candidate in enumerate(metadata["candidates"]):
                self.assertAlmostEqual(
                    candidate["latent_cosine"],
                    cosines[sample_index][candidate_index],
                    places=6,
                )
            self.assertAlmostEqual(
                row["affective"]["latent_cosine"],
                cosines[sample_index][selected_index],
                places=6,
            )

            saved = generator.saved_images[sample_index]
            self.assertEqual(saved["global_index"], sample_index)
            self.assertEqual(saved["sample_id"], sample_ids[sample_index])
            self.assertEqual(tuple(saved["image"].shape), (3, 4, 4))
            self.assertEqual(int(saved["image"][0, 0, 0].item()), selected_index)
            self.assertEqual(int(saved["image"][0, 0, 1].item()), sample_index)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_records_selection_metadata(self) -> None:
        import torch

        from safa.utils.sampling import make_x_init_for_sample_ids

        result, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 2,
                "single_face_priority": True,
                "selection_metric": "latent_cosine",
            },
            cosines=[0.7, 0.85],
            face_counts=[0, 1],
            face_detection_enabled=True,
        )

        self.assertEqual(result["sampling"]["candidate_rerank"]["enabled"], True)
        self.assertEqual(result["sampling"]["candidate_rerank"]["num_candidates"], 2)
        expected_candidate_0 = make_x_init_for_sample_ids(
            ["sample-0::candidate::0"],
            1337,
            4,
            generator.x_inits[0].device,
            generator.x_inits[0].dtype,
        )
        self.assertTrue(torch.equal(generator.x_inits[0], expected_candidate_0))
        metadata = rows[0]["candidate_rerank"]
        self.assertEqual(metadata["enabled"], True)
        self.assertEqual(metadata["num_candidates"], 2)
        self.assertEqual(metadata["selected_candidate_index"], 1)
        self.assertAlmostEqual(metadata["selected_latent_cosine"], 0.85)
        self.assertEqual(metadata["selected_face_count"], 1)
        self.assertEqual([candidate["index"] for candidate in metadata["candidates"]], [0, 1])
        self.assertEqual([candidate["face_count"] for candidate in metadata["candidates"]], [0, 1])
        self.assertAlmostEqual(metadata["candidates"][0]["latent_cosine"], 0.7)
        self.assertAlmostEqual(metadata["candidates"][1]["latent_cosine"], 0.85)

    def test_candidate_rerank_parses_adaptive_k_config(self) -> None:
        from safa.evaluation import runner

        config = runner._candidate_rerank_config(
            {
                "candidate_rerank": {
                    "enabled": True,
                    "num_candidates": 48,
                    "single_face_priority": True,
                    "selection_metric": "latent_cosine",
                    "adaptive_k": {
                        "enabled": True,
                        "min_candidates": 8,
                        "accept_latent_cosine_threshold": 0.95,
                        "require_single_face": True,
                    },
                }
            }
        )

        self.assertEqual(
            config["adaptive_k"],
            {
                "enabled": True,
                "min_candidates": 8,
                "accept_latent_cosine_threshold": 0.95,
                "require_single_face": True,
            },
        )
        self.assertEqual(
            runner._candidate_rerank_result_metadata(config)["adaptive_k"],
            {
                "enabled": True,
                "min_candidates": 8,
                "accept_latent_cosine_threshold": 0.95,
                "require_single_face": True,
            },
        )

    def test_candidate_rerank_rejects_invalid_adaptive_k_config(self) -> None:
        from safa.evaluation import runner

        def parse_adaptive(adaptive_k: dict):
            return runner._candidate_rerank_config(
                {
                    "candidate_rerank": {
                        "enabled": True,
                        "num_candidates": 4,
                        "single_face_priority": True,
                        "selection_metric": "latent_cosine",
                        "adaptive_k": adaptive_k,
                    }
                }
            )

        cases = [
            ({}, "candidate_rerank.adaptive_k.enabled"),
            (
                {"enabled": True, "min_candidates": 0, "accept_latent_cosine_threshold": 0.95, "require_single_face": True},
                "candidate_rerank.adaptive_k.min_candidates",
            ),
            (
                {"enabled": True, "min_candidates": 5, "accept_latent_cosine_threshold": 0.95, "require_single_face": True},
                "min_candidates.*num_candidates",
            ),
            (
                {"enabled": True, "min_candidates": 2, "accept_latent_cosine_threshold": math.nan, "require_single_face": True},
                "candidate_rerank.adaptive_k.accept_latent_cosine_threshold",
            ),
            (
                {"enabled": True, "min_candidates": 2, "accept_latent_cosine_threshold": 0.95, "require_single_face": "yes"},
                "candidate_rerank.adaptive_k.require_single_face",
            ),
        ]
        for adaptive_k, message in cases:
            with self.subTest(adaptive_k=adaptive_k):
                with self.assertRaisesRegex(ValueError, message):
                    parse_adaptive(adaptive_k)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_adaptive_stops_each_sample_independently(self) -> None:
        cosines = [
            [0.1, 0.95, 0.99, 0.99],
            [0.1, 0.2, 0.3, 0.4],
        ]

        result, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 4,
                "single_face_priority": False,
                "selection_metric": "latent_cosine",
                "adaptive_k": {
                    "enabled": True,
                    "min_candidates": 2,
                    "accept_latent_cosine_threshold": 0.9,
                    "require_single_face": False,
                },
            },
            cosines=cosines,
            face_counts=None,
            face_detection_enabled=False,
            dataset_len=2,
            batch_size=2,
        )

        self.assertEqual(result["sampling"]["candidate_rerank"]["adaptive_k"]["enabled"], True)
        self.assertEqual(
            [tuple(x_init.shape) for x_init in generator.x_inits],
            [(2, 3, 4, 4), (2, 3, 4, 4), (1, 3, 4, 4), (1, 3, 4, 4)],
        )

        first = rows[0]["candidate_rerank"]
        self.assertEqual(first["num_candidates"], 4)
        self.assertEqual(first["num_candidates_evaluated"], 2)
        self.assertEqual(first["stop_reason"], "threshold_passed")
        self.assertEqual(first["selected_candidate_index"], 1)
        self.assertAlmostEqual(first["best_score"], 0.95)
        self.assertTrue(first["threshold_passed"])
        self.assertEqual([candidate["index"] for candidate in first["candidates"]], [0, 1])

        second = rows[1]["candidate_rerank"]
        self.assertEqual(second["num_candidates_evaluated"], 4)
        self.assertEqual(second["stop_reason"], "max_k_no_threshold_passed")
        self.assertEqual(second["selected_candidate_index"], 3)
        self.assertAlmostEqual(second["best_score"], 0.4)
        self.assertFalse(second["threshold_passed"])
        self.assertEqual([candidate["index"] for candidate in second["candidates"]], [0, 1, 2, 3])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_adaptive_respects_min_candidates_before_stopping(self) -> None:
        _, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 5,
                "single_face_priority": False,
                "selection_metric": "latent_cosine",
                "adaptive_k": {
                    "enabled": True,
                    "min_candidates": 3,
                    "accept_latent_cosine_threshold": 0.9,
                    "require_single_face": False,
                },
            },
            cosines=[0.99, 0.91, 0.2, 0.1, 0.1],
            face_counts=None,
            face_detection_enabled=False,
        )

        metadata = rows[0]["candidate_rerank"]
        self.assertEqual(len(generator.x_inits), 3)
        self.assertEqual(metadata["num_candidates_evaluated"], 3)
        self.assertEqual(metadata["stop_reason"], "threshold_passed")
        self.assertEqual(metadata["selected_candidate_index"], 0)
        self.assertAlmostEqual(metadata["best_score"], 0.99)
        self.assertTrue(metadata["threshold_passed"])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for candidate rerank tests")
    def test_candidate_rerank_adaptive_requires_single_face_for_threshold(self) -> None:
        _, rows, generator = self._run_candidate_rerank_eval(
            candidate_rerank={
                "enabled": True,
                "num_candidates": 4,
                "single_face_priority": True,
                "selection_metric": "latent_cosine",
                "adaptive_k": {
                    "enabled": True,
                    "min_candidates": 2,
                    "accept_latent_cosine_threshold": 0.9,
                    "require_single_face": True,
                },
            },
            cosines=[0.1, 0.96, 0.94, 0.2],
            face_counts=[1, 2, 1, 1],
            face_detection_enabled=True,
        )

        metadata = rows[0]["candidate_rerank"]
        self.assertEqual(len(generator.x_inits), 3)
        self.assertEqual(metadata["num_candidates_evaluated"], 3)
        self.assertEqual(metadata["stop_reason"], "threshold_passed")
        self.assertEqual(metadata["selected_candidate_index"], 2)
        self.assertAlmostEqual(metadata["best_score"], 0.94)
        self.assertTrue(metadata["threshold_passed"])
        self.assertEqual(metadata["selected_face_count"], 1)

    def test_candidate_rerank_rejects_invalid_config(self) -> None:
        from safa.evaluation import runner

        with self.assertRaisesRegex(ValueError, "candidate_rerank.num_candidates"):
            runner._candidate_rerank_config(
                {
                    "candidate_rerank": {
                        "enabled": True,
                        "num_candidates": 0,
                        "single_face_priority": True,
                        "selection_metric": "latent_cosine",
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "candidate_rerank.selection_metric"):
            runner._candidate_rerank_config(
                {
                    "candidate_rerank": {
                        "enabled": True,
                        "num_candidates": 4,
                        "single_face_priority": True,
                        "selection_metric": "pixel_mse",
                    }
                }
            )

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
            def __init__(self):
                super().__init__()
                self.generated_embeddings = [
                    torch.tensor([[1.0, 0.0, 0.0]]),
                    torch.tensor([[1.0, 0.0, 0.0]]),
                    torch.tensor([[0.0, 1.0, 0.0]]),
                ]
                self.calls = 0

            def forward(self, images):
                batch = images.shape[0]
                generated_call = self.calls % 2 == 1
                self.calls += 1
                if generated_call:
                    embedding = self.generated_embeddings.pop(0).to(images.device)
                else:
                    embedding = torch.zeros(batch, 3, device=images.device)
                    embedding[:, 0] = 1.0
                return {"embedding": embedding, "logits": torch.zeros(batch, 2, device=images.device)}

        generator = DummyGenerator()
        loader = [
            {"image": torch.zeros(1, 3, 4, 4), "z": torch.tensor([[1.0, 0.0, 0.0]]), "sample_id": ["same-sample"]},
            {"image": torch.zeros(1, 3, 4, 4), "z": torch.tensor([[1.0, 0.0, 0.0]]), "sample_id": ["same-sample"]},
            {"image": torch.zeros(1, 3, 4, 4), "z": torch.tensor([[0.0, 1.0, 0.0]]), "sample_id": ["different-sample"]},
        ]

        _evaluate_validation(
            generator,
            DummyE0(),
            loader,
            detector=None,
            device=torch.device("cpu"),
            generator_config=FlowGeneratorConfig(embedding_dim=3, image_size=4, sample_steps=1),
            sampling_seed=1337,
        )

        self.assertEqual(len(generator.x_inits), 3)
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
            def __init__(self, generated_embeddings):
                super().__init__()
                self.generated_embeddings = generated_embeddings
                self.calls = 0

            def forward(self, images):
                batch = images.shape[0]
                generated_call = self.calls % 2 == 1
                self.calls += 1
                if generated_call:
                    embedding = self.generated_embeddings.to(images.device)
                else:
                    embedding = torch.zeros(batch, self.generated_embeddings.shape[1], device=images.device)
                    embedding[:, 0] = 1.0
                return {"embedding": embedding, "logits": torch.zeros(batch, 2, device=images.device)}

        class DummyDetector:
            def detect_counts(self, images):
                return [0, 1, 2]

        z = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0**-0.5, 2.0**-0.5, 0.0],
            ]
        )
        loader = [
            {
                "image": torch.zeros(3, 3, 4, 4),
                "z": z,
                "sample_id": ["zero", "single", "multi"],
            }
        ]

        metrics = _evaluate_validation(
            DummyGenerator(),
            DummyE0(z),
            loader,
            detector=DummyDetector(),
            device=torch.device("cpu"),
            generator_config=FlowGeneratorConfig(embedding_dim=3, image_size=4, sample_steps=1),
            sampling_seed=1337,
        )

        self.assertAlmostEqual(metrics["repr_point_loss"], 0.0)
        self.assertAlmostEqual(metrics["repr_relation_loss"], 0.0)
        self.assertAlmostEqual(metrics["offdiag_gram_mse"], 0.0)
        self.assertAlmostEqual(metrics["offdiag_gram_mae"], 0.0)
        self.assertAlmostEqual(metrics["pairwise_pearson"], 1.0)
        self.assertAlmostEqual(metrics["pairwise_spearman"], 1.0)
        self.assertAlmostEqual(metrics["face_detect_ge1_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["single_face_eq1_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["zero_face_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["multi_face_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["face_detection_rate"], metrics["face_detect_ge1_rate"])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for validation metric tests")
    def test_validation_relation_metrics_fail_fast_above_dense_gram_cap(self) -> None:
        import torch

        from safa.evaluation.metrics import DEFAULT_DENSE_GRAM_MAX_SAMPLES
        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _evaluate_validation

        class DummyGenerator(torch.nn.Module):
            def sample(self, z, **kwargs):
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        class DummyE0(torch.nn.Module):
            def forward(self, images):
                embedding = torch.zeros(images.shape[0], 2, device=images.device)
                embedding[:, 0] = 1.0
                return {"embedding": embedding, "logits": torch.zeros(images.shape[0], 2, device=images.device)}

        sample_count = DEFAULT_DENSE_GRAM_MAX_SAMPLES + 1
        z = torch.zeros(sample_count, 2)
        z[:, 0] = 1.0
        loader = [
            {
                "image": torch.zeros(sample_count, 3, 4, 4),
                "z": z,
                "sample_id": [f"sample-{index}" for index in range(sample_count)],
            }
        ]

        with patch("safa.training.g_loop.compute_validation_relation_metrics", side_effect=AssertionError("dense Gram should not run")):
            with self.assertRaisesRegex(ValueError, "dense Gram cap.*2048"):
                _evaluate_validation(
                    DummyGenerator(),
                    DummyE0(),
                    loader,
                    detector=None,
                    device=torch.device("cpu"),
                    generator_config=FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1),
                    sampling_seed=1337,
                )

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
