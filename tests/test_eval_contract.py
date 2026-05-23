from __future__ import annotations

import inspect
import importlib.util
import math
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from safa.evaluation.metrics import flatten_finite_numbers, summarize
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
                "detected": {"mean": 0.99},
                "face_detect_ge1_rate": {"mean": 0.99},
                "single_face_eq1_rate": {"mean": 0.98},
                "zero_face_rate": {"mean": 0.01},
                "multi_face_rate": {"mean": 0.01},
            },
            "latent_cosine": {"mean": 0.94},
        }
        guard = _guard_result(metrics, {"enabled": True, "threshold": 0.95, "latent_cosine_threshold": 0.95})
        self.assertFalse(guard["passed"])
        self.assertEqual(guard["face_detection_rate"], metrics["face_detection"]["detected"]["mean"])
        self.assertEqual(guard["face_detect_ge1_rate"], metrics["face_detection"]["face_detect_ge1_rate"]["mean"])
        self.assertEqual(guard["single_face_eq1_rate"], metrics["face_detection"]["single_face_eq1_rate"]["mean"])
        self.assertEqual(guard["zero_face_rate"], metrics["face_detection"]["zero_face_rate"]["mean"])
        self.assertEqual(guard["multi_face_rate"], metrics["face_detection"]["multi_face_rate"]["mean"])
        metrics["latent_cosine"]["mean"] = 0.96
        guard = _guard_result(metrics, {"enabled": True, "threshold": 0.95, "latent_cosine_threshold": 0.95})
        self.assertTrue(guard["passed"])

    def test_face_detection_guard_rejects_missing_detection_metrics(self) -> None:
        with self.assertRaises(RuntimeError):
            _guard_result({"latent_cosine": {"mean": 0.99}}, {"enabled": True})

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
