from __future__ import annotations

import inspect
import importlib.util
import math
import unittest

from safa.evaluation.metrics import flatten_finite_numbers, summarize
from safa.evaluation import perturbations
from safa.evaluation.runner import _guard_result, _run_privacy_pass, deterministic_impostor_indices


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
            "face_detection": {"detected": {"mean": 0.99}},
            "latent_cosine": {"mean": 0.94},
        }
        guard = _guard_result(metrics, {"enabled": True, "threshold": 0.95, "latent_cosine_threshold": 0.95})
        self.assertFalse(guard["passed"])
        metrics["latent_cosine"]["mean"] = 0.96
        guard = _guard_result(metrics, {"enabled": True, "threshold": 0.95, "latent_cosine_threshold": 0.95})
        self.assertTrue(guard["passed"])

    def test_face_detection_guard_rejects_missing_detection_metrics(self) -> None:
        with self.assertRaises(RuntimeError):
            _guard_result({"latent_cosine": {"mean": 0.99}}, {"enabled": True})

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
