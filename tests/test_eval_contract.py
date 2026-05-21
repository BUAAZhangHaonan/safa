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
