from __future__ import annotations

import unittest

from safa.training.g_loop import _assert_stage1_gate_allows_stage2


class StageGateTests(unittest.TestCase):
    def test_blocks_stage2_when_detection_rate_missing(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 1}}
        with self.assertRaises(RuntimeError):
            _assert_stage1_gate_allows_stage2(stages, stable_hits=0, detection_rate=None, allow_bypass=False)

    def test_blocks_stage2_when_detection_gate_not_stable(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 2}}
        with self.assertRaises(RuntimeError):
            _assert_stage1_gate_allows_stage2(stages, stable_hits=1, detection_rate=0.99, allow_bypass=False)

    def test_allows_stage2_after_gate(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 2}}
        _assert_stage1_gate_allows_stage2(stages, stable_hits=2, detection_rate=0.99, allow_bypass=False)

    def test_smoke_bypass_is_explicit(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 1}}
        _assert_stage1_gate_allows_stage2(stages, stable_hits=0, detection_rate=0.0, allow_bypass=True)

    def test_stage2_cycle_uses_stable_x_init_and_unclamped_sampling(self) -> None:
        import torch
        from torch import nn

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _GeneratorTrainingStep
        from safa.utils.sampling import make_x_init_for_sample_ids

        class DummyGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_calls = []

            def flow_matching_loss(self, images, z):
                loss = images.sum() * 0.0 + z.sum() * 0.0
                return loss, {"flow_matching_mse": loss.detach()}

            def sample(self, z, **kwargs):
                self.sample_calls.append(kwargs)
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        class DummyE0(nn.Module):
            def forward(self, images):
                return {"embedding": torch.ones(images.shape[0], 2, device=images.device), "logits": torch.zeros(images.shape[0], 2, device=images.device)}

        generator = DummyGenerator()
        module = _GeneratorTrainingStep(
            generator,
            DummyE0(),
            FlowGeneratorConfig(embedding_dim=2, image_size=4, train_cycle_steps=1),
            sampling_seed=1337,
        )
        z = torch.ones(2, 2)

        module(torch.zeros(2, 3, 4, 4), z, ["sample-b", "sample-a"], True, 1.0)

        self.assertEqual(len(generator.sample_calls), 1)
        sample_kwargs = generator.sample_calls[0]
        self.assertIs(sample_kwargs["clamp_output"], False)
        self.assertIsNotNone(sample_kwargs["x_init"])
        expected = make_x_init_for_sample_ids(["sample-b", "sample-a"], 1337, 4, z.device, z.dtype)
        self.assertTrue(torch.equal(sample_kwargs["x_init"], expected))

    def test_stage2_gradient_conflict_config_requires_explicit_setting(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        with self.assertRaisesRegex(ValueError, "stages.stage2.gradient_conflict"):
            _stage2_gradient_conflict_config({"stage1": {"epochs": 0}, "stage2": {"epochs": 1}})

    def test_stage2_gradient_conflict_config_rejects_invalid_interval(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        stages = {
            "stage1": {"epochs": 0},
            "stage2": {"epochs": 1, "gradient_conflict": {"enabled": True, "interval": 0}},
        }

        with self.assertRaisesRegex(ValueError, "interval"):
            _stage2_gradient_conflict_config(stages)

    def test_stage2_gradient_conflict_config_is_not_required_without_stage2(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        config = _stage2_gradient_conflict_config({"stage1": {"epochs": 1}, "stage2": {"epochs": 0}})

        self.assertFalse(config.enabled)

    def test_stage2_gradient_conflict_metrics_compute_cosine_and_norms(self) -> None:
        import torch

        from safa.training.g_loop import _compute_gradient_conflict_metrics

        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        flow_loss = parameter[0] * 2.0
        cycle_loss = parameter[1] * 3.0

        metrics = _compute_gradient_conflict_metrics(flow_loss, cycle_loss, [parameter])

        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle"], 0.0, places=6)
        self.assertAlmostEqual(metrics["gradient_norm_fm"], 2.0, places=6)
        self.assertAlmostEqual(metrics["gradient_norm_cycle"], 3.0, places=6)

    def test_stage2_gradient_conflict_metrics_reject_zero_norm_gradient(self) -> None:
        import torch

        from safa.training.g_loop import _compute_gradient_conflict_metrics

        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        flow_loss = parameter[0] * 0.0
        cycle_loss = parameter[1] * 3.0

        with self.assertRaisesRegex(RuntimeError, "zero norm"):
            _compute_gradient_conflict_metrics(flow_loss, cycle_loss, [parameter])

    def test_checkpoint_composite_uses_single_face_eq1_rate_not_legacy_ge1(self) -> None:
        from safa.training.g_loop import _composite_score

        ge1_high_single_low = {
            "validation_latent_cosine_mean": 0.90,
            "validation_face_detection_rate": 1.00,
            "validation_single_face_eq1_rate": 0.10,
        }
        ge1_low_single_high = {
            "validation_latent_cosine_mean": 0.80,
            "validation_face_detection_rate": 0.20,
            "validation_single_face_eq1_rate": 0.90,
        }

        self.assertAlmostEqual(_composite_score(ge1_high_single_low), 0.09)
        self.assertAlmostEqual(_composite_score(ge1_low_single_high), 0.72)
        self.assertGreater(_composite_score(ge1_low_single_high), _composite_score(ge1_high_single_low))

    def test_checkpoint_composite_requires_single_face_eq1_rate(self) -> None:
        from safa.training.g_loop import _composite_score

        with self.assertRaisesRegex(KeyError, "validation_single_face_eq1_rate"):
            _composite_score({"validation_latent_cosine_mean": 0.90, "validation_face_detection_rate": 1.00})

    def test_epoch_metrics_include_gradient_conflict_when_recorded(self) -> None:
        import torch

        from safa.training.g_loop import _reduce_epoch_metrics
        from safa.utils.distributed import DistributedContext

        totals = {
            "loss": 8.0,
            "flow_matching_mse": 4.0,
            "cycle": 2.0,
            "grad_norm": 0.0,
            "gradient_conflict_count": 2.0,
            "gradient_cosine_fm_cycle": -0.5,
            "gradient_norm_fm": 4.0,
            "gradient_norm_cycle": 6.0,
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

        metrics = _reduce_epoch_metrics(totals, seen=4, device=torch.device("cpu"), distributed=distributed)

        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle"], -0.25)
        self.assertAlmostEqual(metrics["gradient_norm_fm"], 2.0)
        self.assertAlmostEqual(metrics["gradient_norm_cycle"], 3.0)


if __name__ == "__main__":
    unittest.main()
