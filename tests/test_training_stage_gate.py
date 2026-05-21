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


if __name__ == "__main__":
    unittest.main()
