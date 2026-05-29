from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_medium_v2_stage2_configs_use_explicit_paths_batches_and_objectives() -> None:
    from safa.training import g_loop

    expected = {
        "train_g_medium_v2_stage2_m2_gram_weighted.yaml": "gram_weighted_sum",
        "train_g_medium_v2_stage2_m3_gram_projected.yaml": "gram_projected_two_step",
    }
    for filename, objective_type in expected.items():
        path = REPO_ROOT / "configs" / "medium_v2" / filename
        assert path.is_file(), filename
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["train_features"] == "artifacts/e0_features/train_balanced_medium_e0_medium_v1"
        assert config["validation"]["features"] == "artifacts/e0_features/val_single_face_e0_medium_v1"
        assert config["e0_checkpoint"] == "artifacts/checkpoints/e0_medium_v1/best.pt"
        assert config["resume_from"] == "artifacts/checkpoints/g_medium_v1_stage1_long200_v4/best_stage1.pt"
        assert config["global_batch_size"] == 96
        assert config["per_device_batch_size"] == 24
        assert "batch_size" not in config
        assert config["stages"]["stage2"]["epochs"] == 120
        assert config["stages"]["stage2"]["gradient_monitor"] == {"enabled": True, "interval": 20, "max_samples": 8}
        quality_eval = config["stages"]["stage2"]["quality_eval"]
        assert quality_eval["niqe_interval_epochs"] == 1
        assert quality_eval["distribution_interval_epochs"] == 20
        assert config["generator"]["train_cycle_steps"] == 16
        assert config["generator"]["cycle_steps_schedule"] == []
        assert config["stages"]["stage2"]["stage2_objective"]["type"] == objective_type

        g_loop._validate_train_g_config(config)


def test_generator_training_step_gram_weighted_sum_outputs_repr_metrics() -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training.g_loop import _GeneratorTrainingStep, _stage2_objective_from_config

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.tensor([0.2, -0.1]))

        def flow_matching_loss(self, images, z):
            loss = self.offset.pow(2).sum() + images.sum() * 0.0 + z.sum() * 0.0
            return loss, {"flow_matching_mse": loss.detach()}

        def sample(self, z, **kwargs):
            embedding = torch.nn.functional.normalize(z + self.offset.unsqueeze(0), dim=1)
            pad = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
            image = torch.cat([embedding, pad], dim=1).reshape(z.shape[0], 3, 1, 1)
            return image.expand(z.shape[0], 3, 4, 4)

    class DummyE0(nn.Module):
        def forward(self, images):
            embedding = torch.nn.functional.normalize(images[:, :2, 0, 0], dim=1)
            return {"embedding": embedding}

    objective = _stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "gram_weighted_sum",
                    "lambda_repr": 0.5,
                    "point_weight": 1.0,
                    "relation_weight": 2.0,
                    "offdiag_only": True,
                },
            },
        }
    )
    module = _GeneratorTrainingStep(
        DummyGenerator(),
        DummyE0(),
        FlowGeneratorConfig(embedding_dim=2, image_size=4, train_cycle_steps=1),
        1337,
        stage2_objective=objective,
    )
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss, _, repr_loss, flow_loss, _ = module(torch.zeros(2, 3, 4, 4), z, ["a", "b"], True, 0.0)

    metrics = module.last_loss_metrics
    assert metrics["stage2_objective_type"] == "gram_weighted_sum"
    assert metrics["repr_loss"] > 0.0
    assert metrics["repr_point_loss"] > 0.0
    assert metrics["repr_relation_loss"] > 0.0
    assert torch.allclose(repr_loss, torch.as_tensor(metrics["repr_loss"], dtype=repr_loss.dtype))
    assert torch.allclose(loss.detach(), flow_loss.detach() + 0.5 * repr_loss.detach())


def test_projected_repr_manual_step_uses_param_data_add_not_optimizer_step() -> None:
    from safa.training.g_loop import _apply_projected_repr_step

    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    _apply_projected_repr_step([param], [torch.tensor([0.25, -0.5])], repr_learning_rate=0.1)

    assert torch.allclose(param.detach(), torch.tensor([0.975, -1.95]))
    source = inspect.getsource(_apply_projected_repr_step)
    assert ".data.add_" in source
    assert "optimizer.step" not in source
    assert "AdamW" not in source
