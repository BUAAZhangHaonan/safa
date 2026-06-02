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
        "train_g_medium_v2_stage2_m2_gram_weighted.yaml": {
            "objective": "gram_weighted_sum",
            "global_batch_size": 96,
            "epochs": 120,
            "gradient_monitor": {"enabled": True, "interval": 20, "max_samples": 8},
            "flow_condition": "embedding",
        },
        "train_g_medium_v2_stage2_m3_gram_projected.yaml": {
            "objective": "gram_projected_two_step",
            "global_batch_size": 96,
            "epochs": 120,
            "gradient_monitor": {"enabled": True, "interval": 20, "max_samples": 8},
            "flow_condition": "embedding",
        },
        "train_g_medium_v2_stage2_m3_point_projected.yaml": {
            "objective": "point_projected_two_step",
            "global_batch_size": 96,
            "epochs": 120,
            "gradient_monitor": {"enabled": True, "interval": 20, "max_samples": 8},
            "flow_condition": "embedding",
        },
        "train_g_medium_v2_stage2_fm_only_probe.yaml": {
            "objective": "fm_only_probe",
            "global_batch_size": 24,
            "epochs": 20,
            "gradient_monitor": {"enabled": False},
            "flow_condition": "embedding",
        },
        "train_g_medium_v2_stage2_null_fm.yaml": {
            "objective": "fm_only_probe",
            "global_batch_size": 24,
            "epochs": 120,
            "gradient_monitor": {"enabled": False},
            "flow_condition": "fixed_null_condition",
        },
        "train_g_medium_v2_stage2_point_only_cl_only.yaml": {
            "objective": "gram_repr_only_probe",
            "global_batch_size": 48,
            "epochs": 120,
            "gradient_monitor": {"enabled": True, "interval": 20, "max_samples": 8},
        },
        "train_g_medium_v2_stage2_point_gram_cl_only.yaml": {
            "objective": "gram_repr_only_probe",
            "global_batch_size": 48,
            "epochs": 120,
            "gradient_monitor": {"enabled": True, "interval": 20, "max_samples": 8},
        },
        "train_g_medium_v2_stage2_gram_only_probe.yaml": {
            "objective": "gram_repr_only_probe",
            "global_batch_size": 24,
            "epochs": 20,
            "gradient_monitor": {"enabled": False},
        },
    }
    for filename, expected_config in expected.items():
        path = REPO_ROOT / "configs" / "medium_v2" / filename
        assert path.is_file(), filename
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["train_features"] == "artifacts/e0_features/train_balanced_medium_e0_medium_v1"
        assert config["validation"]["features"] == "artifacts/e0_features/val_single_face_e0_medium_v1"
        assert config["e0_checkpoint"] == "artifacts/checkpoints/e0_medium_v1/best.pt"
        assert config["resume_from"] == "artifacts/checkpoints/g_medium_v1_stage1_long200_v4/best_stage1.pt"
        assert config["global_batch_size"] == expected_config["global_batch_size"]
        assert config["per_device_batch_size"] == 24
        assert "batch_size" not in config
        assert config["stages"]["stage2"]["epochs"] == expected_config["epochs"]
        assert config["stages"]["stage2"]["gradient_monitor"] == expected_config["gradient_monitor"]
        quality_eval = config["stages"]["stage2"]["quality_eval"]
        assert quality_eval["niqe_interval_epochs"] == 1
        assert quality_eval["distribution_interval_epochs"] == 20
        assert config["generator"]["train_cycle_steps"] == 16
        assert config["generator"]["cycle_steps_schedule"] == []
        objective = config["stages"]["stage2"]["stage2_objective"]
        assert objective["type"] == expected_config["objective"]
        if "flow_condition" in expected_config:
            assert objective["flow_condition"] == expected_config["flow_condition"]
        else:
            assert "flow_condition" not in objective
        if expected_config["objective"] == "point_projected_two_step":
            assert "relation_weight" not in objective
            assert "offdiag_only" not in objective
            assert objective["point_weight"] == 1.0
            assert objective["repr_learning_rate"] > 0.0
            assert "grad_clip_norm" not in config

        g_loop._validate_train_g_config(config)


def test_probe_configs_use_distinct_outputs_and_raw_quality_device() -> None:
    expected = {
        "train_g_medium_v2_stage2_fm_only_probe.yaml": (
            "artifacts/checkpoints/g_medium_v2_stage2_fm_only_probe",
            "artifacts/eval/g_medium_v2_stage2_fm_only_probe/quality",
            "fm_only_probe",
        ),
        "train_g_medium_v2_stage2_gram_only_probe.yaml": (
            "artifacts/checkpoints/g_medium_v2_stage2_gram_only_probe",
            "artifacts/eval/g_medium_v2_stage2_gram_only_probe/quality",
            "gram_repr_only_probe",
        ),
    }
    for filename, (out_dir, eval_dir, objective_type) in expected.items():
        config = yaml.safe_load((REPO_ROOT / "configs" / "medium_v2" / filename).read_text(encoding="utf-8"))
        assert config["out_dir"] == out_dir
        assert config["stages"]["stage2"]["quality_eval"]["output_dir"] == eval_dir
        assert "distribution_cuda_visible_devices" not in config["stages"]["stage2"]["quality_eval"]
        assert config["stages"]["stage2"]["stage2_objective"]["type"] == objective_type


def test_medium_v2_stage1_long1000_continue_config_extends_pure_fm_stage1() -> None:
    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "train_g_medium_v2_stage1_long1000_continue.yaml"
    assert path.is_file()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["out_dir"] == "artifacts/checkpoints/g_medium_v2_stage1_long1000_continue"
    assert config["resume_from"] == "artifacts/checkpoints/g_medium_v1_stage1_long200_v4/last.pt"
    assert config["global_batch_size"] == 32
    assert config["per_device_batch_size"] == 32
    assert "batch_size" not in config
    assert config["stages"]["stage1"]["epochs"] == 1000
    assert config["stages"]["stage1"]["flow_condition"] == "embedding"
    assert config["stages"]["stage1"]["stable_epochs"] == 1001
    assert config["stages"]["stage2"]["epochs"] == 0
    assert "stage2_objective" not in config["stages"]["stage2"]
    assert "gradient_monitor" not in config["stages"]["stage2"]
    assert "gradient_conflict" in config["stages"]["stage2"]
    quality_eval = config["stages"]["stage1"]["quality_eval"]
    assert quality_eval["output_dir"] == "artifacts/eval/g_medium_v2_stage1_long1000_continue/quality"
    assert quality_eval["niqe_interval_epochs"] == 1
    assert quality_eval["distribution_interval_epochs"] == 20
    assert quality_eval["metrics"] == ["niqe", "fid", "kid"]
    assert "distribution_cuda_visible_devices" not in quality_eval

    g_loop._validate_train_g_config(config)


def test_medium_v2_stage2_config_fails_fast_without_stage2_objective() -> None:
    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "train_g_medium_v2_stage2_m2_gram_weighted.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    del config["stages"]["stage2"]["stage2_objective"]

    with pytest.raises(ValueError, match="medium_v2.*stage2_objective"):
        g_loop._validate_train_g_config(config)


def test_flow_objective_requires_explicit_flow_condition() -> None:
    from safa.training import g_loop

    config = {
        "stage1": {"epochs": 0},
        "stage2": {"epochs": 1, "stage2_objective": {"type": "fm_only_probe"}},
    }

    with pytest.raises(ValueError, match="flow_condition"):
        g_loop._stage2_objective_from_config(config)


def test_gram_weighted_sum_config_validation_allows_relation_weight_two() -> None:
    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "train_g_medium_v2_stage2_m2_gram_weighted.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["stages"]["stage2"]["stage2_objective"]["relation_weight"] = 2.0

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
                    "flow_condition": "embedding",
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


def test_generator_training_step_fm_only_probe_does_not_sample_repr_or_cycle() -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def flow_matching_loss(self, images, z):
            loss = self.weight.square() + images.sum() * 0.0 + z.sum() * 0.0
            return loss, {"flow_matching_mse": loss.detach()}

        def sample(self, z, **kwargs):
            raise AssertionError("FM-only probe must not sample for repr or cycle loss")

    class DummyE0(nn.Module):
        def forward(self, images):
            raise AssertionError("FM-only probe must not call E0")

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {"epochs": 1, "stage2_objective": {"type": "fm_only_probe", "flow_condition": "embedding"}},
        }
    )
    module = g_loop._GeneratorTrainingStep(
        DummyGenerator(),
        DummyE0(),
        FlowGeneratorConfig(embedding_dim=2, image_size=4, train_cycle_steps=1),
        1337,
        stage2_objective=objective,
    )

    loss, _, secondary, flow_loss, secondary_raw = module(torch.zeros(2, 3, 4, 4), torch.eye(2), ["a", "b"], True, 0.0)

    assert torch.allclose(loss, flow_loss)
    assert torch.allclose(secondary, torch.tensor(0.0))
    assert torch.allclose(secondary_raw, torch.tensor(0.0))
    assert module.last_loss_metrics["stage2_objective_type"] == "fm_only_probe"
    assert module.last_loss_metrics["effective_repr_loss_weight"] == 0.0
    assert module.last_loss_metrics["effective_cycle_loss_weight"] == 0.0


def test_generator_training_step_fm_only_probe_uses_configured_fixed_null_condition() -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.flow_condition_z = None

        def flow_matching_loss(self, images, z):
            self.flow_condition_z = z.detach().clone()
            loss = self.weight.square() + images.sum() * 0.0 + z.sum() * 0.0
            return loss, {"flow_matching_mse": loss.detach()}

        def sample(self, z, **kwargs):
            raise AssertionError("FM-only probe must not sample for repr or cycle loss")

    class DummyE0(nn.Module):
        def forward(self, images):
            raise AssertionError("FM-only probe must not call E0")

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "fm_only_probe",
                    "flow_condition": "fixed_null_condition",
                },
            },
        }
    )
    generator = DummyGenerator()
    module = g_loop._GeneratorTrainingStep(
        generator,
        DummyE0(),
        FlowGeneratorConfig(embedding_dim=2, image_size=4, train_cycle_steps=1),
        1337,
        stage2_objective=objective,
    )
    z = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])

    loss, _, _, flow_loss, _ = module(torch.zeros(2, 3, 4, 4), z, ["a", "b"], True, 0.0)

    assert torch.allclose(loss, flow_loss)
    assert torch.equal(generator.flow_condition_z, torch.zeros_like(z))
    assert module.last_loss_metrics["flow_condition"] == "fixed_null_condition"


def test_stage1_flow_condition_config_supports_explicit_fixed_null_and_rejects_unknown() -> None:
    from safa.training import g_loop

    stages = {
        "stage1": {"epochs": 1, "flow_condition": "fixed_null_condition"},
        "stage2": {"epochs": 0},
    }

    assert g_loop._flow_condition_for_stage(stages, "stage1", None) == "fixed_null_condition"

    stages["stage1"]["flow_condition"] = "zero"
    with pytest.raises(ValueError, match="stages.stage1.flow_condition"):
        g_loop._flow_condition_for_stage(stages, "stage1", None)


def test_stage2_objective_accepts_only_named_repr_weight_modes() -> None:
    from safa.training import g_loop

    for relation_weight in (0.0, 1.0):
        objective = g_loop._stage2_objective_from_config(
            {
                "stage1": {"epochs": 0},
                "stage2": {
                    "epochs": 1,
                    "stage2_objective": {
                        "type": "gram_repr_only_probe",
                        "lambda_repr": 1.0,
                        "point_weight": 1.0,
                        "relation_weight": relation_weight,
                        "offdiag_only": True,
                    },
                },
            }
        )
        assert objective.point_weight == 1.0
        assert objective.relation_weight == relation_weight

    with pytest.raises(ValueError, match="point_weight=1.0.*relation_weight"):
        g_loop._stage2_objective_from_config(
            {
                "stage1": {"epochs": 0},
                "stage2": {
                    "epochs": 1,
                    "stage2_objective": {
                        "type": "gram_repr_only_probe",
                        "lambda_repr": 1.0,
                        "point_weight": 0.5,
                        "relation_weight": 0.0,
                        "offdiag_only": True,
                    },
                },
            }
        )


def test_point_projected_objective_requires_point_only_contract() -> None:
    from safa.training import g_loop

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "point_projected_two_step",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "repr_learning_rate": 0.00003,
                    "projection_eps": 1e-12,
                },
            },
        }
    )

    assert objective.type == "point_projected_two_step"
    assert objective.relation_weight == 0.0
    assert objective.offdiag_only is False

    for forbidden in ("relation_weight", "offdiag_only"):
        payload = {
            "type": "point_projected_two_step",
            "flow_condition": "embedding",
            "lambda_repr": 1.0,
            "point_weight": 1.0,
            "repr_learning_rate": 0.00003,
            "projection_eps": 1e-12,
            forbidden: 0.0,
        }
        with pytest.raises(ValueError, match=forbidden):
            g_loop._stage2_objective_from_config(
                {"stage1": {"epochs": 0}, "stage2": {"epochs": 1, "stage2_objective": payload}}
            )


def test_generator_training_step_gram_repr_only_probe_does_not_compute_flow_loss(monkeypatch) -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def flow_matching_loss(self, images, z):
            raise AssertionError("repr-only probe must not compute FM loss")

        def sample(self, z, **kwargs):
            pad = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
            image = torch.cat([z, pad], dim=1).reshape(z.shape[0], 3, 1, 1)
            return image.expand(z.shape[0], 3, 4, 4)

    class DummyE0(nn.Module):
        def forward(self, images):
            return {"embedding": torch.nn.functional.normalize(images[:, :2, 0, 0], dim=1)}

    def fake_hyperspherical_gram_loss(pred_embedding, target_embedding, point_weight, relation_weight, offdiag_only=True):
        del pred_embedding, target_embedding, point_weight, relation_weight, offdiag_only
        base = torch.tensor(1.25)
        return {"repr": base, "point": base * 0.2, "relation": base * 0.8}

    monkeypatch.setattr(g_loop, "hyperspherical_gram_loss", fake_hyperspherical_gram_loss)
    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "gram_repr_only_probe",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "relation_weight": 1.0,
                    "offdiag_only": True,
                },
            },
        }
    )
    module = g_loop._GeneratorTrainingStep(
        DummyGenerator(),
        DummyE0(),
        FlowGeneratorConfig(embedding_dim=2, image_size=4, train_cycle_steps=1),
        1337,
        stage2_objective=objective,
    )

    loss, flow_mse, repr_loss, flow_loss, raw_repr_loss = module(torch.zeros(2, 3, 4, 4), torch.eye(2), ["a", "b"], True, 0.0)

    assert torch.allclose(flow_mse, torch.tensor(0.0))
    assert torch.allclose(flow_loss, torch.tensor(0.0))
    assert torch.allclose(repr_loss, torch.tensor(1.25))
    assert torch.allclose(raw_repr_loss, torch.tensor(1.25))
    assert torch.allclose(loss.detach(), repr_loss.detach())
    assert module.last_loss_metrics["stage2_objective_type"] == "gram_repr_only_probe"
    assert module.last_loss_metrics["flow_loss_raw"] == 0.0
    assert module.last_loss_metrics["effective_flow_loss_weight"] == 0.0
    assert module.last_loss_metrics["effective_repr_loss_weight"] == 1.0


def test_generator_training_step_point_projected_uses_point_loss_without_gram(monkeypatch) -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.25))

        def flow_matching_loss(self, images, z):
            loss = self.weight.square() + images.sum() * 0.0 + z.sum() * 0.0
            return loss, {"flow_matching_mse": loss.detach()}

        def sample(self, z, **kwargs):
            pad = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
            image = torch.cat([z, pad], dim=1).reshape(z.shape[0], 3, 1, 1)
            return image.expand(z.shape[0], 3, 4, 4)

    class DummyE0(nn.Module):
        def forward(self, images):
            return {"embedding": torch.nn.functional.normalize(images[:, :2, 0, 0], dim=1)}

    def forbidden_gram_loss(*args, **kwargs):
        raise AssertionError("point-projected objective must not compute Gram loss")

    def fake_point_loss(pred_embedding, target_embedding, point_weight):
        del pred_embedding, target_embedding, point_weight
        base = torch.tensor(1.5)
        return {"repr": base, "point": base, "relation": base.new_tensor(0.0)}

    monkeypatch.setattr(g_loop, "hyperspherical_gram_loss", forbidden_gram_loss)
    monkeypatch.setattr(g_loop, "hyperspherical_point_cosine_loss", fake_point_loss, raising=False)
    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "point_projected_two_step",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "repr_learning_rate": 0.00003,
                    "projection_eps": 1e-12,
                },
            },
        }
    )
    module = g_loop._GeneratorTrainingStep(
        DummyGenerator(), DummyE0(), FlowGeneratorConfig(embedding_dim=2, image_size=4), 1337, stage2_objective=objective
    )

    loss, _, repr_loss, flow_loss, raw_repr_loss = module(torch.zeros(2, 3, 4, 4), torch.eye(2), ["a", "b"], True, 0.0)

    assert torch.allclose(repr_loss, torch.tensor(1.5))
    assert torch.allclose(raw_repr_loss, torch.tensor(1.5))
    assert torch.allclose(loss.detach(), repr_loss.detach())
    assert torch.allclose(flow_loss.detach(), torch.tensor(0.25).square())
    assert module.last_loss_metrics["stage2_objective_type"] == "point_projected_two_step"
    assert module.last_loss_metrics["repr_relation_loss"] == 0.0


def test_generator_training_step_prefers_spec_repr_metric_fields(monkeypatch) -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def flow_matching_loss(self, images, z):
            loss = z.sum() * 0.0 + torch.tensor(0.25, dtype=z.dtype, device=z.device)
            return loss, {"flow_matching_mse": loss.detach()}

        def sample(self, z, **kwargs):
            pad = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
            image = torch.cat([z, pad], dim=1).reshape(z.shape[0], 3, 1, 1)
            return image.expand(z.shape[0], 3, 4, 4)

    class DummyE0(nn.Module):
        def forward(self, images):
            return {"embedding": torch.nn.functional.normalize(images[:, :2, 0, 0], dim=1)}

    def fake_hyperspherical_gram_loss(pred_embedding, target_embedding, point_weight, relation_weight, offdiag_only=True):
        del pred_embedding, target_embedding, point_weight, relation_weight, offdiag_only
        base = torch.tensor(1.0)
        return {"repr": base, "point": base * 0.25, "relation": base * 0.75}

    monkeypatch.setattr(g_loop, "hyperspherical_gram_loss", fake_hyperspherical_gram_loss)
    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "gram_weighted_sum",
                    "flow_condition": "embedding",
                    "lambda_repr": 0.5,
                    "point_weight": 1.0,
                    "relation_weight": 1.0,
                    "offdiag_only": True,
                },
            },
        }
    )
    module = g_loop._GeneratorTrainingStep(
        DummyGenerator(),
        DummyE0(),
        FlowGeneratorConfig(embedding_dim=2, image_size=4, train_cycle_steps=1),
        1337,
        stage2_objective=objective,
    )

    loss, _, repr_loss, flow_loss, _ = module(torch.zeros(2, 3, 4, 4), torch.eye(2), ["a", "b"], True, 0.0)

    assert torch.allclose(repr_loss, torch.tensor(1.0))
    assert module.last_loss_metrics["repr_loss"] == 1.0
    assert module.last_loss_metrics["repr_point_loss"] == 0.25
    assert module.last_loss_metrics["repr_relation_loss"] == 0.75
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


def test_projected_stage2_fm_clip_fails_fast_on_nonfinite_gradients() -> None:
    from safa.training.g_loop import _run_projected_stage2_batch

    source = inspect.getsource(_run_projected_stage2_batch)
    assert "error_if_nonfinite=True" in source
