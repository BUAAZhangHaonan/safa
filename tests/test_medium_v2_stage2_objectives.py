from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _small_generator_payload() -> dict:
    return {
        "embedding_dim": 8,
        "image_size": 8,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 8,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "euler",
    }


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
        "train_g_medium_v2_stage2_m3_point_descent_credit_projected_smoke10.yaml": {
            "objective": "point_descent_credit_projected",
            "global_batch_size": 96,
            "epochs": 10,
            "gradient_monitor": {"enabled": True, "interval": 20, "max_samples": 8},
            "flow_condition": "embedding",
        },
        "train_g_medium_v2_stage2_m3_fm_anchored_cagrad_smoke10.yaml": {
            "objective": "fm_anchored_cagrad",
            "global_batch_size": 48,
            "per_device_batch_size": 24,
            "epochs": 10,
            "gradient_monitor": {"enabled": True, "interval": 20, "max_samples": 8},
            "flow_condition": "embedding",
        },
        "train_g_medium_v2_stage2_m3_fm_primary_constrained_famo_smoke10.yaml": {
            "objective": "fm_primary_constrained_famo",
            "global_batch_size": 48,
            "per_device_batch_size": 24,
            "epochs": 10,
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
        assert config["per_device_batch_size"] == expected_config.get("per_device_batch_size", 24)
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
        if expected_config["objective"] in {"point_projected_two_step", "point_descent_credit_projected"}:
            assert "relation_weight" not in objective
            assert "offdiag_only" not in objective
            assert objective["point_weight"] == 1.0
            assert objective["repr_learning_rate"] > 0.0
            assert "grad_clip_norm" not in config
        if expected_config["objective"] == "fm_anchored_cagrad":
            for forbidden in ("relation_weight", "offdiag_only", "repr_learning_rate"):
                assert forbidden not in objective
            assert objective["lambda_repr"] == 1.0
            assert objective["point_weight"] == 1.0
            assert objective["cagrad_c"] == 0.5
            assert objective["fm_descent_floor_fraction"] == 0.1
            assert config["out_dir"] == "artifacts/checkpoints/g_medium_v2_stage2_m3_fm_anchored_cagrad_smoke10"
            assert (
                config["stages"]["stage2"]["quality_eval"]["output_dir"]
                == "artifacts/eval/g_medium_v2_stage2_m3_fm_anchored_cagrad_smoke10/quality"
            )
        if expected_config["objective"] == "fm_primary_constrained_famo":
            for forbidden in ("relation_weight", "offdiag_only", "repr_learning_rate", "projection_eps"):
                assert forbidden not in objective
            assert objective["lambda_repr"] == 1.0
            assert objective["point_weight"] == 1.0
            assert objective["cagrad_c"] == 0.5
            assert objective["fm_descent_floor_fraction"] == 0.2
            assert objective["famo_beta"] > 0.0
            assert objective["famo_gamma"] >= 0.0
            assert objective["famo_eps"] > 0.0
            assert "famo_min_loss_fm" not in objective
            assert "famo_min_loss_cl" not in objective
            assert config["out_dir"] == "artifacts/checkpoints/g_medium_v2_stage2_m3_fm_primary_constrained_famo_smoke10"
            assert (
                config["stages"]["stage2"]["quality_eval"]["output_dir"]
                == "artifacts/eval/g_medium_v2_stage2_m3_fm_primary_constrained_famo_smoke10/quality"
            )
        if expected_config["objective"] == "point_descent_credit_projected":
            assert config["out_dir"] == "artifacts/checkpoints/g_medium_v2_stage2_m3_point_descent_credit_projected_smoke10"
            assert (
                config["stages"]["stage2"]["quality_eval"]["output_dir"]
                == "artifacts/eval/g_medium_v2_stage2_m3_point_descent_credit_projected_smoke10/quality"
            )

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
        "train_g_medium_v2_stage2_frozen_fm_conditioning_only_probe_gpu0_bs24_weights_only_20260608.yaml": (
            "artifacts/checkpoints/g_medium_v2_stage2_frozen_fm_conditioning_only_probe_gpu0_bs24_weights_only_20260608",
            "artifacts/eval/g_medium_v2_stage2_frozen_fm_conditioning_only_probe_gpu0_bs24_weights_only_20260608/quality",
            "point_projected_two_step",
        ),
    }
    for filename, (out_dir, eval_dir, objective_type) in expected.items():
        config = yaml.safe_load((REPO_ROOT / "configs" / "medium_v2" / filename).read_text(encoding="utf-8"))
        assert config["out_dir"] == out_dir
        assert config["stages"]["stage2"]["quality_eval"]["output_dir"] == eval_dir
        assert "distribution_cuda_visible_devices" not in config["stages"]["stage2"]["quality_eval"]
        assert config["stages"]["stage2"]["stage2_objective"]["type"] == objective_type
        if "weights_only" in filename:
            assert config["resume_mode"] == "model_weights_only"
            assert config["device"] == "cuda:0"
            assert config["generator_trainable"] == "conditioning_only"


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


def test_point_projected_sample_condition_defaults_to_flow_condition_and_accepts_override() -> None:
    from safa.training import g_loop

    stages = {
        "stage1": {"epochs": 0},
        "stage2": {
            "epochs": 1,
            "stage2_objective": {
                "type": "point_projected_two_step",
                "flow_condition": "fixed_null_condition",
                "lambda_repr": 0.5,
                "point_weight": 1.0,
                "repr_learning_rate": 3.0e-5,
                "projection_eps": 1.0e-12,
            },
        },
    }

    objective = g_loop._stage2_objective_from_config(stages)
    assert objective.sample_condition == "fixed_null_condition"

    stages["stage2"]["stage2_objective"]["sample_condition"] = "embedding"
    objective = g_loop._stage2_objective_from_config(stages)
    assert objective.sample_condition == "embedding"

    stages["stage2"]["stage2_objective"]["sample_condition"] = "unknown"
    with pytest.raises(ValueError, match="sample_condition"):
        g_loop._stage2_objective_from_config(stages)


def test_independent_prior_m2m_requires_null_fm_embedding_sampling_and_zero_lpips() -> None:
    import copy

    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "experiments" / "r6_m2m_full_pu_l05_lr5e5_gpu2.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["many_to_many"]["semantics"] = "independent_prior"
    objective = config["stages"]["stage2"]["stage2_objective"]
    objective["flow_condition"] = "learned_null_condition"
    objective["sample_condition"] = "embedding"
    objective["lambda_lpips"] = 0.0

    g_loop._validate_train_g_config(config)

    invalid = []
    bad_flow = copy.deepcopy(config)
    bad_flow["stages"]["stage2"]["stage2_objective"]["flow_condition"] = "embedding"
    invalid.append((bad_flow, "flow_condition"))
    bad_sample = copy.deepcopy(config)
    bad_sample["stages"]["stage2"]["stage2_objective"]["sample_condition"] = "fixed_null_condition"
    invalid.append((bad_sample, "sample_condition"))
    missing_sample = copy.deepcopy(config)
    del missing_sample["stages"]["stage2"]["stage2_objective"]["sample_condition"]
    invalid.append((missing_sample, "sample_condition"))
    bad_lpips = copy.deepcopy(config)
    bad_lpips["stages"]["stage2"]["stage2_objective"]["lambda_lpips"] = 0.1
    invalid.append((bad_lpips, "lambda_lpips"))

    for bad_config, field in invalid:
        with pytest.raises(ValueError, match=field):
            g_loop._validate_train_g_config(bad_config)


def test_generator_trainable_defaults_to_full() -> None:
    from safa.models.generator import ConditionalFlowGenerator
    from safa.training import g_loop

    generator = ConditionalFlowGenerator(_small_generator_payload())

    mode = g_loop._generator_trainable_mode({})
    g_loop._apply_generator_trainable_mode(generator, mode)

    assert mode == "full"
    assert all(param.requires_grad for param in generator.parameters())


def test_generator_trainable_conditioning_only_keeps_only_z_mlp_and_film_condition() -> None:
    from safa.models.generator import ConditionalFlowGenerator
    from safa.training import g_loop

    generator = ConditionalFlowGenerator(_small_generator_payload())
    mode = g_loop._generator_trainable_mode({"generator_trainable": "conditioning_only"})

    g_loop._apply_generator_trainable_mode(generator, mode)

    trainable_names = {name for name, param in generator.named_parameters() if param.requires_grad}
    expected_trainable_names = {
        "vector_field.z_mlp.0.weight",
        "vector_field.z_mlp.0.bias",
        "vector_field.z_mlp.2.weight",
        "vector_field.z_mlp.2.bias",
        "vector_field.down_blocks.0.condition.weight",
        "vector_field.down_blocks.0.condition.bias",
        "vector_field.mid.condition.weight",
        "vector_field.mid.condition.bias",
        "vector_field.up_blocks.0.condition.weight",
        "vector_field.up_blocks.0.condition.bias",
    }
    assert trainable_names == expected_trainable_names

    parameters_by_name = dict(generator.named_parameters())
    expected_frozen_names = {
        "vector_field.input.weight",
        "vector_field.input.bias",
        "vector_field.time_mlp.0.weight",
        "vector_field.time_mlp.0.bias",
        "vector_field.down_blocks.0.in_conv.weight",
        "vector_field.down_blocks.0.out_conv.bias",
        "vector_field.mid.in_norm.weight",
        "vector_field.mid.in_conv.weight",
        "vector_field.up_blocks.0.skip.weight",
        "vector_field.upsamplers.0.bias",
        "vector_field.output.2.weight",
    }
    assert expected_frozen_names <= set(parameters_by_name)
    for name in expected_frozen_names:
        assert not parameters_by_name[name].requires_grad, name


def test_optimizer_param_groups_filter_frozen_generator_params() -> None:
    from torch import nn

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trainable = nn.Parameter(torch.tensor(1.0))
            self.frozen = nn.Parameter(torch.tensor(2.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.uncertainty_loss = None

    module = DummyTrainingModule()
    module.generator.frozen.requires_grad_(False)
    groups = g_loop._optimizer_param_groups(
        module,
        {"learning_rate": 0.01, "weight_decay": 0.0},
        g_loop._LossWeightingRuntime(type="legacy"),
    )

    generator_params = list(groups[0]["params"])
    assert [id(param) for param in generator_params] == [id(module.generator.trainable)]
    assert all(param.requires_grad for group in groups for param in group["params"])

    module.generator.trainable.requires_grad_(False)
    with pytest.raises(RuntimeError, match="trainable generator"):
        g_loop._optimizer_param_groups(
            module,
            {"learning_rate": 0.01, "weight_decay": 0.0},
            g_loop._LossWeightingRuntime(type="legacy"),
        )


def test_optimizer_resume_skips_full_state_for_conditioning_only_param_group_mismatch(capsys) -> None:
    from safa.models.generator import ConditionalFlowGenerator
    from safa.training import g_loop

    generator = ConditionalFlowGenerator(_small_generator_payload())
    full_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.001)
    full_optimizer_state = full_optimizer.state_dict()

    g_loop._apply_generator_trainable_mode(generator, "conditioning_only")
    conditioning_params = [param for param in generator.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(conditioning_params, lr=0.001)

    resumed = g_loop._load_resume_optimizer_state(
        optimizer,
        full_optimizer_state,
        generator_trainable_mode="conditioning_only",
        is_main=True,
    )

    assert resumed is False
    assert "optimizer_resumed: false" in capsys.readouterr().out
    assert len(optimizer.param_groups[0]["params"]) == len(conditioning_params)
    assert len(optimizer.param_groups[0]["params"]) < len(full_optimizer_state["param_groups"][0]["params"])
    assert optimizer.state_dict()["state"] == {}


def test_optimizer_resume_full_mode_keeps_pytorch_param_group_mismatch_error() -> None:
    from torch import nn

    from safa.training import g_loop

    source_model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    full_optimizer_state = torch.optim.AdamW(source_model.parameters(), lr=0.001).state_dict()
    optimizer = torch.optim.AdamW(source_model[0].parameters(), lr=0.001)

    with pytest.raises(ValueError, match="parameter group"):
        g_loop._load_resume_optimizer_state(
            optimizer,
            full_optimizer_state,
            generator_trainable_mode="full",
            is_main=False,
        )


def test_optimizer_resume_conditioning_only_does_not_swallow_malformed_state() -> None:
    from torch import nn

    from safa.training import g_loop

    optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=0.001)

    with pytest.raises(KeyError):
        g_loop._load_resume_optimizer_state(
            optimizer,
            {"state": {}},
            generator_trainable_mode="conditioning_only",
            is_main=False,
        )


def test_train_g_applies_generator_trainable_after_resume_before_ddp_and_optimizer_groups(monkeypatch, tmp_path) -> None:
    from torch import nn

    from safa.training import g_loop
    from safa.utils.distributed import DistributedContext

    events: list[str] = []
    resume_path = tmp_path / "resume.pt"
    resume_path.write_bytes(b"checkpoint")

    class StopAtOptimizerGroups(RuntimeError):
        pass

    class FakeGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def load_state_dict(self, state_dict, strict=True):
            del strict
            del state_dict
            events.append("load_resume")
            return [], []

    class FakeTrainingModule(nn.Module):
        def __init__(self, generator, *args, **kwargs) -> None:
            super().__init__()
            del args, kwargs
            self.generator = generator
            self.uncertainty_loss = None

    generator = FakeGenerator()
    generator_config = g_loop.FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1, train_cycle_steps=1)
    distributed = DistributedContext(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        is_main=True,
        device=torch.device("cpu"),
        backend="gloo",
    )

    monkeypatch.setattr(g_loop, "set_seed", lambda seed: None)
    monkeypatch.setattr(g_loop, "audit_no_identity_supervision", lambda config, paths: None)
    monkeypatch.setattr(g_loop, "_validate_train_g_config", lambda config: None)
    monkeypatch.setattr(g_loop, "init_distributed", lambda config: distributed)
    monkeypatch.setattr(g_loop, "barrier", lambda distributed: None)
    monkeypatch.setattr(g_loop, "load_e0_checkpoint", lambda path, device: (nn.Identity(), {}))
    monkeypatch.setattr(g_loop, "freeze_e0", lambda e0: None)
    monkeypatch.setattr(g_loop, "_generator_config_from_train_config", lambda config: generator_config)
    monkeypatch.setattr(g_loop, "_stage_config", lambda config: {"stage1": {"epochs": 0}, "stage2": {"epochs": 0}})
    monkeypatch.setattr(
        g_loop,
        "_ema_config",
        lambda config: {
            "enabled": False,
            "decay": 0.999,
            "evaluate_raw": True,
            "evaluate_ema": False,
            "save_ema_checkpoint": False,
        },
    )
    monkeypatch.setattr(g_loop, "_best_model", lambda config, ema_config: "raw")
    monkeypatch.setattr(g_loop, "_loss_weighting_runtime_from_config", lambda config: g_loop._LossWeightingRuntime(type="legacy"))
    monkeypatch.setattr(g_loop, "_stage2_objective_from_config", lambda stages: None)
    monkeypatch.setattr(g_loop, "sampling_base_seed_from_config", lambda config: 123)
    monkeypatch.setattr(g_loop, "build_generator", lambda payload: generator)
    monkeypatch.setattr(torch, "load", lambda path, map_location, weights_only: {"model_state_dict": {}, "metrics": {}})
    monkeypatch.setattr(g_loop, "_resume_stage_progress_from_metrics", lambda metrics, path: g_loop._ResumeProgress("stage1", 0))

    def fake_apply_generator_trainable_mode(actual_generator, mode):
        assert actual_generator is generator
        assert mode == "conditioning_only"
        events.append("apply_generator_trainable")

    monkeypatch.setattr(g_loop, "_apply_generator_trainable_mode", fake_apply_generator_trainable_mode)
    monkeypatch.setattr(g_loop, "_GeneratorTrainingStep", FakeTrainingModule)
    monkeypatch.setattr(g_loop, "_verify_e0_feature_cache_consistency", lambda config: None)
    monkeypatch.setattr(g_loop, "FeatureAlignedAffectNet", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(g_loop, "_build_train_loader", lambda train_set, *, train_sampler, batch_config, num_workers: [])
    monkeypatch.setattr(g_loop, "_build_validation_loader", lambda config: None)
    monkeypatch.setattr(g_loop, "_build_detector", lambda config, device: None)
    monkeypatch.setattr(g_loop, "_stage2_gradient_conflict_config", lambda stages: g_loop._GradientConflictConfig(enabled=False))

    class FakeDDP:
        def __init__(self, module, *, device_ids=None, output_device=None, **kwargs) -> None:
            del device_ids, output_device, kwargs
            events.append("ddp_wrap")
            self.module = module

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    def fake_optimizer_param_groups(training_module, config, loss_weighting):
        del config, loss_weighting
        assert training_module.generator is generator
        events.append("optimizer_param_groups")
        raise StopAtOptimizerGroups

    monkeypatch.setattr(g_loop, "_optimizer_param_groups", fake_optimizer_param_groups)

    config = {
        "seed": 1,
        "out_dir": str(tmp_path / "out"),
        "num_workers": 1,
        "e0_checkpoint": "e0.pt",
        "train_index": "train.csv",
        "train_features": "features",
        "image_size": 4,
        "global_batch_size": 2,
        "per_device_batch_size": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "resume_from": str(resume_path),
        "generator_trainable": "conditioning_only",
    }

    with pytest.raises(StopAtOptimizerGroups):
        g_loop.train_g_from_config(config)

    assert events == ["load_resume", "apply_generator_trainable", "ddp_wrap", "optimizer_param_groups"]




def test_train_g_model_weights_only_resume_skips_training_state_and_initializes_ema_from_loaded_model(monkeypatch, tmp_path) -> None:
    from torch import nn

    from safa.training import g_loop
    from safa.utils.distributed import DistributedContext

    events: list[str] = []
    resume_path = tmp_path / "resume.pt"
    resume_path.write_bytes(b"checkpoint")

    class StopAfterOptimizerResume(RuntimeError):
        pass

    class FakeGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def load_state_dict(self, state_dict, strict=True):
            del strict
            with torch.no_grad():
                self.weight.copy_(state_dict["weight"])
            events.append(f"load_resume:{float(self.weight.detach())}")
            return [], []

    class FakeEMA:
        def __init__(self, model, decay: float) -> None:
            del decay
            events.append(f"ema_init:{float(model.weight.detach())}")

        def load_state_dict(self, state_dict) -> None:
            del state_dict
            events.append("ema_load_old")

    class FakeTrainingModule(nn.Module):
        def __init__(self, generator, *args, **kwargs) -> None:
            super().__init__()
            del args, kwargs
            self.generator = generator
            self.uncertainty_loss = nn.Linear(1, 1)

    generator = FakeGenerator()
    generator_config = g_loop.FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1, train_cycle_steps=1)
    distributed = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_main=True,
        device=torch.device("cpu"),
        backend="gloo",
    )

    monkeypatch.setattr(g_loop, "set_seed", lambda seed: None)
    monkeypatch.setattr(g_loop, "audit_no_identity_supervision", lambda config, paths: None)
    monkeypatch.setattr(g_loop, "_validate_train_g_config", lambda config: None)
    monkeypatch.setattr(g_loop, "init_distributed", lambda config: distributed)
    monkeypatch.setattr(g_loop, "cleanup_distributed", lambda distributed: None)
    monkeypatch.setattr(g_loop, "barrier", lambda distributed: None)
    monkeypatch.setattr(g_loop, "load_e0_checkpoint", lambda path, device: (nn.Identity(), {}))
    monkeypatch.setattr(g_loop, "freeze_e0", lambda e0: None)
    monkeypatch.setattr(g_loop, "_generator_config_from_train_config", lambda config: generator_config)
    monkeypatch.setattr(g_loop, "_stage_config", lambda config: {"stage1": {"epochs": 0}, "stage2": {"epochs": 3}})
    monkeypatch.setattr(
        g_loop,
        "_ema_config",
        lambda config: {
            "enabled": True,
            "decay": 0.999,
            "evaluate_raw": True,
            "evaluate_ema": True,
            "save_ema_checkpoint": True,
        },
    )
    monkeypatch.setattr(g_loop, "ExponentialMovingAverage", FakeEMA)
    monkeypatch.setattr(g_loop, "_best_model", lambda config, ema_config: "raw")
    monkeypatch.setattr(
        g_loop,
        "_loss_weighting_runtime_from_config",
        lambda config: g_loop._LossWeightingRuntime(
            type="uncertainty",
            calibration_batches=1,
            log_var_lr=0.001,
            log_var_weight_decay=0.0,
        ),
    )
    monkeypatch.setattr(g_loop, "_stage2_objective_from_config", lambda stages: None)
    monkeypatch.setattr(g_loop, "sampling_base_seed_from_config", lambda config: 123)
    monkeypatch.setattr(g_loop, "build_generator", lambda payload: generator)
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, map_location, weights_only: {
            "model_state_dict": {"weight": torch.tensor(7.0)},
            "history": [{"stage": "stage2"}],
            "metrics": {"stage": "stage2", "stage_epoch": 2},
            "ema_model_state_dict": {"weight": torch.tensor(3.0)},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "loss_weighting_state": {"type": "uncertainty"},
        },
    )
    monkeypatch.setattr(
        g_loop,
        "_resume_history_for_checkpoint_selection",
        lambda history, path, config, stages: events.append("resume_history") or list(history),
    )
    monkeypatch.setattr(
        g_loop,
        "_resume_stage_progress_from_metrics",
        lambda metrics, path: events.append("resume_progress") or g_loop._ResumeProgress("stage2", 2),
    )

    def fake_apply_generator_trainable_mode(actual_generator, mode):
        assert actual_generator is generator
        assert mode == "conditioning_only"
        events.append("apply_generator_trainable")

    monkeypatch.setattr(g_loop, "_apply_generator_trainable_mode", fake_apply_generator_trainable_mode)
    monkeypatch.setattr(g_loop, "_GeneratorTrainingStep", FakeTrainingModule)
    monkeypatch.setattr(g_loop, "_verify_e0_feature_cache_consistency", lambda config: None)
    monkeypatch.setattr(g_loop, "FeatureAlignedAffectNet", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(g_loop, "_build_train_loader", lambda train_set, *, train_sampler, batch_config, num_workers: [])
    monkeypatch.setattr(g_loop, "_build_validation_loader", lambda config: None)
    monkeypatch.setattr(g_loop, "_build_detector", lambda config, device: None)
    monkeypatch.setattr(g_loop, "_stage2_gradient_conflict_config", lambda stages: g_loop._GradientConflictConfig(enabled=False))

    def fake_restore_or_calibrate_uncertainty_loss(
        training_module,
        train_loader,
        device,
        *,
        use_amp,
        calibration_batches,
        distributed,
        resume_progress,
        resume_loss_weighting_state,
        resume_path,
    ):
        del training_module, train_loader, device, use_amp, calibration_batches, distributed, resume_progress, resume_path
        state = "None" if resume_loss_weighting_state is None else "present"
        events.append(f"loss_weighting_state:{state}")
        return "calibrated"

    monkeypatch.setattr(g_loop, "_restore_or_calibrate_uncertainty_loss", fake_restore_or_calibrate_uncertainty_loss)

    def fake_optimizer_param_groups(training_module, config, loss_weighting):
        del config, loss_weighting
        assert training_module.generator is generator
        events.append("optimizer_param_groups")
        return [{"params": [training_module.generator.weight], "lr": 0.001, "weight_decay": 0.0}]

    monkeypatch.setattr(g_loop, "_optimizer_param_groups", fake_optimizer_param_groups)
    monkeypatch.setattr(
        g_loop,
        "_assert_required_resume_optimizer_state",
        lambda config, stages, resume_progress, resume_optimizer_state_dict, resume_path: events.append("assert_optimizer_state"),
    )
    monkeypatch.setattr(
        g_loop,
        "_load_resume_optimizer_state",
        lambda optimizer, resume_optimizer_state_dict, *, generator_trainable_mode, is_main: events.append("load_optimizer_state") or True,
    )

    def fake_assert_e0_frozen(e0, optimizer):
        del e0, optimizer
        events.append("assert_e0_frozen")
        raise StopAfterOptimizerResume

    monkeypatch.setattr(g_loop, "assert_e0_frozen", fake_assert_e0_frozen)

    config = {
        "seed": 1,
        "out_dir": str(tmp_path / "out"),
        "num_workers": 1,
        "e0_checkpoint": "e0.pt",
        "train_index": "train.csv",
        "train_features": "features",
        "image_size": 4,
        "global_batch_size": 1,
        "per_device_batch_size": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "resume_from": str(resume_path),
        "resume_mode": "model_weights_only",
        "generator_trainable": "conditioning_only",
    }

    with pytest.raises(StopAfterOptimizerResume):
        g_loop.train_g_from_config(config)

    assert events == [
        "load_resume:7.0",
        "apply_generator_trainable",
        "ema_init:7.0",
        "loss_weighting_state:None",
        "optimizer_param_groups",
        "assert_e0_frozen",
    ]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "training_state"),
        ({"resume_mode": "training_state"}, "training_state"),
        ({"resume_mode": "model_weights_only"}, "model_weights_only"),
    ],
)
def test_resume_mode_defaults_to_training_state_and_accepts_known_values(config, expected) -> None:
    from safa.training import g_loop

    assert g_loop._resume_mode(config) == expected


def test_validate_train_g_config_rejects_bad_resume_mode() -> None:
    from safa.training import g_loop

    with pytest.raises(ValueError, match="resume_mode.*bad"):
        g_loop._validate_train_g_config({"resume_mode": "bad"})

def test_generator_trainable_rejects_unknown_value() -> None:
    from safa.training import g_loop

    with pytest.raises(ValueError, match="generator_trainable"):
        g_loop._generator_trainable_mode({"generator_trainable": "adapter"})


def test_validate_train_g_config_rejects_bad_generator_trainable() -> None:
    from safa.training import g_loop

    with pytest.raises(ValueError, match="generator_trainable.*bad"):
        g_loop._validate_train_g_config({"generator_trainable": "bad"})


@pytest.mark.parametrize(
    ("train_config", "expected_mode"),
    [
        ({"generator_trainable": "conditioning_only"}, "conditioning_only"),
        ({}, "full"),
    ],
)
def test_save_generator_records_generator_trainable_explicit_and_default(tmp_path, train_config, expected_mode) -> None:
    from torch import nn

    from safa.training import g_loop

    generator = nn.Linear(2, 2)
    generator_config = g_loop.FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1, train_cycle_steps=1)
    checkpoint_path = tmp_path / "generator.pt"
    metrics = {
        "stage": "stage2",
        "loss": 1.0,
        "validation_latent_cosine_mean": 0.9,
        "validation_single_face_eq1_rate": 0.8,
    }
    ema_config = {
        "enabled": False,
        "decay": 0.999,
        "evaluate_raw": True,
        "evaluate_ema": False,
        "save_ema_checkpoint": False,
    }

    g_loop._save_generator(
        checkpoint_path,
        generator,
        generator_config,
        {"stages": {}, "validation": {}, **train_config},
        metrics,
        [],
        ema_config=ema_config,
        best_model="raw",
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["training_config"]["generator_trainable"] == expected_mode


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

        def flow_matching_loss(self, images, z, generator=None):
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

        def flow_matching_loss(self, images, z, generator=None):
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

        def flow_matching_loss(self, images, z, generator=None):
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


def test_generator_training_step_fm_only_probe_uses_configured_learned_null_condition() -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.null_condition = nn.Parameter(torch.tensor([7.0, -5.0]))
            self.flow_condition_z = None
            self.flow_condition_requires_grad = None

        def make_null_condition(self, *, batch_size: int, device, dtype):
            return self.null_condition.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

        def flow_matching_loss(self, images, z, generator=None):
            self.flow_condition_z = z.detach().clone()
            self.flow_condition_requires_grad = z.requires_grad
            loss = self.null_condition.square().sum() + images.sum() * 0.0 + z.sum() * 0.0
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
                    "flow_condition": "learned_null_condition",
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

    loss, _, _, flow_loss, _ = module(torch.zeros(2, 3, 4, 4), torch.eye(2), ["a", "b"], True, 0.0)

    assert torch.allclose(loss, flow_loss)
    assert torch.equal(generator.flow_condition_z, torch.tensor([[7.0, -5.0], [7.0, -5.0]]))
    assert generator.flow_condition_requires_grad is True
    assert module.last_loss_metrics["flow_condition"] == "learned_null_condition"


def test_point_projected_repr_sampling_uses_sample_condition_not_flow_condition() -> None:
    from torch import nn
    import torch.nn.functional as F

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.flow_condition_z = None
            self.sample_condition_z = None

        def make_null_condition(self, *, batch_size: int, device, dtype):
            return torch.tensor([0.0, 1.0], device=device, dtype=dtype).expand(batch_size, -1)

        def flow_matching_loss(self, images, z, generator=None):
            del generator
            self.flow_condition_z = z.detach().clone()
            loss = self.weight.square() + images.sum() * 0.0 + z.sum() * 0.0
            return loss, {"flow_matching_mse": loss.detach()}

        def sample(self, z, **kwargs):
            del kwargs
            self.sample_condition_z = z.detach().clone()
            pad = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
            image = torch.cat([z, pad], dim=1).reshape(z.shape[0], 3, 1, 1)
            return image.expand(z.shape[0], 3, 4, 4)

    class DummyE0(nn.Module):
        def forward(self, images):
            return {"embedding": F.normalize(images[:, :2, 0, 0], dim=1)}

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "point_projected_two_step",
                    "flow_condition": "embedding",
                    "sample_condition": "learned_null_condition",
                    "lambda_repr": 0.5,
                    "point_weight": 1.0,
                    "repr_learning_rate": 3.0e-5,
                    "projection_eps": 1.0e-12,
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
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    module(torch.zeros(2, 3, 4, 4), z, ["a", "b"], True, 0.0)

    assert torch.equal(generator.flow_condition_z, z)
    assert torch.equal(generator.sample_condition_z, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))


def test_validation_sampling_uses_configured_learned_null_condition() -> None:
    from torch import nn
    import torch.nn.functional as F

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.null_condition = nn.Parameter(torch.tensor([7.0, -5.0]))
            self.sample_condition_z = None

        def make_null_condition(self, *, batch_size: int, device, dtype):
            return self.null_condition.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

        def sample(self, z, **kwargs):
            del kwargs
            self.sample_condition_z = z.detach().clone()
            return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

    class DummyE0(nn.Module):
        def forward(self, images):
            batch_size = images.shape[0]
            base = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                device=images.device,
                dtype=images.dtype,
            )
            embedding = F.normalize(base[:batch_size], dim=1)
            logits = torch.zeros(batch_size, 2, device=images.device, dtype=images.dtype)
            return {"embedding": embedding, "logits": logits}

    generator = DummyGenerator()
    loader = [
        {
            "image": torch.zeros(3, 3, 4, 4),
            "z": F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=1),
            "sample_id": ["a", "b", "c"],
        }
    ]

    metrics = g_loop._evaluate_validation(
        generator,
        DummyE0(),
        loader,
        None,
        torch.device("cpu"),
        FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1),
        sampling_seed=1337,
        flow_condition="learned_null_condition",
    )

    assert torch.equal(generator.sample_condition_z, torch.tensor([[7.0, -5.0], [7.0, -5.0], [7.0, -5.0]]))
    assert metrics["latent_cosine_mean"] == pytest.approx(1.0)


def test_quality_eval_sampling_uses_configured_learned_null_condition(tmp_path) -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.null_condition = nn.Parameter(torch.tensor([7.0, -5.0]))
            self.sample_condition_z = None

        def make_null_condition(self, *, batch_size: int, device, dtype):
            return self.null_condition.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

        def sample(self, z, **kwargs):
            del kwargs
            self.sample_condition_z = z.detach().clone()
            return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

    generator = DummyGenerator()
    loader = [{"z": torch.eye(2), "sample_id": ["a", "b"]}]
    generated_dir = tmp_path / "generated"

    count = g_loop._generate_quality_eval_images(
        generator=generator,
        loader=loader,
        generated_dir=generated_dir,
        device=torch.device("cpu"),
        generator_config=FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1),
        sampling_seed=1337,
        max_samples=2,
        use_amp=False,
        flow_condition="learned_null_condition",
    )

    assert count == 2
    assert torch.equal(generator.sample_condition_z, torch.tensor([[7.0, -5.0], [7.0, -5.0]]))
    assert len(list(generated_dir.glob("*.png"))) == 2


def test_quality_eval_releases_cuda_cache_before_distribution_subprocess(tmp_path, monkeypatch) -> None:
    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    real_index = tmp_path / "real.jsonl"
    config = {
        "validation": {"enabled": True},
        "stages": {
            "stage2": {
                "quality_eval": {
                    "enabled": True,
                    "metrics": ["fid"],
                    "distribution_interval_epochs": 1,
                    "distribution_max_samples": 1,
                    "distribution_timeout_seconds": 30,
                    "real_index": str(real_index),
                    "distribution_device": "cuda:0",
                    "output_dir": str(tmp_path / "quality"),
                    "model": "ema",
                }
            }
        },
    }
    calls: list[str] = []

    monkeypatch.setattr(g_loop, "_build_quality_eval_loader", lambda *args, **kwargs: [{"z": torch.zeros(1, 2), "sample_id": ["a"]}])
    monkeypatch.setattr(g_loop, "_quality_eval_current_generator", lambda *args, **kwargs: object())
    monkeypatch.setattr(g_loop, "_generate_quality_eval_images", lambda **kwargs: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    def fake_distribution_eval(**kwargs):
        calls.append("subprocess")
        assert "empty_cache" in calls
        return {"metrics": ["fid"], "num_generated": 1, "num_real": 1, "fid": 1.0}

    monkeypatch.setattr(g_loop, "_evaluate_generation_quality_subprocess", fake_distribution_eval)

    metrics = g_loop._run_quality_eval_hook(
        config,
        "stage2",
        0,
        generator=object(),
        ema=object(),
        device=torch.device("cuda:0"),
        generator_config=FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1),
        sampling_seed=1337,
    )

    assert calls == ["empty_cache", "subprocess"]
    assert metrics["quality_ema_fid"] == pytest.approx(1.0)


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


def test_stage2_sample_condition_is_separate_from_fm_condition() -> None:
    from safa.training import g_loop

    stages = {
        "stage1": {"epochs": 0},
        "stage2": {
            "epochs": 1,
            "stage2_objective": {
                "type": "point_projected_two_step",
                "flow_condition": "learned_null_condition",
                "sample_condition": "embedding",
                "lambda_repr": 0.5,
                "point_weight": 1.0,
                "repr_learning_rate": 3.0e-5,
                "projection_eps": 1.0e-12,
            },
        },
    }
    objective = g_loop._stage2_objective_from_config(stages)

    assert g_loop._flow_condition_for_stage(stages, "stage2", objective) == "learned_null_condition"
    assert g_loop._sample_condition_for_stage(stages, "stage2", objective) == "embedding"


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


def test_gram_projected_objective_parses_repr_step_ratio_cap() -> None:
    from safa.training import g_loop

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "gram_projected_two_step",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "relation_weight": 1.0,
                    "offdiag_only": True,
                    "repr_learning_rate": 0.00003,
                    "projection_eps": 1e-12,
                    "repr_step_ratio_cap": 0.125,
                },
            },
        }
    )

    assert objective.repr_step_ratio_cap == pytest.approx(0.125)


def test_e1_pu_sgd_config_declares_repr_step_ratio_cap() -> None:
    with (REPO_ROOT / "configs/medium_v2/experiments/e1_pu_sgd_200ep.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    objective = config["stages"]["stage2"]["stage2_objective"]

    assert objective["optimizer_type"] == "sgd"
    assert objective["repr_step_ratio_cap"] == pytest.approx(0.25)


@pytest.mark.parametrize("objective_type", ["point_projected_two_step", "gram_projected_two_step"])
@pytest.mark.parametrize("bad_cap", [-0.1, float("nan")])
def test_projected_objective_rejects_invalid_repr_step_ratio_cap(objective_type: str, bad_cap: float) -> None:
    from safa.training import g_loop

    payload = {
        "type": objective_type,
        "flow_condition": "embedding",
        "lambda_repr": 1.0,
        "point_weight": 1.0,
        "repr_learning_rate": 0.00003,
        "projection_eps": 1e-12,
        "repr_step_ratio_cap": bad_cap,
    }
    if objective_type == "gram_projected_two_step":
        payload.update({"relation_weight": 1.0, "offdiag_only": True})

    with pytest.raises(ValueError, match="repr_step_ratio_cap"):
        g_loop._stage2_objective_from_config(
            {"stage1": {"epochs": 0}, "stage2": {"epochs": 1, "stage2_objective": payload}}
        )


@pytest.mark.parametrize("objective_type", ["point_projected_two_step", "point_descent_credit_projected"])
def test_point_projected_objective_requires_point_only_contract(objective_type: str) -> None:
    from safa.training import g_loop

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": objective_type,
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "repr_learning_rate": 0.00003,
                    "projection_eps": 1e-12,
                },
            },
        }
    )

    assert objective.type == objective_type
    assert objective.relation_weight == 0.0
    assert objective.offdiag_only is False

    for forbidden in ("relation_weight", "offdiag_only"):
        payload = {
            "type": objective_type,
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


def test_fm_anchored_cagrad_objective_requires_point_only_unit_lambda_contract() -> None:
    from safa.training import g_loop

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "fm_anchored_cagrad",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "cagrad_c": 0.5,
                    "fm_descent_floor_fraction": 0.1,
                    "projection_eps": 1e-12,
                },
            },
        }
    )

    assert objective.type == "fm_anchored_cagrad"
    assert objective.lambda_repr == 1.0
    assert objective.point_weight == 1.0
    assert objective.relation_weight == 0.0
    assert objective.offdiag_only is False
    assert objective.cagrad_c == 0.5
    assert objective.fm_descent_floor_fraction == 0.1

    invalid_payloads = [
        ({"lambda_repr": 0.5}, "lambda_repr"),
        ({"relation_weight": 0.0}, "relation_weight"),
        ({"offdiag_only": False}, "offdiag_only"),
        ({"repr_learning_rate": 0.00003}, "repr_learning_rate"),
        ({"cagrad_c": 1.0}, "cagrad_c"),
        ({"fm_descent_floor_fraction": 1.5}, "fm_descent_floor_fraction"),
    ]
    for override, match in invalid_payloads:
        payload = {
            "type": "fm_anchored_cagrad",
            "flow_condition": "embedding",
            "lambda_repr": 1.0,
            "point_weight": 1.0,
            "cagrad_c": 0.5,
            "fm_descent_floor_fraction": 0.1,
            "projection_eps": 1e-12,
            **override,
        }
        with pytest.raises(ValueError, match=match):
            g_loop._stage2_objective_from_config(
                {"stage1": {"epochs": 0}, "stage2": {"epochs": 1, "stage2_objective": payload}}
            )


def test_fm_primary_constrained_famo_objective_requires_point_only_unit_lambda_contract() -> None:
    from safa.training import g_loop

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "fm_primary_constrained_famo",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "cagrad_c": 0.5,
                    "fm_descent_floor_fraction": 0.2,
                    "famo_beta": 0.05,
                    "famo_gamma": 0.01,
                    "famo_eps": 1e-8,
                },
            },
        }
    )

    assert objective.type == "fm_primary_constrained_famo"
    assert objective.lambda_repr == 1.0
    assert objective.point_weight == 1.0
    assert objective.relation_weight == 0.0
    assert objective.offdiag_only is False
    assert objective.cagrad_c == 0.5
    assert objective.fm_descent_floor_fraction == 0.2
    assert objective.famo_beta == 0.05
    assert objective.famo_gamma == 0.01
    assert objective.famo_eps == 1e-8
    assert objective.famo_min_loss_fm == 0.0
    assert objective.famo_min_loss_cl == 0.0

    invalid_payloads = [
        ({"lambda_repr": 0.5}, "lambda_repr"),
        ({"relation_weight": 0.0}, "relation_weight"),
        ({"offdiag_only": False}, "offdiag_only"),
        ({"repr_learning_rate": 0.00003}, "repr_learning_rate"),
        ({"projection_eps": 1e-12}, "projection_eps"),
        ({"cagrad_c": 1.0}, "cagrad_c"),
        ({"fm_descent_floor_fraction": 1.5}, "fm_descent_floor_fraction"),
        ({"famo_beta": -0.1}, "famo_beta"),
        ({"famo_gamma": -0.1}, "famo_gamma"),
        ({"famo_eps": 0.0}, "famo_eps"),
    ]
    for override, match in invalid_payloads:
        payload = {
            "type": "fm_primary_constrained_famo",
            "flow_condition": "embedding",
            "lambda_repr": 1.0,
            "point_weight": 1.0,
            "cagrad_c": 0.5,
            "fm_descent_floor_fraction": 0.2,
            "famo_beta": 0.05,
            "famo_gamma": 0.01,
            "famo_eps": 1e-8,
            **override,
        }
        with pytest.raises(ValueError, match=match):
            g_loop._stage2_objective_from_config(
                {"stage1": {"epochs": 0}, "stage2": {"epochs": 1, "stage2_objective": payload}}
            )


def test_generator_training_step_gram_repr_only_probe_does_not_compute_flow_loss(monkeypatch) -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def flow_matching_loss(self, images, z, generator=None):
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


@pytest.mark.parametrize("objective_type", ["point_projected_two_step", "point_descent_credit_projected"])
def test_generator_training_step_point_projected_uses_point_loss_without_gram(monkeypatch, objective_type: str) -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.25))

        def flow_matching_loss(self, images, z, generator=None):
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
                    "type": objective_type,
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
    assert module.last_loss_metrics["stage2_objective_type"] == objective_type
    assert module.last_loss_metrics["repr_relation_loss"] == 0.0


def test_sgd_projected_batch_caps_repr_parameter_step_and_logs_euclidean_metrics() -> None:
    from contextlib import nullcontext
    from torch import nn

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.last_loss_metrics = {}

        def forward(self, images, z, sample_ids, include_repr, lambda_cycle, flow_condition, noise_generator=None):
            del images, z, sample_ids, lambda_cycle, flow_condition, noise_generator
            if include_repr:
                repr_loss = 100.0 * self.generator.weight
                zero = repr_loss.new_tensor(0.0)
                self.last_loss_metrics = {
                    "repr_loss": float(repr_loss.detach().cpu()),
                    "repr_point_loss": float(repr_loss.detach().cpu()),
                    "repr_relation_loss": 0.0,
                    "stage2_objective_type": "point_projected_two_step",
                    "lambda_repr": 1.0,
                    "effective_flow_loss_weight": 1.0,
                    "effective_repr_loss_weight": 1.0,
                    "effective_cycle_loss_weight": 0.0,
                    "flow_condition": "embedding",
                }
                return repr_loss, zero, repr_loss.detach(), zero, repr_loss
            flow_loss = 0.5 * self.generator.weight.square()
            zero = flow_loss.new_tensor(0.0)
            return flow_loss, flow_loss.detach(), zero, flow_loss, zero

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
                    "repr_learning_rate": 1.0,
                    "projection_eps": 1e-12,
                    "optimizer_type": "sgd",
                    "pu_gradient_normalization": True,
                    "repr_step_ratio_cap": 0.25,
                },
            },
        }
    )
    module = DummyTrainingModule()
    optimizer = torch.optim.SGD(module.generator.parameters(), lr=0.1)

    g_loop._run_projected_stage2_batch(
        training_module=module,
        optimizer=optimizer,
        images=torch.zeros(1, 3, 4, 4),
        z=torch.zeros(1, 2),
        sample_ids=["sample"],
        lambda_cycle=0.0,
        amp_ctx=nullcontext(),
        grad_clip_norm=None,
        ema=None,
        stage2_objective=objective,
        flow_condition="embedding",
    )

    metrics = module.last_loss_metrics
    assert metrics["fm_param_step_norm"] == pytest.approx(0.1, abs=1e-6)
    assert metrics["repr_param_step_norm_before_clip"] > metrics["repr_param_step_norm_after_clip"]
    assert metrics["repr_param_step_norm_after_clip"] == pytest.approx(0.025, abs=1e-6)
    assert metrics["repr_to_fm_param_step_ratio"] == pytest.approx(0.25, abs=1e-6)
    assert metrics["pu_norm_ratio_euclidean"] == pytest.approx(metrics["pu_norm_ratio"])
    assert "repr_grad_euclidean_norm_before_proj" in metrics
    assert "repr_grad_euclidean_norm_after_proj" in metrics
    assert "repr_grad_qnorm_before_proj" not in metrics
    assert "repr_grad_qnorm_after_proj" not in metrics
    assert module.generator.weight.detach().item() == pytest.approx(0.875, abs=1e-6)
    source = inspect.getsource(g_loop._run_projected_stage2_batch)
    assert "un_scale" not in source
    assert "grad_clip_norm or 1.0" not in source


def test_sgd_projected_batch_does_not_apply_default_unit_gradient_clip() -> None:
    from contextlib import nullcontext
    from torch import nn

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.last_loss_metrics = {}

        def forward(self, images, z, sample_ids, include_repr, lambda_cycle, flow_condition, noise_generator=None):
            del images, z, sample_ids, lambda_cycle, flow_condition, noise_generator
            if include_repr:
                repr_loss = 5.0 * self.generator.weight
                zero = repr_loss.new_tensor(0.0)
                self.last_loss_metrics = {
                    "repr_loss": float(repr_loss.detach().cpu()),
                    "repr_point_loss": float(repr_loss.detach().cpu()),
                    "repr_relation_loss": 0.0,
                    "stage2_objective_type": "point_projected_two_step",
                }
                return repr_loss, zero, repr_loss.detach(), zero, repr_loss
            flow_loss = 0.1 * self.generator.weight
            return flow_loss, flow_loss.detach(), flow_loss.detach(), flow_loss, flow_loss.detach()

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
                    "repr_learning_rate": 1.0,
                    "projection_eps": 1e-12,
                    "optimizer_type": "sgd",
                    "repr_step_ratio_cap": 1000.0,
                },
            },
        }
    )
    module = DummyTrainingModule()
    optimizer = torch.optim.SGD(module.generator.parameters(), lr=0.1)

    g_loop._run_projected_stage2_batch(
        training_module=module,
        optimizer=optimizer,
        images=torch.zeros(1, 3, 4, 4),
        z=torch.zeros(1, 2),
        sample_ids=["sample"],
        lambda_cycle=0.0,
        amp_ctx=nullcontext(),
        grad_clip_norm=None,
        ema=None,
        stage2_objective=objective,
        flow_condition="embedding",
    )

    assert module.generator.weight.detach().item() == pytest.approx(-4.01, abs=1e-6)
    assert module.last_loss_metrics["repr_param_step_norm_before_clip"] == pytest.approx(5.0, abs=1e-6)


def test_projected_batch_backtracking_preserves_repr_metrics_snapshot() -> None:
    from contextlib import nullcontext
    from torch import nn

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.last_loss_metrics = {}

        def forward(self, images, z, sample_ids, include_repr, lambda_cycle, flow_condition, noise_generator=None):
            del images, z, sample_ids, lambda_cycle, flow_condition, noise_generator
            if include_repr:
                repr_loss = 2.0 * self.generator.weight
                zero = repr_loss.new_tensor(0.0)
                self.last_loss_metrics = {
                    "repr_loss": float(repr_loss.detach().cpu()),
                    "repr_point_loss": float(repr_loss.detach().cpu()),
                    "repr_relation_loss": 0.0,
                    "stage2_objective_type": "point_projected_two_step",
                    "flow_condition": "embedding",
                }
                return repr_loss, zero, repr_loss.detach(), zero, repr_loss
            flow_loss = 0.5 * self.generator.weight.square()
            zero = flow_loss.new_tensor(0.0)
            self.last_loss_metrics = {
                "guard_metric": 1.0,
                "repr_loss": -999.0,
                "stage2_objective_type": "flow_guard",
            }
            return flow_loss, flow_loss.detach(), zero, flow_loss, zero

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
                    "repr_learning_rate": 0.1,
                    "projection_eps": 1e-12,
                    "optimizer_type": "sgd",
                    "pu_backtrack_max_retries": 1,
                    "pu_fm_increase_budget": 999.0,
                    "repr_step_ratio_cap": 100.0,
                },
            },
        }
    )
    module = DummyTrainingModule()
    optimizer = torch.optim.SGD(module.generator.parameters(), lr=0.1)

    _, _, _, _, _, _, metrics = g_loop._run_projected_stage2_batch(
        training_module=module,
        optimizer=optimizer,
        images=torch.zeros(1, 3, 4, 4),
        z=torch.zeros(1, 2),
        sample_ids=["sample"],
        lambda_cycle=0.0,
        amp_ctx=nullcontext(),
        grad_clip_norm=None,
        ema=None,
        stage2_objective=objective,
        flow_condition="embedding",
    )

    assert metrics["repr_loss"] == pytest.approx(1.8, abs=1e-6)
    assert metrics["repr_point_loss"] == pytest.approx(1.8, abs=1e-6)
    assert "guard_metric" not in metrics
    assert metrics["stage2_objective_type"] == "point_projected_two_step"
    assert metrics["pu_backtrack_count"] == 0.0


def test_descent_credit_projected_batch_records_credit_budget_metrics() -> None:
    from contextlib import nullcontext
    from torch import nn

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.last_loss_metrics = {}

        def forward(self, images, z, sample_ids, include_repr, lambda_cycle, flow_condition, noise_generator=None):
            del images, z, sample_ids, lambda_cycle, flow_condition
            if include_repr:
                repr_loss = -self.generator.weight
                zero = repr_loss.new_tensor(0.0)
                self.last_loss_metrics = {
                    "repr_loss": float(repr_loss.detach().cpu()),
                    "repr_point_loss": float(repr_loss.detach().cpu()),
                    "repr_relation_loss": 0.0,
                    "stage2_objective_type": "point_descent_credit_projected",
                    "lambda_repr": 1.0,
                    "effective_flow_loss_weight": 1.0,
                    "effective_repr_loss_weight": 1.0,
                    "effective_cycle_loss_weight": 0.0,
                    "flow_condition": "embedding",
                }
                return repr_loss, zero, repr_loss.detach(), zero, repr_loss
            flow_loss = self.generator.weight.square()
            zero = flow_loss.new_tensor(0.0)
            return flow_loss, flow_loss.detach(), zero, flow_loss, zero

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "point_descent_credit_projected",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "repr_learning_rate": 1.0,
                    "projection_eps": 1e-12,
                    "repr_step_ratio_cap": 100.0,
                },
            },
        }
    )
    module = DummyTrainingModule()
    optimizer = torch.optim.AdamW(module.generator.parameters(), lr=0.1, weight_decay=0.0)

    _, _, _, flow_loss_guard, _, _, metrics = g_loop._run_projected_stage2_batch(
        training_module=module,
        optimizer=optimizer,
        images=torch.zeros(1, 3, 4, 4),
        z=torch.zeros(1, 2),
        sample_ids=["sample"],
        lambda_cycle=0.0,
        amp_ctx=nullcontext(),
        grad_clip_norm=None,
        ema=None,
        stage2_objective=objective,
        flow_condition="embedding",
    )

    assert metrics["stage2_objective_type"] == "point_descent_credit_projected"
    assert metrics["pre_flow_loss_before_fm_step"] == pytest.approx(1.0)
    assert metrics["flow_loss_guard"] == pytest.approx(float(flow_loss_guard.detach().cpu()))
    assert metrics["fm_descent_credit"] == pytest.approx(metrics["pre_flow_loss_before_fm_step"] - metrics["flow_loss_guard"])
    assert metrics["credit_dot_lower_bound"] == pytest.approx(-metrics["fm_descent_credit"])
    assert metrics["dot_after"] == pytest.approx(metrics["credit_dot_lower_bound"])
    assert metrics["credit_budget_used_fraction"] == pytest.approx(1.0)
    assert metrics["projection_applied_fraction"] == 1.0
    assert "repr_grad_qnorm_before_proj" in metrics
    assert "repr_grad_qnorm_after_proj" in metrics
    assert "pu_norm_ratio_qweighted" in metrics
    assert "pu_norm_ratio_euclidean" not in metrics
    assert "repr_grad_euclidean_norm_before_proj" not in metrics
    assert "repr_grad_euclidean_norm_after_proj" not in metrics


def test_fm_anchored_cagrad_batch_uses_single_optimizer_step_and_logs_metrics() -> None:
    from contextlib import nullcontext
    from torch import nn

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.last_loss_metrics = {}

        def forward(self, images, z, sample_ids, include_repr, lambda_cycle, flow_condition, noise_generator=None):
            del images, z, sample_ids, lambda_cycle, flow_condition
            flow_loss = self.generator.weight.square()
            flow_mse = flow_loss.detach()
            if include_repr:
                repr_loss = -0.5 * self.generator.weight
                self.last_loss_metrics = {
                    "flow_loss_raw": float(flow_loss.detach().cpu()),
                    "cycle_loss_raw": 0.0,
                    "repr_loss": float(repr_loss.detach().cpu()),
                    "repr_point_loss": float(repr_loss.detach().cpu()),
                    "repr_relation_loss": 0.0,
                    "stage2_objective_type": "fm_anchored_cagrad",
                    "lambda_repr": 1.0,
                    "effective_flow_loss_weight": 1.0,
                    "effective_repr_loss_weight": 1.0,
                    "effective_cycle_loss_weight": 0.0,
                    "flow_condition": "embedding",
                }
                return repr_loss, flow_mse, repr_loss.detach(), flow_loss, repr_loss
            return flow_loss, flow_mse, flow_loss.new_tensor(0.0), flow_loss, flow_loss.new_tensor(0.0)

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params) -> None:
            super().__init__(params, lr=0.1)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "fm_anchored_cagrad",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "cagrad_c": 0.5,
                    "fm_descent_floor_fraction": 0.1,
                    "projection_eps": 1e-12,
                },
            },
        }
    )
    module = DummyTrainingModule()
    optimizer = CountingSGD(module.generator.parameters())

    _, _, _, _, _, batch_grad_norm, metrics = g_loop._run_fm_anchored_cagrad_stage2_batch(
        training_module=module,
        optimizer=optimizer,
        images=torch.zeros(1, 3, 4, 4),
        z=torch.zeros(1, 2),
        sample_ids=["sample"],
        lambda_cycle=0.0,
        amp_ctx=nullcontext(),
        grad_clip_norm=None,
        ema=None,
        stage2_objective=objective,
        flow_condition="embedding",
    )

    assert optimizer.step_calls == 1
    assert batch_grad_norm == pytest.approx(metrics["combined_grad_norm"])
    for key in (
        "cagrad_fm_weight",
        "cagrad_cl_weight",
        "cagrad_raw_fm_weight",
        "cagrad_raw_cl_weight",
        "fm_descent_floor_fraction",
        "fm_descent_floor_value",
        "gradient_cosine_fm_repr",
        "combined_grad_norm",
        "fm_descent_after_cagrad",
        "fm_descent_after_anchor",
        "fm_anchor_active",
    ):
        assert key in metrics
    assert metrics["cagrad_fm_weight"] + metrics["cagrad_cl_weight"] == pytest.approx(1.0)
    assert metrics["fm_descent_after_anchor"] + 1e-6 >= metrics["fm_descent_floor_value"]

    source = inspect.getsource(g_loop._run_fm_anchored_cagrad_stage2_batch)
    assert "optimizer.step()" in source
    assert source.count("optimizer.step()") == 1
    assert "torch.autograd.grad" not in source
    assert "_gradient_vector_for_loss" not in source


def test_fm_primary_constrained_famo_batch_logs_weights_floor_and_gate_metrics() -> None:
    from contextlib import nullcontext
    from torch import nn

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.last_loss_metrics = {}

        def forward(self, images, z, sample_ids, include_repr, lambda_cycle, flow_condition, noise_generator=None):
            del images, z, sample_ids, lambda_cycle, flow_condition
            flow_loss = self.generator.weight.square()
            flow_mse = flow_loss.detach()
            if include_repr:
                repr_loss = 7.0 - 6.0 * self.generator.weight
                self.last_loss_metrics = {
                    "flow_loss_raw": float(flow_loss.detach().cpu()),
                    "cycle_loss_raw": 0.0,
                    "repr_loss": float(repr_loss.detach().cpu()),
                    "repr_point_loss": float(repr_loss.detach().cpu()),
                    "repr_relation_loss": 0.0,
                    "stage2_objective_type": "fm_primary_constrained_famo",
                    "lambda_repr": 1.0,
                    "effective_flow_loss_weight": 1.0,
                    "effective_repr_loss_weight": 1.0,
                    "effective_cycle_loss_weight": 0.0,
                    "flow_condition": "embedding",
                }
                return repr_loss, flow_mse, repr_loss.detach(), flow_loss, repr_loss
            return flow_loss, flow_mse, flow_loss.new_tensor(0.0), flow_loss, flow_loss.new_tensor(0.0)

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params) -> None:
            super().__init__(params, lr=0.1)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    objective = g_loop._stage2_objective_from_config(
        {
            "stage1": {"epochs": 0},
            "stage2": {
                "epochs": 1,
                "stage2_objective": {
                    "type": "fm_primary_constrained_famo",
                    "flow_condition": "embedding",
                    "lambda_repr": 1.0,
                    "point_weight": 1.0,
                    "cagrad_c": 0.5,
                    "fm_descent_floor_fraction": 0.2,
                    "famo_beta": 0.05,
                    "famo_gamma": 0.01,
                    "famo_eps": 1e-8,
                },
            },
        }
    )
    module = DummyTrainingModule()
    optimizer = CountingSGD(module.generator.parameters())

    _, _, _, _, _, batch_grad_norm, metrics = g_loop._run_fm_primary_constrained_famo_stage2_batch(
        training_module=module,
        optimizer=optimizer,
        images=torch.zeros(1, 3, 4, 4),
        z=torch.zeros(1, 2),
        sample_ids=["sample"],
        lambda_cycle=0.0,
        amp_ctx=nullcontext(),
        grad_clip_norm=None,
        ema=None,
        stage2_objective=objective,
        flow_condition="embedding",
    )

    assert optimizer.step_calls == 1
    assert batch_grad_norm == pytest.approx(metrics["combined_grad_norm"])
    for key in (
        "famo_weight_fm",
        "famo_weight_cl",
        "famo_logit_fm",
        "famo_logit_cl",
        "famo_delta_log_loss_fm",
        "famo_delta_log_loss_cl",
        "fm_floor_ratio",
        "fm_floor_active",
        "cl_gate_scale",
        "gradient_cosine_fm_cl",
        "combined_grad_norm",
    ):
        assert key in metrics
    assert metrics["stage2_objective_type"] == "fm_primary_constrained_famo"
    assert metrics["famo_weight_fm"] + metrics["famo_weight_cl"] == pytest.approx(1.0)
    assert metrics["gradient_cosine_fm_cl"] == pytest.approx(-1.0)
    assert metrics["fm_floor_ratio"] + 1e-6 >= objective.fm_descent_floor_fraction
    assert metrics["fm_floor_active"] == 1.0
    assert 0.0 <= metrics["cl_gate_scale"] < 1.0

    source = inspect.getsource(g_loop._run_fm_primary_constrained_famo_stage2_batch)
    assert "optimizer.step()" in source
    assert source.count("optimizer.step()") == 1
    assert "torch.autograd.grad" not in source
    assert "_gradient_vector_for_loss" not in source


def test_generator_training_step_prefers_spec_repr_metric_fields(monkeypatch) -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def flow_matching_loss(self, images, z, generator=None):
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


def test_projected_stage2_rejects_nonfinite_fm_gradients_without_clip_before_optimizer_step() -> None:
    from contextlib import nullcontext
    from torch import nn

    from safa.training import g_loop

    class FiniteLossWithNonfiniteGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.new_tensor(1.0)

        @staticmethod
        def backward(ctx, grad_output):
            (value,) = ctx.saved_tensors
            return torch.full_like(value, float("inf")) * grad_output

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    class DummyTrainingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = DummyGenerator()
            self.last_loss_metrics = {"flow_loss_raw": 1.0}

        def forward(self, images, z, sample_ids, include_repr, lambda_cycle, flow_condition, noise_generator=None):
            del images, z, sample_ids, include_repr, lambda_cycle, flow_condition
            loss = FiniteLossWithNonfiniteGrad.apply(self.generator.weight)
            finite = loss.detach()
            return loss, finite, finite, finite, finite

    class OptimizerMustNotStep:
        def __init__(self, parameters) -> None:
            self.parameters = list(parameters)
            self.step_calls = 0

        def zero_grad(self, set_to_none=True):
            for parameter in self.parameters:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad.zero_()

        def step(self):
            self.step_calls += 1
            raise AssertionError("optimizer.step must not run after non-finite gradients")

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
    module = DummyTrainingModule()
    optimizer = OptimizerMustNotStep(module.generator.parameters())

    with pytest.raises(RuntimeError, match="M3 flow matching gradient 0 contains non-finite values"):
        g_loop._run_projected_stage2_batch(
            training_module=module,
            optimizer=optimizer,
            images=torch.zeros(1, 3, 4, 4),
            z=torch.zeros(1, 2),
            sample_ids=["sample"],
            lambda_cycle=0.0,
            amp_ctx=nullcontext(),
            grad_clip_norm=None,
            ema=None,
            stage2_objective=objective,
            flow_condition="embedding",
        )
    assert optimizer.step_calls == 0
