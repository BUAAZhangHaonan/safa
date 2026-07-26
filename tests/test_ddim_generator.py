from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from safa.utils.hashing import sha256_file

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIVE_M_FM_PARAMETER_COUNT = 5_004_291


def _small_ddim_config(**overrides) -> dict:
    config = {
        "model_type": "ddim",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "ddim",
        "learned_null_condition": True,
        "ddim_num_train_timesteps": 1000,
        "ddim_beta_schedule": "linear",
        "ddim_beta_start": 0.0001,
        "ddim_beta_end": 0.02,
        "ddim_eta": 0.0,
    }
    config.update(overrides)
    return config


def test_build_generator_supports_ddim_and_preserves_model_type_roundtrip() -> None:
    from safa.models.generator import FlowGeneratorConfig, build_generator

    config = FlowGeneratorConfig.from_dict(_small_ddim_config())

    assert config.model_type == "ddim"
    assert config.to_dict()["model_type"] == "ddim"
    generator = build_generator(config.to_dict())
    assert generator.config.model_type == "ddim"
    assert generator.config.sample_steps == 1
    assert generator.config.sampler == "ddim"


def test_e10_ddim_config_is_200_epoch_one_step_gpu6_safe_and_larger_than_5m() -> None:
    from safa.models.generator import build_generator
    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "experiments" / "e10_ddim_200ep.yaml"
    assert path.is_file()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment_name"] == "e10_ddim_200ep"
    assert config["device"] == "cuda:0"
    assert config["global_batch_size"] == 4
    assert config["per_device_batch_size"] == 4
    assert config["amp"] is True
    assert config["out_dir"] == "artifacts/checkpoints/g_medium_v2_ddim_200ep"
    assert config["stages"]["stage2"]["epochs"] == 200
    assert config["generator"]["model_type"] == "ddim"
    assert config["generator"]["sample_steps"] == 1
    assert config["generator"]["train_cycle_steps"] == 1
    assert config["generator"]["sampler"] == "ddim"
    assert config["generator"]["learned_null_condition"] is True
    assert config["generator"]["ddim_num_train_timesteps"] == 1000
    assert config["generator"]["ddim_eta"] == 0.0
    assert config["stages"]["stage2"]["stage2_objective"]["flow_condition"] == "learned_null_condition"
    quality_eval = config["stages"]["stage2"]["quality_eval"]
    assert quality_eval["distribution_cuda_visible_devices"] == "6"
    assert quality_eval["distribution_device"] == "cuda:0"
    assert quality_eval["output_dir"] == "artifacts/eval/e10_ddim_200ep/quality"

    g_loop._validate_train_g_config(config)
    generator_config = dict(config["generator"])
    generator_config["embedding_dim"] = config["embedding_dim"]
    generator_config["image_size"] = config["image_size"]
    generator = build_generator(generator_config)
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    assert parameter_count > FIVE_M_FM_PARAMETER_COUNT


def test_ddim_denoising_loss_returns_scalar_metrics_and_backpropagates() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_small_ddim_config())
    images = torch.rand(2, 3, 16, 16)
    z = torch.randn(2, 16)

    loss, metrics = generator.denoising_loss(images, z)
    wrapper_loss, wrapper_metrics = generator.flow_matching_loss(images, z)
    loss.backward()

    assert tuple(loss.shape) == ()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["ddim_denoising_mse"])
    assert "flow_matching_mse" in wrapper_metrics
    assert torch.isfinite(wrapper_loss)
    finite_grads = [
        parameter.grad.detach()
        for parameter in generator.parameters()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    ]
    assert finite_grads


def test_ddim_sample_shape_and_one_step_timesteps() -> None:
    from safa.models.generator import build_generator

    torch.manual_seed(123)
    generator = build_generator(_small_ddim_config())
    z = torch.randn(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    default_steps = generator.sample(z, steps=None, x_init=x_init, clamp_output=False)
    one_step = generator.sample(z, steps=1, x_init=x_init, clamp_output=False)
    two_steps = generator.sample(z, steps=2, x_init=x_init, clamp_output=False)

    assert default_steps.shape == (2, 3, 16, 16)
    assert torch.allclose(default_steps, one_step)
    assert not torch.allclose(default_steps, two_steps)
    assert generator.ddim_timesteps(1).tolist() == [999]
    assert generator.ddim_timesteps(2).tolist() == [999, 0]


def test_ddim_checkpoint_roundtrip_and_existing_generators_still_load() -> None:
    from safa.evaluation.runner import _load_generator
    from safa.models.generator import ConditionalFlowGenerator, build_generator

    ddim = build_generator(_small_ddim_config())
    meanflow = build_generator(
        {
            "model_type": "meanflow",
            "embedding_dim": 16,
            "image_size": 16,
            "base_channels": 4,
            "channel_multipliers": [1],
            "time_embedding_dim": 8,
            "condition_dim": 16,
            "sample_steps": 1,
            "train_cycle_steps": 1,
            "sampler": "meanflow",
            "learned_null_condition": True,
            "meanflow_ratio": 0.75,
            "meanflow_adaptive_weighting": True,
            "meanflow_norm_p": 0.75,
            "meanflow_norm_eps": 0.001,
            "meanflow_jvp_mode": "torch_func",
        }
    )
    flow_config = {
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "euler",
    }
    flow = ConditionalFlowGenerator(flow_config)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = {
            "ddim": tmp_path / "ddim.pt",
            "meanflow": tmp_path / "meanflow.pt",
            "flow": tmp_path / "flow.pt",
        }
        torch.save({"model_state_dict": ddim.state_dict(), "model_config": ddim.config.to_dict(), "metrics": {}}, paths["ddim"])
        torch.save({"model_state_dict": meanflow.state_dict(), "model_config": meanflow.config.to_dict(), "metrics": {}}, paths["meanflow"])
        torch.save({"model_state_dict": flow.state_dict(), "model_config": flow.config.to_dict(), "metrics": {}}, paths["flow"])

        loaded_ddim = _load_generator(
            str(paths["ddim"]),
            {
                "checkpoint_model": "raw",
                "checkpoint_sha256": sha256_file(paths["ddim"]),
            },
            "cpu",
        )
        loaded_meanflow = _load_generator(
            str(paths["meanflow"]),
            {
                "checkpoint_model": "raw",
                "checkpoint_sha256": sha256_file(paths["meanflow"]),
            },
            "cpu",
        )
        loaded_flow = _load_generator(
            str(paths["flow"]),
            {
                "checkpoint_model": "raw",
                "checkpoint_sha256": sha256_file(paths["flow"]),
            },
            "cpu",
        )

    assert loaded_ddim.config.model_type == "ddim"
    assert loaded_meanflow.config.model_type == "meanflow"
    assert loaded_flow.config.model_type == "conditional_flow_matching"
    assert tuple(loaded_ddim(torch.randn(1, 16)).shape) == (1, 3, 16, 16)
    assert tuple(loaded_meanflow(torch.randn(1, 16)).shape) == (1, 3, 16, 16)
    assert tuple(loaded_flow(torch.randn(1, 16)).shape) == (1, 3, 16, 16)


def test_ddim_is_separate_from_flow_sampler_contract() -> None:
    from safa.models.generator import ConditionalFlowGenerator, build_generator

    flow_payload = {
        "model_type": "conditional_flow_matching",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "ddim",
    }

    with pytest.raises(ValueError, match="sampler must be euler or heun"):
        ConditionalFlowGenerator(flow_payload)
    with pytest.raises(ValueError, match="sampler must be euler or heun"):
        build_generator(flow_payload)
