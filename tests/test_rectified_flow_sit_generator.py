from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _tiny_rectified_flow_sit_config() -> dict:
    return {
        "model_type": "rectified_flow_sit",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 4,
        "train_cycle_steps": 4,
        "sampler": "euler",
        "learned_null_condition": True,
        "sit_input_channels": 3,
        "sit_patch_size": 4,
        "sit_hidden_size": 32,
        "sit_depth": 2,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 32,
        "attention_backend": "native",
    }


def test_flow_generator_config_supports_rectified_flow_sit_roundtrip() -> None:
    from safa.models.generator import FlowGeneratorConfig

    config = FlowGeneratorConfig.from_dict(_tiny_rectified_flow_sit_config())
    payload = config.to_dict()

    assert config.model_type == "rectified_flow_sit"
    assert config.sampler == "euler"
    assert config.sample_steps == 4
    assert payload["model_type"] == "rectified_flow_sit"
    assert payload["sampler"] == "euler"
    assert payload["sit_hidden_size"] == 32
    assert payload["sit_data_space"] == "pixel"


def test_rectified_flow_sit_loss_and_multi_step_sample_are_finite() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_rectified_flow_sit_config())
    z = torch.zeros(2, 16)
    x = torch.rand(2, 3, 16, 16)
    x_init = torch.randn(2, 3, 16, 16)

    loss, metrics = generator.flow_matching_loss(x, z)
    loss.backward()
    sample = generator.sample(z, steps=4, x_init=x_init, clamp_output=False)

    assert generator.config.model_type == "rectified_flow_sit"
    assert tuple(loss.shape) == ()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["flow_matching_mse"])
    assert metrics["rectified_flow_backbone"] == "sit"
    assert metrics["rectified_flow_sampler"] == "euler"
    assert tuple(sample.shape) == (2, 3, 16, 16)
    assert torch.isfinite(sample).all()
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_rectified_flow_sit_requires_flow_sampler() -> None:
    from safa.models.generator import build_generator

    config = _tiny_rectified_flow_sit_config()
    config["sampler"] = "ddim"

    with pytest.raises(ValueError, match="rectified_flow_sit sampler must be euler or heun"):
        build_generator(config)
