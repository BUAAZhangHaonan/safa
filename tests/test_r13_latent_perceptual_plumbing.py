from __future__ import annotations

from types import MethodType

import pytest


torch = pytest.importorskip("torch")
from torch import nn


def _latent_meanflow_config() -> dict:
    return {
        "model_type": "meanflow_sit",
        "embedding_dim": 8,
        "image_size": 8,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 8,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "meanflow",
        "learned_null_condition": False,
        "meanflow_ratio": 0.25,
        "meanflow_ratio_r_not_equal_t": 1.0,
        "meanflow_adaptive_weighting": False,
        "meanflow_norm_p": 1.0,
        "meanflow_norm_eps": 0.001,
        "meanflow_jvp_mode": "first_order",
        "sit_input_channels": 4,
        "sit_data_space": "latent",
        "sit_patch_size": 2,
        "sit_hidden_size": 16,
        "sit_depth": 1,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 16,
    }


class _FeatureBlock(nn.Module):
    def __init__(self, channels: int, *, scale_factor: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale_factor = int(scale_factor)

    def forward(self, sample, latent_embeds=None):
        if latent_embeds is not None:
            raise AssertionError("the frozen SD VAE path must not supply latent_embeds")
        if self.scale_factor != 1:
            sample = torch.nn.functional.interpolate(sample, scale_factor=self.scale_factor, mode="nearest")
        return torch.nn.functional.silu(self.proj(sample))


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 4, kernel_size=1)
        self.mid_block = _FeatureBlock(4)
        self.up_blocks = nn.ModuleList(
            [
                _FeatureBlock(4),
                _FeatureBlock(4, scale_factor=2),
                _FeatureBlock(4, scale_factor=2),
                _FeatureBlock(4, scale_factor=2),
            ]
        )


class _FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_tiling = False
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1)
        self.decoder = _FakeDecoder()

    def decode(self, _latents):
        raise AssertionError("intermediate features must use the explicit decoder path")


def test_decoder_intermediate_features_are_explicit_finite_and_freeze_vae() -> None:
    from safa.training.latent_codec import LatentCodec, LatentCodecConfig

    torch.manual_seed(17)
    vae = _FakeVAE()
    codec = LatentCodec(vae, LatentCodecConfig(source="fake", scaling_factor=0.5))
    latents = torch.randn(2, 4, 4, 4, requires_grad=True)

    features = codec.decode_intermediate_features(latents)
    loss = sum(feature.square().mean() for feature in features.values())
    loss.backward()

    assert tuple(features) == (
        "conv_in",
        "mid_block",
        "up_block_0",
        "up_block_1",
        "up_block_2",
        "up_block_3",
    )
    assert all(feature.ndim == 4 and torch.isfinite(feature).all() for feature in features.values())
    assert not vae.training
    assert all(not parameter.requires_grad for parameter in vae.parameters())
    assert all(parameter.grad is None for parameter in vae.parameters())
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert torch.count_nonzero(latents.grad).item() > 0


def test_decoder_intermediate_features_fail_closed_on_runtime_contract_errors() -> None:
    from safa.training.latent_codec import LatentCodec, LatentCodecConfig

    latents = torch.zeros(1, 4, 2, 2)
    vae = _FakeVAE()
    codec = LatentCodec(vae, LatentCodecConfig(source="fake", scaling_factor=1.0))

    vae.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        codec.decode_intermediate_features(latents)
    vae.eval()
    vae.use_tiling = True
    with pytest.raises(RuntimeError, match="tiling"):
        codec.decode_intermediate_features(latents)
    vae.use_tiling = False
    del vae.decoder.mid_block
    with pytest.raises(RuntimeError, match="mid_block"):
        codec.decode_intermediate_features(latents)


def test_decoder_intermediate_features_fail_closed_on_nonfinite_values() -> None:
    from safa.training.latent_codec import LatentCodec, LatentCodecConfig

    codec = LatentCodec(_FakeVAE(), LatentCodecConfig(source="fake", scaling_factor=1.0))
    bad = torch.zeros(1, 4, 2, 2)
    bad[0, 0, 0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="input latents.*non-finite"):
        codec.decode_intermediate_features(bad)


class _SyntheticMeanFlow(nn.Module):
    """Average velocity of the synthetic curve x(s)=s^2, scaled per channel."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.queried_r: list[torch.Tensor] = []
        self.resolved_attention_backend = "synthetic"

    def forward(self, x, r, t, _z):
        self.queried_r.append(r.detach().clone())
        return self.scale * (t + r).view(-1, 1, 1, 1).expand_as(x)


def _fixed_interval(self, batch_size, *, device, dtype, generator=None):
    del generator
    if batch_size != 2:
        raise AssertionError("the fixed interval test requires batch_size=2")
    return (
        torch.tensor([0.5, 0.25], device=device, dtype=dtype),
        torch.tensor([0.5, 0.75], device=device, dtype=dtype),
    )


def test_meanflow_clean_latent_uses_only_instantaneous_r_equal_t_rows() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_latent_meanflow_config())
    field = _SyntheticMeanFlow()
    generator.vector_field = field
    generator._sample_t_r = MethodType(_fixed_interval, generator)
    x_data = torch.randn(2, 4, 8, 8)
    condition = torch.randn(2, 8)

    _, _, clean = generator.flow_matching_loss(
        x_data,
        condition,
        generator=torch.Generator().manual_seed(11),
        return_clean_latents=True,
    )

    active = clean["active_mask"]
    active_t = clean["sampled_t"][active].view(-1, 1, 1, 1)
    active_velocity = clean["sampled_interval_velocity"][active]
    expected_z_0 = clean["noisy_latent"][active] - active_t * active_velocity

    assert len(field.queried_r) == 1
    assert torch.equal(field.queried_r[0], clean["sampled_r"])
    assert torch.equal(active, torch.tensor([True, False]))
    assert torch.equal(clean["target_clean_latent"], x_data[:1])
    assert torch.allclose(clean["predicted_clean_latent"], expected_z_0)
    assert clean["predicted_clean_latent"].shape[0] == 1


def test_meanflow_clean_latent_opt_in_preserves_control_loss_and_has_gradient() -> None:
    from safa.models.generator import build_generator

    torch.manual_seed(29)
    control = build_generator(_latent_meanflow_config())
    lpl = build_generator(_latent_meanflow_config())
    lpl.load_state_dict(control.state_dict())
    with torch.no_grad():
        control.vector_field.final_layer.linear.weight.normal_()
        control.vector_field.final_layer.linear.bias.normal_()
    lpl.load_state_dict(control.state_dict())
    control._sample_t_r = MethodType(_fixed_interval, control)
    lpl._sample_t_r = MethodType(_fixed_interval, lpl)
    x_data = torch.randn(2, 4, 8, 8)
    condition = torch.randn(2, 8)

    control_loss, control_metrics = control.flow_matching_loss(
        x_data,
        condition,
        generator=torch.Generator().manual_seed(31),
    )
    lpl_loss, lpl_metrics, clean = lpl.flow_matching_loss(
        x_data,
        condition,
        generator=torch.Generator().manual_seed(31),
        return_clean_latents=True,
    )
    perceptual_probe = clean["predicted_clean_latent"].square().mean()
    perceptual_probe.backward()

    assert torch.equal(control_loss, lpl_loss)
    assert torch.equal(control_metrics["flow_matching_mse"], lpl_metrics["flow_matching_mse"])
    assert torch.equal(clean["target_clean_latent"], x_data[clean["active_mask"]])
    assert not clean["target_clean_latent"].requires_grad
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in lpl.parameters()
    )


def test_meanflow_clean_latent_rejects_pixel_mode_and_non_bool_opt_in() -> None:
    from safa.models.generator import build_generator

    latent = build_generator(_latent_meanflow_config())
    with pytest.raises(TypeError, match="return_clean_latents must be a bool"):
        latent.flow_matching_loss(torch.randn(1, 4, 8, 8), torch.randn(1, 8), return_clean_latents=1)

    pixel_config = _latent_meanflow_config()
    pixel_config.update({"sit_input_channels": 3, "sit_data_space": "pixel"})
    pixel = build_generator(pixel_config)
    with pytest.raises(RuntimeError, match="requires latent MeanFlow"):
        pixel.flow_matching_loss(
            torch.rand(1, 3, 8, 8),
            torch.randn(1, 8),
            return_clean_latents=True,
        )


def _lpl_config_payload(*, enabled: bool) -> dict:
    from safa.training.latent_perceptual_loss import (
        R13_LPL_CONTRACT,
        R13_LPL_FEATURE_NAMES,
        R13_LPL_FLOW_SUBSET,
        R13_LPL_LAYER_WEIGHTING,
        R13_LPL_NORMALIZATION,
        R13_LPL_SPATIAL_VALIDITY,
    )

    return {
        "contract_type": R13_LPL_CONTRACT,
        "enabled": enabled,
        "weight": 3.0,
        "snr_tau": 3.0,
        "feature_names": list(R13_LPL_FEATURE_NAMES),
        "normalization": R13_LPL_NORMALIZATION,
        "layer_weighting": R13_LPL_LAYER_WEIGHTING,
        "flow_subset": R13_LPL_FLOW_SUBSET,
        "spatial_validity": R13_LPL_SPATIAL_VALIDITY,
    }


def test_r13_lpl_config_is_versioned_exact_and_latent_only() -> None:
    from safa.models.generator import FlowGeneratorConfig
    from safa.training.latent_perceptual_loss import latent_perceptual_loss_runtime_from_config

    generator_config = FlowGeneratorConfig.from_dict(_latent_meanflow_config())
    config = {"latent_training": True, "latent_perceptual_loss": _lpl_config_payload(enabled=True)}
    runtime = latent_perceptual_loss_runtime_from_config(config, generator_config)

    assert runtime.enabled
    assert runtime.weight == 3.0
    assert runtime.snr_tau == 3.0

    wrong = {"latent_training": True, "latent_perceptual_loss": _lpl_config_payload(enabled=True)}
    wrong["latent_perceptual_loss"]["weight"] = 2.0
    with pytest.raises(ValueError, match="weight must equal 3.0"):
        latent_perceptual_loss_runtime_from_config(wrong, generator_config)

    missing = {"latent_training": True, "latent_perceptual_loss": _lpl_config_payload(enabled=True)}
    del missing["latent_perceptual_loss"]["normalization"]
    with pytest.raises(ValueError, match="missing required field 'normalization'"):
        latent_perceptual_loss_runtime_from_config(missing, generator_config)


def _active_clean_payload(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    count = target.shape[0]
    sampled_t = torch.full((count,), 0.5, dtype=target.dtype, device=target.device)
    sampled_r = sampled_t.clone()
    return {
        "target_clean_latent": target,
        "predicted_clean_latent": prediction,
        "active_mask": torch.ones(count, dtype=torch.bool, device=target.device),
        "sampled_r": sampled_r,
        "sampled_t": sampled_t,
        "snr": sampled_t / (1.0 - sampled_t),
    }


def test_decoder_lpl_uses_five_cross_normalized_features_and_inverse_upsampling_weights() -> None:
    from safa.training.latent_codec import LatentCodec, LatentCodecConfig
    from safa.training.latent_perceptual_loss import (
        LatentPerceptualLossRuntime,
        decoder_latent_perceptual_loss,
    )

    torch.manual_seed(43)
    vae = _FakeVAE()
    codec = LatentCodec(vae, LatentCodecConfig(source="fake", scaling_factor=1.0))
    target = torch.randn(2, 4, 4, 4)
    prediction = (target + 0.2 * torch.randn_like(target)).detach().requires_grad_(True)
    runtime = LatentPerceptualLossRuntime(enabled=True)

    loss, metrics = decoder_latent_perceptual_loss(codec, _active_clean_payload(target, prediction), runtime)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert [metrics[f"latent_perceptual_layer_{name}_weight"] for name in runtime.feature_names] == [
        1.0,
        1.0,
        0.5,
        0.25,
        0.125,
    ]
    assert metrics["latent_perceptual_active_count"] == 2.0
    assert metrics["latent_perceptual_active_fraction"] == 1.0
    assert metrics["latent_perceptual_min_prediction_variance"] > 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad).item() > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in vae.parameters())


def test_decoder_lpl_zero_active_batch_is_zero_and_differentiable() -> None:
    from safa.training.latent_codec import LatentCodec, LatentCodecConfig
    from safa.training.latent_perceptual_loss import (
        LatentPerceptualLossRuntime,
        decoder_latent_perceptual_loss,
    )

    codec = LatentCodec(_FakeVAE(), LatentCodecConfig(source="fake", scaling_factor=1.0))
    base = torch.randn(2, 4, 4, 4, requires_grad=True)
    active = torch.zeros(2, dtype=torch.bool)
    sampled_r = torch.tensor([0.1, 0.2])
    sampled_t = torch.tensor([0.5, 0.6])
    payload = {
        "target_clean_latent": base.detach()[active],
        "predicted_clean_latent": base[active],
        "active_mask": active,
        "sampled_r": sampled_r,
        "sampled_t": sampled_t,
        "snr": sampled_t / (1.0 - sampled_t),
    }

    loss, metrics = decoder_latent_perceptual_loss(
        codec,
        payload,
        LatentPerceptualLossRuntime(enabled=True),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert metrics["latent_perceptual_active_count"] == 0.0
    assert base.grad is not None
    assert torch.count_nonzero(base.grad).item() == 0


class _ConstantFeatureCodec:
    def decode_intermediate_features(self, latents):
        from safa.training.latent_perceptual_loss import R13_LPL_FEATURE_NAMES

        resolutions = (2, 2, 4, 8, 16)
        return {
            name: latents.sum() * 0.0 + torch.ones(latents.shape[0], 2, size, size)
            for name, size in zip(R13_LPL_FEATURE_NAMES, resolutions, strict=True)
        }


def test_decoder_lpl_fails_closed_on_zero_prediction_variance_and_invalid_mask() -> None:
    from safa.training.latent_perceptual_loss import (
        LatentPerceptualLossRuntime,
        decoder_latent_perceptual_loss,
    )

    target = torch.randn(1, 4, 2, 2)
    prediction = torch.randn(1, 4, 2, 2, requires_grad=True)
    payload = _active_clean_payload(target, prediction)
    runtime = LatentPerceptualLossRuntime(enabled=True)

    with pytest.raises(RuntimeError, match="variance must be positive"):
        decoder_latent_perceptual_loss(_ConstantFeatureCodec(), payload, runtime)

    payload["sampled_r"] = torch.zeros_like(payload["sampled_t"])
    with pytest.raises(RuntimeError, match="active_mask violates"):
        decoder_latent_perceptual_loss(_ConstantFeatureCodec(), payload, runtime)
