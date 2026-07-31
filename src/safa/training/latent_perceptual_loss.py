from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from safa.models.generator import FlowGeneratorConfig


R13_LPL_CONTRACT = "safa_r13_decoder_latent_perceptual_loss_v1"
R13_LPL_FEATURE_NAMES = (
    "mid_block",
    "up_block_0",
    "up_block_1",
    "up_block_2",
    "up_block_3",
)
R13_LPL_WEIGHT = 3.0
R13_LPL_SNR_TAU = 3.0
R13_LPL_NORMALIZATION = "prediction_spatial_mean_variance_cross_normalization"
R13_LPL_LAYER_WEIGHTING = "inverse_linear_upsampling"
R13_LPL_FLOW_SUBSET = "r_equals_t_and_snr_lte_tau"
R13_LPL_SPATIAL_VALIDITY = "all_features_fail_closed"


@dataclass(frozen=True)
class LatentPerceptualLossRuntime:
    enabled: bool
    weight: float = R13_LPL_WEIGHT
    snr_tau: float = R13_LPL_SNR_TAU
    feature_names: tuple[str, ...] = R13_LPL_FEATURE_NAMES


def latent_perceptual_loss_runtime_from_config(
    config: Mapping[str, Any],
    generator_config: FlowGeneratorConfig,
) -> LatentPerceptualLossRuntime:
    payload = config.get("latent_perceptual_loss")
    if payload is None:
        return LatentPerceptualLossRuntime(enabled=False)
    if not isinstance(payload, Mapping):
        raise ValueError("train_g config.latent_perceptual_loss must be a mapping")
    context = "train_g config.latent_perceptual_loss"
    required = {
        "contract_type": R13_LPL_CONTRACT,
        "weight": R13_LPL_WEIGHT,
        "snr_tau": R13_LPL_SNR_TAU,
        "normalization": R13_LPL_NORMALIZATION,
        "layer_weighting": R13_LPL_LAYER_WEIGHTING,
        "flow_subset": R13_LPL_FLOW_SUBSET,
        "spatial_validity": R13_LPL_SPATIAL_VALIDITY,
    }
    for field, expected in required.items():
        if field not in payload:
            raise ValueError(f"{context} missing required field {field!r}")
        value = payload[field]
        if isinstance(expected, float):
            if isinstance(value, bool):
                raise ValueError(f"{context}.{field} must equal {expected}, got bool")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}.{field} must equal {expected}, got {value!r}") from exc
            if not math.isfinite(numeric) or numeric != expected:
                raise ValueError(f"{context}.{field} must equal {expected}, got {value!r}")
        elif value != expected:
            raise ValueError(f"{context}.{field} must equal {expected!r}, got {value!r}")
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"{context}.enabled must be true or false, got {enabled!r}")
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or tuple(feature_names) != R13_LPL_FEATURE_NAMES:
        raise ValueError(f"{context}.feature_names must equal {list(R13_LPL_FEATURE_NAMES)!r}")
    unexpected = sorted(set(payload) - {"enabled", "feature_names", *required})
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {unexpected!r}")
    runtime = LatentPerceptualLossRuntime(enabled=enabled)
    if not enabled:
        return runtime
    if config.get("latent_training") is not True:
        raise ValueError(f"{context}.enabled=true requires train_g config.latent_training=true")
    if generator_config.model_type != "meanflow_sit":
        raise ValueError(f"{context}.enabled=true requires generator.model_type='meanflow_sit'")
    if generator_config.sit_data_space != "latent" or generator_config.sit_input_channels != 4:
        raise ValueError(
            f"{context}.enabled=true requires generator.sit_data_space='latent' and sit_input_channels=4"
        )
    return runtime


def decoder_latent_perceptual_loss(
    codec,
    clean_latents: Mapping[str, torch.Tensor],
    runtime: LatentPerceptualLossRuntime,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not runtime.enabled:
        raise RuntimeError("decoder_latent_perceptual_loss requires an enabled runtime")
    if codec is None:
        raise RuntimeError("decoder_latent_perceptual_loss requires a latent codec")
    required = (
        "target_clean_latent",
        "predicted_clean_latent",
        "active_mask",
        "sampled_r",
        "sampled_t",
        "snr",
    )
    missing = [field for field in required if field not in clean_latents]
    if missing:
        raise RuntimeError(f"MeanFlow clean latent payload missing fields: {missing!r}")
    target = clean_latents["target_clean_latent"]
    prediction = clean_latents["predicted_clean_latent"]
    active_mask = clean_latents["active_mask"]
    sampled_r = clean_latents["sampled_r"]
    sampled_t = clean_latents["sampled_t"]
    snr = clean_latents["snr"]
    _validate_clean_latent_payload(target, prediction, active_mask, sampled_r, sampled_t, snr, runtime)
    active_count = int(active_mask.sum().item())
    batch_size = int(active_mask.numel())
    if active_count == 0:
        zero = prediction.sum() * 0.0
        return zero, {
            "latent_perceptual_loss_raw": 0.0,
            "latent_perceptual_loss_weighted": 0.0,
            "latent_perceptual_active_count": 0.0,
            "latent_perceptual_active_fraction": 0.0,
            "latent_perceptual_min_prediction_variance": 0.0,
            "latent_perceptual_weight": runtime.weight,
        }

    with torch.no_grad():
        target_features = codec.decode_intermediate_features(target)
    prediction_features = codec.decode_intermediate_features(prediction)
    _require_feature_contract(target_features, prediction_features, runtime.feature_names)

    base_height, base_width = target_features[runtime.feature_names[0]].shape[-2:]
    total = prediction.new_zeros(())
    metrics: dict[str, float] = {}
    minimum_variance: torch.Tensor | None = None
    for name in runtime.feature_names:
        target_feature = target_features[name].detach()
        prediction_feature = prediction_features[name]
        if target_feature.shape != prediction_feature.shape:
            raise RuntimeError(
                f"LPL feature shape mismatch for {name}: target={tuple(target_feature.shape)} "
                f"prediction={tuple(prediction_feature.shape)}"
            )
        channels = int(prediction_feature.shape[1])
        if channels <= 0:
            raise RuntimeError(f"LPL feature {name} has no channels")
        prediction_mean = prediction_feature.mean(dim=(-2, -1), keepdim=True)
        centered_prediction = prediction_feature - prediction_mean
        prediction_variance = centered_prediction.square().mean(dim=(-2, -1), keepdim=True)
        if not torch.isfinite(prediction_variance).all().item():
            raise RuntimeError(f"LPL prediction variance is non-finite for {name}")
        if torch.any(prediction_variance <= 0.0).item():
            raise RuntimeError(f"LPL prediction variance must be positive for every active channel in {name}")
        prediction_std = prediction_variance.sqrt()
        normalized_prediction = centered_prediction / prediction_std
        normalized_target = (target_feature - prediction_mean) / prediction_std
        difference = normalized_target - normalized_prediction
        per_sample = difference.flatten(1).square().sum(dim=1) / float(channels)
        layer_loss = per_sample.mean()
        _assert_finite_scalar(f"LPL layer loss {name}", layer_loss)
        layer_weight = _inverse_linear_upsampling_weight(
            base_height,
            base_width,
            int(prediction_feature.shape[-2]),
            int(prediction_feature.shape[-1]),
            name,
        )
        total = total + layer_weight * layer_loss
        layer_minimum = prediction_variance.detach().min()
        minimum_variance = layer_minimum if minimum_variance is None else torch.minimum(minimum_variance, layer_minimum)
        metrics[f"latent_perceptual_layer_{name}_loss"] = float(layer_loss.detach().cpu())
        metrics[f"latent_perceptual_layer_{name}_weight"] = layer_weight
    _assert_finite_scalar("latent perceptual loss", total)
    if minimum_variance is None or not torch.isfinite(minimum_variance).item() or minimum_variance.item() <= 0.0:
        raise RuntimeError("LPL minimum prediction variance must be positive and finite")
    raw = float(total.detach().cpu())
    metrics.update(
        {
            "latent_perceptual_loss_raw": raw,
            "latent_perceptual_loss_weighted": runtime.weight * raw,
            "latent_perceptual_active_count": float(active_count),
            "latent_perceptual_active_fraction": float(active_count) / float(batch_size),
            "latent_perceptual_min_prediction_variance": float(minimum_variance.cpu()),
            "latent_perceptual_weight": runtime.weight,
        }
    )
    return total, metrics


def _validate_clean_latent_payload(target, prediction, active_mask, sampled_r, sampled_t, snr, runtime) -> None:
    for name, tensor in (
        ("target_clean_latent", target),
        ("predicted_clean_latent", prediction),
        ("active_mask", active_mask),
        ("sampled_r", sampled_r),
        ("sampled_t", sampled_t),
        ("snr", snr),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"MeanFlow clean latent payload {name} must be a tensor")
    if active_mask.ndim != 1 or active_mask.dtype != torch.bool:
        raise RuntimeError("MeanFlow clean latent active_mask must be a rank-one bool tensor")
    if sampled_r.shape != active_mask.shape or sampled_t.shape != active_mask.shape or snr.shape != active_mask.shape:
        raise RuntimeError("MeanFlow clean latent time tensors must match active_mask shape")
    if target.ndim != 4 or prediction.ndim != 4 or target.shape != prediction.shape:
        raise RuntimeError(
            "MeanFlow active clean latents must have matching [B,4,H,W] shapes, "
            f"got target={tuple(target.shape)} prediction={tuple(prediction.shape)}"
        )
    if target.shape[0] != int(active_mask.sum().item()) or target.shape[1] != 4:
        raise RuntimeError("MeanFlow active clean latent count/channels do not match active_mask")
    expected_active = torch.eq(sampled_r, sampled_t) & (snr <= runtime.snr_tau)
    if not torch.equal(active_mask, expected_active):
        raise RuntimeError("MeanFlow clean latent active_mask violates the r==t and SNR threshold contract")
    if target.requires_grad:
        raise RuntimeError("LPL target clean latent must be detached")
    if prediction.shape[0] > 0 and not prediction.requires_grad:
        raise RuntimeError("LPL predicted clean latent must remain differentiable")
    for name, tensor in (("target", target), ("prediction", prediction), ("snr", snr)):
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"LPL {name} contains non-finite values")


def _require_feature_contract(target, prediction, feature_names: tuple[str, ...]) -> None:
    if not isinstance(target, Mapping) or not isinstance(prediction, Mapping):
        raise RuntimeError("VAE decoder intermediate features must be mappings")
    for name in feature_names:
        if name not in target or name not in prediction:
            raise RuntimeError(f"VAE decoder intermediate features missing required LPL feature {name!r}")


def _inverse_linear_upsampling_weight(
    base_height: int,
    base_width: int,
    height: int,
    width: int,
    name: str,
) -> float:
    if min(base_height, base_width, height, width) <= 0:
        raise RuntimeError(f"LPL feature {name} has invalid spatial resolution {height}x{width}")
    height_weight = float(base_height) / float(height)
    width_weight = float(base_width) / float(width)
    if not math.isclose(height_weight, width_weight, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(f"LPL feature {name} must use isotropic linear upsampling")
    if height_weight > 1.0:
        raise RuntimeError(f"LPL feature {name} resolution must not be below the decoder base resolution")
    if not math.isfinite(height_weight) or height_weight <= 0.0:
        raise RuntimeError(f"LPL feature {name} has invalid inverse upsampling weight {height_weight}")
    return height_weight


def _assert_finite_scalar(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value).item():
        raise RuntimeError(f"{name} must be a finite scalar tensor")
