from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from safa.training.latent_perceptual_loss import (
    R13_LPL_CONTRACT,
    R13_LPL_FEATURE_NAMES,
    R13_LPL_FLOW_SUBSET,
    R13_LPL_LAYER_WEIGHTING,
    R13_LPL_NORMALIZATION,
    R13_LPL_SPATIAL_VALIDITY,
    R13_LPL_SNR_TAU,
    R13_LPL_WEIGHT,
)


R13_TRAINING_CONTRACT_TYPE = "safa_r13_control_lpl_training_v1"
R13_SOURCE_CHECKPOINT = (
    "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/"
    "last_nopretrained.pt"
)
R13_SOURCE_CHECKPOINT_SHA256 = (
    "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d"
)
R13_CONDITIONING_PARAMETER_NUMEL = 44_688_384
R13_TRAIN_ORDER_PATH = (
    "artifacts/r13_control_lpl_training/preparation_v1/train_order_seed1337.jsonl"
)
R13_TRAIN_ORDER_SHA256 = "05e805511058f241cc35f1b7c1086b30354f6d80756009cc9997a47904424e41"
R13_OPTIMIZER_STEP_CONTRACT = {
    "contract_type": "safa_r13_exact_optimizer_steps_v1",
    "required_steps": 7500,
}
R13_OPTIMIZER_CHECKPOINT_CONTRACT = {
    "contract_type": "safa_r13_optimizer_checkpoint_steps_v1",
    "save_steps": [0, 2500, 5000, 7500],
}
R13_FLOW_MATCHING_RNG_CONTRACT = {
    "contract_type": "safa_r13_dedicated_flow_rng_v1",
    "seed": 1337,
    "algorithm": "torch.Generator",
    "draw_order": ["eps", "r_t_pairs", "equality_mask"],
    "ledger_filename": "flow_rng_ledger.jsonl",
}
R13_TRAIN_ORDER_CONTRACT = {
    "contract_type": "safa_r13_locked_train_order_v1",
    "path": R13_TRAIN_ORDER_PATH,
    "sha256": R13_TRAIN_ORDER_SHA256,
    "seed": 1337,
    "sample_count": 30000,
    "batch_size": 4,
}
R13_CONDITIONING_PARAMETER_NAMES = tuple(
    sorted(
        [
            *(f"vector_field.blocks.{index}.adaLN_modulation.1.{suffix}" for index in range(12) for suffix in ("bias", "weight")),
            "vector_field.final_layer.adaLN_modulation.1.bias",
            "vector_field.final_layer.adaLN_modulation.1.weight",
            "vector_field.z_embedder.0.bias",
            "vector_field.z_embedder.0.weight",
            "vector_field.z_embedder.2.bias",
            "vector_field.z_embedder.2.weight",
        ]
    )
)


def build_r13_training_contract(config: Mapping[str, Any], generator) -> dict[str, Any] | None:
    arm_id = config.get("r13_arm_id")
    if arm_id is None:
        return None
    if arm_id not in {"control", "lpl"}:
        raise ValueError("r13_arm_id must be 'control' or 'lpl'")
    lpl = config.get("latent_perceptual_loss")
    if not isinstance(lpl, Mapping) or lpl.get("enabled") is not (arm_id == "lpl"):
        raise ValueError("r13_arm_id and latent_perceptual_loss.enabled disagree")
    actual = tuple(sorted(name for name, parameter in generator.named_parameters() if parameter.requires_grad))
    if actual != R13_CONDITIONING_PARAMETER_NAMES:
        raise RuntimeError("R13 conditioning-only trainable parameter allowlist differs")
    parameter_by_name = dict(generator.named_parameters())
    parameter_numel = sum(int(parameter_by_name[name].numel()) for name in actual)
    if parameter_numel != R13_CONDITIONING_PARAMETER_NUMEL:
        raise RuntimeError("R13 conditioning-only trainable parameter count differs")
    source_path = config.get("resume_from")
    source_sha256 = config.get("resume_from_sha256")
    if source_path != R13_SOURCE_CHECKPOINT or source_sha256 != R13_SOURCE_CHECKPOINT_SHA256:
        raise ValueError("R13 source checkpoint binding differs")
    return {
        "schema_version": 1,
        "contract_type": R13_TRAINING_CONTRACT_TYPE,
        "arm_id": arm_id,
        "latent_perceptual_loss": dict(lpl),
        "optimizer_step_contract": dict(config["optimizer_step_contract"]),
        "optimizer_checkpoint_contract": (
            None
            if config.get("optimizer_checkpoint_contract") is None
            else dict(config["optimizer_checkpoint_contract"])
        ),
        "flow_matching_rng_contract": dict(config["flow_matching_rng_contract"]),
        "train_order_contract": dict(config["train_order_contract"]),
        "resume": {
            "mode": config.get("resume_mode"),
            "checkpoint_model": config.get("resume_checkpoint_model"),
            "source_checkpoint": source_path,
            "source_checkpoint_sha256": source_sha256,
        },
        "trainable": {
            "selector": config.get("generator_trainable"),
            "parameter_names": list(actual),
            "parameter_count": len(actual),
            "parameter_numel": parameter_numel,
        },
    }


def validate_r13_training_contract_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "contract_type",
        "arm_id",
        "latent_perceptual_loss",
        "optimizer_step_contract",
        "optimizer_checkpoint_contract",
        "flow_matching_rng_contract",
        "train_order_contract",
        "resume",
        "trainable",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("R13 checkpoint training contract fields differ")
    if payload.get("schema_version") != 1 or payload.get("contract_type") != R13_TRAINING_CONTRACT_TYPE:
        raise ValueError("R13 checkpoint training contract identity differs")
    arm_id = payload.get("arm_id")
    lpl = payload.get("latent_perceptual_loss")
    if arm_id not in {"control", "lpl"} or not isinstance(lpl, Mapping):
        raise ValueError("R13 checkpoint arm declaration differs")
    if lpl.get("enabled") is not (arm_id == "lpl"):
        raise ValueError("R13 checkpoint arm and LPL enabled flag disagree")
    expected_lpl = {
        "contract_type": R13_LPL_CONTRACT,
        "enabled": arm_id == "lpl",
        "weight": R13_LPL_WEIGHT,
        "snr_tau": R13_LPL_SNR_TAU,
        "feature_names": list(R13_LPL_FEATURE_NAMES),
        "normalization": R13_LPL_NORMALIZATION,
        "layer_weighting": R13_LPL_LAYER_WEIGHTING,
        "flow_subset": R13_LPL_FLOW_SUBSET,
        "spatial_validity": R13_LPL_SPATIAL_VALIDITY,
    }
    if dict(lpl) != expected_lpl:
        raise ValueError("R13 checkpoint latent perceptual loss block differs")
    if payload.get("optimizer_step_contract") != R13_OPTIMIZER_STEP_CONTRACT:
        raise ValueError("R13 checkpoint optimizer step contract differs")
    if payload.get("optimizer_checkpoint_contract") != R13_OPTIMIZER_CHECKPOINT_CONTRACT:
        raise ValueError("R13 checkpoint optimizer checkpoint contract differs")
    if payload.get("flow_matching_rng_contract") != R13_FLOW_MATCHING_RNG_CONTRACT:
        raise ValueError("R13 checkpoint flow RNG contract differs")
    if payload.get("train_order_contract") != R13_TRAIN_ORDER_CONTRACT:
        raise ValueError("R13 checkpoint train order contract differs")
    resume = payload.get("resume")
    expected_resume = {
        "mode": "model_weights_only",
        "checkpoint_model": "ema",
        "source_checkpoint": R13_SOURCE_CHECKPOINT,
        "source_checkpoint_sha256": R13_SOURCE_CHECKPOINT_SHA256,
    }
    if resume != expected_resume:
        raise ValueError("R13 checkpoint resume binding differs")
    trainable = payload.get("trainable")
    if not isinstance(trainable, Mapping) or dict(trainable) != {
        "selector": "conditioning_only",
        "parameter_names": list(R13_CONDITIONING_PARAMETER_NAMES),
        "parameter_count": len(R13_CONDITIONING_PARAMETER_NAMES),
        "parameter_numel": R13_CONDITIONING_PARAMETER_NUMEL,
    }:
        raise ValueError("R13 checkpoint trainable allowlist differs")
    return dict(payload)
