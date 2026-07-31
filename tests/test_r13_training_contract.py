from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
from torch import nn


def _contract_config(arm_id: str = "lpl") -> dict:
    return {
        "r13_arm_id": arm_id,
        "latent_perceptual_loss": {
            "contract_type": "safa_r13_decoder_latent_perceptual_loss_v1",
            "enabled": arm_id == "lpl",
            "weight": 3.0,
            "snr_tau": 3.0,
            "feature_names": [
                "mid_block",
                "up_block_0",
                "up_block_1",
                "up_block_2",
                "up_block_3",
            ],
            "normalization": "prediction_spatial_mean_variance_cross_normalization",
            "layer_weighting": "inverse_linear_upsampling",
            "flow_subset": "r_equals_t_and_snr_lte_tau",
            "spatial_validity": "all_features_fail_closed",
        },
        "optimizer_step_contract": {
            "contract_type": "safa_r13_exact_optimizer_steps_v1",
            "required_steps": 7500,
        },
        "optimizer_checkpoint_contract": {
            "contract_type": "safa_r13_optimizer_checkpoint_steps_v1",
            "save_steps": [0, 2500, 5000, 7500],
        },
        "flow_matching_rng_contract": {
            "contract_type": "safa_r13_dedicated_flow_rng_v1",
            "seed": 1337,
            "algorithm": "torch.Generator",
            "draw_order": ["eps", "r_t_pairs", "equality_mask"],
            "ledger_filename": "flow_rng_ledger.jsonl",
        },
        "train_order_contract": {
            "contract_type": "safa_r13_locked_train_order_v1",
            "path": "artifacts/r13_control_lpl_training/preparation_v1/train_order_seed1337.jsonl",
            "sha256": "05e805511058f241cc35f1b7c1086b30354f6d80756009cc9997a47904424e41",
            "seed": 1337,
            "sample_count": 30000,
            "batch_size": 4,
        },
        "resume_mode": "model_weights_only",
        "resume_checkpoint_model": "ema",
        "resume_from": (
            "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/"
            "last_nopretrained.pt"
        ),
        "resume_from_sha256": "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d",
        "generator_trainable": "conditioning_only",
    }


class _FakeParameter:
    requires_grad = True

    def __init__(self, numel: int):
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class _ExactConditioningGenerator:
    def named_parameters(self):
        from safa.training.r13_training_contract import (
            R13_CONDITIONING_PARAMETER_NAMES,
            R13_CONDITIONING_PARAMETER_NUMEL,
        )

        sizes = [1] * len(R13_CONDITIONING_PARAMETER_NAMES)
        sizes[0] += R13_CONDITIONING_PARAMETER_NUMEL - len(sizes)
        return iter(
            (name, _FakeParameter(size))
            for name, size in zip(R13_CONDITIONING_PARAMETER_NAMES, sizes, strict=True)
        )


@pytest.mark.parametrize("arm_id", ["control", "lpl"])
def test_r13_training_contract_builds_exact_arm_semantics(arm_id: str) -> None:
    from safa.training.r13_training_contract import (
        R13_CONDITIONING_PARAMETER_NAMES,
        build_r13_training_contract,
        validate_r13_training_contract_payload,
    )

    payload = build_r13_training_contract(_contract_config(arm_id), _ExactConditioningGenerator())
    assert payload is not None
    assert validate_r13_training_contract_payload(payload) == payload
    assert payload["arm_id"] == arm_id
    assert payload["latent_perceptual_loss"]["enabled"] is (arm_id == "lpl")
    assert payload["trainable"]["parameter_names"] == list(R13_CONDITIONING_PARAMETER_NAMES)
    assert payload["trainable"]["parameter_count"] == 30
    assert payload["trainable"]["parameter_numel"] == 44_688_384
    tampered = {**payload, "optimizer_step_contract": {**payload["optimizer_step_contract"], "required_steps": 7499}}
    with pytest.raises(ValueError, match="optimizer step contract differs"):
        validate_r13_training_contract_payload(tampered)


def test_save_generator_serializes_r13_training_contract(tmp_path: Path, monkeypatch) -> None:
    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop
    from safa.training.r13_training_contract import build_r13_training_contract

    payload = build_r13_training_contract(_contract_config(), _ExactConditioningGenerator())
    assert payload is not None
    monkeypatch.setattr(g_loop, "build_r13_training_contract", lambda _config, _generator: payload)
    generator = nn.Linear(2, 2)
    generator_config = FlowGeneratorConfig(
        image_size=8,
        embedding_dim=2,
        base_channels=4,
        channel_multipliers=(1,),
        condition_dim=2,
        model_type="flow",
    )
    output = tmp_path / "checkpoint.pt"
    train_config = {
        "validation": {"enabled": False, "face_detection": {"enabled": False}},
        "ema": {
            "enabled": False,
            "decay": 0.9999,
            "evaluate_raw": True,
            "evaluate_ema": False,
            "save_ema_checkpoint": False,
        },
        "best_model": "raw",
        "generator_trainable": "conditioning_only",
        "stages": {"stage1": {"epochs": 0}, "stage2": {"epochs": 1}},
    }
    g_loop._save_generator(
        output,
        generator,
        generator_config,
        train_config,
        {"stage": "stage2", "loss": 1.0},
        [],
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["r13_training_contract"] == payload


def test_optimizer_checkpoint_filename_and_contract_are_exact() -> None:
    from safa.training.g_loop import (
        _optimizer_checkpoint_steps,
        _optimizer_step_checkpoint_filename,
    )

    config = {
        "optimizer_checkpoint_contract": {
            "contract_type": "safa_r13_optimizer_checkpoint_steps_v1",
            "save_steps": [0, 2500, 5000, 7500],
        }
    }
    assert _optimizer_checkpoint_steps(config, 7500) == (0, 2500, 5000, 7500)
    assert [_optimizer_step_checkpoint_filename(step) for step in (0, 2500, 5000, 7500)] == [
        "step_00000000.pt",
        "step_00002500.pt",
        "step_00005000.pt",
        "step_00007500.pt",
    ]
