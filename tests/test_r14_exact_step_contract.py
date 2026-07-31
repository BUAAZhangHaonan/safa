from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from safa.training import g_loop


def _contract_inputs() -> tuple[dict, dict]:
    config = {
        "optimizer_step_contract": {
            "contract_type": g_loop._R14_OPTIMIZER_STEP_CONTRACT,
            "required_steps": 2560,
        },
        "optimizer_checkpoint_contract": {
            "contract_type": g_loop._R14_OPTIMIZER_CHECKPOINT_CONTRACT,
            "save_steps": [0, 2560],
        },
        "resume_from": g_loop.R13_SOURCE_CHECKPOINT,
        "resume_from_sha256": g_loop.R13_SOURCE_CHECKPOINT_SHA256,
        "r14_spatial": {
            "contract_type": g_loop._R14_SPATIAL_TRAINING_CONTRACT,
            "pair_manifest": "pairs.jsonl",
            "horizontal_flip_probability": 0.0,
        },
    }
    kwargs = {
        "generator_config": SimpleNamespace(learned_null_condition=True),
        "generator_trainable_mode": g_loop._GENERATOR_TRAINABLE_FULL,
        "resume_mode": g_loop._RESUME_MODE_MODEL_WEIGHTS_ONLY,
        "resume_checkpoint_model": g_loop._RESUME_CHECKPOINT_MODEL_EMA,
        "required_optimizer_steps": 2560,
        "optimizer_checkpoint_steps": (0, 2560),
        "stages": {"stage1": {"epochs": 0}, "stage2": {"epochs": 20}},
        "stage2_objective": SimpleNamespace(
            type="fm_only_probe",
            flow_condition=g_loop._FLOW_CONDITION_EMBEDDING,
        ),
        "ema_config": {"enabled": True, "save_ema_checkpoint": True},
        "batch_config": SimpleNamespace(
            per_device_batch_size=2,
            global_batch_size=8,
        ),
        "gradient_conflict": SimpleNamespace(enabled=False),
        "validation": {"enabled": False},
    }
    return config, kwargs


def test_r14_contract_is_exactly_twenty_epochs_and_2560_successful_steps() -> None:
    config, kwargs = _contract_inputs()

    assert g_loop._optimizer_step_contract(config) == 2560
    assert g_loop._optimizer_checkpoint_steps(config, 2560) == (0, 2560)
    g_loop._validate_r14_inpaint_training_contract(config, **kwargs)
    assert g_loop._advance_global_step(
        2559,
        2560,
        optimizer_step_succeeded=True,
    ) == (2560, True)
    with pytest.raises(RuntimeError, match="explicit successful optimizer.step"):
        g_loop._advance_global_step(
            2559,
            2560,
            optimizer_step_succeeded=False,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("required_steps", "exactly 2560 optimizer steps"),
        ("checkpoint_steps", r"exactly \[0, 2560\]"),
        ("epochs", "exactly 20 stages.stage2 epochs"),
    ),
)
def test_r14_contract_rejects_old_or_partial_duration(mutation: str, match: str) -> None:
    config, kwargs = _contract_inputs()
    config = deepcopy(config)
    kwargs = deepcopy(kwargs)
    if mutation == "required_steps":
        config["optimizer_step_contract"]["required_steps"] = 256
        kwargs["required_optimizer_steps"] = 256
        kwargs["optimizer_checkpoint_steps"] = (0, 256)
    elif mutation == "checkpoint_steps":
        kwargs["optimizer_checkpoint_steps"] = (0, 256)
    elif mutation == "epochs":
        kwargs["stages"]["stage2"]["epochs"] = 2
    else:
        raise AssertionError(f"unexpected mutation {mutation}")

    with pytest.raises(ValueError, match=match):
        g_loop._validate_r14_inpaint_training_contract(config, **kwargs)
