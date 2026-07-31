from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

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


def _resume_contract_inputs() -> tuple[dict, dict]:
    config, kwargs = _contract_inputs()
    contract = {
        "contract_type": g_loop._R14_RESUME_CONTRACT,
        "source_global_step": 2432,
        "source_completed_stage2_epochs": 19,
        "source_world_size": 4,
        "source_global_batch_size": 8,
        "source_per_device_batch_size": 2,
        "samples_per_epoch": 1024,
        "target_world_size": 2,
        "target_global_batch_size": 4,
        "target_per_device_batch_size": 2,
        "additional_optimizer_steps": 256,
        "target_global_step": 2688,
    }
    config.update(
        {
            "resume_from": g_loop._R14_RESUME_SOURCE_CHECKPOINT,
            "resume_optimizer_state": True,
            "r14_resume_contract": contract,
        }
    )
    config.pop("resume_from_sha256")
    config["optimizer_step_contract"]["required_steps"] = 2688
    config["optimizer_checkpoint_contract"]["save_steps"] = [2688]
    kwargs.update(
        {
            "resume_mode": g_loop._RESUME_MODE_TRAINING_STATE,
            "resume_checkpoint_model": g_loop._RESUME_CHECKPOINT_MODEL_RAW,
            "required_optimizer_steps": 2688,
            "optimizer_checkpoint_steps": (2688,),
            "batch_config": SimpleNamespace(per_device_batch_size=2, global_batch_size=4),
        }
    )
    return config, kwargs


def _resume_checkpoint() -> dict[str, Any]:
    history = []
    for epoch in range(19):
        history.append(
            {
                "stage": "stage2",
                "stage_epoch": epoch,
                "loss": 1.0,
                "global_step": (epoch + 1) * 128,
                "required_optimizer_steps": 2560,
                "world_size": 4,
                "global_batch_size": 8,
                "per_device_batch_size": 2,
            }
        )
    metrics = dict(history[-1])
    metrics.update({"stage_epoch_0based": 18, "stage_epoch_1based": 19})
    return {
        "metrics": metrics,
        "history": history,
        "training_config": {
            "world_size": 4,
            "global_batch_size": 8,
            "per_device_batch_size": 2,
        },
        "optimizer_state_dict": {
            "state": {0: {"step": 2432}},
            "param_groups": [{"params": [0]}],
        },
        "ema_model_state_dict": {"weight": object()},
    }


def test_r14_epoch_boundary_resume_contract_runs_one_full_world2_epoch() -> None:
    config, kwargs = _resume_contract_inputs()
    contract = g_loop._r14_resume_contract(config)

    assert contract is not None
    assert g_loop._optimizer_step_contract(config) == 2688
    assert g_loop._optimizer_checkpoint_steps(config, 2688) == (2688,)
    g_loop._validate_r14_inpaint_training_contract(config, **kwargs)
    g_loop._validate_r14_resume_checkpoint(_resume_checkpoint(), "last.pt", contract)
    progress = g_loop._resume_stage_progress_from_metrics(_resume_checkpoint()["metrics"], "last.pt")
    assert g_loop._resume_stage_start_epoch("stage2", kwargs["stages"], progress) == 19
    step = 2432
    for _ in range(256):
        step, reached = g_loop._advance_global_step(step, 2688, optimizer_step_succeeded=True)
    assert (step, reached) == (2688, True)
    with pytest.raises(RuntimeError, match="exceeded required_steps=2688"):
        g_loop._advance_global_step(step, 2688, optimizer_step_succeeded=True)


def test_r14_world2_epoch19_sampler_covers_1024_samples_once() -> None:
    from torch.utils.data import DistributedSampler

    shards = []
    for rank in range(2):
        sampler = DistributedSampler(
            range(1024),
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=1337,
            drop_last=False,
        )
        sampler.set_epoch(19)
        shard = list(sampler)
        assert len(shard) == 512
        assert len(shard) // 2 == 256
        shards.append(shard)
    assert set(shards[0]).isdisjoint(shards[1])
    assert set(shards[0]) | set(shards[1]) == set(range(1024))


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("half_epoch", "target_global_step"),
        ("weights_only", "training_state"),
        ("ema", "checkpoint_model='raw'"),
        ("optimizer_false", "optimizer_state=true"),
        ("global8", "global_batch_size=4"),
    ),
)
def test_r14_resume_rejects_partial_or_non_training_state_contract(mutation: str, match: str) -> None:
    config, kwargs = _resume_contract_inputs()
    if mutation == "half_epoch":
        config["r14_resume_contract"]["target_global_step"] = 2560
    elif mutation == "weights_only":
        kwargs["resume_mode"] = g_loop._RESUME_MODE_MODEL_WEIGHTS_ONLY
    elif mutation == "ema":
        kwargs["resume_checkpoint_model"] = g_loop._RESUME_CHECKPOINT_MODEL_EMA
    elif mutation == "optimizer_false":
        config["resume_optimizer_state"] = False
    elif mutation == "global8":
        kwargs["batch_config"] = SimpleNamespace(per_device_batch_size=2, global_batch_size=8)
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        g_loop._validate_r14_inpaint_training_contract(config, **kwargs)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("step", "metrics.global_step"),
        ("history", "19 completed epochs"),
        ("optimizer", "optimizer_state_dict"),
        ("ema", "ema_model_state_dict"),
    ),
)
def test_r14_resume_checkpoint_must_be_complete_epoch_boundary(mutation: str, match: str) -> None:
    config, _ = _resume_contract_inputs()
    checkpoint = _resume_checkpoint()
    if mutation == "step":
        checkpoint["metrics"]["global_step"] = 2400
    elif mutation == "history":
        checkpoint["history"].pop()
    elif mutation == "optimizer":
        checkpoint.pop("optimizer_state_dict")
    elif mutation == "ema":
        checkpoint.pop("ema_model_state_dict")
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        g_loop._validate_r14_resume_checkpoint(
            checkpoint,
            "last.pt",
            g_loop._r14_resume_contract(config),
        )


def test_r14_training_state_load_requires_complete_topology() -> None:
    g_loop._validate_r14_training_state_load([], [])
    with pytest.raises(RuntimeError, match="complete trained topology"):
        g_loop._validate_r14_training_state_load(["context_embedder.weight"], [])


def test_validation_disabled_resume_preserves_last_only_history() -> None:
    history = _resume_checkpoint()["history"]
    config = {"validation": {"enabled": False}}
    stages = {"stage1": {"epochs": 0}, "stage2": {"epochs": 20}}

    assert g_loop._resume_history_for_checkpoint_selection(history, "last.pt", config, stages) == history


def test_r14_resume_completion_requires_twenty_epochs_and_1024_samples() -> None:
    config, _ = _resume_contract_inputs()
    history = _resume_checkpoint()["history"]
    history.append(
        {
            "stage": "stage2",
            "stage_epoch": 19,
            "stage_epoch_1based": 20,
            "global_step": 2688,
            "world_size": 2,
            "global_batch_size": 4,
            "per_device_batch_size": 2,
            "epoch_sample_count": 1024,
        }
    )
    g_loop._validate_r14_resume_completion(history, g_loop._r14_resume_contract(config))
    history[-1]["epoch_sample_count"] = 512
    with pytest.raises(RuntimeError, match="epoch_sample_count"):
        g_loop._validate_r14_resume_completion(history, g_loop._r14_resume_contract(config))
