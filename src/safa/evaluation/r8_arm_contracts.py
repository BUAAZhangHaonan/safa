from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


ALGORITHM_FIELDS = (
    "sample_mode",
    "optimization_mode",
    "step_size",
    "eta",
    "projection",
    "typical_delta",
    "num_optim_iters",
    "num_updates",
    "guided_steps",
    "unguided_tail_intervals",
    "t_cut",
)

FIXED_ASSET_FIELDS = (
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_model",
    "expected_stage",
    "expected_stage_epoch_1based",
    "expected_model_type",
    "expected_sit_patch_size",
    "transport_condition",
    "e0_checkpoint",
    "e0_sha256",
    "edev_checkpoint",
    "edev_sha256",
    "heldout_e1_checkpoint",
    "heldout_e1_sha256",
    "heldout_e2_checkpoint",
    "heldout_e2_sha256",
    "vae_path",
    "vae_digest",
    "vae_scaling_factor",
    "index",
    "index_sha256",
    "feature_source",
    "features",
    "features_digest",
    "sampling_seed",
)


def canonical_arm_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(config.get("mode", config.get("route", "")))
    if mode == "noise_oracle":
        mode = "initial_noise"
    locked_schedule = config.get("locked_schedule")
    schedule_digest = (
        locked_schedule.get("schedule_contract_sha256")
        if isinstance(locked_schedule, Mapping)
        else config.get("schedule_contract_sha256")
    )
    fixed_assets = {
        field: _json_value(config.get(field)) for field in FIXED_ASSET_FIELDS
    }
    r13_contract = config.get("r13_evaluator_contract")
    if r13_contract is not None:
        if not isinstance(r13_contract, Mapping):
            raise ValueError("r13_evaluator_contract must be a mapping")
        encoded_contract = json.dumps(
            dict(r13_contract),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        fixed_assets["r13_evaluator_contract_sha256"] = hashlib.sha256(
            encoded_contract
        ).hexdigest()
    return {
        "schema_version": 1,
        "mode": mode,
        "algorithm": {field: _json_value(config.get(field)) for field in ALGORITHM_FIELDS},
        "schedule_contract_sha256": _json_value(schedule_digest),
        "fixed_assets": fixed_assets,
    }


def canonical_arm_config_digest(config: Mapping[str, Any]) -> str:
    payload = canonical_arm_config_payload(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require_arm_config_digest(value: Any, label: str = "arm config SHA256") -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"arm config contract contains a non-scalar value: {value!r}")
