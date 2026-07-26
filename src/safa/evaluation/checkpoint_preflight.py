"""Fail-closed generator checkpoint reconstruction and validation.

The checkpoint's recorded stage-2 objective is the only accepted source for
adapter reconstruction.  State-dict key patterns are used to verify that the
recorded objective and serialized tensors agree; they are never used to guess
an adapter configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from safa.models.generator import (
    build_generator,
    generator_sample_channels,
    require_generator_model_config,
)
from safa.utils.hashing import sha256_file


CONTRACT_TYPE = "safa_generator_checkpoint_preflight_v1"
SCHEMA_VERSION = 1
ADAPTER_OBJECTIVE_TYPES = {
    "lora_sweep",
    "peft_fm",
    "peft_lora",
    "peft_mlp",
}


class CheckpointPreflightError(ValueError):
    """A hard checkpoint failure with a machine-readable result."""

    def __init__(self, result: Mapping[str, Any]):
        self.result = dict(result)
        super().__init__(str(self.result["failure_message"]))


def _base_result(
    path: Path,
    selector: Any,
    expected_checkpoint_sha256: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_type": CONTRACT_TYPE,
        "status": "invalid",
        "checkpoint_path": str(path),
        "checkpoint_sha256": None,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "sha256_binding": None,
        "checkpoint_model": selector,
        "declared_checkpoint_model": None,
        "available_state_dict_fields": [],
        "selector_binding": None,
        "state_dict_field": None,
        "tensor_count": 0,
        "finite_tensor_count": 0,
        "nonfinite_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "reconstruction_messages": [],
        "adapter": {
            "type": None,
            "objective_type": None,
            "configuration_source": None,
            "state_key_count": 0,
            "mounted_key_count": 0,
            "mounted": False,
        },
        "smoke": {
            "requested_sample_count": 0,
            "executed_sample_count": 0,
            "output_shape": None,
        },
        "failure_code": None,
        "failure_message": None,
    }


def _fail(
    result: Mapping[str, Any],
    code: str,
    message: str,
    **updates: Any,
) -> None:
    payload = dict(result)
    payload.update(updates)
    payload["status"] = "invalid"
    payload["failure_code"] = str(code)
    payload["failure_message"] = str(message)
    raise CheckpointPreflightError(payload)


def _mapping(value: Any, label: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(result, "invalid_mapping", f"{label} must be a mapping")
    return value


def _expected_sha256(value: Any, result: Mapping[str, Any]) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            result,
            "invalid_expected_checkpoint_sha256",
            "expected_checkpoint_sha256 must be a lowercase SHA256 digest",
        )
    return value


def _stage2_objective(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    training = payload.get("training_config")
    if training is None:
        return None
    training = _mapping(training, "training_config", result)
    stages = training.get("stages")
    if stages is None:
        return None
    stages = _mapping(stages, "training_config.stages", result)
    stage2 = stages.get("stage2")
    if stage2 is None:
        return None
    stage2 = _mapping(stage2, "training_config.stages.stage2", result)
    objective = stage2.get("stage2_objective")
    if objective is None:
        return None
    return _mapping(
        objective,
        "training_config.stages.stage2.stage2_objective",
        result,
    )


def _adapter_state_type(keys: list[str], result: Mapping[str, Any]) -> tuple[str, int]:
    groups = {
        "ip_adapter": [key for key in keys if ".ip_adapter." in key],
        "peft_mlp": [key for key in keys if ".cond_mlp_adapter." in key],
        "lora": [
            key
            for key in keys
            if ".lora_a." in key or ".lora_b." in key
        ],
        "peft_lora_support": [
            key
            for key in keys
            if (
                "gated_low_rank_z" in key
                or "generic_bank" in key
                or "_peft_lora_null_embed" in key
            )
        ],
    }
    active = {
        name
        for name in ("ip_adapter", "peft_mlp", "lora")
        if groups[name]
    }
    if len(active) > 1:
        _fail(
            result,
            "mixed_adapter_state",
            f"checkpoint contains incompatible adapter state groups: {sorted(active)}",
        )
    if groups["peft_lora_support"] and active - {"lora"}:
        _fail(
            result,
            "mixed_adapter_state",
            "PEFT-LoRA support tensors coexist with another adapter type",
        )
    if groups["peft_lora_support"]:
        adapter_type = "peft_lora"
        adapter_keys = groups["peft_lora_support"] + groups["lora"]
    elif groups["lora"]:
        adapter_type = "lora_target"
        adapter_keys = groups["lora"]
    elif groups["ip_adapter"]:
        adapter_type = "ip_adapter"
        adapter_keys = groups["ip_adapter"]
    elif groups["peft_mlp"]:
        adapter_type = "peft_mlp"
        adapter_keys = groups["peft_mlp"]
    else:
        adapter_type = "none"
        adapter_keys = []
    return adapter_type, len(set(adapter_keys))


def _objective_adapter_type(
    objective: Mapping[str, Any] | None,
    result: Mapping[str, Any],
) -> tuple[str, str | None]:
    if objective is None:
        return "none", None
    raw_type = objective.get("type")
    if not isinstance(raw_type, str) or not raw_type:
        _fail(
            result,
            "adapter_configuration_invalid",
            "stage2_objective.type must be a non-empty string",
        )
    objective_type = str(raw_type)
    if objective_type == "peft_fm":
        return "ip_adapter", objective_type
    if objective_type == "peft_mlp":
        return "peft_mlp", objective_type
    if objective_type == "peft_lora":
        return "peft_lora", objective_type
    if objective_type == "lora_sweep":
        return "lora_target", objective_type
    if objective_type == "point_projected_two_step":
        targets = objective.get("lora_target_modules")
        if targets is None:
            return "none", objective_type
        if (
            not isinstance(targets, (list, tuple))
            or not targets
            or any(not isinstance(item, str) or not item for item in targets)
        ):
            _fail(
                result,
                "adapter_configuration_invalid",
                "point_projected_two_step lora_target_modules must be non-empty strings",
            )
        return "lora_target", objective_type
    return "none", objective_type


def _require_nonempty_string_list(
    value: Any,
    label: str,
    result: Mapping[str, Any],
) -> list[str]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        _fail(
            result,
            "adapter_configuration_invalid",
            f"{label} must be a non-empty list of strings",
        )
    return [str(item) for item in value]


def _mount_adapter(
    generator: Any,
    objective: Mapping[str, Any] | None,
    adapter_type: str,
    result: Mapping[str, Any],
) -> None:
    if adapter_type == "none":
        return
    if objective is None:
        _fail(
            result,
            "adapter_configuration_missing",
            "checkpoint contains adapter tensors but no recorded stage2_objective",
        )
    objective_type = str(objective["type"])
    context = "checkpoint.training_config.stages.stage2.stage2_objective"
    if objective_type == "peft_fm":
        from safa.training.peft_runner import (
            init_peft_generator,
            peft_stage2_objective_from_config,
        )

        parsed = peft_stage2_objective_from_config(dict(objective), context)
        init_peft_generator(generator, parsed)
        return
    if objective_type == "peft_mlp":
        from safa.training.peft_runner import (
            init_peft_mlp_generator,
            peft_mlp_objective_from_config,
        )

        parsed = peft_mlp_objective_from_config(dict(objective), context)
        init_peft_mlp_generator(generator, parsed)
        return
    if objective_type == "peft_lora":
        from safa.training.peft_runner import (
            init_peft_lora_generator,
            peft_lora_objective_from_config,
        )

        parsed = peft_lora_objective_from_config(dict(objective), context)
        init_peft_lora_generator(generator, parsed)
        return
    if objective_type in {"lora_sweep", "point_projected_two_step"}:
        from safa.models.peft_lora import wrap_backbone_with_lora_target

        targets = _require_nonempty_string_list(
            objective.get("lora_target_modules"),
            f"{context}.lora_target_modules",
            result,
        )
        rank = int(objective.get("lora_rank", 8))
        alpha = float(objective.get("lora_alpha", 4.0))
        if rank <= 0 or alpha <= 0:
            _fail(
                result,
                "adapter_configuration_invalid",
                f"LoRA rank and alpha must be positive, got rank={rank}, alpha={alpha}",
            )
        wrap_backbone_with_lora_target(
            generator.vector_field,
            target_modules=targets,
            rank=rank,
            alpha=alpha,
        )
        return
    _fail(
        result,
        "unsupported_adapter_objective",
        f"unsupported adapter objective type: {objective_type!r}",
    )


def _assert_finite_state(
    state_dict: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[int, int]:
    import torch

    nonfinite = []
    finite_count = 0
    for key, value in state_dict.items():
        if not isinstance(key, str):
            _fail(
                result,
                "invalid_state_dict_key",
                f"state_dict key must be str, got {type(key).__name__}",
            )
        if not torch.is_tensor(value):
            _fail(
                result,
                "invalid_state_dict_value",
                f"state_dict[{key!r}] is not a tensor",
            )
        if torch.is_floating_point(value) or torch.is_complex(value):
            if not bool(torch.isfinite(value).all().item()):
                nonfinite.append(key)
            else:
                finite_count += 1
    if nonfinite:
        _fail(
            result,
            "nonfinite_tensor",
            f"checkpoint contains non-finite tensors: {nonfinite[:8]}",
            nonfinite_keys=nonfinite,
            tensor_count=len(state_dict),
            finite_tensor_count=finite_count,
        )
    return len(state_dict), finite_count


def _smoke_generator(
    generator: Any,
    sample_count: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    if sample_count <= 0:
        return {
            "requested_sample_count": 0,
            "executed_sample_count": 0,
            "output_shape": None,
        }
    config = generator.config
    embedding_dim = int(config.embedding_dim)
    image_size = int(config.image_size)
    channels = int(generator_sample_channels(config))
    device = next(generator.parameters()).device
    dtype = next(generator.parameters()).dtype
    z = torch.zeros(sample_count, embedding_dim, device=device, dtype=dtype)
    x_init = torch.zeros(
        sample_count,
        channels,
        image_size,
        image_size,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        output = generator.sample(z, steps=1, x_init=x_init, clamp_output=False)
    expected = (sample_count, channels, image_size, image_size)
    if tuple(output.shape) != expected:
        _fail(
            result,
            "smoke_shape_mismatch",
            f"smoke output shape {tuple(output.shape)} != {expected}",
        )
    if not bool(torch.isfinite(output).all().item()):
        _fail(result, "smoke_nonfinite", "smoke output contains non-finite values")
    return {
        "requested_sample_count": sample_count,
        "executed_sample_count": sample_count,
        "output_shape": list(expected),
    }


def strict_load_generator_checkpoint(
    checkpoint_path: str | Path,
    checkpoint_model: str | None,
    device: str = "cpu",
    *,
    expected_checkpoint_sha256: str | None = None,
    compute_sha256: bool = False,
    smoke_samples: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct, validate, and strictly load one generator checkpoint."""
    import torch

    path = Path(checkpoint_path)
    selector = checkpoint_model if isinstance(checkpoint_model, str) else None
    result = _base_result(path, selector, expected_checkpoint_sha256)
    if not path.is_file():
        _fail(result, "checkpoint_missing", f"checkpoint does not exist: {path}")
    if smoke_samples < 0:
        _fail(
            result,
            "invalid_smoke_sample_count",
            f"smoke_samples must be >= 0, got {smoke_samples}",
        )
    expected_sha256 = _expected_sha256(expected_checkpoint_sha256, result)
    if compute_sha256 or expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        result["checkpoint_sha256"] = actual_sha256
        if expected_sha256 is None:
            result["sha256_binding"] = "computed_unbound"
        elif actual_sha256 == expected_sha256:
            result["sha256_binding"] = "expected_exact"
        else:
            result["sha256_binding"] = "expected_mismatch"
            _fail(
                result,
                "checkpoint_sha256_mismatch",
                "checkpoint content does not match expected_checkpoint_sha256",
            )
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        _fail(
            result,
            "checkpoint_deserialization_failed",
            f"failed to deserialize checkpoint: {type(exc).__name__}: {exc}",
        )
    payload = _mapping(payload, "checkpoint payload", result)
    try:
        model_config = dict(require_generator_model_config(payload, str(path)))
    except Exception as exc:
        _fail(
            result,
            "model_config_invalid",
            f"invalid generator model_config: {type(exc).__name__}: {exc}",
        )
    if selector not in {"raw", "ema"}:
        message = (
            "checkpoint_model is required for eval and must be 'raw' or 'ema'"
            if selector is None
            else f"checkpoint_model must be 'raw' or 'ema', got {selector!r}"
        )
        _fail(result, "invalid_selector", message)
    declared_selector = payload.get("checkpoint_model")
    result["declared_checkpoint_model"] = declared_selector
    available_state_fields = [
        field
        for field in ("model_state_dict", "ema_model_state_dict")
        if isinstance(payload.get(field), Mapping) and bool(payload.get(field))
    ]
    result["available_state_dict_fields"] = available_state_fields
    if declared_selector is not None:
        if declared_selector not in {"raw", "ema"}:
            _fail(
                result,
                "invalid_declared_selector",
                f"checkpoint declares invalid checkpoint_model={declared_selector!r}",
            )
        if declared_selector != selector:
            _fail(
                result,
                "selector_mismatch",
                f"requested {selector!r} but checkpoint declares {declared_selector!r}",
            )
        result["selector_binding"] = "explicit_checkpoint_model"
    elif len(available_state_fields) == 1:
        result["selector_binding"] = "single_available_state_dict"
    else:
        result["selector_binding"] = "explicit_request_with_multiple_states"
    state_field = (
        "model_state_dict" if selector == "raw" else "ema_model_state_dict"
    )
    result["state_dict_field"] = state_field
    state_dict = payload.get(state_field)
    if not isinstance(state_dict, Mapping) or not state_dict:
        _fail(
            result,
            "selector_state_missing",
            f"checkpoint requested {selector!r} but missing non-empty {state_field}",
        )
    tensor_count, finite_count = _assert_finite_state(state_dict, result)
    result["tensor_count"] = tensor_count
    result["finite_tensor_count"] = finite_count

    objective = _stage2_objective(payload, result)
    state_adapter_type, adapter_key_count = _adapter_state_type(
        sorted(str(key) for key in state_dict),
        result,
    )
    objective_adapter_type, objective_type = _objective_adapter_type(
        objective,
        result,
    )
    result["adapter"] = {
        "type": state_adapter_type,
        "objective_type": objective_type,
        "configuration_source": (
            "training_config.stages.stage2.stage2_objective"
            if objective is not None
            else None
        ),
        "state_key_count": adapter_key_count,
        "mounted_key_count": 0,
        "mounted": state_adapter_type == "none",
    }
    if state_adapter_type != "none" and objective is None:
        _fail(
            result,
            "adapter_configuration_missing",
            "checkpoint contains adapter tensors but no recorded stage2_objective",
        )
    if state_adapter_type != objective_adapter_type:
        _fail(
            result,
            "adapter_contract_mismatch",
            "serialized adapter type "
            f"{state_adapter_type!r} != objective adapter type "
            f"{objective_adapter_type!r} ({objective_type!r})",
        )

    reconstruction_stdout = StringIO()
    try:
        # A complete checkpoint load must not depend on an initialization-only
        # pretrained asset.  The serialized state remains authoritative.
        if "sit_pretrained_path" in model_config:
            model_config["sit_pretrained_path"] = ""
        with redirect_stdout(reconstruction_stdout):
            generator = build_generator(model_config)
            _mount_adapter(generator, objective, state_adapter_type, result)
    except CheckpointPreflightError:
        raise
    except Exception as exc:
        _fail(
            result,
            "model_reconstruction_failed",
            f"failed to reconstruct checkpoint model: {type(exc).__name__}: {exc}",
        )
    result["reconstruction_messages"] = [
        line
        for line in reconstruction_stdout.getvalue().splitlines()
        if line.strip()
    ]

    model_keys = set(generator.state_dict())
    checkpoint_keys = set(str(key) for key in state_dict)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    mounted_adapter_keys = len(
        [
            key
            for key in model_keys
            if (
                ".ip_adapter." in key
                or ".cond_mlp_adapter." in key
                or ".lora_a." in key
                or ".lora_b." in key
                or "gated_low_rank_z" in key
                or "generic_bank" in key
                or "_peft_lora_null_embed" in key
            )
        ]
    )
    result["adapter"] = {
        **result["adapter"],
        "mounted_key_count": mounted_adapter_keys,
        "mounted": (
            state_adapter_type == "none"
            or (
                adapter_key_count > 0
                and mounted_adapter_keys == adapter_key_count
            )
        ),
    }
    if missing or unexpected:
        code = (
            "adapter_not_mounted"
            if unexpected
            and any(
                (
                    ".ip_adapter." in key
                    or ".cond_mlp_adapter." in key
                    or ".lora_a." in key
                    or ".lora_b." in key
                    or "gated_low_rank_z" in key
                    or "generic_bank" in key
                    or "_peft_lora_null_embed" in key
                )
                for key in unexpected
            )
            else "state_dict_key_mismatch"
        )
        _fail(
            result,
            code,
            f"strict state_dict mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}",
            missing_keys=missing,
            unexpected_keys=unexpected,
        )
    if state_adapter_type != "none" and not result["adapter"]["mounted"]:
        _fail(
            result,
            "adapter_not_mounted",
            "adapter state keys were not mounted exactly",
        )
    model_state = generator.state_dict()
    shape_mismatches = []
    for key in sorted(model_keys):
        checkpoint_value = state_dict[key]
        model_value = model_state[key]
        if tuple(checkpoint_value.shape) != tuple(model_value.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(checkpoint_value.shape),
                    "model_shape": list(model_value.shape),
                }
            )
    if shape_mismatches:
        _fail(
            result,
            "state_dict_shape_mismatch",
            f"strict state_dict shape mismatch count={len(shape_mismatches)}",
            shape_mismatches=shape_mismatches,
        )
    try:
        generator.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        _fail(
            result,
            "state_dict_load_failed",
            f"strict state_dict load failed: {type(exc).__name__}: {exc}",
        )
    generator = generator.to(device).eval()
    result["smoke"] = _smoke_generator(generator, int(smoke_samples), result)
    result["status"] = "valid"
    result["failure_code"] = None
    result["failure_message"] = None
    setattr(generator, "_safa_checkpoint_preflight", dict(result))
    return generator, result


def preflight_generator_checkpoint(
    checkpoint_path: str | Path,
    checkpoint_model: str | None,
    device: str = "cpu",
    *,
    expected_checkpoint_sha256: str | None = None,
    compute_sha256: bool = True,
    smoke_samples: int = 0,
) -> dict[str, Any]:
    """Return a machine-readable result instead of raising on validation failure."""
    try:
        _, result = strict_load_generator_checkpoint(
            checkpoint_path,
            checkpoint_model,
            device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            compute_sha256=compute_sha256,
            smoke_samples=smoke_samples,
        )
        return result
    except CheckpointPreflightError as exc:
        return dict(exc.result)
