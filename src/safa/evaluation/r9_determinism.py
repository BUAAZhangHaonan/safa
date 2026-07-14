from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from safa.evaluation.r8_arm_contracts import canonical_arm_config_digest


R9_EXPERIMENT_CONTRACT = "safa_r9_meanflow_v1"
R9_ATTENTION_BACKEND = "native"
R9_DETERMINISM_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


R9_DETERMINISM_POLICY_SHA256 = canonical_json_sha256(R9_DETERMINISM_POLICY)


def is_r9_guidance_config(config: Mapping[str, Any]) -> bool:
    return config.get("experiment_contract") == R9_EXPERIMENT_CONTRACT


def validate_r9_execution_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("experiment_contract") != R9_EXPERIMENT_CONTRACT:
        raise ValueError(f"R9 experiment_contract must be {R9_EXPERIMENT_CONTRACT!r}")
    policy = config.get("determinism_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("R9 config requires a determinism_policy mapping")
    normalized_policy = dict(policy)
    if normalized_policy != R9_DETERMINISM_POLICY:
        raise ValueError(
            "R9 determinism_policy must match the strict registered policy"
        )
    policy_sha256 = canonical_json_sha256(normalized_policy)
    declared_policy_sha256 = config.get("determinism_policy_sha256")
    if declared_policy_sha256 is not None and declared_policy_sha256 != policy_sha256:
        raise ValueError(
            "R9 determinism_policy_sha256 disagrees with the canonical policy"
        )
    if config.get("attention_backend") != R9_ATTENTION_BACKEND:
        raise ValueError("R9 attention_backend must be explicitly locked to 'native'")
    return {
        "experiment_contract": R9_EXPERIMENT_CONTRACT,
        "determinism_policy": normalized_policy,
        "determinism_policy_sha256": policy_sha256,
        "attention_backend": R9_ATTENTION_BACKEND,
    }


def apply_r9_strict_cuda_determinism(
    config: Mapping[str, Any], *, torch_module
) -> dict[str, Any]:
    contract = validate_r9_execution_config(config)
    if torch_module.cuda.is_initialized():
        raise RuntimeError(
            "R9 strict determinism must be applied before CUDA initialization"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        R9_DETERMINISM_POLICY["cublas_workspace_config"]
    )
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    assert_r9_strict_cuda_determinism(torch_module=torch_module)
    return contract


def assert_r9_strict_cuda_determinism(*, torch_module) -> None:
    expected_cublas = str(R9_DETERMINISM_POLICY["cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_cublas:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG is not locked to the R9 policy")
    if not torch_module.are_deterministic_algorithms_enabled():
        raise RuntimeError("torch deterministic algorithms are not enabled")
    if torch_module.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("torch deterministic algorithms must use hard-error mode")
    actual = {
        "cudnn_deterministic": torch_module.backends.cudnn.deterministic,
        "cudnn_benchmark": torch_module.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch_module.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch_module.backends.cudnn.allow_tf32,
    }
    expected = {
        field: R9_DETERMINISM_POLICY[field]
        for field in (
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        )
    }
    if actual != expected:
        raise RuntimeError(
            f"active torch backend flags disagree with R9 determinism policy: {actual!r}"
        )


def canonical_r9_arm_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_r9_execution_config(config)
    active_intervals = None
    if config.get("mode") in {
        "official_head_current_xt",
        "paper_algorithm_split",
    }:
        value = config.get("active_guidance_intervals")
        if not isinstance(value, list) or any(
            interval not in {"I1", "I2", "I3"} for interval in value
        ):
            raise ValueError("R9 FMRG arm digest requires a canonical interval mask")
        canonical = [interval for interval in ("I1", "I2", "I3") if interval in value]
        if value != canonical or len(set(value)) != len(value):
            raise ValueError("R9 FMRG interval mask must follow canonical order")
        active_intervals = list(value)
    bound_digests = {}
    for field in (
        "semigroup_preflight_contract_sha256",
        "r9_semigroup_gate_contract_sha256",
    ):
        value = config.get(field)
        if value is not None:
            value = _require_sha256(value, field)
        bound_digests[field] = value
    return {
        "schema_version": 1,
        "experiment_contract": R9_EXPERIMENT_CONTRACT,
        "base_arm_config_sha256": canonical_arm_config_digest(config),
        "determinism_policy": contract["determinism_policy"],
        "determinism_policy_sha256": contract["determinism_policy_sha256"],
        "attention_backend": contract["attention_backend"],
        "active_guidance_intervals": active_intervals,
        **bound_digests,
    }


def canonical_r9_arm_config_digest(config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(canonical_r9_arm_config_payload(config))


def canonical_guidance_arm_config_digest(config: Mapping[str, Any]) -> str:
    if is_r9_guidance_config(config):
        return canonical_r9_arm_config_digest(config)
    return canonical_arm_config_digest(config)


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text
