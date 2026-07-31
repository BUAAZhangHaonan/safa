from __future__ import annotations

import math
import os
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from safa.evaluation.r9_determinism import (
    R9_DETERMINISM_POLICY,
    R9_DETERMINISM_POLICY_SHA256,
    assert_r9_strict_cuda_determinism,
)


R13_EVALUATOR_CONTRACT_TYPE = "safa_r13_r12_protocol_initial_noise_v1"
R13_EVALUATOR_CONTRACT_FIELD = "r13_evaluator_contract"
R13_FINAL_GLOBAL_STEP = 7500
R13_STAGE_EPOCH_1BASED = 1

R13_ARM_CHECKPOINT_ROOTS = {
    "control": PurePosixPath(
        "artifacts/checkpoints/r13_control_conditioning_1epoch_seed1337"
    ),
    "lpl": PurePosixPath(
        "artifacts/checkpoints/r13_lpl_conditioning_1epoch_seed1337"
    ),
}

R13_SAMPLE_MANIFESTS = {
    "regular32": {
        "path": "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl",
        "sha256": "fc47ae9372f23667e6b59ab2f33b22bc6cc8405be0095aaee1565d1158be1b05",
    },
    "tail32": {
        "path": "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl",
        "sha256": "f38fb6f6542c267b6c7d9cbec9ce57abdf9b0657edfe8d060fe533178a9f5b29",
    },
}

R13_LOCKED_ASSETS: dict[str, Any] = {
    "e0_checkpoint": "artifacts/checkpoints/e0_medium_v1/best.pt",
    "e0_sha256": "d7d2c57a552155776b8c15a4e52e43ec5082fc046aa0aabb4e9709685f7e3d1a",
    "edev_checkpoint": "artifacts/checkpoints/e0_resnet18/best.pt",
    "edev_sha256": "373b331c917834467e854ddf3fe20f39000532f189ec73f76a1abc55d82e560e",
    "vae_path": "artifacts/checkpoints/external/sd-vae-ft-ema",
    "vae_digest": "ac188e7f6ff31ff1a3bbde37fea3c345ec72f9e10589cf8aa8a3ec7e86afb188",
    "vae_scaling_factor": 0.18215,
    "index": "data/index/val_face_mixed_e14.jsonl",
    "index_sha256": "da14e23eacefecbc2948d1374fb93961a13d017a9183aa1fe2a2f62b33a4b4ea",
    "feature_source": "cached_features",
    "features": "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1",
    "features_digest": "287b8163f093f290e75e8ef09fbbedc986e6934ab0ac458ad786e655889fbe45",
    "heldout_e1_checkpoint": "artifacts/checkpoints/e0_dinov2_large_v2/best.pt",
    "heldout_e1_sha256": "cce0de2f1eab097cb6091886f587a9f334dd84ced1ca4dd5e08c3a765718a14c",
    "heldout_e2_checkpoint": "artifacts/checkpoints/e0_convnext_tiny/best.pt",
    "heldout_e2_sha256": "09c88bd416057222abefeba52ebe88d710715ede791ec34198a23ae5e6e850a8",
}

_CONTRACT_FIELDS = {
    "schema_version",
    "contract_type",
    "arm_id",
    "sample_set",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_model",
    "stage",
    "stage_epoch_1based",
    "global_step",
    "phase",
    "mode",
    "transport_condition",
    "seed",
    "sampling_seed",
    "projection",
    "eta",
    "num_updates",
    "batch_size",
    "max_samples",
    "pixel_image_size",
    "attention_backend",
    "determinism_policy_sha256",
    "sample_id_manifest",
    "sample_id_manifest_sha256",
}


def is_r13_evaluator_config(config: Mapping[str, Any]) -> bool:
    return (
        config.get("experiment_contract") == R13_EVALUATOR_CONTRACT_TYPE
        and R13_EVALUATOR_CONTRACT_FIELD in config
    )


def validate_r13_evaluator_contract(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    declared_type = config.get("experiment_contract")
    has_payload = R13_EVALUATOR_CONTRACT_FIELD in config
    if declared_type != R13_EVALUATOR_CONTRACT_TYPE and not has_payload:
        return None
    if declared_type != R13_EVALUATOR_CONTRACT_TYPE:
        raise ValueError(
            f"{R13_EVALUATOR_CONTRACT_FIELD} requires "
            f"experiment_contract={R13_EVALUATOR_CONTRACT_TYPE!r}"
        )
    raw = config.get(R13_EVALUATOR_CONTRACT_FIELD)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{R13_EVALUATOR_CONTRACT_FIELD} must be a mapping")
    keys = set(raw)
    if keys != _CONTRACT_FIELDS:
        missing = sorted(_CONTRACT_FIELDS - keys)
        extra = sorted(keys - _CONTRACT_FIELDS)
        raise ValueError(
            f"{R13_EVALUATOR_CONTRACT_FIELD} fields must match the registered "
            f"schema exactly; missing={missing!r} extra={extra!r}"
        )
    contract = dict(raw)
    if contract["schema_version"] != 1:
        raise ValueError("R13 evaluator schema_version must be 1")
    if contract["contract_type"] != R13_EVALUATOR_CONTRACT_TYPE:
        raise ValueError("R13 evaluator contract_type mismatch")

    arm_id = contract["arm_id"]
    if arm_id not in R13_ARM_CHECKPOINT_ROOTS:
        raise ValueError("R13 evaluator arm_id must be 'control' or 'lpl'")
    sample_set = contract["sample_set"]
    if sample_set not in R13_SAMPLE_MANIFESTS:
        raise ValueError("R13 evaluator sample_set must be 'regular32' or 'tail32'")

    checkpoint_path = _locked_checkpoint_path(contract["checkpoint_path"], arm_id)
    checkpoint_sha256 = _require_sha256(
        contract["checkpoint_sha256"], "R13 checkpoint_sha256"
    )
    _require_exact(contract, "checkpoint_model", "ema")
    _require_exact(contract, "stage", "stage2")
    _require_exact_int(
        contract,
        "stage_epoch_1based",
        R13_STAGE_EPOCH_1BASED,
    )
    _require_exact_int(contract, "global_step", R13_FINAL_GLOBAL_STEP)
    _require_exact(contract, "phase", "diagnose")
    _require_exact(contract, "mode", "initial_noise")
    _require_exact(contract, "transport_condition", "learned_null_condition")
    _require_exact_int(contract, "seed", 7919)
    _require_exact_int(contract, "sampling_seed", 7919)
    _require_exact(contract, "projection", "fixed_radius")
    _require_exact_finite_float(contract, "eta", 0.5)
    _require_exact_int(contract, "num_updates", 16)
    _require_exact_int(contract, "batch_size", 2)
    _require_exact_int(contract, "max_samples", 32)
    _require_exact_int(contract, "pixel_image_size", 256)
    _require_exact(contract, "attention_backend", "native")
    _require_exact(
        contract,
        "determinism_policy_sha256",
        R9_DETERMINISM_POLICY_SHA256,
    )

    manifest = R13_SAMPLE_MANIFESTS[str(sample_set)]
    _require_exact(contract, "sample_id_manifest", manifest["path"])
    _require_exact(
        contract,
        "sample_id_manifest_sha256",
        manifest["sha256"],
    )

    top_level_bindings = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_model": "ema",
        "expected_stage": "stage2",
        "expected_stage_epoch_1based": R13_STAGE_EPOCH_1BASED,
        "expected_global_step": R13_FINAL_GLOBAL_STEP,
        "expected_model_type": "meanflow_sit",
        "expected_sit_patch_size": 4,
        "phase": "diagnose",
        "mode": "initial_noise",
        "seed": 7919,
        "sampling_seed": 7919,
        "transport_condition": "learned_null_condition",
        "projection": "fixed_radius",
        "eta": 0.5,
        "num_updates": 16,
        "batch_size": 2,
        "max_samples": 32,
        "pixel_image_size": 256,
        "attention_backend": "native",
        "determinism_policy_sha256": R9_DETERMINISM_POLICY_SHA256,
        "sample_id_manifest": manifest["path"],
        "sample_id_manifest_sha256": manifest["sha256"],
        **R13_LOCKED_ASSETS,
    }
    for field, expected in top_level_bindings.items():
        actual = config.get(field)
        if field == "eta":
            _require_exact_finite_float(config, field, float(expected))
        elif field in {
            "seed",
            "sampling_seed",
            "num_updates",
            "expected_stage_epoch_1based",
            "expected_global_step",
            "expected_sit_patch_size",
            "batch_size",
            "max_samples",
            "pixel_image_size",
        }:
            _require_exact_int(config, field, int(expected))
        elif actual != expected:
            raise ValueError(
                f"R13 evaluator config {field} must be {expected!r}, got {actual!r}"
            )
    policy = config.get("determinism_policy")
    if not isinstance(policy, Mapping) or dict(policy) != R9_DETERMINISM_POLICY:
        raise ValueError(
            "R13 evaluator determinism_policy must match the registered strict policy"
        )
    return contract


def validate_r13_checkpoint_declaration(
    declaration: Mapping[str, Any], checkpoint_path: str | os.PathLike[str]
) -> dict[str, Any]:
    if not isinstance(declaration, Mapping):
        raise ValueError("R13 checkpoint declaration must be a mapping")
    synthetic_config = {
        "experiment_contract": R13_EVALUATOR_CONTRACT_TYPE,
        R13_EVALUATOR_CONTRACT_FIELD: declaration,
        "checkpoint": declaration.get("checkpoint_path"),
        "checkpoint_sha256": declaration.get("checkpoint_sha256"),
        "checkpoint_model": declaration.get("checkpoint_model"),
        "expected_stage": declaration.get("stage"),
        "expected_stage_epoch_1based": declaration.get("stage_epoch_1based"),
        "expected_global_step": declaration.get("global_step"),
        "expected_model_type": "meanflow_sit",
        "expected_sit_patch_size": 4,
        "phase": declaration.get("phase"),
        "mode": declaration.get("mode"),
        "transport_condition": declaration.get("transport_condition"),
        "seed": declaration.get("seed"),
        "sampling_seed": declaration.get("sampling_seed"),
        "projection": declaration.get("projection"),
        "eta": declaration.get("eta"),
        "num_updates": declaration.get("num_updates"),
        "batch_size": declaration.get("batch_size"),
        "max_samples": declaration.get("max_samples"),
        "pixel_image_size": declaration.get("pixel_image_size"),
        "attention_backend": declaration.get("attention_backend"),
        "determinism_policy": dict(R9_DETERMINISM_POLICY),
        "determinism_policy_sha256": declaration.get(
            "determinism_policy_sha256"
        ),
        "sample_id_manifest": declaration.get("sample_id_manifest"),
        "sample_id_manifest_sha256": declaration.get(
            "sample_id_manifest_sha256"
        ),
        **R13_LOCKED_ASSETS,
    }
    contract = validate_r13_evaluator_contract(synthetic_config)
    if contract is None:
        raise AssertionError("R13 checkpoint declaration validation returned no contract")
    actual_path = os.fspath(checkpoint_path)
    if actual_path != contract["checkpoint_path"]:
        raise ValueError(
            "R13 checkpoint loader path must match the declared repository-relative path: "
            f"declared={contract['checkpoint_path']!r} actual={actual_path!r}"
        )
    return contract


def apply_r13_strict_cuda_determinism(
    config: Mapping[str, Any], *, torch_module
) -> dict[str, Any]:
    execution_contract = r13_execution_contract(config)
    if torch_module.cuda.is_initialized():
        raise RuntimeError(
            "R13 strict determinism must be applied before CUDA initialization"
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
    return execution_contract


def r13_execution_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_r13_evaluator_contract(config)
    if contract is None:
        raise ValueError("R13 strict determinism requires an R13 evaluator contract")
    return {
        "determinism_policy": dict(R9_DETERMINISM_POLICY),
        "determinism_policy_sha256": R9_DETERMINISM_POLICY_SHA256,
        "attention_backend": "native",
    }


def assert_r13_strict_cuda_determinism(
    config: Mapping[str, Any], *, torch_module
) -> None:
    if validate_r13_evaluator_contract(config) is None:
        raise ValueError("R13 strict determinism requires an R13 evaluator contract")
    assert_r9_strict_cuda_determinism(torch_module=torch_module)


def _locked_checkpoint_path(value: Any, arm_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("R13 checkpoint_path must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise ValueError("R13 checkpoint_path must use a normalized POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError("R13 checkpoint_path must be a normalized repository-relative path")
    root = R13_ARM_CHECKPOINT_ROOTS[arm_id]
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"R13 {arm_id} checkpoint_path must be under {root.as_posix()!r}"
        ) from exc
    if not relative.parts or path.suffix != ".pt":
        raise ValueError("R13 checkpoint_path must name a .pt file under its locked arm root")
    return value


def _require_exact(mapping: Mapping[str, Any], field: str, expected: Any) -> None:
    actual = mapping.get(field)
    if actual != expected:
        raise ValueError(
            f"R13 evaluator {field} must be {expected!r}, got {actual!r}"
        )


def _require_exact_int(
    mapping: Mapping[str, Any], field: str, expected: int
) -> None:
    actual = mapping.get(field)
    if type(actual) is not int or actual != expected:
        raise ValueError(
            f"R13 evaluator {field} must be integer {expected}, got {actual!r}"
        )


def _require_exact_finite_float(
    mapping: Mapping[str, Any], field: str, expected: float
) -> None:
    actual = mapping.get(field)
    if isinstance(actual, bool):
        raise ValueError(
            f"R13 evaluator {field} must be finite {expected}, got {actual!r}"
        )
    try:
        parsed = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"R13 evaluator {field} must be finite {expected}, got {actual!r}"
        ) from exc
    if not math.isfinite(parsed) or not math.isclose(
        parsed, expected, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"R13 evaluator {field} must be finite {expected}, got {actual!r}"
        )


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text
