#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from safa.evaluation.r13_evaluator_contract import validate_r13_evaluator_contract
from safa.training.g_loop import _locked_train_order_indices, _validate_train_g_config
from safa.training.r13_training_contract import (
    R13_SOURCE_CHECKPOINT,
    R13_SOURCE_CHECKPOINT_SHA256,
    R13_TRAIN_ORDER_PATH,
    R13_TRAIN_ORDER_SHA256,
    validate_r13_training_contract_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARATION = REPO_ROOT / "artifacts/r13_control_lpl_training/preparation_v1"
FULL_CONFIGS = {
    "control": REPO_ROOT / "configs/medium_v2/experiments/r13_control_conditioning_1epoch_seed1337.yaml",
    "lpl": REPO_ROOT / "configs/medium_v2/experiments/r13_lpl_conditioning_1epoch_seed1337.yaml",
}
PROBE_CONFIGS = {
    "control": REPO_ROOT / "configs/medium_v2/experiments/r13_probe_control_conditioning_seed1337.yaml",
    "lpl": REPO_ROOT / "configs/medium_v2/experiments/r13_probe_lpl_conditioning_seed1337.yaml",
}
EVAL_TEMPLATES = {
    ("control", "regular32"): REPO_ROOT / "configs/medium_v2/experiments/r13_eval_control_regular32.template.yaml",
    ("control", "tail32"): REPO_ROOT / "configs/medium_v2/experiments/r13_eval_control_tail32.template.yaml",
    ("lpl", "regular32"): REPO_ROOT / "configs/medium_v2/experiments/r13_eval_lpl_regular32.template.yaml",
    ("lpl", "tail32"): REPO_ROOT / "configs/medium_v2/experiments/r13_eval_lpl_tail32.template.yaml",
}


class R13PreparationValidationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R13PreparationValidationError(f"JSON payload is not an object: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R13PreparationValidationError(f"YAML payload is not a mapping: {path}")
    return value


def _without_paths(value: Mapping[str, Any], ignored: Sequence[str]) -> dict[str, Any]:
    result = json.loads(json.dumps(value, allow_nan=False))
    for path in ignored:
        parts = path.split(".")
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.get(part)
            if not isinstance(cursor, dict):
                break
        else:
            cursor.pop(parts[-1], None)
    return result


def _assert_arm_equality(control: Mapping[str, Any], lpl: Mapping[str, Any], *, probe: bool) -> None:
    ignored = ["experiment_name", "r13_arm_id", "out_dir", "latent_perceptual_loss.enabled"]
    if probe:
        ignored.extend([
            "r13_active_row_probe.arm_id",
            "r13_active_row_probe.require_cumulative_active_rows",
            "r13_resource_binding.physical_gpu_index",
            "r13_resource_binding.physical_gpu_uuid",
        ])
    if _without_paths(control, ignored) != _without_paths(lpl, ignored):
        raise R13PreparationValidationError("R13 control/LPL configs differ outside the registered arm switch")


class _RecordsOnly:
    def __init__(self, sample_ids: Sequence[str]):
        self.records = [type("Record", (), {"sample_id": value})() for value in sample_ids]


def _validate_order(config: Mapping[str, Any]) -> None:
    path = REPO_ROOT / R13_TRAIN_ORDER_PATH
    if _sha256(path) != R13_TRAIN_ORDER_SHA256:
        raise R13PreparationValidationError("R13 train order SHA-256 differs")
    source_ids = []
    with (REPO_ROOT / "data/index/train_face_mixed_e14_4029avail.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            source_ids.append(str(json.loads(line)["sample_id"]))
    indices = _locked_train_order_indices(_RecordsOnly(source_ids), config["train_order_contract"])
    if len(indices) != 30000 or len(set(indices)) != 30000:
        raise R13PreparationValidationError("R13 train order is not an exact 30000-row permutation")


def _validate_eval_template(path: Path, arm_id: str, sample_set: str) -> None:
    config = _yaml(path)
    placeholder = "__R13_CONTROL_LAST_PT_SHA256__" if arm_id == "control" else "__R13_LPL_LAST_PT_SHA256__"
    if config.get("checkpoint_sha256") != placeholder or config.get("r13_evaluator_contract", {}).get("checkpoint_sha256") != placeholder:
        raise R13PreparationValidationError(f"R13 evaluator template placeholder differs: {path}")
    materialized = json.loads(json.dumps(config).replace(placeholder, "a" * 64))
    contract = validate_r13_evaluator_contract(materialized)
    if contract is None or contract["arm_id"] != arm_id or contract["sample_set"] != sample_set:
        raise R13PreparationValidationError(f"R13 evaluator template contract differs: {path}")


def _validate_ledgers(preparation: Path) -> None:
    training = _json(preparation / "training_ledger.json")
    probe = _json(preparation / "probe_ledger.json")
    training_jobs = training.get("jobs")
    probe_jobs = probe.get("jobs")
    if not isinstance(training_jobs, list) or [job["physical_gpu"]["index"] for job in training_jobs] != [0, 1]:
        raise R13PreparationValidationError("R13 training ledger must map control/LPL to GPU0/GPU1")
    if not isinstance(probe_jobs, list) or [job["physical_gpu"]["index"] for job in probe_jobs] != [2, 1]:
        raise R13PreparationValidationError("R13 probe ledger must map control/LPL to GPU2/GPU1")
    resource = _json(preparation / "resource_contract.json")
    if resource.get("allowed_physical_gpus") != [0, 1, 2] or resource.get("training_gpu_bindings") != {"control": 0, "lpl": 1} or resource.get("probe_gpu_bindings") != {"control": 2, "lpl": 1} or resource.get("probe_and_training_are_sequential") is not True:
        raise R13PreparationValidationError("R13 resource binding contract differs")
    for job in [*training_jobs, *probe_jobs]:
        if job.get("batch_size") != 4 or job.get("attempt_limit") != 1 or job.get("retry_count") != 0:
            raise R13PreparationValidationError("R13 ledger batch/retry semantics differ")
    if [job.get("required_steps") for job in training_jobs] != [7500, 7500] or [job.get("required_steps") for job in probe_jobs] != [8, 8]:
        raise R13PreparationValidationError("R13 ledger optimizer-step counts differ")


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise R13PreparationValidationError(f"{label} must be positive and finite")
    return float(value)


def _post_probe() -> dict[str, Any]:
    roots = {arm: REPO_ROOT / f"artifacts/r13_control_lpl_training/probes_v1/{arm}" for arm in ("control", "lpl")}
    metrics = {arm: _json(root / "last_metrics.json") for arm, root in roots.items()}
    if any(value.get("global_step") != 8 for value in metrics.values()):
        raise R13PreparationValidationError("R13 probe did not complete exactly 8 optimizer steps")
    _finite_positive(metrics["lpl"].get("r13_cumulative_active_rows"), "LPL cumulative active rows")
    _finite_positive(metrics["lpl"].get("latent_perceptual_loss_raw"), "LPL loss")
    control_rng = (roots["control"] / "flow_rng_ledger.jsonl").read_bytes()
    lpl_rng = (roots["lpl"] / "flow_rng_ledger.jsonl").read_bytes()
    if control_rng != lpl_rng or len(control_rng.splitlines()) != 8:
        raise R13PreparationValidationError("R13 probe flow RNG ledgers differ or do not contain 8 rows")
    return {"status": "probe_validated", "lpl_cumulative_active_rows": metrics["lpl"]["r13_cumulative_active_rows"]}


def _state_dict_equal(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    import torch

    if set(left) != set(right):
        raise R13PreparationValidationError(f"{label} keys differ")
    for name in sorted(left):
        if not torch.equal(left[name], right[name]):
            raise R13PreparationValidationError(f"{label} tensor differs: {name}")


def _post_training() -> dict[str, Any]:
    import torch

    roots = {arm: REPO_ROOT / f"artifacts/checkpoints/r13_{arm}_conditioning_1epoch_seed1337" for arm in ("control", "lpl")}
    rng = {arm: (root / "flow_rng_ledger.jsonl").read_bytes() for arm, root in roots.items()}
    if rng["control"] != rng["lpl"] or len(rng["control"].splitlines()) != 7500:
        raise R13PreparationValidationError("R13 full-run flow RNG ledgers differ or do not contain 7500 rows")
    start = {arm: torch.load(root / "step_00000000.pt", map_location="cpu", weights_only=True, mmap=True) for arm, root in roots.items()}
    _state_dict_equal(start["control"]["model_state_dict"], start["lpl"]["model_state_dict"], "R13 step0 model state")
    _state_dict_equal(start["control"]["ema_model_state_dict"], start["lpl"]["ema_model_state_dict"], "R13 step0 EMA state")
    final = {arm: torch.load(root / "last.pt", map_location="cpu", weights_only=True, mmap=True) for arm, root in roots.items()}
    for arm, checkpoint in final.items():
        if checkpoint.get("metrics", {}).get("global_step") != 7500:
            raise R13PreparationValidationError(f"R13 {arm} final checkpoint is not global_step 7500")
        contract = validate_r13_training_contract_payload(checkpoint.get("r13_training_contract"))
        if contract["arm_id"] != arm:
            raise R13PreparationValidationError(f"R13 {arm} final checkpoint arm declaration differs")
    return {"status": "training_validated", "flow_rng_rows": 7500}


def validate(preparation: Path, *, post_probe: bool = False, post_training: bool = False) -> dict[str, Any]:
    summary = _json(preparation / "preparation_summary.json")
    if summary.get("status") != "prepared_not_launched" or summary.get("probe_launched") is not False or summary.get("training_launched") is not False:
        raise R13PreparationValidationError("R13 preparation summary launch status differs")
    source = _json(preparation / "e15_ema_start_binding.json")
    if source.get("checkpoint_path") != R13_SOURCE_CHECKPOINT or source.get("checkpoint_sha256") != R13_SOURCE_CHECKPOINT_SHA256 or source.get("checkpoint_model") != "ema":
        raise R13PreparationValidationError("R13 E15 EMA source binding differs")
    full = {arm: _yaml(path) for arm, path in FULL_CONFIGS.items()}
    probes = {arm: _yaml(path) for arm, path in PROBE_CONFIGS.items()}
    for config in [*full.values(), *probes.values()]:
        _validate_train_g_config(config)
    _assert_arm_equality(full["control"], full["lpl"], probe=False)
    _assert_arm_equality(probes["control"], probes["lpl"], probe=True)
    expected_probe_bindings = {
        "control": {"contract_type": "safa_r13_disposable_probe_resource_binding_v1", "physical_gpu_index": 2, "physical_gpu_uuid": "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02"},
        "lpl": {"contract_type": "safa_r13_disposable_probe_resource_binding_v1", "physical_gpu_index": 1, "physical_gpu_uuid": "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1"},
    }
    if {arm: config.get("r13_resource_binding") for arm, config in probes.items()} != expected_probe_bindings:
        raise R13PreparationValidationError("R13 probe config resource bindings differ")
    recorded_probe_configs = {row["arm_id"]: row for row in summary.get("probe_configs", [])}
    if set(recorded_probe_configs) != set(PROBE_CONFIGS) or any(recorded_probe_configs[arm].get("sha256") != _sha256(path) for arm, path in PROBE_CONFIGS.items()):
        raise R13PreparationValidationError("R13 recorded probe config SHA-256 differs")
    _validate_order(full["control"])
    _validate_ledgers(preparation)
    for (arm, sample_set), path in EVAL_TEMPLATES.items():
        _validate_eval_template(path, arm, sample_set)
    result = {
        "contract_type": "safa_r13_control_lpl_preparation_validation_v1",
        "status": "validated_prepared_not_launched",
        "train_order_rows": 30000,
        "full_required_steps": 7500,
        "probe_required_steps": 8,
        "full_batch_size": 4,
    }
    if post_probe:
        result["post_probe"] = _post_probe()
    if post_training:
        result["post_training"] = _post_training()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only validation of the R13 control/LPL preparation.")
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--post-probe", action="store_true")
    parser.add_argument("--post-training", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.preparation.resolve(), post_probe=args.post_probe, post_training=args.post_training), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
