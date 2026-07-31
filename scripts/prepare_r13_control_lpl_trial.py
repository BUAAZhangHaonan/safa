#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from safa.training.g_loop import _validate_train_g_config
from safa.training.r13_training_contract import (
    R13_SOURCE_CHECKPOINT,
    R13_SOURCE_CHECKPOINT_SHA256,
    R13_TRAIN_ORDER_PATH,
    R13_TRAIN_ORDER_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/r13_control_lpl_training/preparation_v1"
INTERPRETER = Path("/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python")
TRAIN_INDEX = REPO_ROOT / "data/index/train_face_mixed_e14_4029avail.jsonl"
TRAIN_FEATURES = REPO_ROOT / "artifacts/e0_features/train_face_mixed_e14_e0_medium_v1"
GPU_UUIDS = (
    "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
    "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02",
    "GPU-61ea2925-9905-7f56-cd64-7a792a32efef",
)
FULL_CONFIGS = {
    "control": "configs/medium_v2/experiments/r13_control_conditioning_1epoch_seed1337.yaml",
    "lpl": "configs/medium_v2/experiments/r13_lpl_conditioning_1epoch_seed1337.yaml",
}
PROBE_CONFIGS = {
    "control": "configs/medium_v2/experiments/r13_probe_control_conditioning_seed1337.yaml",
    "lpl": "configs/medium_v2/experiments/r13_probe_lpl_conditioning_seed1337.yaml",
}
EVAL_TEMPLATES = (
    "configs/medium_v2/experiments/r13_eval_control_regular32.template.yaml",
    "configs/medium_v2/experiments/r13_eval_control_tail32.template.yaml",
    "configs/medium_v2/experiments/r13_eval_lpl_regular32.template.yaml",
    "configs/medium_v2/experiments/r13_eval_lpl_tail32.template.yaml",
)


class R13PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise R13PreparationError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise R13PreparationError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(value)
    return rows


def build_train_order_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(records) != 30000:
        raise R13PreparationError(f"R13 training index must contain 30000 rows, got {len(records)}")
    sample_ids = [record.get("sample_id") for record in records]
    if any(not isinstance(value, str) or not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise R13PreparationError("R13 training index sample IDs must be unique non-empty strings")
    indices = list(range(len(records)))
    random.Random(1337).shuffle(indices)
    return [
        {
            "batch_index": order_ordinal // 4,
            "batch_offset": order_ordinal % 4,
            "order_ordinal": order_ordinal,
            "sample_id": sample_ids[source_index],
            "source_index": source_index,
        }
        for order_ordinal, source_index in enumerate(indices)
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _load_config(relative_path: str) -> dict[str, Any]:
    value = yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R13PreparationError(f"config is not a mapping: {relative_path}")
    _validate_train_g_config(value)
    return value


def _config_binding(relative_path: str, value: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    return {
        "path": relative_path,
        "sha256": sha256(path),
        "arm_id": value["r13_arm_id"],
        "lpl_enabled": value["latent_perceptual_loss"]["enabled"],
        "required_steps": value["optimizer_step_contract"]["required_steps"],
        "batch_size": value["per_device_batch_size"],
        "out_dir": value["out_dir"],
    }


def _source_binding() -> dict[str, Any]:
    import torch

    path = REPO_ROOT / R13_SOURCE_CHECKPOINT
    if sha256(path) != R13_SOURCE_CHECKPOINT_SHA256:
        raise R13PreparationError("locked E15 checkpoint SHA-256 differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ema_state = checkpoint.get("ema_model_state_dict")
    model_state = checkpoint.get("model_state_dict")
    metrics = checkpoint.get("metrics")
    if not isinstance(ema_state, Mapping) or not isinstance(model_state, Mapping) or set(ema_state) != set(model_state):
        raise R13PreparationError("locked E15 raw/EMA state topology differs")
    if not isinstance(metrics, Mapping) or checkpoint.get("stage") != "stage2" or metrics.get("stage_epoch_1based") != 1652:
        raise R13PreparationError("locked E15 stage binding differs")
    return {
        "contract_type": "safa_r13_e15_ema_start_binding_v1",
        "checkpoint_path": R13_SOURCE_CHECKPOINT,
        "checkpoint_sha256": R13_SOURCE_CHECKPOINT_SHA256,
        "checkpoint_model": "ema",
        "source_stage": "stage2",
        "source_stage_epoch_1based": 1652,
        "state_tensor_count": len(ema_state),
        "initialization": {
            "generator": "load ema_model_state_dict",
            "ema": "fresh EMA clone of the loaded generator",
            "optimizer": "fresh AdamW with no restored state",
        },
    }


def _job(*, arm_id: str, config_path: str, physical_gpu: int, probe: bool) -> dict[str, Any]:
    config = yaml.safe_load((REPO_ROOT / config_path).read_text(encoding="utf-8"))
    out_dir = str(config["out_dir"])
    kind = "probe" if probe else "training"
    if probe:
        binding = config.get("r13_resource_binding")
        expected = {
            "contract_type": "safa_r13_disposable_probe_resource_binding_v1",
            "physical_gpu_index": physical_gpu,
            "physical_gpu_uuid": GPU_UUIDS[physical_gpu],
        }
        if binding != expected:
            raise R13PreparationError(f"R13 {arm_id} probe resource binding differs")
    return {
        "job_id": f"r13_{kind}_{arm_id}",
        "arm_id": arm_id,
        "kind": kind,
        "physical_gpu": {"index": physical_gpu, "uuid": GPU_UUIDS[physical_gpu]},
        "environment": {"CUDA_VISIBLE_DEVICES": GPU_UUIDS[physical_gpu]},
        "logical_device": "cuda:0",
        "tmux_session": f"safa-r13-{kind}-{arm_id}",
        "config": config_path,
        "argv": [str(INTERPRETER), "-m", "safa.cli.train_g", "--config", config_path],
        "log_path": f"artifacts/r13_control_lpl_training/logs_v1/{kind}_{arm_id}.log",
        "output_root": out_dir,
        "last_checkpoint": f"{out_dir}/last.pt",
        "required_steps": int(config["optimizer_step_contract"]["required_steps"]),
        "batch_size": 4,
        "attempt_limit": 1,
        "retry_count": 0,
    }


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R13 preparation output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    order_path = output / "train_order_seed1337.jsonl"
    order_rows = build_train_order_rows(read_jsonl(TRAIN_INDEX))
    write_jsonl(order_path, order_rows)
    if str(order_path.relative_to(REPO_ROOT)) != R13_TRAIN_ORDER_PATH or sha256(order_path) != R13_TRAIN_ORDER_SHA256:
        raise R13PreparationError("materialized R13 train order differs from the registered contract")

    full_values = {arm: _load_config(path) for arm, path in FULL_CONFIGS.items()}
    probe_values = {arm: _load_config(path) for arm, path in PROBE_CONFIGS.items()}
    source_binding = _source_binding()
    write_json(output / "e15_ema_start_binding.json", source_binding)

    training_jobs = [_job(arm_id="control", config_path=FULL_CONFIGS["control"], physical_gpu=0, probe=False), _job(arm_id="lpl", config_path=FULL_CONFIGS["lpl"], physical_gpu=1, probe=False)]
    for job in training_jobs:
        root = job["output_root"]
        job["checkpoint_paths"] = [
            f"{root}/step_{step:08d}.pt" for step in (0, 2500, 5000, 7500)
        ] + [job["last_checkpoint"]]
    write_json(
        output / "training_ledger.json",
        {
            "contract_type": "safa_r13_single_gpu_training_ledger_v1",
            "status": "prepared_not_launched",
            "jobs": training_jobs,
        },
    )
    probe_jobs = [_job(arm_id="control", config_path=PROBE_CONFIGS["control"], physical_gpu=2, probe=True), _job(arm_id="lpl", config_path=PROBE_CONFIGS["lpl"], physical_gpu=1, probe=True)]
    write_json(
        output / "probe_ledger.json",
        {
            "contract_type": "safa_r13_disposable_probe_ledger_v1",
            "status": "prepared_not_launched",
            "jobs": probe_jobs,
            "lpl_acceptance": {
                "global_step": 8,
                "r13_cumulative_active_rows": ">0",
                "latent_perceptual_loss_raw": ">0 and finite",
                "flow_rng_rows": 8,
            },
        },
    )
    write_json(
        output / "resource_contract.json",
        {
            "contract_type": "safa_r13_resource_admission_v1",
            "allowed_physical_gpus": [0, 1, 2],
            "training_gpu_bindings": {"control": 0, "lpl": 1},
            "probe_gpu_bindings": {"control": 2, "lpl": 1},
            "probe_and_training_are_sequential": True,
            "max_cpu_percent": 90.0,
            "max_ram_percent": 90.0,
            "swap_policy": "observe_only_main_memory_is_the_admission_gate",
            "swap_io_sample_seconds": 1.0,
            "maximum_swap_in_out_delta_bytes": 0,
            "max_gpu_memory_percent_after_launch": 85.0,
            "minimum_gpu_free_bytes_after_launch": 4 * 1024**3,
            "minimum_repo_filesystem_free_bytes": 24 * 1024**3,
            "batch_size": 4,
            "one_worker_per_gpu": True,
            "probe_before_training": True,
            "automatic_retry": False,
            "automatic_batch_change": False,
        },
    )
    write_json(
        output / "equality_contract.json",
        {
            "contract_type": "safa_r13_control_lpl_equality_v1",
            "allowed_config_differences": ["experiment_name", "r13_arm_id", "out_dir", "latent_perceptual_loss.enabled", "r13_resource_binding.physical_gpu_index", "r13_resource_binding.physical_gpu_uuid"],
            "required_equal_semantics": ["seed", "sampling_seed", "train_index", "train_features", "train_order_contract", "flow_matching_rng_contract", "optimizer_step_contract", "optimizer_checkpoint_contract", "optimizer_type", "learning_rate", "weight_decay", "generator", "ema", "stages"],
            "post_run_checks": {
                "flow_rng_ledger": "byte_for_byte_equal",
                "step_00000000_model_state": "byte_for_byte_equal",
                "step_00000000_ema_state": "byte_for_byte_equal",
                "trainable_parameter_names": "exact_registered_30_name_allowlist",
            },
        },
    )
    write_json(
        output / "evaluation_template_ledger.json",
        {
            "contract_type": "safa_r13_post_training_evaluation_templates_v1",
            "status": "template_not_launchable_until_sha_substitution",
            "templates": [
                {"path": path, "sha256": sha256(REPO_ROOT / path)} for path in EVAL_TEMPLATES
            ],
            "placeholders": {
                "control": "__R13_CONTROL_LAST_PT_SHA256__",
                "lpl": "__R13_LPL_LAST_PT_SHA256__",
            },
        },
    )
    summary = {
        "contract_type": "safa_r13_control_lpl_preparation_v1",
        "status": "prepared_not_launched",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "train_order": {
            "path": R13_TRAIN_ORDER_PATH,
            "sha256": R13_TRAIN_ORDER_SHA256,
            "sample_count": len(order_rows),
            "batch_count": len(order_rows) // 4,
            "seed": 1337,
        },
        "source_binding": source_binding,
        "full_configs": [_config_binding(FULL_CONFIGS[arm], full_values[arm]) for arm in ("control", "lpl")],
        "probe_configs": [_config_binding(PROBE_CONFIGS[arm], probe_values[arm]) for arm in ("control", "lpl")],
        "training_launched": False,
        "probe_launched": False,
    }
    write_json(output / "preparation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the locked R13 control/LPL trial without launching GPU work.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
