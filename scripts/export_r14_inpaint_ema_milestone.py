#!/usr/bin/env python3
"""Export one R14 long-run EMA checkpoint without optimizer state."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch


CHECKPOINT_STEPS = (2560, 5120, 7680, 10240, 12800)
TRAINING_CONTRACT = "safa_r14_face_region_inpaint_feasibility_v1"
EXPORT_CONTRACT = "safa_r14_inpaint_ema_export_v1"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def export_ema(checkpoint_path: Path, output: Path, metadata_output: Path) -> dict[str, object]:
    for path in (output, metadata_output):
        if path.exists():
            raise FileExistsError(f"refusing to replace EMA export: {path}")
    payload = _mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True),
        "checkpoint",
    )
    metrics = _mapping(payload.get("metrics"), "checkpoint.metrics")
    step = metrics.get("global_step")
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        raise RuntimeError(f"checkpoint global_step must be one of {CHECKPOINT_STEPS}, got {step!r}")
    model_config = _mapping(payload.get("model_config"), "checkpoint.model_config")
    if model_config.get("model_type") != "meanflow_sit_inpaint":
        raise RuntimeError("checkpoint model_type must be meanflow_sit_inpaint")
    training_config = _mapping(payload.get("training_config"), "checkpoint.training_config")
    if training_config.get("r14_contract") != TRAINING_CONTRACT:
        raise RuntimeError("checkpoint training contract differs")
    ema = _mapping(payload.get("ema_model_state_dict"), "checkpoint.ema_model_state_dict")
    if not ema:
        raise RuntimeError("checkpoint EMA state is empty")
    for name, value in ema.items():
        if not isinstance(name, str) or not name or not torch.is_tensor(value):
            raise RuntimeError("checkpoint EMA state has an invalid entry")
        if torch.is_floating_point(value) and not torch.isfinite(value).all().item():
            raise FloatingPointError(f"checkpoint EMA tensor is non-finite: {name}")
    required_context = {
        "vector_field.context_embedder.weight",
        "vector_field.context_embedder.bias",
    }
    if not required_context.issubset(ema):
        raise RuntimeError("checkpoint EMA is missing the inpaint context projection")
    export = {
        "schema_version": 1,
        "contract_type": EXPORT_CONTRACT,
        "checkpoint_model": "ema",
        "source_checkpoint": str(checkpoint_path),
        "stage": payload.get("stage"),
        "metrics": dict(metrics),
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "r14_training_contract": payload.get("r14_training_contract"),
        "ema_model_state_dict": dict(ema),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, output)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("EMA export was not materialized")
    metadata = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_ema_export_metadata_v1",
        "checkpoint_model": "ema",
        "optimizer_steps": step,
        "source_checkpoint": str(checkpoint_path),
        "output": str(output),
        "state_tensor_count": len(ema),
        "size_bytes": output.stat().st_size,
    }
    if not math.isfinite(float(metadata["size_bytes"])):
        raise RuntimeError("EMA export size is invalid")
    with metadata_output.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return metadata


def main() -> None:
    args = _parse_args()
    print(json.dumps(export_ema(args.checkpoint, args.output, args.metadata_output), sort_keys=True))


if __name__ == "__main__":
    main()
