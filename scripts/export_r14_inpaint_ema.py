#!/usr/bin/env python3
"""Export the only R14 256-step EMA without optimizer state."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch


CONTRACT = "safa_r14_face_region_inpaint_feasibility_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def main() -> None:
    args = parse_args()
    for output in (args.output, args.metadata_output):
        if output.exists():
            raise FileExistsError(f"refusing to replace existing EMA export: {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    payload = _mapping(checkpoint, "checkpoint")
    metrics = _mapping(payload.get("metrics"), "checkpoint.metrics")
    if metrics.get("global_step") != 256:
        raise RuntimeError(f"R14 checkpoint must end at optimizer step 256, got {metrics.get('global_step')!r}")
    model_config = _mapping(payload.get("model_config"), "checkpoint.model_config")
    if model_config.get("model_type") != "meanflow_sit_inpaint":
        raise RuntimeError("R14 checkpoint model_type must be meanflow_sit_inpaint")
    training_config = _mapping(payload.get("training_config"), "checkpoint.training_config")
    if training_config.get("r14_contract") != CONTRACT:
        raise RuntimeError("R14 checkpoint training contract differs")
    ema = _mapping(payload.get("ema_model_state_dict"), "checkpoint.ema_model_state_dict")
    if not ema:
        raise RuntimeError("R14 checkpoint has an empty EMA state")
    for name, value in ema.items():
        if not isinstance(name, str) or not name or not torch.is_tensor(value):
            raise RuntimeError("R14 EMA state has an invalid entry")
        if torch.is_floating_point(value) and not torch.isfinite(value).all().item():
            raise FloatingPointError(f"R14 EMA tensor is non-finite: {name}")
    required_context = {
        "vector_field.context_embedder.weight",
        "vector_field.context_embedder.bias",
    }
    if not required_context.issubset(ema):
        raise RuntimeError("trained R14 EMA is missing the inpaint context projection")
    export = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_ema_export_v1",
        "checkpoint_model": "ema",
        "source_checkpoint": str(args.checkpoint),
        "stage": payload.get("stage"),
        "metrics": dict(metrics),
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "r14_training_contract": payload.get("r14_training_contract"),
        "ema_model_state_dict": dict(ema),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, args.output)
    if not args.output.is_file() or args.output.stat().st_size <= 0:
        raise RuntimeError("R14 EMA export was not materialized")
    metadata = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_ema_export_metadata_v1",
        "checkpoint_model": "ema",
        "optimizer_steps": 256,
        "source_checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "state_tensor_count": len(ema),
        "size_bytes": args.output.stat().st_size,
    }
    if not math.isfinite(float(metadata["size_bytes"])):
        raise RuntimeError("R14 EMA export size is invalid")
    with args.metadata_output.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
