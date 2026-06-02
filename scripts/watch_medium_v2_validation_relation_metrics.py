#!/usr/bin/env python3
"""Watch checkpoints and compute validation relation metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safa.evaluation.metrics import (
    DEFAULT_DENSE_GRAM_MAX_SAMPLES,
    validate_dense_gram_cap,
    validate_dense_gram_sample_count,
    validate_dense_gram_sample_limit,
)

DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints/g_medium_v2_stage2_m2_gram_weighted")
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "last.pt"
DEFAULT_CONFIG = Path("configs/medium_v2/train_g_medium_v2_stage2_m2_gram_weighted.yaml")
DEFAULT_INDEX = Path("data/index/val_single_face.jsonl")
DEFAULT_FEATURES = Path("artifacts/e0_features/val_single_face_e0_medium_v1")
DEFAULT_OUT_DIR = Path("artifacts/metrics/medium_v2/validation_relation_metrics")
DEFAULT_SUMMARY = DEFAULT_OUT_DIR / "validation_relation_metrics_summary.json"
DEFAULT_EVENTS = DEFAULT_OUT_DIR / "validation_relation_metrics_events.jsonl"
DEFAULT_STATE = DEFAULT_OUT_DIR / "validation_relation_metrics_state.json"
DEFAULT_LOG = DEFAULT_OUT_DIR / "validation_relation_metrics.log"


class WatcherPaths(NamedTuple):
    checkpoints: tuple[Path, ...] = (DEFAULT_CHECKPOINT,)
    config: Path = DEFAULT_CONFIG
    index: Path = DEFAULT_INDEX
    features: Path = DEFAULT_FEATURES
    out_dir: Path = DEFAULT_OUT_DIR
    summary: Path = DEFAULT_SUMMARY
    events: Path = DEFAULT_EVENTS
    state: Path = DEFAULT_STATE
    log: Path = DEFAULT_LOG


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            detail = row.get("checkpoint") or row.get("summary") or row.get("error") or ""
            handle.write(f"{row['time']} {row['type']}: {detail}\n")


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def wait_for_stable_checkpoint(path: Path, *, stable_seconds: float, check_interval: float) -> None:
    if stable_seconds < 0:
        raise ValueError(f"stable_seconds must be non-negative, got {stable_seconds!r}")
    if check_interval <= 0:
        raise ValueError(f"check_interval must be positive, got {check_interval!r}")
    last_signature: tuple[int, int] | None = None
    stable_since: float | None = None
    while True:
        if not path.is_file():
            time.sleep(check_interval)
            continue
        stat = path.stat()
        signature = (int(stat.st_size), int(stat.st_mtime_ns))
        now = time.monotonic()
        if signature != last_signature:
            last_signature = signature
            stable_since = now
        elif stable_since is not None and now - stable_since >= stable_seconds:
            return
        if stable_seconds == 0:
            return
        time.sleep(check_interval)


def load_checkpoint_with_retries(path: Path, *, retries: int, retry_interval: float):
    if isinstance(retries, bool) or int(retries) <= 0:
        raise ValueError(f"retries must be a positive integer, got {retries!r}")
    if retry_interval < 0:
        raise ValueError(f"retry_interval must be non-negative, got {retry_interval!r}")
    import torch

    attempts = int(retries)
    for attempt in range(1, attempts + 1):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            if attempt == attempts:
                raise RuntimeError(f"failed to load checkpoint after {attempts} attempts: {path}: {type(exc).__name__}: {exc}") from exc
            time.sleep(retry_interval)
    raise RuntimeError("unreachable checkpoint load retry state")


def load_config(path: Path) -> dict[str, Any]:
    from safa.utils.config import load_yaml

    config = load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def model_sources(selection: str) -> tuple[str, ...]:
    if selection == "raw":
        return ("raw",)
    if selection == "ema":
        return ("ema",)
    if selection == "both":
        return ("raw", "ema")
    raise ValueError(f"model must be raw, ema, or both, got {selection!r}")


def build_validation_loader(
    *,
    config: dict[str, Any],
    paths: WatcherPaths,
    model_config: dict[str, Any],
    max_samples: int | None,
    dense_gram_cap: int,
    batch_size: int | None,
    num_workers: int | None,
    device: str,
):
    from torch.utils.data import DataLoader, Subset

    from safa.data.feature_dataset import FeatureAlignedAffectNet
    from safa.training.transforms import generator_image_transform

    if "e0_checkpoint" not in config:
        raise ValueError(f"config missing e0_checkpoint: {paths.config}")
    validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
    configured_max_samples = int(validation.get("max_samples", 0))
    effective_max_samples = configured_max_samples if max_samples is None else int(max_samples)
    validate_dense_gram_sample_limit(
        effective_max_samples,
        dense_gram_cap=dense_gram_cap,
        context="validation relation watcher",
    )
    effective_batch_size = int(batch_size if batch_size is not None else validation.get("batch_size", config.get("batch_size", 8)))
    effective_num_workers = int(num_workers if num_workers is not None else config.get("num_workers", 0))
    if effective_batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {effective_batch_size!r}")
    if effective_num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {effective_num_workers!r}")

    dataset = FeatureAlignedAffectNet(
        paths.index,
        paths.features,
        config["e0_checkpoint"],
        transform=generator_image_transform(int(model_config["image_size"])),
    )
    if effective_max_samples > 0:
        dataset = Subset(dataset, list(range(min(effective_max_samples, len(dataset)))))
    loader_kwargs = {
        "batch_size": effective_batch_size,
        "shuffle": False,
        "num_workers": effective_num_workers,
        "pin_memory": str(device).startswith("cuda"),
        "persistent_workers": effective_num_workers > 0,
    }
    if effective_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


def load_e0_for_validation(config: dict[str, Any], *, device: str):
    from safa.models.e0 import freeze_e0, load_e0_checkpoint

    e0, _payload = load_e0_checkpoint(config["e0_checkpoint"], device=device)
    e0.to(device)
    freeze_e0(e0)
    return e0.eval()


def build_generator_variant(payload: dict[str, Any], checkpoint_path: Path, *, source: str, device: str):
    from safa.models.generator import build_generator, require_generator_model_config

    model_config = require_generator_model_config(payload, str(checkpoint_path))
    state_key = "ema_model_state_dict" if source == "ema" else "model_state_dict"
    state_dict = payload.get(state_key)
    if state_dict is None:
        raise ValueError(f"{checkpoint_path} requested {source} weights but missing {state_key}")
    generator = build_generator(model_config).to(device)
    generator.load_state_dict(state_dict)
    return generator.eval(), model_config


def evaluate_relation_metrics(
    *,
    generator,
    e0,
    loader,
    device: str,
    model_config: dict[str, Any],
    sampling_seed: int,
    dense_gram_cap: int,
) -> dict[str, float]:
    import torch

    from safa.evaluation.metrics import compute_validation_relation_metrics
    from safa.training.losses import normalize_for_e0
    from safa.utils.device import assert_finite_tensor
    from safa.utils.sampling import make_x_init_for_sample_ids

    generator.eval()
    e0.eval()
    target_chunks = []
    generated_chunks = []
    total = 0
    image_size = int(model_config["image_size"])
    sample_steps = int(model_config.get("sample_steps", 32))
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device, non_blocking=True)
            sample_ids = list(batch["sample_id"])
            x_init = make_x_init_for_sample_ids(sample_ids, sampling_seed, image_size, z.device, z.dtype)
            generated = generator.sample(z, steps=sample_steps, x_init=x_init)
            assert_finite_tensor("validation_relation_generated_image", generated)
            generated_out = e0(normalize_for_e0(generated))
            if "embedding" not in generated_out:
                raise RuntimeError("E0 output missing embedding for validation relation metrics")
            target_chunks.append(z.detach().to(dtype=torch.float32).cpu())
            generated_chunks.append(generated_out["embedding"].detach().to(dtype=torch.float32).cpu())
            total += int(z.shape[0])
    if total == 0:
        raise ValueError("validation dataset produced zero samples")
    validate_dense_gram_sample_count(total, dense_gram_cap=dense_gram_cap, context="validation relation watcher")
    metrics = compute_validation_relation_metrics(
        torch.cat(generated_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
    )
    metrics["sample_count"] = float(total)
    return metrics


def compute_checkpoint_relation_metrics(
    *,
    checkpoint_path: Path,
    paths: WatcherPaths,
    model: str,
    device: str,
    max_samples: int | None,
    dense_gram_cap: int,
    batch_size: int | None,
    num_workers: int | None,
    stable_seconds: float,
    stable_check_interval: float,
    load_retries: int,
    load_retry_interval: float,
) -> dict[str, Any]:
    from safa.models.generator import require_generator_model_config
    from safa.utils.sampling import sampling_base_seed_from_config

    wait_for_stable_checkpoint(checkpoint_path, stable_seconds=stable_seconds, check_interval=stable_check_interval)
    checkpoint = load_checkpoint_with_retries(checkpoint_path, retries=load_retries, retry_interval=load_retry_interval)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {checkpoint_path}")
    config = load_config(paths.config)
    model_config = require_generator_model_config(checkpoint, str(checkpoint_path))
    loader = build_validation_loader(
        config=config,
        paths=paths,
        model_config=model_config,
        max_samples=max_samples,
        dense_gram_cap=dense_gram_cap,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    e0 = load_e0_for_validation(config, device=device)
    sampling_seed = int(sampling_base_seed_from_config(config))
    variants: dict[str, Any] = {}
    for source in model_sources(model):
        generator, variant_model_config = build_generator_variant(checkpoint, checkpoint_path, source=source, device=device)
        variants[source] = evaluate_relation_metrics(
            generator=generator,
            e0=e0,
            loader=loader,
            device=device,
            model_config=variant_model_config,
            sampling_seed=sampling_seed,
            dense_gram_cap=dense_gram_cap,
        )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_stat": checkpoint_stat(checkpoint_path),
        "stage": _jsonable_scalar(checkpoint.get("stage")),
        "stage_epoch_1based": _checkpoint_stage_epoch(checkpoint.get("metrics")),
        "model": model,
        "variants": variants,
    }


def checkpoint_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def checkpoint_state_key(checkpoint_path: Path, stat: dict[str, int], *, model_source: str, stage_epoch_1based: int | None) -> str:
    payload = {
        "checkpoint": str(checkpoint_path),
        "size_bytes": int(stat["size_bytes"]),
        "mtime_ns": int(stat["mtime_ns"]),
        "model_source": str(model_source),
        "stage_epoch_1based": None if stage_epoch_1based is None else int(stage_epoch_1based),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checkpoint_state_record(
    checkpoint_path: Path,
    stat: dict[str, int],
    *,
    model_source: str,
    stage_epoch_1based: int | None,
) -> dict[str, Any]:
    return {
        "key": checkpoint_state_key(
            checkpoint_path,
            stat,
            model_source=model_source,
            stage_epoch_1based=stage_epoch_1based,
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_stat": {"size_bytes": int(stat["size_bytes"]), "mtime_ns": int(stat["mtime_ns"])},
        "model_source": str(model_source),
        "stage_epoch_1based": None if stage_epoch_1based is None else int(stage_epoch_1based),
        "time": utc_now(),
    }


def checkpoint_state_records_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    variants = result.get("variants")
    stat = result.get("checkpoint_stat")
    if not isinstance(variants, dict) or not isinstance(stat, dict):
        return []
    checkpoint_path = Path(str(result["checkpoint"]))
    stage_epoch = result.get("stage_epoch_1based")
    if isinstance(stage_epoch, bool):
        stage_epoch = None
    return [
        checkpoint_state_record(
            checkpoint_path,
            stat,
            model_source=str(model_source),
            stage_epoch_1based=stage_epoch if stage_epoch is None else int(stage_epoch),
        )
        for model_source in sorted(variants)
    ]


def _processed_checkpoint_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records = state.get("processed_checkpoints")
    if not isinstance(records, list):
        return []
    parsed = []
    for record in records:
        if not isinstance(record, dict):
            continue
        key = record.get("key")
        stat = record.get("checkpoint_stat")
        if not isinstance(key, str) or not isinstance(stat, dict):
            continue
        if "checkpoint" not in record or "model_source" not in record:
            continue
        try:
            parsed.append(
                {
                    "key": key,
                    "checkpoint": str(record["checkpoint"]),
                    "checkpoint_stat": {"size_bytes": int(stat["size_bytes"]), "mtime_ns": int(stat["mtime_ns"])},
                    "model_source": str(record["model_source"]),
                    "stage_epoch_1based": record.get("stage_epoch_1based"),
                    "time": str(record.get("time", "")),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


def _merge_checkpoint_state_records(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for record in group:
            key = record.get("key")
            if isinstance(key, str):
                merged[key] = record
    return [merged[key] for key in sorted(merged)]


def _checkpoint_processed_in_state(state: dict[str, Any], checkpoint_path: Path, stat: dict[str, int], *, model: str) -> bool:
    records = _processed_checkpoint_records(state)
    required_sources = set(model_sources(model))
    seen_sources = set()
    for record in records:
        record_stat = record["checkpoint_stat"]
        if record["checkpoint"] != str(checkpoint_path):
            continue
        if int(record_stat["size_bytes"]) != int(stat["size_bytes"]):
            continue
        if int(record_stat["mtime_ns"]) != int(stat["mtime_ns"]):
            continue
        source = str(record["model_source"])
        if source in required_sources:
            seen_sources.add(source)
    return seen_sources == required_sources


def _checkpoint_stage_epoch(metrics: Any) -> int | None:
    if not isinstance(metrics, dict):
        return None
    for field in ("stage_epoch_1based", "stage_epoch"):
        value = metrics.get(field)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _jsonable_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_state(
    paths: WatcherPaths,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    new_processed_checkpoints: list[dict[str, Any]] | None = None,
    skipped_checkpoint_count: int = 0,
) -> None:
    existing_state = read_state(paths.state)
    processed_checkpoints = _merge_checkpoint_state_records(
        _processed_checkpoint_records(existing_state),
        new_processed_checkpoints or [],
    )
    payload = {
        "time": utc_now(),
        "summary": str(paths.summary),
        "config": str(paths.config),
        "index": str(paths.index),
        "features": str(paths.features),
        "checkpoints": [str(path) for path in paths.checkpoints],
        "new_event_count": len(events),
        "result_count": len(summary.get("results", [])),
        "skipped_checkpoint_count": int(skipped_checkpoint_count),
        "processed_checkpoints": processed_checkpoints,
    }
    write_json_atomic(paths.state, payload)
    append_jsonl(paths.events, events)
    append_log(paths.log, events)


def run_once(
    paths: WatcherPaths,
    *,
    model: str = "both",
    device: str = "cuda:0",
    max_samples: int | None = None,
    dense_gram_cap: int = DEFAULT_DENSE_GRAM_MAX_SAMPLES,
    batch_size: int | None = None,
    num_workers: int | None = None,
    stable_seconds: float = 15.0,
    stable_check_interval: float = 2.0,
    load_retries: int = 5,
    load_retry_interval: float = 5.0,
    skip_unchanged: bool = False,
) -> int:
    results = []
    events = []
    new_processed_checkpoints = []
    skipped_checkpoint_count = 0
    state = read_state(paths.state) if skip_unchanged else {}
    for checkpoint_path in paths.checkpoints:
        if skip_unchanged and checkpoint_path.is_file():
            wait_for_stable_checkpoint(
                checkpoint_path,
                stable_seconds=stable_seconds,
                check_interval=stable_check_interval,
            )
            stat = checkpoint_stat(checkpoint_path)
            if _checkpoint_processed_in_state(state, checkpoint_path, stat, model=model):
                skipped_checkpoint_count += 1
                continue
        result = compute_checkpoint_relation_metrics(
            checkpoint_path=checkpoint_path,
            paths=paths,
            model=model,
            device=device,
            max_samples=max_samples,
            dense_gram_cap=dense_gram_cap,
            batch_size=batch_size,
            num_workers=num_workers,
            stable_seconds=stable_seconds,
            stable_check_interval=stable_check_interval,
            load_retries=load_retries,
            load_retry_interval=load_retry_interval,
        )
        results.append(result)
        events.append(
            {
                "time": utc_now(),
                "type": "validation_relation_metrics_computed",
                "checkpoint": str(checkpoint_path),
                "summary": str(paths.summary),
                "model": model,
                "variants": sorted(result["variants"].keys()),
            }
        )
        new_processed_checkpoints.extend(checkpoint_state_records_from_result(result))
    if not results:
        return 0
    summary = {
        "created_at": utc_now(),
        "model": model,
        "device": device,
        "config": str(paths.config),
        "index": str(paths.index),
        "features": str(paths.features),
        "results": results,
    }
    write_json_atomic(paths.summary, summary)
    write_state(
        paths,
        summary,
        events,
        new_processed_checkpoints=new_processed_checkpoints,
        skipped_checkpoint_count=skipped_checkpoint_count,
    )
    return 0


def loop(paths: WatcherPaths, **kwargs) -> int:
    interval_seconds = float(kwargs.pop("interval_seconds"))
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds!r}")
    validate_dense_gram_sample_limit(
        kwargs.get("max_samples"),
        dense_gram_cap=int(kwargs.get("dense_gram_cap", DEFAULT_DENSE_GRAM_MAX_SAMPLES)),
        context="validation relation watcher loop",
    )
    while True:
        try:
            run_once(paths, skip_unchanged=True, **kwargs)
        except Exception as exc:
            event = {"time": utc_now(), "type": "validation_relation_metrics_error", "error": f"{type(exc).__name__}: {exc}"}
            append_jsonl(paths.events, [event])
            append_log(paths.log, [event])
        time.sleep(interval_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch medium v2 checkpoints and compute validation relation metrics.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--run-name", default="validation_relation_metrics")
    parser.add_argument("--model", choices=("raw", "ema", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dense-gram-cap", type=int, default=DEFAULT_DENSE_GRAM_MAX_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stable-seconds", type=float, default=15.0)
    parser.add_argument("--stable-check-interval", type=float, default=2.0)
    parser.add_argument("--load-retries", type=int, default=5)
    parser.add_argument("--load-retry-interval", type=float, default=5.0)
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    dense_gram_cap = validate_dense_gram_cap(args.dense_gram_cap, context="validation relation watcher dense Gram cap")
    if not args.once and args.max_samples is None:
        raise ValueError("--max-samples is required for validation relation watcher loop mode")
    if args.max_samples is not None:
        validate_dense_gram_sample_limit(
            int(args.max_samples),
            dense_gram_cap=dense_gram_cap,
            context="validation relation watcher",
        )


def resolve_paths(args: argparse.Namespace) -> WatcherPaths:
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    run_name = str(args.run_name)
    checkpoints = tuple(Path(path) for path in args.checkpoint) if args.checkpoint else (checkpoint_dir / "last.pt",)
    return WatcherPaths(
        checkpoints=checkpoints,
        config=Path(args.config),
        index=Path(args.index),
        features=Path(args.features),
        out_dir=out_dir,
        summary=Path(args.summary) if args.summary is not None else out_dir / f"{run_name}_summary.json",
        events=Path(args.events) if args.events is not None else out_dir / f"{run_name}_events.jsonl",
        state=Path(args.state) if args.state is not None else out_dir / f"{run_name}_state.json",
        log=Path(args.log) if args.log is not None else out_dir / f"{run_name}.log",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli_args(args)
    paths = resolve_paths(args)
    common_kwargs = {
        "model": args.model,
        "device": args.device,
        "max_samples": args.max_samples,
        "dense_gram_cap": args.dense_gram_cap,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "stable_seconds": args.stable_seconds,
        "stable_check_interval": args.stable_check_interval,
        "load_retries": args.load_retries,
        "load_retry_interval": args.load_retry_interval,
    }
    if args.once:
        return run_once(paths, **common_kwargs)
    return loop(paths, interval_seconds=args.interval, **common_kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
