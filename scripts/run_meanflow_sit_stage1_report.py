#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Sequence

import yaml


DEFAULT_OUTPUT_JSON = Path("artifacts/reports/e11_meanflow_sit_stage1_report.json")
DEFAULT_OUTPUT_MD = Path("artifacts/reports/e11_meanflow_sit_stage1_report.md")
DEFAULT_TRAIN_SESSION = "safa_e11_meanflow_sit_k100_200ep_20260613_053015"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LOG_TIMESTAMP_RE = re.compile(r"(20\d\d-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)")


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    config: Path
    checkpoint_dir: Path
    quality_dir: Path
    log_glob: str | None = None


DEFAULT_RUNS: dict[str, RunSpec] = {
    "e11": RunSpec(
        name="e11",
        label="MeanFlow-SiT-B/4 latent Stage1",
        config=Path("configs/medium_v2/experiments/e11_meanflow_sit_b_stage1_200ep.yaml"),
        checkpoint_dir=Path("artifacts/checkpoints/e11_meanflow_sit_b_stage1_200ep"),
        quality_dir=Path("artifacts/eval/e11_meanflow_sit_b_stage1_200ep/quality"),
        log_glob="artifacts/logs/e11_meanflow_sit_k100_200ep_*.log",
    ),
    "e8": RunSpec(
        name="e8",
        label="5M FM E8 baseline",
        config=Path("configs/medium_v2/experiments/e8_fm_only_200ep.yaml"),
        checkpoint_dir=Path("artifacts/checkpoints/e8_fm_only_200ep"),
        quality_dir=Path("artifacts/eval/e8_fm_only_200ep/quality"),
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize e11 MeanFlow-SiT Stage1 progress against the e8 5M FM baseline.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", nargs="+", choices=tuple(DEFAULT_RUNS), default=["e11", "e8"])
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--train-session", default=DEFAULT_TRAIN_SESSION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root)
    train_session = str(args.train_session or "")
    session_alive = tmux_session_exists(train_session) if train_session else False
    runs = {
        name: summarize_run(repo_root, DEFAULT_RUNS[name], session_alive=session_alive if name == "e11" else False)
        for name in args.runs
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "train_session": train_session,
        "train_session_alive": session_alive,
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "runs": runs,
    }


def summarize_run(repo_root: Path, spec: RunSpec, *, session_alive: bool) -> dict[str, Any]:
    config_path = _resolve(repo_root, spec.config)
    checkpoint_dir = _resolve(repo_root, spec.checkpoint_dir)
    quality_dir = _resolve(repo_root, spec.quality_dir)
    metrics_path = checkpoint_dir / "metrics_history.jsonl"
    config = _load_yaml_if_exists(config_path)
    records = _read_jsonl_if_exists(metrics_path)
    latest = records[-1] if records else {}
    target_epochs = _target_epochs(config)
    latest_epoch = _latest_epoch(latest, len(records))
    status = _run_status(config_path, metrics_path, latest_epoch, target_epochs, session_alive)
    latest_quality_dir = quality_dir / f"epoch_{latest_epoch:04d}" if latest_epoch else None
    log_path = _latest_log(repo_root, spec.log_glob) if spec.log_glob else None

    return {
        "name": spec.name,
        "label": spec.label,
        "status": status,
        "config_path": str(spec.config),
        "metrics_history": str(spec.checkpoint_dir / "metrics_history.jsonl"),
        "metrics_count": len(records),
        "latest_epoch": latest_epoch,
        "target_epochs": target_epochs,
        "metrics": {
            "raw": _model_metrics(records, latest, "raw"),
            "ema": _model_metrics(records, latest, "ema"),
        },
        "loss": _loss_metrics(latest),
        "checkpoint_paths": _checkpoint_paths(checkpoint_dir, spec.checkpoint_dir),
        "samples": _sample_summary(latest_quality_dir, spec.quality_dir, latest_epoch),
        "quality_files": _quality_files(latest_quality_dir, spec.quality_dir, latest_epoch),
        "sampling": _sampling_summary(config),
        "training_time": _training_time_summary(log_path, [metrics_path, checkpoint_dir / "last.pt", checkpoint_dir / "last_metrics.json"]),
        "files": {
            "config_exists": config_path.is_file(),
            "metrics_history_exists": metrics_path.is_file(),
            "quality_dir_exists": quality_dir.is_dir(),
        },
    }


def write_report(summary: dict[str, Any], output_json: Path, output_md: Path, repo_root: Path) -> tuple[Path, Path]:
    json_path = _resolve(repo_root, output_json)
    md_path = _resolve(repo_root, output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# MeanFlow-SiT Stage1 Report", ""]
    lines.append(f"generated_at: `{summary['generated_at']}`")
    lines.append(f"train_session_alive: `{summary['train_session_alive']}`")
    lines.append("")
    lines.append("| run | status | epoch | raw face | raw zero | ema face | ema zero | ema NIQE | ema FID | ema KID | samples | checkpoint |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, run in summary["runs"].items():
        raw = run["metrics"]["raw"]
        ema = run["metrics"]["ema"]
        checkpoint = run["checkpoint_paths"]["best_stage2"]
        checkpoint_text = checkpoint["path"] if checkpoint["exists"] else "missing"
        lines.append(
            "| {name} | {status} | {epoch}/{target} | {raw_face} | {raw_zero} | {ema_face} | {ema_zero} | {niqe} | {fid} | {kid} | {samples} | `{checkpoint}` |".format(
                name=name,
                status=run["status"],
                epoch=run["latest_epoch"] or 0,
                target=run["target_epochs"] or "?",
                raw_face=_fmt(raw.get("face_detection_rate")),
                raw_zero=_fmt(raw.get("zero_face_rate")),
                ema_face=_fmt(ema.get("face_detection_rate")),
                ema_zero=_fmt(ema.get("zero_face_rate")),
                niqe=_fmt(ema.get("niqe")),
                fid=_fmt(ema.get("fid")),
                kid=_fmt(ema.get("kid_mean")),
                samples=run["samples"].get("generated_count") or 0,
                checkpoint=checkpoint_text,
            )
        )
    lines.append("")
    lines.append("## Sampling")
    for name, run in summary["runs"].items():
        sampling = run["sampling"]
        lines.append(
            f"- {name}: model={sampling.get('model_type')}, nfe={sampling.get('nfe')}, "
            f"sample_steps={sampling.get('sample_steps')}, sampler={sampling.get('sampler')}, "
            f"data_space={sampling.get('data_space')}."
        )
    lines.append("")
    lines.append("## Artifacts")
    for name, run in summary["runs"].items():
        lines.append(f"- {name}: samples=`{run['samples'].get('generated_dir')}`, metrics=`{run['metrics_history']}`")
    return "\n".join(lines) + "\n"


def tmux_session_exists(session: str) -> bool:
    if not session:
        return False
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _run_status(config_path: Path, metrics_path: Path, latest_epoch: int | None, target_epochs: int | None, session_alive: bool) -> str:
    if not config_path.is_file():
        return "missing_config"
    if not metrics_path.is_file() or latest_epoch is None:
        return "missing_metrics"
    if target_epochs is not None and latest_epoch >= target_epochs:
        return "complete"
    return "running" if session_alive else "partial"


def _model_metrics(records: list[dict[str, Any]], latest: dict[str, Any], model: str) -> dict[str, Any]:
    prefix = f"validation_{model}_"
    return {
        "face_detection_rate": _first_number(latest, (f"{prefix}face_detection_rate", "validation_face_detection_rate" if model == "raw" else "")),
        "single_face_eq1_rate": _first_number(latest, (f"{prefix}single_face_eq1_rate", "validation_single_face_eq1_rate" if model == "raw" else "")),
        "zero_face_rate": _first_number(latest, (f"{prefix}zero_face_rate", "validation_zero_face_rate" if model == "raw" else "")),
        "niqe": _latest_number(records, (f"quality_{model}_niqe", f"quality_{model}_niqe_mean", "quality_niqe" if model == "raw" else "")),
        "fid": _latest_number(records, (f"quality_{model}_fid", "quality_fid" if model == "raw" else "")),
        "kid_mean": _latest_number(records, (f"quality_{model}_kid_mean", "quality_kid_mean" if model == "raw" else "")),
        "kid_std": _latest_number(records, (f"quality_{model}_kid_std", "quality_kid_std" if model == "raw" else "")),
    }


def _loss_metrics(latest: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": _first_number(latest, ("loss",)),
        "flow_matching_mse": _first_number(latest, ("flow_matching_mse",)),
        "grad_norm": _first_number(latest, ("grad_norm",)),
    }


def _checkpoint_paths(checkpoint_dir: Path, display_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(display_dir / filename), "exists": (checkpoint_dir / filename).is_file()}
        for name, filename in (
            ("best_stage2", "best_stage2.pt"),
            ("best", "best.pt"),
            ("last", "last.pt"),
            ("last_metrics", "last_metrics.json"),
        )
    }


def _sample_summary(latest_quality_dir: Path | None, display_quality_dir: Path, latest_epoch: int | None) -> dict[str, Any]:
    if latest_quality_dir is None or latest_epoch is None:
        return {"latest_epoch": None, "generated_dir": None, "generated_count": 0}
    generated_dir = latest_quality_dir / "generated_images"
    count = 0
    if generated_dir.is_dir():
        count = sum(1 for path in generated_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return {
        "latest_epoch": latest_epoch,
        "generated_dir": str(display_quality_dir / f"epoch_{latest_epoch:04d}" / "generated_images"),
        "generated_count": count,
    }


def _quality_files(latest_quality_dir: Path | None, display_quality_dir: Path, latest_epoch: int | None) -> list[str]:
    if latest_quality_dir is None or latest_epoch is None or not latest_quality_dir.is_dir():
        return []
    display_epoch_dir = display_quality_dir / f"epoch_{latest_epoch:04d}"
    return [str(display_epoch_dir / path.name) for path in sorted(latest_quality_dir.glob("*.json"))]


def _sampling_summary(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    generator = config.get("generator") if isinstance(config.get("generator"), dict) else {}
    model_type = generator.get("model_type")
    sample_steps = generator.get("sample_steps")
    nfe = 1 if model_type == "meanflow_sit" else sample_steps
    stage2 = _stage2_config(config)
    objective = stage2.get("stage2_objective") if isinstance(stage2.get("stage2_objective"), dict) else {}
    return {
        "model_type": model_type,
        "sample_steps": sample_steps,
        "nfe": nfe,
        "sampler": generator.get("sampler"),
        "data_space": generator.get("sit_data_space") or ("latent" if config.get("latent_training") else "pixel"),
        "image_size": config.get("image_size"),
        "pixel_image_size": config.get("pixel_image_size"),
        "flow_condition": objective.get("flow_condition"),
        "amp": config.get("amp"),
        "global_batch_size": config.get("global_batch_size"),
        "per_device_batch_size": config.get("per_device_batch_size"),
        "pretrained_path": generator.get("sit_pretrained_path"),
    }


def _training_time_summary(log_path: Path | None, artifact_paths: list[Path]) -> dict[str, Any]:
    start = _first_log_timestamp(log_path) if log_path else None
    latest_mtime = max((path.stat().st_mtime for path in artifact_paths if path.exists()), default=None)
    elapsed = latest_mtime - start.timestamp() if start is not None and latest_mtime is not None else None
    return {
        "log_path": str(log_path) if log_path is not None else None,
        "start_time": start.isoformat() if start is not None else None,
        "latest_artifact_time": _timestamp(latest_mtime),
        "elapsed_seconds_estimate": elapsed,
    }


def _first_log_timestamp(path: Path | None) -> datetime | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = LOG_TIMESTAMP_RE.search(line)
            if match:
                return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}").astimezone()
    return None


def _latest_log(repo_root: Path, pattern: str | None) -> Path | None:
    if not pattern:
        return None
    candidates = sorted(repo_root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return candidates[-1] if candidates else None


def _target_epochs(config: dict[str, Any] | None) -> int | None:
    stage2 = _stage2_config(config or {})
    epochs = stage2.get("epochs")
    return int(epochs) if isinstance(epochs, int) else None


def _stage2_config(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("stages")
    if not isinstance(stages, dict):
        return {}
    stage2 = stages.get("stage2")
    return stage2 if isinstance(stage2, dict) else {}


def _latest_epoch(latest: dict[str, Any], count: int) -> int | None:
    for key in ("stage_epoch_1based", "epoch_1based"):
        value = latest.get(key)
        if isinstance(value, int):
            return value
    for key in ("stage_epoch", "stage_epoch_0based", "epoch"):
        value = latest.get(key)
        if isinstance(value, int):
            return value + 1
    return count or None


def _latest_number(records: list[dict[str, Any]], keys: tuple[str, ...]) -> float | int | None:
    for row in reversed(records):
        value = _first_number(row, keys)
        if value is not None:
            return value
    return None


def _first_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | int | None:
    for key in keys:
        if not key:
            continue
        value = row.get(key)
        if isinstance(value, Number) and not isinstance(value, bool):
            return value
    return None


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"metrics row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _load_yaml_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _timestamp(value: float | None) -> str | None:
    return datetime.fromtimestamp(value).astimezone().isoformat() if value is not None else None


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, Number) and not isinstance(value, bool) else "NA"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_report(args)
    if args.dry_run:
        print(render_markdown(summary), end="")
        return 0
    json_path, md_path = write_report(summary, args.output_json, args.output_md, Path(args.repo_root))
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
