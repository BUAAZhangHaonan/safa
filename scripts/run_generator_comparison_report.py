#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Sequence

import yaml


DEFAULT_BASE_EVAL_CONFIG = Path("configs/medium_v2/eval_g_medium_v2_stage2_fm_only_baseline_k1_20260606.yaml")
DEFAULT_REAL_INDEX = Path("data/index/val_single_face.jsonl")
DEFAULT_CONFIG_DIR = Path("artifacts/eval/comparison_configs")
DEFAULT_SUMMARY_OUT = Path("artifacts/eval/generator_comparison_summary.json")
DEFAULT_MAX_GENERATED = 3969
DEFAULT_MAX_REAL = 3969
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "cuda:0"
DEFAULT_VISUAL_NUM_SAMPLES = 16


@dataclass(frozen=True)
class RunSpec:
    name: str
    checkpoint: Path
    out_dir: Path
    artifact_name: str


@dataclass(frozen=True)
class RunPlan:
    name: str
    checkpoint: Path
    out_dir: Path
    artifact_name: str
    eval_config: Path
    result_json: Path
    per_sample_jsonl: Path
    sample_dir: Path
    generated_image_dir: Path
    quality_json: Path
    visualization_path: Path
    commands: list[tuple[str, list[str]]]


@dataclass(frozen=True)
class ComparisonPlan:
    runs: list[RunPlan]
    base_eval_config: Path
    real_index: Path
    summary_out: Path
    max_generated: int
    max_real: int
    seed: int
    device: str
    python: str
    dry_run: bool


DEFAULT_RUNS: dict[str, RunSpec] = {
    "e8": RunSpec(
        name="e8",
        checkpoint=Path("artifacts/checkpoints/e8_fm_only_200ep/best_stage2.pt"),
        out_dir=Path("artifacts/eval/e8_fm_only_200ep/formal_baseline_k1"),
        artifact_name="e8_fm_only_200ep",
    ),
    "e9": RunSpec(
        name="e9",
        checkpoint=Path("artifacts/checkpoints/g_medium_v2_meanflow_200ep/best_stage2.pt"),
        out_dir=Path("artifacts/eval/g_medium_v2_meanflow_200ep/formal_baseline_k1"),
        artifact_name="g_medium_v2_meanflow_200ep",
    ),
    "e10": RunSpec(
        name="e10",
        checkpoint=Path("artifacts/checkpoints/g_medium_v2_ddim_200ep/best_stage2.pt"),
        out_dir=Path("artifacts/eval/g_medium_v2_ddim_200ep/formal_baseline_k1"),
        artifact_name="g_medium_v2_ddim_200ep",
    ),
}
DEFAULT_RUN_ORDER = tuple(DEFAULT_RUNS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal eval, quality metrics, visuals, and summary for e8/e9/e10 generator comparisons.",
        epilog=(
            "After training finishes, run: CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src "
            "python scripts/run_generator_comparison_report.py --runs e8 e9 e10 --device cuda:0. "
            "Use --dry-run first to inspect commands without writing files or running heavy work."
        ),
    )
    parser.add_argument("--runs", nargs="+", choices=DEFAULT_RUN_ORDER, default=list(DEFAULT_RUN_ORDER))
    parser.add_argument("--base-eval-config", type=Path, default=DEFAULT_BASE_EVAL_CONFIG)
    parser.add_argument("--real-index", type=Path, default=DEFAULT_REAL_INDEX)
    parser.add_argument("--max-generated", type=int, default=DEFAULT_MAX_GENERATED)
    parser.add_argument("--max-real", type=int, default=DEFAULT_MAX_REAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-visuals", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    return parser.parse_args(argv)


def build_comparison_plan(args: argparse.Namespace) -> ComparisonPlan:
    runs = [
        _build_run_plan(DEFAULT_RUNS[name], args)
        for name in args.runs
    ]
    return ComparisonPlan(
        runs=runs,
        base_eval_config=Path(args.base_eval_config),
        real_index=Path(args.real_index),
        summary_out=Path(args.summary_out),
        max_generated=int(args.max_generated),
        max_real=int(args.max_real),
        seed=int(args.seed),
        device=str(args.device),
        python=str(args.python),
        dry_run=bool(args.dry_run),
    )


def _build_run_plan(spec: RunSpec, args: argparse.Namespace) -> RunPlan:
    config_path = DEFAULT_CONFIG_DIR / f"eval_{spec.name}.yaml"
    result_json = spec.out_dir / "result.json"
    per_sample_jsonl = spec.out_dir / "per_sample.jsonl"
    sample_dir = spec.out_dir / "samples"
    generated_image_dir = spec.out_dir / "generated_images"
    quality_json = spec.out_dir / "quality_fid_kid_niqe.json"
    visualization_path = Path("artifacts/visualizations") / f"{spec.artifact_name}_formal_pairs.png"
    commands: list[tuple[str, list[str]]] = []

    if not args.skip_eval:
        commands.append(
            (
                "eval",
                [str(args.python), "-m", "safa.cli.eval", "--config", str(config_path)],
            )
        )
    if not args.skip_quality:
        commands.append(
            (
                "quality",
                [
                    str(args.python),
                    "scripts/eval_generation_quality.py",
                    "--real-index",
                    str(args.real_index),
                    "--generated-dir",
                    str(generated_image_dir),
                    "--output",
                    str(quality_json),
                    "--metrics",
                    "fid",
                    "kid",
                    "niqe",
                    "--max-generated",
                    str(args.max_generated),
                    "--max-real",
                    str(args.max_real),
                    "--device",
                    str(args.device),
                    "--seed",
                    str(args.seed),
                ],
            )
        )
    if not args.skip_visuals:
        commands.append(
            (
                "visuals",
                [
                    str(args.python),
                    "scripts/visualize_eval_pairs.py",
                    "--result-json",
                    str(result_json),
                    "--out-path",
                    str(visualization_path),
                    "--num-samples",
                    str(DEFAULT_VISUAL_NUM_SAMPLES),
                    "--seed",
                    str(args.seed),
                ],
            )
        )

    return RunPlan(
        name=spec.name,
        checkpoint=spec.checkpoint,
        out_dir=spec.out_dir,
        artifact_name=spec.artifact_name,
        eval_config=config_path,
        result_json=result_json,
        per_sample_jsonl=per_sample_jsonl,
        sample_dir=sample_dir,
        generated_image_dir=generated_image_dir,
        quality_json=quality_json,
        visualization_path=visualization_path,
        commands=commands,
    )


def write_eval_configs(plan: ComparisonPlan) -> list[Path]:
    base_config = _load_yaml_mapping(plan.base_eval_config, "base eval config")
    written: list[Path] = []
    for run in plan.runs:
        config = dict(base_config)
        config.update(
            {
                "seed": plan.seed,
                "sampling_seed": plan.seed,
                "device": plan.device,
                "g_checkpoint": str(run.checkpoint),
                "checkpoint_model": "raw",
                "out_json": str(run.result_json),
                "per_sample_jsonl": str(run.per_sample_jsonl),
                "sample_dir": str(run.sample_dir),
                "generated_image_dir": str(run.generated_image_dir),
            }
        )
        run.eval_config.parent.mkdir(parents=True, exist_ok=True)
        run.eval_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(run.eval_config)
    return written


def build_summary(plan: ComparisonPlan, *, generated_at: str | None = None) -> dict[str, Any]:
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": timestamp,
        "base_eval_config": str(plan.base_eval_config),
        "real_index": str(plan.real_index),
        "summary_out": str(plan.summary_out),
        "max_generated": plan.max_generated,
        "max_real": plan.max_real,
        "seed": plan.seed,
        "device": plan.device,
        "runs": {run.name: _summarize_run(run) for run in plan.runs},
    }


def write_summary(plan: ComparisonPlan) -> Path:
    summary = build_summary(plan)
    plan.summary_out.parent.mkdir(parents=True, exist_ok=True)
    plan.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan.summary_out


def render_dry_run(plan: ComparisonPlan) -> str:
    lines = ["DRY RUN: no commands executed and no files written."]
    lines.append(f"base_eval_config: {plan.base_eval_config}")
    lines.append(f"real_index: {plan.real_index}")
    lines.append(f"summary_out: {plan.summary_out}")
    lines.append("eval configs:")
    for run in plan.runs:
        lines.append(f"  {run.name}: {run.eval_config}")
    lines.append("commands:")
    for run in plan.runs:
        for kind, command in run.commands:
            lines.append(f"  [{run.name}:{kind}] {shlex.join(command)}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_comparison_plan(args)
    if plan.dry_run:
        print(render_dry_run(plan), end="")
        return 0

    write_eval_configs(plan)
    for run in plan.runs:
        for kind, command in run.commands:
            print(f"[{run.name}:{kind}] {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True)
    summary_path = write_summary(plan)
    print(summary_path)
    return 0


def _summarize_run(run: RunPlan) -> dict[str, Any]:
    result = _read_json_if_exists(run.result_json)
    quality = _read_json_if_exists(run.quality_json)
    metrics: dict[str, Any] = {}
    if result is not None:
        metrics.update(_extract_result_metrics(result))
    if quality is not None:
        metrics.update(_extract_quality_metrics(quality))

    return {
        "checkpoint": str(run.checkpoint),
        "out_dir": str(run.out_dir),
        "eval_config": str(run.eval_config),
        "result_json": str(run.result_json),
        "quality_json": str(run.quality_json),
        "visualization": str(run.visualization_path),
        "files": {
            "result_json_exists": run.result_json.is_file(),
            "quality_json_exists": run.quality_json.is_file(),
            "visualization_exists": run.visualization_path.is_file(),
        },
        "metrics": metrics,
    }


def _extract_result_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("num_samples", "num_rows", "sample_count"):
        value = result.get(key)
        if _is_number(value):
            metrics[key] = value

    raw_metrics = result.get("metrics")
    if isinstance(raw_metrics, dict):
        for key, value in raw_metrics.items():
            scalar = _scalar_metric_value(value)
            if scalar is not None:
                metrics[key] = scalar
    return metrics


def _extract_quality_metrics(quality: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("num_real", "num_generated", "fid", "kid_mean", "kid_std"):
        value = quality.get(key)
        if _is_number(value):
            metrics[key] = value

    iqa = quality.get("iqa")
    if isinstance(iqa, dict):
        method = iqa.get("method")
        if isinstance(method, str) and method:
            metrics["iqa_method"] = method
            mean_key = f"{method}_mean"
            std_key = f"{method}_std"
        else:
            mean_key = "iqa_mean"
            std_key = "iqa_std"
        if _is_number(iqa.get("mean")):
            metrics[mean_key] = iqa["mean"]
        if _is_number(iqa.get("std")):
            metrics[std_key] = iqa["std"]
    return metrics


def _scalar_metric_value(value: Any) -> Any | None:
    if _is_number(value):
        return value
    if isinstance(value, dict) and _is_number(value.get("mean")):
        return value["mean"]
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
