#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


REQUIRED_METRIC_KEYS = {
    "actual_fm_delta_after_repr_step",
    "conflict_fraction",
    "delta_deg",
    "dot_after_mean",
    "dot_before_mean",
    "generation_mse_to_x1",
    "lambda_repr",
    "method",
    "projected_repr_norm_ratio",
    "repr_cosine_mean",
    "repr_point_loss",
    "repr_relation_loss",
    "soft_margin",
    "step",
    "valid_fm_loss",
}


class EmptyMetricsFileError(RuntimeError):
    pass


def refresh_visuals(run_dir: str | Path, expected_experiments: int, steps_per_experiment: int) -> dict[str, Any]:
    if expected_experiments <= 0:
        raise ValueError("expected_experiments must be positive")
    if steps_per_experiment <= 0:
        raise ValueError("steps_per_experiment must be positive")
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing toy metrics file: {metrics_path}")
    rows = _read_metric_rows(metrics_path)
    if not rows:
        raise EmptyMetricsFileError(f"Toy metrics file is empty: {metrics_path}")
    live_curves = run_path / "live_curves.png"
    live_tradeoff = run_path / "live_tradeoff.png"
    _plot_live_curves(rows, live_curves)
    _plot_live_tradeoff(rows, live_tradeoff)
    progress = _build_progress(rows, expected_experiments, steps_per_experiment)
    (run_path / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    return progress


def watch_visuals(
    run_dir: str | Path,
    expected_experiments: int,
    steps_per_experiment: int,
    interval_seconds: int,
    stop_when_summary_exists: bool,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    run_path = Path(run_dir)
    while True:
        try:
            progress = refresh_visuals(run_path, expected_experiments, steps_per_experiment)
            print(json.dumps(progress, sort_keys=True), flush=True)
        except (FileNotFoundError, EmptyMetricsFileError) as error:
            print(json.dumps({"status": "waiting_for_metrics", "message": str(error)}, sort_keys=True), flush=True)
        if stop_when_summary_exists and (run_path / "summary.json").is_file():
            if (run_path / "metrics.jsonl").is_file():
                progress = refresh_visuals(run_path, expected_experiments, steps_per_experiment)
                print(json.dumps({"status": "summary_detected", "progress": progress}, sort_keys=True), flush=True)
            return
        time.sleep(interval_seconds)


def _read_metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"Metric row {line_number} must be a JSON object")
        missing = sorted(REQUIRED_METRIC_KEYS - set(row))
        if missing:
            raise KeyError(f"Metric row {line_number} is missing keys: {missing}")
        rows.append(row)
    return rows


def _build_progress(rows: list[dict[str, Any]], expected_experiments: int, steps_per_experiment: int) -> dict[str, Any]:
    complete_rows = [row for row in rows if int(float(row["step"])) == steps_per_experiment]
    latest_rows = _latest_rows_by_experiment(rows)
    return {
        "metric_rows": len(rows),
        "expected_experiments": expected_experiments,
        "steps_per_experiment": steps_per_experiment,
        "completed_experiments_estimate": len(complete_rows),
        "latest_experiment_count": len(latest_rows),
        "latest_step_max": max(float(row["step"]) for row in rows),
        "latest_repr_cosine_best": max(float(row["repr_cosine_mean"]) for row in latest_rows),
        "latest_fm_loss_best": min(float(row["valid_fm_loss"]) for row in latest_rows),
    }


def _latest_rows_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[float, str, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (
            float(row["delta_deg"]),
            str(row["method"]),
            float(row["lambda_repr"]),
            float(row["soft_margin"]),
        )
        if key not in latest or float(row["step"]) >= float(latest[key]["step"]):
            latest[key] = row
    return list(latest.values())


def _plot_live_curves(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple[float, str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["delta_deg"]),
            str(row["method"]),
            float(row["lambda_repr"]),
            float(row["soft_margin"]),
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_specs = [
        ("repr_cosine_mean", "repr cosine"),
        ("valid_fm_loss", "valid FM loss"),
        ("generation_mse_to_x1", "generation MSE"),
        ("actual_fm_delta_after_repr_step", "actual FM delta after repr"),
    ]
    for axis, (metric_key, title) in zip(axes.flatten(), plot_specs):
        for key, group_rows in groups.items():
            group_rows = sorted(group_rows, key=lambda item: float(item["step"]))
            label = f"d={key[0]:g} {key[1]} l={key[2]:g} m={key[3]:g}"
            axis.plot([float(item["step"]) for item in group_rows], [float(item[metric_key]) for item in group_rows], label=label, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("step")
    axes[0, 0].legend(fontsize=4, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_live_tradeoff(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest_rows = _latest_rows_by_experiment(rows)
    fig, axis = plt.subplots(figsize=(10, 6))
    scatter = axis.scatter(
        [float(row["valid_fm_loss"]) for row in latest_rows],
        [float(row["repr_cosine_mean"]) for row in latest_rows],
        c=[float(row["delta_deg"]) for row in latest_rows],
        cmap="viridis",
    )
    for row in latest_rows:
        label = f"d={float(row['delta_deg']):g} {row['method']} l={float(row['lambda_repr']):g} m={float(row['soft_margin']):g}"
        axis.annotate(label, (float(row["valid_fm_loss"]), float(row["repr_cosine_mean"])), fontsize=5, alpha=0.7)
    axis.set_xlabel("validation FM loss")
    axis.set_ylabel("representation cosine")
    axis.set_title("Live toy FM/CL trade-off")
    fig.colorbar(scatter, ax=axis, label="delta_deg")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live visualizations for the toy FM+CL projected-update sweep.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-experiments", required=True, type=int)
    parser.add_argument("--steps-per-experiment", required=True, type=int)
    parser.add_argument("--interval-seconds", required=True, type=int)
    parser.add_argument("--stop-when-summary-exists", action="store_true")
    args = parser.parse_args()
    watch_visuals(
        run_dir=args.run_dir,
        expected_experiments=args.expected_experiments,
        steps_per_experiment=args.steps_per_experiment,
        interval_seconds=args.interval_seconds,
        stop_when_summary_exists=args.stop_when_summary_exists,
    )


if __name__ == "__main__":
    main()
