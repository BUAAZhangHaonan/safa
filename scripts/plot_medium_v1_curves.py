#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def load_medium_v1_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required medium_v1 JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history") if isinstance(payload, dict) else payload
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} must contain a non-empty history list")
    rows = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: history[{index}] must be an object")
        rows.append(item)
    return rows


def _series(history: list[dict[str, Any]], field: str, label: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for index, row in enumerate(history, start=1):
        if field not in row:
            raise ValueError(f"{label}: missing required curve field {field!r} at history index {index - 1}")
        value = row[field]
        if isinstance(value, bool):
            raise ValueError(f"{label}: curve field {field!r} must be numeric, got bool")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label}: curve field {field!r} must be finite, got {value!r}")
        xs.append(index)
        ys.append(number)
    return xs, ys


def _utility_series(history: list[dict[str, Any]], label: str, model: str = "raw") -> tuple[list[int], list[float]]:
    cosine_field = f"validation_{model}_latent_cosine_mean"
    single_field = f"validation_{model}_single_face_eq1_rate"
    xs = []
    ys = []
    for index, row in enumerate(history, start=1):
        if cosine_field not in row and model == "raw":
            cosine_field = "validation_latent_cosine_mean"
        if single_field not in row and model == "raw":
            single_field = "validation_single_face_eq1_rate"
        if cosine_field not in row or single_field not in row:
            raise ValueError(f"{label}: missing utility curve fields for {model}")
        cosine = float(row[cosine_field])
        single = float(row[single_field])
        if not math.isfinite(cosine) or not math.isfinite(single):
            raise ValueError(f"{label}: utility curve contains non-finite values")
        xs.append(index)
        ys.append(cosine * single)
    return xs, ys


def _plot_lines(curves: list[tuple[str, list[int], list[float]]], title: str, ylabel: str, output: Path) -> None:
    plt = _import_pyplot()

    if not curves:
        raise ValueError(f"no curves provided for {title}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, xs, ys in curves:
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot medium_v1 curves") from exc
    return plt


def _finite_optional(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _first_number(row: dict[str, Any], fields: tuple[str, ...], label: str) -> float | None:
    for field in fields:
        if field in row:
            return _finite_optional(row[field], f"{label}.{field}")
    return None


def _load_stage1_history_source(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required Stage1 history source is missing: {path}")
    if path.suffix == ".json":
        return load_medium_v1_history(path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to read Stage1 checkpoint history") from exc
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        metrics = checkpoint.get("metrics")
        if isinstance(metrics, dict):
            history = [metrics]
        else:
            raise ValueError(f"{path} must contain a non-empty history list or metrics object")
    rows = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: history[{index}] must be an object")
        rows.append(dict(item))
    return rows


def _load_last_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a metrics object")
    return payload


def _stage_epoch(row: dict[str, Any]) -> int | None:
    if "stage_epoch" not in row:
        return None
    value = _finite_optional(row["stage_epoch"], "stage_epoch")
    if value is None:
        return None
    if int(value) != value or value < 0:
        raise ValueError(f"stage_epoch must be a non-negative integer, got {row['stage_epoch']!r}")
    return int(value)


def _merge_last_metrics(history: list[dict[str, Any]], last_metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = [dict(item) for item in history]
    if last_metrics is None:
        return rows
    last_epoch = _stage_epoch(last_metrics)
    if last_epoch is None:
        rows.append(dict(last_metrics))
        return rows
    for index in range(len(rows) - 1, -1, -1):
        if _stage_epoch(rows[index]) == last_epoch:
            rows[index] = dict(last_metrics)
            return rows
    rows.append(dict(last_metrics))
    return rows


def _latest_stage_epoch_segment(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not history:
        raise ValueError("Stage1 history is empty")
    start = 0
    previous: int | None = None
    for index, row in enumerate(history):
        current = _stage_epoch(row)
        if current is not None and previous is not None and current <= previous:
            start = index
        if current is not None:
            previous = current
    segment = [dict(item) for item in history[start:]]
    if not segment:
        raise ValueError("Stage1 history has no latest segment")
    return segment


def _quality_epoch_from_path(path: Path) -> int | None:
    for text in (path.parent.name, path.name):
        match = re.search(r"epoch_(\d{4,})", text)
        if match:
            return int(match.group(1))
    return None


def load_stage1_quality_timeseries(quality_dir: Path) -> dict[int, dict[str, float]]:
    if not quality_dir.is_dir():
        return {}
    rows: dict[int, dict[str, float]] = {}
    for path in sorted(quality_dir.glob("epoch_*/*.json")):
        epoch = _quality_epoch_from_path(path)
        if epoch is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a quality metrics object")
        row = rows.setdefault(epoch, {})
        iqa = payload.get("iqa")
        if isinstance(iqa, dict) and str(iqa.get("method", "")).lower() == "niqe":
            niqe = _finite_optional(iqa.get("mean"), f"{path}.iqa.mean")
            if niqe is not None:
                row["niqe"] = niqe
            niqe_std = _finite_optional(iqa.get("std"), f"{path}.iqa.std")
            if niqe_std is not None:
                row["niqe_std"] = niqe_std
        for source_field, output_field in (
            ("fid", "fid"),
            ("kid_mean", "kid_mean"),
            ("kid_std", "kid_std"),
        ):
            if source_field in payload:
                value = _finite_optional(payload[source_field], f"{path}.{source_field}")
                if value is not None:
                    row[output_field] = value
    return rows


def _epoch_number(row: dict[str, Any], fallback_index: int) -> int:
    current = _stage_epoch(row)
    if current is not None:
        return current + 1
    return fallback_index


def _stage1_timeseries_row(row: dict[str, Any], epoch: int, quality: dict[str, float]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "stage_epoch": _stage_epoch(row),
        "loss": _first_number(row, ("loss",), "history"),
        "flow_mse": _first_number(row, ("flow_matching_mse", "flow_loss_raw", "flow_loss_normalized"), "history"),
        "grad_norm": _first_number(row, ("grad_norm",), "history"),
        "latent_cosine": _first_number(
            row,
            ("validation_raw_latent_cosine_mean", "validation_latent_cosine_mean"),
            "history",
        ),
        "source_prediction_preserved": _first_number(
            row,
            ("validation_raw_source_prediction_preserved", "validation_source_prediction_preserved"),
            "history",
        ),
        "face_detect_ge1": _first_number(
            row,
            ("validation_raw_face_detect_ge1_rate", "validation_face_detect_ge1_rate", "validation_face_detection_rate"),
            "history",
        ),
        "single_face_eq1": _first_number(
            row,
            ("validation_raw_single_face_eq1_rate", "validation_single_face_eq1_rate"),
            "history",
        ),
        "zero_face": _first_number(row, ("validation_raw_zero_face_rate", "validation_zero_face_rate"), "history"),
        "multi_face": _first_number(row, ("validation_raw_multi_face_rate", "validation_multi_face_rate"), "history"),
        "niqe": quality.get("niqe", _first_number(row, ("quality_raw_niqe", "quality_niqe"), "history")),
        "niqe_std": quality.get("niqe_std"),
        "fid": quality.get("fid", _first_number(row, ("quality_raw_fid", "quality_fid"), "history")),
        "kid_mean": quality.get("kid_mean", _first_number(row, ("quality_raw_kid_mean", "quality_kid_mean"), "history")),
        "kid_std": quality.get("kid_std", _first_number(row, ("quality_raw_kid_std", "quality_kid_std"), "history")),
    }


def build_stage1_long200_timeseries(
    history_path: Path,
    last_metrics_path: Path | None,
    quality_dir: Path,
    *,
    run_name: str = "stage1_long200_v4",
) -> dict[str, Any]:
    history = _load_stage1_history_source(history_path)
    history = _merge_last_metrics(history, _load_last_metrics(last_metrics_path))
    history = _latest_stage_epoch_segment(history)
    quality = load_stage1_quality_timeseries(quality_dir)
    epochs = [
        _stage1_timeseries_row(row, _epoch_number(row, index), quality.get(_epoch_number(row, index), {}))
        for index, row in enumerate(history, start=1)
    ]
    return {
        "run": run_name,
        "sources": {
            "history": str(history_path),
            "last_metrics": str(last_metrics_path) if last_metrics_path is not None else None,
            "quality_dir": str(quality_dir),
        },
        "epochs": epochs,
    }


def _points(rows: list[dict[str, Any]], field: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        number = _finite_optional(value, field)
        if number is None:
            continue
        xs.append(int(row["epoch"]))
        ys.append(number)
    return xs, ys


def _plot_or_pending(ax: Any, rows: list[dict[str, Any]], field: str, label: str, ylabel: str) -> None:
    xs, ys = _points(rows, field)
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)
        ax.legend()
    else:
        ax.text(0.5, 0.5, f"{label} pending", ha="center", va="center", transform=ax.transAxes, alpha=0.65)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _plot_quality_curves(rows: list[dict[str, Any]], output: Path) -> None:
    plt = _import_pyplot()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    _plot_or_pending(axes[0], rows, "niqe", "NIQE", "NIQE")
    _plot_or_pending(axes[1], rows, "fid", "FID", "FID")
    _plot_or_pending(axes[2], rows, "kid_mean", "KID mean", "KID")
    axes[0].set_title("stage1_long200_v4 quality")
    axes[2].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_named_fields(rows: list[dict[str, Any]], fields: tuple[tuple[str, str], ...], title: str, ylabel: str, output: Path) -> None:
    plt = _import_pyplot()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for field, label in fields:
        xs, ys = _points(rows, field)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_training_curves(rows: list[dict[str, Any]], output: Path) -> None:
    plt = _import_pyplot()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for field, label in (("loss", "loss"), ("flow_mse", "flow_mse")):
        xs, ys = _points(rows, field)
        if xs:
            axes[0].plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)
    axes[0].set_title("stage1_long200_v4 training")
    axes[0].set_ylabel("loss")
    if axes[0].lines:
        axes[0].legend()
    _plot_or_pending(axes[1], rows, "grad_norm", "grad_norm", "grad_norm")
    _plot_or_pending(axes[2], rows, "latent_cosine", "latent_cosine", "latent cosine")
    _plot_or_pending(
        axes[3],
        rows,
        "source_prediction_preserved",
        "source_prediction_preserved",
        "source preserved",
    )
    axes[3].set_xlabel("epoch")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_stage1_long200_curves(
    *,
    history_path: Path,
    last_metrics_path: Path | None,
    quality_dir: Path,
    out_dir: Path,
    output_prefix: str = "stage1_long200_v4",
) -> list[Path]:
    payload = build_stage1_long200_timeseries(
        history_path,
        last_metrics_path,
        quality_dir,
        run_name=output_prefix,
    )
    rows = payload["epochs"]
    if not rows:
        raise ValueError("Stage1 long200 timeseries has no epochs")
    outputs = [
        out_dir / f"{output_prefix}_quality_curves.png",
        out_dir / f"{output_prefix}_face_curves.png",
        out_dir / f"{output_prefix}_training_curves.png",
        out_dir / f"{output_prefix}_metrics_timeseries.json",
    ]
    _plot_quality_curves(rows, outputs[0])
    _plot_named_fields(
        rows,
        (
            ("face_detect_ge1", "face_detect_ge1"),
            ("single_face_eq1", "single_face_eq1"),
            ("zero_face", "zero_face"),
            ("multi_face", "multi_face"),
        ),
        "stage1_long200_v4 face metrics",
        "rate",
        outputs[1],
    )
    _plot_training_curves(rows, outputs[2])
    outputs[3].parent.mkdir(parents=True, exist_ok=True)
    outputs[3].write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return outputs


def plot_medium_v1_curves(*, stage1_json: Path, m0_json: Path, m1_json: Path, out_dir: Path) -> list[Path]:
    histories = {
        "stage1": load_medium_v1_history(stage1_json),
        "m0": load_medium_v1_history(m0_json),
        "m1_uw": load_medium_v1_history(m1_json),
    }
    outputs = [
        out_dir / "medium_v1_loss.png",
        out_dir / "medium_v1_flow_loss_raw.png",
        out_dir / "medium_v1_cycle_loss_raw.png",
        out_dir / "medium_v1_flow_loss_normalized.png",
        out_dir / "medium_v1_cycle_loss_normalized.png",
        out_dir / "medium_v1_raw_utility.png",
    ]
    _plot_lines(
        [(name, *_series(history, "loss", name)) for name, history in histories.items()],
        "medium_v1 loss",
        "loss",
        outputs[0],
    )
    _plot_lines(
        [(name, *_series(history, "flow_loss_raw", name)) for name, history in histories.items()],
        "medium_v1 raw flow loss",
        "flow loss",
        outputs[1],
    )
    _plot_lines(
        [(name, *_series(history, "cycle_loss_raw", name)) for name, history in histories.items()],
        "medium_v1 raw cycle loss",
        "cycle loss",
        outputs[2],
    )
    _plot_lines(
        [(name, *_series(history, "flow_loss_normalized", name)) for name, history in histories.items()],
        "medium_v1 normalized flow loss",
        "normalized flow loss",
        outputs[3],
    )
    _plot_lines(
        [(name, *_series(history, "cycle_loss_normalized", name)) for name, history in histories.items()],
        "medium_v1 normalized cycle loss",
        "normalized cycle loss",
        outputs[4],
    )
    _plot_lines(
        [(name, *_utility_series(history, name, "raw")) for name, history in histories.items()],
        "medium_v1 raw utility",
        "latent cosine x single-face rate",
        outputs[5],
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot medium_v1 G training and evaluation curves.")
    parser.add_argument("--stage1-json", type=Path)
    parser.add_argument("--m0-json", type=Path)
    parser.add_argument("--m1-json", type=Path)
    parser.add_argument("--stage1-history", type=Path, help="Stage1 checkpoint .pt or JSON containing history.")
    parser.add_argument("--stage1-last-metrics-json", type=Path)
    parser.add_argument("--stage1-quality-dir", type=Path)
    parser.add_argument("--stage1-output-prefix", default="stage1_long200_v4")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.stage1_history is not None or args.stage1_quality_dir is not None:
            if args.stage1_history is None or args.stage1_quality_dir is None:
                raise ValueError("--stage1-history and --stage1-quality-dir must be provided together")
            outputs = plot_stage1_long200_curves(
                history_path=args.stage1_history,
                last_metrics_path=args.stage1_last_metrics_json,
                quality_dir=args.stage1_quality_dir,
                out_dir=args.out_dir,
                output_prefix=args.stage1_output_prefix,
            )
        else:
            if args.stage1_json is None or args.m0_json is None or args.m1_json is None:
                raise ValueError("--stage1-json, --m0-json, and --m1-json are required for medium_v1 comparison plots")
            outputs = plot_medium_v1_curves(
                stage1_json=args.stage1_json,
                m0_json=args.m0_json,
                m1_json=args.m1_json,
                out_dir=args.out_dir,
            )
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
