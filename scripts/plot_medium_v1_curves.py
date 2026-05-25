#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot medium_v1 curves") from exc

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
    parser.add_argument("--stage1-json", required=True, type=Path)
    parser.add_argument("--m0-json", required=True, type=Path)
    parser.add_argument("--m1-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
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
