#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


BASELINES = [
    {
        "label": "dual_step",
        "method": "primal_dual_projected",
        "valid_fm_loss": 0.07888,
        "repr_cosine_mean": 0.70763,
        "conflict_fraction": 0.443,
    },
    {
        "label": "adaptive_trust",
        "method": "adaptive_trust_projected",
        "valid_fm_loss": 0.09216,
        "repr_cosine_mean": 0.70601,
        "conflict_fraction": 0.473,
    },
    {
        "label": "cagrad",
        "method": "cagrad",
        "valid_fm_loss": 0.118409,
        "repr_cosine_mean": 0.743446,
        "conflict_fraction": 1.0,
        "cagrad_fm_weight": 0.0,
        "cagrad_cl_weight": 1.0,
    },
    {
        "label": "uncertainty",
        "method": "uncertainty_weighted",
        "valid_fm_loss": 0.092128,
        "repr_cosine_mean": 0.719974,
        "conflict_fraction": 1.0,
    },
]


def build_comparison(summary_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    observed = _read_observed(summary_paths)
    rows = BASELINES + observed
    best_fm = min(rows, key=lambda item: float(item["valid_fm_loss"]))
    best_cosine = max(rows, key=lambda item: float(item["repr_cosine_mean"]))
    comparison = {
        "complete": len(observed) == len(summary_paths),
        "expected_new_runs": len(summary_paths),
        "observed_new_runs": len(observed),
        "best_fm_label": best_fm["label"],
        "best_cosine_label": best_cosine["label"],
        "rows": rows,
    }
    (out_path / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    (out_path / "comparison.md").write_text(_format_markdown(comparison), encoding="utf-8")
    return comparison


def watch_comparison(
    summary_paths: list[str | Path],
    output_dir: str | Path,
    interval_seconds: int,
    stop_when_complete: bool,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    while True:
        comparison = build_comparison(summary_paths, output_dir)
        print(json.dumps({key: comparison[key] for key in ["complete", "observed_new_runs", "expected_new_runs"]}, sort_keys=True), flush=True)
        if stop_when_complete and comparison["complete"]:
            return
        time.sleep(interval_seconds)


def _read_observed(summary_paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        path = Path(summary_path)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for experiment in payload.get("experiments", []):
            final = experiment["final"]
            row = {
                "label": payload["run_name"],
                "method": experiment["method"],
                "valid_fm_loss": float(final["valid_fm_loss"]),
                "repr_cosine_mean": float(final["repr_cosine_mean"]),
                "conflict_fraction": float(final["conflict_fraction"]),
            }
            for key in [
                "cagrad_fm_weight",
                "cagrad_cl_weight",
                "cagrad_raw_fm_weight",
                "cagrad_raw_cl_weight",
                "fm_descent_floor",
                "fm_descent_after_cagrad",
                "fm_descent_after_anchor",
                "fm_anchor_active",
                "fm_budget",
                "fm_descent_after_budget",
                "fm_budget_active",
                "fm_budget_scale",
            ]:
                if key in final:
                    row[key] = float(final[key])
            rows.append(row)
    return rows


def _format_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Toy CAGrad Comparison",
        "",
        f"- Complete: {comparison['complete']}",
        f"- New runs: {comparison['observed_new_runs']}/{comparison['expected_new_runs']}",
        f"- Best FM loss: {comparison['best_fm_label']}",
        f"- Best cosine: {comparison['best_cosine_label']}",
        "",
        "| label | method | FM loss | cosine | conflict | FM weight | CL weight | anchor/budget |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        action_value = row.get("fm_anchor_active", row.get("fm_budget_active", ""))
        lines.append(
            "| {label} | {method} | {fm:.6f} | {cos:.6f} | {conf:.3f} | {fm_w} | {cl_w} | {active} |".format(
                label=row["label"],
                method=row["method"],
                fm=float(row["valid_fm_loss"]),
                cos=float(row["repr_cosine_mean"]),
                conf=float(row["conflict_fraction"]),
                fm_w=_format_optional(row.get("cagrad_fm_weight")),
                cl_w=_format_optional(row.get("cagrad_cl_weight")),
                active=_format_optional(action_value),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    if value == "":
        return ""
    if value is None:
        return ""
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch toy CAGrad summaries and write a CPU-only comparison report.")
    parser.add_argument("--summary", action="append", required=True, help="Expected summary.json path. Repeat for each new run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--stop-when-complete", action="store_true")
    args = parser.parse_args()
    watch_comparison(
        summary_paths=args.summary,
        output_dir=args.output_dir,
        interval_seconds=args.interval_seconds,
        stop_when_complete=args.stop_when_complete,
    )


if __name__ == "__main__":
    main()
