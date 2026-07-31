#!/usr/bin/env python3
"""Apply the locked R14 regular32 stop decision without tuning."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _classification(gate: Mapping[str, object], visual: Mapping[str, object]) -> tuple[str, str]:
    for key in ("exact_one", "representation", "privacy", "full_quality", "roi_quality"):
        if not isinstance(gate.get(key), bool):
            raise RuntimeError(f"evaluation gate {key} is not explicit boolean")
    if not gate["exact_one"]:
        return "no_go_exact_one", "ArcFace exact-one failed; stop without tuning."
    if not gate["representation"]:
        return "no_go_representation", "The provisional representation gate failed; stop without tuning."
    if not gate["privacy"]:
        return "no_go_privacy", "The provisional privacy gate failed; stop without tuning."
    if not gate["full_quality"]:
        return "no_go_full_quality", "Full-image NIQE or sharpness failed; stop without tuning."
    if not gate["roi_quality"]:
        return (
            "no_go_copied_background_metric_inflation",
            "Full-image quality passed but exact-bbox face ROI quality failed; copied background inflated the full-image score.",
        )
    severe = visual.get("severe_count")
    if severe is None:
        return "numeric_pass_visual_review_pending", "All numeric gates passed; the fixed visual8 severe review is pending."
    if isinstance(severe, bool) or not isinstance(severe, int) or severe < 0:
        raise RuntimeError("visual severe_count must be null or a non-negative integer")
    if severe != 0:
        return "no_go_visual_severe", "The fixed visual8 review contains a severe failure; stop without tuning."
    return "feasibility_pass_stage128_allowed", "All regular32 numeric gates and visual8 severe=0 passed; one stage128 is allowed."


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "summary.json"
    conclusion_path = args.output_dir / "conclusion.md"
    if summary_path.exists() or conclusion_path.exists():
        raise FileExistsError("refusing to replace an existing R14 conclusion")
    evaluation = _read(args.evaluation)
    visual = _read(args.visual)
    gate = evaluation.get("gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("evaluation lacks an explicit gate mapping")
    classification, reason = _classification(gate, visual)
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_feasibility_closeout_v1",
        "classification": classification,
        "reason": reason,
        "training_contract": {
            "epochs": 20,
            "optimizer_steps": 2560,
            "global_batch_size": 8,
            "per_device_batch_size": 2,
        },
        "regular32_gate": dict(gate),
        "metrics": evaluation.get("metrics"),
        "visual8": {
            "review_status": visual.get("review_status"),
            "severe_count": visual.get("severe_count"),
            "contact_sheet": visual.get("contact_sheet"),
        },
        "next_action": "one_stage128" if classification == "feasibility_pass_stage128_allowed" else "stop",
        "search_forbidden": ["mask", "learning_rate", "loss_weight", "training_steps"],
        "claim_boundary": "regular32 feasibility evidence only; not a survivor, privacy proof, Full success, or formal winner",
    }
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    conclusion = "\n".join(
        (
            "# R14 20-epoch face-region inpainting feasibility",
            "",
            f"Classification: `{classification}`.",
            "",
            reason,
            "",
            "The only training arm ran 20 epochs and exactly 2,560 optimizer steps at global batch 8.",
            "",
            "This is regular32 feasibility evidence only. It is not a screening survivor, privacy proof, Full success, or formal winner.",
            "",
        )
    )
    conclusion_path.write_text(conclusion, encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
