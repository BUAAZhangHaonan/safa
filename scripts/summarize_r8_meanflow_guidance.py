#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


CHECKPOINT_SHA256 = "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d"
FULL_MANIFEST_SHA256 = "7f830ad3f84089bcf83d092fbffaf2b5c3335cf68a4b397f04b65f362f79ae5b"
ALLOWED_VISUAL_CATEGORIES = frozenset(
    {
        "blank_or_near_constant",
        "unstructured_noise",
        "repeated_patch_or_tiled_artifact",
        "severe_color_clipping_or_saturation",
        "broken_global_structure",
        "broken_global_face_or_image_structure",
        "large_non_image_texture_region",
    }
)
TABLE_FIELDS = (
    "arm_id",
    "mode",
    "eligible",
    "fid",
    "kid_mean",
    "niqe",
    "sharpness_mean",
    "e0_cosine",
    "edev_cosine",
    "nfe",
    "images_per_second",
    "peak_vram_bytes",
    "severe_failure_count",
)


def validate_calibration_arm(
    arm_id: str,
    generation: Mapping[str, Any],
    quality: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    if not arm_id:
        raise ValueError("arm ID must be non-empty")
    _reject_heldout_outputs(generation)
    if generation.get("status") != "complete":
        raise ValueError(f"arm {arm_id}: generation is not complete")
    generation_count = _integer(generation.get("sample_count"), "generation sample count")
    quality_count = _integer(quality.get("num_generated"), "quality sample count")
    manifest_count = _integer(quality.get("sample_id_count"), "quality manifest sample count")
    if generation_count != 64 or quality_count != 64 or manifest_count != 64:
        raise ValueError(f"arm {arm_id}: calibration sample count must be exactly 64")
    if quality.get("num_real") != 64:
        raise ValueError(f"arm {arm_id}: real quality sample count must be exactly 64")
    generation_digest = _sha(generation.get("sample_id_sha256"), "generation sample-ID digest")
    quality_digest = _sha(quality.get("sample_id_sha256"), "quality sample-ID digest")
    if generation_digest != quality_digest:
        raise ValueError(f"arm {arm_id}: generation/quality sample-ID digests disagree")
    checkpoint = generation.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"arm {arm_id}: missing checkpoint contract")
    checkpoint_sha = _sha(checkpoint.get("sha256"), "checkpoint SHA256")
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError(f"arm {arm_id}: checkpoint SHA256 is not the fixed epoch-1652 EMA")
    config = generation.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"arm {arm_id}: missing resolved config")
    seed = _integer(config.get("sampling_seed", config.get("seed")), "sampling seed")
    if seed != 1337:
        raise ValueError(f"arm {arm_id}: sampling seed must be 1337")
    expected_metrics = {"fid", "kid", "niqe", "sharpness"}
    if set(quality.get("metrics", ())) != expected_metrics:
        raise ValueError(f"arm {arm_id}: quality metrics must be {sorted(expected_metrics)!r}")

    cosine = generation.get("cosine")
    if not isinstance(cosine, Mapping):
        raise ValueError(f"arm {arm_id}: missing cosine summaries")
    e0 = _summary_mean(cosine, "candidate_e0_target")
    native_e0 = _summary_mean(cosine, "native_e0_target")
    edev = _summary_mean(cosine, "candidate_edev_source")
    native_edev = _summary_mean(cosine, "native_edev_source")
    sharpness = _nested_finite(quality, ("sharpness", "mean"), "Sharpness mean")
    native_sharpness = _nested_finite(
        generation, ("native_sharpness", "mean"), "matched-native Sharpness mean"
    )
    if native_sharpness <= 0.0:
        raise ValueError("matched-native Sharpness mean must be positive")
    severe = _integer(review.get("severe_failure_count"), "severe visual failure count")
    failures = review.get("failures")
    if not isinstance(failures, list) or len(failures) != severe:
        raise ValueError(f"arm {arm_id}: visual failure rows/count disagree")
    for failure in failures:
        if not isinstance(failure, Mapping) or not failure.get("sample_id"):
            raise ValueError(f"arm {arm_id}: invalid visual failure record")
        if failure.get("category") not in ALLOWED_VISUAL_CATEGORIES:
            raise ValueError(f"arm {arm_id}: unknown visual failure category")
    if not 0 <= severe <= 64:
        raise ValueError(f"arm {arm_id}: severe visual failure count is outside [0,64]")
    nfe = _integer(_nested(generation, ("nfe", "candidate"), "candidate NFE"), "candidate NFE")
    if nfe <= 0:
        raise ValueError(f"arm {arm_id}: candidate NFE must be positive")
    mode = str(generation.get("mode", ""))
    schedule_contract = _fmrg_schedule_contract(
        arm_id=arm_id,
        mode=mode,
        generation=generation,
        config=config,
        checkpoint_sha256=checkpoint_sha,
    )

    row = {
        "arm_id": arm_id,
        "mode": mode,
        "sample_count": generation_count,
        "sample_id_sha256": generation_digest,
        "checkpoint_sha256": checkpoint_sha,
        "seed": seed,
        "fid": _finite(quality.get("fid"), "FID"),
        "kid_mean": _finite(quality.get("kid_mean"), "KID mean"),
        "kid_std": _finite(quality.get("kid_std"), "KID std"),
        "niqe": _nested_finite(quality, ("iqa", "mean"), "NIQE mean"),
        "sharpness_mean": sharpness,
        "native_sharpness_mean": native_sharpness,
        "sharpness_retention": sharpness / native_sharpness,
        "e0_cosine": e0,
        "native_e0_cosine": native_e0,
        "e0_delta": e0 - native_e0,
        "edev_cosine": edev,
        "native_edev_cosine": native_edev,
        "edev_delta": edev - native_edev,
        "nfe": nfe,
        "images_per_second": _nested_finite(
            generation, ("timing", "images_per_second"), "images per second"
        ),
        "wall_seconds": _nested_finite(generation, ("timing", "wall_seconds"), "wall seconds"),
        "peak_vram_bytes": _nested_finite(
            generation, ("max_memory", "reserved_bytes"), "peak reserved VRAM"
        ),
        "severe_failure_count": severe,
        "visual_failure_rate": severe / 64.0,
        "config": dict(config),
        "generation_result": dict(generation),
        "quality_result": dict(quality),
        **schedule_contract,
    }
    reasons = []
    if row["e0_delta"] < 0.02:
        reasons.append("e0_delta")
    if row["edev_delta"] < 0.0:
        reasons.append("edev_direction")
    if row["sharpness_retention"] < 0.80:
        reasons.append("sharpness_retention")
    if row["visual_failure_rate"] > 0.10:
        reasons.append("visual_failure_rate")
    row["ineligible_reasons"] = reasons
    row["eligible"] = not reasons
    _require_finite_tree(row, "calibration arm")
    return row


def select_calibration_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_digests = {str(row.get("sample_id_sha256")) for row in rows}
    if len(sample_digests) != 1:
        raise ValueError("all calibration arms must use the same sample-ID membership")
    eligible = [dict(row) for row in rows if row.get("eligible") is True]
    if not eligible:
        raise ValueError("no eligible R8 calibration candidate")
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["e0_cosine"]),
            -float(row["edev_cosine"]),
            int(row["severe_failure_count"]),
            -float(row["sharpness_retention"]),
            int(row["nfe"]),
            str(row["arm_id"]),
        ),
    )
    winner = ranked[0]
    winner_contract = {
        key: winner[key]
        for key in (
            "arm_id",
            "mode",
            "e0_cosine",
            "edev_cosine",
            "severe_failure_count",
            "sharpness_retention",
            "nfe",
            "checkpoint_sha256",
            "sample_id_sha256",
            "seed",
        )
    }
    for key in (
        "t_cut",
        "schedule_manifest",
        "schedule_manifest_sha256",
        "schedule_contract_sha256",
    ):
        if key in winner:
            winner_contract[key] = winner[key]
    return {
        "selection_rule": (
            "higher E0 cosine, higher Edev cosine, fewer severe visual failures, "
            "higher Sharpness retention, lower NFE, lexical arm ID"
        ),
        "fid_policy": "64-sample FID is diagnostic only",
        "winner": winner_contract,
        "eligible_arm_ids": [str(row["arm_id"]) for row in ranked],
    }


def summarize_calibration(root: Path) -> dict[str, Any]:
    root = Path(root)
    visual_path = root / "visual_review.json"
    if not visual_path.is_file():
        raise FileNotFoundError(f"required visual_review.json does not exist: {visual_path}")
    visual = _read_json(visual_path, "visual_review")
    if visual.get("reviewed_sample_count") != 64:
        raise ValueError("visual_review must record exactly 64 reviewed samples")
    reviews = visual.get("arms")
    if not isinstance(reviews, Mapping):
        raise ValueError("visual_review must contain an arms mapping")
    calibration = root / "calibration"
    if not calibration.is_dir():
        raise FileNotFoundError(f"calibration directory does not exist: {calibration}")
    arm_dirs = sorted(
        path
        for path in calibration.iterdir()
        if path.is_dir() and (path / "generation_result.json").is_file()
    )
    if not arm_dirs:
        raise ValueError("no completed R8 calibration arms were found")
    arm_ids = {path.name for path in arm_dirs}
    if set(reviews) != arm_ids:
        raise ValueError(
            "visual_review arm IDs must exactly match completed calibration arm IDs: "
            f"completed={sorted(arm_ids)!r} reviewed={sorted(reviews)!r}"
        )
    rows = []
    for arm_dir in arm_dirs:
        generation = _read_json(arm_dir / "generation_result.json", "generation result")
        quality = _read_json(arm_dir / "quality.json", "quality result")
        if "native_sharpness" not in generation:
            generation["native_sharpness"] = {
                "mean": _native_sharpness_mean(arm_dir / "per_sample.jsonl")
            }
        rows.append(
            validate_calibration_arm(arm_dir.name, generation, quality, reviews[arm_dir.name])
        )
    selection = select_calibration_winner(rows)
    winner_row = next(row for row in rows if row["arm_id"] == selection["winner"]["arm_id"])
    winner_config_path = root / "locked_winner_config.yaml"
    _write_exclusive_text(
        winner_config_path,
        yaml.safe_dump(winner_row["config"], sort_keys=False),
    )
    config_sha = _sha256_file(winner_config_path)
    selection["winner"].update(
        {
            "config": str(winner_config_path),
            "config_sha256": config_sha,
        }
    )
    selection.update(
        {
            "schema_version": 1,
            "winner_locked_before_heldout": True,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "calibration_sample_count": 64,
            "calibration_sample_id_sha256": winner_row["sample_id_sha256"],
            "full_sample_count": 2048,
            "full_sample_id_manifest_sha256": FULL_MANIFEST_SHA256,
            "full_sample_id_sha256": _optional_full_id_digest(root),
            "prospective_heldout_status": "not_evaluated",
        }
    )
    _write_exclusive_json(root / "selection.json", selection)
    write_summary_tables(rows, root / "summary.csv", root / "summary.md")
    return selection


def write_summary_tables(rows: Sequence[Mapping[str, Any]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    headers = (
        "Arm",
        "Mode",
        "Eligible",
        "FID",
        "KID",
        "NIQE",
        "Sharpness",
        "E0 cosine",
        "Edev cosine",
        "NFE",
        "images/s",
        "peak VRAM",
        "severe",
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = (
            row["arm_id"],
            row["mode"],
            row["eligible"],
            _fmt(row["fid"]),
            _fmt(row["kid_mean"]),
            _fmt(row["niqe"]),
            _fmt(row["sharpness_mean"]),
            _fmt(row["e0_cosine"]),
            _fmt(row["edev_cosine"]),
            row["nfe"],
            _fmt(row["images_per_second"]),
            int(row["peak_vram_bytes"]),
            row["severe_failure_count"],
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    _write_exclusive_text(md_path, "\n".join(lines) + "\n")


def classify_full_result(
    native: Mapping[str, Any],
    winner: Mapping[str, Any],
    heldout: Mapping[str, Any],
    *,
    severe_failure_count: int,
) -> dict[str, Any]:
    _require_finite_tree(native, "full native result")
    _require_finite_tree(winner, "full winner result")
    severe = _integer(severe_failure_count, "severe visual failure count")
    heldout_improvements = _heldout_improvements(heldout)
    solved_checks = {
        "fid": _finite(winner["fid"], "winner FID") <= _finite(native["fid"], "native FID") + 3.0,
        "sharpness_absolute": _finite(winner["sharpness_mean"], "winner Sharpness") >= 300.0,
        "sharpness_retention": _finite(winner["sharpness_mean"], "winner Sharpness")
        >= 0.95 * _finite(native["sharpness_mean"], "native Sharpness"),
        "kid": _finite(winner["kid_mean"], "winner KID")
        <= _finite(native["kid_mean"], "native KID") + 0.005,
        "e0": _finite(winner["e0_cosine"], "winner E0 cosine") >= 0.516,
        "heldout_e1_e2": all(heldout_improvements.values()),
        "visual": severe <= 3,
        "finite": bool(native.get("all_finite")) and bool(winner.get("all_finite")),
    }
    directional_checks = {
        "e0_delta": _finite(winner["e0_cosine"], "winner E0 cosine")
        - _finite(native["e0_cosine"], "native E0 cosine")
        >= 0.05,
        "sharpness_safety": _finite(winner["sharpness_mean"], "winner Sharpness")
        >= 0.80 * _finite(native["sharpness_mean"], "native Sharpness"),
        "visual_safety": severe <= 6,
        "finite": bool(native.get("all_finite")) and bool(winner.get("all_finite")),
    }
    if all(solved_checks.values()):
        label = "solved"
    elif all(directional_checks.values()):
        label = "directional_evidence"
    else:
        label = "failed"
    return {
        "label": label,
        "solved_checks": solved_checks,
        "directional_checks": directional_checks,
        "heldout_improvements": heldout_improvements,
    }


def summarize_full(root: Path) -> dict[str, Any]:
    root = Path(root)
    selection = _read_json(root / "selection.json", "selection")
    heldout = _read_json(root / "heldout_e1_e2.json", "held-out evaluation")
    visual = _read_json(root / "visual_review.json", "visual_review")
    winner_id = str(_nested(selection, ("winner", "arm_id"), "winner arm ID"))
    severe = _integer(
        _nested(visual, ("arms", winner_id, "severe_failure_count"), "winner severe count"),
        "winner severe count",
    )
    native = _full_quality_row(root / "full/merged/native", e0_key="native")
    winner = _full_quality_row(root / "full/merged/winner", e0_key="candidate")
    decision = classify_full_result(native, winner, heldout, severe_failure_count=severe)
    payload = {
        "schema_version": 1,
        "winner": selection["winner"],
        "native": native,
        "winner_result": winner,
        "heldout": heldout,
        "decision": decision,
    }
    _write_exclusive_json(root / "final_summary.json", payload)
    return payload


def _full_quality_row(path: Path, *, e0_key: str) -> dict[str, Any]:
    quality = _read_json(path / "quality.json", f"{e0_key} quality")
    generation_path = path / "generation_result.json"
    if generation_path.is_file():
        generation = _read_json(generation_path, f"{e0_key} generation")
        cosine_key = "native_e0_target" if e0_key == "native" else "candidate_e0_target"
        e0_cosine = _summary_mean(generation["cosine"], cosine_key)
    else:
        per_sample_key = "native_cosine" if e0_key == "native" else "candidate_cosine"
        values = [
            _finite(row.get(per_sample_key), f"{e0_key} per-sample cosine")
            for row in _read_jsonl(path / "per_sample.jsonl")
        ]
        if not values:
            raise ValueError(f"{e0_key} merged per-sample manifest contains no rows")
        e0_cosine = sum(values) / len(values)
    return {
        "fid": _finite(quality["fid"], "FID"),
        "kid_mean": _finite(quality["kid_mean"], "KID"),
        "sharpness_mean": _nested_finite(quality, ("sharpness", "mean"), "Sharpness"),
        "e0_cosine": e0_cosine,
        "all_finite": True,
    }


def _heldout_improvements(heldout: Mapping[str, Any]) -> dict[str, bool]:
    encoders = heldout.get("encoders")
    if not isinstance(encoders, Mapping) or set(encoders) != {
        "e1_dinov2_large_v2",
        "e2_convnext_tiny",
    }:
        raise ValueError("held-out result must contain exactly fixed E1 and E2")
    result = {}
    for name, payload in encoders.items():
        native = _nested_finite(
            payload,
            ("native", "paired_source_generated_cosine", "mean"),
            f"{name} native cosine",
        )
        winner = _nested_finite(
            payload,
            ("winner", "paired_source_generated_cosine", "mean"),
            f"{name} winner cosine",
        )
        result[name] = winner > native
    return result


def _fmrg_schedule_contract(
    *,
    arm_id: str,
    mode: str,
    generation: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if mode not in {"official_head_current_xt", "paper_algorithm_split"}:
        return {}
    schedule = generation.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError(f"arm {arm_id}: FMRG generation is missing its locked schedule")
    if schedule.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"arm {arm_id}: locked schedule checkpoint SHA256 disagrees")
    t_cut = _finite(schedule.get("t_cut"), "locked schedule t_cut")
    config_t_cut = _finite(config.get("t_cut"), "config t_cut")
    if not 0.0 < t_cut < 1.0 or not math.isclose(
        config_t_cut, t_cut, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"arm {arm_id}: config t_cut disagrees with the locked schedule")
    guided = [
        1.0,
        1.0 - (1.0 - t_cut) / 3.0,
        1.0 - 2.0 * (1.0 - t_cut) / 3.0,
        t_cut,
    ]
    unguided = [t_cut, t_cut / 2.0, 0.0]
    if not _float_sequences_equal(schedule.get("guided_times"), guided):
        raise ValueError(f"arm {arm_id}: locked guided schedule is not the registered 3-step schedule")
    if not _float_sequences_equal(schedule.get("unguided_times"), unguided):
        raise ValueError(f"arm {arm_id}: locked unguided schedule is not the registered 2-step tail")
    manifest = str(schedule.get("manifest", ""))
    if not manifest or str(config.get("schedule_manifest", "")) != manifest:
        raise ValueError(f"arm {arm_id}: schedule manifest path disagrees with the resolved config")
    return {
        "t_cut": t_cut,
        "schedule_manifest": manifest,
        "schedule_manifest_sha256": _sha(
            schedule.get("manifest_sha256"), "schedule manifest SHA256"
        ),
        "schedule_contract_sha256": _sha(
            schedule.get("schedule_contract_sha256"), "schedule contract SHA256"
        ),
    }


def _float_sequences_equal(value: Any, expected: Sequence[float]) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != len(expected):
        return False
    try:
        return all(
            math.isclose(float(actual), target, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, target in zip(value, expected, strict=True)
        )
    except (TypeError, ValueError):
        return False


def _reject_heldout_outputs(payload: Mapping[str, Any]) -> None:
    forbidden = {
        "heldout_results",
        "heldout_metrics",
        "e1_cosine",
        "e2_cosine",
        "heldout_e1_e2",
    }
    present = sorted(forbidden.intersection(payload))
    if present:
        raise ValueError(f"calibration contains forbidden held-out outputs: {present!r}")


def _native_sharpness_mean(per_sample_path: Path) -> float:
    if not per_sample_path.is_file():
        raise FileNotFoundError(
            "matched-native Sharpness is absent and per_sample.jsonl is unavailable: "
            f"{per_sample_path}"
        )
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for matched-native Sharpness") from exc
    values = []
    for row in _read_jsonl(per_sample_path):
        path = Path(str(row.get("native", "")))
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"failed to read matched-native image: {path}")
        values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    if len(values) != 64 or not np.isfinite(values).all():
        raise ValueError("matched-native Sharpness requires exactly 64 finite images")
    return float(np.mean(values))


def _optional_full_id_digest(root: Path) -> str | None:
    manifest = root / "manifests/full_2048.jsonl"
    if not manifest.is_file():
        return None
    ids = [str(row["sample_id"]) for row in _read_jsonl(manifest)]
    if len(ids) != 2048 or len(set(ids)) != 2048:
        raise ValueError("full sample manifest must contain exactly 2048 unique IDs")
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the locked R8 MeanFlow study.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "full"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = (
            summarize_calibration(args.root)
            if args.phase == "calibration"
            else summarize_full(args.root)
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _summary_mean(payload: Mapping[str, Any], key: str) -> float:
    return _nested_finite(payload, (key, "mean"), f"{key} mean")


def _nested(payload: Mapping[str, Any], keys: Sequence[str], label: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing {label}")
        value = value[key]
    return value


def _nested_finite(payload: Mapping[str, Any], keys: Sequence[str], label: str) -> float:
    return _finite(_nested(payload, keys, label), label)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def _require_finite_tree(payload: Any, label: str) -> None:
    if isinstance(payload, Mapping):
        for value in payload.values():
            _require_finite_tree(value, label)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _require_finite_tree(value, label)
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        if not math.isfinite(float(payload)):
            raise ValueError(f"{label} contains a non-finite value")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_exclusive_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_exclusive_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, content.encode())
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    return f"{float(value):.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
