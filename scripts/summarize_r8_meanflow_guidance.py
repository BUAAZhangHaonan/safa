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

from safa.evaluation.r8_visual_evidence import validate_visual_review_arm
from safa.evaluation.r8_arm_contracts import (
    canonical_arm_config_digest,
    require_arm_config_digest,
)


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
        "repeated_tiled_face_or_image_pattern",
        "near_uniform_or_blank_frame",
        "severe_saturation_or_clipping_destroying_global_structure",
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
    if str(config.get("mode", "")) != str(generation.get("mode", "")):
        raise ValueError(f"arm {arm_id}: generation/config modes disagree")
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
    digest_config = dict(config)
    if schedule_contract:
        digest_config["schedule_contract_sha256"] = schedule_contract[
            "schedule_contract_sha256"
        ]
    arm_config_sha256 = canonical_arm_config_digest(digest_config)
    generation_arm_digest = require_arm_config_digest(
        generation.get("arm_config_sha256"), f"arm {arm_id} config SHA256"
    )
    if generation_arm_digest != arm_config_sha256:
        raise ValueError(f"arm {arm_id}: generation arm config SHA256 disagrees")

    row = {
        "arm_id": arm_id,
        "mode": mode,
        "sample_count": generation_count,
        "sample_id_sha256": generation_digest,
        "checkpoint_sha256": checkpoint_sha,
        "seed": seed,
        "arm_config_sha256": arm_config_sha256,
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
            "arm_config_sha256",
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
    evidence = _read_json(root / "calibration/visual_evidence.json", "calibration visual evidence")
    if visual.get("reviewed_sample_count") != 64 or evidence.get("sample_count") != 64:
        raise ValueError("visual_review must record exactly 64 reviewed samples")
    reviews = visual.get("arms")
    evidence_arms = evidence.get("arms")
    if not isinstance(reviews, Mapping) or not isinstance(evidence_arms, Mapping):
        raise ValueError("visual_review and evidence must contain arms mappings")
    if set(reviews) != set(evidence_arms):
        raise ValueError("visual_review arm IDs must exactly match visual evidence")
    normalized_reviews = {
        arm_id: validate_visual_review_arm(reviews[arm_id], evidence_arms[arm_id])
        for arm_id in reviews
    }
    if visual.get("passed") is not all(review["passed"] for review in normalized_reviews.values()):
        raise ValueError("visual_review top-level pass disagrees with reviewed arms")
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
    if set(normalized_reviews) != arm_ids:
        raise ValueError(
            "visual_review arm IDs must exactly match completed calibration arm IDs: "
            f"completed={sorted(arm_ids)!r} reviewed={sorted(normalized_reviews)!r}"
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
            validate_calibration_arm(
                arm_dir.name, generation, quality, normalized_reviews[arm_dir.name]
            )
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
        "contracts": bool(native.get("contract_validated"))
        and bool(winner.get("contract_validated")),
    }
    directional_checks = {
        "e0_delta": _finite(winner["e0_cosine"], "winner E0 cosine")
        - _finite(native["e0_cosine"], "native E0 cosine")
        >= 0.05,
        "sharpness_safety": _finite(winner["sharpness_mean"], "winner Sharpness")
        >= 0.80 * _finite(native["sharpness_mean"], "native Sharpness"),
        "visual_safety": severe <= 6,
        "finite": bool(native.get("all_finite")) and bool(winner.get("all_finite")),
        "contracts": bool(native.get("contract_validated"))
        and bool(winner.get("contract_validated")),
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
    marker = _read_json(root / "heldout_protocol_marker.json", "held-out protocol marker")
    visual_path = root / "full/visual_review.json"
    visual = _read_json(visual_path, "full visual_review")
    visual_evidence_path = root / "full/visual_evidence.json"
    visual_evidence = _read_json(visual_evidence_path, "full visual evidence")
    finalization = _read_json(
        root / "full/finalization_completion.json", "full finalization completion"
    )
    winner_id = str(_nested(selection, ("winner", "arm_id"), "winner arm ID"))
    _validate_full_selection(selection, root)
    evidence_arms = visual_evidence.get("arms")
    review_arms = visual.get("arms")
    if not isinstance(evidence_arms, Mapping) or set(evidence_arms) != {winner_id}:
        raise ValueError("full visual evidence must contain exactly the locked winner arm")
    if not isinstance(review_arms, Mapping) or set(review_arms) != {winner_id}:
        raise ValueError("full visual review must contain exactly the locked winner arm")
    normalized_review = validate_visual_review_arm(
        review_arms[winner_id], evidence_arms[winner_id]
    )
    if visual.get("reviewed_sample_count") != 64 or visual.get("passed") is not True:
        raise ValueError("full visual review must pass exactly 64 locked samples")
    severe = normalized_review["severe_failure_count"]
    if finalization.get("status") != "complete":
        raise ValueError("full finalization completion marker is not complete")
    if finalization.get("sample_id_manifest_sha256") != _sha256_file(
        root / "manifests/full_2048.jsonl"
    ):
        raise ValueError("full finalization manifest SHA256 disagrees")
    if finalization.get("visual_evidence_sha256") != _sha256_file(visual_evidence_path):
        raise ValueError("full finalization visual evidence SHA256 disagrees")
    if finalization.get("visual_review_sha256") != _sha256_file(visual_path):
        raise ValueError("full finalization visual review SHA256 disagrees")
    native = _full_quality_row(root / "full/merged/native", e0_key="native")
    winner = _full_quality_row(root / "full/merged/winner", e0_key="candidate")
    for arm_id, result in (("native", native), ("winner", winner)):
        arm_finalization = _nested(
            finalization, ("arms", arm_id), f"full {arm_id} finalization contract"
        )
        if not isinstance(arm_finalization, Mapping) or arm_finalization.get(
            "merge_contract_sha256"
        ) != result.get(
            "merge_contract_sha256"
        ) or arm_finalization.get("arm_config_sha256") != result.get(
            "arm_config_sha256"
        ) or arm_finalization.get("quality_sha256") != _sha256_file(
            root / f"full/merged/{arm_id}/quality.json"
        ):
            raise ValueError(f"full {arm_id} finalization assets disagree")
    _validate_full_result_contracts(selection, native, winner, root)
    _validate_heldout_result_contract(selection, heldout, marker, root)
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


def _validate_full_selection(selection: Mapping[str, Any], root: Path) -> None:
    if selection.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("full selection checkpoint SHA256 disagrees")
    if selection.get("full_sample_count") != 2048:
        raise ValueError("full selection must lock exactly 2048 samples")
    manifest_path = root / "manifests/full_2048.jsonl"
    if selection.get("full_sample_id_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("full selection manifest file SHA256 disagrees")
    manifest_ids = [str(row.get("sample_id", "")) for row in _read_jsonl(manifest_path)]
    if len(manifest_ids) != 2048 or len(set(manifest_ids)) != 2048:
        raise ValueError("full selection manifest must contain exactly 2048 unique IDs")
    if selection.get("full_sample_id_sha256") != _sample_id_digest(manifest_ids):
        raise ValueError("full selection ordered sample-ID digest disagrees")
    winner = selection.get("winner")
    if not isinstance(winner, Mapping):
        raise ValueError("full selection is missing the locked winner")
    config_path = Path(str(winner.get("config", "")))
    if _sha256_file(config_path) != _sha(winner.get("config_sha256"), "winner config SHA256"):
        raise ValueError("full selection winner config file SHA256 disagrees")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("full selection winner config must be a mapping")
    locked_arm_digest = require_arm_config_digest(
        winner.get("arm_config_sha256"), "winner arm config SHA256"
    )
    if canonical_arm_config_digest(config) != locked_arm_digest:
        raise ValueError("full selection winner canonical arm config SHA256 disagrees")


def _validate_full_result_contracts(
    selection: Mapping[str, Any],
    native: Mapping[str, Any],
    winner: Mapping[str, Any],
    root: Path,
) -> None:
    manifest_ids = [
        str(row["sample_id"]) for row in _read_jsonl(root / "manifests/full_2048.jsonl")
    ]
    expected_sample_digest = _sample_id_digest(manifest_ids)
    if native.get("sample_id_sha256") != expected_sample_digest or winner.get(
        "sample_id_sha256"
    ) != expected_sample_digest:
        raise ValueError("full native/winner sample-ID contracts disagree with the locked manifest")
    if native.get("mode") != "native" or native.get("schedule") is not None:
        raise ValueError("full native result must use native mode without a guidance schedule")
    winner_contract = selection["winner"]
    if winner.get("mode") != winner_contract.get("mode"):
        raise ValueError("full winner mode disagrees with the locked selection")
    if winner.get("arm_config_sha256") != winner_contract.get("arm_config_sha256"):
        raise ValueError("full winner arm config SHA256 disagrees with locked selection")
    if winner_contract.get("mode") in {"official_head_current_xt", "paper_algorithm_split"}:
        schedule = winner.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError("full FMRG winner is missing the locked schedule")
        for field in ("schedule_manifest_sha256", "schedule_contract_sha256"):
            schedule_field = "manifest_sha256" if field == "schedule_manifest_sha256" else field
            if schedule.get(schedule_field) != winner_contract.get(field):
                raise ValueError(f"full winner {field} disagrees with selection")
        if not math.isclose(
            _finite(schedule.get("t_cut"), "full winner schedule t_cut"),
            _finite(winner_contract.get("t_cut"), "selection winner t_cut"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("full winner t_cut disagrees with selection")


def _validate_heldout_result_contract(
    selection: Mapping[str, Any],
    heldout: Mapping[str, Any],
    marker: Mapping[str, Any],
    root: Path,
) -> None:
    contract = heldout.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("held-out result is missing its prospective contract")
    contract_sha = _canonical_contract_sha256(contract, "contract_sha256")
    if contract.get("contract_sha256") != contract_sha:
        raise ValueError("held-out result contract canonical SHA256 disagrees")
    if marker.get("status") != "complete" or marker.get("contract_sha256") != contract_sha:
        raise ValueError("held-out protocol marker/result contract digests disagree")
    if marker.get("contract") != contract:
        raise ValueError("held-out protocol marker contract payload disagrees with result")
    result_path = root / "heldout_e1_e2.json"
    if Path(str(marker.get("result", ""))).resolve() != result_path.resolve():
        raise ValueError("held-out protocol marker result path disagrees")
    if marker.get("result_sha256") != _sha256_file(result_path):
        raise ValueError("held-out protocol marker result SHA256 disagrees")
    if contract.get("winner_config_sha256") != selection["winner"].get("config_sha256"):
        raise ValueError("held-out winner config contract disagrees with selection")
    if contract.get("winner_arm_config_sha256") != selection["winner"].get(
        "arm_config_sha256"
    ):
        raise ValueError("held-out winner arm config digest disagrees with selection")
    if contract.get("sample_id_sha256") != selection.get("full_sample_id_sha256"):
        raise ValueError("held-out sample-ID contract disagrees with selection")
    if contract.get("full_finalization_sha256") != _sha256_file(
        root / "full/finalization_completion.json"
    ):
        raise ValueError("held-out full finalization contract was replaced")
    for arm_id in ("native", "winner"):
        per_sample = root / f"full/merged/{arm_id}/per_sample.jsonl"
        if _nested(
            contract, (arm_id, "per_sample_sha256"), f"held-out {arm_id} per-sample SHA"
        ) != _sha256_file(per_sample):
            raise ValueError(f"held-out {arm_id} per-sample evidence was replaced")
        if _nested(
            contract,
            (arm_id, "ordered_image_manifest_sha256"),
            f"held-out {arm_id} ordered image manifest SHA",
        ) != _ordered_image_manifest_sha256(per_sample):
            raise ValueError(f"held-out {arm_id} generated image evidence was replaced")


def _full_quality_row(path: Path, *, e0_key: str) -> dict[str, Any]:
    quality = _read_json(path / "quality.json", f"{e0_key} quality")
    generation_path = path / "generation_result.json"
    generation = _read_json(generation_path, f"{e0_key} generation")
    completion = _read_json(path / "completion.json", f"{e0_key} merge completion")
    merge_contract = _read_json(path / "merge_contract.json", f"{e0_key} merge contract")
    per_sample_path = path / "per_sample.jsonl"
    rows = _read_jsonl(per_sample_path)
    if len(rows) != 2048 or generation.get("sample_count") != 2048:
        raise ValueError(f"{e0_key} full generation must contain exactly 2048 samples")
    if quality.get("num_generated") != 2048 or quality.get("num_real") != 2048:
        raise ValueError(f"{e0_key} full quality must contain exactly 2048 generated/real samples")
    if quality.get("sample_id_count") != 2048:
        raise ValueError(f"{e0_key} quality manifest must contain exactly 2048 IDs")
    ordered_ids = [str(row.get("sample_id", "")) for row in rows]
    sample_digest = _sample_id_digest(ordered_ids)
    if generation.get("sample_id_sha256") != sample_digest or quality.get(
        "sample_id_sha256"
    ) != sample_digest:
        raise ValueError(f"{e0_key} generation/quality ordered sample digests disagree")
    if completion.get("status") != "complete" or completion.get("sample_count") != 2048:
        raise ValueError(f"{e0_key} merge completion contract is incomplete")
    if completion.get("per_sample_sha256") != _sha256_file(per_sample_path):
        raise ValueError(f"{e0_key} per-sample file SHA256 disagrees with completion")
    if completion.get("generation_result_sha256") != _sha256_file(generation_path):
        raise ValueError(f"{e0_key} generation file SHA256 disagrees with completion")
    arm_config_sha256 = require_arm_config_digest(
        generation.get("arm_config_sha256"), f"{e0_key} arm config SHA256"
    )
    if completion.get("arm_config_sha256") != arm_config_sha256:
        raise ValueError(f"{e0_key} completion arm config SHA256 disagrees")
    config = generation.get("config")
    if not isinstance(config, Mapping) or canonical_arm_config_digest(config) != arm_config_sha256:
        raise ValueError(f"{e0_key} canonical arm config SHA256 disagrees")
    if completion.get("merge_contract_sha256") != merge_contract.get(
        "merge_contract_sha256"
    ):
        raise ValueError(f"{e0_key} merge completion/contract digests disagree")
    checkpoint = generation.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError(f"{e0_key} checkpoint SHA256 is not the locked epoch-1652 checkpoint")
    expected_checkpoint_state = {
        "stage": "stage2",
        "stage_epoch_1based": 1652,
        "sit_patch_size": 4,
        "weight_source": "ema_model_state_dict",
    }
    for field, expected in expected_checkpoint_state.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"{e0_key} checkpoint state field {field} disagrees")
    if generation.get("seed") != 1337:
        raise ValueError(f"{e0_key} full generation seed must be 1337")
    cosine_key = "native_e0_target" if e0_key == "native" else "candidate_e0_target"
    e0_cosine = _summary_mean(generation["cosine"], cosine_key)
    return {
        "fid": _finite(quality["fid"], "FID"),
        "kid_mean": _finite(quality["kid_mean"], "KID"),
        "sharpness_mean": _nested_finite(quality, ("sharpness", "mean"), "Sharpness"),
        "e0_cosine": e0_cosine,
        "all_finite": True,
        "contract_validated": True,
        "mode": generation.get("mode"),
        "sample_id_sha256": sample_digest,
        "checkpoint": dict(checkpoint),
        "seed": 1337,
        "schedule": generation.get("schedule"),
        "merge_contract_sha256": merge_contract.get("merge_contract_sha256"),
        "arm_config_sha256": arm_config_sha256,
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


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _ordered_image_manifest_sha256(per_sample_path: Path) -> str:
    lines = []
    for row in _read_jsonl(per_sample_path):
        sample_id = str(row.get("sample_id", ""))
        generated = Path(str(row.get("generated", "")))
        lines.append(f"{sample_id}\t{generated}\t{_sha256_file(generated)}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _canonical_contract_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    contract = dict(payload)
    contract.pop(digest_field, None)
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fmt(value: Any) -> str:
    return f"{float(value):.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
