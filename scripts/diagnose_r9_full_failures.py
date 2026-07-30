#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Mapping, Sequence


ROLES = ("source", "native", "candidate")
FACE_COUNT_FIELDS = {
    "source": "source_face_count",
    "native": "native_face_count",
    "candidate": "candidate_face_count",
}
PAIR_FIELDS = (
    "sample_id",
    "seed",
    "native_sharpness",
    "candidate_sharpness",
    "delta_sharpness",
)
DETECTOR_MODEL = "buffalo_l"
DETECTOR_SIZE = [224, 224]


class DiagnosticError(ValueError):
    """Raised when archived or newly observed diagnostic evidence is invalid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the four archived R9 Full ArcFace failures."
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def _reject_json_constant(value: str) -> None:
    raise DiagnosticError(f"JSON contains non-finite constant: {value}")


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise DiagnosticError(f"invalid {label}: {path}") from exc


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"{label} must be a JSON object")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise DiagnosticError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise DiagnosticError(f"{label} must be finite")
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample_paths(
    samples: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(samples, list) or not samples:
        raise DiagnosticError("ArcFace request samples must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    role_paths: dict[str, set[Path]] = {role: set() for role in ROLES}
    for index, raw_sample in enumerate(samples):
        sample = require_mapping(raw_sample, f"sample {index}")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise DiagnosticError(f"sample {index} has an invalid sample_id")
        if sample_id in by_id:
            raise DiagnosticError(f"duplicate sample_id: {sample_id}")
        row: dict[str, Any] = {"sample_id": sample_id}
        sample_paths: set[Path] = set()
        for role in ROLES:
            raw_path = sample.get(role)
            if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                raise DiagnosticError(
                    f"{sample_id}: {role} path must be absolute"
                )
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"{sample_id}: missing {role} image: {path}"
                )
            if path in sample_paths:
                raise DiagnosticError(
                    f"{sample_id}: duplicate source/native/candidate path: {path}"
                )
            if path in role_paths[role]:
                raise DiagnosticError(f"duplicate {role} image path: {path}")
            sample_paths.add(path)
            role_paths[role].add(path)
            digest_field = f"{role}_sha256"
            expected_digest = sample.get(digest_field)
            if (
                not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or any(c not in "0123456789abcdef" for c in expected_digest)
            ):
                raise DiagnosticError(
                    f"{sample_id}: invalid {digest_field}"
                )
            row[role] = path
            row[digest_field] = expected_digest
        normalized.append(row)
        by_id[sample_id] = row
    return normalized, by_id


def normalize_pair_rows(
    pair_rows: Any, sample_ids: Sequence[str], observation_count: Any
) -> list[dict[str, Any]]:
    if not isinstance(pair_rows, list) or len(pair_rows) != 2048:
        raise DiagnosticError("automatic evidence must contain 2048 paired rows")
    if observation_count != len(pair_rows):
        raise DiagnosticError("paired observation_count disagrees with rows")
    normalized_pairs: list[dict[str, Any]] = []
    pair_ids: list[str] = []
    seen_pair_ids: set[str] = set()
    for index, raw_pair in enumerate(pair_rows):
        pair = require_mapping(raw_pair, f"paired row {index}")
        sample_id = pair.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in seen_pair_ids:
            raise DiagnosticError("paired rows contain an invalid or duplicate ID")
        seen_pair_ids.add(sample_id)
        pair_ids.append(sample_id)
        seed = pair.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DiagnosticError(f"{sample_id}: paired seed is invalid")
        native = finite_float(pair.get("native_sharpness"), "native sharpness")
        candidate = finite_float(
            pair.get("candidate_sharpness"), "candidate sharpness"
        )
        normalized_pairs.append(
            {
                "sample_id": sample_id,
                "seed": seed,
                "native_sharpness": native,
                "candidate_sharpness": candidate,
                "delta_sharpness": candidate - native,
            }
        )
    if pair_ids != list(sample_ids):
        raise DiagnosticError("paired sharpness rows do not align with request samples")
    return normalized_pairs


def validate_archives(
    automatic_evidence: Any,
    request: Any,
    result: Any,
    formal_evidence: Any,
) -> dict[str, Any]:
    automatic = require_mapping(automatic_evidence, "automatic evidence")
    request_map = require_mapping(request, "ArcFace request")
    result_map = require_mapping(result, "ArcFace result")
    formal = require_mapping(formal_evidence, "formal ArcFace evidence")
    if request_map.get("task") != "arcface":
        raise DiagnosticError("archived evaluator request is not ArcFace")
    payload = require_mapping(request_map.get("payload"), "ArcFace request payload")
    samples, samples_by_id = validate_sample_paths(payload.get("samples"))
    sample_ids = [sample["sample_id"] for sample in samples]

    config = require_mapping(request_map.get("config"), "evaluator config")
    contract = require_mapping(config.get("arcface"), "ArcFace contract")
    if contract.get("model_name") != DETECTOR_MODEL:
        raise DiagnosticError("ArcFace request does not use buffalo_l")
    if contract.get("det_size") != DETECTOR_SIZE:
        raise DiagnosticError("ArcFace request does not use det_size 224x224")

    raw_rows = result_map.get("result")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(samples):
        raise DiagnosticError("ArcFace raw rows do not align with request samples")
    formal_arcface = require_mapping(formal.get("arcface"), "formal ArcFace section")
    formal_rows = formal_arcface.get("rows")
    if not isinstance(formal_rows, list) or len(formal_rows) != len(raw_rows):
        raise DiagnosticError("formal ArcFace rows do not align with raw rows")
    raw_ids: list[str] = []
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    failure_ids: list[str] = []
    for index, raw_row in enumerate(raw_rows):
        row = require_mapping(raw_row, f"ArcFace row {index}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in raw_by_id:
            raise DiagnosticError("ArcFace rows contain an invalid or duplicate ID")
        raw_ids.append(sample_id)
        raw_by_id[sample_id] = row
        counts = []
        for role in ROLES:
            count = row.get(FACE_COUNT_FIELDS[role])
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise DiagnosticError(
                    f"{sample_id}: archived {role} face count is invalid"
                )
            counts.append(count)
        if counts != [1, 1, 1]:
            failure_ids.append(sample_id)
    if raw_ids != sample_ids:
        raise DiagnosticError("ArcFace raw row order disagrees with request samples")
    for index, (raw_row, formal_row) in enumerate(
        zip(raw_rows, formal_rows, strict=True)
    ):
        formal_row = require_mapping(formal_row, f"formal ArcFace row {index}")
        binding_fields = ("sample_id", *FACE_COUNT_FIELDS.values())
        if any(formal_row.get(field) != raw_row.get(field) for field in binding_fields):
            raise DiagnosticError(
                "formal ArcFace IDs/counts disagree with raw evaluator rows"
            )
    archived_failure_ids = formal_arcface.get("failure_sample_ids")
    if archived_failure_ids != failure_ids:
        raise DiagnosticError("formal failure IDs disagree with raw ArcFace rows")
    if len(failure_ids) != 4:
        raise DiagnosticError(
            f"expected four union ArcFace failure IDs, found {len(failure_ids)}"
        )

    arms = automatic.get("arms")
    if not isinstance(arms, list):
        raise DiagnosticError("automatic evidence arms must be a list")
    matching_arms = [
        require_mapping(arm, "automatic evidence arm")
        for arm in arms
        if isinstance(arm, Mapping) and arm.get("arm_id") == payload.get("arm_id")
    ]
    if len(matching_arms) != 1:
        raise DiagnosticError("automatic evidence has a missing or duplicate winner arm")
    paired = require_mapping(
        matching_arms[0].get("paired_metric_rows"), "paired metric rows"
    )
    normalized_pairs = normalize_pair_rows(
        paired.get("rows"), sample_ids, paired.get("observation_count")
    )

    for sample_id in failure_ids:
        sample = samples_by_id[sample_id]
        for role in ROLES:
            actual = sha256_file(sample[role])
            if actual != sample[f"{role}_sha256"]:
                raise DiagnosticError(
                    f"{sample_id}: archived {role} image digest mismatch"
                )
    return {
        "samples": samples,
        "samples_by_id": samples_by_id,
        "raw_by_id": raw_by_id,
        "failure_ids": failure_ids,
        "pair_rows": normalized_pairs,
        "arcface_contract": contract,
        "archived_device": config.get("device"),
    }


def serialize_faces(faces: Any, image_shape: Sequence[int]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for ArcFace diagnostics") from exc
    if not isinstance(faces, Sequence):
        raise DiagnosticError("ArcFace analyzer returned a non-sequence")
    if len(image_shape) not in (2, 3):
        raise DiagnosticError("decoded image shape must have two or three axes")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise DiagnosticError("decoded image has an invalid shape")
    serialized = []
    for index, face in enumerate(faces):
        bbox = np.asarray(getattr(face, "bbox", None), dtype=np.float64)
        kps = np.asarray(getattr(face, "kps", None), dtype=np.float64)
        score = finite_float(
            getattr(face, "det_score", None), f"face {index} det_score"
        )
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise DiagnosticError(f"face {index} has an invalid bbox")
        if kps.shape != (5, 2) or not np.isfinite(kps).all():
            raise DiagnosticError(f"face {index} has invalid five-point landmarks")
        area = max(0.0, float(bbox[2] - bbox[0])) * max(
            0.0, float(bbox[3] - bbox[1])
        )
        serialized.append(
            {
                "bbox": [float(value) for value in bbox],
                "det_score": score,
                "kps": [[float(value) for value in point] for point in kps],
                "bbox_area_ratio": area / float(height * width),
            }
        )
    return {
        "image_shape": [int(value) for value in image_shape],
        "face_count": len(serialized),
        "faces": serialized,
    }


def observe_twice(analyzer: Any, image: Any) -> list[dict[str, Any]]:
    observations = [
        serialize_faces(analyzer.get(image), image.shape),
        serialize_faces(analyzer.get(image), image.shape),
    ]
    counts = [observation["face_count"] for observation in observations]
    if counts[0] != counts[1]:
        raise DiagnosticError(
            f"ArcFace repeated face counts are nondeterministic: {counts}"
        )
    return observations


def laplacian_variance(image: Any) -> float:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for sharpness validation") from exc
    if image is None:
        raise DiagnosticError("cannot compute sharpness for an absent image")
    gray = (
        image
        if getattr(image, "ndim", None) == 2
        else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )
    value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if not math.isfinite(value):
        raise DiagnosticError("Laplacian variance is non-finite")
    return value


def summarize_sharpness(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 2048:
        raise DiagnosticError("sharpness summary requires exactly 2048 paired rows")
    ordered = sorted(rows, key=lambda row: (row["native_sharpness"], row["sample_id"]))
    buckets: list[list[Mapping[str, Any]]] = [[] for _ in range(10)]
    for rank, row in enumerate(ordered):
        buckets[min(9, rank * 10 // len(ordered))].append(row)

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        native_values = [float(row["native_sharpness"]) for row in group]
        candidate_values = [float(row["candidate_sharpness"]) for row in group]
        deltas = [float(row["delta_sharpness"]) for row in group]
        native_mean = statistics.fmean(native_values)
        candidate_mean = statistics.fmean(candidate_values)
        return {
            "count": len(group),
            "native_mean": native_mean,
            "candidate_mean": candidate_mean,
            "delta_mean": statistics.fmean(deltas),
            "delta_median": statistics.median(deltas),
            "candidate_gt_native_count": sum(delta > 0.0 for delta in deltas),
            "candidate_gt_native_fraction": (
                sum(delta > 0.0 for delta in deltas) / len(group)
            ),
            "candidate_to_native_mean_ratio": (
                candidate_mean / native_mean if native_mean > 0.0 else None
            ),
        }

    deciles = []
    for index, bucket in enumerate(buckets, start=1):
        values = summarize(bucket)
        values["native_sharpness_decile"] = index
        values["native_min"] = float(bucket[0]["native_sharpness"])
        values["native_max"] = float(bucket[-1]["native_sharpness"])
        deciles.append(values)
    overall = summarize(rows)
    overall["native_sharpness_definition"] = "grayscale_laplacian_variance"
    overall["decile_basis"] = "ranked_native_sharpness"
    return deciles, overall


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_overlay(path: Path, image: Any, observation: Mapping[str, Any]) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for overlays") from exc
    canvas = image.copy()
    for index, face in enumerate(observation["faces"]):
        x1, y1, x2, y2 = [int(round(value)) for value in face["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{index}: {face['det_score']:.5f}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        for point in face["kps"]:
            cv2.circle(
                canvas,
                (int(round(point[0])), int(round(point[1]))),
                2,
                (0, 0, 255),
                -1,
            )
    cv2.putText(
        canvas,
        f"faces={observation['face_count']} det=buffalo_l/224",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise DiagnosticError(f"failed to write overlay: {path}")


def build_conclusion(
    arcface_rows: Sequence[Mapping[str, Any]], sharpness: Mapping[str, Any]
) -> str:
    archived_failures = []
    for row in arcface_rows:
        archived_failures.append(
            f"- `{row['sample_id']}`: archived counts "
            f"source/native/candidate = "
            f"{row['archived_counts']['source']}/"
            f"{row['archived_counts']['native']}/"
            f"{row['archived_counts']['candidate']}; "
            f"diagnostic repeat counts match."
        )
    return (
        "# R9 Full failure diagnostic\n\n"
        "The archived exact-one gate fails on four union sample IDs. "
        "This diagnostic reproduces the same detector-count pattern twice per "
        "source/native/candidate image with the locked buffalo_l, 224x224 CUDA "
        "analyzer. It does not change or replace the formal evaluator.\n\n"
        + "\n".join(archived_failures)
        + "\n\n"
        "The existing 2048 paired sharpness rows were analyzed without rerunning "
        "FID or KID. The overall candidate-minus-native mean is "
        f"{sharpness['delta_mean']:.6f}, and the candidate is sharper in "
        f"{sharpness['candidate_gt_native_count']} of {sharpness['count']} pairs. "
        "See `sharpness_deciles.csv` for the native-sharpness-ranked deciles.\n"
    )


def run(args: argparse.Namespace, *, analyzer_factory: Any = None) -> Path:
    full_root = args.full_root.resolve()
    output_dir = args.output_dir.resolve()
    if not full_root.is_dir():
        raise FileNotFoundError(f"missing Full evidence root: {full_root}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    request_path = full_root / "evaluator_runs/arcface/winner/request.json"
    result_path = full_root / "evaluator_runs/arcface/winner/result.json"
    formal_path = full_root / "evaluator_evidence/arcface/winner.json"
    automatic_path = full_root / "automatic_evidence.json"
    archives = validate_archives(
        read_json(automatic_path, "automatic evidence"),
        read_json(request_path, "ArcFace request"),
        read_json(result_path, "ArcFace raw evidence"),
        read_json(formal_path, "formal ArcFace evidence"),
    )
    if not isinstance(args.device, str) or not args.device.startswith("cuda:"):
        raise DiagnosticError("diagnostic analyzer requires an explicit cuda:N device")
    if analyzer_factory is None:
        from safa.evaluation.r9_evaluator_worker import (
            _production_face_analyzer_factory,
        )

        analyzer_factory = _production_face_analyzer_factory
    analyzer = analyzer_factory(archives["arcface_contract"], args.device)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for diagnostics") from exc
    parent = output_dir.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {parent}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        overlays = staging / "overlays"
        overlays.mkdir()
        arcface_rows = []
        for failure_index, sample_id in enumerate(archives["failure_ids"]):
            sample = archives["samples_by_id"][sample_id]
            archived = archives["raw_by_id"][sample_id]
            role_rows = {}
            for role in ROLES:
                image = cv2.imread(str(sample[role]), cv2.IMREAD_COLOR)
                if image is None:
                    raise DiagnosticError(
                        f"{sample_id}: could not decode {role} image"
                    )
                observations = observe_twice(analyzer, image)
                archived_count = archived[FACE_COUNT_FIELDS[role]]
                if observations[0]["face_count"] != archived_count:
                    raise DiagnosticError(
                        f"{sample_id}: diagnostic {role} count disagrees with archive"
                    )
                role_rows[role] = {
                    "path": str(sample[role]),
                    "sha256": sample[f"{role}_sha256"],
                    "runs": observations,
                }
                write_overlay(
                    overlays / f"{failure_index:02d}_{role}.png",
                    image,
                    observations[0],
                )
            arcface_rows.append(
                {
                    "sample_id": sample_id,
                    "archived_counts": {
                        role: archived[FACE_COUNT_FIELDS[role]] for role in ROLES
                    },
                    "roles": role_rows,
                }
            )
        deciles, sharpness_summary = summarize_sharpness(archives["pair_rows"])
        write_json(staging / "arcface_rows.json", arcface_rows)
        write_csv(staging / "sharpness_pairs.csv", archives["pair_rows"], PAIR_FIELDS)
        decile_fields = (
            "native_sharpness_decile",
            "count",
            "native_min",
            "native_max",
            "native_mean",
            "candidate_mean",
            "delta_mean",
            "delta_median",
            "candidate_gt_native_count",
            "candidate_gt_native_fraction",
            "candidate_to_native_mean_ratio",
        )
        write_csv(staging / "sharpness_deciles.csv", deciles, decile_fields)
        summary = {
            "schema_version": 1,
            "contract_type": "safa_r9_full_failure_diagnostic_v1",
            "formal_evaluator_modified": False,
            "arcface": {
                "model_name": DETECTOR_MODEL,
                "det_size": DETECTOR_SIZE,
                "device": args.device,
                "failure_sample_ids": archives["failure_ids"],
                "failure_count": len(archives["failure_ids"]),
                "repeats_per_image": 2,
                "all_repeated_face_counts_equal": True,
            },
            "sharpness": sharpness_summary,
            "inputs": {
                "automatic_evidence": str(automatic_path),
                "arcface_request": str(request_path),
                "arcface_raw_evidence": str(result_path),
                "arcface_formal_evidence": str(formal_path),
            },
            "fid_kid_rerun": False,
        }
        write_json(staging / "diagnostic_summary.json", summary)
        (staging / "conclusion.md").write_text(
            build_conclusion(arcface_rows, sharpness_summary),
            encoding="utf-8",
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
