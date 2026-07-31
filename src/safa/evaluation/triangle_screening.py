from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SELECTOR_DOMAIN = b"safa-triangle-512-v1\0"
BOOTSTRAP_SEED = 91637
STAGE_CAPS = {32: 12, 128: 4, 512: 2}
BREAKTHROUGH_STATUSES = (
    "no_gate_survivor",
    "gate_survivor_no_breakthrough",
    "triangle_breakthrough",
    "privacy_positive_breakthrough",
)
ROW_FIELDS = (
    "sample_id",
    "native_e0",
    "candidate_e0",
    "native_edev",
    "candidate_edev",
    "native_niqe",
    "candidate_niqe",
    "native_sharpness",
    "candidate_sharpness",
    "source_face_count",
    "native_face_count",
    "candidate_face_count",
    "source_native_cosine",
    "source_candidate_cosine",
)
HISTORICAL_METRIC_FIELDS = (
    "e0",
    "edev",
    "fid",
    "kid",
    "niqe",
    "sharpness",
    "arcface_source_candidate_cosine",
)
HISTORICAL_FAMILY_IDS = (
    "E11",
    "E12",
    "E13",
    "E15",
    "E2",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
)


class TriangleScreeningError(ValueError):
    """Raised when triangle-screening inputs violate the locked protocol."""


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    sample_count: int
    candidate_exact_one_count: int
    e0: float
    delta_e0: float
    delta_edev: float
    niqe: float
    native_niqe: float
    fid: float | None
    native_fid: float | None
    kid: float | None
    native_kid: float | None
    sharpness: float
    native_sharpness: float
    arcface_delta: float | None
    arcface_delta_u95: float | None
    hard_gate_pass: bool
    failed_gates: tuple[str, ...]
    r_margin: float
    q_margin: float
    p_margin: float | None
    pareto: bool = False
    selected: bool = False
    status: str = "gate_survivor_no_breakthrough"

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "sample_count": self.sample_count,
            "candidate_exact_one_count": self.candidate_exact_one_count,
            "e0": self.e0,
            "delta_e0": self.delta_e0,
            "delta_edev": self.delta_edev,
            "niqe": self.niqe,
            "native_niqe": self.native_niqe,
            "fid": self.fid,
            "native_fid": self.native_fid,
            "kid": self.kid,
            "native_kid": self.native_kid,
            "sharpness": self.sharpness,
            "native_sharpness": self.native_sharpness,
            "arcface_delta": self.arcface_delta,
            "arcface_delta_u95": self.arcface_delta_u95,
            "hard_gate_pass": self.hard_gate_pass,
            "failed_gates": list(self.failed_gates),
            "R": self.r_margin,
            "Q": self.q_margin,
            "P": self.p_margin,
            "pareto": self.pareto,
            "selected": self.selected,
            "status": self.status,
        }


def _average_tie_percentiles(
    candidates: Sequence[Mapping[str, Any]], field: str, *, sign: float
) -> dict[str, float]:
    ordered = sorted(
        (
            sign * _finite_number(candidate[field], f"{candidate['candidate_id']}.{field}"),
            candidate["candidate_id"],
        )
        for candidate in candidates
    )
    denominator = len(ordered) - 1
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_zero_based_rank = (start + end - 1) / 2.0
        percentile = (
            1.0 if denominator == 0 else average_zero_based_rank / denominator
        )
        for _, candidate_id in ordered[start:end]:
            result[candidate_id] = percentile
        start = end
    return result


def select_historical24(
    candidate_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    family_ids: Sequence[str],
    expected_candidate_count: int = 193,
) -> list[dict[str, Any]]:
    """Select the locked family-balanced 24 from primary historical smoke-8 rows."""
    if len(family_ids) != 12 or len(set(family_ids)) != 12:
        raise TriangleScreeningError("exactly 12 unique family IDs are required")
    if any(not isinstance(family_id, str) or not family_id for family_id in family_ids):
        raise TriangleScreeningError("family IDs must be non-empty strings")
    manifest_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(manifest_rows):
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TriangleScreeningError(
                f"manifest row {index} candidate_id must be a non-empty string"
            )
        if candidate_id in manifest_by_id:
            raise TriangleScreeningError(
                f"duplicate manifest candidate_id: {candidate_id}"
            )
        manifest_by_id[candidate_id] = row
    required = (
        "candidate_id",
        "family_id",
        "smoke8_primary",
        *HISTORICAL_METRIC_FIELDS,
    )
    primary: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(candidate_rows):
        if not isinstance(row, Mapping):
            raise TriangleScreeningError(f"candidate row {index} must be an object")
        missing = [field for field in required if field not in row]
        if missing:
            raise TriangleScreeningError(
                f"candidate row {index} is missing fields: {', '.join(missing)}"
            )
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TriangleScreeningError(
                f"candidate row {index} candidate_id must be non-empty"
            )
        if candidate_id in seen:
            raise TriangleScreeningError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        if row["smoke8_primary"] is True:
            family_id = row["family_id"]
            if family_id not in family_ids:
                raise TriangleScreeningError(
                    f"{candidate_id} has unknown family_id {family_id!r}"
                )
            normalized = dict(row)
            for field in HISTORICAL_METRIC_FIELDS:
                normalized[field] = _finite_number(
                    row[field], f"{candidate_id}.{field}"
                )
            primary.append(normalized)
        elif row["smoke8_primary"] is not False:
            raise TriangleScreeningError(
                f"{candidate_id}.smoke8_primary must be boolean"
            )
    if len(primary) != expected_candidate_count:
        raise TriangleScreeningError(
            f"primary historical candidate count must be {expected_candidate_count}, "
            f"got {len(primary)}"
        )
    primary_ids = {row["candidate_id"] for row in primary}
    if primary_ids != set(manifest_by_id):
        missing = sorted(primary_ids - set(manifest_by_id))
        extra = sorted(set(manifest_by_id) - primary_ids)
        raise TriangleScreeningError(
            "candidate-to-manifest mapping is not exact: "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )

    percentiles = {
        "e0": _average_tie_percentiles(primary, "e0", sign=1.0),
        "edev": _average_tie_percentiles(primary, "edev", sign=1.0),
        "fid": _average_tie_percentiles(primary, "fid", sign=-1.0),
        "kid": _average_tie_percentiles(primary, "kid", sign=-1.0),
        "niqe": _average_tie_percentiles(primary, "niqe", sign=-1.0),
        "sharpness": _average_tie_percentiles(primary, "sharpness", sign=1.0),
        "privacy": _average_tie_percentiles(
            primary, "arcface_source_candidate_cosine", sign=-1.0
        ),
    }
    scored: list[dict[str, Any]] = []
    for row in primary:
        candidate_id = row["candidate_id"]
        r_axis = min(
            percentiles["e0"][candidate_id], percentiles["edev"][candidate_id]
        )
        q_axis = min(
            percentiles["fid"][candidate_id],
            percentiles["kid"][candidate_id],
            percentiles["niqe"][candidate_id],
            percentiles["sharpness"][candidate_id],
        )
        p_axis = percentiles["privacy"][candidate_id]
        scored.append(
            {
                **row,
                "R": r_axis,
                "Q": q_axis,
                "P": p_axis,
                "balance": min(r_axis, q_axis, p_axis),
            }
        )

    selected: list[tuple[dict[str, Any], str]] = []
    selected_ids: set[str] = set()
    for family_id in family_ids:
        family = [row for row in scored if row["family_id"] == family_id]
        if not family:
            raise TriangleScreeningError(f"family {family_id!r} has no candidate")
        winner = min(
            family, key=lambda row: (-row["balance"], row["candidate_id"])
        )
        selected.append((winner, "family_balance"))
        selected_ids.add(winner["candidate_id"])
    for axis in ("R", "Q", "P"):
        remaining = [row for row in scored if row["candidate_id"] not in selected_ids]
        if len(remaining) < 4:
            raise TriangleScreeningError(
                f"fewer than four candidates remain for {axis} selection"
            )
        winners = sorted(
            remaining, key=lambda row: (-row[axis], row["candidate_id"])
        )[:4]
        for winner in winners:
            selected.append((winner, f"top_{axis}"))
            selected_ids.add(winner["candidate_id"])
    if len(selected) != 24 or len(selected_ids) != 24:
        raise TriangleScreeningError("historical selector did not produce 24 unique rows")

    output: list[dict[str, Any]] = []
    for selection_rank, (row, selection_group) in enumerate(selected, 1):
        manifest_row = manifest_by_id[row["candidate_id"]]
        result = {
            "selection_rank": selection_rank,
            "selection_group": selection_group,
            "candidate_id": row["candidate_id"],
            "family_id": row["family_id"],
            "R": row["R"],
            "Q": row["Q"],
            "P": row["P"],
            "balance": row["balance"],
        }
        checkpoint_path = manifest_row.get("checkpoint_path")
        checkpoint_sha256 = manifest_row.get("checkpoint_sha256")
        if checkpoint_path is not None:
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                raise TriangleScreeningError(
                    f"{row['candidate_id']} checkpoint_path must be non-empty"
                )
            result["checkpoint_path"] = checkpoint_path
        if checkpoint_sha256 is not None:
            if (
                not isinstance(checkpoint_sha256, str)
                or len(checkpoint_sha256) != 64
                or any(char not in "0123456789abcdef" for char in checkpoint_sha256)
            ):
                raise TriangleScreeningError(
                    f"{row['candidate_id']} checkpoint_sha256 is invalid"
                )
            result["checkpoint_sha256"] = checkpoint_sha256
        output.append(result)
    return output


def load_historical_primary_artifacts(
    historical_primary_root: Path,
    manifest_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Adapt the authoritative result directory and current candidate manifest."""
    historical_primary_root = historical_primary_root.resolve()
    if not historical_primary_root.is_dir():
        raise TriangleScreeningError(
            f"historical primary root is missing: {historical_primary_root}"
        )
    manifest_candidates = manifest_payload.get("candidates")
    if not isinstance(manifest_candidates, list):
        raise TriangleScreeningError(
            "manifest JSON must contain a candidates list"
        )
    manifest_by_id: dict[str, dict[str, Any]] = {}
    observed_families: set[str] = set()
    for index, row in enumerate(manifest_candidates):
        if not isinstance(row, Mapping):
            raise TriangleScreeningError(
                f"manifest candidate {index} must be an object"
            )
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TriangleScreeningError(
                f"manifest candidate {index} has invalid candidate_id"
            )
        if candidate_id in manifest_by_id:
            raise TriangleScreeningError(
                f"duplicate manifest candidate_id: {candidate_id}"
            )
        family_values = row.get("source_logical_experiment_ids")
        if (
            not isinstance(family_values, list)
            or len(family_values) != 1
            or not isinstance(family_values[0], str)
        ):
            raise TriangleScreeningError(
                f"{candidate_id} must have one source_logical_experiment_id"
            )
        family_id = family_values[0]
        observed_families.add(family_id)
        manifest_by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "checkpoint_path": row.get("checkpoint_path"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "family_id": family_id,
        }
    if observed_families != set(HISTORICAL_FAMILY_IDS):
        raise TriangleScreeningError(
            "manifest families do not equal the locked 12: "
            f"{sorted(observed_families)!r}"
        )
    if len(manifest_by_id) != 193:
        raise TriangleScreeningError(
            f"manifest must contain 193 candidates, got {len(manifest_by_id)}"
        )

    result_paths = sorted(historical_primary_root.glob("*/result.json"))
    if len(result_paths) != 193:
        raise TriangleScreeningError(
            f"historical primary root must contain 193 result.json files, "
            f"got {len(result_paths)}"
        )
    candidate_rows: list[dict[str, Any]] = []
    result_ids: set[str] = set()
    for result_path in result_paths:
        candidate_id = result_path.parent.name
        if candidate_id in result_ids:
            raise TriangleScreeningError(
                f"duplicate historical result candidate: {candidate_id}"
            )
        result_ids.add(candidate_id)
        if candidate_id not in manifest_by_id:
            raise TriangleScreeningError(
                f"historical result is absent from manifest: {candidate_id}"
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TriangleScreeningError(
                f"invalid JSON in {result_path}"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("status") != "completed"
            or payload.get("failure") is not None
        ):
            raise TriangleScreeningError(
                f"historical primary result is not completed: {result_path}"
            )
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TriangleScreeningError(f"missing evidence in {result_path}")
        if (
            evidence.get("mode") != "smoke8"
            or evidence.get("replicate") != "primary"
            or evidence.get("sample_count") != 8
        ):
            raise TriangleScreeningError(
                f"result is not authoritative smoke8 primary: {result_path}"
            )
        quality = evidence.get("quality")
        arcface = evidence.get("arcface")
        if not isinstance(quality, Mapping) or not isinstance(arcface, Mapping):
            raise TriangleScreeningError(
                f"quality/ArcFace evidence is missing in {result_path}"
            )
        canonical_kid = quality.get("canonical_kid")
        iqa = quality.get("iqa")
        sharpness = quality.get("sharpness")
        if (
            not isinstance(canonical_kid, Mapping)
            or not isinstance(iqa, Mapping)
            or not isinstance(sharpness, Mapping)
        ):
            raise TriangleScreeningError(
                f"quality metric objects are missing in {result_path}"
            )
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "family_id": manifest_by_id[candidate_id]["family_id"],
                "smoke8_primary": True,
                "e0": evidence.get("e0_mean"),
                "edev": evidence.get("edev_mean"),
                "fid": quality.get("fid"),
                "kid": canonical_kid.get("kid_mean"),
                "niqe": iqa.get("mean"),
                "sharpness": sharpness.get("mean"),
                "arcface_source_candidate_cosine": arcface.get(
                    "mean_source_candidate_cosine"
                ),
            }
        )
    if result_ids != set(manifest_by_id):
        raise TriangleScreeningError(
            "historical primary results do not map exactly to the manifest"
        )
    manifest_rows = [
        {
            key: value
            for key, value in manifest_by_id[candidate_id].items()
            if key != "family_id" and value is not None
        }
        for candidate_id in sorted(manifest_by_id)
    ]
    return candidate_rows, manifest_rows


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TriangleScreeningError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TriangleScreeningError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise TriangleScreeningError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TriangleScreeningError(f"{label} must be an integer")
    return value


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("sample_id")
    if not isinstance(value, str) or not value or "\0" in value:
        raise TriangleScreeningError(
            f"row {index} sample_id must be a non-empty string without NUL"
        )
    return value


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], *, required_fields: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TriangleScreeningError(f"row {index} must be an object")
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise TriangleScreeningError(
                f"row {index} is missing fields: {', '.join(missing)}"
            )
        sample_id = _sample_id(row, index)
        if sample_id in result:
            raise TriangleScreeningError(f"duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def select_eligible512(
    rows: Sequence[Mapping[str, Any]], *, expected_eligible_count: int | None = 2045
) -> list[dict[str, Any]]:
    """Build the locked nested 32/128/512 manifest from full-2048 evidence."""
    required = (
        "sample_id",
        "label",
        "native_sharpness",
        "source_detector",
        "native_detector",
        "source_face_count",
        "native_face_count",
    )
    indexed = _unique_rows(rows, required_fields=required)
    eligible: list[dict[str, Any]] = []
    for sample_id, row in indexed.items():
        label = _integer(row["label"], f"{sample_id}.label")
        if label not in range(8):
            raise TriangleScreeningError(f"{sample_id}.label must be in 0..7")
        sharpness = _finite_number(
            row["native_sharpness"], f"{sample_id}.native_sharpness"
        )
        source_count = _integer(
            row["source_face_count"], f"{sample_id}.source_face_count"
        )
        native_count = _integer(
            row["native_face_count"], f"{sample_id}.native_face_count"
        )
        if (
            row["source_detector"] == "buffalo_l"
            and row["native_detector"] == "buffalo_l"
            and source_count == 1
            and native_count == 1
        ):
            eligible.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "native_sharpness": sharpness,
                }
            )
    if expected_eligible_count is not None and len(eligible) != expected_eligible_count:
        raise TriangleScreeningError(
            f"eligible count must be {expected_eligible_count}, got {len(eligible)}"
        )

    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for label in range(8):
        label_rows = sorted(
            (row for row in eligible if row["label"] == label),
            key=lambda row: (row["native_sharpness"], row["sample_id"]),
        )
        if len(label_rows) < 64:
            raise TriangleScreeningError(
                f"label {label} has {len(label_rows)} eligible rows; at least 64 required"
            )
        for position, row in enumerate(label_rows):
            quartile = min(3, (position * 4) // len(label_rows))
            cells.setdefault((label, quartile), []).append(row)

    selected_cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for label in range(8):
        for quartile in range(4):
            cell = cells.get((label, quartile), [])
            if len(cell) < 16:
                raise TriangleScreeningError(
                    f"label {label} quartile {quartile} has fewer than 16 rows"
                )
            ranked = sorted(
                cell,
                key=lambda row: (
                    hashlib.sha256(
                        SELECTOR_DOMAIN + row["sample_id"].encode("utf-8")
                    ).digest(),
                    row["sample_id"],
                ),
            )
            selected_cells[(label, quartile)] = ranked[:16]

    manifest: list[dict[str, Any]] = []
    for rank in range(16):
        for label in range(8):
            for quartile in range(4):
                row = selected_cells[(label, quartile)][rank]
                manifest.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": label,
                        "sharpness_quartile": quartile,
                        "cell_rank": rank,
                    }
                )
    return manifest


def join_eligibility_evidence(
    full_evidence_rows: Sequence[Mapping[str, Any]],
    affectnet_label_rows: Sequence[Mapping[str, Any]],
    native_sharpness_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Strictly join detector evidence, AffectNet labels, and native sharpness."""
    full = _unique_rows(
        full_evidence_rows,
        required_fields=(
            "sample_id",
            "source_detector",
            "native_detector",
            "source_face_count",
            "native_face_count",
        ),
    )
    labels = _unique_rows(
        affectnet_label_rows, required_fields=("sample_id", "label")
    )
    sharpness = _unique_rows(
        native_sharpness_rows,
        required_fields=("sample_id", "native_sharpness"),
    )
    if set(full) != set(labels) or set(full) != set(sharpness):
        raise TriangleScreeningError(
            "eligibility evidence, AffectNet labels, and native sharpness "
            "must contain exactly the same sample IDs"
        )
    return [
        {
            **full[sample_id],
            "label": labels[sample_id]["label"],
            "native_sharpness": sharpness[sample_id]["native_sharpness"],
        }
        for sample_id in sorted(full)
    ]


def paired_bootstrap_upper(
    paired_deltas: Sequence[float],
    *,
    iterations: int,
    seed: int = BOOTSTRAP_SEED,
    indices: np.ndarray | None = None,
) -> float:
    values = np.asarray(
        [_finite_number(value, "paired delta") for value in paired_deltas],
        dtype=np.float64,
    )
    if values.ndim != 1 or values.size == 0:
        raise TriangleScreeningError("paired bootstrap requires non-empty 1D values")
    if iterations <= 0:
        raise TriangleScreeningError("bootstrap iterations must be positive")
    if indices is None:
        indices = np.random.Generator(np.random.PCG64(seed)).integers(
            0, values.size, size=(iterations, values.size), dtype=np.int64
        )
    if indices.shape != (iterations, values.size):
        raise TriangleScreeningError("shared bootstrap index shape mismatch")
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.95, method="higher"))


def _bootstrap_iterations(stage: int) -> int:
    if stage == 512:
        return 10_000
    if stage in (8, 32, 128):
        return 2_000
    raise TriangleScreeningError("stage must be one of 8, 32, 128, 512")


def _mean(values: Iterable[float]) -> float:
    array = np.fromiter(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise TriangleScreeningError("metric aggregation requires finite values")
    return float(array.mean())


def _evaluate_arm(
    arm_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    fid: float | None,
    kid: float | None,
    native_fid: float | None,
    native_kid: float | None,
    bootstrap_indices: np.ndarray,
    stage: int,
) -> ArmResult:
    if not isinstance(arm_id, str) or not arm_id:
        raise TriangleScreeningError("arm_id must be a non-empty string")
    indexed = _unique_rows(rows, required_fields=ROW_FIELDS)
    ordered = [indexed[sample_id] for sample_id in sorted(indexed)]
    numeric: dict[str, list[float]] = {}
    for field in ROW_FIELDS[1:9]:
        numeric[field] = [
            _finite_number(row[field], f"{arm_id}.{row['sample_id']}.{field}")
            for row in ordered
        ]
    counts: dict[str, list[int]] = {}
    for field in ROW_FIELDS[9:12]:
        counts[field] = [
            _integer(row[field], f"{arm_id}.{row['sample_id']}.{field}")
            for row in ordered
        ]
    n = len(ordered)
    source_exact_one = sum(count == 1 for count in counts["source_face_count"])
    native_exact_one = sum(count == 1 for count in counts["native_face_count"])
    exact_one = sum(count == 1 for count in counts["candidate_face_count"])
    e0 = _mean(numeric["candidate_e0"])
    delta_e0 = _mean(
        candidate - native
        for candidate, native in zip(
            numeric["candidate_e0"], numeric["native_e0"], strict=True
        )
    )
    delta_edev = _mean(
        candidate - native
        for candidate, native in zip(
            numeric["candidate_edev"], numeric["native_edev"], strict=True
        )
    )
    niqe = _mean(numeric["candidate_niqe"])
    native_niqe = _mean(numeric["native_niqe"])
    sharpness = _mean(numeric["candidate_sharpness"])
    native_sharpness = _mean(numeric["native_sharpness"])
    privacy_available = source_exact_one == native_exact_one == exact_one == n
    arcface_deltas: list[float] = []
    for row in ordered:
        pair_values: list[float | None] = []
        for field, pair_exact_one in (
            (
                "source_native_cosine",
                row["source_face_count"] == row["native_face_count"] == 1,
            ),
            (
                "source_candidate_cosine",
                row["source_face_count"] == row["candidate_face_count"] == 1,
            ),
        ):
            if pair_exact_one:
                if row[field] is None and not privacy_available:
                    pair_values.append(None)
                else:
                    pair_values.append(
                        _finite_number(
                            row[field], f"{arm_id}.{row['sample_id']}.{field}"
                        )
                    )
            else:
                if row[field] is not None:
                    raise TriangleScreeningError(
                        f"{arm_id}.{row['sample_id']}.{field} must be null "
                        "when its role pair is not exact-one"
                    )
                pair_values.append(None)
        if privacy_available:
            source_native, source_candidate = pair_values
            assert source_native is not None and source_candidate is not None
            arcface_deltas.append(
                source_candidate - source_native
            )
    arcface_delta = _mean(arcface_deltas) if privacy_available else None
    if stage == 32:
        if any(value is not None for value in (fid, kid, native_fid, native_kid)):
            raise TriangleScreeningError(
                "stage32 provisional screening forbids FID/KID values"
            )
        fid = kid = native_fid = native_kid = None
        arcface_delta_u95 = None
        privacy_statistic = arcface_delta
    else:
        fid = _finite_number(fid, f"{arm_id}.fid")
        kid = _finite_number(kid, f"{arm_id}.kid")
        native_fid = _finite_number(native_fid, "native_fid")
        native_kid = _finite_number(native_kid, "native_kid")
        arcface_delta_u95 = (
            paired_bootstrap_upper(
                arcface_deltas,
                iterations=bootstrap_indices.shape[0],
                indices=bootstrap_indices,
            )
            if privacy_available
            else None
        )
        privacy_statistic = arcface_delta_u95
    sharpness_floor = max(300.0, 0.95 * native_sharpness)
    checks = (
        ("source_exact_one", source_exact_one == n),
        ("native_exact_one", native_exact_one == n),
        ("candidate_exact_one", exact_one == n),
        ("arcface_privacy_available", privacy_statistic is not None),
        ("e0", e0 >= 0.75),
        ("delta_e0", delta_e0 >= 0.30),
        ("delta_edev", delta_edev >= 0.05),
        ("niqe", niqe <= native_niqe + 0.10),
        ("sharpness", sharpness >= sharpness_floor),
    )
    if privacy_statistic is not None:
        checks = checks[:4] + (
            (
                "arcface_delta" if stage == 32 else "arcface_delta_u95",
                privacy_statistic <= 0.02,
            ),
        ) + checks[4:]
    if stage != 32:
        checks = checks[:-1] + (
            ("fid", fid <= native_fid + 3.0),
            ("kid", kid <= native_kid + 0.005),
            checks[-1],
        )
    failed = tuple(name for name, passed in checks if not passed)
    r_margin = min(
        (e0 - 0.75) / 0.75,
        (delta_e0 - 0.30) / 0.30,
        (delta_edev - 0.05) / 0.05,
    )
    q_terms = [
        (native_niqe + 0.10 - niqe) / 0.10,
        sharpness / sharpness_floor - 1.0,
    ]
    if stage != 32:
        q_terms[:0] = [
            (native_fid + 3.0 - fid) / 3.0,
            (native_kid + 0.005 - kid) / 0.005,
        ]
    q_margin = min(q_terms)
    p_margin = (
        (0.02 - privacy_statistic) / 0.02
        if privacy_statistic is not None
        else None
    )
    return ArmResult(
        arm_id=arm_id,
        sample_count=n,
        candidate_exact_one_count=exact_one,
        e0=e0,
        delta_e0=delta_e0,
        delta_edev=delta_edev,
        niqe=niqe,
        native_niqe=native_niqe,
        fid=fid,
        native_fid=native_fid,
        kid=kid,
        native_kid=native_kid,
        sharpness=sharpness,
        native_sharpness=native_sharpness,
        arcface_delta=arcface_delta,
        arcface_delta_u95=arcface_delta_u95,
        hard_gate_pass=not failed,
        failed_gates=failed,
        r_margin=r_margin,
        q_margin=q_margin,
        p_margin=p_margin,
    )


def pareto_frontier(results: Sequence[ArmResult]) -> list[ArmResult]:
    survivors = [result for result in results if result.hard_gate_pass]
    frontier: list[ArmResult] = []
    for candidate in survivors:
        dominated = any(
            other.arm_id != candidate.arm_id
            and other.r_margin >= candidate.r_margin
            and other.q_margin >= candidate.q_margin
            and other.p_margin >= candidate.p_margin
            and (
                other.r_margin > candidate.r_margin
                or other.q_margin > candidate.q_margin
                or other.p_margin > candidate.p_margin
            )
            for other in survivors
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda result: result.arm_id)


def _axis_best(
    results: Sequence[ArmResult], axis: str, *, final_tie: bool = False
) -> ArmResult:
    if axis == "P":
        key = lambda result: (
            -result.p_margin,
            -result.r_margin if final_tie else 0.0,
            result.arm_id,
        )
    elif axis == "Q":
        key = lambda result: (
            -result.q_margin,
            -result.r_margin if final_tie else 0.0,
            result.arm_id,
        )
    elif axis == "R":
        key = lambda result: (-result.r_margin, result.arm_id)
    else:
        raise TriangleScreeningError(f"unknown axis: {axis}")
    return min(results, key=key)


def apply_stage_cap(frontier: Sequence[ArmResult], stage: int) -> list[ArmResult]:
    if stage == 8:
        return list(frontier)
    cap = STAGE_CAPS.get(stage)
    if cap is None:
        raise TriangleScreeningError("stage must be one of 8, 32, 128, 512")
    if len(frontier) <= cap:
        return list(frontier)
    if stage == 512:
        picks = [
            _axis_best(frontier, "P", final_tie=True),
            _axis_best(frontier, "Q", final_tie=True),
        ]
    else:
        picks = [_axis_best(frontier, axis) for axis in ("P", "Q", "R")]
        remaining = sorted(
            frontier,
            key=lambda result: (
                -min(result.r_margin, result.q_margin, result.p_margin),
                result.arm_id,
            ),
        )
        picks.extend(remaining)
    selected: list[ArmResult] = []
    for result in picks:
        if result.arm_id not in {item.arm_id for item in selected}:
            selected.append(result)
        if len(selected) == cap:
            break
    return selected


def _breakthrough_status(result: ArmResult, baseline: ArmResult) -> str:
    if not result.hard_gate_pass:
        return "no_gate_survivor"
    if result.p_margin is None or baseline.p_margin is None:
        return "gate_survivor_no_breakthrough"
    axes = (result.r_margin, result.q_margin, result.p_margin)
    baseline_axes = (baseline.r_margin, baseline.q_margin, baseline.p_margin)
    strict_improvements = sum(
        value > baseline_value
        for value, baseline_value in zip(axes, baseline_axes, strict=True)
    )
    no_worse = all(
        value >= baseline_value
        for value, baseline_value in zip(axes, baseline_axes, strict=True)
    )
    if strict_improvements >= 2 and no_worse:
        privacy_statistic = (
            result.arcface_delta
            if result.arcface_delta_u95 is None
            else result.arcface_delta_u95
        )
        if privacy_statistic <= 0.0:
            return "privacy_positive_breakthrough"
        return "triangle_breakthrough"
    return "gate_survivor_no_breakthrough"


def evaluate_arms(
    arm_inputs: Sequence[Mapping[str, Any]],
    *,
    stage: int,
    native_fid: float,
    native_kid: float,
    baseline_arm_id: str = "paper_eta_0p125",
    expected_sample_ids: Sequence[str] | None = None,
) -> list[ArmResult]:
    iterations = _bootstrap_iterations(stage)
    if not arm_inputs:
        raise TriangleScreeningError("at least one arm is required")
    sample_counts = {len(arm.get("rows", [])) for arm in arm_inputs}
    if sample_counts != {stage}:
        raise TriangleScreeningError(
            f"every arm must have exactly {stage} per-sample rows"
        )
    sample_ids: tuple[str, ...] | None = None
    normalized: list[tuple[str, list[Mapping[str, Any]], float, float]] = []
    arm_ids: set[str] = set()
    for arm in arm_inputs:
        arm_id = arm.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise TriangleScreeningError("each arm requires a non-empty arm_id")
        if arm_id in arm_ids:
            raise TriangleScreeningError(f"duplicate arm_id: {arm_id}")
        arm_ids.add(arm_id)
        rows = arm.get("rows")
        if not isinstance(rows, list):
            raise TriangleScreeningError(f"{arm_id}.rows must be a list")
        ids = tuple(sorted(_unique_rows(rows, required_fields=ROW_FIELDS)))
        if sample_ids is None:
            sample_ids = ids
        elif ids != sample_ids:
            raise TriangleScreeningError("all arms must use the same sample manifest")
        normalized.append((arm_id, rows, arm.get("fid"), arm.get("kid")))
    if expected_sample_ids is not None:
        expected = tuple(sorted(expected_sample_ids))
        if len(expected) != stage or len(set(expected)) != stage:
            raise TriangleScreeningError(
                "expected sample manifest must contain exactly the stage count"
            )
        if sample_ids != expected:
            raise TriangleScreeningError(
                "arm rows do not match the locked eligible512 stage prefix"
            )
    if baseline_arm_id not in arm_ids:
        raise TriangleScreeningError(
            f"baseline arm {baseline_arm_id!r} is missing"
        )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    shared_indices = rng.integers(
        0, stage, size=(iterations, stage), dtype=np.int64
    )
    evaluated = [
        _evaluate_arm(
            arm_id,
            rows,
            fid=fid,
            kid=kid,
            native_fid=native_fid,
            native_kid=native_kid,
            bootstrap_indices=shared_indices,
            stage=stage,
        )
        for arm_id, rows, fid, kid in normalized
    ]
    baseline = next(
        result for result in evaluated if result.arm_id == baseline_arm_id
    )
    frontier_ids = {result.arm_id for result in pareto_frontier(evaluated)}
    selected_ids = {
        result.arm_id
        for result in apply_stage_cap(
            [result for result in evaluated if result.arm_id in frontier_ids],
            stage,
        )
    }
    return [
        ArmResult(
            **{
                **result.__dict__,
                "pareto": result.arm_id in frontier_ids,
                "selected": result.arm_id in selected_ids,
                "status": _breakthrough_status(result, baseline),
            }
        )
        for result in evaluated
    ]


def write_outputs(
    output_dir: Path,
    results: Sequence[ArmResult],
    *,
    stage: int,
    baseline_arm_id: str,
    selection_manifest: Sequence[Mapping[str, Any]] | None = None,
    selection_metadata: Mapping[str, Any] | None = None,
) -> None:
    output_dir = output_dir.resolve()
    targets = (
        output_dir / "summary.json",
        output_dir / "arms.csv",
        output_dir / "arcface_failures.json",
        output_dir / "conclusion.md",
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TriangleScreeningError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in targets):
        raise TriangleScreeningError("refusing to overwrite triangle outputs")
    rows = [result.as_dict() for result in results]
    selected = [result.arm_id for result in results if result.selected]
    summary = {
        "schema_version": 1,
        "contract_type": "safa_triangle_screening_v1",
        "stage": stage,
        "bootstrap": (
            None
            if stage == 32
            else {
                "bit_generator": "PCG64",
                "seed": BOOTSTRAP_SEED,
                "iterations": _bootstrap_iterations(stage),
                "paired": True,
                "shared_across_arms": True,
            }
        ),
        "privacy_inference": (
            {
                "statistic": "point_arcface_delta",
                "provisional": True,
                "bootstrap": None,
            }
            if stage == 32
            else {
                "statistic": "arcface_delta_u95",
                "provisional": False,
                "bootstrap": {
                    "bit_generator": "PCG64",
                    "seed": BOOTSTRAP_SEED,
                    "iterations": _bootstrap_iterations(stage),
                    "paired": True,
                    "shared_across_arms": True,
                },
            }
        ),
        "baseline_arm_id": baseline_arm_id,
        "arm_count": len(results),
        "gate_survivor_count": sum(result.hard_gate_pass for result in results),
        "pareto_arm_ids": [
            result.arm_id for result in results if result.pareto
        ],
        "selected_arm_ids": selected,
        "arms": rows,
    }
    if selection_manifest is not None:
        if selection_metadata is None:
            summary["selection"] = {
                "selector": "safa-triangle-512-v1",
                "eligible_count": 2045,
                "prefix_count": len(selection_manifest),
                "samples": list(selection_manifest),
            }
        else:
            summary["selection"] = {
                **dict(selection_metadata),
                "sample_count": len(selection_manifest),
                "samples": list(selection_manifest),
            }
    with targets[0].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with targets[1].open("x", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else ["arm_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "failed_gates": ";".join(row["failed_gates"]),
                }
            )
    failures = {
        "schema_version": 1,
        "failures": [
            {
                "arm_id": result.arm_id,
                "sample_count": result.sample_count,
                "candidate_exact_one_count": result.candidate_exact_one_count,
                "arcface_delta_u95": result.arcface_delta_u95,
                "arcface_delta": result.arcface_delta,
                "failed_gates": list(result.failed_gates),
            }
            for result in results
            if (
                result.candidate_exact_one_count != result.sample_count
                or result.p_margin is None
                or (
                    result.arcface_delta_u95 is not None
                    and result.arcface_delta_u95 > 0.02
                )
                or (
                    result.arcface_delta_u95 is None
                    and result.arcface_delta is not None
                    and result.arcface_delta > 0.02
                )
            )
        ],
    }
    with targets[2].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(failures, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Triangle screening conclusion",
        "",
        f"- Stage: {stage}",
        f"- Gate survivors: {summary['gate_survivor_count']}/{len(results)}",
        f"- Selected arms: {', '.join(selected) if selected else 'none'}",
        "",
        "## Arm statuses",
        "",
    ]
    lines.extend(f"- `{result.arm_id}`: {result.status}" for result in results)
    with targets[3].open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
