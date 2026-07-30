from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from safa.evaluation.triangle_screening import TriangleScreeningError


LEGACY_ARM_IDS = (
    "eta0p125_baseline",
    "eta0p125_disable_i1",
    "eta0p125_disable_i2",
    "eta0p125_disable_i3",
)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Triangle32ArmSet:
    contract_type: str
    manifest_path: Path | None
    arm_ids: tuple[str, ...]
    selection_manifest: Path | None
    selection_manifest_sha256: str | None
    seed: int | None
    result_filename: str
    baseline_arm_id: str | None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_line_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path, label: str) -> Mapping[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise TriangleScreeningError(f"{label} is missing: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TriangleScreeningError(f"{label} is invalid JSON: {resolved}") from exc
    if not isinstance(value, Mapping):
        raise TriangleScreeningError(f"{label} must be an object")
    return value


def _bound_file(binding: Any, label: str) -> Path:
    if not isinstance(binding, Mapping) or set(binding) < {"path", "sha256"}:
        raise TriangleScreeningError(f"{label} binding is invalid")
    path = Path(str(binding["path"])).resolve()
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise TriangleScreeningError(f"{label} binding differs: {path}")
    return path


def _arm_ids(values: Sequence[Any], label: str) -> tuple[str, ...]:
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise TriangleScreeningError(f"{label} arm {index} has an unsafe ID")
        if value in result:
            raise TriangleScreeningError(f"{label} duplicate arm ID: {value}")
        result.append(value)
    if not result:
        raise TriangleScreeningError(f"{label} must contain at least one arm")
    return tuple(result)


def load_arm_set(path: Path | None) -> Triangle32ArmSet:
    if path is None:
        return Triangle32ArmSet(
            contract_type="safa_r10_triangle_fixed32_legacy_v1",
            manifest_path=None,
            arm_ids=LEGACY_ARM_IDS,
            selection_manifest=None,
            selection_manifest_sha256=None,
            seed=None,
            result_filename="generation_result.json",
            baseline_arm_id=LEGACY_ARM_IDS[0],
        )
    manifest_path = path.resolve()
    value = read_json(manifest_path, "triangle32 arm-set manifest")
    contract_type = value.get("contract_type")
    if contract_type == "safa_r10_triangle_fixed32_diagnostic_preparation_v1":
        if value.get("registration") != "new_diagnostic_not_r9_continuation":
            raise TriangleScreeningError("fixed32 diagnostic registration differs")
        arms = value.get("arms")
        if not isinstance(arms, list):
            raise TriangleScreeningError("fixed32 diagnostic arms are missing")
        arm_ids = _arm_ids(
            [
                arm.get("arm_id") if isinstance(arm, Mapping) else None
                for arm in arms
            ],
            "fixed32 diagnostic",
        )
        if arm_ids != LEGACY_ARM_IDS:
            raise TriangleScreeningError("fixed32 diagnostic arm IDs/order disagree")
        shared = value.get("shared")
        if not isinstance(shared, Mapping):
            raise TriangleScreeningError("fixed32 diagnostic shared binding is missing")
        selection = Path(str(shared.get("sample_id_manifest", ""))).resolve()
        selection_sha = shared.get("sample_id_manifest_sha256")
        if not selection.is_file() or sha256_file(selection) != selection_sha:
            raise TriangleScreeningError("fixed32 diagnostic selection binding differs")
        seed = shared.get("sampling_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TriangleScreeningError("fixed32 diagnostic seed is invalid")
        return Triangle32ArmSet(
            contract_type=str(contract_type),
            manifest_path=manifest_path,
            arm_ids=arm_ids,
            selection_manifest=selection,
            selection_manifest_sha256=str(selection_sha),
            seed=seed,
            result_filename="generation_result.json",
            baseline_arm_id=LEGACY_ARM_IDS[0],
        )
    if contract_type != "safa_r10_triangle32_pilot_preparation_v1":
        raise TriangleScreeningError("unknown triangle32 arm-set contract")
    if value.get("candidate_count") != 24 or value.get("generation_only") is not True:
        raise TriangleScreeningError("triangle32 pilot preparation fields differ")
    selected_path = _bound_file(value.get("selected24"), "selected24")
    selected = read_json(selected_path, "selected24")
    selected_rows = selected.get("selected")
    if (
        selected.get("contract_type") != "safa_triangle_historical24_v1"
        or selected.get("selected_count") != 24
        or not isinstance(selected_rows, list)
        or [
            row.get("selection_rank") for row in selected_rows
            if isinstance(row, Mapping)
        ] != list(range(1, 25))
    ):
        raise TriangleScreeningError("selected24 contract differs")
    arm_ids = _arm_ids(
        [
            row.get("candidate_id") if isinstance(row, Mapping) else None
            for row in selected_rows
        ],
        "triangle32 pilot",
    )
    sample_binding = value.get("sample_manifest")
    selection = _bound_file(sample_binding, "triangle32 sample manifest")
    if not isinstance(sample_binding, Mapping) or sample_binding.get("sample_count") != 32:
        raise TriangleScreeningError("triangle32 sample manifest count differs")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TriangleScreeningError("triangle32 pilot seed is invalid")
    return Triangle32ArmSet(
        contract_type=str(contract_type),
        manifest_path=manifest_path,
        arm_ids=arm_ids,
        selection_manifest=selection,
        selection_manifest_sha256=str(sample_binding["sha256"]),
        seed=seed,
        result_filename="result.json",
        baseline_arm_id=None,
    )


def validate_generation_result(
    arm_set: Triangle32ArmSet, runs_root: Path, arm_id: str
) -> Path:
    run_root = runs_root.resolve() / arm_id
    per_sample = run_root / "per_sample.jsonl"
    result_path = run_root / arm_set.result_filename
    result = read_json(result_path, f"{arm_id} generation result")
    if arm_set.result_filename == "generation_result.json":
        if result.get("status") != "complete" or result.get("sample_count") != 32:
            raise TriangleScreeningError(f"{arm_id} generation is not complete")
    else:
        binding = result.get("per_sample")
        if (
            result.get("contract_type") != "safa_r10_triangle32_worker_result_v1"
            or result.get("candidate_id") != arm_id
            or result.get("status") != "completed"
            or result.get("failure") is not None
            or result.get("generation_only") is not True
            or result.get("evaluation_status") != "not_started"
            or result.get("sample_count") != 32
            or result.get("worker_result_sha256")
            != canonical_line_digest(result, "worker_result_sha256")
            or not isinstance(binding, Mapping)
            or Path(str(binding.get("path", ""))).resolve() != per_sample.resolve()
            or not per_sample.is_file()
            or binding.get("sha256") != sha256_file(per_sample)
        ):
            raise TriangleScreeningError(
                f"{arm_id} triangle32 generation result binding differs"
            )
    if not per_sample.is_file():
        raise TriangleScreeningError(f"{arm_id} per-sample evidence is missing")
    return per_sample
