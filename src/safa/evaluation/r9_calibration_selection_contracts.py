from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from safa.evaluation.r9_campaign_contracts import (
    privacy_delta_cluster_bootstrap,
    validate_gate_contract,
)


SOURCE_CAMPAIGN_ID = "r9-report-only-formal-v6"
CHILD_CAMPAIGN_ID = "r9-report-only-formal-v7"
SOURCE_PHASE = "calibrate"
SOURCE_ROOT = Path(
    "artifacts/r9_meanflow_flow_map_guidance/campaigns"
) / SOURCE_CAMPAIGN_ID
SOURCE_GATE_SHA256 = (
    "84c4aa802965601bfeccc03fa0e9da2baef25d8cc98cb9dbbc536058037520b9"
)
SOURCE_PHASE_RESULTS_SHA256 = (
    "2be463aaadc7b5cf9f4cfd87b452034bdcecd3bf65d13ecb3bebb4b68844a35c"
)
SOURCE_AUTOMATIC_EVIDENCE_SHA256 = (
    "c9840ff3a4c96b64db386e64a543e2c637b69ec0f9cd2453913070267ecaffbe"
)
SOURCE_REPAIR_SHA256 = (
    "716355ccf9171d3b6d35f51c124139e110b99986393ed7e2b397c02d7c0fb355"
)
SOURCE_GENERATION_INVENTORY_SHA256 = (
    "e40516b8dc852c6b6930e38b89b656b28274c91f40003884ab688d8768ab145a"
)
SOURCE_RUNTIME_SHA256 = (
    "9a529a086b79522b50a40fb73586fa44151c191f11cc11d19707f33bee9aeeb9"
)
MISS_SEED = 2027
MISS_SAMPLE_ID = (
    "val:Manually_Annotated_Images/1003/"
    "5a46f394c9709f851bdb273c33f8ef136fe8c1c384b0975b8047c47b.jpg"
)
SEEDS = (1337, 2027, 3407)
SELECTED_ARM_IDS = (
    "paper_eta_0p125",
    "flow_map2_normalized_eta_0p125",
)
EXPECTED_BOOTSTRAP_SHA256 = {
    "paper_eta_0p125": (
        "ccd31757505ccd2c1249528276ec50466d8663b9ad9a235a5bbfd8835f9adfe0"
    ),
    "flow_map2_normalized_eta_0p125": (
        "b51a43bab079f770d1f99d7ce8bc3be2b633a635242104cd74cfbf5f75d331e3"
    ),
}


class CalibrationSelectionContractError(ValueError):
    """Raised when frozen Phase-B evidence cannot support the v7 transition."""


def build_calibration_report_only_selection_contract(
    *, repo_root: Path, child_campaign_id: str = CHILD_CAMPAIGN_ID
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if child_campaign_id != CHILD_CAMPAIGN_ID:
        raise CalibrationSelectionContractError("selection child campaign must be v7")
    phase_root = root / SOURCE_ROOT / SOURCE_PHASE
    runtime_path = root / SOURCE_ROOT / "campaign_runtime.json"
    gate_path = phase_root / "gate_contract.json"
    phase_path = phase_root / "phase_results.json"
    automatic_path = phase_root / "automatic_evidence.json"
    repair_path = phase_root / "evaluation_repair_contract_v3.json"

    runtime = _read_contract(runtime_path, "campaign_runtime_sha256")
    gate = validate_gate_contract(_read_json(gate_path))
    phase = _read_contract(phase_path, "phase_results_sha256")
    automatic = _read_contract(automatic_path, "automatic_evidence_sha256")
    repair = _read_contract(repair_path, "repair_contract_sha256")
    _validate_source_chain(runtime, gate, phase, automatic, repair)

    automatic_by_arm = _unique_arms(automatic.get("arms"), "automatic evidence")
    phase_by_arm = _unique_arms(phase.get("arms"), "phase results")
    gate_by_arm = _unique_arms(gate.get("arms"), "gate")
    if set(automatic_by_arm) != set(phase_by_arm) or set(phase_by_arm) != set(
        gate_by_arm
    ):
        raise CalibrationSelectionContractError("Phase-B arm sets disagree")

    raw_bindings: list[dict[str, Any]] = []
    bootstraps: dict[str, dict[str, Any]] = {}
    evaluated_arms = []
    for arm_id in sorted(automatic_by_arm):
        automatic_arm = automatic_by_arm[arm_id]
        phase_arm = phase_by_arm[arm_id]
        gate_arm = gate_by_arm[arm_id]
        _validate_arm_identity(automatic_arm, phase_arm, gate_arm)
        rows, bindings = _complete_case_rows(
            root=root,
            arm_id=arm_id,
            automatic_arm=automatic_arm,
        )
        raw_bindings.extend(bindings)
        bootstrap = privacy_delta_cluster_bootstrap(
            rows,
            expected_seeds=SEEDS,
            bootstrap_seed=91637,
        )
        expected_bootstrap = EXPECTED_BOOTSTRAP_SHA256.get(arm_id)
        if expected_bootstrap is not None and bootstrap["bootstrap_sha256"] != expected_bootstrap:
            raise CalibrationSelectionContractError(
                f"{arm_id} complete-case bootstrap changed"
            )
        bootstraps[arm_id] = bootstrap
        failures = gate_arm.get("failures")
        if failures != ["seed_2027:arcface_not_exactly_one_face_per_image"]:
            raise CalibrationSelectionContractError(
                f"{arm_id} has a non-coverage Phase-B failure"
            )
        evaluated_arms.append(
            {
                "arm_id": arm_id,
                "family": str(phase_arm["family"]),
                "config_sha256": _sha(
                    phase_arm.get("config_sha256"), f"{arm_id} config SHA256"
                ),
                "output_sha256": _sha(
                    phase_arm.get("output_sha256"), f"{arm_id} output SHA256"
                ),
                "evaluator_evidence_sha256": _sha(
                    phase_arm.get("evaluator_evidence_sha256"),
                    f"{arm_id} evaluator evidence SHA256",
                ),
                "severe_count": sum(
                    _nonnegative_int(row.get("severe_count"), "severe count")
                    for row in gate_arm.get("seed_results", ())
                ),
                "complete_case_privacy": bootstrap,
                "selected": arm_id in SELECTED_ARM_IDS,
            }
        )

    selected_arms = []
    evaluated_by_id = {row["arm_id"]: row for row in evaluated_arms}
    expected_families = ("paper_split_constant", "flow_map2")
    for arm_id, family in zip(SELECTED_ARM_IDS, expected_families, strict=True):
        row = evaluated_by_id.get(arm_id)
        if row is None or row["family"] != family:
            raise CalibrationSelectionContractError(
                "fixed dual-mainline selection is unavailable"
            )
        selected_arms.append(
            {
                key: row[key]
                for key in (
                    "arm_id",
                    "family",
                    "config_sha256",
                    "output_sha256",
                    "evaluator_evidence_sha256",
                )
            }
        )

    generation = _mapping(repair.get("generation_evidence"), "generation evidence")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_calibration_report_only_selection_v1",
        "child_campaign_id": CHILD_CAMPAIGN_ID,
        "source": {
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "phase": SOURCE_PHASE,
            "gate_contract_sha256": SOURCE_GATE_SHA256,
            "phase_results_sha256": SOURCE_PHASE_RESULTS_SHA256,
            "automatic_evidence_sha256": SOURCE_AUTOMATIC_EVIDENCE_SHA256,
            "evaluation_repair_contract_sha256": SOURCE_REPAIR_SHA256,
            "generation_inventory_sha256": SOURCE_GENERATION_INVENTORY_SHA256,
        },
        "evidence": {
            "campaign_runtime": _binding(
                root, runtime_path, runtime["campaign_runtime_sha256"]
            ),
            "calibrate_gate": _binding(
                root, gate_path, gate["gate_contract_sha256"]
            ),
            "phase_results": _binding(
                root, phase_path, phase["phase_results_sha256"]
            ),
            "automatic_evidence": _binding(
                root, automatic_path, automatic["automatic_evidence_sha256"]
            ),
            "evaluation_repair": _binding(
                root, repair_path, repair["repair_contract_sha256"]
            ),
            "generation_inventory": {
                key: generation[key]
                for key in (
                    "inventory_sha256",
                    "logical_run_count",
                    "shard_count",
                    "completion_count",
                    "generation_result_count",
                    "file_count",
                    "png_count",
                )
            },
            "arcface_raw_evidence": raw_bindings,
        },
        "policy": {
            "scope": "promotion_decision_only",
            "original_gate_mutation": False,
            "numerical_metrics_role": "report_only",
            "visual_metrics_role": "observation_only",
            "arcface_coverage_role": "report_only",
            "privacy_metrics_role": "report_only_complete_case",
            "complete_case_exclusion": "sample_id_across_all_seeds",
            "eligible_families": ["paper_split_constant", "flow_map2"],
            "family_order": ["paper_split_constant", "flow_map2"],
            "max_selected": 2,
            "noncoverage_failures_allowed": False,
        },
        "coverage_report": {
            "seeds": list(SEEDS),
            "miss_seed": MISS_SEED,
            "excluded_sample_ids": [MISS_SAMPLE_ID],
            "miss_face_counts": {"source": 1, "native": 2, "candidate": 1},
            "sample_count": 63,
            "observation_count": 189,
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 91637,
            "complete_case_bootstraps": bootstraps,
        },
        "evaluated_arms": evaluated_arms,
        "selected_arms": selected_arms,
        "supersedes": {
            "scope": "promotion_decision_only",
            "original_verdict": "stop_zero_candidates",
            "original_selected_arm_ids": [],
            "original_gate_contract_sha256": SOURCE_GATE_SHA256,
        },
        "verdict": "continue_to_confirm512",
        "reselection_allowed": False,
    }
    payload["calibration_selection_sha256"] = _canonical_digest(
        payload, "calibration_selection_sha256"
    )
    return payload


def validate_calibration_report_only_selection_contract(
    value: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    normalized = _mapping(value, "calibration report-only selection")
    declared = _sha(
        normalized.get("calibration_selection_sha256"),
        "calibration selection SHA256",
    )
    if declared != _canonical_digest(normalized, "calibration_selection_sha256"):
        raise CalibrationSelectionContractError("selection canonical digest mismatch")
    expected = build_calibration_report_only_selection_contract(repo_root=repo_root)
    if normalized != expected:
        raise CalibrationSelectionContractError(
            "selection disagrees with frozen Phase-B evidence"
        )
    return normalized


def calibration_selection_contract_binding(
    payload: Mapping[str, Any], *, repo_root: Path
) -> tuple[Path, bytes, dict[str, str]]:
    normalized = _mapping(payload, "calibration report-only selection")
    declared = _sha(
        normalized.get("calibration_selection_sha256"),
        "calibration selection SHA256",
    )
    if declared != _canonical_digest(normalized, "calibration_selection_sha256"):
        raise CalibrationSelectionContractError("selection canonical digest mismatch")
    root = Path(repo_root).resolve()
    path = root / (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        f"{CHILD_CAMPAIGN_ID}/calibration_report_only_selection.json"
    )
    content = _contract_bytes(normalized)
    return path, content, {
        "path": str(path.relative_to(root)),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "contract_sha256": declared,
    }


def materialize_calibration_report_only_selection_contract(
    *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = build_calibration_report_only_selection_contract(repo_root=repo_root)
    path, content, binding = calibration_selection_contract_binding(
        payload, repo_root=repo_root
    )
    _write_exclusive(path, content)
    return payload, binding


def _validate_source_chain(
    runtime: Mapping[str, Any],
    gate: Mapping[str, Any],
    phase: Mapping[str, Any],
    automatic: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> None:
    if runtime.get("campaign_id") != SOURCE_CAMPAIGN_ID or runtime.get(
        "campaign_runtime_sha256"
    ) != SOURCE_RUNTIME_SHA256:
        raise CalibrationSelectionContractError("source runtime changed")
    if (
        gate.get("phase") != SOURCE_PHASE
        or gate.get("gate_contract_sha256") != SOURCE_GATE_SHA256
        or gate.get("verdict") != "stop_zero_candidates"
        or gate.get("selected_arm_ids") != []
    ):
        raise CalibrationSelectionContractError("source gate identity changed")
    if (
        phase.get("phase") != SOURCE_PHASE
        or phase.get("phase_results_sha256") != SOURCE_PHASE_RESULTS_SHA256
        or automatic.get("phase") != SOURCE_PHASE
        or automatic.get("automatic_evidence_sha256")
        != SOURCE_AUTOMATIC_EVIDENCE_SHA256
        or repair.get("phase") != SOURCE_PHASE
        or repair.get("repair_contract_sha256") != SOURCE_REPAIR_SHA256
    ):
        raise CalibrationSelectionContractError("source evidence identity changed")
    context = _mapping(gate.get("context"), "gate context")
    for actual, expected, label in (
        (context.get("campaign_runtime_sha256"), SOURCE_RUNTIME_SHA256, "runtime"),
        (
            context.get("phase_results_sha256"),
            SOURCE_PHASE_RESULTS_SHA256,
            "phase results",
        ),
        (
            context.get("automatic_evidence_sha256"),
            SOURCE_AUTOMATIC_EVIDENCE_SHA256,
            "automatic evidence",
        ),
        (phase.get("automatic_evidence_sha256"), SOURCE_AUTOMATIC_EVIDENCE_SHA256, "phase automatic evidence"),
        (automatic.get("run_plan_sha256"), phase.get("run_plan_sha256"), "run plan"),
    ):
        if actual != expected:
            raise CalibrationSelectionContractError(f"source {label} binding changed")
    generation = _mapping(repair.get("generation_evidence"), "generation evidence")
    expected_generation = {
        "inventory_sha256": SOURCE_GENERATION_INVENTORY_SHA256,
        "logical_run_count": 12,
        "shard_count": 12,
        "completion_count": 12,
        "generation_result_count": 12,
        "file_count": 1440,
        "png_count": 1344,
    }
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise CalibrationSelectionContractError("generation inventory changed")


def _validate_arm_identity(
    automatic: Mapping[str, Any],
    phase: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    arm_id = str(phase.get("arm_id", ""))
    shared = ("arm_id", "family", "config_sha256")
    if any(automatic.get(key) != phase.get(key) for key in shared) or any(
        phase.get(key) != gate.get(key) for key in shared
    ):
        raise CalibrationSelectionContractError(f"{arm_id} identity changed")
    if phase.get("output_sha256") != gate.get("output_sha256"):
        raise CalibrationSelectionContractError(f"{arm_id} output changed")


def _complete_case_rows(
    *, root: Path, arm_id: str, automatic_arm: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_results = automatic_arm.get("seed_results")
    if not isinstance(seed_results, list) or [row.get("seed") for row in seed_results] != list(SEEDS):
        raise CalibrationSelectionContractError(f"{arm_id} seed evidence changed")
    complete_rows = []
    bindings = []
    for seed_row in seed_results:
        seed = int(seed_row["seed"])
        summary = _mapping(seed_row.get("arcface_summary"), "ArcFace summary")
        expected_summary = (
            {
                "source_exact_one_count": 64,
                "native_exact_one_count": 63,
                "candidate_exact_one_count": 64,
                "paired_exact_one_count": 63,
                "failure_sample_ids": [MISS_SAMPLE_ID],
            }
            if seed == MISS_SEED
            else {
                "source_exact_one_count": 64,
                "native_exact_one_count": 64,
                "candidate_exact_one_count": 64,
                "paired_exact_one_count": 64,
                "failure_sample_ids": [],
            }
        )
        if summary != expected_summary:
            raise CalibrationSelectionContractError(
                f"{arm_id} seed {seed} coverage changed"
            )
        raw_path = Path(str(seed_row.get("arcface_raw_evidence_path", ""))).resolve()
        if not raw_path.is_file() or raw_path.is_symlink() or root not in raw_path.parents:
            raise CalibrationSelectionContractError("ArcFace raw path escaped repo")
        raw_contract_sha = _sha(
            seed_row.get("arcface_raw_evidence_sha256"), "ArcFace raw SHA256"
        )
        raw = _read_json(raw_path)
        if (
            raw.get("arcface_raw_evidence_sha256") != raw_contract_sha
            or _canonical_digest(raw, "arcface_raw_evidence_sha256")
            != raw_contract_sha
        ):
            raise CalibrationSelectionContractError("ArcFace raw evidence changed")
        arcface = _mapping(raw.get("arcface"), "raw ArcFace evidence")
        rows = arcface.get("rows")
        if not isinstance(rows, list) or len(rows) != 64:
            raise CalibrationSelectionContractError("ArcFace raw rows changed")
        seen = set()
        for row in rows:
            normalized = _mapping(row, "ArcFace row")
            sample_id = str(normalized.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                raise CalibrationSelectionContractError("ArcFace row IDs changed")
            seen.add(sample_id)
            expected_native = 2 if seed == MISS_SEED and sample_id == MISS_SAMPLE_ID else 1
            if (
                normalized.get("source_face_count") != 1
                or normalized.get("candidate_face_count") != 1
                or normalized.get("native_face_count") != expected_native
            ):
                raise CalibrationSelectionContractError("ArcFace face counts changed")
            if sample_id == MISS_SAMPLE_ID:
                continue
            complete_rows.append(
                {
                    "sample_id": sample_id,
                    "seed": seed,
                    "source_candidate_cosine": normalized.get(
                        "source_candidate_cosine"
                    ),
                    "source_native_cosine": normalized.get("source_native_cosine"),
                }
            )
        bindings.append(
            {
                "arm_id": arm_id,
                "seed": seed,
                "path": str(raw_path.relative_to(root)),
                "file_sha256": _file_sha256(raw_path),
                "contract_sha256": raw_contract_sha,
                "arcface_evidence_sha256": _sha(
                    seed_row.get("arcface_evidence_sha256"),
                    "ArcFace evidence SHA256",
                ),
            }
        )
    if len(complete_rows) != 189:
        raise CalibrationSelectionContractError("complete-case row count changed")
    return complete_rows, bindings


def _unique_arms(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 3:
        raise CalibrationSelectionContractError(f"{label} must contain three arms")
    result = {}
    for row in value:
        normalized = _mapping(row, f"{label} arm")
        arm_id = str(normalized.get("arm_id", ""))
        if not arm_id or arm_id in result:
            raise CalibrationSelectionContractError(f"{label} arm IDs changed")
        result[arm_id] = normalized
    return result


def _binding(root: Path, path: Path, contract_sha256: str) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "file_sha256": _file_sha256(path),
        "contract_sha256": _sha(contract_sha256, "contract SHA256"),
    }


def _read_contract(path: Path, field: str) -> dict[str, Any]:
    payload = _read_json(path)
    declared = _sha(payload.get(field), field)
    if declared != _canonical_digest(payload, field):
        raise CalibrationSelectionContractError(f"{field} canonical digest mismatch")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CalibrationSelectionContractError(f"missing immutable evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationSelectionContractError(f"invalid JSON evidence: {path}") from error
    return _mapping(value, str(path))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _contract_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise CalibrationSelectionContractError("selection contract already differs")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
        os.link(temporary, path)
    except FileExistsError as error:
        raise CalibrationSelectionContractError("selection contract creation raced") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationSelectionContractError(f"{label} must be a mapping")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise CalibrationSelectionContractError(f"{label} must be lowercase SHA256")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationSelectionContractError(f"{label} must be nonnegative int")
    return value
