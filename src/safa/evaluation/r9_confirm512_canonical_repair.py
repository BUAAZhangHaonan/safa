"""Strict zero-generation canonical-native repair for the R9 v8 C campaign."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from safa.evaluation.r9_campaign_contracts import (
    paired_metric_cluster_bootstrap,
    privacy_delta_cluster_bootstrap,
    write_immutable_contract,
)
from safa.evaluation.r9_calibration_selection_contracts import (
    validate_calibration_report_only_selection_contract,
)
from safa.evaluation.r9_phase_results import (
    PhaseResultsRequest,
    _build_paired_metric_rows_contract,
    _canonical_json_sha256,
    _evaluate_arcface,
    _evaluate_quality,
    _load_run_evidence,
    _mean_metric,
    _paired_metric_rows,
    _sample_evidence,
    _validate_request,
)
from safa.utils.sampling import make_x_init_for_sample_ids, stable_sample_seed


SOURCE_CAMPAIGN_ID = "r9-report-only-formal-v8"
SOURCE_FAILURE_MESSAGE = (
    "safa.evaluation.r9_phase_results.PhaseResultsError: "
    "candidate matched-native PNG differs from native run"
)
EXPECTED_ARMS = (
    "native",
    "flow_map2_normalized_eta_0p125",
    "paper_eta_0p125",
)
EXPECTED_ROOT_COUNT = 48
EXPECTED_ROOT_FILE_COUNT = 2944
EXPECTED_SHARED_FILE_COUNT = 9
EXPECTED_FILE_COUNT = 2953
EXPECTED_PNG_COUNT = 2560
EXPECTED_SAMPLE_COUNT = 512
EXPECTED_SHARD_COUNT = 16
EXPECTED_SEED = 4549
EXPECTED_BATCH_SIZE = 2
EXPECTED_FLOW_MISMATCH_SHARD = 14
CONTRACT_TYPE = "safa_r9_confirm512_canonical_native_repair_v1"


class CanonicalNativeRepairError(RuntimeError):
    """Raised when immutable confirm512 evidence violates the repair contract."""


@dataclass(frozen=True)
class CanonicalNativeRepairRequest:
    repo_root: Path
    source_campaign_root: Path
    repair_root: Path
    repair_id: str
    campaign_id: str
    source_failure_sha256: str
    phase_request: PhaseResultsRequest


@dataclass(frozen=True)
class PreparedCanonicalNativeRepair:
    request: CanonicalNativeRepairRequest
    contract: Mapping[str, Any]
    contract_sha256: str
    namespace_root: Path
    phase_request: PhaseResultsRequest
    validated_phase_request: Mapping[str, Any]
    source_runs: Mapping[str, Mapping[str, Any]]
    canonical_runs: Mapping[str, Mapping[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CanonicalNativeRepairError(f"{label} is not a lowercase SHA256")
    return value


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalNativeRepairError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise CanonicalNativeRepairError(f"{label} is not an object: {path}")
    return value


def _contained(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise CanonicalNativeRepairError(
            f"{label} escapes its registered root"
        ) from error
    return resolved


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _canonical_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return _canonical_json_sha256(canonical)


def _declared_contract_binding(
    path: Path,
    *,
    digest_field: str,
    contract_type: str | None,
) -> dict[str, Any]:
    payload = _read_mapping(path, digest_field)
    if contract_type is not None and payload.get("contract_type") != contract_type:
        raise CanonicalNativeRepairError(f"{digest_field} contract type mismatch")
    declared = _require_sha256(payload.get(digest_field), digest_field)
    if _canonical_digest(payload, digest_field) != declared:
        raise CanonicalNativeRepairError(f"{digest_field} canonical digest mismatch")
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "contract_sha256": declared,
    }


def validate_canonical_native_inventory(
    *,
    campaign_root: Path,
    expected_roots: Sequence[Path],
) -> Mapping[str, Any]:
    """Hash the exact 48 shard roots plus their three shared asset directories."""
    root = campaign_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise CanonicalNativeRepairError(
            "source campaign root is not a regular directory"
        )
    resolved_roots = tuple(
        _contained(root, Path(path), "registered shard root") for path in expected_roots
    )
    if len(resolved_roots) != EXPECTED_ROOT_COUNT:
        raise CanonicalNativeRepairError(
            f"repair requires exactly {EXPECTED_ROOT_COUNT} shard roots"
        )
    if len(set(resolved_roots)) != len(resolved_roots):
        raise CanonicalNativeRepairError("registered shard roots contain duplicates")
    root_rows: list[dict[str, Any]] = []
    all_paths: set[Path] = set()
    total_root_files = 0
    total_pngs = 0
    for shard_root in sorted(resolved_roots, key=lambda value: str(value)):
        if not shard_root.is_dir() or shard_root.is_symlink():
            raise CanonicalNativeRepairError(
                f"registered shard root is invalid: {shard_root}"
            )
        files = []
        pngs = []
        for child in sorted(shard_root.rglob("*"), key=lambda value: str(value)):
            if child.is_symlink():
                raise CanonicalNativeRepairError(
                    f"generation inventory forbids symlinks: {child}"
                )
            if not child.is_file():
                continue
            resolved = child.resolve()
            if resolved in all_paths:
                raise CanonicalNativeRepairError(
                    f"generation inventory repeats a file: {resolved}"
                )
            all_paths.add(resolved)
            row = {
                "path": _relative(root, resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
            files.append(row)
            if resolved.suffix.lower() == ".png":
                pngs.append(row)
        relative_root = _relative(root, shard_root)
        arm_id = shard_root.parents[1].name
        try:
            shard_index = int(shard_root.name.removeprefix("shard_"))
        except ValueError as error:
            raise CanonicalNativeRepairError(
                f"invalid registered shard root name: {shard_root.name}"
            ) from error
        if arm_id not in EXPECTED_ARMS or not 0 <= shard_index < EXPECTED_SHARD_COUNT:
            raise CanonicalNativeRepairError(
                f"unexpected arm/shard root: {relative_root}"
            )
        expected_pngs = 32 if arm_id == "native" else 64
        if len(pngs) != expected_pngs:
            raise CanonicalNativeRepairError(
                f"{relative_root} PNG count {len(pngs)} != {expected_pngs}"
            )
        for required in ("generation_result.json", "completion.json"):
            if not (shard_root / required).is_file():
                raise CanonicalNativeRepairError(
                    f"{relative_root} lacks required {required}"
                )
        total_root_files += len(files)
        total_pngs += len(pngs)
        root_payload = {
            "arm_id": arm_id,
            "shard_index": shard_index,
            "root": relative_root,
            "file_count": len(files),
            "png_count": len(pngs),
            "files": files,
            "png_inventory_sha256": _canonical_json_sha256(
                {"pngs": [row["sha256"] for row in pngs]}
            ),
        }
        root_payload["root_inventory_sha256"] = _canonical_json_sha256(root_payload)
        root_rows.append(root_payload)
    shared_rows: list[dict[str, Any]] = []
    total_shared_files = 0
    shared_roots = sorted(
        {shard_root.parent / "shared" for shard_root in resolved_roots},
        key=lambda value: str(value),
    )
    if len(shared_roots) != len(EXPECTED_ARMS):
        raise CanonicalNativeRepairError("shared asset root set changed")
    for shared_root in shared_roots:
        if not shared_root.is_dir() or shared_root.is_symlink():
            raise CanonicalNativeRepairError(
                f"registered shared asset root is invalid: {shared_root}"
            )
        arm_id = shared_root.parents[1].name
        if arm_id not in EXPECTED_ARMS:
            raise CanonicalNativeRepairError(
                f"unexpected shared asset root: {_relative(root, shared_root)}"
            )
        files = []
        for child in sorted(shared_root.rglob("*"), key=lambda value: str(value)):
            if child.is_symlink():
                raise CanonicalNativeRepairError(
                    f"generation inventory forbids symlinks: {child}"
                )
            if not child.is_file():
                continue
            resolved = child.resolve()
            if resolved in all_paths:
                raise CanonicalNativeRepairError(
                    f"generation inventory repeats a file: {resolved}"
                )
            all_paths.add(resolved)
            files.append(
                {
                    "path": _relative(root, resolved),
                    "size_bytes": resolved.stat().st_size,
                    "sha256": _sha256_file(resolved),
                }
            )
        total_shared_files += len(files)
        shared_payload = {
            "arm_id": arm_id,
            "root": _relative(root, shared_root),
            "file_count": len(files),
            "files": files,
        }
        shared_payload["shared_inventory_sha256"] = _canonical_json_sha256(
            shared_payload
        )
        shared_rows.append(shared_payload)
    observed_pairs = {(row["arm_id"], int(row["shard_index"])) for row in root_rows}
    expected_pairs = {
        (arm_id, shard_index)
        for arm_id in EXPECTED_ARMS
        for shard_index in range(EXPECTED_SHARD_COUNT)
    }
    if observed_pairs != expected_pairs:
        raise CanonicalNativeRepairError(
            "registered root set is not the exact 3x16 grid"
        )
    total_files = total_root_files + total_shared_files
    if (
        total_root_files != EXPECTED_ROOT_FILE_COUNT
        or total_shared_files != EXPECTED_SHARED_FILE_COUNT
        or total_files != EXPECTED_FILE_COUNT
        or total_pngs != EXPECTED_PNG_COUNT
    ):
        raise CanonicalNativeRepairError(
            "frozen v8 inventory counts changed: "
            f"root_files={total_root_files}, shared_files={total_shared_files}, "
            f"files={total_files}, pngs={total_pngs}"
        )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_confirm512_generation_inventory_v1",
        "root_count": len(root_rows),
        "root_file_count": total_root_files,
        "shared_root_count": len(shared_rows),
        "shared_file_count": total_shared_files,
        "file_count": total_files,
        "png_count": total_pngs,
        "roots": sorted(
            root_rows, key=lambda row: (str(row["arm_id"]), int(row["shard_index"]))
        ),
        "shared_roots": sorted(shared_rows, key=lambda row: str(row["arm_id"])),
    }
    payload["inventory_sha256"] = _canonical_digest(payload, "inventory_sha256")
    return payload


def _strict_failure_binding(request: CanonicalNativeRepairRequest) -> dict[str, Any]:
    log_path = request.source_campaign_root / "confirm512.controller.log"
    expected = _require_sha256(
        request.source_failure_sha256, "source strict failure SHA256"
    )
    if _sha256_file(log_path) != expected:
        raise CanonicalNativeRepairError("source strict-failure log SHA256 changed")
    text = log_path.read_text(encoding="utf-8")
    if SOURCE_FAILURE_MESSAGE not in text:
        raise CanonicalNativeRepairError(
            "source log lacks the registered strict failure"
        )
    if (request.source_campaign_root / "confirm512" / "phase_results.json").exists():
        raise CanonicalNativeRepairError(
            "strict C failure unexpectedly produced phase results"
        )
    return {
        "classification": "strict_cross_worker_matched_native_png_sha_mismatch",
        "exception_type": "PhaseResultsError",
        "message": "candidate matched-native PNG differs from native run",
        "log_path": str(log_path),
        "log_sha256": expected,
        "phase_results_materialized": False,
    }


def _load_source_runs(
    phase_request: PhaseResultsRequest,
) -> tuple[Mapping[str, Any], dict[str, dict[str, Any]]]:
    validated = _validate_request(phase_request)
    loaded: dict[str, dict[str, Any]] = {}
    for spec in phase_request.runs:
        run = _load_run_evidence(validated, spec)
        arm_id = str(run["arm_id"])
        if arm_id in loaded:
            raise CanonicalNativeRepairError(f"duplicate source arm: {arm_id}")
        loaded[arm_id] = run
    if tuple(sorted(loaded)) != tuple(sorted(EXPECTED_ARMS)):
        raise CanonicalNativeRepairError("source C run arms changed")
    return validated, loaded


def _matched_native_diagnostic(
    validated: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_ids = list(validated["manifest_ids"])
    if len(manifest_ids) != EXPECTED_SAMPLE_COUNT:
        raise CanonicalNativeRepairError("validate_512 membership changed")
    native_rows = {str(row["sample_id"]): row for row in runs["native"]["rows"]}
    result: dict[str, Any] = {}
    for arm_id in EXPECTED_ARMS[1:]:
        mismatch_png = []
        mismatch_e0 = []
        mismatch_edev = []
        overlays = []
        for row in runs[arm_id]["rows"]:
            sample_id = str(row["sample_id"])
            baseline = native_rows[sample_id]
            if row["native_sha256"] != baseline["candidate_sha256"]:
                mismatch_png.append(sample_id)
            if (
                row["metrics"]["native_cosine"]
                != baseline["metrics"]["candidate_cosine"]
            ):
                mismatch_e0.append(sample_id)
            if (
                row["metrics"]["native_edev_cosine"]
                != baseline["metrics"]["edev_cosine"]
            ):
                mismatch_edev.append(sample_id)
            overlays.append(
                {
                    "sample_id": sample_id,
                    "candidate_embedded_native_sha256": row["native_sha256"],
                    "canonical_standalone_native_sha256": baseline["candidate_sha256"],
                }
            )
        expected_mismatch = (
            manifest_ids[EXPECTED_FLOW_MISMATCH_SHARD::EXPECTED_SHARD_COUNT]
            if arm_id == "flow_map2_normalized_eta_0p125"
            else []
        )
        if mismatch_png != expected_mismatch:
            raise CanonicalNativeRepairError(
                f"{arm_id} matched-native PNG mismatch list changed"
            )
        if mismatch_e0 != expected_mismatch or mismatch_edev != expected_mismatch:
            raise CanonicalNativeRepairError(
                f"{arm_id} matched-native metric mismatch list changed"
            )
        result[arm_id] = {
            "role": "diagnostic_only_excluded_from_evaluation",
            "mismatch_count": len(mismatch_png),
            "mismatch_sample_ids": mismatch_png,
            "overlay_sha256": _canonical_json_sha256({"rows": overlays}),
        }
    return {
        "standalone_native_role": "sole_canonical_baseline",
        "candidate_embedded_native_role": "diagnostic_only_excluded_from_evaluation",
        "paper_mismatch_count": 0,
        "flow_mismatch_count": 32,
        "flow_mismatch_shard": EXPECTED_FLOW_MISMATCH_SHARD,
        "arms": result,
    }


def _stable_noise_binding(
    validated: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_ids = list(validated["manifest_ids"])
    config_bindings = []
    image_size = None
    channels = None
    for arm_id in EXPECTED_ARMS:
        run = runs[arm_id]
        shards = run["output_contract"]["shards"]
        first = _read_mapping(
            Path(str(shards[0]["generation_result_path"])), "generation result"
        )
        config = first["config"]
        checkpoint = first["checkpoint"]
        if (
            config.get("sampling_seed") != EXPECTED_SEED
            or config.get("seed") != EXPECTED_SEED
            or config.get("batch_size") != EXPECTED_BATCH_SIZE
        ):
            raise CanonicalNativeRepairError(f"{arm_id} seed/batch binding changed")
        current_image_size = int(config.get("image_size", 32))
        current_channels = int(checkpoint["model_config"]["sit_input_channels"])
        if image_size is None:
            image_size = current_image_size
            channels = current_channels
        elif (image_size, channels) != (current_image_size, current_channels):
            raise CanonicalNativeRepairError("source runs changed latent shape")
        config_bindings.append(
            {
                "arm_id": arm_id,
                "algorithm_config_sha256": run["algorithm_config_sha256"],
                "runner_arm_config_sha256": run["runner_arm_config_sha256"],
                "generation_result_set_sha256": _canonical_json_sha256(
                    {
                        "generation_results": [
                            shard["generation_result_sha256"] for shard in shards
                        ]
                    }
                ),
                "per_sample_set_sha256": _canonical_json_sha256(
                    {"per_sample": [shard["per_sample_sha256"] for shard in shards]}
                ),
            }
        )
    assert image_size is not None and channels is not None
    rows = []
    for sample_id in manifest_ids:
        sample_seed = stable_sample_seed(EXPECTED_SEED, sample_id)
        tensor = make_x_init_for_sample_ids(
            [sample_id],
            EXPECTED_SEED,
            image_size,
            "cpu",
            torch.float32,
            channels=channels,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "stable_sample_seed": sample_seed,
                "derived_x_init_sha256": hashlib.sha256(
                    tensor.contiguous().numpy().tobytes()
                ).hexdigest(),
            }
        )
    return {
        "base_seed": EXPECTED_SEED,
        "batch_size": EXPECTED_BATCH_SIZE,
        "image_size": image_size,
        "channels": channels,
        "derivation": "sha256(base_seed NUL sample_id) -> CPU torch.Generator -> float32 randn",
        "actual_x_init_sha256_persisted_by_v8": False,
        "actual_gpu_uuid_persisted_by_v8": False,
        "sample_count": len(rows),
        "ordered_rows_sha256": _canonical_json_sha256({"rows": rows}),
        "config_bindings": config_bindings,
    }


def _canonicalize_candidate(
    candidate: Mapping[str, Any],
    native: Mapping[str, Any],
) -> dict[str, Any]:
    native_by_id = {str(row["sample_id"]): row for row in native["rows"]}
    rows = []
    overlays = []
    for source_row in candidate["rows"]:
        row = dict(source_row)
        row["metrics"] = dict(source_row["metrics"])
        baseline = native_by_id[str(row["sample_id"])]
        overlays.append(
            {
                "sample_id": row["sample_id"],
                "embedded_native_sha256": row["native_sha256"],
                "standalone_native_sha256": baseline["candidate_sha256"],
            }
        )
        row["native"] = baseline["candidate"]
        row["native_sha256"] = baseline["candidate_sha256"]
        row["metrics"]["native_cosine"] = baseline["metrics"]["candidate_cosine"]
        row["metrics"]["native_edev_cosine"] = baseline["metrics"]["edev_cosine"]
        rows.append(row)
    output_contract = dict(candidate["output_contract"])
    output_contract["images"] = [
        {
            "sample_id": row["sample_id"],
            "source": row["source"],
            "native": row["native"],
            "candidate": row["candidate"],
            "source_sha256": row["source_sha256"],
            "native_sha256": row["native_sha256"],
            "candidate_sha256": row["candidate_sha256"],
        }
        for row in rows
    ]
    output_contract["canonical_native_overlay_sha256"] = _canonical_json_sha256(
        {"rows": overlays}
    )
    canonical = dict(candidate)
    canonical["rows"] = rows
    canonical["output_contract"] = output_contract
    canonical["evidence_binding_sha256"] = _canonical_json_sha256(output_contract)
    canonical["canonical_native_overlay_sha256"] = output_contract[
        "canonical_native_overlay_sha256"
    ]
    canonical["source_generation_output_sha256"] = candidate["output_sha256"]
    return canonical


def _canonical_views(
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    native = runs["native"]
    canonical = {"native": native}
    for arm_id in EXPECTED_ARMS[1:]:
        canonical[arm_id] = _canonicalize_candidate(runs[arm_id], native)
    return canonical


def _worker_gpu_provenance_binding(
    request: CanonicalNativeRepairRequest,
) -> dict[str, Any]:
    expected_ids = {
        f"confirm512:{spec.logical_run_id}:shard-{index}"
        for spec in request.phase_request.runs
        for index in range(EXPECTED_SHARD_COUNT)
    }
    observed = {}
    for path in (request.source_campaign_root / "worker_status").glob("*.json"):
        payload = _read_mapping(path, "worker status")
        worker_id = payload.get("worker_id")
        if worker_id not in expected_ids:
            continue
        if payload.get("state") != "succeeded":
            raise CanonicalNativeRepairError(
                "source generation worker is not succeeded"
            )
        if "gpu_uuid" in payload or "gpu_index" in payload or "gpu_slot" in payload:
            raise CanonicalNativeRepairError(
                "v8 worker status unexpectedly changed GPU provenance schema"
            )
        observed[str(worker_id)] = {
            "status_file_sha256": _sha256_file(path),
            "pid": payload.get("pid"),
            "process_start_ticks": payload.get("process_start_ticks"),
        }
    if set(observed) != expected_ids:
        raise CanonicalNativeRepairError("source worker-status coverage changed")
    return {
        "status": "unavailable",
        "reason": "v8 worker artifacts did not persist actual GPU UUID or slot",
        "worker_count": len(observed),
        "worker_status_set_sha256": _canonical_json_sha256({"workers": observed}),
    }


def build_canonical_native_repair(
    request: CanonicalNativeRepairRequest,
) -> PreparedCanonicalNativeRepair:
    repo_root = request.repo_root.resolve()
    source_root = _contained(repo_root, request.source_campaign_root, "campaign root")
    repair_root = _contained(repo_root, request.repair_root, "repair root")
    if request.campaign_id != SOURCE_CAMPAIGN_ID:
        raise CanonicalNativeRepairError("canonical repair is bound only to formal v8")
    if request.phase_request.campaign_id != request.campaign_id:
        raise CanonicalNativeRepairError("phase request campaign ID mismatch")
    if (
        request.phase_request.phase != "confirm512"
        or request.phase_request.confirm_seed != EXPECTED_SEED
        or request.phase_request.phase_root.resolve()
        != (source_root / "confirm512").resolve()
    ):
        raise CanonicalNativeRepairError("phase request is not the frozen v8 C request")
    if not request.repair_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in request.repair_id
    ):
        raise CanonicalNativeRepairError("repair ID is not lowercase filesystem-safe")
    validated, runs = _load_source_runs(request.phase_request)
    expected_roots = tuple(
        path for spec in request.phase_request.runs for path in spec.shard_output_dirs
    )
    inventory = validate_canonical_native_inventory(
        campaign_root=source_root,
        expected_roots=expected_roots,
    )
    diagnostic = _matched_native_diagnostic(validated, runs)
    canonical_runs = _canonical_views(runs)
    calibration_selection_path = source_root / "calibration_report_only_selection.json"
    calibration_selection = validate_calibration_report_only_selection_contract(
        _read_mapping(calibration_selection_path, "calibration selection"),
        repo_root=repo_root,
    )
    continuation_path = source_root / "confirm_continuation_contract.json"
    continuation = _declared_contract_binding(
        continuation_path,
        digest_field="confirm_continuation_sha256",
        contract_type="safa_r9_confirm_continuation_contract_v1",
    )
    contract: dict[str, Any] = {
        "schema_version": 1,
        "contract_type": CONTRACT_TYPE,
        "campaign_id": request.campaign_id,
        "repair_id": request.repair_id,
        "phase": "confirm512",
        "strict_failure": _strict_failure_binding(request),
        "source_campaign": {
            "path": str(source_root),
            "campaign_runtime_sha256": request.phase_request.campaign_runtime_sha256,
            "manifest_contracts_sha256": request.phase_request.manifest_contracts_sha256,
            "validate_512_manifest_sha256": request.phase_request.manifest_sha256,
            "source_index_sha256": request.phase_request.source_index_sha256,
            "checkpoint_sha256": request.phase_request.checkpoint_sha256,
            "calibration_selection": {
                "path": str(calibration_selection_path),
                "file_sha256": _sha256_file(calibration_selection_path),
                "selection_sha256": calibration_selection[
                    "calibration_selection_sha256"
                ],
            },
            "confirm_continuation": continuation,
        },
        "generation_inventory": inventory,
        "matched_native_diagnostic": diagnostic,
        "stable_input_noise_config_binding": _stable_noise_binding(validated, runs),
        "actual_generation_gpu": _worker_gpu_provenance_binding(request),
        "canonical_native_policy": {
            "standalone_native": "sole_baseline_for_all_C_evaluation",
            "candidate_embedded_native": "diagnostic_only_excluded_from_all_C_evaluation",
            "candidate_generated_images": "immutable_source_evidence",
            "generation_execution_count": 0,
            "generation_retry_allowed": False,
            "source_generation_mutation_allowed": False,
            "evaluation_parallelism": 4,
            "quality_unit_count": 3,
            "arcface_unit_count": 2,
            "visual_coverage_role": "report_only",
            "selection": "one_post_hoc_winner_bound_to_repair",
        },
        "canonical_views": {
            arm_id: {
                "source_generation_output_sha256": runs[arm_id]["output_sha256"],
                "canonical_evidence_binding_sha256": canonical_runs[arm_id][
                    "evidence_binding_sha256"
                ],
                "canonical_native_overlay_sha256": (
                    None
                    if arm_id == "native"
                    else canonical_runs[arm_id]["canonical_native_overlay_sha256"]
                ),
            }
            for arm_id in EXPECTED_ARMS
        },
    }
    contract["repair_contract_sha256"] = _canonical_digest(
        contract, "repair_contract_sha256"
    )
    namespace_root = (
        repair_root / request.repair_id / contract["repair_contract_sha256"]
    )
    return PreparedCanonicalNativeRepair(
        request=request,
        contract=contract,
        contract_sha256=str(contract["repair_contract_sha256"]),
        namespace_root=namespace_root,
        phase_request=request.phase_request,
        validated_phase_request=validated,
        source_runs=runs,
        canonical_runs=canonical_runs,
    )


def materialize_repair_contract(
    prepared: PreparedCanonicalNativeRepair,
) -> Path:
    path = prepared.namespace_root / "repair_contract.json"
    write_immutable_contract(
        path,
        prepared.contract,
        digest_field="repair_contract_sha256",
    )
    return path


def _bind_run_to_repair(
    run: Mapping[str, Any],
    repair_sha256: str,
) -> dict[str, Any]:
    bound = dict(run)
    logical_run_id = f"repair_{repair_sha256}__{run['logical_run_id']}"
    bound["logical_run_id"] = logical_run_id
    output_contract = dict(run["output_contract"])
    output_contract["logical_run_id"] = logical_run_id
    bound["output_contract"] = output_contract
    bound["evidence_binding_sha256"] = _canonical_json_sha256(output_contract)
    return bound


def _reference_misses(
    candidate_quality: Mapping[str, Any],
    native_quality: Mapping[str, Any],
    *,
    e0: float,
    delta_e0: float,
    delta_edev: float,
    privacy_upper: float | None,
) -> list[str]:
    misses = []
    if candidate_quality["fid"] > native_quality["fid"] + 3.0:
        misses.append("fid_above_native_plus_3")
    if candidate_quality["kid"] > native_quality["kid"] + 0.005:
        misses.append("kid_above_native_plus_0.005")
    if candidate_quality["niqe"] > native_quality["niqe"] + 0.10:
        misses.append("niqe_above_native_plus_0.10")
    if candidate_quality["sharpness"] < max(300.0, 0.95 * native_quality["sharpness"]):
        misses.append("sharpness_below_reference")
    if e0 < 0.75:
        misses.append("e0_below_0.75")
    if delta_e0 < 0.30:
        misses.append("delta_e0_below_0.30")
    if delta_edev < 0.05:
        misses.append("delta_edev_below_0.05")
    if privacy_upper is not None and privacy_upper > 0.02:
        misses.append("privacy_delta_upper_gt_0.02")
    return misses


def _canonical_visual_coverage(run: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "canonical_native_sha256": row["native_sha256"],
            "candidate_sha256": row["candidate_sha256"],
        }
        for row in run["rows"]
    ]
    return {
        "role": "report_only",
        "review_status": "not_reinterpreted_by_repair",
        "sample_count": len(rows),
        "coverage_sha256": _canonical_json_sha256({"rows": rows}),
    }


def _arm_report(
    *,
    request: PhaseResultsRequest,
    validated: Mapping[str, Any],
    run: Mapping[str, Any],
    candidate_quality: Mapping[str, Any],
    native_quality: Mapping[str, Any],
    arcface: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_ids = list(validated["manifest_ids"])
    paired_rows = _paired_metric_rows(
        run,
        candidate_quality=candidate_quality,
        native_quality=native_quality,
        manifest_ids=manifest_ids,
    )
    paired_contract = _build_paired_metric_rows_contract(
        paired_rows,
        manifest_ids=manifest_ids,
        expected_seeds=(EXPECTED_SEED,),
    )
    paired_bootstrap = paired_metric_cluster_bootstrap(
        paired_contract,
        expected_seeds=(EXPECTED_SEED,),
        expected_sample_count=EXPECTED_SAMPLE_COUNT,
        bootstrap_seed=request.bootstrap_seed,
    )
    exact_one = bool(arcface["exact_one"])
    privacy_rows = []
    privacy_bootstrap = None
    if exact_one:
        privacy_rows = [
            {
                "sample_id": row["sample_id"],
                "seed": EXPECTED_SEED,
                "source_candidate_cosine": row["source_candidate_cosine"],
                "source_native_cosine": row["source_native_cosine"],
            }
            for row in arcface["rows"]
        ]
        privacy_bootstrap = privacy_delta_cluster_bootstrap(
            privacy_rows,
            expected_seeds=(EXPECTED_SEED,),
            bootstrap_seed=request.bootstrap_seed,
        )
    e0 = _mean_metric(run["rows"], "candidate_cosine")
    native_e0 = _mean_metric(run["rows"], "native_cosine")
    delta_e0 = e0 - native_e0
    delta_edev = _mean_metric(run["rows"], "edev_cosine") - _mean_metric(
        run["rows"], "native_edev_cosine"
    )
    privacy_upper = (
        None
        if privacy_bootstrap is None
        else float(privacy_bootstrap["upper_95_one_sided"])
    )
    coverage_failures = [] if exact_one else ["arcface_not_exactly_one_face_per_image"]
    report = {
        "arm_id": run["arm_id"],
        "family": run["family"],
        "config_sha256": run["algorithm_config_sha256"],
        "source_generation_output_sha256": run["source_generation_output_sha256"],
        "canonical_evidence_binding_sha256": run["evidence_binding_sha256"],
        "seed": EXPECTED_SEED,
        "quality": {
            "fid": candidate_quality["fid"],
            "native_fid": native_quality["fid"],
            "kid": candidate_quality["kid"],
            "native_kid": native_quality["kid"],
            "niqe": candidate_quality["niqe"],
            "native_niqe": native_quality["niqe"],
            "sharpness": candidate_quality["sharpness"],
            "native_sharpness": native_quality["sharpness"],
            "candidate_quality_evidence_sha256": candidate_quality[
                "quality_evidence_sha256"
            ],
            "native_quality_evidence_sha256": native_quality["quality_evidence_sha256"],
        },
        "representation": {
            "e0": e0,
            "native_e0": native_e0,
            "delta_e0": delta_e0,
            "delta_edev": delta_edev,
            "paired_metric_bootstrap": paired_bootstrap,
        },
        "privacy": {
            "arcface_exact_one": exact_one,
            "source_exact_one_count": arcface["source_exact_one_count"],
            "native_exact_one_count": arcface["native_exact_one_count"],
            "candidate_exact_one_count": arcface["candidate_exact_one_count"],
            "paired_exact_one_count": arcface["paired_exact_one_count"],
            "failure_sample_ids": arcface["failure_sample_ids"],
            "privacy_bootstrap": privacy_bootstrap,
            "arcface_evidence_sha256": arcface["arcface_evidence_sha256"],
        },
        "visual_coverage": _canonical_visual_coverage(run),
        "observations": {
            "numerical_metrics_role": "report_only",
            "privacy_metrics_role": "report_only",
            "visual_metrics_role": "report_only_coverage",
            "reference_misses": _reference_misses(
                candidate_quality,
                native_quality,
                e0=e0,
                delta_e0=delta_e0,
                delta_edev=delta_edev,
                privacy_upper=privacy_upper,
            ),
        },
        "coverage_failures": coverage_failures,
        "passed_coverage": not coverage_failures,
    }
    report["evaluator_evidence_sha256"] = _canonical_json_sha256(
        {
            "candidate_quality": candidate_quality["quality_evidence_sha256"],
            "native_quality": native_quality["quality_evidence_sha256"],
            "arcface": arcface["arcface_evidence_sha256"],
            "paired_metrics": paired_bootstrap["paired_metric_bootstrap_sha256"],
            "privacy": (
                None
                if privacy_bootstrap is None
                else privacy_bootstrap["bootstrap_sha256"]
            ),
        }
    )
    return report


def _post_hoc_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    quality = row["quality"]
    representation = row["representation"]
    return (
        quality["kid"],
        quality["fid"],
        -representation["delta_edev"],
        -representation["e0"],
        row["arm_id"],
    )


def execute_canonical_evaluations(
    prepared: PreparedCanonicalNativeRepair,
    *,
    quality_evaluator: Any,
    arcface_evaluator: Any,
) -> Mapping[str, Any]:
    repair_request = replace(
        prepared.phase_request,
        phase_root=prepared.namespace_root / "confirm512",
    )
    bound_runs = {
        arm_id: _bind_run_to_repair(run, prepared.contract_sha256)
        for arm_id, run in prepared.canonical_runs.items()
    }
    native = bound_runs["native"]
    native_samples = _sample_evidence(native)
    tasks: dict[str, tuple[Any, tuple[Any, ...]]] = {
        "quality:native": (
            _evaluate_quality,
            (repair_request, native, native_samples, "native", quality_evaluator),
        )
    }
    for arm_id in EXPECTED_ARMS[1:]:
        run = bound_runs[arm_id]
        samples = _sample_evidence(run)
        tasks[f"quality:{arm_id}"] = (
            _evaluate_quality,
            (repair_request, run, samples, "candidate", quality_evaluator),
        )
        tasks[f"arcface:{arm_id}"] = (
            _evaluate_arcface,
            (repair_request, run, samples, arcface_evaluator),
        )
    futures: dict[str, Future[Any]] = {}
    with ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="safa-r9-canonical-repair",
    ) as executor:
        for key, (function, arguments) in tasks.items():
            futures[key] = executor.submit(function, *arguments)
        try:
            evaluated = {key: futures[key].result() for key in tasks}
        except BaseException:
            for future in futures.values():
                future.cancel()
            raise
    native_quality = evaluated["quality:native"]
    arms = [
        _arm_report(
            request=repair_request,
            validated=prepared.validated_phase_request,
            run=bound_runs[arm_id],
            candidate_quality=evaluated[f"quality:{arm_id}"],
            native_quality=native_quality,
            arcface=evaluated[f"arcface:{arm_id}"],
        )
        for arm_id in EXPECTED_ARMS[1:]
    ]
    passing = sorted(
        (row for row in arms if row["passed_coverage"]),
        key=_post_hoc_rank,
    )
    selected = [] if not passing else [str(passing[0]["arm_id"])]
    automatic = {
        "schema_version": 1,
        "contract_type": "safa_r9_confirm512_canonical_automatic_v1",
        "campaign_id": prepared.request.campaign_id,
        "phase": "confirm512",
        "repair_contract_sha256": prepared.contract_sha256,
        "generation_inventory_sha256": prepared.contract["generation_inventory"][
            "inventory_sha256"
        ],
        "evaluator_unit_count": len(tasks),
        "quality_unit_count": 3,
        "arcface_unit_count": 2,
        "canonical_native_role": "sole_baseline",
        "candidate_embedded_native_role": "diagnostic_only",
        "arms": arms,
    }
    automatic["automatic_evidence_sha256"] = _canonical_digest(
        automatic, "automatic_evidence_sha256"
    )
    gate = {
        "schema_version": 1,
        "contract_type": "safa_r9_confirm512_canonical_report_only_gate_v1",
        "campaign_id": prepared.request.campaign_id,
        "phase": "confirm512",
        "repair_contract_sha256": prepared.contract_sha256,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "policy": {
            "coverage_role": "hard_requirement",
            "numerical_metrics_role": "report_only",
            "privacy_metrics_role": "report_only",
            "visual_metrics_role": "report_only_coverage",
            "ranking": ["kid", "fid", "-delta_edev", "-e0", "arm_id"],
            "reselection_allowed": False,
        },
        "evaluated": [
            {
                "arm_id": row["arm_id"],
                "config_sha256": row["config_sha256"],
                "evaluator_evidence_sha256": row["evaluator_evidence_sha256"],
                "passed_coverage": row["passed_coverage"],
                "coverage_failures": row["coverage_failures"],
            }
            for row in arms
        ],
        "selected_arm_ids": selected,
        "verdict": "winner_locked" if selected else "stop_zero_coverage_candidates",
    }
    gate["gate_contract_sha256"] = _canonical_digest(gate, "gate_contract_sha256")
    phase_root = repair_request.phase_root
    write_immutable_contract(
        phase_root / "automatic_evidence.json",
        automatic,
        digest_field="automatic_evidence_sha256",
    )
    write_immutable_contract(
        phase_root / "gate_contract.json",
        gate,
        digest_field="gate_contract_sha256",
    )
    selection = None
    if selected:
        winner = next(row for row in arms if row["arm_id"] == selected[0])
        selection = {
            "schema_version": 1,
            "contract_type": "safa_r9_confirm512_canonical_selection_v1",
            "campaign_id": prepared.request.campaign_id,
            "repair_contract_sha256": prepared.contract_sha256,
            "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
            "gate_contract_sha256": gate["gate_contract_sha256"],
            "generation_inventory_sha256": automatic["generation_inventory_sha256"],
            "manifest_sha256": prepared.phase_request.manifest_sha256,
            "winner": {
                "arm_id": winner["arm_id"],
                "config_sha256": winner["config_sha256"],
                "source_generation_output_sha256": winner[
                    "source_generation_output_sha256"
                ],
                "canonical_evidence_binding_sha256": winner[
                    "canonical_evidence_binding_sha256"
                ],
                "evaluator_evidence_sha256": winner["evaluator_evidence_sha256"],
            },
            "reselection_allowed": False,
        }
        selection["selection_sha256"] = _canonical_digest(selection, "selection_sha256")
        write_immutable_contract(
            prepared.namespace_root / "selection.json",
            selection,
            digest_field="selection_sha256",
        )
    result = {
        "repair_contract_sha256": prepared.contract_sha256,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "gate_contract_sha256": gate["gate_contract_sha256"],
        "selection_sha256": (
            None if selection is None else selection["selection_sha256"]
        ),
        "winner_arm_id": None if not selected else selected[0],
        "verdict": gate["verdict"],
        "evaluator_unit_count": len(tasks),
        "generation_execution_count": 0,
    }
    result["repair_result_sha256"] = _canonical_digest(result, "repair_result_sha256")
    write_immutable_contract(
        prepared.namespace_root / "repair_result.json",
        result,
        digest_field="repair_result_sha256",
    )
    return result


def materialize_canonical_native_repair(
    request: CanonicalNativeRepairRequest,
    *,
    quality_runner: Any,
    arcface_runner: Any,
) -> Mapping[str, Any]:
    prepared = build_canonical_native_repair(request)
    materialize_repair_contract(prepared)
    return execute_canonical_evaluations(
        prepared,
        quality_evaluator=quality_runner,
        arcface_evaluator=arcface_runner,
    )
