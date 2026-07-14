"""Superseding report-only C selection over immutable canonical-native v1 evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safa.evaluation.r9_campaign_contracts import (
    derive_visual_arm_pass,
    privacy_delta_cluster_bootstrap,
    write_immutable_contract,
)
from safa.evaluation.r9_confirm512_canonical_repair import (
    EXPECTED_ARMS,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_SEED,
    PreparedCanonicalNativeRepair,
)
from safa.evaluation.r9_phase_results import (
    _canonical_json_sha256,
    _materialize_visual_unit,
    validate_visual_review,
)


SOURCE_REPAIR_SHA256 = (
    "e9e523e5a8da863b6a1ffed8e99a28816fa7403daf5c14ca3b128b90e2750528"
)
UNIQUE_ARCFACE_MISS = (
    "val:Manually_Annotated_Images/344/"
    "f22b6a6870032b7296c7f03febb8c9f8fee7e8199a38b4933b523219.jpg"
)
CONTRACT_TYPE = "safa_r9_confirm512_report_only_supersession_v2"
AWAITING_VISUAL_REVIEW_EXIT_CODE = 20


class Confirm512SupersessionError(RuntimeError):
    """Raised when v1 evidence or v2 review coverage violates the contract."""


@dataclass(frozen=True)
class Confirm512SupersessionRequest:
    repo_root: Path
    campaign_root: Path
    source_namespace_root: Path
    supersession_root: Path
    supersession_id: str
    campaign_id: str
    source: PreparedCanonicalNativeRepair


@dataclass(frozen=True)
class PreparedConfirm512Supersession:
    request: Confirm512SupersessionRequest
    contract: Mapping[str, Any]
    contract_sha256: str
    namespace_root: Path
    source_automatic: Mapping[str, Any]
    evaluator_results: Mapping[str, Mapping[str, Any]]
    arms: tuple[Mapping[str, Any], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Confirm512SupersessionError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise Confirm512SupersessionError(f"{label} is not an object")
    return value


def _canonical_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return _canonical_json_sha256(canonical)


def _bind_digest(
    path: Path, *, digest_field: str, contract_type: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_mapping(path, digest_field)
    if contract_type is not None and payload.get("contract_type") != contract_type:
        raise Confirm512SupersessionError(f"{path} contract type changed")
    declared = payload.get(digest_field)
    if (
        not isinstance(declared, str)
        or _canonical_digest(payload, digest_field) != declared
    ):
        raise Confirm512SupersessionError(f"{path} canonical digest changed")
    return payload, {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "contract_sha256": declared,
    }


def _complete_case_privacy(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_seed: int
) -> dict[str, Any]:
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise Confirm512SupersessionError(
            "ArcFace result must contain 512 ordered rows"
        )
    complete = []
    excluded = []
    for row in rows:
        counts = tuple(
            row.get(field)
            for field in (
                "source_face_count",
                "native_face_count",
                "candidate_face_count",
            )
        )
        if counts == (1, 1, 1):
            complete.append(
                {
                    "sample_id": row["sample_id"],
                    "seed": EXPECTED_SEED,
                    "source_candidate_cosine": row["source_candidate_cosine"],
                    "source_native_cosine": row["source_native_cosine"],
                }
            )
        else:
            excluded.append({"sample_id": row.get("sample_id"), "face_counts": counts})
    if excluded != [{"sample_id": UNIQUE_ARCFACE_MISS, "face_counts": (1, 0, 1)}]:
        raise Confirm512SupersessionError("ArcFace unique-miss contract changed")
    bootstrap = privacy_delta_cluster_bootstrap(
        complete,
        expected_seeds=(EXPECTED_SEED,),
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "role": "report_only_complete_case",
        "observation_count": len(complete),
        "excluded_sample_ids": [UNIQUE_ARCFACE_MISS],
        "bootstrap": bootstrap,
    }


def _materialize_visual_evidence(
    prepared: PreparedConfirm512Supersession,
) -> tuple[dict[str, Any], ...]:
    phase_root = prepared.namespace_root / "confirm512"
    request = replace(prepared.source.phase_request, phase_root=phase_root)
    validated = dict(prepared.source.validated_phase_request)
    validated["request"] = request
    units = []
    for arm_id in EXPECTED_ARMS[1:]:
        run = dict(prepared.source.canonical_runs[arm_id])
        run["logical_run_id"] = f"report_only_v2__{arm_id}"
        units.append(_materialize_visual_unit(validated, run))
    if len(units) != 2:
        raise Confirm512SupersessionError("v2 requires two complete visual units")
    return tuple(units)


def _post_hoc_rank(row: Mapping[str, Any], severe_count: int) -> tuple[Any, ...]:
    return (
        severe_count,
        row["quality"]["kid"],
        row["quality"]["fid"],
        -row["representation"]["delta_edev"],
        -row["representation"]["e0"],
        row["arm_id"],
    )


def materialize_visual_stage(
    prepared: PreparedConfirm512Supersession,
) -> Mapping[str, Any]:
    """Create 2x512 canonical overlays and stop until both official reviews exist."""
    write_immutable_contract(
        prepared.namespace_root / "supersession_contract.json",
        prepared.contract,
        digest_field="supersession_contract_sha256",
    )
    units = _materialize_visual_evidence(prepared)
    automatic = {
        "schema_version": 2,
        "contract_type": "safa_r9_confirm512_report_only_automatic_v2",
        "campaign_id": prepared.request.campaign_id,
        "supersession_contract_sha256": prepared.contract_sha256,
        "source_automatic_evidence_sha256": prepared.source_automatic[
            "automatic_evidence_sha256"
        ],
        "generation_inventory_sha256": prepared.contract["generation_inventory_sha256"],
        "generation_execution_count": 0,
        "evaluator_execution_count": 0,
        "arms": list(prepared.arms),
        "visual_units": list(units),
    }
    automatic["automatic_evidence_sha256"] = _canonical_digest(
        automatic, "automatic_evidence_sha256"
    )
    write_immutable_contract(
        prepared.namespace_root / "confirm512" / "automatic_evidence_v2.json",
        automatic,
        digest_field="automatic_evidence_sha256",
    )
    awaiting = {
        "schema_version": 2,
        "contract_type": "safa_r9_confirm512_report_only_status_v2",
        "status": "awaiting_visual_review",
        "campaign_id": prepared.request.campaign_id,
        "supersession_contract_sha256": prepared.contract_sha256,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "bounded_exit_code": AWAITING_VISUAL_REVIEW_EXIT_CODE,
        "required_reviews": [
            {
                "unit_id": unit["unit_id"],
                "arm_id": unit["arm_id"],
                "evidence_path": unit["evidence_path"],
                "evidence_contract_sha256": unit["evidence_contract_sha256"],
                "review_path": unit["review_path"],
            }
            for unit in units
        ],
    }
    awaiting["awaiting_visual_review_sha256"] = _canonical_digest(
        awaiting, "awaiting_visual_review_sha256"
    )
    write_immutable_contract(
        prepared.namespace_root / "confirm512" / "awaiting_visual_review.json",
        awaiting,
        digest_field="awaiting_visual_review_sha256",
    )
    return awaiting


def _expected_evaluator_paths(source_root: Path, source_sha256: str) -> dict[str, Path]:
    prefix = f"repair_{source_sha256}__"
    evaluator_root = source_root / "confirm512" / "evaluator_runs"
    return {
        "quality:native": evaluator_root
        / "quality"
        / f"{prefix}native__native"
        / "result.json",
        "quality:flow_map2_normalized_eta_0p125": evaluator_root
        / "quality"
        / f"{prefix}flow_map2_normalized_eta_0p125__candidate"
        / "result.json",
        "quality:paper_eta_0p125": evaluator_root
        / "quality"
        / f"{prefix}paper_eta_0p125__candidate"
        / "result.json",
        "arcface:flow_map2_normalized_eta_0p125": evaluator_root
        / "arcface"
        / f"{prefix}flow_map2_normalized_eta_0p125"
        / "result.json",
        "arcface:paper_eta_0p125": evaluator_root
        / "arcface"
        / f"{prefix}paper_eta_0p125"
        / "result.json",
    }


def build_report_only_supersession(
    request: Confirm512SupersessionRequest,
) -> PreparedConfirm512Supersession:
    """Bind v1 closure and compute report-only complete-case evidence without writes."""
    if request.campaign_id != request.source.request.campaign_id:
        raise Confirm512SupersessionError("campaign ID disagrees with source repair")
    if request.source.contract_sha256 != SOURCE_REPAIR_SHA256:
        raise Confirm512SupersessionError("source repair SHA256 is not frozen v1")
    campaign_root = request.campaign_root.resolve()
    source_root = request.source_namespace_root.resolve()
    supersession_root = request.supersession_root.resolve()
    for path, label in (
        (source_root, "source namespace"),
        (supersession_root, "supersession root"),
    ):
        try:
            path.relative_to(campaign_root)
        except ValueError as error:
            raise Confirm512SupersessionError(
                f"{label} escapes the campaign"
            ) from error
    source_contract, source_contract_binding = _bind_digest(
        source_root / "repair_contract.json",
        digest_field="repair_contract_sha256",
        contract_type="safa_r9_confirm512_canonical_native_repair_v1",
    )
    if source_contract["repair_contract_sha256"] != request.source.contract_sha256:
        raise Confirm512SupersessionError("materialized source repair disagrees")
    automatic, automatic_binding = _bind_digest(
        source_root / "confirm512" / "automatic_evidence.json",
        digest_field="automatic_evidence_sha256",
        contract_type="safa_r9_confirm512_canonical_automatic_v1",
    )
    gate, gate_binding = _bind_digest(
        source_root / "confirm512" / "gate_contract.json",
        digest_field="gate_contract_sha256",
        contract_type="safa_r9_confirm512_canonical_report_only_gate_v1",
    )
    result, result_binding = _bind_digest(
        source_root / "repair_result.json",
        digest_field="repair_result_sha256",
        contract_type=None,
    )
    if (
        gate.get("policy", {}).get("coverage_role") != "hard_requirement"
        or gate.get("verdict") != "stop_zero_coverage_candidates"
        or result.get("winner_arm_id") is not None
        or result.get("selection_sha256") is not None
        or result.get("generation_execution_count") != 0
        or result.get("evaluator_unit_count") != 5
    ):
        raise Confirm512SupersessionError("source v1 closure state changed")
    inventory = source_contract.get("generation_inventory")
    if (
        not isinstance(inventory, Mapping)
        or inventory.get("root_count") != 48
        or inventory.get("root_file_count") != 2944
        or inventory.get("shared_file_count") != 9
        or inventory.get("png_count") != 2560
        or automatic.get("generation_inventory_sha256")
        != inventory.get("inventory_sha256")
    ):
        raise Confirm512SupersessionError("source 48-root inventory binding changed")
    evaluator_results: dict[str, Mapping[str, Any]] = {}
    evaluator_bindings = []
    privacy_by_arm = {}
    for key, path in _expected_evaluator_paths(
        source_root, SOURCE_REPAIR_SHA256
    ).items():
        payload, binding = _bind_digest(
            path,
            digest_field="evaluator_output_sha256",
            contract_type="safa_r9_phase_evaluator_output_v1",
        )
        evaluator_results[key] = payload
        evaluator_bindings.append({"unit_id": key, **binding})
        if key.startswith("arcface:"):
            rows = payload.get("result")
            if not isinstance(rows, list):
                raise Confirm512SupersessionError(
                    "ArcFace evaluator result is not rows"
                )
            privacy_by_arm[key.split(":", 1)[1]] = _complete_case_privacy(
                rows, bootstrap_seed=request.source.phase_request.bootstrap_seed
            )
    if len(evaluator_results) != 5:
        raise Confirm512SupersessionError(
            "v2 must bind exactly five v1 evaluator results"
        )
    source_arms = {
        str(row["arm_id"]): row
        for row in automatic.get("arms", ())
        if isinstance(row, Mapping)
    }
    if set(source_arms) != set(EXPECTED_ARMS[1:]):
        raise Confirm512SupersessionError("source automatic candidate set changed")
    arms = []
    for arm_id in EXPECTED_ARMS[1:]:
        source_arm = source_arms[arm_id]
        privacy = privacy_by_arm[arm_id]
        arms.append(
            {
                "arm_id": arm_id,
                "family": source_arm["family"],
                "config_sha256": source_arm["config_sha256"],
                "source_generation_output_sha256": source_arm[
                    "source_generation_output_sha256"
                ],
                "canonical_evidence_binding_sha256": source_arm[
                    "canonical_evidence_binding_sha256"
                ],
                "evaluator_evidence_sha256": source_arm["evaluator_evidence_sha256"],
                "quality": source_arm["quality"],
                "representation": source_arm["representation"],
                "coverage": {
                    "role": "report_only",
                    "source_exact_one_count": source_arm["privacy"][
                        "source_exact_one_count"
                    ],
                    "native_exact_one_count": source_arm["privacy"][
                        "native_exact_one_count"
                    ],
                    "candidate_exact_one_count": source_arm["privacy"][
                        "candidate_exact_one_count"
                    ],
                    "paired_exact_one_count": source_arm["privacy"][
                        "paired_exact_one_count"
                    ],
                    "failure_sample_ids": source_arm["privacy"]["failure_sample_ids"],
                },
                "complete_case_privacy": privacy,
                "visual_review": {
                    "role": "report_only",
                    "status": "pending_full_512_review",
                    "required_sample_count": EXPECTED_SAMPLE_COUNT,
                    "severe_count": None,
                },
            }
        )
    contract = {
        "schema_version": 2,
        "contract_type": CONTRACT_TYPE,
        "campaign_id": request.campaign_id,
        "supersession_id": request.supersession_id,
        "supersedes": {
            "reason": "v1 incorrectly treated C ArcFace coverage as a hard gate",
            "source_repair": source_contract_binding,
            "source_automatic": automatic_binding,
            "source_gate": gate_binding,
            "source_repair_result": result_binding,
            "source_gate_verdict": gate["verdict"],
        },
        "evaluator_results": evaluator_bindings,
        "generation_inventory_sha256": inventory["inventory_sha256"],
        "generation_inventory_counts": {
            "root_count": 48,
            "root_file_count": 2944,
            "shared_file_count": 9,
            "file_count": 2953,
            "png_count": 2560,
        },
        "unique_arcface_miss": UNIQUE_ARCFACE_MISS,
        "complete_case_observation_count": 511,
        "policy": {
            "coverage_role": "report_only",
            "numerical_metrics_role": "report_only",
            "privacy_metrics_role": "report_only_complete_case",
            "visual_metrics_role": "report_only_full_512_required_before_selection",
            "ranking": [
                "severe_count",
                "kid",
                "fid",
                "-delta_edev",
                "-e0",
                "arm_id",
            ],
            "reselection_allowed": False,
        },
        "execution": {
            "generation_execution_count": 0,
            "evaluator_execution_count": 0,
            "visual_evidence_unit_count": 2,
        },
        "arms": arms,
    }
    contract["supersession_contract_sha256"] = _canonical_digest(
        contract, "supersession_contract_sha256"
    )
    namespace = (
        supersession_root
        / request.supersession_id
        / contract["supersession_contract_sha256"]
    )
    return PreparedConfirm512Supersession(
        request=request,
        contract=contract,
        contract_sha256=contract["supersession_contract_sha256"],
        namespace_root=namespace,
        source_automatic=automatic,
        evaluator_results=evaluator_results,
        arms=tuple(arms),
    )


def finalize_report_only_selection(
    prepared: PreparedConfirm512Supersession,
) -> Mapping[str, Any]:
    """Lock one winner only after both immutable 512-sample reviews validate."""
    awaiting = materialize_visual_stage(prepared)
    automatic, _ = _bind_digest(
        prepared.namespace_root / "confirm512" / "automatic_evidence_v2.json",
        digest_field="automatic_evidence_sha256",
        contract_type="safa_r9_confirm512_report_only_automatic_v2",
    )
    review_rows = []
    missing = []
    for unit in automatic["visual_units"]:
        review_path = Path(unit["review_path"])
        evidence_path = Path(unit["evidence_path"])
        if not review_path.is_file():
            missing.append(str(review_path))
            continue
        review = validate_visual_review(review_path, evidence_path)
        evidence, evidence_binding = _bind_digest(
            evidence_path,
            digest_field="evidence_contract_sha256",
            contract_type=None,
        )
        derived = derive_visual_arm_pass(
            review, evidence, severe_limit=EXPECTED_SAMPLE_COUNT
        )
        if derived["reviewed_sample_count"] != EXPECTED_SAMPLE_COUNT:
            raise Confirm512SupersessionError("visual review is not complete for 512")
        review_rows.append(
            {
                "arm_id": unit["arm_id"],
                "severe_count": derived["severe_count"],
                "review_sha256": review["review_sha256"],
                "review_file_sha256": _sha256_file(review_path),
                "evidence": evidence_binding,
            }
        )
    if missing:
        return {
            "status": "awaiting_visual_review",
            "bounded_exit_code": AWAITING_VISUAL_REVIEW_EXIT_CODE,
            "missing_review_paths": missing,
            "awaiting_visual_review_sha256": awaiting["awaiting_visual_review_sha256"],
        }
    if len(review_rows) != 2:
        raise Confirm512SupersessionError("v2 requires exactly two complete reviews")
    arm_by_id = {str(row["arm_id"]): row for row in prepared.arms}
    ranked = sorted(
        review_rows,
        key=lambda row: _post_hoc_rank(
            arm_by_id[str(row["arm_id"])], int(row["severe_count"])
        ),
    )
    winner_review = ranked[0]
    winner = arm_by_id[str(winner_review["arm_id"])]
    gate = {
        "schema_version": 2,
        "contract_type": "safa_r9_confirm512_report_only_gate_v2",
        "campaign_id": prepared.request.campaign_id,
        "supersession_contract_sha256": prepared.contract_sha256,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "policy": prepared.contract["policy"],
        "evaluated": [
            {
                "arm_id": row["arm_id"],
                "severe_count": row["severe_count"],
                "review_sha256": row["review_sha256"],
                "evidence_contract_sha256": row["evidence"]["contract_sha256"],
                "complete_case_privacy_bootstrap_sha256": arm_by_id[str(row["arm_id"])][
                    "complete_case_privacy"
                ]["bootstrap"]["bootstrap_sha256"],
                "rank": list(
                    _post_hoc_rank(
                        arm_by_id[str(row["arm_id"])], int(row["severe_count"])
                    )
                ),
            }
            for row in ranked
        ],
        "selected_arm_ids": [winner["arm_id"]],
        "verdict": "winner_locked_report_only",
    }
    gate["gate_contract_sha256"] = _canonical_digest(gate, "gate_contract_sha256")
    phase_root = prepared.namespace_root / "confirm512"
    write_immutable_contract(
        phase_root / "gate_contract_v2.json",
        gate,
        digest_field="gate_contract_sha256",
    )
    selection = {
        "schema_version": 2,
        "contract_type": "safa_r9_confirm512_report_only_selection_v2",
        "campaign_id": prepared.request.campaign_id,
        "supersession_contract_sha256": prepared.contract_sha256,
        "source_repair_sha256": SOURCE_REPAIR_SHA256,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "gate_contract_sha256": gate["gate_contract_sha256"],
        "generation_inventory_sha256": prepared.contract["generation_inventory_sha256"],
        "manifest_sha256": prepared.source.phase_request.manifest_sha256,
        "visual_reviews": review_rows,
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
        "next_stage": "new_v9_full_continuation_required",
        "reselection_allowed": False,
    }
    selection["selection_sha256"] = _canonical_digest(selection, "selection_sha256")
    write_immutable_contract(
        prepared.namespace_root / "selection.json",
        selection,
        digest_field="selection_sha256",
    )
    result = {
        "supersession_contract_sha256": prepared.contract_sha256,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "gate_contract_sha256": gate["gate_contract_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "winner_arm_id": winner["arm_id"],
        "verdict": "winner_locked_report_only",
        "generation_execution_count": 0,
        "evaluator_execution_count": 0,
    }
    result["supersession_result_sha256"] = _canonical_digest(
        result, "supersession_result_sha256"
    )
    write_immutable_contract(
        prepared.namespace_root / "supersession_result.json",
        result,
        digest_field="supersession_result_sha256",
    )
    return result
