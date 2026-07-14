#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import run_r9_meanflow_campaign as driver
from safa.evaluation.r9_campaign_contracts import write_immutable_contract
from safa.evaluation.r9_phase_results import (
    AWAITING_VISUAL_REVIEW_EXIT_CODE,
    EVALUATION_REPAIR_FILENAME,
    EVALUATION_REPAIR_V2_FILENAME,
    EVALUATION_REPAIR_V3_FILENAME,
    PhaseResultsError,
    PhaseResultsRequest,
    evaluation_attempt_inventory,
    evaluation_repair_binding,
    generation_evidence_inventory,
    materialize_phase_results,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_RESULTS_PATH = Path("src/safa/evaluation/r9_phase_results.py")
DRIVER_PATH = Path("scripts/run_r9_meanflow_campaign.py")
WORKER_PATH = Path("src/safa/evaluation/r9_evaluator_worker.py")
QUALITY_PATH = Path("scripts/eval_generation_quality.py")
EXPECTED_V6_GENERATION_COUNTS = {
    "logical_run_count": 12,
    "shard_count": 12,
    "completion_count": 12,
    "generation_result_count": 12,
    "file_count": 1440,
    "png_count": 1344,
}


class EvaluationRepairError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--phase", choices=("calibrate",), required=True)
    parser.add_argument("--failed-unit-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prior-phase-results-sha256", required=True)
    parser.add_argument("--supersedes-repair-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute or not args.allow_busy_gpus:
        parser.error("evaluation repair requires --execute --allow-busy-gpus")
    return args


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationRepairError(f"{label} is not a SHA256")
    return value


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationRepairError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise EvaluationRepairError(f"{label} is not an object")
    return value


def _contract_binding(
    path: Path, *, digest_field: str, contract_type: str | None
) -> dict[str, str]:
    label = contract_type or digest_field
    payload = _read_mapping(path, label)
    if contract_type is not None and payload.get("contract_type") != contract_type:
        raise EvaluationRepairError(f"{contract_type} type mismatch")
    declared = _require_sha256(payload.get(digest_field), digest_field)
    canonical = dict(payload)
    canonical.pop(digest_field)
    if driver._canonical_json_sha256(canonical) != declared:
        raise EvaluationRepairError(f"{label} canonical digest mismatch")
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "file_sha256": _sha256(path),
        "contract_sha256": declared,
    }


def _implementation_binding(path: Path) -> dict[str, str]:
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise EvaluationRepairError(f"repair implementation is invalid: {path}")
    return {"path": str(path), "sha256": _sha256(resolved)}


def _source_phase_results_sha256(source_commit: str) -> str:
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise EvaluationRepairError("source commit must be a full lowercase SHA1")
    completed = subprocess.run(
        ["git", "show", f"{source_commit}:{PHASE_RESULTS_PATH.as_posix()}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise EvaluationRepairError("source commit does not contain phase results")
    return hashlib.sha256(completed.stdout).hexdigest()


def _load_campaign(
    campaign_id: str, phase: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    driver.PhasePlan,
    PhaseResultsRequest,
]:
    runtime, runtime_path, source = driver.load_campaign_configuration(campaign_id)
    effective, manifests, diagnose = driver.build_effective_campaign_runtime(
        runtime,
        campaign_id=campaign_id,
        repo_root=REPO_ROOT,
        runtime_config_path=runtime_path,
        continuation_source=source,
    )
    campaign_root = REPO_ROOT / str(effective["campaign_root"])
    stored = _read_mapping(campaign_root / "campaign_runtime.json", "campaign runtime")
    if effective != stored:
        raise EvaluationRepairError("live campaign runtime changed before repair")
    promoted, winner = driver.resolve_phase_promotion(
        runtime, effective, phase=phase, campaign_id=campaign_id
    )
    plan = driver.build_phase_plan(
        runtime,
        phase=phase,
        campaign_id=campaign_id,
        promoted_arm_ids=promoted,
        winner_arm_id=winner,
    )
    request = driver.build_phase_results_request(
        runtime,
        effective,
        manifests,
        diagnose,
        plan=plan,
        campaign_id=campaign_id,
    )
    return runtime, effective, manifests, diagnose, plan, request


def _assert_generation_terminal(
    request: PhaseResultsRequest, plan: driver.PhasePlan
) -> None:
    expected = {
        f"{run.phase}:{run.logical_run_id}:shard-{run.shard_index}" for run in plan.runs
    }
    status_root = request.phase_root.parent / "worker_status"
    states = {}
    for path in status_root.glob("*.json"):
        payload = _read_mapping(path, "worker status")
        worker_id = payload.get("worker_id")
        if worker_id in expected:
            states[str(worker_id)] = payload.get("state")
    if states != {worker_id: "succeeded" for worker_id in expected}:
        raise EvaluationRepairError("generation workers are not exactly terminal-success")
    for run in plan.runs:
        driver.validate_worker_completion(run)


def _assert_v6_generation_inventory(inventory: Mapping[str, Any]) -> None:
    observed = {
        field: inventory.get(field) for field in EXPECTED_V6_GENERATION_COUNTS
    }
    if observed != EXPECTED_V6_GENERATION_COUNTS:
        raise EvaluationRepairError("frozen v6 generation inventory counts changed")


def _build_repair_contract(
    request: PhaseResultsRequest,
    effective: Mapping[str, Any],
    *,
    failed_unit_id: str,
    source_commit: str,
    prior_phase_results_sha256: str,
    supersedes_repair_sha256: str | None,
) -> dict[str, Any]:
    prior = _require_sha256(
        prior_phase_results_sha256, "prior phase-results SHA256"
    )
    if _source_phase_results_sha256(source_commit) != prior:
        raise EvaluationRepairError("source commit phase-results SHA256 mismatch")
    current_phase = _implementation_binding(PHASE_RESULTS_PATH)
    if current_phase["sha256"] == prior:
        raise EvaluationRepairError("repair did not change phase-results implementation")
    failed_root = (
        request.phase_root / "evaluator_runs" / "quality" / failed_unit_id
    )
    request_binding = _contract_binding(
        failed_root / "request.json",
        digest_field="evaluator_request_sha256",
        contract_type="safa_r9_phase_evaluator_request_v1",
    )
    result_binding = _contract_binding(
        failed_root / "result.json",
        digest_field="evaluator_output_sha256",
        contract_type="safa_r9_phase_evaluator_output_v1",
    )
    failed_request = _read_mapping(failed_root / "request.json", "failed request")
    failed_result = _read_mapping(failed_root / "result.json", "failed result")
    raw_binding = failed_result.get("result", {}).get("r9_evidence_binding")
    request_payload = failed_request.get("payload")
    if (
        not isinstance(raw_binding, Mapping)
        or "source_index_path" in raw_binding
        or not isinstance(request_payload, Mapping)
        or not isinstance(request_payload.get("source_index_path"), str)
        or raw_binding.get("source_index_sha256")
        != request_payload.get("source_index_sha256")
        or failed_result.get("evaluator_request_sha256")
        != request_binding["contract_sha256"]
    ):
        raise EvaluationRepairError("failed evaluator evidence is not the registered mismatch")
    runtime_path = request.phase_root.parent / "campaign_runtime.json"
    runtime_binding = _contract_binding(
        runtime_path,
        digest_field="campaign_runtime_sha256",
        contract_type=None,
    )
    if runtime_binding["contract_sha256"] != request.campaign_runtime_sha256:
        raise EvaluationRepairError("repair runtime differs from phase request")
    continuation_path = request.phase_root.parent / "continuation_contract.json"
    continuation = _read_mapping(continuation_path, "continuation contract")
    locked = continuation.get("bindings", {}).get("implementations", {})
    expected_locked = {
        "driver": _implementation_binding(DRIVER_PATH),
        "evaluator_worker": _implementation_binding(WORKER_PATH),
        "quality": _implementation_binding(QUALITY_PATH),
    }
    for name, binding in expected_locked.items():
        locked_name = {
            "driver": "driver",
            "evaluator_worker": "evaluator_implementation",
            "quality": "quality",
        }[name]
        if locked.get(locked_name) != binding:
            raise EvaluationRepairError(f"v6 locked {name} implementation changed")
    generation_inventory = generation_evidence_inventory(request)
    supersedes = None
    superseded_version = None
    if supersedes_repair_sha256 is not None:
        supersedes, superseded_version = _build_supersedes_binding(
            request,
            supersedes_repair_sha256=supersedes_repair_sha256,
            prior_phase_results_sha256=prior,
            generation_inventory=generation_inventory,
        )
    contract = {
        "schema_version": 1,
        "contract_type": {
            None: "safa_r9_evaluation_repair_contract_v1",
            1: "safa_r9_evaluation_repair_contract_v2",
            2: "safa_r9_evaluation_repair_contract_v3",
        }[superseded_version],
        "campaign_id": request.campaign_id,
        "phase": request.phase,
        "campaign_runtime": runtime_binding,
        "generation_evidence": generation_inventory,
        "failed_evaluation": {
            "evaluator": "quality",
            "unit_id": failed_unit_id,
            "request": request_binding,
            "result": result_binding,
            "mismatch": {
                "field": "r9_evidence_binding.source_index_path",
                "classification": "request_transport_field_in_raw_content_binding",
                "producer_has_field": False,
                "consumer_required_field": True,
            },
        },
        "implementations": {
            "source_git_commit": source_commit,
            "prior_phase_results_sha256": prior,
            "repaired_phase_results": current_phase,
            **expected_locked,
            "repair_runner": _implementation_binding(
                Path("scripts/run_r9_evaluation_repair.py")
            ),
        },
        "policy": {
            "generation_execution": "forbidden",
            "expected_generation_worker_count": 0,
            "old_failed_result_usage": "input_evidence_only",
            "old_attempt_retry_allowed": False,
            "evaluation_namespace": "evaluation_repairs/{repair_contract_sha256}",
            "request_binding": "full_repair_sha256_in_logical_run_id",
        },
    }
    if supersedes is not None:
        contract["supersedes"] = supersedes
    return contract


def _build_supersedes_binding(
    request: PhaseResultsRequest,
    *,
    supersedes_repair_sha256: str,
    prior_phase_results_sha256: str,
    generation_inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    expected_sha256 = _require_sha256(
        supersedes_repair_sha256, "superseded repair SHA256"
    )
    v2_path = request.phase_root / EVALUATION_REPAIR_V2_FILENAME
    if v2_path.exists():
        prior_path = v2_path
        prior_contract_type = "safa_r9_evaluation_repair_contract_v2"
        prior_version = 2
        controller_log = request.phase_root / "evaluation_repair_v2_controller.log"
        expected_failure = {
            "exception_type": "CampaignContractError",
            "symbol": "raw_evidence_namespace",
            "message": "immutable contract already exists with other content",
        }
        expected_counts = {
            "file_count": 4,
            "result_count": 1,
            "quality_result_count": 1,
            "arcface_result_count": 0,
        }
    else:
        prior_path = request.phase_root / EVALUATION_REPAIR_FILENAME
        prior_contract_type = "safa_r9_evaluation_repair_contract_v1"
        prior_version = 1
        controller_log = request.phase_root / "evaluation_repair_controller.log"
        expected_failure = {
            "exception_type": "NameError",
            "symbol": "manifest_ids",
            "message": "NameError: name 'manifest_ids' is not defined",
        }
        expected_counts = {
            "file_count": 8,
            "result_count": 2,
            "quality_result_count": 2,
            "arcface_result_count": 0,
        }
    prior_binding = _contract_binding(
        prior_path,
        digest_field="repair_contract_sha256",
        contract_type=prior_contract_type,
    )
    if prior_binding["contract_sha256"] != expected_sha256:
        raise EvaluationRepairError("superseded repair SHA256 mismatch")
    prior = _read_mapping(prior_path, "superseded repair")
    prior_implementations = prior.get("implementations")
    if (
        prior.get("campaign_id") != request.campaign_id
        or prior.get("phase") != request.phase
        or prior.get("generation_evidence") != generation_inventory
        or not isinstance(prior_implementations, Mapping)
        or not isinstance(prior_implementations.get("repaired_phase_results"), Mapping)
        or prior_implementations["repaired_phase_results"].get("sha256")
        != prior_phase_results_sha256
    ):
        raise EvaluationRepairError("superseded repair binding changed")
    if prior.get("policy") != {
        "generation_execution": "forbidden",
        "expected_generation_worker_count": 0,
        "old_failed_result_usage": "input_evidence_only",
        "old_attempt_retry_allowed": False,
        "evaluation_namespace": "evaluation_repairs/{repair_contract_sha256}",
        "request_binding": "full_repair_sha256_in_logical_run_id",
    }:
        raise EvaluationRepairError("superseded repair policy changed")
    namespace_root = (
        request.phase_root.parent / "evaluation_repairs" / expected_sha256
    )
    attempt = evaluation_attempt_inventory(REPO_ROOT, namespace_root)
    if {
        "file_count": attempt["file_count"],
        "result_count": attempt["result_count"],
        "quality_result_count": attempt["quality_result_count"],
        "arcface_result_count": attempt["arcface_result_count"],
    } != expected_counts:
        raise EvaluationRepairError("superseded evaluation attempt counts changed")
    log_binding = _implementation_binding(controller_log.relative_to(REPO_ROOT))
    try:
        log_text = controller_log.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise EvaluationRepairError("superseded controller log is not UTF-8") from error
    if expected_failure["message"] not in log_text:
        raise EvaluationRepairError("superseded controller log lacks registered error")
    return {
        "repair": prior_binding,
        "failure": {
            "exception_type": expected_failure["exception_type"],
            "symbol": expected_failure["symbol"],
            "controller_log": log_binding,
        },
        "evaluation_attempt": attempt,
        "policy": {
            "prior_repair_usage": "input_evidence_only",
            "prior_evaluation_results_usage": "input_evidence_only",
            "prior_namespace_reuse": False,
        },
    }, prior_version


def _repair_logical_run_id(repair_sha256: str, logical_run_id: str) -> str:
    _require_sha256(repair_sha256, "repair contract SHA256")
    return f"repair_{repair_sha256}__{logical_run_id}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if "TMUX" not in os.environ:
        raise EvaluationRepairError("evaluation repair must run inside tmux")
    runtime, effective, manifests, diagnose, plan, request = _load_campaign(
        str(args.campaign_id), str(args.phase)
    )
    _assert_generation_terminal(request, plan)
    before = generation_evidence_inventory(request)
    _assert_v6_generation_inventory(before)
    contract = _build_repair_contract(
        request,
        effective,
        failed_unit_id=str(args.failed_unit_id),
        source_commit=str(args.source_commit),
        prior_phase_results_sha256=str(args.prior_phase_results_sha256),
        supersedes_repair_sha256=(
            None
            if args.supersedes_repair_sha256 is None
            else str(args.supersedes_repair_sha256)
        ),
    )
    contract["repair_contract_sha256"] = driver._canonical_json_sha256(contract)
    repair_path = request.phase_root / {
        "safa_r9_evaluation_repair_contract_v1": EVALUATION_REPAIR_FILENAME,
        "safa_r9_evaluation_repair_contract_v2": EVALUATION_REPAIR_V2_FILENAME,
        "safa_r9_evaluation_repair_contract_v3": EVALUATION_REPAIR_V3_FILENAME,
    }[str(contract["contract_type"])]
    write_immutable_contract(
        repair_path, contract, digest_field="repair_contract_sha256"
    )
    repair_sha256 = str(contract["repair_contract_sha256"])
    if evaluation_repair_binding(request) is None:
        raise AssertionError("materialized repair contract is not bound")
    repair_runtime = json.loads(json.dumps(effective))
    repair_runtime["campaign_root"] = str(
        Path(str(effective["campaign_root"]))
        / "evaluation_repairs"
        / repair_sha256
    )
    scheduler, gpu_bindings, status = driver.build_resource_scheduler(effective)
    callbacks = driver.R9ProductionEvaluatorCallbacks(
        runtime=runtime,
        campaign_runtime=repair_runtime,
        scheduler=scheduler,
        gpu_bindings=gpu_bindings,
        peer_status_store=status,
    )

    def quality(evaluation):
        return callbacks.quality(
            replace(
                evaluation,
                logical_run_id=_repair_logical_run_id(
                    repair_sha256, evaluation.logical_run_id
                ),
            )
        )

    def arcface(evaluation):
        return callbacks.arcface(
            replace(
                evaluation,
                logical_run_id=_repair_logical_run_id(
                    repair_sha256, evaluation.logical_run_id
                ),
            )
        )

    closure = materialize_phase_results(
        request,
        quality_evaluator=quality,
        arcface_evaluator=arcface,
    )
    after = generation_evidence_inventory(request)
    if after != before:
        raise PhaseResultsError("evaluation repair changed frozen generation evidence")
    _assert_v6_generation_inventory(after)
    _assert_generation_terminal(request, plan)
    print(
        json.dumps(
            {
                "repair_contract_sha256": repair_sha256,
                "generation_evidence": after,
                "closure_status": closure.status,
                "required_review_count": closure.required_review_count,
                "completed_review_count": closure.completed_review_count,
            },
            sort_keys=True,
        )
    )
    if closure.status == "awaiting_visual_review":
        return AWAITING_VISUAL_REVIEW_EXIT_CODE
    if closure.status != "complete":
        raise EvaluationRepairError("evaluation repair did not close the phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
