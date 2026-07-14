#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_r9_meanflow_campaign as driver
from safa.evaluation.r9_confirm512_canonical_repair import (
    CanonicalNativeRepairError,
    CanonicalNativeRepairRequest,
    build_canonical_native_repair,
    execute_canonical_evaluations,
    materialize_repair_contract,
    validate_canonical_native_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--repair-id", required=True)
    parser.add_argument("--source-failure-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute or not args.allow_busy_gpus:
        parser.error("canonical-native repair requires --execute --allow-busy-gpus")
    return args


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalNativeRepairError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise CanonicalNativeRepairError(f"{label} is not an object")
    return value


def _load_campaign(
    campaign_id: str,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    driver.PhasePlan,
    Any,
]:
    runtime, _, _ = driver.load_campaign_configuration(campaign_id)
    campaign_root = REPO_ROOT / str(runtime["campaign_root"]) / campaign_id
    stored = _read_mapping(campaign_root / "campaign_runtime.json", "campaign runtime")
    declared = stored.get("campaign_runtime_sha256")
    canonical = dict(stored)
    canonical.pop("campaign_runtime_sha256", None)
    if (
        stored.get("campaign_id") != campaign_id
        or not isinstance(declared, str)
        or driver._canonical_json_sha256(canonical) != declared
        or (REPO_ROOT / str(stored.get("campaign_root"))).resolve()
        != campaign_root.resolve()
    ):
        raise CanonicalNativeRepairError(
            "frozen v8 campaign runtime binding is invalid"
        )
    effective = stored
    manifests = {
        "manifest_contracts_sha256": stored["manifest_contracts_sha256"],
        "manifests": stored["manifests"],
    }
    diagnose: dict[str, Any] = {}
    promoted, winner = driver.resolve_phase_promotion(
        runtime,
        effective,
        phase="confirm512",
        campaign_id=campaign_id,
    )
    if winner is not None:
        raise CanonicalNativeRepairError("failed v8 C campaign already has a winner")
    plan = driver.build_phase_plan(
        runtime,
        phase="confirm512",
        campaign_id=campaign_id,
        promoted_arm_ids=promoted,
        winner_arm_id=None,
    )
    phase_request = driver.build_phase_results_request(
        runtime,
        effective,
        manifests,
        diagnose,
        plan=plan,
        campaign_id=campaign_id,
    )
    return runtime, effective, plan, phase_request


def _assert_generation_terminal(plan: driver.PhasePlan) -> None:
    if plan.phase != "confirm512" or len(plan.runs) != 48:
        raise CanonicalNativeRepairError("repair requires the exact 48-run C plan")
    for run in plan.runs:
        driver.validate_worker_completion(run)


def _inventory_for(
    campaign_root: Path,
    phase_request: Any,
) -> Mapping[str, Any]:
    roots = tuple(
        output for spec in phase_request.runs for output in spec.shard_output_dirs
    )
    return validate_canonical_native_inventory(
        campaign_root=campaign_root,
        expected_roots=roots,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if "TMUX" not in os.environ:
        raise CanonicalNativeRepairError(
            "canonical-native repair must execute inside tmux"
        )
    runtime, effective, plan, phase_request = _load_campaign(str(args.campaign_id))
    _assert_generation_terminal(plan)
    campaign_root = REPO_ROOT / str(effective["campaign_root"])
    before = _inventory_for(campaign_root, phase_request)
    repair_base = campaign_root / "canonical_native_repairs"
    request = CanonicalNativeRepairRequest(
        repo_root=REPO_ROOT,
        source_campaign_root=campaign_root,
        repair_root=repair_base,
        repair_id=str(args.repair_id),
        campaign_id=str(args.campaign_id),
        source_failure_sha256=str(args.source_failure_sha256),
        phase_request=phase_request,
    )
    prepared = build_canonical_native_repair(request)
    contract_path = materialize_repair_contract(prepared)
    repair_runtime = json.loads(json.dumps(effective))
    repair_runtime["campaign_root"] = str(prepared.namespace_root)
    scheduler, gpu_bindings, status = driver.build_resource_scheduler(effective)
    callbacks = driver.R9ProductionEvaluatorCallbacks(
        runtime=runtime,
        campaign_runtime=repair_runtime,
        scheduler=scheduler,
        gpu_bindings=gpu_bindings,
        peer_status_store=status,
    )
    result = execute_canonical_evaluations(
        prepared,
        quality_evaluator=callbacks.quality,
        arcface_evaluator=callbacks.arcface,
    )
    after = _inventory_for(campaign_root, phase_request)
    if before != after:
        raise CanonicalNativeRepairError(
            "canonical-native repair changed frozen generation evidence"
        )
    if prepared.contract["canonical_native_policy"]["generation_execution_count"] != 0:
        raise CanonicalNativeRepairError("repair contract did not forbid generation")
    print(
        json.dumps(
            {
                "repair_contract": str(contract_path),
                "repair_contract_sha256": prepared.contract_sha256,
                "namespace_root": str(prepared.namespace_root),
                "generation_inventory_sha256": after["inventory_sha256"],
                "generation_execution_count": 0,
                "evaluator_unit_count": result["evaluator_unit_count"],
                "winner_arm_id": result["winner_arm_id"],
                "selection_sha256": result["selection_sha256"],
                "status": result["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["winner_arm_id"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
