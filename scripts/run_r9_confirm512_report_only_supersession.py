#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import run_r9_confirm512_canonical_repair as source_runner
from safa.evaluation.r9_confirm512_canonical_repair import (
    CanonicalNativeRepairRequest,
    build_canonical_native_repair,
)
from safa.evaluation.r9_confirm512_report_only_supersession import (
    AWAITING_VISUAL_REVIEW_EXIT_CODE,
    Confirm512SupersessionError,
    Confirm512SupersessionRequest,
    FAILED_V2_CONTRACT_SHA256,
    SOURCE_REPAIR_SHA256,
    build_report_only_supersession,
    finalize_report_only_selection,
    materialize_visual_stage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--source-repair-id", required=True)
    parser.add_argument("--source-repair-sha256", required=True)
    parser.add_argument("--failed-v2-sha256", required=True)
    parser.add_argument("--supersession-id", required=True)
    parser.add_argument("--phase", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _build(args: argparse.Namespace):
    if str(args.source_repair_sha256) != SOURCE_REPAIR_SHA256:
        raise Confirm512SupersessionError("source repair SHA256 is not formal v1")
    if str(args.failed_v2_sha256) != FAILED_V2_CONTRACT_SHA256:
        raise Confirm512SupersessionError("failed v2 SHA256 is not frozen")
    _, effective, plan, phase_request = source_runner._load_campaign(
        str(args.campaign_id)
    )
    source_runner._assert_generation_terminal(plan)
    campaign_root = REPO_ROOT / str(effective["campaign_root"])
    source_request = CanonicalNativeRepairRequest(
        repo_root=REPO_ROOT,
        source_campaign_root=campaign_root,
        repair_root=campaign_root / "canonical_native_repairs",
        repair_id=str(args.source_repair_id),
        campaign_id=str(args.campaign_id),
        source_failure_sha256=(
            "db27dc5588fd42955890ca8ec249070cc1051728f17debaaf5d95541edc85ebb"
        ),
        phase_request=phase_request,
    )
    source = build_canonical_native_repair(source_request)
    source_namespace = (
        campaign_root
        / "canonical_native_repairs"
        / str(args.source_repair_id)
        / str(args.source_repair_sha256)
    )
    if source.namespace_root.resolve() != source_namespace.resolve():
        raise Confirm512SupersessionError("reconstructed source namespace changed")
    return build_report_only_supersession(
        Confirm512SupersessionRequest(
            failed_v2_namespace_root=(
                campaign_root
                / "confirm512_supersessions"
                / "report-only-v2"
                / str(args.failed_v2_sha256)
            ),
            repo_root=REPO_ROOT,
            campaign_root=campaign_root,
            source_namespace_root=source_namespace,
            supersession_root=campaign_root / "confirm512_supersessions",
            supersession_id=str(args.supersession_id),
            campaign_id=str(args.campaign_id),
            source=source,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = _build(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_validated",
                    "phase": args.phase,
                    "supersession_contract_sha256": prepared.contract_sha256,
                    "namespace_root": str(prepared.namespace_root),
                    "source_repair_sha256": SOURCE_REPAIR_SHA256,
                    "generation_inventory_sha256": prepared.contract[
                        "generation_inventory_sha256"
                    ],
                    "bound_evaluator_result_count": len(
                        prepared.contract["evaluator_results"]
                    ),
                    "planned_visual_unit_count": 2,
                    "generation_execution_count": 0,
                    "evaluator_execution_count": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.phase == "prepare":
        outcome = materialize_visual_stage(prepared)
    else:
        outcome = finalize_report_only_selection(prepared)
    print(
        json.dumps(
            {
                "status": (
                    outcome["status"] if "status" in outcome else outcome["verdict"]
                ),
                "supersession_contract_sha256": prepared.contract_sha256,
                "namespace_root": str(prepared.namespace_root),
                "generation_execution_count": 0,
                "evaluator_execution_count": 0,
                "winner_arm_id": outcome.get("winner_arm_id"),
                "selection_sha256": outcome.get("selection_sha256"),
            },
            sort_keys=True,
        )
    )
    if outcome.get("status") == "awaiting_visual_review":
        return AWAITING_VISUAL_REVIEW_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
