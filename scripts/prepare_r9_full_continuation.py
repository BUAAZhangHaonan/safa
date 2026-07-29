#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence

import yaml

from safa.evaluation.r9_full_continuation_contracts import (
    full_continuation_contract_binding,
    full_selection_binding,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO_ROOT / "scripts/run_r9_meanflow_campaign.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=(
            "evaluator-smoke-requests",
            "e2e-prepare",
            "e2e-run",
            "e2e-finalize",
            "e2e-monitor",
            "formal-monitor",
            "validate",
        ),
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _driver():
    spec = importlib.util.spec_from_file_location("r9_campaign_driver", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load R9 campaign driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    driver = _driver()
    if args.phase == "evaluator-smoke-requests":
        runtime, request_path, source = driver.load_full_continuation_request(
            allow_pre_e2e_profiles=True
        )
        if runtime["evaluation"]["resource_smokes"] != {
            "profile_state": "pre_e2e_not_execution_authority"
        }:
            raise ValueError(
                "historical smoke validation is only valid before E2E profiles"
            )
        contract = driver.build_full_continuation_contract(
            repo_root=driver.REPO_ROOT, expected_source=source
        )
        historical = contract["bindings"]["evaluator_smoke_requests"]
        print(
            json.dumps(
                {
                    "status": (
                        "validated_historical_provenance"
                        if args.execute
                        else "dry_run_validated"
                    ),
                    "campaign_id": driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
                    "runtime_config": str(request_path),
                    "classification": historical["classification"],
                    "request_set_sha256": historical["request_set_sha256"],
                    "artifact_write_count": 0,
                    "gpu_execution_count": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.phase == "e2e-prepare":
        result = _prepare_e2e(driver, materialize=args.execute)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase == "e2e-run":
        if not args.execute:
            raise RuntimeError("e2e-run requires --execute")
        result = _run_e2e(driver)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase == "e2e-finalize":
        if not args.execute:
            raise RuntimeError("e2e-finalize requires --execute")
        result = _finalize_e2e(driver)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase == "e2e-monitor":
        if not args.execute:
            raise RuntimeError("e2e-monitor requires --execute")
        return _monitor_e2e(driver)
    if args.phase == "formal-monitor":
        if not args.execute:
            raise RuntimeError("formal-monitor requires --execute")
        return _monitor_formal_full(driver)
    root = _e2e_root(driver)
    profile_exists = (root / "resource_profiles.json").is_file()
    gate_exists = (root / "gate_contract.json").is_file()
    if profile_exists != gate_exists:
        raise RuntimeError(
            "Full E2E profile/gate materialization is incomplete"
        )
    if not profile_exists:
        runtime, request_path, source = driver.load_full_continuation_request(
            allow_pre_e2e_profiles=True
        )
        contract = driver.build_full_continuation_contract(
            repo_root=driver.REPO_ROOT, expected_source=source
        )
        provisional = _build_provisional_e2e_runtime(driver)
        manifest = _e2e_manifest(driver, provisional)
        print(
            json.dumps(
                {
                    "status": "valid_pre_e2e_blocked_full",
                    "campaign_id": driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
                    "runtime_config": str(request_path),
                    "full_continuation_sha256": contract[
                        "full_continuation_sha256"
                    ],
                    "provisional_runtime_sha256": provisional[
                        "campaign_runtime_sha256"
                    ],
                    "manifest_sha256": manifest["sha256"],
                    "profile_state": runtime["evaluation"][
                        "resource_smokes"
                    ]["profile_state"],
                    "artifact_write_count": 0,
                    "gpu_execution_count": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    runtime, request_path, source = driver.load_full_continuation_request()
    contract = driver.build_full_continuation_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    print(
        json.dumps(
            {
                "status": "validated",
                "campaign_id": driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
                "runtime_config": str(request_path),
                "full_continuation_sha256": contract[
                    "full_continuation_sha256"
                ],
                "batch_policy": contract["bindings"]["generation_batch_policy"],
                "evaluator_smoke_request_set_sha256": contract["bindings"][
                    "evaluator_smoke_requests"
                ]["request_set_sha256"],
                "gpu_execution_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _e2e_root(driver) -> Path:
    return (
        driver.REPO_ROOT
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        / "full_e2e"
    )


def _provisional_runtime_path(driver) -> Path:
    return _e2e_root(driver) / "provisional_runtime.json"


def _load_e2e_request(driver) -> dict:
    path = driver.REPO_ROOT / (
        "configs/medium_v2/experiments/r9_full_e2e_v1.yaml"
    )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Full E2E request must be a mapping")
    return value


def _build_provisional_e2e_runtime(driver) -> dict:
    current_runtime, _, source = driver.load_full_continuation_request(
        allow_pre_e2e_profiles=True
    )
    continuation = driver.build_full_continuation_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    selection = driver.build_full_continuation_selection_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    source_path = driver.REPO_ROOT / (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        "r9-report-only-formal-v8/campaign_runtime.json"
    )
    source_runtime = driver._read_json_mapping(
        source_path, "v8 source runtime"
    )
    source_canonical = dict(source_runtime)
    source_digest = source_canonical.pop("campaign_runtime_sha256", None)
    if driver._canonical_json_sha256(source_canonical) != source_digest:
        raise ValueError("v8 source runtime digest changed")
    request = _load_e2e_request(driver)
    policy = request["resource_policy"]
    generation = policy["generation"]
    evaluator = policy["evaluator"]
    provisional = dict(source_runtime)
    provisional["campaign_id"] = driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID
    provisional["campaign_root"] = (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        f"{driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID}"
    )
    continuation_path, continuation_bytes, continuation_binding = (
        full_continuation_contract_binding(
            continuation, repo_root=driver.REPO_ROOT
        )
    )
    del continuation_path, continuation_bytes
    provisional["continuation"] = continuation_binding
    evaluation = dict(current_runtime["evaluation"])
    current_binding = continuation["bindings"]["current_evaluation"]
    if (
        evaluation["worker"] != current_binding["worker"]
        or evaluation["quality"]["script"]
        != current_binding["quality_script"]
    ):
        raise ValueError(
            "Full E2E provisional runtime does not bind current evaluator bytes"
        )
    evaluation["resource_smokes"] = {
        "arcface": {
            "mode": "frozen_conservative_e2e_claim_v1",
            "ram_slot_budget_bytes": evaluator["ram_slot_budget_bytes"],
        },
        "quality": {
            "mode": "frozen_conservative_e2e_claim_v1",
            "ram_slot_budget_bytes": evaluator["ram_slot_budget_bytes"],
        },
        "heldout": {
            "mode": "exclusive_single_official_run",
            "smoke_execution": "sealed_until_winner_lock",
            "global_exclusive_slots": 16,
            "ram_admission_percent": 85,
            "ram_hard_limit_percent": 90,
        },
    }
    provisional["evaluation"] = evaluation
    resources = dict(source_runtime["resources"])
    resources.update(
        {
            "physical_gpus": list(policy["gpu_indices"]),
            "max_slots_per_gpu": generation["max_slots_per_gpu"],
            "generation_slots_per_gpu": generation["max_slots_per_gpu"],
            "gpu_slot_claim_bytes": generation["gpu_slot_claim_bytes"],
            "ram_slot_budget_bytes": generation["ram_slot_budget_bytes"],
            "generation_batch_size": 2,
        }
    )
    provisional["resources"] = resources
    provisional["full_e2e_bootstrap"] = {
        "contract_type": "safa_r9_full_e2e_provisional_runtime_v1",
        "selection_sha256": selection["selection_sha256"],
        "resource_policy": policy,
    }
    provisional.pop("campaign_runtime_sha256", None)
    provisional["campaign_runtime_sha256"] = driver._canonical_json_sha256(
        provisional
    )
    return provisional


def _load_effective_runtime(driver, *, allow_provisional: bool = False) -> dict:
    path = (
        driver.REPO_ROOT
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        / "campaign_runtime.json"
    )
    if path.is_file():
        value = driver._read_json_mapping(path, "v9 campaign runtime")
        canonical = dict(value)
        declared = canonical.pop("campaign_runtime_sha256", None)
        continuation = driver._continuation_for_runtime(value)
        validated = driver.validate_campaign_runtime(
            canonical,
            driver.REPO_ROOT,
            continuation_contract=continuation,
        )
        if (
            value.get("campaign_runtime_sha256") != declared
            or validated != value
        ):
            raise ValueError("v9 materialized runtime changed")
        return value
    if not allow_provisional:
        raise FileNotFoundError(f"v9 campaign runtime is missing: {path}")
    provisional = _build_provisional_e2e_runtime(driver)
    provisional_path = _provisional_runtime_path(driver)
    if provisional_path.is_file():
        observed = driver._read_json_mapping(
            provisional_path, "Full E2E provisional runtime"
        )
        if observed != provisional:
            raise ValueError("Full E2E provisional runtime changed")
    return provisional


def _e2e_manifest(driver, campaign_runtime: dict) -> dict:
    path = Path(
        "configs/medium_v2/experiments/r9_manifests/full_smoke_8.jsonl"
    )
    absolute = driver.REPO_ROOT / path
    sha256 = _file_sha256(absolute)
    if sha256 != "04a7d89db541b065755c965505bb26b1e58aea306cc59c1717f251ec32dfc87f":
        raise ValueError("Full E2E manifest SHA256 changed")
    parent = campaign_runtime["manifests"]["full_2048"]
    parent_ids = _manifest_ids(driver.REPO_ROOT / parent["path"])
    sample_ids = _manifest_ids(absolute)
    if sample_ids != parent_ids[:8]:
        raise ValueError("Full E2E IDs are not the frozen first 8 full_2048 IDs")
    return {
        "path": str(path),
        "sha256": sha256,
        "sample_count": 8,
        "parent_path": parent["path"],
        "parent_sha256": parent["sha256"],
        "sample_ids": sample_ids,
    }


def _prepare_e2e(driver, *, materialize: bool) -> dict:
    runtime, request_path, source = driver.load_full_continuation_request(
        allow_pre_e2e_profiles=True
    )
    if runtime["evaluation"]["resource_smokes"] != {
        "profile_state": "pre_e2e_not_execution_authority"
    }:
        raise ValueError(
            "Full E2E prepare expected the pre-E2E non-authoritative profile state"
        )
    campaign_runtime = _load_effective_runtime(driver, allow_provisional=True)
    continuation = driver.build_full_continuation_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    selection = driver.build_full_continuation_selection_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    manifest = _e2e_manifest(driver, campaign_runtime)
    root = _e2e_root(driver)
    original_manifest_contract = {
        "manifest_contracts_sha256": campaign_runtime[
            "manifest_contracts_sha256"
        ],
        "manifests": campaign_runtime["manifests"],
    }
    base_plan = driver.build_phase_plan(
        runtime,
        phase="full",
        campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        winner_arm_id="paper_eta_0p125",
    )
    first_by_arm = {}
    for run in base_plan.runs:
        first_by_arm.setdefault(run.arm_ref, run)
    records = []
    config_writes: list[tuple[Path, bytes]] = []
    for arm_id in ("native", "paper_eta_0p125"):
        source_run = first_by_arm[arm_id]
        config_path = root / "runtime_configs" / f"{arm_id}.yaml"
        output_dir = root / "generation" / arm_id
        run = driver.RunSpec(
            phase="full",
            logical_run_id=f"formal_e2e_{arm_id}_8",
            arm_ref=arm_id,
            seed=7919,
            repeat_index=None,
            shard_index=0,
            num_shards=1,
            sample_count=8,
            manifest_key="full_2048",
            runtime_config=config_path.relative_to(driver.REPO_ROOT),
            output_dir=output_dir.relative_to(driver.REPO_ROOT),
            command=(
                str(runtime["python"]),
                str(runtime["generation_script"]),
                "--config",
                str(config_path.relative_to(driver.REPO_ROOT)),
                "--output-dir",
                str(output_dir.relative_to(driver.REPO_ROOT)),
                "--shard-index",
                "0",
                "--num-shards",
                "1",
            ),
        )
        del source_run
        config = driver.build_run_runtime_config(
            runtime,
            campaign_runtime,
            original_manifest_contract,
            run,
            continuation_contract=continuation,
        )
        config.update(
            {
                "experiment_name": f"full_e2e__{arm_id}",
                "out_dir": str(run.output_dir),
                "max_samples": 8,
                "sample_id_manifest": manifest["path"],
                "sample_id_manifest_sha256": manifest["sha256"],
                "r9_phase_manifest_sha256": manifest["sha256"],
                "r9_full_e2e_role": "formal_gate_v1",
            }
        )
        content = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
        config_writes.append((config_path, content))
        records.append(
            {
                "arm_id": arm_id,
                "runtime_config": str(config_path.relative_to(driver.REPO_ROOT)),
                "runtime_config_sha256": hashlib.sha256(content).hexdigest(),
                "output_dir": str(output_dir.relative_to(driver.REPO_ROOT)),
                "command": list(run.command),
            }
        )
    plan = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_plan_v1",
        "campaign_id": driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        "continuation_contract_sha256": driver._continuation_digest(continuation),
        "selection_sha256": selection["selection_sha256"],
        "request_config": {
            "path": str(request_path),
            "sha256": _file_sha256(driver.REPO_ROOT / request_path),
        },
        "e2e_request": continuation["bindings"]["full_e2e_requirement"][
            "request"
        ],
        "generation_batch_benchmark": campaign_runtime[
            "generation_batch_benchmark"
        ],
        "provisional_runtime": {
            "path": str(
                _provisional_runtime_path(driver).relative_to(driver.REPO_ROOT)
            ),
            "contract_sha256": campaign_runtime["campaign_runtime_sha256"],
        },
        "manifest": manifest,
        "generation_policy": {
            "phase": "full",
            "seed": 7919,
            "batch_size": 2,
            "arms": ["native", "paper_eta_0p125"],
            "retry_count": 0,
        },
        "runs": records,
    }
    plan["full_e2e_plan_sha256"] = driver._canonical_json_sha256(plan)
    continuation_path, continuation_content, _ = full_continuation_contract_binding(
        continuation, repo_root=driver.REPO_ROOT
    )
    selection_path, selection_content, _ = full_selection_binding(
        selection, repo_root=driver.REPO_ROOT
    )
    pending_writes = [
        (selection_path, selection_content),
        (continuation_path, continuation_content),
        (_provisional_runtime_path(driver), _contract_bytes(campaign_runtime)),
        *config_writes,
    ]
    pending_writes.append((root / "plan.json", _contract_bytes(plan)))
    if materialize:
        for path, content in pending_writes:
            driver._write_immutable_bytes(path, content)
    return {
        "status": "prepared" if materialize else "dry_run_validated",
        "plan_sha256": plan["full_e2e_plan_sha256"],
        "generation_execution_count": 0,
        "evaluator_execution_count": 0,
        "artifact_write_count": len(pending_writes) if materialize else 0,
    }


def _run_e2e(driver) -> dict:
    if not os.environ.get("TMUX"):
        raise RuntimeError("Full E2E execution must run inside tmux")
    campaign_runtime = _load_effective_runtime(driver, allow_provisional=True)
    root = _e2e_root(driver)
    plan = _load_e2e_plan(driver)
    declared = plan["full_e2e_plan_sha256"]
    admission = driver._full_admission_preflight()
    scheduler, gpu_bindings, peer_status_store = driver.build_resource_scheduler(
        campaign_runtime
    )
    e2e_bootstrap = driver._mapping(
        campaign_runtime.get("full_e2e_bootstrap"), "Full E2E bootstrap"
    )
    resource_policy = driver._mapping(
        e2e_bootstrap.get("resource_policy"), "Full E2E resource policy"
    )
    runtime_guard = driver.FullRuntimeGuard(
        resource_policy,
        monitor_path=root / "monitor/resource_samples.jsonl",
    )
    scheduler_admission = _pre_admit_e2e_plan(
        driver,
        scheduler=scheduler,
        gpu_bindings=gpu_bindings,
        campaign_runtime=campaign_runtime,
        plan=plan,
    )
    monitor_claim = _materialize_e2e_monitor_claim(driver)
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_execution_v1",
        "plan_sha256": declared,
        "admission": admission,
        "scheduler_admission": scheduler_admission,
        "monitor_claim": monitor_claim,
        "retry_allowed": False,
    }
    claim["execution_claim_sha256"] = driver._canonical_json_sha256(claim)
    driver._write_exclusive_bytes(
        root / "execution_claim.json", _contract_bytes(claim)
    )
    try:
        _start_e2e_monitor(driver, monitor_claim)
        runtime_guard.bind_monitor(
            session_name=monitor_claim["session_name"],
            claim_path=root / "monitor/claim.json",
            claim_sha256=monitor_claim["monitor_claim_sha256"],
        )
    except BaseException:
        subprocess.run(
            ["tmux", "kill-session", "-t", monitor_claim["session_name"]],
            check=False,
            capture_output=True,
        )
        terminal = {
            "schema_version": 1,
            "contract_type": "safa_r9_full_e2e_execution_terminal_v1",
            "status": "failed_before_worker",
            "execution_claim_sha256": claim["execution_claim_sha256"],
            "error_type": "MonitorStartError",
        }
        terminal["execution_terminal_sha256"] = (
            driver._canonical_json_sha256(terminal)
        )
        driver._write_exclusive_bytes(
            root / "execution_terminal.json", _contract_bytes(terminal)
        )
        raise
    runs = tuple(
        driver.RunSpec(
            phase="full",
            logical_run_id=f"formal_e2e_{row['arm_id']}_8",
            arm_ref=row["arm_id"],
            seed=7919,
            repeat_index=None,
            shard_index=0,
            num_shards=1,
            sample_count=8,
            manifest_key="full_2048",
            runtime_config=Path(row["runtime_config"]),
            output_dir=Path(row["output_dir"]),
            command=tuple(row["command"]),
        )
        for row in plan["runs"]
    )
    try:
        driver.execute_campaign(
            (
                driver.PhasePlan(
                    phase="full",
                    campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
                    campaign_root=Path(campaign_runtime["campaign_root"]),
                    logical_run_count=2,
                    runs=runs,
                ),
            ),
            scheduler=scheduler,
            gpu_bindings=gpu_bindings,
            peer_status_store=peer_status_store,
            runtime_guard=runtime_guard,
        )
        result = _run_e2e_evaluators(
            driver,
            campaign_runtime,
            plan,
            scheduler,
            gpu_bindings,
            peer_status_store,
            runtime_guard,
        )
    except BaseException as error:
        terminal = {
            "schema_version": 1,
            "contract_type": "safa_r9_full_e2e_execution_terminal_v1",
            "status": "failed",
            "execution_claim_sha256": claim["execution_claim_sha256"],
            "error_type": type(error).__name__,
        }
        terminal["execution_terminal_sha256"] = (
            driver._canonical_json_sha256(terminal)
        )
        driver._write_exclusive_bytes(
            root / "execution_terminal.json", _contract_bytes(terminal)
        )
        _wait_e2e_monitor(driver)
        raise
    terminal = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_execution_terminal_v1",
        "status": "succeeded",
        "execution_claim_sha256": claim["execution_claim_sha256"],
        "full_e2e_result_sha256": result["full_e2e_result_sha256"],
    }
    terminal["execution_terminal_sha256"] = driver._canonical_json_sha256(
        terminal
    )
    driver._write_exclusive_bytes(
        root / "execution_terminal.json", _contract_bytes(terminal)
    )
    _wait_e2e_monitor(driver)
    return result


def _materialize_e2e_monitor_claim(driver) -> dict:
    root = _e2e_root(driver)
    session = "safa-r9-v9-e2e-monitor"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "e2e-monitor",
        "--execute",
    ]
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_monitor_claim_v1",
        "session_name": session,
        "plan_sha256": _load_e2e_plan(driver)["full_e2e_plan_sha256"],
        "command": command,
        "records": [
            "gpu",
            "cpu",
            "ram",
            "disk",
            "log_byte_progress",
            "png_count",
            "result_count",
        ],
    }
    claim["monitor_claim_sha256"] = driver._canonical_json_sha256(claim)
    driver._write_exclusive_bytes(
        root / "monitor/claim.json", _contract_bytes(claim)
    )
    return claim


def _start_e2e_monitor(driver, claim: dict) -> None:
    session = claim["session_name"]
    if subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        capture_output=True,
    ).returncode == 0:
        raise RuntimeError("Full E2E monitor tmux session already exists")
    started = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, *claim["command"]],
        cwd=driver.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        raise RuntimeError(
            f"Full E2E monitor tmux start failed: {started.stderr.strip()}"
        )
    live = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        capture_output=True,
    )
    if live.returncode != 0:
        raise RuntimeError("Full E2E monitor tmux did not stay alive")


def _monitor_e2e(driver) -> int:
    if not os.environ.get("TMUX"):
        raise RuntimeError("Full E2E monitor must run inside its tmux session")
    root = _e2e_root(driver)
    claim = driver._read_json_mapping(
        root / "monitor/claim.json", "Full E2E monitor claim"
    )
    observed_session = subprocess.run(
        ["tmux", "display-message", "-p", "#S"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_session != claim["session_name"]:
        raise RuntimeError("Full E2E monitor tmux session name changed")
    samples_path = root / "monitor/session_samples.jsonl"
    sample_count = 0
    last_log_bytes = -1
    log_progress_count = 0
    previous_cpu = driver._read_cpu_times()
    while not (root / "execution_terminal.json").is_file():
        probe = driver.SystemResourceProbe()
        ram = probe.ram_snapshot()
        gpu = [
            {
                "index": row.index,
                "uuid": row.uuid,
                "used_bytes": row.total_bytes - row.free_bytes,
                "total_bytes": row.total_bytes,
            }
            for row in probe.gpu_snapshots()
            if row.index in {0, 1, 2, 3}
        ]
        usage = shutil.disk_usage(driver.REPO_ROOT)
        current_cpu = driver._read_cpu_times()
        total_delta = current_cpu[0] - previous_cpu[0]
        idle_delta = current_cpu[1] - previous_cpu[1]
        previous_cpu = current_cpu
        if total_delta <= 0 or not 0 <= idle_delta <= total_delta:
            raise RuntimeError("Full E2E monitor CPU counters changed")
        cpu_busy_percent = 100.0 * (total_delta - idle_delta) / total_delta
        log_bytes = sum(
            path.stat().st_size
            for path in root.rglob("*.log")
            if path.is_file()
        )
        if last_log_bytes >= 0 and log_bytes > last_log_bytes:
            log_progress_count += 1
        last_log_bytes = log_bytes
        sample = {
            "schema_version": 1,
            "contract_type": "safa_r9_full_e2e_monitor_sample_v1",
            "monotonic_ns": time.monotonic_ns(),
            "gpu": gpu,
            "ram_used_bytes": ram.used_bytes,
            "ram_total_bytes": ram.total_bytes,
            "disk_used_bytes": usage.used,
            "disk_total_bytes": usage.total,
            "cpu_busy_percent": cpu_busy_percent,
            "log_bytes": log_bytes,
            "png_count": sum(1 for _ in root.rglob("*.png")),
            "result_count": sum(1 for _ in root.rglob("*result.json")),
        }
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(sample, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        sample_count += 1
        time.sleep(2)
    terminal = driver._read_json_mapping(
        root / "execution_terminal.json", "Full E2E execution terminal"
    )
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_monitor_summary_v1",
        "monitor_claim_sha256": claim["monitor_claim_sha256"],
        "execution_terminal_sha256": terminal["execution_terminal_sha256"],
        "tmux_session": os.environ["TMUX"],
        "sample_count": sample_count,
        "log_progress_count": log_progress_count,
        "samples": {
            "path": str(samples_path.relative_to(driver.REPO_ROOT)),
            "file_sha256": _file_sha256(samples_path),
        },
    }
    summary["monitor_summary_sha256"] = driver._canonical_json_sha256(
        summary
    )
    driver._write_exclusive_bytes(
        root / "monitor/summary.json", _contract_bytes(summary)
    )
    return 0


def _wait_e2e_monitor(driver) -> None:
    summary = _e2e_root(driver) / "monitor/summary.json"
    deadline = time.monotonic() + 30
    while not summary.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("Full E2E monitor did not finalize")
        time.sleep(0.5)


def _monitor_formal_full(driver) -> int:
    if not os.environ.get("TMUX"):
        raise RuntimeError("formal Full monitor must run inside tmux")
    root = (
        driver.REPO_ROOT
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID
    )
    claim = driver._read_json_mapping(
        root / "formal_monitor/claim.json", "formal Full monitor claim"
    )
    observed_session = subprocess.run(
        ["tmux", "display-message", "-p", "#S"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_session != claim["session_name"]:
        raise RuntimeError("formal Full monitor tmux session name changed")
    samples_path = root / "formal_monitor/session_samples.jsonl"
    terminal_path = root / "formal_execution_terminal.json"
    sample_count = 0
    last_log_bytes = -1
    log_progress_count = 0
    previous_cpu = driver._read_cpu_times()
    while not terminal_path.is_file():
        probe = driver.SystemResourceProbe()
        ram = probe.ram_snapshot()
        usage = shutil.disk_usage(driver.REPO_ROOT)
        current_cpu = driver._read_cpu_times()
        total_delta = current_cpu[0] - previous_cpu[0]
        idle_delta = current_cpu[1] - previous_cpu[1]
        previous_cpu = current_cpu
        if total_delta <= 0 or not 0 <= idle_delta <= total_delta:
            raise RuntimeError("formal Full monitor CPU counters changed")
        cpu_busy_percent = 100.0 * (total_delta - idle_delta) / total_delta
        log_bytes = sum(
            path.stat().st_size
            for path in root.rglob("*.log")
            if path.is_file()
        )
        if last_log_bytes >= 0 and log_bytes > last_log_bytes:
            log_progress_count += 1
        last_log_bytes = log_bytes
        sample = {
            "schema_version": 1,
            "contract_type": "safa_r9_formal_full_monitor_sample_v1",
            "monotonic_ns": time.monotonic_ns(),
            "gpu": [
                {
                    "index": row.index,
                    "uuid": row.uuid,
                    "used_bytes": row.total_bytes - row.free_bytes,
                    "total_bytes": row.total_bytes,
                }
                for row in probe.gpu_snapshots()
                if row.index in {0, 1, 2, 3}
            ],
            "ram_used_bytes": ram.used_bytes,
            "ram_total_bytes": ram.total_bytes,
            "disk_used_bytes": usage.used,
            "disk_total_bytes": usage.total,
            "cpu_busy_percent": cpu_busy_percent,
            "log_bytes": log_bytes,
            "png_count": sum(1 for _ in root.rglob("*.png")),
            "result_count": sum(1 for _ in root.rglob("*result.json")),
        }
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(sample, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        sample_count += 1
        time.sleep(5)
    terminal = driver._read_json_mapping(
        terminal_path, "formal Full execution terminal"
    )
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r9_formal_full_monitor_summary_v1",
        "monitor_claim_sha256": claim["monitor_claim_sha256"],
        "formal_execution_terminal_sha256": terminal[
            "formal_execution_terminal_sha256"
        ],
        "tmux_session": os.environ["TMUX"],
        "sample_count": sample_count,
        "log_progress_count": log_progress_count,
        "samples": {
            "path": str(samples_path.relative_to(driver.REPO_ROOT)),
            "file_sha256": _file_sha256(samples_path),
        },
    }
    summary["monitor_summary_sha256"] = driver._canonical_json_sha256(
        summary
    )
    driver._write_exclusive_bytes(
        root / "formal_monitor/summary.json", _contract_bytes(summary)
    )
    return 0


def _pre_admit_e2e_plan(
    driver,
    *,
    scheduler,
    gpu_bindings,
    campaign_runtime,
    plan,
) -> dict:
    admitted = []
    generation_workers = []
    for index, row in enumerate(plan["runs"]):
        worker_id = f"e2e-preflight:generation:{row['arm_id']}"
        lease = driver._admit_worker(
            scheduler,
            worker_id=worker_id,
            launch_ordinal=90_000 + index,
            gpu_bindings=gpu_bindings,
            ram_slot_budget_bytes=scheduler.ram_slot_budget_bytes,
            start_gpu_index=index,
        )
        if lease is None:
            raise driver.ResourceContractError(
                f"Full E2E scheduler rejected planned worker {worker_id}"
            )
        generation_workers.append(worker_id)
        admitted.append(
            {
                "worker_id": worker_id,
                "kind": "generation",
                "gpu_uuid": lease.gpu_uuid,
                "slot_index": lease.slot_index,
            }
        )
    for worker_id in generation_workers:
        scheduler.release_worker(worker_id)
    evaluator_budget = campaign_runtime["evaluation"]["resource_smokes"][
        "arcface"
    ]["ram_slot_budget_bytes"]
    for index, unit in enumerate(
        ("arcface", "quality_native", "quality_candidate")
    ):
        worker_id = f"e2e-preflight:evaluator:{unit}"
        lease = driver._admit_worker(
            scheduler,
            worker_id=worker_id,
            launch_ordinal=91_000 + index,
            gpu_bindings=gpu_bindings,
            ram_slot_budget_bytes=evaluator_budget,
            start_gpu_index=index % 4,
        )
        if lease is None:
            raise driver.ResourceContractError(
                f"Full E2E scheduler rejected planned worker {worker_id}"
            )
        admitted.append(
            {
                "worker_id": worker_id,
                "kind": unit,
                "gpu_uuid": lease.gpu_uuid,
                "slot_index": lease.slot_index,
            }
        )
        scheduler.release_worker(worker_id)
    evidence = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_scheduler_admission_v1",
        "plan_sha256": plan["full_e2e_plan_sha256"],
        "claim_type": (
            "preregistered_exclusive_upper_bound_not_measured_profile"
        ),
        "admitted": admitted,
        "active_lease_count_after_preflight": len(scheduler.active_leases),
    }
    if evidence["active_lease_count_after_preflight"] != 0:
        raise driver.ResourceContractError(
            "Full E2E pre-admission leaked scheduler leases"
        )
    evidence["scheduler_admission_sha256"] = driver._canonical_json_sha256(
        evidence
    )
    return evidence


def _run_e2e_evaluators(
    driver,
    campaign_runtime,
    plan,
    scheduler,
    gpu_bindings,
    peer_status_store,
    runtime_guard,
) -> dict:
    runtime, _, _ = driver.load_full_continuation_request()
    root = _e2e_root(driver)
    rows = {
        arm: _per_sample_rows(root / "generation" / arm / "per_sample.jsonl")
        for arm in ("native", "paper_eta_0p125")
    }
    samples = []
    for sample_id in plan["manifest"]["sample_ids"]:
        native = rows["native"][sample_id]
        winner = rows["paper_eta_0p125"][sample_id]
        source = Path(native["source"]).resolve()
        native_path = Path(native["native"]).resolve()
        candidate = Path(winner["generated"]).resolve()
        samples.append(
            driver.SampleEvidence(
                sample_id=sample_id,
                source=source,
                native=native_path,
                candidate=candidate,
                source_sha256=_file_sha256(source),
                native_sha256=_file_sha256(native_path),
                candidate_sha256=_file_sha256(candidate),
            )
        )
    evaluation = campaign_runtime["evaluation"]
    source_index = evaluation["quality"]["real_index"]
    callbacks = driver.R9ProductionEvaluatorCallbacks(
        runtime=runtime,
        campaign_runtime=campaign_runtime,
        scheduler=scheduler,
        gpu_bindings=gpu_bindings,
        peer_status_store=peer_status_store,
        runtime_guard=runtime_guard,
    )
    arc_request = driver.ArcFaceEvaluationRequest(
        phase="full_e2e",
        logical_run_id="formal_e2e_arcface_8",
        arm_id="paper_eta_0p125",
        seed=7919,
        source_index_path=driver.REPO_ROOT / source_index["path"],
        source_index_sha256=source_index["sha256"],
        samples=tuple(samples),
    )
    winner = driver._require_full_selection_binding(
        driver._continuation_for_runtime(campaign_runtime), campaign_runtime
    )["winner"]
    native_config = yaml.safe_load(
        (
            driver.REPO_ROOT
            / next(row["runtime_config"] for row in plan["runs"] if row["arm_id"] == "native")
        ).read_text(encoding="utf-8")
    )
    if not isinstance(native_config, dict):
        raise ValueError("Full E2E native runtime config changed")
    native_config_sha256 = native_config["arm_config_sha256"]
    common_quality = {
        "phase": "full_e2e",
        "logical_run_id": "formal_e2e_quality_8",
        "seed": 7919,
        "manifest_path": driver.REPO_ROOT / plan["manifest"]["path"],
        "source_index_path": driver.REPO_ROOT / source_index["path"],
        "source_index_sha256": source_index["sha256"],
        "samples": tuple(samples),
        "evidence_binding_sha256": driver._canonical_json_sha256(
            [
                {
                    "sample_id": row.sample_id,
                    "source": row.source_sha256,
                    "native": row.native_sha256,
                    "candidate": row.candidate_sha256,
                }
                for row in samples
            ]
        ),
        "generation_result_set_sha256": driver._canonical_json_sha256(
            [
                _file_sha256(root / "generation" / arm / "generation_result.json")
                for arm in ("native", "paper_eta_0p125")
            ]
        ),
        "per_sample_set_sha256": driver._canonical_json_sha256(
            [
                _file_sha256(root / "generation" / arm / "per_sample.jsonl")
                for arm in ("native", "paper_eta_0p125")
            ]
        ),
    }
    native_quality_request = driver.QualityEvaluationRequest(
        **common_quality,
        arm_id="native",
        image_role="native",
        algorithm_config_sha256=native_config_sha256,
        runner_arm_config_sha256=native_config_sha256,
        semantic_output_sha256=driver._canonical_json_sha256(
            [
                {"sample_id": row.sample_id, "sha256": row.native_sha256}
                for row in samples
            ]
        ),
    )
    candidate_quality_request = driver.QualityEvaluationRequest(
        **common_quality,
        arm_id="paper_eta_0p125",
        image_role="candidate",
        algorithm_config_sha256=winner["config_sha256"],
        runner_arm_config_sha256=winner["config_sha256"],
        semantic_output_sha256=driver._canonical_json_sha256(
            [
                {"sample_id": row.sample_id, "sha256": row.candidate_sha256}
                for row in samples
            ]
        ),
    )
    callbacks.arcface(arc_request)
    callbacks.quality(native_quality_request)
    callbacks.quality(candidate_quality_request)
    driver.materialize_full_e2e_resource_profiles(campaign_runtime)
    result = _collect_e2e_result(driver, campaign_runtime, plan)
    driver._write_exclusive_bytes(
        root / "run_result.json", _contract_bytes(result)
    )
    return result


def _collect_e2e_result(driver, campaign_runtime, plan) -> dict:
    rebuilt = driver._rebuild_full_e2e_evidence(
        campaign_runtime, require_materialized_result=False
    )
    if rebuilt["plan"] != plan:
        raise ValueError("Full E2E plan changed during result collection")
    return rebuilt["result"]


def _finalize_e2e(driver) -> dict:
    provisional_runtime = _load_effective_runtime(
        driver, allow_provisional=True
    )
    root = _e2e_root(driver)
    plan = _load_e2e_plan(driver)
    result = driver._read_json_mapping(root / "run_result.json", "Full E2E result")
    if result != _collect_e2e_result(driver, provisional_runtime, plan):
        raise ValueError("Full E2E result bindings changed")
    runtime, runtime_path, source = driver.load_full_continuation_request()
    final_continuation = driver.build_full_continuation_contract(
        repo_root=driver.REPO_ROOT, expected_source=source
    )
    effective_runtime, _, _ = driver.build_effective_campaign_runtime(
        runtime,
        campaign_id=driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        repo_root=driver.REPO_ROOT,
        runtime_config_path=runtime_path,
        continuation_contract_override=final_continuation,
    )
    if effective_runtime.get("campaign_runtime_sha256") is None:
        raise RuntimeError("Full E2E did not produce a final v9 campaign runtime")
    rebuilt = driver._rebuild_full_e2e_evidence(effective_runtime)
    if rebuilt["result"] != result:
        raise ValueError("Final v9 runtime changes the Full E2E result")
    gate = rebuilt["gate"]
    runtime_content = _contract_bytes(effective_runtime)
    gate_content = _contract_bytes(gate)
    # Authorization is fail-closed: the runtime may exist without a gate, but
    # the pass gate is always linked last after every in-memory validation.
    driver._write_immutable_bytes(
        driver.REPO_ROOT
        / effective_runtime["campaign_root"]
        / "campaign_runtime.json",
        runtime_content,
    )
    driver._write_exclusive_bytes(
        root / "gate_contract.json",
        gate_content,
    )
    driver._require_full_e2e_gate(effective_runtime)
    return gate


def _load_e2e_plan(driver) -> dict:
    plan = driver._read_json_mapping(
        _e2e_root(driver) / "plan.json", "Full E2E plan"
    )
    expected = {
        "schema_version",
        "contract_type",
        "campaign_id",
        "continuation_contract_sha256",
        "selection_sha256",
        "request_config",
        "e2e_request",
        "generation_batch_benchmark",
        "provisional_runtime",
        "manifest",
        "generation_policy",
        "runs",
        "full_e2e_plan_sha256",
    }
    if (
        set(plan) != expected
        or plan.get("schema_version") != 1
        or plan.get("contract_type") != "safa_r9_full_e2e_plan_v1"
        or plan.get("campaign_id") != driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID
    ):
        raise ValueError("Full E2E plan fields changed")
    declared = plan["full_e2e_plan_sha256"]
    canonical = dict(plan)
    canonical.pop("full_e2e_plan_sha256")
    if driver._canonical_json_sha256(canonical) != declared:
        raise ValueError("Full E2E plan digest mismatch")
    return plan


def _manifest_ids(path: Path) -> list[str]:
    return [
        json.loads(line)["sample_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _per_sample_rows(path: Path) -> dict[str, dict]:
    return {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
