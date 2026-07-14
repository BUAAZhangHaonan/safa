#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from run_r9_bootstrap_resource_smoke import (
    _canonical_sha256,
    _process_tree,
    _sha256,
    _terminate,
    _tree_gpu_bytes,
    _tree_rss_bytes,
    _write_exclusive,
)
from safa.evaluation.r9_resources import (
    AdmissionStatus,
    CampaignFailedError,
    FcntlSlotLockBackend,
    R9PeerStatusStore,
    R9ResourceScheduler,
    SystemResourceProbe,
    WorkerRequest,
    ram_slot_budget_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_SLOT_ROOT = Path("/tmp/safa-r9-gpu-slots-v1")


class EvaluatorSmokeError(RuntimeError):
    pass


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluatorSmokeError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluatorSmokeError(f"{label} is not an object")
    return value


def _validate_digest(value: Mapping[str, Any], field: str, label: str) -> str:
    declared = value.get(field)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise EvaluatorSmokeError(f"{label} digest is invalid")
    canonical = dict(value)
    canonical.pop(field)
    if _canonical_sha256(canonical) != declared:
        raise EvaluatorSmokeError(f"{label} digest mismatch")
    return declared


def _validate_worker_output(
    path: Path,
    *,
    request: Mapping[str, Any],
    request_claim: Mapping[str, Any],
) -> dict[str, Any]:
    output = _read_mapping(path, "evaluator worker output")
    expected_fields = {
        "schema_version",
        "contract_type",
        "task",
        "evaluator_request_sha256",
        "worker_contract",
        "arcface_contract_sha256",
        "quality_script_sha256",
        "result",
        "evaluator_output_sha256",
    }
    if set(output) != expected_fields:
        raise EvaluatorSmokeError("evaluator worker output fields are not canonical")
    if (
        output.get("schema_version") != 1
        or output.get("contract_type") != "safa_r9_phase_evaluator_output_v1"
        or output.get("task") != request.get("task")
        or output.get("evaluator_request_sha256")
        != request.get("evaluator_request_sha256")
        or output.get("worker_contract") != request_claim.get("worker_contract")
        or output.get("arcface_contract_sha256")
        != request_claim.get("arcface_contract_sha256")
        or output.get("quality_script_sha256")
        != request_claim.get("quality_script_sha256")
    ):
        raise EvaluatorSmokeError("evaluator worker output binding mismatch")
    _validate_digest(output, "evaluator_output_sha256", "evaluator worker output")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("arcface",), required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--profile-result", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute or not args.allow_busy_gpus:
        parser.error("execution requires --execute --allow-busy-gpus")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    request_path = artifact_root / "request.json"
    request_claim_path = artifact_root / "request_claim.json"
    execution_claim_path = artifact_root / "execution_claim.json"
    worker_output_path = artifact_root / "worker_result.json"
    smoke_result_path = artifact_root / "resource_result.json"
    worker_log_path = artifact_root / "worker.log"
    for path in (
        execution_claim_path,
        worker_output_path,
        smoke_result_path,
        worker_log_path,
    ):
        if path.exists():
            raise EvaluatorSmokeError(
                f"evaluator resource smoke already attempted: {path}"
            )
    request = _read_mapping(request_path, "evaluator request")
    request_claim = _read_mapping(request_claim_path, "evaluator request claim")
    request_claim_sha = _validate_digest(
        request_claim, "smoke_request_claim_sha256", "evaluator request claim"
    )
    if (
        request.get("task") != args.kind
        or request_claim.get("kind") != args.kind
        or request.get("evaluator_request_sha256")
        != request_claim.get("evaluator_request_sha256")
    ):
        raise EvaluatorSmokeError("evaluator request/claim kind or digest mismatch")
    request_sha = _validate_digest(
        request, "evaluator_request_sha256", "evaluator request"
    )
    if request_sha != request_claim["evaluator_request_sha256"]:
        raise EvaluatorSmokeError("evaluator request digest disagrees with claim")
    profile_result_path = args.profile_result.resolve(strict=True)
    profile_result = _read_mapping(profile_result_path, "ArcFace profile result")
    profile_result_sha = _validate_digest(
        profile_result, "bootstrap_result_sha256", "ArcFace profile result"
    )
    if (
        profile_result.get("status") != "succeeded"
        or profile_result.get("failure_reason") is not None
        or profile_result.get("retry_allowed") is not False
    ):
        raise EvaluatorSmokeError(
            "ArcFace profile result is not a successful final result"
        )
    peak_rss = profile_result.get("peak_process_tree_rss_bytes")
    profile_budget = profile_result.get("ram_slot_budget_bytes")
    if (
        isinstance(peak_rss, bool)
        or not isinstance(peak_rss, int)
        or peak_rss <= 0
        or profile_budget != ram_slot_budget_bytes(peak_rss)
    ):
        raise EvaluatorSmokeError("ArcFace profile RAM budget mismatch")
    worker_contract = request_claim.get("worker_contract")
    if not isinstance(worker_contract, Mapping):
        raise EvaluatorSmokeError("worker contract is missing")
    wrapper_path = Path(str(worker_contract.get("path"))).resolve(strict=True)
    if _sha256(wrapper_path) != worker_contract.get("sha256"):
        raise EvaluatorSmokeError("worker wrapper digest mismatch")
    implementation_path = Path(str(worker_contract.get("implementation_path"))).resolve(
        strict=True
    )
    if _sha256(implementation_path) != worker_contract.get("implementation_sha256"):
        raise EvaluatorSmokeError("worker implementation digest mismatch")
    execution_claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_evaluator_resource_smoke_execution_v1",
        "campaign_id": args.campaign_id,
        "kind": args.kind,
        "request_claim_sha256": request_claim_sha,
        "evaluator_request_sha256": request_sha,
        "profile_result": str(profile_result_path),
        "profile_result_sha256": profile_result_sha,
        "profile_ram_slot_budget_bytes": profile_budget,
        "launcher_path": str(Path(__file__).resolve()),
        "launcher_sha256": _sha256(Path(__file__).resolve()),
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "global_slot_root": str(GLOBAL_SLOT_ROOT),
        "retry_allowed": False,
    }
    execution_claim["execution_claim_sha256"] = _canonical_sha256(execution_claim)
    _write_exclusive(execution_claim_path, execution_claim)
    resource_contract_sha = _canonical_sha256(
        {
            "execution_claim_sha256": execution_claim["execution_claim_sha256"],
            "worker_contract": dict(worker_contract),
            "arcface_contract_sha256": request_claim["arcface_contract_sha256"],
            "quality_script_sha256": request_claim["quality_script_sha256"],
        }
    )
    probe = SystemResourceProbe()
    peer_store = R9PeerStatusStore(
        artifact_root.parent,
        campaign_id=args.campaign_id,
    )
    scheduler = R9ResourceScheduler(
        campaign_id=args.campaign_id,
        resource_contract_sha256=resource_contract_sha,
        smoke_peak_rss_bytes=peak_rss,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(GLOBAL_SLOT_ROOT),
        peer_status_probe=peer_store,
    )
    decision = scheduler.admit_worker(
        WorkerRequest(
            worker_id=f"evaluator-smoke:{args.kind}",
            gpu_index=args.gpu_index,
            expected_gpu_uuid=args.gpu_uuid,
            resource_contract_sha256=resource_contract_sha,
            launch_ordinal=0,
            ram_slot_budget_bytes=profile_budget,
        )
    )
    if (
        decision.status
        not in {
            AdmissionStatus.ADMITTED,
            AdmissionStatus.RESUMED,
            AdmissionStatus.RECLAIMED,
        }
        or decision.lease is None
    ):
        raise EvaluatorSmokeError(
            f"ArcFace evaluator smoke admission failed: {decision.status.value}"
        )
    worker_id = f"evaluator-smoke:{args.kind}"
    peer_store.record_admitted(worker_id)
    command = [
        sys.executable,
        str(wrapper_path),
        "--request",
        str(request_path),
        "--output",
        str(worker_output_path),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "SAFA_R9_WORKER_ID": worker_id,
            "SAFA_R9_GPU_UUID": args.gpu_uuid,
            "SAFA_R9_GPU_SLOT": str(decision.lease.slot_index),
        }
    )
    log_fd = os.open(
        worker_log_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    process: subprocess.Popen[Any] | None = None
    peak_actual_rss = 0
    peak_gpu = 0
    failure_reason: str | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        peer_store.record_running(worker_id, pid=process.pid)
        while process.poll() is None:
            pids = _process_tree(process.pid)
            peak_actual_rss = max(peak_actual_rss, _tree_rss_bytes(pids))
            peak_gpu = max(peak_gpu, _tree_gpu_bytes(pids, args.gpu_uuid))
            try:
                scheduler.enforce_actual_ram_limit()
            except CampaignFailedError as exc:
                failure_reason = exc.failure.reason
                _terminate(process)
                break
            time.sleep(0.1)
        returncode = process.wait()
    finally:
        os.close(log_fd)
    if failure_reason is None and returncode != 0:
        failure_reason = f"evaluator worker exited once with status {returncode}"
    worker_output: dict[str, Any] | None = None
    if failure_reason is None:
        try:
            worker_output = _validate_worker_output(
                worker_output_path,
                request=request,
                request_claim=request_claim,
            )
        except BaseException as exc:
            failure_reason = str(exc)
    if failure_reason is None and peak_actual_rss <= 0:
        failure_reason = "evaluator smoke measured no positive process-tree RSS"
    status = "succeeded" if failure_reason is None else "failed"
    if status == "succeeded":
        peer_store.record_terminal(worker_id, state="succeeded")
        scheduler.release_worker(worker_id)
    else:
        peer_store.record_terminal(worker_id, state="failed")
        if worker_id in {lease.worker_id for lease in scheduler.active_leases}:
            scheduler.release_worker(worker_id)
    result = {
        "schema_version": 1,
        "contract_type": "safa_r9_evaluator_resource_smoke_result_v1",
        "execution_claim_sha256": execution_claim["execution_claim_sha256"],
        "status": status,
        "failure_reason": failure_reason,
        "returncode": returncode,
        "profile_ram_slot_budget_bytes": profile_budget,
        "peak_process_tree_rss_bytes": peak_actual_rss,
        "peak_gpu_memory_bytes": peak_gpu,
        "ram_slot_budget_bytes": (
            ram_slot_budget_bytes(peak_actual_rss) if peak_actual_rss > 0 else None
        ),
        "worker_output_sha256": (
            _sha256(worker_output_path) if worker_output_path.is_file() else None
        ),
        "worker_evaluator_output_sha256": (
            None if worker_output is None else worker_output["evaluator_output_sha256"]
        ),
        "worker_log_sha256": _sha256(worker_log_path),
        "gpu_uuid": args.gpu_uuid,
        "retry_allowed": False,
    }
    result["resource_smoke_result_sha256"] = _canonical_sha256(result)
    _write_exclusive(smoke_result_path, result)
    print(json.dumps(result, sort_keys=True))
    if failure_reason is not None:
        raise EvaluatorSmokeError(failure_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
