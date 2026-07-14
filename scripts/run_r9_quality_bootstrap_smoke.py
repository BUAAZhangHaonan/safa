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
    RAM_ADMISSION_PERCENT,
    RAM_HARD_LIMIT_PERCENT,
    _canonical_sha256,
    _gpu_snapshots,
    _lock_all_slots,
    _process_tree,
    _ram_snapshot,
    _release_locks,
    _sha256,
    _terminate,
    _tree_gpu_bytes,
    _tree_rss_bytes,
    _write_exclusive,
)
from run_r9_evaluator_resource_smoke import (
    EvaluatorSmokeError,
    _read_mapping,
    _validate_digest,
    _validate_worker_output,
)
from safa.evaluation.r9_resources import ram_slot_budget_bytes


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
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
    result_path = artifact_root / "resource_result.json"
    log_path = artifact_root / "worker.log"
    for path in (execution_claim_path, worker_output_path, result_path, log_path):
        if path.exists():
            raise EvaluatorSmokeError(f"quality bootstrap already attempted: {path}")
    request = _read_mapping(request_path, "quality evaluator request")
    request_claim = _read_mapping(request_claim_path, "quality request claim")
    request_sha = _validate_digest(
        request, "evaluator_request_sha256", "quality evaluator request"
    )
    request_claim_sha = _validate_digest(
        request_claim, "smoke_request_claim_sha256", "quality request claim"
    )
    if (
        request.get("task") != "quality"
        or request_claim.get("kind") != "quality"
        or request_claim.get("evaluator_request_sha256") != request_sha
    ):
        raise EvaluatorSmokeError("quality evaluator request/claim mismatch")
    worker_contract = request_claim.get("worker_contract")
    if not isinstance(worker_contract, Mapping):
        raise EvaluatorSmokeError("quality worker contract is missing")
    wrapper = Path(str(worker_contract.get("path"))).resolve(strict=True)
    implementation = Path(str(worker_contract.get("implementation_path"))).resolve(
        strict=True
    )
    if _sha256(wrapper) != worker_contract.get("sha256") or _sha256(
        implementation
    ) != worker_contract.get("implementation_sha256"):
        raise EvaluatorSmokeError("quality worker implementation digest mismatch")
    gpus = _gpu_snapshots()
    if args.gpu_index not in gpus or gpus[args.gpu_index][0] != args.gpu_uuid:
        raise EvaluatorSmokeError("quality bootstrap GPU index/UUID mismatch")
    total_ram, available_ram = _ram_snapshot()
    if (total_ram - available_ram) * 100 >= total_ram * RAM_ADMISSION_PERCENT:
        raise EvaluatorSmokeError("quality bootstrap cannot start at or above 85% RAM")
    descriptors = _lock_all_slots(gpus)
    execution_claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_quality_bootstrap_smoke_execution_v1",
        "campaign_id": args.campaign_id,
        "request_claim_sha256": request_claim_sha,
        "evaluator_request_sha256": request_sha,
        "launcher_path": str(Path(__file__).resolve()),
        "launcher_sha256": _sha256(Path(__file__).resolve()),
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "ram": {
            "total_bytes": total_ram,
            "available_bytes_at_admission": available_ram,
            "admission_percent": RAM_ADMISSION_PERCENT,
            "hard_limit_percent": RAM_HARD_LIMIT_PERCENT,
        },
        "global_exclusive_slots": 16,
        "retry_allowed": False,
    }
    execution_claim["execution_claim_sha256"] = _canonical_sha256(execution_claim)
    command = [
        sys.executable,
        str(wrapper),
        "--request",
        str(request_path),
        "--output",
        str(worker_output_path),
    ]
    log_fd: int | None = None
    process: subprocess.Popen[Any] | None = None
    peak_rss = 0
    peak_gpu = 0
    failure_reason: str | None = None
    try:
        _write_exclusive(execution_claim_path, execution_claim)
        log_fd = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "SAFA_R9_WORKER_ID": "evaluator-smoke:quality",
                "SAFA_R9_GPU_UUID": args.gpu_uuid,
                "SAFA_R9_GPU_SLOT": "bootstrap-exclusive",
            }
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            pids = _process_tree(process.pid)
            peak_rss = max(peak_rss, _tree_rss_bytes(pids))
            peak_gpu = max(peak_gpu, _tree_gpu_bytes(pids, args.gpu_uuid))
            current_total, current_available = _ram_snapshot()
            if (
                current_total - current_available
            ) * 100 >= current_total * RAM_HARD_LIMIT_PERCENT:
                failure_reason = "actual system RAM reached the R9 90% hard limit"
                _terminate(process)
                break
            time.sleep(0.1)
        returncode = process.wait()
    finally:
        if log_fd is not None:
            os.close(log_fd)
        _release_locks(descriptors)
    if failure_reason is None and returncode != 0:
        failure_reason = f"quality worker exited once with status {returncode}"
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
    if failure_reason is None and peak_rss <= 0:
        failure_reason = "quality bootstrap measured no positive process-tree RSS"
    work_root = artifact_root / "work"
    if work_root.exists():
        if work_root.is_symlink() or not work_root.is_dir() or any(work_root.iterdir()):
            failure_reason = (
                failure_reason or "quality evaluator work root was not empty"
            )
        else:
            work_root.rmdir()
    result = {
        "schema_version": 1,
        "contract_type": "safa_r9_quality_bootstrap_smoke_result_v1",
        "execution_claim_sha256": execution_claim["execution_claim_sha256"],
        "status": "succeeded" if failure_reason is None else "failed",
        "failure_reason": failure_reason,
        "returncode": returncode,
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_gpu_memory_bytes": peak_gpu,
        "ram_slot_budget_bytes": (
            ram_slot_budget_bytes(peak_rss) if peak_rss > 0 else None
        ),
        "worker_output_sha256": (
            _sha256(worker_output_path) if worker_output_path.is_file() else None
        ),
        "worker_evaluator_output_sha256": (
            None if worker_output is None else worker_output["evaluator_output_sha256"]
        ),
        "worker_log_sha256": _sha256(log_path),
        "gpu_uuid": args.gpu_uuid,
        "retry_allowed": False,
    }
    result["resource_smoke_result_sha256"] = _canonical_sha256(result)
    _write_exclusive(result_path, result)
    print(json.dumps(result, sort_keys=True))
    if failure_reason is not None:
        raise EvaluatorSmokeError(failure_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
