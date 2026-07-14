#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


GPU_SLOT_CLAIM_BYTES = 4_938_792_960
GPU_HEADROOM_BYTES = 2 * 1024**3
MAX_GPU_SLOTS = 4
RAM_ADMISSION_PERCENT = 85
RAM_HARD_LIMIT_PERCENT = 90
LOCK_ROOT = Path("/tmp/safa-r9-gpu-slots-v1")


class BootstrapError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        payload = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _ram_snapshot() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            fields = raw.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise BootstrapError(f"invalid /proc/meminfo field: {key}")
            values[key] = int(fields[0]) * 1024
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise BootstrapError("/proc/meminfo omitted required fields")
    return values["MemTotal"], values["MemAvailable"]


def _gpu_snapshots() -> dict[int, tuple[str, int, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[int, tuple[str, int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise BootstrapError("invalid nvidia-smi GPU row")
        index, uuid, free_mib, total_mib = fields
        numeric_index = int(index)
        if numeric_index in rows:
            raise BootstrapError("nvidia-smi reported a duplicate GPU index")
        free_bytes = int(free_mib) * 1024**2
        total_bytes = int(total_mib) * 1024**2
        if not uuid or total_bytes <= 0 or free_bytes < 0 or free_bytes > total_bytes:
            raise BootstrapError("nvidia-smi reported an invalid GPU snapshot")
        rows[numeric_index] = (uuid, free_bytes, total_bytes)
    registered = {0, 1, 2, 3}
    if not registered.issubset(rows):
        raise BootstrapError("bootstrap requires registered physical GPUs 0,1,2,3")
    configured = {index: rows[index] for index in sorted(registered)}
    if len({snapshot[0] for snapshot in configured.values()}) != len(configured):
        raise BootstrapError("registered physical GPU UUIDs must be unique")
    return configured


def _lock_all_slots(gpus: dict[int, tuple[str, int, int]]) -> list[int]:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    descriptors: list[int] = []
    bootstrap = os.open(
        LOCK_ROOT / "bootstrap_resource_smoke.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(bootstrap, fcntl.LOCK_EX | fcntl.LOCK_NB)
        descriptors.append(bootstrap)
        for index in sorted(gpus):
            uuid = gpus[index][0]
            uuid_digest = hashlib.sha256(uuid.encode()).hexdigest()[:24]
            for slot_index in range(MAX_GPU_SLOTS):
                path = LOCK_ROOT / f"gpu_{uuid_digest}.slot_{slot_index}.lock"
                descriptor = os.open(
                    path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
                )
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(descriptor)
                    raise
                descriptors.append(descriptor)
    except BaseException:
        for descriptor in reversed(descriptors):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if bootstrap not in descriptors:
            os.close(bootstrap)
        raise BootstrapError("global R9 bootstrap/slot lock is contended")
    return descriptors


def _release_locks(descriptors: Sequence[int]) -> None:
    for descriptor in reversed(descriptors):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _process_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            content = stat_path.read_text(encoding="utf-8")
            closing = content.rfind(")")
            fields = content[closing + 2 :].split()
            parents[int(stat_path.parent.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def _tree_rss_bytes(pids: set[int]) -> int:
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    total += int(fields[1]) * 1024
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def _tree_gpu_bytes(pids: set[int], gpu_uuid: str) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total = 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise BootstrapError("invalid nvidia-smi compute-process row")
        row_uuid, raw_pid, raw_mib = fields
        if row_uuid == gpu_uuid and int(raw_pid) in pids:
            total += int(raw_mib) * 1024**2
    return total


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--kind", choices=("arcface_profile",), required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute or not args.allow_busy_gpus:
        parser.error("bootstrap execution requires --execute --allow-busy-gpus")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    probe_script = args.probe_script.resolve(strict=True)
    model_root = args.model_root.resolve(strict=True)
    probe_output = args.probe_output.resolve()
    artifact_root = args.artifact_root.resolve()
    if probe_output.exists():
        raise BootstrapError("probe output already exists; retry is forbidden")
    claim_path = artifact_root / "claim.json"
    result_path = artifact_root / "result.json"
    if claim_path.exists() or result_path.exists():
        raise BootstrapError(
            "bootstrap claim/result already exists; retry is forbidden"
        )
    gpus = _gpu_snapshots()
    if args.gpu_index not in gpus or gpus[args.gpu_index][0] != args.gpu_uuid:
        raise BootstrapError("requested GPU index/UUID binding mismatch")
    gpu_uuid, free_bytes, total_bytes = gpus[args.gpu_index]
    capacity = min(
        MAX_GPU_SLOTS,
        max(0, free_bytes - GPU_HEADROOM_BYTES) // GPU_SLOT_CLAIM_BYTES,
    )
    if capacity < 1:
        raise BootstrapError("target GPU cannot admit one exact R9 slot")
    total_ram, available_ram = _ram_snapshot()
    used_ram = total_ram - available_ram
    if used_ram * 100 >= total_ram * RAM_ADMISSION_PERCENT:
        raise BootstrapError("bootstrap cannot start at or above 85% RAM")
    descriptors = _lock_all_slots(gpus)
    command = [
        sys.executable,
        str(probe_script),
        "--model-root",
        str(model_root),
        "--device-id",
        "0",
        "--output",
        str(probe_output),
    ]
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_bootstrap_resource_smoke_claim_v1",
        "campaign_id": args.campaign_id,
        "kind": args.kind,
        "controller_path": str(Path(__file__).resolve()),
        "controller_sha256": _sha256(Path(__file__).resolve()),
        "probe_script": str(probe_script),
        "probe_script_sha256": _sha256(probe_script),
        "probe_output": str(probe_output),
        "model_root": str(model_root),
        "command": command,
        "environment": {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "gpu": {
            "physical_index": args.gpu_index,
            "uuid": gpu_uuid,
            "free_bytes_at_admission": free_bytes,
            "total_bytes": total_bytes,
            "slot_capacity_at_admission": capacity,
            "slot_claim_bytes": GPU_SLOT_CLAIM_BYTES,
            "headroom_bytes": GPU_HEADROOM_BYTES,
        },
        "ram": {
            "total_bytes": total_ram,
            "available_bytes_at_admission": available_ram,
            "admission_percent": RAM_ADMISSION_PERCENT,
            "hard_limit_percent": RAM_HARD_LIMIT_PERCENT,
        },
        "global_exclusive_lock": {
            "root": str(LOCK_ROOT),
            "all_physical_gpu_slots_locked": 16,
        },
        "retry_allowed": False,
    }
    claim["bootstrap_claim_sha256"] = _canonical_sha256(claim)
    log_path = artifact_root / "worker.log"
    try:
        _write_exclusive(claim_path, claim)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = os.open(
            log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        environment = dict(os.environ)
        environment.update(claim["environment"])
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        peak_rss = 0
        peak_gpu = 0
        failure_reason: str | None = None
        try:
            while process.poll() is None:
                pids = _process_tree(process.pid)
                peak_rss = max(peak_rss, _tree_rss_bytes(pids))
                peak_gpu = max(peak_gpu, _tree_gpu_bytes(pids, gpu_uuid))
                ram_total, ram_available = _ram_snapshot()
                if (
                    ram_total - ram_available
                ) * 100 >= ram_total * RAM_HARD_LIMIT_PERCENT:
                    failure_reason = "actual system RAM reached the R9 90% hard limit"
                    _terminate(process)
                    break
                time.sleep(0.1)
            returncode = process.wait()
        finally:
            os.close(log_fd)
        status = "succeeded"
        if failure_reason is None and returncode != 0:
            failure_reason = f"probe exited once with status {returncode}"
        if failure_reason is None and not probe_output.is_file():
            failure_reason = "probe output is missing"
        if failure_reason is None and peak_rss <= 0:
            failure_reason = "bootstrap measured no positive process-tree RSS"
        if failure_reason is not None:
            status = "failed"
        result = {
            "schema_version": 1,
            "contract_type": "safa_r9_bootstrap_resource_smoke_result_v1",
            "bootstrap_claim_sha256": claim["bootstrap_claim_sha256"],
            "status": status,
            "failure_reason": failure_reason,
            "returncode": returncode,
            "peak_process_tree_rss_bytes": peak_rss,
            "peak_gpu_memory_bytes": peak_gpu,
            "ram_slot_budget_bytes": (peak_rss * 110 + 99) // 100,
            "gpu_uuid": gpu_uuid,
            "probe_output_sha256": _sha256(probe_output)
            if probe_output.is_file()
            else None,
            "worker_log_sha256": _sha256(log_path),
            "retry_allowed": False,
        }
        result["bootstrap_result_sha256"] = _canonical_sha256(result)
        _write_exclusive(result_path, result)
        print(json.dumps(result, sort_keys=True))
        if failure_reason is not None:
            raise BootstrapError(failure_reason)
        return 0
    finally:
        _release_locks(descriptors)


if __name__ == "__main__":
    raise SystemExit(main())
