#!/usr/bin/env python3
"""Prepare and control the fail-closed historical canonical screening campaign."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    build_preflight_result,
    build_candidate_manifest,
    build_checkpoint_plan,
    canonical_json,
    iter_run_requests,
    load_json,
    load_jsonl,
    sha256_file,
    validate_candidate_manifest,
    validate_checkpoint_plan,
    validate_policy,
    validate_preflight_request,
    validate_preflight_result,
    validate_run_claim,
    validate_run_request,
    validate_run_result,
    write_exclusive_json,
    write_preflight_requests,
)
from safa.closeout.canonical_screening_worker import execute_screening_request
from safa.evaluation.checkpoint_preflight import preflight_generator_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "plan",
    "prepare",
    "preflight",
    "prepare-screening",
    "smoke8",
    "screen512",
    "monitor",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control the canonical all-checkpoint 512 screening campaign."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("artifacts/closeout/historical-canonical-512-v1"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--monitor-target", choices=("preflight", "smoke8", "screen512"))
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root(path: Path) -> Path:
    value = path if path.is_absolute() else REPO_ROOT / path
    return value.resolve()


def _paths(campaign_root: Path, policy_sha256: str) -> dict[str, Path]:
    policy_root = campaign_root / "by_policy" / policy_sha256
    return {
        "root": campaign_root,
        "policy_root": policy_root,
        "preflight_requests": policy_root / "checkpoint_preflight" / "requests",
        "preflight_results": policy_root / "checkpoint_preflight" / "results",
        "checkpoint_plan": policy_root / "checkpoint_plan.json",
        "candidate_manifest": policy_root / "candidate_manifest.json",
        "run_requests": policy_root / "run_requests",
        "runs": policy_root / "runs",
        "logs": policy_root / "logs",
        "admissions": policy_root / "admissions",
        "summaries": policy_root / "summaries",
    }


def _final_plan_path(paths: Mapping[str, Path], policy: Mapping[str, Any]) -> Path:
    return (
        paths["root"]
        / f"checkpoint_plan_final__{str(policy['policy_sha256'])[:16]}.json"
    )


def _candidate_manifest_path(
    paths: Mapping[str, Path], policy: Mapping[str, Any]
) -> Path:
    return (
        paths["root"]
        / f"candidate_manifest__{str(policy['policy_sha256'])[:16]}.json"
    )


def _memory_percent() -> float:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            name, raw = line.split(":", maxsplit=1)
            values[name] = int(raw.strip().split()[0])
    total = values["MemTotal"]
    available = values["MemAvailable"]
    return 100.0 * (total - available) / total


def _cpu_times() -> tuple[int, int]:
    line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    fields = line.split()
    if fields[0] != "cpu" or len(fields) < 6:
        raise CanonicalScreeningError("/proc/stat aggregate CPU row is invalid")
    values = [int(value) for value in fields[1:]]
    total = sum(values)
    idle = values[3] + values[4]
    return total, idle


def _cpu_load_percent() -> float:
    total_before, idle_before = _cpu_times()
    time.sleep(0.1)
    total_after, idle_after = _cpu_times()
    total_delta = total_after - total_before
    idle_delta = idle_after - idle_before
    if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
        raise CanonicalScreeningError("aggregate CPU utilization sample is invalid")
    return 100.0 * (total_delta - idle_delta) / total_delta


def _disk_percent(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return 100.0 * usage.used / usage.total


def _gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            raise CanonicalScreeningError(f"unexpected nvidia-smi GPU row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "temperature_c": int(fields[5]),
            }
        )
    return rows


def _gpu_compute_processes() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise CanonicalScreeningError(
                f"unexpected nvidia-smi process row: {line!r}"
            )
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": fields[3],
            }
        )
    return rows


def assert_resource_admission(
    policy: Mapping[str, Any], campaign_root: Path, *, require_idle_gpus: bool
) -> dict[str, Any]:
    resources = policy["resources"]
    cpu_percent = _cpu_load_percent()
    memory_percent = _memory_percent()
    disk_percent = _disk_percent(campaign_root.parent)
    if cpu_percent >= float(resources["cpu_admission_percent"]):
        raise CanonicalScreeningError(
            f"CPU admission failed: {cpu_percent:.2f}% >= "
            f"{resources['cpu_admission_percent']}%"
        )
    if memory_percent >= float(resources["ram_admission_percent"]):
        raise CanonicalScreeningError(
            f"RAM admission failed: {memory_percent:.2f}% >= "
            f"{resources['ram_admission_percent']}%"
        )
    if disk_percent >= float(resources["disk_admission_percent"]):
        raise CanonicalScreeningError(
            f"disk admission failed: {disk_percent:.2f}% >= "
            f"{resources['disk_admission_percent']}%"
        )
    gpus = [
        row for row in _gpu_snapshot() if row["index"] in resources["physical_gpus"]
    ]
    if [row["index"] for row in gpus] != resources["physical_gpus"]:
        raise CanonicalScreeningError("physical GPU 0..3 registry is unavailable")
    required_free_mib = int(resources["gpu_headroom_bytes"]) // 1024**2
    blocked = [row for row in gpus if row["memory_free_mib"] < required_free_mib]
    if blocked:
        raise CanonicalScreeningError(f"GPU headroom admission failed: {blocked}")
    target_uuids = {row["uuid"] for row in gpus}
    processes = [
        row for row in _gpu_compute_processes() if row["gpu_uuid"] in target_uuids
    ]
    if require_idle_gpus and processes:
        raise CanonicalScreeningError(
            "GPU execution is blocked because compute PIDs already exist; "
            f"busy override is forbidden: {processes}"
        )
    return {
        "observed_at": _utc_now(),
        "cpu_load_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_percent": disk_percent,
        "swap_pages": {
            "in": _swap_pages()[0],
            "out": _swap_pages()[1],
        },
        "gpus": gpus,
        "compute_processes": processes,
    }


def _write_admission(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_resource_admission_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "snapshot": dict(snapshot),
    }
    value["admission_sha256"] = hashlib.sha256(
        canonical_json(value)
    ).hexdigest()
    path = paths["admissions"] / f"{phase}__{value['admission_sha256']}.json"
    write_exclusive_json(path, value)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "canonical_sha256": value["admission_sha256"],
    }


def _load_preflight_request(
    path: Path, policy: Mapping[str, Any]
) -> dict[str, Any]:
    return validate_preflight_request(
        load_json(path, "checkpoint preflight request"), policy
    )


def materialize_preflights(
    policy: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, int]:
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("CPU checkpoint preflight must run inside tmux")
    requests = sorted(paths["preflight_requests"].glob("*.json"))
    if not requests:
        raise CanonicalScreeningError("no checkpoint preflight requests exist")
    paths["preflight_results"].mkdir(parents=True, exist_ok=True)
    completed = 0
    reused = 0
    valid = 0
    invalid = 0
    for request_path in requests:
        request = _load_preflight_request(request_path, policy)
        result_path = paths["preflight_results"] / request_path.name
        if result_path.is_file():
            reused += 1
            is_valid, _ = validate_preflight_result(
                load_json(result_path, "checkpoint preflight result"),
                request,
                policy,
            )
            if is_valid:
                valid += 1
            else:
                invalid += 1
            continue
        assert_resource_admission(policy, paths["root"], require_idle_gpus=False)
        checkpoint = _root(Path(str(request["checkpoint_path"])))
        strict_result = preflight_generator_checkpoint(
            checkpoint,
            str(request["checkpoint_model"]),
            "cpu",
            expected_checkpoint_sha256=str(request["checkpoint_sha256"]),
            compute_sha256=True,
            smoke_samples=0,
        )
        result = build_preflight_result(request, policy, strict_result)
        validate_preflight_result(result, request, policy)
        write_exclusive_json(result_path, result)
        completed += 1
        if strict_result["status"] == "valid":
            valid += 1
        else:
            invalid += 1
        _append_monitor_sample(policy, paths, "preflight")
    _append_monitor_sample(policy, paths, "preflight", terminal=True)
    return {
        "request_count": len(requests),
        "completed": completed,
        "reused": reused,
        "valid": valid,
        "invalid": invalid,
    }


def _write_run_requests(
    policy: Mapping[str, Any],
    policy_path: Path,
    candidate_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    mode: str,
    replicate: str,
    admission: Mapping[str, Any],
) -> list[Path]:
    request_root = paths["run_requests"] / f"{mode}_{replicate}"
    written: list[Path] = []
    candidate_manifest_path = _candidate_manifest_path(paths, policy)
    for request in iter_run_requests(
        policy,
        policy_path,
        candidate_manifest,
        candidate_manifest_path,
        mode,
        replicate,
        paths["runs"],
        admission,
    ):
        validate_run_request(request, policy)
        path = request_root / f"{request['candidate']['candidate_id']}.json"
        write_exclusive_json(path, request)
        written.append(path)
    return written


def _require_smoke_success(
    policy: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    for candidate in candidate_manifest["candidates"]:
        determinism_rows: list[list[tuple[str, str]]] = []
        for replicate in ("primary", "repeat"):
            request_path = (
                paths["run_requests"]
                / f"smoke8_{replicate}"
                / f"{candidate['candidate_id']}.json"
            )
            request = validate_run_request(
                load_json(request_path, f"smoke8 {replicate} request"), policy
            )
            if (
                request["candidate"] != candidate
                or request["replicate"] != replicate
                or request["candidate_manifest"]["canonical_sha256"]
                != candidate_manifest["candidate_manifest_sha256"]
            ):
                raise CanonicalScreeningError(
                    "screen512 is blocked by stale smoke8 request: "
                    f"{candidate['candidate_id']}/{replicate}"
                )
            output_dir = (
                paths["runs"] / f"smoke8_{replicate}" / candidate["candidate_id"]
            )
            claim = validate_run_claim(
                load_json(output_dir / "claim.json", f"smoke8 {replicate} claim"),
                request,
                policy,
            )
            result = validate_run_result(
                load_json(output_dir / "result.json", f"smoke8 {replicate} result"),
                request,
                claim,
                policy,
            )
            if result["status"] != "completed":
                raise CanonicalScreeningError(
                    "screen512 is blocked by smoke8 result: "
                    f"{candidate['candidate_id']}/{replicate}"
                )
            per_sample_path = output_dir / "per_sample.jsonl"
            if sha256_file(per_sample_path) != result["evidence"]["per_sample_sha256"]:
                raise CanonicalScreeningError("smoke8 per-sample digest mismatch")
            rows = [
                (str(row["sample_id"]), str(row["candidate_sha256"]))
                for row in load_jsonl(per_sample_path, f"smoke8 {replicate} per-sample")
            ]
            if len(rows) != 8:
                raise CanonicalScreeningError("smoke8 per-sample coverage must be 8")
            determinism_rows.append(rows)
        if determinism_rows[0] != determinism_rows[1]:
            raise CanonicalScreeningError(
                "screen512 is blocked by batch2 repeat determinism mismatch: "
                f"{candidate['candidate_id']}"
            )


def _tmux_commands(
    policy: Mapping[str, Any], config: Path, campaign_root: Path, phase: str
) -> dict[str, list[str]]:
    python = str(policy["python"])
    script = str(Path(__file__).resolve())
    controller = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        f"safa-screening-{phase}-controller",
        " ".join(
            [
                python,
                script,
                "--config",
                str(config.resolve()),
                "--campaign-root",
                str(campaign_root.resolve()),
                "--phase",
                phase,
                "--execute",
            ]
        ),
    ]
    monitor = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        f"safa-screening-{phase}-monitor",
        " ".join(
            [
                python,
                script,
                "--config",
                str(config.resolve()),
                "--campaign-root",
                str(campaign_root.resolve()),
                "--phase",
                "monitor",
                "--monitor-target",
                phase,
                "--execute",
            ]
        ),
    ]
    return {"controller": controller, "monitor": monitor}


def _run_monitor(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    target: str,
) -> dict[str, Any]:
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("resource monitor must run inside tmux")
    controller_session = f"safa-screening-{target}-controller"
    path = paths["logs"] / f"{target}__observer.jsonl"
    samples = 0
    while True:
        exists = subprocess.run(
            ["tmux", "has-session", "-t", controller_session],
            capture_output=True,
            text=True,
        ).returncode == 0
        sample = _monitor_sample(policy, paths, target, terminal=not exists)
        _append_jsonl(path, sample)
        samples += 1
        if not exists:
            break
        time.sleep(30)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "samples": samples}


def _swap_pages() -> tuple[int, int]:
    values = {"pswpin": 0, "pswpout": 0}
    with Path("/proc/vmstat").open("r", encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split()
            if key in values:
                values[key] = int(raw)
    return values["pswpin"], values["pswpout"]


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    content = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        if os.write(descriptor, content) != len(content):
            raise CanonicalScreeningError("short append to monitor log")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_progress(paths: Mapping[str, Path], phase: str) -> dict[str, int]:
    roots = {
        "request_files": paths["run_requests"],
        "result_files": paths["runs"],
        "generated_png": paths["runs"],
        "preflight_requests": paths["preflight_requests"],
        "preflight_results": paths["preflight_results"],
    }
    patterns = {
        "request_files": "*.json",
        "result_files": "result.json",
        "generated_png": "*.png",
        "preflight_requests": "*.json",
        "preflight_results": "*.json",
    }
    return {
        name: (
            sum(1 for _ in root.rglob(patterns[name])) if root.exists() else 0
        )
        for name, root in roots.items()
    }


def _monitor_sample(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    *,
    terminal: bool,
) -> dict[str, Any]:
    log_rows = []
    if paths["logs"].exists():
        for path in sorted(paths["logs"].glob(f"{phase}*__*.log")):
            stat = path.stat()
            log_rows.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "contract_type": "safa_canonical_resource_monitor_sample_v1",
        "observed_at": _utc_now(),
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "terminal": terminal,
        "cpu_load_percent": _cpu_load_percent(),
        "memory_percent": _memory_percent(),
        "disk_percent": _disk_percent(paths["root"].parent),
        "swap_pages": {"in": _swap_pages()[0], "out": _swap_pages()[1]},
        "gpus": _gpu_snapshot(),
        "compute_processes": _gpu_compute_processes(),
        "logs": log_rows,
        "artifacts": _artifact_progress(paths, phase),
    }


def _append_monitor_sample(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    *,
    terminal: bool = False,
) -> Path:
    path = paths["logs"] / f"{phase}__monitor.jsonl"
    _append_jsonl(
        path,
        _monitor_sample(policy, paths, phase, terminal=terminal),
    )
    return path


class FreeSlotPool:
    def __init__(self, slots: Sequence[tuple[int, int]]) -> None:
        if not slots or len(set(slots)) != len(slots):
            raise CanonicalScreeningError("GPU slot registry is empty or repeated")
        self._all = frozenset(slots)
        self._free = set(slots)

    def acquire(self) -> tuple[int, int]:
        if not self._free:
            raise CanonicalScreeningError("no free GPU slot is available")
        slot = min(self._free)
        self._free.remove(slot)
        return slot

    def release(self, slot: tuple[int, int]) -> None:
        if slot not in self._all or slot in self._free:
            raise CanonicalScreeningError(f"invalid GPU slot release: {slot}")
        self._free.add(slot)

    @property
    def free_count(self) -> int:
        return len(self._free)


def _cleanup_active_workers(
    active: list[dict[str, Any]], slot_pool: FreeSlotPool
) -> None:
    for item in active:
        process = item["process"]
        if process.poll() is None:
            process.terminate()
    for item in active:
        process = item["process"]
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        item["log_handle"].close()
        item["lock"].unlink(missing_ok=True)
        slot_pool.release(item["slot"])
    active.clear()


def _hard_resource_violation(
    policy: Mapping[str, Any],
    campaign_root: Path,
    previous_swap: tuple[int, int],
    sustained_swap_intervals: int,
) -> tuple[str | None, tuple[int, int], int]:
    resources = policy["resources"]
    cpu_percent = _cpu_load_percent()
    memory_percent = _memory_percent()
    disk_percent = _disk_percent(campaign_root.parent)
    if cpu_percent >= float(resources["cpu_hard_limit_percent"]):
        return (
            f"CPU hard limit reached: {cpu_percent:.2f}%",
            previous_swap,
            sustained_swap_intervals,
        )
    if memory_percent >= float(resources["ram_hard_limit_percent"]):
        return (
            f"RAM hard limit reached: {memory_percent:.2f}%",
            previous_swap,
            sustained_swap_intervals,
        )
    if disk_percent >= float(resources["disk_hard_limit_percent"]):
        return (
            f"disk hard limit reached: {disk_percent:.2f}%",
            previous_swap,
            sustained_swap_intervals,
        )
    for gpu in _gpu_snapshot():
        if gpu["index"] not in resources["physical_gpus"]:
            continue
        percent = 100.0 * gpu["memory_used_mib"] / gpu["memory_total_mib"]
        if percent >= 90.0:
            return (
                f"GPU{gpu['index']} memory hard limit reached: {percent:.2f}%",
                previous_swap,
                sustained_swap_intervals,
            )
        if gpu["temperature_c"] > 85:
            return (
                f"GPU{gpu['index']} temperature hard limit reached: "
                f"{gpu['temperature_c']}C",
                previous_swap,
                sustained_swap_intervals,
            )
    current_swap = _swap_pages()
    if current_swap[0] > previous_swap[0] or current_swap[1] > previous_swap[1]:
        sustained_swap_intervals += 1
    else:
        sustained_swap_intervals = 0
    if sustained_swap_intervals >= 3:
        return (
            "sustained swap I/O observed for three resource intervals",
            current_swap,
            sustained_swap_intervals,
        )
    return None, current_swap, sustained_swap_intervals


def _claim_slot(lock_root: Path, gpu_index: int, slot_index: int, request: Path) -> Path:
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / f"gpu{gpu_index}.slot{slot_index}.json"
    payload = {
        "gpu_index": gpu_index,
        "slot_index": slot_index,
        "request": str(request.resolve()),
        "controller_pid": os.getpid(),
        "claimed_at": _utc_now(),
    }
    write_exclusive_json(path, payload)
    return path


def _worker_command(
    policy: Mapping[str, Any],
    config: Path,
    campaign_root: Path,
    phase: str,
    request: Path,
    gpu_index: int,
) -> list[str]:
    return [
        str(policy["python"]),
        str(Path(__file__).resolve()),
        "--config",
        str(config.resolve()),
        "--campaign-root",
        str(campaign_root.resolve()),
        "--phase",
        phase,
        "--execute",
        "--request",
        str(request.resolve()),
        "--gpu-index",
        str(gpu_index),
    ]


def _run_gpu_phase(
    policy: Mapping[str, Any],
    config: Path,
    paths: Mapping[str, Path],
    phase: str,
) -> dict[str, Any]:
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("GPU screening controller must run inside tmux")
    admission_snapshot = assert_resource_admission(
        policy, paths["root"], require_idle_gpus=True
    )
    admission = _write_admission(policy, paths, phase, admission_snapshot)
    candidate_manifest = load_json(
        _candidate_manifest_path(paths, policy), "candidate manifest"
    )
    plan_path = Path(str(candidate_manifest["checkpoint_plan"]["path"]))
    plan = validate_checkpoint_plan(
        load_json(plan_path, "checkpoint plan"),
        repo_root=REPO_ROOT,
        policy=policy,
        preflight_root=paths["preflight_results"],
    )
    validate_candidate_manifest(
        candidate_manifest,
        policy=policy,
        plan=plan,
        plan_path=plan_path,
        repo_root=REPO_ROOT,
        preflight_root=paths["preflight_results"],
    )
    if phase == "screen512":
        _require_smoke_success(policy, candidate_manifest, paths)
    requests: list[Path] = []
    replicates = ("primary", "repeat") if phase == "smoke8" else ("primary",)
    for replicate in replicates:
        requests.extend(
            _write_run_requests(
                policy,
                config,
                candidate_manifest,
                paths,
                phase,
                replicate,
                admission,
            )
        )
    gpus = list(policy["resources"]["physical_gpus"])
    capacity = int(policy["resources"]["workers_per_gpu"])
    slots = [(gpu, slot) for gpu in gpus for slot in range(capacity)]
    slot_pool = FreeSlotPool(slots)
    lock_root = Path(str(policy["resources"]["global_lock_root"]))
    paths["logs"].mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    active: list[dict[str, Any]] = []
    request_queue = deque(requests)
    previous_swap = _swap_pages()
    sustained_swap_intervals = 0
    stop_reason: str | None = None
    unexpected: BaseException | None = None
    monitor_path = _append_monitor_sample(policy, paths, phase)
    try:
        while request_queue or active:
            while request_queue and slot_pool.free_count:
                request = request_queue.popleft()
                gpu, slot_index = slot_pool.acquire()
                lock: Path | None = None
                log_handle: Any | None = None
                try:
                    lock = _claim_slot(lock_root, gpu, slot_index, request)
                    log_path = (
                        paths["logs"]
                        / f"{request.parent.name}__{request.stem}.log"
                    )
                    log_handle = log_path.open("x", encoding="utf-8")
                    process = subprocess.Popen(
                        _worker_command(
                            policy,
                            config,
                            paths["root"],
                            phase,
                            request,
                            gpu,
                        ),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                except BaseException:
                    if log_handle is not None:
                        log_handle.close()
                    if lock is not None:
                        lock.unlink(missing_ok=True)
                    slot_pool.release((gpu, slot_index))
                    raise
                active.append(
                    {
                        "process": process,
                        "request": request,
                        "lock": lock,
                        "log_handle": log_handle,
                        "slot": (gpu, slot_index),
                    }
                )
            for item in list(active):
                process = item["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                item["log_handle"].close()
                item["lock"].unlink()
                slot_pool.release(item["slot"])
                active.remove(item)
                if return_code != 0:
                    failures.append(
                        f"{item['request']}: exit_code={return_code}"
                    )
            if failures:
                stop_reason = "worker_nonzero_exit"
                break
            violation, previous_swap, sustained_swap_intervals = _hard_resource_violation(
                policy,
                paths["root"],
                previous_swap,
                sustained_swap_intervals,
            )
            if violation is not None:
                failures.append(violation)
                stop_reason = "resource_hard_stop"
                break
            _append_monitor_sample(policy, paths, phase)
            if active:
                time.sleep(10)
    except BaseException as exc:
        unexpected = exc
        stop_reason = "controller_exception"
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        if failures or unexpected is not None:
            _cleanup_active_workers(active, slot_pool)
        _append_monitor_sample(policy, paths, phase, terminal=True)
    if failures:
        stop = {
            "phase": phase,
            "stopped_at": _utc_now(),
            "reason": stop_reason,
            "failures": failures,
            "admission": admission,
            "monitor_log": {
                "path": str(monitor_path.resolve()),
                "sha256": sha256_file(monitor_path),
            },
        }
        stop["controller_summary_sha256"] = hashlib.sha256(
            canonical_json(stop)
        ).hexdigest()
        write_exclusive_json(
            paths["summaries"] / f"{phase}__failed.json", stop
        )
        raise CanonicalScreeningError(
            "GPU screening failed without retry or batch change: " + " | ".join(failures)
        )
    summary = {
        "phase": phase,
        "completed_at": _utc_now(),
        "request_count": len(requests),
        "failures": [],
        "admission": admission,
        "monitor_log": {
            "path": str(monitor_path.resolve()),
            "sha256": sha256_file(monitor_path),
        },
    }
    summary["controller_summary_sha256"] = hashlib.sha256(
        canonical_json(summary)
    ).hexdigest()
    write_exclusive_json(
        paths["summaries"] / f"{phase}__completed.json", summary
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _root(args.config)
    campaign_root = _root(args.campaign_root)
    policy = validate_policy(REPO_ROOT, config)
    paths = _paths(campaign_root, policy["policy_sha256"])
    if args.phase == "monitor":
        if args.dry_run or args.monitor_target is None:
            raise CanonicalScreeningError(
                "monitor requires --execute and --monitor-target"
            )
        print(json.dumps(_run_monitor(policy, paths, args.monitor_target), sort_keys=True))
        return 0
    if args.request is not None:
        if args.gpu_index is None or args.phase not in {"smoke8", "screen512"}:
            raise CanonicalScreeningError(
                "--request requires --gpu-index and a GPU screening phase"
            )
        if args.dry_run:
            raise CanonicalScreeningError("worker requests cannot use --dry-run")
        execute_screening_request(_root(args.request), args.gpu_index, policy)
        return 0
    plan = build_checkpoint_plan(REPO_ROOT, policy, paths["preflight_results"])

    if args.dry_run:
        payload: dict[str, Any] = {
            "phase": args.phase,
            "execute": False,
            "policy_sha256": policy["policy_sha256"],
            "checkpoint_plan_sha256": plan["checkpoint_plan_sha256"],
            "counts": plan["counts"],
            "tmux": (
                _tmux_commands(policy, config, campaign_root, args.phase)
                if args.phase in {"preflight", "smoke8", "screen512"}
                else None
            ),
        }
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0

    if args.phase == "plan":
        raise CanonicalScreeningError("plan is read-only; use --dry-run")
    if args.phase == "prepare":
        write_exclusive_json(paths["checkpoint_plan"], plan)
        validate_checkpoint_plan(
            plan,
            repo_root=REPO_ROOT,
            policy=policy,
            preflight_root=paths["preflight_results"],
        )
        request_paths = write_preflight_requests(
            plan, paths["preflight_requests"]
        )
        print(
            json.dumps(
                {
                    "checkpoint_plan": str(paths["checkpoint_plan"]),
                    "preflight_requests": len(request_paths),
                    "counts": plan["counts"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.phase == "preflight":
        summary = materialize_preflights(policy, paths)
        refreshed = build_checkpoint_plan(
            REPO_ROOT, policy, paths["preflight_results"]
        )
        write_exclusive_json(_final_plan_path(paths, policy), refreshed)
        validate_checkpoint_plan(
            refreshed,
            repo_root=REPO_ROOT,
            policy=policy,
            preflight_root=paths["preflight_results"],
        )
        print(json.dumps({"preflight": summary, "counts": refreshed["counts"]}, sort_keys=True))
        return 0
    if args.phase == "prepare-screening":
        final_plan_path = _final_plan_path(paths, policy)
        if final_plan_path.is_file():
            final_plan = validate_checkpoint_plan(
                load_json(final_plan_path, "final checkpoint plan"),
                repo_root=REPO_ROOT,
                policy=policy,
                preflight_root=paths["preflight_results"],
            )
        else:
            final_plan = build_checkpoint_plan(
                REPO_ROOT, policy, paths["preflight_results"]
            )
            write_exclusive_json(final_plan_path, final_plan)
        manifest = build_candidate_manifest(
            policy,
            final_plan,
            plan_path=final_plan_path,
            repo_root=REPO_ROOT,
            preflight_root=paths["preflight_results"],
        )
        candidate_manifest_path = _candidate_manifest_path(paths, policy)
        write_exclusive_json(candidate_manifest_path, manifest)
        validate_candidate_manifest(
            manifest,
            policy=policy,
            plan=final_plan,
            plan_path=final_plan_path,
            repo_root=REPO_ROOT,
            preflight_root=paths["preflight_results"],
        )
        print(
            json.dumps(
                {
                    "candidate_manifest": str(candidate_manifest_path),
                    "candidate_count": manifest["candidate_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    result = _run_gpu_phase(policy, config, paths, args.phase)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CanonicalScreeningError as exc:
        print(f"CANONICAL SCREENING BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
