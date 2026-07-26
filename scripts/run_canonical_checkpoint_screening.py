#!/usr/bin/env python3
"""Prepare and control the fail-closed historical canonical screening campaign."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    build_preflight_result,
    build_candidate_manifest,
    build_checkpoint_plan,
    canonical_digest,
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
    parser.add_argument("--gpu-uuid")
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
        "preflight_control": policy_root / "preflight_control",
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


def _memory_snapshot_bytes() -> dict[str, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            name, raw = line.split(":", maxsplit=1)
            values[name] = int(raw.strip().split()[0])
    total = values["MemTotal"] * 1024
    available = values["MemAvailable"] * 1024
    if total <= 0 or available < 0 or available > total:
        raise CanonicalScreeningError("/proc/meminfo RAM counters are invalid")
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": total - available,
    }


def _memory_percent() -> float:
    snapshot = _memory_snapshot_bytes()
    return 100.0 * snapshot["used_bytes"] / snapshot["total_bytes"]


def _ram_reservation_projection(
    *,
    total_bytes: int,
    used_bytes: int,
    slot_budget_bytes: int,
    slot_count: int,
    admission_limit_percent: float,
    budget_source: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        type(total_bytes) is not int
        or type(used_bytes) is not int
        or type(slot_budget_bytes) is not int
        or type(slot_count) is not int
        or total_bytes <= 0
        or used_bytes < 0
        or used_bytes > total_bytes
        or slot_budget_bytes <= 0
        or slot_count <= 0
        or not 0.0 < admission_limit_percent < 100.0
    ):
        raise CanonicalScreeningError("RAM reservation inputs are invalid")
    reserved_bytes = slot_budget_bytes * slot_count
    projected_used_bytes = used_bytes + reserved_bytes
    projected_used_percent = 100.0 * projected_used_bytes / total_bytes
    reservation = {
        "slot_count": slot_count,
        "slot_budget_bytes": slot_budget_bytes,
        "reserved_bytes": reserved_bytes,
        "memory_total_bytes": total_bytes,
        "memory_used_bytes": used_bytes,
        "projected_used_bytes": projected_used_bytes,
        "projected_used_percent": projected_used_percent,
        "admission_limit_percent": admission_limit_percent,
        "budget_source": dict(budget_source),
    }
    if projected_used_percent >= admission_limit_percent:
        raise CanonicalScreeningError(
            "RAM reservation admission failed: projected "
            f"{projected_used_percent:.6f}% >= {admission_limit_percent:.0f}% "
            f"for {slot_count} slots x {slot_budget_bytes} bytes"
        )
    return reservation


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
    if resources["ram_budget_status"] != "sealed":
        raise CanonicalScreeningError(
            "GPU admission is blocked until the single-worker RAM probe "
            "and slot budget are sealed"
        )
    cpu_percent = _cpu_load_percent()
    memory = _memory_snapshot_bytes()
    memory_percent = 100.0 * memory["used_bytes"] / memory["total_bytes"]
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
    authorized_gpu_registry = [
        {
            "physical_gpu_index": row["index"],
            "physical_gpu_uuid": row["uuid"],
        }
        for row in gpus
    ]
    if (
        any(
            not row["physical_gpu_uuid"].startswith("GPU-")
            for row in authorized_gpu_registry
        )
        or len({row["physical_gpu_uuid"] for row in authorized_gpu_registry})
        != len(authorized_gpu_registry)
    ):
        raise CanonicalScreeningError("physical GPU UUID registry is invalid")
    slot_count = len(resources["physical_gpus"]) * int(
        resources["workers_per_gpu"]
    )
    ram_reservation = _ram_reservation_projection(
        total_bytes=memory["total_bytes"],
        used_bytes=memory["used_bytes"],
        slot_budget_bytes=int(resources["ram_slot_budget_bytes"]),
        slot_count=slot_count,
        admission_limit_percent=float(resources["ram_admission_percent"]),
        budget_source=resources["ram_slot_budget_source"],
    )
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
        "authorized_gpu_registry": authorized_gpu_registry,
        "ram_reservation": ram_reservation,
        "compute_processes": processes,
    }


def assert_cpu_resource_admission(
    policy: Mapping[str, Any], campaign_root: Path
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
    swap_in, swap_out = _swap_pages()
    return {
        "observed_at": _utc_now(),
        "admission_kind": "cpu_only",
        "cpu_load_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_percent": disk_percent,
        "swap_pages": {"in": swap_in, "out": swap_out},
    }


class CpuWindowState:
    def __init__(self, hard_limit_percent: float, consecutive_limit: int) -> None:
        self.hard_limit_percent = hard_limit_percent
        self.consecutive_limit = consecutive_limit
        self.consecutive_high = 0
        self.window_count = 0
        self.violated = False

    def record(self, cpu_percent: float) -> bool:
        self.window_count += 1
        if self.violated:
            return True
        if cpu_percent >= self.hard_limit_percent:
            self.consecutive_high += 1
        else:
            self.consecutive_high = 0
        if self.consecutive_high >= self.consecutive_limit:
            self.violated = True
        return self.violated


class RuntimeResourceGuard:
    def __init__(
        self,
        policy: Mapping[str, Any],
        sample_path: Path,
        disk_path: Path,
    ) -> None:
        resources = policy["resources"]
        self.policy_sha256 = str(policy["policy_sha256"])
        self.window_seconds = int(resources["cpu_window_seconds"])
        self.poll_seconds = int(resources["resource_poll_seconds"])
        self.cpu_state = CpuWindowState(
            float(resources["cpu_hard_limit_percent"]),
            int(resources["cpu_consecutive_hard_windows"]),
        )
        self.ram_hard_limit_percent = float(resources["ram_hard_limit_percent"])
        self.disk_hard_limit_percent = float(resources["disk_hard_limit_percent"])
        self.disk_path = disk_path
        self.sample_path = sample_path
        self.swap_consecutive_io = 0
        self.swap_consecutive_limit = int(
            resources["swap_consecutive_hard_intervals"]
        )
        self._violation_reason: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="safa-canonical-runtime-resource-guard",
            daemon=True,
        )
        self._lock = threading.Lock()
        self._thread_failure: BaseException | None = None
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def _run(self) -> None:
        try:
            previous_total, previous_idle = _cpu_times()
            cpu_window_started = time.monotonic()
            previous_swap = _swap_pages()
            sequence = 0
            while not self._stop.wait(self.poll_seconds):
                cpu_percent: float | None = None
                now = time.monotonic()
                if now - cpu_window_started >= self.window_seconds:
                    current_total, current_idle = _cpu_times()
                    total_delta = current_total - previous_total
                    idle_delta = current_idle - previous_idle
                    if (
                        total_delta <= 0
                        or idle_delta < 0
                        or idle_delta > total_delta
                    ):
                        raise CanonicalScreeningError(
                            "CPU runtime window counters are invalid"
                        )
                    cpu_percent = (
                        100.0 * (total_delta - idle_delta) / total_delta
                    )
                    previous_total, previous_idle = current_total, current_idle
                    cpu_window_started = now
                memory_percent = _memory_percent()
                disk_percent = _disk_percent(self.disk_path)
                current_swap = _swap_pages()
                sequence += 1
                with self._lock:
                    cpu_violated = (
                        self.cpu_state.record(cpu_percent)
                        if cpu_percent is not None
                        else False
                    )
                    if (
                        current_swap[0] > previous_swap[0]
                        or current_swap[1] > previous_swap[1]
                    ):
                        self.swap_consecutive_io += 1
                    else:
                        self.swap_consecutive_io = 0
                    if cpu_violated and self._violation_reason is None:
                        self._violation_reason = (
                            f"CPU runtime hard stop: "
                            f"{self.cpu_state.consecutive_high} consecutive "
                            f"{self.window_seconds}s windows met or exceeded "
                            f"{self.cpu_state.hard_limit_percent:.0f}%"
                        )
                    if (
                        memory_percent >= self.ram_hard_limit_percent
                        and self._violation_reason is None
                    ):
                        self._violation_reason = (
                            f"RAM runtime hard stop: {memory_percent:.2f}% >= "
                            f"{self.ram_hard_limit_percent:.0f}%"
                        )
                    if (
                        disk_percent >= self.disk_hard_limit_percent
                        and self._violation_reason is None
                    ):
                        self._violation_reason = (
                            f"disk runtime hard stop: {disk_percent:.2f}% >= "
                            f"{self.disk_hard_limit_percent:.0f}%"
                        )
                    if (
                        self.swap_consecutive_io >= self.swap_consecutive_limit
                        and self._violation_reason is None
                    ):
                        self._violation_reason = (
                            "sustained swap I/O observed for "
                            f"{self.swap_consecutive_limit} consecutive "
                            f"{self.poll_seconds}s resource intervals"
                        )
                    consecutive = self.cpu_state.consecutive_high
                    swap_consecutive = self.swap_consecutive_io
                    violation_reason = self._violation_reason
                sample = {
                    "schema_version": 1,
                    "contract_type": "safa_canonical_runtime_resource_window_v1",
                    "policy_sha256": self.policy_sha256,
                    "sequence": sequence,
                    "resource_poll_seconds": self.poll_seconds,
                    "window_seconds": self.window_seconds,
                    "cpu_percent": cpu_percent,
                    "cpu_hard_limit_percent": self.cpu_state.hard_limit_percent,
                    "cpu_consecutive_high": consecutive,
                    "cpu_consecutive_limit": self.cpu_state.consecutive_limit,
                    "memory_percent": memory_percent,
                    "ram_hard_limit_percent": self.ram_hard_limit_percent,
                    "disk_percent": disk_percent,
                    "disk_hard_limit_percent": self.disk_hard_limit_percent,
                    "swap_pages": {"in": current_swap[0], "out": current_swap[1]},
                    "swap_consecutive_io": swap_consecutive,
                    "swap_consecutive_limit": self.swap_consecutive_limit,
                    "violation_reason": violation_reason,
                    "violated": violation_reason is not None,
                    "completed_at": _utc_now(),
                }
                sample["resource_window_sha256"] = hashlib.sha256(
                    canonical_json(sample)
                ).hexdigest()
                _append_jsonl(self.sample_path, sample)
                previous_swap = current_swap
        except BaseException as exc:
            with self._lock:
                self._thread_failure = exc

    def raise_if_violated(self) -> None:
        with self._lock:
            violation_reason = self._violation_reason
            thread_failure = self._thread_failure
        if thread_failure is not None:
            raise CanonicalScreeningError(
                "runtime resource guard failed: "
                f"{type(thread_failure).__name__}: {thread_failure}"
            )
        if violation_reason is not None:
            raise CanonicalScreeningError(violation_reason)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._started:
            self._thread.join()
        with self._lock:
            summary = {
                "started": self._started,
                "window_seconds": self.window_seconds,
                "resource_poll_seconds": self.poll_seconds,
                "cpu_hard_limit_percent": self.cpu_state.hard_limit_percent,
                "cpu_consecutive_limit": self.cpu_state.consecutive_limit,
                "window_count": self.cpu_state.window_count,
                "final_cpu_consecutive_high": self.cpu_state.consecutive_high,
                "final_swap_consecutive_io": self.swap_consecutive_io,
                "swap_consecutive_limit": self.swap_consecutive_limit,
                "violation_reason": self._violation_reason,
                "violated": self._violation_reason is not None,
                "thread_failure": (
                    None
                    if self._thread_failure is None
                    else {
                        "type": type(self._thread_failure).__name__,
                        "message": str(self._thread_failure),
                    }
                ),
            }
        summary["samples"] = (
            {
                "path": str(self.sample_path.resolve()),
                "sha256": sha256_file(self.sample_path),
            }
            if self.sample_path.is_file()
            else None
        )
        return summary


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
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    resource_guard: RuntimeResourceGuard,
    startup_admission_sha256: str,
) -> dict[str, int]:
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("CPU checkpoint preflight must run inside tmux")
    requests = sorted(paths["preflight_requests"].glob("*.json"))
    if not requests:
        raise CanonicalScreeningError("no checkpoint preflight requests exist")
    paths["preflight_results"].mkdir(parents=True, exist_ok=True)
    if any(paths["preflight_results"].glob("*.json")):
        raise CanonicalScreeningError(
            "current-policy preflight refuses result reuse; use a new policy namespace"
        )
    completed = 0
    valid = 0
    invalid = 0
    attempt_root = paths["preflight_control"] / "attempts"
    for sequence, request_path in enumerate(requests, start=1):
        resource_guard.raise_if_violated()
        request = _load_preflight_request(request_path, policy)
        result_path = paths["preflight_results"] / request_path.name
        claim = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_attempt_claim_v1",
            "policy_sha256": policy["policy_sha256"],
            "sequence": sequence,
            "request_path": str(request_path.resolve()),
            "preflight_request_sha256": request["preflight_request_sha256"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "checkpoint_model": request["checkpoint_model"],
            "startup_admission_sha256": startup_admission_sha256,
            "started_at": _utc_now(),
            "worker_pid": os.getpid(),
        }
        claim["attempt_claim_sha256"] = hashlib.sha256(
            canonical_json(claim)
        ).hexdigest()
        claim_path = attempt_root / f"{request_path.stem}.claim.json"
        terminal_path = attempt_root / f"{request_path.stem}.terminal.json"
        write_exclusive_json(claim_path, claim)
        try:
            checkpoint = _root(Path(str(request["checkpoint_path"])))
            strict_result = preflight_generator_checkpoint(
                checkpoint,
                str(request["checkpoint_model"]),
                "cpu",
                expected_checkpoint_sha256=str(request["checkpoint_sha256"]),
                compute_sha256=True,
                smoke_samples=0,
                output_decoder_registry=request["output_decoder_registry"],
            )
            resource_guard.raise_if_violated()
            result = build_preflight_result(request, policy, strict_result)
            is_valid, _ = validate_preflight_result(result, request, policy)
            write_exclusive_json(result_path, result)
            terminal = {
                "schema_version": 1,
                "contract_type": "safa_canonical_preflight_attempt_terminal_v1",
                "policy_sha256": policy["policy_sha256"],
                "attempt_claim_sha256": claim["attempt_claim_sha256"],
                "preflight_request_sha256": request["preflight_request_sha256"],
                "status": "completed",
                "valid": is_valid,
                "result_path": str(result_path.resolve()),
                "result_file_sha256": sha256_file(result_path),
                "preflight_result_sha256": result["preflight_result_sha256"],
                "failure": None,
                "completed_at": _utc_now(),
            }
            terminal["attempt_terminal_sha256"] = hashlib.sha256(
                canonical_json(terminal)
            ).hexdigest()
            write_exclusive_json(terminal_path, terminal)
            completed += 1
            if is_valid:
                valid += 1
            else:
                invalid += 1
        except BaseException as exc:
            terminal = {
                "schema_version": 1,
                "contract_type": "safa_canonical_preflight_attempt_terminal_v1",
                "policy_sha256": policy["policy_sha256"],
                "attempt_claim_sha256": claim["attempt_claim_sha256"],
                "preflight_request_sha256": request["preflight_request_sha256"],
                "status": "failed",
                "valid": None,
                "result_path": None,
                "result_file_sha256": None,
                "preflight_result_sha256": None,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "completed_at": _utc_now(),
            }
            terminal["attempt_terminal_sha256"] = hashlib.sha256(
                canonical_json(terminal)
            ).hexdigest()
            write_exclusive_json(terminal_path, terminal)
            raise
        _append_monitor_sample(policy, paths, "preflight")
        resource_guard.raise_if_violated()
    _append_monitor_sample(policy, paths, "preflight", terminal=True)
    return {
        "request_count": len(requests),
        "completed": completed,
        "reused": 0,
        "valid": valid,
        "invalid": invalid,
    }


def _execute_preflight_controller(
    policy: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("CPU checkpoint preflight must run inside tmux")
    startup_snapshot = assert_cpu_resource_admission(policy, paths["root"])
    startup_admission = _write_admission(
        policy, paths, "preflight_cpu_startup", startup_snapshot
    )
    control = paths["preflight_control"]
    claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_controller_claim_v1",
        "campaign_id": policy["campaign_id"],
        "policy_sha256": policy["policy_sha256"],
        "supersedes_policy_sha256": policy["supersedes_policy_sha256"],
        "startup_admission": startup_admission,
        "request_count": len(list(paths["preflight_requests"].glob("*.json"))),
        "controller_pid": os.getpid(),
        "started_at": _utc_now(),
        "external_timeout_seconds": None,
    }
    claim["controller_claim_sha256"] = hashlib.sha256(
        canonical_json(claim)
    ).hexdigest()
    claim_path = control / "controller_claim.json"
    terminal_path = control / "controller_terminal.json"
    summary_path = control / "controller_summary.json"
    log_path = control / "controller.log"
    write_exclusive_json(claim_path, claim)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] | None = None
    caught: BaseException | None = None
    resource_guard = RuntimeResourceGuard(
        policy,
        control / "runtime_resource_windows.jsonl",
        paths["root"].parent,
    )
    with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                resource_guard.start()
                print(canonical_json({"event": "controller_started", **claim}).decode(), end="")
                materialized = materialize_preflights(
                    policy,
                    paths,
                    resource_guard,
                    startup_admission["canonical_sha256"],
                )
                refreshed = build_checkpoint_plan(
                    REPO_ROOT, policy, paths["preflight_results"]
                )
                final_plan = _final_plan_path(paths, policy)
                write_exclusive_json(final_plan, refreshed)
                validate_checkpoint_plan(
                    refreshed,
                    repo_root=REPO_ROOT,
                    policy=policy,
                    preflight_root=paths["preflight_results"],
                )
                summary = {
                    "preflight": materialized,
                    "counts": refreshed["counts"],
                    "final_plan": {
                        "path": str(final_plan.resolve()),
                        "sha256": sha256_file(final_plan),
                        "canonical_sha256": refreshed["checkpoint_plan_sha256"],
                    },
                }
            except BaseException as exc:
                caught = exc
                print(
                    canonical_json(
                        {
                            "event": "controller_exception",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ).decode(),
                    end="",
                )
            finally:
                resource_guard_summary = resource_guard.stop()
                if (
                    caught is None
                    and resource_guard_summary["thread_failure"] is not None
                ):
                    failure = resource_guard_summary["thread_failure"]
                    caught = CanonicalScreeningError(
                        "runtime resource guard failed before controller terminal: "
                        f"{failure['type']}: {failure['message']}"
                    )
                    summary = None
                if (
                    caught is None
                    and resource_guard_summary["violation_reason"] is not None
                ):
                    caught = CanonicalScreeningError(
                        str(resource_guard_summary["violation_reason"])
                    )
                    summary = None
                result_count = len(list(paths["preflight_results"].glob("*.json")))
                attempt_claim_count = len(
                    list((control / "attempts").glob("*.claim.json"))
                )
                attempt_terminal_count = len(
                    list((control / "attempts").glob("*.terminal.json"))
                )
                terminal = {
                    "schema_version": 1,
                    "contract_type": "safa_canonical_preflight_controller_terminal_v1",
                    "policy_sha256": policy["policy_sha256"],
                    "controller_claim_sha256": claim["controller_claim_sha256"],
                    "status": "completed" if caught is None else "failed",
                    "result_count": result_count,
                    "pending_count": claim["request_count"] - result_count,
                    "attempt_claim_count": attempt_claim_count,
                    "attempt_terminal_count": attempt_terminal_count,
                    "runtime_resource_guard": resource_guard_summary,
                    "controller_monitor_samples": (
                        {
                            "path": str(
                                (paths["logs"] / "preflight__monitor.jsonl").resolve()
                            ),
                            "sha256": sha256_file(
                                paths["logs"] / "preflight__monitor.jsonl"
                            ),
                        }
                        if (
                            paths["logs"] / "preflight__monitor.jsonl"
                        ).is_file()
                        else None
                    ),
                    "failure": (
                        None
                        if caught is None
                        else {"type": type(caught).__name__, "message": str(caught)}
                    ),
                    "completed_at": _utc_now(),
                }
                terminal["controller_terminal_sha256"] = hashlib.sha256(
                    canonical_json(terminal)
                ).hexdigest()
                write_exclusive_json(terminal_path, terminal)
                if summary is not None:
                    summary_value = {
                        "schema_version": 1,
                        "contract_type": "safa_canonical_preflight_controller_summary_v1",
                        "policy_sha256": policy["policy_sha256"],
                        "controller_claim_sha256": claim["controller_claim_sha256"],
                        "controller_terminal_sha256": terminal[
                            "controller_terminal_sha256"
                        ],
                        **summary,
                    }
                    summary_value["controller_summary_sha256"] = hashlib.sha256(
                        canonical_json(summary_value)
                    ).hexdigest()
                    write_exclusive_json(summary_path, summary_value)
                print(canonical_json({"event": "controller_terminal", **terminal}).decode(), end="")
    if caught is not None:
        raise caught
    if summary is None:
        raise CanonicalScreeningError("preflight controller produced no summary")
    return summary


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


def _validate_smoke_per_sample_rows(
    request: Mapping[str, Any],
    per_sample_path: Path,
    expected_manifest_ids: Sequence[str],
) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    rows = load_jsonl(per_sample_path, "smoke8 per-sample evidence")
    required = {
        "sample_id",
        "run_request_sha256",
        "checkpoint_sha256",
        "checkpoint_model",
        "source_path",
        "source_sha256",
        "candidate_path",
        "candidate_sha256",
        "output_contract_sha256",
        "output_contract_type",
        "decoder_registry_sha256",
        "output_space",
        "native_output_sha256",
        "native_output_shape",
        "native_rgb_shape",
        "native_rgb_size",
        "quality_protocol_family",
        "nfe",
        "e0_cosine",
        "edev_cosine",
        "arcface_source_face_count",
        "arcface_candidate_face_count",
        "arcface_source_candidate_cosine",
    }
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if (
        sample_ids != list(expected_manifest_ids)
        or len(sample_ids) != 8
        or len(sample_ids) != len(set(sample_ids))
    ):
        raise CanonicalScreeningError(
            "smoke8 per-sample IDs differ from the frozen ordered manifest"
        )
    contract = request["output_contract"]
    capability = contract["capability"]
    native_tensor = capability["generator_output_tensor"]
    native_shape = [
        native_tensor["channels"],
        native_tensor["height"],
        native_tensor["width"],
    ]
    rgb_contract = contract["rgb_contract"]
    rgb_shape = [
        rgb_contract["channels"],
        rgb_contract["height"],
        rgb_contract["width"],
    ]
    expected_bindings = {
        "run_request_sha256": request["run_request_sha256"],
        "checkpoint_sha256": request["candidate"]["checkpoint_sha256"],
        "checkpoint_model": request["candidate"]["checkpoint_model"],
        "output_contract_sha256": contract["output_contract_sha256"],
        "output_contract_type": contract["contract_type"],
        "decoder_registry_sha256": request["output_decoder_registry"][
            "decoder_registry_sha256"
        ],
        "output_space": capability["output_space"],
        "native_output_shape": native_shape,
        "native_rgb_shape": rgb_shape,
        "native_rgb_size": request["native_rgb_size"],
        "quality_protocol_family": request["quality_protocol_family"],
        "nfe": request["nfe"],
    }
    deterministic = []
    for index, row in enumerate(rows):
        if set(row) != required:
            raise CanonicalScreeningError(
                "smoke8 per-sample evidence fields differ"
            )
        for field, expected in expected_bindings.items():
            if row[field] != expected:
                raise CanonicalScreeningError(
                    f"smoke8 per-sample {field} binding differs"
                )
        for field in (
            "source_sha256",
            "candidate_sha256",
            "native_output_sha256",
        ):
            digest = row[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise CanonicalScreeningError(
                    f"smoke8 per-sample {field} is invalid"
                )
        source_path = Path(str(row["source_path"])).resolve()
        candidate_path = Path(str(row["candidate_path"])).resolve()
        expected_candidate = (
            Path(str(request["output_dir"])).resolve()
            / "generated"
            / f"{index:06d}.png"
        )
        if (
            not source_path.is_file()
            or sha256_file(source_path) != row["source_sha256"]
            or candidate_path != expected_candidate
            or not candidate_path.is_file()
            or sha256_file(candidate_path) != row["candidate_sha256"]
        ):
            raise CanonicalScreeningError(
                "smoke8 per-sample image binding differs"
            )
        for field in ("e0_cosine", "edev_cosine"):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                raise CanonicalScreeningError(
                    f"smoke8 per-sample {field} is non-finite"
                )
        source_faces = row["arcface_source_face_count"]
        candidate_faces = row["arcface_candidate_face_count"]
        if (
            type(source_faces) is not int
            or source_faces < 0
            or type(candidate_faces) is not int
            or candidate_faces < 0
        ):
            raise CanonicalScreeningError(
                "smoke8 per-sample ArcFace counts are invalid"
            )
        arcface_cosine = row["arcface_source_candidate_cosine"]
        if source_faces == candidate_faces == 1:
            if not isinstance(arcface_cosine, (int, float)) or not math.isfinite(
                float(arcface_cosine)
            ):
                raise CanonicalScreeningError(
                    "smoke8 per-sample ArcFace cosine is invalid"
                )
        elif arcface_cosine is not None:
            raise CanonicalScreeningError(
                "smoke8 per-sample ArcFace coverage/cosine differs"
            )
        deterministic.append(
            (
                row["sample_id"],
                row["candidate_sha256"],
                row["native_output_sha256"],
                row["output_contract_sha256"],
                tuple(int(item) for item in row["native_rgb_shape"]),
            )
        )
    return deterministic


def _require_smoke_success(
    policy: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    expected_manifest_ids = [
        str(row["sample_id"])
        for row in load_jsonl(
            Path(str(policy["protocol"]["manifests"]["smoke8"]["path"])),
            "frozen smoke8 manifest",
        )
    ]
    if (
        len(expected_manifest_ids) != 8
        or len(expected_manifest_ids) != len(set(expected_manifest_ids))
    ):
        raise CanonicalScreeningError("frozen smoke8 manifest IDs are invalid")
    for candidate in candidate_manifest["candidates"]:
        determinism_rows: list[list[tuple[str, str, str, str, tuple[int, ...]]]] = []
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
            rows = _validate_smoke_per_sample_rows(
                request,
                per_sample_path,
                expected_manifest_ids,
            )
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
    controller_invocation = " ".join(
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
    )
    if phase == "preflight":
        controller = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "safa-screening-preflight-controller",
            "-c",
            str(REPO_ROOT),
            python,
            str(REPO_ROOT / "scripts/run_canonical_preflight_wrapper.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--config",
            str(config.resolve()),
            "--campaign-root",
            str(campaign_root.resolve()),
            "--policy-sha256",
            str(policy["policy_sha256"]),
            "--python",
            python,
        ]
    else:
        controller = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            f"safa-screening-{phase}-controller",
            controller_invocation,
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
    admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cpu_only = phase == "preflight"
    gpu_rows = None if cpu_only else _gpu_snapshot()
    gpu_binding = None
    if not cpu_only:
        if admission is None:
            admission_paths = sorted(paths["admissions"].glob(f"{phase}__*.json"))
            if len(admission_paths) != 1:
                raise CanonicalScreeningError(
                    f"{phase} monitor requires exactly one admission artifact"
                )
            admission_path = admission_paths[0]
            admission_value = load_json(admission_path, "resource admission")
            admission = {
                "path": str(admission_path.resolve()),
                "sha256": sha256_file(admission_path),
                "canonical_sha256": admission_value["admission_sha256"],
            }
        admission_path = Path(str(admission["path"])).resolve()
        if (
            not admission_path.is_file()
            or sha256_file(admission_path) != admission["sha256"]
        ):
            raise CanonicalScreeningError("monitor admission file binding mismatch")
        admission_value = load_json(admission_path, "resource admission")
        if (
            admission_value.get("admission_sha256")
            != admission["canonical_sha256"]
            or canonical_digest(admission_value, "admission_sha256")
            != admission["canonical_sha256"]
            or admission_value.get("policy_sha256") != policy["policy_sha256"]
        ):
            raise CanonicalScreeningError("monitor admission contract mismatch")
        snapshot = admission_value.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise CanonicalScreeningError("monitor admission snapshot is invalid")
        authorized_registry = snapshot.get("authorized_gpu_registry")
        runtime_registry = [
            {
                "physical_gpu_index": row["index"],
                "physical_gpu_uuid": row["uuid"],
            }
            for row in gpu_rows
            if row["index"] in policy["resources"]["physical_gpus"]
        ]
        if runtime_registry != authorized_registry:
            raise CanonicalScreeningError(
                "monitor physical GPU UUID registry differs from admission"
            )
        gpu_binding = {
            "admission_sha256": admission["canonical_sha256"],
            "authorized_gpu_registry": authorized_registry,
            "runtime_gpu_registry": runtime_registry,
            "ram_reservation": snapshot.get("ram_reservation"),
        }
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
        "gpus": gpu_rows,
        "gpu_binding": gpu_binding,
        "compute_processes": None if cpu_only else _gpu_compute_processes(),
        "logs": log_rows,
        "artifacts": _artifact_progress(paths, phase),
    }


def _append_monitor_sample(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    *,
    terminal: bool = False,
    admission: Mapping[str, Any] | None = None,
) -> Path:
    path = paths["logs"] / f"{phase}__monitor.jsonl"
    _append_jsonl(
        path,
        _monitor_sample(
            policy,
            paths,
            phase,
            terminal=terminal,
            admission=admission,
        ),
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


def _gpu_hard_resource_violation(
    policy: Mapping[str, Any],
) -> str | None:
    resources = policy["resources"]
    for gpu in _gpu_snapshot():
        if gpu["index"] not in resources["physical_gpus"]:
            continue
        percent = 100.0 * gpu["memory_used_mib"] / gpu["memory_total_mib"]
        if percent >= 90.0:
            return (
                f"GPU{gpu['index']} memory hard limit reached: {percent:.2f}%"
            )
        if gpu["temperature_c"] > 85:
            return (
                f"GPU{gpu['index']} temperature hard limit reached: "
                f"{gpu['temperature_c']}C"
            )
    return None


def _claim_slot(
    lock_root: Path,
    gpu_index: int,
    gpu_uuid: str,
    slot_index: int,
    request: Path,
    admission: Mapping[str, Any],
    ram_slot_budget_bytes: int,
) -> Path:
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / f"gpu{gpu_index}.slot{slot_index}.json"
    payload = {
        "physical_gpu_index": gpu_index,
        "physical_gpu_uuid": gpu_uuid,
        "logical_cuda_index": 0,
        "cuda_visible_devices": gpu_uuid,
        "slot_index": slot_index,
        "ram_slot_budget_bytes": ram_slot_budget_bytes,
        "admission_sha256": admission["canonical_sha256"],
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
    gpu_uuid: str,
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
        "--gpu-uuid",
        gpu_uuid,
    ]


def _worker_environment(gpu_uuid: str) -> dict[str, str]:
    if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise CanonicalScreeningError("worker physical GPU UUID is invalid")
    return {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
    }


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
    gpu_uuid_by_index = {
        row["physical_gpu_index"]: row["physical_gpu_uuid"]
        for row in admission_snapshot["authorized_gpu_registry"]
    }
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
    stop_reason: str | None = None
    unexpected: BaseException | None = None
    monitor_path = _append_monitor_sample(
        policy, paths, phase, admission=admission
    )
    resource_guard = RuntimeResourceGuard(
        policy,
        paths["logs"] / f"{phase}__runtime_resource_windows.jsonl",
        paths["root"].parent,
    )
    try:
        resource_guard.start()
        while request_queue or active:
            resource_guard.raise_if_violated()
            while request_queue and slot_pool.free_count:
                request = request_queue.popleft()
                gpu, slot_index = slot_pool.acquire()
                gpu_uuid = gpu_uuid_by_index[gpu]
                lock: Path | None = None
                log_handle: Any | None = None
                try:
                    lock = _claim_slot(
                        lock_root,
                        gpu,
                        gpu_uuid,
                        slot_index,
                        request,
                        admission,
                        int(policy["resources"]["ram_slot_budget_bytes"]),
                    )
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
                            gpu_uuid,
                        ),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=_worker_environment(gpu_uuid),
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
            violation = _gpu_hard_resource_violation(policy)
            if violation is not None:
                failures.append(violation)
                stop_reason = "resource_hard_stop"
                break
            _append_monitor_sample(
                policy, paths, phase, admission=admission
            )
            if active:
                time.sleep(10)
    except BaseException as exc:
        unexpected = exc
        stop_reason = "controller_exception"
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        resource_guard_summary = resource_guard.stop()
        late_resource_failure = (
            resource_guard_summary["violation_reason"]
            or (
                "runtime resource guard failed: "
                f"{resource_guard_summary['thread_failure']['type']}: "
                f"{resource_guard_summary['thread_failure']['message']}"
                if resource_guard_summary["thread_failure"] is not None
                else None
            )
        )
        if late_resource_failure is not None and late_resource_failure not in failures:
            failures.append(late_resource_failure)
            stop_reason = "resource_hard_stop"
        if failures or unexpected is not None:
            _cleanup_active_workers(active, slot_pool)
        _append_monitor_sample(
            policy, paths, phase, terminal=True, admission=admission
        )
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
            "runtime_resource_guard": resource_guard_summary,
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
        "runtime_resource_guard": resource_guard_summary,
        "capability_completion": {
            output_space: {
                "request_count": sum(
                    1
                    for request_path in requests
                    if load_json(
                        request_path, "completed run request"
                    )["output_contract"]["capability"]["output_space"]
                    == output_space
                ),
                "completed_count": sum(
                    1
                    for request_path in requests
                    if load_json(
                        Path(
                            load_json(
                                request_path, "completed run request"
                            )["output_dir"]
                        )
                        / "result.json",
                        "completed run result",
                    )["status"]
                    == "completed"
                    and load_json(
                        request_path, "completed run request"
                    )["output_contract"]["capability"]["output_space"]
                    == output_space
                ),
            }
            for output_space in ("latent", "pixel")
        },
    }
    for output_space, completion in summary["capability_completion"].items():
        if completion["request_count"] == 0 or (
            completion["completed_count"] != completion["request_count"]
        ):
            raise CanonicalScreeningError(
                f"{phase} capability completion differs for {output_space}: "
                f"{completion}"
            )
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
        if (
            args.gpu_index is None
            or args.gpu_uuid is None
            or args.phase not in {"smoke8", "screen512"}
        ):
            raise CanonicalScreeningError(
                "--request requires --gpu-index, --gpu-uuid, and a GPU "
                "screening phase"
            )
        if args.dry_run:
            raise CanonicalScreeningError("worker requests cannot use --dry-run")
        execute_screening_request(
            _root(args.request), args.gpu_index, args.gpu_uuid, policy
        )
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
        summary = _execute_preflight_controller(policy, paths)
        print(json.dumps(summary, sort_keys=True))
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
