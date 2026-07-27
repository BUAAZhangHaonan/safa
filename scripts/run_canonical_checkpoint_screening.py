#!/usr/bin/env python3
"""Prepare and control the fail-closed historical canonical screening campaign."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from safa.closeout.canonical_screening import (
        CONTROLLER_LAUNCH_REHASH_CONTRACT,
        CanonicalScreeningError,
        WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        WORKER_RELEASE_CONTRACT,
        build_candidate_manifest,
        build_checkpoint_plan,
        build_preflight_result,
        canonical_digest,
        canonical_json,
        iter_run_requests,
        load_json,
        load_jsonl,
        publish_exclusive_json,
        sha256_file,
        validate_candidate_manifest,
        validate_checkpoint_plan,
        validate_controller_launch_rehash_value,
        validate_policy,
        validate_preflight_request,
        validate_preflight_result,
        validate_run_claim,
        validate_run_request,
        validate_run_result,
        validate_worker_ready_value,
        validate_worker_release_value,
        validate_worker_terminal_value,
        write_exclusive_json,
        write_preflight_requests,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_EXPORTS = (
    "CanonicalScreeningError",
    "WORKER_EXTERNAL_GPU_RACE_CONTRACT",
    "WORKER_RELEASE_CONTRACT",
    "CONTROLLER_LAUNCH_REHASH_CONTRACT",
    "build_preflight_result",
    "build_candidate_manifest",
    "build_checkpoint_plan",
    "canonical_digest",
    "canonical_json",
    "iter_run_requests",
    "load_json",
    "load_jsonl",
    "publish_exclusive_json",
    "sha256_file",
    "validate_candidate_manifest",
    "validate_checkpoint_plan",
    "validate_policy",
    "validate_preflight_request",
    "validate_preflight_result",
    "validate_run_claim",
    "validate_run_request",
    "validate_run_result",
    "validate_worker_ready_value",
    "validate_worker_release_value",
    "validate_worker_terminal_value",
    "validate_controller_launch_rehash_value",
    "write_exclusive_json",
    "write_preflight_requests",
)
_CONTRACT_API_INSTALLED = False
PHASES = (
    "plan",
    "prepare",
    "preflight",
    "prepare-screening",
    "smoke8",
    "screen512",
    "monitor",
)


class ControllerBootstrapError(RuntimeError):
    """Raised before any policy-bound SAFA implementation is imported."""


if not TYPE_CHECKING:
    CanonicalScreeningError = ControllerBootstrapError


def _stdlib_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stdlib_validate_implementation_bindings(
    config: Path,
) -> tuple[dict[str, Any], str]:
    resolved_config = config.resolve()
    try:
        raw = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerBootstrapError(
            f"controller bootstrap cannot read policy: {resolved_config}"
        ) from exc
    if not isinstance(raw, dict):
        raise ControllerBootstrapError("controller bootstrap policy is not a mapping")
    implementations = raw.get("implementations")
    if not isinstance(implementations, dict) or not implementations:
        raise ControllerBootstrapError(
            "controller bootstrap policy omits implementations"
        )
    required = {"screening_contracts", "screening_worker", "controller"}
    if not required.issubset(implementations):
        raise ControllerBootstrapError(
            "controller bootstrap policy omits trust-boundary implementations"
        )
    root = REPO_ROOT.resolve()
    normalized: dict[str, dict[str, str]] = {}
    for name, value in implementations.items():
        if not isinstance(name, str) or not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
        }:
            raise ControllerBootstrapError(
                f"controller bootstrap implementation binding differs: {name!r}"
            )
        raw_path = Path(str(value["path"]))
        path = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ControllerBootstrapError(
                f"controller bootstrap implementation escapes repository: {name}"
            ) from exc
        expected = value["sha256"]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not path.is_file()
            or _stdlib_sha256_file(path) != expected
        ):
            raise ControllerBootstrapError(
                f"controller bootstrap implementation digest differs: {name}"
            )
        normalized[name] = {"path": str(path), "sha256": expected}
    if Path(normalized["controller"]["path"]) != Path(__file__).resolve():
        raise ControllerBootstrapError(
            "controller bootstrap policy does not bind this controller"
        )
    return normalized, _stdlib_sha256_file(resolved_config)


def _install_verified_contract_api(
    config: Path,
    *,
    verify_historical_output_evidence: bool = True,
) -> dict[str, Any]:
    global _CONTRACT_API_INSTALLED
    implementations, config_sha256 = _stdlib_validate_implementation_bindings(
        config
    )
    contracts_path = Path(implementations["screening_contracts"]["path"])
    module = importlib.import_module("safa.closeout.canonical_screening")
    if Path(module.__file__).resolve() != contracts_path:
        raise ControllerBootstrapError(
            "imported screening contracts path differs from policy binding"
        )
    missing = [name for name in _CONTRACT_EXPORTS if not hasattr(module, name)]
    if missing:
        raise ControllerBootstrapError(
            f"screening contracts omit controller API: {missing}"
        )
    for name in _CONTRACT_EXPORTS:
        globals()[name] = getattr(module, name)
    policy = module.validate_policy(
        REPO_ROOT,
        config.resolve(),
        verify_historical_output_evidence=verify_historical_output_evidence,
    )
    if (
        _stdlib_sha256_file(config.resolve()) != config_sha256
        or {
            name: {
                "path": str(Path(binding["path"]).resolve()),
                "sha256": binding["sha256"],
            }
            for name, binding in policy["implementations"].items()
        }
        != implementations
    ):
        raise ControllerBootstrapError(
            "policy or implementation bindings changed during controller bootstrap"
        )
    _CONTRACT_API_INSTALLED = True
    return policy


def _stdlib_canonical_digest(
    value: Mapping[str, Any], excluded_field: str
) -> str:
    payload = {
        key: item for key, item in dict(value).items() if key != excluded_field
    }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stdlib_publish_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = destination.parent / (
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        if os.write(descriptor, payload) != len(payload):
            raise ControllerBootstrapError(
                f"short bootstrap terminal write: {temporary}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _write_stdlib_main_bootstrap_failure(
    campaign_root: Path,
    config: Path,
    phase: str,
    failure: BaseException,
) -> None:
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_main_bootstrap_terminal_v1",
        "phase": phase,
        "config": {
            "path": str(config.resolve()),
            "sha256": (
                _stdlib_sha256_file(config.resolve()) if config.is_file() else None
            ),
        },
        "status": "failed",
        "failure": {
            "type": type(failure).__name__,
            "message": str(failure),
        },
        "completed_at": _utc_now(),
    }
    value["main_bootstrap_terminal_sha256"] = _stdlib_canonical_digest(
        value, "main_bootstrap_terminal_sha256"
    )
    _stdlib_publish_exclusive_json(
        campaign_root.resolve()
        / "bootstrap_control"
        / phase
        / "main_bootstrap_terminal.json",
        value,
    )


def _stdlib_optional_artifact_binding(
    path: Path | None, digest_field: str | None = None
) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        return {
            "path": str(resolved),
            "sha256": None,
            "canonical_sha256": None,
        }
    canonical_sha256 = None
    if digest_field is not None:
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            value = {}
        candidate = (
            value.get(digest_field) if isinstance(value, dict) else None
        )
        if (
            isinstance(candidate, str)
            and len(candidate) == 64
            and all(character in "0123456789abcdef" for character in candidate)
        ):
            canonical_sha256 = candidate
    return {
        "path": str(resolved),
        "sha256": _stdlib_sha256_file(resolved),
        "canonical_sha256": canonical_sha256,
    }


def _write_stdlib_worker_bootstrap_failure(
    *,
    config: Path,
    phase: str,
    request_path: Path | None,
    worker_ready_path: Path,
    worker_release_path: Path | None,
    stage: str,
    failure: BaseException,
) -> None:
    resolved_config = config.resolve()
    policy_sha256 = None
    if resolved_config.is_file():
        try:
            raw_policy = json.loads(
                resolved_config.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raw_policy = {}
        candidate = (
            raw_policy.get("policy_sha256")
            if isinstance(raw_policy, dict)
            else None
        )
        if isinstance(candidate, str):
            policy_sha256 = candidate
    value = {
        "schema_version": 1,
        "contract_type":
            "safa_canonical_worker_bootstrap_terminal_v1",
        "policy_sha256": policy_sha256,
        "phase": phase,
        "worker_pid": os.getpid(),
        "stage": stage,
        "config": _stdlib_optional_artifact_binding(config),
        "request": _stdlib_optional_artifact_binding(
            request_path, "run_request_sha256"
        ),
        "worker_ready_path": str(worker_ready_path.resolve()),
        "worker_release_path": (
            None
            if worker_release_path is None
            else str(worker_release_path.resolve())
        ),
        "status": "failed",
        "failure": {
            "type": type(failure).__name__,
            "message": str(failure),
        },
        "completed_at": _utc_now(),
    }
    value["worker_bootstrap_terminal_sha256"] = (
        _stdlib_canonical_digest(
            value, "worker_bootstrap_terminal_sha256"
        )
    )
    _stdlib_publish_exclusive_json(
        worker_ready_path.resolve().parent
        / "worker_bootstrap_terminal.json",
        value,
    )


def preflight_generator_checkpoint(*args: Any, **kwargs: Any) -> Any:
    from safa.evaluation.checkpoint_preflight import (
        preflight_generator_checkpoint as implementation,
    )

    return implementation(*args, **kwargs)


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
    parser.add_argument("--final-release-path", type=Path)
    parser.add_argument("--final-release-sha256")
    parser.add_argument("--final-release-canonical-sha256")
    parser.add_argument("--worker-ready-path", type=Path)
    parser.add_argument("--worker-release-path", type=Path)
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
        "gpu_control": policy_root / "gpu_control",
        "request_intents": policy_root / "request_intents",
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
    swap_in, swap_out = _swap_pages()
    return {
        "observed_at": _utc_now(),
        "cpu_load_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_percent": disk_percent,
        "swap_pages": {
            "in": swap_in,
            "out": swap_out,
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
        authorized_gpu_registry: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        resources = policy["resources"]
        self.policy_resources = dict(resources)
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
        self.authorized_gpu_registry = (
            None
            if authorized_gpu_registry is None
            else [dict(row) for row in authorized_gpu_registry]
        )
        if self.authorized_gpu_registry is not None:
            expected_indices = resources["physical_gpus"]
            if (
                [
                    row.get("physical_gpu_index")
                    for row in self.authorized_gpu_registry
                ]
                != expected_indices
                or any(
                    not isinstance(row.get("physical_gpu_uuid"), str)
                    or not row["physical_gpu_uuid"].startswith("GPU-")
                    for row in self.authorized_gpu_registry
                )
            ):
                raise CanonicalScreeningError(
                    "runtime guard GPU registry is invalid"
                )
        self._active_worker_pids: set[int] = set()
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
        self._lock = threading.RLock()
        self._thread_failure: BaseException | None = None
        self._started = False
        self._first_sample = threading.Event()
        self._stop_summary: dict[str, Any] | None = None

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def launch_authorized_worker(
        self, factory: Any
    ) -> subprocess.Popen[Any]:
        with self._lock:
            if self._thread_failure is not None:
                raise CanonicalScreeningError(
                    "runtime resource guard failed before worker launch"
                )
            if self._violation_reason is not None:
                raise CanonicalScreeningError(self._violation_reason)
            self._sample_gpu_state_locked()
            if self._violation_reason is not None:
                raise CanonicalScreeningError(self._violation_reason)
            process = factory()
            if type(process.pid) is not int or process.pid <= 0:
                process.terminate()
                process.wait()
                raise CanonicalScreeningError(
                    "worker launch returned an invalid PID"
                )
            self._active_worker_pids.add(process.pid)
            return process

    def launch_cpu_worker(self, factory: Any, validator: Any) -> Any:
        with self._lock:
            if self._thread_failure is not None:
                raise CanonicalScreeningError(
                    "runtime resource guard failed before CPU worker launch"
                )
            if self._violation_reason is not None:
                raise CanonicalScreeningError(self._violation_reason)
            validation = validator()
            self._sample_gpu_state_locked()
            if self._violation_reason is not None:
                raise CanonicalScreeningError(self._violation_reason)
            process = factory()
            if type(process.pid) is not int or process.pid <= 0:
                process.terminate()
                process.wait()
                raise CanonicalScreeningError(
                    "CPU worker launch returned an invalid PID"
                )
            return process, validation

    def release_worker_after_handshake(
        self,
        worker_pid: int,
        validator: Any,
        publisher: Any,
    ) -> Any:
        with self._lock:
            if worker_pid in self._active_worker_pids:
                raise CanonicalScreeningError(
                    f"worker PID is already registered: {worker_pid}"
                )
            self._active_worker_pids.add(worker_pid)
            try:
                if self._thread_failure is not None:
                    raise CanonicalScreeningError(
                        "runtime resource guard failed before worker release"
                    ) from self._thread_failure
                if self._violation_reason is not None:
                    raise CanonicalScreeningError(self._violation_reason)
                validation = validator()
                swap_before = _swap_pages()
                cpu_load_percent = _cpu_load_percent()
                memory_percent = _memory_percent()
                disk_percent = _disk_percent(self.disk_path)
                swap_after = _swap_pages()
                gpu_rows = _gpu_snapshot()
                compute_rows = _gpu_compute_processes()
                (
                    runtime_registry,
                    compute_processes,
                    unknown_processes,
                ) = self._sample_gpu_state_locked(
                    gpu_rows=gpu_rows,
                    compute_processes=compute_rows,
                )
                if self._violation_reason is not None:
                    raise CanonicalScreeningError(self._violation_reason)
                authorized_indices = {
                    row["physical_gpu_index"]
                    for row in self.authorized_gpu_registry or []
                }
                authorized_gpu_rows = [
                    row for row in gpu_rows
                    if row["index"] in authorized_indices
                ]
                swap_delta = {
                    "in": swap_after[0] - swap_before[0],
                    "out": swap_after[1] - swap_before[1],
                }
                hard_limits = {
                    "cpu_percent": self.cpu_state.hard_limit_percent,
                    "ram_percent": self.ram_hard_limit_percent,
                    "disk_percent": self.disk_hard_limit_percent,
                    "gpu_memory_percent": 90.0,
                    "gpu_temperature_c": 85,
                    "gpu_free_mib": int(
                        self.policy_resources["gpu_headroom_bytes"]
                    ) // 1024**2,
                    "swap_io_delta_pages": 0,
                    "swap_consecutive_io": 0,
                }
                violation = self._release_resource_violation(
                    cpu_load_percent=cpu_load_percent,
                    memory_percent=memory_percent,
                    disk_percent=disk_percent,
                    swap_delta=swap_delta,
                    gpu_rows=authorized_gpu_rows,
                )
                if violation is not None:
                    self._violation_reason = violation
                    raise CanonicalScreeningError(violation)
                resource_snapshot = {
                    "schema_version": 1,
                    "contract_type":
                        "safa_canonical_worker_release_resource_snapshot_v2",
                    "policy_sha256": self.policy_sha256,
                    "observed_at": _utc_now(),
                    "runtime_gpu_registry": runtime_registry,
                    "compute_processes": compute_processes,
                    "unknown_compute_processes": unknown_processes,
                    "cpu_load_percent": cpu_load_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "swap_pages_before": {
                        "in": swap_before[0], "out": swap_before[1]
                    },
                    "swap_pages_after": {
                        "in": swap_after[0], "out": swap_after[1]
                    },
                    "swap_io_delta": swap_delta,
                    "swap_consecutive_io": self.swap_consecutive_io,
                    "gpu": authorized_gpu_rows,
                    "active_worker_pids": sorted(
                        self._active_worker_pids
                    ),
                    "hard_limits": hard_limits,
                    "guard_thread_failure": None,
                    "guard_violation_reason": None,
                }
                publisher(validation, resource_snapshot)
                return validation
            except BaseException:
                self._active_worker_pids.remove(worker_pid)
                raise

    def unregister_worker_pid(self, worker_pid: int) -> None:
        with self._lock:
            if worker_pid not in self._active_worker_pids:
                raise CanonicalScreeningError(
                    f"runtime guard worker PID is not registered: {worker_pid}"
                )
            self._active_worker_pids.remove(worker_pid)

    def _release_resource_violation(
        self,
        *,
        cpu_load_percent: float,
        memory_percent: float,
        disk_percent: float,
        swap_delta: Mapping[str, int],
        gpu_rows: Sequence[Mapping[str, Any]],
    ) -> str | None:
        if cpu_load_percent >= self.cpu_state.hard_limit_percent:
            return (
                "CPU release hard gate failed: "
                f"{cpu_load_percent:.2f}% >= "
                f"{self.cpu_state.hard_limit_percent:.0f}%"
            )
        if memory_percent >= self.ram_hard_limit_percent:
            return (
                "RAM release hard gate failed: "
                f"{memory_percent:.2f}% >= "
                f"{self.ram_hard_limit_percent:.0f}%"
            )
        if disk_percent >= self.disk_hard_limit_percent:
            return (
                "disk release hard gate failed: "
                f"{disk_percent:.2f}% >= "
                f"{self.disk_hard_limit_percent:.0f}%"
            )
        if (
            swap_delta["in"] != 0
            or swap_delta["out"] != 0
            or self.swap_consecutive_io != 0
        ):
            return "swap I/O release hard gate failed"
        for gpu in gpu_rows:
            percent = (
                100.0
                * gpu["memory_used_mib"]
                / gpu["memory_total_mib"]
            )
            if percent >= 90.0:
                return (
                    f"GPU{gpu['index']} release memory hard gate failed: "
                    f"{percent:.2f}% >= 90%"
                )
            if gpu["temperature_c"] > 85:
                return (
                    f"GPU{gpu['index']} release temperature hard gate failed: "
                    f"{gpu['temperature_c']}C > 85C"
                )
            required_free_mib = int(
                self.policy_resources["gpu_headroom_bytes"]
            ) // 1024**2
            if gpu["memory_free_mib"] < required_free_mib:
                return (
                    f"GPU{gpu['index']} release headroom hard gate failed: "
                    f"{gpu['memory_free_mib']} MiB < "
                    f"{required_free_mib} MiB"
                )
        return None

    def _sample_gpu_state_locked(
        self,
        *,
        gpu_rows: Sequence[Mapping[str, Any]] | None = None,
        compute_processes: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[
        list[dict[str, Any]] | None,
        list[dict[str, Any]] | None,
        list[dict[str, Any]] | None,
    ]:
        if self.authorized_gpu_registry is None:
            return None, None, None
        authorized_indices = [
            row["physical_gpu_index"]
            for row in self.authorized_gpu_registry
        ]
        authorized_uuids = {
            row["physical_gpu_uuid"]
            for row in self.authorized_gpu_registry
        }
        observed_gpu_rows = [
            dict(row)
            for row in (
                _gpu_snapshot() if gpu_rows is None else gpu_rows
            )
            if row["index"] in authorized_indices
        ]
        runtime_gpu_registry = [
            {
                "physical_gpu_index": row["index"],
                "physical_gpu_uuid": row["uuid"],
            }
            for row in observed_gpu_rows
        ]
        observed_compute_processes = [
            dict(row)
            for row in (
                _gpu_compute_processes()
                if compute_processes is None
                else compute_processes
            )
            if row["gpu_uuid"] in authorized_uuids
        ]
        unknown_compute_processes = [
            row
            for row in observed_compute_processes
            if row["pid"] not in self._active_worker_pids
        ]
        if (
            runtime_gpu_registry != self.authorized_gpu_registry
            and self._violation_reason is None
        ):
            self._violation_reason = (
                "runtime GPU UUID registry drifted from admission"
            )
        if unknown_compute_processes and self._violation_reason is None:
            self._violation_reason = (
                "unknown compute PID observed on GPU0-3: "
                f"{unknown_compute_processes}"
            )
        return (
            runtime_gpu_registry,
            observed_compute_processes,
            unknown_compute_processes,
        )

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
                    runtime_gpu_registry = None
                    compute_processes = None
                    unknown_compute_processes = None
                    active_worker_pids = sorted(self._active_worker_pids)
                    if self.authorized_gpu_registry is not None:
                        (
                            runtime_gpu_registry,
                            compute_processes,
                            unknown_compute_processes,
                        ) = self._sample_gpu_state_locked()
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
                    "authorized_gpu_registry": self.authorized_gpu_registry,
                    "runtime_gpu_registry": runtime_gpu_registry,
                    "active_worker_pids": active_worker_pids,
                    "compute_processes": compute_processes,
                    "unknown_compute_processes": unknown_compute_processes,
                    "violation_reason": violation_reason,
                    "violated": violation_reason is not None,
                    "completed_at": _utc_now(),
                }
                sample["resource_window_sha256"] = hashlib.sha256(
                    canonical_json(sample)
                ).hexdigest()
                _append_jsonl(self.sample_path, sample)
                self._first_sample.set()
                previous_swap = current_swap
        except BaseException as exc:
            with self._lock:
                self._thread_failure = exc
            self._first_sample.set()

    def wait_first_sample(self, timeout_seconds: float) -> dict[str, Any]:
        if not self._started:
            raise CanonicalScreeningError("runtime resource guard is not started")
        if not self._first_sample.wait(timeout_seconds):
            raise CanonicalScreeningError(
                "runtime resource guard first sample timed out"
            )
        self.raise_if_violated()
        rows = load_jsonl(self.sample_path, "runtime resource guard samples")
        first = rows[0]
        if (
            first.get("contract_type")
            != "safa_canonical_runtime_resource_window_v1"
            or first.get("policy_sha256") != self.policy_sha256
            or first.get("sequence") != 1
            or first.get("violated") is not False
            or first.get("resource_window_sha256")
            != canonical_digest(first, "resource_window_sha256")
        ):
            raise CanonicalScreeningError(
                "runtime resource guard first sample contract mismatch"
            )
        return first

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
        if self._stop_summary is not None:
            return dict(self._stop_summary)
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
                "authorized_gpu_registry": self.authorized_gpu_registry,
                "final_active_worker_pids": sorted(self._active_worker_pids),
            }
        summary["samples"] = (
            {
                "path": str(self.sample_path.resolve()),
                "sha256": sha256_file(self.sample_path),
            }
            if self.sample_path.is_file()
            else None
        )
        self._stop_summary = dict(summary)
        return dict(summary)


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
    publish_exclusive_json(path, value)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "canonical_sha256": value["admission_sha256"],
    }


def _artifact_binding(path: Path, canonical_sha256: str) -> dict[str, str]:
    if not path.is_file():
        raise CanonicalScreeningError(f"bound artifact does not exist: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "canonical_sha256": canonical_sha256,
    }


def _validate_json_artifact_binding(
    binding: Mapping[str, Any], label: str, digest_field: str
) -> dict[str, Any]:
    if set(binding) != {"path", "sha256", "canonical_sha256"}:
        raise CanonicalScreeningError(f"{label} binding fields differ")
    path = Path(str(binding["path"])).resolve()
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise CanonicalScreeningError(f"{label} file binding mismatch")
    value = load_json(path, label)
    if (
        value.get(digest_field) != binding["canonical_sha256"]
        or canonical_digest(value, digest_field) != binding["canonical_sha256"]
    ):
        raise CanonicalScreeningError(f"{label} canonical binding mismatch")
    return value


def _gpu_phase_control(paths: Mapping[str, Path], phase: str) -> Path:
    if phase not in {"smoke8", "screen512"}:
        raise CanonicalScreeningError("GPU control phase is invalid")
    return paths["gpu_control"] / phase


def _validate_gpu_wrapper_provenance(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
) -> tuple[dict[str, str], dict[str, str]]:
    control = _gpu_phase_control(paths, phase)
    wrapper_claim_path = control / "wrapper_claim.json"
    observer_launch_path = control / "observer_launch.json"
    wrapper_claim = load_json(wrapper_claim_path, "GPU wrapper claim")
    if (
        wrapper_claim.get("contract_type")
        != "safa_canonical_gpu_wrapper_claim_v1"
        or wrapper_claim.get("policy_sha256") != policy["policy_sha256"]
        or wrapper_claim.get("phase") != phase
        or wrapper_claim.get("wrapper_claim_sha256")
        != canonical_digest(wrapper_claim, "wrapper_claim_sha256")
    ):
        raise CanonicalScreeningError("GPU wrapper claim contract mismatch")
    wrapper_binding = _artifact_binding(
        wrapper_claim_path, wrapper_claim["wrapper_claim_sha256"]
    )
    observer_launch = load_json(
        observer_launch_path, "GPU observer launch"
    )
    if (
        observer_launch.get("contract_type")
        != "safa_canonical_gpu_observer_launch_v2"
        or observer_launch.get("policy_sha256") != policy["policy_sha256"]
        or observer_launch.get("phase") != phase
        or observer_launch.get("status") != "launched"
        or observer_launch.get("failure") is not None
        or observer_launch.get("wrapper_claim") != wrapper_binding
        or observer_launch.get("wrapper_claim_sha256")
        != wrapper_claim["wrapper_claim_sha256"]
        or observer_launch.get("observer_launch_sha256")
        != canonical_digest(observer_launch, "observer_launch_sha256")
    ):
        raise CanonicalScreeningError("GPU observer launch contract mismatch")
    command = observer_launch.get("command")
    if (
        not isinstance(command, list)
        or "--monitor-target" not in command
        or command[command.index("--monitor-target") + 1] != phase
        or command[-1] != "--execute"
    ):
        raise CanonicalScreeningError(
            "GPU observer launch command differs from the phase"
        )
    return wrapper_binding, _artifact_binding(
        observer_launch_path, observer_launch["observer_launch_sha256"]
    )


def _write_gpu_controller_claim(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    wrapper_claim: Mapping[str, str],
    observer_launch: Mapping[str, str],
) -> tuple[dict[str, Any], Path]:
    claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_claim_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "wrapper_claim": dict(wrapper_claim),
        "observer_launch": dict(observer_launch),
        "controller_pid": os.getpid(),
        "started_at": _utc_now(),
    }
    claim["controller_claim_sha256"] = canonical_digest(
        claim, "controller_claim_sha256"
    )
    path = _gpu_phase_control(paths, phase) / "controller_claim.json"
    write_exclusive_json(path, claim)
    return claim, path


def _write_gpu_controller_terminal(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    claim: Mapping[str, Any],
    *,
    status: str,
    stage: str,
    failure: Mapping[str, Any] | None,
    controller_ready: Mapping[str, str] | None,
    observer_ready: Mapping[str, str] | None,
    final_release_admission: Mapping[str, str] | None,
    runtime_resource_guard: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
) -> Path:
    if status not in {"completed", "failed"}:
        raise CanonicalScreeningError("GPU controller terminal status is invalid")
    terminal = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_terminal_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "controller_claim_sha256": claim["controller_claim_sha256"],
        "status": status,
        "stage": stage,
        "failure": None if failure is None else dict(failure),
        "controller_ready": controller_ready,
        "observer_ready": observer_ready,
        "final_release_admission": final_release_admission,
        "runtime_resource_guard": (
            None if runtime_resource_guard is None else dict(runtime_resource_guard)
        ),
        "summary": None if summary is None else dict(summary),
        "completed_at": _utc_now(),
    }
    terminal["controller_terminal_sha256"] = canonical_digest(
        terminal, "controller_terminal_sha256"
    )
    path = _gpu_phase_control(paths, phase) / "controller_terminal.json"
    publish_exclusive_json(path, terminal)
    return path


def _write_gpu_bootstrap_terminal(
    policy: Mapping[str, Any],
    config: Path,
    paths: Mapping[str, Path],
    phase: str,
    *,
    status: str,
    stage: str,
    failure: Mapping[str, Any] | None,
    controller_claim: Mapping[str, Any] | None,
) -> Path:
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_bootstrap_terminal_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "config": {
            "path": str(config.resolve()),
            "sha256": sha256_file(config.resolve()),
        },
        "status": status,
        "stage": stage,
        "failure": None if failure is None else dict(failure),
        "controller_claim": (
            None if controller_claim is None else dict(controller_claim)
        ),
        "completed_at": _utc_now(),
    }
    value["controller_bootstrap_terminal_sha256"] = canonical_digest(
        value, "controller_bootstrap_terminal_sha256"
    )
    path = _gpu_phase_control(paths, phase) / "bootstrap_terminal.json"
    publish_exclusive_json(path, value)
    return path


def _write_main_bootstrap_failure(
    campaign_root: Path,
    config: Path,
    phase: str,
    failure: BaseException,
) -> None:
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_main_bootstrap_terminal_v1",
        "phase": phase,
        "config": {
            "path": str(config.resolve()),
            "sha256": (
                sha256_file(config.resolve()) if config.is_file() else None
            ),
        },
        "status": "failed",
        "failure": {
            "type": type(failure).__name__,
            "message": str(failure),
        },
        "completed_at": _utc_now(),
    }
    value["main_bootstrap_terminal_sha256"] = canonical_digest(
        value, "main_bootstrap_terminal_sha256"
    )
    publish_exclusive_json(
        campaign_root.resolve()
        / "bootstrap_control"
        / phase
        / "main_bootstrap_terminal.json",
        value,
    )


def _write_request_intent_manifest(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    replicates: Sequence[str],
    candidate_manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    rows = []
    for replicate in replicates:
        for candidate in candidate_manifest["candidates"]:
            rows.append(
                {
                    "sequence": len(rows) + 1,
                    "candidate_id": candidate["candidate_id"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "checkpoint_model": candidate["checkpoint_model"],
                    "mode": phase,
                    "replicate": replicate,
                    "sample_count": 8 if phase == "smoke8" else 512,
                    "seed": 4549,
                    "batch_size": 2,
                    "admission_sha256": admission["canonical_sha256"],
                }
            )
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_run_request_intent_manifest_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "replicates": list(replicates),
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "candidate_manifest_sha256": candidate_manifest[
            "candidate_manifest_sha256"
        ],
        "request_count": len(rows),
        "requests": rows,
    }
    value["request_intent_manifest_sha256"] = canonical_digest(
        value, "request_intent_manifest_sha256"
    )
    path = (
        paths["request_intents"]
        / phase
        / "request_intent_manifest.json"
    )
    write_exclusive_json(path, value)
    return value, path


def _validate_final_requests_against_intents(
    request_paths: Sequence[Path],
    intent_manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    controller_ready: Mapping[str, Any],
    observer_ready: Mapping[str, Any],
) -> None:
    intents = {
        (row["candidate_id"], row["replicate"]): row
        for row in intent_manifest["requests"]
    }
    if len(intents) != intent_manifest["request_count"]:
        raise CanonicalScreeningError(
            "immutable request intents contain duplicate logical requests"
        )
    observed: set[tuple[str, str]] = set()
    for request_path in request_paths:
        request = validate_run_request(
            load_json(request_path, "barrier-bound run request"), policy
        )
        key = (request["candidate"]["candidate_id"], request["replicate"])
        intent = intents.get(key)
        if intent is None or key in observed:
            raise CanonicalScreeningError(
                "final run request set differs from immutable intents"
            )
        observed.add(key)
        if (
            request["candidate"]["checkpoint_sha256"]
            != intent["checkpoint_sha256"]
            or request["candidate"]["checkpoint_model"]
            != intent["checkpoint_model"]
            or request["mode"] != intent["mode"]
            or request["sample_count"] != intent["sample_count"]
            or request["seed"] != intent["seed"]
            or request["batch_size"] != intent["batch_size"]
            or request["admission"]["canonical_sha256"]
            != intent["admission_sha256"]
            or request["controller_ready"] != controller_ready
            or request["observer_ready"] != observer_ready
        ):
            raise CanonicalScreeningError(
                "final run request fields differ from immutable intent"
            )
    if observed != set(intents):
        raise CanonicalScreeningError(
            "final run request coverage differs from immutable intents"
        )


def _write_final_release_admission(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    admission: Mapping[str, Any],
    controller_ready_binding: Mapping[str, Any],
    observer_ready_binding: Mapping[str, Any],
    request_paths: Sequence[Path],
    resource_guard: "RuntimeResourceGuard",
) -> tuple[dict[str, Any], dict[str, str]]:
    controller_ready_path = Path(str(controller_ready_binding["path"])).resolve()
    controller_ready = _validate_controller_ready(
        load_json(controller_ready_path, "release controller ready"),
        policy,
        phase,
        admission,
    )
    observer_ready_path = Path(str(observer_ready_binding["path"])).resolve()
    observer_ready = _validate_observer_ready(
        load_json(observer_ready_path, "release observer ready"),
        policy,
        phase,
        controller_ready,
        admission,
    )
    if (
        controller_ready["wrapper_claim"] != observer_ready["wrapper_claim"]
        or controller_ready["observer_launch"]
        != observer_ready["observer_launch"]
    ):
        raise CanonicalScreeningError(
            "release wrapper provenance differs across ready barriers"
        )
    intent_binding = controller_ready["request_intent_manifest"]
    intent = _validate_json_artifact_binding(
        intent_binding,
        "release request intent manifest",
        "request_intent_manifest_sha256",
    )
    _validate_final_requests_against_intents(
        request_paths,
        intent,
        policy,
        controller_ready_binding,
        observer_ready_binding,
    )
    if len(request_paths) != controller_ready["request_count"]:
        raise CanonicalScreeningError(
            "release request count differs from controller ready"
        )
    resource_guard.raise_if_violated()
    snapshot = assert_resource_admission(
        policy, paths["root"], require_idle_gpus=True
    )
    admission_value = load_json(
        Path(str(admission["path"])), "release initial admission"
    )
    if (
        snapshot["authorized_gpu_registry"]
        != admission_value["snapshot"]["authorized_gpu_registry"]
        or snapshot["compute_processes"] != []
    ):
        raise CanonicalScreeningError(
            "final release admission differs from initial GPU registry"
        )
    requests = []
    for path in sorted(request_paths, key=lambda item: str(item.resolve())):
        request = validate_run_request(
            load_json(path, "release run request"), policy
        )
        requests.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "canonical_sha256": request["run_request_sha256"],
            }
        )
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_final_release_admission_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "initial_admission_sha256": admission["canonical_sha256"],
        "controller_ready_sha256": controller_ready[
            "controller_ready_sha256"
        ],
        "observer_ready_sha256": observer_ready["observer_ready_sha256"],
        "wrapper_claim": dict(controller_ready["wrapper_claim"]),
        "wrapper_claim_sha256": controller_ready["wrapper_claim_sha256"],
        "observer_launch": dict(controller_ready["observer_launch"]),
        "observer_launch_sha256": controller_ready[
            "observer_launch_sha256"
        ],
        "authorized_gpu_registry": snapshot["authorized_gpu_registry"],
        "request_count": len(requests),
        "requests": requests,
        "snapshot": snapshot,
        "released_at": _utc_now(),
    }
    value["final_release_admission_sha256"] = canonical_digest(
        value, "final_release_admission_sha256"
    )
    path = _gpu_phase_control(paths, phase) / "final_release_admission.json"
    publish_exclusive_json(path, value)
    return value, _artifact_binding(
        path, value["final_release_admission_sha256"]
    )


def _validate_monitor_sample(
    sample: Mapping[str, Any],
    policy: Mapping[str, Any],
    phase: str,
    admission: Mapping[str, Any],
    *,
    terminal: bool,
) -> dict[str, Any]:
    value = dict(sample)
    if (
        value.get("contract_type")
        != "safa_canonical_resource_monitor_sample_v1"
        or value.get("policy_sha256") != policy["policy_sha256"]
        or value.get("phase") != phase
        or value.get("terminal") is not terminal
        or value.get("monitor_sample_sha256")
        != canonical_digest(value, "monitor_sample_sha256")
        or value.get("gpu_binding", {}).get("admission_sha256")
        != admission["canonical_sha256"]
    ):
        raise CanonicalScreeningError("resource monitor sample contract mismatch")
    return value


def _write_gpu_resource_recheck(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    admission: Mapping[str, Any],
    first_guard_sample: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    snapshot = assert_resource_admission(
        policy, paths["root"], require_idle_gpus=True
    )
    admission_value = load_json(Path(str(admission["path"])), "resource admission")
    original = admission_value["snapshot"]
    if (
        snapshot["authorized_gpu_registry"]
        != original["authorized_gpu_registry"]
        or snapshot["compute_processes"] != []
        or any(gpu["temperature_c"] > 85 for gpu in snapshot["gpus"])
        or first_guard_sample.get("violated") is not False
        or first_guard_sample.get("swap_consecutive_io") != 0
    ):
        raise CanonicalScreeningError("GPU resource recheck differs from admission")
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_resource_recheck_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "snapshot": snapshot,
        "first_guard_sample_sha256": first_guard_sample[
            "resource_window_sha256"
        ],
        "completed_at": _utc_now(),
    }
    value["resource_recheck_sha256"] = canonical_digest(
        value, "resource_recheck_sha256"
    )
    path = _gpu_phase_control(paths, phase) / "resource_recheck.json"
    write_exclusive_json(path, value)
    return value, path


def _write_controller_ready(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    claim: Mapping[str, Any],
    admission: Mapping[str, Any],
    intent_manifest: Mapping[str, Any],
    intent_path: Path,
    internal_monitor_sample: Mapping[str, Any],
    internal_monitor_path: Path,
    first_guard_sample: Mapping[str, Any],
    guard_path: Path,
    recheck: Mapping[str, Any],
    recheck_path: Path,
    claim_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, str]]:
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_claim_sha256": claim["controller_claim_sha256"],
        "controller_claim": _artifact_binding(
            claim_path, claim["controller_claim_sha256"]
        ),
        "wrapper_claim": dict(claim["wrapper_claim"]),
        "wrapper_claim_sha256": claim["wrapper_claim"]["canonical_sha256"],
        "observer_launch": dict(claim["observer_launch"]),
        "observer_launch_sha256": claim["observer_launch"][
            "canonical_sha256"
        ],
        "admission": dict(admission),
        "request_count": intent_manifest["request_count"],
        "request_intent_manifest": _artifact_binding(
            intent_path, intent_manifest["request_intent_manifest_sha256"]
        ),
        "internal_monitor": _artifact_binding(
            internal_monitor_path,
            internal_monitor_sample["monitor_sample_sha256"],
        ),
        "runtime_guard_first_sample": _artifact_binding(
            guard_path, first_guard_sample["resource_window_sha256"]
        ),
        "resource_recheck": _artifact_binding(
            recheck_path, recheck["resource_recheck_sha256"]
        ),
        "ready_at": _utc_now(),
    }
    value["controller_ready_sha256"] = canonical_digest(
        value, "controller_ready_sha256"
    )
    path = _gpu_phase_control(paths, phase) / "controller_ready.json"
    publish_exclusive_json(path, value)
    binding = _artifact_binding(path, value["controller_ready_sha256"])
    return value, path, binding


def _validate_controller_ready(
    ready: Mapping[str, Any],
    policy: Mapping[str, Any],
    phase: str,
    admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = dict(ready)
    if (
        value.get("contract_type") != "safa_canonical_gpu_controller_ready_v1"
        or value.get("campaign_id") != policy["campaign_id"]
        or value.get("phase") != phase
        or value.get("policy_sha256") != policy["policy_sha256"]
        or value.get("controller_ready_sha256")
        != canonical_digest(value, "controller_ready_sha256")
        or value.get("request_count") != (386 if phase == "smoke8" else 193)
        or (
            admission is not None
            and value.get("admission_sha256") != admission["canonical_sha256"]
        )
    ):
        raise CanonicalScreeningError("controller ready contract mismatch")
    for field, digest_field in {
        "controller_claim": "controller_claim_sha256",
        "wrapper_claim": "wrapper_claim_sha256",
        "observer_launch": "observer_launch_sha256",
        "admission": "admission_sha256",
        "request_intent_manifest": "request_intent_manifest_sha256",
        "internal_monitor": "monitor_sample_sha256",
        "runtime_guard_first_sample": "resource_window_sha256",
        "resource_recheck": "resource_recheck_sha256",
    }.items():
        if not isinstance(value.get(field), Mapping):
            raise CanonicalScreeningError(
                f"controller ready omits {field} binding"
            )
        _validate_json_artifact_binding(
            value[field], f"controller ready {field}", digest_field
        )
    if (
        value["controller_claim"]["canonical_sha256"]
        != value["controller_claim_sha256"]
        or value["wrapper_claim"]["canonical_sha256"]
        != value["wrapper_claim_sha256"]
        or value["observer_launch"]["canonical_sha256"]
        != value["observer_launch_sha256"]
        or value["admission"]["canonical_sha256"]
        != value["admission_sha256"]
    ):
        raise CanonicalScreeningError("controller ready primary binding mismatch")
    return value


def _validate_observer_ready(
    ready: Mapping[str, Any],
    policy: Mapping[str, Any],
    phase: str,
    controller_ready: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(ready)
    if (
        value.get("contract_type") != "safa_canonical_gpu_observer_ready_v1"
        or value.get("campaign_id") != policy["campaign_id"]
        or value.get("phase") != phase
        or value.get("policy_sha256") != policy["policy_sha256"]
        or value.get("admission_sha256") != admission["canonical_sha256"]
        or value.get("controller_ready_sha256")
        != controller_ready["controller_ready_sha256"]
        or value.get("observer_ready_sha256")
        != canonical_digest(value, "observer_ready_sha256")
    ):
        raise CanonicalScreeningError("observer ready contract mismatch")
    for field, digest_field in {
        "observer_claim": "observer_claim_sha256",
        "wrapper_claim": "wrapper_claim_sha256",
        "observer_launch": "observer_launch_sha256",
        "controller_ready": "controller_ready_sha256",
        "admission": "admission_sha256",
        "first_observer_sample": "monitor_sample_sha256",
    }.items():
        if not isinstance(value.get(field), Mapping):
            raise CanonicalScreeningError(
                f"observer ready omits {field} binding"
            )
        _validate_json_artifact_binding(
            value[field], f"observer ready {field}", digest_field
        )
    if (
        value["observer_claim"]["canonical_sha256"]
        != value["observer_claim_sha256"]
        or value["wrapper_claim"]["canonical_sha256"]
        != value["wrapper_claim_sha256"]
        or value["observer_launch"]["canonical_sha256"]
        != value["observer_launch_sha256"]
        or value["controller_ready"]["canonical_sha256"]
        != value["controller_ready_sha256"]
        or value["admission"]["canonical_sha256"]
        != value["admission_sha256"]
    ):
        raise CanonicalScreeningError("observer ready primary binding mismatch")
    return value


def _wait_observer_ready(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    controller_ready: Mapping[str, Any],
    admission: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = _gpu_phase_control(paths, phase) / "observer_ready.json"
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise CanonicalScreeningError("observer ready barrier timed out")
        time.sleep(0.1)
    value = _validate_observer_ready(
        load_json(path, "observer ready"),
        policy,
        phase,
        controller_ready,
        admission,
    )
    return value, _artifact_binding(path, value["observer_ready_sha256"])


def _assert_observer_live(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    phase: str,
    observer_ready: Mapping[str, Any],
) -> None:
    ready_path = Path(str(observer_ready["path"])).resolve()
    if (
        not ready_path.is_file()
        or sha256_file(ready_path) != observer_ready["sha256"]
    ):
        raise CanonicalScreeningError("observer ready file changed after release")
    ready = load_json(ready_path, "observer ready liveness binding")
    observer_path = paths["logs"] / f"{phase}__observer.jsonl"
    rows = load_jsonl(observer_path, "observer heartbeat samples")
    if not rows:
        raise CanonicalScreeningError("external observer heartbeat is absent")
    latest = _validate_monitor_sample(
        rows[-1],
        policy,
        phase,
        ready["admission"],
        terminal=False,
    )
    try:
        completed_at = datetime.fromisoformat(str(latest["observed_at"]))
    except (KeyError, ValueError) as exc:
        raise CanonicalScreeningError(
            "external observer heartbeat timestamp is invalid"
        ) from exc
    heartbeat_age = (datetime.now(timezone.utc) - completed_at).total_seconds()
    heartbeat_limit = max(
        30.0, 3.0 * float(policy["resources"]["resource_poll_seconds"])
    )
    if heartbeat_age < 0 or heartbeat_age > heartbeat_limit:
        raise CanonicalScreeningError(
            "external observer heartbeat is stale: "
            f"age={heartbeat_age:.2f}s, limit={heartbeat_limit:.2f}s"
        )
    terminal_path = _gpu_phase_control(paths, phase) / "observer_terminal.json"
    if terminal_path.exists():
        terminal = load_json(terminal_path, "observer terminal")
        raise CanonicalScreeningError(
            "external observer terminated before controller: "
            f"status={terminal.get('status')}, failure={terminal.get('failure')}"
        )
    session = f"safa-screening-{phase}-monitor"
    alive = (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    if not alive:
        raise CanonicalScreeningError(
            "external observer tmux died after ready release"
        )


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
    controller_ready: Mapping[str, Any],
    observer_ready: Mapping[str, Any],
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
        controller_ready,
        observer_ready,
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
        monitor = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "safa-screening-preflight-monitor",
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
                    "preflight",
                    "--execute",
                ]
            ),
        ]
    else:
        controller = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            f"safa-screening-{phase}-controller",
            "-c",
            str(REPO_ROOT),
            python,
            str(REPO_ROOT / "scripts/run_canonical_gpu_wrapper.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--config",
            str(config.resolve()),
            "--campaign-root",
            str(campaign_root.resolve()),
            "--policy-sha256",
            str(policy["policy_sha256"]),
            "--phase",
            phase,
            "--python",
            python,
        ]
        monitor = []
    return {"controller": controller, "monitor": monitor}


def _run_monitor(
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    target: str,
) -> dict[str, Any]:
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("resource monitor must run inside tmux")
    controller_session = f"safa-screening-{target}-controller"
    control = _gpu_phase_control(paths, target)
    controller_process_exit_path = control / "controller_process_exit.json"
    claim_path = control / "observer_claim.json"
    if claim_path.exists():
        raise CanonicalScreeningError(
            "observer claim already exists for this phase"
        )
    provenance_deadline = time.monotonic() + 180.0
    while not (
        (control / "wrapper_claim.json").is_file()
        and (control / "observer_launch.json").is_file()
    ):
        if controller_process_exit_path.is_file():
            raise CanonicalScreeningError(
                "controller exited before observer provenance release"
            )
        if time.monotonic() >= provenance_deadline:
            raise CanonicalScreeningError(
                "observer wrapper provenance barrier timed out"
            )
        time.sleep(0.1)
    wrapper_claim, observer_launch = _validate_gpu_wrapper_provenance(
        policy, paths, target
    )
    claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_claim_v1",
        "campaign_id": policy["campaign_id"],
        "phase": target,
        "policy_sha256": policy["policy_sha256"],
        "wrapper_claim": wrapper_claim,
        "observer_launch": observer_launch,
        "observer_pid": os.getpid(),
        "started_at": _utc_now(),
    }
    claim["observer_claim_sha256"] = canonical_digest(
        claim, "observer_claim_sha256"
    )
    write_exclusive_json(claim_path, claim)
    path = paths["logs"] / f"{target}__observer.jsonl"
    terminal_path = control / "observer_terminal.json"
    samples = 0
    status = "failed"
    failure: dict[str, str] | None = None
    ready_binding: dict[str, str] | None = None
    try:
        exists = subprocess.run(
            ["tmux", "has-session", "-t", controller_session],
            capture_output=True,
            text=True,
        ).returncode == 0
        if not exists:
            raise CanonicalScreeningError(
                "observer refuses to start without the controller session"
            )
        artifact_deadline = time.monotonic() + 180.0
        admission_paths = sorted(paths["admissions"].glob(f"{target}__*.json"))
        while not admission_paths:
            if controller_process_exit_path.is_file():
                raise CanonicalScreeningError(
                    "controller exited before observer admission release"
                )
            if time.monotonic() >= artifact_deadline:
                raise CanonicalScreeningError(
                    "observer admission barrier timed out"
                )
            time.sleep(0.1)
            admission_paths = sorted(
                paths["admissions"].glob(f"{target}__*.json")
            )
        if len(admission_paths) != 1:
            raise CanonicalScreeningError(
                "observer requires exactly one current-phase admission"
            )
        admission_path = admission_paths[0]
        admission_value = load_json(admission_path, "observer admission")
        admission = _artifact_binding(
            admission_path, admission_value["admission_sha256"]
        )
        controller_ready_path = control / "controller_ready.json"
        while not controller_ready_path.is_file():
            if controller_process_exit_path.is_file():
                raise CanonicalScreeningError(
                    "controller exited before observer ready release"
                )
            if time.monotonic() >= artifact_deadline:
                raise CanonicalScreeningError(
                    "observer controller ready barrier timed out"
                )
            time.sleep(0.1)
        controller_ready = _validate_controller_ready(
            load_json(controller_ready_path, "controller ready"),
            policy,
            target,
            admission,
        )
        first = _validate_monitor_sample(
            _monitor_sample(
                policy,
                paths,
                target,
                terminal=False,
                admission=admission,
            ),
            policy,
            target,
            admission,
            terminal=False,
        )
        _append_jsonl(path, first)
        samples += 1
        first_sample_path = control / "observer_first_sample.json"
        write_exclusive_json(first_sample_path, first)
        ready = {
            "schema_version": 1,
            "contract_type": "safa_canonical_gpu_observer_ready_v1",
            "campaign_id": policy["campaign_id"],
            "phase": target,
            "policy_sha256": policy["policy_sha256"],
            "admission_sha256": admission["canonical_sha256"],
            "controller_ready_sha256": controller_ready[
                "controller_ready_sha256"
            ],
            "observer_claim_sha256": claim["observer_claim_sha256"],
            "wrapper_claim_sha256": wrapper_claim["canonical_sha256"],
            "observer_launch_sha256": observer_launch["canonical_sha256"],
            "observer_claim": _artifact_binding(
                claim_path, claim["observer_claim_sha256"]
            ),
            "wrapper_claim": wrapper_claim,
            "observer_launch": observer_launch,
            "controller_ready": _artifact_binding(
                controller_ready_path,
                controller_ready["controller_ready_sha256"],
            ),
            "admission": dict(admission),
            "first_observer_sample": _artifact_binding(
                first_sample_path, first["monitor_sample_sha256"]
            ),
            "ready_at": _utc_now(),
        }
        ready["observer_ready_sha256"] = canonical_digest(
            ready, "observer_ready_sha256"
        )
        ready_path = control / "observer_ready.json"
        publish_exclusive_json(ready_path, ready)
        ready_binding = _artifact_binding(
            ready_path, ready["observer_ready_sha256"]
        )
        controller_terminal_path = control / "controller_terminal.json"
        while True:
            time.sleep(float(policy["resources"]["resource_poll_seconds"]))
            controller_alive = subprocess.run(
                ["tmux", "has-session", "-t", controller_session],
                capture_output=True,
                text=True,
            ).returncode == 0
            terminal_exists = controller_terminal_path.is_file()
            process_exited = controller_process_exit_path.is_file()
            if not controller_alive and not terminal_exists:
                raise CanonicalScreeningError(
                    "controller disappeared without a durable terminal"
                )
            if process_exited and not terminal_exists:
                raise CanonicalScreeningError(
                    "controller process exited without a durable terminal"
                )
            if terminal_exists:
                controller_terminal = load_json(
                    controller_terminal_path, "GPU controller terminal"
                )
                if (
                    controller_terminal.get("contract_type")
                    != "safa_canonical_gpu_controller_terminal_v1"
                    or controller_terminal.get("policy_sha256")
                    != policy["policy_sha256"]
                    or controller_terminal.get("phase") != target
                    or controller_terminal.get("controller_terminal_sha256")
                    != canonical_digest(
                        controller_terminal, "controller_terminal_sha256"
                    )
                ):
                    raise CanonicalScreeningError(
                        "observer controller terminal contract mismatch"
                    )
            sample = _validate_monitor_sample(
                _monitor_sample(
                    policy,
                    paths,
                    target,
                    terminal=terminal_exists,
                    admission=admission,
                ),
                policy,
                target,
                admission,
                terminal=terminal_exists,
            )
            _append_jsonl(path, sample)
            samples += 1
            if terminal_exists:
                break
        status = "completed"
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        terminal = {
            "schema_version": 1,
            "contract_type": "safa_canonical_gpu_observer_terminal_v1",
            "campaign_id": policy["campaign_id"],
            "phase": target,
            "policy_sha256": policy["policy_sha256"],
            "observer_claim_sha256": claim["observer_claim_sha256"],
            "status": status,
            "failure": failure,
            "observer_ready": ready_binding,
            "samples": samples,
            "completed_at": _utc_now(),
        }
        terminal["observer_terminal_sha256"] = canonical_digest(
            terminal, "observer_terminal_sha256"
        )
        publish_exclusive_json(terminal_path, terminal)
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
    replicates = ("primary", "repeat") if phase == "smoke8" else ("primary",)
    request_roots = [
        paths["run_requests"] / f"{phase}_{replicate}" for replicate in replicates
    ]
    run_roots = [
        paths["runs"] / f"{phase}_{replicate}" for replicate in replicates
    ]

    def count(roots: Sequence[Path], pattern: str) -> int:
        return sum(
            1
            for root in roots
            if root.exists()
            for _ in root.rglob(pattern)
        )

    return {
        "request_files": count(request_roots, "*.json"),
        "result_files": count(run_roots, "result.json"),
        "generated_png": count(run_roots, "*.png"),
        "preflight_requests": (
            sum(1 for _ in paths["preflight_requests"].glob("*.json"))
            if paths["preflight_requests"].exists()
            else 0
        ),
        "preflight_results": (
            sum(1 for _ in paths["preflight_results"].glob("*.json"))
            if paths["preflight_results"].exists()
            else 0
        ),
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
    sample = {
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
    sample["monitor_sample_sha256"] = canonical_digest(
        sample, "monitor_sample_sha256"
    )
    return sample


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
    active: list[dict[str, Any]],
    slot_pool: FreeSlotPool,
    resource_guard: RuntimeResourceGuard,
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
        resource_guard.unregister_worker_pid(process.pid)
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


def _validate_launch_integrity(
    policy: Mapping[str, Any],
    config: Path,
    paths: Mapping[str, Path],
    request: Path,
    final_release_admission: Mapping[str, Any],
    *,
    worker_pid: int | None = None,
    gpu_index: int | None = None,
    gpu_uuid: str | None = None,
    worker_ready: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    implementations, config_sha256 = _stdlib_validate_implementation_bindings(
        config
    )
    asset_verification_audit: list[dict[str, Any]] = []
    current_policy = validate_policy(
        REPO_ROOT,
        config.resolve(),
        verify_historical_output_evidence=False,
        asset_verification_audit=asset_verification_audit,
    )
    if current_policy != dict(policy):
        raise CanonicalScreeningError(
            "controller policy/config changed before worker launch"
        )
    if (
        _stdlib_sha256_file(config.resolve()) != config_sha256
        or {
            name: {
                "path": str(Path(binding["path"]).resolve()),
                "sha256": binding["sha256"],
            }
            for name, binding in current_policy["implementations"].items()
        }
        != implementations
    ):
        raise CanonicalScreeningError(
            "controller bootstrap bindings changed before worker validator import"
        )
    from safa.closeout.canonical_screening_worker import (
        validate_pre_cuda_request,
    )

    worker_module = sys.modules[
        "safa.closeout.canonical_screening_worker"
    ]
    if (
        Path(worker_module.__file__).resolve()
        != Path(implementations["screening_worker"]["path"])
    ):
        raise CanonicalScreeningError(
            "imported screening worker path differs from policy binding"
        )
    pre_cuda = validate_pre_cuda_request(
        request,
        policy,
        final_release_admission,
        config_path=config,
        require_heavy_modules_absent=False,
    )
    resources = assert_resource_admission(
        policy, paths["root"], require_idle_gpus=False
    )
    value = {
        "schema_version": 1,
        "contract_type": CONTROLLER_LAUNCH_REHASH_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "run_request_sha256": pre_cuda["request"]["run_request_sha256"],
        "worker_pid": worker_pid,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "worker_ready": (
            None if worker_ready is None else dict(worker_ready)
        ),
        "verification_order": pre_cuda["verification_order"],
        "rehashed_bindings": pre_cuda["rehashed_bindings"],
        "rehashed_bindings_sha256": pre_cuda[
            "rehashed_bindings_sha256"
        ],
        "resource_snapshot": resources,
        "asset_content_verification": asset_verification_audit[0],
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "validated_at": _utc_now(),
    }
    value["controller_launch_rehash_sha256"] = canonical_digest(
        value, "controller_launch_rehash_sha256"
    )
    return value


def _worker_handshake_paths(
    paths: Mapping[str, Path],
    phase: str,
    request: Path,
    gpu_index: int,
    slot_index: int,
) -> tuple[Path, Path, Path]:
    request_value = load_json(request, "worker handshake request")
    root = (
        _gpu_phase_control(paths, phase)
        / "worker_handshakes"
        / (
            f"{request_value['run_request_sha256'][:16]}"
            f"__gpu{gpu_index}_slot{slot_index}"
        )
    )
    return (
        root / "worker_ready.json",
        root / "worker_release.json",
        root / "controller_launch_rehash.json",
    )


def _wait_worker_ready(
    process: Any,
    ready_path: Path,
    request_path: Path,
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    gpu_index: int,
    gpu_uuid: str,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    deadline = time.monotonic() + timeout_seconds
    while not ready_path.is_file():
        return_code = process.poll()
        if return_code is not None:
            bootstrap_path = (
                ready_path.resolve().parent
                / "worker_bootstrap_terminal.json"
            )
            if bootstrap_path.is_file():
                bootstrap = load_json(
                    bootstrap_path, "worker bootstrap terminal"
                )
                required = {
                    "schema_version",
                    "contract_type",
                    "policy_sha256",
                    "phase",
                    "worker_pid",
                    "stage",
                    "config",
                    "request",
                    "worker_ready_path",
                    "worker_release_path",
                    "status",
                    "failure",
                    "completed_at",
                    "worker_bootstrap_terminal_sha256",
                }
                request_binding = dict(
                    bootstrap.get("request")
                    if isinstance(bootstrap.get("request"), Mapping)
                    else {}
                )
                failure = bootstrap.get("failure")
                if (
                    set(bootstrap) != required
                    or bootstrap["schema_version"] != 1
                    or bootstrap["contract_type"]
                    != "safa_canonical_worker_bootstrap_terminal_v1"
                    or bootstrap["policy_sha256"]
                    not in {None, policy["policy_sha256"]}
                    or bootstrap["phase"] != request["mode"]
                    or bootstrap["worker_pid"] != process.pid
                    or bootstrap["stage"]
                    not in {
                        "policy_bootstrap",
                        "worker_arguments",
                        "worker_import",
                    }
                    or bootstrap["worker_ready_path"]
                    != str(ready_path.resolve())
                    or bootstrap["status"] != "failed"
                    or not isinstance(failure, Mapping)
                    or set(failure) != {"type", "message"}
                    or not failure["type"]
                    or not failure["message"]
                    or set(request_binding)
                    != {"path", "sha256", "canonical_sha256"}
                    or Path(str(request_binding["path"])).resolve()
                    != request_path.resolve()
                    or request_binding["canonical_sha256"]
                    != request["run_request_sha256"]
                    or not request_path.is_file()
                    or request_binding["sha256"]
                    != sha256_file(request_path)
                    or bootstrap["worker_bootstrap_terminal_sha256"]
                    != canonical_digest(
                        bootstrap,
                        "worker_bootstrap_terminal_sha256",
                    )
                ):
                    raise CanonicalScreeningError(
                        "CPU worker bootstrap terminal contract mismatch"
                    )
                raise CanonicalScreeningError(
                    "CPU worker bootstrap failed before worker_ready: "
                    f"{failure['type']}: {failure['message']}"
                )
            if return_code < 0:
                raise CanonicalScreeningError(
                    "CPU worker was terminated by signal before worker_ready: "
                    f"signal={-return_code}; bootstrap terminal unavailable"
                )
            raise CanonicalScreeningError(
                "CPU worker exited before worker_ready: "
                f"exit_code={return_code}; bootstrap terminal unavailable"
            )
        if time.monotonic() >= deadline:
            raise CanonicalScreeningError("worker_ready barrier timed out")
        time.sleep(0.05)
    ready = validate_worker_ready_value(
        load_json(ready_path, "worker_ready"),
        request,
        policy,
        expected_worker_pid=process.pid,
        expected_gpu_index=gpu_index,
        expected_gpu_uuid=gpu_uuid,
    )
    binding = _artifact_binding(
        ready_path, ready["worker_ready_sha256"]
    )
    return ready, binding


def _publish_worker_release(
    release_path: Path,
    policy: Mapping[str, Any],
    request: Mapping[str, Any],
    worker_pid: int,
    worker_ready: Mapping[str, str],
    controller_rehash: Mapping[str, str],
    resource_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": WORKER_RELEASE_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "phase": request["mode"],
        "worker_pid": worker_pid,
        "run_request_sha256": request["run_request_sha256"],
        "worker_ready": dict(worker_ready),
        "controller_launch_rehash": dict(controller_rehash),
        "resource_snapshot": dict(resource_snapshot),
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "released_at": _utc_now(),
    }
    value["worker_release_sha256"] = canonical_digest(
        value, "worker_release_sha256"
    )
    validate_worker_release_value(
        value,
        request,
        policy,
        expected_worker_pid=worker_pid,
    )
    publish_exclusive_json(release_path, value)
    return value


def _worker_command(
    policy: Mapping[str, Any],
    config: Path,
    campaign_root: Path,
    phase: str,
    request: Path,
    gpu_index: int,
    gpu_uuid: str,
    final_release_admission: Mapping[str, Any],
    worker_ready_path: Path,
    worker_release_path: Path,
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
        "--final-release-path",
        str(final_release_admission["path"]),
        "--final-release-sha256",
        str(final_release_admission["sha256"]),
        "--final-release-canonical-sha256",
        str(final_release_admission["canonical_sha256"]),
        "--worker-ready-path",
        str(worker_ready_path.resolve()),
        "--worker-release-path",
        str(worker_release_path.resolve()),
    ]


def _worker_environment(gpu_uuid: str) -> dict[str, str]:
    if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise CanonicalScreeningError("worker physical GPU UUID is invalid")
    return {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
    }


def _prepare_gpu_ready_barrier(
    policy: Mapping[str, Any],
    config: Path,
    paths: Mapping[str, Path],
    phase: str,
    claim: Mapping[str, Any],
    claim_path: Path,
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    lifecycle["stage"] = "startup_admission"
    admission_snapshot = assert_resource_admission(
        policy, paths["root"], require_idle_gpus=True
    )
    admission = _write_admission(policy, paths, phase, admission_snapshot)
    lifecycle["admission"] = admission
    lifecycle["stage"] = "manifest_and_smoke_validation"
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
    replicates = ("primary", "repeat") if phase == "smoke8" else ("primary",)
    lifecycle["stage"] = "request_intents"
    intent, intent_path = _write_request_intent_manifest(
        policy,
        paths,
        phase,
        replicates,
        candidate_manifest,
        admission,
    )
    lifecycle["stage"] = "internal_monitor_first_sample"
    monitor_path = _append_monitor_sample(
        policy, paths, phase, admission=admission
    )
    lifecycle["monitor_path"] = monitor_path
    internal_sample = _validate_monitor_sample(
        load_jsonl(monitor_path, "internal monitor samples")[0],
        policy,
        phase,
        admission,
        terminal=False,
    )
    internal_sample_path = (
        _gpu_phase_control(paths, phase) / "internal_monitor_first_sample.json"
    )
    write_exclusive_json(internal_sample_path, internal_sample)
    lifecycle["stage"] = "runtime_guard_first_sample"
    guard_path = paths["logs"] / f"{phase}__runtime_resource_windows.jsonl"
    resource_guard = RuntimeResourceGuard(
        policy,
        guard_path,
        paths["root"].parent,
        admission_snapshot["authorized_gpu_registry"],
    )
    lifecycle["resource_guard"] = resource_guard
    resource_guard.start()
    first_guard = resource_guard.wait_first_sample(
        max(30.0, 3.0 * resource_guard.poll_seconds)
    )
    first_guard_path = (
        _gpu_phase_control(paths, phase) / "runtime_guard_first_sample.json"
    )
    write_exclusive_json(first_guard_path, first_guard)
    lifecycle["stage"] = "resource_recheck"
    recheck, recheck_path = _write_gpu_resource_recheck(
        policy, paths, phase, admission, first_guard
    )
    lifecycle["stage"] = "controller_ready"
    controller_ready, _, controller_ready_binding = _write_controller_ready(
        policy,
        paths,
        phase,
        claim,
        admission,
        intent,
        intent_path,
        internal_sample,
        internal_sample_path,
        first_guard,
        first_guard_path,
        recheck,
        recheck_path,
        claim_path,
    )
    lifecycle["controller_ready"] = controller_ready_binding
    lifecycle["stage"] = "observer_ready"
    _, observer_ready_binding = _wait_observer_ready(
        policy,
        paths,
        phase,
        controller_ready,
        admission,
        timeout_seconds=180.0,
    )
    lifecycle["observer_ready"] = observer_ready_binding
    lifecycle["stage"] = "final_run_requests"
    requests: list[Path] = []
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
                controller_ready_binding,
                observer_ready_binding,
            )
        )
    if len(requests) != intent["request_count"]:
        raise CanonicalScreeningError(
            "final run request count differs from immutable intents"
        )
    _validate_final_requests_against_intents(
        requests,
        intent,
        policy,
        controller_ready_binding,
        observer_ready_binding,
    )
    return {
        "claim": claim,
        "claim_path": claim_path,
        "admission_snapshot": admission_snapshot,
        "admission": admission,
        "candidate_manifest": candidate_manifest,
        "requests": requests,
        "resource_guard": resource_guard,
        "monitor_path": monitor_path,
        "controller_ready": controller_ready_binding,
        "observer_ready": observer_ready_binding,
    }


def _build_gpu_completion_summary(
    policy: Mapping[str, Any],
    phase: str,
    paths: Mapping[str, Path],
    requests: Sequence[Path],
    admission: Mapping[str, Any],
    final_release_admission: Mapping[str, Any],
    monitor_path: Path,
    resource_guard_summary: Mapping[str, Any],
) -> dict[str, Any]:
    completed_runs = []
    capability_counts = {
        "latent": {"request_count": 0, "completed_count": 0},
        "pixel": {"request_count": 0, "completed_count": 0},
    }
    for request_path in requests:
        request = validate_run_request(
            load_json(request_path, "completed run request"),
            policy,
        )
        if request["mode"] != phase:
            raise CanonicalScreeningError(
                "completion request phase differs"
            )
        output_dir = Path(str(request["output_dir"])).resolve()
        claim_path = output_dir / "claim.json"
        result_path = output_dir / "result.json"
        claim = validate_run_claim(
            load_json(claim_path, "completed run claim"),
            request,
            policy,
        )
        result = validate_run_result(
            load_json(result_path, "completed run result"),
            request,
            claim,
            policy,
        )
        if result["status"] != "completed":
            raise CanonicalScreeningError(
                "completion result status is not completed"
            )
        worker_terminal_path = (
            Path(str(claim["worker_ready"]["path"])).parent
            / "worker_terminal.json"
        )
        worker_terminal = validate_worker_terminal_value(
            load_json(worker_terminal_path, "completed worker terminal"),
            request_path,
            policy,
            expected_worker_pid=claim["worker_pid"],
            require_completed=True,
        )
        if (
            worker_terminal["request"]
            != _artifact_binding(
                request_path, request["run_request_sha256"]
            )
            or worker_terminal["claim"]
            != _artifact_binding(claim_path, claim["run_claim_sha256"])
            or worker_terminal["result"]
            != _artifact_binding(result_path, result["run_result_sha256"])
        ):
            raise CanonicalScreeningError(
                "completion worker terminal artifact chain differs"
            )
        output_space = request["output_contract"]["capability"][
            "output_space"
        ]
        if output_space not in capability_counts:
            raise CanonicalScreeningError(
                f"completion output space is unregistered: {output_space}"
            )
        capability_counts[output_space]["request_count"] += 1
        capability_counts[output_space]["completed_count"] += 1
        completed_runs.append(
            {
                "run_request_sha256": request["run_request_sha256"],
                "run_claim_sha256": claim["run_claim_sha256"],
                "run_result_sha256": result["run_result_sha256"],
                "worker_terminal_sha256": worker_terminal[
                    "worker_terminal_sha256"
                ],
                "request": _artifact_binding(
                    request_path, request["run_request_sha256"]
                ),
                "claim": _artifact_binding(
                    claim_path, claim["run_claim_sha256"]
                ),
                "result": _artifact_binding(
                    result_path, result["run_result_sha256"]
                ),
                "worker_terminal": _artifact_binding(
                    worker_terminal_path,
                    worker_terminal["worker_terminal_sha256"],
                ),
                "output_space": output_space,
            }
        )
    summary = {
        "phase": phase,
        "completed_at": _utc_now(),
        "request_count": len(requests),
        "failures": [],
        "admission": dict(admission),
        "final_release_admission": dict(final_release_admission),
        "monitor_log": {
            "path": str(monitor_path.resolve()),
            "sha256": sha256_file(monitor_path),
        },
        "runtime_resource_guard": dict(resource_guard_summary),
        "completed_runs": completed_runs,
        "capability_completion": capability_counts,
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
    return summary


def _run_gpu_phase(
    policy: Mapping[str, Any],
    config: Path,
    paths: Mapping[str, Path],
    phase: str,
) -> dict[str, Any]:
    claim: dict[str, Any] | None = None
    claim_path: Path | None = None
    lifecycle: dict[str, Any] = {
        "stage": "bootstrap",
        "admission": None,
        "controller_ready": None,
        "observer_ready": None,
        "final_release_admission": None,
        "resource_guard": None,
        "monitor_path": None,
    }
    failures: list[str] = []
    active: list[dict[str, Any]] = []
    slot_pool: FreeSlotPool | None = None
    resource_guard_summary: dict[str, Any] | None = None
    monitor_terminal_written = False
    terminal_written = False
    caught: BaseException | None = None
    summary: dict[str, Any] | None = None
    try:
        lifecycle["stage"] = "tmux_bootstrap"
        if "TMUX" not in os.environ:
            raise CanonicalScreeningError(
                "GPU screening controller must run inside tmux"
            )
        lifecycle["stage"] = "wrapper_provenance"
        wrapper_claim, observer_launch = _validate_gpu_wrapper_provenance(
            policy, paths, phase
        )
        lifecycle["stage"] = "controller_claim"
        claim, claim_path = _write_gpu_controller_claim(
            policy, paths, phase, wrapper_claim, observer_launch
        )
        _write_gpu_bootstrap_terminal(
            policy,
            config,
            paths,
            phase,
            status="completed",
            stage="controller_claim",
            failure=None,
            controller_claim=_artifact_binding(
                claim_path, claim["controller_claim_sha256"]
            ),
        )
        barrier = _prepare_gpu_ready_barrier(
            policy,
            config,
            paths,
            phase,
            claim,
            claim_path,
            lifecycle,
        )
        admission_snapshot = barrier["admission_snapshot"]
        admission = barrier["admission"]
        requests = list(barrier["requests"])
        resource_guard = barrier["resource_guard"]
        _assert_observer_live(
            policy, paths, phase, barrier["observer_ready"]
        )
        gpu_uuid_by_index = {
            row["physical_gpu_index"]: row["physical_gpu_uuid"]
            for row in admission_snapshot["authorized_gpu_registry"]
        }
        gpus = list(policy["resources"]["physical_gpus"])
        capacity = int(policy["resources"]["workers_per_gpu"])
        slots = [(gpu, slot) for gpu in gpus for slot in range(capacity)]
        slot_pool = FreeSlotPool(slots)
        lock_root = Path(str(policy["resources"]["global_lock_root"]))
        paths["logs"].mkdir(parents=True, exist_ok=True)
        request_queue = deque(requests)
        lifecycle["stage"] = "final_release_admission"
        _assert_observer_live(
            policy, paths, phase, barrier["observer_ready"]
        )
        _, final_release_binding = _write_final_release_admission(
            policy,
            paths,
            phase,
            admission,
            barrier["controller_ready"],
            barrier["observer_ready"],
            requests,
            resource_guard,
        )
        lifecycle["final_release_admission"] = final_release_binding
        lifecycle["stage"] = "worker_execution"
        while request_queue or active:
            resource_guard.raise_if_violated()
            _assert_observer_live(
                policy, paths, phase, barrier["observer_ready"]
            )
            while request_queue and slot_pool.free_count:
                request = request_queue.popleft()
                gpu, slot_index = slot_pool.acquire()
                gpu_uuid = gpu_uuid_by_index[gpu]
                lock: Path | None = None
                log_handle: Any | None = None
                process: Any | None = None
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
                    (
                        ready_path,
                        release_path,
                        controller_rehash_path,
                    ) = _worker_handshake_paths(
                        paths, phase, request, gpu, slot_index
                    )
                    request_value = validate_run_request(
                        load_json(request, "launch run request"), policy
                    )
                    process, initial_rehash = resource_guard.launch_cpu_worker(
                        lambda: subprocess.Popen(
                            _worker_command(
                                policy,
                                config,
                                paths["root"],
                                phase,
                                request,
                                gpu,
                                gpu_uuid,
                                final_release_binding,
                                ready_path,
                                release_path,
                            ),
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=_worker_environment(gpu_uuid),
                        ),
                        lambda: _validate_launch_integrity(
                            policy,
                            config,
                            paths,
                            request,
                            final_release_binding,
                        ),
                    )
                    _, ready_binding = _wait_worker_ready(
                        process,
                        ready_path,
                        request,
                        request_value,
                        policy,
                        gpu,
                        gpu_uuid,
                        timeout_seconds=180.0,
                    )
                    final_rehash: dict[str, Any] | None = None

                    def publish_release(
                        validation: Mapping[str, Any],
                        guard_snapshot: Mapping[str, Any],
                    ) -> None:
                        nonlocal final_rehash
                        final_rehash = dict(validation)
                        validate_controller_launch_rehash_value(
                            final_rehash,
                            request_value,
                            policy,
                        )
                        publish_exclusive_json(
                            controller_rehash_path, final_rehash
                        )
                        controller_rehash_binding = _artifact_binding(
                            controller_rehash_path,
                            final_rehash[
                                "controller_launch_rehash_sha256"
                            ],
                        )
                        release_resources = {
                            "admission": validation["resource_snapshot"],
                            "runtime_guard": dict(guard_snapshot),
                        }
                        _publish_worker_release(
                            release_path,
                            policy,
                            request_value,
                            process.pid,
                            ready_binding,
                            controller_rehash_binding,
                            release_resources,
                        )

                    resource_guard.release_worker_after_handshake(
                        process.pid,
                        lambda: _validate_launch_integrity(
                            policy,
                            config,
                            paths,
                            request,
                            final_release_binding,
                            worker_pid=process.pid,
                            gpu_index=gpu,
                            gpu_uuid=gpu_uuid,
                            worker_ready=ready_binding,
                        ),
                        publish_release,
                    )
                except BaseException:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        process.wait()
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
                        "worker_ready": ready_binding,
                        "worker_release": _artifact_binding(
                            release_path,
                            load_json(
                                release_path, "worker release"
                            )["worker_release_sha256"],
                        ),
                        "initial_launch_rehash": initial_rehash,
                        "final_launch_rehash": final_rehash,
                        "controller_launch_rehash": _artifact_binding(
                            controller_rehash_path,
                            final_rehash[
                                "controller_launch_rehash_sha256"
                            ],
                        ),
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
                resource_guard.unregister_worker_pid(process.pid)
                active.remove(item)
                worker_terminal_path = (
                    Path(str(item["worker_ready"]["path"])).parent
                    / "worker_terminal.json"
                )
                worker_terminal = (
                    load_json(worker_terminal_path, "worker terminal")
                    if worker_terminal_path.is_file()
                    else None
                )
                if return_code == 0 and (
                    worker_terminal is None
                    or worker_terminal.get("contract_type")
                    != "safa_canonical_worker_terminal_v1"
                    or worker_terminal.get("worker_pid") != process.pid
                    or worker_terminal.get("status") != "completed"
                    or worker_terminal.get("worker_terminal_sha256")
                    != canonical_digest(
                        worker_terminal, "worker_terminal_sha256"
                    )
                ):
                    failures.append(
                        f"{item['request']}: completed without a valid "
                        "worker terminal"
                    )
                if return_code != 0:
                    terminal_semantics = (
                        "worker terminal absent as expected after signal"
                        if return_code < 0 and worker_terminal is None
                        else "worker terminal present"
                    )
                    failures.append(
                        f"{item['request']}: exit_code={return_code}; "
                        f"{terminal_semantics}"
                    )
            if failures:
                raise CanonicalScreeningError(" | ".join(failures))
            violation = _gpu_hard_resource_violation(policy)
            if violation is not None:
                raise CanonicalScreeningError(violation)
            _append_monitor_sample(
                policy, paths, phase, admission=admission
            )
            if active:
                time.sleep(10)
        lifecycle["stage"] = "observer_final_gate"
        _assert_observer_live(
            policy, paths, phase, barrier["observer_ready"]
        )
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
        if late_resource_failure is not None:
            raise CanonicalScreeningError(late_resource_failure)
        if resource_guard_summary["final_active_worker_pids"] != []:
            raise CanonicalScreeningError(
                "runtime guard retained worker PIDs at completion"
            )
        _append_monitor_sample(
            policy, paths, phase, terminal=True, admission=admission
        )
        monitor_terminal_written = True
        lifecycle["stage"] = "completion_summary"
        summary = _build_gpu_completion_summary(
            policy,
            phase,
            paths,
            requests,
            admission,
            final_release_binding,
            barrier["monitor_path"],
            resource_guard_summary,
        )
        completed_summary_path = (
            paths["summaries"] / f"{phase}__completed.json"
        )
        write_exclusive_json(completed_summary_path, summary)
        completed_summary_binding = _artifact_binding(
            completed_summary_path,
            summary["controller_summary_sha256"],
        )
        lifecycle["stage"] = "completed_terminal"
        _write_gpu_controller_terminal(
            policy,
            paths,
            phase,
            claim,
            status="completed",
            stage="completed",
            failure=None,
            controller_ready=barrier["controller_ready"],
            observer_ready=barrier["observer_ready"],
            final_release_admission=final_release_binding,
            runtime_resource_guard=resource_guard_summary,
            summary=completed_summary_binding,
        )
        terminal_written = True
    except BaseException as exc:
        caught = exc
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        resource_guard = lifecycle["resource_guard"]
        admission = lifecycle["admission"]
        monitor_path = lifecycle["monitor_path"]
        if active and slot_pool is not None and resource_guard is not None:
            try:
                _cleanup_active_workers(active, slot_pool, resource_guard)
            except BaseException as cleanup_exc:
                failures.append(
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                if caught is None:
                    caught = cleanup_exc
        if resource_guard is not None:
            try:
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
                if (
                    late_resource_failure is not None
                    and late_resource_failure not in failures
                ):
                    failures.append(late_resource_failure)
                    if caught is None:
                        caught = CanonicalScreeningError(
                            late_resource_failure
                        )
            except BaseException as guard_stop_exc:
                failures.append(
                    f"{type(guard_stop_exc).__name__}: {guard_stop_exc}"
                )
                if caught is None:
                    caught = guard_stop_exc
        if (
            not monitor_terminal_written
            and monitor_path is not None
            and admission is not None
        ):
            try:
                _append_monitor_sample(
                    policy, paths, phase, terminal=True, admission=admission
                )
                monitor_terminal_written = True
            except BaseException as monitor_exc:
                failures.append(
                    f"{type(monitor_exc).__name__}: {monitor_exc}"
                )
                if caught is None:
                    caught = monitor_exc
        if caught is not None and not terminal_written:
            failure = {
                "type": type(caught).__name__,
                "message": " | ".join(failures),
            }
            failed_summary_binding: dict[str, str] | None = None
            try:
                stop = {
                    "phase": phase,
                    "stopped_at": _utc_now(),
                    "reason": lifecycle["stage"],
                    "failures": failures,
                    "admission": admission,
                    "final_release_admission": lifecycle[
                        "final_release_admission"
                    ],
                    "monitor_log": (
                        None
                        if monitor_path is None
                        else {
                            "path": str(monitor_path.resolve()),
                            "sha256": sha256_file(monitor_path),
                        }
                    ),
                    "runtime_resource_guard": resource_guard_summary,
                }
                stop["controller_summary_sha256"] = hashlib.sha256(
                    canonical_json(stop)
                ).hexdigest()
                failed_summary_path = (
                    paths["summaries"] / f"{phase}__failed.json"
                )
                write_exclusive_json(failed_summary_path, stop)
                failed_summary_binding = _artifact_binding(
                    failed_summary_path,
                    stop["controller_summary_sha256"],
                )
            except BaseException as summary_exc:
                failures.append(
                    f"{type(summary_exc).__name__}: {summary_exc}"
                )
                failure["message"] = " | ".join(failures)
            if claim is None:
                _write_gpu_bootstrap_terminal(
                    policy,
                    config,
                    paths,
                    phase,
                    status="failed",
                    stage=lifecycle["stage"],
                    failure=failure,
                    controller_claim=None,
                )
            else:
                _write_gpu_controller_terminal(
                    policy,
                    paths,
                    phase,
                    claim,
                    status="failed",
                    stage=lifecycle["stage"],
                    failure=failure,
                    controller_ready=lifecycle["controller_ready"],
                    observer_ready=lifecycle["observer_ready"],
                    final_release_admission=lifecycle[
                        "final_release_admission"
                    ],
                    runtime_resource_guard=resource_guard_summary,
                    summary=failed_summary_binding,
                )
            terminal_written = True
    if caught is not None:
        raise caught.with_traceback(caught.__traceback__)
    if summary is None:
        raise CanonicalScreeningError(
            "GPU controller completed without a summary"
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _root(args.config)
    campaign_root = _root(args.campaign_root)
    try:
        policy = _install_verified_contract_api(config)
    except BaseException as exc:
        if args.execute:
            if (
                args.request is not None
                and args.worker_ready_path is not None
            ):
                _write_stdlib_worker_bootstrap_failure(
                    config=config,
                    phase=args.phase,
                    request_path=_root(args.request),
                    worker_ready_path=_root(args.worker_ready_path),
                    worker_release_path=(
                        None
                        if args.worker_release_path is None
                        else _root(args.worker_release_path)
                    ),
                    stage="policy_bootstrap",
                    failure=exc,
                )
            elif args.request is None:
                _write_stdlib_main_bootstrap_failure(
                    campaign_root, config, args.phase, exc
                )
        raise
    paths = _paths(campaign_root, policy["policy_sha256"])
    if args.phase == "monitor":
        if args.dry_run or args.monitor_target is None:
            raise CanonicalScreeningError(
                "monitor requires --execute and --monitor-target"
            )
        print(json.dumps(_run_monitor(policy, paths, args.monitor_target), sort_keys=True))
        return 0
    if args.request is not None:
        try:
            if (
                args.gpu_index is None
                or args.gpu_uuid is None
                or args.final_release_path is None
                or args.final_release_sha256 is None
                or args.final_release_canonical_sha256 is None
                or args.worker_ready_path is None
                or args.worker_release_path is None
                or args.phase not in {"smoke8", "screen512"}
            ):
                raise CanonicalScreeningError(
                    "--request requires --gpu-index, --gpu-uuid, and a GPU "
                    "screening phase"
                )
            if args.dry_run:
                raise CanonicalScreeningError(
                    "worker requests cannot use --dry-run"
                )
        except BaseException as exc:
            if args.worker_ready_path is not None:
                _write_stdlib_worker_bootstrap_failure(
                    config=config,
                    phase=args.phase,
                    request_path=_root(args.request),
                    worker_ready_path=_root(args.worker_ready_path),
                    worker_release_path=(
                        None
                        if args.worker_release_path is None
                        else _root(args.worker_release_path)
                    ),
                    stage="worker_arguments",
                    failure=exc,
                )
            raise
        try:
            from safa.closeout.canonical_screening_worker import (
                execute_screening_request,
            )
        except BaseException as exc:
            _write_stdlib_worker_bootstrap_failure(
                config=config,
                phase=args.phase,
                request_path=_root(args.request),
                worker_ready_path=_root(args.worker_ready_path),
                worker_release_path=_root(args.worker_release_path),
                stage="worker_import",
                failure=exc,
            )
            raise

        execute_screening_request(
            _root(args.request),
            args.gpu_index,
            args.gpu_uuid,
            policy,
            {
                "path": str(_root(args.final_release_path)),
                "sha256": args.final_release_sha256,
                "canonical_sha256": args.final_release_canonical_sha256,
            },
            _root(args.worker_ready_path),
            _root(args.worker_release_path),
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
