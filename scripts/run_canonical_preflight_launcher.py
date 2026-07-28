#!/usr/bin/env python3
"""Evidence-complete launcher for the canonical CPU preflight wrapper."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
import traceback
import types
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from safa.closeout.preflight_launch_contract import (
        FAULT_RECORD_CONTRACT_TYPE,
        LAUNCH_RECEIPT_CONTRACT_TYPE,
        LAUNCH_TERMINAL_CONTRACT_TYPE,
        PreflightLaunchContractError,
        build_artifact_binding,
        build_bound_lifecycle_evidence,
        build_file_identity,
        build_finalization_secondary_failure,
        build_gate_ready,
        build_invalid_claim_evidence,
        build_lifecycle_wait_status,
        build_launch_accepted,
        build_launch_terminal_v2,
        build_ownership_release,
        build_ownership_terminal,
        build_pane_fault_consumer_chain,
        build_pane_fault_consumer_registration,
        build_pane_owner_seal,
        build_preclaim_failure_intent,
        build_process_identity,
        build_publish_failure_record,
        build_sealed_lifecycle_artifact,
        build_terminal_failure,
        build_tmux_server_identity,
        build_tmux_started,
        build_verified_implementations,
        build_wrapper_started,
        validate_claim_v3,
        validate_artifact_binding,
        validate_file_identity,
        validate_gate_ready,
        validate_launch_receipt_schema,
        validate_launch_terminal_v2,
        validate_lifecycle_wait_status,
        validate_ownership_chain,
        validate_pane_fault_consumer_chain,
        validate_pane_fault_consumer_registration,
        validate_preclaim_failure_intent,
        validate_publish_failure_record,
        validate_verified_implementations,
        validate_wrapper_started,
    )

CONTROLLER_SESSION = "safa-screening-preflight-controller"
OBSERVER_SESSION_PREFIX = "safa-screening-preflight-monitor-"
TMUX_OWNER_ENV = "SAFA_OWNER_NONCE"
OBSERVER_SESSION_ENV = "SAFA_PREFLIGHT_OBSERVER_SESSION"
LAUNCH_RECEIPT_PATH_ENV = "SAFA_PREFLIGHT_LAUNCH_RECEIPT_PATH"
LAUNCH_ACCEPTED_PATH_ENV = "SAFA_PREFLIGHT_LAUNCH_ACCEPTED_PATH"
LAUNCH_RELEASE_PATH_ENV = "SAFA_PREFLIGHT_LAUNCH_RELEASE_PATH"
PANE_LOG_PATH_ENV = "SAFA_PREFLIGHT_PANE_LOG_PATH"
FAULT_CHANNEL_FD_ENV = "SAFA_PREFLIGHT_FAULT_CHANNEL_FD"
FAULT_CHANNEL_MAX_RECORD_BYTES = 65536
FAULT_CHANNEL_PREFIX = b"SAFA-PREFLIGHT-FAULT-V1\n"
FAULT_CHANNEL_SHA_PREFIX = b"sha256:"
LIFECYCLE_WAIT_CHANNEL_PREFIX = (
    b"SAFA-PREFLIGHT-LIFECYCLE-WAIT-V1\n"
)
LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES = 65536
LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX = b"commit:"
LIFECYCLE_WAIT_EXPECTED_BINDING_KEYS = {
    "policy_sha256",
    "attempt_id",
    "source_artifact",
    "publisher",
    "supervisor_owner_seal",
    "supervisor_process",
    "supervisor_executable",
    "supervisor_command",
    "worker_started",
    "child_process",
    "child_executable",
    "child_command",
    "terminal",
}
PANE_GATE_MODE = "__pane_gate__"
GATE_WAIT_SUPERVISOR_MODE = "__gate_wait_supervisor__"
GATE_ADJUDICATED_EXIT = 117
CONSUMER_WAIT_SUPERVISOR_MODE = "__consumer_wait_supervisor__"
CONSUMER_ADJUDICATED_EXIT = 118
PANE_FAULT_CONSUMER_MODE = "__pane_fault_consumer__"
PANE_FAULT_CONSUMER_JOIN_MODE = "__join_pane_fault_consumer__"
PANE_FAULT_CONSUMER_OWNER_ENV = (
    "SAFA_PANE_FAULT_CONSUMER_OWNER_NONCE"
)
PANE_FAULT_CONSUMER_SESSION_PREFIX = (
    "safa-pane-fault-consumer-"
)
ARCHIVE_FAILURE_MODE = "archive-untracked-failure"
ARCHIVE_LEGACY_FAILURE_V2_MODE = (
    "archive-legacy-untracked-failure-v2"
)
LEGACY_FAILURE_ARCHIVE_CONTRACT_TYPE = (
    "safa_canonical_legacy_untracked_launch_failure_archive_v2"
)
LEGACY_FAILURE_EVIDENCE_CONTRACT_TYPE = (
    "safa_canonical_legacy_untracked_launch_failure_evidence_v2"
)
LEGACY_FAILURE_ARCHIVE_ID_DERIVATION = (
    "sha256_canonical_legacy_untracked_evidence_v1"
)
LEGACY_POLICY_TREE_DERIVATION = (
    "sha256_relative_posix_nul_size_nul_content_sha256_lf_v1"
)
LEGACY_REQUEST_SET_DERIVATION = (
    "sha256_canonical_preflight_request_bindings_v1"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
REPO_ROOT = Path(__file__).resolve().parents[1]
_VERIFIED_LOADER_RELATIVE_PATH = (
    "src/safa/closeout/verified_preflight_module_loader.py"
)
_SHARED_CONTRACT_RELATIVE_PATH = (
    "src/safa/closeout/preflight_launch_contract.py"
)
_VERIFIED_LOADER_EXPORTS = (
    "VerifiedPreflightModuleError",
    "load_verified_preflight_module",
    "reverify_verified_preflight_module",
)
_SHARED_CONTRACT_EXPORTS = (
    "FAULT_RECORD_CONTRACT_TYPE",
    "LAUNCH_RECEIPT_CONTRACT_TYPE",
    "LAUNCH_RECEIPT_V5_CONTRACT_TYPE",
    "LAUNCH_TERMINAL_CONTRACT_TYPE",
    "LAUNCH_TERMINAL_V2_CONTRACT_TYPE",
    "LIFECYCLE_WAIT_STATUS_CONTRACT_TYPE",
    "LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE",
    "LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_CONTRACT_TYPE",
    "POSTCLAIM_FINALIZATION_PROFILE_V1_CONTRACT_TYPE",
    "OS_ERROR_TYPE_TOKENS",
    "PRECLAIM_FAILURE_INTENT_CONTRACT_TYPE",
    "POST_HANDOFF_FINALIZATION_FAILURE_CONTRACT_TYPE",
    "PreflightLaunchContractError",
    "build_artifact_binding",
    "build_bound_lifecycle_evidence",
    "build_file_identity",
    "build_finalization_inner_failure",
    "build_finalization_secondary_failure",
    "build_gate_ready",
    "build_invalid_claim_evidence",
    "build_lifecycle_raw_wait_publish_failure_v1",
    "build_lifecycle_raw_wait_v3",
    "build_lifecycle_wait_status",
    "build_launch_accepted",
    "build_launch_receipt_v5",
    "build_launch_terminal_v2",
    "build_ownership_release",
    "build_ownership_terminal",
    "build_pane_fault_consumer_chain",
    "build_pane_fault_consumer_registration",
    "build_preclaim_failure_intent",
    "build_postclaim_finalization_profile_v1",
    "build_pane_owner_seal",
    "build_post_handoff_finalization_failure",
    "build_process_identity",
    "build_publish_failure_record",
    "build_sealed_lifecycle_artifact",
    "build_terminal_failure",
    "build_tmux_server_identity",
    "build_tmux_started",
    "build_verified_implementations",
    "build_wrapper_started",
    "derive_lifecycle_wait_outcome",
    "validate_claim_v3",
    "validate_artifact_binding",
    "validate_file_identity",
    "validate_finalization_secondary_failure",
    "validate_gate_ready",
    "validate_invalid_claim_evidence",
    "validate_launch_receipt_schema",
    "validate_launch_receipt_v5",
    "validate_launch_terminal_v2",
    "validate_lifecycle_raw_wait_publish_failure_v1",
    "validate_lifecycle_raw_wait_v3",
    "validate_lifecycle_wait_status",
    "validate_ownership_chain",
    "validate_pane_fault_consumer_chain",
    "validate_pane_fault_consumer_registration",
    "validate_publish_failure_record",
    "validate_preclaim_failure_intent",
    "validate_postclaim_finalization_profile_v1",
    "validate_post_handoff_finalization_failure",
    "validate_tmux_started",
    "validate_verified_implementations",
    "validate_wrapper_started",
)
_VERIFIED_LOADER_HANDLE: dict[str, Any] | None = None
_SHARED_CONTRACT_HANDLE: dict[str, Any] | None = None


def build_file_identity(
    *,
    path: str,
    device: int,
    inode: int,
    mode: int,
    size: int,
) -> dict[str, Any]:
    """Build bootstrap-safe identity before the verified API is installed."""
    return {
        "path": path,
        "device": device,
        "inode": inode,
        "mode": mode,
        "size": size,
    }


def build_process_identity(
    *,
    pid: int,
    ppid: int,
    pgid: int,
    sid: int,
    start_ticks: int,
) -> dict[str, int]:
    """Build bootstrap-safe process identity before verified API install."""
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": pgid,
        "sid": sid,
        "start_ticks": start_ticks,
    }


def _bootstrap_read_file(
    path: Path,
    expected_sha256: str | None,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_CLOEXEC"):
        raise RuntimeError(
            "preflight bootstrap requires no-follow descriptors"
        )
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or path.is_symlink()
    ):
        raise RuntimeError(f"{label} path is not exact")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    source = b"".join(chunks)
    identity = {
        "path": str(path),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mode": int(before.st_mode),
        "size": int(before.st_size),
    }
    if (
        identity
        != {
            "path": str(path),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "mode": int(after.st_mode),
            "size": int(after.st_size),
        }
        or not stat.S_ISREG(before.st_mode)
        or before.st_size != len(source)
        or (
            expected_sha256 is not None
            and hashlib.sha256(source).hexdigest()
            != expected_sha256
        )
    ):
        raise RuntimeError(f"{label} identity or SHA-256 differs")
    return source, identity


def _bootstrap_implementation(
    policy: Mapping[str, Any],
    name: str,
    relative_path: str,
) -> tuple[Path, str]:
    implementations = policy.get("implementations")
    raw = (
        implementations.get(name)
        if isinstance(implementations, Mapping)
        else None
    )
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"path", "sha256"}
        or raw.get("path") != relative_path
        or not isinstance(raw.get("sha256"), str)
        or not HEX64.fullmatch(str(raw["sha256"]))
    ):
        raise RuntimeError(
            f"preflight bootstrap implementation differs: {name}"
        )
    path = REPO_ROOT / relative_path
    if path.parent.resolve(strict=True) != path.parent:
        raise RuntimeError(
            f"preflight bootstrap parent path differs: {name}"
        )
    return path, str(raw["sha256"])


def _reverify_verified_loader() -> dict[str, Any]:
    if _VERIFIED_LOADER_HANDLE is None:
        raise RuntimeError("verified preflight loader is not installed")
    handle = _VERIFIED_LOADER_HANDLE
    config_source, config_identity = _bootstrap_read_file(
        Path(handle["config_path"]),
        handle["config_sha256"],
        "preflight policy",
    )
    _caller_source, caller_identity = _bootstrap_read_file(
        Path(handle["caller_path"]),
        handle["caller_sha256"],
        "preflight launcher",
    )
    _loader_source, loader_identity = _bootstrap_read_file(
        Path(handle["loader_path"]),
        handle["loader_sha256"],
        "verified preflight loader",
    )
    module = handle["module"]
    exports = handle["exports"]
    if (
        hashlib.sha256(config_source).hexdigest()
        != handle["config_sha256"]
        or config_identity != handle["config_identity"]
        or caller_identity != handle["caller_identity"]
        or loader_identity != handle["loader_identity"]
        or any(
            getattr(module, name, None) is not value
            for name, value in exports.items()
        )
    ):
        raise RuntimeError(
            "verified preflight loader changed after bootstrap"
        )
    return {
        "path": handle["loader_path"],
        "sha256": handle["loader_sha256"],
        "file_identity": dict(loader_identity),
    }


def _install_verified_preflight_apis(config: Path) -> dict[str, Any]:
    global _VERIFIED_LOADER_HANDLE
    global _SHARED_CONTRACT_HANDLE
    config = config.resolve(strict=True)
    config_source, config_identity = _bootstrap_read_file(
        config, None, "preflight policy"
    )
    try:
        policy = json.loads(config_source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("preflight policy is not valid JSON") from exc
    if not isinstance(policy, dict):
        raise RuntimeError("preflight policy is not a mapping")
    caller_path, caller_sha256 = _bootstrap_implementation(
        policy,
        "preflight_launcher",
        "scripts/run_canonical_preflight_launcher.py",
    )
    caller_source, caller_identity = _bootstrap_read_file(
        caller_path, caller_sha256, "preflight launcher"
    )
    if caller_path != Path(__file__).resolve():
        raise RuntimeError("policy does not bind this preflight launcher")
    loader_path, loader_sha256 = _bootstrap_implementation(
        policy,
        "preflight_verified_loader",
        _VERIFIED_LOADER_RELATIVE_PATH,
    )
    loader_source, loader_identity = _bootstrap_read_file(
        loader_path,
        loader_sha256,
        "verified preflight loader",
    )
    loader_module = types.ModuleType(
        f"_safa_verified_preflight_loader_{loader_sha256}"
    )
    loader_module.__file__ = str(loader_path)
    loader_module.__package__ = "safa.closeout"
    try:
        exec(
            compile(loader_source, str(loader_path), "exec"),
            loader_module.__dict__,
        )
    except BaseException as exc:
        raise RuntimeError(
            "verified preflight loader execution failed"
        ) from exc
    loader_exports = {}
    for name in _VERIFIED_LOADER_EXPORTS:
        if not hasattr(loader_module, name):
            raise RuntimeError("verified preflight loader API differs")
        loader_exports[name] = getattr(loader_module, name)
    _VERIFIED_LOADER_HANDLE = {
        "module": loader_module,
        "exports": loader_exports,
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config_source).hexdigest(),
        "config_identity": config_identity,
        "caller_path": str(caller_path),
        "caller_sha256": caller_sha256,
        "caller_identity": caller_identity,
        "loader_path": str(loader_path),
        "loader_sha256": loader_sha256,
        "loader_identity": loader_identity,
    }
    _reverify_verified_loader()
    shared_handle = loader_exports[
        "load_verified_preflight_module"
    ](
        config_path=str(config),
        repo_root=str(REPO_ROOT),
        caller_name="preflight_launcher",
        caller_relative_path=(
            "scripts/run_canonical_preflight_launcher.py"
        ),
        target_name="preflight_launch_contract",
        target_relative_path=_SHARED_CONTRACT_RELATIVE_PATH,
        expected_exports=_SHARED_CONTRACT_EXPORTS,
    )
    for name in _SHARED_CONTRACT_EXPORTS:
        globals()[name] = shared_handle["exports"][name]
    _SHARED_CONTRACT_HANDLE = shared_handle
    return _reverify_verified_preflight_apis()


def _reverify_verified_preflight_apis() -> dict[str, Any]:
    loader_binding = _reverify_verified_loader()
    if _SHARED_CONTRACT_HANDLE is None or _VERIFIED_LOADER_HANDLE is None:
        raise RuntimeError("verified preflight contract is not installed")
    shared_binding = _VERIFIED_LOADER_HANDLE["exports"][
        "reverify_verified_preflight_module"
    ](_SHARED_CONTRACT_HANDLE)
    value = build_verified_implementations(
        verified_loader=loader_binding,
        preflight_launch_contract=shared_binding,
    )
    return validate_verified_implementations(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(value: Mapping[str, Any], excluded: str) -> str:
    payload = {
        key: item for key, item in value.items() if key != excluded
    }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_secure_directory(path: Path) -> tuple[int, os.stat_result]:
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or path.is_symlink()
    ):
        raise RuntimeError(
            f"secure directory path is not exact: {path}"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC,
    )
    value = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o755
    ):
        os.close(descriptor)
        raise RuntimeError(
            f"secure directory identity or permissions differ: {path}"
        )
    return descriptor, value


def _ensure_secure_leaf_directories(
    trusted_root: Path,
    relative_parts: Sequence[str],
    *,
    final_must_be_new: bool = False,
) -> Path:
    current_path = trusted_root
    current_descriptor, _identity = _open_secure_directory(
        trusted_root
    )
    try:
        for index, name in enumerate(relative_parts):
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\0" in name
            ):
                raise RuntimeError(
                    "secure directory component is invalid"
                )
            created = False
            try:
                os.mkdir(
                    name,
                    0o755,
                    dir_fd=current_descriptor,
                )
                created = True
            except FileExistsError:
                if (
                    final_must_be_new
                    and index == len(relative_parts) - 1
                ):
                    raise
            child_descriptor = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                dir_fd=current_descriptor,
            )
            if created:
                os.fchmod(child_descriptor, 0o755)
            child = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(child.st_mode)
                or child.st_uid != os.geteuid()
                or stat.S_IMODE(child.st_mode) != 0o755
            ):
                os.close(child_descriptor)
                raise RuntimeError(
                    "secure directory component identity or "
                    "permissions differ"
                )
            if created:
                os.fsync(current_descriptor)
            parent_descriptor = current_descriptor
            current_descriptor = child_descriptor
            os.close(parent_descriptor)
            current_path = current_path / name
    finally:
        os.close(current_descriptor)
    return current_path


def _create_fault_channel(path: Path) -> dict[str, Any]:
    directory_descriptor, directory = _open_secure_directory(
        path.parent
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_DSYNC
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or opened.st_size != 0
        ):
            raise RuntimeError(
                "preflight fault channel identity differs"
            )
        os.fsync(descriptor)
        descriptor_to_close = descriptor
        descriptor = -1
        os.close(descriptor_to_close)
        os.fsync(directory_descriptor)
        return {
            "path": str(path),
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "mode": int(opened.st_mode),
            "uid": int(opened.st_uid),
            "nlink": int(opened.st_nlink),
            "size": int(opened.st_size),
            "sha256": hashlib.sha256(b"").hexdigest(),
            "directory_device": int(directory.st_dev),
            "directory_inode": int(directory.st_ino),
        }
    finally:
        if descriptor >= 0:
            descriptor_to_close = descriptor
            descriptor = -1
            os.close(descriptor_to_close)
        os.close(directory_descriptor)


def _open_presealed_fault_channel(
    attempt_root: Path,
    binding: Mapping[str, Any],
    *,
    name: str = "wrapper_fault.channel",
) -> int:
    path = attempt_root / name
    expected_keys = {
        "path",
        "device",
        "inode",
        "mode",
        "uid",
        "nlink",
        "size",
        "sha256",
        "directory_device",
        "directory_inode",
    }
    directory_descriptor, directory = _open_secure_directory(
        attempt_root
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        opened = os.fstat(descriptor)
        named = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        observed = {
            "path": str(path),
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "mode": int(opened.st_mode),
            "uid": int(opened.st_uid),
            "nlink": int(opened.st_nlink),
            "size": int(opened.st_size),
            "sha256": hashlib.sha256(b"").hexdigest(),
            "directory_device": int(directory.st_dev),
            "directory_inode": int(directory.st_ino),
        }
        if (
            set(binding) != expected_keys
            or dict(binding) != observed
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or opened.st_size != 0
        ):
            raise RuntimeError(
                "presealed fault channel identity differs"
            )
        os.set_inheritable(descriptor, False)
        return_descriptor = descriptor
        descriptor = -1
        return return_descriptor
    finally:
        if descriptor >= 0:
            descriptor_to_close = descriptor
            descriptor = -1
            os.close(descriptor_to_close)
        os.close(directory_descriptor)


def _close_fault_channel(descriptor: int) -> dict[str, str] | None:
    try:
        os.close(descriptor)
    except OSError as exc:
        return {
            "type": type(exc).__name__,
            "message": str(exc),
            "errno": str(exc.errno),
        }
    return None


def _require_named_lifecycle_wait_channel(
    descriptor: int,
    binding: Mapping[str, Any],
    *,
    directory_descriptor: int | None = None,
    expected_size: int | None = None,
) -> os.stat_result:
    path = Path(str(binding["path"]))
    expected_keys = {
        "path",
        "device",
        "inode",
        "mode",
        "uid",
        "nlink",
        "size",
        "sha256",
        "directory_device",
        "directory_inode",
    }
    if (
        set(binding) != expected_keys
        or not path.is_absolute()
        or path.name in {"", ".", ".."}
    ):
        raise RuntimeError(
            "lifecycle wait channel binding differs"
        )
    named_directory_descriptor = -1
    try:
        named_directory_descriptor, named_directory = (
            _open_secure_directory(path.parent)
        )
        if directory_descriptor is None:
            directory_descriptor = named_directory_descriptor
        directory = os.fstat(directory_descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_dev != binding["directory_device"]
            or directory.st_ino != binding["directory_inode"]
            or named_directory.st_dev != directory.st_dev
            or named_directory.st_ino != directory.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_dev != binding["device"]
            or opened.st_ino != binding["inode"]
            or opened.st_mode != binding["mode"]
            or opened.st_uid != binding["uid"]
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != binding["nlink"]
            or opened.st_nlink != 1
            or (
                expected_size is not None
                and opened.st_size != expected_size
            )
            or named.st_dev != opened.st_dev
            or named.st_ino != opened.st_ino
            or named.st_mode != opened.st_mode
            or named.st_uid != opened.st_uid
            or named.st_nlink != opened.st_nlink
            or named.st_size != opened.st_size
        ):
            raise RuntimeError(
                "lifecycle wait channel named identity differs"
            )
        return opened
    finally:
        if named_directory_descriptor >= 0:
            os.close(named_directory_descriptor)


def _open_presealed_lifecycle_wait_channel(
    attempt_root: Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> tuple[int, int]:
    path = attempt_root / name
    if str(path) != binding.get("path"):
        raise RuntimeError(
            "lifecycle wait channel path binding differs"
        )
    directory_descriptor, _directory = _open_secure_directory(
        attempt_root
    )
    descriptor = -1
    try:
        named_before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            named_before.st_dev != binding["device"]
            or named_before.st_ino != binding["inode"]
            or named_before.st_mode != binding["mode"]
            or named_before.st_uid != binding["uid"]
            or named_before.st_nlink != 1
            or named_before.st_size != 0
        ):
            raise RuntimeError(
                "presealed lifecycle wait channel identity differs"
            )
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_DSYNC
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        _require_named_lifecycle_wait_channel(
            descriptor,
            binding,
            directory_descriptor=directory_descriptor,
            expected_size=0,
        )
        fcntl.flock(
            descriptor,
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
        _require_named_lifecycle_wait_channel(
            descriptor,
            binding,
            directory_descriptor=directory_descriptor,
            expected_size=0,
        )
        os.set_inheritable(descriptor, False)
        os.set_inheritable(directory_descriptor, False)
        result = (descriptor, directory_descriptor)
        descriptor = -1
        directory_descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _open_lifecycle_wait_channel_reader(
    attempt_root: Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> tuple[int, int]:
    path = attempt_root / name
    if str(path) != binding.get("path"):
        raise RuntimeError(
            "lifecycle wait channel path binding differs"
        )
    directory_descriptor, _directory = _open_secure_directory(
        attempt_root
    )
    descriptor = -1
    try:
        named_before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            named_before.st_dev != binding["device"]
            or named_before.st_ino != binding["inode"]
            or named_before.st_mode != binding["mode"]
            or named_before.st_uid != binding["uid"]
            or named_before.st_nlink != 1
            or named_before.st_size
            > LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES
        ):
            raise RuntimeError(
                "lifecycle wait channel reader identity differs"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        fcntl.flock(
            descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB
        )
        _require_named_lifecycle_wait_channel(
            descriptor,
            binding,
            directory_descriptor=directory_descriptor,
        )
        os.set_inheritable(descriptor, False)
        os.set_inheritable(directory_descriptor, False)
        result = (descriptor, directory_descriptor)
        descriptor = -1
        directory_descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _write_launcher_fault_channel_record(
    descriptor: int,
    binding: Mapping[str, Any],
    *,
    attempt_id: str,
    owner_nonce: str,
    launch_receipt_sha256: str,
    publisher: Mapping[str, str],
    failure: "LauncherExclusivePublishError",
) -> dict[str, Any]:
    failure_record = build_publish_failure_record(
        commit_state=failure.commit_state,
        stage=failure.stage,
        message=str(failure),
        directory_seal=failure.directory_seal,
        payload=failure.payload,
        temporary=failure.temporary,
        error_number=failure.error_number,
        secondary_failures=[],
    )
    record = {
        "schema_version": 1,
        "contract_type": FAULT_RECORD_CONTRACT_TYPE,
        "attempt_id": attempt_id,
        "owner_nonce": owner_nonce,
        "launch_receipt_sha256": launch_receipt_sha256,
        "publisher": dict(publisher),
        "fault_channel": dict(binding),
        "failure": failure_record,
        "recorded_at": _utc_now(),
    }
    record["fault_record_sha256"] = _canonical_digest(
        record, "fault_record_sha256"
    )
    payload = _canonical_json_bytes(record)
    frame = (
        FAULT_CHANNEL_PREFIX
        + f"{len(payload):08x}\n".encode("ascii")
        + payload
        + FAULT_CHANNEL_SHA_PREFIX
        + hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"\n"
    )
    if len(frame) > FAULT_CHANNEL_MAX_RECORD_BYTES:
        raise RuntimeError(
            "launcher fault channel record exceeds bound"
        )
    before = os.fstat(descriptor)
    if (
        before.st_dev != binding["device"]
        or before.st_ino != binding["inode"]
        or before.st_mode != binding["mode"]
        or before.st_uid != binding["uid"]
        or before.st_nlink != 1
        or before.st_size != 0
    ):
        raise RuntimeError(
            "launcher fault channel changed before write"
        )
    offset = 0
    while offset < len(frame):
        written = os.pwrite(
            descriptor, frame[offset:], offset
        )
        if written <= 0:
            raise RuntimeError(
                "launcher fault channel write made no progress"
            )
        offset += written
    os.fsync(descriptor)
    after = os.fstat(descriptor)
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_mode != before.st_mode
        or after.st_size != len(frame)
        or os.pread(descriptor, len(frame), 0) != frame
    ):
        raise RuntimeError(
            "launcher fault channel write differs"
        )
    return record


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_lifecycle_wait_channel_frame(
    record: Mapping[str, Any],
) -> tuple[bytes, bytes, bytes]:
    payload = _canonical_json_bytes(record)
    body = (
        LIFECYCLE_WAIT_CHANNEL_PREFIX
        + f"{len(payload):08x}\n".encode("ascii")
        + payload
        + FAULT_CHANNEL_SHA_PREFIX
        + hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"\n"
    )
    commit = (
        LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
        + hashlib.sha256(body).hexdigest().encode("ascii")
        + b"\n"
    )
    frame = body + commit
    if len(frame) > LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES:
        raise RuntimeError(
            "lifecycle wait status exceeds channel bound"
        )
    return body, commit, frame


def _pwrite_all(
    descriptor: int,
    data: bytes,
    offset: int,
    *,
    label: str,
) -> None:
    written_total = 0
    while written_total < len(data):
        try:
            written = os.pwrite(
                descriptor,
                data[written_total:],
                offset + written_total,
            )
        except InterruptedError:
            continue
        if written <= 0:
            raise RuntimeError(f"{label} write made no progress")
        written_total += written


def _write_lifecycle_wait_status(
    descriptor: int,
    directory_descriptor: int,
    binding: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    record = validate_lifecycle_wait_status(
        value,
        role=role,
        label=f"{role} lifecycle wait writer",
    )
    if record["wait_channel"] != dict(binding):
        raise RuntimeError(
            "lifecycle wait channel binding differs before write"
        )
    if (
        fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_DSYNC
    ) == 0:
        raise RuntimeError(
            "lifecycle wait channel descriptor is not O_DSYNC"
        )
    body, commit, frame = _build_lifecycle_wait_channel_frame(
        record
    )
    _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
        expected_size=0,
    )
    _pwrite_all(
        descriptor,
        body,
        0,
        label="lifecycle wait channel body",
    )
    os.fsync(descriptor)
    _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
        expected_size=len(body),
    )
    if os.pread(descriptor, len(body), 0) != body:
        raise RuntimeError(
            "lifecycle wait channel body differs"
        )
    _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
        expected_size=len(body),
    )
    # The descriptor was opened with O_DSYNC. A successful commit
    # pwrite is therefore the durability boundary; a second fsync
    # would create an ambiguous "committed bytes but fsync failed"
    # state.
    _pwrite_all(
        descriptor,
        commit,
        len(body),
        label="lifecycle wait channel commit",
    )
    _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
        expected_size=len(frame),
    )
    if os.pread(descriptor, len(frame), 0) != frame:
        raise RuntimeError("lifecycle wait channel frame differs")
    _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
        expected_size=len(frame),
    )
    return record


def _read_lifecycle_wait_status(
    descriptor: int,
    directory_descriptor: int,
    binding: Mapping[str, Any],
    *,
    role: str,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if set(expected_bindings) != (
        LIFECYCLE_WAIT_EXPECTED_BINDING_KEYS
    ):
        raise RuntimeError(
            "lifecycle wait expected bindings differ"
        )
    before = _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
    )
    if before.st_size > LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES:
        raise RuntimeError(
            "lifecycle wait channel exceeds bound before read"
        )
    chunks: list[bytes] = []
    offset = 0
    while offset <= LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES:
        try:
            chunk = os.pread(
                descriptor,
                min(
                    4096,
                    LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES
                    + 1
                    - offset,
                ),
                offset,
            )
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    content = b"".join(chunks)
    after = _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
    )
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_uid != before.st_uid
        or after.st_mode != before.st_mode
        or after.st_nlink != before.st_nlink
        or after.st_size != before.st_size
        or len(content) != before.st_size
    ):
        raise RuntimeError(
            "lifecycle wait channel changed during read"
        )
    if not content:
        final = _require_named_lifecycle_wait_channel(
            descriptor,
            binding,
            directory_descriptor=directory_descriptor,
            expected_size=0,
        )
        return {
            "state": "empty",
            "record": None,
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "channel_authority": {
                "path": binding["path"],
                "device": int(final.st_dev),
                "inode": int(final.st_ino),
                "mode": int(final.st_mode),
                "uid": int(final.st_uid),
                "nlink": int(final.st_nlink),
                "size": int(final.st_size),
                "directory_device": binding[
                    "directory_device"
                ],
                "directory_inode": binding["directory_inode"],
            },
        }
    if (
        len(content) > LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES
        or not content.startswith(
            LIFECYCLE_WAIT_CHANNEL_PREFIX
        )
    ):
        raise RuntimeError(
            "lifecycle wait channel magic or bound differs"
        )
    length_start = len(LIFECYCLE_WAIT_CHANNEL_PREFIX)
    length_end = length_start + 9
    length_line = content[length_start:length_end]
    if (
        len(length_line) != 9
        or length_line[-1:] != b"\n"
        or any(
            character not in b"0123456789abcdef"
            for character in length_line[:8]
        )
    ):
        raise RuntimeError(
            "lifecycle wait channel length differs"
        )
    payload_length = int(length_line[:8], 16)
    payload_start = length_end
    payload_end = payload_start + payload_length
    body_end = (
        payload_end
        + len(FAULT_CHANNEL_SHA_PREFIX)
        + 65
    )
    expected_size = (
        body_end
        + len(LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX)
        + 65
    )
    if len(content) != expected_size:
        raise RuntimeError(
            "lifecycle wait channel is uncommitted, partial, or trailing"
        )
    payload = content[payload_start:payload_end]
    expected_trailer = (
        FAULT_CHANNEL_SHA_PREFIX
        + hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"\n"
    )
    if content[payload_end:body_end] != expected_trailer:
        raise RuntimeError(
            "lifecycle wait channel SHA differs"
        )
    expected_commit = (
        LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
        + hashlib.sha256(content[:body_end]).hexdigest().encode(
            "ascii"
        )
        + b"\n"
    )
    if content[body_end:] != expected_commit:
        raise RuntimeError(
            "lifecycle wait channel commit differs"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "lifecycle wait channel JSON is invalid"
        ) from exc
    if (
        not isinstance(value, dict)
        or _canonical_json_bytes(value) != payload
    ):
        raise RuntimeError(
            "lifecycle wait channel JSON is not canonical"
        )
    try:
        record = validate_lifecycle_wait_status(
            value,
            role=role,
            label=f"{role} lifecycle wait reader",
        )
    except PreflightLaunchContractError as exc:
        raise RuntimeError(
            "lifecycle wait channel schema differs"
        ) from exc
    if record["wait_channel"] != dict(binding):
        raise RuntimeError(
            "lifecycle wait channel binding differs"
        )
    record_bindings = {
        key: record[key]
        for key in LIFECYCLE_WAIT_EXPECTED_BINDING_KEYS
    }
    if record_bindings != dict(expected_bindings):
        raise RuntimeError(
            "lifecycle wait status semantic bindings differ"
        )
    content_sha256 = hashlib.sha256(content).hexdigest()
    final = _require_named_lifecycle_wait_channel(
        descriptor,
        binding,
        directory_descriptor=directory_descriptor,
        expected_size=len(content),
    )
    return {
        "state": "valid_wait_status",
        "record": record,
        "size": len(content),
        "sha256": content_sha256,
        "channel_authority": {
            "path": binding["path"],
            "device": int(final.st_dev),
            "inode": int(final.st_ino),
            "mode": int(final.st_mode),
            "uid": int(final.st_uid),
            "nlink": int(final.st_nlink),
            "size": int(final.st_size),
            "directory_device": binding["directory_device"],
            "directory_inode": binding["directory_inode"],
        },
    }


def _read_fault_channel(
    descriptor: int,
    binding: Mapping[str, Any],
    *,
    attempt_id: str,
    owner_nonce: str,
    launch_receipt_sha256: str,
    publisher: Mapping[str, str],
) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != binding["device"]
        or before.st_ino != binding["inode"]
        or before.st_uid != binding["uid"]
        or before.st_mode != binding["mode"]
        or before.st_nlink != binding["nlink"]
        or before.st_size > FAULT_CHANNEL_MAX_RECORD_BYTES
    ):
        raise RuntimeError(
            "fault channel identity or size differs before read"
        )
    chunks: list[bytes] = []
    offset = 0
    while offset <= FAULT_CHANNEL_MAX_RECORD_BYTES:
        try:
            chunk = os.pread(
                descriptor,
                min(
                    4096,
                    FAULT_CHANNEL_MAX_RECORD_BYTES + 1 - offset,
                ),
                offset,
            )
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    content = b"".join(chunks)
    after = os.fstat(descriptor)
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_uid != before.st_uid
        or after.st_mode != before.st_mode
        or after.st_nlink != before.st_nlink
        or after.st_size != before.st_size
        or len(content) != before.st_size
    ):
        raise RuntimeError("fault channel changed during read")
    if not content:
        return {
            "state": "empty",
            "record": None,
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    if len(content) > FAULT_CHANNEL_MAX_RECORD_BYTES:
        raise RuntimeError("fault channel record exceeds bound")
    if not content.startswith(FAULT_CHANNEL_PREFIX):
        raise RuntimeError("fault channel magic or version differs")
    length_start = len(FAULT_CHANNEL_PREFIX)
    length_end = length_start + 9
    length_line = content[length_start:length_end]
    if (
        len(length_line) != 9
        or length_line[-1:] != b"\n"
        or any(
            character not in b"0123456789abcdef"
            for character in length_line[:8]
        )
    ):
        raise RuntimeError("fault channel length field differs")
    payload_length = int(length_line[:8], 16)
    payload_start = length_end
    payload_end = payload_start + payload_length
    trailer = (
        FAULT_CHANNEL_SHA_PREFIX
        + b"0" * 64
        + b"\n"
    )
    expected_size = payload_end + len(trailer)
    if len(content) != expected_size:
        raise RuntimeError(
            "fault channel record is partial or has trailing bytes"
        )
    payload = content[payload_start:payload_end]
    observed_trailer = content[payload_end:]
    expected_trailer = (
        FAULT_CHANNEL_SHA_PREFIX
        + hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"\n"
    )
    if observed_trailer != expected_trailer:
        raise RuntimeError("fault channel SHA trailer differs")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("fault channel JSON is invalid") from exc
    if (
        not isinstance(value, dict)
        or _canonical_json_bytes(value) != payload
    ):
        raise RuntimeError("fault channel JSON is not canonical")
    expected_keys = {
        "schema_version",
        "contract_type",
        "attempt_id",
        "owner_nonce",
        "launch_receipt_sha256",
        "publisher",
        "fault_channel",
        "failure",
        "recorded_at",
        "fault_record_sha256",
    }
    failure = value.get("failure")
    try:
        validated_failure = validate_publish_failure_record(
            failure if isinstance(failure, dict) else {}
        )
    except PreflightLaunchContractError as exc:
        raise RuntimeError(
            "fault channel publication failure schema differs"
        ) from exc
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != FAULT_RECORD_CONTRACT_TYPE
        or value.get("attempt_id") != attempt_id
        or value.get("owner_nonce") != owner_nonce
        or value.get("launch_receipt_sha256")
        != launch_receipt_sha256
        or value.get("publisher") != dict(publisher)
        or value.get("fault_channel") != dict(binding)
        or failure != validated_failure
        or value.get("fault_record_sha256")
        != _canonical_digest(value, "fault_record_sha256")
    ):
        raise RuntimeError(
            "fault channel schema or binding differs"
        )
    return {
        "state": "valid_fault",
        "record": value,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _read_exact_wrapper_exit(
    path: Path,
    *,
    policy_sha256: str,
) -> dict[str, Any]:
    directory_descriptor, _directory = _open_secure_directory(
        path.parent
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o644
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > 1024 * 1024
        ):
            raise RuntimeError(
                "wrapper exit identity or size differs"
            )
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        content = b"".join(chunks)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_uid != before.st_uid
            or after.st_mode != before.st_mode
            or after.st_nlink != before.st_nlink
            or after.st_size != before.st_size
            or len(content) != before.st_size
        ):
            raise RuntimeError("wrapper exit changed during read")
    finally:
        if descriptor >= 0:
            descriptor_to_close = descriptor
            descriptor = -1
            os.close(descriptor_to_close)
        os.close(directory_descriptor)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("wrapper exit JSON is invalid") from exc
    if (
        not isinstance(value, dict)
        or _canonical_json_bytes(value) != content
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_canonical_preflight_wrapper_exit_v4"
        or value.get("policy_sha256") != policy_sha256
        or value.get("wrapper_exit_sha256")
        != _canonical_digest(value, "wrapper_exit_sha256")
    ):
        raise RuntimeError(
            "wrapper exit schema, binding, or digest differs"
        )
    return {
        "value": value,
        "binding": build_artifact_binding(
            path=str(path),
            sha256=hashlib.sha256(content).hexdigest(),
            canonical_sha256=value["wrapper_exit_sha256"],
        ),
    }


def _evaluate_wrapper_outcome(
    *,
    returncode: int,
    fault_snapshot: Mapping[str, Any] | None,
    fault_validation_failure: Mapping[str, Any] | None,
    fault_close_failure: Mapping[str, Any] | None,
    wrapper_exit_reader,
) -> dict[str, Any]:
    if fault_validation_failure is not None:
        return {
            "status": "invalid_fault_channel",
            "exit_code": 125,
            "failure": dict(fault_validation_failure),
            "wrapper_exit": None,
        }
    if fault_close_failure is not None:
        return {
            "status": "fault_channel_close_failed",
            "exit_code": 125,
            "failure": dict(fault_close_failure),
            "wrapper_exit": None,
        }
    if fault_snapshot is None:
        return {
            "status": "fault_channel_snapshot_absent",
            "exit_code": 125,
            "failure": None,
            "wrapper_exit": None,
        }
    if fault_snapshot.get("state") == "valid_fault":
        return {
            "status": "typed_publish_failure",
            "exit_code": 125,
            "failure": dict(fault_snapshot["record"]["failure"]),
            "wrapper_exit": None,
        }
    if fault_snapshot.get("state") != "empty":
        return {
            "status": "fault_channel_state_invalid",
            "exit_code": 125,
            "failure": None,
            "wrapper_exit": None,
        }
    if returncode != 0:
        return {
            "status": "wrapper_child_failed",
            "exit_code": (
                returncode if returncode > 0 else 128 - returncode
            ),
            "failure": None,
            "wrapper_exit": None,
        }
    try:
        wrapper_exit = wrapper_exit_reader()
    except BaseException as exc:
        return {
            "status": "wrapper_exit_invalid",
            "exit_code": 125,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "wrapper_exit": None,
        }
    value = wrapper_exit["value"]
    if (
        value.get("exit_code") != 0
        or value.get("controller_exit_code") != 0
        or value.get("launch_failure") is not None
    ):
        return {
            "status": "wrapper_exit_not_success",
            "exit_code": 125,
            "failure": None,
            "wrapper_exit": wrapper_exit["binding"],
        }
    return {
        "status": "success",
        "exit_code": 0,
        "failure": None,
        "wrapper_exit": wrapper_exit["binding"],
    }


def _evaluate_gate_outcome(
    *,
    returncode: int | None,
    exec_failure: Mapping[str, Any] | None,
    fault_snapshot: Mapping[str, Any] | None,
    fault_validation_failure: Mapping[str, Any] | None,
    fault_close_failure: Mapping[str, Any] | None,
    wrapper_exit_reader,
) -> dict[str, Any]:
    if returncode is not None:
        if exec_failure is not None:
            raise RuntimeError(
                "gate outcome has both returncode and exec failure"
            )
        return _evaluate_wrapper_outcome(
            returncode=returncode,
            fault_snapshot=fault_snapshot,
            fault_validation_failure=fault_validation_failure,
            fault_close_failure=fault_close_failure,
            wrapper_exit_reader=wrapper_exit_reader,
        )
    if exec_failure is None:
        raise RuntimeError(
            "gate exec outcome is missing its typed failure"
        )
    if fault_validation_failure is not None:
        return {
            "status": "invalid_fault_channel",
            "exit_code": 125,
            "failure": dict(fault_validation_failure),
            "wrapper_exit": None,
        }
    if fault_close_failure is not None:
        return {
            "status": "fault_channel_close_failed",
            "exit_code": 125,
            "failure": dict(fault_close_failure),
            "wrapper_exit": None,
        }
    if (
        fault_snapshot is None
        or fault_snapshot.get("state") != "empty"
    ):
        return {
            "status": "exec_error_fault_channel_not_empty",
            "exit_code": 125,
            "failure": None,
            "wrapper_exit": None,
        }
    return {
        "status": "exec_error",
        "exit_code": 126,
        "failure": dict(exec_failure),
        "wrapper_exit": None,
    }


def _adjudicate_gate_execution_outcome(
    gate_execution: Mapping[str, Any],
    *,
    wrapper_exit_path: Path,
    policy_sha256: str,
) -> dict[str, Any]:
    outcome = gate_execution.get("wrapper_outcome")
    if (
        not isinstance(outcome, Mapping)
        or set(outcome)
        != {"status", "exit_code", "failure", "wrapper_exit"}
    ):
        raise RuntimeError(
            "gate execution wrapper outcome schema differs"
        )
    expected = _evaluate_gate_outcome(
        returncode=gate_execution.get("returncode"),
        exec_failure=gate_execution.get("failure"),
        fault_snapshot=gate_execution.get(
            "fault_channel_snapshot"
        ),
        fault_validation_failure=gate_execution.get(
            "fault_channel_validation_failure"
        ),
        fault_close_failure=gate_execution.get(
            "fault_channel_close_failure"
        ),
        wrapper_exit_reader=lambda: _read_exact_wrapper_exit(
            wrapper_exit_path,
            policy_sha256=policy_sha256,
        ),
    )
    if dict(outcome) != expected:
        raise RuntimeError(
            "gate execution wrapper outcome differs"
        )
    controller_exit_code = expected["exit_code"]
    if (
        type(controller_exit_code) is not int
        or controller_exit_code < 0
        or controller_exit_code > 255
    ):
        raise RuntimeError(
            "gate execution controller exit code differs"
        )
    adjudicated_outcome = (
        "completed"
        if expected["status"] == "success"
        and controller_exit_code == 0
        else "controller_failed"
    )
    return {
        "wrapper_outcome": expected,
        "controller_exit_code": controller_exit_code,
        "adjudicated_outcome": adjudicated_outcome,
    }


class LauncherExclusivePublishError(RuntimeError):
    """Typed fail-closed state for a launcher publication."""

    def __init__(
        self,
        commit_state: str,
        message: str,
        *,
        stage: str,
        directory_seal: Mapping[str, int],
        payload: Mapping[str, Any],
        temporary: Mapping[str, Any] | None,
        error_number: int | None,
        quarantined: bool,
    ) -> None:
        if commit_state not in {
            "precommit_failed_clean",
            "durability_unknown_quarantined",
            "committed_cleanup_error",
            "collision",
        }:
            raise ValueError(
                "launcher publication commit state is invalid: "
                f"{commit_state}"
            )
        super().__init__(f"{commit_state}: {message}")
        self.commit_state = commit_state
        self.status = commit_state
        self.stage = stage
        self.directory_seal = dict(directory_seal)
        self.payload = dict(payload)
        self.temporary = (
            None if temporary is None else dict(temporary)
        )
        self.error_number = error_number
        self.quarantined = quarantined
        self.secondary_failures: list[dict[str, str]] = []

    def add_secondary_failure(
        self, *, stage: str, failure: BaseException
    ) -> None:
        self.secondary_failures.append(
            build_finalization_secondary_failure(
                stage=stage,
                failure_type=type(failure).__name__,
                message=str(failure),
            )
        )


class LauncherTerminalPublishError(OSError):
    """A terminal publication failed before a typed commit state existed."""

    def __init__(
        self, path: Path, failure: BaseException
    ) -> None:
        super().__init__(
            f"terminal publication failed for {path}: {failure}"
        )
        self.path = path
        self.failure = failure
        self.secondary_failures: list[dict[str, str]] = []

    def add_secondary_failure(
        self, *, stage: str, failure: BaseException
    ) -> None:
        self.secondary_failures.append(
            build_finalization_secondary_failure(
                stage=stage,
                failure_type=type(failure).__name__,
                message=str(failure),
            )
        )


class LauncherGateFaultError(RuntimeError):
    """A gate fault channel became valid or invalid."""

    def __init__(
        self,
        status: str,
        *,
        snapshot: Mapping[str, Any] | None,
        failure: BaseException | None = None,
    ) -> None:
        super().__init__(
            status if failure is None else f"{status}: {failure}"
        )
        self.status = status
        self.snapshot = (
            None if snapshot is None else dict(snapshot)
        )
        self.failure = failure
        self.secondary_failures: list[dict[str, str]] = []

    def add_secondary_failure(
        self, *, stage: str, failure: BaseException
    ) -> None:
        self.secondary_failures.append(
            build_finalization_secondary_failure(
                stage=stage,
                failure_type=type(failure).__name__,
                message=str(failure),
            )
        )


class PaneFaultConsumerReservationError(RuntimeError):
    """A consumer reservation failed, with exact cleanup evidence."""

    def __init__(self, failure: BaseException) -> None:
        super().__init__(
            "pane fault consumer reservation failed: "
            f"{type(failure).__name__}: {failure}"
        )
        self.failure = failure
        self.secondary_failures: list[dict[str, str]] = []

    def add_secondary_failure(
        self, *, stage: str, failure: BaseException
    ) -> None:
        self.secondary_failures.append(
            build_finalization_secondary_failure(
                stage=stage,
                failure_type=type(failure).__name__,
                message=str(failure),
            )
        )


def _propagate_launcher_publish_error(
    failure: BaseException,
) -> None:
    if isinstance(
        failure,
        (
            LauncherExclusivePublishError,
            LauncherTerminalPublishError,
        ),
    ):
        raise failure


class _LauncherDirectoryDurabilityUnknown(RuntimeError):
    """A launcher directory mutation has unknown durability."""


class _LauncherPublicationSealMismatch(RuntimeError):
    """A launcher publication name no longer refers to its sealed inode."""


def _launcher_checked_close(descriptor: int, label: str) -> None:
    try:
        os.close(descriptor)
    except OSError as exc:
        raise RuntimeError(f"{label} close failed") from exc


def _launcher_open_publication_directory(
    path: Path,
) -> tuple[int, os.stat_result]:
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or path.is_symlink()
    ):
        raise RuntimeError(
            f"launcher publication directory path is not exact: {path}"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC,
    )
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) & 0o022
        ):
            raise RuntimeError(
                "launcher publication directory identity or "
                f"permissions differ: {path}"
            )
    except BaseException:
        _launcher_checked_close(
            descriptor, "launcher publication directory"
        )
        raise
    return descriptor, opened


def _launcher_read_publication_file(
    directory_descriptor: int,
    name: str,
    *,
    expected_content: bytes | None = None,
    expected_identity: tuple[int, int] | None = None,
    fsync_file: bool = False,
) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o644
            or before.st_nlink not in {1, 2}
            or (
                expected_identity is not None
                and (before.st_dev, before.st_ino)
                != expected_identity
            )
        ):
            raise RuntimeError(
                "launcher publication file identity or permissions "
                f"differ: {name}"
            )
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_mode != after.st_mode
            or before.st_uid != after.st_uid
            or before.st_nlink != after.st_nlink
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(content) != after.st_size
            or (
                expected_content is not None
                and content != expected_content
            )
        ):
            raise RuntimeError(
                "launcher publication file changed or content "
                f"differs: {name}"
            )
        if fsync_file:
            os.fsync(descriptor)
    finally:
        _launcher_checked_close(
            descriptor, "launcher publication file"
        )
    return content, before


def _launcher_fsync_dirfd(descriptor: int) -> None:
    os.fsync(descriptor)


def _launcher_cleanup_temporary(
    directory_descriptor: int,
    temporary_name: str,
    temporary_seal: Mapping[str, Any],
) -> None:
    current = os.stat(
        temporary_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != temporary_seal["device"]
        or current.st_ino != temporary_seal["inode"]
        or (
            temporary_seal.get("ctime_ns") is not None
            and current.st_ctime_ns != temporary_seal["ctime_ns"]
        )
        or (
            temporary_seal.get("size") is not None
            and current.st_size != temporary_seal["size"]
        )
        or (
            temporary_seal.get("nlink") is not None
            and current.st_nlink != temporary_seal["nlink"]
        )
    ):
        raise _LauncherPublicationSealMismatch(
            "launcher publication temporary identity changed"
        )
    expected_content = temporary_seal.get("content")
    if expected_content is not None:
        observed, _identity = _launcher_read_publication_file(
            directory_descriptor,
            temporary_name,
            expected_content=expected_content,
            expected_identity=(
                int(temporary_seal["device"]),
                int(temporary_seal["inode"]),
            ),
        )
        if (
            hashlib.sha256(observed).hexdigest()
            != temporary_seal["sha256"]
        ):
            raise _LauncherPublicationSealMismatch(
                "launcher publication temporary content changed"
            )
    os.unlink(temporary_name, dir_fd=directory_descriptor)
    try:
        _launcher_fsync_dirfd(directory_descriptor)
    except BaseException as exc:
        raise _LauncherDirectoryDurabilityUnknown(
            "launcher temporary unlink directory fsync failed"
        ) from exc


def _launcher_rollback_final(
    directory_descriptor: int,
    final_name: str,
    final_identity: tuple[int, int],
) -> None:
    current = os.stat(
        final_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != final_identity
    ):
        raise _LauncherPublicationSealMismatch(
            "launcher uncommitted final identity changed"
        )
    os.unlink(final_name, dir_fd=directory_descriptor)
    try:
        _launcher_fsync_dirfd(directory_descriptor)
    except BaseException as exc:
        raise _LauncherDirectoryDurabilityUnknown(
            "launcher final rollback directory fsync failed"
        ) from exc


def _write_exclusive(
    path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    content = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    payload_seal = {
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    directory_descriptor, directory_identity = (
        _launcher_open_publication_directory(path.parent)
    )
    directory_seal = {
        "device": int(directory_identity.st_dev),
        "inode": int(directory_identity.st_ino),
        "uid": int(directory_identity.st_uid),
        "mode": int(stat.S_IMODE(directory_identity.st_mode)),
    }
    temporary_name = (
        f".{path.name}.publish-{secrets.token_hex(32)}"
    )
    temporary_descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    temporary_seal: dict[str, Any] | None = None
    linked = False
    committed = False
    quarantined = False

    def failure(
        commit_state: str,
        stage: str,
        message: str,
        cause: BaseException | None = None,
    ) -> LauncherExclusivePublishError:
        return LauncherExclusivePublishError(
            commit_state,
            message,
            stage=stage,
            directory_seal=directory_seal,
            payload=payload_seal,
            temporary=temporary_seal,
            error_number=(
                cause.errno if isinstance(cause, OSError) else None
            ),
            quarantined=quarantined,
        )

    def rollback_final(stage: str) -> None:
        nonlocal linked, quarantined
        if not linked or temporary_identity is None:
            return
        try:
            _launcher_rollback_final(
                directory_descriptor,
                path.name,
                temporary_identity,
            )
        except BaseException as exc:
            quarantined = True
            raise failure(
                "durability_unknown_quarantined",
                stage,
                "launcher uncommitted final rollback failed",
                exc,
            ) from exc
        linked = False

    def cleanup_temporary(
        *, stage: str, after_commit: bool
    ) -> None:
        nonlocal temporary_identity, temporary_seal, quarantined
        if temporary_seal is None:
            return
        try:
            _launcher_cleanup_temporary(
                directory_descriptor,
                temporary_name,
                temporary_seal,
            )
        except _LauncherPublicationSealMismatch as exc:
            quarantined = True
            raise failure(
                (
                    "committed_cleanup_error"
                    if after_commit
                    else "collision"
                ),
                stage,
                "launcher sealed temporary changed",
                exc,
            ) from exc
        except BaseException as exc:
            quarantined = True
            raise failure(
                (
                    "committed_cleanup_error"
                    if after_commit
                    else "durability_unknown_quarantined"
                ),
                stage,
                "launcher temporary cleanup failed",
                exc,
            ) from exc
        temporary_identity = None
        temporary_seal = None

    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        opened = os.fstat(temporary_descriptor)
        temporary_identity = (opened.st_dev, opened.st_ino)
        temporary_seal = {
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "ctime_ns": None,
            "size": None,
            "sha256": None,
            "content": None,
            "nlink": 1,
        }
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
        ):
            raise RuntimeError(
                "launcher publication temporary identity differs"
            )
        offset = 0
        while offset < len(content):
            try:
                written = os.write(
                    temporary_descriptor, content[offset:]
                )
            except InterruptedError:
                continue
            if written <= 0:
                raise RuntimeError(
                    "launcher publication write made no progress"
                )
            offset += written
        os.fchmod(temporary_descriptor, 0o644)
        os.fsync(temporary_descriptor)
        descriptor_to_close = temporary_descriptor
        temporary_descriptor = -1
        _launcher_checked_close(
            descriptor_to_close, "launcher publication temporary"
        )
        reopened_content, reopened = (
            _launcher_read_publication_file(
                directory_descriptor,
                temporary_name,
                expected_content=content,
                expected_identity=temporary_identity,
            )
        )
        if (
            reopened_content != content
            or reopened.st_nlink != 1
        ):
            raise RuntimeError(
                "launcher publication temporary verification differs"
            )
        temporary_seal = {
            "device": int(reopened.st_dev),
            "inode": int(reopened.st_ino),
            "ctime_ns": int(reopened.st_ctime_ns),
            "size": int(reopened.st_size),
            "sha256": payload_seal["sha256"],
            "content": content,
            "nlink": 1,
        }
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError as exc:
            try:
                existing_content, _existing = (
                    _launcher_read_publication_file(
                        directory_descriptor,
                        path.name,
                        expected_content=content,
                        fsync_file=True,
                    )
                )
            except BaseException as collision:
                raise failure(
                    "collision",
                    "existing_final_verify",
                    "launcher final exists with different or unsafe "
                    "content",
                    collision,
                ) from collision
            if existing_content != content:
                raise failure(
                    "collision",
                    "existing_final_compare",
                    "launcher final exists with different content",
                    exc,
                ) from exc
            try:
                _launcher_fsync_dirfd(directory_descriptor)
            except BaseException as sync_failure:
                quarantined = True
                raise failure(
                    "durability_unknown_quarantined",
                    "existing_final_directory_fsync",
                    "launcher exact existing final fsync failed",
                    sync_failure,
                ) from sync_failure
            committed = True
            cleanup_temporary(
                stage="existing_final_temporary_cleanup",
                after_commit=True,
            )
            return {
                "status": "already_committed_exact",
                "path": str(path),
                "payload_size": len(content),
                "payload_sha256": payload_seal["sha256"],
            }
        linked_final = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        temporary_after_link = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(linked_final.st_mode)
            or (
                linked_final.st_dev,
                linked_final.st_ino,
            )
            != temporary_identity
            or (
                temporary_after_link.st_dev,
                temporary_after_link.st_ino,
            )
            != temporary_identity
            or linked_final.st_nlink != 2
            or temporary_after_link.st_nlink != 2
        ):
            rollback_final("linked_identity_rollback")
            raise failure(
                "precommit_failed_clean",
                "linked_identity_verify",
                "launcher linked names differ",
            )
        linked_temporary_content, linked_temporary = (
            _launcher_read_publication_file(
                directory_descriptor,
                temporary_name,
                expected_content=content,
                expected_identity=temporary_identity,
            )
        )
        temporary_seal = {
            **temporary_seal,
            "ctime_ns": int(linked_temporary.st_ctime_ns),
            "nlink": 2,
        }
        final_content, final_identity = (
            _launcher_read_publication_file(
                directory_descriptor,
                path.name,
                expected_content=content,
                expected_identity=temporary_identity,
            )
        )
        if (
            linked_temporary_content != content
            or final_content != content
            or linked_temporary.st_nlink != 2
            or final_identity.st_nlink != 2
        ):
            rollback_final("linked_content_rollback")
            raise failure(
                "precommit_failed_clean",
                "linked_content_verify",
                "launcher linked content differs",
            )
        try:
            _launcher_fsync_dirfd(directory_descriptor)
        except BaseException as exc:
            quarantined = True
            raise failure(
                "durability_unknown_quarantined",
                "final_link_directory_fsync",
                "launcher final link directory fsync failed",
                exc,
            ) from exc
        committed = True
        cleanup_temporary(
            stage="committed_temporary_cleanup",
            after_commit=True,
        )
        return {
            "status": "committed",
            "path": str(path),
            "payload_size": len(content),
            "payload_sha256": payload_seal["sha256"],
            "directory_device": int(directory_identity.st_dev),
            "directory_inode": int(directory_identity.st_ino),
        }
    except BaseException as exc:
        if temporary_descriptor >= 0:
            descriptor_to_close = temporary_descriptor
            temporary_descriptor = -1
            try:
                _launcher_checked_close(
                    descriptor_to_close,
                    "launcher publication temporary",
                )
            except BaseException as close_failure:
                if not isinstance(
                    exc, LauncherExclusivePublishError
                ):
                    exc = failure(
                        "precommit_failed_clean",
                        "temporary_close",
                        f"{exc}; close failure: {close_failure}",
                        close_failure,
                    )
        if isinstance(exc, LauncherExclusivePublishError):
            quarantined = quarantined or exc.quarantined
        if not quarantined and linked and not committed:
            rollback_final("exception_rollback")
        if not quarantined and temporary_seal is not None:
            cleanup_temporary(
                stage="exception_temporary_cleanup",
                after_commit=committed,
            )
        if isinstance(exc, LauncherExclusivePublishError):
            raise
        raise failure(
            (
                "committed_cleanup_error"
                if committed
                else "precommit_failed_clean"
            ),
            "publication",
            str(exc),
            exc,
        ) from exc
    finally:
        active_failure = sys.exc_info()[1]
        descriptor_to_close = directory_descriptor
        directory_descriptor = -1
        try:
            _launcher_checked_close(
                descriptor_to_close,
                "launcher publication directory",
            )
        except BaseException as close_failure:
            if active_failure is not None:
                state = (
                    active_failure.commit_state
                    if isinstance(
                        active_failure,
                        LauncherExclusivePublishError,
                    )
                    else (
                        "committed_cleanup_error"
                        if committed
                        else "precommit_failed_clean"
                    )
                )
                raise failure(
                    state,
                    "directory_close",
                    f"{active_failure}; directory close failed: "
                    f"{close_failure}",
                    close_failure,
                ) from active_failure
            raise failure(
                (
                    "committed_cleanup_error"
                    if committed
                    else "precommit_failed_clean"
                ),
                "directory_close",
                str(close_failure),
                close_failure,
            ) from close_failure


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_exact_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"{label} path is not absolute")
    try:
        resolved = path.resolve(strict=True)
        value = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} directory is unavailable") from exc
    if (
        resolved != path
        or path.is_symlink()
        or not stat.S_ISDIR(value.st_mode)
    ):
        raise RuntimeError(f"{label} directory path is not exact")
    return path


def _read_exact_json_artifact(
    path: Path,
    *,
    expected_path: Path,
    canonical_field: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path != expected_path or not path.is_absolute():
        raise RuntimeError(f"{label} path differs")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_CLOEXEC"):
        raise RuntimeError(f"{label} requires no-follow descriptors")
    try:
        if (
            path.resolve(strict=True) != path
            or path.is_symlink()
            or path.parent.resolve(strict=True) != path.parent
        ):
            raise RuntimeError(f"{label} path is not exact")
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    source = b"".join(chunks)
    identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_mode),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    if (
        identity
        != (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        or not stat.S_ISREG(before.st_mode)
        or before.st_size != len(source)
    ):
        raise RuntimeError(f"{label} identity changed during read")
    try:
        value = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} JSON differs") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    canonical = value.get(canonical_field)
    if (
        not isinstance(canonical, str)
        or not HEX64.fullmatch(canonical)
        or canonical != _canonical_digest(value, canonical_field)
    ):
        raise RuntimeError(f"{label} canonical digest differs")
    return value, {
        "path": str(path),
        "sha256": hashlib.sha256(source).hexdigest(),
        "canonical_sha256": canonical,
        "mtime_ns": int(before.st_mtime_ns),
    }


def _exact_regular_tree_files(root: Path) -> list[Path]:
    _require_exact_directory(root, "legacy prepared policy root")
    files: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(
                os.scandir(directory), key=lambda item: item.name
            )
        except OSError as exc:
            raise RuntimeError(
                "legacy prepared policy tree is unreadable"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise RuntimeError(
                    "legacy prepared policy tree contains a symlink"
                )
            if entry.is_dir(follow_symlinks=False):
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                files.append(path)
            else:
                raise RuntimeError(
                    "legacy prepared policy tree contains a special entry"
                )

    visit(root)
    return sorted(
        files, key=lambda path: path.relative_to(root).as_posix()
    )


def _legacy_policy_tree_snapshot(root: Path) -> dict[str, Any]:
    files = _exact_regular_tree_files(root)
    rows: list[bytes] = []
    content_bytes = 0
    for path in files:
        source, _identity = _bootstrap_read_file(
            path, None, "legacy prepared policy tree file"
        )
        relative = path.relative_to(root).as_posix()
        content_bytes += len(source)
        rows.append(
            relative.encode("utf-8")
            + b"\0"
            + str(len(source)).encode("ascii")
            + b"\0"
            + hashlib.sha256(source).hexdigest().encode("ascii")
            + b"\n"
        )
    return {
        "derivation": LEGACY_POLICY_TREE_DERIVATION,
        "sha256": hashlib.sha256(b"".join(rows)).hexdigest(),
        "file_count": len(files),
        "content_bytes": content_bytes,
        "serialized_bytes": sum(len(row) for row in rows),
        "symlink_count": 0,
    }


def _ensure_exact_archive_directory(
    campaign_root: Path,
    policy_sha256: str,
) -> Path:
    exact_root = _require_exact_directory(
        campaign_root, "legacy archive campaign root"
    )
    return _ensure_secure_leaf_directories(
        exact_root,
        (
            "untracked_failure_archives",
            "by_policy",
            policy_sha256,
        ),
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _json_binding(path: Path, canonical_field: str) -> dict[str, str]:
    value = _load_json(path, path.name)
    canonical = value.get(canonical_field)
    if (
        not isinstance(canonical, str)
        or not HEX64.fullmatch(canonical)
        or canonical != _canonical_digest(value, canonical_field)
    ):
        raise RuntimeError(f"{path.name} canonical digest differs")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "canonical_sha256": canonical,
    }


def _file_binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"bound file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    value = path.stat()
    return build_file_identity(
        path=str(path.resolve()),
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
    )


def _opened_file_identity(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"bound file is not regular: {path}")
    return build_file_identity(
        path=str(path.resolve()),
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
    )


def _sealed_launcher_publication_read(
    path: Path,
    *,
    expected_content: bytes,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not path.is_absolute()
        or path.name != "preclaim_failure_intent.json"
    ):
        raise RuntimeError(f"{label} path differs")
    directory_descriptor, _directory_identity = (
        _launcher_open_publication_directory(path.parent)
    )
    try:
        content, opened = _launcher_read_publication_file(
            directory_descriptor,
            path.name,
            expected_content=expected_content,
            fsync_file=True,
        )
        named = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or opened.st_mode != named.st_mode
            or opened.st_uid != named.st_uid
            or opened.st_nlink != 1
            or named.st_nlink != 1
            or opened.st_nlink != named.st_nlink
            or opened.st_size != named.st_size
            or opened.st_mtime_ns != named.st_mtime_ns
        ):
            raise RuntimeError(f"{label} named identity changed")
        _launcher_fsync_dirfd(directory_descriptor)
    finally:
        _launcher_checked_close(
            directory_descriptor, f"{label} directory"
        )
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} content is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} content is not an object")
    identity = build_file_identity(
        path=str(path),
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
        mode=int(opened.st_mode),
        size=int(opened.st_size),
    )
    return value, identity


def _sealed_invalid_wrapper_claim_evidence(
    path: Path,
) -> dict[str, Any]:
    if not path.is_absolute():
        raise RuntimeError("invalid wrapper claim path is not absolute")
    directory_descriptor, _directory_identity = (
        _launcher_open_publication_directory(path.parent)
    )
    try:
        content, opened = _launcher_read_publication_file(
            directory_descriptor,
            path.name,
            fsync_file=True,
        )
        named = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or opened.st_mode != named.st_mode
            or opened.st_uid != named.st_uid
            or opened.st_nlink != named.st_nlink
            or opened.st_size != named.st_size
            or opened.st_mtime_ns != named.st_mtime_ns
        ):
            raise RuntimeError(
                "invalid wrapper claim named identity changed"
            )
    finally:
        _launcher_checked_close(
            directory_descriptor,
            "invalid wrapper claim directory",
        )
    return build_invalid_claim_evidence(
        raw_content_sha256=hashlib.sha256(content).hexdigest(),
        file_identity=build_file_identity(
            path=str(path),
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
            mode=int(opened.st_mode),
            size=int(opened.st_size),
        ),
    )


def _publish_or_resume_preclaim_failure_intent(
    path: Path,
    *,
    attempt_id: str,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    wrapper_claim_path: Path,
    pane_fault_consumer_chain: Mapping[str, Any],
    tmux_owner_seal: Mapping[str, Any],
    reason: str,
    deadline_observation: Mapping[str, Any],
    observed_at: str,
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.name != "preclaim_failure_intent.json"
    ):
        raise RuntimeError("preclaim failure intent path differs")
    if not wrapper_claim_path.is_absolute():
        raise RuntimeError("wrapper claim path is not absolute")
    invalid_claim_evidence = (
        _sealed_invalid_wrapper_claim_evidence(
            wrapper_claim_path
        )
        if reason == "invalid_claim"
        else None
    )
    stage = {
        "invalid_claim": "wrapper_claim_validation",
        "claim_timeout": "wrapper_claim_wait_deadline",
    }.get(reason)
    if stage is None:
        raise RuntimeError("preclaim failure reason differs")
    intent = build_preclaim_failure_intent(
        attempt_id=attempt_id,
        launch_receipt=launch_receipt,
        launch_receipt_identity=launch_receipt_identity,
        verified_implementations=verified_implementations,
        wrapper_claim_path=str(wrapper_claim_path),
        pane_fault_consumer_chain=pane_fault_consumer_chain,
        controller_owner_seal=tmux_owner_seal,
        reason=reason,
        stage=stage,
        deadline_observation=deadline_observation,
        invalid_claim_evidence=invalid_claim_evidence,
        observed_at=observed_at,
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
    )
    expected_content = (
        json.dumps(
            intent,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_exclusive(path, intent)
    observed, identity = _sealed_launcher_publication_read(
        path,
        expected_content=expected_content,
        label="preclaim failure intent",
    )
    validated = validate_preclaim_failure_intent(
        observed,
        verified_implementations=verified_implementations,
        expected_wrapper_claim_path=str(wrapper_claim_path),
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
        expected_receipt=launch_receipt,
        expected_receipt_identity=launch_receipt_identity,
        expected_consumer_chain=pane_fault_consumer_chain,
        label="published preclaim failure intent",
    )
    if validated != intent:
        raise RuntimeError("published preclaim failure intent differs")
    return {
        "intent": validated,
        "artifact": build_artifact_binding(
            path=str(path),
            sha256=hashlib.sha256(expected_content).hexdigest(),
            canonical_sha256=validated[
                "preclaim_failure_intent_sha256"
            ],
        ),
        "file_identity": identity,
    }


def _optional_file_binding(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    binding: dict[str, Any] = _file_binding(path)
    binding["size"] = path.stat().st_size
    return binding


def _require_hex64(value: str, label: str) -> str:
    if not HEX64.fullmatch(value):
        raise RuntimeError(f"{label} is not 64 lowercase hex characters")
    return value


def _require_git_oid(value: str, label: str) -> str:
    if not GIT_OID.fullmatch(value):
        raise RuntimeError(f"{label} is not a valid Git object ID")
    return value


def _git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _verified_git_state(repo_root: Path) -> dict[str, str]:
    head = _git_output(repo_root, "rev-parse", "HEAD")
    origin = _git_output(repo_root, "rev-parse", "origin/master")
    branch = _git_output(repo_root, "branch", "--show-current")
    status = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if branch != "master":
        raise RuntimeError(f"launcher branch differs: {branch!r}")
    if head != origin:
        raise RuntimeError("launcher HEAD differs from origin/master")
    if status:
        raise RuntimeError("launcher Git worktree is not clean")
    return {
        "branch": branch,
        "head_sha": _require_git_oid(head, "HEAD SHA"),
        "origin_master_sha": _require_git_oid(
            origin, "origin/master SHA"
        ),
    }


def _validate_implementation_binding(
    repo_root: Path,
    implementations: Mapping[str, Any],
    name: str,
    expected_path: Path,
) -> dict[str, str]:
    raw = implementations.get(name)
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise RuntimeError(f"{name} implementation binding is malformed")
    path = (repo_root / str(raw["path"])).resolve()
    if path != expected_path.resolve():
        raise RuntimeError(f"{name} implementation path differs")
    observed = _file_binding(path)
    if observed["sha256"] != raw["sha256"]:
        raise RuntimeError(f"{name} implementation SHA256 differs")
    return observed


def _verified_bindings(
    repo_root: Path,
    config: Path,
    launcher_path: Path,
    wrapper_path: Path,
    controller_path: Path,
) -> dict[str, Any]:
    raw = _load_json(config, "canonical screening config")
    implementations = raw.get("implementations")
    if not isinstance(implementations, dict):
        raise RuntimeError("canonical implementations are malformed")
    return {
        "config": _file_binding(config),
        "launcher": _validate_implementation_binding(
            repo_root,
            implementations,
            "preflight_launcher",
            launcher_path,
        ),
        "wrapper": _validate_implementation_binding(
            repo_root,
            implementations,
            "preflight_wrapper",
            wrapper_path,
        ),
        "controller": _validate_implementation_binding(
            repo_root,
            implementations,
            "controller",
            controller_path,
        ),
        "verified_loader": _validate_implementation_binding(
            repo_root,
            implementations,
            "preflight_verified_loader",
            (
                repo_root
                / "src/safa/closeout/verified_preflight_module_loader.py"
            ),
        ),
        "preflight_launch_contract": (
            _validate_implementation_binding(
                repo_root,
                implementations,
                "preflight_launch_contract",
                (
                    repo_root
                    / "src/safa/closeout/preflight_launch_contract.py"
                ),
            )
        ),
    }


def _verified_implementations_from_receipt(
    receipt_path: Path,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, "launch receipt")
    sealed = validate_verified_implementations(
        receipt.get("verified_implementations"),
        "launch receipt verified implementations",
    )
    live = _reverify_verified_preflight_apis()
    if live != sealed:
        raise RuntimeError(
            "launch receipt verified implementations differ live"
        )
    return live


def _process_identity(pid: int) -> dict[str, int]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError(f"process stat is malformed for PID {pid}")
    fields = raw[closing + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError(f"process stat is incomplete for PID {pid}")
    return build_process_identity(
        pid=pid,
        ppid=int(fields[1]),
        pgid=int(fields[2]),
        sid=int(fields[3]),
        start_ticks=int(fields[19]),
    )


def _process_command(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    command = [
        item.decode("utf-8")
        for item in raw.split(b"\0")
        if item
    ]
    if not command:
        raise RuntimeError(f"process command is empty for PID {pid}")
    return command


def _process_command_bytes(pid: int) -> bytes:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    if not raw or not raw.endswith(b"\0"):
        raise RuntimeError(
            f"process command bytes are malformed for PID {pid}"
        )
    return raw


def _command_bytes(arguments: Sequence[str]) -> bytes:
    if not arguments or any(not isinstance(item, str) for item in arguments):
        raise RuntimeError("expected process arguments are invalid")
    return b"\0".join(os.fsencode(item) for item in arguments) + b"\0"


def _process_executable(pid: int) -> dict[str, Any]:
    return _opened_file_identity(
        Path(os.readlink(f"/proc/{pid}/exe")).resolve()
    )


def _wrapper_child_setup() -> None:
    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_pdeathsig = 1
    if libc.prctl(pr_set_pdeathsig, signal.SIGKILL) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def _tmux_pane(session: str) -> dict[str, Any] | None:
    result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            session,
            "-F",
            (
                "#{session_name}\t#{pane_id}\t#{pane_pid}\t"
                "#{pane_dead}\t#{pane_dead_status}"
            ),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    rows = [line.split("\t") for line in result.stdout.splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("launcher tmux pane identity is malformed")
    row = rows[0]
    return {
        "session": row[0],
        "pane": row[1],
        "pane_pid": int(row[2]),
        "pane_dead": row[3] == "1",
        "pane_dead_status": None if row[4] == "" else int(row[4]),
    }


def _tmux_owner_nonce(session: str) -> str:
    result = subprocess.run(
        ["tmux", "show-environment", "-t", session, TMUX_OWNER_ENV],
        capture_output=True,
        text=True,
    )
    prefix = f"{TMUX_OWNER_ENV}="
    if (
        result.returncode != 0
        or result.stderr.strip()
        or not result.stdout.strip().startswith(prefix)
    ):
        raise RuntimeError("launcher tmux owner nonce is unavailable")
    return _require_hex64(
        result.stdout.strip()[len(prefix) :],
        "tmux owner nonce",
    )


def _tmux_server_identity(pane: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "tmux",
            "display-message",
            "-p",
            "-t",
            pane,
            "#{pid}\t#{socket_path}",
        ],
        capture_output=True,
        text=True,
    )
    rows = [
        line.split("\t") for line in result.stdout.splitlines() if line
    ]
    if (
        result.returncode != 0
        or result.stderr.strip()
        or len(rows) != 1
        or len(rows[0]) != 2
    ):
        raise RuntimeError("launcher tmux server identity is malformed")
    socket_path = Path(rows[0][1])
    socket_stat = socket_path.stat()
    server_pid = int(rows[0][0])
    return build_tmux_server_identity(
        server_pid=server_pid,
        server_process=_process_identity(server_pid),
        socket_path=str(socket_path.resolve()),
        socket_device=int(socket_stat.st_dev),
        socket_inode=int(socket_stat.st_ino),
    )


def _tmux_owner_seal(
    session: str,
    expected_owner_nonce: str,
) -> dict[str, Any]:
    pane = _tmux_pane(session)
    if pane is None:
        raise RuntimeError("launcher tmux pane is absent")
    owner_nonce = _tmux_owner_nonce(session)
    if owner_nonce != expected_owner_nonce:
        raise RuntimeError("launcher tmux owner nonce differs")
    pane_process: dict[str, int] | None = None
    if not pane["pane_dead"]:
        try:
            pane_process = _process_identity(int(pane["pane_pid"]))
        except (FileNotFoundError, ProcessLookupError):
            deadline = time.monotonic() + 1.0
            while True:
                transitioned = _tmux_pane(session)
                if transitioned is None:
                    raise RuntimeError(
                        "launcher tmux pane disappeared during owner seal"
                    )
                if (
                    transitioned["session"] != pane["session"]
                    or transitioned["pane"] != pane["pane"]
                    or transitioned["pane_pid"] != pane["pane_pid"]
                ):
                    raise RuntimeError(
                        "launcher tmux pane changed during owner seal"
                    )
                if transitioned["pane_dead"]:
                    pane = transitioned
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "launcher tmux pane process vanished before tmux "
                        "recorded the exact gate exit"
                    )
                time.sleep(0.01)
    return {
        "session": session,
        "pane": pane["pane"],
        "pane_pid": pane["pane_pid"],
        "pane_dead": pane["pane_dead"],
        "pane_dead_status": pane["pane_dead_status"],
        "pane_process": pane_process,
        "owner_nonce": owner_nonce,
        "tmux_server": _tmux_server_identity(str(pane["pane"])),
    }


def _set_remain_on_exit(pane: str, enabled: bool) -> None:
    result = subprocess.run(
        [
            "tmux",
            "set-window-option",
            "-t",
            pane,
            "remain-on-exit",
            "on" if enabled else "off",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError(
            "launcher remain-on-exit update failed: "
            f"{result.stderr.strip()}"
        )


def _verify_remain_on_exit(pane: str, expected: str) -> None:
    result = subprocess.run(
        [
            "tmux",
            "show-window-options",
            "-v",
            "-t",
            pane,
            "remain-on-exit",
        ],
        capture_output=True,
        text=True,
    )
    if (
        result.returncode != 0
        or result.stderr.strip()
        or result.stdout.strip() != expected
    ):
        raise RuntimeError("launcher remain-on-exit verification failed")


def _kill_exact_session(
    session: str,
    owner_nonce: str,
    owner_seal: Mapping[str, Any] | None,
) -> bool:
    pane = _tmux_pane(session)
    if pane is None:
        return False
    if owner_seal is None:
        raise RuntimeError(
            "launcher refuses cleanup without an exact owner seal"
        )
    if _tmux_owner_nonce(session) != owner_nonce:
        raise RuntimeError("launcher refuses to kill a foreign tmux session")
    observed = _tmux_owner_seal(session, owner_nonce)
    expected = dict(owner_seal)
    if (
        observed["session"] != expected["session"]
        or observed["pane"] != expected["pane"]
        or observed["pane_pid"] != expected["pane_pid"]
        or observed["owner_nonce"] != expected["owner_nonce"]
        or observed["tmux_server"] != expected["tmux_server"]
        or (
            expected.get("pane_process") is not None
            and observed.get("pane_process")
            not in (expected["pane_process"], None)
        )
    ):
        raise RuntimeError(
            "launcher refuses to kill a replaced tmux owner"
        )
    result = subprocess.run(
        ["tmux", "kill-session", "-t", session],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if _tmux_pane(session) is None:
            return False
        raise RuntimeError(
            f"launcher tmux cleanup failed: {result.stderr.strip()}"
        )
    deadline = time.monotonic() + 1.0
    while _tmux_pane(session) is not None:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "launcher tmux cleanup left a session residual"
            )
        time.sleep(0.01)
    return True


def _terminate_exact_wrapper_child(
    wrapper_started_path: Path,
    owner_seal: Mapping[str, Any] | None,
) -> bool:
    if not wrapper_started_path.is_file():
        return False
    if owner_seal is None:
        raise RuntimeError(
            "launcher refuses child cleanup without an owner seal"
        )
    started = _load_json(
        wrapper_started_path, "wrapper-start cleanup evidence"
    )
    if (
        started.get("wrapper_started_sha256")
        != _canonical_digest(started, "wrapper_started_sha256")
        or started.get("pane_gate_process")
        != owner_seal.get("pane_process")
    ):
        raise RuntimeError(
            "launcher refuses cleanup from invalid wrapper-start evidence"
        )
    expected = dict(started["wrapper_process"])
    pid = int(expected["pid"])
    try:
        current = _process_identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        return False
    if current != expected:
        raise RuntimeError(
            "launcher refuses to signal a replaced wrapper child"
        )
    if (
        _process_command_bytes(pid)
        != _command_bytes(started["wrapper_arguments"])
        or _process_executable(pid) != started["wrapper_executable"]
    ):
        raise RuntimeError(
            "launcher refuses to signal a wrapper with a changed "
            "cmdline/executable seal"
        )
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 1.0
    while True:
        try:
            current = _process_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            return True
        if current != expected:
            raise RuntimeError(
                "wrapper PID changed during exact cleanup"
            )
        if time.monotonic() >= deadline:
            os.killpg(pid, signal.SIGKILL)
            break
        time.sleep(0.01)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            current = _process_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            return True
        if current != expected:
            raise RuntimeError(
                "wrapper PID changed after exact kill"
            )
        time.sleep(0.01)
    raise RuntimeError("exact wrapper child cleanup did not terminate")


def _cleanup_exact_attempt(
    *,
    session: str,
    owner_nonce: str,
    owner_seal: Mapping[str, Any] | None,
    wrapper_started_path: Path,
) -> None:
    session_failure: BaseException | None = None
    try:
        _kill_exact_session(session, owner_nonce, owner_seal)
    except BaseException as exc:
        session_failure = exc
    child_failure: BaseException | None = None
    try:
        _terminate_exact_wrapper_child(
            wrapper_started_path, owner_seal
        )
    except BaseException as exc:
        child_failure = exc
    if session_failure is not None:
        raise RuntimeError(
            f"exact tmux cleanup failed: {session_failure}"
        ) from session_failure
    if child_failure is None or not wrapper_started_path.is_file():
        return
    started = _load_json(
        wrapper_started_path, "wrapper-start cleanup verification"
    )
    expected = dict(started["wrapper_process"])
    try:
        current = _process_identity(int(expected["pid"]))
    except (FileNotFoundError, ProcessLookupError):
        return
    if current != expected:
        raise RuntimeError(
            "wrapper PID changed after cleanup failure"
        ) from child_failure
    raise RuntimeError(
        f"exact wrapper cleanup left a residual: {child_failure}"
    ) from child_failure


def _validate_wrapper_claim(
    path: Path,
    *,
    policy_sha256: str,
    config_binding: Mapping[str, str],
    receipt_binding: Mapping[str, str],
    receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    gate_ready_binding: Mapping[str, str],
    tmux_started_binding: Mapping[str, str],
    wrapper_started: Mapping[str, Any],
    wrapper_started_binding: Mapping[str, str],
    wrapper_arguments: Sequence[str],
    pane_log_identity: Mapping[str, Any],
    git: Mapping[str, str],
    supervisor_pid: int,
    gate_pid: int,
    pane_fault_consumer_chain: Mapping[str, Any],
) -> dict[str, Any]:
    value = _load_json(path, "preflight wrapper claim")
    validate_file_identity(
        receipt_identity, "launcher receipt identity"
    )
    validate_claim_v3(
        value,
        verified_implementations=verified_implementations,
        wrapper_started=wrapper_started,
        pane_fault_consumer_chain=pane_fault_consumer_chain,
        label="launcher wrapper claim v3",
    )
    child = dict(wrapper_started["wrapper_process"])
    if (
        value.get("policy_sha256") != policy_sha256
        or value.get("config") != dict(config_binding)
        or value.get("preflight_launch_receipt")
        != dict(receipt_binding)
        or value.get("preflight_launch_receipt_identity")
        != dict(receipt_identity)
        or value.get("pane_gate_ready")
        != dict(gate_ready_binding)
        or value.get("preflight_launch_tmux_started")
        != dict(tmux_started_binding)
        or value.get("preflight_wrapper_started")
        != dict(wrapper_started_binding)
        or value.get("pane_gate_process")
        != wrapper_started["pane_gate_process"]
        or value.get("wrapper_arguments") != list(wrapper_arguments)
        or value.get("wrapper_executable")
        != wrapper_started["wrapper_executable"]
        or value.get("pane_log") != dict(pane_log_identity)
        or value.get("git") != dict(git)
        or value.get("controller_session") != CONTROLLER_SESSION
        or value.get("wrapper_pid") != child["pid"]
        or value.get("controller_tmux", {}).get("pane_pid")
        != supervisor_pid
        or value.get("wrapper_process") != child
        or value.get("wrapper_launch_process") != child
        or child["ppid"] != gate_pid
        or child["pgid"] != child["pid"]
        or child["sid"] != child["pid"]
        or _process_identity(child["pid"]) != child
        or _process_command(child["pid"]) != list(wrapper_arguments)
        or _process_executable(child["pid"])
        != wrapper_started["wrapper_executable"]
        or value.get("wrapper_claim_sha256")
        != _canonical_digest(value, "wrapper_claim_sha256")
        or value.get("pane_fault_consumer_chain")
        != validate_pane_fault_consumer_chain(
            pane_fault_consumer_chain,
            label="launcher expected pane fault consumer chain",
        )
    ):
        raise RuntimeError("preflight wrapper claim differs at launch gate")
    return value


def _validate_wrapper_started(
    path: Path,
    *,
    receipt_binding: Mapping[str, str],
    receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    gate_ready_binding: Mapping[str, str],
    gate_process: Mapping[str, int],
    wrapper_arguments: Sequence[str],
) -> dict[str, Any]:
    value = _load_json(path, "preflight wrapper started")
    validate_file_identity(
        receipt_identity, "launcher receipt identity"
    )
    validate_wrapper_started(
        value,
        verified_implementations=verified_implementations,
        label="launcher wrapper started",
    )
    child = value.get("wrapper_process")
    if (
        value.get("launch_receipt") != dict(receipt_binding)
        or value.get("launch_receipt_identity")
        != dict(receipt_identity)
        or value.get("pane_gate_ready") != dict(gate_ready_binding)
        or value.get("pane_gate_process") != dict(gate_process)
        or value.get("wrapper_arguments") != list(wrapper_arguments)
        or not isinstance(child, dict)
        or child.get("ppid") != gate_process["pid"]
        or child.get("pgid") != child.get("pid")
        or child.get("sid") != child.get("pid")
        or _process_identity(child["pid"]) != child
        or _process_command_bytes(child["pid"])
        != _command_bytes(wrapper_arguments)
        or _process_executable(child["pid"])
        != value.get("wrapper_executable")
        or value.get("wrapper_started_sha256")
        != _canonical_digest(value, "wrapper_started_sha256")
    ):
        raise RuntimeError("preflight wrapper-start contract differs")
    return value


def _is_artifact_binding(raw: Any) -> bool:
    try:
        validate_artifact_binding(
            raw, "gate execution terminal artifact binding"
        )
    except PreflightLaunchContractError:
        return False
    return True


def _validate_gate_execution_terminal_value(
    value: Mapping[str, Any],
    *,
    receipt_binding: Mapping[str, str],
    receipt_identity: Mapping[str, Any],
    gate_ready_binding: Mapping[str, str],
    wrapper_arguments: Sequence[str],
    expected_ownership_chain: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract_type",
        "launch_receipt",
        "launch_receipt_identity",
        "pane_gate_ready",
        "wrapper_started",
        "launch_terminal",
        "launch_accepted",
        "launch_ownership_release",
        "returncode",
        "exit_kind",
        "exit_code",
        "signal_number",
        "failure",
        "publication_failures",
        "fault_channel",
        "fault_channel_snapshot",
        "fault_channel_validation_failure",
        "fault_channel_close_failure",
        "wrapper_outcome",
        "wrapper_arguments",
        "completed_at",
        "gate_execution_terminal_sha256",
    }
    ownership_values = tuple(
        value.get(field)
        for field in (
            "launch_accepted",
            "launch_terminal",
            "launch_ownership_release",
        )
    )
    ownership_absent = all(
        item is None for item in ownership_values
    )
    ownership_bound = all(
        _is_artifact_binding(item) for item in ownership_values
    )
    returncode = value.get("returncode")
    exit_kind = value.get("exit_kind")
    if exit_kind != "exec_error":
        expected_keys.add("supervisor_signals")
    exit_code = value.get("exit_code")
    signal_number = value.get("signal_number")
    classified = (
        exit_kind == "exec_error"
        and returncode is None
        and exit_code is None
        and signal_number is None
        and isinstance(value.get("failure"), dict)
    ) or (
        exit_kind == "exit"
        and type(returncode) is int
        and returncode >= 0
        and exit_code == returncode
        and signal_number is None
        and value.get("failure") is None
    ) or (
        exit_kind == "signal"
        and type(returncode) is int
        and returncode < 0
        and exit_code is None
        and signal_number == -returncode
        and value.get("failure") is None
    )
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_canonical_preflight_gate_execution_terminal_v1"
        or value.get("launch_receipt") != dict(receipt_binding)
        or value.get("launch_receipt_identity")
        != dict(receipt_identity)
        or value.get("pane_gate_ready") != dict(gate_ready_binding)
        or value.get("wrapper_arguments") != list(wrapper_arguments)
        or not isinstance(value.get("publication_failures"), list)
        or not (ownership_absent or ownership_bound)
        or {
            "launch_accepted": value.get("launch_accepted"),
            "launch_terminal": value.get("launch_terminal"),
            "launch_ownership_release": value.get(
                "launch_ownership_release"
            ),
        }
        != dict(expected_ownership_chain)
        or not classified
        or value.get("gate_execution_terminal_sha256")
        != _canonical_digest(
            value, "gate_execution_terminal_sha256"
        )
    ):
        raise RuntimeError("gate execution terminal differs")
    return dict(value)


def _validate_gate_execution_terminal(
    path: Path,
    *,
    receipt_binding: Mapping[str, str],
    receipt_identity: Mapping[str, Any],
    gate_ready_binding: Mapping[str, str],
    wrapper_arguments: Sequence[str],
) -> dict[str, Any]:
    value, _artifact, _identity = _sealed_finalization_json(
        path,
        digest_field="gate_execution_terminal_sha256",
        label="gate execution terminal",
    )
    try:
        _ownership_state, expected_ownership_chain = (
            _validate_gate_execution_ownership_chain(
                receipt_path=Path(str(receipt_binding["path"])),
                gate_execution=value,
            )
        )
    except RuntimeError as exc:
        raise RuntimeError("gate execution terminal differs") from exc
    return _validate_gate_execution_terminal_value(
        value,
        receipt_binding=receipt_binding,
        receipt_identity=receipt_identity,
        gate_ready_binding=gate_ready_binding,
        wrapper_arguments=wrapper_arguments,
        expected_ownership_chain=expected_ownership_chain,
    )


def _publish_gate_execution_terminal(
    path: Path, value: dict[str, Any]
) -> dict[str, Any]:
    value["gate_execution_terminal_sha256"] = _canonical_digest(
        value, "gate_execution_terminal_sha256"
    )
    try:
        _write_exclusive(path, value)
    except LauncherExclusivePublishError:
        raise
    except BaseException as exc:
        raise LauncherTerminalPublishError(path, exc) from exc
    return value


def _publish_terminal(
    terminal_path: Path,
    *,
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    status: str,
    failure_type: str,
    message: str,
    client: Mapping[str, Any] | None,
    pane: Mapping[str, Any] | None,
    tmux_started_path: Path | None,
    log_path: Path,
    session_residual: bool | None,
    started_at: str,
    gate_execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": LAUNCH_TERMINAL_CONTRACT_TYPE,
        "launch_receipt": _json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        "launch_receipt_identity": dict(receipt_identity),
        "status": status,
        "failure": {
            "type": failure_type,
            "message": message,
            "secondary_failures": [],
        },
        "tmux_client": None if client is None else dict(client),
        "pane": None if pane is None else dict(pane),
        "tmux_started": (
            None
            if tmux_started_path is None
            or not tmux_started_path.is_file()
            else _json_binding(
                tmux_started_path, "launch_tmux_started_sha256"
            )
        ),
        "gate_execution": (
            None
            if gate_execution is None
            else dict(gate_execution)
        ),
        "pane_log": _optional_file_binding(log_path),
        "session_residual": session_residual,
        "started_at": started_at,
        "completed_at": _utc_now(),
    }
    value["launch_terminal_sha256"] = _canonical_digest(
        value, "launch_terminal_sha256"
    )
    try:
        _write_exclusive(terminal_path, value)
    except LauncherExclusivePublishError:
        raise
    except BaseException as exc:
        raise LauncherTerminalPublishError(
            terminal_path, exc
        ) from exc
    return value


def _publish_setup_terminal(
    *,
    campaign_root: Path,
    attempt_id: str,
    policy_sha256: str,
    started_registry_path: Path,
    attempt_root: Path,
    stage: str,
    failure: BaseException,
    started_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_setup_terminal_v1"
        ),
        "attempt_id": attempt_id,
        "policy_sha256": policy_sha256,
        "started_registry": _json_binding(
            started_registry_path,
            "launch_started_registry_sha256",
        ),
        "stage": stage,
        "failure": {
            "type": type(failure).__name__,
            "message": str(failure),
        },
        "tmux_execution_count": 0,
        "scientific_execution_started": False,
        "started_at": started_at,
        "completed_at": _utc_now(),
    }
    value["launch_setup_terminal_sha256"] = _canonical_digest(
        value, "launch_setup_terminal_sha256"
    )
    path = (
        campaign_root.resolve()
        / "preflight_launch_attempts"
        / "setup_terminals"
        / f"{attempt_id}.json"
    )
    _write_exclusive(path, value)
    return value


def _publish_accepted(
    accepted_path: Path,
    *,
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    claim_path: Path,
    tmux_started_path: Path,
    pane: Mapping[str, Any],
    log_path: Path,
    started_at: str,
    pane_fault_consumer_chain: Mapping[str, Any],
) -> dict[str, Any]:
    value = build_launch_accepted(
        attempt_id=_load_json(
            receipt_path, "launch receipt"
        )["attempt_id"],
        launch_receipt=_json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        launch_receipt_identity=receipt_identity,
        verified_implementations=(
            _verified_implementations_from_receipt(receipt_path)
        ),
        wrapper_claim=_json_binding(
            claim_path, "wrapper_claim_sha256"
        ),
        tmux_started=_json_binding(
            tmux_started_path, "launch_tmux_started_sha256"
        ),
        pane=pane,
        pane_log_path=str(log_path.resolve()),
        started_at=started_at,
        accepted_at=_utc_now(),
        pane_fault_consumer_chain=pane_fault_consumer_chain,
    )
    _write_exclusive(accepted_path, value)
    return value


def _publish_ownership_terminal(
    terminal_path: Path,
    *,
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    accepted_path: Path,
    tmux_started_path: Path,
    claim_path: Path,
    pane: Mapping[str, Any],
    log_path: Path,
    started_at: str,
    pane_fault_consumer_chain: Mapping[str, Any],
) -> dict[str, Any]:
    value = build_ownership_terminal(
        launch_receipt=_json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        launch_receipt_identity=receipt_identity,
        verified_implementations=(
            _verified_implementations_from_receipt(receipt_path)
        ),
        launch_accepted=_json_binding(
            accepted_path, "launch_accepted_sha256"
        ),
        wrapper_claim=_json_binding(
            claim_path, "wrapper_claim_sha256"
        ),
        tmux_started=_json_binding(
            tmux_started_path, "launch_tmux_started_sha256"
        ),
        pane=pane,
        pane_log=_file_identity(log_path),
        started_at=started_at,
        completed_at=_utc_now(),
        pane_fault_consumer_chain=pane_fault_consumer_chain,
    )
    _write_exclusive(terminal_path, value)
    return value


def _publish_ownership_release(
    path: Path,
    *,
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    accepted_path: Path,
    terminal_path: Path,
    claim_path: Path,
    pane_fault_consumer_chain: Mapping[str, Any],
) -> dict[str, Any]:
    value = build_ownership_release(
        launch_receipt=_json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        launch_receipt_identity=receipt_identity,
        verified_implementations=(
            _verified_implementations_from_receipt(receipt_path)
        ),
        launch_accepted=_json_binding(
            accepted_path, "launch_accepted_sha256"
        ),
        launch_terminal=_json_binding(
            terminal_path, "launch_terminal_sha256"
        ),
        wrapper_claim=_json_binding(
            claim_path, "wrapper_claim_sha256"
        ),
        released_at=_utc_now(),
        pane_fault_consumer_chain=pane_fault_consumer_chain,
    )
    _write_exclusive(path, value)
    return value


def _terminate_spawned_child(child: subprocess.Popen[Any]) -> None:
    if child.poll() is not None:
        child.wait()
        return
    process = _process_identity(child.pid)
    if (
        process["ppid"] != os.getpid()
        or process["pgid"] != child.pid
        or process["sid"] != child.pid
    ):
        raise RuntimeError(
            "gate refuses to terminate an unbound wrapper child"
        )
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    current = _process_identity(child.pid)
    if current != process:
        raise RuntimeError(
            "wrapper child identity changed before SIGKILL"
        )
    os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=1.0)


def _supervise_wrapper_child(
    *,
    child: subprocess.Popen[Any],
    wrapper_arguments: Sequence[str],
    receipt_path: Path,
    gate_ready_path: Path,
    wrapper_started_path: Path,
    execution_terminal_path: Path,
    launch_terminal_path: Path,
    accepted_path: Path,
    ownership_release_path: Path,
    supervisor_signals: list[int],
    fault_descriptor: int,
    fault_channel: Mapping[str, Any],
    fault_attempt_id: str,
    fault_owner_nonce: str,
    fault_receipt_sha256: str,
    fault_publisher: Mapping[str, str],
    wrapper_exit_path: Path,
    policy_sha256: str,
) -> int:
    wrapper_started: dict[str, Any] | None = None
    process_deadline = time.monotonic() + 2.0
    while child.poll() is None:
        try:
            process = _process_identity(child.pid)
            command = _process_command(child.pid)
            executable = _process_executable(child.pid)
        except (OSError, RuntimeError, UnicodeError):
            if time.monotonic() >= process_deadline:
                break
            time.sleep(0.005)
            continue
        if (
            command == list(wrapper_arguments)
            and process["ppid"] == os.getpid()
            and process["pgid"] == child.pid
            and process["sid"] == child.pid
        ):
            wrapper_started = build_wrapper_started(
                launch_receipt=_json_binding(
                    receipt_path, "launch_receipt_sha256"
                ),
                launch_receipt_identity=_opened_file_identity(
                    receipt_path
                ),
                verified_implementations=(
                    _verified_implementations_from_receipt(
                        receipt_path
                    )
                ),
                pane_gate_ready=_json_binding(
                    gate_ready_path, "pane_gate_ready_sha256"
                ),
                pane_gate_process=_process_identity(os.getpid()),
                wrapper_arguments=list(wrapper_arguments),
                wrapper_process=process,
                wrapper_executable=executable,
                started_at=_utc_now(),
                gate_ready=_load_json(
                    gate_ready_path, "pane gate ready"
                ),
            )
            _write_exclusive(wrapper_started_path, wrapper_started)
            break
        if time.monotonic() >= process_deadline:
            break
        time.sleep(0.005)
    for signum in supervisor_signals:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
    while True:
        try:
            returncode = child.wait()
            break
        except InterruptedError:
            continue
    fault_snapshot: dict[str, Any] | None = None
    fault_validation_failure: dict[str, str] | None = None
    try:
        fault_snapshot = _read_fault_channel(
            fault_descriptor,
            fault_channel,
            attempt_id=fault_attempt_id,
            owner_nonce=fault_owner_nonce,
            launch_receipt_sha256=fault_receipt_sha256,
            publisher=fault_publisher,
        )
    except BaseException as exc:
        fault_validation_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    fault_close_failure = _close_fault_channel(fault_descriptor)
    outcome = _evaluate_gate_outcome(
        returncode=returncode,
        exec_failure=None,
        fault_snapshot=fault_snapshot,
        fault_validation_failure=fault_validation_failure,
        fault_close_failure=fault_close_failure,
        wrapper_exit_reader=lambda: _read_exact_wrapper_exit(
            wrapper_exit_path,
            policy_sha256=policy_sha256,
        ),
    )
    exit_kind = "exit" if returncode >= 0 else "signal"
    terminal = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_gate_execution_terminal_v1"
        ),
        "launch_receipt": _json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        "launch_receipt_identity": _opened_file_identity(receipt_path),
        "pane_gate_ready": _json_binding(
            gate_ready_path, "pane_gate_ready_sha256"
        ),
        "wrapper_started": (
            None
            if wrapper_started is None
            else _json_binding(
                wrapper_started_path, "wrapper_started_sha256"
            )
        ),
        "launch_terminal": (
            _json_binding(
                launch_terminal_path, "launch_terminal_sha256"
            )
            if launch_terminal_path.is_file()
            else None
        ),
        "launch_accepted": (
            _json_binding(
                accepted_path, "launch_accepted_sha256"
            )
            if accepted_path.is_file()
            else None
        ),
        "launch_ownership_release": (
            _json_binding(
                ownership_release_path,
                "launch_ownership_release_sha256",
            )
            if ownership_release_path.is_file()
            else None
        ),
        "returncode": returncode,
        "exit_kind": exit_kind,
        "exit_code": returncode if returncode >= 0 else None,
        "signal_number": -returncode if returncode < 0 else None,
        "supervisor_signals": supervisor_signals,
        "failure": None,
        "publication_failures": [],
        "fault_channel": dict(fault_channel),
        "fault_channel_snapshot": fault_snapshot,
        "fault_channel_validation_failure": (
            fault_validation_failure
        ),
        "fault_channel_close_failure": fault_close_failure,
        "wrapper_outcome": outcome,
        "wrapper_arguments": list(wrapper_arguments),
        "completed_at": _utc_now(),
    }
    _publish_gate_execution_terminal(execution_terminal_path, terminal)
    return GATE_ADJUDICATED_EXIT


def _publish_pre_wrapper_gate_failure(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    gate_ready_path: Path,
    execution_terminal_path: Path,
    wrapper_arguments: Sequence[str],
    failure: BaseException,
) -> int:
    fault_channel = receipt.get("fault_channel")
    if not isinstance(fault_channel, Mapping):
        raise RuntimeError(
            "pre-wrapper gate fault channel binding differs"
        )
    descriptor = _open_presealed_fault_channel(
        receipt_path.parent, fault_channel
    )
    try:
        fault_snapshot = _read_fault_channel(
            descriptor,
            fault_channel,
            attempt_id=str(receipt["attempt_id"]),
            owner_nonce=str(receipt["controller_owner_nonce"]),
            launch_receipt_sha256=str(
                receipt["launch_receipt_sha256"]
            ),
            publisher=dict(receipt["bindings"]["wrapper"]),
        )
    finally:
        os.close(descriptor)
    typed_failure = {
        "type": type(failure).__name__,
        "message": str(failure),
    }
    outcome = _evaluate_gate_outcome(
        returncode=None,
        exec_failure=typed_failure,
        fault_snapshot=fault_snapshot,
        fault_validation_failure=None,
        fault_close_failure=None,
        wrapper_exit_reader=lambda: None,
    )
    terminal = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_gate_execution_terminal_v1"
        ),
        "launch_receipt": _json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        "launch_receipt_identity": _opened_file_identity(receipt_path),
        "pane_gate_ready": _json_binding(
            gate_ready_path, "pane_gate_ready_sha256"
        ),
        "wrapper_started": None,
        "launch_terminal": None,
        "launch_accepted": None,
        "launch_ownership_release": None,
        "returncode": None,
        "exit_kind": "exec_error",
        "exit_code": None,
        "signal_number": None,
        "failure": typed_failure,
        "publication_failures": [],
        "fault_channel": dict(fault_channel),
        "fault_channel_snapshot": fault_snapshot,
        "fault_channel_validation_failure": None,
        "fault_channel_close_failure": None,
        "wrapper_outcome": outcome,
        "wrapper_arguments": list(wrapper_arguments),
        "completed_at": _utc_now(),
    }
    _publish_gate_execution_terminal(
        execution_terminal_path, terminal
    )
    return GATE_ADJUDICATED_EXIT


def _pane_gate_owned(
    *,
    attempt_root: Path,
    release_path: Path,
    log_path: Path,
    wrapper_arguments: Sequence[str],
) -> int:
    receipt_path = attempt_root / "launch_receipt.json"
    gate_ready_path = attempt_root / "pane_gate_ready.json"
    wrapper_started_path = attempt_root / "wrapper_started.json"
    execution_terminal_path = (
        attempt_root / "gate_execution_terminal.json"
    )
    launch_terminal_path = attempt_root / "launch_terminal.json"
    accepted_path = attempt_root / "launch_accepted.json"
    ownership_release_path = (
        attempt_root / "launch_ownership_release.json"
    )
    receipt = _load_json(receipt_path, "launch receipt")
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_APPEND,
        0o644,
    )
    descriptor_stat = os.fstat(descriptor)
    expected_log = receipt.get("pane_log")
    if (
        not isinstance(expected_log, dict)
        or expected_log.get("path") != str(log_path.resolve())
        or expected_log.get("device") != int(descriptor_stat.st_dev)
        or expected_log.get("inode") != int(descriptor_stat.st_ino)
        or expected_log.get("mode") != int(descriptor_stat.st_mode)
    ):
        os.close(descriptor)
        raise RuntimeError("pane log identity differs before exec")
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    os.close(descriptor)
    ready = build_gate_ready(
        launch_receipt=_json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        launch_receipt_identity=_opened_file_identity(receipt_path),
        verified_implementations=(
            _verified_implementations_from_receipt(receipt_path)
        ),
        process=_process_identity(os.getpid()),
        wrapper_arguments=list(wrapper_arguments),
        ready_at=_utc_now(),
    )
    receipt_identity = dict(ready["launch_receipt_identity"])
    _write_exclusive(gate_ready_path, ready)
    deadline = time.monotonic() + DEFAULT_STARTUP_TIMEOUT_SECONDS
    while not release_path.is_file():
        if time.monotonic() >= deadline:
            failure = RuntimeError("pane gate release timed out")
            print(str(failure), flush=True)
            return _publish_pre_wrapper_gate_failure(
                receipt=receipt,
                receipt_path=receipt_path,
                gate_ready_path=gate_ready_path,
                execution_terminal_path=execution_terminal_path,
                wrapper_arguments=wrapper_arguments,
                failure=failure,
            )
        time.sleep(0.02)
    release = _load_json(release_path, "pane gate release")
    started_binding = release.get("tmux_started")
    started_valid = False
    if isinstance(started_binding, dict):
        try:
            started_valid = _json_binding(
                Path(str(started_binding["path"])),
                "launch_tmux_started_sha256",
            ) == started_binding
        except (KeyError, OSError, RuntimeError, ValueError):
            started_valid = False
    if (
        release.get("launch_receipt")
        != _json_binding(receipt_path, "launch_receipt_sha256")
        or release.get("launch_receipt_identity")
        != receipt_identity
        or _opened_file_identity(receipt_path) != receipt_identity
        or release.get("pane_gate_ready")
        != _json_binding(gate_ready_path, "pane_gate_ready_sha256")
        or not started_valid
        or release.get("wrapper_arguments") != list(wrapper_arguments)
        or validate_pane_fault_consumer_chain(
            release.get("pane_fault_consumer_chain"),
            registration=receipt.get("pane_fault_consumer"),
            label="pane gate release consumer chain",
        )
        != release.get("pane_fault_consumer_chain")
        or release.get("pane_gate_release_sha256")
        != _canonical_digest(release, "pane_gate_release_sha256")
    ):
        failure = RuntimeError(
            "pane gate release contract differs"
        )
        print(str(failure), flush=True)
        return _publish_pre_wrapper_gate_failure(
            receipt=receipt,
            receipt_path=receipt_path,
            gate_ready_path=gate_ready_path,
            execution_terminal_path=execution_terminal_path,
            wrapper_arguments=wrapper_arguments,
            failure=failure,
        )
    fault_channel = receipt.get("fault_channel")
    if not isinstance(fault_channel, dict):
        print("pane gate fault channel binding differs", flush=True)
        return 125
    fault_descriptor = _open_presealed_fault_channel(
        attempt_root, fault_channel
    )
    environment = dict(os.environ)
    environment[FAULT_CHANNEL_FD_ENV] = str(fault_descriptor)
    child: subprocess.Popen[Any] | None = None
    supervisor_signals: list[int] = []

    def forward_signal(
        signum: int, _frame: Any
    ) -> None:
        supervisor_signals.append(signum)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    old_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        child = subprocess.Popen(
            list(wrapper_arguments),
            shell=False,
            preexec_fn=_wrapper_child_setup,
            env=environment,
            pass_fds=(fault_descriptor,),
        )
    except BaseException as exc:
        traceback.print_exc()
        fault_snapshot: dict[str, Any] | None = None
        fault_validation_failure: dict[str, str] | None = None
        try:
            fault_snapshot = _read_fault_channel(
                fault_descriptor,
                fault_channel,
                attempt_id=str(receipt["attempt_id"]),
                owner_nonce=str(
                    receipt["controller_owner_nonce"]
                ),
                launch_receipt_sha256=str(
                    receipt["launch_receipt_sha256"]
                ),
                publisher=dict(receipt["bindings"]["wrapper"]),
            )
        except BaseException as fault_exc:
            fault_validation_failure = {
                "type": type(fault_exc).__name__,
                "message": str(fault_exc),
            }
        fault_close_failure = _close_fault_channel(
            fault_descriptor
        )
        outcome = _evaluate_gate_outcome(
            returncode=None,
            exec_failure={
                "type": type(exc).__name__,
                "message": str(exc),
            },
            fault_snapshot=fault_snapshot,
            fault_validation_failure=fault_validation_failure,
            fault_close_failure=fault_close_failure,
            wrapper_exit_reader=lambda: None,
        )
        terminal = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_gate_execution_terminal_v1"
            ),
            "launch_receipt": _json_binding(
                receipt_path, "launch_receipt_sha256"
            ),
            "launch_receipt_identity": _opened_file_identity(receipt_path),
            "pane_gate_ready": _json_binding(
                gate_ready_path, "pane_gate_ready_sha256"
            ),
            "wrapper_started": None,
            "launch_terminal": None,
            "launch_accepted": None,
            "launch_ownership_release": None,
            "returncode": None,
            "exit_kind": "exec_error",
            "exit_code": None,
            "signal_number": None,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "publication_failures": [],
            "fault_channel": dict(fault_channel),
            "fault_channel_snapshot": fault_snapshot,
            "fault_channel_validation_failure": (
                fault_validation_failure
            ),
            "fault_channel_close_failure": fault_close_failure,
            "wrapper_outcome": outcome,
            "wrapper_arguments": list(wrapper_arguments),
            "completed_at": _utc_now(),
        }
        _publish_gate_execution_terminal(
            execution_terminal_path, terminal
        )
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        return GATE_ADJUDICATED_EXIT
    assert child is not None
    try:
        return _supervise_wrapper_child(
            child=child,
            wrapper_arguments=wrapper_arguments,
            receipt_path=receipt_path,
            gate_ready_path=gate_ready_path,
            wrapper_started_path=wrapper_started_path,
            execution_terminal_path=execution_terminal_path,
            launch_terminal_path=launch_terminal_path,
            accepted_path=accepted_path,
            ownership_release_path=ownership_release_path,
            supervisor_signals=supervisor_signals,
            fault_descriptor=fault_descriptor,
            fault_channel=fault_channel,
            fault_attempt_id=str(receipt["attempt_id"]),
            fault_owner_nonce=str(
                receipt["controller_owner_nonce"]
            ),
            fault_receipt_sha256=str(
                receipt["launch_receipt_sha256"]
            ),
            fault_publisher=dict(
                receipt["bindings"]["wrapper"]
            ),
            wrapper_exit_path=(
                Path(str(receipt["wrapper_claim_path"])).parent
                / "wrapper_exit.json"
            ),
            policy_sha256=str(receipt["policy_sha256"]),
        )
    except BaseException:
        _terminate_spawned_child(child)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def _pane_gate(
    *,
    attempt_root: Path,
    release_path: Path,
    log_path: Path,
    wrapper_arguments: Sequence[str],
) -> int:
    receipt = _load_json(
        attempt_root / "launch_receipt.json",
        "launch receipt for pane fault channel",
    )
    binding = receipt.get("pane_gate_fault_channel")
    publisher = receipt.get("pane_gate_fault_publisher")
    if (
        not isinstance(binding, dict)
        or not isinstance(publisher, dict)
    ):
        raise RuntimeError(
            "pane gate fault channel receipt binding differs"
        )
    descriptor = _open_presealed_fault_channel(
        attempt_root,
        binding,
        name="pane_gate_fault.channel",
    )
    try:
        try:
            return _pane_gate_owned(
                attempt_root=attempt_root,
                release_path=release_path,
                log_path=log_path,
                wrapper_arguments=wrapper_arguments,
            )
        except LauncherExclusivePublishError as exc:
            _write_launcher_fault_channel_record(
                descriptor,
                binding,
                attempt_id=str(receipt["attempt_id"]),
                owner_nonce=str(
                    receipt["controller_owner_nonce"]
                ),
                launch_receipt_sha256=str(
                    receipt["launch_receipt_sha256"]
                ),
                publisher=publisher,
                failure=exc,
            )
            return 123
    finally:
        os.close(descriptor)


def _sealed_lifecycle_artifact(
    path: Path, *, digest_field: str, kind: str
) -> dict[str, Any]:
    return build_sealed_lifecycle_artifact(
        kind=kind,
        binding=_json_binding(path, digest_field),
        file_identity=_opened_file_identity(path),
    )


def _read_sealed_json_artifact_at(
    directory_descriptor: int,
    attempt_root: Path,
    *,
    name: str,
    digest_field: str,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES
        ):
            raise RuntimeError(
                f"sealed {kind} identity or size differs"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            len(raw) != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mode != before.st_mode
            or after.st_uid != before.st_uid
            or after.st_nlink != before.st_nlink
            or after.st_size != before.st_size
            or named.st_dev != before.st_dev
            or named.st_ino != before.st_ino
            or named.st_mode != before.st_mode
            or named.st_uid != before.st_uid
            or named.st_nlink != before.st_nlink
            or named.st_size != before.st_size
        ):
            raise RuntimeError(
                f"sealed {kind} changed during read"
            )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"sealed {kind} JSON is invalid"
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get(digest_field)
            != _canonical_digest(value, digest_field)
        ):
            raise RuntimeError(
                f"sealed {kind} canonical digest differs"
            )
        path = attempt_root / name
        artifact = build_sealed_lifecycle_artifact(
            kind=kind,
            binding=build_artifact_binding(
                path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                canonical_sha256=str(value[digest_field]),
            ),
            file_identity=build_file_identity(
                path=str(path),
                device=int(before.st_dev),
                inode=int(before.st_ino),
                mode=int(before.st_mode),
                size=int(before.st_size),
            ),
        )
        return artifact, value
    finally:
        os.close(descriptor)


def _expected_consumer_worker_arguments_from_receipt(
    receipt: Mapping[str, Any],
) -> list[str]:
    registration = receipt["pane_fault_consumer"]
    return [
        sys.executable,
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        PANE_FAULT_CONSUMER_MODE,
        "--attempt-path",
        str(registration["artifacts"]["attempt"]),
        "--config",
        str(receipt["bindings"]["config"]["path"]),
    ]


def _read_formal_gate_lifecycle_status(
    *,
    attempt_root: Path,
    pane: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = attempt_root / "launch_receipt.json"
    receipt = _load_json(receipt_path, "formal gate launch receipt")
    supervisor_ready_path = Path(
        str(receipt["gate_lifecycle_wait_supervisor_ready_path"])
    )
    supervisor_ready = _validate_gate_wait_supervisor_ready(
        supervisor_ready_path,
        receipt_path=receipt_path,
        receipt_identity=_opened_file_identity(receipt_path),
        receipt=receipt,
    )
    validate_launch_receipt_schema(
        receipt,
        expected_gate_worker_arguments=supervisor_ready[
            "gate_worker_arguments"
        ],
        expected_consumer_worker_arguments=(
            _expected_consumer_worker_arguments_from_receipt(receipt)
        ),
        label="formal gate launch receipt v3",
    )
    supervisor = supervisor_ready["supervisor_process"]
    if (
        pane.get("pane_pid") != supervisor["pid"]
        or pane.get("pane_dead") is not True
    ):
        raise RuntimeError(
            "formal gate lifecycle supervisor is not retired"
        )
    channel_descriptor, directory_descriptor = (
        _open_lifecycle_wait_channel_reader(
            attempt_root,
            receipt["gate_lifecycle_wait_channel"],
            name=Path(
                str(receipt["gate_lifecycle_wait_status_path"])
            ).name,
        )
    )
    try:
        source_artifact, sealed_receipt = (
            _read_sealed_json_artifact_at(
                directory_descriptor,
                attempt_root,
                name="launch_receipt.json",
                digest_field="launch_receipt_sha256",
                kind="launch_receipt",
            )
        )
        if sealed_receipt != receipt:
            raise RuntimeError(
                "formal gate launch receipt changed"
            )
        supervisor_artifact, sealed_supervisor = (
            _read_sealed_json_artifact_at(
                directory_descriptor,
                attempt_root,
                name="gate_wait_supervisor_ready.json",
                digest_field=(
                    "gate_wait_supervisor_ready_sha256"
                ),
                kind="gate_wait_supervisor_ready",
            )
        )
        del supervisor_artifact
        if sealed_supervisor != supervisor_ready:
            raise RuntimeError(
                "formal gate supervisor ready changed"
            )
        worker_started, gate_ready = (
            _read_sealed_json_artifact_at(
                directory_descriptor,
                attempt_root,
                name="pane_gate_ready.json",
                digest_field="pane_gate_ready_sha256",
                kind="gate_worker_started",
            )
        )
        validate_gate_ready(
            gate_ready,
            verified_implementations=receipt[
                "verified_implementations"
            ],
            label="formal gate worker started",
        )
        if (
            gate_ready.get("process")
            != supervisor_ready["gate_worker_process"]
        ):
            raise RuntimeError(
                "formal gate worker process differs"
            )
        terminal_name = Path(
            str(receipt["gate_execution_terminal_path"])
        ).name
        try:
            os.stat(
                terminal_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            terminal_ref = None
            terminal = None
        else:
            terminal_ref, terminal = (
                _read_sealed_json_artifact_at(
                    directory_descriptor,
                    attempt_root,
                    name=terminal_name,
                    digest_field=(
                        "gate_execution_terminal_sha256"
                    ),
                    kind="gate_execution_terminal",
                )
            )
            _ownership_state, expected_ownership = (
                _validate_gate_execution_ownership_chain(
                    receipt_path=receipt_path,
                    gate_execution=terminal,
                )
            )
            _validate_gate_execution_terminal_value(
                terminal,
                receipt_binding=source_artifact["binding"],
                receipt_identity=source_artifact[
                    "file_identity"
                ],
                gate_ready_binding=worker_started["binding"],
                wrapper_arguments=receipt["wrapper_arguments"],
                expected_ownership_chain=expected_ownership,
            )
        expected_bindings = {
            "policy_sha256": receipt["policy_sha256"],
            "attempt_id": receipt["attempt_id"],
            "source_artifact": source_artifact,
            "publisher": receipt[
                "gate_lifecycle_wait_publisher"
            ],
            "supervisor_owner_seal": supervisor_ready[
                "owner_seal"
            ],
            "supervisor_process": supervisor,
            "supervisor_executable": supervisor_ready[
                "supervisor_executable"
            ],
            "supervisor_command": supervisor_ready[
                "supervisor_arguments"
            ],
            "worker_started": worker_started,
            "child_process": supervisor_ready[
                "gate_worker_process"
            ],
            "child_executable": supervisor_ready[
                "gate_worker_executable"
            ],
            "child_command": supervisor_ready[
                "gate_worker_arguments"
            ],
            "terminal": terminal_ref,
        }
        snapshot = _read_lifecycle_wait_status(
            channel_descriptor,
            directory_descriptor,
            receipt["gate_lifecycle_wait_channel"],
            role="gate",
            expected_bindings=expected_bindings,
        )
        record = snapshot.get("record")
        if (
            snapshot.get("state") != "valid_wait_status"
            or not isinstance(record, Mapping)
            or record.get("exit_kind") != "exit"
            or record.get("returncode") != GATE_ADJUDICATED_EXIT
            or terminal is None
        ):
            raise RuntimeError(
                "formal gate lifecycle outcome is not controlled"
            )
        adjudication = _adjudicate_gate_execution_outcome(
            terminal,
            wrapper_exit_path=(
                Path(str(receipt["wrapper_claim_path"])).parent
                / "wrapper_exit.json"
            ),
            policy_sha256=str(receipt["policy_sha256"]),
        )
        ownership_chain_state, ownership_chain = (
            _validate_gate_execution_ownership_chain(
                receipt_path=receipt_path,
                gate_execution=terminal,
            )
        )
        if (
            adjudication["adjudicated_outcome"] == "completed"
            and ownership_chain_state != "bound"
        ):
            raise RuntimeError(
                "completed gate execution ownership chain is absent"
            )
        return {
            "snapshot": snapshot,
            "gate_execution": terminal,
            "adjudication": adjudication,
            "ownership_chain_state": ownership_chain_state,
            "ownership_chain": ownership_chain,
        }
    finally:
        os.close(channel_descriptor)
        os.close(directory_descriptor)


def _read_formal_consumer_lifecycle_status(
    *,
    attempt_path: Path,
    pane: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_root = attempt_path.parent
    attempt = _load_json(
        attempt_path, "formal consumer attempt"
    )
    supervisor_ready_path = Path(
        str(
            attempt[
                "consumer_lifecycle_wait_supervisor_ready_path"
            ]
        )
    )
    supervisor_ready = _validate_consumer_wait_supervisor_ready(
        supervisor_ready_path,
        attempt_path=attempt_path,
        attempt_identity=_opened_file_identity(attempt_path),
        attempt=attempt,
    )
    supervisor = supervisor_ready["supervisor_process"]
    if (
        pane.get("pane_pid") != supervisor["pid"]
        or pane.get("pane_dead") is not True
    ):
        raise RuntimeError(
            "formal consumer lifecycle supervisor is not retired"
        )
    channel_descriptor, directory_descriptor = (
        _open_lifecycle_wait_channel_reader(
            attempt_root,
            attempt["consumer_lifecycle_wait_channel"],
            name=Path(
                str(attempt["consumer_lifecycle_wait_status_path"])
            ).name,
        )
    )
    try:
        source_artifact, sealed_attempt = (
            _read_sealed_json_artifact_at(
                directory_descriptor,
                attempt_root,
                name=attempt_path.name,
                digest_field="consumer_attempt_sha256",
                kind="consumer_attempt",
            )
        )
        if sealed_attempt != attempt:
            raise RuntimeError(
                "formal consumer attempt changed"
            )
        supervisor_ready_binding, sealed_supervisor = (
            _read_sealed_json_artifact_at(
                directory_descriptor,
                attempt_root,
                name=supervisor_ready_path.name,
                digest_field=(
                    "consumer_wait_supervisor_ready_sha256"
                ),
                kind="consumer_wait_supervisor_ready",
            )
        )
        if sealed_supervisor != supervisor_ready:
            raise RuntimeError(
                "formal consumer supervisor ready changed"
            )
        worker_started, worker_ready = (
            _read_sealed_json_artifact_at(
                directory_descriptor,
                attempt_root,
                name=Path(str(attempt["artifacts"]["ready"])).name,
                digest_field="consumer_ready_sha256",
                kind="consumer_worker_started",
            )
        )
        if (
            worker_ready.get("worker_process")
            != supervisor_ready["consumer_worker_process"]
            or worker_ready.get("supervisor_process")
            != supervisor
            or worker_ready.get("supervisor_owner_seal")
            != supervisor_ready["owner_seal"]
        ):
            raise RuntimeError(
                "formal consumer worker started differs"
            )
        terminal_path = Path(str(attempt["artifacts"]["terminal"]))
        try:
            os.stat(
                terminal_path.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            terminal_ref = None
            terminal = None
        else:
            terminal_ref, terminal = (
                _read_sealed_json_artifact_at(
                    directory_descriptor,
                    attempt_root,
                    name=terminal_path.name,
                    digest_field="consumer_terminal_sha256",
                    kind="consumer_terminal",
                )
            )
        expected_bindings = {
            "policy_sha256": attempt["policy_sha256"],
            "attempt_id": attempt["attempt_id"],
            "source_artifact": source_artifact,
            "publisher": attempt[
                "consumer_lifecycle_wait_publisher"
            ],
            "supervisor_owner_seal": supervisor_ready[
                "owner_seal"
            ],
            "supervisor_process": supervisor,
            "supervisor_executable": supervisor_ready[
                "supervisor_executable"
            ],
            "supervisor_command": supervisor_ready[
                "supervisor_arguments"
            ],
            "worker_started": worker_started,
            "child_process": supervisor_ready[
                "consumer_worker_process"
            ],
            "child_executable": supervisor_ready[
                "consumer_worker_executable"
            ],
            "child_command": supervisor_ready[
                "consumer_worker_arguments"
            ],
            "terminal": terminal_ref,
        }
        snapshot = _read_lifecycle_wait_status(
            channel_descriptor,
            directory_descriptor,
            attempt["consumer_lifecycle_wait_channel"],
            role="consumer",
            expected_bindings=expected_bindings,
        )
        record = snapshot.get("record")
        if (
            snapshot.get("state") != "valid_wait_status"
            or not isinstance(record, Mapping)
            or record.get("exit_kind") != "exit"
            or record.get("returncode")
            != CONSUMER_ADJUDICATED_EXIT
            or terminal is None
        ):
            raise RuntimeError(
                "formal consumer lifecycle outcome is not controlled"
            )
        return {
            "snapshot": snapshot,
            "terminal": terminal,
            "supervisor_ready": supervisor_ready,
            "supervisor_ready_binding": supervisor_ready_binding,
            "worker_ready": worker_ready,
            "worker_ready_binding": worker_started,
        }
    finally:
        os.close(channel_descriptor)
        os.close(directory_descriptor)


def _validate_gate_execution_ownership_chain(
    *,
    receipt_path: Path,
    gate_execution: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    specifications = (
        (
            "launch_accepted",
            receipt_path.with_name("launch_accepted.json"),
            "launch_accepted_sha256",
        ),
        (
            "launch_terminal",
            receipt_path.with_name("launch_terminal.json"),
            "launch_terminal_sha256",
        ),
        (
            "launch_ownership_release",
            receipt_path.with_name("launch_ownership_release.json"),
            "launch_ownership_release_sha256",
        ),
    )
    values = {
        field: gate_execution.get(field)
        for field, _, _ in specifications
    }
    if all(value is None for value in values.values()):
        return "absent", values
    if any(value is None for value in values.values()):
        raise RuntimeError(
            "gate execution ownership chain is partial"
        )
    expected: dict[str, Any] = {}
    for field, path, digest_field in specifications:
        expected[field] = _json_binding(path, digest_field)
    if values != expected:
        raise RuntimeError(
            "gate execution ownership chain differs"
        )
    return "bound", expected


def _publish_gate_wait_supervisor_ready(
    path: Path,
    *,
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    receipt: Mapping[str, Any],
    owner_seal: Mapping[str, Any],
    supervisor_process: Mapping[str, Any],
    supervisor_executable: Mapping[str, Any],
    supervisor_arguments: Sequence[str],
    child_process: Mapping[str, Any],
    child_executable: Mapping[str, Any],
    child_arguments: Sequence[str],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": "safa_gate_wait_supervisor_ready_v1",
        "launch_receipt": _json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        "launch_receipt_identity": dict(receipt_identity),
        "verified_implementations": dict(
            receipt["verified_implementations"]
        ),
        "wait_channel": dict(
            receipt["gate_lifecycle_wait_channel"]
        ),
        "publisher": dict(
            receipt["gate_lifecycle_wait_publisher"]
        ),
        "owner_seal": dict(owner_seal),
        "supervisor_process": dict(supervisor_process),
        "supervisor_executable": dict(supervisor_executable),
        "supervisor_arguments": list(supervisor_arguments),
        "gate_worker_process": dict(child_process),
        "gate_worker_executable": dict(child_executable),
        "gate_worker_arguments": list(child_arguments),
        "ready_at": _utc_now(),
    }
    value["gate_wait_supervisor_ready_sha256"] = _canonical_digest(
        value, "gate_wait_supervisor_ready_sha256"
    )
    _write_exclusive(path, value)
    return value


def _validate_gate_wait_supervisor_ready(
    path: Path,
    *,
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = _load_json(path, "gate wait supervisor ready")
    expected_keys = {
        "schema_version",
        "contract_type",
        "launch_receipt",
        "launch_receipt_identity",
        "verified_implementations",
        "wait_channel",
        "publisher",
        "owner_seal",
        "supervisor_process",
        "supervisor_executable",
        "supervisor_arguments",
        "gate_worker_process",
        "gate_worker_executable",
        "gate_worker_arguments",
        "ready_at",
        "gate_wait_supervisor_ready_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_gate_wait_supervisor_ready_v1"
        or value.get("launch_receipt")
        != _json_binding(receipt_path, "launch_receipt_sha256")
        or value.get("launch_receipt_identity")
        != dict(receipt_identity)
        or _opened_file_identity(receipt_path)
        != dict(receipt_identity)
        or value.get("verified_implementations")
        != receipt["verified_implementations"]
        or value.get("wait_channel")
        != receipt["gate_lifecycle_wait_channel"]
        or value.get("publisher")
        != receipt["gate_lifecycle_wait_publisher"]
        or value.get("supervisor_arguments")
        != receipt["gate_lifecycle_wait_supervisor_arguments"]
        or value.get("gate_worker_arguments")
        != receipt["gate_worker_arguments"]
        or value.get("gate_wait_supervisor_ready_sha256")
        != _canonical_digest(
            value, "gate_wait_supervisor_ready_sha256"
        )
    ):
        raise RuntimeError(
            "gate wait supervisor ready contract differs"
        )
    supervisor = value.get("supervisor_process")
    child = value.get("gate_worker_process")
    owner = value.get("owner_seal")
    if (
        not isinstance(supervisor, dict)
        or not isinstance(child, dict)
        or not isinstance(owner, dict)
        or owner.get("pane_process") != supervisor
        or owner.get("pane_pid") != supervisor.get("pid")
        or child.get("ppid") != supervisor.get("pid")
        or child.get("pgid") != child.get("pid")
        or child.get("sid") != child.get("pid")
    ):
        raise RuntimeError(
            "gate wait supervisor process relation differs"
        )
    return value


def _publish_consumer_wait_supervisor_ready(
    path: Path,
    *,
    attempt_path: Path,
    attempt_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    owner_seal: Mapping[str, Any],
    supervisor_process: Mapping[str, Any],
    supervisor_executable: Mapping[str, Any],
    supervisor_arguments: Sequence[str],
    child_process: Mapping[str, Any],
    child_executable: Mapping[str, Any],
    child_arguments: Sequence[str],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": "safa_consumer_wait_supervisor_ready_v1",
        "consumer_attempt": _json_binding(
            attempt_path, "consumer_attempt_sha256"
        ),
        "consumer_attempt_identity": dict(attempt_identity),
        "wait_channel": dict(
            attempt["consumer_lifecycle_wait_channel"]
        ),
        "publisher": dict(
            attempt["consumer_lifecycle_wait_publisher"]
        ),
        "owner_seal": dict(owner_seal),
        "supervisor_process": dict(supervisor_process),
        "supervisor_executable": dict(supervisor_executable),
        "supervisor_arguments": list(supervisor_arguments),
        "consumer_worker_process": dict(child_process),
        "consumer_worker_executable": dict(child_executable),
        "consumer_worker_arguments": list(child_arguments),
        "ready_at": _utc_now(),
    }
    value["consumer_wait_supervisor_ready_sha256"] = (
        _canonical_digest(
            value, "consumer_wait_supervisor_ready_sha256"
        )
    )
    _write_exclusive(path, value)
    return value


def _validate_consumer_wait_supervisor_ready(
    path: Path,
    *,
    attempt_path: Path,
    attempt_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    value = _load_json(path, "consumer wait supervisor ready")
    expected_keys = {
        "schema_version",
        "contract_type",
        "consumer_attempt",
        "consumer_attempt_identity",
        "wait_channel",
        "publisher",
        "owner_seal",
        "supervisor_process",
        "supervisor_executable",
        "supervisor_arguments",
        "consumer_worker_process",
        "consumer_worker_executable",
        "consumer_worker_arguments",
        "ready_at",
        "consumer_wait_supervisor_ready_sha256",
    }
    supervisor = value.get("supervisor_process")
    child = value.get("consumer_worker_process")
    owner = value.get("owner_seal")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_consumer_wait_supervisor_ready_v1"
        or value.get("consumer_attempt")
        != _json_binding(
            attempt_path, "consumer_attempt_sha256"
        )
        or value.get("consumer_attempt_identity")
        != dict(attempt_identity)
        or _opened_file_identity(attempt_path)
        != dict(attempt_identity)
        or value.get("wait_channel")
        != attempt["consumer_lifecycle_wait_channel"]
        or value.get("publisher")
        != attempt["consumer_lifecycle_wait_publisher"]
        or value.get("supervisor_arguments")
        != attempt["consumer_lifecycle_wait_supervisor_arguments"]
        or value.get("consumer_worker_arguments")
        != attempt["consumer_worker_arguments"]
        or not isinstance(supervisor, dict)
        or not isinstance(child, dict)
        or not isinstance(owner, dict)
        or owner.get("pane_process") != supervisor
        or owner.get("pane_pid") != supervisor.get("pid")
        or child.get("ppid") != supervisor.get("pid")
        or child.get("pgid") != child.get("pid")
        or child.get("sid") != child.get("pid")
        or value.get("consumer_wait_supervisor_ready_sha256")
        != _canonical_digest(
            value, "consumer_wait_supervisor_ready_sha256"
        )
    ):
        raise RuntimeError(
            "consumer wait supervisor ready contract differs"
        )
    return value


def _waitid_then_waitpid(
    child: subprocess.Popen[Any],
) -> tuple[Any, int, int]:
    while True:
        try:
            info = os.waitid(
                os.P_PID,
                child.pid,
                os.WEXITED | os.WNOWAIT,
            )
            break
        except InterruptedError:
            continue
    if info is None or info.si_pid != child.pid:
        raise RuntimeError("gate worker waitid identity differs")
    while True:
        try:
            waited_pid, raw_status = os.waitpid(child.pid, 0)
            break
        except InterruptedError:
            continue
    if waited_pid != child.pid:
        raise RuntimeError("gate worker waitpid identity differs")
    if os.WIFEXITED(raw_status):
        child.returncode = os.WEXITSTATUS(raw_status)
    elif os.WIFSIGNALED(raw_status):
        child.returncode = -os.WTERMSIG(raw_status)
    else:
        raise RuntimeError("gate worker wait status is not terminal")
    return info, waited_pid, raw_status


def _gate_wait_supervisor(
    *,
    receipt_path: Path,
    attempt_id: str,
    wait_channel_path: Path,
    gate_worker_arguments: Sequence[str],
) -> int:
    receipt = _load_json(receipt_path, "gate wait supervisor receipt")
    validate_launch_receipt_schema(
        receipt,
        expected_gate_worker_arguments=gate_worker_arguments,
        expected_consumer_worker_arguments=(
            _expected_consumer_worker_arguments_from_receipt(receipt)
        ),
        label="gate wait supervisor receipt v3",
    )
    supervisor_arguments = _process_command(os.getpid())
    receipt_identity = _opened_file_identity(receipt_path)
    attempt_root = receipt_path.parent
    if (
        receipt.get("attempt_id") != attempt_id
        or receipt.get("gate_lifecycle_wait_status_path")
        != str(wait_channel_path)
        or receipt.get("gate_worker_arguments")
        != list(gate_worker_arguments)
        or receipt.get("gate_lifecycle_wait_supervisor_arguments")
        != supervisor_arguments
    ):
        raise RuntimeError(
            "gate wait supervisor invocation differs from receipt"
        )
    log_path = attempt_root / "pane.log"
    log_descriptor = os.open(
        log_path, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC
    )
    try:
        log_identity = os.fstat(log_descriptor)
        expected_log = receipt.get("pane_log")
        if (
            not isinstance(expected_log, Mapping)
            or expected_log.get("path") != str(log_path)
            or expected_log.get("device") != log_identity.st_dev
            or expected_log.get("inode") != log_identity.st_ino
            or expected_log.get("mode") != log_identity.st_mode
        ):
            raise RuntimeError(
                "gate wait supervisor pane log identity differs"
            )
        os.dup2(log_descriptor, sys.stdout.fileno())
        os.dup2(log_descriptor, sys.stderr.fileno())
    finally:
        os.close(log_descriptor)
    channel_descriptor, directory_descriptor = (
        _open_presealed_lifecycle_wait_channel(
            attempt_root,
            receipt["gate_lifecycle_wait_channel"],
            name=wait_channel_path.name,
        )
    )
    child: subprocess.Popen[Any] | None = None
    forwarded_signals: list[int] = []

    def forward_signal(signum: int, _frame: Any) -> None:
        forwarded_signals.append(signum)
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    old_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    started_at = _utc_now()
    try:
        owner_seal = _tmux_owner_seal(
            str(receipt["controller_session"]),
            str(receipt["controller_owner_nonce"]),
        )
        supervisor_process = _process_identity(os.getpid())
        supervisor_executable = _process_executable(os.getpid())
        if (
            owner_seal.get("pane_process") != supervisor_process
            or owner_seal.get("pane_pid") != os.getpid()
            or owner_seal.get("pane_dead")
        ):
            raise RuntimeError(
                "gate wait supervisor tmux owner differs"
            )
        child = subprocess.Popen(
            list(gate_worker_arguments),
            shell=False,
            preexec_fn=_wrapper_child_setup,
            close_fds=True,
        )
        process_deadline = time.monotonic() + 2.0
        while True:
            try:
                child_process = _process_identity(child.pid)
                child_command = _process_command(child.pid)
                child_executable = _process_executable(child.pid)
            except (OSError, RuntimeError, UnicodeError):
                if time.monotonic() >= process_deadline:
                    raise RuntimeError(
                        "gate worker process seal timed out"
                    )
                time.sleep(0.005)
                continue
            if (
                child_process["ppid"] == os.getpid()
                and child_process["pgid"] == child.pid
                and child_process["sid"] == child.pid
                and child_command == list(gate_worker_arguments)
            ):
                break
            if time.monotonic() >= process_deadline:
                raise RuntimeError(
                    "gate worker process seal differs"
                )
            time.sleep(0.005)
        ready_path = Path(
            str(receipt["gate_lifecycle_wait_supervisor_ready_path"])
        )
        _publish_gate_wait_supervisor_ready(
            ready_path,
            receipt_path=receipt_path,
            receipt_identity=receipt_identity,
            receipt=receipt,
            owner_seal=owner_seal,
            supervisor_process=supervisor_process,
            supervisor_executable=supervisor_executable,
            supervisor_arguments=supervisor_arguments,
            child_process=child_process,
            child_executable=child_executable,
            child_arguments=gate_worker_arguments,
        )
        info, waited_pid, raw_status = _waitid_then_waitpid(child)
        gate_ready_path = attempt_root / "pane_gate_ready.json"
        gate_ready = _load_json(gate_ready_path, "pane gate ready")
        if (
            gate_ready.get("process") != child_process
            or gate_ready.get("wrapper_arguments")
            != receipt["wrapper_arguments"]
        ):
            raise RuntimeError(
                "gate worker ready differs after wait"
            )
        tmux_started_path = attempt_root / "launch_tmux_started.json"
        tmux_started = _load_json(
            tmux_started_path, "launch tmux started"
        )
        started_owner = tmux_started.get("owner_seal")
        owner_server = owner_seal["tmux_server"]
        if (
            not isinstance(started_owner, Mapping)
            or started_owner.get("session")
            != owner_seal["session"]
            or started_owner.get("pane") != owner_seal["pane"]
            or started_owner.get("pane_pid")
            != owner_seal["pane_pid"]
            or started_owner.get("pane_process")
            != owner_seal["pane_process"]
            or started_owner.get("owner_nonce")
            != owner_seal["owner_nonce"]
            or started_owner.get("server_pid")
            != owner_server["server_pid"]
            or started_owner.get("server_start_ticks")
            != owner_server["server_process"]["start_ticks"]
            or started_owner.get("socket_path")
            != owner_server["socket_path"]
            or started_owner.get("socket_device")
            != owner_server["socket_device"]
            or started_owner.get("socket_inode")
            != owner_server["socket_inode"]
        ):
            raise RuntimeError(
                "gate wait supervisor owner ancestor differs"
            )
        terminal_path = Path(
            str(receipt["gate_execution_terminal_path"])
        )
        terminal_ref: dict[str, Any] | None = None
        terminal: dict[str, Any] | None = None
        if terminal_path.is_file():
            terminal = _validate_gate_execution_terminal(
                terminal_path,
                receipt_binding=_json_binding(
                    receipt_path, "launch_receipt_sha256"
                ),
                receipt_identity=receipt_identity,
                gate_ready_binding=_json_binding(
                    gate_ready_path, "pane_gate_ready_sha256"
                ),
                wrapper_arguments=receipt["wrapper_arguments"],
            )
            terminal_ref = _sealed_lifecycle_artifact(
                terminal_path,
                digest_field="gate_execution_terminal_sha256",
                kind="gate_execution_terminal",
            )
        if info.si_code == os.CLD_EXITED:
            if terminal is None:
                raise RuntimeError(
                    "controlled gate worker exit lacks terminal"
                )
            if child.returncode != GATE_ADJUDICATED_EXIT:
                raise RuntimeError(
                    "gate worker controlled exit marker differs"
                )
        record = build_lifecycle_wait_status(
            role="gate",
            policy_sha256=str(receipt["policy_sha256"]),
            attempt_id=attempt_id,
            source_artifact=_sealed_lifecycle_artifact(
                receipt_path,
                digest_field="launch_receipt_sha256",
                kind="launch_receipt",
            ),
            wait_channel=receipt["gate_lifecycle_wait_channel"],
            publisher=receipt["gate_lifecycle_wait_publisher"],
            supervisor_owner_seal=owner_seal,
            supervisor_process=supervisor_process,
            supervisor_executable=supervisor_executable,
            supervisor_command=supervisor_arguments,
            worker_started=_sealed_lifecycle_artifact(
                gate_ready_path,
                digest_field="pane_gate_ready_sha256",
                kind="gate_worker_started",
            ),
            child_process=child_process,
            child_executable=child_executable,
            child_command=list(gate_worker_arguments),
            terminal=terminal_ref,
            waitid_si_pid=int(info.si_pid),
            waitid_si_code=int(info.si_code),
            waitid_si_status=int(info.si_status),
            waited_pid=waited_pid,
            wait_status_raw=raw_status,
            started_at=started_at,
            completed_at=_utc_now(),
        )
        _write_lifecycle_wait_status(
            channel_descriptor,
            directory_descriptor,
            receipt["gate_lifecycle_wait_channel"],
            record,
            role="gate",
        )
        return (
            int(child.returncode)
            if child.returncode is not None
            and child.returncode >= 0
            else 128 + int(-child.returncode)
        )
    finally:
        if child is not None and child.returncode is None:
            _terminate_spawned_child(child)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        os.close(channel_descriptor)
        os.close(directory_descriptor)


def _create_secure_append_log(path: Path) -> dict[str, Any]:
    directory_descriptor, _directory = _open_secure_directory(
        path.parent
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o644,
            dir_fd=directory_descriptor,
        )
        os.fchmod(descriptor, 0o644)
        opened = os.fstat(descriptor)
        named = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o644
            or opened.st_nlink != 1
            or opened.st_size != 0
        ):
            raise RuntimeError(
                "pane fault consumer log identity differs"
            )
        os.fsync(descriptor)
        os.fsync(directory_descriptor)
        return build_file_identity(
            path=str(path),
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
            mode=int(opened.st_mode),
            size=int(opened.st_size),
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def _open_consumer_log(
    path: Path, binding: Mapping[str, Any]
) -> int:
    directory_descriptor, _directory = _open_secure_directory(
        path.parent
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_APPEND
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        opened = os.fstat(descriptor)
        named = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        observed = build_file_identity(
            path=str(path),
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
            mode=int(opened.st_mode),
            size=int(opened.st_size),
        )
        if (
            dict(binding) != observed
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o644
            or opened.st_nlink != 1
        ):
            raise RuntimeError(
                "pane fault consumer log seal differs"
            )
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def _consumer_wait_supervisor(
    *,
    attempt_path: Path,
    config: Path,
    wait_channel_path: Path,
    consumer_worker_arguments: Sequence[str],
) -> int:
    _install_verified_preflight_apis(config)
    attempt = _load_json(
        attempt_path, "consumer wait supervisor attempt"
    )
    attempt_identity = _opened_file_identity(attempt_path)
    receipt_path = Path(str(attempt["launch_receipt"]["path"]))
    receipt = _load_json(
        receipt_path, "consumer wait supervisor receipt"
    )
    validate_launch_receipt_schema(
        receipt,
        expected_gate_worker_arguments=receipt[
            "gate_worker_arguments"
        ],
        expected_consumer_worker_arguments=(
            consumer_worker_arguments
        ),
        label="consumer wait supervisor receipt v4",
    )
    supervisor_arguments = _process_command(os.getpid())
    if (
        attempt.get("consumer_lifecycle_wait_status_path")
        != str(wait_channel_path)
        or attempt.get("consumer_worker_arguments")
        != list(consumer_worker_arguments)
        or attempt.get(
            "consumer_lifecycle_wait_supervisor_arguments"
        )
        != supervisor_arguments
        or receipt.get("consumer_worker_arguments")
        != list(consumer_worker_arguments)
        or receipt.get(
            "consumer_lifecycle_wait_supervisor_arguments"
        )
        != supervisor_arguments
    ):
        raise RuntimeError(
            "consumer wait supervisor invocation differs"
        )
    log_binding = attempt["consumer_log"]
    log_descriptor = _open_consumer_log(
        Path(str(log_binding["path"])), log_binding
    )
    os.dup2(log_descriptor, sys.stdout.fileno())
    os.dup2(log_descriptor, sys.stderr.fileno())
    os.close(log_descriptor)
    channel_descriptor, directory_descriptor = (
        _open_presealed_lifecycle_wait_channel(
            attempt_path.parent,
            attempt["consumer_lifecycle_wait_channel"],
            name=wait_channel_path.name,
        )
    )
    child: subprocess.Popen[Any] | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    old_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    started_at = _utc_now()
    try:
        owner_seal = _tmux_owner_seal(
            str(attempt["consumer_session"]),
            str(attempt["consumer_owner_nonce"]),
        )
        supervisor_process = _process_identity(os.getpid())
        supervisor_executable = _process_executable(os.getpid())
        if (
            owner_seal.get("pane_process") != supervisor_process
            or owner_seal.get("pane_pid") != os.getpid()
            or owner_seal.get("pane_dead")
            or owner_seal.get("pane_dead_status") is not None
        ):
            raise RuntimeError(
                "consumer wait supervisor tmux owner differs"
            )
        child = subprocess.Popen(
            list(consumer_worker_arguments),
            shell=False,
            preexec_fn=_wrapper_child_setup,
            close_fds=True,
        )
        process_deadline = time.monotonic() + 2.0
        while True:
            try:
                child_process = _process_identity(child.pid)
                child_command = _process_command(child.pid)
                child_executable = _process_executable(child.pid)
            except (OSError, RuntimeError, UnicodeError):
                if time.monotonic() >= process_deadline:
                    raise RuntimeError(
                        "consumer worker process seal timed out"
                    )
                time.sleep(0.005)
                continue
            if (
                child_process["ppid"] == os.getpid()
                and child_process["pgid"] == child.pid
                and child_process["sid"] == child.pid
                and child_command == list(consumer_worker_arguments)
            ):
                break
            if time.monotonic() >= process_deadline:
                raise RuntimeError(
                    "consumer worker process seal differs"
                )
            time.sleep(0.005)
        ready_path = Path(
            str(
                attempt[
                    "consumer_lifecycle_wait_supervisor_ready_path"
                ]
            )
        )
        _publish_consumer_wait_supervisor_ready(
            ready_path,
            attempt_path=attempt_path,
            attempt_identity=attempt_identity,
            attempt=attempt,
            owner_seal=owner_seal,
            supervisor_process=supervisor_process,
            supervisor_executable=supervisor_executable,
            supervisor_arguments=supervisor_arguments,
            child_process=child_process,
            child_executable=child_executable,
            child_arguments=consumer_worker_arguments,
        )
        info, waited_pid, raw_status = _waitid_then_waitpid(child)
        worker_ready_path = Path(str(attempt["artifacts"]["ready"]))
        worker_ready = _load_json(
            worker_ready_path, "consumer worker ready after wait"
        )
        if (
            worker_ready.get("worker_process") != child_process
            or worker_ready.get("supervisor_owner_seal")
            != owner_seal
            or worker_ready.get("supervisor_process")
            != supervisor_process
        ):
            raise RuntimeError(
                "consumer worker ready differs after wait"
            )
        terminal_path = Path(str(attempt["artifacts"]["terminal"]))
        terminal_ref: dict[str, Any] | None = None
        terminal: dict[str, Any] | None = None
        if terminal_path.is_file():
            terminal = _load_json(
                terminal_path, "consumer worker terminal"
            )
            if (
                terminal.get("consumer_terminal_sha256")
                != _canonical_digest(
                    terminal, "consumer_terminal_sha256"
                )
                or terminal.get("consumer_attempt")
                != _json_binding(
                    attempt_path, "consumer_attempt_sha256"
                )
                or terminal.get("supervisor_owner_seal")
                != owner_seal
                or terminal.get("supervisor_process")
                != supervisor_process
                or terminal.get("worker_process") != child_process
                or terminal.get("exit_code") != 0
            ):
                raise RuntimeError(
                    "consumer worker terminal differs"
                )
            terminal_ref = _sealed_lifecycle_artifact(
                terminal_path,
                digest_field="consumer_terminal_sha256",
                kind="consumer_terminal",
            )
        if info.si_code == os.CLD_EXITED:
            if terminal is None:
                raise RuntimeError(
                    "controlled consumer worker exit lacks terminal"
                )
            if child.returncode != CONSUMER_ADJUDICATED_EXIT:
                raise RuntimeError(
                    "consumer worker controlled exit marker differs"
                )
        record = build_lifecycle_wait_status(
            role="consumer",
            policy_sha256=str(attempt["policy_sha256"]),
            attempt_id=str(attempt["attempt_id"]),
            source_artifact=_sealed_lifecycle_artifact(
                attempt_path,
                digest_field="consumer_attempt_sha256",
                kind="consumer_attempt",
            ),
            wait_channel=attempt["consumer_lifecycle_wait_channel"],
            publisher=attempt[
                "consumer_lifecycle_wait_publisher"
            ],
            supervisor_owner_seal=owner_seal,
            supervisor_process=supervisor_process,
            supervisor_executable=supervisor_executable,
            supervisor_command=supervisor_arguments,
            worker_started=_sealed_lifecycle_artifact(
                worker_ready_path,
                digest_field="consumer_ready_sha256",
                kind="consumer_worker_started",
            ),
            child_process=child_process,
            child_executable=child_executable,
            child_command=list(consumer_worker_arguments),
            terminal=terminal_ref,
            waitid_si_pid=int(info.si_pid),
            waitid_si_code=int(info.si_code),
            waitid_si_status=int(info.si_status),
            waited_pid=waited_pid,
            wait_status_raw=raw_status,
            started_at=started_at,
            completed_at=_utc_now(),
        )
        _write_lifecycle_wait_status(
            channel_descriptor,
            directory_descriptor,
            attempt["consumer_lifecycle_wait_channel"],
            record,
            role="consumer",
        )
        return (
            int(child.returncode)
            if child.returncode is not None
            and child.returncode >= 0
            else 128 + int(-child.returncode)
        )
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        os.close(channel_descriptor)
        os.close(directory_descriptor)


def _pane_fault_consumer(
    *,
    attempt_path: Path,
    config: Path,
) -> int:
    _install_verified_preflight_apis(config)
    attempt = _load_json(
        attempt_path, "pane fault consumer attempt"
    )
    if (
        attempt.get("consumer_attempt_sha256")
        != _canonical_digest(
            attempt, "consumer_attempt_sha256"
        )
        or attempt.get("contract_type")
        != "safa_pane_fault_consumer_attempt_v1"
    ):
        raise RuntimeError(
            "pane fault consumer attempt contract differs"
        )
    log_binding = attempt.get("consumer_log")
    channel_binding = attempt.get("pane_fault_channel")
    self_channel_binding = attempt.get(
        "consumer_self_fault_channel"
    )
    if (
        not isinstance(log_binding, Mapping)
        or not isinstance(channel_binding, Mapping)
        or not isinstance(self_channel_binding, Mapping)
    ):
        raise RuntimeError(
            "pane fault consumer sealed files differ"
        )
    log_descriptor = _open_consumer_log(
        Path(str(log_binding["path"])), log_binding
    )
    os.dup2(log_descriptor, sys.stdout.fileno())
    os.dup2(log_descriptor, sys.stderr.fileno())
    os.close(log_descriptor)
    fault_descriptor = -1
    self_fault_descriptor = -1
    try:
        fault_descriptor = _open_presealed_fault_channel(
            Path(str(channel_binding["path"])).parent,
            channel_binding,
            name=Path(str(channel_binding["path"])).name,
        )
        self_fault_descriptor = _open_presealed_fault_channel(
            Path(str(self_channel_binding["path"])).parent,
            self_channel_binding,
            name=Path(str(self_channel_binding["path"])).name,
        )
        session = str(attempt["consumer_session"])
        owner_nonce = str(attempt["consumer_owner_nonce"])
        supervisor_ready_path = Path(
            str(
                attempt[
                    "consumer_lifecycle_wait_supervisor_ready_path"
                ]
            )
        )
        supervisor_deadline = time.monotonic() + 5.0
        while not supervisor_ready_path.is_file():
            if time.monotonic() >= supervisor_deadline:
                raise RuntimeError(
                    "consumer wait supervisor ready timed out"
                )
            time.sleep(0.005)
        supervisor_ready = (
            _validate_consumer_wait_supervisor_ready(
                supervisor_ready_path,
                attempt_path=attempt_path,
                attempt_identity=_opened_file_identity(attempt_path),
                attempt=attempt,
            )
        )
        owner_seal = supervisor_ready["owner_seal"]
        supervisor_process = supervisor_ready[
            "supervisor_process"
        ]
        supervisor_executable = supervisor_ready[
            "supervisor_executable"
        ]
        supervisor_command = supervisor_ready[
            "supervisor_arguments"
        ]
        process = _process_identity(os.getpid())
        command = _process_command(os.getpid())
        executable = _process_executable(os.getpid())
        if (
            owner_seal["pane_dead"]
            or owner_seal["pane_pid"] != supervisor_process["pid"]
            or owner_seal["pane_process"] != supervisor_process
            or process
            != supervisor_ready["consumer_worker_process"]
            or process["ppid"] != supervisor_process["pid"]
            or command != attempt.get("consumer_worker_arguments")
            or _command_bytes(command)
            != _process_command_bytes(os.getpid())
            or executable
            != supervisor_ready["consumer_worker_executable"]
        ):
            raise RuntimeError(
                "pane fault consumer live process seal differs"
            )
        ready = {
            "schema_version": 1,
            "contract_type": (
                "safa_pane_fault_consumer_ready_v1"
            ),
            "policy_sha256": attempt["policy_sha256"],
            "attempt_id": attempt["attempt_id"],
            "consumer_attempt": _json_binding(
                attempt_path, "consumer_attempt_sha256"
            ),
            "consumer_session": session,
            "consumer_owner_nonce": owner_nonce,
            "consumer_wait_supervisor_ready": _json_binding(
                supervisor_ready_path,
                "consumer_wait_supervisor_ready_sha256",
            ),
            "supervisor_owner_seal": owner_seal,
            "supervisor_process": supervisor_process,
            "supervisor_command": supervisor_command,
            "supervisor_executable": supervisor_executable,
            "tmux": {
                key: owner_seal[key]
                for key in (
                    "session",
                    "pane",
                    "pane_pid",
                    "pane_dead",
                    "pane_dead_status",
                )
            },
            "tmux_server": owner_seal["tmux_server"],
            "worker_process": process,
            "worker_command": command,
            "worker_executable": executable,
            "fault_descriptor": fault_descriptor,
            "pane_fault_channel": dict(channel_binding),
            "self_fault_descriptor": self_fault_descriptor,
            "consumer_self_fault_channel": dict(
                self_channel_binding
            ),
            "ready_at": _utc_now(),
        }
        ready["consumer_ready_sha256"] = _canonical_digest(
            ready, "consumer_ready_sha256"
        )
        _write_exclusive(
            Path(str(attempt["artifacts"]["ready"])), ready
        )
        try:
            _pane_fault_consumer_transfer(
                attempt=attempt,
                attempt_path=attempt_path,
                ready=ready,
                owner_seal=owner_seal,
                process=process,
                fault_descriptor=fault_descriptor,
                self_fault_descriptor=self_fault_descriptor,
            )
            while True:
                final_snapshots = (
                    _require_empty_pane_fault_consumer_channels(
                        attempt=attempt,
                        fault_descriptor=fault_descriptor,
                        self_fault_descriptor=self_fault_descriptor,
                    )
                )
                gate_pane = _tmux_pane(
                    str(attempt["gate_owner_seal"]["session"])
                )
                if gate_pane is None:
                    raise RuntimeError(
                        "controller pane disappeared before a sealed dead "
                        "status was observed"
                    )
                if (
                    not gate_pane["pane_dead"]
                    and gate_pane["pane_dead_status"] is not None
                ):
                    raise RuntimeError(
                        "controller live pane exposed an exit status"
                    )
                if gate_pane["pane_dead"]:
                    final_snapshots = (
                        _require_empty_pane_fault_consumer_channels(
                            attempt=attempt,
                            fault_descriptor=fault_descriptor,
                            self_fault_descriptor=self_fault_descriptor,
                        )
                    )
                    dead_owner_seal = _tmux_owner_seal(
                        str(
                            attempt["gate_owner_seal"]["session"]
                        ),
                        str(
                            attempt["gate_owner_seal"]["owner_nonce"]
                        ),
                    )
                    original_gate_owner = attempt["gate_owner_seal"]
                    dead_owner_seal = (
                        _validate_pane_owner_lifecycle_transition(
                            original_gate_owner,
                            dead_owner_seal,
                            label="controller pane owner",
                        )
                    )
                    receipt_path = Path(
                        str(attempt["launch_receipt"]["path"])
                    )
                    receipt = _load_json(
                        receipt_path,
                        "consumer final launch receipt",
                    )
                    formal_gate = (
                        _read_formal_gate_lifecycle_status(
                            attempt_root=receipt_path.parent,
                            pane=gate_pane,
                        )
                    )
                    gate_execution_path = Path(
                        str(
                            receipt[
                                "gate_execution_terminal_path"
                            ]
                        )
                    )
                    gate_execution = formal_gate["gate_execution"]
                    gate_adjudication = formal_gate["adjudication"]
                    ownership_chain_state = formal_gate[
                        "ownership_chain_state"
                    ]
                    ownership_chain = formal_gate["ownership_chain"]
                    launch_accepted = ownership_chain[
                        "launch_accepted"
                    ]
                    launch_terminal = ownership_chain[
                        "launch_terminal"
                    ]
                    launch_release = ownership_chain[
                        "launch_ownership_release"
                    ]
                    if (
                        gate_execution["launch_accepted"]
                        != launch_accepted
                        or gate_execution["launch_terminal"]
                        != launch_terminal
                        or gate_execution[
                            "launch_ownership_release"
                        ]
                        != launch_release
                    ):
                        raise RuntimeError(
                            "consumer final gate execution chain differs"
                        )
                    _kill_exact_session(
                        str(original_gate_owner["session"]),
                        str(original_gate_owner["owner_nonce"]),
                        original_gate_owner,
                    )
                    if (
                        _tmux_pane(
                            str(original_gate_owner["session"])
                        )
                        is not None
                    ):
                        raise RuntimeError(
                            "controller cleanup left session residual"
                        )
                    cleanup = {
                        "schema_version": 1,
                        "contract_type": (
                            "safa_pane_fault_consumer_controller_cleanup_v2"
                        ),
                        "policy_sha256": attempt["policy_sha256"],
                        "attempt_id": attempt["attempt_id"],
                        "consumer_attempt": _json_binding(
                            attempt_path,
                            "consumer_attempt_sha256",
                        ),
                        "gate_owner_seal": attempt["gate_owner_seal"],
                        "gate_execution_terminal": _json_binding(
                            gate_execution_path,
                            "gate_execution_terminal_sha256",
                        ),
                        "launch_accepted": launch_accepted,
                        "launch_terminal": launch_terminal,
                        "launch_ownership_release": launch_release,
                        "consumer_reader_release": _json_binding(
                            Path(
                                str(
                                    attempt["artifacts"][
                                        "reader_release"
                                    ]
                                )
                            ),
                            "consumer_reader_release_sha256",
                        ),
                        "dead_owner_seal": dead_owner_seal,
                        "controller_exit_code": gate_adjudication[
                            "controller_exit_code"
                        ],
                        "adjudicated_outcome": gate_adjudication[
                            "adjudicated_outcome"
                        ],
                        "ownership_chain_state": (
                            ownership_chain_state
                        ),
                        "status": "controller_dead_cleaned",
                        "cleanup_performed": True,
                        "session_residual": False,
                        "process_residual": False,
                        "final_empty_snapshots": final_snapshots,
                        "completed_at": _utc_now(),
                    }
                    cleanup[
                        "consumer_controller_cleanup_sha256"
                    ] = _canonical_digest(
                        cleanup,
                        "consumer_controller_cleanup_sha256",
                    )
                    cleanup_path = Path(
                        str(
                            attempt["artifacts"][
                                "controller_cleanup"
                            ]
                        )
                    )
                    _write_exclusive(cleanup_path, cleanup)
                    terminal = {
                        "schema_version": 1,
                        "contract_type": (
                            "safa_pane_fault_consumer_terminal_v2"
                        ),
                        "policy_sha256": attempt["policy_sha256"],
                        "attempt_id": attempt["attempt_id"],
                        "consumer_attempt": _json_binding(
                            attempt_path,
                            "consumer_attempt_sha256",
                        ),
                        "consumer_started": _json_binding(
                            Path(
                                str(attempt["artifacts"]["started"])
                            ),
                            "consumer_started_sha256",
                        ),
                        "consumer_active": _json_binding(
                            Path(
                                str(attempt["artifacts"]["active"])
                            ),
                            "consumer_active_sha256",
                        ),
                        "consumer_reader_release": _json_binding(
                            Path(
                                str(
                                    attempt["artifacts"][
                                        "reader_release"
                                    ]
                                )
                            ),
                            "consumer_reader_release_sha256",
                        ),
                        "consumer_release_observed": _json_binding(
                            Path(
                                str(
                                    attempt["artifacts"][
                                        "release_observed"
                                    ]
                                )
                            ),
                            "consumer_release_observed_sha256",
                        ),
                        "controller_cleanup": _json_binding(
                            cleanup_path,
                            "consumer_controller_cleanup_sha256",
                        ),
                        "gate_execution_terminal": _json_binding(
                            gate_execution_path,
                            "gate_execution_terminal_sha256",
                        ),
                        "launch_accepted": launch_accepted,
                        "launch_terminal": launch_terminal,
                        "launch_ownership_release": launch_release,
                        "ownership_chain_state": (
                            ownership_chain_state
                        ),
                        "consumer_session": session,
                        "consumer_owner_nonce": owner_nonce,
                        "supervisor_owner_seal": dict(owner_seal),
                        "supervisor_process": dict(
                            supervisor_process
                        ),
                        "worker_process": dict(process),
                        "final_empty_snapshots": final_snapshots,
                        "controller_exit_code": gate_adjudication[
                            "controller_exit_code"
                        ],
                        "status": gate_adjudication[
                            "adjudicated_outcome"
                        ],
                        "exit_code": 0,
                        "completed_at": _utc_now(),
                    }
                    terminal[
                        "consumer_terminal_sha256"
                    ] = _canonical_digest(
                        terminal, "consumer_terminal_sha256"
                    )
                    _write_exclusive(
                        Path(
                            str(attempt["artifacts"]["terminal"])
                        ),
                        terminal,
                    )
                    return CONSUMER_ADJUDICATED_EXIT
                _require_empty_pane_fault_consumer_channels(
                    attempt=attempt,
                    fault_descriptor=fault_descriptor,
                    self_fault_descriptor=self_fault_descriptor,
                )
                if (
                    _tmux_owner_seal(session, owner_nonce)
                    != owner_seal
                ):
                    raise RuntimeError(
                        "pane fault consumer live owner changed"
                    )
                time.sleep(0.02)
        except BaseException as exc:
            poison = LauncherExclusivePublishError(
                "precommit_failed_clean",
                (
                    "pane fault consumer runtime monitor failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                stage="pane_fault_consumer_runtime_monitor",
                directory_seal={},
                payload={
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                },
                temporary=None,
                error_number=(
                    exc.errno if isinstance(exc, OSError) else None
                ),
                quarantined=False,
            )
            _write_launcher_fault_channel_record(
                self_fault_descriptor,
                self_channel_binding,
                attempt_id=str(attempt["attempt_id"]),
                owner_nonce=owner_nonce,
                launch_receipt_sha256=str(
                    attempt["launch_receipt"]["canonical_sha256"]
                ),
                publisher=attempt[
                    "consumer_self_fault_publisher"
                ],
                failure=poison,
            )
            raise
    finally:
        if self_fault_descriptor >= 0:
            os.close(self_fault_descriptor)
        if fault_descriptor >= 0:
            os.close(fault_descriptor)


def _require_empty_pane_fault_consumer_channels(
    *,
    attempt: Mapping[str, Any],
    fault_descriptor: int,
    self_fault_descriptor: int,
) -> dict[str, dict[str, Any]]:
    receipt_binding = attempt["launch_receipt"]
    receipt_sha256 = str(receipt_binding["canonical_sha256"])
    checks = (
        (
            fault_descriptor,
            attempt["pane_fault_channel"],
            attempt["gate_owner_seal"]["owner_nonce"],
            attempt["pane_fault_publisher"],
            "pane_gate",
        ),
        (
            self_fault_descriptor,
            attempt["consumer_self_fault_channel"],
            attempt["consumer_owner_nonce"],
            attempt["consumer_self_fault_publisher"],
            "consumer_self",
        ),
    )
    snapshots: dict[str, dict[str, Any]] = {}
    for descriptor, binding, owner_nonce, publisher, label in checks:
        try:
            snapshot = _read_fault_channel(
                int(descriptor),
                binding,
                attempt_id=str(attempt["attempt_id"]),
                owner_nonce=str(owner_nonce),
                launch_receipt_sha256=receipt_sha256,
                publisher=publisher,
            )
        except BaseException as exc:
            raise LauncherGateFaultError(
                f"{label}_fault_channel_invalid",
                snapshot=None,
                failure=exc,
            ) from exc
        if snapshot.get("state") == "valid_fault":
            raise LauncherGateFaultError(
                f"{label}_typed_publish_failure",
                snapshot=snapshot,
            )
        if snapshot.get("state") != "empty":
            raise LauncherGateFaultError(
                f"{label}_fault_channel_nonempty",
                snapshot=snapshot,
            )
        snapshots[label] = snapshot
    return snapshots


def _pane_fault_consumer_transfer(
    *,
    attempt: Mapping[str, Any],
    attempt_path: Path,
    ready: Mapping[str, Any],
    owner_seal: Mapping[str, Any],
    process: Mapping[str, Any],
    fault_descriptor: int,
    self_fault_descriptor: int,
) -> None:
    artifacts = attempt["artifacts"]
    offer_path = Path(str(artifacts["offer"]))
    accepted_path = Path(str(artifacts["accepted"]))
    commit_path = Path(str(artifacts["commit"]))
    active_path = Path(str(artifacts["active"]))
    reader_release_path = Path(
        str(artifacts["reader_release"])
    )
    release_observed_path = Path(
        str(artifacts["release_observed"])
    )
    while not offer_path.is_file():
        _require_empty_pane_fault_consumer_channels(
            attempt=attempt,
            fault_descriptor=fault_descriptor,
            self_fault_descriptor=self_fault_descriptor,
        )
        if (
            _tmux_owner_seal(
                str(attempt["consumer_session"]),
                str(attempt["consumer_owner_nonce"]),
            )
            != owner_seal
        ):
            raise RuntimeError(
                "pane fault consumer owner changed before offer"
            )
        time.sleep(0.02)
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=fault_descriptor,
        self_fault_descriptor=self_fault_descriptor,
    )
    offer = _load_json(offer_path, "pane fault consumer offer")
    expected_offer = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_offer_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": _json_binding(
            attempt_path, "consumer_attempt_sha256"
        ),
        "consumer_ready": _json_binding(
            Path(str(artifacts["ready"])),
            "consumer_ready_sha256",
        ),
        "consumer_started": _json_binding(
            Path(str(artifacts["started"])),
            "consumer_started_sha256",
        ),
        "launch_receipt": attempt["launch_receipt"],
        "launch_receipt_identity": (
            attempt["launch_receipt_identity"]
        ),
        "gate_owner_seal": attempt["gate_owner_seal"],
        "pane_fault_channel": attempt["pane_fault_channel"],
        "consumer_self_fault_channel": (
            attempt["consumer_self_fault_channel"]
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
    }
    if (
        set(offer)
        != set(expected_offer)
        | {"offered_at", "consumer_offer_sha256"}
        or any(offer.get(key) != value for key, value in expected_offer.items())
        or offer.get("consumer_offer_sha256")
        != _canonical_digest(offer, "consumer_offer_sha256")
    ):
        raise RuntimeError(
            "pane fault consumer transfer offer differs"
        )
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=fault_descriptor,
        self_fault_descriptor=self_fault_descriptor,
    )
    accepted = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_accepted_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": expected_offer["consumer_attempt"],
        "consumer_ready": expected_offer["consumer_ready"],
        "consumer_started": expected_offer["consumer_started"],
        "consumer_offer": _json_binding(
            offer_path, "consumer_offer_sha256"
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
        "owner_seal": dict(owner_seal),
        "supervisor_process": dict(
            ready["supervisor_process"]
        ),
        "worker_process": dict(process),
        "pane_fault_channel": attempt["pane_fault_channel"],
        "consumer_self_fault_channel": (
            attempt["consumer_self_fault_channel"]
        ),
        "accepted_at": _utc_now(),
    }
    accepted["consumer_accepted_sha256"] = _canonical_digest(
        accepted, "consumer_accepted_sha256"
    )
    _write_exclusive(accepted_path, accepted)
    while not commit_path.is_file():
        _require_empty_pane_fault_consumer_channels(
            attempt=attempt,
            fault_descriptor=fault_descriptor,
            self_fault_descriptor=self_fault_descriptor,
        )
        if (
            _tmux_owner_seal(
                str(attempt["consumer_session"]),
                str(attempt["consumer_owner_nonce"]),
            )
            != owner_seal
        ):
            raise RuntimeError(
                "pane fault consumer owner changed before commit"
            )
        time.sleep(0.02)
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=fault_descriptor,
        self_fault_descriptor=self_fault_descriptor,
    )
    commit = _load_json(
        commit_path, "pane fault consumer transfer commit"
    )
    expected_commit = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_commit_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": expected_offer["consumer_attempt"],
        "consumer_offer": accepted["consumer_offer"],
        "consumer_accepted": _json_binding(
            accepted_path, "consumer_accepted_sha256"
        ),
        "gate_owner_seal": attempt["gate_owner_seal"],
        "consumer_owner_seal": dict(owner_seal),
    }
    if (
        set(commit)
        != set(expected_commit)
        | {"committed_at", "consumer_commit_sha256"}
        or any(
            commit.get(key) != value
            for key, value in expected_commit.items()
        )
        or commit.get("consumer_commit_sha256")
        != _canonical_digest(commit, "consumer_commit_sha256")
    ):
        raise RuntimeError(
            "pane fault consumer transfer commit differs"
        )
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=fault_descriptor,
        self_fault_descriptor=self_fault_descriptor,
    )
    active = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_active_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": expected_offer["consumer_attempt"],
        "consumer_accepted": expected_commit["consumer_accepted"],
        "consumer_commit": _json_binding(
            commit_path, "consumer_commit_sha256"
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
        "owner_seal": dict(owner_seal),
        "supervisor_process": dict(
            ready["supervisor_process"]
        ),
        "worker_process": dict(process),
        "pane_fault_channel": attempt["pane_fault_channel"],
        "consumer_self_fault_channel": (
            attempt["consumer_self_fault_channel"]
        ),
        "active_at": _utc_now(),
    }
    active["consumer_active_sha256"] = _canonical_digest(
        active, "consumer_active_sha256"
    )
    _write_exclusive(active_path, active)
    while not reader_release_path.is_file():
        _require_empty_pane_fault_consumer_channels(
            attempt=attempt,
            fault_descriptor=fault_descriptor,
            self_fault_descriptor=self_fault_descriptor,
        )
        time.sleep(0.02)
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=fault_descriptor,
        self_fault_descriptor=self_fault_descriptor,
    )
    reader_release = _load_json(
        reader_release_path,
        "pane fault consumer launcher reader release",
    )
    expected_release = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_reader_release_intent_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": expected_offer["consumer_attempt"],
        "consumer_commit": active["consumer_commit"],
        "consumer_active": _json_binding(
            active_path, "consumer_active_sha256"
        ),
        "launcher_gate_reader_release_intent": True,
        "last_empty_snapshots": {
            "pane_gate": {
                "state": "empty",
                "record": None,
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
            "consumer_self": {
                "state": "empty",
                "record": None,
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
        },
    }
    if (
        set(reader_release)
        != set(expected_release)
        | {"released_at", "consumer_reader_release_sha256"}
        or any(
            reader_release.get(key) != value
            for key, value in expected_release.items()
        )
        or reader_release.get("consumer_reader_release_sha256")
        != _canonical_digest(
            reader_release, "consumer_reader_release_sha256"
        )
    ):
        raise RuntimeError(
            "pane fault consumer reader release differs"
        )
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=fault_descriptor,
        self_fault_descriptor=self_fault_descriptor,
    )
    observed = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_release_observed_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": expected_offer["consumer_attempt"],
        "consumer_active": expected_release["consumer_active"],
        "consumer_reader_release": _json_binding(
            reader_release_path,
            "consumer_reader_release_sha256",
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
        "owner_seal": dict(owner_seal),
        "supervisor_process": dict(
            ready["supervisor_process"]
        ),
        "worker_process": dict(process),
        "release_observed_at": _utc_now(),
    }
    observed[
        "consumer_release_observed_sha256"
    ] = _canonical_digest(
        observed, "consumer_release_observed_sha256"
    )
    _write_exclusive(release_observed_path, observed)


def _cleanup_failed_pane_fault_consumer(
    session: str,
    owner_nonce: str,
) -> None:
    if _tmux_pane(session) is None:
        return
    owner_seal = _tmux_owner_seal(session, owner_nonce)
    _kill_exact_session(session, owner_nonce, owner_seal)


def _validate_pane_owner_lifecycle_transition(
    live_owner: Mapping[str, Any],
    dead_owner: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    keys = {
        "session",
        "pane",
        "pane_pid",
        "pane_dead",
        "pane_dead_status",
        "pane_process",
        "owner_nonce",
        "tmux_server",
    }
    live_process = live_owner.get("pane_process")
    if (
        set(live_owner) != keys
        or set(dead_owner) != keys
        or live_owner.get("pane_dead") is not False
        or live_owner.get("pane_dead_status") is not None
        or not isinstance(live_process, Mapping)
        or live_process.get("pid") != live_owner.get("pane_pid")
        or type(live_process.get("start_ticks")) is not int
        or int(live_process["start_ticks"]) <= 0
        or dead_owner.get("pane_dead") is not True
        or (
            dead_owner.get("pane_dead_status") is not None
            and type(dead_owner.get("pane_dead_status")) is not int
        )
        or dead_owner.get("pane_process") is not None
        or any(
            dead_owner.get(key) != live_owner.get(key)
            for key in (
                "session",
                "pane",
                "pane_pid",
                "owner_nonce",
                "tmux_server",
            )
        )
    ):
        raise RuntimeError(f"{label} lifecycle transition differs")
    return dict(dead_owner)


def _poison_and_cleanup_pane_fault_consumer(
    consumer: Mapping[str, Any],
    failure: BaseException,
) -> PaneFaultConsumerReservationError:
    poison = (
        failure
        if isinstance(failure, PaneFaultConsumerReservationError)
        else PaneFaultConsumerReservationError(failure)
    )
    try:
        os.close(int(consumer["self_fault_reader_descriptor"]))
    except BaseException as close_exc:
        poison.add_secondary_failure(
            stage="close_consumer_self_fault_reader",
            failure=close_exc,
        )
    try:
        _cleanup_failed_pane_fault_consumer(
            str(consumer["session"]),
            str(consumer["owner_nonce"]),
        )
    except BaseException as cleanup_exc:
        poison.add_secondary_failure(
            stage="exact_consumer_tmux_cleanup",
            failure=cleanup_exc,
        )
    return poison


def _spawn_ready_pane_fault_consumer(
    *,
    repo_root: Path,
    consumer_session: str,
    consumer_owner_nonce: str,
    consumer_worker_arguments: Sequence[str],
    consumer_supervisor_arguments: Sequence[str],
    consumer_tmux_arguments: Sequence[str],
    attempt_path: Path,
    policy_sha256: str,
    attempt_id: str,
    artifacts: Mapping[str, str],
    pane_fault_channel: Mapping[str, Any],
    self_fault_channel: Mapping[str, Any],
    ready_timeout_seconds: float,
) -> dict[str, Any]:
    del repo_root
    client_result = subprocess.run(
        list(consumer_tmux_arguments),
        capture_output=True,
        text=True,
    )
    client = {
        "returncode": client_result.returncode,
        "stdout": client_result.stdout,
        "stderr": client_result.stderr,
    }
    if client_result.returncode != 0:
        raise RuntimeError(
            "pane fault consumer tmux spawn failed: "
            f"{client_result.stderr.strip()}"
        )
    attempt = _load_json(
        attempt_path, "spawned consumer attempt"
    )
    supervisor_ready_path = Path(
        str(
            attempt[
                "consumer_lifecycle_wait_supervisor_ready_path"
            ]
        )
    )
    ready_path = Path(artifacts["ready"])
    deadline = time.monotonic() + ready_timeout_seconds
    while not (
        supervisor_ready_path.is_file() and ready_path.is_file()
    ):
        pane = _tmux_pane(consumer_session)
        if pane is None or pane["pane_dead"]:
            raise RuntimeError(
                "pane fault consumer exited before ready"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "pane fault consumer ready timed out"
            )
        time.sleep(0.02)
    ready = _load_json(
        ready_path, "pane fault consumer ready"
    )
    owner_seal = _tmux_owner_seal(
        consumer_session, consumer_owner_nonce
    )
    supervisor_ready = _validate_consumer_wait_supervisor_ready(
        supervisor_ready_path,
        attempt_path=attempt_path,
        attempt_identity=_opened_file_identity(attempt_path),
        attempt=attempt,
    )
    supervisor_process = _process_identity(
        int(owner_seal["pane_pid"])
    )
    worker_process = supervisor_ready[
        "consumer_worker_process"
    ]
    fd_stat = os.stat(
        f"/proc/{worker_process['pid']}/fd/"
        f"{int(ready['fault_descriptor'])}"
    )
    self_fd_stat = os.stat(
        f"/proc/{worker_process['pid']}/fd/"
        f"{int(ready['self_fault_descriptor'])}"
    )
    if (
        ready.get("consumer_ready_sha256")
        != _canonical_digest(ready, "consumer_ready_sha256")
        or ready.get("consumer_attempt")
        != _json_binding(
            attempt_path, "consumer_attempt_sha256"
        )
        or ready.get("consumer_session") != consumer_session
        or ready.get("consumer_owner_nonce")
        != consumer_owner_nonce
        or ready.get("tmux_server")
        != owner_seal["tmux_server"]
        or owner_seal != supervisor_ready["owner_seal"]
        or owner_seal["pane_process"] != supervisor_process
        or supervisor_ready["supervisor_arguments"]
        != list(consumer_supervisor_arguments)
        or ready.get("supervisor_command")
        != list(consumer_supervisor_arguments)
        or _process_command_bytes(supervisor_process["pid"])
        != _command_bytes(consumer_supervisor_arguments)
        or ready.get("supervisor_executable")
        != _process_executable(supervisor_process["pid"])
        or ready.get("supervisor_owner_seal") != owner_seal
        or ready.get("supervisor_process") != supervisor_process
        or ready.get("consumer_wait_supervisor_ready")
        != _json_binding(
            supervisor_ready_path,
            "consumer_wait_supervisor_ready_sha256",
        )
        or ready.get("worker_process") != worker_process
        or _process_identity(worker_process["pid"])
        != worker_process
        or ready.get("worker_command")
        != list(consumer_worker_arguments)
        or _process_command_bytes(worker_process["pid"])
        != _command_bytes(consumer_worker_arguments)
        or ready.get("worker_executable")
        != _process_executable(worker_process["pid"])
        or int(fd_stat.st_dev) != pane_fault_channel["device"]
        or int(fd_stat.st_ino) != pane_fault_channel["inode"]
        or ready.get("pane_fault_channel")
        != dict(pane_fault_channel)
        or int(self_fd_stat.st_dev)
        != self_fault_channel["device"]
        or int(self_fd_stat.st_ino)
        != self_fault_channel["inode"]
        or ready.get("consumer_self_fault_channel")
        != dict(self_fault_channel)
    ):
        raise RuntimeError(
            "pane fault consumer ready live seal differs"
        )
    _set_remain_on_exit(str(owner_seal["pane"]), True)
    _verify_remain_on_exit(str(owner_seal["pane"]), "on")
    started = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_started_v1"
        ),
        "policy_sha256": policy_sha256,
        "attempt_id": attempt_id,
        "consumer_attempt": _json_binding(
            attempt_path, "consumer_attempt_sha256"
        ),
        "consumer_ready": _json_binding(
            ready_path, "consumer_ready_sha256"
        ),
        "consumer_wait_supervisor_ready": _json_binding(
            supervisor_ready_path,
            "consumer_wait_supervisor_ready_sha256",
        ),
        "owner_seal": owner_seal,
        "supervisor_process": supervisor_process,
        "worker_process": worker_process,
        "pane_fault_channel": dict(pane_fault_channel),
        "consumer_self_fault_channel": dict(
            self_fault_channel
        ),
        "remain_on_exit": "on",
        "started_at": _utc_now(),
    }
    started["consumer_started_sha256"] = _canonical_digest(
        started, "consumer_started_sha256"
    )
    started_path = Path(artifacts["started"])
    _write_exclusive(started_path, started)
    return {
        "ready": ready,
        "ready_path": ready_path,
        "supervisor_ready": supervisor_ready,
        "supervisor_ready_path": supervisor_ready_path,
        "started": started,
        "started_path": started_path,
        "client": client,
        "owner_seal": owner_seal,
    }


def _wait_for_pane_fault_consumer_artifact(
    *,
    path: Path,
    label: str,
    consumer: Mapping[str, Any],
    launcher_gate_reader: Mapping[str, Any],
    deadline: float,
) -> dict[str, Any]:
    attempt = consumer["attempt"]
    while not path.is_file():
        _require_empty_pane_fault_consumer_channels(
            attempt=attempt,
            fault_descriptor=int(
                launcher_gate_reader["descriptor"]
            ),
            self_fault_descriptor=int(
                consumer["self_fault_reader_descriptor"]
            ),
        )
        _require_live_pane_fault_consumer(consumer, label)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"pane fault consumer {label} timed out"
            )
        time.sleep(0.02)
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=int(
            launcher_gate_reader["descriptor"]
        ),
        self_fault_descriptor=int(
            consumer["self_fault_reader_descriptor"]
        ),
    )
    _require_live_pane_fault_consumer(consumer, label)
    return _load_json(path, f"pane fault consumer {label}")


def _require_live_pane_fault_consumer(
    consumer: Mapping[str, Any],
    label: str,
) -> None:
    observed = _tmux_owner_seal(
        str(consumer["session"]),
        str(consumer["owner_nonce"]),
    )
    if observed != consumer["owner_seal"]:
        raise RuntimeError(
            f"pane fault consumer changed before {label}"
        )
    worker = consumer["ready"]["worker_process"]
    if (
        _process_identity(int(worker["pid"])) != worker
        or worker["ppid"] != observed["pane_pid"]
        or _process_command_bytes(int(worker["pid"]))
        != _command_bytes(
            consumer["attempt"]["consumer_worker_arguments"]
        )
    ):
        raise RuntimeError(
            f"pane fault consumer worker changed before {label}"
        )


def _require_post_handoff_pane_fault_consumer(
    consumer: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    attempt = consumer["attempt"]
    descriptor = int(consumer["self_fault_reader_descriptor"])
    binding = attempt["consumer_self_fault_channel"]
    try:
        snapshot = _read_fault_channel(
            descriptor,
            binding,
            attempt_id=str(attempt["attempt_id"]),
            owner_nonce=str(attempt["consumer_owner_nonce"]),
            launch_receipt_sha256=str(
                attempt["launch_receipt"]["canonical_sha256"]
            ),
            publisher=attempt["consumer_self_fault_publisher"],
        )
    except BaseException as exc:
        raise LauncherGateFaultError(
            "consumer_self_fault_channel_invalid",
            snapshot=None,
            failure=exc,
        ) from exc
    if snapshot.get("state") == "valid_fault":
        raise LauncherGateFaultError(
            "consumer_self_typed_publish_failure",
            snapshot=snapshot,
        )
    if snapshot.get("state") != "empty":
        raise LauncherGateFaultError(
            "consumer_self_fault_channel_nonempty",
            snapshot=snapshot,
        )
    _require_live_pane_fault_consumer(consumer, label)
    return snapshot


def _transfer_pane_fault_consumer(
    *,
    consumer: Mapping[str, Any],
    launcher_gate_reader: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    attempt = consumer["attempt"]
    artifacts = attempt["artifacts"]
    attempt_path = Path(str(consumer["attempt_path"]))
    ready_path = Path(str(consumer["ready_path"]))
    started_path = Path(str(consumer["started_path"]))
    offer_path = Path(str(artifacts["offer"]))
    accepted_path = Path(str(artifacts["accepted"]))
    commit_path = Path(str(artifacts["commit"]))
    active_path = Path(str(artifacts["active"]))
    reader_release_path = Path(
        str(artifacts["reader_release"])
    )
    release_observed_path = Path(
        str(artifacts["release_observed"])
    )
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=int(
            launcher_gate_reader["descriptor"]
        ),
        self_fault_descriptor=int(
            consumer["self_fault_reader_descriptor"]
        ),
    )
    _require_live_pane_fault_consumer(consumer, "transfer offer")
    offer = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_offer_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": _json_binding(
            attempt_path, "consumer_attempt_sha256"
        ),
        "consumer_ready": _json_binding(
            ready_path, "consumer_ready_sha256"
        ),
        "consumer_started": _json_binding(
            started_path, "consumer_started_sha256"
        ),
        "launch_receipt": attempt["launch_receipt"],
        "launch_receipt_identity": (
            attempt["launch_receipt_identity"]
        ),
        "gate_owner_seal": attempt["gate_owner_seal"],
        "pane_fault_channel": attempt["pane_fault_channel"],
        "consumer_self_fault_channel": (
            attempt["consumer_self_fault_channel"]
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
        "offered_at": _utc_now(),
    }
    offer["consumer_offer_sha256"] = _canonical_digest(
        offer, "consumer_offer_sha256"
    )
    _write_exclusive(offer_path, offer)
    accepted = _wait_for_pane_fault_consumer_artifact(
        path=accepted_path,
        label="transfer accepted",
        consumer=consumer,
        launcher_gate_reader=launcher_gate_reader,
        deadline=time.monotonic() + timeout_seconds,
    )
    expected_accepted = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_accepted_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": offer["consumer_attempt"],
        "consumer_ready": offer["consumer_ready"],
        "consumer_started": offer["consumer_started"],
        "consumer_offer": _json_binding(
            offer_path, "consumer_offer_sha256"
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
        "owner_seal": consumer["owner_seal"],
        "supervisor_process": consumer["ready"][
            "supervisor_process"
        ],
        "worker_process": consumer["ready"]["worker_process"],
        "pane_fault_channel": attempt["pane_fault_channel"],
        "consumer_self_fault_channel": (
            attempt["consumer_self_fault_channel"]
        ),
    }
    if (
        set(accepted)
        != set(expected_accepted)
        | {"accepted_at", "consumer_accepted_sha256"}
        or any(
            accepted.get(key) != value
            for key, value in expected_accepted.items()
        )
        or accepted.get("consumer_accepted_sha256")
        != _canonical_digest(
            accepted, "consumer_accepted_sha256"
        )
    ):
        raise RuntimeError(
            "pane fault consumer accepted shadow differs"
        )
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=int(
            launcher_gate_reader["descriptor"]
        ),
        self_fault_descriptor=int(
            consumer["self_fault_reader_descriptor"]
        ),
    )
    _require_live_pane_fault_consumer(consumer, "transfer commit")
    commit = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_commit_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": offer["consumer_attempt"],
        "consumer_offer": expected_accepted["consumer_offer"],
        "consumer_accepted": _json_binding(
            accepted_path, "consumer_accepted_sha256"
        ),
        "gate_owner_seal": attempt["gate_owner_seal"],
        "consumer_owner_seal": consumer["owner_seal"],
        "committed_at": _utc_now(),
    }
    commit["consumer_commit_sha256"] = _canonical_digest(
        commit, "consumer_commit_sha256"
    )
    _write_exclusive(commit_path, commit)
    active = _wait_for_pane_fault_consumer_artifact(
        path=active_path,
        label="transfer active",
        consumer=consumer,
        launcher_gate_reader=launcher_gate_reader,
        deadline=time.monotonic() + timeout_seconds,
    )
    expected_active = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_transfer_active_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": offer["consumer_attempt"],
        "consumer_accepted": commit["consumer_accepted"],
        "consumer_commit": _json_binding(
            commit_path, "consumer_commit_sha256"
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
        "owner_seal": consumer["owner_seal"],
        "supervisor_process": consumer["ready"][
            "supervisor_process"
        ],
        "worker_process": consumer["ready"]["worker_process"],
        "pane_fault_channel": attempt["pane_fault_channel"],
        "consumer_self_fault_channel": (
            attempt["consumer_self_fault_channel"]
        ),
    }
    if (
        set(active)
        != set(expected_active)
        | {"active_at", "consumer_active_sha256"}
        or any(
            active.get(key) != value
            for key, value in expected_active.items()
        )
        or active.get("consumer_active_sha256")
        != _canonical_digest(active, "consumer_active_sha256")
    ):
        raise RuntimeError(
            "pane fault consumer active authority differs"
        )
    last_empty_snapshots = (
        _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=int(
            launcher_gate_reader["descriptor"]
        ),
        self_fault_descriptor=int(
            consumer["self_fault_reader_descriptor"]
        ),
        )
    )
    _require_live_pane_fault_consumer(
        consumer, "reader release intent"
    )
    reader_release = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_reader_release_intent_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": offer["consumer_attempt"],
        "consumer_commit": expected_active["consumer_commit"],
        "consumer_active": _json_binding(
            active_path, "consumer_active_sha256"
        ),
        "launcher_gate_reader_release_intent": True,
        "last_empty_snapshots": last_empty_snapshots,
        "released_at": _utc_now(),
    }
    reader_release[
        "consumer_reader_release_sha256"
    ] = _canonical_digest(
        reader_release, "consumer_reader_release_sha256"
    )
    _write_exclusive(reader_release_path, reader_release)
    release_observed = _wait_for_pane_fault_consumer_artifact(
        path=release_observed_path,
        label="reader release observed",
        consumer=consumer,
        launcher_gate_reader=launcher_gate_reader,
        deadline=time.monotonic() + timeout_seconds,
    )
    expected_observed = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_release_observed_v1"
        ),
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": offer["consumer_attempt"],
        "consumer_active": reader_release["consumer_active"],
        "consumer_reader_release": _json_binding(
            reader_release_path,
            "consumer_reader_release_sha256",
        ),
        "consumer_session": attempt["consumer_session"],
        "consumer_owner_nonce": (
            attempt["consumer_owner_nonce"]
        ),
        "owner_seal": consumer["owner_seal"],
        "supervisor_process": consumer["ready"][
            "supervisor_process"
        ],
        "worker_process": consumer["ready"]["worker_process"],
    }
    if (
        set(release_observed)
        != set(expected_observed)
        | {
            "release_observed_at",
            "consumer_release_observed_sha256",
        }
        or any(
            release_observed.get(key) != value
            for key, value in expected_observed.items()
        )
        or release_observed.get(
            "consumer_release_observed_sha256"
        )
        != _canonical_digest(
            release_observed,
            "consumer_release_observed_sha256",
        )
    ):
        raise RuntimeError(
            "pane fault consumer release observation differs"
        )
    _require_empty_pane_fault_consumer_channels(
        attempt=attempt,
        fault_descriptor=int(
            launcher_gate_reader["descriptor"]
        ),
        self_fault_descriptor=int(
            consumer["self_fault_reader_descriptor"]
        ),
    )
    _require_live_pane_fault_consumer(
        consumer, "launcher gate reader close"
    )
    try:
        os.close(int(launcher_gate_reader["descriptor"]))
    except BaseException as close_exc:
        failure = PaneFaultConsumerReservationError(
            RuntimeError(
                "launcher gate reader close failed after transfer ACK"
            )
        )
        failure.add_secondary_failure(
            stage="close_launcher_gate_reader",
            failure=close_exc,
        )
        raise failure from close_exc
    launcher_gate_reader["closed"] = True
    return {
        "offer": offer,
        "offer_path": offer_path,
        "accepted": accepted,
        "accepted_path": accepted_path,
        "commit": commit,
        "commit_path": commit_path,
        "active": active,
        "active_path": active_path,
        "reader_release": reader_release,
        "reader_release_path": reader_release_path,
        "release_observed": release_observed,
        "release_observed_path": release_observed_path,
    }


def _reserve_spawn_ready_pane_fault_consumer(
    *,
    repo_root: Path,
    config: Path,
    attempt_root: Path,
    policy_sha256: str,
    attempt_id: str,
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    gate_owner_seal: Mapping[str, Any],
    pane_fault_channel: Mapping[str, Any],
    pane_fault_publisher: Mapping[str, str],
    python: str,
    ready_timeout_seconds: float,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    registered = validate_pane_fault_consumer_registration(
        registration,
        expected_namespace=str(
            attempt_root / "pane_fault_consumer"
        ),
        label="launcher pane fault consumer registration",
    )
    namespace = _ensure_secure_leaf_directories(
        attempt_root,
        ("pane_fault_consumer",),
    )
    if str(namespace) != registered["namespace"]:
        raise RuntimeError(
            "pane fault consumer namespace registration differs"
        )
    self_fault_channel = _create_fault_channel(
        namespace / "consumer_self_fault.channel"
    )
    artifacts = {
        name: path
        for name, path in registered["artifacts"].items()
        if name
        not in {"attempt", "log", "self_fault_channel"}
    }
    attempt_path = Path(registered["artifacts"]["attempt"])
    log_path = Path(registered["artifacts"]["log"])
    if (
        str(namespace / "consumer_self_fault.channel")
        != registered["artifacts"]["self_fault_channel"]
        or self_fault_channel["path"]
        != registered["artifacts"]["self_fault_channel"]
    ):
        raise RuntimeError(
            "pane fault consumer channel registration differs"
        )
    log_binding = _create_secure_append_log(log_path)
    receipt = _load_json(
        receipt_path, "consumer reservation launch receipt"
    )
    consumer_session = str(receipt["consumer_session"])
    consumer_owner_nonce = str(receipt["consumer_owner_nonce"])
    consumer_worker_arguments = list(
        receipt["consumer_worker_arguments"]
    )
    consumer_supervisor_arguments = list(
        receipt["consumer_lifecycle_wait_supervisor_arguments"]
    )
    consumer_tmux_arguments = list(
        receipt["consumer_tmux_arguments"]
    )
    consumer_lifecycle_wait_channel = dict(
        receipt["consumer_lifecycle_wait_channel"]
    )
    if (
        consumer_session
        != PANE_FAULT_CONSUMER_SESSION_PREFIX + attempt_id
        or consumer_worker_arguments
        != [
            python,
            "-B",
            "-u",
            str(Path(__file__).resolve()),
            PANE_FAULT_CONSUMER_MODE,
            "--attempt-path",
            str(attempt_path),
            "--config",
            str(config),
        ]
        or consumer_lifecycle_wait_channel["path"]
        != registered["artifacts"]["lifecycle_wait_channel"]
    ):
        raise RuntimeError(
            "consumer reservation receipt authority differs"
        )
    value = {
        "schema_version": 1,
        "contract_type": (
            "safa_pane_fault_consumer_attempt_v1"
        ),
        "policy_sha256": policy_sha256,
        "attempt_id": attempt_id,
        "launch_receipt": _json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        "launch_receipt_identity": dict(receipt_identity),
        "gate_owner_seal": dict(gate_owner_seal),
        "pane_fault_channel": dict(pane_fault_channel),
        "pane_fault_publisher": dict(pane_fault_publisher),
        "consumer_self_fault_channel": self_fault_channel,
        "consumer_self_fault_publisher": {
            **dict(registered["publishers"]["consumer"]),
        },
        "consumer_session": consumer_session,
        "consumer_owner_nonce": consumer_owner_nonce,
        "consumer_worker_arguments": consumer_worker_arguments,
        "consumer_lifecycle_wait_channel": (
            consumer_lifecycle_wait_channel
        ),
        "consumer_lifecycle_wait_publisher": dict(
            receipt["consumer_lifecycle_wait_publisher"]
        ),
        "consumer_lifecycle_wait_supervisor_arguments": (
            consumer_supervisor_arguments
        ),
        "consumer_lifecycle_wait_supervisor_ready_path": str(
            receipt[
                "consumer_lifecycle_wait_supervisor_ready_path"
            ]
        ),
        "consumer_lifecycle_wait_status_path": str(
            receipt["consumer_lifecycle_wait_status_path"]
        ),
        "consumer_tmux_arguments": consumer_tmux_arguments,
        "consumer_log": log_binding,
        "artifacts": artifacts,
        "reserved_at": _utc_now(),
    }
    value["consumer_attempt_sha256"] = _canonical_digest(
        value, "consumer_attempt_sha256"
    )
    _write_exclusive(attempt_path, value)
    self_fault_reader_descriptor = (
        _open_presealed_fault_channel(
            namespace,
            self_fault_channel,
            name="consumer_self_fault.channel",
        )
    )
    try:
        spawned = _spawn_ready_pane_fault_consumer(
            repo_root=repo_root,
            consumer_session=consumer_session,
            consumer_owner_nonce=consumer_owner_nonce,
            consumer_worker_arguments=consumer_worker_arguments,
            consumer_supervisor_arguments=(
                consumer_supervisor_arguments
            ),
            consumer_tmux_arguments=consumer_tmux_arguments,
            attempt_path=attempt_path,
            policy_sha256=policy_sha256,
            attempt_id=attempt_id,
            artifacts=artifacts,
            pane_fault_channel=pane_fault_channel,
            self_fault_channel=self_fault_channel,
            ready_timeout_seconds=ready_timeout_seconds,
        )
    except BaseException as exc:
        failure = PaneFaultConsumerReservationError(exc)
        try:
            os.close(self_fault_reader_descriptor)
        except BaseException as close_exc:
            failure.add_secondary_failure(
                stage="close_self_fault_reader",
                failure=close_exc,
            )
        try:
            _cleanup_failed_pane_fault_consumer(
                consumer_session, consumer_owner_nonce
            )
        except BaseException as cleanup_exc:
            failure.add_secondary_failure(
                stage="exact_consumer_tmux_cleanup",
                failure=cleanup_exc,
            )
        raise failure from exc
    return {
        "namespace": namespace,
        "attempt": value,
        "attempt_path": attempt_path,
        **spawned,
        "session": consumer_session,
        "owner_nonce": consumer_owner_nonce,
        "self_fault_channel": self_fault_channel,
        "self_fault_reader_descriptor": (
            self_fault_reader_descriptor
        ),
    }


def _build_pane_fault_consumer_registration(
    *,
    attempt_root: Path,
    launcher_binding: Mapping[str, str],
) -> dict[str, Any]:
    namespace = attempt_root / "pane_fault_consumer"
    artifacts = {
        name: str(namespace / f"consumer_{name}.json")
        for name in (
            "ready",
            "started",
            "offer",
            "accepted",
            "commit",
            "active",
            "reader_release",
            "release_observed",
            "controller_cleanup",
            "terminal",
            "join",
            "cleanup",
            "wait_supervisor_ready",
        )
    }
    artifacts.update(
        {
            "attempt": str(namespace / "consumer_attempt.json"),
            "log": str(namespace / "consumer.log"),
            "self_fault_channel": str(
                namespace / "consumer_self_fault.channel"
            ),
            "lifecycle_wait_channel": str(
                namespace / "consumer_lifecycle_wait.channel"
            ),
        }
    )
    registration = build_pane_fault_consumer_registration(
        namespace=str(namespace),
        artifacts=artifacts,
        publishers={
            "launcher": {
                **dict(launcher_binding),
                "role": "launcher_pane_fault_consumer_handoff",
            },
            "consumer": {
                **dict(launcher_binding),
                "role": "pane_fault_consumer",
            },
        },
    )
    return validate_pane_fault_consumer_registration(
        registration,
        expected_namespace=str(namespace),
        label="launch receipt pane fault consumer registration",
    )


def join_pane_fault_consumer(
    *,
    attempt_path: Path,
    config: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    _install_verified_preflight_apis(config.resolve())
    attempt_path = attempt_path.resolve()
    attempt = _load_json(
        attempt_path, "pane fault consumer join attempt"
    )
    _validate_consumer_attempt_evidence(
        attempt,
        expected_attempt_id=str(attempt["attempt_id"]),
        expected_receipt=attempt["launch_receipt"],
        expected_receipt_identity=attempt["launch_receipt_identity"],
        expected_gate_owner_seal=attempt["gate_owner_seal"],
        error_message=(
            "pane fault consumer join attempt contract differs"
        ),
    )
    artifacts = attempt["artifacts"]
    terminal_path = Path(str(artifacts["terminal"]))
    join_path = Path(str(artifacts["join"]))
    cleanup_path = Path(str(artifacts["cleanup"]))
    deadline = time.monotonic() + timeout_seconds
    session = str(attempt["consumer_session"])
    pane: dict[str, Any] | None = None
    preexisting_join: dict[str, Any] | None = None
    if join_path.is_file():
        preexisting_join = _load_json(
            join_path, "preexisting pane fault consumer join"
        )
        retired = preexisting_join.get("retired_pane")
        if not isinstance(retired, dict):
            raise RuntimeError(
                "preexisting pane fault consumer join lacks retired pane"
            )
        pane = retired
    else:
        while True:
            pane = _tmux_pane(session)
            if (
                pane is not None
                and pane["pane_dead"]
                and terminal_path.is_file()
            ):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "pane fault consumer formal lifecycle join timed out"
                )
            time.sleep(0.02)
    formal_consumer = _read_formal_consumer_lifecycle_status(
        attempt_path=attempt_path,
        pane=pane,
    )
    terminal = formal_consumer["terminal"]
    receipt_path = Path(str(attempt["launch_receipt"]["path"]))
    receipt = _load_json(
        receipt_path, "pane fault consumer join receipt"
    )
    controller_cleanup_path = Path(
        str(artifacts["controller_cleanup"])
    )
    controller_cleanup = _load_json(
        controller_cleanup_path,
        "pane fault consumer controller cleanup",
    )
    dead_controller_owner = controller_cleanup.get(
        "dead_owner_seal"
    )
    formal_gate = _read_formal_gate_lifecycle_status(
        attempt_root=receipt_path.parent,
        pane=dead_controller_owner,
    )
    gate_adjudication = formal_gate["adjudication"]
    try:
        validated_dead_controller_owner = (
            _validate_pane_owner_lifecycle_transition(
                attempt["gate_owner_seal"],
                dead_controller_owner,
                label="joined controller pane owner",
            )
        )
    except (KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "pane fault consumer controller cleanup differs"
        ) from exc
    adjudicated_outcome = gate_adjudication[
        "adjudicated_outcome"
    ]
    expected_empty = {
        "state": "empty",
        "record": None,
        "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    expected_chain = build_pane_fault_consumer_chain(
        consumer_started=_json_binding(
            Path(str(artifacts["started"])),
            "consumer_started_sha256",
        ),
        consumer_active=_json_binding(
            Path(str(artifacts["active"])),
            "consumer_active_sha256",
        ),
        consumer_reader_release=_json_binding(
            Path(str(artifacts["reader_release"])),
            "consumer_reader_release_sha256",
        ),
        consumer_release_observed=_json_binding(
            Path(str(artifacts["release_observed"])),
            "consumer_release_observed_sha256",
        ),
        registration=receipt["pane_fault_consumer"],
    )
    expected_final_chain = {
        "gate_execution_terminal": _json_binding(
            receipt_path.with_name("gate_execution_terminal.json"),
            "gate_execution_terminal_sha256",
        ),
        **formal_gate["ownership_chain"],
    }
    gate_terminal_binding = expected_final_chain[
        "gate_execution_terminal"
    ]
    _validate_controller_cleanup_evidence(
        controller_cleanup,
        attempt=attempt,
        attempt_binding=_json_binding(
            attempt_path, "consumer_attempt_sha256"
        ),
        gate_terminal_binding=gate_terminal_binding,
        formal_gate=formal_gate,
        expected_consumer_chain=expected_chain,
        dead_gate_owner_seal=validated_dead_controller_owner,
        contract_error=(
            "pane fault consumer controller cleanup differs"
        ),
        chain_error=(
            "pane fault consumer controller cleanup chain differs"
        ),
        exit_error=(
            "pane fault consumer controller cleanup exit differs"
        ),
    )
    _validate_consumer_terminal_evidence(
        terminal,
        attempt=attempt,
        attempt_binding=_json_binding(
            attempt_path, "consumer_attempt_sha256"
        ),
        controller_cleanup_binding=_json_binding(
            controller_cleanup_path,
            "consumer_controller_cleanup_sha256",
        ),
        gate_terminal_binding=gate_terminal_binding,
        formal_gate=formal_gate,
        formal_consumer=formal_consumer,
        expected_consumer_chain=expected_chain,
        error_message=(
            "pane fault consumer terminal join contract differs"
        ),
    )
    self_descriptor = _open_presealed_fault_channel(
        Path(str(attempt["consumer_self_fault_channel"]["path"])).parent,
        attempt["consumer_self_fault_channel"],
        name=Path(
            str(attempt["consumer_self_fault_channel"]["path"])
        ).name,
    )
    try:
        snapshot = _read_fault_channel(
            self_descriptor,
            attempt["consumer_self_fault_channel"],
            attempt_id=str(attempt["attempt_id"]),
            owner_nonce=str(attempt["consumer_owner_nonce"]),
            launch_receipt_sha256=str(
                attempt["launch_receipt"]["canonical_sha256"]
            ),
            publisher=attempt["consumer_self_fault_publisher"],
        )
    finally:
        os.close(self_descriptor)
    if snapshot != expected_empty:
        raise RuntimeError(
            "pane fault consumer join self channel is not empty"
        )
    owner_seal = attempt["gate_owner_seal"]
    consumer_owner_seal = terminal["supervisor_owner_seal"]
    expected_attempt_binding = _json_binding(
        attempt_path, "consumer_attempt_sha256"
    )
    expected_terminal_binding = _json_binding(
        terminal_path, "consumer_terminal_sha256"
    )
    expected_supervisor_ready_binding = formal_consumer[
        "supervisor_ready_binding"
    ]
    expected_worker_ready_binding = formal_consumer[
        "worker_ready_binding"
    ]
    expected_lifecycle = formal_consumer["snapshot"]

    def validate_join(value: Mapping[str, Any]) -> dict[str, Any]:
        return _validate_consumer_join_evidence(
            value,
            attempt=attempt,
            attempt_binding=expected_attempt_binding,
            terminal_binding=expected_terminal_binding,
            supervisor_ready_binding=(
                expected_supervisor_ready_binding
            ),
            worker_ready_binding=expected_worker_ready_binding,
            formal_consumer=formal_consumer,
            dead_consumer_owner_seal={
                **dict(consumer_owner_seal),
                "pane_dead": True,
                "pane_dead_status": value.get("retired_pane", {}).get(
                    "pane_dead_status"
                ),
                "pane_process": None,
            },
            adjudicated_outcome=adjudicated_outcome,
            error_message=(
                "pane fault consumer join authorization differs"
            ),
        )

    def validate_cleanup(
        value: Mapping[str, Any],
        join_binding: Mapping[str, str],
    ) -> dict[str, Any]:
        return _validate_consumer_cleanup_evidence(
            value,
            attempt=attempt,
            attempt_binding=expected_attempt_binding,
            terminal_binding=expected_terminal_binding,
            join_binding=join_binding,
            formal_consumer=formal_consumer,
            adjudicated_outcome=adjudicated_outcome,
            error_message=(
                "pane fault consumer cleanup contract differs"
            ),
        )

    join: dict[str, Any] | None = None
    if preexisting_join is not None:
        join = validate_join(preexisting_join)
    if cleanup_path.is_file():
        if join is None:
            raise RuntimeError(
                "pane fault consumer cleanup exists without join"
            )
        cleanup = validate_cleanup(
            _load_json(
                cleanup_path, "pane fault consumer cleanup"
            ),
            _json_binding(join_path, "consumer_join_sha256"),
        )
        if _tmux_pane(session) is not None:
            raise RuntimeError(
                "completed pane fault consumer cleanup has residual"
            )
        return cleanup

    if join is None:
        if (
            pane is None
            or
            pane["session"] != session
            or pane["pane"] != consumer_owner_seal["pane"]
            or pane["pane_pid"] != consumer_owner_seal["pane_pid"]
            or _tmux_owner_nonce(session)
            != attempt["consumer_owner_nonce"]
        ):
            raise RuntimeError(
                "pane fault consumer dead owner seal differs"
            )
        join = {
            "schema_version": 1,
            "contract_type": "safa_pane_fault_consumer_join_v3",
            "policy_sha256": attempt["policy_sha256"],
            "attempt_id": attempt["attempt_id"],
            "consumer_attempt": expected_attempt_binding,
            "consumer_terminal": expected_terminal_binding,
            "consumer_lifecycle": expected_lifecycle,
            "consumer_wait_supervisor_ready": (
                expected_supervisor_ready_binding
            ),
            "consumer_worker_ready": (
                expected_worker_ready_binding
            ),
            "consumer_session": session,
            "consumer_owner_nonce": attempt[
                "consumer_owner_nonce"
            ],
            "retired_pane": pane,
            "adjudicated_outcome": adjudicated_outcome,
            "consumer_adjudicated_exit": (
                CONSUMER_ADJUDICATED_EXIT
            ),
            "status": "cleanup_authorized",
            "session_residual": True,
            "authorized_at": _utc_now(),
        }
        join["consumer_join_sha256"] = _canonical_digest(
            join, "consumer_join_sha256"
        )
        _write_exclusive(join_path, join)

    pane = _tmux_pane(session)
    if pane is not None:
        if (
            pane.get("session")
            != join["retired_pane"]["session"]
            or pane.get("pane") != join["retired_pane"]["pane"]
            or pane.get("pane_pid")
            != join["retired_pane"]["pane_pid"]
            or pane.get("pane_dead") is not True
            or _tmux_owner_nonce(session)
            != attempt["consumer_owner_nonce"]
        ):
            raise RuntimeError(
                "pane fault consumer join found foreign session"
            )
        current_owner_seal = _tmux_owner_seal(
            session, str(attempt["consumer_owner_nonce"])
        )
        if (
            current_owner_seal["session"] != session
            or current_owner_seal["pane"]
            != consumer_owner_seal["pane"]
            or current_owner_seal["pane_pid"]
            != consumer_owner_seal["pane_pid"]
            or current_owner_seal["tmux_server"]
            != consumer_owner_seal["tmux_server"]
            or not current_owner_seal["pane_dead"]
        ):
            raise RuntimeError(
                "pane fault consumer join owner authority differs"
            )
        _kill_exact_session(
            session,
            str(attempt["consumer_owner_nonce"]),
            current_owner_seal,
        )
    if _tmux_pane(session) is not None:
        raise RuntimeError(
            "pane fault consumer join left session residual"
        )
    cleanup = {
        "schema_version": 1,
        "contract_type": "safa_pane_fault_consumer_cleanup_v3",
        "policy_sha256": attempt["policy_sha256"],
        "attempt_id": attempt["attempt_id"],
        "consumer_attempt": _json_binding(
            attempt_path, "consumer_attempt_sha256"
        ),
        "consumer_terminal": _json_binding(
            terminal_path, "consumer_terminal_sha256"
        ),
        "consumer_join": _json_binding(
            join_path, "consumer_join_sha256"
        ),
        "consumer_lifecycle": expected_lifecycle,
        "controller_owner_seal": owner_seal,
        "adjudicated_outcome": adjudicated_outcome,
        "status": "cleaned",
        "session_residual": False,
        "completed_at": _utc_now(),
    }
    cleanup["consumer_cleanup_sha256"] = _canonical_digest(
        cleanup, "consumer_cleanup_sha256"
    )
    _write_exclusive(cleanup_path, cleanup)
    return validate_cleanup(
        cleanup,
        _json_binding(join_path, "consumer_join_sha256"),
    )


def _sealed_finalization_json(
    path: Path,
    *,
    digest_field: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not path.is_absolute():
        raise RuntimeError(f"{label} path is not absolute")
    directory_descriptor, _directory_identity = (
        _launcher_open_publication_directory(path.parent)
    )
    try:
        content, opened = _launcher_read_publication_file(
            directory_descriptor,
            path.name,
            fsync_file=True,
        )
        named = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or opened.st_mode != named.st_mode
            or opened.st_uid != named.st_uid
            or opened.st_nlink != 1
            or named.st_nlink != 1
            or opened.st_size != named.st_size
            or opened.st_mtime_ns != named.st_mtime_ns
        ):
            raise RuntimeError(f"{label} named identity changed")
        _launcher_fsync_dirfd(directory_descriptor)
    finally:
        _launcher_checked_close(
            directory_descriptor, f"{label} directory"
        )
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} content is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get(digest_field)
        != _canonical_digest(value, digest_field)
    ):
        raise RuntimeError(f"{label} canonical digest differs")
    artifact = build_artifact_binding(
        path=str(path),
        sha256=hashlib.sha256(content).hexdigest(),
        canonical_sha256=str(value[digest_field]),
    )
    identity = build_file_identity(
        path=str(path),
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
        mode=int(opened.st_mode),
        size=int(opened.st_size),
    )
    return value, artifact, identity


def _require_exact_finalization_owner(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    runtime_keys = {
        "session",
        "pane",
        "pane_pid",
        "pane_dead",
        "pane_dead_status",
        "pane_process",
        "owner_nonce",
        "tmux_server",
    }
    stable_fields = (
        "session",
        "pane",
        "pane_pid",
        "owner_nonce",
        "tmux_server",
    )
    if (
        set(observed) != runtime_keys
        or set(expected) != runtime_keys
        or expected.get("pane_dead") is not False
        or expected.get("pane_dead_status") is not None
        or not isinstance(expected.get("pane_process"), Mapping)
        or type(observed.get("pane_dead")) is not bool
        or (
            observed.get("pane_dead") is True
            and (
                observed.get("pane_process") is not None
                or (
                    observed.get("pane_dead_status") is not None
                    and type(observed.get("pane_dead_status"))
                    is not int
                )
            )
        )
        or (
            observed.get("pane_dead") is False
            and (
                observed.get("pane_dead_status") is not None
                or observed.get("pane_process")
                != expected["pane_process"]
            )
        )
        or any(
            observed.get(field) != expected.get(field)
            for field in stable_fields
        )
        or (
            expected.get("pane_process") is not None
            and observed.get("pane_process")
            not in (expected["pane_process"], None)
        )
    ):
        raise RuntimeError(f"{label} owner seal differs")
    return dict(observed)


def _require_formal_finalization_owner(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    runtime_keys = {
        "session",
        "pane",
        "pane_pid",
        "pane_dead",
        "pane_dead_status",
        "pane_process",
        "owner_nonce",
        "tmux_server",
    }
    server = observed.get("tmux_server")
    if (
        set(observed) != runtime_keys
        or observed.get("pane_dead") is not False
        or observed.get("pane_dead_status") is not None
        or not isinstance(observed.get("pane_process"), Mapping)
        or not isinstance(server, Mapping)
    ):
        raise RuntimeError(f"{label} runtime owner seal differs")
    formal = build_pane_owner_seal(
        server_pid=server["server_pid"],
        server_start_ticks=server["server_process"]["start_ticks"],
        socket_path=server["socket_path"],
        socket_device=server["socket_device"],
        socket_inode=server["socket_inode"],
        session=str(observed["session"]),
        pane=str(observed["pane"]),
        pane_pid=int(observed["pane_pid"]),
        pane_process=observed["pane_process"],
        owner_nonce=str(observed["owner_nonce"]),
        tmux_identity={
            "session": observed["session"],
            "pane": observed["pane"],
            "pane_pid": observed["pane_pid"],
        },
        tmux_server=server,
    )
    if formal != dict(expected):
        raise RuntimeError(f"{label} formal owner seal differs")
    return formal


def _validate_consumer_attempt_evidence(
    value: Mapping[str, Any],
    *,
    expected_attempt_id: str,
    expected_receipt: Mapping[str, Any],
    expected_receipt_identity: Mapping[str, Any],
    expected_gate_owner_seal: Mapping[str, Any],
    error_message: str = (
        "preclaim finalization consumer attempt differs"
    ),
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "attempt_id",
        "launch_receipt",
        "launch_receipt_identity",
        "gate_owner_seal",
        "pane_fault_channel",
        "pane_fault_publisher",
        "consumer_self_fault_channel",
        "consumer_self_fault_publisher",
        "consumer_session",
        "consumer_owner_nonce",
        "consumer_worker_arguments",
        "consumer_lifecycle_wait_channel",
        "consumer_lifecycle_wait_publisher",
        "consumer_lifecycle_wait_supervisor_arguments",
        "consumer_lifecycle_wait_supervisor_ready_path",
        "consumer_lifecycle_wait_status_path",
        "consumer_tmux_arguments",
        "consumer_log",
        "artifacts",
        "reserved_at",
        "consumer_attempt_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_pane_fault_consumer_attempt_v1"
        or value.get("attempt_id") != expected_attempt_id
        or value.get("launch_receipt") != dict(expected_receipt)
        or value.get("launch_receipt_identity")
        != dict(expected_receipt_identity)
        or value.get("gate_owner_seal")
        != dict(expected_gate_owner_seal)
        or value.get("consumer_attempt_sha256")
        != _canonical_digest(value, "consumer_attempt_sha256")
    ):
        raise RuntimeError(error_message)
    return dict(value)


def _validate_controller_cleanup_evidence(
    value: Mapping[str, Any],
    *,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    gate_terminal_binding: Mapping[str, Any],
    formal_gate: Mapping[str, Any],
    expected_consumer_chain: Mapping[str, Any],
    dead_gate_owner_seal: Mapping[str, Any],
    contract_error: str = (
        "preclaim finalization controller cleanup differs"
    ),
    chain_error: str = (
        "preclaim finalization controller cleanup chain differs"
    ),
    exit_error: str = (
        "preclaim finalization controller cleanup exit differs"
    ),
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "attempt_id",
        "consumer_attempt",
        "gate_owner_seal",
        "gate_execution_terminal",
        "launch_accepted",
        "launch_terminal",
        "launch_ownership_release",
        "consumer_reader_release",
        "dead_owner_seal",
        "controller_exit_code",
        "adjudicated_outcome",
        "ownership_chain_state",
        "status",
        "cleanup_performed",
        "session_residual",
        "process_residual",
        "final_empty_snapshots",
        "completed_at",
        "consumer_controller_cleanup_sha256",
    }
    adjudication = formal_gate["adjudication"]
    ownership = formal_gate["ownership_chain"]
    if type(value.get("controller_exit_code")) is not int:
        raise RuntimeError(exit_error)
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_pane_fault_consumer_controller_cleanup_v2"
        or value.get("policy_sha256") != attempt["policy_sha256"]
        or value.get("attempt_id") != attempt["attempt_id"]
        or value.get("consumer_attempt") != dict(attempt_binding)
        or value.get("gate_owner_seal")
        != attempt["gate_owner_seal"]
        or value.get("dead_owner_seal")
        != dict(dead_gate_owner_seal)
        or value.get("controller_exit_code")
        != adjudication["controller_exit_code"]
        or value.get("adjudicated_outcome")
        != adjudication["adjudicated_outcome"]
        or value.get("ownership_chain_state")
        != formal_gate["ownership_chain_state"]
        or value.get("status") != "controller_dead_cleaned"
        or value.get("cleanup_performed") is not True
        or value.get("session_residual") is not False
        or value.get("process_residual") is not False
        or value.get("consumer_controller_cleanup_sha256")
        != _canonical_digest(
            value, "consumer_controller_cleanup_sha256"
        )
    ):
        raise RuntimeError(contract_error)
    if (
        value.get("gate_execution_terminal")
        != dict(gate_terminal_binding)
        or value.get("launch_accepted")
        != ownership["launch_accepted"]
        or value.get("launch_terminal")
        != ownership["launch_terminal"]
        or value.get("launch_ownership_release")
        != ownership["launch_ownership_release"]
        or value.get("consumer_reader_release")
        != expected_consumer_chain["consumer_reader_release"]
    ):
        raise RuntimeError(chain_error)
    return dict(value)


def _continue_wrapper_early_exit_from_durable_cleanup(
    *,
    attempt_root: Path,
    launch_receipt: Mapping[str, Any],
    launch_receipt_path: Path,
    launch_receipt_identity: Mapping[str, Any],
    gate_ready_path: Path,
    live_gate_owner_seal: Mapping[str, Any],
    expected_consumer_chain: Mapping[str, Any],
    pane_fault_consumer: Mapping[str, Any],
    config: Path,
    deadline: float,
    startup_timeout_seconds: float,
) -> dict[str, Any]:
    """Continue an early wrapper exit from the consumer's durable authority."""
    consumer_artifacts = launch_receipt[
        "pane_fault_consumer"
    ]["artifacts"]
    attempt_path = Path(str(consumer_artifacts["attempt"]))
    controller_cleanup_path = Path(
        str(consumer_artifacts["controller_cleanup"])
    )
    gate_terminal_path = attempt_root / "gate_execution_terminal.json"
    while not controller_cleanup_path.is_file():
        _require_post_handoff_pane_fault_consumer(
            pane_fault_consumer,
            "preclaim controller cleanup",
        )
        if time.monotonic() >= deadline:
            raise PaneFaultConsumerReservationError(
                RuntimeError("preclaim formal consumer timed out")
            )
        time.sleep(0.02)
    consumer_attempt, attempt_binding, _ = _sealed_finalization_json(
        attempt_path,
        digest_field="consumer_attempt_sha256",
        label="preclaim early-exit consumer attempt",
    )
    _validate_consumer_attempt_evidence(
        consumer_attempt,
        expected_attempt_id=str(launch_receipt["attempt_id"]),
        expected_receipt=_json_binding(
            launch_receipt_path, "launch_receipt_sha256"
        ),
        expected_receipt_identity=launch_receipt_identity,
        expected_gate_owner_seal=live_gate_owner_seal,
        error_message="preclaim early-exit consumer attempt differs",
    )
    gate_terminal, gate_terminal_binding, _ = _sealed_finalization_json(
        gate_terminal_path,
        digest_field="gate_execution_terminal_sha256",
        label="preclaim early-exit gate terminal",
    )
    _validate_gate_execution_terminal_value(
        gate_terminal,
        receipt_binding=_json_binding(
            launch_receipt_path, "launch_receipt_sha256"
        ),
        receipt_identity=launch_receipt_identity,
        gate_ready_binding=_json_binding(
            gate_ready_path, "pane_gate_ready_sha256"
        ),
        wrapper_arguments=launch_receipt["wrapper_arguments"],
        expected_ownership_chain={
            "launch_accepted": None,
            "launch_terminal": None,
            "launch_ownership_release": None,
        },
    )
    controller_cleanup, controller_cleanup_binding, _ = (
        _sealed_finalization_json(
            controller_cleanup_path,
            digest_field="consumer_controller_cleanup_sha256",
            label="preclaim early-exit controller cleanup",
        )
    )
    dead_gate_owner_seal = controller_cleanup.get("dead_owner_seal")
    if not isinstance(dead_gate_owner_seal, Mapping):
        raise RuntimeError(
            "preclaim early-exit cleanup dead owner differs"
        )
    _validate_pane_owner_lifecycle_transition(
        live_gate_owner_seal,
        dead_gate_owner_seal,
        label="preclaim early-exit gate",
    )
    formal_gate = _read_formal_gate_lifecycle_status(
        attempt_root=attempt_root,
        pane=dead_gate_owner_seal,
    )
    if formal_gate["gate_execution"] != gate_terminal:
        raise RuntimeError("preclaim early-exit formal gate differs")
    _validate_controller_cleanup_evidence(
        controller_cleanup,
        attempt=consumer_attempt,
        attempt_binding=attempt_binding,
        gate_terminal_binding=gate_terminal_binding,
        formal_gate=formal_gate,
        expected_consumer_chain=expected_consumer_chain,
        dead_gate_owner_seal=dead_gate_owner_seal,
        contract_error=(
            "preclaim early-exit controller cleanup differs"
        ),
        chain_error=(
            "preclaim early-exit controller cleanup chain differs"
        ),
        exit_error=(
            "preclaim early-exit controller cleanup exit differs"
        ),
    )
    joined_cleanup = join_pane_fault_consumer(
        attempt_path=attempt_path,
        config=config,
        timeout_seconds=startup_timeout_seconds,
    )
    consumer_terminal_path = Path(
        str(consumer_artifacts["terminal"])
    )
    join_path = Path(str(consumer_artifacts["join"]))
    cleanup_path = Path(str(consumer_artifacts["cleanup"]))
    consumer_terminal, consumer_terminal_binding, _ = (
        _sealed_finalization_json(
            consumer_terminal_path,
            digest_field="consumer_terminal_sha256",
            label="preclaim early-exit consumer terminal",
        )
    )
    join, join_binding, _ = _sealed_finalization_json(
        join_path,
        digest_field="consumer_join_sha256",
        label="preclaim early-exit consumer join",
    )
    cleanup, _cleanup_binding, _ = _sealed_finalization_json(
        cleanup_path,
        digest_field="consumer_cleanup_sha256",
        label="preclaim early-exit consumer cleanup",
    )
    if cleanup != joined_cleanup:
        raise RuntimeError(
            "preclaim early-exit joined cleanup differs"
        )
    retired_consumer_pane = join.get("retired_pane")
    if not isinstance(retired_consumer_pane, Mapping):
        raise RuntimeError(
            "preclaim early-exit consumer dead owner differs"
        )
    formal_consumer = _read_formal_consumer_lifecycle_status(
        attempt_path=attempt_path,
        pane=retired_consumer_pane,
    )
    dead_consumer_owner_seal = {
        **dict(consumer_terminal["supervisor_owner_seal"]),
        "pane_dead": True,
        "pane_dead_status": retired_consumer_pane[
            "pane_dead_status"
        ],
        "pane_process": None,
    }
    _validate_pane_owner_lifecycle_transition(
        formal_consumer["supervisor_ready"]["owner_seal"],
        dead_consumer_owner_seal,
        label="preclaim early-exit consumer",
    )
    _validate_consumer_terminal_evidence(
        consumer_terminal,
        attempt=consumer_attempt,
        attempt_binding=attempt_binding,
        controller_cleanup_binding=controller_cleanup_binding,
        gate_terminal_binding=gate_terminal_binding,
        formal_gate=formal_gate,
        formal_consumer=formal_consumer,
        expected_consumer_chain=expected_consumer_chain,
        error_message=(
            "preclaim early-exit consumer terminal differs"
        ),
    )
    adjudicated_outcome = formal_gate["adjudication"][
        "adjudicated_outcome"
    ]
    _validate_consumer_join_evidence(
        join,
        attempt=consumer_attempt,
        attempt_binding=attempt_binding,
        terminal_binding=consumer_terminal_binding,
        supervisor_ready_binding=formal_consumer[
            "supervisor_ready_binding"
        ],
        worker_ready_binding=formal_consumer[
            "worker_ready_binding"
        ],
        formal_consumer=formal_consumer,
        dead_consumer_owner_seal=dead_consumer_owner_seal,
        adjudicated_outcome=adjudicated_outcome,
        error_message="preclaim early-exit consumer join differs",
    )
    _validate_consumer_cleanup_evidence(
        cleanup,
        attempt=consumer_attempt,
        attempt_binding=attempt_binding,
        terminal_binding=consumer_terminal_binding,
        join_binding=join_binding,
        formal_consumer=formal_consumer,
        adjudicated_outcome=adjudicated_outcome,
        error_message="preclaim early-exit consumer cleanup differs",
    )
    return dict(formal_gate["gate_execution"])


def _validate_consumer_terminal_evidence(
    value: Mapping[str, Any],
    *,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    controller_cleanup_binding: Mapping[str, Any],
    gate_terminal_binding: Mapping[str, Any],
    formal_gate: Mapping[str, Any],
    formal_consumer: Mapping[str, Any],
    expected_consumer_chain: Mapping[str, Any],
    error_message: str = (
        "preclaim finalization consumer terminal differs"
    ),
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "attempt_id",
        "consumer_attempt",
        "consumer_started",
        "consumer_active",
        "consumer_reader_release",
        "consumer_release_observed",
        "controller_cleanup",
        "gate_execution_terminal",
        "launch_accepted",
        "launch_terminal",
        "launch_ownership_release",
        "ownership_chain_state",
        "consumer_session",
        "consumer_owner_nonce",
        "supervisor_owner_seal",
        "supervisor_process",
        "worker_process",
        "final_empty_snapshots",
        "controller_exit_code",
        "status",
        "exit_code",
        "completed_at",
        "consumer_terminal_sha256",
    }
    ownership = formal_gate["ownership_chain"]
    adjudication = formal_gate["adjudication"]
    empty = {
        "state": "empty",
        "record": None,
        "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_pane_fault_consumer_terminal_v2"
        or value.get("policy_sha256") != attempt["policy_sha256"]
        or value.get("attempt_id") != attempt["attempt_id"]
        or value.get("consumer_attempt") != dict(attempt_binding)
        or any(
            value.get(key) != expected
            for key, expected in expected_consumer_chain.items()
        )
        or value.get("controller_cleanup")
        != dict(controller_cleanup_binding)
        or value.get("gate_execution_terminal")
        != dict(gate_terminal_binding)
        or value.get("launch_accepted")
        != ownership["launch_accepted"]
        or value.get("launch_terminal")
        != ownership["launch_terminal"]
        or value.get("launch_ownership_release")
        != ownership["launch_ownership_release"]
        or value.get("ownership_chain_state")
        != formal_gate["ownership_chain_state"]
        or value.get("consumer_session")
        != attempt["consumer_session"]
        or value.get("consumer_owner_nonce")
        != attempt["consumer_owner_nonce"]
        or value.get("supervisor_owner_seal")
        != formal_consumer["supervisor_ready"]["owner_seal"]
        or value.get("supervisor_process")
        != formal_consumer["supervisor_ready"]["supervisor_process"]
        or value.get("worker_process")
        != formal_consumer["supervisor_ready"][
            "consumer_worker_process"
        ]
        or value.get("final_empty_snapshots")
        != {"pane_gate": empty, "consumer_self": empty}
        or value.get("controller_exit_code")
        != adjudication["controller_exit_code"]
        or value.get("status")
        != adjudication["adjudicated_outcome"]
        or value.get("exit_code") != 0
        or value.get("consumer_terminal_sha256")
        != _canonical_digest(value, "consumer_terminal_sha256")
    ):
        raise RuntimeError(error_message)
    return dict(value)


def _validate_consumer_join_evidence(
    value: Mapping[str, Any],
    *,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    terminal_binding: Mapping[str, Any],
    supervisor_ready_binding: Mapping[str, Any],
    worker_ready_binding: Mapping[str, Any],
    formal_consumer: Mapping[str, Any],
    dead_consumer_owner_seal: Mapping[str, Any],
    adjudicated_outcome: str,
    error_message: str = (
        "preclaim finalization consumer join differs"
    ),
) -> dict[str, Any]:
    retired = {
        key: dead_consumer_owner_seal[key]
        for key in (
            "session",
            "pane",
            "pane_pid",
            "pane_dead",
            "pane_dead_status",
        )
    }
    expected_keys = {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "attempt_id",
        "consumer_attempt",
        "consumer_terminal",
        "consumer_lifecycle",
        "consumer_wait_supervisor_ready",
        "consumer_worker_ready",
        "consumer_session",
        "consumer_owner_nonce",
        "retired_pane",
        "adjudicated_outcome",
        "consumer_adjudicated_exit",
        "status",
        "session_residual",
        "authorized_at",
        "consumer_join_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_pane_fault_consumer_join_v3"
        or value.get("policy_sha256") != attempt["policy_sha256"]
        or value.get("attempt_id") != attempt["attempt_id"]
        or value.get("consumer_attempt") != dict(attempt_binding)
        or value.get("consumer_terminal") != dict(terminal_binding)
        or value.get("consumer_lifecycle")
        != formal_consumer["snapshot"]
        or value.get("consumer_wait_supervisor_ready")
        != dict(supervisor_ready_binding)
        or value.get("consumer_worker_ready")
        != dict(worker_ready_binding)
        or value.get("consumer_session")
        != attempt["consumer_session"]
        or value.get("consumer_owner_nonce")
        != attempt["consumer_owner_nonce"]
        or value.get("retired_pane") != retired
        or value.get("adjudicated_outcome") != adjudicated_outcome
        or value.get("consumer_adjudicated_exit")
        != CONSUMER_ADJUDICATED_EXIT
        or value.get("status") != "cleanup_authorized"
        or value.get("session_residual") is not True
        or not isinstance(value.get("authorized_at"), str)
        or not value["authorized_at"]
        or value.get("consumer_join_sha256")
        != _canonical_digest(value, "consumer_join_sha256")
    ):
        raise RuntimeError(error_message)
    return dict(value)


def _validate_consumer_cleanup_evidence(
    value: Mapping[str, Any],
    *,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    terminal_binding: Mapping[str, Any],
    join_binding: Mapping[str, Any],
    formal_consumer: Mapping[str, Any],
    adjudicated_outcome: str,
    error_message: str = (
        "preclaim finalization consumer cleanup differs"
    ),
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "attempt_id",
        "consumer_attempt",
        "consumer_terminal",
        "consumer_join",
        "consumer_lifecycle",
        "controller_owner_seal",
        "adjudicated_outcome",
        "status",
        "session_residual",
        "completed_at",
        "consumer_cleanup_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_pane_fault_consumer_cleanup_v3"
        or value.get("policy_sha256") != attempt["policy_sha256"]
        or value.get("attempt_id") != attempt["attempt_id"]
        or value.get("consumer_attempt") != dict(attempt_binding)
        or value.get("consumer_terminal") != dict(terminal_binding)
        or value.get("consumer_join") != dict(join_binding)
        or value.get("consumer_lifecycle")
        != formal_consumer["snapshot"]
        or value.get("controller_owner_seal")
        != attempt["gate_owner_seal"]
        or value.get("adjudicated_outcome") != adjudicated_outcome
        or value.get("status") != "cleaned"
        or value.get("session_residual") is not False
        or not isinstance(value.get("completed_at"), str)
        or not value["completed_at"]
        or value.get("consumer_cleanup_sha256")
        != _canonical_digest(value, "consumer_cleanup_sha256")
    ):
        raise RuntimeError(error_message)
    return dict(value)


class PreclaimFinalizationEvidenceState(Enum):
    INTENT_ONLY = "intent_only"
    GATE_EVIDENCE = "gate_evidence"
    CONTROLLER_CLEANUP_PRESENT = "controller_cleanup_present"
    CONSUMER_TERMINAL_CHAIN = "consumer_terminal_chain"
    CONSUMER_JOIN_PRESENT = "consumer_join_present"
    CONSUMER_CLEANUP_PRESENT = "consumer_cleanup_present"
    LAUNCH_TERMINAL = "launch_terminal"


def _finalization_artifact_present(
    path: Path, *, label: str
) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"{label} is not a regular file")
    return True


def _load_validate_preclaim_finalization_evidence(
    *,
    attempt_root: Path,
    intent_publication: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    expected_gate_owner_seal: Mapping[str, Any],
    expected_consumer_owner_seal: Mapping[str, Any],
    launch_terminal_path: Path,
) -> PreclaimFinalizationEvidenceState:
    """Read and classify only the exact durable preclaim-finalization state."""
    if (
        not attempt_root.is_absolute()
        or not launch_terminal_path.is_absolute()
        or launch_terminal_path.parent != attempt_root
    ):
        raise RuntimeError("preclaim finalization evidence paths differ")
    intent_path = Path(str(intent_publication["artifact"]["path"]))
    intent, intent_binding, intent_identity = (
        _sealed_finalization_json(
            intent_path,
            digest_field="preclaim_failure_intent_sha256",
            label="preclaim failure intent",
        )
    )
    if (
        intent != intent_publication.get("intent")
        or intent_binding != intent_publication.get("artifact")
        or intent_identity != intent_publication.get("file_identity")
    ):
        raise RuntimeError("preclaim failure intent seal differs")
    receipt, receipt_binding, receipt_identity = (
        _sealed_finalization_json(
            Path(str(intent["launch_receipt"]["path"])),
            digest_field="launch_receipt_sha256",
            label="preclaim finalization launch receipt",
        )
    )
    if (
        receipt != launch_receipt
        or receipt_binding != intent["launch_receipt"]
        or receipt_identity != dict(launch_receipt_identity)
    ):
        raise RuntimeError("preclaim finalization authority differs")
    validate_launch_receipt_schema(
        receipt,
        expected_gate_worker_arguments=receipt[
            "gate_worker_arguments"
        ],
        expected_consumer_worker_arguments=receipt[
            "consumer_worker_arguments"
        ],
        label="preclaim finalization launch receipt",
    )
    consumer_artifacts = receipt["pane_fault_consumer"]["artifacts"]
    consumer_attempt, consumer_attempt_binding, _ = (
        _sealed_finalization_json(
            Path(str(consumer_artifacts["attempt"])),
            digest_field="consumer_attempt_sha256",
            label="preclaim finalization consumer attempt",
        )
    )
    attempt_gate_owner_seal = consumer_attempt.get(
        "gate_owner_seal"
    )
    if not isinstance(attempt_gate_owner_seal, Mapping):
        raise RuntimeError(
            "preclaim finalization consumer attempt owner differs"
        )
    _require_exact_finalization_owner(
        expected_gate_owner_seal,
        attempt_gate_owner_seal,
        label="preclaim finalization gate",
    )
    runtime_tmux_server = expected_gate_owner_seal.get(
        "tmux_server"
    )
    if not isinstance(runtime_tmux_server, Mapping):
        raise RuntimeError(
            "preclaim finalization runtime tmux server differs"
        )
    validate_preclaim_failure_intent(
        intent,
        verified_implementations=intent[
            "verified_implementations"
        ],
        expected_wrapper_claim_path=str(
            launch_receipt["wrapper_claim_path"]
        ),
        tmux_identity={
            "session": expected_gate_owner_seal["session"],
            "pane": expected_gate_owner_seal["pane"],
            "pane_pid": expected_gate_owner_seal["pane_pid"],
        },
        tmux_server=runtime_tmux_server,
        expected_receipt=intent["launch_receipt"],
        expected_receipt_identity=launch_receipt_identity,
        expected_consumer_chain=intent[
            "pane_fault_consumer_chain"
        ],
        label="preclaim finalization intent",
    )
    _require_formal_finalization_owner(
        attempt_gate_owner_seal,
        intent["controller_owner_seal"],
        label="preclaim finalization gate",
    )
    _validate_consumer_attempt_evidence(
        consumer_attempt,
        expected_attempt_id=intent["attempt_id"],
        expected_receipt=receipt_binding,
        expected_receipt_identity=receipt_identity,
        expected_gate_owner_seal=attempt_gate_owner_seal,
    )
    if (
        expected_consumer_owner_seal.get("session")
        != consumer_attempt["consumer_session"]
        or expected_consumer_owner_seal.get("owner_nonce")
        != consumer_attempt["consumer_owner_nonce"]
    ):
        raise RuntimeError(
            "preclaim finalization consumer owner differs"
        )

    gate_terminal_path = Path(
        str(receipt["gate_execution_terminal_path"])
    )
    controller_cleanup_path = Path(
        str(consumer_artifacts["controller_cleanup"])
    )
    consumer_terminal_path = Path(
        str(consumer_artifacts["terminal"])
    )
    join_path = Path(str(consumer_artifacts["join"]))
    cleanup_path = Path(str(consumer_artifacts["cleanup"]))
    paths = (
        gate_terminal_path,
        controller_cleanup_path,
        consumer_terminal_path,
        join_path,
        cleanup_path,
        launch_terminal_path,
    )
    present = tuple(
        _finalization_artifact_present(
            path, label=f"preclaim finalization {path.name}"
        )
        for path in paths
    )
    if not present[0]:
        if any(present[1:]):
            raise RuntimeError(
                "preclaim finalization evidence precedes gate terminal"
            )
        return PreclaimFinalizationEvidenceState.INTENT_ONLY

    gate_terminal, gate_terminal_binding, _ = (
        _sealed_finalization_json(
            gate_terminal_path,
            digest_field="gate_execution_terminal_sha256",
            label="preclaim finalization gate terminal",
        )
    )
    gate_ready_path = attempt_root / "pane_gate_ready.json"
    _validate_gate_execution_terminal_value(
        gate_terminal,
        receipt_binding=receipt_binding,
        receipt_identity=receipt_identity,
        gate_ready_binding=_json_binding(
            gate_ready_path, "pane_gate_ready_sha256"
        ),
        wrapper_arguments=receipt["wrapper_arguments"],
        expected_ownership_chain={
            "launch_accepted": None,
            "launch_terminal": None,
            "launch_ownership_release": None,
        },
    )
    cleanup_present = present[1]
    terminal_present = present[2]
    if terminal_present and not cleanup_present:
        raise RuntimeError(
            "consumer terminal exists without controller cleanup"
        )
    if present[3] and not terminal_present:
        raise RuntimeError(
            "consumer join exists without terminal chain"
        )
    if present[4] and not present[3]:
        raise RuntimeError(
            "consumer cleanup exists without join"
        )
    if present[5] and not present[4]:
        raise RuntimeError(
            "launch terminal exists before consumer cleanup"
        )
    formal_gate: dict[str, Any] | None = None
    controller_cleanup: dict[str, Any] | None = None
    controller_cleanup_binding: dict[str, Any] | None = None
    if cleanup_present:
        if expected_gate_owner_seal.get("pane_dead") is not True:
            raise RuntimeError(
                "controller cleanup lacks a dead gate owner seal"
            )
        _validate_pane_owner_lifecycle_transition(
            attempt_gate_owner_seal,
            expected_gate_owner_seal,
            label="preclaim finalization gate",
        )
        formal_gate = _read_formal_gate_lifecycle_status(
            attempt_root=attempt_root,
            pane=expected_gate_owner_seal,
        )
        if formal_gate["gate_execution"] != gate_terminal:
            raise RuntimeError(
                "preclaim finalization formal gate differs"
            )
        controller_cleanup, controller_cleanup_binding, _ = (
            _sealed_finalization_json(
            controller_cleanup_path,
            digest_field="consumer_controller_cleanup_sha256",
            label="preclaim finalization controller cleanup",
            )
        )
        _validate_controller_cleanup_evidence(
            controller_cleanup,
            attempt=consumer_attempt,
            attempt_binding=consumer_attempt_binding,
            gate_terminal_binding=gate_terminal_binding,
            formal_gate=formal_gate,
            expected_consumer_chain=intent[
                "pane_fault_consumer_chain"
            ],
            dead_gate_owner_seal=expected_gate_owner_seal,
        )
    formal_consumer: dict[str, Any] | None = None
    consumer_terminal: dict[str, Any] | None = None
    consumer_terminal_binding: dict[str, Any] | None = None
    if terminal_present:
        if (
            formal_gate is None
            or controller_cleanup_binding is None
            or expected_consumer_owner_seal.get("pane_dead") is not True
        ):
            raise RuntimeError(
                "consumer terminal lacks formal dead authority"
            )
        formal_consumer = _read_formal_consumer_lifecycle_status(
            attempt_path=Path(str(consumer_artifacts["attempt"])),
            pane=expected_consumer_owner_seal,
        )
        _validate_pane_owner_lifecycle_transition(
            formal_consumer["supervisor_ready"]["owner_seal"],
            expected_consumer_owner_seal,
            label="preclaim finalization consumer",
        )
        consumer_terminal, consumer_terminal_binding, _ = (
            _sealed_finalization_json(
                consumer_terminal_path,
                digest_field="consumer_terminal_sha256",
                label="preclaim finalization consumer terminal",
            )
        )
        _validate_consumer_terminal_evidence(
            consumer_terminal,
            attempt=consumer_attempt,
            attempt_binding=consumer_attempt_binding,
            controller_cleanup_binding=controller_cleanup_binding,
            gate_terminal_binding=gate_terminal_binding,
            formal_gate=formal_gate,
            formal_consumer=formal_consumer,
            expected_consumer_chain=intent[
                "pane_fault_consumer_chain"
            ],
        )
    join: dict[str, Any] | None = None
    join_binding: dict[str, Any] | None = None
    if present[3]:
        if (
            formal_gate is None
            or formal_consumer is None
            or consumer_terminal_binding is None
        ):
            raise RuntimeError(
                "consumer join lacks formal terminal authority"
            )
        join, join_binding, _ = _sealed_finalization_json(
            join_path,
            digest_field="consumer_join_sha256",
            label="preclaim finalization consumer join",
        )
        _validate_consumer_join_evidence(
            join,
            attempt=consumer_attempt,
            attempt_binding=consumer_attempt_binding,
            terminal_binding=consumer_terminal_binding,
            supervisor_ready_binding=formal_consumer[
                "supervisor_ready_binding"
            ],
            worker_ready_binding=formal_consumer[
                "worker_ready_binding"
            ],
            formal_consumer=formal_consumer,
            dead_consumer_owner_seal=expected_consumer_owner_seal,
            adjudicated_outcome=formal_gate["adjudication"][
                "adjudicated_outcome"
            ],
        )
    if present[4]:
        if (
            formal_gate is None
            or formal_consumer is None
            or consumer_terminal_binding is None
            or join_binding is None
        ):
            raise RuntimeError(
                "consumer cleanup lacks join authority"
            )
        cleanup, cleanup_binding, _ = _sealed_finalization_json(
            cleanup_path,
            digest_field="consumer_cleanup_sha256",
            label="preclaim finalization consumer cleanup",
        )
        _validate_consumer_cleanup_evidence(
            cleanup,
            attempt=consumer_attempt,
            attempt_binding=consumer_attempt_binding,
            terminal_binding=consumer_terminal_binding,
            join_binding=join_binding,
            formal_consumer=formal_consumer,
            adjudicated_outcome=formal_gate["adjudication"][
                "adjudicated_outcome"
            ],
        )
    if present[5]:
        launch_terminal, _binding, _identity = (
            _sealed_finalization_json(
                launch_terminal_path,
                digest_field="launch_terminal_sha256",
                label="preclaim finalization launch terminal",
            )
        )
        validate_launch_terminal_v2(
            launch_terminal,
            preclaim_failure_intent=intent,
            preclaim_failure_intent_binding=intent_binding,
            verified_implementations=intent[
                "verified_implementations"
            ],
            label="preclaim finalization launch terminal",
        )
        return PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL
    if present[4]:
        return (
            PreclaimFinalizationEvidenceState.CONSUMER_CLEANUP_PRESENT
        )
    if present[3]:
        return PreclaimFinalizationEvidenceState.CONSUMER_JOIN_PRESENT
    if terminal_present:
        return PreclaimFinalizationEvidenceState.CONSUMER_TERMINAL_CHAIN
    if cleanup_present:
        return (
            PreclaimFinalizationEvidenceState.CONTROLLER_CLEANUP_PRESENT
        )
    return PreclaimFinalizationEvidenceState.GATE_EVIDENCE


_PRECLAIM_FINALIZATION_STATE_ORDER = {
    PreclaimFinalizationEvidenceState.INTENT_ONLY: 0,
    PreclaimFinalizationEvidenceState.GATE_EVIDENCE: 1,
    PreclaimFinalizationEvidenceState.CONTROLLER_CLEANUP_PRESENT: 2,
    PreclaimFinalizationEvidenceState.CONSUMER_TERMINAL_CHAIN: 3,
    PreclaimFinalizationEvidenceState.CONSUMER_JOIN_PRESENT: 4,
    PreclaimFinalizationEvidenceState.CONSUMER_CLEANUP_PRESENT: 5,
    PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL: 6,
}


class PreclaimFinalizationTimeoutError(TimeoutError):
    """One preclaim-finalization action exhausted its own time budget."""

    def __init__(
        self,
        *,
        action: str,
        started: float,
        deadline: float,
        ended: float,
    ) -> None:
        super().__init__(
            f"preclaim finalization action timed out: {action}"
        )
        self.action = action
        self.started = started
        self.deadline = deadline
        self.ended = ended


def _start_preclaim_finalization_action(
    action: str, timeout_seconds: float
) -> dict[str, float | str]:
    if timeout_seconds <= 0:
        raise RuntimeError(
            "preclaim finalization timeout is not positive"
        )
    started = time.monotonic()
    return {
        "action": action,
        "started": started,
        "deadline": started + timeout_seconds,
    }


def _require_preclaim_finalization_action_open(
    action: Mapping[str, Any],
) -> float:
    ended = time.monotonic()
    if ended >= action["deadline"]:
        raise PreclaimFinalizationTimeoutError(
            action=str(action["action"]),
            started=float(action["started"]),
            deadline=float(action["deadline"]),
            ended=ended,
        )
    return ended


def _finish_preclaim_finalization_action(
    action: Mapping[str, Any],
) -> dict[str, float | str]:
    ended = _require_preclaim_finalization_action_open(action)
    return {
        "action": str(action["action"]),
        "started": float(action["started"]),
        "deadline": float(action["deadline"]),
        "ended": ended,
    }


def _require_monotonic_preclaim_finalization_state(
    before: PreclaimFinalizationEvidenceState,
    after: PreclaimFinalizationEvidenceState,
    *,
    action: str,
) -> None:
    if (
        _PRECLAIM_FINALIZATION_STATE_ORDER[after]
        < _PRECLAIM_FINALIZATION_STATE_ORDER[before]
    ):
        raise RuntimeError(
            f"preclaim finalization state regressed after {action}"
        )


def _preclaim_finalizer_artifact_paths(
    *,
    launch_receipt: Mapping[str, Any],
    launch_terminal_path: Path,
) -> dict[str, Path]:
    artifacts = launch_receipt["pane_fault_consumer"][
        "artifacts"
    ]
    return {
        "gate_terminal": Path(
            str(launch_receipt["gate_execution_terminal_path"])
        ),
        "controller_cleanup": Path(
            str(artifacts["controller_cleanup"])
        ),
        "consumer_terminal": Path(str(artifacts["terminal"])),
        "consumer_join": Path(str(artifacts["join"])),
        "consumer_cleanup": Path(str(artifacts["cleanup"])),
        "launch_terminal": launch_terminal_path,
    }


def _require_preclaim_finalizer_partial_order(
    paths: Mapping[str, Path],
) -> dict[str, bool]:
    present = {
        key: _finalization_artifact_present(
            path, label=f"preclaim finalizer {path.name}"
        )
        for key, path in paths.items()
    }
    if (
        present["consumer_terminal"]
        and not present["controller_cleanup"]
    ):
        raise RuntimeError(
            "consumer terminal exists without controller cleanup"
        )
    if (
        present["consumer_join"]
        and not present["consumer_terminal"]
    ):
        raise RuntimeError(
            "consumer join exists without terminal chain"
        )
    if (
        present["consumer_cleanup"]
        and not present["consumer_join"]
    ):
        raise RuntimeError(
            "consumer cleanup exists without join"
        )
    if (
        present["launch_terminal"]
        and not present["consumer_cleanup"]
    ):
        raise RuntimeError(
            "launch terminal exists before consumer cleanup"
        )
    return present


def _preclaim_finalizer_dead_consumer_owner(
    *,
    live_owner: Mapping[str, Any],
    terminal_path: Path,
    join_path: Path,
) -> dict[str, Any]:
    terminal, _terminal_binding, _terminal_identity = (
        _sealed_finalization_json(
            terminal_path,
            digest_field="consumer_terminal_sha256",
            label="preclaim finalizer consumer terminal",
        )
    )
    if terminal.get("supervisor_owner_seal") != dict(live_owner):
        raise RuntimeError(
            "preclaim finalizer consumer live owner differs"
        )
    join, _join_binding, _join_identity = (
        _sealed_finalization_json(
            join_path,
            digest_field="consumer_join_sha256",
            label="preclaim finalizer consumer join",
        )
    )
    retired = join.get("retired_pane")
    retired_keys = {
        "session",
        "pane",
        "pane_pid",
        "pane_dead",
        "pane_dead_status",
    }
    if (
        not isinstance(retired, Mapping)
        or set(retired) != retired_keys
    ):
        raise RuntimeError(
            "preclaim finalizer retired consumer pane differs"
        )
    dead_owner = {
        **dict(live_owner),
        "pane_dead": retired["pane_dead"],
        "pane_dead_status": retired["pane_dead_status"],
        "pane_process": None,
    }
    return _validate_pane_owner_lifecycle_transition(
        live_owner,
        dead_owner,
        label="preclaim finalizer consumer",
    )


def _preclaim_finalizer_owner_seals(
    *,
    launch_receipt: Mapping[str, Any],
    launch_terminal_path: Path,
    live_gate_owner_seal: Mapping[str, Any],
    live_consumer_owner_seal: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _preclaim_finalizer_artifact_paths(
        launch_receipt=launch_receipt,
        launch_terminal_path=launch_terminal_path,
    )
    present = _require_preclaim_finalizer_partial_order(paths)
    if present["controller_cleanup"]:
        cleanup, _binding, _identity = (
            _sealed_finalization_json(
                paths["controller_cleanup"],
                digest_field=(
                    "consumer_controller_cleanup_sha256"
                ),
                label="preclaim finalizer controller cleanup",
            )
        )
        dead_gate_owner = cleanup.get("dead_owner_seal")
        if not isinstance(dead_gate_owner, Mapping):
            raise RuntimeError(
                "preclaim finalizer dead gate owner differs"
            )
        gate_owner = _validate_pane_owner_lifecycle_transition(
            live_gate_owner_seal,
            dead_gate_owner,
            label="preclaim finalizer gate",
        )
    else:
        while _tmux_pane(
            str(live_gate_owner_seal["session"])
        ) is None:
            _require_preclaim_finalization_action_open(action)
            present = _require_preclaim_finalizer_partial_order(
                paths
            )
            if present["controller_cleanup"]:
                break
            time.sleep(0.02)
        if present["controller_cleanup"]:
            return _preclaim_finalizer_owner_seals(
                launch_receipt=launch_receipt,
                launch_terminal_path=launch_terminal_path,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                action=action,
            )
        gate_owner = _tmux_owner_seal(
            str(live_gate_owner_seal["session"]),
            str(live_gate_owner_seal["owner_nonce"]),
        )
        _require_exact_finalization_owner(
            gate_owner,
            live_gate_owner_seal,
            label="preclaim finalizer gate",
        )
    if present["consumer_join"]:
        consumer_owner = _preclaim_finalizer_dead_consumer_owner(
            live_owner=live_consumer_owner_seal,
            terminal_path=paths["consumer_terminal"],
            join_path=paths["consumer_join"],
        )
    else:
        while _tmux_pane(
            str(live_consumer_owner_seal["session"])
        ) is None:
            _require_preclaim_finalization_action_open(action)
            present = _require_preclaim_finalizer_partial_order(
                paths
            )
            if present["consumer_join"]:
                break
            time.sleep(0.02)
        if present["consumer_join"]:
            return _preclaim_finalizer_owner_seals(
                launch_receipt=launch_receipt,
                launch_terminal_path=launch_terminal_path,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                action=action,
            )
        consumer_owner = _tmux_owner_seal(
            str(live_consumer_owner_seal["session"]),
            str(live_consumer_owner_seal["owner_nonce"]),
        )
        _require_exact_finalization_owner(
            consumer_owner,
            live_consumer_owner_seal,
            label="preclaim finalizer consumer",
        )
    return dict(gate_owner), dict(consumer_owner)


def _read_preclaim_finalizer_state(
    *,
    attempt_root: Path,
    intent_publication: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    live_gate_owner_seal: Mapping[str, Any],
    live_consumer_owner_seal: Mapping[str, Any],
    launch_terminal_path: Path,
    action: Mapping[str, Any],
) -> tuple[
    PreclaimFinalizationEvidenceState,
    dict[str, Any],
    dict[str, Any],
]:
    gate_owner, consumer_owner = (
        _preclaim_finalizer_owner_seals(
            launch_receipt=launch_receipt,
            launch_terminal_path=launch_terminal_path,
            live_gate_owner_seal=live_gate_owner_seal,
            live_consumer_owner_seal=live_consumer_owner_seal,
            action=action,
        )
    )
    state = _load_validate_preclaim_finalization_evidence(
        attempt_root=attempt_root,
        intent_publication=intent_publication,
        launch_receipt=launch_receipt,
        launch_receipt_identity=launch_receipt_identity,
        expected_gate_owner_seal=gate_owner,
        expected_consumer_owner_seal=consumer_owner,
        launch_terminal_path=launch_terminal_path,
    )
    return state, gate_owner, consumer_owner


def _wait_preclaim_gate_terminal_and_dead(
    *,
    launch_receipt: Mapping[str, Any],
    launch_terminal_path: Path,
    live_gate_owner_seal: Mapping[str, Any],
    live_consumer_owner_seal: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _preclaim_finalizer_artifact_paths(
        launch_receipt=launch_receipt,
        launch_terminal_path=launch_terminal_path,
    )
    while True:
        _require_preclaim_finalization_action_open(action)
        present = _require_preclaim_finalizer_partial_order(paths)
        if present["controller_cleanup"]:
            return _preclaim_finalizer_owner_seals(
                launch_receipt=launch_receipt,
                launch_terminal_path=launch_terminal_path,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                action=action,
            )
        if _tmux_pane(
            str(live_gate_owner_seal["session"])
        ) is None:
            time.sleep(0.02)
            continue
        gate_owner = _tmux_owner_seal(
            str(live_gate_owner_seal["session"]),
            str(live_gate_owner_seal["owner_nonce"]),
        )
        _require_exact_finalization_owner(
            gate_owner,
            live_gate_owner_seal,
            label="preclaim finalizer gate wait",
        )
        if gate_owner["pane_dead"] is True:
            if not present["gate_terminal"]:
                raise RuntimeError(
                    "dead gate lacks execution terminal"
                )
            consumer_owner = _tmux_owner_seal(
                str(live_consumer_owner_seal["session"]),
                str(live_consumer_owner_seal["owner_nonce"]),
            )
            _require_exact_finalization_owner(
                consumer_owner,
                live_consumer_owner_seal,
                label="preclaim finalizer consumer wait",
            )
            return dict(gate_owner), dict(consumer_owner)
        time.sleep(0.02)


def _wait_preclaim_controller_cleanup(
    *,
    launch_receipt: Mapping[str, Any],
    launch_terminal_path: Path,
    live_gate_owner_seal: Mapping[str, Any],
    live_consumer_owner_seal: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _preclaim_finalizer_artifact_paths(
        launch_receipt=launch_receipt,
        launch_terminal_path=launch_terminal_path,
    )
    while True:
        _require_preclaim_finalization_action_open(action)
        present = _require_preclaim_finalizer_partial_order(paths)
        if present["controller_cleanup"]:
            return _preclaim_finalizer_owner_seals(
                launch_receipt=launch_receipt,
                launch_terminal_path=launch_terminal_path,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                action=action,
            )
        time.sleep(0.02)


def _wait_preclaim_consumer_terminal_and_dead(
    *,
    launch_receipt: Mapping[str, Any],
    launch_terminal_path: Path,
    live_gate_owner_seal: Mapping[str, Any],
    live_consumer_owner_seal: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _preclaim_finalizer_artifact_paths(
        launch_receipt=launch_receipt,
        launch_terminal_path=launch_terminal_path,
    )
    while True:
        _require_preclaim_finalization_action_open(action)
        present = _require_preclaim_finalizer_partial_order(paths)
        if present["consumer_join"]:
            return _preclaim_finalizer_owner_seals(
                launch_receipt=launch_receipt,
                launch_terminal_path=launch_terminal_path,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                action=action,
            )
        if (
            present["controller_cleanup"]
            and present["consumer_terminal"]
        ):
            gate_owner, consumer_owner = (
                _preclaim_finalizer_owner_seals(
                    launch_receipt=launch_receipt,
                    launch_terminal_path=launch_terminal_path,
                    live_gate_owner_seal=live_gate_owner_seal,
                    live_consumer_owner_seal=(
                        live_consumer_owner_seal
                    ),
                    action=action,
                )
            )
            if consumer_owner["pane_dead"] is True:
                return gate_owner, consumer_owner
        else:
            if _tmux_pane(
                str(live_consumer_owner_seal["session"])
            ) is None:
                time.sleep(0.02)
                continue
            consumer_owner = _tmux_owner_seal(
                str(live_consumer_owner_seal["session"]),
                str(live_consumer_owner_seal["owner_nonce"]),
            )
            _require_exact_finalization_owner(
                consumer_owner,
                live_consumer_owner_seal,
                label="preclaim finalizer consumer wait",
            )
            if (
                consumer_owner["pane_dead"] is True
                and not present["consumer_terminal"]
            ):
                raise RuntimeError(
                    "dead consumer lacks terminal evidence"
                )
        time.sleep(0.02)


def _bound_preclaim_lifecycle(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    record = snapshot["record"]
    return build_bound_lifecycle_evidence(
        artifact=build_artifact_binding(
            path=snapshot["channel_authority"]["path"],
            sha256=snapshot["sha256"],
            canonical_sha256=record[
                "lifecycle_wait_status_sha256"
            ],
        ),
        record=record,
        role=record["role"],
        attempt_id=record["attempt_id"],
    )


def _publish_preclaim_launch_terminal_v2(
    *,
    attempt_root: Path,
    intent_publication: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    gate_owner_seal: Mapping[str, Any],
    consumer_owner_seal: Mapping[str, Any],
    launch_terminal_path: Path,
) -> dict[str, Any]:
    artifacts = launch_receipt["pane_fault_consumer"][
        "artifacts"
    ]
    gate_terminal_path = Path(
        str(launch_receipt["gate_execution_terminal_path"])
    )
    controller_cleanup_path = Path(
        str(artifacts["controller_cleanup"])
    )
    consumer_terminal_path = Path(str(artifacts["terminal"]))
    join_path = Path(str(artifacts["join"]))
    cleanup_path = Path(str(artifacts["cleanup"]))
    gate_terminal, gate_terminal_binding, _ = (
        _sealed_finalization_json(
            gate_terminal_path,
            digest_field="gate_execution_terminal_sha256",
            label="preclaim terminal gate execution",
        )
    )
    del gate_terminal
    _controller_cleanup, controller_cleanup_binding, _ = (
        _sealed_finalization_json(
            controller_cleanup_path,
            digest_field="consumer_controller_cleanup_sha256",
            label="preclaim terminal controller cleanup",
        )
    )
    _consumer_terminal, consumer_terminal_binding, _ = (
        _sealed_finalization_json(
            consumer_terminal_path,
            digest_field="consumer_terminal_sha256",
            label="preclaim terminal consumer terminal",
        )
    )
    _join, join_binding, _ = _sealed_finalization_json(
        join_path,
        digest_field="consumer_join_sha256",
        label="preclaim terminal consumer join",
    )
    _cleanup, cleanup_binding, _ = _sealed_finalization_json(
        cleanup_path,
        digest_field="consumer_cleanup_sha256",
        label="preclaim terminal consumer cleanup",
    )
    formal_gate = _read_formal_gate_lifecycle_status(
        attempt_root=attempt_root,
        pane=gate_owner_seal,
    )
    formal_consumer = _read_formal_consumer_lifecycle_status(
        attempt_path=Path(str(artifacts["attempt"])),
        pane=consumer_owner_seal,
    )
    intent = intent_publication["intent"]
    status = {
        "invalid_claim": "launcher_failed",
        "claim_timeout": "wrapper_claim_timeout",
    }[intent["reason"]]
    failure_type, message = {
        "invalid_claim": (
            "InvalidWrapperClaim",
            "wrapper claim validation failed",
        ),
        "claim_timeout": (
            "WrapperClaimTimeout",
            "wrapper claim deadline reached",
        ),
    }[intent["reason"]]
    terminal = build_launch_terminal_v2(
        attempt_id=intent["attempt_id"],
        preclaim_failure_intent=intent,
        preclaim_failure_intent_binding=intent_publication[
            "artifact"
        ],
        launch_receipt=intent["launch_receipt"],
        launch_receipt_identity=launch_receipt_identity,
        verified_implementations=intent[
            "verified_implementations"
        ],
        pane_fault_consumer_chain=intent[
            "pane_fault_consumer_chain"
        ],
        gate_execution_terminal=gate_terminal_binding,
        gate_lifecycle=_bound_preclaim_lifecycle(
            formal_gate["snapshot"]
        ),
        controller_cleanup=controller_cleanup_binding,
        consumer_terminal=consumer_terminal_binding,
        consumer_lifecycle=_bound_preclaim_lifecycle(
            formal_consumer["snapshot"]
        ),
        consumer_join=join_binding,
        consumer_cleanup=cleanup_binding,
        status=status,
        failure=build_terminal_failure(
            reason=intent["reason"],
            stage=intent["stage"],
            failure_type=failure_type,
            message=message,
        ),
        started_at=launch_receipt["started_at"],
        completed_at=_cleanup["completed_at"],
    )
    try:
        _write_exclusive(launch_terminal_path, terminal)
        observed, _binding, _identity = _sealed_finalization_json(
            launch_terminal_path,
            digest_field="launch_terminal_sha256",
            label="published preclaim launch terminal",
        )
        if observed != terminal:
            raise RuntimeError(
                "published preclaim launch terminal differs"
            )
    except LauncherTerminalPublishError:
        raise
    except LauncherExclusivePublishError:
        raise
    except BaseException as exc:
        raise LauncherTerminalPublishError(
            launch_terminal_path, exc
        ) from exc
    return observed


def _run_preclaim_finalization_action(
    *,
    action_name: str,
    timeout_seconds: float,
    expected_states: set[PreclaimFinalizationEvidenceState],
    attempt_root: Path,
    intent_publication: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    live_gate_owner_seal: Mapping[str, Any],
    live_consumer_owner_seal: Mapping[str, Any],
    launch_terminal_path: Path,
    operation: Callable[
        [
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ],
        None,
    ],
) -> tuple[
    PreclaimFinalizationEvidenceState,
    dict[str, Any],
    dict[str, Any],
    dict[str, float | str] | None,
]:
    action = _start_preclaim_finalization_action(
        action_name, timeout_seconds
    )
    before, gate_owner, consumer_owner = (
        _read_preclaim_finalizer_state(
            attempt_root=attempt_root,
            intent_publication=intent_publication,
            launch_receipt=launch_receipt,
            launch_receipt_identity=launch_receipt_identity,
            live_gate_owner_seal=live_gate_owner_seal,
            live_consumer_owner_seal=live_consumer_owner_seal,
            launch_terminal_path=launch_terminal_path,
            action=action,
        )
    )
    if before is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return before, gate_owner, consumer_owner, None
    if before not in expected_states:
        raise RuntimeError(
            f"preclaim finalization state differs before {action_name}"
        )
    _require_preclaim_finalization_action_open(action)
    operation(action, gate_owner, consumer_owner)
    _require_preclaim_finalization_action_open(action)
    after, gate_owner, consumer_owner = (
        _read_preclaim_finalizer_state(
            attempt_root=attempt_root,
            intent_publication=intent_publication,
            launch_receipt=launch_receipt,
            launch_receipt_identity=launch_receipt_identity,
            live_gate_owner_seal=live_gate_owner_seal,
            live_consumer_owner_seal=live_consumer_owner_seal,
            launch_terminal_path=launch_terminal_path,
            action=action,
        )
    )
    _require_monotonic_preclaim_finalization_state(
        before, after, action=action_name
    )
    timing = _finish_preclaim_finalization_action(action)
    return after, gate_owner, consumer_owner, timing


def _resume_or_finalize_preclaim_failure(
    *,
    config: Path,
    timeout_seconds: float,
    attempt_root: Path,
    intent_publication: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    live_gate_owner_seal: Mapping[str, Any],
    live_consumer_owner_seal: Mapping[str, Any],
    launch_terminal_path: Path,
) -> dict[str, Any]:
    """Resume the unique durable preclaim finalization action."""
    if timeout_seconds <= 0:
        raise RuntimeError(
            "preclaim finalization timeout is not positive"
        )
    action_timings: list[dict[str, float | str]] = []

    def read_state() -> tuple[
        PreclaimFinalizationEvidenceState,
        dict[str, Any],
        dict[str, Any],
    ]:
        discovery = _start_preclaim_finalization_action(
            "state_discovery", timeout_seconds
        )
        state_value = _read_preclaim_finalizer_state(
            attempt_root=attempt_root,
            intent_publication=intent_publication,
            launch_receipt=launch_receipt,
            launch_receipt_identity=launch_receipt_identity,
            live_gate_owner_seal=live_gate_owner_seal,
            live_consumer_owner_seal=live_consumer_owner_seal,
            launch_terminal_path=launch_terminal_path,
            action=discovery,
        )
        _finish_preclaim_finalization_action(discovery)
        return state_value

    def run_action(
        action_name: str,
        expected_states: set[PreclaimFinalizationEvidenceState],
        operation: Callable[
            [
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
            ],
            None,
        ],
    ) -> tuple[
        PreclaimFinalizationEvidenceState,
        dict[str, Any],
        dict[str, Any],
    ]:
        state_value, gate_value, consumer_value, timing = (
            _run_preclaim_finalization_action(
                action_name=action_name,
                timeout_seconds=timeout_seconds,
                expected_states=expected_states,
                attempt_root=attempt_root,
                intent_publication=intent_publication,
                launch_receipt=launch_receipt,
                launch_receipt_identity=launch_receipt_identity,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                launch_terminal_path=launch_terminal_path,
                operation=operation,
            )
        )
        if timing is not None:
            action_timings.append(timing)
        return state_value, gate_value, consumer_value

    def final_result(
        state: PreclaimFinalizationEvidenceState,
        gate_owner: Mapping[str, Any],
        consumer_owner: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": state.value,
            "gate_owner_seal": dict(gate_owner),
            "consumer_owner_seal": dict(consumer_owner),
            "action_timings": list(action_timings),
        }

    state, gate_owner, consumer_owner = read_state()
    if state is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return final_result(state, gate_owner, consumer_owner)

    if state is PreclaimFinalizationEvidenceState.INTENT_ONLY:
        state, gate_owner, consumer_owner = run_action(
            "exact_wrapper_termination",
            {PreclaimFinalizationEvidenceState.INTENT_ONLY},
            lambda _action, _gate, _consumer: (
                _terminate_exact_wrapper_child(
                    attempt_root / "wrapper_started.json",
                    live_gate_owner_seal,
                )
            ),
        )
    if state is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return final_result(state, gate_owner, consumer_owner)

    if state is PreclaimFinalizationEvidenceState.INTENT_ONLY:
        def wait_gate(
            action: Mapping[str, Any],
            _gate: Mapping[str, Any],
            _consumer: Mapping[str, Any],
        ) -> None:
            _wait_preclaim_gate_terminal_and_dead(
                launch_receipt=launch_receipt,
                launch_terminal_path=launch_terminal_path,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                action=action,
            )

        state, gate_owner, consumer_owner = run_action(
            "gate_terminal_and_dead",
            {PreclaimFinalizationEvidenceState.INTENT_ONLY},
            wait_gate,
        )
    if state is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return final_result(state, gate_owner, consumer_owner)

    if state is PreclaimFinalizationEvidenceState.GATE_EVIDENCE:
        def read_formal_gate(
            _action: Mapping[str, Any],
            gate: Mapping[str, Any],
            _consumer: Mapping[str, Any],
        ) -> None:
            _read_formal_gate_lifecycle_status(
                attempt_root=attempt_root,
                pane=gate,
            )

        state, gate_owner, consumer_owner = run_action(
            "formal_gate_lifecycle",
            {PreclaimFinalizationEvidenceState.GATE_EVIDENCE},
            read_formal_gate,
        )
    if state is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return final_result(state, gate_owner, consumer_owner)

    if state in {
        PreclaimFinalizationEvidenceState.GATE_EVIDENCE,
        PreclaimFinalizationEvidenceState.CONTROLLER_CLEANUP_PRESENT,
    }:
        def wait_consumer_chain(
            action: Mapping[str, Any],
            _gate: Mapping[str, Any],
            _consumer: Mapping[str, Any],
        ) -> None:
            _wait_preclaim_controller_cleanup(
                launch_receipt=launch_receipt,
                launch_terminal_path=launch_terminal_path,
                live_gate_owner_seal=live_gate_owner_seal,
                live_consumer_owner_seal=live_consumer_owner_seal,
                action=action,
            )
            dead_gate, dead_consumer = (
                _wait_preclaim_consumer_terminal_and_dead(
                    launch_receipt=launch_receipt,
                    launch_terminal_path=launch_terminal_path,
                    live_gate_owner_seal=live_gate_owner_seal,
                    live_consumer_owner_seal=(
                        live_consumer_owner_seal
                    ),
                    action=action,
                )
            )
            _read_formal_gate_lifecycle_status(
                attempt_root=attempt_root,
                pane=dead_gate,
            )
            _read_formal_consumer_lifecycle_status(
                attempt_path=Path(
                    str(
                        launch_receipt["pane_fault_consumer"][
                            "artifacts"
                        ]["attempt"]
                    )
                ),
                pane=dead_consumer,
            )

        state, gate_owner, consumer_owner = run_action(
            "consumer_terminal_chain_and_formal_dead",
            {
                PreclaimFinalizationEvidenceState.GATE_EVIDENCE,
                (
                    PreclaimFinalizationEvidenceState
                    .CONTROLLER_CLEANUP_PRESENT
                ),
            },
            wait_consumer_chain,
        )
    if state is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return final_result(state, gate_owner, consumer_owner)

    if state in {
        PreclaimFinalizationEvidenceState.CONSUMER_TERMINAL_CHAIN,
        PreclaimFinalizationEvidenceState.CONSUMER_JOIN_PRESENT,
    }:
        def join_and_cleanup(
            _action: Mapping[str, Any],
            _gate: Mapping[str, Any],
            _consumer: Mapping[str, Any],
        ) -> None:
            join_pane_fault_consumer(
                attempt_path=Path(
                    str(
                        launch_receipt["pane_fault_consumer"][
                            "artifacts"
                        ]["attempt"]
                    )
                ),
                config=config,
                timeout_seconds=timeout_seconds,
            )

        state, gate_owner, consumer_owner = run_action(
            "consumer_join_and_cleanup",
            {
                (
                    PreclaimFinalizationEvidenceState
                    .CONSUMER_TERMINAL_CHAIN
                ),
                PreclaimFinalizationEvidenceState.CONSUMER_JOIN_PRESENT,
            },
            join_and_cleanup,
        )
    if state is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return final_result(state, gate_owner, consumer_owner)

    if (
        state
        is PreclaimFinalizationEvidenceState.CONSUMER_CLEANUP_PRESENT
    ):
        def publish_terminal(
            _action: Mapping[str, Any],
            gate: Mapping[str, Any],
            consumer: Mapping[str, Any],
        ) -> None:
            _publish_preclaim_launch_terminal_v2(
                attempt_root=attempt_root,
                intent_publication=intent_publication,
                launch_receipt=launch_receipt,
                launch_receipt_identity=launch_receipt_identity,
                gate_owner_seal=gate,
                consumer_owner_seal=consumer,
                launch_terminal_path=launch_terminal_path,
            )

        state, gate_owner, consumer_owner = run_action(
            "launch_terminal_v2_publication",
            {
                (
                    PreclaimFinalizationEvidenceState
                    .CONSUMER_CLEANUP_PRESENT
                )
            },
            publish_terminal,
        )
    if state is PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL:
        return final_result(state, gate_owner, consumer_owner)

    raise RuntimeError("preclaim finalization state is unsupported")


def launch_preflight(
    *,
    repo_root: Path,
    config: Path,
    campaign_root: Path,
    policy_sha256: str,
    python: str,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    attempt_id: str | None = None,
    owner_nonce: str | None = None,
    observer_suffix: str | None = None,
    wrapper_arguments_override: Sequence[str] | None = None,
    tmux_binary: str = "tmux",
) -> dict[str, Any]:
    started_at = _utc_now()
    repo_root = repo_root.resolve()
    config = config.resolve()
    campaign_root = campaign_root.resolve()
    policy_sha256 = _require_hex64(policy_sha256, "policy SHA256")
    attempt_id = _require_hex64(
        secrets.token_hex(32) if attempt_id is None else attempt_id,
        "launch attempt ID",
    )
    owner_nonce = _require_hex64(
        secrets.token_hex(32) if owner_nonce is None else owner_nonce,
        "controller owner nonce",
    )
    observer_suffix = _require_hex64(
        secrets.token_hex(32)
        if observer_suffix is None
        else observer_suffix,
        "observer session suffix",
    )
    if _legacy_archive_id_exists(campaign_root, attempt_id):
        raise RuntimeError(
            "launch attempt ID collides with a legacy archive ID"
        )
    observer_session = OBSERVER_SESSION_PREFIX + observer_suffix
    launcher_path = Path(__file__).resolve()
    wrapper_path = (
        repo_root / "scripts/run_canonical_preflight_wrapper.py"
    ).resolve()
    controller_path = (
        repo_root / "scripts/run_canonical_checkpoint_screening.py"
    ).resolve()
    launch_attempts_root = _ensure_secure_leaf_directories(
        campaign_root, ("preflight_launch_attempts",)
    )
    started_root = _ensure_secure_leaf_directories(
        launch_attempts_root, ("started",)
    )
    _ensure_secure_leaf_directories(
        launch_attempts_root, ("setup_terminals",)
    )
    policy_attempt_root = _ensure_secure_leaf_directories(
        launch_attempts_root,
        ("by_policy", policy_sha256),
    )
    started_registry_path = (
        started_root / f"{attempt_id}.json"
    )
    started_registry = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_started_registry_v1"
        ),
        "attempt_id": attempt_id,
        "policy_sha256": policy_sha256,
        "reserved_at": started_at,
    }
    started_registry["launch_started_registry_sha256"] = (
        _canonical_digest(
            started_registry, "launch_started_registry_sha256"
        )
    )
    attempt_root = policy_attempt_root / attempt_id
    try:
        _write_exclusive(started_registry_path, started_registry)
    except BaseException as exc:
        _propagate_launcher_publish_error(exc)
        if started_registry_path.is_file():
            _publish_setup_terminal(
                campaign_root=campaign_root,
                attempt_id=attempt_id,
                policy_sha256=policy_sha256,
                started_registry_path=started_registry_path,
                attempt_root=attempt_root,
                stage="started_registry_create_or_fsync",
                failure=exc,
                started_at=started_at,
            )
        raise
    try:
        _ensure_secure_leaf_directories(
            policy_attempt_root,
            (attempt_id,),
            final_must_be_new=True,
        )
    except BaseException as exc:
        _publish_setup_terminal(
            campaign_root=campaign_root,
            attempt_id=attempt_id,
            policy_sha256=policy_sha256,
            started_registry_path=started_registry_path,
            attempt_root=attempt_root,
            stage="attempt_root_create_or_fsync",
            failure=exc,
            started_at=started_at,
        )
        raise
    try:
        verified_implementations = _install_verified_preflight_apis(
            config
        )
        git = _verified_git_state(repo_root)
        bindings = _verified_bindings(
            repo_root,
            config,
            launcher_path,
            wrapper_path,
            controller_path,
        )
        python_binding = _file_binding(Path(python).resolve())
        if (
            bindings["verified_loader"]["path"]
            != verified_implementations["verified_loader"]["path"]
            or bindings["verified_loader"]["sha256"]
            != verified_implementations["verified_loader"]["sha256"]
            or bindings["preflight_launch_contract"]["path"]
            != verified_implementations[
                "preflight_launch_contract"
            ]["path"]
            or bindings["preflight_launch_contract"]["sha256"]
            != verified_implementations[
                "preflight_launch_contract"
            ]["sha256"]
        ):
            raise RuntimeError(
                "verified implementation bindings differ"
            )
    except BaseException as exc:
        _publish_setup_terminal(
            campaign_root=campaign_root,
            attempt_id=attempt_id,
            policy_sha256=policy_sha256,
            started_registry_path=started_registry_path,
            attempt_root=attempt_root,
            stage="verified_implementation_bootstrap",
            failure=exc,
            started_at=started_at,
        )
        raise
    receipt_path = attempt_root / "launch_receipt.json"
    terminal_path = attempt_root / "launch_terminal.json"
    accepted_path = attempt_root / "launch_accepted.json"
    ownership_release_path = (
        attempt_root / "launch_ownership_release.json"
    )
    gate_ready_path = attempt_root / "pane_gate_ready.json"
    tmux_started_path = attempt_root / "launch_tmux_started.json"
    wrapper_started_path = attempt_root / "wrapper_started.json"
    gate_execution_terminal_path = (
        attempt_root / "gate_execution_terminal.json"
    )
    release_path = attempt_root / "pane_gate_release.json"
    log_path = attempt_root / "pane.log"
    fault_channel_path = attempt_root / "wrapper_fault.channel"
    pane_gate_fault_channel_path = (
        attempt_root / "pane_gate_fault.channel"
    )
    gate_lifecycle_wait_channel_path = (
        attempt_root / "gate_lifecycle_wait.channel"
    )
    gate_wait_supervisor_ready_path = (
        attempt_root / "gate_wait_supervisor_ready.json"
    )
    pane_fault_consumer_registration = (
        _build_pane_fault_consumer_registration(
            attempt_root=attempt_root,
            launcher_binding=bindings["launcher"],
        )
    )
    consumer_namespace = _ensure_secure_leaf_directories(
        attempt_root,
        ("pane_fault_consumer",),
        final_must_be_new=True,
    )
    if (
        str(consumer_namespace)
        != pane_fault_consumer_registration["namespace"]
    ):
        raise RuntimeError(
            "presealed consumer namespace registration differs"
        )
    consumer_lifecycle_wait_channel_path = Path(
        pane_fault_consumer_registration["artifacts"][
            "lifecycle_wait_channel"
        ]
    )
    consumer_wait_supervisor_ready_path = Path(
        pane_fault_consumer_registration["artifacts"][
            "wait_supervisor_ready"
        ]
    )
    consumer_session = PANE_FAULT_CONSUMER_SESSION_PREFIX + attempt_id
    consumer_owner_nonce = secrets.token_hex(32)
    claim_path = (
        campaign_root
        / "by_policy"
        / policy_sha256
        / "preflight_control/wrapper_claim.json"
    )
    wrapper_arguments = list(
        wrapper_arguments_override
        if wrapper_arguments_override is not None
        else [
            python,
            "-B",
            "-u",
            str(wrapper_path),
            "--repo-root",
            str(repo_root),
            "--config",
            str(config),
            "--campaign-root",
            str(campaign_root),
            "--policy-sha256",
            policy_sha256,
            "--python",
            python,
            "--launch-receipt",
            str(receipt_path),
            "--attempt-id",
            attempt_id,
            "--launch-accepted",
            str(accepted_path),
            "--launch-release",
            str(ownership_release_path),
            "--pane-log",
            str(log_path),
        ]
    )
    if not wrapper_arguments or any(
        not isinstance(item, str) or not item
        for item in wrapper_arguments
    ):
        raise RuntimeError("wrapper argument token array is invalid")
    try:
        fault_channel = _create_fault_channel(
            fault_channel_path
        )
        pane_gate_fault_channel = _create_fault_channel(
            pane_gate_fault_channel_path
        )
        gate_lifecycle_wait_channel = _create_fault_channel(
            gate_lifecycle_wait_channel_path
        )
        consumer_lifecycle_wait_channel = _create_fault_channel(
            consumer_lifecycle_wait_channel_path
        )
    except BaseException as exc:
        _publish_setup_terminal(
            campaign_root=campaign_root,
            attempt_id=attempt_id,
            policy_sha256=policy_sha256,
            started_registry_path=started_registry_path,
            attempt_root=attempt_root,
            stage="fault_channels_create_or_fsync",
            failure=exc,
            started_at=started_at,
        )
        raise
    try:
        log_descriptor = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            os.fsync(log_descriptor)
        finally:
            os.close(log_descriptor)
        _fsync_directory(attempt_root)
    except BaseException as exc:
        _publish_setup_terminal(
            campaign_root=campaign_root,
            attempt_id=attempt_id,
            policy_sha256=policy_sha256,
            started_registry_path=started_registry_path,
            attempt_root=attempt_root,
            stage="pane_log_create_or_fsync",
            failure=exc,
            started_at=started_at,
        )
        raise
    pane_log_identity = _file_identity(log_path)
    gate_arguments = [
        python,
        "-B",
        "-u",
        str(launcher_path),
        PANE_GATE_MODE,
        "--attempt-root",
        str(attempt_root),
        "--release-path",
        str(release_path),
        "--log-path",
        str(log_path),
        "--wrapper-arguments-json",
        json.dumps(wrapper_arguments, separators=(",", ":")),
    ]
    gate_wait_supervisor_arguments = [
        python,
        "-B",
        "-u",
        str(launcher_path),
        GATE_WAIT_SUPERVISOR_MODE,
        "--launch-receipt",
        str(receipt_path),
        "--attempt-id",
        attempt_id,
        "--wait-channel-path",
        str(gate_lifecycle_wait_channel_path),
        "--gate-worker-arguments-json",
        json.dumps(gate_arguments, separators=(",", ":")),
    ]
    tmux_arguments = [
        tmux_binary,
        "new-session",
        "-d",
        "-s",
        CONTROLLER_SESSION,
        "-c",
        str(repo_root),
        "-e",
        f"{TMUX_OWNER_ENV}={owner_nonce}",
        "-e",
        f"{OBSERVER_SESSION_ENV}={observer_session}",
        "-e",
        f"{LAUNCH_RECEIPT_PATH_ENV}={receipt_path}",
        "-e",
        f"{LAUNCH_ACCEPTED_PATH_ENV}={accepted_path}",
        "-e",
        f"{LAUNCH_RELEASE_PATH_ENV}={ownership_release_path}",
        "-e",
        f"{PANE_LOG_PATH_ENV}={log_path}",
        *gate_wait_supervisor_arguments,
    ]
    verified_implementations = _reverify_verified_preflight_apis()
    consumer_worker_arguments = [
        python,
        "-B",
        "-u",
        str(launcher_path),
        PANE_FAULT_CONSUMER_MODE,
        "--attempt-path",
        pane_fault_consumer_registration["artifacts"]["attempt"],
        "--config",
        str(config),
    ]
    consumer_wait_supervisor_arguments = [
        python,
        "-B",
        "-u",
        str(launcher_path),
        CONSUMER_WAIT_SUPERVISOR_MODE,
        "--attempt-path",
        pane_fault_consumer_registration["artifacts"]["attempt"],
        "--config",
        str(config),
        "--wait-channel-path",
        str(consumer_lifecycle_wait_channel_path),
        "--consumer-worker-arguments-json",
        json.dumps(
            consumer_worker_arguments, separators=(",", ":")
        ),
    ]
    consumer_tmux_arguments = [
        tmux_binary,
        "new-session",
        "-d",
        "-s",
        consumer_session,
        "-c",
        str(repo_root),
        "-e",
        f"{TMUX_OWNER_ENV}={consumer_owner_nonce}",
        *consumer_wait_supervisor_arguments,
    ]
    receipt = {
        "schema_version": 4,
        "contract_type": LAUNCH_RECEIPT_CONTRACT_TYPE,
        "attempt_id": attempt_id,
        "started_registry": _json_binding(
            started_registry_path,
            "launch_started_registry_sha256",
        ),
        "policy_sha256": policy_sha256,
        "git": git,
        "bindings": bindings,
        "verified_implementations": verified_implementations,
        "python_executable": python_binding,
        "controller_session": CONTROLLER_SESSION,
        "controller_owner_nonce": owner_nonce,
        "observer_session": observer_session,
        "wrapper_arguments": wrapper_arguments,
        "gate_lifecycle_wait_channel": (
            gate_lifecycle_wait_channel
        ),
        "gate_lifecycle_wait_publisher": {
            **dict(bindings["launcher"]),
            "file_identity": _opened_file_identity(launcher_path),
            "role": "gate_lifecycle_wait_supervisor",
        },
        "gate_lifecycle_wait_supervisor_arguments": (
            gate_wait_supervisor_arguments
        ),
        "gate_lifecycle_wait_supervisor_ready_path": str(
            gate_wait_supervisor_ready_path
        ),
        "gate_lifecycle_wait_status_path": str(
            gate_lifecycle_wait_channel_path
        ),
        "gate_worker_arguments": gate_arguments,
        "consumer_lifecycle_wait_channel": (
            consumer_lifecycle_wait_channel
        ),
        "consumer_lifecycle_wait_publisher": {
            **dict(bindings["launcher"]),
            "file_identity": _opened_file_identity(launcher_path),
            "role": "consumer_lifecycle_wait_supervisor",
        },
        "consumer_lifecycle_wait_supervisor_arguments": (
            consumer_wait_supervisor_arguments
        ),
        "consumer_lifecycle_wait_supervisor_ready_path": str(
            consumer_wait_supervisor_ready_path
        ),
        "consumer_lifecycle_wait_status_path": str(
            consumer_lifecycle_wait_channel_path
        ),
        "consumer_worker_arguments": consumer_worker_arguments,
        "consumer_session": consumer_session,
        "consumer_owner_nonce": consumer_owner_nonce,
        "consumer_tmux_arguments": consumer_tmux_arguments,
        "tmux_arguments": tmux_arguments,
        "shell": False,
        "pane_log": pane_log_identity,
        "fault_channel": fault_channel,
        "pane_gate_fault_channel": pane_gate_fault_channel,
        "pane_gate_fault_publisher": {
            **dict(bindings["launcher"]),
            "role": "launcher_pane_gate",
        },
        "pane_fault_consumer": pane_fault_consumer_registration,
        "wrapper_claim_path": str(claim_path),
        "wrapper_started_path": str(wrapper_started_path),
        "gate_execution_terminal_path": str(
            gate_execution_terminal_path
        ),
        "started_at": started_at,
    }
    receipt["launch_receipt_sha256"] = _canonical_digest(
        receipt, "launch_receipt_sha256"
    )
    validate_launch_receipt_schema(
        receipt,
        expected_gate_worker_arguments=gate_arguments,
        expected_consumer_worker_arguments=(
            consumer_worker_arguments
        ),
        label="launcher launch receipt v4",
    )
    try:
        _write_exclusive(receipt_path, receipt)
    except BaseException as exc:
        _propagate_launcher_publish_error(exc)
        _publish_setup_terminal(
            campaign_root=campaign_root,
            attempt_id=attempt_id,
            policy_sha256=policy_sha256,
            started_registry_path=started_registry_path,
            attempt_root=attempt_root,
            stage="launch_receipt_create_or_fsync",
            failure=exc,
            started_at=started_at,
        )
        raise
    receipt_identity = _opened_file_identity(receipt_path)
    pane_gate_fault_descriptor = _open_presealed_fault_channel(
        attempt_root,
        pane_gate_fault_channel,
        name="pane_gate_fault.channel",
    )

    def require_empty_pane_gate_fault_channel() -> None:
        try:
            snapshot = _read_fault_channel(
                pane_gate_fault_descriptor,
                pane_gate_fault_channel,
                attempt_id=attempt_id,
                owner_nonce=owner_nonce,
                launch_receipt_sha256=str(
                    receipt["launch_receipt_sha256"]
                ),
                publisher=dict(
                    receipt["pane_gate_fault_publisher"]
                ),
            )
        except BaseException as exc:
            raise LauncherGateFaultError(
                "pane_gate_fault_channel_invalid",
                snapshot=None,
                failure=exc,
            ) from exc
        if snapshot.get("state") == "valid_fault":
            raise LauncherGateFaultError(
                "pane_gate_typed_publish_failure",
                snapshot=snapshot,
            )
        if snapshot.get("state") != "empty":
            raise LauncherGateFaultError(
                "pane_gate_fault_channel_nonempty",
                snapshot=snapshot,
            )

    client: dict[str, Any] | None = None
    pane: dict[str, Any] | None = None
    owner_seal: dict[str, Any] | None = None
    pane_fault_consumer: dict[str, Any] | None = None
    pane_fault_consumer_chain: dict[str, Any] | None = None
    launcher_gate_reader = {
        "descriptor": pane_gate_fault_descriptor,
        "closed": False,
    }
    post_handoff_finalized = False

    def poison_consumer_after_failure(
        failure: BaseException,
    ) -> BaseException:
        poison: BaseException = failure
        if pane_fault_consumer is not None:
            poison = _poison_and_cleanup_pane_fault_consumer(
                pane_fault_consumer, failure
            )
        if not launcher_gate_reader["closed"]:
            try:
                os.close(int(launcher_gate_reader["descriptor"]))
                launcher_gate_reader["closed"] = True
            except BaseException as close_exc:
                if isinstance(
                    poison, PaneFaultConsumerReservationError
                ):
                    poison.add_secondary_failure(
                        stage="close_launcher_gate_reader",
                        failure=close_exc,
                    )
                else:
                    poison = PaneFaultConsumerReservationError(
                        failure
                    )
                    poison.add_secondary_failure(
                        stage="close_launcher_gate_reader",
                        failure=close_exc,
                    )
        return poison
    try:
        require_empty_pane_gate_fault_channel()
        if (
            _reverify_verified_preflight_apis()
            != receipt["verified_implementations"]
        ):
            raise RuntimeError(
                "verified implementations changed before tmux launch"
            )
        result = subprocess.run(
            tmux_arguments,
            capture_output=True,
            text=True,
            shell=False,
        )
        client = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode != 0:
            return _publish_terminal(
                terminal_path,
                receipt_path=receipt_path,
                receipt_identity=receipt_identity,
                status="tmux_launch_failed",
                failure_type="TmuxLaunchFailed",
                message=result.stderr.strip() or result.stdout.strip(),
                client=client,
                pane=None,
                tmux_started_path=None,
                log_path=log_path,
                session_residual=_tmux_pane(CONTROLLER_SESSION)
                is not None,
                started_at=started_at,
            )
        deadline = time.monotonic() + startup_timeout_seconds
        while True:
            require_empty_pane_gate_fault_channel()
            if (
                gate_wait_supervisor_ready_path.is_file()
                and gate_ready_path.is_file()
            ):
                break
            pane = _tmux_pane(CONTROLLER_SESSION)
            if pane is None or pane["pane_dead"]:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        pane = _tmux_pane(CONTROLLER_SESSION)
        require_empty_pane_gate_fault_channel()
        if pane is None:
            return _publish_terminal(
                terminal_path,
                receipt_path=receipt_path,
                receipt_identity=receipt_identity,
                status="pane_absent_before_release",
                failure_type="PaneAbsent",
                message="tmux pane disappeared before gate release",
                client=client,
                pane=None,
                tmux_started_path=None,
                log_path=log_path,
                session_residual=False,
                started_at=started_at,
            )
        owner_seal = _tmux_owner_seal(
            CONTROLLER_SESSION, owner_nonce
        )
        if (
            pane["pane_dead"]
            or not gate_wait_supervisor_ready_path.is_file()
            or not gate_ready_path.is_file()
        ):
            _cleanup_exact_attempt(
                session=CONTROLLER_SESSION,
                owner_nonce=owner_nonce,
                owner_seal=owner_seal,
                wrapper_started_path=wrapper_started_path,
            )
            return _publish_terminal(
                terminal_path,
                receipt_path=receipt_path,
                receipt_identity=receipt_identity,
                status="pane_failed_before_release",
                failure_type="PaneGateFailed",
                message=(
                    "gate wait supervisor or gate worker did not "
                    "publish ready evidence"
                ),
                client=client,
                pane=pane,
                tmux_started_path=None,
                log_path=log_path,
                session_residual=False,
                started_at=started_at,
            )
        supervisor_ready = _validate_gate_wait_supervisor_ready(
            gate_wait_supervisor_ready_path,
            receipt_path=receipt_path,
            receipt_identity=receipt_identity,
            receipt=receipt,
        )
        gate_ready = _load_json(gate_ready_path, "pane gate ready")
        verified_implementations = (
            _verified_implementations_from_receipt(receipt_path)
        )
        validate_gate_ready(
            gate_ready,
            verified_implementations=verified_implementations,
            label="launcher pane gate ready",
        )
        if (
            gate_ready.get("launch_receipt")
            != _json_binding(receipt_path, "launch_receipt_sha256")
            or gate_ready.get("launch_receipt_identity")
            != receipt_identity
            or _opened_file_identity(receipt_path)
            != receipt_identity
            or gate_ready.get("wrapper_arguments") != wrapper_arguments
            or gate_ready.get("process")
            != supervisor_ready["gate_worker_process"]
            or _process_identity(
                int(gate_ready["process"]["pid"])
            )
            != gate_ready["process"]
            or _process_command_bytes(
                int(gate_ready["process"]["pid"])
            )
            != _command_bytes(gate_arguments)
            or _process_executable(
                int(gate_ready["process"]["pid"])
            )
            != supervisor_ready["gate_worker_executable"]
            or gate_ready.get("pane_gate_ready_sha256")
            != _canonical_digest(
                gate_ready, "pane_gate_ready_sha256"
            )
        ):
            raise RuntimeError("pane gate ready contract differs")
        if (
            owner_seal["pane"] != pane["pane"]
            or owner_seal["pane_pid"] != pane["pane_pid"]
            or owner_seal["pane_process"]
            != supervisor_ready["supervisor_process"]
            or owner_seal != supervisor_ready["owner_seal"]
            or owner_seal["pane_dead"]
        ):
            raise RuntimeError("launcher exact tmux owner seal differs")
        _set_remain_on_exit(str(pane["pane"]), True)
        _verify_remain_on_exit(str(pane["pane"]), "on")
        tmux_server = owner_seal["tmux_server"]
        tmux_owner_seal = build_pane_owner_seal(
            server_pid=tmux_server["server_pid"],
            server_start_ticks=tmux_server["server_process"][
                "start_ticks"
            ],
            socket_path=tmux_server["socket_path"],
            socket_device=tmux_server["socket_device"],
            socket_inode=tmux_server["socket_inode"],
            session=pane["session"],
            pane=pane["pane"],
            pane_pid=pane["pane_pid"],
            pane_process=supervisor_ready["supervisor_process"],
            owner_nonce=owner_seal["owner_nonce"],
            tmux_identity=pane,
            tmux_server=tmux_server,
        )
        tmux_started = build_tmux_started(
            launch_receipt=_json_binding(
                receipt_path, "launch_receipt_sha256"
            ),
            launch_receipt_identity=receipt_identity,
            verified_implementations=verified_implementations,
            pane_gate_ready=_json_binding(
                gate_ready_path, "pane_gate_ready_sha256"
            ),
            tmux_client=client,
            owner_seal=tmux_owner_seal,
            started_at=_utc_now(),
            tmux_identity=pane,
            tmux_server=tmux_server,
        )
        _write_exclusive(tmux_started_path, tmux_started)
        pane_fault_consumer = (
            _reserve_spawn_ready_pane_fault_consumer(
                repo_root=repo_root,
                config=config,
                attempt_root=attempt_root,
                policy_sha256=policy_sha256,
                attempt_id=attempt_id,
                receipt_path=receipt_path,
                receipt_identity=receipt_identity,
                gate_owner_seal=owner_seal,
                pane_fault_channel=pane_gate_fault_channel,
                pane_fault_publisher=dict(
                    receipt["pane_gate_fault_publisher"]
                ),
                python=python,
                ready_timeout_seconds=startup_timeout_seconds,
                registration=receipt["pane_fault_consumer"],
            )
        )
        transfer = _transfer_pane_fault_consumer(
            consumer=pane_fault_consumer,
            launcher_gate_reader=launcher_gate_reader,
            timeout_seconds=startup_timeout_seconds,
        )
        pane_fault_consumer["transfer"] = transfer
        pane_fault_consumer_chain = build_pane_fault_consumer_chain(
            consumer_started=_json_binding(
                Path(str(pane_fault_consumer["started_path"])),
                "consumer_started_sha256",
            ),
            consumer_active=_json_binding(
                Path(str(transfer["active_path"])),
                "consumer_active_sha256",
            ),
            consumer_reader_release=_json_binding(
                Path(str(transfer["reader_release_path"])),
                "consumer_reader_release_sha256",
            ),
            consumer_release_observed=_json_binding(
                Path(str(transfer["release_observed_path"])),
                "consumer_release_observed_sha256",
            ),
            registration=receipt["pane_fault_consumer"],
        )
        _require_post_handoff_pane_fault_consumer(
            pane_fault_consumer, "pane gate release"
        )
        release = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_pane_gate_release_v1"
            ),
            "launch_receipt": _json_binding(
                receipt_path, "launch_receipt_sha256"
            ),
            "launch_receipt_identity": receipt_identity,
            "verified_implementations": (
                _verified_implementations_from_receipt(receipt_path)
            ),
            "pane_gate_ready": _json_binding(
                gate_ready_path, "pane_gate_ready_sha256"
            ),
            "tmux_started": _json_binding(
                tmux_started_path, "launch_tmux_started_sha256"
            ),
            "wrapper_arguments": wrapper_arguments,
            "pane_fault_consumer_chain": (
                pane_fault_consumer_chain
            ),
            "released_at": _utc_now(),
        }
        release["pane_gate_release_sha256"] = _canonical_digest(
            release, "pane_gate_release_sha256"
        )
        _write_exclusive(release_path, release)
        while True:
            assert pane_fault_consumer is not None
            assert pane_fault_consumer_chain is not None
            _require_post_handoff_pane_fault_consumer(
                pane_fault_consumer, "launch ownership return"
            )
            pane = _tmux_pane(CONTROLLER_SESSION)
            if claim_path.is_file():
                if pane is None or pane["pane_dead"]:
                    break
                current_owner = _tmux_owner_seal(
                    CONTROLLER_SESSION, owner_nonce
                )
                if current_owner != owner_seal:
                    raise RuntimeError(
                        "tmux gate owner changed before acceptance"
                    )
                if (
                    _process_identity(int(pane["pane_pid"]))
                    != supervisor_ready["supervisor_process"]
                    or _process_command_bytes(int(pane["pane_pid"]))
                    != _command_bytes(
                        gate_wait_supervisor_arguments
                    )
                    or _process_executable(
                        int(pane["pane_pid"])
                    )["path"]
                    != python_binding["path"]
                ):
                    raise RuntimeError(
                        "tmux gate process seal changed before acceptance"
                    )
                wrapper_started = _validate_wrapper_started(
                    wrapper_started_path,
                    receipt_binding=_json_binding(
                        receipt_path, "launch_receipt_sha256"
                    ),
                    receipt_identity=receipt_identity,
                    verified_implementations=verified_implementations,
                    gate_ready_binding=_json_binding(
                        gate_ready_path, "pane_gate_ready_sha256"
                    ),
                    gate_process=gate_ready["process"],
                    wrapper_arguments=wrapper_arguments,
                )
                _validate_wrapper_claim(
                    claim_path,
                    policy_sha256=policy_sha256,
                    config_binding=bindings["config"],
                    receipt_binding=_json_binding(
                        receipt_path, "launch_receipt_sha256"
                    ),
                    receipt_identity=receipt_identity,
                    verified_implementations=verified_implementations,
                    gate_ready_binding=_json_binding(
                        gate_ready_path, "pane_gate_ready_sha256"
                    ),
                    tmux_started_binding=_json_binding(
                        tmux_started_path,
                        "launch_tmux_started_sha256",
                    ),
                    wrapper_started=wrapper_started,
                    wrapper_started_binding=_json_binding(
                        wrapper_started_path,
                        "wrapper_started_sha256",
                    ),
                    wrapper_arguments=wrapper_arguments,
                    pane_log_identity=pane_log_identity,
                    git=git,
                    supervisor_pid=int(
                        supervisor_ready[
                            "supervisor_process"
                        ]["pid"]
                    ),
                    gate_pid=int(gate_ready["process"]["pid"]),
                    pane_fault_consumer_chain=(
                        pane_fault_consumer_chain
                    ),
                )
                if _opened_file_identity(receipt_path) != receipt_identity:
                    raise RuntimeError(
                        "launch receipt identity changed before acceptance"
                    )
                accepted = _publish_accepted(
                    accepted_path,
                    receipt_path=receipt_path,
                    receipt_identity=receipt_identity,
                    claim_path=claim_path,
                    tmux_started_path=tmux_started_path,
                    pane=pane,
                    log_path=log_path,
                    started_at=started_at,
                    pane_fault_consumer_chain=(
                        pane_fault_consumer_chain
                    ),
                )
                ownership_terminal = _publish_ownership_terminal(
                    terminal_path,
                    receipt_path=receipt_path,
                    receipt_identity=receipt_identity,
                    accepted_path=accepted_path,
                    tmux_started_path=tmux_started_path,
                    claim_path=claim_path,
                    pane=pane,
                    log_path=log_path,
                    started_at=started_at,
                    pane_fault_consumer_chain=(
                        pane_fault_consumer_chain
                    ),
                )
                _verify_remain_on_exit(str(pane["pane"]), "on")
                ownership_release = _publish_ownership_release(
                    ownership_release_path,
                    receipt_path=receipt_path,
                    receipt_identity=receipt_identity,
                    accepted_path=accepted_path,
                    terminal_path=terminal_path,
                    claim_path=claim_path,
                    pane_fault_consumer_chain=(
                        pane_fault_consumer_chain
                    ),
                )
                validate_ownership_chain(
                    accepted,
                    ownership_terminal,
                    ownership_release,
                    receipt_binding=_json_binding(
                        receipt_path, "launch_receipt_sha256"
                    ),
                    receipt_identity=receipt_identity,
                    wrapper_binding=_json_binding(
                        claim_path, "wrapper_claim_sha256"
                    ),
                    accepted_binding=_json_binding(
                        accepted_path, "launch_accepted_sha256"
                    ),
                    terminal_binding=_json_binding(
                        terminal_path, "launch_terminal_sha256"
                    ),
                    verified_implementations=(
                        _verified_implementations_from_receipt(
                            receipt_path
                        )
                    ),
                    pane_fault_consumer_chain=(
                        pane_fault_consumer_chain
                    ),
                    label="published preflight launch ownership chain",
                )
                return ownership_release
            if gate_execution_terminal_path.is_file() and (
                pane is None or pane["pane_dead"]
            ):
                assert pane_fault_consumer is not None
                execution = (
                    _continue_wrapper_early_exit_from_durable_cleanup(
                    attempt_root=attempt_root,
                    launch_receipt=receipt,
                    launch_receipt_path=receipt_path,
                    launch_receipt_identity=receipt_identity,
                    gate_ready_path=gate_ready_path,
                    live_gate_owner_seal=owner_seal,
                    expected_consumer_chain=(
                        pane_fault_consumer_chain
                    ),
                    pane_fault_consumer=pane_fault_consumer,
                    config=config,
                    deadline=deadline,
                    startup_timeout_seconds=(
                        startup_timeout_seconds
                    ),
                    )
                )
                if not launcher_gate_reader["closed"]:
                    os.close(
                        int(launcher_gate_reader["descriptor"])
                    )
                    launcher_gate_reader["closed"] = True
                post_handoff_finalized = True
                return _publish_terminal(
                    terminal_path,
                    receipt_path=receipt_path,
                    receipt_identity=receipt_identity,
                    status="wrapper_exited_before_claim",
                    failure_type="WrapperEarlyExit",
                    message=(
                        "wrapper process ended before a durable claim: "
                        f"{execution['exit_kind']}"
                    ),
                    client=client,
                    pane=pane,
                    tmux_started_path=tmux_started_path,
                    log_path=log_path,
                    session_residual=False,
                    started_at=started_at,
                    gate_execution=execution,
                )
            if pane is None or pane["pane_dead"]:
                break
            if time.monotonic() >= deadline:
                _cleanup_exact_attempt(
                    session=CONTROLLER_SESSION,
                    owner_nonce=owner_nonce,
                    owner_seal=owner_seal,
                    wrapper_started_path=wrapper_started_path,
                )
                raise PaneFaultConsumerReservationError(
                    RuntimeError(
                        "wrapper claim was not published in time"
                    )
                )
            time.sleep(0.02)
        _cleanup_exact_attempt(
            session=CONTROLLER_SESSION,
            owner_nonce=owner_nonce,
            owner_seal=owner_seal,
            wrapper_started_path=wrapper_started_path,
        )
        raise PaneFaultConsumerReservationError(
            RuntimeError(
                "wrapper process exited before publishing a durable claim"
            )
        )
    except (
        LauncherExclusivePublishError,
        LauncherTerminalPublishError,
        LauncherGateFaultError,
        PaneFaultConsumerReservationError,
    ) as exc:
        if (
            post_handoff_finalized
            and isinstance(exc, LauncherTerminalPublishError)
        ):
            raise
        try:
            cleanup_owner_seal = owner_seal
            if (
                cleanup_owner_seal is None
                and _tmux_pane(CONTROLLER_SESSION) is not None
            ):
                cleanup_owner_seal = _tmux_owner_seal(
                    CONTROLLER_SESSION, owner_nonce
                )
            _kill_exact_session(
                CONTROLLER_SESSION,
                owner_nonce,
                cleanup_owner_seal,
            )
        except BaseException as cleanup_exc:
            exc.add_secondary_failure(
                stage="external_tmux_cleanup",
                failure=cleanup_exc,
            )
        poison = poison_consumer_after_failure(exc)
        raise poison from exc
    except BaseException as exc:
        exc = poison_consumer_after_failure(exc)
        session_residual: bool | None
        cleanup_failure: dict[str, str] | None = None
        try:
            _cleanup_exact_attempt(
                session=CONTROLLER_SESSION,
                owner_nonce=owner_nonce,
                owner_seal=owner_seal,
                wrapper_started_path=wrapper_started_path,
            )
            session_residual = _tmux_pane(CONTROLLER_SESSION) is not None
        except BaseException as cleanup_exc:
            session_residual = (
                _tmux_pane(CONTROLLER_SESSION) is not None
            )
            cleanup_failure = {
                "type": type(cleanup_exc).__name__,
                "message": str(cleanup_exc),
            }
        if terminal_path.exists():
            post_terminal = {
                "schema_version": 1,
                "contract_type": (
                    "safa_canonical_preflight_launch_post_terminal_"
                    "failure_v1"
                ),
                "launch_receipt": _json_binding(
                    receipt_path, "launch_receipt_sha256"
                ),
                "launch_terminal": _json_binding(
                    terminal_path, "launch_terminal_sha256"
                ),
                "status": "launch_release_failed",
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "cleanup_failure": cleanup_failure,
                },
                "pane_log": _optional_file_binding(log_path),
                "session_residual": session_residual,
                "failed_at": _utc_now(),
            }
            post_terminal[
                "launch_post_terminal_failure_sha256"
            ] = _canonical_digest(
                post_terminal,
                "launch_post_terminal_failure_sha256",
            )
            _write_exclusive(
                attempt_root / "launch_post_terminal_failure.json",
                post_terminal,
            )
            return post_terminal
        return _publish_terminal(
            terminal_path,
            receipt_path=receipt_path,
            receipt_identity=receipt_identity,
            status="launcher_failed",
            failure_type=type(exc).__name__,
            message=(
                str(exc)
                if cleanup_failure is None
                else f"{exc}; cleanup={cleanup_failure}"
            ),
            client=client,
            pane=pane,
            tmux_started_path=(
                tmux_started_path
                if tmux_started_path.is_file()
                else None
            ),
            log_path=log_path,
            session_residual=session_residual,
            started_at=started_at,
        )
    finally:
        if not launcher_gate_reader["closed"]:
            _launcher_checked_close(
                int(launcher_gate_reader["descriptor"]),
                "launcher pane fault gate reader",
            )
            launcher_gate_reader["closed"] = True


def archive_untracked_failure(
    *,
    campaign_root: Path,
    policy_sha256: str,
    attempt_id: str,
    controller_owner_nonce: str,
    observer_session: str,
    occurred_at: str,
) -> dict[str, Any]:
    policy_sha256 = _require_hex64(policy_sha256, "policy SHA256")
    attempt_id = _require_hex64(attempt_id, "archive attempt ID")
    controller_owner_nonce = _require_hex64(
        controller_owner_nonce, "controller owner nonce"
    )
    if not re.fullmatch(
        rf"{re.escape(OBSERVER_SESSION_PREFIX)}[0-9a-f]{{64}}",
        observer_session,
    ):
        raise RuntimeError("archived observer session is invalid")
    exact_campaign_root = campaign_root.resolve()
    launch_attempts_root = _ensure_secure_leaf_directories(
        exact_campaign_root, ("preflight_launch_attempts",)
    )
    started_root = _ensure_secure_leaf_directories(
        launch_attempts_root, ("started",)
    )
    policy_attempt_root = _ensure_secure_leaf_directories(
        launch_attempts_root,
        ("by_policy", policy_sha256),
    )
    started_registry_path = (
        started_root / f"{attempt_id}.json"
    )
    started_registry = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_started_registry_v1"
        ),
        "attempt_id": attempt_id,
        "policy_sha256": policy_sha256,
        "reserved_at": occurred_at,
        "archived_untracked_attempt": True,
    }
    started_registry["launch_started_registry_sha256"] = (
        _canonical_digest(
            started_registry, "launch_started_registry_sha256"
        )
    )
    _write_exclusive(started_registry_path, started_registry)
    attempt_root = _ensure_secure_leaf_directories(
        policy_attempt_root,
        (attempt_id,),
        final_must_be_new=True,
    )
    policy_root = (
        campaign_root.resolve() / "by_policy" / policy_sha256
    )

    def count(pattern: str) -> int:
        return sum(1 for path in policy_root.glob(pattern) if path.is_file())

    value = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_untracked_launch_failure_v1"
        ),
        "attempt_id": attempt_id,
        "started_registry": _json_binding(
            started_registry_path,
            "launch_started_registry_sha256",
        ),
        "policy_sha256": policy_sha256,
        "controller_session": CONTROLLER_SESSION,
        "controller_owner_nonce": controller_owner_nonce,
        "observer_session": observer_session,
        "status": "failed_before_durable_wrapper_claim",
        "failure": {
            "type": "UntrackedTmuxLaunchFailure",
            "message": (
                "tmux new-session returned zero but the session disappeared "
                "before a wrapper claim; the legacy launch path captured no "
                "pane log or exact remote argv"
            ),
        },
        "evidence_limitations": {
            "wrapper_python_entry_proven": False,
            "exact_remote_argv_recoverable": False,
            "error_text_recoverable": False,
            "pane_log_present": False,
            "launch_receipt_present": False,
        },
        "scientific_execution": {
            "preflight_results": count(
                "checkpoint_preflight/results/*.json"
            ),
            "attempt_claims": count(
                "checkpoint_preflight/attempt_claims/**/*.json"
            ),
            "attempt_terminals": count(
                "checkpoint_preflight/attempt_terminals/**/*.json"
            ),
            "generated_png": count("**/*.png"),
            "run_requests": count("**/run_requests/*.json"),
        },
        "occurred_at": occurred_at,
        "archived_at": _utc_now(),
    }
    if any(value["scientific_execution"].values()):
        raise RuntimeError(
            "untracked failure archive is not zero scientific execution"
        )
    value["untracked_launch_failure_sha256"] = _canonical_digest(
        value, "untracked_launch_failure_sha256"
    )
    path = attempt_root / "untracked_launch_failure.json"
    _write_exclusive(path, value)
    return {
        "archive": _json_binding(
            path, "untracked_launch_failure_sha256"
        )
    }


def _legacy_absence_snapshot(
    policy_root: Path, files: Sequence[Path]
) -> dict[str, Any]:
    relatives = [
        path.relative_to(policy_root).as_posix() for path in files
    ]
    results = sum(
        relative.startswith("checkpoint_preflight/results/")
        and relative.endswith(".json")
        for relative in relatives
    )
    claims = sum(
        (
            relative.startswith(
                "checkpoint_preflight/attempt_claims/"
            )
            or (
                relative.startswith("preflight_control/attempts/")
                and relative.endswith(".claim.json")
            )
        )
        and relative.endswith(".json")
        for relative in relatives
    )
    terminals = sum(
        (
            relative.startswith(
                "checkpoint_preflight/attempt_terminals/"
            )
            or (
                relative.startswith("preflight_control/attempts/")
                and relative.endswith(".terminal.json")
            )
        )
        and relative.endswith(".json")
        for relative in relatives
    )
    control = sum(
        relative.startswith("preflight_control/")
        for relative in relatives
    )
    png = sum(
        relative.lower().endswith(".png") for relative in relatives
    )
    run_requests = sum(
        "run_requests" in Path(relative).parts
        and relative.endswith(".json")
        for relative in relatives
    )
    snapshot = {
        "policy_root": str(policy_root),
        "preflight_results": results,
        "attempt_claims": claims,
        "attempt_terminals": terminals,
        "preflight_control_files": control,
        "generated_png": png,
        "run_requests": run_requests,
        "scientific_execution": 0,
        "scientific_execution_started": False,
    }
    if any(
        snapshot[field]
        for field in (
            "preflight_results",
            "attempt_claims",
            "attempt_terminals",
            "preflight_control_files",
            "generated_png",
            "run_requests",
            "scientific_execution",
        )
    ):
        raise RuntimeError(
            "legacy archive evidence is not zero scientific execution"
        )
    return snapshot


def _binding_without_mtime(
    binding: Mapping[str, Any],
) -> dict[str, str]:
    return dict(
        path=str(binding["path"]),
        sha256=str(binding["sha256"]),
        canonical_sha256=str(binding["canonical_sha256"]),
    )


def _build_legacy_failure_immutable_evidence(
    *,
    campaign_root: Path,
    policy_sha256: str,
    controller_owner_nonce: str,
    observer_session: str,
    prepare_completion_path: Path,
    prepare_completion_file_sha256: str,
    prepare_completion_canonical_sha256: str,
    prepare_completion_mtime_ns: int,
    old_policy_tree_sha256: str,
) -> dict[str, Any]:
    campaign_root = _require_exact_directory(
        campaign_root, "legacy archive campaign root"
    )
    policy_sha256 = _require_hex64(
        policy_sha256, "legacy policy SHA256"
    )
    controller_owner_nonce = _require_hex64(
        controller_owner_nonce, "reported controller owner nonce"
    )
    old_policy_tree_sha256 = _require_hex64(
        old_policy_tree_sha256, "legacy policy tree SHA256"
    )
    prepare_completion_file_sha256 = _require_hex64(
        prepare_completion_file_sha256,
        "prepare completion file SHA256",
    )
    prepare_completion_canonical_sha256 = _require_hex64(
        prepare_completion_canonical_sha256,
        "prepare completion canonical SHA256",
    )
    if (
        type(prepare_completion_mtime_ns) is not int
        or prepare_completion_mtime_ns <= 0
    ):
        raise RuntimeError("prepare completion mtime_ns is invalid")
    if not re.fullmatch(
        rf"{re.escape(OBSERVER_SESSION_PREFIX)}[0-9a-f]{{64}}",
        observer_session,
    ):
        raise RuntimeError("reported observer session is invalid")
    policy_root = (
        campaign_root / "by_policy" / policy_sha256
    )
    _require_exact_directory(
        policy_root, "legacy prepared policy root"
    )
    expected_manifest_path = (
        policy_root
        / "checkpoint_preflight"
        / "preflight_request_manifest.json"
    )
    manifest, manifest_binding = _read_exact_json_artifact(
        prepare_completion_path,
        expected_path=expected_manifest_path,
        canonical_field="preflight_request_manifest_sha256",
        label="legacy prepare completion manifest",
    )
    if (
        manifest_binding["sha256"]
        != prepare_completion_file_sha256
        or manifest_binding["canonical_sha256"]
        != prepare_completion_canonical_sha256
        or manifest_binding["mtime_ns"]
        != prepare_completion_mtime_ns
    ):
        raise RuntimeError(
            "legacy prepare completion binding differs"
        )
    if (
        manifest.get("contract_type")
        != "safa_canonical_preflight_request_manifest_v1"
        or manifest.get("policy_sha256") != policy_sha256
        or type(manifest.get("request_count")) is not int
        or manifest["request_count"] <= 0
        or not isinstance(manifest.get("requests"), list)
        or manifest["request_count"] != len(manifest["requests"])
    ):
        raise RuntimeError(
            "legacy prepare completion manifest differs"
        )
    raw_plan_binding = manifest.get("checkpoint_plan")
    if (
        not isinstance(raw_plan_binding, Mapping)
        or len(raw_plan_binding) != 3
        or "path" not in raw_plan_binding
        or "sha256" not in raw_plan_binding
        or "canonical_sha256" not in raw_plan_binding
    ):
        raise RuntimeError("legacy checkpoint plan binding differs")
    expected_plan_path = policy_root / "checkpoint_plan.json"
    plan_path = Path(str(raw_plan_binding["path"]))
    plan, plan_binding_with_mtime = _read_exact_json_artifact(
        plan_path,
        expected_path=expected_plan_path,
        canonical_field="checkpoint_plan_sha256",
        label="legacy checkpoint plan",
    )
    plan_binding = _binding_without_mtime(
        plan_binding_with_mtime
    )
    if plan_binding != dict(raw_plan_binding):
        raise RuntimeError(
            "legacy checkpoint plan file binding differs"
        )
    counts = plan.get("counts")
    plan_requests = plan.get("preflight_requests")
    if (
        plan.get("contract_type")
        != "safa_canonical_checkpoint_plan_v1"
        or plan.get("policy_sha256") != policy_sha256
        or not isinstance(counts, Mapping)
        or not isinstance(plan_requests, list)
        or counts.get("preflight_requests")
        != manifest["request_count"]
        or counts.get("distinct_checkpoint_sha256")
        != manifest["request_count"]
        or counts.get("distinct_raw_checkpoint_sha256")
        + counts.get("distinct_ema_checkpoint_sha256")
        != manifest["request_count"]
    ):
        raise RuntimeError("legacy checkpoint plan differs")
    plan_by_digest = {
        request.get("preflight_request_sha256"): request
        for request in plan_requests
        if isinstance(request, Mapping)
    }
    if len(plan_by_digest) != manifest["request_count"]:
        raise RuntimeError(
            "legacy checkpoint plan request set differs"
        )
    request_root = (
        policy_root / "checkpoint_preflight" / "requests"
    )
    _require_exact_directory(
        request_root, "legacy preflight request root"
    )
    normalized_requests: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_binding in manifest["requests"]:
        if (
            not isinstance(raw_binding, Mapping)
            or set(raw_binding)
            != {
                "checkpoint_model",
                "checkpoint_sha256",
                "path",
                "preflight_request_sha256",
                "sha256",
            }
        ):
            raise RuntimeError(
                "legacy preflight request manifest binding differs"
            )
        checkpoint_sha256 = _require_hex64(
            str(raw_binding["checkpoint_sha256"]),
            "legacy checkpoint SHA256",
        )
        checkpoint_model = raw_binding["checkpoint_model"]
        if checkpoint_model not in {"raw", "ema"}:
            raise RuntimeError(
                "legacy checkpoint model selector differs"
            )
        key = (checkpoint_sha256, str(checkpoint_model))
        if key in seen:
            raise RuntimeError(
                "legacy preflight request selector is duplicated"
            )
        seen.add(key)
        expected_request_path = (
            request_root
            / f"{checkpoint_sha256}__{checkpoint_model}.json"
        )
        request_path = Path(str(raw_binding["path"]))
        request, request_binding = _read_exact_json_artifact(
            request_path,
            expected_path=expected_request_path,
            canonical_field="preflight_request_sha256",
            label="legacy preflight request",
        )
        if (
            request_binding["sha256"] != raw_binding["sha256"]
            or request_binding["canonical_sha256"]
            != raw_binding["preflight_request_sha256"]
            or request.get("policy_sha256") != policy_sha256
            or request.get("checkpoint_sha256")
            != checkpoint_sha256
            or request.get("checkpoint_model")
            != checkpoint_model
            or plan_by_digest.get(
                request_binding["canonical_sha256"]
            )
            != request
        ):
            raise RuntimeError(
                "legacy preflight request evidence differs"
            )
        normalized_requests.append(
            {
                "checkpoint_model": str(checkpoint_model),
                "checkpoint_sha256": checkpoint_sha256,
                "path": str(request_binding["path"]),
                "sha256": str(request_binding["sha256"]),
                "canonical_sha256": str(
                    request_binding["canonical_sha256"]
                ),
            }
        )
    normalized_requests.sort(
        key=lambda item: (
            item["checkpoint_sha256"],
            item["checkpoint_model"],
        )
    )
    raw_count = sum(
        item["checkpoint_model"] == "raw"
        for item in normalized_requests
    )
    ema_count = sum(
        item["checkpoint_model"] == "ema"
        for item in normalized_requests
    )
    if (
        raw_count != counts["distinct_raw_checkpoint_sha256"]
        or ema_count != counts["distinct_ema_checkpoint_sha256"]
    ):
        raise RuntimeError(
            "legacy preflight request selector counts differ"
        )
    request_set_payload = {
        "schema_version": 1,
        "derivation": LEGACY_REQUEST_SET_DERIVATION,
        "requests": normalized_requests,
    }
    tree_before = _legacy_policy_tree_snapshot(policy_root)
    if tree_before["sha256"] != old_policy_tree_sha256:
        raise RuntimeError("legacy prepared policy tree SHA256 differs")
    files = _exact_regular_tree_files(policy_root)
    absence = _legacy_absence_snapshot(policy_root, files)
    tree_after = _legacy_policy_tree_snapshot(policy_root)
    if tree_after != tree_before:
        raise RuntimeError(
            "legacy prepared policy tree changed during capture"
        )
    return {
        "schema_version": 2,
        "contract_type": LEGACY_FAILURE_EVIDENCE_CONTRACT_TYPE,
        "policy_sha256": policy_sha256,
        "original_attempt_id": None,
        "original_started_registry": None,
        "launch_receipt": None,
        "wrapper_claim": None,
        "failure_stage": "launch_before_ownership",
        "evidence_level": "operator_observed_unsealed",
        "reported_invocation_identifiers": {
            "controller_session": CONTROLLER_SESSION,
            "controller_owner_nonce": controller_owner_nonce,
            "observer_session": observer_session,
        },
        "occurrence_time": {
            "exact": None,
            "not_before": {
                "relation": "strictly_after_prepare_completion",
                "prepare_completion_artifact": dict(
                    manifest_binding
                ),
            },
            "not_after": None,
            "precision": "lower_bound_only_or_unknown",
        },
        "prepared_policy": {
            "policy_sha256": policy_sha256,
            "checkpoint_plan": plan_binding,
            "preflight_request_manifest": (
                _binding_without_mtime(manifest_binding)
            ),
            "request_set": {
                "derivation": LEGACY_REQUEST_SET_DERIVATION,
                "sha256": _canonical_payload_sha256(
                    request_set_payload
                ),
                "request_count": len(normalized_requests),
                "raw_count": raw_count,
                "ema_count": ema_count,
            },
            "counts": dict(counts),
            "policy_tree": tree_before,
        },
        "absence_snapshot": absence,
    }


def _derive_legacy_failure_archive_id(
    immutable_evidence: Mapping[str, Any],
) -> str:
    return _canonical_payload_sha256(
        {
            "schema_version": 1,
            "derivation": LEGACY_FAILURE_ARCHIVE_ID_DERIVATION,
            "immutable_evidence": dict(immutable_evidence),
        }
    )


def validate_legacy_untracked_failure_archive_v2(
    raw: Any,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("legacy failure archive is not a mapping")
    value = dict(raw)
    if set(value) != {
        "schema_version",
        "contract_type",
        "archive_id",
        "archive_id_derivation",
        "immutable_evidence",
        "archived_at",
        "legacy_failure_archive_sha256",
    }:
        raise RuntimeError("legacy failure archive keys differ")
    evidence = value["immutable_evidence"]
    derivation = value["archive_id_derivation"]
    if (
        value["schema_version"] != 2
        or value["contract_type"]
        != LEGACY_FAILURE_ARCHIVE_CONTRACT_TYPE
        or not isinstance(evidence, Mapping)
        or not isinstance(derivation, Mapping)
        or derivation
        != {
            "schema_version": 1,
            "algorithm": "sha256_canonical_json_lf",
            "derivation": LEGACY_FAILURE_ARCHIVE_ID_DERIVATION,
        }
        or value["archive_id"]
        != _derive_legacy_failure_archive_id(evidence)
        or not isinstance(value["archived_at"], str)
        or not value["archived_at"]
        or value["legacy_failure_archive_sha256"]
        != _canonical_digest(
            value, "legacy_failure_archive_sha256"
        )
    ):
        raise RuntimeError("legacy failure archive relation differs")
    if set(evidence) != {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "original_attempt_id",
        "original_started_registry",
        "launch_receipt",
        "wrapper_claim",
        "failure_stage",
        "evidence_level",
        "reported_invocation_identifiers",
        "occurrence_time",
        "prepared_policy",
        "absence_snapshot",
    }:
        raise RuntimeError("legacy failure evidence keys differ")
    identifiers = evidence["reported_invocation_identifiers"]
    occurrence = evidence["occurrence_time"]
    prepared = evidence["prepared_policy"]
    absence = evidence["absence_snapshot"]
    if (
        evidence["schema_version"] != 2
        or evidence["contract_type"]
        != LEGACY_FAILURE_EVIDENCE_CONTRACT_TYPE
        or not HEX64.fullmatch(str(evidence["policy_sha256"]))
        or evidence["original_attempt_id"] is not None
        or evidence["original_started_registry"] is not None
        or evidence["launch_receipt"] is not None
        or evidence["wrapper_claim"] is not None
        or evidence["failure_stage"] != "launch_before_ownership"
        or evidence["evidence_level"]
        != "operator_observed_unsealed"
        or not isinstance(identifiers, Mapping)
        or set(identifiers)
        != {
            "controller_session",
            "controller_owner_nonce",
            "observer_session",
        }
        or identifiers["controller_session"] != CONTROLLER_SESSION
        or not HEX64.fullmatch(
            str(identifiers["controller_owner_nonce"])
        )
        or not re.fullmatch(
            rf"{re.escape(OBSERVER_SESSION_PREFIX)}[0-9a-f]{{64}}",
            str(identifiers["observer_session"]),
        )
        or not isinstance(occurrence, Mapping)
        or set(occurrence)
        != {"exact", "not_before", "not_after", "precision"}
        or occurrence.get("exact") is not None
        or occurrence.get("not_after") is not None
        or occurrence.get("precision")
        != "lower_bound_only_or_unknown"
        or not isinstance(occurrence.get("not_before"), Mapping)
        or set(occurrence["not_before"])
        != {"relation", "prepare_completion_artifact"}
        or occurrence["not_before"].get("relation")
        != "strictly_after_prepare_completion"
        or not isinstance(
            occurrence["not_before"].get(
                "prepare_completion_artifact"
            ),
            Mapping,
        )
        or not isinstance(prepared, Mapping)
        or set(prepared)
        != {
            "policy_sha256",
            "checkpoint_plan",
            "preflight_request_manifest",
            "request_set",
            "counts",
            "policy_tree",
        }
        or prepared.get("policy_sha256")
        != evidence["policy_sha256"]
        or not isinstance(absence, Mapping)
        or set(absence)
        != {
            "policy_root",
            "preflight_results",
            "attempt_claims",
            "attempt_terminals",
            "preflight_control_files",
            "generated_png",
            "run_requests",
            "scientific_execution",
            "scientific_execution_started",
        }
        or absence.get("scientific_execution_started") is not False
        or any(
            absence.get(field) != 0
            for field in (
                "preflight_results",
                "attempt_claims",
                "attempt_terminals",
                "preflight_control_files",
                "generated_png",
                "run_requests",
                "scientific_execution",
            )
        )
    ):
        raise RuntimeError("legacy failure evidence relation differs")
    completion = occurrence["not_before"][
        "prepare_completion_artifact"
    ]
    plan_binding = prepared["checkpoint_plan"]
    manifest_binding = prepared["preflight_request_manifest"]
    request_set = prepared["request_set"]
    counts = prepared["counts"]
    policy_tree = prepared["policy_tree"]
    if (
        set(completion)
        != {
            "path",
            "sha256",
            "canonical_sha256",
            "mtime_ns",
        }
        or not isinstance(completion["path"], str)
        or not completion["path"].startswith("/")
        or not HEX64.fullmatch(str(completion["sha256"]))
        or not HEX64.fullmatch(
            str(completion["canonical_sha256"])
        )
        or type(completion["mtime_ns"]) is not int
        or completion["mtime_ns"] <= 0
        or not isinstance(plan_binding, Mapping)
        or len(plan_binding) != 3
        or "path" not in plan_binding
        or "sha256" not in plan_binding
        or "canonical_sha256" not in plan_binding
        or not isinstance(manifest_binding, Mapping)
        or len(manifest_binding) != 3
        or "path" not in manifest_binding
        or "sha256" not in manifest_binding
        or "canonical_sha256" not in manifest_binding
        or any(
            not isinstance(binding["path"], str)
            or not binding["path"].startswith("/")
            or not HEX64.fullmatch(str(binding["sha256"]))
            or not HEX64.fullmatch(
                str(binding["canonical_sha256"])
            )
            for binding in (plan_binding, manifest_binding)
        )
        or completion["path"] != manifest_binding["path"]
        or completion["sha256"] != manifest_binding["sha256"]
        or completion["canonical_sha256"]
        != manifest_binding["canonical_sha256"]
        or not isinstance(request_set, Mapping)
        or set(request_set)
        != {
            "derivation",
            "sha256",
            "request_count",
            "raw_count",
            "ema_count",
        }
        or request_set["derivation"]
        != LEGACY_REQUEST_SET_DERIVATION
        or not HEX64.fullmatch(str(request_set["sha256"]))
        or any(
            type(request_set[field]) is not int
            or request_set[field] < 0
            for field in ("request_count", "raw_count", "ema_count")
        )
        or request_set["request_count"] <= 0
        or request_set["raw_count"] + request_set["ema_count"]
        != request_set["request_count"]
        or not isinstance(counts, Mapping)
        or counts.get("preflight_requests")
        != request_set["request_count"]
        or counts.get("distinct_checkpoint_sha256")
        != request_set["request_count"]
        or counts.get("distinct_raw_checkpoint_sha256")
        != request_set["raw_count"]
        or counts.get("distinct_ema_checkpoint_sha256")
        != request_set["ema_count"]
        or not isinstance(policy_tree, Mapping)
        or set(policy_tree)
        != {
            "derivation",
            "sha256",
            "file_count",
            "content_bytes",
            "serialized_bytes",
            "symlink_count",
        }
        or policy_tree["derivation"]
        != LEGACY_POLICY_TREE_DERIVATION
        or not HEX64.fullmatch(str(policy_tree["sha256"]))
        or any(
            type(policy_tree[field]) is not int
            or policy_tree[field] < 0
            for field in (
                "file_count",
                "content_bytes",
                "serialized_bytes",
                "symlink_count",
            )
        )
        or policy_tree["file_count"] <= 0
        or policy_tree["symlink_count"] != 0
        or not isinstance(absence["policy_root"], str)
        or not absence["policy_root"].startswith("/")
    ):
        raise RuntimeError(
            "legacy failure prepare completion binding differs"
        )
    policy_root = Path(absence["policy_root"])
    if (
        Path(plan_binding["path"])
        != policy_root / "checkpoint_plan.json"
        or Path(manifest_binding["path"])
        != (
            policy_root
            / "checkpoint_preflight"
            / "preflight_request_manifest.json"
        )
        or policy_root.name != evidence["policy_sha256"]
        or ".." in policy_root.parts
    ):
        raise RuntimeError(
            "legacy failure prepared policy path differs"
        )
    return value


def archive_legacy_untracked_failure_v2(
    *,
    campaign_root: Path,
    policy_sha256: str,
    controller_owner_nonce: str,
    observer_session: str,
    prepare_completion_path: Path,
    prepare_completion_file_sha256: str,
    prepare_completion_canonical_sha256: str,
    prepare_completion_mtime_ns: int,
    old_policy_tree_sha256: str,
    archived_at: str | None = None,
) -> dict[str, Any]:
    immutable_evidence = (
        _build_legacy_failure_immutable_evidence(
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            controller_owner_nonce=controller_owner_nonce,
            observer_session=observer_session,
            prepare_completion_path=prepare_completion_path,
            prepare_completion_file_sha256=(
                prepare_completion_file_sha256
            ),
            prepare_completion_canonical_sha256=(
                prepare_completion_canonical_sha256
            ),
            prepare_completion_mtime_ns=(
                prepare_completion_mtime_ns
            ),
            old_policy_tree_sha256=old_policy_tree_sha256,
        )
    )
    archive_id = _derive_legacy_failure_archive_id(
        immutable_evidence
    )
    archive_directory = _ensure_exact_archive_directory(
        campaign_root, policy_sha256
    )
    path = archive_directory / f"{archive_id}.json"
    value = {
        "schema_version": 2,
        "contract_type": LEGACY_FAILURE_ARCHIVE_CONTRACT_TYPE,
        "archive_id": archive_id,
        "archive_id_derivation": {
            "schema_version": 1,
            "algorithm": "sha256_canonical_json_lf",
            "derivation": LEGACY_FAILURE_ARCHIVE_ID_DERIVATION,
        },
        "immutable_evidence": immutable_evidence,
        "archived_at": _utc_now() if archived_at is None else archived_at,
    }
    value["legacy_failure_archive_sha256"] = _canonical_digest(
        value, "legacy_failure_archive_sha256"
    )
    validate_legacy_untracked_failure_archive_v2(value)
    created = True
    try:
        _write_exclusive(path, value)
    except LauncherExclusivePublishError as exc:
        if exc.commit_state != "collision":
            raise
        existing, existing_binding = _read_exact_json_artifact(
            path,
            expected_path=path,
            canonical_field="legacy_failure_archive_sha256",
            label="legacy failure archive",
        )
        validate_legacy_untracked_failure_archive_v2(existing)
        if (
            existing.get("archive_id") != archive_id
            or existing.get("immutable_evidence")
            != immutable_evidence
        ):
            raise RuntimeError(
                "legacy failure archive ID collision differs"
            )
        value = existing
        created = False
        binding = _binding_without_mtime(existing_binding)
    else:
        binding = _json_binding(
            path, "legacy_failure_archive_sha256"
        )
    return {
        "archive_id": archive_id,
        "archive": binding,
        "created": created,
        "value": value,
    }


def _legacy_archive_id_exists(
    campaign_root: Path, candidate_id: str
) -> bool:
    archive_root = (
        campaign_root
        / "untracked_failure_archives"
        / "by_policy"
    )
    if not archive_root.exists():
        return False
    _require_exact_directory(
        archive_root, "legacy archive policy namespace"
    )
    for entry in os.scandir(archive_root):
        if entry.is_symlink():
            raise RuntimeError(
                "legacy archive namespace contains a symlink"
            )
        if not entry.is_dir(follow_symlinks=False):
            raise RuntimeError(
                "legacy archive namespace contains a non-directory"
            )
        candidate = Path(entry.path) / f"{candidate_id}.json"
        if candidate.exists() or candidate.is_symlink():
            return True
    return False


def _parse_launch_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    return parser.parse_args(argv)


def _parse_gate_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--release-path", required=True, type=Path)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--wrapper-arguments-json", required=True)
    return parser.parse_args(argv)


def _parse_gate_wait_supervisor_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch-receipt", required=True, type=Path
    )
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--wait-channel-path", required=True, type=Path
    )
    parser.add_argument(
        "--gate-worker-arguments-json", required=True
    )
    return parser.parse_args(argv)


def _parse_pane_fault_consumer_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempt-path", required=True, type=Path
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def _parse_consumer_wait_supervisor_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempt-path", required=True, type=Path
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--wait-channel-path", required=True, type=Path
    )
    parser.add_argument(
        "--consumer-worker-arguments-json", required=True
    )
    return parser.parse_args(argv)


def _parse_pane_fault_consumer_join_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempt-path", required=True, type=Path
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--timeout-seconds", required=True, type=float
    )
    return parser.parse_args(argv)


def _parse_archive_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--controller-owner-nonce", required=True)
    parser.add_argument("--observer-session", required=True)
    parser.add_argument("--occurred-at", required=True)
    return parser.parse_args(argv)


def _parse_legacy_archive_v2_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--controller-owner-nonce", required=True)
    parser.add_argument("--observer-session", required=True)
    parser.add_argument(
        "--prepare-completion-path", required=True, type=Path
    )
    parser.add_argument(
        "--prepare-completion-file-sha256", required=True
    )
    parser.add_argument(
        "--prepare-completion-canonical-sha256", required=True
    )
    parser.add_argument(
        "--prepare-completion-mtime-ns", required=True, type=int
    )
    parser.add_argument(
        "--old-policy-tree-sha256", required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == GATE_WAIT_SUPERVISOR_MODE:
        args = _parse_gate_wait_supervisor_args(raw[1:])
        bootstrap_receipt = _load_json(
            args.launch_receipt.resolve(),
            "gate wait supervisor bootstrap receipt",
        )
        bootstrap_bindings = bootstrap_receipt.get("bindings")
        bootstrap_config = (
            bootstrap_bindings.get("config")
            if isinstance(bootstrap_bindings, Mapping)
            else None
        )
        if (
            not isinstance(bootstrap_config, Mapping)
            or set(bootstrap_config) != {"path", "sha256"}
        ):
            raise RuntimeError(
                "gate wait supervisor config binding is malformed"
            )
        config_path = Path(str(bootstrap_config["path"]))
        if (
            _sha256_file(config_path)
            != bootstrap_config["sha256"]
        ):
            raise RuntimeError(
                "gate wait supervisor config SHA-256 differs"
            )
        live_implementations = _install_verified_preflight_apis(
            config_path
        )
        if (
            validate_verified_implementations(
                bootstrap_receipt.get("verified_implementations")
            )
            != live_implementations
        ):
            raise RuntimeError(
                "gate wait supervisor verified implementations differ"
            )
        gate_worker_arguments = json.loads(
            args.gate_worker_arguments_json
        )
        if (
            not isinstance(gate_worker_arguments, list)
            or any(
                not isinstance(item, str) or not item
                for item in gate_worker_arguments
            )
        ):
            raise RuntimeError(
                "gate wait supervisor worker arguments are invalid"
            )
        return _gate_wait_supervisor(
            receipt_path=args.launch_receipt.resolve(),
            attempt_id=args.attempt_id,
            wait_channel_path=args.wait_channel_path.resolve(),
            gate_worker_arguments=gate_worker_arguments,
        )
    if raw and raw[0] == PANE_FAULT_CONSUMER_MODE:
        args = _parse_pane_fault_consumer_args(raw[1:])
        return _pane_fault_consumer(
            attempt_path=args.attempt_path.resolve(),
            config=args.config.resolve(),
        )
    if raw and raw[0] == CONSUMER_WAIT_SUPERVISOR_MODE:
        args = _parse_consumer_wait_supervisor_args(raw[1:])
        consumer_worker_arguments = json.loads(
            args.consumer_worker_arguments_json
        )
        if (
            not isinstance(consumer_worker_arguments, list)
            or any(
                not isinstance(item, str) or not item
                for item in consumer_worker_arguments
            )
        ):
            raise RuntimeError(
                "consumer wait supervisor worker arguments are invalid"
            )
        return _consumer_wait_supervisor(
            attempt_path=args.attempt_path.resolve(),
            config=args.config.resolve(),
            wait_channel_path=args.wait_channel_path.resolve(),
            consumer_worker_arguments=consumer_worker_arguments,
        )
    if raw and raw[0] == PANE_FAULT_CONSUMER_JOIN_MODE:
        args = _parse_pane_fault_consumer_join_args(raw[1:])
        if (
            not args.timeout_seconds > 0
            or args.timeout_seconds > 300
        ):
            raise RuntimeError(
                "pane fault consumer join timeout is outside (0, 300]"
            )
        result = join_pane_fault_consumer(
            attempt_path=args.attempt_path,
            config=args.config,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    if raw and raw[0] == PANE_GATE_MODE:
        args = _parse_gate_args(raw[1:])
        bootstrap_receipt = _load_json(
            args.attempt_root.resolve() / "launch_receipt.json",
            "launch receipt",
        )
        bootstrap_bindings = bootstrap_receipt.get("bindings")
        bootstrap_config = (
            bootstrap_bindings.get("config")
            if isinstance(bootstrap_bindings, Mapping)
            else None
        )
        if (
            not isinstance(bootstrap_config, Mapping)
            or set(bootstrap_config) != {"path", "sha256"}
        ):
            raise RuntimeError(
                "pane gate config binding is malformed"
            )
        config_path = Path(str(bootstrap_config["path"]))
        if (
            _sha256_file(config_path)
            != bootstrap_config["sha256"]
        ):
            raise RuntimeError("pane gate config SHA-256 differs")
        live_implementations = _install_verified_preflight_apis(
            config_path
        )
        if (
            validate_verified_implementations(
                bootstrap_receipt.get("verified_implementations")
            )
            != live_implementations
        ):
            raise RuntimeError(
                "pane gate verified implementations differ"
            )
        wrapper_arguments = json.loads(args.wrapper_arguments_json)
        if not isinstance(wrapper_arguments, list) or any(
            not isinstance(item, str) or not item
            for item in wrapper_arguments
        ):
            raise RuntimeError("pane wrapper arguments are invalid")
        return _pane_gate(
            attempt_root=args.attempt_root.resolve(),
            release_path=args.release_path.resolve(),
            log_path=args.log_path.resolve(),
            wrapper_arguments=wrapper_arguments,
        )
    if raw and raw[0] == ARCHIVE_FAILURE_MODE:
        args = _parse_archive_args(raw[1:])
        result = archive_untracked_failure(
            campaign_root=args.campaign_root,
            policy_sha256=args.policy_sha256,
            attempt_id=args.attempt_id,
            controller_owner_nonce=args.controller_owner_nonce,
            observer_session=args.observer_session,
            occurred_at=args.occurred_at,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if raw and raw[0] == ARCHIVE_LEGACY_FAILURE_V2_MODE:
        args = _parse_legacy_archive_v2_args(raw[1:])
        result = archive_legacy_untracked_failure_v2(
            campaign_root=args.campaign_root,
            policy_sha256=args.policy_sha256,
            controller_owner_nonce=args.controller_owner_nonce,
            observer_session=args.observer_session,
            prepare_completion_path=args.prepare_completion_path,
            prepare_completion_file_sha256=(
                args.prepare_completion_file_sha256
            ),
            prepare_completion_canonical_sha256=(
                args.prepare_completion_canonical_sha256
            ),
            prepare_completion_mtime_ns=(
                args.prepare_completion_mtime_ns
            ),
            old_policy_tree_sha256=args.old_policy_tree_sha256,
        )
        print(
            json.dumps(
                {
                    "archive_id": result["archive_id"],
                    "archive": result["archive"],
                    "created": result["created"],
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    args = _parse_launch_args(raw)
    if (
        not args.startup_timeout_seconds > 0
        or not args.startup_timeout_seconds <= 300
    ):
        raise RuntimeError("startup timeout is outside (0, 300]")
    result = launch_preflight(
        repo_root=args.repo_root,
        config=args.config,
        campaign_root=args.campaign_root,
        policy_sha256=args.policy_sha256,
        python=args.python,
        startup_timeout_seconds=args.startup_timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result.get("status") is None else 125


if __name__ == "__main__":
    raise SystemExit(main())
