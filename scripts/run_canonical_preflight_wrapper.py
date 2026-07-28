#!/usr/bin/env python3
"""Durable process wrapper for the canonical CPU preflight controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from safa.closeout.preflight_launch_contract import (
        FAULT_RECORD_CONTRACT_TYPE,
        LAUNCH_RECEIPT_CONTRACT_TYPE,
        PreflightLaunchContractError,
        build_artifact_binding,
        build_claim_v3,
        build_file_identity,
        build_finalization_secondary_failure,
        build_pane_fault_consumer_chain,
        build_pane_owner_seal,
        build_process_identity,
        build_publish_failure_record,
        build_tmux_server_identity,
        build_verified_implementations,
        validate_artifact_binding,
        validate_executable_identity,
        validate_file_identity,
        validate_gate_ready,
        validate_launch_receipt_schema,
        validate_ownership_chain,
        validate_pane_fault_consumer_registration,
        validate_pane_owner_seal,
        validate_publish_failure_record,
        validate_tmux_server_identity,
        validate_tmux_started,
        validate_verified_implementations,
        validate_wrapper_started,
    )

CONTROLLER_SESSION = "safa-screening-preflight-controller"
OBSERVER_SESSION_PREFIX = "safa-screening-preflight-monitor"
OBSERVER_SESSION_ENV = "SAFA_PREFLIGHT_OBSERVER_SESSION"
_OBSERVER_SESSION_FROM_ENV = os.environ.get(OBSERVER_SESSION_ENV)
if _OBSERVER_SESSION_FROM_ENV is None:
    OBSERVER_SESSION = (
        f"{OBSERVER_SESSION_PREFIX}-{secrets.token_hex(32)}"
    )
elif (
    not _OBSERVER_SESSION_FROM_ENV.startswith(
        f"{OBSERVER_SESSION_PREFIX}-"
    )
    or len(
        _OBSERVER_SESSION_FROM_ENV[
            len(OBSERVER_SESSION_PREFIX) + 1 :
        ]
    )
    != 64
    or any(
        character not in "0123456789abcdef"
        for character in _OBSERVER_SESSION_FROM_ENV[
            len(OBSERVER_SESSION_PREFIX) + 1 :
        ]
    )
):
    raise RuntimeError("preflight observer session environment is invalid")
else:
    OBSERVER_SESSION = _OBSERVER_SESSION_FROM_ENV
OBSERVER_TERMINAL_WAIT_SECONDS = 180.0
OBSERVER_IDENTITY_WAIT_SECONDS = 10.0
PROCESS_TERMINATION_WAIT_SECONDS = 30.0
TMUX_CONDITIONAL_KILL_REJECTED = (
    "SAFA_TMUX_CONDITIONAL_KILL_REJECTED"
)
TMUX_CONDITIONAL_REMAIN_REJECTED = (
    "SAFA_TMUX_CONDITIONAL_REMAIN_REJECTED"
)
TMUX_OWNER_ENV = "SAFA_OWNER_NONCE"
TMUX_OWNER_NONCE_HEX_LENGTH = 64
LAUNCH_RECEIPT_PATH_ENV = "SAFA_PREFLIGHT_LAUNCH_RECEIPT_PATH"
LAUNCH_ACCEPTED_PATH_ENV = "SAFA_PREFLIGHT_LAUNCH_ACCEPTED_PATH"
LAUNCH_RELEASE_PATH_ENV = "SAFA_PREFLIGHT_LAUNCH_RELEASE_PATH"
PANE_LOG_PATH_ENV = "SAFA_PREFLIGHT_PANE_LOG_PATH"
FAULT_CHANNEL_FD_ENV = "SAFA_PREFLIGHT_FAULT_CHANNEL_FD"
FAULT_CHANNEL_MAX_RECORD_BYTES = 65536
FAULT_CHANNEL_PREFIX = b"SAFA-PREFLIGHT-FAULT-V1\n"
FAULT_CHANNEL_SHA_PREFIX = b"sha256:"
FAULT_REPORTED_EXIT_CODE = 123
FAULT_CHANNEL_WRITE_FAILED_EXIT_CODE = 122
FAULT_CHANNEL_CLOSE_FAILED_EXIT_CODE = 121
LAUNCH_ACCEPTED_WAIT_SECONDS = 30.0
OBSERVER_BOOTSTRAP_PATH_ENV = "SAFA_PREFLIGHT_OBSERVER_BOOTSTRAP_PATH"
OBSERVER_BOOTSTRAP_POLICY_ENV = "SAFA_PREFLIGHT_OBSERVER_POLICY_SHA256"
OBSERVER_BOOTSTRAP_WRAPPER_ENV = "SAFA_PREFLIGHT_WRAPPER_CLAIM"
OBSERVER_BOOTSTRAP_NONCE_ENV = "SAFA_PREFLIGHT_OBSERVER_OWNER_NONCE"
OBSERVER_GATE_MODE = "--observer-bootstrap-gate"
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
    "PreflightLaunchContractError",
    "build_artifact_binding",
    "build_claim_v3",
    "build_file_identity",
    "build_finalization_secondary_failure",
    "build_pane_fault_consumer_chain",
    "build_pane_owner_seal",
    "build_publish_failure_record",
    "build_process_identity",
    "build_tmux_server_identity",
    "build_tmux_started",
    "build_verified_implementations",
    "validate_artifact_binding",
    "validate_claim_v3",
    "validate_executable_identity",
    "validate_file_identity",
    "validate_gate_ready",
    "validate_launch_receipt_schema",
    "validate_ownership_chain",
    "validate_pane_fault_consumer_registration",
    "validate_pane_owner_seal",
    "validate_publish_failure_record",
    "validate_tmux_server_identity",
    "validate_tmux_started",
    "validate_verified_implementations",
    "validate_wrapper_started",
)
_VERIFIED_LOADER_HANDLE: dict[str, Any] | None = None
_SHARED_CONTRACT_HANDLE: dict[str, Any] | None = None


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
        or len(str(raw["sha256"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(raw["sha256"])
        )
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
        "preflight wrapper",
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
        "preflight_wrapper",
        "scripts/run_canonical_preflight_wrapper.py",
    )
    _caller_source, caller_identity = _bootstrap_read_file(
        caller_path, caller_sha256, "preflight wrapper"
    )
    if caller_path != Path(__file__).resolve():
        raise RuntimeError("policy does not bind this preflight wrapper")
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
        caller_name="preflight_wrapper",
        caller_relative_path=(
            "scripts/run_canonical_preflight_wrapper.py"
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


class TmuxTargetAbsent(RuntimeError):
    """The requested tmux server/session/pane is explicitly absent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: Mapping[str, Any], excluded: str) -> str:
    return hashlib.sha256(
        _canonical_json({key: item for key, item in value.items() if key != excluded})
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExclusivePublishError(RuntimeError):
    """Fail-closed result for a one-writer durable publication."""

    def __init__(
        self,
        commit_state: str,
        message: str,
        *,
        stage: str,
        directory_seal: Mapping[str, int] | None,
        payload: Mapping[str, Any],
        temporary: Mapping[str, Any] | None,
        error_number: int | None,
    ) -> None:
        if commit_state not in {
            "precommit_failed_clean",
            "durability_unknown_quarantined",
            "committed_cleanup_error",
            "collision",
        }:
            raise ValueError(
                f"exclusive publication commit state is invalid: "
                f"{commit_state}"
            )
        super().__init__(
            f"{commit_state} at {stage}: {message}"
        )
        self.commit_state = commit_state
        self.status = commit_state
        self.stage = stage
        self.directory_seal = (
            None if directory_seal is None else dict(directory_seal)
        )
        self.payload = dict(payload)
        self.temporary = (
            None if temporary is None else dict(temporary)
        )
        self.error_number = error_number
        self.quarantined = commit_state in {
            "durability_unknown_quarantined",
            "committed_cleanup_error",
        }
        self.secondary_failures: list[dict[str, str]] = []

    def add_secondary_failure(
        self, *, stage: str, failure: BaseException
    ) -> None:
        if len(self.secondary_failures) >= 8:
            raise RuntimeError(
                "typed publication secondary failure bound exceeded"
            )
        error_number = getattr(failure, "errno", None)
        if type(error_number) is not int:
            error_number = None
        identity = getattr(failure, "identity", None)
        if not isinstance(identity, Mapping):
            identity = None
        self.secondary_failures.append(
            {
                "stage": stage,
                "type": type(failure).__name__,
                "message": str(failure),
                "errno": error_number,
                "identity": (
                    None if identity is None else dict(identity)
                ),
            }
        )

    def as_record(self) -> dict[str, Any]:
        return build_publish_failure_record(
            commit_state=self.commit_state,
            stage=self.stage,
            message=str(self),
            directory_seal=self.directory_seal,
            payload=self.payload,
            temporary=self.temporary,
            error_number=self.error_number,
            secondary_failures=list(self.secondary_failures),
        )


class _DirectoryDurabilityUnknown(RuntimeError):
    """A directory mutation completed but its fsync failed."""


class _PublicationSealMismatch(RuntimeError):
    """A sealed publication inode changed before a permitted mutation."""


def _checked_close(descriptor: int, label: str) -> None:
    try:
        os.close(descriptor)
    except OSError as exc:
        raise RuntimeError(f"{label} close failed") from exc


def _bind_inherited_fault_channel(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    raw_descriptor = os.environ.get(FAULT_CHANNEL_FD_ENV)
    if (
        raw_descriptor is None
        or not raw_descriptor.isascii()
        or not raw_descriptor.isdecimal()
    ):
        raise RuntimeError(
            "inherited fault channel descriptor is not decimal"
        )
    descriptor = int(raw_descriptor)
    if descriptor <= 2 or str(descriptor) != raw_descriptor:
        raise RuntimeError(
            "inherited fault channel descriptor is not canonical"
        )
    binding = receipt.get("fault_channel")
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
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise RuntimeError(
            "inherited fault channel binding is malformed"
        )
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(
            "inherited fault channel descriptor is not open"
        ) from exc
    observed = {
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
        "mode": int(opened.st_mode),
        "uid": int(opened.st_uid),
        "nlink": int(opened.st_nlink),
        "size": int(opened.st_size),
    }
    if (
        not os.get_inheritable(descriptor)
        or not stat.S_ISREG(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_uid != os.geteuid()
        or opened.st_nlink != 1
        or opened.st_size != 0
        or any(
            observed[name] != binding[name]
            for name in observed
        )
        or binding["sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(
            "inherited fault channel identity differs"
        )
    attempt_id = receipt.get("attempt_id")
    owner_nonce = receipt.get("controller_owner_nonce")
    receipt_sha256 = receipt.get("launch_receipt_sha256")
    publisher = receipt.get("bindings", {}).get("wrapper")
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 64
        or any(character not in "0123456789abcdef" for character in attempt_id)
        or not isinstance(owner_nonce, str)
        or len(owner_nonce) != 64
        or any(character not in "0123456789abcdef" for character in owner_nonce)
        or not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt_sha256
        )
        or not isinstance(publisher, dict)
        or set(publisher) != {"path", "sha256"}
        or not isinstance(publisher.get("path"), str)
        or not isinstance(publisher.get("sha256"), str)
        or len(publisher["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in publisher["sha256"]
        )
    ):
        raise RuntimeError(
            "inherited fault channel receipt binding differs"
        )
    os.set_inheritable(descriptor, False)
    return {
        "descriptor": descriptor,
        "binding": dict(binding),
        "attempt_id": attempt_id,
        "owner_nonce": owner_nonce,
        "launch_receipt_sha256": receipt_sha256,
        "publisher": dict(publisher),
    }


def _fault_channel_frame(payload: bytes) -> bytes:
    payload_sha256 = hashlib.sha256(payload).hexdigest().encode(
        "ascii"
    )
    frame = (
        FAULT_CHANNEL_PREFIX
        + f"{len(payload):08x}\n".encode("ascii")
        + payload
        + FAULT_CHANNEL_SHA_PREFIX
        + payload_sha256
        + b"\n"
    )
    if len(frame) > FAULT_CHANNEL_MAX_RECORD_BYTES:
        raise RuntimeError("fault channel record exceeds bound")
    return frame


_FAULT_CHANNEL_BINDING_KEYS = {
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


def _create_named_fault_channel(path: Path) -> dict[str, Any]:
    directory_descriptor, directory = _open_sealed_directory(
        path.parent
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_descriptor,
        )
        os.fchmod(descriptor, 0o600)
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
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or opened.st_size != 0
        ):
            raise RuntimeError(
                "named fault channel identity differs"
            )
        os.fsync(descriptor)
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
            os.close(descriptor)
        os.close(directory_descriptor)


def _open_named_fault_channel(
    path: Path, binding: Mapping[str, Any]
) -> int:
    directory_descriptor, directory = _open_sealed_directory(
        path.parent
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
            set(binding) != _FAULT_CHANNEL_BINDING_KEYS
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
                "named presealed fault channel differs"
            )
        os.set_inheritable(descriptor, False)
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def _read_named_fault_channel(
    descriptor: int,
    binding: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if (
        before.st_dev != binding.get("device")
        or before.st_ino != binding.get("inode")
        or before.st_uid != binding.get("uid")
        or before.st_mode != binding.get("mode")
        or before.st_nlink != 1
        or before.st_size > FAULT_CHANNEL_MAX_RECORD_BYTES
    ):
        raise RuntimeError(
            "named fault channel changed before read"
        )
    content = os.pread(
        descriptor, FAULT_CHANNEL_MAX_RECORD_BYTES + 1, 0
    )
    after = os.fstat(descriptor)
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_mode != before.st_mode
        or after.st_size != before.st_size
        or len(content) != before.st_size
    ):
        raise RuntimeError(
            "named fault channel changed during read"
        )
    if not content:
        return {"state": "empty", "record": None}
    if (
        len(content) > FAULT_CHANNEL_MAX_RECORD_BYTES
        or not content.startswith(FAULT_CHANNEL_PREFIX)
    ):
        raise RuntimeError("named fault channel frame differs")
    length_start = len(FAULT_CHANNEL_PREFIX)
    length_line = content[length_start : length_start + 9]
    if (
        len(length_line) != 9
        or length_line[-1:] != b"\n"
    ):
        raise RuntimeError(
            "named fault channel length differs"
        )
    payload_start = length_start + 9
    payload_end = payload_start + int(length_line[:8], 16)
    payload = content[payload_start:payload_end]
    trailer = (
        FAULT_CHANNEL_SHA_PREFIX
        + hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"\n"
    )
    if content[payload_end:] != trailer:
        raise RuntimeError(
            "named fault channel trailer differs"
        )
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "named fault channel JSON differs"
        ) from exc
    if (
        not isinstance(record, dict)
        or _canonical_json(record) != payload
        or record.get("contract_type")
        != FAULT_RECORD_CONTRACT_TYPE
        or record.get("fault_channel") != dict(binding)
        or record.get("attempt_id") != context["attempt_id"]
        or record.get("owner_nonce") != context["owner_nonce"]
        or record.get("launch_receipt_sha256")
        != context["launch_receipt_sha256"]
        or record.get("publisher") != context["publisher"]
        or record.get("fault_record_sha256")
        != _canonical_digest(record, "fault_record_sha256")
    ):
        raise RuntimeError(
            "named fault channel record binding differs"
        )
    try:
        validate_publish_failure_record(record["failure"])
    except (KeyError, PreflightLaunchContractError) as exc:
        raise RuntimeError(
            "named fault channel failure differs"
        ) from exc
    return {"state": "valid_fault", "record": record}


def _write_fault_channel_record(
    context: Mapping[str, Any],
    failure: ExclusivePublishError,
) -> dict[str, Any]:
    descriptor = int(context["descriptor"])
    binding = dict(context["binding"])
    record = {
        "schema_version": 1,
        "contract_type": FAULT_RECORD_CONTRACT_TYPE,
        "attempt_id": context["attempt_id"],
        "owner_nonce": context["owner_nonce"],
        "launch_receipt_sha256": context[
            "launch_receipt_sha256"
        ],
        "publisher": dict(context["publisher"]),
        "fault_channel": binding,
        "failure": failure.as_record(),
        "recorded_at": _utc_now(),
    }
    record["fault_record_sha256"] = _canonical_digest(
        record, "fault_record_sha256"
    )
    payload = _canonical_json(record)
    frame = _fault_channel_frame(payload)
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != binding["device"]
        or before.st_ino != binding["inode"]
        or before.st_uid != binding["uid"]
        or before.st_mode != binding["mode"]
        or before.st_nlink != binding["nlink"]
        or before.st_size != 0
    ):
        raise RuntimeError(
            "fault channel changed before single-record write"
        )
    offset = 0
    while offset < len(frame):
        try:
            written = os.pwrite(
                descriptor, frame[offset:], offset
            )
        except InterruptedError:
            continue
        if written <= 0:
            raise RuntimeError(
                "fault channel write made no progress"
            )
        offset += written
    os.fsync(descriptor)
    after = os.fstat(descriptor)
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_uid != before.st_uid
        or after.st_mode != before.st_mode
        or after.st_nlink != before.st_nlink
        or after.st_size != len(frame)
    ):
        raise RuntimeError(
            "fault channel changed after single-record write"
        )
    observed_parts: list[bytes] = []
    offset = 0
    while offset < len(frame):
        try:
            chunk = os.pread(
                descriptor, len(frame) - offset, offset
            )
        except InterruptedError:
            continue
        if not chunk:
            break
        observed_parts.append(chunk)
        offset += len(chunk)
    if b"".join(observed_parts) != frame:
        raise RuntimeError(
            "fault channel readback differs after write"
        )
    return record


def _execute_with_fault_reporting(
    context: dict[str, Any],
    operation: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any] | None, int | None, dict[str, Any] | None]:
    value: dict[str, Any] | None = None
    dedicated_failure_code: int | None = None
    dedicated_failure: dict[str, Any] | None = None
    try:
        value = operation()
    except ExclusivePublishError as exc:
        try:
            _write_fault_channel_record(context, exc)
            dedicated_failure_code = FAULT_REPORTED_EXIT_CODE
            dedicated_failure = {
                "status": "typed_publish_failure_reported",
                "failure": exc.as_record(),
            }
        except BaseException as channel_exc:
            dedicated_failure_code = (
                FAULT_CHANNEL_WRITE_FAILED_EXIT_CODE
            )
            dedicated_failure = {
                "status": "fault_channel_write_failed",
                "failure": exc.as_record(),
                "channel_failure": {
                    "type": type(channel_exc).__name__,
                    "message": str(channel_exc),
                },
            }
    finally:
        descriptor_to_close = int(context["descriptor"])
        context["descriptor"] = -1
        try:
            _checked_close(
                descriptor_to_close, "inherited fault channel"
            )
        except BaseException as close_exc:
            dedicated_failure_code = (
                FAULT_CHANNEL_CLOSE_FAILED_EXIT_CODE
            )
            dedicated_failure = {
                "status": "fault_channel_close_failed",
                "channel_failure": {
                    "type": type(close_exc).__name__,
                    "message": str(close_exc),
                },
            }
    return value, dedicated_failure_code, dedicated_failure


def _open_sealed_directory(path: Path) -> tuple[int, os.stat_result]:
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or path.is_symlink()
    ):
        raise RuntimeError(f"publication directory is not exact: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
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
                f"publication directory identity or permissions differ: {path}"
            )
    except BaseException:
        _checked_close(descriptor, "publication directory")
        raise
    return descriptor, opened


def _ensure_secure_leaf_directories(
    trusted_root: Path,
    relative_parts: Sequence[str],
) -> Path:
    current_path = trusted_root
    current_descriptor, _identity = _open_sealed_directory(
        trusted_root
    )
    try:
        for name in relative_parts:
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
                pass
            flags = (
                os.O_RDONLY
                | os.O_CLOEXEC
                | os.O_NOFOLLOW
                | os.O_DIRECTORY
            )
            child_descriptor = os.open(
                name, flags, dir_fd=current_descriptor
            )
            try:
                if created:
                    os.fchmod(child_descriptor, 0o755)
                child = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(child.st_mode)
                    or child.st_uid != os.geteuid()
                    or stat.S_IMODE(child.st_mode) != 0o755
                ):
                    raise RuntimeError(
                        "secure directory component identity or "
                        "permissions differ"
                    )
                if created:
                    _fsync_dirfd(current_descriptor)
            except BaseException:
                _checked_close(
                    child_descriptor,
                    "secure child directory",
                )
                raise
            _checked_close(
                current_descriptor, "secure parent directory"
            )
            current_descriptor = child_descriptor
            current_path = current_path / name
    finally:
        _checked_close(
            current_descriptor, "secure directory walk"
        )
    return current_path


def _read_dirfd_regular(
    directory_descriptor: int,
    name: str,
    *,
    expected_content: bytes | None = None,
    expected_identity: tuple[int, int] | None = None,
    fsync_file: bool = False,
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
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
                f"publication file identity or permissions differ: {name}"
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
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_mode != after.st_mode
            or before.st_uid != after.st_uid
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(content) != after.st_size
            or (
                expected_content is not None
                and content != expected_content
            )
        ):
            raise RuntimeError(
                f"publication file changed or content differs: {name}"
            )
        if fsync_file:
            os.fsync(descriptor)
    finally:
        _checked_close(descriptor, "publication file")
    return content, before


def _secure_read_file(
    path: Path, *, missing_ok: bool = False
) -> tuple[bytes, os.stat_result] | None:
    directory_descriptor, _directory = _open_sealed_directory(
        path.parent
    )
    try:
        try:
            return _read_dirfd_regular(
                directory_descriptor, path.name
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
    finally:
        _checked_close(
            directory_descriptor, "publication directory"
        )


def _secure_json_snapshot(
    path: Path,
    *,
    digest_field: str | None = None,
    missing_ok: bool = False,
) -> dict[str, Any] | None:
    read = _secure_read_file(path, missing_ok=missing_ok)
    if read is None:
        return None
    content, identity = read
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"secure JSON snapshot is invalid: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"secure JSON snapshot is not a mapping: {path}"
        )
    canonical: str | None = None
    if digest_field is not None:
        canonical = value.get(digest_field)
        if (
            not isinstance(canonical, str)
            or len(canonical) != 64
            or _canonical_digest(value, digest_field) != canonical
        ):
            raise RuntimeError(
                f"secure JSON snapshot digest differs: {path}"
            )
    snapshot = {
        "path": str(path),
        "content": content,
        "value": value,
        "sha256": hashlib.sha256(content).hexdigest(),
        "identity": identity,
    }
    if canonical is not None:
        snapshot["binding"] = build_artifact_binding(
            path=str(path),
            sha256=snapshot["sha256"],
            canonical_sha256=canonical,
        )
    return snapshot


def _fsync_dirfd(descriptor: int) -> None:
    os.fsync(descriptor)


def _cleanup_sealed_temporary(
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
        raise _PublicationSealMismatch(
            "publication temporary identity changed before cleanup"
        )
    expected_content = temporary_seal.get("content")
    if expected_content is not None:
        observed, _identity = _read_dirfd_regular(
            directory_descriptor,
            temporary_name,
            expected_content=expected_content,
            expected_identity=(
                int(temporary_seal["device"]),
                int(temporary_seal["inode"]),
            ),
        )
        if hashlib.sha256(observed).hexdigest() != temporary_seal["sha256"]:
            raise _PublicationSealMismatch(
                "publication temporary content changed before cleanup"
            )
    os.unlink(temporary_name, dir_fd=directory_descriptor)
    try:
        _fsync_dirfd(directory_descriptor)
    except BaseException as exc:
        raise _DirectoryDurabilityUnknown(
            "temporary unlink directory fsync failed"
        ) from exc


def _rollback_uncommitted_final(
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
        raise _PublicationSealMismatch(
            "uncommitted final identity changed before rollback"
        )
    os.unlink(final_name, dir_fd=directory_descriptor)
    try:
        _fsync_dirfd(directory_descriptor)
    except BaseException as exc:
        raise _DirectoryDurabilityUnknown(
            "final rollback directory fsync failed"
        ) from exc


def _write_exclusive(
    path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    content = _canonical_json(dict(value))
    content_sha256 = hashlib.sha256(content).hexdigest()
    payload_seal = {
        "size": len(content),
        "sha256": content_sha256,
    }
    directory_descriptor, directory_identity = (
        _open_sealed_directory(path.parent)
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
    ) -> ExclusivePublishError:
        error_number = (
            cause.errno if isinstance(cause, OSError) else None
        )
        return ExclusivePublishError(
            commit_state,
            message,
            stage=stage,
            directory_seal=directory_seal,
            payload=payload_seal,
            temporary=temporary_seal,
            error_number=error_number,
        )

    def rollback_final(
        stage: str,
        final_identity: tuple[int, int] | None = None,
    ) -> None:
        nonlocal linked, quarantined
        if not linked or temporary_identity is None:
            return
        sealed_final_identity = (
            temporary_identity
            if final_identity is None
            else final_identity
        )
        try:
            _rollback_uncommitted_final(
                directory_descriptor,
                path.name,
                sealed_final_identity,
            )
        except BaseException as exc:
            quarantined = True
            raise failure(
                "durability_unknown_quarantined",
                stage,
                "uncommitted final rollback failed; target "
                "directory is quarantined",
                exc,
            ) from exc
        linked = False

    def cleanup_temporary(
        *,
        stage: str,
        after_commit: bool,
    ) -> None:
        nonlocal temporary_identity, temporary_seal, quarantined
        if temporary_seal is None:
            return
        try:
            _cleanup_sealed_temporary(
                directory_descriptor,
                temporary_name,
                temporary_seal,
            )
        except _PublicationSealMismatch as exc:
            quarantined = True
            raise failure(
                (
                    "committed_cleanup_error"
                    if after_commit
                    else "collision"
                ),
                stage,
                "sealed temporary changed; target directory is "
                "quarantined",
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
                "temporary cleanup failed; target directory is "
                "quarantined",
                exc,
            ) from exc
        temporary_identity = None
        temporary_seal = None

    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
        )
        temporary_descriptor = os.open(
            temporary_name,
            flags,
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
                "publication temporary identity differs"
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
                    "publication write made no progress"
                )
            offset += written
        os.fchmod(temporary_descriptor, 0o644)
        os.fsync(temporary_descriptor)
        descriptor_to_close = temporary_descriptor
        temporary_descriptor = -1
        _checked_close(
            descriptor_to_close, "publication temporary"
        )
        reopened_content, reopened = _read_dirfd_regular(
            directory_descriptor,
            temporary_name,
            expected_content=content,
            expected_identity=temporary_identity,
        )
        if (
            len(reopened_content) != len(content)
            or hashlib.sha256(reopened_content).hexdigest()
            != content_sha256
            or reopened.st_nlink != 1
        ):
            raise RuntimeError(
                "publication temporary verification differs"
            )
        temporary_seal = {
            "device": int(reopened.st_dev),
            "inode": int(reopened.st_ino),
            "ctime_ns": int(reopened.st_ctime_ns),
            "size": int(reopened.st_size),
            "sha256": content_sha256,
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
                    _read_dirfd_regular(
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
                    "final path exists with different or unsafe content",
                    collision,
                ) from collision
            if (
                len(existing_content) != len(content)
                or hashlib.sha256(existing_content).hexdigest()
                != content_sha256
            ):
                raise failure(
                    "collision",
                    "existing_final_compare",
                    "final path exists with different content",
                    exc,
                ) from exc
            try:
                _fsync_dirfd(directory_descriptor)
            except BaseException as sync_failure:
                quarantined = True
                raise failure(
                    "durability_unknown_quarantined",
                    "existing_final_directory_fsync",
                    "exact existing final directory fsync failed",
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
                "payload_sha256": content_sha256,
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
            or (linked_final.st_dev, linked_final.st_ino)
            != temporary_identity
            or temporary_after_link.st_nlink != 2
            or linked_final.st_nlink != 2
            or (temporary_after_link.st_dev, temporary_after_link.st_ino)
            != temporary_identity
        ):
            rollback_final(
                "linked_identity_rollback",
                (linked_final.st_dev, linked_final.st_ino),
            )
            raise failure(
                "precommit_failed_clean",
                "linked_identity_verify",
                "linked final and temporary identity differ",
            )
        try:
            linked_temporary_content, linked_temporary = (
                _read_dirfd_regular(
                    directory_descriptor,
                    temporary_name,
                    expected_content=content,
                    expected_identity=temporary_identity,
                )
            )
        except BaseException as exc:
            rollback_final("linked_temporary_read_rollback")
            raise failure(
                "precommit_failed_clean",
                "linked_temporary_read",
                str(exc),
                exc,
            ) from exc
        if (
            linked_temporary_content != content
            or linked_temporary.st_nlink != 2
        ):
            rollback_final("linked_temporary_compare_rollback")
            raise failure(
                "precommit_failed_clean",
                "linked_temporary_compare",
                "linked temporary content differs",
            )
        temporary_seal = {
            **temporary_seal,
            "ctime_ns": int(linked_temporary.st_ctime_ns),
            "nlink": 2,
        }
        try:
            final_content, final_identity = _read_dirfd_regular(
                directory_descriptor,
                path.name,
                expected_content=content,
                expected_identity=temporary_identity,
            )
        except BaseException as exc:
            rollback_final("linked_final_read_rollback")
            raise failure(
                "precommit_failed_clean",
                "linked_final_read",
                str(exc),
                exc,
            ) from exc
        if final_content != content or final_identity.st_nlink != 2:
            rollback_final("linked_final_compare_rollback")
            raise failure(
                "precommit_failed_clean",
                "linked_final_compare",
                "linked final content differs",
            )
        try:
            _fsync_dirfd(directory_descriptor)
        except BaseException as exc:
            quarantined = True
            raise failure(
                "durability_unknown_quarantined",
                "final_link_directory_fsync",
                "final link directory fsync failed",
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
            "payload_sha256": content_sha256,
            "directory_device": int(directory_identity.st_dev),
            "directory_inode": int(directory_identity.st_ino),
        }
    except BaseException as exc:
        if temporary_descriptor >= 0:
            descriptor_to_close = temporary_descriptor
            temporary_descriptor = -1
            try:
                _checked_close(
                    descriptor_to_close,
                    "publication temporary",
                )
            except BaseException as close_failure:
                if not isinstance(exc, ExclusivePublishError):
                    exc = failure(
                        "precommit_failed_clean",
                        "temporary_close",
                        f"{exc}; close failure: {close_failure}",
                        close_failure,
                    )
        if isinstance(exc, ExclusivePublishError):
            quarantined = quarantined or exc.quarantined
        if not quarantined and linked and not committed:
            try:
                rollback_final("exception_rollback")
            except ExclusivePublishError:
                raise
        if not quarantined and temporary_seal is not None:
            try:
                cleanup_temporary(
                    stage="exception_temporary_cleanup",
                    after_commit=committed,
                )
            except ExclusivePublishError:
                raise
        if isinstance(exc, ExclusivePublishError):
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
            _checked_close(
                descriptor_to_close, "publication directory"
            )
        except BaseException as close_failure:
            if active_failure is not None:
                state = (
                    active_failure.commit_state
                    if isinstance(
                        active_failure, ExclusivePublishError
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
            if committed:
                raise failure(
                    "committed_cleanup_error",
                    "directory_close",
                    f"final committed; directory close failed: "
                    f"{close_failure}",
                    close_failure,
                ) from close_failure
            raise failure(
                "precommit_failed_clean",
                "directory_close",
                str(close_failure),
                close_failure,
            ) from close_failure


def _json_binding(path: Path, digest_field: str) -> dict[str, str]:
    snapshot = _secure_json_snapshot(
        path, digest_field=digest_field
    )
    assert snapshot is not None
    return dict(snapshot["binding"])


def _optional_binding(path: Path) -> dict[str, str] | None:
    read = _secure_read_file(path, missing_ok=True)
    if read is None:
        return None
    content, _identity = read
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _merge_launch_failure(
    current: Mapping[str, Any] | None,
    *,
    stage: str,
    failure_type: str,
    message: str,
) -> dict[str, Any]:
    entry = build_finalization_secondary_failure(
        stage=stage,
        failure_type=failure_type,
        message=message,
    )
    if current is None:
        return {
            **entry,
            "secondary_failures": [],
        }
    merged = dict(current)
    secondary = list(merged.get("secondary_failures", []))
    secondary.append(entry)
    merged["secondary_failures"] = secondary
    return merged


def _publish_wrapper_exit_total(
    path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    published = dict(value)
    _write_exclusive(path, published)
    return published


def _propagate_publish_error(exc: BaseException) -> None:
    if isinstance(exc, ExclusivePublishError):
        raise exc


def _tmux_session() -> str:
    if "TMUX" not in os.environ:
        raise RuntimeError("CPU preflight wrapper must run inside tmux")
    result = subprocess.run(
        ["tmux", "display-message", "-p", "#S"],
        check=True,
        capture_output=True,
        text=True,
    )
    session = result.stdout.strip()
    if session != CONTROLLER_SESSION:
        raise RuntimeError(
            f"CPU preflight wrapper tmux session differs: {session!r}"
        )
    return session


def _read_process_stat(
    pid: int,
) -> tuple[dict[str, int], str] | None:
    if type(pid) is not int or pid <= 0:
        raise RuntimeError(f"process PID is invalid: {pid!r}")
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        raise RuntimeError(
            f"process stat permission denied for PID {pid}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"process stat read failed for PID {pid}") from exc
    opening = raw_stat.find("(")
    closing = raw_stat.rfind(")")
    if opening <= 0 or closing <= opening:
        raise RuntimeError(f"process identity stat is malformed for PID {pid}")
    try:
        stat_pid = int(raw_stat[:opening].strip())
    except ValueError as exc:
        raise RuntimeError(
            f"process identity stat is malformed for PID {pid}"
        ) from exc
    fields = raw_stat[closing + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError(
            f"process identity stat is malformed for PID {pid}"
        )
    try:
        state = fields[0]
        stat_ppid = int(fields[1])
        stat_pgid = int(fields[2])
        stat_sid = int(fields[3])
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"process identity stat is malformed for PID {pid}"
        ) from exc
    if (
        stat_pid != pid
        or len(state) != 1
        or stat_ppid <= 0
        or stat_pgid <= 0
        or stat_sid <= 0
        or start_ticks <= 0
    ):
        raise RuntimeError(
            f"process identity stat is malformed for PID {pid}"
        )
    return (
        build_process_identity(
            pid=stat_pid,
            ppid=stat_ppid,
            pgid=stat_pgid,
            sid=stat_sid,
            start_ticks=start_ticks,
        ),
        state,
    )


def _revalidate_process_after_missing_probe(
    pid: int,
    initial_identity: Mapping[str, int],
    probe: str,
) -> tuple[dict[str, int], str] | None:
    current = _read_process_stat(pid)
    if current is None:
        return None
    current_identity, current_state = current
    if current_identity != dict(initial_identity):
        raise RuntimeError(
            f"process identity changed after {probe} disappeared for PID {pid}"
        )
    if current_state == "Z":
        return current
    raise RuntimeError(
        f"process {probe} is absent while PID {pid} remains live "
        f"in state {current_state}"
    )


def _process_identity_state(
    pid: int,
) -> tuple[dict[str, int], str] | None:
    initial = _read_process_stat(pid)
    if initial is None:
        return None
    initial_identity, initial_state = initial
    if initial_state == "Z":
        return initial
    try:
        os.readlink(f"/proc/{pid}/exe")
    except (FileNotFoundError, ProcessLookupError):
        return _revalidate_process_after_missing_probe(
            pid, initial_identity, "executable"
        )
    except PermissionError as exc:
        raise RuntimeError(
            f"process executable permission denied for PID {pid}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"process executable read failed for PID {pid}"
        ) from exc
    try:
        live_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return _revalidate_process_after_missing_probe(
            pid, initial_identity, "process group"
        )
    except PermissionError as exc:
        raise RuntimeError(
            f"process group permission denied for PID {pid}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"process group read failed for PID {pid}"
        ) from exc
    final = _read_process_stat(pid)
    if final is None:
        return None
    final_identity, final_state = final
    if final_identity != initial_identity:
        raise RuntimeError(
            f"process identity changed during snapshot for PID {pid}"
        )
    if final_state == "Z":
        return final
    if final_identity["pgid"] != live_pgid:
        raise RuntimeError(
            f"process group changed during snapshot for PID {pid}"
        )
    return final


def _process_identity(pid: int) -> dict[str, int] | None:
    snapshot = _process_identity_state(pid)
    return None if snapshot is None else snapshot[0]


def _require_process_identity(pid: int, label: str) -> dict[str, int]:
    identity = _process_identity(pid)
    if identity is None:
        raise RuntimeError(f"{label} process is absent")
    return identity


def _tmux_identity(session: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            session,
            "-F",
            "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if (
            error.startswith("no server running on ")
            or error == f"can't find session: {session}"
            or error == f"can't find window: {session}"
            or error == f"can't find pane: {session}"
        ):
            raise TmuxTargetAbsent(error)
        raise RuntimeError(
            f"tmux identity command failed for {session}: {error}"
        )
    rows = [line.split("\t") for line in result.stdout.splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0] != session:
        raise RuntimeError(f"tmux identity differs for session {session}")
    identity = {
        "session": rows[0][0],
        "pane": rows[0][1],
        "pane_pid": int(rows[0][2]),
        "pane_current_command": rows[0][3],
    }
    _validate_tmux_identity(identity, session)
    return identity


def _tmux_runtime_status(session: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            session,
            "-F",
            (
                "#{session_name}\t#{pane_id}\t#{pane_pid}\t"
                "#{pane_dead}\t#{pane_dead_status}\t"
                "#{pane_pipe}\t#{pane_current_command}"
            ),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "tmux runtime status command failed: "
            f"{result.stderr.strip()}"
        )
    rows = [line.split("\t") for line in result.stdout.splitlines() if line]
    if (
        len(rows) != 1
        or len(rows[0]) != 7
        or rows[0][0] != session
        or rows[0][3] not in {"0", "1"}
        or rows[0][5] not in {"0", "1"}
        or not rows[0][6]
    ):
        raise RuntimeError(
            f"tmux runtime status differs for session {session}"
        )
    return {
        "session": rows[0][0],
        "pane": rows[0][1],
        "pane_pid": int(rows[0][2]),
        "pane_dead": rows[0][3] == "1",
        "pane_dead_status": (
            None if rows[0][4] == "" else int(rows[0][4])
        ),
        "pane_pipe": rows[0][5] == "1",
        "pane_current_command": rows[0][6],
    }


def _validate_tmux_identity(
    identity: Mapping[str, Any], expected_session: str
) -> None:
    pane = identity.get("pane")
    if (
        set(identity)
        != {"session", "pane", "pane_pid", "pane_current_command"}
        or identity.get("session") != expected_session
        or not isinstance(pane, str)
        or not pane.startswith("%")
        or not pane[1:].isdecimal()
        or type(identity.get("pane_pid")) is not int
        or int(identity["pane_pid"]) <= 1
        or not isinstance(identity.get("pane_current_command"), str)
        or not identity["pane_current_command"]
    ):
        raise RuntimeError(
            f"invalid public tmux identity for session {expected_session}"
        )


def _tmux_pane_identity(pane: str) -> dict[str, Any]:
    if (
        not isinstance(pane, str)
        or not pane.startswith("%")
        or not pane[1:].isdecimal()
    ):
        raise RuntimeError("tmux pane target is not an opaque pane ID")
    result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            pane,
            "-F",
            "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if (
            error.startswith("no server running on ")
            or error == f"can't find pane: {pane}"
        ):
            raise TmuxTargetAbsent(error)
        raise RuntimeError(
            f"tmux pane identity command failed for {pane}: {error}"
        )
    rows = [line.split("\t") for line in result.stdout.splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][1] != pane:
        raise RuntimeError(f"tmux identity differs for sealed pane {pane}")
    identity = {
        "session": rows[0][0],
        "pane": rows[0][1],
        "pane_pid": int(rows[0][2]),
        "pane_current_command": rows[0][3],
    }
    _validate_tmux_identity(identity, identity["session"])
    return identity


def _tmux_server_identity(target: str | None = None) -> dict[str, Any]:
    command = ["tmux", "display-message", "-p"]
    if target is not None:
        command.extend(["-t", target])
    command.append("#{pid}\t#{socket_path}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        error = result.stderr.strip()
        if (
            error.startswith("no server running on ")
            or (
                target is not None
                and (
                    error == f"can't find session: {target}"
                    or error == f"can't find window: {target}"
                    or error == f"can't find pane: {target}"
                )
            )
        ):
            raise TmuxTargetAbsent(error)
        raise RuntimeError(f"tmux server identity command failed: {error}")
    rows = [
        line.split("\t") for line in result.stdout.splitlines() if line
    ]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError("tmux server identity is malformed")
    server_pid = int(rows[0][0])
    server_process = _require_process_identity(
        server_pid, "tmux server"
    )
    socket_identity = _tmux_socket_identity(rows[0][1])
    return build_tmux_server_identity(
        server_pid=server_pid,
        server_process=server_process,
        socket_path=str(socket_identity["socket_path"]),
        socket_device=int(socket_identity["socket_device"]),
        socket_inode=int(socket_identity["socket_inode"]),
    )


def _tmux_socket_identity(socket_path: str) -> dict[str, Any]:
    path = Path(socket_path)
    if not path.is_absolute():
        raise RuntimeError("tmux socket path is not absolute")
    value = os.lstat(path)
    if not stat.S_ISSOCK(value.st_mode):
        raise RuntimeError("tmux socket identity is not a socket")
    return {
        "socket_path": str(path),
        "socket_device": int(value.st_dev),
        "socket_inode": int(value.st_ino),
    }


def _validate_tmux_owner_nonce(owner_nonce: object) -> str:
    if (
        not isinstance(owner_nonce, str)
        or len(owner_nonce) != TMUX_OWNER_NONCE_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in owner_nonce)
    ):
        raise RuntimeError("tmux owner nonce is invalid")
    return owner_nonce


def _tmux_owner_nonce_raw(
    session: str,
    socket_path: str,
) -> str:
    result = subprocess.run(
        [
            "tmux",
            "-S",
            socket_path,
            "show-environment",
            "-t",
            session,
            TMUX_OWNER_ENV,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "tmux owner nonce command failed: "
            f"{result.stderr.strip()}"
        )
    if result.stderr.strip():
        raise RuntimeError("tmux owner nonce command returned stderr")
    lines = result.stdout.splitlines()
    prefix = f"{TMUX_OWNER_ENV}="
    if (
        len(lines) != 1
        or not lines[0].startswith(prefix)
        or lines[0].count("=") != 1
    ):
        raise RuntimeError(
            "tmux owner nonce environment is malformed"
        )
    return lines[0][len(prefix) :]


def _tmux_owner_nonce(
    session: str,
    socket_path: str,
) -> str:
    return _validate_tmux_owner_nonce(
        _tmux_owner_nonce_raw(session, socket_path)
    )


def _build_tmux_owner_seal(
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
    expected_owner_nonce: str,
) -> dict[str, Any]:
    expected_owner_nonce = _validate_tmux_owner_nonce(
        expected_owner_nonce
    )
    server_process = _process_identity(int(tmux_server["server_pid"]))
    if server_process is None:
        raise RuntimeError("tmux server process is absent")
    socket_identity = _tmux_socket_identity(
        str(tmux_server["socket_path"])
    )
    pane_process = _launch_process_identity(
        int(tmux_identity["pane_pid"])
    )
    seal = build_pane_owner_seal(
        server_pid=int(tmux_server["server_pid"]),
        server_start_ticks=int(server_process["start_ticks"]),
        socket_path=str(socket_identity["socket_path"]),
        socket_device=int(socket_identity["socket_device"]),
        socket_inode=int(socket_identity["socket_inode"]),
        session=str(tmux_identity["session"]),
        pane=str(tmux_identity["pane"]),
        pane_pid=int(tmux_identity["pane_pid"]),
        pane_process=pane_process,
        owner_nonce=_tmux_owner_nonce(
            str(tmux_identity["session"]),
            str(tmux_server["socket_path"]),
        ),
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
    )
    if seal["owner_nonce"] != expected_owner_nonce:
        raise RuntimeError("tmux owner nonce differs after launch")
    _validate_tmux_owner_seal(seal, tmux_identity, tmux_server)
    return seal


def _validate_tmux_owner_seal(
    owner_seal: Mapping[str, Any],
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> None:
    validate_pane_owner_seal(
        owner_seal,
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
        label="wrapper pane owner seal",
    )
    if (
        owner_seal.get("server_pid") != tmux_server.get("server_pid")
        or owner_seal.get("socket_path") != tmux_server.get("socket_path")
        or owner_seal.get("session") != tmux_identity.get("session")
        or owner_seal.get("pane") != tmux_identity.get("pane")
        or owner_seal.get("pane_pid") != tmux_identity.get("pane_pid")
        or type(owner_seal.get("server_start_ticks")) is not int
        or int(owner_seal["server_start_ticks"]) <= 0
        or type(owner_seal.get("socket_device")) is not int
        or type(owner_seal.get("socket_inode")) is not int
        or _validate_tmux_owner_nonce(owner_seal.get("owner_nonce"))
        != owner_seal.get("owner_nonce")
    ):
        raise RuntimeError("tmux owner seal is invalid")


def _validate_tmux_owner_host_identity(
    owner_seal: Mapping[str, Any],
) -> None:
    server_process = _process_identity(int(owner_seal["server_pid"]))
    if (
        server_process is None
        or server_process["start_ticks"]
        != owner_seal["server_start_ticks"]
    ):
        raise RuntimeError("tmux owner server process identity differs")
    if _tmux_socket_identity(str(owner_seal["socket_path"])) != {
        "socket_path": owner_seal["socket_path"],
        "socket_device": owner_seal["socket_device"],
        "socket_inode": owner_seal["socket_inode"],
    }:
        raise RuntimeError("tmux owner socket identity differs")
    try:
        pane_process = _launch_process_identity(
            int(owner_seal["pane_pid"])
        )
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise RuntimeError(
            "tmux owner pane process is absent"
        ) from exc
    if pane_process != owner_seal["pane_process"]:
        raise RuntimeError("tmux owner pane process identity differs")


def _read_strict_json_contract(
    path: Path,
    digest_field: str,
) -> dict[str, Any]:
    snapshot = _secure_json_snapshot(
        path, digest_field=digest_field
    )
    assert snapshot is not None
    return dict(snapshot["value"])


def _observer_gate_ready(
    *,
    expected_session: str,
    owner_nonce: str,
    ready_path: Path,
    release_path: Path,
    bootstrap_path: Path,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    observer_command: Sequence[str],
) -> int:
    owner_nonce = _validate_tmux_owner_nonce(owner_nonce)
    tmux_identity = _tmux_identity(expected_session)
    process = _require_process_identity(
        os.getpid(), "CPU preflight observer bootstrap gate"
    )
    if (
        tmux_identity["pane_pid"] != process["pid"]
        or process["pgid"] != process["pid"]
    ):
        raise RuntimeError(
            "CPU preflight observer bootstrap gate pane/process differs"
        )
    tmux_server = _tmux_server_identity(tmux_identity["pane"])
    gate_executable = os.readlink(f"/proc/{process['pid']}/exe")
    gate_command = _process_command(process["pid"])
    if (
        _tmux_owner_nonce(
            expected_session,
            str(tmux_server["socket_path"]),
        )
        != owner_nonce
    ):
        raise RuntimeError(
            "CPU preflight observer bootstrap gate environment differs"
        )
    ready = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_observer_gate_ready_v1"
        ),
        "policy_sha256": policy_sha256,
        "verified_implementations": (
            _reverify_verified_preflight_apis()
        ),
        "wrapper_claim": dict(wrapper_binding),
        "observer_session": expected_session,
        "owner_nonce": owner_nonce,
        "process": process,
        "gate_executable": gate_executable,
        "gate_command": gate_command,
        "tmux": tmux_identity,
        "tmux_server": tmux_server,
        "release_path": str(release_path.resolve()),
        "bootstrap_path": str(bootstrap_path.resolve()),
        "observer_command": list(observer_command),
        "published_at": _utc_now(),
    }
    ready["observer_gate_ready_sha256"] = _canonical_digest(
        ready, "observer_gate_ready_sha256"
    )
    _write_exclusive(ready_path, ready)
    while _secure_read_file(release_path, missing_ok=True) is None:
        time.sleep(0.02)
    release = _read_strict_json_contract(
        release_path, "observer_gate_release_sha256"
    )
    ready_binding = _json_binding(
        ready_path, "observer_gate_ready_sha256"
    )
    if (
        set(release)
        != {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "verified_implementations",
            "wrapper_claim",
            "observer_gate_ready",
            "observer_session",
            "owner_nonce",
            "observer_command",
            "released_at",
            "observer_gate_release_sha256",
        }
        or release.get("schema_version") != 1
        or release.get("contract_type")
        != "safa_canonical_preflight_observer_gate_release_v1"
        or release.get("policy_sha256") != policy_sha256
        or release.get("verified_implementations")
        != _reverify_verified_preflight_apis()
        or release.get("wrapper_claim") != dict(wrapper_binding)
        or release.get("observer_gate_ready") != ready_binding
        or release.get("observer_session") != expected_session
        or release.get("owner_nonce") != owner_nonce
        or release.get("observer_command") != list(observer_command)
    ):
        raise RuntimeError(
            "CPU preflight observer bootstrap gate release differs"
        )
    current_tmux = _tmux_pane_identity(str(tmux_identity["pane"]))
    current_server = _tmux_server_identity(str(tmux_identity["pane"]))
    if (
        current_tmux != tmux_identity
        or current_server != tmux_server
        or _process_identity(process["pid"]) != process
        or os.readlink(f"/proc/{process['pid']}/exe") != gate_executable
        or _process_command_bytes(process["pid"])
        != _command_bytes(gate_command)
        or _tmux_owner_nonce(
            expected_session,
            str(tmux_server["socket_path"]),
        )
        != owner_nonce
    ):
        raise RuntimeError(
            "CPU preflight observer bootstrap gate identity changed"
        )
    environment = dict(os.environ)
    environment[OBSERVER_BOOTSTRAP_PATH_ENV] = str(
        bootstrap_path.resolve()
    )
    environment[OBSERVER_BOOTSTRAP_POLICY_ENV] = policy_sha256
    environment[OBSERVER_BOOTSTRAP_WRAPPER_ENV] = json.dumps(
        dict(wrapper_binding), separators=(",", ":")
    )
    environment[OBSERVER_BOOTSTRAP_NONCE_ENV] = owner_nonce
    environment[OBSERVER_SESSION_ENV] = expected_session
    os.execve(
        str(observer_command[0]),
        list(observer_command),
        environment,
    )
    raise RuntimeError("observer bootstrap gate execve returned")


def _validate_observer_gate_ready(
    ready_path: Path,
    *,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    owner_nonce: str,
    release_path: Path,
    bootstrap_path: Path,
    observer_command: Sequence[str],
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> dict[str, Any]:
    ready = _read_strict_json_contract(
        ready_path, "observer_gate_ready_sha256"
    )
    if (
        set(ready)
        != {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "verified_implementations",
            "wrapper_claim",
            "observer_session",
            "owner_nonce",
            "process",
            "gate_executable",
            "gate_command",
            "tmux",
            "tmux_server",
            "release_path",
            "bootstrap_path",
            "observer_command",
            "published_at",
            "observer_gate_ready_sha256",
        }
        or ready.get("schema_version") != 1
        or ready.get("contract_type")
        != "safa_canonical_preflight_observer_gate_ready_v1"
        or ready.get("policy_sha256") != policy_sha256
        or ready.get("verified_implementations")
        != _reverify_verified_preflight_apis()
        or ready.get("wrapper_claim") != dict(wrapper_binding)
        or ready.get("observer_session") != tmux_identity.get("session")
        or ready.get("owner_nonce") != owner_nonce
        or ready.get("release_path") != str(release_path.resolve())
        or ready.get("bootstrap_path") != str(bootstrap_path.resolve())
        or ready.get("observer_command") != list(observer_command)
        or ready.get("tmux") != dict(tmux_identity)
        or ready.get("tmux_server") != dict(tmux_server)
        or ready.get("process", {}).get("pid")
        != tmux_identity.get("pane_pid")
        or ready.get("process", {}).get("pgid")
        != ready.get("process", {}).get("pid")
        or ready.get("gate_executable")
        != os.readlink(f"/proc/{ready['process']['pid']}/exe")
        or ready.get("gate_command")
        != _process_command(ready["process"]["pid"])
        or _process_command_bytes(ready["process"]["pid"])
        != _command_bytes(ready["gate_command"])
    ):
        raise RuntimeError(
            "CPU preflight observer bootstrap gate ready differs"
        )
    return ready


def _probe_observer_gate(
    *,
    ready_path: Path,
    release_path: Path,
    bootstrap_path: Path,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    owner_nonce: str,
    observer_command: Sequence[str],
    owner_recorder: Callable[
        [
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, int] | None,
        ],
        None,
    ]
    | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + OBSERVER_IDENTITY_WAIT_SECONDS
    last_probe: dict[str, Any] | None = None
    best_tmux: dict[str, Any] | None = None
    best_tmux_server: dict[str, Any] | None = None
    best_owner_seal: dict[str, Any] | None = None
    best_process: dict[str, int] | None = None
    best_process_probe: dict[str, Any] | None = None

    def with_best(probe: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(probe)
        value.update(
            {
                "best_tmux": best_tmux,
                "best_tmux_server": best_tmux_server,
                "best_tmux_owner_seal": best_owner_seal,
                "best_process": best_process,
                "best_process_probe": best_process_probe,
            }
        )
        return value

    def remember_exact(
        tmux_identity: Mapping[str, Any],
        tmux_server: Mapping[str, Any],
        owner_seal: Mapping[str, Any],
        process: Mapping[str, int] | None = None,
        process_probe: Mapping[str, Any] | None = None,
    ) -> dict[str, str] | None:
        nonlocal best_tmux
        nonlocal best_tmux_server
        nonlocal best_owner_seal
        nonlocal best_process
        nonlocal best_process_probe
        candidate_tmux = dict(tmux_identity)
        candidate_server = dict(tmux_server)
        candidate_seal = dict(owner_seal)
        candidate_tmux_owner = {
            key: candidate_tmux[key]
            for key in ("session", "pane", "pane_pid")
        }
        if best_owner_seal is None:
            best_tmux = candidate_tmux
            best_tmux_server = candidate_server
            best_owner_seal = candidate_seal
        elif (
            candidate_tmux_owner
            != {
                key: best_tmux[key]
                for key in ("session", "pane", "pane_pid")
            }
            or candidate_server != best_tmux_server
            or candidate_seal != best_owner_seal
        ):
            return {
                "type": "TmuxOwnerEvidenceConflict",
                "message": (
                    "later exact owner evidence differs from the "
                    "monotonic owner seal"
                ),
            }
        else:
            best_tmux = candidate_tmux
        if owner_recorder is not None:
            owner_recorder(
                candidate_tmux,
                candidate_server,
                candidate_seal,
                None if process is None else dict(process),
            )
        if process is not None:
            candidate_process = dict(process)
            if best_process is None:
                best_process = candidate_process
                best_process_probe = (
                    None
                    if process_probe is None
                    else dict(process_probe)
                )
            elif candidate_process != best_process:
                return {
                    "type": "ProcessOwnerEvidenceConflict",
                    "message": (
                        "later process evidence differs from the "
                        "monotonic owner process"
                    ),
                }
        return None

    while True:
        try:
            tmux_identity = _tmux_identity(OBSERVER_SESSION)
        except TmuxTargetAbsent:
            last_probe = {
                "status": "absent",
                "tmux": None,
                "tmux_server": None,
                "tmux_owner_seal": None,
                "process": None,
                "process_probe": {"status": "not_observed"},
                "gate_ready": None,
                "failure": None,
                "session_residual": False,
            }
        except BaseException as exc:
            last_probe = {
                "status": "owner_unsealed_unknown",
                "tmux": None,
                "tmux_server": None,
                "tmux_owner_seal": None,
                "process": None,
                "process_probe": {"status": "not_observed"},
                "gate_ready": None,
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "session_residual": True,
            }
        else:
            tmux_server: dict[str, Any] | None = None
            try:
                tmux_server = _tmux_server_identity(
                    str(tmux_identity["pane"])
                )
                observed_nonce = _tmux_owner_nonce_raw(
                    str(tmux_identity["session"]),
                    str(tmux_server["socket_path"]),
                )
            except BaseException as exc:
                last_probe = {
                    "status": "owner_unsealed_unknown",
                    "tmux": tmux_identity,
                    "tmux_server": tmux_server,
                    "tmux_owner_seal": None,
                    "process": None,
                    "process_probe": {"status": "not_observed"},
                    "gate_ready": None,
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "session_residual": True,
                }
            else:
                if observed_nonce != owner_nonce:
                    last_probe = {
                        "status": "foreign_or_incomplete_owner",
                        "tmux": tmux_identity,
                        "tmux_server": tmux_server,
                        "tmux_owner_seal": None,
                        "process": None,
                        "process_probe": {"status": "not_observed"},
                        "gate_ready": None,
                        "failure": {
                            "type": "TmuxOwnerMarkerMismatch",
                            "message": (
                                "observer tmux owner marker differs"
                            ),
                        },
                        "session_residual": True,
                    }
                else:
                    try:
                        owner_seal = _build_tmux_owner_seal(
                            tmux_identity, tmux_server, owner_nonce
                        )
                    except BaseException as exc:
                        last_probe = {
                            "status": "owner_unsealed_unknown",
                            "tmux": tmux_identity,
                            "tmux_server": tmux_server,
                            "tmux_owner_seal": None,
                            "process": None,
                            "process_probe": {
                                "status": "not_observed"
                            },
                            "gate_ready": None,
                            "failure": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            "session_residual": True,
                        }
                    else:
                        evidence_conflict = remember_exact(
                            tmux_identity,
                            tmux_server,
                            owner_seal,
                        )
                        if evidence_conflict is not None:
                            return with_best(
                                {
                                    "status": (
                                        "exact_owner_evidence_conflict"
                                    ),
                                    "tmux": tmux_identity,
                                    "tmux_server": tmux_server,
                                    "tmux_owner_seal": owner_seal,
                                    "process": None,
                                    "process_probe": {
                                        "status": "not_observed"
                                    },
                                    "gate_ready": None,
                                    "failure": evidence_conflict,
                                    "session_residual": True,
                                }
                            )
                        try:
                            process_snapshot = _read_process_stat(
                                int(tmux_identity["pane_pid"])
                            )
                        except BaseException as exc:
                            return with_best({
                                "status": (
                                    "exact_owner_process_probe_failed"
                                ),
                                "tmux": tmux_identity,
                                "tmux_server": tmux_server,
                                "tmux_owner_seal": owner_seal,
                                "process": None,
                                "process_probe": {
                                    "status": "error",
                                    "pid": tmux_identity["pane_pid"],
                                    "failure": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                },
                                "gate_ready": None,
                                "failure": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                },
                                "session_residual": True,
                            })
                        if process_snapshot is None:
                            process = None
                            process_probe = {
                                "status": "absent",
                                "pid": tmux_identity["pane_pid"],
                            }
                        else:
                            process, process_state = process_snapshot
                            process_probe = {
                                "status": (
                                    "zombie"
                                    if process_state == "Z"
                                    else "live"
                                ),
                                "pid": tmux_identity["pane_pid"],
                                "state": process_state,
                                "identity": dict(process),
                            }
                        evidence_conflict = remember_exact(
                            tmux_identity,
                            tmux_server,
                            owner_seal,
                            process,
                            process_probe,
                        )
                        if evidence_conflict is not None:
                            return with_best(
                                {
                                    "status": (
                                        "exact_owner_evidence_conflict"
                                    ),
                                    "tmux": tmux_identity,
                                    "tmux_server": tmux_server,
                                    "tmux_owner_seal": owner_seal,
                                    "process": process,
                                    "process_probe": process_probe,
                                    "gate_ready": None,
                                    "failure": evidence_conflict,
                                    "session_residual": True,
                                }
                            )
                        if (
                            _secure_read_file(
                                ready_path, missing_ok=True
                            )
                            is not None
                        ):
                            try:
                                ready = _validate_observer_gate_ready(
                                    ready_path,
                                    policy_sha256=policy_sha256,
                                    wrapper_binding=wrapper_binding,
                                    owner_nonce=owner_nonce,
                                    release_path=release_path,
                                    bootstrap_path=bootstrap_path,
                                    observer_command=observer_command,
                                    tmux_identity=tmux_identity,
                                    tmux_server=tmux_server,
                                )
                                if (
                                    process is not None
                                    and ready["process"] != process
                                ):
                                    raise RuntimeError(
                                        "observer gate process differs "
                                        "from stat-only provisional "
                                        "process"
                                    )
                            except BaseException as exc:
                                return with_best({
                                    "status": (
                                        "exact_owner_ready_invalid"
                                    ),
                                    "tmux": tmux_identity,
                                    "tmux_server": tmux_server,
                                    "tmux_owner_seal": owner_seal,
                                    "process": process,
                                    "process_probe": process_probe,
                                    "gate_ready": None,
                                    "failure": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    "session_residual": True,
                                })
                            return with_best({
                                "status": "exact_ready",
                                "tmux": tmux_identity,
                                "tmux_server": tmux_server,
                                "tmux_owner_seal": owner_seal,
                                "process": process,
                                "process_probe": process_probe,
                                "gate_ready": ready,
                                "failure": None,
                                "session_residual": True,
                            })
                        last_probe = {
                            "status": "exact_owner_not_ready",
                            "tmux": tmux_identity,
                            "tmux_server": tmux_server,
                            "tmux_owner_seal": owner_seal,
                            "process": process,
                            "process_probe": process_probe,
                            "gate_ready": None,
                            "failure": None,
                            "session_residual": True,
                        }
        if time.monotonic() >= deadline:
            assert last_probe is not None
            return with_best(last_probe)
        time.sleep(0.02)


def _observer_gate_command(
    *,
    ready_path: Path,
    release_path: Path,
    bootstrap_path: Path,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    owner_nonce: str,
    observer_command: Sequence[str],
) -> list[str]:
    try:
        config_index = list(observer_command).index("--config") + 1
        config_path = Path(str(observer_command[config_index])).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError(
            "observer command omits exact config path"
        ) from exc
    return [
        str(observer_command[0]),
        "-u",
        str(Path(__file__).resolve()),
        OBSERVER_GATE_MODE,
        "--expected-session",
        OBSERVER_SESSION,
        "--owner-nonce",
        owner_nonce,
        "--ready-path",
        str(ready_path.resolve()),
        "--release-path",
        str(release_path.resolve()),
        "--bootstrap-path",
        str(bootstrap_path.resolve()),
        "--config",
        str(config_path),
        "--policy-sha256",
        policy_sha256,
        "--wrapper-binding-json",
        json.dumps(dict(wrapper_binding), separators=(",", ":")),
        "--observer-command-json",
        json.dumps(list(observer_command), separators=(",", ":")),
    ]


def _launch_and_probe_observer_gate(
    *,
    repo_root: Path,
    ready_path: Path,
    release_path: Path,
    bootstrap_path: Path,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    owner_nonce: str,
    observer_command: Sequence[str],
    owner_recorder: Callable[
        [
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, int] | None,
        ],
        None,
    ]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_command = _observer_gate_command(
        ready_path=ready_path,
        release_path=release_path,
        bootstrap_path=bootstrap_path,
        policy_sha256=policy_sha256,
        wrapper_binding=wrapper_binding,
        owner_nonce=owner_nonce,
        observer_command=observer_command,
    )
    client: dict[str, Any]
    shell_command = "exec " + shlex.join(gate_command)
    tmux_command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        OBSERVER_SESSION,
        "-c",
        str(repo_root.resolve()),
        "-e",
        f"{TMUX_OWNER_ENV}={owner_nonce}",
        "-e",
        f"{OBSERVER_SESSION_ENV}={OBSERVER_SESSION}",
        shell_command,
    ]
    try:
        result = subprocess.run(
            tmux_command,
            capture_output=True,
            text=True,
        )
        client = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "failure": None,
            "command": tmux_command,
        }
    except BaseException as exc:
        client = {
            "returncode": None,
            "stdout": None,
            "stderr": None,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "command": tmux_command,
        }
    probe = _probe_observer_gate(
        ready_path=ready_path,
        release_path=release_path,
        bootstrap_path=bootstrap_path,
        policy_sha256=policy_sha256,
        wrapper_binding=wrapper_binding,
        owner_nonce=owner_nonce,
        observer_command=observer_command,
        owner_recorder=owner_recorder,
    )
    return probe, client


def _tmux_owner_condition(owner_seal: Mapping[str, Any]) -> str:
    pane = str(owner_seal["pane"])
    session = str(owner_seal["session"])
    server_pid = int(owner_seal["server_pid"])
    pane_pid = int(owner_seal["pane_pid"])
    owner_nonce = _validate_tmux_owner_nonce(
        owner_seal["owner_nonce"]
    )
    return (
        "#{&&:"
        f"#{{==:#{{pid}},{server_pid}}},"
        "#{&&:"
        f"#{{==:#{{session_name}},{session}}},"
        "#{&&:"
        f"#{{==:#{{pane_id}},{pane}}},"
        "#{&&:"
        f"#{{==:#{{pane_pid}},{pane_pid}}},"
        f"#{{==:#{{E:{TMUX_OWNER_ENV}}},{owner_nonce}}}"
        "}}"
        "}}"
        "}}"
        "}"
    )


def _set_observer_remain_on_exit(
    owner_seal: Mapping[str, Any],
) -> None:
    _validate_tmux_owner_host_identity(owner_seal)
    pane = str(owner_seal["pane"])
    result = subprocess.run(
        [
            "tmux",
            "-S",
            str(owner_seal["socket_path"]),
            "if-shell",
            "-t",
            pane,
            "-F",
            _tmux_owner_condition(owner_seal),
            f"set-window-option -t {pane} remain-on-exit on",
            (
                "display-message -p "
                f"{TMUX_CONDITIONAL_REMAIN_REJECTED}"
            ),
        ],
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        raise RuntimeError(
            "observer remain-on-exit configuration failed: "
            f"{stderr}"
        )
    if (
        stdout == TMUX_CONDITIONAL_REMAIN_REJECTED
        and not stderr
    ):
        raise RuntimeError(
            "observer remain-on-exit owner condition rejected"
        )
    if stdout or stderr:
        raise RuntimeError(
            "observer remain-on-exit configuration returned "
            "unexpected output"
        )


def _conditional_kill_tmux_owner(
    owner_seal: Mapping[str, Any],
) -> tuple[str, subprocess.CompletedProcess[str]]:
    _validate_tmux_owner_host_identity(owner_seal)
    pane = str(owner_seal["pane"])
    condition = _tmux_owner_condition(owner_seal)
    result = subprocess.run(
        [
            "tmux",
            "-S",
            str(owner_seal["socket_path"]),
            "if-shell",
            "-t",
            pane,
            "-F",
            condition,
            f"kill-pane -t {pane}",
            (
                "display-message -p "
                f"{TMUX_CONDITIONAL_KILL_REJECTED}"
            ),
        ],
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        return "command_failed", result
    if stdout == TMUX_CONDITIONAL_KILL_REJECTED and not stderr:
        return "condition_rejected", result
    if stdout or stderr:
        raise RuntimeError(
            "tmux conditional owner kill returned unexpected output"
        )
    return "executed", result


def _provisional_tmux_owner_state(
    owner_seal: Mapping[str, Any],
) -> tuple[
    bool,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    pane = str(owner_seal["pane"])
    try:
        server = _tmux_server_identity(pane)
    except TmuxTargetAbsent:
        return False, None, None
    try:
        tmux_identity = _tmux_pane_identity(pane)
    except TmuxTargetAbsent:
        return False, server, None
    exact_owner = (
        server.get("server_pid") == owner_seal.get("server_pid")
        and server.get("socket_path") == owner_seal.get("socket_path")
        and tmux_identity.get("session") == owner_seal.get("session")
        and tmux_identity.get("pane") == owner_seal.get("pane")
        and tmux_identity.get("pane_pid") == owner_seal.get("pane_pid")
        and _tmux_owner_nonce(
            str(owner_seal["session"]),
            str(owner_seal["socket_path"]),
        )
        == owner_seal.get("owner_nonce")
    )
    return exact_owner, server, tmux_identity


def _provisional_process_residual(
    sealed_process: Mapping[str, int] | None,
    process_probe_failure: Mapping[str, str] | None = None,
) -> bool | None:
    if process_probe_failure is not None:
        return None
    if sealed_process is None:
        return False
    snapshot = _read_process_stat(int(sealed_process["pid"]))
    if snapshot is None:
        return False
    identity, state = snapshot
    return identity == dict(sealed_process) and state != "Z"


def _terminate_provisional_tmux_owner(
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
    owner_seal: Mapping[str, Any],
    sealed_process: Mapping[str, int] | None,
    process_probe_failure: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started_at = _utc_now()
    _validate_tmux_identity(tmux_identity, OBSERVER_SESSION)
    _validate_tmux_owner_seal(
        owner_seal, tmux_identity, tmux_server
    )
    kill_status, kill_result = _conditional_kill_tmux_owner(
        owner_seal
    )
    exact_owner = True
    observed_server: dict[str, Any] | None = None
    observed_tmux: dict[str, Any] | None = None
    if kill_status == "executed":
        deadline = time.monotonic() + PROCESS_TERMINATION_WAIT_SECONDS
        while time.monotonic() < deadline:
            (
                exact_owner,
                observed_server,
                observed_tmux,
            ) = _provisional_tmux_owner_state(owner_seal)
            if not exact_owner:
                break
            time.sleep(0.05)
    else:
        (
            exact_owner,
            observed_server,
            observed_tmux,
        ) = _provisional_tmux_owner_state(owner_seal)
    foreign_residual = bool(
        observed_tmux is not None and not exact_owner
    )
    process_residual = _provisional_process_residual(
        sealed_process, process_probe_failure
    )
    if kill_status == "executed":
        if not exact_owner and process_residual is False:
            status = "closed_provisional_observer"
        elif not exact_owner and process_residual is None:
            status = "cleanup_indeterminate_process_residual"
        else:
            status = "cleanup_timeout_with_residual"
    elif kill_status == "condition_rejected":
        status = (
            "identity_replaced_not_terminated"
            if not exact_owner
            else "conditional_kill_refused_owner_residual"
        )
    else:
        status = "conditional_kill_command_failed"
    failure = None
    if kill_status != "executed":
        failure = {
            "type": (
                "TmuxConditionalKillRejected"
                if kill_status == "condition_rejected"
                else "TmuxConditionalKillCommandError"
            ),
            "message": (
                TMUX_CONDITIONAL_KILL_REJECTED
                if kill_status == "condition_rejected"
                else kill_result.stderr.strip()
            ),
        }
    return {
        "session": OBSERVER_SESSION,
        "sealed_tmux": dict(tmux_identity),
        "sealed_tmux_server": dict(tmux_server),
        "sealed_tmux_owner": dict(owner_seal),
        "sealed_process": (
            None if sealed_process is None else dict(sealed_process)
        ),
        "status": status,
        "session_residual": exact_owner,
        "process_residual": process_residual,
        "foreign_session_residual": foreign_residual,
        "foreign_pane_residual": foreign_residual,
        "foreign_tmux": (
            dict(observed_tmux) if foreign_residual else None
        ),
        "foreign_tmux_server": (
            dict(observed_server) if foreign_residual else None
        ),
        "tmux_kill_status": kill_status,
        "process_probe_failure": (
            None
            if process_probe_failure is None
            else dict(process_probe_failure)
        ),
        "failure": failure,
        "started_at": started_at,
        "completed_at": _utc_now(),
    }


def _assert_process_identity(identity: Mapping[str, int], label: str) -> None:
    current = _process_identity(int(identity["pid"]))
    if current is None:
        raise RuntimeError(f"{label} process is absent")
    if current != dict(identity):
        raise RuntimeError(
            f"{label} process identity differs: "
            f"expected={dict(identity)}, current={current}"
        )


def _assert_tmux_process_identity(
    session: str,
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
    process_identity: Mapping[str, int],
) -> None:
    _validate_tmux_identity(tmux_identity, session)
    validate_tmux_server_identity(
        tmux_server, "sealed tmux server identity"
    )
    pane = str(tmux_identity["pane"])
    current_server = _tmux_server_identity(pane)
    if current_server != dict(tmux_server):
        raise RuntimeError(
            f"tmux server identity differs for sealed pane {pane}"
        )
    current_tmux = _tmux_pane_identity(pane)
    if current_tmux != dict(tmux_identity):
        raise RuntimeError(
            f"tmux identity differs for sealed session {session}: "
            f"expected={dict(tmux_identity)}, current={current_tmux}"
        )
    if (
        current_tmux["pane_pid"] != process_identity["pid"]
        or process_identity["pgid"] != process_identity["pid"]
    ):
        raise RuntimeError(
            f"tmux pane/process binding differs for sealed session {session}"
        )
    _assert_process_identity(process_identity, f"tmux session {session}")


def _process_command(pid: int) -> list[str]:
    try:
        content = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise RuntimeError(
            f"process command is absent for PID {pid}"
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            f"process command permission denied for PID {pid}"
        ) from exc
    try:
        command = [
            item.decode("utf-8")
            for item in content.split(b"\0")
            if item
        ]
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"process command is not UTF-8 for PID {pid}"
        ) from exc
    if not command:
        raise RuntimeError(f"process command is empty for PID {pid}")
    return command


def _process_command_bytes(pid: int) -> bytes:
    try:
        value = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise RuntimeError(
            f"process command bytes are absent for PID {pid}"
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            f"process command bytes permission denied for PID {pid}"
        ) from exc
    if not value or not value.endswith(b"\0"):
        raise RuntimeError(
            f"process command bytes are malformed for PID {pid}"
        )
    return value


def _command_bytes(arguments: Sequence[str]) -> bytes:
    if not arguments or any(not isinstance(item, str) for item in arguments):
        raise RuntimeError("expected process arguments are invalid")
    return b"\0".join(os.fsencode(item) for item in arguments) + b"\0"


def _process_executable_identity(pid: int) -> dict[str, Any]:
    proc_path = f"/proc/{pid}/exe"
    descriptor = os.open(proc_path, os.O_RDONLY)
    try:
        value = os.fstat(descriptor)
        raw_target = os.readlink(proc_path)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(
            f"process executable target is not regular for PID {pid}"
        )
    target = Path(raw_target).resolve(strict=True)
    target_value = target.stat()
    if (
        target_value.st_dev != value.st_dev
        or target_value.st_ino != value.st_ino
    ):
        raise RuntimeError(
            f"process executable target changed for PID {pid}"
        )
    return build_file_identity(
        path=str(target),
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
    )


def _launch_process_identity(pid: int) -> dict[str, int]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError(
            f"launch process stat is malformed for PID {pid}"
        )
    fields = raw[closing + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError(
            f"launch process stat is incomplete for PID {pid}"
        )
    return build_process_identity(
        pid=pid,
        ppid=int(fields[1]),
        pgid=int(fields[2]),
        sid=int(fields[3]),
        start_ticks=int(fields[19]),
    )


def _wait_tmux_process_identity(
    session: str,
    expected_owner_nonce: str,
    bootstrap_path: Path,
    *,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    expected_command: Sequence[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, int],
    dict[str, Any],
]:
    deadline = time.monotonic() + OBSERVER_IDENTITY_WAIT_SECONDS
    while _secure_read_file(bootstrap_path, missing_ok=True) is None:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"tmux observer bootstrap timed out for session {session}"
            )
        time.sleep(0.05)
    bootstrap_snapshot = _secure_json_snapshot(
        bootstrap_path,
        digest_field="observer_bootstrap_sha256",
    )
    assert bootstrap_snapshot is not None
    bootstrap = dict(bootstrap_snapshot["value"])
    expected_keys = {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "verified_implementations",
        "wrapper_claim",
        "observer_session",
        "owner_nonce",
        "process",
        "executable",
        "executable_identity",
        "command",
        "tmux",
        "published_at",
        "observer_bootstrap_sha256",
    }
    if (
        set(bootstrap) != expected_keys
        or bootstrap.get("schema_version") != 1
        or bootstrap.get("contract_type")
        != "safa_canonical_preflight_observer_bootstrap_v1"
        or bootstrap.get("policy_sha256") != policy_sha256
        or bootstrap.get("verified_implementations")
        != _reverify_verified_preflight_apis()
        or bootstrap.get("wrapper_claim") != dict(wrapper_binding)
        or bootstrap.get("observer_session") != session
        or bootstrap.get("owner_nonce") != expected_owner_nonce
        or bootstrap.get("command") != list(expected_command)
        or bootstrap.get("observer_bootstrap_sha256")
        != _canonical_digest(bootstrap, "observer_bootstrap_sha256")
    ):
        raise RuntimeError("tmux observer bootstrap contract mismatch")
    validate_executable_identity(
        bootstrap["executable_identity"],
        "observer bootstrap executable identity",
    )
    tmux_identity = bootstrap["tmux"]
    process_identity = bootstrap["process"]
    _validate_tmux_identity(tmux_identity, session)
    if (
        tmux_identity["pane_pid"] != process_identity.get("pid")
        or process_identity != _process_identity(process_identity["pid"])
        or bootstrap["executable"]
        != os.readlink(f"/proc/{process_identity['pid']}/exe")
        or bootstrap["executable_identity"]
        != _process_executable_identity(process_identity["pid"])
        or bootstrap["command"]
        != _process_command(process_identity["pid"])
        or _process_command_bytes(process_identity["pid"])
        != _command_bytes(bootstrap["command"])
    ):
        raise RuntimeError("tmux observer bootstrap process differs")
    tmux_server = _tmux_server_identity(tmux_identity["pane"])
    tmux_owner_seal = _build_tmux_owner_seal(
        tmux_identity,
        tmux_server,
        expected_owner_nonce,
    )
    _assert_tmux_process_identity(
        session, tmux_identity, tmux_server, process_identity
    )
    return (
        tmux_identity,
        tmux_server,
        tmux_owner_seal,
        process_identity,
        bootstrap,
    )


def _terminate_owned_process(
    process: subprocess.Popen[Any],
    expected_identity: Mapping[str, int],
) -> None:
    if process.poll() is not None:
        return
    if expected_identity["pid"] != process.pid:
        raise RuntimeError("owned controller PID differs from sealed identity")
    _assert_process_identity(expected_identity, "owned controller")
    if expected_identity["pgid"] != process.pid:
        raise RuntimeError("owned controller process group differs")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=PROCESS_TERMINATION_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        _assert_process_identity(expected_identity, "owned controller")
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _close_owned_controller_process(
    process: subprocess.Popen[Any],
    expected_identity: Mapping[str, int] | None,
    *,
    terminate: bool,
) -> tuple[int | None, dict[str, Any]]:
    """Close an owned controller without allowing cleanup errors to escape.

    The returned code is populated only by a successful ``Popen.wait`` call.
    Every exception is retained in the durable closure report.  The exact
    Popen child can still be terminated safely when its process-group identity
    could not be sealed because an unreaped child PID cannot be reused.
    """

    started_at = _utc_now()
    failures: list[dict[str, str]] = []
    term_sent = False
    kill_sent = False
    waited = False
    return_code: int | None = None
    group_identity_verified = False

    def record(stage: str, exc: BaseException) -> None:
        failures.append(
            build_finalization_secondary_failure(
                stage=stage,
                failure_type=type(exc).__name__,
                message=str(exc),
            )
        )

    def wait_owned(
        stage: str, timeout: float | None = None
    ) -> int | None:
        nonlocal waited, return_code
        try:
            if timeout is None:
                observed = process.wait()
            else:
                observed = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise
        except BaseException as exc:
            record(stage, exc)
            return None
        waited = True
        return_code = int(observed)
        return return_code

    try:
        running = process.poll() is None
    except BaseException as exc:
        record("initial_poll", exc)
        running = True

    if running and terminate:
        try:
            if expected_identity is None:
                raise RuntimeError(
                    "owned controller process identity was not sealed"
                )
            if expected_identity["pid"] != process.pid:
                raise RuntimeError(
                    "owned controller PID differs from sealed identity"
                )
            _assert_process_identity(
                expected_identity, "owned controller termination"
            )
            if expected_identity["pgid"] != process.pid:
                raise RuntimeError(
                    "owned controller process group differs"
                )
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError(
                    "live owned controller process group differs"
                )
            group_identity_verified = True
        except BaseException as exc:
            record("termination_initial_identity", exc)
        try:
            if group_identity_verified:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            term_sent = True
        except BaseException as exc:
            record("termination_sigterm", exc)
        try:
            wait_owned(
                "termination_sigterm_wait",
                PROCESS_TERMINATION_WAIT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            try:
                if group_identity_verified:
                    assert expected_identity is not None
                    _assert_process_identity(
                        expected_identity,
                        "owned controller SIGKILL recheck",
                    )
                    if os.getpgid(process.pid) != process.pid:
                        raise RuntimeError(
                            "owned controller process group changed "
                            "before SIGKILL"
                        )
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                kill_sent = True
            except BaseException as exc:
                record("termination_sigkill", exc)
            wait_owned("termination_sigkill_wait")
    elif running:
        wait_owned("natural_wait")
    else:
        wait_owned("already_exited_wait")

    if not waited:
        try:
            if process.poll() is None:
                if not kill_sent:
                    process.kill()
                    kill_sent = True
                wait_owned("final_exact_child_wait")
            else:
                wait_owned("final_reap_wait")
        except BaseException as exc:
            record("final_reap", exc)

    try:
        residual: bool | None = process.poll() is None
    except BaseException as exc:
        record("final_poll", exc)
        residual = None
    status = (
        "reaped"
        if waited and residual is False
        else "live_residual"
        if residual is True
        else "unknown_residual"
    )
    report = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_controller_process_closure_v1"
        ),
        "controller_pid": process.pid,
        "sealed_process": (
            None if expected_identity is None else dict(expected_identity)
        ),
        "terminate_requested": terminate,
        "group_identity_verified": group_identity_verified,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "wait_observed": waited,
        "wait_return_code": return_code,
        "process_residual": residual,
        "status": status,
        "failures": failures,
        "started_at": started_at,
        "completed_at": _utc_now(),
    }
    report["controller_process_closure_sha256"] = _canonical_digest(
        report, "controller_process_closure_sha256"
    )
    return return_code, report


def _terminate_bound_observer(
    observer_tmux: Mapping[str, Any],
    observer_tmux_server: Mapping[str, Any],
    observer_tmux_owner_seal: Mapping[str, Any],
    observer_process: Mapping[str, int],
    *,
    normal_close: bool = False,
) -> dict[str, Any]:
    started_at = _utc_now()
    sealed_tmux = dict(observer_tmux)
    sealed_server = dict(observer_tmux_server)
    sealed_owner = dict(observer_tmux_owner_seal)
    sealed_process = dict(observer_process)
    pane = str(sealed_tmux["pane"])
    _validate_tmux_identity(sealed_tmux, OBSERVER_SESSION)
    validate_tmux_server_identity(
        sealed_server, "sealed tmux server identity"
    )
    _validate_tmux_owner_seal(
        sealed_owner, sealed_tmux, sealed_server
    )

    def current_process() -> dict[str, int] | None:
        return _process_identity(int(sealed_process["pid"]))

    def process_running_by_stat() -> bool:
        snapshot = _read_process_stat(int(sealed_process["pid"]))
        if snapshot is None:
            return False
        identity, state = snapshot
        return identity == sealed_process and state != "Z"

    def wait_process_not_running(seconds: float) -> str | None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            snapshot = _read_process_stat(
                int(sealed_process["pid"])
            )
            if snapshot is None:
                return "absent"
            current, state = snapshot
            if current != sealed_process:
                return "absent"
            if state == "Z":
                return "zombie"
            time.sleep(0.05)
        return None

    def wait_process_absent() -> bool:
        deadline = time.monotonic() + PROCESS_TERMINATION_WAIT_SECONDS
        while time.monotonic() < deadline:
            if current_process() != sealed_process:
                return True
            time.sleep(0.05)
        return False

    def kill_sealed_process() -> str:
        if current_process() != sealed_process:
            return "already_absent"
        if (
            sealed_process["pgid"] != sealed_process["pid"]
            or sealed_process["pid"] <= 1
        ):
            raise RuntimeError(
                "sealed CPU preflight observer process group differs"
            )
        _assert_process_identity(
            sealed_process, "sealed CPU preflight observer"
        )
        snapshot = _process_identity_state(
            int(sealed_process["pid"])
        )
        if snapshot is None:
            return "already_absent"
        current, state = snapshot
        if current != sealed_process:
            return "already_absent"
        if state == "Z":
            return "zombie"
        try:
            os.killpg(int(sealed_process["pgid"]), signal.SIGKILL)
        except ProcessLookupError:
            if current_process() != sealed_process:
                return "absent"
            raise RuntimeError(
                "sealed CPU preflight observer killpg reported ESRCH "
                "while identity remained live"
            )
        except PermissionError as exc:
            raise RuntimeError(
                "sealed CPU preflight observer killpg permission denied"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "sealed CPU preflight observer killpg failed"
            ) from exc
        disposition = wait_process_not_running(
            PROCESS_TERMINATION_WAIT_SECONDS
        )
        if disposition is None:
            return "survived"
        return disposition

    def current_server() -> dict[str, Any] | None:
        try:
            return _tmux_server_identity()
        except TmuxTargetAbsent:
            return None

    def current_sealed_pane() -> dict[str, Any] | None:
        server = current_server()
        if server != sealed_server:
            return None
        try:
            return _tmux_pane_identity(pane)
        except TmuxTargetAbsent:
            return None

    def owner_residuals() -> tuple[bool, bool]:
        return (
            current_sealed_pane() == sealed_tmux,
            process_running_by_stat(),
        )

    def result(
        status: str,
        *,
        disposition: str | None = None,
        observed_tmux: Mapping[str, Any] | None = None,
        kill_failure: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        pane_residual, process_residual = owner_residuals()
        try:
            named_server = _tmux_server_identity()
            try:
                named_tmux = _tmux_identity(OBSERVER_SESSION)
            except TmuxTargetAbsent:
                named_tmux = None
        except TmuxTargetAbsent:
            named_server = None
            named_tmux = None
        foreign_residual = bool(
            named_tmux is not None
            and (
                named_server != sealed_server
                or named_tmux != sealed_tmux
            )
        )
        value: dict[str, Any] = {
            "session": OBSERVER_SESSION,
            "sealed_tmux": sealed_tmux,
            "sealed_tmux_server": sealed_server,
            "sealed_tmux_owner": sealed_owner,
            "sealed_process": sealed_process,
            "status": status,
            "session_residual": pane_residual,
            "process_residual": process_residual,
            "foreign_session_residual": foreign_residual,
            "foreign_pane_residual": foreign_residual,
            "foreign_tmux": (
                dict(named_tmux) if foreign_residual else None
            ),
            "foreign_tmux_server": (
                dict(named_server) if foreign_residual else None
            ),
            "started_at": started_at,
            "completed_at": _utc_now(),
        }
        if disposition is not None:
            value["sealed_process_disposition"] = disposition
        if observed_tmux is not None:
            value["observed_tmux"] = dict(observed_tmux)
        if kill_failure is not None:
            value["tmux_kill_failure"] = dict(kill_failure)
        return value

    try:
        observed_server = _tmux_server_identity()
    except TmuxTargetAbsent:
        observed_server = None
    try:
        observed_tmux = _tmux_identity(OBSERVER_SESSION)
    except TmuxTargetAbsent:
        observed_tmux = None
    current_pane = current_sealed_pane()
    observed_process = current_process()
    observed_state: str | None = None
    if observed_process == sealed_process:
        observed_snapshot = _process_identity_state(
            int(sealed_process["pid"])
        )
        if observed_snapshot is None:
            observed_process = None
        else:
            observed_process_again, observed_state = observed_snapshot
            if observed_process_again != sealed_process:
                observed_process = observed_process_again
    if current_pane is None and observed_process != sealed_process:
        status = (
            "already_absent"
            if observed_server == sealed_server and observed_tmux is None
            else "identity_replaced_not_terminated"
        )
        return result(status, observed_tmux=observed_tmux)
    if current_pane is None:
        disposition = kill_sealed_process()
        process_absent = wait_process_absent()
        return result(
            (
                f"cleaned_detached_process_{disposition}"
                if process_absent
                else "cleanup_timeout_detached_process_residual"
            ),
            disposition=disposition,
            observed_tmux=observed_tmux,
        )
    if current_pane != sealed_tmux:
        process_disposition = kill_sealed_process()
        process_absent = wait_process_absent()
        return result(
            (
                "identity_replaced_not_terminated"
                if process_absent
                else "cleanup_timeout_detached_process_residual"
            ),
            disposition=process_disposition,
            observed_tmux=current_pane,
        )
    if observed_process == sealed_process:
        _assert_tmux_process_identity(
            OBSERVER_SESSION,
            sealed_tmux,
            sealed_server,
            sealed_process,
        )
    normal_close_verified = normal_close
    disposition = (
        "terminal_consumed"
        if normal_close_verified
        else "terminal_observer_not_live"
        if normal_close
        else "tmux_owner_termination"
    )
    kill_status, kill_result = _conditional_kill_tmux_owner(
        sealed_owner
    )
    kill_failure: dict[str, str] | None = None
    if kill_status != "executed":
        kill_failure = {
            "type": (
                "TmuxConditionalKillRejected"
                if kill_status == "condition_rejected"
                else "TmuxConditionalKillCommandError"
            ),
            "message": (
                TMUX_CONDITIONAL_KILL_REJECTED
                if kill_status == "condition_rejected"
                else kill_result.stderr.strip()
            ),
        }
        post_failure_pane = current_sealed_pane()
        post_failure_process = current_process()
        if (
            post_failure_pane == sealed_tmux
            and post_failure_process == sealed_process
        ):
            raise RuntimeError(
                "failed to kill sealed CPU preflight observer pane while "
                "the sealed pane/process remained live: "
                f"{kill_result.stderr.strip()}"
            )
        if post_failure_process == sealed_process:
            disposition = kill_sealed_process()
    else:
        post_close_snapshot = _read_process_stat(
            int(sealed_process["pid"])
        )
        if post_close_snapshot is None:
            disposition = "absent"
        else:
            post_close_process, post_close_state = post_close_snapshot
            if post_close_process != sealed_process:
                disposition = "absent"
            elif post_close_state == "Z":
                disposition = "zombie"
            else:
                disposition = wait_process_not_running(
                    PROCESS_TERMINATION_WAIT_SECONDS
                )
                if disposition is None:
                    disposition = kill_sealed_process()

    pane_absent = False
    process_absent = False
    cleanup_deadline = time.monotonic() + PROCESS_TERMINATION_WAIT_SECONDS
    while time.monotonic() < cleanup_deadline:
        pane_absent = current_sealed_pane() != sealed_tmux
        process_absent = not process_running_by_stat()
        if pane_absent and process_absent:
            break
        time.sleep(0.05)
    return result(
        (
            (
                "closed_terminal_observer"
                if normal_close_verified
                else f"cleaned_tmux_{disposition}"
            )
            if pane_absent and process_absent
            else "cleanup_timeout_with_residual"
        ),
        disposition=disposition,
        kill_failure=kill_failure,
    )


def _wait_bound_observer_exit(
    observer_tmux: Mapping[str, Any],
    observer_tmux_server: Mapping[str, Any],
    observer_process: Mapping[str, int],
) -> bool:
    deadline = time.monotonic() + PROCESS_TERMINATION_WAIT_SECONDS
    while time.monotonic() < deadline:
        session_alive = (
            subprocess.run(
                ["tmux", "has-session", "-t", OBSERVER_SESSION],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        current_process = _process_identity(int(observer_process["pid"]))
        process_alive = current_process == dict(observer_process)
        if not session_alive and not process_alive:
            return True
        if session_alive:
            try:
                current_tmux = _tmux_identity(OBSERVER_SESSION)
                if current_tmux != dict(observer_tmux):
                    raise RuntimeError(
                        "CPU preflight observer tmux identity differs"
                    )
                if not process_alive:
                    return False
                snapshot = _process_identity_state(
                    int(observer_process["pid"])
                )
                if snapshot is None:
                    return False
                _, state = snapshot
                if state == "Z":
                    return False
                _assert_tmux_process_identity(
                    OBSERVER_SESSION,
                    observer_tmux,
                    observer_tmux_server,
                    observer_process,
                )
            except TmuxTargetAbsent:
                time.sleep(0.05)
                continue
        elif process_alive:
            _assert_process_identity(
                observer_process, "detached CPU preflight observer"
            )
        time.sleep(0.05)
    return False


def _validate_observer_stop(
    path: Path,
    *,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    observer_launch_binding: Mapping[str, str],
    observer_process: Mapping[str, int],
    process_start_binding: Mapping[str, str],
    require_live_identity: bool = True,
) -> dict[str, Any]:
    stop_snapshot = _secure_json_snapshot(
        path, digest_field="observer_stop_sha256"
    )
    assert stop_snapshot is not None
    value = dict(stop_snapshot["value"])
    process_start_snapshot = _secure_json_snapshot(
        Path(process_start_binding["path"]),
        digest_field="controller_process_start_sha256",
    )
    assert process_start_snapshot is not None
    expected_keys = {
        "schema_version",
        "contract_type",
        "campaign_id",
        "policy_sha256",
        "wrapper_claim",
        "observer_launch",
        "observer_claim",
        "observer_ready",
        "controller_process_start",
        "controller_ready",
        "observer_session",
        "observer_pid",
        "observer_process",
        "observer_tmux",
        "controller_process",
        "failure",
        "requested_at",
        "observer_stop_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_canonical_preflight_observer_stop_v2"
        or value.get("policy_sha256") != policy_sha256
        or value.get("wrapper_claim") != dict(wrapper_binding)
        or value.get("observer_launch") != dict(observer_launch_binding)
        or value.get("controller_process_start")
        != dict(process_start_binding)
        or value.get("observer_session") != OBSERVER_SESSION
        or value.get("observer_pid") != observer_process["pid"]
        or value.get("observer_process") != dict(observer_process)
        or value.get("controller_process")
        != process_start_snapshot["value"].get("process")
        or not isinstance(value.get("failure"), Mapping)
        or value.get("observer_stop_sha256")
        != _canonical_digest(value, "observer_stop_sha256")
    ):
        raise RuntimeError("CPU preflight observer stop contract mismatch")
    if value.get("observer_claim") is not None:
        _json_binding(
            Path(value["observer_claim"]["path"]), "observer_claim_sha256"
        )
    if value.get("observer_ready") is not None:
        _json_binding(
            Path(value["observer_ready"]["path"]), "observer_ready_sha256"
        )
    if value.get("controller_ready") is not None:
        _json_binding(
            Path(value["controller_ready"]["path"]), "controller_ready_sha256"
        )
    launch_snapshot = _secure_json_snapshot(
        Path(observer_launch_binding["path"]),
        digest_field="observer_launch_sha256",
    )
    assert launch_snapshot is not None
    launch = dict(launch_snapshot["value"])
    if (
        value.get("observer_tmux") != launch.get("tmux")
        or value.get("observer_process") != launch.get("process")
    ):
        raise RuntimeError(
            "CPU preflight observer stop launch identity differs"
        )
    if require_live_identity:
        if value.get("observer_tmux") != dict(
            _tmux_identity(OBSERVER_SESSION)
        ):
            raise RuntimeError(
                "CPU preflight observer stop tmux identity differs"
            )
        _assert_tmux_process_identity(
            OBSERVER_SESSION,
            value["observer_tmux"],
            launch["tmux_server"],
            observer_process,
        )
    return value


def _normalized_exit(return_code: int) -> tuple[int, int | None]:
    if return_code >= 0:
        return return_code, None
    signal_number = -return_code
    return 128 + signal_number, signal_number


def _read_observer_terminal(
    path: Path,
    process_exit_path: Path,
    *,
    policy_sha256: str,
    observer_launch_binding: Mapping[str, str],
    observer_process: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, str]]:
    def read_bound_json(
        binding: Any,
        *,
        expected_path: Path | None,
        digest_field: str,
        label: str,
    ) -> dict[str, Any]:
        try:
            binding = validate_artifact_binding(
                binding,
                f"CPU preflight observer terminal {label} binding",
            )
        except RuntimeError:
            raise RuntimeError(
                f"CPU preflight observer terminal {label} binding differs"
            )
        bound_path = Path(str(binding["path"]))
        if (
            expected_path is not None
            and bound_path != expected_path
        ):
            raise RuntimeError(
                f"CPU preflight observer terminal {label} path differs"
            )
        try:
            snapshot = _secure_json_snapshot(
                bound_path, digest_field=digest_field
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"CPU preflight observer terminal {label} file is absent"
            ) from exc
        except PermissionError as exc:
            raise RuntimeError(
                f"CPU preflight observer terminal {label} "
                "file permission denied"
            ) from exc
        assert snapshot is not None
        if snapshot["sha256"] != binding["sha256"]:
            raise RuntimeError(
                f"CPU preflight observer terminal {label} file SHA differs"
            )
        if (
            snapshot["binding"]["canonical_sha256"]
            != binding["canonical_sha256"]
        ):
            raise RuntimeError(
                f"CPU preflight observer terminal {label} "
                "canonical SHA differs"
            )
        return dict(snapshot["value"])

    terminal_snapshot = _secure_json_snapshot(
        path, digest_field="observer_terminal_sha256"
    )
    assert terminal_snapshot is not None
    value = dict(terminal_snapshot["value"])
    canonical = value.get("observer_terminal_sha256")
    if (
        value.get("contract_type")
        != "safa_canonical_preflight_observer_terminal_v1"
        or value.get("policy_sha256") != policy_sha256
        or value.get("status") not in {"completed", "failed"}
        or (
            value.get("status") == "completed"
            and value.get("failure") is not None
        )
        or (
            value.get("status") == "completed"
            and (
                not isinstance(value.get("observer_claim"), Mapping)
                or not isinstance(value.get("observer_ready"), Mapping)
            )
        )
        or (
            value.get("status") == "failed"
            and not isinstance(value.get("failure"), Mapping)
        )
        or not isinstance(canonical, str)
        or _canonical_digest(value, "observer_terminal_sha256") != canonical
    ):
        raise RuntimeError("CPU preflight observer terminal binding differs")
    read_bound_json(
        value.get("controller_process_exit"),
        expected_path=process_exit_path,
        digest_field="controller_process_exit_sha256",
        label="controller process exit",
    )
    claim: dict[str, Any] | None = None
    if value.get("observer_claim") is not None:
        claim = read_bound_json(
            value["observer_claim"],
            expected_path=path.parent / "observer_claim.json",
            digest_field="observer_claim_sha256",
            label="claim",
        )
        if (
            claim.get("contract_type")
            != "safa_canonical_preflight_observer_claim_v1"
            or claim.get("phase") != "preflight"
            or claim.get("policy_sha256") != policy_sha256
            or claim.get("observer_session") != OBSERVER_SESSION
            or claim.get("observer_pid") != observer_process["pid"]
            or claim.get("observer_launch") != dict(observer_launch_binding)
            or claim.get("observer_process") != dict(observer_process)
        ):
            raise RuntimeError(
                "CPU preflight observer terminal claim identity differs"
            )
    if value.get("observer_ready") is not None:
        ready = read_bound_json(
            value["observer_ready"],
            expected_path=path.parent / "observer_ready.json",
            digest_field="observer_ready_sha256",
            label="ready",
        )
        if (
            ready.get("contract_type")
            != "safa_canonical_preflight_observer_ready_v1"
            or ready.get("phase") != "preflight"
            or ready.get("policy_sha256") != policy_sha256
            or ready.get("observer_session") != OBSERVER_SESSION
            or ready.get("observer_pid") != observer_process["pid"]
            or ready.get("observer_claim") != value.get("observer_claim")
            or ready.get("observer_claim_sha256")
            != (
                None
                if claim is None
                else claim.get("observer_claim_sha256")
            )
            or ready.get("observer_launch") != dict(observer_launch_binding)
            or ready.get("observer_process") != dict(observer_process)
        ):
            raise RuntimeError(
                "CPU preflight observer terminal ready identity differs"
            )
    return value, dict(terminal_snapshot["binding"])


def _wait_observer_terminal(
    path: Path,
    process_exit_path: Path,
    *,
    policy_sha256: str,
    observer_launch_binding: Mapping[str, str],
    observer_process: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, str]] | None:
    deadline = time.monotonic() + OBSERVER_TERMINAL_WAIT_SECONDS
    while _secure_read_file(path, missing_ok=True) is None:
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)
    return _read_observer_terminal(
        path,
        process_exit_path,
        policy_sha256=policy_sha256,
        observer_launch_binding=observer_launch_binding,
        observer_process=observer_process,
    )


def _validate_terminal_stop_binding(
    terminal_value: Mapping[str, Any],
    *,
    observer_stop_path: Path,
    policy_sha256: str,
    wrapper_binding: Mapping[str, str],
    observer_launch_binding: Mapping[str, str],
    observer_process: Mapping[str, int],
    process_start_binding: Mapping[str, str],
    observer_stop_binding: dict[str, str] | None,
) -> dict[str, str] | None:
    if (
        observer_stop_binding is None
        and _secure_read_file(
            observer_stop_path, missing_ok=True
        )
        is not None
    ):
        _validate_observer_stop(
            observer_stop_path,
            policy_sha256=policy_sha256,
            wrapper_binding=wrapper_binding,
            observer_launch_binding=observer_launch_binding,
            observer_process=observer_process,
            process_start_binding=process_start_binding,
            require_live_identity=False,
        )
        observer_stop_binding = _json_binding(
            observer_stop_path, "observer_stop_sha256"
        )
    if observer_stop_binding is not None:
        if terminal_value.get("observer_stop") != observer_stop_binding:
            raise RuntimeError(
                "CPU preflight observer terminal stop binding differs"
            )
    elif terminal_value.get("status") == "failed":
        raise RuntimeError("failed CPU preflight observer terminal has no stop")
    return observer_stop_binding


def _launcher_git_state(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, arguments in (
        ("head_sha", ("rev-parse", "HEAD")),
        ("origin_master_sha", ("rev-parse", "origin/master")),
        ("branch", ("branch", "--show-current")),
    ):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stderr.strip():
            raise RuntimeError(
                f"wrapper Git {name} query failed: "
                f"{result.stderr.strip()}"
            )
        values[name] = result.stdout.strip()
    if (
        values["branch"] != "master"
        or values["head_sha"] != values["origin_master_sha"]
    ):
        raise RuntimeError("wrapper Git state differs from master/origin")
    return values


def _wrapper_file_identity(path: Path) -> dict[str, Any]:
    value = path.stat()
    return build_file_identity(
        path=str(path.resolve()),
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
    )


def _open_regular_json_with_identity(
    path: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = _secure_json_snapshot(path)
    if snapshot is None:
        raise RuntimeError(f"{label} is absent")
    value = snapshot["identity"]
    return dict(snapshot["value"]), build_file_identity(
        path=str(path),
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
    )


def _opened_file_identity(path: Path) -> dict[str, Any]:
    _payload, identity = _open_regular_json_with_identity(
        path, "opened file"
    )
    return identity


def _validate_pane_fault_consumer_runtime(
    *,
    receipt: Mapping[str, Any],
    expected_attempt_root: Path,
    label: str,
) -> dict[str, Any]:
    registration = validate_pane_fault_consumer_registration(
        receipt.get("pane_fault_consumer"),
        expected_namespace=str(
            expected_attempt_root / "pane_fault_consumer"
        ),
        label=f"{label} registration",
    )
    artifacts = registration["artifacts"]
    records = {
        name: _secure_json_snapshot(
            Path(artifacts[name]), digest_field=digest_field
        )
        for name, digest_field in {
            "ready": "consumer_ready_sha256",
            "started": "consumer_started_sha256",
            "active": "consumer_active_sha256",
            "reader_release": "consumer_reader_release_sha256",
            "release_observed": (
                "consumer_release_observed_sha256"
            ),
        }.items()
    }
    if any(snapshot is None for snapshot in records.values()):
        raise RuntimeError(
            f"{label} required consumer artifact is absent"
        )
    values = {
        name: dict(snapshot["value"])
        for name, snapshot in records.items()
        if snapshot is not None
    }
    expected_keys = {
        "ready": {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "attempt_id",
            "consumer_attempt",
            "consumer_session",
            "consumer_owner_nonce",
            "consumer_wait_supervisor_ready",
            "supervisor_owner_seal",
            "supervisor_process",
            "supervisor_command",
            "supervisor_executable",
            "tmux",
            "tmux_server",
            "worker_process",
            "worker_command",
            "worker_executable",
            "fault_descriptor",
            "pane_fault_channel",
            "self_fault_descriptor",
            "consumer_self_fault_channel",
            "ready_at",
            "consumer_ready_sha256",
        },
        "started": {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "attempt_id",
            "consumer_attempt",
            "consumer_ready",
            "consumer_wait_supervisor_ready",
            "owner_seal",
            "supervisor_process",
            "worker_process",
            "pane_fault_channel",
            "consumer_self_fault_channel",
            "remain_on_exit",
            "started_at",
            "consumer_started_sha256",
        },
        "active": {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "attempt_id",
            "consumer_attempt",
            "consumer_accepted",
            "consumer_commit",
            "consumer_session",
            "consumer_owner_nonce",
            "owner_seal",
            "supervisor_process",
            "worker_process",
            "pane_fault_channel",
            "consumer_self_fault_channel",
            "active_at",
            "consumer_active_sha256",
        },
        "reader_release": {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "attempt_id",
            "consumer_attempt",
            "consumer_commit",
            "consumer_active",
            "launcher_gate_reader_release_intent",
            "last_empty_snapshots",
            "released_at",
            "consumer_reader_release_sha256",
        },
        "release_observed": {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "attempt_id",
            "consumer_attempt",
            "consumer_active",
            "consumer_reader_release",
            "consumer_session",
            "consumer_owner_nonce",
            "owner_seal",
            "supervisor_process",
            "worker_process",
            "release_observed_at",
            "consumer_release_observed_sha256",
        },
    }
    expected_types = {
        "ready": "safa_pane_fault_consumer_ready_v1",
        "started": "safa_pane_fault_consumer_started_v1",
        "active": "safa_pane_fault_consumer_transfer_active_v1",
        "reader_release": (
            "safa_pane_fault_consumer_reader_release_intent_v1"
        ),
        "release_observed": (
            "safa_pane_fault_consumer_release_observed_v1"
        ),
    }
    expected_digests = {
        "ready": "consumer_ready_sha256",
        "started": "consumer_started_sha256",
        "active": "consumer_active_sha256",
        "reader_release": "consumer_reader_release_sha256",
        "release_observed": (
            "consumer_release_observed_sha256"
        ),
    }
    for name, value in values.items():
        digest_field = expected_digests[name]
        if (
            set(value) != expected_keys[name]
            or value.get("schema_version") != 1
            or value.get("contract_type") != expected_types[name]
            or value.get("policy_sha256")
            != receipt.get("policy_sha256")
            or value.get("attempt_id") != receipt.get("attempt_id")
            or value.get(digest_field)
            != _canonical_digest(value, digest_field)
        ):
            raise RuntimeError(
                f"{label} {name} contract differs"
            )
    chain = build_pane_fault_consumer_chain(
        consumer_started=records["started"]["binding"],
        consumer_active=records["active"]["binding"],
        consumer_reader_release=records["reader_release"][
            "binding"
        ],
        consumer_release_observed=records["release_observed"][
            "binding"
        ],
        registration=registration,
    )
    ready = values["ready"]
    active = values["active"]
    reader_release = values["reader_release"]
    observed = values["release_observed"]
    if (
        reader_release["consumer_active"]
        != chain["consumer_active"]
        or observed["consumer_active"]
        != chain["consumer_active"]
        or observed["consumer_reader_release"]
        != chain["consumer_reader_release"]
        or active["consumer_attempt"]
        != reader_release["consumer_attempt"]
        or active["consumer_attempt"]
        != observed["consumer_attempt"]
        or observed["owner_seal"] != active["owner_seal"]
        or observed["supervisor_process"]
        != active["supervisor_process"]
        or observed["worker_process"] != active["worker_process"]
        or observed["consumer_session"]
        != active["consumer_session"]
        or observed["consumer_owner_nonce"]
        != active["consumer_owner_nonce"]
        or ready["consumer_attempt"]
        != active["consumer_attempt"]
        or ready["consumer_session"]
        != active["consumer_session"]
        or ready["consumer_owner_nonce"]
        != active["consumer_owner_nonce"]
        or ready["supervisor_process"]
        != active["supervisor_process"]
        or ready["worker_process"] != active["worker_process"]
    ):
        raise RuntimeError(
            f"{label} artifact reference relation differs"
    )
    session = str(active["consumer_session"])
    live_tmux = _tmux_identity(session)
    live_tmux_status = _tmux_runtime_status(session)
    live_tmux_server = _tmux_server_identity(
        str(live_tmux["pane"])
    )
    live_owner_nonce = _tmux_owner_nonce(
        session, str(live_tmux_server["socket_path"])
    )
    live_owner_authority = _build_tmux_owner_seal(
        live_tmux,
        live_tmux_server,
        live_owner_nonce,
    )
    live_supervisor = _require_process_identity(
        int(active["supervisor_process"]["pid"]),
        f"{label} live supervisor",
    )
    live_worker = _require_process_identity(
        int(active["worker_process"]["pid"]),
        f"{label} live worker",
    )
    live_supervisor_command = _process_command(
        live_supervisor["pid"]
    )
    live_worker_command = _process_command(live_worker["pid"])
    live_process_command_name = Path(
        f"/proc/{live_supervisor['pid']}/comm"
    ).read_text(encoding="utf-8").rstrip("\n")
    if not live_process_command_name:
        raise RuntimeError(
            f"{label} live process command name is empty"
        )
    live_supervisor_executable = _process_executable_identity(
        live_supervisor["pid"]
    )
    live_worker_executable = _process_executable_identity(
        live_worker["pid"]
    )
    live_cwd = Path(
        os.readlink(f"/proc/{live_worker['pid']}/cwd")
    ).resolve(strict=True)
    expected_cwd = Path(
        str(receipt["bindings"]["launcher"]["path"])
    ).parent.parent.resolve(strict=True)
    live_owner_seal = {
        "owner_nonce": live_owner_authority["owner_nonce"],
        "pane": live_tmux_status["pane"],
        "pane_dead": live_tmux_status["pane_dead"],
        "pane_dead_status": live_tmux_status["pane_dead_status"],
        "pane_pid": live_tmux_status["pane_pid"],
        "pane_process": live_supervisor,
        "session": live_tmux_status["session"],
        "tmux_server": live_tmux_server,
    }
    expected_ready_tmux = {
        key: live_tmux_status[key]
        for key in (
            "session",
            "pane",
            "pane_pid",
            "pane_dead",
            "pane_dead_status",
        )
    }
    live_checks = {
        "supervisor_process": (
            live_supervisor == active["supervisor_process"]
        ),
        "worker_process": live_worker == active["worker_process"],
        "pane_pid": live_tmux_status.get("pane_pid")
        == live_supervisor["pid"],
        "direct_worker": (
            live_worker["ppid"] == live_supervisor["pid"]
            and live_worker["pgid"] == live_worker["pid"]
            and live_worker["sid"] == live_worker["pid"]
        ),
        "pane_live": live_tmux_status.get("pane_dead") is False,
        "pane_pipe": live_tmux_status.get("pane_pipe") is False,
        "tmux_identity_status": {
            key: live_tmux_status[key]
            for key in ("session", "pane", "pane_pid")
        }
        == {
            key: live_tmux[key]
            for key in ("session", "pane", "pane_pid")
        },
        "pane_current_command": (
            live_tmux_status["pane_current_command"]
            == live_tmux["pane_current_command"]
            == live_process_command_name
        ),
        "owner_nonce": live_owner_nonce
        == active["consumer_owner_nonce"],
        "owner_seal": live_owner_seal == active["owner_seal"],
        "ready_tmux": ready["tmux"] == expected_ready_tmux,
        "ready_tmux_server": ready["tmux_server"]
        == live_tmux_server,
        "supervisor_command": (
            ready["supervisor_command"]
            == live_supervisor_command
        ),
        "supervisor_command_bytes": (
            _command_bytes(live_supervisor_command)
            == _process_command_bytes(live_supervisor["pid"])
        ),
        "supervisor_executable": (
            ready["supervisor_executable"]
            == live_supervisor_executable
        ),
        "worker_command": (
            ready["worker_command"] == live_worker_command
        ),
        "worker_command_bytes": (
            _command_bytes(live_worker_command)
            == _process_command_bytes(live_worker["pid"])
        ),
        "worker_executable": (
            ready["worker_executable"] == live_worker_executable
        ),
        "cwd": live_cwd == expected_cwd,
    }
    failed_live_checks = [
        name for name, passed in live_checks.items() if not passed
    ]
    if failed_live_checks:
        raise RuntimeError(
            f"{label} live consumer seal differs: "
            f"{failed_live_checks}"
        )
    return {
        "registration": registration,
        "chain": chain,
        "active": active,
    }


def _validate_preflight_launch_receipt(
    *,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    controller_tmux: Mapping[str, Any],
    controller_tmux_server: Mapping[str, Any],
) -> dict[str, Any]:
    raw_receipt_path = os.environ.get(LAUNCH_RECEIPT_PATH_ENV)
    raw_accepted_path = os.environ.get(LAUNCH_ACCEPTED_PATH_ENV)
    raw_release_path = os.environ.get(LAUNCH_RELEASE_PATH_ENV)
    raw_log_path = os.environ.get(PANE_LOG_PATH_ENV)
    if (
        raw_receipt_path is None
        or raw_accepted_path is None
        or raw_release_path is None
        or raw_log_path is None
    ):
        raise RuntimeError(
            "formal preflight launcher environment is incomplete"
        )
    receipt_path = Path(raw_receipt_path).resolve()
    accepted_path = Path(raw_accepted_path).resolve()
    release_path = Path(raw_release_path).resolve()
    log_path = Path(raw_log_path).resolve()
    receipt_snapshot = _secure_json_snapshot(
        receipt_path, digest_field="launch_receipt_sha256"
    )
    assert receipt_snapshot is not None
    receipt = dict(receipt_snapshot["value"])
    validate_launch_receipt_schema(
        receipt,
        expected_gate_worker_arguments=_process_command(
            os.getppid()
        ),
        expected_consumer_worker_arguments=[
            sys.executable,
            "-B",
            "-u",
            str(
                Path(__file__).resolve().with_name(
                    "run_canonical_preflight_launcher.py"
                )
            ),
            "__pane_fault_consumer__",
            "--attempt-path",
            str(
                receipt["pane_fault_consumer"]["artifacts"][
                    "attempt"
                ]
            ),
            "--config",
            str(config.resolve()),
        ],
        label="wrapper launch receipt v4",
    )
    receipt_stat = receipt_snapshot["identity"]
    receipt_identity = build_file_identity(
        path=str(receipt_path),
        device=int(receipt_stat.st_dev),
        inode=int(receipt_stat.st_ino),
        mode=int(receipt_stat.st_mode),
        size=int(receipt_stat.st_size),
    )
    validate_file_identity(
        receipt_identity, "wrapper launch receipt identity"
    )
    attempt_id = receipt.get("attempt_id")
    expected_attempt_root = (
        policy_root.parents[1]
        / "preflight_launch_attempts"
        / "by_policy"
        / policy_sha256
        / str(attempt_id)
    ).resolve()
    gate_ready_path = expected_attempt_root / "pane_gate_ready.json"
    supervisor_ready_path = (
        expected_attempt_root / "gate_wait_supervisor_ready.json"
    )
    tmux_started_path = expected_attempt_root / "launch_tmux_started.json"
    wrapper_started_path = expected_attempt_root / "wrapper_started.json"
    gate_execution_terminal_path = (
        expected_attempt_root / "gate_execution_terminal.json"
    )
    gate_ready_snapshot = _secure_json_snapshot(
        gate_ready_path, digest_field="pane_gate_ready_sha256"
    )
    supervisor_ready_snapshot = _secure_json_snapshot(
        supervisor_ready_path,
        digest_field="gate_wait_supervisor_ready_sha256",
    )
    tmux_started_snapshot = _secure_json_snapshot(
        tmux_started_path,
        digest_field="launch_tmux_started_sha256",
    )
    assert gate_ready_snapshot is not None
    assert supervisor_ready_snapshot is not None
    assert tmux_started_snapshot is not None
    gate_ready_binding = dict(gate_ready_snapshot["binding"])
    tmux_started_binding = dict(
        tmux_started_snapshot["binding"]
    )
    process_arguments = _process_command(os.getpid())
    executable_path = str(
        Path(os.readlink(f"/proc/{os.getpid()}/exe")).resolve()
    )
    executable = _process_executable_identity(os.getpid())
    config_binding = {
        "path": str(config.resolve()),
        "sha256": _sha256_file(config),
    }
    receipt_binding = dict(receipt_snapshot["binding"])
    live_implementations = _reverify_verified_preflight_apis()
    tmux_started = dict(tmux_started_snapshot["value"])
    supervisor_ready = dict(supervisor_ready_snapshot["value"])
    validate_tmux_started(
        tmux_started,
        verified_implementations=live_implementations,
        tmux_identity=controller_tmux,
        tmux_server=controller_tmux_server,
        label="wrapper launch tmux started",
    )
    sealed_implementations = validate_verified_implementations(
        receipt.get("verified_implementations"),
        "wrapper launch receipt verified implementations",
    )
    log_identity = _wrapper_file_identity(log_path)
    if (
        receipt.get("schema_version") != 4
        or receipt.get("contract_type")
        != LAUNCH_RECEIPT_CONTRACT_TYPE
        or not isinstance(attempt_id, str)
        or len(attempt_id) != 64
        or any(character not in "0123456789abcdef" for character in attempt_id)
        or receipt_path != expected_attempt_root / "launch_receipt.json"
        or accepted_path != expected_attempt_root / "launch_accepted.json"
        or release_path
        != expected_attempt_root / "launch_ownership_release.json"
        or log_path != expected_attempt_root / "pane.log"
        or receipt.get("wrapper_started_path")
        != str(wrapper_started_path)
        or receipt.get("gate_execution_terminal_path")
        != str(gate_execution_terminal_path)
        or receipt.get("gate_lifecycle_wait_supervisor_ready_path")
        != str(supervisor_ready_path)
        or receipt.get("policy_sha256") != policy_sha256
        or receipt.get("controller_session") != CONTROLLER_SESSION
        or receipt.get("controller_owner_nonce")
        != _validate_tmux_owner_nonce(
            os.environ.get(TMUX_OWNER_ENV)
        )
        or receipt.get("observer_session") != OBSERVER_SESSION
        or receipt.get("wrapper_arguments") != process_arguments
        or receipt.get("shell") is not False
        or receipt.get("pane_log") != log_identity
        or receipt.get("fault_channel", {}).get("path")
        != str(expected_attempt_root / "wrapper_fault.channel")
        or receipt.get(
            "pane_gate_fault_channel", {}
        ).get("path")
        != str(
            expected_attempt_root / "pane_gate_fault.channel"
        )
        or receipt.get("pane_gate_fault_publisher")
        != {
            **dict(
                receipt.get("bindings", {}).get(
                    "launcher", {}
                )
            ),
            "role": "launcher_pane_gate",
        }
        or receipt.get("git") != _launcher_git_state(repo_root)
        or receipt.get("bindings", {}).get("config")
        != config_binding
        or sealed_implementations != live_implementations
        or receipt.get("bindings", {}).get("wrapper")
        != {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        }
        or receipt.get("python_executable", {}).get("path")
        != executable_path
        or receipt.get("python_executable", {}).get("sha256")
        != _sha256_file(Path(executable_path))
        or receipt.get("launch_receipt_sha256")
        != _canonical_digest(receipt, "launch_receipt_sha256")
    ):
        raise RuntimeError("formal preflight launch receipt differs")
    consumer_runtime = _validate_pane_fault_consumer_runtime(
        receipt=receipt,
        expected_attempt_root=expected_attempt_root,
        label="wrapper receipt pane fault consumer",
    )
    deadline = time.monotonic() + LAUNCH_ACCEPTED_WAIT_SECONDS
    while (
        _secure_read_file(
            wrapper_started_path, missing_ok=True
        )
        is None
    ):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "formal preflight wrapper-start evidence timed out"
            )
        time.sleep(0.005)
    wrapper_started_snapshot = _secure_json_snapshot(
        wrapper_started_path,
        digest_field="wrapper_started_sha256",
    )
    assert wrapper_started_snapshot is not None
    wrapper_started = dict(wrapper_started_snapshot["value"])
    wrapper_launch_process = _launch_process_identity(os.getpid())
    gate_ready = dict(gate_ready_snapshot["value"])
    validate_gate_ready(
        gate_ready,
        verified_implementations=live_implementations,
        label="wrapper pane gate ready",
    )
    validate_wrapper_started(
        wrapper_started,
        verified_implementations=live_implementations,
        gate_ready=gate_ready,
        label="wrapper start evidence",
    )
    if (
        not isinstance(wrapper_started, dict)
        or gate_ready.get("launch_receipt") != receipt_binding
        or gate_ready.get("launch_receipt_identity")
        != receipt_identity
        or gate_ready.get("verified_implementations")
        != live_implementations
        or tmux_started.get("launch_receipt") != receipt_binding
        or tmux_started.get("launch_receipt_identity")
        != receipt_identity
        or tmux_started.get("verified_implementations")
        != live_implementations
        or tmux_started.get("pane_gate_ready")
        != gate_ready_binding
        or tmux_started.get("owner_seal", {}).get("pane_process")
        != supervisor_ready.get("supervisor_process")
        or supervisor_ready.get("gate_worker_process")
        != gate_ready.get("process")
        or gate_ready.get("wrapper_arguments")
        != process_arguments
        or gate_ready.get("pane_gate_ready_sha256")
        != _canonical_digest(gate_ready, "pane_gate_ready_sha256")
        or wrapper_started.get("launch_receipt") != receipt_binding
        or wrapper_started.get("launch_receipt_identity")
        != receipt_identity
        or wrapper_started.get("verified_implementations")
        != live_implementations
        or wrapper_started.get("pane_gate_ready")
        != gate_ready_binding
        or wrapper_started.get("pane_gate_process")
        != gate_ready.get("process")
        or gate_ready.get("process")
        != _launch_process_identity(os.getppid())
        or gate_ready.get("process", {}).get("pgid") != os.getppid()
        or gate_ready.get("process", {}).get("sid") != os.getppid()
        or _process_command_bytes(os.getppid())
        != _command_bytes(receipt["gate_worker_arguments"])
        or str(Path(os.readlink(f"/proc/{os.getppid()}/exe")).resolve())
        != receipt["python_executable"]["path"]
        or wrapper_started.get("wrapper_arguments")
        != process_arguments
        or wrapper_started.get("wrapper_process")
        != wrapper_launch_process
        or wrapper_launch_process.get("ppid")
        != wrapper_started.get("pane_gate_process", {}).get("pid")
        or wrapper_launch_process.get("pgid") != os.getpid()
        or wrapper_launch_process.get("sid") != os.getpid()
        or wrapper_started.get("wrapper_executable") != executable
        or _process_command_bytes(os.getpid())
        != _command_bytes(process_arguments)
        or wrapper_started.get("wrapper_started_sha256")
        != _canonical_digest(
            wrapper_started, "wrapper_started_sha256"
        )
    ):
        raise RuntimeError("formal preflight wrapper-start evidence differs")
    return {
        "attempt_id": attempt_id,
        "receipt": receipt,
        "receipt_binding": receipt_binding,
        "receipt_identity": receipt_identity,
        "verified_implementations": live_implementations,
        "gate_ready_binding": gate_ready_binding,
        "gate_ready": gate_ready,
        "tmux_started_binding": tmux_started_binding,
        "wrapper_started_binding": dict(
            wrapper_started_snapshot["binding"]
        ),
        "wrapper_started": wrapper_started,
        "gate_supervisor_process": supervisor_ready[
            "supervisor_process"
        ],
        "gate_process": wrapper_started["pane_gate_process"],
        "wrapper_launch_process": wrapper_launch_process,
        "accepted_path": accepted_path,
        "release_path": release_path,
        "wrapper_arguments": process_arguments,
        "wrapper_executable": executable,
        "pane_log": log_identity,
        "git": dict(receipt["git"]),
        "pane_fault_consumer_registration": (
            consumer_runtime["registration"]
        ),
        "pane_fault_consumer_chain": consumer_runtime["chain"],
        "pane_fault_consumer_active": consumer_runtime["active"],
    }


def _wait_preflight_launch_release(
    *,
    launch: Mapping[str, Any],
    wrapper_binding: Mapping[str, str],
) -> dict[str, Any]:
    accepted_path = Path(str(launch["accepted_path"]))
    release_path = Path(str(launch["release_path"]))
    terminal_path = accepted_path.with_name("launch_terminal.json")
    deadline = time.monotonic() + LAUNCH_ACCEPTED_WAIT_SECONDS
    while _secure_read_file(release_path, missing_ok=True) is None:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "formal preflight launch acceptance timed out"
            )
        time.sleep(0.02)
    accepted_snapshot = _secure_json_snapshot(
        accepted_path, digest_field="launch_accepted_sha256"
    )
    terminal_snapshot = _secure_json_snapshot(
        terminal_path, digest_field="launch_terminal_sha256"
    )
    release_snapshot = _secure_json_snapshot(
        release_path,
        digest_field="launch_ownership_release_sha256",
    )
    assert accepted_snapshot is not None
    assert terminal_snapshot is not None
    assert release_snapshot is not None
    accepted = dict(accepted_snapshot["value"])
    terminal = dict(terminal_snapshot["value"])
    release = dict(release_snapshot["value"])
    validate_ownership_chain(
        accepted,
        terminal,
        release,
        receipt_binding=launch["receipt_binding"],
        receipt_identity=launch["receipt_identity"],
        wrapper_binding=wrapper_binding,
        accepted_binding=dict(accepted_snapshot["binding"]),
        terminal_binding=dict(terminal_snapshot["binding"]),
        verified_implementations=launch[
            "verified_implementations"
        ],
        pane_fault_consumer_chain=launch[
            "pane_fault_consumer_chain"
        ],
        label="formal preflight launch ownership chain",
    )
    if (
        accepted.get("attempt_id") != launch["attempt_id"]
        or _opened_file_identity(
            Path(str(launch["receipt_identity"]["path"]))
        )
        != launch["receipt_identity"]
    ):
        raise RuntimeError("formal preflight launch release differs")
    return release


def _run_wrapped_controller_owned(
    *,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    command: Sequence[str],
    observer_command: Sequence[str],
    emergency_state: dict[str, Any],
) -> dict[str, Any]:
    verified_implementations = _install_verified_preflight_apis(
        config
    )
    exact_policy_root = policy_root.resolve()
    control = _ensure_secure_leaf_directories(
        exact_policy_root, ("preflight_control",)
    )
    wrapper_claim_path = control / "wrapper_claim.json"
    process_log_path = control / "controller_process.log"
    process_exit_path = control / "controller_process_exit.json"
    process_closure_path = control / "controller_process_closure.json"
    process_start_path = control / "controller_process_start.json"
    observer_launch_path = control / "observer_launch.json"
    observer_bootstrap_path = control / "observer_bootstrap.json"
    observer_gate_ready_path = control / "observer_gate_ready.json"
    observer_gate_release_path = control / "observer_gate_release.json"
    observer_stop_path = control / "observer_stop.json"
    observer_cleanup_path = control / "observer_cleanup.json"
    wrapper_exit_path = control / "wrapper_exit.json"
    plan_path = policy_root.resolve() / "checkpoint_plan.json"
    request_manifest_path = (
        policy_root.resolve() / "checkpoint_preflight/preflight_request_manifest.json"
    )
    started_at = _utc_now()
    emergency_state.update(
        {
            "control": control,
            "wrapper_exit_path": wrapper_exit_path,
            "observer_cleanup_path": observer_cleanup_path,
            "process_closure_path": process_closure_path,
            "policy_sha256": policy_sha256,
            "command": list(command),
            "started_at": started_at,
        }
    )
    controller_session = _tmux_session()
    controller_tmux = _tmux_identity(CONTROLLER_SESSION)
    controller_tmux_server = _tmux_server_identity(
        controller_tmux["pane"]
    )
    launch = _validate_preflight_launch_receipt(
        repo_root=repo_root,
        policy_root=policy_root,
        policy_sha256=policy_sha256,
        config=config,
        controller_tmux=controller_tmux,
        controller_tmux_server=controller_tmux_server,
    )
    wrapper_process = _require_process_identity(
        os.getpid(), "CPU preflight wrapper"
    )
    _assert_tmux_process_identity(
        CONTROLLER_SESSION,
        controller_tmux,
        controller_tmux_server,
        launch["gate_supervisor_process"],
    )
    _assert_process_identity(
        launch["gate_process"], "CPU preflight gate worker"
    )
    if (
        launch["gate_process"]["ppid"]
        != launch["gate_supervisor_process"]["pid"]
        or wrapper_process["ppid"] != launch["gate_process"]["pid"]
    ):
        raise RuntimeError(
            "preflight supervisor/gate/wrapper process chain differs"
        )
    verified_implementations = _reverify_verified_preflight_apis()
    if verified_implementations != launch["verified_implementations"]:
        raise RuntimeError(
            "verified implementations changed before wrapper claim"
        )
    claim = build_claim_v3(
        attempt_id=launch["attempt_id"],
        preflight_launch_receipt=launch["receipt_binding"],
        preflight_launch_receipt_identity=launch["receipt_identity"],
        verified_implementations=verified_implementations,
        pane_gate_ready=launch["gate_ready_binding"],
        preflight_launch_tmux_started=launch["tmux_started_binding"],
        preflight_wrapper_started=launch["wrapper_started_binding"],
        pane_gate_process=launch["gate_process"],
        wrapper_arguments=launch["wrapper_arguments"],
        wrapper_executable=launch["wrapper_executable"],
        pane_log=launch["pane_log"],
        git=launch["git"],
        policy_sha256=policy_sha256,
        config={
            "path": str(config.resolve()),
            "sha256": _sha256_file(config),
        },
        checkpoint_plan=_json_binding(
            plan_path, "checkpoint_plan_sha256"
        ),
        preflight_request_manifest=_json_binding(
            request_manifest_path, "preflight_request_manifest_sha256"
        ),
        controller_session=controller_session,
        controller_tmux=controller_tmux,
        controller_tmux_server=controller_tmux_server,
        observer_session=OBSERVER_SESSION,
        command=list(command),
        observer_command=list(observer_command),
        wrapper_pid=os.getpid(),
        wrapper_process=wrapper_process,
        wrapper_launch_process=launch["wrapper_launch_process"],
        started_at=started_at,
        external_timeout_seconds=None,
        pane_fault_consumer_chain=(
            launch["pane_fault_consumer_chain"]
        ),
        gate_ready=launch["gate_ready"],
        wrapper_started=launch["wrapper_started"],
    )
    if (
        _reverify_verified_preflight_apis()
        != claim["verified_implementations"]
    ):
        raise RuntimeError(
            "verified implementations changed after claim construction"
        )
    _write_exclusive(wrapper_claim_path, claim)
    wrapper_binding = _json_binding(
        wrapper_claim_path, "wrapper_claim_sha256"
    )
    emergency_state["claim"] = claim
    emergency_state["wrapper_binding"] = wrapper_binding
    emergency_state["launch"] = launch
    _wait_preflight_launch_release(
        launch=launch,
        wrapper_binding=wrapper_binding,
    )
    observer_launch_failure: dict[str, str] | None = None
    observer_owner_nonce = secrets.token_hex(
        TMUX_OWNER_NONCE_HEX_LENGTH // 2
    )
    provisional_tmux: dict[str, Any] | None = None
    provisional_tmux_server: dict[str, Any] | None = None
    provisional_tmux_owner_seal: dict[str, Any] | None = None
    provisional_process: dict[str, int] | None = None
    provisional_process_snapshot_failure: dict[str, str] | None = None
    observer_gate_probe: dict[str, Any] | None = None
    observer_gate_client: dict[str, Any] | None = None
    observer_gate_ready: dict[str, Any] | None = None
    observer_gate_release: dict[str, Any] | None = None
    observer_tmux: dict[str, Any] | None = None
    observer_tmux_server: dict[str, Any] | None = None
    observer_tmux_owner_seal: dict[str, Any] | None = None
    observer_process: dict[str, int] | None = None
    observer_bootstrap: dict[str, Any] | None = None

    def record_exact_owner(
        tmux_identity: Mapping[str, Any],
        tmux_server: Mapping[str, Any],
        owner_seal: Mapping[str, Any],
        process_identity: Mapping[str, int] | None,
    ) -> None:
        candidate = {
            "tmux": dict(tmux_identity),
            "tmux_server": dict(tmux_server),
            "owner_seal": dict(owner_seal),
            "process": (
                None
                if process_identity is None
                else dict(process_identity)
            ),
            "process_snapshot_failure": None,
        }
        previous = emergency_state.get("provisional_observer")
        if previous is not None:
            previous_tmux = previous.get("tmux")
            if (
                not isinstance(previous_tmux, Mapping)
                or {
                    key: previous_tmux.get(key)
                    for key in ("session", "pane", "pane_pid")
                }
                != {
                    key: candidate["tmux"][key]
                    for key in ("session", "pane", "pane_pid")
                }
                or previous.get("tmux_server")
                != candidate["tmux_server"]
                or previous.get("owner_seal")
                != candidate["owner_seal"]
            ):
                raise RuntimeError(
                    "emergency observer owner identity changed"
                )
            previous_process = previous.get("process")
            if (
                previous_process is not None
                and candidate["process"] is not None
                and previous_process != candidate["process"]
            ):
                raise RuntimeError(
                    "emergency observer process identity changed"
                )
            if candidate["process"] is None:
                candidate["process"] = previous_process
        emergency_state["provisional_observer"] = candidate
        if candidate["process"] is not None:
            emergency_state["observer"] = {
                "tmux": candidate["tmux"],
                "tmux_server": candidate["tmux_server"],
                "owner_seal": candidate["owner_seal"],
                "process": candidate["process"],
            }

    try:
        (
            observer_gate_probe,
            observer_gate_client,
        ) = _launch_and_probe_observer_gate(
            repo_root=repo_root,
            ready_path=observer_gate_ready_path,
            release_path=observer_gate_release_path,
            bootstrap_path=observer_bootstrap_path,
            policy_sha256=policy_sha256,
            wrapper_binding=wrapper_binding,
            owner_nonce=observer_owner_nonce,
            observer_command=observer_command,
            owner_recorder=record_exact_owner,
        )
        provisional_tmux = observer_gate_probe.get(
            "best_tmux"
        ) or observer_gate_probe.get("tmux")
        provisional_tmux_server = observer_gate_probe.get(
            "best_tmux_server"
        ) or observer_gate_probe.get("tmux_server")
        provisional_tmux_owner_seal = observer_gate_probe.get(
            "best_tmux_owner_seal"
        ) or observer_gate_probe.get("tmux_owner_seal")
        provisional_process = observer_gate_probe.get(
            "best_process"
        ) or observer_gate_probe.get("process")
        process_probe = observer_gate_probe.get("process_probe")
        if (
            isinstance(process_probe, Mapping)
            and process_probe.get("status") == "error"
            and isinstance(process_probe.get("failure"), Mapping)
        ):
            provisional_process_snapshot_failure = dict(
                process_probe["failure"]
            )
        emergency_state["provisional_observer"] = {
            "tmux": provisional_tmux,
            "tmux_server": provisional_tmux_server,
            "owner_seal": provisional_tmux_owner_seal,
            "process": provisional_process,
            "process_snapshot_failure": (
                provisional_process_snapshot_failure
            ),
        }
        observer_gate_ready = observer_gate_probe.get("gate_ready")
        if observer_gate_probe.get("status") != "exact_ready":
            raise RuntimeError(
                "CPU preflight observer gate probe did not establish "
                f"an exact ready owner: {observer_gate_probe!r}"
            )
        assert provisional_tmux_owner_seal is not None
        assert observer_gate_ready is not None
        _set_observer_remain_on_exit(
            provisional_tmux_owner_seal
        )
        observer_gate_ready_binding = _json_binding(
            observer_gate_ready_path, "observer_gate_ready_sha256"
        )
        observer_gate_release = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_observer_gate_release_v1"
            ),
            "policy_sha256": policy_sha256,
            "verified_implementations": (
                _reverify_verified_preflight_apis()
            ),
            "wrapper_claim": wrapper_binding,
            "observer_gate_ready": observer_gate_ready_binding,
            "observer_session": OBSERVER_SESSION,
            "owner_nonce": observer_owner_nonce,
            "observer_command": list(observer_command),
            "released_at": _utc_now(),
        }
        observer_gate_release[
            "observer_gate_release_sha256"
        ] = _canonical_digest(
            observer_gate_release,
            "observer_gate_release_sha256",
        )
        _write_exclusive(
            observer_gate_release_path, observer_gate_release
        )
        (
            observer_tmux,
            observer_tmux_server,
            observer_tmux_owner_seal,
            observer_process,
            observer_bootstrap,
        ) = _wait_tmux_process_identity(
            OBSERVER_SESSION,
            observer_owner_nonce,
            observer_bootstrap_path,
            policy_sha256=policy_sha256,
            wrapper_binding=wrapper_binding,
            expected_command=observer_command,
        )
        emergency_state["observer"] = {
            "tmux": observer_tmux,
            "tmux_server": observer_tmux_server,
            "owner_seal": observer_tmux_owner_seal,
            "process": observer_process,
        }
        if (
            {
                key: observer_tmux[key]
                for key in ("session", "pane", "pane_pid")
            }
            != {
                key: provisional_tmux[key]
                for key in ("session", "pane", "pane_pid")
            }
            or observer_tmux_server != provisional_tmux_server
            or observer_tmux_owner_seal
            != provisional_tmux_owner_seal
            or (
                provisional_process is not None
                and observer_process != provisional_process
            )
        ):
            raise RuntimeError(
                "tmux observer identity differs from provisional owner seal"
            )
        observer_launch_status = "launched"
    except BaseException as exc:
        _propagate_publish_error(exc)
        recorded_owner = emergency_state.get(
            "provisional_observer"
        )
        if recorded_owner is not None:
            provisional_tmux = recorded_owner.get("tmux")
            provisional_tmux_server = recorded_owner.get(
                "tmux_server"
            )
            provisional_tmux_owner_seal = recorded_owner.get(
                "owner_seal"
            )
            provisional_process = recorded_owner.get("process")
            provisional_process_snapshot_failure = (
                recorded_owner.get("process_snapshot_failure")
            )
        observer_tmux = None
        observer_tmux_server = None
        observer_tmux_owner_seal = None
        observer_process = None
        observer_bootstrap = None
        observer_launch_status = "failed"
        observer_launch_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    observer_launch = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_observer_launch_v3"
            if observer_launch_status == "launched"
            else "safa_canonical_preflight_observer_launch_failed_v1"
        ),
        "policy_sha256": policy_sha256,
        "verified_implementations": (
            _reverify_verified_preflight_apis()
        ),
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        "observer_session": OBSERVER_SESSION,
        "command": list(observer_command),
        "observer_bootstrap": (
            None
            if observer_bootstrap is None
            else _json_binding(
                observer_bootstrap_path, "observer_bootstrap_sha256"
            )
        ),
        "tmux": observer_tmux,
        "tmux_server": observer_tmux_server,
        "tmux_owner_seal": observer_tmux_owner_seal,
        "process": observer_process,
        "observer_gate_ready": (
            None
            if observer_gate_ready is None
            else _json_binding(
                observer_gate_ready_path, "observer_gate_ready_sha256"
            )
        ),
        "observer_gate_release": (
            None
            if observer_gate_release is None
            else _json_binding(
                observer_gate_release_path,
                "observer_gate_release_sha256",
            )
        ),
        "status": observer_launch_status,
        "failure": observer_launch_failure,
        "completed_at": _utc_now(),
    }
    if observer_launch_status == "failed":
        observer_launch.update(
            {
                "observer_gate_client": observer_gate_client,
                "observer_gate_probe": observer_gate_probe,
                "provisional_tmux": provisional_tmux,
                "provisional_tmux_server": provisional_tmux_server,
                "provisional_tmux_owner_seal": (
                    provisional_tmux_owner_seal
                ),
                "provisional_process": provisional_process,
                "provisional_process_snapshot_failure": (
                    provisional_process_snapshot_failure
                ),
            }
        )
    observer_launch["observer_launch_sha256"] = _canonical_digest(
        observer_launch, "observer_launch_sha256"
    )
    try:
        _write_exclusive(observer_launch_path, observer_launch)
        observer_launch_binding: dict[str, str] | None = _json_binding(
            observer_launch_path, "observer_launch_sha256"
        )
        emergency_state["observer_launch_binding"] = (
            observer_launch_binding
        )
    except BaseException as exc:
        _propagate_publish_error(exc)
        observer_launch_failure = _merge_launch_failure(
            observer_launch_failure,
            stage="observer_launch_write",
            failure_type=type(exc).__name__,
            message=str(exc),
        )
        observer_launch_binding = None
    if observer_launch_failure is not None:
        durable_launch_failure = dict(observer_launch_failure)
        if "stage" not in durable_launch_failure:
            durable_launch_failure = _merge_launch_failure(
                None,
                stage="observer_launch",
                failure_type=str(observer_launch_failure["type"]),
                message=str(observer_launch_failure["message"]),
            )
        if (
            provisional_tmux is not None
            and provisional_tmux_server is not None
            and provisional_tmux_owner_seal is not None
        ):
            try:
                launch_termination = _terminate_provisional_tmux_owner(
                    provisional_tmux,
                    provisional_tmux_server,
                    provisional_tmux_owner_seal,
                    provisional_process,
                    provisional_process_snapshot_failure,
                )
            except BaseException as exc:
                launch_termination = {
                    "session": OBSERVER_SESSION,
                    "sealed_tmux": dict(provisional_tmux),
                    "sealed_tmux_server": dict(
                        provisional_tmux_server
                    ),
                    "sealed_tmux_owner": dict(
                        provisional_tmux_owner_seal
                    ),
                    "sealed_process": (
                        None
                        if provisional_process is None
                        else dict(provisional_process)
                    ),
                    "status": "cleanup_failed",
                    "session_residual": True,
                    "process_residual": (
                        None
                        if provisional_process_snapshot_failure is not None
                        else provisional_process is not None
                    ),
                    "foreign_session_residual": None,
                    "foreign_pane_residual": None,
                    "foreign_tmux": None,
                    "foreign_tmux_server": None,
                    "tmux_kill_status": "exception",
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "started_at": _utc_now(),
                    "completed_at": _utc_now(),
                }
        else:
            launch_termination = {
                "session": OBSERVER_SESSION,
                "sealed_tmux": provisional_tmux,
                "sealed_tmux_server": provisional_tmux_server,
                "sealed_tmux_owner": provisional_tmux_owner_seal,
                "sealed_process": provisional_process,
                "status": "observer_owner_not_sealed",
                "session_residual": bool(
                    observer_gate_probe
                    and observer_gate_probe.get("session_residual")
                ),
                "process_residual": (
                    False
                    if observer_gate_probe
                    and observer_gate_probe.get("status") == "absent"
                    else None
                ),
                "foreign_session_residual": None,
                "foreign_pane_residual": None,
                "foreign_tmux": None,
                "foreign_tmux_server": None,
                "tmux_kill_status": "not_attempted",
                "failure": {
                    "type": "ObserverOwnerSealUnavailable",
                    "message": (
                        "observer owner seal was unavailable after "
                        "tmux new-session"
                    ),
                },
                "started_at": _utc_now(),
                "completed_at": _utc_now(),
            }
        launch_cleanup = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_observer_cleanup_v1"
            ),
            "policy_sha256": policy_sha256,
            "wrapper_claim": wrapper_binding,
            "observer_launch": observer_launch_binding,
            "reason": "observer_launch_failed",
            **launch_termination,
        }
        launch_cleanup["observer_cleanup_sha256"] = _canonical_digest(
            launch_cleanup, "observer_cleanup_sha256"
        )
        try:
            _write_exclusive(observer_cleanup_path, launch_cleanup)
            observer_cleanup_binding = _json_binding(
                observer_cleanup_path, "observer_cleanup_sha256"
            )
        except BaseException as exc:
            _propagate_publish_error(exc)
            durable_launch_failure = _merge_launch_failure(
                durable_launch_failure,
                stage="observer_cleanup_write",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            observer_cleanup_binding = None
        not_started = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_controller_process_"
                "not_started_v1"
            ),
            "policy_sha256": policy_sha256,
            "wrapper_claim": wrapper_binding,
            "observer_launch": observer_launch_binding,
            "command": list(command),
            "status": "not_started",
            "process": None,
            "reason": dict(durable_launch_failure),
            "started_at": None,
            "completed_at": _utc_now(),
        }
        not_started["controller_process_start_sha256"] = (
            _canonical_digest(
                not_started, "controller_process_start_sha256"
            )
        )
        try:
            _write_exclusive(process_start_path, not_started)
            not_started_binding = _json_binding(
                process_start_path, "controller_process_start_sha256"
            )
        except BaseException as exc:
            _propagate_publish_error(exc)
            durable_launch_failure = _merge_launch_failure(
                durable_launch_failure,
                stage="controller_not_started_write",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            not_started_binding = None
        not_started_exit = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_controller_process_"
                "exit_not_started_v1"
            ),
            "policy_sha256": policy_sha256,
            "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
            "observer_launch": observer_launch_binding,
            "controller_process_start": not_started_binding,
            "observer_stop": None,
            "controller_pid": None,
            "command": list(command),
            "status": "not_started",
            "exit_code": None,
            "signal": None,
            "launch_failure": dict(durable_launch_failure),
            "controller_process_log": None,
            "controller_claim": None,
            "controller_terminal": None,
            "completed_at": _utc_now(),
        }
        not_started_exit[
            "controller_process_exit_sha256"
        ] = _canonical_digest(
            not_started_exit, "controller_process_exit_sha256"
        )
        try:
            _write_exclusive(process_exit_path, not_started_exit)
            process_exit_binding = _json_binding(
                process_exit_path, "controller_process_exit_sha256"
            )
        except BaseException as exc:
            _propagate_publish_error(exc)
            durable_launch_failure = _merge_launch_failure(
                durable_launch_failure,
                stage="controller_not_started_exit_write",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            process_exit_binding = None
        failed_wrapper_exit = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_wrapper_exit_v4"
            ),
            "policy_sha256": policy_sha256,
            "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
            "command": list(command),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "exit_code": 125,
            "controller_exit_code": None,
            "signal": None,
            "launch_failure": dict(durable_launch_failure),
            "controller_process_exit": process_exit_binding,
            "controller_process_log": None,
            "observer_launch": observer_launch_binding,
            "controller_process_start": not_started_binding,
            "controller_claim": None,
            "controller_terminal": None,
            "observer_terminal": None,
            "observer_terminal_snapshot": None,
            "late_observer_terminal": None,
            "late_observer_terminal_snapshot": None,
            "observer_terminal_validation_failure": None,
            "late_observer_terminal_validation_failure": None,
            "observer_stop": None,
            "observer_cleanup": observer_cleanup_binding,
        }
        failed_wrapper_exit["wrapper_exit_sha256"] = (
            _canonical_digest(
                failed_wrapper_exit, "wrapper_exit_sha256"
            )
        )
        return _publish_wrapper_exit_total(
            wrapper_exit_path, failed_wrapper_exit
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    process: subprocess.Popen[Any] | None = None
    return_code: int | None = None
    controller_pid: int | None = None
    controller_process_start: dict[str, Any] | None = None
    controller_process_start_binding: dict[str, str] | None = None
    controller_process_closure: dict[str, Any] | None = None
    controller_process_closure_binding: dict[str, str] | None = None
    process_exit_binding: dict[str, str] | None = None
    observer_stop_binding: dict[str, str] | None = None
    observer_cleanup_binding: dict[str, str] | None = None
    observer_cleanup_residual = False
    observer_cleanup_failed = False
    observer_terminal: dict[str, str] | None = None
    observer_terminal_snapshot: dict[str, str] | None = None
    late_observer_terminal: dict[str, str] | None = None
    late_observer_terminal_snapshot: dict[str, str] | None = None
    observer_terminal_value: dict[str, Any] | None = None
    observer_terminal_validation_failure: dict[str, str] | None = None
    late_observer_terminal_validation_failure: dict[str, str] | None = None
    launch_failure: dict[str, Any] | None = None
    publication_poison: ExclusivePublishError | None = None

    def optional_artifact_binding(
        path: Path, stage: str
    ) -> dict[str, str] | None:
        nonlocal launch_failure
        try:
            return _optional_binding(path)
        except BaseException as exc:
            _propagate_publish_error(exc)
            launch_failure = _merge_launch_failure(
                launch_failure,
                stage=stage,
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            return None

    try:
        live_consumer = _validate_pane_fault_consumer_runtime(
            receipt=launch["receipt"],
            expected_attempt_root=Path(
                launch[
                    "pane_fault_consumer_registration"
                ]["namespace"]
            ).parent,
            label="wrapper controller conversion pane fault consumer",
        )
        if (
            live_consumer["chain"]
            != launch["pane_fault_consumer_chain"]
        ):
            raise RuntimeError(
                "pane fault consumer chain changed before controller start"
            )
        descriptor = os.open(process_log_path, flags, 0o644)
        controller_environment = dict(os.environ)
        controller_environment[OBSERVER_SESSION_ENV] = OBSERVER_SESSION
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=descriptor,
            close_fds=True,
            start_new_session=True,
            env=controller_environment,
        )
        emergency_state["controller_process"] = process
        controller_pid = process.pid
        controller_process_start = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_controller_process_start_v1",
            "policy_sha256": policy_sha256,
            "verified_implementations": (
                _reverify_verified_preflight_apis()
            ),
            "wrapper_claim": wrapper_binding,
            "observer_launch": observer_launch_binding,
            "command": list(command),
            "process": _require_process_identity(
                process.pid, "CPU preflight controller"
            ),
            "started_at": _utc_now(),
        }
        if controller_process_start["process"]["pgid"] != process.pid:
            raise RuntimeError("CPU preflight controller process group differs")
        emergency_state["controller_identity"] = (
            controller_process_start["process"]
        )
        controller_process_start[
            "controller_process_start_sha256"
        ] = _canonical_digest(
            controller_process_start,
            "controller_process_start_sha256",
        )
        try:
            _write_exclusive(
                process_start_path, controller_process_start
            )
        except BaseException as exc:
            launch_failure = _merge_launch_failure(
                launch_failure,
                stage="controller_process_start_write",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        try:
            controller_process_start_binding = _json_binding(
                process_start_path, "controller_process_start_sha256"
            )
        except BaseException as exc:
            launch_failure = _merge_launch_failure(
                launch_failure,
                stage="controller_process_start_binding",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        while process.poll() is None:
            try:
                _assert_tmux_process_identity(
                    OBSERVER_SESSION,
                    observer_tmux,
                    observer_tmux_server,
                    observer_process,
                )
                if (
                    _secure_read_file(
                        observer_stop_path, missing_ok=True
                    )
                    is not None
                ):
                    stop = _validate_observer_stop(
                        observer_stop_path,
                        policy_sha256=policy_sha256,
                        wrapper_binding=wrapper_binding,
                        observer_launch_binding=observer_launch_binding,
                        observer_process=observer_process,
                        process_start_binding=controller_process_start_binding,
                    )
                    observer_stop_binding = _json_binding(
                        observer_stop_path, "observer_stop_sha256"
                    )
                    launch_failure = _merge_launch_failure(
                        launch_failure,
                        stage="controller_monitor_observer_hard_stop",
                        failure_type="ObserverHardStop",
                        message=str(stop["failure"]),
                    )
                    break
            except BaseException as exc:
                _propagate_publish_error(exc)
                launch_failure = _merge_launch_failure(
                    launch_failure,
                    stage="controller_monitor",
                    failure_type=type(exc).__name__,
                    message=str(exc),
                )
                break
            time.sleep(0.1)
    except ExclusivePublishError as exc:
        publication_poison = exc
        raise
    except BaseException as exc:
        launch_failure = _merge_launch_failure(
            launch_failure,
            stage="controller_launch_or_start",
            failure_type=type(exc).__name__,
            message=str(exc),
        )
    finally:
        if process is not None:
            sealed_process = (
                None
                if controller_process_start is None
                else controller_process_start.get("process")
            )
            try:
                return_code, controller_process_closure = (
                    _close_owned_controller_process(
                        process,
                        sealed_process,
                        terminate=(
                            publication_poison is not None
                            or launch_failure is not None
                        ),
                    )
                )
            except BaseException as exc:
                if publication_poison is not None:
                    publication_poison.add_secondary_failure(
                        stage="controller_process_reap",
                        failure=exc,
                    )
                else:
                    raise
            if (
                publication_poison is None
                and controller_process_closure is not None
            ):
                if controller_process_closure["failures"]:
                    for closure_failure in controller_process_closure[
                        "failures"
                    ]:
                        launch_failure = _merge_launch_failure(
                            launch_failure,
                            stage=(
                                "controller_closure."
                                f"{closure_failure['stage']}"
                            ),
                            failure_type=str(
                                closure_failure["type"]
                            ),
                            message=str(
                                closure_failure["message"]
                            ),
                        )
                if (
                    not controller_process_closure["wait_observed"]
                    or controller_process_closure[
                        "process_residual"
                    ]
                    is not False
                ):
                    launch_failure = _merge_launch_failure(
                        launch_failure,
                        stage="controller_closure.residual",
                        failure_type="ControllerProcessResidual",
                        message=(
                            "controller process was not proven reaped "
                            "by wait"
                        ),
                    )
                try:
                    _write_exclusive(
                        process_closure_path,
                        controller_process_closure,
                    )
                    controller_process_closure_binding = (
                        _json_binding(
                            process_closure_path,
                            "controller_process_closure_sha256",
                        )
                    )
                except BaseException as exc:
                    _propagate_publish_error(exc)
                    launch_failure = _merge_launch_failure(
                        launch_failure,
                        stage="controller_closure_write",
                        failure_type=type(exc).__name__,
                        message=str(exc),
                    )
        if (
            publication_poison is not None
            and observer_tmux is not None
            and observer_tmux_server is not None
            and observer_tmux_owner_seal is not None
            and observer_process is not None
        ):
            try:
                observer_termination = (
                    _terminate_bound_observer(
                        observer_tmux,
                        observer_tmux_server,
                        observer_tmux_owner_seal,
                        observer_process,
                    )
                )
                if (
                    observer_termination.get(
                        "session_residual"
                    )
                    is not False
                    or observer_termination.get(
                        "process_residual"
                    )
                    is not False
                ):
                    publication_poison.add_secondary_failure(
                        stage="observer_reap",
                        failure=RuntimeError(
                            "observer reap left a residual"
                        ),
                    )
            except BaseException as exc:
                publication_poison.add_secondary_failure(
                    stage="observer_reap",
                    failure=exc,
                )
        if descriptor is not None:
            if publication_poison is not None:
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    publication_poison.add_secondary_failure(
                        stage="controller_log_close",
                        failure=exc,
                    )
            else:
                try:
                    try:
                        os.fsync(descriptor)
                    except BaseException as exc:
                        launch_failure = _merge_launch_failure(
                            launch_failure,
                            stage="controller_log_fsync",
                            failure_type=type(exc).__name__,
                            message=str(exc),
                        )
                finally:
                    try:
                        os.close(descriptor)
                    except BaseException as exc:
                        launch_failure = _merge_launch_failure(
                            launch_failure,
                            stage="controller_log_close",
                            failure_type=type(exc).__name__,
                            message=str(exc),
                        )
    if (
        process is not None
        and controller_process_closure is not None
        and controller_process_closure["wait_observed"]
        and return_code is not None
    ):
        exit_code, signal_number = _normalized_exit(return_code)
        if (
            _secure_read_file(
                observer_stop_path, missing_ok=True
            )
            is not None
            and observer_stop_binding is None
        ):
            try:
                stop = _validate_observer_stop(
                    observer_stop_path,
                    policy_sha256=policy_sha256,
                    wrapper_binding=wrapper_binding,
                    observer_launch_binding=observer_launch_binding,
                    observer_process=observer_process,
                    process_start_binding=controller_process_start_binding,
                    require_live_identity=False,
                )
                observer_stop_binding = _json_binding(
                    observer_stop_path, "observer_stop_sha256"
                )
            except BaseException as exc:
                launch_failure = _merge_launch_failure(
                    launch_failure,
                    stage="observer_stop_validation",
                    failure_type=type(exc).__name__,
                    message=str(exc),
                )
        process_exit = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_controller_process_exit_v2",
            "policy_sha256": policy_sha256,
            "verified_implementations": (
                _reverify_verified_preflight_apis()
            ),
            "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
            "observer_launch": observer_launch_binding,
            "controller_process_start": controller_process_start_binding,
            "controller_process_closure": (
                controller_process_closure_binding
            ),
            "observer_stop": observer_stop_binding,
            "controller_pid": controller_pid,
            "command": list(command),
            "exit_code": exit_code,
            "signal": signal_number,
            "launch_failure": launch_failure,
            "controller_process_log": optional_artifact_binding(
                process_log_path, "controller_process_log_binding"
            ),
            "controller_claim": optional_artifact_binding(
                control / "controller_claim.json",
                "controller_claim_binding",
            ),
            "controller_terminal": optional_artifact_binding(
                control / "controller_terminal.json",
                "controller_terminal_binding",
            ),
            "completed_at": _utc_now(),
        }
        process_exit["controller_process_exit_sha256"] = _canonical_digest(
            process_exit, "controller_process_exit_sha256"
        )
        try:
            _write_exclusive(process_exit_path, process_exit)
        except BaseException as exc:
            _propagate_publish_error(exc)
            launch_failure = _merge_launch_failure(
                launch_failure,
                stage="controller_process_exit_write",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            process_exit = None
            process_exit_binding = None
        else:
            try:
                process_exit_binding = _json_binding(
                    process_exit_path, "controller_process_exit_sha256"
                )
            except BaseException as exc:
                launch_failure = _merge_launch_failure(
                    launch_failure,
                    stage="controller_process_exit_binding",
                    failure_type=type(exc).__name__,
                    message=str(exc),
                )
                process_exit = None
                process_exit_binding = None
    else:
        exit_code = 125
        signal_number = None
        process_exit = None
    try:
        observer_terminal_snapshot_result = _wait_observer_terminal(
            control / "observer_terminal.json",
            process_exit_path,
            policy_sha256=policy_sha256,
            observer_launch_binding=observer_launch_binding,
            observer_process=observer_process,
        )
        if observer_terminal_snapshot_result is not None:
            (
                observer_terminal_value,
                observer_terminal_snapshot,
            ) = observer_terminal_snapshot_result
            if observer_stop_binding is None:
                observer_stop_binding = _validate_terminal_stop_binding(
                    observer_terminal_value,
                    observer_stop_path=observer_stop_path,
                    policy_sha256=policy_sha256,
                    wrapper_binding=wrapper_binding,
                    observer_launch_binding=observer_launch_binding,
                    observer_process=observer_process,
                    process_start_binding=controller_process_start_binding,
                    observer_stop_binding=observer_stop_binding,
                )
            observer_terminal = observer_terminal_snapshot
    except BaseException as exc:
        observer_terminal = None
        observer_terminal_validation_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        launch_failure = _merge_launch_failure(
            launch_failure,
            stage="observer_terminal_validation",
            failure_type="ObserverTerminalValidationError",
            message=str(exc),
        )
    if observer_terminal is None:
        try:
            termination = _terminate_bound_observer(
                observer_tmux,
                observer_tmux_server,
                observer_tmux_owner_seal,
                observer_process,
            )
        except BaseException as exc:
            termination = {
                "session": OBSERVER_SESSION,
                "sealed_tmux": dict(observer_tmux),
                "sealed_tmux_server": dict(observer_tmux_server),
                "sealed_tmux_owner": dict(observer_tmux_owner_seal),
                "sealed_process": dict(observer_process),
                "status": "cleanup_failed",
                "session_residual": True,
                "process_residual": True,
                "foreign_session_residual": None,
                "foreign_pane_residual": None,
                "foreign_tmux": None,
                "foreign_tmux_server": None,
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "started_at": _utc_now(),
                "completed_at": _utc_now(),
            }
            observer_cleanup_failed = True
        if (
            not termination["process_residual"]
            and observer_terminal_validation_failure is None
        ):
            try:
                (
                    late_terminal_value,
                    late_observer_terminal_snapshot,
                ) = _read_observer_terminal(
                    control / "observer_terminal.json",
                    process_exit_path,
                    policy_sha256=policy_sha256,
                    observer_launch_binding=observer_launch_binding,
                    observer_process=observer_process,
                )
            except FileNotFoundError:
                late_terminal_value = None
            except BaseException as exc:
                late_terminal_value = None
                late_observer_terminal_snapshot = None
                late_observer_terminal = None
                late_observer_terminal_validation_failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            if late_terminal_value is not None:
                try:
                    observer_stop_binding = _validate_terminal_stop_binding(
                        late_terminal_value,
                        observer_stop_path=observer_stop_path,
                        policy_sha256=policy_sha256,
                        wrapper_binding=wrapper_binding,
                        observer_launch_binding=observer_launch_binding,
                        observer_process=observer_process,
                        process_start_binding=controller_process_start_binding,
                        observer_stop_binding=observer_stop_binding,
                    )
                    late_observer_terminal = (
                        late_observer_terminal_snapshot
                    )
                except BaseException as exc:
                    late_observer_terminal = None
                    late_observer_terminal_validation_failure = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
        cleanup = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_observer_cleanup_v1",
            "policy_sha256": policy_sha256,
            "wrapper_claim": wrapper_binding,
            "observer_launch": observer_launch_binding,
            "reason": (
                "observer_terminal_validation_failed"
                if observer_terminal_validation_failure is not None
                else "observer_terminal_timeout"
            ),
            "late_observer_terminal": late_observer_terminal,
            "observer_terminal_snapshot": observer_terminal_snapshot,
            "late_observer_terminal_snapshot": (
                late_observer_terminal_snapshot
            ),
            "observer_terminal_validation_failure": (
                observer_terminal_validation_failure
            ),
            "late_observer_terminal_validation_failure": (
                late_observer_terminal_validation_failure
            ),
            **termination,
        }
        cleanup["observer_cleanup_sha256"] = _canonical_digest(
            cleanup, "observer_cleanup_sha256"
        )
        try:
            _write_exclusive(observer_cleanup_path, cleanup)
            observer_cleanup_binding = _json_binding(
                observer_cleanup_path, "observer_cleanup_sha256"
            )
        except BaseException as exc:
            _propagate_publish_error(exc)
            launch_failure = _merge_launch_failure(
                launch_failure,
                stage="observer_cleanup_write",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            observer_cleanup_binding = None
            observer_cleanup_failed = True
        observer_cleanup_residual = bool(
            cleanup["session_residual"] or cleanup["process_residual"]
        )
    else:
        try:
            termination = _terminate_bound_observer(
                observer_tmux,
                observer_tmux_server,
                observer_tmux_owner_seal,
                observer_process,
                normal_close=True,
            )
        except BaseException as exc:
            termination = {
                "session": OBSERVER_SESSION,
                "sealed_tmux": dict(observer_tmux),
                "sealed_tmux_server": dict(observer_tmux_server),
                "sealed_tmux_owner": dict(observer_tmux_owner_seal),
                "sealed_process": dict(observer_process),
                "status": "cleanup_failed",
                "session_residual": True,
                "process_residual": True,
                "foreign_session_residual": None,
                "foreign_pane_residual": None,
                "foreign_tmux": None,
                "foreign_tmux_server": None,
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "started_at": _utc_now(),
                "completed_at": _utc_now(),
            }
        cleanup = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_observer_cleanup_v1",
            "policy_sha256": policy_sha256,
            "wrapper_claim": wrapper_binding,
            "observer_launch": observer_launch_binding,
            "reason": "observer_terminal_consumed",
            **termination,
        }
        cleanup["observer_cleanup_sha256"] = _canonical_digest(
            cleanup, "observer_cleanup_sha256"
        )
        try:
            _write_exclusive(observer_cleanup_path, cleanup)
            observer_cleanup_binding = _json_binding(
                observer_cleanup_path, "observer_cleanup_sha256"
            )
        except BaseException as exc:
            _propagate_publish_error(exc)
            launch_failure = _merge_launch_failure(
                launch_failure,
                stage="observer_cleanup_write",
                failure_type=type(exc).__name__,
                message=str(exc),
            )
            observer_cleanup_binding = None
            observer_cleanup_failed = True
        observer_cleanup_residual = bool(
            cleanup["session_residual"] or cleanup["process_residual"]
        )
        observer_cleanup_failed = (
            observer_cleanup_failed
            or cleanup["status"] != "closed_terminal_observer"
        )
    observer_status = (
        None
        if observer_terminal_value is None
        else observer_terminal_value.get("status")
    )
    effective_exit_code = exit_code
    strict_binding_failure: dict[str, str] | None = None
    try:
        if controller_process_start_binding is None:
            raise RuntimeError(
                "controller process start binding is absent"
            )
        if controller_process_closure_binding is None:
            raise RuntimeError(
                "controller process closure binding is absent"
            )
        if process_exit is None:
            raise RuntimeError("controller process exit binding is absent")
        if wrapper_binding != _json_binding(
            wrapper_claim_path, "wrapper_claim_sha256"
        ):
            raise RuntimeError("wrapper claim binding changed")
        if observer_launch_binding != _json_binding(
            observer_launch_path, "observer_launch_sha256"
        ):
            raise RuntimeError("observer launch binding changed")
        if controller_process_start_binding != _json_binding(
            process_start_path, "controller_process_start_sha256"
        ):
            raise RuntimeError(
                "controller process start binding changed"
            )
        if controller_process_closure_binding != _json_binding(
            process_closure_path, "controller_process_closure_sha256"
        ):
            raise RuntimeError(
                "controller process closure binding changed"
            )
        process_exit_binding = _json_binding(
            process_exit_path, "controller_process_exit_sha256"
        )
        if process_exit_binding["canonical_sha256"] != process_exit[
            "controller_process_exit_sha256"
        ]:
            raise RuntimeError(
                "controller process exit binding changed"
            )
        if observer_cleanup_binding != _json_binding(
            observer_cleanup_path, "observer_cleanup_sha256"
        ):
            raise RuntimeError("observer cleanup binding changed")
        if observer_terminal != observer_terminal_snapshot:
            raise RuntimeError(
                "observer terminal strict snapshot binding changed"
            )
    except BaseException as exc:
        strict_binding_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    strict_success = (
        launch_failure is None
        and strict_binding_failure is None
        and process_exit is not None
        and exit_code == 0
        and process_exit.get("launch_failure") is None
        and controller_process_closure is not None
        and controller_process_closure.get("status") == "reaped"
        and controller_process_closure.get("wait_observed") is True
        and controller_process_closure.get("wait_return_code")
        == return_code
        and controller_process_closure.get("process_residual") is False
        and controller_process_closure.get("failures") == []
        and observer_terminal is not None
        and observer_status == "completed"
        and observer_terminal_validation_failure is None
        and late_observer_terminal is None
        and late_observer_terminal_validation_failure is None
        and not observer_cleanup_residual
        and not observer_cleanup_failed
        and cleanup.get("status") == "closed_terminal_observer"
        and observer_cleanup_binding is not None
    )
    if not strict_success:
        if effective_exit_code == 0:
            effective_exit_code = 124
        if strict_binding_failure is not None:
            launch_failure = _merge_launch_failure(
                launch_failure,
                stage="strict_binding_validation",
                failure_type=str(strict_binding_failure["type"]),
                message=str(strict_binding_failure["message"]),
            )
        if launch_failure is None:
            launch_failure = _merge_launch_failure(
                None,
                stage="strict_success_gate",
                failure_type="RuntimeError",
                message=(
                    "CPU preflight wrapper strict success conjunction "
                    "was not satisfied"
                    + (
                        ""
                        if strict_binding_failure is None
                        else f": {strict_binding_failure}"
                    )
                ),
            )
    process_log_binding = optional_artifact_binding(
        process_log_path, "wrapper_process_log_binding"
    )
    controller_claim_binding = optional_artifact_binding(
        control / "controller_claim.json",
        "wrapper_controller_claim_binding",
    )
    controller_terminal_binding = optional_artifact_binding(
        control / "controller_terminal.json",
        "wrapper_controller_terminal_binding",
    )
    if launch_failure is not None and effective_exit_code == 0:
        effective_exit_code = 124
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_exit_v4",
        "policy_sha256": policy_sha256,
        "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        "command": list(command),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "exit_code": effective_exit_code,
        "controller_exit_code": exit_code,
        "signal": signal_number,
        "launch_failure": launch_failure,
        "controller_process_exit": process_exit_binding,
        "controller_process_closure": (
            controller_process_closure_binding
        ),
        "controller_process_log": process_log_binding,
        "observer_launch": observer_launch_binding,
        "controller_process_start": controller_process_start_binding,
        "controller_claim": controller_claim_binding,
        "controller_terminal": controller_terminal_binding,
        "observer_terminal": observer_terminal,
        "observer_terminal_snapshot": observer_terminal_snapshot,
        "late_observer_terminal": late_observer_terminal,
        "late_observer_terminal_snapshot": (
            late_observer_terminal_snapshot
        ),
        "observer_terminal_validation_failure": (
            observer_terminal_validation_failure
        ),
        "late_observer_terminal_validation_failure": (
            late_observer_terminal_validation_failure
        ),
        "observer_stop": observer_stop_binding,
        "observer_cleanup": observer_cleanup_binding,
    }
    value["wrapper_exit_sha256"] = _canonical_digest(
        value, "wrapper_exit_sha256"
    )
    return _publish_wrapper_exit_total(wrapper_exit_path, value)


def run_wrapped_controller(
    *,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    command: Sequence[str],
    observer_command: Sequence[str],
) -> dict[str, Any]:
    emergency_state: dict[str, Any] = {}
    try:
        return _run_wrapped_controller_owned(
            repo_root=repo_root,
            policy_root=policy_root,
            policy_sha256=policy_sha256,
            config=config,
            command=command,
            observer_command=observer_command,
            emergency_state=emergency_state,
        )
    except BaseException as exc:
        if isinstance(exc, ExclusivePublishError):
            raise
        failure = _merge_launch_failure(
            None,
            stage="outer_emergency_closure",
            failure_type=type(exc).__name__,
            message=str(exc),
        )
        control = emergency_state.get(
            "control",
            policy_root.resolve() / "preflight_control",
        )
        wrapper_exit_path = emergency_state.get(
            "wrapper_exit_path", control / "wrapper_exit.json"
        )
        cleanup_path = emergency_state.get(
            "observer_cleanup_path",
            control / "observer_cleanup.json",
        )
        closure_path = emergency_state.get(
            "process_closure_path",
            control / "controller_process_closure.json",
        )
        controller_closure_binding: dict[str, str] | None = None
        controller_return_code: int | None = None
        process = emergency_state.get("controller_process")
        if process is not None:
            controller_return_code, controller_closure = (
                _close_owned_controller_process(
                    process,
                    emergency_state.get("controller_identity"),
                    terminate=True,
                )
            )
            for item in controller_closure["failures"]:
                failure = _merge_launch_failure(
                    failure,
                    stage=f"outer_controller.{item['stage']}",
                    failure_type=str(item["type"]),
                    message=str(item["message"]),
                )
            try:
                _write_exclusive(
                    closure_path, controller_closure
                )
                controller_closure_binding = _json_binding(
                    closure_path,
                    "controller_process_closure_sha256",
                )
            except BaseException as closure_exc:
                _propagate_publish_error(closure_exc)
                failure = _merge_launch_failure(
                    failure,
                    stage="outer_controller_closure_write",
                    failure_type=type(closure_exc).__name__,
                    message=str(closure_exc),
                )
        observer_termination: dict[str, Any] | None = None
        observer = emergency_state.get("observer")
        provisional = emergency_state.get("provisional_observer")
        try:
            if observer is not None:
                observer_termination = _terminate_bound_observer(
                    observer["tmux"],
                    observer["tmux_server"],
                    observer["owner_seal"],
                    observer["process"],
                )
            elif (
                provisional is not None
                and provisional.get("tmux") is not None
                and provisional.get("tmux_server") is not None
                and provisional.get("owner_seal") is not None
            ):
                observer_termination = (
                    _terminate_provisional_tmux_owner(
                        provisional["tmux"],
                        provisional["tmux_server"],
                        provisional["owner_seal"],
                        provisional.get("process"),
                        provisional.get(
                            "process_snapshot_failure"
                        ),
                    )
                )
        except BaseException as observer_exc:
            failure = _merge_launch_failure(
                failure,
                stage="outer_observer_termination",
                failure_type=type(observer_exc).__name__,
                message=str(observer_exc),
            )
            observer_termination = {
                "session": OBSERVER_SESSION,
                "status": "cleanup_failed",
                "session_residual": True,
                "process_residual": True,
                "foreign_session_residual": None,
                "foreign_pane_residual": None,
                "failure": {
                    "type": type(observer_exc).__name__,
                    "message": str(observer_exc),
                },
                "started_at": _utc_now(),
                "completed_at": _utc_now(),
            }
        cleanup_binding: dict[str, str] | None = None
        if observer_termination is not None:
            cleanup = {
                "schema_version": 1,
                "contract_type": (
                    "safa_canonical_preflight_observer_cleanup_v1"
                ),
                "policy_sha256": policy_sha256,
                "wrapper_claim": emergency_state.get(
                    "wrapper_binding"
                ),
                "observer_launch": emergency_state.get(
                    "observer_launch_binding"
                ),
                "reason": "outer_emergency_closure",
                **observer_termination,
            }
            cleanup["observer_cleanup_sha256"] = (
                _canonical_digest(
                    cleanup, "observer_cleanup_sha256"
                )
            )
            try:
                _write_exclusive(cleanup_path, cleanup)
                cleanup_binding = _json_binding(
                    cleanup_path, "observer_cleanup_sha256"
                )
            except BaseException as cleanup_exc:
                _propagate_publish_error(cleanup_exc)
                failure = _merge_launch_failure(
                    failure,
                    stage="outer_observer_cleanup_write",
                    failure_type=type(cleanup_exc).__name__,
                    message=str(cleanup_exc),
                )

        def optional(path: Path, stage: str) -> dict[str, str] | None:
            nonlocal failure
            try:
                return _optional_binding(path)
            except BaseException as binding_exc:
                failure = _merge_launch_failure(
                    failure,
                    stage=stage,
                    failure_type=type(binding_exc).__name__,
                    message=str(binding_exc),
                )
                return None

        claim = emergency_state.get("claim")
        normalized_controller_exit = (
            None
            if controller_return_code is None
            else _normalized_exit(controller_return_code)[0]
        )
        value = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_wrapper_exit_v4"
            ),
            "policy_sha256": policy_sha256,
            "wrapper_claim_sha256": (
                None
                if claim is None
                else claim.get("wrapper_claim_sha256")
            ),
            "command": list(command),
            "started_at": emergency_state.get(
                "started_at", _utc_now()
            ),
            "completed_at": _utc_now(),
            "exit_code": 125,
            "controller_exit_code": normalized_controller_exit,
            "signal": (
                None
                if controller_return_code is None
                else _normalized_exit(controller_return_code)[1]
            ),
            "launch_failure": failure,
            "controller_process_exit": optional(
                control / "controller_process_exit.json",
                "outer_controller_process_exit_binding",
            ),
            "controller_process_closure": (
                controller_closure_binding
            ),
            "controller_process_log": optional(
                control / "controller_process.log",
                "outer_controller_process_log_binding",
            ),
            "observer_launch": emergency_state.get(
                "observer_launch_binding"
            ),
            "controller_process_start": optional(
                control / "controller_process_start.json",
                "outer_controller_process_start_binding",
            ),
            "controller_claim": optional(
                control / "controller_claim.json",
                "outer_controller_claim_binding",
            ),
            "controller_terminal": optional(
                control / "controller_terminal.json",
                "outer_controller_terminal_binding",
            ),
            "observer_terminal": optional(
                control / "observer_terminal.json",
                "outer_observer_terminal_binding",
            ),
            "observer_terminal_snapshot": None,
            "late_observer_terminal": None,
            "late_observer_terminal_snapshot": None,
            "observer_terminal_validation_failure": None,
            "late_observer_terminal_validation_failure": None,
            "observer_stop": optional(
                control / "observer_stop.json",
                "outer_observer_stop_binding",
            ),
            "observer_cleanup": cleanup_binding,
        }
        value["wrapper_exit_sha256"] = _canonical_digest(
            value, "wrapper_exit_sha256"
        )
        return _publish_wrapper_exit_total(
            wrapper_exit_path, value
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--launch-accepted", required=True, type=Path)
    parser.add_argument("--launch-release", required=True, type=Path)
    parser.add_argument("--pane-log", required=True, type=Path)
    return parser.parse_args(argv)


def parse_gate_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-session", required=True)
    parser.add_argument("--owner-nonce", required=True)
    parser.add_argument("--ready-path", required=True, type=Path)
    parser.add_argument("--release-path", required=True, type=Path)
    parser.add_argument("--bootstrap-path", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--wrapper-binding-json", required=True)
    parser.add_argument("--observer-command-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == OBSERVER_GATE_MODE:
        gate_args = parse_gate_args(raw_argv[1:])
        _install_verified_preflight_apis(
            gate_args.config.resolve()
        )
        wrapper_binding = json.loads(
            gate_args.wrapper_binding_json
        )
        observer_command = json.loads(
            gate_args.observer_command_json
        )
        if (
            not isinstance(wrapper_binding, dict)
            or not isinstance(observer_command, list)
            or not observer_command
            or any(not isinstance(item, str) for item in observer_command)
        ):
            raise RuntimeError(
                "observer bootstrap gate arguments are invalid"
            )
        return _observer_gate_ready(
            expected_session=gate_args.expected_session,
            owner_nonce=gate_args.owner_nonce,
            ready_path=gate_args.ready_path.resolve(),
            release_path=gate_args.release_path.resolve(),
            bootstrap_path=gate_args.bootstrap_path.resolve(),
            policy_sha256=gate_args.policy_sha256,
            wrapper_binding=wrapper_binding,
            observer_command=observer_command,
        )
    args = parse_args(raw_argv)
    if (
        os.environ.get(LAUNCH_RECEIPT_PATH_ENV)
        != str(args.launch_receipt.resolve())
        or os.environ.get(LAUNCH_ACCEPTED_PATH_ENV)
        != str(args.launch_accepted.resolve())
        or os.environ.get(LAUNCH_RELEASE_PATH_ENV)
        != str(args.launch_release.resolve())
        or os.environ.get(PANE_LOG_PATH_ENV)
        != str(args.pane_log.resolve())
        or len(args.attempt_id) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.attempt_id
        )
    ):
        raise RuntimeError(
            "formal preflight launcher CLI/environment differs"
        )
    cli_receipt_snapshot = _secure_json_snapshot(
        args.launch_receipt,
        digest_field="launch_receipt_sha256",
    )
    assert cli_receipt_snapshot is not None
    receipt = dict(cli_receipt_snapshot["value"])
    if receipt.get("attempt_id") != args.attempt_id:
        raise RuntimeError("formal preflight attempt ID differs")
    _fault_channel_context = _bind_inherited_fault_channel(
        receipt
    )
    repo_root = args.repo_root.resolve()
    config = args.config.resolve()
    campaign_root = args.campaign_root.resolve()
    policy_root = campaign_root / "by_policy" / args.policy_sha256
    command = [
        args.python,
        "-u",
        str(repo_root / "scripts/run_canonical_checkpoint_screening.py"),
        "--config",
        str(config),
        "--campaign-root",
        str(campaign_root),
        "--phase",
        "preflight",
        "--execute",
    ]
    observer_command = [
        args.python,
        "-u",
        str(repo_root / "scripts/run_canonical_checkpoint_screening.py"),
        "--config",
        str(config),
        "--campaign-root",
        str(campaign_root),
        "--phase",
        "monitor",
        "--monitor-target",
        "preflight",
        "--execute",
    ]
    def run_operation() -> dict[str, Any]:
        return run_wrapped_controller(
            repo_root=repo_root,
            policy_root=policy_root,
            policy_sha256=args.policy_sha256,
            config=config,
            command=command,
            observer_command=observer_command,
        )

    (
        value,
        dedicated_failure_code,
        dedicated_failure,
    ) = _execute_with_fault_reporting(
        _fault_channel_context, run_operation
    )
    if dedicated_failure_code is not None:
        print(
            json.dumps(
                dedicated_failure,
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return dedicated_failure_code
    assert value is not None
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return int(value["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
