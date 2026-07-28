#!/usr/bin/env python3
"""Evidence-complete launcher for the canonical CPU preflight wrapper."""

from __future__ import annotations

import argparse
import ctypes
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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from safa.closeout.preflight_launch_contract import (
        LAUNCH_TERMINAL_CONTRACT_TYPE,
        build_file_identity,
        build_gate_ready,
        build_launch_accepted,
        build_ownership_release,
        build_ownership_terminal,
        build_pane_owner_seal,
        build_process_identity,
        build_tmux_server_identity,
        build_tmux_started,
        build_verified_implementations,
        build_wrapper_started,
        validate_claim_v3,
        validate_file_identity,
        validate_gate_ready,
        validate_ownership_chain,
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
PANE_GATE_MODE = "__pane_gate__"
ARCHIVE_FAILURE_MODE = "archive-untracked-failure"
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
    "LAUNCH_TERMINAL_CONTRACT_TYPE",
    "PreflightLaunchContractError",
    "build_artifact_binding",
    "build_file_identity",
    "build_gate_ready",
    "build_launch_accepted",
    "build_ownership_release",
    "build_ownership_terminal",
    "build_pane_owner_seal",
    "build_process_identity",
    "build_tmux_server_identity",
    "build_tmux_started",
    "build_verified_implementations",
    "build_wrapper_started",
    "validate_claim_v3",
    "validate_file_identity",
    "validate_gate_ready",
    "validate_ownership_chain",
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
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


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
    gate_pid: int,
) -> dict[str, Any]:
    value = _load_json(path, "preflight wrapper claim")
    validate_file_identity(
        receipt_identity, "launcher receipt identity"
    )
    validate_claim_v3(
        value,
        verified_implementations=verified_implementations,
        wrapper_started=wrapper_started,
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
        or value.get("controller_tmux", {}).get("pane_pid") != gate_pid
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


def _validate_gate_execution_terminal(
    path: Path,
    *,
    receipt_binding: Mapping[str, str],
    receipt_identity: Mapping[str, Any],
    gate_ready_binding: Mapping[str, str],
    wrapper_arguments: Sequence[str],
) -> dict[str, Any]:
    value = _load_json(path, "gate execution terminal")
    returncode = value.get("returncode")
    exit_kind = value.get("exit_kind")
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
        value.get("schema_version") != 1
        or value.get("contract_type")
        != "safa_canonical_preflight_gate_execution_terminal_v1"
        or value.get("launch_receipt") != dict(receipt_binding)
        or value.get("launch_receipt_identity")
        != dict(receipt_identity)
        or value.get("pane_gate_ready") != dict(gate_ready_binding)
        or value.get("wrapper_arguments") != list(wrapper_arguments)
        or not isinstance(value.get("publication_failures"), list)
        or (
            value.get("launch_ownership_release") is not None
            and (
                value.get("launch_accepted") is None
                or value.get("launch_terminal") is None
            )
        )
        or not classified
        or value.get("gate_execution_terminal_sha256")
        != _canonical_digest(
            value, "gate_execution_terminal_sha256"
        )
    ):
        raise RuntimeError("gate execution terminal differs")
    return value


def _publish_gate_execution_terminal(
    path: Path, value: dict[str, Any]
) -> dict[str, Any]:
    value["gate_execution_terminal_sha256"] = _canonical_digest(
        value, "gate_execution_terminal_sha256"
    )
    try:
        _write_exclusive(path, value)
    except BaseException as exc:
        if path.is_file():
            existing = _load_json(path, "existing gate execution terminal")
            if existing.get("gate_execution_terminal_sha256") == (
                _canonical_digest(
                    existing, "gate_execution_terminal_sha256"
                )
            ):
                return existing
        value["publication_failures"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        value["gate_execution_terminal_sha256"] = _canonical_digest(
            value, "gate_execution_terminal_sha256"
        )
        _write_exclusive(path, value)
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
    except BaseException as exc:
        if terminal_path.is_file():
            try:
                existing = _load_json(
                    terminal_path, "existing launch terminal"
                )
                if (
                    existing.get("launch_terminal_sha256")
                    == _canonical_digest(
                        existing, "launch_terminal_sha256"
                    )
                ):
                    return existing
            except BaseException:
                pass
        value["failure"]["secondary_failures"].append(
            {
                "stage": "launch_terminal_write",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        value["completed_at"] = _utc_now()
        value["launch_terminal_sha256"] = _canonical_digest(
            value, "launch_terminal_sha256"
        )
        target = (
            terminal_path
            if not terminal_path.exists()
            else terminal_path.with_name(
                "launch_terminal_emergency.json"
            )
        )
        _write_exclusive(target, value)
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
        "wrapper_arguments": list(wrapper_arguments),
        "completed_at": _utc_now(),
    }
    _publish_gate_execution_terminal(execution_terminal_path, terminal)
    return returncode if returncode >= 0 else 128 - returncode


def _pane_gate(
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
            print("pane gate release timed out", flush=True)
            return 124
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
        or release.get("pane_gate_release_sha256")
        != _canonical_digest(release, "pane_gate_release_sha256")
    ):
        print("pane gate release contract differs", flush=True)
        return 125
    environment = dict(os.environ)
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
        )
    except BaseException as exc:
        traceback.print_exc()
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
            "wrapper_arguments": list(wrapper_arguments),
            "completed_at": _utc_now(),
        }
        _publish_gate_execution_terminal(
            execution_terminal_path, terminal
        )
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        return 126
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
        )
    except BaseException:
        _terminate_spawned_child(child)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


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
    observer_session = OBSERVER_SESSION_PREFIX + observer_suffix
    launcher_path = Path(__file__).resolve()
    wrapper_path = (
        repo_root / "scripts/run_canonical_preflight_wrapper.py"
    ).resolve()
    controller_path = (
        repo_root / "scripts/run_canonical_checkpoint_screening.py"
    ).resolve()
    started_registry_path = (
        campaign_root
        / "preflight_launch_attempts"
        / "started"
        / f"{attempt_id}.json"
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
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts"
        / "by_policy"
        / policy_sha256
        / attempt_id
    )
    try:
        _write_exclusive(started_registry_path, started_registry)
    except BaseException as exc:
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
        attempt_root.parent.mkdir(parents=True, exist_ok=True)
        attempt_root.mkdir(exist_ok=False)
        _fsync_directory(attempt_root.parent)
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
        *gate_arguments,
    ]
    verified_implementations = _reverify_verified_preflight_apis()
    receipt = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_launch_receipt_v1",
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
        "pane_gate_arguments": gate_arguments,
        "tmux_arguments": tmux_arguments,
        "shell": False,
        "pane_log": pane_log_identity,
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
    try:
        _write_exclusive(receipt_path, receipt)
    except BaseException as exc:
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
    client: dict[str, Any] | None = None
    pane: dict[str, Any] | None = None
    owner_seal: dict[str, Any] | None = None
    try:
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
        while not gate_ready_path.is_file():
            pane = _tmux_pane(CONTROLLER_SESSION)
            if pane is None or pane["pane_dead"]:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        pane = _tmux_pane(CONTROLLER_SESSION)
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
        if pane["pane_dead"] or not gate_ready_path.is_file():
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
                message="pane gate did not publish ready evidence",
                client=client,
                pane=pane,
                tmux_started_path=None,
                log_path=log_path,
                session_residual=False,
                started_at=started_at,
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
            or gate_ready.get("process", {}).get("pid")
            != pane["pane_pid"]
            or gate_ready.get("process")
            != _process_identity(int(pane["pane_pid"]))
            or _process_command_bytes(int(pane["pane_pid"]))
            != _command_bytes(gate_arguments)
            or _process_executable(int(pane["pane_pid"]))["path"]
            != python_binding["path"]
            or gate_ready.get("pane_gate_ready_sha256")
            != _canonical_digest(
                gate_ready, "pane_gate_ready_sha256"
            )
        ):
            raise RuntimeError("pane gate ready contract differs")
        if (
            owner_seal["pane"] != pane["pane"]
            or owner_seal["pane_pid"] != pane["pane_pid"]
            or owner_seal["pane_process"] != gate_ready["process"]
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
            pane_process=gate_ready["process"],
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
            "released_at": _utc_now(),
        }
        release["pane_gate_release_sha256"] = _canonical_digest(
            release, "pane_gate_release_sha256"
        )
        _write_exclusive(release_path, release)
        while True:
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
                    != gate_ready["process"]
                    or _process_command_bytes(int(pane["pane_pid"]))
                    != _command_bytes(gate_arguments)
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
                    gate_pid=int(pane["pane_pid"]),
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
                )
                _set_remain_on_exit(str(pane["pane"]), False)
                _verify_remain_on_exit(str(pane["pane"]), "off")
                ownership_release = _publish_ownership_release(
                    ownership_release_path,
                    receipt_path=receipt_path,
                    receipt_identity=receipt_identity,
                    accepted_path=accepted_path,
                    terminal_path=terminal_path,
                    claim_path=claim_path,
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
                    label="published preflight launch ownership chain",
                )
                return ownership_release
            if gate_execution_terminal_path.is_file():
                execution = _validate_gate_execution_terminal(
                    gate_execution_terminal_path,
                    receipt_binding=_json_binding(
                        receipt_path, "launch_receipt_sha256"
                    ),
                    receipt_identity=receipt_identity,
                    gate_ready_binding=_json_binding(
                        gate_ready_path, "pane_gate_ready_sha256"
                    ),
                    wrapper_arguments=wrapper_arguments,
                )
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
                return _publish_terminal(
                    terminal_path,
                    receipt_path=receipt_path,
                    receipt_identity=receipt_identity,
                    status="wrapper_claim_timeout",
                    failure_type="WrapperClaimTimeout",
                    message="wrapper claim was not published in time",
                    client=client,
                    pane=pane,
                    tmux_started_path=tmux_started_path,
                    log_path=log_path,
                    session_residual=False,
                    started_at=started_at,
                )
            time.sleep(0.02)
        dead_pane = pane
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
            status="wrapper_exited_before_claim",
            failure_type="WrapperEarlyExit",
            message=(
                "wrapper process exited before publishing a durable claim"
            ),
            client=client,
            pane=dead_pane,
            tmux_started_path=tmux_started_path,
            log_path=log_path,
            session_residual=False,
            started_at=started_at,
        )
    except BaseException as exc:
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
    started_registry_path = (
        campaign_root.resolve()
        / "preflight_launch_attempts"
        / "started"
        / f"{attempt_id}.json"
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
    attempt_root = (
        campaign_root.resolve()
        / "preflight_launch_attempts"
        / "by_policy"
        / policy_sha256
        / attempt_id
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
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


def _parse_archive_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--controller-owner-nonce", required=True)
    parser.add_argument("--observer-session", required=True)
    parser.add_argument("--occurred-at", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
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
