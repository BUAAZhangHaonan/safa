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
from typing import Any, Mapping, Sequence


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
OBSERVER_BOOTSTRAP_PATH_ENV = "SAFA_PREFLIGHT_OBSERVER_BOOTSTRAP_PATH"
OBSERVER_BOOTSTRAP_POLICY_ENV = "SAFA_PREFLIGHT_OBSERVER_POLICY_SHA256"
OBSERVER_BOOTSTRAP_WRAPPER_ENV = "SAFA_PREFLIGHT_WRAPPER_CLAIM"
OBSERVER_BOOTSTRAP_NONCE_ENV = "SAFA_PREFLIGHT_OBSERVER_OWNER_NONCE"
OBSERVER_GATE_MODE = "--observer-bootstrap-gate"


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


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    content = _canonical_json(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    complete = False
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            path.unlink(missing_ok=True)


def _json_binding(path: Path, digest_field: str) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"required wrapper artifact is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = value.get(digest_field)
    if (
        not isinstance(canonical, str)
        or len(canonical) != 64
        or _canonical_digest(value, digest_field) != canonical
    ):
        raise RuntimeError(f"wrapper artifact canonical digest differs: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "canonical_sha256": canonical,
    }


def _optional_binding(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


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
        stat_pgid = int(fields[2])
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"process identity stat is malformed for PID {pid}"
        ) from exc
    if (
        stat_pid != pid
        or len(state) != 1
        or stat_pgid <= 0
        or start_ticks <= 0
    ):
        raise RuntimeError(
            f"process identity stat is malformed for PID {pid}"
        )
    return (
        {
            "pid": stat_pid,
            "pgid": stat_pgid,
            "start_ticks": start_ticks,
        },
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
    identity = {
        "server_pid": int(rows[0][0]),
        "socket_path": rows[0][1],
    }
    if (
        set(identity) != {"server_pid", "socket_path"}
        or type(identity["server_pid"]) is not int
        or identity["server_pid"] <= 1
        or not isinstance(identity["socket_path"], str)
        or not Path(identity["socket_path"]).is_absolute()
    ):
        raise RuntimeError("tmux server identity is invalid")
    return identity


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
    seal = {
        "server_pid": int(tmux_server["server_pid"]),
        "server_start_ticks": int(server_process["start_ticks"]),
        **socket_identity,
        "session": str(tmux_identity["session"]),
        "pane": str(tmux_identity["pane"]),
        "pane_pid": int(tmux_identity["pane_pid"]),
        "owner_nonce": _tmux_owner_nonce(
            str(tmux_identity["session"]),
            str(tmux_server["socket_path"]),
        ),
    }
    if seal["owner_nonce"] != expected_owner_nonce:
        raise RuntimeError("tmux owner nonce differs after launch")
    _validate_tmux_owner_seal(seal, tmux_identity, tmux_server)
    return seal


def _validate_tmux_owner_seal(
    owner_seal: Mapping[str, Any],
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> None:
    expected_keys = {
        "server_pid",
        "server_start_ticks",
        "socket_path",
        "socket_device",
        "socket_inode",
        "session",
        "pane",
        "pane_pid",
        "owner_nonce",
    }
    if (
        set(owner_seal) != expected_keys
        or owner_seal.get("server_pid") != tmux_server.get("server_pid")
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


def _read_strict_json_contract(
    path: Path,
    digest_field: str,
) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"contract is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(content) != after.st_size
    ):
        raise RuntimeError(f"contract changed during atomic read: {path}")
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"contract is not a mapping: {path}")
    canonical = value.get(digest_field)
    if (
        not isinstance(canonical, str)
        or len(canonical) != 64
        or _canonical_digest(value, digest_field) != canonical
    ):
        raise RuntimeError(f"contract canonical digest differs: {path}")
    return value


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
    while not release_path.is_file():
        time.sleep(0.02)
    release = _read_strict_json_contract(
        release_path, "observer_gate_release_sha256"
    )
    ready_binding = {
        "path": str(ready_path.resolve()),
        "sha256": _sha256_file(ready_path),
        "canonical_sha256": ready["observer_gate_ready_sha256"],
    }
    if (
        set(release)
        != {
            "schema_version",
            "contract_type",
            "policy_sha256",
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
        if best_owner_seal is None:
            best_tmux = candidate_tmux
            best_tmux_server = candidate_server
            best_owner_seal = candidate_seal
        elif (
            candidate_tmux != best_tmux
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
                        if ready_path.is_file():
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
    if set(tmux_server) != {"server_pid", "socket_path"}:
        raise RuntimeError("sealed tmux server identity is invalid")
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
    while not bootstrap_path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"tmux observer bootstrap timed out for session {session}"
            )
        time.sleep(0.05)
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "contract_type",
        "policy_sha256",
        "wrapper_claim",
        "observer_session",
        "owner_nonce",
        "process",
        "executable",
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
        or bootstrap.get("wrapper_claim") != dict(wrapper_binding)
        or bootstrap.get("observer_session") != session
        or bootstrap.get("owner_nonce") != expected_owner_nonce
        or bootstrap.get("command") != list(expected_command)
        or bootstrap.get("observer_bootstrap_sha256")
        != _canonical_digest(bootstrap, "observer_bootstrap_sha256")
    ):
        raise RuntimeError("tmux observer bootstrap contract mismatch")
    tmux_identity = bootstrap["tmux"]
    process_identity = bootstrap["process"]
    _validate_tmux_identity(tmux_identity, session)
    if (
        tmux_identity["pane_pid"] != process_identity.get("pid")
        or process_identity != _process_identity(process_identity["pid"])
        or bootstrap["executable"]
        != os.readlink(f"/proc/{process_identity['pid']}/exe")
        or bootstrap["command"]
        != _process_command(process_identity["pid"])
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
    if set(sealed_server) != {"server_pid", "socket_path"}:
        raise RuntimeError("sealed tmux server identity is invalid")
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
    value = json.loads(path.read_text(encoding="utf-8"))
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
        != json.loads(
            Path(process_start_binding["path"]).read_text(encoding="utf-8")
        ).get("process")
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
    launch = json.loads(
        Path(observer_launch_binding["path"]).read_text(encoding="utf-8")
    )
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
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {"path", "sha256", "canonical_sha256"}
            or not all(
                isinstance(binding.get(field), str)
                for field in ("path", "sha256", "canonical_sha256")
            )
        ):
            raise RuntimeError(
                f"CPU preflight observer terminal {label} binding differs"
            )
        bound_path = Path(str(binding["path"]))
        if (
            expected_path is not None
            and bound_path.resolve() != expected_path.resolve()
        ):
            raise RuntimeError(
                f"CPU preflight observer terminal {label} path differs"
            )
        bound_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            bound_flags |= os.O_NOFOLLOW
        try:
            bound_descriptor = os.open(bound_path, bound_flags)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"CPU preflight observer terminal {label} file is absent"
            ) from exc
        except PermissionError as exc:
            raise RuntimeError(
                f"CPU preflight observer terminal {label} "
                "file permission denied"
            ) from exc
        try:
            bound_before = os.fstat(bound_descriptor)
            if not stat.S_ISREG(bound_before.st_mode):
                raise RuntimeError(
                    f"CPU preflight observer terminal {label} "
                    "is not a regular file"
                )
            bound_chunks: list[bytes] = []
            while True:
                bound_chunk = os.read(bound_descriptor, 1024 * 1024)
                if not bound_chunk:
                    break
                bound_chunks.append(bound_chunk)
            bound_after = os.fstat(bound_descriptor)
        finally:
            os.close(bound_descriptor)
        bound_content = b"".join(bound_chunks)
        if (
            bound_before.st_dev != bound_after.st_dev
            or bound_before.st_ino != bound_after.st_ino
            or bound_before.st_size != bound_after.st_size
            or bound_before.st_mtime_ns != bound_after.st_mtime_ns
            or len(bound_content) != bound_after.st_size
        ):
            raise RuntimeError(
                f"CPU preflight observer terminal {label} "
                "changed during atomic read"
            )
        if hashlib.sha256(bound_content).hexdigest() != binding["sha256"]:
            raise RuntimeError(
                f"CPU preflight observer terminal {label} file SHA differs"
            )
        bound_value = json.loads(bound_content.decode("utf-8"))
        canonical = bound_value.get(digest_field)
        if (
            canonical != binding["canonical_sha256"]
            or _canonical_digest(bound_value, digest_field) != canonical
        ):
            raise RuntimeError(
                f"CPU preflight observer terminal {label} "
                "canonical SHA differs"
            )
        return bound_value

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(
                "CPU preflight observer terminal is not a regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(content) != after.st_size
    ):
        raise RuntimeError(
            "CPU preflight observer terminal changed during atomic read"
        )
    value = json.loads(content.decode("utf-8"))
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
    return value, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "canonical_sha256": canonical,
    }


def _wait_observer_terminal(
    path: Path,
    process_exit_path: Path,
    *,
    policy_sha256: str,
    observer_launch_binding: Mapping[str, str],
    observer_process: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, str]] | None:
    deadline = time.monotonic() + OBSERVER_TERMINAL_WAIT_SECONDS
    while not path.is_file():
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
    if observer_stop_binding is None and observer_stop_path.is_file():
        stop = _validate_observer_stop(
            observer_stop_path,
            policy_sha256=policy_sha256,
            wrapper_binding=wrapper_binding,
            observer_launch_binding=observer_launch_binding,
            observer_process=observer_process,
            process_start_binding=process_start_binding,
            require_live_identity=False,
        )
        observer_stop_binding = {
            "path": str(observer_stop_path.resolve()),
            "sha256": _sha256_file(observer_stop_path),
            "canonical_sha256": stop["observer_stop_sha256"],
        }
    if observer_stop_binding is not None:
        if terminal_value.get("observer_stop") != observer_stop_binding:
            raise RuntimeError(
                "CPU preflight observer terminal stop binding differs"
            )
    elif terminal_value.get("status") == "failed":
        raise RuntimeError("failed CPU preflight observer terminal has no stop")
    return observer_stop_binding


def run_wrapped_controller(
    *,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    command: Sequence[str],
    observer_command: Sequence[str],
) -> dict[str, Any]:
    control = policy_root.resolve() / "preflight_control"
    wrapper_claim_path = control / "wrapper_claim.json"
    process_log_path = control / "controller_process.log"
    process_exit_path = control / "controller_process_exit.json"
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
    controller_session = _tmux_session()
    controller_tmux = _tmux_identity(CONTROLLER_SESSION)
    controller_tmux_server = _tmux_server_identity(
        controller_tmux["pane"]
    )
    wrapper_process = _require_process_identity(
        os.getpid(), "CPU preflight wrapper"
    )
    _assert_tmux_process_identity(
        CONTROLLER_SESSION,
        controller_tmux,
        controller_tmux_server,
        wrapper_process,
    )
    claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_claim_v2",
        "policy_sha256": policy_sha256,
        "config": {
            "path": str(config.resolve()),
            "sha256": _sha256_file(config),
        },
        "checkpoint_plan": _json_binding(plan_path, "checkpoint_plan_sha256"),
        "preflight_request_manifest": _json_binding(
            request_manifest_path, "preflight_request_manifest_sha256"
        ),
        "controller_session": controller_session,
        "controller_tmux": controller_tmux,
        "controller_tmux_server": controller_tmux_server,
        "observer_session": OBSERVER_SESSION,
        "command": list(command),
        "observer_command": list(observer_command),
        "wrapper_pid": os.getpid(),
        "wrapper_process": wrapper_process,
        "started_at": started_at,
        "external_timeout_seconds": None,
    }
    claim["wrapper_claim_sha256"] = _canonical_digest(
        claim, "wrapper_claim_sha256"
    )
    _write_exclusive(wrapper_claim_path, claim)
    wrapper_binding = {
        "path": str(wrapper_claim_path.resolve()),
        "sha256": _sha256_file(wrapper_claim_path),
        "canonical_sha256": claim["wrapper_claim_sha256"],
    }
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
        observer_gate_ready_binding = {
            "path": str(observer_gate_ready_path.resolve()),
            "sha256": _sha256_file(observer_gate_ready_path),
            "canonical_sha256": observer_gate_ready[
                "observer_gate_ready_sha256"
            ],
        }
        observer_gate_release = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_observer_gate_release_v1"
            ),
            "policy_sha256": policy_sha256,
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
        if (
            observer_tmux != provisional_tmux
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
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        "observer_session": OBSERVER_SESSION,
        "command": list(observer_command),
        "observer_bootstrap": (
            None
            if observer_bootstrap is None
            else {
                "path": str(observer_bootstrap_path.resolve()),
                "sha256": _sha256_file(observer_bootstrap_path),
                "canonical_sha256": observer_bootstrap[
                    "observer_bootstrap_sha256"
                ],
            }
        ),
        "tmux": observer_tmux,
        "tmux_server": observer_tmux_server,
        "tmux_owner_seal": observer_tmux_owner_seal,
        "process": observer_process,
        "observer_gate_ready": (
            None
            if observer_gate_ready is None
            else {
                "path": str(observer_gate_ready_path.resolve()),
                "sha256": _sha256_file(observer_gate_ready_path),
                "canonical_sha256": observer_gate_ready[
                    "observer_gate_ready_sha256"
                ],
            }
        ),
        "observer_gate_release": (
            None
            if observer_gate_release is None
            else {
                "path": str(observer_gate_release_path.resolve()),
                "sha256": _sha256_file(observer_gate_release_path),
                "canonical_sha256": observer_gate_release[
                    "observer_gate_release_sha256"
                ],
            }
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
    _write_exclusive(observer_launch_path, observer_launch)
    observer_launch_binding = {
        "path": str(observer_launch_path.resolve()),
        "sha256": _sha256_file(observer_launch_path),
        "canonical_sha256": observer_launch[
            "observer_launch_sha256"
        ],
    }
    if observer_launch_failure is not None:
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
        _write_exclusive(observer_cleanup_path, launch_cleanup)
        observer_cleanup_binding = {
            "path": str(observer_cleanup_path.resolve()),
            "sha256": _sha256_file(observer_cleanup_path),
            "canonical_sha256": launch_cleanup[
                "observer_cleanup_sha256"
            ],
        }
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
            "reason": dict(observer_launch_failure),
            "started_at": None,
            "completed_at": _utc_now(),
        }
        not_started["controller_process_start_sha256"] = (
            _canonical_digest(
                not_started, "controller_process_start_sha256"
            )
        )
        _write_exclusive(process_start_path, not_started)
        not_started_binding = {
            "path": str(process_start_path.resolve()),
            "sha256": _sha256_file(process_start_path),
            "canonical_sha256": not_started[
                "controller_process_start_sha256"
            ],
        }
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
            "launch_failure": dict(observer_launch_failure),
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
        _write_exclusive(process_exit_path, not_started_exit)
        process_exit_binding = {
            "path": str(process_exit_path.resolve()),
            "sha256": _sha256_file(process_exit_path),
            "canonical_sha256": not_started_exit[
                "controller_process_exit_sha256"
            ],
        }
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
            "launch_failure": dict(observer_launch_failure),
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
        _write_exclusive(wrapper_exit_path, failed_wrapper_exit)
        return failed_wrapper_exit
    process_log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    return_code: int | None = None
    controller_pid: int | None = None
    controller_process_start: dict[str, Any] | None = None
    controller_process_start_binding: dict[str, str] | None = None
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
    launch_failure: dict[str, str] | None = None
    try:
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
        controller_pid = process.pid
        controller_process_start = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_controller_process_start_v1",
            "policy_sha256": policy_sha256,
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
        controller_process_start[
            "controller_process_start_sha256"
        ] = _canonical_digest(
            controller_process_start,
            "controller_process_start_sha256",
        )
        _write_exclusive(process_start_path, controller_process_start)
        controller_process_start_binding = {
            "path": str(process_start_path.resolve()),
            "sha256": _sha256_file(process_start_path),
            "canonical_sha256": controller_process_start[
                "controller_process_start_sha256"
            ],
        }
        while process.poll() is None:
            try:
                _assert_tmux_process_identity(
                    OBSERVER_SESSION,
                    observer_tmux,
                    observer_tmux_server,
                    observer_process,
                )
                if observer_stop_path.is_file():
                    stop = _validate_observer_stop(
                        observer_stop_path,
                        policy_sha256=policy_sha256,
                        wrapper_binding=wrapper_binding,
                        observer_launch_binding=observer_launch_binding,
                        observer_process=observer_process,
                        process_start_binding=controller_process_start_binding,
                    )
                    observer_stop_binding = {
                        "path": str(observer_stop_path.resolve()),
                        "sha256": _sha256_file(observer_stop_path),
                        "canonical_sha256": stop["observer_stop_sha256"],
                    }
                    launch_failure = {
                        "type": "ObserverHardStop",
                        "message": str(stop["failure"]),
                    }
                    _terminate_owned_process(
                        process, controller_process_start["process"]
                    )
                    break
            except BaseException as exc:
                launch_failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                _terminate_owned_process(
                    process, controller_process_start["process"]
                )
                break
            time.sleep(0.1)
        return_code = process.wait()
        if observer_stop_path.is_file() and observer_stop_binding is None:
            stop = _validate_observer_stop(
                observer_stop_path,
                policy_sha256=policy_sha256,
                wrapper_binding=wrapper_binding,
                observer_launch_binding=observer_launch_binding,
                observer_process=observer_process,
                process_start_binding=controller_process_start_binding,
            )
            observer_stop_binding = {
                "path": str(observer_stop_path.resolve()),
                "sha256": _sha256_file(observer_stop_path),
                "canonical_sha256": stop["observer_stop_sha256"],
            }
    except BaseException as exc:
        launch_failure = {"type": type(exc).__name__, "message": str(exc)}
        return_code = 125
    finally:
        if descriptor is not None:
            os.fsync(descriptor)
            os.close(descriptor)
    exit_code, signal_number = _normalized_exit(return_code)
    process_exit = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_controller_process_exit_v2",
        "policy_sha256": policy_sha256,
        "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        "observer_launch": observer_launch_binding,
        "controller_process_start": controller_process_start_binding,
        "observer_stop": observer_stop_binding,
        "controller_pid": controller_pid,
        "command": list(command),
        "exit_code": exit_code,
        "signal": signal_number,
        "launch_failure": launch_failure,
        "controller_process_log": _optional_binding(process_log_path),
        "controller_claim": _optional_binding(control / "controller_claim.json"),
        "controller_terminal": _optional_binding(control / "controller_terminal.json"),
        "completed_at": _utc_now(),
    }
    process_exit["controller_process_exit_sha256"] = _canonical_digest(
        process_exit, "controller_process_exit_sha256"
    )
    _write_exclusive(process_exit_path, process_exit)
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
        if launch_failure is None:
            launch_failure = {
                "type": "ObserverTerminalValidationError",
                "message": str(exc),
            }
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
        _write_exclusive(observer_cleanup_path, cleanup)
        observer_cleanup_residual = bool(
            cleanup["session_residual"] or cleanup["process_residual"]
        )
        observer_cleanup_binding = {
            "path": str(observer_cleanup_path.resolve()),
            "sha256": _sha256_file(observer_cleanup_path),
            "canonical_sha256": cleanup["observer_cleanup_sha256"],
        }
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
        _write_exclusive(observer_cleanup_path, cleanup)
        observer_cleanup_residual = bool(
            cleanup["session_residual"] or cleanup["process_residual"]
        )
        observer_cleanup_failed = (
            cleanup["status"] != "closed_terminal_observer"
        )
        observer_cleanup_binding = {
            "path": str(observer_cleanup_path.resolve()),
            "sha256": _sha256_file(observer_cleanup_path),
            "canonical_sha256": cleanup["observer_cleanup_sha256"],
        }
    observer_status = (
        None
        if observer_terminal_value is None
        else observer_terminal_value.get("status")
    )
    effective_exit_code = exit_code
    if (
        observer_terminal is None
        or observer_status != "completed"
        or observer_terminal_validation_failure is not None
        or late_observer_terminal is not None
        or observer_cleanup_residual
        or observer_cleanup_failed
    ):
        if effective_exit_code == 0:
            effective_exit_code = 124
        if launch_failure is None:
            launch_failure = {
                "type": "RuntimeError",
                "message": (
                    "CPU preflight observer crossed the terminal boundary "
                    "or did not complete without residuals"
                ),
            }
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
        "controller_process_exit": {
            "path": str(process_exit_path.resolve()),
            "sha256": _sha256_file(process_exit_path),
            "canonical_sha256": process_exit[
                "controller_process_exit_sha256"
            ],
        },
        "controller_process_log": _optional_binding(process_log_path),
        "observer_launch": observer_launch_binding,
        "controller_process_start": process_exit[
            "controller_process_start"
        ],
        "controller_claim": _optional_binding(control / "controller_claim.json"),
        "controller_terminal": _optional_binding(control / "controller_terminal.json"),
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
    _write_exclusive(wrapper_exit_path, value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--python", required=True)
    return parser.parse_args(argv)


def parse_gate_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-session", required=True)
    parser.add_argument("--owner-nonce", required=True)
    parser.add_argument("--ready-path", required=True, type=Path)
    parser.add_argument("--release-path", required=True, type=Path)
    parser.add_argument("--bootstrap-path", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--wrapper-binding-json", required=True)
    parser.add_argument("--observer-command-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == OBSERVER_GATE_MODE:
        gate_args = parse_gate_args(raw_argv[1:])
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
    value = run_wrapped_controller(
        repo_root=repo_root,
        policy_root=policy_root,
        policy_sha256=args.policy_sha256,
        config=config,
        command=command,
        observer_command=observer_command,
    )
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return int(value["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
