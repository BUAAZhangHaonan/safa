#!/usr/bin/env python3
"""Subprocess fixtures for the real tmux preflight-wrapper lifecycle tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_json({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, canonical_json(value))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path: Path, field: str) -> dict[str, str]:
    value = load(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha(path),
        "canonical_sha256": value[field],
    }


def file_identity(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError("fixture identity target is not regular")
    return {
        "path": str(path.resolve()),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
    }


def verified_implementations(config: Path) -> dict[str, Any]:
    policy = load(config)
    root = config.resolve().parents[2]
    result: dict[str, Any] = {}
    for output_name, implementation_name in (
        ("verified_loader", "preflight_verified_loader"),
        (
            "preflight_launch_contract",
            "preflight_launch_contract",
        ),
    ):
        raw = policy["implementations"][implementation_name]
        path = (root / raw["path"]).resolve(strict=True)
        observed_sha256 = file_sha(path)
        if observed_sha256 != raw["sha256"]:
            raise RuntimeError(
                f"fixture verified implementation differs: "
                f"{implementation_name}"
            )
        result[output_name] = {
            "path": str(path),
            "sha256": observed_sha256,
            "file_identity": file_identity(path),
        }
    return result


def process_identity(pid: int) -> dict[str, int]:
    raw_stat = Path(f"/proc/{pid}/stat").read_text()
    closing = raw_stat.rfind(")")
    if closing < 0:
        raise RuntimeError("fixture process stat is malformed")
    fields = raw_stat[closing + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def launch_process_identity(pid: int) -> dict[str, int]:
    raw_stat = Path(f"/proc/{pid}/stat").read_text()
    closing = raw_stat.rfind(")")
    if closing < 0:
        raise RuntimeError("fixture launch process stat is malformed")
    fields = raw_stat[closing + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def tmux_identity(session: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            session,
            "-F",
            "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    row = result.stdout.strip().split("\t")
    return {
        "session": row[0],
        "pane": row[1],
        "pane_pid": int(row[2]),
        "pane_current_command": row[3],
    }


def wait_file(path: Path, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"fixture timed out waiting for {path}")
        time.sleep(0.02)


def current_observer_session() -> str:
    return os.environ.get(
        "SAFA_PREFLIGHT_OBSERVER_SESSION",
        "safa-screening-preflight-monitor",
    )


def install_synthetic_launcher_contract(
    *,
    module: Any,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
) -> threading.Thread:
    campaign_root = policy_root.parents[1]
    attempt_id = hashlib.sha256(
        f"{policy_root.resolve()}:{os.getpid()}".encode()
    ).hexdigest()
    owner_nonce = hashlib.sha256(
        f"owner:{attempt_id}".encode()
    ).hexdigest()
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts"
        / "by_policy"
        / policy_sha256
        / attempt_id
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    started_path = (
        campaign_root
        / "preflight_launch_attempts"
        / "started"
        / f"{attempt_id}.json"
    )
    started = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_started_registry_v1"
        ),
        "attempt_id": attempt_id,
        "policy_sha256": policy_sha256,
        "reserved_at": utc_now(),
    }
    started["launch_started_registry_sha256"] = digest(
        started, "launch_started_registry_sha256"
    )
    write_exclusive(started_path, started)
    log_path = attempt_root / "pane.log"
    descriptor = os.open(
        log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    os.fsync(descriptor)
    os.close(descriptor)
    log_stat = log_path.stat()
    receipt_path = attempt_root / "launch_receipt.json"
    accepted_path = attempt_root / "launch_accepted.json"
    release_path = attempt_root / "launch_ownership_release.json"
    claim_path = policy_root / "preflight_control/wrapper_claim.json"
    command = [
        item.decode("utf-8")
        for item in Path(f"/proc/{os.getpid()}/cmdline")
        .read_bytes()
        .split(b"\0")
        if item
    ]
    git: dict[str, str] = {}
    for name, arguments in (
        ("head_sha", ("rev-parse", "HEAD")),
        ("origin_master_sha", ("rev-parse", "origin/master")),
        ("branch", ("branch", "--show-current")),
    ):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        git[name] = result.stdout.strip()
    proc_executable = Path(
        os.readlink(f"/proc/{os.getpid()}/exe")
    ).resolve()
    wrapper_path = Path(module.__file__).resolve()
    receipt = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_launch_receipt_v1",
        "attempt_id": attempt_id,
        "started_registry": binding(
            started_path, "launch_started_registry_sha256"
        ),
        "policy_sha256": policy_sha256,
        "git": git,
        "bindings": {
            "config": {
                "path": str(config.resolve()),
                "sha256": file_sha(config),
            },
            "launcher": {
                "path": str(wrapper_path),
                "sha256": file_sha(wrapper_path),
            },
            "wrapper": {
                "path": str(wrapper_path),
                "sha256": file_sha(wrapper_path),
            },
            "controller": {
                "path": str(wrapper_path),
                "sha256": file_sha(wrapper_path),
            },
        },
        "python_executable": {
            "path": str(proc_executable),
            "sha256": file_sha(proc_executable),
        },
        "controller_session": module.CONTROLLER_SESSION,
        "controller_owner_nonce": owner_nonce,
        "observer_session": module.OBSERVER_SESSION,
        "wrapper_arguments": command,
        "pane_gate_arguments": [sys.executable, "fixture-pane-gate"],
        "tmux_arguments": ["tmux", "new-session", "fixture"],
        "shell": False,
        "pane_log": {
            "path": str(log_path.resolve()),
            "device": int(log_stat.st_dev),
            "inode": int(log_stat.st_ino),
            "mode": int(log_stat.st_mode),
            "size": int(log_stat.st_size),
        },
        "wrapper_claim_path": str(claim_path.resolve()),
        "started_at": utc_now(),
    }
    receipt["launch_receipt_sha256"] = digest(
        receipt, "launch_receipt_sha256"
    )
    write_exclusive(receipt_path, receipt)
    receipt_identity = file_identity(receipt_path)
    receipt_binding = binding(
        receipt_path, "launch_receipt_sha256"
    )
    gate_ready_path = attempt_root / "pane_gate_ready.json"
    current_process = process_identity(os.getpid())
    gate_ready = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_pane_gate_ready_v1"
        ),
        "launch_receipt": receipt_binding,
        "launch_receipt_identity": receipt_identity,
        "process": current_process,
        "wrapper_arguments": command,
        "ready_at": utc_now(),
    }
    gate_ready["pane_gate_ready_sha256"] = digest(
        gate_ready, "pane_gate_ready_sha256"
    )
    write_exclusive(gate_ready_path, gate_ready)
    tmux_started_path = attempt_root / "launch_tmux_started.json"
    tmux_started = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_tmux_started_v1"
        ),
        "launch_receipt": receipt_binding,
        "launch_receipt_identity": receipt_identity,
        "pane_gate_ready": binding(
            gate_ready_path, "pane_gate_ready_sha256"
        ),
        "tmux_client": {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
        "owner_seal": {"pane_process": current_process},
        "remain_on_exit": "on",
        "started_at": utc_now(),
    }
    tmux_started["launch_tmux_started_sha256"] = digest(
        tmux_started, "launch_tmux_started_sha256"
    )
    write_exclusive(tmux_started_path, tmux_started)
    os.environ[module.TMUX_OWNER_ENV] = owner_nonce
    os.environ[module.LAUNCH_RECEIPT_PATH_ENV] = str(receipt_path.resolve())
    os.environ[module.LAUNCH_ACCEPTED_PATH_ENV] = str(
        accepted_path.resolve()
    )
    os.environ[module.LAUNCH_RELEASE_PATH_ENV] = str(
        release_path.resolve()
    )
    os.environ[module.PANE_LOG_PATH_ENV] = str(log_path.resolve())

    def accept_claim() -> None:
        deadline = time.monotonic() + 45
        while not claim_path.is_file():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.02)
        claim_binding = binding(
            claim_path, "wrapper_claim_sha256"
        )
        accepted = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_launch_accepted_v1"
            ),
            "attempt_id": attempt_id,
            "launch_receipt": receipt_binding,
            "launch_receipt_identity": receipt_identity,
            "wrapper_claim": claim_binding,
            "startup_window_closed": False,
        }
        accepted["launch_accepted_sha256"] = digest(
            accepted, "launch_accepted_sha256"
        )
        write_exclusive(accepted_path, accepted)
        terminal_path = attempt_root / "launch_terminal.json"
        terminal = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_launch_terminal_v1"
            ),
            "launch_receipt": receipt_binding,
            "launch_receipt_identity": receipt_identity,
            "wrapper_claim": claim_binding,
            "status": "ownership_transferred",
            "failure": None,
        }
        terminal["launch_terminal_sha256"] = digest(
            terminal, "launch_terminal_sha256"
        )
        write_exclusive(terminal_path, terminal)
        release = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_launch_ownership_release_v1"
            ),
            "launch_receipt": receipt_binding,
            "launch_receipt_identity": receipt_identity,
            "launch_accepted": binding(
                accepted_path, "launch_accepted_sha256"
            ),
            "launch_terminal": binding(
                terminal_path, "launch_terminal_sha256"
            ),
            "wrapper_claim": claim_binding,
            "startup_window_closed": True,
        }
        release["launch_ownership_release_sha256"] = digest(
            release, "launch_ownership_release_sha256"
        )
        write_exclusive(release_path, release)

    thread = threading.Thread(target=accept_claim, daemon=True)
    thread.start()
    return thread


def prepare_supervised_launcher_contract(
    *,
    module: Any,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    wrapper_arguments: list[str],
) -> dict[str, Any]:
    module._install_verified_preflight_apis(config)
    verified = module._reverify_verified_preflight_apis()
    campaign_root = policy_root.parents[1]
    attempt_id = hashlib.sha256(
        (
            f"supervised:{policy_root.resolve()}:{os.getpid()}:"
            f"{wrapper_arguments}"
        ).encode()
    ).hexdigest()
    owner_nonce = hashlib.sha256(
        f"owner:{attempt_id}".encode()
    ).hexdigest()
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts"
        / "by_policy"
        / policy_sha256
        / attempt_id
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    started_path = (
        campaign_root
        / "preflight_launch_attempts"
        / "started"
        / f"{attempt_id}.json"
    )
    started = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_started_registry_v1"
        ),
        "attempt_id": attempt_id,
        "policy_sha256": policy_sha256,
        "reserved_at": utc_now(),
    }
    started["launch_started_registry_sha256"] = digest(
        started, "launch_started_registry_sha256"
    )
    write_exclusive(started_path, started)
    log_path = attempt_root / "pane.log"
    descriptor = os.open(
        log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    os.fsync(descriptor)
    os.close(descriptor)
    log_stat = log_path.stat()
    receipt_path = attempt_root / "launch_receipt.json"
    accepted_path = attempt_root / "launch_accepted.json"
    release_path = attempt_root / "launch_ownership_release.json"
    claim_path = policy_root / "preflight_control/wrapper_claim.json"
    gate_ready_path = attempt_root / "pane_gate_ready.json"
    tmux_started_path = attempt_root / "launch_tmux_started.json"
    wrapper_started_path = attempt_root / "wrapper_started.json"
    gate_terminal_path = attempt_root / "gate_execution_terminal.json"
    git: dict[str, str] = {}
    for name, arguments in (
        ("head_sha", ("rev-parse", "HEAD")),
        ("origin_master_sha", ("rev-parse", "origin/master")),
        ("branch", ("branch", "--show-current")),
    ):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        git[name] = result.stdout.strip()
    proc_executable = Path(
        os.readlink(f"/proc/{os.getpid()}/exe")
    ).resolve()
    wrapper_path = Path(module.__file__).resolve()
    gate_process = launch_process_identity(os.getpid())
    gate_arguments = [
        item.decode("utf-8")
        for item in Path(f"/proc/{os.getpid()}/cmdline")
        .read_bytes()
        .split(b"\0")
        if item
    ]
    receipt = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_launch_receipt_v1",
        "attempt_id": attempt_id,
        "started_registry": binding(
            started_path, "launch_started_registry_sha256"
        ),
        "policy_sha256": policy_sha256,
        "git": git,
        "bindings": {
            "config": {
                "path": str(config.resolve()),
                "sha256": file_sha(config),
            },
            "launcher": {
                "path": str(wrapper_path),
                "sha256": file_sha(wrapper_path),
            },
            "wrapper": {
                "path": str(wrapper_path),
                "sha256": file_sha(wrapper_path),
            },
            "controller": {
                "path": str(wrapper_path),
                "sha256": file_sha(wrapper_path),
            },
        },
        "verified_implementations": verified,
        "python_executable": {
            "path": str(proc_executable),
            "sha256": file_sha(proc_executable),
        },
        "controller_session": module.CONTROLLER_SESSION,
        "controller_owner_nonce": owner_nonce,
        "observer_session": module.OBSERVER_SESSION,
        "wrapper_arguments": wrapper_arguments,
        "pane_gate_arguments": gate_arguments,
        "tmux_arguments": ["tmux", "new-session", "fixture"],
        "shell": False,
        "pane_log": {
            "path": str(log_path.resolve()),
            "device": int(log_stat.st_dev),
            "inode": int(log_stat.st_ino),
            "mode": int(log_stat.st_mode),
            "size": int(log_stat.st_size),
        },
        "wrapper_claim_path": str(claim_path.resolve()),
        "wrapper_started_path": str(wrapper_started_path.resolve()),
        "gate_execution_terminal_path": str(
            gate_terminal_path.resolve()
        ),
        "started_at": utc_now(),
    }
    receipt["launch_receipt_sha256"] = digest(
        receipt, "launch_receipt_sha256"
    )
    write_exclusive(receipt_path, receipt)
    receipt_identity = file_identity(receipt_path)
    receipt_binding = binding(
        receipt_path, "launch_receipt_sha256"
    )
    gate_ready = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_pane_gate_ready_v1"
        ),
        "launch_receipt": receipt_binding,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": verified,
        "process": gate_process,
        "wrapper_arguments": wrapper_arguments,
        "ready_at": utc_now(),
    }
    gate_ready["pane_gate_ready_sha256"] = digest(
        gate_ready, "pane_gate_ready_sha256"
    )
    write_exclusive(gate_ready_path, gate_ready)
    subprocess.run(
        [
            "tmux",
            "set-environment",
            "-t",
            module.CONTROLLER_SESSION,
            module.TMUX_OWNER_ENV,
            owner_nonce,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    controller_tmux = module._tmux_identity(
        module.CONTROLLER_SESSION
    )
    controller_tmux_server = module._tmux_server_identity(
        controller_tmux["pane"]
    )
    tmux_owner_seal = module._build_tmux_owner_seal(
        controller_tmux,
        controller_tmux_server,
        owner_nonce,
    )
    tmux_started = module.build_tmux_started(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified,
        pane_gate_ready=binding(
            gate_ready_path, "pane_gate_ready_sha256"
        ),
        tmux_client={"returncode": 0, "stdout": "", "stderr": ""},
        owner_seal=tmux_owner_seal,
        started_at=utc_now(),
        tmux_identity=controller_tmux,
        tmux_server=controller_tmux_server,
    )
    write_exclusive(tmux_started_path, tmux_started)
    environment = {
        module.TMUX_OWNER_ENV: owner_nonce,
        module.LAUNCH_RECEIPT_PATH_ENV: str(receipt_path.resolve()),
        module.LAUNCH_ACCEPTED_PATH_ENV: str(accepted_path.resolve()),
        module.LAUNCH_RELEASE_PATH_ENV: str(release_path.resolve()),
        module.PANE_LOG_PATH_ENV: str(log_path.resolve()),
    }
    return {
        "attempt_root": attempt_root,
        "receipt_path": receipt_path,
        "accepted_path": accepted_path,
        "release_path": release_path,
        "claim_path": claim_path,
        "gate_ready_path": gate_ready_path,
        "wrapper_started_path": wrapper_started_path,
        "tmux_started_path": tmux_started_path,
        "log_path": log_path,
        "controller_pane": controller_tmux,
        "wrapper_arguments": wrapper_arguments,
        "gate_process": gate_process,
        "environment": environment,
        "receipt_identity": receipt_identity,
        "verified_implementations": verified,
    }


def complete_supervised_launcher_contract(
    context: Mapping[str, Any], child_pid: int
) -> threading.Thread:
    verified = dict(context["verified_implementations"])
    wrapper_arguments = list(context["wrapper_arguments"])
    process = launch_process_identity(child_pid)
    deadline = time.monotonic() + 5
    while True:
        command = [
            item.decode("utf-8")
            for item in Path(f"/proc/{child_pid}/cmdline")
            .read_bytes()
            .split(b"\0")
            if item
        ]
        if command == wrapper_arguments:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("fixture child command did not stabilize")
        time.sleep(0.005)
    executable_path = Path(
        os.readlink(f"/proc/{child_pid}/exe")
    ).resolve()
    executable_stat = executable_path.stat()
    executable = {
        "path": str(executable_path),
        "device": int(executable_stat.st_dev),
        "inode": int(executable_stat.st_ino),
        "mode": int(executable_stat.st_mode),
        "size": int(executable_stat.st_size),
    }
    wrapper_started = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_wrapper_started_v1"
        ),
        "launch_receipt": binding(
            context["receipt_path"], "launch_receipt_sha256"
        ),
        "launch_receipt_identity": dict(
            context["receipt_identity"]
        ),
        "verified_implementations": verified,
        "pane_gate_ready": binding(
            context["gate_ready_path"], "pane_gate_ready_sha256"
        ),
        "pane_gate_process": dict(context["gate_process"]),
        "wrapper_arguments": wrapper_arguments,
        "wrapper_process": process,
        "wrapper_executable": executable,
        "started_at": utc_now(),
    }
    wrapper_started["wrapper_started_sha256"] = digest(
        wrapper_started, "wrapper_started_sha256"
    )
    write_exclusive(
        context["wrapper_started_path"], wrapper_started
    )

    def accept_claim() -> None:
        claim_path = Path(context["claim_path"])
        deadline = time.monotonic() + 45
        while not claim_path.is_file():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.02)
        claim_binding = binding(
            claim_path, "wrapper_claim_sha256"
        )
        receipt_binding = binding(
            context["receipt_path"], "launch_receipt_sha256"
        )
        accepted = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_launch_accepted_v1"
            ),
            "attempt_id": load(context["receipt_path"])["attempt_id"],
            "launch_receipt": receipt_binding,
            "launch_receipt_identity": dict(
                context["receipt_identity"]
            ),
            "verified_implementations": verified,
            "wrapper_claim": claim_binding,
            "tmux_started": binding(
                context["tmux_started_path"],
                "launch_tmux_started_sha256",
            ),
            "pane": dict(context["controller_pane"]),
            "pane_log_path": str(
                Path(context["log_path"]).resolve()
            ),
            "startup_window_closed": False,
            "started_at": load(context["receipt_path"])["started_at"],
            "accepted_at": utc_now(),
        }
        accepted["launch_accepted_sha256"] = digest(
            accepted, "launch_accepted_sha256"
        )
        write_exclusive(context["accepted_path"], accepted)
        terminal_path = Path(context["accepted_path"]).with_name(
            "launch_terminal.json"
        )
        terminal = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_launch_terminal_v1"
            ),
            "launch_receipt": receipt_binding,
            "launch_receipt_identity": dict(
                context["receipt_identity"]
            ),
            "verified_implementations": verified,
            "launch_accepted": binding(
                context["accepted_path"], "launch_accepted_sha256"
            ),
            "wrapper_claim": claim_binding,
            "tmux_started": binding(
                context["tmux_started_path"],
                "launch_tmux_started_sha256",
            ),
            "status": "ownership_transferred",
            "failure": None,
            "tmux_client": None,
            "pane": dict(context["controller_pane"]),
            "pane_log": file_identity(Path(context["log_path"])),
            "session_residual": True,
            "started_at": load(context["receipt_path"])["started_at"],
            "completed_at": utc_now(),
        }
        terminal["launch_terminal_sha256"] = digest(
            terminal, "launch_terminal_sha256"
        )
        write_exclusive(terminal_path, terminal)
        release = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_launch_ownership_release_v1"
            ),
            "launch_receipt": receipt_binding,
            "launch_receipt_identity": dict(
                context["receipt_identity"]
            ),
            "verified_implementations": verified,
            "launch_accepted": binding(
                context["accepted_path"], "launch_accepted_sha256"
            ),
            "launch_terminal": binding(
                terminal_path, "launch_terminal_sha256"
            ),
            "wrapper_claim": claim_binding,
            "startup_window_closed": True,
            "released_at": utc_now(),
        }
        release["launch_ownership_release_sha256"] = digest(
            release, "launch_ownership_release_sha256"
        )
        write_exclusive(context["release_path"], release)

    thread = threading.Thread(target=accept_claim, daemon=True)
    thread.start()
    return thread


def publish_observer_bootstrap(config: Path) -> None:
    raw_path = os.environ.get("SAFA_PREFLIGHT_OBSERVER_BOOTSTRAP_PATH")
    if raw_path is None:
        return
    path = Path(raw_path)
    wrapper_binding = json.loads(
        os.environ["SAFA_PREFLIGHT_WRAPPER_CLAIM"]
    )
    policy_sha256 = os.environ[
        "SAFA_PREFLIGHT_OBSERVER_POLICY_SHA256"
    ]
    owner_nonce = os.environ[
        "SAFA_PREFLIGHT_OBSERVER_OWNER_NONCE"
    ]
    process = process_identity(os.getpid())
    observer_session = current_observer_session()
    tmux = tmux_identity(observer_session)
    executable = os.readlink(f"/proc/{os.getpid()}/exe")
    command = [
        item.decode("utf-8")
        for item in Path(f"/proc/{os.getpid()}/cmdline")
        .read_bytes()
        .split(b"\0")
        if item
    ]
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_observer_bootstrap_v1",
        "policy_sha256": policy_sha256,
        "verified_implementations": verified_implementations(config),
        "wrapper_claim": wrapper_binding,
        "observer_session": observer_session,
        "owner_nonce": owner_nonce,
        "process": process,
        "executable": executable,
        "executable_identity": file_identity(
            Path(executable).resolve(strict=True)
        ),
        "command": command,
        "tmux": tmux,
        "published_at": utc_now(),
    }
    value["observer_bootstrap_sha256"] = digest(
        value, "observer_bootstrap_sha256"
    )
    write_exclusive(path, value)


def observer(
    root: Path, mode: str, policy: str, config: Path
) -> int:
    publish_observer_bootstrap(config)
    observer_session = current_observer_session()
    control = root / "preflight_control"
    launch_path = control / "observer_launch.json"
    process_start_path = control / "controller_process_start.json"
    process_exit_path = control / "controller_process_exit.json"
    wait_file(launch_path)
    launch = load(launch_path)
    if launch["process"] != process_identity(os.getpid()):
        raise RuntimeError("fixture observer process differs from sealed launch")
    stop_binding = None
    failure = None
    if mode == "timeout":
        while True:
            time.sleep(1.0)
    if mode == "stop":
        wait_file(process_start_path)
        start = load(process_start_path)
        stop = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_observer_stop_v2",
            "campaign_id": "integration",
            "policy_sha256": policy,
            "wrapper_claim": launch["wrapper_claim"],
            "observer_launch": binding(
                launch_path, "observer_launch_sha256"
            ),
            "observer_claim": None,
            "observer_ready": None,
            "controller_process_start": binding(
                process_start_path, "controller_process_start_sha256"
            ),
            "controller_ready": None,
            "observer_session": observer_session,
            "observer_pid": os.getpid(),
            "observer_process": process_identity(os.getpid()),
            "observer_tmux": tmux_identity(observer_session),
            "controller_process": start["process"],
            "failure": {
                "type": "CanonicalScreeningError",
                "message": "controlled resource hard stop",
            },
            "requested_at": utc_now(),
        }
        stop["observer_stop_sha256"] = digest(
            stop, "observer_stop_sha256"
        )
        stop_path = control / "observer_stop.json"
        write_exclusive(stop_path, stop)
        stop_binding = binding(stop_path, "observer_stop_sha256")
        failure = stop["failure"]
    wait_file(process_exit_path)
    process_exit_binding = binding(
        process_exit_path, "controller_process_exit_sha256"
    )
    if mode in {"failure", "snapshot_failed_to_completed"}:
        failure = {
            "type": "ControllerEarlyExit",
            "message": "controller exited without terminal",
        }
        wait_file(process_start_path)
        start = load(process_start_path)
        stop = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_observer_stop_v2",
            "campaign_id": "integration",
            "policy_sha256": policy,
            "wrapper_claim": launch["wrapper_claim"],
            "observer_launch": binding(
                launch_path, "observer_launch_sha256"
            ),
            "observer_claim": None,
            "observer_ready": None,
            "controller_process_start": binding(
                process_start_path, "controller_process_start_sha256"
            ),
            "controller_ready": None,
            "observer_session": observer_session,
            "observer_pid": os.getpid(),
            "observer_process": process_identity(os.getpid()),
            "observer_tmux": tmux_identity(observer_session),
            "controller_process": start["process"],
            "failure": failure,
            "requested_at": utc_now(),
        }
        stop["observer_stop_sha256"] = digest(
            stop, "observer_stop_sha256"
        )
        stop_path = control / "observer_stop.json"
        write_exclusive(stop_path, stop)
        stop_binding = binding(stop_path, "observer_stop_sha256")
    claim_binding = None
    ready_binding = None
    if failure is None or mode == "snapshot_failed_to_completed":
        observer_launch_binding = binding(
            launch_path, "observer_launch_sha256"
        )
        observer_process = process_identity(os.getpid())
        claim = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_observer_claim_v1",
            "campaign_id": "integration",
            "phase": "preflight",
            "policy_sha256": policy,
            "observer_launch": observer_launch_binding,
            "observer_session": observer_session,
            "observer_pid": os.getpid(),
            "observer_process": observer_process,
            "claimed_at": utc_now(),
        }
        claim["observer_claim_sha256"] = digest(
            claim, "observer_claim_sha256"
        )
        claim_path = control / "observer_claim.json"
        write_exclusive(claim_path, claim)
        claim_binding = binding(claim_path, "observer_claim_sha256")
        ready = {
            "schema_version": 1,
            "contract_type": "safa_canonical_preflight_observer_ready_v1",
            "campaign_id": "integration",
            "phase": "preflight",
            "policy_sha256": policy,
            "observer_claim": claim_binding,
            "observer_claim_sha256": claim["observer_claim_sha256"],
            "observer_launch": observer_launch_binding,
            "observer_session": observer_session,
            "observer_pid": os.getpid(),
            "observer_process": observer_process,
            "ready_at": utc_now(),
        }
        ready["observer_ready_sha256"] = digest(
            ready, "observer_ready_sha256"
        )
        ready_path = control / "observer_ready.json"
        write_exclusive(ready_path, ready)
        ready_binding = binding(ready_path, "observer_ready_sha256")
    terminal = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_observer_terminal_v1",
        "campaign_id": "integration",
        "phase": "preflight",
        "policy_sha256": policy,
        "observer_claim": claim_binding,
        "observer_ready": ready_binding,
        "status": "failed" if failure is not None else "completed",
        "failure": failure,
        "samples": 0,
        "progress_samples": None,
        "resource_guard": None,
        "controller_terminal": None,
        "controller_process_exit": process_exit_binding,
        "observer_stop": stop_binding,
        "completed_at": utc_now(),
    }
    if mode == "terminal_process_exit_null":
        terminal["controller_process_exit"] = None
    elif mode == "terminal_process_exit_path":
        terminal["controller_process_exit"] = {
            **process_exit_binding,
            "path": str((control / "wrong_process_exit.json").resolve()),
        }
    elif mode == "terminal_process_exit_sha":
        terminal["controller_process_exit"] = {
            **process_exit_binding,
            "sha256": "0" * 64,
        }
    elif mode == "terminal_process_exit_canonical":
        terminal["controller_process_exit"] = {
            **process_exit_binding,
            "canonical_sha256": "0" * 64,
        }
    if mode == "terminal_malformed":
        descriptor = os.open(
            control / "observer_terminal.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            os.write(descriptor, b"{")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        while True:
            time.sleep(1.0)
    terminal["observer_terminal_sha256"] = digest(
        terminal, "observer_terminal_sha256"
    )
    write_exclusive(control / "observer_terminal.json", terminal)
    while True:
        time.sleep(1.0)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("preflight_wrapper_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load wrapper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrapper(args: argparse.Namespace) -> int:
    module = load_module(args.wrapper_module)
    module.OBSERVER_TERMINAL_WAIT_SECONDS = args.terminal_timeout
    module.PROCESS_TERMINATION_WAIT_SECONDS = (
        0.5
        if args.observer_mode
        in {
            "controller_fault_monitor",
            "controller_fault_monitor_fsync",
            "controller_fault_monitor_cleanup_write",
        }
        else 0.05
        if args.observer_mode.startswith("controller_fault_")
        else 10.0
    )
    helper = Path(__file__).resolve()
    controller_code = (
        (
            "import signal,time,sys;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            f"time.sleep({args.controller_seconds});"
            "sys.exit(0)"
        )
        if args.observer_mode
        in {
            "controller_fault_monitor",
            "controller_fault_monitor_fsync",
            "controller_fault_monitor_cleanup_write",
        }
        else (
            "import time,sys;"
            f"time.sleep({args.controller_seconds});"
            f"sys.exit({args.controller_exit})"
        )
    )
    if args.observer_mode == "controller_fault_identity":
        original_require_identity = module._require_process_identity

        def fail_controller_identity(pid: int, label: str):
            if label == "CPU preflight controller":
                raise RuntimeError("fixture controller identity failure")
            return original_require_identity(pid, label)

        module._require_process_identity = fail_controller_identity
    if args.observer_mode == "controller_fault_pgid":
        original_require_identity = module._require_process_identity

        def wrong_controller_pgid(pid: int, label: str):
            value = original_require_identity(pid, label)
            if label == "CPU preflight controller":
                value = {**value, "pgid": pid + 1}
            return value

        module._require_process_identity = wrong_controller_pgid
    if args.observer_mode in {
        "controller_fault_start_write",
        "controller_fault_start_write_process_exit_write",
    }:
        original_write = module._write_exclusive

        def fail_start_write(path: Path, value: Mapping[str, Any]) -> None:
            if path.name == "controller_process_start.json":
                raise RuntimeError("fixture controller start write failure")
            if (
                args.observer_mode
                == "controller_fault_start_write_process_exit_write"
                and path.name == "controller_process_exit.json"
            ):
                raise RuntimeError("fixture controller exit write failure")
            original_write(path, value)

        module._write_exclusive = fail_start_write
    if args.observer_mode in {
        "controller_fault_monitor",
        "controller_fault_monitor_fsync",
        "controller_fault_monitor_cleanup_write",
    }:
        original_assert_tmux = module._assert_tmux_process_identity
        process_start_path = (
            args.policy_root
            / "preflight_control/controller_process_start.json"
        )
        monitor_failure_injected = False

        def fail_monitor_identity(*call_args: Any, **call_kwargs: Any):
            nonlocal monitor_failure_injected
            if (
                not monitor_failure_injected
                and call_args
                and call_args[0] == module.OBSERVER_SESSION
                and process_start_path.is_file()
            ):
                monitor_failure_injected = True
                time.sleep(0.1)
                raise RuntimeError("fixture controller monitor failure")
            return original_assert_tmux(*call_args, **call_kwargs)

        module._assert_tmux_process_identity = fail_monitor_identity
    if args.observer_mode in {
        "controller_fault_log_fsync",
        "controller_fault_log_close",
        "controller_fault_monitor_fsync",
        "controller_fault_log_fsync_close",
    }:
        process_log_path = (
            args.policy_root / "preflight_control/controller_process.log"
        ).resolve()

        def is_process_log_descriptor(descriptor: int) -> bool:
            try:
                return Path(
                    os.readlink(f"/proc/self/fd/{descriptor}")
                ).resolve() == process_log_path
            except FileNotFoundError:
                return False

        if args.observer_mode in {
            "controller_fault_log_fsync",
            "controller_fault_monitor_fsync",
        }:
            original_fsync = module.os.fsync

            def fail_log_fsync(descriptor: int) -> None:
                if is_process_log_descriptor(descriptor):
                    raise OSError("fixture controller log fsync failure")
                original_fsync(descriptor)

            module.os.fsync = fail_log_fsync
        elif args.observer_mode == "controller_fault_log_close":
            original_close = module.os.close

            def fail_log_close(descriptor: int) -> None:
                is_log = is_process_log_descriptor(descriptor)
                original_close(descriptor)
                if is_log:
                    raise OSError("fixture controller log close failure")

            module.os.close = fail_log_close
        else:
            original_fsync = module.os.fsync
            original_close = module.os.close

            def fail_log_fsync_and_close(descriptor: int) -> None:
                if is_process_log_descriptor(descriptor):
                    raise OSError("fixture controller log fsync failure")
                original_fsync(descriptor)

            def fail_log_close_after_fsync(descriptor: int) -> None:
                is_log = is_process_log_descriptor(descriptor)
                original_close(descriptor)
                if is_log:
                    raise OSError("fixture controller log close failure")

            module.os.fsync = fail_log_fsync_and_close
            module.os.close = fail_log_close_after_fsync
    if args.observer_mode in {
        "controller_fault_observer_launch_write",
        "controller_fault_observer_cleanup_write",
        "controller_fault_monitor_cleanup_write",
        "controller_fault_wrapper_exit_write",
    }:
        original_contract_write = module._write_exclusive
        contract_failure_injected = False

        def fail_contract_write(
            path: Path, value: Mapping[str, Any]
        ) -> None:
            nonlocal contract_failure_injected
            target = {
                "controller_fault_observer_launch_write": (
                    "observer_launch.json"
                ),
                "controller_fault_observer_cleanup_write": (
                    "observer_cleanup.json"
                ),
                "controller_fault_monitor_cleanup_write": (
                    "observer_cleanup.json"
                ),
                "controller_fault_wrapper_exit_write": "wrapper_exit.json",
            }[args.observer_mode]
            if not contract_failure_injected and path.name == target:
                contract_failure_injected = True
                raise OSError(f"fixture {target} write failure")
            original_contract_write(path, value)

        module._write_exclusive = fail_contract_write
    if args.observer_mode == "controller_fault_process_exit_binding":
        original_sha256_file = module._sha256_file

        def fail_process_exit_binding(path: Path) -> str:
            if path.name == "controller_process_exit.json":
                raise OSError("fixture process exit binding hash failure")
            return original_sha256_file(path)

        module._sha256_file = fail_process_exit_binding
    if args.observer_mode == "controller_fault_observer_launch_binding":
        original_sha256_file = module._sha256_file

        def fail_observer_launch_binding(path: Path) -> str:
            if path.name == "observer_bootstrap.json":
                raise OSError(
                    "fixture observer launch binding hash failure"
                )
            return original_sha256_file(path)

        module._sha256_file = fail_observer_launch_binding
    if args.observer_mode == "controller_fault_process_log_mkdir":
        original_mkdir = module.Path.mkdir
        control_path = (
            args.policy_root / "preflight_control"
        ).resolve()
        mkdir_failure_injected = False

        def fail_process_log_mkdir(
            self: Path, *call_args: Any, **call_kwargs: Any
        ) -> None:
            nonlocal mkdir_failure_injected
            if (
                not mkdir_failure_injected
                and self.resolve() == control_path
                and (control_path / "observer_launch.json").is_file()
                and not (
                    control_path / "controller_process.log"
                ).exists()
            ):
                mkdir_failure_injected = True
                raise OSError("fixture process log parent mkdir failure")
            original_mkdir(self, *call_args, **call_kwargs)

        module.Path.mkdir = fail_process_log_mkdir
    if args.observer_mode == "controller_fault_final_binding":
        original_optional_binding = module._optional_binding
        process_log_binding_calls = 0

        def fail_final_binding(path: Path):
            nonlocal process_log_binding_calls
            if path.name == "controller_process.log":
                process_log_binding_calls += 1
                if process_log_binding_calls == 2:
                    raise OSError("fixture final binding failure")
            return original_optional_binding(path)

        module._optional_binding = fail_final_binding
    if (
        args.observer_mode
        == "controller_fault_after_exact_owner_seal"
    ):
        original_probe = module._probe_observer_gate
        exact_owner_failure_injected = False

        def fail_after_exact_owner(*call_args: Any, **call_kwargs: Any):
            recorder = call_kwargs["owner_recorder"]

            def record_then_fail(*owner_args: Any) -> None:
                nonlocal exact_owner_failure_injected
                recorder(*owner_args)
                if not exact_owner_failure_injected:
                    exact_owner_failure_injected = True
                    raise OSError(
                        "fixture failure after exact owner seal"
                    )

            return original_probe(
                *call_args,
                **{
                    **call_kwargs,
                    "owner_recorder": record_then_fail,
                },
            )

        module._probe_observer_gate = fail_after_exact_owner
    if args.observer_mode == "terminal_validator_exception":
        def reject_terminal(*_args: Any, **_kwargs: Any):
            raise RuntimeError("fixture terminal validator failure")

        module._validate_terminal_stop_binding = reject_terminal
    if args.observer_mode in {
        "snapshot_completed_to_failed",
        "snapshot_failed_to_completed",
        "snapshot_delete",
        "snapshot_exception_replacement",
    }:
        original_wait = module._wait_observer_terminal
        terminal_path = (
            args.policy_root / "preflight_control/observer_terminal.json"
        )
        snapshot_mode = args.observer_mode

        def wait_then_replace(*call_args: Any, **call_kwargs: Any):
            snapshot = original_wait(*call_args, **call_kwargs)
            if snapshot is None:
                raise RuntimeError("fixture terminal snapshot is absent")
            value, terminal_binding = snapshot
            terminal_path.unlink()
            if snapshot_mode != "snapshot_delete":
                replacement = dict(value)
                if value["status"] == "completed":
                    replacement["status"] = "failed"
                    replacement["failure"] = {
                        "type": "FixtureReplacement",
                        "message": "replacement failed",
                    }
                else:
                    replacement["status"] = "completed"
                    replacement["failure"] = None
                replacement["observer_terminal_sha256"] = digest(
                    replacement, "observer_terminal_sha256"
                )
                write_exclusive(terminal_path, replacement)
            return value, terminal_binding

        module._wait_observer_terminal = wait_then_replace
        if snapshot_mode == "snapshot_exception_replacement":
            def reject_snapshot(*_args: Any, **_kwargs: Any):
                raise RuntimeError("fixture snapshot validator failure")

            module._validate_terminal_stop_binding = reject_snapshot
    if not args.supervised_child:
        raise RuntimeError("wrapper fixture child is not supervised")
    value = module.run_wrapped_controller(
        repo_root=args.repo_root,
        policy_root=args.policy_root,
        policy_sha256=args.policy,
        config=args.config,
        command=[sys.executable, "-c", controller_code],
        observer_command=[
            sys.executable,
            str(helper),
            "observer",
            "--config",
            str(args.config),
            "--policy-root",
            str(args.policy_root),
            "--policy",
            args.policy,
            "--mode",
            args.observer_mode,
        ],
    )
    print(json.dumps(value, sort_keys=True))
    return int(value["exit_code"])


def controlled_probes(module: Any, *, violate_memory: bool) -> None:
    gpu_rows = [
        {
            "index": index,
            "uuid": f"GPU-INTEGRATION-{index}",
            "memory_total_mib": 24000,
            "memory_used_mib": 0,
            "memory_free_mib": 24000,
            "temperature_c": 30,
        }
        for index in range(4)
    ]
    module._gpu_snapshot = lambda: [dict(row) for row in gpu_rows]
    module._gpu_compute_processes = lambda: []
    module._cpu_load_percent = lambda: 1.0
    cpu_counter = {"total": 1000, "idle": 900}

    def cpu_times() -> tuple[int, int]:
        cpu_counter["total"] += 100
        cpu_counter["idle"] += 90
        return cpu_counter["total"], cpu_counter["idle"]

    module._cpu_times = cpu_times
    module._memory_snapshot_bytes = lambda: {
        "total_bytes": 10_000_000,
        "used_bytes": 1_000_000,
        "available_bytes": 9_000_000,
    }
    module._memory_percent = lambda: 95.0 if violate_memory else 10.0
    module._disk_percent = lambda _path: 10.0
    module._swap_pages = lambda: (0, 0)


def production_role(args: argparse.Namespace) -> int:
    fixture = load(args.fixture)
    module = load_module(args.controller_module)
    module._install_verified_preflight_contract_api(
        module.REPO_ROOT
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    if args.role == "observer":
        module._publish_preflight_observer_bootstrap_from_environment()
    contracts = importlib.import_module("safa.closeout.canonical_screening")
    for name in module._CONTRACT_EXPORTS:
        setattr(module, name, getattr(contracts, name))
    policy = fixture["policy"]
    paths = module._paths(
        Path(fixture["campaign_root"]), policy["policy_sha256"]
    )
    module._expected_preflight_controller_command = (
        lambda _policy, _paths: fixture["controller_command"]
    )
    module._expected_preflight_observer_command = (
        lambda _policy, _paths: fixture["observer_command"]
    )
    module.PREFLIGHT_BARRIER_TIMEOUT_SECONDS = float(
        fixture.get(
            "barrier_timeout",
            module.PREFLIGHT_BARRIER_TIMEOUT_SECONDS,
        )
    )
    controlled_probes(
        module,
        violate_memory=(
            args.role == "observer" and fixture["resource_stop"]
        ),
    )
    if args.role == "controller":
        if fixture.get("mode") == "early_exit":
            os._exit(2)

        def strict_preflight(*_args: Any, **_kwargs: Any) -> Any:
            time.sleep(float(fixture.get("checkpoint_delay", 0.0)))
            return fixture["strict_preflight"]

        module.preflight_generator_checkpoint = strict_preflight
        module._execute_preflight_controller(policy, paths)
    else:
        module._run_preflight_monitor(policy, paths)
    return 0


def production_wrapper(args: argparse.Namespace) -> int:
    fixture = load(args.fixture)
    module = load_module(args.wrapper_module)
    module.OBSERVER_TERMINAL_WAIT_SECONDS = float(
        fixture.get("terminal_timeout", 10.0)
    )
    module.PROCESS_TERMINATION_WAIT_SECONDS = float(
        fixture.get("process_termination_wait", 10.0)
    )
    if fixture.get("mode") in {
        "process_exit_delay",
        "process_exit_barrier_timeout",
    }:
        original_write = module._write_exclusive
        process_exit_path = (
            args.policy_root
            / "preflight_control/controller_process_exit.json"
        ).resolve()
        controller_terminal_path = (
            args.policy_root / "preflight_control/controller_terminal.json"
        )
        observer_terminal_path = (
            args.policy_root / "preflight_control/observer_terminal.json"
        )
        marker_path = (
            args.policy_root / "preflight_control/process_exit_barrier.json"
        )
        barrier_mode = fixture["mode"]

        def delayed_process_exit(path: Path, value: Mapping[str, Any]) -> None:
            if path.resolve() != process_exit_path:
                original_write(path, value)
                return
            wait_file(controller_terminal_path, seconds=5.0)
            if observer_terminal_path.exists():
                raise RuntimeError(
                    "observer terminal preceded process-exit barrier"
                )
            if barrier_mode == "process_exit_delay":
                time.sleep(0.5)
                if observer_terminal_path.exists():
                    raise RuntimeError(
                        "observer terminal crossed delayed process-exit barrier"
                    )
                observed_status = None
            else:
                wait_file(observer_terminal_path, seconds=5.0)
                observed_terminal = load(observer_terminal_path)
                if (
                    observed_terminal.get("status") != "failed"
                    or observed_terminal.get("controller_process_exit")
                    is not None
                    or observed_terminal.get("observer_stop") is None
                ):
                    raise RuntimeError(
                        "observer process-exit timeout contract differs"
                    )
                observed_status = observed_terminal["status"]
            marker = {
                "schema_version": 1,
                "contract_type": "safa_test_process_exit_barrier_v1",
                "mode": barrier_mode,
                "controller_terminal_before_process_exit": True,
                "observer_terminal_before_process_exit": (
                    observer_terminal_path.exists()
                ),
                "observer_status_before_process_exit": observed_status,
                "observed_at": utc_now(),
            }
            marker["process_exit_barrier_sha256"] = digest(
                marker, "process_exit_barrier_sha256"
            )
            write_exclusive(marker_path, marker)
            original_write(path, value)

        module._write_exclusive = delayed_process_exit
    if fixture.get("mode") in {
        "late_terminal_race",
        "late_snapshot_replacement",
        "late_snapshot_delete",
    }:
        original_terminate = module._terminate_bound_observer
        terminal_path = (
            args.policy_root / "preflight_control/observer_terminal.json"
        )
        race_path = (
            args.policy_root
            / "preflight_control/late_terminal_race_window.json"
        )

        def terminate_after_late_terminal(*call_args: Any, **call_kwargs: Any):
            if terminal_path.exists():
                raise RuntimeError(
                    "late-terminal race fixture missed the timeout boundary"
                )
            race = {
                "schema_version": 1,
                "contract_type": "safa_test_late_terminal_race_window_v1",
                "terminal_absent_after_wait": True,
                "opened_at": utc_now(),
            }
            race["race_window_sha256"] = digest(
                race, "race_window_sha256"
            )
            write_exclusive(race_path, race)
            wait_file(terminal_path, seconds=5.0)
            terminal = load(terminal_path)
            if terminal.get("observer_terminal_sha256") != digest(
                terminal, "observer_terminal_sha256"
            ):
                raise RuntimeError(
                    "late-terminal race fixture observed invalid terminal"
                )
            return original_terminate(*call_args, **call_kwargs)

        module._terminate_bound_observer = terminate_after_late_terminal
    if fixture.get("mode") in {
        "late_snapshot_replacement",
        "late_snapshot_delete",
    }:
        original_read_terminal = module._read_observer_terminal
        late_snapshot_mode = fixture["mode"]

        def read_late_snapshot_then_mutate(
            path: Path,
            *call_args: Any,
            **call_kwargs: Any,
        ):
            value, terminal_binding = original_read_terminal(
                path, *call_args, **call_kwargs
            )
            path.unlink()
            if late_snapshot_mode == "late_snapshot_replacement":
                replacement = dict(value)
                replacement["status"] = "failed"
                replacement["failure"] = {
                    "type": "FixtureLateReplacement",
                    "message": "late replacement failed",
                }
                replacement["observer_terminal_sha256"] = digest(
                    replacement, "observer_terminal_sha256"
                )
                write_exclusive(path, replacement)
            return value, terminal_binding

        module._read_observer_terminal = read_late_snapshot_then_mutate
    if fixture.get("mode") == "late_terminal_foreign_replacement":
        original_terminate = module._terminate_bound_observer
        terminal_path = (
            args.policy_root / "preflight_control/observer_terminal.json"
        )
        replacement_path = (
            args.policy_root
            / "preflight_control/late_terminal_foreign_replacement.json"
        )

        def terminate_with_foreign_replacement(
            observer_tmux: Mapping[str, Any],
            observer_tmux_server: Mapping[str, Any],
            observer_tmux_owner_seal: Mapping[str, Any],
            observer_process: Mapping[str, int],
            **call_kwargs: Any,
        ):
            if terminal_path.exists():
                raise RuntimeError(
                    "late foreign fixture missed the timeout boundary"
                )
            original_run = module.subprocess.run
            injected = False

            def run_with_foreign_replacement(command, **run_kwargs):
                nonlocal injected
                conditional_owner_kill = (
                    len(command) == 10
                    and command[:4]
                    == [
                        "tmux",
                        "-S",
                        observer_tmux_owner_seal["socket_path"],
                        "if-shell",
                    ]
                    and command[4:7]
                    == ["-t", observer_tmux["pane"], "-F"]
                    and command[8]
                    == f"kill-pane -t {observer_tmux['pane']}"
                    and command[9]
                    == (
                        "display-message -p "
                        f"{module.TMUX_CONDITIONAL_KILL_REJECTED}"
                    )
                )
                if conditional_owner_kill:
                    if injected:
                        raise RuntimeError(
                            "late foreign fixture repeated conditional kill"
                        )
                    wait_file(terminal_path, seconds=5.0)
                    terminal = load(terminal_path)
                    if terminal.get(
                        "observer_terminal_sha256"
                    ) != digest(terminal, "observer_terminal_sha256"):
                        raise RuntimeError(
                            "late foreign fixture observed invalid terminal"
                        )
                    kill_result = original_run(
                        [
                            "tmux",
                            "-S",
                            observer_tmux_owner_seal["socket_path"],
                            "kill-pane",
                            "-t",
                            observer_tmux["pane"],
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if kill_result.returncode != 0:
                        return kill_result
                    process_deadline = time.monotonic() + 5.0
                    while True:
                        try:
                            current_snapshot = module._read_process_stat(
                                int(observer_process["pid"])
                            )
                        except (FileNotFoundError, ProcessLookupError):
                            current_snapshot = None
                        if current_snapshot is None:
                            break
                        current_process, current_state = current_snapshot
                        if (
                            current_process != dict(observer_process)
                            or current_state == "Z"
                        ):
                            break
                        if time.monotonic() >= process_deadline:
                            raise RuntimeError(
                                "sealed observer survived pane kill"
                            )
                        time.sleep(0.02)
                    original_run(
                        [
                            "tmux",
                            "new-session",
                            "-d",
                            "-s",
                            module.OBSERVER_SESSION,
                            sys.executable,
                            "-c",
                            "import time;time.sleep(30)",
                        ],
                        check=True,
                    )
                    foreign_tmux = module._tmux_identity(
                        module.OBSERVER_SESSION
                    )
                    foreign_server = module._tmux_server_identity(
                        foreign_tmux["pane"]
                    )
                    replacement = {
                        "schema_version": 1,
                        "contract_type": (
                            "safa_test_late_terminal_foreign_replacement_v1"
                        ),
                        "sealed_tmux": dict(observer_tmux),
                        "sealed_tmux_server": dict(observer_tmux_server),
                        "sealed_tmux_owner": dict(
                            observer_tmux_owner_seal
                        ),
                        "sealed_process": dict(observer_process),
                        "foreign_tmux": foreign_tmux,
                        "foreign_tmux_server": foreign_server,
                        "observer_terminal": binding(
                            terminal_path,
                            "observer_terminal_sha256",
                        ),
                        "created_at": utc_now(),
                    }
                    replacement[
                        "foreign_replacement_sha256"
                    ] = digest(
                        replacement, "foreign_replacement_sha256"
                    )
                    write_exclusive(replacement_path, replacement)
                    injected = True
                    return original_run(command, **run_kwargs)
                if command == [
                    "tmux",
                    "kill-pane",
                    "-t",
                    observer_tmux["pane"],
                ]:
                    raise RuntimeError(
                        "late foreign fixture observed non-atomic pane kill"
                    )
                return original_run(command, **run_kwargs)

            module.subprocess.run = run_with_foreign_replacement
            try:
                result = original_terminate(
                    observer_tmux,
                    observer_tmux_server,
                    observer_tmux_owner_seal,
                    observer_process,
                    **call_kwargs,
                )
            finally:
                module.subprocess.run = original_run
            if not injected:
                raise RuntimeError(
                    "late foreign fixture did not reach pane kill"
                )
            return result

        module._terminate_bound_observer = (
            terminate_with_foreign_replacement
        )
    if fixture.get("mode") == "proc_snapshot_absent":
        original_terminate = module._terminate_bound_observer
        original_snapshot = module._process_identity_state

        def terminate_with_absent_snapshot(
            observer_tmux: Mapping[str, Any],
            observer_tmux_server: Mapping[str, Any],
            observer_tmux_owner_seal: Mapping[str, Any],
            observer_process: Mapping[str, int],
            **call_kwargs: Any,
        ):
            observer_pid = int(observer_process["pid"])

            def absent_observer_snapshot(pid: int):
                if pid == observer_pid:
                    return None
                return original_snapshot(pid)

            module._process_identity_state = absent_observer_snapshot
            try:
                return original_terminate(
                    observer_tmux,
                    observer_tmux_server,
                    observer_tmux_owner_seal,
                    observer_process,
                    **call_kwargs,
                )
            finally:
                module._process_identity_state = original_snapshot

        module._terminate_bound_observer = (
            terminate_with_absent_snapshot
        )
    if not args.supervised_child:
        raise RuntimeError(
            "production wrapper fixture child is not supervised"
        )
    try:
        value = module.run_wrapped_controller(
            repo_root=args.repo_root,
            policy_root=args.policy_root,
            policy_sha256=args.policy,
            config=args.config,
            command=fixture["controller_command"],
            observer_command=fixture["observer_command"],
        )
    except BaseException:
        args.policy_root.mkdir(parents=True, exist_ok=True)
        (args.policy_root / "wrapper_fixture_error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    print(json.dumps(value, sort_keys=True))
    return int(value["exit_code"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    observe = sub.add_parser("observer")
    observe.add_argument("--config", type=Path, required=True)
    observe.add_argument("--policy-root", type=Path, required=True)
    observe.add_argument("--policy", required=True)
    observe.add_argument(
        "--mode",
        choices=(
            "success",
            "stop",
            "timeout",
            "failure",
            "terminal_process_exit_null",
            "terminal_process_exit_path",
            "terminal_process_exit_sha",
            "terminal_process_exit_canonical",
            "terminal_malformed",
            "terminal_validator_exception",
            "snapshot_completed_to_failed",
            "snapshot_failed_to_completed",
            "snapshot_delete",
            "snapshot_exception_replacement",
            "controller_fault_identity",
            "controller_fault_pgid",
            "controller_fault_start_write",
            "controller_fault_monitor",
            "controller_fault_log_fsync",
            "controller_fault_log_close",
            "controller_fault_monitor_fsync",
            "controller_fault_log_fsync_close",
            "controller_fault_start_write_process_exit_write",
            "controller_fault_observer_launch_write",
            "controller_fault_observer_cleanup_write",
            "controller_fault_monitor_cleanup_write",
            "controller_fault_process_exit_binding",
            "controller_fault_final_binding",
            "controller_fault_wrapper_exit_write",
            "controller_fault_observer_launch_binding",
            "controller_fault_process_log_mkdir",
            "controller_fault_after_exact_owner_seal",
        ),
        required=True,
    )
    wrap = sub.add_parser("wrapper")
    wrap.add_argument("--wrapper-module", type=Path, required=True)
    wrap.add_argument("--repo-root", type=Path, required=True)
    wrap.add_argument("--policy-root", type=Path, required=True)
    wrap.add_argument("--policy", required=True)
    wrap.add_argument("--config", type=Path, required=True)
    wrap.add_argument(
        "--observer-mode",
        choices=(
            "success",
            "stop",
            "timeout",
            "failure",
            "terminal_process_exit_null",
            "terminal_process_exit_path",
            "terminal_process_exit_sha",
            "terminal_process_exit_canonical",
            "terminal_malformed",
            "terminal_validator_exception",
            "snapshot_completed_to_failed",
            "snapshot_failed_to_completed",
            "snapshot_delete",
            "snapshot_exception_replacement",
            "controller_fault_identity",
            "controller_fault_pgid",
            "controller_fault_start_write",
            "controller_fault_monitor",
            "controller_fault_log_fsync",
            "controller_fault_log_close",
            "controller_fault_monitor_fsync",
            "controller_fault_log_fsync_close",
            "controller_fault_start_write_process_exit_write",
            "controller_fault_observer_launch_write",
            "controller_fault_observer_cleanup_write",
            "controller_fault_monitor_cleanup_write",
            "controller_fault_process_exit_binding",
            "controller_fault_final_binding",
            "controller_fault_wrapper_exit_write",
            "controller_fault_observer_launch_binding",
            "controller_fault_process_log_mkdir",
            "controller_fault_after_exact_owner_seal",
        ),
        required=True,
    )
    wrap.add_argument("--controller-seconds", type=float, default=0.2)
    wrap.add_argument("--controller-exit", type=int, default=0)
    wrap.add_argument("--terminal-timeout", type=float, default=3.0)
    wrap.add_argument("--supervised-child", action="store_true")
    production = sub.add_parser("production-role")
    production.add_argument("--controller-module", type=Path, required=True)
    production.add_argument("--fixture", type=Path, required=True)
    production.add_argument("--config", type=Path, required=True)
    production.add_argument(
        "--role", choices=("controller", "observer"), required=True
    )
    production_wrap = sub.add_parser("production-wrapper")
    production_wrap.add_argument("--wrapper-module", type=Path, required=True)
    production_wrap.add_argument("--repo-root", type=Path, required=True)
    production_wrap.add_argument("--policy-root", type=Path, required=True)
    production_wrap.add_argument("--policy", required=True)
    production_wrap.add_argument("--config", type=Path, required=True)
    production_wrap.add_argument("--fixture", type=Path, required=True)
    production_wrap.add_argument(
        "--supervised-child", action="store_true"
    )
    return parser.parse_args()


def supervise_wrapper_action(args: argparse.Namespace) -> int:
    module = load_module(args.wrapper_module)
    child_arguments = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--supervised-child",
    ]
    context = prepare_supervised_launcher_contract(
        module=module,
        repo_root=args.repo_root,
        policy_root=args.policy_root,
        policy_sha256=args.policy,
        config=args.config,
        wrapper_arguments=child_arguments,
    )
    environment = os.environ.copy()
    environment.update(context["environment"])
    child = subprocess.Popen(
        child_arguments,
        shell=False,
        preexec_fn=os.setsid,
        env=environment,
    )
    try:
        complete_supervised_launcher_contract(context, child.pid)
        return child.wait()
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=1)
        raise


def main() -> int:
    args = parse_args()
    if args.action == "observer":
        return observer(
            args.policy_root, args.mode, args.policy, args.config
        )
    if args.action == "production-role":
        return production_role(args)
    if args.action == "production-wrapper":
        if not args.supervised_child:
            return supervise_wrapper_action(args)
        return production_wrapper(args)
    if not args.supervised_child:
        return supervise_wrapper_action(args)
    return wrapper(args)


if __name__ == "__main__":
    raise SystemExit(main())
