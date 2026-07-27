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
import subprocess
import sys
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


def process_identity(pid: int) -> dict[str, int]:
    raw_stat = Path(f"/proc/{pid}/stat").read_text()
    closing = raw_stat.rfind(")")
    if closing < 0:
        raise RuntimeError("fixture process stat is malformed")
    fields = raw_stat[closing + 2 :].split()
    return {
        "pid": pid,
        "pgid": int(fields[2]),
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


def publish_observer_bootstrap() -> None:
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
        "wrapper_claim": wrapper_binding,
        "observer_session": observer_session,
        "owner_nonce": owner_nonce,
        "process": process,
        "executable": executable,
        "command": command,
        "tmux": tmux,
        "published_at": utc_now(),
    }
    value["observer_bootstrap_sha256"] = digest(
        value, "observer_bootstrap_sha256"
    )
    write_exclusive(path, value)


def observer(root: Path, mode: str, policy: str) -> int:
    publish_observer_bootstrap()
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
    module.PROCESS_TERMINATION_WAIT_SECONDS = 10.0
    helper = Path(__file__).resolve()
    controller_code = (
        "import time,sys;"
        f"time.sleep({args.controller_seconds});"
        f"sys.exit({args.controller_exit})"
    )
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
        ),
        required=True,
    )
    wrap.add_argument("--controller-seconds", type=float, default=0.2)
    wrap.add_argument("--controller-exit", type=int, default=0)
    wrap.add_argument("--terminal-timeout", type=float, default=3.0)
    production = sub.add_parser("production-role")
    production.add_argument("--controller-module", type=Path, required=True)
    production.add_argument("--fixture", type=Path, required=True)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "observer":
        return observer(args.policy_root, args.mode, args.policy)
    if args.action == "production-role":
        return production_role(args)
    if args.action == "production-wrapper":
        return production_wrapper(args)
    return wrapper(args)


if __name__ == "__main__":
    raise SystemExit(main())
