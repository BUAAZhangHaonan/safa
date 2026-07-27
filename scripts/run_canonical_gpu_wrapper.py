#!/usr/bin/env python3
"""Durable launcher for one fail-closed canonical GPU controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence


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


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    content = _canonical_json(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.publish-{os.getpid()}-{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _optional_binding(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _normalized_exit(return_code: int) -> tuple[int, int | None]:
    if return_code >= 0:
        return return_code, None
    signal_number = -return_code
    return 128 + signal_number, signal_number


def _wait_observer_terminal(
    path: Path,
    policy_sha256: str,
    phase: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("observer terminal barrier timed out")
        time.sleep(0.1)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    ready_binding = value.get("observer_ready")
    ready_path = path.parent / "observer_ready.json"
    if not isinstance(ready_binding, Mapping) or (
        set(ready_binding) != {"path", "sha256", "canonical_sha256"}
        or Path(str(ready_binding["path"])).resolve() != ready_path.resolve()
        or not ready_path.is_file()
        or ready_binding["sha256"] != _sha256_file(ready_path)
    ):
        raise RuntimeError("observer terminal ready binding mismatch")
    with ready_path.open("r", encoding="utf-8") as handle:
        ready = json.load(handle)
    if (
        value.get("contract_type")
        != "safa_canonical_gpu_observer_terminal_v1"
        or value.get("policy_sha256") != policy_sha256
        or value.get("phase") != phase
        or value.get("observer_terminal_sha256")
        != _canonical_digest(value, "observer_terminal_sha256")
        or value.get("status") != "completed"
        or value.get("failure") is not None
        or ready.get("observer_ready_sha256")
        != ready_binding["canonical_sha256"]
        or ready.get("observer_ready_sha256")
        != _canonical_digest(ready, "observer_ready_sha256")
    ):
        raise RuntimeError("observer terminal contract mismatch")
    return value


def _load_valid_controller_ready(
    path: Path, policy_sha256: str, phase: str
) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("contract_type")
        != "safa_canonical_gpu_controller_ready_v1"
        or value.get("policy_sha256") != policy_sha256
        or value.get("phase") != phase
        or value.get("controller_ready_sha256")
        != _canonical_digest(value, "controller_ready_sha256")
    ):
        raise RuntimeError("controller ready contract mismatch")
    return value


def _launch_observer(
    *,
    repo_root: Path,
    python: str,
    config: Path,
    campaign_root: Path,
    phase: str,
) -> dict[str, Any]:
    session = f"safa-screening-{phase}-monitor"
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-c",
        str(repo_root),
        python,
        "-u",
        str(repo_root / "scripts/run_canonical_checkpoint_screening.py"),
        "--config",
        str(config),
        "--campaign-root",
        str(campaign_root),
        "--phase",
        "monitor",
        "--monitor-target",
        phase,
        "--execute",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "observer tmux launch failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return {
        "session": session,
        "command": command,
        "launched_at": _utc_now(),
    }


def _run_wrapped_controller_impl(
    *,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    campaign_root: Path,
    phase: str,
    python: str,
    command: Sequence[str],
) -> dict[str, Any]:
    control = policy_root.resolve() / "gpu_control" / phase
    wrapper_claim_path = control / "wrapper_claim.json"
    process_log_path = control / "controller_process.log"
    observer_launch_path = control / "observer_launch.json"
    process_exit_path = control / "controller_process_exit.json"
    wrapper_exit_path = control / "wrapper_exit.json"
    started_at = _utc_now()
    claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_wrapper_claim_v1",
        "policy_sha256": policy_sha256,
        "phase": phase,
        "config": {
            "path": str(config.resolve()),
            "sha256": _sha256_file(config),
        },
        "command": list(command),
        "wrapper_pid": os.getpid(),
        "started_at": started_at,
    }
    claim["wrapper_claim_sha256"] = _canonical_digest(
        claim, "wrapper_claim_sha256"
    )
    _write_exclusive(wrapper_claim_path, claim)
    try:
        observer_launch = _launch_observer(
            repo_root=repo_root,
            python=python,
            config=config,
            campaign_root=campaign_root,
            phase=phase,
        )
        observer_launch["status"] = "launched"
        observer_launch["failure"] = None
    except BaseException as exc:
        observer_launch = {
            "session": f"safa-screening-{phase}-monitor",
            "command": [],
            "launched_at": None,
            "status": "failed",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    observer_launch.update(
        {
            "schema_version": 1,
            "contract_type": "safa_canonical_gpu_observer_launch_v2",
            "policy_sha256": policy_sha256,
            "phase": phase,
            "wrapper_claim": {
                "path": str(wrapper_claim_path.resolve()),
                "sha256": _sha256_file(wrapper_claim_path),
                "canonical_sha256": claim["wrapper_claim_sha256"],
            },
            "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        }
    )
    observer_launch["observer_launch_sha256"] = _canonical_digest(
        observer_launch, "observer_launch_sha256"
    )
    _write_exclusive(observer_launch_path, observer_launch)
    process_log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    process: subprocess.Popen[bytes] | None = None
    return_code = 125
    launch_failure: dict[str, str] | None = None
    try:
        if observer_launch["status"] != "launched":
            raise RuntimeError("observer launch failed before controller release")
        descriptor = os.open(process_log_path, flags, 0o644)
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=descriptor,
            close_fds=True,
        )
        return_code = process.wait()
    except BaseException as exc:
        launch_failure = {"type": type(exc).__name__, "message": str(exc)}
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
    finally:
        if descriptor is not None:
            os.fsync(descriptor)
            os.close(descriptor)
    process_exit = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_process_exit_v1",
        "policy_sha256": policy_sha256,
        "phase": phase,
        "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        "return_code": return_code,
        "completed_at": _utc_now(),
    }
    process_exit["controller_process_exit_sha256"] = _canonical_digest(
        process_exit, "controller_process_exit_sha256"
    )
    _write_exclusive(process_exit_path, process_exit)
    if observer_launch is not None and observer_launch["status"] == "launched":
        try:
            _wait_observer_terminal(
                control / "observer_terminal.json",
                policy_sha256,
                phase,
                timeout_seconds=120.0,
            )
        except BaseException as exc:
            launch_failure = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if return_code == 0:
                return_code = 124
    exit_code, signal_number = _normalized_exit(return_code)
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_wrapper_exit_v1",
        "policy_sha256": policy_sha256,
        "phase": phase,
        "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        "command": list(command),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "exit_code": exit_code,
        "signal": signal_number,
        "launch_failure": launch_failure,
        "controller_process_log": _optional_binding(process_log_path),
        "controller_process_exit": _optional_binding(process_exit_path),
        "controller_claim": _optional_binding(control / "controller_claim.json"),
        "controller_ready": _optional_binding(control / "controller_ready.json"),
        "observer_launch": _optional_binding(observer_launch_path),
        "observer_ready": _optional_binding(control / "observer_ready.json"),
        "observer_terminal": _optional_binding(
            control / "observer_terminal.json"
        ),
        "controller_terminal": _optional_binding(
            control / "controller_terminal.json"
        ),
    }
    value["wrapper_exit_sha256"] = _canonical_digest(
        value, "wrapper_exit_sha256"
    )
    _write_exclusive(wrapper_exit_path, value)
    return value


def run_wrapped_controller(
    *,
    repo_root: Path,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    campaign_root: Path,
    phase: str,
    python: str,
    command: Sequence[str],
) -> dict[str, Any]:
    control = policy_root.resolve() / "gpu_control" / phase
    terminal_path = control / "wrapper_terminal.json"
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    started_at = _utc_now()
    try:
        result = _run_wrapped_controller_impl(
            repo_root=repo_root,
            policy_root=policy_root,
            policy_sha256=policy_sha256,
            config=config,
            campaign_root=campaign_root,
            phase=phase,
            python=python,
            command=command,
        )
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_failure: dict[str, str] | None = None
        if failure is not None:
            completed = subprocess.run(
                [
                    "tmux",
                    "kill-session",
                    "-t",
                    f"safa-screening-{phase}-monitor",
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode not in {0, 1}:
                cleanup_failure = {
                    "type": "ObserverCleanupError",
                    "message": (
                        completed.stderr.strip()
                        or completed.stdout.strip()
                        or f"exit_code={completed.returncode}"
                    ),
                }
        terminal = {
            "schema_version": 1,
            "contract_type": "safa_canonical_gpu_wrapper_terminal_v1",
            "policy_sha256": policy_sha256,
            "phase": phase,
            "config": {
                "path": str(config.resolve()),
                "sha256": (
                    _sha256_file(config) if config.is_file() else None
                ),
            },
            "command": list(command),
            "status": (
                "completed"
                if result is not None and result.get("exit_code") == 0
                else "failed"
            ),
            "failure": (
                (
                    None
                    if result is not None and result.get("exit_code") == 0
                    else {
                        "type": "ControllerExit",
                        "message": (
                            f"controller exit_code={result.get('exit_code')}"
                        ),
                    }
                )
                if failure is None
                else {
                    "type": type(failure).__name__,
                    "message": str(failure),
                }
            ),
            "cleanup_failure": cleanup_failure,
            "wrapper_claim": _optional_binding(
                control / "wrapper_claim.json"
            ),
            "observer_launch": _optional_binding(
                control / "observer_launch.json"
            ),
            "controller_process_exit": _optional_binding(
                control / "controller_process_exit.json"
            ),
            "controller_terminal": _optional_binding(
                control / "controller_terminal.json"
            ),
            "wrapper_exit": _optional_binding(
                control / "wrapper_exit.json"
            ),
            "started_at": started_at,
            "completed_at": _utc_now(),
        }
        terminal["wrapper_terminal_sha256"] = _canonical_digest(
            terminal, "wrapper_terminal_sha256"
        )
        _write_exclusive(terminal_path, terminal)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke8", "screen512"))
    parser.add_argument("--python", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
        args.phase,
        "--execute",
    ]
    value = run_wrapped_controller(
        repo_root=repo_root,
        policy_root=policy_root,
        policy_sha256=args.policy_sha256,
        config=config,
        campaign_root=campaign_root,
        phase=args.phase,
        python=args.python,
        command=command,
    )
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return int(value["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
