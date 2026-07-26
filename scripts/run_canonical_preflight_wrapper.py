#!/usr/bin/env python3
"""Process-level durable wrapper for the canonical CPU preflight controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
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


def _optional_binding(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _normalized_exit(return_code: int) -> tuple[int, int | None]:
    if return_code >= 0:
        return return_code, None
    signal_number = -return_code
    return 128 + signal_number, signal_number


def run_wrapped_controller(
    *,
    policy_root: Path,
    policy_sha256: str,
    config: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    control = policy_root.resolve() / "preflight_control"
    wrapper_claim_path = control / "wrapper_claim.json"
    process_log_path = control / "controller_process.log"
    wrapper_exit_path = control / "wrapper_exit.json"
    started_at = _utc_now()
    claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_claim_v1",
        "policy_sha256": policy_sha256,
        "config": {
            "path": str(config.resolve()),
            "sha256": _sha256_file(config),
        },
        "command": list(command),
        "wrapper_pid": os.getpid(),
        "started_at": started_at,
        "external_timeout_seconds": None,
    }
    claim["wrapper_claim_sha256"] = hashlib.sha256(_canonical_json(claim)).hexdigest()
    _write_exclusive(wrapper_claim_path, claim)
    process_log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    return_code: int | None = None
    launch_failure: dict[str, str] | None = None
    try:
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
        return_code = 125
    finally:
        if descriptor is not None:
            os.fsync(descriptor)
            os.close(descriptor)
    exit_code, signal_number = _normalized_exit(return_code)
    controller_claim_path = control / "controller_claim.json"
    controller_terminal_path = control / "controller_terminal.json"
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_exit_v2",
        "policy_sha256": policy_sha256,
        "wrapper_claim_sha256": claim["wrapper_claim_sha256"],
        "command": list(command),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "exit_code": exit_code,
        "signal": signal_number,
        "launch_failure": launch_failure,
        "controller_process_log": _optional_binding(process_log_path),
        "controller_claim": _optional_binding(controller_claim_path),
        "controller_terminal": _optional_binding(controller_terminal_path),
    }
    value["wrapper_exit_sha256"] = hashlib.sha256(_canonical_json(value)).hexdigest()
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
        "preflight",
        "--execute",
    ]
    value = run_wrapped_controller(
        policy_root=policy_root,
        policy_sha256=args.policy_sha256,
        config=config,
        command=command,
    )
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return int(value["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
