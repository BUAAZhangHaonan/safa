#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _verified_config_binding(
    job: Mapping[str, Any], job_id: str
) -> dict[str, str]:
    binding = job.get("config_binding")
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{job_id} config_binding fields differ")
    path_text = binding.get("path")
    declared_sha256 = binding.get("sha256")
    if not isinstance(path_text, str) or not path_text:
        raise ValueError(f"{job_id} config path is invalid")
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
    ):
        raise ValueError(f"{job_id} config SHA256 is invalid")
    path = _resolve(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"{job_id} config is missing: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != declared_sha256:
        raise ValueError(f"{job_id} config SHA256 differs")
    argv = job.get("argv")
    if job.get("job_type") == "generation":
        config_positions = [
            index for index, token in enumerate(argv) if token == "--config"
        ]
        if (
            len(config_positions) != 1
            or config_positions[0] + 1 >= len(argv)
            or _resolve(argv[config_positions[0] + 1]).resolve()
            != path.resolve()
        ):
            raise ValueError(f"{job_id} argv config differs from config_binding")
    return {"path": path_text, "sha256": actual_sha256}


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"attempt receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _select_job(ledger: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    if (
        ledger.get("schema_version") != 1
        or ledger.get("contract_type")
        != "safa_r11_schedule_nfe2_launch_ledger_v1"
        or ledger.get("status") != "prepared_not_launched"
        or ledger.get("retry_limit") != 0
    ):
        raise ValueError("NFE2 launch ledger identity differs")
    jobs = ledger.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("NFE2 launch ledger jobs must be a list")
    matches = [
        job
        for job in jobs
        if isinstance(job, Mapping) and job.get("job_id") == job_id
    ]
    if len(matches) != 1:
        raise ValueError(f"ledger must contain exactly one job {job_id!r}")
    job = matches[0]
    if job.get("retry_limit") != 0:
        raise ValueError(f"{job_id} retry_limit must be 0")
    if job.get("attempt_limit") != 1:
        raise ValueError(f"{job_id} attempt_limit must be 1")
    argv = job.get("argv")
    if (
        not isinstance(argv, list)
        or any(not isinstance(token, str) or not token for token in argv)
        or not argv
        or argv[0] != PYTHON
    ):
        raise ValueError(f"{job_id} argv must use the locked interpreter")
    env = job.get("env")
    gpu = job.get("gpu")
    if (
        not isinstance(env, Mapping)
        or not isinstance(gpu, Mapping)
        or env.get("CUDA_VISIBLE_DEVICES") != gpu.get("uuid")
    ):
        raise ValueError(f"{job_id} CUDA UUID binding differs")
    return job


def run_job(ledger_path: Path, job_id: str) -> int:
    resolved_ledger = ledger_path.resolve()
    ledger = _json(resolved_ledger, "NFE2 launch ledger")
    job = _select_job(ledger, job_id)
    config_binding = _verified_config_binding(job, job_id)
    fresh_paths = job.get("fresh_output_paths")
    if not isinstance(fresh_paths, list) or any(
        not isinstance(path, str) or not path for path in fresh_paths
    ):
        raise ValueError(f"{job_id} fresh_output_paths must be a path list")
    existing = [path for path in fresh_paths if _resolve(path).exists()]
    if existing:
        raise FileExistsError(
            f"{job_id} refuses a repeated attempt; fresh paths exist: {existing!r}"
        )
    prerequisites = job.get("prerequisite_paths")
    if not isinstance(prerequisites, list):
        raise ValueError(f"{job_id} prerequisite_paths must be a list")
    missing = [
        path
        for path in prerequisites
        if not isinstance(path, str) or not _resolve(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{job_id} prerequisites are missing: {missing!r}"
        )
    log_path = _resolve(str(job["log_path"]))
    receipt_path = _resolve(str(job["attempt_receipt_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "contract_type": "safa_r11_schedule_nfe2_attempt_receipt_v1",
        "job_id": job_id,
        "attempt_count": 1,
        "retry_count": 0,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(resolved_ledger),
        "argv": list(job["argv"]),
        "config_binding": config_binding,
        "env": dict(job["env"]),
        "gpu": dict(job["gpu"]),
        "started_at_utc": started,
    }
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in job["env"].items()})
    with log_path.open("x", encoding="utf-8") as log_handle:
        try:
            completed = subprocess.run(
                list(job["argv"]),
                cwd=REPO,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        except OSError as exc:
            receipt.update(
                {
                    "status": "launch_failed",
                    "exit_status": None,
                    "finished_at_utc": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_receipt(receipt_path, receipt)
            return 3
    receipt["exit_status"] = completed.returncode
    receipt["finished_at_utc"] = _utc_now()
    if completed.returncode != 0:
        receipt["status"] = "failed"
        _write_receipt(receipt_path, receipt)
        return completed.returncode
    success_outputs = job.get("success_output_paths")
    if not isinstance(success_outputs, list) or any(
        not isinstance(path, str) or not _resolve(path).exists()
        for path in success_outputs
    ):
        receipt["status"] = "failed_output_contract"
        _write_receipt(receipt_path, receipt)
        return 4
    receipt["status"] = "complete"
    receipt["success_output_paths"] = list(success_outputs)
    _write_receipt(receipt_path, receipt)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute exactly one retry-0 R11 NFE2 ledger job."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_job(args.ledger, args.job_id)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"NFE2 ledger job refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
