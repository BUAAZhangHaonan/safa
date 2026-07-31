#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARATION = REPO_ROOT / "artifacts/r13_control_lpl_training/preparation_v1"
EXPECTED_UUIDS = {
    0: "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    1: "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
    2: "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02",
    3: "GPU-61ea2925-9905-7f56-cd64-7a792a32efef",
}


class R13ResourceAdmissionError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R13ResourceAdmissionError(f"JSON payload is not an object: {path}")
    return value


def _gpu_rows() -> dict[int, dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise R13ResourceAdmissionError(f"unexpected nvidia-smi row: {line!r}")
        index = int(fields[0])
        rows[index] = {
            "uuid": fields[1],
            "total_bytes": int(fields[2]) * 1024**2,
            "used_bytes": int(fields[3]) * 1024**2,
            "utilization_percent": float(fields[4]),
        }
    return rows


def _peak_bytes(path: Path | None, arms: tuple[str, ...]) -> dict[str, int]:
    if path is None:
        return {arm: 0 for arm in arms}
    payload = _read_json(path)
    if payload.get("contract_type") != "safa_r13_probe_peak_memory_v1":
        raise R13ResourceAdmissionError("peak probe result contract differs")
    values = payload.get("peak_bytes")
    if not isinstance(values, Mapping):
        raise R13ResourceAdmissionError("peak probe result lacks peak_bytes mapping")
    result = {}
    for arm in arms:
        value = values.get(arm)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise R13ResourceAdmissionError(f"peak probe result for {arm} is invalid")
        result[arm] = value
    return result


def validate(preparation: Path, *, mode: str, peak_results: Path | None) -> dict[str, Any]:
    contract = _read_json(preparation / "resource_contract.json")
    if contract.get("contract_type") != "safa_r13_resource_admission_v1":
        raise R13ResourceAdmissionError("resource contract identity differs")
    ledger_name = "probe_ledger.json" if mode == "probe" else "training_ledger.json"
    ledger = _read_json(preparation / ledger_name)
    jobs = ledger.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2:
        raise R13ResourceAdmissionError(f"{mode} ledger must contain exactly two jobs")
    binding_key = "probe_gpu_bindings" if mode == "probe" else "training_gpu_bindings"
    expected_bindings = {"control": 2, "lpl": 1} if mode == "probe" else {"control": 0, "lpl": 1}
    if contract.get(binding_key) != expected_bindings or contract.get("probe_and_training_are_sequential") is not True:
        raise R13ResourceAdmissionError(f"{mode} resource binding contract differs")
    ledger_bindings = {str(job["arm_id"]): int(job["physical_gpu"]["index"]) for job in jobs}
    if ledger_bindings != expected_bindings:
        raise R13ResourceAdmissionError(f"{mode} ledger resource bindings differ")
    arms = tuple(str(job["arm_id"]) for job in jobs)
    peaks = _peak_bytes(peak_results, arms)
    if mode == "training" and peak_results is None:
        raise R13ResourceAdmissionError("training admission requires measured 8-step peak results")

    swap_before = psutil.swap_memory()
    sample_seconds = float(contract.get("swap_io_sample_seconds", 0.0))
    if sample_seconds <= 0.0:
        raise R13ResourceAdmissionError("swap I/O sample duration must be positive")
    cpu = float(psutil.cpu_percent(interval=min(0.2, sample_seconds)))
    remaining = sample_seconds - min(0.2, sample_seconds)
    if remaining > 0.0:
        time.sleep(remaining)
    swap_after = psutil.swap_memory()
    ram = float(psutil.virtual_memory().percent)
    swap = float(swap_after.percent)
    if cpu >= float(contract["max_cpu_percent"]):
        raise R13ResourceAdmissionError(f"CPU utilization is too high: {cpu}")
    if ram >= float(contract["max_ram_percent"]):
        raise R13ResourceAdmissionError(f"RAM utilization is too high: {ram}")
    if contract.get("swap_policy") != "observe_only_main_memory_is_the_admission_gate":
        raise R13ResourceAdmissionError("swap observation policy differs")
    swap_in_delta = int(swap_after.sin) - int(swap_before.sin)
    swap_out_delta = int(swap_after.sout) - int(swap_before.sout)
    maximum_swap_delta = int(contract.get("maximum_swap_in_out_delta_bytes", -1))
    if swap_in_delta > maximum_swap_delta or swap_out_delta > maximum_swap_delta:
        raise R13ResourceAdmissionError(
            f"active swap I/O detected: swap_in_delta={swap_in_delta} swap_out_delta={swap_out_delta}"
        )
    disk = shutil.disk_usage(REPO_ROOT)
    if disk.free < int(contract["minimum_repo_filesystem_free_bytes"]):
        raise R13ResourceAdmissionError(f"repo filesystem free bytes are too low: {disk.free}")

    gpu_rows = _gpu_rows()
    admitted = []
    for job in jobs:
        index = int(job["physical_gpu"]["index"])
        arm = str(job["arm_id"])
        row = gpu_rows.get(index)
        if row is None or row["uuid"] != EXPECTED_UUIDS[index] or job["physical_gpu"]["uuid"] != EXPECTED_UUIDS[index] or job.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != EXPECTED_UUIDS[index]:
            raise R13ResourceAdmissionError(f"GPU identity differs for physical GPU {index}")
        predicted_used = int(row["used_bytes"]) + peaks[arm]
        predicted_percent = 100.0 * predicted_used / int(row["total_bytes"])
        predicted_free = int(row["total_bytes"]) - predicted_used
        if predicted_percent >= float(contract["max_gpu_memory_percent_after_launch"]):
            raise R13ResourceAdmissionError(f"GPU {index} predicted memory percent is too high: {predicted_percent}")
        if predicted_free < int(contract["minimum_gpu_free_bytes_after_launch"]):
            raise R13ResourceAdmissionError(f"GPU {index} predicted free memory is too low: {predicted_free}")
        admitted.append({"index": index, "arm_id": arm, **row, "predicted_peak_bytes": peaks[arm], "predicted_used_percent": predicted_percent, "predicted_free_bytes": predicted_free})

    sessions = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], text=True, capture_output=True)
    active_sessions = set(sessions.stdout.splitlines()) if sessions.returncode == 0 else set()
    conflicts = sorted(str(job["tmux_session"]) for job in jobs if str(job["tmux_session"]) in active_sessions)
    if conflicts:
        raise R13ResourceAdmissionError(f"R13 tmux sessions already exist: {conflicts}")
    command_lines = [" ".join(process.info["cmdline"] or []) for process in psutil.process_iter(["cmdline"])]
    running = [str(job["config"]) for job in jobs if any(str(job["config"]) in command for command in command_lines)]
    if running:
        raise R13ResourceAdmissionError(f"R13 jobs already run for configs: {running}")
    existing_outputs = [str(job["output_root"]) for job in jobs if (REPO_ROOT / str(job["output_root"])).exists()]
    if existing_outputs:
        raise R13ResourceAdmissionError(f"R13 output roots must be fresh: {existing_outputs}")
    return {
        "contract_type": "safa_r13_resource_admission_result_v1",
        "status": "admitted",
        "mode": mode,
        "cpu_percent": cpu,
        "ram_percent": ram,
        "swap_percent": swap,
        "swap_in_delta_bytes": swap_in_delta,
        "swap_out_delta_bytes": swap_out_delta,
        "disk_free_bytes": disk.free,
        "gpus": admitted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate live R13 resources without launching work.")
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--mode", choices=("probe", "training"), required=True)
    parser.add_argument("--peak-results", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.preparation.resolve(), mode=args.mode, peak_results=None if args.peak_results is None else args.peak_results.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
