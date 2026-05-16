#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _memory_used_fraction() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        values[key] = int(raw_value.strip().split()[0])
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None or total <= 0:
        raise RuntimeError("Cannot read MemTotal and MemAvailable from /proc/meminfo")
    return 1.0 - (available / total)


def _terminate(process: subprocess.Popen, reason: str) -> int:
    print(reason, file=sys.stderr, flush=True)
    if process.poll() is not None:
        return int(process.returncode)
    process.terminate()
    try:
        return int(process.wait(timeout=30))
    except subprocess.TimeoutExpired:
        process.kill()
        return int(process.wait())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command with a hard Linux RAM guard.")
    parser.add_argument("--max-ram-fraction", type=float, default=0.90)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("No command provided")
    if args.command[0] == "--":
        args.command = args.command[1:]
    if not (0.0 < args.max_ram_fraction < 1.0):
        raise SystemExit("--max-ram-fraction must be in (0,1)")
    used = _memory_used_fraction()
    if used >= args.max_ram_fraction:
        raise SystemExit(f"RAM guard refused to start: used_fraction={used:.4f} threshold={args.max_ram_fraction:.4f}")
    process = subprocess.Popen(args.command, start_new_session=True)
    try:
        while process.poll() is None:
            time.sleep(args.poll_seconds)
            used = _memory_used_fraction()
            if used >= args.max_ram_fraction:
                code = _terminate(
                    process,
                    f"RAM guard stopped command: used_fraction={used:.4f} threshold={args.max_ram_fraction:.4f}",
                )
                raise SystemExit(code if code != 0 else 137)
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        raise
    raise SystemExit(int(process.returncode))


if __name__ == "__main__":
    main()
