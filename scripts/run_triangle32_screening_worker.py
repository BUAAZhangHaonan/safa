#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
from typing import Sequence

from safa.closeout.canonical_screening import CanonicalScreeningError, load_json
from safa.closeout.canonical_screening_worker import (
    execute_triangle32_request,
    validate_triangle32_request,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one direct generation-only R10 triangle32 worker request."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--gpu-uuid", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request_path = args.request.resolve()
    request = validate_triangle32_request(
        load_json(request_path, "triangle32 worker request")
    )
    log_path = Path(str(request["log_path"])).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise CanonicalScreeningError(
            f"triangle32 log path already exists: {log_path}"
        )
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            print(
                json.dumps(
                    {
                        "event": "triangle32_worker_start",
                        "request": str(request_path),
                        "candidate_id": request["candidate"]["candidate_id"],
                        "gpu_index": args.gpu_index,
                        "gpu_uuid": args.gpu_uuid,
                        "retry_count": 0,
                        "generation_only": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            execute_triangle32_request(
                request_path, args.gpu_index, args.gpu_uuid
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
