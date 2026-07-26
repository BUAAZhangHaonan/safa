#!/usr/bin/env python3
"""Run one fail-closed generator checkpoint preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from safa.evaluation.checkpoint_preflight import preflight_generator_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-model",
        choices=("raw", "ema"),
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument("--out-json")
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow an unbound diagnostic preflight that is not formal evidence.",
    )
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args(argv)


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    complete = False
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            path.unlink(missing_ok=True)


def _require_sha256(value: str | None, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SystemExit(f"{label} must be a lowercase SHA256 digest")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.skip_sha256 and not args.diagnostic:
        raise SystemExit("--skip-sha256 is only allowed with --diagnostic")
    if not args.diagnostic and args.expected_sha256 is None:
        raise SystemExit("--expected-sha256 is required for formal preflight")
    expected_sha256 = (
        args.expected_sha256
        if args.diagnostic
        else _require_sha256(args.expected_sha256, "--expected-sha256")
    )
    result = preflight_generator_checkpoint(
        args.checkpoint,
        args.checkpoint_model,
        args.device,
        expected_checkpoint_sha256=expected_sha256,
        compute_sha256=not args.skip_sha256,
        smoke_samples=args.smoke_samples,
    )
    if not args.diagnostic and result["checkpoint_sha256"] is None:
        raise RuntimeError("formal preflight did not bind the expected checkpoint SHA256")
    if (
        not args.diagnostic
        and result["status"] == "valid"
        and result["sha256_binding"] != "expected_exact"
    ):
        raise RuntimeError("valid formal preflight lacks an exact checkpoint SHA256 binding")
    rendered = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    if args.out_json:
        _write_exclusive(Path(args.out_json), rendered)
    print(rendered.decode("utf-8"), end="")
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
