#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_OUTPUT = Path("artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_4_imagenet256.pt")
ZHUYU_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1oWt6tdm5WIeVaZnBuUVheKIG3cNDffl9?usp=drive_link"
HAOYI_CONVERTED_B4_FILE = "https://drive.google.com/file/d/1jcsM02gvPWe0IkXkZhDmLk6bIucsJG-b/view?usp=sharing"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or verify the external MeanFlow-SiT checkpoint used by e11.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default="", help="Direct file URL. The zhuyu Google Drive folder is not a direct file URL.")
    parser.add_argument("--sha256", default="", help="Expected SHA256. If omitted, the script only reports the observed hash.")
    parser.add_argument("--check-only", action="store_true", help="Only verify the target path; do not download.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output
    if output.is_file():
        observed = sha256_file(output)
        payload = {"status": "present", "path": str(output), "sha256": observed, "size_bytes": output.stat().st_size}
        print(json.dumps(payload, sort_keys=True))
        if args.sha256 and observed != args.sha256:
            print(f"SHA256 mismatch: expected {args.sha256}, got {observed}", file=sys.stderr)
            return 3
        return 0
    if args.check_only:
        print(json.dumps({"status": "missing", "path": str(output), "source_folder": ZHUYU_DRIVE_FOLDER, "backup_file": HAOYI_CONVERTED_B4_FILE}, sort_keys=True))
        return 2
    if not args.url:
        print(
            "No direct checkpoint URL was provided. Download the SiT-B/4 ImageNet256 checkpoint "
            f"from {ZHUYU_DRIVE_FOLDER}, or use {HAOYI_CONVERTED_B4_FILE}, or rerun with --url DIRECT_FILE_URL.",
            file=sys.stderr,
        )
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        urllib.request.urlretrieve(args.url, tmp_path)
        observed = sha256_file(tmp_path)
        if args.sha256 and observed != args.sha256:
            print(f"SHA256 mismatch: expected {args.sha256}, got {observed}", file=sys.stderr)
            tmp_path.unlink(missing_ok=True)
            return 3
        tmp_path.replace(output)
        print(json.dumps({"status": "downloaded", "path": str(output), "sha256": observed, "size_bytes": output.stat().st_size}, sort_keys=True))
        return 0
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
