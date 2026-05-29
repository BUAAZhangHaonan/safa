#!/usr/bin/env python3
"""Watch M2 quality epochs and create source/generated pair grids."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Sequence


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from scripts.visualize_epoch_quality_pairs import visualize_epoch_quality_pairs
except ModuleNotFoundError:
    from visualize_epoch_quality_pairs import visualize_epoch_quality_pairs


DEFAULT_QUALITY_DIR = Path("artifacts/eval/g_medium_v2_stage2_m2_gram_weighted/quality")
DEFAULT_INDEX = Path("data/index/val_single_face.jsonl")
DEFAULT_OUT_DIR = Path("artifacts/plots/medium_v2/m2")
DEFAULT_EVENTS = DEFAULT_OUT_DIR / "epoch_visuals_events.jsonl"
DEFAULT_LOG = DEFAULT_OUT_DIR / "epoch_visuals.log"
DEFAULT_STATE = DEFAULT_OUT_DIR / "epoch_visuals_state.json"


class WatcherPaths(NamedTuple):
    quality_dir: Path = DEFAULT_QUALITY_DIR
    index: Path = DEFAULT_INDEX
    out_dir: Path = DEFAULT_OUT_DIR
    events: Path = DEFAULT_EVENTS
    log: Path = DEFAULT_LOG
    state: Path = DEFAULT_STATE


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_read_error": f"invalid json: {exc}"}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            detail = row.get("out_path") or row.get("error") or row.get("epoch_dir") or ""
            handle.write(f"{row['time']} {row['type']}: {detail}\n")


def epoch_number(path: Path) -> int | None:
    match = re.fullmatch(r"epoch_(\d{4,})", path.name)
    if not match:
        return None
    return int(match.group(1))


def quality_epoch_dirs(quality_dir: Path) -> list[Path]:
    if not quality_dir.is_dir():
        return []
    dirs = [path for path in quality_dir.iterdir() if path.is_dir() and epoch_number(path) is not None]
    return sorted(dirs, key=lambda path: epoch_number(path) or -1)


def run_once(paths: WatcherPaths, num_samples: int = 16) -> int:
    now = utc_now()
    previous = read_json(paths.state)
    previous_state = previous if isinstance(previous, dict) else {}
    completed = dict(previous_state.get("completed", {})) if isinstance(previous_state.get("completed"), dict) else {}
    error_signatures = (
        dict(previous_state.get("error_signatures", {}))
        if isinstance(previous_state.get("error_signatures"), dict)
        else {}
    )
    events: list[dict[str, Any]] = []

    for epoch_dir in quality_epoch_dirs(paths.quality_dir):
        epoch = epoch_number(epoch_dir)
        if epoch is None:
            continue
        epoch_key = f"{epoch:04d}"
        out_path = paths.out_dir / f"epoch_{epoch:04d}_pairs.png"
        if epoch_key in completed and out_path.is_file():
            continue
        try:
            visualize_epoch_quality_pairs(
                quality_epoch_dir=epoch_dir,
                index=paths.index,
                out_path=out_path,
                num_samples=num_samples,
            )
        except Exception as exc:
            signature = f"{type(exc).__name__}: {exc}"
            if error_signatures.get(epoch_key) != signature:
                events.append(
                    {
                        "time": now,
                        "type": "epoch_visual_error",
                        "epoch": epoch,
                        "epoch_dir": epoch_dir.as_posix(),
                        "error": signature,
                    }
                )
            error_signatures[epoch_key] = signature
            continue

        completed[epoch_key] = out_path.as_posix()
        error_signatures.pop(epoch_key, None)
        events.append(
            {
                "time": now,
                "type": "epoch_visual_created",
                "epoch": epoch,
                "epoch_dir": epoch_dir.as_posix(),
                "out_path": out_path.as_posix(),
            }
        )

    state = {
        "time": now,
        "quality_dir": paths.quality_dir.as_posix(),
        "index": paths.index.as_posix(),
        "out_dir": paths.out_dir.as_posix(),
        "completed": completed,
        "error_signatures": error_signatures,
        "new_event_count": len(events),
    }
    write_json_atomic(paths.state, state)
    append_jsonl(paths.events, events)
    append_log(paths.log, events)
    return 0


def loop(paths: WatcherPaths, *, num_samples: int, interval_seconds: int) -> int:
    while True:
        run_once(paths, num_samples=num_samples)
        time.sleep(interval_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch medium v2 M2 quality epochs and create pair visuals.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--quality-dir", type=Path, default=DEFAULT_QUALITY_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--num-samples", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = WatcherPaths(
        quality_dir=args.quality_dir,
        index=args.index,
        out_dir=args.out_dir,
        events=args.events,
        log=args.log,
        state=args.state,
    )
    if args.once:
        return run_once(paths, num_samples=args.num_samples)
    return loop(paths, num_samples=args.num_samples, interval_seconds=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
