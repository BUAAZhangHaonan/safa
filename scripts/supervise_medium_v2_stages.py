#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from scripts.monitor_medium_v2_m2 import (
    CommandResult,
    capture_tmux,
    check_tmux,
    latest_quality_metrics,
    list_processes,
    parse_live_progress,
    query_gpus,
    read_error_signatures,
    read_json,
    summarize_metrics,
    write_json_atomic,
)
from scripts.plot_m2_m3_curves import plot_m2_m3_curves


DEFAULT_STATUS = Path("artifacts/monitor/medium_v2_stage_supervisor_status.json")
DEFAULT_EVENTS = Path("artifacts/monitor/medium_v2_stage_supervisor_events.jsonl")
DEFAULT_LOG = Path("artifacts/monitor/medium_v2_stage_supervisor.log")
DEFAULT_REPORT_DIR = Path("artifacts/monitor")
DEFAULT_DRAFT_DIR = Path("docs/experiments/drafts")
DEFAULT_PLOT_DIR = Path("artifacts/plots/medium_v2")
DEFAULT_MARKER = Path("artifacts/monitor/medium_v2_start_m3.approved")
DEFAULT_PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
DEFAULT_GPUS = "3,4,5,6"
DEFAULT_REQUIRED_EPOCHS = 120

M2_NAME = "m2"
M3_NAME = "m3"
M2_RUN = Path("artifacts/checkpoints/g_medium_v2_stage2_m2_gram_weighted")
M3_RUN = Path("artifacts/checkpoints/g_medium_v2_stage2_m3_gram_projected")
M2_DOC = "MEDIUM_V2_STAGE2_M2_GRAM_WEIGHTED_DRAFT.md"
M3_DOC = "MEDIUM_V2_STAGE2_M3_GRAM_PROJECTED_DRAFT.md"
COMPARISON_DOC = "MEDIUM_V2_M0_M2_M3_COMPARISON_DRAFT.md"


class StageSupervisorPaths(NamedTuple):
    status: Path = DEFAULT_STATUS
    events: Path = DEFAULT_EVENTS
    log: Path = DEFAULT_LOG
    report_dir: Path = DEFAULT_REPORT_DIR
    draft_dir: Path = DEFAULT_DRAFT_DIR
    plot_dir: Path = DEFAULT_PLOT_DIR
    marker: Path = DEFAULT_MARKER
    m2_metrics: Path = M2_RUN / "last_metrics.json"
    m2_history: Path = M2_RUN / "history.json"
    m2_quality_dir: Path = Path("artifacts/eval/g_medium_v2_stage2_m2_gram_weighted/quality")
    m2_train_log: Path = Path("artifacts/logs/train_g_medium_v2_stage2_m2_gram_weighted_gpu3_6.log")
    m2_last_checkpoint: Path = M2_RUN / "last.pt"
    m2_best_checkpoint: Path = M2_RUN / "best_raw_utility.pt"
    m3_metrics: Path = M3_RUN / "last_metrics.json"
    m3_history: Path = M3_RUN / "history.json"
    m3_quality_dir: Path = Path("artifacts/eval/g_medium_v2_stage2_m3_gram_projected/quality")
    m3_train_log: Path = Path("artifacts/logs/train_g_medium_v2_stage2_m3_gram_projected_gpu3_6.log")
    m3_last_checkpoint: Path = M3_RUN / "last.pt"
    m3_best_checkpoint: Path = M3_RUN / "best_raw_utility.pt"
    privacy_skip_json: Path = DEFAULT_REPORT_DIR / "medium_v2_m3_privacy_skipped.json"


class StageSupervisorConfig(NamedTuple):
    m2_session: str = "train_g_medium_v2_stage2_m2_gram_weighted_gpu3_6"
    m3_session: str = "train_g_medium_v2_stage2_m3_gram_projected_gpu3_6"
    m2_process_pattern: str = "safa.cli.train_g --config configs/medium_v2/train_g_medium_v2_stage2_m2_gram_weighted.yaml"
    m3_process_pattern: str = "safa.cli.train_g --config configs/medium_v2/train_g_medium_v2_stage2_m3_gram_projected.yaml"
    m3_config: str = "configs/medium_v2/train_g_medium_v2_stage2_m3_gram_projected.yaml"
    gpu_visible_devices: str = DEFAULT_GPUS
    python_bin: str = DEFAULT_PYTHON
    required_epochs: int = DEFAULT_REQUIRED_EPOCHS
    run_completion_actions: bool = True
    allow_launch: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_command(command: Sequence[str]) -> CommandResult:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


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
            detail = row.get("summary") or row.get("reason") or row.get("path") or ""
            handle.write(f"{row['time']} {row['type']}: {detail}\n")


def _stage_paths(paths: StageSupervisorPaths, stage: str) -> dict[str, Path]:
    prefix = "m2" if stage == M2_NAME else "m3"
    return {
        "metrics": getattr(paths, f"{prefix}_metrics"),
        "history": getattr(paths, f"{prefix}_history"),
        "quality_dir": getattr(paths, f"{prefix}_quality_dir"),
        "train_log": getattr(paths, f"{prefix}_train_log"),
        "last_checkpoint": getattr(paths, f"{prefix}_last_checkpoint"),
        "best_checkpoint": getattr(paths, f"{prefix}_best_checkpoint"),
    }


def _merge_epoch(metric_epoch: dict[str, Any], live_progress: dict[str, Any]) -> dict[str, Any]:
    if metric_epoch:
        result = dict(metric_epoch)
        result["source"] = "metrics"
        return result
    if live_progress:
        result = dict(live_progress)
        result["source"] = "tmux_capture"
        return result
    return {}


def _stage_status(
    stage: str,
    paths: StageSupervisorPaths,
    config: StageSupervisorConfig,
    command_runner: Callable[[Sequence[str]], CommandResult],
    previous_status: dict[str, Any],
) -> dict[str, Any]:
    stage_paths = _stage_paths(paths, stage)
    session = config.m2_session if stage == M2_NAME else config.m3_session
    pattern = config.m2_process_pattern if stage == M2_NAME else config.m3_process_pattern
    tmux_alive = check_tmux(session, command_runner)
    tmux_capture = capture_tmux(session, command_runner) if tmux_alive else ""
    processes = list_processes(pattern, command_runner)
    metrics = read_json(stage_paths["metrics"])
    metric_summary = summarize_metrics(metrics if isinstance(metrics, dict) else {})
    errors, log_offset = read_error_signatures(stage_paths["train_log"], previous_status)
    latest_epoch = _merge_epoch(metric_summary.get("latest_epoch", {}), parse_live_progress(tmux_capture))
    epoch = latest_epoch.get("stage_epoch_1based")
    complete = isinstance(epoch, int) and epoch >= config.required_epochs
    checkpoint_ready = stage_paths["last_checkpoint"].exists() and stage_paths["best_checkpoint"].exists()
    return {
        "stage": stage,
        "tmux_alive": tmux_alive,
        "process_alive": bool(processes),
        "process_count": len(processes),
        "latest_epoch": latest_epoch,
        "latest_validation": metric_summary.get("validation", {}),
        "latest_quality": latest_quality_metrics(stage_paths["quality_dir"]),
        "training_metrics": metric_summary.get("training", {}),
        "error_signatures": errors,
        "train_log_offset": log_offset,
        "complete": complete,
        "checkpoint_ready": checkpoint_ready,
        "paths": {key: value.as_posix() for key, value in stage_paths.items()},
    }


def _report_nodes(stage_status: dict[str, Any], required_epochs: int) -> list[tuple[str, int | None]]:
    epoch = stage_status.get("latest_epoch", {}).get("stage_epoch_1based")
    if not isinstance(epoch, int):
        return []
    nodes: list[tuple[str, int | None]] = []
    if epoch >= 1:
        nodes.append(("epoch_0001", 1))
    for node_epoch in range(20, min(epoch, required_epochs) + 1, 20):
        nodes.append((f"epoch_{node_epoch:04d}", node_epoch))
    if stage_status.get("complete"):
        nodes.append(("complete", None))
    return nodes


def _format_report_md(payload: dict[str, Any]) -> str:
    status = payload["status"]
    lines = [
        f"# Medium V2 {payload['stage'].upper()} {payload['node']} Report",
        "",
        f"- Time: {payload['time']}",
        f"- Epoch: {status.get('latest_epoch', {})}",
        f"- Complete: {status.get('complete')}",
        f"- Checkpoint ready: {status.get('checkpoint_ready')}",
        f"- Validation: {status.get('latest_validation', {})}",
        f"- Quality: {status.get('latest_quality', {})}",
        f"- Error signatures: {len(status.get('error_signatures', []))}",
    ]
    if status.get("error_signatures"):
        lines.append("")
        lines.append("## Error Signatures")
        for line in status["error_signatures"][-10:]:
            lines.append(f"- `{line}`")
    return "\n".join(lines) + "\n"


def write_stage_reports(
    stage_status: dict[str, Any],
    paths: StageSupervisorPaths,
    config: StageSupervisorConfig,
    now: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    stage = stage_status["stage"]
    for node, _ in _report_nodes(stage_status, config.required_epochs):
        stem = f"medium_v2_{stage}_{node}_report"
        json_path = paths.report_dir / f"{stem}.json"
        md_path = paths.report_dir / f"{stem}.md"
        if json_path.exists() and md_path.exists():
            continue
        payload = {"time": now, "stage": stage, "node": node, "status": stage_status}
        write_json_atomic(json_path, payload)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_format_report_md(payload), encoding="utf-8")
        events.append({"time": now, "type": "stage_report", "stage": stage, "path": json_path.as_posix(), "summary": stem})
    return events


def _metric_row(metrics: dict[str, Any]) -> str:
    keys = (
        "loss",
        "flow_loss_raw",
        "cycle_loss_raw",
        "repr_point_loss",
        "repr_relation_loss",
        "repr_loss",
        "validation_raw_latent_cosine_mean",
        "validation_raw_source_prediction_preserved",
        "validation_raw_single_face_eq1_rate",
    )
    return "\n".join(f"- {key}: {metrics.get(key, 'pending')}" for key in keys)


def write_stage_draft(stage: str, paths: StageSupervisorPaths, now: str) -> Path:
    stage_paths = _stage_paths(paths, stage)
    metrics = read_json(stage_paths["metrics"])
    if not isinstance(metrics, dict):
        metrics = {}
    quality = latest_quality_metrics(stage_paths["quality_dir"])
    doc_name = M2_DOC if stage == M2_NAME else M3_DOC
    curve_names = ["m2_curves.png"] if stage == M2_NAME else ["m3_curves.png", "m3_projection_diagnostics.png"]
    lines = [
        f"# Medium V2 {stage.upper()} Draft",
        "",
        f"Updated: {now}",
        "",
        "## Artifacts",
        "",
        f"- Checkpoint dir: {Path(stage_paths['metrics']).parent.as_posix()}",
        f"- Metrics JSON: {stage_paths['metrics'].as_posix()}",
        f"- History JSON: {stage_paths['history'].as_posix()}",
        f"- Quality dir: {stage_paths['quality_dir'].as_posix()}",
        f"- Curves: {', '.join(str(paths.plot_dir / name) for name in curve_names)}",
        "",
        "## Latest Metrics",
        "",
        _metric_row(metrics),
        "",
        "## Latest Quality",
        "",
        json.dumps(quality, indent=2, sort_keys=True),
        "",
        "Privacy eval remains pending unless the formal guard passes.",
    ]
    output = paths.draft_dir / doc_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_comparison_draft(paths: StageSupervisorPaths, now: str) -> Path:
    output = paths.draft_dir / COMPARISON_DOC
    lines = [
        "# Medium V2 M0/M2/M3 Comparison Draft",
        "",
        f"Updated: {now}",
        "",
        "M2 and M3 can be compared only from generated curves, completed history JSON, quality JSON, and any formal privacy eval that passed its guard.",
        "",
        "## Curves",
        "",
        f"- M2: {paths.plot_dir / 'm2_curves.png'}",
        f"- M3: {paths.plot_dir / 'm3_curves.png'}",
        f"- Projection: {paths.plot_dir / 'm3_projection_diagnostics.png'}",
        f"- Comparison: {paths.plot_dir / 'm0_m2_m3_comparison.png'}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def completion_actions(stage: str, paths: StageSupervisorPaths, now: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        outputs = plot_m2_m3_curves(
            m0_run=Path("artifacts/checkpoints/g_medium_v1_stage2_m0"),
            m2_run=M2_RUN,
            m3_run=M3_RUN,
            out_dir=paths.plot_dir,
            only=stage,
            m2_history_json=paths.m2_history,
            m3_history_json=paths.m3_history,
            m2_quality_dir=paths.m2_quality_dir,
            m3_quality_dir=paths.m3_quality_dir,
        )
        events.append({"time": now, "type": "curves_ready", "stage": stage, "summary": ", ".join(path.as_posix() for path in outputs)})
    except Exception as exc:
        events.append({"time": now, "type": "curves_failed", "stage": stage, "reason": str(exc), "summary": str(exc)})
        return events

    draft = write_stage_draft(stage, paths, now)
    events.append({"time": now, "type": "draft_ready", "stage": stage, "path": draft.as_posix(), "summary": draft.as_posix()})
    if stage == M3_NAME:
        comparison = write_comparison_draft(paths, now)
        events.append({"time": now, "type": "comparison_draft_ready", "path": comparison.as_posix(), "summary": comparison.as_posix()})
    return events


def docs_and_curves_ready(paths: StageSupervisorPaths) -> bool:
    return (paths.plot_dir / "m2_curves.png").is_file() and (paths.draft_dir / M2_DOC).is_file()


def build_m3_command(paths: StageSupervisorPaths, config: StageSupervisorConfig) -> str:
    log_path = paths.m3_train_log.as_posix()
    return (
        f"cd {Path.cwd().as_posix()} && "
        f"mkdir -p {Path(log_path).parent.as_posix()} && "
        f"CUDA_VISIBLE_DEVICES={config.gpu_visible_devices} "
        "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HTTP_PROXY= HTTPS_PROXY= PYTHONPATH=src "
        f"{config.python_bin} scripts/guarded_run.py --max-ram-fraction 0.90 -- "
        f"{config.python_bin} -m torch.distributed.run --standalone --nproc_per_node=4 "
        f"-m safa.cli.train_g --config {config.m3_config} 2>&1 | tee {log_path}"
    )


def maybe_start_m3(
    paths: StageSupervisorPaths,
    config: StageSupervisorConfig,
    m2_status: dict[str, Any],
    m3_status: dict[str, Any],
    command_runner: Callable[[Sequence[str]], CommandResult],
    now: str,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    ready = bool(m2_status.get("complete")) and bool(m2_status.get("checkpoint_ready")) and docs_and_curves_ready(paths)
    marker_present = paths.marker.is_file()
    gate = {
        "marker": paths.marker.as_posix(),
        "marker_present": marker_present,
        "ready": ready,
        "m2_complete": bool(m2_status.get("complete")),
        "m2_checkpoint_ready": bool(m2_status.get("checkpoint_ready")),
        "m2_docs_curves_ready": docs_and_curves_ready(paths),
    }
    events: list[dict[str, Any]] = []
    if m3_status.get("tmux_alive") or m3_status.get("process_alive"):
        return "m3_running", gate, events
    if m3_status.get("checkpoint_ready"):
        return "m3_exists", gate, events
    if not ready:
        return "waiting_m2", gate, events
    if not marker_present:
        return "waiting_m3_approval", gate, events
    if not config.allow_launch:
        gate["launch_suppressed"] = True
        return "m3_ready_launch_suppressed", gate, events

    launch_command = build_m3_command(paths, config)
    launch = command_runner(("tmux", "new-session", "-d", "-s", config.m3_session, launch_command))
    gate["launch_command"] = launch_command
    if launch.returncode == 0:
        events.append({"time": now, "type": "m3_started", "summary": f"started {config.m3_session} on GPU {config.gpu_visible_devices}"})
        return "m3_started", gate, events
    gate["launch_stderr"] = launch.stderr[-1000:]
    events.append({"time": now, "type": "error", "reason": "m3_launch_failed", "summary": launch.stderr[-500:] or launch.stdout[-500:]})
    return "blocked", gate, events


def privacy_guard_pass(metrics: dict[str, Any]) -> bool:
    value = metrics.get("privacy_guard_pass")
    if isinstance(value, bool):
        return value
    validation = summarize_metrics(metrics).get("validation", {})
    cosine = validation.get("cosine")
    single_face = validation.get("single_face")
    return isinstance(cosine, float) and cosine >= 0.95 and isinstance(single_face, float) and single_face >= 1.0


def write_privacy_skip_if_needed(paths: StageSupervisorPaths, m3_status: dict[str, Any], now: str) -> list[dict[str, Any]]:
    if not m3_status.get("complete") or not m3_status.get("checkpoint_ready"):
        return []
    metrics = read_json(paths.m3_metrics)
    if isinstance(metrics, dict) and privacy_guard_pass(metrics):
        return []
    payload = {
        "time": now,
        "stage": "m3",
        "privacy_skipped": True,
        "reason": "privacy_guard_not_passed_or_missing",
        "metrics": metrics if isinstance(metrics, dict) else {},
    }
    write_json_atomic(paths.privacy_skip_json, payload)
    return [{"time": now, "type": "privacy_skipped", "path": paths.privacy_skip_json.as_posix(), "summary": payload["reason"]}]


def run_once(
    paths: StageSupervisorPaths,
    config: StageSupervisorConfig,
    command_runner: Callable[[Sequence[str]], CommandResult] = run_command,
) -> int:
    now = utc_now()
    previous_status = read_json(paths.status)
    if not isinstance(previous_status, dict):
        previous_status = {}
    m2_status = _stage_status(M2_NAME, paths, config, command_runner, previous_status.get("m2", {}))
    m3_status = _stage_status(M3_NAME, paths, config, command_runner, previous_status.get("m3", {}))
    events = []
    events.extend(write_stage_reports(m2_status, paths, config, now))
    events.extend(write_stage_reports(m3_status, paths, config, now))
    if config.run_completion_actions:
        if m2_status.get("complete") and not docs_and_curves_ready(paths):
            events.extend(completion_actions(M2_NAME, paths, now))
        if m3_status.get("complete") and not (paths.plot_dir / "m3_curves.png").is_file():
            events.extend(completion_actions(M3_NAME, paths, now))
    events.extend(write_privacy_skip_if_needed(paths, m3_status, now))
    state, gate, gate_events = maybe_start_m3(paths, config, m2_status, m3_status, command_runner, now)
    events.extend(gate_events)
    status = {
        "time": now,
        "state": state,
        "m2": m2_status,
        "m3": m3_status,
        "m3_gate": gate,
        "gpus": query_gpus(tuple(int(part) for part in config.gpu_visible_devices.split(",") if part.strip()), command_runner),
        "paths": {
            "status": paths.status.as_posix(),
            "events": paths.events.as_posix(),
            "log": paths.log.as_posix(),
            "report_dir": paths.report_dir.as_posix(),
            "draft_dir": paths.draft_dir.as_posix(),
            "plot_dir": paths.plot_dir.as_posix(),
        },
        "new_event_count": len(events),
        "new_event_types": [event["type"] for event in events],
    }
    write_json_atomic(paths.status, status)
    append_jsonl(paths.events, events)
    append_log(paths.log, events)
    return 0


def loop(paths: StageSupervisorPaths, config: StageSupervisorConfig, interval_seconds: int) -> int:
    while True:
        run_once(paths, config)
        time.sleep(interval_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise medium_v2 M2/M3 reports and gated M3 launch.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--no-launch", action="store_true", help="monitor and report, but never launch M3")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--draft-dir", type=Path, default=DEFAULT_DRAFT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--required-epochs", type=int, default=DEFAULT_REQUIRED_EPOCHS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = StageSupervisorPaths(
        status=args.status,
        events=args.events,
        log=args.log,
        report_dir=args.report_dir,
        draft_dir=args.draft_dir,
        plot_dir=args.plot_dir,
        marker=args.marker,
    )
    config = StageSupervisorConfig(
        gpu_visible_devices=args.gpus,
        python_bin=args.python_bin,
        required_epochs=args.required_epochs,
        allow_launch=not args.no_launch,
    )
    if args.once:
        return run_once(paths, config)
    return loop(paths, config, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
