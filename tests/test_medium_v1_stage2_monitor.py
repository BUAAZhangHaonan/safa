from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "monitor_medium_v1_stage2.py"
    spec = importlib.util.spec_from_file_location("monitor_medium_v1_stage2", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_events_are_emitted_only_for_new_state() -> None:
    module = _load_script()
    previous = {
        "state": {
            "last_epoch_1based": 5,
            "seen_quality_jsons": ["old_fid.json"],
            "seen_error_signatures": ["RuntimeError: old"],
            "last_health": {"tmux_alive": True, "process_alive": True, "gpu_abnormal": False},
        }
    }
    snapshot = {
        "last_metrics": {"stage_epoch_1based": 6, "loss": 0.12, "quality_raw_niqe_mean": 4.5},
        "quality_jsons": ["old_fid.json", "new_kid.json"],
        "errors": ["RuntimeError: old", "CUDA out of memory at step 4"],
        "tmux_alive": True,
        "processes": [{"pid": 123, "cmd": "train_g"}],
        "gpus": [{"index": 3, "memory_used_mb": 1000, "memory_total_mb": 2000, "utilization_gpu_pct": 80}],
        "gpu_abnormal": False,
        "gpu_abnormal_reasons": [],
    }

    events, state = module.build_events(snapshot, previous, now="2026-05-26T00:00:00Z")

    assert [event["type"] for event in events] == ["epoch_completed", "quality_json", "error_keyword"]
    assert state["last_epoch_1based"] == 6
    assert state["seen_quality_jsons"] == ["old_fid.json", "new_kid.json"]
    assert "CUDA out of memory at step 4" in state["seen_error_signatures"]

    repeated_events, _ = module.build_events(snapshot, {"state": state}, now="2026-05-26T00:05:00Z")
    assert repeated_events == []


def test_single_shot_writes_status_and_only_event_summaries(tmp_path: Path) -> None:
    module = _load_script()
    metrics = tmp_path / "last_metrics.json"
    metrics.write_text(json.dumps({"stage_epoch_1based": 2, "loss": 0.2}), encoding="utf-8")
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    (quality_dir / "stage2_epoch_0002_fid_kid.json").write_text("{}", encoding="utf-8")
    log_path = tmp_path / "train.log"
    log_path.write_text("ok\nRuntimeError: synthetic failure\n", encoding="utf-8")

    paths = module.MonitorPaths(
        status=tmp_path / "status.json",
        events=tmp_path / "events.jsonl",
        log=tmp_path / "monitor.log",
        metrics=metrics,
        quality_dir=quality_dir,
        train_log=log_path,
    )
    config = module.MonitorConfig(
        tmux_session="unused",
        process_pattern="synthetic-train-pattern",
        gpu_indices=(3,),
        gpu_memory_high_ratio=0.98,
        gpu_memory_low_mb=0,
    )

    def fake_runner(command):
        if command[0] == "nvidia-smi":
            return module.CommandResult(0, "3, 1200, 2000, 80\n", "")
        return module.CommandResult(1, "", "")

    exit_code = module.run_once(paths=paths, config=config, command_runner=fake_runner)

    assert exit_code == 0
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["last_metrics"]["stage_epoch_1based"] == 2
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["epoch_completed", "quality_json", "error_keyword", "tmux_dead"]
    assert "epoch_completed" in paths.log.read_text(encoding="utf-8")


def test_single_shot_detects_nested_raw_distribution_quality_json(tmp_path: Path) -> None:
    module = _load_script()
    metrics = tmp_path / "last_metrics.json"
    metrics.write_text(json.dumps({"stage_epoch_1based": 21, "loss": 0.2}), encoding="utf-8")
    quality_dir = tmp_path / "quality"
    epoch_dir = quality_dir / "epoch_0020"
    epoch_dir.mkdir(parents=True)
    distribution_json = epoch_dir / "stage2_epoch_0020_raw_distribution.json"
    distribution_json.write_text(json.dumps({"fid": 12.3, "kid": 0.04}), encoding="utf-8")
    log_path = tmp_path / "train.log"
    log_path.write_text("ok\n", encoding="utf-8")

    paths = module.MonitorPaths(
        status=tmp_path / "status.json",
        events=tmp_path / "events.jsonl",
        log=tmp_path / "monitor.log",
        metrics=metrics,
        quality_dir=quality_dir,
        train_log=log_path,
    )
    config = module.MonitorConfig(
        tmux_session="unused",
        process_pattern="synthetic-train-pattern",
        gpu_indices=(3,),
        gpu_memory_high_ratio=0.98,
        gpu_memory_low_mb=0,
    )

    def fake_runner(command):
        if command[0] == "nvidia-smi":
            return module.CommandResult(0, "3, 1200, 2000, 80\n", "")
        return module.CommandResult(1, "", "")

    module.run_once(paths=paths, config=config, command_runner=fake_runner)

    status = json.loads(paths.status.read_text(encoding="utf-8"))
    distribution_path = distribution_json.as_posix()
    assert status["quality_json_count"] == 1
    assert status["latest_quality_jsons"] == [distribution_path]
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    quality_events = [event for event in events if event["type"] == "quality_json"]
    assert [event["path"] for event in quality_events] == [distribution_path]


def test_error_keyword_detection_does_not_match_nan_inside_paths_or_warnings(tmp_path: Path) -> None:
    module = _load_script()
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "/home/hdd3/zhanghaonan/file.py: FutureWarning: harmless warning\n"
        "loss became nan at step 7\n",
        encoding="utf-8",
    )

    errors, _ = module.read_new_error_lines(log_path, {})

    assert errors == ["loss became nan at step 7"]


def test_single_all_idle_sample_does_not_emit_gpu_abnormal_when_process_is_alive() -> None:
    module = _load_script()
    snapshot = {
        "last_metrics": {},
        "quality_jsons": [],
        "errors": [],
        "tmux_alive": True,
        "processes": [{"pid": 123, "cmd": "train_g"}],
        "gpus": [
            {"index": 3, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
            {"index": 4, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
            {"index": 5, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
            {"index": 6, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
        ],
    }
    previous = {"state": {"last_health": {"tmux_alive": True, "process_alive": True, "gpu_abnormal": False}}}
    config = module.MonitorConfig()

    module.apply_gpu_abnormal_debounce(snapshot, previous, config)
    events, state = module.build_events(snapshot, previous, now="2026-05-27T00:00:00Z")

    assert snapshot["gpu_abnormal"] is False
    assert "gpu3-6:all_idle" in snapshot["gpu_abnormal_observed_reasons"]
    assert [event["type"] for event in events] == []
    assert state["gpu_all_idle_consecutive_samples"] == 1


def test_repeated_all_idle_samples_emit_gpu_abnormal_when_process_is_alive() -> None:
    module = _load_script()
    snapshot = {
        "last_metrics": {},
        "quality_jsons": [],
        "errors": [],
        "tmux_alive": True,
        "processes": [{"pid": 123, "cmd": "train_g"}],
        "gpus": [
            {"index": 3, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
            {"index": 4, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
            {"index": 5, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
            {"index": 6, "memory_used_mb": 22000, "memory_total_mb": 24576, "memory_used_ratio": 0.895, "utilization_gpu_pct": 0},
        ],
    }
    previous = {
        "state": {
            "gpu_all_idle_consecutive_samples": 1,
            "last_health": {"tmux_alive": True, "process_alive": True, "gpu_abnormal": False},
        }
    }
    config = module.MonitorConfig(gpu_all_idle_min_samples=2)

    module.apply_gpu_abnormal_debounce(snapshot, previous, config)
    events, state = module.build_events(snapshot, previous, now="2026-05-27T00:05:00Z")

    assert snapshot["gpu_abnormal"] is True
    assert snapshot["gpu_abnormal_reasons"] == ["gpu3-6:all_idle"]
    assert [event["type"] for event in events] == ["gpu_abnormal"]
    assert state["gpu_all_idle_consecutive_samples"] == 2
