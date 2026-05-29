from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "monitor_medium_v2_m2.py"
    spec = importlib.util.spec_from_file_location("monitor_medium_v2_m2", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracts_live_epoch_from_tmux_capture_when_metrics_are_absent() -> None:
    module = _load_script()
    text = "train_g stage2 epoch=0:   7%| | 23/313 [04:19<38:07,  7.89s/it]"

    progress = module.parse_live_progress(text)

    assert progress == {"stage_epoch_0based": 0, "stage_epoch_1based": 1, "batch": 23, "total_batches": 313, "percent": 7.0}


def test_extracts_required_metrics_and_latest_quality_values(tmp_path: Path) -> None:
    module = _load_script()
    metrics = tmp_path / "last_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "stage_epoch_1based": 2,
                "validation_raw_latent_cosine_mean": 0.61,
                "validation_raw_single_face_eq1_rate": 0.91,
                "validation_raw_source_prediction_preserved": 0.73,
                "loss": 1.2,
            }
        ),
        encoding="utf-8",
    )
    quality_dir = tmp_path / "quality" / "epoch_0002"
    quality_dir.mkdir(parents=True)
    (quality_dir / "stage2_epoch_0002_raw_niqe.json").write_text(
        json.dumps({"iqa": {"method": "niqe", "mean": 4.2}}), encoding="utf-8"
    )
    (quality_dir / "stage2_epoch_0002_raw_distribution.json").write_text(
        json.dumps({"fid": 18.3, "kid_mean": 0.031, "kid_std": 0.002}), encoding="utf-8"
    )

    summary = module.summarize_metrics(module.read_json(metrics))
    quality = module.latest_quality_metrics(tmp_path / "quality")

    assert summary["latest_epoch"]["stage_epoch_1based"] == 2
    assert summary["validation"]["cosine"] == 0.61
    assert summary["validation"]["single_face"] == 0.91
    assert summary["validation"]["source_preserved"] == 0.73
    assert quality["niqe"] == 4.2
    assert quality["fid"] == 18.3
    assert quality["kid_mean"] == 0.031
    assert quality["kid_std"] == 0.002


def test_run_once_writes_required_status_and_event_paths(tmp_path: Path) -> None:
    module = _load_script()
    train_log = tmp_path / "train.log"
    train_log.write_text("ok\nRuntimeError: synthetic failure\n", encoding="utf-8")
    paths = module.MonitorPaths(
        status=tmp_path / "status.json",
        events=tmp_path / "events.jsonl",
        log=tmp_path / "events.log",
        metrics=tmp_path / "missing_metrics.json",
        history=tmp_path / "missing_history.json",
        quality_dir=tmp_path / "quality",
        train_log=train_log,
    )
    config = module.MonitorConfig(tmux_session="m2", process_pattern="train_m2", gpu_indices=(3,))

    def fake_runner(command):
        if command[:3] == ("tmux", "has-session", "-t"):
            return module.CommandResult(0, "", "")
        if command[:3] == ("tmux", "capture-pane", "-pt"):
            return module.CommandResult(0, "train_g stage2 epoch=0: 7%| | 23/313 [04:19<38:07, 7.89s/it]\n", "")
        if command[0] == "pgrep":
            return module.CommandResult(0, "123 python -m safa.cli.train_g --config configs/medium_v2/train_g_medium_v2_stage2_m2_gram_weighted.yaml\n", "")
        if command[0] == "nvidia-smi":
            return module.CommandResult(0, "3, 100, 2048, 24576\n", "")
        return module.CommandResult(1, "", "")

    exit_code = module.run_once(paths, config, command_runner=fake_runner)

    assert exit_code == 0
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["tmux_alive"] is True
    assert status["process_alive"] is True
    assert status["gpus"] == [{"index": 3, "utilization_gpu_pct": 100, "memory_used_mb": 2048, "memory_total_mb": 24576}]
    assert status["latest_epoch"]["stage_epoch_1based"] == 1
    assert status["latest_epoch"]["source"] == "tmux_capture"
    assert status["error_signatures"] == ["RuntimeError: synthetic failure"]
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["epoch_observed", "error_signature"]
    assert "error_signature" in paths.log.read_text(encoding="utf-8")
