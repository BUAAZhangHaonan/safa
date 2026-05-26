from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "supervise_medium_v1_stage2.py"
    spec = importlib.util.spec_from_file_location("supervise_medium_v1_stage2", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(module, tmp_path: Path):
    return module.SupervisorPaths(
        status=tmp_path / "status.json",
        events=tmp_path / "events.jsonl",
        log=tmp_path / "supervisor.log",
        m0_metrics=tmp_path / "m0" / "last_metrics.json",
        m0_last_checkpoint=tmp_path / "m0" / "last.pt",
        m0_best_checkpoint=tmp_path / "m0" / "best_raw_utility.pt",
        m1_last_checkpoint=tmp_path / "m1" / "last.pt",
        m1_best_checkpoint=tmp_path / "m1" / "best_raw_utility.pt",
        m0_train_log=tmp_path / "logs" / "m0.log",
    )


def _runner(module, alive_sessions: set[str], calls: list[tuple[str, ...]]):
    def fake_runner(command):
        calls.append(tuple(command))
        if command[:3] == ("tmux", "has-session", "-t"):
            return module.CommandResult(0 if command[3] in alive_sessions else 1, "", "")
        if command[:3] == ("tmux", "new-session", "-d"):
            alive_sessions.add(command[4])
            return module.CommandResult(0, "", "")
        return module.CommandResult(0, "", "")

    return fake_runner


def test_m0_incomplete_does_not_start_m1(tmp_path: Path) -> None:
    module = _load_script()
    paths = _paths(module, tmp_path)
    paths.m0_metrics.parent.mkdir(parents=True)
    paths.m0_metrics.write_text(json.dumps({"stage_epoch_1based": 199}), encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    exit_code = module.run_once(paths, module.SupervisorConfig(), _runner(module, {module.DEFAULT_M0_SESSION}, calls))

    assert exit_code == 0
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["state"] == "waiting_m0"
    assert not any(call[:3] == ("tmux", "new-session", "-d") for call in calls)


def test_m0_complete_without_required_checkpoints_blocks_without_launch(tmp_path: Path) -> None:
    module = _load_script()
    paths = _paths(module, tmp_path)
    paths.m0_metrics.parent.mkdir(parents=True)
    paths.m0_metrics.write_text(json.dumps({"stage_epoch_1based": 200}), encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    exit_code = module.run_once(paths, module.SupervisorConfig(), _runner(module, set(), calls))

    assert exit_code == 0
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["state"] == "blocked"
    assert status["reason"] == "m0_checkpoints_missing"
    assert not any(call[:3] == ("tmux", "new-session", "-d") for call in calls)
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "blocked"


def test_m1_existing_session_is_not_started_again(tmp_path: Path) -> None:
    module = _load_script()
    paths = _paths(module, tmp_path)
    paths.m0_metrics.parent.mkdir(parents=True)
    paths.m0_metrics.write_text(json.dumps({"stage_epoch_1based": 200}), encoding="utf-8")
    paths.m0_last_checkpoint.write_text("last", encoding="utf-8")
    paths.m0_best_checkpoint.write_text("best", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    exit_code = module.run_once(
        paths, module.SupervisorConfig(), _runner(module, {module.DEFAULT_M1_SESSION}, calls)
    )

    assert exit_code == 0
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["state"] == "m1_running"
    assert not any(call[:3] == ("tmux", "new-session", "-d") for call in calls)


def test_launch_command_uses_gpu3_6_and_m1_config(tmp_path: Path) -> None:
    module = _load_script()
    paths = _paths(module, tmp_path)
    paths.m0_metrics.parent.mkdir(parents=True)
    paths.m0_metrics.write_text(json.dumps({"stage_epoch_1based": 200}), encoding="utf-8")
    paths.m0_last_checkpoint.write_text("last", encoding="utf-8")
    paths.m0_best_checkpoint.write_text("best", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    exit_code = module.run_once(paths, module.SupervisorConfig(), _runner(module, set(), calls))

    assert exit_code == 0
    launch = next(call for call in calls if call[:3] == ("tmux", "new-session", "-d"))
    assert launch[4] == "train_g_medium_v1_stage2_m1_uw_gpu3_6"
    command_text = launch[5]
    assert "CUDA_VISIBLE_DEVICES=3,4,5,6" in command_text
    assert "configs/medium_v1/train_g_medium_v1_stage2_m1_uw.yaml" in command_text
    assert "torch.distributed.run --standalone --nproc_per_node=4" in command_text
