from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_gpu6_generator_queue.py"
    spec = importlib.util.spec_from_file_location("run_gpu6_generator_queue", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_default_plan_builds_expected_gpu6_paths_and_sessions() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--timestamp",
            "20260612_120000",
            "--repo-root",
            "/repo",
        ]
    )

    plan = module.build_queue_plan(args)

    assert plan.meanflow.session == "safa_meanflow_e9_gpu6_200ep_20260612_034831"
    assert plan.meanflow.log_path == Path("artifacts/logs/e9_meanflow_200ep_gpu6_20260612_034831_b4.log")
    assert plan.meanflow.checkpoints == (
        Path("artifacts/checkpoints/g_medium_v2_meanflow_200ep/best_stage2.pt"),
        Path("artifacts/checkpoints/g_medium_v2_meanflow_200ep/last.pt"),
    )
    assert plan.ddim.session == "safa_ddim_e10_gpu6_200ep_20260612_120000"
    assert plan.ddim.log_path == Path("artifacts/logs/e10_ddim_200ep_gpu6_20260612_120000.log")
    assert plan.queue_log == Path("artifacts/logs/gpu6_generator_queue_20260612_120000.log")
    assert plan.poll_seconds == 300
    assert plan.repo_root == Path("/repo")


def test_ddim_tmux_command_pins_gpu6_and_e10_config() -> None:
    module = _load_script()
    args = module.parse_args(["--timestamp", "20260612_120000", "--repo-root", "/repo"])
    plan = module.build_queue_plan(args)

    command = module.build_tmux_start_command(plan.ddim, plan)
    joined = " ".join(command)
    shell_command = command[-1]

    assert command[:5] == ["tmux", "new-session", "-d", "-s", "safa_ddim_e10_gpu6_200ep_20260612_120000"]
    assert "CUDA_VISIBLE_DEVICES=6" in shell_command
    assert "PYTHONPATH=src" in shell_command
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in shell_command
    assert "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m safa.cli.train_g" in shell_command
    assert "configs/medium_v2/experiments/e10_ddim_200ep.yaml" in shell_command
    assert "artifacts/logs/e10_ddim_200ep_gpu6_20260612_120000.log" in shell_command
    assert "CUDA_VISIBLE_DEVICES=0" not in joined


def test_report_command_uses_gpu6_environment_and_expected_runs() -> None:
    module = _load_script()
    args = module.parse_args(["--timestamp", "20260612_120000", "--repo-root", "/repo"])
    plan = module.build_queue_plan(args)

    command = module.build_report_command(plan)
    env = module.build_gpu6_env({})

    assert command == [
        "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python",
        "scripts/run_generator_comparison_report.py",
        "--runs",
        "e8",
        "e9",
        "e10",
        "--device",
        "cuda:0",
        "--python",
        "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python",
    ]
    assert env["CUDA_VISIBLE_DEVICES"] == "6"
    assert env["PYTHONPATH"] == "src"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_error_keyword_scan_reports_only_configured_failures(tmp_path: Path) -> None:
    module = _load_script()
    log_path = tmp_path / "train.log"
    log_path.write_text("ok\nRuntimeError: failed\nCUDA error: bad\n", encoding="utf-8")

    assert module.find_error_keywords(log_path) == ["RuntimeError", "CUDA error"]


def test_checkpoint_selection_prefers_best_stage2(tmp_path: Path) -> None:
    module = _load_script()
    best = tmp_path / "best_stage2.pt"
    last = tmp_path / "last.pt"
    last.write_text("last", encoding="utf-8")
    best.write_text("best", encoding="utf-8")

    assert module.first_existing_checkpoint((best, last)) == best
    assert module.first_existing_checkpoint((tmp_path / "missing.pt",)) is None


def test_dry_run_prints_commands_without_writing_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script()

    exit_code = module.main(
        [
            "--dry-run",
            "--timestamp",
            "20260612_120000",
            "--repo-root",
            str(tmp_path),
            "--meanflow-session",
            "safa_meanflow_e9_gpu6_200ep_20260612_034831",
            "--poll-seconds",
            "1",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "no waits" in output
    assert "safa_meanflow_e9_gpu6_200ep_20260612_034831" in output
    assert "safa_ddim_e10_gpu6_200ep_20260612_120000" in output
    assert "scripts/run_generator_comparison_report.py --runs e8 e9 e10 --device cuda:0" in output
    assert not (tmp_path / "artifacts").exists()
