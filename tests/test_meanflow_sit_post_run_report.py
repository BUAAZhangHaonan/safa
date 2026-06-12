from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_meanflow_sit_post_run_report.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("run_meanflow_sit_post_run_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_post_run_monitor_builds_non_gpu_report_command() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--repo-root",
            "/repo",
            "--python",
            "/env/bin/python",
            "--timestamp",
            "20260613_120000",
            "--sleep-seconds",
            "60",
        ]
    )
    plan = module.build_monitor_plan(args)

    assert plan.session == "safa_e11_meanflow_sit_k100_200ep_20260613_053015"
    assert plan.monitor_session == "safa_e11_meanflow_sit_report_monitor_20260613_120000"
    assert plan.output_json == Path("artifacts/reports/e11_meanflow_sit_stage1_report.json")
    command = module.build_report_command(plan)
    assert command[:2] == ["/env/bin/python", "scripts/run_meanflow_sit_stage1_report.py"]
    assert "--runs" in command
    assert "e11" in command
    assert "e8" in command
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(command)


def test_post_run_monitor_tmux_command_waits_for_training_session() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--repo-root",
            "/repo",
            "--python",
            "/env/bin/python",
            "--timestamp",
            "20260613_120000",
        ]
    )
    plan = module.build_monitor_plan(args)
    tmux_command = module.build_tmux_start_command(plan)
    shell_command = tmux_command[-1]

    assert tmux_command[:5] == ["tmux", "new-session", "-d", "-s", "safa_e11_meanflow_sit_report_monitor_20260613_120000"]
    assert "run_meanflow_sit_post_run_report.py" in shell_command
    assert "--wait-only" in shell_command
    assert "safa_e11_meanflow_sit_k100_200ep_20260613_053015" in shell_command
    assert "artifacts/logs/e11_meanflow_sit_report_monitor_20260613_120000.log" in shell_command
