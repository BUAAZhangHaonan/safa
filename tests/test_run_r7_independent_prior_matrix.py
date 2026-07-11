from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_r7_independent_prior_matrix.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("run_r7_independent_prior_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_r7_matrix_plan_has_four_pinned_train_and_eval_contracts() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--python",
            sys.executable,
            "--phase",
            "all",
            "--dry-run",
        ]
    )

    plan = module.validate_preflight(
        module.build_matrix_plan(args),
        gpu_states=_idle_gpu_states(module),
    )

    assert plan.allow_busy_gpus is False
    assert [run.physical_gpu for run in plan.runs] == [0, 1, 2, 3]
    assert len({run.config for run in plan.runs}) == 4
    assert len({run.checkpoint for run in plan.runs}) == 4
    for run in plan.runs:
        assert run.gpu_uuid == f"GPU-{run.physical_gpu}"
        assert run.env["CUDA_VISIBLE_DEVICES"] == run.gpu_uuid
        assert run.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert run.env["PYTHONPATH"] == "src"
        assert run.env["SAFA_REPO_ROOT"] == str(REPO_ROOT)
        assert run.train_command == (
            sys.executable,
            "-m",
            "safa.cli.train_g",
            "--config",
            str(run.config),
        )
        assert run.eval_command == (
            sys.executable,
            "scripts/r5_eval.py",
            str(run.checkpoint),
            run.experiment_name,
            "0",
            "--full-ft",
        )


def test_r7_matrix_preflight_accepts_current_train_then_eval_contract() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--python",
            sys.executable,
            "--phase",
            "all",
            "--dry-run",
        ]
    )
    plan = module.build_matrix_plan(args)

    bound_plan = module.validate_preflight(plan, gpu_states=_idle_gpu_states(module))
    assert [run.gpu_uuid for run in bound_plan.runs] == ["GPU-0", "GPU-1", "GPU-2", "GPU-3"]


def test_r7_matrix_dry_run_never_launches_commands(monkeypatch, capsys) -> None:
    module = _load_script()

    def forbidden_launch(_plan):
        raise AssertionError("dry-run attempted to launch commands")

    monkeypatch.setattr(module, "launch_matrix", forbidden_launch)
    monkeypatch.setattr(module, "query_gpu_states", lambda: _idle_gpu_states(module))

    exit_code = module.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--python",
            sys.executable,
            "--phase",
            "all",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "PREFLIGHT OK" in output
    assert "DRY RUN: no commands executed" in output
    for gpu in range(4):
        assert f"CUDA_VISIBLE_DEVICES=GPU-{gpu}" in output
    assert "safa.cli.train_g" in output
    assert "scripts/r5_eval.py" in output


def test_r7_matrix_supervisor_waits_for_all_runs_and_records_failures(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--python",
            sys.executable,
            "--phase",
            "all",
            "--execute",
            "--allow-busy-gpus",
        ]
    )
    gpu_states = _idle_gpu_states(module)
    gpu_states[0] = module.GpuState(
        index=0,
        uuid="GPU-0",
        free_memory_mib=22000,
        compute_processes=("pid=2197322 python",),
    )
    plan = module.validate_preflight(
        module.build_matrix_plan(args),
        gpu_states=gpu_states,
    )
    plan = replace(plan, log_dir=tmp_path / "logs")
    return_codes = iter([0, 7, 0, 0])
    processes = []

    class FakeProcess:
        def __init__(self, command, *, cwd, env):
            self.command = command
            self.cwd = cwd
            self.env = env
            self.returncode = next(return_codes)
            processes.append(self)

        def wait(self):
            assert len(processes) == 4
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)

    exit_code = module.launch_matrix(plan)

    assert exit_code == 1
    assert len(processes) == 4
    status = json.loads((plan.log_dir / "matrix_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "failed"
    assert status["allow_busy_gpus"] is True
    assert status["external_compute_processes"] == {"0": ["pid=2197322 python"]}
    assert [run["exit_code"] for run in status["runs"]] == [0, 7, 0, 0]
    assert [run["status"] for run in status["runs"]] == ["passed", "failed", "passed", "passed"]


def test_r7_matrix_preflight_rejects_busy_or_low_memory_gpus() -> None:
    module = _load_script()
    args = module.parse_args(["--repo-root", str(REPO_ROOT), "--python", sys.executable, "--dry-run"])
    plan = module.build_matrix_plan(args)
    states = _idle_gpu_states(module)

    busy = dict(states)
    busy[2] = module.GpuState(index=2, uuid="GPU-2", free_memory_mib=24000, compute_processes=("pid=123 python",))
    try:
        module.validate_preflight(plan, gpu_states=busy)
    except RuntimeError as exc:
        assert "compute process" in str(exc)
    else:
        raise AssertionError("busy GPU passed preflight")

    low_memory = dict(states)
    low_memory[3] = module.GpuState(index=3, uuid="GPU-3", free_memory_mib=19999, compute_processes=())
    try:
        module.validate_preflight(plan, gpu_states=low_memory)
    except RuntimeError as exc:
        assert "20000 MiB" in str(exc)
    else:
        raise AssertionError("low-memory GPU passed preflight")


def test_r7_matrix_authorized_busy_gpu_still_requires_memory_and_binds_uuid() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--python",
            sys.executable,
            "--dry-run",
            "--allow-busy-gpus",
        ]
    )
    plan = module.build_matrix_plan(args)
    states = _idle_gpu_states(module)
    states[0] = module.GpuState(
        index=0,
        uuid="GPU-busy-0",
        free_memory_mib=22000,
        compute_processes=("pid=2197322 python",),
    )

    bound_plan = module.validate_preflight(plan, gpu_states=states)

    assert bound_plan.allow_busy_gpus is True
    assert bound_plan.runs[0].gpu_uuid == "GPU-busy-0"
    assert bound_plan.runs[0].env["CUDA_VISIBLE_DEVICES"] == "GPU-busy-0"
    assert bound_plan.external_compute_processes == {0: ("pid=2197322 python",)}

    states[0] = replace(states[0], free_memory_mib=19999)
    try:
        module.validate_preflight(plan, gpu_states=states)
    except RuntimeError as exc:
        assert "20000 MiB" in str(exc)
    else:
        raise AssertionError("authorized busy GPU bypassed the memory floor")


def test_r7_matrix_artifact_preflight_refuses_overwrite_and_eval_without_checkpoint(tmp_path: Path) -> None:
    module = _load_script()
    args = module.parse_args(["--repo-root", str(REPO_ROOT), "--python", sys.executable, "--phase", "train"])
    plan = replace(module.build_matrix_plan(args), log_dir=tmp_path / "existing-logs")
    plan.log_dir.mkdir()

    try:
        module.validate_artifact_paths(plan)
    except FileExistsError as exc:
        assert "log" in str(exc)
    else:
        raise AssertionError("existing log directory passed preflight")

    missing_checkpoint = tmp_path / "missing.pt"
    eval_runs = tuple(replace(run, checkpoint=missing_checkpoint) for run in plan.runs)
    eval_plan = replace(plan, phase="eval", log_dir=tmp_path / "eval-logs", runs=eval_runs)
    try:
        module.validate_artifact_paths(eval_plan)
    except FileNotFoundError as exc:
        assert "checkpoint" in str(exc)
    else:
        raise AssertionError("eval without checkpoint passed preflight")


def _idle_gpu_states(module):
    return {
        index: module.GpuState(index=index, uuid=f"GPU-{index}", free_memory_mib=24000, compute_processes=())
        for index in range(4)
    }


def test_r7_matrix_cleans_up_owned_children_when_popen_launch_fails(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    args = module.parse_args(["--repo-root", str(REPO_ROOT), "--python", sys.executable, "--execute"])
    plan = module.validate_preflight(
        module.build_matrix_plan(args),
        gpu_states=_idle_gpu_states(module),
    )
    plan = replace(plan, log_dir=tmp_path / "logs")
    owned = []

    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return -15

    def fake_popen(command, *, cwd, env):
        del command, cwd, env
        if len(owned) == 2:
            raise OSError("launch failed")
        process = FakeProcess()
        owned.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    exit_code = module.launch_matrix(plan)

    assert exit_code == 1
    assert all(process.terminated and process.waited for process in owned)
    status = json.loads((plan.log_dir / "matrix_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "failed"
    assert "launch failed" in status["launch_error"]
    assert len(status["runs"]) == 4
    assert [run["status"] for run in status["runs"]] == [
        "terminated_after_launch_error",
        "terminated_after_launch_error",
        "launch_failed",
        "not_started",
    ]


def test_r7_matrix_execute_uses_atomic_lock_and_removes_it(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    args = module.parse_args(["--repo-root", str(REPO_ROOT), "--python", sys.executable, "--execute"])
    plan = replace(
        module.build_matrix_plan(args),
        lock_path=tmp_path / "matrix.lock",
        log_dir=tmp_path / "logs",
    )
    plan.lock_path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(module, "query_gpu_states", lambda: _idle_gpu_states(module))

    try:
        module.execute_matrix(plan)
    except FileExistsError as exc:
        assert "lock" in str(exc)
    else:
        raise AssertionError("existing matrix lock was ignored")

    plan.lock_path.unlink()
    monkeypatch.setattr(module, "launch_matrix", lambda _plan: 0)
    assert module.execute_matrix(plan) == 0
    assert not plan.lock_path.exists()
