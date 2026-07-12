from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_r8_meanflow_guidance_matrix.py"
    spec = importlib.util.spec_from_file_location("run_r8_meanflow_guidance_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _args(module, *extra: str):
    return module.parse_args(
        ["--repo-root", str(REPO_ROOT), "--python", sys.executable, *extra]
    )


def _idle_gpu_states(module):
    return {
        index: module.GpuState(
            index=index,
            uuid=f"GPU-{index}",
            free_memory_mib=24000,
            compute_processes=(),
        )
        for index in range(4)
    }


def _passing_gate(module) -> dict:
    return {
        "gate_passed": True,
        "checkpoint_sha256": module.CHECKPOINT_SHA256,
        "selected_t_cut": 0.25,
        "t_cut": 0.25,
        "sample_id_manifest_sha256": module.CALIBRATION_MANIFEST_SHA256,
    }


def test_matrix_pins_exact_physical_gpus_zero_through_three() -> None:
    module = _load_script()
    plan = module.build_matrix_plan(_args(module, "--phase", "semigroup", "--dry-run"))

    assert [run.physical_gpu for run in plan.runs] == [0, 1, 2, 3]
    assert [run.shard_index for run in plan.runs] == [0, 1, 2, 3]
    assert all(run.num_shards == 4 for run in plan.runs)


def test_matrix_uses_gpu_uuid_as_cuda_visible_devices() -> None:
    module = _load_script()
    plan = module.validate_preflight(
        module.build_matrix_plan(_args(module, "--phase", "semigroup")),
        gpu_states=_idle_gpu_states(module),
    )

    assert [run.gpu_uuid for run in plan.runs] == ["GPU-0", "GPU-1", "GPU-2", "GPU-3"]
    assert [run.env["CUDA_VISIBLE_DEVICES"] for run in plan.runs] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    ]
    assert all(run.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for run in plan.runs)


def test_matrix_dry_run_is_default_and_has_no_writes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "query_gpu_states", lambda: _idle_gpu_states(module))
    monkeypatch.setattr(module, "launch_matrix", lambda plan: pytest.fail("dry-run launched"))
    before = set((REPO_ROOT / "artifacts").iterdir())

    assert module.main(["--repo-root", str(REPO_ROOT), "--python", sys.executable, "--phase", "all"]) == 0

    assert set((REPO_ROOT / "artifacts").iterdir()) == before
    output = capsys.readouterr().out
    assert "DRY RUN: no commands executed" in output
    assert "phase: semigroup" in output
    assert "phase: calibrate" in output
    assert "phase: full" in output


def test_matrix_requires_explicit_execute() -> None:
    module = _load_script()
    assert _args(module).execute is False
    assert _args(module, "--dry-run").execute is False
    assert _args(module, "--execute").execute is True


@pytest.mark.parametrize(
    "field",
    ["checkpoint", "e0_checkpoint", "vae_path", "index", "features"],
)
def test_matrix_rejects_missing_checkpoint_e0_vae_index_or_features(
    tmp_path: Path, field: str
) -> None:
    module = _load_script()
    config_path = REPO_ROOT / module.SEMIGROUP_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config[field] = str(tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match=field):
        module.validate_config_assets(REPO_ROOT, config)


def test_matrix_rejects_existing_output_directory(tmp_path: Path) -> None:
    module = _load_script()
    plan = module.build_matrix_plan(_args(module, "--phase", "semigroup"))
    output = tmp_path / "existing"
    output.mkdir()
    run = replace(plan.runs[0], output_dir=output)

    with pytest.raises(FileExistsError, match="overwrite"):
        module.validate_artifact_paths(replace(plan, runs=(run,)))


def test_matrix_requires_passing_semigroup_report_for_fmrg(tmp_path: Path) -> None:
    module = _load_script()
    with pytest.raises(FileNotFoundError, match="semigroup"):
        args = _args(module, "--phase", "calibrate")
        args.repo_root = tmp_path
        module.build_matrix_plan(args)

    failed = {**_passing_gate(module), "gate_passed": False}
    fallback = module.build_matrix_plan(
        _args(module, "--phase", "calibrate"), semigroup_gate=failed
    )
    assert {run.family for run in fallback.runs} == {"initial_noise_fallback"}


def test_matrix_semigroup_shards_64_ids_across_all_four_gpus() -> None:
    module = _load_script()
    plan = module.build_matrix_plan(_args(module, "--phase", "semigroup"))

    assert len(plan.runs) == 4
    for gpu, run in enumerate(plan.runs):
        command = " ".join(run.command)
        assert f"--shard-index {gpu}" in command
        assert "--num-shards 4" in command
        assert run.sample_count == 16
        assert run.sample_manifest == module.CALIBRATION_MANIFEST


def test_matrix_semigroup_resume_skips_only_owned_completed_shards() -> None:
    module = _load_script()
    plan = module.build_matrix_plan(_args(module, "--phase", "semigroup"))

    for gpu, run in enumerate(plan.runs):
        command = " ".join(run.command)
        assert (
            f"if test -f artifacts/r8_meanflow_flow_map_guidance/semigroup/shards/"
            f"shard_{gpu}/completion.json"
        ) in command
        assert "pkill" not in command
        assert "killall" not in command


def _write_semigroup_shards(root: Path, *, duplicate: bool = False, missing: bool = False) -> tuple[list[Path], Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    ids = [f"sample-{index}" for index in range(64)]
    manifest.write_text("".join(json.dumps({"sample_id": value}) + "\n" for value in ids), encoding="utf-8")
    paths = []
    for shard in range(4):
        shard_dir = root / f"shard_{shard}"
        shard_dir.mkdir()
        shard_ids = ids[shard::4]
        if duplicate and shard == 1:
            shard_ids[0] = ids[0]
        if missing and shard == 3:
            shard_ids.pop()
        rows = [
            {
                "sample_id": sample_id,
                "splits": {
                    "0.25": {
                        "latent_residual": 0.05,
                        "decoded_pixel_l1": 0.01,
                        "decoded_psnr": 30.0,
                        "endpoint_e0_cosine": 0.98,
                    },
                    "0.5": {
                        "latent_residual": 0.15,
                        "decoded_pixel_l1": 0.02,
                        "decoded_psnr": 25.0,
                        "endpoint_e0_cosine": 0.90,
                    },
                    "0.75": {
                        "latent_residual": 0.25,
                        "decoded_pixel_l1": 0.03,
                        "decoded_psnr": 20.0,
                        "endpoint_e0_cosine": 0.80,
                    },
                },
            }
            for sample_id in shard_ids
        ]
        path = shard_dir / "semigroup.json"
        path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        paths.append(path)
    return paths, manifest


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_matrix_semigroup_merge_rejects_missing_or_duplicate_ids(tmp_path: Path, case: str) -> None:
    module = _load_script()
    paths, manifest = _write_semigroup_shards(
        tmp_path,
        duplicate=case == "duplicate",
        missing=case == "missing",
    )

    with pytest.raises(ValueError, match=case):
        module.merge_semigroup_shards(
            paths,
            manifest_path=manifest,
            thresholds={"median": 0.1, "p90": 0.2, "endpoint_e0_cosine": 0.95},
            visual_pass_by_split={"0.25": True, "0.5": True, "0.75": True},
            checkpoint_sha256=module.CHECKPOINT_SHA256,
        )


def test_matrix_semigroup_merge_sorts_and_locks_smallest_full_pass(tmp_path: Path) -> None:
    module = _load_script()
    paths, manifest = _write_semigroup_shards(tmp_path)

    report = module.merge_semigroup_shards(
        paths,
        manifest_path=manifest,
        thresholds={"median": 0.1, "p90": 0.2, "endpoint_e0_cosine": 0.95},
        visual_pass_by_split={"0.75": True, "0.5": True, "0.25": True},
        checkpoint_sha256=module.CHECKPOINT_SHA256,
    )

    assert report["gate_passed"] is True
    assert report["selected_t_cut"] == 0.25
    assert [row["t_cut"] for row in report["candidates"]] == [0.25, 0.5, 0.75]


def test_matrix_semigroup_waits_for_direct_visual_review_before_gate(tmp_path: Path) -> None:
    module = _load_script()
    source_paths, source_manifest = _write_semigroup_shards(tmp_path / "source")
    manifest = tmp_path / module.CALIBRATION_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(source_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    for gpu, source in enumerate(source_paths):
        target = tmp_path / module.ROOT / f"semigroup/shards/shard_{gpu}/semigroup.json"
        target.parent.mkdir(parents=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / module.SEMIGROUP_CONFIG
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "semigroup_thresholds": {
                    "median": 0.1,
                    "p90": 0.2,
                    "endpoint_e0_cosine": 0.95,
                }
            }
        ),
        encoding="utf-8",
    )
    plan = replace(
        module.build_matrix_plan(_args(module, "--phase", "semigroup")),
        repo_root=tmp_path,
    )

    assert module.finalize_phase(plan) == 2
    assert (tmp_path / module.SEMIGROUP_GATE_DRAFT).is_file()
    assert not (tmp_path / module.SEMIGROUP_GATE).exists()
    assert not (tmp_path / module.SCHEDULE_MANIFEST).exists()


def test_matrix_calibration_launches_four_processes_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    plan = module.validate_preflight(
        module.build_matrix_plan(
            _args(module, "--phase", "calibrate", "--allow-busy-gpus"),
            semigroup_gate=_passing_gate(module),
        ),
        gpu_states=_idle_gpu_states(module),
    )
    plan = replace(plan, status_dir=tmp_path / "status")
    processes = []

    class FakeProcess:
        def __init__(self, command, *, cwd, env, start_new_session):
            del command, cwd, env, start_new_session
            self.pid = 1000 + len(processes)
            self.returncode = 0
            processes.append(self)

        def wait(self):
            assert len(processes) == 4
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)

    assert module.launch_matrix(plan) == 0
    assert len(processes) == 4


def test_matrix_failed_semigroup_replaces_all_four_arms_with_noise_configs() -> None:
    module = _load_script()
    plan = module.build_matrix_plan(
        _args(module, "--phase", "calibrate"),
        semigroup_gate={**_passing_gate(module), "gate_passed": False},
    )

    assert [run.physical_gpu for run in plan.runs] == [0, 1, 2, 3]
    assert [run.config for run in plan.runs] == list(module.NOISE_CONFIGS)
    assert all("official" not in " ".join(run.command) for run in plan.runs)
    assert all("paper_split" not in " ".join(run.command) for run in plan.runs)


def test_matrix_full_requires_2048_samples_and_visual_review(tmp_path: Path) -> None:
    module = _load_script()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"winner": {"config": str(module.NOISE_CONFIGS[0])}}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="visual_review"):
        module.load_full_contract(selection, tmp_path / "visual_review.json")

    review = tmp_path / "visual_review.json"
    review.write_text(json.dumps({"reviewed_sample_count": 63, "passed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="64"):
        module.load_full_contract(selection, review)


def test_matrix_full_shards_locked_native_and_winner_across_four_gpus() -> None:
    module = _load_script()
    contract = {
        "winner": {"arm_id": "winner", "config": str(module.NOISE_CONFIGS[0])},
        "visual_review": {"reviewed_sample_count": 64, "passed": True},
        "manifest_count": 2048,
        "manifest_sha256": module.FULL_MANIFEST_SHA256,
    }
    plan = module.build_matrix_plan(
        _args(module, "--phase", "full"), full_contract=contract
    )

    assert len(plan.runs) == 4
    assert all(run.sample_count == 512 and run.num_shards == 4 for run in plan.runs)
    for run in plan.runs:
        command = " ".join(run.command)
        assert run.arm_ids == ("native", "winner")
        assert "runtime_configs/native.yaml" in command
        assert "runtime_configs/winner.yaml" in command
        assert run.sample_manifest == module.FULL_MANIFEST


def test_matrix_quality_commands_require_manifest_and_per_sample_join() -> None:
    module = _load_script()
    command = module.build_quality_command(
        python=sys.executable,
        output_dir=Path("arm"),
        manifest=module.CALIBRATION_MANIFEST,
    )
    text = " ".join(command)

    assert "--sample-id-manifest" in text
    assert "--per-sample-jsonl arm/per_sample.jsonl" in text
    assert "--generated-dir arm/generated_images" in text
    assert "--metrics fid kid niqe sharpness" in text


def test_matrix_quality_commands_never_use_max_count_flags() -> None:
    module = _load_script()
    for manifest in (module.CALIBRATION_MANIFEST, module.FULL_MANIFEST):
        text = " ".join(
            module.build_quality_command(
                python=sys.executable,
                output_dir=Path("arm"),
                manifest=manifest,
            )
        )
        assert "--max-real" not in text
        assert "--max-generated" not in text


def test_matrix_calibration_never_overwrites_completed_quality() -> None:
    module = _load_script()
    plan = module.build_matrix_plan(
        _args(module, "--phase", "calibrate"), semigroup_gate=_passing_gate(module)
    )

    for run in plan.runs:
        command = " ".join(run.command)
        assert "/quality.json; then :; else" in command


def test_matrix_records_exit_codes_peak_memory_and_external_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    states = _idle_gpu_states(module)
    states[0] = replace(states[0], compute_processes=("pid=77 python",))
    plan = module.validate_preflight(
        module.build_matrix_plan(
            _args(module, "--phase", "semigroup", "--allow-busy-gpus")
        ),
        gpu_states=states,
    )
    plan = replace(plan, status_dir=tmp_path / "status")
    processes = []

    class FakeProcess:
        def __init__(self, command, *, cwd, env, start_new_session):
            del command, cwd, env, start_new_session
            self.pid = 2000 + len(processes)
            self.returncode = 3 if len(processes) == 1 else 0
            processes.append(self)

        def wait(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module, "run_peak_memory", lambda run: {"allocated": 12, "reserved": 34})

    assert module.launch_matrix(plan) == 1
    status = json.loads((plan.status_dir / "matrix_status.json").read_text(encoding="utf-8"))
    assert status["external_compute_processes"] == {"0": ["pid=77 python"]}
    assert [row["exit_code"] for row in status["runs"]] == [0, 3, 0, 0]
    assert all(row["peak_memory"] == {"allocated": 12, "reserved": 34} for row in status["runs"])
    assert all(row["pid"] is not None for row in status["runs"])


def test_matrix_terminates_started_children_after_partial_launch_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    plan = module.validate_preflight(
        module.build_matrix_plan(_args(module, "--phase", "semigroup")),
        gpu_states=_idle_gpu_states(module),
    )
    plan = replace(plan, status_dir=tmp_path / "status")
    started = []

    class FakeProcess:
        def __init__(self):
            self.pid = 3000 + len(started)
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return -15

    def fake_popen(command, *, cwd, env, start_new_session):
        del command, cwd, env, start_new_session
        if len(started) == 2:
            raise OSError("launch failed")
        process = FakeProcess()
        started.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    assert module.launch_matrix(plan) == 1
    assert all(process.terminated and process.waited for process in started)
    status = json.loads((plan.status_dir / "matrix_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "failed"
    assert "launch failed" in status["launch_error"]


def test_fmrg_full_plan_requires_and_uses_locked_tcut_and_schedule_digest() -> None:
    module = _load_script()
    contract = {
        "winner": {
            "arm_id": "official",
            "config": str(module.FLOW_MAP1_CONFIG),
            "mode": "official_head_current_xt",
            "t_cut": 0.25,
            "schedule_manifest": str(module.SCHEDULE_MANIFEST),
            "schedule_manifest_sha256": "d" * 64,
            "schedule_contract_sha256": "e" * 64,
        },
        "visual_review": {"reviewed_sample_count": 64, "passed": True},
        "manifest_count": 2048,
        "manifest_sha256": module.FULL_MANIFEST_SHA256,
    }

    plan = module.build_matrix_plan(
        _args(module, "--phase", "full"), full_contract=contract
    )

    assert plan.schedule_manifest == module.SCHEDULE_MANIFEST
    assert all("--t-cut 0.25" in " ".join(run.command) for run in plan.runs)
    missing = {**contract, "winner": {**contract["winner"]}}
    missing["winner"].pop("t_cut")
    with pytest.raises(ValueError, match="t_cut"):
        module.build_matrix_plan(_args(module, "--phase", "full"), full_contract=missing)


def test_semigroup_merge_requires_registered_splits_and_four_exact_modulo_shards(
    tmp_path: Path,
) -> None:
    module = _load_script()
    paths, manifest = _write_semigroup_shards(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["rows"][0]["splits"]["0.125"] = dict(payload["rows"][0]["splits"]["0.25"])
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registered split"):
        module.merge_semigroup_shards(
            paths,
            manifest_path=manifest,
            thresholds={"median": 0.1, "p90": 0.2, "endpoint_e0_cosine": 0.95},
            visual_pass_by_split={"0.25": True, "0.5": True, "0.75": True},
            checkpoint_sha256=module.CHECKPOINT_SHA256,
        )

    paths, manifest = _write_semigroup_shards(tmp_path / "short")
    payload = json.loads(paths[3].read_text(encoding="utf-8"))
    payload["rows"].pop()
    paths[3].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="16"):
        module.merge_semigroup_shards(
            paths,
            manifest_path=manifest,
            thresholds={"median": 0.1, "p90": 0.2, "endpoint_e0_cosine": 0.95},
            visual_pass_by_split={"0.25": True, "0.5": True, "0.75": True},
            checkpoint_sha256=module.CHECKPOINT_SHA256,
        )


def test_schedule_payload_binds_report_manifest_times_and_self_digest(tmp_path: Path) -> None:
    module = _load_script()
    paths, manifest = _write_semigroup_shards(tmp_path)
    report = module.merge_semigroup_shards(
        paths,
        manifest_path=manifest,
        thresholds={"median": 0.1, "p90": 0.2, "endpoint_e0_cosine": 0.95},
        visual_pass_by_split={"0.25": True, "0.5": True, "0.75": True},
        checkpoint_sha256=module.CHECKPOINT_SHA256,
    )

    schedule = module._schedule_payload(report, semigroup_report_sha256="f" * 64)

    assert schedule["guided_steps"] == 3
    assert schedule["unguided_tail_intervals"] == 2
    assert schedule["guided_times"] == [1.0, 0.75, 0.5, 0.25]
    assert schedule["unguided_times"] == [0.25, 0.125, 0.0]
    assert schedule["semigroup_report_sha256"] == "f" * 64
    assert schedule["sample_id_manifest_sha256"] == module._sha256_file(manifest)
    assert module._schedule_contract_digest(schedule) == schedule["schedule_contract_sha256"]


def test_peak_memory_recurses_over_every_owned_arm(tmp_path: Path) -> None:
    module = _load_script()
    plan = module.build_matrix_plan(
        _args(module, "--phase", "calibrate"), semigroup_gate=_passing_gate(module)
    )
    run = replace(plan.runs[0], output_dir=tmp_path)
    for name, allocated, reserved in (("first", 10, 20), ("nested/second", 30, 25)):
        path = tmp_path / name / "generation_result.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "max_memory": {
                        "allocated_bytes": allocated,
                        "reserved_bytes": reserved,
                    }
                }
            ),
            encoding="utf-8",
        )

    assert module.run_peak_memory(run) == {"allocated": 30, "reserved": 25}


def test_terminate_process_group_reaps_shell_and_child(tmp_path: Path) -> None:
    module = _load_script()
    child_pid_path = tmp_path / "child.pid"
    process = subprocess.Popen(
        [
            "/bin/bash",
            "-lc",
            f"sleep 60 & echo $! > {child_pid_path}; wait",
        ],
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while not child_pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    module._terminate_process_group(process, terminate_timeout=1.0)

    assert process.poll() is not None
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_execute_plan_marks_finalize_failure_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    plan = replace(
        module.build_matrix_plan(_args(module, "--phase", "semigroup", "--execute")),
        repo_root=tmp_path,
        status_dir=Path("status"),
        lock_path=Path("matrix.lock"),
    )
    monkeypatch.setattr(module, "validate_artifact_paths", lambda value: None)
    monkeypatch.setattr(module, "validate_preflight", lambda value: value)
    monkeypatch.setattr(module, "materialize_locked_manifests", lambda value: None)
    monkeypatch.setattr(module, "materialize_full_runtime_configs", lambda value: None)

    def fake_launch(value):
        status = value.repo_root / value.status_dir / "matrix_status.json"
        status.parent.mkdir(parents=True)
        status.write_text(
            json.dumps({"overall_status": "children_passed_pending_finalize"}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "launch_matrix", fake_launch)
    monkeypatch.setattr(module, "finalize_phase", lambda value: (_ for _ in ()).throw(RuntimeError("merge failed")))

    assert module.execute_plan(plan) == 1
    status = json.loads((tmp_path / "status/matrix_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "failed"
    assert "merge failed" in status["finalize_error"]


def test_full_merge_is_idempotent_and_recovers_partial_owned_output(tmp_path: Path) -> None:
    module = _load_script()
    manifest = tmp_path / module.FULL_MANIFEST
    manifest.parent.mkdir(parents=True)
    ids = [f"sample-{index}" for index in range(4)]
    manifest.write_text(
        "".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in ids),
        encoding="utf-8",
    )
    for gpu, sample_id in enumerate(ids):
        shard = tmp_path / module.ROOT / f"full/shards/shard_{gpu}/native"
        generated = shard / f"{gpu}.png"
        generated.parent.mkdir(parents=True)
        generated.write_bytes(f"image-{gpu}".encode())
        (shard / "per_sample.jsonl").write_text(
            json.dumps({"sample_id": sample_id, "generated": str(generated)}) + "\n",
            encoding="utf-8",
        )
        (shard / "completion.json").write_text(
            json.dumps({"status": "complete", "sample_count": 1}), encoding="utf-8"
        )
    combined = tmp_path / module.ROOT / "full/merged/native"

    first = module._merge_full_arm(tmp_path, "native", combined)
    second = module._merge_full_arm(tmp_path, "native", combined)
    assert first == second
    completion = json.loads((combined / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "complete"

    (combined / "completion.json").unlink()
    victim = next((combined / "generated_images").iterdir())
    victim.unlink()
    resumed = module._merge_full_arm(tmp_path, "native", combined)
    assert resumed["status"] == "complete"
    assert victim.is_file()
