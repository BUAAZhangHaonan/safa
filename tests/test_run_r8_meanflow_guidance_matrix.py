from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

from safa.evaluation.meanflow_guidance_runner import resolve_locked_schedule
from safa.evaluation.r8_arm_contracts import canonical_arm_config_digest


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
    values = list(extra)
    phase = values[values.index("--phase") + 1] if "--phase" in values else "all"
    if phase in {"calibrate", "all"} and "--campaign-id" not in values:
        values.extend(("--campaign-id", "test-campaign"))
    return module.parse_args(
        ["--repo-root", str(REPO_ROOT), "--python", sys.executable, *values]
    )
def _calibration_args(module, campaign_id: str):
    return module.argparse.Namespace(
        repo_root=REPO_ROOT,
        python=sys.executable,
        phase="calibrate",
        dry_run=False,
        execute=False,
        allow_busy_gpus=False,
        campaign_id=campaign_id,
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

    assert module.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--python",
            sys.executable,
            "--phase",
            "all",
            "--campaign-id",
            "test-campaign",
        ]
    ) == 0

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

def test_campaign_id_is_required_strict_and_calibration_only() -> None:
    module = _load_script()
    base = ["--repo-root", str(REPO_ROOT), "--python", sys.executable]

    with pytest.raises(SystemExit):
        module.parse_args([*base, "--phase", "calibrate"])
    valid = module.parse_args(
        [*base, "--phase", "calibrate", "--campaign-id", "campaign-2026-07-13"]
    )
    assert valid.campaign_id == "campaign-2026-07-13"

    for invalid in (
        "../escape",
        "nested/path",
        ".hidden",
        "Uppercase",
        "under_score",
        "double--dash",
        "trailing-",
    ):
        with pytest.raises(SystemExit):
            module.parse_args(
                [*base, "--phase", "calibrate", "--campaign-id", invalid]
            )
    for phase in ("semigroup", "full"):
        with pytest.raises(SystemExit):
            module.parse_args(
                [*base, "--phase", phase, "--campaign-id", "campaign-a"]
            )


def test_calibration_campaign_paths_are_isolated_and_shared_inputs_stay_locked() -> None:
    module = _load_script()
    plan = module.build_matrix_plan(
        _calibration_args(module, "campaign-a"),
        semigroup_gate=_passing_gate(module),
    )
    campaign_root = module.ROOT / "campaigns" / "campaign-a"

    assert plan.campaign_id == "campaign-a"
    assert plan.campaign_root == campaign_root
    assert plan.status_dir == campaign_root / "status"
    assert plan.lock_path == campaign_root / ".calibrate.lock"
    assert len(plan.runs) == 21
    assert all(
        run.output_dir == campaign_root / "calibration" / run.arm_ids[0]
        for run in plan.runs
    )
    assert all(run.log_path.is_relative_to(campaign_root / "logs") for run in plan.runs)
    assert plan.sample_manifest == module.CALIBRATION_MANIFEST
    assert plan.sample_manifest_sha256 == module.CALIBRATION_MANIFEST_SHA256
    assert plan.schedule_manifest == module.SCHEDULE_MANIFEST

    for run in plan.runs:
        command = " ".join(run.command)
        old_completion = module.ROOT / "calibration" / run.arm_ids[0] / "completion.json"
        assert str(run.output_dir / "completion.json") in command
        assert str(old_completion) not in command


def test_two_calibration_campaigns_have_disjoint_writable_paths() -> None:
    module = _load_script()
    plans = [
        module.build_matrix_plan(
            _calibration_args(module, campaign_id),
            semigroup_gate=_passing_gate(module),
        )
        for campaign_id in ("campaign-a", "campaign-b")
    ]

    def writable_paths(plan):
        return {
            plan.status_dir,
            plan.lock_path,
            plan.campaign_root / "campaign_contract.json",
            plan.campaign_root / "visual_review.json",
            plan.campaign_root / "calibration/visual_evidence.json",
            *(run.output_dir for run in plan.runs),
            *(run.log_path for run in plan.runs),
            *(run.runtime_config for run in plan.runs if run.runtime_config is not None),
        }

    assert writable_paths(plans[0]).isdisjoint(writable_paths(plans[1]))


def _campaign_plan_in_tmp(module, tmp_path: Path):
    plan = module.build_matrix_plan(
        _calibration_args(module, "contract-test"),
        semigroup_gate=_passing_gate(module),
    )
    runs = tuple(
        replace(
            run,
            config=REPO_ROOT / run.config,
            source_configs=tuple(REPO_ROOT / path for path in run.source_configs),
            sample_manifest=REPO_ROOT / run.sample_manifest,
        )
        for run in plan.runs
    )
    return replace(
        plan,
        repo_root=tmp_path,
        runs=runs,
        sample_manifest=REPO_ROOT / plan.sample_manifest,
        schedule_manifest=REPO_ROOT / plan.schedule_manifest,
    )


def _kernel_lock_execute_worker(
    repo_root: str,
    host_lock_dir: str,
    campaign_id: str,
    barrier,
    release,
    events,
    mode: str,
) -> None:
    module = _load_script()
    module.HOST_GPU_LOCK_DIR = Path(host_lock_dir)
    plan = replace(
        module.build_matrix_plan(
            _calibration_args(module, campaign_id),
            semigroup_gate=_passing_gate(module),
        ),
        repo_root=Path(repo_root),
        execute=True,
    )
    module.validate_artifact_paths = lambda value: None
    module.materialize_locked_manifests = lambda value: None
    module.ensure_campaign_contract = lambda value: {}
    module.materialize_calibration_runtime_configs = lambda value: None
    module.materialize_full_runtime_configs = lambda value: None

    def preflight(value):
        events.put((campaign_id, os.getpid(), "query"))
        if mode == "hold_preflight" and not release.wait(10.0):
            raise TimeoutError("test release barrier timed out")
        return value

    def launch(value):
        events.put((campaign_id, os.getpid(), "launch"))
        if mode == "crash_launch":
            os._exit(17)
        return 1

    module.validate_preflight = preflight
    module.launch_matrix = launch
    try:
        barrier.wait(timeout=10.0)
        module.execute_plan(plan)
        events.put((campaign_id, os.getpid(), "done"))
    except Exception as exc:
        events.put((campaign_id, os.getpid(), "rejected", type(exc).__name__, str(exc)))


def _collect_process_events(events, count: int) -> list[tuple]:
    return [events.get(timeout=10.0) for _ in range(count)]


def _kernel_lock_hold_gpu_worker(
    host_lock_dir: str, physical_gpu: int, ready, release, events
) -> None:
    module = _load_script()
    module.HOST_GPU_LOCK_DIR = Path(host_lock_dir)
    path = module.HOST_GPU_LOCK_DIR / f"gpu{physical_gpu}.lock"
    fd = module._acquire_kernel_lease(
        path,
        module.HOST_GPU_LOCK_DIR.parent,
        f"test physical GPU {physical_gpu} lease",
    )
    try:
        events.put(("held", physical_gpu))
        ready.set()
        if not release.wait(10.0):
            raise TimeoutError("test GPU lock release timed out")
    finally:
        os.close(fd)


def _kernel_lock_probe_worker(
    repo_root: str, host_lock_dir: str, campaign_id: str, events
) -> None:
    module = _load_script()
    module.HOST_GPU_LOCK_DIR = Path(host_lock_dir)
    plan = module.build_matrix_plan(
        _calibration_args(module, campaign_id),
        semigroup_gate=_passing_gate(module),
    )
    gpu0_run = next(run for run in plan.runs if run.physical_gpu == 0)
    plan = replace(plan, repo_root=Path(repo_root), runs=(gpu0_run,))
    try:
        with module.execution_leases(plan):
            events.put(("acquired", os.getpid()))
    except Exception as exc:
        events.put(("rejected", type(exc).__name__, str(exc)))


def test_campaign_contract_is_atomic_immutable_and_allows_exact_resume(tmp_path: Path) -> None:
    module = _load_script()
    plan = _campaign_plan_in_tmp(module, tmp_path)

    expected = module.ensure_campaign_contract(plan)
    contract_path = tmp_path / plan.campaign_root / "campaign_contract.json"

    assert json.loads(contract_path.read_text(encoding="utf-8")) == expected
    assert module._canonical_contract_digest(
        expected, "campaign_contract_sha256"
    ) == expected["campaign_contract_sha256"]
    assert not list(contract_path.parent.glob(".*.tmp"))
    assert module.ensure_campaign_contract(plan) == expected


def test_native_unguided_calibration_baseline_has_complete_locked_contract(
    tmp_path: Path,
) -> None:
    module = _load_script()
    plan = module.build_matrix_plan(
        _calibration_args(module, "native-baseline"),
        semigroup_gate=_passing_gate(module),
    )

    assert len(plan.runs) == 21
    run = next(candidate for candidate in plan.runs if candidate.arm_ids == ("native_unguided_64",))
    assert run.physical_gpu == 3
    assert run.family == "native_unguided"
    assert run.config == module.NATIVE_CONFIG
    assert run.source_configs == (module.NATIVE_CONFIG,)
    assert run.shard_index == 0
    assert run.num_shards == 1
    assert run.sample_count == 64
    assert run.sample_manifest == module.CALIBRATION_MANIFEST
    assert run.sample_manifest_sha256 == module.CALIBRATION_MANIFEST_SHA256
    assert run.output_dir == plan.campaign_root / "calibration/native_unguided_64"
    assert run.log_path == plan.campaign_root / "logs/gpu3_native_unguided_64.log"
    assert run.runtime_config == plan.campaign_root / "runtime_configs/native_unguided_64.yaml"

    command = " ".join(run.command)
    assert f"--config {run.runtime_config}" in command
    assert "--mode native" in command
    assert "--semigroup-report" not in command
    assert "--schedule-manifest" not in command
    assert "--t-cut" not in command
    assert str(run.output_dir / "completion.json") in command
    assert "--reuse-valid-output" in command
    assert "--generation-result" in command
    assert "--seed 1337" in command

    tmp_plan = _campaign_plan_in_tmp(module, tmp_path)
    tmp_run = next(candidate for candidate in tmp_plan.runs if candidate.family == "native_unguided")
    runtime = module._native_unguided_runtime_config(tmp_plan, tmp_run)
    assert runtime["mode"] == "native"
    assert runtime["phase"] == "calibration"
    assert runtime["max_samples"] == 64
    assert runtime["sample_id_manifest"] == str(module.CALIBRATION_MANIFEST)
    assert runtime["sample_id_manifest_sha256"] == module.CALIBRATION_MANIFEST_SHA256
    assert runtime["sampling_seed"] == 1337
    assert runtime["contact_sheets"] is True
    assert "schedule_manifest" not in runtime
    assert runtime["asset_digest_cache"] == str(module.ROOT / "shared/asset_digests.json")

    contract = module.build_campaign_contract(tmp_plan)
    native = next(row for row in contract["runs"] if row["arm_ids"] == ["native_unguided_64"])
    assert native["sampling_seed"] == 1337
    assert native["runtime_config"]["path"] == str(tmp_run.runtime_config)
    assert len(native["runtime_config"]["sha256"]) == 64

    module.materialize_calibration_runtime_configs(tmp_plan)
    runtime_path = tmp_path / tmp_run.runtime_config
    assert yaml.safe_load(runtime_path.read_text(encoding="utf-8")) == runtime
    module.materialize_calibration_runtime_configs(tmp_plan)


def test_native_unguided_calibration_requires_registered_seed_1337(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    real_load_yaml = module._load_yaml

    def tampered_seed(path: Path):
        payload = real_load_yaml(path)
        if Path(path).resolve() == (REPO_ROOT / module.NATIVE_CONFIG).resolve():
            payload["sampling_seed"] = 1338
        return payload

    monkeypatch.setattr(module, "_load_yaml", tampered_seed)
    with pytest.raises(ValueError, match="sampling_seed.*1337"):
        module.build_matrix_plan(
            _calibration_args(module, "wrong-seed"),
            semigroup_gate=_passing_gate(module),
        )


def test_campaign_contract_rejects_new_nonempty_root_without_contract(tmp_path: Path) -> None:
    module = _load_script()
    plan = _campaign_plan_in_tmp(module, tmp_path)
    campaign_root = tmp_path / plan.campaign_root
    campaign_root.mkdir(parents=True)
    (campaign_root / "unowned.txt").write_text("junk", encoding="utf-8")

    with pytest.raises(FileExistsError, match="entries but no contract"):
        module.ensure_campaign_contract(plan)
    assert not (campaign_root / "campaign_contract.json").exists()


@pytest.mark.parametrize("symlink_level", ["campaigns", "campaign"])
def test_campaign_contract_rejects_symlinked_path_components_before_outside_write(
    tmp_path: Path, symlink_level: str
) -> None:
    module = _load_script()
    repo_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    plan = _campaign_plan_in_tmp(module, repo_root)
    campaigns_root = repo_root / module.ROOT / "campaigns"
    if symlink_level == "campaigns":
        campaigns_root.parent.mkdir(parents=True)
        campaigns_root.symlink_to(outside, target_is_directory=True)
    else:
        campaigns_root.mkdir(parents=True)
        (repo_root / plan.campaign_root).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink path component"):
        module.ensure_campaign_contract(plan)

    assert list(outside.iterdir()) == []


def test_execute_rejects_symlinked_campaigns_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    repo_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    plan = replace(_campaign_plan_in_tmp(module, repo_root), execute=True)
    campaigns_root = repo_root / module.ROOT / "campaigns"
    campaigns_root.parent.mkdir(parents=True)
    campaigns_root.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(module, "validate_artifact_paths", lambda value: None)
    monkeypatch.setattr(module, "validate_preflight", lambda value: value)
    monkeypatch.setattr(
        module,
        "materialize_locked_manifests",
        lambda value: pytest.fail("execute wrote manifests before campaign path validation"),
    )

    with pytest.raises(ValueError, match="symlink path component"):
        module.execute_plan(plan)

    assert list(outside.iterdir()) == []


def test_campaign_contract_rejects_any_semantic_change(tmp_path: Path) -> None:
    module = _load_script()
    plan = _campaign_plan_in_tmp(module, tmp_path)
    module.ensure_campaign_contract(plan)
    first = plan.runs[0]

    changed_command = list(first.command)
    changed_command[-1] = changed_command[-1].replace(
        "--optimization-mode official_adam",
        "--optimization-mode paper_normalized_direct_autograd",
    )
    changed_seed = list(first.command)
    changed_seed[-1] = changed_seed[-1].replace("--seed 1337", "--seed 1338")
    assert changed_command != list(first.command)
    assert changed_seed != list(first.command)

    mutations = {
        "arm": replace(plan, runs=(replace(first, arm_ids=("changed-arm",)), *plan.runs[1:])),
        "config": replace(
            plan,
            runs=(replace(first, config=REPO_ROOT / module.FLOW_MAP2_CONFIG), *plan.runs[1:]),
        ),
        "mode": replace(
            plan,
            runs=(replace(first, command=tuple(changed_command)), *plan.runs[1:]),
        ),
        "manifest": replace(plan, sample_manifest=REPO_ROOT / module.FULL_MANIFEST),
        "hash": replace(plan, sample_manifest_sha256="0" * 64),
        "seed": replace(
            plan,
            runs=(replace(first, command=tuple(changed_seed)), *plan.runs[1:]),
        ),
    }
    for label, changed in mutations.items():
        with pytest.raises(ValueError, match="disagrees"):
            module.ensure_campaign_contract(changed)


def test_calibration_visual_evidence_is_written_only_inside_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    plan = replace(
        module.build_matrix_plan(
            _calibration_args(module, "evidence-test"),
            semigroup_gate=_passing_gate(module),
        ),
        repo_root=tmp_path,
    )
    read_paths = []
    monkeypatch.setattr(
        module,
        "_read_jsonl",
        lambda path: read_paths.append(path) or [{"sample_id": "sample-a"}],
    )
    monkeypatch.setattr(
        module,
        "_read_json",
        lambda path, label: {"columns": ["source", "native", "candidate"], "pages": []},
    )
    monkeypatch.setattr(module, "build_visual_evidence_contract", lambda **kwargs: {})
    monkeypatch.setattr(module, "_sha256_file", lambda path: module.CALIBRATION_MANIFEST_SHA256)

    module._build_calibration_visual_evidence(plan)

    campaign_root = tmp_path / plan.campaign_root
    assert (campaign_root / "calibration/visual_evidence.json").is_file()
    assert not (tmp_path / module.CALIBRATION_VISUAL_EVIDENCE).exists()
    assert all(path.is_relative_to(campaign_root / "calibration") for path in read_paths)


def test_calibration_finalize_reads_only_campaign_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    plan = replace(
        module.build_matrix_plan(
            _calibration_args(module, "review-test"),
            semigroup_gate=_passing_gate(module),
        ),
        repo_root=tmp_path,
    )
    global_review = tmp_path / module.VISUAL_REVIEW
    global_review.parent.mkdir(parents=True)
    global_review.write_text(json.dumps({"source": "global"}), encoding="utf-8")
    monkeypatch.setattr(module, "_build_calibration_visual_evidence", lambda value: {"arms": {}})

    assert module.finalize_phase(plan) == 2

    campaign_review = tmp_path / plan.campaign_root / "visual_review.json"
    campaign_review.parent.mkdir(parents=True)
    campaign_review.write_text(json.dumps({"source": "campaign"}), encoding="utf-8")
    reviewed = []
    monkeypatch.setattr(
        module,
        "_validate_multi_arm_review",
        lambda review, evidence, require_passed: reviewed.append(review) or review,
    )

    assert module.finalize_phase(plan) == 0
    assert reviewed == [{"source": "campaign"}]


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


def test_matrix_semigroup_waits_for_direct_visual_review_before_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setattr(
        module,
        "_build_semigroup_visual_evidence",
        lambda value: {"sample_count": 64, "arms": {}},
    )

    assert module.finalize_phase(plan) == 2
    assert (tmp_path / module.SEMIGROUP_GATE_DRAFT).is_file()
    assert not (tmp_path / module.SEMIGROUP_GATE).exists()
    assert not (tmp_path / module.SCHEDULE_MANIFEST).exists()


def test_matrix_calibration_uses_one_owned_output_per_arm_and_four_gpu_queues(
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
    active_gpus: set[str] = set()
    max_active = 0

    class FakeProcess:
        def __init__(self, command, *, cwd, env, start_new_session):
            nonlocal max_active
            del command, cwd, start_new_session
            self.pid = 1000 + len(processes)
            self.returncode = 0
            self.gpu = env["CUDA_VISIBLE_DEVICES"]
            assert self.gpu not in active_gpus
            active_gpus.add(self.gpu)
            max_active = max(max_active, len(active_gpus))
            processes.append(self)

        def wait(self):
            return self.returncode

        def poll(self):
            active_gpus.remove(self.gpu)
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)

    assert module.launch_matrix(plan) == 0
    assert len(processes) == 21
    assert max_active == 4
    assert all(
        run.output_dir == plan.campaign_root / "calibration" / run.arm_ids[0]
        for run in plan.runs
    )
    assert all(len(run.arm_ids) == 1 for run in plan.runs)


def test_matrix_refills_fast_gpu_queues_without_waiting_for_gpu_zero(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source_plan = module.build_matrix_plan(
        _args(module, "--phase", "calibrate"), semigroup_gate=_passing_gate(module)
    )
    selected = []
    for gpu in range(4):
        gpu_runs = [run for run in source_plan.runs if run.physical_gpu == gpu][:4]
        for queue_index, run in enumerate(gpu_runs):
            start_path = tmp_path / f"gpu{gpu}_{queue_index}.start"
            end_path = tmp_path / f"gpu{gpu}_{queue_index}.end"
            delay = 0.9 if gpu == 0 and queue_index == 0 else (0.05 if gpu == 0 else 0.2)
            selected.append(
                replace(
                    run,
                    command=(
                        "/bin/bash",
                        "-lc",
                        f"date +%s%N > {start_path}; sleep {delay}; date +%s%N > {end_path}",
                    ),
                    output_dir=tmp_path / f"out_gpu{gpu}_{queue_index}",
                    log_path=tmp_path / f"gpu{gpu}_{queue_index}.log",
                )
            )
    plan = replace(
        source_plan,
        repo_root=tmp_path,
        runs=tuple(selected),
        status_dir=tmp_path / "status",
    )

    started = time.monotonic()
    assert module.launch_matrix(plan) == 0
    elapsed = time.monotonic() - started

    gpu0_first_end = int((tmp_path / "gpu0_0.end").read_text(encoding="utf-8"))
    assert elapsed < 1.8
    for gpu in range(1, 4):
        fast_second_start = int(
            (tmp_path / f"gpu{gpu}_1.start").read_text(encoding="utf-8")
        )
        assert fast_second_start < gpu0_first_end


def test_matrix_fast_peer_failure_immediately_terminates_slow_sessions(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source_plan = module.build_matrix_plan(_args(module, "--phase", "semigroup"))
    runs = []
    pid_paths = []
    for gpu, run in enumerate(source_plan.runs):
        if gpu == 1:
            command = ("/bin/bash", "-lc", "sleep 0.1; exit 7")
        else:
            pid_path = tmp_path / f"gpu{gpu}.child_pid"
            pid_paths.append(pid_path)
            command = (
                "/bin/bash",
                "-lc",
                f"sleep 5 & echo $! > {pid_path}; wait",
            )
        runs.append(
            replace(
                run,
                command=command,
                output_dir=tmp_path / f"out_gpu{gpu}",
                log_path=tmp_path / f"gpu{gpu}.log",
            )
        )
    plan = replace(
        source_plan,
        repo_root=tmp_path,
        runs=tuple(runs),
        status_dir=tmp_path / "status",
    )

    started = time.monotonic()
    assert module.launch_matrix(plan) == 1
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    status = json.loads((tmp_path / "status/matrix_status.json").read_text(encoding="utf-8"))
    assert status["runs"][1]["exit_code"] == 7
    for pid_path in pid_paths:
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 1.0
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{child_pid}").exists()


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
        "winner": {
            "arm_id": "winner",
            "config": str(module.NOISE_CONFIGS[0]),
            "arm_config_sha256": canonical_arm_config_digest(
                yaml.safe_load((REPO_ROOT / module.NOISE_CONFIGS[0]).read_text())
            ),
        },
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


def test_matrix_calibration_quality_reuse_is_contract_validated() -> None:
    module = _load_script()
    plan = module.build_matrix_plan(
        _args(module, "--phase", "calibrate"), semigroup_gate=_passing_gate(module)
    )

    for run in plan.runs:
        command = " ".join(run.command)
        assert "--reuse-valid-output" in command
        assert "--generation-result" in command
        assert "/quality.json; then :; else" not in command


def test_matrix_calibration_resume_contract_is_uniform_across_all_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    plan = module.build_matrix_plan(
        _args(module, "--phase", "calibrate"), semigroup_gate=_passing_gate(module)
    )
    selected = []
    for family in (
        "official_flow_map1",
        "official_flow_map2",
        "paper_algorithm_split",
        "initial_noise_oracle",
    ):
        run = next(candidate for candidate in plan.runs if candidate.family == family)
        output = tmp_path / run.output_dir
        output.mkdir(parents=True)
        (output / "resume_contract.json").write_text("{}", encoding="utf-8")
        selected.append(run)
    resumed_plan = replace(plan, repo_root=tmp_path, runs=tuple(selected))

    module.validate_artifact_paths(resumed_plan)

    gpu3 = next(run for run in selected if run.physical_gpu == 3)
    command = " ".join(gpu3.command)
    assert f"{gpu3.output_dir}/completion.json" in command
    assert "--reuse-valid-output" in command
    assert gpu3.output_dir == plan.campaign_root / "calibration" / gpu3.arm_ids[0]
    finalized_arms: list[str] = []

    def fake_evidence(value):
        finalized_arms.extend(arm_id for run in value.runs for arm_id in run.arm_ids)
        return {"arms": {}}

    monkeypatch.setattr(module, "_build_calibration_visual_evidence", fake_evidence)
    assert module.finalize_phase(resumed_plan) == 2
    assert gpu3.arm_ids[0] in finalized_arms


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

        def poll(self):
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
            "arm_config_sha256": "f" * 64,
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


@pytest.mark.parametrize("winner_source_name", ["FLOW_MAP1_CONFIG", "PAPER_CONFIG"])
def test_materialized_full_fmrg_keeps_semigroup_schedule_manifest_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    winner_source_name: str,
) -> None:
    module = _load_script()
    winner_source = getattr(module, winner_source_name)
    for relative in (module.NATIVE_CONFIG, winner_source):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((REPO_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    semigroup_manifest = tmp_path / module.CALIBRATION_MANIFEST
    semigroup_manifest.parent.mkdir(parents=True, exist_ok=True)
    semigroup_manifest.write_text('{"sample_id":"semigroup-0"}\n', encoding="utf-8")
    report_path = tmp_path / module.SEMIGROUP_GATE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "gate_passed": True,
                "checkpoint_sha256": module.CHECKPOINT_SHA256,
                "selected_t_cut": 0.25,
            }
        ),
        encoding="utf-8",
    )
    schedule = {
        "schema_version": 2,
        "gate_passed": True,
        "checkpoint_sha256": module.CHECKPOINT_SHA256,
        "semigroup_report": str(module.SEMIGROUP_GATE),
        "semigroup_report_sha256": module._sha256_file(report_path),
        "semigroup_sample_id_manifest": str(module.CALIBRATION_MANIFEST),
        "semigroup_sample_id_manifest_sha256": module._sha256_file(semigroup_manifest),
        "t_cut": 0.25,
        "guided_steps": 3,
        "guided_times": [1.0, 0.75, 0.5, 0.25],
        "unguided_tail_intervals": 2,
        "unguided_times": [0.25, 0.125, 0.0],
        "selection_rule": "test",
    }
    schedule["schedule_contract_sha256"] = module._schedule_contract_digest(schedule)
    schedule_path = tmp_path / module.SCHEDULE_MANIFEST
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    winner_config = yaml.safe_load((tmp_path / winner_source).read_text(encoding="utf-8"))
    winner_config.update(
        {
            "t_cut": 0.25,
            "schedule_manifest": str(module.SCHEDULE_MANIFEST),
            "semigroup_report": str(module.SEMIGROUP_GATE),
            "semigroup_sample_id_manifest": str(module.CALIBRATION_MANIFEST),
            "semigroup_sample_id_manifest_sha256": module._sha256_file(
                semigroup_manifest
            ),
            "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        }
    )
    locked_path = tmp_path / "locked_winner.yaml"
    locked_path.write_text(yaml.safe_dump(winner_config, sort_keys=False), encoding="utf-8")
    winner = {
        "arm_id": "locked-winner",
        "config": str(locked_path.relative_to(tmp_path)),
        "config_sha256": module._sha256_file(locked_path),
        "arm_config_sha256": canonical_arm_config_digest(winner_config),
        "mode": winner_config["mode"],
        "t_cut": 0.25,
        "schedule_manifest": str(module.SCHEDULE_MANIFEST),
        "schedule_manifest_sha256": module._sha256_file(schedule_path),
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
    }
    args = _args(module, "--phase", "full")
    args.repo_root = tmp_path
    plan = module.build_matrix_plan(
        args,
        full_contract={
            "winner": winner,
            "visual_review": {"reviewed_sample_count": 64, "passed": True},
            "manifest_count": 2048,
            "manifest_sha256": module.FULL_MANIFEST_SHA256,
        },
    )
    module.materialize_full_runtime_configs(plan)
    runtime = yaml.safe_load(
        (tmp_path / module.ROOT / "full/runtime_configs/winner.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert runtime["sample_id_manifest"] == str(module.FULL_MANIFEST)
    assert runtime["semigroup_sample_id_manifest"] == str(module.CALIBRATION_MANIFEST)
    assert runtime["arm_config_sha256"] == winner["arm_config_sha256"]
    assert canonical_arm_config_digest(runtime) == winner["arm_config_sha256"]
    monkeypatch.chdir(tmp_path)
    resolved = resolve_locked_schedule(
        runtime, checkpoint_sha256=module.CHECKPOINT_SHA256, explicit_t_cut=0.25
    )
    assert resolved["semigroup_sample_id_manifest_sha256"] == module._sha256_file(
        semigroup_manifest
    )
    semigroup_manifest.write_text('{"sample_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="semigroup sample manifest SHA256"):
        resolve_locked_schedule(
            runtime, checkpoint_sha256=module.CHECKPOINT_SHA256, explicit_t_cut=0.25
        )


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
    assert schedule["semigroup_sample_id_manifest_sha256"] == module._sha256_file(
        manifest
    )
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


@pytest.mark.parametrize(
    "campaign_ids",
    [("campaign-a", "campaign-b"), ("same-campaign", "same-campaign")],
    ids=("different-campaigns", "same-campaign"),
)
def test_kernel_gpu_leases_admit_only_one_process_before_query(
    tmp_path: Path, campaign_ids: tuple[str, str]
) -> None:
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    release = context.Event()
    events = context.Queue()
    repo_root = tmp_path / "repo"
    host_lock_dir = tmp_path / "host-locks"
    processes = [
        context.Process(
            target=_kernel_lock_execute_worker,
            args=(
                str(repo_root),
                str(host_lock_dir),
                campaign_id,
                barrier,
                release,
                events,
                "hold_preflight",
            ),
        )
        for campaign_id in campaign_ids
    ]
    for process in processes:
        process.start()
    try:
        before_release = _collect_process_events(events, 2)
        assert [event[2] for event in before_release].count("query") == 1
        assert [event[2] for event in before_release].count("rejected") == 1
    finally:
        release.set()
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
    assert [process.exitcode for process in processes] == [0, 0]
    after_release = _collect_process_events(events, 2)
    assert [event[2] for event in after_release] == ["launch", "done"]


def test_kernel_leases_recover_after_holder_process_abrupt_exit(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    repo_root = tmp_path / "repo"
    host_lock_dir = tmp_path / "host-locks"
    release = context.Event()
    release.set()
    crash_events = context.Queue()
    crashed = context.Process(
        target=_kernel_lock_execute_worker,
        args=(
            str(repo_root),
            str(host_lock_dir),
            "recover-campaign",
            context.Barrier(1),
            release,
            crash_events,
            "crash_launch",
        ),
    )
    crashed.start()
    crashed.join(timeout=10.0)
    if crashed.is_alive():
        crashed.kill()
        crashed.join(timeout=5.0)
    assert crashed.exitcode == 17

    recovered_events = context.Queue()
    recovered = context.Process(
        target=_kernel_lock_execute_worker,
        args=(
            str(repo_root),
            str(host_lock_dir),
            "recover-campaign",
            context.Barrier(1),
            release,
            recovered_events,
            "normal",
        ),
    )
    recovered.start()
    recovered.join(timeout=10.0)
    if recovered.is_alive():
        recovered.kill()
        recovered.join(timeout=5.0)
    assert recovered.exitcode == 0
    stages = [event[2] for event in _collect_process_events(recovered_events, 3)]
    assert stages == ["query", "launch", "done"]


def test_kernel_lease_partial_acquisition_failure_releases_earlier_leases(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    host_lock_dir = tmp_path / "host-locks"
    ready = context.Event()
    release = context.Event()
    holder_events = context.Queue()
    holder = context.Process(
        target=_kernel_lock_hold_gpu_worker,
        args=(str(host_lock_dir), 1, ready, release, holder_events),
    )
    holder.start()
    try:
        assert ready.wait(10.0)
        assert holder_events.get(timeout=10.0) == ("held", 1)
        module = _load_script()
        module.HOST_GPU_LOCK_DIR = host_lock_dir
        plan = _campaign_plan_in_tmp(module, tmp_path / "repo")
        gpu_runs = tuple(
            next(run for run in plan.runs if run.physical_gpu == gpu)
            for gpu in (0, 1)
        )
        plan = replace(plan, runs=gpu_runs)
        with pytest.raises(RuntimeError, match="physical GPU 1 lease"):
            with module.execution_leases(plan):
                pytest.fail("held GPU lease was acquired")

        probe_events = context.Queue()
        probe = context.Process(
            target=_kernel_lock_probe_worker,
            args=(
                str(tmp_path / "repo"),
                str(host_lock_dir),
                str(plan.campaign_id),
                probe_events,
            ),
        )
        probe.start()
        probe.join(timeout=10.0)
        if probe.is_alive():
            probe.kill()
            probe.join(timeout=5.0)
        assert probe.exitcode == 0
        assert probe_events.get(timeout=10.0)[0] == "acquired"
    finally:
        release.set()
        holder.join(timeout=10.0)
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=5.0)
    assert holder.exitcode == 0


@pytest.mark.parametrize("lock_kind", ["campaign-leaf", "gpu-directory", "gpu-leaf"])
def test_kernel_lease_paths_reject_symlinks(tmp_path: Path, lock_kind: str) -> None:
    module = _load_script()
    module.HOST_GPU_LOCK_DIR = tmp_path / "host-locks"
    repo_root = tmp_path / "repo"
    plan = replace(_campaign_plan_in_tmp(module, repo_root), execute=True)
    outside_file = tmp_path / "outside.lock"
    outside_file.write_text("unchanged", encoding="utf-8")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    if lock_kind == "campaign-leaf":
        lock_path = repo_root / plan.lock_path
        lock_path.parent.mkdir(parents=True)
        lock_path.symlink_to(outside_file)
    elif lock_kind == "gpu-directory":
        module.HOST_GPU_LOCK_DIR.symlink_to(outside_dir, target_is_directory=True)
    else:
        module.HOST_GPU_LOCK_DIR.mkdir()
        (module.HOST_GPU_LOCK_DIR / "gpu0.lock").symlink_to(outside_file)

    with pytest.raises(ValueError, match="symlink"):
        with module.execution_leases(plan):
            pytest.fail("symlinked lock path was acquired")

    assert outside_file.read_text(encoding="utf-8") == "unchanged"
    assert list(outside_dir.iterdir()) == []


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


def test_execute_plan_keyboard_interrupt_reaps_session_before_unlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    module.HOST_GPU_LOCK_DIR = tmp_path / "host-locks"
    child_pid_path = tmp_path / "child.pid"
    run = replace(
        module.build_matrix_plan(_args(module, "--phase", "semigroup")).runs[0],
        command=(
            "/bin/bash",
            "-lc",
            f"sleep 60 & echo $! > {child_pid_path}; wait",
        ),
        output_dir=Path("owned"),
        log_path=Path("logs/run.log"),
    )
    plan = replace(
        module.build_matrix_plan(_args(module, "--phase", "semigroup", "--execute")),
        repo_root=tmp_path,
        runs=(run,),
        status_dir=Path("status"),
        lock_path=Path("matrix.lock"),
    )
    real_popen = subprocess.Popen

    class InterruptingProcess:
        def __init__(self, command, *, cwd, env, start_new_session):
            self._process = real_popen(
                command, cwd=cwd, env=env, start_new_session=start_new_session
            )
            self.pid = self._process.pid
            self.interrupted = False

        def wait(self, *args, **kwargs):
            return self._process.wait(*args, **kwargs)

        def poll(self):
            if not self.interrupted:
                self.interrupted = True
                deadline = time.monotonic() + 5.0
                while not child_pid_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert (tmp_path / "matrix.lock").is_file()
                raise KeyboardInterrupt
            return self._process.poll()

        def terminate(self):
            self._process.terminate()

        def kill(self):
            self._process.kill()

    monkeypatch.setattr(module.subprocess, "Popen", InterruptingProcess)
    monkeypatch.setattr(module, "validate_artifact_paths", lambda value: None)
    monkeypatch.setattr(module, "validate_preflight", lambda value: value)
    monkeypatch.setattr(module, "materialize_locked_manifests", lambda value: None)
    monkeypatch.setattr(module, "materialize_full_runtime_configs", lambda value: None)

    with pytest.raises(KeyboardInterrupt):
        module.execute_plan(plan)

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()
    assert (tmp_path / "matrix.lock").is_file()
    with module.execution_leases(plan):
        pass


def test_execute_plan_marks_finalize_failure_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    module.HOST_GPU_LOCK_DIR = tmp_path / "host-locks"
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
    native_config = {"mode": "native", "sampling_seed": 1337}
    native_digest = canonical_arm_config_digest(native_config)
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
            json.dumps(
                {
                    "sample_id": sample_id,
                    "generated": str(generated),
                    "candidate_cosine": 0.5,
                    "native_cosine": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (shard / "completion.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "sample_count": 1,
                    "arm_config_sha256": native_digest,
                }
            ),
            encoding="utf-8",
        )
        (shard / "generation_result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": "native",
                    "checkpoint": {
                        "sha256": module.CHECKPOINT_SHA256,
                        "stage": "stage2",
                        "stage_epoch_1based": 1652,
                        "sit_patch_size": 4,
                        "weight_source": "ema_model_state_dict",
                    },
                    "sample_count": 1,
                    "sample_id_sha256": module._sample_id_digest([sample_id]),
                    "arm_config_sha256": native_digest,
                    "schedule": None,
                    "config": native_config,
                }
            ),
            encoding="utf-8",
        )
    combined = tmp_path / module.ROOT / "full/merged/native"

    first = module._merge_full_arm(
        tmp_path, "native", combined, expected_arm_config_sha256=native_digest
    )
    second = module._merge_full_arm(
        tmp_path, "native", combined, expected_arm_config_sha256=native_digest
    )
    assert first == second
    completion = json.loads((combined / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "complete"

    (combined / "completion.json").unlink()
    victim = next((combined / "generated_images").iterdir())
    expected = victim.read_bytes()
    victim.write_bytes(expected[: max(1, len(expected) // 2)])
    resumed = module._merge_full_arm(
        tmp_path, "native", combined, expected_arm_config_sha256=native_digest
    )
    assert resumed["status"] == "complete"
    assert victim.read_bytes() == expected


def test_full_merge_rejects_symlink_escape_and_unknown_output(tmp_path: Path) -> None:
    module = _load_script()
    native_config = {"mode": "native", "sampling_seed": 1337}
    digest = canonical_arm_config_digest(native_config)
    manifest = tmp_path / module.FULL_MANIFEST
    manifest.parent.mkdir(parents=True)
    ids = [f"sample-{index}" for index in range(4)]
    manifest.write_text(
        "".join(json.dumps({"sample_id": value}) + "\n" for value in ids),
        encoding="utf-8",
    )
    for gpu, sample_id in enumerate(ids):
        shard = tmp_path / module.ROOT / f"full/shards/shard_{gpu}/native"
        shard.mkdir(parents=True)
        image = shard / f"{gpu}.png"
        image.write_bytes(b"image")
        (shard / "per_sample.jsonl").write_text(
            json.dumps({"sample_id": sample_id, "generated": str(image), "candidate_cosine": 0.5, "native_cosine": 0.5}) + "\n",
            encoding="utf-8",
        )
        (shard / "completion.json").write_text(
            json.dumps({"status": "complete", "sample_count": 1, "arm_config_sha256": digest}),
            encoding="utf-8",
        )
        (shard / "generation_result.json").write_text(
            json.dumps({
                "status": "complete", "mode": "native", "checkpoint": {"sha256": module.CHECKPOINT_SHA256},
                "sample_count": 1, "sample_id_sha256": module._sample_id_digest([sample_id]),
                "arm_config_sha256": digest, "schedule": None, "config": native_config,
            }),
            encoding="utf-8",
        )
    combined = tmp_path / module.ROOT / "full/merged/native"
    outside = tmp_path / "outside"
    outside.mkdir()
    combined.mkdir(parents=True)
    (combined / "generated_images").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module._merge_full_arm(tmp_path, "native", combined, expected_arm_config_sha256=digest)
    assert not list(outside.iterdir())

    (combined / "generated_images").unlink()
    (combined / "unknown.txt").write_text("unknown", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError), match="unowned"):
        module._merge_full_arm(tmp_path, "native", combined, expected_arm_config_sha256=digest)


@pytest.mark.parametrize(
    ("config_index", "field", "old_value"),
    [(0, "step_size", 3.0), (0, "eta", 0.5), (2, "typical_delta", 0.1)],
)
def test_full_preflight_rejects_old_winner_completion_with_different_algorithm_field(
    tmp_path: Path, config_index: int, field: str, old_value: float
) -> None:
    module = _load_script()
    config_path = module.NOISE_CONFIGS[config_index]
    locked_config = yaml.safe_load((REPO_ROOT / config_path).read_text())
    locked_digest = canonical_arm_config_digest(locked_config)
    old_digest = canonical_arm_config_digest({**locked_config, field: old_value})
    assert old_digest != locked_digest
    contract = {
        "winner": {
            "arm_id": "noise-winner",
            "config": str(config_path),
            "arm_config_sha256": locked_digest,
        },
        "visual_review": {"reviewed_sample_count": 64, "passed": True},
        "manifest_count": 2048,
        "manifest_sha256": module.FULL_MANIFEST_SHA256,
    }
    plan = module.build_matrix_plan(
        _args(module, "--phase", "full"), full_contract=contract
    )
    output = tmp_path / "shard_0"
    winner = output / "winner"
    winner.mkdir(parents=True)
    (winner / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sample_count": 512,
                "arm_config_sha256": old_digest,
            }
        ),
        encoding="utf-8",
    )
    plan = replace(plan, runs=(replace(plan.runs[0], output_dir=output),))

    with pytest.raises(ValueError, match="locked winner arm config"):
        module.validate_artifact_paths(plan)
