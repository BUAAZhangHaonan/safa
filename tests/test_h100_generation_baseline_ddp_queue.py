from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "h100" / "run_generation_baseline_ddp_h100.sh"

B2_CONFIGS = (
    "e22_sit_diffusion_b2_face_mixed_2400ep",
    "e23_latent_consistency_b2_face_mixed_2400ep",
    "e19_meanflow_sit_b2_face_mixed_2400ep",
    "e20_rectified_flow_sit_b2_face_mixed_2400ep",
)
L2_CONFIGS = (
    "e17_sit_diffusion_l2_face_mixed_2400ep",
    "e18_latent_consistency_l2_face_mixed_2400ep",
    "e21_rectified_flow_sit_l2_face_mixed_2400ep",
)
QUEUE_ORDER = (*B2_CONFIGS[:2], *B2_CONFIGS[2:], *L2_CONFIGS)


def _copy_config_tree(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    experiments = repo_root / "configs" / "medium_v2" / "experiments"
    experiments.mkdir(parents=True)
    for name in QUEUE_ORDER:
        shutil.copy2(
            REPO_ROOT / "configs" / "medium_v2" / "experiments" / f"{name}.yaml",
            experiments / f"{name}.yaml",
        )
    return repo_root


def _run_queue(repo_root: Path, *args: str, marker: str = "safa_no_h100_ddp_e16_marker_for_test") -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["SAFA_E16_PATTERN"] = marker
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo_root),
            "--timestamp",
            "20260701_120000",
            "--python",
            sys.executable,
            *args,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _runtime_path(repo_root: Path, name: str) -> Path:
    return repo_root / "configs" / "medium_v2" / "experiments" / f"{name}_h100_ddp_runtime.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_dry_run_prints_runtime_configs_batches_commands_and_artifacts(tmp_path: Path) -> None:
    repo_root = _copy_config_tree(tmp_path)

    result = _run_queue(repo_root)

    assert result.returncode == 0, result.stderr
    assert "DRY RUN: pass --run to start training" in result.stdout
    positions = [result.stdout.index(name) for name in QUEUE_ORDER]
    assert positions == sorted(positions)

    for name in QUEUE_ORDER:
        runtime_config = f"configs/medium_v2/experiments/{name}_h100_ddp_runtime.yaml"
        assert f"runtime_config: {runtime_config}" in result.stdout
        assert "command: " in result.stdout
        assert "torchrun --standalone --nproc_per_node=4" in result.stdout
        assert f"--config {runtime_config}" in result.stdout
        assert f"log: artifacts/logs/{name}_h100_ddp_20260701_120000.log" in result.stdout
        assert f"pid: artifacts/run/{name}_h100_ddp_20260701_120000.pid" in result.stdout
        assert f"status: artifacts/run/{name}_h100_ddp_20260701_120000.status" in result.stdout

    assert result.stdout.count("batch: per_device=32 global=128") == len(B2_CONFIGS)
    assert result.stdout.count("batch: per_device=16 global=64") == len(L2_CONFIGS)


def test_dry_run_writes_runtime_yaml_without_mutating_source_configs(tmp_path: Path) -> None:
    repo_root = _copy_config_tree(tmp_path)
    source_b2 = repo_root / "configs" / "medium_v2" / "experiments" / f"{B2_CONFIGS[0]}.yaml"
    source_l2 = repo_root / "configs" / "medium_v2" / "experiments" / f"{L2_CONFIGS[0]}.yaml"
    before_b2 = source_b2.read_text(encoding="utf-8")
    before_l2 = source_l2.read_text(encoding="utf-8")

    result = _run_queue(repo_root)

    assert result.returncode == 0, result.stderr
    assert source_b2.read_text(encoding="utf-8") == before_b2
    assert source_l2.read_text(encoding="utf-8") == before_l2

    b2_runtime = _load_yaml(_runtime_path(repo_root, B2_CONFIGS[0]))
    assert b2_runtime["device"] == "cuda:0"
    assert b2_runtime["distributed"]["backend"] == "nccl"
    assert b2_runtime["per_device_batch_size"] == 32
    assert b2_runtime["global_batch_size"] == 128
    assert b2_runtime["num_workers"] == 8
    assert b2_runtime["validation"]["batch_size"] == 16
    b2_quality = b2_runtime["stages"]["stage2"]["quality_eval"]
    assert b2_quality["output_dir"].endswith("/quality_h100_ddp")
    assert b2_quality["distribution_cuda_visible_devices"] == "0"
    assert b2_quality["distribution_device"] == "cuda:0"
    assert b2_runtime["train_index"] == _load_yaml(source_b2)["train_index"]
    assert b2_runtime["train_features"] == _load_yaml(source_b2)["train_features"]

    l2_runtime = _load_yaml(_runtime_path(repo_root, L2_CONFIGS[0]))
    assert l2_runtime["per_device_batch_size"] == 16
    assert l2_runtime["global_batch_size"] == 64


def test_e16_guard_exits_2_and_skip_e16_check_bypasses_it(tmp_path: Path) -> None:
    repo_root = _copy_config_tree(tmp_path)
    marker = "safa_h100_ddp_generation_queue_e16_marker_for_test"
    process = subprocess.Popen(["bash", "-lc", f"exec -a '{marker}' sleep 60"])
    try:
        time.sleep(0.2)
        guarded = _run_queue(repo_root, marker=marker)
        skipped = _run_queue(repo_root, "--skip-e16-check", marker=marker)
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert guarded.returncode == 2
    assert "E16 training is still running" in guarded.stdout
    assert marker in guarded.stdout
    assert skipped.returncode == 0, skipped.stderr
    assert "DRY RUN: pass --run to start training" in skipped.stdout


def test_one_selects_single_config_by_name_or_config_path(tmp_path: Path) -> None:
    repo_root_by_name = _copy_config_tree(tmp_path / "by_name")
    by_name = _run_queue(repo_root_by_name, "--one", "e18_latent_consistency_l2_face_mixed_2400ep")

    assert by_name.returncode == 0, by_name.stderr
    assert "runtime_config: configs/medium_v2/experiments/e18_latent_consistency_l2_face_mixed_2400ep_h100_ddp_runtime.yaml" in by_name.stdout
    assert "e22_sit_diffusion_b2_face_mixed_2400ep_h100_ddp_runtime.yaml" not in by_name.stdout
    assert _runtime_path(repo_root_by_name, "e18_latent_consistency_l2_face_mixed_2400ep").exists()
    assert not _runtime_path(repo_root_by_name, "e22_sit_diffusion_b2_face_mixed_2400ep").exists()

    repo_root_by_path = _copy_config_tree(tmp_path / "by_path")
    config_path = "configs/medium_v2/experiments/e20_rectified_flow_sit_b2_face_mixed_2400ep.yaml"
    by_path = _run_queue(repo_root_by_path, "--one", config_path)

    assert by_path.returncode == 0, by_path.stderr
    assert "runtime_config: configs/medium_v2/experiments/e20_rectified_flow_sit_b2_face_mixed_2400ep_h100_ddp_runtime.yaml" in by_path.stdout
    assert "batch: per_device=32 global=128" in by_path.stdout
    assert "e21_rectified_flow_sit_l2_face_mixed_2400ep_h100_ddp_runtime.yaml" not in by_path.stdout


def test_command_uses_requested_nproc_and_runtime_config(tmp_path: Path) -> None:
    repo_root = _copy_config_tree(tmp_path)

    result = _run_queue(repo_root, "--nproc-per-node", "2", "--one", "e22_sit_diffusion_b2_face_mixed_2400ep")

    assert result.returncode == 0, result.stderr
    assert "torchrun --standalone --nproc_per_node=2" in result.stdout
    assert "batch: per_device=32 global=128" in result.stdout
    assert "--config configs/medium_v2/experiments/e22_sit_diffusion_b2_face_mixed_2400ep_h100_ddp_runtime.yaml" in result.stdout
