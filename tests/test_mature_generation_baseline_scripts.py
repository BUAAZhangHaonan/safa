from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
K100_SCRIPT = REPO_ROOT / "scripts" / "k100" / "run_mature_generation_baselines_k100.sh"
H100_SCRIPT = REPO_ROOT / "scripts" / "h100" / "run_mature_generation_baselines_ddp_h100.sh"

E16 = "e16_meanflow_sit_l2_face_mixed_2400ep"
E19 = "e19_meanflow_sit_b2_face_mixed_2400ep"


def _copy_mature_config_tree(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    experiments = repo_root / "configs" / "medium_v2" / "experiments"
    experiments.mkdir(parents=True)
    for name in (E16, E19):
        shutil.copy2(
            REPO_ROOT / "configs" / "medium_v2" / "experiments" / f"{name}.yaml",
            experiments / f"{name}.yaml",
        )
    return repo_root


def _run(script: Path, repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["SAFA_E16_PATTERN"] = "safa_no_mature_script_e16_marker_for_test"
    return subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(repo_root),
            "--timestamp",
            "20260702_120000",
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


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _runtime(repo_root: Path, name: str, suffix: str) -> Path:
    return repo_root / "configs" / "medium_v2" / "experiments" / f"{name}{suffix}.yaml"


def test_k100_mature_dry_run_writes_runtime_resume_and_lists_only_mature_family(tmp_path: Path) -> None:
    repo_root = _copy_mature_config_tree(tmp_path)
    e16_last = repo_root / "artifacts" / "checkpoints" / E16 / "last.pt"
    e16_last.parent.mkdir(parents=True)
    e16_last.write_bytes(b"checkpoint")

    result = _run(K100_SCRIPT, repo_root)

    assert result.returncode == 0, result.stderr
    assert "DRY RUN: mature MeanFlow-SiT K100 queue" in result.stdout
    assert result.stdout.index(E16) < result.stdout.index(E19)
    assert "e17_sit_diffusion" not in result.stdout
    assert "e20_rectified_flow" not in result.stdout
    assert "-m safa.cli.train_g --config configs/medium_v2/experiments/e16_meanflow_sit_l2_face_mixed_2400ep_k100_runtime.yaml" in result.stdout

    e16_runtime = _load_yaml(_runtime(repo_root, E16, "_k100_runtime"))
    assert e16_runtime["resume_from"] == f"artifacts/checkpoints/{E16}/last.pt"
    assert e16_runtime["resume_mode"] == "training_state"
    assert e16_runtime["resume_optimizer_state"] is True
    assert e16_runtime["per_device_batch_size"] == 32
    assert e16_runtime["global_batch_size"] == 32
    assert e16_runtime["distributed"]["backend"] == "gloo"

    e19_runtime = _load_yaml(_runtime(repo_root, E19, "_k100_runtime"))
    assert e19_runtime["resume_from"] == ""
    assert e19_runtime["resume_mode"] == "model_weights_only"
    assert e19_runtime["resume_optimizer_state"] is False


def test_k100_mature_one_selects_single_config(tmp_path: Path) -> None:
    repo_root = _copy_mature_config_tree(tmp_path)

    result = _run(K100_SCRIPT, repo_root, "--one", E19)

    assert result.returncode == 0, result.stderr
    assert f"runtime_config: configs/medium_v2/experiments/{E19}_k100_runtime.yaml" in result.stdout
    assert f"{E16}_k100_runtime.yaml" not in result.stdout
    assert _runtime(repo_root, E19, "_k100_runtime").exists()
    assert not _runtime(repo_root, E16, "_k100_runtime").exists()


def test_h100_mature_default_skips_current_e16_and_include_e16_opt_in(tmp_path: Path) -> None:
    repo_root = _copy_mature_config_tree(tmp_path)
    e19_last = repo_root / "artifacts" / "checkpoints" / E19 / "last.pt"
    e19_last.parent.mkdir(parents=True)
    e19_last.write_bytes(b"checkpoint")

    default = _run(H100_SCRIPT, repo_root)

    assert default.returncode == 0, default.stderr
    assert "DRY RUN: mature MeanFlow-SiT H100 DDP queue" in default.stdout
    assert "skip-current: e16_meanflow_sit_l2_face_mixed_2400ep" in default.stdout
    assert f"runtime_config: configs/medium_v2/experiments/{E19}_h100_mature_ddp_runtime.yaml" in default.stdout
    assert f"{E16}_h100_mature_ddp_runtime.yaml" not in default.stdout
    assert "torchrun --standalone --nproc_per_node=4" in default.stdout

    e19_runtime = _load_yaml(_runtime(repo_root, E19, "_h100_mature_ddp_runtime"))
    assert e19_runtime["resume_from"] == f"artifacts/checkpoints/{E19}/last.pt"
    assert e19_runtime["resume_mode"] == "training_state"
    assert e19_runtime["resume_optimizer_state"] is True
    assert e19_runtime["distributed"]["backend"] == "nccl"
    assert e19_runtime["per_device_batch_size"] == 32
    assert e19_runtime["global_batch_size"] == 128

    include = _run(H100_SCRIPT, _copy_mature_config_tree(tmp_path / "include"), "--include-e16")

    assert include.returncode == 0, include.stderr
    assert f"runtime_config: configs/medium_v2/experiments/{E16}_h100_mature_ddp_runtime.yaml" in include.stdout
    assert include.stdout.index(E19) < include.stdout.index(E16)


def test_h100_mature_one_selects_e16_even_without_include_flag(tmp_path: Path) -> None:
    repo_root = _copy_mature_config_tree(tmp_path)

    result = _run(H100_SCRIPT, repo_root, "--one", E16)

    assert result.returncode == 0, result.stderr
    assert f"runtime_config: configs/medium_v2/experiments/{E16}_h100_mature_ddp_runtime.yaml" in result.stdout
    assert f"{E19}_h100_mature_ddp_runtime.yaml" not in result.stdout
