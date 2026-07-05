from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "k100" / "run_generation_baseline_queue.sh"


def _run_queue(*args: str, marker: str = "safa_no_e16_marker_for_test") -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["SAFA_E16_PATTERN"] = marker
    return subprocess.run(
        ["bash", str(SCRIPT), "--repo-root", str(REPO_ROOT), "--timestamp", "20260630_120000", "--python", "python", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def test_generation_baseline_queue_refuses_without_ablation_only() -> None:
    result = _run_queue()

    assert result.returncode == 64
    assert "internal ablation" in result.stderr
    assert "--ablation-only" in result.stderr
    assert "not paper main-table mature baselines" in result.stderr


def test_generation_baseline_queue_ablation_only_dry_run_lists_order() -> None:
    result = _run_queue("--ablation-only")

    assert result.returncode == 0
    assert "DRY RUN: internal ablation queue" in result.stdout
    expected_order = [
        "e22_sit_diffusion_b2_face_mixed_2400ep",
        "e23_latent_consistency_b2_face_mixed_2400ep",
        "e19_meanflow_sit_b2_face_mixed_2400ep",
        "e20_rectified_flow_sit_b2_face_mixed_2400ep",
        "e17_sit_diffusion_l2_face_mixed_2400ep",
        "e18_latent_consistency_l2_face_mixed_2400ep",
        "e21_rectified_flow_sit_l2_face_mixed_2400ep",
    ]
    positions = [result.stdout.index(name) for name in expected_order]
    assert positions == sorted(positions)
    assert "python -m safa.cli.train_g --config configs/medium_v2/experiments/e22_sit_diffusion_b2_face_mixed_2400ep.yaml" in result.stdout
    assert "artifacts/logs/e22_sit_diffusion_b2_face_mixed_2400ep_20260630_120000.log" in result.stdout
    assert "artifacts/run/e22_sit_diffusion_b2_face_mixed_2400ep_20260630_120000.pid" in result.stdout


def test_generation_baseline_queue_exits_when_e16_process_is_running() -> None:
    marker = "safa_generation_queue_e16_marker_for_test"
    process = subprocess.Popen(["bash", "-lc", f"exec -a '{marker}' sleep 60"])
    try:
        time.sleep(0.2)
        result = _run_queue("--ablation-only", marker=marker)
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert result.returncode == 2
    assert "E16 training is still running" in result.stdout
    assert marker in result.stdout
