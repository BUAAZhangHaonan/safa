from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "external" / "verify_generation_baseline_weights.py"


def _run_verify(root: Path, manifest: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--manifest", str(manifest), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def test_verify_generation_baseline_weights_require_existing_all_fails_when_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"

    result = _run_verify(tmp_path, manifest, "--require-existing", "all")

    assert result.returncode != 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["missing_required"] == ["meanflow_sit_b2", "meanflow_sit_l2"]


def test_verify_generation_baseline_weights_manifest_records_sha256(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    weight_dir = tmp_path / "artifacts" / "checkpoints" / "external" / "meanflow_sit"
    weight_dir.mkdir(parents=True)
    checkpoint = weight_dir / "zhuyu_sit_b_2_imagenet256.pt"
    torch.save(
        {
            "model": {
                "pos_embed": torch.zeros(1, 256, 768),
                "x_embedder.proj.weight": torch.zeros(768, 4, 2, 2),
                "blocks.11.attn.qkv.weight": torch.zeros(2304, 768),
                "final_layer.linear.weight": torch.zeros(16, 768),
            }
        },
        checkpoint,
    )
    manifest = tmp_path / "manifest.json"

    result = _run_verify(tmp_path, manifest, "--require-existing", "meanflow_sit_b2")

    assert result.returncode == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    b2 = {item["name"]: item for item in payload["meanflow_sit_weights"]}["meanflow_sit_b2"]
    assert b2["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
