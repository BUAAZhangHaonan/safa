from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "external" / "verify_generation_baseline_weights.py"
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "external" / "prepare_generation_baselines.sh"


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


def test_prepare_generation_baselines_prefers_k100_conda_python_before_path_python() -> None:
    text = PREPARE_SCRIPT.read_text(encoding="utf-8")

    assert 'PREFERRED_PYTHON="/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python"' in text
    assert text.index('if [[ -x "${PREFERRED_PYTHON}" ]]') < text.index("command -v python3")


def test_prepare_generation_baselines_uses_python_module_gdown_before_cli(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "external"
    script_dir.mkdir(parents=True)
    script_path = script_dir / "prepare_generation_baselines.sh"
    shutil.copy2(PREPARE_SCRIPT, script_path)
    (script_dir / "verify_generation_baseline_weights.py").write_text("", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == "clone" ]]; then\n'
        '  mkdir -p "${5}/.git"\n'
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    fake_python = bin_dir / "python-with-gdown-module"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${PYTHON_CALL_LOG}"\n'
        'if [[ "${1:-}" == "-m" && "${2:-}" == "gdown" ]]; then\n'
        '  if [[ "${3:-}" == "--help" ]]; then\n'
        "    exit 0\n"
        "  fi\n"
        '  out=""\n'
        "  while [[ $# -gt 0 ]]; do\n"
        '    if [[ "$1" == "-O" ]]; then\n'
        "      shift\n"
        '      out="$1"\n'
        "    fi\n"
        "    shift || true\n"
        "  done\n"
        '  if [[ -n "${out}" ]]; then\n'
        '    mkdir -p "$(dirname "${out}")"\n'
        '    : > "${out}"\n'
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "${1:-}" == *verify_generation_baseline_weights.py ]]; then\n'
        "  exit 0\n"
        "fi\n"
        "exit 88\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    python_log = tmp_path / "python-calls.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "PYTHON": str(fake_python),
        "PYTHON_CALL_LOG": str(python_log),
        "MEANFLOW_SIT_B2_GDRIVE_ID": "fake-id",
    }

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    calls = python_log.read_text(encoding="utf-8")
    assert "-m gdown --help" in calls
    assert "-m gdown fake-id -O" in calls


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
