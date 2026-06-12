from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_k100_meanflow_sit_e11.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("run_k100_meanflow_sit_e11", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_k100_runner_env_pins_single_gpu_and_cu13_library_path() -> None:
    module = _load_script()

    env = module.build_k100_env({"LD_LIBRARY_PATH": "/keep/lib"})

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONPATH"] == "src"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    ld_parts = env["LD_LIBRARY_PATH"].split(":")
    assert ld_parts[0] == module.DEFAULT_CU13_LIBRARY_PATH
    assert "/keep/lib" in ld_parts


def test_k100_runner_tmux_command_uses_e11_config_and_unique_log() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--timestamp",
            "20260613_120000",
            "--repo-root",
            "/repo",
            "--python",
            "/env/bin/python",
        ]
    )
    plan = module.build_run_plan(args)

    command = module.build_tmux_start_command(plan)
    shell_command = command[-1]

    assert command[:5] == ["tmux", "new-session", "-d", "-s", "safa_e11_meanflow_sit_k100_200ep_20260613_120000"]
    assert "CUDA_VISIBLE_DEVICES=0" in shell_command
    assert "CUDA_VISIBLE_DEVICES=6" not in shell_command
    assert module.DEFAULT_CU13_LIBRARY_PATH in shell_command
    assert "configs/medium_v2/experiments/e11_meanflow_sit_b_stage1_200ep.yaml" in shell_command
    assert "artifacts/logs/e11_meanflow_sit_k100_200ep_20260613_120000.log" in shell_command
    assert " > artifacts/logs/e11_meanflow_sit_k100_200ep_20260613_120000.log 2>&1" in shell_command


def test_k100_runner_validates_required_weight_and_vae_artifacts(tmp_path: Path) -> None:
    module = _load_script()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "generator": {
                    "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/weights.pt",
                },
                "vae_path": "artifacts/checkpoints/external/sd-vae-ft-ema",
            }
        ),
        encoding="utf-8",
    )
    args = module.parse_args(
        [
            "--timestamp",
            "20260613_120000",
            "--repo-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--log",
            "artifacts/logs/e11.log",
        ]
    )
    plan = module.build_run_plan(args)

    with pytest.raises(FileNotFoundError, match="MeanFlow-SiT pretrained checkpoint"):
        module.validate_prerequisites(plan)

    weight = tmp_path / "artifacts/checkpoints/external/meanflow_sit/weights.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError, match="VAE path"):
        module.validate_prerequisites(plan)
