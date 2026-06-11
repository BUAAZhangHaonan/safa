from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_generator_comparison_report.py"
    spec = importlib.util.spec_from_file_location("run_generator_comparison_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _write_base_eval_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "sampling_seed": 2,
                "device": "cpu",
                "num_workers": 0,
                "batch_size": 2,
                "image_size": 224,
                "index": "data/index/old.jsonl",
                "features": "artifacts/e0_features/old",
                "e0_checkpoint": "artifacts/checkpoints/e0.pt",
                "g_checkpoint": "old.pt",
                "checkpoint_model": "ema",
                "out_json": "old/result.json",
                "per_sample_jsonl": "old/per_sample.jsonl",
                "sample_dir": "old/samples",
                "generated_image_dir": "old/generated_images",
                "face_detection": {"enabled": False},
                "privacy": {"enabled": False},
                "anti_steg": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_dry_run_builds_default_commands_for_three_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script()
    base_config = tmp_path / "base_eval.yaml"
    _write_base_eval_config(base_config)

    args = module.parse_args(
        [
            "--base-eval-config",
            str(base_config),
            "--python",
            "/custom/python",
            "--dry-run",
        ]
    )
    plan = module.build_comparison_plan(args)

    assert [run.name for run in plan.runs] == ["e8", "e9", "e10"]
    flattened = [(run.name, kind, command) for run in plan.runs for kind, command in run.commands]
    assert len(flattened) == 9
    assert flattened[0] == (
        "e8",
        "eval",
        [
            "/custom/python",
            "-m",
            "safa.cli.eval",
            "--config",
            "artifacts/eval/comparison_configs/eval_e8.yaml",
        ],
    )
    assert any(command[1] == "scripts/eval_generation_quality.py" for _, _, command in flattened)
    assert any(command[1] == "scripts/visualize_eval_pairs.py" for _, _, command in flattened)

    exit_code = module.main(
        [
            "--base-eval-config",
            str(base_config),
            "--python",
            "/custom/python",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "safa.cli.eval" in output
    assert "artifacts/eval/comparison_configs/eval_e8.yaml" in output
    assert not (Path.cwd() / "artifacts" / "eval" / "comparison_configs" / "eval_e8.yaml").exists()


def test_runs_subset_filters_commands(tmp_path: Path) -> None:
    module = _load_script()
    base_config = tmp_path / "base_eval.yaml"
    _write_base_eval_config(base_config)

    args = module.parse_args(
        [
            "--base-eval-config",
            str(base_config),
            "--runs",
            "e9",
            "e10",
            "--skip-quality",
        ]
    )
    plan = module.build_comparison_plan(args)

    assert [run.name for run in plan.runs] == ["e9", "e10"]
    assert all(run.name != "e8" for run in plan.runs)
    assert {kind for run in plan.runs for kind, _ in run.commands} == {"eval", "visuals"}


def test_write_eval_configs_uses_expected_checkpoint_and_output_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "base_eval.yaml"
    _write_base_eval_config(base_config)

    args = module.parse_args(
        [
            "--base-eval-config",
            str(base_config),
            "--runs",
            "e9",
            "--device",
            "cuda:0",
            "--seed",
            "1337",
        ]
    )
    plan = module.build_comparison_plan(args)

    module.write_eval_configs(plan)

    eval_config = yaml.safe_load((tmp_path / "artifacts/eval/comparison_configs/eval_e9.yaml").read_text(encoding="utf-8"))
    assert eval_config["g_checkpoint"] == "artifacts/checkpoints/g_medium_v2_meanflow_200ep/best_stage2.pt"
    assert eval_config["checkpoint_model"] == "raw"
    assert eval_config["out_json"] == "artifacts/eval/g_medium_v2_meanflow_200ep/formal_baseline_k1/result.json"
    assert eval_config["per_sample_jsonl"] == "artifacts/eval/g_medium_v2_meanflow_200ep/formal_baseline_k1/per_sample.jsonl"
    assert eval_config["sample_dir"] == "artifacts/eval/g_medium_v2_meanflow_200ep/formal_baseline_k1/samples"
    assert eval_config["generated_image_dir"] == "artifacts/eval/g_medium_v2_meanflow_200ep/formal_baseline_k1/generated_images"
    assert eval_config["device"] == "cuda:0"
    assert eval_config["seed"] == 1337
    assert eval_config["sampling_seed"] == 1337
    assert eval_config["features"] == "artifacts/e0_features/old"


def test_summary_loader_extracts_fake_result_and_quality_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.chdir(tmp_path)
