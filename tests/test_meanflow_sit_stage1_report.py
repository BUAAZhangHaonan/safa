from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_meanflow_sit_stage1_report.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("run_meanflow_sit_stage1_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _write_config(path: Path, *, experiment_name: str, model_type: str, sample_steps: int, epochs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": experiment_name,
                "amp": False,
                "global_batch_size": 16,
                "per_device_batch_size": 16,
                "image_size": 32,
                "pixel_image_size": 256,
                "latent_training": model_type == "meanflow_sit",
                "out_dir": f"artifacts/checkpoints/{experiment_name}",
                "generator": {
                    "model_type": model_type,
                    "sample_steps": sample_steps,
                    "sampler": "heun" if model_type == "conditional_flow_matching" else "meanflow_1nfe",
                    "sit_input_channels": 4,
                    "sit_patch_size": 4,
                    "sit_data_space": "latent",
                    "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/weights.pt",
                },
                "stages": {
                    "stage2": {
                        "epochs": epochs,
                        "stage2_objective": {"type": "fm_only_probe", "flow_condition": "learned_null_condition"},
                        "quality_eval": {
                            "enabled": True,
                            "metrics": ["niqe", "fid", "kid"],
                            "output_dir": f"artifacts/eval/{experiment_name}/quality",
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _append_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_partial_report_summarizes_e11_metrics_and_marks_missing_e8(tmp_path: Path) -> None:
    module = _load_script()
    _write_config(
        tmp_path / "configs/medium_v2/experiments/e11_meanflow_sit_b_stage1_200ep.yaml",
        experiment_name="e11_meanflow_sit_b_stage1_200ep",
        model_type="meanflow_sit",
        sample_steps=1,
        epochs=200,
    )
    _write_config(
        tmp_path / "configs/medium_v2/experiments/e8_fm_only_200ep.yaml",
        experiment_name="e8_fm_only_200ep",
        model_type="conditional_flow_matching",
        sample_steps=32,
        epochs=200,
    )
    checkpoint_dir = tmp_path / "artifacts/checkpoints/e11_meanflow_sit_b_stage1_200ep"
    (checkpoint_dir / "best_stage2.pt").parent.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best_stage2.pt").write_bytes(b"checkpoint")
    _append_metrics(
        checkpoint_dir / "metrics_history.jsonl",
        [
            {
                "stage_epoch_1based": 1,
                "validation_raw_face_detection_rate": 0.9,
                "validation_raw_zero_face_rate": 0.1,
                "quality_ema_niqe": 7.0,
            },
            {
                "stage_epoch_1based": 5,
                "validation_raw_face_detection_rate": 0.9921875,
                "validation_raw_single_face_eq1_rate": 0.9921875,
                "validation_raw_zero_face_rate": 0.0078125,
                "validation_ema_face_detection_rate": 0.0078125,
                "validation_ema_zero_face_rate": 0.9921875,
                "quality_ema_niqe": 5.286571992026357,
                "quality_ema_fid": 123.4,
                "quality_ema_kid_mean": 0.12,
            },
        ],
    )
    quality_dir = tmp_path / "artifacts/eval/e11_meanflow_sit_b_stage1_200ep/quality/epoch_0005"
    generated_dir = quality_dir / "generated_images"
    generated_dir.mkdir(parents=True)
    (generated_dir / "000000.png").write_bytes(b"png")
    (quality_dir / "stage2_epoch_0005_ema_niqe.json").write_text(
        json.dumps({"iqa": {"method": "niqe", "mean": 5.28, "std": 1.2}, "num_generated": 1}),
        encoding="utf-8",
    )

    args = module.parse_args(["--repo-root", str(tmp_path), "--runs", "e11", "e8", "--train-session", ""])
    summary = module.build_report(args)

    e11 = summary["runs"]["e11"]
    assert e11["status"] == "partial"
    assert e11["latest_epoch"] == 5
    assert e11["target_epochs"] == 200
    assert e11["metrics"]["raw"]["face_detection_rate"] == 0.9921875
    assert e11["metrics"]["raw"]["zero_face_rate"] == 0.0078125
    assert e11["metrics"]["ema"]["niqe"] == 5.286571992026357
    assert e11["metrics"]["ema"]["fid"] == 123.4
    assert e11["metrics"]["ema"]["kid_mean"] == 0.12
    assert e11["samples"]["generated_count"] == 1
    assert e11["sampling"]["nfe"] == 1
    assert e11["checkpoint_paths"]["best_stage2"]["exists"] is True
    assert summary["runs"]["e8"]["status"] == "missing_metrics"


def test_report_main_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_script()
    _write_config(
        tmp_path / "configs/medium_v2/experiments/e11_meanflow_sit_b_stage1_200ep.yaml",
        experiment_name="e11_meanflow_sit_b_stage1_200ep",
        model_type="meanflow_sit",
        sample_steps=1,
        epochs=1,
    )
    _append_metrics(
        tmp_path / "artifacts/checkpoints/e11_meanflow_sit_b_stage1_200ep/metrics_history.jsonl",
        [{"stage_epoch_1based": 1, "validation_raw_face_detection_rate": 1.0, "validation_raw_zero_face_rate": 0.0}],
    )
    output_json = tmp_path / "artifacts/reports/e11_report.json"
    output_md = tmp_path / "artifacts/reports/e11_report.md"

    exit_code = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--runs",
            "e11",
            "--train-session",
            "",
            "--output-json",
            str(output_json.relative_to(tmp_path)),
            "--output-md",
            str(output_md.relative_to(tmp_path)),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["runs"]["e11"]["status"] == "complete"
    markdown = output_md.read_text(encoding="utf-8")
    assert "e11" in markdown
    assert "1.0000" in markdown
