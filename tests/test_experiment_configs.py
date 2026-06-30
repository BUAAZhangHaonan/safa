from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "medium_v2" / "experiments"


def _load_experiment_config(filename: str) -> dict:
    path = EXPERIMENT_DIR / filename
    assert path.is_file(), f"missing experiment config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "e17_sit_diffusion_l2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e17_sit_diffusion_l2_face_mixed_2400ep",
                "model_type": "sit_diffusion",
                "sampler": "ddim",
                "sample_steps": 16,
                "train_cycle_steps": 16,
                "quality_dir": "artifacts/eval/e17_sit_diffusion_l2_face_mixed_2400ep/quality",
            },
        ),
        (
            "e18_latent_consistency_l2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e18_latent_consistency_l2_face_mixed_2400ep",
                "model_type": "latent_consistency",
                "sampler": "consistency",
                "sample_steps": 4,
                "train_cycle_steps": 4,
                "quality_dir": "artifacts/eval/e18_latent_consistency_l2_face_mixed_2400ep/quality",
            },
        ),
    ],
)
def test_e17_e18_baseline_configs_validate_and_keep_stage1_null_prior(filename: str, expected: dict) -> None:
    from safa.models.generator import FlowGeneratorConfig
    from safa.training import g_loop

    config = _load_experiment_config(filename)

    assert config["experiment_name"] == expected["experiment_name"]
    assert config["latent_training"] is True
    assert config["pixel_image_size"] == 256
    assert config["image_size"] == 32
    assert config["train_index"] == "data/index/train_face_mixed_e14.jsonl"
    assert config["train_features"] == "artifacts/e0_features/train_face_mixed_e14_e0_medium_v1"
    assert config["out_dir"] == f"artifacts/checkpoints/{expected['experiment_name']}"
    assert config["resume_from"] == ""
    assert config["resume_optimizer_state"] is False
    assert config["global_batch_size"] == 16
    assert config["per_device_batch_size"] == 16

    generator = config["generator"]
    assert generator["model_type"] == expected["model_type"]
    assert generator["sampler"] == expected["sampler"]
    assert generator["sample_steps"] == expected["sample_steps"]
    assert generator["train_cycle_steps"] == expected["train_cycle_steps"]
    assert generator["learned_null_condition"] is True
    assert generator["sit_input_channels"] == 4
    assert generator["sit_data_space"] == "latent"
    assert generator["sit_patch_size"] == 2
    assert generator["sit_hidden_size"] == 1024
    assert generator["sit_depth"] == 24
    assert generator["sit_num_heads"] == 16
    assert generator["sit_pretrained_path"] == "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt"

    objective = config["stages"]["stage2"]["stage2_objective"]
    assert objective["type"] == "fm_only_probe"
    assert objective["flow_condition"] == "learned_null_condition"

    quality_eval = config["stages"]["stage2"]["quality_eval"]
    assert quality_eval["real_index"] == "data/index/val_face_mixed_e14.jsonl"
    assert quality_eval["output_dir"] == expected["quality_dir"]
    assert quality_eval["model"] == "ema"

    validation = config["validation"]
    assert validation["index"] == "data/index/val_face_mixed_e14.jsonl"
    assert validation["features"] == "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"

    generator_payload = dict(generator)
    generator_payload["embedding_dim"] = config["embedding_dim"]
    generator_payload["image_size"] = config["image_size"]
    FlowGeneratorConfig.from_dict(generator_payload)
    g_loop._validate_train_g_config(config)


def test_e18_declares_analytic_x0_consistency_surrogate() -> None:
    config = _load_experiment_config("e18_latent_consistency_l2_face_mixed_2400ep.yaml")
    generator = config["generator"]

    assert generator["consistency_train_timesteps"] == 1000
    assert generator["consistency_prediction_type"] == "x0"
    assert generator["consistency_target"] == "analytic_x0"
    assert generator["consistency_min_step_gap"] == 1
