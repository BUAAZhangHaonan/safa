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
            "e16_meanflow_sit_l2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e16_meanflow_sit_l2_face_mixed_2400ep",
                "model_type": "meanflow_sit",
                "sampler": "meanflow",
                "sample_steps": 1,
                "train_cycle_steps": 1,
                "quality_dir": "artifacts/eval/e16_meanflow_sit_l2_face_mixed_2400ep/quality",
                "global_batch_size": 64,
                "per_device_batch_size": 64,
                "sit_hidden_size": 1024,
                "sit_depth": 24,
                "sit_num_heads": 16,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt",
                "eval_cells": [{"name": "meanflow_l2_1step", "sampler": "meanflow", "sample_steps": 1}],
            },
        ),
        (
            "e19_meanflow_sit_b2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e19_meanflow_sit_b2_face_mixed_2400ep",
                "model_type": "meanflow_sit",
                "sampler": "meanflow",
                "sample_steps": 1,
                "train_cycle_steps": 1,
                "quality_dir": "artifacts/eval/e19_meanflow_sit_b2_face_mixed_2400ep/quality",
                "sit_hidden_size": 768,
                "sit_depth": 12,
                "sit_num_heads": 12,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_2_imagenet256.pt",
                "eval_cells": [{"name": "meanflow_b2_1step", "sampler": "meanflow", "sample_steps": 1}],
            },
        ),
        (
            "e20_rectified_flow_sit_b2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e20_rectified_flow_sit_b2_face_mixed_2400ep",
                "model_type": "rectified_flow_sit",
                "sampler": "euler",
                "sample_steps": 16,
                "train_cycle_steps": 16,
                "quality_dir": "artifacts/eval/e20_rectified_flow_sit_b2_face_mixed_2400ep/quality",
                "sit_hidden_size": 768,
                "sit_depth": 12,
                "sit_num_heads": 12,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_2_imagenet256.pt",
                "eval_cells": [{"name": "rectified_flow_sit_b2_euler16", "sampler": "euler", "sample_steps": 16}],
            },
        ),
        (
            "e17_sit_diffusion_l2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e17_sit_diffusion_l2_face_mixed_2400ep",
                "model_type": "sit_diffusion",
                "sampler": "ddim",
                "sample_steps": 16,
                "train_cycle_steps": 16,
                "quality_dir": "artifacts/eval/e17_sit_diffusion_l2_face_mixed_2400ep/quality",
                "sit_hidden_size": 1024,
                "sit_depth": 24,
                "sit_num_heads": 16,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt",
                "eval_cells": [
                    {"name": "sit_diffusion_l2_ddim1", "sampler": "ddim", "sample_steps": 1},
                    {"name": "sit_diffusion_l2_ddim16", "sampler": "ddim", "sample_steps": 16},
                ],
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
                "sit_hidden_size": 1024,
                "sit_depth": 24,
                "sit_num_heads": 16,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt",
                "eval_cells": [
                    {"name": "latent_consistency_l2_1step", "sampler": "consistency", "sample_steps": 1},
                    {"name": "latent_consistency_l2_4step", "sampler": "consistency", "sample_steps": 4},
                ],
            },
        ),
        (
            "e21_rectified_flow_sit_l2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e21_rectified_flow_sit_l2_face_mixed_2400ep",
                "model_type": "rectified_flow_sit",
                "sampler": "euler",
                "sample_steps": 16,
                "train_cycle_steps": 16,
                "quality_dir": "artifacts/eval/e21_rectified_flow_sit_l2_face_mixed_2400ep/quality",
                "sit_hidden_size": 1024,
                "sit_depth": 24,
                "sit_num_heads": 16,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt",
                "eval_cells": [{"name": "rectified_flow_sit_l2_euler16", "sampler": "euler", "sample_steps": 16}],
            },
        ),
        (
            "e22_sit_diffusion_b2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e22_sit_diffusion_b2_face_mixed_2400ep",
                "model_type": "sit_diffusion",
                "sampler": "ddim",
                "sample_steps": 16,
                "train_cycle_steps": 16,
                "quality_dir": "artifacts/eval/e22_sit_diffusion_b2_face_mixed_2400ep/quality",
                "sit_hidden_size": 768,
                "sit_depth": 12,
                "sit_num_heads": 12,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_2_imagenet256.pt",
                "eval_cells": [
                    {"name": "sit_diffusion_b2_ddim1", "sampler": "ddim", "sample_steps": 1},
                    {"name": "sit_diffusion_b2_ddim16", "sampler": "ddim", "sample_steps": 16},
                ],
            },
        ),
        (
            "e23_latent_consistency_b2_face_mixed_2400ep.yaml",
            {
                "experiment_name": "e23_latent_consistency_b2_face_mixed_2400ep",
                "model_type": "latent_consistency",
                "sampler": "consistency",
                "sample_steps": 4,
                "train_cycle_steps": 4,
                "quality_dir": "artifacts/eval/e23_latent_consistency_b2_face_mixed_2400ep/quality",
                "sit_hidden_size": 768,
                "sit_depth": 12,
                "sit_num_heads": 12,
                "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_2_imagenet256.pt",
                "eval_cells": [
                    {"name": "latent_consistency_b2_1step", "sampler": "consistency", "sample_steps": 1},
                    {"name": "latent_consistency_b2_4step", "sampler": "consistency", "sample_steps": 4},
                ],
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
    assert config["global_batch_size"] == expected.get("global_batch_size", 16)
    assert config["per_device_batch_size"] == expected.get("per_device_batch_size", 16)

    generator = config["generator"]
    assert generator["model_type"] == expected["model_type"]
    assert generator["sampler"] == expected["sampler"]
    assert generator["sample_steps"] == expected["sample_steps"]
    assert generator["train_cycle_steps"] == expected["train_cycle_steps"]
    assert generator["learned_null_condition"] is True
    assert generator["sit_input_channels"] == 4
    assert generator["sit_data_space"] == "latent"
    assert generator["sit_patch_size"] == 2
    assert generator["sit_hidden_size"] == expected["sit_hidden_size"]
    assert generator["sit_depth"] == expected["sit_depth"]
    assert generator["sit_num_heads"] == expected["sit_num_heads"]
    assert generator["sit_pretrained_path"] == expected["sit_pretrained_path"]
    eval_cells = generator["eval_cells"]
    assert len(eval_cells) == len(expected["eval_cells"])
    for actual, expected_cell in zip(eval_cells, expected["eval_cells"]):
        for key, value in expected_cell.items():
            assert actual[key] == value
        assert actual["note"]

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


@pytest.mark.parametrize(
    "filename",
    [
        "e19_meanflow_sit_b2_face_mixed_2400ep.yaml",
        "e20_rectified_flow_sit_b2_face_mixed_2400ep.yaml",
        "e21_rectified_flow_sit_l2_face_mixed_2400ep.yaml",
        "e22_sit_diffusion_b2_face_mixed_2400ep.yaml",
        "e23_latent_consistency_b2_face_mixed_2400ep.yaml",
    ],
)
def test_new_generation_baseline_configs_have_buildable_model_types(filename: str) -> None:
    from safa.models.generator import build_generator

    config = _load_experiment_config(filename)
    generator = dict(config["generator"])
    generator.update(
        {
            "embedding_dim": 16,
            "image_size": 16,
            "base_channels": 4,
            "channel_multipliers": [1],
            "time_embedding_dim": 8,
            "condition_dim": 16,
            "learned_null_condition": True,
            "sit_input_channels": 3,
            "sit_data_space": "pixel",
            "sit_patch_size": 4,
            "sit_hidden_size": 32,
            "sit_depth": 2,
            "sit_num_heads": 4,
            "sit_mlp_ratio": 2.0,
            "sit_time_embedding_dim": 32,
            "sit_pretrained_path": "",
            "attention_backend": "native",
        }
    )
    built = build_generator(generator)

    assert built.config.model_type == config["generator"]["model_type"]


def test_e18_declares_analytic_x0_consistency_surrogate() -> None:
    config = _load_experiment_config("e18_latent_consistency_l2_face_mixed_2400ep.yaml")
    generator = config["generator"]

    assert generator["consistency_train_timesteps"] == 1000
    assert generator["consistency_prediction_type"] == "x0"
    assert generator["consistency_target"] == "analytic_x0"
    assert generator["consistency_min_step_gap"] == 1


def test_generation_baseline_matrix_declares_all_12_report_cells() -> None:
    configs = [
        "e16_meanflow_sit_l2_face_mixed_2400ep.yaml",
        "e17_sit_diffusion_l2_face_mixed_2400ep.yaml",
        "e18_latent_consistency_l2_face_mixed_2400ep.yaml",
        "e19_meanflow_sit_b2_face_mixed_2400ep.yaml",
        "e20_rectified_flow_sit_b2_face_mixed_2400ep.yaml",
        "e21_rectified_flow_sit_l2_face_mixed_2400ep.yaml",
        "e22_sit_diffusion_b2_face_mixed_2400ep.yaml",
        "e23_latent_consistency_b2_face_mixed_2400ep.yaml",
    ]
    expected_cells = {
        "meanflow_l2_1step",
        "sit_diffusion_l2_ddim1",
        "sit_diffusion_l2_ddim16",
        "latent_consistency_l2_1step",
        "latent_consistency_l2_4step",
        "meanflow_b2_1step",
        "rectified_flow_sit_b2_euler16",
        "rectified_flow_sit_l2_euler16",
        "sit_diffusion_b2_ddim1",
        "sit_diffusion_b2_ddim16",
        "latent_consistency_b2_1step",
        "latent_consistency_b2_4step",
    }

    cells = []
    for filename in configs:
        generator = _load_experiment_config(filename)["generator"]
        cells.extend(generator["eval_cells"])

    assert len(cells) == 12
    assert {cell["name"] for cell in cells} == expected_cells
