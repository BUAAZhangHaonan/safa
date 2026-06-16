from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIVE_M_FM_PARAMETER_COUNT = 5_004_291


def _tiny_meanflow_sit_config() -> dict:
    return {
        "model_type": "meanflow_sit",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "meanflow",
        "learned_null_condition": True,
        "meanflow_ratio": 0.25,
        "meanflow_ratio_r_not_equal_t": 0.75,
        "meanflow_adaptive_weighting": True,
        "meanflow_norm_p": 1.0,
        "meanflow_norm_eps": 0.001,
        "meanflow_jvp_mode": "torch_func",
        "sit_input_channels": 3,
        "sit_patch_size": 4,
        "sit_hidden_size": 32,
        "sit_depth": 2,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 32,
    }


def test_build_generator_supports_meanflow_sit_and_preserves_model_type_roundtrip() -> None:
    from safa.models.generator import FlowGeneratorConfig, build_generator

    config = FlowGeneratorConfig.from_dict(_tiny_meanflow_sit_config())

    assert config.model_type == "meanflow_sit"
    assert config.to_dict()["model_type"] == "meanflow_sit"
    assert config.to_dict()["meanflow_ratio"] == 0.25
    assert config.to_dict()["meanflow_ratio_r_not_equal_t"] == 0.75
    generator = build_generator(config.to_dict())
    assert generator.config.model_type == "meanflow_sit"
    assert generator.config.sample_steps == 1
    assert generator.config.train_cycle_steps == 1


def test_meanflow_sit_forward_loss_and_one_step_sample_shapes() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_meanflow_sit_config())
    z = torch.zeros(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    output = generator(z)
    one_step = generator.sample(z, steps=1, x_init=x_init, clamp_output=False)
    ignored_multi_step = generator.sample(z, steps=8, x_init=x_init, clamp_output=False)
    loss, metrics = generator.flow_matching_loss(torch.rand(2, 3, 16, 16), z)
    loss.backward()

    assert tuple(output.shape) == (2, 3, 16, 16)
    assert tuple(one_step.shape) == (2, 3, 16, 16)
    assert torch.allclose(one_step, ignored_multi_step)
    assert tuple(loss.shape) == ()
    assert torch.isfinite(loss)
    assert metrics["meanflow_backbone"] == "sit"
    assert metrics["meanflow_jvp_mode"] == "torch_func"
    assert torch.isfinite(metrics["meanflow_raw_mse"])
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_meanflow_sit_null_condition_and_embedding_shape_errors_are_clear() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_meanflow_sit_config())
    null_z = generator.make_null_condition(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    assert tuple(null_z.shape) == (2, 16)
    assert tuple(generator.sample(null_z, steps=1).shape) == (2, 3, 16, 16)
    with pytest.raises(ValueError, match=r"G expects z with shape \[B,16\]"):
        generator.sample(torch.zeros(2, 15))
    with pytest.raises(ValueError, match=r"G expects z with shape \[B,16\]"):
        generator.flow_matching_loss(torch.rand(2, 3, 16, 16), torch.zeros(2, 17))


def test_meanflow_sit_flow_loss_accepts_expanded_null_condition_for_jvp() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_meanflow_sit_config())
    z = generator.make_null_condition(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.rand(2, 3, 16, 16)

    assert z.stride(0) == 0
    loss, metrics = generator.flow_matching_loss(x, z)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["meanflow_raw_mse"])


def test_meanflow_sit_checkpoint_loader_reports_missing_and_unexpected_keys() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_meanflow_sit_config())
    missing = generator.load_pretrained("/no/such/meanflow_sit.pt", allow_missing=True)
    assert missing["loaded"] is False
    assert missing["missing_file"] is True
    assert missing["missing_keys"] == []
    assert missing["unexpected_keys"] == []

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.pt"
        torch.save({"model": {"unexpected.weight": torch.zeros(1)}}, path)
        report = generator.load_pretrained(str(path), strict=False)

    assert report["loaded"] is False
    assert report["missing_file"] is False
    assert report["loaded_keys"] == []
    assert report["missing_keys"]
    assert report["unexpected_keys"] == ["unexpected.weight"]


def test_meanflow_sit_loads_zhuyu_style_checkpoint_keys_and_reports_condition_mismatch() -> None:
    from safa.models.generator import build_generator

    config = _tiny_meanflow_sit_config()
    config.update({"sit_input_channels": 4, "image_size": 16, "sit_patch_size": 4})
    generator = build_generator(config)
    target = generator.vector_field.state_dict()

    def like_target(key: str, value: float) -> torch.Tensor:
        return torch.full_like(target[key], value)

    zhuyu_state = {
        "x_embedder.proj.weight": like_target("x_embedder.weight", 0.11),
        "x_embedder.proj.bias": like_target("x_embedder.bias", 0.12),
        "t_embedder.mlp.0.weight": like_target("t_embedder.mlp.0.weight", 0.21),
        "r_embedder.mlp.2.bias": like_target("r_embedder.mlp.2.bias", 0.31),
        "blocks.0.attn.qkv.weight": like_target("blocks.0.attn.qkv.weight", 0.41),
        "blocks.0.mlp.fc1.weight": like_target("blocks.0.mlp.0.weight", 0.51),
        "blocks.0.mlp.fc2.bias": like_target("blocks.0.mlp.2.bias", 0.52),
        "final_layer.linear.weight": like_target("final_layer.linear.weight", 0.61),
        "y_embedder.embedding_table.weight": torch.zeros(1001, config["sit_hidden_size"]),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "zhuyu.pt"
        torch.save({"model": zhuyu_state}, path)
        report = generator.load_pretrained(str(path), strict=False)

    assert report["loaded"] is True
    assert report["source_format"] == "zhuyu_meanflow_sit"
    assert "x_embedder.weight" in report["loaded_keys"]
    assert "blocks.0.mlp.0.weight" in report["loaded_keys"]
    assert "final_layer.linear.weight" in report["loaded_keys"]
    assert torch.allclose(generator.vector_field.x_embedder.weight, like_target("x_embedder.weight", 0.11))
    assert any(
        item["source_key"] == "y_embedder.embedding_table.weight"
        and item["target_key"] == "z_embedder.0.weight"
        and item["reason"] == "shape_mismatch"
        for item in report["mismatched_keys"]
    )


def test_meanflow_sit_loads_raw_ordered_dict_when_state_key_is_empty() -> None:
    from collections import OrderedDict

    from safa.models.generator import build_generator

    config = _tiny_meanflow_sit_config()
    config.update({"sit_input_channels": 4, "image_size": 16, "sit_patch_size": 4})
    generator = build_generator(config)
    target = generator.vector_field.state_dict()
    raw_state = OrderedDict(
        [
            ("x_embedder.proj.weight", torch.full_like(target["x_embedder.weight"], 0.17)),
            ("x_embedder.proj.bias", torch.full_like(target["x_embedder.bias"], 0.18)),
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "raw_zhuyu.pt"
        torch.save(raw_state, path)
        report = generator.load_pretrained(str(path), state_key=None, strict=False)

    assert report["loaded"] is True
    assert report["source_format"] == "zhuyu_meanflow_sit"
    assert report["loaded_keys"] == ["x_embedder.bias", "x_embedder.weight"]
    assert torch.allclose(generator.vector_field.x_embedder.weight, raw_state["x_embedder.proj.weight"])


def test_meanflow_sit_raw_ordered_dict_with_nonexistent_state_key_errors_clearly() -> None:
    from collections import OrderedDict

    from safa.models.generator import build_generator

    generator = build_generator(_tiny_meanflow_sit_config())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "raw.pt"
        torch.save(OrderedDict([("x_embedder.proj.weight", torch.zeros(1))]), path)
        with pytest.raises(KeyError, match="checkpoint missing state_key.*model"):
            generator.load_pretrained(str(path), state_key="model", strict=False)


def test_meanflow_sit_checkpoint_loader_reports_shape_mismatches_without_silent_success() -> None:
    from safa.models.generator import build_generator

    config = _tiny_meanflow_sit_config()
    config.update({"sit_input_channels": 4, "image_size": 16, "sit_patch_size": 4})
    generator = build_generator(config)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_shape.pt"
        torch.save({"model": {"x_embedder.proj.weight": torch.zeros(1)}}, path)
        report = generator.load_pretrained(str(path), strict=False)

    assert report["loaded"] is False
    assert report["loaded_keys"] == []
    assert report["mismatched_keys"] == [
        {
            "source_key": "x_embedder.proj.weight",
            "target_key": "x_embedder.weight",
            "source_shape": [1],
            "target_shape": [32, 4, 4, 4],
            "reason": "shape_mismatch",
        }
    ]


def test_e11_meanflow_sit_config_is_k100_stage1_null_conditioned_and_larger_than_5m() -> None:
    from safa.models.generator import build_generator
    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "experiments" / "e11_meanflow_sit_b_stage1_200ep.yaml"
    assert path.is_file()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment_name"] == "e11_meanflow_sit_b_stage1_200ep"
    assert config["device"] == "cuda:0"
    assert config["amp"] is False
    assert config["global_batch_size"] == 64
    assert config["per_device_batch_size"] == 64
    assert config["num_workers"] == 16
    assert config["pixel_image_size"] == 256
    assert config["image_size"] == config["pixel_image_size"] // 8
    assert config["out_dir"] == "artifacts/checkpoints/e11_meanflow_sit_b_stage1_200ep"
    assert config["out_dir"] != "artifacts/checkpoints/g_medium_v2_meanflow_200ep"
    assert config["out_dir"] != "artifacts/checkpoints/g_medium_v2_ddim_200ep"
    assert config["stages"]["stage2"]["epochs"] == 200
    gradient_monitor = config["stages"]["stage2"]["gradient_monitor"]
    assert gradient_monitor["enabled"] is False
    assert config["generator"]["model_type"] == "meanflow_sit"
    assert config["generator"]["sample_steps"] == 1
    assert config["generator"]["train_cycle_steps"] == 1
    assert config["generator"]["sampler"] == "meanflow"
    assert config["generator"]["learned_null_condition"] is True
    assert config["generator"]["meanflow_ratio"] == 0.25
    assert config["generator"]["meanflow_ratio_r_not_equal_t"] == 0.75
    assert config["generator"]["sit_input_channels"] == 4
    assert config["generator"]["sit_patch_size"] == 4
    assert config["generator"]["sit_depth"] == 12
    assert config["generator"]["sit_hidden_size"] == 768
    assert config["generator"]["sit_num_heads"] == 12
    assert config["generator"]["sit_pretrained_path"] == (
        "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_4_imagenet256.pt"
    )
    assert not config["generator"].get("sit_pretrained_state_key")
    assert config["generator"]["sit_pretrained_source"].startswith("https://drive.google.com/drive/folders/")
    assert config["stages"]["stage2"]["stage2_objective"]["flow_condition"] == "learned_null_condition"
    validation = config["validation"]
    assert validation["max_samples"] == 128
    assert validation["batch_size"] == 64
    quality_eval = config["stages"]["stage2"]["quality_eval"]
    assert quality_eval["niqe_interval_epochs"] == 5
    assert quality_eval["niqe_max_samples"] == 256
    assert quality_eval["distribution_interval_epochs"] == 50
    assert quality_eval["distribution_max_samples"] == 2048
    assert quality_eval["quality_num_workers"] == 4
    assert quality_eval["distribution_cuda_visible_devices"] == "0"
    assert quality_eval["distribution_device"] == "cuda:0"
    assert quality_eval["output_dir"] == "artifacts/eval/e11_meanflow_sit_b_stage1_200ep/quality"

    g_loop._validate_train_g_config(config)
    generator_config = dict(config["generator"])
    generator_config["sit_pretrained_path"] = ""
    generator_config["embedding_dim"] = config["embedding_dim"]
    generator_config["image_size"] = config["image_size"]
    generator = build_generator(generator_config)
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    assert parameter_count > FIVE_M_FM_PARAMETER_COUNT


def test_e12_e13_meanflow_sit_stage2_configs_target_gpu4_gpu5() -> None:
    from safa.training import g_loop

    expected = {
        "e12_pu_sgd_meanflow_sit_stage2_gpu4_200ep.yaml": {
            "experiment_name": "e12_pu_sgd_meanflow_sit_stage2_gpu4_200ep",
            "device": "cuda:0",
            "optimizer_type": "sgd",
            "pu_optimizer_type": "sgd",
            "out_dir": "artifacts/checkpoints/e12_pu_sgd_meanflow_sit_stage2_gpu4_200ep",
            "quality_dir": "artifacts/eval/e12_pu_sgd_meanflow_sit_stage2_gpu4_200ep/quality",
        },
        "e13_pu_adamw_meanflow_sit_stage2_gpu5_200ep.yaml": {
            "experiment_name": "e13_pu_adamw_meanflow_sit_stage2_gpu5_200ep",
            "device": "cuda:0",
            "optimizer_type": "adamw",
            "pu_optimizer_type": "adamw",
            "out_dir": "artifacts/checkpoints/e13_pu_adamw_meanflow_sit_stage2_gpu5_200ep",
            "quality_dir": "artifacts/eval/e13_pu_adamw_meanflow_sit_stage2_gpu5_200ep/quality",
        },
    }

    seen_out_dirs = set()
    for filename, values in expected.items():
        path = REPO_ROOT / "configs" / "medium_v2" / "experiments" / filename
        assert path.is_file()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["experiment_name"] == values["experiment_name"]
        assert config["device"] == values["device"]
        assert config["optimizer_type"] == values["optimizer_type"]
        assert config["amp"] is True
        assert config["global_batch_size"] == 8
        assert config["per_device_batch_size"] == 8
        assert config["num_workers"] == 8
        assert config["latent_training"] is True
        assert config["pixel_image_size"] == 256
        assert config["image_size"] == 32
        assert config["vae_path"] == "artifacts/checkpoints/external/sd-vae-ft-ema"
        assert config["resume_from"] == "artifacts/checkpoints/e11_meanflow_sit_b_stage1_200ep/best.pt"
        assert config["resume_mode"] == "model_weights_only"
        assert config["resume_optimizer_state"] is False
        assert config["out_dir"] == values["out_dir"]
        assert config["out_dir"] not in seen_out_dirs
        seen_out_dirs.add(config["out_dir"])

        generator = config["generator"]
        assert generator["model_type"] == "meanflow_sit"
        assert generator["sample_steps"] == 1
        assert generator["train_cycle_steps"] == 1
        assert generator["sampler"] == "meanflow"
        assert generator["sit_input_channels"] == 4
        assert generator["sit_data_space"] == "latent"
        assert generator["sit_patch_size"] == 4
        assert generator["sit_hidden_size"] == 768
        assert generator["sit_depth"] == 12
        assert generator["sit_num_heads"] == 12
        assert generator["sit_pretrained_path"] == (
            "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_4_imagenet256.pt"
        )

        objective = config["stages"]["stage2"]["stage2_objective"]
        assert config["stages"]["stage2"]["gradient_monitor"]["enabled"] is False
        assert objective["type"] == "point_projected_two_step"
        assert objective["flow_condition"] == "embedding"
        assert objective["optimizer_type"] == values["pu_optimizer_type"]
        assert objective["repr_step_ratio_cap"] == 0.25

        quality_eval = config["stages"]["stage2"]["quality_eval"]
        assert quality_eval["distribution_cuda_visible_devices"] == "6"
        assert quality_eval["distribution_device"] == "cuda:0"
        assert quality_eval["output_dir"] == values["quality_dir"]
        assert quality_eval["model"] == "ema"

        validation = config["validation"]
        assert validation["max_samples"] == 256
        assert validation["batch_size"] == 8

        g_loop._validate_train_g_config(config)


def test_e14_meanflow_sit_mixed_face_continuation_config() -> None:
    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "experiments" / "e14_meanflow_sit_b_face_mixed_continue_200ep.yaml"
    assert path.is_file()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment_name"] == "e14_meanflow_sit_b_face_mixed_continue_200ep"
    assert config["device"] == "cuda:0"
    assert config["global_batch_size"] == 128
    assert config["per_device_batch_size"] == 128
    assert config["train_index"] == "data/index/train_face_mixed_e14.jsonl"
    assert config["train_features"] == "artifacts/e0_features/train_face_mixed_e14_e0_medium_v1"
    assert config["resume_from"] == "artifacts/checkpoints/e11_meanflow_sit_b_stage1_200ep/best_stage2.pt"
    assert config["resume_mode"] == "model_weights_only"
    assert config["resume_optimizer_state"] is False
    assert config["out_dir"] == "artifacts/checkpoints/e14_meanflow_sit_b_face_mixed_continue_200ep"

    generator = config["generator"]
    assert generator["model_type"] == "meanflow_sit"
    assert generator["sample_steps"] == 1
    assert generator["train_cycle_steps"] == 1
    assert generator["sampler"] == "meanflow"
    assert generator["learned_null_condition"] is True
    assert generator["sit_input_channels"] == 4
    assert generator["sit_data_space"] == "latent"
    assert generator["sit_patch_size"] == 4
    assert generator["sit_hidden_size"] == 768
    assert generator["sit_depth"] == 12
    assert generator["sit_num_heads"] == 12

    stage2 = config["stages"]["stage2"]
    assert stage2["epochs"] == 200
    assert stage2["gradient_monitor"]["enabled"] is False
    objective = stage2["stage2_objective"]
    assert objective["type"] == "fm_only_probe"
    assert objective["flow_condition"] == "learned_null_condition"

    quality_eval = stage2["quality_eval"]
    assert quality_eval["real_index"] == "data/index/val_face_mixed_e14.jsonl"
    assert quality_eval["output_dir"] == "artifacts/eval/e14_meanflow_sit_b_face_mixed_continue_200ep/quality"
    assert quality_eval["model"] == "ema"

    validation = config["validation"]
    assert validation["index"] == "data/index/val_face_mixed_e14.jsonl"
    assert validation["features"] == "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"

    g_loop._validate_train_g_config(config)
