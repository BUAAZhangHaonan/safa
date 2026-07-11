from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "medium_v2" / "experiments"
CONFIG_NAMES = (
    "r7_coupled_embedding_cap025_lr1e4_gpu0.yaml",
    "r7_independent_prior_cap025_lr1e4_gpu1.yaml",
    "r7_independent_prior_cap005_lr1e4_gpu2.yaml",
    "r7_independent_prior_cap025_lr5e5_gpu3.yaml",
)


@pytest.fixture()
def configs() -> dict[str, dict]:
    paths = {name: CONFIG_DIR / name for name in CONFIG_NAMES}
    missing = [str(path.relative_to(REPO_ROOT)) for path in paths.values() if not path.is_file()]
    assert not missing, f"missing R7 config files: {missing}"
    return {
        name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }


def test_r7_configs_validate_and_reference_existing_inputs(configs: dict[str, dict]) -> None:
    from safa.training import g_loop

    for name, config in configs.items():
        g_loop._validate_train_g_config(config)
        input_paths = (
            config["train_index"],
            config["train_features"],
            config["e0_checkpoint"],
            config["resume_from"],
            config["vae_path"],
            config["many_to_many"]["target_index"],
            config["validation"]["index"],
            config["validation"]["features"],
            config["stages"]["stage2"]["stage2_objective"]["ffhq_index"],
        )
        missing = [path for path in input_paths if not (REPO_ROOT / path).exists()]
        assert not missing, f"{name} references missing inputs: {missing}"


def test_r7_configs_share_the_required_training_contract(configs: dict[str, dict]) -> None:
    for config in configs.values():
        objective = config["stages"]["stage2"]["stage2_objective"]
        many_to_many = config["many_to_many"]

        assert config["seed"] == config["sampling_seed"] == 1337
        assert config["global_batch_size"] == config["per_device_batch_size"] == 4
        assert config["learning_rate"] in {1.0e-4, 5.0e-5}
        assert config["train_index"] == "data/index/train_face_mixed_e14_4029avail.jsonl"
        assert config["train_features"] == "artifacts/e0_features/train_face_mixed_e14_e0_medium_v1"
        assert config["resume_from"] == "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt"
        assert config["resume_mode"] == "model_weights_only"
        assert config["generator_trainable"] == "full"
        assert config["ema"]["enabled"] is False
        assert config["stages"]["stage2"]["epochs"] == 1
        assert config["stages"]["stage2"]["quality_eval"]["enabled"] is False
        assert many_to_many["enabled"] is True
        assert many_to_many["pairing_strategy"] == "balanced_epoch_cycle"
        assert objective["type"] == "point_projected_two_step"
        assert objective["sample_condition"] == "embedding"
        assert objective["lambda_lpips"] == pytest.approx(0.0)
        assert objective["pu_fm_increase_budget"] == pytest.approx(0.0)


def test_r7_matrix_changes_only_the_declared_factors(configs: dict[str, dict]) -> None:
    coupled = _without_run_identity(configs[CONFIG_NAMES[0]])
    anchor = _without_run_identity(configs[CONFIG_NAMES[1]])
    tight_cap = _without_run_identity(configs[CONFIG_NAMES[2]])
    low_lr = _without_run_identity(configs[CONFIG_NAMES[3]])

    assert _different_leaf_paths(coupled, anchor) == {
        ("many_to_many", "semantics"),
        ("stages", "stage2", "stage2_objective", "flow_condition"),
    }
    assert _different_leaf_paths(anchor, tight_cap) == {
        ("stages", "stage2", "stage2_objective", "repr_step_ratio_cap"),
    }
    assert _different_leaf_paths(anchor, low_lr) == {("learning_rate",)}

    assert "semantics" not in coupled["many_to_many"]
    assert coupled["stages"]["stage2"]["stage2_objective"]["flow_condition"] == "embedding"
    for config in (anchor, tight_cap, low_lr):
        assert config["many_to_many"]["semantics"] == "independent_prior"
        assert config["stages"]["stage2"]["stage2_objective"]["flow_condition"] == "learned_null_condition"

    assert anchor["learning_rate"] == tight_cap["learning_rate"] == 1.0e-4
    assert low_lr["learning_rate"] == 5.0e-5
    assert anchor["stages"]["stage2"]["stage2_objective"]["repr_step_ratio_cap"] == pytest.approx(0.25)
    assert tight_cap["stages"]["stage2"]["stage2_objective"]["repr_step_ratio_cap"] == pytest.approx(0.05)


def test_r7_configs_use_unique_output_directories(configs: dict[str, dict]) -> None:
    experiment_names = [config["experiment_name"] for config in configs.values()]
    output_dirs = [config["out_dir"] for config in configs.values()]

    assert len(set(experiment_names)) == len(CONFIG_NAMES)
    assert len(set(output_dirs)) == len(CONFIG_NAMES)
    for experiment_name, output_dir in zip(experiment_names, output_dirs, strict=True):
        assert experiment_name.startswith("r7_")
        assert output_dir == f"artifacts/checkpoints/{experiment_name}"


def _without_run_identity(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in {"experiment_name", "out_dir"}}


def _different_leaf_paths(left, right, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[tuple[str, ...]] = set()
        for key in left.keys() | right.keys():
            if key not in left or key not in right:
                differences.add((*path, str(key)))
            else:
                differences.update(_different_leaf_paths(left[key], right[key], (*path, str(key))))
        return differences
    return set() if left == right else {path}
