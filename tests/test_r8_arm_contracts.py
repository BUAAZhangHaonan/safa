from __future__ import annotations

from safa.evaluation.r8_arm_contracts import canonical_arm_config_digest


def _config() -> dict:
    return {
        "mode": "official_head_current_xt",
        "sample_mode": "flow_map1",
        "optimization_mode": "official_adam",
        "step_size": 1.0,
        "eta": 0.25,
        "num_optim_iters": 1,
        "t_cut": 0.25,
        "schedule_contract_sha256": "a" * 64,
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_model": "ema",
        "transport_condition": "learned_null_condition",
        "e0_checkpoint": "e0.pt",
        "e0_sha256": "c" * 64,
        "edev_checkpoint": "edev.pt",
        "edev_sha256": "d" * 64,
        "vae_path": "vae",
        "vae_digest": "e" * 64,
        "vae_scaling_factor": 0.18215,
        "index": "index.jsonl",
        "index_sha256": "f" * 64,
        "features": "features",
        "features_digest": "1" * 64,
        "feature_source": "cached_features",
        "sampling_seed": 1337,
    }


def test_arm_digest_excludes_runtime_shard_manifest_and_visual_fields() -> None:
    config = _config()
    runtime = {
        **config,
        "output_dir": "other-output",
        "out_dir": "other-out",
        "shard_index": 3,
        "num_shards": 4,
        "sample_id_manifest": "full-2048.jsonl",
        "sample_id_manifest_sha256": "2" * 64,
        "max_samples": 2048,
        "contact_sheets": False,
        "contact_sheet_rows": 4,
    }

    assert canonical_arm_config_digest(runtime) == canonical_arm_config_digest(config)


def test_arm_digest_rejects_same_mode_with_different_algorithm_variant() -> None:
    config = _config()
    original = canonical_arm_config_digest(config)

    for field, value in (
        ("sample_mode", "flow_map2"),
        ("optimization_mode", "paper_normalized_direct_autograd"),
        ("step_size", 3.0),
        ("eta", 0.5),
        ("num_optim_iters", 2),
        ("schedule_contract_sha256", "9" * 64),
        ("checkpoint_sha256", "8" * 64),
    ):
        assert canonical_arm_config_digest({**config, field: value}) != original


def test_noise_shell_typical_delta_is_part_of_arm_digest() -> None:
    shell = {
        **_config(),
        "mode": "initial_noise",
        "projection": "typical_shell",
        "eta": 1.0,
        "num_updates": 16,
        "typical_delta": 0.05,
    }

    assert canonical_arm_config_digest(shell) != canonical_arm_config_digest(
        {**shell, "typical_delta": 0.1}
    )
