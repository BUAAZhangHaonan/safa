from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from safa.evaluation.meanflow_guidance_runner import (
    EXPECTED_CHECKPOINT_PATH,
    EXPECTED_E0_CHECKPOINT_PATH,
    EXPECTED_EDEV_CHECKPOINT_PATH,
    EXPECTED_VAE_PATH,
    resolve_locked_schedule,
    validate_guidance_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "medium_v2" / "experiments"
CONFIG_NAMES = (
    "r8_meanflow_semigroup_preflight.yaml",
    "r8_meanflow_native_ema.yaml",
    "r8_meanflow_official_xt_flow_map1_gpu0.yaml",
    "r8_meanflow_official_xt_flow_map2_gpu1.yaml",
    "r8_meanflow_paper_split_gpu2.yaml",
    "r8_meanflow_noise_fixed_eta025.yaml",
    "r8_meanflow_noise_fixed_eta05.yaml",
    "r8_meanflow_noise_shell_eta1.yaml",
    "r8_meanflow_noise_shell_eta2.yaml",
)
EXPECTED_HASHES = {
    "checkpoint_sha256": "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d",
    "e0_sha256": "d7d2c57a552155776b8c15a4e52e43ec5082fc046aa0aabb4e9709685f7e3d1a",
    "edev_sha256": "373b331c917834467e854ddf3fe20f39000532f189ec73f76a1abc55d82e560e",
    "heldout_e1_sha256": "cce0de2f1eab097cb6091886f587a9f334dd84ced1ca4dd5e08c3a765718a14c",
    "heldout_e2_sha256": "09c88bd416057222abefeba52ebe88d710715ede791ec34198a23ae5e6e850a8",
    "vae_digest": "ac188e7f6ff31ff1a3bbde37fea3c345ec72f9e10589cf8aa8a3ec7e86afb188",
    "index_sha256": "da14e23eacefecbc2948d1374fb93961a13d017a9183aa1fe2a2f62b33a4b4ea",
    "features_digest": "287b8163f093f290e75e8ef09fbbedc986e6934ab0ac458ad786e655889fbe45",
}
CALIBRATION_MANIFEST = "artifacts/r8_meanflow_flow_map_guidance/manifests/calibration_64.jsonl"
FULL_MANIFEST = "artifacts/r8_meanflow_flow_map_guidance/manifests/full_2048.jsonl"
CALIBRATION_MANIFEST_SHA256 = "ffc1f04f671533ee1498f4b03565826920afcc4e5c6ab244fc6f9b7aa680f964"
FULL_MANIFEST_SHA256 = "7f830ad3f84089bcf83d092fbffaf2b5c3335cf68a4b397f04b65f362f79ae5b"


def _load_configs() -> dict[str, dict]:
    configs = {}
    for name in CONFIG_NAMES:
        payload = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        configs[name] = payload
    return configs


def test_r8_guidance_configs_lock_assets_hashes_and_model_contract() -> None:
    configs = _load_configs()
    expected_paths = {
        "checkpoint": EXPECTED_CHECKPOINT_PATH,
        "e0_checkpoint": EXPECTED_E0_CHECKPOINT_PATH,
        "edev_checkpoint": EXPECTED_EDEV_CHECKPOINT_PATH,
        "heldout_e1_checkpoint": "artifacts/checkpoints/e0_dinov2_large_v2/best.pt",
        "heldout_e2_checkpoint": "artifacts/checkpoints/e0_convnext_tiny/best.pt",
        "vae_path": EXPECTED_VAE_PATH,
        "index": "data/index/val_face_mixed_e14.jsonl",
        "features": "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1",
    }
    for name, config in configs.items():
        assert validate_guidance_config(config)["mode"] == config["mode"], name
        assert {field: config[field] for field in expected_paths} == expected_paths
        assert {field: config[field] for field in EXPECTED_HASHES} == EXPECTED_HASHES
        assert config["checkpoint_model"] == "ema"
        assert config["expected_stage"] == "stage2"
        assert config["expected_stage_epoch_1based"] == 1652
        assert config["expected_model_type"] == "meanflow_sit"
        assert config["expected_sit_patch_size"] == 4
        assert config["transport_condition"] == "learned_null_condition"
        assert config["vae_scaling_factor"] == 0.18215
        assert config["feature_source"] == "cached_features"
        assert config["seed"] == config["sampling_seed"] == 1337
        assert config["asset_digest_cache"] == (
            "artifacts/r8_meanflow_flow_map_guidance/shared/asset_digests.json"
        )
        assert config["heldout_eval"] == "prospective_after_winner_lock"


def test_r8_guidance_configs_have_unique_names_outputs_and_fixed_counts() -> None:
    configs = _load_configs().values()
    assert len({config["experiment_name"] for config in configs}) == len(CONFIG_NAMES)
    assert len({config["out_dir"] for config in configs}) == len(CONFIG_NAMES)
    for config in configs:
        assert config["calibration_samples"] == 64
        assert config["visual_review_samples"] == 64
        assert config["full_samples"] == 2048
        assert config["quality_metrics"] == ["fid", "kid", "niqe", "sharpness"]
        assert "heldout_e1" not in config["calibration_metrics"]
        assert "heldout_e2" not in config["calibration_metrics"]
        assert config["calibration_sample_id_manifest"] == CALIBRATION_MANIFEST
        assert config["calibration_sample_id_manifest_sha256"] == CALIBRATION_MANIFEST_SHA256
        assert config["full_sample_id_manifest"] == FULL_MANIFEST
        assert config["full_sample_id_manifest_sha256"] == FULL_MANIFEST_SHA256


def test_r8_semigroup_and_guided_candidates_are_closed() -> None:
    configs = _load_configs()
    semigroup = configs["r8_meanflow_semigroup_preflight.yaml"]
    assert semigroup["registered_t_cut_candidates"] == [0.75, 0.5, 0.25]
    assert semigroup["split_times"] == [0.25, 0.5, 0.75]

    flow1 = configs["r8_meanflow_official_xt_flow_map1_gpu0.yaml"]
    flow2 = configs["r8_meanflow_official_xt_flow_map2_gpu1.yaml"]
    for config, sample_mode in ((flow1, "flow_map1"), (flow2, "flow_map2")):
        assert config["mode"] == "official_head_current_xt"
        assert config["sample_mode"] == sample_mode
        assert config["num_optim_iters"] == 1
        assert config["guided_steps"] == 3
        assert config["unguided_tail_intervals"] == 2
        assert config["official_adam_step_size_candidates"] == [1.0, 3.0]
        assert config["normalized_eta_candidates"] == [0.25, 0.5, 1.0, 2.0]

    paper = configs["r8_meanflow_paper_split_gpu2.yaml"]
    assert paper["mode"] == "paper_algorithm_split"
    assert paper["normalized_eta_candidates"] == [0.25, 0.5, 1.0, 2.0]
    assert paper["mode"] != flow1["mode"]

    noise = [
        configs["r8_meanflow_noise_fixed_eta025.yaml"],
        configs["r8_meanflow_noise_fixed_eta05.yaml"],
        configs["r8_meanflow_noise_shell_eta1.yaml"],
        configs["r8_meanflow_noise_shell_eta2.yaml"],
    ]
    assert [(config["projection"], config["eta"]) for config in noise] == [
        ("fixed_radius", 0.25),
        ("fixed_radius", 0.5),
        ("typical_shell", 1.0),
        ("typical_shell", 2.0),
    ]
    assert all(config["mode"] == "initial_noise" for config in noise)
    assert all(config["registered_eta_candidates"] == [0.25, 0.5, 1.0, 2.0] for config in noise)


def test_r8_guided_configs_resolve_one_locked_uniform_schedule(tmp_path: Path) -> None:
    configs = _load_configs()
    guided_names = (
        "r8_meanflow_official_xt_flow_map1_gpu0.yaml",
        "r8_meanflow_official_xt_flow_map2_gpu1.yaml",
        "r8_meanflow_paper_split_gpu2.yaml",
    )
    expected_manifest = (
        "artifacts/r8_meanflow_flow_map_guidance/semigroup/locked_schedule_manifest.json"
    )
    assert {configs[name]["schedule_manifest"] for name in guided_names} == {expected_manifest}
    assert {
        configs[name]["semigroup_sample_id_manifest"] for name in guided_names
    } == {CALIBRATION_MANIFEST}
    assert {
        configs[name]["semigroup_sample_id_manifest_sha256"] for name in guided_names
    } == {CALIBRATION_MANIFEST_SHA256}

    report = tmp_path / "semigroup_gate.json"
    report.write_text(json.dumps({"gate_passed": True}), encoding="utf-8")
    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"sample_id":"sample-0"}\n', encoding="utf-8")
    manifest = tmp_path / "locked_schedule_manifest.json"
    payload = {
        "schema_version": 2,
        "gate_passed": True,
        "checkpoint_sha256": EXPECTED_HASHES["checkpoint_sha256"],
        "semigroup_report": str(report),
        "semigroup_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "semigroup_sample_id_manifest": str(samples),
        "semigroup_sample_id_manifest_sha256": hashlib.sha256(
            samples.read_bytes()
        ).hexdigest(),
        "t_cut": 0.25,
        "guided_steps": 3,
        "guided_times": [1.0, 0.75, 0.5, 0.25],
        "unguided_tail_intervals": 2,
        "unguided_times": [0.25, 0.125, 0.0],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["schedule_contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    for name in guided_names:
        config = {
            **configs[name],
            "schedule_manifest": str(manifest),
            "semigroup_report": str(report),
            "semigroup_sample_id_manifest": str(samples),
            "semigroup_sample_id_manifest_sha256": hashlib.sha256(
                samples.read_bytes()
            ).hexdigest(),
            "t_cut": 0.25,
        }
        schedule = resolve_locked_schedule(
            config,
            checkpoint_sha256=EXPECTED_HASHES["checkpoint_sha256"],
            explicit_t_cut=0.25,
        )
        assert schedule["guided_times"] == [1.0, 0.75, 0.5, 0.25]
        assert schedule["unguided_times"] == [0.25, 0.125, 0.0]
        with pytest.raises(ValueError, match="t_cut"):
            resolve_locked_schedule(
                config,
                checkpoint_sha256=EXPECTED_HASHES["checkpoint_sha256"],
                explicit_t_cut=0.5,
            )
