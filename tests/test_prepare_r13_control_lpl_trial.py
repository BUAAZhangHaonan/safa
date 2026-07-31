from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed1337_order_is_exact_one_epoch_of_batch4() -> None:
    module = _load_script("prepare_r13_control_lpl_trial.py")
    records = [{"sample_id": f"sample-{index}"} for index in range(30000)]
    rows = module.build_train_order_rows(records)
    assert len(rows) == 30000
    assert rows[0]["source_index"] == 23500
    assert rows[-1]["order_ordinal"] == 29999
    assert rows[-1]["batch_index"] == 7499
    assert rows[-1]["batch_offset"] == 3
    assert sorted(row["source_index"] for row in rows) == list(range(30000))


def test_r13_configs_lock_full_and_probe_matrix() -> None:
    from safa.training.g_loop import _validate_train_g_config

    configs = {}
    for arm in ("control", "lpl"):
        for kind, name in (
            ("full", f"r13_{arm}_conditioning_1epoch_seed1337.yaml"),
            ("probe", f"r13_probe_{arm}_conditioning_seed1337.yaml"),
        ):
            config = yaml.safe_load((ROOT / "configs/medium_v2/experiments" / name).read_text(encoding="utf-8"))
            _validate_train_g_config(config)
            configs[(arm, kind)] = config
            assert config["global_batch_size"] == config["per_device_batch_size"] == 4
            assert config["r13_arm_id"] == arm
            assert config["latent_perceptual_loss"]["enabled"] is (arm == "lpl")
            assert config["resume_mode"] == "model_weights_only"
            assert config["resume_checkpoint_model"] == "ema"
            assert config["resume_optimizer_state"] is False
            assert config["optimizer_type"] == "adamw"
    assert [configs[(arm, "full")]["optimizer_step_contract"]["required_steps"] for arm in ("control", "lpl")] == [7500, 7500]
    assert [configs[(arm, "probe")]["optimizer_step_contract"]["required_steps"] for arm in ("control", "lpl")] == [8, 8]
    assert configs[("control", "full")]["optimizer_checkpoint_contract"]["save_steps"] == [0, 2500, 5000, 7500]
    assert "optimizer_checkpoint_contract" not in configs[("lpl", "probe")]


def test_materialized_preparation_validates_and_maps_all_four_gpus() -> None:
    preparation = ROOT / "artifacts/r13_control_lpl_training/preparation_v1"
    validator = _load_script("validate_r13_control_lpl_preparation.py")
    result = validator.validate(preparation)
    assert result["status"] == "validated_prepared_not_launched"
    assert result["train_order_rows"] == 30000
    training = json.loads((preparation / "training_ledger.json").read_text(encoding="utf-8"))
    probe = json.loads((preparation / "probe_ledger.json").read_text(encoding="utf-8"))
    assert [job["physical_gpu"]["index"] for job in training["jobs"]] == [0, 1]
    assert [job["physical_gpu"]["index"] for job in probe["jobs"]] == [2, 3]
    assert all(job["batch_size"] == 4 and job["retry_count"] == 0 for job in [*training["jobs"], *probe["jobs"]])


def test_train_order_artifact_digest_is_registered() -> None:
    from safa.training.r13_training_contract import R13_TRAIN_ORDER_SHA256

    path = ROOT / "artifacts/r13_control_lpl_training/preparation_v1/train_order_seed1337.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == R13_TRAIN_ORDER_SHA256
    assert len(path.read_text(encoding="utf-8").splitlines()) == 30000
