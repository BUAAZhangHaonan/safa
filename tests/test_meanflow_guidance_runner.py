from __future__ import annotations

import hashlib
import json
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.evaluation.meanflow_guidance_runner import (  # noqa: E402
    EXPECTED_CHECKPOINT_PATH,
    EXPECTED_E0_CHECKPOINT_PATH,
    EXPECTED_EDEV_CHECKPOINT_PATH,
    EXPECTED_VAE_PATH,
    EXPECTED_VAE_SCALING_FACTOR,
    GuidanceRuntime,
    aggregate_session_memory,
    asset_contract_from_config,
    build_frozen_runtime,
    cached_asset_digest,
    deterministic_shard,
    execute_guidance_mode,
    load_ema_generator,
    read_ordered_sample_manifest,
    resolve_locked_schedule,
    resume_remaining_ids,
    run_guidance_from_config,
    run_guidance_records,
    validate_checkpoint_contract,
    validate_guidance_config,
)
from safa.evaluation import meanflow_guidance_runner as runner_module  # noqa: E402
from safa.guidance.meanflow_flow_map import freeze_guidance_stack  # noqa: E402
from safa.training.latent_codec import LatentCodec, LatentCodecConfig  # noqa: E402
from safa.training.losses import normalize_for_e0  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_payload() -> dict:
    return {
        "stage": "stage2",
        "metrics": {"stage_epoch_1based": 1652},
        "model_config": {
            "model_type": "meanflow_sit",
            "sit_patch_size": 4,
            "sit_hidden_size": 768,
            "sit_depth": 12,
            "sit_num_heads": 12,
            "sit_input_channels": 4,
            "image_size": 32,
            "embedding_dim": 512,
            "learned_null_condition": True,
            "sample_steps": 1,
            "sit_data_space": "latent",
        },
        "ema_model_state_dict": {"weight": torch.ones(())},
        "model_state_dict": {"weight": torch.zeros(())},
        "optimizer_state_dict": {"state": {0: {"exp_avg": torch.ones(100)}}},
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("ema_model_state_dict"), "ema_model_state_dict"),
        (lambda payload: payload.__setitem__("stage", "stage1"), "stage.*stage2"),
        (lambda payload: payload["metrics"].__setitem__("stage_epoch_1based", 1651), "1652"),
        (lambda payload: payload["model_config"].__setitem__("sit_patch_size", 2), "sit_patch_size.*4"),
        (
            lambda payload: payload["model_config"].__setitem__("learned_null_condition", False),
            "learned_null_condition.*True",
        ),
    ],
)
def test_checkpoint_contract_rejects_any_target_metadata_mismatch(mutate, message: str) -> None:
    payload = _checkpoint_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        validate_checkpoint_contract(payload)


def test_checkpoint_contract_accepts_stage2_epoch1652_sit_b4_ema() -> None:
    metadata = validate_checkpoint_contract(_checkpoint_payload())

    assert metadata["stage"] == "stage2"
    assert metadata["stage_epoch_1based"] == 1652
    assert metadata["sit_patch_size"] == 4
    assert metadata["weight_source"] == "ema_model_state_dict"


def test_load_ema_generator_strictly_loads_only_ema_before_device_move(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    payload = _checkpoint_payload()
    calls: dict[str, object] = {}

    class FakeGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def load_state_dict(self, state_dict, strict: bool = True):
            calls["loaded_state"] = state_dict
            calls["strict"] = strict
            return super().load_state_dict(state_dict, strict=strict)

        def to(self, device):
            calls["device"] = str(device)
            return self

    def fake_load(path, *, map_location, weights_only):
        calls["load"] = (Path(path), map_location, weights_only)
        return payload

    monkeypatch.setattr(torch, "load", fake_load)
    generator, metadata = load_ema_generator(
        checkpoint,
        device=torch.device("cpu"),
        generator_builder=lambda config: FakeGenerator(),
    )

    assert calls["load"] == (checkpoint, "cpu", True)
    assert calls["loaded_state"] is payload["ema_model_state_dict"]
    assert calls["strict"] is True
    assert calls["device"] == "cpu"
    assert generator.weight.item() == 1.0
    assert metadata["weight_source"] == "ema_model_state_dict"


def _r8_guidance_config() -> dict:
    return {
        "checkpoint": EXPECTED_CHECKPOINT_PATH,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_model": "ema",
        "transport_condition": "learned_null_condition",
        "mode": "native",
        "sampling_seed": 1337,
        "e0_checkpoint": EXPECTED_E0_CHECKPOINT_PATH,
        "e0_sha256": "b" * 64,
        "edev_checkpoint": EXPECTED_EDEV_CHECKPOINT_PATH,
        "edev_sha256": "c" * 64,
        "vae_path": EXPECTED_VAE_PATH,
        "vae_digest": "d" * 64,
        "vae_scaling_factor": EXPECTED_VAE_SCALING_FACTOR,
        "index": "data/index/val.jsonl",
        "index_sha256": "e" * 64,
        "feature_source": "cached_features",
        "features": "artifacts/e0_features/val",
        "features_digest": "f" * 64,
        "sample_id_manifest": "artifacts/r8/sample_ids.jsonl",
        "sample_id_manifest_sha256": "1" * 64,
        "heldout_e1_checkpoint": "artifacts/checkpoints/e1.pt",
        "heldout_e1_sha256": "2" * 64,
        "heldout_e2_checkpoint": "artifacts/checkpoints/e2.pt",
        "heldout_e2_sha256": "3" * 64,
    }


def test_guidance_config_requires_exact_checkpoint_ema_and_learned_null() -> None:
    valid = _r8_guidance_config()
    assert validate_guidance_config(valid)["mode"] == "native"
    assert validate_guidance_config({**valid, "mode": "noise_oracle"})["mode"] == "initial_noise"

    for field, value, message in (
        ("checkpoint", "other.pt", "checkpoint"),
        ("checkpoint_model", "raw", "checkpoint_model"),
        ("transport_condition", "target_z0", "learned_null_condition"),
    ):
        invalid = dict(valid)
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            validate_guidance_config(invalid)


def test_guidance_config_carries_heldout_encoder_assets_without_rejecting_them() -> None:
    resolved = validate_guidance_config(_r8_guidance_config())

    assert resolved["heldout_e1_checkpoint"] == "artifacts/checkpoints/e1.pt"
    assert resolved["heldout_e2_sha256"] == "3" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", "other.pt"),
        ("e0_checkpoint", "other-e0.pt"),
        ("edev_checkpoint", "other-edev.pt"),
        ("vae_path", "other-vae"),
        ("vae_scaling_factor", 1.0),
    ],
)
def test_guidance_config_rejects_any_fixed_r8_asset_substitution(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        validate_guidance_config({**_r8_guidance_config(), field: value})


def test_guidance_config_requires_every_expected_digest() -> None:
    config = _r8_guidance_config()
    config.pop("features_digest")

    with pytest.raises(ValueError, match="features_digest"):
        validate_guidance_config(config)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schedule_contract_sha256(payload: dict) -> str:
    contract = dict(payload)
    contract.pop("schedule_contract_sha256", None)
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _write_locked_schedule_fixture(tmp_path: Path) -> tuple[Path, dict]:
    sample_manifest = tmp_path / "sample_ids.jsonl"
    _write_jsonl(sample_manifest, [{"sample_id": "sample-0"}])
    report_path = tmp_path / "semigroup_gate.json"
    report = {
        "gate_passed": True,
        "checkpoint_sha256": "a" * 64,
        "selected_t_cut": 0.25,
    }
    _write_json(report_path, report)
    schedule_path = tmp_path / "schedule.json"
    schedule = {
        "schema_version": 2,
        "gate_passed": True,
        "checkpoint_sha256": "a" * 64,
        "semigroup_report": str(report_path),
        "semigroup_report_sha256": _file_sha256(report_path),
        "sample_id_manifest": str(sample_manifest),
        "sample_id_manifest_sha256": _file_sha256(sample_manifest),
        "t_cut": 0.25,
        "guided_steps": 3,
        "guided_times": [1.0, 0.75, 0.5, 0.25],
        "unguided_tail_intervals": 2,
        "unguided_times": [0.25, 0.125, 0.0],
        "selection_rule": "smallest_numeric_t_cut_passing_all_registered_thresholds",
    }
    schedule["schedule_contract_sha256"] = _schedule_contract_sha256(schedule)
    _write_json(schedule_path, schedule)
    config = {
        "t_cut": 0.25,
        "schedule_manifest": str(schedule_path),
        "semigroup_report": str(report_path),
        "sample_id_manifest": str(sample_manifest),
    }
    return schedule_path, config


def test_locked_schedule_is_uniform_and_rejects_t_cut_or_hash_disagreement(tmp_path: Path) -> None:
    schedule_path, config = _write_locked_schedule_fixture(tmp_path)
    schedule = resolve_locked_schedule(
        config,
        checkpoint_sha256="a" * 64,
        explicit_t_cut=0.25,
    )

    assert schedule["guided_times"] == [1.0, 0.75, 0.5, 0.25]
    assert schedule["unguided_times"] == [0.25, 0.125, 0.0]
    for config_t_cut, explicit_t_cut, checkpoint_hash, message in (
        (0.5, 0.25, "a" * 64, "t_cut"),
        (0.25, 0.5, "a" * 64, "t_cut"),
        (0.25, 0.25, "b" * 64, "checkpoint"),
    ):
        with pytest.raises(ValueError, match=message):
            resolve_locked_schedule(
                {**config, "t_cut": config_t_cut},
                checkpoint_sha256=checkpoint_hash,
                explicit_t_cut=explicit_t_cut,
            )


def test_locked_schedule_rejects_report_manifest_or_self_digest_tampering(tmp_path: Path) -> None:
    schedule_path, config = _write_locked_schedule_fixture(tmp_path)
    report_path = Path(config["semigroup_report"])
    sample_manifest = Path(config["sample_id_manifest"])

    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="semigroup report SHA256"):
        resolve_locked_schedule(config, checkpoint_sha256="a" * 64)

    schedule_path, config = _write_locked_schedule_fixture(tmp_path)
    sample_manifest.write_text('{"sample_id":"replacement"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sample manifest SHA256"):
        resolve_locked_schedule(config, checkpoint_sha256="a" * 64)

    schedule_path, config = _write_locked_schedule_fixture(tmp_path)
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    payload["guided_times"][1] = 0.7
    _write_json(schedule_path, payload)
    with pytest.raises(ValueError, match="schedule contract SHA256"):
        resolve_locked_schedule(config, checkpoint_sha256="a" * 64)


def test_asset_contract_locks_all_r8_assets_features_and_input_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _r8_guidance_config()
    config["mode"] = "initial_noise"
    expected_by_path = {
        Path(config["checkpoint"]): config["checkpoint_sha256"],
        Path(config["e0_checkpoint"]): config["e0_sha256"],
        Path(config["edev_checkpoint"]): config["edev_sha256"],
        Path(config["vae_path"]): config["vae_digest"],
        Path(config["index"]): config["index_sha256"],
        Path(config["features"]): config["features_digest"],
        Path(config["sample_id_manifest"]): config["sample_id_manifest_sha256"],
        Path(config["heldout_e1_checkpoint"]): config["heldout_e1_sha256"],
        Path(config["heldout_e2_checkpoint"]): config["heldout_e2_sha256"],
    }
    monkeypatch.setattr(runner_module, "_digest_path", lambda path: expected_by_path[Path(path)])
    monkeypatch.setattr(
        runner_module,
        "read_ordered_sample_manifest",
        lambda path: [{"sample_id": "a"}, {"sample_id": "b"}],
    )

    contract = asset_contract_from_config(config)

    assert contract["e0"] == {
        "path": EXPECTED_E0_CHECKPOINT_PATH,
        "sha256": config["e0_sha256"],
    }
    assert contract["edev"]["path"] == EXPECTED_EDEV_CHECKPOINT_PATH
    assert contract["vae"]["digest"] == config["vae_digest"]
    assert contract["target_features"] == {
        "path": config["features"],
        "digest": config["features_digest"],
        "feature_source": "cached_features",
    }
    assert contract["sample_manifest"]["sha256"] == config["sample_id_manifest_sha256"]
    assert contract["heldout_e1"]["sha256"] == config["heldout_e1_sha256"]
    assert contract["seed"] == 1337
    assert contract["mode"] == "initial_noise"

    with pytest.raises(ValueError, match="E0.*digest"):
        asset_contract_from_config({**config, "e0_sha256": "9" * 64})


def test_shared_digest_cache_hashes_once_and_invalidates_on_stat_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"first")
    cache = tmp_path / "digests.json"
    real_digest = runner_module._digest_path
    calls = 0

    def counted_digest(path):
        nonlocal calls
        calls += 1
        return real_digest(path)

    monkeypatch.setattr(runner_module, "_digest_path", counted_digest)
    first_digest = hashlib.sha256(b"first").hexdigest()
    for _ in range(4):
        assert cached_asset_digest(asset, first_digest, cache) == first_digest
    assert calls == 1

    asset.write_bytes(b"second-content")
    second_digest = hashlib.sha256(b"second-content").hexdigest()
    assert cached_asset_digest(asset, second_digest, cache) == second_digest
    assert calls == 2


def test_completion_marker_is_checked_before_config_validation_or_asset_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "complete"
    output.mkdir()
    (output / "completion.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner_module,
        "asset_contract_from_config",
        lambda config: pytest.fail("completed run must not hash assets"),
    )

    with pytest.raises(FileExistsError, match="completed output"):
        run_guidance_from_config({}, output_dir=output)


def test_multishard_requires_cache_before_hash_or_nfe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hash_calls = 0
    nfe_calls = 0

    def counted_hash(config):
        nonlocal hash_calls
        hash_calls += 1
        return config

    def counted_nfe(**kwargs):
        nonlocal nfe_calls
        nfe_calls += 1
        return kwargs

    monkeypatch.setattr(runner_module, "asset_contract_from_config", counted_hash)
    monkeypatch.setattr(runner_module, "execute_guidance_mode", counted_nfe)
    config = {**_r8_guidance_config(), "device": "cuda:0"}

    with pytest.raises(ValueError, match="asset_digest_cache"):
        run_guidance_from_config(
            config,
            output_dir=tmp_path / "shard-0",
            shard_index=0,
            num_shards=4,
        )

    assert hash_calls == 0
    assert nfe_calls == 0


def test_shared_cache_must_be_in_repo_or_explicit_asset_root(tmp_path: Path) -> None:
    outside_cache = tmp_path / "cache" / "digests.json"
    config = {**_r8_guidance_config(), "asset_digest_cache": str(outside_cache)}

    with pytest.raises(ValueError, match="allowed.*root"):
        runner_module._prepare_sharded_guidance_config(config, shard_index=0, num_shards=4)

    resolved = runner_module._prepare_sharded_guidance_config(
        {**config, "asset_digest_cache_root": str(tmp_path)},
        shard_index=0,
        num_shards=4,
    )
    assert resolved["asset_digest_cache"] == str(outside_cache.resolve())


def test_four_shards_share_one_exact_cache_contract(tmp_path: Path) -> None:
    cache = tmp_path / "shared" / "digests.json"
    config = {
        **_r8_guidance_config(),
        "asset_digest_cache": str(cache),
        "asset_digest_cache_root": str(tmp_path),
    }

    for shard_index in range(4):
        runner_module._prepare_sharded_guidance_config(
            config,
            shard_index=shard_index,
            num_shards=4,
        )

    contract_path = cache.with_name(f"{cache.name}.shards.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["registered_shards"] == [0, 1, 2, 3]
    assert contract["contract"]["num_shards"] == 4

    with pytest.raises(ValueError, match="shard asset cache contract"):
        runner_module._prepare_sharded_guidance_config(
            {**config, "sampling_seed": 1338},
            shard_index=3,
            num_shards=4,
        )


def test_single_shard_does_not_require_asset_digest_cache() -> None:
    resolved = runner_module._prepare_sharded_guidance_config(
        _r8_guidance_config(),
        shard_index=0,
        num_shards=1,
    )

    assert "asset_digest_cache" not in resolved


class _DifferentiableVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def decode(self, latent):
        return SimpleNamespace(sample=latent[:, :3] * self.scale)


def test_build_frozen_runtime_uses_real_e0_loader_and_preserves_input_gradient(
    monkeypatch, tmp_path: Path
) -> None:
    generator = _FakeGenerator()
    loaded_encoder_paths: list[str] = []
    from safa.models import e0 as e0_module

    def generator_loader(path, *, device):
        del path, device
        return generator, {"weight_source": "ema_model_state_dict"}

    def fake_torch_load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        loaded_encoder_paths.append(str(path))
        return {
            "model_config": {"num_classes": 8, "embedding_dim": 3, "backbone": "fake"},
            "model_state_dict": {"scale": torch.ones(())},
        }

    def codec_builder(config, device):
        del config, device
        return LatentCodec(_DifferentiableVAE(), LatentCodecConfig(source="fake", scaling_factor=1.0))

    e0_path = tmp_path / "e0.pt"
    edev_path = tmp_path / "edev.pt"
    e0_path.touch()
    edev_path.touch()
    config = {
        "checkpoint": EXPECTED_CHECKPOINT_PATH,
        "e0_checkpoint": str(e0_path),
        "edev_checkpoint": str(edev_path),
        "heldout_e1_checkpoint": "/models/e1.pt",
        "heldout_e2_checkpoint": "/models/e2.pt",
        "phase": "calibration",
    }
    monkeypatch.setattr(torch, "load", fake_torch_load)
    monkeypatch.setattr(e0_module, "build_e0", lambda config, allow_random_init: _FakeE0())
    assets = {
        "e0": {"path": str(e0_path), "sha256": "0" * 64},
        "edev": {"path": str(edev_path), "sha256": "1" * 64},
        "vae": {"path": "/models/vae", "digest": "2" * 64, "scaling_factor": 1.0},
        "real_index": {"path": "/dataset/index.jsonl", "sha256": "3" * 64},
        "target_features": {
            "path": "/features",
            "digest": "4" * 64,
            "feature_source": "cached_features",
        },
        "sample_manifest": {
            "path": "/samples.jsonl",
            "sha256": "5" * 64,
            "sample_count": 2,
            "ordered_sample_id_sha256": "8" * 64,
        },
        "heldout_e1": {"path": "/models/e1.pt", "sha256": "6" * 64},
        "heldout_e2": {"path": "/models/e2.pt", "sha256": "7" * 64},
    }

    runtime = build_frozen_runtime(
        config,
        device=torch.device("cpu"),
        checkpoint_sha256="c" * 64,
        asset_contract=assets,
        generator_loader=generator_loader,
        encoder_loader=e0_module.load_e0_checkpoint,
        codec_builder=codec_builder,
    )

    assert loaded_encoder_paths == [str(e0_path), str(edev_path)]
    assert not runtime.codec.vae.training
    assert all(not parameter.requires_grad for parameter in runtime.codec.vae.parameters())
    assert all(not parameter.requires_grad for parameter in runtime.e0.parameters())
    assert runtime.edev is not None and all(not parameter.requires_grad for parameter in runtime.edev.parameters())
    latent = torch.randn(1, 4, 2, 2, requires_grad=True)
    embedding = runtime.e0(normalize_for_e0(runtime.codec.decode(latent)))["embedding"]
    embedding[:, 0].sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert torch.count_nonzero(latent.grad) > 0
    assert all(parameter.grad is None for parameter in runtime.e0.parameters())
    assert all(parameter.grad is None for parameter in runtime.codec.vae.parameters())


def test_manifest_sharding_and_resume_are_deterministic_and_strict(tmp_path: Path) -> None:
    manifest = tmp_path / "samples.jsonl"
    _write_jsonl(manifest, [{"sample_id": f"id-{index}"} for index in range(7)])
    rows = read_ordered_sample_manifest(manifest)

    assert [row["sample_id"] for row in deterministic_shard(rows, 0, 3)] == ["id-0", "id-3", "id-6"]
    assert [row["sample_id"] for row in deterministic_shard(rows, 1, 3)] == ["id-1", "id-4"]
    assert resume_remaining_ids(["id-0", "id-3", "id-6"], [{"sample_id": "id-0"}]) == ["id-3", "id-6"]
    with pytest.raises(ValueError, match="prefix"):
        resume_remaining_ids(["id-0", "id-3", "id-6"], [{"sample_id": "id-3"}])

    _write_jsonl(manifest, [{"sample_id": "duplicate"}, {"sample_id": "duplicate"}])
    with pytest.raises(ValueError, match="duplicate"):
        read_ordered_sample_manifest(manifest)


def test_session_memory_uses_maximum_across_resumed_sessions() -> None:
    sessions = [
        {"max_memory": {"allocated_bytes": 900, "reserved_bytes": 1200}},
        {"max_memory": {"allocated_bytes": 500, "reserved_bytes": 700}},
    ]

    assert aggregate_session_memory(sessions) == {
        "allocated_bytes": 900,
        "reserved_bytes": 1200,
    }


def test_finite_summary_reports_p05_p10_p90_and_p95() -> None:
    summary = runner_module._finite_summary([0.0, 10.0])

    assert summary["p05"] == pytest.approx(0.5)
    assert summary["p10"] == pytest.approx(1.0)
    assert summary["p90"] == pytest.approx(9.0)
    assert summary["p95"] == pytest.approx(9.5)


class _FakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.null_calls = 0

    def make_null_condition(self, *, batch_size: int, device, dtype):
        self.null_calls += 1
        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def flow_map(self, x, z, *, t, r):
        assert torch.count_nonzero(z) == 0
        horizon = torch.as_tensor(t, device=x.device, dtype=x.dtype) - torch.as_tensor(
            r, device=x.device, dtype=x.dtype
        )
        return torch.exp(-horizon) * x * self.scale


class _FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))


class _FakeCodec:
    def __init__(self) -> None:
        self.vae = _FakeVAE()

    def decode(self, latent):
        return torch.sigmoid(latent[:, :3] * self.vae.scale)


class _FakeE0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, image):
        return {"embedding": torch.nn.functional.normalize(image.mean(dim=(2, 3)) * self.scale, dim=1)}


def _fake_stack() -> tuple[_FakeGenerator, _FakeCodec, _FakeE0]:
    generator = _FakeGenerator()
    codec = _FakeCodec()
    e0 = _FakeE0()
    freeze_guidance_stack(generator, codec, e0)
    return generator, codec, e0


@pytest.mark.parametrize(
    ("mode", "expected_nfe"),
    [
        ("native", 1),
        ("semigroup", 7),
        ("official_head_current_xt", 5),
        ("paper_algorithm_split", 8),
        ("initial_noise", 3),
    ],
)
def test_execute_guidance_mode_supports_every_route_with_counted_nfe(mode: str, expected_nfe: int) -> None:
    generator, codec, e0 = _fake_stack()
    x_init = torch.randn(2, 4, 2, 2)
    target_z0 = torch.nn.functional.normalize(torch.rand(2, 3), dim=1)
    config = {
        "mode": mode,
        "split_times": [0.25, 0.5, 0.75],
        "sample_mode": "flow_map1",
        "optimization_mode": "paper_normalized_direct_autograd",
        "num_optim_iters": 1,
        "step_size": 0.25,
        "num_updates": 2,
        "eta": 0.25,
        "projection": "fixed_radius",
    }
    schedule = {
        "t_cut": 0.25,
        "guided_times": [1.0, 0.75, 0.5, 0.25],
        "unguided_times": [0.25, 0.125, 0.0],
    }

    result = execute_guidance_mode(
        config=config,
        generator=generator,
        codec=codec,
        e0=e0,
        x_init=x_init,
        target_z0=target_z0,
        schedule=None if mode == "initial_noise" else schedule,
    )

    assert result.latent.shape == x_init.shape
    assert result.nfe == expected_nfe
    assert generator.null_calls == 1
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in codec.vae.parameters())
    assert all(parameter.grad is None for parameter in e0.parameters())


def _guidance_runtime(*, edev=None, checkpoint_sha256: str = "c" * 64) -> GuidanceRuntime:
    generator, codec, e0 = _fake_stack()
    return GuidanceRuntime(
        generator=generator,
        codec=codec,
        e0=e0,
        edev=edev,
        device=torch.device("cpu"),
        checkpoint_path=Path(EXPECTED_CHECKPOINT_PATH),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_state={
            "stage": "stage2",
            "stage_epoch_1based": 1652,
            "sit_patch_size": 4,
            "weight_source": "ema_model_state_dict",
        },
        e0_checkpoint_path=Path("/models/e0.pt"),
        e0_checkpoint_sha256="0" * 64,
        edev_checkpoint_path=Path("/models/edev.pt"),
        edev_checkpoint_sha256="1" * 64,
        vae_path=Path("/models/vae"),
        vae_digest="2" * 64,
        vae_scaling_factor=0.18215,
        real_index_path=Path("/dataset/index.jsonl"),
        real_index_sha256="3" * 64,
        target_features_path=Path("/features"),
        target_features_digest="4" * 64,
        feature_source="cached_features",
        input_sample_manifest_path=Path("/samples.jsonl"),
        input_sample_manifest_sha256="5" * 64,
        input_sample_manifest_id_sha256="8" * 64,
        input_sample_manifest_count=3,
        heldout_e1={"path": "/models/e1.pt", "sha256": "6" * 64},
        heldout_e2={"path": "/models/e2.pt", "sha256": "7" * 64},
    )


def _generation_records(tmp_path: Path, count: int = 3) -> list[dict]:
    records = []
    for index in range(count):
        source = tmp_path / f"source-{index}.png"
        Image.new("RGB", (8, 8), color=(20 * index, 50, 100)).save(source)
        records.append(
            {
                "sample_id": f"sample-{index}",
                "source": str(source),
                "z": torch.nn.functional.normalize(torch.rand(3), dim=0),
            }
        )
    return records


def test_runner_writes_bound_rows_generation_result_and_prevents_overwrite(tmp_path: Path) -> None:
    runtime = _guidance_runtime()
    records = _generation_records(tmp_path)
    output_dir = tmp_path / "run"
    config = {
        "mode": "native",
        "sampling_seed": 9,
        "image_size": 2,
        "batch_size": 2,
        "phase": "full",
    }
    manifest = run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
    )

    rows = [json.loads(line) for line in (output_dir / "per_sample.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["sample-0", "sample-1", "sample-2"]
    assert all(
        set(row).issuperset(
            {
                "ordinal",
                "sample_id",
                "source",
                "generated",
                "native",
                "candidate_cosine",
                "native_cosine",
                "candidate_nfe",
                "native_nfe",
                "mode",
                "shard",
            }
        )
        for row in rows
    )
    assert [row["ordinal"] for row in rows] == [0, 1, 2]
    assert all(row["native"] == row["generated"] for row in rows)
    assert all(Path(row["generated"]).is_file() for row in rows)
    assert manifest["checkpoint"]["sha256"] == "c" * 64
    assert manifest["checkpoint"]["weight_source"] == "ema_model_state_dict"
    assert manifest["nfe"] == {"candidate": 1, "matched_native": 1}
    assert manifest["resume_contract"]["target_features"]["digest"] == "4" * 64
    assert manifest["resume_contract"]["input_sample_manifest"]["sha256"] == "5" * 64
    assert len(manifest["resume_contract"]["shard"]["ordered_sample_id_sha256"]) == 64
    assert manifest["sample_count"] == 3
    assert manifest["timing"]["generation_seconds"] > 0.0
    assert manifest["timing"]["io_seconds"] >= 0.0
    assert manifest["timing"]["wall_seconds"] >= manifest["timing"]["generation_seconds"]
    assert set(manifest["max_memory"]) == {"allocated_bytes", "reserved_bytes"}
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "generation_result.json").is_file()
    assert not (output_dir / "contact_sheet_columns.json").exists()

    with pytest.raises(FileExistsError, match="completed output"):
        run_guidance_records(
            config=config,
            records=records,
            runtime=runtime,
            output_dir=output_dir,
            shard_index=0,
            num_shards=1,
        )


def test_calibration_runner_requires_edev_before_writing_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires.*Edev"):
        run_guidance_records(
            config={"mode": "native", "sampling_seed": 1, "phase": "calibration"},
            records=_generation_records(tmp_path, count=1),
            runtime=_guidance_runtime(edev=None),
            output_dir=tmp_path / "missing-edev",
            shard_index=0,
            num_shards=1,
        )
    assert not (tmp_path / "missing-edev" / "per_sample.jsonl").exists()

    with pytest.raises(ValueError, match="calibration requires.*contact sheets"):
        run_guidance_records(
            config={
                "mode": "native",
                "sampling_seed": 1,
                "phase": "calibration",
                "contact_sheets": False,
            },
            records=_generation_records(tmp_path, count=1),
            runtime=_guidance_runtime(edev=_FakeE0().eval().requires_grad_(False)),
            output_dir=tmp_path / "disabled-contacts",
            shard_index=0,
            num_shards=1,
        )


def test_candidate_writes_matched_native_edev_traces_and_pil_contact_sheets(tmp_path: Path) -> None:
    edev = _FakeE0().eval().requires_grad_(False)
    runtime = _guidance_runtime(edev=edev)
    records = _generation_records(tmp_path, count=2)
    output_dir = tmp_path / "candidate"
    config = {
        "mode": "initial_noise",
        "sampling_seed": 11,
        "image_size": 2,
        "pixel_image_size": 8,
        "batch_size": 2,
        "num_updates": 2,
        "eta": 0.25,
        "projection": "fixed_radius",
        "phase": "calibration",
        "contact_sheets": True,
        "contact_sheet_rows": 1,
    }

    result = run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
    )

    rows = [json.loads(line) for line in (output_dir / "per_sample.jsonl").read_text().splitlines()]
    assert all(row["mode"] == "initial_noise" for row in rows)
    assert all(Path(row["generated"]).is_file() and Path(row["native"]).is_file() for row in rows)
    assert all(row["candidate_nfe"] == 3 and row["native_nfe"] == 1 for row in rows)
    assert all("edev_cosine" in row and "native_edev_cosine" in row for row in rows)
    assert all(item["kind"] == "initial_noise" for item in rows[0]["candidate_trace"])
    assert rows[0]["native_trace"] == [
        {"kind": "matched_native", "r": 0.0, "t": 1.0}
    ]
    assert result["nfe"] == {"candidate": 3, "matched_native": 1}
    assert set(result["cosine"]) == {
        "candidate_e0_target",
        "native_e0_target",
        "candidate_edev_source",
        "native_edev_source",
    }
    assert all({"p05", "p10", "p90", "p95"}.issubset(summary) for summary in result["cosine"].values())
    contact_manifest = json.loads((output_dir / "contact_sheet_columns.json").read_text())
    assert contact_manifest["columns"] == ["source", "native", "candidate"]
    assert len(contact_manifest["pages"]) == 2
    assert all(Path(page["path"]).is_file() for page in contact_manifest["pages"])
    assert [page["sample_ids"] for page in contact_manifest["pages"]] == [["sample-0"], ["sample-1"]]
    with Image.open(contact_manifest["pages"][0]["path"]) as page:
        assert page.mode == "RGB"
        assert page.size == (128 * 3, 128)
        assert page.getpixel((0, 0)) == (0, 50, 100)


@pytest.mark.parametrize("batch_size", [4, 3])
def test_runner_serializes_initial_noise_channel_diagnostics_per_sample(
    tmp_path: Path, batch_size: int
) -> None:
    output = tmp_path / f"diagnostics-{batch_size}"
    run_guidance_records(
        config={
            "mode": "initial_noise",
            "sampling_seed": 5,
            "image_size": 2,
            "batch_size": batch_size,
            "num_updates": 1,
            "eta": 0.25,
            "projection": "fixed_radius",
            "phase": "full",
        },
        records=_generation_records(tmp_path, count=batch_size),
        runtime=_guidance_runtime(),
        output_dir=output,
        shard_index=0,
        num_shards=1,
    )

    rows = [json.loads(line) for line in (output / "per_sample.jsonl").read_text().splitlines()]
    assert len(rows) == batch_size
    assert all(len(row["route_diagnostics"]["channel_mean"]) == 4 for row in rows)
    assert all(len(row["route_diagnostics"]["channel_std"]) == 4 for row in rows)
    assert all("final_noise" not in row["route_diagnostics"] for row in rows)


def test_resume_replaces_only_exact_crash_orphan_and_aggregates_all_rows(tmp_path: Path) -> None:
    runtime = _guidance_runtime()
    records = _generation_records(tmp_path, count=3)
    output_dir = tmp_path / "resume"
    config = {
        "mode": "initial_noise",
        "sampling_seed": 17,
        "image_size": 2,
        "batch_size": 1,
        "num_updates": 1,
        "eta": 0.25,
        "projection": "fixed_radius",
        "phase": "full",
    }
    first = run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
    )
    rows = [json.loads(line) for line in (output_dir / "per_sample.jsonl").read_text().splitlines()]
    orphan_candidate = Path(rows[-1]["generated"])
    orphan_native = Path(rows[-1]["native"])
    expected_candidate = orphan_candidate.read_bytes()
    expected_native = orphan_native.read_bytes()
    for artifact in ("run_manifest.json", "generation_result.json", "completion.json"):
        (output_dir / artifact).unlink()
    _write_jsonl(output_dir / "per_sample.jsonl", rows[:-1])

    with pytest.raises(ValueError, match="resume contract"):
        run_guidance_records(
            config=config,
            records=records,
            runtime=replace(runtime, e0_checkpoint_sha256="9" * 64),
            output_dir=output_dir,
            shard_index=0,
            num_shards=1,
        )

    resumed = run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
    )

    assert orphan_candidate.read_bytes() == expected_candidate
    assert orphan_native.read_bytes() == expected_native
    assert resumed["timing"]["resumed_count"] == 2
    assert resumed["timing"]["generated_this_invocation"] == 1
    assert resumed["timing"]["generation_seconds"] == pytest.approx(
        sum(row["candidate_generation_seconds"] + row["native_generation_seconds"] for row in rows),
        rel=0.25,
    )
    assert resumed["timing"]["generation_seconds"] >= first["timing"]["generation_seconds"] * 0.5


def test_real_png_before_row_crash_window_resumes_transactionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _guidance_runtime()
    records = _generation_records(tmp_path, count=1)
    output = tmp_path / "crash-window"
    config = {
        "mode": "initial_noise",
        "sampling_seed": 23,
        "image_size": 2,
        "batch_size": 1,
        "num_updates": 1,
        "eta": 0.25,
        "projection": "fixed_radius",
        "phase": "full",
    }
    real_append = runner_module._append_jsonl

    def crash_before_row(path, row):
        if Path(path).name == "per_sample.jsonl":
            (output / ".tmp-per-sample-interrupted.jsonl").write_text(
                '{"half"', encoding="utf-8"
            )
            raise RuntimeError("injected row commit crash")
        return real_append(path, row)

    monkeypatch.setattr(runner_module, "_append_jsonl", crash_before_row)
    with pytest.raises(RuntimeError, match="injected row commit crash"):
        run_guidance_records(
            config=config,
            records=records,
            runtime=runtime,
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )
    assert len(list((output / "generated_images").glob("*.png"))) == 1
    assert len(list((output / "native_images").glob("*.png"))) == 1
    assert not (output / "per_sample.jsonl").exists()
    assert (output / "session_journal.json").is_file()
    assert (output / ".tmp-per-sample-interrupted.jsonl").is_file()

    monkeypatch.setattr(runner_module, "_append_jsonl", real_append)
    result = run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output,
        shard_index=0,
        num_shards=1,
    )

    rows = [json.loads(line) for line in (output / "per_sample.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["ordinal"] == 0
    assert result["timing"]["session_count"] == 2
    assert not (output / "session_journal.json").exists()
    assert not (output / ".tmp-per-sample-interrupted.jsonl").exists()


def test_session_history_commit_is_idempotent_when_unlink_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "history-crash"
    real_unlink = Path.unlink
    injected = False

    def crash_after_history(path, *args, **kwargs):
        nonlocal injected
        if (
            Path(path) == output / "session_journal.json"
            and (output / "session_history.jsonl").exists()
            and not injected
        ):
            injected = True
            raise RuntimeError("injected journal unlink crash")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", crash_after_history)
    with pytest.raises(RuntimeError, match="journal unlink crash"):
        run_guidance_records(
            config={"mode": "native", "sampling_seed": 4, "image_size": 2, "phase": "full"},
            records=_generation_records(tmp_path, count=1),
            runtime=_guidance_runtime(),
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )
    first_history = [
        json.loads(line) for line in (output / "session_history.jsonl").read_text().splitlines()
    ]
    assert len(first_history) == 1
    assert (output / "session_journal.json").is_file()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    result = run_guidance_records(
        config={"mode": "native", "sampling_seed": 4, "image_size": 2, "phase": "full"},
        records=_generation_records(tmp_path, count=1),
        runtime=_guidance_runtime(),
        output_dir=output,
        shard_index=0,
        num_shards=1,
    )
    final_history = [
        json.loads(line) for line in (output / "session_history.jsonl").read_text().splitlines()
    ]
    session_ids = [row["session_id"] for row in final_history]
    assert len(final_history) == result["timing"]["session_count"] == 2
    assert len(session_ids) == len(set(session_ids))


def test_partial_finalization_between_result_jsons_rebuilds_until_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "partial-final"
    real_atomic_json = runner_module._atomic_write_json
    injected = False

    def fail_between_final_jsons(path, payload, **kwargs):
        nonlocal injected
        if Path(path).name == "run_manifest.json" and not injected:
            injected = True
            raise RuntimeError("injected finalization crash")
        return real_atomic_json(path, payload, **kwargs)

    monkeypatch.setattr(runner_module, "_atomic_write_json", fail_between_final_jsons)
    with pytest.raises(RuntimeError, match="finalization crash"):
        run_guidance_records(
            config={"mode": "native", "sampling_seed": 2, "image_size": 2, "phase": "full"},
            records=_generation_records(tmp_path, count=1),
            runtime=_guidance_runtime(),
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )
    assert (output / "generation_result.json").is_file()
    assert not (output / "completion.json").exists()

    monkeypatch.setattr(runner_module, "_atomic_write_json", real_atomic_json)
    result = run_guidance_records(
        config={"mode": "native", "sampling_seed": 2, "image_size": 2, "phase": "full"},
        records=_generation_records(tmp_path, count=1),
        runtime=_guidance_runtime(),
        output_dir=output,
        shard_index=0,
        num_shards=1,
    )
    assert result["status"] == "complete"
    assert (output / "completion.json").is_file()
    with pytest.raises(FileExistsError, match="completed output"):
        run_guidance_records(
            config={"mode": "native", "sampling_seed": 2, "image_size": 2, "phase": "full"},
            records=_generation_records(tmp_path, count=1),
            runtime=_guidance_runtime(),
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )


def test_resume_rejects_swapped_row_bindings_and_unowned_png(tmp_path: Path) -> None:
    runtime = _guidance_runtime()
    records = _generation_records(tmp_path, count=2)
    output_dir = tmp_path / "invalid-resume"
    config = {"mode": "native", "sampling_seed": 3, "image_size": 2, "batch_size": 1, "phase": "full"}
    run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
    )
    rows = [json.loads(line) for line in (output_dir / "per_sample.jsonl").read_text().splitlines()]
    for artifact in ("run_manifest.json", "generation_result.json", "completion.json"):
        (output_dir / artifact).unlink()
    rows[0]["source"], rows[1]["source"] = rows[1]["source"], rows[0]["source"]
    _write_jsonl(output_dir / "per_sample.jsonl", rows)

    with pytest.raises(ValueError, match="row binding"):
        run_guidance_records(
            config=config,
            records=records,
            runtime=runtime,
            output_dir=output_dir,
            shard_index=0,
            num_shards=1,
        )

    _write_jsonl(output_dir / "per_sample.jsonl", [])
    extra = output_dir / "generated_images" / "not-owned.png"
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(extra)
    with pytest.raises(ValueError, match="extra PNG"):
        run_guidance_records(
            config=config,
            records=records,
            runtime=runtime,
            output_dir=output_dir,
            shard_index=0,
            num_shards=1,
        )


def test_native_mode_rejects_unowned_native_directory(tmp_path: Path) -> None:
    output = tmp_path / "native-extra"
    extra_dir = output / "native_images"
    extra_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(extra_dir / "extra.png")

    with pytest.raises(ValueError, match="unexpected files"):
        run_guidance_records(
            config={"mode": "native", "sampling_seed": 1, "phase": "full"},
            records=_generation_records(tmp_path, count=1),
            runtime=_guidance_runtime(),
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )


def test_runner_rejects_owned_subdirectory_symlink_escape(tmp_path: Path) -> None:
    output = tmp_path / "symlink-run"
    external = tmp_path / "external"
    output.mkdir()
    external.mkdir()
    (output / "generated_images").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        run_guidance_records(
            config={"mode": "native", "sampling_seed": 1, "phase": "full"},
            records=_generation_records(tmp_path, count=1),
            runtime=_guidance_runtime(),
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )
    assert list(external.iterdir()) == []



def test_semigroup_mode_writes_per_sample_semigroup_json(tmp_path: Path) -> None:
    generator, codec, e0 = _fake_stack()
    runtime = GuidanceRuntime(
        generator=generator,
        codec=codec,
        e0=e0,
        device=torch.device("cpu"),
        checkpoint_path=Path(EXPECTED_CHECKPOINT_PATH),
        checkpoint_sha256="d" * 64,
        checkpoint_state={"weight_source": "ema_model_state_dict"},
    )
    record = {
        "sample_id": "semigroup-sample",
        "source": "/dataset/source.png",
        "z": torch.nn.functional.normalize(torch.rand(3), dim=0),
    }
    output_dir = tmp_path / "semigroup"

    manifest = run_guidance_records(
        config={
            "mode": "semigroup",
            "sampling_seed": 9,
            "image_size": 2,
            "batch_size": 1,
            "split_times": [0.25, 0.5, 0.75],
            "phase": "full",
        },
        records=[record],
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
    )

    payload = json.loads((output_dir / "semigroup.json").read_text())
    assert manifest["nfe"] == {"candidate": 7, "matched_native": 1}
    assert payload["rows"][0]["sample_id"] == "semigroup-sample"
    assert set(payload["rows"][0]["splits"]) == {"0.25", "0.5", "0.75"}
    assert set(payload["rows"][0]["splits"]["0.25"]) == {
        "latent_residual",
        "decoded_pixel_l1",
        "decoded_psnr",
        "endpoint_e0_cosine",
        "decoded_image",
    }
    assert Path(payload["rows"][0]["splits"]["0.25"]["decoded_image"]).is_file()


def test_guidance_cli_accepts_required_config_output_and_shard_controls() -> None:
    path = REPO_ROOT / "scripts" / "run_meanflow_flow_map_guidance.py"
    spec = importlib.util.spec_from_file_location("run_meanflow_flow_map_guidance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--config",
            "config.yaml",
            "--output-dir",
            "output",
            "--shard-index",
            "2",
            "--num-shards",
            "4",
        ]
    )
    assert args.config == Path("config.yaml")
    assert args.output_dir == Path("output")
    assert args.shard_index == 2
    assert args.num_shards == 4
    with pytest.raises(SystemExit):
        module.parse_args(
            ["--config", "config.yaml", "--output-dir", "output", "--allow-overwrite"]
        )
