from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.evaluation.meanflow_guidance_runner import (  # noqa: E402
    EXPECTED_CHECKPOINT_PATH,
    GuidanceRuntime,
    deterministic_shard,
    execute_guidance_mode,
    load_ema_generator,
    read_ordered_sample_manifest,
    resolve_locked_schedule,
    resume_remaining_ids,
    run_guidance_records,
    validate_checkpoint_contract,
    validate_guidance_config,
)
from safa.guidance.meanflow_flow_map import freeze_guidance_stack  # noqa: E402


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


def test_guidance_config_requires_exact_checkpoint_ema_and_learned_null() -> None:
    valid = {
        "checkpoint": EXPECTED_CHECKPOINT_PATH,
        "checkpoint_model": "ema",
        "transport_condition": "learned_null_condition",
        "mode": "native",
        "sampling_seed": 1337,
    }
    assert validate_guidance_config(valid)["mode"] == "native"

    for field, value, message in (
        ("checkpoint", "other.pt", "checkpoint"),
        ("checkpoint_model", "raw", "checkpoint_model"),
        ("transport_condition", "target_z0", "learned_null_condition"),
    ):
        invalid = dict(valid)
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            validate_guidance_config(invalid)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_locked_schedule_is_uniform_and_rejects_t_cut_or_hash_disagreement(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedule.json"
    _write_json(
        schedule_path,
        {"t_cut": 0.25, "checkpoint_sha256": "a" * 64, "gate_passed": True},
    )
    schedule = resolve_locked_schedule(
        {"t_cut": 0.25, "schedule_manifest": str(schedule_path)},
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
                {"t_cut": config_t_cut, "schedule_manifest": str(schedule_path)},
                checkpoint_sha256=checkpoint_hash,
                explicit_t_cut=explicit_t_cut,
            )


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
        ("noise_oracle", 3),
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
        schedule=None if mode == "noise_oracle" else schedule,
    )

    assert result.latent.shape == x_init.shape
    assert result.nfe == expected_nfe
    assert generator.null_calls == 1
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in codec.vae.parameters())
    assert all(parameter.grad is None for parameter in e0.parameters())


def test_runner_writes_png_rows_manifest_and_prevents_overwrite(tmp_path: Path) -> None:
    generator, codec, e0 = _fake_stack()
    runtime = GuidanceRuntime(
        generator=generator,
        codec=codec,
        e0=e0,
        device=torch.device("cpu"),
        checkpoint_path=Path(EXPECTED_CHECKPOINT_PATH),
        checkpoint_sha256="c" * 64,
        checkpoint_state={
            "stage": "stage2",
            "stage_epoch_1based": 1652,
            "sit_patch_size": 4,
            "weight_source": "ema_model_state_dict",
        },
    )
    records = [
        {
            "sample_id": f"sample-{index}",
            "source": f"/dataset/source-{index}.png",
            "z": torch.nn.functional.normalize(torch.rand(3), dim=0),
        }
        for index in range(3)
    ]
    output_dir = tmp_path / "run"
    config = {"mode": "native", "sampling_seed": 9, "image_size": 2, "batch_size": 2}
    manifest = run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
        allow_overwrite=False,
    )

    rows = [json.loads(line) for line in (output_dir / "per_sample.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["sample-0", "sample-1", "sample-2"]
    assert all(set(row).issuperset({"sample_id", "source", "generated", "cosine", "mode", "shard"}) for row in rows)
    assert all(Path(row["generated"]).is_file() for row in rows)
    assert manifest["checkpoint"]["sha256"] == "c" * 64
    assert manifest["checkpoint"]["weight_source"] == "ema_model_state_dict"
    assert manifest["nfe"] == 1
    assert manifest["sample_count"] == 3
    assert manifest["timing"]["images_per_second"] > 0.0
    assert set(manifest["max_memory"]) == {"allocated_bytes", "reserved_bytes"}
    assert (output_dir / "run_manifest.json").is_file()

    with pytest.raises(FileExistsError, match="completed output"):
        run_guidance_records(
            config=config,
            records=records,
            runtime=runtime,
            output_dir=output_dir,
            shard_index=0,
            num_shards=1,
            allow_overwrite=False,
        )

    last_path = Path(rows[-1]["generated"])
    expected_last_bytes = last_path.read_bytes()
    (output_dir / "run_manifest.json").unlink()
    _write_jsonl(output_dir / "per_sample.jsonl", rows[:-1])
    last_path.unlink()
    with pytest.raises(ValueError, match="resume contract"):
        run_guidance_records(
            config={**config, "sampling_seed": 10},
            records=records,
            runtime=runtime,
            output_dir=output_dir,
            shard_index=0,
            num_shards=1,
            allow_overwrite=False,
        )
    resumed = run_guidance_records(
        config=config,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
        allow_overwrite=False,
    )

    assert last_path.read_bytes() == expected_last_bytes
    assert resumed["timing"]["resumed_count"] == 2
    assert resumed["timing"]["generated_this_invocation"] == 1


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
        },
        records=[record],
        runtime=runtime,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
        allow_overwrite=False,
    )

    payload = json.loads((output_dir / "semigroup.json").read_text())
    assert manifest["nfe"] == 7
    assert payload["rows"][0]["sample_id"] == "semigroup-sample"
    assert set(payload["rows"][0]["splits"]) == {"0.25", "0.5", "0.75"}
    assert set(payload["rows"][0]["splits"]["0.25"]) == {
        "latent_residual",
        "decoded_pixel_l1",
        "decoded_psnr",
        "endpoint_e0_cosine",
    }


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
            "--allow-overwrite",
        ]
    )
    assert args.config == Path("config.yaml")
    assert args.output_dir == Path("output")
    assert args.shard_index == 2
    assert args.num_shards == 4
    assert args.allow_overwrite is True
