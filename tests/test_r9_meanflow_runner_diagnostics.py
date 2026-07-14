from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.evaluation import meanflow_guidance_runner as runner  # noqa: E402
from safa.evaluation.meanflow_guidance_runner import (  # noqa: E402
    EXPECTED_CHECKPOINT_PATH,
    EXPECTED_E0_CHECKPOINT_PATH,
    EXPECTED_EDEV_CHECKPOINT_PATH,
    EXPECTED_VAE_PATH,
    EXPECTED_VAE_SCALING_FACTOR,
    GuidanceRuntime,
    R9_GUIDANCE_INTERVAL_CONTRACT_FIELD,
    R9_PHASE_CONTRACT_FIELD,
    execute_guidance_mode,
    run_guidance_records,
    validate_guidance_config,
    validate_r9_interval_guidance_config,
    validate_r9_phase_contract,
)
from safa.evaluation.r9_determinism import (  # noqa: E402
    R9_ATTENTION_BACKEND,
    R9_DETERMINISM_POLICY,
    R9_DETERMINISM_POLICY_SHA256,
    R9_EXPERIMENT_CONTRACT,
    apply_r9_strict_cuda_determinism,
)
from safa.guidance.meanflow_flow_map import freeze_guidance_stack  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCKED_SCHEDULE = {
    "t_cut": 0.25,
    "guided_times": [1.0, 0.75, 0.5, 0.25],
    "unguided_times": [0.25, 0.125, 0.0],
}


class _FakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))

    def make_null_condition(self, *, batch_size: int, device, dtype):
        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def flow_map(self, x, z, *, t, r):
        assert torch.count_nonzero(z) == 0
        horizon = torch.as_tensor(t, device=x.device, dtype=x.dtype) - torch.as_tensor(
            r, device=x.device, dtype=x.dtype
        )
        return x - horizon * self.scale * (x.square() + 0.25)


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
        embedding = image.mean(dim=(2, 3)) * self.scale
        return {"embedding": torch.nn.functional.normalize(embedding, dim=1)}


def _fake_stack():
    generator = _FakeGenerator()
    codec = _FakeCodec()
    e0 = _FakeE0()
    freeze_guidance_stack(generator, codec, e0)
    return generator, codec, e0


def _asset_complete_config() -> dict:
    return {
        "checkpoint": EXPECTED_CHECKPOINT_PATH,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_model": "ema",
        "transport_condition": "learned_null_condition",
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
        "sample_id_manifest": "artifacts/r9/sample_ids.jsonl",
        "sample_id_manifest_sha256": "1" * 64,
        "heldout_e1_checkpoint": "artifacts/checkpoints/e1.pt",
        "heldout_e1_sha256": "2" * 64,
        "heldout_e2_checkpoint": "artifacts/checkpoints/e2.pt",
        "heldout_e2_sha256": "3" * 64,
    }


def _r9_config(*, mode="official_head_current_xt", collect=True, active=None) -> dict:
    config = {
        **_asset_complete_config(),
        "experiment_contract": R9_EXPERIMENT_CONTRACT,
        "determinism_policy": dict(R9_DETERMINISM_POLICY),
        "determinism_policy_sha256": R9_DETERMINISM_POLICY_SHA256,
        "attention_backend": R9_ATTENTION_BACKEND,
        "phase": "diagnose",
        "mode": mode,
        "step_size": 0.25,
        "active_guidance_intervals": ["I1", "I3"] if active is None else active,
        "collect_interval_diagnostics": collect,
    }
    if mode == "official_head_current_xt":
        config.update(
            {
                "sample_mode": "flow_map2",
                "optimization_mode": "paper_normalized_direct_autograd",
                "num_optim_iters": 1,
            }
        )
    return config


def _execution_config(*, mode: str, collect: bool) -> dict:
    config = _r9_config(mode=mode, collect=collect)
    for field in tuple(config):
        if field not in {
            "experiment_contract",
            "mode",
            "step_size",
            "active_guidance_intervals",
            "collect_interval_diagnostics",
            "sample_mode",
            "optimization_mode",
            "num_optim_iters",
        }:
            config.pop(field)
    return config


def _runtime() -> GuidanceRuntime:
    generator, codec, e0 = _fake_stack()
    edev = _FakeE0().eval().requires_grad_(False)
    return GuidanceRuntime(
        generator=generator,
        codec=codec,
        e0=e0,
        device=torch.device("cpu"),
        checkpoint_path=Path(EXPECTED_CHECKPOINT_PATH),
        checkpoint_sha256="a" * 64,
        checkpoint_state={"stage": "stage2"},
        edev=edev,
        e0_checkpoint_path=Path("/models/e0.pt"),
        e0_checkpoint_sha256="b" * 64,
        edev_checkpoint_path=Path("/models/edev.pt"),
        edev_checkpoint_sha256="c" * 64,
        vae_path=Path("/models/vae"),
        vae_digest="d" * 64,
        vae_scaling_factor=EXPECTED_VAE_SCALING_FACTOR,
        real_index_path=Path("/data/index.jsonl"),
        real_index_sha256="e" * 64,
        target_features_path=Path("/features"),
        target_features_digest="f" * 64,
        feature_source="cached_features",
        input_sample_manifest_path=Path("/samples.jsonl"),
        input_sample_manifest_sha256="1" * 64,
        input_sample_manifest_id_sha256="4" * 64,
        input_sample_manifest_count=2,
        heldout_e1={"path": "/models/e1.pt", "sha256": "2" * 64},
        heldout_e2={"path": "/models/e2.pt", "sha256": "3" * 64},
    )


def _records(tmp_path: Path) -> list[dict]:
    rows = []
    for index in range(2):
        source = tmp_path / f"source-{index}.png"
        Image.new("RGB", (8, 8), color=(10 * index, 50, 90)).save(source)
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "source": str(source),
                "z": torch.nn.functional.normalize(
                    torch.tensor([1.0, 2.0, 3.0]), dim=0
                ),
            }
        )
    return rows


@contextmanager
def _strict_r9_torch_state():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    contract = apply_r9_strict_cuda_determinism(_r9_config(), torch_module=torch)
    try:
        yield contract
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        if previous_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas


def _run_records_config(*, collect: bool, execution_contract: dict) -> dict:
    config = _r9_config(mode="paper_algorithm_split", collect=collect)
    interval_contract = validate_r9_interval_guidance_config(config)
    assert interval_contract is not None
    config.update(
        {
            "r9_execution_contract": execution_contract,
            R9_GUIDANCE_INTERVAL_CONTRACT_FIELD: interval_contract,
            "locked_schedule": dict(LOCKED_SCHEDULE),
            "image_size": 2,
            "batch_size": 2,
            "phase": "full",
        }
    )
    config[R9_PHASE_CONTRACT_FIELD] = validate_r9_phase_contract(config)
    return config


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config.__setitem__("active_guidance_intervals", ("I1",)),
            "YAML list",
        ),
        (
            lambda config: config.__setitem__("active_guidance_intervals", ["I4"]),
            "unknown",
        ),
        (
            lambda config: config.__setitem__(
                "active_guidance_intervals", ["I1", "I1"]
            ),
            "duplicates",
        ),
        (
            lambda config: config.__setitem__(
                "active_guidance_intervals", ["I3", "I1"]
            ),
            "canonical",
        ),
        (
            lambda config: config.__setitem__("collect_interval_diagnostics", 1),
            "boolean",
        ),
        (lambda config: config.__setitem__("sample_mode", "flow_map1"), "flow_map2"),
        (
            lambda config: config.__setitem__("optimization_mode", "official_adam"),
            "paper_normalized",
        ),
        (lambda config: config.__setitem__("num_optim_iters", 2), "num_optim_iters=1"),
    ],
)
def test_r9_config_strictly_rejects_invalid_interval_or_mode_contract(
    mutate, message
) -> None:
    config = _r9_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        validate_guidance_config(config)


def test_r8_config_rejects_r9_only_interval_fields() -> None:
    config = {**_asset_complete_config(), "mode": "native"}
    config["active_guidance_intervals"] = ["I1"]

    with pytest.raises(ValueError, match="R9 experiment contract"):
        validate_guidance_config(config)


@pytest.mark.parametrize("mode", ["official_head_current_xt", "paper_algorithm_split"])
def test_r9_validated_config_binds_interval_contract(mode) -> None:
    config = _r9_config(mode=mode)
    validated = validate_guidance_config(config)
    contract = validated[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]

    assert contract["active_guidance_intervals"] == ["I1", "I3"]
    assert contract["collect_interval_diagnostics"] is True
    assert contract["expected_algorithm_nfe"] == 7
    assert contract["expected_diagnostic_nfe"] == 8


@pytest.mark.parametrize("mode", ["official_head_current_xt", "paper_algorithm_split"])
def test_runner_diagnostic_toggle_preserves_latent_png_and_algorithm_trace(
    mode, tmp_path
) -> None:
    generator, codec, e0 = _fake_stack()
    x_init = torch.linspace(-0.7, 0.8, 32).reshape(2, 4, 2, 2)
    target = torch.nn.functional.normalize(
        torch.tensor([[1.0, 2.0, 3.0]]).repeat(2, 1), dim=1
    )
    plain = execute_guidance_mode(
        config=_execution_config(mode=mode, collect=False),
        generator=generator,
        codec=codec,
        e0=e0,
        x_init=x_init,
        target_z0=target,
        schedule=LOCKED_SCHEDULE,
    )
    generator, codec, e0 = _fake_stack()
    diagnosed = execute_guidance_mode(
        config=_execution_config(mode=mode, collect=True),
        generator=generator,
        codec=codec,
        e0=e0,
        x_init=x_init,
        target_z0=target,
        schedule=LOCKED_SCHEDULE,
    )

    assert torch.equal(plain.latent, diagnosed.latent)
    assert plain.nfe == diagnosed.nfe == 7
    assert (
        plain.diagnostics["flow_map_trace"] == diagnosed.diagnostics["flow_map_trace"]
    )
    assert plain.diagnostics["diagnostic_nfe"] == 0
    assert diagnosed.diagnostics["diagnostic_nfe"] == 8
    plain_png = tmp_path / f"{mode}-plain.png"
    diagnosed_png = tmp_path / f"{mode}-diagnosed.png"
    runner._atomic_save_image(codec.decode(plain.latent)[0], plain_png)
    runner._atomic_save_image(codec.decode(diagnosed.latent)[0], diagnosed_png)
    assert plain_png.read_bytes() == diagnosed_png.read_bytes()


def test_per_sample_serializer_recurses_only_interval_mapping_sequences() -> None:
    payload = {
        "interval_diagnostics": [
            {
                "interval_id": "I1",
                "active": True,
                "gradient_norm": torch.tensor([1.0, 2.0]),
            },
            {
                "interval_id": "I2",
                "active": False,
                "gradient_norm": torch.tensor([3.0, 4.0]),
            },
        ],
        "flow_map_trace": [{"kind": "algorithm", "t": 1.0, "r": 0.75}],
    }

    converted = runner._per_sample_diagnostics(payload, 1, 2)

    assert converted["interval_diagnostics"] == [
        {"interval_id": "I1", "active": True, "gradient_norm": 2.0},
        {"interval_id": "I2", "active": False, "gradient_norm": 4.0},
    ]
    assert "flow_map_trace" not in converted


def test_r9_generation_and_resume_bind_separate_nfe_and_diagnostic_trace(
    tmp_path,
) -> None:
    records = _records(tmp_path)
    with _strict_r9_torch_state() as execution_contract:
        config = _run_records_config(
            collect=True, execution_contract=execution_contract
        )
        output = tmp_path / "r9-diagnostics"
        manifest = run_guidance_records(
            config=config,
            records=records,
            runtime=_runtime(),
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )

        rows = [
            json.loads(line)
            for line in (output / "per_sample.jsonl").read_text().splitlines()
        ]
        assert all(
            row["candidate_nfe"] == row["candidate_algorithm_nfe"] == 7 for row in rows
        )
        assert all(row["candidate_diagnostic_nfe"] == 8 for row in rows)
        assert all(len(row["candidate_trace"]) == 7 for row in rows)
        assert all(len(row["candidate_diagnostic_trace"]) == 8 for row in rows)
        assert all(
            set(row["route_diagnostics"]["interval_diagnostics"]) == {"I1", "I2", "I3"}
            for row in rows
        )
        assert manifest["nfe"] == {
            "candidate": 7,
            "matched_native": 1,
            "candidate_algorithm": 7,
            "candidate_diagnostic": 8,
        }
        assert (
            manifest[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
            == config[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
        )
        assert manifest[R9_PHASE_CONTRACT_FIELD] == config[R9_PHASE_CONTRACT_FIELD]
        assert (
            manifest["resume_contract"][R9_PHASE_CONTRACT_FIELD]
            == config[R9_PHASE_CONTRACT_FIELD]
        )
        assert set(manifest["cosine"]) == {
            "candidate_e0_target",
            "native_e0_target",
            "candidate_edev_source",
            "native_edev_source",
        }
        assert all("edev_cosine" in row and "native_edev_cosine" in row for row in rows)
        assert (
            manifest["resume_contract"][R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
            == config[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
        )
        assert manifest["config"]["active_guidance_intervals"] == ["I1", "I3"]
        assert len(manifest["diagnostic_flow_map_traces"]) == 2

        for name in ("completion.json", "generation_result.json", "run_manifest.json"):
            (output / name).unlink()
        rows[0]["candidate_diagnostic_nfe"] = 7
        (output / "per_sample.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="diagnostic NFE disagrees"):
            run_guidance_records(
                config=config,
                records=records,
                runtime=_runtime(),
                output_dir=output,
                shard_index=0,
                num_shards=1,
            )
        rows[0]["candidate_diagnostic_nfe"] = 8
        rows[0]["candidate_diagnostic_trace"][0]["t"] = 0.9
        (output / "per_sample.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="diagnostic trace disagrees"):
            run_guidance_records(
                config=config,
                records=records,
                runtime=_runtime(),
                output_dir=output,
                shard_index=0,
                num_shards=1,
            )


def test_cli_has_no_r9_interval_semantic_override_and_effective_config_binds_contract() -> (
    None
):
    path = REPO_ROOT / "scripts" / "run_meanflow_flow_map_guidance.py"
    spec = importlib.util.spec_from_file_location(
        "run_meanflow_flow_map_guidance_r9", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="unsupported semantic"):
        module.resolve_guidance_semantics(
            _r9_config(), {"active_guidance_intervals": ["I2"]}
        )
    validated = validate_guidance_config(_r9_config())
    effective = module.finalize_effective_guidance_config(
        validated,
        locked_schedule=LOCKED_SCHEDULE,
    )
    assert (
        effective[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
        == validated[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
    )
    assert effective["active_guidance_intervals"] == ["I1", "I3"]
    assert len(effective["arm_config_sha256"]) == 64
