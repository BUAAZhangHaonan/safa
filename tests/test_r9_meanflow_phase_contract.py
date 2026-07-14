from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.evaluation import meanflow_guidance_runner as runner  # noqa: E402
from safa.evaluation.meanflow_guidance_runner import (  # noqa: E402
    EXPECTED_VAE_SCALING_FACTOR,
    R9_EDEV_PHASES,
    R9_PHASE_CONTRACT_FIELD,
    build_frozen_runtime,
    validate_r9_phase_contract,
)
from safa.evaluation.r9_determinism import (  # noqa: E402
    R9_EXPERIMENT_CONTRACT,
)


SHA = "a" * 64


class _Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))


class _Codec:
    def __init__(self) -> None:
        self.vae = _Module()


def _r9_config(phase: str) -> dict[str, object]:
    config: dict[str, object] = {
        "experiment_contract": R9_EXPERIMENT_CONTRACT,
        "phase": phase,
        "attention_backend": "native",
        "checkpoint": "checkpoint.pt",
        "e0_checkpoint": "e0.pt",
        "edev_checkpoint": "edev.pt",
        "vae_scaling_factor": EXPECTED_VAE_SCALING_FACTOR,
    }
    config[R9_PHASE_CONTRACT_FIELD] = validate_r9_phase_contract(config)
    return config


def _asset_contract() -> dict[str, object]:
    return {
        "e0": {"path": "e0.pt", "sha256": SHA},
        "edev": {"path": "edev.pt", "sha256": SHA},
        "vae": {
            "path": "vae",
            "digest": SHA,
            "scaling_factor": EXPECTED_VAE_SCALING_FACTOR,
        },
        "real_index": {"path": "index.jsonl", "sha256": SHA},
        "target_features": {
            "path": "features",
            "digest": SHA,
            "feature_source": "cached_features",
        },
        "sample_manifest": {
            "path": "samples.jsonl",
            "sha256": SHA,
            "sample_count": 1,
            "ordered_sample_id_sha256": SHA,
        },
        "heldout_e1": {"path": "e1.pt", "sha256": SHA},
        "heldout_e2": {"path": "e2.pt", "sha256": SHA},
    }


@pytest.mark.parametrize(
    ("phase", "edev_required"),
    (
        ("semigroup", False),
        ("preflight", False),
        ("resource_smoke", True),
        ("diagnose", True),
        ("calibrate", True),
        ("confirm512", True),
        ("full", True),
    ),
)
def test_r9_phase_contract_is_explicit_and_binds_edev_outputs(
    phase: str, edev_required: bool
) -> None:
    contract = validate_r9_phase_contract(
        {"experiment_contract": R9_EXPERIMENT_CONTRACT, "phase": phase}
    )

    assert contract == {
        "schema_version": 1,
        "phase": phase,
        "edev_required": edev_required,
        "required_per_sample_edev_fields": (
            ["edev_cosine", "native_edev_cosine"] if edev_required else []
        ),
        "required_summary_edev_fields": (
            ["candidate_edev_source", "native_edev_source"] if edev_required else []
        ),
    }


@pytest.mark.parametrize("phase", R9_EDEV_PHASES)
def test_r9_abcd_phases_load_frozen_edev(phase: str) -> None:
    loaded: list[str] = []

    def generator_loader(path, **kwargs):
        return _Module(), {"stage": "stage2"}

    def encoder_loader(path, *, device):
        loaded.append(str(path))
        return _Module(), {}

    runtime = build_frozen_runtime(
        _r9_config(phase),
        device=torch.device("cpu"),
        checkpoint_sha256=SHA,
        asset_contract=_asset_contract(),
        generator_loader=generator_loader,
        encoder_loader=encoder_loader,
        codec_builder=lambda config, device: _Codec(),
    )

    assert loaded == ["e0.pt", "edev.pt"]
    assert runtime.edev is not None
    assert runtime.edev.training is False
    assert all(
        parameter.requires_grad is False for parameter in runtime.edev.parameters()
    )


@pytest.mark.parametrize("phase", ("semigroup", "preflight"))
def test_r9_preflight_phases_do_not_load_edev(phase: str) -> None:
    loaded: list[str] = []

    def encoder_loader(path, *, device):
        loaded.append(str(path))
        return _Module(), {}

    runtime = build_frozen_runtime(
        _r9_config(phase),
        device=torch.device("cpu"),
        checkpoint_sha256=SHA,
        asset_contract=_asset_contract(),
        generator_loader=lambda path, **kwargs: (_Module(), {"stage": "stage2"}),
        encoder_loader=encoder_loader,
        codec_builder=lambda config, device: _Codec(),
    )

    assert loaded == ["e0.pt"]
    assert runtime.edev is None


@pytest.mark.parametrize(
    ("phase", "expected_calls"),
    (("calibration", ["e0.pt", "edev.pt"]), ("full", ["e0.pt"])),
)
def test_r8_phase_semantics_are_unchanged(
    phase: str, expected_calls: list[str]
) -> None:
    loaded: list[str] = []

    def encoder_loader(path, *, device):
        loaded.append(str(path))
        return _Module(), {}

    config = dict(_r9_config("preflight"))
    config.pop("experiment_contract")
    config.pop(R9_PHASE_CONTRACT_FIELD)
    config["phase"] = phase
    runtime = build_frozen_runtime(
        config,
        device=torch.device("cpu"),
        checkpoint_sha256=SHA,
        asset_contract=_asset_contract(),
        generator_loader=lambda path, **kwargs: (_Module(), {"stage": "stage2"}),
        encoder_loader=encoder_loader,
        codec_builder=lambda config, device: _Codec(),
    )

    assert loaded == expected_calls
    assert (runtime.edev is not None) is (phase == "calibration")


@pytest.mark.parametrize("phase", (None, "calibration", "unknown"))
def test_r9_phase_contract_rejects_missing_or_unregistered_phase(phase) -> None:
    config = {"experiment_contract": R9_EXPERIMENT_CONTRACT}
    if phase is not None:
        config["phase"] = phase

    with pytest.raises(ValueError, match="explicit phase|phase must be one of"):
        validate_r9_phase_contract(config)


def test_r9_phase_contract_rejects_tamper_and_resume_rows_require_edev(
    tmp_path: Path,
) -> None:
    config = _r9_config("full")
    config[R9_PHASE_CONTRACT_FIELD]["edev_required"] = False
    with pytest.raises(ValueError, match="disagrees"):
        validate_r9_phase_contract(config)

    generated = tmp_path / "generated.png"
    native = tmp_path / "native.png"
    generated.write_bytes(b"png")
    native.write_bytes(b"png")
    binding = {"generated": str(generated), "native": str(native)}
    row = {
        **binding,
        "candidate_nfe": 1,
        "native_nfe": 1,
        "candidate_trace": [{"t": 1.0}],
        "native_trace": [{"t": 1.0}],
    }
    phase_contract = validate_r9_phase_contract(
        {"experiment_contract": R9_EXPERIMENT_CONTRACT, "phase": "full"}
    )
    with pytest.raises(ValueError, match="missing Edev fields"):
        runner._validate_resume_rows([row], [binding], r9_phase_contract=phase_contract)

    row.update({"edev_cosine": 0.4, "native_edev_cosine": 0.3})
    runner._validate_resume_rows([row], [binding], r9_phase_contract=phase_contract)
