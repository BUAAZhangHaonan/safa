from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

import safa.evaluation.meanflow_guidance_runner as runner  # noqa: E402
from safa.evaluation.meanflow_guidance_runner import (  # noqa: E402
    EXPECTED_MODEL_CONFIG,
    asset_contract_from_config,
    build_frozen_runtime,
    load_ema_generator,
    validate_checkpoint_contract,
    validate_guidance_config,
)
from safa.evaluation.r12_latent_fourier_replay import (  # noqa: E402
    validate_replay_config,
)
from safa.evaluation.r13_evaluator_contract import (  # noqa: E402
    R13_ARM_CHECKPOINT_ROOTS,
    R13_EVALUATOR_CONTRACT_FIELD,
    R13_EVALUATOR_CONTRACT_TYPE,
    R13_FINAL_GLOBAL_STEP,
    R13_LOCKED_ASSETS,
    R13_SAMPLE_MANIFESTS,
    apply_r13_strict_cuda_determinism,
    validate_r13_checkpoint_declaration,
    validate_r13_evaluator_contract,
)
from safa.evaluation.r8_arm_contracts import (  # noqa: E402
    canonical_arm_config_digest,
)
from safa.evaluation.r9_determinism import (  # noqa: E402
    R9_DETERMINISM_POLICY,
    R9_DETERMINISM_POLICY_SHA256,
    canonical_guidance_arm_config_digest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _declaration(arm_id: str = "control", sample_set: str = "regular32") -> dict:
    root = R13_ARM_CHECKPOINT_ROOTS[arm_id].as_posix()
    manifest = R13_SAMPLE_MANIFESTS[sample_set]
    return {
        "schema_version": 1,
        "contract_type": R13_EVALUATOR_CONTRACT_TYPE,
        "arm_id": arm_id,
        "sample_set": sample_set,
        "checkpoint_path": f"{root}/last.pt",
        "checkpoint_sha256": ("a" if arm_id == "control" else "b") * 64,
        "checkpoint_model": "ema",
        "stage": "stage2",
        "stage_epoch_1based": 1,
        "global_step": R13_FINAL_GLOBAL_STEP,
        "phase": "diagnose",
        "mode": "initial_noise",
        "transport_condition": "learned_null_condition",
        "seed": 7919,
        "sampling_seed": 7919,
        "projection": "fixed_radius",
        "eta": 0.5,
        "num_updates": 16,
        "batch_size": 2,
        "max_samples": 32,
        "pixel_image_size": 256,
        "attention_backend": "native",
        "determinism_policy_sha256": R9_DETERMINISM_POLICY_SHA256,
        "sample_id_manifest": manifest["path"],
        "sample_id_manifest_sha256": manifest["sha256"],
    }


def _config(arm_id: str = "control", sample_set: str = "regular32") -> dict:
    declaration = _declaration(arm_id, sample_set)
    return {
        "experiment_contract": R13_EVALUATOR_CONTRACT_TYPE,
        R13_EVALUATOR_CONTRACT_FIELD: declaration,
        "experiment_name": f"r13_{arm_id}_{sample_set}",
        "device": "cuda:0",
        "checkpoint": declaration["checkpoint_path"],
        "checkpoint_sha256": declaration["checkpoint_sha256"],
        "checkpoint_model": "ema",
        "expected_stage": "stage2",
        "expected_stage_epoch_1based": 1,
        "expected_global_step": R13_FINAL_GLOBAL_STEP,
        "expected_model_type": "meanflow_sit",
        "expected_sit_patch_size": 4,
        "phase": "diagnose",
        "mode": "initial_noise",
        "transport_condition": "learned_null_condition",
        "seed": 7919,
        "sampling_seed": 7919,
        "projection": "fixed_radius",
        "eta": 0.5,
        "num_updates": 16,
        "batch_size": 2,
        "max_samples": 32,
        "pixel_image_size": 256,
        "attention_backend": "native",
        "determinism_policy": dict(R9_DETERMINISM_POLICY),
        "determinism_policy_sha256": R9_DETERMINISM_POLICY_SHA256,
        "sample_id_manifest": declaration["sample_id_manifest"],
        "sample_id_manifest_sha256": declaration[
            "sample_id_manifest_sha256"
        ],
        **R13_LOCKED_ASSETS,
    }


def _checkpoint_payload(
    *,
    epoch: object = 1,
    global_step: object = R13_FINAL_GLOBAL_STEP,
    required_optimizer_steps: object = R13_FINAL_GLOBAL_STEP,
) -> dict:
    return {
        "stage": "stage2",
        "metrics": {
            "stage_epoch_1based": epoch,
            "global_step": global_step,
            "required_optimizer_steps": required_optimizer_steps,
        },
        "model_config": dict(EXPECTED_MODEL_CONFIG),
        "ema_model_state_dict": {"weight": torch.ones(())},
    }


@pytest.mark.parametrize("arm_id", ("control", "lpl"))
@pytest.mark.parametrize("sample_set", ("regular32", "tail32"))
def test_registered_r13_contract_accepts_only_the_four_declared_pairings(
    arm_id: str, sample_set: str
) -> None:
    config = _config(arm_id, sample_set)

    assert validate_r13_evaluator_contract(config) == config[
        R13_EVALUATOR_CONTRACT_FIELD
    ]
    assert validate_guidance_config(config)["mode"] == "initial_noise"


def test_historical_checkpoint_default_remains_epoch_1652_without_step_fields() -> None:
    payload = _checkpoint_payload(epoch=1652.0)
    payload["metrics"].pop("global_step")
    payload["metrics"].pop("required_optimizer_steps")

    metadata = validate_checkpoint_contract(payload)

    assert metadata["stage_epoch_1based"] == 1652
    assert "global_step" not in metadata


def test_new_checkpoint_path_is_rejected_without_exact_opt_in() -> None:
    config = _config()
    config.pop(R13_EVALUATOR_CONTRACT_FIELD)
    config["experiment_contract"] = "safa_r9_meanflow_v1"

    with pytest.raises(ValueError, match="checkpoint must be exactly"):
        validate_guidance_config(config)


@pytest.mark.parametrize("remove", (True, False))
def test_r13_schema_rejects_missing_and_extra_fields(remove: bool) -> None:
    config = _config()
    if remove:
        config[R13_EVALUATOR_CONTRACT_FIELD].pop("sampling_seed")
    else:
        config[R13_EVALUATOR_CONTRACT_FIELD]["unexpected"] = "forbidden"

    with pytest.raises(ValueError, match="fields must match"):
        validate_r13_evaluator_contract(config)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("contract_type", "unknown"),
        ("arm_id", "other"),
        ("sample_set", "other32"),
        ("checkpoint_path", "/absolute/last.pt"),
        ("checkpoint_path", "artifacts/checkpoints/../escape.pt"),
        ("checkpoint_path", "artifacts\\checkpoints\\last.pt"),
        ("checkpoint_path", "artifacts/checkpoints/r13_lpl_conditioning_1epoch_seed1337/last.pt"),
        ("checkpoint_path", "artifacts/checkpoints/r13_control_conditioning_1epoch_seed1337/last.bin"),
        ("checkpoint_sha256", "A" * 64),
        ("checkpoint_model", "raw"),
        ("stage", "stage1"),
        ("stage_epoch_1based", 1.0),
        ("global_step", 7499),
        ("phase", "calibrate"),
        ("mode", "native"),
        ("transport_condition", "source_condition"),
        ("seed", True),
        ("sampling_seed", 1337),
        ("projection", "typical_shell"),
        ("eta", float("nan")),
        ("eta", float("inf")),
        ("eta", 0.25),
        ("num_updates", 15),
        ("batch_size", 4),
        ("max_samples", 31),
        ("pixel_image_size", 128),
        ("attention_backend", "auto"),
        ("determinism_policy_sha256", "0" * 64),
        ("sample_id_manifest", R13_SAMPLE_MANIFESTS["tail32"]["path"]),
        ("sample_id_manifest_sha256", "0" * 64),
    ),
)
def test_r13_contract_rejects_nested_drift(field: str, value: object) -> None:
    config = _config()
    config[R13_EVALUATOR_CONTRACT_FIELD][field] = value

    with pytest.raises(ValueError):
        validate_r13_evaluator_contract(config)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint", "artifacts/checkpoints/r13_control_conditioning_1epoch_seed1337/other.pt"),
        ("checkpoint_sha256", "f" * 64),
        ("expected_stage_epoch_1based", 1.0),
        ("expected_global_step", float("nan")),
        ("expected_model_type", "other"),
        ("expected_sit_patch_size", 2),
        ("phase", "full"),
        ("mode", "native"),
        ("seed", 1337),
        ("sampling_seed", 1337),
        ("eta", float("nan")),
        ("num_updates", 12),
        ("batch_size", 4),
        ("max_samples", 31),
        ("attention_backend", "sdpa"),
        ("sample_id_manifest_sha256", "0" * 64),
        ("e0_sha256", "0" * 64),
    ),
)
def test_r13_contract_rejects_top_level_binding_drift(
    field: str, value: object
) -> None:
    config = _config()
    config[field] = value

    with pytest.raises(ValueError):
        validate_r13_evaluator_contract(config)


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("stage_epoch_1based", 1652),
        ("stage_epoch_1based", 1.0),
        ("global_step", None),
        ("global_step", 7499),
        ("global_step", 7500.0),
        ("global_step", float("nan")),
        ("required_optimizer_steps", None),
        ("required_optimizer_steps", 7499),
        ("required_optimizer_steps", 7500.0),
        ("required_optimizer_steps", float("inf")),
    ),
)
def test_r13_checkpoint_rejects_missing_nonfinite_or_drifted_step_evidence(
    metric: str, value: object
) -> None:
    payload = _checkpoint_payload()
    payload["metrics"][metric] = value

    with pytest.raises(ValueError):
        validate_checkpoint_contract(
            payload,
            expected_stage_epoch_1based=1,
            expected_global_step=R13_FINAL_GLOBAL_STEP,
        )


def test_r13_checkpoint_accepts_exact_internal_step_evidence() -> None:
    metadata = validate_checkpoint_contract(
        _checkpoint_payload(),
        expected_stage_epoch_1based=1,
        expected_global_step=R13_FINAL_GLOBAL_STEP,
    )

    assert metadata["global_step"] == R13_FINAL_GLOBAL_STEP
    assert metadata["required_optimizer_steps"] == R13_FINAL_GLOBAL_STEP
    assert metadata["weight_source"] == "ema_model_state_dict"


def test_r13_checkpoint_declaration_binds_the_loader_path() -> None:
    declaration = _declaration()
    assert validate_r13_checkpoint_declaration(
        declaration, declaration["checkpoint_path"]
    ) == declaration

    with pytest.raises(ValueError, match="loader path must match"):
        validate_r13_checkpoint_declaration(declaration, "other.pt")


def test_r13_ema_loader_uses_exact_declaration_and_internal_step_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _declaration()
    payload = _checkpoint_payload()
    captured: dict[str, object] = {}

    class FakeGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
            self.requested_attention_backend = "native"
            self.attention_backend = "native"

    def builder(config):
        captured.update(config)
        return FakeGenerator()

    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    _, metadata = load_ema_generator(
        declaration["checkpoint_path"],
        device=torch.device("cpu"),
        r9_attention_backend="native",
        checkpoint_contract=declaration,
        generator_builder=builder,
    )

    assert captured["attention_backend"] == "native"
    assert metadata["weight_source"] == "ema_model_state_dict"
    assert metadata["global_step"] == R13_FINAL_GLOBAL_STEP
    assert metadata["required_optimizer_steps"] == R13_FINAL_GLOBAL_STEP


def _mock_asset_digests(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    expected = {
        str(config["checkpoint"]): str(config["checkpoint_sha256"]),
        str(config["e0_checkpoint"]): str(config["e0_sha256"]),
        str(config["edev_checkpoint"]): str(config["edev_sha256"]),
        str(config["vae_path"]): str(config["vae_digest"]),
        str(config["index"]): str(config["index_sha256"]),
        str(config["features"]): str(config["features_digest"]),
        str(config["sample_id_manifest"]): str(
            config["sample_id_manifest_sha256"]
        ),
        str(config["heldout_e1_checkpoint"]): str(config["heldout_e1_sha256"]),
        str(config["heldout_e2_checkpoint"]): str(config["heldout_e2_sha256"]),
    }
    monkeypatch.setattr(runner, "_digest_path", lambda path: expected[str(path)])
    monkeypatch.setattr(
        runner,
        "read_ordered_sample_manifest",
        lambda path: [{"sample_id": f"sample-{index:02d}"} for index in range(32)],
    )


def test_r13_asset_contract_recomputes_sha_and_requires_exact_32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    _mock_asset_digests(monkeypatch, config)

    assets = asset_contract_from_config(config)
    assert assets["checkpoint"]["sha256"] == config["checkpoint_sha256"]
    assert assets["sample_manifest"]["sample_count"] == 32

    monkeypatch.setattr(
        runner,
        "read_ordered_sample_manifest",
        lambda path: [{"sample_id": f"sample-{index:02d}"} for index in range(31)],
    )
    with pytest.raises(ValueError, match="exactly 32"):
        asset_contract_from_config(config)


def test_r13_asset_contract_rejects_actual_checkpoint_sha_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    _mock_asset_digests(monkeypatch, config)
    original_digest = runner._digest_path
    monkeypatch.setattr(
        runner,
        "_digest_path",
        lambda path: "f" * 64
        if str(path) == config["checkpoint"]
        else original_digest(path),
    )

    with pytest.raises(ValueError, match="checkpoint asset digest mismatch"):
        asset_contract_from_config(config)


def test_build_runtime_for_r13_forces_native_ema_contract_and_edev() -> None:
    config = _config()
    captured: dict[str, object] = {}
    encoder_paths: list[str] = []

    def generator_loader(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return nn.Linear(1, 1), {"weight_source": "ema_model_state_dict"}

    def encoder_loader(path, *, device):
        encoder_paths.append(str(path))
        assert device == "cpu"
        return nn.Linear(1, 1), {}

    def codec_builder(config, device):
        del config, device
        return SimpleNamespace(vae=nn.Linear(1, 1))

    assets = {
        "e0": {"path": config["e0_checkpoint"], "sha256": config["e0_sha256"]},
        "edev": {
            "path": config["edev_checkpoint"],
            "sha256": config["edev_sha256"],
        },
        "vae": {
            "path": config["vae_path"],
            "digest": config["vae_digest"],
            "scaling_factor": config["vae_scaling_factor"],
        },
        "real_index": {"path": config["index"], "sha256": config["index_sha256"]},
        "target_features": {
            "path": config["features"],
            "digest": config["features_digest"],
            "feature_source": config["feature_source"],
        },
        "sample_manifest": {
            "path": config["sample_id_manifest"],
            "sha256": config["sample_id_manifest_sha256"],
            "sample_count": 32,
            "ordered_sample_id_sha256": "c" * 64,
        },
        "heldout_e1": {
            "path": config["heldout_e1_checkpoint"],
            "sha256": config["heldout_e1_sha256"],
        },
        "heldout_e2": {
            "path": config["heldout_e2_checkpoint"],
            "sha256": config["heldout_e2_sha256"],
        },
    }

    runtime = build_frozen_runtime(
        config,
        device=torch.device("cpu"),
        checkpoint_sha256=str(config["checkpoint_sha256"]),
        asset_contract=assets,
        generator_loader=generator_loader,
        encoder_loader=encoder_loader,
        codec_builder=codec_builder,
    )

    assert captured["path"] == config["checkpoint"]
    assert captured["r9_attention_backend"] == "native"
    assert captured["checkpoint_contract"] == config[R13_EVALUATOR_CONTRACT_FIELD]
    assert encoder_paths == [config["e0_checkpoint"], config["edev_checkpoint"]]
    assert runtime.edev is not None


class _FakeTorch:
    def __init__(self, *, cuda_initialized: bool = False) -> None:
        self.cuda = SimpleNamespace(is_initialized=lambda: cuda_initialized)
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(
                deterministic=False,
                benchmark=True,
                allow_tf32=True,
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )
        self._deterministic = False
        self._warn_only = True

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self._deterministic = enabled
        self._warn_only = warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self._warn_only


def test_r13_determinism_is_applied_before_cuda_and_rejects_late_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake_torch = _FakeTorch()

    contract = apply_r13_strict_cuda_determinism(
        _config(), torch_module=fake_torch
    )

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert fake_torch.are_deterministic_algorithms_enabled()
    assert not fake_torch.is_deterministic_algorithms_warn_only_enabled()
    assert contract["attention_backend"] == "native"

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        apply_r13_strict_cuda_determinism(
            _config(), torch_module=_FakeTorch(cuda_initialized=True)
        )


def test_r13_contract_changes_arm_digest_without_changing_historical_golden() -> None:
    r12_manifest = (
        REPO_ROOT
        / "artifacts"
        / "r12_seed_aligned_trajectory"
        / "runs_v1"
        / "u16_regular32"
        / "run_manifest.json"
    )
    r12_config = json.loads(r12_manifest.read_text(encoding="utf-8"))["config"]
    assert canonical_guidance_arm_config_digest(r12_config) == (
        "f6bf6460217cffc50805a1cdd34877b6de26ba581a7b551d74a55340dae73d35"
    )

    config = _config()
    original = canonical_arm_config_digest(config)
    mutated = deepcopy(config)
    mutated[R13_EVALUATOR_CONTRACT_FIELD]["checkpoint_sha256"] = "f" * 64
    assert canonical_arm_config_digest(mutated) != original


def test_r12_fourier_replay_explicitly_rejects_r13_opt_in_field() -> None:
    with pytest.raises(ValueError, match="does not accept the R13"):
        validate_replay_config(
            {
                "experiment_contract": "safa_r9_meanflow_v1",
                R13_EVALUATOR_CONTRACT_FIELD: {},
            },
            "regular32",
        )
