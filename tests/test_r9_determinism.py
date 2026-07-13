from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.evaluation.meanflow_guidance_runner import (  # noqa: E402
    load_ema_generator,
    validate_guidance_config,
)
from safa.evaluation.r9_determinism import (  # noqa: E402
    R9_ATTENTION_BACKEND,
    R9_DETERMINISM_POLICY,
    R9_DETERMINISM_POLICY_SHA256,
    apply_r9_strict_cuda_determinism,
    canonical_r9_arm_config_digest,
    validate_r9_execution_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
R9_CONFIG = (
    REPO_ROOT
    / "configs"
    / "medium_v2"
    / "experiments"
    / "r9_meanflow_semigroup_preflight.yaml"
)


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


def _config() -> dict:
    payload = yaml.safe_load(R9_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_registered_r9_config_carries_exact_strict_policy_and_native_backend() -> None:
    config = _config()
    execution = validate_r9_execution_config(config)
    validated = validate_guidance_config(config)

    assert execution["determinism_policy"] == R9_DETERMINISM_POLICY
    assert execution["determinism_policy_sha256"] == R9_DETERMINISM_POLICY_SHA256
    assert execution["attention_backend"] == R9_ATTENTION_BACKEND == "native"
    assert validated["determinism_policy_sha256"] == R9_DETERMINISM_POLICY_SHA256


@pytest.mark.parametrize(
    "mutation",
    (
        lambda config: config["determinism_policy"].__setitem__(
            "deterministic_warn_only", True
        ),
        lambda config: config["determinism_policy"].__setitem__(
            "cuda_matmul_allow_tf32", True
        ),
        lambda config: config.__setitem__("attention_backend", "auto"),
        lambda config: config.__setitem__("determinism_policy_sha256", "0" * 64),
    ),
)
def test_r9_execution_contract_rejects_every_policy_or_backend_substitution(
    mutation,
) -> None:
    config = deepcopy(_config())
    mutation(config)

    with pytest.raises(ValueError):
        validate_r9_execution_config(config)


def test_apply_r9_determinism_sets_hard_error_policy_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake_torch = _FakeTorch()

    contract = apply_r9_strict_cuda_determinism(_config(), torch_module=fake_torch)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert fake_torch.are_deterministic_algorithms_enabled()
    assert not fake_torch.is_deterministic_algorithms_warn_only_enabled()
    assert fake_torch.backends.cudnn.deterministic
    assert not fake_torch.backends.cudnn.benchmark
    assert not fake_torch.backends.cuda.matmul.allow_tf32
    assert not fake_torch.backends.cudnn.allow_tf32
    assert contract["determinism_policy_sha256"] == R9_DETERMINISM_POLICY_SHA256


def test_apply_r9_determinism_fails_if_cuda_is_already_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake_torch = _FakeTorch(cuda_initialized=True)

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        apply_r9_strict_cuda_determinism(_config(), torch_module=fake_torch)

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert not fake_torch.are_deterministic_algorithms_enabled()


def test_r9_arm_digest_binds_preflight_contract_digest() -> None:
    config = _config()
    original = canonical_r9_arm_config_digest(config)

    assert canonical_r9_arm_config_digest(
        {**config, "semigroup_preflight_contract_sha256": "0" * 64}
    ) != original
    assert canonical_r9_arm_config_digest(
        {**config, "r9_semigroup_gate_contract_sha256": "1" * 64}
    ) != original


def test_load_ema_generator_overrides_checkpoint_auto_with_locked_native(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    captured: dict[str, object] = {}
    model_config = {
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
        "attention_backend": "auto",
    }
    payload = {
        "stage": "stage2",
        "metrics": {"stage_epoch_1based": 1652},
        "model_config": model_config,
        "ema_model_state_dict": {"weight": torch.ones(())},
    }

    class FakeGenerator(nn.Module):
        def __init__(self, attention_backend: str) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
            self.requested_attention_backend = attention_backend
            self.attention_backend = attention_backend

    def builder(config):
        captured.update(config)
        return FakeGenerator(str(config["attention_backend"]))

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    _, metadata = load_ema_generator(
        checkpoint,
        device=torch.device("cpu"),
        r9_attention_backend="native",
        generator_builder=builder,
    )

    assert captured["attention_backend"] == "native"
    assert metadata["model_config"]["attention_backend"] == "auto"
    assert metadata["attention_backend_requested"] == "native"
    assert metadata["attention_backend_resolved"] == "native"


def test_load_ema_generator_rejects_non_native_r9_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    payload = {
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
            "attention_backend": "auto",
        },
        "ema_model_state_dict": {"weight": torch.ones(())},
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    with pytest.raises(ValueError, match="locked to 'native'"):
        load_ema_generator(
            checkpoint,
            device=torch.device("cpu"),
            r9_attention_backend="sdpa",
            generator_builder=lambda config: pytest.fail(
                "invalid R9 backend must fail before model construction"
            ),
        )
