from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _tiny_meanflow_sit_config(attention_backend: str = "native") -> dict:
    return {
        "model_type": "meanflow_sit",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "meanflow",
        "learned_null_condition": True,
        "meanflow_ratio": 0.25,
        "meanflow_ratio_r_not_equal_t": 0.75,
        "meanflow_adaptive_weighting": True,
        "meanflow_norm_p": 1.0,
        "meanflow_norm_eps": 0.001,
        "meanflow_jvp_mode": "torch_func",
        "sit_input_channels": 3,
        "sit_patch_size": 4,
        "sit_hidden_size": 32,
        "sit_depth": 2,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 32,
        "attention_backend": attention_backend,
    }


def _run_generator_smoke(attention_backend: str) -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_meanflow_sit_config(attention_backend))
    z = torch.zeros(2, 16)
    x_init = torch.randn(2, 3, 16, 16)
    sample = generator.sample(z, steps=1, x_init=x_init, clamp_output=False)
    loss, metrics = generator.flow_matching_loss(torch.rand(2, 3, 16, 16), z)
    loss.backward()

    assert tuple(sample.shape) == (2, 3, 16, 16)
    assert torch.isfinite(loss)
    assert generator.attention_backend == attention_backend
    assert metrics["meanflow_attention_backend"] == attention_backend
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_meanflow_sit_native_attention_forward_sample_and_loss() -> None:
    _run_generator_smoke("native")


def test_meanflow_sit_sdpa_attention_forward_sample_and_loss() -> None:
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("PyTorch SDPA is not available")
    _run_generator_smoke("sdpa")


def test_meanflow_sit_sdpa_uses_native_attention_inside_jvp(monkeypatch) -> None:
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("PyTorch SDPA is not available")
    from safa.models.generator import build_generator

    sdpa_calls = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def recording_sdpa(*args, **kwargs):
        sdpa_calls.append(int(torch.autograd.forward_ad._current_level))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)

    generator = build_generator(_tiny_meanflow_sit_config("sdpa"))
    z = torch.zeros(2, 16)
    loss, _ = generator.flow_matching_loss(torch.rand(2, 3, 16, 16), z)
    loss.backward()

    assert sdpa_calls
    assert all(level < 0 for level in sdpa_calls)


def test_meanflow_sit_auto_attention_backend_priority(monkeypatch) -> None:
    from safa.models import meanflow_sit

    availability = {"native": True, "sdpa": True, "fa2": True, "fa4": True}

    def fake_available(backend: str) -> bool:
        return availability[backend]

    monkeypatch.setattr(meanflow_sit, "_is_attention_backend_available", fake_available)

    assert meanflow_sit.resolve_meanflow_sit_attention_backend("auto") == "fa4"
    availability["fa4"] = False
    assert meanflow_sit.resolve_meanflow_sit_attention_backend("auto") == "fa2"
    availability["fa2"] = False
    assert meanflow_sit.resolve_meanflow_sit_attention_backend("auto") == "sdpa"
    availability["sdpa"] = False
    assert meanflow_sit.resolve_meanflow_sit_attention_backend("auto") == "native"


def test_meanflow_sit_explicit_unavailable_attention_backend_errors(monkeypatch) -> None:
    from safa.models import meanflow_sit

    monkeypatch.setattr(meanflow_sit, "_is_attention_backend_available", lambda backend: backend == "native")

    with pytest.raises(RuntimeError, match="attention backend 'fa4' is not available"):
        meanflow_sit.resolve_meanflow_sit_attention_backend("fa4")


def test_meanflow_sit_config_passes_attention_backend_to_generator() -> None:
    from safa.models.generator import FlowGeneratorConfig, build_generator

    config = FlowGeneratorConfig.from_dict(_tiny_meanflow_sit_config("sdpa"))
    assert config.attention_backend == "sdpa"
    assert config.to_dict()["attention_backend"] == "sdpa"

    generator = build_generator(config.to_dict())
    assert generator.requested_attention_backend == "sdpa"
    assert generator.attention_backend == "sdpa"
