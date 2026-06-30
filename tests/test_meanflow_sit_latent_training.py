from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_e11_config() -> dict:
    path = REPO_ROOT / "configs" / "medium_v2" / "experiments" / "e11_meanflow_sit_b_stage1_200ep.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _latent_generator_config():
    from safa.models.generator import FlowGeneratorConfig

    return FlowGeneratorConfig(
        model_type="meanflow_sit",
        embedding_dim=2,
        image_size=32,
        sample_steps=1,
        train_cycle_steps=1,
        sampler="meanflow",
        sit_input_channels=4,
        sit_patch_size=4,
    )


def _latent_sit_diffusion_generator_config():
    from safa.models.generator import FlowGeneratorConfig

    return FlowGeneratorConfig(
        model_type="sit_diffusion",
        embedding_dim=2,
        image_size=32,
        sample_steps=4,
        train_cycle_steps=4,
        sampler="ddim",
        sit_input_channels=4,
        sit_patch_size=4,
        sit_hidden_size=32,
        sit_depth=2,
        sit_num_heads=4,
        sit_mlp_ratio=2.0,
        sit_time_embedding_dim=32,
        sit_data_space="latent",
        diffusion_train_timesteps=32,
    )


class FakeLatentCodec:
    def __init__(self) -> None:
        self.encoded_images_shape = None
        self.decoded_latents_shape = None

    def encode(self, images):
        self.encoded_images_shape = tuple(images.shape)
        batch_size = int(images.shape[0])
        return torch.full((batch_size, 4, 32, 32), 0.25, device=images.device, dtype=images.dtype)

    def decode(self, latents):
        self.decoded_latents_shape = tuple(latents.shape)
        batch_size = int(latents.shape[0])
        return torch.full((batch_size, 3, 256, 256), 0.75, device=latents.device, dtype=latents.dtype)


class BFloat16OutOfRangeLatentCodec:
    def __init__(self) -> None:
        self.decoded_latents_shape = None

    def decode(self, latents):
        self.decoded_latents_shape = tuple(latents.shape)
        decoded = torch.full((latents.shape[0], 3, 256, 256), 1.5, device=latents.device, dtype=torch.bfloat16)
        decoded[:, :, 0, 0] = -0.5
        return decoded


def test_e11_latent_training_declares_vae_and_uses_pixel_image_size_for_transforms() -> None:
    from safa.training import g_loop

    config = _load_e11_config()

    assert config["latent_training"] is True
    assert config["pixel_image_size"] == 256
    assert config["image_size"] == 32
    assert config["vae_model"] == "stabilityai/sd-vae-ft-ema"
    assert config["vae_path"] == "artifacts/checkpoints/external/sd-vae-ft-ema"
    assert config["vae_scaling_factor"] == pytest.approx(0.18215)
    assert g_loop._generator_image_transform_size(config) == 256

    non_latent = copy.deepcopy(config)
    non_latent["latent_training"] = False
    non_latent.pop("pixel_image_size", None)
    assert g_loop._generator_image_transform_size(non_latent) == 32


def test_latent_training_config_requires_vae_source_before_training_starts() -> None:
    from safa.training import g_loop

    config = _load_e11_config()
    missing_vae = copy.deepcopy(config)
    missing_vae["latent_training"] = True
    missing_vae.pop("vae_model", None)
    missing_vae.pop("vae_path", None)

    with pytest.raises(ValueError, match="latent_training requires vae_model or vae_path"):
        g_loop._validate_train_g_config(missing_vae)


def test_latent_training_config_accepts_sit_diffusion_latent_generator() -> None:
    from safa.training.latent_codec import validate_latent_training_config

    validate_latent_training_config(_load_e11_config(), _latent_sit_diffusion_generator_config())


def test_latent_training_config_rejects_non_latent_capable_generator() -> None:
    from safa.models.generator import FlowGeneratorConfig
    from safa.training.latent_codec import validate_latent_training_config

    with pytest.raises(ValueError, match="latent_training requires generator.model_type"):
        validate_latent_training_config(_load_e11_config(), FlowGeneratorConfig(model_type="ddim", sampler="ddim"))


def test_sit_diffusion_latent_training_x_init_uses_four_channels() -> None:
    from safa.training import g_loop

    x_init = g_loop._make_x_init_for_generator_config(
        ["a", "b"],
        1337,
        _latent_sit_diffusion_generator_config(),
        torch.device("cpu"),
        torch.float32,
    )

    assert tuple(x_init.shape) == (2, 4, 32, 32)


def test_eval_runner_sit_diffusion_latent_x_init_uses_four_channels() -> None:
    from torch import nn

    from safa.evaluation import runner

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _latent_sit_diffusion_generator_config()
            self.seen_x_init_shape = None

        def sample(self, z, **kwargs):
            self.seen_x_init_shape = tuple(kwargs["x_init"].shape)
            return torch.zeros(z.shape[0], 4, 32, 32, device=z.device, dtype=z.dtype)

    generator = DummyGenerator()
    generated = runner._sample_generated_for_eval(
        generator,
        torch.eye(2),
        ["a", "b"],
        1337,
        256,
    )

    assert generator.seen_x_init_shape == (2, 4, 32, 32)
    assert tuple(generated.shape) == (2, 4, 32, 32)


def test_generator_training_step_encodes_pixels_to_latents_before_flow_loss() -> None:
    from torch import nn

    from safa.training.g_loop import _GeneratorTrainingStep

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.seen_image_shape = None

        def flow_matching_loss(self, images, z, generator=None):
            del z, generator
            self.seen_image_shape = tuple(images.shape)
            loss = images.mean() * self.weight
            return loss, {"flow_matching_mse": loss.detach()}

    class DummyE0(nn.Module):
        pass

    codec = FakeLatentCodec()
    generator = DummyGenerator()
    module = _GeneratorTrainingStep(generator, DummyE0(), _latent_generator_config(), 1337, latent_codec=codec)
    pixel_images = torch.zeros(2, 3, 256, 256)
    z = torch.eye(2)

    loss, flow_mse, cycle, flow_loss, cycle_loss = module(
        pixel_images,
        z,
        ["a", "b"],
        False,
        0.0,
        flow_condition="embedding",
    )

    assert codec.encoded_images_shape == (2, 3, 256, 256)
    assert generator.seen_image_shape == (2, 4, 32, 32)
    assert torch.isfinite(loss)
    assert torch.isfinite(flow_mse)
    assert torch.equal(cycle, torch.tensor(0.0))
    assert torch.allclose(flow_loss, loss)
    assert torch.equal(cycle_loss, torch.tensor(0.0))


def test_generator_training_step_keeps_non_latent_flow_loss_on_pixels() -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig
    from safa.training.g_loop import _GeneratorTrainingStep

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.seen_image_shape = None

        def flow_matching_loss(self, images, z, generator=None):
            del z, generator
            self.seen_image_shape = tuple(images.shape)
            loss = images.mean() * self.weight
            return loss, {"flow_matching_mse": loss.detach()}

    class DummyE0(nn.Module):
        pass

    generator = DummyGenerator()
    module = _GeneratorTrainingStep(generator, DummyE0(), FlowGeneratorConfig(embedding_dim=2, image_size=64), 1337)
    pixel_images = torch.zeros(2, 3, 64, 64)

    module(pixel_images, torch.eye(2), ["a", "b"], False, 0.0, flow_condition="embedding")

    assert generator.seen_image_shape == (2, 3, 64, 64)


def test_latent_decoded_eval_samples_are_float32_clamped_before_validation_detector() -> None:
    from torch import nn
    import torch.nn.functional as F

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def sample(self, z, **kwargs):
            del kwargs
            return torch.zeros(z.shape[0], 4, 32, 32, device=z.device, dtype=torch.bfloat16)

    class StrictE0(nn.Module):
        def forward(self, images):
            assert images.dtype == torch.float32
            batch_size = int(images.shape[0])
            base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=images.device, dtype=images.dtype)
            embedding = F.normalize(base[:batch_size], dim=1)
            logits = torch.zeros(batch_size, 2, device=images.device, dtype=images.dtype)
            return {"embedding": embedding, "logits": logits}

    class StrictDetector:
        def detect_counts(self, images):
            assert images.dtype == torch.float32
            assert float(images.min()) >= 0.0
            assert float(images.max()) <= 1.0
            return [1 for _ in range(int(images.shape[0]))]

    codec = BFloat16OutOfRangeLatentCodec()
    loader = [
        {
            "image": torch.zeros(3, 3, 256, 256),
            "z": F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=1),
            "sample_id": ["a", "b", "c"],
        }
    ]

    metrics = g_loop._evaluate_validation(
        DummyGenerator(),
        StrictE0(),
        loader,
        StrictDetector(),
        torch.device("cpu"),
        _latent_generator_config(),
        sampling_seed=1337,
        latent_codec=codec,
    )

    assert codec.decoded_latents_shape == (3, 4, 32, 32)
    assert metrics["face_detect_ge1_rate"] == pytest.approx(1.0)


def test_validation_decodes_latent_samples_before_e0_and_face_detector() -> None:
    from torch import nn
    import torch.nn.functional as F

    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_x_init_shape = None

        def sample(self, z, **kwargs):
            self.seen_x_init_shape = tuple(kwargs["x_init"].shape)
            return torch.zeros(z.shape[0], 4, 32, 32, device=z.device, dtype=z.dtype)

    class DummyE0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shapes = []

        def forward(self, images):
            self.seen_shapes.append(tuple(images.shape))
            batch_size = int(images.shape[0])
            base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=images.device, dtype=images.dtype)
            embedding = F.normalize(base[:batch_size], dim=1)
            logits = torch.zeros(batch_size, 2, device=images.device, dtype=images.dtype)
            return {"embedding": embedding, "logits": logits}

    class DummyDetector:
        def __init__(self) -> None:
            self.seen_shape = None

        def detect_counts(self, images):
            self.seen_shape = tuple(images.shape)
            return [1 for _ in range(int(images.shape[0]))]

    codec = FakeLatentCodec()
    generator = DummyGenerator()
    e0 = DummyE0()
    detector = DummyDetector()
    loader = [
        {
            "image": torch.zeros(3, 3, 256, 256),
            "z": F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=1),
            "sample_id": ["a", "b", "c"],
        }
    ]

    metrics = g_loop._evaluate_validation(
        generator,
        e0,
        loader,
        detector,
        torch.device("cpu"),
        _latent_generator_config(),
        sampling_seed=1337,
        latent_codec=codec,
    )

    assert generator.seen_x_init_shape == (3, 4, 32, 32)
    assert codec.decoded_latents_shape == (3, 4, 32, 32)
    assert e0.seen_shapes == [(3, 3, 256, 256), (3, 3, 256, 256)]
    assert detector.seen_shape == (3, 3, 256, 256)
    assert metrics["face_detect_ge1_rate"] == pytest.approx(1.0)


def test_quality_eval_decodes_latent_samples_before_saving(tmp_path, monkeypatch) -> None:
    from torch import nn

    from safa.evaluation import runner
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_x_init_shape = None

        def sample(self, z, **kwargs):
            self.seen_x_init_shape = tuple(kwargs["x_init"].shape)
            return torch.zeros(z.shape[0], 4, 32, 32, device=z.device, dtype=z.dtype)

    saved_shapes = []

    def fake_save(image, generated_dir, *, global_index, sample_id, row):
        del generated_dir, global_index, sample_id, row
        saved_shapes.append(tuple(image.shape))

    monkeypatch.setattr(runner, "_save_generated_image_for_eval", fake_save)
    codec = FakeLatentCodec()
    generator = DummyGenerator()

    count = g_loop._generate_quality_eval_images(
        generator=generator,
        loader=[{"z": torch.eye(2), "sample_id": ["a", "b"]}],
        generated_dir=tmp_path / "generated",
        device=torch.device("cpu"),
        generator_config=_latent_generator_config(),
        sampling_seed=1337,
        max_samples=2,
        use_amp=False,
        flow_condition="embedding",
        latent_codec=codec,
    )

    assert count == 2
    assert generator.seen_x_init_shape == (2, 4, 32, 32)
    assert codec.decoded_latents_shape == (2, 4, 32, 32)
    assert saved_shapes == [(3, 256, 256), (3, 256, 256)]


def test_latent_decoded_quality_samples_are_float32_clamped_before_saving(tmp_path, monkeypatch) -> None:
    from torch import nn

    from safa.evaluation import runner
    from safa.training import g_loop

    class DummyGenerator(nn.Module):
        def sample(self, z, **kwargs):
            del kwargs
            return torch.zeros(z.shape[0], 4, 32, 32, device=z.device, dtype=torch.bfloat16)

    saved = []

    def fake_save(image, generated_dir, *, global_index, sample_id, row):
        del generated_dir, global_index, sample_id, row
        assert image.dtype == torch.float32
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0
        saved.append(tuple(image.shape))

    monkeypatch.setattr(runner, "_save_generated_image_for_eval", fake_save)

    count = g_loop._generate_quality_eval_images(
        generator=DummyGenerator(),
        loader=[{"z": torch.eye(2), "sample_id": ["a", "b"]}],
        generated_dir=tmp_path / "generated",
        device=torch.device("cpu"),
        generator_config=_latent_generator_config(),
        sampling_seed=1337,
        max_samples=2,
        use_amp=False,
        flow_condition="embedding",
        latent_codec=BFloat16OutOfRangeLatentCodec(),
    )

    assert count == 2
    assert saved == [(3, 256, 256), (3, 256, 256)]


def test_meanflow_sit_latent_data_space_samples_raw_latents_without_pixel_clamp() -> None:
    from safa.models.generator import build_generator

    config = {
        "model_type": "meanflow_sit",
        "embedding_dim": 2,
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
        "meanflow_adaptive_weighting": False,
        "meanflow_norm_p": 1.0,
        "meanflow_norm_eps": 0.001,
        "meanflow_jvp_mode": "first_order",
        "sit_input_channels": 4,
        "sit_patch_size": 4,
        "sit_hidden_size": 32,
        "sit_depth": 2,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 32,
        "sit_data_space": "latent",
    }
    generator = build_generator(config)
    z = torch.zeros(1, 2)
    x_init = torch.full((1, 4, 16, 16), 2.0)

    sample = generator.sample(z, x_init=x_init, clamp_output=True)

    assert torch.equal(sample, x_init)


def test_latent_codec_encode_decode_with_cpu_fake_vae() -> None:
    from torch import nn

    from safa.training.latent_codec import LatentCodec, LatentCodecConfig

    class FakeLatentDist:
        def __init__(self, latents) -> None:
            self.latents = latents

        def sample(self):
            return self.latents

    class FakeEncodeOutput:
        def __init__(self, latents) -> None:
            self.latent_dist = FakeLatentDist(latents)

    class FakeDecodeOutput:
        def __init__(self, sample) -> None:
            self.sample = sample

    class FakeVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
            self.encode_input = None
            self.decode_input = None

        def encode(self, images):
            self.encode_input = images.detach().clone()
            latents = torch.full((images.shape[0], 4, 2, 2), 2.0, dtype=images.dtype, device=images.device)
            return FakeEncodeOutput(latents)

        def decode(self, latents):
            self.decode_input = latents.detach().clone()
            sample = torch.zeros(latents.shape[0], 3, 16, 16, dtype=latents.dtype, device=latents.device)
            return FakeDecodeOutput(sample)

    vae = FakeVAE()
    codec = LatentCodec(vae, LatentCodecConfig(source="fake", scaling_factor=0.5))

    encoded = codec.encode(torch.zeros(2, 3, 16, 16))
    decoded = codec.decode(encoded)

    assert torch.equal(vae.encode_input, torch.full((2, 3, 16, 16), -1.0))
    assert torch.equal(encoded, torch.ones(2, 4, 2, 2))
    assert torch.equal(vae.decode_input, torch.full((2, 4, 2, 2), 2.0))
    assert torch.equal(decoded, torch.full((2, 3, 16, 16), 0.5))
    assert all(parameter.requires_grad is False for parameter in vae.parameters())


def test_latent_codec_decode_preserves_latent_gradients_for_stage2_repr() -> None:
    from torch import nn

    from safa.training.latent_codec import LatentCodec, LatentCodecConfig

    class FakeDecodeOutput:
        def __init__(self, sample) -> None:
            self.sample = sample

    class FakeVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def decode(self, latents):
            sample = latents[:, :3].repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
            return FakeDecodeOutput(sample * self.weight)

    vae = FakeVAE()
    codec = LatentCodec(vae, LatentCodecConfig(source="fake", scaling_factor=1.0))
    latents = torch.zeros(1, 4, 2, 2, requires_grad=True)

    decoded = codec.decode(latents)
    loss = decoded.mean()
    loss.backward()

    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert float(latents.grad.abs().sum()) > 0.0
    assert all(parameter.requires_grad is False for parameter in vae.parameters())
