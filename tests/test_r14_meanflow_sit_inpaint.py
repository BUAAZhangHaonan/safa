from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _config(model_type: str = "meanflow_sit_inpaint") -> dict:
    return {
        "model_type": model_type,
        "embedding_dim": 8,
        "image_size": 8,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 8,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "meanflow",
        "learned_null_condition": True,
        "meanflow_ratio": 0.25,
        "meanflow_ratio_r_not_equal_t": 0.75,
        "meanflow_adaptive_weighting": True,
        "meanflow_norm_p": 0.75,
        "meanflow_norm_eps": 0.001,
        "meanflow_jvp_mode": "first_order",
        "sit_input_channels": 4,
        "sit_patch_size": 2,
        "sit_hidden_size": 16,
        "sit_depth": 1,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 16,
        "sit_data_space": "latent",
        "attention_backend": "native",
    }


def _inputs(batch_size: int = 2):
    z = torch.randn(batch_size, 8)
    state = torch.randn(batch_size, 4, 8, 8)
    context = torch.randn(batch_size, 4, 8, 8)
    mask = torch.zeros(batch_size, 1, 8, 8, dtype=torch.bool)
    mask[:, :, 2:6, 2:6] = True
    context = torch.where(mask.expand_as(context), torch.zeros_like(context), context)
    return z, state, context, mask


def test_inpaint_factory_is_versioned_and_context_embedder_is_zero_initialized() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_config())

    assert generator.config.model_type == "meanflow_sit_inpaint"
    assert generator.inpaint is True
    assert generator.inpaint_contract == "safa_meanflow_sit_inpaint_v1"
    assert torch.count_nonzero(generator.vector_field.context_embedder.weight) == 0
    assert torch.count_nonzero(generator.vector_field.context_embedder.bias) == 0

    legacy = build_generator(_config("meanflow_sit"))
    assert legacy.inpaint is False
    assert legacy.vector_field.context_embedder is None


def test_flow_map_projects_background_exactly_for_partial_and_empty_masks() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_config())
    z, state, context, mask = _inputs()

    output = generator.flow_map(
        state,
        z,
        t=1.0,
        r=0.0,
        context_latent=context,
        latent_mask=mask,
    )
    assert torch.equal(output.masked_select(~mask.expand_as(output)), context.masked_select(~mask.expand_as(context)))

    empty_mask = torch.zeros_like(mask)
    empty_context = torch.randn_like(context)
    empty_output = generator.flow_map(
        state,
        z,
        t=1.0,
        r=0.0,
        context_latent=empty_context,
        latent_mask=empty_mask,
    )
    assert torch.equal(empty_output, empty_context)


def test_full_mask_is_valid_and_masked_loss_has_finite_gradients() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_config())
    z = torch.randn(2, 8)
    target = torch.randn(2, 4, 8, 8)
    full_mask = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    context = torch.zeros_like(target)

    loss, metrics = generator.flow_matching_loss(
        target,
        z,
        context_latent=context,
        latent_mask=full_mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["flow_matching_mse"])
    assert metrics["meanflow_inpaint_contract"] == "safa_meanflow_sit_inpaint_v1"
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_training_rejects_empty_mask_and_context_inside_generation_region() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_config())
    z, target, context, mask = _inputs()

    with pytest.raises(ValueError, match="non-empty latent_mask"):
        generator.flow_matching_loss(
            target,
            z,
            context_latent=torch.randn_like(context),
            latent_mask=torch.zeros_like(mask),
        )

    invalid_context = context.clone()
    invalid_context[:, :, 3, 3] = 1.0
    with pytest.raises(ValueError, match="exactly zero inside latent_mask"):
        generator.flow_matching_loss(
            target,
            z,
            context_latent=invalid_context,
            latent_mask=mask,
        )


def test_pixel_assembly_preserves_original_background_bit_exactly() -> None:
    from safa.models.meanflow_sit import assemble_inpainted_pixels

    original = torch.randn(2, 3, 8, 8)
    generated = torch.randn_like(original)
    mask = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    mask[:, :, 1:5, 2:7] = True

    output = assemble_inpainted_pixels(original, generated, mask)

    assert torch.equal(output.masked_select(~mask.expand_as(output)), original.masked_select(~mask.expand_as(original)))
    assert torch.equal(output.masked_select(mask.expand_as(output)), generated.masked_select(mask.expand_as(generated)))


class _RecordingCodec:
    def __init__(self):
        self.inputs = []

    def encode(self, value):
        self.inputs.append(value.detach().clone())
        return torch.nn.functional.avg_pool2d(value[:, :1].repeat(1, 4, 1, 1), 2)


def test_context_encoder_api_never_accepts_source_pixels_and_zeros_face_before_latent_context() -> None:
    from safa.models.meanflow_sit import encode_inpaint_training_latents

    codec = _RecordingCodec()
    source_pixels = torch.full((2, 3, 8, 8), 99.0)
    target_pixels = torch.randn(2, 3, 8, 8)
    mask = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    mask[:, :, 2:6, 2:6] = True
    context_pixels = torch.where(mask.expand_as(target_pixels), torch.zeros_like(target_pixels), target_pixels)

    target_latent, context_latent, latent_mask = encode_inpaint_training_latents(
        codec,
        target_pixels,
        context_pixels,
        mask,
    )

    assert len(codec.inputs) == 2
    assert torch.equal(codec.inputs[0], target_pixels)
    assert torch.equal(codec.inputs[1], context_pixels)
    assert not torch.equal(codec.inputs[1], source_pixels)
    assert torch.count_nonzero(codec.inputs[1].masked_select(mask.expand_as(context_pixels))) == 0
    assert torch.count_nonzero(context_latent.masked_select(latent_mask.expand_as(context_latent))) == 0
    assert target_latent.shape == context_latent.shape

