from __future__ import annotations

import torch
import torch.nn.functional as F

from safa.models.meanflow_sit import build_meanflow_sit_generator


def build_rectified_flow_sit_generator(config):
    base = build_meanflow_sit_generator(config)

    def forward(z):
        return sample(z, steps=config.sample_steps)

    def flow_matching_loss(x_1, z, generator=None):
        base._validate_z(z)
        if x_1.ndim != 4 or tuple(x_1.shape[1:]) != (config.sit_input_channels, base.image_size, base.image_size):
            raise ValueError(
                f"x_1 must have shape [B,{config.sit_input_channels},{base.image_size},{base.image_size}], got {tuple(x_1.shape)}"
            )
        x_data = base._data_to_model_space(x_1)
        eps = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype, generator=generator)
        t = torch.rand(x_data.shape[0], device=x_data.device, dtype=x_data.dtype, generator=generator)
        view_t = t.view(-1, 1, 1, 1)
        x_t = (1.0 - view_t) * x_data + view_t * eps
        target_velocity = eps - x_data
        predicted_velocity = base.vector_field(x_t, t, t, z)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        if base.null_condition is not None:
            loss = loss + 0.0 * base.null_condition.embedding.sum()
        return loss, {
            "flow_matching_mse": loss.detach(),
            "rectified_flow_mse": loss.detach(),
            "rectified_flow_backbone": "sit",
            "rectified_flow_sampler": config.sampler,
            "rectified_flow_t_mean": t.detach().mean(),
            "rectified_flow_attention_backend": base.attention_backend,
            "rectified_flow_attention_backend_requested": base.requested_attention_backend,
            "target_velocity_abs_mean": target_velocity.detach().abs().mean(),
            "predicted_velocity_abs_mean": predicted_velocity.detach().abs().mean(),
        }

    def sample(z, steps: int | None = None, checkpoint_steps: bool = False, *, x_init=None, clamp_output: bool = True):
        del checkpoint_steps
        base._validate_z(z)
        step_count = int(steps or config.sample_steps)
        if step_count <= 0:
            raise ValueError(f"sample steps must be positive, got {step_count}")
        if x_init is None:
            x = torch.randn(
                z.shape[0],
                config.sit_input_channels,
                base.image_size,
                base.image_size,
                device=z.device,
                dtype=z.dtype,
            )
        else:
            base._validate_x_init(x_init, z)
            x = x_init
        dt = 1.0 / float(step_count)
        for index in range(step_count):
            t_value = 1.0 - float(index) * dt
            t = torch.full((z.shape[0],), t_value, device=z.device, dtype=z.dtype)
            velocity = base.vector_field(x, t, t, z)
            if config.sampler == "heun" and index + 1 < step_count:
                proposal = x - dt * velocity
                next_t = torch.full((z.shape[0],), max(t_value - dt, 0.0), device=z.device, dtype=z.dtype)
                next_velocity = base.vector_field(proposal, next_t, next_t, z)
                velocity = 0.5 * (velocity + next_velocity)
            x = x - dt * velocity
        if base.null_condition is not None:
            x = x + 0.0 * base.null_condition.embedding.sum()
        return base._model_to_data_space(x, clamp_output=clamp_output)

    base.forward = forward
    base.flow_matching_loss = flow_matching_loss
    base.sample = sample
    return base
