from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.evaluation.r12_latent_fourier_replay import (  # noqa: E402
    R12FourierReplayError,
    frozen_criteria,
    merge_and_bind_replay_rows,
    pair_preserving_lane,
)
from safa.guidance.meanflow_flow_map import (  # noqa: E402
    CountedFlowMap,
    freeze_guidance_stack,
    optimize_initial_noise,
    radial_shell_rfft_energy,
)


class _Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.1))

    def flow_map(self, x, z, *, t, r):
        del z
        return x - (float(t) - float(r)) * self.velocity


class _Codec:
    def __init__(self) -> None:
        self.vae = nn.Identity()

    def decode(self, latent):
        return latent


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, image):
        return {"embedding": image.flatten(1) * self.scale}


def _run(snapshot_steps=None):
    torch.manual_seed(7)
    generator = _Generator()
    codec = _Codec()
    encoder = _Encoder()
    freeze_guidance_stack(generator, codec, encoder)
    x_init = torch.randn(2, 3, 4, 4)
    target = torch.flip(x_init.flatten(1), dims=(1,))
    result = optimize_initial_noise(
        flow_map=CountedFlowMap(generator),
        codec=codec,
        e0=encoder,
        x_init=x_init,
        transport_condition=torch.zeros(2, 4),
        target_z0=target,
        num_updates=2,
        eta=0.5,
        projection="fixed_radius",
        spectral_snapshot_steps=snapshot_steps,
    )
    return result


def test_rfft_shell_energy_obeys_parseval_and_constant_is_dc() -> None:
    noise = torch.randn(2, 3, 8, 8)
    summary = radial_shell_rfft_energy(noise)
    assert summary["norm"] == "ortho"
    assert sum(summary["full_spectrum_coefficient_count"]) == 64
    assert torch.allclose(
        summary["per_channel_shell_energy"].sum(dim=-1),
        noise.double().square().sum(dim=(-2, -1)),
        rtol=1.0e-10,
        atol=1.0e-8,
    )
    constant = radial_shell_rfft_energy(torch.ones(1, 1, 8, 8))
    assert constant["radius_squared"][0] == 0
    assert constant["per_channel_shell_energy"][0, 0, 0] == pytest.approx(64.0)
    assert constant["per_channel_shell_energy"][0, 0, 1:].abs().max() < 1.0e-20


def test_opt_in_snapshots_do_not_change_default_trajectory() -> None:
    baseline = _run()
    replay = _run((0, 1, 2))
    assert "spectral_snapshots" not in baseline.diagnostics
    assert [row["step"] for row in replay.diagnostics["spectral_snapshots"]] == [0, 1, 2]
    assert torch.equal(baseline.latent, replay.latent)
    assert baseline.diagnostics["loss_history"] == replay.diagnostics["loss_history"]


@pytest.mark.parametrize("steps", [(1, 0), (0, 0), (-1,), (3,), (False,)])
def test_snapshot_steps_fail_closed(steps) -> None:
    with pytest.raises(ValueError, match="spectral_snapshot_steps"):
        _run(steps)


def test_pair_preserving_lane_keeps_original_batch_pairs() -> None:
    rows = [{"sample_id": str(index)} for index in range(8)]
    lane0 = pair_preserving_lane(rows, batch_size=2, lane_index=0, lane_count=2)
    lane1 = pair_preserving_lane(rows, batch_size=2, lane_index=1, lane_count=2)
    assert [row["original_ordinal"] for row in lane0] == [0, 1, 4, 5]
    assert [row["original_ordinal"] for row in lane1] == [2, 3, 6, 7]


def test_merge_requires_exact_order_and_loss_history(tmp_path: Path) -> None:
    expected = [f"s{index}" for index in range(4)]
    lane_paths = []
    for lane, ordinals in enumerate(((0, 1), (2, 3))):
        path = tmp_path / f"lane{lane}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": expected[ordinal],
                        "original_ordinal": ordinal,
                        "loss_history": [1.0, 0.5],
                        "candidate_nfe": 2,
                        "initial_norm": 4.0,
                    }
                )
                + "\n"
                for ordinal in ordinals
            ),
            encoding="utf-8",
        )
        lane_paths.append(path)
    existing = [
        {
            "sample_id": sample_id,
            "candidate_nfe": 2,
            "route_diagnostics": {"loss_history": [1.0, 0.5], "initial_norm": 4.0},
        }
        for sample_id in expected
    ]
    merged = merge_and_bind_replay_rows(
        lane_paths, expected_sample_ids=expected, existing_u16_rows=existing
    )
    assert [row["sample_id"] for row in merged] == expected
    existing[0]["route_diagnostics"]["loss_history"] = [1.0, 0.4]
    with pytest.raises(R12FourierReplayError, match="loss-history binding failed"):
        merge_and_bind_replay_rows(
            lane_paths, expected_sample_ids=expected, existing_u16_rows=existing
        )


def test_frozen_criteria_license_only_joint_signature() -> None:
    sample_ids = [f"s{index:02d}" for index in range(32)]
    regular_privacy = [float((index * 7) % 32) for index in range(32)]
    regular = [
        {
            "sample_id": sample_id,
            "h12": float(index),
            "h16": float(index),
            "arcface_delta": regular_privacy[index],
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    tail = [
        {
            "sample_id": sample_id,
            "h12": float(index),
            "h16": float(index),
            "sharpness_retention": float(index),
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    result = frozen_criteria(regular, tail, iterations=300)
    assert result["fourier_projection_licensed"] is True
    for index, row in enumerate(tail):
        row["h12"] = float(31 - index)
    result = frozen_criteria(regular, tail, iterations=300)
    assert result["fourier_projection_licensed"] is False
