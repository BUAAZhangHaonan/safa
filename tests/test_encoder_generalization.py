from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from safa.evaluation.encoder_generalization import (
    EncoderBatch,
    multi_encoder_median,
    within_encoder_generalization,
)


def _batch(name: str, embeddings: torch.Tensor, predictions: list[int]) -> EncoderBatch:
    logits = torch.full((len(predictions), 8), -4.0)
    logits[torch.arange(len(predictions)), torch.tensor(predictions)] = 4.0
    return EncoderBatch(
        encoder_name=name,
        embeddings=F.normalize(embeddings.float(), dim=1),
        logits=logits,
    )


def test_within_encoder_metrics_compute_cosine_spearman_and_eight_class_accuracy() -> None:
    source = _batch(
        "e1_dinov2_large_v2",
        torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0]]),
        [0, 1, 2],
    )
    generated = _batch(
        "e1_dinov2_large_v2",
        torch.tensor([[2.0, 0.0, 0.0], [2.4, 1.8, 0.0], [0.0, 4.0, 0.0]]),
        [0, 5, 2],
    )

    metrics = within_encoder_generalization(source, generated, torch.tensor([0, 1, 2]))

    assert metrics["encoder"] == "e1_dinov2_large_v2"
    assert metrics["sample_count"] == 3
    assert metrics["paired_source_generated_cosine"]["mean"] == pytest.approx(1.0)
    assert metrics["pairwise_cosine_distance_spearman"] == pytest.approx(1.0)
    assert metrics["source_accuracy_8class"] == pytest.approx(1.0)
    assert metrics["generated_accuracy_8class"] == pytest.approx(2 / 3)


def test_within_encoder_metrics_reject_cross_coordinate_cosine() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    source = _batch("e1_dinov2_large_v2", embeddings, [0, 1, 2])
    generated = _batch("e2_convnext_tiny", embeddings, [0, 1, 2])

    with pytest.raises(ValueError, match="cross-coordinate"):
        within_encoder_generalization(source, generated, torch.tensor([0, 1, 2]))


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (torch.tensor([0, 1]), "sample count"),
        (torch.tensor([0, 1, 8]), "0..7"),
    ],
)
def test_within_encoder_metrics_require_exact_eight_class_labels(labels, message: str) -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    source = _batch("e1", embeddings, [0, 1, 2])
    generated = _batch("e1", embeddings, [0, 1, 2])

    with pytest.raises(ValueError, match=message):
        within_encoder_generalization(source, generated, labels)


def test_multi_encoder_median_uses_encoder_level_results() -> None:
    results = {
        "e1": {
            "paired_source_generated_cosine": {"mean": 0.6},
            "pairwise_cosine_distance_spearman": 0.4,
            "generated_accuracy_8class": 0.5,
        },
        "e2": {
            "paired_source_generated_cosine": {"mean": 0.8},
            "pairwise_cosine_distance_spearman": 0.6,
            "generated_accuracy_8class": 0.7,
        },
    }

    assert multi_encoder_median(results) == {
        "encoder_count": 2,
        "paired_cosine_mean_median": pytest.approx(0.7),
        "pairwise_distance_spearman_median": pytest.approx(0.5),
        "generated_accuracy_8class_median": pytest.approx(0.6),
    }
