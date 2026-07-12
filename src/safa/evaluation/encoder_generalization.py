from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from safa.evaluation.metrics import compute_validation_relation_metrics


@dataclass(frozen=True)
class EncoderBatch:
    encoder_name: str
    embeddings: torch.Tensor
    logits: torch.Tensor


def within_encoder_generalization(
    source: EncoderBatch,
    generated: EncoderBatch,
    labels: torch.Tensor,
) -> dict[str, Any]:
    if source.encoder_name != generated.encoder_name:
        raise ValueError(
            "cross-coordinate cosine is forbidden: source and generated encoder names differ"
        )
    if not source.encoder_name:
        raise ValueError("encoder_name must be non-empty")
    _validate_batch(source, "source")
    _validate_batch(generated, "generated")
    if source.embeddings.shape != generated.embeddings.shape:
        raise ValueError("source and generated embeddings must have the same shape")
    if source.logits.shape != generated.logits.shape:
        raise ValueError("source and generated logits must have the same shape")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional torch.Tensor")
    if labels.shape[0] != source.embeddings.shape[0]:
        raise ValueError("labels sample count must match encoder batches")
    labels = labels.detach().to(dtype=torch.long, device="cpu")
    if bool(((labels < 0) | (labels > 7)).any()):
        raise ValueError("8-class labels must be integers in 0..7")

    source_embeddings = F.normalize(source.embeddings.detach().float(), dim=1)
    generated_embeddings = F.normalize(generated.embeddings.detach().float(), dim=1)
    paired = F.cosine_similarity(source_embeddings, generated_embeddings, dim=1)
    relation = compute_validation_relation_metrics(generated_embeddings, source_embeddings)
    source_predictions = source.logits.detach().argmax(dim=1).cpu()
    generated_predictions = generated.logits.detach().argmax(dim=1).cpu()
    return {
        "encoder": source.encoder_name,
        "sample_count": int(labels.shape[0]),
        "paired_source_generated_cosine": _summary(paired),
        "pairwise_cosine_distance_spearman": float(relation["pairwise_spearman"]),
        "source_accuracy_8class": float((source_predictions == labels).float().mean()),
        "generated_accuracy_8class": float((generated_predictions == labels).float().mean()),
    }


def multi_encoder_median(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("multi-encoder median requires at least one encoder result")
    cosine = []
    spearman = []
    accuracy = []
    for name, result in results.items():
        paired = result.get("paired_source_generated_cosine")
        if not isinstance(paired, Mapping):
            raise ValueError(f"encoder {name!r} is missing paired cosine summary")
        cosine.append(_finite(paired.get("mean"), f"{name} paired cosine mean"))
        spearman.append(
            _finite(
                result.get("pairwise_cosine_distance_spearman"),
                f"{name} pairwise-distance Spearman",
            )
        )
        accuracy.append(
            _finite(result.get("generated_accuracy_8class"), f"{name} generated accuracy")
        )
    return {
        "encoder_count": len(results),
        "paired_cosine_mean_median": float(statistics.median(cosine)),
        "pairwise_distance_spearman_median": float(statistics.median(spearman)),
        "generated_accuracy_8class_median": float(statistics.median(accuracy)),
    }


def _validate_batch(batch: EncoderBatch, label: str) -> None:
    if not isinstance(batch.embeddings, torch.Tensor) or batch.embeddings.ndim != 2:
        raise ValueError(f"{label} embeddings must be a two-dimensional torch.Tensor")
    if not isinstance(batch.logits, torch.Tensor) or batch.logits.ndim != 2:
        raise ValueError(f"{label} logits must be a two-dimensional torch.Tensor")
    if batch.embeddings.shape[0] != batch.logits.shape[0]:
        raise ValueError(f"{label} embedding/logit sample counts differ")
    if batch.embeddings.shape[0] <= 1:
        raise ValueError(f"{label} batch requires at least two samples")
    if batch.logits.shape[1] != 8:
        raise ValueError(f"{label} logits must have exactly 8 classes")
    if not torch.isfinite(batch.embeddings).all() or not torch.isfinite(batch.logits).all():
        raise ValueError(f"{label} encoder outputs contain non-finite values")


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(dtype=torch.float64, device="cpu")
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("cosine summary requires finite values")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "median": float(torch.quantile(values, 0.5)),
        "p10": float(torch.quantile(values, 0.1)),
        "p90": float(torch.quantile(values, 0.9)),
    }


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
