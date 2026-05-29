from __future__ import annotations


_UNIT_NORM_ATOL = 1e-4
_UNIT_NORM_RTOL = 1e-4


def hyperspherical_gram_loss(
    pred_embedding,
    target_embedding,
    point_weight,
    relation_weight,
    offdiag_only: bool = True,
) -> dict[str, "torch.Tensor"]:
    import torch

    _validate_embedding_pair(pred_embedding, target_embedding)
    point_weight_tensor = _validate_scalar_weight("point_weight", point_weight, pred_embedding)
    relation_weight_tensor = _validate_scalar_weight("relation_weight", relation_weight, pred_embedding)
    if not isinstance(offdiag_only, bool):
        raise TypeError("offdiag_only must be bool")

    point_loss = (1.0 - (pred_embedding * target_embedding).sum(dim=1)).mean()
    pred_gram = pred_embedding @ pred_embedding.T
    target_gram = target_embedding @ target_embedding.T
    relation_error = (pred_gram - target_gram).pow(2)
    if offdiag_only:
        batch_size = pred_embedding.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=pred_embedding.device)
        relation_loss = relation_error[mask].mean()
    else:
        relation_loss = relation_error.mean()

    total_loss = point_weight_tensor * point_loss + relation_weight_tensor * relation_loss
    return {
        "point_loss": point_loss,
        "relation_loss": relation_loss,
        "total_loss": total_loss,
    }


def _validate_embedding_pair(pred_embedding, target_embedding) -> None:
    import torch

    if not isinstance(pred_embedding, torch.Tensor):
        raise TypeError("pred_embedding must be a torch.Tensor")
    if not isinstance(target_embedding, torch.Tensor):
        raise TypeError("target_embedding must be a torch.Tensor")
    if pred_embedding.ndim != 2 or target_embedding.ndim != 2:
        raise ValueError("pred_embedding and target_embedding must be 2D")
    if pred_embedding.shape != target_embedding.shape:
        raise ValueError("pred_embedding and target_embedding must have the same shape")
    if pred_embedding.shape[0] <= 1:
        raise ValueError("Batch dimension B > 1 is required")
    if not torch.isfinite(pred_embedding).all() or not torch.isfinite(target_embedding).all():
        raise FloatingPointError("pred_embedding and target_embedding must be finite")
    _validate_unit_norm("pred_embedding", pred_embedding)
    _validate_unit_norm("target_embedding", target_embedding)


def _validate_unit_norm(name: str, tensor) -> None:
    import torch

    norms = tensor.norm(dim=1)
    expected = torch.ones_like(norms)
    if not torch.allclose(norms, expected, rtol=_UNIT_NORM_RTOL, atol=_UNIT_NORM_ATOL):
        raise ValueError(f"{name} rows must be unit-norm within tolerance")


def _validate_scalar_weight(name: str, value, reference):
    import torch

    weight = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if weight.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not torch.isfinite(weight):
        raise FloatingPointError(f"{name} must be finite")
    return weight
