from __future__ import annotations


def cosine_cycle_loss(pred_embedding, target_embedding):
    import torch.nn.functional as F

    return (1.0 - F.cosine_similarity(pred_embedding, target_embedding, dim=1)).mean()


def total_variation_loss(images):
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def normalize_for_e0(images):
    import torch

    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std

