from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class E0Config:
    num_classes: int = 8
    embedding_dim: int = 512
    imagenet_weights: str = "IMAGENET1K_V2"


def build_e0(config: E0Config, allow_random_init: bool = False):
    import torch
    from torch import nn
    from torchvision.models import ResNet50_Weights, resnet50

    if config.embedding_dim != 512:
        raise ValueError(f"E0 embedding_dim must be 512 for the minimal validation, got {config.embedding_dim}")
    weights = None
    if config.imagenet_weights:
        try:
            weights = getattr(ResNet50_Weights, config.imagenet_weights)
        except AttributeError as exc:
            raise ValueError(f"Unknown torchvision ResNet-50 weights: {config.imagenet_weights}") from exc
    elif not allow_random_init:
        raise RuntimeError("Random E0 initialization is not allowed for experiment runs")
    backbone = resnet50(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return EmotionEncoder(backbone=backbone, in_features=in_features, embedding_dim=config.embedding_dim, num_classes=config.num_classes)


class EmotionEncoder:
    def __new__(cls, backbone, in_features: int, embedding_dim: int, num_classes: int):
        import torch
        from torch import nn
        import torch.nn.functional as F

        class _EmotionEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.projector = nn.Linear(in_features, embedding_dim)
                self.classifier = nn.Linear(embedding_dim, num_classes)
                self.embedding_dim = embedding_dim
                self.num_classes = num_classes

            def forward(self, images):
                features = self.backbone(images)
                embedding = F.normalize(self.projector(features), p=2, dim=1)
                logits = self.classifier(embedding)
                return {"embedding": embedding, "logits": logits}

        return _EmotionEncoder()


def freeze_e0(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def assert_e0_frozen(model, optimizer=None) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"E0 has trainable parameters after freezing: {trainable[:20]}")
    if optimizer is not None:
        e0_param_ids = {id(parameter) for parameter in model.parameters()}
        for group in optimizer.param_groups:
            overlap = [parameter for parameter in group["params"] if id(parameter) in e0_param_ids]
            if overlap:
                raise RuntimeError("Optimizer contains E0 parameters")


def load_e0_checkpoint(path: str | Path, device: str | None = None):
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"E0 checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    cfg = checkpoint.get("model_config")
    if not isinstance(cfg, dict):
        raise ValueError(f"E0 checkpoint missing model_config: {checkpoint_path}")
    load_config = E0Config(
        num_classes=int(cfg["num_classes"]),
        embedding_dim=int(cfg["embedding_dim"]),
        imagenet_weights="",
    )
    model = build_e0(load_config, allow_random_init=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def checkpoint_payload(model, config: E0Config, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_classes": config.num_classes,
            "embedding_dim": config.embedding_dim,
            "imagenet_weights": config.imagenet_weights,
        },
        "metrics": metrics,
    }
