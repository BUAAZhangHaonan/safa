from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class E0Config:
    num_classes: int = 8
    embedding_dim: int = 512
    imagenet_weights: str = "IMAGENET1K_V2"
    backbone: str = "resnet50"
    backbone_path: str = ""


_SUPPORTED_BACKBONES = (
    "resnet18",
    "resnet50",
    "vgg16",
    "dinov2_large",
    "dinov3_vitl16",
    "convnext_tiny",
    "swin_tiny",
    "iresnet100",
    "mobilenetv3_large",
)


def _build_resnet(name: str, imagenet_weights: str):
    from torch import nn
    from torchvision.models import ResNet50_Weights, ResNet18_Weights, resnet18, resnet50

    if name == "resnet50":
        weights_enum = ResNet50_Weights
        ctor = resnet50
    elif name == "resnet18":
        weights_enum = ResNet18_Weights
        ctor = resnet18
    else:
        raise ValueError(f"Unsupported resnet variant: {name}")

    weights = None
    if imagenet_weights:
        try:
            weights = getattr(weights_enum, imagenet_weights)
        except AttributeError as exc:
            raise ValueError(f"Unknown torchvision {name} weights: {imagenet_weights}") from exc
    model = ctor(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, in_features


def _build_vgg16(imagenet_weights: str):
    from torch import nn
    from torchvision.models import VGG16_Weights, vgg16

    weights = None
    if imagenet_weights:
        try:
            weights = getattr(VGG16_Weights, imagenet_weights)
        except AttributeError as exc:
            raise ValueError(f"Unknown torchvision VGG16 weights: {imagenet_weights}") from exc
    model = vgg16(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    return model, in_features


def _build_dinov2_large(backbone_path: str):
    from torch import nn

    if not backbone_path:
        raise ValueError("dinov2_large backbone requires backbone_path pointing to the local model dir")
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required for dinov2_large backbone") from exc

    class _DINOv2Wrapper(nn.Module):
        def __init__(self, path: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(path)
            self.hidden_size = int(self.encoder.config.hidden_size)

        def forward(self, images):
            outputs = self.encoder(pixel_values=images)
            cls_token = outputs.last_hidden_state[:, 0, :]
            return cls_token

    wrapper = _DINOv2Wrapper(backbone_path)
    return wrapper, wrapper.hidden_size


def _build_dinov3_vitl16(backbone_path: str):
    from torch import nn

    if not backbone_path:
        raise ValueError("dinov3_vitl16 backbone requires backbone_path pointing to the local model dir")
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required for dinov3_vitl16 backbone") from exc

    class _DINOv3Wrapper(nn.Module):
        def __init__(self, path: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(path)
            self.hidden_size = int(self.encoder.config.hidden_size)

        def forward(self, images):
            outputs = self.encoder(pixel_values=images)
            cls_token = outputs.last_hidden_state[:, 0, :]
            return cls_token

    wrapper = _DINOv3Wrapper(backbone_path)
    return wrapper, wrapper.hidden_size


def _build_convnext_tiny(imagenet_weights: str):
    from torch import nn
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    weights = None
    if imagenet_weights:
        try:
            weights = getattr(ConvNeXt_Tiny_Weights, imagenet_weights)
        except AttributeError as exc:
            raise ValueError(f"Unknown torchvision ConvNeXt-Tiny weights: {imagenet_weights}") from exc
    model = convnext_tiny(weights=weights)
    # m.features -> (B,768,7,7); m.avgpool -> (B,768,1,1); m.classifier[0] is LayerNorm2d(768) — must apply on 4D
    features = nn.Sequential(model.features, model.avgpool, model.classifier[0], nn.Flatten(1))
    return features, 768


def _build_swin_tiny(imagenet_weights: str):
    from torch import nn
    from torchvision.models import swin_t, Swin_T_Weights

    weights = None
    if imagenet_weights:
        try:
            weights = getattr(Swin_T_Weights, imagenet_weights)
        except AttributeError as exc:
            raise ValueError(f"Unknown torchvision Swin-Tiny weights: {imagenet_weights}") from exc
    model = swin_t(weights=weights)
    # m.features outputs (B,H,W,768) channels-last; m.norm final LN over last dim; m.permute -> (B,768,H,W); m.avgpool -> (B,768,1,1)
    features = nn.Sequential(model.features, model.norm, model.permute, model.avgpool, nn.Flatten(1))
    return features, 768


def _build_mobilenetv3_large(imagenet_weights: str):
    from torch import nn
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    weights = None
    if imagenet_weights:
        try:
            weights = getattr(MobileNet_V3_Large_Weights, imagenet_weights)
        except AttributeError as exc:
            raise ValueError(f"Unknown torchvision MobileNetV3-Large weights: {imagenet_weights}") from exc
    model = mobilenet_v3_large(weights=weights)
    # m.features -> (B,960,7,7); m.avgpool -> (B,960,1,1)
    features = nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    return features, 960


def _build_iresnet100(backbone_path: str):
    import torch

    if not backbone_path:
        raise ValueError("iresnet100 backbone requires backbone_path pointing to a local ArcFace checkpoint")
    from safa.models.backbones.iresnet import iresnet100

    model = iresnet100()
    state = torch.load(backbone_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # drop ArcFace classification head weight (key "weight" — the final 512x512 FC after the IResNet features)
    state = {k: v for k, v in state.items() if k != "weight"}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"iresnet100 load_state_dict missing keys: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"iresnet100 load_state_dict unexpected keys: {unexpected[:10]}")
    return model, 512


def build_e0(config: E0Config, allow_random_init: bool = False):
    from torch import nn

    if config.embedding_dim <= 0:
        raise ValueError(f"E0 embedding_dim must be positive, got {config.embedding_dim}")
    if config.backbone not in _SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone {config.backbone!r}, supported: {_SUPPORTED_BACKBONES}")

    needs_imagenet = config.backbone in {
        "resnet18",
        "resnet50",
        "vgg16",
        "convnext_tiny",
        "swin_tiny",
        "mobilenetv3_large",
    }
    if needs_imagenet:
        if not config.imagenet_weights and not allow_random_init:
            raise RuntimeError("Random E0 initialization is not allowed for experiment runs")

    if config.backbone == "resnet50":
        backbone, in_features = _build_resnet("resnet50", config.imagenet_weights)
    elif config.backbone == "resnet18":
        backbone, in_features = _build_resnet("resnet18", config.imagenet_weights)
    elif config.backbone == "vgg16":
        backbone, in_features = _build_vgg16(config.imagenet_weights)
    elif config.backbone == "dinov2_large":
        backbone, in_features = _build_dinov2_large(config.backbone_path)
    elif config.backbone == "dinov3_vitl16":
        backbone, in_features = _build_dinov3_vitl16(config.backbone_path)
    elif config.backbone == "convnext_tiny":
        backbone, in_features = _build_convnext_tiny(config.imagenet_weights)
    elif config.backbone == "swin_tiny":
        backbone, in_features = _build_swin_tiny(config.imagenet_weights)
    elif config.backbone == "mobilenetv3_large":
        backbone, in_features = _build_mobilenetv3_large(config.imagenet_weights)
    elif config.backbone == "iresnet100":
        backbone, in_features = _build_iresnet100(config.backbone_path)
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")

    return EmotionEncoder(backbone=backbone, in_features=in_features, embedding_dim=config.embedding_dim, num_classes=config.num_classes)


class EmotionEncoder:
    def __new__(cls, backbone, in_features: int, embedding_dim: int, num_classes: int):
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
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
    cfg = checkpoint.get("model_config")
    if not isinstance(cfg, dict):
        raise ValueError(f"E0 checkpoint missing model_config: {checkpoint_path}")
    load_config = E0Config(
        num_classes=int(cfg["num_classes"]),
        embedding_dim=int(cfg["embedding_dim"]),
        imagenet_weights="",
        backbone=str(cfg.get("backbone", "resnet50")),
        backbone_path=str(cfg.get("backbone_path", "")),
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
            "backbone": config.backbone,
            "backbone_path": config.backbone_path,
        },
        "metrics": metrics,
    }
