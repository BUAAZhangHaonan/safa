from __future__ import annotations

from pathlib import Path

from safa.utils.hashing import sha256_file


class TorchScriptRecognizer:
    def __init__(self, name: str, checkpoint: str | Path, device: str, embedding_dim: int, input_size: int):
        import torch

        self.name = name
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Recognizer checkpoint missing for {name}: {checkpoint_path}")
        self.model = torch.jit.load(str(checkpoint_path), map_location=self.device).eval()
        self.embedding_dim = int(embedding_dim)
        self.input_size = int(input_size)
        if self.embedding_dim <= 0:
            raise ValueError(f"Recognizer {name} embedding_dim must be positive, got {embedding_dim}")
        if self.input_size <= 0:
            raise ValueError(f"Recognizer {name} input_size must be positive, got {input_size}")

    def embed(self, images):
        import torch.nn.functional as F

        resized = F.interpolate(images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        normalized = (resized - 0.5) / 0.5
        output = self.model(normalized.to(self.device))
        if output.ndim != 2 or output.shape[1] != self.embedding_dim:
            raise ValueError(f"Recognizer {self.name} emitted shape {tuple(output.shape)}, expected [B,{self.embedding_dim}]")
        return F.normalize(output.float(), p=2, dim=1)


class InsightFaceRecognizer:
    def __init__(self, name: str, model_name: str, device: str):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("insightface is required for ArcFace privacy evaluation") from exc
        if not device.startswith("cuda"):
            raise RuntimeError("InsightFace evaluation requires a CUDA device")
        ctx_id = int(device.split(":")[1]) if ":" in device else 0
        self.name = name
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=(224, 224))

    def embed(self, images):
        import numpy as np
        import torch
        import torch.nn.functional as F

        embeddings = []
        for image in images.detach().cpu():
            array = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            bgr = array[:, :, ::-1]
            faces = self.app.get(bgr)
            if len(faces) != 1:
                raise RuntimeError(f"Recognizer {self.name} expected exactly one face, detected {len(faces)}")
            embeddings.append(torch.from_numpy(faces[0].embedding).float())
        return F.normalize(torch.stack(embeddings, dim=0).to(images.device), p=2, dim=1)


class InsightFaceDetector:
    def __init__(self, model_name: str, device: str):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("insightface is required for ArcFace face detection monitoring") from exc
        if not device.startswith("cuda"):
            raise RuntimeError("ArcFace face detection monitoring requires a CUDA device")
        ctx_id = int(device.split(":")[1]) if ":" in device else 0
        self.name = model_name
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=(224, 224))

    def detect_counts(self, images) -> list[int]:
        import numpy as np

        counts = []
        for image in images.detach().cpu():
            array = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            bgr = array[:, :, ::-1]
            counts.append(len(self.app.get(bgr)))
        return counts


def validate_recognizer_configs(configs: list[dict]) -> None:
    if not isinstance(configs, list):
        raise ValueError("privacy.recognizers must be a list")
    for index, config in enumerate(configs):
        context = f"privacy.recognizers[{index}]"
        if not isinstance(config, dict):
            raise ValueError(f"{context} must be a mapping")
        _require_field(config, "name", context)
        kind = _require_field(config, "type", context)
        if kind == "insightface":
            _require_field(config, "model_name", context)
        elif kind == "torchscript":
            _require_field(config, "checkpoint", context)
            _positive_int_field(config, "embedding_dim", context)
            _positive_int_field(config, "input_size", context)
        else:
            raise ValueError(f"Unknown recognizer type: {kind}")


def _require_field(config: dict, field: str, context: str):
    if field not in config:
        raise ValueError(f"{context}.{field} is required")
    return config[field]


def _positive_int_field(config: dict, field: str, context: str) -> int:
    value = _require_field(config, field, context)
    if isinstance(value, bool):
        raise ValueError(f"{context}.{field} must be a positive integer, got bool")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{context}.{field} must be a positive integer, got {value!r}")
    return parsed


def build_recognizers(configs: list[dict], device: str):
    validate_recognizer_configs(configs)
    recognizers = []
    for config in configs:
        kind = config["type"]
        if kind == "insightface":
            recognizers.append(InsightFaceRecognizer(config["name"], config["model_name"], device))
        elif kind == "torchscript":
            recognizers.append(
                TorchScriptRecognizer(
                    name=config["name"],
                    checkpoint=config["checkpoint"],
                    device=device,
                    embedding_dim=int(config["embedding_dim"]),
                    input_size=int(config["input_size"]),
                )
            )
        else:
            raise ValueError(f"Unknown recognizer type: {kind}")
    return recognizers


def describe_recognizer_assets(configs: list[dict]) -> list[dict]:
    validate_recognizer_configs(configs)
    assets = []
    for config in configs:
        kind = config["type"]
        if kind == "insightface":
            assets.append({"name": config["name"], "type": kind, "model_name": config["model_name"]})
        elif kind == "torchscript":
            checkpoint = Path(config["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Recognizer checkpoint missing for {config['name']}: {checkpoint}")
            assets.append(
                {
                    "name": config["name"],
                    "type": kind,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "embedding_dim": int(config["embedding_dim"]),
                    "input_size": int(config["input_size"]),
                }
            )
        else:
            raise ValueError(f"Unknown recognizer type: {kind}")
    return assets
