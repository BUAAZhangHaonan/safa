from __future__ import annotations

from pathlib import Path


class TorchScriptRecognizer:
    def __init__(self, name: str, checkpoint: str | Path, device: str, embedding_dim: int = 512):
        import torch

        self.name = name
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Recognizer checkpoint missing for {name}: {checkpoint_path}")
        self.model = torch.jit.load(str(checkpoint_path), map_location=self.device).eval()
        self.embedding_dim = int(embedding_dim)

    def embed(self, images):
        import torch.nn.functional as F

        resized = F.interpolate(images, size=(112, 112), mode="bilinear", align_corners=False)
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


def build_recognizers(configs: list[dict], device: str):
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
                    embedding_dim=int(config.get("embedding_dim", 512)),
                )
            )
        else:
            raise ValueError(f"Unknown recognizer type: {kind}")
    return recognizers

