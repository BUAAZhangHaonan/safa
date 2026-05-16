#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


class _EmbeddingWrapper:
    def __init__(self, model):
        import torch

        self.module = _TorchEmbeddingWrapper(model)
        self.module.eval()
        self.torch = torch

    def export(self, output_path: Path, input_size: int) -> None:
        example = self.torch.randn(1, 3, input_size, input_size)
        traced = self.torch.jit.trace(self.module, example, strict=True)
        test_output = traced(example)
        if tuple(test_output.shape) != (1, 512):
            raise RuntimeError(f"Exported recognizer emitted {tuple(test_output.shape)}, expected (1, 512)")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(output_path))


def _normalize_output(output):
    import torch
    import torch.nn.functional as F

    if isinstance(output, dict):
        for key in ("embedding", "embeddings", "features", "last_hidden_state"):
            if key in output:
                output = output[key]
                break
        else:
            raise RuntimeError(f"Unsupported dict output keys from recognizer: {sorted(output)}")
    elif isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Recognizer returned an empty tuple/list")
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"Recognizer returned unsupported output type: {type(output)!r}")
    if output.ndim != 2 or output.shape[1] != 512:
        raise RuntimeError(f"Recognizer emitted {tuple(output.shape)}, expected [B,512]")
    return F.normalize(output.float(), p=2, dim=1)


def _make_torch_wrapper_class():
    import torch

    class TorchEmbeddingWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.eval()

        def forward(self, images):
            return _normalize_output(self.model(images))

    return TorchEmbeddingWrapper


_TorchEmbeddingWrapper = _make_torch_wrapper_class()


def export_facenet(output_path: Path) -> None:
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError as exc:
        raise RuntimeError("facenet-pytorch is required to export FaceNet") from exc
    model = InceptionResnetV1(pretrained="vggface2").eval()
    _EmbeddingWrapper(model).export(output_path, input_size=160)


def export_adaface(output_path: Path, repo_id: str) -> None:
    try:
        from transformers import AutoModel
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("transformers and huggingface_hub are required to export AdaFace") from exc
    import os
    import sys

    snapshot_path = Path(snapshot_download(repo_id=repo_id))
    sys.path.insert(0, str(snapshot_path))
    cwd = Path.cwd()
    try:
        os.chdir(snapshot_path)
        model = AutoModel.from_pretrained(str(snapshot_path), trust_remote_code=True).eval()
    finally:
        os.chdir(cwd)
    backbone = getattr(model, "model", model)
    _EmbeddingWrapper(backbone).export(output_path, input_size=112)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export evaluation-only privacy recognizers to TorchScript.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/privacy"))
    parser.add_argument("--which", choices=("facenet", "adaface", "both"), default="both")
    parser.add_argument("--adaface-repo", default="minchul/cvlface_adaface_ir50_webface4m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.which in {"facenet", "both"}:
        export_facenet(args.out_dir / "facenet.pt")
        print(f"wrote {args.out_dir / 'facenet.pt'}")
    if args.which in {"adaface", "both"}:
        export_adaface(args.out_dir / "adaface.pt", args.adaface_repo)
        print(f"wrote {args.out_dir / 'adaface.pt'}")


if __name__ == "__main__":
    main()
