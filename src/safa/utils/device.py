from __future__ import annotations


def require_cuda_device(device: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model training") from exc
    if not device.startswith("cuda"):
        raise RuntimeError(f"CPU execution is not allowed for experiment runs: requested device={device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required and is not available")
    parsed = torch.device(device)
    if parsed.index is not None and parsed.index not in {0, 1, 2, 3}:
        raise RuntimeError(f"Only GPU indices 0,1,2,3 are allowed; requested {parsed.index}")
    return parsed


def assert_finite_tensor(name: str, tensor) -> None:
    import torch

    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values detected in tensor: {name}")

