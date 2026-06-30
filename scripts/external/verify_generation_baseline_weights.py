from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MEANFLOW_WEIGHTS = {
    "meanflow_sit_b2": {
        "path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_2_imagenet256.pt",
        "source": "zhuyu-cs/MeanFlow ImageNet256 SiT-B/2 MeanFlow EMA",
        "required_shapes": {
            "pos_embed": [1, 256, 768],
            "x_embedder.proj.weight": [768, 4, 2, 2],
            "blocks.11.attn.qkv.weight": [2304, 768],
            "final_layer.linear.weight": [16, 768],
        },
    },
    "meanflow_sit_l2": {
        "path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt",
        "source": "zhuyu-cs/MeanFlow ImageNet256 SiT-L/2 MeanFlow EMA",
        "required_shapes": {
            "pos_embed": [1, 256, 1024],
            "x_embedder.proj.weight": [1024, 4, 2, 2],
            "blocks.23.attn.qkv.weight": [3072, 1024],
            "final_layer.linear.weight": [16, 1024],
        },
    },
}

NO_RELIABLE_PUBLIC_PRETRAINED = {
    "sit_diffusion_b2": {
        "configs": ["configs/medium_v2/experiments/e22_sit_diffusion_b2_face_mixed_2400ep.yaml"],
        "reason": "No reliable public SAFA-compatible SiT-Diffusion-B/2 pretrained checkpoint is known for this matrix.",
    },
    "sit_diffusion_l2": {
        "configs": ["configs/medium_v2/experiments/e17_sit_diffusion_l2_face_mixed_2400ep.yaml"],
        "reason": "No reliable public SAFA-compatible SiT-Diffusion-L/2 pretrained checkpoint is known for this matrix.",
    },
    "latent_consistency_b2": {
        "configs": ["configs/medium_v2/experiments/e23_latent_consistency_b2_face_mixed_2400ep.yaml"],
        "reason": "No reliable public SAFA-compatible Latent-Consistency-B/2 pretrained checkpoint is known for this matrix.",
    },
    "latent_consistency_l2": {
        "configs": ["configs/medium_v2/experiments/e18_latent_consistency_l2_face_mixed_2400ep.yaml"],
        "reason": "No reliable public SAFA-compatible Latent-Consistency-L/2 pretrained checkpoint is known for this matrix.",
    },
}


def _extract_state_dict(payload: Any, state_key: str | None) -> dict[str, Any]:
    if state_key:
        if not isinstance(payload, dict) or state_key not in payload:
            raise KeyError(f"checkpoint missing state_key {state_key!r}")
        state = payload[state_key]
    elif isinstance(payload, dict):
        for key in ("ema", "model", "model_state_dict", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                state = value
                break
        else:
            state = payload
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state_dict must be a dict, got {type(state).__name__}")
    return state


def _strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if not all(isinstance(key, str) for key in state):
        return state
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def verify_weight(root: Path, name: str, spec: dict[str, Any], state_key: str | None) -> tuple[dict[str, Any], bool]:
    path = root / spec["path"]
    result: dict[str, Any] = {
        "name": name,
        "path": spec["path"],
        "source": spec["source"],
        "exists": path.is_file(),
        "ok": False,
        "shape_checks": [],
    }
    if not path.is_file():
        result["status"] = "missing"
        return result, True

    try:
        import torch
    except ImportError as exc:
        result["status"] = "error"
        result["error"] = f"torch import failed: {exc}"
        return result, False

    try:
        payload = torch.load(path, map_location="cpu")
        state = _strip_module_prefix(_extract_state_dict(payload, state_key))
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result, False

    all_ok = True
    for key, expected_shape in spec["required_shapes"].items():
        actual_shape = _shape_of(state.get(key))
        check_ok = actual_shape == expected_shape
        all_ok = all_ok and check_ok
        result["shape_checks"].append(
            {
                "key": key,
                "expected": expected_shape,
                "actual": actual_shape,
                "ok": check_ok,
            }
        )
    result["ok"] = all_ok
    result["status"] = "verified" if all_ok else "shape_mismatch"
    return result, all_ok


def build_manifest(root: Path, state_key: str | None) -> tuple[dict[str, Any], bool]:
    weights = []
    all_ok = True
    for name, spec in MEANFLOW_WEIGHTS.items():
        result, ok = verify_weight(root, name, spec, state_key)
        weights.append(result)
        all_ok = all_ok and ok
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "meanflow_sit_weights": weights,
        "unavailable_public_pretrained": NO_RELIABLE_PUBLIC_PRETRAINED,
        "policy": {
            "do_not_fake_downloads": True,
            "notes": [
                "Only MeanFlow-SiT B/2 and L/2 weights are auto-handled.",
                "Diffusion and consistency B/L configs are matrix training targets when no reliable public checkpoint is available.",
            ],
        },
    }
    return manifest, all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generation baseline external checkpoint shapes.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/generation_baseline_weights_manifest.json"))
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--require-existing", action="append", choices=sorted(MEANFLOW_WEIGHTS), default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest, ok = build_manifest(root, args.state_key)
    by_name = {item["name"]: item for item in manifest["meanflow_sit_weights"]}
    missing_required = [name for name in args.require_existing if not by_name[name]["exists"]]
    if missing_required:
        ok = False
        manifest["missing_required"] = missing_required
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
