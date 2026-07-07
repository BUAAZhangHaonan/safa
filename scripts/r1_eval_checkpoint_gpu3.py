#!/usr/bin/env python3
"""Round 1 GPU 3: eval a checkpoint with LoRA wrap (for r1_lora_repr_only_*)."""
from __future__ import annotations
import sys
import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path

REPO = Path("/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization")
sys.path.insert(0, str(REPO / "src"))

# Allow override via env var
CKPT_NAME = os.environ.get("R1_CKPT_NAME", "r1_lora_repr_only_lr0.5_gpu1")
CKPT = REPO / f"artifacts/checkpoints/{CKPT_NAME}/best.pt"
LORA_TARGETS = os.environ.get("R1_LORA_TARGETS", "adaLN_modulation.1").split(",")
LORA_RANK = int(os.environ.get("R1_LORA_RANK", "8"))
LORA_ALPHA = float(os.environ.get("R1_LORA_ALPHA", "4.0"))
VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
VAL_FEATURES = REPO / "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"
E0_CKPT = REPO / "artifacts/checkpoints/e0_medium_v1/best.pt"
VAE_PATH = REPO / "artifacts/checkpoints/external/sd-vae-ft-ema"
OUT_DIR = REPO / f"artifacts/r1_eval2048_{CKPT_NAME}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "result.json"

DEVICE = "cuda:0"
SEED = 1337


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    has_lora = any("lora_a" in k for k in state_dict.keys())
    print(f"[r1_eval] CKPT={CKPT_NAME} has_lora={has_lora} targets={LORA_TARGETS}")

    if has_lora:
        from safa.models.generator import build_generator, require_generator_model_config
        from safa.models.peft_lora import wrap_backbone_with_peft_lora
        import safa.evaluation.runner as runner

        # Detect peft_lora full stack vs pure LoRA sweep
        has_gated = any("gated_low_rank_z" in k for k in state_dict.keys())
        has_bank = any("generic_bank" in k for k in state_dict.keys())
        has_null = any("_peft_lora_null_embed" in k for k in state_dict.keys())
        has_final_lora = any("final_layer.adaLN_modulation.1.lora" in k for k in state_dict.keys())
        # If checkpoint only has block-level LoRA (no gated/bank/null/final) -> pure lora_sweep wrap
        is_peft_lora_full = has_gated or has_bank or has_null or has_final_lora
        # Auto-detect LoRA targets from checkpoint
        detected_targets = set()
        for k in state_dict.keys():
            if "lora_a" in k:
                parts = k.split(".")
                # vector_field.blocks.{N}.{path}.lora_a.weight -> parts[3:-2]
                if len(parts) > 5 and parts[0] == "vector_field" and parts[1] == "blocks":
                    target_path = ".".join(parts[3:-2])
                    detected_targets.add(target_path)
        print(f"[r1_eval] gated={has_gated} bank={has_bank} null={has_null} final_lora={has_final_lora} -> is_peft_lora_full={is_peft_lora_full}")
        print(f"[r1_eval] detected LoRA targets: {sorted(detected_targets)}")
        # If pure lora_sweep, override LORA_TARGETS with detected
        if not is_peft_lora_full and detected_targets:
            LORA_TARGETS_OVERRIDE = sorted(detected_targets)
        else:
            LORA_TARGETS_OVERRIDE = LORA_TARGETS

        def patched_load(checkpoint_path, cfg, dev):
            pl = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)
            mc = require_generator_model_config(pl, str(checkpoint_path))
            if "sit_pretrained_path" in mc:
                mc["sit_pretrained_path"] = ""
            gen = build_generator(mc)
            if is_peft_lora_full:
                # Full peft_lora stack (training-time idempotent wrap, same flags as config)
                bb = gen.vector_field
                z_dim = int(bb.z_embedder[0].in_features)
                hidden_size = int(bb.x_embedder.out_channels)
                # Detect generic_bank size from checkpoint
                bank_keys = [k for k in state_dict.keys() if "generic_bank.embeddings" in k]
                if bank_keys:
                    bank_shape = state_dict[bank_keys[0]].shape
                    num_gen = bank_shape[0] if len(bank_shape) >= 1 else 16
                    if num_gen == 1:
                        generic_mode = "null"
                        enable_bank = False  # null anchor uses 1 embedding but no bank learning
                    else:
                        generic_mode = "bank"
                        enable_bank = True
                else:
                    num_gen = 16
                    generic_mode = "bank"
                    enable_bank = True
                print(f"[r1_eval] num_generic={num_gen} generic_mode={generic_mode} enable_bank={enable_bank}")
                wrap_backbone_with_peft_lora(
                    bb,
                    lora_rank=LORA_RANK, lora_alpha=1.0,
                    z_dim=z_dim, hidden_size=hidden_size,
                    num_generic_embeddings=num_gen,
                    enable_lora=True,
                    enable_gated_low_rank=has_gated,
                    enable_generic_bank=enable_bank,
                    lora_blocks="all",
                    generic_mode=generic_mode,
                    freeze_null_embed=False,
                )
            else:
                # Pure lora_sweep wrap
                from safa.models.peft_lora import wrap_backbone_with_lora_target
                wrap_backbone_with_lora_target(
                    gen.vector_field, target_modules=LORA_TARGETS_OVERRIDE,
                    rank=LORA_RANK, alpha=LORA_ALPHA,
                )
            gen.load_state_dict(pl["model_state_dict"])
            return gen.to(dev).eval()
        runner._load_generator = patched_load

    config = {
        "seed": SEED, "device": DEVICE, "num_workers": 4, "batch_size": 32,
        "image_size": 256, "index": str(VAL_INDEX), "features": str(VAL_FEATURES),
        "e0_checkpoint": str(E0_CKPT), "g_checkpoint": str(CKPT),
        "checkpoint_model": "raw", "out_json": str(OUT_JSON),
        "per_sample_jsonl": str(OUT_DIR / "per_sample.jsonl"),
        "sample_dir": str(OUT_DIR / "samples"),
        "generated_image_dir": str(OUT_DIR / "generated_images"),
        "face_detection": {"enabled": True, "model_name": "buffalo_l",
                          "threshold": 0.5, "single_face_eq1_threshold": 0.98,
                          "latent_cosine_threshold": 0.95},
        "privacy": {"enabled": False}, "anti_steg": {"enabled": False},
        "vae_path": str(VAE_PATH), "vae_scaling_factor": 0.18215,
        "pixel_image_size": 256, "latent_training": True,
    }

    print(f"[r1_eval] running full eval (3969 samples)")
    from safa.evaluation.runner import run_eval_from_config
    run_eval_from_config(config)
    print(f"[r1_eval] eval done")

    with VAL_INDEX.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    real_paths = [Path(r["image_path"]) for r in rows[:2048]]
    gen_files = sorted((OUT_DIR / "generated_images").glob("*.png"))[:2048]

    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    fid_metric.eval()
    from torchvision import transforms
    from PIL import Image
    to_tensor = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        t = (to_tensor(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid_metric.update(t, real=True)
    for gf in gen_files:
        img = Image.open(gf).convert("RGB")
        t = (to_tensor(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid_metric.update(t, real=False)
    fid_value = fid_metric.compute().item()
    print(f"[r1_eval] FID(2048) = {fid_value:.4f}")

    sharp_values = []
    for gf in gen_files[:512]:
        img_np = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
        if img_np is None: continue
        sharp_values.append(cv2.Laplacian(img_np, cv2.CV_64F).var())
    sharp_mean = float(np.mean(sharp_values)) if sharp_values else float("nan")

    payload_out = json.loads(OUT_JSON.read_text())
    payload_out["fid_2048"] = fid_value
    payload_out["sharpness_mean_generated"] = sharp_mean
    payload_out["n_real"] = len(real_paths)
    payload_out["n_generated"] = len(gen_files)
    OUT_JSON.write_text(json.dumps(payload_out, indent=2))
    print(f"[r1_eval] DONE {CKPT_NAME} FID={fid_value:.4f} sharp={sharp_mean:.2f}")


if __name__ == "__main__":
    main()
