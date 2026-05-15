from __future__ import annotations

from pathlib import Path
import json

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.models.e0 import assert_e0_frozen, freeze_e0, load_e0_checkpoint
from safa.models.generator import ZOnlyGenerator
from safa.training.audit import audit_no_identity_supervision
from safa.training.losses import cosine_cycle_loss, normalize_for_e0, total_variation_loss
from safa.training.transforms import generator_image_transform
from safa.utils.device import assert_finite_tensor, require_cuda_device
from safa.utils.seed import set_seed


def train_g_from_config(config: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    set_seed(int(config["seed"]))
    audit_no_identity_supervision(config)
    device = require_cuda_device(str(config["device"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    e0, _ = load_e0_checkpoint(config["e0_checkpoint"], device=str(device))
    e0.to(device)
    freeze_e0(e0)

    generator = ZOnlyGenerator(embedding_dim=int(config["embedding_dim"]), image_size=int(config["image_size"])).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    assert_e0_frozen(e0, optimizer)

    train_set = FeatureAlignedAffectNet(
        config["train_index"],
        config["train_features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
    loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )
    weights = config["loss_weights"]
    metrics = {}
    for epoch in range(int(config["epochs"])):
        generator.train()
        totals = {"loss": 0.0, "cycle": 0.0, "semantic_ce": 0.0, "image_tv": 0.0}
        seen = 0
        for batch in tqdm(loader, desc=f"train_g epoch={epoch}"):
            z = batch["z"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            generated = generator(z)
            assert_finite_tensor("generated_image", generated)
            e0_out = e0(normalize_for_e0(generated))
            pred_z = e0_out["embedding"]
            cycle = cosine_cycle_loss(pred_z, z)
            semantic_ce = F.cross_entropy(e0_out["logits"], labels)
            image_tv = total_variation_loss(generated)
            loss = (
                float(weights["cycle"]) * cycle
                + float(weights["semantic_ce"]) * semantic_ce
                + float(weights["image_tv"]) * image_tv
            )
            assert_finite_tensor("g_loss", loss)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.numel())
            seen += batch_size
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["cycle"] += float(cycle.detach().cpu()) * batch_size
            totals["semantic_ce"] += float(semantic_ce.detach().cpu()) * batch_size
            totals["image_tv"] += float(image_tv.detach().cpu()) * batch_size
        metrics = {key: value / max(seen, 1) for key, value in totals.items()}
        metrics["epoch"] = epoch
        _save_generator(out_dir / "last.pt", generator, config, metrics)
        _write_json(out_dir / "last_metrics.json", metrics)
        _save_generator(out_dir / "best.pt", generator, config, metrics)
    manifest = {"checkpoint": str(out_dir / "best.pt"), "metrics": metrics, "generator_input": "z_only", "identity_supervision": False}
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _save_generator(path: Path, generator, config: dict, metrics: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "model_config": {
                "embedding_dim": int(config["embedding_dim"]),
                "image_size": int(config["image_size"]),
            },
            "metrics": metrics,
        },
        path,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
