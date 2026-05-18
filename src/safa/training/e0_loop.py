from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math

from safa.data.dataset import AffectNetRecords
from safa.models.e0 import E0Config, build_e0, checkpoint_payload
from safa.training.transforms import eval_transform, train_transform, train_transform_strong
from safa.utils.device import assert_finite_tensor, require_cuda_device
from safa.utils.seed import set_seed


def train_e0_from_config(config: dict) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    set_seed(int(config["seed"]))
    device = require_cuda_device(str(config["device"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- augmentation selection ---
    augmentation = str(config.get("augmentation", "default"))
    if augmentation == "strong":
        t_transform = train_transform_strong(int(config["image_size"]))
    else:
        t_transform = train_transform(int(config["image_size"]))

    train_set = AffectNetRecords(config["train_index"], transform=t_transform)
    val_set = AffectNetRecords(config["val_index"], transform=eval_transform(int(config["image_size"])))
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )
    model_config = E0Config(
        num_classes=int(config["num_classes"]),
        embedding_dim=int(config["embedding_dim"]),
        imagenet_weights=str(config["imagenet_weights"]),
    )
    model = build_e0(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))

    # --- class weighting (effective number of samples) ---
    class_weights = None
    if config.get("class_weight", False):
        class_counts = Counter(record.label for record in train_set.records)
        num_classes = int(config["num_classes"])
        beta = 0.9999
        effective_num = 1.0 - beta ** torch.tensor(
            [float(class_counts.get(i, 0)) for i in range(num_classes)], dtype=torch.float64
        )
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        class_weights = weights.float().to(device)

    # --- label smoothing ---
    label_smoothing = float(config.get("label_smoothing", 0.0))

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # --- LR scheduler ---
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- warmup ---
    warmup_epochs = int(config.get("warmup_epochs", 0))
    base_lr = float(config["learning_rate"])

    # --- early stopping ---
    early_stopping_patience = int(config.get("early_stopping_patience", 0))

    majority = _majority_baseline(train_set.records, val_set.records)
    best_acc = -1.0
    best_metrics: dict = {}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # --- warmup LR adjustment ---
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * warmup_factor

        model.train()
        train_loss = 0.0
        seen = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"train_e0 epoch={epoch}")):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            assert_finite_tensor("e0_embedding", output["embedding"])
            assert_finite_tensor("e0_logits", output["logits"])
            loss = criterion(output["logits"], labels)
            assert_finite_tensor("e0_loss", loss)
            # --- NaN / Inf check ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: loss is {loss.item()} at epoch={epoch} batch={batch_idx}, skipping backward pass")
                continue
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * labels.numel()
            seen += labels.numel()

        # --- step scheduler (after warmup phase, cosine takes over) ---
        if epoch >= warmup_epochs:
            scheduler.step()

        metrics = evaluate_e0(model, val_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(seen, 1),
                "majority_val_accuracy": majority,
            }
        )
        _write_json(out_dir / "last_metrics.json", metrics)
        torch.save(checkpoint_payload(model, model_config, metrics), out_dir / "last.pt")
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_metrics = metrics
            epochs_without_improvement = 0
            torch.save(checkpoint_payload(model, model_config, metrics), out_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        # --- early stopping check ---
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no improvement for {early_stopping_patience} epochs")
            break

    manifest = {
        "checkpoint": str(out_dir / "best.pt"),
        "embedding_dim": model_config.embedding_dim,
        "num_classes": model_config.num_classes,
        "l2_normalized": True,
        "best_metrics": best_metrics,
        "majority_val_accuracy": majority,
        "passes_majority_baseline": bool(best_metrics.get("accuracy", 0.0) > majority),
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def evaluate_e0(model, loader, device) -> dict:
    import torch

    model.eval()
    correct = 0
    total = 0
    logits_abs_sum = 0.0
    # --- per-class accuracy tracking ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            output = model(images)
            assert_finite_tensor("eval_e0_embedding", output["embedding"])
            norms = output["embedding"].float().norm(dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
                raise RuntimeError("E0 embeddings are not L2-normalized")
            predictions = output["logits"].argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            logits_abs_sum += float(output["logits"].abs().sum().item())
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
    if total == 0:
        raise ValueError("Validation loader produced zero samples")

    # --- compute per-class accuracy ---
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    per_class_acc = {}
    for cls in sorted(all_labels.unique().tolist()):
        cls = int(cls)
        mask = all_labels == cls
        cls_correct = int((all_preds[mask] == cls).sum().item())
        cls_total = int(mask.sum().item())
        acc = cls_correct / cls_total if cls_total > 0 else 0.0
        per_class_acc[f"class_{cls}"] = acc
        print(f"  class_{cls}: accuracy={acc:.4f} ({cls_correct}/{cls_total})")

    result = {
        "accuracy": correct / total,
        "num_samples": total,
        "mean_abs_logit": logits_abs_sum / total,
        "per_class_accuracy": per_class_acc,
    }
    return result


def _majority_baseline(train_records, val_records) -> float:
    counts = Counter(record.label for record in train_records)
    if not counts:
        raise ValueError("Cannot compute majority baseline from empty training records")
    majority_label = counts.most_common(1)[0][0]
    return sum(1 for record in val_records if record.label == majority_label) / len(val_records)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
