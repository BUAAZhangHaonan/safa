from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math

from safa.data.dataset import AffectNetRecords
from safa.models.e0 import E0Config, build_e0, checkpoint_payload
from safa.training.transforms import eval_transform, train_transform, train_transform_strong
from safa.utils.device import assert_finite_tensor
from safa.utils.distributed import (
    DistributedContext,
    barrier,
    broadcast_early_stop,
    cleanup_distributed,
    init_distributed,
    reduce_train_metrics,
    unwrap_model,
)
from safa.utils.seed import set_seed


def train_e0_from_config(config: dict) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm

    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    distributed = init_distributed(config)
    device = distributed.device
    num_workers = int(config["num_workers"])
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1 for persistent_workers, got {num_workers}")
    out_dir = Path(config["out_dir"])
    if distributed.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed)

    # --- augmentation selection ---
    augmentation = str(config.get("augmentation", "default"))
    if augmentation == "strong":
        t_transform = train_transform_strong(int(config["image_size"]))
    else:
        t_transform = train_transform(int(config["image_size"]))

    train_set = AffectNetRecords(config["train_index"], transform=t_transform)
    val_set = AffectNetRecords(config["val_index"], transform=eval_transform(int(config["image_size"])))

    train_sampler = (
        DistributedSampler(
            train_set,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=int(config["seed"]),
            drop_last=False,
        )
        if distributed.enabled
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        if distributed.is_main
        else None
    )

    model_config = E0Config(
        num_classes=int(config["num_classes"]),
        embedding_dim=int(config["embedding_dim"]),
        imagenet_weights=str(config["imagenet_weights"]),
    )
    model = build_e0(model_config).to(device)
    if distributed.enabled:
        model = DistributedDataParallel(model, device_ids=[distributed.local_rank], output_device=distributed.local_rank)
    optimizer = torch.optim.AdamW(unwrap_model(model).parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))

    # --- class weighting (effective number of samples) ---
    class_weights = None
    if config.get("class_weight", False):
        class_counts = Counter(record.label for record in train_set.records)
        num_classes = int(config["num_classes"])
        missing = [i for i in range(num_classes) if class_counts.get(i, 0) == 0]
        if missing:
            raise ValueError(f"Classes with zero training samples: {missing}. Cannot compute class weights.")
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

    # --- warmup + scheduler ---
    epochs = int(config["epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 0))
    base_lr = float(config["learning_rate"])

    # --- LR scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)

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

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        set_seed(int(config["seed"]) + distributed.rank + epoch)

        model.train()
        train_loss_sum = 0.0
        seen = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"train_e0 epoch={epoch}", disable=not distributed.is_main)):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            assert_finite_tensor("e0_embedding", output["embedding"])
            assert_finite_tensor("e0_logits", output["logits"])
            loss = criterion(output["logits"], labels)
            _assert_finite_e0_loss(loss, epoch, batch_idx)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * labels.numel()
            seen += labels.numel()

        # --- step scheduler (after warmup phase, cosine takes over) ---
        if epoch >= warmup_epochs:
            scheduler.step()

        # --- reduce training metrics across ranks ---
        train_metrics = reduce_train_metrics(train_loss_sum, seen, device, distributed)

        should_break = False
        if distributed.is_main:
            metrics = evaluate_e0(unwrap_model(model), val_loader, device)
            metrics.update(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "majority_val_accuracy": majority,
                }
            )
            _write_json(out_dir / "last_metrics.json", metrics)
            torch.save(checkpoint_payload(unwrap_model(model), model_config, metrics), out_dir / "last.pt")
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_metrics = metrics
                epochs_without_improvement = 0
                torch.save(checkpoint_payload(unwrap_model(model), model_config, metrics), out_dir / "best.pt")
            else:
                epochs_without_improvement += 1

            # --- early stopping check ---
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}: no improvement for {early_stopping_patience} epochs")
                should_break = True

        # --- broadcast early stopping from main rank ---
        should_break = broadcast_early_stop(should_break, device, distributed)
        barrier(distributed)
        if should_break:
            break

    manifest = {}
    if distributed.is_main:
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
    barrier(distributed)
    cleanup_distributed(distributed)
    return manifest


def _assert_finite_e0_loss(loss, epoch: int, batch_idx: int) -> None:
    loss_val = float(loss.detach().cpu()) if hasattr(loss, "detach") else float(loss)
    if not math.isfinite(loss_val):
        raise RuntimeError(f"non-finite E0 loss detected: {loss_val} at epoch={epoch} batch={batch_idx}")
    assert_finite_tensor("e0_loss", loss)


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
