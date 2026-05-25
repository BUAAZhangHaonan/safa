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
from safa.utils.config import require_keys


REQUIRED_E0_TRAIN_KEYS = (
    "seed",
    "device",
    "num_workers",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "num_classes",
    "embedding_dim",
    "image_size",
    "imagenet_weights",
    "train_index",
    "val_index",
    "out_dir",
    "warmup_epochs",
    "early_stopping_patience",
    "augmentation",
    "class_weight",
    "label_smoothing",
)


def require_e0_train_config(config: dict) -> None:
    require_keys(config, REQUIRED_E0_TRAIN_KEYS)
    augmentation = str(config["augmentation"])
    if augmentation not in {"default", "strong"}:
        raise ValueError(f"augmentation must be 'default' or 'strong', got {augmentation!r}")
    if not isinstance(config["class_weight"], bool):
        raise ValueError("class_weight must be true or false")
    epochs = int(config["epochs"])
    warmup_epochs = int(config["warmup_epochs"])
    early_stopping_patience = int(config["early_stopping_patience"])
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
    if warmup_epochs > epochs:
        raise ValueError(f"warmup_epochs must be <= epochs, got {warmup_epochs} > {epochs}")
    if early_stopping_patience < 0:
        raise ValueError(f"early_stopping_patience must be non-negative, got {early_stopping_patience}")
    if int(config["num_classes"]) <= 0:
        raise ValueError(f"num_classes must be positive, got {config['num_classes']}")
    if int(config["embedding_dim"]) <= 0:
        raise ValueError(f"embedding_dim must be positive, got {config['embedding_dim']}")


def train_e0_from_config(config: dict) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm

    require_e0_train_config(config)
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
    augmentation = str(config["augmentation"])
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
    if config["class_weight"]:
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
    label_smoothing = float(config["label_smoothing"])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # --- warmup + scheduler ---
    epochs = int(config["epochs"])
    warmup_epochs = int(config["warmup_epochs"])
    base_lr = float(config["learning_rate"])

    # --- LR scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)

    # --- early stopping ---
    early_stopping_patience = int(config["early_stopping_patience"])

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
            metrics = evaluate_e0(unwrap_model(model), val_loader, device, num_classes=model_config.num_classes)
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


def evaluate_e0(model, loader, device, num_classes: int | None = None) -> dict:
    import torch

    model.eval()
    correct = 0
    total = 0
    logits_abs_sum = 0.0
    all_preds = []
    all_labels = []
    all_norms = []
    parsed_num_classes = int(num_classes if num_classes is not None else getattr(model, "num_classes"))
    if parsed_num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {parsed_num_classes}")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            output = model(images)
            assert_finite_tensor("eval_e0_embedding", output["embedding"])
            if output["logits"].shape[1] != parsed_num_classes:
                raise RuntimeError(
                    "E0 logits class dimension does not match num_classes: "
                    f"logits={output['logits'].shape[1]} num_classes={parsed_num_classes}"
                )
            norms = output["embedding"].float().norm(dim=1)
            all_norms.append(norms.detach().cpu())
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
    invalid_labels = sorted(
        int(value) for value in all_labels[(all_labels < 0) | (all_labels >= parsed_num_classes)].unique().tolist()
    )
    if invalid_labels:
        raise ValueError(f"Validation labels out of range for num_classes={parsed_num_classes}: {invalid_labels}")
    invalid_preds = sorted(
        int(value) for value in all_preds[(all_preds < 0) | (all_preds >= parsed_num_classes)].unique().tolist()
    )
    if invalid_preds:
        raise RuntimeError(f"E0 predictions out of range for num_classes={parsed_num_classes}: {invalid_preds}")
    confusion = torch.zeros((parsed_num_classes, parsed_num_classes), dtype=torch.int64)
    for label, pred in zip(all_labels.tolist(), all_preds.tolist()):
        confusion[int(label), int(pred)] += 1

    per_class_acc = {}
    per_class_support = {}
    per_class_correct = {}
    per_class_f1 = {}
    recalls = []
    f1_values = []
    for cls in range(parsed_num_classes):
        key = f"class_{cls}"
        cls_correct = int(confusion[cls, cls].item())
        cls_total = int(confusion[cls].sum().item())
        cls_predicted = int(confusion[:, cls].sum().item())
        per_class_support[key] = cls_total
        per_class_correct[key] = cls_correct
        if cls_total == 0:
            per_class_acc[key] = None
            per_class_f1[key] = None
            print(f"  {key}: accuracy=undefined (0 validation samples)")
            continue
        recall = cls_correct / cls_total
        precision = cls_correct / cls_predicted if cls_predicted > 0 else 0.0
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_class_acc[key] = recall
        per_class_f1[key] = f1
        recalls.append(recall)
        f1_values.append(f1)
        print(f"  {key}: accuracy={recall:.4f} ({cls_correct}/{cls_total})")

    norm_values = torch.cat(all_norms)
    norm_deviation = (norm_values - 1.0).abs()
    embedding_norm_check = {
        "passed": True,
        "mean": float(norm_values.mean().item()),
        "min": float(norm_values.min().item()),
        "max": float(norm_values.max().item()),
        "max_abs_deviation": float(norm_deviation.max().item()),
        "rtol": 1e-4,
        "atol": 1e-4,
    }

    result = {
        "accuracy": correct / total,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix": [[int(value) for value in row] for row in confusion.tolist()],
        "embedding_norm_check": embedding_norm_check,
        "num_samples": total,
        "mean_abs_logit": logits_abs_sum / total,
        "per_class_accuracy": per_class_acc,
        "per_class_accuracy_note": "null means the class has zero validation samples",
        "per_class_support": per_class_support,
        "per_class_correct": per_class_correct,
        "per_class_f1": per_class_f1,
        "metric_averaging": {
            "balanced_accuracy": "mean recall over classes with validation support",
            "macro_f1": "mean F1 over classes with validation support",
        },
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
