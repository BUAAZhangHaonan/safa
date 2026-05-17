from __future__ import annotations

from pathlib import Path
import json

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.recognizers import InsightFaceDetector
from safa.models.e0 import assert_e0_frozen, freeze_e0, load_e0_checkpoint
from safa.models.generator import FlowGeneratorConfig, build_generator
from safa.training.audit import audit_no_identity_supervision
from safa.training.losses import cosine_cycle_loss, normalize_for_e0
from safa.training.transforms import generator_image_transform
from safa.utils.device import assert_finite_tensor, require_cuda_device
from safa.utils.seed import set_seed


def train_g_from_config(config: dict) -> dict:
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    set_seed(int(config["seed"]))
    audit_no_identity_supervision(config)
    device = require_cuda_device(str(config["device"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    e0, _ = load_e0_checkpoint(config["e0_checkpoint"], device=str(device))
    e0.to(device)
    freeze_e0(e0)

    generator_config = _generator_config_from_train_config(config)
    generator = build_generator(generator_config.to_dict()).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    assert_e0_frozen(e0, optimizer)

    train_set = FeatureAlignedAffectNet(
        config["train_index"],
        config["train_features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )
    validation_loader = _build_validation_loader(config, train_set)
    detector = _build_detector(config, str(device))
    stages = _stage_config(config)
    lambda_cycle = float(stages["stage2"]["lambda_initial"])
    lambda_max = float(stages["stage2"]["lambda_max"])
    lambda_growth = float(stages["stage2"]["lambda_growth"])
    detection_drop_tolerance = float(config.get("validation", {}).get("face_detection", {}).get("drop_tolerance", 0.02))
    baseline_detection_rate = None
    best_detection_rate = -1.0
    best_checkpoint = out_dir / "best.pt"
    best_detectable_checkpoint = out_dir / "best_detectable.pt"
    history: list[dict] = []
    stage1_stable_hits = 0
    allow_stage2_without_stage1_gate = bool(config.get("allow_stage2_without_stage1_gate", False))

    for stage_name in ("stage1", "stage2"):
        if stage_name == "stage2":
            try:
                _assert_stage1_gate_allows_stage2(stages, stage1_stable_hits, baseline_detection_rate, allow_stage2_without_stage1_gate)
            except RuntimeError as exc:
                final_checkpoint = best_checkpoint if best_checkpoint.is_file() else out_dir / "last.pt"
                _write_json(
                    out_dir / "manifest.json",
                    {
                        "checkpoint": str(final_checkpoint),
                        "metrics": history[-1] if history else {},
                        "history": history,
                        "generator_input": "z_only",
                        "model_type": "conditional_flow_matching",
                        "identity_supervision": False,
                        "blocked": True,
                        "block_reason": str(exc),
                    },
                )
                raise
        epochs = int(stages[stage_name]["epochs"])
        for stage_epoch in range(epochs):
            generator.train()
            totals = {"loss": 0.0, "flow_matching_mse": 0.0, "cycle": 0.0}
            seen = 0
            for batch in tqdm(train_loader, desc=f"train_g {stage_name} epoch={stage_epoch}"):
                images = batch["image"].to(device, non_blocking=True)
                z = batch["z"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                flow_loss, flow_metrics = generator.flow_matching_loss(images, z)
                loss = flow_loss
                cycle = flow_loss.new_tensor(0.0)
                if stage_name == "stage2":
                    generated = generator.sample(z, steps=generator_config.train_cycle_steps)
                    assert_finite_tensor("stage2_generated_image", generated)
                    e0_out = e0(normalize_for_e0(generated))
                    cycle = cosine_cycle_loss(e0_out["embedding"], z)
                    loss = flow_loss + lambda_cycle * cycle
                assert_finite_tensor("g_loss", loss)
                loss.backward()
                optimizer.step()
                batch_size = int(z.shape[0])
                seen += batch_size
                totals["loss"] += float(loss.detach().cpu()) * batch_size
                totals["flow_matching_mse"] += float(flow_metrics["flow_matching_mse"].cpu()) * batch_size
                totals["cycle"] += float(cycle.detach().cpu()) * batch_size

            metrics = {key: value / max(seen, 1) for key, value in totals.items()}
            metrics.update({"stage": stage_name, "stage_epoch": stage_epoch, "lambda_cycle": lambda_cycle})
            validation_metrics = _evaluate_validation(generator, e0, validation_loader, detector, device, generator_config)
            metrics.update({f"validation_{key}": value for key, value in validation_metrics.items()})
            if stage_name == "stage1" and validation_metrics.get("face_detection_rate") is not None:
                baseline_detection_rate = validation_metrics["face_detection_rate"]
                threshold = float(stages["stage1"].get("face_detection_threshold", 0.95))
                stable_epochs = int(stages["stage1"].get("stable_epochs", 1))
                if baseline_detection_rate >= threshold:
                    stage1_stable_hits += 1
                    metrics["stage1_stable_hits"] = stage1_stable_hits
                else:
                    stage1_stable_hits = 0
            if stage_name == "stage2" and baseline_detection_rate is not None:
                current_rate = validation_metrics.get("face_detection_rate")
                if current_rate is not None and current_rate < baseline_detection_rate - detection_drop_tolerance:
                    lambda_cycle = max(float(stages["stage2"]["lambda_initial"]), lambda_cycle * 0.5)
                    metrics["lambda_action"] = "reduced_after_detection_drop"
                    if best_detectable_checkpoint.is_file() and current_rate < best_detection_rate - detection_drop_tolerance:
                        payload = torch.load(best_detectable_checkpoint, map_location=device)
                        generator.load_state_dict(payload["model_state_dict"])
                        metrics["checkpoint_action"] = f"restored:{best_detectable_checkpoint}"
                else:
                    lambda_cycle = min(lambda_max, lambda_cycle + lambda_growth)

            history.append(metrics)
            _save_generator(out_dir / "last.pt", generator, generator_config, config, metrics, history)
            _write_json(out_dir / "last_metrics.json", metrics)
            if _is_better(metrics, history[:-1]):
                _save_generator(best_checkpoint, generator, generator_config, config, metrics, history)
            face_rate = validation_metrics.get("face_detection_rate")
            if face_rate is not None and face_rate >= best_detection_rate:
                best_detection_rate = float(face_rate)
                _save_generator(best_detectable_checkpoint, generator, generator_config, config, metrics, history)
            if stage_name == "stage1" and stage1_stable_hits >= int(stages["stage1"].get("stable_epochs", 1)):
                break
    final_checkpoint = best_checkpoint if best_checkpoint.is_file() else out_dir / "last.pt"
    final_metrics = history[-1] if history else {}
    manifest = {
        "checkpoint": str(final_checkpoint),
        "metrics": final_metrics,
        "history": history,
        "generator_input": "z_only",
        "model_type": "conditional_flow_matching",
        "identity_supervision": False,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _generator_config_from_train_config(config: dict) -> FlowGeneratorConfig:
    model_config = dict(config.get("generator", {}))
    model_config.setdefault("embedding_dim", int(config["embedding_dim"]))
    model_config.setdefault("image_size", int(config["image_size"]))
    return FlowGeneratorConfig.from_dict(model_config)


def _stage_config(config: dict) -> dict:
    stages = config.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("train_g config requires a stages block")
    for name in ("stage1", "stage2"):
        if name not in stages:
            raise ValueError(f"train_g stages missing {name}")
    return stages


def _assert_stage1_gate_allows_stage2(stages: dict, stable_hits: int, detection_rate: float | None, allow_bypass: bool) -> None:
    stage1 = stages["stage1"]
    if allow_bypass:
        return
    if not bool(stage1.get("require_face_detection_gate", True)):
        return
    threshold = float(stage1.get("face_detection_threshold", 0.95))
    stable_epochs = int(stage1.get("stable_epochs", 1))
    if detection_rate is None:
        raise RuntimeError("Stage 2 is blocked because Stage 1 did not produce ArcFace detection metrics")
    if stable_hits < stable_epochs:
        raise RuntimeError(
            "Stage 2 is blocked because Stage 1 face detection gate failed: "
            f"face_detection_rate={detection_rate}, threshold={threshold}, "
            f"stable_hits={stable_hits}, required_stable_epochs={stable_epochs}"
        )


def _build_validation_loader(config: dict, train_set):
    from torch.utils.data import DataLoader, Subset

    validation = config.get("validation", {})
    if not validation.get("enabled", False):
        return None
    val_set = FeatureAlignedAffectNet(
        validation["index"],
        validation["features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
    max_samples = int(validation.get("max_samples", 0))
    if max_samples > 0:
        val_set = Subset(val_set, list(range(min(max_samples, len(val_set)))))
    return DataLoader(
        val_set,
        batch_size=int(validation.get("batch_size", config["batch_size"])),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )


def _build_detector(config: dict, device: str):
    validation = config.get("validation", {})
    detection = validation.get("face_detection", {})
    if not validation.get("enabled", False) or not detection.get("enabled", False):
        return None
    return InsightFaceDetector(model_name=str(detection["model_name"]), device=device)


def _evaluate_validation(generator, e0, loader, detector, device, generator_config: FlowGeneratorConfig) -> dict:
    if loader is None:
        return {}
    import torch
    import torch.nn.functional as F

    generator.eval()
    e0.eval()
    total = 0
    detection_success = 0
    latent_cosine_sum = 0.0
    source_preserved_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            source = batch["image"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            generated = generator.sample(z, steps=generator_config.sample_steps)
            assert_finite_tensor("validation_generated_image", generated)
            source_out = e0(normalize_for_e0(source))
            generated_out = e0(normalize_for_e0(generated))
            cosine = F.cosine_similarity(generated_out["embedding"], z, dim=1)
            latent_cosine_sum += float(cosine.detach().sum().cpu())
            source_preserved_sum += float((generated_out["logits"].argmax(dim=1) == source_out["logits"].argmax(dim=1)).float().sum().cpu())
            if detector is not None:
                counts = detector.detect_counts(generated)
                detection_success += sum(1 for count in counts if count >= 1)
            total += int(z.shape[0])
    if total == 0:
        raise ValueError("Validation monitor received zero samples")
    metrics = {
        "latent_cosine_mean": latent_cosine_sum / total,
        "source_prediction_preserved": source_preserved_sum / total,
    }
    if detector is not None:
        metrics["face_detection_rate"] = detection_success / total
    return metrics


def _is_better(metrics: dict, previous: list[dict]) -> bool:
    if not previous:
        return True
    current_face = metrics.get("validation_face_detection_rate")
    current_cosine = metrics.get("validation_latent_cosine_mean", -1.0)
    best = max(previous, key=lambda item: (item.get("validation_face_detection_rate", -1.0), item.get("validation_latent_cosine_mean", -1.0), -item["loss"]))
    best_face = best.get("validation_face_detection_rate")
    if current_face is not None or best_face is not None:
        return (current_face or -1.0, current_cosine, -metrics["loss"]) > (
            best_face or -1.0,
            best.get("validation_latent_cosine_mean", -1.0),
            -best["loss"],
        )
    return metrics["loss"] < best["loss"]


def _save_generator(path: Path, generator, generator_config: FlowGeneratorConfig, train_config: dict, metrics: dict, history: list[dict]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "model_config": generator_config.to_dict(),
            "sampler_config": {
                "sample_steps": generator_config.sample_steps,
                "train_cycle_steps": generator_config.train_cycle_steps,
                "sampler": generator_config.sampler,
            },
            "stage": metrics.get("stage"),
            "metrics": metrics,
            "history": history,
            "training_config": {
                "stages": train_config.get("stages"),
                "validation": train_config.get("validation"),
            },
        },
        path,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
