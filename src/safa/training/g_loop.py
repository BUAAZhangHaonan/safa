from __future__ import annotations

from pathlib import Path
import math
import json
from contextlib import nullcontext
from dataclasses import dataclass

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.recognizers import InsightFaceDetector
from safa.models.e0 import assert_e0_frozen, freeze_e0, load_e0_checkpoint
from safa.models.generator import FlowGeneratorConfig, build_generator
from safa.training.audit import audit_no_identity_supervision
from safa.training.losses import cosine_cycle_loss, normalize_for_e0
from safa.training.transforms import generator_image_transform
from safa.utils.device import assert_finite_tensor, require_cuda_device
from safa.utils.seed import set_seed


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    device: object
    backend: str


class _GeneratorTrainingStep:
    def __new__(cls, generator, e0, generator_config: FlowGeneratorConfig):
        from torch import nn
        schedule = generator_config.cycle_steps_schedule

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.generator = generator
                self.e0 = e0
                self.generator_config = generator_config
                self._schedule = schedule
                self._batch_idx = 0

            def forward(self, images, z, use_cycle: bool, lambda_cycle: float):
                import torch
                flow_loss, flow_metrics = self.generator.flow_matching_loss(images, z)
                cycle = flow_loss.new_tensor(0.0)
                loss = flow_loss
                if use_cycle:
                    if self._schedule:
                        cycle_steps = self._schedule[self._batch_idx % len(self._schedule)]
                    else:
                        cycle_steps = self.generator_config.train_cycle_steps
                    generated = self.generator.sample(
                        z,
                        steps=cycle_steps,
                        checkpoint_steps=True,
                    )
                    assert_finite_tensor("stage2_generated_image", generated)
                    self.e0.eval()
                    e0_out = self.e0(normalize_for_e0(generated))
                    cycle = cosine_cycle_loss(e0_out["embedding"], z)
                    loss = flow_loss + float(lambda_cycle) * cycle
                    self._batch_idx += 1
                return loss, flow_metrics["flow_matching_mse"].detach(), cycle.detach()

        return _Module()


def _verify_e0_feature_cache_consistency(config: dict) -> None:
    from safa.data.feature_cache import load_manifest
    from safa.utils.hashing import sha256_file

    e0_path = config["e0_checkpoint"]
    feature_dir = config["train_features"]
    manifest = load_manifest(feature_dir)
    actual_sha256 = sha256_file(e0_path)
    if manifest.encoder_checkpoint_sha256 != actual_sha256:
        raise RuntimeError(
            f"SHA256 mismatch between E0 checkpoint and feature cache manifest. "
            f"E0 checkpoint: {e0_path} (sha256={actual_sha256}) "
            f"Feature cache manifest expects: {manifest.encoder_checkpoint_sha256}. "
            f"Regenerate the feature cache with the current E0 checkpoint."
        )


def train_g_from_config(config: dict) -> dict:
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm

    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    audit_no_identity_supervision(config)
    distributed = _init_distributed(config)
    device = distributed.device
    num_workers = int(config["num_workers"])
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1 for persistent_workers, got {num_workers}")
    out_dir = Path(config["out_dir"])
    if distributed.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(config.get("amp", False))
    if distributed.is_main:
        print(f"AMP (bfloat16): {"enabled" if use_amp else "disabled"}")
    _barrier(distributed)

    e0, _ = load_e0_checkpoint(config["e0_checkpoint"], device=str(device))
    e0.to(device)
    freeze_e0(e0)

    generator_config = _generator_config_from_train_config(config)
    generator = build_generator(generator_config.to_dict()).to(device)
    if config.get("resume_from"):
        resume_path = Path(config["resume_from"])
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(ckpt["model_state_dict"])
        if distributed.is_main:
            print(f"Resumed generator from {resume_path}")
    training_module = _GeneratorTrainingStep(generator, e0, generator_config).to(device)
    if distributed.enabled:
        training_module = DistributedDataParallel(training_module, device_ids=[distributed.local_rank], output_device=distributed.local_rank)
    optimizer = torch.optim.AdamW(_unwrap_model(training_module).generator.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    assert_e0_frozen(e0, optimizer)
    set_seed(int(config["seed"]) + distributed.rank)

    _verify_e0_feature_cache_consistency(config)
    train_set = FeatureAlignedAffectNet(
        config["train_index"],
        config["train_features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
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
    validation_loader = _build_validation_loader(config) if distributed.is_main else None
    detector = _build_detector(config, str(device)) if distributed.is_main else None
    stages = _stage_config(config)
    lambda_cycle = float(stages["stage2"]["lambda_initial"])
    lambda_max = float(stages["stage2"]["lambda_max"])
    lambda_growth = float(stages["stage2"]["lambda_growth"])
    baseline_detection_rate = None
    best_checkpoint = out_dir / "best.pt"
    history: list[dict] = []
    stage1_stable_hits = 0
    allow_stage2_without_stage1_gate = bool(config.get("allow_stage2_without_stage1_gate", False))

    total_epoch = 0
    for stage_name in ("stage1", "stage2"):
        if stage_name == "stage2":
            blocked = _stage2_blocked(
                distributed,
                device,
                stages,
                stage1_stable_hits,
                baseline_detection_rate,
                allow_stage2_without_stage1_gate,
                out_dir,
                best_checkpoint,
                history,
            )
            if blocked:
                _cleanup_distributed(distributed)
                raise RuntimeError("Stage 2 is blocked by the Stage 1 face detection gate; see manifest.json on rank 0")
        epochs = int(stages[stage_name]["epochs"])
        for stage_epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(total_epoch + stage_epoch)
            training_module.train()
            e0.eval()
            totals = {"loss": 0.0, "flow_matching_mse": 0.0, "cycle": 0.0, "grad_norm": 0.0}
            seen = 0
            for batch in tqdm(train_loader, desc=f"train_g {stage_name} epoch={stage_epoch}", disable=not distributed.is_main):
                images = batch["image"].to(device, non_blocking=True)
                z = batch["z"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
                with amp_ctx:
                    loss, flow_mse, cycle = training_module(images, z, stage_name == "stage2", lambda_cycle)
                loss_val = float(loss.detach().cpu())
                if not math.isfinite(loss_val):
                    print(f"WARNING: non-finite G loss detected: {loss_val}, skipping batch entirely")
                    dummy = sum(p.sum() for p in _unwrap_model(training_module).generator.parameters())
                    (0.0 * dummy).backward()
                    optimizer.step()
                    continue
                assert_finite_tensor("g_loss", loss)
                loss.backward()
                batch_grad_norm = 0.0
                if "grad_clip_norm" in config:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        _unwrap_model(training_module).generator.parameters(),
                        config["grad_clip_norm"],
                    )
                    batch_grad_norm = float(grad_norm) if isinstance(grad_norm, float) else float(grad_norm.detach().cpu())
                optimizer.step()
                batch_size = int(z.shape[0])
                seen += batch_size
                totals["loss"] += float(loss.detach().cpu()) * batch_size
                totals["flow_matching_mse"] += float(flow_mse.cpu()) * batch_size
                totals["cycle"] += float(cycle.detach().cpu()) * batch_size
                totals["grad_norm"] += batch_grad_norm * batch_size

            metrics = _reduce_epoch_metrics(totals, seen, device, distributed)
            should_break = False
            if distributed.is_main:
                metrics.update({"stage": stage_name, "stage_epoch": stage_epoch, "lambda_cycle": lambda_cycle})
                validation_metrics = _evaluate_validation(_unwrap_model(training_module).generator, e0, validation_loader, detector, device, generator_config, use_amp=use_amp)
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
                if stage_name == "stage2":
                    next_lambda = min(lambda_max, lambda_cycle + lambda_growth)
                    metrics["next_lambda_cycle"] = next_lambda
                    lambda_cycle = next_lambda

                history.append(metrics)
                _save_generator(out_dir / "last.pt", _unwrap_model(training_module).generator, generator_config, config, metrics, history)
                _write_json(out_dir / "last_metrics.json", metrics)
                stage_best_path = out_dir / f"best_{stage_name}.pt"
                if _is_better(metrics, history[:-1]):
                    _save_generator(stage_best_path, _unwrap_model(training_module).generator, generator_config, config, metrics, history)
                if _is_better_overall(metrics, history[:-1]):
                    _save_generator(best_checkpoint, _unwrap_model(training_module).generator, generator_config, config, metrics, history)
                should_break = stage_name == "stage1" and stage1_stable_hits >= int(stages["stage1"].get("stable_epochs", 1))
            lambda_cycle, baseline_detection_rate, stage1_stable_hits, should_break = _sync_epoch_control(
                lambda_cycle,
                baseline_detection_rate,
                stage1_stable_hits,
                should_break,
                device,
                distributed,
            )
            if should_break:
                break
        total_epoch += epochs

    manifest = {}
    if distributed.is_main:
        final_checkpoint = best_checkpoint if best_checkpoint.is_file() else out_dir / "last.pt"
        final_metrics = history[-1] if history else {}
        manifest = {
            "checkpoint": str(final_checkpoint),
            "metrics": final_metrics,
            "history": history,
            "generator_input": "z_only",
            "model_type": "conditional_flow_matching",
            "identity_supervision": False,
            "distributed": _distributed_manifest(distributed),
        }
        _write_json(out_dir / "manifest.json", manifest)
    _barrier(distributed)
    _cleanup_distributed(distributed)
    return manifest


def _init_distributed(config: dict) -> DistributedContext:
    import os
    import torch
    import torch.distributed as dist

    world_size_raw = os.environ.get("WORLD_SIZE")
    rank_raw = os.environ.get("RANK")
    local_rank_raw = os.environ.get("LOCAL_RANK")
    world_size = int(world_size_raw) if world_size_raw else 1
    if world_size > 1:
        if rank_raw is None or local_rank_raw is None:
            raise RuntimeError("DDP requires RANK and LOCAL_RANK when WORLD_SIZE > 1")
        rank = int(rank_raw)
        local_rank = int(local_rank_raw)
        backend = str(config.get("distributed", {}).get("backend", "nccl"))
        if backend not in {"nccl", "gloo"}:
            raise ValueError(f"Unsupported DDP backend: {backend}")
        device = require_cuda_device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            if backend == "nccl":
                dist.init_process_group(backend=backend, device_id=device)
            else:
                dist.init_process_group(backend=backend)
        return DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_main=rank == 0,
            device=device,
            backend=backend,
        )
    device = require_cuda_device(str(config["device"]))
    if device.index is not None:
        torch.cuda.set_device(device)
    return DistributedContext(
        enabled=False,
        rank=0,
        local_rank=device.index or 0,
        world_size=1,
        is_main=True,
        device=device,
        backend="single",
    )


def _cleanup_distributed(distributed: DistributedContext) -> None:
    if not distributed.enabled:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(distributed: DistributedContext) -> None:
    if not distributed.enabled:
        return
    import torch.distributed as dist

    if distributed.backend == "nccl":
        dist.barrier(device_ids=[distributed.local_rank])
    else:
        dist.barrier()


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _reduce_epoch_metrics(totals: dict, seen: int, device, distributed: DistributedContext) -> dict:
    import torch

    values = torch.tensor(
        [
            float(totals["loss"]),
            float(totals["flow_matching_mse"]),
            float(totals["cycle"]),
            float(totals["grad_norm"]),
            float(seen),
        ],
        device=device,
        dtype=torch.float64,
    )
    if distributed.enabled:
        import torch.distributed as dist

        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_seen = max(float(values[4].item()), 1.0)
    return {
        "loss": float(values[0].item() / total_seen),
        "flow_matching_mse": float(values[1].item() / total_seen),
        "cycle": float(values[2].item() / total_seen),
        "grad_norm": float(values[3].item() / total_seen),
    }


def _sync_epoch_control(
    lambda_cycle: float,
    baseline_detection_rate: float | None,
    stage1_stable_hits: int,
    should_break: bool,
    device,
    distributed: DistributedContext,
) -> tuple[float, float | None, int, bool]:
    if not distributed.enabled:
        return lambda_cycle, baseline_detection_rate, stage1_stable_hits, should_break
    import torch
    import torch.distributed as dist

    payload = torch.tensor(
        [
            float(lambda_cycle),
            -1.0 if baseline_detection_rate is None else float(baseline_detection_rate),
            float(stage1_stable_hits),
            1.0 if should_break else 0.0,
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.broadcast(payload, src=0)
    synced_baseline = float(payload[1].item())
    return (
        float(payload[0].item()),
        None if synced_baseline < 0.0 else synced_baseline,
        int(payload[2].item()),
        bool(int(payload[3].item())),
    )


def _stage2_blocked(
    distributed: DistributedContext,
    device,
    stages: dict,
    stage1_stable_hits: int,
    baseline_detection_rate: float | None,
    allow_stage2_without_stage1_gate: bool,
    out_dir: Path,
    best_checkpoint: Path,
    history: list[dict],
) -> bool:
    import torch

    blocked = False
    if distributed.is_main:
        try:
            _assert_stage1_gate_allows_stage2(
                stages,
                stage1_stable_hits,
                baseline_detection_rate,
                allow_stage2_without_stage1_gate,
            )
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
                    "distributed": _distributed_manifest(distributed),
                },
            )
            blocked = True
    if distributed.enabled:
        import torch.distributed as dist

        flag = torch.tensor([1 if blocked else 0], device=device, dtype=torch.int64)
        dist.broadcast(flag, src=0)
        return bool(flag.item())
    return blocked


def _distributed_manifest(distributed: DistributedContext) -> dict:
    return {
        "enabled": distributed.enabled,
        "world_size": distributed.world_size,
        "backend": distributed.backend,
    }


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


def _build_validation_loader(config: dict):
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
        persistent_workers=True,
        prefetch_factor=4,
    )


def _build_detector(config: dict, device: str):
    validation = config.get("validation", {})
    detection = validation.get("face_detection", {})
    if not validation.get("enabled", False) or not detection.get("enabled", False):
        return None
    return InsightFaceDetector(model_name=str(detection["model_name"]), device=device)


def _evaluate_validation(generator, e0, loader, detector, device, generator_config: FlowGeneratorConfig, *, use_amp: bool = False) -> dict:
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
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.no_grad(), amp_ctx:
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


def _composite_score(item: dict) -> float:
    """cosine x face_detection_rate. Penalizes degenerate checkpoints
    that achieve high cosine by generating non-face outputs."""
    cosine = item.get("validation_latent_cosine_mean", -1.0)
    face_det = item.get("validation_face_detection_rate", 1.0)
    return cosine * face_det


def _is_better(metrics: dict, previous: list[dict]) -> bool:
    if not previous:
        return True
    stage = metrics.get("stage", "stage1")
    same_stage = [m for m in previous if m.get("stage") == stage]
    if not same_stage:
        return True
    current_score = _composite_score(metrics)
    best = max(same_stage, key=lambda item: (_composite_score(item), -item["loss"]))
    return (current_score, -metrics["loss"]) > (_composite_score(best), -best["loss"])


def _is_better_overall(metrics: dict, previous: list[dict]) -> bool:
    """Compare current epoch against ALL previous epochs regardless of stage.

    Unlike _is_better which only compares within the same stage, this function
    compares across stages. Uses composite score (cosine x face_det) to prevent
    selecting degenerate checkpoints with high cosine but zero face quality.
    """
    if not previous:
        return True
    current_score = _composite_score(metrics)
    best = max(previous, key=lambda item: (_composite_score(item), -item["loss"]))
    return (current_score, -metrics["loss"]) > (_composite_score(best), -best["loss"])

def _save_generator(path: Path, generator, generator_config: FlowGeneratorConfig, train_config: dict, metrics: dict, history: list[dict]) -> None:
    import torch

    generator = _unwrap_model(generator)
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
