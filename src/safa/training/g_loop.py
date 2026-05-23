from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import math
import json
from contextlib import nullcontext
from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.metrics import face_count_rates
from safa.evaluation.recognizers import InsightFaceDetector
from safa.models.e0 import assert_e0_frozen, freeze_e0, load_e0_checkpoint
from safa.models.generator import FlowGeneratorConfig, build_generator
from safa.training.audit import audit_no_identity_supervision
from safa.training.losses import cosine_cycle_loss, normalize_for_e0
from safa.training.transforms import generator_image_transform
from safa.utils.device import assert_finite_tensor
from safa.utils.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    init_distributed,
    unwrap_model,
)
from safa.utils.sampling import make_x_init_for_sample_ids, optional_sampling_base_seed_from_config, sampling_base_seed_from_config
from safa.utils.seed import set_seed


_init_distributed = init_distributed

_SAFA_PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NO_IDENTITY_SOURCE_PATHS = (Path(__file__).resolve().parent, _SAFA_PACKAGE_DIR / "models")


@dataclass(frozen=True)
class _GradientConflictConfig:
    enabled: bool
    interval: int | None = None


class _GeneratorTrainingStep:
    def __new__(cls, generator, e0, generator_config: FlowGeneratorConfig, sampling_seed: int):
        from torch import nn
        schedule = generator_config.cycle_steps_schedule

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.generator = generator
                self.e0 = e0
                self.generator_config = generator_config
                self.sampling_seed = int(sampling_seed)
                self._schedule = schedule
                self._batch_idx = 0

            def reset_batch_idx(self):
                self._batch_idx = 0

            def forward(self, images, z, sample_ids, use_cycle: bool, lambda_cycle: float):
                import torch
                flow_loss, flow_metrics = self.generator.flow_matching_loss(images, z)
                cycle_loss = flow_loss.new_tensor(0.0)
                loss = flow_loss
                if use_cycle:
                    if self._schedule:
                        cycle_steps = self._schedule[self._batch_idx % len(self._schedule)]
                    else:
                        cycle_steps = self.generator_config.train_cycle_steps
                    x_init = make_x_init_for_sample_ids(
                        sample_ids,
                        self.sampling_seed,
                        self.generator_config.image_size,
                        z.device,
                        z.dtype,
                    )
                    generated = self.generator.sample(
                        z,
                        steps=cycle_steps,
                        checkpoint_steps=True,
                        x_init=x_init,
                        clamp_output=False,
                    )
                    assert_finite_tensor("stage2_generated_image", generated)
                    self.e0.eval()
                    e0_out = self.e0(normalize_for_e0(generated))
                    cycle_loss = cosine_cycle_loss(e0_out["embedding"], z)
                    loss = flow_loss + float(lambda_cycle) * cycle_loss
                    self._batch_idx += 1
                return loss, flow_metrics["flow_matching_mse"].detach(), cycle_loss.detach(), flow_loss, cycle_loss

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
    audit_no_identity_supervision(config, DEFAULT_NO_IDENTITY_SOURCE_PATHS)
    _validate_train_g_config(config)
    distributed = init_distributed(config)
    device = distributed.device
    num_workers = int(config["num_workers"])
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1 for persistent_workers, got {num_workers}")
    out_dir = Path(config["out_dir"])
    if distributed.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Optional: absent amp means full precision.
    use_amp = bool(config.get("amp", False))
    if distributed.is_main:
        amp_status = "enabled" if use_amp else "disabled"
        print(f"AMP (bfloat16): {amp_status}")
    barrier(distributed)

    e0, _ = load_e0_checkpoint(config["e0_checkpoint"], device=str(device))
    e0.to(device)
    freeze_e0(e0)

    generator_config = _generator_config_from_train_config(config)
    sampling_seed = sampling_base_seed_from_config(config)
    generator = build_generator(generator_config.to_dict()).to(device)
    resume_history = None
    resume_stage_epoch = None
    # Optional: absent resume_from starts a fresh generator run.
    if config.get("resume_from"):
        resume_path = Path(config["resume_from"])
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(ckpt["model_state_dict"])
        if "history" in ckpt:
            resume_history = ckpt["history"]
        if "metrics" in ckpt and "stage_epoch" in ckpt["metrics"]:
            resume_stage_epoch = ckpt["metrics"]["stage_epoch"]
        if distributed.is_main:
            restored = ["model_state_dict"]
            if resume_history is not None:
                restored.append("history")
            if resume_stage_epoch is not None:
                restored.append("stage_epoch")
            sep = ", ".join(restored)
            print(f"Resumed generator from {resume_path} (restored: {sep})")
    training_module = _GeneratorTrainingStep(generator, e0, generator_config, sampling_seed).to(device)
    if distributed.enabled:
        training_module = DistributedDataParallel(training_module, device_ids=[distributed.local_rank], output_device=distributed.local_rank)
    optimizer = torch.optim.AdamW(unwrap_model(training_module).generator.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
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
    gradient_conflict_config = _stage2_gradient_conflict_config(stages)
    lambda_cycle = float(stages["stage2"]["lambda_initial"])
    lambda_max = float(stages["stage2"]["lambda_max"])
    lambda_growth = float(stages["stage2"]["lambda_growth"])
    baseline_detection_rate = None
    best_checkpoint = out_dir / "best.pt"
    history: list[dict] = resume_history if resume_history is not None else []
    stage1_stable_hits = 0
    allow_stage2_without_stage1_gate = _require_bool(config, "allow_stage2_without_stage1_gate", "train_g config")

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
                cleanup_distributed(distributed)
                raise RuntimeError("Stage 2 is blocked by the Stage 1 face detection gate; see manifest.json on rank 0")
        epochs = int(stages[stage_name]["epochs"])
        stage_epoch = -1
        for stage_epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(total_epoch + stage_epoch)
            training_module.train()
            unwrap_model(training_module).reset_batch_idx()
            e0.eval()
            totals = {
                "loss": 0.0,
                "flow_matching_mse": 0.0,
                "cycle": 0.0,
                "grad_norm": 0.0,
                "gradient_conflict_count": 0.0,
                "gradient_cosine_fm_cycle": 0.0,
                "gradient_norm_fm": 0.0,
                "gradient_norm_cycle": 0.0,
            }
            seen = 0
            for batch_index, batch in enumerate(tqdm(train_loader, desc=f"train_g {stage_name} epoch={stage_epoch}", disable=not distributed.is_main)):
                images = batch["image"].to(device, non_blocking=True)
                z = batch["z"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
                sample_ids = list(batch["sample_id"])
                with amp_ctx:
                    loss, flow_mse, cycle, flow_loss, cycle_loss = training_module(images, z, sample_ids, stage_name == "stage2", lambda_cycle)
                _assert_finite_training_scalars(loss, flow_mse, cycle)
                if _should_record_gradient_conflict(stage_name, batch_index, gradient_conflict_config):
                    gradient_metrics = _compute_gradient_conflict_metrics(
                        flow_loss,
                        cycle_loss,
                        unwrap_model(training_module).generator.parameters(),
                    )
                    totals["gradient_conflict_count"] += 1.0
                    totals["gradient_cosine_fm_cycle"] += gradient_metrics["gradient_cosine_fm_cycle"]
                    totals["gradient_norm_fm"] += gradient_metrics["gradient_norm_fm"]
                    totals["gradient_norm_cycle"] += gradient_metrics["gradient_norm_cycle"]
                assert_finite_tensor("g_loss", loss)
                loss.backward()
                batch_grad_norm = 0.0
                if "grad_clip_norm" in config:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        unwrap_model(training_module).generator.parameters(),
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
                validation_metrics = _evaluate_validation(unwrap_model(training_module).generator, e0, validation_loader, detector, device, generator_config, sampling_seed=sampling_seed, use_amp=use_amp)
                metrics.update({f"validation_{key}": value for key, value in validation_metrics.items()})
                if stage_name == "stage1" and validation_metrics.get("face_detection_rate") is not None:
                    baseline_detection_rate = validation_metrics["face_detection_rate"]
                    threshold = float(stages["stage1"]["face_detection_threshold"])
                    stable_epochs = int(stages["stage1"]["stable_epochs"])
                    if baseline_detection_rate >= threshold:
                        stage1_stable_hits += 1
                        metrics["stage1_stable_hits"] = stage1_stable_hits
                    else:
                        stage1_stable_hits = 0
                if stage_name == "stage2":
                    next_lambda = min(lambda_max, lambda_cycle + lambda_growth)
                    metrics["next_lambda_cycle"] = next_lambda
                    lambda_cycle = next_lambda

                _validate_checkpoint_selection_metrics(metrics)
                history.append(metrics)
                _save_generator(out_dir / "last.pt", unwrap_model(training_module).generator, generator_config, config, metrics, history)
                _write_json(out_dir / "last_metrics.json", metrics)
                stage_best_path = out_dir / f"best_{stage_name}.pt"
                if _is_better(metrics, history[:-1]):
                    _save_generator(stage_best_path, unwrap_model(training_module).generator, generator_config, config, metrics, history)
                if _is_better_overall(metrics, history[:-1]):
                    _save_generator(best_checkpoint, unwrap_model(training_module).generator, generator_config, config, metrics, history)
                should_break = stage_name == "stage1" and stage1_stable_hits >= int(stages["stage1"]["stable_epochs"])
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
        total_epoch += stage_epoch + 1

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
            "sampling": {"base_seed": sampling_seed, "stable_x_init": True},
        }
        _write_json(out_dir / "manifest.json", manifest)
    barrier(distributed)
    cleanup_distributed(distributed)
    return manifest


def _reduce_epoch_metrics(totals: dict, seen: int, device, distributed: DistributedContext) -> dict:
    import torch

    values = torch.tensor(
        [
            float(totals["loss"]),
            float(totals["flow_matching_mse"]),
            float(totals["cycle"]),
            float(totals["grad_norm"]),
            float(seen),
            # These totals are present only when the Stage 2 gradient-conflict monitor records a batch.
            float(totals.get("gradient_conflict_count", 0.0)),
            float(totals.get("gradient_cosine_fm_cycle", 0.0)),
            float(totals.get("gradient_norm_fm", 0.0)),
            float(totals.get("gradient_norm_cycle", 0.0)),
        ],
        device=device,
        dtype=torch.float64,
    )
    if distributed.enabled:
        import torch.distributed as dist

        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    metric_names = (
        "loss",
        "flow_matching_mse",
        "cycle",
        "grad_norm",
        "seen",
        "gradient_conflict_count",
        "gradient_cosine_fm_cycle",
        "gradient_norm_fm",
        "gradient_norm_cycle",
    )
    for index, name in enumerate(metric_names):
        if not bool(torch.isfinite(values[index]).item()):
            raise RuntimeError(f"Epoch metric {name} is not finite")
    total_seen = float(values[4].item())
    if total_seen <= 0.0:
        raise RuntimeError("Cannot reduce epoch metrics from zero samples")
    metrics = {
        "loss": float(values[0].item() / total_seen),
        "flow_matching_mse": float(values[1].item() / total_seen),
        "cycle": float(values[2].item() / total_seen),
        "grad_norm": float(values[3].item() / total_seen),
    }
    gradient_conflict_count = float(values[5].item())
    if gradient_conflict_count > 0.0:
        metrics.update(
            {
                "gradient_cosine_fm_cycle": float(values[6].item() / gradient_conflict_count),
                "gradient_norm_fm": float(values[7].item() / gradient_conflict_count),
                "gradient_norm_cycle": float(values[8].item() / gradient_conflict_count),
                "gradient_conflict_count": int(gradient_conflict_count),
            }
        )
    return metrics


def _assert_finite_training_scalars(loss, flow_mse, cycle) -> None:
    import torch

    for name, value in (("loss", loss), ("flow_matching_mse", flow_mse), ("cycle", cycle)):
        if hasattr(value, "detach"):
            finite = bool(torch.isfinite(value.detach()).all().item())
        else:
            finite = math.isfinite(float(value))
        if not finite:
            raise RuntimeError(f"{name} is not finite")


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


def _validate_train_g_config(config: dict) -> None:
    _generator_config_from_train_config(config)
    _require_bool(config, "allow_stage2_without_stage1_gate", "train_g config")
    stages = _stage_config(config)
    _validate_stage1_gate_config(stages["stage1"])
    _stage2_gradient_conflict_config(stages)
    _validate_validation_block(config)
    if int(_require_field(stages["stage2"], "epochs", "stages.stage2")) > 0:
        _validate_stage2_validation_config(config)


def _require_mapping(config: dict, field: str, context: str) -> dict:
    if field not in config:
        raise ValueError(f"{context}.{field} is required")
    value = config[field]
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{field} must be a mapping")
    return value


def _require_field(config: dict, field: str, context: str):
    if field not in config:
        raise ValueError(f"{context}.{field} is required")
    return config[field]


def _require_bool(config: dict, field: str, context: str) -> bool:
    value = _require_field(config, field, context)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{field} must be true or false")
    return value


def _validate_stage1_gate_config(stage1: dict) -> None:
    if not _require_bool(stage1, "require_face_detection_gate", "stages.stage1"):
        return
    _require_field(stage1, "face_detection_threshold", "stages.stage1")
    _require_field(stage1, "stable_epochs", "stages.stage1")


def _validate_validation_block(config: dict) -> dict:
    validation = _require_mapping(config, "validation", "train_g config")
    _require_bool(validation, "enabled", "validation")
    detection = _require_mapping(validation, "face_detection", "validation")
    if _require_bool(detection, "enabled", "validation.face_detection"):
        _require_field(detection, "model_name", "validation.face_detection")
    return validation


def _validate_stage2_validation_config(config: dict) -> None:
    validation = _validate_validation_block(config)
    if not _require_bool(validation, "enabled", "validation"):
        raise ValueError("validation.enabled must be true when Stage 2 epochs > 0")
    for field in ("index", "features", "max_samples", "batch_size"):
        _require_field(validation, field, "validation")
    detection = _require_mapping(validation, "face_detection", "validation")
    if not _require_bool(detection, "enabled", "validation.face_detection"):
        raise ValueError("validation.face_detection.enabled must be true when Stage 2 epochs > 0")
    _require_field(detection, "model_name", "validation.face_detection")


def _generator_config_from_train_config(config: dict) -> FlowGeneratorConfig:
    model_config = dict(_require_mapping(config, "generator", "train_g config"))
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
        if not isinstance(stages[name], dict):
            raise ValueError(f"train_g stages.{name} must be a mapping")
    return stages


def _stage2_gradient_conflict_config(stages: dict) -> _GradientConflictConfig:
    stage2 = stages["stage2"]
    epochs = int(_require_field(stage2, "epochs", "stages.stage2"))
    payload = stage2.get("gradient_conflict")
    if payload is None:
        if epochs <= 0:
            return _GradientConflictConfig(enabled=False)
        raise ValueError("stages.stage2.gradient_conflict is required when Stage 2 epochs > 0")
    if not isinstance(payload, dict):
        raise ValueError("stages.stage2.gradient_conflict must be a mapping")
    if "enabled" not in payload:
        raise ValueError("stages.stage2.gradient_conflict.enabled is required")
    enabled = payload["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("stages.stage2.gradient_conflict.enabled must be true or false")
    if not enabled:
        if "interval" in payload:
            _validate_gradient_conflict_interval(payload["interval"])
        return _GradientConflictConfig(enabled=False)
    if "interval" not in payload:
        raise ValueError("stages.stage2.gradient_conflict.interval is required when enabled")
    return _GradientConflictConfig(enabled=True, interval=_validate_gradient_conflict_interval(payload["interval"]))


def _validate_gradient_conflict_interval(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"stages.stage2.gradient_conflict.interval must be a positive integer, got {value!r}")
    return int(value)


def _should_record_gradient_conflict(stage_name: str, batch_index: int, config: _GradientConflictConfig) -> bool:
    if stage_name != "stage2" or not config.enabled:
        return False
    if config.interval is None:
        raise RuntimeError("Stage 2 gradient conflict monitor is enabled without an interval")
    return batch_index % config.interval == 0


def _compute_gradient_conflict_metrics(flow_loss, cycle_loss, parameters) -> dict[str, float]:
    import torch

    params = [param for param in parameters if param.requires_grad]
    if not params:
        raise RuntimeError("Cannot compute gradient conflict metrics without trainable generator parameters")
    flow_gradient = _gradient_vector_for_loss("flow matching", flow_loss, params)
    cycle_gradient = _gradient_vector_for_loss("cycle", cycle_loss, params)
    flow_norm = torch.linalg.vector_norm(flow_gradient)
    cycle_norm = torch.linalg.vector_norm(cycle_gradient)
    if not torch.isfinite(flow_norm):
        raise RuntimeError("flow matching gradient norm is not finite")
    if not torch.isfinite(cycle_norm):
        raise RuntimeError("cycle gradient norm is not finite")
    if float(flow_norm.detach().cpu()) <= 0.0:
        raise RuntimeError("flow matching gradient has zero norm")
    if float(cycle_norm.detach().cpu()) <= 0.0:
        raise RuntimeError("cycle gradient has zero norm")
    cosine = torch.dot(flow_gradient, cycle_gradient) / (flow_norm * cycle_norm)
    if not torch.isfinite(cosine):
        raise RuntimeError("gradient cosine between flow matching and cycle losses is not finite")
    return {
        "gradient_cosine_fm_cycle": float(cosine.detach().cpu()),
        "gradient_norm_fm": float(flow_norm.detach().cpu()),
        "gradient_norm_cycle": float(cycle_norm.detach().cpu()),
    }


def _gradient_vector_for_loss(name: str, loss, params) -> object:
    import torch

    if not hasattr(loss, "requires_grad") or not loss.requires_grad:
        raise RuntimeError(f"{name} loss is not connected to a gradient graph")
    if not torch.isfinite(loss.detach()).all():
        raise RuntimeError(f"{name} loss is not finite")
    gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    chunks = []
    has_gradient = False
    for param, gradient in zip(params, gradients):
        if gradient is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float64))
            continue
        has_gradient = True
        flat = gradient.detach().reshape(-1).to(dtype=torch.float64)
        if not torch.isfinite(flat).all():
            raise RuntimeError(f"{name} gradient contains non-finite values")
        chunks.append(flat)
    if not has_gradient:
        raise RuntimeError(f"No valid gradient for {name} loss")
    vector = torch.cat(chunks)
    if vector.numel() == 0:
        raise RuntimeError(f"No valid gradient entries for {name} loss")
    return vector


def _assert_stage1_gate_allows_stage2(stages: dict, stable_hits: int, detection_rate: float | None, allow_bypass: bool) -> None:
    stage1 = stages["stage1"]
    if allow_bypass:
        return
    if not _require_bool(stage1, "require_face_detection_gate", "stages.stage1"):
        return
    _validate_stage1_gate_config(stage1)
    threshold = float(stage1["face_detection_threshold"])
    stable_epochs = int(stage1["stable_epochs"])
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

    validation = _require_mapping(config, "validation", "train_g config")
    if not _require_bool(validation, "enabled", "validation"):
        return None
    val_set = FeatureAlignedAffectNet(
        validation["index"],
        validation["features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
    max_samples = int(validation["max_samples"])
    if max_samples > 0:
        val_set = Subset(val_set, list(range(min(max_samples, len(val_set)))))
    return DataLoader(
        val_set,
        batch_size=int(validation["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


def _build_detector(config: dict, device: str):
    validation = _require_mapping(config, "validation", "train_g config")
    detection = _require_mapping(validation, "face_detection", "validation")
    if not _require_bool(validation, "enabled", "validation") or not _require_bool(detection, "enabled", "validation.face_detection"):
        return None
    return InsightFaceDetector(model_name=str(detection["model_name"]), device=device)


def _evaluate_validation(generator, e0, loader, detector, device, generator_config: FlowGeneratorConfig, *, sampling_seed: int, use_amp: bool = False) -> dict:
    if loader is None:
        return {}
    import torch
    import torch.nn.functional as F

    generator.eval()
    e0.eval()
    total = 0
    detected_counts = []
    latent_cosine_sum = 0.0
    source_preserved_sum = 0.0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.no_grad(), amp_ctx:
        for batch in loader:
            source = batch["image"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            sample_ids = list(batch["sample_id"])
            x_init = make_x_init_for_sample_ids(sample_ids, sampling_seed, generator_config.image_size, z.device, z.dtype)
            generated = generator.sample(z, steps=generator_config.sample_steps, x_init=x_init)
            assert_finite_tensor("validation_generated_image", generated)
            source_out = e0(normalize_for_e0(source))
            generated_out = e0(normalize_for_e0(generated))
            cosine = F.cosine_similarity(generated_out["embedding"], z, dim=1)
            latent_cosine_sum += float(cosine.detach().sum().cpu())
            source_preserved_sum += float((generated_out["logits"].argmax(dim=1) == source_out["logits"].argmax(dim=1)).float().sum().cpu())
            if detector is not None:
                counts = detector.detect_counts(generated)
                if len(counts) != int(z.shape[0]):
                    raise RuntimeError(f"Validation face detection count mismatch: batch={int(z.shape[0])} counts={len(counts)}")
                detected_counts.extend(counts)
            total += int(z.shape[0])
    if total == 0:
        raise ValueError("Validation monitor received zero samples")
    metrics = {
        "latent_cosine_mean": latent_cosine_sum / total,
        "source_prediction_preserved": source_preserved_sum / total,
    }
    if detector is not None:
        metrics.update(face_count_rates(detected_counts))
        metrics["face_detection_rate"] = metrics["face_detect_ge1_rate"]
    return metrics


def _composite_score(item: dict) -> float:
    """New checkpoint composite: cosine x single_face_eq1_rate.

    Old reports used validation_face_detection_rate, which is the ge1 rate.
    """
    cosine = item["validation_latent_cosine_mean"]
    single_face = item["validation_single_face_eq1_rate"]
    return cosine * single_face


def _validate_checkpoint_selection_metrics(metrics: dict, context: str = "checkpoint metrics") -> None:
    for field in ("validation_latent_cosine_mean", "validation_single_face_eq1_rate", "loss", "stage"):
        _require_field(metrics, field, context)
    for field in ("validation_latent_cosine_mean", "validation_single_face_eq1_rate", "loss"):
        value = metrics[field]
        if isinstance(value, bool):
            raise ValueError(f"{context}.{field} must be numeric, got bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.{field} must be numeric, got {value!r}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{context}.{field} must be finite, got {value!r}")


def _is_better(metrics: dict, previous: list[dict]) -> bool:
    _validate_checkpoint_selection_metrics(metrics)
    current_score = _composite_score(metrics)
    if not previous:
        return True
    stage = metrics["stage"]
    same_stage = []
    for item in previous:
        _validate_checkpoint_selection_metrics(item, "checkpoint history item")
        if item["stage"] == stage:
            same_stage.append(item)
    if not same_stage:
        return True
    best = max(same_stage, key=lambda item: (_composite_score(item), -item["loss"]))
    return (current_score, -metrics["loss"]) > (_composite_score(best), -best["loss"])


def _is_better_overall(metrics: dict, previous: list[dict]) -> bool:
    """Compare current epoch against ALL previous epochs regardless of stage.

    Unlike _is_better which only compares within the same stage, this function
    compares across stages. Uses composite score (cosine x single_face_eq1_rate)
    to prevent selecting degenerate checkpoints with high cosine but invalid face counts.
    """
    _validate_checkpoint_selection_metrics(metrics)
    current_score = _composite_score(metrics)
    if not previous:
        return True
    for item in previous:
        _validate_checkpoint_selection_metrics(item, "checkpoint history item")
    best = max(previous, key=lambda item: (_composite_score(item), -item["loss"]))
    return (current_score, -metrics["loss"]) > (_composite_score(best), -best["loss"])


def _save_generator(path: Path, generator, generator_config: FlowGeneratorConfig, train_config: dict, metrics: dict, history: list[dict]) -> None:
    import torch

    _validate_checkpoint_selection_metrics(metrics)
    generator = unwrap_model(generator)
    path.parent.mkdir(parents=True, exist_ok=True)
    training_config = {
        "stages": train_config.get("stages"),
        "validation": train_config.get("validation"),
    }
    if "seed" in train_config and train_config["seed"] is not None:
        training_config["seed"] = train_config["seed"]
    if "sampling_seed" in train_config and train_config["sampling_seed"] is not None:
        training_config["sampling_seed"] = train_config["sampling_seed"]
    payload = {
        "model_state_dict": generator.state_dict(),
        "model_config": generator_config.to_dict(),
        "sampler_config": {
            "sample_steps": generator_config.sample_steps,
            "train_cycle_steps": generator_config.train_cycle_steps,
            "sampler": generator_config.sampler,
        },
        "stage": metrics["stage"],
        "metrics": metrics,
        "history": history,
        "training_config": training_config,
    }
    sampling_seed = optional_sampling_base_seed_from_config(train_config)
    if sampling_seed is not None:
        payload["sampling"] = {"base_seed": sampling_seed, "stable_x_init": True}
    torch.save(payload, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
