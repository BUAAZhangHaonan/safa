from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import math
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.metrics import face_count_rates
from safa.evaluation.recognizers import InsightFaceDetector
from safa.models.e0 import assert_e0_frozen, freeze_e0, load_e0_checkpoint
from safa.models.generator import FlowGeneratorConfig, build_generator
from safa.training.audit import audit_no_identity_supervision
from safa.training.losses import cosine_cycle_loss, normalize_for_e0
from safa.training.multitask_loss import UncertaintyWeightedLoss
from safa.training.transforms import generator_image_transform
from safa.utils.device import assert_finite_tensor
from safa.utils.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    init_distributed,
    unwrap_model,
)
from safa.utils.ema import ExponentialMovingAverage
from safa.utils.sampling import make_x_init_for_sample_ids, optional_sampling_base_seed_from_config, sampling_base_seed_from_config
from safa.utils.seed import set_seed


_init_distributed = init_distributed

_SAFA_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NO_IDENTITY_SOURCE_PATHS = (Path(__file__).resolve().parent, _SAFA_PACKAGE_DIR / "models")


@dataclass(frozen=True)
class _GradientConflictConfig:
    enabled: bool
    interval: int | None = None


@dataclass(frozen=True)
class _LossWeightingRuntime:
    type: str
    flow_weight: float = 1.0
    cycle_weight: float = 0.0
    calibration_batches: int | None = None
    log_var_lr: float | None = None
    log_var_weight_decay: float | None = None


@dataclass(frozen=True)
class _QualityEvalGroup:
    name: str
    metrics: tuple[str, ...]
    max_samples: int


@dataclass(frozen=True)
class _ResumeProgress:
    stage: str
    stage_epoch: int


class _GeneratorTrainingStep:
    def __new__(
        cls,
        generator,
        e0,
        generator_config: FlowGeneratorConfig,
        sampling_seed: int,
        loss_weighting: _LossWeightingRuntime | None = None,
    ):
        from torch import nn
        schedule = generator_config.cycle_steps_schedule

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.generator = generator
                self.e0 = e0
                self.generator_config = generator_config
                self.sampling_seed = int(sampling_seed)
                self.loss_weighting = loss_weighting if loss_weighting is not None else _LossWeightingRuntime(type="legacy")
                self.uncertainty_loss = UncertaintyWeightedLoss(["flow", "cycle"]) if self.loss_weighting.type == "uncertainty" else None
                self._schedule = schedule
                self._batch_idx = 0
                self.last_loss_metrics: dict[str, float | str] = {}
                import torch

                self.register_buffer("_flow_loss_initial", torch.tensor(float("nan"), dtype=torch.float64), persistent=True)
                self.register_buffer("_cycle_loss_initial", torch.tensor(float("nan"), dtype=torch.float64), persistent=True)

            def reset_batch_idx(self):
                self._batch_idx = 0

            def configure_uncertainty_scales(self, *, flow_loss_initial: float, cycle_loss_initial: float) -> None:
                if self.loss_weighting.type != "uncertainty":
                    raise RuntimeError("configure_uncertainty_scales requires loss_weighting.type == 'uncertainty'")
                _ensure_positive_finite_scale("flow_loss_initial", flow_loss_initial)
                _ensure_positive_finite_scale("cycle_loss_initial", cycle_loss_initial)
                self._flow_loss_initial.fill_(float(flow_loss_initial))
                self._cycle_loss_initial.fill_(float(cycle_loss_initial))

            def loss_weighting_checkpoint_state(self) -> dict:
                state = {
                    "type": self.loss_weighting.type,
                    "flow_weight": self.loss_weighting.flow_weight,
                    "cycle_weight": self.loss_weighting.cycle_weight,
                }
                if self.loss_weighting.type == "uncertainty":
                    if self.uncertainty_loss is None:
                        raise RuntimeError("uncertainty loss state requested before uncertainty_loss is initialized")
                    state.update(
                        {
                            "task_names": list(self.uncertainty_loss.task_names),
                            "initial_scales": {
                                "flow": float(self._flow_loss_initial.detach().cpu()),
                                "cycle": float(self._cycle_loss_initial.detach().cpu()),
                            },
                            "state_dict": self.uncertainty_loss.state_dict(),
                        }
                    )
                return state

            def forward(self, images, z, sample_ids, use_cycle: bool, lambda_cycle: float):
                import torch
                flow_loss, flow_metrics = self.generator.flow_matching_loss(images, z)
                cycle_loss = flow_loss.new_tensor(0.0)
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
                    self._batch_idx += 1
                loss, loss_metrics = self._combine_losses(flow_loss, cycle_loss, use_cycle=use_cycle, lambda_cycle=lambda_cycle)
                self.last_loss_metrics = loss_metrics
                return loss, flow_metrics["flow_matching_mse"].detach(), cycle_loss.detach(), flow_loss, cycle_loss

            def _combine_losses(self, flow_loss, cycle_loss, *, use_cycle: bool, lambda_cycle: float):
                metrics = {
                    "flow_loss_raw": float(flow_loss.detach().cpu()),
                    "cycle_loss_raw": float(cycle_loss.detach().cpu()),
                    "loss_weighting_type": self.loss_weighting.type,
                }
                if self.loss_weighting.type == "legacy":
                    cycle_weight = float(lambda_cycle) if use_cycle else 0.0
                    loss = flow_loss + cycle_weight * cycle_loss
                    metrics.update(
                        {
                            "flow_loss_normalized": metrics["flow_loss_raw"],
                            "cycle_loss_normalized": metrics["cycle_loss_raw"],
                            "loss_weighting_flow_weight": 1.0,
                            "loss_weighting_cycle_weight": cycle_weight,
                            "effective_cycle_loss_weight": cycle_weight,
                        }
                    )
                    return loss, metrics
                if self.loss_weighting.type == "fixed":
                    flow_weight = float(self.loss_weighting.flow_weight)
                    cycle_weight = float(self.loss_weighting.cycle_weight) if use_cycle else 0.0
                    loss = flow_weight * flow_loss + cycle_weight * cycle_loss
                    metrics.update(
                        {
                            "flow_loss_normalized": metrics["flow_loss_raw"],
                            "cycle_loss_normalized": metrics["cycle_loss_raw"],
                            "loss_weighting_flow_weight": flow_weight,
                            "loss_weighting_cycle_weight": cycle_weight,
                            "effective_cycle_loss_weight": cycle_weight,
                        }
                    )
                    return loss, metrics
                if self.loss_weighting.type != "uncertainty":
                    raise RuntimeError(f"Unsupported loss_weighting.type {self.loss_weighting.type!r}")
                if not use_cycle:
                    raise RuntimeError("loss_weighting.type='uncertainty' requires cycle loss and can only run on Stage 2 batches")
                if self.uncertainty_loss is None:
                    raise RuntimeError("uncertainty_loss is not initialized")
                flow_scale = float(self._flow_loss_initial.detach().cpu())
                cycle_scale = float(self._cycle_loss_initial.detach().cpu())
                _ensure_positive_finite_scale("flow_loss_initial", flow_scale)
                _ensure_positive_finite_scale("cycle_loss_initial", cycle_scale)
                normalized = {"flow": flow_loss / flow_scale, "cycle": cycle_loss / cycle_scale}
                loss, uw_metrics = self.uncertainty_loss(normalized)
                metrics.update(uw_metrics)
                metrics.update(
                    {
                        "flow_loss_initial": flow_scale,
                        "cycle_loss_initial": cycle_scale,
                        "flow_loss_normalized": metrics["loss_weighting_uw_flow_normalized"],
                        "cycle_loss_normalized": metrics["loss_weighting_uw_cycle_normalized"],
                        "loss_weighting_flow_weight": 0.5 * metrics["loss_weighting_uw_flow_precision"] / flow_scale,
                        "loss_weighting_cycle_weight": 0.5 * metrics["loss_weighting_uw_cycle_precision"] / cycle_scale,
                        "effective_cycle_loss_weight": 0.5 * metrics["loss_weighting_uw_cycle_precision"] / cycle_scale,
                    }
                )
                return loss, metrics

        return _Module()


def _ensure_positive_finite_scale(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise RuntimeError(f"{name} must be positive and finite, got {value!r}")


def _finalize_uncertainty_calibration(*, flow_sum: float, cycle_sum: float, batches: int) -> dict[str, float]:
    if batches <= 0:
        raise RuntimeError(f"calibration_batches produced no batches: {batches}")
    flow_loss_initial = float(flow_sum) / float(batches)
    cycle_loss_initial = float(cycle_sum) / float(batches)
    _ensure_positive_finite_scale("flow_loss_initial", flow_loss_initial)
    _ensure_positive_finite_scale("cycle_loss_initial", cycle_loss_initial)
    return {"flow_loss_initial": flow_loss_initial, "cycle_loss_initial": cycle_loss_initial}


def _loss_weighting_runtime_from_config(config: dict) -> _LossWeightingRuntime:
    payload = config.get("loss_weighting")
    if payload is None:
        return _LossWeightingRuntime(type="legacy")
    if not isinstance(payload, dict):
        raise ValueError("loss_weighting must be a mapping")
    loss_type = _require_field(payload, "type", "loss_weighting")
    if loss_type == "fixed":
        flow_weight = _require_numeric(payload, "flow_weight", "loss_weighting")
        cycle_weight = _require_numeric(payload, "cycle_weight", "loss_weighting")
        if flow_weight < 0.0:
            raise ValueError(f"loss_weighting.flow_weight must be non-negative, got {flow_weight!r}")
        if cycle_weight < 0.0:
            raise ValueError(f"loss_weighting.cycle_weight must be non-negative, got {cycle_weight!r}")
        return _LossWeightingRuntime(type="fixed", flow_weight=flow_weight, cycle_weight=cycle_weight)
    if loss_type == "uncertainty":
        calibration_batches = _require_positive_int(payload, "calibration_batches", "loss_weighting")
        log_var_lr = _require_numeric(payload, "log_var_lr", "loss_weighting")
        log_var_weight_decay = _require_numeric(payload, "log_var_weight_decay", "loss_weighting")
        if log_var_lr <= 0.0:
            raise ValueError(f"loss_weighting.log_var_lr must be positive, got {log_var_lr!r}")
        if log_var_weight_decay < 0.0:
            raise ValueError(f"loss_weighting.log_var_weight_decay must be non-negative, got {log_var_weight_decay!r}")
        return _LossWeightingRuntime(
            type="uncertainty",
            calibration_batches=calibration_batches,
            log_var_lr=log_var_lr,
            log_var_weight_decay=log_var_weight_decay,
        )
    raise ValueError(f"loss_weighting.type must be 'fixed' or 'uncertainty', got {loss_type!r}")


def _require_positive_int(config: dict, field: str, context: str) -> int:
    value = _require_field(config, field, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{field} must be a positive integer, got {value!r}")
    return int(value)


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


def _optimizer_param_groups(training_module, config: dict, loss_weighting: _LossWeightingRuntime) -> list[dict]:
    groups = [
        {
            "params": training_module.generator.parameters(),
            "lr": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
        }
    ]
    if loss_weighting.type == "uncertainty":
        if training_module.uncertainty_loss is None:
            raise RuntimeError("loss_weighting.type='uncertainty' requires uncertainty_loss parameters")
        groups.append(
            {
                "params": training_module.uncertainty_loss.parameters(),
                "lr": float(loss_weighting.log_var_lr),
                "weight_decay": float(loss_weighting.log_var_weight_decay),
            }
        )
    return groups


def _stage2_lambda_schedule(stages: dict, loss_weighting: _LossWeightingRuntime) -> tuple[float, float, float]:
    stage2 = stages["stage2"]
    if loss_weighting.type == "legacy":
        return (
            float(_require_numeric(stage2, "lambda_initial", "stages.stage2")),
            float(_require_numeric(stage2, "lambda_max", "stages.stage2")),
            float(_require_numeric(stage2, "lambda_growth", "stages.stage2")),
        )
    if loss_weighting.type == "fixed":
        cycle_weight = float(loss_weighting.cycle_weight)
        return cycle_weight, cycle_weight, 0.0
    return 0.0, 0.0, 0.0


def _calibrate_uncertainty_loss(
    training_module,
    train_loader,
    device,
    *,
    use_amp: bool,
    calibration_batches: int,
    distributed: DistributedContext,
) -> None:
    import torch

    if calibration_batches <= 0:
        raise RuntimeError(f"loss_weighting.calibration_batches must be positive, got {calibration_batches!r}")
    was_training = training_module.training
    training_module.eval()
    training_module.reset_batch_idx()
    training_module.configure_uncertainty_scales(flow_loss_initial=1.0, cycle_loss_initial=1.0)
    flow_sum = 0.0
    cycle_sum = 0.0
    batches = 0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.no_grad():
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= calibration_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            sample_ids = list(batch["sample_id"])
            with amp_ctx:
                _, _, _, flow_loss, cycle_loss = training_module(images, z, sample_ids, True, 0.0)
            flow_value = float(flow_loss.detach().cpu())
            cycle_value = float(cycle_loss.detach().cpu())
            _ensure_positive_finite_scale("flow_loss_initial", flow_value)
            _ensure_positive_finite_scale("cycle_loss_initial", cycle_value)
            flow_sum += flow_value
            cycle_sum += cycle_value
            batches += 1
    if distributed.enabled:
        import torch.distributed as dist

        values = torch.tensor([flow_sum, cycle_sum, float(batches)], device=device, dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        flow_sum = float(values[0].item())
        cycle_sum = float(values[1].item())
        batches = int(values[2].item())
    scales = _finalize_uncertainty_calibration(flow_sum=flow_sum, cycle_sum=cycle_sum, batches=batches)
    training_module.configure_uncertainty_scales(**scales)
    training_module.reset_batch_idx()
    if was_training:
        training_module.train()


def _restore_uncertainty_loss_checkpoint_state(training_module, state: dict, checkpoint_path: str) -> None:
    context = f"loss_weighting_state in {checkpoint_path}"
    if not isinstance(state, dict):
        raise RuntimeError(f"{context} must be a mapping")
    state_type = state.get("type")
    if state_type != "uncertainty":
        raise RuntimeError(f"{context}.type must be 'uncertainty', got {state_type!r}")
    if getattr(training_module, "uncertainty_loss", None) is None:
        raise RuntimeError("Cannot restore uncertainty loss state because uncertainty_loss is not initialized")
    expected_tasks = list(training_module.uncertainty_loss.task_names)
    task_names = state.get("task_names")
    if task_names != expected_tasks:
        raise RuntimeError(f"{context}.task_names must be {expected_tasks!r}, got {task_names!r}")
    initial_scales = state.get("initial_scales")
    if not isinstance(initial_scales, dict):
        raise RuntimeError(f"{context}.initial_scales must be a mapping")
    state_dict = state.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"{context}.state_dict must be a mapping")
    training_module.configure_uncertainty_scales(
        flow_loss_initial=float(_require_field(initial_scales, "flow", context + ".initial_scales")),
        cycle_loss_initial=float(_require_field(initial_scales, "cycle", context + ".initial_scales")),
    )
    try:
        training_module.uncertainty_loss.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(f"{context}.state_dict is invalid: {exc}") from exc


def _restore_or_calibrate_uncertainty_loss(
    training_module,
    train_loader,
    device,
    *,
    use_amp: bool,
    calibration_batches: int,
    distributed: DistributedContext,
    resume_progress: _ResumeProgress | None,
    resume_loss_weighting_state: dict | None,
    resume_path: str | None,
) -> str:
    if resume_loss_weighting_state is not None:
        if not isinstance(resume_loss_weighting_state, dict):
            raise RuntimeError(f"loss_weighting_state in {resume_path} must be a mapping")
        if resume_loss_weighting_state.get("type") == "uncertainty":
            _restore_uncertainty_loss_checkpoint_state(training_module, resume_loss_weighting_state, str(resume_path or "<fresh run>"))
            return "restored"
    if resume_progress is not None and resume_progress.stage == "stage2":
        raise RuntimeError(
            "Stage 2 uncertainty resume checkpoint is missing required UW loss_weighting_state; "
            f"refusing to recalibrate UW scales: {resume_path}"
        )
    _calibrate_uncertainty_loss(
        training_module,
        train_loader,
        device,
        use_amp=use_amp,
        calibration_batches=calibration_batches,
        distributed=distributed,
    )
    return "calibrated"


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
    stages = _stage_config(config)
    ema_config = _ema_config(config)
    best_model = _best_model(config, ema_config)
    loss_weighting_runtime = _loss_weighting_runtime_from_config(config)
    sampling_seed = sampling_base_seed_from_config(config)
    generator = build_generator(generator_config.to_dict()).to(device)
    resume_history = None
    resume_progress = None
    resume_ema_state_dict = None
    resume_optimizer_state_dict = None
    resume_loss_weighting_state = None
    # Optional: absent resume_from starts a fresh generator run.
    if config.get("resume_from"):
        resume_path = Path(config["resume_from"])
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(ckpt["model_state_dict"])
        if "history" in ckpt:
            resume_history = _resume_history_for_checkpoint_selection(ckpt["history"], str(resume_path), config, stages)
        resume_progress = _resume_stage_progress_from_metrics(ckpt.get("metrics"), str(resume_path))
        if "ema_model_state_dict" in ckpt:
            resume_ema_state_dict = ckpt["ema_model_state_dict"]
        if "optimizer_state_dict" in ckpt:
            resume_optimizer_state_dict = ckpt["optimizer_state_dict"]
        if "loss_weighting_state" in ckpt:
            resume_loss_weighting_state = ckpt["loss_weighting_state"]
        if distributed.is_main:
            restored = ["model_state_dict"]
            if resume_history is not None:
                restored.append("history")
            restored.append(f"progress={resume_progress.stage}:{resume_progress.stage_epoch}")
            if resume_ema_state_dict is not None:
                restored.append("ema_model_state_dict")
            if resume_loss_weighting_state is not None:
                restored.append("loss_weighting_state")
            sep = ", ".join(restored)
            print(f"Resumed generator from {resume_path} (restored: {sep})")
    ema = None
    if ema_config["enabled"]:
        ema = ExponentialMovingAverage(generator, decay=float(ema_config["decay"]))
        if resume_ema_state_dict is not None:
            ema.load_state_dict(resume_ema_state_dict)
    training_module = _GeneratorTrainingStep(generator, e0, generator_config, sampling_seed, loss_weighting_runtime).to(device)
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
    gradient_conflict_config = _stage2_gradient_conflict_config(stages)
    if loss_weighting_runtime.type == "uncertainty":
        uw_state_action = _restore_or_calibrate_uncertainty_loss(
            training_module,
            train_loader,
            device,
            use_amp=use_amp,
            calibration_batches=int(loss_weighting_runtime.calibration_batches),
            distributed=distributed,
            resume_progress=resume_progress,
            resume_loss_weighting_state=resume_loss_weighting_state,
            resume_path=str(config.get("resume_from")) if config.get("resume_from") else None,
        )
        if distributed.is_main:
            print(f"Uncertainty loss state: {uw_state_action}")
    if distributed.enabled:
        training_module = DistributedDataParallel(training_module, device_ids=[distributed.local_rank], output_device=distributed.local_rank)
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(unwrap_model(training_module), config, loss_weighting_runtime),
    )
    optimizer_resumed = False
    if config.get("resume_from"):
        if resume_optimizer_state_dict is None:
            if distributed.is_main:
                print("Resume checkpoint has no optimizer_state_dict; optimizer_resumed: false")
        else:
            optimizer.load_state_dict(resume_optimizer_state_dict)
            optimizer_resumed = True
            if distributed.is_main:
                print("Resumed optimizer state from checkpoint; optimizer_resumed: true")
    assert_e0_frozen(e0, optimizer)
    lambda_cycle, lambda_max, lambda_growth = _stage2_lambda_schedule(stages, loss_weighting_runtime)
    baseline_detection_rate = None
    best_checkpoint = out_dir / "best.pt"
    history: list[dict] = resume_history if resume_history is not None else []
    stage1_stable_hits = 0
    allow_stage2_without_stage1_gate = _require_bool(config, "allow_stage2_without_stage1_gate", "train_g config")

    total_epoch = 0
    for stage_name in ("stage1", "stage2"):
        if _should_check_stage2_gate(stage_name, resume_progress):
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
        start_stage_epoch = _resume_stage_start_epoch(stage_name, stages, resume_progress)
        stage_epoch = start_stage_epoch - 1
        completed_stage_epochs = start_stage_epoch
        for stage_epoch in range(start_stage_epoch, epochs):
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
                "weighted_gradient_norm_cycle": 0.0,
                "weighted_gradient_ratio_cycle_to_fm": 0.0,
                "gradient_conflict_samples": [],
                "extra_metric_sums": {},
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
                batch_size = int(z.shape[0])
                _accumulate_extra_loss_metrics(totals, unwrap_model(training_module).last_loss_metrics, batch_size)
                if _should_record_gradient_conflict(stage_name, batch_index, gradient_conflict_config):
                    gradient_cycle_weight = float(
                        unwrap_model(training_module).last_loss_metrics.get("loss_weighting_cycle_weight", lambda_cycle)
                    )
                    gradient_metrics = _compute_gradient_conflict_metrics(
                        flow_loss,
                        cycle_loss,
                        unwrap_model(training_module).generator.parameters(),
                        lambda_cycle=gradient_cycle_weight,
                    )
                    totals["gradient_conflict_count"] += 1.0
                    totals["gradient_cosine_fm_cycle"] += gradient_metrics["gradient_cosine_fm_cycle"]
                    totals["gradient_norm_fm"] += gradient_metrics["gradient_norm_fm"]
                    totals["gradient_norm_cycle"] += gradient_metrics["gradient_norm_cycle"]
                    totals["weighted_gradient_norm_cycle"] += gradient_metrics["weighted_gradient_norm_cycle"]
                    totals["weighted_gradient_ratio_cycle_to_fm"] += gradient_metrics["weighted_gradient_ratio_cycle_to_fm"]
                    totals["gradient_conflict_samples"].append(gradient_metrics)
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
                if ema is not None:
                    ema.update(unwrap_model(training_module).generator)
                seen += batch_size
                totals["loss"] += float(loss.detach().cpu()) * batch_size
                totals["flow_matching_mse"] += float(flow_mse.cpu()) * batch_size
                totals["cycle"] += float(cycle.detach().cpu()) * batch_size
                totals["grad_norm"] += batch_grad_norm * batch_size

            metrics = _reduce_epoch_metrics(totals, seen, device, distributed)
            should_break = False
            if distributed.is_main:
                effective_cycle_loss_weight = _finite_metric_value(metrics, "effective_cycle_loss_weight", "epoch metrics")
                metrics.update(
                    {
                        "stage": stage_name,
                        "stage_epoch": stage_epoch,
                        "stage_epoch_0based": stage_epoch,
                        "stage_epoch_1based": stage_epoch + 1,
                        "lambda_cycle": effective_cycle_loss_weight,
                        "effective_cycle_loss_weight": effective_cycle_loss_weight,
                        "loss_weighting_type": loss_weighting_runtime.type,
                    }
                )
                if config.get("resume_from"):
                    metrics["optimizer_resumed"] = optimizer_resumed
                if loss_weighting_runtime.type == "uncertainty":
                    metrics["lambda_cycle_legacy_schedule"] = lambda_cycle
                raw_validation_metrics, ema_validation_metrics = _evaluate_validation_variants(
                    unwrap_model(training_module).generator,
                    ema,
                    e0,
                    validation_loader,
                    detector,
                    device,
                    generator_config,
                    sampling_seed=sampling_seed,
                    use_amp=use_amp,
                    ema_config=ema_config,
                )
                _attach_validation_metrics(metrics, raw_validation_metrics, ema_validation_metrics)
                metrics.update(
                    _run_quality_eval_hook(
                        config,
                        stage_name,
                        stage_epoch,
                        generator=unwrap_model(training_module).generator,
                        ema=ema,
                        device=device,
                        generator_config=generator_config,
                        sampling_seed=sampling_seed,
                        use_amp=use_amp,
                        ema_config=ema_config,
                    )
                )
                if stage_name == "stage1" and raw_validation_metrics is not None and raw_validation_metrics.get("face_detect_ge1_rate") is not None:
                    baseline_detection_rate = raw_validation_metrics["face_detect_ge1_rate"]
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

                _validate_checkpoint_selection_metrics(metrics, best_model=best_model)
                history.append(metrics)
                checkpoint_kwargs = {
                    "ema_model_state_dict": ema.state_dict() if ema is not None and ema_config["save_ema_checkpoint"] else None,
                    "metrics_raw": raw_validation_metrics,
                    "metrics_ema": ema_validation_metrics,
                    "ema_config": ema_config,
                    "best_model": best_model,
                    "loss_weighting_state": unwrap_model(training_module).loss_weighting_checkpoint_state(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                _save_generator(out_dir / "last.pt", unwrap_model(training_module).generator, generator_config, config, metrics, history, **checkpoint_kwargs)
                _write_json(out_dir / "last_metrics.json", metrics)
                stage_best_path = out_dir / f"best_{stage_name}.pt"
                if _is_better(metrics, history[:-1], best_model=best_model):
                    _save_generator(stage_best_path, unwrap_model(training_module).generator, generator_config, config, metrics, history, **checkpoint_kwargs)
                if stage_name == "stage1":
                    for filename in _stage1_single_face_checkpoint_filenames_to_save(metrics, history[:-1]):
                        _save_generator(out_dir / filename, unwrap_model(training_module).generator, generator_config, config, metrics, history, **checkpoint_kwargs)
                if _is_better_overall(metrics, history[:-1], best_model=best_model):
                    _save_generator(best_checkpoint, unwrap_model(training_module).generator, generator_config, config, metrics, history, **checkpoint_kwargs)
                if stage_name == "stage2":
                    for filename in _stage2_checkpoint_filenames_to_save(metrics, history[:-1]):
                        _save_generator(out_dir / filename, unwrap_model(training_module).generator, generator_config, config, metrics, history, **checkpoint_kwargs)
                should_break = stage_name == "stage1" and stage1_stable_hits >= int(stages["stage1"]["stable_epochs"])
            lambda_cycle, baseline_detection_rate, stage1_stable_hits, should_break = _sync_epoch_control(
                lambda_cycle,
                baseline_detection_rate,
                stage1_stable_hits,
                should_break,
                device,
                distributed,
            )
            completed_stage_epochs = stage_epoch + 1
            if should_break:
                break
        total_epoch += completed_stage_epochs

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
            "ema_config": ema_config,
            "best_model": best_model,
            "resume_from_legacy_stage1_metrics": bool(config.get("resume_from_legacy_stage1_metrics", False)),
        }
        _write_json(out_dir / "manifest.json", manifest)
    barrier(distributed)
    cleanup_distributed(distributed)
    return manifest


def _reduce_epoch_metrics(totals: dict, seen: int, device, distributed: DistributedContext) -> dict:
    import torch

    local_gradient_conflict_count = float(totals.get("gradient_conflict_count", 0.0))
    weighted_gradient_norm_cycle = 0.0
    weighted_gradient_ratio_cycle_to_fm = 0.0
    if local_gradient_conflict_count > 0.0:
        weighted_gradient_norm_cycle = float(totals["weighted_gradient_norm_cycle"])
        weighted_gradient_ratio_cycle_to_fm = float(totals["weighted_gradient_ratio_cycle_to_fm"])
    values = torch.tensor(
        [
            float(totals["loss"]),
            float(totals["flow_matching_mse"]),
            float(totals["cycle"]),
            float(totals["grad_norm"]),
            float(seen),
            # These totals are present only when the Stage 2 gradient-conflict monitor records a batch.
            local_gradient_conflict_count,
            float(totals.get("gradient_cosine_fm_cycle", 0.0)),
            float(totals.get("gradient_norm_fm", 0.0)),
            float(totals.get("gradient_norm_cycle", 0.0)),
            weighted_gradient_norm_cycle,
            weighted_gradient_ratio_cycle_to_fm,
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
        "weighted_gradient_norm_cycle",
        "weighted_gradient_ratio_cycle_to_fm",
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
    metrics.update(_reduce_extra_epoch_metrics(totals.get("extra_metric_sums", {}), total_seen, device, distributed))
    gradient_conflict_count = float(values[5].item())
    if gradient_conflict_count > 0.0:
        gradient_samples = _gather_gradient_conflict_samples(totals.get("gradient_conflict_samples", []), distributed)
        if len(gradient_samples) != int(gradient_conflict_count):
            raise RuntimeError(
                "Gradient conflict monitor sample count mismatch: "
                f"samples={len(gradient_samples)} count={int(gradient_conflict_count)}"
            )
        metrics.update(
            {
                "gradient_cosine_fm_cycle": float(values[6].item() / gradient_conflict_count),
                "gradient_norm_fm": float(values[7].item() / gradient_conflict_count),
                "gradient_norm_cycle": float(values[8].item() / gradient_conflict_count),
                "weighted_gradient_norm_cycle": float(values[9].item() / gradient_conflict_count),
                "weighted_gradient_ratio_cycle_to_fm": float(values[10].item() / gradient_conflict_count),
                "gradient_conflict_count": int(gradient_conflict_count),
                **_summarize_gradient_conflict_samples(gradient_samples),
            }
        )
    return metrics


def _accumulate_extra_loss_metrics(totals: dict, metrics: dict, batch_size: int) -> None:
    extra = totals.setdefault("extra_metric_sums", {})
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError(f"Loss metric {key} is not finite")
        extra[key] = float(extra.get(key, 0.0)) + numeric * batch_size


def _reduce_extra_epoch_metrics(extra_sums: dict, total_seen: float, device, distributed: DistributedContext) -> dict:
    if not extra_sums:
        return {}
    import torch

    names = sorted(extra_sums)
    values = torch.tensor([float(extra_sums[name]) for name in names], device=device, dtype=torch.float64)
    if distributed.enabled:
        import torch.distributed as dist

        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    output = {}
    for index, name in enumerate(names):
        if not bool(torch.isfinite(values[index]).item()):
            raise RuntimeError(f"Epoch metric {name} is not finite")
        output[name] = float(values[index].item() / total_seen)
    return output


def _gather_gradient_conflict_samples(samples: list[dict], distributed: DistributedContext) -> list[dict]:
    if not distributed.enabled:
        return list(samples)
    import torch.distributed as dist

    gathered: list[list[dict]] = [list() for _ in range(distributed.world_size)]
    dist.all_gather_object(gathered, list(samples))
    flattened = []
    for rank_samples in gathered:
        flattened.extend(rank_samples)
    return flattened


def _summarize_gradient_conflict_samples(samples: list[dict]) -> dict[str, float]:
    import torch

    if not samples:
        raise RuntimeError("Gradient conflict monitor recorded no samples")
    cosines = torch.tensor([_finite_sample_value(sample, "gradient_cosine_fm_cycle") for sample in samples], dtype=torch.float64)
    norm_fm = torch.tensor([_finite_sample_value(sample, "gradient_norm_fm") for sample in samples], dtype=torch.float64)
    norm_cycle = torch.tensor([_finite_sample_value(sample, "gradient_norm_cycle") for sample in samples], dtype=torch.float64)
    weighted_norm_cycle = torch.tensor([_finite_sample_value(sample, "weighted_gradient_norm_cycle") for sample in samples], dtype=torch.float64)
    weighted_ratios = torch.tensor([_finite_sample_value(sample, "weighted_gradient_ratio_cycle_to_fm") for sample in samples], dtype=torch.float64)
    ratios = norm_cycle / norm_fm
    if not torch.isfinite(ratios).all():
        raise RuntimeError("Gradient norm ratio contains non-finite values")
    if not torch.isfinite(weighted_ratios).all():
        raise RuntimeError("Weighted gradient norm ratio contains non-finite values")
    if bool((norm_fm <= 0.0).any().item()) or bool((norm_cycle <= 0.0).any().item()):
        raise RuntimeError("Gradient norm samples must be positive")
    if bool((weighted_norm_cycle <= 0.0).any().item()) or bool((weighted_ratios <= 0.0).any().item()):
        raise RuntimeError("Weighted gradient norm samples must be positive")
    return {
        "gradient_cosine_fm_cycle_mean": float(cosines.mean().item()),
        "gradient_cosine_fm_cycle_p10": float(torch.quantile(cosines, 0.10).item()),
        "gradient_cosine_fm_cycle_p50": float(torch.quantile(cosines, 0.50).item()),
        "gradient_cosine_fm_cycle_p90": float(torch.quantile(cosines, 0.90).item()),
        "gradient_norm_fm_mean": float(norm_fm.mean().item()),
        "gradient_norm_cycle_mean": float(norm_cycle.mean().item()),
        "gradient_norm_ratio_cycle_to_fm_mean": float(ratios.mean().item()),
        "weighted_gradient_norm_cycle_mean": float(weighted_norm_cycle.mean().item()),
        "weighted_gradient_ratio_cycle_to_fm_mean": float(weighted_ratios.mean().item()),
        "gradient_conflict_fraction": float((cosines < 0.0).to(dtype=torch.float64).mean().item()),
    }


def _finite_sample_value(sample: dict, field: str) -> float:
    if field not in sample:
        raise RuntimeError(f"Gradient conflict sample missing {field}")
    value = sample[field]
    if isinstance(value, bool):
        raise RuntimeError(f"Gradient conflict sample {field} must be numeric, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Gradient conflict sample {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"Gradient conflict sample {field} must be finite, got {value!r}")
    return parsed


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
    ema_config = _ema_config(config)
    _best_model(config, ema_config)
    loss_weighting = _loss_weighting_runtime_from_config(config)
    if loss_weighting.type == "uncertainty":
        if int(_require_field(stages["stage1"], "epochs", "stages.stage1")) != 0:
            raise ValueError("loss_weighting.type='uncertainty' requires stages.stage1.epochs == 0")
        if int(_require_field(stages["stage2"], "epochs", "stages.stage2")) <= 0:
            raise ValueError("loss_weighting.type='uncertainty' requires stages.stage2.epochs > 0")
    _stage2_lambda_schedule(stages, loss_weighting)
    _validate_stage1_gate_config(stages["stage1"])
    _stage2_gradient_conflict_config(stages)
    _validate_validation_block(config)
    _validate_quality_eval_configs(config, stages)
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


def _ema_config(config: dict) -> dict:
    payload = dict(_require_mapping(config, "ema", "train_g config"))
    enabled = _require_bool(payload, "enabled", "ema")
    decay = _require_numeric(payload, "decay", "ema")
    if not 0.0 < decay < 1.0:
        raise ValueError(f"ema.decay must be in (0, 1), got {payload['decay']!r}")
    evaluate_raw = _require_bool(payload, "evaluate_raw", "ema")
    evaluate_ema = _require_bool(payload, "evaluate_ema", "ema")
    save_ema_checkpoint = _require_bool(payload, "save_ema_checkpoint", "ema")
    if enabled:
        if not evaluate_raw:
            raise ValueError("ema.evaluate_raw must be true when ema.enabled is true")
        if not evaluate_ema:
            raise ValueError("ema.evaluate_ema must be true when ema.enabled is true")
        if not save_ema_checkpoint:
            raise ValueError("ema.save_ema_checkpoint must be true when ema.enabled is true")
    else:
        if evaluate_ema:
            raise ValueError("ema.evaluate_ema must be false when ema.enabled is false")
        if save_ema_checkpoint:
            raise ValueError("ema.save_ema_checkpoint must be false when ema.enabled is false")
    return {
        "enabled": enabled,
        "decay": decay,
        "evaluate_raw": evaluate_raw,
        "evaluate_ema": evaluate_ema,
        "save_ema_checkpoint": save_ema_checkpoint,
    }


def _best_model(config: dict, ema_config: dict) -> str:
    value = _require_field(config, "best_model", "train_g config")
    if value not in ("raw", "ema"):
        raise ValueError(f"train_g config.best_model must be 'raw' or 'ema', got {value!r}")
    if value == "raw" and not ema_config["evaluate_raw"]:
        raise ValueError("ema.evaluate_raw must be true when best_model is raw")
    if value == "ema":
        if not ema_config["enabled"]:
            raise ValueError("ema.enabled must be true when best_model is ema")
        if not ema_config["evaluate_ema"]:
            raise ValueError("ema.evaluate_ema must be true when best_model is ema")
    return str(value)


def _require_numeric(config: dict, field: str, context: str) -> float:
    value = _require_field(config, field, context)
    if isinstance(value, bool):
        raise ValueError(f"{context}.{field} must be numeric, got bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{context}.{field} must be finite, got {value!r}")
    return numeric


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


def _validate_quality_eval_configs(config: dict, stages: dict) -> None:
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        payload = stage.get("quality_eval")
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"stages.{stage_name}.quality_eval must be a mapping")
        enabled = _quality_eval_enabled(payload, stage_name)
        if not enabled:
            continue
        context = f"stages.{stage_name}.quality_eval"
        _require_field(payload, "output_dir", context)
        _quality_eval_num_workers(payload, context)
        _quality_eval_variants(payload, stage_name)
        groups = _quality_eval_due_groups(payload, stage_name, 1)
        metric_names = _quality_eval_metric_names(payload, stage_name)
        if not groups and all(name != "niqe" for name in metric_names):
            _quality_eval_due_groups(payload, stage_name, _require_positive_int(payload, "distribution_interval_epochs", f"stages.{stage_name}.quality_eval"))
        if _quality_eval_needs_real_index(metric_names):
            _require_field(payload, "real_index", f"stages.{stage_name}.quality_eval")
        validation = _validate_validation_block(config)
        if not _require_bool(validation, "enabled", "validation"):
            raise ValueError("validation.enabled must be true when quality_eval is enabled")
        for field in ("index", "features", "batch_size"):
            _require_field(validation, field, "validation")


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


def _resume_stage_progress_from_metrics(metrics: dict | None, checkpoint_path: str) -> _ResumeProgress:
    context = f"resume_from checkpoint metrics: {checkpoint_path}"
    if not isinstance(metrics, dict):
        raise ValueError(f"{context} must be a mapping with stage and stage_epoch progress")
    stage = _require_field(metrics, "stage", context)
    if stage not in ("stage1", "stage2"):
        raise ValueError(f"{context}.stage must be stage1 or stage2, got {stage!r}")
    if "stage_epoch" in metrics:
        field = "stage_epoch"
    elif "stage_epoch_0based" in metrics:
        field = "stage_epoch_0based"
    else:
        raise ValueError(f"{context} must include stage_epoch or stage_epoch_0based")
    value = metrics[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}.{field} must be a non-negative integer, got {value!r}")
    return _ResumeProgress(stage=str(stage), stage_epoch=int(value))


def _resume_stage_start_epoch(stage_name: str, stages: dict, resume_progress: _ResumeProgress | None) -> int:
    if stage_name not in ("stage1", "stage2"):
        raise ValueError(f"stage_name must be stage1 or stage2, got {stage_name!r}")
    epochs_by_stage = _resume_stage_epochs(stages)
    epochs = epochs_by_stage[stage_name]
    if resume_progress is None:
        return 0
    stage_order = {"stage1": 0, "stage2": 1}
    if stage_order[stage_name] < stage_order[resume_progress.stage]:
        return epochs
    if stage_name != resume_progress.stage:
        return 0
    if _stage1_checkpoint_initializes_stage2_only(stages, resume_progress):
        return 0
    start_epoch = resume_progress.stage_epoch + 1
    if start_epoch > epochs:
        raise ValueError(
            f"resume_from checkpoint progress {resume_progress.stage} stage_epoch={resume_progress.stage_epoch} "
            f"exceeds configured {stage_name}.epochs={epochs}"
        )
    return start_epoch


def _resume_stage_epochs(stages: dict) -> dict[str, int]:
    if not isinstance(stages, dict):
        raise ValueError("stages must be a mapping")
    epochs_by_stage: dict[str, int] = {}
    for name in ("stage1", "stage2"):
        stage_config = stages.get(name)
        if not isinstance(stage_config, dict):
            raise ValueError(f"stages.{name} must be a mapping")
        value = _require_field(stage_config, "epochs", f"stages.{name}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"stages.{name}.epochs must be a non-negative integer, got {value!r}")
        epochs_by_stage[name] = int(value)
    return epochs_by_stage


def _stage1_checkpoint_initializes_stage2_only(stages: dict, resume_progress: _ResumeProgress) -> bool:
    if resume_progress.stage != "stage1":
        return False
    epochs_by_stage = _resume_stage_epochs(stages)
    return epochs_by_stage["stage1"] == 0 and epochs_by_stage["stage2"] > 0


def _should_check_stage2_gate(stage_name: str, resume_progress: _ResumeProgress | None) -> bool:
    return stage_name == "stage2" and not (resume_progress is not None and resume_progress.stage == "stage2")


def _resume_history_for_checkpoint_selection(history: list[dict], checkpoint_path: str, config: dict, stages: dict) -> list[dict]:
    if not isinstance(history, list):
        raise ValueError(f"resume_from checkpoint history must be a list: {checkpoint_path}")
    invalid_history = []
    for index, item in enumerate(history):
        try:
            _validate_checkpoint_selection_metrics(item, f"resume history item {index}")
        except (KeyError, ValueError) as exc:
            invalid_history.append((index, item, exc))
    if not invalid_history:
        return list(history)
    if not _require_bool(config, "resume_from_legacy_stage1_metrics", "train_g config"):
        first_index, _, first_error = invalid_history[0]
        raise ValueError(
            "resume_from_legacy_stage1_metrics must be true to use a legacy Stage 1 checkpoint "
            f"whose history is missing current checkpoint-selection metrics: {checkpoint_path} "
            f"history_index={first_index} error={first_error}"
        )
    if int(_require_field(stages["stage1"], "epochs", "stages.stage1")) != 0:
        raise ValueError("resume_from_legacy_stage1_metrics requires stages.stage1.epochs == 0")
    if int(_require_field(stages["stage2"], "epochs", "stages.stage2")) <= 0:
        raise ValueError("resume_from_legacy_stage1_metrics requires stages.stage2.epochs > 0")
    non_stage1 = [index for index, item, _ in invalid_history if item.get("stage") != "stage1"]
    if non_stage1:
        raise ValueError(
            "resume_from_legacy_stage1_metrics only permits legacy Stage 1 history; "
            f"non_stage1_indices={non_stage1}"
        )
    return []


def _compute_gradient_conflict_metrics(flow_loss, cycle_loss, parameters, *, lambda_cycle: float) -> dict[str, float]:
    import torch

    if isinstance(lambda_cycle, bool):
        raise RuntimeError("lambda_cycle must be numeric, got bool")
    lambda_cycle = float(lambda_cycle)
    if not math.isfinite(lambda_cycle):
        raise RuntimeError("lambda_cycle must be finite")
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
    weighted_cycle_norm = lambda_cycle * cycle_norm
    weighted_ratio = weighted_cycle_norm / flow_norm
    if not torch.isfinite(weighted_cycle_norm):
        raise RuntimeError("weighted cycle gradient norm is not finite")
    if not torch.isfinite(weighted_ratio):
        raise RuntimeError("weighted cycle-to-flow gradient norm ratio is not finite")
    return {
        "gradient_cosine_fm_cycle": float(cosine.detach().cpu()),
        "gradient_norm_fm": float(flow_norm.detach().cpu()),
        "gradient_norm_cycle": float(cycle_norm.detach().cpu()),
        "weighted_gradient_norm_cycle": float(weighted_cycle_norm.detach().cpu()),
        "weighted_gradient_ratio_cycle_to_fm": float(weighted_ratio.detach().cpu()),
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
            f"face_detect_ge1_rate={detection_rate}, threshold={threshold}, "
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


def _evaluate_validation_variants(
    generator,
    ema: ExponentialMovingAverage | None,
    e0,
    loader,
    detector,
    device,
    generator_config: FlowGeneratorConfig,
    *,
    sampling_seed: int,
    use_amp: bool,
    ema_config: dict,
) -> tuple[dict | None, dict | None]:
    raw_metrics = None
    if ema_config["evaluate_raw"]:
        raw_metrics = _evaluate_validation(
            generator,
            e0,
            loader,
            detector,
            device,
            generator_config,
            sampling_seed=sampling_seed,
            use_amp=use_amp,
        )
    ema_metrics = None
    if ema_config["enabled"] and ema_config["evaluate_ema"]:
        if ema is None:
            raise RuntimeError("EMA validation requested but EMA state is not initialized")
        ema_generator = build_generator(generator_config.to_dict()).to(device)
        ema.copy_to(ema_generator)
        ema_metrics = _evaluate_validation(
            ema_generator,
            e0,
            loader,
            detector,
            device,
            generator_config,
            sampling_seed=sampling_seed,
            use_amp=use_amp,
        )
    return raw_metrics, ema_metrics


def _attach_validation_metrics(metrics: dict, raw_metrics: dict | None, ema_metrics: dict | None) -> None:
    if raw_metrics is not None:
        metrics.update({f"validation_raw_{key}": value for key, value in raw_metrics.items()})
        metrics.update({f"validation_{key}": value for key, value in raw_metrics.items()})
    if ema_metrics is not None:
        metrics.update({f"validation_ema_{key}": value for key, value in ema_metrics.items()})


def _evaluate_generation_quality(**kwargs):
    from scripts.eval_generation_quality import evaluate_generation_quality

    return evaluate_generation_quality(**kwargs)


def _evaluate_generation_quality_subprocess(
    *,
    real_index: Path | None,
    generated_dir: Path,
    output: Path,
    iqa_method: str,
    metrics: tuple[str, ...],
    max_generated: int | None,
    max_real: int | None,
    subset_seed: int,
    device: str,
    cuda_visible_devices: str | None,
    timeout_seconds: int,
) -> dict:
    script_path = _REPO_ROOT / "scripts" / "eval_generation_quality.py"
    if output.exists():
        if output.is_dir():
            raise IsADirectoryError(f"quality_eval distribution output path is a directory: {output}")
        output.unlink()
    command = [
        sys.executable,
        str(script_path),
        "--generated-dir",
        str(generated_dir),
        "--output",
        str(output),
        "--iqa-method",
        iqa_method,
        "--seed",
        str(int(subset_seed)),
        "--device",
        str(device),
        "--metrics",
        *[str(name) for name in metrics],
    ]
    if real_index is not None:
        command.extend(["--real-index", str(real_index)])
    if max_generated is not None:
        command.extend(["--max-generated", str(int(max_generated))])
    if max_real is not None:
        command.extend(["--max-real", str(int(max_real))])

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)

    try:
        completed = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=int(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        details = _subprocess_output_details(exc.stderr, exc.stdout)
        suffix = f": {details}" if details else ""
        raise RuntimeError(f"quality_eval distribution subprocess timed out after {int(timeout_seconds)} seconds{suffix}") from exc
    if completed.returncode != 0:
        details = "\n".join(part for part in (completed.stderr.strip(), completed.stdout.strip()) if part)
        suffix = f": {details}" if details else ""
        raise RuntimeError(f"quality_eval distribution subprocess failed with exit code {completed.returncode}{suffix}")
    if not output.is_file():
        raise FileNotFoundError(f"quality_eval distribution subprocess did not write JSON: {output}")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"quality_eval distribution subprocess wrote invalid JSON: {output}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"quality_eval distribution subprocess JSON must be an object: {output}")
    return payload


def _subprocess_output_details(stderr, stdout) -> str:
    parts = []
    for value in (stderr, stdout):
        if value is None:
            continue
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        text = text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _run_quality_eval_hook(
    config: dict,
    stage_name: str,
    stage_epoch: int,
    *,
    generator=None,
    ema: ExponentialMovingAverage | None = None,
    device=None,
    generator_config: FlowGeneratorConfig | None = None,
    sampling_seed: int | None = None,
    use_amp: bool = False,
    ema_config: dict | None = None,
) -> dict[str, float]:
    stages = config.get("stages")
    if not isinstance(stages, dict):
        return {}
    stage = stages.get(stage_name)
    if not isinstance(stage, dict):
        return {}
    payload = stage.get("quality_eval")
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"stages.{stage_name}.quality_eval must be a mapping")
    enabled = _quality_eval_enabled(payload, stage_name)
    if not enabled:
        return {}
    epoch_number = int(stage_epoch) + 1
    groups = _quality_eval_due_groups(payload, stage_name, epoch_number)
    if not groups:
        return {}
    _require_quality_eval_runtime(
        generator=generator,
        device=device,
        generator_config=generator_config,
        sampling_seed=sampling_seed,
    )
    output_dir = Path(str(_require_field(payload, "output_dir", f"stages.{stage_name}.quality_eval")))
    iqa_method = str(payload.get("iqa_method", "niqe"))
    subset_seed = int(payload.get("subset_seed", sampling_seed))
    quality_device = str(payload.get("device", device))
    distribution_device = str(payload.get("distribution_device", payload.get("device", "auto")))
    distribution_timeout_seconds = (
        _quality_eval_distribution_timeout_seconds(payload, f"stages.{stage_name}.quality_eval")
        if any(_quality_eval_needs_real_index(group.metrics) for group in groups)
        else None
    )
    distribution_cuda_visible_devices = payload.get("distribution_cuda_visible_devices")
    if distribution_cuda_visible_devices is not None:
        distribution_cuda_visible_devices = str(distribution_cuda_visible_devices)
    variants = _quality_eval_variants(payload, stage_name)
    metrics: dict[str, float] = {}
    generation_max_samples = max(group.max_samples for group in groups)
    loader = _build_quality_eval_loader(
        config,
        generation_max_samples,
        quality_eval_config=payload,
        quality_eval_context=f"stages.{stage_name}.quality_eval",
    )
    multiple_variants = len(variants) > 1
    for model_name in variants:
        current_generator = _quality_eval_current_generator(
            model_name,
            generator=generator,
            ema=ema,
            device=device,
            generator_config=generator_config,
            ema_config=ema_config,
        )
        epoch_dir = _quality_eval_epoch_dir(output_dir, epoch_number, model_name, multiple_variants)
        generated_dir = epoch_dir / "generated_images"
        generated_count = _generate_quality_eval_images(
            generator=current_generator,
            loader=loader,
            generated_dir=generated_dir,
            device=device,
            generator_config=generator_config,
            sampling_seed=int(sampling_seed),
            max_samples=generation_max_samples,
            use_amp=use_amp,
        )
        for group in groups:
            real_index = (
                Path(str(_require_field(payload, "real_index", f"stages.{stage_name}.quality_eval")))
                if _quality_eval_needs_real_index(group.metrics)
                else None
            )
            eval_kwargs = {
                "real_index": real_index,
                "generated_dir": generated_dir,
                "output": epoch_dir / f"{stage_name}_epoch_{epoch_number:04d}_{model_name}_{group.name}.json",
                "iqa_method": iqa_method,
                "metrics": group.metrics,
                "max_generated": min(group.max_samples, generated_count),
                "max_real": group.max_samples if real_index is not None else None,
                "subset_seed": subset_seed,
            }
            if _quality_eval_needs_real_index(group.metrics):
                result = _evaluate_generation_quality_subprocess(
                    **eval_kwargs,
                    device=distribution_device,
                    cuda_visible_devices=distribution_cuda_visible_devices,
                    timeout_seconds=distribution_timeout_seconds,
                )
            else:
                result = _evaluate_generation_quality(
                    **eval_kwargs,
                    device=quality_device,
                )
            metrics.update(_quality_payload_to_metrics(result, model_name, group.metrics))
    return metrics


def _require_quality_eval_runtime(*, generator, device, generator_config, sampling_seed) -> None:
    if generator is None:
        raise RuntimeError("quality_eval requires the current generator instance")
    if device is None:
        raise RuntimeError("quality_eval requires the current training device")
    if generator_config is None:
        raise RuntimeError("quality_eval requires the current generator config")
    if sampling_seed is None:
        raise RuntimeError("quality_eval requires the sampling seed")


def _quality_eval_due_groups(payload: dict, stage_name: str, epoch_number: int) -> list[_QualityEvalGroup]:
    if epoch_number <= 0:
        raise ValueError(f"quality_eval epoch_number must be positive, got {epoch_number!r}")
    context = f"stages.{stage_name}.quality_eval"
    metric_names = _quality_eval_metric_names(payload, stage_name)
    groups: list[_QualityEvalGroup] = []
    if "niqe" in metric_names:
        interval = _require_positive_int(payload, "niqe_interval_epochs", context)
        max_samples = _require_positive_int(payload, "niqe_max_samples", context)
        if epoch_number % interval == 0:
            groups.append(_QualityEvalGroup("niqe", ("niqe",), max_samples))
    distribution_metrics = tuple(name for name in metric_names if name in ("fid", "kid"))
    if distribution_metrics:
        interval = _require_positive_int(payload, "distribution_interval_epochs", context)
        max_samples = _quality_eval_distribution_max_samples(payload, context)
        _quality_eval_distribution_timeout_seconds(payload, context)
        _require_field(payload, "real_index", context)
        if epoch_number % interval == 0:
            groups.append(_QualityEvalGroup("distribution", distribution_metrics, max_samples))
    return groups


def _quality_eval_distribution_max_samples(payload: dict, context: str) -> int:
    if "distribution_max_samples" not in payload:
        raise ValueError(f"{context}.distribution_max_samples is required")
    return _require_positive_int(payload, "distribution_max_samples", context)


def _quality_eval_distribution_timeout_seconds(payload: dict, context: str) -> int:
    if "distribution_timeout_seconds" not in payload:
        raise ValueError(f"{context}.distribution_timeout_seconds is required")
    return _require_positive_int(payload, "distribution_timeout_seconds", context)


def _quality_eval_num_workers(payload: dict, context: str) -> int:
    if "quality_num_workers" not in payload:
        raise ValueError(f"{context}.quality_num_workers is required")
    value = payload["quality_num_workers"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}.quality_num_workers must be a non-negative integer, got {value!r}")
    return int(value)


def _quality_eval_enabled(payload: dict, stage_name: str) -> bool:
    if "enabled" not in payload:
        raise ValueError(f"stages.{stage_name}.quality_eval.enabled is required")
    enabled = payload["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(f"stages.{stage_name}.quality_eval.enabled must be true or false")
    return enabled


def _quality_eval_interval_epochs(payload: dict, stage_name: str) -> int:
    context = f"stages.{stage_name}.quality_eval"
    if "interval_epochs" in payload:
        return _require_positive_int(payload, "interval_epochs", context)
    return _require_positive_int(payload, "interval", context)


def _quality_eval_metric_names(payload: dict, stage_name: str) -> tuple[str, ...]:
    payload_context = f"stages.{stage_name}.quality_eval"
    value = _require_field(payload, "metrics", payload_context)
    context = f"{payload_context}.metrics"
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a non-empty list")
    if not value:
        raise ValueError(f"{context} must be a non-empty list")
    parsed = []
    for item in value:
        name = str(item).lower()
        if name not in ("fid", "kid", "niqe"):
            raise ValueError(f"{context} contains unsupported metric {item!r}")
        if name in parsed:
            raise ValueError(f"{context} contains duplicate metric {name!r}")
        parsed.append(name)
    return tuple(parsed)


def _quality_eval_needs_real_index(metric_names: tuple[str, ...]) -> bool:
    return any(name in ("fid", "kid") for name in metric_names)


def _quality_eval_variants(payload: dict, stage_name: str) -> list[str]:
    variants = payload.get("variants")
    if variants is not None:
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"stages.{stage_name}.quality_eval.variants must be a non-empty mapping")
        parsed = []
        for model_name, variant_payload in variants.items():
            if model_name not in ("raw", "ema"):
                raise ValueError(f"stages.{stage_name}.quality_eval variant must be raw or ema, got {model_name!r}")
            if not isinstance(variant_payload, dict):
                raise ValueError(f"stages.{stage_name}.quality_eval.variants.{model_name} must be a mapping")
            parsed.append(str(model_name))
        return parsed
    model_name = str(payload.get("model", "raw"))
    if model_name not in ("raw", "ema"):
        raise ValueError(f"stages.{stage_name}.quality_eval.model must be raw or ema, got {model_name!r}")
    return [model_name]


def _quality_eval_epoch_dir(output_dir: Path, epoch_number: int, model_name: str, multiple_variants: bool) -> Path:
    root = output_dir / f"epoch_{epoch_number:04d}"
    return root / model_name if multiple_variants else root


def _quality_eval_current_generator(
    model_name: str,
    *,
    generator,
    ema: ExponentialMovingAverage | None,
    device,
    generator_config: FlowGeneratorConfig,
    ema_config: dict | None,
):
    if model_name == "raw":
        return generator
    if model_name != "ema":
        raise ValueError(f"quality_eval model must be raw or ema, got {model_name!r}")
    if ema is None or not (ema_config or {}).get("enabled", False):
        raise RuntimeError("quality_eval requested EMA images but EMA is not enabled")
    ema_generator = build_generator(generator_config.to_dict()).to(device)
    ema.copy_to(ema_generator)
    return ema_generator


def _build_quality_eval_loader(config: dict, max_samples: int, *, quality_eval_config: dict, quality_eval_context: str):
    from torch.utils.data import DataLoader, Subset

    validation = _require_mapping(config, "validation", "train_g config")
    if not _require_bool(validation, "enabled", "validation"):
        raise ValueError("validation.enabled must be true when quality_eval is enabled")
    for field in ("index", "features", "batch_size"):
        _require_field(validation, field, "validation")
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError(f"quality_eval max_samples must be positive, got {max_samples!r}")
    val_set = FeatureAlignedAffectNet(
        validation["index"],
        validation["features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
    val_set = Subset(val_set, list(range(min(max_samples, len(val_set)))))
    if len(val_set) == 0:
        raise ValueError("quality_eval validation dataset contains no samples")
    num_workers = _quality_eval_num_workers(quality_eval_config, quality_eval_context)
    loader_kwargs = {
        "batch_size": int(validation["batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": False, "prefetch_factor": 4})
    return DataLoader(val_set, **loader_kwargs)


def _generate_quality_eval_images(
    *,
    generator,
    loader,
    generated_dir: Path,
    device,
    generator_config: FlowGeneratorConfig,
    sampling_seed: int,
    max_samples: int,
    use_amp: bool,
) -> int:
    import torch
    from safa.evaluation.runner import _save_generated_image_for_eval

    if generated_dir.exists():
        raise FileExistsError(f"quality_eval generated image directory already exists: {generated_dir}")
    generated_dir.parent.mkdir(parents=True, exist_ok=True)
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError(f"quality_eval max_samples must be positive, got {max_samples!r}")
    was_training = bool(getattr(generator, "training", False))
    generator.eval()
    count = 0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    try:
        with torch.no_grad(), amp_ctx:
            for batch in loader:
                if count >= max_samples:
                    break
                z = batch["z"].to(device, non_blocking=True)
                sample_ids = list(batch["sample_id"])
                remaining = max_samples - count
                if int(z.shape[0]) > remaining:
                    z = z[:remaining]
                    sample_ids = sample_ids[:remaining]
                x_init = make_x_init_for_sample_ids(sample_ids, sampling_seed, generator_config.image_size, z.device, z.dtype)
                generated = generator.sample(z, steps=generator_config.sample_steps, x_init=x_init)
                assert_finite_tensor("quality_eval_generated_image", generated)
                for index, sample_id in enumerate(sample_ids):
                    _save_generated_image_for_eval(
                        generated[index],
                        generated_dir,
                        global_index=count + index,
                        sample_id=sample_id,
                        row={},
                    )
                count += len(sample_ids)
    finally:
        if was_training:
            generator.train()
    if count <= 0:
        raise RuntimeError("quality_eval generated zero images")
    return count


def _quality_payload_to_metrics(payload: dict, model_name: str, metric_names: tuple[str, ...] = ("fid", "kid", "niqe")) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "fid" in metric_names:
        metrics[f"quality_{model_name}_fid"] = _finite_quality_value(payload, "fid")
    if "kid" in metric_names:
        metrics[f"quality_{model_name}_kid_mean"] = _finite_quality_value(payload, "kid_mean")
        metrics[f"quality_{model_name}_kid_std"] = _finite_quality_value(payload, "kid_std")
    if "niqe" in metric_names:
        iqa = payload.get("iqa")
        if not isinstance(iqa, dict):
            raise ValueError("quality_eval payload missing iqa metrics")
        method = str(iqa.get("method", ""))
        if method.lower() != "niqe":
            raise ValueError(f"quality_eval iqa.method must be niqe for medium_v1 checkpoints, got {method!r}")
        niqe_mean = _finite_quality_value(iqa, "mean")
        metrics[f"quality_{model_name}_niqe_mean"] = niqe_mean
        metrics[f"quality_{model_name}_niqe"] = niqe_mean
        if "std" in iqa:
            metrics[f"quality_{model_name}_niqe_std"] = _finite_quality_value(iqa, "std")
    return metrics


def _finite_quality_value(payload: dict, field: str) -> float:
    if field not in payload:
        raise ValueError(f"quality_eval payload missing {field}")
    value = payload[field]
    if isinstance(value, bool):
        raise ValueError(f"quality_eval {field} must be numeric, got bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"quality_eval {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"quality_eval {field} must be finite, got {value!r}")
    return numeric


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


def _composite_score(item: dict, best_model: str = "raw") -> float:
    """New checkpoint composite: cosine x single_face_eq1_rate.

    Old reports used validation_face_detection_rate, which is the ge1 rate.
    """
    cosine = item[_checkpoint_metric_field(item, "latent_cosine_mean", best_model)]
    single_face = item[_checkpoint_metric_field(item, "single_face_eq1_rate", best_model)]
    return cosine * single_face


def _stage2_checkpoint_filenames_to_save(metrics: dict, previous: list[dict]) -> list[str]:
    if metrics.get("stage") != "stage2":
        return []
    filenames = []
    for model_name in ("raw", "ema"):
        if _has_utility_metrics(metrics, model_name) and _is_better_utility_for_model(metrics, previous, model_name):
            filenames.append(f"best_{model_name}_utility.pt")
        if _has_quality_metrics(metrics, model_name) and _is_better_quality_for_model(metrics, previous, model_name):
            filenames.append(f"best_{model_name}_quality.pt")
    return filenames


def _stage1_single_face_checkpoint_filenames_to_save(metrics: dict, previous: list[dict]) -> list[str]:
    if metrics.get("stage") != "stage1":
        return []
    if not _is_better_single_face_stage1(metrics, previous):
        return []
    epoch_number = _stage_epoch_index(metrics, "stage1 single-face checkpoint metrics") + 1
    return ["best_single_face.pt", f"best_single_face_epoch_{epoch_number:04d}.pt"]


def _is_better_single_face_stage1(metrics: dict, previous: list[dict]) -> bool:
    if metrics.get("stage") != "stage1":
        return False
    current_score = _stage1_single_face_score(metrics)
    candidates = [item for item in previous if item.get("stage") == "stage1"]
    if not candidates:
        return True
    best = max(candidates, key=_stage1_single_face_score)
    return current_score > _stage1_single_face_score(best)


def _stage1_single_face_score(item: dict) -> tuple[float, float, float, float, float, float, int]:
    context = "stage1 single-face checkpoint metrics"
    single_face = _finite_metric_value(item, "validation_raw_single_face_eq1_rate", context)
    multi_face = _finite_metric_value(item, "validation_raw_multi_face_rate", context)
    zero_face = _finite_metric_value(item, "validation_raw_zero_face_rate", context)
    face_detect_ge1 = _finite_metric_value(item, "validation_raw_face_detect_ge1_rate", context)
    loss = _finite_metric_value(item, "loss", context)
    epoch = _stage_epoch_index(item, context)
    multi_face_is_zero = 1.0 if multi_face == 0.0 else 0.0
    return (multi_face_is_zero, single_face, -multi_face, -zero_face, face_detect_ge1, -loss, -epoch)


def _stage_epoch_index(item: dict, context: str) -> int:
    epoch = _finite_metric_value(item, "stage_epoch", context)
    if epoch < 0 or int(epoch) != epoch:
        raise ValueError(f"{context}.stage_epoch must be a non-negative integer, got {epoch!r}")
    return int(epoch)


def _has_utility_metrics(item: dict, model_name: str) -> bool:
    try:
        _checkpoint_metric_field(item, "latent_cosine_mean", model_name)
        _checkpoint_metric_field(item, "single_face_eq1_rate", model_name)
    except KeyError:
        return False
    return True


def _utility_score_for_model(item: dict, model_name: str) -> float:
    cosine = _finite_metric_value(item, _checkpoint_metric_field(item, "latent_cosine_mean", model_name), "utility checkpoint metrics")
    single_face = _finite_metric_value(item, _checkpoint_metric_field(item, "single_face_eq1_rate", model_name), "utility checkpoint metrics")
    return cosine * single_face


def _is_better_utility_for_model(metrics: dict, previous: list[dict], model_name: str) -> bool:
    current_score = _utility_score_for_model(metrics, model_name)
    candidates = [item for item in previous if item.get("stage") == "stage2" and _has_utility_metrics(item, model_name)]
    if not candidates:
        return True
    best = max(candidates, key=lambda item: (_utility_score_for_model(item, model_name), -float(item["loss"])))
    return (current_score, -float(metrics["loss"])) > (_utility_score_for_model(best, model_name), -float(best["loss"]))


def _has_quality_metrics(item: dict, model_name: str) -> bool:
    return all(f"quality_{model_name}_{field}" in item for field in ("fid", "kid_mean", "niqe"))


def _quality_tuple_for_model(item: dict, model_name: str) -> tuple[float, float, float, float]:
    context = f"quality {model_name} checkpoint metrics"
    fid = _finite_metric_value(item, f"quality_{model_name}_fid", context)
    kid = _finite_metric_value(item, f"quality_{model_name}_kid_mean", context)
    niqe = _finite_metric_value(item, f"quality_{model_name}_niqe", context)
    loss = _finite_metric_value(item, "loss", context)
    return fid, kid, niqe, loss


def _is_better_quality_for_model(metrics: dict, previous: list[dict], model_name: str) -> bool:
    current = _quality_tuple_for_model(metrics, model_name)
    candidates = [item for item in previous if item.get("stage") == "stage2" and _has_quality_metrics(item, model_name)]
    if not candidates:
        return True
    best = min(candidates, key=lambda item: _quality_tuple_for_model(item, model_name))
    return current < _quality_tuple_for_model(best, model_name)


def _finite_metric_value(item: dict, field: str, context: str) -> float:
    if field not in item:
        raise ValueError(f"{context}.{field} is required")
    value = item[field]
    if isinstance(value, bool):
        raise ValueError(f"{context}.{field} must be numeric, got bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{context}.{field} must be finite, got {value!r}")
    return numeric


def _checkpoint_metric_field(item: dict, metric_name: str, best_model: str) -> str:
    if best_model not in ("raw", "ema"):
        raise ValueError(f"best_model must be 'raw' or 'ema', got {best_model!r}")
    prefixed = f"validation_{best_model}_{metric_name}"
    if prefixed in item:
        return prefixed
    if best_model == "raw":
        legacy = f"validation_{metric_name}"
        if legacy in item:
            return legacy
        raise KeyError(legacy)
    raise KeyError(prefixed)


def _validate_checkpoint_selection_metrics(metrics: dict, context: str = "checkpoint metrics", best_model: str = "raw") -> None:
    for field in ("loss", "stage"):
        _require_field(metrics, field, context)
    try:
        metric_fields = (
            _checkpoint_metric_field(metrics, "latent_cosine_mean", best_model),
            _checkpoint_metric_field(metrics, "single_face_eq1_rate", best_model),
            "loss",
        )
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"{context}.{missing} is required") from exc
    for field in metric_fields:
        value = metrics[field]
        if isinstance(value, bool):
            raise ValueError(f"{context}.{field} must be numeric, got bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.{field} must be numeric, got {value!r}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{context}.{field} must be finite, got {value!r}")


def _is_better(metrics: dict, previous: list[dict], best_model: str = "raw") -> bool:
    _validate_checkpoint_selection_metrics(metrics, best_model=best_model)
    current_score = _composite_score(metrics, best_model)
    if not previous:
        return True
    stage = metrics["stage"]
    same_stage = []
    for item in previous:
        _validate_checkpoint_selection_metrics(item, "checkpoint history item", best_model=best_model)
        if item["stage"] == stage:
            same_stage.append(item)
    if not same_stage:
        return True
    best = max(same_stage, key=lambda item: (_composite_score(item, best_model), -item["loss"]))
    return (current_score, -metrics["loss"]) > (_composite_score(best, best_model), -best["loss"])


def _is_better_overall(metrics: dict, previous: list[dict], best_model: str = "raw") -> bool:
    """Compare current epoch against ALL previous epochs regardless of stage.

    Unlike _is_better which only compares within the same stage, this function
    compares across stages. Uses composite score (cosine x single_face_eq1_rate)
    to prevent selecting degenerate checkpoints with high cosine but invalid face counts.
    """
    _validate_checkpoint_selection_metrics(metrics, best_model=best_model)
    current_score = _composite_score(metrics, best_model)
    if not previous:
        return True
    for item in previous:
        _validate_checkpoint_selection_metrics(item, "checkpoint history item", best_model=best_model)
    best = max(previous, key=lambda item: (_composite_score(item, best_model), -item["loss"]))
    return (current_score, -metrics["loss"]) > (_composite_score(best, best_model), -best["loss"])


def _save_generator(
    path: Path,
    generator,
    generator_config: FlowGeneratorConfig,
    train_config: dict,
    metrics: dict,
    history: list[dict],
    *,
    ema_model_state_dict: dict | None = None,
    metrics_raw: dict | None = None,
    metrics_ema: dict | None = None,
    ema_config: dict | None = None,
    best_model: str | None = None,
    loss_weighting_state: dict | None = None,
    optimizer_state_dict: dict | None = None,
) -> None:
    import torch

    ema_config = dict(ema_config if ema_config is not None else _ema_config(train_config))
    best_model = str(best_model if best_model is not None else _best_model(train_config, ema_config))
    _validate_checkpoint_selection_metrics(metrics, best_model=best_model)
    if ema_config["enabled"] and ema_config["save_ema_checkpoint"] and ema_model_state_dict is None:
        raise ValueError("ema_model_state_dict is required when ema.enabled and ema.save_ema_checkpoint are true")
    if not ema_config["enabled"] and ema_model_state_dict is not None:
        raise ValueError("ema_model_state_dict must not be provided when ema.enabled is false")
    generator = unwrap_model(generator)
    path.parent.mkdir(parents=True, exist_ok=True)
    training_config = {
        "stages": train_config.get("stages"),
        "validation": train_config.get("validation"),
        "ema": ema_config,
        "best_model": best_model,
    }
    if "loss_weighting" in train_config:
        training_config["loss_weighting"] = train_config["loss_weighting"]
    if "resume_from_legacy_stage1_metrics" in train_config:
        training_config["resume_from_legacy_stage1_metrics"] = train_config["resume_from_legacy_stage1_metrics"]
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
        "metrics_raw": metrics_raw,
        "metrics_ema": metrics_ema,
        "ema_config": ema_config,
        "history": history,
        "training_config": training_config,
    }
    if loss_weighting_state is not None:
        payload["loss_weighting_state"] = loss_weighting_state
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if ema_config["enabled"] and ema_config["save_ema_checkpoint"]:
        payload["ema_model_state_dict"] = ema_model_state_dict
    sampling_seed = optional_sampling_base_seed_from_config(train_config)
    if sampling_seed is not None:
        payload["sampling"] = {"base_seed": sampling_seed, "stable_x_init": True}
    torch.save(payload, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
