from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from safa.guidance.meanflow_flow_map import (
    CountedFlowMap,
    GuidanceResult,
    freeze_guidance_stack,
    optimize_initial_noise,
    sample_official_head_current_xt,
    sample_paper_algorithm_split,
    semigroup_probe,
)
from safa.training.losses import normalize_for_e0
from safa.utils.sampling import make_x_init_for_sample_ids


EXPECTED_CHECKPOINT_PATH = (
    "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last_nopretrained.pt"
)
EXPECTED_STAGE = "stage2"
EXPECTED_STAGE_EPOCH = 1652
EXPECTED_MODEL_CONFIG: dict[str, Any] = {
    "model_type": "meanflow_sit",
    "sit_patch_size": 4,
    "sit_hidden_size": 768,
    "sit_depth": 12,
    "sit_num_heads": 12,
    "sit_input_channels": 4,
    "image_size": 32,
    "embedding_dim": 512,
    "learned_null_condition": True,
    "sample_steps": 1,
    "sit_data_space": "latent",
}
SUPPORTED_MODES = frozenset(
    {"native", "semigroup", "official_head_current_xt", "paper_algorithm_split", "initial_noise"}
)
_MODE_ALIASES = {"noise_oracle": "initial_noise"}
FMRG_MODES = frozenset({"official_head_current_xt", "paper_algorithm_split"})
_SAFE_SAMPLE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class GuidanceRuntime:
    generator: Any
    codec: Any
    e0: Any
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_state: dict[str, Any]
    edev: Any | None = None
    e0_checkpoint_path: Path | None = None
    e0_checkpoint_sha256: str = ""
    edev_checkpoint_path: Path | None = None
    edev_checkpoint_sha256: str = ""
    vae_path: Path | None = None
    vae_digest: str = ""
    vae_scaling_factor: float = 0.0
    real_index_path: Path | None = None
    real_index_sha256: str = ""


def validate_checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("MeanFlow checkpoint must contain a mapping")
    if payload.get("stage") != EXPECTED_STAGE:
        raise ValueError(
            f"checkpoint stage must be {EXPECTED_STAGE!r}, got {payload.get('stage')!r}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("checkpoint missing metrics mapping")
    epoch = metrics.get("stage_epoch_1based")
    if epoch != EXPECTED_STAGE_EPOCH:
        raise ValueError(
            f"checkpoint metrics.stage_epoch_1based must be {EXPECTED_STAGE_EPOCH}, got {epoch!r}"
        )
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint missing model_config mapping")
    for field, expected in EXPECTED_MODEL_CONFIG.items():
        actual = model_config.get(field)
        if actual != expected:
            raise ValueError(f"checkpoint model_config.{field} must be {expected!r}, got {actual!r}")
    ema_state = payload.get("ema_model_state_dict")
    if not isinstance(ema_state, Mapping) or not ema_state:
        raise ValueError("checkpoint missing non-empty ema_model_state_dict")
    return {
        "stage": EXPECTED_STAGE,
        "stage_epoch_1based": EXPECTED_STAGE_EPOCH,
        "model_type": EXPECTED_MODEL_CONFIG["model_type"],
        "sit_patch_size": EXPECTED_MODEL_CONFIG["sit_patch_size"],
        "model_config": dict(model_config),
        "weight_source": "ema_model_state_dict",
        "strict_state_load": True,
    }


def load_ema_generator(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    generator_builder=None,
) -> tuple[Any, dict[str, Any]]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MeanFlow checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = validate_checkpoint_contract(payload)
    model_config = dict(payload["model_config"])
    # The target EMA is complete. Do not load an older pretrained SiT before replacing it.
    model_config["sit_pretrained_path"] = ""
    model_config["sit_pretrained_state_key"] = ""
    if generator_builder is None:
        from safa.models.generator import build_generator

        generator_builder = build_generator
    generator = generator_builder(model_config)
    ema_state = payload["ema_model_state_dict"]
    generator.load_state_dict(ema_state, strict=True)
    del ema_state
    del payload
    generator = generator.to(device).eval()
    generator.requires_grad_(False)
    return generator, metadata


def validate_guidance_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("guidance config must be a mapping")
    resolved = dict(config)
    present_forbidden = sorted(
        str(field)
        for field in resolved
        if re.search(r"(^|_)e[12]($|_)", str(field).lower())
    )
    if present_forbidden:
        raise ValueError(f"guidance runner must not accept or load E1/E2 fields: {present_forbidden!r}")
    checkpoint = str(resolved.get("checkpoint", ""))
    if checkpoint != EXPECTED_CHECKPOINT_PATH:
        raise ValueError(
            f"guidance checkpoint must be exactly {EXPECTED_CHECKPOINT_PATH!r}, got {checkpoint!r}"
        )
    if resolved.get("checkpoint_model") != "ema":
        raise ValueError("guidance checkpoint_model must be 'ema'")
    if resolved.get("transport_condition") != "learned_null_condition":
        raise ValueError("guidance transport_condition must be learned_null_condition")
    mode = str(resolved.get("mode", resolved.get("route", "")))
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"guidance mode must be one of {sorted(SUPPORTED_MODES)}, got {mode!r}")
    resolved["mode"] = mode
    resolved.pop("route", None)
    if "sampling_seed" not in resolved and "seed" not in resolved:
        raise ValueError("guidance config requires sampling_seed or seed")
    resolved["sampling_seed"] = int(resolved.get("sampling_seed", resolved.get("seed")))
    for field, expected in (
        ("expected_stage", EXPECTED_STAGE),
        ("expected_stage_epoch_1based", EXPECTED_STAGE_EPOCH),
        ("expected_model_type", EXPECTED_MODEL_CONFIG["model_type"]),
        ("expected_sit_patch_size", EXPECTED_MODEL_CONFIG["sit_patch_size"]),
    ):
        if field in resolved and resolved[field] != expected:
            raise ValueError(f"guidance config {field} must be {expected!r}, got {resolved[field]!r}")
    return resolved


def asset_contract_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "e0_checkpoint",
        "edev_checkpoint",
        "vae_path",
        "vae_scaling_factor",
        "index",
    )
    missing = [field for field in required if not config.get(field)]
    if missing:
        raise ValueError(f"guidance asset contract missing required fields: {missing!r}")
    mode = _MODE_ALIASES.get(str(config.get("mode", "")), str(config.get("mode", "")))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"guidance asset contract has unsupported mode {mode!r}")
    e0_path = Path(str(config["e0_checkpoint"]))
    edev_path = Path(str(config["edev_checkpoint"]))
    vae_path = Path(str(config["vae_path"]))
    index_path = Path(str(config["index"]))
    e0_digest = _digest_path(e0_path)
    edev_digest = _digest_path(edev_path)
    vae_digest = _digest_path(vae_path)
    index_digest = _digest_path(index_path)
    _validate_expected_digest(config, ("e0_sha256", "e0_checkpoint_sha256"), e0_digest, "E0")
    _validate_expected_digest(config, ("edev_sha256", "edev_checkpoint_sha256"), edev_digest, "Edev")
    _validate_expected_digest(config, ("vae_digest", "vae_sha256"), vae_digest, "VAE")
    _validate_expected_digest(config, ("index_sha256", "real_index_sha256"), index_digest, "real index")
    scale = float(config["vae_scaling_factor"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"vae_scaling_factor must be positive and finite, got {scale!r}")
    seed = int(config.get("sampling_seed", config.get("seed")))
    return {
        "e0": {"path": str(e0_path), "sha256": e0_digest},
        "edev": {"path": str(edev_path), "sha256": edev_digest},
        "vae": {"path": str(vae_path), "digest": vae_digest, "scaling_factor": scale},
        "real_index": {"path": str(index_path), "sha256": index_digest},
        "seed": seed,
        "schedule": _json_safe(config.get("locked_schedule")),
        "mode": mode,
    }


def build_frozen_runtime(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    checkpoint_sha256: str,
    asset_contract: Mapping[str, Any],
    generator_loader=None,
    encoder_loader=None,
    codec_builder=None,
) -> GuidanceRuntime:
    if generator_loader is None:
        generator_loader = load_ema_generator
    if encoder_loader is None:
        from safa.models.e0 import load_e0_checkpoint

        encoder_loader = load_e0_checkpoint
    if codec_builder is None:
        from safa.training.latent_codec import build_latent_codec_from_train_config

        codec_builder = build_latent_codec_from_train_config
    generator, checkpoint_state = generator_loader(config["checkpoint"], device=device)
    e0, _ = encoder_loader(config["e0_checkpoint"], device="cpu")
    e0 = e0.to(device)
    edev = None
    if str(config.get("phase", "full")) == "calibration":
        edev, _ = encoder_loader(config["edev_checkpoint"], device="cpu")
        edev = edev.to(device).eval().requires_grad_(False)
    codec_config = dict(config)
    codec_config.update(
        {
            "latent_training": True,
            "image_size": int(EXPECTED_MODEL_CONFIG["image_size"]),
            "pixel_image_size": int(config.get("pixel_image_size", 256)),
        }
    )
    codec = codec_builder(codec_config, device)
    if codec is None:
        raise RuntimeError("MeanFlow guidance requires the configured latent VAE")
    freeze_guidance_stack(generator, codec, e0)
    if edev is not None:
        edev.eval().requires_grad_(False)
    return GuidanceRuntime(
        generator=generator,
        codec=codec,
        e0=e0,
        edev=edev,
        device=device,
        checkpoint_path=Path(str(config["checkpoint"])),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_state=checkpoint_state,
        e0_checkpoint_path=Path(str(asset_contract["e0"]["path"])),
        e0_checkpoint_sha256=str(asset_contract["e0"]["sha256"]),
        edev_checkpoint_path=Path(str(asset_contract["edev"]["path"])),
        edev_checkpoint_sha256=str(asset_contract["edev"]["sha256"]),
        vae_path=Path(str(asset_contract["vae"]["path"])),
        vae_digest=str(asset_contract["vae"]["digest"]),
        vae_scaling_factor=float(asset_contract["vae"]["scaling_factor"]),
        real_index_path=Path(str(asset_contract["real_index"]["path"])),
        real_index_sha256=str(asset_contract["real_index"]["sha256"]),
    )


def resolve_locked_schedule(
    config: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    explicit_t_cut: float | None = None,
) -> dict[str, Any]:
    manifest_value = config.get("schedule_manifest")
    if not manifest_value:
        raise ValueError("FMRG guidance requires a locked schedule_manifest")
    manifest_path = Path(str(manifest_value))
    manifest = _read_json_mapping(manifest_path, "locked schedule manifest")
    if manifest.get("gate_passed") is not True:
        raise ValueError("locked schedule manifest must record gate_passed=true")
    manifest_hash = manifest.get("checkpoint_sha256")
    if manifest_hash != checkpoint_sha256:
        raise ValueError(
            "locked schedule checkpoint SHA256 does not match the loaded checkpoint: "
            f"manifest={manifest_hash!r} loaded={checkpoint_sha256!r}"
        )
    t_cut = _finite_open_unit(manifest.get("t_cut"), "locked schedule t_cut")
    candidates = {
        "config": config.get("t_cut"),
        "explicit CLI": explicit_t_cut,
    }
    for source, value in candidates.items():
        if value is None:
            continue
        parsed = _finite_open_unit(value, f"{source} t_cut")
        if not math.isclose(parsed, t_cut, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{source} t_cut {parsed} disagrees with locked manifest t_cut {t_cut}")
    guided = [1.0 - index * (1.0 - t_cut) / 3.0 for index in range(4)]
    unguided = [t_cut, t_cut / 2.0, 0.0]
    guided[-1] = t_cut
    for field, expected in (("guided_times", guided), ("unguided_times", unguided)):
        if field in config and not _float_sequences_equal(config[field], expected):
            raise ValueError(f"config {field} disagrees with the locked uniform schedule")
    return {
        "manifest": str(manifest_path),
        "checkpoint_sha256": checkpoint_sha256,
        "t_cut": t_cut,
        "guided_times": guided,
        "unguided_times": unguided,
        "gate_passed": True,
    }


def read_ordered_sample_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_no}: expected JSON object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{manifest_path}:{line_no}: sample_id must be a non-empty string")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id in manifest: {sample_id!r}")
            seen.add(sample_id)
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"sample manifest contains no rows: {manifest_path}")
    return rows


def deterministic_shard(rows: Sequence[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if isinstance(num_shards, bool) or int(num_shards) != num_shards or num_shards <= 0:
        raise ValueError(f"num_shards must be a positive integer, got {num_shards!r}")
    if isinstance(shard_index, bool) or int(shard_index) != shard_index or not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0,{num_shards}), got {shard_index!r}")
    return [dict(row) for position, row in enumerate(rows) if position % int(num_shards) == int(shard_index)]


def resume_remaining_ids(expected_ids: Sequence[str], completed_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    expected = list(expected_ids)
    completed = [row.get("sample_id") for row in completed_rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in completed):
        raise ValueError("resume rows require non-empty string sample_id values")
    if len(set(completed)) != len(completed):
        raise ValueError("resume rows contain duplicate sample IDs")
    if completed != expected[: len(completed)]:
        raise ValueError("resume rows must be an exact prefix of the deterministic shard")
    return expected[len(completed) :]


def execute_guidance_mode(
    *,
    config: Mapping[str, Any],
    generator,
    codec,
    e0,
    x_init: torch.Tensor,
    target_z0: torch.Tensor,
    schedule: Mapping[str, Any] | None,
) -> GuidanceResult:
    mode = _MODE_ALIASES.get(str(config.get("mode", "")), str(config.get("mode", "")))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported guidance mode {mode!r}")
    transport_condition = generator.make_null_condition(
        batch_size=x_init.shape[0], device=x_init.device, dtype=x_init.dtype
    )
    counted = CountedFlowMap(generator, kind=mode)
    if mode == "native":
        with torch.no_grad():
            latent = counted(x_init, transport_condition, t=1.0, r=0.0).detach()
        return GuidanceResult(
            latent=latent,
            nfe=1,
            diagnostics={"mode": mode, "flow_map_trace": list(counted.trace)},
        )
    if mode == "semigroup":
        report = semigroup_probe(
            counted,
            x_init,
            transport_condition,
            config.get("split_times", (0.25, 0.5, 0.75)),
        )
        return GuidanceResult(
            latent=report["direct_endpoint"],
            nfe=int(report["nfe"]),
            diagnostics={"mode": mode, "semigroup": report, "flow_map_trace": list(counted.trace)},
        )
    if mode == "initial_noise":
        result = optimize_initial_noise(
            flow_map=counted,
            codec=codec,
            e0=e0,
            x_init=x_init,
            transport_condition=transport_condition,
            target_z0=target_z0,
            num_updates=int(config.get("num_updates", 0)),
            eta=float(config.get("eta", 0.25)),
            projection=str(config.get("projection", "fixed_radius")),
            typical_delta=float(config.get("typical_delta", 0.05)),
        )
        return _result_with_trace(result, counted.trace, mode)
    if schedule is None:
        raise ValueError(f"{mode} requires a locked schedule")
    guided_times = schedule["guided_times"]
    unguided_times = schedule["unguided_times"]
    if mode == "official_head_current_xt":
        result = sample_official_head_current_xt(
            flow_map=counted,
            codec=codec,
            e0=e0,
            x_init=x_init,
            transport_condition=transport_condition,
            target_z0=target_z0,
            guided_times=guided_times,
            unguided_times=unguided_times,
            sample_mode=str(config.get("sample_mode", "flow_map1")),
            optimization_mode=str(
                config.get("optimization_mode", "paper_normalized_direct_autograd")
            ),
            num_optim_iters=int(config.get("num_optim_iters", 1)),
            step_size=float(config.get("step_size", config.get("eta", 0.25))),
        )
        return _result_with_trace(result, counted.trace, mode)
    if mode == "paper_algorithm_split":
        result = sample_paper_algorithm_split(
            flow_map=counted,
            codec=codec,
            e0=e0,
            x_init=x_init,
            transport_condition=transport_condition,
            target_z0=target_z0,
            guided_times=guided_times,
            unguided_times=unguided_times,
            step_size=float(config.get("step_size", config.get("eta", 0.25))),
        )
        return _result_with_trace(result, counted.trace, mode)
    raise AssertionError(f"unhandled guidance mode {mode!r}")


def execute_matched_native(*, generator, x_init: torch.Tensor) -> GuidanceResult:
    condition = generator.make_null_condition(
        batch_size=x_init.shape[0], device=x_init.device, dtype=x_init.dtype
    )
    counted = CountedFlowMap(generator, kind="matched_native")
    with torch.no_grad():
        latent = counted(x_init, condition, t=1.0, r=0.0).detach()
    return GuidanceResult(
        latent=latent,
        nfe=1,
        diagnostics={"mode": "matched_native", "flow_map_trace": list(counted.trace)},
    )


def run_guidance_records(
    *,
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    runtime: GuidanceRuntime,
    output_dir: str | Path,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    invocation_started = time.perf_counter()
    output = Path(output_dir)
    selected = deterministic_shard([dict(row) for row in records], shard_index, num_shards)
    _validate_generation_records(selected)
    expected_ids = [str(row["sample_id"]) for row in selected]
    if not expected_ids:
        raise ValueError(f"shard {shard_index}/{num_shards} contains no samples")
    mode = _MODE_ALIASES.get(str(config.get("mode", "")), str(config.get("mode", "")))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported guidance mode {mode!r}")
    resolved_config = dict(config)
    resolved_config["mode"] = mode
    resolved_config.pop("route", None)
    generated_dir = output / "generated_images"
    native_dir = output / "native_images"
    per_sample_path = output / "per_sample.jsonl"
    run_manifest_path = output / "run_manifest.json"
    generation_result_path = output / "generation_result.json"
    sample_manifest_path = output / "sample_id_manifest.jsonl"
    semigroup_path = output / "semigroup.json"
    resume_contract_path = output / "resume_contract.json"
    session_history_path = output / "session_history.jsonl"
    session_journal_path = output / "session_journal.json"

    phase = str(resolved_config.get("phase", "full"))
    contact_enabled = _contact_sheets_enabled(resolved_config)
    _validate_output_entries(
        output,
        mode=mode,
        contact_sheets=contact_enabled,
    )
    if phase == "calibration" and runtime.edev is None:
        raise ValueError("calibration guidance requires a frozen Edev checkpoint")
    if phase != "calibration" and runtime.edev is not None:
        raise ValueError("Edev must not be loaded or scored outside calibration")
    completed_artifacts = [path for path in (run_manifest_path, generation_result_path) if path.exists()]
    if completed_artifacts:
        raise FileExistsError(f"refusing to replace completed output: {completed_artifacts[0]}")
    output.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    if mode != "native":
        native_dir.mkdir(parents=True, exist_ok=True)

    resume_contract = {
        "checkpoint": {
            "path": str(runtime.checkpoint_path),
            "sha256": runtime.checkpoint_sha256,
            "state": runtime.checkpoint_state,
        },
        "e0": {
            "path": "" if runtime.e0_checkpoint_path is None else str(runtime.e0_checkpoint_path),
            "sha256": runtime.e0_checkpoint_sha256,
        },
        "edev": {
            "path": "" if runtime.edev_checkpoint_path is None else str(runtime.edev_checkpoint_path),
            "sha256": runtime.edev_checkpoint_sha256,
        },
        "vae": {
            "path": "" if runtime.vae_path is None else str(runtime.vae_path),
            "digest": runtime.vae_digest,
            "scaling_factor": runtime.vae_scaling_factor,
        },
        "real_index": {
            "path": "" if runtime.real_index_path is None else str(runtime.real_index_path),
            "sha256": runtime.real_index_sha256,
        },
        "seed": int(resolved_config.get("sampling_seed", resolved_config.get("seed", 0))),
        "schedule": resolved_config.get("locked_schedule"),
        "mode": mode,
        "config": resolved_config,
        "sample_id_sha256": _sample_id_digest(expected_ids),
        "shard": {"index": int(shard_index), "count": int(num_shards)},
    }
    if resume_contract_path.exists():
        existing_contract = _read_json_mapping(resume_contract_path, "resume contract")
        if existing_contract != _json_safe(resume_contract):
            raise ValueError("existing resume contract disagrees with the fixed run contract")
    else:
        _atomic_write_json(resume_contract_path, resume_contract, exclusive=True)

    expected_manifest_rows = [
        {"ordinal": ordinal, "sample_id": row["sample_id"], "source": str(row["source"])}
        for ordinal, row in enumerate(selected)
    ]
    if sample_manifest_path.exists():
        existing_manifest = _read_optional_jsonl(sample_manifest_path)
        if existing_manifest != expected_manifest_rows:
            raise ValueError("existing sample_id_manifest.jsonl disagrees with the deterministic shard")
    else:
        _write_jsonl(sample_manifest_path, expected_manifest_rows, mode="x")

    completed_rows = _read_optional_jsonl(per_sample_path)
    expected_bindings = _expected_row_bindings(selected, generated_dir, native_dir, mode)
    _validate_resume_rows(completed_rows, expected_bindings)
    completed_count = len(completed_rows)
    remaining_records = selected[completed_count:]
    _validate_owned_png_state(
        generated_dir=generated_dir,
        native_dir=native_dir,
        expected_bindings=expected_bindings,
        completed_count=completed_count,
        mode=mode,
    )

    schedule = resolved_config.get("locked_schedule")
    if schedule is not None and not isinstance(schedule, Mapping):
        raise ValueError("config.locked_schedule must be a mapping")
    batch_size = _positive_int(resolved_config.get("batch_size", 1), "batch_size")
    sampling_seed = int(resolved_config.get("sampling_seed", resolved_config.get("seed", 0)))
    image_size = _positive_int(resolved_config.get("image_size", 32), "image_size")
    channels = _sample_channels(runtime.generator, resolved_config)
    sessions = _read_optional_jsonl(session_history_path)
    _validate_session_history(sessions)
    if session_journal_path.exists():
        recovered = _read_json_mapping(session_journal_path, "session journal")
        if int(recovered.get("session_index", -1)) != len(sessions):
            raise ValueError("session journal index does not follow session history")
        recovered["recovered_after_crash"] = True
        _append_jsonl(session_history_path, recovered)
        session_journal_path.unlink()
        sessions.append(recovered)
        _validate_session_history(sessions)
    _cuda_reset(runtime.device)
    session_candidate_seconds = 0.0
    session_native_seconds = 0.0
    session_row_io_seconds = 0.0
    session_artifact_io_seconds = 0.0
    session_index = len(sessions)
    _atomic_write_json(
        session_journal_path,
        _session_snapshot(
            session_index=session_index,
            generated_count=0,
            resumed_count=completed_count,
            candidate_generation_seconds=0.0,
            native_generation_seconds=0.0,
            row_io_seconds=0.0,
            artifact_io_seconds=0.0,
            wall_seconds=time.perf_counter() - invocation_started,
            device=runtime.device,
        ),
    )
    for batch_start in range(0, len(remaining_records), batch_size):
        batch = remaining_records[batch_start : batch_start + batch_size]
        sample_ids = [str(row["sample_id"]) for row in batch]
        target_z0 = torch.stack([_as_float_tensor(row["z"]) for row in batch]).to(runtime.device)
        x_init = make_x_init_for_sample_ids(
            sample_ids,
            sampling_seed,
            image_size,
            runtime.device,
            target_z0.dtype,
            channels=channels,
        )
        candidate_started = _algorithm_timer_start(runtime.device)
        candidate_result = execute_guidance_mode(
            config=resolved_config,
            generator=runtime.generator,
            codec=runtime.codec,
            e0=runtime.e0,
            x_init=x_init,
            target_z0=target_z0,
            schedule=schedule,
        )
        candidate_seconds = _algorithm_timer_stop(runtime.device, candidate_started)
        if mode == "native":
            native_result = candidate_result
            native_seconds = 0.0
        else:
            native_started = _algorithm_timer_start(runtime.device)
            native_result = execute_matched_native(generator=runtime.generator, x_init=x_init)
            native_seconds = _algorithm_timer_stop(runtime.device, native_started)
        session_candidate_seconds += candidate_seconds
        session_native_seconds += native_seconds
        source_io_seconds = 0.0
        source_images = None
        if runtime.edev is not None:
            source_io_started = time.perf_counter()
            source_images = _load_source_images(
                [str(row["source"]) for row in batch],
                int(resolved_config.get("pixel_image_size", 256)),
                runtime.device,
                target_z0.dtype,
            )
            source_io_seconds = time.perf_counter() - source_io_started
        with torch.no_grad():
            generated = runtime.codec.decode(candidate_result.latent)
            native_images = generated if mode == "native" else runtime.codec.decode(native_result.latent)
            candidate_embedding = runtime.e0(normalize_for_e0(generated))["embedding"]
            native_embedding = runtime.e0(normalize_for_e0(native_images))["embedding"]
            candidate_cosine = F.cosine_similarity(candidate_embedding, target_z0, dim=1)
            native_cosine = F.cosine_similarity(native_embedding, target_z0, dim=1)
            edev_cosine = None
            if runtime.edev is not None:
                if source_images is None:
                    raise RuntimeError("calibration source images were not loaded")
                edev_generated = runtime.edev(normalize_for_e0(generated))["embedding"]
                edev_source = runtime.edev(normalize_for_e0(source_images))["embedding"]
                edev_cosine = F.cosine_similarity(edev_generated, edev_source, dim=1)
        tensors_to_check = [generated, native_images, candidate_cosine, native_cosine]
        if edev_cosine is not None:
            tensors_to_check.append(edev_cosine)
        if any(not torch.isfinite(tensor).all() for tensor in tensors_to_check):
            raise FloatingPointError("guidance runner produced non-finite images or cosine values")
        batch_semigroup = {}
        if mode == "semigroup":
            batch_semigroup = {
                row["sample_id"]: row["splits"]
                for row in _semigroup_batch_rows(
                    result=candidate_result,
                    sample_ids=sample_ids,
                    codec=runtime.codec,
                    e0=runtime.e0,
                    direct_images=generated,
                )
            }
        for local_index, sample_id in enumerate(sample_ids):
            ordinal = completed_count + batch_start + local_index
            binding = expected_bindings[ordinal]
            row_io_started = time.perf_counter()
            _atomic_save_image(generated[local_index], Path(binding["generated"]))
            if mode != "native":
                _atomic_save_image(native_images[local_index], Path(binding["native"]))
            per_sample_candidate_seconds = candidate_seconds / len(sample_ids)
            per_sample_native_seconds = native_seconds / len(sample_ids)
            row = {
                **binding,
                "shard": int(shard_index),
                "candidate_cosine": float(candidate_cosine[local_index].detach().cpu()),
                "native_cosine": float(native_cosine[local_index].detach().cpu()),
                "cosine": float(candidate_cosine[local_index].detach().cpu()),
                "candidate_nfe": int(candidate_result.nfe),
                "native_nfe": int(native_result.nfe),
                "candidate_trace": list(candidate_result.diagnostics["flow_map_trace"]),
                "native_trace": list(native_result.diagnostics["flow_map_trace"]),
                "candidate_generation_seconds": per_sample_candidate_seconds,
                "native_generation_seconds": per_sample_native_seconds,
                "generation_seconds": per_sample_candidate_seconds + per_sample_native_seconds,
                "io_seconds": 0.0,
                "sample_id": sample_id,
                "route_diagnostics": _per_sample_diagnostics(
                    candidate_result.diagnostics, local_index, len(sample_ids)
                ),
            }
            if edev_cosine is not None:
                row["edev_cosine"] = float(edev_cosine[local_index].detach().cpu())
            if mode == "semigroup":
                row["semigroup"] = batch_semigroup[sample_id]
            source_io_per_sample = source_io_seconds / len(sample_ids)
            row["io_seconds"] = time.perf_counter() - row_io_started + source_io_per_sample
            committed_this_session = batch_start + local_index + 1
            _atomic_write_json(
                session_journal_path,
                _session_snapshot(
                    session_index=session_index,
                    generated_count=committed_this_session,
                    resumed_count=completed_count,
                    candidate_generation_seconds=session_candidate_seconds,
                    native_generation_seconds=session_native_seconds,
                    row_io_seconds=session_row_io_seconds + float(row["io_seconds"]),
                    artifact_io_seconds=session_artifact_io_seconds,
                    wall_seconds=time.perf_counter() - invocation_started,
                    device=runtime.device,
                ),
            )
            _append_jsonl(per_sample_path, row)
            session_row_io_seconds += time.perf_counter() - row_io_started + source_io_per_sample
            _atomic_write_json(
                session_journal_path,
                _session_snapshot(
                    session_index=session_index,
                    generated_count=committed_this_session,
                    resumed_count=completed_count,
                    candidate_generation_seconds=session_candidate_seconds,
                    native_generation_seconds=session_native_seconds,
                    row_io_seconds=session_row_io_seconds,
                    artifact_io_seconds=session_artifact_io_seconds,
                    wall_seconds=time.perf_counter() - invocation_started,
                    device=runtime.device,
                ),
            )

    generated_count = len(remaining_records)
    final_rows = _read_optional_jsonl(per_sample_path)
    if resume_remaining_ids(expected_ids, final_rows):
        raise RuntimeError("guidance run ended before every deterministic shard sample was written")
    _validate_resume_rows(final_rows, expected_bindings)
    artifact_io_started = time.perf_counter()
    if mode == "semigroup":
        _atomic_write_json(
            semigroup_path,
            {
                "mode": mode,
                "split_times": [
                    float(value) for value in resolved_config.get("split_times", (0.25, 0.5, 0.75))
                ],
                "rows": [
                    {"sample_id": row["sample_id"], "splits": row["semigroup"]}
                    for row in final_rows
                ],
            },
        )
    contact_manifest = None
    if contact_enabled:
        contact_manifest = _write_contact_sheets(
            final_rows,
            output,
            rows_per_page=_positive_int(
                resolved_config.get("contact_sheet_rows", 8), "contact_sheet_rows"
            ),
            tile_size=_positive_int(
                resolved_config.get("contact_sheet_tile_size", 128), "contact_sheet_tile_size"
            ),
        )
    session_artifact_io_seconds += time.perf_counter() - artifact_io_started
    session = _session_snapshot(
        session_index=session_index,
        generated_count=generated_count,
        resumed_count=completed_count,
        candidate_generation_seconds=session_candidate_seconds,
        native_generation_seconds=session_native_seconds,
        row_io_seconds=session_row_io_seconds,
        artifact_io_seconds=session_artifact_io_seconds,
        wall_seconds=time.perf_counter() - invocation_started,
        device=runtime.device,
    )
    _atomic_write_json(session_journal_path, session)
    _append_jsonl(session_history_path, session)
    session_journal_path.unlink()
    sessions = _read_optional_jsonl(session_history_path)
    _validate_session_history(sessions)
    timing = _aggregate_timing(final_rows, sessions, completed_count, generated_count)
    max_memory = aggregate_session_memory(sessions)
    candidate_nfe = _single_row_value(final_rows, "candidate_nfe")
    native_nfe = _single_row_value(final_rows, "native_nfe")
    cosine = {
        "candidate_e0_target": _finite_summary(
            [float(row["candidate_cosine"]) for row in final_rows]
        ),
        "native_e0_target": _finite_summary([float(row["native_cosine"]) for row in final_rows]),
    }
    if all("edev_cosine" in row for row in final_rows):
        cosine["candidate_edev_source"] = _finite_summary(
            [float(row["edev_cosine"]) for row in final_rows]
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "checkpoint": {
            "path": str(runtime.checkpoint_path),
            "sha256": runtime.checkpoint_sha256,
            **runtime.checkpoint_state,
        },
        "sample_count": len(expected_ids),
        "sample_id_sha256": _sample_id_digest(expected_ids),
        "sample_id_manifest": str(sample_manifest_path),
        "cosine": cosine,
        "shard": {"index": int(shard_index), "count": int(num_shards)},
        "nfe": {"candidate": candidate_nfe, "matched_native": native_nfe},
        "flow_map_traces": [
            {
                "sample_id": row["sample_id"],
                "candidate": row["candidate_trace"],
                "matched_native": row["native_trace"],
            }
            for row in final_rows
        ],
        "timing": timing,
        "max_memory": max_memory,
        "schedule": _json_safe(schedule),
        "config": _json_safe(resolved_config),
        "resume_contract": _json_safe(resume_contract),
        "artifacts": {
            "generated_dir": str(generated_dir),
            "native_dir": str(native_dir) if mode != "native" else None,
            "per_sample_jsonl": str(per_sample_path),
            "semigroup_json": str(semigroup_path) if mode == "semigroup" else None,
            "resume_contract": str(resume_contract_path),
            "session_history": str(session_history_path),
            "contact_sheet_manifest": None
            if contact_manifest is None
            else str(output / "contact_sheet_columns.json"),
            "generation_result": str(generation_result_path),
            "run_manifest": str(run_manifest_path),
        },
    }
    _atomic_write_json(generation_result_path, manifest)
    _atomic_write_json(run_manifest_path, manifest)
    return manifest


def run_guidance_from_config(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    shard_index: int = 0,
    num_shards: int = 1,
    explicit_t_cut: float | None = None,
) -> dict[str, Any]:
    resolved = validate_guidance_config(config)
    from safa.data.feature_dataset import FeatureAlignedAffectNet
    from safa.training.transforms import generator_image_transform
    from safa.utils.device import require_cuda_device
    from safa.utils.hashing import sha256_file

    required = (
        "device",
        "index",
        "features",
        "e0_checkpoint",
        "edev_checkpoint",
        "vae_path",
        "vae_scaling_factor",
    )
    missing = [field for field in required if field not in resolved]
    if missing:
        raise ValueError(f"guidance config missing required fields: {missing!r}")
    checkpoint = Path(str(resolved["checkpoint"]))
    checkpoint_sha256 = sha256_file(checkpoint)
    mode = resolved["mode"]
    if mode in FMRG_MODES:
        schedule = resolve_locked_schedule(
            resolved,
            checkpoint_sha256=checkpoint_sha256,
            explicit_t_cut=explicit_t_cut,
        )
        _validate_semigroup_gate(resolved, checkpoint_sha256, schedule["t_cut"])
        resolved["locked_schedule"] = schedule
    output_manifest = Path(output_dir) / "run_manifest.json"
    output_result = Path(output_dir) / "generation_result.json"
    if output_manifest.exists() or output_result.exists():
        raise FileExistsError(f"refusing to replace completed output: {output_manifest}")

    assets = asset_contract_from_config(resolved)
    dataset = FeatureAlignedAffectNet(
        resolved["index"],
        resolved["features"],
        resolved["e0_checkpoint"],
        transform=generator_image_transform(int(resolved.get("pixel_image_size", 256))),
    )
    records = _records_from_feature_dataset(dataset, resolved)
    _preflight_existing_resume_contract(
        output_dir=Path(output_dir),
        config=resolved,
        asset_contract=assets,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        records=records,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    device = require_cuda_device(str(resolved["device"]))
    runtime = build_frozen_runtime(
        resolved,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        asset_contract=assets,
    )
    return run_guidance_records(
        config=resolved,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=shard_index,
        num_shards=num_shards,
    )


def _records_from_feature_dataset(dataset, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    all_records = [
        {
            "sample_id": record.sample_id,
            "source": str(record.image_path),
            "z": dataset.features[index],
        }
        for index, record in enumerate(dataset.records)
    ]
    manifest_value = config.get("sample_id_manifest")
    if manifest_value:
        manifest = read_ordered_sample_manifest(str(manifest_value))
        by_id = {row["sample_id"]: row for row in all_records}
        requested = [row["sample_id"] for row in manifest]
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"sample ID manifest is not covered by the feature dataset: {missing!r}")
        all_records = [by_id[sample_id] for sample_id in requested]
    max_samples = config.get("max_samples")
    if max_samples is not None:
        all_records = all_records[: _positive_int(max_samples, "max_samples")]
    return all_records


def _preflight_existing_resume_contract(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    asset_contract: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    records,
    shard_index: int,
    num_shards: int,
) -> None:
    path = output_dir / "resume_contract.json"
    if not path.exists():
        return
    selected = deterministic_shard([dict(row) for row in records], shard_index, num_shards)
    existing = _read_json_mapping(path, "resume contract")
    expected = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "e0": asset_contract["e0"],
        "edev": asset_contract["edev"],
        "vae": asset_contract["vae"],
        "real_index": asset_contract["real_index"],
        "seed": int(config["sampling_seed"]),
        "schedule": config.get("locked_schedule"),
        "mode": config["mode"],
        "config": dict(config),
        "sample_id_sha256": _sample_id_digest(str(row["sample_id"]) for row in selected),
        "shard": {"index": int(shard_index), "count": int(num_shards)},
    }
    actual = {
        "checkpoint_path": existing.get("checkpoint", {}).get("path"),
        "checkpoint_sha256": existing.get("checkpoint", {}).get("sha256"),
        "e0": existing.get("e0"),
        "edev": existing.get("edev"),
        "vae": existing.get("vae"),
        "real_index": existing.get("real_index"),
        "seed": existing.get("seed"),
        "schedule": existing.get("schedule"),
        "mode": existing.get("mode"),
        "config": existing.get("config"),
        "sample_id_sha256": existing.get("sample_id_sha256"),
        "shard": existing.get("shard"),
    }
    if _json_safe(actual) != _json_safe(expected):
        raise ValueError("existing resume contract disagrees before CUDA/model loading")


def _validate_semigroup_gate(config: Mapping[str, Any], checkpoint_hash: str, t_cut: float) -> None:
    report_value = config.get("semigroup_report")
    if not report_value:
        raise ValueError("FMRG guidance requires semigroup_report")
    report = _read_json_mapping(Path(str(report_value)), "semigroup report")
    if report.get("gate_passed") is not True:
        raise ValueError("semigroup report must record gate_passed=true")
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("semigroup report checkpoint SHA256 disagrees with the loaded checkpoint")
    selected = _finite_open_unit(
        report.get("selected_t_cut", report.get("t_cut")), "semigroup report selected t_cut"
    )
    if not math.isclose(selected, t_cut, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("semigroup report selected t_cut disagrees with the locked schedule")


def _semigroup_batch_rows(*, result, sample_ids, codec, e0, direct_images) -> list[dict[str, Any]]:
    report = result.diagnostics["semigroup"]
    rows = [{"sample_id": sample_id, "splits": {}} for sample_id in sample_ids]
    with torch.no_grad():
        direct_embedding = e0(normalize_for_e0(direct_images))["embedding"]
        for split, endpoints in report["split_endpoints"].items():
            split_images = codec.decode(endpoints)
            split_embedding = e0(normalize_for_e0(split_images))["embedding"]
            pixel_l1 = (direct_images - split_images).abs().flatten(1).mean(dim=1)
            mse = (direct_images - split_images).square().flatten(1).mean(dim=1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
            cosine = F.cosine_similarity(direct_embedding, split_embedding, dim=1)
            residual = report["residuals"][split]
            for index, row in enumerate(rows):
                row["splits"][str(float(split))] = {
                    "latent_residual": float(residual[index].detach().cpu()),
                    "decoded_pixel_l1": float(pixel_l1[index].detach().cpu()),
                    "decoded_psnr": float(psnr[index].detach().cpu()),
                    "endpoint_e0_cosine": float(cosine[index].detach().cpu()),
                }
    return rows


def _validate_generation_records(records: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(records):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"generation record {index} requires a non-empty string sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate generation sample_id: {sample_id!r}")
        seen.add(sample_id)
        if "source" not in row or "z" not in row:
            raise ValueError(f"generation record {sample_id!r} requires source and z")


def _expected_row_bindings(selected, generated_dir: Path, native_dir: Path, mode: str):
    bindings = []
    for ordinal, record in enumerate(selected):
        sample_id = str(record["sample_id"])
        filename = f"{ordinal:08d}__{_safe_sample_id(sample_id)}.png"
        generated = generated_dir / filename
        native = generated if mode == "native" else native_dir / filename
        bindings.append(
            {
                "ordinal": ordinal,
                "sample_id": sample_id,
                "source": str(record["source"]),
                "generated": str(generated),
                "native": str(native),
                "mode": mode,
            }
        )
    return bindings


def _validate_resume_rows(rows, expected_bindings) -> None:
    if len(rows) > len(expected_bindings):
        raise ValueError("resume rows exceed the deterministic shard")
    for index, row in enumerate(rows):
        binding = expected_bindings[index]
        for field, expected in binding.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"resume row binding mismatch at ordinal {index}: "
                    f"{field}={row.get(field)!r} expected={expected!r}"
                )
        for field in ("candidate_nfe", "native_nfe", "candidate_trace", "native_trace"):
            if field not in row:
                raise ValueError(f"resume row binding is missing required field {field!r}")
        if len(row["candidate_trace"]) != int(row["candidate_nfe"]):
            raise ValueError(f"resume row candidate trace/NFE mismatch at ordinal {index}")
        if len(row["native_trace"]) != int(row["native_nfe"]):
            raise ValueError(f"resume row native trace/NFE mismatch at ordinal {index}")
        for path_field in ("generated", "native"):
            if not Path(str(row[path_field])).is_file():
                raise FileNotFoundError(
                    f"resume row binding points to missing {path_field} PNG: {row[path_field]}"
                )


def _validate_owned_png_state(
    *, generated_dir, native_dir, expected_bindings, completed_count: int, mode: str
) -> None:
    actual_generated = {path.resolve() for path in generated_dir.iterdir() if path.is_file()}
    actual_native = (
        set()
        if mode == "native"
        else {path.resolve() for path in native_dir.iterdir() if path.is_file()}
    )
    completed_generated = {
        Path(binding["generated"]).resolve() for binding in expected_bindings[:completed_count]
    }
    completed_native = (
        set()
        if mode == "native"
        else {Path(binding["native"]).resolve() for binding in expected_bindings[:completed_count]}
    )
    missing = (completed_generated - actual_generated) | (completed_native - actual_native)
    if missing:
        raise FileNotFoundError(f"completed resume row is missing owned PNGs: {sorted(map(str, missing))!r}")
    allowed_orphan_generated = set()
    allowed_orphan_native = set()
    if completed_count < len(expected_bindings):
        binding = expected_bindings[completed_count]
        allowed_orphan_generated.add(Path(binding["generated"]).resolve())
        if mode != "native":
            allowed_orphan_native.add(Path(binding["native"]).resolve())
    extra = (actual_generated - completed_generated - allowed_orphan_generated) | (
        actual_native - completed_native - allowed_orphan_native
    )
    if extra:
        raise ValueError(
            "output contains extra PNG/files outside the exact crash orphan: "
            f"{sorted(map(str, extra))!r}"
        )


def _validate_output_entries(output: Path, *, mode: str, contact_sheets: bool) -> None:
    if not output.exists():
        return
    allowed = {
        "generated_images",
        "per_sample.jsonl",
        "run_manifest.json",
        "generation_result.json",
        "sample_id_manifest.jsonl",
        "resume_contract.json",
        "session_history.jsonl",
        "session_journal.json",
    }
    if mode != "native":
        allowed.add("native_images")
    if mode == "semigroup":
        allowed.add("semigroup.json")
    if contact_sheets:
        allowed.update({"contact_sheets", "contact_sheet_columns.json"})
    extras = sorted(path.name for path in output.iterdir() if path.name not in allowed)
    if extras:
        raise ValueError(f"output contains unexpected files: {extras!r}")


def _sample_channels(generator, config: Mapping[str, Any]) -> int:
    generator_config = getattr(generator, "config", None)
    value = getattr(generator_config, "sit_input_channels", None)
    if value is None:
        value = config.get("channels", EXPECTED_MODEL_CONFIG["sit_input_channels"])
    return _positive_int(value, "sample channels")


def _as_float_tensor(value) -> torch.Tensor:
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != 1 or not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim != 1 or not torch.isfinite(tensor).all():
        raise ValueError("target z must be a finite one-dimensional tensor")
    return tensor


def _per_sample_diagnostics(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return float(detached)
        if detached.ndim == 1 and detached.shape[0] == batch_size:
            return float(detached[index])
        return None
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            converted = _per_sample_diagnostics(item, index, batch_size)
            if converted is not None:
                result[str(key)] = converted
        return result
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            return list(value)
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _safe_sample_id(sample_id: str) -> str:
    safe = _SAFE_SAMPLE_ID_RE.sub("_", sample_id.replace("/", "_").replace("\\", "_"))
    return safe.strip("._-") or "sample"


def _tensor_to_pil(image: torch.Tensor):
    from PIL import Image

    tensor = image.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"PNG output requires [3,H,W], got {tuple(tensor.shape)}")
    array = tensor.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _atomic_save_image(image: torch.Tensor, path: Path) -> None:
    _atomic_save_pil(_tensor_to_pil(image), path)


def _atomic_save_pil(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        image.save(temporary, format="PNG")
        with temporary.open("rb") as reader:
            os.fsync(reader.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_source_images(paths, image_size: int, device, dtype) -> torch.Tensor:
    from PIL import Image

    tensors = []
    for path in paths:
        with Image.open(path) as source:
            rgb = source.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            tensor = data.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(255.0)
            tensors.append(tensor)
    return torch.stack(tensors).to(device=device, dtype=dtype)


def _contact_sheets_enabled(config: Mapping[str, Any]) -> bool:
    calibration = str(config.get("phase", "full")) == "calibration"
    value = config.get("contact_sheets")
    if value is None:
        return calibration
    if not isinstance(value, bool):
        raise ValueError(f"contact_sheets must be true or false, got {value!r}")
    if calibration and value is False:
        raise ValueError("calibration requires source/native/candidate contact sheets")
    return value


def _write_contact_sheets(rows, output: Path, *, rows_per_page: int, tile_size: int):
    from PIL import Image

    contact_dir = output / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    expected_pages = math.ceil(len(rows) / rows_per_page)
    expected_names = {f"page_{index:03d}.png" for index in range(expected_pages)}
    extras = sorted(
        path.name for path in contact_dir.iterdir() if path.is_file() and path.name not in expected_names
    )
    if extras:
        raise ValueError(f"contact sheet directory contains unexpected pages: {extras!r}")
    for page_index in range(expected_pages):
        page_rows = rows[page_index * rows_per_page : (page_index + 1) * rows_per_page]
        sheet = Image.new("RGB", (tile_size * 3, tile_size * len(page_rows)))
        for row_index, row in enumerate(page_rows):
            for column, field in enumerate(("source", "native", "generated")):
                with Image.open(row[field]) as image:
                    tile = image.convert("RGB").resize(
                        (tile_size, tile_size), Image.Resampling.BILINEAR
                    )
                sheet.paste(tile, (column * tile_size, row_index * tile_size))
        path = contact_dir / f"page_{page_index:03d}.png"
        _atomic_save_pil(sheet, path)
        pages.append(
            {
                "page_index": page_index,
                "path": str(path),
                "sample_ids": [row["sample_id"] for row in page_rows],
                "ordinals": [int(row["ordinal"]) for row in page_rows],
            }
        )
    manifest = {"columns": ["source", "native", "candidate"], "pages": pages}
    _atomic_write_json(output / "contact_sheet_columns.json", manifest)
    return manifest


def _sample_id_digest(sample_ids: Iterable[str]) -> str:
    payload = "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to replace existing JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        temporary.write_text(
            json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as reader:
            os.fsync(reader.fileno())
        if exclusive and path.exists():
            raise FileExistsError(f"refusing to replace existing JSON: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.tolist() if detached.ndim else float(detached)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cannot serialize a non-finite float")
    return value


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _finite_open_unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and within (0,1), got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and within (0,1), got {value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError(f"{label} must be finite and within (0,1), got {value!r}")
    return parsed


def _float_sequences_equal(left: Any, right: Sequence[float]) -> bool:
    if not isinstance(left, Sequence) or isinstance(left, (str, bytes)) or len(left) != len(right):
        return False
    try:
        return all(
            math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(left, right, strict=True)
        )
    except (TypeError, ValueError):
        return False


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return int(value)


def _digest_path(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"asset digest rejects symlink paths: {path}")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"asset path does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise ValueError(f"asset directory contains no files: {path}")
    for file_path in files:
        if file_path.is_symlink():
            raise ValueError(f"asset digest rejects symlink entries: {file_path}")
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_expected_digest(
    config: Mapping[str, Any], fields: Sequence[str], actual: str, label: str
) -> None:
    values = [str(config[field]) for field in fields if config.get(field)]
    if len(set(values)) > 1:
        raise ValueError(f"{label} expected digest fields disagree: {values!r}")
    if values and values[0] != actual:
        raise ValueError(f"{label} asset digest mismatch: expected={values[0]} actual={actual}")


def _result_with_trace(result: GuidanceResult, trace, mode: str) -> GuidanceResult:
    if len(trace) != int(result.nfe):
        raise RuntimeError(f"{mode} flow-map trace length does not match NFE")
    diagnostics = dict(result.diagnostics)
    diagnostics["mode"] = mode
    diagnostics["flow_map_trace"] = list(trace)
    return GuidanceResult(latent=result.latent, nfe=result.nfe, diagnostics=diagnostics)


def _single_row_value(rows, field: str) -> int:
    values = {int(row[field]) for row in rows}
    if len(values) != 1:
        raise RuntimeError(f"per-sample {field} values disagree: {sorted(values)!r}")
    return values.pop()


def _session_snapshot(
    *,
    session_index: int,
    generated_count: int,
    resumed_count: int,
    candidate_generation_seconds: float,
    native_generation_seconds: float,
    row_io_seconds: float,
    artifact_io_seconds: float,
    wall_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    allocated, reserved = _cuda_peak_memory(device)
    generation_seconds = candidate_generation_seconds + native_generation_seconds
    return {
        "session_index": int(session_index),
        "generated_count": int(generated_count),
        "resumed_count": int(resumed_count),
        "candidate_generation_seconds": float(candidate_generation_seconds),
        "native_generation_seconds": float(native_generation_seconds),
        "generation_seconds": float(generation_seconds),
        "row_io_seconds": float(row_io_seconds),
        "artifact_io_seconds": float(artifact_io_seconds),
        "io_seconds": float(row_io_seconds + artifact_io_seconds),
        "wall_seconds": float(wall_seconds),
        "max_memory": {
            "allocated_bytes": int(allocated),
            "reserved_bytes": int(reserved),
        },
    }


def _validate_session_history(sessions) -> None:
    required = {
        "session_index",
        "generated_count",
        "resumed_count",
        "candidate_generation_seconds",
        "native_generation_seconds",
        "generation_seconds",
        "row_io_seconds",
        "artifact_io_seconds",
        "io_seconds",
        "wall_seconds",
        "max_memory",
    }
    for index, session in enumerate(sessions):
        if not isinstance(session, Mapping) or not required.issubset(session):
            raise ValueError(f"invalid session history row at index {index}")
        if int(session["session_index"]) != index:
            raise ValueError(f"session history index mismatch at row {index}")
        memory = session["max_memory"]
        if not isinstance(memory, Mapping) or not {
            "allocated_bytes",
            "reserved_bytes",
        }.issubset(memory):
            raise ValueError(f"invalid session memory record at index {index}")


def aggregate_session_memory(sessions) -> dict[str, int]:
    if not sessions:
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": max(int(item["max_memory"]["allocated_bytes"]) for item in sessions),
        "reserved_bytes": max(int(item["max_memory"]["reserved_bytes"]) for item in sessions),
    }


def _aggregate_timing(rows, sessions, resumed_count: int, generated_count: int) -> dict[str, Any]:
    candidate = sum(float(row["candidate_generation_seconds"]) for row in rows)
    native = sum(float(row["native_generation_seconds"]) for row in rows)
    generation = candidate + native
    io_seconds = sum(float(session.get("io_seconds", 0.0)) for session in sessions)
    wall_seconds = sum(float(session["wall_seconds"]) for session in sessions)
    return {
        "candidate_generation_seconds": candidate,
        "native_generation_seconds": native,
        "generation_seconds": generation,
        "io_seconds": io_seconds,
        "wall_seconds": wall_seconds,
        "images_per_second": float(len(rows) / generation) if generation > 0.0 else 0.0,
        "generated_this_invocation": generated_count,
        "resumed_count": resumed_count,
        "session_count": len(sessions),
    }


def _finite_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("metric summary requires non-empty finite values")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
        "std": float(tensor.std(unbiased=False)),
        "p05": float(torch.quantile(tensor, 0.05)),
        "p95": float(torch.quantile(tensor, 0.95)),
    }


def _cuda_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _algorithm_timer_start(device: torch.device) -> float:
    _cuda_sync(device)
    return time.perf_counter()


def _algorithm_timer_stop(device: torch.device, started: float) -> float:
    _cuda_sync(device)
    return time.perf_counter() - started


def _cuda_peak_memory(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
