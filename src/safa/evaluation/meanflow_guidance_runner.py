from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

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
from safa.evaluation.r8_arm_contracts import (
    require_arm_config_digest,
)
from safa.evaluation.r9_determinism import (
    R9_ATTENTION_BACKEND,
    apply_r9_strict_cuda_determinism,
    assert_r9_strict_cuda_determinism,
    canonical_guidance_arm_config_digest,
    is_r9_guidance_config,
    validate_r9_execution_config,
)
from safa.evaluation.r9_semigroup_contracts import (
    canonical_r9_schedule_contract_sha256,
    validate_r9_locked_schedule_bindings,
    validate_r9_semigroup_preflight_config,
)
from safa.utils.sampling import make_x_init_for_sample_ids


EXPECTED_CHECKPOINT_PATH = "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last_nopretrained.pt"
EXPECTED_E0_CHECKPOINT_PATH = "artifacts/checkpoints/e0_medium_v1/best.pt"
EXPECTED_EDEV_CHECKPOINT_PATH = "artifacts/checkpoints/e0_resnet18/best.pt"
EXPECTED_VAE_PATH = "artifacts/checkpoints/external/sd-vae-ft-ema"
EXPECTED_VAE_SCALING_FACTOR = 0.18215
EXPECTED_STAGE = "stage2"
EXPECTED_STAGE_EPOCH = 1652
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
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
    {
        "native",
        "semigroup",
        "official_head_current_xt",
        "paper_algorithm_split",
        "initial_noise",
    }
)
_MODE_ALIASES = {"noise_oracle": "initial_noise"}
FMRG_MODES = frozenset({"official_head_current_xt", "paper_algorithm_split"})
_SAFE_SAMPLE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
R9_GUIDANCE_INTERVAL_IDS = ("I1", "I2", "I3")
R9_GUIDANCE_INTERVAL_CONTRACT_FIELD = "r9_guidance_interval_contract"
R9_PHASE_CONTRACT_FIELD = "r9_phase_contract"
R9_PHASES = (
    "semigroup",
    "preflight",
    "resource_smoke",
    "diagnose",
    "calibrate",
    "confirm512",
    "full",
)
R9_EDEV_PHASES = (
    "resource_smoke",
    "diagnose",
    "calibrate",
    "confirm512",
    "full",
)


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
    target_features_path: Path | None = None
    target_features_digest: str = ""
    feature_source: str = ""
    input_sample_manifest_path: Path | None = None
    input_sample_manifest_sha256: str = ""
    input_sample_manifest_id_sha256: str = ""
    input_sample_manifest_count: int = 0
    heldout_e1: dict[str, str] | None = None
    heldout_e2: dict[str, str] | None = None


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
            raise ValueError(
                f"checkpoint model_config.{field} must be {expected!r}, got {actual!r}"
            )
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
    r9_attention_backend: str | None = None,
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
    if r9_attention_backend is not None:
        if r9_attention_backend != R9_ATTENTION_BACKEND:
            raise ValueError(
                "R9 generator attention backend must be locked to 'native'"
            )
        model_config["attention_backend"] = str(r9_attention_backend)
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
    if r9_attention_backend is not None:
        requested = getattr(generator, "requested_attention_backend", None)
        resolved = getattr(generator, "attention_backend", None)
        if requested != r9_attention_backend or resolved != r9_attention_backend:
            raise RuntimeError(
                "MeanFlow generator did not honor the locked attention backend: "
                f"requested={requested!r} resolved={resolved!r} "
                f"expected={r9_attention_backend!r}"
            )
        metadata["attention_backend_requested"] = requested
        metadata["attention_backend_resolved"] = resolved
    return generator, metadata


def validate_guidance_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("guidance config must be a mapping")
    resolved = dict(config)
    allowed_heldout_fields = {
        "heldout_e1_checkpoint",
        "heldout_e1_sha256",
        "heldout_e2_checkpoint",
        "heldout_e2_sha256",
    }
    present_forbidden = sorted(
        str(field)
        for field in resolved
        if re.search(r"(^|_)e[12]($|_)", str(field).lower())
        and str(field) not in allowed_heldout_fields
    )
    if present_forbidden:
        raise ValueError(
            f"guidance runner accepts only locked heldout E1/E2 assets: {present_forbidden!r}"
        )
    checkpoint = str(resolved.get("checkpoint", ""))
    if checkpoint != EXPECTED_CHECKPOINT_PATH:
        raise ValueError(
            f"guidance checkpoint must be exactly {EXPECTED_CHECKPOINT_PATH!r}, got {checkpoint!r}"
        )
    if resolved.get("checkpoint_model") != "ema":
        raise ValueError("guidance checkpoint_model must be 'ema'")
    if resolved.get("transport_condition") != "learned_null_condition":
        raise ValueError("guidance transport_condition must be learned_null_condition")
    for field, expected in (
        ("e0_checkpoint", EXPECTED_E0_CHECKPOINT_PATH),
        ("edev_checkpoint", EXPECTED_EDEV_CHECKPOINT_PATH),
        ("vae_path", EXPECTED_VAE_PATH),
        ("vae_scaling_factor", EXPECTED_VAE_SCALING_FACTOR),
    ):
        if resolved.get(field) != expected:
            raise ValueError(
                f"guidance config {field} must be {expected!r}, got {resolved.get(field)!r}"
            )
    required_digests = (
        "checkpoint_sha256",
        "e0_sha256",
        "edev_sha256",
        "vae_digest",
        "index_sha256",
        "features_digest",
        "sample_id_manifest_sha256",
        "heldout_e1_sha256",
        "heldout_e2_sha256",
    )
    for field in required_digests:
        _require_sha256(resolved.get(field), field)
    required_paths = (
        "index",
        "features",
        "sample_id_manifest",
        "heldout_e1_checkpoint",
        "heldout_e2_checkpoint",
    )
    missing_paths = [field for field in required_paths if not resolved.get(field)]
    if missing_paths:
        raise ValueError(
            f"guidance config missing locked asset paths: {missing_paths!r}"
        )
    if resolved.get("feature_source") != "cached_features":
        raise ValueError("guidance config feature_source must be 'cached_features'")
    mode = str(resolved.get("mode", resolved.get("route", "")))
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"guidance mode must be one of {sorted(SUPPORTED_MODES)}, got {mode!r}"
        )
    resolved["mode"] = mode
    resolved.pop("route", None)
    if mode == "official_head_current_xt":
        required_algorithm_settings = (
            "sample_mode",
            "optimization_mode",
            "num_optim_iters",
            "step_size",
        )
        missing_algorithm_settings = [
            field for field in required_algorithm_settings if field not in resolved
        ]
        if missing_algorithm_settings:
            raise ValueError(
                "guidance mode 'official_head_current_xt' requires explicit algorithm settings: "
                f"{missing_algorithm_settings!r}"
            )
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
            raise ValueError(
                f"guidance config {field} must be {expected!r}, got {resolved[field]!r}"
            )
    if is_r9_guidance_config(resolved):
        execution_contract = validate_r9_execution_config(resolved)
        resolved.update(execution_contract)
        phase_contract = validate_r9_phase_contract(resolved)
        resolved[R9_PHASE_CONTRACT_FIELD] = phase_contract
        interval_contract = validate_r9_interval_guidance_config(resolved, mode=mode)
        if interval_contract is not None:
            resolved[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD] = interval_contract
        if mode == "semigroup":
            validate_r9_semigroup_preflight_config(resolved)
    else:
        validate_r9_interval_guidance_config(resolved, mode=mode)
    return resolved


def finalize_effective_guidance_config(
    validated_config: Mapping[str, Any],
    *,
    locked_schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the exact config consumed by generation and bind its arm digest."""
    effective = dict(validated_config)
    mode = str(effective["mode"])
    if mode in FMRG_MODES:
        if locked_schedule is None:
            raise ValueError("FMRG effective config requires a locked schedule")
        effective["locked_schedule"] = dict(locked_schedule)
    elif locked_schedule is not None or "locked_schedule" in effective:
        raise ValueError("non-FMRG effective config must not contain a locked schedule")
    computed = canonical_guidance_arm_config_digest(effective)
    declared = effective.get("arm_config_sha256")
    if declared is not None and require_arm_config_digest(declared) != computed:
        raise ValueError(
            "declared arm config SHA256 disagrees with the effective guidance config"
        )
    effective["arm_config_sha256"] = computed
    return effective


def resolve_frozen_effective_guidance_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one YAML-owned config exactly as the generation entry point does."""
    validated = validate_guidance_config(config)
    schedule = None
    if validated["mode"] in FMRG_MODES:
        checkpoint_sha256 = str(validated["checkpoint_sha256"])
        schedule = resolve_locked_schedule(
            validated,
            checkpoint_sha256=checkpoint_sha256,
            explicit_t_cut=None,
        )
        _validate_semigroup_gate(
            validated,
            checkpoint_sha256,
            float(schedule["t_cut"]),
            str(schedule["semigroup_sample_id_manifest_sha256"]),
        )
    return finalize_effective_guidance_config(
        validated,
        locked_schedule=schedule,
    )


def validate_r9_phase_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if not is_r9_guidance_config(config):
        if R9_PHASE_CONTRACT_FIELD in config:
            raise ValueError("R9 phase contract requires the R9 experiment contract")
        raise ValueError("R9 phase contract requires the R9 experiment contract")
    if "phase" not in config:
        raise ValueError("R9 guidance config requires an explicit phase")
    phase = config["phase"]
    if not isinstance(phase, str) or phase not in R9_PHASES:
        raise ValueError(f"R9 guidance phase must be one of {list(R9_PHASES)!r}")
    edev_required = phase in R9_EDEV_PHASES
    contract = {
        "schema_version": 1,
        "phase": phase,
        "edev_required": edev_required,
        "required_per_sample_edev_fields": (
            ["edev_cosine", "native_edev_cosine"] if edev_required else []
        ),
        "required_summary_edev_fields": (
            ["candidate_edev_source", "native_edev_source"] if edev_required else []
        ),
    }
    declared = config.get(R9_PHASE_CONTRACT_FIELD)
    if declared is not None and declared != contract:
        raise ValueError("R9 phase contract disagrees with the effective config")
    return contract


def _edev_required_for_config(config: Mapping[str, Any]) -> bool:
    if is_r9_guidance_config(config):
        contract = validate_r9_phase_contract(config)
        if config.get(R9_PHASE_CONTRACT_FIELD) != contract:
            raise ValueError(
                "R9 execution requires the validated phase contract from config preflight"
            )
        return bool(contract["edev_required"])
    if R9_PHASE_CONTRACT_FIELD in config:
        raise ValueError("R9 phase contract requires the R9 experiment contract")
    return str(config.get("phase", "full")) == "calibration"


def validate_r9_interval_guidance_config(
    config: Mapping[str, Any], *, mode: str | None = None
) -> dict[str, Any] | None:
    resolved_mode = _MODE_ALIASES.get(
        str(mode if mode is not None else config.get("mode", config.get("route", ""))),
        str(mode if mode is not None else config.get("mode", config.get("route", ""))),
    )
    interval_fields = {
        "active_guidance_intervals",
        "collect_interval_diagnostics",
        R9_GUIDANCE_INTERVAL_CONTRACT_FIELD,
    }
    present_fields = sorted(field for field in interval_fields if field in config)
    if not is_r9_guidance_config(config):
        if present_fields:
            raise ValueError(
                f"R9 interval guidance fields require the R9 experiment contract: {present_fields!r}"
            )
        return None
    if resolved_mode not in FMRG_MODES:
        if present_fields:
            raise ValueError(
                f"R9 interval guidance fields are invalid for mode {resolved_mode!r}"
            )
        return None

    missing = [
        field
        for field in ("active_guidance_intervals", "collect_interval_diagnostics")
        if field not in config
    ]
    if missing:
        raise ValueError(
            f"R9 FMRG config is missing interval guidance fields: {missing!r}"
        )
    active_value = config["active_guidance_intervals"]
    if not isinstance(active_value, list):
        raise ValueError("R9 active_guidance_intervals must be a YAML list")
    if any(not isinstance(interval_id, str) for interval_id in active_value):
        raise ValueError("R9 active_guidance_intervals must contain only interval IDs")
    if len(set(active_value)) != len(active_value):
        raise ValueError("R9 active_guidance_intervals must not contain duplicates")
    unknown = sorted(set(active_value) - set(R9_GUIDANCE_INTERVAL_IDS))
    if unknown:
        raise ValueError(
            f"R9 active_guidance_intervals contains unknown IDs: {unknown!r}"
        )
    canonical_active = [
        interval_id
        for interval_id in R9_GUIDANCE_INTERVAL_IDS
        if interval_id in active_value
    ]
    if active_value != canonical_active:
        raise ValueError(
            "R9 active_guidance_intervals must follow canonical I1/I2/I3 order"
        )
    collect = config["collect_interval_diagnostics"]
    if not isinstance(collect, bool):
        raise ValueError("R9 collect_interval_diagnostics must be a boolean")
    step_size = config.get("step_size")
    if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
        raise ValueError("R9 FMRG config requires an explicit numeric step_size")
    if not math.isfinite(float(step_size)) or float(step_size) <= 0.0:
        raise ValueError("R9 FMRG step_size must be positive and finite")

    if resolved_mode == "official_head_current_xt":
        if config.get("sample_mode") != "flow_map2":
            raise ValueError(
                "R9 official_head_current_xt requires sample_mode='flow_map2'"
            )
        if config.get("optimization_mode") != "paper_normalized_direct_autograd":
            raise ValueError(
                "R9 official_head_current_xt requires "
                "optimization_mode='paper_normalized_direct_autograd'"
            )
        if config.get("num_optim_iters") != 1:
            raise ValueError("R9 official_head_current_xt requires num_optim_iters=1")
    else:
        ignored = sorted(
            field
            for field in ("sample_mode", "optimization_mode", "num_optim_iters")
            if field in config
        )
        if ignored:
            raise ValueError(
                f"R9 paper_algorithm_split rejects unused official algorithm fields: {ignored!r}"
            )

    expected_algorithm_trace, expected_diagnostic_trace = _expected_r9_interval_traces(
        resolved_mode,
        canonical_active,
        collect=collect,
    )
    contract = {
        "schema_version": 1,
        "mode": resolved_mode,
        "active_guidance_intervals": canonical_active,
        "collect_interval_diagnostics": collect,
        "expected_algorithm_nfe": len(expected_algorithm_trace),
        "expected_diagnostic_nfe": len(expected_diagnostic_trace),
        "expected_algorithm_trace": expected_algorithm_trace,
        "expected_diagnostic_trace": expected_diagnostic_trace,
    }
    declared = config.get(R9_GUIDANCE_INTERVAL_CONTRACT_FIELD)
    if declared is not None and declared != contract:
        raise ValueError(
            "R9 guidance interval contract disagrees with the effective config"
        )
    return contract


def _expected_r9_interval_traces(
    mode: str,
    active_guidance_intervals: Sequence[str],
    *,
    collect: bool,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    active = set(active_guidance_intervals)
    algorithm_trace: list[dict[str, float | str]] = []
    diagnostic_trace: list[dict[str, float | str]] = []
    for interval_id, t, s in (
        ("I1", 1.0, 0.75),
        ("I2", 0.75, 0.5),
        ("I3", 0.5, 0.25),
    ):
        if mode == "official_head_current_xt":
            if interval_id in active:
                algorithm_trace.append({"t": t, "r": 0.0, "kind": mode})
            algorithm_trace.append({"t": t, "r": s, "kind": mode})
        elif mode == "paper_algorithm_split":
            algorithm_trace.append({"t": t, "r": s, "kind": mode})
            if interval_id in active:
                algorithm_trace.append({"t": s, "r": 0.0, "kind": mode})
        else:
            raise ValueError(f"unsupported R9 interval guidance mode {mode!r}")
        if collect:
            diagnostic_trace.extend(
                (
                    {"t": t, "r": 0.0, "kind": "interval_diagnostic"},
                    {"t": s, "r": 0.0, "kind": "interval_diagnostic"},
                )
            )
            if interval_id in active:
                diagnostic_trace.append(
                    {"t": s, "r": 0.0, "kind": "interval_diagnostic"}
                )
    algorithm_trace.extend(
        (
            {"t": 0.25, "r": 0.125, "kind": mode},
            {"t": 0.125, "r": 0.0, "kind": mode},
        )
    )
    return algorithm_trace, diagnostic_trace


def _prepare_sharded_guidance_config(
    config: Mapping[str, Any],
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    _validate_shard_coordinates(shard_index, num_shards)
    cache_path, allowed_roots = _resolve_asset_digest_cache(
        config, num_shards=num_shards
    )
    resolved = validate_guidance_config(config)
    if cache_path is None:
        return resolved
    resolved["asset_digest_cache"] = str(cache_path)
    explicit_root = config.get("asset_digest_cache_root")
    if explicit_root:
        resolved["asset_digest_cache_root"] = str(
            _repository_relative_candidate(explicit_root).resolve(strict=False)
        )
    if num_shards > 1:
        _register_shard_asset_cache_contract(
            resolved,
            cache_path=cache_path,
            allowed_roots=allowed_roots,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    return resolved


def _validate_shard_coordinates(shard_index: int, num_shards: int) -> None:
    if (
        isinstance(num_shards, bool)
        or not isinstance(num_shards, int)
        or num_shards <= 0
    ):
        raise ValueError(f"num_shards must be a positive integer, got {num_shards!r}")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < num_shards
    ):
        raise ValueError(
            f"shard_index must be in [0,{num_shards}), got {shard_index!r}"
        )


def _resolve_asset_digest_cache(
    config: Mapping[str, Any], *, num_shards: int
) -> tuple[Path | None, tuple[Path, ...]]:
    cache_value = config.get("asset_digest_cache")
    if not cache_value or not str(cache_value).strip():
        if num_shards > 1:
            raise ValueError(
                "num_shards > 1 requires a non-empty shared asset_digest_cache"
            )
        return None, (REPOSITORY_ROOT,)

    repository_root = REPOSITORY_ROOT.resolve()
    allowed_roots = [repository_root]
    explicit_root_value = config.get("asset_digest_cache_root")
    if explicit_root_value:
        explicit_root = _repository_relative_candidate(explicit_root_value)
        if explicit_root.is_symlink():
            raise ValueError(
                f"asset_digest_cache_root must not be a symlink: {explicit_root}"
            )
        if explicit_root.exists() and not explicit_root.is_dir():
            raise ValueError(
                f"asset_digest_cache_root must be a directory: {explicit_root}"
            )
        allowed_roots.append(explicit_root.resolve(strict=False))

    cache_candidate = _repository_relative_candidate(cache_value)
    _require_asset_cache_target(
        cache_candidate, tuple(allowed_roots), "asset_digest_cache"
    )
    cache_path = cache_candidate.resolve(strict=False)
    if cache_path.exists() and not cache_path.is_file():
        raise ValueError(f"asset_digest_cache must be a file path: {cache_path}")
    return cache_path, tuple(allowed_roots)


def _repository_relative_candidate(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return Path(os.path.abspath(path))


def _require_asset_cache_target(
    path: Path, allowed_roots: Sequence[Path], label: str
) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=False)
    matched_root = None
    for root in allowed_roots:
        root_resolved = root.resolve(strict=False)
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        matched_root = root_resolved
        break
    if matched_root is None:
        raise ValueError(
            f"{label} must be inside the repository or an explicitly allowed root"
        )

    relative = resolved.relative_to(matched_root)
    cursor = matched_root
    if cursor.is_symlink():
        raise ValueError(f"{label} allowed root must not be a symlink: {cursor}")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} contains a symlink path component: {cursor}")


def _register_shard_asset_cache_contract(
    config: Mapping[str, Any],
    *,
    cache_path: Path,
    allowed_roots: Sequence[Path],
    shard_index: int,
    num_shards: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path = cache_path.with_name(f"{cache_path.name}.shards.json")
    lock_path = cache_path.with_name(f"{cache_path.name}.lock")
    for path, label in (
        (cache_path, "asset_digest_cache"),
        (contract_path, "shard asset cache contract"),
        (lock_path, "asset digest cache lock"),
    ):
        _require_asset_cache_target(path, allowed_roots, label)

    contract = {
        "num_shards": int(num_shards),
        "config": _json_safe(dict(config)),
    }
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if contract_path.exists():
                payload = _read_json_mapping(
                    contract_path, "shard asset cache contract"
                )
                if payload.get("contract") != contract:
                    raise ValueError("existing shard asset cache contract disagrees")
                registered = payload.get("registered_shards")
                if not isinstance(registered, list):
                    raise ValueError("invalid shard asset cache contract registrations")
                registered_shards = {int(value) for value in registered}
                if any(not 0 <= value < num_shards for value in registered_shards):
                    raise ValueError("invalid shard asset cache contract registrations")
            else:
                registered_shards = set()
            registered_shards.add(int(shard_index))
            _atomic_write_json(
                contract_path,
                {
                    "schema_version": 1,
                    "contract": contract,
                    "registered_shards": sorted(registered_shards),
                },
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def asset_contract_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "checkpoint",
        "checkpoint_sha256",
        "e0_checkpoint",
        "e0_sha256",
        "edev_checkpoint",
        "edev_sha256",
        "vae_path",
        "vae_digest",
        "vae_scaling_factor",
        "index",
        "index_sha256",
        "features",
        "features_digest",
        "sample_id_manifest",
        "sample_id_manifest_sha256",
        "heldout_e1_checkpoint",
        "heldout_e1_sha256",
        "heldout_e2_checkpoint",
        "heldout_e2_sha256",
    )
    missing = [field for field in required if not config.get(field)]
    if missing:
        raise ValueError(
            f"guidance asset contract missing required fields: {missing!r}"
        )
    mode = _MODE_ALIASES.get(str(config.get("mode", "")), str(config.get("mode", "")))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"guidance asset contract has unsupported mode {mode!r}")
    for field, expected in (
        ("checkpoint", EXPECTED_CHECKPOINT_PATH),
        ("e0_checkpoint", EXPECTED_E0_CHECKPOINT_PATH),
        ("edev_checkpoint", EXPECTED_EDEV_CHECKPOINT_PATH),
        ("vae_path", EXPECTED_VAE_PATH),
        ("vae_scaling_factor", EXPECTED_VAE_SCALING_FACTOR),
    ):
        if config.get(field) != expected:
            raise ValueError(f"guidance asset {field} must be {expected!r}")
    checkpoint_path = Path(str(config["checkpoint"]))
    e0_path = Path(str(config["e0_checkpoint"]))
    edev_path = Path(str(config["edev_checkpoint"]))
    vae_path = Path(str(config["vae_path"]))
    index_path = Path(str(config["index"]))
    features_path = Path(str(config["features"]))
    sample_manifest_path = Path(str(config["sample_id_manifest"]))
    heldout_e1_path = Path(str(config["heldout_e1_checkpoint"]))
    heldout_e2_path = Path(str(config["heldout_e2_checkpoint"]))
    cache_path = config.get("asset_digest_cache")

    def digest(path: Path, expected_field: str) -> str:
        if cache_path:
            return cached_asset_digest(
                path, str(config[expected_field]), str(cache_path)
            )
        return _digest_path(path)

    checkpoint_digest = digest(checkpoint_path, "checkpoint_sha256")
    e0_digest = digest(e0_path, "e0_sha256")
    edev_digest = digest(edev_path, "edev_sha256")
    vae_digest = digest(vae_path, "vae_digest")
    index_digest = digest(index_path, "index_sha256")
    features_digest = digest(features_path, "features_digest")
    sample_manifest_digest = digest(sample_manifest_path, "sample_id_manifest_sha256")
    ordered_manifest_rows = read_ordered_sample_manifest(sample_manifest_path)
    ordered_manifest_ids = [str(row["sample_id"]) for row in ordered_manifest_rows]
    heldout_e1_digest = digest(heldout_e1_path, "heldout_e1_sha256")
    heldout_e2_digest = digest(heldout_e2_path, "heldout_e2_sha256")
    _validate_expected_digest(
        config, ("checkpoint_sha256",), checkpoint_digest, "checkpoint"
    )
    _validate_expected_digest(config, ("e0_sha256",), e0_digest, "E0")
    _validate_expected_digest(config, ("edev_sha256",), edev_digest, "Edev")
    _validate_expected_digest(config, ("vae_digest",), vae_digest, "VAE")
    _validate_expected_digest(config, ("index_sha256",), index_digest, "real index")
    _validate_expected_digest(
        config, ("features_digest",), features_digest, "target features"
    )
    _validate_expected_digest(
        config,
        ("sample_id_manifest_sha256",),
        sample_manifest_digest,
        "sample manifest",
    )
    _validate_expected_digest(
        config, ("heldout_e1_sha256",), heldout_e1_digest, "heldout E1"
    )
    _validate_expected_digest(
        config, ("heldout_e2_sha256",), heldout_e2_digest, "heldout E2"
    )
    scale = float(config["vae_scaling_factor"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"vae_scaling_factor must be positive and finite, got {scale!r}"
        )
    seed = int(config.get("sampling_seed", config.get("seed")))
    return {
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_digest},
        "e0": {"path": str(e0_path), "sha256": e0_digest},
        "edev": {"path": str(edev_path), "sha256": edev_digest},
        "vae": {"path": str(vae_path), "digest": vae_digest, "scaling_factor": scale},
        "real_index": {"path": str(index_path), "sha256": index_digest},
        "target_features": {
            "path": str(features_path),
            "digest": features_digest,
            "feature_source": str(config.get("feature_source")),
        },
        "sample_manifest": {
            "path": str(sample_manifest_path),
            "sha256": sample_manifest_digest,
            "sample_count": len(ordered_manifest_ids),
            "ordered_sample_id_sha256": _sample_id_digest(ordered_manifest_ids),
        },
        "heldout_e1": {"path": str(heldout_e1_path), "sha256": heldout_e1_digest},
        "heldout_e2": {"path": str(heldout_e2_path), "sha256": heldout_e2_digest},
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
    generator_loader_kwargs: dict[str, Any] = {"device": device}
    if is_r9_guidance_config(config):
        generator_loader_kwargs["r9_attention_backend"] = str(
            config["attention_backend"]
        )
    generator, checkpoint_state = generator_loader(
        config["checkpoint"], **generator_loader_kwargs
    )
    e0, _ = encoder_loader(config["e0_checkpoint"], device="cpu")
    e0 = e0.to(device)
    edev = None
    if _edev_required_for_config(config):
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
        target_features_path=Path(str(asset_contract["target_features"]["path"])),
        target_features_digest=str(asset_contract["target_features"]["digest"]),
        feature_source=str(asset_contract["target_features"]["feature_source"]),
        input_sample_manifest_path=Path(str(asset_contract["sample_manifest"]["path"])),
        input_sample_manifest_sha256=str(asset_contract["sample_manifest"]["sha256"]),
        input_sample_manifest_id_sha256=str(
            asset_contract["sample_manifest"]["ordered_sample_id_sha256"]
        ),
        input_sample_manifest_count=int(
            asset_contract["sample_manifest"]["sample_count"]
        ),
        heldout_e1=dict(asset_contract["heldout_e1"]),
        heldout_e2=dict(asset_contract["heldout_e2"]),
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
    r9_contract = is_r9_guidance_config(config)
    expected_schema_version = 3 if r9_contract else 2
    if manifest.get("schema_version") != expected_schema_version:
        raise ValueError(
            "locked schedule manifest must use "
            f"schema_version={expected_schema_version}"
        )
    if manifest.get("gate_passed") is not True:
        raise ValueError("locked schedule manifest must record gate_passed=true")
    schedule_contract_sha256 = _require_sha256(
        manifest.get("schedule_contract_sha256"), "schedule_contract_sha256"
    )
    computed_schedule_sha256 = (
        canonical_r9_schedule_contract_sha256(manifest)
        if r9_contract
        else _schedule_contract_sha256(manifest)
    )
    if computed_schedule_sha256 != schedule_contract_sha256:
        raise ValueError(
            "locked schedule contract SHA256 does not match its canonical payload"
        )
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
            raise ValueError(
                f"{source} t_cut {parsed} disagrees with locked manifest t_cut {t_cut}"
            )
    guided = [1.0 - index * (1.0 - t_cut) / 3.0 for index in range(4)]
    unguided = [t_cut, t_cut / 2.0, 0.0]
    guided[-1] = t_cut
    if (
        manifest.get("guided_steps") != 3
        or manifest.get("unguided_tail_intervals") != 2
    ):
        raise ValueError(
            "locked schedule must register exactly 3 guided and 2 unguided intervals"
        )
    for field, expected in (("guided_times", guided), ("unguided_times", unguided)):
        if not _float_sequences_equal(manifest.get(field), expected):
            raise ValueError(
                f"locked schedule {field} disagrees with the uniform schedule"
            )
    for field, expected in (("guided_times", guided), ("unguided_times", unguided)):
        if field in config and not _float_sequences_equal(config[field], expected):
            raise ValueError(
                f"config {field} disagrees with the locked uniform schedule"
            )
    report_path = _locked_path_binding(config, manifest, "semigroup_report")
    report_sha256 = _require_sha256(
        manifest.get("semigroup_report_sha256"), "semigroup_report_sha256"
    )
    if _digest_path(report_path) != report_sha256:
        raise ValueError(
            "locked semigroup report SHA256 does not match the report file"
        )
    sample_manifest_path = _locked_path_binding(
        config, manifest, "semigroup_sample_id_manifest"
    )
    sample_manifest_sha256 = _require_sha256(
        manifest.get("semigroup_sample_id_manifest_sha256"),
        "semigroup_sample_id_manifest_sha256",
    )
    config_semigroup_sha256 = _require_sha256(
        config.get("semigroup_sample_id_manifest_sha256"),
        "config semigroup_sample_id_manifest_sha256",
    )
    if config_semigroup_sha256 != sample_manifest_sha256:
        raise ValueError(
            "config semigroup sample manifest SHA256 disagrees with locked schedule"
        )
    if _digest_path(sample_manifest_path) != sample_manifest_sha256:
        raise ValueError(
            "locked semigroup sample manifest SHA256 does not match the manifest file"
        )
    resolved_schedule = {
        "manifest": str(manifest_path),
        "manifest_sha256": _digest_path(manifest_path),
        "schedule_contract_sha256": schedule_contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "semigroup_report": str(report_path),
        "semigroup_report_sha256": report_sha256,
        "semigroup_sample_id_manifest": str(sample_manifest_path),
        "semigroup_sample_id_manifest_sha256": sample_manifest_sha256,
        "t_cut": t_cut,
        "guided_times": guided,
        "unguided_times": unguided,
        "gate_passed": True,
    }
    if r9_contract:
        gate_contract = validate_r9_locked_schedule_bindings(config, manifest)
        for field in (
            "semigroup_preflight_contract",
            "semigroup_preflight_contract_sha256",
            "r9_semigroup_gate_contract",
            "r9_semigroup_gate_contract_sha256",
        ):
            resolved_schedule[field] = manifest[field]
        resolved_schedule["r9_gate_contract"] = gate_contract
    return resolved_schedule


def _bind_arm_config_digest(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    computed = canonical_guidance_arm_config_digest(resolved)
    declared = resolved.get("arm_config_sha256")
    if declared is not None and require_arm_config_digest(declared) != computed:
        raise ValueError(
            "declared arm config SHA256 disagrees with the canonical algorithm/asset contract"
        )
    resolved["arm_config_sha256"] = computed
    return resolved


def _schedule_contract_sha256(payload: Mapping[str, Any]) -> str:
    contract = dict(payload)
    contract.pop("schedule_contract_sha256", None)
    encoded = json.dumps(
        _json_safe(contract), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_path_binding(
    config: Mapping[str, Any], manifest: Mapping[str, Any], field: str
) -> Path:
    config_value = config.get(field)
    manifest_value = manifest.get(field)
    if not config_value or not manifest_value:
        raise ValueError(
            f"locked schedule requires {field} in both config and manifest"
        )
    config_path = Path(str(config_value))
    manifest_path = Path(str(manifest_value))
    if config_path.resolve() != manifest_path.resolve():
        raise ValueError(f"config {field} disagrees with the locked schedule")
    return config_path


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
                raise ValueError(
                    f"{manifest_path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_no}: expected JSON object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"{manifest_path}:{line_no}: sample_id must be a non-empty string"
                )
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id in manifest: {sample_id!r}")
            seen.add(sample_id)
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"sample manifest contains no rows: {manifest_path}")
    return rows


def deterministic_shard(
    rows: Sequence[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    if isinstance(num_shards, bool) or int(num_shards) != num_shards or num_shards <= 0:
        raise ValueError(f"num_shards must be a positive integer, got {num_shards!r}")
    if (
        isinstance(shard_index, bool)
        or int(shard_index) != shard_index
        or not 0 <= shard_index < num_shards
    ):
        raise ValueError(
            f"shard_index must be in [0,{num_shards}), got {shard_index!r}"
        )
    return [
        dict(row)
        for position, row in enumerate(rows)
        if position % int(num_shards) == int(shard_index)
    ]


def resume_remaining_ids(
    expected_ids: Sequence[str], completed_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    expected = list(expected_ids)
    completed = [row.get("sample_id") for row in completed_rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in completed):
        raise ValueError("resume rows require non-empty string sample_id values")
    if len(set(completed)) != len(completed):
        raise ValueError("resume rows contain duplicate sample IDs")
    if completed != expected[: len(completed)]:
        raise ValueError(
            "resume rows must be an exact prefix of the deterministic shard"
        )
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
    interval_contract = validate_r9_interval_guidance_config(config, mode=mode)
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
            diagnostics={
                "mode": mode,
                "semigroup": report,
                "flow_map_trace": list(counted.trace),
            },
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
            sample_mode=str(config["sample_mode"]),
            optimization_mode=str(config["optimization_mode"]),
            num_optim_iters=int(config["num_optim_iters"]),
            step_size=float(config["step_size"]),
            active_guidance_intervals=None
            if interval_contract is None
            else interval_contract["active_guidance_intervals"],
            collect_interval_diagnostics=False
            if interval_contract is None
            else bool(interval_contract["collect_interval_diagnostics"]),
        )
        return _result_with_trace(
            result,
            counted.trace,
            mode,
            interval_contract=interval_contract,
        )
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
            active_guidance_intervals=None
            if interval_contract is None
            else interval_contract["active_guidance_intervals"],
            collect_interval_diagnostics=False
            if interval_contract is None
            else bool(interval_contract["collect_interval_diagnostics"]),
        )
        return _result_with_trace(
            result,
            counted.trace,
            mode,
            interval_contract=interval_contract,
        )
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
    selected = deterministic_shard(
        [dict(row) for row in records], shard_index, num_shards
    )
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
    r9_execution_contract = None
    r9_phase_contract = None
    r9_interval_contract = validate_r9_interval_guidance_config(
        resolved_config, mode=mode
    )
    if is_r9_guidance_config(resolved_config):
        r9_execution_contract = validate_r9_execution_config(resolved_config)
        if resolved_config.get("r9_execution_contract") != r9_execution_contract:
            raise ValueError(
                "R9 run requires the applied strict execution contract from the runner preflight"
            )
        r9_phase_contract = validate_r9_phase_contract(resolved_config)
        if resolved_config.get(R9_PHASE_CONTRACT_FIELD) != r9_phase_contract:
            raise ValueError(
                "R9 run requires the validated phase contract from config preflight"
            )
        if (
            r9_interval_contract is not None
            and resolved_config.get(R9_GUIDANCE_INTERVAL_CONTRACT_FIELD)
            != r9_interval_contract
        ):
            raise ValueError(
                "R9 run requires the validated interval guidance contract from config preflight"
            )
        assert_r9_strict_cuda_determinism(torch_module=torch)
    resolved_config = _bind_arm_config_digest(resolved_config)
    generated_dir = output / "generated_images"
    native_dir = output / "native_images"
    per_sample_path = output / "per_sample.jsonl"
    run_manifest_path = output / "run_manifest.json"
    generation_result_path = output / "generation_result.json"
    sample_manifest_path = output / "sample_id_manifest.jsonl"
    semigroup_path = output / "semigroup.json"
    semigroup_split_dir = output / "semigroup_split_images"
    resume_contract_path = output / "resume_contract.json"
    session_history_path = output / "session_history.jsonl"
    session_journal_path = output / "session_journal.json"
    completion_path = output / "completion.json"

    phase = str(resolved_config.get("phase", "full"))
    contact_enabled = _contact_sheets_enabled(resolved_config)
    _require_safe_output_root(output)
    owned_paths = [
        generated_dir,
        per_sample_path,
        run_manifest_path,
        generation_result_path,
        sample_manifest_path,
        resume_contract_path,
        session_history_path,
        session_journal_path,
        completion_path,
    ]
    if mode != "native":
        owned_paths.append(native_dir)
    if mode == "semigroup":
        owned_paths.extend((semigroup_path, semigroup_split_dir))
    if contact_enabled:
        owned_paths.extend(
            [output / "contact_sheets", output / "contact_sheet_columns.json"]
        )
    for path in owned_paths:
        _require_contained(output, path, "guidance output")
    if completion_path.exists():
        raise FileExistsError(
            f"refusing to replace completed output: {completion_path}"
        )
    _validate_output_entries(
        output,
        mode=mode,
        contact_sheets=contact_enabled,
    )
    edev_required = _edev_required_for_config(resolved_config)
    if edev_required and runtime.edev is None:
        raise ValueError(f"guidance phase {phase!r} requires a frozen Edev checkpoint")
    if not edev_required and runtime.edev is not None:
        raise ValueError(
            f"Edev must not be loaded or scored in guidance phase {phase!r}"
        )
    output.mkdir(parents=True, exist_ok=True)
    _require_safe_output_root(output)
    _prepare_owned_directory(output, generated_dir, "generated image directory")
    if mode != "native":
        _prepare_owned_directory(output, native_dir, "native image directory")
    if mode == "semigroup":
        _prepare_owned_directory(
            output, semigroup_split_dir, "semigroup split image directory"
        )
    if contact_enabled and (output / "contact_sheets").exists():
        _prepare_owned_directory(
            output, output / "contact_sheets", "contact sheet directory"
        )

    resume_contract = {
        "checkpoint": {
            "path": str(runtime.checkpoint_path),
            "sha256": runtime.checkpoint_sha256,
            "state": runtime.checkpoint_state,
        },
        "e0": {
            "path": ""
            if runtime.e0_checkpoint_path is None
            else str(runtime.e0_checkpoint_path),
            "sha256": runtime.e0_checkpoint_sha256,
        },
        "edev": {
            "path": ""
            if runtime.edev_checkpoint_path is None
            else str(runtime.edev_checkpoint_path),
            "sha256": runtime.edev_checkpoint_sha256,
        },
        "vae": {
            "path": "" if runtime.vae_path is None else str(runtime.vae_path),
            "digest": runtime.vae_digest,
            "scaling_factor": runtime.vae_scaling_factor,
        },
        "real_index": {
            "path": ""
            if runtime.real_index_path is None
            else str(runtime.real_index_path),
            "sha256": runtime.real_index_sha256,
        },
        "target_features": {
            "path": ""
            if runtime.target_features_path is None
            else str(runtime.target_features_path),
            "digest": runtime.target_features_digest,
            "feature_source": runtime.feature_source,
        },
        "input_sample_manifest": {
            "path": ""
            if runtime.input_sample_manifest_path is None
            else str(runtime.input_sample_manifest_path),
            "sha256": runtime.input_sample_manifest_sha256,
            "sample_count": runtime.input_sample_manifest_count,
            "ordered_sample_id_sha256": runtime.input_sample_manifest_id_sha256,
        },
        "heldout_e1": runtime.heldout_e1,
        "heldout_e2": runtime.heldout_e2,
        "r9_execution_contract": r9_execution_contract,
        "seed": int(
            resolved_config.get("sampling_seed", resolved_config.get("seed", 0))
        ),
        "schedule": resolved_config.get("locked_schedule"),
        "mode": mode,
        "arm_config_sha256": resolved_config["arm_config_sha256"],
        "config": resolved_config,
        "sample_id_sha256": _sample_id_digest(expected_ids),
        "shard": {
            "index": int(shard_index),
            "count": int(num_shards),
            "sample_count": len(expected_ids),
            "ordered_sample_id_sha256": _sample_id_digest(expected_ids),
        },
    }
    if r9_phase_contract is not None:
        resume_contract[R9_PHASE_CONTRACT_FIELD] = r9_phase_contract
    if r9_interval_contract is not None:
        resume_contract[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD] = r9_interval_contract
    if resume_contract_path.exists():
        existing_contract = _read_json_mapping(resume_contract_path, "resume contract")
        if existing_contract != _json_safe(resume_contract):
            raise ValueError(
                "existing resume contract disagrees with the fixed run contract"
            )
    else:
        _atomic_write_json(resume_contract_path, resume_contract, exclusive=True)
    _cleanup_known_temps(output)

    expected_manifest_rows = [
        {
            "ordinal": ordinal,
            "sample_id": row["sample_id"],
            "source": str(row["source"]),
        }
        for ordinal, row in enumerate(selected)
    ]
    if sample_manifest_path.exists():
        existing_manifest = _read_optional_jsonl(sample_manifest_path)
        if existing_manifest != expected_manifest_rows:
            raise ValueError(
                "existing sample_id_manifest.jsonl disagrees with the deterministic shard"
            )
    else:
        _write_jsonl(sample_manifest_path, expected_manifest_rows, mode="x")

    completed_rows = _read_optional_jsonl(per_sample_path)
    expected_bindings = _expected_row_bindings(
        selected, generated_dir, native_dir, mode
    )
    semigroup_split_bindings = (
        _expected_semigroup_split_bindings(
            selected,
            semigroup_split_dir,
            resolved_config.get("split_times", (0.25, 0.5, 0.75)),
        )
        if mode == "semigroup"
        else []
    )
    for binding in expected_bindings:
        _require_contained(output, Path(str(binding["generated"])), "generated image")
        if mode != "native":
            _require_contained(output, Path(str(binding["native"])), "native image")
    _validate_resume_rows(
        completed_rows,
        expected_bindings,
        r9_interval_contract=r9_interval_contract,
        r9_phase_contract=r9_phase_contract,
    )
    completed_count = len(completed_rows)
    remaining_records = selected[completed_count:]
    _validate_owned_png_state(
        generated_dir=generated_dir,
        native_dir=native_dir,
        expected_bindings=expected_bindings,
        completed_count=completed_count,
        mode=mode,
    )
    if mode == "semigroup":
        _validate_semigroup_split_state(
            completed_rows, semigroup_split_bindings, semigroup_split_dir
        )

    schedule = resolved_config.get("locked_schedule")
    if schedule is not None and not isinstance(schedule, Mapping):
        raise ValueError("config.locked_schedule must be a mapping")
    batch_size = _positive_int(resolved_config.get("batch_size", 1), "batch_size")
    sampling_seed = int(
        resolved_config.get("sampling_seed", resolved_config.get("seed", 0))
    )
    image_size = _positive_int(resolved_config.get("image_size", 32), "image_size")
    channels = _sample_channels(runtime.generator, resolved_config)
    sessions = _read_optional_jsonl(session_history_path)
    _validate_session_history(sessions)
    if session_journal_path.exists():
        recovered = _read_json_mapping(session_journal_path, "session journal")
        _validate_session_history([recovered], require_contiguous_indices=False)
        recovered_id = str(recovered["session_id"])
        existing_by_id = {str(session["session_id"]): session for session in sessions}
        if recovered_id in existing_by_id:
            if not _same_session(existing_by_id[recovered_id], recovered):
                raise ValueError(
                    "session journal disagrees with its committed history row"
                )
        else:
            if int(recovered.get("session_index", -1)) != len(sessions):
                raise ValueError(
                    "session journal index does not follow session history"
                )
            recovered["recovered_after_crash"] = True
            _append_jsonl(session_history_path, recovered)
            sessions.append(recovered)
        session_journal_path.unlink()
        _fsync_directory(output)
        _validate_session_history(sessions)
    _cuda_reset(runtime.device)
    session_candidate_seconds = 0.0
    session_native_seconds = 0.0
    session_row_io_seconds = 0.0
    session_artifact_io_seconds = 0.0
    session_index = len(sessions)
    session_id = uuid.uuid4().hex
    _atomic_write_json(
        session_journal_path,
        _session_snapshot(
            session_id=session_id,
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
        target_z0 = torch.stack([_as_float_tensor(row["z"]) for row in batch]).to(
            runtime.device
        )
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
            native_result = execute_matched_native(
                generator=runtime.generator, x_init=x_init
            )
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
            native_images = (
                generated
                if mode == "native"
                else runtime.codec.decode(native_result.latent)
            )
            candidate_embedding = runtime.e0(normalize_for_e0(generated))["embedding"]
            native_embedding = runtime.e0(normalize_for_e0(native_images))["embedding"]
            candidate_cosine = F.cosine_similarity(
                candidate_embedding, target_z0, dim=1
            )
            native_cosine = F.cosine_similarity(native_embedding, target_z0, dim=1)
            edev_cosine = None
            native_edev_cosine = None
            if runtime.edev is not None:
                if source_images is None:
                    raise RuntimeError("Edev source images were not loaded")
                edev_generated = runtime.edev(normalize_for_e0(generated))["embedding"]
                edev_native = runtime.edev(normalize_for_e0(native_images))["embedding"]
                edev_source = runtime.edev(normalize_for_e0(source_images))["embedding"]
                edev_cosine = F.cosine_similarity(edev_generated, edev_source, dim=1)
                native_edev_cosine = F.cosine_similarity(
                    edev_native, edev_source, dim=1
                )
        tensors_to_check = [generated, native_images, candidate_cosine, native_cosine]
        if edev_cosine is not None:
            tensors_to_check.append(edev_cosine)
        if native_edev_cosine is not None:
            tensors_to_check.append(native_edev_cosine)
        if any(not torch.isfinite(tensor).all() for tensor in tensors_to_check):
            raise FloatingPointError(
                "guidance runner produced non-finite images or cosine values"
            )
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
                    split_image_bindings=semigroup_split_bindings[
                        completed_count + batch_start : completed_count
                        + batch_start
                        + len(batch)
                    ],
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
                "generation_seconds": per_sample_candidate_seconds
                + per_sample_native_seconds,
                "io_seconds": 0.0,
                "sample_id": sample_id,
                "route_diagnostics": _per_sample_diagnostics(
                    candidate_result.diagnostics, local_index, len(sample_ids)
                ),
            }
            if r9_interval_contract is not None:
                row.update(
                    {
                        "candidate_algorithm_nfe": int(
                            candidate_result.diagnostics["algorithm_nfe"]
                        ),
                        "candidate_diagnostic_nfe": int(
                            candidate_result.diagnostics["diagnostic_nfe"]
                        ),
                        "candidate_diagnostic_trace": list(
                            candidate_result.diagnostics["diagnostic_flow_map_trace"]
                        ),
                    }
                )
            if edev_cosine is not None:
                row["edev_cosine"] = float(edev_cosine[local_index].detach().cpu())
            if native_edev_cosine is not None:
                row["native_edev_cosine"] = float(
                    native_edev_cosine[local_index].detach().cpu()
                )
            if mode == "semigroup":
                row["semigroup"] = batch_semigroup[sample_id]
            source_io_per_sample = source_io_seconds / len(sample_ids)
            row["io_seconds"] = (
                time.perf_counter() - row_io_started + source_io_per_sample
            )
            committed_this_session = batch_start + local_index + 1
            _atomic_write_json(
                session_journal_path,
                _session_snapshot(
                    session_id=session_id,
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
            session_row_io_seconds += (
                time.perf_counter() - row_io_started + source_io_per_sample
            )
            _atomic_write_json(
                session_journal_path,
                _session_snapshot(
                    session_id=session_id,
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
        raise RuntimeError(
            "guidance run ended before every deterministic shard sample was written"
        )
    _validate_resume_rows(
        final_rows,
        expected_bindings,
        r9_interval_contract=r9_interval_contract,
    )
    artifact_io_started = time.perf_counter()
    if mode == "semigroup":
        _atomic_write_json(
            semigroup_path,
            {
                "mode": mode,
                "split_times": [
                    float(value)
                    for value in resolved_config.get("split_times", (0.25, 0.5, 0.75))
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
                resolved_config.get("contact_sheet_tile_size", 128),
                "contact_sheet_tile_size",
            ),
        )
    session_artifact_io_seconds += time.perf_counter() - artifact_io_started
    session = _session_snapshot(
        session_id=session_id,
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
    _fsync_directory(output)
    sessions = _read_optional_jsonl(session_history_path)
    _validate_session_history(sessions)
    timing = _aggregate_timing(final_rows, sessions, completed_count, generated_count)
    max_memory = aggregate_session_memory(sessions)
    candidate_nfe = _single_row_value(final_rows, "candidate_nfe")
    native_nfe = _single_row_value(final_rows, "native_nfe")
    candidate_algorithm_nfe = None
    candidate_diagnostic_nfe = None
    if r9_interval_contract is not None:
        candidate_algorithm_nfe = _single_row_value(
            final_rows, "candidate_algorithm_nfe"
        )
        candidate_diagnostic_nfe = _single_row_value(
            final_rows, "candidate_diagnostic_nfe"
        )
    cosine = {
        "candidate_e0_target": _finite_summary(
            [float(row["candidate_cosine"]) for row in final_rows]
        ),
        "native_e0_target": _finite_summary(
            [float(row["native_cosine"]) for row in final_rows]
        ),
    }
    if edev_required and any("edev_cosine" not in row for row in final_rows):
        raise ValueError(f"guidance phase {phase!r} is missing candidate Edev evidence")
    if edev_required and any("native_edev_cosine" not in row for row in final_rows):
        raise ValueError(f"guidance phase {phase!r} is missing native Edev evidence")
    if all("edev_cosine" in row for row in final_rows):
        cosine["candidate_edev_source"] = _finite_summary(
            [float(row["edev_cosine"]) for row in final_rows]
        )
    if all("native_edev_cosine" in row for row in final_rows):
        cosine["native_edev_source"] = _finite_summary(
            [float(row["native_edev_cosine"]) for row in final_rows]
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "arm_config_sha256": resolved_config["arm_config_sha256"],
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
        "r9_execution_contract": _json_safe(r9_execution_contract),
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
            "completion": str(completion_path),
        },
    }
    if r9_interval_contract is not None:
        manifest[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD] = _json_safe(r9_interval_contract)
        manifest["nfe"].update(
            {
                "candidate_algorithm": candidate_algorithm_nfe,
                "candidate_diagnostic": candidate_diagnostic_nfe,
            }
        )
        manifest["diagnostic_flow_map_traces"] = [
            {
                "sample_id": row["sample_id"],
                "candidate": row["candidate_diagnostic_trace"],
            }
            for row in final_rows
        ]
    if r9_phase_contract is not None:
        manifest[R9_PHASE_CONTRACT_FIELD] = _json_safe(r9_phase_contract)
    _atomic_write_json(generation_result_path, manifest)
    _atomic_write_json(run_manifest_path, manifest)
    _atomic_write_json(
        completion_path,
        {
            "schema_version": 1,
            "status": "complete",
            "sample_count": len(expected_ids),
            "sample_id_sha256": _sample_id_digest(expected_ids),
            "arm_config_sha256": resolved_config["arm_config_sha256"],
            "generation_result": str(generation_result_path),
            "run_manifest": str(run_manifest_path),
        },
        exclusive=True,
    )
    return manifest


def run_guidance_from_config(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    shard_index: int = 0,
    num_shards: int = 1,
    explicit_t_cut: float | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    completion_path = output / "completion.json"
    _require_safe_output_root(output)
    _require_contained(output, completion_path, "completion marker")
    if completion_path.exists():
        raise FileExistsError(
            f"refusing to replace completed output: {completion_path}"
        )
    resolved = _prepare_sharded_guidance_config(
        config,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if is_r9_guidance_config(resolved):
        resolved["r9_execution_contract"] = apply_r9_strict_cuda_determinism(
            resolved, torch_module=torch
        )
    from safa.data.feature_dataset import FeatureAlignedAffectNet
    from safa.training.transforms import generator_image_transform
    from safa.utils.device import require_cuda_device

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
    assets = asset_contract_from_config(resolved)
    checkpoint_sha256 = str(assets["checkpoint"]["sha256"])
    mode = resolved["mode"]
    if mode in FMRG_MODES:
        schedule = resolve_locked_schedule(
            resolved,
            checkpoint_sha256=checkpoint_sha256,
            explicit_t_cut=explicit_t_cut,
        )
        _validate_semigroup_gate(
            resolved,
            checkpoint_sha256,
            schedule["t_cut"],
            schedule["semigroup_sample_id_manifest_sha256"],
        )
        resolved["locked_schedule"] = schedule
    resolved = _bind_arm_config_digest(resolved)
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


def _records_from_feature_dataset(
    dataset, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    seen_sample_ids = set()
    duplicate_sample_ids = set()
    for record in dataset.records:
        sample_id = record.sample_id
        if sample_id in seen_sample_ids:
            duplicate_sample_ids.add(sample_id)
        seen_sample_ids.add(sample_id)
    if duplicate_sample_ids:
        raise ValueError(
            "feature dataset contains duplicate sample_id values: "
            f"{sorted(duplicate_sample_ids)!r}"
        )
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
            raise ValueError(
                f"sample ID manifest is not covered by the feature dataset: {missing!r}"
            )
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
    selected = deterministic_shard(
        [dict(row) for row in records], shard_index, num_shards
    )
    existing = _read_json_mapping(path, "resume contract")
    expected = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "e0": asset_contract["e0"],
        "edev": asset_contract["edev"],
        "vae": asset_contract["vae"],
        "real_index": asset_contract["real_index"],
        "target_features": asset_contract["target_features"],
        "input_sample_manifest": asset_contract["sample_manifest"],
        "heldout_e1": asset_contract["heldout_e1"],
        "heldout_e2": asset_contract["heldout_e2"],
        "arm_config_sha256": config["arm_config_sha256"],
        "seed": int(config["sampling_seed"]),
        "schedule": config.get("locked_schedule"),
        "mode": config["mode"],
        "config": dict(config),
        "sample_id_sha256": _sample_id_digest(
            str(row["sample_id"]) for row in selected
        ),
        "shard": {
            "index": int(shard_index),
            "count": int(num_shards),
            "sample_count": len(selected),
            "ordered_sample_id_sha256": _sample_id_digest(
                str(row["sample_id"]) for row in selected
            ),
        },
        R9_GUIDANCE_INTERVAL_CONTRACT_FIELD: config.get(
            R9_GUIDANCE_INTERVAL_CONTRACT_FIELD
        ),
    }
    actual = {
        "checkpoint_path": existing.get("checkpoint", {}).get("path"),
        "checkpoint_sha256": existing.get("checkpoint", {}).get("sha256"),
        "e0": existing.get("e0"),
        "edev": existing.get("edev"),
        "vae": existing.get("vae"),
        "real_index": existing.get("real_index"),
        "target_features": existing.get("target_features"),
        "input_sample_manifest": existing.get("input_sample_manifest"),
        "heldout_e1": existing.get("heldout_e1"),
        "heldout_e2": existing.get("heldout_e2"),
        "arm_config_sha256": existing.get("arm_config_sha256"),
        "seed": existing.get("seed"),
        "schedule": existing.get("schedule"),
        "mode": existing.get("mode"),
        "config": existing.get("config"),
        "sample_id_sha256": existing.get("sample_id_sha256"),
        "shard": existing.get("shard"),
        R9_GUIDANCE_INTERVAL_CONTRACT_FIELD: existing.get(
            R9_GUIDANCE_INTERVAL_CONTRACT_FIELD
        ),
    }
    if _json_safe(actual) != _json_safe(expected):
        raise ValueError("existing resume contract disagrees before CUDA/model loading")


def _validate_semigroup_gate(
    config: Mapping[str, Any],
    checkpoint_hash: str,
    t_cut: float,
    locked_sample_id_manifest_sha256: str,
) -> None:
    report_value = config.get("semigroup_report")
    if not report_value:
        raise ValueError("FMRG guidance requires semigroup_report")
    report = _read_json_mapping(Path(str(report_value)), "semigroup report")
    if report.get("gate_passed") is not True:
        raise ValueError("semigroup report must record gate_passed=true")
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(
            "semigroup report checkpoint SHA256 disagrees with the loaded checkpoint"
        )
    locked_sample_manifest_sha256 = _require_sha256(
        locked_sample_id_manifest_sha256,
        "locked schedule semigroup_sample_id_manifest_sha256",
    )
    config_sample_manifest_sha256 = _require_sha256(
        config.get("semigroup_sample_id_manifest_sha256"),
        "config semigroup_sample_id_manifest_sha256",
    )
    if config_sample_manifest_sha256 != locked_sample_manifest_sha256:
        raise ValueError(
            "config semigroup sample manifest SHA256 disagrees with locked schedule"
        )
    report_sample_manifest_sha256 = _require_sha256(
        report.get("sample_id_manifest_sha256"),
        "semigroup report sample_id_manifest_sha256",
    )
    if report_sample_manifest_sha256 != locked_sample_manifest_sha256:
        raise ValueError(
            "semigroup report sample_id_manifest_sha256 disagrees with locked schedule"
        )
    selected = _finite_open_unit(
        report.get("selected_t_cut", report.get("t_cut")),
        "semigroup report selected t_cut",
    )
    if not math.isclose(selected, t_cut, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "semigroup report selected t_cut disagrees with the locked schedule"
        )


def _semigroup_batch_rows(
    *, result, sample_ids, codec, e0, direct_images, split_image_bindings
) -> list[dict[str, Any]]:
    report = result.diagnostics["semigroup"]
    rows = [{"sample_id": sample_id, "splits": {}} for sample_id in sample_ids]
    with torch.no_grad():
        direct_embedding = e0(normalize_for_e0(direct_images))["embedding"]
        for split, endpoints in report["split_endpoints"].items():
            split_key = str(float(split))
            split_images = codec.decode(endpoints)
            split_embedding = e0(normalize_for_e0(split_images))["embedding"]
            pixel_l1 = (direct_images - split_images).abs().flatten(1).mean(dim=1)
            mse = (direct_images - split_images).square().flatten(1).mean(dim=1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
            cosine = F.cosine_similarity(direct_embedding, split_embedding, dim=1)
            residual = report["residuals"][split]
            for index, row in enumerate(rows):
                decoded_image = Path(str(split_image_bindings[index][split_key]))
                _atomic_save_image(split_images[index], decoded_image)
                row["splits"][split_key] = {
                    "latent_residual": float(residual[index].detach().cpu()),
                    "decoded_pixel_l1": float(pixel_l1[index].detach().cpu()),
                    "decoded_psnr": float(psnr[index].detach().cpu()),
                    "endpoint_e0_cosine": float(cosine[index].detach().cpu()),
                    "decoded_image": str(decoded_image),
                }
    return rows


def _validate_generation_records(records: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(records):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                f"generation record {index} requires a non-empty string sample_id"
            )
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


def _expected_semigroup_split_bindings(selected, output_dir: Path, split_times):
    split_keys = [str(float(value)) for value in split_times]
    if set(split_keys) != {"0.25", "0.5", "0.75"} or len(split_keys) != 3:
        raise ValueError(
            "semigroup split image bindings require registered splits 0.25,0.5,0.75"
        )
    bindings = []
    for ordinal, record in enumerate(selected):
        filename = f"{ordinal:08d}__{_safe_sample_id(str(record['sample_id']))}.png"
        bindings.append(
            {
                split_key: str(
                    output_dir / f"t_cut_{split_key.replace('.', 'p')}" / filename
                )
                for split_key in split_keys
            }
        )
    return bindings


def _validate_semigroup_split_state(rows, expected_bindings, output_dir: Path) -> None:
    expected_paths = {
        Path(path).resolve()
        for binding in expected_bindings
        for path in binding.values()
    }
    actual_paths = {
        path.resolve() for path in output_dir.glob("**/*") if path.is_file()
    }
    if actual_paths - expected_paths:
        raise ValueError("semigroup split image directory contains unowned files")
    for index, row in enumerate(rows):
        splits = row.get("semigroup")
        if not isinstance(splits, Mapping) or set(splits) != set(
            expected_bindings[index]
        ):
            raise ValueError("resumed semigroup row has the wrong registered split set")
        for split_key, expected_path in expected_bindings[index].items():
            if splits[split_key].get("decoded_image") != expected_path:
                raise ValueError("resumed semigroup decoded image binding disagrees")
            if not Path(expected_path).is_file():
                raise FileNotFoundError("resumed semigroup decoded image is missing")


def _validate_resume_rows(
    rows,
    expected_bindings,
    *,
    r9_interval_contract: Mapping[str, Any] | None = None,
    r9_phase_contract: Mapping[str, Any] | None = None,
) -> None:
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
                raise ValueError(
                    f"resume row binding is missing required field {field!r}"
                )
        if len(row["candidate_trace"]) != int(row["candidate_nfe"]):
            raise ValueError(
                f"resume row candidate trace/NFE mismatch at ordinal {index}"
            )
        if len(row["native_trace"]) != int(row["native_nfe"]):
            raise ValueError(f"resume row native trace/NFE mismatch at ordinal {index}")
        if r9_phase_contract is not None and r9_phase_contract["edev_required"]:
            missing_edev = [
                field
                for field in r9_phase_contract["required_per_sample_edev_fields"]
                if field not in row
            ]
            if missing_edev:
                raise ValueError(
                    f"R9 resume row is missing Edev fields at ordinal {index}: "
                    f"{missing_edev!r}"
                )
            for field in r9_phase_contract["required_per_sample_edev_fields"]:
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"R9 resume row {index} {field} must be finite numeric data"
                    )
        if r9_interval_contract is not None:
            required_r9_fields = (
                "candidate_algorithm_nfe",
                "candidate_diagnostic_nfe",
                "candidate_diagnostic_trace",
                "route_diagnostics",
            )
            missing_r9_fields = [
                field for field in required_r9_fields if field not in row
            ]
            if missing_r9_fields:
                raise ValueError(
                    f"R9 resume row is missing interval diagnostic fields at ordinal {index}: "
                    f"{missing_r9_fields!r}"
                )
            algorithm_nfe = int(row["candidate_algorithm_nfe"])
            diagnostic_nfe = int(row["candidate_diagnostic_nfe"])
            if algorithm_nfe != int(row["candidate_nfe"]):
                raise ValueError(
                    f"R9 resume row legacy/algorithm NFE mismatch at ordinal {index}"
                )
            if algorithm_nfe != int(r9_interval_contract["expected_algorithm_nfe"]):
                raise ValueError(
                    f"R9 resume row algorithm NFE disagrees with its contract at ordinal {index}"
                )
            if (
                row["candidate_trace"]
                != r9_interval_contract["expected_algorithm_trace"]
            ):
                raise ValueError(
                    f"R9 resume row algorithm trace disagrees with its contract at ordinal {index}"
                )
            if diagnostic_nfe != int(r9_interval_contract["expected_diagnostic_nfe"]):
                raise ValueError(
                    f"R9 resume row diagnostic NFE disagrees with its contract at ordinal {index}"
                )
            diagnostic_trace = row["candidate_diagnostic_trace"]
            if (
                not isinstance(diagnostic_trace, list)
                or len(diagnostic_trace) != diagnostic_nfe
            ):
                raise ValueError(
                    f"R9 resume row diagnostic trace/NFE mismatch at ordinal {index}"
                )
            if diagnostic_trace != r9_interval_contract["expected_diagnostic_trace"]:
                raise ValueError(
                    f"R9 resume row diagnostic trace disagrees with its contract at ordinal {index}"
                )
            if any(
                not isinstance(entry, Mapping)
                or entry.get("kind") != "interval_diagnostic"
                for entry in diagnostic_trace
            ):
                raise ValueError(
                    f"R9 resume row diagnostic trace has an invalid kind at ordinal {index}"
                )
            route_diagnostics = row["route_diagnostics"]
            if not isinstance(route_diagnostics, Mapping):
                raise ValueError(
                    f"R9 resume row route_diagnostics must be a mapping at ordinal {index}"
                )
            expected_route_contract = {
                "active_guidance_intervals": r9_interval_contract[
                    "active_guidance_intervals"
                ],
                "interval_diagnostics_enabled": r9_interval_contract[
                    "collect_interval_diagnostics"
                ],
                "algorithm_nfe": algorithm_nfe,
                "diagnostic_nfe": diagnostic_nfe,
            }
            actual_route_contract = {
                field: route_diagnostics.get(field) for field in expected_route_contract
            }
            if actual_route_contract != expected_route_contract:
                raise ValueError(
                    f"R9 resume row route diagnostic contract mismatch at ordinal {index}"
                )
            interval_diagnostics = route_diagnostics.get("interval_diagnostics")
            if r9_interval_contract["collect_interval_diagnostics"]:
                if (
                    not isinstance(interval_diagnostics, Mapping)
                    or tuple(interval_diagnostics) != R9_GUIDANCE_INTERVAL_IDS
                ):
                    raise ValueError(
                        f"R9 resume row interval diagnostics are incomplete at ordinal {index}"
                    )
            elif interval_diagnostics != {}:
                raise ValueError(
                    f"R9 resume row unexpectedly contains interval diagnostics at ordinal {index}"
                )
            _require_finite_diagnostic_tree(
                route_diagnostics,
                label=f"R9 resume row {index} route diagnostics",
            )
            _require_finite_diagnostic_tree(
                diagnostic_trace,
                label=f"R9 resume row {index} diagnostic trace",
            )
        for path_field in ("generated", "native"):
            if not Path(str(row[path_field])).is_file():
                raise FileNotFoundError(
                    f"resume row binding points to missing {path_field} PNG: {row[path_field]}"
                )


def _validate_owned_png_state(
    *, generated_dir, native_dir, expected_bindings, completed_count: int, mode: str
) -> None:
    actual_generated = {
        path.resolve() for path in generated_dir.iterdir() if path.is_file()
    }
    actual_native = (
        set()
        if mode == "native"
        else {path.resolve() for path in native_dir.iterdir() if path.is_file()}
    )
    completed_generated = {
        Path(binding["generated"]).resolve()
        for binding in expected_bindings[:completed_count]
    }
    completed_native = (
        set()
        if mode == "native"
        else {
            Path(binding["native"]).resolve()
            for binding in expected_bindings[:completed_count]
        }
    )
    missing = (completed_generated - actual_generated) | (
        completed_native - actual_native
    )
    if missing:
        raise FileNotFoundError(
            f"completed resume row is missing owned PNGs: {sorted(map(str, missing))!r}"
        )
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
        "completion.json",
    }
    if mode != "native":
        allowed.add("native_images")
    if mode == "semigroup":
        allowed.update({"semigroup.json", "semigroup_split_images"})
    if contact_sheets:
        allowed.update({"contact_sheets", "contact_sheet_columns.json"})
    extras = sorted(
        path.name
        for path in output.iterdir()
        if path.name not in allowed and not path.name.startswith(".tmp-")
    )
    if extras:
        raise ValueError(f"output contains unexpected files: {extras!r}")


def _require_safe_output_root(output: Path) -> None:
    if output.is_symlink():
        raise ValueError(f"guidance output root must not be a symlink: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"guidance output root must be a directory: {output}")


def _require_contained(root: Path, target: Path, label: str) -> None:
    _require_safe_output_root(root)
    root_absolute = Path(os.path.abspath(root))
    target_absolute = Path(os.path.abspath(target))
    try:
        relative = target_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the guidance output root: {target}") from exc

    cursor = root_absolute
    if cursor.is_symlink():
        raise ValueError(f"guidance output root must not be a symlink: {root}")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} contains a symlink path component: {cursor}")

    root_resolved = root_absolute.resolve(strict=False)
    target_resolved = target_absolute.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"{label} resolves outside the guidance output root: {target}"
        ) from exc


def _prepare_owned_directory(root: Path, directory: Path, label: str) -> None:
    _require_contained(root, directory, label)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"{label} must be a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    _require_contained(root, directory, label)


def _cleanup_known_temps(output: Path) -> None:
    removed = False
    for current, directory_names, file_names in os.walk(
        output, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        for name in directory_names:
            child = current_path / name
            if child.is_symlink():
                raise ValueError(
                    f"guidance output contains a symlink directory: {child}"
                )
            if name.startswith(".tmp-"):
                raise ValueError(
                    f"guidance output temporary entry is not a file: {child}"
                )
        for name in file_names:
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"guidance output contains a symlink file: {child}")
            if not name.startswith(".tmp-"):
                continue
            _require_contained(output, child, "temporary guidance output")
            if not child.is_file():
                raise ValueError(
                    f"guidance output temporary entry is not a file: {child}"
                )
            child.unlink()
            removed = True
    if removed:
        _fsync_directory(output)


def _sample_channels(generator, config: Mapping[str, Any]) -> int:
    generator_config = getattr(generator, "config", None)
    value = getattr(generator_config, "sit_input_channels", None)
    if value is None:
        value = config.get("channels", EXPECTED_MODEL_CONFIG["sit_input_channels"])
    return _positive_int(value, "sample channels")


def _as_float_tensor(value) -> torch.Tensor:
    tensor = (
        value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    )
    if tensor.ndim != 1 or not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim != 1 or not torch.isfinite(tensor).all():
        raise ValueError("target z must be a finite one-dimensional tensor")
    return tensor


def _per_sample_diagnostics(
    value: Any,
    index: int,
    batch_size: int,
    *,
    _path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if not torch.isfinite(detached).all().item():
            raise FloatingPointError(
                f"non-finite route diagnostic tensor at {'.'.join(_path) or '<root>'}"
            )
        if detached.ndim == 0:
            return float(detached)
        if detached.ndim in {1, 2} and detached.shape[0] == batch_size:
            selected = detached[index]
            return float(selected) if selected.ndim == 0 else selected.tolist()
        return None
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            converted = _per_sample_diagnostics(
                item,
                index,
                batch_size,
                _path=(*_path, str(key)),
            )
            if converted is not None:
                result[str(key)] = converted
        return result
    if isinstance(value, (list, tuple)):
        if all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        ):
            _require_finite_diagnostic_tree(
                value,
                label=f"route diagnostics {'.'.join(_path) or '<root>'}",
            )
            return list(value)
        if _path and _path[-1] == "interval_diagnostics":
            converted_items = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise ValueError(
                        "interval_diagnostics sequences must contain only mappings"
                    )
                converted_items.append(
                    _per_sample_diagnostics(
                        item,
                        index,
                        batch_size,
                        _path=_path,
                    )
                )
            return converted_items
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(
            f"non-finite route diagnostic value at {'.'.join(_path) or '<root>'}"
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _require_finite_diagnostic_tree(value: Any, *, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"{label} contains a non-finite tensor")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_diagnostic_tree(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_diagnostic_tree(item, label=label)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{label} contains a non-finite value")


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
        prefix=f".tmp-{path.stem}-", suffix=".png", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        image.save(temporary, format="PNG")
        with temporary.open("rb") as reader:
            os.fsync(reader.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_source_images(paths, image_size: int, device, dtype) -> torch.Tensor:
    from PIL import Image

    tensors = []
    for path in paths:
        with Image.open(path) as source:
            rgb = source.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            tensor = (
                data.reshape(image_size, image_size, 3)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
            )
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
    _prepare_owned_directory(output, contact_dir, "contact sheet directory")
    pages = []
    expected_pages = math.ceil(len(rows) / rows_per_page)
    expected_names = {f"page_{index:03d}.png" for index in range(expected_pages)}
    extras = sorted(
        path.name
        for path in contact_dir.iterdir()
        if path.is_file() and path.name not in expected_names
    )
    if extras:
        raise ValueError(
            f"contact sheet directory contains unexpected pages: {extras!r}"
        )
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
        _require_contained(output, path, "contact sheet page")
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
    manifest_path = output / "contact_sheet_columns.json"
    _require_contained(output, manifest_path, "contact sheet manifest")
    _atomic_write_json(manifest_path, manifest)
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
    rows = _read_optional_jsonl(path)
    _atomic_replace_jsonl(path, [*rows, dict(row)])


def _write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, mode: str = "w"
) -> None:
    if mode not in {"w", "x"}:
        raise ValueError(f"unsupported JSONL write mode: {mode!r}")
    if mode == "x" and path.exists():
        raise FileExistsError(f"refusing to replace existing JSONL: {path}")
    _atomic_replace_jsonl(path, rows, exclusive=mode == "x")


def _atomic_replace_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    exclusive: bool = False,
) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to replace existing JSONL: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".tmp-{path.stem}-",
        suffix=".jsonl",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(
                    json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False)
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(f"refusing to replace existing JSONL: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to replace existing JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".tmp-{path.stem}-",
        suffix=".json",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(
                json.dumps(
                    _json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(f"refusing to replace existing JSON: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        raise ValueError(
            f"{label} must be finite and within (0,1), got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError(f"{label} must be finite and within (0,1), got {value!r}")
    return parsed


def _float_sequences_equal(left: Any, right: Sequence[float]) -> bool:
    if (
        not isinstance(left, Sequence)
        or isinstance(left, (str, bytes))
        or len(left) != len(right)
    ):
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
    files = sorted(
        item for item in path.rglob("*") if item.is_file() or item.is_symlink()
    )
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


def cached_asset_digest(
    path: str | Path, expected_digest: str, cache_path: str | Path
) -> str:
    asset = Path(path)
    expected = _require_sha256(expected_digest, "expected_digest")
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache.with_name(f"{cache.name}.lock")
    fingerprint = _stat_fingerprint(asset)
    key_payload = {
        "path": str(asset.resolve()),
        "expected_digest": expected,
        "stat_fingerprint": fingerprint,
    }
    key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            payload = _read_digest_cache(cache)
            entry = payload.get(key)
            if isinstance(entry, Mapping) and entry.get("digest") == expected:
                if _stat_fingerprint(asset) == fingerprint:
                    return expected
            actual = _digest_path(asset)
            if actual != expected:
                raise ValueError(
                    f"asset digest mismatch for {asset}: expected={expected} actual={actual}"
                )
            payload[key] = {**key_payload, "digest": actual}
            _atomic_write_json(cache, payload)
            return actual
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_digest_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _read_json_mapping(path, "asset digest cache")
    return dict(payload)


def _stat_fingerprint(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink():
        raise ValueError(f"asset fingerprint rejects symlink: {path}")
    if path.is_file():
        items = [(".", path)]
    elif path.is_dir():
        items = [
            (item.relative_to(path).as_posix(), item)
            for item in sorted(path.rglob("*"))
            if item.is_file() or item.is_symlink()
        ]
    else:
        raise FileNotFoundError(f"asset path does not exist: {path}")
    fingerprint = []
    for relative, item in items:
        if item.is_symlink():
            raise ValueError(f"asset fingerprint rejects symlink entry: {item}")
        stat = item.stat()
        fingerprint.append(
            {
                "relative": relative,
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
            }
        )
    return fingerprint


def _validate_expected_digest(
    config: Mapping[str, Any], fields: Sequence[str], actual: str, label: str
) -> None:
    values = [str(config[field]) for field in fields if config.get(field)]
    if len(set(values)) > 1:
        raise ValueError(f"{label} expected digest fields disagree: {values!r}")
    if values and values[0] != actual:
        raise ValueError(
            f"{label} asset digest mismatch: expected={values[0]} actual={actual}"
        )


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"guidance config {field} must be a lowercase SHA256 digest")
    return text


def _result_with_trace(
    result: GuidanceResult,
    trace,
    mode: str,
    *,
    interval_contract: Mapping[str, Any] | None = None,
) -> GuidanceResult:
    if len(trace) != int(result.nfe):
        raise RuntimeError(f"{mode} flow-map trace length does not match NFE")
    diagnostics = dict(result.diagnostics)
    if interval_contract is not None:
        required = {
            "active_guidance_intervals",
            "interval_diagnostics_enabled",
            "interval_diagnostics",
            "algorithm_nfe",
            "diagnostic_nfe",
            "diagnostic_flow_map_trace",
        }
        missing = sorted(required - set(diagnostics))
        if missing:
            raise RuntimeError(
                f"R9 guidance result is missing diagnostics: {missing!r}"
            )
        algorithm_nfe = int(diagnostics["algorithm_nfe"])
        diagnostic_nfe = int(diagnostics["diagnostic_nfe"])
        diagnostic_trace = diagnostics["diagnostic_flow_map_trace"]
        if algorithm_nfe != int(result.nfe):
            raise RuntimeError(
                "R9 guidance result algorithm NFE disagrees with legacy NFE"
            )
        if algorithm_nfe != int(interval_contract["expected_algorithm_nfe"]):
            raise RuntimeError(
                "R9 guidance result algorithm NFE disagrees with its contract"
            )
        if diagnostic_nfe != int(interval_contract["expected_diagnostic_nfe"]):
            raise RuntimeError(
                "R9 guidance result diagnostic NFE disagrees with its contract"
            )
        if list(trace) != interval_contract["expected_algorithm_trace"]:
            raise RuntimeError(
                "R9 guidance result algorithm trace disagrees with its contract"
            )
        if (
            not isinstance(diagnostic_trace, list)
            or len(diagnostic_trace) != diagnostic_nfe
        ):
            raise RuntimeError(
                "R9 guidance result diagnostic trace length disagrees with NFE"
            )
        if diagnostic_trace != interval_contract["expected_diagnostic_trace"]:
            raise RuntimeError(
                "R9 guidance result diagnostic trace disagrees with its contract"
            )
        if any(
            not isinstance(entry, Mapping) or entry.get("kind") != "interval_diagnostic"
            for entry in diagnostic_trace
        ):
            raise RuntimeError(
                "R9 guidance result diagnostic trace contains an invalid kind"
            )
        if (
            diagnostics["active_guidance_intervals"]
            != interval_contract["active_guidance_intervals"]
        ):
            raise RuntimeError(
                "R9 guidance result active interval mask disagrees with its contract"
            )
        if (
            diagnostics["interval_diagnostics_enabled"]
            is not interval_contract["collect_interval_diagnostics"]
        ):
            raise RuntimeError(
                "R9 guidance result diagnostic toggle disagrees with its contract"
            )
        interval_diagnostics = diagnostics["interval_diagnostics"]
        if interval_contract["collect_interval_diagnostics"]:
            if (
                not isinstance(interval_diagnostics, Mapping)
                or tuple(interval_diagnostics) != R9_GUIDANCE_INTERVAL_IDS
            ):
                raise RuntimeError(
                    "R9 guidance result interval diagnostics are incomplete"
                )
        elif interval_diagnostics != {}:
            raise RuntimeError(
                "R9 guidance result unexpectedly contains interval diagnostics"
            )
        _require_finite_diagnostic_tree(
            diagnostics["interval_diagnostics"],
            label="R9 guidance interval diagnostics",
        )
        _require_finite_diagnostic_tree(
            diagnostic_trace,
            label="R9 guidance diagnostic trace",
        )
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
    session_id: str,
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
        "session_id": str(session_id),
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


def _validate_session_history(
    sessions, *, require_contiguous_indices: bool = True
) -> None:
    required = {
        "session_id",
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
    session_ids: set[str] = set()
    for index, session in enumerate(sessions):
        if not isinstance(session, Mapping) or not required.issubset(session):
            raise ValueError(f"invalid session history row at index {index}")
        session_id = str(session["session_id"])
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError(f"invalid session ID at row {index}")
        if session_id in session_ids:
            raise ValueError(f"duplicate session ID at row {index}")
        session_ids.add(session_id)
        if require_contiguous_indices and int(session["session_index"]) != index:
            raise ValueError(f"session history index mismatch at row {index}")
        if int(session["session_index"]) < 0:
            raise ValueError(f"negative session history index at row {index}")
        memory = session["max_memory"]
        if not isinstance(memory, Mapping) or not {
            "allocated_bytes",
            "reserved_bytes",
        }.issubset(memory):
            raise ValueError(f"invalid session memory record at index {index}")


def _same_session(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_core = {
        key: value for key, value in left.items() if key != "recovered_after_crash"
    }
    right_core = {
        key: value for key, value in right.items() if key != "recovered_after_crash"
    }
    return _json_safe(left_core) == _json_safe(right_core)


def aggregate_session_memory(sessions) -> dict[str, int]:
    if not sessions:
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": max(
            int(item["max_memory"]["allocated_bytes"]) for item in sessions
        ),
        "reserved_bytes": max(
            int(item["max_memory"]["reserved_bytes"]) for item in sessions
        ),
    }


def _aggregate_timing(
    rows, sessions, resumed_count: int, generated_count: int
) -> dict[str, Any]:
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
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
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
    return torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(
        device
    )
