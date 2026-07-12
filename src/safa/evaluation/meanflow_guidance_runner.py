from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
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
    {"native", "semigroup", "official_head_current_xt", "paper_algorithm_split", "noise_oracle"}
)
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
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"guidance mode must be one of {sorted(SUPPORTED_MODES)}, got {mode!r}")
    resolved["mode"] = mode
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
    mode = str(config.get("mode", ""))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported guidance mode {mode!r}")
    transport_condition = generator.make_null_condition(
        batch_size=x_init.shape[0], device=x_init.device, dtype=x_init.dtype
    )
    counted = CountedFlowMap(generator)
    if mode == "native":
        with torch.no_grad():
            latent = counted(x_init, transport_condition, t=1.0, r=0.0).detach()
        return GuidanceResult(latent=latent, nfe=1, diagnostics={"mode": mode})
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
            diagnostics={"mode": mode, "semigroup": report},
        )
    if mode == "noise_oracle":
        return optimize_initial_noise(
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
    if schedule is None:
        raise ValueError(f"{mode} requires a locked schedule")
    guided_times = schedule["guided_times"]
    unguided_times = schedule["unguided_times"]
    if mode == "official_head_current_xt":
        return sample_official_head_current_xt(
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
    if mode == "paper_algorithm_split":
        return sample_paper_algorithm_split(
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
    raise AssertionError(f"unhandled guidance mode {mode!r}")


def run_guidance_records(
    *,
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    runtime: GuidanceRuntime,
    output_dir: str | Path,
    shard_index: int,
    num_shards: int,
    allow_overwrite: bool,
) -> dict[str, Any]:
    output = Path(output_dir)
    selected = deterministic_shard([dict(row) for row in records], shard_index, num_shards)
    _validate_generation_records(selected)
    expected_ids = [str(row["sample_id"]) for row in selected]
    if not expected_ids:
        raise ValueError(f"shard {shard_index}/{num_shards} contains no samples")
    generated_dir = output / "generated_images"
    per_sample_path = output / "per_sample.jsonl"
    run_manifest_path = output / "run_manifest.json"
    sample_manifest_path = output / "sample_id_manifest.jsonl"
    semigroup_path = output / "semigroup.json"
    resume_contract_path = output / "resume_contract.json"

    if run_manifest_path.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to replace completed output: {run_manifest_path}")
    if allow_overwrite:
        _clear_owned_outputs(output)
    output.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    resume_contract = {
        "checkpoint_path": str(runtime.checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "checkpoint_state": runtime.checkpoint_state,
        "config": dict(config),
        "sample_id_sha256": _sample_id_digest(expected_ids),
        "shard": {"index": int(shard_index), "count": int(num_shards)},
    }
    if resume_contract_path.exists():
        existing_contract = _read_json_mapping(resume_contract_path, "resume contract")
        if existing_contract != _json_safe(resume_contract):
            raise ValueError("existing resume contract disagrees with checkpoint, config, seed, mode, or shard")
    else:
        _write_json(resume_contract_path, resume_contract)

    expected_manifest_rows = [
        {"sample_id": row["sample_id"], "source": str(row["source"])} for row in selected
    ]
    if sample_manifest_path.exists():
        existing_manifest = read_ordered_sample_manifest(sample_manifest_path)
        if existing_manifest != expected_manifest_rows:
            raise ValueError("existing sample_id_manifest.jsonl disagrees with the deterministic shard")
    else:
        _write_jsonl(sample_manifest_path, expected_manifest_rows, mode="x")

    completed_rows = _read_optional_jsonl(per_sample_path)
    remaining_ids = resume_remaining_ids(expected_ids, completed_rows)
    completed_count = len(expected_ids) - len(remaining_ids)
    for row in completed_rows:
        generated = Path(str(row.get("generated", "")))
        if not generated.is_file():
            raise FileNotFoundError(f"resume row points to a missing generated image: {generated}")
    record_by_id = {str(row["sample_id"]): row for row in selected}
    remaining_records = [record_by_id[sample_id] for sample_id in remaining_ids]

    schedule = config.get("locked_schedule")
    if schedule is not None and not isinstance(schedule, Mapping):
        raise ValueError("config.locked_schedule must be a mapping")
    mode = str(config.get("mode", ""))
    batch_size = _positive_int(config.get("batch_size", 1), "batch_size")
    sampling_seed = int(config.get("sampling_seed", config.get("seed", 0)))
    image_size = _positive_int(config.get("image_size", 32), "image_size")
    channels = _sample_channels(runtime.generator, config)
    completed_nfe_values = [int(row["nfe"]) for row in completed_rows if "nfe" in row]
    batch_nfe_values: list[int] = []
    _cuda_start(runtime.device)
    start = time.perf_counter()
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
        result = execute_guidance_mode(
            config=config,
            generator=runtime.generator,
            codec=runtime.codec,
            e0=runtime.e0,
            x_init=x_init,
            target_z0=target_z0,
            schedule=schedule,
        )
        batch_nfe_values.append(int(result.nfe))
        with torch.no_grad():
            generated = runtime.codec.decode(result.latent)
            generated_embedding = runtime.e0(normalize_for_e0(generated))["embedding"]
            cosine = F.cosine_similarity(generated_embedding, target_z0, dim=1)
        if not torch.isfinite(generated).all() or not torch.isfinite(cosine).all():
            raise FloatingPointError("guidance runner produced non-finite generated output or cosine")
        batch_semigroup = {}
        if mode == "semigroup":
            batch_semigroup = {
                row["sample_id"]: row["splits"]
                for row in _semigroup_batch_rows(
                    result=result,
                    sample_ids=sample_ids,
                    codec=runtime.codec,
                    e0=runtime.e0,
                    direct_images=generated,
                )
            }
        for local_index, sample_id in enumerate(sample_ids):
            ordinal = expected_ids.index(sample_id)
            path = generated_dir / f"{ordinal:08d}__{_safe_sample_id(sample_id)}.png"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite generated image: {path}")
            _save_image(generated[local_index], path)
            source = str(batch[local_index]["source"])
            row = {
                "sample_id": sample_id,
                "source": source,
                "generated": str(path),
                "cosine": float(cosine[local_index].detach().cpu()),
                "mode": mode,
                "shard": int(shard_index),
                "nfe": int(result.nfe),
                "route_diagnostics": _per_sample_diagnostics(
                    result.diagnostics, local_index, len(sample_ids)
                ),
            }
            if mode == "semigroup":
                row["semigroup"] = batch_semigroup[sample_id]
            _append_jsonl(per_sample_path, row)
    _cuda_sync(runtime.device)
    elapsed = time.perf_counter() - start
    generated_count = len(remaining_records)
    all_nfe_values = completed_nfe_values + batch_nfe_values
    if all_nfe_values and len(set(all_nfe_values)) != 1:
        raise RuntimeError(f"route NFE changed across batches or resume rows: {all_nfe_values!r}")
    route_nfe = all_nfe_values[0] if all_nfe_values else 0
    allocated, reserved = _cuda_peak_memory(runtime.device)
    final_rows = _read_optional_jsonl(per_sample_path)
    if resume_remaining_ids(expected_ids, final_rows):
        raise RuntimeError("guidance run ended before every deterministic shard sample was written")
    generated_files = {path.resolve() for path in generated_dir.glob("*.png")}
    recorded_files = {Path(str(row["generated"])).resolve() for row in final_rows}
    if generated_files != recorded_files:
        raise RuntimeError("generated image files do not exactly match per_sample.jsonl")
    cosine_summary = _finite_summary([float(row["cosine"]) for row in final_rows])
    if mode == "semigroup":
        _write_json(
            semigroup_path,
            {
                "mode": mode,
                "split_times": [float(value) for value in config.get("split_times", (0.25, 0.5, 0.75))],
                "rows": [
                    {"sample_id": row["sample_id"], "splits": row["semigroup"]}
                    for row in final_rows
                ],
            },
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
        "cosine": cosine_summary,
        "shard": {"index": int(shard_index), "count": int(num_shards)},
        "nfe": int(route_nfe),
        "nfe_batch_calls_this_invocation": int(sum(batch_nfe_values)),
        "timing": {
            "elapsed_seconds": float(elapsed),
            "images_per_second": float(generated_count / elapsed) if generated_count else 0.0,
            "generated_this_invocation": generated_count,
            "resumed_count": completed_count,
        },
        "max_memory": {
            "allocated_bytes": int(allocated),
            "reserved_bytes": int(reserved),
        },
        "schedule": _json_safe(schedule),
        "config": _json_safe(dict(config)),
        "artifacts": {
            "generated_dir": str(generated_dir),
            "per_sample_jsonl": str(per_sample_path),
            "semigroup_json": str(semigroup_path) if mode == "semigroup" else None,
            "resume_contract": str(resume_contract_path),
        },
    }
    _write_json(run_manifest_path, manifest)
    return manifest


def run_guidance_from_config(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    shard_index: int = 0,
    num_shards: int = 1,
    allow_overwrite: bool = False,
    explicit_t_cut: float | None = None,
) -> dict[str, Any]:
    resolved = validate_guidance_config(config)
    from safa.data.feature_dataset import FeatureAlignedAffectNet
    from safa.models.e0 import freeze_e0, load_e0_checkpoint
    from safa.training.latent_codec import build_latent_codec_from_train_config
    from safa.training.transforms import generator_image_transform
    from safa.utils.device import require_cuda_device
    from safa.utils.hashing import sha256_file

    required = ("device", "index", "features", "e0_checkpoint", "vae_path", "vae_scaling_factor")
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
    if output_manifest.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to replace completed output: {output_manifest}")

    device = require_cuda_device(str(resolved["device"]))
    generator, checkpoint_state = load_ema_generator(checkpoint, device=device)
    e0, _ = load_e0_checkpoint(resolved["e0_checkpoint"], device="cpu")
    e0 = e0.to(device)
    freeze_e0(e0)
    codec_config = dict(resolved)
    codec_config.update(
        {
            "latent_training": True,
            "image_size": int(EXPECTED_MODEL_CONFIG["image_size"]),
            "pixel_image_size": int(resolved.get("pixel_image_size", 256)),
        }
    )
    codec = build_latent_codec_from_train_config(codec_config, device)
    if codec is None:
        raise RuntimeError("MeanFlow guidance requires the configured latent VAE")
    freeze_guidance_stack(generator, codec, e0)

    dataset = FeatureAlignedAffectNet(
        resolved["index"],
        resolved["features"],
        resolved["e0_checkpoint"],
        transform=generator_image_transform(int(resolved.get("pixel_image_size", 256))),
    )
    records = _records_from_feature_dataset(dataset, resolved)
    runtime = GuidanceRuntime(
        generator=generator,
        codec=codec,
        e0=e0,
        device=device,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_state=checkpoint_state,
    )
    return run_guidance_records(
        config=resolved,
        records=records,
        runtime=runtime,
        output_dir=output_dir,
        shard_index=shard_index,
        num_shards=num_shards,
        allow_overwrite=allow_overwrite,
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


def _save_image(image: torch.Tensor, path: Path) -> None:
    from torchvision.utils import save_image

    save_image(image.detach().cpu(), path)


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def _clear_owned_outputs(output: Path) -> None:
    for path in (
        output / "per_sample.jsonl",
        output / "run_manifest.json",
        output / "sample_id_manifest.jsonl",
        output / "semigroup.json",
        output / "resume_contract.json",
    ):
        if path.exists():
            path.unlink()
    generated_dir = output / "generated_images"
    if generated_dir.is_dir():
        for path in generated_dir.glob("*.png"):
            path.unlink()


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


def _cuda_start(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_memory(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
