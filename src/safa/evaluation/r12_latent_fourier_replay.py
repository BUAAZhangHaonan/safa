from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from safa.evaluation.meanflow_guidance_runner import (
    _as_float_tensor,
    _records_from_feature_dataset,
    _sample_channels,
    asset_contract_from_config,
    build_frozen_runtime,
    execute_guidance_mode,
    materialize_runtime_guidance_config,
)
from safa.evaluation.r9_determinism import (
    apply_r9_strict_cuda_determinism,
    assert_r9_strict_cuda_determinism,
)
from safa.utils.sampling import make_x_init_for_sample_ids


class R12FourierReplayError(RuntimeError):
    pass


SNAPSHOT_STEPS = (0, 12, 16)
BOOTSTRAP_SEED = 91637
BOOTSTRAP_ITERATIONS = 10_000


def pair_preserving_lane(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    lane_index: int,
    lane_count: int,
) -> list[dict[str, Any]]:
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(lane_count, bool) or lane_count <= 0:
        raise ValueError("lane_count must be positive")
    if isinstance(lane_index, bool) or not 0 <= lane_index < lane_count:
        raise ValueError("lane_index must be within lane_count")
    selected: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch = records[start : start + batch_size]
        if len(batch) != batch_size:
            raise ValueError("pair-preserving replay forbids a partial final batch")
        if batch_index % lane_count != lane_index:
            continue
        for offset, record in enumerate(batch):
            selected.append({**dict(record), "original_ordinal": start + offset})
    if not selected:
        raise ValueError("pair-preserving lane contains no samples")
    return selected


def _sample_spectral_snapshots(
    snapshots: Sequence[Mapping[str, Any]], *, index: int, batch_size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        shell = snapshot["per_channel_shell_energy"]
        spatial = snapshot["per_channel_spatial_energy"]
        high = snapshot["per_channel_high_frequency_energy"]
        for name, tensor in (("shell", shell), ("spatial", spatial), ("high", high)):
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != batch_size:
                raise R12FourierReplayError(
                    f"spectral {name} tensor has an invalid batch axis"
                )
            if not torch.isfinite(tensor).all().item():
                raise R12FourierReplayError(f"spectral {name} tensor is non-finite")
        rows.append(
            {
                "step": int(snapshot["step"]),
                "norm": str(snapshot["norm"]),
                "height": int(snapshot["height"]),
                "width": int(snapshot["width"]),
                "radius_squared": [int(value) for value in snapshot["radius_squared"]],
                "full_spectrum_coefficient_count": [
                    int(value)
                    for value in snapshot["full_spectrum_coefficient_count"]
                ],
                "high_frequency_min_radius": float(
                    snapshot["high_frequency_min_radius"]
                ),
                "per_channel_shell_energy": shell[index].detach().cpu().tolist(),
                "per_channel_spatial_energy": spatial[index].detach().cpu().tolist(),
                "per_channel_high_frequency_energy": high[index]
                .detach()
                .cpu()
                .tolist(),
            }
        )
    if tuple(row["step"] for row in rows) != SNAPSHOT_STEPS:
        raise R12FourierReplayError("spectral snapshot steps are not exactly 0/12/16")
    return rows


def validate_replay_config(config: Mapping[str, Any], dataset_id: str) -> None:
    expected_manifest_suffix = {
        "regular32": "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl",
        "sharpness_tail32": "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl",
    }
    if dataset_id not in expected_manifest_suffix:
        raise ValueError(f"unsupported R12 replay dataset {dataset_id!r}")
    exact = {
        "experiment_contract": "safa_r9_meanflow_v1",
        "attention_backend": "native",
        "mode": "initial_noise",
        "phase": "diagnose",
        "projection": "fixed_radius",
        "eta": 0.5,
        "num_updates": 16,
        "seed": 7919,
        "sampling_seed": 7919,
        "batch_size": 2,
        "max_samples": 32,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise ValueError(
                f"R12 replay requires config.{field}={expected!r}, got {config.get(field)!r}"
            )
    if str(config.get("sample_id_manifest")) != expected_manifest_suffix[dataset_id]:
        raise ValueError("R12 replay config uses the wrong ordered sample manifest")


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    encoded = json.dumps(list(sample_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def run_replay_lane(
    config: Mapping[str, Any],
    *,
    dataset_id: str,
    lane_index: int,
    lane_count: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    validate_replay_config(config, dataset_id)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to reuse replay output {output}")
    resolved = materialize_runtime_guidance_config(config)
    validate_replay_config(resolved, dataset_id)
    applied = apply_r9_strict_cuda_determinism(resolved, torch_module=torch)
    if resolved.get("r9_execution_contract") != applied:
        raise R12FourierReplayError("strict R9 execution contract was not applied")
    assert_r9_strict_cuda_determinism(torch_module=torch)

    from safa.data.feature_dataset import FeatureAlignedAffectNet
    from safa.training.transforms import generator_image_transform
    from safa.utils.device import require_cuda_device

    dataset = FeatureAlignedAffectNet(
        resolved["index"],
        resolved["features"],
        resolved["e0_checkpoint"],
        transform=generator_image_transform(int(resolved.get("pixel_image_size", 256))),
    )
    records = _records_from_feature_dataset(dataset, resolved)
    batch_size = int(resolved["batch_size"])
    selected = pair_preserving_lane(
        records,
        batch_size=batch_size,
        lane_index=lane_index,
        lane_count=lane_count,
    )
    assets = asset_contract_from_config(resolved)
    device = require_cuda_device(str(resolved["device"]))
    runtime = build_frozen_runtime(
        resolved,
        device=device,
        checkpoint_sha256=str(assets["checkpoint"]["sha256"]),
        asset_contract=assets,
    )
    channels = _sample_channels(runtime.generator, resolved)
    sampling_seed = int(resolved["sampling_seed"])
    image_size = int(resolved.get("image_size", 32))
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
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
        result = execute_guidance_mode(
            config=resolved,
            generator=runtime.generator,
            codec=runtime.codec,
            e0=runtime.e0,
            x_init=x_init,
            target_z0=target_z0,
            schedule=None,
            initial_noise_snapshot_steps=SNAPSHOT_STEPS,
        )
        if result.nfe != 17:
            raise R12FourierReplayError(f"replay candidate NFE must be 17, got {result.nfe}")
        history = [float(value) for value in result.diagnostics["loss_history"]]
        if len(history) != 17 or not all(math.isfinite(value) for value in history):
            raise R12FourierReplayError("replay loss history is invalid")
        snapshots = result.diagnostics.get("spectral_snapshots")
        if not isinstance(snapshots, Sequence):
            raise R12FourierReplayError("replay did not return spectral snapshots")
        for local_index, record in enumerate(batch):
            rows.append(
                {
                    "schema_version": 1,
                    "contract_type": "safa_r12_latent_fourier_replay_row_v1",
                    "dataset_id": dataset_id,
                    "lane_index": lane_index,
                    "lane_count": lane_count,
                    "original_ordinal": int(record["original_ordinal"]),
                    "sample_id": str(record["sample_id"]),
                    "candidate_nfe": int(result.nfe),
                    "loss_history": history,
                    "projection": str(result.diagnostics["projection"]),
                    "eta": float(result.diagnostics["eta"]),
                    "num_updates": int(result.diagnostics["num_updates"]),
                    "initial_norm": float(
                        result.diagnostics["initial_norm"][local_index].detach().cpu()
                    ),
                    "final_norm": float(
                        result.diagnostics["final_norm"][local_index].detach().cpu()
                    ),
                    "spectral_snapshots": _sample_spectral_snapshots(
                        snapshots, index=local_index, batch_size=len(batch)
                    ),
                }
            )
    rows.sort(key=lambda row: int(row["original_ordinal"]))
    sample_ids = [str(row["sample_id"]) for row in rows]
    _write_jsonl(output / "per_sample.jsonl", rows)
    completion = {
        "schema_version": 1,
        "contract_type": "safa_r12_latent_fourier_replay_lane_v1",
        "status": "complete",
        "dataset_id": dataset_id,
        "lane_index": lane_index,
        "lane_count": lane_count,
        "sample_count": len(rows),
        "ordered_sample_id_sha256": _sample_id_digest(sample_ids),
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "batch_size": batch_size,
        "candidate_nfe": 17,
        "seed": sampling_seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    _write_json(output / "completion.json", completion)
    return completion


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise R12FourierReplayError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def merge_and_bind_replay_rows(
    lane_paths: Sequence[str | Path],
    *,
    expected_sample_ids: Sequence[str],
    existing_u16_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [row for path in lane_paths for row in read_jsonl(path)]
    rows.sort(key=lambda row: int(row["original_ordinal"]))
    ordinals = [int(row["original_ordinal"]) for row in rows]
    if ordinals != list(range(len(expected_sample_ids))):
        raise R12FourierReplayError("merged replay ordinals are not the exact full order")
    if [row["sample_id"] for row in rows] != list(expected_sample_ids):
        raise R12FourierReplayError("merged replay sample IDs disagree with manifest")
    existing = {str(row["sample_id"]): row for row in existing_u16_rows}
    if set(existing) != set(expected_sample_ids):
        raise R12FourierReplayError("existing u16 rows disagree with manifest")
    for row in rows:
        reference = existing[str(row["sample_id"])]
        if row["loss_history"] != reference["route_diagnostics"]["loss_history"]:
            raise R12FourierReplayError(
                f"loss-history binding failed for {row['sample_id']}"
            )
        if int(row["candidate_nfe"]) != int(reference["candidate_nfe"]):
            raise R12FourierReplayError(
                f"candidate-NFE binding failed for {row['sample_id']}"
            )
        if float(row["initial_norm"]) != float(
            reference["route_diagnostics"]["initial_norm"]
        ):
            raise R12FourierReplayError(
                f"initial-noise binding failed for {row['sample_id']}"
            )
    return rows


def high_frequency_drift(row: Mapping[str, Any], step: int) -> float:
    snapshots = {int(item["step"]): item for item in row["spectral_snapshots"]}
    if set(snapshots) != set(SNAPSHOT_STEPS):
        raise R12FourierReplayError("row snapshot steps are not exactly 0/12/16")
    initial = math.fsum(
        float(value) for value in snapshots[0]["per_channel_high_frequency_energy"]
    )
    final = math.fsum(
        float(value) for value in snapshots[step]["per_channel_high_frequency_energy"]
    )
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0 or final <= 0:
        raise R12FourierReplayError("high-frequency energy must be positive and finite")
    return math.log(final / initial)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("rank input must be a finite vector with at least two values")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(values: Sequence[float], outcomes: Sequence[float]) -> float:
    if len(values) != len(outcomes):
        raise ValueError("Spearman vectors must have equal length")
    left = _rankdata(values)
    right = _rankdata(outcomes)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant vector")
    return float(np.dot(left, right) / denominator)


def association(
    values: Sequence[float],
    outcomes: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    point = spearman(values, outcomes)
    left = np.asarray(values, dtype=np.float64)
    right = np.asarray(outcomes, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = []
    dropped = 0
    for _ in range(iterations):
        index = rng.integers(0, left.size, size=left.size)
        try:
            boot.append(spearman(left[index], right[index]))
        except ValueError:
            dropped += 1
    if len(boot) < max(100, iterations // 2):
        raise R12FourierReplayError("too many degenerate bootstrap resamples")
    lower, upper = np.quantile(np.asarray(boot), [0.025, 0.975])
    return {
        "spearman": point,
        "spearman_ci95": [float(lower), float(upper)],
        "bootstrap_kept": len(boot),
        "bootstrap_dropped_degenerate": dropped,
    }


def bottom_worst_overlap(
    sample_ids: Sequence[str],
    values: Sequence[float],
    outcomes: Sequence[float],
    *,
    k: int = 8,
) -> dict[str, Any]:
    if not (len(sample_ids) == len(values) == len(outcomes)) or len(sample_ids) < k:
        raise ValueError("overlap inputs have incompatible sizes")
    bottom = {
        sample_id
        for _, sample_id in sorted(
            zip(values, sample_ids, strict=True), key=lambda item: (item[0], item[1])
        )[:k]
    }
    worst = {
        sample_id
        for _, sample_id in sorted(
            zip(outcomes, sample_ids, strict=True), key=lambda item: (item[0], item[1])
        )[:k]
    }
    return {
        "k": k,
        "overlap": len(bottom & worst),
        "bottom_h_sample_ids": sorted(bottom),
        "worst_retention_sample_ids": sorted(worst),
    }


def frozen_criteria(
    regular_rows: Sequence[Mapping[str, Any]],
    tail_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    if len(regular_rows) != 32 or len(tail_rows) != 32:
        raise R12FourierReplayError("frozen criteria require regular32 and tail32")
    tail: dict[str, Any] = {}
    regular: dict[str, Any] = {}
    for step in (12, 16):
        h_tail = [float(row[f"h{step}"]) for row in tail_rows]
        retention = [float(row["sharpness_retention"]) for row in tail_rows]
        tail_assoc = association(h_tail, retention, iterations=iterations)
        tail[str(step)] = {
            **tail_assoc,
            "bottom8_worst8": bottom_worst_overlap(
                [str(row["sample_id"]) for row in tail_rows], h_tail, retention
            ),
        }
        h_regular = [float(row[f"h{step}"]) for row in regular_rows]
        privacy = [float(row["arcface_delta"]) for row in regular_rows]
        regular[str(step)] = association(h_regular, privacy, iterations=iterations)

    tail_same_positive_sign = all(tail[str(step)]["spearman"] > 0 for step in (12, 16))
    tail_rho_threshold = all(
        tail[str(step)]["spearman"] >= 0.40 for step in (12, 16)
    )
    tail_ci_evidence = any(
        tail[str(step)]["spearman_ci95"][0] > 0 for step in (12, 16)
    )
    tail_overlap = all(
        tail[str(step)]["bottom8_worst8"]["overlap"] >= 4 for step in (12, 16)
    )
    regular_decoupled = all(
        abs(regular[str(step)]["spearman"]) < 0.20
        and regular[str(step)]["spearman_ci95"][0] <= 0.0
        <= regular[str(step)]["spearman_ci95"][1]
        for step in (12, 16)
    )
    licensed = all(
        (
            tail_same_positive_sign,
            tail_rho_threshold,
            tail_ci_evidence,
            tail_overlap,
            regular_decoupled,
        )
    )
    return {
        "tail": tail,
        "regular": regular,
        "checks": {
            "tail_same_positive_sign": tail_same_positive_sign,
            "tail_rho_at_least_0p40_both": tail_rho_threshold,
            "tail_bootstrap_lower_positive_at_least_one": tail_ci_evidence,
            "tail_bottom8_overlap_at_least4_both": tail_overlap,
            "regular_abs_rho_below_0p20_and_ci_contains_zero_both": regular_decoupled,
        },
        "fourier_projection_licensed": licensed,
        "stop_required": not licensed,
    }
