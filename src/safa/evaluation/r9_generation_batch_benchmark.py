from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


HEADROOM_BYTES = 2 * 1024**3
MAX_SLOTS_PER_GPU = 4
BENCHMARK_BATCH_SIZES = (2, 4)
_TIMING_FIELDS = frozenset(
    {
        "candidate_generation_seconds",
        "native_generation_seconds",
        "generation_seconds",
        "io_seconds",
    }
)
_PATH_FIELDS = frozenset({"generated", "native", "source"})


class GenerationBatchBenchmarkError(ValueError):
    """Raised when measured batch evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class BatchRunEvidence:
    arm_id: str
    batch_size: int
    output_dir: Path
    gpu_uuid: str
    free_vram_before_bytes: int
    peak_process_tree_rss_bytes: int

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise GenerationBatchBenchmarkError("benchmark arm ID is required")
        if self.batch_size not in BENCHMARK_BATCH_SIZES:
            raise GenerationBatchBenchmarkError("benchmark batch size must be 2 or 4")
        if not self.gpu_uuid:
            raise GenerationBatchBenchmarkError("benchmark GPU UUID is required")
        _positive_int(self.free_vram_before_bytes, "free VRAM before")
        _positive_int(self.peak_process_tree_rss_bytes, "peak process-tree RSS")


@dataclass(frozen=True)
class BenchmarkGpuSnapshot:
    index: int
    uuid: str
    total_bytes: int
    free_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise GenerationBatchBenchmarkError("GPU index must be nonnegative")
        if not self.uuid:
            raise GenerationBatchBenchmarkError("GPU UUID is required")
        total = _positive_int(self.total_bytes, "GPU total VRAM")
        free = _positive_int(self.free_bytes, "GPU free VRAM")
        if free > total:
            raise GenerationBatchBenchmarkError("GPU free VRAM exceeds total VRAM")


def build_generation_batch_benchmark_contract(
    *,
    repo_root: Path,
    campaign_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    sample_count: int,
    seed: int,
    continuation_contract_sha256: str,
    request_sha256: str,
    required_arm_ids: Sequence[str],
    gpu_snapshots: Sequence[BenchmarkGpuSnapshot],
    evidence: Sequence[BatchRunEvidence],
) -> dict[str, Any]:
    """Compare measured batch-2/4 runs and bind the only allowed C batch policy."""
    root = Path(repo_root).resolve()
    _sha(manifest_sha256, "manifest SHA256")
    _sha(continuation_contract_sha256, "continuation contract SHA256")
    _sha(request_sha256, "benchmark request SHA256")
    _positive_int(sample_count, "sample count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GenerationBatchBenchmarkError("seed must be a nonnegative integer")
    manifest = _repo_file(root, manifest_path, "benchmark manifest")
    if _file_sha256(manifest) != manifest_sha256:
        raise GenerationBatchBenchmarkError("benchmark manifest SHA256 mismatch")
    expected_arms = tuple(required_arm_ids)
    if (
        len(expected_arms) != 3
        or len(set(expected_arms)) != 3
        or "native" not in expected_arms
    ):
        raise GenerationBatchBenchmarkError(
            "benchmark requires native and exactly two candidate arms"
        )
    snapshots = tuple(gpu_snapshots)
    if (
        tuple(sorted(row.index for row in snapshots)) != (0, 1, 2, 3)
        or len({row.uuid for row in snapshots}) != 4
    ):
        raise GenerationBatchBenchmarkError(
            "benchmark requires unique snapshots for GPU indices 0-3"
        )
    grouped: dict[str, dict[int, BatchRunEvidence]] = {}
    for row in evidence:
        by_batch = grouped.setdefault(row.arm_id, {})
        if row.batch_size in by_batch:
            raise GenerationBatchBenchmarkError("duplicate arm/batch evidence")
        by_batch[row.batch_size] = row
    if set(grouped) != set(expected_arms) or any(
        set(rows) != {2, 4} for rows in grouped.values()
    ):
        raise GenerationBatchBenchmarkError(
            "every required arm needs exactly batch-2 and batch-4 evidence"
        )

    arm_results: list[dict[str, Any]] = []
    all_equivalent = True
    loaded_by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for arm_id in sorted(grouped):
        loaded = {
            batch_size: _load_run(root, row, sample_count=sample_count, seed=seed)
            for batch_size, row in grouped[arm_id].items()
        }
        loaded_by_arm[arm_id] = loaded
        comparison = _compare_runs(loaded[2], loaded[4])
        all_equivalent = all_equivalent and comparison["bit_identical"]
        arm_results.append(
            {
                "arm_id": arm_id,
                "comparison": comparison,
                "runs": [
                    _run_contract_row(grouped[arm_id][batch_size], loaded[batch_size])
                    for batch_size in BENCHMARK_BATCH_SIZES
                ],
            }
        )

    slot_claims = {
        batch_size: max(
            loaded_by_arm[arm_id][batch_size]["peak_reserved_bytes"]
            for arm_id in grouped
        )
        for batch_size in BENCHMARK_BATCH_SIZES
    }
    ram_budgets = {
        batch_size: math.ceil(
            max(
                grouped[arm_id][batch_size].peak_process_tree_rss_bytes
                for arm_id in grouped
            )
            * 1.10
        )
        for batch_size in BENCHMARK_BATCH_SIZES
    }
    capacities = {
        batch_size: {
            row.uuid: _slot_capacity(
                free_bytes=row.free_bytes,
                peak_reserved_bytes=slot_claims[batch_size],
            )
            for row in sorted(snapshots, key=lambda item: item.index)
        }
        for batch_size in BENCHMARK_BATCH_SIZES
    }
    min_capacities = {
        batch_size: min(by_gpu.values())
        for batch_size, by_gpu in capacities.items()
    }
    if all_equivalent and min_capacities[4] >= 2:
        selected_batch_size = 4
        selected_slots_per_gpu = 2
        decision = "batch4_exact_and_multi_slot_safe"
        status = "ready"
    elif min_capacities[2] >= 4:
        selected_batch_size = 2
        selected_slots_per_gpu = 4
        decision = (
            "batch2_required_due_to_batch4_non_equivalence"
            if not all_equivalent
            else "batch2_required_due_to_batch4_capacity"
        )
        status = "ready"
    else:
        selected_batch_size = None
        selected_slots_per_gpu = None
        decision = "blocked_batch2_cannot_support_four_slots_per_gpu"
        status = "blocked"
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_generation_batch_benchmark_v1",
        "campaign_id": campaign_id,
        "continuation_contract_sha256": continuation_contract_sha256,
        "request_sha256": request_sha256,
        "status": status,
        "manifest": {
            "path": str(manifest.relative_to(root)),
            "sha256": manifest_sha256,
            "sample_count": sample_count,
        },
        "seed": seed,
        "requirements": {
            "batch_sizes": [2, 4],
            "png_sha256_exact": True,
            "final_latent_sha256_exact": True,
            "semantic_rows_exact": True,
            "headroom_bytes": HEADROOM_BYTES,
            "batch4_min_slots_per_gpu": 2,
            "batch2_required_slots_per_gpu": 4,
        },
        "arms": arm_results,
        "gpu_snapshots": [
            {
                "index": row.index,
                "uuid": row.uuid,
                "total_bytes": row.total_bytes,
                "free_bytes": row.free_bytes,
            }
            for row in sorted(snapshots, key=lambda item: item.index)
        ],
        "slot_claim_bytes_by_batch_size": {
            str(key): value for key, value in sorted(slot_claims.items())
        },
        "ram_slot_budget_bytes_by_batch_size": {
            str(key): value for key, value in sorted(ram_budgets.items())
        },
        "capacity_by_batch_size": {
            str(key): value for key, value in sorted(capacities.items())
        },
        "decision": {
            "reason": decision,
            "selected_batch_size": selected_batch_size,
            "selected_slots_per_gpu": selected_slots_per_gpu,
            "aggregate_batch_per_gpu": (
                None
                if selected_batch_size is None
                else selected_batch_size * selected_slots_per_gpu
            ),
            "aggregate_batch_four_gpus": (
                None
                if selected_batch_size is None
                else selected_batch_size * selected_slots_per_gpu * 4
            ),
            "selected_gpu_slot_claim_bytes": (
                None if selected_batch_size is None else slot_claims[selected_batch_size]
            ),
            "selected_ram_slot_budget_bytes": (
                None if selected_batch_size is None else ram_budgets[selected_batch_size]
            ),
            "all_arms_bit_identical": all_equivalent,
        },
    }
    payload["generation_batch_benchmark_sha256"] = _canonical_digest(
        payload, "generation_batch_benchmark_sha256"
    )
    return payload


def validate_generation_batch_benchmark_contract(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_campaign_id: str,
    expected_continuation_contract_sha256: str,
) -> dict[str, Any]:
    normalized = _mapping(value, "generation batch benchmark")
    if (
        normalized.get("schema_version") != 1
        or normalized.get("contract_type")
        != "safa_r9_generation_batch_benchmark_v1"
        or normalized.get("campaign_id") != expected_campaign_id
        or normalized.get("continuation_contract_sha256")
        != expected_continuation_contract_sha256
    ):
        raise GenerationBatchBenchmarkError("generation batch benchmark identity mismatch")
    declared = _sha(
        normalized.get("generation_batch_benchmark_sha256"),
        "generation batch benchmark SHA256",
    )
    if declared != _canonical_digest(normalized, "generation_batch_benchmark_sha256"):
        raise GenerationBatchBenchmarkError("generation batch benchmark digest mismatch")
    manifest = _mapping(normalized.get("manifest"), "benchmark manifest")
    path = _repo_file(Path(repo_root).resolve(), manifest.get("path"), "manifest")
    if _file_sha256(path) != _sha(manifest.get("sha256"), "manifest SHA256"):
        raise GenerationBatchBenchmarkError("generation benchmark manifest changed")
    decision = _mapping(normalized.get("decision"), "benchmark decision")
    if normalized.get("status") == "ready" and decision.get(
        "selected_batch_size"
    ) not in BENCHMARK_BATCH_SIZES:
        raise GenerationBatchBenchmarkError("benchmark selected an invalid batch size")
    if normalized.get("status") not in {"ready", "blocked"}:
        raise GenerationBatchBenchmarkError("benchmark status is invalid")
    return normalized


def materialize_generation_batch_benchmark_contract(
    path: Path, payload: Mapping[str, Any]
) -> None:
    _write_exclusive(Path(path), _contract_bytes(payload))


def _load_run(
    root: Path, evidence: BatchRunEvidence, *, sample_count: int, seed: int
) -> dict[str, Any]:
    output = Path(evidence.output_dir)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if root not in output.parents:
        raise GenerationBatchBenchmarkError("benchmark output escapes repository")
    result_path = output / "generation_result.json"
    rows_path = output / "per_sample.jsonl"
    result = _mapping(json.loads(result_path.read_text(encoding="utf-8")), "result")
    config = _mapping(result.get("config"), "result config")
    if (
        result.get("status") != "complete"
        or result.get("sample_count") != sample_count
        or config.get("batch_size") != evidence.batch_size
        or config.get("seed") != seed
        or config.get("record_final_latent_sha256") is not True
    ):
        raise GenerationBatchBenchmarkError("benchmark result contract mismatch")
    memory = _mapping(result.get("max_memory"), "result max memory")
    peak_reserved = _positive_int(memory.get("reserved_bytes"), "peak reserved VRAM")
    rows: list[dict[str, Any]] = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(_mapping(json.loads(line), "per-sample row"))
    if len(rows) != sample_count or len({row.get("sample_id") for row in rows}) != sample_count:
        raise GenerationBatchBenchmarkError("benchmark per-sample coverage mismatch")
    for row in rows:
        for field in ("candidate_latent_sha256", "native_latent_sha256"):
            _sha(row.get(field), field)
    return {
        "output_dir": output,
        "generation_result_path": result_path,
        "per_sample_path": rows_path,
        "generation_result": result,
        "rows": rows,
        "peak_reserved_bytes": peak_reserved,
    }


def _compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_rows = {str(row["sample_id"]): row for row in left["rows"]}
    right_rows = {str(row["sample_id"]): row for row in right["rows"]}
    if set(left_rows) != set(right_rows):
        raise GenerationBatchBenchmarkError("batch runs used different sample IDs")
    png_match = True
    latent_match = True
    semantic_match = True
    sample_rows: list[dict[str, Any]] = []
    for sample_id in sorted(left_rows):
        left_row, right_row = left_rows[sample_id], right_rows[sample_id]
        left_png = _png_hashes(left["output_dir"], left_row)
        right_png = _png_hashes(right["output_dir"], right_row)
        row_png_match = left_png == right_png
        row_latent_match = all(
            left_row[field] == right_row[field]
            for field in ("candidate_latent_sha256", "native_latent_sha256")
        )
        row_semantic_match = _semantic_row(left_row) == _semantic_row(right_row)
        png_match = png_match and row_png_match
        latent_match = latent_match and row_latent_match
        semantic_match = semantic_match and row_semantic_match
        sample_rows.append(
            {
                "sample_id": sample_id,
                "png_sha256_match": row_png_match,
                "final_latent_sha256_match": row_latent_match,
                "semantic_row_match": row_semantic_match,
                "batch2_png": left_png,
                "batch4_png": right_png,
            }
        )
    return {
        "sample_count": len(sample_rows),
        "png_sha256_exact": png_match,
        "final_latent_sha256_exact": latent_match,
        "semantic_rows_exact": semantic_match,
        "bit_identical": png_match and latent_match and semantic_match,
        "samples_sha256": hashlib.sha256(_contract_bytes(sample_rows)).hexdigest(),
        "mismatched_sample_ids": [
            row["sample_id"]
            for row in sample_rows
            if not (
                row["png_sha256_match"]
                and row["final_latent_sha256_match"]
                and row["semantic_row_match"]
            )
        ],
    }


def _semantic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in _TIMING_FIELDS | _PATH_FIELDS | {"ordinal", "shard"}
    }


def _png_hashes(output: Path, row: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for role in ("generated", "native"):
        path = Path(str(row[role]))
        if not path.is_absolute():
            candidate = (output / path).resolve()
            path = candidate if candidate.is_file() else path.resolve()
        if not path.is_file() or path.is_symlink():
            raise GenerationBatchBenchmarkError(f"benchmark {role} PNG is missing")
        hashes[role] = _file_sha256(path)
    return hashes


def _run_contract_row(evidence: BatchRunEvidence, loaded: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": evidence.batch_size,
        "gpu_uuid": evidence.gpu_uuid,
        "free_vram_before_bytes": evidence.free_vram_before_bytes,
        "peak_reserved_vram_bytes": loaded["peak_reserved_bytes"],
        "peak_process_tree_rss_bytes": evidence.peak_process_tree_rss_bytes,
        "slot_capacity": _slot_capacity(
            free_bytes=evidence.free_vram_before_bytes,
            peak_reserved_bytes=loaded["peak_reserved_bytes"],
        ),
        "generation_result_sha256": _file_sha256(loaded["generation_result_path"]),
        "per_sample_sha256": _file_sha256(loaded["per_sample_path"]),
    }


def _slot_capacity(*, free_bytes: int, peak_reserved_bytes: int) -> int:
    available = free_bytes - HEADROOM_BYTES
    if available <= 0:
        return 0
    return min(MAX_SLOTS_PER_GPU, available // peak_reserved_bytes)


def _repo_file(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        try:
            path = path.resolve()
            path.relative_to(root)
        except ValueError as error:
            raise GenerationBatchBenchmarkError(f"{label} escapes repository") from error
    else:
        path = (root / path).resolve()
    if not path.is_file() or path.is_symlink():
        raise GenerationBatchBenchmarkError(f"{label} is not a regular file")
    return path


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationBatchBenchmarkError(f"{label} must be a mapping")
    return dict(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GenerationBatchBenchmarkError(f"{label} must be a positive integer")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise GenerationBatchBenchmarkError(f"{label} must be lowercase SHA256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(_contract_bytes(payload)).hexdigest()


def _contract_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise GenerationBatchBenchmarkError("generation benchmark already differs")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
        os.link(temporary, path)
    except FileExistsError as error:
        raise GenerationBatchBenchmarkError("generation benchmark creation raced") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
