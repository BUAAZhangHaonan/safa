from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from safa.evaluation.r9_generation_batch_benchmark import (
    BatchRunEvidence,
    BenchmarkGpuSnapshot,
    GenerationBatchBenchmarkError,
    build_generation_batch_benchmark_contract,
    materialize_generation_batch_benchmark_contract,
    validate_generation_batch_benchmark_contract,
)


SHA = "a" * 64
ARMS = ("native", "paper_eta_0p125", "flow_map2_normalized_eta_0p125")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(
    root: Path,
    *,
    arm_id: str,
    batch_size: int,
    peak_reserved: int,
    semantic_delta: float = 0.0,
    latent_suffix: str = "",
    png_suffix: bytes = b"",
) -> BatchRunEvidence:
    output = root / "runs" / arm_id / f"batch_{batch_size}"
    output.mkdir(parents=True)
    rows = []
    for ordinal, sample_id in enumerate(("sample-a", "sample-b")):
        generated = output / f"candidate-{ordinal}.png"
        native = output / f"native-{ordinal}.png"
        generated.write_bytes(f"{arm_id}-candidate-{ordinal}".encode() + png_suffix)
        native.write_bytes(f"{arm_id}-native-{ordinal}".encode() + png_suffix)
        rows.append(
            {
                "sample_id": sample_id,
                "ordinal": ordinal,
                "shard": 0,
                "generated": str(generated),
                "native": str(native),
                "source": str(root / "source.png"),
                "candidate_cosine": 0.5 + semantic_delta,
                "native_cosine": 0.1,
                "edev_cosine": 0.4,
                "native_edev_cosine": 0.2,
                "candidate_nfe": 7,
                "native_nfe": 1,
                "candidate_trace": [{"kind": "paper"}],
                "native_trace": [{"kind": "native"}],
                "route_diagnostics": {"finite": True},
                "candidate_latent_sha256": hashlib.sha256(
                    f"{arm_id}-candidate-{ordinal}{latent_suffix}".encode()
                ).hexdigest(),
                "native_latent_sha256": hashlib.sha256(
                    f"{arm_id}-native-{ordinal}{latent_suffix}".encode()
                ).hexdigest(),
                "candidate_generation_seconds": float(batch_size),
                "native_generation_seconds": float(batch_size),
                "io_seconds": float(batch_size),
            }
        )
    (output / "per_sample.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "generation_result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sample_count": 2,
                "config": {
                    "batch_size": batch_size,
                    "seed": 4549,
                    "record_final_latent_sha256": True,
                },
                "max_memory": {
                    "allocated_bytes": peak_reserved - 1,
                    "reserved_bytes": peak_reserved,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return BatchRunEvidence(
        arm_id=arm_id,
        batch_size=batch_size,
        output_dir=output.relative_to(root),
        gpu_uuid=f"GPU-{batch_size}",
        free_vram_before_bytes=30 * 1024**3,
        peak_process_tree_rss_bytes=2 * 1024**3,
    )


def _fixture(
    root: Path,
    *,
    free_bytes: int = 30 * 1024**3,
    batch4_peak_reserved: int = 4 * 1024**3,
    mutate_arm: str | None = None,
    mutation: str | None = None,
) -> tuple[dict, Path]:
    (root / "source.png").write_bytes(b"source")
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        '{"sample_id":"sample-a"}\n{"sample_id":"sample-b"}\n',
        encoding="utf-8",
    )
    evidence = []
    for arm_id in ARMS:
        for batch_size in (2, 4):
            kwargs = {}
            if arm_id == mutate_arm and batch_size == 4:
                if mutation == "semantic":
                    kwargs["semantic_delta"] = 0.01
                elif mutation == "latent":
                    kwargs["latent_suffix"] = "changed"
                elif mutation == "png":
                    kwargs["png_suffix"] = b"changed"
            evidence.append(
                _write_run(
                    root,
                    arm_id=arm_id,
                    batch_size=batch_size,
                    peak_reserved=(
                        batch4_peak_reserved
                        if batch_size == 4
                        else 4 * 1024**3
                    ),
                    **kwargs,
                )
            )
    snapshots = [
        BenchmarkGpuSnapshot(
            index=index,
            uuid=f"GPU-{index}",
            total_bytes=32 * 1024**3,
            free_bytes=free_bytes,
        )
        for index in range(4)
    ]
    contract = build_generation_batch_benchmark_contract(
        repo_root=root,
        campaign_id="r9-report-only-formal-v7",
        manifest_path=manifest.relative_to(root),
        manifest_sha256=_sha(manifest),
        sample_count=2,
        seed=4549,
        continuation_contract_sha256=SHA,
        request_sha256="b" * 64,
        required_arm_ids=ARMS,
        gpu_snapshots=snapshots,
        evidence=evidence,
    )
    return contract, manifest


def test_exact_three_arm_evidence_selects_batch4_two_slots(tmp_path: Path) -> None:
    contract, _ = _fixture(tmp_path)
    assert contract["status"] == "ready"
    assert contract["decision"] == {
        "reason": "batch4_exact_and_multi_slot_safe",
        "selected_batch_size": 4,
        "selected_slots_per_gpu": 2,
        "aggregate_batch_per_gpu": 8,
        "aggregate_batch_four_gpus": 32,
        "selected_gpu_slot_claim_bytes": 4 * 1024**3,
        "selected_ram_slot_budget_bytes": 2362232013,
        "all_arms_bit_identical": True,
    }


@pytest.mark.parametrize("mutation", ["png", "latent", "semantic"])
def test_any_exactness_mismatch_selects_batch2(
    tmp_path: Path, mutation: str
) -> None:
    contract, _ = _fixture(
        tmp_path, mutate_arm="paper_eta_0p125", mutation=mutation
    )
    assert contract["status"] == "ready"
    assert contract["decision"]["selected_batch_size"] == 2
    assert contract["decision"]["all_arms_bit_identical"] is False


def test_batch4_capacity_falls_back_to_measured_batch2(tmp_path: Path) -> None:
    contract, _ = _fixture(
        tmp_path,
        free_bytes=18 * 1024**3,
        batch4_peak_reserved=9 * 1024**3,
    )
    assert contract["capacity_by_batch_size"]["4"]["GPU-0"] == 1
    assert contract["decision"]["selected_batch_size"] == 2


def test_both_batch_policies_unsafe_blocks_confirm(tmp_path: Path) -> None:
    contract, _ = _fixture(
        tmp_path,
        free_bytes=17 * 1024**3,
        batch4_peak_reserved=9 * 1024**3,
    )
    assert contract["status"] == "blocked"
    assert contract["decision"]["selected_batch_size"] is None


def test_requires_native_and_two_candidates(tmp_path: Path) -> None:
    contract, manifest = _fixture(tmp_path)
    with pytest.raises(GenerationBatchBenchmarkError, match="exactly two"):
        build_generation_batch_benchmark_contract(
            repo_root=tmp_path,
            campaign_id=contract["campaign_id"],
            manifest_path=manifest.relative_to(tmp_path),
            manifest_sha256=_sha(manifest),
            sample_count=2,
            seed=4549,
            continuation_contract_sha256=SHA,
            request_sha256="b" * 64,
            required_arm_ids=ARMS[:2],
            gpu_snapshots=[
                BenchmarkGpuSnapshot(i, f"GPU-{i}", 32 * 1024**3, 30 * 1024**3)
                for i in range(4)
            ],
            evidence=[],
        )


def test_digest_validation_and_immutable_materialization(tmp_path: Path) -> None:
    contract, _ = _fixture(tmp_path)
    validated = validate_generation_batch_benchmark_contract(
        contract,
        repo_root=tmp_path,
        expected_campaign_id="r9-report-only-formal-v7",
        expected_continuation_contract_sha256=SHA,
    )
    assert validated == contract
    destination = tmp_path / "contract.json"
    materialize_generation_batch_benchmark_contract(destination, contract)
    materialize_generation_batch_benchmark_contract(destination, contract)
    tampered = dict(contract)
    tampered["status"] = "blocked"
    with pytest.raises(GenerationBatchBenchmarkError, match="already differs"):
        materialize_generation_batch_benchmark_contract(destination, tampered)
    tampered = dict(contract)
    tampered["seed"] = 1
    with pytest.raises(GenerationBatchBenchmarkError, match="digest mismatch"):
        validate_generation_batch_benchmark_contract(
            tampered,
            repo_root=tmp_path,
            expected_campaign_id="r9-report-only-formal-v7",
            expected_continuation_contract_sha256=SHA,
        )
