from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import types

import pytest

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    CONTROLLER_LAUNCH_REHASH_CONTRACT,
    WORKER_EXTERNAL_GPU_RACE_CONTRACT,
    WORKER_PRE_CUDA_VERIFICATION_ORDER,
    WORKER_READY_CONTRACT,
    WORKER_RELEASE_CONTRACT,
    _require_no_repo_path_component_symlinks,
    _require_tree_without_symlinks,
    _validate_6b_failed_probe_root_identity,
    _validate_ram_probe_artifact_seal,
    _validate_ram_slot_budget_source,
    build_candidate_manifest,
    build_checkpoint_plan,
    build_preflight_result,
    build_run_claim,
    build_run_request,
    build_run_result,
    canonical_digest,
    canonical_gpu_registry,
    canonicalize_nvidia_gpu_uuid,
    canonical_json,
    hash_asset_directory_content,
    load_json,
    publish_exclusive_json,
    ram_probe_admission_evidence_digest,
    ram_probe_contract_digest,
    ram_probe_execution_digest,
    sha256_file,
    validate_arcface_execution_probe_binding,
    validate_candidate_manifest,
    validate_checkpoint_plan,
    validate_preflight_result,
    validate_run_request,
    validate_run_result,
    validate_controller_launch_rehash_value,
    validate_worker_ready_value,
    validate_worker_release_value,
    validate_worker_terminal_value,
    validate_supersession_evidence,
    validate_policy,
    write_exclusive_json,
)
from safa.closeout.canonical_screening_worker import (
    _assert_ready_barrier,
    _assert_runtime_cuda_binding,
    _load_arcface_contract,
    _load_source_pixel_batch,
    _representation_cosines,
    _wait_worker_release,
    _write_validated_run_result,
    execute_screening_request,
)
import safa.closeout.canonical_screening_worker as screening_worker_module
from safa.closeout.canonical_quality import evaluate_locked_kid
from safa.closeout.generator_output_contract import (
    bind_output_contract,
    decoder_registry_digest,
    resolve_checkpoint_output_capability,
)


def _gpu_uuid(index: int) -> str:
    return f"GPU-0000000{index}-0000-0000-0000-00000000000{index}"


def _raw_controller_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_checkpoint_screening.py"
    spec = importlib.util.spec_from_file_location("canonical_controller_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controller_module():
    module = _raw_controller_module()
    module._install_verified_contract_api(
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json",
        verify_historical_output_evidence=False,
    )
    return module


def _wrapper_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_preflight_wrapper.py"
    spec = importlib.util.spec_from_file_location("canonical_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_wrapper_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_gpu_wrapper.py"
    spec = importlib.util.spec_from_file_location("canonical_gpu_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ram_probe_module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_canonical_screening_ram_probe.py"
    )
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "canonical_ram_probe_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(row) for row in rows))


def _bound(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"x")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _decoder_registry(tmp_path: Path) -> dict:
    bound = _bound(tmp_path / "decoder-bound.bin")
    registry = {
        "schema_version": 1,
        "contract_type": "safa_canonical_output_decoder_registry_v1",
        "pixel": {
            "decoder_type": "native_rgb_unit_interval",
            "output_range": [0.0, 1.0],
            "channels": 3,
            "height": 224,
            "width": 224,
            "model_type": "conditional_flow_matching",
            "sampler": "heun",
            "sample_steps": 32,
            "model_space": "rgb_neg1_pos1",
            "sample_api": "clamp_output=true",
            "clamp_output": True,
            "postprocess": (
                "in_generator_clamp_minus1_1_then_affine_then_"
                "clamp_unit_interval"
            ),
            "decoder_forbidden": True,
            "sampling_implementation": dict(bound),
        },
        "latent": {
            "decoder_type": "r9_frozen_sd_vae_ft_ema",
            "vae_source_path": "artifacts/checkpoints/external/sd-vae-ft-ema",
            "directory": {"path": str(tmp_path), "digest": "a" * 64},
            "config": dict(bound),
            "weights": dict(bound),
            "scaling_factor": 0.18215,
            "implementation": dict(bound),
            "trusted_runtime_config": dict(bound),
            "trusted_runner": dict(bound),
            "trusted_reference_checkpoint": dict(bound),
            "trusted_resolved_config": dict(bound),
            "trusted_generation_result": dict(bound),
            "environment": {
                "provenance_snapshot": dict(bound),
                "packages_sha256": (
                    "35196c0c7f5a8a2db3dcb31a67c0102"
                    "fbd713db6d67af72eacfffe8f8b82be7b"
                ),
                "python_version": "3.12.13",
                "torch_version": "2.11.0+cu128",
                "diffusers_version": "0.38.0",
            },
            "directory_digest_algorithm": (
                "sha256_relative_posix_nul_content_nul_v1"
            ),
            "asset_digest_cache": {"path": str(tmp_path / "cache.json")},
            "asset_digest_cache_algorithm": dict(bound),
            "latent_shape": ["B", 4, 32, 32],
            "decoded_rgb_shape": ["B", 3, 256, 256],
            "output_range": [0.0, 1.0],
        },
        "decoder_registry_sha256": "",
    }
    registry["decoder_registry_sha256"] = decoder_registry_digest(registry)
    return registry


def test_asset_content_hash_rejects_same_size_restored_time_and_forged_cache(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "vae"
    asset.mkdir()
    weights = asset / "weights.bin"
    weights.write_bytes(b"AAAA")
    expected = hashlib.sha256(
        b"weights.bin\0" + b"AAAA" + b"\0"
    ).hexdigest()
    verification = hash_asset_directory_content(asset, expected)
    original_mtime_ns = weights.stat().st_mtime_ns
    weights.write_bytes(b"BBBB")
    os.utime(weights, ns=(original_mtime_ns, original_mtime_ns))
    current = weights.stat()
    forged_cache = {
        "forged": {
            "path": str(asset.resolve()),
            "expected_digest": expected,
            "stat_fingerprint": [
                {
                    "relative": "weights.bin",
                    "device": int(current.st_dev),
                    "inode": int(current.st_ino),
                    "size": int(current.st_size),
                    "mtime_ns": int(current.st_mtime_ns),
                    "ctime_ns": int(current.st_ctime_ns),
                }
            ],
            "digest": expected,
        }
    }
    (tmp_path / "forged-cache.json").write_text(
        json.dumps(forged_cache), encoding="utf-8"
    )
    assert verification["total_bytes"] == 4
    with pytest.raises(
        CanonicalScreeningError,
        match="asset directory content digest differs",
    ):
        hash_asset_directory_content(asset, expected)


def _pixel_output_contract(checkpoint_sha256: str, registry: dict) -> dict:
    capability = resolve_checkpoint_output_capability(
        {
            "model_config": {
                "model_type": "conditional_flow_matching",
                "embedding_dim": 512,
                "image_size": 224,
                "base_channels": 32,
                "channel_multipliers": [1, 2, 4, 4],
                "condition_dim": 512,
                "sample_steps": 32,
                "train_cycle_steps": 8,
                "sampler": "heun",
            },
            "training_config": {},
        },
        checkpoint_sha256,
    )
    return bind_output_contract(capability, registry)


def _asset_content_verification(policy: dict) -> dict:
    directory = policy["output_decoder_registry"]["latent"]["directory"]
    return {
        "schema_version": 1,
        "contract_type": "safa_canonical_asset_content_verification_v1",
        "path": directory["path"],
        "digest_algorithm": "sha256_relative_posix_nul_content_nul_v1",
        "expected_digest": directory["digest"],
        "observed_digest": directory["digest"],
        "file_count": 2,
        "total_bytes": 2,
        "elapsed_seconds": 0.01,
        "started_at": "2026-07-27T00:00:00+00:00",
        "completed_at": "2026-07-27T00:00:00+00:00",
    }


def _policy(tmp_path: Path, ledger: Path) -> tuple[dict, Path, dict]:
    bound = _bound(tmp_path / "bound.bin")
    smoke_manifest = tmp_path / "smoke8.jsonl"
    screen_manifest = tmp_path / "screen512.jsonl"
    _write_jsonl(
        smoke_manifest,
        [{"sample_id": f"s{index}"} for index in range(8)],
    )
    _write_jsonl(
        screen_manifest,
        [{"sample_id": f"s{index}"} for index in range(512)],
    )
    implementations = {
        name: dict(bound)
        for name in (
            "checkpoint_preflight",
            "arcface_evaluator",
            "e0_loader",
            "canonical_quality",
            "screening_contracts",
            "screening_worker",
            "controller",
            "ram_probe_launcher",
            "preflight_wrapper",
            "generator_sampling",
            "meanflow_sampling",
            "latent_codec",
            "output_contract",
        )
    }
    policy = {
        "campaign_id": "historical-canonical-512-v1",
        "supersedes_policy_sha256": "f7d9b8e263bdd54af7754889c7e7ce92d3ec7212d3784ac11c819fc3c07381cd",
        "python": "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python",
        "policy_sha256": "1" * 64,
        "source": {
            "ledger": {
                "path": str(ledger.resolve()),
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            }
        },
        "protocol": {
            "seed": 4549,
            "batch_size": 2,
            "manifests": {
                "smoke8": {**_bound(smoke_manifest), "sample_count": 8},
                "screen512": {
                    **_bound(screen_manifest),
                    "sample_count": 512,
                },
            },
            "source_index": bound,
            "features": {"directory": str(tmp_path), "manifest": bound, "shard": bound},
            "e0": bound,
            "edev": bound,
            "quality_script": bound,
            "pixel_image_size": 256,
            "pixel_protocol_config": bound,
            "kid_subset_sizes": {"smoke8": 8, "screen512": 50},
            "metrics": [],
        },
        "resources": {
            "physical_gpus": [0, 1, 2, 3],
            "workers_per_gpu": 2,
            "ram_budget_status": "sealed",
            "ram_slot_budget_bytes": 1100,
            "ram_slot_budget_source": {
                "contract_type": "safa_canonical_screening_ram_budget_source_v1",
                "method": (
                    "ceil(single_worker_process_tree_peak_rss_bytes*11/10)"
                ),
                "measurement_factor_numerator": 11,
                "measurement_factor_denominator": 10,
                "peak_process_tree_rss_bytes": 1000,
                "ram_slot_budget_bytes": 1100,
                "probe_result": bound,
            },
            "gpu_headroom_bytes": 2 * 1024**3,
            "cpu_admission_percent": 90,
            "cpu_hard_limit_percent": 90,
            "cpu_window_seconds": 60,
            "cpu_consecutive_hard_windows": 2,
            "resource_poll_seconds": 10,
            "swap_consecutive_hard_intervals": 3,
            "ram_admission_percent": 85,
            "ram_hard_limit_percent": 90,
            "disk_admission_percent": 85,
            "disk_hard_limit_percent": 90,
            "retry_count": 0,
            "require_tmux": True,
            "global_lock_root": str(tmp_path / "locks"),
        },
        "implementations": implementations,
        "arcface": {"model_name": "buffalo_l"},
        "output_decoder_registry": _decoder_registry(tmp_path),
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"policy":"fixture"}\n', encoding="utf-8")
    admission_value = {
        "contract_type": "safa_canonical_resource_admission_v1",
        "policy_sha256": policy["policy_sha256"],
        "snapshot": _admission_snapshot(policy),
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "admission.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        "path": str(admission_path.resolve()),
        "sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    return policy, policy_path, admission


def _admission_snapshot(policy: dict) -> dict:
    slot_count = (
        len(policy["resources"]["physical_gpus"])
        * int(policy["resources"]["workers_per_gpu"])
    )
    slot_budget_bytes = int(
        policy["resources"]["ram_slot_budget_bytes"]
    )
    reserved_bytes = slot_count * slot_budget_bytes
    memory_total_bytes = max(100000, reserved_bytes * 10)
    memory_used_bytes = memory_total_bytes // 10
    projected_used_bytes = memory_used_bytes + reserved_bytes
    return {
        "gpus": [],
        "authorized_gpu_registry": [
            {
                "physical_gpu_index": index,
                "physical_gpu_uuid": _gpu_uuid(index),
            }
            for index in range(4)
        ],
        "ram_reservation": {
            "slot_count": slot_count,
            "slot_budget_bytes": slot_budget_bytes,
            "reserved_bytes": reserved_bytes,
            "memory_total_bytes": memory_total_bytes,
            "memory_used_bytes": memory_used_bytes,
            "projected_used_bytes": projected_used_bytes,
            "projected_used_percent": (
                100.0 * projected_used_bytes / memory_total_bytes
            ),
            "admission_limit_percent": policy["resources"][
                "ram_admission_percent"
            ],
            "budget_source": policy["resources"]["ram_slot_budget_source"],
        },
    }


def _row(run_id: str, sha: str | None, selector: str = "raw", path: str | None = None) -> dict:
    return {
        "run_id": run_id,
        "status": "config_only_never_started" if sha is None else "started_incomplete",
        "logical_experiment_id": "R6",
        "protocol_family": "family",
        "comparability_group": "group",
        "evidence_level": "strong_provenance_historical_baseline",
        "checkpoint": {
            "files": [] if sha is None else [{
                "path": path or f"artifacts/{run_id}.pt",
                "sha256": sha,
                "size_bytes": 10,
            }],
            "selector": selector,
        },
    }


def _strict_preflight(
    sha: str,
    selector: str,
    registry: dict,
    status: str = "valid",
) -> dict:
    valid = status == "valid"
    return {
        "schema_version": 1,
        "contract_type": "safa_generator_checkpoint_preflight_v1",
        "status": status,
        "checkpoint_path": "/checkpoint",
        "checkpoint_sha256": sha,
        "expected_checkpoint_sha256": sha,
        "sha256_binding": "expected_exact",
        "checkpoint_model": selector,
        "declared_checkpoint_model": None,
        "available_state_dict_fields": ["model_state_dict"],
        "selector_binding": "single_available_state_dict",
        "state_dict_field": "model_state_dict",
        "tensor_count": 2,
        "finite_tensor_count": 2,
        "nonfinite_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "reconstruction_messages": [],
        "adapter": {
            "type": "none",
            "objective_type": None,
            "configuration_source": None,
            "state_key_count": 0,
            "mounted_key_count": 0,
            "mounted": False,
        },
        "output_capability": (
            _pixel_output_contract(sha, registry)["capability"]
            if valid
            else None
        ),
        "output_contract": (
            _pixel_output_contract(sha, registry)
            if valid
            else None
        ),
        "smoke": {"requested_sample_count": 0, "executed_sample_count": 0, "output_shape": None},
        "failure_code": None if valid else "strict_load_failed",
        "failure_message": None if valid else "cannot reconstruct",
    }


def _complete_plan(tmp_path: Path, rows: list[dict]) -> tuple[dict, dict, Path]:
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, rows)
    policy, _, _ = _policy(tmp_path, ledger)
    result_root = tmp_path / "results"
    pending = build_checkpoint_plan(tmp_path, policy, result_root)
    for request in pending["preflight_requests"]:
        strict = _strict_preflight(
            request["checkpoint_sha256"],
            request["checkpoint_model"],
            policy["output_decoder_registry"],
        )
        envelope = build_preflight_result(request, policy, strict)
        write_exclusive_json(
            result_root
            / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json",
            envelope,
        )
    return build_checkpoint_plan(tmp_path, policy, result_root), policy, result_root


def _manifest_fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path, dict, dict]:
    checkpoint = tmp_path / "artifacts" / "candidate.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"canonical-screening-fixture-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    plan, policy, result_root = _complete_plan(
        tmp_path,
        [
            _row(
                "candidate",
                checkpoint_sha256,
                path=str(checkpoint.resolve()),
            )
        ],
    )
    plan_path = tmp_path / "plan.json"
    write_exclusive_json(plan_path, plan)
    manifest = build_candidate_manifest(
        policy,
        plan,
        plan_path=plan_path,
        repo_root=tmp_path,
        preflight_root=result_root,
    )
    manifest_path = tmp_path / "manifest.json"
    write_exclusive_json(manifest_path, manifest)
    policy_path = tmp_path / "policy.json"
    admission_value = {
        "contract_type": "safa_canonical_resource_admission_v1",
        "policy_sha256": policy["policy_sha256"],
        "snapshot": _admission_snapshot(policy),
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "admission2.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        "path": str(admission_path.resolve()),
        "sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    return policy, manifest, manifest_path, policy_path, admission, plan


def test_plan_counts_real_reference_semantics_and_dedup(tmp_path: Path) -> None:
    sha = "4" * 64
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            _row("raw_a", sha, path="artifacts/a.pt"),
            _row("raw_b", sha, path="artifacts/b.pt"),
            _row("config", None),
        ],
    )
    policy, _, _ = _policy(tmp_path, ledger)
    plan = build_checkpoint_plan(tmp_path, policy, tmp_path / "results")
    counts = plan["counts"]
    assert counts["checkpoint_references"] == 2
    assert counts["raw_checkpoint_references"] == 2
    assert counts["ema_checkpoint_references"] == 0
    assert counts["distinct_checkpoint_sha256"] == 1
    assert counts["distinct_raw_checkpoint_sha256"] == 1
    assert counts["distinct_ema_checkpoint_sha256"] == 0
    assert counts["duplicate_checkpoint_references"] == 1
    assert counts["selector_conflicts"] == 0


def test_old_unbound_preflight_result_is_rejected(tmp_path: Path) -> None:
    sha = "5" * 64
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", sha)])
    policy, _, _ = _policy(tmp_path, ledger)
    results = tmp_path / "results"
    write_exclusive_json(
        results / f"{sha}__raw.json",
        _strict_preflight(sha, "raw", policy["output_decoder_registry"]),
    )
    with pytest.raises(CanonicalScreeningError, match="fields differ"):
        build_checkpoint_plan(tmp_path, policy, results)


def test_preflight_result_binds_request_policy_ledger_and_tool(tmp_path: Path) -> None:
    sha = "6" * 64
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", sha)])
    policy, _, _ = _policy(tmp_path, ledger)
    pending = build_checkpoint_plan(tmp_path, policy, tmp_path / "results")
    request = pending["preflight_requests"][0]
    envelope = build_preflight_result(
        request,
        policy,
        _strict_preflight(
            sha,
            "raw",
            policy["output_decoder_registry"],
        ),
    )
    assert validate_preflight_result(envelope, request, policy)[0] is True
    tampered = json.loads(json.dumps(envelope))
    tampered["policy_sha256"] = "7" * 64
    tampered["preflight_result_sha256"] = canonical_digest(
        tampered, "preflight_result_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="binding mismatch"):
        validate_preflight_result(tampered, request, policy)


@pytest.mark.parametrize("mutation", ("digest", "drop", "count", "policy"))
def test_plan_validator_rederives_and_rejects_tamper(tmp_path: Path, mutation: str) -> None:
    plan, policy, result_root = _complete_plan(
        tmp_path, [_row("candidate", "9" * 64)]
    )
    changed = json.loads(json.dumps(plan))
    if mutation == "digest":
        changed["checkpoint_plan_sha256"] = "a" * 64
    elif mutation == "drop":
        changed["eligible"] = []
        changed["checkpoint_plan_sha256"] = canonical_digest(
            changed, "checkpoint_plan_sha256"
        )
    elif mutation == "count":
        changed["counts"]["eligible_candidates"] = 2
        changed["checkpoint_plan_sha256"] = canonical_digest(
            changed, "checkpoint_plan_sha256"
        )
    else:
        changed["policy_sha256"] = "b" * 64
        changed["checkpoint_plan_sha256"] = canonical_digest(
            changed, "checkpoint_plan_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_checkpoint_plan(
            changed,
            repo_root=tmp_path,
            policy=policy,
            preflight_root=result_root,
        )


def test_candidate_manifest_exactly_binds_validated_plan(tmp_path: Path) -> None:
    policy, manifest, manifest_path, _, _, plan = _manifest_fixture(tmp_path)
    result_root = tmp_path / "results"
    actual_plan_path = Path(manifest["checkpoint_plan"]["path"])
    assert validate_candidate_manifest(
        manifest,
        policy=policy,
        plan=plan,
        plan_path=actual_plan_path,
        repo_root=tmp_path,
        preflight_root=result_root,
    ) == manifest
    changed = json.loads(json.dumps(manifest))
    changed["candidate_count"] = 0
    changed["candidate_manifest_sha256"] = canonical_digest(
        changed, "candidate_manifest_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="differs"):
        validate_candidate_manifest(
            changed,
            policy=policy,
            plan=plan,
            plan_path=actual_plan_path,
            repo_root=tmp_path,
            preflight_root=result_root,
        )
    assert manifest_path.is_file()


def _run_fixture(tmp_path: Path, mode: str = "smoke8", replicate: str = "primary"):
    policy, manifest, manifest_path, policy_path, admission, _ = _manifest_fixture(tmp_path)
    candidate = manifest["candidates"][0]
    controller_ready, observer_ready = _ready_bindings(
        tmp_path, policy, admission, mode
    )
    request = build_run_request(
        policy,
        policy_path,
        manifest,
        manifest_path,
        candidate,
        mode,
        replicate,
        tmp_path / "runs",
        admission,
        controller_ready,
        observer_ready,
    )
    return policy, request


def _real_policy_run_fixture(
    tmp_path: Path,
    module,
) -> tuple[dict, Path, dict, Path]:
    repo_root = Path(__file__).parents[1].resolve()
    config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    policy = validate_policy(
        repo_root,
        config,
        verify_historical_output_evidence=False,
    )
    preflight_root = tmp_path / "real-policy-preflight"
    pending = build_checkpoint_plan(repo_root, policy, preflight_root)
    available = []
    for request in pending["preflight_requests"]:
        raw_path = Path(str(request["checkpoint_path"]))
        checkpoint_path = (
            raw_path if raw_path.is_absolute() else repo_root / raw_path
        ).resolve()
        if checkpoint_path.is_file():
            available.append(
                (checkpoint_path.stat().st_size, checkpoint_path, request)
            )
    for _, checkpoint_path, preflight_request in sorted(
        available, key=lambda item: (item[0], str(item[1]))
    ):
        if sha256_file(checkpoint_path) == preflight_request[
            "checkpoint_sha256"
        ]:
            break
    else:
        raise AssertionError(
            "real policy has no SHA-exact checkpoint for CPU integration"
        )
    selected_sha256 = preflight_request["checkpoint_sha256"]
    for item in pending["preflight_requests"]:
        selected = item["checkpoint_sha256"] == selected_sha256
        strict = _strict_preflight(
            item["checkpoint_sha256"],
            item["checkpoint_model"],
            policy["output_decoder_registry"],
            status="valid" if selected else "invalid",
        )
        strict["checkpoint_path"] = (
            str(checkpoint_path)
            if selected
            else str(item["checkpoint_path"])
        )
        preflight_result = build_preflight_result(
            item, policy, strict
        )
        write_exclusive_json(
            preflight_root
            / (
                f"{item['checkpoint_sha256']}__"
                f"{item['checkpoint_model']}.json"
            ),
            preflight_result,
        )
    plan = build_checkpoint_plan(repo_root, policy, preflight_root)
    assert plan["counts"]["eligible_candidates"] == 1
    plan_path = tmp_path / "real-policy-plan.json"
    write_exclusive_json(plan_path, plan)
    manifest = build_candidate_manifest(
        policy,
        plan,
        plan_path=plan_path,
        repo_root=repo_root,
        preflight_root=preflight_root,
    )
    manifest_path = tmp_path / "real-policy-manifest.json"
    write_exclusive_json(manifest_path, manifest)
    admission_value = {
        "contract_type": "safa_canonical_resource_admission_v1",
        "policy_sha256": policy["policy_sha256"],
        "snapshot": _admission_snapshot(policy),
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "real-policy-admission.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        **_bound(admission_path),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    controller_ready, observer_ready = _production_ready_bindings(
        tmp_path, module, policy, manifest, admission
    )
    request = build_run_request(
        policy,
        config,
        manifest,
        manifest_path,
        manifest["candidates"][0],
        "smoke8",
        "primary",
        tmp_path / "real-policy-runs",
        admission,
        controller_ready,
        observer_ready,
    )
    request_path = tmp_path / "real-policy-run-request.json"
    write_exclusive_json(request_path, request)
    return policy, config, request, request_path


def _production_ready_bindings(
    tmp_path: Path,
    module,
    policy: dict,
    manifest: dict,
    admission: dict,
) -> tuple[dict, dict]:
    paths = module._paths(
        tmp_path / "real-ready-campaign", policy["policy_sha256"]
    )
    wrapper, observer_launch = _wrapper_bindings(
        tmp_path, policy, "smoke8"
    )
    claim, claim_path = module._write_gpu_controller_claim(
        policy, paths, "smoke8", wrapper, observer_launch
    )
    intent, intent_path = module._write_request_intent_manifest(
        policy,
        paths,
        "smoke8",
        ("primary", "repeat"),
        manifest,
        admission,
    )

    def artifact(name: str, digest_field: str) -> tuple[dict, Path]:
        value = {
            "kind": name,
            "policy_sha256": policy["policy_sha256"],
        }
        value[digest_field] = canonical_digest(value, digest_field)
        path = tmp_path / "real-ready-artifacts" / f"{name}.json"
        write_exclusive_json(path, value)
        return value, path

    internal, internal_path = artifact(
        "internal", "monitor_sample_sha256"
    )
    first_guard, first_guard_path = artifact(
        "runtime_guard", "resource_window_sha256"
    )
    recheck, recheck_path = artifact(
        "resource_recheck", "resource_recheck_sha256"
    )
    controller, _, controller_binding = module._write_controller_ready(
        policy,
        paths,
        "smoke8",
        claim,
        admission,
        intent,
        intent_path,
        internal,
        internal_path,
        first_guard,
        first_guard_path,
        recheck,
        recheck_path,
        claim_path,
    )
    observer_claim, observer_claim_path = artifact(
        "observer_claim", "observer_claim_sha256"
    )
    observer_sample, observer_sample_path = artifact(
        "observer_sample", "monitor_sample_sha256"
    )
    observer = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": "smoke8",
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_ready_sha256": controller[
            "controller_ready_sha256"
        ],
        "observer_claim_sha256": observer_claim[
            "observer_claim_sha256"
        ],
        "wrapper_claim_sha256": wrapper["canonical_sha256"],
        "observer_launch_sha256": observer_launch[
            "canonical_sha256"
        ],
        "observer_claim": module._artifact_binding(
            observer_claim_path,
            observer_claim["observer_claim_sha256"],
        ),
        "wrapper_claim": wrapper,
        "observer_launch": observer_launch,
        "controller_ready": controller_binding,
        "admission": admission,
        "first_observer_sample": module._artifact_binding(
            observer_sample_path,
            observer_sample["monitor_sample_sha256"],
        ),
    }
    observer["observer_ready_sha256"] = canonical_digest(
        observer, "observer_ready_sha256"
    )
    observer_path = (
        paths["gpu_control"] / "smoke8" / "observer_ready.json"
    )
    write_exclusive_json(observer_path, observer)
    observer_binding = module._artifact_binding(
        observer_path, observer["observer_ready_sha256"]
    )
    module._validate_observer_ready(
        observer, policy, "smoke8", controller, admission
    )
    return controller_binding, observer_binding


def _final_release_for_single_request(
    tmp_path: Path,
    policy: dict,
    request: dict,
    request_path: Path,
) -> dict:
    controller_ready = load_json(
        Path(request["controller_ready"]["path"]),
        "real pre-CUDA controller ready",
    )
    release = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_final_release_admission_v1",
        "campaign_id": policy["campaign_id"],
        "phase": request["mode"],
        "policy_sha256": policy["policy_sha256"],
        "initial_admission_sha256": request["admission"][
            "canonical_sha256"
        ],
        "controller_ready_sha256": request["controller_ready"][
            "canonical_sha256"
        ],
        "observer_ready_sha256": request["observer_ready"][
            "canonical_sha256"
        ],
        "wrapper_claim": controller_ready["wrapper_claim"],
        "wrapper_claim_sha256": controller_ready[
            "wrapper_claim_sha256"
        ],
        "observer_launch": controller_ready["observer_launch"],
        "observer_launch_sha256": controller_ready[
            "observer_launch_sha256"
        ],
        "authorized_gpu_registry": request[
            "authorized_gpu_registry"
        ],
        "request_count": 1,
        "requests": [
            {
                **_bound(request_path),
                "canonical_sha256": request["run_request_sha256"],
            }
        ],
        "snapshot": {
            "authorized_gpu_registry": request[
                "authorized_gpu_registry"
            ],
            "compute_processes": [],
        },
        "released_at": "2026-07-27T00:00:00+00:00",
    }
    release["final_release_admission_sha256"] = canonical_digest(
        release, "final_release_admission_sha256"
    )
    release_path = tmp_path / "real-policy-final-release.json"
    write_exclusive_json(release_path, release)
    return {
        **_bound(release_path),
        "canonical_sha256": release[
            "final_release_admission_sha256"
        ],
    }


def _ready_bindings(
    tmp_path: Path, policy: dict, admission: dict, mode: str
) -> tuple[dict, dict]:
    wrapper_claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_wrapper_claim_v1",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
    }
    wrapper_claim["wrapper_claim_sha256"] = canonical_digest(
        wrapper_claim, "wrapper_claim_sha256"
    )
    wrapper_claim_path = tmp_path / "ready" / mode / "wrapper_claim.json"
    write_exclusive_json(wrapper_claim_path, wrapper_claim)
    wrapper_binding = {
        **_bound(wrapper_claim_path),
        "canonical_sha256": wrapper_claim["wrapper_claim_sha256"],
    }
    observer_launch = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_launch_v2",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
        "status": "launched",
        "failure": None,
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper_claim["wrapper_claim_sha256"],
    }
    observer_launch["observer_launch_sha256"] = canonical_digest(
        observer_launch, "observer_launch_sha256"
    )
    observer_launch_path = (
        tmp_path / "ready" / mode / "observer_launch.json"
    )
    write_exclusive_json(observer_launch_path, observer_launch)
    observer_launch_binding = {
        **_bound(observer_launch_path),
        "canonical_sha256": observer_launch["observer_launch_sha256"],
    }
    controller_claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_claim_v1",
        "campaign_id": policy["campaign_id"],
        "phase": mode,
        "policy_sha256": policy["policy_sha256"],
        "wrapper_claim": wrapper_binding,
        "observer_launch": observer_launch_binding,
        "controller_pid": 77,
        "started_at": "2026-07-27T00:00:00+00:00",
    }
    controller_claim["controller_claim_sha256"] = canonical_digest(
        controller_claim, "controller_claim_sha256"
    )
    controller_claim_path = (
        tmp_path / "ready" / mode / "controller_claim.json"
    )
    write_exclusive_json(controller_claim_path, controller_claim)
    controller_claim_binding = {
        **_bound(controller_claim_path),
        "canonical_sha256": controller_claim[
            "controller_claim_sha256"
        ],
    }
    controller = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": mode,
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_claim": controller_claim_binding,
        "controller_claim_sha256": controller_claim[
            "controller_claim_sha256"
        ],
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper_claim["wrapper_claim_sha256"],
        "observer_launch": observer_launch_binding,
        "observer_launch_sha256": observer_launch[
            "observer_launch_sha256"
        ],
    }
    controller["controller_ready_sha256"] = canonical_digest(
        controller, "controller_ready_sha256"
    )
    controller_path = tmp_path / "ready" / mode / "controller_ready.json"
    write_exclusive_json(controller_path, controller)
    controller_binding = {
        **_bound(controller_path),
        "canonical_sha256": controller["controller_ready_sha256"],
    }
    observer = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": mode,
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_ready_sha256": controller["controller_ready_sha256"],
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper_claim["wrapper_claim_sha256"],
        "observer_launch": observer_launch_binding,
        "observer_launch_sha256": observer_launch[
            "observer_launch_sha256"
        ],
    }
    observer["observer_ready_sha256"] = canonical_digest(
        observer, "observer_ready_sha256"
    )
    observer_path = tmp_path / "ready" / mode / "observer_ready.json"
    write_exclusive_json(observer_path, observer)
    observer_binding = {
        **_bound(observer_path),
        "canonical_sha256": observer["observer_ready_sha256"],
    }
    return controller_binding, observer_binding


def _wrapper_bindings(
    tmp_path: Path, policy: dict, mode: str
) -> tuple[dict, dict]:
    wrapper = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_wrapper_claim_v1",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
    }
    wrapper["wrapper_claim_sha256"] = canonical_digest(
        wrapper, "wrapper_claim_sha256"
    )
    wrapper_path = tmp_path / "wrapper" / mode / "wrapper_claim.json"
    write_exclusive_json(wrapper_path, wrapper)
    wrapper_binding = {
        **_bound(wrapper_path),
        "canonical_sha256": wrapper["wrapper_claim_sha256"],
    }
    launch = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_launch_v2",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
        "status": "launched",
        "failure": None,
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper["wrapper_claim_sha256"],
        "command": ["controller", "--monitor-target", mode, "--execute"],
    }
    launch["observer_launch_sha256"] = canonical_digest(
        launch, "observer_launch_sha256"
    )
    launch_path = tmp_path / "wrapper" / mode / "observer_launch.json"
    write_exclusive_json(launch_path, launch)
    return wrapper_binding, {
        **_bound(launch_path),
        "canonical_sha256": launch["observer_launch_sha256"],
    }


def _mock_controller_claim(path: Path, claim: dict) -> tuple[dict, Path]:
    write_exclusive_json(path, claim)
    return claim, path


def _evidence(policy: dict, request: dict) -> dict:
    return {
        "mode": request["mode"],
        "replicate": request["replicate"],
        "seed": 4549,
        "batch_size": 2,
        "sample_count": request["sample_count"],
        "sample_manifest_sha256": request["sample_manifest"]["sha256"],
        "candidate_manifest_sha256": request["candidate_manifest"]["canonical_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "implementations": policy["implementations"],
        "checkpoint_sha256": request["candidate"]["checkpoint_sha256"],
        "checkpoint_model": request["candidate"]["checkpoint_model"],
        "output_contract_sha256": request["output_contract"][
            "output_contract_sha256"
        ],
        "output_contract_type": request["output_contract"]["contract_type"],
        "decoder_registry_sha256": request["output_decoder_registry"][
            "decoder_registry_sha256"
        ],
        "output_space": request["output_contract"]["capability"]["output_space"],
        "native_rgb_size": request["native_rgb_size"],
        "quality_protocol_family": request["quality_protocol_family"],
        "nfe": request["nfe"],
        "pixel_image_size": 256,
        "pixel_protocol_config_sha256": policy["protocol"]["pixel_protocol_config"]["sha256"],
        "kid_subset_size": policy["protocol"]["kid_subset_sizes"][request["mode"]],
        "e0_mean": 0.8,
        "edev_mean": 0.7,
        "arcface": {"coverage": request["sample_count"]},
        "quality": {"kid_mean": 0.01},
        "per_sample_sha256": "c" * 64,
    }


def _handshake_fixture(
    policy: dict,
    request: dict,
    gpu_index: int = 0,
    worker_pid: int = 123,
) -> dict:
    gpu_uuid = request["authorized_gpu_registry"][gpu_index][
        "physical_gpu_uuid"
    ]
    request_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}.json"
    )
    write_exclusive_json(request_path, request)
    release = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_final_release_admission_v1",
        "campaign_id": policy["campaign_id"],
        "phase": request["mode"],
        "policy_sha256": policy["policy_sha256"],
        "initial_admission_sha256": request["admission"][
            "canonical_sha256"
        ],
        "controller_ready_sha256": request["controller_ready"][
            "canonical_sha256"
        ],
        "observer_ready_sha256": request["observer_ready"][
            "canonical_sha256"
        ],
        "wrapper_claim": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["wrapper_claim"],
        "wrapper_claim_sha256": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["wrapper_claim_sha256"],
        "observer_launch": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["observer_launch"],
        "observer_launch_sha256": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["observer_launch_sha256"],
        "authorized_gpu_registry": request["authorized_gpu_registry"],
        "request_count": 1,
        "requests": [
            {
                **_bound(request_path),
                "canonical_sha256": request["run_request_sha256"],
            }
        ],
        "snapshot": {
            "authorized_gpu_registry": request["authorized_gpu_registry"],
            "compute_processes": [],
        },
        "released_at": "2026-07-26T00:00:00+00:00",
    }
    release["final_release_admission_sha256"] = canonical_digest(
        release, "final_release_admission_sha256"
    )
    release_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-release.json"
    )
    write_exclusive_json(release_path, release)
    release_binding = {
        **_bound(release_path),
        "canonical_sha256": release["final_release_admission_sha256"],
    }
    manifest = load_json(
        Path(request["candidate_manifest"]["path"]),
        "handshake candidate manifest",
    )
    checkpoint_path = Path(request["candidate"]["checkpoint_path"]).resolve()
    rehashed_bindings = {
        "config": {
            "path": str(Path(request["policy"]["path"]).resolve()),
            "sha256": request["policy"]["sha256"],
        },
        "implementations": {
            name: {
                "path": str(Path(binding["path"]).resolve()),
                "sha256": binding["sha256"],
            }
            for name, binding in request["implementations"].items()
        },
        "request": {
            "path": str(request_path.resolve()),
            "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "canonical_sha256": request["run_request_sha256"],
        },
        "candidate_manifest": dict(request["candidate_manifest"]),
        "checkpoint_plan": dict(manifest["checkpoint_plan"]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "data_and_evaluators": {
            name: request[name]
            for name in (
                "sample_manifest",
                "source_index",
                "features",
                "e0",
                "edev",
                "quality_script",
                "pixel_protocol_config",
                "arcface",
            )
        },
        "final_release": dict(release_binding),
        "controller_ready": dict(request["controller_ready"]),
        "observer_ready": dict(request["observer_ready"]),
    }
    controller_ready = load_json(
        Path(request["controller_ready"]["path"]),
        "handshake controller ready",
    )
    worker_ready = {
        "schema_version": 1,
        "contract_type": WORKER_READY_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "phase": request["mode"],
        "worker_pid": worker_pid,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "run_request_sha256": request["run_request_sha256"],
        "request": rehashed_bindings["request"],
        "final_release": release_binding,
        "verification_order": list(WORKER_PRE_CUDA_VERIFICATION_ORDER),
        "rehashed_bindings": rehashed_bindings,
        "rehashed_bindings_sha256": hashlib.sha256(
            canonical_json(rehashed_bindings)
        ).hexdigest(),
        "controller_claim": controller_ready["controller_claim"],
        "screening_worker_sha256": request["implementations"][
            "screening_worker"
        ]["sha256"],
        "controller_implementation_sha256": request["implementations"][
            "controller"
        ]["sha256"],
        "cuda_visible_devices": gpu_uuid,
        "heavy_modules_absent": True,
        "loaded_heavy_modules": [],
        "asset_content_verification": _asset_content_verification(policy),
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "ready_at": "2026-07-27T00:00:01+00:00",
    }
    worker_ready["worker_ready_sha256"] = canonical_digest(
        worker_ready, "worker_ready_sha256"
    )
    worker_ready_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-worker-ready.json"
    )
    write_exclusive_json(worker_ready_path, worker_ready)
    worker_ready_binding = {
        **_bound(worker_ready_path),
        "canonical_sha256": worker_ready["worker_ready_sha256"],
    }
    gpus = [
        {
            "index": row["physical_gpu_index"],
            "uuid": row["physical_gpu_uuid"],
            "memory_total_mib": 24576,
            "memory_used_mib": 3,
            "memory_free_mib": 24573,
            "temperature_c": 35,
        }
        for row in request["authorized_gpu_registry"]
    ]
    controller_resource_snapshot = {
        "observed_at": "2026-07-27T00:00:02+00:00",
        "cpu_load_percent": 1.0,
        "memory_percent": 2.0,
        "disk_percent": 3.0,
        "swap_pages": {"in": 0, "out": 0},
        "gpus": gpus,
        "authorized_gpu_registry": request["authorized_gpu_registry"],
        "ram_reservation": {"validated": True},
        "compute_processes": [],
    }
    controller_rehash = {
        "schema_version": 1,
        "contract_type": CONTROLLER_LAUNCH_REHASH_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "run_request_sha256": request["run_request_sha256"],
        "worker_pid": worker_pid,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "worker_ready": worker_ready_binding,
        "verification_order": list(WORKER_PRE_CUDA_VERIFICATION_ORDER),
        "rehashed_bindings": rehashed_bindings,
        "rehashed_bindings_sha256": worker_ready[
            "rehashed_bindings_sha256"
        ],
        "resource_snapshot": controller_resource_snapshot,
        "asset_content_verification": _asset_content_verification(policy),
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "validated_at": "2026-07-27T00:00:02+00:00",
    }
    controller_rehash["controller_launch_rehash_sha256"] = canonical_digest(
        controller_rehash, "controller_launch_rehash_sha256"
    )
    controller_rehash_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-controller-rehash.json"
    )
    write_exclusive_json(controller_rehash_path, controller_rehash)
    controller_rehash_binding = {
        **_bound(controller_rehash_path),
        "canonical_sha256": controller_rehash[
            "controller_launch_rehash_sha256"
        ],
    }
    worker_release = {
        "schema_version": 1,
        "contract_type": WORKER_RELEASE_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "phase": request["mode"],
        "worker_pid": worker_pid,
        "run_request_sha256": request["run_request_sha256"],
        "worker_ready": worker_ready_binding,
        "controller_launch_rehash": controller_rehash_binding,
        "resource_snapshot": {
            "admission": controller_resource_snapshot,
            "runtime_guard": {
                "schema_version": 1,
                "contract_type":
                    "safa_canonical_worker_release_resource_snapshot_v2",
                "policy_sha256": policy["policy_sha256"],
                "observed_at": "2026-07-27T00:00:03+00:00",
                "runtime_gpu_registry": request[
                    "authorized_gpu_registry"
                ],
                "compute_processes": [],
                "unknown_compute_processes": [],
                "cpu_load_percent": 1.0,
                "memory_percent": 2.0,
                "disk_percent": 3.0,
                "swap_pages_before": {"in": 0, "out": 0},
                "swap_pages_after": {"in": 0, "out": 0},
                "swap_io_delta": {"in": 0, "out": 0},
                "swap_consecutive_io": 0,
                "gpu": gpus,
                "active_worker_pids": [worker_pid],
                "hard_limits": {
                    "cpu_percent": 90,
                    "ram_percent": 90,
                    "disk_percent": 90,
                    "gpu_memory_percent": 90.0,
                    "gpu_temperature_c": 85,
                    "gpu_free_mib": 2048,
                    "swap_io_delta_pages": 0,
                    "swap_consecutive_io": 0,
                },
                "guard_thread_failure": None,
                "guard_violation_reason": None,
            },
        },
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "released_at": "2026-07-27T00:00:03+00:00",
    }
    worker_release["worker_release_sha256"] = canonical_digest(
        worker_release, "worker_release_sha256"
    )
    worker_release_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-worker-release.json"
    )
    write_exclusive_json(worker_release_path, worker_release)
    worker_release_binding = {
        **_bound(worker_release_path),
        "canonical_sha256": worker_release["worker_release_sha256"],
    }
    validate_worker_ready_value(
        worker_ready,
        request,
        policy,
        expected_worker_pid=worker_pid,
        expected_gpu_index=gpu_index,
        expected_gpu_uuid=gpu_uuid,
    )
    validate_controller_launch_rehash_value(
        controller_rehash, request, policy
    )
    validate_worker_release_value(
        worker_release,
        request,
        policy,
        expected_worker_pid=worker_pid,
    )
    return {
        "final_release": release_binding,
        "worker_ready": worker_ready,
        "worker_ready_binding": worker_ready_binding,
        "controller_rehash": controller_rehash,
        "controller_rehash_binding": controller_rehash_binding,
        "worker_release": worker_release,
        "worker_release_binding": worker_release_binding,
    }


def _run_claim(
    policy: dict, request: dict, gpu_index: int = 0, worker_pid: int = 123
) -> dict:
    gpu_uuid = request["authorized_gpu_registry"][gpu_index][
        "physical_gpu_uuid"
    ]
    handshake = _handshake_fixture(
        policy,
        request,
        gpu_index=gpu_index,
        worker_pid=worker_pid,
    )
    return build_run_claim(
        request,
        policy,
        handshake["final_release"],
        handshake["worker_ready_binding"],
        handshake["worker_release_binding"],
        gpu_index,
        gpu_uuid,
        gpu_uuid,
        gpu_uuid,
        worker_pid,
        "2026-07-26T00:00:00+00:00",
    )


def _completed_worker_terminal_fixture(
    policy: dict, request: dict, worker_pid: int = 123
) -> dict:
    handshake = _handshake_fixture(
        policy, request, worker_pid=worker_pid
    )
    gpu_uuid = request["authorized_gpu_registry"][0][
        "physical_gpu_uuid"
    ]
    claim = build_run_claim(
        request,
        policy,
        handshake["final_release"],
        handshake["worker_ready_binding"],
        handshake["worker_release_binding"],
        0,
        gpu_uuid,
        gpu_uuid,
        gpu_uuid,
        worker_pid,
        "2026-07-26T00:00:00+00:00",
    )
    output_dir = Path(request["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / "claim.json"
    write_exclusive_json(claim_path, claim)
    result = build_run_result(
        request,
        claim,
        policy,
        status="completed",
        completed_at="2026-07-26T00:01:00+00:00",
        evidence=_evidence(policy, request),
    )
    result_path = output_dir / "result.json"
    write_exclusive_json(result_path, result)
    request_path = Path(
        handshake["worker_ready"]["request"]["path"]
    ).resolve()
    terminal = {
        "schema_version": 1,
        "contract_type": "safa_canonical_worker_terminal_v1",
        "policy_sha256": policy["policy_sha256"],
        "worker_pid": worker_pid,
        "request": {
            **_bound(request_path),
            "canonical_sha256": request["run_request_sha256"],
        },
        "claim": {
            **_bound(claim_path),
            "canonical_sha256": claim["run_claim_sha256"],
        },
        "result": {
            **_bound(result_path),
            "canonical_sha256": result["run_result_sha256"],
        },
        "worker_ready": handshake["worker_ready_binding"],
        "worker_release": handshake["worker_release_binding"],
        "status": "completed",
        "failure": None,
        "started_at": "2026-07-26T00:00:00+00:00",
        "completed_at": "2026-07-26T00:01:00+00:00",
    }
    terminal["worker_terminal_sha256"] = canonical_digest(
        terminal, "worker_terminal_sha256"
    )
    terminal_path = (
        Path(handshake["worker_ready_binding"]["path"]).parent
        / "worker_terminal.json"
    )
    write_exclusive_json(terminal_path, terminal)
    return {
        "request_path": request_path,
        "claim_path": claim_path,
        "result_path": result_path,
        "terminal_path": terminal_path,
        "claim": claim,
        "result": result,
        "terminal": terminal,
    }


@pytest.mark.parametrize(
    "target",
    (
        "request",
        "claim",
        "result",
        "terminal",
        "missing_claim",
        "missing_result",
        "missing_terminal",
    ),
)
def test_completion_rejects_post_exit_artifact_tamper(
    tmp_path: Path, target: str
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    artifacts = _completed_worker_terminal_fixture(policy, request)
    validate_worker_terminal_value(
        artifacts["terminal"],
        artifacts["request_path"],
        policy,
        expected_worker_pid=123,
        require_completed=True,
    )
    if target == "request":
        value = load_json(artifacts["request_path"], "request")
        value["replicate"] = "post-exit-tamper"
        value["run_request_sha256"] = canonical_digest(
            value, "run_request_sha256"
        )
        artifacts["request_path"].write_bytes(canonical_json(value))
    elif target == "claim":
        value = load_json(artifacts["claim_path"], "claim")
        value["started_at"] = "2026-07-26T00:00:01+00:00"
        value["run_claim_sha256"] = canonical_digest(
            value, "run_claim_sha256"
        )
        artifacts["claim_path"].write_bytes(canonical_json(value))
    elif target == "result":
        value = load_json(artifacts["result_path"], "result")
        value["completed_at"] = "2026-07-26T00:01:01+00:00"
        value["run_result_sha256"] = canonical_digest(
            value, "run_result_sha256"
        )
        artifacts["result_path"].write_bytes(canonical_json(value))
    elif target == "terminal":
        value = load_json(artifacts["terminal_path"], "terminal")
        value["result"]["sha256"] = "f" * 64
        value["worker_terminal_sha256"] = canonical_digest(
            value, "worker_terminal_sha256"
        )
        artifacts["terminal_path"].write_bytes(canonical_json(value))
    else:
        artifacts[target.removeprefix("missing_") + "_path"].unlink()
    with pytest.raises((CanonicalScreeningError, FileNotFoundError)):
        module._build_gpu_completion_summary(
            policy,
            request["mode"],
            module._paths(tmp_path / "campaign", policy["policy_sha256"]),
            [artifacts["request_path"]],
            {},
            {},
            tmp_path / "monitor.jsonl",
            {},
        )


def test_run_request_rejects_stale_policy_and_wrong_kid_subset(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    assert request["kid_subset_size"] == 8
    assert validate_run_request(request, policy) == request
    changed = json.loads(json.dumps(request))
    changed["kid_subset_size"] = 50
    changed["run_request_sha256"] = canonical_digest(changed, "run_request_sha256")
    with pytest.raises(CanonicalScreeningError, match="frozen"):
        validate_run_request(changed, policy)
    stale = json.loads(json.dumps(request))
    stale["policy"]["canonical_sha256"] = "d" * 64
    stale["run_request_sha256"] = canonical_digest(stale, "run_request_sha256")
    with pytest.raises(CanonicalScreeningError, match="policy binding"):
        validate_run_request(stale, policy)


def test_screen512_locks_kid_subset_50(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path, mode="screen512")
    assert request["kid_subset_size"] == 50
    assert validate_run_request(request, policy) == request


def test_run_result_binds_smoke_manifest_policy_tool_and_admission(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    claim = _run_claim(policy, request, gpu_index=3)
    result = build_run_result(
        request,
        claim,
        policy,
        status="completed",
        completed_at="2026-07-26T00:01:00+00:00",
        evidence=_evidence(policy, request),
    )
    assert validate_run_result(result, request, claim, policy) == result
    changed = json.loads(json.dumps(result))
    changed["evidence"]["candidate_manifest_sha256"] = "e" * 64
    changed["run_result_sha256"] = canonical_digest(changed, "run_result_sha256")
    with pytest.raises(CanonicalScreeningError, match="candidate_manifest"):
        validate_run_result(changed, request, claim, policy)


def test_run_request_and_claim_reject_gpu_uuid_tampering(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    changed_request = json.loads(json.dumps(request))
    changed_request["authorized_gpu_registry"][0]["physical_gpu_uuid"] = (
        "GPU-tampered"
    )
    changed_request["run_request_sha256"] = canonical_digest(
        changed_request, "run_request_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="GPU UUID registry"):
        validate_run_request(changed_request, policy)

    claim = _run_claim(policy, request)
    changed_claim = json.loads(json.dumps(claim))
    changed_claim["runtime_cuda_uuid"] = "GPU-tampered"
    changed_claim["run_claim_sha256"] = canonical_digest(
        changed_claim, "run_claim_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="CUDA/RAM"):
        build_run_result(
            request,
            changed_claim,
            policy,
            status="failed",
            completed_at="2026-07-26T00:01:00+00:00",
            failure={"type": "probe", "message": "probe"},
        )


def test_worker_cuda_binding_refuses_remap_and_runtime_uuid_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    policy, request = _run_fixture(tmp_path)
    expected_uuid = request["authorized_gpu_registry"][0]["physical_gpu_uuid"]
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(CanonicalScreeningError, match="CUDA_VISIBLE_DEVICES"):
        _assert_runtime_cuda_binding(request, 0, expected_uuid)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", expected_uuid)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: types.SimpleNamespace(uuid=_gpu_uuid(1)),
    )
    with pytest.raises(CanonicalScreeningError, match="runtime CUDA UUID"):
        _assert_runtime_cuda_binding(request, 0, expected_uuid)

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: types.SimpleNamespace(uuid=expected_uuid),
    )
    selected: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    binding = _assert_runtime_cuda_binding(request, 0, expected_uuid)
    assert binding["physical_gpu_index"] == 0
    assert binding["logical_cuda_index"] == 0
    assert binding["physical_gpu_uuid"] == expected_uuid
    assert binding["runtime_cuda_uuid"] == expected_uuid
    assert binding["cuda_visible_devices"] == expected_uuid
    assert {
        binding["uuid_evidence"][name]["canonical"]
        for name in (
            "admission",
            "worker_argument",
            "cuda_visible_devices",
            "runtime_cuda_uuid",
        )
    } == {expected_uuid}
    assert selected == [0]


@pytest.mark.parametrize(
    "raw",
    [
        "GPU-7BA69FC7-12AC-3DFB-8265-3476CE2504B6",
        "7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
        b"GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    ],
)
def test_gpu_uuid_canonicalizer_accepts_verified_representations(
    raw: str | bytes,
) -> None:
    assert canonicalize_nvidia_gpu_uuid(raw, "fixture")["canonical"] == (
        "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6"
    )


@pytest.mark.parametrize(
    "raw",
    [
        " GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
        "GPU-7ba69fc7",
        "7ba69fc712ac3dfb82653476ce2504b6",
        "MIG-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
        b"\xff",
        object(),
    ],
)
def test_gpu_uuid_canonicalizer_rejects_malformed_values(raw: object) -> None:
    with pytest.raises(CanonicalScreeningError):
        canonicalize_nvidia_gpu_uuid(raw, "fixture")  # type: ignore[arg-type]


def test_gpu_registry_rejects_duplicate_canonical_uuid() -> None:
    with pytest.raises(CanonicalScreeningError, match="duplicate UUIDs"):
        canonical_gpu_registry(
            [
                {
                    "physical_gpu_index": 0,
                    "physical_gpu_uuid": _gpu_uuid(0),
                },
                {
                    "physical_gpu_index": 1,
                    "physical_gpu_uuid": _gpu_uuid(0).removeprefix("GPU-"),
                },
            ]
        )


def test_failed_probe_root_identity_rejects_root_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "clone"
    target.mkdir()
    declared = (
        tmp_path
        / "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__6b088236579f7311"
    )
    declared.parent.mkdir(parents=True)
    declared.symlink_to(target, target_is_directory=True)
    with pytest.raises(CanonicalScreeningError, match="root identity"):
        _validate_6b_failed_probe_root_identity(
            tmp_path,
            {
                "path": (
                    "artifacts/closeout/historical-canonical-512-v1/"
                    "ram_probe__6b088236579f7311"
                ),
                "digest": "0" * 64,
                "digest_algorithm": (
                    "sha256_relative_posix_nul_content_nul_v1"
                ),
            },
        )


def test_ram_probe_contract_and_execution_digests_are_split() -> None:
    module = _ram_probe_module()
    static = {
        "schema_version": 1,
        "contract_type": module.PROBE_CONTRACT,
        "sample_count": 8,
        "authorized_gpu_registry": None,
        "admission": None,
        "probe_contract_sha256": None,
        "probe_execution_sha256": None,
    }
    contract = module._probe_contract_digest(static)
    registry = [
        {
            "physical_gpu_index": 0,
            "physical_gpu_uuid": _gpu_uuid(0),
        }
    ]
    live = {
        **static,
        "authorized_gpu_registry": registry,
        "admission": {"path": "/bound", "sha256": "0" * 64},
        "probe_contract_sha256": contract,
        "probe_execution_sha256": "1" * 64,
    }
    assert module._probe_contract_digest(live) == contract
    admission = {
        "schema_version": 1,
        "contract_type": module.PROBE_ADMISSION_CONTRACT,
        "probe_contract_sha256": contract,
        "host": {"ram_used_percent": 20.0},
        "gpu_snapshot": [{"index": 0, "uuid": _gpu_uuid(0)}],
        "authorized_gpu_registry": registry,
        "observed_at": "2026-07-27T00:00:00+00:00",
    }
    evidence = module._admission_evidence_digest(admission)
    execution = module._probe_execution_digest(contract, registry, evidence)
    changed_registry = [{**registry[0], "physical_gpu_uuid": _gpu_uuid(1)}]
    assert (
        module._probe_execution_digest(contract, changed_registry, evidence)
        != execution
    )
    tampered = {
        **admission,
        "host": {"ram_used_percent": 21.0},
    }
    assert module._admission_evidence_digest(tampered) != evidence


def test_ram_probe_build_spec_separates_admission_binding_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    config = tmp_path / "policy.json"
    manifest_path = tmp_path / "manifest.json"
    smoke = tmp_path / "smoke.jsonl"
    for path in (config, manifest_path, smoke):
        path.write_text("{}\n", encoding="utf-8")
    policy = {
        "policy_sha256": "1" * 64,
        "protocol": {"manifests": {"smoke8": _bound(smoke)}},
        "implementations": {"worker": _bound(smoke)},
    }
    manifest = {"candidate_manifest_sha256": "2" * 64}
    monkeypatch.setattr(module, "_select_probe_candidates", lambda _value: [])
    dry = module._build_spec(
        policy,
        config,
        manifest,
        manifest_path,
        tmp_path / "probe",
        None,
    )
    binding = {
        "path": str((tmp_path / "admission.json").resolve()),
        "sha256": "3" * 64,
        "canonical_sha256": "4" * 64,
    }
    execution = "5" * 64
    live = module._build_spec(
        policy,
        config,
        manifest,
        manifest_path,
        tmp_path / "probe",
        [
            {
                "physical_gpu_index": 0,
                "physical_gpu_uuid": _gpu_uuid(0),
            }
        ],
        binding,
        execution,
    )
    assert live["admission"] == binding
    assert live["probe_execution_sha256"] == execution
    assert live["probe_contract_sha256"] == dry["probe_contract_sha256"]
    with pytest.raises(CanonicalScreeningError, match="must be paired"):
        module._build_spec(
            policy,
            config,
            manifest,
            manifest_path,
            tmp_path / "probe",
            [],
            binding,
            None,
        )
    with pytest.raises(CanonicalScreeningError, match="fields differ"):
        module._build_spec(
            policy,
            config,
            manifest,
            manifest_path,
            tmp_path / "probe",
            [],
            {**binding, "probe_execution_sha256": execution},
            execution,
        )


def test_ram_probe_controller_writes_one_preworker_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    artifact_root = tmp_path / "probe"
    claim = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_screening_ram_probe_controller_claim_v1"
        ),
        "controller_claim_sha256": "6" * 64,
    }

    def fail_after_claim(*_args: object, **_kwargs: object) -> None:
        artifact_root.mkdir()
        write_exclusive_json(artifact_root / "controller_claim.json", claim)
        (artifact_root / "input_policy.json").write_text(
            "{}\n", encoding="utf-8"
        )
        write_exclusive_json(artifact_root / "admission.json", {})
        raise KeyError("probe_execution_sha256")

    monkeypatch.setattr(module, "_run_controller_once", fail_after_claim)
    with pytest.raises(KeyError, match="probe_execution_sha256"):
        module._run_controller({}, tmp_path / "p", {}, tmp_path / "m", artifact_root)
    terminal = load_json(
        artifact_root / "controller_terminal.json", "controller terminal"
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "admission_to_spec"
    assert terminal["exception"]["type"] == "KeyError"
    assert terminal["retry_count"] == 0
    assert terminal["worker_started"] is False
    assert not (artifact_root / "probe_result.json").exists()
    before = (artifact_root / "controller_terminal.json").read_bytes()
    with pytest.raises(FileExistsError):
        module._write_controller_failure_terminal(
            artifact_root, RuntimeError("collision")
        )
    assert (artifact_root / "controller_terminal.json").read_bytes() == before


def test_ram_probe_controller_positive_mock_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    root = Path(__file__).parents[1]
    config = root / "configs/closeout/canonical_screening_512_v1.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_root = tmp_path / "probe"
    policy = validate_policy(root, config)
    policy["resources"] = {
        key: value
        for key, value in policy["resources"].items()
        if key not in {"ram_slot_budget_bytes", "ram_slot_budget_source"}
    }
    policy["resources"]["ram_budget_status"] = "probe_required"
    manifest = {"candidate_manifest_sha256": "2" * 64}
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(module, "_select_probe_candidates", lambda _value: [])
    monkeypatch.setattr(
        module,
        "assert_cpu_resource_admission",
        lambda *_args: {"memory_percent": 20.0},
    )
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {
                "index": index,
                "uuid": _gpu_uuid(index),
                "memory_free_mib": 24_000,
            }
            for index in range(4)
        ],
    )
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    monkeypatch.setattr(module, "_worker_environment", lambda _uuid: {})

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> dict:
            return {
                "violated": False,
                "violation_reason": None,
                "thread_failure": None,
            }

    class Process:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    def monitor(
        _process: object, _policy: dict, _guard: object
    ) -> tuple[int, int, None, None]:
        spec = load_json(artifact_root / "probe_spec.json", "probe spec")
        worker = {
            "schema_version": 1,
            "contract_type": module.PROBE_WORKER_RESULT_CONTRACT,
            "status": "succeeded",
            "probe_contract_sha256": spec["probe_contract_sha256"],
            "probe_execution_sha256": spec["probe_execution_sha256"],
            "purpose": spec["purpose"],
            "device_binding": {"physical_gpu_index": 0},
            "steps": [],
            "worker_vmhwm_bytes": 900,
            "failure": None,
            "completed_at": "2026-07-27T00:01:00+00:00",
        }
        worker["worker_result_sha256"] = canonical_digest(
            worker, "worker_result_sha256"
        )
        write_exclusive_json(artifact_root / "worker_result.json", worker)
        return 1000, 0, None, None

    monkeypatch.setattr(module, "RuntimeResourceGuard", Guard)
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "_monitor_probe_process", monitor)
    result = module._run_controller(
        policy, config, manifest, manifest_path, artifact_root
    )
    spec = load_json(artifact_root / "probe_spec.json", "probe spec")
    admission = load_json(artifact_root / "admission.json", "admission")
    assert result["status"] == "succeeded"
    assert spec["admission"] == {
        "path": str((artifact_root / "admission.json").resolve()),
        "sha256": hashlib.sha256(
            (artifact_root / "admission.json").read_bytes()
        ).hexdigest(),
        "canonical_sha256": admission["admission_sha256"],
    }
    assert (
        spec["probe_execution_sha256"]
        == admission["probe_execution_sha256"]
    )
    assert (artifact_root / "controller_claim.json").is_file()
    assert not (artifact_root / "controller_terminal.json").exists()
    assert (artifact_root / "probe_result.json").is_file()


def test_worker_environment_overrides_inherited_cuda_remap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _controller_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,1,0")
    env = module._worker_environment("GPU-authorized")
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-authorized"
    with pytest.raises(CanonicalScreeningError, match="UUID"):
        module._worker_environment("0")


def test_ram_projection_accepts_84_99_and_rejects_exact_85() -> None:
    module = _controller_module()
    source = {"contract_type": "fixture"}
    accepted = module._ram_reservation_projection(
        total_bytes=1_000_000,
        used_bytes=0,
        slot_budget_bytes=849_900,
        slot_count=1,
        admission_limit_percent=85,
        budget_source=source,
    )
    assert accepted["projected_used_percent"] == 84.99
    with pytest.raises(CanonicalScreeningError, match="RAM reservation"):
        module._ram_reservation_projection(
            total_bytes=1_000_000,
            used_bytes=0,
            slot_budget_bytes=850_000,
            slot_count=1,
            admission_limit_percent=85,
            budget_source=source,
        )


def test_sealed_4d_ram_probe_artifact_tree_and_source() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = validate_policy(
        root,
        root / "configs/closeout/canonical_screening_512_v1.json",
        verify_historical_output_evidence=False,
    )
    resources = policy["resources"]
    source = resources["ram_slot_budget_source"]
    seal = source["probe_artifact_seal"]
    assert resources["ram_budget_status"] == "sealed"
    assert resources["ram_slot_budget_bytes"] == 3_768_299_111
    assert source["peak_sampled_process_tree_rss_bytes"] == 3_275_694_080
    assert source["worker_vmhwm_bytes"] == 3_425_726_464
    assert source["ram_budget_basis_bytes"] == 3_425_726_464
    assert seal["file_count"] == 28
    assert seal["directory_count"] == 5
    assert seal["symlink_count"] == 0
    assert len([name for name in seal["files"] if name.endswith(".png")]) == 16
    assert seal["controller_terminal"] == "absent_by_contract"
    assert seal["scientific_result_reuse"] == "forbidden"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("controller_terminal", "present"),
        ("scientific_result_reuse", "allowed"),
        ("file_count", 27),
    ],
)
def test_sealed_4d_ram_probe_rejects_contract_tamper(
    field: str, value: object
) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "canonical policy",
    )
    seal = dict(raw["resources"]["ram_slot_budget_source"]["probe_artifact_seal"])
    seal[field] = value
    with pytest.raises(CanonicalScreeningError, match="artifact tree"):
        _validate_ram_probe_artifact_seal(root, seal)


@pytest.mark.parametrize("symlink_kind", ["root", "ancestor"])
def test_sealed_4d_ram_probe_rejects_path_component_symlinks(
    tmp_path: Path, symlink_kind: str
) -> None:
    source_repo = Path(__file__).resolve().parents[1]
    raw = load_json(
        source_repo / "configs/closeout/canonical_screening_512_v1.json",
        "canonical policy",
    )
    seal = raw["resources"]["ram_slot_budget_source"]["probe_artifact_seal"]
    source_root = (
        source_repo
        / "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__4d0345b6fc29cc8e"
    )
    expected_root = (
        tmp_path
        / "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__4d0345b6fc29cc8e"
    )
    if symlink_kind == "root":
        expected_root.parent.mkdir(parents=True)
        expected_root.symlink_to(source_root, target_is_directory=True)
    else:
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/closeout").symlink_to(
            source_repo / "artifacts/closeout",
            target_is_directory=True,
        )
    with pytest.raises(CanonicalScreeningError, match="symlink"):
        _validate_ram_probe_artifact_seal(tmp_path, seal)


def test_eight_worker_ram_projection_is_strictly_below_85_percent() -> None:
    module = _controller_module()
    total = 100_000_000_000
    slot_budget = 3_768_299_111
    slots = 8
    reserved = slots * slot_budget
    exact_85_used = 85_000_000_000 - reserved
    source = {"contract_type": "safa_canonical_screening_ram_budget_source_v2"}
    accepted = module._ram_reservation_projection(
        total_bytes=total,
        used_bytes=exact_85_used - 1,
        slot_budget_bytes=slot_budget,
        slot_count=slots,
        admission_limit_percent=85,
        budget_source=source,
    )
    assert accepted["slot_count"] == 8
    assert accepted["reserved_bytes"] == 30_146_392_888
    assert accepted["projected_used_bytes"] == 84_999_999_999
    assert accepted["projected_used_percent"] < 85
    with pytest.raises(CanonicalScreeningError, match="RAM reservation"):
        module._ram_reservation_projection(
            total_bytes=total,
            used_bytes=exact_85_used,
            slot_budget_bytes=slot_budget,
            slot_count=slots,
            admission_limit_percent=85,
            budget_source=source,
        )


def test_ram_probe_selects_largest_checkpoint_per_output_space(
    tmp_path: Path,
) -> None:
    module = _ram_probe_module()
    candidates = []
    for output_space, sizes in (("latent", (3, 7)), ("pixel", (5, 2))):
        for index, size in enumerate(sizes):
            checkpoint = tmp_path / f"{output_space}-{index}.pt"
            checkpoint.write_bytes(bytes([index + 1]) * size)
            candidates.append(
                {
                    "candidate_id": f"{output_space}-{index}",
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "checkpoint_model": "raw",
                    "output_contract": {
                        "output_contract_sha256": str(index) * 64,
                        "capability": {"output_space": output_space},
                    },
                }
            )
    selected = module._select_probe_candidates({"candidates": candidates})
    assert [
        (row["output_space"], row["candidate_id"], row["checkpoint_size_bytes"])
        for row in selected
    ] == [("latent", "latent-1", 7), ("pixel", "pixel-0", 5)]


def test_ram_probe_manifest_uses_current_plan_and_candidate_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    plan_path = tmp_path / "plan.json"
    plan = {"preflight_result_root": str((tmp_path / "preflight").resolve())}
    write_exclusive_json(plan_path, plan)
    manifest = {
        "checkpoint_plan": _bound(plan_path),
        "candidates": [{"candidate_id": "candidate"}],
    }
    manifest_path = tmp_path / "manifest.json"
    write_exclusive_json(manifest_path, manifest)
    policy = {"policy_sha256": "1" * 64}
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        module,
        "validate_checkpoint_plan",
        lambda value, **kwargs: calls.append(
            ("plan", (value, kwargs["policy"]))
        )
        or value,
    )
    monkeypatch.setattr(
        module,
        "validate_candidate_manifest",
        lambda value, **kwargs: calls.append(
            ("manifest", (value, kwargs["policy"]))
        )
        or value,
    )
    assert module._validate_manifest_envelope(manifest_path, policy) == manifest
    assert calls == [
        ("plan", (plan, policy)),
        ("manifest", (manifest, policy)),
    ]
    manifest["checkpoint_plan"]["sha256"] = "0" * 64
    manifest_path.write_bytes(canonical_json(manifest))
    with pytest.raises(CanonicalScreeningError, match="plan binding"):
        module._validate_manifest_envelope(manifest_path, policy)


def _sealed_ram_probe_fixture(tmp_path: Path) -> tuple[dict, int]:
    purpose = "resource_measurement_only_scientific_reuse_forbidden"
    gpu_uuid = _gpu_uuid(0)
    input_file = tmp_path / "input.json"
    input_file.write_text("{}\n", encoding="utf-8")
    input_binding = _bound(input_file)
    policy_snapshot = tmp_path / "input_policy.json"
    policy_snapshot.write_text(
        '{"resources":{"ram_budget_status":"probe_required"}}\n',
        encoding="utf-8",
    )
    policy_binding = {
        "path": str(
            (
                tmp_path / "configs/closeout/canonical_screening_512_v1.json"
            ).resolve()
        ),
        "sha256": hashlib.sha256(policy_snapshot.read_bytes()).hexdigest(),
        "canonical_sha256": "1" * 64,
        "snapshot": _bound(policy_snapshot),
    }
    selected = [
        {
            "candidate_id": f"{space}-candidate",
            "checkpoint_sha256": str(index + 2) * 64,
            "checkpoint_model": "raw",
            "checkpoint_size_bytes": 100 + index,
            "output_space": space,
            "output_contract_sha256": str(index + 4) * 64,
        }
        for index, space in enumerate(("latent", "pixel"))
    ]
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": (
                gpu_uuid if index == 0 else _gpu_uuid(index)
            ),
        }
        for index in range(4)
    ]
    spec = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_v1",
        "purpose": purpose,
        "policy": policy_binding,
        "candidate_manifest": input_binding,
        "selected_candidates": selected,
        "sample_manifest": input_binding,
        "sample_count": 8,
        "seed": 4549,
        "batch_size": 2,
        "authorized_gpu_registry": registry,
        "artifact_root": str(tmp_path.resolve()),
        "implementations": {"worker": input_binding},
        "retry_count": 0,
        "probe_sha256": None,
    }
    spec["probe_sha256"] = canonical_digest(spec, "probe_sha256")
    write_exclusive_json(tmp_path / "probe_spec.json", spec)
    admission = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_admission_v1",
        "probe_sha256": spec["probe_sha256"],
        "host": {},
        "gpu_snapshot": [],
        "authorized_gpu_registry": registry,
        "observed_at": "2026-07-26T00:00:00+00:00",
    }
    admission["admission_sha256"] = canonical_digest(
        admission, "admission_sha256"
    )
    write_exclusive_json(tmp_path / "admission.json", admission)
    worker = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_screening_ram_probe_worker_result_v1"
        ),
        "probe_sha256": spec["probe_sha256"],
        "purpose": purpose,
        "device_binding": {
            "physical_gpu_index": 0,
            "physical_gpu_uuid": gpu_uuid,
            "logical_cuda_index": 0,
            "runtime_cuda_uuid": gpu_uuid,
            "cuda_visible_devices": gpu_uuid,
        },
        "steps": [
            {
                **descriptor,
                "sample_count": 8,
                "generated_png_manifest_sha256": "8" * 64,
            }
            for descriptor in selected
        ],
        "worker_vmhwm_bytes": 900,
        "completed_at": "2026-07-26T00:01:00+00:00",
    }
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    write_exclusive_json(tmp_path / "worker_result.json", worker)
    worker_log = tmp_path / "worker.log"
    worker_log.write_text("probe\n", encoding="utf-8")
    peak = 1000
    budget = 1100
    method = (
        "ceil(max(peak_sampled_process_tree_rss_bytes,"
        "worker_vmhwm_bytes)*11/10);sampled_tree_every_0.1s_"
        "plus_worker_vmhwm_not_a_mathematical_instantaneous_tree_peak"
    )
    result = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_result_v1",
        "status": "succeeded",
        "purpose": purpose,
        "probe_sha256": spec["probe_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "worker_result_sha256": worker["worker_result_sha256"],
        "worker_log_sha256": hashlib.sha256(worker_log.read_bytes()).hexdigest(),
        "worker_returncode": 0,
        "termination": None,
        "peak_sampled_process_tree_rss_bytes": peak,
        "worker_vmhwm_bytes": 900,
        "ram_budget_basis_bytes": peak,
        "ram_slot_budget_bytes": budget,
        "budget_method": method,
        "measurement_factor_numerator": 11,
        "measurement_factor_denominator": 10,
        "runtime_resource_guard": {
            "violated": False,
            "violation_reason": None,
            "thread_failure": None,
        },
        "failure": None,
        "retry_count": 0,
        "completed_at": "2026-07-26T00:02:00+00:00",
    }
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path = tmp_path / "probe_result.json"
    write_exclusive_json(result_path, result)
    source = {
        "contract_type": "safa_canonical_screening_ram_budget_source_v1",
        "method": method,
        "measurement_factor_numerator": 11,
        "measurement_factor_denominator": 10,
        "peak_sampled_process_tree_rss_bytes": peak,
        "worker_vmhwm_bytes": 900,
        "ram_budget_basis_bytes": peak,
        "ram_slot_budget_bytes": budget,
        "probe_result": _bound(result_path),
    }
    return source, budget


def _reseal_ram_probe_fixture(
    tmp_path: Path, source: dict
) -> None:
    snapshot_path = tmp_path / "input_policy.json"
    spec_path = tmp_path / "probe_spec.json"
    spec = load_json(spec_path, "RAM probe spec")
    snapshot_binding = _bound(snapshot_path)
    spec["policy"]["sha256"] = snapshot_binding["sha256"]
    spec["policy"]["snapshot"] = snapshot_binding
    spec["probe_sha256"] = canonical_digest(spec, "probe_sha256")
    spec_path.write_bytes(canonical_json(spec))

    admission_path = tmp_path / "admission.json"
    admission = load_json(admission_path, "RAM probe admission")
    admission["probe_sha256"] = spec["probe_sha256"]
    admission["admission_sha256"] = canonical_digest(
        admission, "admission_sha256"
    )
    admission_path.write_bytes(canonical_json(admission))

    worker_path = tmp_path / "worker_result.json"
    worker = load_json(worker_path, "RAM probe worker result")
    worker["probe_sha256"] = spec["probe_sha256"]
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    worker_path.write_bytes(canonical_json(worker))

    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result["probe_sha256"] = spec["probe_sha256"]
    result["admission_sha256"] = admission["admission_sha256"]
    result["worker_result_sha256"] = worker["worker_result_sha256"]
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["probe_result"] = _bound(result_path)


def _upgrade_ram_probe_fixture_to_v2(tmp_path: Path, source: dict) -> None:
    spec_path = tmp_path / "probe_spec.json"
    spec = load_json(spec_path, "RAM probe spec")
    spec.pop("probe_sha256")
    spec["contract_type"] = "safa_canonical_screening_ram_probe_v2"
    spec["admission"] = None
    spec["probe_contract_sha256"] = None
    spec["probe_execution_sha256"] = None
    contract = ram_probe_contract_digest(spec)

    admission_path = tmp_path / "admission.json"
    admission = load_json(admission_path, "RAM probe admission")
    admission.pop("probe_sha256")
    admission["contract_type"] = (
        "safa_canonical_screening_ram_probe_admission_v2"
    )
    admission["probe_contract_sha256"] = contract
    admission["admission_evidence_sha256"] = (
        ram_probe_admission_evidence_digest(admission)
    )
    execution = ram_probe_execution_digest(
        contract,
        admission["authorized_gpu_registry"],
        admission["admission_evidence_sha256"],
    )
    admission["probe_execution_sha256"] = execution
    admission["admission_sha256"] = canonical_digest(
        admission, "admission_sha256"
    )
    admission_path.write_bytes(canonical_json(admission))

    spec["probe_contract_sha256"] = contract
    spec["probe_execution_sha256"] = execution
    spec["admission"] = {
        **_bound(admission_path),
        "canonical_sha256": admission["admission_sha256"],
    }
    spec_path.write_bytes(canonical_json(spec))

    worker_path = tmp_path / "worker_result.json"
    worker = load_json(worker_path, "RAM probe worker result")
    worker.pop("probe_sha256")
    worker["contract_type"] = (
        "safa_canonical_screening_ram_probe_worker_result_v2"
    )
    worker["status"] = "succeeded"
    worker["failure"] = None
    worker["probe_contract_sha256"] = contract
    worker["probe_execution_sha256"] = execution
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    worker_path.write_bytes(canonical_json(worker))

    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result.pop("probe_sha256")
    result["contract_type"] = "safa_canonical_screening_ram_probe_result_v2"
    result["probe_contract_sha256"] = contract
    result["probe_execution_sha256"] = execution
    result["worker_device_binding"] = worker["device_binding"]
    result["admission_sha256"] = admission["admission_sha256"]
    result["worker_result_sha256"] = worker["worker_result_sha256"]
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["probe_result"] = _bound(result_path)


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("probe_spec.json", "probe_contract_sha256"),
        ("probe_spec.json", "probe_execution_sha256"),
        ("admission.json", "authorized_gpu_registry"),
        ("worker_result.json", "probe_execution_sha256"),
        ("probe_result.json", "probe_contract_sha256"),
    ],
)
def test_sealed_ram_probe_v2_rejects_digest_chain_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    _upgrade_ram_probe_fixture_to_v2(tmp_path, source)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    assert (
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )["ram_slot_budget_bytes"]
        == budget
    )
    path = tmp_path / artifact
    changed = load_json(path, "tampered v2 RAM probe artifact")
    if field == "authorized_gpu_registry":
        changed[field][0]["physical_gpu_uuid"] = _gpu_uuid(1)
    else:
        changed[field] = "0" * 64
    own_digest = {
        "admission.json": "admission_sha256",
        "worker_result.json": "worker_result_sha256",
        "probe_result.json": "probe_result_sha256",
    }.get(artifact)
    if own_digest is not None:
        changed[own_digest] = canonical_digest(changed, own_digest)
    path.write_bytes(canonical_json(changed))
    if artifact == "probe_result.json":
        source["probe_result"] = _bound(path)
    with pytest.raises(CanonicalScreeningError, match="evidence chain"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    (
        ("probe_result.json", "status", "failed", "semantics"),
        ("input_policy.json", "policy", "forged", "snapshot binding"),
        ("probe_spec.json", "purpose", "scientific", "evidence chain"),
        ("admission.json", "probe_sha256", "0" * 64, "evidence chain"),
        (
            "worker_result.json",
            "device_binding",
            {
                "physical_gpu_index": 0,
                "physical_gpu_uuid": "GPU-tampered",
                "logical_cuda_index": 0,
                "runtime_cuda_uuid": "GPU-tampered",
                "cuda_visible_devices": "GPU-tampered",
            },
            "evidence chain",
        ),
    ),
)
def test_sealed_ram_probe_chain_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
    value: object,
    message: str,
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    assert (
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )["ram_slot_budget_bytes"]
        == budget
    )
    path = tmp_path / artifact
    changed = load_json(path, "tampered RAM probe artifact")
    changed[field] = value
    digest_fields = {
        "probe_result.json": "probe_result_sha256",
        "probe_spec.json": "probe_sha256",
        "admission.json": "admission_sha256",
        "worker_result.json": "worker_result_sha256",
    }
    if artifact in digest_fields:
        changed[digest_fields[artifact]] = canonical_digest(
            changed, digest_fields[artifact]
        )
    path.write_bytes(canonical_json(changed))
    if artifact == "probe_result.json":
        source["probe_result"] = _bound(path)
    with pytest.raises(CanonicalScreeningError, match=message):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_ram_probe_sampler_exception_terminates_and_reaps_worker_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 123

        def poll(self):
            return None

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: ((123, 1),)
    )
    monkeypatch.setattr(
        module,
        "_sample_or_reap_process_tree",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("sampler injected")),
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda process, **_kwargs: (
            cleanup_calls.append(process.pid)
            or {
                "term_sent": True,
                "kill_sent": False,
                "reaped_returncode": -15,
            }
        ),
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0
    assert returncode == -15
    assert failure == "RuntimeError: sampler injected"
    assert termination["term_sent"] is True
    assert cleanup_calls == [123]


def test_ram_probe_descendant_appearance_terminates_and_reaps_worker_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 321

        def poll(self):
            return None

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        module, "_process_descendants", lambda _root: ((654, 2),)
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda process, **_kwargs: (
            cleanup_calls.append(process.pid)
            or {
                "term_sent": True,
                "kill_sent": False,
                "reaped_returncode": -15,
            }
        ),
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0
    assert returncode == -15
    assert "forbidden descendant processes" in failure
    assert termination["term_sent"] is True
    assert cleanup_calls == [321]


def test_ram_probe_descendant_scan_tracks_children_that_escape_process_group(
    tmp_path: Path,
) -> None:
    module = _ram_probe_module()
    proc_root = tmp_path / "proc"

    def write_stat(
        pid: int, *, parent: int, process_group: int, start_time: int
    ) -> None:
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        fields = (
            ["S", str(parent), str(process_group)]
            + ["0"] * 16
            + [str(start_time)]
        )
        (directory / "stat").write_text(
            f"{pid} (fixture worker) {' '.join(fields)}\n",
            encoding="utf-8",
        )

    write_stat(100, parent=1, process_group=100, start_time=10)
    write_stat(101, parent=100, process_group=999, start_time=11)
    write_stat(102, parent=101, process_group=998, start_time=12)
    write_stat(200, parent=1, process_group=200, start_time=20)

    assert module._process_group_members(100, proc_root=proc_root) == (
        (100, 10),
    )
    assert module._process_descendants(100, proc_root=proc_root) == (
        (101, 11),
        (102, 12),
    )


def test_ram_probe_process_group_cleanup_escalates_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 456
        returncode = None
        waits = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout: int):
            assert timeout == 10
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("probe", timeout)
            self.returncode = -9
            return self.returncode

    process = Process()
    signals: list[tuple[int, object]] = []
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module,
        "_process_group_members",
        lambda _group: (
            ((process.pid, 1),) if process.returncode is None else ()
        ),
    )
    monotonic = iter((0.0, 11.0, 11.0, 11.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(module, "_signal_process_identities", lambda *_args: 0)

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == module.signal.SIGKILL:
            process.returncode = -9

    monkeypatch.setattr(
        module.os, "killpg", killpg
    )
    result = module._terminate_process_group(process)
    assert result == {
        "term_sent": True,
        "kill_sent": True,
        "reaped_returncode": -9,
    }
    assert signals == [
        (process.pid, module.signal.SIGTERM),
        (process.pid, module.signal.SIGKILL),
    ]


def test_ram_probe_cleanup_reaps_initial_zombie_before_group_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 457
        returncode = None

        def poll(self):
            self.returncode = 0
            return self.returncode

    process = Process()
    group_scans: list[int] = []
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module,
        "_process_group_members",
        lambda group: group_scans.append(group) or (),
    )
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("reaped zombie group must not be signalled")
        ),
    )
    assert module._terminate_process_group(process) == {
        "term_sent": False,
        "kill_sent": False,
        "reaped_returncode": 0,
    }
    assert group_scans == [process.pid]


def test_ram_probe_cleanup_accepts_esrch_only_after_empty_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 458
        returncode = None
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls > 1:
                self.returncode = 0
            return self.returncode

    process = Process()
    member_samples = iter((((process.pid, 1),), ()))
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: next(member_samples)
    )
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    result = module._terminate_process_group(process)
    assert result == {
        "term_sent": False,
        "kill_sent": False,
        "reaped_returncode": 0,
    }


def test_ram_probe_cleanup_esrch_with_live_members_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 459
        returncode = None

        def poll(self):
            return None

    process = Process()
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module,
        "_process_group_members",
        lambda _group: ((process.pid, 1),),
    )
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: process.pid)

    def killpg(_group, sig):
        if sig == module.signal.SIGKILL:
            raise ProcessLookupError()

    monkeypatch.setattr(module.os, "killpg", killpg)
    monotonic = iter((0.0, 11.0, 11.0, 22.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(CanonicalScreeningError, match="survived SIGKILL"):
        module._terminate_process_group(process)


def test_ram_probe_root_exit_cleans_residual_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 777
        returncode = 0

        def poll(self):
            return 0

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: ((888, 2),)
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda _process, **_kwargs: {
            "term_sent": True,
            "kill_sent": False,
            "reaped_returncode": 0,
        },
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0 and returncode == 0
    assert "residual process-group members" in failure
    assert termination["term_sent"] is True


def test_ram_probe_sampling_exit_cleans_residual_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 778
        returncode = 0

        def poll(self):
            return None

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: ((889, 2),)
    )
    monkeypatch.setattr(
        module, "_sample_or_reap_process_tree", lambda *_args: (None, 0)
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda _process, **_kwargs: {
            "term_sent": True,
            "kill_sent": False,
            "reaped_returncode": 0,
        },
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0 and returncode == 0
    assert "exited during RSS sampling" in failure
    assert termination["term_sent"] is True


def test_sealed_ram_budget_uses_higher_worker_vmhwm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result["peak_sampled_process_tree_rss_bytes"] = 500
    result["worker_vmhwm_bytes"] = 1000
    result["ram_budget_basis_bytes"] = 1000
    worker_path = tmp_path / "worker_result.json"
    worker = load_json(worker_path, "RAM probe worker result")
    worker["worker_vmhwm_bytes"] = 1000
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    worker_path.write_bytes(canonical_json(worker))
    result["worker_result_sha256"] = worker["worker_result_sha256"]
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["peak_sampled_process_tree_rss_bytes"] = 500
    source["worker_vmhwm_bytes"] = 1000
    source["ram_budget_basis_bytes"] = 1000
    source["probe_result"] = _bound(result_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    validated = _validate_ram_slot_budget_source(
        tmp_path,
        source,
        declared_budget_bytes=budget,
        expected_predecessor_policy_sha256="1" * 64,
    )
    assert validated["ram_budget_basis_bytes"] == 1000


def test_sealed_ram_budget_rejects_vmhwm_chain_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result["peak_sampled_process_tree_rss_bytes"] = 500
    result["worker_vmhwm_bytes"] = 1000
    result["ram_budget_basis_bytes"] = 1000
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["peak_sampled_process_tree_rss_bytes"] = 500
    source["worker_vmhwm_bytes"] = 1000
    source["ram_budget_basis_bytes"] = 1000
    source["probe_result"] = _bound(result_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    with pytest.raises(CanonicalScreeningError, match="evidence chain"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_sealed_ram_budget_rejects_forged_snapshot_canonical_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "2" * 64},
    )
    with pytest.raises(CanonicalScreeningError, match="predecessor policy"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_sealed_ram_budget_rejects_recursive_sealed_snapshot_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    snapshot_path = tmp_path / "input_policy.json"
    snapshot_path.write_text(
        '{"resources":{"ram_budget_status":"sealed"}}\n',
        encoding="utf-8",
    )
    _reseal_ram_probe_fixture(tmp_path, source)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recursive policy validation must not start")
        ),
    )
    with pytest.raises(CanonicalScreeningError, match="probe-required"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_screen512_gate_requires_exact_primary_repeat_smoke(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy, manifest, manifest_path, policy_path, admission, _ = _manifest_fixture(
        tmp_path
    )
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    candidate = manifest["candidates"][0]
    controller_ready, observer_ready = _ready_bindings(
        tmp_path, policy, admission, "smoke8"
    )
    baseline_rows = {}
    baseline_results = {}
    for replicate in ("primary", "repeat"):
        request = build_run_request(
            policy,
            policy_path,
            manifest,
            manifest_path,
            candidate,
            "smoke8",
            replicate,
            paths["runs"],
            admission,
            controller_ready,
            observer_ready,
        )
        request_path = (
            paths["run_requests"]
            / f"smoke8_{replicate}"
            / f"{candidate['candidate_id']}.json"
        )
        write_exclusive_json(request_path, request)
        output = Path(request["output_dir"])
        output.mkdir(parents=True)
        generated = output / "generated"
        generated.mkdir()
        rows = []
        for index in range(8):
            source_path = tmp_path / "sources" / f"{index:06d}.png"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(f"source{index}".encode())
            candidate_path = generated / f"{index:06d}.png"
            candidate_path.write_bytes(f"png{index}".encode())
            rows.append(
                {
                    "sample_id": f"s{index}",
                    "run_request_sha256": request["run_request_sha256"],
                    "checkpoint_sha256": request["candidate"][
                        "checkpoint_sha256"
                    ],
                    "checkpoint_model": request["candidate"][
                        "checkpoint_model"
                    ],
                    "source_path": str(source_path.resolve()),
                    "source_sha256": hashlib.sha256(
                        source_path.read_bytes()
                    ).hexdigest(),
                    "candidate_path": str(candidate_path.resolve()),
                    "candidate_sha256": hashlib.sha256(
                        candidate_path.read_bytes()
                    ).hexdigest(),
                    "native_output_sha256": hashlib.sha256(
                        f"native{index}".encode()
                    ).hexdigest(),
                    "output_contract_sha256": request["output_contract"][
                        "output_contract_sha256"
                    ],
                    "output_contract_type": request["output_contract"][
                        "contract_type"
                    ],
                    "decoder_registry_sha256": request[
                        "output_decoder_registry"
                    ]["decoder_registry_sha256"],
                    "output_space": request["output_contract"]["capability"][
                        "output_space"
                    ],
                    "native_output_shape": [3, 224, 224],
                    "native_rgb_shape": [3, 224, 224],
                    "native_rgb_size": [224, 224],
                    "quality_protocol_family": request[
                        "quality_protocol_family"
                    ],
                    "nfe": request["nfe"],
                    "e0_cosine": 0.8,
                    "edev_cosine": 0.7,
                    "arcface_source_face_count": 1,
                    "arcface_candidate_face_count": 1,
                    "arcface_source_candidate_cosine": 0.1,
                }
            )
        per_sample = output / "per_sample.jsonl"
        _write_jsonl(per_sample, rows)
        claim = _run_claim(
            policy,
            request,
            worker_pid=100 + (replicate == "repeat"),
        )
        write_exclusive_json(output / "claim.json", claim)
        evidence = _evidence(policy, request)
        evidence["per_sample_sha256"] = hashlib.sha256(
            per_sample.read_bytes()
        ).hexdigest()
        result = build_run_result(
            request,
            claim,
            policy,
            status="completed",
            completed_at="2026-07-26T00:01:00+00:00",
            evidence=evidence,
        )
        write_exclusive_json(output / "result.json", result)
        baseline_rows[replicate] = json.loads(json.dumps(rows))
        baseline_results[replicate] = json.loads(json.dumps(result))
    module._require_smoke_success(policy, manifest, paths)
    mutations = (
        lambda rows: rows[0].__setitem__("sample_id", "tampered"),
        lambda rows: rows[1].__setitem__("sample_id", rows[0]["sample_id"]),
        lambda rows: (
            rows[0].__setitem__("sample_id", "s1"),
            rows[1].__setitem__("sample_id", "s0"),
        ),
        lambda rows: rows[0].__setitem__(
            "native_rgb_shape", [3, 256, 256]
        ),
        lambda rows: rows[0].__setitem__(
            "output_contract_sha256", "f" * 64
        ),
    )
    for mutate in mutations:
        for replicate in ("primary", "repeat"):
            output = (
                paths["runs"]
                / f"smoke8_{replicate}"
                / candidate["candidate_id"]
            )
            changed_rows = json.loads(json.dumps(baseline_rows[replicate]))
            mutate(changed_rows)
            per_sample = output / "per_sample.jsonl"
            per_sample.unlink()
            _write_jsonl(per_sample, changed_rows)
            changed_result = json.loads(
                json.dumps(baseline_results[replicate])
            )
            changed_result["evidence"]["per_sample_sha256"] = hashlib.sha256(
                per_sample.read_bytes()
            ).hexdigest()
            changed_result["run_result_sha256"] = canonical_digest(
                changed_result,
                "run_result_sha256",
            )
            (output / "result.json").write_bytes(
                canonical_json(changed_result)
            )
        with pytest.raises(CanonicalScreeningError, match="smoke8 per-sample"):
            module._require_smoke_success(policy, manifest, paths)
        for replicate in ("primary", "repeat"):
            output = (
                paths["runs"]
                / f"smoke8_{replicate}"
                / candidate["candidate_id"]
            )
            per_sample = output / "per_sample.jsonl"
            per_sample.unlink()
            _write_jsonl(per_sample, baseline_rows[replicate])
            (output / "result.json").write_bytes(
                canonical_json(baseline_results[replicate])
            )


def test_e0_cosine_uses_locked_target_z_not_source_embedding() -> None:
    torch = pytest.importorskip("torch")
    generated = torch.tensor([[1.0, 0.0]])
    target_z = torch.tensor([[1.0, 0.0]])
    source_e0 = torch.tensor([[0.0, 1.0]])
    generated_edev = torch.tensor([[0.0, 1.0]])
    source_edev = torch.tensor([[0.0, 1.0]])
    e0_cosine, edev_cosine = _representation_cosines(
        generated, target_z, generated_edev, source_edev
    )
    assert e0_cosine.item() == pytest.approx(1.0)
    assert edev_cosine.item() == pytest.approx(1.0)
    assert torch.nn.functional.cosine_similarity(generated, source_e0).item() == 0.0


def test_edev_source_loader_is_locked_to_256(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    black = tmp_path / "black.png"
    white = tmp_path / "white.png"
    image_module.new("RGB", (17, 23), color=(0, 0, 0)).save(black)
    image_module.new("RGB", (17, 23), color=(255, 255, 255)).save(white)
    batch = _load_source_pixel_batch([black, white], 256, "cpu")
    assert tuple(batch.shape) == (2, 3, 256, 256)
    assert float(batch.min()) == 0.0
    assert float(batch.max()) == 1.0
    assert float(batch[0].max()) == 0.0
    assert float(batch[1].min()) == 1.0


def test_kid_subset_8_accepts_eight_real_and_fake_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safa.closeout.canonical_quality as canonical_quality
    torch = pytest.importorskip("torch")
    root = tmp_path
    real_paths = [root / f"real_{index}.png" for index in range(8)]
    generated_paths = [root / f"generated_{index}.png" for index in range(8)]
    manifest = root / "canonical_kid_test_manifest.jsonl"
    per_sample = root / "canonical_kid_test_per_sample.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    per_sample.write_text("{}\n", encoding="utf-8")

    class FakeKid:
        def __init__(self, subset_size: int, normalize: bool) -> None:
            self.subset_size = subset_size
            self.real = 0
            self.fake = 0

        def update(self, _image, *, real: bool) -> None:
            if real:
                self.real += 1
            else:
                self.fake += 1

        def compute(self):
            assert self.real >= self.subset_size
            assert self.fake >= self.subset_size
            return 0.1, 0.01

    fake_quality = types.SimpleNamespace(
        manifest_image_paths=lambda **_kwargs: (
            [f"s{index}" for index in range(8)],
            real_paths,
            generated_paths,
        ),
        quality_eval_device=lambda _device: torch.device("cpu"),
        prepare_metric_for_device=lambda metric, device: (metric, device),
        load_image_uint8=lambda _path: torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
        image_to_device=lambda image, _device: image,
        seed_metric_randomness=lambda _seed, _device: None,
        metric_scalar=float,
        asset_manifest_digest=lambda paths, labels: hashlib.sha256(
            canonical_json([str(path) for path in paths] + list(labels))
        ).hexdigest(),
    )
    monkeypatch.setattr(canonical_quality, "_load_quality_module", lambda _binding: fake_quality)
    torchmetrics = types.ModuleType("torchmetrics")
    image = types.ModuleType("torchmetrics.image")
    kid = types.ModuleType("torchmetrics.image.kid")
    kid.KernelInceptionDistance = FakeKid
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", image)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", kid)
    result = evaluate_locked_kid(
        quality_script={"path": "/locked", "sha256": "a" * 64},
        real_index=root / "index.jsonl",
        generated_dir=root,
        sample_id_manifest=manifest,
        per_sample_jsonl=per_sample,
        subset_seed=4549,
        subset_size=8,
        device="cpu",
    )
    assert result["kid_mean"] == 0.1
    assert result["kid_subset_size"] == 8


def test_invalid_result_validation_leaves_no_immutable_result(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    claim = _run_claim(policy, request)
    result = build_run_result(
        request,
        claim,
        policy,
        status="completed",
        completed_at="2026-07-26T00:01:00+00:00",
        evidence=_evidence(policy, request),
    )
    result["evidence"]["policy_sha256"] = "f" * 64
    result["run_result_sha256"] = canonical_digest(result, "run_result_sha256")
    path = tmp_path / "result.json"
    with pytest.raises(CanonicalScreeningError, match="policy_sha256"):
        _write_validated_run_result(path, result, request, claim, policy)
    assert not path.exists()


def test_free_slot_pool_reuses_exact_out_of_order_completion() -> None:
    module = _controller_module()
    pool = module.FreeSlotPool([(0, 0), (0, 1), (1, 0)])
    assert pool.acquire() == (0, 0)
    second = pool.acquire()
    third = pool.acquire()
    assert (second, third) == ((0, 1), (1, 0))
    pool.release(third)
    assert pool.acquire() == third
    pool.release(second)
    with pytest.raises(CanonicalScreeningError, match="invalid GPU slot release"):
        pool.release(second)


def test_controller_cleanup_terminates_workers_and_releases_owned_lock(
    tmp_path: Path,
) -> None:
    module = _controller_module()

    class Process:
        def __init__(self) -> None:
            self.pid = 4242
            self.terminated = False
            self.waited = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None) -> int:
            self.waited = True
            return -15

    class Log:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    pool = module.FreeSlotPool([(0, 0)])
    slot = pool.acquire()
    lock = tmp_path / "owned.lock"
    lock.write_text("owned", encoding="utf-8")
    process = Process()
    log = Log()
    active = [{
        "process": process,
        "request": tmp_path / "request.json",
        "lock": lock,
        "log_handle": log,
        "slot": slot,
    }]
    guard = types.SimpleNamespace(unregister_worker_pid=lambda _pid: None)
    module._cleanup_active_workers(active, pool, guard)
    assert active == []
    assert process.terminated and process.waited and log.closed
    assert not lock.exists()
    assert pool.free_count == 1


def test_monitor_is_append_only_and_audit_reconstructable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 12.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (1, 2))
    gpu_rows = [
        {
            "index": index,
            "uuid": _gpu_uuid(index),
            "temperature_c": 40,
        }
        for index in range(4)
    ]
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: gpu_rows)
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    admission = module._write_admission(
        policy,
        paths,
        "smoke8",
        {
            **_admission_snapshot(policy),
            "gpus": gpu_rows,
        },
    )
    path = module._append_monitor_sample(
        policy, paths, "smoke8", admission=admission
    )
    module._append_monitor_sample(
        policy, paths, "smoke8", terminal=True, admission=admission
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["terminal"] is False and rows[1]["terminal"] is True
    assert rows[0]["gpus"][0]["uuid"] == _gpu_uuid(0)
    assert rows[0]["artifacts"]["generated_png"] == 0


def test_cpu_admission_never_depends_on_gpu_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    policy["resources"].update({
        "cpu_admission_percent": 90,
        "ram_admission_percent": 85,
        "disk_admission_percent": 85,
    })
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (1, 2))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    snapshot = module.assert_cpu_resource_admission(policy, tmp_path)
    assert snapshot["admission_kind"] == "cpu_only"


@pytest.mark.parametrize(
    ("cpu_percent", "should_pass"),
    [(89.1, True), (90.0, False)],
)
def test_cpu_startup_admission_uses_strict_below_90_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_percent: float,
    should_pass: bool,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    policy["resources"]["cpu_admission_percent"] = 90
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: cpu_percent)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    if should_pass:
        assert module.assert_cpu_resource_admission(
            policy, tmp_path
        )["cpu_load_percent"] == cpu_percent
    else:
        with pytest.raises(CanonicalScreeningError, match="CPU admission failed"):
            module.assert_cpu_resource_admission(policy, tmp_path)


def test_cpu_window_requires_two_consecutive_windows_and_latches() -> None:
    module = _controller_module()
    single = module.CpuWindowState(90.0, 2)
    assert single.record(93.0) is False
    assert single.record(10.0) is False
    assert single.consecutive_high == 0
    exact = module.CpuWindowState(90.0, 2)
    assert exact.record(90.0) is False
    assert exact.consecutive_high == 1
    assert exact.record(90.0) is True
    consecutive = module.CpuWindowState(90.0, 2)
    assert consecutive.record(91.0) is False
    assert consecutive.record(92.0) is True
    assert consecutive.record(10.0) is True
    assert consecutive.violated is True


def test_runtime_guard_preserves_ram_disk_and_swap_hard_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(module, "_cpu_times", lambda: (100, 50))

    class FiniteWait:
        def __init__(self, intervals: int) -> None:
            self.intervals = intervals
            self.calls = 0

        def wait(self, _seconds: int) -> bool:
            self.calls += 1
            return self.calls > self.intervals

    def run_case(
        name: str,
        *,
        memory_percent: float,
        disk_percent: float,
        swaps: list[tuple[int, int]],
        intervals: int,
    ) -> str:
        monotonic_values = iter(float(10 * index) for index in range(intervals + 1))
        swap_values = iter(swaps)
        monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
        monkeypatch.setattr(module, "_memory_percent", lambda: memory_percent)
        monkeypatch.setattr(module, "_disk_percent", lambda _path: disk_percent)
        monkeypatch.setattr(module, "_swap_pages", lambda: next(swap_values))
        guard = module.RuntimeResourceGuard(
            policy, tmp_path / f"{name}.jsonl", tmp_path
        )
        guard._stop = FiniteWait(intervals)
        guard._run()
        with pytest.raises(CanonicalScreeningError) as error:
            guard.raise_if_violated()
        return str(error.value)

    assert "RAM runtime hard stop" in run_case(
        "ram",
        memory_percent=90.0,
        disk_percent=10.0,
        swaps=[(0, 0), (0, 0)],
        intervals=1,
    )
    assert "disk runtime hard stop" in run_case(
        "disk",
        memory_percent=10.0,
        disk_percent=90.0,
        swaps=[(0, 0), (0, 0)],
        intervals=1,
    )
    assert "sustained swap I/O" in run_case(
        "swap",
        memory_percent=10.0,
        disk_percent=10.0,
        swaps=[(0, 0), (1, 0), (2, 0), (3, 0)],
        intervals=3,
    )


def test_runtime_guard_exposes_monitor_thread_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(
        module,
        "_cpu_times",
        lambda: (_ for _ in ()).throw(RuntimeError("proc stat injected")),
    )
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "guard.jsonl", tmp_path
    )
    guard._run()
    with pytest.raises(CanonicalScreeningError, match="proc stat injected"):
        guard.raise_if_violated()


def test_runtime_guard_hard_stops_unknown_gpu_pid_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(module, "_cpu_times", lambda: (100, 50))
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 10.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {"index": row["physical_gpu_index"], "uuid": row["physical_gpu_uuid"]}
            for row in registry
        ],
    )
    monkeypatch.setattr(
        module,
        "_gpu_compute_processes",
        lambda: [
            {
                "gpu_uuid": registry[0]["physical_gpu_uuid"],
                "pid": 99991,
                "process_name": "foreign",
            }
        ],
    )

    class OneSample:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _seconds: int) -> bool:
            self.calls += 1
            return self.calls > 1

        def set(self) -> None:
            self.calls = 2

    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "unknown_pid.jsonl", tmp_path, registry
    )
    launched = False

    def forbidden_factory():
        nonlocal launched
        launched = True
        return types.SimpleNamespace(pid=99991)

    with pytest.raises(CanonicalScreeningError, match="unknown compute PID"):
        guard.launch_authorized_worker(forbidden_factory)
    assert launched is False
    guard._stop = OneSample()
    guard._run()
    with pytest.raises(CanonicalScreeningError, match="unknown compute PID"):
        guard.raise_if_violated()
    sample = json.loads(
        (tmp_path / "unknown_pid.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert sample["unknown_compute_processes"][0]["pid"] == 99991


def test_runtime_guard_allows_atomically_registered_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(module, "_cpu_times", lambda: (100, 50))
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 10.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {"index": row["physical_gpu_index"], "uuid": row["physical_gpu_uuid"]}
            for row in registry
        ],
    )
    launched = False

    def compute_processes():
        return (
            [{"gpu_uuid": registry[0]["physical_gpu_uuid"], "pid": 4242}]
            if launched
            else []
        )

    monkeypatch.setattr(module, "_gpu_compute_processes", compute_processes)

    class OneSample:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _seconds: int) -> bool:
            self.calls += 1
            return self.calls > 1

        def set(self) -> None:
            self.calls = 2

    process = types.SimpleNamespace(pid=4242)
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "allowed_pid.jsonl", tmp_path, registry
    )

    def launch_worker():
        nonlocal launched
        launched = True
        return process

    assert guard.launch_authorized_worker(launch_worker) is process
    guard._stop = OneSample()
    guard._run()
    guard.raise_if_violated()
    guard.unregister_worker_pid(4242)
    assert guard.stop()["final_active_worker_pids"] == []


def test_cpu_worker_handshake_orders_rehash_launch_register_check_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 10.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: [])
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    events = []
    process = types.SimpleNamespace(pid=5151)
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "handshake.jsonl", tmp_path
    )

    def initial_validator():
        events.append("initial_rehash")
        return {"stage": "initial"}

    def factory():
        events.append("popen")
        return process

    launched, validation = guard.launch_cpu_worker(
        factory, initial_validator
    )
    assert launched is process
    assert validation == {"stage": "initial"}

    def final_validator():
        assert 5151 in guard._active_worker_pids
        events.append("final_rehash")
        return {"stage": "final"}

    def publisher(_validation, snapshot):
        assert snapshot["active_worker_pids"] == [5151]
        events.append("release")

    guard.release_worker_after_handshake(
        5151, final_validator, publisher
    )
    assert events == [
        "initial_rehash",
        "popen",
        "final_rehash",
        "release",
    ]
    guard.unregister_worker_pid(5151)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu", "CPU release hard gate"),
        ("ram", "RAM release hard gate"),
        ("disk", "disk release hard gate"),
        ("swap", "swap I/O release hard gate"),
        ("gpu_memory", "release memory hard gate"),
        ("gpu_temperature", "release temperature hard gate"),
        ("unknown_pid", "unknown compute PID"),
        ("thread_failure", "guard failed before worker release"),
    ),
)
def test_release_lock_hard_gates_fresh_resource_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        module,
        "_cpu_load_percent",
        lambda: 90.0 if case == "cpu" else 10.0,
    )
    monkeypatch.setattr(
        module,
        "_memory_percent",
        lambda: 90.0 if case == "ram" else 10.0,
    )
    monkeypatch.setattr(
        module,
        "_disk_percent",
        lambda _path: 90.0 if case == "disk" else 10.0,
    )
    swaps = iter(
        ((0, 0), (1, 0))
        if case == "swap"
        else ((0, 0), (0, 0))
    )
    monkeypatch.setattr(module, "_swap_pages", lambda: next(swaps))
    gpus = [
        {
            "index": row["physical_gpu_index"],
            "uuid": row["physical_gpu_uuid"],
            "memory_total_mib": 24576,
            "memory_used_mib": (
                23000
                if case == "gpu_memory"
                and row["physical_gpu_index"] == 0
                else 3
            ),
            "memory_free_mib": (
                1576
                if case == "gpu_memory"
                and row["physical_gpu_index"] == 0
                else 24573
            ),
            "temperature_c": (
                86
                if case == "gpu_temperature"
                and row["physical_gpu_index"] == 0
                else 35
            ),
        }
        for row in registry
    ]
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: gpus)
    monkeypatch.setattr(
        module,
        "_gpu_compute_processes",
        lambda: (
            [
                {
                    "gpu_uuid": registry[0]["physical_gpu_uuid"],
                    "pid": 9999,
                    "process_name": "foreign",
                }
            ]
            if case == "unknown_pid"
            else []
        ),
    )
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "release.jsonl", tmp_path, registry
    )
    if case == "thread_failure":
        guard._thread_failure = RuntimeError("injected guard failure")
    published = False

    def publisher(*_args):
        nonlocal published
        published = True

    with pytest.raises(CanonicalScreeningError, match=message):
        guard.release_worker_after_handshake(
            5151, lambda: {"validated": True}, publisher
        )
    assert published is False
    assert 5151 not in guard._active_worker_pids


def test_worker_release_pid_tamper_fails_before_heavy_import(
    tmp_path: Path,
) -> None:
    policy, request = _run_fixture(tmp_path, mode="screen512")
    handshake = _handshake_fixture(
        policy, request, worker_pid=os.getpid()
    )
    ready = handshake["worker_ready"]
    ready_binding = handshake["worker_ready_binding"]
    release = json.loads(json.dumps(handshake["worker_release"]))
    release["worker_pid"] = 999999
    release["worker_release_sha256"] = canonical_digest(
        release, "worker_release_sha256"
    )
    release_path = tmp_path / "tampered-worker-release.json"
    write_exclusive_json(release_path, release)
    with pytest.raises(CanonicalScreeningError, match="contract mismatch"):
        _wait_worker_release(
            release_path,
            ready,
            ready_binding,
            request,
            policy,
            timeout_seconds=0.01,
        )


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "contract_type",
        "policy_sha256",
        "phase",
        "worker_pid",
        "gpu_index",
        "gpu_uuid",
        "run_request_sha256",
        "request",
        "final_release",
        "verification_order",
        "rehashed_bindings",
        "rehashed_bindings_sha256",
        "controller_claim",
        "screening_worker_sha256",
        "controller_implementation_sha256",
        "cuda_visible_devices",
        "heavy_modules_absent",
        "loaded_heavy_modules",
        "asset_content_verification",
        "external_gpu_race_contract",
        "ready_at",
        "worker_ready_sha256",
    ),
)
def test_worker_ready_rejects_each_missing_field(
    tmp_path: Path, field: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    ready = json.loads(
        json.dumps(_handshake_fixture(policy, request)["worker_ready"])
    )
    ready.pop(field)
    if field != "worker_ready_sha256":
        ready["worker_ready_sha256"] = canonical_digest(
            ready, "worker_ready_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_ready_value(ready, request, policy)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "contract_type",
        "policy_sha256",
        "run_request_sha256",
        "worker_pid",
        "gpu_index",
        "gpu_uuid",
        "worker_ready",
        "verification_order",
        "rehashed_bindings",
        "rehashed_bindings_sha256",
        "resource_snapshot",
        "asset_content_verification",
        "external_gpu_race_contract",
        "validated_at",
        "controller_launch_rehash_sha256",
    ),
)
def test_controller_rehash_rejects_each_missing_field(
    tmp_path: Path, field: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    value = json.loads(
        json.dumps(
            _handshake_fixture(policy, request)["controller_rehash"]
        )
    )
    value.pop(field)
    if field != "controller_launch_rehash_sha256":
        value["controller_launch_rehash_sha256"] = canonical_digest(
            value, "controller_launch_rehash_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_controller_launch_rehash_value(value, request, policy)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "contract_type",
        "policy_sha256",
        "phase",
        "worker_pid",
        "run_request_sha256",
        "worker_ready",
        "controller_launch_rehash",
        "resource_snapshot",
        "external_gpu_race_contract",
        "released_at",
        "worker_release_sha256",
    ),
)
def test_worker_release_rejects_each_missing_field(
    tmp_path: Path, field: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    value = json.loads(
        json.dumps(_handshake_fixture(policy, request)["worker_release"])
    )
    value.pop(field)
    if field != "worker_release_sha256":
        value["worker_release_sha256"] = canonical_digest(
            value, "worker_release_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_release_value(value, request, policy)


@pytest.mark.parametrize(
    "mutation",
    (
        "heavy_false",
        "loaded_torch",
        "gpu_uuid",
        "external_race",
        "nested_request",
        "asset_digest",
        "extra_field",
    ),
)
def test_worker_ready_rejects_semantic_tamper(
    tmp_path: Path, mutation: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    value = json.loads(
        json.dumps(_handshake_fixture(policy, request)["worker_ready"])
    )
    if mutation == "heavy_false":
        value["heavy_modules_absent"] = False
    elif mutation == "loaded_torch":
        value["loaded_heavy_modules"] = ["torch"]
    elif mutation == "gpu_uuid":
        value["gpu_uuid"] = _gpu_uuid(1)
    elif mutation == "external_race":
        value["external_gpu_race_contract"]["compute_mode_changed"] = True
    elif mutation == "nested_request":
        value["rehashed_bindings"]["request"]["sha256"] = "f" * 64
        value["rehashed_bindings_sha256"] = hashlib.sha256(
            canonical_json(value["rehashed_bindings"])
        ).hexdigest()
    elif mutation == "asset_digest":
        value["asset_content_verification"]["observed_digest"] = "f" * 64
    else:
        value["unexpected"] = True
    value["worker_ready_sha256"] = canonical_digest(
        value, "worker_ready_sha256"
    )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_ready_value(value, request, policy)


@pytest.mark.parametrize(
    "mutation",
    (
        "ready_rehash",
        "admission",
        "runtime_registry",
        "unknown_pid",
        "active_pid",
        "cpu_limit",
        "swap_delta",
        "gpu_memory",
        "gpu_headroom",
        "gpu_temperature",
        "guard_failure",
        "external_race",
        "extra_field",
    ),
)
def test_worker_release_rejects_semantic_tamper(
    tmp_path: Path, mutation: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    handshake = _handshake_fixture(policy, request)
    value = json.loads(json.dumps(handshake["worker_release"]))
    if mutation == "ready_rehash":
        rehash = json.loads(json.dumps(handshake["controller_rehash"]))
        rehash["rehashed_bindings_sha256"] = "e" * 64
        rehash["controller_launch_rehash_sha256"] = canonical_digest(
            rehash, "controller_launch_rehash_sha256"
        )
        path = tmp_path / "tampered-controller-rehash.json"
        write_exclusive_json(path, rehash)
        value["controller_launch_rehash"] = {
            **_bound(path),
            "canonical_sha256": rehash[
                "controller_launch_rehash_sha256"
            ],
        }
    elif mutation == "admission":
        value["resource_snapshot"]["admission"]["cpu_load_percent"] = 4.0
    elif mutation == "runtime_registry":
        value["resource_snapshot"]["runtime_guard"][
            "runtime_gpu_registry"
        ] = list(reversed(request["authorized_gpu_registry"]))
    elif mutation == "unknown_pid":
        value["resource_snapshot"]["runtime_guard"][
            "unknown_compute_processes"
        ] = [{"pid": 999}]
    elif mutation == "active_pid":
        value["resource_snapshot"]["runtime_guard"][
            "active_worker_pids"
        ] = []
    elif mutation == "cpu_limit":
        value["resource_snapshot"]["runtime_guard"][
            "cpu_load_percent"
        ] = 90.0
    elif mutation == "swap_delta":
        value["resource_snapshot"]["runtime_guard"][
            "swap_io_delta"
        ]["in"] = 1
    elif mutation == "gpu_memory":
        gpu = value["resource_snapshot"]["runtime_guard"]["gpu"][0]
        gpu["memory_used_mib"] = 23000
        gpu["memory_free_mib"] = 1576
    elif mutation == "gpu_headroom":
        gpu = value["resource_snapshot"]["runtime_guard"]["gpu"][0]
        gpu["memory_used_mib"] = 23000
        gpu["memory_free_mib"] = 1576
    elif mutation == "gpu_temperature":
        value["resource_snapshot"]["runtime_guard"]["gpu"][0][
            "temperature_c"
        ] = 86
    elif mutation == "guard_failure":
        value["resource_snapshot"]["runtime_guard"][
            "guard_thread_failure"
        ] = {"type": "RuntimeError"}
    elif mutation == "external_race":
        value["external_gpu_race_contract"]["compute_mode_changed"] = True
    else:
        value["unexpected"] = True
    value["worker_release_sha256"] = canonical_digest(
        value, "worker_release_sha256"
    )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_release_value(value, request, policy)


def test_real_cpu_subprocess_completes_production_pre_cuda_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy, config, request, request_path = _real_policy_run_fixture(
        tmp_path, module
    )
    final_release = _final_release_for_single_request(
        tmp_path, policy, request, request_path
    )
    gpu_index = 0
    gpu_uuid = request["authorized_gpu_registry"][0][
        "physical_gpu_uuid"
    ]
    ready_path = tmp_path / "real-handshake/worker_ready.json"
    release_path = tmp_path / "real-handshake/worker_release.json"
    controller_rehash_path = (
        tmp_path / "real-handshake/controller_rehash.json"
    )
    gpu_rows = [
        {
            "index": row["physical_gpu_index"],
            "uuid": row["physical_gpu_uuid"],
            "memory_total_mib": 24576,
            "memory_used_mib": 3,
            "memory_free_mib": 24573,
            "temperature_c": 35,
        }
        for row in request["authorized_gpu_registry"]
    ]
    controller_resources = {
        "observed_at": "2026-07-27T00:00:00+00:00",
        "cpu_load_percent": 1.0,
        "memory_percent": 2.0,
        "disk_percent": 3.0,
        "swap_pages": {"in": 0, "out": 0},
        "gpus": gpu_rows,
        "authorized_gpu_registry": request[
            "authorized_gpu_registry"
        ],
        "ram_reservation": _admission_snapshot(policy)[
            "ram_reservation"
        ],
        "compute_processes": [],
    }
    monkeypatch.setattr(
        module,
        "assert_resource_admission",
        lambda *_args, **_kwargs: json.loads(
            json.dumps(controller_resources)
        ),
    )
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 1.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 2.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 3.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: json.loads(json.dumps(gpu_rows)),
    )
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    child_code = """
import json
from pathlib import Path
import sys
from safa.closeout.canonical_screening import load_json, validate_policy
from safa.closeout.canonical_screening_worker import (
    HEAVY_MODULE_ROOTS,
    prepare_screening_request_for_cuda,
)
repo_root = Path(sys.argv[1]).resolve()
config = Path(sys.argv[2]).resolve()
request_path = Path(sys.argv[3]).resolve()
final_release_path = Path(sys.argv[4]).resolve()
ready_path = Path(sys.argv[5]).resolve()
release_path = Path(sys.argv[6]).resolve()
gpu_index = int(sys.argv[7])
gpu_uuid = sys.argv[8]
policy = validate_policy(
    repo_root, config, verify_historical_output_evidence=False
)
prepared = prepare_screening_request_for_cuda(
    request_path,
    gpu_index,
    gpu_uuid,
    policy,
    {
        "path": str(final_release_path),
        "sha256": __import__("hashlib").sha256(
            final_release_path.read_bytes()
        ).hexdigest(),
        "canonical_sha256": load_json(
            final_release_path, "child final release"
        )["final_release_admission_sha256"],
    },
    ready_path,
    release_path,
)
print(json.dumps({
    "next_stage": prepared["next_stage"],
    "worker_ready_sha256": prepared["worker_ready"]["worker_ready_sha256"],
    "worker_release_sha256": prepared["worker_release"]["worker_release_sha256"],
    "pre_asset_digest": prepared["pre_cuda"]["asset_content_verification"][
        "observed_digest"
    ],
    "post_asset_digest": prepared["post_release"][
        "asset_content_verification"
    ]["observed_digest"],
    "heavy_modules": sorted(
        name for name in HEAVY_MODULE_ROOTS if name in sys.modules
    ),
}, sort_keys=True))
"""
    child_env = {
        **os.environ,
        "TMUX": "canonical-pre-cuda-integration",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child_env.pop("CUDA_DEVICE_ORDER", None)
    guard = module.RuntimeResourceGuard(
        policy,
        tmp_path / "real-handshake/resource.jsonl",
        tmp_path,
        request["authorized_gpu_registry"],
    )
    process = None
    try:
        process, initial_rehash = guard.launch_cpu_worker(
            lambda: subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    str(Path(__file__).parents[1]),
                    str(config),
                    str(request_path),
                    str(final_release["path"]),
                    str(ready_path),
                    str(release_path),
                    str(gpu_index),
                    gpu_uuid,
                ],
                cwd=Path(__file__).parents[1],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
            lambda: module._validate_launch_integrity(
                policy,
                config,
                module._paths(
                    tmp_path / "real-campaign",
                    policy["policy_sha256"],
                ),
                request_path,
                final_release,
            ),
        )
        assert initial_rehash["worker_pid"] is None
        ready, ready_binding = module._wait_worker_ready(
            process,
            ready_path,
            request_path,
            request,
            policy,
            gpu_index,
            gpu_uuid,
            timeout_seconds=120.0,
        )
        final_rehash = None

        def publish_release(validation, guard_snapshot):
            nonlocal final_rehash
            final_rehash = dict(validation)
            validate_controller_launch_rehash_value(
                final_rehash, request, policy
            )
            publish_exclusive_json(
                controller_rehash_path, final_rehash
            )
            controller_rehash_binding = {
                **_bound(controller_rehash_path),
                "canonical_sha256": final_rehash[
                    "controller_launch_rehash_sha256"
                ],
            }
            module._publish_worker_release(
                release_path,
                policy,
                request,
                process.pid,
                ready_binding,
                controller_rehash_binding,
                {
                    "admission": final_rehash[
                        "resource_snapshot"
                    ],
                    "runtime_guard": dict(guard_snapshot),
                },
            )

        guard.release_worker_after_handshake(
            process.pid,
            lambda: module._validate_launch_integrity(
                policy,
                config,
                module._paths(
                    tmp_path / "real-campaign",
                    policy["policy_sha256"],
                ),
                request_path,
                final_release,
                worker_pid=process.pid,
                gpu_index=gpu_index,
                gpu_uuid=gpu_uuid,
                worker_ready=ready_binding,
            ),
            publish_release,
        )
        stdout, stderr = process.communicate(timeout=120.0)
        assert process.returncode == 0, stderr
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.pid in guard._active_worker_pids:
                guard.unregister_worker_pid(process.pid)
    payload = json.loads(stdout)
    release = validate_worker_release_value(
        load_json(release_path, "real worker release"),
        request,
        policy,
        expected_worker_pid=process.pid,
    )
    assert final_rehash is not None
    assert payload["next_stage"] == "runtime_cuda_binding"
    assert payload["heavy_modules"] == []
    assert payload["worker_ready_sha256"] == ready[
        "worker_ready_sha256"
    ]
    assert payload["worker_release_sha256"] == release[
        "worker_release_sha256"
    ]
    expected_asset_digest = policy["output_decoder_registry"]["latent"][
        "directory"
    ]["digest"]
    assert payload["pre_asset_digest"] == expected_asset_digest
    assert payload["post_asset_digest"] == expected_asset_digest
    controller_ready = load_json(
        Path(request["controller_ready"]["path"]),
        "real controller ready",
    )
    assert ready["controller_claim"] == controller_ready[
        "controller_claim"
    ]
    assert release["controller_launch_rehash"][
        "canonical_sha256"
    ] == final_rehash["controller_launch_rehash_sha256"]
    output_dir = Path(request["output_dir"])
    assert not output_dir.exists()
    assert not (output_dir / "claim.json").exists()


def test_malformed_release_still_writes_atomic_worker_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    write_exclusive_json(request_path, {"request": "fixture"})
    ready_path = tmp_path / "handshake" / "worker_ready.json"
    release_path = tmp_path / "handshake" / "worker_release.json"
    write_exclusive_json(
        ready_path,
        {"worker_ready_sha256": "a" * 64},
    )
    write_exclusive_json(release_path, {"malformed": True})

    def reject_malformed(*_args, **_kwargs):
        release = load_json(release_path, "malformed release")
        if "worker_release_sha256" not in release:
            raise CanonicalScreeningError(
                "malformed worker release is missing its digest"
            )
        raise AssertionError("malformed release unexpectedly passed")

    monkeypatch.setattr(
        screening_worker_module,
        "_execute_screening_request_impl",
        reject_malformed,
    )
    with pytest.raises(CanonicalScreeningError, match="malformed"):
        execute_screening_request(
            request_path,
            0,
            _gpu_uuid(0),
            {"policy_sha256": "b" * 64},
            {},
            ready_path,
            release_path,
        )
    terminal_path = ready_path.parent / "worker_terminal.json"
    terminal = load_json(terminal_path, "malformed release terminal")
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "CanonicalScreeningError"
    assert terminal["worker_release"]["canonical_sha256"] is None
    assert terminal["worker_terminal_sha256"] == canonical_digest(
        terminal, "worker_terminal_sha256"
    )
    assert terminal_path.read_bytes().endswith(b"\n")


def test_launch_bootstrap_rejects_invalid_config_before_resource_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    request_path = tmp_path / "bootstrap-request.json"
    write_exclusive_json(request_path, request)
    sampled = False

    def forbidden_sample(*_args, **_kwargs):
        nonlocal sampled
        sampled = True
        raise AssertionError("resource sample must follow rehash")

    monkeypatch.setattr(module, "assert_resource_admission", forbidden_sample)
    with pytest.raises(
        module.ControllerBootstrapError,
        match="omits implementations",
    ):
        module._validate_launch_integrity(
            policy,
            Path(request["policy"]["path"]),
            module._paths(tmp_path / "campaign", policy["policy_sha256"]),
            request_path,
            {},
        )
    assert sampled is False


def test_real_controller_main_cpu_dry_run_uses_current_policy_without_writes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "scripts/run_canonical_checkpoint_screening.py"
    config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    campaign = tmp_path / "dry-run-campaign"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--phase",
            "plan",
            "--campaign-root",
            str(campaign),
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(completed.stdout)
    assert payload["phase"] == "plan"
    assert payload["execute"] is False
    assert payload["policy_sha256"]
    assert not campaign.exists()


def test_real_worker_policy_bootstrap_failure_writes_atomic_terminal(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "scripts/run_canonical_checkpoint_screening.py"
    source_config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    raw = json.loads(source_config.read_text(encoding="utf-8"))
    raw["implementations"]["controller"]["sha256"] = "0" * 64
    config = tmp_path / "tampered-policy.json"
    config.write_bytes(canonical_json(raw))
    request_path = tmp_path / "request.json"
    request_path.write_text('{"malformed":true}\n', encoding="utf-8")
    ready_path = tmp_path / "handshake/worker_ready.json"
    release_path = tmp_path / "handshake/worker_release.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--phase",
            "smoke8",
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--execute",
            "--request",
            str(request_path),
            "--worker-ready-path",
            str(ready_path),
            "--worker-release-path",
            str(release_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    terminal_path = (
        ready_path.parent / "worker_bootstrap_terminal.json"
    )
    terminal = load_json(terminal_path, "worker bootstrap terminal")
    assert terminal["stage"] == "policy_bootstrap"
    assert terminal["status"] == "failed"
    assert terminal["worker_pid"] > 0
    assert terminal["request"]["path"] == str(request_path.resolve())
    assert terminal["worker_bootstrap_terminal_sha256"] == canonical_digest(
        terminal, "worker_bootstrap_terminal_sha256"
    )
    assert not ready_path.exists()
    assert not release_path.exists()
    assert not (ready_path.parent / "worker_terminal.json").exists()


def test_preflight_monitor_never_queries_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (1, 2))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    monkeypatch.setattr(
        module,
        "_gpu_compute_processes",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    sample = module._monitor_sample(
        policy, paths, "preflight", terminal=False
    )
    assert sample["gpus"] is None
    assert sample["compute_processes"] is None


def test_supersession_evidence_binds_ea7_failed_smoke_chain(
    tmp_path: Path,
) -> None:
    old_policy = (
        "ea7ae71fd662526b9a45bf3cc6d283884"
        "aefc380b292c8f273169a35f42ffc28"
    )
    policy_root = (
        tmp_path
        / "artifacts/closeout/historical-canonical-512-v1/by_policy"
        / old_policy
    )
    primary_requests = policy_root / "run_requests/smoke8_primary"
    repeat_requests = policy_root / "run_requests/smoke8_repeat"
    run_requests = []
    run_claims = []
    failed_results = []
    worker_logs = []
    failure_message = (
        "The size of tensor a (4) must match the size of tensor b (3) "
        "at non-singleton dimension 1"
    )
    for index in range(193):
        candidate_id = f"g_{index:016x}_raw"
        request_path = primary_requests / f"{candidate_id}.json"
        if index < 8:
            request = {
                "contract_type": "safa_canonical_screening_run_request_v1",
                "mode": "smoke8",
                "replicate": "primary",
                "sample_count": 8,
                "batch_size": 2,
                "seed": 4549,
                "policy": {"canonical_sha256": old_policy},
                "candidate": {"candidate_id": candidate_id},
            }
            request["run_request_sha256"] = canonical_digest(
                request, "run_request_sha256"
            )
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_bytes(canonical_json(request))
            run_dir = policy_root / "runs/smoke8_primary" / candidate_id
            claim = {
                "contract_type": "safa_canonical_screening_run_claim_v1",
                "run_request_sha256": request["run_request_sha256"],
            }
            claim["run_claim_sha256"] = canonical_digest(
                claim, "run_claim_sha256"
            )
            claim_path = run_dir / "claim.json"
            claim_path.parent.mkdir(parents=True, exist_ok=True)
            claim_path.write_bytes(canonical_json(claim))
            result = {
                "contract_type": "safa_canonical_screening_run_result_v1",
                "run_request_sha256": request["run_request_sha256"],
                "run_claim_sha256": claim["run_claim_sha256"],
                "status": "failed",
                "failure": {
                    "type": "RuntimeError",
                    "message": failure_message,
                },
            }
            result["run_result_sha256"] = canonical_digest(
                result, "run_result_sha256"
            )
            result_path = run_dir / "result.json"
            result_path.write_bytes(canonical_json(result))
            log_path = policy_root / "logs" / f"smoke8_primary__{candidate_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(failure_message, encoding="utf-8")
            run_requests.append(_bound(request_path))
            run_claims.append(_bound(claim_path))
            failed_results.append(_bound(result_path))
            worker_logs.append(_bound(log_path))
        else:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text("{}\n", encoding="utf-8")
        repeat_path = repeat_requests / f"{candidate_id}.json"
        repeat_path.parent.mkdir(parents=True, exist_ok=True)
        repeat_path.write_text("{}\n", encoding="utf-8")
    monitor = _bound(policy_root / "logs/smoke8__monitor.jsonl")
    runtime = _bound(policy_root / "logs/smoke8__runtime_resource_windows.jsonl")
    summary_path = policy_root / "summaries/smoke8__failed.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(
        canonical_json(
            {
                "phase": "smoke8",
                "reason": "worker_nonzero_exit",
                "failures": [
                    f"{binding['path']}: exit_code=1"
                    for binding in run_requests
                ],
                "monitor_log": monitor,
                "runtime_resource_guard": {
                    "samples": runtime,
                    "violated": False,
                    "violation_reason": None,
                    "thread_failure": None,
                    "final_cpu_consecutive_high": 0,
                    "final_swap_consecutive_io": 0,
                },
            }
        )
    )
    supersedes = {
        "policy_sha256": old_policy,
        "classification": "started_incomplete",
        "phase": "smoke8",
        "request_count": 386,
        "primary_failed_count": 8,
        "repeat_result_count": 0,
        "screen512_result_count": 0,
        "generated_png_count": 0,
        "failed_summary": _bound(summary_path),
        "run_requests": run_requests,
        "run_claims": run_claims,
        "failed_results": failed_results,
        "worker_logs": worker_logs,
        "resource_monitor": monitor,
        "runtime_resource_windows": runtime,
    }
    assert (
        validate_supersession_evidence(tmp_path, supersedes)["classification"]
        == "started_incomplete"
    )
    tampered = json.loads(json.dumps(supersedes))
    tampered["run_claims"][0]["sha256"] = "f" * 64
    with pytest.raises(CanonicalScreeningError, match="SHA256 mismatch"):
        validate_supersession_evidence(tmp_path, tampered)


def test_current_arcface_binding_reaches_official_validator_and_probe_request() -> None:
    root = Path(__file__).parents[1]
    policy_path = root / "configs/closeout/canonical_screening_512_v1.json"
    policy = validate_policy(root, policy_path)
    expected_fields = {
        "path",
        "sha256",
        "bootstrap_claim_path",
        "bootstrap_claim_sha256",
        "bootstrap_claim_file_sha256",
        "bootstrap_result_path",
        "bootstrap_result_sha256",
        "bootstrap_result_file_sha256",
    }
    assert set(policy["arcface"]["execution_probe"]) == expected_fields
    contract = _load_arcface_contract(
        {
            "arcface": policy["arcface"],
            "source_index": policy["protocol"]["source_index"],
        }
    )
    assert set(contract["execution_probe"]) == expected_fields

    module = _ram_probe_module()
    manifest_path = (
        root
        / "artifacts/closeout/historical-canonical-512-v1/"
        "candidate_manifest__4c5ecb55501fa6b0.json"
    )
    manifest = load_json(manifest_path, "4c5 candidate manifest")
    request = module._probe_request(
        policy,
        manifest,
        manifest["candidates"][0],
        {"probe_execution_sha256": "e" * 64},
    )
    assert set(request["arcface"]["execution_probe"]) == expected_fields


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "probe_file_sha",
        "claim_canonical_sha",
        "claim_file_sha",
        "result_canonical_sha",
        "result_file_sha",
        "path_escape",
    ],
)
def test_arcface_execution_probe_binding_tamper_fails_closed(
    mutation: str,
) -> None:
    root = Path(__file__).parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "current policy",
    )
    arcface = json.loads(json.dumps(raw["arcface"]))
    binding = arcface["execution_probe"]
    if mutation == "missing":
        binding.pop("bootstrap_claim_sha256")
    elif mutation == "extra":
        binding["unexpected"] = "x"
    elif mutation == "probe_file_sha":
        binding["sha256"] = "0" * 64
    elif mutation == "claim_canonical_sha":
        binding["bootstrap_claim_sha256"] = "0" * 64
    elif mutation == "claim_file_sha":
        binding["bootstrap_claim_file_sha256"] = "0" * 64
    elif mutation == "result_canonical_sha":
        binding["bootstrap_result_sha256"] = "0" * 64
    elif mutation == "result_file_sha":
        binding["bootstrap_result_file_sha256"] = "0" * 64
    elif mutation == "path_escape":
        binding["bootstrap_claim_path"] = "../outside.json"
    else:
        raise AssertionError(mutation)
    with pytest.raises(CanonicalScreeningError):
        validate_arcface_execution_probe_binding(
            root, binding, arcface_contract=arcface
        )


def test_arcface_binding_rejects_ancestor_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "probe.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(CanonicalScreeningError, match="must not be symlinks"):
        _require_no_repo_path_component_symlinks(
            tmp_path, "alias/probe.json", "ArcFace probe"
        )


def test_arcface_binding_delegates_coherent_semantic_tamper(
    tmp_path: Path,
) -> None:
    from safa.evaluation.r9_evaluator_worker import _canonical_digest

    root = Path(__file__).parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "current policy",
    )
    arcface = json.loads(json.dumps(raw["arcface"]))
    original = arcface["execution_probe"]
    probe = tmp_path / "probe.json"
    claim_path = tmp_path / "claim.json"
    result_path = tmp_path / "result.json"
    probe.write_bytes((root / original["path"]).read_bytes())
    claim = load_json(root / original["bootstrap_claim_path"], "claim")
    claim["kind"] = "not_arcface_profile"
    claim["probe_output"] = str(probe.resolve())
    claim["bootstrap_claim_sha256"] = _canonical_digest(
        claim, "bootstrap_claim_sha256"
    )
    write_exclusive_json(claim_path, claim)
    result = load_json(root / original["bootstrap_result_path"], "result")
    result["bootstrap_claim_sha256"] = claim["bootstrap_claim_sha256"]
    result["bootstrap_result_sha256"] = _canonical_digest(
        result, "bootstrap_result_sha256"
    )
    write_exclusive_json(result_path, result)
    arcface["execution_probe"] = {
        "path": str(probe.resolve()),
        "sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "bootstrap_claim_path": str(claim_path.resolve()),
        "bootstrap_claim_sha256": claim["bootstrap_claim_sha256"],
        "bootstrap_claim_file_sha256": hashlib.sha256(
            claim_path.read_bytes()
        ).hexdigest(),
        "bootstrap_result_path": str(result_path.resolve()),
        "bootstrap_result_sha256": result["bootstrap_result_sha256"],
        "bootstrap_result_file_sha256": hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest(),
    }
    with pytest.raises(CanonicalScreeningError, match="claim policy mismatch"):
        validate_arcface_execution_probe_binding(
            tmp_path,
            arcface["execution_probe"],
            arcface_contract=arcface,
        )


def test_supersession_tree_rejects_recursive_symlink(
    tmp_path: Path,
) -> None:
    (tmp_path / "root/nested").mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    (tmp_path / "root/nested/alias").symlink_to(target)
    with pytest.raises(CanonicalScreeningError, match="must not contain symlinks"):
        _require_tree_without_symlinks(
            tmp_path / "root", "4c5 failed root"
        )


def test_worker_rejects_truncated_arcface_probe_request() -> None:
    from safa.evaluation.r9_evaluator_worker import R9EvaluatorError

    root = Path(__file__).parents[1]
    policy = validate_policy(
        root, root / "configs/closeout/canonical_screening_512_v1.json"
    )
    arcface = json.loads(json.dumps(policy["arcface"]))
    arcface["execution_probe"].pop("bootstrap_result_file_sha256")
    with pytest.raises(
        R9EvaluatorError,
        match="provenance fields are not canonical",
    ):
        _load_arcface_contract(
            {
                "arcface": arcface,
                "source_index": policy["protocol"]["source_index"],
            }
        )


def test_current_policy_binds_5dbb_preflight_smoke_and_execution_audit() -> None:
    root = Path(__file__).parents[1]
    policy_path = root / "configs/closeout/canonical_screening_512_v1.json"

    policy = validate_policy(root, policy_path)

    supersedes = policy["supersedes"]
    assert supersedes["policy_sha256"] == (
        "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af"
    )
    assert supersedes["classification"] == (
        "completed_preflight_and_smoke8_scientific_success_"
        "execution_barrier_incomplete_superseded"
    )
    assert supersedes["counts"]["preflight_request_count"] == 193
    assert supersedes["counts"]["preflight_result_count"] == 193
    assert supersedes["counts"]["preflight_valid_count"] == 193
    assert supersedes["counts"]["run_request_count"] == 386
    assert supersedes["counts"]["run_result_count"] == 386
    assert supersedes["counts"]["generated_png_count"] == 3088
    assert supersedes["counts"]["screen512_request_count"] == 0
    assert supersedes["execution_audit"]["execution_compliance"] == "p1_failed"
    assert supersedes["execution_audit"]["external_observer_ready"] == "absent"
    assert supersedes["execution_audit"]["screen512"] == "never_started"
    assert supersedes["scientific_result_reuse"] == "forbidden"
    assert (
        supersedes["successor_execution"]
        == "fresh_full_193_preflight_and_smoke8"
    )
    assert supersedes["ram_budget_source_policy_sha256"] == (
        "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
    )
    assert "gpu_wrapper" in policy["implementations"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("request_count", "counts differ"),
        ("reuse", "status differs"),
        ("root_digest", "evidence root differs"),
        ("preflight_path_alias", "file identity differs"),
        ("observer_audit", "execution audit differs"),
        ("successor", "status differs"),
        ("ram_lineage", "status differs"),
    ],
)
def test_5dbb_supersession_tampering_fails_closed(
    mutation: str,
    match: str,
) -> None:
    root = Path(__file__).parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "current policy",
    )
    supersedes = json.loads(json.dumps(raw["supersedes"]))
    if mutation == "request_count":
        supersedes["counts"]["run_request_count"] = 385
    elif mutation == "reuse":
        supersedes["scientific_result_reuse"] = "allowed"
    elif mutation == "root_digest":
        supersedes["evidence_root"]["digest"] = "0" * 64
    elif mutation == "preflight_path_alias":
        supersedes["files"]["final_plan"]["path"] = supersedes["files"][
            "candidate_manifest"
        ]["path"]
    elif mutation == "observer_audit":
        supersedes["execution_audit"]["external_observer_ready"] = "present"
    elif mutation == "successor":
        supersedes["successor_execution"] = "fresh_full_193_preflight"
    elif mutation == "ram_lineage":
        supersedes["ram_budget_source_policy_sha256"] = "0" * 64
    else:
        raise AssertionError(mutation)

    with pytest.raises(CanonicalScreeningError, match=match):
        validate_supersession_evidence(root, supersedes)


def test_unknown_supersession_policy_sha_fails_closed() -> None:
    with pytest.raises(CanonicalScreeningError, match="unknown supersession"):
        validate_supersession_evidence(
            Path(__file__).parents[1],
            {"policy_sha256": "0" * 64},
        )


def test_preflight_attempt_failure_writes_claim_and_terminal_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "a" * 64)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    request = plan["preflight_requests"][0]
    request_path = (
        paths["preflight_requests"]
        / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json"
    )
    write_exclusive_json(request_path, request)
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "assert_cpu_resource_admission",
        lambda *_args: {"admission_kind": "cpu_only"},
    )
    monkeypatch.setattr(
        module,
        "preflight_generator_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    guard = types.SimpleNamespace(raise_if_violated=lambda: None)
    with pytest.raises(RuntimeError, match="injected"):
        module.materialize_preflights(policy, paths, guard, "d" * 64)
    attempts = paths["preflight_control"] / "attempts"
    claim = load_json(next(attempts.glob("*.claim.json")), "attempt claim")
    terminal = load_json(next(attempts.glob("*.terminal.json")), "attempt terminal")
    assert terminal["attempt_claim_sha256"] == claim["attempt_claim_sha256"]
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "RuntimeError"
    assert list(paths["preflight_results"].glob("*.json")) == []


def test_runtime_stop_before_result_writes_one_failed_attempt_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "a" * 64)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    request = plan["preflight_requests"][0]
    write_exclusive_json(
        paths["preflight_requests"]
        / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json",
        request,
    )
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module, "preflight_generator_checkpoint", lambda *_args, **_kwargs: {}
    )

    class Guard:
        def __init__(self) -> None:
            self.calls = 0

        def raise_if_violated(self) -> None:
            self.calls += 1
            if self.calls == 2:
                raise CanonicalScreeningError("CPU runtime hard stop")

    with pytest.raises(CanonicalScreeningError, match="CPU runtime hard stop"):
        module.materialize_preflights(policy, paths, Guard(), "d" * 64)
    attempts = paths["preflight_control"] / "attempts"
    terminals = list(attempts.glob("*.terminal.json"))
    assert len(terminals) == 1
    terminal = load_json(terminals[0], "attempt terminal")
    assert terminal["status"] == "failed"
    assert terminal["failure"]["message"] == "CPU runtime hard stop"
    assert list(paths["preflight_results"].glob("*.json")) == []


def test_controller_failure_persists_log_and_global_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setenv("TMUX", "fixture")
    admission_calls = 0

    def admit(*_args):
        nonlocal admission_calls
        admission_calls += 1
        return {"admission_kind": "cpu_only"}

    class FakeGuard:
        def __init__(self, _policy, sample_path: Path, _disk_path: Path) -> None:
            self.started = False
            self.sample_path = sample_path

        def start(self) -> None:
            self.started = True
            self.sample_path.parent.mkdir(parents=True, exist_ok=True)
            self.sample_path.write_bytes(b'{"sample":1}\n')

        def stop(self) -> dict:
            return {
                "started": self.started,
                "thread_failure": None,
                "violation_reason": None,
                "violated": False,
                "samples": {
                    "path": str(self.sample_path.resolve()),
                    "sha256": hashlib.sha256(
                        self.sample_path.read_bytes()
                    ).hexdigest(),
                },
            }

    monkeypatch.setattr(module, "assert_cpu_resource_admission", admit)
    monkeypatch.setattr(module, "RuntimeResourceGuard", FakeGuard)
    monkeypatch.setattr(
        module,
        "materialize_preflights",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("controller injected")),
    )
    with pytest.raises(RuntimeError, match="controller injected"):
        module._execute_preflight_controller(policy, paths)
    control = paths["preflight_control"]
    terminal = load_json(control / "controller_terminal.json", "controller terminal")
    assert admission_calls == 1
    assert terminal["status"] == "failed"
    assert terminal["failure"]["message"] == "controller injected"
    assert terminal["runtime_resource_guard"]["started"] is True
    samples = terminal["runtime_resource_guard"]["samples"]
    assert Path(samples["path"]).is_file()
    assert samples["sha256"] == hashlib.sha256(
        Path(samples["path"]).read_bytes()
    ).hexdigest()
    assert "controller_exception" in (control / "controller.log").read_text(
        encoding="utf-8"
    )


def test_wrapper_records_native_stderr_and_sigkill_without_controller_claim(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    policy_root = tmp_path / "campaign" / "by_policy" / ("1" * 64)
    value = wrapper.run_wrapped_controller(
        policy_root=policy_root,
        policy_sha256="1" * 64,
        config=config,
        command=[
            sys.executable,
            "-c",
            (
                "import os,signal;"
                "os.write(2,b'native-before-kill\\n');"
                "os.kill(os.getpid(),signal.SIGKILL)"
            ),
        ],
    )
    assert value["exit_code"] == 137
    assert value["signal"] == 9
    assert value["controller_claim"] is None
    assert value["controller_terminal"] is None
    log_path = Path(value["controller_process_log"]["path"])
    assert log_path.read_bytes() == b"native-before-kill\n"
    assert load_json(
        policy_root / "preflight_control" / "wrapper_exit.json", "wrapper exit"
    ) == value


def test_wrapper_records_pre_main_failure_without_controller_artifacts(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    config = tmp_path / "policy.json"
    config.write_text("{bad policy}\n", encoding="utf-8")
    policy_root = tmp_path / "campaign" / "by_policy" / ("2" * 64)
    value = wrapper.run_wrapped_controller(
        policy_root=policy_root,
        policy_sha256="2" * 64,
        config=config,
        command=[
            sys.executable,
            "-c",
            "import os,sys;os.write(1,b'pre-main\\n');sys.exit(2)",
        ],
    )
    assert value["exit_code"] == 2
    assert value["signal"] is None
    assert value["controller_claim"] is None
    assert value["controller_terminal"] is None
    assert Path(value["controller_process_log"]["path"]).read_bytes() == b"pre-main\n"


def test_preflight_tmux_wrapper_has_exit_recorder_and_no_timeout(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    commands = module._tmux_commands(
        policy, policy_path, tmp_path / "campaign", "preflight"
    )
    controller = " ".join(commands["controller"])
    assert "run_canonical_preflight_wrapper.py" in controller
    assert "--policy-sha256" in controller
    assert "timeout" not in controller.lower()


def test_current_policy_preflight_refuses_partial_result_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "c" * 64)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    request = plan["preflight_requests"][0]
    write_exclusive_json(
        paths["preflight_requests"]
        / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json",
        request,
    )
    write_exclusive_json(paths["preflight_results"] / "partial.json", {"partial": True})
    monkeypatch.setenv("TMUX", "fixture")
    guard = types.SimpleNamespace(raise_if_violated=lambda: None)
    with pytest.raises(CanonicalScreeningError, match="refuses result reuse"):
        module.materialize_preflights(policy, paths, guard, "d" * 64)


def test_write_exclusive_json_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    write_exclusive_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_exclusive_json(path, {"value": 2})


def test_atomic_publish_exposes_only_complete_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    original_link = __import__("os").link
    observed = []

    def inspect_link(source, target):
        assert Path(target) == path
        assert not path.exists()
        observed.append(load_json(Path(source), "temporary publication"))
        return original_link(source, target)

    module = sys.modules[publish_exclusive_json.__module__]
    monkeypatch.setattr(module.os, "link", inspect_link)
    publish_exclusive_json(path, value)
    assert observed == [value]
    assert load_json(path, "published ready") == value
    assert list(tmp_path.glob(".ready.json.publish-*")) == []


def test_atomic_publish_race_has_one_winner_and_valid_final(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observer_ready.json"
    barrier = threading.Barrier(2)
    successes = []
    failures = []

    def publish(value: dict) -> None:
        barrier.wait()
        try:
            publish_exclusive_json(path, value)
            successes.append(value)
        except CanonicalScreeningError as exc:
            failures.append(exc)

    values = [{"writer": 1}, {"writer": 2}]
    threads = [
        threading.Thread(target=publish, args=(value,)) for value in values
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 1
    assert len(failures) == 1
    assert load_json(path, "race winner") == successes[0]
    assert list(tmp_path.glob(".observer_ready.json.publish-*")) == []


def test_gpu_pre_ready_admission_failure_writes_terminal_without_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "1" * 64}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}\n", encoding="utf-8")
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setattr(
        module,
        "assert_resource_admission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CanonicalScreeningError("fixture admission race")
        ),
    )
    claim = {
        "controller_claim_sha256": "3" * 64,
        "wrapper_claim": {"canonical_sha256": "4" * 64},
        "observer_launch": {"canonical_sha256": "5" * 64},
    }
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (
            claim["wrapper_claim"],
            claim["observer_launch"],
        ),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "claim.json", claim
        ),
    )
    with pytest.raises(CanonicalScreeningError, match="admission race"):
        module._run_gpu_phase(
            policy, policy_path, paths, "screen512"
        )
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "controller_terminal.json",
        "GPU terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "startup_admission"
    assert not paths["run_requests"].exists()


def test_gpu_preclaim_failure_writes_only_bootstrap_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "6" * 64}
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.delenv("TMUX", raising=False)
    with pytest.raises(CanonicalScreeningError, match="inside tmux"):
        module._run_gpu_phase(policy, config, paths, "screen512")
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "bootstrap_terminal.json",
        "bootstrap terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "tmux_bootstrap"
    assert terminal["controller_claim"] is None
    assert not (
        paths["gpu_control"] / "screen512" / "controller_terminal.json"
    ).exists()


def test_observer_ready_timeout_is_bounded_and_writes_no_requests(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "2" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    with pytest.raises(CanonicalScreeningError, match="timed out"):
        module._wait_observer_ready(
            policy,
            paths,
            "screen512",
            {"controller_ready_sha256": "3" * 64},
            {"canonical_sha256": "4" * 64},
            timeout_seconds=0.01,
        )
    assert not paths["run_requests"].exists()


def test_duplicate_observer_claim_fails_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "5" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    claim_path = (
        paths["gpu_control"] / "smoke8" / "observer_claim.json"
    )
    write_exclusive_json(claim_path, {"occupied": True})
    monkeypatch.setenv("TMUX", "fixture")
    with pytest.raises(CanonicalScreeningError, match="already exists"):
        module._run_monitor(policy, paths, "smoke8")


def test_gpu_resource_recheck_rejects_uuid_registry_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "6" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    original_snapshot = {
        "authorized_gpu_registry": [
            {"physical_gpu_index": 0, "physical_gpu_uuid": _gpu_uuid(0)}
        ],
        "compute_processes": [],
        "gpus": [{"temperature_c": 40}],
    }
    admission_value = {
        "policy_sha256": policy["policy_sha256"],
        "snapshot": original_snapshot,
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "admission.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        **_bound(admission_path),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    raced_snapshot = json.loads(json.dumps(original_snapshot))
    raced_snapshot["authorized_gpu_registry"][0]["physical_gpu_uuid"] = _gpu_uuid(1)
    monkeypatch.setattr(
        module,
        "assert_resource_admission",
        lambda *_args, **_kwargs: raced_snapshot,
    )
    with pytest.raises(CanonicalScreeningError, match="differs"):
        module._write_gpu_resource_recheck(
            policy,
            paths,
            "screen512",
            admission,
            {
                "violated": False,
                "swap_consecutive_io": 0,
                "resource_window_sha256": "7" * 64,
            },
        )


def test_final_request_set_rejects_partial_intent_coverage(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    request_path = tmp_path / "request.json"
    write_exclusive_json(request_path, request)
    candidate = request["candidate"]
    base = {
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "checkpoint_model": candidate["checkpoint_model"],
        "mode": "smoke8",
        "sample_count": 8,
        "seed": 4549,
        "batch_size": 2,
        "admission_sha256": request["admission"]["canonical_sha256"],
    }
    intents = {
        "request_count": 2,
        "requests": [
            {
                **base,
                "candidate_id": candidate["candidate_id"],
                "replicate": "primary",
            },
            {
                **base,
                "candidate_id": "missing-candidate",
                "replicate": "repeat",
            },
        ],
    }
    with pytest.raises(CanonicalScreeningError, match="coverage"):
        module._validate_final_requests_against_intents(
            [request_path],
            intents,
            policy,
            request["controller_ready"],
            request["observer_ready"],
        )


def test_runtime_guard_first_sample_exposes_monitor_thread_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(
        module,
        "_cpu_times",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture thread failure")),
    )
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "guard.jsonl", tmp_path
    )
    guard.start()
    try:
        with pytest.raises(CanonicalScreeningError, match="thread failure"):
            guard.wait_first_sample(1.0)
    finally:
        summary = guard.stop()
    assert summary["thread_failure"]["type"] == "RuntimeError"


def test_worker_revalidates_ready_files_before_cuda(
    tmp_path: Path,
) -> None:
    policy, request = _run_fixture(tmp_path)
    controller_path = Path(request["controller_ready"]["path"])
    controller_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CanonicalScreeningError, match="file binding"):
        _assert_ready_barrier(request, policy)


def test_controller_raw_import_executes_no_policy_bound_module() -> None:
    root = Path(__file__).parents[1]
    controller = root / "scripts/run_canonical_checkpoint_screening.py"
    code = (
        "import importlib.util,json,sys;"
        f"p={str(controller)!r};"
        "s=importlib.util.spec_from_file_location('raw_controller',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(sorted(x for x in ("
        "'safa.closeout.canonical_screening',"
        "'safa.closeout.canonical_screening_worker') if x in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize(
    "implementation",
    (
        "checkpoint_preflight",
        "arcface_evaluator",
        "e0_loader",
        "canonical_quality",
        "screening_contracts",
        "screening_worker",
        "controller",
        "ram_probe_launcher",
        "preflight_wrapper",
        "gpu_wrapper",
        "generator_sampling",
        "meanflow_sampling",
        "latent_codec",
        "output_contract",
    ),
)
def test_controller_stdlib_bootstrap_rejects_each_implementation_tamper(
    tmp_path: Path, implementation: str
) -> None:
    module = _raw_controller_module()
    root = Path(__file__).parents[1]
    config = json.loads(
        (
            root / "configs/closeout/canonical_screening_512_v1.json"
        ).read_text(encoding="utf-8")
    )
    config["implementations"][implementation]["sha256"] = "0" * 64
    path = tmp_path / f"{implementation}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        module.ControllerBootstrapError,
        match="implementation digest differs",
    ):
        module._stdlib_validate_implementation_bindings(path)


def test_controller_tampered_worker_fails_before_dynamic_import(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    controller = root / "scripts/run_canonical_checkpoint_screening.py"
    config = json.loads(
        (
            root / "configs/closeout/canonical_screening_512_v1.json"
        ).read_text(encoding="utf-8")
    )
    config["implementations"]["screening_worker"]["sha256"] = "0" * 64
    config_path = tmp_path / "tampered-worker-policy.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    code = (
        "import importlib.util,json,sys;"
        f"p={str(controller)!r};c={str(config_path)!r};"
        "s=importlib.util.spec_from_file_location('raw_controller',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "\ntry:m._install_verified_contract_api(__import__('pathlib').Path(c))\n"
        "except m.ControllerBootstrapError:pass\n"
        "else:raise SystemExit(7)\n"
        "print(json.dumps(sorted(x for x in ("
        "'safa.closeout.canonical_screening',"
        "'safa.closeout.canonical_screening_worker') if x in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_real_worker_bootstrap_imports_no_heavy_modules() -> None:
    root = Path(__file__).parents[1]
    controller = (
        root
        / "scripts"
        / "run_canonical_checkpoint_screening.py"
    )
    policy_path = root / "configs/closeout/canonical_screening_512_v1.json"
    code = (
        "import importlib.util,json,sys;"
        f"p={str(controller)!r};"
        "s=importlib.util.spec_from_file_location('worker_bootstrap',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"m._install_verified_contract_api(__import__('pathlib').Path({str(policy_path)!r}),"
        "verify_historical_output_evidence=False);"
        "print(json.dumps(sorted(x for x in "
        "('torch','torchvision','onnxruntime','diffusers') if x in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_gpu_wrapper_records_sigkill_before_controller_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _gpu_wrapper_module()
    monkeypatch.setattr(
        wrapper,
        "_launch_observer",
        lambda **_kwargs: {
            "session": "fixture-monitor",
            "command": [
                "monitor",
                "--monitor-target",
                "screen512",
                "--execute",
            ],
            "launched_at": "2026-07-27T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        wrapper,
        "_wait_observer_terminal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fixture observer exited")
        ),
    )
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    policy_root = tmp_path / "campaign" / "by_policy" / ("8" * 64)
    value = wrapper.run_wrapped_controller(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy_sha256="8" * 64,
        config=config,
        campaign_root=tmp_path / "campaign",
        phase="screen512",
        python=sys.executable,
        command=[
            sys.executable,
            "-c",
            "import os,signal;os.kill(os.getpid(),signal.SIGKILL)",
        ],
    )
    assert value["exit_code"] == 137
    assert value["signal"] == 9
    assert value["controller_claim"] is None
    assert value["controller_terminal"] is None
    assert load_json(
        policy_root / "gpu_control/screen512/wrapper_exit.json",
        "GPU wrapper exit",
    ) == value
    terminal = load_json(
        policy_root / "gpu_control/screen512/wrapper_terminal.json",
        "GPU wrapper terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "ControllerExit"


def test_gpu_wrapper_preclaim_failure_still_writes_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _gpu_wrapper_module()
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    policy_root = tmp_path / "campaign" / "by_policy" / ("7" * 64)
    claim_path = (
        policy_root / "gpu_control/screen512/wrapper_claim.json"
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text("{}\n", encoding="utf-8")
    popen_calls = 0

    def forbidden_popen(*_args, **_kwargs):
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("controller Popen must not run")

    monkeypatch.setattr(wrapper.subprocess, "Popen", forbidden_popen)
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1, stderr="", stdout=""
        ),
    )
    with pytest.raises((FileExistsError, RuntimeError)):
        wrapper.run_wrapped_controller(
            repo_root=tmp_path,
            policy_root=policy_root,
            policy_sha256="7" * 64,
            config=config,
            campaign_root=tmp_path / "campaign",
            phase="screen512",
            python=sys.executable,
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )
    assert popen_calls == 0
    terminal = load_json(
        policy_root / "gpu_control/screen512/wrapper_terminal.json",
        "wrapper preclaim terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["wrapper_claim"] is not None


def test_gpu_wrapper_validates_observer_terminal_barrier(
    tmp_path: Path,
) -> None:
    wrapper = _gpu_wrapper_module()
    ready = {
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "policy_sha256": "9" * 64,
        "phase": "screen512",
    }
    ready["observer_ready_sha256"] = wrapper._canonical_digest(
        ready, "observer_ready_sha256"
    )
    ready_path = tmp_path / "observer_ready.json"
    wrapper._write_exclusive(ready_path, ready)
    terminal = {
        "contract_type": "safa_canonical_gpu_observer_terminal_v1",
        "policy_sha256": "9" * 64,
        "phase": "screen512",
        "status": "completed",
        "failure": None,
        "observer_ready": {
            "path": str(ready_path.resolve()),
            "sha256": wrapper._sha256_file(ready_path),
            "canonical_sha256": ready["observer_ready_sha256"],
        },
    }
    terminal["observer_terminal_sha256"] = wrapper._canonical_digest(
        terminal, "observer_terminal_sha256"
    )
    path = tmp_path / "observer_terminal.json"
    wrapper._write_exclusive(path, terminal)
    assert (
        wrapper._wait_observer_terminal(
            path, "9" * 64, "screen512", timeout_seconds=0.1
        )
        == terminal
    )
    failed = dict(terminal)
    failed["status"] = "failed"
    failed["failure"] = {"type": "RuntimeError", "message": "fixture"}
    failed["observer_terminal_sha256"] = wrapper._canonical_digest(
        failed, "observer_terminal_sha256"
    )
    failed_path = tmp_path / "failed_observer_terminal.json"
    wrapper._write_exclusive(failed_path, failed)
    with pytest.raises(RuntimeError, match="observer terminal contract"):
        wrapper._wait_observer_terminal(
            failed_path, "9" * 64, "screen512", timeout_seconds=0.1
        )


def test_gpu_tmux_command_uses_durable_wrapper_and_managed_observer(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    commands = module._tmux_commands(
        policy, policy_path, tmp_path / "campaign", "screen512"
    )
    assert "run_canonical_gpu_wrapper.py" in " ".join(commands["controller"])
    assert commands["monitor"] == []


def test_controller_rejects_observer_death_after_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    ready = load_json(Path(request["observer_ready"]["path"]), "observer ready")
    ready["admission"] = request["admission"]
    ready["observer_ready_sha256"] = canonical_digest(
        ready, "observer_ready_sha256"
    )
    ready_path = tmp_path / "observer_liveness_ready.json"
    write_exclusive_json(ready_path, ready)
    ready_binding = {
        **_bound(ready_path),
        "canonical_sha256": ready["observer_ready_sha256"],
    }
    admission_value = load_json(
        Path(request["admission"]["path"]), "liveness admission"
    )
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {
                "index": row["physical_gpu_index"],
                "uuid": row["physical_gpu_uuid"],
            }
            for row in admission_value["snapshot"]["authorized_gpu_registry"]
        ],
    )
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 1.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 2.0)
    monkeypatch.setattr(module, "_disk_percent", lambda *_args: 3.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    heartbeat = module._monitor_sample(
        policy,
        paths,
        "smoke8",
        terminal=False,
        admission=request["admission"],
    )
    assert "observed_at" in heartbeat
    assert "completed_at" not in heartbeat
    _write_jsonl(paths["logs"] / "smoke8__observer.jsonl", [heartbeat])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=1),
    )
    with pytest.raises(CanonicalScreeningError, match="observer tmux died"):
        module._assert_observer_live(
            policy, paths, "smoke8", ready_binding
        )


def test_controller_has_final_observer_liveness_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monitor_path = tmp_path / "monitor.jsonl"
    monitor_path.write_text("{}\n", encoding="utf-8")
    guard = types.SimpleNamespace(
        raise_if_violated=lambda: None,
        stop=lambda: {
            "violation_reason": None,
            "thread_failure": None,
        },
    )
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setenv("TMUX", "fixture")
    wrapper, observer_launch = _wrapper_bindings(
        tmp_path, policy, "screen512"
    )
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (wrapper, observer_launch),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "controller_claim.json",
            {
                "controller_claim_sha256": "b" * 64,
                "wrapper_claim": wrapper,
                "observer_launch": observer_launch,
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_gpu_ready_barrier",
        lambda *_args: {
            "admission_snapshot": {"authorized_gpu_registry": registry},
            "admission": {"canonical_sha256": "a" * 64},
            "requests": [],
            "resource_guard": guard,
            "monitor_path": monitor_path,
            "observer_ready": {"path": "fixture"},
            "controller_ready": {"path": "fixture"},
            "claim": {"controller_claim_sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(
        module, "_append_monitor_sample", lambda *_args, **_kwargs: monitor_path
    )
    monkeypatch.setattr(
        module, "_write_gpu_controller_terminal", lambda *_args, **_kwargs: None
    )
    calls = 0

    def check_observer(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CanonicalScreeningError("fixture final observer death")

    monkeypatch.setattr(module, "_assert_observer_live", check_observer)
    with pytest.raises(CanonicalScreeningError, match="final observer death"):
        module._run_gpu_phase(policy, policy_path, paths, "screen512")
    assert calls == 2


def test_final_release_gpu_pid_race_blocks_first_worker_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monitor_path = tmp_path / "monitor.jsonl"
    monitor_path.write_text("{}\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    guard = types.SimpleNamespace(
        raise_if_violated=lambda: None,
        stop=lambda: {
            "violation_reason": None,
            "thread_failure": None,
            "final_active_worker_pids": [],
        },
    )
    wrapper, launch = _wrapper_bindings(tmp_path, policy, "screen512")
    claim = {
        "controller_claim_sha256": "b" * 64,
        "wrapper_claim": wrapper,
        "observer_launch": launch,
    }
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (wrapper, launch),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "claim.json", claim
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_gpu_ready_barrier",
        lambda *_args: {
            "admission_snapshot": {"authorized_gpu_registry": registry},
            "admission": {"canonical_sha256": "a" * 64},
            "requests": [request_path],
            "resource_guard": guard,
            "monitor_path": monitor_path,
            "observer_ready": {"path": "observer"},
            "controller_ready": {"path": "controller"},
        },
    )
    monkeypatch.setattr(module, "_assert_observer_live", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_write_final_release_admission",
        lambda *_args: (_ for _ in ()).throw(
            CanonicalScreeningError(
                "unknown compute PID observed at final release"
            )
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker Popen must not run")
        ),
    )
    monkeypatch.setattr(
        module, "_append_monitor_sample", lambda *_args, **_kwargs: monitor_path
    )
    with pytest.raises(CanonicalScreeningError, match="unknown compute PID"):
        module._run_gpu_phase(policy, policy_path, paths, "screen512")
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "controller_terminal.json",
        "race terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "final_release_admission"


def test_release_ready_tamper_fails_before_artifact_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "1" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    ready_path = tmp_path / "tampered_ready.json"
    ready_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_validate_controller_ready",
        lambda *_args: (_ for _ in ()).throw(
            CanonicalScreeningError("release controller ready tampered")
        ),
    )
    with pytest.raises(CanonicalScreeningError, match="tampered"):
        module._write_final_release_admission(
            policy,
            paths,
            "screen512",
            {"canonical_sha256": "2" * 64},
            {"path": str(ready_path.resolve())},
            {"path": str(ready_path.resolve())},
            [],
            types.SimpleNamespace(raise_if_violated=lambda: None),
        )
    assert not (
        paths["gpu_control"]
        / "screen512"
        / "final_release_admission.json"
    ).exists()


def test_post_worker_summary_exception_writes_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monitor_path = tmp_path / "monitor.jsonl"
    monitor_path.write_text("{}\n", encoding="utf-8")
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    stop_summary = {
        "violation_reason": None,
        "thread_failure": None,
        "final_active_worker_pids": [],
    }
    guard = types.SimpleNamespace(
        raise_if_violated=lambda: None,
        stop=lambda: stop_summary,
    )
    wrapper, launch = _wrapper_bindings(tmp_path, policy, "screen512")
    claim = {
        "controller_claim_sha256": "b" * 64,
        "wrapper_claim": wrapper,
        "observer_launch": launch,
    }
    release = {
        "path": str((tmp_path / "release.json").resolve()),
        "sha256": "c" * 64,
        "canonical_sha256": "d" * 64,
    }
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (wrapper, launch),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "claim.json", claim
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_gpu_ready_barrier",
        lambda *_args: {
            "admission_snapshot": {"authorized_gpu_registry": registry},
            "admission": {"canonical_sha256": "a" * 64},
            "requests": [],
            "resource_guard": guard,
            "monitor_path": monitor_path,
            "observer_ready": {"path": "observer"},
            "controller_ready": {"path": "controller"},
        },
    )
    monkeypatch.setattr(module, "_assert_observer_live", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_write_final_release_admission",
        lambda *_args: ({}, release),
    )
    monkeypatch.setattr(
        module, "_append_monitor_sample", lambda *_args, **_kwargs: monitor_path
    )
    monkeypatch.setattr(
        module,
        "_build_gpu_completion_summary",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("summary injected")
        ),
    )
    with pytest.raises(RuntimeError, match="summary injected"):
        module._run_gpu_phase(policy, policy_path, paths, "screen512")
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "controller_terminal.json",
        "summary failure terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "completion_summary"
    assert "summary injected" in terminal["failure"]["message"]


def test_gpu_ready_barrier_positive_contract_chain(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy, manifest, manifest_path, policy_path, admission, _ = _manifest_fixture(
        tmp_path
    )
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    wrapper, observer_launch = _wrapper_bindings(
        tmp_path, policy, "smoke8"
    )
    claim, claim_path = module._write_gpu_controller_claim(
        policy, paths, "smoke8", wrapper, observer_launch
    )
    candidates = []
    template = manifest["candidates"][0]
    for index in range(193):
        candidates.append(
            {
                "candidate_id": f"candidate-{index:03d}",
                "checkpoint_sha256": f"{index:064x}",
                "checkpoint_model": template["checkpoint_model"],
            }
        )
    intent, intent_path = module._write_request_intent_manifest(
        policy,
        paths,
        "smoke8",
        ("primary", "repeat"),
        {
            "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
            "candidates": candidates,
        },
        admission,
    )

    def write_artifact(
        name: str, digest_field: str, value: dict
    ) -> tuple[dict, Path]:
        value[digest_field] = canonical_digest(value, digest_field)
        path = tmp_path / "barrier" / f"{name}.json"
        write_exclusive_json(path, value)
        return value, path

    internal, internal_path = write_artifact(
        "internal",
        "monitor_sample_sha256",
        {"kind": "internal", "policy_sha256": policy["policy_sha256"]},
    )
    guard, guard_path = write_artifact(
        "guard",
        "resource_window_sha256",
        {"kind": "guard", "policy_sha256": policy["policy_sha256"]},
    )
    recheck, recheck_path = write_artifact(
        "recheck",
        "resource_recheck_sha256",
        {"kind": "recheck", "policy_sha256": policy["policy_sha256"]},
    )
    controller, controller_path, controller_binding = (
        module._write_controller_ready(
            policy,
            paths,
            "smoke8",
            claim,
            admission,
            intent,
            intent_path,
            internal,
            internal_path,
            guard,
            guard_path,
            recheck,
            recheck_path,
            claim_path,
        )
    )
    assert (
        module._validate_controller_ready(
            controller, policy, "smoke8", admission
        )
        == controller
    )
    observer_claim, observer_claim_path = write_artifact(
        "observer_claim",
        "observer_claim_sha256",
        {"kind": "observer_claim", "policy_sha256": policy["policy_sha256"]},
    )
    observer_sample, observer_sample_path = write_artifact(
        "observer_sample",
        "monitor_sample_sha256",
        {"kind": "observer_sample", "policy_sha256": policy["policy_sha256"]},
    )
    observer = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": "smoke8",
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_ready_sha256": controller["controller_ready_sha256"],
        "observer_claim_sha256": observer_claim["observer_claim_sha256"],
        "wrapper_claim_sha256": wrapper["canonical_sha256"],
        "observer_launch_sha256": observer_launch["canonical_sha256"],
        "observer_claim": module._artifact_binding(
            observer_claim_path, observer_claim["observer_claim_sha256"]
        ),
        "wrapper_claim": wrapper,
        "observer_launch": observer_launch,
        "controller_ready": controller_binding,
        "admission": admission,
        "first_observer_sample": module._artifact_binding(
            observer_sample_path, observer_sample["monitor_sample_sha256"]
        ),
    }
    observer["observer_ready_sha256"] = canonical_digest(
        observer, "observer_ready_sha256"
    )
    observer_path = (
        paths["gpu_control"] / "smoke8" / "observer_ready.json"
    )
    write_exclusive_json(observer_path, observer)
    observer_binding = module._artifact_binding(
        observer_path, observer["observer_ready_sha256"]
    )
    assert (
        module._validate_observer_ready(
            observer, policy, "smoke8", controller, admission
        )
        == observer
    )
    request = build_run_request(
        policy,
        policy_path,
        manifest,
        manifest_path,
        manifest["candidates"][0],
        "smoke8",
        "primary",
        tmp_path / "runs",
        admission,
        controller_binding,
        observer_binding,
    )
    assert validate_run_request(request, policy) == request
    _assert_ready_barrier(request, policy)
    internal_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CanonicalScreeningError, match="file binding"):
        _assert_ready_barrier(request, policy)
