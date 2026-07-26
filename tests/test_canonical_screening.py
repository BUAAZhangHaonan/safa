from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
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
    load_json,
    ram_probe_admission_evidence_digest,
    ram_probe_contract_digest,
    ram_probe_execution_digest,
    validate_arcface_execution_probe_binding,
    validate_candidate_manifest,
    validate_checkpoint_plan,
    validate_preflight_result,
    validate_run_request,
    validate_run_result,
    validate_supersession_evidence,
    validate_policy,
    write_exclusive_json,
)
from safa.closeout.canonical_screening_worker import (
    _assert_runtime_cuda_binding,
    _load_arcface_contract,
    _load_source_pixel_batch,
    _representation_cosines,
    _write_validated_run_result,
)
from safa.closeout.canonical_quality import evaluate_locked_kid
from safa.closeout.generator_output_contract import (
    bind_output_contract,
    decoder_registry_digest,
    resolve_checkpoint_output_capability,
)


def _gpu_uuid(index: int) -> str:
    return f"GPU-0000000{index}-0000-0000-0000-00000000000{index}"


def _controller_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_checkpoint_screening.py"
    spec = importlib.util.spec_from_file_location("canonical_controller_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrapper_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_preflight_wrapper.py"
    spec = importlib.util.spec_from_file_location("canonical_wrapper_test", path)
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
            "slot_count": 8,
            "slot_budget_bytes": 1100,
            "reserved_bytes": 8800,
            "memory_total_bytes": 100000,
            "memory_used_bytes": 10000,
            "projected_used_bytes": 18800,
            "projected_used_percent": 18.8,
            "admission_limit_percent": 85,
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
    plan, policy, result_root = _complete_plan(
        tmp_path, [_row("candidate", "8" * 64)]
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
    )
    return policy, request


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


def _run_claim(
    policy: dict, request: dict, gpu_index: int = 0, worker_pid: int = 123
) -> dict:
    gpu_uuid = request["authorized_gpu_registry"][gpu_index][
        "physical_gpu_uuid"
    ]
    return build_run_claim(
        request,
        policy,
        gpu_index,
        gpu_uuid,
        gpu_uuid,
        gpu_uuid,
        worker_pid,
        "2026-07-26T00:00:00+00:00",
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
    module._cleanup_active_workers(active, pool)
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


def test_current_policy_binds_4d_preflight_successful_probe_and_arcface() -> None:
    root = Path(__file__).parents[1]
    policy_path = root / "configs/closeout/canonical_screening_512_v1.json"

    policy = validate_policy(root, policy_path)

    supersedes = policy["supersedes"]
    assert supersedes["policy_sha256"] == (
        "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
    )
    assert supersedes["classification"] == (
        "completed_preflight_and_successful_resource_only_probe_superseded"
    )
    assert supersedes["preflight"]["request_count"] == 193
    assert supersedes["preflight"]["result_count"] == 193
    assert supersedes["preflight"]["valid_count"] == 193
    assert supersedes["preflight"]["invalid_count"] == 0
    assert supersedes["preflight"]["reused_count"] == 0
    probe = supersedes["successful_ram_probe"]
    assert probe["classification"] == "successful_resource_measurement_only"
    assert probe["controller_terminal"] == "absent_by_contract"
    assert probe["resource_measurement_only"] is True
    assert probe["scientific_result_reuse"] == "forbidden"
    assert probe["retry_count"] == 0
    assert probe["artifact_seal"]["file_count"] == 28
    assert len(
        [
            name
            for name in probe["artifact_seal"]["files"]
            if name.endswith(".png")
        ]
    ) == 16
    assert supersedes["scientific_result_reuse"] == "forbidden"
    assert supersedes["successor_execution"] == "fresh_full_193_preflight"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("request_count", "counts differ"),
        ("reuse", "status differs"),
        ("root_digest", "evidence root differs"),
        ("preflight_path_alias", "file identity differs"),
        ("probe_root_digest", "artifact tree differs"),
        ("probe_file_sha", "file differs"),
        ("controller_terminal", "artifact tree differs"),
        ("resource_only", "probe semantics differ"),
    ],
)
def test_4d_supersession_tampering_fails_closed(
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
        supersedes["preflight"]["request_count"] = 192
    elif mutation == "reuse":
        supersedes["scientific_result_reuse"] = "allowed"
    elif mutation == "root_digest":
        supersedes["preflight"]["evidence_root"]["digest"] = "0" * 64
    elif mutation == "preflight_path_alias":
        supersedes["preflight"]["final_plan"]["path"] = supersedes[
            "preflight"
        ]["candidate_manifest"]["path"]
    elif mutation == "probe_root_digest":
        supersedes["successful_ram_probe"]["artifact_seal"]["root"][
            "digest"
        ] = "0" * 64
    elif mutation == "probe_file_sha":
        supersedes["successful_ram_probe"]["artifact_seal"]["files"][
            "admission.json"
        ] = "0" * 64
    elif mutation == "controller_terminal":
        supersedes["successful_ram_probe"]["artifact_seal"][
            "controller_terminal"
        ] = "present"
    elif mutation == "resource_only":
        supersedes["successful_ram_probe"]["resource_measurement_only"] = False
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
