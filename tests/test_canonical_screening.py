from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    build_candidate_manifest,
    build_checkpoint_plan,
    build_preflight_result,
    build_run_claim,
    build_run_request,
    build_run_result,
    canonical_digest,
    canonical_json,
    load_json,
    validate_candidate_manifest,
    validate_checkpoint_plan,
    validate_preflight_result,
    validate_run_request,
    validate_run_result,
    write_exclusive_json,
)
from safa.closeout.canonical_screening_worker import (
    _load_source_pixel_batch,
    _representation_cosines,
    _write_validated_run_result,
)
from safa.closeout.canonical_quality import evaluate_locked_kid


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(row) for row in rows))


def _bound(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"x")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _policy(tmp_path: Path, ledger: Path) -> tuple[dict, Path, dict]:
    bound = _bound(tmp_path / "bound.bin")
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
            "preflight_wrapper",
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
                "smoke8": {**bound, "sample_count": 8},
                "screen512": {**bound, "sample_count": 512},
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
        "resources": {"physical_gpus": [0, 1, 2, 3]},
        "implementations": implementations,
        "arcface": {"model_name": "buffalo_l"},
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"policy":"fixture"}\n', encoding="utf-8")
    admission_value = {
        "contract_type": "safa_canonical_resource_admission_v1",
        "policy_sha256": policy["policy_sha256"],
        "snapshot": {"gpus": []},
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


def _strict_preflight(sha: str, selector: str, status: str = "valid") -> dict:
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
            request["checkpoint_sha256"], request["checkpoint_model"]
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
        "snapshot": {"gpus": []},
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
    write_exclusive_json(results / f"{sha}__raw.json", _strict_preflight(sha, "raw"))
    with pytest.raises(CanonicalScreeningError, match="fields differ"):
        build_checkpoint_plan(tmp_path, policy, results)


def test_preflight_result_binds_request_policy_ledger_and_tool(tmp_path: Path) -> None:
    sha = "6" * 64
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", sha)])
    policy, _, _ = _policy(tmp_path, ledger)
    pending = build_checkpoint_plan(tmp_path, policy, tmp_path / "results")
    request = pending["preflight_requests"][0]
    envelope = build_preflight_result(request, policy, _strict_preflight(sha, "raw"))
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
        "pixel_image_size": 256,
        "pixel_protocol_config_sha256": policy["protocol"]["pixel_protocol_config"]["sha256"],
        "kid_subset_size": policy["protocol"]["kid_subset_sizes"][request["mode"]],
        "e0_mean": 0.8,
        "edev_mean": 0.7,
        "arcface": {"coverage": request["sample_count"]},
        "quality": {"kid_mean": 0.01},
        "per_sample_sha256": "c" * 64,
    }


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
    claim = build_run_claim(request, policy, 3, 123, "2026-07-26T00:00:00+00:00")
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


def test_screen512_gate_requires_exact_primary_repeat_smoke(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy, manifest, manifest_path, policy_path, admission, _ = _manifest_fixture(
        tmp_path
    )
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    candidate = manifest["candidates"][0]
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
        rows = [
            {
                "sample_id": f"s{index}",
                "candidate_sha256": hashlib.sha256(f"png{index}".encode()).hexdigest(),
            }
            for index in range(8)
        ]
        per_sample = output / "per_sample.jsonl"
        _write_jsonl(per_sample, rows)
        claim = build_run_claim(
            request,
            policy,
            0,
            100 + (replicate == "repeat"),
            "2026-07-26T00:00:00+00:00",
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
    module._require_smoke_success(policy, manifest, paths)
    repeat_rows = paths["runs"] / "smoke8_repeat" / candidate["candidate_id"] / "per_sample.jsonl"
    rows = [json.loads(line) for line in repeat_rows.read_text(encoding="utf-8").splitlines()]
    rows[0]["candidate_sha256"] = "f" * 64
    _write_jsonl(repeat_rows, rows)
    with pytest.raises(CanonicalScreeningError, match="per-sample digest mismatch"):
        module._require_smoke_success(policy, manifest, paths)


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
    claim = build_run_claim(request, policy, 0, 123, "2026-07-26T00:00:00+00:00")
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
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [{"index": 0, "uuid": "GPU-a", "temperature_c": 40}],
    )
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    path = module._append_monitor_sample(policy, paths, "smoke8")
    module._append_monitor_sample(policy, paths, "smoke8", terminal=True)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["terminal"] is False and rows[1]["terminal"] is True
    assert rows[0]["gpus"][0]["uuid"] == "GPU-a"
    assert rows[0]["artifacts"]["generated_png"] == 0


def test_cpu_admission_never_depends_on_gpu_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    policy["resources"].update({
        "cpu_admission_percent": 85,
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
    with pytest.raises(RuntimeError, match="injected"):
        module.materialize_preflights(policy, paths)
    attempts = paths["preflight_control"] / "attempts"
    claim = load_json(next(attempts.glob("*.claim.json")), "attempt claim")
    terminal = load_json(next(attempts.glob("*.terminal.json")), "attempt terminal")
    assert terminal["attempt_claim_sha256"] == claim["attempt_claim_sha256"]
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "RuntimeError"
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
    monkeypatch.setattr(
        module,
        "materialize_preflights",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("controller injected")),
    )
    with pytest.raises(RuntimeError, match="controller injected"):
        module._execute_preflight_controller(policy, paths)
    control = paths["preflight_control"]
    terminal = load_json(control / "controller_terminal.json", "controller terminal")
    assert terminal["status"] == "failed"
    assert terminal["failure"]["message"] == "controller injected"
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
    with pytest.raises(CanonicalScreeningError, match="refuses result reuse"):
        module.materialize_preflights(policy, paths)


def test_write_exclusive_json_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    write_exclusive_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_exclusive_json(path, {"value": 2})
