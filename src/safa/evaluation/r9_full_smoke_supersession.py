from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from safa.evaluation.r9_evaluator_worker import (
    ProductionEvaluatorConfig,
    build_worker_request,
)
from safa.evaluation.r9_phase_results import (
    ArcFaceEvaluationRequest,
    QualityEvaluationRequest,
    SampleEvidence,
)

CAMPAIGN_ID = "r9-report-only-formal-v9"
WINNER_ARM_ID = "paper_eta_0p125"
FULL_SEED = 7919
MANIFEST_PATH = (
    "configs/medium_v2/experiments/r9_manifests/resource_smoke_8.jsonl"
)
MANIFEST_SHA256 = (
    "a5c1f71cb940135a53bcd5c75e3cff97c8a3ca659f6a0e33c448bfebf55ca2fb"
)
FAILED_ROOT = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9/evaluator_smoke"
)
SUPERSESSION_ROOT = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9/evaluator_smoke_supersessions/full-smoke-v2"
)
NATIVE_ROOT = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v8/generation_batch_benchmark/native__batch_2"
)
WINNER_ROOT = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v8/generation_batch_benchmark/"
    "paper_eta_0p125__batch_2"
)
V9_CONFIG = (
    "configs/medium_v2/experiments/r9_meanflow_full_continuation_campaign_v9.yaml"
)
V3_ROOT = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v8/confirm512_supersessions/report-only-v3/"
    "f4323db51df0c4980a3b8160bd741ec72aa45a4836bf3c1f4fde5ee0f86a83f0"
)
FAILED_EXECUTION_CLAIM_SHA256 = (
    "cd448bfa50c64472c8db87fa1951a2659aab5de01eecd55508c7f35aaee86c95"
)
FAILED_WORKER_STATUS = (
    f"{FAILED_ROOT}/r9-report-only-formal-v9/worker_status/"
    "9fceef6060adf6904a4808f96d9346debad4192d987a8eb66e36cca72a33b460.json"
)
FAILED_TERMINAL_OBSERVATION = (
    f"{FAILED_ROOT}/process_terminal_observation.json"
)
PREVIOUS_V2_SHA256 = (
    "a92d29bae05dec8e202cdd0c87878275e28e8e2e770498576e22c94c44a8db0f"
)


class FullSmokeSupersessionError(ValueError):
    pass


def build_full_smoke_supersession_contract(*, repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    failed = _bind_failed_v1(root)
    manifest = _file(root, MANIFEST_PATH, "full smoke manifest")
    if _sha256_file(manifest) != MANIFEST_SHA256:
        raise FullSmokeSupersessionError("full smoke manifest SHA256 changed")
    sample_ids = _manifest_ids(manifest)
    samples, evidence = _samples(root, sample_ids)
    del samples
    selection, selection_binding = _read_contract(
        root / V3_ROOT / "selection.json",
        "selection_sha256",
        "safa_r9_confirm512_report_only_selection_v3",
        root,
    )
    result, result_binding = _read_contract(
        root / V3_ROOT / "supersession_result.json",
        "supersession_result_sha256",
        None,
        root,
    )
    gate, gate_binding = _read_contract(
        root / V3_ROOT / "confirm512/gate_contract_v3.json",
        "gate_contract_sha256",
        "safa_r9_confirm512_report_only_gate_v3",
        root,
    )
    winner = _mapping(selection.get("winner"), "v3 winner")
    gate_sha = _sha(selection.get("gate_contract_sha256"), "selection gate SHA256")
    if (
        gate["gate_contract_sha256"] != gate_sha
        or
        result.get("gate_contract_sha256") != gate_sha
        or result.get("selection_sha256") != selection["selection_sha256"]
        or result.get("winner_arm_id") != WINNER_ARM_ID
        or winner.get("arm_id") != WINNER_ARM_ID
        or selection.get("reselection_allowed") is not False
    ):
        raise FullSmokeSupersessionError("v3 winner/gate evidence changed")
    config = _mapping(
        yaml.safe_load((root / V9_CONFIG).read_text(encoding="utf-8")), "v9 config"
    )
    base = _mapping(
        yaml.safe_load(
            (root / config["base_runtime"]["path"]).read_text(encoding="utf-8")
        ),
        "base runtime",
    )
    worker = _mapping(
        _mapping(base["evaluation"], "evaluation")["worker"], "worker"
    )
    worker_binding = {
        "path": worker["path"],
        "sha256": _sha256_file(_file(root, worker["path"], "worker wrapper")),
        "implementation_path": worker["implementation_path"],
        "implementation_sha256": _sha256_file(
            _file(root, worker["implementation_path"], "worker implementation")
        ),
    }
    payload = {
        "schema_version": 2,
        "contract_type": "safa_r9_full_smoke_supersession_v2",
        "campaign_id": CAMPAIGN_ID,
        "supersession_id": "full-smoke-v2",
        "failed_v1": failed,
        "prepared_v2_superseded": _bind_prepared_v2(root),
        "failure_reason": (
            "v1_started_arcface_with_legacy_r8_calibration_payload_"
            "instead_of_v9_full_winner_smoke"
        ),
        "source_selection": selection_binding,
        "source_supersession_result": result_binding,
        "source_gate": gate_binding,
        "winner": {
            "arm_id": WINNER_ARM_ID,
            "config_sha256": winner["config_sha256"],
            "source_generation_output_sha256": winner[
                "source_generation_output_sha256"
            ],
        },
        "manifest": {
            "path": MANIFEST_PATH,
            "sha256": MANIFEST_SHA256,
            "sample_count": 8,
            "ordered_sample_id_sha256": _digest(sample_ids),
        },
        "evidence": evidence,
        "worker": worker_binding,
        "request_policy": {
            "phase": "full",
            "arm_id": WINNER_ARM_ID,
            "matched_native": True,
            "sample_count": 8,
            "seed": FULL_SEED,
            "batch_size": 2,
            "tasks": ["arcface", "quality"],
            "same_sample_set_required": True,
            "retry_count": 0,
        },
        "execution": {
            "v1_arcface_execution_count": 1,
            "v1_quality_execution_count": 0,
            "v2_execution_count": 0,
        },
    }
    payload["smoke_supersession_sha256"] = _canonical(
        payload, "smoke_supersession_sha256"
    )
    return payload


def _bind_prepared_v2(root: Path) -> dict[str, Any]:
    previous = root / SUPERSESSION_ROOT / PREVIOUS_V2_SHA256
    expected = {
        "smoke_supersession_contract.json",
        "arcface/request.json",
        "arcface/request_claim.json",
        "quality/request.json",
        "quality/request_claim.json",
    }
    observed = {
        str(path.relative_to(previous))
        for path in previous.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise FullSmokeSupersessionError("prepared v2 inventory changed")
    contract, contract_binding = _read_contract(
        previous / "smoke_supersession_contract.json",
        "smoke_supersession_sha256",
        "safa_r9_full_smoke_supersession_v2",
        root,
    )
    if contract["smoke_supersession_sha256"] != PREVIOUS_V2_SHA256:
        raise FullSmokeSupersessionError("prepared v2 digest changed")
    files = {
        name: _binding(root, previous / name)
        for name in sorted(expected - {"smoke_supersession_contract.json"})
    }
    return {
        "classification": "prepared_zero_execution_superseded_before_launch",
        "contract": contract_binding,
        "files": files,
        "execution_count": 0,
    }


def smoke_namespace(*, repo_root: Path) -> Path:
    contract = build_full_smoke_supersession_contract(repo_root=repo_root)
    return (
        Path(repo_root).resolve()
        / SUPERSESSION_ROOT
        / contract["smoke_supersession_sha256"]
    )


def build_full_smoke_requests(
    *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    root = Path(repo_root).resolve()
    contract = build_full_smoke_supersession_contract(repo_root=root)
    namespace = root / SUPERSESSION_ROOT / contract["smoke_supersession_sha256"]
    manifest = root / MANIFEST_PATH
    sample_ids = _manifest_ids(manifest)
    samples, _ = _samples(root, sample_ids)
    config = _production_config(root, namespace)
    source_index = _mapping(
        yaml.safe_load(
            (root / "configs/medium_v2/experiments/r9_meanflow_campaign.yaml").read_text(
                encoding="utf-8"
            )
        )["evaluation"]["quality"]["real_index"],
        "source index",
    )
    source_index_path = _file(root, source_index["path"], "source index")
    native_root = root / NATIVE_ROOT
    winner_root = root / WINNER_ROOT
    requests: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    evaluators = {
        "arcface": ArcFaceEvaluationRequest(
            phase="full",
            logical_run_id="resource_smoke_arcface_full_8",
            arm_id=WINNER_ARM_ID,
            seed=FULL_SEED,
            source_index_path=source_index_path,
            source_index_sha256=source_index["sha256"],
            samples=samples,
        ),
        "quality": QualityEvaluationRequest(
            phase="full",
            logical_run_id="resource_smoke_quality_full_8",
            arm_id=WINNER_ARM_ID,
            seed=FULL_SEED,
            image_role="candidate",
            manifest_path=manifest,
            source_index_path=source_index_path,
            source_index_sha256=source_index["sha256"],
            samples=samples,
            algorithm_config_sha256=contract["winner"]["config_sha256"],
            runner_arm_config_sha256=contract["winner"]["config_sha256"],
            semantic_output_sha256=_digest(
                [
                    {"sample_id": s.sample_id, "candidate_sha256": s.candidate_sha256}
                    for s in samples
                ]
            ),
            evidence_binding_sha256=_digest(
                [
                    {
                        "sample_id": s.sample_id,
                        "source_sha256": s.source_sha256,
                        "native_sha256": s.native_sha256,
                        "candidate_sha256": s.candidate_sha256,
                    }
                    for s in samples
                ]
            ),
            generation_result_set_sha256=_digest(
                [
                    _sha256_file(native_root / "generation_result.json"),
                    _sha256_file(winner_root / "generation_result.json"),
                ]
            ),
            per_sample_set_sha256=_digest(
                [
                    _sha256_file(native_root / "per_sample.jsonl"),
                    _sha256_file(winner_root / "per_sample.jsonl"),
                ]
            ),
        ),
    }
    for task, evaluator in evaluators.items():
        request = build_worker_request(task, evaluator, config=config)
        _validate_v2_request(request, task=task, sample_ids=sample_ids)
        claim = {
            "schema_version": 2,
            "contract_type": "safa_r9_evaluator_resource_smoke_request_v2",
            "campaign_id": CAMPAIGN_ID,
            "kind": task,
            "sample_count": 8,
            "phase": "full",
            "arm_id": WINNER_ARM_ID,
            "batch_size": 2,
            "runtime_config": str(root / V9_CONFIG),
            "runtime_config_sha256": _sha256_file(root / V9_CONFIG),
            "manifest": contract["manifest"],
            "smoke_supersession_sha256": contract["smoke_supersession_sha256"],
            "evaluator_request_sha256": request["evaluator_request_sha256"],
            "worker_contract": contract["worker"],
            "arcface_contract_sha256": _digest(config.arcface),
            "quality_script_sha256": config.quality_script["sha256"],
            "retry_allowed": False,
        }
        claim["smoke_request_claim_sha256"] = _canonical(
            claim, "smoke_request_claim_sha256"
        )
        requests[task] = (request, claim)
    return contract, requests


def materialize_full_smoke_supersession(*, repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    contract, requests = build_full_smoke_requests(repo_root=root)
    namespace = root / SUPERSESSION_ROOT / contract["smoke_supersession_sha256"]
    _write_exclusive(
        namespace / "smoke_supersession_contract.json", _bytes(contract)
    )
    result: dict[str, Any] = {
        "namespace": str(namespace.relative_to(root)),
        "smoke_supersession_sha256": contract["smoke_supersession_sha256"],
        "execution_count": 0,
    }
    for task, (request, claim) in requests.items():
        task_root = namespace / task
        _write_exclusive(task_root / "request_claim.json", _bytes(claim))
        _write_exclusive(task_root / "request.json", _bytes(request))
        result[task] = {
            "request_sha256": request["evaluator_request_sha256"],
            "claim_sha256": claim["smoke_request_claim_sha256"],
        }
    return result


def validate_materialized_full_smoke(*, repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    contract, expected = build_full_smoke_requests(repo_root=root)
    namespace = root / SUPERSESSION_ROOT / contract["smoke_supersession_sha256"]
    observed, _ = _read_contract(
        namespace / "smoke_supersession_contract.json",
        "smoke_supersession_sha256",
        "safa_r9_full_smoke_supersession_v2",
        root,
    )
    if observed != contract:
        raise FullSmokeSupersessionError("materialized supersession changed")
    for task, (request, claim) in expected.items():
        actual_request, _ = _read_contract(
            namespace / task / "request.json",
            "evaluator_request_sha256",
            "safa_r9_phase_evaluator_request_v1",
            root,
        )
        actual_claim, _ = _read_contract(
            namespace / task / "request_claim.json",
            "smoke_request_claim_sha256",
            "safa_r9_evaluator_resource_smoke_request_v2",
            root,
        )
        if actual_request != request or actual_claim != claim:
            raise FullSmokeSupersessionError(f"{task} v2 request changed")
    return {"contract": contract, "namespace": namespace}


def _bind_failed_v1(root: Path) -> dict[str, Any]:
    arc = root / FAILED_ROOT / "arcface-full-v1"
    quality = root / FAILED_ROOT / "quality-full-v1"
    arc_files = {p.name for p in arc.iterdir() if p.is_file()}
    quality_files = {p.name for p in quality.iterdir() if p.is_file()}
    if arc_files != {
        "request.json",
        "request_claim.json",
        "execution_claim.json",
        "controller.log",
        "worker.log",
    }:
        raise FullSmokeSupersessionError("failed ArcFace v1 inventory changed")
    if quality_files != {"request.json", "request_claim.json"}:
        raise FullSmokeSupersessionError("unstarted quality v1 inventory changed")
    for name in ("worker_result.json", "resource_result.json"):
        if (arc / name).exists():
            raise FullSmokeSupersessionError(
                f"failed ArcFace v1 unexpectedly contains {name}"
            )
        if (quality / name).exists():
            raise FullSmokeSupersessionError(
                f"unstarted quality v1 unexpectedly contains {name}"
            )
    request, request_binding = _read_contract(
        arc / "request.json",
        "evaluator_request_sha256",
        "safa_r9_phase_evaluator_request_v1",
        root,
    )
    claim, claim_binding = _read_contract(
        arc / "request_claim.json",
        "smoke_request_claim_sha256",
        "safa_r9_evaluator_resource_smoke_request_v1",
        root,
    )
    execution, execution_binding = _read_contract(
        arc / "execution_claim.json",
        "execution_claim_sha256",
        "safa_r9_evaluator_resource_smoke_execution_v1",
        root,
    )
    worker_status_path = root / FAILED_WORKER_STATUS
    worker_status = _mapping(
        json.loads(worker_status_path.read_text(encoding="utf-8")),
        "failed worker status",
    )
    terminal_observation_path = root / FAILED_TERMINAL_OBSERVATION
    terminal_observation = _mapping(
        json.loads(terminal_observation_path.read_text(encoding="utf-8")),
        "failed process terminal observation",
    )
    quality_request, quality_request_binding = _read_contract(
        quality / "request.json",
        "evaluator_request_sha256",
        "safa_r9_phase_evaluator_request_v1",
        root,
    )
    quality_claim, quality_claim_binding = _read_contract(
        quality / "request_claim.json",
        "smoke_request_claim_sha256",
        "safa_r9_evaluator_resource_smoke_request_v1",
        root,
    )
    payload = _mapping(request.get("payload"), "failed request payload")
    samples = payload.get("samples")
    if (
        request.get("task") != "arcface"
        or payload.get("phase") != "calibrate"
        or payload.get("arm_id") != "paper_split_eta0.25"
        or not isinstance(samples, list)
        or len(samples) != 64
        or execution["execution_claim_sha256"] != FAILED_EXECUTION_CLAIM_SHA256
        or execution.get("retry_allowed") is not False
        or claim.get("retry_allowed") is not False
        or worker_status
        != {
            "campaign_id": CAMPAIGN_ID,
            "pid": 3615176,
            "process_start_ticks": 545202990,
            "schema_version": 1,
            "state": "running",
            "worker_id": "evaluator-smoke:arcface",
        }
        or terminal_observation
        != {
            "campaign_id": CAMPAIGN_ID,
            "host": "WS-4029GP-TRT",
            "observed_at_utc": "2026-07-26T09:05:30Z",
            "pid": 3615176,
            "proc_entry_present": False,
            "process_start_ticks": 545202990,
            "schema_version": 1,
            "worker_id": "evaluator-smoke:arcface",
        }
        or quality_request.get("task") != "quality"
        or quality_claim.get("retry_allowed") is not False
    ):
        raise FullSmokeSupersessionError("failed v1 classification changed")
    return {
        "classification": "invalid_legacy_calibration_payload_started_without_result",
        "request": request_binding,
        "request_claim": claim_binding,
        "execution_claim": execution_binding,
        "worker_status": _binding(root, worker_status_path),
        "terminal_observation": _binding(root, terminal_observation_path),
        "controller_log": _binding(root, arc / "controller.log"),
        "worker_log": _binding(root, arc / "worker.log"),
        "expected_absent_outputs": {
            "arcface_worker_result": str(
                (arc / "worker_result.json").relative_to(root)
            ),
            "arcface_resource_result": str(
                (arc / "resource_result.json").relative_to(root)
            ),
            "quality_worker_result": str(
                (quality / "worker_result.json").relative_to(root)
            ),
            "quality_resource_result": str(
                (quality / "resource_result.json").relative_to(root)
            ),
        },
        "quality_request": quality_request_binding,
        "quality_request_claim": quality_claim_binding,
        "quality_execution_started": False,
        "retry_allowed": False,
    }


def _validate_v2_request(
    request: Mapping[str, Any], *, task: str, sample_ids: list[str]
) -> None:
    payload = _mapping(request.get("payload"), "v2 payload")
    samples = payload.get("samples")
    observed = (
        [row.get("sample_id") for row in samples]
        if isinstance(samples, list)
        else []
    )
    config = _mapping(request.get("config"), "v2 config")
    if (
        payload.get("phase") != "full"
        or payload.get("arm_id") != WINNER_ARM_ID
        or payload.get("logical_run_id") != f"resource_smoke_{task}_full_8"
        or observed != sample_ids
        or config.get("batch_size") != 2
    ):
        raise FullSmokeSupersessionError(
            "v2 materializer rejects legacy calibration/arm/sample/batch fields"
        )


def _production_config(root: Path, namespace: Path) -> ProductionEvaluatorConfig:
    base = _mapping(
        yaml.safe_load(
            (root / "configs/medium_v2/experiments/r9_meanflow_campaign.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "base runtime",
    )
    evaluation = _mapping(base["evaluation"], "evaluation")
    worker = _mapping(evaluation["worker"], "worker")
    quality = _mapping(evaluation["quality"], "quality")
    quality_script = _mapping(quality["script"], "quality script")
    worker_contract = {
        "path": str(_file(root, worker["path"], "worker wrapper")),
        "sha256": _sha256_file(_file(root, worker["path"], "worker wrapper")),
        "implementation_path": str(
            _file(root, worker["implementation_path"], "worker implementation")
        ),
        "implementation_sha256": _sha256_file(
            _file(root, worker["implementation_path"], "worker implementation")
        ),
    }
    arcface = dict(_mapping(evaluation["arcface"], "arcface"))
    probe = _mapping(
        json.loads(_file(root, arcface["execution_probe"]["path"], "probe").read_text()),
        "probe",
    )
    arcface["execution"] = dict(_mapping(probe["execution"], "probe execution"))
    return ProductionEvaluatorConfig(
        repo_root=root,
        device="cuda:0",
        work_root=namespace / "work",
        quality_script={
            "path": str(_file(root, quality_script["path"], "quality script")),
            "sha256": _sha256_file(_file(root, quality_script["path"], "quality script")),
        },
        arcface=arcface,
        worker_contract=worker_contract,
        batch_size=2,
    )


def _samples(
    root: Path, sample_ids: list[str]
) -> tuple[tuple[SampleEvidence, ...], dict[str, Any]]:
    native_path = root / NATIVE_ROOT / "per_sample.jsonl"
    winner_path = root / WINNER_ROOT / "per_sample.jsonl"
    native_rows = _rows(native_path)
    winner_rows = _rows(winner_path)
    samples = []
    for sample_id in sample_ids:
        native = native_rows[sample_id]
        winner = winner_rows[sample_id]
        if native.get("source") != winner.get("source"):
            raise FullSmokeSupersessionError("matched source changed")
        source_path = Path(native["source"]).resolve()
        native_image = _file(root, native["native"], "matched native")
        candidate = _file(root, winner["generated"], "winner image")
        samples.append(
            SampleEvidence(
                sample_id=sample_id,
                source=source_path,
                native=native_image,
                candidate=candidate,
                source_sha256=_sha256_file(source_path),
                native_sha256=_sha256_file(native_image),
                candidate_sha256=_sha256_file(candidate),
            )
        )
    return tuple(samples), {
        "native_per_sample": _binding(root, native_path),
        "winner_per_sample": _binding(root, winner_path),
        "native_generation_result": _binding(
            root, root / NATIVE_ROOT / "generation_result.json"
        ),
        "winner_generation_result": _binding(
            root, root / WINNER_ROOT / "generation_result.json"
        ),
        "source_role": "v8_batch2_frozen_images_for_v9_full_resource_smoke_only",
    }


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = _mapping(json.loads(line), "per-sample row")
        rows[row["sample_id"]] = row
    return rows


def _manifest_ids(path: Path) -> list[str]:
    rows = [
        _mapping(json.loads(line), "manifest row")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != 8 or len(set(ids)) != 8 or any(not isinstance(x, str) for x in ids):
        raise FullSmokeSupersessionError("full smoke manifest is not fixed 8")
    return ids  # type: ignore[return-value]


def _read_contract(
    path: Path, field: str, contract_type: str | None, root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    value = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    if contract_type is not None and value.get("contract_type") != contract_type:
        raise FullSmokeSupersessionError(f"{path} contract type changed")
    if _sha(value.get(field), field) != _canonical(value, field):
        raise FullSmokeSupersessionError(f"{path} digest changed")
    return value, {
        "path": str(path.resolve().relative_to(root.resolve())),
        "file_sha256": _sha256_file(path),
        "contract_sha256": value[field],
    }


def _binding(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
    }


def _file(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FullSmokeSupersessionError(f"{label} is not a regular file")
    return resolved


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("exclusive contract write made no progress")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _canonical(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _digest(payload)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullSmokeSupersessionError(f"{label} must be a mapping")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise FullSmokeSupersessionError(f"{label} must be SHA256")
    return value
