#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

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


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG = REPO_ROOT / "configs/medium_v2/experiments/r9_meanflow_campaign.yaml"
CALIBRATION_MANIFEST = (
    REPO_ROOT / "configs/medium_v2/experiments/r9_manifests/calibration_64.jsonl"
)
R8_NATIVE_ROOT = (
    REPO_ROOT
    / "artifacts/r8_meanflow_flow_map_guidance/calibration/official_flow_map2_normalized_eta1"
)
R8_CANDIDATE_ROOT = (
    REPO_ROOT
    / "artifacts/r8_meanflow_flow_map_guidance/calibration/paper_split_eta0.25"
)


class SmokeRequestError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SmokeRequestError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in rows:
            raise SmokeRequestError(f"{path} has an invalid or duplicate sample ID")
        rows[sample_id] = row
    return rows


def _resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SmokeRequestError(f"{label} path is invalid")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if not resolved.is_file():
        raise SmokeRequestError(f"{label} is missing: {resolved}")
    return resolved


def _build_samples() -> tuple[SampleEvidence, ...]:
    manifest_rows = _read_jsonl(CALIBRATION_MANIFEST)
    sample_ids = [row.get("sample_id") for row in manifest_rows]
    if (
        len(sample_ids) != 64
        or any(
            not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids
        )
        or len(set(sample_ids)) != 64
    ):
        raise SmokeRequestError("calibration_64 must contain 64 unique ordered IDs")
    native_rows = _rows_by_id(R8_NATIVE_ROOT / "per_sample.jsonl")
    candidate_rows = _rows_by_id(R8_CANDIDATE_ROOT / "per_sample.jsonl")
    if set(native_rows) != set(sample_ids) or set(candidate_rows) != set(sample_ids):
        raise SmokeRequestError("R8 smoke assets do not exactly match calibration_64")
    samples = []
    for sample_id in sample_ids:
        assert isinstance(sample_id, str)
        native_row = native_rows[sample_id]
        candidate_row = candidate_rows[sample_id]
        source = _resolve_repo_path(native_row.get("source"), "R8 source")
        if Path(str(candidate_row.get("source"))).resolve() != source:
            raise SmokeRequestError("R8 native/candidate source binding mismatch")
        native = _resolve_repo_path(native_row.get("native"), "R8 native")
        candidate = _resolve_repo_path(candidate_row.get("generated"), "R8 candidate")
        samples.append(
            SampleEvidence(
                sample_id=sample_id,
                source=source,
                native=native,
                candidate=candidate,
                source_sha256=_sha256(source),
                native_sha256=_sha256(native),
                candidate_sha256=_sha256(candidate),
            )
        )
    return tuple(samples)


def _runtime() -> dict[str, Any]:
    value = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SmokeRequestError("R9 runtime config is not a mapping")
    return value


def build_request(
    kind: str, artifact_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if kind not in {"arcface", "quality"}:
        raise SmokeRequestError("evaluator smoke kind must be arcface or quality")
    runtime = _runtime()
    evaluation = runtime.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise SmokeRequestError("R9 evaluation contract is missing")
    worker = evaluation.get("worker")
    arcface = evaluation.get("arcface")
    heldout = evaluation.get("heldout")
    quality = evaluation.get("quality")
    if not all(
        isinstance(value, Mapping) for value in (worker, arcface, heldout, quality)
    ):
        raise SmokeRequestError("R9 evaluator runtime sections are incomplete")
    assert isinstance(worker, Mapping)
    assert isinstance(arcface, Mapping)
    assert isinstance(heldout, Mapping)
    assert isinstance(quality, Mapping)
    wrapper = _resolve_repo_path(worker.get("path"), "worker wrapper")
    implementation = _resolve_repo_path(
        worker.get("implementation_path"), "worker implementation"
    )
    worker_contract = {
        "path": str(wrapper),
        "sha256": str(worker.get("sha256")),
        "implementation_path": str(implementation),
        "implementation_sha256": str(worker.get("implementation_sha256")),
    }
    if _sha256(wrapper) != worker_contract["sha256"]:
        raise SmokeRequestError("worker wrapper digest mismatch")
    if _sha256(implementation) != worker_contract["implementation_sha256"]:
        raise SmokeRequestError("worker implementation digest mismatch")
    samples = _build_samples()
    source_index = _resolve_repo_path(
        quality.get("real_index", {}).get("path"), "quality source index"
    )
    declared_source_sha = quality.get("real_index", {}).get("sha256")
    if _sha256(source_index) != declared_source_sha:
        raise SmokeRequestError("quality source index digest mismatch")
    raw_arcface = dict(arcface)
    execution_probe = raw_arcface.get("execution_probe")
    if not isinstance(execution_probe, Mapping):
        raise SmokeRequestError("ArcFace execution probe provenance is missing")
    probe_path = _resolve_repo_path(
        execution_probe.get("path"), "ArcFace execution probe"
    )
    if _sha256(probe_path) != execution_probe.get("sha256"):
        raise SmokeRequestError("ArcFace execution probe digest mismatch")
    probe_payload = json.loads(probe_path.read_text(encoding="utf-8"))
    if not isinstance(probe_payload, dict) or not isinstance(
        probe_payload.get("execution"), Mapping
    ):
        raise SmokeRequestError("ArcFace execution probe omitted execution")
    raw_arcface["execution"] = dict(probe_payload["execution"])
    config = ProductionEvaluatorConfig(
        repo_root=REPO_ROOT,
        device="cuda:0",
        work_root=artifact_root / "work",
        arcface=raw_arcface,
        quality_script=dict(quality["script"]),
        worker_contract=worker_contract,
        batch_size=int(heldout.get("batch_size")),
    )
    if kind == "arcface":
        evaluator_request = ArcFaceEvaluationRequest(
            phase="calibrate",
            logical_run_id="resource_smoke_arcface_calibration_64",
            arm_id="paper_split_eta0.25",
            seed=1337,
            source_index_path=source_index,
            source_index_sha256=str(declared_source_sha),
            samples=samples,
        )
    else:
        native_generation = R8_NATIVE_ROOT / "generation_result.json"
        candidate_generation = R8_CANDIDATE_ROOT / "generation_result.json"
        native_per_sample = R8_NATIVE_ROOT / "per_sample.jsonl"
        candidate_per_sample = R8_CANDIDATE_ROOT / "per_sample.jsonl"
        evaluator_request = QualityEvaluationRequest(
            phase="calibrate",
            logical_run_id="resource_smoke_quality_calibration_64",
            arm_id="paper_split_eta0.25",
            seed=1337,
            image_role="candidate",
            manifest_path=CALIBRATION_MANIFEST,
            source_index_path=source_index,
            source_index_sha256=str(declared_source_sha),
            samples=samples,
            algorithm_config_sha256=_sha256(candidate_generation),
            runner_arm_config_sha256=_sha256(native_generation),
            semantic_output_sha256=_canonical_sha256(
                [
                    {
                        "sample_id": sample.sample_id,
                        "candidate_sha256": sample.candidate_sha256,
                    }
                    for sample in samples
                ]
            ),
            evidence_binding_sha256=_canonical_sha256(
                [
                    {
                        "sample_id": sample.sample_id,
                        "source_sha256": sample.source_sha256,
                        "native_sha256": sample.native_sha256,
                        "candidate_sha256": sample.candidate_sha256,
                    }
                    for sample in samples
                ]
            ),
            generation_result_set_sha256=_canonical_sha256(
                [_sha256(native_generation), _sha256(candidate_generation)]
            ),
            per_sample_set_sha256=_canonical_sha256(
                [_sha256(native_per_sample), _sha256(candidate_per_sample)]
            ),
        )
    request = build_worker_request(kind, evaluator_request, config=config)
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_evaluator_resource_smoke_request_v1",
        "kind": kind,
        "sample_count": len(samples),
        "runtime_config": str(RUNTIME_CONFIG),
        "runtime_config_sha256": _sha256(RUNTIME_CONFIG),
        "calibration_manifest": str(CALIBRATION_MANIFEST),
        "calibration_manifest_sha256": _sha256(CALIBRATION_MANIFEST),
        "source_index": str(source_index),
        "source_index_sha256": str(declared_source_sha),
        "native_per_sample_sha256": _sha256(R8_NATIVE_ROOT / "per_sample.jsonl"),
        "candidate_per_sample_sha256": _sha256(R8_CANDIDATE_ROOT / "per_sample.jsonl"),
        "worker_contract": worker_contract,
        "arcface_contract_sha256": _canonical_sha256(config.arcface),
        "quality_script_sha256": config.quality_script["sha256"],
        "evaluator_request_sha256": request["evaluator_request_sha256"],
        "retry_allowed": False,
    }
    claim["smoke_request_claim_sha256"] = _canonical_sha256(claim)
    return request, claim


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("arcface", "quality"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    request_path = artifact_root / "request.json"
    claim_path = artifact_root / "request_claim.json"
    if request_path.exists() or claim_path.exists():
        raise SmokeRequestError("evaluator smoke request already exists")
    request, claim = build_request(args.kind, artifact_root)
    _write_exclusive(claim_path, claim)
    _write_exclusive(request_path, request)
    print(json.dumps(claim, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
