from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from safa.evaluation.r9_campaign_contracts import validate_gate_contract
from safa.evaluation.r9_evaluator_worker import (
    ProductionEvaluatorConfig,
    build_worker_request,
)
from safa.evaluation.r9_phase_results import (
    ArcFaceEvaluationRequest,
    QualityEvaluationRequest,
    SampleEvidence,
)


R9_CONTINUATION_PARENT_CAMPAIGN_ID = "r9-report-only-formal-v2"
R9_CONTINUATION_PARENT_GATE_SHA256 = (
    "748ff3a78157db3cce0c5161dc8a209d204a6f356fa19325b7e92a01e40d5cef"
)
R9_CONTINUATION_PARENT_PHASE_RESULTS_SHA256 = (
    "54bc6ba437986aedc695016c53cb2a82d69c545f443ad4c068502dedab4c882f"
)
R9_CONTINUATION_REQUEST_PATH = (
    "configs/medium_v2/experiments/r9_meanflow_continuation_campaign.yaml"
)
R9_CONTINUATION_BASE_RUNTIME_PATH = (
    "configs/medium_v2/experiments/r9_meanflow_campaign.yaml"
)
R9_CONTINUATION_SOURCE_FIELDS = frozenset(
    {
        "parent_campaign_id",
        "diagnose_gate_contract_sha256",
        "diagnose_phase_results_sha256",
    }
)
R9_CONTINUATION_MANIFESTS = frozenset(
    {
        "calibration_64",
        "validate_512",
        "full_2048",
        "full_visual_64",
        "arcface_clean_pool",
    }
)
R9_CONTINUATION_CALIBRATION_MANIFEST = (
    "configs/medium_v2/experiments/r9_manifests/calibration_64.jsonl"
)
R9_CONTINUATION_SMOKE_NATIVE_ROOT = (
    "artifacts/r8_meanflow_flow_map_guidance/calibration/"
    "official_flow_map2_normalized_eta1"
)
R9_CONTINUATION_SMOKE_CANDIDATE_ROOT = (
    "artifacts/r8_meanflow_flow_map_guidance/calibration/paper_split_eta0.25"
)
_CID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class ContinuationContractError(ValueError):
    """Raised when the sealed parent-to-child evidence chain is invalid."""


def normalize_continuation_source(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != R9_CONTINUATION_SOURCE_FIELDS:
        raise ContinuationContractError(
            "continuation source must contain only parent CID and expected A digests"
        )
    normalized = {
        "parent_campaign_id": _nonempty(
            value.get("parent_campaign_id"), "parent campaign ID"
        ),
        "diagnose_gate_contract_sha256": _sha(
            value.get("diagnose_gate_contract_sha256"), "parent A gate SHA256"
        ),
        "diagnose_phase_results_sha256": _sha(
            value.get("diagnose_phase_results_sha256"),
            "parent A phase-results SHA256",
        ),
    }
    expected = {
        "parent_campaign_id": R9_CONTINUATION_PARENT_CAMPAIGN_ID,
        "diagnose_gate_contract_sha256": R9_CONTINUATION_PARENT_GATE_SHA256,
        "diagnose_phase_results_sha256": (
            R9_CONTINUATION_PARENT_PHASE_RESULTS_SHA256
        ),
    }
    if normalized != expected:
        raise ContinuationContractError("continuation source is not the sealed R9 v2 A gate")
    return normalized


def build_continuation_contract(
    *,
    repo_root: Path,
    child_campaign_id: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    child = _nonempty(child_campaign_id, "child campaign ID")
    if _CID.fullmatch(child) is None or child == R9_CONTINUATION_PARENT_CAMPAIGN_ID:
        raise ContinuationContractError("child campaign ID must be a distinct slug")
    locked_source = normalize_continuation_source(source)
    parent_root = root / (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        + locked_source["parent_campaign_id"]
    )
    runtime_path = parent_root / "campaign_runtime.json"
    phase_path = parent_root / "diagnose/phase_results.json"
    gate_path = parent_root / "diagnose/gate_contract.json"
    runtime = _read_contract(runtime_path, "campaign_runtime_sha256")
    phase = _read_contract(phase_path, "phase_results_sha256")
    gate = validate_gate_contract(_read_json(gate_path))
    if runtime.get("campaign_id") != locked_source["parent_campaign_id"]:
        raise ContinuationContractError("parent runtime campaign ID mismatch")
    if phase.get("phase") != "diagnose" or gate.get("phase") != "diagnose":
        raise ContinuationContractError("parent evidence is not the diagnose phase")
    if gate["gate_contract_sha256"] != locked_source[
        "diagnose_gate_contract_sha256"
    ]:
        raise ContinuationContractError("parent A gate contract SHA256 mismatch")
    if phase["phase_results_sha256"] != locked_source[
        "diagnose_phase_results_sha256"
    ]:
        raise ContinuationContractError("parent A phase-results SHA256 mismatch")
    runtime_sha = _sha(runtime.get("campaign_runtime_sha256"), "parent runtime SHA256")
    gate_context = _mapping(gate.get("context"), "parent gate context")
    for actual, expected, label in (
        (phase.get("campaign_runtime_sha256"), runtime_sha, "phase runtime"),
        (gate_context.get("campaign_runtime_sha256"), runtime_sha, "gate runtime"),
        (
            gate_context.get("phase_results_sha256"),
            phase["phase_results_sha256"],
            "gate phase-results",
        ),
    ):
        if actual != expected:
            raise ContinuationContractError(f"parent {label} binding mismatch")
    selected_ids = gate.get("selected_arm_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != 3:
        raise ContinuationContractError("parent A gate must select exactly three arms")
    arm_by_id = {
        row.get("arm_id"): row
        for row in gate.get("arms", [])
        if isinstance(row, Mapping)
    }
    selected_arms = []
    for arm_id in selected_ids:
        row = arm_by_id.get(arm_id)
        if row is None or row.get("passed") is not True:
            raise ContinuationContractError("parent selected arm is missing or failed")
        selected_arms.append(
            {
                "arm_id": _nonempty(row.get("arm_id"), "selected arm ID"),
                "family": _nonempty(row.get("family"), "selected arm family"),
                "config_sha256": _sha(
                    row.get("config_sha256"), "selected arm config SHA256"
                ),
                "output_sha256": _sha(
                    row.get("output_sha256"), "selected arm output SHA256"
                ),
            }
        )
    if {row["family"] for row in selected_arms} != {
        "flow_map2",
        "paper_split_constant",
        "paper_split_interval_ablation",
    }:
        raise ContinuationContractError("parent A selections do not cover three families")
    manifests = _mapping(runtime.get("manifests"), "parent manifests")
    if set(manifests) != R9_CONTINUATION_MANIFESTS:
        raise ContinuationContractError("parent runtime must bind exactly five manifests")
    normalized_manifests = {
        name: _verified_file_binding(root, manifests[name], f"manifest {name}")
        for name in sorted(R9_CONTINUATION_MANIFESTS)
    }
    checkpoint = _verified_file_binding(
        root, _mapping(runtime.get("checkpoint"), "checkpoint"), "checkpoint"
    )
    evaluation = _mapping(runtime.get("evaluation"), "parent evaluation")
    worker = _mapping(evaluation.get("worker"), "parent evaluator worker")
    quality = _mapping(
        _mapping(evaluation.get("quality"), "parent quality").get("script"),
        "parent quality script",
    )
    implementations = {
        "continuation_request": _actual_file_binding(
            root, R9_CONTINUATION_REQUEST_PATH
        ),
        "base_runtime_template": _actual_file_binding(
            root, R9_CONTINUATION_BASE_RUNTIME_PATH
        ),
        "driver": _actual_file_binding(root, "scripts/run_r9_meanflow_campaign.py"),
        "evaluator_entrypoint": _actual_file_binding(root, worker["path"]),
        "evaluator_implementation": _actual_file_binding(
            root, worker["implementation_path"]
        ),
        "quality": _actual_file_binding(root, quality["path"]),
    }
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_continuation_contract_v1",
        "child_campaign_id": child,
        "source": locked_source,
        "parent": {
            "campaign_id": runtime["campaign_id"],
            "runtime": _contract_binding(root, runtime_path, runtime_sha),
            "diagnose_gate": _contract_binding(
                root, gate_path, gate["gate_contract_sha256"]
            ),
            "diagnose_phase_results": _contract_binding(
                root, phase_path, phase["phase_results_sha256"]
            ),
        },
        "selected_arms": selected_arms,
        "bindings": {
            "manifest_contracts_sha256": _sha(
                runtime.get("manifest_contracts_sha256"),
                "parent manifest contracts SHA256",
            ),
            "manifests": normalized_manifests,
            "checkpoint": checkpoint,
            "determinism_policy_sha256": _sha(
                runtime.get("determinism_policy_sha256"), "determinism policy SHA256"
            ),
            "attention_backend": _nonempty(
                runtime.get("attention_backend"), "attention backend"
            ),
            "schedule": _verified_contract_binding(
                root, runtime.get("schedule"), "schedule"
            ),
            "semigroup_gate": _verified_contract_binding(
                root, runtime.get("semigroup_gate"), "semigroup gate"
            ),
            "implementations": implementations,
        },
    }
    if payload["bindings"]["attention_backend"] != "native":
        raise ContinuationContractError("continuation requires native attention")
    payload["continuation_contract_sha256"] = _canonical_digest(
        payload, "continuation_contract_sha256"
    )
    return payload


def validate_continuation_contract(
    value: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    normalized = _mapping(value, "continuation contract")
    if normalized.get("contract_type") == "safa_r9_confirm_continuation_contract_v1":
        from safa.evaluation.r9_confirm_continuation_contracts import (
            validate_confirm_continuation_contract,
        )

        return validate_confirm_continuation_contract(
            normalized, repo_root=repo_root
        )
    declared = _sha(
        normalized.get("continuation_contract_sha256"), "continuation contract SHA256"
    )
    if declared != _canonical_digest(normalized, "continuation_contract_sha256"):
        raise ContinuationContractError("continuation contract canonical digest mismatch")
    source = normalize_continuation_source(normalized.get("source"))
    expected = build_continuation_contract(
        repo_root=repo_root,
        child_campaign_id=_nonempty(
            normalized.get("child_campaign_id"), "child campaign ID"
        ),
        source=source,
    )
    if normalized != expected:
        raise ContinuationContractError("continuation contract disagrees with live bindings")
    return normalized


def materialize_continuation_contract(
    *, repo_root: Path, child_campaign_id: str, source: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = build_continuation_contract(
        repo_root=repo_root, child_campaign_id=child_campaign_id, source=source
    )
    path, content, binding = continuation_contract_binding(
        payload, repo_root=repo_root
    )
    _write_exclusive(path, content)
    return payload, binding


def continuation_contract_binding(
    payload: Mapping[str, Any], *, repo_root: Path
) -> tuple[Path, bytes, dict[str, str]]:
    normalized = _mapping(payload, "continuation contract")
    if normalized.get("contract_type") == "safa_r9_confirm_continuation_contract_v1":
        from safa.evaluation.r9_confirm_continuation_contracts import (
            confirm_continuation_contract_binding,
        )

        return confirm_continuation_contract_binding(
            normalized, repo_root=repo_root
        )
    child_campaign_id = _nonempty(
        normalized.get("child_campaign_id"), "child campaign ID"
    )
    declared = _sha(
        normalized.get("continuation_contract_sha256"), "continuation contract SHA256"
    )
    if declared != _canonical_digest(normalized, "continuation_contract_sha256"):
        raise ContinuationContractError("continuation contract canonical digest mismatch")
    root = Path(repo_root).resolve()
    path = (
        root
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / child_campaign_id
        / "continuation_contract.json"
    )
    content = (
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    return path, content, {
        "path": str(path.relative_to(root)),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "contract_sha256": declared,
    }


def materialize_continuation_evaluator_smoke_requests(
    *,
    repo_root: Path,
    child_campaign_id: str,
    runtime_config_path: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    child = _nonempty(child_campaign_id, "child campaign ID")
    if _CID.fullmatch(child) is None:
        raise ContinuationContractError("child campaign ID must be a slug")
    evaluation = _mapping(runtime.get("evaluation"), "evaluation")
    resources = _mapping(
        evaluation.get("resource_smokes"), "evaluator resource smokes"
    )
    if set(resources) != {"arcface", "quality", "heldout"}:
        raise ContinuationContractError(
            "child evaluator resource declarations are not canonical"
        )
    expected_modes = {
        "arcface": "measured_single_worker",
        "quality": "measured_exclusive_bootstrap",
    }
    materialized: dict[str, Any] = {}
    for kind, mode in expected_modes.items():
        declaration = _mapping(resources.get(kind), f"{kind} resource declaration")
        if set(declaration) != {"mode", "artifact_root"}:
            raise ContinuationContractError(
                f"{kind} resource declaration is not canonical"
            )
        if declaration.get("mode") != mode:
            raise ContinuationContractError(f"{kind} resource mode mismatch")
        artifact_root = _child_smoke_root(
            root,
            declaration.get("artifact_root"),
            child_campaign_id=child,
            kind=kind,
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        request, claim = build_continuation_evaluator_smoke_request(
            kind=kind,
            repo_root=root,
            artifact_root=artifact_root,
            runtime_config_path=runtime_config_path,
            runtime=runtime,
        )
        request_content = _json_contract_bytes(request)
        claim_content = _json_contract_bytes(claim)
        _write_exclusive(artifact_root / "request_claim.json", claim_content)
        _write_exclusive(artifact_root / "request.json", request_content)
        materialized[kind] = {
            "artifact_root": str(artifact_root.relative_to(root)),
            "request": {
                "path": str((artifact_root / "request.json").relative_to(root)),
                "file_sha256": hashlib.sha256(request_content).hexdigest(),
                "contract_sha256": request["evaluator_request_sha256"],
            },
            "request_claim": {
                "path": str(
                    (artifact_root / "request_claim.json").relative_to(root)
                ),
                "file_sha256": hashlib.sha256(claim_content).hexdigest(),
                "contract_sha256": claim["smoke_request_claim_sha256"],
            },
        }
    heldout = _mapping(resources.get("heldout"), "heldout resource declaration")
    expected_heldout = {
        "mode": "exclusive_single_official_run",
        "smoke_execution": "sealed_until_winner_lock",
        "global_exclusive_slots": 16,
        "ram_admission_percent": 85,
        "ram_hard_limit_percent": 90,
    }
    if heldout != expected_heldout:
        raise ContinuationContractError("child heldout evaluator must remain sealed")
    materialized["heldout"] = heldout
    materialized["request_set_sha256"] = _canonical_value_digest(materialized)
    return materialized


def build_continuation_evaluator_smoke_request(
    *,
    kind: str,
    repo_root: Path,
    artifact_root: Path,
    runtime_config_path: Path,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if kind not in {"arcface", "quality"}:
        raise ContinuationContractError(
            "evaluator smoke kind must be arcface or quality"
        )
    root = Path(repo_root).resolve()
    runtime_path = _contained(
        root, str(runtime_config_path), "continuation request"
    )
    evaluation = _mapping(runtime.get("evaluation"), "evaluation")
    worker = _mapping(evaluation.get("worker"), "evaluator worker")
    quality = _mapping(evaluation.get("quality"), "quality evaluator")
    arcface = _mapping(evaluation.get("arcface"), "ArcFace evaluator")
    heldout = _mapping(evaluation.get("heldout"), "heldout evaluator")
    wrapper = _contained(root, worker.get("path"), "worker wrapper")
    implementation = _contained(
        root, worker.get("implementation_path"), "worker implementation"
    )
    worker_contract = {
        "path": str(wrapper),
        "sha256": _verified_current_sha(
            wrapper, worker.get("sha256"), "worker wrapper"
        ),
        "implementation_path": str(implementation),
        "implementation_sha256": _verified_current_sha(
            implementation,
            worker.get("implementation_sha256"),
            "worker implementation",
        ),
    }
    quality_script = _mapping(quality.get("script"), "quality script")
    quality_path = _contained(root, quality_script.get("path"), "quality script")
    quality_binding = {
        "path": str(quality_path),
        "sha256": _verified_current_sha(
            quality_path, quality_script.get("sha256"), "quality script"
        ),
    }
    raw_arcface = dict(arcface)
    probe_binding = _mapping(
        raw_arcface.get("execution_probe"), "ArcFace execution probe"
    )
    probe_path = _contained(
        root, probe_binding.get("path"), "ArcFace execution probe"
    )
    _verified_current_sha(
        probe_path, probe_binding.get("sha256"), "ArcFace execution probe"
    )
    probe = _read_json(probe_path)
    execution = probe.get("execution")
    if not isinstance(execution, Mapping):
        raise ContinuationContractError(
            "ArcFace execution probe omitted execution"
        )
    raw_arcface["execution"] = dict(execution)
    config = ProductionEvaluatorConfig(
        repo_root=root,
        device="cuda:0",
        work_root=artifact_root / "work",
        quality_script=quality_binding,
        arcface=raw_arcface,
        worker_contract=worker_contract,
        batch_size=int(heldout.get("batch_size")),
    )
    samples = _continuation_smoke_samples(root)
    source_index_binding = _mapping(quality.get("real_index"), "quality real index")
    source_index = _contained(
        root, source_index_binding.get("path"), "quality real index"
    )
    source_index_sha = _verified_current_sha(
        source_index,
        source_index_binding.get("sha256"),
        "quality real index",
    )
    calibration_manifest = _contained(
        root, R9_CONTINUATION_CALIBRATION_MANIFEST, "calibration manifest"
    )
    native_root = root / R9_CONTINUATION_SMOKE_NATIVE_ROOT
    candidate_root = root / R9_CONTINUATION_SMOKE_CANDIDATE_ROOT
    if kind == "arcface":
        evaluator_request: Any = ArcFaceEvaluationRequest(
            phase="calibrate",
            logical_run_id="resource_smoke_arcface_calibration_64",
            arm_id="paper_split_eta0.25",
            seed=1337,
            source_index_path=source_index,
            source_index_sha256=source_index_sha,
            samples=samples,
        )
    else:
        native_generation = _regular_file(
            native_root / "generation_result.json", "native generation result"
        )
        candidate_generation = _regular_file(
            candidate_root / "generation_result.json",
            "candidate generation result",
        )
        native_per_sample = _regular_file(
            native_root / "per_sample.jsonl", "native per-sample evidence"
        )
        candidate_per_sample = _regular_file(
            candidate_root / "per_sample.jsonl", "candidate per-sample evidence"
        )
        evaluator_request = QualityEvaluationRequest(
            phase="calibrate",
            logical_run_id="resource_smoke_quality_calibration_64",
            arm_id="paper_split_eta0.25",
            seed=1337,
            image_role="candidate",
            manifest_path=calibration_manifest,
            source_index_path=source_index,
            source_index_sha256=source_index_sha,
            samples=samples,
            algorithm_config_sha256=_file_sha256(candidate_generation),
            runner_arm_config_sha256=_file_sha256(native_generation),
            semantic_output_sha256=_canonical_value_digest(
                [
                    {
                        "sample_id": sample.sample_id,
                        "candidate_sha256": sample.candidate_sha256,
                    }
                    for sample in samples
                ]
            ),
            evidence_binding_sha256=_canonical_value_digest(
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
            generation_result_set_sha256=_canonical_value_digest(
                [
                    _file_sha256(native_generation),
                    _file_sha256(candidate_generation),
                ]
            ),
            per_sample_set_sha256=_canonical_value_digest(
                [
                    _file_sha256(native_per_sample),
                    _file_sha256(candidate_per_sample),
                ]
            ),
        )
    request = build_worker_request(kind, evaluator_request, config=config)
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_evaluator_resource_smoke_request_v1",
        "kind": kind,
        "sample_count": len(samples),
        "runtime_config": str(runtime_path),
        "runtime_config_sha256": _file_sha256(runtime_path),
        "calibration_manifest": str(calibration_manifest),
        "calibration_manifest_sha256": _file_sha256(calibration_manifest),
        "source_index": str(source_index),
        "source_index_sha256": source_index_sha,
        "native_per_sample_sha256": _file_sha256(
            native_root / "per_sample.jsonl"
        ),
        "candidate_per_sample_sha256": _file_sha256(
            candidate_root / "per_sample.jsonl"
        ),
        "worker_contract": worker_contract,
        "arcface_contract_sha256": _canonical_value_digest(config.arcface),
        "quality_script_sha256": quality_binding["sha256"],
        "evaluator_request_sha256": request["evaluator_request_sha256"],
        "retry_allowed": False,
    }
    claim["smoke_request_claim_sha256"] = _canonical_value_digest(claim)
    return request, claim


def _continuation_smoke_samples(root: Path) -> tuple[SampleEvidence, ...]:
    manifest = _contained(
        root, R9_CONTINUATION_CALIBRATION_MANIFEST, "calibration manifest"
    )
    manifest_rows = _read_jsonl(manifest)
    sample_ids = [row.get("sample_id") for row in manifest_rows]
    if (
        len(sample_ids) != 64
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != 64
    ):
        raise ContinuationContractError(
            "calibration manifest must contain 64 unique ordered IDs"
        )
    native_root = root / R9_CONTINUATION_SMOKE_NATIVE_ROOT
    candidate_root = root / R9_CONTINUATION_SMOKE_CANDIDATE_ROOT
    native_rows = _rows_by_id(native_root / "per_sample.jsonl")
    candidate_rows = _rows_by_id(candidate_root / "per_sample.jsonl")
    if set(native_rows) != set(sample_ids) or set(candidate_rows) != set(sample_ids):
        raise ContinuationContractError(
            "R8 evaluator smoke evidence does not match calibration_64"
        )
    samples = []
    for sample_id in sample_ids:
        assert isinstance(sample_id, str)
        native_row = native_rows[sample_id]
        candidate_row = candidate_rows[sample_id]
        source = _contained_evidence_file(
            root, native_row.get("source"), "smoke source"
        )
        candidate_source = _contained_evidence_file(
            root, candidate_row.get("source"), "candidate smoke source"
        )
        if candidate_source != source:
            raise ContinuationContractError(
                "native/candidate smoke source binding mismatch"
            )
        native = _contained_evidence_file(
            root, native_row.get("native"), "smoke native"
        )
        candidate = _contained_evidence_file(
            root, candidate_row.get("generated"), "smoke candidate"
        )
        samples.append(
            SampleEvidence(
                sample_id=sample_id,
                source=source,
                native=native,
                candidate=candidate,
                source_sha256=_file_sha256(source),
                native_sha256=_file_sha256(native),
                candidate_sha256=_file_sha256(candidate),
            )
        )
    return tuple(samples)


def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(_regular_file(path, "per-sample evidence")):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in rows:
            raise ContinuationContractError(
                "per-sample evidence has an invalid or duplicate ID"
            )
        rows[sample_id] = row
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContinuationContractError(
                f"invalid JSONL row {path}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ContinuationContractError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _child_smoke_root(
    root: Path,
    value: Any,
    *,
    child_campaign_id: str,
    kind: str,
) -> Path:
    relative = Path(_nonempty(value, f"{kind} artifact root"))
    if relative.is_absolute():
        raise ContinuationContractError("child smoke root must be repo-relative")
    path = (root / relative).resolve()
    expected_parent = (
        root
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / child_campaign_id
        / "evaluator_smoke"
    ).resolve()
    if expected_parent not in path.parents:
        raise ContinuationContractError(
            f"{kind} smoke root is outside the child campaign"
        )
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise ContinuationContractError(f"{kind} smoke root is not a real directory")
    return path


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ContinuationContractError(f"{label} is not a regular file")
    return path


def _contained_evidence_file(root: Path, value: Any, label: str) -> Path:
    raw = Path(_nonempty(value, f"{label} path"))
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not path.is_file() or path.is_symlink():
        raise ContinuationContractError(f"{label} is not a regular file")
    return path


def _verified_current_sha(path: Path, value: Any, label: str) -> str:
    declared = _sha(value, f"{label} SHA256")
    if _file_sha256(path) != declared:
        raise ContinuationContractError(f"{label} SHA256 mismatch")
    return declared


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_value_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json_contract_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _read_contract(path: Path, field: str) -> dict[str, Any]:
    payload = _read_json(path)
    declared = _sha(payload.get(field), field)
    if declared != _canonical_digest(payload, field):
        raise ContinuationContractError(f"{field} canonical digest mismatch")
    return payload


def _contract_binding(
    root: Path, path: Path, contract_sha256: str
) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "contract_sha256": contract_sha256,
    }


def _verified_file_binding(root: Path, value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, label)
    path = _contained(root, row.get("path"), label)
    declared = _sha(row.get("sha256"), f"{label} SHA256")
    if hashlib.sha256(path.read_bytes()).hexdigest() != declared:
        raise ContinuationContractError(f"{label} file SHA256 mismatch")
    return dict(row)


def _actual_file_binding(root: Path, value: Any) -> dict[str, str]:
    path = _contained(root, value, "implementation")
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _verified_contract_binding(root: Path, value: Any, label: str) -> dict[str, str]:
    row = _mapping(value, label)
    if set(row) != {"path", "file_sha256", "contract_sha256"}:
        raise ContinuationContractError(f"{label} binding fields mismatch")
    path = _contained(root, row["path"], label)
    if hashlib.sha256(path.read_bytes()).hexdigest() != _sha(
        row["file_sha256"], f"{label} file SHA256"
    ):
        raise ContinuationContractError(f"{label} file SHA256 mismatch")
    _sha(row["contract_sha256"], f"{label} contract SHA256")
    return dict(row)


def _contained(root: Path, value: Any, label: str) -> Path:
    relative = Path(_nonempty(value, f"{label} path"))
    if relative.is_absolute():
        raise ContinuationContractError(f"{label} path must be repo-relative")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ContinuationContractError(f"{label} path is not a regular repo file")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationContractError(f"missing immutable contract: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContinuationContractError(f"invalid JSON contract: {path}") from error
    return _mapping(payload, str(path))


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ContinuationContractError("continuation contract already differs")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
        os.link(temporary, path)
    except FileExistsError as error:
        raise ContinuationContractError("continuation contract creation raced") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuationContractError(f"{label} must be a mapping")
    return dict(value)


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContinuationContractError(f"{label} must be non-empty")
    return value


def _sha(value: Any, label: str) -> str:
    result = _nonempty(value, label)
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ContinuationContractError(f"{label} must be lowercase SHA256")
    return result
