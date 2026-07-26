from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class EvaluatorResourceContractError(ValueError):
    """Raised when an R9 evaluator resource profile is not exact."""


_SMOKE_FILES = frozenset(
    {
        "request.json",
        "request_claim.json",
        "execution_claim.json",
        "worker_result.json",
        "resource_result.json",
        "worker.log",
        "controller.log",
    }
)


def materialize_evaluator_resource_profiles(
    value: Any,
    *,
    repo_root: Path,
    worker_contract: Mapping[str, Any],
    arcface_contract_sha256: str,
    quality_script_sha256: str,
    runtime_config_path: Any,
    runtime_config_sha256: str,
) -> dict[str, Any]:
    expected_runtime = _normalize_expected_runtime_config(
        repo_root,
        runtime_config_path,
        runtime_config_sha256,
    )
    raw = _mapping(value, "evaluator resource profiles")
    if set(raw) != {"arcface", "quality", "heldout"}:
        raise EvaluatorResourceContractError(
            "evaluator resource profile fields are not canonical"
        )
    expected_modes = {
        "arcface": "measured_single_worker",
        "quality": "measured_exclusive_bootstrap",
    }
    normalized = {}
    claim_runtimes = {}
    for kind, mode in expected_modes.items():
        declaration = _mapping(raw.get(kind), f"{kind} resource profile")
        if set(declaration) != {"mode", "artifact_root"}:
            raise EvaluatorResourceContractError(
                f"{kind} resource profile declaration is not canonical"
            )
        if declaration.get("mode") != mode:
            raise EvaluatorResourceContractError(f"{kind} resource mode mismatch")
        artifact_root = _repo_dir(
            repo_root, declaration.get("artifact_root"), f"{kind} smoke root"
        )
        normalized[kind], claim_runtimes[kind] = _materialize_measured_profile(
            kind,
            artifact_root=artifact_root,
            repo_root=repo_root,
            worker_contract=worker_contract,
            arcface_contract_sha256=arcface_contract_sha256,
            quality_script_sha256=quality_script_sha256,
            expected_runtime=expected_runtime,
        )
    if claim_runtimes["arcface"] != claim_runtimes["quality"]:
        raise EvaluatorResourceContractError(
            "ArcFace and quality smoke claims bind different runtime configs"
        )
    heldout = _mapping(raw.get("heldout"), "heldout resource profile")
    expected_heldout = {
        "mode": "exclusive_single_official_run",
        "smoke_execution": "sealed_until_winner_lock",
        "global_exclusive_slots": 16,
        "ram_admission_percent": 85,
        "ram_hard_limit_percent": 90,
    }
    if heldout != expected_heldout:
        raise EvaluatorResourceContractError(
            "heldout resource profile must remain sealed and globally exclusive"
        )
    normalized["heldout"] = expected_heldout
    normalized["resource_profiles_sha256"] = _canonical_sha256(normalized)
    return normalized


def validate_evaluator_resource_profiles(
    value: Any,
    *,
    repo_root: Path,
    worker_contract: Mapping[str, Any],
    arcface_contract_sha256: str,
    quality_script_sha256: str,
    runtime_config_path: Any,
    runtime_config_sha256: str,
) -> dict[str, Any]:
    declared = _mapping(value, "evaluator resource profiles")
    if "resource_profile_binding" in declared:
        return _validate_full_e2e_resource_profiles(
            declared,
            repo_root=repo_root,
            worker_contract=worker_contract,
            arcface_contract_sha256=arcface_contract_sha256,
            quality_script_sha256=quality_script_sha256,
        )
    if set(declared) != {
        "arcface",
        "quality",
        "heldout",
        "resource_profiles_sha256",
    }:
        raise EvaluatorResourceContractError(
            "effective evaluator resource profile fields are not canonical"
        )
    digest = _sha256(declared.get("resource_profiles_sha256"), "resource profiles")
    canonical = dict(declared)
    canonical.pop("resource_profiles_sha256")
    if _canonical_sha256(canonical) != digest:
        raise EvaluatorResourceContractError(
            "evaluator resource profile canonical digest mismatch"
        )
    raw = {
        kind: {
            "mode": _mapping(declared.get(kind), f"{kind} profile").get("mode"),
            "artifact_root": _mapping(declared.get(kind), f"{kind} profile").get(
                "artifact_root"
            ),
        }
        for kind in ("arcface", "quality")
    }
    raw["heldout"] = dict(_mapping(declared.get("heldout"), "heldout profile"))
    rematerialized = materialize_evaluator_resource_profiles(
        raw,
        repo_root=repo_root,
        worker_contract=worker_contract,
        arcface_contract_sha256=arcface_contract_sha256,
        quality_script_sha256=quality_script_sha256,
        runtime_config_path=runtime_config_path,
        runtime_config_sha256=runtime_config_sha256,
    )
    if rematerialized != declared:
        raise EvaluatorResourceContractError(
            "evaluator resource profile disagrees with immutable smoke artifacts"
        )
    return rematerialized


def _validate_full_e2e_resource_profiles(
    declared: Mapping[str, Any],
    *,
    repo_root: Path,
    worker_contract: Mapping[str, Any],
    arcface_contract_sha256: str,
    quality_script_sha256: str,
) -> dict[str, Any]:
    value = _mapping(declared, "Full E2E resource profiles")
    if set(value) != {
        "arcface",
        "quality",
        "heldout",
        "resource_profiles_sha256",
        "resource_profile_binding",
    }:
        raise EvaluatorResourceContractError(
            "Full E2E resource profile fields are not canonical"
        )
    binding = _mapping(
        value["resource_profile_binding"], "Full E2E resource profile binding"
    )
    path = _repo_file(
        repo_root, binding.get("path"), "Full E2E resource profile contract"
    )
    profile = _read_mapping(path)
    declared_digest = _sha256(
        profile.get("resource_profiles_sha256"),
        "Full E2E resource profiles",
    )
    canonical = dict(profile)
    canonical.pop("resource_profiles_sha256")
    if (
        binding
        != {
            "path": str(path.relative_to(repo_root.resolve())),
            "file_sha256": _file_sha256(path),
            "contract_sha256": declared_digest,
        }
        or _canonical_sha256(canonical) != declared_digest
        or profile.get("contract_type")
        != "safa_r9_full_e2e_resource_profiles_v1"
        or profile.get("worker_contract")
        != _normalize_expected_worker(worker_contract, repo_root)
        or profile.get("arcface_contract_sha256")
        != _sha256(arcface_contract_sha256, "ArcFace contract")
        or profile.get("quality_script_sha256")
        != _sha256(quality_script_sha256, "quality script")
        or value["resource_profiles_sha256"] != declared_digest
        or value["arcface"] != profile.get("arcface")
        or value["quality"] != profile.get("quality")
        or value["heldout"] != profile.get("heldout")
    ):
        raise EvaluatorResourceContractError(
            "Full runtime resource profile does not bind current evaluator bytes"
        )
    for kind, mode in (
        ("arcface", "measured_single_worker"),
        ("quality", "measured_exclusive_bootstrap"),
    ):
        row = _mapping(profile.get(kind), f"Full E2E {kind} profile")
        peak = _positive_int(
            row.get("peak_process_tree_rss_bytes"), f"{kind} peak RSS"
        )
        if (
            row.get("mode") != mode
            or row.get("ram_slot_budget_bytes") != (peak * 110 + 99) // 100
        ):
            raise EvaluatorResourceContractError(
                f"Full E2E {kind} resource budget changed"
            )
    return dict(value)


def _materialize_measured_profile(
    kind: str,
    *,
    artifact_root: Path,
    repo_root: Path,
    worker_contract: Mapping[str, Any],
    arcface_contract_sha256: str,
    quality_script_sha256: str,
    expected_runtime: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    inventory = {path.name for path in artifact_root.iterdir()}
    if inventory != _SMOKE_FILES or any(
        path.is_symlink() or not path.is_file() for path in artifact_root.iterdir()
    ):
        raise EvaluatorResourceContractError(
            f"{kind} smoke artifact inventory is not canonical"
        )
    request_path = artifact_root / "request.json"
    claim_path = artifact_root / "request_claim.json"
    execution_path = artifact_root / "execution_claim.json"
    worker_path = artifact_root / "worker_result.json"
    result_path = artifact_root / "resource_result.json"
    request = _digest_contract(
        request_path,
        digest_field="evaluator_request_sha256",
        contract_type="safa_r9_phase_evaluator_request_v1",
    )
    raw_claim = _read_mapping(claim_path)
    claim_type = raw_claim.get("contract_type")
    if claim_type not in {
        "safa_r9_evaluator_resource_smoke_request_v1",
        "safa_r9_evaluator_resource_smoke_request_v2",
    }:
        raise EvaluatorResourceContractError("smoke request claim type changed")
    claim = _digest_contract(
        claim_path,
        digest_field="smoke_request_claim_sha256",
        contract_type=str(claim_type),
    )
    claim_runtime = _normalize_claim_runtime_config(claim, repo_root)
    if claim_runtime != expected_runtime:
        raise EvaluatorResourceContractError(
            f"{kind} smoke claim runtime config binding mismatch"
        )
    execution_type = {
        "arcface": "safa_r9_evaluator_resource_smoke_execution_v1",
        "quality": "safa_r9_quality_bootstrap_smoke_execution_v1",
    }[kind]
    execution = _digest_contract(
        execution_path,
        digest_field="execution_claim_sha256",
        contract_type=execution_type,
    )
    worker = _digest_contract(
        worker_path,
        digest_field="evaluator_output_sha256",
        contract_type="safa_r9_phase_evaluator_output_v1",
    )
    result_type = {
        "arcface": "safa_r9_evaluator_resource_smoke_result_v1",
        "quality": "safa_r9_quality_bootstrap_smoke_result_v1",
    }[kind]
    result = _digest_contract(
        result_path,
        digest_field="resource_smoke_result_sha256",
        contract_type=result_type,
    )
    normalized_worker = _normalize_expected_worker(worker_contract, repo_root)
    request_config = _mapping(request.get("config"), "smoke request config")
    request_quality = _mapping(
        request_config.get("quality_script"), "smoke quality script"
    )
    request_arcface_sha = _canonical_sha256(
        _mapping(request_config.get("arcface"), "smoke ArcFace contract")
    )
    request_digest = request["evaluator_request_sha256"]
    claim_digest = claim["smoke_request_claim_sha256"]
    expected_sample_count = (
        8
        if claim_type == "safa_r9_evaluator_resource_smoke_request_v2"
        else 64
    )
    if (
        request.get("task") != kind
        or claim.get("kind") != kind
        or claim.get("sample_count") != expected_sample_count
        or claim.get("retry_allowed") is not False
        or claim.get("evaluator_request_sha256") != request_digest
        or claim.get("worker_contract") != normalized_worker
        or request_config.get("worker_contract") != normalized_worker
        or claim.get("arcface_contract_sha256")
        != _sha256(arcface_contract_sha256, "ArcFace contract")
        or request_arcface_sha != arcface_contract_sha256
        or claim.get("quality_script_sha256")
        != _sha256(quality_script_sha256, "quality script")
        or request_quality.get("sha256") != quality_script_sha256
    ):
        raise EvaluatorResourceContractError(
            f"{kind} smoke request does not bind the current evaluator contract"
        )
    if claim_type == "safa_r9_evaluator_resource_smoke_request_v2":
        request_payload = _mapping(request.get("payload"), "v2 smoke payload")
        request_samples = request_payload.get("samples")
        manifest = _mapping(claim.get("manifest"), "v2 smoke manifest")
        if (
            claim.get("campaign_id") != "r9-report-only-formal-v9"
            or claim.get("phase") != "full"
            or claim.get("arm_id") != "paper_eta_0p125"
            or claim.get("batch_size") != 2
            or set(manifest) != {
                "path",
                "sha256",
                "sample_count",
                "ordered_sample_id_sha256",
            }
            or manifest.get("sample_count") != 8
            or not isinstance(claim.get("smoke_supersession_sha256"), str)
            or len(str(claim["smoke_supersession_sha256"])) != 64
            or request_payload.get("phase") != "full"
            or request_payload.get("arm_id") != "paper_eta_0p125"
            or request_payload.get("logical_run_id")
            != f"resource_smoke_{kind}_full_8"
            or not isinstance(request_samples, list)
            or len(request_samples) != 8
            or request_config.get("batch_size") != 2
        ):
            raise EvaluatorResourceContractError(
                f"{kind} v2 smoke request violates the frozen Full policy"
            )
    quality_path = Path(str(request_quality.get("path"))).resolve()
    if _contained_file(repo_root, quality_path, "quality script") is None:
        raise AssertionError("unreachable")
    if _file_sha256(quality_path) != quality_script_sha256:
        raise EvaluatorResourceContractError("quality script bytes changed")
    if (
        execution.get("evaluator_request_sha256") != request_digest
        or execution.get("request_claim_sha256") != claim_digest
        or execution.get("retry_allowed") is not False
        or worker.get("task") != kind
        or worker.get("evaluator_request_sha256") != request_digest
        or worker.get("worker_contract") != normalized_worker
        or worker.get("arcface_contract_sha256") != arcface_contract_sha256
        or worker.get("quality_script_sha256") != quality_script_sha256
        or result.get("execution_claim_sha256") != execution["execution_claim_sha256"]
        or result.get("status") != "succeeded"
        or result.get("failure_reason") is not None
        or result.get("returncode") != 0
        or result.get("retry_allowed") is not False
        or result.get("worker_output_sha256") != _file_sha256(worker_path)
        or result.get("worker_evaluator_output_sha256")
        != worker["evaluator_output_sha256"]
        or result.get("worker_log_sha256") != _file_sha256(artifact_root / "worker.log")
    ):
        raise EvaluatorResourceContractError(
            f"{kind} smoke execution/result binding mismatch"
        )
    peak_rss = _positive_int(
        result.get("peak_process_tree_rss_bytes"), f"{kind} peak RSS"
    )
    peak_gpu = _positive_int(
        result.get("peak_gpu_memory_bytes"), f"{kind} peak GPU memory"
    )
    budget = _positive_int(
        result.get("ram_slot_budget_bytes"), f"{kind} RAM slot budget"
    )
    if budget != (peak_rss * 110 + 99) // 100:
        raise EvaluatorResourceContractError(
            f"{kind} RAM budget is not ceil(peak RSS * 1.10)"
        )
    gpu_uuid = result.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise EvaluatorResourceContractError(f"{kind} GPU UUID is invalid")
    if kind == "arcface":
        _validate_arcface_result(
            worker.get("result"), expected_sample_count=expected_sample_count
        )
        if execution.get("kind") != "arcface":
            raise EvaluatorResourceContractError("ArcFace smoke kind mismatch")
    else:
        _validate_quality_result(
            worker.get("result"), expected_sample_count=expected_sample_count
        )
        ram = _mapping(execution.get("ram"), "quality smoke RAM policy")
        if (
            execution.get("global_exclusive_slots") != 16
            or ram.get("admission_percent") != 85
            or ram.get("hard_limit_percent") != 90
        ):
            raise EvaluatorResourceContractError(
                "quality smoke did not hold all slots under the 85/90 RAM policy"
            )
    bindings = {
        "request": _binding(request_path, request_digest, repo_root),
        "request_claim": _binding(claim_path, claim_digest, repo_root),
        "execution_claim": _binding(
            execution_path, execution["execution_claim_sha256"], repo_root
        ),
        "worker_result": _binding(
            worker_path, worker["evaluator_output_sha256"], repo_root
        ),
        "resource_result": _binding(
            result_path, result["resource_smoke_result_sha256"], repo_root
        ),
    }
    inventory_rows = [
        {"name": name, "sha256": _file_sha256(artifact_root / name)}
        for name in sorted(_SMOKE_FILES)
    ]
    return {
        "mode": {
            "arcface": "measured_single_worker",
            "quality": "measured_exclusive_bootstrap",
        }[kind],
        "artifact_root": str(artifact_root.relative_to(repo_root.resolve())),
        **bindings,
        "worker_log": _file_binding(artifact_root / "worker.log", repo_root),
        "controller_log": _file_binding(artifact_root / "controller.log", repo_root),
        "artifact_inventory_sha256": _canonical_sha256(inventory_rows),
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_gpu_memory_bytes": peak_gpu,
        "ram_slot_budget_bytes": budget,
        "gpu_uuid": gpu_uuid,
    }, claim_runtime


def _validate_arcface_result(value: Any, *, expected_sample_count: int) -> None:
    if not isinstance(value, list) or len(value) != expected_sample_count:
        raise EvaluatorResourceContractError(
            f"ArcFace smoke must cover {expected_sample_count} rows"
        )
    ids = []
    for row in value:
        item = _mapping(row, "ArcFace smoke row")
        ids.append(item.get("sample_id"))
        if any(
            item.get(field) != 1
            for field in (
                "source_face_count",
                "native_face_count",
                "candidate_face_count",
            )
        ) or any(
            not _finite(item.get(field))
            for field in ("source_native_cosine", "source_candidate_cosine")
        ):
            raise EvaluatorResourceContractError(
                "ArcFace smoke requires exact-one finite evidence"
            )
    if (
        any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != expected_sample_count
    ):
        raise EvaluatorResourceContractError("ArcFace smoke IDs are not unique")


def _validate_quality_result(value: Any, *, expected_sample_count: int) -> None:
    result = _mapping(value, "quality smoke result")
    if (
        result.get("metrics") != ["fid", "kid", "niqe", "sharpness"]
        or result.get("num_generated") != expected_sample_count
        or result.get("num_real") != expected_sample_count
        or result.get("sample_id_count") != expected_sample_count
    ):
        raise EvaluatorResourceContractError("quality smoke coverage mismatch")
    iqa = _mapping(result.get("iqa"), "quality IQA")
    sharpness = _mapping(result.get("sharpness"), "quality sharpness")
    scalars = [
        result.get("fid"),
        result.get("kid_mean"),
        result.get("kid_std"),
        iqa.get("mean"),
        iqa.get("std"),
        *(
            sharpness.get(field)
            for field in ("mean", "std", "median", "p05", "p10", "p90", "p95")
        ),
    ]
    if not all(_finite(value) for value in scalars):
        raise EvaluatorResourceContractError("quality smoke metrics are non-finite")


def _normalize_expected_worker(
    value: Mapping[str, Any], repo_root: Path
) -> dict[str, str]:
    expected = {"path", "sha256", "implementation_path", "implementation_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EvaluatorResourceContractError("worker contract fields are not canonical")
    normalized = {}
    for path_field, sha_field in (
        ("path", "sha256"),
        ("implementation_path", "implementation_sha256"),
    ):
        path = _repo_file(repo_root, value[path_field], f"worker {path_field}")
        digest = _sha256(value[sha_field], f"worker {sha_field}")
        if _file_sha256(path) != digest:
            raise EvaluatorResourceContractError("worker contract bytes changed")
        normalized[path_field] = str(path)
        normalized[sha_field] = digest
    return normalized


def _normalize_expected_runtime_config(
    repo_root: Path,
    path_value: Any,
    sha256_value: Any,
) -> dict[str, str]:
    path = _repo_file(repo_root, path_value, "expected runtime config")
    digest = _sha256(sha256_value, "expected runtime config")
    if _file_sha256(path) != digest:
        raise EvaluatorResourceContractError(
            "expected runtime config bytes changed"
        )
    return {"path": str(path), "sha256": digest}


def _normalize_claim_runtime_config(
    claim: Mapping[str, Any], repo_root: Path
) -> dict[str, str]:
    raw_path = claim.get("runtime_config")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise EvaluatorResourceContractError(
            "smoke claim runtime config path must be canonical and absolute"
        )
    path = _repo_file(repo_root, raw_path, "smoke claim runtime config")
    if str(path) != raw_path:
        raise EvaluatorResourceContractError(
            "smoke claim runtime config path is not canonical"
        )
    digest = _sha256(
        claim.get("runtime_config_sha256"), "smoke claim runtime config"
    )
    if _file_sha256(path) != digest:
        raise EvaluatorResourceContractError(
            "smoke claim runtime config SHA256 mismatch"
        )
    return {"path": str(path), "sha256": digest}


def _digest_contract(
    path: Path, *, digest_field: str, contract_type: str
) -> dict[str, Any]:
    value = _read_mapping(path)
    if value.get("schema_version") != 1 or value.get("contract_type") != contract_type:
        raise EvaluatorResourceContractError(f"contract identity mismatch: {path}")
    declared = _sha256(value.get(digest_field), f"{path.name} canonical digest")
    canonical = dict(value)
    canonical.pop(digest_field, None)
    if _canonical_sha256(canonical) != declared:
        raise EvaluatorResourceContractError(f"contract digest mismatch: {path}")
    return value


def _binding(path: Path, contract_sha256: str, repo_root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(repo_root.resolve())),
        "file_sha256": _file_sha256(path),
        "contract_sha256": contract_sha256,
    }


def _file_binding(path: Path, repo_root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(repo_root.resolve())),
        "file_sha256": _file_sha256(path),
    }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluatorResourceContractError(f"{label} must be a mapping")
    return dict(value)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluatorResourceContractError(f"invalid JSON contract: {path}") from exc
    return _mapping(value, str(path))


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise EvaluatorResourceContractError("contract is not finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluatorResourceContractError(f"{label} is not a SHA256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluatorResourceContractError(f"{label} must be a positive integer")
    return value


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _repo_dir(repo_root: Path, value: Any, label: str) -> Path:
    path = _contained(repo_root, value, label)
    if not path.is_dir() or path.is_symlink():
        raise EvaluatorResourceContractError(f"{label} is not a real directory")
    return path


def _repo_file(repo_root: Path, value: Any, label: str) -> Path:
    path = _contained(repo_root, value, label)
    if not path.is_file() or path.is_symlink():
        raise EvaluatorResourceContractError(f"{label} is not a real file")
    return path


def _contained(repo_root: Path, value: Any, label: str) -> Path:
    root = repo_root.resolve()
    raw = Path(str(value))
    path = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvaluatorResourceContractError(f"{label} escapes repository") from exc
    return path


def _contained_file(repo_root: Path, path: Path, label: str) -> Path:
    return _repo_file(repo_root, path, label)
