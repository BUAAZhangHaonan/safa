"""Fail-closed contracts for the historical canonical 512 screening campaign."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

from safa.closeout.generator_output_contract import (
    DECODER_REGISTRY_CONTRACT,
    GeneratorOutputContractError,
    LATENT_DECODER_TYPE,
    PIXEL_DECODER_TYPE,
    decoder_registry_digest,
    validate_decoder_registry,
    validate_output_contract,
)

POLICY_CONTRACT = "safa_canonical_screening_policy_v1"
PLAN_CONTRACT = "safa_canonical_checkpoint_plan_v1"
PREFLIGHT_REQUEST_CONTRACT = "safa_canonical_checkpoint_preflight_request_v1"
PREFLIGHT_RESULT_CONTRACT = "safa_canonical_checkpoint_preflight_result_v1"
CANDIDATE_MANIFEST_CONTRACT = "safa_canonical_screening_candidate_manifest_v1"
RUN_REQUEST_CONTRACT = "safa_canonical_screening_run_request_v1"
RUN_CLAIM_CONTRACT = "safa_canonical_screening_run_claim_v1"
RUN_RESULT_CONTRACT = "safa_canonical_screening_run_result_v1"
CONTROLLER_READY_CONTRACT = "safa_canonical_gpu_controller_ready_v1"
OBSERVER_READY_CONTRACT = "safa_canonical_gpu_observer_ready_v1"
FINAL_RELEASE_ADMISSION_CONTRACT = (
    "safa_canonical_gpu_final_release_admission_v1"
)
WORKER_READY_CONTRACT = "safa_canonical_worker_pre_cuda_ready_v2"
WORKER_RELEASE_CONTRACT = "safa_canonical_worker_cuda_release_v2"
CONTROLLER_LAUNCH_REHASH_CONTRACT = (
    "safa_canonical_controller_launch_rehash_v2"
)
WORKER_PRE_CUDA_VERIFICATION_ORDER = (
    "policy_config",
    "implementations",
    "run_request",
    "candidate_manifest",
    "checkpoint_plan",
    "checkpoint",
    "data_and_evaluators",
    "final_release",
    "ready_barrier",
)
WORKER_EXTERNAL_GPU_RACE_CONTRACT = {
    "external_process_exclusion_guarantee": False,
    "scope": "controller_worker_processes_only",
    "residual_race": (
        "GPU process and resource snapshots are point-in-time observations; "
        "an unrelated external process can start after the final snapshot."
    ),
    "compute_mode_changed": False,
}
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_MODES = {"smoke8": 8, "screen512": 512}
CHECKPOINT_SELECTORS = {"raw", "ema"}
NVIDIA_GPU_UUID_RE = re.compile(
    r"^(?:GPU-)?([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})$"
)


class CanonicalScreeningError(RuntimeError):
    """A fail-closed canonical screening contract violation."""


def canonicalize_nvidia_gpu_uuid(
    value: Any, label: str
) -> dict[str, str]:
    """Return strict raw evidence and the canonical NVIDIA GPU UUID."""
    raw_type = f"{type(value).__module__}.{type(value).__qualname__}"
    raw_repr = repr(value)
    if isinstance(value, bytes):
        try:
            raw_string = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise CanonicalScreeningError(
                f"{label} is not an ASCII NVIDIA GPU UUID"
            ) from exc
    elif isinstance(value, str):
        raw_string = value
    elif (
        type(value).__module__ == "torch._C"
        and type(value).__qualname__ == "_CUuuid"
    ):
        raw_string = str(value)
    else:
        raise CanonicalScreeningError(
            f"{label} must be a string, ASCII bytes, or torch._C._CUuuid"
        )
    match = NVIDIA_GPU_UUID_RE.fullmatch(raw_string)
    if match is None:
        raise CanonicalScreeningError(
            f"{label} is not a canonicalizable NVIDIA GPU UUID"
        )
    return {
        "raw_type": raw_type,
        "raw_repr": raw_repr,
        "raw_string": raw_string,
        "canonical": f"GPU-{match.group(1).lower()}",
    }


def canonical_gpu_registry(
    registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in registry:
        index = row.get("physical_gpu_index")
        if type(index) is not int:
            raise CanonicalScreeningError("GPU registry index is invalid")
        uuid = canonicalize_nvidia_gpu_uuid(
            row.get("physical_gpu_uuid"), "GPU registry UUID"
        )
        normalized.append(
            {
                "physical_gpu_index": index,
                "physical_gpu_uuid": uuid["canonical"],
            }
        )
    if len(normalized) != len({row["physical_gpu_index"] for row in normalized}):
        raise CanonicalScreeningError("GPU registry has duplicate indices")
    if len(normalized) != len(
        {row["physical_gpu_uuid"] for row in normalized}
    ):
        raise CanonicalScreeningError("GPU registry has duplicate UUIDs")
    return normalized


def ram_probe_contract_digest(spec: Mapping[str, Any]) -> str:
    payload = dict(spec)
    for field in (
        "authorized_gpu_registry",
        "admission",
        "probe_contract_sha256",
        "probe_execution_sha256",
    ):
        payload.pop(field, None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def ram_probe_admission_evidence_digest(
    admission: Mapping[str, Any],
) -> str:
    payload = dict(admission)
    for field in (
        "probe_execution_sha256",
        "admission_evidence_sha256",
        "admission_sha256",
    ):
        payload.pop(field, None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def ram_probe_execution_digest(
    probe_contract_sha256: str,
    registry: Sequence[Mapping[str, Any]],
    admission_evidence_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema_version": 1,
                "contract_type": "safa_canonical_screening_ram_probe_execution_v1",
                "probe_contract_sha256": probe_contract_sha256,
                "authorized_gpu_registry": list(registry),
                "admission_evidence_sha256": admission_evidence_sha256,
            }
        )
    ).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_digest(value: Mapping[str, Any], digest_field: str) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_asset_directory_content(
    path: Path, expected_digest: str
) -> dict[str, Any]:
    """Read and hash every asset byte; stat metadata and caches are not trusted."""
    if not SHA256_RE.fullmatch(expected_digest):
        raise CanonicalScreeningError("asset expected digest is invalid")
    asset = path.resolve()
    if asset.is_symlink() or not asset.is_dir():
        raise CanonicalScreeningError(
            f"asset directory is missing or symlinked: {asset}"
        )
    files = sorted(
        item for item in asset.rglob("*") if item.is_file() or item.is_symlink()
    )
    if not files:
        raise CanonicalScreeningError(
            f"asset directory contains no files: {asset}"
        )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    total_bytes = 0
    digest = hashlib.sha256()
    for file_path in files:
        if file_path.is_symlink():
            raise CanonicalScreeningError(
                f"asset directory contains a symlink: {file_path}"
            )
        digest.update(file_path.relative_to(asset).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total_bytes += len(chunk)
                digest.update(chunk)
        digest.update(b"\0")
    observed_digest = digest.hexdigest()
    completed_at = datetime.now(timezone.utc).isoformat()
    if observed_digest != expected_digest:
        raise CanonicalScreeningError("asset directory content digest differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_type": "safa_canonical_asset_content_verification_v1",
        "path": str(asset),
        "digest_algorithm": "sha256_relative_posix_nul_content_nul_v1",
        "expected_digest": expected_digest,
        "observed_digest": observed_digest,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def sha256_directory_tree(path: Path) -> str:
    """Hash every regular file by relative POSIX path and content."""
    root = path.resolve()
    if not root.is_dir():
        raise CanonicalScreeningError(f"evidence directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_ram_probe_artifact_seal(
    repo_root: Path, raw_seal: Mapping[str, Any]
) -> dict[str, Any]:
    seal = _require_mapping(raw_seal, "RAM probe artifact seal")
    _require_exact_keys(
        seal,
        {
            "root",
            "file_count",
            "directory_count",
            "symlink_count",
            "files",
            "controller_terminal",
            "scientific_result_reuse",
        },
        "RAM probe artifact seal",
    )
    root_binding = _require_mapping(seal["root"], "RAM probe sealed root")
    _require_exact_keys(
        root_binding,
        {"path", "digest", "digest_algorithm"},
        "RAM probe sealed root",
    )
    expected_relative_root = (
        "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__4d0345b6fc29cc8e"
    )
    if root_binding["path"] != expected_relative_root:
        raise CanonicalScreeningError("sealed RAM probe root identity differs")
    _require_no_repo_path_component_symlinks(
        repo_root,
        expected_relative_root,
        "sealed RAM probe root",
    )
    unresolved_root = repo_root.resolve() / expected_relative_root
    _require_tree_without_symlinks(unresolved_root, "sealed RAM probe root")
    root = unresolved_root.resolve()
    expected_root = (
        repo_root.resolve()
        / "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__4d0345b6fc29cc8e"
    ).resolve()
    if root != expected_root or not root.is_dir():
        raise CanonicalScreeningError("sealed RAM probe root differs")
    entries = list(root.rglob("*"))
    symlinks = [entry for entry in entries if entry.is_symlink()]
    files = [entry for entry in entries if entry.is_file()]
    directories = [entry for entry in entries if entry.is_dir()]
    expected_files = _require_mapping(seal["files"], "RAM probe sealed files")
    actual_paths = {entry.relative_to(root).as_posix() for entry in files}
    if (
        root_binding["digest_algorithm"]
        != "sha256_relative_posix_nul_content_nul_v1"
        or root_binding["digest"] != sha256_directory_tree(root)
        or seal["file_count"] != 28
        or seal["file_count"] != len(files)
        or seal["directory_count"] != 5
        or seal["directory_count"] != len(directories)
        or seal["symlink_count"] != 0
        or symlinks
        or set(expected_files) != actual_paths
        or len([path for path in actual_paths if path.endswith(".png")]) != 16
        or seal["controller_terminal"] != "absent_by_contract"
        or (root / "controller_terminal.json").exists()
        or seal["scientific_result_reuse"] != "forbidden"
    ):
        raise CanonicalScreeningError("sealed RAM probe artifact tree differs")
    for relative_path, expected_sha256 in expected_files.items():
        _require_sha256(expected_sha256, f"RAM probe sealed file {relative_path}")
        if sha256_file(root / relative_path) != expected_sha256:
            raise CanonicalScreeningError(
                f"sealed RAM probe file differs: {relative_path}"
            )
    return dict(seal)


def write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    content = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    complete = False
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            path.unlink(missing_ok=True)


def publish_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a complete cross-process JSON contract once."""
    content = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.publish.{os.getpid()}.{threading.get_ident()}."
        f"{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    linked = False
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path)
        linked = True
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if not linked and path.exists():
            raise CanonicalScreeningError(
                "atomic JSON publish failed after final path appeared"
            )


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalScreeningError(f"{label} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalScreeningError(f"{label} must be a JSON object: {path}")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CanonicalScreeningError(f"{label} is not readable: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanonicalScreeningError(
                f"{label} has invalid JSON at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise CanonicalScreeningError(
                f"{label} line {line_number} must be a JSON object"
            )
        rows.append(row)
    if not rows:
        raise CanonicalScreeningError(f"{label} is empty: {path}")
    return rows


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalScreeningError(f"{label} must be a mapping")
    return dict(value)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CanonicalScreeningError(f"{label} must be a lowercase SHA256 digest")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise CanonicalScreeningError(
            f"{label} fields differ: missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )


def _repo_path(repo_root: Path, raw: Any, label: str, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw:
        raise CanonicalScreeningError(f"{label} path must be a non-empty string")
    root = repo_root.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CanonicalScreeningError(f"{label} escapes repository root: {raw}") from exc
    if must_exist and not resolved.is_file():
        raise CanonicalScreeningError(f"{label} does not exist: {resolved}")
    return resolved


def _validate_bound_file(
    repo_root: Path,
    value: Any,
    label: str,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    binding = _require_mapping(value, label)
    required = {"path", "sha256"}
    if expected_count is not None:
        required.add("sample_count")
    _require_exact_keys(binding, required, label)
    path = _repo_path(repo_root, binding["path"], label)
    expected = _require_sha256(binding["sha256"], f"{label} SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise CanonicalScreeningError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    if expected_count is not None and binding["sample_count"] != expected_count:
        raise CanonicalScreeningError(
            f"{label} sample_count must be {expected_count}, got {binding['sample_count']!r}"
        )
    return {"path": str(path), "sha256": expected, **(
        {"sample_count": expected_count} if expected_count is not None else {}
    )}


def _require_no_repo_path_component_symlinks(
    repo_root: Path, raw_path: Any, label: str
) -> None:
    root = repo_root.resolve()
    unresolved = Path(str(raw_path))
    candidate = unresolved if unresolved.is_absolute() else root / unresolved
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise CanonicalScreeningError(
            f"{label} escapes repository root"
        ) from exc
    component = root
    for part in relative.parts:
        component = component / part
        if component.is_symlink():
            raise CanonicalScreeningError(
                f"{label} path components must not be symlinks"
            )


def _require_tree_without_symlinks(root: Path, label: str) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise CanonicalScreeningError(
            f"{label} must not contain symlinks"
        )


def validate_arcface_execution_probe_binding(
    repo_root: Path,
    value: Any,
    *,
    arcface_contract: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the complete frozen ArcFace bootstrap provenance chain."""
    binding = _require_mapping(value, "ArcFace execution probe")
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
    _require_exact_keys(
        binding, expected_fields, "ArcFace execution probe"
    )

    def bound_path(
        path_field: str, file_sha_field: str, label: str
    ) -> Path:
        path = _repo_path(repo_root, binding[path_field], label)
        _require_no_repo_path_component_symlinks(
            repo_root, binding[path_field], label
        )
        expected_file_sha = _require_sha256(
            binding[file_sha_field], f"{label} file SHA256"
        )
        if sha256_file(path) != expected_file_sha:
            raise CanonicalScreeningError(f"{label} file SHA256 mismatch")
        return path

    probe_path = bound_path("path", "sha256", "ArcFace execution probe")
    claim_path = bound_path(
        "bootstrap_claim_path",
        "bootstrap_claim_file_sha256",
        "ArcFace bootstrap claim",
    )
    result_path = bound_path(
        "bootstrap_result_path",
        "bootstrap_result_file_sha256",
        "ArcFace bootstrap result",
    )
    normalized = {
        "path": str(probe_path),
        "sha256": _require_sha256(
            binding["sha256"], "ArcFace execution probe SHA256"
        ),
        "bootstrap_claim_path": str(claim_path),
        "bootstrap_claim_sha256": _require_sha256(
            binding["bootstrap_claim_sha256"],
            "ArcFace bootstrap claim canonical SHA256",
        ),
        "bootstrap_claim_file_sha256": _require_sha256(
            binding["bootstrap_claim_file_sha256"],
            "ArcFace bootstrap claim file SHA256",
        ),
        "bootstrap_result_path": str(result_path),
        "bootstrap_result_sha256": _require_sha256(
            binding["bootstrap_result_sha256"],
            "ArcFace bootstrap result canonical SHA256",
        ),
        "bootstrap_result_file_sha256": _require_sha256(
            binding["bootstrap_result_file_sha256"],
            "ArcFace bootstrap result file SHA256",
        ),
    }
    probe = load_json(probe_path, "ArcFace execution probe")
    execution = probe.get("execution")
    if not isinstance(execution, Mapping):
        raise CanonicalScreeningError(
            "ArcFace execution probe omits execution"
        )
    declared = dict(arcface_contract)
    declared["execution"] = dict(execution)
    declared["execution_probe"] = normalized
    try:
        from safa.evaluation.r9_evaluator_worker import (
            R9EvaluatorError,
            _validate_arcface_contract,
        )

        validated = _validate_arcface_contract(
            declared, repo_root=repo_root.resolve()
        )
    except (FileNotFoundError, R9EvaluatorError) as exc:
        raise CanonicalScreeningError(str(exc)) from exc
    return dict(validated["execution_probe"])


def _validate_ea7_smoke_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "classification",
            "phase",
            "request_count",
            "primary_failed_count",
            "repeat_result_count",
            "screen512_result_count",
            "generated_png_count",
            "failed_summary",
            "run_requests",
            "run_claims",
            "failed_results",
            "worker_logs",
            "resource_monitor",
            "runtime_resource_windows",
        },
        "supersession evidence",
    )
    if (
        supersedes["policy_sha256"]
        != "ea7ae71fd662526b9a45bf3cc6d283884aefc380b292c8f273169a35f42ffc28"
        or supersedes["classification"] != "started_incomplete"
        or supersedes["phase"] != "smoke8"
        or supersedes["request_count"] != 386
        or supersedes["primary_failed_count"] != 8
        or supersedes["repeat_result_count"] != 0
        or supersedes["screen512_result_count"] != 0
        or supersedes["generated_png_count"] != 0
    ):
        raise CanonicalScreeningError("canonical supersession status differs")
    root = repo_root.resolve()
    failed_results = supersedes["failed_results"]
    run_requests = supersedes["run_requests"]
    run_claims = supersedes["run_claims"]
    worker_logs = supersedes["worker_logs"]
    if (
        not isinstance(run_requests, list)
        or len(run_requests) != 8
        or not isinstance(run_claims, list)
        or len(run_claims) != 8
        or not isinstance(failed_results, list)
        or len(failed_results) != 8
        or not isinstance(worker_logs, list)
        or len(worker_logs) != 8
    ):
        raise CanonicalScreeningError(
            "smoke supersession must bind eight results and eight logs"
        )
    bound = {
        "policy_sha256": supersedes["policy_sha256"],
        "classification": "started_incomplete",
        "phase": "smoke8",
        "request_count": 386,
        "primary_failed_count": 8,
        "repeat_result_count": 0,
        "screen512_result_count": 0,
        "generated_png_count": 0,
        "failed_summary": _validate_bound_file(
            root, supersedes["failed_summary"], "supersession failed summary"
        ),
        "run_requests": [
            _validate_bound_file(root, value, "supersession run request")
            for value in run_requests
        ],
        "run_claims": [
            _validate_bound_file(root, value, "supersession run claim")
            for value in run_claims
        ],
        "failed_results": [
            _validate_bound_file(root, value, "supersession failed result")
            for value in failed_results
        ],
        "worker_logs": [
            _validate_bound_file(root, value, "supersession worker log")
            for value in worker_logs
        ],
        "resource_monitor": _validate_bound_file(
            root, supersedes["resource_monitor"], "supersession resource monitor"
        ),
        "runtime_resource_windows": _validate_bound_file(
            root,
            supersedes["runtime_resource_windows"],
            "supersession runtime resource windows",
        ),
    }
    summary = load_json(
        Path(bound["failed_summary"]["path"]),
        "superseded smoke failed summary",
    )
    if (
        summary.get("phase") != "smoke8"
        or summary.get("reason") != "worker_nonzero_exit"
        or summary.get("failures")
        != [
            f"{binding['path']}: exit_code=1"
            for binding in bound["run_requests"]
        ]
        or summary.get("monitor_log") != bound["resource_monitor"]
        or _require_mapping(
            summary.get("runtime_resource_guard"),
            "superseded runtime resource guard",
        ).get("samples")
        != bound["runtime_resource_windows"]
    ):
        raise CanonicalScreeningError("superseded smoke summary semantics differ")
    failure_message = (
        "The size of tensor a (4) must match the size of tensor b (3) "
        "at non-singleton dimension 1"
    )
    candidate_ids: list[str] = []
    for request_binding, claim_binding, result_binding in zip(
        bound["run_requests"],
        bound["run_claims"],
        bound["failed_results"],
        strict=True,
    ):
        request = load_json(
            Path(request_binding["path"]), "superseded smoke run request"
        )
        claim = load_json(
            Path(claim_binding["path"]), "superseded smoke run claim"
        )
        result = load_json(
            Path(result_binding["path"]), "superseded smoke failed result"
        )
        failure = _require_mapping(result.get("failure"), "smoke result failure")
        candidate = _require_mapping(
            request.get("candidate"), "superseded request candidate"
        )
        candidate_id = candidate.get("candidate_id")
        if (
            request.get("contract_type") != RUN_REQUEST_CONTRACT
            or request.get("mode") != "smoke8"
            or request.get("replicate") != "primary"
            or request.get("sample_count") != 8
            or request.get("batch_size") != 2
            or request.get("seed") != 4549
            or _require_mapping(
                request.get("policy"), "superseded request policy"
            ).get("canonical_sha256")
            != supersedes["policy_sha256"]
            or request.get("run_request_sha256")
            != canonical_digest(request, "run_request_sha256")
            or not isinstance(candidate_id, str)
            or not candidate_id
            or claim.get("contract_type") != RUN_CLAIM_CONTRACT
            or claim.get("run_request_sha256")
            != request.get("run_request_sha256")
            or claim.get("run_claim_sha256")
            != canonical_digest(claim, "run_claim_sha256")
            or result.get("run_request_sha256")
            != request.get("run_request_sha256")
            or result.get("run_claim_sha256") != claim.get("run_claim_sha256")
            or result.get("run_result_sha256")
            != canonical_digest(result, "run_result_sha256")
            or result.get("status") != "failed"
            or failure.get("type") != "RuntimeError"
            or failure.get("message") != failure_message
        ):
            raise CanonicalScreeningError(
                "superseded smoke result semantics differ"
            )
        candidate_ids.append(candidate_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CanonicalScreeningError(
            "superseded smoke candidate IDs are not unique"
        )
    campaign_policy_root = (
        Path(bound["run_requests"][0]["path"]).resolve().parents[2]
    )
    primary_requests = list(
        (campaign_policy_root / "run_requests" / "smoke8_primary").glob("*.json")
    )
    repeat_requests = list(
        (campaign_policy_root / "run_requests" / "smoke8_repeat").glob("*.json")
    )
    repeat_results = list(
        (campaign_policy_root / "runs" / "smoke8_repeat").glob("*/result.json")
    )
    primary_results = list(
        (campaign_policy_root / "runs" / "smoke8_primary").glob("*/result.json")
    )
    screen512_results = list(
        (campaign_policy_root / "runs" / "screen512_primary").glob("*/result.json")
    )
    generated_pngs = list((campaign_policy_root / "runs").glob("**/*.png"))
    if (
        len(primary_requests) + len(repeat_requests) != 386
        or len(primary_requests) != 193
        or len(repeat_requests) != 193
        or len(primary_results) != 8
        or len(repeat_results) != 0
        or len(screen512_results) != 0
        or len(generated_pngs) != 0
    ):
        raise CanonicalScreeningError(
            "superseded smoke filesystem counts differ"
        )
    resource_guard = _require_mapping(
        summary.get("runtime_resource_guard"),
        "superseded runtime resource guard",
    )
    if (
        resource_guard.get("violated") is not False
        or resource_guard.get("violation_reason") is not None
        or resource_guard.get("thread_failure") is not None
        or resource_guard.get("final_cpu_consecutive_high") != 0
        or resource_guard.get("final_swap_consecutive_io") != 0
    ):
        raise CanonicalScreeningError(
            "superseded smoke resource guard semantics differ"
        )
    for log_binding in bound["worker_logs"]:
        if failure_message not in Path(log_binding["path"]).read_text(
            encoding="utf-8"
        ):
            raise CanonicalScreeningError(
                "superseded smoke worker log semantics differ"
            )
    return bound


def _validate_c83_preflight_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "c83 supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "previous_policy_sha256",
            "classification",
            "phase",
            "request_count",
            "result_count",
            "valid_count",
            "false_invalid_count",
            "pending_count",
            "attempt_claim_count",
            "attempt_terminal_count",
            "false_invalid_failure_code",
            "false_invalid_checkpoint_sha256",
            "scientific_result_reuse",
            "successor_execution",
            "ram_budget_source_policy_sha256",
            "evidence_root",
            "controller_claim",
            "controller_terminal",
            "wrapper_claim",
            "wrapper_exit",
            "resource_monitor",
            "resource_observer",
            "runtime_resource_windows",
        },
        "c83 supersession evidence",
    )
    c83 = "c83b95e0ca49a0cf5b5b3d67c337000a31b8d2a3299d434fc1051256f18fea50"
    ea7 = "ea7ae71fd662526b9a45bf3cc6d283884aefc380b292c8f273169a35f42ffc28"
    false_invalid_sha256 = [
        "1058090b828d8e0243c3ccc9563526e40407a1554fa128e9169fb1c2b546f6b4",
        "1244b3a3aa790f6fcb941158acf4a9088d035b2309b24248e15470410140c684",
        "1c2cd0a53f837c3e20e6be07e3a2e7860e4bed3e0cf5ec50b47dfb99fb5a2f35",
        "1d199b8a32ba9b1a6225157416674aa0bcd05766765a3c619fcc25a97178b3cb",
        "33a4c9bfb46080e8a75c503c269e21e2c6d436913d2a42db7ed0ea74ea31ce9d",
    ]
    expected_scalars = {
        "policy_sha256": c83,
        "previous_policy_sha256": ea7,
        "classification": "started_incomplete",
        "phase": "preflight",
        "request_count": 193,
        "result_count": 34,
        "valid_count": 29,
        "false_invalid_count": 5,
        "pending_count": 159,
        "attempt_claim_count": 34,
        "attempt_terminal_count": 34,
        "false_invalid_failure_code": "loaded_output_capability_mismatch",
        "false_invalid_checkpoint_sha256": false_invalid_sha256,
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
    }
    if any(supersedes[key] != value for key, value in expected_scalars.items()):
        raise CanonicalScreeningError("c83 supersession status differs")

    evidence_root = _require_mapping(
        supersedes["evidence_root"], "c83 evidence root"
    )
    _require_exact_keys(
        evidence_root,
        {"path", "digest", "digest_algorithm"},
        "c83 evidence root",
    )
    root = _repo_path(
        repo_root,
        evidence_root["path"],
        "c83 evidence root",
        must_exist=False,
    )
    expected_root = (
        repo_root.resolve()
        / "artifacts/closeout/historical-canonical-512-v1/by_policy"
        / c83
    ).resolve()
    if (
        root != expected_root
        or not root.is_dir()
        or evidence_root["digest_algorithm"]
        != "sha256_relative_posix_nul_content_nul_v1"
        or sha256_directory_tree(root)
        != _require_sha256(evidence_root["digest"], "c83 evidence root digest")
    ):
        raise CanonicalScreeningError("c83 evidence root binding differs")

    bound_files = {
        name: _validate_bound_file(
            repo_root, supersedes[name], f"c83 {name.replace('_', ' ')}"
        )
        for name in (
            "controller_claim",
            "controller_terminal",
            "wrapper_claim",
            "wrapper_exit",
            "resource_monitor",
            "resource_observer",
            "runtime_resource_windows",
        )
    }
    expected_relative_paths = {
        "controller_claim": "preflight_control/controller_claim.json",
        "controller_terminal": "preflight_control/controller_terminal.json",
        "wrapper_claim": "preflight_control/wrapper_claim.json",
        "wrapper_exit": "preflight_control/wrapper_exit.json",
        "resource_monitor": "logs/preflight__monitor.jsonl",
        "resource_observer": "logs/preflight__observer.jsonl",
        "runtime_resource_windows": (
            "preflight_control/runtime_resource_windows.jsonl"
        ),
    }
    for name, relative_path in expected_relative_paths.items():
        if Path(bound_files[name]["path"]).resolve() != (
            root / relative_path
        ).resolve():
            raise CanonicalScreeningError(
                f"c83 {name.replace('_', ' ')} path differs"
            )
    controller_claim = load_json(
        Path(bound_files["controller_claim"]["path"]), "c83 controller claim"
    )
    controller_terminal = load_json(
        Path(bound_files["controller_terminal"]["path"]), "c83 controller terminal"
    )
    wrapper_claim = load_json(
        Path(bound_files["wrapper_claim"]["path"]), "c83 wrapper claim"
    )
    wrapper_exit = load_json(
        Path(bound_files["wrapper_exit"]["path"]), "c83 wrapper exit"
    )
    expected_config = (
        repo_root.resolve()
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    wrapper_config = _require_mapping(
        wrapper_claim.get("config"), "c83 wrapper config"
    )
    wrapper_command = wrapper_claim.get("command")
    guard = _require_mapping(
        controller_terminal.get("runtime_resource_guard"),
        "c83 runtime resource guard",
    )
    if (
        controller_claim.get("contract_type")
        != "safa_canonical_preflight_controller_claim_v1"
        or controller_claim.get("policy_sha256") != c83
        or controller_claim.get("supersedes_policy_sha256") != ea7
        or controller_claim.get("request_count") != 193
        or controller_claim.get("controller_claim_sha256")
        != canonical_digest(controller_claim, "controller_claim_sha256")
        or controller_terminal.get("contract_type")
        != "safa_canonical_preflight_controller_terminal_v1"
        or controller_terminal.get("policy_sha256") != c83
        or controller_terminal.get("controller_claim_sha256")
        != controller_claim.get("controller_claim_sha256")
        or controller_terminal.get("status") != "failed"
        or controller_terminal.get("result_count") != 34
        or controller_terminal.get("pending_count") != 159
        or controller_terminal.get("attempt_claim_count") != 34
        or controller_terminal.get("attempt_terminal_count") != 34
        or controller_terminal.get("failure")
        != {"type": "KeyboardInterrupt", "message": ""}
        or controller_terminal.get("controller_terminal_sha256")
        != canonical_digest(
            controller_terminal, "controller_terminal_sha256"
        )
        or controller_terminal.get("controller_monitor_samples")
        != bound_files["resource_monitor"]
        or guard.get("violated") is not False
        or guard.get("violation_reason") is not None
        or guard.get("thread_failure") is not None
        or guard.get("samples") != bound_files["runtime_resource_windows"]
        or wrapper_claim.get("contract_type")
        != "safa_canonical_preflight_wrapper_claim_v1"
        or wrapper_claim.get("policy_sha256") != c83
        or wrapper_claim.get("external_timeout_seconds") is not None
        or Path(str(wrapper_config.get("path"))).resolve() != expected_config
        or wrapper_config.get("sha256")
        != "82d596e02c021a064a0a2156524773bd75599acfeb77ac981902ad9cf9745561"
        or not isinstance(wrapper_command, list)
        or wrapper_command[-2:] != ["preflight", "--execute"]
        or wrapper_claim.get("wrapper_claim_sha256")
        != canonical_digest(wrapper_claim, "wrapper_claim_sha256")
        or wrapper_exit.get("contract_type")
        != "safa_canonical_preflight_wrapper_exit_v2"
        or wrapper_exit.get("policy_sha256") != c83
        or wrapper_exit.get("wrapper_claim_sha256")
        != wrapper_claim.get("wrapper_claim_sha256")
        or wrapper_exit.get("command") != wrapper_command
        or wrapper_exit.get("started_at") != wrapper_claim.get("started_at")
        or wrapper_exit.get("exit_code") != 125
        or wrapper_exit.get("launch_failure")
        != {"type": "KeyboardInterrupt", "message": ""}
        or wrapper_exit.get("wrapper_exit_sha256")
        != canonical_digest(wrapper_exit, "wrapper_exit_sha256")
        or wrapper_exit.get("controller_claim") != bound_files["controller_claim"]
        or wrapper_exit.get("controller_terminal")
        != bound_files["controller_terminal"]
    ):
        raise CanonicalScreeningError("c83 stop evidence semantics differ")
    monitor_rows = load_jsonl(
        Path(bound_files["resource_monitor"]["path"]), "c83 resource monitor"
    )
    observer_rows = load_jsonl(
        Path(bound_files["resource_observer"]["path"]), "c83 resource observer"
    )
    if (
        not monitor_rows
        or not observer_rows
        or any(
            row.get("policy_sha256") != c83
            or row.get("phase") != "preflight"
            or row.get("contract_type")
            != "safa_canonical_resource_monitor_sample_v1"
            for row in [*monitor_rows, *observer_rows]
        )
        or observer_rows[-1].get("terminal") is not True
        or observer_rows[-1].get("artifacts", {}).get("preflight_requests")
        != 193
        or observer_rows[-1].get("artifacts", {}).get("preflight_results")
        != 34
    ):
        raise CanonicalScreeningError("c83 resource evidence semantics differ")

    requests = sorted((root / "checkpoint_preflight/requests").glob("*.json"))
    results = sorted((root / "checkpoint_preflight/results").glob("*.json"))
    claims = sorted((root / "preflight_control/attempts").glob("*.claim.json"))
    terminals = sorted(
        (root / "preflight_control/attempts").glob("*.terminal.json")
    )
    if (
        len(requests) != 193
        or len(results) != 34
        or len(claims) != 34
        or len(terminals) != 34
    ):
        raise CanonicalScreeningError("c83 preflight filesystem counts differ")

    request_by_stem: dict[str, dict[str, Any]] = {}
    request_sha256: set[str] = set()
    for path in requests:
        request = load_json(path, "c83 preflight request")
        stem = path.stem
        if (
            request.get("contract_type") != PREFLIGHT_REQUEST_CONTRACT
            or request.get("policy_sha256") != c83
            or request.get("preflight_request_sha256")
            != canonical_digest(request, "preflight_request_sha256")
            or stem
            != f"{request.get('checkpoint_sha256')}__{request.get('checkpoint_model')}"
            or request.get("preflight_request_sha256") in request_sha256
        ):
            raise CanonicalScreeningError("c83 preflight request semantics differ")
        request_sha256.add(str(request["preflight_request_sha256"]))
        request_by_stem[stem] = request

    valid_count = 0
    invalid_sha256: list[str] = []
    for result_path in results:
        stem = result_path.stem
        request = request_by_stem.get(stem)
        result = load_json(result_path, "c83 preflight result")
        strict_result = _require_mapping(
            result.get("strict_result"), "c83 strict preflight result"
        )
        claim_path = root / "preflight_control/attempts" / f"{stem}.claim.json"
        terminal_path = (
            root / "preflight_control/attempts" / f"{stem}.terminal.json"
        )
        if request is None or not claim_path.is_file() or not terminal_path.is_file():
            raise CanonicalScreeningError("c83 preflight chain is incomplete")
        claim = load_json(claim_path, "c83 preflight attempt claim")
        terminal = load_json(terminal_path, "c83 preflight attempt terminal")
        status = strict_result.get("status")
        if status == "valid":
            valid_count += 1
        elif (
            status == "invalid"
            and strict_result.get("failure_code")
            == "loaded_output_capability_mismatch"
            and strict_result.get("failure_message")
            == (
                "strict-loaded generator output capability mismatch: "
                "GeneratorOutputContractError: strict-loaded generator "
                "canonical config digest differs"
            )
            and strict_result.get("tensor_count") == 139
            and strict_result.get("finite_tensor_count") == 139
            and strict_result.get("nonfinite_keys") == []
            and strict_result.get("missing_keys") == []
            and strict_result.get("unexpected_keys") == []
            and strict_result.get("shape_mismatches") == []
        ):
            invalid_sha256.append(str(result.get("checkpoint_sha256")))
        else:
            raise CanonicalScreeningError("c83 preflight status semantics differ")
        if (
            result.get("contract_type") != PREFLIGHT_RESULT_CONTRACT
            or result.get("policy_sha256") != c83
            or result.get("preflight_request_sha256")
            != request.get("preflight_request_sha256")
            or result.get("preflight_result_sha256")
            != canonical_digest(result, "preflight_result_sha256")
            or claim.get("contract_type")
            != "safa_canonical_preflight_attempt_claim_v1"
            or claim.get("policy_sha256") != c83
            or claim.get("preflight_request_sha256")
            != request.get("preflight_request_sha256")
            or claim.get("attempt_claim_sha256")
            != canonical_digest(claim, "attempt_claim_sha256")
            or terminal.get("contract_type")
            != "safa_canonical_preflight_attempt_terminal_v1"
            or terminal.get("policy_sha256") != c83
            or terminal.get("attempt_claim_sha256")
            != claim.get("attempt_claim_sha256")
            or terminal.get("status") != "completed"
            or terminal.get("valid") is not (status == "valid")
            or terminal.get("preflight_result_sha256")
            != result.get("preflight_result_sha256")
            or terminal.get("result_file_sha256") != sha256_file(result_path)
            or terminal.get("attempt_terminal_sha256")
            != canonical_digest(terminal, "attempt_terminal_sha256")
        ):
            raise CanonicalScreeningError("c83 preflight chain semantics differ")
    if valid_count != 29 or sorted(invalid_sha256) != false_invalid_sha256:
        raise CanonicalScreeningError("c83 valid/false-invalid classification differs")

    return {
        **expected_scalars,
        "evidence_root": {
            "path": str(root),
            "digest": evidence_root["digest"],
            "digest_algorithm": evidence_root["digest_algorithm"],
        },
        **bound_files,
    }


def _validate_310_preflight_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "310 supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "previous_policy_sha256",
            "classification",
            "phase",
            "stage",
            "request_count",
            "result_count",
            "pending_count",
            "checkpoint_attempt_claim_count",
            "checkpoint_attempt_terminal_count",
            "wrapper_claim_count",
            "generated_png_count",
            "startup_cpu_observed_percent",
            "superseded_cpu_admission_percent",
            "successor_cpu_admission_percent",
            "failure_message",
            "scientific_result_reuse",
            "successor_execution",
            "ram_budget_source_policy_sha256",
            "evidence_root",
            "checkpoint_plan",
            "wrapper_claim",
            "wrapper_exit",
            "controller_process_log",
            "resource_observer",
        },
        "310 supersession evidence",
    )
    policy_sha256 = (
        "310f5b539315d3bc957530856c0f810bf5b32afc97469fdb9467bf3facdc9cda"
    )
    previous_policy_sha256 = (
        "c83b95e0ca49a0cf5b5b3d67c337000a31b8d2a3299d434fc1051256f18fea50"
    )
    failure_message = "CPU admission failed: 89.10% >= 85%"
    expected_scalars = {
        "policy_sha256": policy_sha256,
        "previous_policy_sha256": previous_policy_sha256,
        "classification": "started_incomplete",
        "phase": "preflight",
        "stage": "startup_admission_before_controller_claim",
        "request_count": 193,
        "result_count": 0,
        "pending_count": 193,
        "checkpoint_attempt_claim_count": 0,
        "checkpoint_attempt_terminal_count": 0,
        "wrapper_claim_count": 1,
        "generated_png_count": 0,
        "startup_cpu_observed_percent": 89.10,
        "superseded_cpu_admission_percent": 85,
        "successor_cpu_admission_percent": 90,
        "failure_message": failure_message,
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
        "ram_budget_source_policy_sha256": (
            "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
        ),
    }
    if any(supersedes[key] != value for key, value in expected_scalars.items()):
        raise CanonicalScreeningError("310 supersession status differs")

    evidence_root = _require_mapping(
        supersedes["evidence_root"], "310 evidence root"
    )
    _require_exact_keys(
        evidence_root,
        {"path", "digest", "digest_algorithm"},
        "310 evidence root",
    )
    root = _repo_path(
        repo_root,
        evidence_root["path"],
        "310 evidence root",
        must_exist=False,
    )
    expected_root = (
        repo_root.resolve()
        / "artifacts/closeout/historical-canonical-512-v1/by_policy"
        / policy_sha256
    ).resolve()
    if (
        root != expected_root
        or not root.is_dir()
        or evidence_root["digest_algorithm"]
        != "sha256_relative_posix_nul_content_nul_v1"
        or sha256_directory_tree(root)
        != _require_sha256(evidence_root["digest"], "310 evidence root digest")
    ):
        raise CanonicalScreeningError("310 evidence root binding differs")

    expected_relative_paths = {
        "checkpoint_plan": "checkpoint_plan.json",
        "wrapper_claim": "preflight_control/wrapper_claim.json",
        "wrapper_exit": "preflight_control/wrapper_exit.json",
        "controller_process_log": "preflight_control/controller_process.log",
        "resource_observer": "logs/preflight__observer.jsonl",
    }
    bound_files = {
        name: _validate_bound_file(
            repo_root, supersedes[name], f"310 {name.replace('_', ' ')}"
        )
        for name in expected_relative_paths
    }
    for name, relative_path in expected_relative_paths.items():
        if Path(bound_files[name]["path"]).resolve() != (
            root / relative_path
        ).resolve():
            raise CanonicalScreeningError(
                f"310 {name.replace('_', ' ')} path differs"
            )

    wrapper_claim = load_json(
        Path(bound_files["wrapper_claim"]["path"]), "310 wrapper claim"
    )
    wrapper_exit = load_json(
        Path(bound_files["wrapper_exit"]["path"]), "310 wrapper exit"
    )
    wrapper_config = _require_mapping(
        wrapper_claim.get("config"), "310 wrapper config"
    )
    wrapper_command = wrapper_claim.get("command")
    if (
        wrapper_claim.get("contract_type")
        != "safa_canonical_preflight_wrapper_claim_v1"
        or wrapper_claim.get("policy_sha256") != policy_sha256
        or wrapper_claim.get("external_timeout_seconds") is not None
        or wrapper_config.get("sha256")
        != "6479b5a207fefdf331f8c988b3b9bc456cbe2d872cba4dfd5bb0491048f845ee"
        or Path(str(wrapper_config.get("path"))).resolve()
        != (
            repo_root.resolve()
            / "configs/closeout/canonical_screening_512_v1.json"
        )
        or not isinstance(wrapper_command, list)
        or wrapper_command[-2:] != ["preflight", "--execute"]
        or wrapper_claim.get("wrapper_claim_sha256")
        != canonical_digest(wrapper_claim, "wrapper_claim_sha256")
        or wrapper_exit.get("contract_type")
        != "safa_canonical_preflight_wrapper_exit_v2"
        or wrapper_exit.get("policy_sha256") != policy_sha256
        or wrapper_exit.get("wrapper_claim_sha256")
        != wrapper_claim.get("wrapper_claim_sha256")
        or wrapper_exit.get("command") != wrapper_command
        or wrapper_exit.get("started_at") != wrapper_claim.get("started_at")
        or wrapper_exit.get("exit_code") != 2
        or wrapper_exit.get("signal") is not None
        or wrapper_exit.get("launch_failure") is not None
        or wrapper_exit.get("controller_claim") is not None
        or wrapper_exit.get("controller_terminal") is not None
        or wrapper_exit.get("controller_process_log")
        != bound_files["controller_process_log"]
        or wrapper_exit.get("wrapper_exit_sha256")
        != canonical_digest(wrapper_exit, "wrapper_exit_sha256")
    ):
        raise CanonicalScreeningError("310 wrapper stop evidence semantics differ")
    if Path(bound_files["controller_process_log"]["path"]).read_text(
        encoding="utf-8"
    ) != f"CANONICAL SCREENING BLOCKED: {failure_message}\n":
        raise CanonicalScreeningError("310 admission failure log differs")

    requests = sorted((root / "checkpoint_preflight/requests").glob("*.json"))
    results = list((root / "checkpoint_preflight/results").glob("*.json"))
    attempts = root / "preflight_control/attempts"
    checkpoint_claims = list(attempts.glob("*.claim.json"))
    checkpoint_terminals = list(attempts.glob("*.terminal.json"))
    generated_pngs = list(root.glob("**/*.png"))
    request_digests: set[str] = set()
    if (
        len(requests) != 193
        or results
        or checkpoint_claims
        or checkpoint_terminals
        or generated_pngs
    ):
        raise CanonicalScreeningError("310 preflight filesystem counts differ")
    for path in requests:
        request = load_json(path, "310 preflight request")
        request_digest = request.get("preflight_request_sha256")
        if (
            request.get("contract_type") != PREFLIGHT_REQUEST_CONTRACT
            or request.get("policy_sha256") != policy_sha256
            or request_digest
            != canonical_digest(request, "preflight_request_sha256")
            or path.stem
            != f"{request.get('checkpoint_sha256')}__{request.get('checkpoint_model')}"
            or request_digest in request_digests
        ):
            raise CanonicalScreeningError("310 preflight request semantics differ")
        request_digests.add(str(request_digest))

    observer_rows = load_jsonl(
        Path(bound_files["resource_observer"]["path"]), "310 resource observer"
    )
    if (
        not observer_rows
        or any(
            row.get("policy_sha256") != policy_sha256
            or row.get("phase") != "preflight"
            or row.get("contract_type")
            != "safa_canonical_resource_monitor_sample_v1"
            or row.get("gpus") is not None
            or row.get("compute_processes") is not None
            for row in observer_rows
        )
        or observer_rows[-1].get("terminal") is not True
        or observer_rows[-1].get("artifacts", {}).get("preflight_requests")
        != 193
        or observer_rows[-1].get("artifacts", {}).get("preflight_results")
        != 0
    ):
        raise CanonicalScreeningError("310 resource observer semantics differ")

    return {
        **expected_scalars,
        "evidence_root": {
            "path": str(root),
            "digest": evidence_root["digest"],
            "digest_algorithm": evidence_root["digest_algorithm"],
        },
        **bound_files,
    }


def _validate_5d_preflight_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "5d supersession evidence")
    file_fields = {
        "controller_claim",
        "controller_terminal",
        "controller_summary",
        "wrapper_claim",
        "wrapper_exit",
        "resource_monitor",
        "resource_observer",
        "runtime_resource_windows",
        "startup_admission",
        "final_plan",
        "candidate_manifest",
    }
    scalar_fields = {
        "policy_sha256",
        "previous_policy_sha256",
        "classification",
        "phase",
        "stage",
        "request_count",
        "result_count",
        "valid_count",
        "invalid_count",
        "reused_count",
        "pending_count",
        "attempt_claim_count",
        "attempt_terminal_count",
        "candidate_count",
        "run_request_count",
        "generated_png_count",
        "supersession_reason",
        "scientific_result_reuse",
        "successor_execution",
    }
    _require_exact_keys(
        supersedes,
        scalar_fields | file_fields | {"evidence_root"},
        "5d supersession evidence",
    )
    policy_sha256 = (
        "5d51185345983fbf9bc2924f43d5a4b671674398581824753c0c155c4cdda2db"
    )
    previous_policy_sha256 = (
        "310f5b539315d3bc957530856c0f810bf5b32afc97469fdb9467bf3facdc9cda"
    )
    expected_scalars = {
        "policy_sha256": policy_sha256,
        "previous_policy_sha256": previous_policy_sha256,
        "classification": "completed_preflight_superseded",
        "phase": "preflight",
        "stage": "completed_preflight_candidate_manifest_prepared_before_gpu",
        "request_count": 193,
        "result_count": 193,
        "valid_count": 193,
        "invalid_count": 0,
        "reused_count": 0,
        "pending_count": 0,
        "attempt_claim_count": 193,
        "attempt_terminal_count": 193,
        "candidate_count": 193,
        "run_request_count": 0,
        "generated_png_count": 0,
        "supersession_reason": "implementation_and_policy_contract_upgrade",
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
    }
    if any(supersedes[key] != value for key, value in expected_scalars.items()):
        raise CanonicalScreeningError("5d supersession status differs")

    evidence_root = _require_mapping(
        supersedes["evidence_root"], "5d evidence root"
    )
    _require_exact_keys(
        evidence_root,
        {"path", "digest", "digest_algorithm"},
        "5d evidence root",
    )
    root = _repo_path(
        repo_root, evidence_root["path"], "5d evidence root", must_exist=False
    )
    expected_root = (
        repo_root.resolve()
        / "artifacts/closeout/historical-canonical-512-v1/by_policy"
        / policy_sha256
    ).resolve()
    if (
        root != expected_root
        or not root.is_dir()
        or evidence_root["digest_algorithm"]
        != "sha256_relative_posix_nul_content_nul_v1"
        or evidence_root["digest"]
        != "7a1c0fba3e7a50b748854d58987a0f21412c1f31849abea87c4f5ef639ecb60e"
        or sha256_directory_tree(root) != evidence_root["digest"]
    ):
        raise CanonicalScreeningError("5d evidence root binding differs")
    bound_files = {
        name: _validate_bound_file(
            repo_root, supersedes[name], f"5d {name.replace('_', ' ')}"
        )
        for name in file_fields
    }
    expected_paths = {
        "controller_claim": root / "preflight_control/controller_claim.json",
        "controller_terminal": root
        / "preflight_control/controller_terminal.json",
        "controller_summary": root
        / "preflight_control/controller_summary.json",
        "wrapper_claim": root / "preflight_control/wrapper_claim.json",
        "wrapper_exit": root / "preflight_control/wrapper_exit.json",
        "resource_monitor": root / "logs/preflight__monitor.jsonl",
        "resource_observer": root / "logs/preflight__observer.jsonl",
        "runtime_resource_windows": root
        / "preflight_control/runtime_resource_windows.jsonl",
        "startup_admission": root
        / "admissions/preflight_cpu_startup__"
        "057d4a37181d2968966ba4934818888592e286baaf2f398ebd828f2471b606d9.json",
        "final_plan": repo_root.resolve()
        / "artifacts/closeout/historical-canonical-512-v1/"
        "checkpoint_plan_final__5d51185345983fbf.json",
        "candidate_manifest": repo_root.resolve()
        / "artifacts/closeout/historical-canonical-512-v1/"
        "candidate_manifest__5d51185345983fbf.json",
    }
    if any(
        Path(bound_files[name]["path"]).resolve() != path.resolve()
        for name, path in expected_paths.items()
    ):
        raise CanonicalScreeningError("5d bound evidence path differs")
    expected_file_sha256 = {
        "controller_claim": (
            "88b19231e5d430c112c70cc406c01a7470967d9fde5d0e9fab9c7c1040ddee61"
        ),
        "controller_terminal": (
            "4a2eb20eefe7f2824d6fe8725c91d3cf5498ac9c50c5c1033fce00ce357096eb"
        ),
        "controller_summary": (
            "9c7fbfc40f4255672daf51c11aafc6c683326745d0f4ac96d4ec4eff27efead8"
        ),
        "wrapper_claim": (
            "990d0d2cfa252ff749757898be8d9fa88d38fc0caac21e0368eb10bde7199a4a"
        ),
        "wrapper_exit": (
            "dddd29617cf6194074a1c0104cf26943ea5c560f78c9225435b4336bf2df1735"
        ),
        "resource_monitor": (
            "0ecc2e5dc5c0746c4dfbe77736e618638eaaa8914c234617fe5f69c3e7fcdc0d"
        ),
        "resource_observer": (
            "ad179025f01b92c3409c619f99bc7cfc7d0b027fb586b63783b1691b0ad329e2"
        ),
        "runtime_resource_windows": (
            "6fd8f5b06187a51214192f784f5e5d2a95f31a5f4063abfee500ba7b574fe225"
        ),
        "startup_admission": (
            "a5eef4a438540eba640b9b515b8f70ce7bd9fa9c295ceb91f9eda2d5b5045f76"
        ),
        "final_plan": (
            "48e4545e641c0f5eb6d5de30c69841af1cb5cc7b979ed5f3af8ffc037b1ec102"
        ),
        "candidate_manifest": (
            "485f008f31a44d7a6ed946d7e9dff51344eeffa43a08214b0422efcb4d4d2957"
        ),
    }
    if any(
        bound_files[name]["sha256"] != digest
        for name, digest in expected_file_sha256.items()
    ):
        raise CanonicalScreeningError("5d bound evidence SHA256 differs")

    summary = load_json(
        Path(bound_files["controller_summary"]["path"]),
        "5d controller summary",
    )
    terminal = load_json(
        Path(bound_files["controller_terminal"]["path"]),
        "5d controller terminal",
    )
    wrapper_exit = load_json(
        Path(bound_files["wrapper_exit"]["path"]), "5d wrapper exit"
    )
    final_plan = load_json(
        Path(bound_files["final_plan"]["path"]), "5d final plan"
    )
    manifest = load_json(
        Path(bound_files["candidate_manifest"]["path"]),
        "5d candidate manifest",
    )
    if (
        summary.get("contract_type")
        != "safa_canonical_preflight_controller_summary_v1"
        or summary.get("policy_sha256") != policy_sha256
        or summary.get("preflight")
        != {
            "request_count": 193,
            "completed": 193,
            "valid": 193,
            "invalid": 0,
            "reused": 0,
        }
        or summary.get("counts", {}).get("pending_preflight") != 0
        or summary.get("counts", {}).get("eligible_candidates") != 193
        or summary.get("controller_summary_sha256")
        != canonical_digest(summary, "controller_summary_sha256")
        or terminal.get("status") != "completed"
        or terminal.get("failure") is not None
        or terminal.get("runtime_resource_guard", {}).get("violated") is not False
        or terminal.get("runtime_resource_guard", {}).get("thread_failure")
        is not None
        or wrapper_exit.get("exit_code") != 0
        or wrapper_exit.get("signal") is not None
        or wrapper_exit.get("launch_failure") is not None
        or wrapper_exit.get("policy_sha256") != policy_sha256
    ):
        raise CanonicalScreeningError("5d controller completion semantics differ")
    if (
        final_plan.get("policy_sha256") != policy_sha256
        or final_plan.get("checkpoint_plan_sha256")
        != canonical_digest(final_plan, "checkpoint_plan_sha256")
        or final_plan.get("counts", {}).get("eligible_candidates") != 193
        or final_plan.get("counts", {}).get("pending_preflight") != 0
        or manifest.get("policy_sha256") != policy_sha256
        or manifest.get("candidate_count") != 193
        or manifest.get("candidate_manifest_sha256")
        != canonical_digest(manifest, "candidate_manifest_sha256")
        or manifest.get("checkpoint_plan", {}).get("canonical_sha256")
        != final_plan.get("checkpoint_plan_sha256")
    ):
        raise CanonicalScreeningError("5d plan/manifest semantics differ")

    requests = sorted((root / "checkpoint_preflight/requests").glob("*.json"))
    results = sorted((root / "checkpoint_preflight/results").glob("*.json"))
    claims = sorted((root / "preflight_control/attempts").glob("*.claim.json"))
    terminals = sorted(
        (root / "preflight_control/attempts").glob("*.terminal.json")
    )
    run_requests = list((root / "run_requests").rglob("*.json"))
    generated_png = list((root / "runs").rglob("*.png"))
    if (
        len(requests) != 193
        or len(results) != 193
        or len(claims) != 193
        or len(terminals) != 193
        or run_requests
        or generated_png
    ):
        raise CanonicalScreeningError("5d evidence filesystem counts differ")
    for result_path in results:
        result = load_json(result_path, "5d preflight result")
        if (
            result.get("policy_sha256") != policy_sha256
            or result.get("strict_result", {}).get("status") != "valid"
            or result.get("preflight_result_sha256")
            != canonical_digest(result, "preflight_result_sha256")
        ):
            raise CanonicalScreeningError("5d preflight result semantics differ")
    return {
        **expected_scalars,
        "evidence_root": {
            "path": str(root),
            "digest": evidence_root["digest"],
            "digest_algorithm": evidence_root["digest_algorithm"],
        },
        **bound_files,
    }


def _validate_6b_failed_probe_root_identity(
    repo_root: Path, raw_binding: Mapping[str, Any]
) -> Path:
    binding = _require_mapping(raw_binding, "failed probe root")
    expected_relative = (
        "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__6b088236579f7311"
    )
    declared = binding.get("path")
    unresolved = repo_root / expected_relative
    if (
        declared != expected_relative
        or unresolved.is_symlink()
        or not unresolved.is_dir()
    ):
        raise CanonicalScreeningError("failed RAM probe root identity differs")
    return _repo_path(
        repo_root,
        declared,
        "failed probe root",
        must_exist=False,
    )


def _validate_6b_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "6b supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "preflight",
            "failed_ram_probe",
        },
        "6b supersession evidence",
    )
    policy_sha = (
        "6b088236579f731183e60c7fc1d7bece31089284aaaf13697a73f3fb6cd42072"
    )
    if {
        "policy_sha256": supersedes["policy_sha256"],
        "classification": supersedes["classification"],
        "supersession_reason": supersedes["supersession_reason"],
        "scientific_result_reuse": supersedes["scientific_result_reuse"],
        "successor_execution": supersedes["successor_execution"],
    } != {
        "policy_sha256": policy_sha,
        "classification": "completed_preflight_and_failed_ram_probe_superseded",
        "supersession_reason": (
            "strict_gpu_uuid_representation_and_probe_digest_contract_upgrade"
        ),
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
    }:
        raise CanonicalScreeningError("6b supersession status differs")

    preflight = _require_mapping(supersedes["preflight"], "6b preflight")
    file_fields = {
        "controller_claim",
        "controller_terminal",
        "controller_summary",
        "wrapper_claim",
        "wrapper_exit",
        "resource_monitor",
        "resource_observer",
        "runtime_resource_windows",
        "startup_admission",
        "final_plan",
        "candidate_manifest",
    }
    count_fields = {
        "request_count": 193,
        "result_count": 193,
        "valid_count": 193,
        "invalid_count": 0,
        "attempt_claim_count": 193,
        "attempt_terminal_count": 193,
        "run_request_count": 0,
        "generated_png_count": 0,
    }
    _require_exact_keys(
        preflight,
        set(count_fields) | file_fields | {"evidence_root"},
        "6b preflight",
    )
    if any(preflight[key] != value for key, value in count_fields.items()):
        raise CanonicalScreeningError("6b preflight counts differ")
    root_binding = _require_mapping(
        preflight["evidence_root"], "6b preflight evidence root"
    )
    root = _repo_path(
        repo_root,
        root_binding["path"],
        "6b preflight evidence root",
        must_exist=False,
    )
    if (
        root_binding
        != {
            "path": str(Path(root_binding["path"])),
            "digest": "17ff26ccfa582367554fea3d8821aa6b0adf482078b2c72ec2bd23fcaa5b1d3f",
            "digest_algorithm": "sha256_relative_posix_nul_content_nul_v1",
        }
        or root
        != (
            repo_root.resolve()
            / "artifacts/closeout/historical-canonical-512-v1/by_policy"
            / policy_sha
        ).resolve()
        or sha256_directory_tree(root) != root_binding["digest"]
    ):
        raise CanonicalScreeningError("6b preflight evidence root differs")
    bound = {
        name: _validate_bound_file(repo_root, preflight[name], f"6b {name}")
        for name in file_fields
    }
    summary = load_json(Path(bound["controller_summary"]["path"]), "6b summary")
    terminal = load_json(
        Path(bound["controller_terminal"]["path"]), "6b terminal"
    )
    wrapper = load_json(Path(bound["wrapper_exit"]["path"]), "6b wrapper")
    summary_preflight = _require_mapping(
        summary.get("preflight"), "6b summary preflight"
    )
    if (
        summary.get("policy_sha256") != policy_sha
        or summary_preflight.get("completed") != 193
        or summary_preflight.get("valid") != 193
        or summary_preflight.get("invalid") != 0
        or summary_preflight.get("reused") != 0
        or terminal.get("status") != "completed"
        or terminal.get("failure") is not None
        or terminal.get("result_count") != 193
        or terminal.get("pending_count") != 0
        or wrapper.get("exit_code") != 0
        or wrapper.get("signal") is not None
        or wrapper.get("launch_failure") is not None
    ):
        raise CanonicalScreeningError("6b terminal semantics differ")

    failed = _require_mapping(
        supersedes["failed_ram_probe"], "6b failed RAM probe"
    )
    expected_failed_scalars = {
        "classification": "failed_before_capability_execution",
        "status": "failed",
        "worker_returncode": 1,
        "retry_count": 0,
        "termination": None,
        "capability_step_count": 0,
        "work_directory_present": False,
        "worker_result_present": False,
        "resource_guard_clean": True,
        "root_cause": "observed_uuid_representation_contract_mismatch",
        "reuse": "forbidden",
        "in_place_retry": "forbidden",
    }
    _require_exact_keys(
        failed, set(expected_failed_scalars) | {"root", "files"}, "failed RAM probe"
    )
    if any(failed[key] != value for key, value in expected_failed_scalars.items()):
        raise CanonicalScreeningError("failed RAM probe classification differs")
    failed_root_binding = _require_mapping(failed["root"], "failed probe root")
    failed_root = _validate_6b_failed_probe_root_identity(
        repo_root, failed_root_binding
    )
    if (
        failed_root
        != (
            repo_root.resolve()
            / "artifacts/closeout/historical-canonical-512-v1/"
            "ram_probe__6b088236579f7311"
        ).resolve()
        or
        failed_root_binding.get("digest_algorithm")
        != "sha256_relative_posix_nul_content_nul_v1"
        or failed_root_binding.get("digest")
        != "33a9ff82fbb453753be6f769dfe6f474f6aa58741e1be54f903db5d408d932f4"
        or sha256_directory_tree(failed_root) != failed_root_binding["digest"]
    ):
        raise CanonicalScreeningError("failed RAM probe root differs")
    expected_files = {
        "admission.json": "e2b9328fd349a6b3e17d4a2cc0006461554a92ae6008910e0c32b223aacdec50",
        "input_policy.json": "93e7238c82cff73afee6e7a0c3067743bd17c7d90289f7e3863739820a0767f3",
        "probe_result.json": "c2c2e86e01f262cfa831b45df8648b5f0a24612849821f739591d46809829d22",
        "probe_spec.json": "4d6d5e3ddbb16e10c9fcde92e8a2e01361eb075b78b02eb80c85cd0ccac306d0",
        "runtime_resource_windows.jsonl": "b32ea7d4b589a6140f35c3ae3ab0987a8b08252eee5ef3f9403a31197d46121f",
        "worker.log": "02bc776e082a1759175fd74bf44f5695952d13eaea08f3ccf23f54ec6610a8b3",
    }
    entries = list(failed_root.iterdir())
    if (
        failed.get("files") != expected_files
        or len(entries) != 6
        or any(not item.is_file() or item.is_symlink() for item in entries)
        or {item.name for item in entries} != set(expected_files)
        or any(
            sha256_file(failed_root / name) != digest
            for name, digest in expected_files.items()
        )
    ):
        raise CanonicalScreeningError("failed RAM probe files differ")
    result = load_json(failed_root / "probe_result.json", "failed probe result")
    guard = _require_mapping(
        result.get("runtime_resource_guard"), "failed probe resource guard"
    )
    if (
        result.get("status") != "failed"
        or result.get("worker_returncode") != 1
        or result.get("retry_count") != 0
        or result.get("termination") is not None
        or result.get("worker_result_sha256") is not None
        or result.get("worker_vmhwm_bytes") is not None
        or result.get("ram_budget_basis_bytes") is not None
        or result.get("ram_slot_budget_bytes") is not None
        or guard.get("violated") is not False
        or guard.get("violation_reason") is not None
        or guard.get("thread_failure") is not None
        or "worker runtime CUDA UUID differs" not in (
            failed_root / "worker.log"
        ).read_text(encoding="utf-8")
    ):
        raise CanonicalScreeningError("failed RAM probe semantics differ")
    return dict(supersedes)


def _validate_fe9_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "fe9 supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "preflight",
            "failed_ram_probe",
        },
        "fe9 supersession evidence",
    )
    policy_sha = (
        "fe9b41136f0b9fa31ce210dfaa5500c3f46f071838ed91288878f57073502060"
    )
    if {
        "policy_sha256": supersedes["policy_sha256"],
        "classification": supersedes["classification"],
        "supersession_reason": supersedes["supersession_reason"],
        "scientific_result_reuse": supersedes["scientific_result_reuse"],
        "successor_execution": supersedes["successor_execution"],
    } != {
        "policy_sha256": policy_sha,
        "classification": (
            "completed_preflight_and_incomplete_ram_probe_superseded"
        ),
        "supersession_reason": (
            "admission_payload_and_file_binding_interface_fix"
        ),
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
    }:
        raise CanonicalScreeningError("fe9 supersession status differs")

    preflight = _require_mapping(supersedes["preflight"], "fe9 preflight")
    counts = {
        "request_count": 193,
        "result_count": 193,
        "valid_count": 193,
        "invalid_count": 0,
        "reused_count": 0,
        "attempt_claim_count": 193,
        "attempt_terminal_count": 193,
        "run_request_count": 0,
        "generated_png_count": 0,
    }
    files = {
        "controller_claim",
        "controller_terminal",
        "controller_summary",
        "wrapper_claim",
        "wrapper_exit",
        "resource_monitor",
        "resource_observer",
        "runtime_resource_windows",
        "startup_admission",
        "final_plan",
        "candidate_manifest",
    }
    _require_exact_keys(
        preflight,
        set(counts) | files | {"evidence_root"},
        "fe9 preflight",
    )
    if any(preflight[key] != value for key, value in counts.items()):
        raise CanonicalScreeningError("fe9 preflight counts differ")
    evidence = _require_mapping(
        preflight["evidence_root"], "fe9 preflight evidence root"
    )
    expected_root_relative = (
        "artifacts/closeout/historical-canonical-512-v1/by_policy/" + policy_sha
    )
    if evidence.get("path") != expected_root_relative:
        raise CanonicalScreeningError("fe9 evidence root identity differs")
    unresolved_root = repo_root / expected_root_relative
    if unresolved_root.is_symlink() or not unresolved_root.is_dir():
        raise CanonicalScreeningError("fe9 evidence root identity differs")
    root = unresolved_root.resolve()
    if (
        evidence.get("digest_algorithm")
        != "sha256_relative_posix_nul_content_nul_v1"
        or evidence.get("digest")
        != "520649529263802abc66b827f585ce2fab4191580d3e00a8aef7d5c5914cd17e"
        or sha256_directory_tree(root) != evidence["digest"]
    ):
        raise CanonicalScreeningError("fe9 evidence root differs")
    bound = {
        name: _validate_bound_file(repo_root, preflight[name], f"fe9 {name}")
        for name in files
    }
    summary = load_json(Path(bound["controller_summary"]["path"]), "fe9 summary")
    terminal = load_json(
        Path(bound["controller_terminal"]["path"]), "fe9 terminal"
    )
    wrapper = load_json(Path(bound["wrapper_exit"]["path"]), "fe9 wrapper")
    if (
        summary.get("policy_sha256") != policy_sha
        or summary.get("preflight")
        != {
            "completed": 193,
            "invalid": 0,
            "request_count": 193,
            "reused": 0,
            "valid": 193,
        }
        or terminal.get("status") != "completed"
        or terminal.get("failure") is not None
        or terminal.get("result_count") != 193
        or terminal.get("pending_count") != 0
        or terminal.get("attempt_claim_count") != 193
        or terminal.get("attempt_terminal_count") != 193
        or wrapper.get("exit_code") != 0
        or wrapper.get("signal") is not None
        or wrapper.get("launch_failure") is not None
    ):
        raise CanonicalScreeningError("fe9 preflight terminal semantics differ")

    failed = _require_mapping(
        supersedes["failed_ram_probe"], "fe9 failed RAM probe"
    )
    expected_failed = {
        "classification": "preworker_controller_failure_incomplete",
        "probe_contract_sha256": (
            "ae3e800037eccef5744924ab7d248cf7c3b0ad757c9f1105e2ff9e8f510127fd"
        ),
        "probe_execution_sha256": (
            "c9c0a3f48b5a8f9e011c5b2be45ebcd512e35b0fd2de43e0374b561b2b021eaa"
        ),
        "admission_evidence_sha256": (
            "dd32e5083c6139795dad945dff8dc104b13370ce053e7ebd22f4719c460b4623"
        ),
        "admission_sha256": (
            "0da039aecf9ad5a0d74a7a13d22a0d51c931a3f1f1298a80640d01c9218fd0b9"
        ),
        "root_cause": (
            "diagnosed_build_spec_admission_binding_interface_mismatch"
        ),
        "evidence_level": "code_path_inference",
        "retry_count": 0,
        "worker_started": False,
        "gpu_execution_count": 0,
        "capability_step_count": 0,
        "reuse": "forbidden",
        "in_place_retry": "forbidden",
    }
    _require_exact_keys(
        failed,
        set(expected_failed) | {"root", "files", "forbidden_entries"},
        "fe9 failed RAM probe",
    )
    if any(failed[key] != value for key, value in expected_failed.items()):
        raise CanonicalScreeningError("fe9 failed RAM probe status differs")
    failed_root_binding = _require_mapping(failed["root"], "fe9 failed root")
    expected_failed_relative = (
        "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__fe9b41136f0b9fa3"
    )
    unresolved_failed = repo_root / expected_failed_relative
    if (
        failed_root_binding.get("path") != expected_failed_relative
        or unresolved_failed.is_symlink()
        or not unresolved_failed.is_dir()
    ):
        raise CanonicalScreeningError("fe9 failed root identity differs")
    if (
        failed_root_binding.get("digest_algorithm")
        != "sha256_relative_posix_nul_content_nul_v1"
        or failed_root_binding.get("digest")
        != "88b07f6d41898c735107fcb3c79af4dc0260790531cf64f65faac2087ca6cb5b"
        or sha256_directory_tree(unresolved_failed)
        != failed_root_binding["digest"]
    ):
        raise CanonicalScreeningError("fe9 failed root differs")
    expected_files = {
        "admission.json": (
            "0f7a8d39d08bf510a292d1bdfb10f95018e8d29e934660a880b3803dde450fe2"
        ),
        "input_policy.json": (
            "bb92566cc14562d4b03fdcab741de233842f69b9bd56acdda074c88f5b8f024d"
        ),
    }
    if (
        failed["files"] != expected_files
        or {path.name for path in unresolved_failed.iterdir()} != set(expected_files)
        or any(
            not path.is_file() or path.is_symlink()
            for path in unresolved_failed.iterdir()
        )
        or any(
            sha256_file(unresolved_failed / name) != digest
            for name, digest in expected_files.items()
        )
    ):
        raise CanonicalScreeningError("fe9 failed root files differ")
    forbidden = [
        "controller_claim.json",
        "controller_terminal.json",
        "probe_spec.json",
        "worker.log",
        "worker_result.json",
        "probe_result.json",
        "runtime_resource_windows.jsonl",
        "work",
    ]
    if failed["forbidden_entries"] != forbidden or any(
        (unresolved_failed / name).exists() for name in forbidden
    ):
        raise CanonicalScreeningError("fe9 forbidden probe artifacts exist")
    admission = load_json(unresolved_failed / "admission.json", "fe9 admission")
    registry = canonical_gpu_registry(admission["authorized_gpu_registry"])
    if (
        admission.get("contract_type")
        != "safa_canonical_screening_ram_probe_admission_v2"
        or admission.get("probe_contract_sha256")
        != failed["probe_contract_sha256"]
        or admission.get("probe_execution_sha256")
        != failed["probe_execution_sha256"]
        or admission.get("admission_evidence_sha256")
        != ram_probe_admission_evidence_digest(admission)
        or admission.get("admission_evidence_sha256")
        != failed["admission_evidence_sha256"]
        or admission.get("admission_sha256")
        != canonical_digest(admission, "admission_sha256")
        or admission.get("admission_sha256") != failed["admission_sha256"]
        or registry != admission["authorized_gpu_registry"]
        or admission.get("probe_execution_sha256")
        != ram_probe_execution_digest(
            admission["probe_contract_sha256"],
            registry,
            admission["admission_evidence_sha256"],
        )
        or [row["physical_gpu_index"] for row in registry] != [0, 1, 2, 3]
    ):
        raise CanonicalScreeningError("fe9 admission evidence differs")
    return dict(supersedes)


def _validate_4c5_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "4c5 supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "preflight",
            "failed_ram_probe",
        },
        "4c5 supersession evidence",
    )
    policy_sha = (
        "4c5ecb55501fa6b09b63377e892f1cee3e0140abd2a02859d33b9b33375a1576"
    )
    expected_status = {
        "policy_sha256": policy_sha,
        "classification": "completed_preflight_and_failed_ram_probe_superseded",
        "supersession_reason": (
            "arcface_execution_probe_provenance_binding_upgrade"
        ),
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
    }
    if {
        key: supersedes[key] for key in expected_status
    } != expected_status:
        raise CanonicalScreeningError("4c5 supersession status differs")

    preflight = _require_mapping(supersedes["preflight"], "4c5 preflight")
    preflight_files = {
        "controller_claim",
        "controller_terminal",
        "controller_summary",
        "wrapper_claim",
        "wrapper_exit",
        "resource_monitor",
        "resource_observer",
        "runtime_resource_windows",
        "startup_admission",
        "final_plan",
        "candidate_manifest",
    }
    preflight_counts = {
        "request_count": 193,
        "result_count": 193,
        "valid_count": 193,
        "invalid_count": 0,
        "reused_count": 0,
        "attempt_claim_count": 193,
        "attempt_terminal_count": 193,
        "run_request_count": 0,
        "generated_png_count": 0,
    }
    _require_exact_keys(
        preflight,
        set(preflight_counts) | preflight_files | {"evidence_root"},
        "4c5 preflight",
    )
    if any(
        preflight[key] != value
        for key, value in preflight_counts.items()
    ):
        raise CanonicalScreeningError("4c5 preflight counts differ")
    evidence = _require_mapping(
        preflight["evidence_root"], "4c5 preflight evidence root"
    )
    expected_preflight_path = (
        "artifacts/closeout/historical-canonical-512-v1/by_policy/" + policy_sha
    )
    unresolved_preflight = repo_root.resolve() / expected_preflight_path
    _require_tree_without_symlinks(
        unresolved_preflight, "4c5 preflight evidence root"
    )
    if (
        evidence
        != {
            "path": expected_preflight_path,
            "digest": (
                "abdc04bcc7d706e66a523121f9dcd77db8a1841403796a92009cb7c37431baff"
            ),
            "digest_algorithm": (
                "sha256_relative_posix_nul_content_nul_v1"
            ),
        }
        or not unresolved_preflight.is_dir()
        or sha256_directory_tree(unresolved_preflight) != evidence["digest"]
    ):
        raise CanonicalScreeningError("4c5 preflight evidence root differs")
    expected_preflight_files = {
        "controller_claim": (
            expected_preflight_path
            + "/preflight_control/controller_claim.json"
        ),
        "controller_terminal": (
            expected_preflight_path
            + "/preflight_control/controller_terminal.json"
        ),
        "controller_summary": (
            expected_preflight_path
            + "/preflight_control/controller_summary.json"
        ),
        "wrapper_claim": (
            expected_preflight_path
            + "/preflight_control/wrapper_claim.json"
        ),
        "wrapper_exit": (
            expected_preflight_path
            + "/preflight_control/wrapper_exit.json"
        ),
        "resource_monitor": (
            expected_preflight_path + "/logs/preflight__monitor.jsonl"
        ),
        "resource_observer": (
            expected_preflight_path + "/logs/preflight__observer.jsonl"
        ),
        "runtime_resource_windows": (
            expected_preflight_path
            + "/preflight_control/runtime_resource_windows.jsonl"
        ),
        "startup_admission": (
            expected_preflight_path
            + "/admissions/"
            "preflight_cpu_startup__fb2850630e589e017d27195c1ca9828b1"
            "b20f273618f7ad296ec633059526464.json"
        ),
        "final_plan": (
            "artifacts/closeout/historical-canonical-512-v1/"
            "checkpoint_plan_final__4c5ecb55501fa6b0.json"
        ),
        "candidate_manifest": (
            "artifacts/closeout/historical-canonical-512-v1/"
            "candidate_manifest__4c5ecb55501fa6b0.json"
        ),
    }
    if any(
        preflight[name].get("path") != expected_preflight_files[name]
        for name in preflight_files
    ):
        raise CanonicalScreeningError(
            "4c5 preflight file identity differs"
        )
    bound = {
        name: _validate_bound_file(
            repo_root, preflight[name], f"4c5 {name}"
        )
        for name in preflight_files
    }
    summary = load_json(Path(bound["controller_summary"]["path"]), "4c5 summary")
    terminal = load_json(
        Path(bound["controller_terminal"]["path"]), "4c5 terminal"
    )
    wrapper = load_json(Path(bound["wrapper_exit"]["path"]), "4c5 wrapper")
    plan = load_json(Path(bound["final_plan"]["path"]), "4c5 final plan")
    manifest = load_json(
        Path(bound["candidate_manifest"]["path"]), "4c5 candidate manifest"
    )
    if (
        summary.get("policy_sha256") != policy_sha
        or summary.get("preflight")
        != {
            "completed": 193,
            "invalid": 0,
            "request_count": 193,
            "reused": 0,
            "valid": 193,
        }
        or terminal.get("status") != "completed"
        or terminal.get("failure") is not None
        or terminal.get("result_count") != 193
        or terminal.get("pending_count") != 0
        or terminal.get("attempt_claim_count") != 193
        or terminal.get("attempt_terminal_count") != 193
        or wrapper.get("exit_code") != 0
        or wrapper.get("signal") is not None
        or wrapper.get("launch_failure") is not None
        or plan.get("checkpoint_plan_sha256")
        != "c7a9329e58b4a31a8e3094e3a60c4e33451dedad36fcc8c9606a70e70439c1ab"
        or manifest.get("candidate_manifest_sha256")
        != "2cd6ba6beeaafe1e6daba605f38668595cb755b826e4a653e474265710751d4e"
        or manifest.get("candidate_count") != 193
    ):
        raise CanonicalScreeningError("4c5 preflight terminal semantics differ")

    failed = _require_mapping(
        supersedes["failed_ram_probe"], "4c5 failed RAM probe"
    )
    expected_failed = {
        "classification": (
            "failed_after_latent_generation_before_arcface_inference"
        ),
        "generation_candidate_count": 1,
        "generation_sample_count": 8,
        "e0_edev_sample_count": 8,
        "arcface_inference_count": 0,
        "quality_evaluation_count": 0,
        "pixel_candidate_count": 0,
        "completed_step_count": 0,
        "worker_result_present": False,
        "ram_budget_sealed": False,
        "resource_guard_clean": True,
        "retry_count": 0,
        "root_cause": (
            "arcface_execution_probe_provenance_fields_truncated"
        ),
        "png_reuse": "forbidden",
        "reuse": "forbidden",
        "in_place_retry": "forbidden",
    }
    _require_exact_keys(
        failed,
        set(expected_failed) | {"root", "files", "png_files"},
        "4c5 failed RAM probe",
    )
    if any(failed[key] != value for key, value in expected_failed.items()):
        raise CanonicalScreeningError("4c5 failed probe classification differs")
    root_binding = _require_mapping(failed["root"], "4c5 failed root")
    expected_failed_path = (
        "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__4c5ecb55501fa6b0"
    )
    failed_root = repo_root.resolve() / expected_failed_path
    _require_tree_without_symlinks(failed_root, "4c5 failed root")
    if (
        root_binding
        != {
            "path": expected_failed_path,
            "digest": (
                "931962816bb7ec702b4985cee40b447517caf8c402bb09d717d674b676bed2b5"
            ),
            "digest_algorithm": (
                "sha256_relative_posix_nul_content_nul_v1"
            ),
        }
        or not failed_root.is_dir()
        or sha256_directory_tree(failed_root) != root_binding["digest"]
    ):
        raise CanonicalScreeningError("4c5 failed root differs")
    expected_files = {
        "controller_claim.json": (
            "1490c328f70620542e408094fdf5b3126fe633b9d7a434efbd39d1e700410409"
        ),
        "input_policy.json": (
            "31adbbe436b4d5546a6a5c7a1f7bb110ded7601445389991915a46f3c8e36963"
        ),
        "admission.json": (
            "d2f1ecf48bccba9c1616511fe7859549571f041b7ea532fe42ec2449f019f524"
        ),
        "probe_spec.json": (
            "8e824f09317baddf27ef2d4c0a2e5d4a9a76cfbcfa408cf8a88dc6aba879e822"
        ),
        "worker.log": (
            "8123b892d91c79276abcb39866fd25bc6fe3bc864377085ef2c00778f88cdb08"
        ),
        "runtime_resource_windows.jsonl": (
            "13ebd01a8c321274ea8178f009e35893a0ca1f2c996a644f50b3a23ed43d7ea0"
        ),
        "probe_result.json": (
            "8d2401461648ab5f5143978e1f527c32524ef85fbd4f413b3dbed03bc3ac1486"
        ),
        "controller_terminal.json": (
            "75bc421ec34ac0463542e4dc0ee13b9d2b2c7b865cb4d4a6d0d0970f2836a082"
        ),
    }
    expected_pngs = {
        "000000.png": (
            "1bc2c6aad34f49afe77d654982bc49cd6105e5726b26a3ba306177df07a1d27b"
        ),
        "000001.png": (
            "b0d5a019429d9c54f2377c7f7bdb48e7fb09fbaeca48b08567d0b2945b996006"
        ),
        "000002.png": (
            "185f7fba318c2828ae3d4895221ee81bf20de0f9fc692606fa0f8245acfb2e8d"
        ),
        "000003.png": (
            "c089a5ea492221896a8e75dd244fb20129786efcd45572398833ecf36697d3a0"
        ),
        "000004.png": (
            "00a790fdc0b23d6637dae29c706c01637b951c08d1e63020c57a33e09dcb4146"
        ),
        "000005.png": (
            "2fa8bc05083d43bea1ddc3a86b1e64da224e21324f1bb28d1492bcbe773f6b68"
        ),
        "000006.png": (
            "e42fa5e6450d5bc9b6662502e3c5d1272046870a7c6edf98a0a952020b9c309e"
        ),
        "000007.png": (
            "a9535e8876f8635ebbbf05bd65c4708c89d993a269008ffd8988452b031cd191"
        ),
    }
    generated = failed_root / "work/latent/generated"
    top_entries = {path.name for path in failed_root.iterdir()}
    work_dirs = {
        str(path.relative_to(failed_root).as_posix())
        for path in failed_root.rglob("*")
        if path.is_dir()
    }
    if (
        failed["files"] != expected_files
        or failed["png_files"] != expected_pngs
        or top_entries != set(expected_files) | {"work"}
        or work_dirs != {"work", "work/latent", "work/latent/generated"}
        or not generated.is_dir()
        or {path.name for path in generated.iterdir()} != set(expected_pngs)
        or any(
            not (failed_root / name).is_file()
            or sha256_file(failed_root / name) != digest
            for name, digest in expected_files.items()
        )
        or any(
            not (generated / name).is_file()
            or sha256_file(generated / name) != digest
            for name, digest in expected_pngs.items()
        )
        or len(list(failed_root.rglob("*"))) != 19
        or (failed_root / "worker_result.json").exists()
        or (failed_root / "work/pixel").exists()
    ):
        raise CanonicalScreeningError("4c5 failed root files differ")

    claim = load_json(failed_root / "controller_claim.json", "4c5 claim")
    admission = load_json(failed_root / "admission.json", "4c5 admission")
    spec = load_json(failed_root / "probe_spec.json", "4c5 spec")
    result = load_json(failed_root / "probe_result.json", "4c5 result")
    failed_terminal = load_json(
        failed_root / "controller_terminal.json", "4c5 failed terminal"
    )
    guard = _require_mapping(
        result.get("runtime_resource_guard"), "4c5 resource guard"
    )
    registry = canonical_gpu_registry(admission["authorized_gpu_registry"])
    if (
        claim.get("controller_claim_sha256")
        != canonical_digest(claim, "controller_claim_sha256")
        or claim.get("controller_claim_sha256")
        != "5f4c1d9875dfaa5a79cbf3ecf2518c3909cb5e8c1d4b4c63c00adddc81accda5"
        or admission.get("admission_sha256")
        != canonical_digest(admission, "admission_sha256")
        or admission.get("admission_sha256")
        != "54f05e25ff9dcf43ddbaf6dd6b9c6317b70974f60fbbce4adf8fe46cab7d08c7"
        or admission.get("admission_evidence_sha256")
        != ram_probe_admission_evidence_digest(admission)
        or admission.get("admission_evidence_sha256")
        != "9b468cb4c0f08cf4c82c67280ed4e89d11c6b60c7d2532fcc63260af408f919e"
        or spec.get("probe_contract_sha256")
        != ram_probe_contract_digest(spec)
        or spec.get("probe_contract_sha256")
        != "1dba8a3d3cafe22d35c02978828e3df3c9fe69e7cd98cf22bd238e6ffd7ffcd4"
        or spec.get("probe_execution_sha256")
        != ram_probe_execution_digest(
            spec["probe_contract_sha256"],
            registry,
            admission["admission_evidence_sha256"],
        )
        or spec.get("probe_execution_sha256")
        != "caaeabaac0877cd3c1ec9e3f0445787b7662250e54a3abde095e6a912a1d260e"
        or admission.get("probe_execution_sha256")
        != spec["probe_execution_sha256"]
        or result.get("probe_result_sha256")
        != canonical_digest(result, "probe_result_sha256")
        or result.get("probe_result_sha256")
        != "924a0c4205b36147818229cea7eab0b1f6189f6aee20f24d739153969a9fa5a8"
        or result.get("status") != "failed"
        or result.get("worker_returncode") != 1
        or result.get("retry_count") != 0
        or result.get("worker_result_sha256") is not None
        or result.get("worker_vmhwm_bytes") is not None
        or result.get("ram_budget_basis_bytes") is not None
        or result.get("ram_slot_budget_bytes") is not None
        or result.get("peak_sampled_process_tree_rss_bytes") != 2384474112
        or guard.get("violated") is not False
        or guard.get("violation_reason") is not None
        or guard.get("thread_failure") is not None
        or guard.get("samples", {}).get("sha256")
        != expected_files["runtime_resource_windows.jsonl"]
        or failed_terminal.get("controller_terminal_sha256")
        != canonical_digest(
            failed_terminal, "controller_terminal_sha256"
        )
        or failed_terminal.get("controller_terminal_sha256")
        != "3ee51f870d5b87bb16079a56f3d41db1d32da2679d03f961d284768cd22cdfcc"
        or failed_terminal.get("controller_claim_sha256")
        != claim["controller_claim_sha256"]
        or failed_terminal.get("status") != "failed"
        or failed_terminal.get("stage") != "worker_execution"
        or failed_terminal.get("worker_started") is not True
        or failed_terminal.get("worker_result_present") is not False
        or failed_terminal.get("probe_result_present") is not True
        or failed_terminal.get("retry_count") != 0
    ):
        raise CanonicalScreeningError("4c5 failed probe evidence chain differs")
    windows = [
        json.loads(line)
        for line in (
            failed_root / "runtime_resource_windows.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    worker_lines = (
        failed_root / "worker.log"
    ).read_text(encoding="utf-8").splitlines()
    binding = json.loads(worker_lines[0])
    uuid = "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6"
    if (
        len(windows) != 5
        or any(
            window.get("resource_window_sha256")
            != canonical_digest(window, "resource_window_sha256")
            or window.get("violated") is not False
            or window.get("swap_consecutive_io") != 0
            for window in windows
        )
        or binding.get("physical_gpu_uuid") != uuid
        or binding.get("runtime_cuda_uuid") != uuid
        or binding.get("cuda_visible_devices") != uuid
        or binding.get("logical_cuda_index") != 0
        or "ArcFace execution probe provenance fields are not canonical"
        not in "\n".join(worker_lines[1:])
        or "_production_face_analyzer_factory" in "\n".join(worker_lines)
    ):
        raise CanonicalScreeningError("4c5 failed runtime evidence differs")
    return dict(supersedes)


def _validate_4d_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "4d supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "preflight",
            "successful_ram_probe",
        },
        "4d supersession evidence",
    )
    policy_sha = (
        "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
    )
    if {
        key: supersedes[key]
        for key in (
            "policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
        )
    } != {
        "policy_sha256": policy_sha,
        "classification": (
            "completed_preflight_and_successful_resource_only_probe_superseded"
        ),
        "supersession_reason": "ram_budget_sealed_from_successful_probe",
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
    }:
        raise CanonicalScreeningError("4d supersession status differs")

    preflight = _require_mapping(supersedes["preflight"], "4d preflight")
    file_names = {
        "controller_claim",
        "controller_terminal",
        "controller_summary",
        "wrapper_claim",
        "wrapper_exit",
        "resource_monitor",
        "resource_observer",
        "runtime_resource_windows",
        "startup_admission",
        "final_plan",
        "candidate_manifest",
    }
    counts = {
        "request_count": 193,
        "result_count": 193,
        "valid_count": 193,
        "invalid_count": 0,
        "reused_count": 0,
        "attempt_claim_count": 193,
        "attempt_terminal_count": 193,
        "run_request_count": 0,
        "generated_png_count": 0,
    }
    _require_exact_keys(
        preflight, set(counts) | file_names | {"evidence_root"}, "4d preflight"
    )
    if any(preflight[key] != value for key, value in counts.items()):
        raise CanonicalScreeningError("4d preflight counts differ")
    relative_root = (
        "artifacts/closeout/historical-canonical-512-v1/by_policy/" + policy_sha
    )
    preflight_root = repo_root.resolve() / relative_root
    evidence = _require_mapping(preflight["evidence_root"], "4d evidence root")
    _require_tree_without_symlinks(preflight_root, "4d preflight evidence root")
    if (
        evidence
        != {
            "path": relative_root,
            "digest": (
                "cc6fcb6b226617c02a1aada2b1f0051d4af15c3ba07215576b3d3d0fd0ae5211"
            ),
            "digest_algorithm": "sha256_relative_posix_nul_content_nul_v1",
        }
        or sha256_directory_tree(preflight_root) != evidence["digest"]
    ):
        raise CanonicalScreeningError("4d preflight evidence root differs")
    expected_paths = {
        "controller_claim": relative_root + "/preflight_control/controller_claim.json",
        "controller_terminal": (
            relative_root + "/preflight_control/controller_terminal.json"
        ),
        "controller_summary": (
            relative_root + "/preflight_control/controller_summary.json"
        ),
        "wrapper_claim": relative_root + "/preflight_control/wrapper_claim.json",
        "wrapper_exit": relative_root + "/preflight_control/wrapper_exit.json",
        "resource_monitor": relative_root + "/logs/preflight__monitor.jsonl",
        "resource_observer": relative_root + "/logs/preflight__observer.jsonl",
        "runtime_resource_windows": (
            relative_root + "/preflight_control/runtime_resource_windows.jsonl"
        ),
        "startup_admission": (
            relative_root
            + "/admissions/preflight_cpu_startup__"
            "01443f03dad7e773895781eea35ecbd514cbc5d88958dd82b5f5768e7331adab.json"
        ),
        "final_plan": (
            "artifacts/closeout/historical-canonical-512-v1/"
            "checkpoint_plan_final__4d0345b6fc29cc8e.json"
        ),
        "candidate_manifest": (
            "artifacts/closeout/historical-canonical-512-v1/"
            "candidate_manifest__4d0345b6fc29cc8e.json"
        ),
    }
    if any(preflight[name].get("path") != path for name, path in expected_paths.items()):
        raise CanonicalScreeningError("4d preflight file identity differs")
    bound = {
        name: _validate_bound_file(repo_root, preflight[name], f"4d {name}")
        for name in file_names
    }
    summary = load_json(Path(bound["controller_summary"]["path"]), "4d summary")
    terminal = load_json(Path(bound["controller_terminal"]["path"]), "4d terminal")
    wrapper = load_json(Path(bound["wrapper_exit"]["path"]), "4d wrapper")
    plan = load_json(Path(bound["final_plan"]["path"]), "4d final plan")
    manifest = load_json(Path(bound["candidate_manifest"]["path"]), "4d manifest")
    if (
        summary.get("policy_sha256") != policy_sha
        or summary.get("preflight")
        != {
            "completed": 193,
            "invalid": 0,
            "request_count": 193,
            "reused": 0,
            "valid": 193,
        }
        or terminal.get("status") != "completed"
        or terminal.get("failure") is not None
        or terminal.get("pending_count") != 0
        or terminal.get("result_count") != 193
        or wrapper.get("exit_code") != 0
        or wrapper.get("signal") is not None
        or wrapper.get("launch_failure") is not None
        or plan.get("checkpoint_plan_sha256")
        != "a1899d29511b08761c5d37c7101aa2a203c0046d244eb0f1ec7b4214ec3ad543"
        or manifest.get("candidate_manifest_sha256")
        != "f2bee5cd0a762e01a275bdd4768658f574384918455cd16a46ff08656b276bb4"
        or manifest.get("candidate_count") != 193
    ):
        raise CanonicalScreeningError("4d preflight terminal semantics differ")

    probe = _require_mapping(
        supersedes["successful_ram_probe"], "4d successful RAM probe"
    )
    _require_exact_keys(
        probe,
        {
            "classification",
            "artifact_seal",
            "probe_result",
            "controller_terminal",
            "resource_measurement_only",
            "scientific_result_reuse",
            "retry_count",
        },
        "4d successful RAM probe",
    )
    seal = _validate_ram_probe_artifact_seal(repo_root, probe["artifact_seal"])
    result_binding = _validate_bound_file(
        repo_root, probe["probe_result"], "4d successful probe result"
    )
    result = load_json(Path(result_binding["path"]), "4d successful probe result")
    if (
        probe["classification"] != "successful_resource_measurement_only"
        or probe["controller_terminal"] != "absent_by_contract"
        or probe["resource_measurement_only"] is not True
        or probe["scientific_result_reuse"] != "forbidden"
        or probe["retry_count"] != 0
        or probe["artifact_seal"] != seal
        or result.get("status") != "succeeded"
        or result.get("failure") is not None
        or result.get("worker_returncode") != 0
        or result.get("termination") is not None
        or result.get("retry_count") != 0
        or result.get("probe_result_sha256")
        != canonical_digest(result, "probe_result_sha256")
    ):
        raise CanonicalScreeningError("4d successful probe semantics differ")
    return {
        **dict(supersedes),
        "successful_ram_probe": {
            **dict(probe),
            "artifact_seal": seal,
            "probe_result": result_binding,
        },
    }


def _validate_5dbb_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "5dbb supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "ram_budget_source_policy_sha256",
            "counts",
            "evidence_root",
            "files",
            "execution_audit",
        },
        "5dbb supersession evidence",
    )
    policy_sha = (
        "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af"
    )
    if {
        key: supersedes[key]
        for key in (
            "policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "ram_budget_source_policy_sha256",
        )
    } != {
        "policy_sha256": policy_sha,
        "classification": (
            "completed_preflight_and_smoke8_scientific_success_"
            "execution_barrier_incomplete_superseded"
        ),
        "supersession_reason": (
            "gpu_ready_barrier_and_phase_local_accounting_upgrade"
        ),
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight_and_smoke8",
        "ram_budget_source_policy_sha256": (
            "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
        ),
    }:
        raise CanonicalScreeningError("5dbb supersession status differs")
    expected_counts = {
        "preflight_request_count": 193,
        "preflight_result_count": 193,
        "preflight_valid_count": 193,
        "run_request_count": 386,
        "run_result_count": 386,
        "run_claim_count": 386,
        "per_sample_count": 386,
        "generated_png_count": 3088,
        "screen512_request_count": 0,
    }
    if supersedes["counts"] != expected_counts:
        raise CanonicalScreeningError("5dbb evidence counts differ")
    relative_root = (
        "artifacts/closeout/historical-canonical-512-v1/by_policy/" + policy_sha
    )
    evidence_root = repo_root.resolve() / relative_root
    evidence = _require_mapping(supersedes["evidence_root"], "5dbb evidence root")
    _require_tree_without_symlinks(evidence_root, "5dbb evidence root")
    if (
        evidence
        != {
            "path": relative_root,
            "digest": (
                "929210949ed4518f6585cd90eb306b9eb856cfae0821cbcdaab86b371ed66978"
            ),
            "digest_algorithm": "sha256_relative_posix_nul_content_nul_v1",
        }
        or sha256_directory_tree(evidence_root) != evidence["digest"]
    ):
        raise CanonicalScreeningError("5dbb evidence root differs")
    expected_paths = {
        "preflight_controller_summary": (
            relative_root + "/preflight_control/controller_summary.json"
        ),
        "preflight_controller_terminal": (
            relative_root + "/preflight_control/controller_terminal.json"
        ),
        "final_plan": (
            "artifacts/closeout/historical-canonical-512-v1/"
            "checkpoint_plan_final__5dbb82fdb1c89d8f.json"
        ),
        "candidate_manifest": (
            "artifacts/closeout/historical-canonical-512-v1/"
            "candidate_manifest__5dbb82fdb1c89d8f.json"
        ),
        "smoke_summary": relative_root + "/summaries/smoke8__completed.json",
        "smoke_admission": (
            relative_root
            + "/admissions/smoke8__"
            "88c500a7256e8f923adf070cc1bdd4271fbedd23d5d71e925e720d35dd4c6870.json"
        ),
        "smoke_monitor": relative_root + "/logs/smoke8__monitor.jsonl",
        "smoke_runtime_guard": (
            relative_root + "/logs/smoke8__runtime_resource_windows.jsonl"
        ),
    }
    files = _require_mapping(supersedes["files"], "5dbb evidence files")
    _require_exact_keys(files, set(expected_paths), "5dbb evidence files")
    if any(files[name].get("path") != path for name, path in expected_paths.items()):
        raise CanonicalScreeningError("5dbb evidence file identity differs")
    bound = {
        name: _validate_bound_file(repo_root, files[name], f"5dbb {name}")
        for name in expected_paths
    }
    preflight_summary = load_json(
        Path(bound["preflight_controller_summary"]["path"]),
        "5dbb preflight summary",
    )
    preflight_terminal = load_json(
        Path(bound["preflight_controller_terminal"]["path"]),
        "5dbb preflight terminal",
    )
    plan = load_json(Path(bound["final_plan"]["path"]), "5dbb final plan")
    manifest = load_json(
        Path(bound["candidate_manifest"]["path"]), "5dbb candidate manifest"
    )
    smoke = load_json(Path(bound["smoke_summary"]["path"]), "5dbb smoke summary")
    if (
        preflight_summary.get("policy_sha256") != policy_sha
        or preflight_summary.get("preflight")
        != {
            "completed": 193,
            "invalid": 0,
            "request_count": 193,
            "reused": 0,
            "valid": 193,
        }
        or preflight_terminal.get("status") != "completed"
        or preflight_terminal.get("failure") is not None
        or preflight_terminal.get("result_count") != 193
        or plan.get("checkpoint_plan_sha256")
        != "ae7de6d768a52915f33ce01b46f60369c22df541e16b166b3df9380c4f9e6f24"
        or manifest.get("candidate_manifest_sha256")
        != "c9ee09a9deced912b491e474fc6eff088dfb696b96ce9cbbd8b719f616fabf3c"
        or manifest.get("candidate_count") != 193
        or smoke.get("phase") != "smoke8"
        or smoke.get("request_count") != 386
        or smoke.get("failures") != []
        or smoke.get("capability_completion")
        != {
            "latent": {"completed_count": 374, "request_count": 374},
            "pixel": {"completed_count": 12, "request_count": 12},
        }
    ):
        raise CanonicalScreeningError("5dbb terminal semantics differ")
    observed_counts = {
        "preflight_request_count": len(
            list((evidence_root / "checkpoint_preflight/requests").glob("*.json"))
        ),
        "preflight_result_count": len(
            list((evidence_root / "checkpoint_preflight/results").glob("*.json"))
        ),
        "preflight_valid_count": preflight_summary["preflight"]["valid"],
        "run_request_count": len(
            list((evidence_root / "run_requests").glob("*/*.json"))
        ),
        "run_result_count": len(list((evidence_root / "runs").glob("*/*/result.json"))),
        "run_claim_count": len(list((evidence_root / "runs").glob("*/*/claim.json"))),
        "per_sample_count": len(
            list((evidence_root / "runs").glob("*/*/per_sample.jsonl"))
        ),
        "generated_png_count": len(
            list((evidence_root / "runs").glob("*/*/generated/*.png"))
        ),
        "screen512_request_count": len(
            list((evidence_root / "run_requests/screen512_primary").glob("*.json"))
        ),
    }
    if observed_counts != expected_counts:
        raise CanonicalScreeningError("5dbb filesystem counts differ")
    audit = supersedes["execution_audit"]
    if audit != {
        "smoke_scientific_contract": "completed_and_deterministic",
        "external_observer_claim": "absent",
        "external_observer_ready": "absent",
        "external_observer_samples": "absent",
        "execution_compliance": "p1_failed",
        "screen512": "never_started",
    }:
        raise CanonicalScreeningError("5dbb execution audit differs")
    for relative in (
        "gpu_control/smoke8/observer_claim.json",
        "gpu_control/smoke8/observer_ready.json",
        "logs/smoke8__observer.jsonl",
    ):
        if (evidence_root / relative).exists():
            raise CanonicalScreeningError(
                "5dbb external observer absence evidence differs"
            )
    return dict(supersedes)


def _validate_9300_zero_result_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(
        raw_supersedes, "9300 zero-result supersession evidence"
    )
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "previous_policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "ram_budget_source_policy_sha256",
            "counts",
            "evidence_root",
            "checkpoint_plan",
            "request_set",
            "absence_evidence",
        },
        "9300 zero-result supersession evidence",
    )
    policy_sha = (
        "9300a01c5f308840918dca8717f06bd6684e3a52967478950b5a9146b8f62508"
    )
    previous_policy_sha = (
        "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af"
    )
    if {
        key: supersedes[key]
        for key in (
            "policy_sha256",
            "previous_policy_sha256",
            "classification",
            "supersession_reason",
            "scientific_result_reuse",
            "successor_execution",
            "ram_budget_source_policy_sha256",
        )
    } != {
        "policy_sha256": policy_sha,
        "previous_policy_sha256": previous_policy_sha,
        "classification": (
            "prepared_execution_barrier_not_crossed_superseded"
        ),
        "supersession_reason": (
            "cpu_preflight_durable_observer_contract_upgrade"
        ),
        "scientific_result_reuse": "forbidden",
        "successor_execution": "fresh_full_193_preflight",
        "ram_budget_source_policy_sha256": (
            "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
        ),
    }:
        raise CanonicalScreeningError(
            "9300 zero-result supersession status differs"
        )
    expected_counts = {
        "checkpoint_plan_count": 1,
        "preflight_request_count": 193,
        "preflight_result_count": 0,
        "attempt_claim_count": 0,
        "attempt_terminal_count": 0,
        "controller_artifact_count": 0,
        "generated_png_count": 0,
    }
    if supersedes["counts"] != expected_counts:
        raise CanonicalScreeningError(
            "9300 zero-result supersession counts differ"
        )
    relative_root = (
        "artifacts/closeout/historical-canonical-512-v1/by_policy/"
        + policy_sha
    )
    evidence_root = repo_root.resolve() / relative_root
    _require_tree_without_symlinks(
        evidence_root, "9300 zero-result evidence root"
    )
    evidence = _require_mapping(
        supersedes["evidence_root"], "9300 zero-result evidence root"
    )
    if (
        evidence
        != {
            "path": relative_root,
            "digest": (
                "3c82e2103c4dc5c0f3c83c4b26d51e9"
                "d9168cb1d8486c7512d89d04cda386ac3"
            ),
            "digest_algorithm": (
                "sha256_relative_posix_nul_content_nul_v1"
            ),
            "file_count": 194,
        }
        or sha256_directory_tree(evidence_root)
        != evidence["digest"]
        or len([path for path in evidence_root.rglob("*") if path.is_file()])
        != evidence["file_count"]
    ):
        raise CanonicalScreeningError(
            "9300 zero-result evidence root differs"
        )
    plan_path = evidence_root / "checkpoint_plan.json"
    plan_binding = _validate_bound_file(
        repo_root,
        supersedes["checkpoint_plan"],
        "9300 zero-result checkpoint plan",
    )
    plan = load_json(plan_path, "9300 zero-result checkpoint plan")
    if (
        Path(plan_binding["path"]) != plan_path.resolve()
        or plan_binding["sha256"]
        != "b2b79a6879e1b771d4e72cd404e332520631407c2c7612f1c32ea7dfad31066b"
        or plan.get("checkpoint_plan_sha256")
        != "f6ddb992eebc528d27d994cc6d536e9c792d6d86429e209ad539fcc555e4410f"
        or plan.get("counts", {}).get("preflight_requests") != 193
        or plan.get("counts", {}).get("pending_preflight") != 193
        or plan.get("counts", {}).get("eligible_candidates") != 0
    ):
        raise CanonicalScreeningError(
            "9300 zero-result checkpoint plan differs"
        )
    request_root = evidence_root / "checkpoint_preflight/requests"
    request_set = _require_mapping(
        supersedes["request_set"], "9300 zero-result request set"
    )
    _require_tree_without_symlinks(
        request_root, "9300 zero-result request set"
    )
    request_paths = sorted(request_root.glob("*.json"))
    if (
        request_set
        != {
            "path": relative_root + "/checkpoint_preflight/requests",
            "digest": (
                "3ca9eab230df733daff633a812d6856e6"
                "8b30cd0a9e81aad4f195632e5ca53e8"
            ),
            "digest_algorithm": (
                "sha256_relative_posix_nul_content_nul_v1"
            ),
            "file_count": 193,
        }
        or sha256_directory_tree(request_root)
        != request_set["digest"]
        or len(request_paths) != request_set["file_count"]
    ):
        raise CanonicalScreeningError(
            "9300 zero-result request set differs"
        )
    request_keys = set()
    for path in request_paths:
        request = load_json(path, "9300 zero-result preflight request")
        if (
            request.get("policy_sha256") != policy_sha
            or request.get("preflight_request_sha256")
            != canonical_digest(
                request, "preflight_request_sha256"
            )
            or request.get("checkpoint_model") not in {"raw", "ema"}
            or path.name
            != (
                f"{request.get('checkpoint_sha256')}__"
                f"{request.get('checkpoint_model')}.json"
            )
        ):
            raise CanonicalScreeningError(
                "9300 zero-result request semantics differ"
            )
        request_keys.add(
            (
                request["checkpoint_sha256"],
                request["checkpoint_model"],
            )
        )
    if len(request_keys) != 193:
        raise CanonicalScreeningError(
            "9300 zero-result request identities differ"
        )
    absence = _require_mapping(
        supersedes["absence_evidence"],
        "9300 zero-result absence evidence",
    )
    expected_absence = {
        "preflight_results": "absent",
        "preflight_control": "absent",
        "preflight_request_manifest": "absent",
        "admissions": "absent",
        "logs": "absent",
        "final_plan": "absent",
        "candidate_manifest": "absent",
    }
    if absence != expected_absence:
        raise CanonicalScreeningError(
            "9300 zero-result absence status differs"
        )
    absent_paths = (
        evidence_root / "checkpoint_preflight/results",
        evidence_root / "preflight_control",
        evidence_root
        / "checkpoint_preflight/preflight_request_manifest.json",
        evidence_root / "admissions",
        evidence_root / "logs",
        repo_root.resolve()
        / (
            "artifacts/closeout/historical-canonical-512-v1/"
            "checkpoint_plan_final__9300a01c5f308840.json"
        ),
        repo_root.resolve()
        / (
            "artifacts/closeout/historical-canonical-512-v1/"
            "candidate_manifest__9300a01c5f308840.json"
        ),
    )
    if any(path.exists() for path in absent_paths):
        raise CanonicalScreeningError(
            "9300 zero-result absence evidence differs"
        )
    return dict(supersedes)


def validate_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "supersession evidence")
    if (
        supersedes.get("policy_sha256")
        == "c83b95e0ca49a0cf5b5b3d67c337000a31b8d2a3299d434fc1051256f18fea50"
    ):
        return _validate_c83_preflight_supersession_evidence(
            repo_root, supersedes
        )
    if (
        supersedes.get("policy_sha256")
        == "310f5b539315d3bc957530856c0f810bf5b32afc97469fdb9467bf3facdc9cda"
    ):
        return _validate_310_preflight_supersession_evidence(
            repo_root, supersedes
        )
    if (
        supersedes.get("policy_sha256")
        == "5d51185345983fbf9bc2924f43d5a4b671674398581824753c0c155c4cdda2db"
    ):
        return _validate_5d_preflight_supersession_evidence(
            repo_root, supersedes
        )
    if (
        supersedes.get("policy_sha256")
        == "6b088236579f731183e60c7fc1d7bece31089284aaaf13697a73f3fb6cd42072"
    ):
        return _validate_6b_supersession_evidence(repo_root, supersedes)
    if (
        supersedes.get("policy_sha256")
        == "fe9b41136f0b9fa31ce210dfaa5500c3f46f071838ed91288878f57073502060"
    ):
        return _validate_fe9_supersession_evidence(repo_root, supersedes)
    if (
        supersedes.get("policy_sha256")
        == "4c5ecb55501fa6b09b63377e892f1cee3e0140abd2a02859d33b9b33375a1576"
    ):
        return _validate_4c5_supersession_evidence(repo_root, supersedes)
    if (
        supersedes.get("policy_sha256")
        == "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
    ):
        return _validate_4d_supersession_evidence(repo_root, supersedes)
    if (
        supersedes.get("policy_sha256")
        == "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af"
    ):
        return _validate_5dbb_supersession_evidence(repo_root, supersedes)
    if (
        supersedes.get("policy_sha256")
        == "9300a01c5f308840918dca8717f06bd6684e3a52967478950b5a9146b8f62508"
    ):
        return _validate_9300_zero_result_supersession_evidence(
            repo_root, supersedes
        )
    if (
        supersedes.get("policy_sha256")
        == "ea7ae71fd662526b9a45bf3cc6d283884aefc380b292c8f273169a35f42ffc28"
    ):
        return _validate_ea7_smoke_supersession_evidence(repo_root, supersedes)
    raise CanonicalScreeningError("unknown supersession policy SHA")


def _validate_output_decoder_registry(
    repo_root: Path,
    raw_registry: Mapping[str, Any],
    *,
    verify_historical_evidence: bool,
    asset_verification_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        registry = validate_decoder_registry(raw_registry)
    except GeneratorOutputContractError as exc:
        raise CanonicalScreeningError(str(exc)) from exc

    def known_binding(
        raw_binding: Mapping[str, Any],
        relative_path: str,
        expected_sha256: str,
        label: str,
        *,
        verify_content: bool,
    ) -> dict[str, Any]:
        binding = _require_mapping(raw_binding, label)
        path = (
            repo_root / str(binding["path"])
            if not Path(str(binding["path"])).is_absolute()
            else Path(str(binding["path"]))
        ).resolve()
        expected_path = (repo_root / relative_path).resolve()
        if (
            path != expected_path
            or binding.get("sha256") != expected_sha256
            or (
                verify_content
                and (
                    not path.is_file()
                    or sha256_file(path) != expected_sha256
                )
            )
        ):
            raise CanonicalScreeningError(f"{label} frozen binding differs")
        return {"path": str(path), "sha256": expected_sha256}

    pixel = _require_mapping(registry["pixel"], "pixel output registry")
    if (
        pixel["decoder_type"] != PIXEL_DECODER_TYPE
        or pixel["channels"] != 3
        or pixel["height"] != 224
        or pixel["width"] != 224
        or pixel["output_range"] != [0.0, 1.0]
        or pixel["model_type"] != "conditional_flow_matching"
        or pixel["sampler"] != "heun"
        or pixel["sample_steps"] != 32
        or pixel["model_space"] != "rgb_neg1_pos1"
        or pixel["sample_api"] != "clamp_output=true"
        or pixel["clamp_output"] is not True
        or pixel["postprocess"]
        != "in_generator_clamp_minus1_1_then_affine_then_clamp_unit_interval"
        or pixel["decoder_forbidden"] is not True
    ):
        raise CanonicalScreeningError("pixel output registry semantics differ")
    bound_pixel = {
        **dict(pixel),
        "sampling_implementation": known_binding(
            pixel["sampling_implementation"],
            "src/safa/models/generator.py",
            "b3d8a699aa4a236c3998681510f7bc1e01ada5415a04342182c91f2f7ae74219",
            "pixel sampling implementation",
            verify_content=True,
        ),
    }
    latent = _require_mapping(registry["latent"], "latent decoder registry")
    if (
        latent["decoder_type"] != LATENT_DECODER_TYPE
        or latent["vae_source_path"]
        != "artifacts/checkpoints/external/sd-vae-ft-ema"
        or latent["scaling_factor"] != 0.18215
        or latent["directory_digest_algorithm"]
        != "sha256_relative_posix_nul_content_nul_v1"
        or latent["latent_shape"] != ["B", 4, 32, 32]
        or latent["decoded_rgb_shape"] != ["B", 3, 256, 256]
        or latent["output_range"] != [0.0, 1.0]
    ):
        raise CanonicalScreeningError("latent decoder registry semantics differ")
    directory = _require_mapping(latent["directory"], "latent decoder directory")
    _require_exact_keys(directory, {"path", "digest"}, "latent decoder directory")
    directory_path = (
        repo_root / str(directory["path"])
        if not Path(str(directory["path"])).is_absolute()
        else Path(str(directory["path"]))
    ).resolve()
    expected_directory = (
        repo_root / "artifacts/checkpoints/external/sd-vae-ft-ema"
    ).resolve()
    if (
        directory_path != expected_directory
        or directory["digest"]
        != "ac188e7f6ff31ff1a3bbde37fea3c345ec72f9e10589cf8aa8a3ec7e86afb188"
    ):
        raise CanonicalScreeningError("latent decoder directory binding differs")
    known_files = {
        "config": (
            "artifacts/checkpoints/external/sd-vae-ft-ema/config.json",
            "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e",
        ),
        "weights": (
            "artifacts/checkpoints/external/sd-vae-ft-ema/"
            "diffusion_pytorch_model.safetensors",
            "32db726da04f06c1b6b14c0043ce115cc87a501482945c5add89a40d838fcb46",
        ),
        "implementation": (
            "src/safa/training/latent_codec.py",
            "b57aded8d27da9ec209c1ec0f04e5d08f9ac53fd92a67f069f32d2259b6c39cb",
        ),
        "trusted_runtime_config": (
            "configs/medium_v2/experiments/r9_meanflow_semigroup_preflight.yaml",
            "0a9536583af4d3b6234e425bca8ad01d6e52110b81827ad97b71dfa08a40b24d",
        ),
        "trusted_runner": (
            "src/safa/evaluation/meanflow_guidance_runner.py",
            "c5c619af8df3edb8c92491350e078064821a8a7e12b9c317aa7e91bca12645ce",
        ),
        "trusted_reference_checkpoint": (
            "artifacts/checkpoints/"
            "e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last_nopretrained.pt",
            "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d",
        ),
        "trusted_resolved_config": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v8/runtime_configs/confirm512/"
            "paper_eta_0p125.yaml",
            "5adb05787baa1130ead3f20c8483603d3696fd65ac04d1c75623f42678fa7819",
        ),
        "trusted_generation_result": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v8/confirm512/paper_eta_0p125/"
            "shards/shard_0/generation_result.json",
            "f96e3afcf618abd93ae1e4fc2196f19ff55cbbf63f8ea6119b939e68d995f149",
        ),
        "asset_digest_cache_algorithm": (
            "src/safa/evaluation/meanflow_guidance_runner.py",
            "c5c619af8df3edb8c92491350e078064821a8a7e12b9c317aa7e91bca12645ce",
        ),
    }
    bound_latent = {
        **dict(latent),
        "directory": {
            "path": str(directory_path),
            "digest": directory["digest"],
        },
        **{
            name: known_binding(
                latent[name],
                relative_path,
                expected_sha256,
                f"latent decoder {name}",
                verify_content=(
                    name == "implementation"
                    or verify_historical_evidence
                ),
            )
            for name, (relative_path, expected_sha256) in known_files.items()
        },
    }
    cache = _require_mapping(
        latent["asset_digest_cache"], "latent asset digest cache"
    )
    cache_path = Path(str(cache["path"]))
    if not cache_path.is_absolute():
        cache_path = repo_root / cache_path
    cache_path = cache_path.resolve()
    expected_cache_path = (
        repo_root
        / "artifacts/closeout/historical-canonical-512-v1/shared/"
        "vae_asset_digests_v1.json"
    ).resolve()
    if cache_path != expected_cache_path:
        raise CanonicalScreeningError("latent asset digest cache binding differs")
    if verify_historical_evidence or asset_verification_audit is not None:
        content_verification = hash_asset_directory_content(
            directory_path, directory["digest"]
        )
        if asset_verification_audit is not None:
            asset_verification_audit.append(content_verification)
    bound_latent["asset_digest_cache"] = {"path": str(cache_path)}
    environment = _require_mapping(latent["environment"], "decoder environment")
    _require_exact_keys(
        environment,
        {
            "provenance_snapshot",
            "packages_sha256",
            "python_version",
            "torch_version",
            "diffusers_version",
        },
        "decoder environment",
    )
    if (
        environment["packages_sha256"]
        != "35196c0c7f5a8a2db3dcb31a67c0102fbd713db6d67af72eacfffe8f8b82be7b"
        or environment["python_version"] != "3.12.13"
        or environment["torch_version"] != "2.11.0+cu128"
        or environment["diffusers_version"] != "0.38.0"
    ):
        raise CanonicalScreeningError("decoder environment binding differs")
    bound_latent["environment"] = {
        **dict(environment),
        "provenance_snapshot": known_binding(
            environment["provenance_snapshot"],
            "artifacts/closeout/"
            "historical-ledger-v1-precommit-5e5ec305-20260726/"
            "provenance_snapshot.json",
            "94507bbd5b2361b29cc3f4a68ade43b9cf3597fecdd20a93835548360c717b68",
            "decoder environment provenance",
            verify_content=verify_historical_evidence,
        ),
    }
    if verify_historical_evidence:
        generation_result = load_json(
            Path(bound_latent["trusted_generation_result"]["path"]),
            "trusted R9 generation result",
        )
        runtime = _require_mapping(
            generation_result.get("config"), "trusted R9 generation config"
        )
        if runtime.get("batch_size") != 2:
            raise CanonicalScreeningError(
                "trusted R9 generation evidence is not batch=2"
            )
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": DECODER_REGISTRY_CONTRACT,
        "pixel": bound_pixel,
        "latent": bound_latent,
        "decoder_registry_sha256": None,
    }
    normalized["decoder_registry_sha256"] = decoder_registry_digest(normalized)
    validate_decoder_registry(normalized)
    return normalized


def _validate_ram_slot_budget_source(
    repo_root: Path,
    raw_source: Mapping[str, Any],
    *,
    declared_budget_bytes: int,
    expected_predecessor_policy_sha256: str,
) -> dict[str, Any]:
    source = _require_mapping(raw_source, "RAM slot budget source")
    source_is_v2 = (
        source.get("contract_type")
        == "safa_canonical_screening_ram_budget_source_v2"
    )
    source_fields = {
        "contract_type",
        "method",
        "measurement_factor_numerator",
        "measurement_factor_denominator",
        "peak_sampled_process_tree_rss_bytes",
        "worker_vmhwm_bytes",
        "ram_budget_basis_bytes",
        "ram_slot_budget_bytes",
        "probe_result",
    }
    if source_is_v2:
        source_fields.add("probe_artifact_seal")
    _require_exact_keys(
        source,
        source_fields,
        "RAM slot budget source",
    )
    probe_binding = _validate_bound_file(
        repo_root, source["probe_result"], "RAM slot budget probe result"
    )
    artifact_seal = (
        _validate_ram_probe_artifact_seal(
            repo_root, source["probe_artifact_seal"]
        )
        if source_is_v2
        else None
    )
    probe_path = Path(probe_binding["path"])
    result = load_json(probe_path, "RAM slot budget probe result")
    is_v2 = (
        result.get("contract_type")
        == "safa_canonical_screening_ram_probe_result_v2"
    )
    probe_digest_fields = (
        {"probe_contract_sha256", "probe_execution_sha256", "worker_device_binding"}
        if is_v2
        else {"probe_sha256"}
    )
    _require_exact_keys(
        result,
        {
            "schema_version",
            "contract_type",
            "status",
            "purpose",
            "admission_sha256",
            "worker_result_sha256",
            "worker_log_sha256",
            "worker_returncode",
            "termination",
            "peak_sampled_process_tree_rss_bytes",
            "worker_vmhwm_bytes",
            "ram_budget_basis_bytes",
            "ram_slot_budget_bytes",
            "budget_method",
            "measurement_factor_numerator",
            "measurement_factor_denominator",
            "runtime_resource_guard",
            "failure",
            "retry_count",
            "completed_at",
            "probe_result_sha256",
        }
        | probe_digest_fields,
        "RAM slot budget probe result",
    )
    expected_method = (
        "ceil(max(peak_sampled_process_tree_rss_bytes,"
        "worker_vmhwm_bytes)*11/10);sampled_tree_every_0.1s_"
        "plus_worker_vmhwm_not_a_mathematical_instantaneous_tree_peak"
    )
    peak = result["peak_sampled_process_tree_rss_bytes"]
    worker_vmhwm = result["worker_vmhwm_bytes"]
    budget_basis = result["ram_budget_basis_bytes"]
    budget = result["ram_slot_budget_bytes"]
    numerator = result["measurement_factor_numerator"]
    denominator = result["measurement_factor_denominator"]
    if (
        result["schema_version"] != 1
        or result["contract_type"]
        not in {
            "safa_canonical_screening_ram_probe_result_v1",
            "safa_canonical_screening_ram_probe_result_v2",
        }
        or result["status"] != "succeeded"
        or result["purpose"]
        != "resource_measurement_only_scientific_reuse_forbidden"
        or result["failure"] is not None
        or result["retry_count"] != 0
        or result["worker_returncode"] != 0
        or result["termination"] is not None
        or result["budget_method"] != expected_method
        or numerator != 11
        or denominator != 10
        or type(peak) is not int
        or peak <= 0
        or type(worker_vmhwm) is not int
        or worker_vmhwm <= 0
        or budget_basis != max(peak, worker_vmhwm)
        or type(budget) is not int
        or budget
        != (budget_basis * numerator + denominator - 1) // denominator
        or result["probe_result_sha256"]
        != canonical_digest(result, "probe_result_sha256")
    ):
        raise CanonicalScreeningError("sealed RAM probe result semantics differ")
    for field in (
        *(
            ("probe_contract_sha256", "probe_execution_sha256")
            if is_v2
            else ("probe_sha256",)
        ),
        "admission_sha256",
        "worker_result_sha256",
        "worker_log_sha256",
        "probe_result_sha256",
    ):
        _require_sha256(result[field], f"RAM probe result {field}")
    guard = _require_mapping(
        result["runtime_resource_guard"], "RAM probe runtime resource guard"
    )
    if (
        guard.get("violated") is not False
        or guard.get("violation_reason") is not None
        or guard.get("thread_failure") is not None
    ):
        raise CanonicalScreeningError(
            "sealed RAM probe runtime resource guard is not clean"
        )

    artifact_root = probe_path.parent
    spec = load_json(artifact_root / "probe_spec.json", "RAM probe spec")
    admission = load_json(
        artifact_root / "admission.json", "RAM probe admission"
    )
    worker = load_json(
        artifact_root / "worker_result.json", "RAM probe worker result"
    )
    legacy_chain_invalid = (
        not is_v2
        and (
            spec.get("contract_type") != "safa_canonical_screening_ram_probe_v1"
            or spec.get("probe_sha256")
            != canonical_digest(spec, "probe_sha256")
            or spec.get("probe_sha256") != result["probe_sha256"]
            or admission.get("contract_type")
            != "safa_canonical_screening_ram_probe_admission_v1"
            or admission.get("probe_sha256") != result["probe_sha256"]
            or worker.get("contract_type")
            != "safa_canonical_screening_ram_probe_worker_result_v1"
            or worker.get("probe_sha256") != result["probe_sha256"]
        )
    )
    v2_chain_invalid = (
        is_v2
        and (
            spec.get("contract_type") != "safa_canonical_screening_ram_probe_v2"
            or spec.get("probe_contract_sha256")
            != ram_probe_contract_digest(spec)
            or admission.get("contract_type")
            != "safa_canonical_screening_ram_probe_admission_v2"
            or admission.get("admission_evidence_sha256")
            != ram_probe_admission_evidence_digest(admission)
            or canonical_gpu_registry(
                admission.get("authorized_gpu_registry", [])
            )
            != admission.get("authorized_gpu_registry")
            or admission.get("authorized_gpu_registry")
            != spec.get("authorized_gpu_registry")
            or not isinstance(spec.get("admission"), Mapping)
            or Path(str(spec["admission"].get("path", ""))).resolve()
            != (artifact_root / "admission.json").resolve()
            or spec["admission"].get("sha256")
            != sha256_file(artifact_root / "admission.json")
            or spec["admission"].get("canonical_sha256")
            != admission.get("admission_sha256")
            or admission.get("probe_contract_sha256")
            != spec.get("probe_contract_sha256")
            or admission.get("probe_execution_sha256")
            != ram_probe_execution_digest(
                spec["probe_contract_sha256"],
                spec["authorized_gpu_registry"],
                admission["admission_evidence_sha256"],
            )
            or spec.get("probe_execution_sha256")
            != admission.get("probe_execution_sha256")
            or result.get("probe_contract_sha256")
            != spec.get("probe_contract_sha256")
            or result.get("probe_execution_sha256")
            != spec.get("probe_execution_sha256")
            or worker.get("contract_type")
            != "safa_canonical_screening_ram_probe_worker_result_v2"
            or worker.get("status") != "succeeded"
            or worker.get("failure") is not None
            or worker.get("probe_contract_sha256")
            != spec.get("probe_contract_sha256")
            or worker.get("probe_execution_sha256")
            != spec.get("probe_execution_sha256")
            or result.get("worker_device_binding")
            != worker.get("device_binding")
        )
    )
    if (
        legacy_chain_invalid
        or v2_chain_invalid
        or spec.get("purpose") != result["purpose"]
        or admission.get("admission_sha256")
        != canonical_digest(admission, "admission_sha256")
        or admission.get("admission_sha256") != result["admission_sha256"]
        or worker.get("worker_result_sha256")
        != canonical_digest(worker, "worker_result_sha256")
        or worker.get("worker_result_sha256")
        != result["worker_result_sha256"]
        or worker.get("purpose") != result["purpose"]
        or type(worker.get("worker_vmhwm_bytes")) is not int
        or worker.get("worker_vmhwm_bytes") <= 0
        or worker.get("worker_vmhwm_bytes") != worker_vmhwm
    ):
        raise CanonicalScreeningError("sealed RAM probe evidence chain differs")
    log_path = artifact_root / "worker.log"
    if not log_path.is_file() or sha256_file(log_path) != result["worker_log_sha256"]:
        raise CanonicalScreeningError("sealed RAM probe worker log binding differs")

    for label in ("candidate_manifest", "sample_manifest"):
        binding = _require_mapping(spec.get(label), f"RAM probe {label}")
        bound_path = Path(str(binding.get("path", ""))).resolve()
        if (
            not bound_path.is_file()
            or sha256_file(bound_path) != binding.get("sha256")
        ):
            raise CanonicalScreeningError(
                f"sealed RAM probe {label} input binding differs"
            )
    policy_binding = _require_mapping(spec.get("policy"), "RAM probe policy")
    snapshot = _require_mapping(
        policy_binding.get("snapshot"), "RAM probe policy snapshot"
    )
    snapshot_path = Path(str(snapshot.get("path", ""))).resolve()
    if (
        not snapshot_path.is_file()
        or sha256_file(snapshot_path) != snapshot.get("sha256")
        or snapshot.get("sha256") != policy_binding.get("sha256")
    ):
        raise CanonicalScreeningError(
            "sealed RAM probe policy snapshot binding differs"
        )
    policy_identity = Path(str(policy_binding.get("path", ""))).resolve()
    expected_policy_identity = (
        repo_root.resolve() / "configs/closeout/canonical_screening_512_v1.json"
    ).resolve()
    if policy_identity != expected_policy_identity:
        raise CanonicalScreeningError(
            "sealed RAM probe policy identity path differs"
        )
    snapshot_raw = load_json(
        snapshot_path, "RAM probe predecessor policy snapshot"
    )
    snapshot_resources = _require_mapping(
        snapshot_raw.get("resources"),
        "RAM probe predecessor policy resources",
    )
    if snapshot_resources.get("ram_budget_status") != "probe_required":
        raise CanonicalScreeningError(
            "sealed RAM probe predecessor must be a probe-required policy"
        )
    implementations = _require_mapping(
        spec.get("implementations"), "RAM probe implementations"
    )
    if source_is_v2:
        if (
            policy_binding.get("sha256") != snapshot.get("sha256")
            or policy_binding.get("canonical_sha256")
            != expected_predecessor_policy_sha256
        ):
            raise CanonicalScreeningError(
                "sealed RAM probe predecessor policy binding differs"
            )
        snapshot_implementations = _require_mapping(
            snapshot_raw.get("implementations"),
            "RAM probe predecessor implementations",
        )
        if set(implementations) != set(snapshot_implementations):
            raise CanonicalScreeningError(
                "sealed RAM probe implementation registry differs"
            )
        for name, binding in implementations.items():
            snapshot_binding = _require_mapping(
                snapshot_implementations[name],
                f"RAM probe predecessor implementation {name}",
            )
            implementation_path = Path(str(binding.get("path", ""))).resolve()
            snapshot_implementation_path = (
                repo_root.resolve() / str(snapshot_binding.get("path", ""))
            ).resolve()
            if (
                implementation_path != snapshot_implementation_path
                or binding.get("sha256") != snapshot_binding.get("sha256")
            ):
                raise CanonicalScreeningError(
                    f"sealed RAM probe implementation binding differs: {name}"
                )
    else:
        probe_policy = validate_policy(
            repo_root,
            snapshot_path,
            verify_historical_output_evidence=False,
            policy_identity_path=policy_identity,
        )
        if (
            policy_binding.get("sha256") != snapshot.get("sha256")
            or policy_binding.get("canonical_sha256")
            != probe_policy["policy_sha256"]
            or expected_predecessor_policy_sha256
            != probe_policy["policy_sha256"]
        ):
            raise CanonicalScreeningError(
                "sealed RAM probe predecessor policy binding differs"
            )
        for name, binding in implementations.items():
            implementation_path = Path(str(binding.get("path", ""))).resolve()
            if (
                not implementation_path.is_file()
                or sha256_file(implementation_path) != binding.get("sha256")
            ):
                raise CanonicalScreeningError(
                    f"sealed RAM probe implementation binding differs: {name}"
                )

    registry = admission.get("authorized_gpu_registry")
    expected_indices = [0, 1, 2, 3]
    if (
        not isinstance(registry, list)
        or [row.get("physical_gpu_index") for row in registry]
        != expected_indices
        or len({row.get("physical_gpu_uuid") for row in registry}) != 4
        or spec.get("authorized_gpu_registry") != registry
    ):
        raise CanonicalScreeningError(
            "sealed RAM probe authorized GPU registry differs"
        )
    device = _require_mapping(
        worker.get("device_binding"), "RAM probe worker device"
    )
    if (
        device.get("physical_gpu_index") != 0
        or device.get("physical_gpu_uuid")
        != registry[0]["physical_gpu_uuid"]
        or device.get("logical_cuda_index") != 0
        or device.get("runtime_cuda_uuid")
        != registry[0]["physical_gpu_uuid"]
        or device.get("cuda_visible_devices")
        != registry[0]["physical_gpu_uuid"]
    ):
        raise CanonicalScreeningError(
            "sealed RAM probe worker device binding differs"
        )
    selected = spec.get("selected_candidates")
    steps = worker.get("steps")
    if (
        not isinstance(selected, list)
        or not isinstance(steps, list)
        or [row.get("output_space") for row in selected]
        != ["latent", "pixel"]
        or len(steps) != len(selected)
    ):
        raise CanonicalScreeningError(
            "sealed RAM probe candidate coverage differs"
        )
    for descriptor, step in zip(selected, steps, strict=True):
        if (
            any(step.get(key) != value for key, value in descriptor.items())
            or step.get("sample_count") != 8
        ):
            raise CanonicalScreeningError(
                "sealed RAM probe worker step binding differs"
            )

    if (
        source["contract_type"]
        not in {
            "safa_canonical_screening_ram_budget_source_v1",
            "safa_canonical_screening_ram_budget_source_v2",
        }
        or source["method"] != expected_method
        or source["measurement_factor_numerator"] != numerator
        or source["measurement_factor_denominator"] != denominator
        or source["peak_sampled_process_tree_rss_bytes"] != peak
        or source["worker_vmhwm_bytes"] != worker_vmhwm
        or source["ram_budget_basis_bytes"] != budget_basis
        or source["ram_slot_budget_bytes"] != budget
        or declared_budget_bytes != budget
    ):
        raise CanonicalScreeningError("sealed RAM slot budget differs")
    normalized_source = {
        **dict(source),
        "probe_result": probe_binding,
    }
    if source_is_v2:
        normalized_source["probe_artifact_seal"] = artifact_seal
    return normalized_source


def validate_policy(
    repo_root: Path,
    policy_path: Path,
    *,
    verify_historical_output_evidence: bool = True,
    policy_identity_path: Path | None = None,
    asset_verification_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    identity_path = (
        policy_path.resolve()
        if policy_identity_path is None
        else policy_identity_path.resolve()
    )
    raw = load_json(policy_path, "canonical screening policy")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "contract_type",
            "campaign_id",
            "supersedes",
            "python",
            "source",
            "protocol",
            "resources",
            "arcface",
            "output_decoder_registry",
            "implementations",
        },
        "canonical screening policy",
    )
    if raw["schema_version"] != SCHEMA_VERSION or raw["contract_type"] != POLICY_CONTRACT:
        raise CanonicalScreeningError("canonical screening policy type/version mismatch")
    if raw["campaign_id"] != "historical-canonical-512-v1":
        raise CanonicalScreeningError("canonical screening campaign_id is not frozen")
    bound_supersedes = validate_supersession_evidence(root, raw["supersedes"])
    if raw["python"] != "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python":
        raise CanonicalScreeningError("canonical screening interpreter is not frozen")

    source = _require_mapping(raw["source"], "policy source")
    _require_exact_keys(
        source,
        {"ledger", "protocol_registry", "artifact_manifest"},
        "policy source",
    )
    source = {
        name: _validate_bound_file(root, source[name], f"source {name}")
        for name in ("ledger", "protocol_registry", "artifact_manifest")
    }

    protocol = _require_mapping(raw["protocol"], "screening protocol")
    _require_exact_keys(
        protocol,
        {
            "seed",
            "batch_size",
            "manifests",
            "source_index",
            "features",
            "e0",
            "edev",
            "quality_script",
            "pixel_image_size",
            "pixel_protocol_config",
            "kid_subset_sizes",
            "metrics",
        },
        "screening protocol",
    )
    if protocol["seed"] != 4549 or protocol["batch_size"] != 2:
        raise CanonicalScreeningError("canonical screening requires seed=4549 and batch_size=2")
    if protocol["pixel_image_size"] != 256:
        raise CanonicalScreeningError("canonical Edev pixel_image_size must be 256")
    if protocol["kid_subset_sizes"] != {"smoke8": 8, "screen512": 50}:
        raise CanonicalScreeningError("canonical KID subset sizes differ")
    manifests = _require_mapping(protocol["manifests"], "screening manifests")
    _require_exact_keys(manifests, set(RUN_MODES), "screening manifests")
    bound_manifests = {
        mode: _validate_bound_file(
            root, manifests[mode], f"{mode} manifest", expected_count=count
        )
        for mode, count in RUN_MODES.items()
    }
    manifest_ids = {
        mode: [str(row.get("sample_id")) for row in load_jsonl(
            Path(binding["path"]), f"{mode} manifest"
        )]
        for mode, binding in bound_manifests.items()
    }
    for mode, ids in manifest_ids.items():
        if any(not item or item == "None" for item in ids) or len(ids) != len(set(ids)):
            raise CanonicalScreeningError(f"{mode} manifest sample IDs are invalid or repeated")
    if manifest_ids["smoke8"] != manifest_ids["screen512"][:8]:
        raise CanonicalScreeningError("smoke8 must be the first eight locked screen512 IDs")

    features = _require_mapping(protocol["features"], "feature cache")
    _require_exact_keys(features, {"directory", "manifest", "shard"}, "feature cache")
    directory = _repo_path(root, f"{features['directory']}/manifest.json", "feature cache")
    if directory.parent != (root / str(features["directory"])).resolve():
        raise CanonicalScreeningError("feature cache directory binding is inconsistent")
    bound_features = {
        "directory": str(directory.parent),
        "manifest": _validate_bound_file(root, features["manifest"], "feature manifest"),
        "shard": _validate_bound_file(root, features["shard"], "feature shard"),
    }
    bound_protocol = {
        "seed": 4549,
        "batch_size": 2,
        "manifests": bound_manifests,
        "source_index": _validate_bound_file(
            root, protocol["source_index"], "source index"
        ),
        "features": bound_features,
        "e0": _validate_bound_file(root, protocol["e0"], "E0 checkpoint"),
        "edev": _validate_bound_file(root, protocol["edev"], "Edev checkpoint"),
        "quality_script": _validate_bound_file(
            root, protocol["quality_script"], "quality evaluator"
        ),
        "pixel_image_size": 256,
        "pixel_protocol_config": _validate_bound_file(
            root, protocol["pixel_protocol_config"], "pixel protocol config"
        ),
        "kid_subset_sizes": {"smoke8": 8, "screen512": 50},
        "metrics": list(protocol["metrics"]),
    }
    if bound_protocol["metrics"] != [
        "e0_cosine",
        "edev_cosine",
        "arcface_source_candidate_cosine",
        "fid",
        "kid",
        "niqe",
        "sharpness",
    ]:
        raise CanonicalScreeningError("canonical metric registry differs")

    resources = _require_mapping(raw["resources"], "screening resources")
    if "ram_budget_status" not in resources:
        raise CanonicalScreeningError(
            "screening resources omit RAM budget status"
        )
    if (
        resources["physical_gpus"] != [0, 1, 2, 3]
        or resources["workers_per_gpu"] != 2
        or resources["ram_budget_status"] not in {"probe_required", "sealed"}
        or resources["retry_count"] != 0
        or resources["require_tmux"] is not True
        or resources["cpu_admission_percent"] != 90
        or resources["cpu_hard_limit_percent"] != 90
        or resources["cpu_window_seconds"] != 60
        or resources["cpu_consecutive_hard_windows"] != 2
        or resources["resource_poll_seconds"] != 10
        or resources["swap_consecutive_hard_intervals"] != 3
        or resources["ram_admission_percent"] != 85
        or resources["ram_hard_limit_percent"] != 90
        or resources["disk_admission_percent"] != 85
        or resources["disk_hard_limit_percent"] != 90
        or resources["gpu_headroom_bytes"] != 2 * 1024**3
    ):
        raise CanonicalScreeningError("canonical resource policy differs")
    if resources["ram_budget_status"] == "probe_required":
        if set(resources) != {
            "physical_gpus",
            "workers_per_gpu",
            "ram_budget_status",
            "gpu_headroom_bytes",
            "cpu_admission_percent",
            "cpu_hard_limit_percent",
            "cpu_window_seconds",
            "cpu_consecutive_hard_windows",
            "resource_poll_seconds",
            "swap_consecutive_hard_intervals",
            "ram_admission_percent",
            "ram_hard_limit_percent",
            "disk_admission_percent",
            "disk_hard_limit_percent",
            "retry_count",
            "require_tmux",
            "global_lock_root",
        }:
            raise CanonicalScreeningError(
                "probe-required resource policy must not preregister a RAM budget"
            )
    else:
        ram_budget_predecessor = bound_supersedes.get(
            "ram_budget_source_policy_sha256",
            bound_supersedes["policy_sha256"],
        )
        expected_keys = {
            "physical_gpus",
            "workers_per_gpu",
            "ram_budget_status",
            "ram_slot_budget_bytes",
            "ram_slot_budget_source",
            "gpu_headroom_bytes",
            "cpu_admission_percent",
            "cpu_hard_limit_percent",
            "cpu_window_seconds",
            "cpu_consecutive_hard_windows",
            "resource_poll_seconds",
            "swap_consecutive_hard_intervals",
            "ram_admission_percent",
            "ram_hard_limit_percent",
            "disk_admission_percent",
            "disk_hard_limit_percent",
            "retry_count",
            "require_tmux",
            "global_lock_root",
        }
        if set(resources) != expected_keys:
            raise CanonicalScreeningError(
                "sealed resource policy omits its RAM budget contract"
            )
        resources = {
            **dict(resources),
            "ram_slot_budget_source": _validate_ram_slot_budget_source(
                root,
                resources["ram_slot_budget_source"],
                declared_budget_bytes=resources["ram_slot_budget_bytes"],
                expected_predecessor_policy_sha256=ram_budget_predecessor,
            ),
        }

    arcface = _require_mapping(raw["arcface"], "ArcFace binding")
    required_arcface = {
        "model_name",
        "model_root",
        "det_size",
        "provider",
        "insightface_version",
        "onnxruntime_version",
        "assets",
        "execution_probe",
    }
    _require_exact_keys(arcface, required_arcface, "ArcFace binding")
    if (
        arcface["model_name"] != "buffalo_l"
        or arcface["det_size"] != [224, 224]
        or arcface["provider"] != "CUDAExecutionProvider"
        or arcface["insightface_version"] != "0.7.3"
        or arcface["onnxruntime_version"] != "1.26.0"
    ):
        raise CanonicalScreeningError("ArcFace runtime binding differs")
    assets = _require_mapping(arcface["assets"], "ArcFace assets")
    if set(assets) != {
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    }:
        raise CanonicalScreeningError("ArcFace asset registry differs")
    for filename, digest in assets.items():
        expected = _require_sha256(digest, f"ArcFace {filename} SHA256")
        path = Path(str(arcface["model_root"])) / "models" / str(arcface["model_name"]) / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise CanonicalScreeningError(f"ArcFace asset mismatch: {path}")
    arcface["execution_probe"] = validate_arcface_execution_probe_binding(
        root,
        arcface["execution_probe"],
        arcface_contract=arcface,
    )
    output_decoder_registry = _validate_output_decoder_registry(
        root,
        raw["output_decoder_registry"],
        verify_historical_evidence=verify_historical_output_evidence,
        asset_verification_audit=asset_verification_audit,
    )

    implementations = _require_mapping(raw["implementations"], "implementations")
    implementation_keys = {
        "checkpoint_preflight",
        "arcface_evaluator",
        "e0_loader",
        "canonical_quality",
        "screening_contracts",
        "preflight_verified_loader",
        "preflight_launch_contract",
        "screening_worker",
        "controller",
        "ram_probe_launcher",
        "preflight_launcher",
        "preflight_wrapper",
        "generator_sampling",
        "meanflow_sampling",
        "latent_codec",
        "output_contract",
    }
    if set(implementations) not in (
        implementation_keys,
        implementation_keys | {"gpu_wrapper"},
    ):
        raise CanonicalScreeningError("implementations keys differ")
    if (
        raw["supersedes"].get("policy_sha256")
        == "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af"
        and "gpu_wrapper" not in implementations
    ):
        raise CanonicalScreeningError("successor policy omits GPU wrapper binding")
    implementations = {
        name: _validate_bound_file(root, value, f"{name} implementation")
        for name, value in implementations.items()
    }

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": POLICY_CONTRACT,
        "campaign_id": raw["campaign_id"],
        "supersedes": bound_supersedes,
        "supersedes_policy_sha256": bound_supersedes["policy_sha256"],
        "python": raw["python"],
        "policy_file": {
            "path": str(identity_path),
            "sha256": sha256_file(policy_path),
        },
        "source": source,
        "protocol": bound_protocol,
        "resources": resources,
        "arcface": arcface,
        "output_decoder_registry": output_decoder_registry,
        "implementations": implementations,
    }
    normalized["policy_sha256"] = canonical_digest(normalized, "policy_sha256")
    return normalized


def _checkpoint_groups(ledger_rows: Sequence[Mapping[str, Any]]) -> tuple[
    dict[str, list[dict[str, Any]]], list[dict[str, Any]]
]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    for row in ledger_rows:
        run_id = str(row.get("run_id", ""))
        status = row.get("status")
        checkpoint = _require_mapping(row.get("checkpoint"), f"ledger checkpoint {run_id}")
        files = checkpoint.get("files")
        if not isinstance(files, list):
            raise CanonicalScreeningError(f"ledger checkpoint files must be a list: {run_id}")
        if not files:
            exclusions.append(
                {
                    "run_id": run_id,
                    "classification": (
                        "config_only_never_started"
                        if status == "config_only_never_started"
                        else "checkpoint_missing_from_ledger"
                    ),
                    "ledger_status": status,
                }
            )
            continue
        selector = checkpoint.get("selector")
        if selector not in CHECKPOINT_SELECTORS:
            for item in files:
                exclusions.append(
                    {
                        "run_id": run_id,
                        "classification": "invalid_checkpoint_selector",
                        "ledger_status": status,
                        "checkpoint_path": item.get("path"),
                        "checkpoint_sha256": item.get("sha256"),
                    }
                )
            continue
        for item in files:
            binding = _require_mapping(item, f"ledger checkpoint file {run_id}")
            sha256 = _require_sha256(
                binding.get("sha256"), f"ledger checkpoint {run_id} SHA256"
            )
            groups[sha256].append(
                {
                    "run_id": run_id,
                    "logical_experiment_id": row.get("logical_experiment_id"),
                    "protocol_family": row.get("protocol_family"),
                    "comparability_group": row.get("comparability_group"),
                    "ledger_status": status,
                    "evidence_level": row.get("evidence_level"),
                    "selector": selector,
                    "path": binding.get("path"),
                    "size_bytes": binding.get("size_bytes"),
                }
            )
    return groups, exclusions


def build_preflight_request(
    *,
    policy: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_model: str,
    checkpoint_path: str,
    path_aliases: Sequence[str],
    source_run_ids: Sequence[str],
) -> dict[str, Any]:
    request = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": PREFLIGHT_REQUEST_CONTRACT,
        "checkpoint_sha256": _require_sha256(
            checkpoint_sha256, "preflight checkpoint SHA256"
        ),
        "checkpoint_model": checkpoint_model,
        "checkpoint_path": checkpoint_path,
        "path_aliases": list(path_aliases),
        "source_run_ids": list(source_run_ids),
        "policy_sha256": policy["policy_sha256"],
        "ledger_sha256": policy["source"]["ledger"]["sha256"],
        "preflight_implementation": dict(
            policy["implementations"]["checkpoint_preflight"]
        ),
        "output_decoder_registry": dict(policy["output_decoder_registry"]),
    }
    request["preflight_request_sha256"] = canonical_digest(
        request, "preflight_request_sha256"
    )
    return validate_preflight_request(request, policy)


def validate_preflight_request(
    request: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(request)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_type",
            "checkpoint_sha256",
            "checkpoint_model",
            "checkpoint_path",
            "path_aliases",
            "source_run_ids",
            "policy_sha256",
            "ledger_sha256",
            "preflight_implementation",
            "output_decoder_registry",
            "preflight_request_sha256",
        },
        "checkpoint preflight request",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != PREFLIGHT_REQUEST_CONTRACT
        or value["checkpoint_model"] not in CHECKPOINT_SELECTORS
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["ledger_sha256"] != policy["source"]["ledger"]["sha256"]
        or value["preflight_implementation"]
        != policy["implementations"]["checkpoint_preflight"]
        or value["output_decoder_registry"] != policy["output_decoder_registry"]
    ):
        raise CanonicalScreeningError("checkpoint preflight request binding mismatch")
    _require_sha256(value["checkpoint_sha256"], "preflight checkpoint SHA256")
    _require_sha256(value["policy_sha256"], "preflight policy SHA256")
    _require_sha256(value["ledger_sha256"], "preflight ledger SHA256")
    _require_sha256(
        value["preflight_request_sha256"], "preflight request SHA256"
    )
    aliases = value["path_aliases"]
    run_ids = value["source_run_ids"]
    if (
        not isinstance(aliases, list)
        or not aliases
        or aliases != sorted(set(aliases))
        or value["checkpoint_path"] != aliases[0]
        or not isinstance(run_ids, list)
        or not run_ids
        or run_ids != sorted(set(run_ids))
    ):
        raise CanonicalScreeningError("checkpoint preflight request lists are not canonical")
    if value["preflight_request_sha256"] != canonical_digest(
        value, "preflight_request_sha256"
    ):
        raise CanonicalScreeningError("checkpoint preflight request digest mismatch")
    return value


def build_preflight_result(
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    strict_result: Mapping[str, Any],
) -> dict[str, Any]:
    validated_request = validate_preflight_request(request, policy)
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": PREFLIGHT_RESULT_CONTRACT,
        "preflight_request_sha256": validated_request["preflight_request_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "ledger_sha256": policy["source"]["ledger"]["sha256"],
        "checkpoint_sha256": validated_request["checkpoint_sha256"],
        "checkpoint_model": validated_request["checkpoint_model"],
        "preflight_implementation": dict(
            policy["implementations"]["checkpoint_preflight"]
        ),
        "strict_result": dict(strict_result),
        "strict_result_sha256": hashlib.sha256(
            canonical_json(dict(strict_result))
        ).hexdigest(),
    }
    result["preflight_result_sha256"] = canonical_digest(
        result, "preflight_result_sha256"
    )
    return result


def validate_preflight_result(
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    value = dict(result)
    validated_request = validate_preflight_request(request, policy)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_type",
            "preflight_request_sha256",
            "policy_sha256",
            "ledger_sha256",
            "checkpoint_sha256",
            "checkpoint_model",
            "preflight_implementation",
            "strict_result",
            "strict_result_sha256",
            "preflight_result_sha256",
        },
        "checkpoint preflight result envelope",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != PREFLIGHT_RESULT_CONTRACT
        or value["preflight_request_sha256"]
        != validated_request["preflight_request_sha256"]
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["ledger_sha256"] != policy["source"]["ledger"]["sha256"]
        or value["checkpoint_sha256"] != validated_request["checkpoint_sha256"]
        or value["checkpoint_model"] != validated_request["checkpoint_model"]
        or value["preflight_implementation"]
        != policy["implementations"]["checkpoint_preflight"]
    ):
        raise CanonicalScreeningError("checkpoint preflight result binding mismatch")
    for field in (
        "preflight_request_sha256",
        "policy_sha256",
        "ledger_sha256",
        "checkpoint_sha256",
        "strict_result_sha256",
        "preflight_result_sha256",
    ):
        _require_sha256(value[field], field)
    strict_result = _require_mapping(
        value["strict_result"], "strict checkpoint preflight result"
    )
    if value["strict_result_sha256"] != hashlib.sha256(
        canonical_json(strict_result)
    ).hexdigest():
        raise CanonicalScreeningError("strict preflight result digest mismatch")
    if value["preflight_result_sha256"] != canonical_digest(
        value, "preflight_result_sha256"
    ):
        raise CanonicalScreeningError("preflight result envelope digest mismatch")

    required = {
        "schema_version",
        "contract_type",
        "status",
        "checkpoint_path",
        "checkpoint_sha256",
        "expected_checkpoint_sha256",
        "sha256_binding",
        "checkpoint_model",
        "declared_checkpoint_model",
        "available_state_dict_fields",
        "selector_binding",
        "state_dict_field",
        "tensor_count",
        "finite_tensor_count",
        "nonfinite_keys",
        "missing_keys",
        "unexpected_keys",
        "shape_mismatches",
        "reconstruction_messages",
        "adapter",
        "output_capability",
        "output_contract",
        "smoke",
        "failure_code",
        "failure_message",
    }
    _require_exact_keys(strict_result, required, "strict checkpoint preflight result")
    expected_sha256 = validated_request["checkpoint_sha256"]
    selector = validated_request["checkpoint_model"]
    if (
        strict_result["contract_type"] != "safa_generator_checkpoint_preflight_v1"
        or strict_result["schema_version"] != 1
        or strict_result["expected_checkpoint_sha256"] != expected_sha256
        or strict_result["checkpoint_model"] != selector
    ):
        raise CanonicalScreeningError(
            f"preflight result does not exactly bind {expected_sha256}/{selector}"
        )
    if strict_result["status"] == "valid" and (
        strict_result["checkpoint_sha256"] != expected_sha256
        or strict_result["sha256_binding"] != "expected_exact"
    ):
        raise CanonicalScreeningError(
            f"valid preflight lacks exact SHA binding for {expected_sha256}/{selector}"
        )
    if strict_result["status"] == "valid":
        output_contract = validate_output_contract(
            _require_mapping(
                strict_result["output_contract"],
                "strict checkpoint output contract",
            ),
            policy["output_decoder_registry"],
        )
        capability = _require_mapping(
            strict_result["output_capability"],
            "strict checkpoint output capability",
        )
        if (
            output_contract["capability"] != capability
            or capability["checkpoint_sha256"] != expected_sha256
        ):
            raise CanonicalScreeningError(
                "valid preflight output contract lacks exact checkpoint binding"
            )
    if strict_result["status"] == "invalid" and (
        strict_result["checkpoint_sha256"] not in {None, expected_sha256}
        or strict_result["sha256_binding"] not in {None, "expected_exact"}
    ):
        raise CanonicalScreeningError(
            f"invalid preflight contradicts expected SHA for {expected_sha256}/{selector}"
        )
    valid = (
        strict_result["status"] == "valid"
        and strict_result["failure_code"] is None
        and strict_result["failure_message"] is None
        and strict_result["nonfinite_keys"] == []
        and strict_result["missing_keys"] == []
        and strict_result["unexpected_keys"] == []
        and strict_result["shape_mismatches"] == []
        and strict_result["tensor_count"] == strict_result["finite_tensor_count"]
        and strict_result["tensor_count"] > 0
    )
    if strict_result["status"] == "valid" and not valid:
        raise CanonicalScreeningError(
            f"preflight claims valid with incomplete strict evidence: {expected_sha256}"
        )
    return valid, strict_result


def _valid_preflight(
    path: Path,
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    return validate_preflight_result(
        load_json(path, "checkpoint preflight result envelope"),
        request,
        policy,
    )


def build_checkpoint_plan(
    repo_root: Path,
    policy: Mapping[str, Any],
    preflight_root: Path,
) -> dict[str, Any]:
    ledger_path = Path(str(policy["source"]["ledger"]["path"]))
    rows = load_jsonl(ledger_path, "experiment ledger")
    groups, checkpointless = _checkpoint_groups(rows)
    checkpoint_references = sum(len(refs) for refs in groups.values())
    raw_references = sum(
        1 for refs in groups.values() for ref in refs if ref["selector"] == "raw"
    )
    ema_references = checkpoint_references - raw_references
    pending: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    exclusions = list(checkpointless)
    request_rows: list[dict[str, Any]] = []
    for sha256 in sorted(groups):
        refs = groups[sha256]
        selectors = sorted({str(ref["selector"]) for ref in refs})
        paths = sorted({str(ref["path"]) for ref in refs})
        if len(selectors) != 1:
            exclusions.append(
                {
                    "checkpoint_sha256": sha256,
                    "classification": "conflicting_checkpoint_selectors",
                    "selectors": selectors,
                    "paths": paths,
                    "run_ids": sorted({str(ref["run_id"]) for ref in refs}),
                }
            )
            continue
        selector = selectors[0]
        request = build_preflight_request(
            policy=policy,
            checkpoint_sha256=sha256,
            checkpoint_model=selector,
            checkpoint_path=paths[0],
            path_aliases=paths,
            source_run_ids=sorted({str(ref["run_id"]) for ref in refs}),
        )
        request_rows.append(request)
        result_path = preflight_root / f"{sha256}__{selector}.json"
        common = {
            "checkpoint_sha256": sha256,
            "checkpoint_model": selector,
            "checkpoint_path": paths[0],
            "path_aliases": paths,
            "source_run_ids": request["source_run_ids"],
            "source_logical_experiment_ids": sorted(
                {str(ref["logical_experiment_id"]) for ref in refs}
            ),
            "protocol_families": sorted({str(ref["protocol_family"]) for ref in refs}),
            "comparability_groups": sorted(
                {str(ref["comparability_group"]) for ref in refs}
            ),
            "ledger_statuses": sorted({str(ref["ledger_status"]) for ref in refs}),
            "evidence_levels": sorted({str(ref["evidence_level"]) for ref in refs}),
            "preflight_request_sha256": request["preflight_request_sha256"],
        }
        if not result_path.is_file():
            pending.append({**common, "classification": "pending_checkpoint_preflight"})
            continue
        valid, result = _valid_preflight(result_path, request, policy)
        result_sha256 = sha256_file(result_path)
        if not valid:
            exclusions.append(
                {
                    **common,
                    "classification": "invalid_checkpoint_preflight",
                    "preflight_result_path": str(result_path.resolve()),
                    "preflight_result_sha256": result_sha256,
                    "failure_code": result["failure_code"],
                    "failure_message": result["failure_message"],
                }
            )
            continue
        eligible.append(
            {
                **common,
                "candidate_id": f"g_{sha256[:16]}_{selector}",
                "preflight_result_path": str(result_path.resolve()),
                "preflight_result_sha256": result_sha256,
                "output_contract": dict(result["output_contract"]),
            }
        )

    plan = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": PLAN_CONTRACT,
        "campaign_id": policy["campaign_id"],
        "policy_sha256": policy["policy_sha256"],
        "ledger_sha256": policy["source"]["ledger"]["sha256"],
        "preflight_implementation": dict(
            policy["implementations"]["checkpoint_preflight"]
        ),
        "preflight_result_root": str(preflight_root.resolve()),
        "counts": {
            "ledger_rows": len(rows),
            "checkpoint_bound_rows": sum(
                1 for row in rows if _require_mapping(row["checkpoint"], "checkpoint")["files"]
            ),
            "checkpointless_rows": len(checkpointless),
            "checkpoint_references": checkpoint_references,
            "raw_checkpoint_references": raw_references,
            "ema_checkpoint_references": ema_references,
            "distinct_checkpoint_sha256": len(groups),
            "distinct_raw_checkpoint_sha256": sum(
                1
                for refs in groups.values()
                if {str(ref["selector"]) for ref in refs} == {"raw"}
            ),
            "distinct_ema_checkpoint_sha256": sum(
                1
                for refs in groups.values()
                if {str(ref["selector"]) for ref in refs} == {"ema"}
            ),
            "duplicate_checkpoint_references": checkpoint_references - len(groups),
            "selector_conflicts": sum(
                1
                for refs in groups.values()
                if len({str(ref["selector"]) for ref in refs}) != 1
            ),
            "preflight_requests": len(request_rows),
            "pending_preflight": len(pending),
            "eligible_candidates": len(eligible),
            "excluded_records": len(exclusions),
        },
        "preflight_requests": request_rows,
        "pending": pending,
        "eligible": eligible,
        "exclusions": sorted(
            exclusions,
            key=lambda item: (
                str(item.get("classification")),
                str(item.get("checkpoint_sha256")),
                str(item.get("run_id")),
            ),
        ),
    }
    plan["checkpoint_plan_sha256"] = canonical_digest(plan, "checkpoint_plan_sha256")
    return plan


def validate_checkpoint_plan(
    plan: Mapping[str, Any],
    *,
    repo_root: Path,
    policy: Mapping[str, Any],
    preflight_root: Path,
) -> dict[str, Any]:
    value = dict(plan)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_type",
            "campaign_id",
            "policy_sha256",
            "ledger_sha256",
            "preflight_implementation",
            "preflight_result_root",
            "counts",
            "preflight_requests",
            "pending",
            "eligible",
            "exclusions",
            "checkpoint_plan_sha256",
        },
        "checkpoint plan",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != PLAN_CONTRACT
        or value["campaign_id"] != policy["campaign_id"]
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["ledger_sha256"] != policy["source"]["ledger"]["sha256"]
        or value["preflight_implementation"]
        != policy["implementations"]["checkpoint_preflight"]
        or value["preflight_result_root"] != str(preflight_root.resolve())
    ):
        raise CanonicalScreeningError("checkpoint plan provenance binding mismatch")
    _require_sha256(value["checkpoint_plan_sha256"], "checkpoint plan SHA256")
    if value["checkpoint_plan_sha256"] != canonical_digest(
        value, "checkpoint_plan_sha256"
    ):
        raise CanonicalScreeningError("checkpoint plan digest mismatch")
    expected = build_checkpoint_plan(repo_root, policy, preflight_root)
    if value != expected:
        raise CanonicalScreeningError(
            "checkpoint plan differs from a fresh ledger/result derivation"
        )
    return value


def build_candidate_manifest(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    repo_root: Path,
    preflight_root: Path,
) -> dict[str, Any]:
    validated_plan = validate_checkpoint_plan(
        plan,
        repo_root=repo_root,
        policy=policy,
        preflight_root=preflight_root,
    )
    if not plan_path.is_file() or load_json(plan_path, "checkpoint plan") != validated_plan:
        raise CanonicalScreeningError("checkpoint plan file binding mismatch")
    if validated_plan.get("pending"):
        raise CanonicalScreeningError(
            "candidate manifest is blocked until every distinct checkpoint has a preflight result"
        )
    candidates = list(validated_plan.get("eligible", []))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": CANDIDATE_MANIFEST_CONTRACT,
        "campaign_id": policy["campaign_id"],
        "policy_sha256": policy["policy_sha256"],
        "ledger_sha256": policy["source"]["ledger"]["sha256"],
        "checkpoint_plan": {
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
            "canonical_sha256": validated_plan["checkpoint_plan_sha256"],
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "protocol": policy["protocol"],
        "implementations": policy["implementations"],
        "arcface": policy["arcface"],
        "output_decoder_registry": policy["output_decoder_registry"],
    }
    manifest["candidate_manifest_sha256"] = canonical_digest(
        manifest, "candidate_manifest_sha256"
    )
    return manifest


def validate_candidate_manifest(
    manifest: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    repo_root: Path,
    preflight_root: Path,
) -> dict[str, Any]:
    value = dict(manifest)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_type",
            "campaign_id",
            "policy_sha256",
            "ledger_sha256",
            "checkpoint_plan",
            "candidate_count",
            "candidates",
            "protocol",
            "implementations",
            "arcface",
            "output_decoder_registry",
            "candidate_manifest_sha256",
        },
        "candidate manifest",
    )
    _require_sha256(value["candidate_manifest_sha256"], "candidate manifest SHA256")
    if value["candidate_manifest_sha256"] != canonical_digest(
        value, "candidate_manifest_sha256"
    ):
        raise CanonicalScreeningError("candidate manifest digest mismatch")
    expected = build_candidate_manifest(
        policy,
        plan,
        plan_path=plan_path,
        repo_root=repo_root,
        preflight_root=preflight_root,
    )
    if value != expected:
        raise CanonicalScreeningError(
            "candidate manifest differs from the validated checkpoint plan"
        )
    return value


def build_run_request(
    policy: Mapping[str, Any],
    policy_path: Path,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidate: Mapping[str, Any],
    mode: str,
    replicate: str,
    output_root: Path,
    admission: Mapping[str, Any],
    controller_ready: Mapping[str, Any],
    observer_ready: Mapping[str, Any],
) -> dict[str, Any]:
    if mode not in RUN_MODES:
        raise CanonicalScreeningError(f"unknown screening mode: {mode}")
    if replicate not in {"primary", "repeat"} or (
        mode == "screen512" and replicate != "primary"
    ):
        raise CanonicalScreeningError("run replicate binding is invalid")
    if candidate not in candidate_manifest.get("candidates", []):
        raise CanonicalScreeningError("run candidate is not in the immutable manifest")
    if not candidate_manifest_path.is_file():
        raise CanonicalScreeningError("candidate manifest file does not exist")
    disk_manifest = load_json(candidate_manifest_path, "candidate manifest")
    if disk_manifest != dict(candidate_manifest):
        raise CanonicalScreeningError(
            "candidate manifest file disagrees with the in-memory contract"
        )
    manifest_binding = policy["protocol"]["manifests"][mode]
    admission_path = Path(str(admission["path"])).resolve()
    admission_value = load_json(admission_path, "resource admission")
    admission_snapshot = _require_mapping(
        admission_value.get("snapshot"), "resource admission snapshot"
    )
    output = (
        output_root / f"{mode}_{replicate}" / str(candidate["candidate_id"])
    ).resolve()
    request = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": RUN_REQUEST_CONTRACT,
        "campaign_id": policy["campaign_id"],
        "mode": mode,
        "replicate": replicate,
        "sample_count": RUN_MODES[mode],
        "seed": 4549,
        "batch_size": 2,
        "policy": {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
            "canonical_sha256": policy["policy_sha256"],
        },
        "implementations": dict(policy["implementations"]),
        "admission": dict(admission),
        "controller_ready": dict(controller_ready),
        "observer_ready": dict(observer_ready),
        "authorized_gpu_registry": list(
            admission_snapshot["authorized_gpu_registry"]
        ),
        "ram_reservation": dict(admission_snapshot["ram_reservation"]),
        "candidate_manifest": {
            "path": str(candidate_manifest_path.resolve()),
            "sha256": sha256_file(candidate_manifest_path),
            "canonical_sha256": candidate_manifest["candidate_manifest_sha256"],
        },
        "candidate": dict(candidate),
        "output_decoder_registry": dict(policy["output_decoder_registry"]),
        "output_contract": dict(candidate["output_contract"]),
        "sample_manifest": dict(manifest_binding),
        "source_index": dict(policy["protocol"]["source_index"]),
        "features": dict(policy["protocol"]["features"]),
        "e0": dict(policy["protocol"]["e0"]),
        "edev": dict(policy["protocol"]["edev"]),
        "quality_script": dict(policy["protocol"]["quality_script"]),
        "pixel_image_size": policy["protocol"]["pixel_image_size"],
        "pixel_protocol_config": dict(
            policy["protocol"]["pixel_protocol_config"]
        ),
        "kid_subset_size": policy["protocol"]["kid_subset_sizes"][mode],
        "arcface": dict(policy["arcface"]),
        "screening_worker": dict(policy["implementations"]["screening_worker"]),
        "quality_protocol_family": candidate["output_contract"][
            "quality_protocol_family"
        ],
        "native_rgb_size": [
            candidate["output_contract"]["rgb_contract"]["height"],
            candidate["output_contract"]["rgb_contract"]["width"],
        ],
        "nfe": candidate["output_contract"]["capability"]["nfe"],
        "output_dir": str(output),
        "retry_count": 0,
    }
    request["run_request_sha256"] = canonical_digest(request, "run_request_sha256")
    return request


def validate_run_request(
    request: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(request)
    required = {
        "schema_version",
        "contract_type",
        "campaign_id",
        "mode",
        "replicate",
        "sample_count",
        "seed",
        "batch_size",
        "policy",
        "implementations",
        "admission",
        "controller_ready",
        "observer_ready",
        "authorized_gpu_registry",
        "ram_reservation",
        "candidate_manifest",
        "candidate",
        "output_decoder_registry",
        "output_contract",
        "sample_manifest",
        "source_index",
        "features",
        "e0",
        "edev",
        "quality_script",
        "pixel_image_size",
        "pixel_protocol_config",
        "kid_subset_size",
        "arcface",
        "screening_worker",
        "quality_protocol_family",
        "native_rgb_size",
        "nfe",
        "output_dir",
        "retry_count",
        "run_request_sha256",
    }
    _require_exact_keys(value, required, "run request")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != RUN_REQUEST_CONTRACT
        or value["mode"] not in RUN_MODES
        or value["replicate"] not in {"primary", "repeat"}
        or (value["mode"] == "screen512" and value["replicate"] != "primary")
        or value["sample_count"] != RUN_MODES[value["mode"]]
        or value["seed"] != 4549
        or value["batch_size"] != 2
        or value["retry_count"] != 0
        or value["campaign_id"] != policy["campaign_id"]
        or value["implementations"] != policy["implementations"]
        or value["output_decoder_registry"] != policy["output_decoder_registry"]
        or value["output_contract"] != value["candidate"].get("output_contract")
        or value["pixel_image_size"] != policy["protocol"]["pixel_image_size"]
        or value["pixel_protocol_config"]
        != policy["protocol"]["pixel_protocol_config"]
        or value["kid_subset_size"]
        != policy["protocol"]["kid_subset_sizes"][value["mode"]]
        or value["native_rgb_size"]
        != [
            value["output_contract"]["rgb_contract"]["height"],
            value["output_contract"]["rgb_contract"]["width"],
        ]
        or value["nfe"] != value["output_contract"]["capability"]["nfe"]
        or value["quality_protocol_family"]
        != value["output_contract"]["quality_protocol_family"]
    ):
        raise CanonicalScreeningError("run request frozen fields differ")
    validate_output_contract(
        _require_mapping(value["output_contract"], "run output contract"),
        policy["output_decoder_registry"],
    )
    policy_binding = _require_mapping(value["policy"], "policy binding")
    _require_exact_keys(
        policy_binding, {"path", "sha256", "canonical_sha256"}, "policy binding"
    )
    if (
        policy_binding["canonical_sha256"] != policy["policy_sha256"]
        or sha256_file(Path(str(policy_binding["path"])).resolve())
        != policy_binding["sha256"]
    ):
        raise CanonicalScreeningError("run request policy binding mismatch")
    for field in ("sha256", "canonical_sha256"):
        _require_sha256(policy_binding[field], f"policy {field}")
    admission = _require_mapping(value["admission"], "admission binding")
    _require_exact_keys(
        admission, {"path", "sha256", "canonical_sha256"}, "admission binding"
    )
    for field in ("sha256", "canonical_sha256"):
        _require_sha256(admission[field], f"admission {field}")
    admission_path = Path(str(admission["path"])).resolve()
    if not admission_path.is_file() or sha256_file(admission_path) != admission["sha256"]:
        raise CanonicalScreeningError("run request admission file binding mismatch")
    admission_value = load_json(admission_path, "resource admission")
    if (
        admission_value.get("admission_sha256") != admission["canonical_sha256"]
        or canonical_digest(admission_value, "admission_sha256")
        != admission["canonical_sha256"]
        or admission_value.get("policy_sha256") != policy["policy_sha256"]
    ):
        raise CanonicalScreeningError("run request admission contract mismatch")
    ready_values: dict[str, dict[str, Any]] = {}
    for field, contract_type, digest_field in (
        (
            "controller_ready",
            CONTROLLER_READY_CONTRACT,
            "controller_ready_sha256",
        ),
        (
            "observer_ready",
            OBSERVER_READY_CONTRACT,
            "observer_ready_sha256",
        ),
    ):
        binding = _require_mapping(value[field], f"{field} binding")
        _require_exact_keys(
            binding,
            {"path", "sha256", "canonical_sha256"},
            f"{field} binding",
        )
        for digest_name in ("sha256", "canonical_sha256"):
            _require_sha256(binding[digest_name], f"{field} {digest_name}")
        path = Path(str(binding["path"])).resolve()
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise CanonicalScreeningError(f"run request {field} file binding mismatch")
        ready = load_json(path, field)
        if (
            ready.get("contract_type") != contract_type
            or ready.get("policy_sha256") != policy["policy_sha256"]
            or ready.get("phase") != value["mode"]
            or ready.get("admission_sha256") != admission["canonical_sha256"]
            or ready.get(digest_field) != binding["canonical_sha256"]
            or canonical_digest(ready, digest_field) != binding["canonical_sha256"]
        ):
            raise CanonicalScreeningError(f"run request {field} contract mismatch")
        ready_values[field] = ready
    if (
        ready_values["observer_ready"].get("controller_ready_sha256")
        != value["controller_ready"]["canonical_sha256"]
    ):
        raise CanonicalScreeningError(
            "run request observer/controller ready binding mismatch"
        )
    for ready_name in ("controller_ready", "observer_ready"):
        ready = ready_values[ready_name]
        for field, digest_field, contract_type in (
            (
                "wrapper_claim",
                "wrapper_claim_sha256",
                "safa_canonical_gpu_wrapper_claim_v1",
            ),
            (
                "observer_launch",
                "observer_launch_sha256",
                "safa_canonical_gpu_observer_launch_v2",
            ),
        ):
            nested = _require_mapping(
                ready.get(field), f"{ready_name} {field} binding"
            )
            _require_exact_keys(
                nested,
                {"path", "sha256", "canonical_sha256"},
                f"{ready_name} {field} binding",
            )
            nested_path = Path(str(nested["path"])).resolve()
            if (
                not nested_path.is_file()
                or sha256_file(nested_path) != nested["sha256"]
            ):
                raise CanonicalScreeningError(
                    f"run request {ready_name} {field} file mismatch"
                )
            artifact = load_json(
                nested_path, f"{ready_name} {field}"
            )
            if (
                artifact.get("contract_type") != contract_type
                or artifact.get("policy_sha256") != policy["policy_sha256"]
                or artifact.get("phase") != value["mode"]
                or artifact.get(digest_field)
                != nested["canonical_sha256"]
                or canonical_digest(artifact, digest_field)
                != nested["canonical_sha256"]
                or ready.get(digest_field)
                != nested["canonical_sha256"]
            ):
                raise CanonicalScreeningError(
                    f"run request {ready_name} {field} contract mismatch"
                )
    if (
        ready_values["controller_ready"]["wrapper_claim"]
        != ready_values["observer_ready"]["wrapper_claim"]
        or ready_values["controller_ready"]["observer_launch"]
        != ready_values["observer_ready"]["observer_launch"]
    ):
        raise CanonicalScreeningError(
            "run request ready wrapper provenance differs"
        )
    snapshot = _require_mapping(
        admission_value.get("snapshot"), "resource admission snapshot"
    )
    authorized_registry = value["authorized_gpu_registry"]
    if (
        not isinstance(authorized_registry, list)
        or authorized_registry != snapshot.get("authorized_gpu_registry")
        or [row.get("physical_gpu_index") for row in authorized_registry]
        != policy["resources"]["physical_gpus"]
        or any(
            set(row) != {"physical_gpu_index", "physical_gpu_uuid"}
            or not isinstance(row["physical_gpu_uuid"], str)
            or not row["physical_gpu_uuid"].startswith("GPU-")
            for row in authorized_registry
        )
        or len({row["physical_gpu_uuid"] for row in authorized_registry})
        != len(authorized_registry)
    ):
        raise CanonicalScreeningError(
            "run request authorized GPU UUID registry mismatch"
        )
    reservation = _require_mapping(
        value["ram_reservation"], "run request RAM reservation"
    )
    if reservation != snapshot.get("ram_reservation"):
        raise CanonicalScreeningError(
            "run request RAM reservation differs from admission"
        )
    required_reservation_fields = {
        "slot_count",
        "slot_budget_bytes",
        "reserved_bytes",
        "memory_total_bytes",
        "memory_used_bytes",
        "projected_used_bytes",
        "projected_used_percent",
        "admission_limit_percent",
        "budget_source",
    }
    _require_exact_keys(
        reservation, required_reservation_fields, "run request RAM reservation"
    )
    expected_slot_count = len(policy["resources"]["physical_gpus"]) * int(
        policy["resources"]["workers_per_gpu"]
    )
    if (
        policy["resources"]["ram_budget_status"] != "sealed"
        or reservation["slot_count"] != expected_slot_count
        or reservation["slot_budget_bytes"]
        != policy["resources"]["ram_slot_budget_bytes"]
        or reservation["reserved_bytes"]
        != reservation["slot_count"] * reservation["slot_budget_bytes"]
        or reservation["projected_used_bytes"]
        != reservation["memory_used_bytes"] + reservation["reserved_bytes"]
        or reservation["projected_used_percent"]
        != 100.0
        * reservation["projected_used_bytes"]
        / reservation["memory_total_bytes"]
        or reservation["admission_limit_percent"]
        != policy["resources"]["ram_admission_percent"]
        or reservation["projected_used_percent"]
        >= reservation["admission_limit_percent"]
        or reservation["budget_source"]
        != policy["resources"]["ram_slot_budget_source"]
    ):
        raise CanonicalScreeningError("run request RAM reservation mismatch")
    candidate_manifest = _require_mapping(
        value["candidate_manifest"], "candidate manifest binding"
    )
    _require_exact_keys(
        candidate_manifest,
        {"path", "sha256", "canonical_sha256"},
        "candidate manifest binding",
    )
    for field in ("sha256", "canonical_sha256"):
        _require_sha256(candidate_manifest[field], f"candidate manifest {field}")
    for field in ("run_request_sha256",):
        _require_sha256(value[field], field)
    expected = canonical_digest(value, "run_request_sha256")
    if value["run_request_sha256"] != expected:
        raise CanonicalScreeningError("run request digest mismatch")
    return value


def validate_final_release_admission(
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    validated_request = validate_run_request(request, policy)
    ready_values = {
        name: load_json(
            Path(str(validated_request[name]["path"])).resolve(), name
        )
        for name in ("controller_ready", "observer_ready")
    }
    bound = _require_mapping(binding, "final release admission binding")
    _require_exact_keys(
        bound,
        {"path", "sha256", "canonical_sha256"},
        "final release admission binding",
    )
    path = Path(str(bound["path"])).resolve()
    if not path.is_file() or sha256_file(path) != bound["sha256"]:
        raise CanonicalScreeningError(
            "final release admission file binding mismatch"
        )
    value = load_json(path, "final release admission")
    required = {
        "schema_version",
        "contract_type",
        "campaign_id",
        "phase",
        "policy_sha256",
        "initial_admission_sha256",
        "controller_ready_sha256",
        "observer_ready_sha256",
        "wrapper_claim",
        "wrapper_claim_sha256",
        "observer_launch",
        "observer_launch_sha256",
        "authorized_gpu_registry",
        "request_count",
        "requests",
        "snapshot",
        "released_at",
        "final_release_admission_sha256",
    }
    _require_exact_keys(value, required, "final release admission")
    requests = value["requests"]
    if not isinstance(requests, list):
        raise CanonicalScreeningError(
            "final release admission requests must be a list"
        )
    matching = [
        row
        for row in requests
        if isinstance(row, Mapping)
        and row.get("canonical_sha256")
        == validated_request["run_request_sha256"]
    ]
    if len(matching) != 1:
        raise CanonicalScreeningError(
            "final release admission does not uniquely bind run request"
        )
    matched = matching[0]
    matched_path = Path(str(matched.get("path", ""))).resolve()
    if (
        set(matched) != {"path", "sha256", "canonical_sha256"}
        or not matched_path.is_file()
        or sha256_file(matched_path) != matched["sha256"]
        or load_json(matched_path, "final release run request")
        != validated_request
    ):
        raise CanonicalScreeningError(
            "final release admission run request binding mismatch"
        )
    snapshot = _require_mapping(
        value["snapshot"], "final release admission snapshot"
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != FINAL_RELEASE_ADMISSION_CONTRACT
        or value["campaign_id"] != policy["campaign_id"]
        or value["phase"] != validated_request["mode"]
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["initial_admission_sha256"]
        != validated_request["admission"]["canonical_sha256"]
        or value["controller_ready_sha256"]
        != validated_request["controller_ready"]["canonical_sha256"]
        or value["observer_ready_sha256"]
        != validated_request["observer_ready"]["canonical_sha256"]
        or value["wrapper_claim"]
        != ready_values["controller_ready"]["wrapper_claim"]
        or value["wrapper_claim"]
        != ready_values["observer_ready"]["wrapper_claim"]
        or value["wrapper_claim_sha256"]
        != value["wrapper_claim"]["canonical_sha256"]
        or value["observer_launch"]
        != ready_values["controller_ready"]["observer_launch"]
        or value["observer_launch"]
        != ready_values["observer_ready"]["observer_launch"]
        or value["observer_launch_sha256"]
        != value["observer_launch"]["canonical_sha256"]
        or value["authorized_gpu_registry"]
        != validated_request["authorized_gpu_registry"]
        or snapshot.get("authorized_gpu_registry")
        != validated_request["authorized_gpu_registry"]
        or snapshot.get("compute_processes") != []
        or value["request_count"] != len(requests)
        or value["final_release_admission_sha256"]
        != bound["canonical_sha256"]
        or canonical_digest(
            value, "final_release_admission_sha256"
        )
        != bound["canonical_sha256"]
    ):
        raise CanonicalScreeningError(
            "final release admission contract mismatch"
        )
    return value


def _load_handshake_artifact(
    binding: Mapping[str, Any],
    *,
    label: str,
    digest_field: str,
    contract_type: str,
) -> dict[str, Any]:
    bound = _require_mapping(binding, f"{label} binding")
    _require_exact_keys(
        bound,
        {"path", "sha256", "canonical_sha256"},
        f"{label} binding",
    )
    path = Path(str(bound["path"])).resolve()
    _require_sha256(bound["sha256"], f"{label} file SHA256")
    _require_sha256(
        bound["canonical_sha256"], f"{label} canonical SHA256"
    )
    if not path.is_file() or sha256_file(path) != bound["sha256"]:
        raise CanonicalScreeningError(f"{label} file binding mismatch")
    value = load_json(path, label)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract_type") != contract_type
        or value.get(digest_field) != bound["canonical_sha256"]
        or canonical_digest(value, digest_field)
        != bound["canonical_sha256"]
    ):
        raise CanonicalScreeningError(f"{label} canonical binding mismatch")
    return value


def _validate_worker_rehashed_bindings(
    raw_bindings: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = dict(
        _require_mapping(raw_bindings, "worker rehashed bindings")
    )
    required = {
        "config",
        "implementations",
        "request",
        "candidate_manifest",
        "checkpoint_plan",
        "checkpoint",
        "data_and_evaluators",
        "final_release",
        "controller_ready",
        "observer_ready",
    }
    _require_exact_keys(bindings, required, "worker rehashed bindings")

    def exact_file_binding(
        value: Any,
        expected: Mapping[str, Any],
        label: str,
        *,
        canonical: bool,
    ) -> dict[str, Any]:
        binding = dict(_require_mapping(value, label))
        fields = {"path", "sha256", "canonical_sha256"} if canonical else {
            "path",
            "sha256",
        }
        _require_exact_keys(binding, fields, label)
        path = Path(str(binding["path"])).resolve()
        expected_path = Path(str(expected["path"])).resolve()
        if (
            path != expected_path
            or binding["sha256"] != expected["sha256"]
            or (canonical and binding["canonical_sha256"] != expected["canonical_sha256"])
            or not path.is_file()
            or sha256_file(path) != binding["sha256"]
        ):
            raise CanonicalScreeningError(f"{label} differs")
        return binding

    exact_file_binding(
        bindings["config"],
        request["policy"],
        "worker rehashed config",
        canonical=False,
    )
    implementations = dict(
        _require_mapping(
            bindings["implementations"],
            "worker rehashed implementations",
        )
    )
    if set(implementations) != set(request["implementations"]):
        raise CanonicalScreeningError(
            "worker rehashed implementation set differs"
        )
    normalized_implementations = {}
    for name, expected in request["implementations"].items():
        normalized_implementations[name] = exact_file_binding(
            implementations[name],
            expected,
            f"worker rehashed {name} implementation",
            canonical=False,
        )
    request_binding = exact_file_binding(
        bindings["request"],
        {
            "path": bindings["request"]["path"],
            "sha256": bindings["request"]["sha256"],
            "canonical_sha256": request["run_request_sha256"],
        },
        "worker rehashed request",
        canonical=True,
    )
    if (
        request_binding["canonical_sha256"]
        != request["run_request_sha256"]
        or load_json(
            Path(str(request_binding["path"])),
            "worker rehashed run request",
        )
        != dict(request)
    ):
        raise CanonicalScreeningError("worker rehashed request differs")
    candidate_manifest = exact_file_binding(
        bindings["candidate_manifest"],
        request["candidate_manifest"],
        "worker rehashed candidate manifest",
        canonical=True,
    )
    manifest_value = load_json(
        Path(str(candidate_manifest["path"])),
        "worker rehashed candidate manifest",
    )
    checkpoint_plan_expected = _require_mapping(
        manifest_value.get("checkpoint_plan"),
        "worker rehashed checkpoint plan expected binding",
    )
    checkpoint_plan = exact_file_binding(
        bindings["checkpoint_plan"],
        checkpoint_plan_expected,
        "worker rehashed checkpoint plan",
        canonical=True,
    )
    checkpoint = exact_file_binding(
        bindings["checkpoint"],
        {
            "path": request["candidate"]["checkpoint_path"],
            "sha256": request["candidate"]["checkpoint_sha256"],
        },
        "worker rehashed checkpoint",
        canonical=False,
    )
    data_and_evaluators = dict(
        _require_mapping(
            bindings["data_and_evaluators"],
            "worker rehashed data and evaluators",
        )
    )
    data_fields = {
        "sample_manifest",
        "source_index",
        "features",
        "e0",
        "edev",
        "quality_script",
        "pixel_protocol_config",
        "arcface",
    }
    _require_exact_keys(
        data_and_evaluators,
        data_fields,
        "worker rehashed data and evaluators",
    )
    if data_and_evaluators != {
        name: request[name] for name in data_fields
    }:
        raise CanonicalScreeningError(
            "worker rehashed data/evaluator bindings differ"
        )
    final_release = dict(
        _require_mapping(
            bindings["final_release"], "worker rehashed final release"
        )
    )
    validate_final_release_admission(final_release, request, policy)
    if (
        bindings["controller_ready"] != request["controller_ready"]
        or bindings["observer_ready"] != request["observer_ready"]
    ):
        raise CanonicalScreeningError(
            "worker rehashed ready barrier bindings differ"
        )
    return {
        "config": dict(bindings["config"]),
        "implementations": normalized_implementations,
        "request": request_binding,
        "candidate_manifest": candidate_manifest,
        "checkpoint_plan": checkpoint_plan,
        "checkpoint": checkpoint,
        "data_and_evaluators": data_and_evaluators,
        "final_release": final_release,
        "controller_ready": dict(bindings["controller_ready"]),
        "observer_ready": dict(bindings["observer_ready"]),
    }


def _validate_controller_resource_snapshot(
    raw_snapshot: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = dict(
        _require_mapping(raw_snapshot, "controller resource snapshot")
    )
    _require_exact_keys(
        snapshot,
        {
            "observed_at",
            "cpu_load_percent",
            "memory_percent",
            "disk_percent",
            "swap_pages",
            "gpus",
            "authorized_gpu_registry",
            "ram_reservation",
            "compute_processes",
        },
        "controller resource snapshot",
    )
    for field, limit in (
        ("cpu_load_percent", policy["resources"]["cpu_admission_percent"]),
        ("memory_percent", policy["resources"]["ram_admission_percent"]),
        ("disk_percent", policy["resources"]["disk_admission_percent"]),
    ):
        value = snapshot[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) >= float(limit)
        ):
            raise CanonicalScreeningError(
                f"controller resource snapshot {field} differs"
            )
    if snapshot["authorized_gpu_registry"] != request[
        "authorized_gpu_registry"
    ]:
        raise CanonicalScreeningError(
            "controller resource GPU registry differs"
        )
    gpus = snapshot["gpus"]
    if (
        not isinstance(gpus, list)
        or [row.get("index") for row in gpus]
        != policy["resources"]["physical_gpus"]
        or [
            {
                "physical_gpu_index": row.get("index"),
                "physical_gpu_uuid": row.get("uuid"),
            }
            for row in gpus
        ]
        != request["authorized_gpu_registry"]
    ):
        raise CanonicalScreeningError(
            "controller resource GPU snapshot differs"
        )
    if (
        not isinstance(snapshot["compute_processes"], list)
        or not isinstance(snapshot["ram_reservation"], Mapping)
        or set(_require_mapping(snapshot["swap_pages"], "resource swap pages"))
        != {"in", "out"}
    ):
        raise CanonicalScreeningError(
            "controller resource snapshot structure differs"
        )
    return snapshot


def _validate_asset_content_verification(
    raw_verification: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    verification = dict(
        _require_mapping(
            raw_verification, "asset content verification"
        )
    )
    _require_exact_keys(
        verification,
        {
            "schema_version",
            "contract_type",
            "path",
            "digest_algorithm",
            "expected_digest",
            "observed_digest",
            "file_count",
            "total_bytes",
            "elapsed_seconds",
            "started_at",
            "completed_at",
        },
        "asset content verification",
    )
    directory = policy["output_decoder_registry"]["latent"]["directory"]
    elapsed = verification["elapsed_seconds"]
    if (
        verification["schema_version"] != SCHEMA_VERSION
        or verification["contract_type"]
        != "safa_canonical_asset_content_verification_v1"
        or Path(str(verification["path"])).resolve()
        != Path(str(directory["path"])).resolve()
        or verification["digest_algorithm"]
        != "sha256_relative_posix_nul_content_nul_v1"
        or verification["expected_digest"] != directory["digest"]
        or verification["observed_digest"] != directory["digest"]
        or type(verification["file_count"]) is not int
        or verification["file_count"] <= 0
        or type(verification["total_bytes"]) is not int
        or verification["total_bytes"] <= 0
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
        or not isinstance(verification["started_at"], str)
        or not verification["started_at"]
        or not isinstance(verification["completed_at"], str)
        or not verification["completed_at"]
    ):
        raise CanonicalScreeningError(
            "asset content verification contract mismatch"
        )
    return verification


def _validate_runtime_resource_snapshot(
    raw_snapshot: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    worker_pid: int,
) -> dict[str, Any]:
    snapshot = dict(
        _require_mapping(raw_snapshot, "runtime guard resource snapshot")
    )
    _require_exact_keys(
        snapshot,
        {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "observed_at",
            "runtime_gpu_registry",
            "compute_processes",
            "unknown_compute_processes",
            "cpu_load_percent",
            "memory_percent",
            "disk_percent",
            "swap_pages_before",
            "swap_pages_after",
            "swap_io_delta",
            "swap_consecutive_io",
            "gpu",
            "active_worker_pids",
            "hard_limits",
            "guard_thread_failure",
            "guard_violation_reason",
        },
        "runtime guard resource snapshot",
    )
    expected_limits = {
        "cpu_percent": policy["resources"]["cpu_hard_limit_percent"],
        "ram_percent": policy["resources"]["ram_hard_limit_percent"],
        "disk_percent": policy["resources"]["disk_hard_limit_percent"],
        "gpu_memory_percent": 90.0,
        "gpu_temperature_c": 85,
        "gpu_free_mib": int(
            policy["resources"]["gpu_headroom_bytes"]
        ) // 1024**2,
        "swap_io_delta_pages": 0,
        "swap_consecutive_io": 0,
    }
    if (
        snapshot["schema_version"] != SCHEMA_VERSION
        or snapshot["contract_type"]
        != "safa_canonical_worker_release_resource_snapshot_v2"
        or snapshot["policy_sha256"] != policy["policy_sha256"]
        or not isinstance(snapshot["observed_at"], str)
        or not snapshot["observed_at"]
        or snapshot["hard_limits"] != expected_limits
        or snapshot["guard_thread_failure"] is not None
        or snapshot["guard_violation_reason"] is not None
        or snapshot["runtime_gpu_registry"]
        != request["authorized_gpu_registry"]
        or snapshot["unknown_compute_processes"] != []
        or not isinstance(snapshot["compute_processes"], list)
        or not isinstance(snapshot["gpu"], list)
        or not isinstance(snapshot["active_worker_pids"], list)
        or worker_pid not in snapshot["active_worker_pids"]
        or any(
            type(pid) is not int or pid <= 0
            for pid in snapshot["active_worker_pids"]
        )
    ):
        raise CanonicalScreeningError(
            "runtime guard resource snapshot differs"
        )
    for field, limit in (
        ("cpu_load_percent", policy["resources"]["cpu_hard_limit_percent"]),
        ("memory_percent", policy["resources"]["ram_hard_limit_percent"]),
        ("disk_percent", policy["resources"]["disk_hard_limit_percent"]),
    ):
        value = snapshot[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) >= float(limit)
        ):
            raise CanonicalScreeningError(
                f"runtime guard resource {field} differs"
            )
    for field in ("swap_pages_before", "swap_pages_after", "swap_io_delta"):
        value = dict(
            _require_mapping(
                snapshot[field], f"runtime guard resource {field}"
            )
        )
        _require_exact_keys(
            value, {"in", "out"}, f"runtime guard resource {field}"
        )
        if any(type(count) is not int or count < 0 for count in value.values()):
            raise CanonicalScreeningError(
                f"runtime guard resource {field} differs"
            )
    if (
        snapshot["swap_io_delta"] != {"in": 0, "out": 0}
        or snapshot["swap_pages_after"] != snapshot["swap_pages_before"]
        or snapshot["swap_consecutive_io"] != 0
    ):
        raise CanonicalScreeningError(
            "runtime guard release swap state differs"
        )
    gpu_rows = snapshot["gpu"]
    if (
        [
            {
                "physical_gpu_index": row.get("index"),
                "physical_gpu_uuid": row.get("uuid"),
            }
            for row in gpu_rows
        ]
        != request["authorized_gpu_registry"]
    ):
        raise CanonicalScreeningError(
            "runtime guard release GPU registry differs"
        )
    allowed_gpu_uuids = {
        row["physical_gpu_uuid"]
        for row in request["authorized_gpu_registry"]
    }
    for row in snapshot["runtime_gpu_registry"]:
        _require_exact_keys(
            _require_mapping(row, "runtime guard GPU registry row"),
            {"physical_gpu_index", "physical_gpu_uuid"},
            "runtime guard GPU registry row",
        )
    for row in snapshot["compute_processes"]:
        process = _require_mapping(
            row, "runtime guard compute process row"
        )
        _require_exact_keys(
            process,
            {"gpu_uuid", "pid", "process_name", "used_memory_mib"},
            "runtime guard compute process row",
        )
        if (
            process["gpu_uuid"] not in allowed_gpu_uuids
            or type(process["pid"]) is not int
            or process["pid"] <= 0
            or not isinstance(process["process_name"], str)
            or not process["process_name"]
            or not isinstance(process["used_memory_mib"], str)
            or not process["used_memory_mib"]
        ):
            raise CanonicalScreeningError(
                "runtime guard compute process row differs"
            )
    for row in gpu_rows:
        _require_exact_keys(
            _require_mapping(row, "runtime guard GPU resource row"),
            {
                "index",
                "uuid",
                "memory_total_mib",
                "memory_used_mib",
                "memory_free_mib",
                "temperature_c",
            },
            "runtime guard GPU resource row",
        )
        total = row.get("memory_total_mib")
        used = row.get("memory_used_mib")
        free = row.get("memory_free_mib")
        temperature = row.get("temperature_c")
        if (
            type(total) is not int
            or total <= 0
            or type(used) is not int
            or used < 0
            or used > total
            or type(free) is not int
            or free < 0
            or free > total
            or used + free != total
            or type(temperature) is not int
            or temperature > 85
            or 100.0 * used / total >= 90.0
            or free < expected_limits["gpu_free_mib"]
        ):
            raise CanonicalScreeningError(
                "runtime guard release GPU resource state differs"
            )
    return snapshot


def validate_worker_ready_value(
    raw_value: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    expected_worker_pid: int | None = None,
    expected_gpu_index: int | None = None,
    expected_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    value = dict(_require_mapping(raw_value, "worker ready"))
    required = {
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
    }
    _require_exact_keys(value, required, "worker ready")
    worker_pid = value["worker_pid"]
    gpu_index = value["gpu_index"]
    gpu_uuid = value["gpu_uuid"]
    registry = {
        row["physical_gpu_index"]: row["physical_gpu_uuid"]
        for row in request["authorized_gpu_registry"]
    }
    rehashed = _validate_worker_rehashed_bindings(
        value["rehashed_bindings"], request, policy
    )
    _validate_asset_content_verification(
        value["asset_content_verification"], policy
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != WORKER_READY_CONTRACT
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["phase"] != request["mode"]
        or type(worker_pid) is not int
        or worker_pid <= 0
        or (expected_worker_pid is not None and worker_pid != expected_worker_pid)
        or type(gpu_index) is not int
        or gpu_index not in registry
        or (
            expected_gpu_index is not None
            and gpu_index != expected_gpu_index
        )
        or gpu_uuid != registry[gpu_index]
        or (
            expected_gpu_uuid is not None
            and gpu_uuid != expected_gpu_uuid
        )
        or value["cuda_visible_devices"] != gpu_uuid
        or value["run_request_sha256"] != request["run_request_sha256"]
        or value["request"] != rehashed["request"]
        or value["final_release"] != rehashed["final_release"]
        or value["verification_order"]
        != list(WORKER_PRE_CUDA_VERIFICATION_ORDER)
        or value["rehashed_bindings_sha256"]
        != hashlib.sha256(canonical_json(rehashed)).hexdigest()
        or value["screening_worker_sha256"]
        != request["implementations"]["screening_worker"]["sha256"]
        or value["controller_implementation_sha256"]
        != request["implementations"]["controller"]["sha256"]
        or value["heavy_modules_absent"] is not True
        or value["loaded_heavy_modules"] != []
        or value["external_gpu_race_contract"]
        != WORKER_EXTERNAL_GPU_RACE_CONTRACT
        or not isinstance(value["ready_at"], str)
        or not value["ready_at"]
        or value["worker_ready_sha256"]
        != canonical_digest(value, "worker_ready_sha256")
    ):
        raise CanonicalScreeningError("worker ready contract mismatch")
    _load_handshake_artifact(
        value["controller_claim"],
        label="worker ready controller claim",
        digest_field="controller_claim_sha256",
        contract_type="safa_canonical_gpu_controller_claim_v1",
    )
    controller_ready = load_json(
        Path(str(request["controller_ready"]["path"])),
        "worker ready controller barrier",
    )
    if value["controller_claim"] != controller_ready.get(
        "controller_claim"
    ):
        raise CanonicalScreeningError(
            "worker ready controller claim binding differs"
        )
    return value


def validate_controller_launch_rehash_value(
    raw_value: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(
        _require_mapping(raw_value, "controller launch rehash")
    )
    required = {
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
    }
    _require_exact_keys(value, required, "controller launch rehash")
    ready = _load_handshake_artifact(
        value["worker_ready"],
        label="controller launch worker ready",
        digest_field="worker_ready_sha256",
        contract_type=WORKER_READY_CONTRACT,
    )
    validate_worker_ready_value(
        ready,
        request,
        policy,
        expected_worker_pid=value["worker_pid"],
        expected_gpu_index=value["gpu_index"],
        expected_gpu_uuid=value["gpu_uuid"],
    )
    rehashed = _validate_worker_rehashed_bindings(
        value["rehashed_bindings"], request, policy
    )
    _validate_controller_resource_snapshot(
        value["resource_snapshot"], request, policy
    )
    _validate_asset_content_verification(
        value["asset_content_verification"], policy
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != CONTROLLER_LAUNCH_REHASH_CONTRACT
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["run_request_sha256"] != request["run_request_sha256"]
        or value["verification_order"]
        != list(WORKER_PRE_CUDA_VERIFICATION_ORDER)
        or value["rehashed_bindings"] != ready["rehashed_bindings"]
        or value["rehashed_bindings_sha256"]
        != hashlib.sha256(canonical_json(rehashed)).hexdigest()
        or value["rehashed_bindings_sha256"]
        != ready["rehashed_bindings_sha256"]
        or value["external_gpu_race_contract"]
        != WORKER_EXTERNAL_GPU_RACE_CONTRACT
        or not isinstance(value["validated_at"], str)
        or not value["validated_at"]
        or value["controller_launch_rehash_sha256"]
        != canonical_digest(
            value, "controller_launch_rehash_sha256"
        )
    ):
        raise CanonicalScreeningError(
            "controller launch rehash contract mismatch"
        )
    return value


def validate_worker_release_value(
    raw_value: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    expected_worker_pid: int | None = None,
) -> dict[str, Any]:
    value = dict(_require_mapping(raw_value, "worker release"))
    required = {
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
    }
    _require_exact_keys(value, required, "worker release")
    worker_pid = value["worker_pid"]
    ready = _load_handshake_artifact(
        value["worker_ready"],
        label="worker release ready",
        digest_field="worker_ready_sha256",
        contract_type=WORKER_READY_CONTRACT,
    )
    validate_worker_ready_value(
        ready,
        request,
        policy,
        expected_worker_pid=worker_pid,
    )
    controller_rehash = _load_handshake_artifact(
        value["controller_launch_rehash"],
        label="worker release controller launch rehash",
        digest_field="controller_launch_rehash_sha256",
        contract_type=CONTROLLER_LAUNCH_REHASH_CONTRACT,
    )
    validate_controller_launch_rehash_value(
        controller_rehash, request, policy
    )
    resource_snapshot = dict(
        _require_mapping(
            value["resource_snapshot"], "worker release resource snapshot"
        )
    )
    _require_exact_keys(
        resource_snapshot,
        {"admission", "runtime_guard"},
        "worker release resource snapshot",
    )
    _validate_runtime_resource_snapshot(
        resource_snapshot["runtime_guard"],
        request,
        policy,
        worker_pid,
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != WORKER_RELEASE_CONTRACT
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["phase"] != request["mode"]
        or type(worker_pid) is not int
        or worker_pid <= 0
        or (expected_worker_pid is not None and worker_pid != expected_worker_pid)
        or value["run_request_sha256"] != request["run_request_sha256"]
        or controller_rehash["worker_pid"] != worker_pid
        or controller_rehash["worker_ready"] != value["worker_ready"]
        or resource_snapshot["admission"]
        != controller_rehash["resource_snapshot"]
        or value["external_gpu_race_contract"]
        != WORKER_EXTERNAL_GPU_RACE_CONTRACT
        or value["external_gpu_race_contract"]
        != controller_rehash["external_gpu_race_contract"]
        or not isinstance(value["released_at"], str)
        or not value["released_at"]
        or value["worker_release_sha256"]
        != canonical_digest(value, "worker_release_sha256")
    ):
        raise CanonicalScreeningError(
            "worker release contract mismatch"
        )
    return value


def validate_worker_terminal_value(
    raw_value: Mapping[str, Any],
    request_path: Path,
    policy: Mapping[str, Any],
    *,
    expected_worker_pid: int | None = None,
    require_completed: bool = False,
) -> dict[str, Any]:
    value = dict(_require_mapping(raw_value, "worker terminal"))
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_type",
            "policy_sha256",
            "worker_pid",
            "request",
            "claim",
            "result",
            "worker_ready",
            "worker_release",
            "status",
            "failure",
            "started_at",
            "completed_at",
            "worker_terminal_sha256",
        },
        "worker terminal",
    )
    request = _load_handshake_artifact(
        value["request"],
        label="worker terminal request",
        digest_field="run_request_sha256",
        contract_type=RUN_REQUEST_CONTRACT,
    )
    validate_run_request(request, policy)
    if Path(str(value["request"]["path"])).resolve() != request_path.resolve():
        raise CanonicalScreeningError(
            "worker terminal request path differs"
        )
    worker_pid = value["worker_pid"]
    claim = _load_handshake_artifact(
        value["claim"],
        label="worker terminal claim",
        digest_field="run_claim_sha256",
        contract_type=RUN_CLAIM_CONTRACT,
    )
    validate_run_claim(claim, request, policy)
    result = _load_handshake_artifact(
        value["result"],
        label="worker terminal result",
        digest_field="run_result_sha256",
        contract_type=RUN_RESULT_CONTRACT,
    )
    validate_run_result(result, request, claim, policy)
    ready = _load_handshake_artifact(
        value["worker_ready"],
        label="worker terminal ready",
        digest_field="worker_ready_sha256",
        contract_type=WORKER_READY_CONTRACT,
    )
    validate_worker_ready_value(
        ready, request, policy, expected_worker_pid=worker_pid
    )
    release = _load_handshake_artifact(
        value["worker_release"],
        label="worker terminal release",
        digest_field="worker_release_sha256",
        contract_type=WORKER_RELEASE_CONTRACT,
    )
    validate_worker_release_value(
        release, request, policy, expected_worker_pid=worker_pid
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != "safa_canonical_worker_terminal_v1"
        or value["policy_sha256"] != policy["policy_sha256"]
        or type(worker_pid) is not int
        or worker_pid <= 0
        or (
            expected_worker_pid is not None
            and worker_pid != expected_worker_pid
        )
        or value["request"]["canonical_sha256"]
        != request["run_request_sha256"]
        or value["claim"]["canonical_sha256"]
        != claim["run_claim_sha256"]
        or value["result"]["canonical_sha256"]
        != result["run_result_sha256"]
        or value["worker_ready"] != claim["worker_ready"]
        or value["worker_release"] != claim["worker_release"]
        or value["status"] not in {"completed", "failed"}
        or value["status"] != result["status"]
        or (value["status"] == "completed" and value["failure"] is not None)
        or (
            value["status"] == "failed"
            and not isinstance(value["failure"], Mapping)
        )
        or not isinstance(value["started_at"], str)
        or not value["started_at"]
        or not isinstance(value["completed_at"], str)
        or not value["completed_at"]
        or (
            require_completed
            and (
                value["status"] != "completed"
                or result["status"] != "completed"
            )
        )
        or value["worker_terminal_sha256"]
        != canonical_digest(value, "worker_terminal_sha256")
    ):
        raise CanonicalScreeningError(
            "worker terminal contract mismatch"
        )
    return value


def _validate_worker_handshake_bindings(
    worker_ready: Mapping[str, Any],
    worker_release: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    worker_pid: int,
) -> None:
    ready_value = _load_handshake_artifact(
        worker_ready,
        label="worker ready",
        digest_field="worker_ready_sha256",
        contract_type=WORKER_READY_CONTRACT,
    )
    validate_worker_ready_value(
        ready_value,
        request,
        policy,
        expected_worker_pid=worker_pid,
    )
    release_value = _load_handshake_artifact(
        worker_release,
        label="worker release",
        digest_field="worker_release_sha256",
        contract_type=WORKER_RELEASE_CONTRACT,
    )
    validate_worker_release_value(
        release_value,
        request,
        policy,
        expected_worker_pid=worker_pid,
    )
    if release_value["worker_ready"] != dict(worker_ready):
        raise CanonicalScreeningError(
            "worker release does not bind worker ready"
        )


def build_run_claim(
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    final_release_admission: Mapping[str, Any],
    worker_ready: Mapping[str, Any],
    worker_release: Mapping[str, Any],
    gpu_index: int,
    gpu_uuid: str,
    runtime_cuda_uuid: str,
    cuda_visible_devices: str,
    worker_pid: int,
    started_at: str,
) -> dict[str, Any]:
    validate_run_request(request, policy)
    validated_release = validate_final_release_admission(
        final_release_admission, request, policy
    )
    if gpu_index not in {0, 1, 2, 3}:
        raise CanonicalScreeningError("screening GPU must be one of 0..3")
    if type(worker_pid) is not int or worker_pid <= 0:
        raise CanonicalScreeningError("worker PID must be positive")
    _validate_worker_handshake_bindings(
        worker_ready, worker_release, request, policy, worker_pid
    )
    claim = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": RUN_CLAIM_CONTRACT,
        "run_request_sha256": request["run_request_sha256"],
        "admission_sha256": request["admission"]["canonical_sha256"],
        "controller_ready_sha256": request["controller_ready"]["canonical_sha256"],
        "observer_ready_sha256": request["observer_ready"]["canonical_sha256"],
        "final_release_admission": dict(final_release_admission),
        "final_release_admission_sha256": validated_release[
            "final_release_admission_sha256"
        ],
        "worker_ready": dict(worker_ready),
        "worker_ready_sha256": worker_ready["canonical_sha256"],
        "worker_release": dict(worker_release),
        "worker_release_sha256": worker_release["canonical_sha256"],
        "physical_gpu_index": gpu_index,
        "physical_gpu_uuid": gpu_uuid,
        "logical_cuda_index": 0,
        "runtime_cuda_uuid": runtime_cuda_uuid,
        "cuda_visible_devices": cuda_visible_devices,
        "ram_slot_budget_bytes": request["ram_reservation"][
            "slot_budget_bytes"
        ],
        "worker_pid": worker_pid,
        "started_at": started_at,
    }
    claim["run_claim_sha256"] = canonical_digest(claim, "run_claim_sha256")
    return claim


def validate_run_claim(
    claim: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    validated_request = validate_run_request(request, policy)
    value = dict(claim)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_type",
            "run_request_sha256",
            "admission_sha256",
            "controller_ready_sha256",
            "observer_ready_sha256",
            "final_release_admission",
            "final_release_admission_sha256",
            "worker_ready",
            "worker_ready_sha256",
            "worker_release",
            "worker_release_sha256",
            "physical_gpu_index",
            "physical_gpu_uuid",
            "logical_cuda_index",
            "runtime_cuda_uuid",
            "cuda_visible_devices",
            "ram_slot_budget_bytes",
            "worker_pid",
            "started_at",
            "run_claim_sha256",
        },
        "run claim",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != RUN_CLAIM_CONTRACT
        or value["run_request_sha256"] != validated_request["run_request_sha256"]
        or value["admission_sha256"]
        != validated_request["admission"]["canonical_sha256"]
        or value["controller_ready_sha256"]
        != validated_request["controller_ready"]["canonical_sha256"]
        or value["observer_ready_sha256"]
        != validated_request["observer_ready"]["canonical_sha256"]
        or value["final_release_admission_sha256"]
        != validate_final_release_admission(
            value["final_release_admission"], validated_request, policy
        )["final_release_admission_sha256"]
        or value["worker_ready_sha256"]
        != value["worker_ready"]["canonical_sha256"]
        or value["worker_release_sha256"]
        != value["worker_release"]["canonical_sha256"]
        or value["physical_gpu_index"]
        not in policy["resources"]["physical_gpus"]
        or type(value["worker_pid"]) is not int
        or value["worker_pid"] <= 0
    ):
        raise CanonicalScreeningError("run claim binding mismatch")
    _validate_worker_handshake_bindings(
        value["worker_ready"],
        value["worker_release"],
        validated_request,
        policy,
        value["worker_pid"],
    )
    registry = {
        row["physical_gpu_index"]: row["physical_gpu_uuid"]
        for row in validated_request["authorized_gpu_registry"]
    }
    expected_uuid = registry[value["physical_gpu_index"]]
    if (
        value["physical_gpu_uuid"] != expected_uuid
        or value["runtime_cuda_uuid"] != expected_uuid
        or value["cuda_visible_devices"] != expected_uuid
        or value["logical_cuda_index"] != 0
        or value["ram_slot_budget_bytes"]
        != validated_request["ram_reservation"]["slot_budget_bytes"]
    ):
        raise CanonicalScreeningError("run claim CUDA/RAM binding mismatch")
    _require_sha256(value["run_claim_sha256"], "run claim SHA256")
    if value["run_claim_sha256"] != canonical_digest(value, "run_claim_sha256"):
        raise CanonicalScreeningError("run claim digest mismatch")
    return value


def build_run_result(
    request: Mapping[str, Any],
    claim: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    status: str,
    completed_at: str,
    evidence: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_run_request(request, policy)
    validate_run_claim(claim, request, policy)
    if status not in {"completed", "failed"}:
        raise CanonicalScreeningError("run result status must be completed or failed")
    if claim.get("run_request_sha256") != request["run_request_sha256"]:
        raise CanonicalScreeningError("run claim/request binding mismatch")
    if (status == "completed") != (evidence is not None) or (
        status == "failed"
    ) != (failure is not None):
        raise CanonicalScreeningError("run result evidence/failure fields disagree")
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": RUN_RESULT_CONTRACT,
        "run_request_sha256": request["run_request_sha256"],
        "run_claim_sha256": claim["run_claim_sha256"],
        "controller_ready_sha256": claim["controller_ready_sha256"],
        "observer_ready_sha256": claim["observer_ready_sha256"],
        "final_release_admission_sha256": claim[
            "final_release_admission_sha256"
        ],
        "worker_ready_sha256": claim["worker_ready_sha256"],
        "worker_release_sha256": claim["worker_release_sha256"],
        "device_binding": {
            "physical_gpu_index": claim["physical_gpu_index"],
            "physical_gpu_uuid": claim["physical_gpu_uuid"],
            "logical_cuda_index": claim["logical_cuda_index"],
            "runtime_cuda_uuid": claim["runtime_cuda_uuid"],
            "cuda_visible_devices": claim["cuda_visible_devices"],
        },
        "ram_slot_budget_bytes": claim["ram_slot_budget_bytes"],
        "status": status,
        "completed_at": completed_at,
        "evidence": None if evidence is None else dict(evidence),
        "failure": None if failure is None else dict(failure),
    }
    result["run_result_sha256"] = canonical_digest(result, "run_result_sha256")
    return result


def validate_run_result(
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    claim: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    validated_request = validate_run_request(request, policy)
    validated_claim = validate_run_claim(claim, validated_request, policy)
    value = dict(result)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_type",
            "run_request_sha256",
            "run_claim_sha256",
            "controller_ready_sha256",
            "observer_ready_sha256",
            "final_release_admission_sha256",
            "worker_ready_sha256",
            "worker_release_sha256",
            "device_binding",
            "ram_slot_budget_bytes",
            "status",
            "completed_at",
            "evidence",
            "failure",
            "run_result_sha256",
        },
        "run result",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != RUN_RESULT_CONTRACT
        or value["run_request_sha256"] != validated_request["run_request_sha256"]
        or value["run_claim_sha256"] != validated_claim["run_claim_sha256"]
        or value["controller_ready_sha256"]
        != validated_claim["controller_ready_sha256"]
        or value["observer_ready_sha256"]
        != validated_claim["observer_ready_sha256"]
        or value["final_release_admission_sha256"]
        != validated_claim["final_release_admission_sha256"]
        or value["worker_ready_sha256"]
        != validated_claim["worker_ready_sha256"]
        or value["worker_release_sha256"]
        != validated_claim["worker_release_sha256"]
        or value["device_binding"]
        != {
            "physical_gpu_index": validated_claim["physical_gpu_index"],
            "physical_gpu_uuid": validated_claim["physical_gpu_uuid"],
            "logical_cuda_index": validated_claim["logical_cuda_index"],
            "runtime_cuda_uuid": validated_claim["runtime_cuda_uuid"],
            "cuda_visible_devices": validated_claim["cuda_visible_devices"],
        }
        or value["ram_slot_budget_bytes"]
        != validated_claim["ram_slot_budget_bytes"]
        or value["status"] not in {"completed", "failed"}
    ):
        raise CanonicalScreeningError("run result binding mismatch")
    _require_sha256(value["run_result_sha256"], "run result SHA256")
    if value["run_result_sha256"] != canonical_digest(value, "run_result_sha256"):
        raise CanonicalScreeningError("run result digest mismatch")
    if value["status"] == "completed":
        if value["failure"] is not None:
            raise CanonicalScreeningError("completed run result contains failure")
        evidence = _require_mapping(value["evidence"], "completed run evidence")
        _require_exact_keys(
            evidence,
            {
                "mode",
                "replicate",
                "seed",
                "batch_size",
                "sample_count",
                "sample_manifest_sha256",
                "candidate_manifest_sha256",
                "policy_sha256",
                "implementations",
                "checkpoint_sha256",
                "checkpoint_model",
                "output_contract_sha256",
                "output_contract_type",
                "decoder_registry_sha256",
                "output_space",
                "native_rgb_size",
                "quality_protocol_family",
                "nfe",
                "pixel_image_size",
                "pixel_protocol_config_sha256",
                "kid_subset_size",
                "e0_mean",
                "edev_mean",
                "arcface",
                "quality",
                "per_sample_sha256",
            },
            "completed run evidence",
        )
        expected = {
            "mode": validated_request["mode"],
            "replicate": validated_request["replicate"],
            "seed": validated_request["seed"],
            "batch_size": validated_request["batch_size"],
            "sample_count": validated_request["sample_count"],
            "sample_manifest_sha256": validated_request["sample_manifest"]["sha256"],
            "candidate_manifest_sha256": validated_request["candidate_manifest"][
                "canonical_sha256"
            ],
            "policy_sha256": policy["policy_sha256"],
            "implementations": policy["implementations"],
            "checkpoint_sha256": validated_request["candidate"]["checkpoint_sha256"],
            "checkpoint_model": validated_request["candidate"]["checkpoint_model"],
            "output_contract_sha256": validated_request["output_contract"][
                "output_contract_sha256"
            ],
            "output_contract_type": validated_request["output_contract"][
                "contract_type"
            ],
            "decoder_registry_sha256": validated_request[
                "output_decoder_registry"
            ]["decoder_registry_sha256"],
            "output_space": validated_request["output_contract"]["capability"][
                "output_space"
            ],
            "native_rgb_size": validated_request["native_rgb_size"],
            "quality_protocol_family": validated_request[
                "quality_protocol_family"
            ],
            "nfe": validated_request["nfe"],
            "pixel_image_size": policy["protocol"]["pixel_image_size"],
            "pixel_protocol_config_sha256": policy["protocol"][
                "pixel_protocol_config"
            ]["sha256"],
            "kid_subset_size": policy["protocol"]["kid_subset_sizes"][
                validated_request["mode"]
            ],
        }
        for field, expected_value in expected.items():
            if evidence[field] != expected_value:
                raise CanonicalScreeningError(
                    f"completed run evidence {field} binding mismatch"
                )
        _require_sha256(evidence["per_sample_sha256"], "per-sample SHA256")
    elif value["evidence"] is not None or not isinstance(value["failure"], Mapping):
        raise CanonicalScreeningError("failed run result evidence/failure mismatch")
    return value


def write_preflight_requests(
    plan: Mapping[str, Any], request_root: Path
) -> list[Path]:
    written: list[Path] = []
    for request in plan.get("preflight_requests", []):
        path = request_root / (
            f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json"
        )
        write_exclusive_json(path, request)
        written.append(path)
    return written


def iter_run_requests(
    policy: Mapping[str, Any],
    policy_path: Path,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    mode: str,
    replicate: str,
    output_root: Path,
    admission: Mapping[str, Any],
    controller_ready: Mapping[str, Any],
    observer_ready: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    for candidate in candidate_manifest.get("candidates", []):
        yield build_run_request(
            policy,
            policy_path,
            candidate_manifest,
            candidate_manifest_path,
            candidate,
            mode,
            replicate,
            output_root,
            admission,
            controller_ready,
            observer_ready,
        )
