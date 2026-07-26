"""Fail-closed contracts for the historical canonical 512 screening campaign."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
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
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_MODES = {"smoke8": 8, "screen512": 512}
CHECKPOINT_SELECTORS = {"raw", "ema"}


class CanonicalScreeningError(RuntimeError):
    """A fail-closed canonical screening contract violation."""


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
    return _validate_ea7_smoke_supersession_evidence(repo_root, supersedes)


def _validate_output_decoder_registry(
    repo_root: Path,
    raw_registry: Mapping[str, Any],
    *,
    verify_historical_evidence: bool,
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
    if verify_historical_evidence:
        from safa.closeout.generator_output_contract import digest_asset_directory

        if digest_asset_directory(directory_path) != directory["digest"]:
            raise CanonicalScreeningError("latent decoder directory digest differs")
    else:
        from safa.evaluation.meanflow_guidance_runner import cached_asset_digest

        if (
            cached_asset_digest(
                directory_path,
                directory["digest"],
                cache_path,
            )
            != directory["digest"]
        ):
            raise CanonicalScreeningError("latent cached asset digest differs")
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
    _require_exact_keys(
        source,
        {
            "contract_type",
            "method",
            "measurement_factor_numerator",
            "measurement_factor_denominator",
            "peak_sampled_process_tree_rss_bytes",
            "worker_vmhwm_bytes",
            "ram_budget_basis_bytes",
            "ram_slot_budget_bytes",
            "probe_result",
        },
        "RAM slot budget source",
    )
    probe_binding = _validate_bound_file(
        repo_root, source["probe_result"], "RAM slot budget probe result"
    )
    probe_path = Path(probe_binding["path"])
    result = load_json(probe_path, "RAM slot budget probe result")
    _require_exact_keys(
        result,
        {
            "schema_version",
            "contract_type",
            "status",
            "purpose",
            "probe_sha256",
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
        },
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
        != "safa_canonical_screening_ram_probe_result_v1"
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
        "probe_sha256",
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
    if (
        spec.get("contract_type") != "safa_canonical_screening_ram_probe_v1"
        or spec.get("purpose") != result["purpose"]
        or spec.get("probe_sha256")
        != canonical_digest(spec, "probe_sha256")
        or spec.get("probe_sha256") != result["probe_sha256"]
        or admission.get("contract_type")
        != "safa_canonical_screening_ram_probe_admission_v1"
        or admission.get("admission_sha256")
        != canonical_digest(admission, "admission_sha256")
        or admission.get("admission_sha256") != result["admission_sha256"]
        or admission.get("probe_sha256") != result["probe_sha256"]
        or worker.get("contract_type")
        != "safa_canonical_screening_ram_probe_worker_result_v1"
        or worker.get("worker_result_sha256")
        != canonical_digest(worker, "worker_result_sha256")
        or worker.get("worker_result_sha256")
        != result["worker_result_sha256"]
        or worker.get("probe_sha256") != result["probe_sha256"]
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
        or expected_predecessor_policy_sha256 != probe_policy["policy_sha256"]
    ):
        raise CanonicalScreeningError(
            "sealed RAM probe predecessor policy binding differs"
        )
    implementations = _require_mapping(
        spec.get("implementations"), "RAM probe implementations"
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
        != "safa_canonical_screening_ram_budget_source_v1"
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
    return {
        **dict(source),
        "probe_result": probe_binding,
    }


def validate_policy(
    repo_root: Path,
    policy_path: Path,
    *,
    verify_historical_output_evidence: bool = True,
    policy_identity_path: Path | None = None,
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
                expected_predecessor_policy_sha256=bound_supersedes[
                    "policy_sha256"
                ],
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
    arcface["execution_probe"] = _validate_bound_file(
        root, arcface["execution_probe"], "ArcFace execution probe"
    )
    output_decoder_registry = _validate_output_decoder_registry(
        root,
        raw["output_decoder_registry"],
        verify_historical_evidence=verify_historical_output_evidence,
    )

    implementations = _require_mapping(raw["implementations"], "implementations")
    _require_exact_keys(
        implementations,
        {
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
        },
        "implementations",
    )
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


def build_run_claim(
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    gpu_index: int,
    gpu_uuid: str,
    runtime_cuda_uuid: str,
    cuda_visible_devices: str,
    worker_pid: int,
    started_at: str,
) -> dict[str, Any]:
    validate_run_request(request, policy)
    if gpu_index not in {0, 1, 2, 3}:
        raise CanonicalScreeningError("screening GPU must be one of 0..3")
    if type(worker_pid) is not int or worker_pid <= 0:
        raise CanonicalScreeningError("worker PID must be positive")
    claim = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": RUN_CLAIM_CONTRACT,
        "run_request_sha256": request["run_request_sha256"],
        "admission_sha256": request["admission"]["canonical_sha256"],
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
        or value["physical_gpu_index"]
        not in policy["resources"]["physical_gpus"]
        or type(value["worker_pid"]) is not int
        or value["worker_pid"] <= 0
    ):
        raise CanonicalScreeningError("run claim binding mismatch")
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
        )
