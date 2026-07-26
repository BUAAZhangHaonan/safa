"""Fail-closed contracts for the historical canonical 512 screening campaign."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


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


def validate_supersession_evidence(
    repo_root: Path, raw_supersedes: Mapping[str, Any]
) -> dict[str, Any]:
    supersedes = _require_mapping(raw_supersedes, "supersession evidence")
    _require_exact_keys(
        supersedes,
        {
            "policy_sha256",
            "classification",
            "result_count",
            "pending_count",
            "controller_terminal",
            "wrapper_exit",
            "controller_log",
            "controller_process_log",
        },
        "supersession evidence",
    )
    if (
        supersedes["policy_sha256"]
        != "8ce4855b042161ff5698ce400f8b80122add90d6025ffd08f31fe49d8ef84a7f"
        or supersedes["classification"] != "started_incomplete"
        or supersedes["result_count"] != 1
        or supersedes["pending_count"] != 192
    ):
        raise CanonicalScreeningError("canonical supersession status differs")
    root = repo_root.resolve()
    bound = {
        "policy_sha256": supersedes["policy_sha256"],
        "classification": "started_incomplete",
        "result_count": 1,
        "pending_count": 192,
        **{
            name: _validate_bound_file(
                root, supersedes[name], f"supersession {name}"
            )
            for name in (
                "controller_terminal",
                "wrapper_exit",
                "controller_log",
                "controller_process_log",
            )
        },
    }
    terminal = load_json(
        Path(bound["controller_terminal"]["path"]),
        "superseded controller terminal",
    )
    if (
        terminal.get("contract_type")
        != "safa_canonical_preflight_controller_terminal_v1"
        or terminal.get("policy_sha256") != bound["policy_sha256"]
        or terminal.get("status") != "failed"
        or terminal.get("result_count") != 1
        or terminal.get("pending_count") != 192
    ):
        raise CanonicalScreeningError(
            "superseded controller terminal semantics differ"
        )
    wrapper_exit = load_json(
        Path(bound["wrapper_exit"]["path"]),
        "superseded wrapper exit",
    )
    if (
        wrapper_exit.get("contract_type")
        != "safa_canonical_preflight_wrapper_exit_v2"
        or wrapper_exit.get("policy_sha256") != bound["policy_sha256"]
        or wrapper_exit.get("exit_code") != 2
        or wrapper_exit.get("signal") is not None
        or wrapper_exit.get("controller_terminal") != bound["controller_terminal"]
        or wrapper_exit.get("controller_process_log")
        != bound["controller_process_log"]
    ):
        raise CanonicalScreeningError("superseded wrapper exit semantics differ")
    return bound


def validate_policy(repo_root: Path, policy_path: Path) -> dict[str, Any]:
    root = repo_root.resolve()
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
    _require_exact_keys(
        resources,
        {
            "physical_gpus",
            "workers_per_gpu",
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
        },
        "screening resources",
    )
    if (
        resources["physical_gpus"] != [0, 1, 2, 3]
        or resources["workers_per_gpu"] != 2
        or resources["retry_count"] != 0
        or resources["require_tmux"] is not True
        or resources["cpu_admission_percent"] != 85
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
            "preflight_wrapper",
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
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
        },
        "source": source,
        "protocol": bound_protocol,
        "resources": resources,
        "arcface": arcface,
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
        "candidate_manifest": {
            "path": str(candidate_manifest_path.resolve()),
            "sha256": sha256_file(candidate_manifest_path),
            "canonical_sha256": candidate_manifest["candidate_manifest_sha256"],
        },
        "candidate": dict(candidate),
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
        "candidate_manifest",
        "candidate",
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
        or value["pixel_image_size"] != policy["protocol"]["pixel_image_size"]
        or value["pixel_protocol_config"]
        != policy["protocol"]["pixel_protocol_config"]
        or value["kid_subset_size"]
        != policy["protocol"]["kid_subset_sizes"][value["mode"]]
    ):
        raise CanonicalScreeningError("run request frozen fields differ")
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
        "gpu_index": gpu_index,
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
            "gpu_index",
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
        or value["gpu_index"] not in policy["resources"]["physical_gpus"]
        or type(value["worker_pid"]) is not int
        or value["worker_pid"] <= 0
    ):
        raise CanonicalScreeningError("run claim binding mismatch")
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
