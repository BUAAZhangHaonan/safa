from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml

from safa.utils.hashing import sha256_file


SCHEMA_VERSION = 1
CONTRACT_TYPE = "safa_experiment_closeout_snapshot_v1"
STATUS_VALUES = (
    "config_only_never_started",
    "started_incomplete",
    "invalid_evaluation",
    "completed_gate_fail",
    "pending_visual_finalize",
    "formal_closed",
)
EVIDENCE_LEVEL_VALUES = (
    "config_only",
    "started_unverified",
    "legacy_untrusted_candidate_discovery",
    "historical_observation",
    "strong_provenance_historical_baseline",
    "formal_gate",
    "formal_closed",
)
EXPERIMENT_CONFIG = re.compile(r"^(?P<series>[er])(?P<number>[1-9]|1\d|2[0-3])(?:_|$)")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".csv",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".onnx",
        ".out",
        ".py",
        ".pt",
        ".pth",
        ".safetensors",
        ".yaml",
        ".yml",
    }
)
CHECKPOINT_SUFFIXES = frozenset({".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"})
RESULT_NAME_TOKENS = (
    "automatic_evidence",
    "evaluation",
    "gate",
    "inventory",
    "manifest",
    "metric",
    "per_sample",
    "phase_result",
    "report",
    "result",
    "selection",
    "supersession",
)
FULL_TERMINAL_CONTRACT_TYPE = "safa_r9_full_continuation_final_result_v1"
FULL_TERMINAL_REQUIRED_SHA_FIELDS = (
    "checkpoint_sha256",
    "evaluator_bundle_sha256",
    "full_continuation_sha256",
    "heldout_seal_sha256",
    "manifest_sha256",
    "selection_sha256",
)
FAIL_VERDICTS = frozenset(
    {"failed", "gate_fail", "no_candidate_passed", "no_winner", "rejected"}
)


class CloseoutError(RuntimeError):
    """Raised when closeout evidence cannot be represented without ambiguity."""


@dataclass(frozen=True)
class EvidenceFile:
    path: str
    sha256: str
    size_bytes: int
    kind: str


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise CloseoutError(f"Evidence path escapes repository root: {path}") from exc


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                value = json.load(handle)
            else:
                value = yaml.safe_load(handle)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CloseoutError(f"Cannot parse required mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CloseoutError(f"Required mapping is not an object: {path}")
    return value


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CloseoutError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.rstrip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise CloseoutError(
            f"git {' '.join(args)} failed: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _discover_configs(repo_root: Path) -> list[Path]:
    root = repo_root / "configs" / "medium_v2" / "experiments"
    if not root.is_dir():
        raise CloseoutError(f"Experiment config directory is missing: {root}")
    paths = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".json", ".yaml", ".yml"}
        and EXPERIMENT_CONFIG.match(path.stem)
    ]
    if not paths:
        raise CloseoutError(f"No R1-R9 or E1-E23 experiment configs found under {root}")
    return sorted(paths, key=lambda item: item.name)


def _walk_scalars(value: Any, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            child = str(child_key)
            yield from _walk_scalars(child_value, child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child_value in value:
            yield from _walk_scalars(child_value, key)
    else:
        yield key, value


def _declared_paths(config: Mapping[str, Any], repo_root: Path) -> list[Path]:
    results: set[Path] = set()
    path_keys = (
        "checkpoint",
        "features",
        "index",
        "manifest",
        "out_dir",
        "output_dir",
        "path",
        "pretrained",
        "resume_from",
        "weights",
    )
    for key, value in _walk_scalars(config):
        if not isinstance(value, str) or not value.strip():
            continue
        if not any(token in key.lower() for token in path_keys):
            continue
        if "://" in value or ";" in value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        try:
            candidate.resolve().relative_to(repo_root.resolve())
        except ValueError:
            continue
        results.add(candidate)
    return sorted(results, key=lambda item: item.as_posix())


def _evaluator_sources(repo_root: Path) -> list[Path]:
    paths = list((repo_root / "src" / "safa" / "evaluation").glob("*.py"))
    for pattern in (
        "eval*.py",
        "identity_privacy_eval.py",
        "probe_r9_arcface_execution.py",
        "run_r9*.py",
    ):
        paths.extend((repo_root / "scripts").glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def _artifact_files(repo_root: Path, configs: Sequence[Path]) -> list[Path]:
    files: set[Path] = set(configs)
    files.update(_evaluator_sources(repo_root))
    for relative_root in ("data/index", "docs/results", "reports"):
        root = repo_root / relative_root
        if not root.exists():
            continue
        if not root.is_dir():
            raise CloseoutError(f"Evidence root is not a directory: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES:
                files.add(path)
    files.update(
        path
        for path in repo_root.glob("safa*.md")
        if path.is_file()
    )
    config_aliases: list[tuple[tuple[str, ...], str]] = []
    for config_path in configs:
        config = _load_mapping(config_path)
        match = EXPERIMENT_CONFIG.match(config_path.stem)
        if match is None:
            raise CloseoutError(f"Internal error: non-experiment config {config_path}")
        logical_id = f"{match.group('series').upper()}{int(match.group('number'))}"
        config_aliases.append((_aliases(config_path, config), logical_id))
        for declared in _declared_paths(config, repo_root):
            if declared.is_file() and declared.suffix.lower() in ARTIFACT_SUFFIXES:
                files.add(declared)
            elif declared.is_dir():
                for child in declared.rglob("*"):
                    if child.is_file() and child.suffix.lower() in ARTIFACT_SUFFIXES:
                        files.add(child)
    for relative_root in ("artifacts", "logs"):
        root = repo_root / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ARTIFACT_SUFFIXES:
                continue
            relative = _relative(path, repo_root)
            if any(
                _is_bound(relative, aliases, logical_id)
                for aliases, logical_id in config_aliases
            ):
                files.add(path)
    return sorted(files, key=lambda item: _relative(item, repo_root))


def _kind(path: Path) -> str:
    lower_parts = tuple(part.lower() for part in path.parts)
    name = path.name.lower()
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        return "checkpoint"
    if path.suffix.lower() == ".py" and (
        "evaluation" in lower_parts or "scripts" in lower_parts
    ):
        return "evaluator_source"
    if "visual_reviews" in lower_parts or "review" in name:
        return "review"
    if "gate" in name:
        return "gate"
    if "logs" in lower_parts or path.suffix.lower() in {".log", ".out"}:
        return "log"
    if path.parts and "configs" in lower_parts:
        return "config"
    if any(token in name for token in RESULT_NAME_TOKENS):
        return "result"
    if "data" in lower_parts and "index" in lower_parts:
        return "data_index"
    return "supporting_evidence"


def _hash_files(paths: Sequence[Path], repo_root: Path) -> list[EvidenceFile]:
    evidence: list[EvidenceFile] = []
    for path in paths:
        digest = sha256_file(path)
        if not SHA256.fullmatch(digest):
            raise CloseoutError(f"Invalid SHA256 returned for {path}: {digest!r}")
        evidence.append(
            EvidenceFile(
                path=_relative(path, repo_root),
                sha256=digest,
                size_bytes=path.stat().st_size,
                kind=_kind(path),
            )
        )
    return evidence


def _primary_aliases(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    values = {config_path.stem}
    for key in ("experiment_name", "campaign_id", "child_campaign_id"):
        value = config.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    return tuple(sorted(values, key=lambda item: (-len(item), item)))


def _aliases(config_path: Path, config: Mapping[str, Any]) -> tuple[str, ...]:
    values = set(_primary_aliases(config_path, config))
    source = config.get("source")
    if isinstance(source, Mapping):
        value = source.get("campaign_id")
        if isinstance(value, str) and value:
            values.add(value)
    return tuple(sorted(values, key=lambda item: (-len(item), item)))


def _is_bound(path: str, aliases: Sequence[str], logical_id: str) -> bool:
    parts = tuple(Path(path).parts)
    for alias in aliases:
        if alias in parts or any(
            part.startswith(f"{alias}_") or part.endswith(f"_{alias}")
            for part in parts
        ):
            return True
    return False


def _selector(config: Mapping[str, Any], bound: Sequence[EvidenceFile]) -> str | None:
    value = config.get("best_model")
    if isinstance(value, str) and value in {"raw", "ema"}:
        return value
    ema = config.get("ema")
    if isinstance(ema, Mapping):
        value = ema.get("evaluate_ema")
        if value is True and ema.get("evaluate_raw") is not True:
            return "ema"
    names = [Path(item.path).name.lower() for item in bound if item.kind == "checkpoint"]
    if any("ema" in name for name in names):
        return "mixed_or_ema_available"
    if names:
        return "unspecified"
    return None


def _data_protocol(
    config: Mapping[str, Any],
    evidence_by_path: Mapping[str, EvidenceFile],
    repo_root: Path,
) -> dict[str, Any]:
    declared: list[dict[str, Any]] = []
    seeds: dict[str, int] = {}
    counts: set[int] = set()
    for key, value in _walk_scalars(config):
        lower = key.lower()
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if lower in {"seed", "sampling_seed", "base_seed", "bootstrap_seed"}:
                seeds[lower] = value
            if (
                "sample_count" in lower
                or "max_samples" in lower
                or lower in {"sample_count", "niqe_max_samples", "distribution_max_samples"}
            ):
                counts.add(value)
        if not isinstance(value, str) or not value:
            continue
        if not ("index" in lower or "manifest" in lower or "features" in lower):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        try:
            relative = _relative(path, repo_root)
        except CloseoutError:
            continue
        item: dict[str, Any] = {"field": key, "path": relative, "exists": path.exists()}
        evidence = evidence_by_path.get(relative)
        if evidence is not None:
            item["sha256"] = evidence.sha256
        declared.append(item)
    return {
        "declared_assets": sorted(declared, key=lambda item: (item["field"], item["path"])),
        "seeds": dict(sorted(seeds.items())),
        "declared_sample_counts": sorted(counts),
    }


def _policy_protocol_key(logical_id: str) -> str:
    number = int(logical_id[1:])
    if logical_id.startswith("R"):
        if number <= 5:
            return "R1-R5"
        if number <= 7:
            return "R6-R7"
        return f"R{number}"
    return "E1-E15" if number <= 15 else "E16-E23"


def _protocol_family(
    logical_id: str, protocol_families: Mapping[str, Any]
) -> str:
    key = _policy_protocol_key(logical_id)
    value = protocol_families.get(key)
    if not isinstance(value, str) or not value:
        raise CloseoutError(
            f"Historical policy has no protocol family for {logical_id} via {key}"
        )
    return value


def _comparability_group(config: Mapping[str, Any], protocol_family: str) -> str:
    generator = config.get("generator")
    model_type = None
    if isinstance(generator, Mapping):
        model_type = generator.get("model_type")
    if not isinstance(model_type, str):
        model_type = config.get("model_type")
    train_index = config.get("train_index")
    if not isinstance(train_index, str):
        train_index = "unspecified_data"
    selector = config.get("best_model", "unspecified_selector")
    payload = {
        "protocol_family": protocol_family,
        "model_type": model_type or "unspecified_model",
        "train_index": train_index,
        "selector": selector,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()[:12]
    return f"{protocol_family}:{digest}"


def _json_objects(bound: Sequence[EvidenceFile], repo_root: Path) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for item in bound:
        if Path(item.path).suffix.lower() != ".json":
            continue
        path = repo_root / item.path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CloseoutError(f"Cannot parse bound JSON evidence {path}: {exc}") from exc
        if isinstance(value, Mapping):
            yield item.path, value


def _self_digest_matches(value: Mapping[str, Any], field: str) -> bool:
    recorded = value.get(field)
    if not isinstance(recorded, str) or not SHA256.fullmatch(recorded):
        return False
    payload = dict(value)
    del payload[field]
    return hashlib.sha256(_canonical_json(payload)).hexdigest() == recorded


def _full_terminal_contract(value: Mapping[str, Any]) -> bool:
    if (
        value.get("contract_type") != FULL_TERMINAL_CONTRACT_TYPE
        or value.get("phase") != "full"
        or value.get("status") != "formal_closed"
        or value.get("verdict") != "passed"
        or not _self_digest_matches(value, "final_result_sha256")
    ):
        return False
    return all(
        isinstance(value.get(field), str)
        and SHA256.fullmatch(str(value[field])) is not None
        for field in FULL_TERMINAL_REQUIRED_SHA_FIELDS
    )


def _pending_is_superseded(
    pending: Mapping[str, Any],
    objects: Sequence[tuple[str, Mapping[str, Any]]],
) -> bool:
    chain = pending.get("supersession_contract_sha256")
    if not isinstance(chain, str) or not SHA256.fullmatch(chain):
        return False
    gates = [
        value
        for _, value in objects
        if value.get("contract_type") == "safa_r9_confirm512_report_only_gate_v3"
        and value.get("supersession_contract_sha256") == chain
        and value.get("verdict") == "winner_locked_report_only"
    ]
    selections = [
        value
        for _, value in objects
        if value.get("contract_type")
        == "safa_r9_confirm512_report_only_selection_v3"
        and value.get("supersession_contract_sha256") == chain
        and value.get("next_stage") == "new_v9_full_continuation_required"
        and value.get("reselection_allowed") is False
    ]
    if len(gates) != 1 or len(selections) != 1:
        return False
    gate = gates[0]
    selection = selections[0]
    gate_sha = gate.get("gate_contract_sha256")
    selection_sha = selection.get("selection_sha256")
    if (
        not isinstance(gate_sha, str)
        or not SHA256.fullmatch(gate_sha)
        or not isinstance(selection_sha, str)
        or not SHA256.fullmatch(selection_sha)
        or selection.get("gate_contract_sha256") != gate_sha
    ):
        return False
    results = [
        value
        for _, value in objects
        if value.get("supersession_contract_sha256") == chain
        and value.get("gate_contract_sha256") == gate_sha
        and value.get("selection_sha256") == selection_sha
        and value.get("verdict") == "winner_locked_report_only"
        and value.get("generation_execution_count") == 0
        and value.get("evaluator_execution_count") == 0
    ]
    return len(results) == 1


def _status_and_level(
    logical_id: str,
    bound: Sequence[EvidenceFile],
    repo_root: Path,
    *,
    legacy_ids: frozenset[str],
    trusted_ids: frozenset[str],
) -> tuple[str, str, str]:
    execution = [item for item in bound if item.kind != "config"]
    if not execution:
        return "config_only_never_started", "config_only", "No execution evidence exists."
    if logical_id in legacy_ids:
        return (
            "invalid_evaluation",
            "legacy_untrusted_candidate_discovery",
            "Historical R1-R5 evaluation is invalid for formal comparison under the approved closeout policy.",
        )
    objects = list(_json_objects(bound, repo_root))
    pending_objects: list[Mapping[str, Any]] = []
    gate_fail = False
    formal_closed = any(_full_terminal_contract(value) for _, value in objects)
    for path, value in objects:
        name = Path(path).name.lower()
        state = value.get("status")
        if name == "awaiting_visual_review.json" or state == "awaiting_visual_review":
            pending_objects.append(value)
        if "gate" in name and (
            value.get("passed") is False or value.get("verdict") in FAIL_VERDICTS
        ):
            gate_fail = True
    if formal_closed:
        return "formal_closed", "formal_closed", "A bound full-phase formal closure contract passed."
    unresolved_pending = [
        pending
        for pending in pending_objects
        if not _pending_is_superseded(pending, objects)
    ]
    if unresolved_pending:
        return (
            "pending_visual_finalize",
            "strong_provenance_historical_baseline"
            if logical_id in trusted_ids
            else "formal_gate",
            "A bound awaiting_visual_review contract has not been superseded by full formal closure.",
        )
    if gate_fail:
        return "completed_gate_fail", "formal_gate", "A bound gate contract records passed=false."
    if logical_id in trusted_ids:
        return (
            "started_incomplete",
            "strong_provenance_historical_baseline",
            "Execution evidence exists, but no bound full formal closure exists.",
        )
    return (
        "started_incomplete",
        "historical_observation" if execution else "started_unverified",
        "Execution evidence exists, but no bound terminal formal contract exists.",
    )


def _resource_cost(config: Mapping[str, Any], bound: Sequence[EvidenceFile]) -> dict[str, Any]:
    fields = {}
    wanted = {
        "device",
        "global_batch_size",
        "num_workers",
        "per_device_batch_size",
        "world_size",
    }
    for key, value in _walk_scalars(config):
        if key in wanted and isinstance(value, (str, int, float)):
            fields[key] = value
    return {
        "declared": dict(sorted(fields.items())),
        "measured_evidence_paths": [
            item.path
            for item in bound
            if any(token in Path(item.path).name.lower() for token in ("benchmark", "resource", "timing"))
        ],
    }


def _build_rows(
    repo_root: Path,
    configs: Sequence[Path],
    evidence: Sequence[EvidenceFile],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    legacy = policy["legacy_invalid_series"]
    trusted = policy["trusted_historical_baselines"]
    protocol_families = policy["protocol_families"]
    legacy_ids = frozenset(legacy["logical_experiment_ids"])
    trusted_ids = frozenset(trusted["logical_experiment_ids"])
    evidence_by_path = {item.path: item for item in evidence}
    rows: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for config_path in configs:
        config = _load_mapping(config_path)
        match = EXPERIMENT_CONFIG.match(config_path.stem)
        if match is None:
            raise CloseoutError(f"Internal error: non-experiment config {config_path}")
        logical_id = f"{match.group('series').upper()}{int(match.group('number'))}"
        aliases = _aliases(config_path, config)
        config_relative = _relative(config_path, repo_root)
        bound = [
            item
            for item in evidence
            if item.path == config_relative or _is_bound(item.path, aliases, logical_id)
        ]
        bound.sort(key=lambda item: item.path)
        primary_aliases = _primary_aliases(config_path, config)
        status_bound = [
            item
            for item in bound
            if item.path == config_relative
            or _is_bound(item.path, primary_aliases, logical_id)
        ]
        run_id = config_path.stem
        if run_id in run_ids:
            raise CloseoutError(f"Duplicate experiment run ID: {run_id}")
        run_ids.add(run_id)
        status, level, termination_reason = _status_and_level(
            logical_id,
            status_bound,
            repo_root,
            legacy_ids=legacy_ids,
            trusted_ids=trusted_ids,
        )
        family = _protocol_family(logical_id, protocol_families)
        checkpoints = [item for item in bound if item.kind == "checkpoint"]
        results = [item for item in bound if item.kind == "result"]
        logs = [item for item in bound if item.kind == "log"]
        gates = [item for item in bound if item.kind == "gate"]
        reviews = [item for item in bound if item.kind == "review"]
        config_evidence = evidence_by_path[config_relative]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "logical_experiment_id": logical_id,
                "run_id": run_id,
                "series": match.group("series").upper(),
                "number": int(match.group("number")),
                "status": status,
                "evidence_level": level,
                "protocol_family": family,
                "comparability_group": _comparability_group(config, family),
                "config": {
                    "path": config_relative,
                    "sha256": config_evidence.sha256,
                    "experiment_name": config.get("experiment_name", run_id),
                },
                "checkpoint": {
                    "selector": _selector(config, bound),
                    "files": [item.__dict__ for item in checkpoints],
                    "distinct_sha256": sorted({item.sha256 for item in checkpoints}),
                },
                "data_protocol": _data_protocol(config, evidence_by_path, repo_root),
                "seed": config.get("seed"),
                "sample_count": config.get("sample_count"),
                "metrics": {"evidence_paths": [item.path for item in results]},
                "resource_cost": _resource_cost(config, bound),
                "termination_reason": termination_reason,
                "evidence": {
                    "aliases": list(aliases),
                    "result_paths": [item.path for item in results],
                    "log_paths": [item.path for item in logs],
                    "gate_paths": [item.path for item in gates],
                    "review_paths": [item.path for item in reviews],
                },
            }
        )
    expected = {f"E{number}" for number in range(1, 24)} | {
        f"R{number}" for number in range(1, 10)
    }
    observed = {row["logical_experiment_id"] for row in rows}
    if observed != expected:
        raise CloseoutError(
            "Experiment config coverage is incomplete: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    for row in rows:
        if row["status"] not in STATUS_VALUES:
            raise CloseoutError(f"Unclassified status in {row['run_id']}: {row['status']}")
        if row["evidence_level"] not in EVIDENCE_LEVEL_VALUES:
            raise CloseoutError(
                f"Unclassified evidence level in {row['run_id']}: {row['evidence_level']}"
            )
    return sorted(rows, key=lambda row: (row["series"], row["number"], row["run_id"]))


def _validated_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if (
        policy.get("schema_version") != 1
        or policy.get("contract_type") != "safa_historical_evidence_policy_v1"
    ):
        raise CloseoutError("Historical evidence policy schema/contract is invalid")
    legacy = policy.get("legacy_invalid_series")
    trusted = policy.get("trusted_historical_baselines")
    families = policy.get("protocol_families")
    if not isinstance(legacy, Mapping) or not isinstance(trusted, Mapping):
        raise CloseoutError("Historical evidence policy series sections are missing")
    if legacy.get("logical_experiment_ids") != ["R1", "R2", "R3", "R4", "R5"]:
        raise CloseoutError("Historical evidence policy must invalidate exactly R1-R5")
    if (
        legacy.get("evidence_level") != "legacy_untrusted_candidate_discovery"
        or legacy.get("status_when_execution_evidence_exists") != "invalid_evaluation"
    ):
        raise CloseoutError("Historical R1-R5 policy status/evidence fields are invalid")
    if trusted.get("logical_experiment_ids") != ["R6", "R7", "R9"]:
        raise CloseoutError("Historical evidence policy trusted set must be R6/R7/R9")
    if trusted.get("evidence_level") != "strong_provenance_historical_baseline":
        raise CloseoutError("Historical trusted baseline evidence level is invalid")
    expected_families = {
        "R1-R5",
        "R6-R7",
        "R8",
        "R9",
        "E1-E15",
        "E16-E23",
    }
    if not isinstance(families, Mapping) or set(families) != expected_families:
        raise CloseoutError(
            "Historical protocol families must cover exactly "
            f"{sorted(expected_families)}"
        )
    if not all(isinstance(value, str) and value for value in families.values()):
        raise CloseoutError("Historical protocol family values must be non-empty strings")
    return dict(policy)


def _packages() -> tuple[list[str], str]:
    rows = sorted(
        f"{dist.metadata.get('Name', '<unnamed>')}=={dist.version}"
        for dist in importlib.metadata.distributions()
    )
    digest = hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
    return rows, digest


def _payload_inventory_bindings(
    evidence: Sequence[EvidenceFile], repo_root: Path
) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for item in evidence:
        if Path(item.path).suffix.lower() != ".json":
            continue
        try:
            value = json.loads((repo_root / item.path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CloseoutError(
                f"Cannot parse evidence JSON while binding payload inventory "
                f"{item.path}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            continue
        inventory_sha = value.get("generation_inventory_sha256")
        if isinstance(inventory_sha, str):
            if not SHA256.fullmatch(inventory_sha):
                raise CloseoutError(
                    f"Invalid generation inventory SHA in {item.path}: {inventory_sha!r}"
                )
            bindings.append(
                {
                    "evidence_path": item.path,
                    "evidence_file_sha256": item.sha256,
                    "generation_inventory_sha256": inventory_sha,
                }
            )
    return sorted(
        bindings,
        key=lambda row: (
            row["generation_inventory_sha256"],
            row["evidence_path"],
        ),
    )


def _missing_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for row in rows:
        missing: list[str] = []
        checkpoint = row["checkpoint"]
        evidence = row["evidence"]
        if not checkpoint["files"]:
            missing.append("checkpoint")
        if not evidence["result_paths"]:
            missing.append("result")
        if not evidence["log_paths"]:
            missing.append("log")
        if not evidence["gate_paths"]:
            missing.append("formal_gate")
        if not evidence["review_paths"]:
            missing.append("visual_review")
        if missing:
            findings.append(
                {
                    "logical_experiment_id": row["logical_experiment_id"],
                    "run_id": row["run_id"],
                    "status": row["status"],
                    "missing": missing,
                }
            )
    counts = Counter(item for finding in findings for item in finding["missing"])
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_type": "safa_missing_evidence_v1",
        "finding_count": len(findings),
        "counts_by_missing_kind": dict(sorted(counts.items())),
        "findings": findings,
    }


def _conflicts(policy: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    legacy = policy.get("legacy_invalid_series")
    if not isinstance(legacy, Mapping):
        raise CloseoutError("Historical evidence policy lacks legacy_invalid_series")
    sources = legacy.get("sources")
    if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
        raise CloseoutError("Historical legacy conflict sources must be a string list")
    missing_sources = [item for item in sources if not (repo_root / item).is_file()]
    if missing_sources:
        raise CloseoutError(f"Historical conflict sources are missing: {missing_sources}")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_type": "safa_documentation_conflicts_v1",
        "conflicts": [
            {
                "conflict_id": "r1_r5_legacy_loader_and_narrative_json_conflict",
                "kind": "protocol_invalidation",
                "logical_experiment_ids": legacy["logical_experiment_ids"],
                "resolution": "candidate_discovery_only",
                "reason": legacy["reason"],
                "sources": sources,
                "source_of_resolution": "configs/closeout/historical_evidence_policy.json",
            }
        ],
    }


def build_closeout_snapshot(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise CloseoutError(f"Repository root is not a Git worktree: {root}")
    policy_path = root / "configs" / "closeout" / "historical_evidence_policy.json"
    policy = _validated_policy(_load_mapping(policy_path))
    configs = _discover_configs(root)
    artifact_paths = _artifact_files(root, configs)
    evidence = _hash_files(artifact_paths, root)
    rows = _build_rows(root, configs, evidence, policy)
    packages, packages_sha256 = _packages()
    dirty_lines = _git(root, "status", "--porcelain").splitlines()
    tracked = frozenset(_git(root, "ls-files").splitlines())
    dirty_paths = frozenset(
        line[3:]
        for line in dirty_lines
        if len(line) >= 4 and " -> " not in line
    )
    evaluator_sources = [
        item for item in evidence if item.kind == "evaluator_source"
    ]
    evaluator_rows = [
        {
            **item.__dict__,
            "git_tracked": item.path in tracked,
            "dirty": item.path in dirty_paths or item.path not in tracked,
        }
        for item in evaluator_sources
    ]
    evaluator_bundle_sha256 = hashlib.sha256(
        _canonical_json(evaluator_rows)
    ).hexdigest()
    evaluator_diff = _git_bytes(
        root,
        "diff",
        "--binary",
        "--",
        "src/safa/evaluation",
        "scripts",
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": "safa_provenance_snapshot_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head_sha": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(dirty_lines),
            "status_porcelain": dirty_lines,
            "evaluator_diff_sha256": hashlib.sha256(evaluator_diff).hexdigest(),
            "evaluator_diff_size_bytes": len(evaluator_diff),
        },
        "environment": {
            "executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages_sha256": packages_sha256,
            "packages": packages,
        },
        "policy": {
            "path": _relative(policy_path, root),
            "sha256": sha256_file(policy_path),
        },
        "evaluator_bundle": {
            "sha256": evaluator_bundle_sha256,
            "source_count": len(evaluator_rows),
            "sources": evaluator_rows,
        },
        "generated_payload_bindings": _payload_inventory_bindings(evidence, root),
    }
    registry_entries = [
        {
            "run_id": row["run_id"],
            "logical_experiment_id": row["logical_experiment_id"],
            "protocol_family": row["protocol_family"],
            "comparability_group": row["comparability_group"],
            "data_protocol": row["data_protocol"],
            "checkpoint_selector": row["checkpoint"]["selector"],
        }
        for row in rows
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_type": CONTRACT_TYPE,
        "rows": rows,
        "artifact_manifest": [item.__dict__ for item in evidence],
        "protocol_registry": {
            "schema_version": SCHEMA_VERSION,
            "contract_type": "safa_protocol_registry_v1",
            "allowed_statuses": list(STATUS_VALUES),
            "allowed_evidence_levels": list(EVIDENCE_LEVEL_VALUES),
            "entries": registry_entries,
        },
        "missing_evidence": _missing_evidence(rows),
        "documentation_conflicts": _conflicts(policy, root),
        "provenance": provenance,
    }


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_closeout_snapshot(snapshot: Mapping[str, Any], output_dir: str | Path) -> Path:
    target = Path(output_dir).resolve()
    if target.exists():
        raise CloseoutError(f"Refusing to overwrite existing closeout output: {target}")
    if not target.parent.is_dir():
        raise CloseoutError(f"Closeout output parent does not exist: {target.parent}")
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise CloseoutError(f"Refusing to reuse stale temporary output: {temporary}")
    temporary.mkdir()
    try:
        rows = snapshot["rows"]
        _write_new(
            temporary / "experiment_ledger.jsonl",
            b"".join(_canonical_json(row) for row in rows),
        )
        csv_fields = (
            "logical_experiment_id",
            "run_id",
            "series",
            "number",
            "status",
            "evidence_level",
            "protocol_family",
            "comparability_group",
            "config",
            "checkpoint",
            "data_protocol",
            "seed",
            "sample_count",
            "metrics",
            "resource_cost",
            "termination_reason",
            "evidence",
        )
        csv_path = temporary / "experiment_ledger.csv"
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field)) for field in csv_fields})
            handle.flush()
            os.fsync(handle.fileno())
        for filename, key in (
            ("protocol_registry.json", "protocol_registry"),
            ("missing_evidence.json", "missing_evidence"),
            ("documentation_conflicts.json", "documentation_conflicts"),
            ("provenance_snapshot.json", "provenance"),
        ):
            _write_new(temporary / filename, _canonical_json(snapshot[key]))
        _write_new(
            temporary / "artifact_sha_manifest.jsonl",
            b"".join(_canonical_json(row) for row in snapshot["artifact_manifest"]),
        )
        output_files = sorted(
            path for path in temporary.iterdir() if path.is_file()
        )
        binding_rows = [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
        ]
        binding = {
            "schema_version": SCHEMA_VERSION,
            "contract_type": "safa_closeout_output_binding_v1",
            "files": binding_rows,
        }
        _write_new(temporary / "closeout_binding.json", _canonical_json(binding))
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
