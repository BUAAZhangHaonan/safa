from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Iterator, Mapping, Sequence

import yaml

from safa.evaluation.meanflow_guidance_runner import (
    resolve_frozen_effective_guidance_config,
)
from safa.evaluation.r9_determinism import (
    R9_ATTENTION_BACKEND,
    canonical_json_sha256,
    canonical_r9_arm_config_digest,
    validate_r9_execution_config,
)
from safa.evaluation.r9_semigroup_contracts import (
    R9_LOCKED_SCHEDULE_SCHEMA_VERSION,
    R9_SEMIGROUP_RECOVERY_AUTHORIZATION_ID,
    R9_SEMIGROUP_RECOVERY_POLICY,
    R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
    R9_SEMIGROUP_RECOVERY_POLICY_VERSION,
    R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
    R9_SELECTION_RULE,
    build_r9_semigroup_gate_contract,
    canonical_r9_schedule_contract_sha256,
    canonical_r9_semigroup_preflight_payload,
    merge_r9_semigroup_shards,
    validate_r9_locked_schedule_bindings,
    validate_r9_semigroup_gate_contract,
    validate_r9_semigroup_preflight_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_BASE = Path("artifacts/r9_meanflow_flow_map_guidance/campaigns")
CLOSURE_BASE = Path(
    "artifacts/r9_meanflow_flow_map_guidance/semigroup_campaign_closures"
)
SPLIT_KEYS = ("0.25", "0.5", "0.75")
SHARD_COUNT = 4
SAMPLE_COUNT = 64
_CAMPAIGN_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_BLINDED_CONDITION_ID = re.compile(r"condition_[0-9a-f]{12}")
_BLINDED_COLUMN_ID = re.compile(r"column_[0-9a-f]{12}")
_BLINDED_ROLES = ("source", "generated_direct", "split")
_REVIEW_ROWS_PER_PAGE = 8
_REVIEW_TILE_SIZE = 128
_REVIEW_HEADER_HEIGHT = 28
_ASSIGNMENT_DECISION_FIELDS = frozenset(
    {"decision", "passed", "severe_count", "severe_sample_ids"}
)
_SHARD_FILES = frozenset(
    {
        "completion.json",
        "generation_result.json",
        "per_sample.jsonl",
        "resume_contract.json",
        "run_manifest.json",
        "sample_id_manifest.jsonl",
        "semigroup.json",
        "session_history.jsonl",
        "verified_completion.json",
    }
)
_SHARD_DIRECTORIES = frozenset(
    {"generated_images", "native_images", "semigroup_split_images"}
)
_CLOSURE_ARTIFACTS = (
    "preflight_contract.json",
    "effective_config.json",
    "executed_config.json",
    "evidence_manifest.json",
    "visual_review_assignment.json",
    "visual_review_blinding_map.json",
    "visual_review.json",
    "semigroup_report.json",
    "gate_contract.json",
    "locked_schedule_manifest.json",
    "closure_seal.json",
)
_PUBLISHED_ARTIFACT_FILES = {
    name.removesuffix(".json"): name
    for name in _CLOSURE_ARTIFACTS
    if name != "closure_seal.json"
}


class CampaignSemigroupClosureError(ValueError):
    """Raised when a campaign-aware semigroup closure is not exact and immutable."""


def build_campaign_semigroup_evidence(
    *,
    config_path: Path,
    shard_root: Path,
    repo_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Rehash every contract and image used by the 64-sample visual review."""

    try:
        context = _load_context(config_path, shard_root, repo_root=repo_root)
        return _build_evidence(context)
    except CampaignSemigroupClosureError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CampaignSemigroupClosureError(
            f"semigroup evidence validation failed: {exc}"
        ) from exc


def prepare_campaign_semigroup_visual_review(
    *,
    config_path: Path,
    shard_root: Path,
    formal_campaign_id: str,
    repo_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Materialize one immutable blinded review assignment without decisions."""

    try:
        context = _load_context(config_path, shard_root, repo_root=repo_root)
        formal_id = _require_campaign_id(str(formal_campaign_id), "formal campaign ID")
        _validate_distinct_unmaterialized_formal_campaign(context, formal_id)
        return _prepare_visual_review_assignment(context, formal_id)
    except CampaignSemigroupClosureError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CampaignSemigroupClosureError(
            f"semigroup visual review preparation failed: {exc}"
        ) from exc


def finalize_campaign_semigroup_closure(
    *,
    config_path: Path,
    shard_root: Path,
    output_root: Path,
    visual_review_path: Path,
    repo_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Publish a new campaign-aware R9 preflight chain exactly once."""

    try:
        context = _load_context(config_path, shard_root, repo_root=repo_root)
        evidence = _build_evidence(context)
        review, formal_campaign_id, assignment = _validate_visual_review_without_map(
            visual_review_path,
            context=context,
            evidence=evidence,
        )
        output = _validate_output_root(
            output_root,
            repo_root=context.repo_root,
            bootstrap_campaign_id=context.bootstrap_campaign_id,
            formal_campaign_id=formal_campaign_id,
        )
        review_conditions = review["conditions"]
        condition_ids = [
            condition["condition_id"] for condition in assignment["conditions"]
        ]
        if all(
            review_conditions[condition_id]["passed"] is False
            for condition_id in condition_ids
        ):
            return _publish_terminal_failure(
                context=context,
                evidence=evidence,
                assignment=assignment,
                review=review,
                review_source=Path(visual_review_path),
                output_root=output,
                formal_campaign_id=formal_campaign_id,
                failure_reason="all_blinded_visual_conditions_failed",
            )

        with _working_directory(context.repo_root):
            numeric_precheck = merge_r9_semigroup_shards(
                context.config,
                context.shard_dirs,
                visual_pass_by_split={split: True for split in SPLIT_KEYS},
            )
        if numeric_precheck.get("gate_passed") is not True:
            return _publish_terminal_failure(
                context=context,
                evidence=evidence,
                assignment=assignment,
                review=review,
                review_source=Path(visual_review_path),
                output_root=output,
                formal_campaign_id=formal_campaign_id,
                failure_reason="no_quantitative_candidate",
                numeric_precheck=numeric_precheck,
            )

        blinding_map = _load_blinding_map_after_complete_review(
            context=context,
            evidence=evidence,
            assignment=assignment,
            formal_campaign_id=formal_campaign_id,
        )
        condition_to_split = _validate_blinding_map_contract(
            blinding_map,
            assignment=assignment,
            evidence_manifest_sha256=str(evidence["evidence_manifest_sha256"]),
            bootstrap_campaign_id=context.bootstrap_campaign_id,
            formal_campaign_id=formal_campaign_id,
            ordered_sample_id_sha256=_sample_id_digest(context.manifest_ids),
        )
        visual_pass = {
            condition_to_split[condition_id]: bool(
                review_conditions[condition_id]["passed"]
            )
            for condition_id in condition_ids
        }
        with _working_directory(context.repo_root):
            report = merge_r9_semigroup_shards(
                context.config,
                context.shard_dirs,
                visual_pass_by_split=visual_pass,
            )
        if report.get("gate_passed") is not True:
            raise CampaignSemigroupClosureError(
                "semigroup gate failed; a passing locked schedule cannot be sealed"
            )
        return _publish_closure(
            context=context,
            evidence=evidence,
            assignment=assignment,
            blinding_map=blinding_map,
            review=review,
            review_source=Path(visual_review_path),
            report=report,
            output_root=output,
            formal_campaign_id=formal_campaign_id,
        )
    except CampaignSemigroupClosureError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CampaignSemigroupClosureError(
            f"semigroup campaign closure failed: {exc}"
        ) from exc


def _validate_recovery_semigroup_metrics_are_finite(context: _Context) -> None:
    for shard_dir in context.shard_dirs:
        raw_semigroup = _read_json(
            shard_dir / "semigroup.json", "semigroup recovery evidence"
        )
        rows = raw_semigroup.get("rows")
        if not isinstance(rows, list):
            raise CampaignSemigroupClosureError(
                "semigroup recovery evidence rows are invalid"
            )
        for row in rows:
            splits = row.get("splits") if isinstance(row, Mapping) else None
            if not isinstance(splits, Mapping):
                raise CampaignSemigroupClosureError(
                    "semigroup recovery split evidence is invalid"
                )
            for split in SPLIT_KEYS:
                metrics = splits.get(split)
                if not isinstance(metrics, Mapping):
                    raise CampaignSemigroupClosureError(
                        "semigroup recovery split evidence is incomplete"
                    )
                for metric in (
                    "latent_residual",
                    "endpoint_e0_cosine",
                    "decoded_pixel_l1",
                    "decoded_psnr",
                ):
                    value = metrics.get(metric)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise CampaignSemigroupClosureError(
                            f"non-finite semigroup recovery metric: {metric}"
                        )


def finalize_campaign_semigroup_policy_recovery(
    *,
    config_path: Path,
    shard_root: Path,
    policy_campaign_id: str,
    formal_campaign_id: str,
    output_root: Path,
    visual_review_path: Path,
    source_terminal_failure_path: Path,
    user_recovery_authorization_id: str,
    repo_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Seal the user-authorized report-only recovery policy exactly once."""

    try:
        context = _load_context(config_path, shard_root, repo_root=repo_root)
        _validate_recovery_semigroup_metrics_are_finite(context)
        evidence = _build_evidence(context)
        review, source_formal_id, assignment = _validate_visual_review_without_map(
            visual_review_path,
            context=context,
            evidence=evidence,
        )
        policy_id = _require_campaign_id(str(policy_campaign_id), "policy campaign ID")
        formal_id = _require_campaign_id(str(formal_campaign_id), "formal campaign ID")
        if policy_id in {context.bootstrap_campaign_id, formal_id}:
            raise CampaignSemigroupClosureError(
                "policy, bootstrap, and formal campaign IDs must be distinct"
            )
        if user_recovery_authorization_id != R9_SEMIGROUP_RECOVERY_AUTHORIZATION_ID:
            raise CampaignSemigroupClosureError(
                "user recovery authorization ID is not registered"
            )
        output = _validate_output_root(
            output_root,
            repo_root=context.repo_root,
            bootstrap_campaign_id=policy_id,
            formal_campaign_id=formal_id,
        )
        source_failure = _existing_file(
            source_terminal_failure_path,
            "source terminal closure failure",
            repo_root=context.repo_root,
        )
        source_failure_payload = _validate_terminal_failure(
            source_failure.parent,
            formal_campaign_id=source_formal_id,
            repo_root=context.repo_root,
        )
        if (
            source_failure.name != "closure_failure.json"
            or source_failure_payload.get("bootstrap_campaign_id")
            != context.bootstrap_campaign_id
        ):
            raise CampaignSemigroupClosureError(
                "source terminal failure does not bind the bootstrap campaign"
            )

        map_binding = assignment["blinding_map"]
        map_path = _existing_file(
            map_binding["path"],
            "revealed visual review blinding map",
            repo_root=context.repo_root,
        )
        if stat.S_IMODE(map_path.stat().st_mode) != stat.S_IRUSR:
            raise CampaignSemigroupClosureError(
                "policy recovery requires the previously revealed read-only map"
            )
        if _sha256_file(map_path) != _require_sha(
            map_binding.get("file_sha256"),
            "visual review blinding map file SHA256",
        ):
            raise CampaignSemigroupClosureError(
                "visual review blinding map file SHA256 mismatch"
            )
        blinding_map = _read_json(map_path, "revealed visual review blinding map")
        condition_to_split = _validate_blinding_map_contract(
            blinding_map,
            assignment=assignment,
            evidence_manifest_sha256=str(evidence["evidence_manifest_sha256"]),
            bootstrap_campaign_id=context.bootstrap_campaign_id,
            formal_campaign_id=source_formal_id,
            ordered_sample_id_sha256=_sample_id_digest(context.manifest_ids),
        )
        visual_assessment: dict[str, dict[str, Any]] = {}
        for condition_id, split in condition_to_split.items():
            decision = review["conditions"][condition_id]
            severe_count = int(decision["severe_count"])
            visual_assessment[split] = {
                "condition_id": condition_id,
                "severe_count": severe_count,
                "severe_sample_ids": list(decision["severe_sample_ids"]),
                "passed": severe_count
                <= int(R9_SEMIGROUP_RECOVERY_POLICY["visual_severe_limit_per_split"]),
            }
        if set(visual_assessment) != set(SPLIT_KEYS) or any(
            row["passed"] is not True for row in visual_assessment.values()
        ):
            raise CampaignSemigroupClosureError(
                "visual severe limit exceeded under recovery policy"
            )
        with _working_directory(context.repo_root):
            report = merge_r9_semigroup_shards(
                context.config,
                context.shard_dirs,
                visual_pass_by_split={split: True for split in SPLIT_KEYS},
            )
        candidates = []
        for candidate in report["candidates"]:
            split = str(candidate["t_cut"])
            normalized = dict(candidate)
            normalized["numeric_threshold_pass"] = bool(candidate["passed"])
            normalized["visual_pass"] = visual_assessment[split]["passed"]
            normalized["passed"] = bool(normalized["visual_pass"])
            candidates.append(normalized)
        report = {
            **report,
            "schema_version": 2,
            "contract_type": "safa_r9_semigroup_recovery_report_v2",
            "gate_passed": True,
            "selected_t_cut": 0.25,
            "t_cut": 0.25,
            "candidates": candidates,
            "selection_rule": R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
            "numerical_metrics_role": "report_only",
            "visual_assessment": visual_assessment,
            "policy_version": R9_SEMIGROUP_RECOVERY_POLICY_VERSION,
            "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
        }
        return _publish_closure(
            context=context,
            evidence=evidence,
            assignment=assignment,
            blinding_map=blinding_map,
            review=review,
            review_source=Path(visual_review_path),
            report=report,
            output_root=output,
            formal_campaign_id=formal_id,
            recovery={
                "policy_campaign_id": policy_id,
                "source_formal_campaign_id": source_formal_id,
                "source_terminal_failure_path": source_failure,
                "authorization_id": user_recovery_authorization_id,
            },
        )
    except CampaignSemigroupClosureError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CampaignSemigroupClosureError(
            f"semigroup policy recovery failed: {exc}"
        ) from exc


def resolve_formal_campaign_semigroup_closure(
    formal_campaign_id: str,
    *,
    repo_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any] | None:
    """Resolve one valid bootstrap closure targeting a formal campaign.

    Absence means that the named campaign can only be used as a bootstrap
    preflight. A matching but invalid closure always fails and never falls back
    to the legacy global schedule.
    """

    try:
        formal_id = _require_campaign_id(str(formal_campaign_id), "formal campaign ID")
        root = Path(repo_root).resolve()
        base = (root / CLOSURE_BASE).resolve(strict=False)
        if not base.exists():
            return None
        if base.is_symlink() or not base.is_dir():
            raise CampaignSemigroupClosureError(
                "semigroup campaign closure base is not a real directory"
            )
        suffix = f"__for__{formal_id}"
        matches = [entry for entry in base.iterdir() if entry.name.endswith(suffix)]
        if not matches:
            return None
        if len(matches) != 1:
            raise CampaignSemigroupClosureError(
                "formal campaign has multiple semigroup campaign closures"
            )
        closure_root = matches[0]
        if closure_root.is_symlink() or not closure_root.is_dir():
            raise CampaignSemigroupClosureError(
                "formal campaign closure must be a real directory"
            )
        entries = {entry.name for entry in closure_root.iterdir()}
        if "closure_failure.json" in entries:
            if entries != {"closure_failure.json"}:
                raise CampaignSemigroupClosureError(
                    "terminal semigroup closure failure inventory is not exact"
                )
            failure = _validate_terminal_failure(
                closure_root,
                formal_campaign_id=formal_id,
                repo_root=root,
            )
            raise CampaignSemigroupClosureError(
                "formal campaign semigroup closure is a terminal failure: "
                f"{failure['failure_reason']}"
            )
        return _validate_published_closure(
            closure_root,
            formal_campaign_id=formal_id,
            repo_root=root,
        )
    except CampaignSemigroupClosureError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CampaignSemigroupClosureError(
            f"formal campaign closure validation failed: {exc}"
        ) from exc


class _Context:
    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        config: dict[str, Any],
        bootstrap_campaign_id: str,
        campaign_root: Path,
        campaign_runtime_path: Path,
        campaign_runtime: dict[str, Any],
        shard_root: Path,
        shard_dirs: tuple[Path, ...],
        manifest_path: Path,
        manifest_ids: tuple[str, ...],
        source_index_path: Path,
        source_by_id: dict[str, Path],
        checkpoint_path: Path,
        executed_config: dict[str, Any],
    ) -> None:
        self.repo_root = repo_root
        self.config_path = config_path
        self.config = config
        self.bootstrap_campaign_id = bootstrap_campaign_id
        self.campaign_root = campaign_root
        self.campaign_runtime_path = campaign_runtime_path
        self.campaign_runtime = campaign_runtime
        self.shard_root = shard_root
        self.shard_dirs = shard_dirs
        self.manifest_path = manifest_path
        self.manifest_ids = manifest_ids
        self.source_index_path = source_index_path
        self.source_by_id = source_by_id
        self.checkpoint_path = checkpoint_path
        self.executed_config = executed_config


def _load_context(config_path: Path, shard_root: Path, *, repo_root: Path) -> _Context:
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise CampaignSemigroupClosureError("repository root is not a directory")
    config_file = _existing_file(config_path, "runtime config", repo_root=root)
    relative_config = _relative_to(config_file, root, "runtime config")
    campaign_prefix = CAMPAIGN_BASE.parts
    parts = relative_config.parts
    if (
        len(parts) != len(campaign_prefix) + 4
        or parts[: len(campaign_prefix)] != campaign_prefix
        or parts[-3:]
        != (
            "runtime_configs",
            "preflight",
            "semigroup_preflight.yaml",
        )
    ):
        raise CampaignSemigroupClosureError(
            "--config must be the exact CID immutable preflight runtime config"
        )
    bootstrap_campaign_id = parts[len(campaign_prefix)]
    _require_campaign_id(bootstrap_campaign_id, "bootstrap campaign ID")
    campaign_root = (root / CAMPAIGN_BASE / bootstrap_campaign_id).resolve()
    expected_config = (
        campaign_root / "runtime_configs" / "preflight" / "semigroup_preflight.yaml"
    )
    if config_file != expected_config:
        raise CampaignSemigroupClosureError("runtime config path is not canonical")
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CampaignSemigroupClosureError("runtime config must be a mapping")
    with _working_directory(root):
        frozen = resolve_frozen_effective_guidance_config(raw)
    if raw != frozen:
        raise CampaignSemigroupClosureError(
            "runtime config is not the exact frozen effective config"
        )
    if raw.get("mode") != "semigroup" or raw.get("phase") != "semigroup":
        raise CampaignSemigroupClosureError(
            "runtime config must be the semigroup preflight"
        )
    if raw.get("r9_campaign_id") != bootstrap_campaign_id:
        raise CampaignSemigroupClosureError(
            "runtime config campaign ID disagrees with its immutable path"
        )
    _require_sha(raw.get("r9_campaign_runtime_sha256"), "campaign runtime SHA256")
    _require_sha(raw.get("r9_manifest_contracts_sha256"), "manifest contracts SHA256")
    validate_r9_semigroup_preflight_config(raw)
    manifest_path = _resolve_config_path(
        root, raw.get("sample_id_manifest"), "sample manifest"
    )
    if _sha256_file(manifest_path) != raw.get("sample_id_manifest_sha256"):
        raise CampaignSemigroupClosureError("sample manifest SHA256 mismatch")
    if raw.get("r9_phase_manifest_sha256") != raw.get("sample_id_manifest_sha256"):
        raise CampaignSemigroupClosureError("preflight phase manifest SHA256 mismatch")
    manifest_ids = tuple(_read_manifest_ids(manifest_path))
    if len(manifest_ids) != SAMPLE_COUNT:
        raise CampaignSemigroupClosureError(
            "semigroup preflight manifest must contain exactly 64 IDs"
        )
    source_index_path = _resolve_config_path(root, raw.get("index"), "source index")
    if _sha256_file(source_index_path) != raw.get("index_sha256"):
        raise CampaignSemigroupClosureError("source index SHA256 mismatch")
    source_by_id = _load_index_sources(
        source_index_path,
        manifest_ids=manifest_ids,
        repo_root=root,
    )
    checkpoint_path = _resolve_config_path(
        root, raw.get("checkpoint"), "MeanFlow checkpoint"
    )
    if _sha256_file(checkpoint_path) != raw.get("checkpoint_sha256"):
        raise CampaignSemigroupClosureError("MeanFlow checkpoint SHA256 mismatch")
    campaign_runtime_path, campaign_runtime = _load_campaign_runtime(
        campaign_root,
        repo_root=root,
        config=raw,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
    )
    expected_shard_root = (
        campaign_root / "preflight" / "semigroup_preflight" / "shards"
    ).resolve()
    actual_shard_root = Path(shard_root)
    if not actual_shard_root.is_absolute():
        actual_shard_root = root / actual_shard_root
    actual_shard_root = actual_shard_root.resolve()
    if actual_shard_root != expected_shard_root or not actual_shard_root.is_dir():
        raise CampaignSemigroupClosureError(
            "--shard-root is not the exact bootstrap CID shard root"
        )
    if actual_shard_root.is_symlink():
        raise CampaignSemigroupClosureError("shard root must not be a symlink")
    allowed_root_entries = {f"shard_{index}" for index in range(SHARD_COUNT)} | {
        "shared"
    }
    actual_root_entries = {path.name for path in actual_shard_root.iterdir()}
    if (
        not actual_root_entries <= allowed_root_entries
        or not {f"shard_{index}" for index in range(SHARD_COUNT)} <= actual_root_entries
    ):
        raise CampaignSemigroupClosureError(
            "shard root does not contain exactly four canonical shard directories"
        )
    shared = actual_shard_root / "shared"
    if shared.exists() and (not shared.is_dir() or shared.is_symlink()):
        raise CampaignSemigroupClosureError(
            "shard shared asset cache root must be a real directory"
        )
    shard_dirs = tuple(
        (actual_shard_root / f"shard_{index}").resolve() for index in range(SHARD_COUNT)
    )
    for shard_dir in shard_dirs:
        if not shard_dir.is_dir() or shard_dir.is_symlink():
            raise CampaignSemigroupClosureError(
                f"invalid semigroup shard directory: {shard_dir}"
            )
    executed_config = _expected_executed_config(raw)
    return _Context(
        repo_root=root,
        config_path=config_file,
        config=raw,
        bootstrap_campaign_id=bootstrap_campaign_id,
        campaign_root=campaign_root,
        campaign_runtime_path=campaign_runtime_path,
        campaign_runtime=campaign_runtime,
        shard_root=actual_shard_root,
        shard_dirs=shard_dirs,
        manifest_path=manifest_path,
        manifest_ids=manifest_ids,
        source_index_path=source_index_path,
        source_by_id=source_by_id,
        checkpoint_path=checkpoint_path,
        executed_config=executed_config,
    )


def _load_campaign_runtime(
    campaign_root: Path,
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    checkpoint_path: Path,
) -> tuple[Path, dict[str, Any]]:
    runtime_path = _existing_file(
        campaign_root / "campaign_runtime.json", "bootstrap campaign runtime"
    )
    runtime = _read_json(runtime_path, "bootstrap campaign runtime")
    declared = _require_sha(
        runtime.get("campaign_runtime_sha256"), "campaign runtime SHA256"
    )
    if _contract_digest(runtime, "campaign_runtime_sha256") != declared:
        raise CampaignSemigroupClosureError("campaign runtime digest mismatch")
    if declared != config.get("r9_campaign_runtime_sha256"):
        raise CampaignSemigroupClosureError(
            "runtime config does not bind the bootstrap campaign runtime"
        )
    campaign_id = config["r9_campaign_id"]
    expected_bindings = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "determinism_policy_sha256": config["determinism_policy_sha256"],
        "attention_backend": config["attention_backend"],
        "manifest_contracts_sha256": config["r9_manifest_contracts_sha256"],
    }
    for field, expected in expected_bindings.items():
        if runtime.get(field) != expected:
            raise CampaignSemigroupClosureError(
                f"campaign runtime binding mismatch: {field}"
            )
    runtime_root = _resolved_optional_path(runtime.get("campaign_root"), repo_root)
    if runtime_root != campaign_root:
        raise CampaignSemigroupClosureError("campaign runtime root mismatch")
    checkpoint = runtime.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or _resolve_config_path(
            repo_root, checkpoint.get("path"), "campaign runtime checkpoint"
        )
        != checkpoint_path
        or checkpoint.get("sha256") != config["checkpoint_sha256"]
    ):
        raise CampaignSemigroupClosureError(
            "campaign runtime checkpoint binding mismatch"
        )
    manifests = runtime.get("manifests")
    calibration = (
        manifests.get("calibration_64") if isinstance(manifests, Mapping) else None
    )
    if (
        not isinstance(calibration, Mapping)
        or _resolve_config_path(
            repo_root,
            calibration.get("path"),
            "campaign runtime calibration manifest",
        )
        != manifest_path
        or calibration.get("sha256") != config["sample_id_manifest_sha256"]
        or calibration.get("sample_count") != SAMPLE_COUNT
        or calibration.get("ordered_sample_id_sha256")
        != _sample_id_digest(_read_manifest_ids(manifest_path))
    ):
        raise CampaignSemigroupClosureError(
            "campaign runtime calibration manifest binding mismatch"
        )
    return runtime_path, runtime


def _expected_executed_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(config)
    if "r9_execution_contract" in expected:
        raise CampaignSemigroupClosureError(
            "immutable preflight runtime config must precede CUDA execution"
        )
    expected["r9_execution_contract"] = validate_r9_execution_config(config)
    expected["arm_config_sha256"] = canonical_r9_arm_config_digest(expected)
    return expected


def _build_evidence(context: _Context) -> dict[str, Any]:
    shard_contracts = []
    actual_executed_config: dict[str, Any] | None = None
    for shard_index, shard_dir in enumerate(context.shard_dirs):
        expected_ids = list(context.manifest_ids[shard_index::SHARD_COUNT])
        shard_contract, shard_executed_config = _validate_shard(
            context,
            shard_index=shard_index,
            shard_dir=shard_dir,
            expected_ids=expected_ids,
        )
        if actual_executed_config is None:
            actual_executed_config = shard_executed_config
        elif shard_executed_config != actual_executed_config:
            raise CampaignSemigroupClosureError(
                "all four shards must share the exact executed config"
            )
        shard_contracts.append(shard_contract)
    if actual_executed_config is None:
        raise AssertionError("four required shards produced no executed config")
    _compare_executed_config(
        actual_executed_config,
        context.executed_config,
        repo_root=context.repo_root,
    )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_semigroup_campaign_evidence_v1",
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "campaign_runtime_sha256": context.config["r9_campaign_runtime_sha256"],
        "campaign_runtime": {
            "path": _path_text(context.campaign_runtime_path, context.repo_root),
            "file_sha256": _sha256_file(context.campaign_runtime_path),
            "contract_sha256": context.campaign_runtime["campaign_runtime_sha256"],
        },
        "runtime_config": {
            "path": _path_text(context.config_path, context.repo_root),
            "sha256": _sha256_file(context.config_path),
        },
        "executed_config_sha256": canonical_json_sha256(actual_executed_config),
        "shard_root": _path_text(context.shard_root, context.repo_root),
        "manifest": {
            "path": _path_text(context.manifest_path, context.repo_root),
            "sha256": _sha256_file(context.manifest_path),
            "sample_count": SAMPLE_COUNT,
            "ordered_sample_id_sha256": _sample_id_digest(context.manifest_ids),
        },
        "source_index": {
            "path": _path_text(context.source_index_path, context.repo_root),
            "sha256": _sha256_file(context.source_index_path),
        },
        "checkpoint": {
            "path": _path_text(context.checkpoint_path, context.repo_root),
            "sha256": _sha256_file(context.checkpoint_path),
        },
        "checkpoint_sha256": context.config["checkpoint_sha256"],
        "determinism_policy_sha256": context.config["determinism_policy_sha256"],
        "attention_backend": context.config["attention_backend"],
        "arm_config_sha256": context.config["arm_config_sha256"],
        "shards": shard_contracts,
    }
    payload["evidence_manifest_sha256"] = _contract_digest(
        payload, "evidence_manifest_sha256"
    )
    return payload


def _prepare_visual_review_assignment(
    context: _Context, formal_campaign_id: str
) -> dict[str, Any]:
    preflight_root = (context.campaign_root / "preflight").resolve()
    evidence_path = preflight_root / "evidence_manifest.json"
    assignment_path = preflight_root / "visual_review_assignment.json"
    blinding_map_path = preflight_root / "visual_review_blinding_map.json"
    sheet_root = preflight_root / "visual_review_sheets"
    review_path = preflight_root / "visual_review.json"
    for path, label in (
        (evidence_path, "evidence manifest"),
        (assignment_path, "visual review assignment"),
        (blinding_map_path, "visual review blinding map"),
        (sheet_root, "visual review sheet root"),
        (review_path, "visual review"),
    ):
        if path.exists() or path.is_symlink():
            raise CampaignSemigroupClosureError(
                f"canonical {label} already exists before review preparation"
            )

    evidence = _build_evidence(context)
    review_rows = _ordered_review_rows(context, evidence)
    _write_exclusive_json(evidence_path, evidence, canonical=False)
    evidence_file_sha256 = _sha256_file(evidence_path)

    blinding_context = {
        "schema_version": 1,
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "formal_campaign_id": formal_campaign_id,
        "evidence_manifest_sha256": evidence["evidence_manifest_sha256"],
        "ordered_sample_id_sha256": _sample_id_digest(context.manifest_ids),
        "registered_splits": list(SPLIT_KEYS),
    }
    blinding_context_sha256 = canonical_json_sha256(blinding_context)
    split_order = _secure_permutation(SPLIT_KEYS)
    used_ids: set[str] = set()
    assignment_conditions = []
    map_conditions: dict[str, Any] = {}
    expected_sheet_files: set[Path] = set()
    sheet_root.mkdir(mode=0o755, parents=False, exist_ok=False)
    for split in split_order:
        condition_id = _new_blinded_id("condition", used_ids)
        role_order = _secure_permutation(_BLINDED_ROLES)
        column_ids = [_new_blinded_id("column", used_ids) for _ in _BLINDED_ROLES]
        condition_root = sheet_root / condition_id
        condition_root.mkdir(mode=0o755, exist_ok=False)
        pages = _write_blinded_contact_pages(
            context=context,
            output_root=condition_root,
            rows=review_rows,
            condition_id=condition_id,
            split=split,
            role_order=role_order,
            column_ids=column_ids,
        )
        expected_sheet_files.update(
            _resolve_config_path(
                context.repo_root,
                page["path"],
                "prepared contact sheet",
            )
            for page in pages
        )
        assignment_conditions.append(
            {
                "condition_id": condition_id,
                "column_ids": column_ids,
                "pages": pages,
            }
        )
        map_conditions[condition_id] = {
            "split": split,
            "columns": [
                {"column_id": column_id, "role": role}
                for column_id, role in zip(column_ids, role_order, strict=True)
            ],
        }
    _validate_exact_file_tree(
        sheet_root,
        expected_sheet_files,
        label="prepared visual review contact sheets",
    )

    blinding_map = {
        "schema_version": 1,
        "contract_type": "safa_r9_semigroup_visual_review_blinding_map_v1",
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "formal_campaign_id": formal_campaign_id,
        "evidence_manifest_sha256": evidence["evidence_manifest_sha256"],
        "blinding_context_sha256": blinding_context_sha256,
        "registered_splits": list(SPLIT_KEYS),
        "conditions": map_conditions,
    }
    blinding_map["blinding_map_sha256"] = _contract_digest(
        blinding_map, "blinding_map_sha256"
    )
    blinding_map_file_sha256 = _write_exclusive_json(
        blinding_map_path,
        blinding_map,
        canonical=False,
        permissions=0,
    )

    assignment = {
        "schema_version": 1,
        "contract_type": "safa_r9_semigroup_visual_review_assignment_v1",
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "formal_campaign_id": formal_campaign_id,
        "review_type": "independent_blinded_semigroup_structure_review",
        "decision_rule": "passed_if_and_only_if_severe_count_equals_zero",
        "sample_count": SAMPLE_COUNT,
        "reviewed_sample_ids": list(context.manifest_ids),
        "ordered_sample_id_sha256": _sample_id_digest(context.manifest_ids),
        "registered_splits": list(SPLIT_KEYS),
        "blinding_context_sha256": blinding_context_sha256,
        "evidence_manifest": {
            "path": _path_text(evidence_path, context.repo_root),
            "file_sha256": evidence_file_sha256,
            "contract_sha256": evidence["evidence_manifest_sha256"],
        },
        "blinding_map": {
            "path": _path_text(blinding_map_path, context.repo_root),
            "file_sha256": blinding_map_file_sha256,
            "contract_sha256": blinding_map["blinding_map_sha256"],
        },
        "conditions": assignment_conditions,
    }
    _reject_assignment_decision_fields(assignment)
    assignment["visual_review_assignment_sha256"] = _contract_digest(
        assignment, "visual_review_assignment_sha256"
    )
    _write_exclusive_json(assignment_path, assignment, canonical=False)
    _validate_review_assignment(
        context=context,
        evidence=evidence,
        formal_campaign_id=formal_campaign_id,
    )
    return {
        "schema_version": 1,
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "formal_campaign_id": formal_campaign_id,
        "evidence_manifest": {
            "path": _path_text(evidence_path, context.repo_root),
            "file_sha256": evidence_file_sha256,
            "contract_sha256": evidence["evidence_manifest_sha256"],
        },
        "visual_review_assignment": {
            "path": _path_text(assignment_path, context.repo_root),
            "file_sha256": _sha256_file(assignment_path),
            "contract_sha256": assignment["visual_review_assignment_sha256"],
        },
        "condition_count": len(assignment_conditions),
        "contact_sheet_count": sum(
            len(condition["pages"]) for condition in assignment_conditions
        ),
        "review_decisions_materialized": False,
    }


def _ordered_review_rows(
    context: _Context, evidence: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    shards = evidence.get("shards")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise CampaignSemigroupClosureError(
            "semigroup evidence has no exact four-shard visual inventory"
        )
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise CampaignSemigroupClosureError("semigroup evidence shard is invalid")
        assets = shard.get("sample_assets")
        if not isinstance(assets, list):
            raise CampaignSemigroupClosureError(
                "semigroup evidence shard has no sample assets"
            )
        for sample in assets:
            if not isinstance(sample, Mapping):
                raise CampaignSemigroupClosureError(
                    "semigroup visual sample asset is invalid"
                )
            sample_id = str(sample.get("sample_id", ""))
            if sample_id in rows_by_id:
                raise CampaignSemigroupClosureError(
                    "semigroup visual evidence contains duplicate sample IDs"
                )
            rows_by_id[sample_id] = sample
    if set(rows_by_id) != set(context.manifest_ids):
        raise CampaignSemigroupClosureError(
            "semigroup visual evidence does not cover the exact 64 IDs"
        )
    rows = []
    for sample_id in context.manifest_ids:
        sample = rows_by_id[sample_id]
        splits = sample.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_KEYS):
            raise CampaignSemigroupClosureError(
                f"semigroup visual evidence split inventory mismatch: {sample_id}"
            )
        rows.append(
            {
                "sample_id": sample_id,
                "source": _bound_review_asset(
                    context, sample.get("source"), f"review source {sample_id}"
                ),
                "generated_direct": _bound_review_asset(
                    context,
                    sample.get("generated_direct"),
                    f"review direct generation {sample_id}",
                ),
                "splits": {
                    split: _bound_review_asset(
                        context,
                        splits[split],
                        f"review split generation {sample_id}/{split}",
                    )
                    for split in SPLIT_KEYS
                },
            }
        )
    return rows


def _bound_review_asset(context: _Context, value: Any, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise CampaignSemigroupClosureError(f"{label} binding is not canonical")
    path = _resolve_config_path(context.repo_root, value.get("path"), label)
    if _sha256_file(path) != _require_sha(value.get("sha256"), f"{label} SHA256"):
        raise CampaignSemigroupClosureError(f"{label} SHA256 mismatch")
    return path


def _write_blinded_contact_pages(
    *,
    context: _Context,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    condition_id: str,
    split: str,
    role_order: Sequence[str],
    column_ids: Sequence[str],
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    if (
        set(role_order) != set(_BLINDED_ROLES)
        or len(role_order) != len(_BLINDED_ROLES)
        or len(column_ids) != len(_BLINDED_ROLES)
    ):
        raise CampaignSemigroupClosureError("blinded contact sheet columns are invalid")
    pages = []
    for page_index, start in enumerate(range(0, len(rows), _REVIEW_ROWS_PER_PAGE)):
        page_rows = rows[start : start + _REVIEW_ROWS_PER_PAGE]
        sheet = Image.new(
            "RGB",
            (
                _REVIEW_TILE_SIZE * len(_BLINDED_ROLES),
                _REVIEW_HEADER_HEIGHT + _REVIEW_TILE_SIZE * len(page_rows),
            ),
            (245, 245, 245),
        )
        draw = ImageDraw.Draw(sheet)
        for column_index, column_id in enumerate(column_ids):
            draw.text(
                (column_index * _REVIEW_TILE_SIZE + 4, 8),
                column_id,
                fill=(0, 0, 0),
            )
        for row_index, row in enumerate(page_rows):
            for column_index, role in enumerate(role_order):
                path = row["splits"][split] if role == "split" else row[role]
                with Image.open(path) as image:
                    tile = image.convert("RGB").resize(
                        (_REVIEW_TILE_SIZE, _REVIEW_TILE_SIZE),
                        Image.Resampling.BILINEAR,
                    )
                sheet.paste(
                    tile,
                    (
                        column_index * _REVIEW_TILE_SIZE,
                        _REVIEW_HEADER_HEIGHT + row_index * _REVIEW_TILE_SIZE,
                    ),
                )
        buffer = BytesIO()
        sheet.save(buffer, format="PNG")
        path = output_root / f"page_{page_index:03d}.png"
        _write_exclusive_bytes(path, buffer.getvalue())
        page = {
            "schema_version": 1,
            "condition_id": condition_id,
            "page_index": page_index,
            "path": _path_text(path, context.repo_root),
            "file_sha256": _sha256_file(path),
            "column_ids": list(column_ids),
            "sample_ids": [str(row["sample_id"]) for row in page_rows],
        }
        page["sheet_contract_sha256"] = _contract_digest(page, "sheet_contract_sha256")
        pages.append(page)
    return pages


def _secure_permutation(values: Sequence[str]) -> list[str]:
    result = list(values)
    for index in range(len(result) - 1, 0, -1):
        swap = secrets.randbelow(index + 1)
        result[index], result[swap] = result[swap], result[index]
    return result


def _new_blinded_id(prefix: str, used: set[str]) -> str:
    while True:
        value = f"{prefix}_{secrets.token_hex(6)}"
        if value not in used:
            used.add(value)
            return value


def _validate_shard(
    context: _Context,
    *,
    shard_index: int,
    shard_dir: Path,
    expected_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = {path.name for path in shard_dir.iterdir()}
    if entries != _SHARD_FILES | _SHARD_DIRECTORIES:
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} contains missing or unregistered evidence files"
        )
    for name in sorted(_SHARD_FILES):
        _existing_file(shard_dir / name, f"shard {shard_index} {name}")
    generation_path = shard_dir / "generation_result.json"
    run_manifest_path = shard_dir / "run_manifest.json"
    if generation_path.read_bytes() != run_manifest_path.read_bytes():
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} run manifest differs from generation result"
        )
    generation = _read_json(generation_path, "generation result")
    if (
        generation.get("status") != "complete"
        or generation.get("mode") != "semigroup"
        or generation.get("sample_count") != len(expected_ids)
        or generation.get("sample_id_sha256") != _sample_id_digest(expected_ids)
        or generation.get("shard") != {"index": shard_index, "count": SHARD_COUNT}
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} generation completion binding mismatch"
        )
    if generation.get("arm_config_sha256") != context.config["arm_config_sha256"]:
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} arm config SHA256 mismatch"
        )
    if generation.get("r9_execution_contract") != validate_r9_execution_config(
        context.config
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} determinism execution contract mismatch"
        )
    checkpoint = generation.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or any(
        checkpoint.get(field) != expected
        for field, expected in (
            ("sha256", context.config["checkpoint_sha256"]),
            ("attention_backend_requested", R9_ATTENTION_BACKEND),
            ("attention_backend_resolved", R9_ATTENTION_BACKEND),
        )
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} checkpoint/backend contract mismatch"
        )
    executed_config = generation.get("config")
    if not isinstance(executed_config, dict):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} executed config is missing"
        )
    _compare_executed_config(
        executed_config,
        context.executed_config,
        repo_root=context.repo_root,
    )
    resume = generation.get("resume_contract")
    if not isinstance(resume, Mapping):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume contract is missing"
        )
    on_disk_resume = _read_json(shard_dir / "resume_contract.json", "resume contract")
    if on_disk_resume != resume:
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume contract bytes disagree with generation"
        )
    _validate_resume_contract(
        resume,
        context=context,
        shard_index=shard_index,
        expected_ids=expected_ids,
        executed_config=executed_config,
    )
    completion = _read_json(shard_dir / "completion.json", "completion marker")
    _validate_completion(
        completion,
        context=context,
        shard_index=shard_index,
        shard_dir=shard_dir,
        expected_ids=expected_ids,
    )
    _validate_verified_completion(
        _read_json(shard_dir / "verified_completion.json", "verified completion"),
        context=context,
        shard_index=shard_index,
        shard_dir=shard_dir,
        expected_ids=expected_ids,
    )
    sample_manifest_rows = _read_jsonl(
        shard_dir / "sample_id_manifest.jsonl", "shard sample manifest"
    )
    per_sample_rows = _read_jsonl(shard_dir / "per_sample.jsonl", "per-sample evidence")
    semigroup = _read_json(shard_dir / "semigroup.json", "semigroup evidence")
    semigroup_rows = semigroup.get("rows")
    if (
        semigroup.get("mode") != "semigroup"
        or semigroup.get("split_times") != [0.25, 0.5, 0.75]
        or not isinstance(semigroup_rows, list)
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} semigroup evidence is not canonical"
        )
    for label, rows in (
        ("sample manifest", sample_manifest_rows),
        ("per-sample", per_sample_rows),
        ("semigroup", semigroup_rows),
    ):
        ids = [row.get("sample_id") for row in rows if isinstance(row, Mapping)]
        if ids != expected_ids:
            raise CampaignSemigroupClosureError(
                f"shard {shard_index} {label} sample order mismatch"
            )
    semigroup_by_id = {row["sample_id"]: row for row in semigroup_rows}
    manifest_by_id = {row["sample_id"]: row for row in sample_manifest_rows}
    sample_assets = []
    referenced_by_directory: dict[str, set[Path]] = {
        name: set() for name in _SHARD_DIRECTORIES
    }
    for row in per_sample_rows:
        sample_id = str(row["sample_id"])
        source = _existing_file(
            row.get("source"),
            f"source image {sample_id}",
            repo_root=context.repo_root,
        )
        if source != context.source_by_id[sample_id]:
            raise CampaignSemigroupClosureError(
                f"shard {shard_index} source index binding mismatch for {sample_id}"
            )
        if manifest_by_id[sample_id].get("source") != row.get("source"):
            raise CampaignSemigroupClosureError(
                f"shard {shard_index} source path disagreement for {sample_id}"
            )
        native = _contained_evidence_file(
            context,
            shard_dir,
            row.get("native"),
            directory="native_images",
            label=f"native image {sample_id}",
        )
        generated = _contained_evidence_file(
            context,
            shard_dir,
            row.get("generated"),
            directory="generated_images",
            label=f"generated image {sample_id}",
        )
        referenced_by_directory["native_images"].add(native)
        referenced_by_directory["generated_images"].add(generated)
        row_splits = row.get("semigroup")
        semigroup_splits = semigroup_by_id[sample_id].get("splits")
        if (
            not isinstance(row_splits, Mapping)
            or not isinstance(semigroup_splits, Mapping)
            or set(row_splits) != set(SPLIT_KEYS)
            or row_splits != semigroup_splits
        ):
            raise CampaignSemigroupClosureError(
                f"shard {shard_index} split evidence disagreement for {sample_id}"
            )
        split_assets = {}
        for split in SPLIT_KEYS:
            split_row = row_splits[split]
            if not isinstance(split_row, Mapping):
                raise CampaignSemigroupClosureError(
                    f"shard {shard_index} split row is invalid for {sample_id}"
                )
            for metric in (
                "latent_residual",
                "endpoint_e0_cosine",
                "decoded_pixel_l1",
                "decoded_psnr",
            ):
                _finite(split_row.get(metric), f"{sample_id}/{split}/{metric}")
            decoded = _contained_evidence_file(
                context,
                shard_dir,
                split_row.get("decoded_image"),
                directory="semigroup_split_images",
                label=f"split image {sample_id}/{split}",
            )
            referenced_by_directory["semigroup_split_images"].add(decoded)
            split_assets[split] = {
                "path": _path_text(decoded, context.repo_root),
                "sha256": _sha256_file(decoded),
            }
        sample_assets.append(
            {
                "sample_id": sample_id,
                "source": {"path": str(source), "sha256": _sha256_file(source)},
                "native": {
                    "path": _path_text(native, context.repo_root),
                    "sha256": _sha256_file(native),
                },
                "generated_direct": {
                    "path": _path_text(generated, context.repo_root),
                    "sha256": _sha256_file(generated),
                },
                "splits": split_assets,
            }
        )
    for directory, referenced in referenced_by_directory.items():
        root = shard_dir / directory
        if not root.is_dir() or root.is_symlink():
            raise CampaignSemigroupClosureError(
                f"shard {shard_index} evidence directory is invalid: {directory}"
            )
        _validate_exact_file_tree(
            root,
            referenced,
            label=f"shard {shard_index} {directory}",
        )
    file_contracts = {
        name: {
            "path": _path_text(shard_dir / name, context.repo_root),
            "sha256": _sha256_file(shard_dir / name),
        }
        for name in sorted(_SHARD_FILES)
    }
    payload = {
        "shard_index": shard_index,
        "sample_count": len(expected_ids),
        "ordered_sample_id_sha256": _sample_id_digest(expected_ids),
        "files": file_contracts,
        "sample_assets": sample_assets,
    }
    payload["shard_evidence_sha256"] = _contract_digest(
        payload, "shard_evidence_sha256"
    )
    return payload, dict(executed_config)


def _compare_executed_config(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, repo_root: Path
) -> None:
    if set(actual) != set(expected):
        raise CampaignSemigroupClosureError(
            "executed config fields disagree with immutable runtime config"
        )
    for field in expected:
        if field in {"asset_digest_cache", "asset_digest_cache_root"}:
            if _resolved_optional_path(
                actual[field], repo_root
            ) != _resolved_optional_path(expected[field], repo_root):
                raise CampaignSemigroupClosureError(
                    f"executed config path mismatch: {field}"
                )
        elif actual[field] != expected[field]:
            raise CampaignSemigroupClosureError(f"executed config mismatch: {field}")


def _validate_resume_contract(
    resume: Mapping[str, Any],
    *,
    context: _Context,
    shard_index: int,
    expected_ids: Sequence[str],
    executed_config: Mapping[str, Any],
) -> None:
    if resume.get("mode") != "semigroup" or resume.get("seed") != context.config.get(
        "sampling_seed"
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume mode/seed mismatch"
        )
    if resume.get("arm_config_sha256") != context.config["arm_config_sha256"]:
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume arm digest mismatch"
        )
    if resume.get("sample_id_sha256") != _sample_id_digest(expected_ids):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume sample digest mismatch"
        )
    shard = resume.get("shard")
    if not isinstance(shard, Mapping) or any(
        shard.get(field) != value
        for field, value in (("index", shard_index), ("count", SHARD_COUNT))
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume coordinates mismatch"
        )
    if resume.get("config") != executed_config:
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume executed config mismatch"
        )
    if resume.get("r9_execution_contract") != validate_r9_execution_config(
        context.config
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume determinism mismatch"
        )
    checkpoint = resume.get("checkpoint")
    manifest = resume.get("input_sample_manifest")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("sha256") != context.config["checkpoint_sha256"]
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume checkpoint mismatch"
        )
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("sha256") != context.config["sample_id_manifest_sha256"]
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume manifest mismatch"
        )
    if (
        _resolve_config_path(
            context.repo_root,
            manifest.get("path"),
            "resume sample manifest",
        )
        != context.manifest_path
    ):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} resume manifest path mismatch"
        )


def _validate_completion(
    completion: Mapping[str, Any],
    *,
    context: _Context,
    shard_index: int,
    shard_dir: Path,
    expected_ids: Sequence[str],
) -> None:
    expected = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": len(expected_ids),
        "sample_id_sha256": _sample_id_digest(expected_ids),
        "arm_config_sha256": context.config["arm_config_sha256"],
    }
    if any(completion.get(field) != value for field, value in expected.items()):
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} completion marker mismatch"
        )
    for field, filename in (
        ("generation_result", "generation_result.json"),
        ("run_manifest", "run_manifest.json"),
    ):
        path = _resolve_config_path(
            context.repo_root, completion.get(field), f"completion {field}"
        )
        if path != (shard_dir / filename).resolve():
            raise CampaignSemigroupClosureError(
                f"shard {shard_index} completion path mismatch: {field}"
            )


def _validate_verified_completion(
    verified: Mapping[str, Any],
    *,
    context: _Context,
    shard_index: int,
    shard_dir: Path,
    expected_ids: Sequence[str],
) -> None:
    expected = {
        "schema_version": 1,
        "contract_type": "safa_r9_verified_worker_completion_v1",
        "worker_id": (f"preflight:semigroup_preflight:shard-{shard_index}"),
        "runtime_config_sha256": _sha256_file(context.config_path),
        "completion_sha256": _sha256_file(shard_dir / "completion.json"),
        "generation_result_sha256": _sha256_file(shard_dir / "generation_result.json"),
        "run_manifest_sha256": _sha256_file(shard_dir / "run_manifest.json"),
        "sample_count": len(expected_ids),
        "sample_id_sha256": _sample_id_digest(expected_ids),
        "arm_config_sha256": context.config["arm_config_sha256"],
        "manifest_contracts_sha256": context.config["r9_manifest_contracts_sha256"],
        "phase_manifest_sha256": context.config["r9_phase_manifest_sha256"],
        "campaign_runtime_sha256": context.config["r9_campaign_runtime_sha256"],
    }
    expected["verified_completion_sha256"] = _contract_digest(
        expected, "verified_completion_sha256"
    )
    if dict(verified) != expected:
        raise CampaignSemigroupClosureError(
            f"shard {shard_index} verified completion binding mismatch"
        )


def _contained_evidence_file(
    context: _Context,
    shard_dir: Path,
    value: Any,
    *,
    directory: str,
    label: str,
) -> Path:
    path = _resolve_config_path(context.repo_root, value, label)
    expected_root = (shard_dir / directory).resolve()
    _relative_to(path, expected_root, label)
    if path.is_symlink():
        raise CampaignSemigroupClosureError(f"{label} must not be a symlink")
    return path


def _validate_exact_file_tree(
    root: Path, expected_files: set[Path], *, label: str
) -> None:
    expected = {path.resolve() for path in expected_files}
    expected_directories: set[Path] = set()
    for path in expected:
        parent = path.parent
        while parent != root:
            expected_directories.add(parent)
            parent = parent.parent
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise CampaignSemigroupClosureError(f"{label} contains a symlink")
    actual_files = {path.resolve() for path in entries if path.is_file()}
    actual_directories = {path.resolve() for path in entries if path.is_dir()}
    if actual_files != expected or actual_directories != expected_directories:
        raise CampaignSemigroupClosureError(f"{label} inventory is not exact")


def _validate_visual_review_without_map(
    visual_review_path: Path,
    *,
    context: _Context,
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    review_path = _existing_file(
        visual_review_path, "visual review", repo_root=context.repo_root
    )
    expected_review_path = (
        context.campaign_root / "preflight" / "visual_review.json"
    ).resolve()
    if review_path != expected_review_path:
        raise CampaignSemigroupClosureError(
            "--visual-review is not the exact bootstrap CID review contract"
        )
    review = _read_json(review_path, "visual review")
    expected_fields = {
        "schema_version",
        "contract_type",
        "bootstrap_campaign_id",
        "formal_campaign_id",
        "review_type",
        "decision_rule",
        "sample_count",
        "reviewed_sample_ids",
        "evidence_manifest_sha256",
        "visual_review_assignment_sha256",
        "conditions",
        "visual_review_sha256",
    }
    if (
        set(review) != expected_fields
        or review.get("schema_version") != 2
        or review.get("contract_type") != "safa_r9_semigroup_visual_review_v2"
    ):
        raise CampaignSemigroupClosureError(
            "visual review contract fields are not canonical"
        )
    if review.get("visual_review_sha256") != _contract_digest(
        review, "visual_review_sha256"
    ):
        raise CampaignSemigroupClosureError("visual review digest mismatch")
    if review.get("bootstrap_campaign_id") != context.bootstrap_campaign_id:
        raise CampaignSemigroupClosureError(
            "visual review bootstrap campaign ID mismatch"
        )
    formal_campaign_id = str(review.get("formal_campaign_id", ""))
    _require_campaign_id(formal_campaign_id, "formal campaign ID")
    if formal_campaign_id == context.bootstrap_campaign_id:
        raise CampaignSemigroupClosureError(
            "bootstrap and formal campaign IDs must be distinct"
        )
    if (
        review.get("review_type") != "independent_blinded_semigroup_structure_review"
        or review.get("decision_rule")
        != "passed_if_and_only_if_severe_count_equals_zero"
        or review.get("sample_count") != SAMPLE_COUNT
        or review.get("reviewed_sample_ids") != list(context.manifest_ids)
    ):
        raise CampaignSemigroupClosureError(
            "visual review does not cover the exact ordered 64-sample manifest"
        )
    if review.get("evidence_manifest_sha256") != evidence.get(
        "evidence_manifest_sha256"
    ):
        raise CampaignSemigroupClosureError(
            "visual review evidence manifest SHA256 mismatch"
        )
    assignment = _validate_review_assignment(
        context=context,
        evidence=evidence,
        formal_campaign_id=formal_campaign_id,
    )
    if review.get("visual_review_assignment_sha256") != assignment.get(
        "visual_review_assignment_sha256"
    ):
        raise CampaignSemigroupClosureError("visual review assignment SHA256 mismatch")
    assignment_conditions = assignment["conditions"]
    condition_ids = [condition["condition_id"] for condition in assignment_conditions]
    conditions = review.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(condition_ids):
        raise CampaignSemigroupClosureError(
            "visual review must cover exactly the three blinded conditions"
        )
    manifest_ids = set(context.manifest_ids)
    for condition_id in condition_ids:
        row = conditions[condition_id]
        if not isinstance(row, Mapping) or set(row) != {
            "passed",
            "severe_count",
            "severe_sample_ids",
        }:
            raise CampaignSemigroupClosureError(
                f"visual review condition fields are not canonical: {condition_id}"
            )
        severe_ids = row.get("severe_sample_ids")
        count = row.get("severe_count")
        passed = row.get("passed")
        if (
            not isinstance(severe_ids, list)
            or any(not isinstance(sample_id, str) for sample_id in severe_ids)
            or len(set(severe_ids)) != len(severe_ids)
            or not set(severe_ids) <= manifest_ids
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count != len(severe_ids)
            or not isinstance(passed, bool)
            or passed != (count == 0)
        ):
            raise CampaignSemigroupClosureError(
                f"visual review passed/severe contract mismatch: {condition_id}"
            )
    return review, formal_campaign_id, assignment


def _validate_review_assignment(
    *,
    context: _Context,
    evidence: Mapping[str, Any],
    formal_campaign_id: str,
) -> dict[str, Any]:
    assignment_path = _existing_file(
        context.campaign_root / "preflight" / "visual_review_assignment.json",
        "visual review assignment",
        repo_root=context.repo_root,
    )
    assignment = _read_json(assignment_path, "visual review assignment")
    _reject_assignment_decision_fields(assignment)
    expected_fields = {
        "schema_version",
        "contract_type",
        "bootstrap_campaign_id",
        "formal_campaign_id",
        "review_type",
        "decision_rule",
        "sample_count",
        "reviewed_sample_ids",
        "ordered_sample_id_sha256",
        "registered_splits",
        "blinding_context_sha256",
        "evidence_manifest",
        "blinding_map",
        "conditions",
        "visual_review_assignment_sha256",
    }
    if (
        set(assignment) != expected_fields
        or assignment.get("schema_version") != 1
        or assignment.get("contract_type")
        != "safa_r9_semigroup_visual_review_assignment_v1"
        or assignment.get("visual_review_assignment_sha256")
        != _contract_digest(assignment, "visual_review_assignment_sha256")
    ):
        raise CampaignSemigroupClosureError(
            "visual review assignment fields or digest are not canonical"
        )
    if (
        assignment.get("bootstrap_campaign_id") != context.bootstrap_campaign_id
        or assignment.get("formal_campaign_id") != formal_campaign_id
        or assignment.get("review_type")
        != "independent_blinded_semigroup_structure_review"
        or assignment.get("decision_rule")
        != "passed_if_and_only_if_severe_count_equals_zero"
        or assignment.get("sample_count") != SAMPLE_COUNT
        or assignment.get("reviewed_sample_ids") != list(context.manifest_ids)
        or assignment.get("ordered_sample_id_sha256")
        != _sample_id_digest(context.manifest_ids)
        or assignment.get("registered_splits") != list(SPLIT_KEYS)
    ):
        raise CampaignSemigroupClosureError(
            "visual review assignment campaign/ID/split binding mismatch"
        )
    expected_blinding_context = canonical_json_sha256(
        {
            "schema_version": 1,
            "bootstrap_campaign_id": context.bootstrap_campaign_id,
            "formal_campaign_id": formal_campaign_id,
            "evidence_manifest_sha256": evidence["evidence_manifest_sha256"],
            "ordered_sample_id_sha256": _sample_id_digest(context.manifest_ids),
            "registered_splits": list(SPLIT_KEYS),
        }
    )
    if assignment.get("blinding_context_sha256") != expected_blinding_context:
        raise CampaignSemigroupClosureError(
            "visual review assignment blinding context mismatch"
        )
    evidence_binding = assignment.get("evidence_manifest")
    expected_evidence_path = (
        context.campaign_root / "preflight" / "evidence_manifest.json"
    ).resolve()
    if not isinstance(evidence_binding, Mapping) or set(evidence_binding) != {
        "path",
        "file_sha256",
        "contract_sha256",
    }:
        raise CampaignSemigroupClosureError(
            "visual review assignment evidence binding is not canonical"
        )
    evidence_path = _resolve_config_path(
        context.repo_root,
        evidence_binding.get("path"),
        "assigned evidence manifest",
    )
    if (
        evidence_path != expected_evidence_path
        or _sha256_file(evidence_path)
        != _require_sha(
            evidence_binding.get("file_sha256"),
            "assigned evidence manifest file SHA256",
        )
        or evidence_binding.get("contract_sha256")
        != evidence.get("evidence_manifest_sha256")
        or _read_json(evidence_path, "assigned evidence manifest") != dict(evidence)
    ):
        raise CampaignSemigroupClosureError(
            "visual review assignment evidence manifest mismatch"
        )
    map_binding = assignment.get("blinding_map")
    expected_map_path = (
        context.campaign_root / "preflight" / "visual_review_blinding_map.json"
    )
    if not isinstance(map_binding, Mapping) or set(map_binding) != {
        "path",
        "file_sha256",
        "contract_sha256",
    }:
        raise CampaignSemigroupClosureError(
            "visual review assignment blinding map binding is not canonical"
        )
    expected_map_text = str(expected_map_path.relative_to(context.repo_root))
    if (
        map_binding.get("path") != expected_map_text
        or str(map_binding.get("file_sha256", ""))
        != _require_sha(
            map_binding.get("file_sha256"),
            "visual review blinding map file SHA256",
        )
        or str(map_binding.get("contract_sha256", ""))
        != _require_sha(
            map_binding.get("contract_sha256"),
            "visual review blinding map contract SHA256",
        )
    ):
        raise CampaignSemigroupClosureError(
            "visual review assignment blinding map binding mismatch"
        )

    conditions = assignment.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(SPLIT_KEYS):
        raise CampaignSemigroupClosureError(
            "visual review assignment must contain three blinded conditions"
        )
    seen_condition_ids: set[str] = set()
    seen_column_ids: set[str] = set()
    expected_sheets: set[Path] = set()
    sheet_root = (
        context.campaign_root / "preflight" / "visual_review_sheets"
    ).resolve()
    for condition in conditions:
        if not isinstance(condition, Mapping) or set(condition) != {
            "condition_id",
            "column_ids",
            "pages",
        }:
            raise CampaignSemigroupClosureError(
                "visual review assignment condition fields are not canonical"
            )
        condition_id = str(condition.get("condition_id", ""))
        if (
            _BLINDED_CONDITION_ID.fullmatch(condition_id) is None
            or condition_id in seen_condition_ids
        ):
            raise CampaignSemigroupClosureError(
                "visual review assignment condition ID is not opaque and unique"
            )
        seen_condition_ids.add(condition_id)
        column_ids = condition.get("column_ids")
        if (
            not isinstance(column_ids, list)
            or len(column_ids) != len(_BLINDED_ROLES)
            or len(set(column_ids)) != len(_BLINDED_ROLES)
            or any(
                not isinstance(column_id, str)
                or _BLINDED_COLUMN_ID.fullmatch(column_id) is None
                or column_id in seen_column_ids
                for column_id in column_ids
            )
        ):
            raise CampaignSemigroupClosureError(
                "visual review assignment column IDs are not opaque and unique"
            )
        seen_column_ids.update(column_ids)
        pages = condition.get("pages")
        expected_page_count = (
            SAMPLE_COUNT + _REVIEW_ROWS_PER_PAGE - 1
        ) // _REVIEW_ROWS_PER_PAGE
        if not isinstance(pages, list) or len(pages) != expected_page_count:
            raise CampaignSemigroupClosureError(
                "visual review assignment does not cover every contact sheet page"
            )
        flattened_ids = []
        condition_root = sheet_root / condition_id
        for expected_index, page in enumerate(pages):
            if not isinstance(page, Mapping) or set(page) != {
                "schema_version",
                "condition_id",
                "page_index",
                "path",
                "file_sha256",
                "column_ids",
                "sample_ids",
                "sheet_contract_sha256",
            }:
                raise CampaignSemigroupClosureError(
                    "visual review assignment page fields are not canonical"
                )
            sample_ids = page.get("sample_ids")
            expected_ids = list(
                context.manifest_ids[
                    expected_index * _REVIEW_ROWS_PER_PAGE : (expected_index + 1)
                    * _REVIEW_ROWS_PER_PAGE
                ]
            )
            path = _resolve_config_path(
                context.repo_root,
                page.get("path"),
                "assigned contact sheet",
            )
            if (
                page.get("schema_version") != 1
                or page.get("condition_id") != condition_id
                or page.get("page_index") != expected_index
                or page.get("column_ids") != column_ids
                or sample_ids != expected_ids
                or path.parent != condition_root
                or path.name != f"page_{expected_index:03d}.png"
                or _sha256_file(path)
                != _require_sha(page.get("file_sha256"), "contact sheet file SHA256")
                or page.get("sheet_contract_sha256")
                != _contract_digest(page, "sheet_contract_sha256")
            ):
                raise CampaignSemigroupClosureError(
                    "visual review contact sheet page/ID/SHA binding mismatch"
                )
            flattened_ids.extend(sample_ids)
            expected_sheets.add(path)
        if flattened_ids != list(context.manifest_ids):
            raise CampaignSemigroupClosureError(
                "visual review assignment page IDs are incomplete or reordered"
            )
    _validate_exact_file_tree(
        sheet_root,
        expected_sheets,
        label="visual review contact sheets",
    )
    return assignment


def _load_blinding_map_after_complete_review(
    *,
    context: _Context,
    evidence: Mapping[str, Any],
    assignment: Mapping[str, Any],
    formal_campaign_id: str,
) -> dict[str, Any]:
    map_binding = assignment["blinding_map"]
    map_path = _existing_file(
        map_binding["path"],
        "visual review blinding map",
        repo_root=context.repo_root,
    )
    if stat.S_IMODE(map_path.stat().st_mode) != 0:
        raise CampaignSemigroupClosureError(
            "visual review blinding map was revealed before finalization"
        )
    map_path.chmod(stat.S_IRUSR)
    if _sha256_file(map_path) != _require_sha(
        map_binding.get("file_sha256"), "visual review blinding map file SHA256"
    ):
        raise CampaignSemigroupClosureError(
            "visual review blinding map file SHA256 mismatch"
        )
    blinding_map = _read_json(map_path, "visual review blinding map")
    _validate_blinding_map_contract(
        blinding_map,
        assignment=assignment,
        evidence_manifest_sha256=str(evidence["evidence_manifest_sha256"]),
        bootstrap_campaign_id=context.bootstrap_campaign_id,
        formal_campaign_id=formal_campaign_id,
        ordered_sample_id_sha256=_sample_id_digest(context.manifest_ids),
    )
    return blinding_map


def _validate_blinding_map_contract(
    blinding_map: Mapping[str, Any],
    *,
    assignment: Mapping[str, Any],
    evidence_manifest_sha256: str,
    bootstrap_campaign_id: str,
    formal_campaign_id: str,
    ordered_sample_id_sha256: str,
) -> dict[str, str]:
    """Validate one revealed map identically before publish and at resolution."""

    expected_fields = {
        "schema_version",
        "contract_type",
        "bootstrap_campaign_id",
        "formal_campaign_id",
        "evidence_manifest_sha256",
        "blinding_context_sha256",
        "registered_splits",
        "conditions",
        "blinding_map_sha256",
    }
    expected_context = canonical_json_sha256(
        {
            "schema_version": 1,
            "bootstrap_campaign_id": bootstrap_campaign_id,
            "formal_campaign_id": formal_campaign_id,
            "evidence_manifest_sha256": evidence_manifest_sha256,
            "ordered_sample_id_sha256": ordered_sample_id_sha256,
            "registered_splits": list(SPLIT_KEYS),
        }
    )
    map_binding = assignment.get("blinding_map")
    if (
        set(blinding_map) != expected_fields
        or blinding_map.get("schema_version") != 1
        or blinding_map.get("contract_type")
        != "safa_r9_semigroup_visual_review_blinding_map_v1"
        or blinding_map.get("blinding_map_sha256")
        != _contract_digest(blinding_map, "blinding_map_sha256")
        or not isinstance(map_binding, Mapping)
        or blinding_map.get("blinding_map_sha256") != map_binding.get("contract_sha256")
        or blinding_map.get("bootstrap_campaign_id") != bootstrap_campaign_id
        or blinding_map.get("formal_campaign_id") != formal_campaign_id
        or blinding_map.get("evidence_manifest_sha256") != evidence_manifest_sha256
        or assignment.get("blinding_context_sha256") != expected_context
        or blinding_map.get("blinding_context_sha256") != expected_context
        or blinding_map.get("registered_splits") != list(SPLIT_KEYS)
    ):
        raise CampaignSemigroupClosureError(
            "visual review blinding map contract or context mismatch"
        )
    assignment_conditions = assignment.get("conditions")
    if not isinstance(assignment_conditions, list) or len(assignment_conditions) != len(
        SPLIT_KEYS
    ):
        raise CampaignSemigroupClosureError(
            "visual review blinding map assignment inventory mismatch"
        )
    assignment_by_id: dict[str, Mapping[str, Any]] = {}
    seen_column_ids: set[str] = set()
    for condition in assignment_conditions:
        if not isinstance(condition, Mapping) or set(condition) != {
            "condition_id",
            "column_ids",
            "pages",
        }:
            raise CampaignSemigroupClosureError(
                "visual review blinding map assignment condition mismatch"
            )
        condition_id = str(condition.get("condition_id", ""))
        column_ids = condition.get("column_ids")
        if (
            _BLINDED_CONDITION_ID.fullmatch(condition_id) is None
            or condition_id in assignment_by_id
            or not isinstance(column_ids, list)
            or len(column_ids) != len(_BLINDED_ROLES)
            or len(set(column_ids)) != len(_BLINDED_ROLES)
            or any(
                not isinstance(column_id, str)
                or _BLINDED_COLUMN_ID.fullmatch(column_id) is None
                or column_id in seen_column_ids
                for column_id in column_ids
            )
        ):
            raise CampaignSemigroupClosureError(
                "visual review blinding map assignment columns mismatch"
            )
        assignment_by_id[condition_id] = condition
        seen_column_ids.update(column_ids)
    conditions = blinding_map.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(assignment_by_id):
        raise CampaignSemigroupClosureError(
            "visual review blinding map condition inventory mismatch"
        )
    seen_splits: set[str] = set()
    condition_to_split: dict[str, str] = {}
    for condition_id, assignment_condition in assignment_by_id.items():
        row = conditions[condition_id]
        if not isinstance(row, Mapping) or set(row) != {"split", "columns"}:
            raise CampaignSemigroupClosureError(
                "visual review blinding map condition fields are not canonical"
            )
        split = str(row.get("split", ""))
        columns = row.get("columns")
        if (
            split not in SPLIT_KEYS
            or split in seen_splits
            or not isinstance(columns, list)
            or len(columns) != len(_BLINDED_ROLES)
            or any(
                not isinstance(column, Mapping) or set(column) != {"column_id", "role"}
                for column in columns
            )
            or [column["column_id"] for column in columns]
            != assignment_condition["column_ids"]
            or [column["role"] for column in columns].count("source") != 1
            or [column["role"] for column in columns].count("generated_direct") != 1
            or [column["role"] for column in columns].count("split") != 1
        ):
            raise CampaignSemigroupClosureError(
                "visual review blinding map split/column mapping mismatch"
            )
        seen_splits.add(split)
        condition_to_split[condition_id] = split
    if seen_splits != set(SPLIT_KEYS):
        raise CampaignSemigroupClosureError(
            "visual review blinding map does not cover the registered splits"
        )
    return condition_to_split


def _reject_assignment_decision_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = set(value) & _ASSIGNMENT_DECISION_FIELDS
        if forbidden:
            raise CampaignSemigroupClosureError(
                "visual review assignment must not contain reviewer decision fields"
            )
        for item in value.values():
            _reject_assignment_decision_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_assignment_decision_fields(item)


def _validate_output_root(
    value: Path,
    *,
    repo_root: Path,
    bootstrap_campaign_id: str,
    formal_campaign_id: str,
) -> Path:
    if bootstrap_campaign_id == formal_campaign_id:
        raise CampaignSemigroupClosureError(
            "bootstrap and formal campaign IDs must be distinct"
        )
    base = (repo_root / CLOSURE_BASE).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve(strict=False)
    expected = (base / f"{bootstrap_campaign_id}__for__{formal_campaign_id}").resolve(
        strict=False
    )
    if path != expected:
        raise CampaignSemigroupClosureError(
            "output root must be the direct canonical campaign closure root"
        )
    if path.exists():
        raise CampaignSemigroupClosureError("closure output root already exists")
    formal_campaign_root = (repo_root / CAMPAIGN_BASE / formal_campaign_id).resolve(
        strict=False
    )
    if formal_campaign_root.exists():
        raise CampaignSemigroupClosureError(
            "formal campaign must not be materialized before bootstrap sealing"
        )
    return path


def _validate_distinct_unmaterialized_formal_campaign(
    context: _Context, formal_campaign_id: str
) -> None:
    if formal_campaign_id == context.bootstrap_campaign_id:
        raise CampaignSemigroupClosureError(
            "bootstrap and formal campaign IDs must be distinct"
        )
    formal_campaign_root = (
        context.repo_root / CAMPAIGN_BASE / formal_campaign_id
    ).resolve(strict=False)
    if formal_campaign_root.exists() or formal_campaign_root.is_symlink():
        raise CampaignSemigroupClosureError(
            "formal campaign must not be materialized before bootstrap sealing"
        )
    closure_root = (
        context.repo_root
        / CLOSURE_BASE
        / f"{context.bootstrap_campaign_id}__for__{formal_campaign_id}"
    ).resolve(strict=False)
    if closure_root.exists() or closure_root.is_symlink():
        raise CampaignSemigroupClosureError(
            "formal campaign semigroup closure already exists"
        )


def _publish_terminal_failure(
    *,
    context: _Context,
    evidence: Mapping[str, Any],
    assignment: Mapping[str, Any],
    review: Mapping[str, Any],
    review_source: Path,
    output_root: Path,
    formal_campaign_id: str,
    failure_reason: str,
    numeric_precheck: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one terminal pre-map failure without touching the blinded map."""

    condition_ids = [
        str(condition["condition_id"]) for condition in assignment["conditions"]
    ]
    conditions = review["conditions"]
    failed_condition_ids = [
        condition_id
        for condition_id in condition_ids
        if conditions[condition_id]["passed"] is False
    ]
    if len(condition_ids) != len(SPLIT_KEYS):
        raise CampaignSemigroupClosureError(
            "terminal failure requires exactly three opaque visual conditions"
        )
    if failure_reason == "all_blinded_visual_conditions_failed":
        if failed_condition_ids != condition_ids or numeric_precheck is not None:
            raise CampaignSemigroupClosureError(
                "all-blinded terminal failure decision contract mismatch"
            )
        schema_version = 1
        contract_type = "safa_r9_semigroup_campaign_closure_failure_v1"
    elif failure_reason == "no_quantitative_candidate":
        candidates = (
            numeric_precheck.get("candidates")
            if isinstance(numeric_precheck, Mapping)
            else None
        )
        if (
            failed_condition_ids == condition_ids
            or not isinstance(numeric_precheck, Mapping)
            or numeric_precheck.get("gate_passed") is not False
            or numeric_precheck.get("selected_t_cut") is not None
            or not isinstance(candidates, list)
            or len(candidates) != len(SPLIT_KEYS)
            or any(
                not isinstance(candidate, Mapping)
                or candidate.get("visual_pass") is not True
                or candidate.get("passed") is not False
                for candidate in candidates
            )
        ):
            raise CampaignSemigroupClosureError(
                "quantitative terminal failure precheck contract mismatch"
            )
        schema_version = 2
        contract_type = "safa_r9_semigroup_campaign_closure_failure_v2"
    else:
        raise CampaignSemigroupClosureError("terminal failure reason is not registered")
    source_review = _existing_file(
        review_source, "visual review", repo_root=context.repo_root
    )
    source_assignment = _existing_file(
        context.campaign_root / "preflight" / "visual_review_assignment.json",
        "visual review assignment",
    )
    source_evidence = _existing_file(
        context.campaign_root / "preflight" / "evidence_manifest.json",
        "evidence manifest",
    )
    failure = {
        "schema_version": schema_version,
        "contract_type": contract_type,
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "formal_campaign_id": formal_campaign_id,
        "relationship": "bootstrap_preflight_for_distinct_formal_campaign",
        "terminal_failure": True,
        "gate_passed": False,
        "selected_t_cut": None,
        "reselection_allowed": False,
        "terminal_path_read_map": False,
        "failure_reason": failure_reason,
        "sample_count": SAMPLE_COUNT,
        "failed_condition_ids": failed_condition_ids,
        "blinding_context_sha256": assignment["blinding_context_sha256"],
        "bindings": {
            "runtime_config": {
                "path": _path_text(context.config_path, context.repo_root),
                "file_sha256": _sha256_file(context.config_path),
            },
            "campaign_runtime": {
                "path": _path_text(context.campaign_runtime_path, context.repo_root),
                "file_sha256": _sha256_file(context.campaign_runtime_path),
                "contract_sha256": context.config["r9_campaign_runtime_sha256"],
            },
            "evidence_manifest": {
                "path": _path_text(source_evidence, context.repo_root),
                "file_sha256": _sha256_file(source_evidence),
                "contract_sha256": evidence["evidence_manifest_sha256"],
            },
            "visual_review_assignment": {
                "path": _path_text(source_assignment, context.repo_root),
                "file_sha256": _sha256_file(source_assignment),
                "contract_sha256": assignment["visual_review_assignment_sha256"],
            },
            "visual_review_source": {
                "path": _path_text(source_review, context.repo_root),
                "file_sha256": _sha256_file(source_review),
                "contract_sha256": review["visual_review_sha256"],
            },
        },
    }
    if numeric_precheck is not None:
        failure["numeric_precheck"] = dict(numeric_precheck)
    failure["failure_contract_sha256"] = _contract_digest(
        failure, "failure_contract_sha256"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(mode=0o755, exist_ok=False)
    failure_path = output_root / "closure_failure.json"
    file_sha256 = _write_exclusive_json(failure_path, failure, canonical=True)
    return {
        "schema_version": schema_version,
        "gate_passed": False,
        "terminal_failure": True,
        "selected_t_cut": None,
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "formal_campaign_id": formal_campaign_id,
        "output_root": _path_text(output_root, context.repo_root),
        "closure_failure": {
            "path": _path_text(failure_path, context.repo_root),
            "file_sha256": file_sha256,
            "contract_sha256": failure["failure_contract_sha256"],
        },
    }


def _validate_terminal_failure(
    closure_root: Path,
    *,
    formal_campaign_id: str,
    repo_root: Path,
) -> dict[str, Any]:
    failure_path = _existing_file(
        closure_root / "closure_failure.json", "terminal closure failure"
    )
    if stat.S_IMODE(failure_path.stat().st_mode) & 0o222:
        raise CampaignSemigroupClosureError(
            "terminal closure failure must be immutable"
        )
    failure = _read_json(failure_path, "terminal closure failure")
    common_fields = {
        "schema_version",
        "contract_type",
        "bootstrap_campaign_id",
        "formal_campaign_id",
        "relationship",
        "terminal_failure",
        "gate_passed",
        "selected_t_cut",
        "reselection_allowed",
        "terminal_path_read_map",
        "failure_reason",
        "sample_count",
        "failed_condition_ids",
        "blinding_context_sha256",
        "bindings",
        "failure_contract_sha256",
    }
    schema_version = failure.get("schema_version")
    if schema_version == 1:
        expected_fields = common_fields
        expected_contract_type = "safa_r9_semigroup_campaign_closure_failure_v1"
        expected_reason = "all_blinded_visual_conditions_failed"
    elif schema_version == 2:
        expected_fields = common_fields | {"numeric_precheck"}
        expected_contract_type = "safa_r9_semigroup_campaign_closure_failure_v2"
        expected_reason = "no_quantitative_candidate"
    else:
        raise CampaignSemigroupClosureError(
            "terminal closure failure schema version is invalid"
        )
    if (
        set(failure) != expected_fields
        or failure.get("contract_type") != expected_contract_type
        or failure.get("formal_campaign_id") != formal_campaign_id
        or failure.get("relationship")
        != "bootstrap_preflight_for_distinct_formal_campaign"
        or failure.get("terminal_failure") is not True
        or failure.get("gate_passed") is not False
        or failure.get("selected_t_cut") is not None
        or failure.get("reselection_allowed") is not False
        or failure.get("terminal_path_read_map") is not False
        or failure.get("failure_reason") != expected_reason
        or failure.get("sample_count") != SAMPLE_COUNT
        or failure.get("failure_contract_sha256")
        != _contract_digest(failure, "failure_contract_sha256")
    ):
        raise CampaignSemigroupClosureError(
            "terminal closure failure contract is invalid"
        )
    if schema_version == 2:
        numeric_precheck = failure["numeric_precheck"]
        candidates = (
            numeric_precheck.get("candidates")
            if isinstance(numeric_precheck, Mapping)
            else None
        )
        if (
            not isinstance(numeric_precheck, Mapping)
            or numeric_precheck.get("gate_passed") is not False
            or numeric_precheck.get("selected_t_cut") is not None
            or not isinstance(candidates, list)
            or len(candidates) != len(SPLIT_KEYS)
            or any(
                not isinstance(candidate, Mapping)
                or candidate.get("visual_pass") is not True
                or candidate.get("passed") is not False
                for candidate in candidates
            )
        ):
            raise CampaignSemigroupClosureError(
                "terminal closure quantitative precheck is invalid"
            )
    bootstrap_id = _require_campaign_id(
        str(failure.get("bootstrap_campaign_id", "")), "bootstrap campaign ID"
    )
    if (
        bootstrap_id == formal_campaign_id
        or closure_root.name != f"{bootstrap_id}__for__{formal_campaign_id}"
    ):
        raise CampaignSemigroupClosureError(
            "terminal closure failure campaign relationship mismatch"
        )
    bindings = failure.get("bindings")
    expected_binding_fields = {
        "runtime_config",
        "campaign_runtime",
        "evidence_manifest",
        "visual_review_assignment",
        "visual_review_source",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_binding_fields:
        raise CampaignSemigroupClosureError(
            "terminal closure failure bindings are invalid"
        )

    campaign_root = (repo_root / CAMPAIGN_BASE / bootstrap_id).resolve()

    def bound_file(
        name: str,
        *,
        expected_path: Path | None = None,
        contract_field: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        binding = bindings[name]
        expected = {"path", "file_sha256"}
        if contract_field is not None:
            expected.add("contract_sha256")
        if not isinstance(binding, Mapping) or set(binding) != expected:
            raise CampaignSemigroupClosureError(
                f"terminal closure failure binding is invalid: {name}"
            )
        path_text = str(binding.get("path", ""))
        if Path(path_text).is_absolute():
            raise CampaignSemigroupClosureError(
                f"terminal closure failure path must be repository-relative: {name}"
            )
        path = _resolve_config_path(
            repo_root, path_text, f"terminal closure failure {name}"
        )
        if expected_path is not None and path != expected_path.resolve():
            raise CampaignSemigroupClosureError(
                f"terminal closure failure path mismatch: {name}"
            )
        if _sha256_file(path) != _require_sha(
            binding.get("file_sha256"),
            f"terminal closure failure {name} file SHA256",
        ):
            raise CampaignSemigroupClosureError(
                f"terminal closure failure file SHA256 mismatch: {name}"
            )
        payload = _read_json(path, f"terminal closure failure {name}")
        if contract_field is not None and (
            payload.get(contract_field) != binding.get("contract_sha256")
            or payload.get(contract_field) != _contract_digest(payload, contract_field)
        ):
            raise CampaignSemigroupClosureError(
                f"terminal closure failure contract mismatch: {name}"
            )
        return path, payload

    runtime_binding = bindings["runtime_config"]
    if not isinstance(runtime_binding, Mapping) or set(runtime_binding) != {
        "path",
        "file_sha256",
    }:
        raise CampaignSemigroupClosureError(
            "terminal closure failure runtime config binding is invalid"
        )
    runtime_text = str(runtime_binding.get("path", ""))
    if Path(runtime_text).is_absolute():
        raise CampaignSemigroupClosureError(
            "terminal closure failure runtime config path must be repository-relative"
        )
    runtime_path = _resolve_config_path(
        repo_root, runtime_text, "terminal closure failure runtime config"
    )
    _relative_to(
        runtime_path,
        campaign_root / "runtime_configs",
        "terminal closure failure runtime config",
    )
    if _sha256_file(runtime_path) != _require_sha(
        runtime_binding.get("file_sha256"),
        "terminal closure failure runtime config file SHA256",
    ):
        raise CampaignSemigroupClosureError(
            "terminal closure failure runtime config SHA256 mismatch"
        )
    _, runtime = bound_file(
        "campaign_runtime",
        expected_path=campaign_root / "campaign_runtime.json",
        contract_field="campaign_runtime_sha256",
    )
    _, evidence = bound_file(
        "evidence_manifest",
        expected_path=campaign_root / "preflight" / "evidence_manifest.json",
        contract_field="evidence_manifest_sha256",
    )
    _, assignment = bound_file(
        "visual_review_assignment",
        expected_path=campaign_root / "preflight" / "visual_review_assignment.json",
        contract_field="visual_review_assignment_sha256",
    )
    _, review = bound_file(
        "visual_review_source",
        expected_path=campaign_root / "preflight" / "visual_review.json",
        contract_field="visual_review_sha256",
    )
    if (
        runtime.get("campaign_id") != bootstrap_id
        or assignment.get("bootstrap_campaign_id") != bootstrap_id
        or review.get("bootstrap_campaign_id") != bootstrap_id
        or assignment.get("formal_campaign_id") != formal_campaign_id
        or review.get("formal_campaign_id") != formal_campaign_id
        or assignment.get("blinding_context_sha256")
        != failure.get("blinding_context_sha256")
        or review.get("evidence_manifest_sha256")
        != evidence.get("evidence_manifest_sha256")
        or review.get("visual_review_assignment_sha256")
        != assignment.get("visual_review_assignment_sha256")
    ):
        raise CampaignSemigroupClosureError(
            "terminal closure failure non-map chain mismatch"
        )
    assignment_conditions = assignment.get("conditions")
    review_conditions = review.get("conditions")
    failed_ids = failure.get("failed_condition_ids")
    if (
        not isinstance(assignment_conditions, list)
        or len(assignment_conditions) != len(SPLIT_KEYS)
        or not isinstance(review_conditions, Mapping)
        or not isinstance(failed_ids, list)
    ):
        raise CampaignSemigroupClosureError(
            "terminal closure failure condition inventory mismatch"
        )
    condition_ids = [
        str(condition.get("condition_id", ""))
        for condition in assignment_conditions
        if isinstance(condition, Mapping)
    ]
    if (
        len(condition_ids) != len(SPLIT_KEYS)
        or len(set(condition_ids)) != len(SPLIT_KEYS)
        or set(review_conditions) != set(condition_ids)
    ):
        raise CampaignSemigroupClosureError(
            "terminal closure failure opaque condition inventory mismatch"
        )
    calculated_failed_ids: list[str] = []
    manifest_ids = set(review.get("reviewed_sample_ids", ()))
    for condition_id in condition_ids:
        decision = review_conditions[condition_id]
        if not isinstance(decision, Mapping) or set(decision) != {
            "passed",
            "severe_count",
            "severe_sample_ids",
        }:
            raise CampaignSemigroupClosureError(
                "terminal closure failure reviewer decision is invalid"
            )
        severe_ids = decision.get("severe_sample_ids")
        severe_count = decision.get("severe_count")
        passed = decision.get("passed")
        if (
            not isinstance(severe_ids, list)
            or len(set(severe_ids)) != len(severe_ids)
            or not set(severe_ids) <= manifest_ids
            or isinstance(severe_count, bool)
            or not isinstance(severe_count, int)
            or severe_count != len(severe_ids)
            or not isinstance(passed, bool)
            or passed != (severe_count == 0)
        ):
            raise CampaignSemigroupClosureError(
                "terminal closure failure reviewer decision contract mismatch"
            )
        if passed is False:
            calculated_failed_ids.append(condition_id)
    if (
        failed_ids != calculated_failed_ids
        or (schema_version == 1 and failed_ids != condition_ids)
        or (schema_version == 2 and failed_ids == condition_ids)
    ):
        raise CampaignSemigroupClosureError(
            "terminal closure failure failed-condition binding mismatch"
        )
    return failure


def _publish_closure(
    *,
    context: _Context,
    evidence: Mapping[str, Any],
    assignment: Mapping[str, Any],
    blinding_map: Mapping[str, Any],
    review: Mapping[str, Any],
    review_source: Path,
    report: Mapping[str, Any],
    output_root: Path,
    formal_campaign_id: str,
    recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preflight = canonical_r9_semigroup_preflight_payload(context.config)
    arm_sha256 = canonical_r9_arm_config_digest(context.config)
    effective = {**context.config, "arm_config_sha256": arm_sha256}
    executed = evidence["executed_config_sha256"]
    selected_t_cut = _finite_open_unit(report.get("selected_t_cut"), "selected_t_cut")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(mode=0o755, exist_ok=False)
    try:
        _write_exclusive_json(
            output_root / "preflight_contract.json", preflight, canonical=True
        )
        _write_exclusive_json(
            output_root / "effective_config.json", effective, canonical=True
        )
        _write_exclusive_json(
            output_root / "executed_config.json",
            _read_json(
                context.shard_dirs[0] / "generation_result.json",
                "generation result",
            )["config"],
            canonical=True,
        )
        _write_exclusive_json(
            output_root / "evidence_manifest.json", evidence, canonical=False
        )
        _write_exclusive_json(
            output_root / "visual_review_assignment.json",
            assignment,
            canonical=False,
        )
        _write_exclusive_json(
            output_root / "visual_review_blinding_map.json",
            blinding_map,
            canonical=False,
        )
        _write_exclusive_json(
            output_root / "visual_review.json", review, canonical=False
        )
        _write_exclusive_json(
            output_root / "semigroup_report.json", report, canonical=False
        )
        if recovery is not None:
            _write_exclusive_json(
                output_root / "recovery_policy.json",
                {
                    **R9_SEMIGROUP_RECOVERY_POLICY,
                    "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                },
                canonical=True,
            )
        report_sha256 = _sha256_file(output_root / "semigroup_report.json")
        if (
            _sha256_file(output_root / "preflight_contract.json")
            != context.config["semigroup_preflight_contract_sha256"]
        ):
            raise CampaignSemigroupClosureError(
                "published preflight contract digest mismatch"
            )
        if (
            _sha256_file(output_root / "effective_config.json")
            != report["effective_config_sha256"]
        ):
            raise CampaignSemigroupClosureError(
                "published effective config digest mismatch"
            )
        if _sha256_file(output_root / "executed_config.json") != executed:
            raise CampaignSemigroupClosureError(
                "published executed config digest mismatch"
            )
        preflight_path = _path_text(
            output_root / "preflight_contract.json", context.repo_root
        )
        report_path = _path_text(
            output_root / "semigroup_report.json", context.repo_root
        )
        gate_path = _path_text(output_root / "gate_contract.json", context.repo_root)
        schedule_path = _path_text(
            output_root / "locked_schedule_manifest.json", context.repo_root
        )
        t_cut = selected_t_cut
        guided = [1.0 - index * (1.0 - t_cut) / 3.0 for index in range(4)]
        guided[-1] = t_cut
        schedule = {
            "schema_version": R9_LOCKED_SCHEDULE_SCHEMA_VERSION,
            "gate_passed": True,
            "checkpoint_sha256": context.config["checkpoint_sha256"],
            "semigroup_report": report_path,
            "semigroup_report_sha256": report_sha256,
            "semigroup_sample_id_manifest": _path_text(
                context.manifest_path, context.repo_root
            ),
            "semigroup_sample_id_manifest_sha256": context.config[
                "sample_id_manifest_sha256"
            ],
            "semigroup_preflight_contract": preflight_path,
            "semigroup_preflight_contract_sha256": _sha256_file(
                output_root / "preflight_contract.json"
            ),
            "t_cut": t_cut,
            "guided_steps": 3,
            "guided_times": guided,
            "unguided_tail_intervals": 2,
            "unguided_times": [t_cut, t_cut / 2.0, 0.0],
            "selection_rule": (
                R9_SEMIGROUP_RECOVERY_SELECTION_RULE
                if recovery is not None
                else R9_SELECTION_RULE
            ),
        }
        if recovery is not None:
            schedule.update(
                {
                    "recovery_policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                    "numerical_metrics_role": "report_only",
                }
            )
        schedule["schedule_contract_sha256"] = canonical_r9_schedule_contract_sha256(
            schedule
        )
        gate = build_r9_semigroup_gate_contract(
            context.config,
            effective_config_sha256=str(report["effective_config_sha256"]),
            semigroup_report_sha256=report_sha256,
            gate_passed=True,
            selected_t_cut=t_cut,
            schedule_contract_sha256=schedule["schedule_contract_sha256"],
        )
        if recovery is not None:
            gate.update(
                {
                    "schema_version": 2,
                    "contract_type": "safa_r9_semigroup_recovery_gate_v2",
                    "recovery_policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                    "numerical_metrics_role": "report_only",
                    "selection_rule": R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
                }
            )
            gate["gate_contract_sha256"] = _contract_digest(
                gate, "gate_contract_sha256"
            )
        else:
            validate_r9_semigroup_gate_contract(gate, context.config)
        _write_exclusive_json(output_root / "gate_contract.json", gate, canonical=True)
        schedule["r9_semigroup_gate_contract"] = gate_path
        schedule["r9_semigroup_gate_contract_sha256"] = _sha256_file(
            output_root / "gate_contract.json"
        )
        _write_exclusive_json(
            output_root / "locked_schedule_manifest.json", schedule, canonical=False
        )
        validation_config = {
            **context.config,
            "semigroup_report": report_path,
            "semigroup_sample_id_manifest": _path_text(
                context.manifest_path, context.repo_root
            ),
            "semigroup_sample_id_manifest_sha256": context.config[
                "sample_id_manifest_sha256"
            ],
            "semigroup_preflight_contract": preflight_path,
            "r9_semigroup_gate_contract": gate_path,
            "r9_semigroup_gate_contract_sha256": schedule[
                "r9_semigroup_gate_contract_sha256"
            ],
        }
        with _working_directory(context.repo_root):
            validate_r9_locked_schedule_bindings(validation_config, schedule)
        visual_review_source = _existing_file(
            review_source, "visual review", repo_root=context.repo_root
        )
        visual_review_source_path = _path_text(visual_review_source, context.repo_root)
        if Path(visual_review_source_path).is_absolute():
            raise CampaignSemigroupClosureError(
                "source visual review path must be repository-relative"
            )
        closure_artifacts = _CLOSURE_ARTIFACTS + (
            ("recovery_policy.json",) if recovery is not None else ()
        )
        artifacts = {
            name.removesuffix(".json"): {
                "path": _path_text(output_root / name, context.repo_root),
                "sha256": _sha256_file(output_root / name),
            }
            for name in closure_artifacts
            if name != "closure_seal.json"
        }
        seal = {
            "schema_version": 1,
            "contract_type": "safa_r9_semigroup_campaign_closure_v1",
            "bootstrap_campaign": {
                "campaign_id": context.bootstrap_campaign_id,
                "campaign_runtime_sha256": context.config["r9_campaign_runtime_sha256"],
                "campaign_runtime": {
                    "path": _path_text(
                        context.campaign_runtime_path, context.repo_root
                    ),
                    "file_sha256": _sha256_file(context.campaign_runtime_path),
                },
                "runtime_config": {
                    "path": _path_text(context.config_path, context.repo_root),
                    "sha256": _sha256_file(context.config_path),
                },
                "shard_root": _path_text(context.shard_root, context.repo_root),
            },
            "formal_campaign": {
                "campaign_id": formal_campaign_id,
                "relationship": "bootstrap_preflight_for_distinct_formal_campaign",
                "runtime_state_at_seal": "not_materialized",
            },
            "gate_passed": True,
            "selected_t_cut": t_cut,
            "reselection_allowed": False,
            "bindings": {
                "checkpoint_sha256": context.config["checkpoint_sha256"],
                "determinism_policy_sha256": context.config[
                    "determinism_policy_sha256"
                ],
                "attention_backend": context.config["attention_backend"],
                "sample_manifest_sha256": context.config["sample_id_manifest_sha256"],
                "preflight_contract_sha256": _sha256_file(
                    output_root / "preflight_contract.json"
                ),
                "effective_config_sha256": report["effective_config_sha256"],
                "executed_config_sha256": executed,
                "evidence_manifest_sha256": evidence["evidence_manifest_sha256"],
                "visual_review_assignment_sha256": assignment[
                    "visual_review_assignment_sha256"
                ],
                "visual_review_blinding_map_sha256": blinding_map[
                    "blinding_map_sha256"
                ],
                "visual_review_sha256": review["visual_review_sha256"],
                "visual_review_source_path": visual_review_source_path,
                "visual_review_source_sha256": _sha256_file(visual_review_source),
                "visual_review_published_copy_sha256": artifacts["visual_review"][
                    "sha256"
                ],
                "semigroup_report_sha256": report_sha256,
                "gate_contract_sha256": gate["gate_contract_sha256"],
                "gate_contract_file_sha256": _sha256_file(
                    output_root / "gate_contract.json"
                ),
                "schedule_contract_sha256": schedule["schedule_contract_sha256"],
                "schedule_manifest_file_sha256": _sha256_file(
                    output_root / "locked_schedule_manifest.json"
                ),
            },
            "artifacts": artifacts,
            "formal_campaign_required_bindings": {
                "closure_seal": _path_text(
                    output_root / "closure_seal.json", context.repo_root
                ),
                "locked_schedule_manifest": schedule_path,
                "gate_contract": gate_path,
            },
        }
        if recovery is not None:
            source_failure_path = _existing_file(
                recovery["source_terminal_failure_path"],
                "source terminal closure failure",
                repo_root=context.repo_root,
            )
            source_failure_payload = _read_json(
                source_failure_path, "source terminal closure failure"
            )
            seal.update(
                {
                    "schema_version": 2,
                    "contract_type": "safa_r9_semigroup_campaign_closure_v2",
                    "policy_campaign": {
                        "campaign_id": recovery["policy_campaign_id"],
                        "relationship": "user_authorized_recovery_of_terminal_preflight",
                    },
                    "source_review": {
                        "state_at_recovery": "previously_revealed",
                        "formal_campaign_id": recovery["source_formal_campaign_id"],
                        "terminal_failure": {
                            "path": _path_text(source_failure_path, context.repo_root),
                            "file_sha256": _sha256_file(source_failure_path),
                            "contract_sha256": source_failure_payload[
                                "failure_contract_sha256"
                            ],
                        },
                    },
                    "policy": {
                        **R9_SEMIGROUP_RECOVERY_POLICY,
                        "policy_version": R9_SEMIGROUP_RECOVERY_POLICY_VERSION,
                        "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                    },
                    "authorization": {
                        "authorization_id": recovery["authorization_id"],
                        "scope": "r9_semigroup_report_only_visual_limit_1_lock_025",
                    },
                }
            )
            seal["bindings"].update(
                {
                    "recovery_policy_file_sha256": artifacts["recovery_policy"][
                        "sha256"
                    ],
                    "recovery_policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                    "source_terminal_failure_file_sha256": _sha256_file(
                        source_failure_path
                    ),
                }
            )
        seal["closure_seal_sha256"] = _contract_digest(seal, "closure_seal_sha256")
        _write_exclusive_json(output_root / "closure_seal.json", seal, canonical=True)
    except Exception:
        # A partially published root is intentionally left visible and cannot be reused.
        raise
    result = {
        "schema_version": 2 if recovery is not None else 1,
        "gate_passed": True,
        "selected_t_cut": selected_t_cut,
        "bootstrap_campaign_id": context.bootstrap_campaign_id,
        "formal_campaign_id": formal_campaign_id,
        "output_root": _path_text(output_root, context.repo_root),
        "closure_seal_sha256": seal["closure_seal_sha256"],
    }
    if recovery is not None:
        result.update(
            {
                "policy_campaign_id": recovery["policy_campaign_id"],
                "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
            }
        )
    return result


def _validate_published_closure(
    closure_root: Path,
    *,
    formal_campaign_id: str,
    repo_root: Path,
) -> dict[str, Any]:
    seal_path = _existing_file(
        closure_root / "closure_seal.json", "campaign closure seal"
    )
    seal = _read_json(seal_path, "campaign closure seal")
    common_seal_fields = {
        "schema_version",
        "contract_type",
        "bootstrap_campaign",
        "formal_campaign",
        "gate_passed",
        "selected_t_cut",
        "reselection_allowed",
        "bindings",
        "artifacts",
        "formal_campaign_required_bindings",
        "closure_seal_sha256",
    }
    schema_version = seal.get("schema_version")
    if schema_version == 1:
        expected_seal_fields = common_seal_fields
        expected_contract_type = "safa_r9_semigroup_campaign_closure_v1"
    elif schema_version == 2:
        expected_seal_fields = common_seal_fields | {
            "policy_campaign",
            "source_review",
            "policy",
            "authorization",
        }
        expected_contract_type = "safa_r9_semigroup_campaign_closure_v2"
    else:
        expected_seal_fields = set()
        expected_contract_type = None
    if (
        set(seal) != expected_seal_fields
        or seal.get("contract_type") != expected_contract_type
        or seal.get("gate_passed") is not True
        or seal.get("reselection_allowed") is not False
    ):
        raise CampaignSemigroupClosureError(
            "campaign closure seal fields are not canonical"
        )
    seal_contract_sha256 = _require_sha(
        seal.get("closure_seal_sha256"), "closure seal SHA256"
    )
    if _contract_digest(seal, "closure_seal_sha256") != seal_contract_sha256:
        raise CampaignSemigroupClosureError("campaign closure seal digest mismatch")
    formal = seal.get("formal_campaign")
    if not isinstance(formal, Mapping) or dict(formal) != {
        "campaign_id": formal_campaign_id,
        "relationship": "bootstrap_preflight_for_distinct_formal_campaign",
        "runtime_state_at_seal": "not_materialized",
    }:
        raise CampaignSemigroupClosureError(
            "campaign closure seal formal relationship mismatch"
        )
    bootstrap = seal.get("bootstrap_campaign")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
        "campaign_id",
        "campaign_runtime_sha256",
        "campaign_runtime",
        "runtime_config",
        "shard_root",
    }:
        raise CampaignSemigroupClosureError(
            "campaign closure seal bootstrap fields are not canonical"
        )
    bootstrap_id = _require_campaign_id(
        str(bootstrap.get("campaign_id", "")), "bootstrap campaign ID"
    )
    if schema_version == 2:
        policy_campaign = seal.get("policy_campaign")
        if (
            not isinstance(policy_campaign, Mapping)
            or set(policy_campaign) != {"campaign_id", "relationship"}
            or policy_campaign.get("relationship")
            != "user_authorized_recovery_of_terminal_preflight"
        ):
            raise CampaignSemigroupClosureError(
                "campaign closure recovery policy relationship mismatch"
            )
        closure_prefix_id = _require_campaign_id(
            str(policy_campaign.get("campaign_id", "")), "policy campaign ID"
        )
        if closure_prefix_id in {bootstrap_id, formal_campaign_id}:
            raise CampaignSemigroupClosureError(
                "campaign closure policy/bootstrap/formal IDs are not distinct"
            )
    else:
        closure_prefix_id = bootstrap_id
    if (
        bootstrap_id == formal_campaign_id
        or closure_root.name != f"{closure_prefix_id}__for__{formal_campaign_id}"
    ):
        raise CampaignSemigroupClosureError(
            "campaign closure directory/CID relationship mismatch"
        )
    runtime_config = bootstrap.get("runtime_config")
    campaign_runtime = bootstrap.get("campaign_runtime")
    if not isinstance(runtime_config, Mapping) or set(runtime_config) != {
        "path",
        "sha256",
    }:
        raise CampaignSemigroupClosureError(
            "campaign closure bootstrap runtime config binding mismatch"
        )
    if not isinstance(campaign_runtime, Mapping) or set(campaign_runtime) != {
        "path",
        "file_sha256",
    }:
        raise CampaignSemigroupClosureError(
            "campaign closure bootstrap campaign runtime binding mismatch"
        )
    runtime_config_path = _resolve_config_path(
        repo_root,
        runtime_config.get("path"),
        "sealed bootstrap runtime config",
    )
    campaign_runtime_path = _resolve_config_path(
        repo_root,
        campaign_runtime.get("path"),
        "sealed bootstrap campaign runtime",
    )
    if _sha256_file(runtime_config_path) != _require_sha(
        runtime_config.get("sha256"), "bootstrap runtime config SHA256"
    ):
        raise CampaignSemigroupClosureError(
            "sealed bootstrap runtime config SHA256 mismatch"
        )
    if _sha256_file(campaign_runtime_path) != _require_sha(
        campaign_runtime.get("file_sha256"),
        "bootstrap campaign runtime file SHA256",
    ):
        raise CampaignSemigroupClosureError(
            "sealed bootstrap campaign runtime file SHA256 mismatch"
        )
    bootstrap_runtime_payload = _read_json(
        campaign_runtime_path, "sealed bootstrap campaign runtime"
    )
    if bootstrap_runtime_payload.get("campaign_runtime_sha256") != _require_sha(
        bootstrap.get("campaign_runtime_sha256"),
        "bootstrap campaign runtime SHA256",
    ) or _contract_digest(
        bootstrap_runtime_payload, "campaign_runtime_sha256"
    ) != bootstrap.get("campaign_runtime_sha256"):
        raise CampaignSemigroupClosureError(
            "sealed bootstrap campaign runtime contract mismatch"
        )
    shard_root = Path(str(bootstrap.get("shard_root", "")))
    if not shard_root.is_absolute():
        shard_root = repo_root / shard_root
    shard_root = shard_root.resolve()
    if shard_root.is_symlink() or not shard_root.is_dir():
        raise CampaignSemigroupClosureError(
            "sealed bootstrap shard root is unavailable"
        )

    expected_artifact_files = dict(_PUBLISHED_ARTIFACT_FILES)
    if schema_version == 2:
        expected_artifact_files["recovery_policy"] = "recovery_policy.json"
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        expected_artifact_files
    ):
        raise CampaignSemigroupClosureError(
            "campaign closure artifact inventory is not canonical"
        )
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for artifact_name, filename in expected_artifact_files.items():
        binding = artifacts[artifact_name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise CampaignSemigroupClosureError(
                f"campaign closure artifact binding is invalid: {artifact_name}"
            )
        path = _resolve_config_path(
            repo_root,
            binding.get("path"),
            f"campaign closure artifact {artifact_name}",
        )
        if path != (closure_root / filename).resolve():
            raise CampaignSemigroupClosureError(
                f"campaign closure artifact path mismatch: {artifact_name}"
            )
        digest = _require_sha(
            binding.get("sha256"), f"campaign closure {artifact_name} SHA256"
        )
        if _sha256_file(path) != digest:
            raise CampaignSemigroupClosureError(
                f"campaign closure artifact SHA256 mismatch: {artifact_name}"
            )
        artifact_paths[artifact_name] = path
        artifact_hashes[artifact_name] = digest

    bindings = seal.get("bindings")
    expected_binding_fields = {
        "checkpoint_sha256",
        "determinism_policy_sha256",
        "attention_backend",
        "sample_manifest_sha256",
        "preflight_contract_sha256",
        "effective_config_sha256",
        "executed_config_sha256",
        "evidence_manifest_sha256",
        "visual_review_assignment_sha256",
        "visual_review_blinding_map_sha256",
        "visual_review_sha256",
        "visual_review_source_path",
        "visual_review_source_sha256",
        "visual_review_published_copy_sha256",
        "semigroup_report_sha256",
        "gate_contract_sha256",
        "gate_contract_file_sha256",
        "schedule_contract_sha256",
        "schedule_manifest_file_sha256",
    }
    if schema_version == 2:
        expected_binding_fields |= {
            "recovery_policy_file_sha256",
            "recovery_policy_sha256",
            "source_terminal_failure_file_sha256",
        }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_binding_fields:
        raise CampaignSemigroupClosureError(
            "campaign closure seal bindings are not canonical"
        )
    preflight = _read_json(
        artifact_paths["preflight_contract"], "sealed preflight contract"
    )
    effective = _read_json(
        artifact_paths["effective_config"], "sealed effective config"
    )
    executed = _read_json(artifact_paths["executed_config"], "sealed executed config")
    evidence = _read_json(
        artifact_paths["evidence_manifest"], "sealed evidence manifest"
    )
    assignment = _read_json(
        artifact_paths["visual_review_assignment"],
        "sealed visual review assignment",
    )
    blinding_map = _read_json(
        artifact_paths["visual_review_blinding_map"],
        "sealed visual review blinding map",
    )
    review = _read_json(artifact_paths["visual_review"], "sealed visual review")
    report = _read_json(artifact_paths["semigroup_report"], "sealed semigroup report")
    gate = _read_json(artifact_paths["gate_contract"], "sealed gate contract")
    schedule = _read_json(
        artifact_paths["locked_schedule_manifest"], "sealed schedule manifest"
    )
    if schema_version == 2:
        candidates = report.get("candidates")
        visual_assessment = report.get("visual_assessment")
        if (
            report.get("schema_version") != 2
            or report.get("contract_type") != "safa_r9_semigroup_recovery_report_v2"
            or report.get("policy_version") != R9_SEMIGROUP_RECOVERY_POLICY_VERSION
            or report.get("policy_sha256") != R9_SEMIGROUP_RECOVERY_POLICY_SHA256
            or report.get("numerical_metrics_role") != "report_only"
            or report.get("selection_rule") != R9_SEMIGROUP_RECOVERY_SELECTION_RULE
            or report.get("selected_t_cut") != 0.25
            or not isinstance(candidates, list)
            or not any(
                isinstance(candidate, Mapping)
                and candidate.get("t_cut") == 0.25
                and candidate.get("passed") is True
                and isinstance(candidate.get("numeric_threshold_pass"), bool)
                for candidate in candidates
            )
            or not isinstance(visual_assessment, Mapping)
            or set(visual_assessment) != set(SPLIT_KEYS)
            or any(
                not isinstance(row, Mapping)
                or row.get("passed") is not True
                or not isinstance(row.get("severe_count"), int)
                or row.get("severe_count")
                > int(R9_SEMIGROUP_RECOVERY_POLICY["visual_severe_limit_per_split"])
                for row in visual_assessment.values()
            )
            or schedule.get("selection_rule") != R9_SEMIGROUP_RECOVERY_SELECTION_RULE
            or schedule.get("recovery_policy_sha256")
            != R9_SEMIGROUP_RECOVERY_POLICY_SHA256
            or schedule.get("numerical_metrics_role") != "report_only"
            or schedule.get("t_cut") != 0.25
            or gate.get("schema_version") != 2
            or gate.get("contract_type") != "safa_r9_semigroup_recovery_gate_v2"
            or gate.get("recovery_policy_sha256") != R9_SEMIGROUP_RECOVERY_POLICY_SHA256
            or gate.get("numerical_metrics_role") != "report_only"
            or gate.get("selection_rule") != R9_SEMIGROUP_RECOVERY_SELECTION_RULE
        ):
            raise CampaignSemigroupClosureError(
                "sealed recovery report/schedule policy semantics mismatch"
            )
    expected_bindings = {
        "preflight_contract_sha256": artifact_hashes["preflight_contract"],
        "effective_config_sha256": canonical_json_sha256(effective),
        "executed_config_sha256": canonical_json_sha256(executed),
        "evidence_manifest_sha256": evidence.get("evidence_manifest_sha256"),
        "visual_review_assignment_sha256": assignment.get(
            "visual_review_assignment_sha256"
        ),
        "visual_review_blinding_map_sha256": blinding_map.get("blinding_map_sha256"),
        "visual_review_sha256": review.get("visual_review_sha256"),
        "visual_review_published_copy_sha256": artifact_hashes["visual_review"],
        "semigroup_report_sha256": artifact_hashes["semigroup_report"],
        "gate_contract_sha256": gate.get("gate_contract_sha256"),
        "gate_contract_file_sha256": artifact_hashes["gate_contract"],
        "schedule_contract_sha256": schedule.get("schedule_contract_sha256"),
        "schedule_manifest_file_sha256": artifact_hashes["locked_schedule_manifest"],
    }
    if schema_version == 2:
        expected_bindings.update(
            {
                "recovery_policy_file_sha256": artifact_hashes["recovery_policy"],
                "recovery_policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                "source_terminal_failure_file_sha256": seal["source_review"][
                    "terminal_failure"
                ]["file_sha256"],
            }
        )
    for field, expected in expected_bindings.items():
        if bindings.get(field) != expected:
            raise CampaignSemigroupClosureError(
                f"campaign closure seal binding mismatch: {field}"
            )
    visual_chain_formal_id = formal_campaign_id
    if schema_version == 2:
        expected_policy = {
            **R9_SEMIGROUP_RECOVERY_POLICY,
            "policy_version": R9_SEMIGROUP_RECOVERY_POLICY_VERSION,
            "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
        }
        published_policy = _read_json(
            artifact_paths["recovery_policy"], "sealed recovery policy"
        )
        if (
            published_policy
            != {
                **R9_SEMIGROUP_RECOVERY_POLICY,
                "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
            }
            or seal.get("policy") != expected_policy
        ):
            raise CampaignSemigroupClosureError(
                "campaign closure recovery policy binding mismatch"
            )
        if seal.get("authorization") != {
            "authorization_id": R9_SEMIGROUP_RECOVERY_AUTHORIZATION_ID,
            "scope": "r9_semigroup_report_only_visual_limit_1_lock_025",
        }:
            raise CampaignSemigroupClosureError(
                "campaign closure recovery authorization mismatch"
            )
        source_review = seal.get("source_review")
        if (
            not isinstance(source_review, Mapping)
            or set(source_review)
            != {"state_at_recovery", "formal_campaign_id", "terminal_failure"}
            or source_review.get("state_at_recovery") != "previously_revealed"
        ):
            raise CampaignSemigroupClosureError(
                "campaign closure source review recovery state mismatch"
            )
        visual_chain_formal_id = _require_campaign_id(
            str(source_review.get("formal_campaign_id", "")),
            "source formal campaign ID",
        )
        terminal_binding = source_review.get("terminal_failure")
        if not isinstance(terminal_binding, Mapping) or set(terminal_binding) != {
            "path",
            "file_sha256",
            "contract_sha256",
        }:
            raise CampaignSemigroupClosureError(
                "campaign closure source terminal failure binding mismatch"
            )
        terminal_path = _resolve_config_path(
            repo_root,
            terminal_binding.get("path"),
            "source terminal closure failure",
        )
        if (
            terminal_path.name != "closure_failure.json"
            or _sha256_file(terminal_path)
            != _require_sha(
                terminal_binding.get("file_sha256"),
                "source terminal failure file SHA256",
            )
            or bindings.get("source_terminal_failure_file_sha256")
            != terminal_binding.get("file_sha256")
        ):
            raise CampaignSemigroupClosureError(
                "campaign closure source terminal failure SHA256 mismatch"
            )
        terminal_payload = _validate_terminal_failure(
            terminal_path.parent,
            formal_campaign_id=visual_chain_formal_id,
            repo_root=repo_root,
        )
        if terminal_payload.get("failure_contract_sha256") != terminal_binding.get(
            "contract_sha256"
        ):
            raise CampaignSemigroupClosureError(
                "campaign closure source terminal failure contract mismatch"
            )
    source_review_text = str(bindings.get("visual_review_source_path", ""))
    if Path(source_review_text).is_absolute():
        raise CampaignSemigroupClosureError(
            "source visual review path must be repository-relative"
        )
    source_review_path = _resolve_config_path(
        repo_root, source_review_text, "source visual review"
    )
    expected_source_review_path = (
        repo_root / CAMPAIGN_BASE / bootstrap_id / "preflight" / "visual_review.json"
    ).resolve()
    if (
        source_review_path != expected_source_review_path
        or _sha256_file(source_review_path)
        != _require_sha(
            bindings.get("visual_review_source_sha256"),
            "source visual review SHA256",
        )
        or _read_json(source_review_path, "source visual review") != review
    ):
        raise CampaignSemigroupClosureError(
            "source visual review path, SHA256, or published copy mismatch"
        )
    if canonical_json_sha256(preflight) != bindings["preflight_contract_sha256"]:
        raise CampaignSemigroupClosureError(
            "sealed preflight canonical digest mismatch"
        )
    if gate.get("gate_contract_sha256") != _contract_digest(
        gate, "gate_contract_sha256"
    ):
        raise CampaignSemigroupClosureError("sealed gate canonical digest mismatch")
    if schedule.get(
        "schedule_contract_sha256"
    ) != canonical_r9_schedule_contract_sha256(schedule):
        raise CampaignSemigroupClosureError("sealed schedule canonical digest mismatch")
    _validate_published_visual_review_chain(
        repo_root=repo_root,
        bootstrap_campaign_id=bootstrap_id,
        formal_campaign_id=visual_chain_formal_id,
        evidence=evidence,
        assignment=assignment,
        blinding_map=blinding_map,
        review=review,
    )
    if (
        gate.get("schedule_contract_sha256") != schedule.get("schedule_contract_sha256")
        or schedule.get("r9_semigroup_gate_contract")
        != _path_text(artifact_paths["gate_contract"], repo_root)
        or schedule.get("r9_semigroup_gate_contract_sha256")
        != artifact_hashes["gate_contract"]
        or report.get("gate_passed") is not True
        or report.get("selected_t_cut") != seal.get("selected_t_cut")
        or gate.get("selected_t_cut") != seal.get("selected_t_cut")
        or schedule.get("t_cut") != seal.get("selected_t_cut")
    ):
        raise CampaignSemigroupClosureError(
            "campaign closure report/gate/schedule chain mismatch"
        )
    required = seal.get("formal_campaign_required_bindings")
    expected_required = {
        "closure_seal": _path_text(seal_path, repo_root),
        "locked_schedule_manifest": _path_text(
            artifact_paths["locked_schedule_manifest"], repo_root
        ),
        "gate_contract": _path_text(artifact_paths["gate_contract"], repo_root),
    }
    if not isinstance(required, Mapping) or dict(required) != expected_required:
        raise CampaignSemigroupClosureError(
            "campaign closure formal required bindings mismatch"
        )
    resolved = {
        "bootstrap_campaign_id": bootstrap_id,
        "formal_campaign_id": formal_campaign_id,
        "closure": {
            "path": _path_text(seal_path, repo_root),
            "file_sha256": _sha256_file(seal_path),
            "contract_sha256": seal_contract_sha256,
        },
        "schedule": {
            "path": _path_text(artifact_paths["locked_schedule_manifest"], repo_root),
            "file_sha256": artifact_hashes["locked_schedule_manifest"],
            "contract_sha256": schedule["schedule_contract_sha256"],
        },
        "gate": {
            "path": _path_text(artifact_paths["gate_contract"], repo_root),
            "file_sha256": artifact_hashes["gate_contract"],
            "contract_sha256": gate["gate_contract_sha256"],
        },
    }
    if schema_version == 2:
        resolved.update(
            {
                "policy_campaign_id": closure_prefix_id,
                "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
            }
        )
    return resolved


def _validate_published_visual_review_chain(
    *,
    repo_root: Path,
    bootstrap_campaign_id: str,
    formal_campaign_id: str,
    evidence: Mapping[str, Any],
    assignment: Mapping[str, Any],
    blinding_map: Mapping[str, Any],
    review: Mapping[str, Any],
) -> None:
    _reject_assignment_decision_fields(assignment)
    if (
        evidence.get("evidence_manifest_sha256")
        != _contract_digest(evidence, "evidence_manifest_sha256")
        or assignment.get("visual_review_assignment_sha256")
        != _contract_digest(assignment, "visual_review_assignment_sha256")
        or blinding_map.get("blinding_map_sha256")
        != _contract_digest(blinding_map, "blinding_map_sha256")
        or review.get("visual_review_sha256")
        != _contract_digest(review, "visual_review_sha256")
    ):
        raise CampaignSemigroupClosureError(
            "sealed visual review chain canonical digest mismatch"
        )
    if any(
        contract.get("bootstrap_campaign_id") != bootstrap_campaign_id
        or contract.get("formal_campaign_id") != formal_campaign_id
        for contract in (assignment, blinding_map, review)
    ):
        raise CampaignSemigroupClosureError(
            "sealed visual review chain campaign ID mismatch"
        )
    evidence_binding = assignment.get("evidence_manifest")
    map_binding = assignment.get("blinding_map")
    if (
        not isinstance(evidence_binding, Mapping)
        or set(evidence_binding) != {"path", "file_sha256", "contract_sha256"}
        or not isinstance(map_binding, Mapping)
        or set(map_binding) != {"path", "file_sha256", "contract_sha256"}
    ):
        raise CampaignSemigroupClosureError(
            "sealed visual review assignment artifact bindings are invalid"
        )
    source_evidence = _resolve_config_path(
        repo_root,
        evidence_binding.get("path"),
        "sealed assigned evidence manifest",
    )
    source_map = _resolve_config_path(
        repo_root,
        map_binding.get("path"),
        "sealed visual review blinding map source",
    )
    if (
        _sha256_file(source_evidence) != evidence_binding.get("file_sha256")
        or _read_json(source_evidence, "sealed assigned evidence manifest")
        != dict(evidence)
        or evidence_binding.get("contract_sha256")
        != evidence.get("evidence_manifest_sha256")
        or _sha256_file(source_map) != map_binding.get("file_sha256")
        or _read_json(source_map, "sealed visual review blinding map source")
        != dict(blinding_map)
        or map_binding.get("contract_sha256") != blinding_map.get("blinding_map_sha256")
    ):
        raise CampaignSemigroupClosureError(
            "sealed visual review assignment source artifact mismatch"
        )
    if (
        review.get("evidence_manifest_sha256")
        != evidence.get("evidence_manifest_sha256")
        or review.get("visual_review_assignment_sha256")
        != assignment.get("visual_review_assignment_sha256")
        or assignment.get("registered_splits") != list(SPLIT_KEYS)
        or blinding_map.get("registered_splits") != list(SPLIT_KEYS)
    ):
        raise CampaignSemigroupClosureError(
            "sealed assignment/evidence/map/review linkage mismatch"
        )
    assignment_conditions = assignment.get("conditions")
    map_conditions = blinding_map.get("conditions")
    review_conditions = review.get("conditions")
    if (
        not isinstance(assignment_conditions, list)
        or len(assignment_conditions) != len(SPLIT_KEYS)
        or not isinstance(map_conditions, Mapping)
        or not isinstance(review_conditions, Mapping)
    ):
        raise CampaignSemigroupClosureError(
            "sealed visual review condition inventory is invalid"
        )
    condition_ids = [
        str(condition.get("condition_id", ""))
        for condition in assignment_conditions
        if isinstance(condition, Mapping)
    ]
    if (
        len(condition_ids) != len(SPLIT_KEYS)
        or len(set(condition_ids)) != len(SPLIT_KEYS)
        or set(map_conditions) != set(condition_ids)
        or set(review_conditions) != set(condition_ids)
    ):
        raise CampaignSemigroupClosureError(
            "sealed visual review blinded condition linkage mismatch"
        )
    expected_sheets: set[Path] = set()
    sheet_root: Path | None = None
    reviewed_ids = assignment.get("reviewed_sample_ids")
    if not isinstance(reviewed_ids, list) or len(reviewed_ids) != SAMPLE_COUNT:
        raise CampaignSemigroupClosureError(
            "sealed visual review assignment does not bind 64 IDs"
        )
    _validate_blinding_map_contract(
        blinding_map,
        assignment=assignment,
        evidence_manifest_sha256=str(evidence["evidence_manifest_sha256"]),
        bootstrap_campaign_id=bootstrap_campaign_id,
        formal_campaign_id=formal_campaign_id,
        ordered_sample_id_sha256=_sample_id_digest(reviewed_ids),
    )
    seen_splits = set()
    manifest_ids = set(reviewed_ids)
    for condition in assignment_conditions:
        condition_id = str(condition["condition_id"])
        pages = condition.get("pages")
        if not isinstance(pages, list):
            raise CampaignSemigroupClosureError(
                "sealed visual review assignment pages are invalid"
            )
        flattened_ids = []
        for page in pages:
            if not isinstance(page, Mapping):
                raise CampaignSemigroupClosureError(
                    "sealed visual review assignment page is invalid"
                )
            path = _resolve_config_path(
                repo_root, page.get("path"), "sealed contact sheet"
            )
            condition_root = path.parent
            candidate_sheet_root = condition_root.parent
            if sheet_root is None:
                sheet_root = candidate_sheet_root
            if (
                candidate_sheet_root != sheet_root
                or condition_root.name != condition_id
                or page.get("condition_id") != condition_id
                or page.get("column_ids") != condition.get("column_ids")
                or _sha256_file(path) != page.get("file_sha256")
                or page.get("sheet_contract_sha256")
                != _contract_digest(page, "sheet_contract_sha256")
            ):
                raise CampaignSemigroupClosureError(
                    "sealed contact sheet path or SHA256 mismatch"
                )
            expected_sheets.add(path)
            flattened_ids.extend(page.get("sample_ids", ()))
        if flattened_ids != reviewed_ids:
            raise CampaignSemigroupClosureError(
                "sealed contact sheets do not cover the ordered 64 IDs"
            )
        map_row = map_conditions[condition_id]
        if not isinstance(map_row, Mapping) or set(map_row) != {"split", "columns"}:
            raise CampaignSemigroupClosureError(
                "sealed blinding map condition is invalid"
            )
        split = str(map_row.get("split", ""))
        if split not in SPLIT_KEYS or split in seen_splits:
            raise CampaignSemigroupClosureError(
                "sealed blinding map split mapping is invalid"
            )
        seen_splits.add(split)
        decision = review_conditions[condition_id]
        if not isinstance(decision, Mapping) or set(decision) != {
            "passed",
            "severe_count",
            "severe_sample_ids",
        }:
            raise CampaignSemigroupClosureError(
                "sealed reviewer decision fields are invalid"
            )
        severe_ids = decision.get("severe_sample_ids")
        count = decision.get("severe_count")
        passed = decision.get("passed")
        if (
            not isinstance(severe_ids, list)
            or len(set(severe_ids)) != len(severe_ids)
            or not set(severe_ids) <= manifest_ids
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count != len(severe_ids)
            or not isinstance(passed, bool)
            or passed != (count == 0)
        ):
            raise CampaignSemigroupClosureError(
                "sealed reviewer decision contract mismatch"
            )
    if sheet_root is None or seen_splits != set(SPLIT_KEYS):
        raise CampaignSemigroupClosureError(
            "sealed visual review chain does not cover three splits"
        )
    _validate_exact_file_tree(
        sheet_root,
        expected_sheets,
        label="sealed visual review contact sheets",
    )


def _existing_file(value: Any, label: str, *, repo_root: Path | None = None) -> Path:
    path = Path(str(value))
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    if path.is_symlink():
        raise CampaignSemigroupClosureError(f"{label} must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_file():
        raise CampaignSemigroupClosureError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_config_path(repo_root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return _existing_file(path, label)


def _resolved_optional_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _relative_to(path: Path, root: Path, label: str) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CampaignSemigroupClosureError(
            f"{label} escapes its required root"
        ) from exc


def _path_text(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignSemigroupClosureError(f"{label} must be a JSON object")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CampaignSemigroupClosureError(
                    f"{label} line {line_number} is not an object"
                )
            rows.append(row)
    return rows


def _read_manifest_ids(path: Path) -> list[str]:
    ids = []
    for row in _read_jsonl(path, "preflight manifest"):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise CampaignSemigroupClosureError(
                "preflight manifest has an invalid or duplicate sample ID"
            )
        ids.append(sample_id)
    return ids


def _load_index_sources(
    path: Path, *, manifest_ids: Sequence[str], repo_root: Path
) -> dict[str, Path]:
    required = set(manifest_ids)
    sources: dict[str, Path] = {}
    for row in _read_jsonl(path, "source index"):
        sample_id = row.get("sample_id")
        if sample_id not in required:
            continue
        if sample_id in sources:
            raise CampaignSemigroupClosureError(
                f"source index contains duplicate target ID: {sample_id}"
            )
        sources[str(sample_id)] = _existing_file(
            row.get("image_path"),
            f"indexed source image {sample_id}",
            repo_root=repo_root,
        )
    if set(sources) != required:
        missing = sorted(required - set(sources))
        raise CampaignSemigroupClosureError(
            f"source index does not cover the exact preflight manifest: {missing!r}"
        )
    return sources


def _write_exclusive_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    canonical: bool,
    permissions: int = 0o444,
) -> str:
    content = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":") if canonical else None,
            indent=None if canonical else 2,
            allow_nan=False,
        )
        + ("" if canonical else "\n")
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, permissions)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("immutable closure write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(content).hexdigest()


def _write_exclusive_bytes(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("immutable contact sheet write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _contract_digest(payload: Mapping[str, Any], field: str) -> str:
    canonical = dict(payload)
    canonical.pop(field, None)
    return canonical_json_sha256(canonical)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise CampaignSemigroupClosureError(f"{label} is not a lowercase SHA256")
    return text


def _require_campaign_id(value: str, label: str) -> str:
    if _CAMPAIGN_ID.fullmatch(value) is None:
        raise CampaignSemigroupClosureError(f"{label} is not canonical")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignSemigroupClosureError(f"{label} is not finite") from exc
    if not math.isfinite(parsed):
        raise CampaignSemigroupClosureError(f"{label} is not finite")
    return parsed


def _finite_open_unit(value: Any, label: str) -> float:
    parsed = _finite(value, label)
    if not 0.0 < parsed < 1.0:
        raise CampaignSemigroupClosureError(f"{label} must be in (0,1)")
    return parsed


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


__all__ = [
    "CampaignSemigroupClosureError",
    "build_campaign_semigroup_evidence",
    "finalize_campaign_semigroup_closure",
    "prepare_campaign_semigroup_visual_review",
    "resolve_formal_campaign_semigroup_closure",
]
