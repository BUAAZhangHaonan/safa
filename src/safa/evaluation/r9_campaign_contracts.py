from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from safa.evaluation.r9_evaluator_resources import (
    EvaluatorResourceContractError,
    validate_evaluator_resource_profiles,
)


R9_CAMPAIGN_CONTRACT = "safa_r9_campaign_v1"
R9_GENERATION_EXPERIMENT_CONTRACT = "safa_r9_meanflow_v1"
R9_DETERMINISM_POLICY_SHA256 = (
    "ea6a4e81627a993066d9b1a3ca4ae791a0bcb3e21e399a5d2cb27811aa22147f"
)
R9_ATTENTION_BACKEND = "native"
R9_BOOTSTRAP_ITERATIONS = 10_000
R9_CALIBRATION_SEEDS = (1337, 2027, 3407)
R9_REQUIRED_MANIFESTS = {
    "calibration_64": 64,
    "validate_512": 512,
    "full_2048": 2048,
    "full_visual_64": 64,
}
R9_MANIFEST_KEYS = frozenset({*R9_REQUIRED_MANIFESTS, "arcface_clean_pool"})
R9_GATE_CONTEXT_FIELDS = frozenset(
    {
        "campaign_id",
        "campaign_runtime_sha256",
        "manifest_contracts_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "phase_results_sha256",
        "automatic_evidence_sha256",
        "run_plan_sha256",
        "evaluator_evidence_sha256",
    }
)
R9_HELDOUT_ASSETS = frozenset({"e1", "e2", "facenet", "adaface"})
R9_IDENTITY_RECOGNIZERS = ("arcface", "facenet", "adaface")
R9_IDENTITY_ROLES = ("native", "winner")
R9_TAR_FAR_KEYS = ("0.001", "0.0001")
R9_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_contract",
        "generation_experiment_contract",
        "campaign_id",
        "campaign_root",
        "campaign_template",
        "base_config",
        "checkpoint",
        "determinism_policy_sha256",
        "attention_backend",
        "schedule",
        "semigroup_gate",
        "seeds",
        "manifests",
        "clean_source",
        "manifest_construction",
        "resources",
        "bootstrap",
        "evaluation",
        "phases",
    }
)


class CampaignContractError(ValueError):
    """Raised when an R9 campaign artifact violates its locked schema."""


def canonical_campaign_runtime_sha256(runtime: Mapping[str, Any]) -> str:
    payload = dict(runtime)
    payload.pop("campaign_runtime_sha256", None)
    return _canonical_json_sha256(payload)


def build_resource_smoke_contract(
    *,
    run_id: str,
    arm_id: str,
    manifest: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    peak_rss_bytes: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_resource_smoke_v1",
        "run_id": _require_nonempty(run_id, "resource smoke run_id"),
        "arm_id": _require_nonempty(arm_id, "resource smoke arm_id"),
        "manifest": _require_nonempty(manifest, "resource smoke manifest"),
        "manifest_sha256": _require_sha256(
            manifest_sha256, "resource smoke manifest SHA256"
        ),
        "checkpoint_sha256": _require_sha256(
            checkpoint_sha256, "resource smoke checkpoint SHA256"
        ),
        "worker_count": 1,
        "measurement": "process_tree_peak_rss_bytes",
        "peak_rss_bytes": _strict_int(peak_rss_bytes, "resource smoke peak RSS bytes"),
        "completed": True,
        "exit_code": 0,
    }
    if payload["peak_rss_bytes"] <= 0:
        raise CampaignContractError("resource smoke peak RSS bytes must be positive")
    payload["resource_smoke_sha256"] = _canonical_contract_sha256(
        payload, "resource_smoke_sha256"
    )
    return validate_resource_smoke_contract(payload)


def validate_resource_smoke_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _json_mapping(contract, "resource smoke contract")
    required = {
        "schema_version",
        "contract_type",
        "run_id",
        "arm_id",
        "manifest",
        "manifest_sha256",
        "checkpoint_sha256",
        "worker_count",
        "measurement",
        "peak_rss_bytes",
        "completed",
        "exit_code",
        "resource_smoke_sha256",
    }
    if set(normalized) != required:
        raise CampaignContractError("resource smoke fields are not canonical")
    _verify_contract_digest(normalized, "resource_smoke_sha256")
    if (
        normalized.get("schema_version") != 1
        or normalized.get("contract_type") != "safa_r9_resource_smoke_v1"
    ):
        raise CampaignContractError("resource smoke contract type mismatch")
    for field in ("run_id", "arm_id", "manifest"):
        _require_nonempty(normalized.get(field), f"resource smoke {field}")
    for field in ("manifest_sha256", "checkpoint_sha256"):
        _require_sha256(normalized.get(field), f"resource smoke {field}")
    peak_rss = _strict_int(
        normalized.get("peak_rss_bytes"), "resource smoke peak RSS bytes"
    )
    if peak_rss <= 0:
        raise CampaignContractError("resource smoke peak RSS bytes must be positive")
    if (
        normalized.get("worker_count") != 1
        or normalized.get("measurement") != "process_tree_peak_rss_bytes"
        or normalized.get("completed") is not True
        or normalized.get("exit_code") != 0
    ):
        raise CampaignContractError("resource smoke must certify one successful worker")
    return normalized


def validate_campaign_runtime(
    runtime: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise CampaignContractError("campaign runtime must be a mapping")
    normalized = _json_mapping(runtime, "campaign runtime")
    if "campaign_runtime_sha256" in normalized:
        raise CampaignContractError(
            "campaign_runtime_sha256 is generated and must not be hand-filled"
        )
    if set(normalized) != R9_RUNTIME_FIELDS:
        missing = sorted(R9_RUNTIME_FIELDS - set(normalized))
        extra = sorted(set(normalized) - R9_RUNTIME_FIELDS)
        raise CampaignContractError(
            f"campaign runtime fields mismatch: missing={missing!r}, extra={extra!r}"
        )
    if normalized.get("schema_version") != 1:
        raise CampaignContractError("campaign runtime schema_version must be 1")
    if normalized.get("experiment_contract") != R9_CAMPAIGN_CONTRACT:
        raise CampaignContractError(
            f"experiment_contract must be {R9_CAMPAIGN_CONTRACT!r}"
        )
    _require_nonempty(normalized.get("campaign_id"), "campaign_id")
    if (
        normalized.get("generation_experiment_contract")
        != R9_GENERATION_EXPERIMENT_CONTRACT
    ):
        raise CampaignContractError(
            "generation_experiment_contract must bind the R9 MeanFlow contract"
        )
    if normalized.get("determinism_policy_sha256") != R9_DETERMINISM_POLICY_SHA256:
        raise CampaignContractError("runtime determinism policy SHA256 mismatch")
    if normalized.get("attention_backend") != R9_ATTENTION_BACKEND:
        raise CampaignContractError("runtime attention backend must be native")
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise CampaignContractError("repo_root must be an existing directory")
    campaign_root = _require_repo_path(
        normalized.get("campaign_root"), root, "campaign_root"
    )
    normalized["campaign_root"] = str(campaign_root.relative_to(root))
    normalized["campaign_template"] = _validate_bound_file(
        normalized.get("campaign_template"),
        repo_root=root,
        label="campaign template",
    )
    normalized["base_config"] = _validate_bound_file(
        normalized.get("base_config"), repo_root=root, label="base config"
    )
    normalized["checkpoint"] = _validate_bound_file(
        normalized.get("checkpoint"), repo_root=root, label="checkpoint"
    )
    seeds = normalized.get("seeds")
    if not isinstance(seeds, Mapping):
        raise CampaignContractError("campaign runtime requires a seeds mapping")
    _validate_seed_contract(seeds)
    manifests = normalized.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != R9_MANIFEST_KEYS:
        raise CampaignContractError(
            "campaign runtime manifests must contain only the five R9 manifests"
        )
    clean_source = _validate_clean_source_contract(normalized.get("clean_source"), root)
    construction = normalized.get("manifest_construction")
    if not isinstance(construction, Mapping) or set(construction) != {
        "r8_calibration_64",
        "diagnose_18",
    }:
        raise CampaignContractError(
            "manifest_construction must bind R8 calibration and diagnose_18"
        )
    manifest_contract = validate_manifest_contracts(
        manifests,
        root,
        clean_source=clean_source,
        r8_calibration_binding=construction["r8_calibration_64"],
        diagnose_manifest=construction["diagnose_18"],
    )
    normalized["manifests"] = manifest_contract["manifests"]
    normalized["clean_source"] = clean_source
    normalized["manifest_construction"] = {
        key: manifest_contract["provenance"][key]
        for key in ("r8_calibration_64", "diagnose_18")
    }
    normalized["manifest_contracts_sha256"] = manifest_contract[
        "manifest_contracts_sha256"
    ]
    schedule, semigroup_gate = _validate_schedule_and_gate(
        normalized.get("schedule"),
        normalized.get("semigroup_gate"),
        repo_root=root,
        checkpoint_sha256=normalized["checkpoint"]["sha256"],
        calibration_manifest_sha256=normalized["manifests"]["calibration_64"]["sha256"],
    )
    normalized["schedule"] = schedule
    normalized["semigroup_gate"] = semigroup_gate
    normalized["resources"] = _validate_runtime_resources(
        normalized.get("resources"),
        repo_root=root,
        manifests=normalized["manifests"],
        checkpoint_sha256=normalized["checkpoint"]["sha256"],
    )
    normalized["bootstrap"] = _validate_runtime_bootstrap(normalized.get("bootstrap"))
    normalized["evaluation"] = _validate_runtime_evaluation(
        normalized.get("evaluation"), repo_root=root
    )
    normalized["phases"] = _validate_runtime_phases(
        normalized.get("phases"), seeds=seeds
    )
    normalized["campaign_runtime_sha256"] = canonical_campaign_runtime_sha256(
        normalized
    )
    return normalized


def validate_manifest_contracts(
    manifests: Mapping[str, Any],
    repo_root: Path,
    *,
    clean_source: Mapping[str, Any],
    r8_calibration_binding: Mapping[str, Any],
    diagnose_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(manifests, Mapping) or set(manifests) != R9_MANIFEST_KEYS:
        raise CampaignContractError(
            "manifest contracts must contain only the five registered manifests"
        )
    if "manifest_contracts_sha256" in manifests:
        raise CampaignContractError(
            "manifest_contracts_sha256 is generated and must not be hand-filled"
        )
    root = Path(repo_root).resolve()
    clean = _validate_clean_source_contract(clean_source, root)
    entries: dict[str, dict[str, Any]] = {}
    ids: dict[str, list[str]] = {}
    for name in sorted(R9_MANIFEST_KEYS):
        declared = manifests[name]
        if not isinstance(declared, Mapping):
            raise CampaignContractError(f"manifest {name!r} must be a mapping")
        expected_count = R9_REQUIRED_MANIFESTS.get(name)
        entry, sample_ids = _validate_manifest_entry(
            name=name,
            declared=declared,
            repo_root=root,
            expected_count=expected_count,
        )
        entries[name] = entry
        ids[name] = sample_ids
    if ids["arcface_clean_pool"] != _read_manifest_ids(root / clean["path"]):
        raise CampaignContractError(
            "arcface_clean_pool must exactly equal the locked ArcFace-clean source"
        )
    if entries["arcface_clean_pool"] != {
        key: clean[key]
        for key in ("path", "sha256", "sample_count", "ordered_sample_id_sha256")
    }:
        raise CampaignContractError(
            "ArcFace-clean pool metadata disagrees with clean_source evidence"
        )
    if clean["arcface_exact_one"] is not True:
        raise CampaignContractError("clean_source must certify exactly one ArcFace")
    r8_entry, r8_ids = _validate_manifest_entry(
        name="r8_calibration_64",
        declared=r8_calibration_binding,
        repo_root=root,
        expected_count=64,
    )
    if (
        ids["calibration_64"] != r8_ids
        or entries["calibration_64"]["ordered_sample_id_sha256"]
        != r8_entry["ordered_sample_id_sha256"]
    ):
        raise CampaignContractError(
            "calibration_64 IDs must exactly equal the fixed R8 64-ID binding"
        )
    diagnose_entry, diagnose_ids = _validate_diagnose_manifest_file(
        diagnose_manifest, repo_root=root
    )
    if not set(diagnose_ids) <= set(ids["calibration_64"]):
        raise CampaignContractError("diagnose_18 IDs must all belong to calibration_64")
    if not set(diagnose_ids) <= set(ids["arcface_clean_pool"]):
        raise CampaignContractError(
            "diagnose_18 IDs must all belong to the ArcFace-clean pool"
        )
    tuning_sets = [
        set(ids["calibration_64"]),
        set(ids["validate_512"]),
        set(ids["full_2048"]),
    ]
    if any(
        tuning_sets[left] & tuning_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise CampaignContractError(
            "calibration_64, validate_512 and full_2048 must be disjoint"
        )
    if not set(ids["full_visual_64"]) < set(ids["full_2048"]):
        raise CampaignContractError(
            "full_visual_64 must be a strict subset of full_2048"
        )
    full_visual_rows = _read_manifest_rows(root / entries["full_visual_64"]["path"])
    full_ids = ids["full_2048"]
    full_indices = []
    for line_number, row in enumerate(full_visual_rows, 1):
        full_index = _strict_int(
            row.get("full_index"), f"full_visual_64 row {line_number} full_index"
        )
        if full_index < 0 or full_index >= len(full_ids):
            raise CampaignContractError("full_visual_64 full_index is out of range")
        if row.get("sample_id") != full_ids[full_index]:
            raise CampaignContractError(
                "full_visual_64 sample_id disagrees with full_2048 full_index"
            )
        full_indices.append(full_index)
    if len(set(full_indices)) != len(full_indices):
        raise CampaignContractError(
            "full_visual_64 contains duplicate full_index values"
        )
    clean_pool = set(ids["arcface_clean_pool"])
    for name in R9_REQUIRED_MANIFESTS:
        if not set(ids[name]) <= clean_pool:
            raise CampaignContractError(
                f"manifest {name!r} contains IDs outside the ArcFace-clean pool"
            )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_manifest_contracts_v1",
        "manifests": entries,
        "provenance": {
            "clean_source": clean,
            "r8_calibration_64": r8_entry,
            "diagnose_18": diagnose_entry,
        },
        "relationships": {
            "calibration_validate_full_disjoint": True,
            "full_visual_strict_subset_of_full": True,
            "all_ids_arcface_clean": True,
        },
    }
    payload["manifest_contracts_sha256"] = _canonical_contract_sha256(
        payload, "manifest_contracts_sha256"
    )
    return payload


def validate_diagnose_manifest_contract(
    diagnose_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the locked 9 difficult/9 matched-control diagnose manifest."""
    return _validate_diagnose_manifest(diagnose_manifest)


def derive_visual_arm_pass(
    review: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    severe_limit: int,
) -> dict[str, Any]:
    _require_nonnegative_int(severe_limit, "visual severe limit")
    if not isinstance(review, Mapping) or not isinstance(evidence, Mapping):
        raise CampaignContractError("visual review and evidence must be mappings")
    evidence_ids = _visual_evidence_ids(evidence)
    review_rows = review.get("samples")
    if not isinstance(review_rows, list) or len(review_rows) != len(evidence_ids):
        raise CampaignContractError("visual review must cover every evidence sample")
    review_ids: list[str] = []
    severe_ids: list[str] = []
    for index, row in enumerate(review_rows):
        if not isinstance(row, Mapping):
            raise CampaignContractError(f"visual review row {index} must be a mapping")
        sample_id = row.get("sample_id")
        _require_nonempty(sample_id, f"visual review row {index} sample_id")
        severe = row.get("severe")
        if not isinstance(severe, bool):
            raise CampaignContractError(
                f"visual review row {index} severe must be boolean"
            )
        review_ids.append(str(sample_id))
        if severe:
            severe_ids.append(str(sample_id))
    if review_ids != evidence_ids or len(set(review_ids)) != len(review_ids):
        raise CampaignContractError(
            "visual review order and membership must exactly match evidence"
        )
    severe_count = len(severe_ids)
    passed = severe_count <= severe_limit
    if "severe_count" in review and review.get("severe_count") != severe_count:
        raise CampaignContractError("hand-filled visual severe_count is inconsistent")
    if "passed" in review and review.get("passed") is not passed:
        raise CampaignContractError("hand-filled visual passed verdict is inconsistent")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_visual_arm_gate_v1",
        "evidence_contract_sha256": _evidence_digest(evidence),
        "reviewed_sample_count": len(review_ids),
        "severe_limit": severe_limit,
        "severe_count": severe_count,
        "severe_sample_ids": severe_ids,
        "passed": passed,
    }
    payload["visual_gate_sha256"] = _canonical_contract_sha256(
        payload, "visual_gate_sha256"
    )
    return payload


def aggregate_seed_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    metric_fields: Sequence[str],
) -> dict[str, Any]:
    seeds = _normalize_expected_seeds(expected_seeds)
    fields = tuple(str(field) for field in metric_fields)
    if (
        not fields
        or len(set(fields)) != len(fields)
        or any(not field for field in fields)
    ):
        raise CampaignContractError("metric_fields must be unique non-empty names")
    by_id: dict[str, dict[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CampaignContractError(f"seed metric row {index} must be a mapping")
        sample_id = _require_nonempty(row.get("sample_id"), f"row {index} sample_id")
        seed = _strict_int(row.get("seed"), f"row {index} seed")
        if seed not in seeds:
            raise CampaignContractError(f"row {index} contains an unregistered seed")
        sample_rows = by_id.setdefault(sample_id, {})
        if seed in sample_rows:
            raise CampaignContractError("duplicate sample_id/seed metric row")
        sample_rows[seed] = row
    if not by_id:
        raise CampaignContractError("seed aggregation requires at least one sample")
    samples = []
    for sample_id in sorted(by_id):
        sample_rows = by_id[sample_id]
        if tuple(sorted(sample_rows)) != tuple(sorted(seeds)):
            raise CampaignContractError(
                f"sample {sample_id!r} does not cover every registered seed"
            )
        metrics = {
            field: statistics.fmean(
                _finite_float(sample_rows[seed].get(field), field) for seed in seeds
            )
            for field in fields
        }
        samples.append({"sample_id": sample_id, "metrics": metrics})
    aggregate = {
        field: statistics.fmean(sample["metrics"][field] for sample in samples)
        for field in fields
    }
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_seed_aggregate_v1",
        "seeds": list(seeds),
        "sample_count": len(samples),
        "observation_count": len(rows),
        "metric_fields": list(fields),
        "samples": samples,
        "aggregate": aggregate,
    }
    payload["seed_aggregate_sha256"] = _canonical_contract_sha256(
        payload, "seed_aggregate_sha256"
    )
    return payload


def privacy_delta_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    bootstrap_seed: int,
    iterations: int = R9_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    if iterations != R9_BOOTSTRAP_ITERATIONS:
        raise CampaignContractError("R9 cluster bootstrap requires exactly 10000 draws")
    _require_nonnegative_int(bootstrap_seed, "bootstrap seed")
    delta_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CampaignContractError(f"privacy row {index} must be a mapping")
        candidate = _finite_float(
            row.get("source_candidate_cosine"), "source_candidate_cosine"
        )
        native = _finite_float(row.get("source_native_cosine"), "source_native_cosine")
        delta_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "seed": row.get("seed"),
                "privacy_delta": candidate - native,
            }
        )
    aggregate = aggregate_seed_metrics(
        delta_rows,
        expected_seeds=expected_seeds,
        metric_fields=("privacy_delta",),
    )
    cluster_values = [
        float(sample["metrics"]["privacy_delta"]) for sample in aggregate["samples"]
    ]
    rng = random.Random(bootstrap_seed)
    sample_count = len(cluster_values)
    draws = []
    for _ in range(iterations):
        draws.append(
            statistics.fmean(
                cluster_values[rng.randrange(sample_count)] for _ in range(sample_count)
            )
        )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_privacy_cluster_bootstrap_v1",
        "direction": "source_candidate_minus_source_native",
        "cluster_unit": "sample_id",
        "seeds": list(_normalize_expected_seeds(expected_seeds)),
        "sample_count": sample_count,
        "observation_count": len(rows),
        "iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "mean_delta": statistics.fmean(cluster_values),
        "lower_95_one_sided": _percentile(draws, 0.05),
        "upper_95_one_sided": _percentile(draws, 0.95),
    }
    payload["bootstrap_sha256"] = _canonical_contract_sha256(
        payload, "bootstrap_sha256"
    )
    return payload


def build_a_gate_contract(
    context: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    *,
    diagnose_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bound_context = _validate_gate_context(context)
    diagnose = _validate_diagnose_manifest(diagnose_manifest)
    if diagnose["sha256"] != bound_context["manifest_sha256"]:
        raise CampaignContractError("A gate context does not bind diagnose manifest")
    evaluated = []
    for arm in arms:
        _reject_derived_arm_fields(arm)
        arm_id, family, config_sha, output_sha = _arm_identity(arm)
        if family not in {
            "flow_map2",
            "paper_split_constant",
            "paper_split_interval_ablation",
        }:
            raise CampaignContractError(f"unknown A-stage family: {family!r}")
        repeats = arm.get("repeat_results")
        if not isinstance(repeats, list) or len(repeats) != 3:
            raise CampaignContractError("A-stage arm requires exactly three repeats")
        repeat_digests = []
        failures: list[str] = []
        normalized_repeats = []
        for expected_index, repeat in enumerate(repeats):
            if not isinstance(repeat, Mapping):
                raise CampaignContractError("A-stage repeat must be a mapping")
            if repeat.get("repeat_index") != expected_index:
                raise CampaignContractError("A-stage repeat indices must be 0,1,2")
            difficult = _require_nonnegative_int(
                repeat.get("difficult_severe_count"), "difficult severe count"
            )
            control = _require_nonnegative_int(
                repeat.get("control_severe_count"), "control severe count"
            )
            e0 = _finite_float(repeat.get("e0_mean"), "A-stage E0 mean")
            edev_delta = _finite_float(
                repeat.get("edev_delta_vs_matched_native"), "A-stage Edev delta"
            )
            diagnostics_finite = repeat.get("diagnostics_finite")
            if not isinstance(diagnostics_finite, bool):
                raise CampaignContractError("diagnostics_finite must be boolean")
            digest = _require_sha256(repeat.get("run_sha256"), "repeat run SHA256")
            repeat_digests.append(digest)
            repeat_failures = []
            if difficult > 3:
                repeat_failures.append("difficult_severe_count_gt_3")
            if control > 1:
                repeat_failures.append("control_severe_count_gt_1")
            if e0 < 0.75:
                repeat_failures.append("e0_mean_lt_0.75")
            if edev_delta < 0:
                repeat_failures.append("edev_below_matched_native")
            if not diagnostics_finite:
                repeat_failures.append("diagnostics_nonfinite_or_contract_mismatch")
            failures.extend(
                f"repeat_{expected_index}:{item}" for item in repeat_failures
            )
            normalized_repeats.append(
                {
                    "repeat_index": expected_index,
                    "run_sha256": digest,
                    "difficult_severe_count": difficult,
                    "control_severe_count": control,
                    "e0_mean": e0,
                    "edev_delta_vs_matched_native": edev_delta,
                    "diagnostics_finite": diagnostics_finite,
                }
            )
        if len(set(repeat_digests)) != 1:
            failures.append("three_repeats_not_bitwise_identical")
        evaluated.append(
            {
                "arm_id": arm_id,
                "family": family,
                "config_sha256": config_sha,
                "output_sha256": output_sha,
                "repeat_results": normalized_repeats,
                "severe_sort_key": max(
                    row["difficult_severe_count"] + row["control_severe_count"]
                    for row in normalized_repeats
                ),
                "edev_sort_value": statistics.fmean(
                    row["edev_delta_vs_matched_native"] for row in normalized_repeats
                ),
                "e0_sort_value": statistics.fmean(
                    row["e0_mean"] for row in normalized_repeats
                ),
                "failures": failures,
                "passed": not failures,
            }
        )
    selected = []
    for family in (
        "flow_map2",
        "paper_split_constant",
        "paper_split_interval_ablation",
    ):
        passing = [
            row for row in evaluated if row["family"] == family and row["passed"]
        ]
        if passing:
            selected.append(
                min(
                    passing,
                    key=lambda row: (
                        row["severe_sort_key"],
                        -row["edev_sort_value"],
                        -row["e0_sort_value"],
                        row["arm_id"],
                    ),
                )["arm_id"]
            )
    return _build_gate_payload(
        phase="diagnose",
        context=bound_context,
        seeds=[1337],
        thresholds={
            "difficult_severe_max": 3,
            "control_severe_max": 1,
            "e0_mean_min": 0.75,
            "edev_delta_min": 0.0,
            "repeat_count": 3,
            "bitwise_identical_required": True,
            "max_one_per_family": True,
        },
        evaluated=evaluated,
        selected=selected,
        verdict="continue" if selected else "stop_zero_candidates",
        extra={"diagnose_manifest": diagnose},
    )


def build_b_gate_contract(
    context: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    bound_context = _validate_gate_context(context)
    if len(arms) > 3:
        raise CampaignContractError("B-stage accepts at most three candidates")
    evaluated = [
        _evaluate_quality_arm(
            arm,
            expected_seeds=R9_CALIBRATION_SEEDS,
            severe_limit=3,
            bootstrap_seed=bootstrap_seed,
            reject_repeated_severe=True,
        )
        for arm in arms
    ]
    passing = [row for row in evaluated if row["passed"]]
    passing.sort(key=_quality_rank_key)
    selected = [row["arm_id"] for row in passing[:2]]
    return _build_gate_payload(
        phase="calibrate",
        context=bound_context,
        seeds=list(R9_CALIBRATION_SEEDS),
        thresholds=_quality_thresholds(severe_limit=3),
        evaluated=evaluated,
        selected=selected,
        verdict="continue" if selected else "stop_zero_candidates",
    )


def build_c_gate_contract(
    context: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    *,
    confirm_seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    bound_context = _validate_gate_context(context)
    _require_nonnegative_int(confirm_seed, "confirm512 seed")
    if len(arms) > 2:
        raise CampaignContractError("C-stage accepts at most two candidates")
    evaluated = [
        _evaluate_quality_arm(
            arm,
            expected_seeds=(confirm_seed,),
            severe_limit=25,
            bootstrap_seed=bootstrap_seed,
            reject_repeated_severe=False,
        )
        for arm in arms
    ]
    passing = [row for row in evaluated if row["passed"]]
    passing.sort(key=_quality_rank_key)
    selected = [passing[0]["arm_id"]] if passing else []
    return _build_gate_payload(
        phase="confirm512",
        context=bound_context,
        seeds=[confirm_seed],
        thresholds=_quality_thresholds(severe_limit=25),
        evaluated=evaluated,
        selected=selected,
        verdict="winner_locked" if selected else "stop_zero_candidates",
    )


def build_selection_contract(
    confirm_gate: Mapping[str, Any],
    *,
    manifest_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    gate = validate_gate_contract(confirm_gate)
    if gate["phase"] != "confirm512" or gate["verdict"] != "winner_locked":
        raise CampaignContractError("selection requires a passing confirm512 gate")
    selected = gate.get("selected_arm_ids")
    if not isinstance(selected, list) or len(selected) != 1:
        raise CampaignContractError("confirm512 gate must select exactly one winner")
    winner_id = selected[0]
    matches = [row for row in gate["arms"] if row.get("arm_id") == winner_id]
    if len(matches) != 1 or matches[0].get("passed") is not True:
        raise CampaignContractError("selected winner is not a unique passing arm")
    if set(manifest_sha256s) != R9_MANIFEST_KEYS:
        raise CampaignContractError("selection must bind all five manifests")
    manifests = {
        name: _require_sha256(manifest_sha256s[name], f"{name} SHA256")
        for name in sorted(R9_MANIFEST_KEYS)
    }
    winner = matches[0]
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_selection_v1",
        "campaign_id": gate["context"]["campaign_id"],
        "campaign_runtime_sha256": gate["context"]["campaign_runtime_sha256"],
        "gate_contract_sha256": gate["gate_contract_sha256"],
        "winner": {
            "arm_id": winner_id,
            "config_sha256": winner["config_sha256"],
            "output_sha256": winner["output_sha256"],
        },
        "manifests": manifests,
        "winner_locked": True,
        "reselection_allowed": False,
    }
    payload["selection_sha256"] = _canonical_contract_sha256(
        payload, "selection_sha256"
    )
    return payload


def validate_selection_contract(
    selection: Mapping[str, Any], confirm_gate: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = _json_mapping(selection, "selection contract")
    _verify_contract_digest(normalized, "selection_sha256")
    gate = validate_gate_contract(confirm_gate)
    expected = build_selection_contract(
        gate, manifest_sha256s=normalized.get("manifests", {})
    )
    if normalized != expected:
        raise CampaignContractError("selection disagrees with its confirm512 gate")
    if set(normalized) != {
        "schema_version",
        "contract_type",
        "campaign_id",
        "campaign_runtime_sha256",
        "gate_contract_sha256",
        "winner",
        "manifests",
        "winner_locked",
        "reselection_allowed",
        "selection_sha256",
    }:
        raise CampaignContractError("selection contains non-winner fields")
    return normalized


def build_heldout_seal_contract(
    selection: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    _verify_contract_digest(selection, "selection_sha256")
    if selection.get("winner_locked") is not True:
        raise CampaignContractError("held-out seal requires a locked winner")
    if set(assets) != R9_HELDOUT_ASSETS:
        raise CampaignContractError("held-out seal requires E1/E2/FaceNet/AdaFace")
    normalized_assets = {}
    for name in sorted(R9_HELDOUT_ASSETS):
        asset = assets[name]
        if not isinstance(asset, Mapping) or set(asset) != {"path", "sha256"}:
            raise CampaignContractError(f"held-out asset {name!r} fields mismatch")
        normalized_assets[name] = {
            "path": _require_nonempty(asset.get("path"), f"{name} path"),
            "sha256": _require_sha256(asset.get("sha256"), f"{name} SHA256"),
            "state": "sealed_unrun",
        }
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_heldout_seal_v1",
        "selection_sha256": selection["selection_sha256"],
        "winner": dict(selection["winner"]),
        "assets": normalized_assets,
        "execution_count": 0,
        "sealed": True,
    }
    payload["heldout_seal_sha256"] = _canonical_contract_sha256(
        payload, "heldout_seal_sha256"
    )
    return payload


def build_d_gate_contract(
    context: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    heldout_seal: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    bound_context = _validate_gate_context(context)
    _verify_contract_digest(selection, "selection_sha256")
    _verify_contract_digest(heldout_seal, "heldout_seal_sha256")
    if heldout_seal.get("selection_sha256") != selection.get("selection_sha256"):
        raise CampaignContractError("held-out seal does not bind selection")
    if (
        heldout_seal.get("execution_count") != 0
        or heldout_seal.get("sealed") is not True
    ):
        raise CampaignContractError("held-out assets were not sealed before Full")
    if not isinstance(result, Mapping):
        raise CampaignContractError("Full result must be a mapping")
    if result.get("execution_count") != 1:
        raise CampaignContractError("held-out evaluators must run exactly once")
    winner = selection.get("winner")
    if not isinstance(winner, Mapping):
        raise CampaignContractError("selection winner is invalid")
    if result.get("winner_arm_id") != winner.get("arm_id"):
        raise CampaignContractError("Full result changed the locked winner")
    if result.get("config_sha256") != winner.get("config_sha256"):
        raise CampaignContractError("Full result changed winner config")
    _require_sha256(result.get("output_sha256"), "Full output SHA256")
    visual_severe = _require_nonnegative_int(
        result.get("full_visual_severe_count"), "Full visual severe count"
    )
    failures = []
    if visual_severe > 3:
        failures.append("full_visual_severe_count_gt_3")
    representations = result.get("representations")
    if not isinstance(representations, Mapping) or set(representations) != {"e1", "e2"}:
        raise CampaignContractError("Full result requires E1 and E2 representations")
    normalized_representations = {}
    for name in ("e1", "e2"):
        row = representations[name]
        if not isinstance(row, Mapping):
            raise CampaignContractError(f"{name} result must be a mapping")
        winner_mean = _finite_float(row.get("winner_mean"), f"{name} winner mean")
        native_mean = _finite_float(row.get("native_mean"), f"{name} native mean")
        lower = _finite_float(
            row.get("paired_bootstrap_lower_95"), f"{name} bootstrap lower"
        )
        if winner_mean <= native_mean:
            failures.append(f"{name}_winner_mean_not_above_native")
        if lower <= 0:
            failures.append(f"{name}_bootstrap_lower_not_positive")
        normalized_representations[name] = {
            "winner_mean": winner_mean,
            "native_mean": native_mean,
            "paired_bootstrap_lower_95": lower,
        }
    recognizers = result.get("recognizers")
    required_recognizers = {"arcface", "facenet", "adaface"}
    if not isinstance(recognizers, Mapping) or set(recognizers) != required_recognizers:
        raise CampaignContractError("Full result requires three identity recognizers")
    normalized_recognizers = {}
    for name in sorted(required_recognizers):
        row = recognizers[name]
        if not isinstance(row, Mapping):
            raise CampaignContractError(f"{name} recognizer result must be a mapping")
        coverage = _strict_int(row.get("coverage"), f"{name} coverage")
        if not 0 <= coverage <= 2048:
            raise CampaignContractError(f"{name} coverage must be in [0, 2048]")
        upper_value = row.get("privacy_delta_upper_95")
        bootstrap_value = row.get("bootstrap_sha256")
        if coverage == 2048:
            upper = _finite_float(upper_value, f"{name} privacy upper")
            bootstrap_sha = _require_sha256(bootstrap_value, f"{name} bootstrap SHA256")
            if upper > 0.02:
                failures.append(f"{name}_privacy_upper_gt_0.02")
        else:
            if upper_value is not None or bootstrap_value is not None:
                raise CampaignContractError(
                    f"{name} incomplete coverage forbids a partial privacy bootstrap"
                )
            upper = None
            bootstrap_sha = None
            failures.append(f"{name}_coverage_not_2048")
        normalized_recognizers[name] = {
            "coverage": coverage,
            "privacy_delta_upper_95": upper,
            "bootstrap_sha256": bootstrap_sha,
        }
    quality = _evaluate_seed_quality(result.get("quality"), severe_limit=2048)
    failures.extend(
        f"quality:{item}"
        for item in quality["failures"]
        if not item.startswith("severe")
    )
    metrics_report = validate_identity_report(
        result.get("identity_report"), expected_count=2048
    )
    for name in R9_IDENTITY_RECOGNIZERS:
        report_row = metrics_report["recognizers"][name]
        if report_row["coverage"] != normalized_recognizers[name]["coverage"]:
            raise CampaignContractError(
                f"{name} identity report coverage disagrees with recognizer evidence"
            )
        if report_row["status"] == "unavailable":
            failures.append(f"{name}_identity_report_unavailable")
    evaluated = [
        {
            "arm_id": winner["arm_id"],
            "config_sha256": winner["config_sha256"],
            "output_sha256": result["output_sha256"],
            "full_visual_severe_count": visual_severe,
            "representations": normalized_representations,
            "recognizers": normalized_recognizers,
            "quality": quality,
            "identity_report": metrics_report,
            "heldout_execution_count": 1,
            "failures": failures,
            "passed": not failures,
        }
    ]
    passed = not failures
    return _build_gate_payload(
        phase="full",
        context=bound_context,
        seeds=[_strict_int(result.get("seed"), "Full seed")],
        thresholds={
            **_quality_thresholds(severe_limit=3),
            "full_visual_severe_max": 3,
            "representation_bootstrap_lower_min_exclusive": 0.0,
            "privacy_delta_upper_max": 0.02,
            "identity_coverage": 2048,
            "heldout_execution_count": 1,
        },
        evaluated=evaluated,
        selected=[winner["arm_id"]],
        verdict="passed_locked_winner" if passed else "failed_locked_winner",
        extra={
            "selection_sha256": selection["selection_sha256"],
            "heldout_seal_sha256": heldout_seal["heldout_seal_sha256"],
            "reselection_allowed": False,
        },
    )


def validate_identity_report(value: Any, *, expected_count: int) -> dict[str, Any]:
    """Validate the complete Full identity report without inventing unavailable metrics."""
    _require_nonnegative_int(expected_count, "identity report expected count")
    if expected_count == 0:
        raise CampaignContractError("identity report expected count must be positive")
    report = _json_mapping(value, "identity report")
    if (
        set(report) != {"schema_version", "recognizers"}
        or report.get("schema_version") != 1
    ):
        raise CampaignContractError("identity report fields are not canonical")
    recognizers = _json_mapping(report.get("recognizers"), "identity recognizers")
    if set(recognizers) != set(R9_IDENTITY_RECOGNIZERS):
        raise CampaignContractError(
            "identity report requires exactly ArcFace, FaceNet and AdaFace"
        )
    normalized: dict[str, Any] = {}
    for name in R9_IDENTITY_RECOGNIZERS:
        row = _json_mapping(recognizers[name], f"{name} identity report")
        if set(row) != {"status", "reason", "coverage", "roles"}:
            raise CampaignContractError(
                f"{name} identity report fields are not canonical"
            )
        coverage = _strict_int(row.get("coverage"), f"{name} identity coverage")
        if not 0 <= coverage <= expected_count:
            raise CampaignContractError(
                f"{name} identity coverage must be in [0, {expected_count}]"
            )
        status = row.get("status")
        reason = row.get("reason")
        roles = _json_mapping(row.get("roles"), f"{name} identity roles")
        if set(roles) != set(R9_IDENTITY_ROLES):
            raise CampaignContractError(
                f"{name} identity report requires native and winner roles"
            )
        if coverage == expected_count:
            if status != "available" or reason is not None:
                raise CampaignContractError(
                    f"{name} complete identity coverage must be available"
                )
            normalized_roles = {
                role: _validate_available_identity_role(
                    roles[role], name=name, role=role
                )
                for role in R9_IDENTITY_ROLES
            }
        else:
            if status != "unavailable" or not isinstance(reason, str) or not reason:
                raise CampaignContractError(
                    f"{name} incomplete identity coverage requires an unavailable reason"
                )
            normalized_roles = {
                role: _validate_unavailable_identity_role(
                    roles[role], name=name, role=role, reason=reason
                )
                for role in R9_IDENTITY_ROLES
            }
        normalized[name] = {
            "status": status,
            "reason": reason,
            "coverage": coverage,
            "roles": normalized_roles,
        }
    return {"schema_version": 1, "recognizers": normalized}


def _validate_available_identity_role(
    value: Any, *, name: str, role: str
) -> dict[str, Any]:
    row = _json_mapping(value, f"{name} {role} identity metrics")
    if (
        set(row) != {"status", "tar_at_far", "eer", "auc"}
        or row.get("status") != "available"
    ):
        raise CampaignContractError(
            f"{name} {role} available identity fields are not canonical"
        )
    tar = _json_mapping(row.get("tar_at_far"), f"{name} {role} TAR@FAR")
    if set(tar) != set(R9_TAR_FAR_KEYS):
        raise CampaignContractError(
            f"{name} {role} TAR@FAR requires exactly 0.001 and 0.0001"
        )
    normalized_tar = {
        key: _unit_interval(tar[key], f"{name} {role} TAR@FAR {key}")
        for key in R9_TAR_FAR_KEYS
    }
    return {
        "status": "available",
        "tar_at_far": normalized_tar,
        "eer": _unit_interval(row.get("eer"), f"{name} {role} EER"),
        "auc": _unit_interval(row.get("auc"), f"{name} {role} AUC"),
    }


def _validate_unavailable_identity_role(
    value: Any, *, name: str, role: str, reason: str
) -> dict[str, Any]:
    row = _json_mapping(value, f"{name} {role} unavailable identity result")
    if set(row) != {"status", "reason"} or row.get("status") != "unavailable":
        raise CampaignContractError(
            f"{name} {role} unavailable identity fields are not canonical"
        )
    if row.get("reason") != reason:
        raise CampaignContractError(
            f"{name} {role} unavailable reason disagrees with recognizer"
        )
    return {"status": "unavailable", "reason": reason}


def _unit_interval(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise CampaignContractError(f"{label} must be in [0, 1]")
    return parsed


def validate_gate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_mapping(contract, "gate contract")
    _verify_contract_digest(normalized, "gate_contract_sha256")
    if (
        normalized.get("schema_version") != 1
        or normalized.get("contract_type") != "safa_r9_gate_contract_v1"
    ):
        raise CampaignContractError("gate contract type mismatch")
    phase = normalized.get("phase")
    selected = normalized.get("selected_arm_ids")
    arms = normalized.get("arms")
    if phase not in {"diagnose", "calibrate", "confirm512", "full"}:
        raise CampaignContractError("gate phase is invalid")
    if not isinstance(selected, list) or len(set(selected)) != len(selected):
        raise CampaignContractError("gate selected_arm_ids are invalid")
    if not isinstance(arms, list):
        raise CampaignContractError("gate arms must be a list")
    passing = {row.get("arm_id") for row in arms if row.get("passed") is True}
    arm_ids = {row.get("arm_id") for row in arms}
    if phase != "full" and not set(selected) <= passing:
        raise CampaignContractError("gate selected a failing or missing arm")
    if phase == "full" and not set(selected) <= arm_ids:
        raise CampaignContractError("Full gate lost its locked winner arm")
    if phase == "diagnose":
        families = [row.get("family") for row in arms if row.get("arm_id") in selected]
        if len(selected) > 3 or len(set(families)) != len(families):
            raise CampaignContractError("A gate violates one-per-family selection")
        expected_verdict = "continue" if selected else "stop_zero_candidates"
    elif phase == "calibrate":
        if len(selected) > 2:
            raise CampaignContractError("B gate selected more than two candidates")
        expected_verdict = "continue" if selected else "stop_zero_candidates"
    elif phase == "confirm512":
        if len(selected) > 1:
            raise CampaignContractError("C gate selected more than one winner")
        expected_verdict = "winner_locked" if selected else "stop_zero_candidates"
    elif phase == "full":
        if len(selected) != 1 or normalized.get("reselection_allowed") is not False:
            raise CampaignContractError("Full gate must retain one locked winner")
        expected_verdict = (
            "passed_locked_winner"
            if arms[0].get("passed") is True
            else "failed_locked_winner"
        )
    if normalized.get("verdict") != expected_verdict:
        raise CampaignContractError("gate verdict disagrees with evaluated selection")
    return normalized


def write_immutable_contract(
    path: Path, payload: Mapping[str, Any], *, digest_field: str
) -> Path:
    normalized = _json_mapping(payload, "immutable contract")
    _verify_contract_digest(normalized, digest_field)
    destination = Path(path)
    if destination.is_symlink():
        raise CampaignContractError("immutable contract path must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise CampaignContractError(
                "immutable contract already exists with other content"
            )
        return destination
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != content:
                raise CampaignContractError(
                    "concurrent immutable contract disagrees with payload"
                )
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_bound_file(value: Any, *, repo_root: Path, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise CampaignContractError(f"{label} must contain only path and sha256")
    path = _require_repo_path(value.get("path"), repo_root, f"{label} path")
    if path.is_symlink() or not path.is_file():
        raise CampaignContractError(f"{label} must be an existing regular file")
    declared = _require_sha256(value.get("sha256"), f"{label} SHA256")
    if _sha256_file(path) != declared:
        raise CampaignContractError(f"{label} file SHA256 mismatch")
    return {"path": str(path.relative_to(repo_root)), "sha256": declared}


def _validate_clean_source_contract(value: Any, repo_root: Path) -> dict[str, Any]:
    required = {
        "path",
        "sha256",
        "sample_count",
        "ordered_sample_id_sha256",
        "arcface_exact_one",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CampaignContractError("clean_source fields are not canonical")
    entry, _ = _validate_manifest_entry(
        name="clean_source",
        declared={key: value[key] for key in required - {"arcface_exact_one"}},
        repo_root=repo_root,
        expected_count=None,
    )
    if value.get("arcface_exact_one") is not True:
        raise CampaignContractError("clean_source must certify arcface_exact_one")
    return {**entry, "arcface_exact_one": True}


def _validate_diagnose_manifest_file(
    value: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], list[str]]:
    declared = _validate_diagnose_manifest(value)
    entry, sample_ids = _validate_manifest_entry(
        name="diagnose_18",
        declared={
            key: declared[key]
            for key in (
                "path",
                "sha256",
                "sample_count",
                "ordered_sample_id_sha256",
            )
        },
        repo_root=repo_root,
        expected_count=18,
    )
    rows = _read_manifest_rows(repo_root / entry["path"])
    by_pair: dict[int, dict[str, Mapping[str, Any]]] = {}
    for line_number, row in enumerate(rows, 1):
        pair_index = _strict_int(
            row.get("pair_index"), f"diagnose row {line_number} pair_index"
        )
        role = row.get("role")
        if pair_index not in range(9) or role not in {"difficult", "control"}:
            raise CampaignContractError("diagnose pair_index/role contract mismatch")
        pair = by_pair.setdefault(pair_index, {})
        if role in pair:
            raise CampaignContractError("diagnose pair repeats a role")
        _finite_float(row.get("native_e0_cosine"), "diagnose native E0 cosine")
        pair[str(role)] = row
    if set(by_pair) != set(range(9)) or any(
        set(pair) != {"difficult", "control"} for pair in by_pair.values()
    ):
        raise CampaignContractError(
            "diagnose manifest must contain nine complete pairs"
        )
    pairs = []
    for pair_index in range(9):
        difficult = by_pair[pair_index]["difficult"]
        control = by_pair[pair_index]["control"]
        difficult_id = str(difficult["sample_id"])
        control_id = str(control["sample_id"])
        if (
            difficult.get("matched_control_sample_id") != control_id
            or control.get("matched_difficult_sample_id") != difficult_id
        ):
            raise CampaignContractError("diagnose matched-pair links are not symmetric")
        pairs.append(
            {
                "pair_index": pair_index,
                "difficult_sample_id": difficult_id,
                "control_sample_id": control_id,
                "difficult_native_e0_cosine": _finite_float(
                    difficult["native_e0_cosine"], "difficult native E0 cosine"
                ),
                "control_native_e0_cosine": _finite_float(
                    control["native_e0_cosine"], "control native E0 cosine"
                ),
            }
        )
    pair_sha = _canonical_json_sha256({"schema_version": 1, "pairs": pairs})
    if pair_sha != declared["matched_pair_sha256"]:
        raise CampaignContractError("diagnose matched-pair digest mismatch")
    return (
        {
            **entry,
            **{
                key: declared[key]
                for key in ("difficult_count", "control_count", "matched_pair_sha256")
            },
        },
        sample_ids,
    )


def _validate_schedule_and_gate(
    schedule_value: Any,
    gate_value: Any,
    *,
    repo_root: Path,
    checkpoint_sha256: str,
    calibration_manifest_sha256: str,
) -> tuple[dict[str, str], dict[str, str]]:
    schedule, schedule_payload = _validate_bound_json_contract(
        schedule_value, repo_root=repo_root, label="schedule"
    )
    gate, gate_payload = _validate_bound_json_contract(
        gate_value, repo_root=repo_root, label="semigroup gate"
    )
    if (
        schedule_payload.get("schema_version") != 3
        or schedule_payload.get("gate_passed") is not True
    ):
        raise CampaignContractError("locked schedule must be passing schema v3")
    if (
        gate_payload.get("contract_type") != "safa_r9_semigroup_gate_v1"
        or gate_payload.get("gate_passed") is not True
    ):
        raise CampaignContractError("semigroup gate must be a passing R9 gate")
    if gate_payload.get("experiment_contract") != R9_GENERATION_EXPERIMENT_CONTRACT:
        raise CampaignContractError("semigroup gate generation contract mismatch")
    if gate_payload.get("determinism_policy_sha256") != R9_DETERMINISM_POLICY_SHA256:
        raise CampaignContractError("semigroup gate determinism policy mismatch")
    if (
        gate_payload.get("attention_backend_requested") != "native"
        or gate_payload.get("attention_backend_resolved") != "native"
    ):
        raise CampaignContractError("semigroup gate attention backend mismatch")
    if (
        gate_payload.get("checkpoint_sha256") != checkpoint_sha256
        or schedule_payload.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise CampaignContractError("schedule/gate checkpoint binding mismatch")
    if (
        gate_payload.get("sample_id_manifest_sha256") != calibration_manifest_sha256
        or schedule_payload.get("semigroup_sample_id_manifest_sha256")
        != calibration_manifest_sha256
    ):
        raise CampaignContractError("schedule/gate calibration manifest mismatch")
    if schedule_payload.get("schedule_contract_sha256") != schedule["contract_sha256"]:
        raise CampaignContractError("schedule canonical contract SHA256 mismatch")
    if gate_payload.get("schedule_contract_sha256") != schedule["contract_sha256"]:
        raise CampaignContractError("semigroup gate does not bind locked schedule")
    if (
        gate_payload.get("gate_contract_sha256") != gate["contract_sha256"]
        or _canonical_contract_sha256(gate_payload, "gate_contract_sha256")
        != gate["contract_sha256"]
    ):
        raise CampaignContractError("semigroup gate canonical digest mismatch")
    gate_path = _require_repo_path(
        schedule_payload.get("r9_semigroup_gate_contract"),
        repo_root,
        "schedule semigroup gate path",
    )
    if (
        gate_path != repo_root / gate["path"]
        or schedule_payload.get("r9_semigroup_gate_contract_sha256")
        != gate["file_sha256"]
    ):
        raise CampaignContractError("schedule semigroup gate file binding mismatch")
    return schedule, gate


def _validate_bound_json_contract(
    value: Any, *, repo_root: Path, label: str
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "contract_sha256",
    }:
        raise CampaignContractError(f"{label} fields are not canonical")
    path = _require_repo_path(value.get("path"), repo_root, f"{label} path")
    if path.is_symlink() or not path.is_file():
        raise CampaignContractError(f"{label} must be an existing regular file")
    file_sha = _require_sha256(value.get("file_sha256"), f"{label} file SHA256")
    contract_sha = _require_sha256(
        value.get("contract_sha256"), f"{label} contract SHA256"
    )
    if _sha256_file(path) != file_sha:
        raise CampaignContractError(f"{label} file SHA256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CampaignContractError(f"{label} file is not valid JSON") from error
    if not isinstance(payload, dict):
        raise CampaignContractError(f"{label} JSON must be an object")
    return (
        {
            "path": str(path.relative_to(repo_root)),
            "file_sha256": file_sha,
            "contract_sha256": contract_sha,
        },
        payload,
    )


def _validate_runtime_resources(
    value: Any,
    *,
    repo_root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    expected = {
        "physical_gpus": [0, 1, 2, 3],
        "global_slot_lock_root": "/tmp/safa-r9-gpu-slots-v1",
        "max_slots_per_gpu": 4,
        "gpu_slot_claim_bytes": 4_938_792_960,
        "gpu_headroom_bytes": 2 * 1024**3,
        "ram_admission_percent": 85,
        "ram_hard_limit_percent": 90,
        "require_tmux": True,
        "retry_count": 0,
    }
    if not isinstance(value, Mapping) or set(value) != {
        *expected,
        "resource_smoke",
        "ram_slot_budget_bytes",
    }:
        raise CampaignContractError("runtime resources fields are not canonical")
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise CampaignContractError(f"runtime resource {field} mismatch")
    declaration = value.get("resource_smoke")
    required_declaration = {
        "required",
        "run_id",
        "arm_id",
        "manifest",
        "output_path",
        "factor",
        "result",
    }
    if not isinstance(declaration, Mapping) or set(declaration) != required_declaration:
        raise CampaignContractError(
            "resource_smoke declaration fields are not canonical"
        )
    if declaration.get("required") is not True or declaration.get("factor") != 1.10:
        raise CampaignContractError("resource_smoke must be required with factor 1.10")
    run_id = _require_nonempty(declaration.get("run_id"), "resource smoke run_id")
    arm_id = _require_nonempty(declaration.get("arm_id"), "resource smoke arm_id")
    manifest = _require_nonempty(declaration.get("manifest"), "resource smoke manifest")
    if manifest not in manifests or manifest == "arcface_clean_pool":
        raise CampaignContractError("resource smoke must bind a generation manifest")
    output_path = _require_repo_path(
        declaration.get("output_path"), repo_root, "resource smoke output path"
    )
    result_binding, result_payload = _validate_bound_json_contract(
        declaration.get("result"),
        repo_root=repo_root,
        label="resource smoke result",
    )
    result = validate_resource_smoke_contract(result_payload)
    if result_binding["contract_sha256"] != result["resource_smoke_sha256"]:
        raise CampaignContractError("resource smoke result contract SHA256 mismatch")
    if output_path != repo_root / result_binding["path"]:
        raise CampaignContractError("resource smoke output/result path mismatch")
    expected_bindings = {
        "run_id": run_id,
        "arm_id": arm_id,
        "manifest": manifest,
        "manifest_sha256": manifests[manifest]["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
    }
    for field, expected_value in expected_bindings.items():
        if result.get(field) != expected_value:
            raise CampaignContractError(f"resource smoke {field} binding mismatch")
    peak_rss = result["peak_rss_bytes"]
    expected_budget = (peak_rss * 110 + 99) // 100
    budget = _strict_int(value.get("ram_slot_budget_bytes"), "RAM slot budget bytes")
    if budget != expected_budget:
        raise CampaignContractError("RAM slot budget must equal ceil(smoke RSS * 1.10)")
    return {
        **expected,
        "resource_smoke": {
            "required": True,
            "run_id": run_id,
            "arm_id": arm_id,
            "manifest": manifest,
            "output_path": str(output_path.relative_to(repo_root)),
            "factor": 1.10,
            "result": {
                **result_binding,
                "peak_rss_bytes": peak_rss,
            },
        },
        "ram_slot_budget_bytes": budget,
    }


def _validate_runtime_evaluation(value: Any, *, repo_root: Path) -> dict[str, Any]:
    evaluation = _json_mapping(value, "evaluation")
    if set(evaluation) != {
        "worker",
        "quality",
        "arcface",
        "heldout",
        "resource_smokes",
    }:
        raise CampaignContractError("evaluation fields are not canonical")
    worker_declared = _json_mapping(evaluation.get("worker"), "evaluation worker")
    if set(worker_declared) != {
        "path",
        "sha256",
        "implementation_path",
        "implementation_sha256",
    }:
        raise CampaignContractError("evaluation worker fields are not canonical")
    worker_wrapper = _validate_bound_file(
        {key: worker_declared[key] for key in ("path", "sha256")},
        repo_root=repo_root,
        label="evaluation worker wrapper",
    )
    worker_implementation = _validate_bound_file(
        {
            "path": worker_declared["implementation_path"],
            "sha256": worker_declared["implementation_sha256"],
        },
        repo_root=repo_root,
        label="evaluation worker implementation",
    )
    worker = {
        **worker_wrapper,
        "implementation_path": worker_implementation["path"],
        "implementation_sha256": worker_implementation["sha256"],
    }
    quality = _json_mapping(evaluation.get("quality"), "quality evaluation")
    if set(quality) != {"script", "real_index", "metrics", "iqa_method", "device"}:
        raise CampaignContractError("quality evaluation fields are not canonical")
    if (
        quality.get("metrics") != ["fid", "kid", "niqe", "sharpness"]
        or quality.get("iqa_method") != "niqe"
        or quality.get("device") != "cuda:0"
    ):
        raise CampaignContractError("quality evaluator settings are not locked")
    normalized_quality = {
        "script": _validate_bound_file(
            quality.get("script"), repo_root=repo_root, label="quality script"
        ),
        "real_index": _validate_bound_file(
            quality.get("real_index"),
            repo_root=repo_root,
            label="quality real index",
        ),
        "metrics": ["fid", "kid", "niqe", "sharpness"],
        "iqa_method": "niqe",
        "device": "cuda:0",
    }
    arcface = _json_mapping(evaluation.get("arcface"), "ArcFace evaluation")
    if set(arcface) != {
        "model_name",
        "model_root",
        "det_size",
        "provider",
        "insightface_version",
        "onnxruntime_version",
        "assets",
        "execution_probe",
    }:
        raise CampaignContractError("ArcFace evaluation fields are not canonical")
    if (
        arcface.get("model_name") != "buffalo_l"
        or arcface.get("det_size") != [224, 224]
        or arcface.get("provider") != "CUDAExecutionProvider"
        or arcface.get("insightface_version") != "0.7.3"
        or arcface.get("onnxruntime_version") != "1.26.0"
    ):
        raise CampaignContractError("ArcFace evaluator settings are not locked")
    model_root = Path(
        _require_nonempty(arcface.get("model_root"), "ArcFace model root")
    )
    if not model_root.is_absolute() or not model_root.resolve().is_dir():
        raise CampaignContractError(
            "ArcFace model root must be an existing absolute path"
        )
    assets = _json_mapping(arcface.get("assets"), "ArcFace assets")
    expected_assets = {
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    }
    if set(assets) != expected_assets:
        raise CampaignContractError("ArcFace buffalo_l assets are incomplete")
    model_dir = model_root.resolve() / "models" / "buffalo_l"
    normalized_assets = {}
    for name in sorted(expected_assets):
        digest = _require_sha256(assets.get(name), f"ArcFace asset {name} SHA256")
        path = model_dir / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise CampaignContractError(f"ArcFace asset {name} SHA256 mismatch")
        normalized_assets[name] = digest
    execution_probe = _json_mapping(
        arcface.get("execution_probe"), "ArcFace execution probe provenance"
    )
    if set(execution_probe) != {
        "path",
        "sha256",
        "bootstrap_claim_path",
        "bootstrap_claim_file_sha256",
        "bootstrap_claim_sha256",
        "bootstrap_result_path",
        "bootstrap_result_file_sha256",
        "bootstrap_result_sha256",
    }:
        raise CampaignContractError(
            "ArcFace execution probe provenance fields are not canonical"
        )
    probe_binding = _validate_bound_file(
        {"path": execution_probe["path"], "sha256": execution_probe["sha256"]},
        repo_root=repo_root,
        label="ArcFace execution probe",
    )
    try:
        probe_payload = json.loads(
            (repo_root / probe_binding["path"]).read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise CampaignContractError(
            "ArcFace execution probe is not valid JSON"
        ) from exc
    if not isinstance(probe_payload, dict):
        raise CampaignContractError("ArcFace execution probe is not an object")
    if not isinstance(probe_payload.get("execution"), Mapping):
        raise CampaignContractError("ArcFace execution probe omitted execution")
    try:
        from safa.evaluation.r9_evaluator_worker import (
            R9EvaluatorError,
            _validate_arcface_contract,
        )

        normalized_arcface = _validate_arcface_contract(
            {
                "model_name": "buffalo_l",
                "model_root": str(model_root.resolve()),
                "det_size": [224, 224],
                "provider": "CUDAExecutionProvider",
                "insightface_version": "0.7.3",
                "onnxruntime_version": "1.26.0",
                "assets": normalized_assets,
                "execution": probe_payload["execution"],
                "execution_probe": execution_probe,
            },
            repo_root=repo_root,
        )
    except (R9EvaluatorError, FileNotFoundError) as exc:
        raise CampaignContractError(str(exc)) from exc
    heldout = _json_mapping(evaluation.get("heldout"), "heldout evaluation")
    if set(heldout) != {
        "batch_size",
        "representation_image_size",
        "facenet",
        "adaface",
    }:
        raise CampaignContractError("heldout evaluation fields are not canonical")
    if (
        heldout.get("batch_size") != 16
        or heldout.get("representation_image_size") != 224
    ):
        raise CampaignContractError("heldout evaluator batch/image size mismatch")
    recognizers = {
        "facenet": {"embedding_dim": 512, "input_size": 160},
        "adaface": {"embedding_dim": 512, "input_size": 112},
    }
    for name, expected in recognizers.items():
        if _json_mapping(heldout.get(name), name) != expected:
            raise CampaignContractError(f"heldout {name} settings mismatch")
    try:
        resource_smokes = validate_evaluator_resource_profiles(
            evaluation.get("resource_smokes"),
            repo_root=repo_root,
            worker_contract=worker,
            arcface_contract_sha256=_canonical_json_sha256(normalized_arcface),
            quality_script_sha256=normalized_quality["script"]["sha256"],
        )
    except EvaluatorResourceContractError as exc:
        raise CampaignContractError(str(exc)) from exc
    return {
        "worker": worker,
        "quality": normalized_quality,
        "arcface": normalized_arcface,
        "heldout": {
            "batch_size": 16,
            "representation_image_size": 224,
            **recognizers,
        },
        "resource_smokes": resource_smokes,
    }


def _validate_runtime_bootstrap(value: Any) -> dict[str, Any]:
    expected = {
        "resamples": 10_000,
        "confidence": 0.95,
        "cluster": "sample_id",
        "identity_delta_direction": "source_candidate_minus_source_native",
    }
    if not isinstance(value, Mapping) or set(value) != {*expected, "seed"}:
        raise CampaignContractError("runtime bootstrap fields are not canonical")
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise CampaignContractError(f"runtime bootstrap {field} mismatch")
    seed = _require_nonnegative_int(value.get("seed"), "bootstrap seed")
    return {**expected, "seed": seed}


def _validate_runtime_phases(value: Any, *, seeds: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "preflight",
        "diagnose",
        "calibrate",
        "confirm512",
        "full",
    }:
        raise CampaignContractError("runtime phases must contain exactly five phases")
    phases = _json_mapping(value, "runtime phases")
    _require_exact_mapping(
        phases["preflight"],
        {
            "manifest": "calibration_64",
            "sample_count": 64,
            "shards_per_logical_run": 4,
            "seed": 1337,
        },
        "preflight phase",
    )
    diagnose = phases["diagnose"]
    if not isinstance(diagnose, Mapping) or set(diagnose) != {
        "manifest",
        "sample_count",
        "shards_per_logical_run",
        "seed",
        "repeats",
        "determinism_repeats_must_match",
        "arms",
        "gate",
    }:
        raise CampaignContractError("diagnose phase fields are not canonical")
    for field, expected in {
        "manifest": "diagnose_18",
        "sample_count": 18,
        "shards_per_logical_run": 1,
        "seed": 1337,
        "repeats": 3,
        "determinism_repeats_must_match": 3,
    }.items():
        if diagnose.get(field) != expected:
            raise CampaignContractError(f"diagnose phase {field} mismatch")
    arms = diagnose.get("arms")
    if not isinstance(arms, list) or len(arms) != 13:
        raise CampaignContractError("diagnose phase must register exactly 13 arms")
    arm_ids = []
    family_counts: dict[str, int] = {}
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise CampaignContractError("diagnose arm must be a mapping")
        arm_id = _require_nonempty(arm.get("arm_id"), "diagnose arm_id")
        family = _require_nonempty(arm.get("family"), "diagnose arm family")
        arm_ids.append(arm_id)
        family_counts[family] = family_counts.get(family, 0) + 1
    if len(set(arm_ids)) != 13 or family_counts != {
        "native": 1,
        "flow_map2": 3,
        "paper_split_constant": 6,
        "paper_split_interval_ablation": 3,
    }:
        raise CampaignContractError("diagnose arm family/count contract mismatch")
    _require_gate_values(
        diagnose.get("gate"),
        {
            "difficult_severe_max": 3,
            "control_severe_max": 1,
            "e0_mean_min": 0.75,
            "edev_vs_matched_native_min": 0.0,
            "require_finite_diagnostics": True,
            "max_candidates_per_family": 1,
            "family_order": [
                "flow_map2",
                "paper_split_constant",
                "paper_split_interval_ablation",
            ],
            "rank_order": ["severe", "edev_desc", "e0_desc", "arm_id"],
        },
        "diagnose gate",
    )
    calibrate = phases["calibrate"]
    if not isinstance(calibrate, Mapping) or set(calibrate) != {
        "manifest",
        "sample_count",
        "shards_per_logical_run",
        "seeds",
        "candidate_slots",
        "collect_interval_diagnostics",
        "gate",
    }:
        raise CampaignContractError("calibrate phase fields are not canonical")
    expected_calibrate = {
        "manifest": "calibration_64",
        "sample_count": 64,
        "shards_per_logical_run": 1,
        "seeds": list(R9_CALIBRATION_SEEDS),
        "candidate_slots": 3,
        "collect_interval_diagnostics": False,
    }
    for field, expected in expected_calibrate.items():
        if calibrate.get(field) != expected:
            raise CampaignContractError(f"calibrate phase {field} mismatch")
    _require_gate_values(
        calibrate.get("gate"),
        {
            "severe_max_per_seed": 3,
            "repeated_severe_same_id_max": 1,
            "fid_native_margin_max": 3.0,
            "kid_native_margin_max": 0.005,
            "niqe_native_margin_max": 0.10,
            "sharpness_absolute_min": 300.0,
            "sharpness_native_ratio_min": 0.95,
            "e0_min": 0.75,
            "e0_delta_min": 0.30,
            "edev_delta_min": 0.05,
            "face_count_exact": 1,
            "identity_delta_upper_95_max": 0.02,
            "max_advancing_candidates": 2,
            "rank_order": [
                "severe",
                "kid",
                "fid",
                "edev_desc",
                "e0_desc",
                "arm_id",
            ],
        },
        "calibrate gate",
    )
    _validate_confirm_phase(phases["confirm512"], seeds["confirm512"])
    _validate_full_phase(phases["full"], seeds["full"])
    return phases


def _validate_confirm_phase(value: Any, seeds: Any) -> None:
    expected = {
        "manifest": "validate_512",
        "sample_count": 512,
        "shards_per_logical_run": 8,
        "seed": _seed_tuple(seeds, "confirm512")[0],
        "candidate_slots": 2,
        "visual_severe_max": 25,
        "gate_ref": "calibrate.gate",
        "rank_order": ["severe", "kid", "fid", "edev_desc", "e0_desc", "arm_id"],
        "winner_count": 1,
    }
    _require_exact_mapping(value, expected, "confirm512 phase")


def _validate_full_phase(value: Any, seeds: Any) -> None:
    expected = {
        "manifest": "full_2048",
        "visual_manifest": "full_visual_64",
        "sample_count": 2048,
        "shards_per_logical_run": 16,
        "seed": _seed_tuple(seeds, "full")[0],
        "visual_severe_max": 3,
        "gate_ref": "calibrate.gate",
        "representation": {
            "e1_mean_delta_lower_95_min_exclusive": 0.0,
            "e2_mean_delta_lower_95_min_exclusive": 0.0,
        },
        "privacy": {
            "recognizers": ["arcface", "facenet", "adaface"],
            "identity_delta_upper_95_max": 0.02,
            "require_complete_pairs": 2048,
        },
        "sealed_until_selection_lock": [
            "heldout_e1",
            "heldout_e2",
            "facenet",
            "adaface",
        ],
        "report_only_metrics": ["tar_at_far", "eer", "auc"],
        "reselect_after_failure": False,
    }
    _require_exact_mapping(value, expected, "full phase")


def _require_exact_mapping(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise CampaignContractError(f"{label} exact contract mismatch")


def _require_gate_values(value: Any, expected: Mapping[str, Any], label: str) -> None:
    _require_exact_mapping(value, expected, label)


def _validate_diagnose_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "path",
        "sha256",
        "sample_count",
        "ordered_sample_id_sha256",
        "difficult_count",
        "control_count",
        "matched_pair_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CampaignContractError("diagnose manifest fields are not canonical")
    result = {
        "path": _require_nonempty(value.get("path"), "diagnose manifest path"),
        "sha256": _require_sha256(value.get("sha256"), "diagnose manifest SHA256"),
        "sample_count": _strict_int(value.get("sample_count"), "diagnose sample_count"),
        "ordered_sample_id_sha256": _require_sha256(
            value.get("ordered_sample_id_sha256"), "diagnose ordered ID SHA256"
        ),
        "difficult_count": _strict_int(
            value.get("difficult_count"), "diagnose difficult_count"
        ),
        "control_count": _strict_int(
            value.get("control_count"), "diagnose control_count"
        ),
        "matched_pair_sha256": _require_sha256(
            value.get("matched_pair_sha256"), "diagnose pairing SHA256"
        ),
    }
    if (
        result["sample_count"] != 18
        or result["difficult_count"] != 9
        or result["control_count"] != 9
    ):
        raise CampaignContractError("diagnose manifest must lock 9 difficult/9 control")
    return result


def _evaluate_quality_arm(
    arm: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int],
    severe_limit: int,
    bootstrap_seed: int,
    reject_repeated_severe: bool,
) -> dict[str, Any]:
    _reject_derived_arm_fields(arm)
    arm_id, family, config_sha, output_sha = _arm_identity(arm)
    seed_rows = arm.get("seed_results")
    if not isinstance(seed_rows, list) or len(seed_rows) != len(expected_seeds):
        raise CampaignContractError("quality arm seed_results count mismatch")
    by_seed = {}
    for row in seed_rows:
        normalized = _evaluate_seed_quality(row, severe_limit=severe_limit)
        seed = normalized["seed"]
        if seed in by_seed:
            raise CampaignContractError("quality arm contains duplicate seed result")
        by_seed[seed] = normalized
    seeds = _normalize_expected_seeds(expected_seeds)
    if tuple(sorted(by_seed)) != tuple(sorted(seeds)):
        raise CampaignContractError("quality arm does not cover registered seeds")
    ordered_rows = [by_seed[seed] for seed in seeds]
    failures = [
        f"seed_{row['seed']}:{failure}"
        for row in ordered_rows
        for failure in row["failures"]
    ]
    if reject_repeated_severe:
        severe_frequency: dict[str, int] = {}
        for row in ordered_rows:
            for sample_id in row["severe_sample_ids"]:
                severe_frequency[sample_id] = severe_frequency.get(sample_id, 0) + 1
        repeated = sorted(
            sample_id for sample_id, count in severe_frequency.items() if count >= 2
        )
        if repeated:
            failures.append("same_sample_severe_in_multiple_seeds")
    privacy_rows = arm.get("privacy_rows")
    if not isinstance(privacy_rows, list):
        raise CampaignContractError("quality arm requires privacy_rows")
    if all(row["arcface_exact_one"] for row in ordered_rows):
        bootstrap = privacy_delta_cluster_bootstrap(
            privacy_rows,
            expected_seeds=seeds,
            bootstrap_seed=bootstrap_seed,
        )
        expected_samples = 64 if len(seeds) == 3 else 512
        if bootstrap["sample_count"] != expected_samples:
            raise CampaignContractError(
                f"privacy bootstrap must cover exactly {expected_samples} sample IDs"
            )
        if bootstrap["upper_95_one_sided"] > 0.02:
            failures.append("privacy_delta_upper_gt_0.02")
    else:
        if privacy_rows:
            raise CampaignContractError(
                "privacy_rows must be empty when ArcFace exact-one coverage fails"
            )
        bootstrap = None
    return {
        "arm_id": arm_id,
        "family": family,
        "config_sha256": config_sha,
        "output_sha256": output_sha,
        "seed_results": ordered_rows,
        "privacy_bootstrap": bootstrap,
        "severe_sort_key": sum(row["severe_count"] for row in ordered_rows),
        "kid_sort_value": statistics.fmean(row["kid"] for row in ordered_rows),
        "fid_sort_value": statistics.fmean(row["fid"] for row in ordered_rows),
        "edev_sort_value": statistics.fmean(row["delta_edev"] for row in ordered_rows),
        "e0_sort_value": statistics.fmean(row["e0"] for row in ordered_rows),
        "failures": failures,
        "passed": not failures,
    }


def _evaluate_seed_quality(value: Any, *, severe_limit: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignContractError("seed quality result must be a mapping")
    seed = _require_nonnegative_int(value.get("seed"), "quality seed")
    severe_ids = value.get("severe_sample_ids")
    if not isinstance(severe_ids, list) or any(
        not isinstance(sample_id, str) or not sample_id for sample_id in severe_ids
    ):
        raise CampaignContractError("severe_sample_ids must be a string list")
    if len(set(severe_ids)) != len(severe_ids):
        raise CampaignContractError("severe_sample_ids contain duplicates")
    severe_count = _require_nonnegative_int(value.get("severe_count"), "severe_count")
    if severe_count != len(severe_ids):
        raise CampaignContractError("severe_count disagrees with severe_sample_ids")
    metrics = {
        field: _finite_float(value.get(field), field)
        for field in (
            "fid",
            "native_fid",
            "kid",
            "native_kid",
            "niqe",
            "native_niqe",
            "sharpness",
            "native_sharpness",
            "e0",
            "delta_e0",
            "delta_edev",
        )
    }
    arcface_exact_one = value.get("arcface_exact_one")
    if not isinstance(arcface_exact_one, bool):
        raise CampaignContractError("arcface_exact_one must be boolean")
    failures = []
    if severe_count > severe_limit:
        failures.append(f"severe_count_gt_{severe_limit}")
    if metrics["fid"] > metrics["native_fid"] + 3:
        failures.append("fid_above_native_plus_3")
    if metrics["kid"] > metrics["native_kid"] + 0.005:
        failures.append("kid_above_native_plus_0.005")
    if metrics["niqe"] > metrics["native_niqe"] + 0.10:
        failures.append("niqe_above_native_plus_0.10")
    if metrics["sharpness"] < max(300.0, 0.95 * metrics["native_sharpness"]):
        failures.append("sharpness_below_gate")
    if metrics["e0"] < 0.75:
        failures.append("e0_below_0.75")
    if metrics["delta_e0"] < 0.30:
        failures.append("delta_e0_below_0.30")
    if metrics["delta_edev"] < 0.05:
        failures.append("delta_edev_below_0.05")
    if not arcface_exact_one:
        failures.append("arcface_not_exactly_one_face_per_image")
    return {
        "seed": seed,
        "severe_count": severe_count,
        "severe_sample_ids": list(severe_ids),
        **metrics,
        "arcface_exact_one": arcface_exact_one,
        "failures": failures,
        "passed": not failures,
    }


def _quality_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["severe_sort_key"],
        row["kid_sort_value"],
        row["fid_sort_value"],
        -row["edev_sort_value"],
        -row["e0_sort_value"],
        row["arm_id"],
    )


def _quality_thresholds(*, severe_limit: int) -> dict[str, Any]:
    return {
        "severe_max": severe_limit,
        "fid_native_delta_max": 3.0,
        "kid_native_delta_max": 0.005,
        "niqe_native_delta_max": 0.10,
        "sharpness_min_absolute": 300.0,
        "sharpness_native_ratio_min": 0.95,
        "e0_min": 0.75,
        "delta_e0_min": 0.30,
        "delta_edev_min": 0.05,
        "arcface_exact_one_required": True,
        "privacy_delta_direction": "source_candidate_minus_source_native",
        "privacy_delta_upper_95_max": 0.02,
        "bootstrap_iterations": R9_BOOTSTRAP_ITERATIONS,
    }


def _validate_gate_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping) or set(context) != R9_GATE_CONTEXT_FIELDS:
        raise CampaignContractError("gate context fields are not canonical")
    return {
        "campaign_id": _require_nonempty(context.get("campaign_id"), "campaign ID"),
        **{
            field: _require_sha256(context.get(field), field)
            for field in sorted(R9_GATE_CONTEXT_FIELDS - {"campaign_id"})
        },
    }


def _arm_identity(arm: Mapping[str, Any]) -> tuple[str, str, str, str]:
    if not isinstance(arm, Mapping):
        raise CampaignContractError("arm result must be a mapping")
    return (
        _require_nonempty(arm.get("arm_id"), "arm_id"),
        _require_nonempty(arm.get("family"), "arm family"),
        _require_sha256(arm.get("config_sha256"), "arm config SHA256"),
        _require_sha256(arm.get("output_sha256"), "arm output SHA256"),
    )


def _reject_derived_arm_fields(arm: Mapping[str, Any]) -> None:
    if any(field in arm for field in ("passed", "failures", "verdict")):
        raise CampaignContractError("arm verdict fields are derived, not hand-filled")


def _build_gate_payload(
    *,
    phase: str,
    context: Mapping[str, Any],
    seeds: Sequence[int],
    thresholds: Mapping[str, Any],
    evaluated: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
    verdict: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    arm_ids = [row.get("arm_id") for row in evaluated]
    if len(set(arm_ids)) != len(arm_ids):
        raise CampaignContractError("gate contains duplicate arm IDs")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_gate_contract_v1",
        "phase": phase,
        "context": dict(context),
        "thresholds": _json_mapping(thresholds, "gate thresholds"),
        "seeds": list(_normalize_expected_seeds(seeds)),
        "arms": [dict(row) for row in evaluated],
        "failures": [
            {"arm_id": row["arm_id"], "reasons": list(row["failures"])}
            for row in evaluated
            if row["failures"]
        ],
        "selected_arm_ids": list(selected),
        "verdict": verdict,
    }
    if extra:
        for key, value in extra.items():
            if key in payload or key == "gate_contract_sha256":
                raise CampaignContractError("gate extra field collides with schema")
            payload[key] = value
    payload["gate_contract_sha256"] = _canonical_contract_sha256(
        payload, "gate_contract_sha256"
    )
    return validate_gate_contract(payload)


def _verify_contract_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    declared = _require_sha256(payload.get(digest_field), digest_field)
    actual = _canonical_contract_sha256(payload, digest_field)
    if declared != actual:
        raise CampaignContractError(f"{digest_field} canonical digest mismatch")
    return declared


def _normalize_expected_seeds(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not value:
        raise CampaignContractError("expected seeds must be a non-empty sequence")
    result = tuple(_require_nonnegative_int(seed, "expected seed") for seed in value)
    if len(set(result)) != len(result):
        raise CampaignContractError("expected seeds must be unique")
    return result


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CampaignContractError(f"{label} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CampaignContractError(f"{label} must be finite numeric data") from error
    if not math.isfinite(result):
        raise CampaignContractError(f"{label} must be finite numeric data")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CampaignContractError("cannot calculate percentile from no values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _validate_seed_contract(seeds: Mapping[str, Any]) -> None:
    required = {"diagnose", "calibrate", "confirm512", "full"}
    allowed = required | {"preflight"}
    if set(seeds) not in (required, allowed):
        raise CampaignContractError("campaign seeds have unknown or missing phases")
    if "preflight" in seeds and _seed_tuple(seeds["preflight"], "preflight") != (1337,):
        raise CampaignContractError("preflight seed must be exactly 1337")
    if _seed_tuple(seeds["diagnose"], "diagnose") != (1337,):
        raise CampaignContractError("diagnose seed must be exactly 1337")
    if _seed_tuple(seeds["calibrate"], "calibrate") != R9_CALIBRATION_SEEDS:
        raise CampaignContractError(
            "calibrate seeds must be exactly 1337, 2027 and 3407"
        )
    confirm = _seed_tuple(seeds["confirm512"], "confirm512")
    full = _seed_tuple(seeds["full"], "full")
    if len(confirm) != 1 or confirm[0] in R9_CALIBRATION_SEEDS:
        raise CampaignContractError("confirm512 requires one new seed")
    if len(full) != 1 or full[0] in {*R9_CALIBRATION_SEEDS, confirm[0]}:
        raise CampaignContractError("full requires one new held-out seed")


def _validate_manifest_entry(
    *,
    name: str,
    declared: Mapping[str, Any],
    repo_root: Path,
    expected_count: int | None,
) -> tuple[dict[str, Any], list[str]]:
    if set(declared) != {
        "path",
        "sha256",
        "sample_count",
        "ordered_sample_id_sha256",
    }:
        raise CampaignContractError(f"manifest {name!r} fields are not canonical")
    path = _require_repo_path(declared.get("path"), repo_root, f"{name} path")
    if path.is_symlink() or not path.is_file():
        raise CampaignContractError(f"manifest {name!r} must be a regular file")
    sample_ids = _read_manifest_ids(path)
    actual_sha = _sha256_file(path)
    actual_ordered = _sample_id_digest(sample_ids)
    declared_sha = _require_sha256(declared.get("sha256"), f"{name} SHA256")
    declared_ordered = _require_sha256(
        declared.get("ordered_sample_id_sha256"), f"{name} ordered ID SHA256"
    )
    declared_count = _strict_int(declared.get("sample_count"), f"{name} sample_count")
    if expected_count is not None and len(sample_ids) != expected_count:
        raise CampaignContractError(
            f"manifest {name!r} must contain exactly {expected_count} IDs"
        )
    if declared_count != len(sample_ids):
        raise CampaignContractError(f"manifest {name!r} sample_count mismatch")
    if declared_sha != actual_sha or declared_ordered != actual_ordered:
        raise CampaignContractError(f"manifest {name!r} digest mismatch")
    return (
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": actual_sha,
            "sample_count": len(sample_ids),
            "ordered_sample_id_sha256": actual_ordered,
        },
        sample_ids,
    )


def _visual_evidence_ids(evidence: Mapping[str, Any]) -> list[str]:
    samples = evidence.get("samples")
    if not isinstance(samples, list) or not samples:
        raise CampaignContractError("visual evidence requires a non-empty sample list")
    ids: list[str] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Mapping):
            raise CampaignContractError(f"visual evidence row {index} is invalid")
        sample_id = row.get("sample_id")
        _require_nonempty(sample_id, f"visual evidence row {index} sample_id")
        ids.append(str(sample_id))
    if len(set(ids)) != len(ids):
        raise CampaignContractError("visual evidence contains duplicate sample IDs")
    declared_count = evidence.get("sample_count")
    if declared_count is not None and declared_count != len(ids):
        raise CampaignContractError("visual evidence sample_count mismatch")
    return ids


def _evidence_digest(evidence: Mapping[str, Any]) -> str:
    declared = evidence.get("evidence_contract_sha256")
    actual = _canonical_contract_sha256(evidence, "evidence_contract_sha256")
    if declared is not None and declared != actual:
        raise CampaignContractError("visual evidence contract digest mismatch")
    return actual


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CampaignContractError(
            f"{label} must contain finite JSON values"
        ) from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise CampaignContractError(f"{label} must be a mapping")
    return decoded


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CampaignContractError(
            "contract contains non-finite JSON values"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _canonical_contract_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return _canonical_json_sha256(canonical)


def _require_repo_path(value: Any, repo_root: Path, label: str) -> Path:
    _require_nonempty(value, label)
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise CampaignContractError(f"{label} escapes repo_root") from error
    return resolved


def _read_manifest_ids(path: Path) -> list[str]:
    return [str(row["sample_id"]) for row in _read_manifest_rows(path)]


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CampaignContractError(
                f"{path}:{line_number}: invalid JSON"
            ) from error
        sample_id = row.get("sample_id") if isinstance(row, Mapping) else None
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise CampaignContractError(
                f"{path}:{line_number}: invalid or duplicate sample_id"
            )
        ids.append(sample_id)
        seen.add(sample_id)
        rows.append(dict(row))
    if not rows:
        raise CampaignContractError(f"manifest {path} must not be empty")
    return rows


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


def _seed_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise CampaignContractError(f"{label} seeds must be a non-empty list")
    result = tuple(_strict_int(seed, f"{label} seed") for seed in value)
    if len(set(result)) != len(result) or any(seed < 0 for seed in result):
        raise CampaignContractError(f"{label} seeds must be unique non-negative ints")
    return result


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignContractError(f"{label} must be an integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    result = _strict_int(value, label)
    if result < 0:
        raise CampaignContractError(f"{label} must be non-negative")
    return result


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignContractError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignContractError(f"{label} must be a lowercase SHA256 digest")
    return value
