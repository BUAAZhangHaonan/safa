from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence

from safa.evaluation.r9_determinism import (
    R9_ATTENTION_BACKEND,
    R9_DETERMINISM_POLICY,
    R9_EXPERIMENT_CONTRACT,
    canonical_json_sha256,
    canonical_r9_arm_config_digest,
    validate_r9_execution_config,
)


R9_SEMIGROUP_PREFLIGHT_CONTRACT = "safa_r9_semigroup_preflight_v1"
R9_SEMIGROUP_GATE_CONTRACT = "safa_r9_semigroup_gate_v1"
R9_PREFLIGHT_SAMPLE_COUNT = 64
R9_LOCKED_SCHEDULE_SCHEMA_VERSION = 3
R9_SELECTION_RULE = "smallest_numeric_t_cut_passing_all_registered_thresholds"
R9_SEMIGROUP_RECOVERY_POLICY_VERSION = "safa_r9_semigroup_recovery_policy_v1"
R9_SEMIGROUP_RECOVERY_SELECTION_RULE = "policy_locked_t_cut_0p25_report_only"
R9_SEMIGROUP_RECOVERY_AUTHORIZATION_ID = (
    "user-authorized-r9-report-only-visual-limit-1-lock-025-2026-07-14"
)
R9_SEMIGROUP_RECOVERY_POLICY = {
    "schema_version": 1,
    "contract_type": R9_SEMIGROUP_RECOVERY_POLICY_VERSION,
    "numerical_metrics_role": "report_only",
    "visual_severe_limit_per_split": 1,
    "selected_t_cut": 0.25,
    "technical_hard_failures": [
        "contract_mismatch",
        "determinism_mismatch",
        "non_finite_metric",
        "out_of_memory",
    ],
    "visual_metrics_role": "reported_with_registered_limit",
}
R9_SEMIGROUP_RECOVERY_POLICY_SHA256 = canonical_json_sha256(
    R9_SEMIGROUP_RECOVERY_POLICY
)


def canonical_r9_schedule_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an R9 schedule without the gate file provenance that it later records.

    The gate binds this canonical schedule hash, while the finalized schedule records
    the gate path and file hash. Excluding only those provenance fields prevents a
    gate/schedule file-hash cycle without weakening either file-level binding.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("R9 locked schedule must be a mapping")
    contract = dict(payload)
    contract.pop("schedule_contract_sha256", None)
    contract.pop("r9_semigroup_gate_contract", None)
    contract.pop("r9_semigroup_gate_contract_sha256", None)
    return canonical_json_sha256(contract)


def validate_r9_locked_schedule_bindings(
    config: Mapping[str, Any], schedule_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every R9 preflight/report/gate file bound by a v3 schedule."""

    if not isinstance(schedule_manifest, Mapping):
        raise ValueError("R9 locked schedule manifest must be a mapping")
    if schedule_manifest.get("schema_version") != R9_LOCKED_SCHEDULE_SCHEMA_VERSION:
        raise ValueError("R9 locked schedule manifest must use schema_version=3")
    if schedule_manifest.get("gate_passed") is not True:
        raise ValueError("R9 locked schedule manifest must record gate_passed=true")
    declared_schedule_sha256 = _require_sha256(
        schedule_manifest.get("schedule_contract_sha256"),
        "schedule_contract_sha256",
    )
    if (
        canonical_r9_schedule_contract_sha256(schedule_manifest)
        != declared_schedule_sha256
    ):
        raise ValueError("R9 locked schedule canonical SHA256 disagrees")

    preflight_path = _bound_contract_path(
        config, schedule_manifest, "semigroup_preflight_contract"
    )
    preflight_file_sha256 = _bound_file_sha256(
        preflight_path,
        schedule_manifest,
        "semigroup_preflight_contract_sha256",
        config=config,
    )
    preflight = _read_json_mapping(preflight_path, "R9 semigroup preflight contract")
    _validate_r9_preflight_payload(preflight)
    if canonical_json_sha256(preflight) != preflight_file_sha256:
        raise ValueError("R9 semigroup preflight canonical SHA256 disagrees")
    execution = validate_r9_execution_config(config)
    if preflight["determinism_policy"] != execution["determinism_policy"]:
        raise ValueError("R9 guided config determinism policy disagrees with preflight")
    if preflight["attention_backend"] != execution["attention_backend"]:
        raise ValueError("R9 guided config attention backend disagrees with preflight")
    if preflight["checkpoint"]["sha256"] != config.get("checkpoint_sha256"):
        raise ValueError("R9 guided config checkpoint disagrees with preflight")
    if preflight["sample_manifest"]["sha256"] != config.get(
        "semigroup_sample_id_manifest_sha256"
    ):
        raise ValueError("R9 guided config semigroup manifest disagrees with preflight")
    if preflight_file_sha256 != _sha256_path(preflight_path):
        raise AssertionError("unreachable preflight SHA256 validation state")

    report_path = _bound_contract_path(config, schedule_manifest, "semigroup_report")
    report_file_sha256 = _bound_file_sha256(
        report_path, schedule_manifest, "semigroup_report_sha256"
    )
    report = _read_json_mapping(report_path, "R9 semigroup report")
    if report.get("gate_passed") is not True:
        raise ValueError("R9 semigroup report must record gate_passed=true")
    recovery_report = (
        report.get("schema_version") == 2
        and report.get("contract_type") == "safa_r9_semigroup_recovery_report_v2"
    )
    if recovery_report:
        candidates = report.get("candidates")
        selected = report.get("selected_t_cut")
        if (
            report.get("policy_sha256") != R9_SEMIGROUP_RECOVERY_POLICY_SHA256
            or report.get("policy_version") != R9_SEMIGROUP_RECOVERY_POLICY_VERSION
            or report.get("numerical_metrics_role") != "report_only"
            or report.get("selection_rule") != R9_SEMIGROUP_RECOVERY_SELECTION_RULE
            or selected != 0.25
            or not isinstance(candidates, list)
            or not any(
                isinstance(candidate, Mapping)
                and candidate.get("t_cut") == 0.25
                and candidate.get("passed") is True
                and isinstance(candidate.get("numeric_threshold_pass"), bool)
                for candidate in candidates
            )
            or schedule_manifest.get("selection_rule")
            != R9_SEMIGROUP_RECOVERY_SELECTION_RULE
            or schedule_manifest.get("recovery_policy_sha256")
            != R9_SEMIGROUP_RECOVERY_POLICY_SHA256
            or schedule_manifest.get("numerical_metrics_role") != "report_only"
        ):
            raise ValueError("R9 recovery report/schedule policy binding mismatch")

    gate_path = _bound_contract_path(
        config, schedule_manifest, "r9_semigroup_gate_contract"
    )
    _bound_file_sha256(
        gate_path,
        schedule_manifest,
        "r9_semigroup_gate_contract_sha256",
        config=config,
    )
    gate = _read_json_mapping(gate_path, "R9 semigroup gate contract")
    validated_gate = _validate_gate_against_locked_files(
        gate,
        preflight=preflight,
        report=report,
        report_sha256=report_file_sha256,
        schedule_sha256=declared_schedule_sha256,
    )
    if validated_gate.get("semigroup_report_sha256") != report_file_sha256:
        raise ValueError(
            "R9 gate semigroup report SHA256 disagrees with the locked report"
        )
    if validated_gate.get("schedule_contract_sha256") != declared_schedule_sha256:
        raise ValueError("R9 gate schedule SHA256 disagrees with the locked schedule")
    if schedule_manifest.get("checkpoint_sha256") != validated_gate.get(
        "checkpoint_sha256"
    ):
        raise ValueError("R9 gate checkpoint SHA256 disagrees with the locked schedule")
    if schedule_manifest.get(
        "semigroup_sample_id_manifest_sha256"
    ) != validated_gate.get("sample_id_manifest_sha256"):
        raise ValueError(
            "R9 gate sample manifest SHA256 disagrees with the locked schedule"
        )
    return validated_gate


def canonical_r9_semigroup_preflight_payload(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    execution = validate_r9_execution_config(config)
    if config.get("mode") != "semigroup" or config.get("phase") != "semigroup":
        raise ValueError(
            "R9 semigroup preflight config requires mode=phase='semigroup'"
        )
    if (
        _positive_int(config.get("max_samples"), "max_samples")
        != R9_PREFLIGHT_SAMPLE_COUNT
    ):
        raise ValueError("R9 semigroup preflight must contain exactly 64 samples")
    split_times = _finite_sequence(config.get("split_times"), "split_times")
    if split_times != [0.25, 0.5, 0.75]:
        raise ValueError("R9 semigroup split_times must be [0.25, 0.5, 0.75]")
    candidates = _finite_sequence(
        config.get("registered_t_cut_candidates"), "registered_t_cut_candidates"
    )
    if candidates != [0.75, 0.5, 0.25]:
        raise ValueError("R9 registered_t_cut_candidates must be [0.75, 0.5, 0.25]")
    thresholds = config.get("semigroup_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("R9 semigroup preflight requires semigroup_thresholds")
    checkpoint_sha256 = _require_sha256(
        config.get("checkpoint_sha256"), "checkpoint_sha256"
    )
    sample_manifest_sha256 = _require_sha256(
        config.get("sample_id_manifest_sha256"), "sample_id_manifest_sha256"
    )
    sample_manifest = str(config.get("sample_id_manifest", ""))
    if not sample_manifest:
        raise ValueError("R9 semigroup preflight requires sample_id_manifest")
    checkpoint = str(config.get("checkpoint", ""))
    if not checkpoint:
        raise ValueError("R9 semigroup preflight requires checkpoint")
    return {
        "schema_version": 1,
        "contract_type": R9_SEMIGROUP_PREFLIGHT_CONTRACT,
        "experiment_contract": R9_EXPERIMENT_CONTRACT,
        "determinism_policy": execution["determinism_policy"],
        "determinism_policy_sha256": execution["determinism_policy_sha256"],
        "attention_backend": execution["attention_backend"],
        "checkpoint": {"path": checkpoint, "sha256": checkpoint_sha256},
        "sample_manifest": {
            "path": sample_manifest,
            "sha256": sample_manifest_sha256,
            "sample_count": R9_PREFLIGHT_SAMPLE_COUNT,
        },
        "sampling_seed": int(config.get("sampling_seed", config.get("seed"))),
        "schedule": {
            "registered_t_cut_candidates": candidates,
            "split_times": split_times,
        },
        "thresholds": dict(thresholds),
    }


def canonical_r9_semigroup_preflight_digest(config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(canonical_r9_semigroup_preflight_payload(config))


def validate_r9_semigroup_preflight_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonical_r9_semigroup_preflight_payload(config)
    computed = canonical_json_sha256(payload)
    declared = _require_sha256(
        config.get("semigroup_preflight_contract_sha256"),
        "semigroup_preflight_contract_sha256",
    )
    if declared != computed:
        raise ValueError(
            "R9 semigroup_preflight_contract_sha256 disagrees with its canonical payload"
        )
    return payload


def build_r9_semigroup_gate_contract(
    config: Mapping[str, Any],
    *,
    effective_config_sha256: str,
    semigroup_report_sha256: str,
    gate_passed: bool,
    selected_t_cut: float | None,
    schedule_contract_sha256: str | None,
) -> dict[str, Any]:
    preflight = validate_r9_semigroup_preflight_config(config)
    report_sha256 = _require_sha256(semigroup_report_sha256, "semigroup_report_sha256")
    effective_sha256 = _require_sha256(
        effective_config_sha256, "effective_config_sha256"
    )
    if not isinstance(gate_passed, bool):
        raise ValueError("gate_passed must be a boolean")
    if gate_passed:
        selected = _finite_open_unit(selected_t_cut, "selected_t_cut")
        registered = preflight["schedule"]["registered_t_cut_candidates"]
        if selected not in registered:
            raise ValueError("selected_t_cut is not registered by the R9 preflight")
        schedule_sha256 = _require_sha256(
            schedule_contract_sha256, "schedule_contract_sha256"
        )
    else:
        if selected_t_cut is not None or schedule_contract_sha256 is not None:
            raise ValueError("failed R9 gate must not bind a selected schedule")
        selected = None
        schedule_sha256 = None
    payload = {
        "schema_version": 1,
        "contract_type": R9_SEMIGROUP_GATE_CONTRACT,
        "experiment_contract": R9_EXPERIMENT_CONTRACT,
        "preflight_contract_sha256": canonical_json_sha256(preflight),
        "determinism_policy_sha256": preflight["determinism_policy_sha256"],
        "attention_backend_requested": preflight["attention_backend"],
        "attention_backend_resolved": R9_ATTENTION_BACKEND,
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "sample_id_manifest_sha256": preflight["sample_manifest"]["sha256"],
        "effective_config_sha256": effective_sha256,
        "arm_config_sha256": canonical_r9_arm_config_digest(config),
        "registered_t_cut_candidates": preflight["schedule"][
            "registered_t_cut_candidates"
        ],
        "split_times": preflight["schedule"]["split_times"],
        "semigroup_report_sha256": report_sha256,
        "gate_passed": gate_passed,
        "selected_t_cut": selected,
        "schedule_contract_sha256": schedule_sha256,
    }
    return {**payload, "gate_contract_sha256": canonical_json_sha256(payload)}


def validate_r9_semigroup_gate_contract(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("R9 semigroup gate contract must be a mapping")
    expected = build_r9_semigroup_gate_contract(
        config,
        effective_config_sha256=_require_sha256(
            payload.get("effective_config_sha256"), "effective_config_sha256"
        ),
        semigroup_report_sha256=_require_sha256(
            payload.get("semigroup_report_sha256"), "semigroup_report_sha256"
        ),
        gate_passed=payload.get("gate_passed"),
        selected_t_cut=payload.get("selected_t_cut"),
        schedule_contract_sha256=payload.get("schedule_contract_sha256"),
    )
    if dict(payload) != expected:
        raise ValueError(
            "R9 semigroup gate contract disagrees with its bound preflight"
        )
    return expected


def _validate_r9_preflight_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("R9 preflight contract must use schema_version=1")
    if payload.get("contract_type") != R9_SEMIGROUP_PREFLIGHT_CONTRACT:
        raise ValueError("R9 preflight contract_type mismatch")
    if payload.get("experiment_contract") != R9_EXPERIMENT_CONTRACT:
        raise ValueError("R9 preflight experiment_contract mismatch")
    if payload.get("determinism_policy") != R9_DETERMINISM_POLICY:
        raise ValueError("R9 preflight determinism policy mismatch")
    if payload.get("determinism_policy_sha256") != canonical_json_sha256(
        R9_DETERMINISM_POLICY
    ):
        raise ValueError("R9 preflight determinism policy SHA256 mismatch")
    if payload.get("attention_backend") != R9_ATTENTION_BACKEND:
        raise ValueError("R9 preflight attention backend mismatch")
    checkpoint = payload.get("checkpoint")
    sample_manifest = payload.get("sample_manifest")
    schedule = payload.get("schedule")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("path"):
        raise ValueError("R9 preflight checkpoint binding is invalid")
    _require_sha256(checkpoint.get("sha256"), "preflight checkpoint_sha256")
    if (
        not isinstance(sample_manifest, Mapping)
        or sample_manifest.get("sample_count") != R9_PREFLIGHT_SAMPLE_COUNT
    ):
        raise ValueError("R9 preflight sample manifest binding is invalid")
    _require_sha256(sample_manifest.get("sha256"), "preflight sample manifest SHA256")
    if not isinstance(schedule, Mapping) or schedule.get("split_times") != [
        0.25,
        0.5,
        0.75,
    ]:
        raise ValueError("R9 preflight split schedule is invalid")
    if schedule.get("registered_t_cut_candidates") != [0.75, 0.5, 0.25]:
        raise ValueError("R9 preflight candidate schedule is invalid")


def _validate_gate_against_locked_files(
    payload: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    report: Mapping[str, Any],
    report_sha256: str,
    schedule_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("R9 semigroup gate contract must be a mapping")
    declared = _require_sha256(
        payload.get("gate_contract_sha256"), "gate_contract_sha256"
    )
    canonical = dict(payload)
    canonical.pop("gate_contract_sha256", None)
    if canonical_json_sha256(canonical) != declared:
        raise ValueError("R9 semigroup gate canonical SHA256 disagrees")
    recovery_report = (
        report.get("schema_version") == 2
        and report.get("contract_type") == "safa_r9_semigroup_recovery_report_v2"
    )
    report_expected = {
        "gate_passed": True,
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "determinism_policy_sha256": preflight["determinism_policy_sha256"],
        "attention_backend_requested": R9_ATTENTION_BACKEND,
        "attention_backend_resolved": R9_ATTENTION_BACKEND,
        "sample_id_manifest_sha256": preflight["sample_manifest"]["sha256"],
        "selection_rule": (
            R9_SEMIGROUP_RECOVERY_SELECTION_RULE
            if recovery_report
            else R9_SELECTION_RULE
        ),
    }
    if recovery_report:
        report_expected.update(
            {
                "policy_version": R9_SEMIGROUP_RECOVERY_POLICY_VERSION,
                "policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                "numerical_metrics_role": "report_only",
            }
        )
    for field, expected_value in report_expected.items():
        if report.get(field) != expected_value:
            raise ValueError(f"R9 semigroup report {field} binding mismatch")
    selected = _finite_open_unit(report.get("selected_t_cut"), "report selected_t_cut")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not any(
        isinstance(candidate, Mapping)
        and candidate.get("passed") is True
        and float(candidate.get("t_cut")) == selected
        for candidate in candidates
    ):
        raise ValueError("R9 semigroup report selected_t_cut has no passing candidate")
    expected = {
        "schema_version": 2 if recovery_report else 1,
        "contract_type": (
            "safa_r9_semigroup_recovery_gate_v2"
            if recovery_report
            else R9_SEMIGROUP_GATE_CONTRACT
        ),
        "experiment_contract": R9_EXPERIMENT_CONTRACT,
        "preflight_contract_sha256": canonical_json_sha256(preflight),
        "determinism_policy_sha256": preflight["determinism_policy_sha256"],
        "attention_backend_requested": R9_ATTENTION_BACKEND,
        "attention_backend_resolved": R9_ATTENTION_BACKEND,
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "sample_id_manifest_sha256": preflight["sample_manifest"]["sha256"],
        "effective_config_sha256": report.get("effective_config_sha256"),
        "arm_config_sha256": report.get("arm_config_sha256"),
        "registered_t_cut_candidates": preflight["schedule"][
            "registered_t_cut_candidates"
        ],
        "split_times": preflight["schedule"]["split_times"],
        "semigroup_report_sha256": report_sha256,
        "gate_passed": True,
        "selected_t_cut": report.get("selected_t_cut"),
        "schedule_contract_sha256": schedule_sha256,
    }
    if recovery_report:
        expected.update(
            {
                "recovery_policy_sha256": R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
                "numerical_metrics_role": "report_only",
                "selection_rule": R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
            }
        )
    for digest_field in ("effective_config_sha256", "arm_config_sha256"):
        _require_sha256(expected[digest_field], digest_field)
    if canonical != expected:
        raise ValueError(
            "R9 semigroup gate disagrees with preflight/report/schedule bindings"
        )
    return dict(payload)


def merge_r9_semigroup_shards(
    config: Mapping[str, Any],
    shard_dirs: Sequence[Path],
    *,
    visual_pass_by_split: Mapping[str, bool],
) -> dict[str, Any]:
    preflight = validate_r9_semigroup_preflight_config(config)
    if len(shard_dirs) != 4:
        raise ValueError("R9 semigroup merge requires exactly four shard directories")
    manifest_path = Path(str(config["sample_id_manifest"]))
    if _sha256_path(manifest_path) != preflight["sample_manifest"]["sha256"]:
        raise ValueError("R9 semigroup sample manifest SHA256 mismatch")
    manifest_ids = _read_manifest_ids(manifest_path)
    if len(manifest_ids) != R9_PREFLIGHT_SAMPLE_COUNT:
        raise ValueError("R9 semigroup manifest must contain exactly 64 IDs")
    arm_sha256 = canonical_r9_arm_config_digest(config)
    effective_config = dict(config)
    effective_config["arm_config_sha256"] = arm_sha256
    effective_config_sha256 = canonical_json_sha256(effective_config)
    execution = validate_r9_execution_config(config)
    split_keys = [str(value) for value in preflight["schedule"]["split_times"]]
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    shard_contracts = []
    for shard_index, shard_dir in enumerate(shard_dirs):
        semigroup_path = Path(shard_dir) / "semigroup.json"
        generation_path = Path(shard_dir) / "generation_result.json"
        semigroup = _read_json_mapping(semigroup_path, "R9 semigroup shard")
        generation = _read_json_mapping(generation_path, "R9 generation result")
        if semigroup.get("mode") != "semigroup" or semigroup.get("split_times") != [
            0.25,
            0.5,
            0.75,
        ]:
            raise ValueError(f"R9 shard {shard_index} semigroup schedule mismatch")
        rows = semigroup.get("rows")
        if not isinstance(rows, list) or len(rows) != 16:
            raise ValueError(f"R9 shard {shard_index} must contain exactly 16 rows")
        expected_ids = manifest_ids[shard_index::4]
        actual_ids = [
            str(row.get("sample_id")) for row in rows if isinstance(row, Mapping)
        ]
        if len(actual_ids) != 16 or actual_ids != expected_ids:
            raise ValueError(f"R9 shard {shard_index} IDs violate modulo-4 order")
        _validate_r9_generation_result(
            generation,
            shard_index=shard_index,
            expected_ids=expected_ids,
            arm_sha256=arm_sha256,
            preflight=preflight,
            execution=execution,
        )
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"R9 shard {shard_index} contains a non-mapping row")
            sample_id = str(row["sample_id"])
            if sample_id in rows_by_id:
                raise ValueError(f"duplicate R9 semigroup sample ID: {sample_id}")
            splits = row.get("splits")
            if not isinstance(splits, Mapping) or set(map(str, splits)) != set(
                split_keys
            ):
                raise ValueError("R9 semigroup row has an incomplete split set")
            rows_by_id[sample_id] = row
        shard_contracts.append(
            {
                "shard_index": shard_index,
                "sample_count": 16,
                "ordered_sample_id_sha256": _sample_id_digest(expected_ids),
                "semigroup_sha256": _sha256_path(semigroup_path),
                "generation_result_sha256": _sha256_path(generation_path),
            }
        )
    if set(rows_by_id) != set(manifest_ids):
        raise ValueError("R9 semigroup shards do not exactly cover the 64-ID manifest")
    if set(visual_pass_by_split) != set(split_keys) or any(
        not isinstance(value, bool) for value in visual_pass_by_split.values()
    ):
        raise ValueError("R9 visual review must cover every registered split")
    thresholds = preflight["thresholds"]
    candidates = []
    for split_key in split_keys:
        split_rows = [
            rows_by_id[sample_id]["splits"][split_key] for sample_id in manifest_ids
        ]
        residuals = [
            _finite_metric(row.get("latent_residual"), "latent_residual")
            for row in split_rows
        ]
        cosines = [
            _finite_metric(row.get("endpoint_e0_cosine"), "endpoint_e0_cosine")
            for row in split_rows
        ]
        pixel_l1 = [
            _finite_metric(row.get("decoded_pixel_l1"), "decoded_pixel_l1")
            for row in split_rows
        ]
        psnr = [
            _finite_metric(row.get("decoded_psnr"), "decoded_psnr")
            for row in split_rows
        ]
        median = float(statistics.median(residuals))
        p90 = _percentile(residuals, 0.90)
        cosine = float(statistics.median(cosines))
        visual_pass = visual_pass_by_split[split_key]
        passed = (
            median <= float(thresholds["median"])
            and p90 <= float(thresholds["p90"])
            and cosine >= float(thresholds["endpoint_e0_cosine"])
            and visual_pass
        )
        candidates.append(
            {
                "t_cut": float(split_key),
                "median": median,
                "p90": p90,
                "endpoint_e0_cosine_median": cosine,
                "decoded_pixel_l1_median": float(statistics.median(pixel_l1)),
                "decoded_psnr_median": float(statistics.median(psnr)),
                "visual_pass": visual_pass,
                "passed": passed,
            }
        )
    candidates.sort(key=lambda row: row["t_cut"])
    passed_candidates = [row for row in candidates if row["passed"]]
    selected = passed_candidates[0]["t_cut"] if passed_candidates else None
    return {
        "schema_version": 1,
        "contract_type": "safa_r9_semigroup_report_v1",
        "gate_passed": selected is not None,
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "determinism_policy_sha256": preflight["determinism_policy_sha256"],
        "attention_backend_requested": R9_ATTENTION_BACKEND,
        "attention_backend_resolved": R9_ATTENTION_BACKEND,
        "effective_config_sha256": effective_config_sha256,
        "arm_config_sha256": arm_sha256,
        "selected_t_cut": selected,
        "t_cut": selected,
        "sample_count": len(manifest_ids),
        "sample_id_manifest": str(manifest_path),
        "sample_id_manifest_sha256": preflight["sample_manifest"]["sha256"],
        "ordered_sample_id_sha256": _sample_id_digest(manifest_ids),
        "shards": shard_contracts,
        "selection_rule": R9_SELECTION_RULE,
        "thresholds": dict(thresholds),
        "candidates": candidates,
    }


def finalize_r9_semigroup_preflight(
    config: Mapping[str, Any],
    shard_dirs: Sequence[Path],
    *,
    output_dir: Path,
    visual_pass_by_split: Mapping[str, bool],
) -> dict[str, Any]:
    preflight = validate_r9_semigroup_preflight_config(config)
    arm_sha256 = canonical_r9_arm_config_digest(config)
    effective = dict(config)
    effective["arm_config_sha256"] = arm_sha256
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight_path = output / "preflight_contract.json"
    effective_path = output / "effective_config.json"
    report_path = output / "semigroup_report.json"
    gate_path = output / "gate_contract.json"
    schedule_path = output / "locked_schedule_manifest.json"
    existing = [
        path for path in (report_path, gate_path, schedule_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"refusing to replace finalized R9 contracts: {existing!r}"
        )
    report = merge_r9_semigroup_shards(
        config, shard_dirs, visual_pass_by_split=visual_pass_by_split
    )
    _atomic_write_canonical_json(preflight_path, preflight)
    _atomic_write_canonical_json(effective_path, effective)
    _atomic_write_json(report_path, report)
    report_sha256 = _sha256_path(report_path)
    if report["gate_passed"] is not True:
        gate = build_r9_semigroup_gate_contract(
            config,
            effective_config_sha256=report["effective_config_sha256"],
            semigroup_report_sha256=report_sha256,
            gate_passed=False,
            selected_t_cut=None,
            schedule_contract_sha256=None,
        )
        _atomic_write_canonical_json(gate_path, gate)
        return {"gate_passed": False, "report": report, "gate_contract": gate}
    t_cut = _finite_open_unit(report["selected_t_cut"], "selected_t_cut")
    guided = [1.0 - index * (1.0 - t_cut) / 3.0 for index in range(4)]
    guided[-1] = t_cut
    schedule = {
        "schema_version": R9_LOCKED_SCHEDULE_SCHEMA_VERSION,
        "gate_passed": True,
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "semigroup_report": str(report_path),
        "semigroup_report_sha256": report_sha256,
        "semigroup_sample_id_manifest": preflight["sample_manifest"]["path"],
        "semigroup_sample_id_manifest_sha256": preflight["sample_manifest"]["sha256"],
        "semigroup_preflight_contract": str(preflight_path),
        "semigroup_preflight_contract_sha256": _sha256_path(preflight_path),
        "t_cut": t_cut,
        "guided_steps": 3,
        "guided_times": guided,
        "unguided_tail_intervals": 2,
        "unguided_times": [t_cut, t_cut / 2.0, 0.0],
        "selection_rule": R9_SELECTION_RULE,
    }
    schedule["schedule_contract_sha256"] = canonical_r9_schedule_contract_sha256(
        schedule
    )
    gate = build_r9_semigroup_gate_contract(
        config,
        effective_config_sha256=report["effective_config_sha256"],
        semigroup_report_sha256=report_sha256,
        gate_passed=True,
        selected_t_cut=t_cut,
        schedule_contract_sha256=schedule["schedule_contract_sha256"],
    )
    validate_r9_semigroup_gate_contract(gate, config)
    _atomic_write_canonical_json(gate_path, gate)
    schedule["r9_semigroup_gate_contract"] = str(gate_path)
    schedule["r9_semigroup_gate_contract_sha256"] = _sha256_path(gate_path)
    _atomic_write_json(schedule_path, schedule)
    validation_config = {
        **config,
        "semigroup_report": str(report_path),
        "semigroup_sample_id_manifest": preflight["sample_manifest"]["path"],
        "semigroup_sample_id_manifest_sha256": preflight["sample_manifest"]["sha256"],
        "semigroup_preflight_contract": str(preflight_path),
        "r9_semigroup_gate_contract": str(gate_path),
        "r9_semigroup_gate_contract_sha256": schedule[
            "r9_semigroup_gate_contract_sha256"
        ],
    }
    validate_r9_locked_schedule_bindings(validation_config, schedule)
    return {
        "gate_passed": True,
        "report": report,
        "gate_contract": gate,
        "schedule": schedule,
        "artifacts": {
            "preflight_contract": str(preflight_path),
            "effective_config": str(effective_path),
            "semigroup_report": str(report_path),
            "gate_contract": str(gate_path),
            "locked_schedule_manifest": str(schedule_path),
        },
    }


def _validate_r9_generation_result(
    payload: Mapping[str, Any],
    *,
    shard_index: int,
    expected_ids: Sequence[str],
    arm_sha256: str,
    preflight: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    if payload.get("status") != "complete" or payload.get("mode") != "semigroup":
        raise ValueError(f"R9 shard {shard_index} generation result is incomplete")
    if payload.get("sample_count") != 16 or payload.get(
        "sample_id_sha256"
    ) != _sample_id_digest(expected_ids):
        raise ValueError(f"R9 shard {shard_index} sample contract mismatch")
    if payload.get("shard") != {"index": shard_index, "count": 4}:
        raise ValueError(f"R9 shard {shard_index} coordinates mismatch")
    if payload.get("arm_config_sha256") != arm_sha256:
        raise ValueError(f"R9 shard {shard_index} arm digest mismatch")
    if payload.get("r9_execution_contract") != execution:
        raise ValueError(f"R9 shard {shard_index} execution policy mismatch")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"R9 shard {shard_index} checkpoint contract is missing")
    for field, expected in (
        ("sha256", preflight["checkpoint"]["sha256"]),
        ("attention_backend_requested", R9_ATTENTION_BACKEND),
        ("attention_backend_resolved", R9_ATTENTION_BACKEND),
    ):
        if checkpoint.get(field) != expected:
            raise ValueError(f"R9 shard {shard_index} checkpoint {field} mismatch")
    run_config = payload.get("config")
    if not isinstance(run_config, Mapping):
        raise ValueError(f"R9 shard {shard_index} effective config is missing")
    if run_config.get("determinism_policy") != R9_DETERMINISM_POLICY:
        raise ValueError(f"R9 shard {shard_index} determinism policy mismatch")
    if run_config.get("attention_backend") != R9_ATTENTION_BACKEND:
        raise ValueError(f"R9 shard {shard_index} attention backend mismatch")
    for field, expected in (
        ("checkpoint_sha256", preflight["checkpoint"]["sha256"]),
        ("sample_id_manifest_sha256", preflight["sample_manifest"]["sha256"]),
        ("semigroup_preflight_contract_sha256", canonical_json_sha256(preflight)),
        ("arm_config_sha256", arm_sha256),
    ):
        if run_config.get(field) != expected:
            raise ValueError(
                f"R9 shard {shard_index} effective config {field} mismatch"
            )
    resume = payload.get("resume_contract")
    manifest = (
        resume.get("input_sample_manifest") if isinstance(resume, Mapping) else None
    )
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("sha256") != preflight["sample_manifest"]["sha256"]
    ):
        raise ValueError(f"R9 shard {shard_index} input manifest mismatch")


def _read_manifest_ids(path: Path) -> list[str]:
    ids = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid R9 manifest JSON at line {line_number}"
                ) from exc
            sample_id = row.get("sample_id") if isinstance(row, Mapping) else None
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(
                    "R9 manifest contains an invalid or duplicate sample_id"
                )
            seen.add(sample_id)
            ids.append(sample_id)
    return ids


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _finite_metric(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid R9 {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite R9 {label}: {value!r}")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate an R9 percentile from no values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"),
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_sequence(value: Any, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result = [float(item) for item in value]
    if not result or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _finite_open_unit(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError(f"{label} must be finite and within (0,1)")
    return parsed


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text


def _bound_contract_path(
    config: Mapping[str, Any], schedule: Mapping[str, Any], field: str
) -> Path:
    config_value = config.get(field)
    schedule_value = schedule.get(field)
    if not config_value or not schedule_value:
        raise ValueError(f"R9 locked schedule requires {field} in config and manifest")
    config_path = Path(str(config_value))
    schedule_path = Path(str(schedule_value))
    if config_path.resolve() != schedule_path.resolve():
        raise ValueError(f"R9 config {field} disagrees with the locked schedule")
    if not schedule_path.is_file():
        raise FileNotFoundError(
            f"R9 locked {field} file does not exist: {schedule_path}"
        )
    return schedule_path


def _bound_file_sha256(
    path: Path,
    schedule: Mapping[str, Any],
    digest_field: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> str:
    declared = _require_sha256(schedule.get(digest_field), digest_field)
    if config is not None:
        configured = _require_sha256(config.get(digest_field), f"config {digest_field}")
        if configured != declared:
            raise ValueError(
                f"R9 config {digest_field} disagrees with the locked schedule"
            )
    actual = _sha256_path(path)
    if actual != declared:
        raise ValueError(f"R9 locked {digest_field} does not match the bound file")
    return declared


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
