from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml

from safa.evaluation.r9_generation_batch_benchmark import (
    validate_generation_batch_benchmark_contract,
)

CHILD_CAMPAIGN_ID = "r9-report-only-formal-v9"
SOURCE_CAMPAIGN_ID = "r9-report-only-formal-v8"
SEMIGROUP_CLOSURE_CAMPAIGN_ID = "r9-report-only-formal-v2"
SOURCE_SUPERSESSION_SHA256 = (
    "f4323db51df0c4980a3b8160bd741ec72aa45a4836bf3c1f4fde5ee0f86a83f0"
)
WINNER_ARM_ID = "paper_eta_0p125"
SOURCE_ROOT = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v8/confirm512_supersessions/report-only-v3/"
    + SOURCE_SUPERSESSION_SHA256
)
SOURCE_RUNTIME = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v8/campaign_runtime.json"
)
BATCH_BENCHMARK = (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v8/generation_batch_benchmark.json"
)
BASE_RUNTIME = "configs/medium_v2/experiments/r9_meanflow_campaign.yaml"
REQUEST_CONFIG = (
    "configs/medium_v2/experiments/"
    "r9_meanflow_full_continuation_campaign_v9.yaml"
)
FULL_E2E_REQUEST = (
    "configs/medium_v2/experiments/r9_full_e2e_v1.yaml"
)


class FullContinuationContractError(ValueError):
    """Raised when the frozen v3-winner-to-Full chain is invalid."""


def expected_source_from_full_continuation(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _mapping(value, "Full continuation")
    source = _mapping(payload.get("source"), "Full continuation source")
    selected = payload.get("selected_arms")
    batch = _mapping(
        _mapping(payload.get("bindings"), "Full bindings").get(
            "generation_batch_benchmark"
        ),
        "batch binding",
    )
    if not isinstance(selected, list) or len(selected) != 1:
        raise FullContinuationContractError("Full continuation must select one arm")
    winner = _mapping(selected[0], "selected winner")
    return {
        "supersession_contract_sha256": _mapping(
            source.get("supersession"), "source supersession"
        ).get("contract_sha256"),
        "gate_contract_sha256": _mapping(
            source.get("gate"), "source gate"
        ).get("contract_sha256"),
        "selection_sha256": _mapping(
            source.get("selection"), "source selection"
        ).get("contract_sha256"),
        "supersession_result_sha256": _mapping(
            source.get("result"), "source result"
        ).get("contract_sha256"),
        "winner_arm_id": winner.get("arm_id"),
        "campaign_runtime_sha256": _mapping(
            source.get("runtime"), "source runtime"
        ).get("contract_sha256"),
        "generation_batch_benchmark_sha256": batch.get("contract_sha256"),
    }


def build_full_continuation_selection_contract(
    *, repo_root: Path, expected_source: Mapping[str, Any]
) -> dict[str, Any]:
    source = _load_source_chain(repo_root, expected_source)
    winner = _mapping(source["selection"]["winner"], "winner")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_continuation_selection_v1",
        "campaign_id": CHILD_CAMPAIGN_ID,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_supersession_contract_sha256": source["supersession"][
            "supersession_contract_sha256"
        ],
        "source_gate_contract_sha256": source["gate"]["gate_contract_sha256"],
        "source_selection_sha256": source["selection"]["selection_sha256"],
        "source_supersession_result_sha256": source["result"][
            "supersession_result_sha256"
        ],
        # Kept for the Full evaluator's explicit upstream gate binding.
        "gate_contract_sha256": source["gate"]["gate_contract_sha256"],
        "winner": {
            "arm_id": winner["arm_id"],
            "config_sha256": winner["config_sha256"],
            "output_sha256": winner["source_generation_output_sha256"],
        },
        "manifests": {
            name: source["runtime"]["manifests"][name]["sha256"]
            for name in sorted(source["runtime"]["manifests"])
        },
        "checkpoint_sha256": source["runtime"]["checkpoint"]["sha256"],
        "winner_locked": True,
        "reselection_allowed": False,
    }
    payload["selection_sha256"] = _canonical_digest(payload, "selection_sha256")
    return payload


def validate_full_continuation_selection_contract(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _mapping(value, "Full selection")
    _verify_digest(normalized, "selection_sha256")
    expected = build_full_continuation_selection_contract(
        repo_root=repo_root, expected_source=expected_source
    )
    if normalized != expected:
        raise FullContinuationContractError(
            "Full selection disagrees with frozen v3 evidence"
        )
    return normalized


def build_full_continuation_contract(
    *, repo_root: Path, expected_source: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    source = _load_source_chain(root, expected_source)
    selection = build_full_continuation_selection_contract(
        repo_root=root, expected_source=expected_source
    )
    path, content, selection_binding = full_selection_binding(
        selection, repo_root=root
    )
    runtime = source["runtime"]
    current_evaluation = _current_evaluation_binding(root)
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_continuation_v1",
        "child_campaign_id": CHILD_CAMPAIGN_ID,
        "start_phase": "full",
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "semigroup_closure_campaign_id": SEMIGROUP_CLOSURE_CAMPAIGN_ID,
        "selection": {
            **selection_binding,
            "path": str(path.relative_to(root)),
            "prospective_file_sha256": hashlib.sha256(content).hexdigest(),
        },
        "source": source["bindings"],
        "selected_arms": [dict(selection["winner"])],
        "bindings": {
            "checkpoint": dict(runtime["checkpoint"]),
            "manifest_contracts_sha256": runtime["manifest_contracts_sha256"],
            "manifests": {
                name: dict(runtime["manifests"][name])
                for name in sorted(runtime["manifests"])
            },
            "generation_batch_benchmark": source["batch_binding"],
            "generation_batch_policy": {
                "batch_size": 2,
                "workers_per_gpu": 2,
                "physical_gpus": [0, 1, 2, 3],
                "batch4_equivalent": False,
                "source_campaign_id": SOURCE_CAMPAIGN_ID,
                "source_continuation_contract_sha256": runtime["continuation"][
                    "contract_sha256"
                ],
            },
            "full_e2e_requirement": _load_full_e2e_requirement(
                root, runtime["manifests"]["full_2048"]
            ),
            "source_evaluation_provenance": {
                "classification": (
                    "historical_v8_runtime_provenance_not_execution_authority"
                ),
                "campaign_runtime_sha256": runtime["campaign_runtime_sha256"],
                "evaluation": dict(runtime["evaluation"]),
            },
            "current_evaluation": current_evaluation,
            "evaluator_smoke_requests": _load_evaluator_smoke_requests(root),
            "heldout_assets": source["heldout_assets"],
            "determinism_policy_sha256": runtime["determinism_policy_sha256"],
            "attention_backend": runtime["attention_backend"],
            "schedule": dict(runtime["schedule"]),
            "semigroup_gate": dict(runtime["semigroup_gate"]),
        },
        "policy": {
            "allowed_phases": ["full"],
            "winner_locked_before_full": True,
            "reselection_allowed": False,
            "upstream_phase_execution_allowed": False,
            "heldout_execution_count": 1,
            "retry_count": 0,
        },
    }
    if payload["bindings"]["attention_backend"] != "native":
        raise FullContinuationContractError("Full continuation requires native attention")
    payload["full_continuation_sha256"] = _canonical_digest(
        payload, "full_continuation_sha256"
    )
    return payload


def _current_evaluation_binding(root: Path) -> dict[str, Any]:
    base = _mapping(
        yaml.safe_load((root / BASE_RUNTIME).read_text(encoding="utf-8")),
        "current base runtime",
    )
    evaluation = _mapping(base.get("evaluation"), "current evaluation")
    worker = _mapping(evaluation.get("worker"), "current worker")
    quality = _mapping(evaluation.get("quality"), "current quality")
    quality_script = _mapping(quality.get("script"), "current quality script")
    wrapper = _contained_file(
        root, root / str(worker["path"]), "current worker wrapper"
    )
    implementation = _contained_file(
        root,
        root / str(worker["implementation_path"]),
        "current worker implementation",
    )
    quality_path = _contained_file(
        root,
        root / str(quality_script["path"]),
        "current quality script",
    )
    payload = {
        "classification": "canonical_current_v9_execution_authority",
        "worker": {
            "path": worker["path"],
            "sha256": _file_sha256(wrapper),
            "implementation_path": worker["implementation_path"],
            "implementation_sha256": _file_sha256(implementation),
        },
        "quality_script": {
            "path": quality_script["path"],
            "sha256": _file_sha256(quality_path),
        },
        "arcface_declaration_sha256": _json_digest(
            _mapping(evaluation.get("arcface"), "current ArcFace")
        ),
    }
    payload["current_evaluation_sha256"] = _canonical_digest(
        payload, "current_evaluation_sha256"
    )
    return payload


def validate_full_continuation_contract(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _mapping(value, "Full continuation")
    _verify_digest(normalized, "full_continuation_sha256")
    expected = build_full_continuation_contract(
        repo_root=repo_root, expected_source=expected_source
    )
    if normalized != expected:
        raise FullContinuationContractError(
            "Full continuation disagrees with frozen v3 evidence"
        )
    return normalized


def full_selection_binding(
    value: Mapping[str, Any], *, repo_root: Path
) -> tuple[Path, bytes, dict[str, str]]:
    payload = _mapping(value, "Full selection")
    _verify_digest(payload, "selection_sha256")
    root = Path(repo_root).resolve()
    path = root / (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        f"{CHILD_CAMPAIGN_ID}/full_continuation_selection.json"
    )
    content = _contract_bytes(payload)
    return path, content, {
        "path": str(path.relative_to(root)),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "contract_sha256": payload["selection_sha256"],
    }


def full_continuation_contract_binding(
    value: Mapping[str, Any], *, repo_root: Path
) -> tuple[Path, bytes, dict[str, str]]:
    payload = _mapping(value, "Full continuation")
    _verify_digest(payload, "full_continuation_sha256")
    root = Path(repo_root).resolve()
    path = root / (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        f"{CHILD_CAMPAIGN_ID}/full_continuation_contract.json"
    )
    content = _contract_bytes(payload)
    return path, content, {
        "path": str(path.relative_to(root)),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "contract_sha256": payload["full_continuation_sha256"],
    }


def materialize_full_continuation_contract(
    *, repo_root: Path, expected_source: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    selection = build_full_continuation_selection_contract(
        repo_root=repo_root, expected_source=expected_source
    )
    selection_path, selection_content, _ = full_selection_binding(
        selection, repo_root=repo_root
    )
    contract = build_full_continuation_contract(
        repo_root=repo_root, expected_source=expected_source
    )
    contract_path, contract_content, binding = full_continuation_contract_binding(
        contract, repo_root=repo_root
    )
    _write_exclusive(selection_path, selection_content)
    _write_exclusive(contract_path, contract_content)
    return contract, binding, selection


def _load_source_chain(
    repo_root: Path, expected_source: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    expected = _mapping(expected_source, "expected source")
    fields = {
        "supersession_contract_sha256",
        "gate_contract_sha256",
        "selection_sha256",
        "supersession_result_sha256",
        "winner_arm_id",
        "campaign_runtime_sha256",
        "generation_batch_benchmark_sha256",
    }
    if set(expected) != fields:
        raise FullContinuationContractError("expected source fields are not canonical")
    if expected["supersession_contract_sha256"] != SOURCE_SUPERSESSION_SHA256:
        raise FullContinuationContractError("v3 supersession SHA256 changed")
    if expected["winner_arm_id"] != WINNER_ARM_ID:
        raise FullContinuationContractError("v3 winner changed")
    source_root = root / SOURCE_ROOT
    specs = {
        "supersession": (
            source_root / "supersession_contract.json",
            "supersession_contract_sha256",
            "safa_r9_confirm512_report_only_supersession_v3",
        ),
        "gate": (
            source_root / "confirm512/gate_contract_v3.json",
            "gate_contract_sha256",
            "safa_r9_confirm512_report_only_gate_v3",
        ),
        "selection": (
            source_root / "selection.json",
            "selection_sha256",
            "safa_r9_confirm512_report_only_selection_v3",
        ),
        "result": (
            source_root / "supersession_result.json",
            "supersession_result_sha256",
            None,
        ),
        "runtime": (
            root / SOURCE_RUNTIME,
            "campaign_runtime_sha256",
            None,
        ),
    }
    payloads: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for name, (path, field, contract_type) in specs.items():
        payloads[name], bindings[name] = _read_bound(
            root, path, field, contract_type
        )
        if payloads[name][field] != _sha(expected[field], field):
            raise FullContinuationContractError(f"{field} changed")
    supersession = payloads["supersession"]
    gate = payloads["gate"]
    selection = payloads["selection"]
    result = payloads["result"]
    runtime = payloads["runtime"]
    winner = _mapping(selection.get("winner"), "winner")
    if (
        any(
            row.get("campaign_id") != SOURCE_CAMPAIGN_ID
            for row in (supersession, gate, selection, runtime)
        )
        or gate.get("supersession_contract_sha256")
        != supersession["supersession_contract_sha256"]
        or selection.get("supersession_contract_sha256")
        != supersession["supersession_contract_sha256"]
        or selection.get("gate_contract_sha256") != gate["gate_contract_sha256"]
        or result.get("gate_contract_sha256") != gate["gate_contract_sha256"]
        or result.get("selection_sha256") != selection["selection_sha256"]
        or gate.get("selected_arm_ids") != [WINNER_ARM_ID]
        or winner.get("arm_id") != WINNER_ARM_ID
        or result.get("winner_arm_id") != WINNER_ARM_ID
        or selection.get("next_stage") != "new_v9_full_continuation_required"
        or selection.get("reselection_allowed") is not False
        or result.get("generation_execution_count") != 0
        or result.get("evaluator_execution_count") != 0
    ):
        raise FullContinuationContractError("v3 winner chain changed")
    runtime_continuation = _mapping(runtime["continuation"], "v8 continuation")
    benchmark_path = root / BATCH_BENCHMARK
    benchmark = validate_generation_batch_benchmark_contract(
        _read_json(benchmark_path, "batch benchmark"),
        repo_root=root,
        expected_campaign_id=SOURCE_CAMPAIGN_ID,
        expected_continuation_contract_sha256=_sha(
            runtime_continuation["contract_sha256"], "v8 continuation SHA256"
        ),
    )
    if benchmark["generation_batch_benchmark_sha256"] != _sha(
        expected["generation_batch_benchmark_sha256"], "batch benchmark SHA256"
    ):
        raise FullContinuationContractError("batch benchmark changed")
    decision = _mapping(benchmark["decision"], "batch decision")
    if (
        benchmark.get("status") != "ready"
        or decision.get("selected_batch_size") != 2
        or decision.get("all_arms_bit_identical") is not False
        or decision.get("reason") != "batch2_required_due_to_batch4_non_equivalence"
        or int(decision.get("selected_slots_per_gpu", 0)) < 2
    ):
        raise FullContinuationContractError("formal batch=2 decision changed")
    manifests = _mapping(runtime["manifests"], "manifests")
    if set(manifests) != {
        "arcface_clean_pool",
        "calibration_64",
        "full_2048",
        "full_visual_64",
        "validate_512",
    }:
        raise FullContinuationContractError("manifest set changed")
    for name, value in manifests.items():
        _verify_file_binding(root, value, f"manifest {name}")
    _verify_file_binding(root, runtime["checkpoint"], "checkpoint")
    base = _mapping(
        yaml.safe_load((root / BASE_RUNTIME).read_text(encoding="utf-8")),
        "base runtime",
    )
    assets = _mapping(base["heldout_assets"], "heldout assets")
    heldout_assets = {}
    for name in ("e1", "e2", "facenet", "adaface"):
        heldout_assets[name] = _mapping(assets[name], f"heldout {name}")
        _verify_file_binding(root, heldout_assets[name], f"heldout {name}")
    return {
        **payloads,
        "bindings": bindings,
        "heldout_assets": heldout_assets,
        "batch_binding": {
            "path": BATCH_BENCHMARK,
            "file_sha256": _file_sha256(benchmark_path),
            "contract_sha256": benchmark["generation_batch_benchmark_sha256"],
        },
    }


def _load_evaluator_smoke_requests(root: Path) -> dict[str, Any]:
    from safa.evaluation.r9_full_smoke_supersession import (
        _bind_failed_v1,
        _bind_prepared_v2,
    )

    namespace = root / (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        "r9-report-only-formal-v9/evaluator_smoke_supersessions/"
        "full-smoke-v2/"
        "ff8152dd8529bae94c0f81668477299fa9f03303fb22dae73c9d479217485df1"
    )
    if not (namespace / "smoke_supersession_contract.json").is_file():
        raise FullContinuationContractError(
            "historical Full smoke provenance is not materialized"
        )
    contract, contract_binding = _read_bound(
        root,
        namespace / "smoke_supersession_contract.json",
        "smoke_supersession_sha256",
        "safa_r9_full_smoke_supersession_v2",
    )
    if (
        contract["smoke_supersession_sha256"]
        != "ff8152dd8529bae94c0f81668477299fa9f03303fb22dae73c9d479217485df1"
        or _mapping(contract.get("execution"), "historical smoke execution").get(
            "v2_execution_count"
        )
        != 0
        or contract.get("failed_v1") != _bind_failed_v1(root)
        or contract.get("prepared_v2_superseded") != _bind_prepared_v2(root)
    ):
        raise FullContinuationContractError(
            "historical Full smoke provenance changed"
        )
    bindings = {}
    for kind in ("arcface", "quality"):
        artifact_root = (namespace / kind).resolve()
        try:
            artifact_root.relative_to(root)
        except ValueError as error:
            raise FullContinuationContractError(
                f"{kind} smoke root escapes repo"
            ) from error
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise FullContinuationContractError(
                f"{kind} smoke requests are not materialized"
            )
        observed_inventory = {
            path.name for path in artifact_root.iterdir() if path.is_file()
        }
        if observed_inventory != {"request.json", "request_claim.json"}:
            raise FullContinuationContractError(
                f"{kind} historical smoke inventory changed"
            )
        request, request_binding = _read_bound(
            root,
            artifact_root / "request.json",
            "evaluator_request_sha256",
            "safa_r9_phase_evaluator_request_v1",
        )
        claim, claim_binding = _read_bound(
            root,
            artifact_root / "request_claim.json",
            "smoke_request_claim_sha256",
            "safa_r9_evaluator_resource_smoke_request_v2",
        )
        if (
            claim.get("kind") != kind
            or claim.get("evaluator_request_sha256")
            != request["evaluator_request_sha256"]
            or claim.get("retry_allowed") is not False
            or claim.get("smoke_supersession_sha256")
            != contract["smoke_supersession_sha256"]
        ):
            raise FullContinuationContractError(
                f"{kind} smoke request binding changed"
            )
        bindings[kind] = {
            "artifact_root": str(artifact_root.relative_to(root)),
            "request": request_binding,
            "request_claim": claim_binding,
        }
    expected_heldout = {
        "mode": "exclusive_single_official_run",
        "smoke_execution": "sealed_until_winner_lock",
        "global_exclusive_slots": 16,
        "ram_admission_percent": 85,
        "ram_hard_limit_percent": 90,
    }
    bindings["heldout"] = expected_heldout
    bindings["smoke_supersession"] = contract_binding
    bindings["classification"] = (
        "historical_invalid_smoke_provenance_only_not_a_full_resource_profile"
    )
    bindings["request_set_sha256"] = _canonical_digest(
        bindings, "request_set_sha256"
    )
    return bindings


def _load_full_e2e_requirement(
    root: Path, parent_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    request_path = _contained_file(
        root, root / FULL_E2E_REQUEST, "Full E2E request"
    )
    request = _mapping(
        yaml.safe_load(request_path.read_text(encoding="utf-8")),
        "Full E2E request",
    )
    expected = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_request_v1",
        "campaign_id": CHILD_CAMPAIGN_ID,
        "manifest": {
            "path": (
                "configs/medium_v2/experiments/r9_manifests/full_smoke_8.jsonl"
            ),
            "sha256": (
                "04a7d89db541b065755c965505bb26b1e58aea306cc59c1717f251ec32dfc87f"
            ),
            "parent_manifest": "full_2048",
        },
        "sample_count": 8,
        "phase": "full",
        "seed": 7919,
        "batch_size": 2,
        "required_arms": ["native", WINNER_ARM_ID],
        "evaluator_tasks": ["arcface", "quality_native", "quality_candidate"],
        "resource_policy": {
            "policy_id": "frozen_conservative_e2e_v1",
            "claim_type": (
                "preregistered_exclusive_upper_bound_not_measured_profile"
            ),
            "source": "pre_execution_protocol_registration",
            "rationale": (
                "e2e_bootstrap_must_not_depend_on_or_reuse_a_prior_campaign_"
                "evaluator_profile"
            ),
            "gpu_indices": [0, 1, 2, 3],
            "generation": {
                "gpu_slot_claim_bytes": 17179869184,
                "ram_slot_budget_bytes": 17179869184,
                "max_slots_per_gpu": 2,
                "concurrent_workers": 2,
            },
            "evaluator": {
                "gpu_slot_claim_bytes": 17179869184,
                "ram_slot_budget_bytes": 17179869184,
                "global_exclusive": True,
                "concurrent_workers": 1,
            },
            "admission": {
                "minimum_free_gpu_bytes": 2147483648,
                "ram_percent_below": 85,
                "disk_percent_below": 85,
                "unknown_gpu_pid_count": 0,
                "initial_swap_io_pages": 0,
            },
            "hard_stop": {
                "gpu_memory_percent_at_or_above": 90,
                "ram_percent_at_or_above": 90,
                "disk_percent_at_or_above": 90,
                "cpu_percent_at_or_above": 90,
                "temperature_c_above": 85,
                "swap_io_positive": True,
                "sustained_sample_count": 2,
            },
        },
        "retry_count": 0,
        "gate_path": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            f"{CHILD_CAMPAIGN_ID}/full_e2e/gate_contract.json"
        ),
    }
    if request != expected:
        raise FullContinuationContractError("Full E2E request changed")
    manifest_binding = _mapping(request["manifest"], "Full E2E manifest")
    e2e_manifest = _contained_file(
        root, root / manifest_binding["path"], "Full E2E manifest"
    )
    if _file_sha256(e2e_manifest) != manifest_binding["sha256"]:
        raise FullContinuationContractError("Full E2E manifest SHA256 changed")
    parent_path = _contained_file(
        root, root / str(parent_manifest["path"]), "full_2048 manifest"
    )
    if _file_sha256(parent_path) != parent_manifest["sha256"]:
        raise FullContinuationContractError("full_2048 manifest SHA256 changed")
    e2e_ids = [
        _mapping(json.loads(line), "Full E2E row").get("sample_id")
        for line in e2e_manifest.read_text(encoding="utf-8").splitlines()
    ]
    parent_ids = [
        _mapping(json.loads(line), "full_2048 row").get("sample_id")
        for line in parent_path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(e2e_ids) != 8
        or len(set(e2e_ids)) != 8
        or e2e_ids != parent_ids[:8]
    ):
        raise FullContinuationContractError(
            "Full E2E manifest is not the frozen first 8 full_2048 IDs"
        )
    return {
        "request": {
            "path": FULL_E2E_REQUEST,
            "file_sha256": _file_sha256(request_path),
        },
        "manifest": {
            "path": manifest_binding["path"],
            "file_sha256": manifest_binding["sha256"],
            "sample_count": 8,
            "parent_path": parent_manifest["path"],
            "parent_sha256": parent_manifest["sha256"],
        },
        "gate_path": request["gate_path"],
        "policy": {
            "phase": "full",
            "seed": 7919,
            "batch_size": 2,
            "arms": ["native", WINNER_ARM_ID],
            "evaluator_tasks": [
                "arcface",
                "quality_native",
                "quality_candidate",
            ],
            "resource_policy": request["resource_policy"],
            "retry_count": 0,
        },
    }


def _read_bound(
    root: Path,
    path: Path,
    digest_field: str,
    contract_type: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = _contained_file(root, path, digest_field)
    payload = _read_json(resolved, digest_field)
    if contract_type is not None and payload.get("contract_type") != contract_type:
        raise FullContinuationContractError(f"{digest_field} contract type changed")
    _verify_digest(payload, digest_field)
    return payload, {
        "path": str(resolved.relative_to(root)),
        "file_sha256": _file_sha256(resolved),
        "contract_sha256": payload[digest_field],
    }


def _verify_file_binding(root: Path, value: Any, label: str) -> None:
    binding = _mapping(value, label)
    if not {"path", "sha256"}.issubset(binding):
        raise FullContinuationContractError(f"{label} binding changed")
    path = _contained_file(root, root / str(binding["path"]), label)
    if _file_sha256(path) != _sha(binding["sha256"], f"{label} SHA256"):
        raise FullContinuationContractError(f"{label} file changed")


def _contained_file(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise FullContinuationContractError(f"{label} path escapes repo") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise FullContinuationContractError(f"{label} is not a regular repo file")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullContinuationContractError(f"invalid {label}: {path}") from error


def _verify_digest(value: Mapping[str, Any], field: str) -> None:
    if _sha(value.get(field), field) != _canonical_digest(value, field):
        raise FullContinuationContractError(f"{field} digest mismatch")


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _contract_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise FullContinuationContractError("Full continuation artifact differs")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
        os.link(temporary, path)
    except FileExistsError as error:
        raise FullContinuationContractError("Full continuation creation raced") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullContinuationContractError(f"{label} must be a mapping")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise FullContinuationContractError(f"{label} must be lowercase SHA256")
    return value
