#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml

from safa.evaluation.r9_campaign_contracts import (
    build_a_gate_contract,
    build_b_gate_contract,
    build_c_gate_contract,
    build_d_gate_contract,
    build_heldout_seal_contract,
    build_resource_smoke_contract,
    build_selection_contract,
    validate_campaign_runtime,
    validate_gate_contract,
    validate_manifest_contracts,
    validate_resource_smoke_contract,
    validate_selection_contract,
    write_immutable_contract,
)
from safa.evaluation.r9_continuation_contracts import (
    R9_CONTINUATION_BASE_RUNTIME_PATH,
    R9_CONTINUATION_REQUEST_PATH,
    build_continuation_contract,
    continuation_contract_binding,
    materialize_continuation_evaluator_smoke_requests as _materialize_continuation_evaluator_smoke_requests,
    materialize_continuation_contract,
    normalize_continuation_source,
    validate_continuation_contract,
)
from safa.evaluation.meanflow_guidance_runner import (
    resolve_frozen_effective_guidance_config,
)
from safa.evaluation.r9_phase_results import (
    AWAITING_VISUAL_REVIEW_EXIT_CODE,
    ArcFaceEvaluationRequest,
    ArcFaceEvaluator,
    HeldoutEvaluationRequest,
    HeldoutEvaluator,
    PhaseClosureOutcome,
    PhaseResultsRequest,
    QualityEvaluationRequest,
    QualityEvaluator,
    RunEvidenceSpec,
    canonical_r9_algorithm_config_digest,
    materialize_phase_results,
    resume_phase_results,
)
from safa.evaluation.r9_evaluator_resources import (
    materialize_evaluator_resource_profiles,
)
from safa.evaluation.r9_resources import (
    AdmissionStatus,
    CampaignFailedError,
    FailureKind,
    FcntlSlotLockBackend,
    R9PeerStatusStore,
    R9ResourceScheduler,
    ResourceContractError,
    StaleSlotLeaseError,
    SystemResourceProbe,
    WorkerRequest,
    gpu_slot_capacity,
)
from safa.evaluation.r9_semigroup_campaign_closure import (
    resolve_formal_campaign_semigroup_closure,
)


RUNTIME_CONFIG = Path("configs/medium_v2/experiments/r9_meanflow_campaign.yaml")
CONTINUATION_RUNTIME_CONFIG = Path(R9_CONTINUATION_REQUEST_PATH)
CONTINUATION_CHILD_CAMPAIGN_ID = "r9-report-only-formal-v6"
REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("preflight", "diagnose", "calibrate", "confirm512", "full")
CAMPAIGN_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
RESOURCE_SMOKE_ROOT_EXIT_WAIT_SECONDS = 1.0


class _ProcessTreeRootExitObserved(RuntimeError):
    """Signal that the root has entered an OS-confirmable exit state."""

    def __init__(self, pid: int, reason: str) -> None:
        self.pid = pid
        self.reason = reason
        super().__init__(f"process-tree root {pid} was observed {reason}")


@dataclass(frozen=True)
class RunSpec:
    phase: str
    logical_run_id: str
    arm_ref: str
    seed: int
    repeat_index: int | None
    shard_index: int
    num_shards: int
    sample_count: int
    manifest_key: str
    runtime_config: Path
    output_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class PhasePlan:
    phase: str
    campaign_id: str
    campaign_root: Path
    logical_run_count: int
    runs: tuple[RunSpec, ...]

    @property
    def shard_count(self) -> int:
        return len(self.runs)


@dataclass
class ActiveWorker:
    run: RunSpec
    worker_id: str
    process: Any
    launch_ordinal: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the immutable SAFA R9 FMRG campaign; dry-run is the default."
    )
    parser.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    args = parser.parse_args(argv)
    if CAMPAIGN_ID_PATTERN.fullmatch(args.campaign_id) is None:
        parser.error("--campaign-id must be an immutable lowercase slug")
    if args.execute and not args.allow_busy_gpus:
        parser.error("R9 execution requires explicit --allow-busy-gpus")
    if (
        args.campaign_id == CONTINUATION_CHILD_CAMPAIGN_ID
        and args.phase in {"preflight", "diagnose"}
    ):
        parser.error("continuation child rejects preflight and diagnose")
    return args


def load_runtime_config(path: Path = RUNTIME_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("R9 campaign runtime YAML must contain a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("R9 campaign runtime YAML must use schema_version=1")
    if payload.get("contract_type") != "safa_r9_campaign_runtime_v1":
        raise ValueError("R9 campaign runtime contract_type mismatch")
    return payload


def load_continuation_request(
    path: Path = CONTINUATION_RUNTIME_CONFIG,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], Path, dict[str, str]]:
    request_path = _repo_path(repo_root, path, "continuation request")
    request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping) or set(request) != {
        "schema_version",
        "contract_type",
        "child_campaign_id",
        "base_runtime",
        "source",
        "evaluator_resources",
    }:
        raise ValueError("continuation request fields are not canonical")
    if (
        request.get("schema_version") != 1
        or request.get("contract_type") != "safa_r9_continuation_request_v1"
        or request.get("child_campaign_id") != CONTINUATION_CHILD_CAMPAIGN_ID
    ):
        raise ValueError("continuation request type or child campaign mismatch")
    base = _mapping(request.get("base_runtime"), "continuation base runtime")
    if set(base) != {"path", "sha256"}:
        raise ValueError("continuation base runtime fields are not canonical")
    base_path = _repo_path(repo_root, base.get("path"), "continuation base runtime")
    if str(base_path.relative_to(repo_root.resolve())) != R9_CONTINUATION_BASE_RUNTIME_PATH:
        raise ValueError("continuation request changed the base runtime path")
    if _sha256_path(base_path) != _require_sha256(
        base.get("sha256"), "continuation base runtime SHA256"
    ):
        raise ValueError("continuation base runtime SHA256 mismatch")
    source = normalize_continuation_source(request.get("source"))
    runtime = load_runtime_config(base_path)
    evaluation = _mapping(runtime.get("evaluation"), "continuation evaluation")
    worker = _mapping(evaluation.get("worker"), "continuation evaluator worker")
    worker["sha256"] = _sha256_path(
        _repo_path(repo_root, worker.get("path"), "continuation evaluator entrypoint")
    )
    worker["implementation_sha256"] = _sha256_path(
        _repo_path(
            repo_root,
            worker.get("implementation_path"),
            "continuation evaluator implementation",
        )
    )
    evaluation["worker"] = worker
    quality = _mapping(evaluation.get("quality"), "continuation quality")
    quality_script = _mapping(quality.get("script"), "continuation quality script")
    quality_script["sha256"] = _sha256_path(
        _repo_path(repo_root, quality_script.get("path"), "continuation quality script")
    )
    quality["script"] = quality_script
    evaluation["quality"] = quality
    evaluator_resources = _mapping(
        request.get("evaluator_resources"), "continuation evaluator resources"
    )
    if set(evaluator_resources) != {"arcface", "quality", "heldout"}:
        raise ValueError("continuation evaluator resource fields are not canonical")
    evaluation["resource_smokes"] = evaluator_resources
    runtime["evaluation"] = evaluation
    return (
        runtime,
        Path(str(request_path.relative_to(repo_root.resolve()))),
        source,
    )


def prepare_continuation_evaluator_smoke_requests(
    *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    runtime, request_path, source = load_continuation_request(repo_root=repo_root)
    normalize_continuation_source(source)
    return _materialize_continuation_evaluator_smoke_requests(
        repo_root=repo_root,
        child_campaign_id=CONTINUATION_CHILD_CAMPAIGN_ID,
        runtime_config_path=request_path,
        runtime=runtime,
    )


def load_campaign_configuration(
    campaign_id: str,
) -> tuple[dict[str, Any], Path, dict[str, str] | None]:
    if campaign_id == CONTINUATION_CHILD_CAMPAIGN_ID:
        return load_continuation_request()
    return load_runtime_config(), RUNTIME_CONFIG, None


def _continuation_for_runtime(
    campaign_runtime: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any] | None:
    binding = campaign_runtime.get("continuation")
    if binding is None:
        return None
    row = _mapping(binding, "continuation binding")
    if set(row) != {"path", "file_sha256", "contract_sha256"}:
        raise ValueError("continuation binding fields are not canonical")
    path = _repo_path(repo_root, row.get("path"), "continuation contract")
    if _sha256_path(path) != _require_sha256(
        row.get("file_sha256"), "continuation file SHA256"
    ):
        raise ValueError("continuation file SHA256 mismatch")
    payload = validate_continuation_contract(
        _read_json_mapping(path, "continuation contract"), repo_root=repo_root
    )
    if payload["continuation_contract_sha256"] != row["contract_sha256"]:
        raise ValueError("continuation internal contract SHA256 mismatch")
    if payload["child_campaign_id"] != campaign_runtime.get("campaign_id"):
        raise ValueError("continuation child campaign ID mismatch")
    return payload


def _formal_closure_for_runtime(
    campaign_runtime: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    continuation_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    campaign_id = str(campaign_runtime.get("campaign_id", ""))
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        raise ValueError("campaign runtime has no canonical campaign ID")
    continuation = (
        validate_continuation_contract(continuation_contract, repo_root=repo_root)
        if continuation_contract is not None
        else _continuation_for_runtime(campaign_runtime, repo_root=repo_root)
    )
    closure_campaign_id = (
        continuation["parent"]["campaign_id"]
        if continuation is not None
        else campaign_id
    )
    closure = resolve_formal_campaign_semigroup_closure(
        closure_campaign_id, repo_root=repo_root
    )
    if closure is None:
        return None
    if (
        campaign_runtime.get("schedule") != closure["schedule"]
        or campaign_runtime.get("semigroup_gate") != closure["gate"]
    ):
        raise ValueError(
            "formal campaign runtime closure schedule/gate binding mismatch"
        )
    return closure


def _validate_requested_campaign_role(
    campaign_runtime: Mapping[str, Any],
    *,
    requested_phase: str,
    continuation_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    closure = _formal_closure_for_runtime(
        campaign_runtime, continuation_contract=continuation_contract
    )
    if campaign_runtime.get("continuation") is not None:
        if requested_phase in {"preflight", "diagnose"}:
            raise ValueError("continuation child rejects preflight and diagnose")
        if closure is None:
            raise ValueError("continuation parent has no sealed semigroup closure")
        return closure
    if closure is None and requested_phase != "preflight":
        raise ValueError(
            "bootstrap campaign can only execute preflight; formal phases require "
            "a sealed campaign-aware semigroup closure"
        )
    return closure


def build_phase_plan(
    runtime: Mapping[str, Any],
    *,
    phase: str,
    campaign_id: str,
    promoted_arm_ids: Sequence[str] | None = None,
    winner_arm_id: str | None = None,
) -> PhasePlan:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES!r}")
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        raise ValueError("campaign_id must be an immutable lowercase slug")
    root = Path(str(runtime["campaign_root"])) / campaign_id
    python = str(runtime["python"])
    generation_script = str(runtime["generation_script"])
    phase_config = _mapping(runtime.get("phases"), "phases").get(phase)
    phase_config = _mapping(phase_config, f"phases.{phase}")
    manifest_key = str(phase_config["manifest"])
    sample_count = _positive_int(phase_config["sample_count"], "sample_count")
    shards_per_logical_run = _positive_int(
        phase_config["shards_per_logical_run"], "shards_per_logical_run"
    )
    logical_runs = _logical_runs(
        runtime,
        phase,
        phase_config,
        promoted_arm_ids=promoted_arm_ids,
        winner_arm_id=winner_arm_id,
    )
    runs: list[RunSpec] = []
    for logical_run_id, arm_ref, seed, repeat_index in logical_runs:
        runtime_config = root / "runtime_configs" / phase / f"{logical_run_id}.yaml"
        logical_output = root / phase / logical_run_id
        for shard_index in range(shards_per_logical_run):
            output_dir = (
                logical_output
                if shards_per_logical_run == 1
                else logical_output / "shards" / f"shard_{shard_index}"
            )
            command = (
                python,
                generation_script,
                "--config",
                str(runtime_config),
                "--output-dir",
                str(output_dir),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(shards_per_logical_run),
            )
            runs.append(
                RunSpec(
                    phase=phase,
                    logical_run_id=logical_run_id,
                    arm_ref=arm_ref,
                    seed=seed,
                    repeat_index=repeat_index,
                    shard_index=shard_index,
                    num_shards=shards_per_logical_run,
                    sample_count=sample_count,
                    manifest_key=manifest_key,
                    runtime_config=runtime_config,
                    output_dir=output_dir,
                    command=command,
                )
            )
    return PhasePlan(
        phase=phase,
        campaign_id=campaign_id,
        campaign_root=root,
        logical_run_count=len(logical_runs),
        runs=tuple(runs),
    )


def build_requested_plans(
    runtime: Mapping[str, Any],
    *,
    phase: str,
    campaign_id: str,
    continuation_selected_arm_ids: Sequence[str] | None = None,
) -> tuple[PhasePlan, ...]:
    continuation = continuation_selected_arm_ids is not None
    if continuation and phase in {"preflight", "diagnose"}:
        raise ValueError("continuation child rejects preflight and diagnose")
    selected = (
        ("calibrate", "confirm512", "full")
        if continuation and phase == "all"
        else (PHASES if phase == "all" else (phase,))
    )
    return tuple(
        build_phase_plan(
            runtime,
            phase=item,
            campaign_id=campaign_id,
            promoted_arm_ids=(
                continuation_selected_arm_ids if item == "calibrate" else None
            ),
        )
        for item in selected
    )


def build_effective_campaign_runtime(
    runtime: Mapping[str, Any],
    *,
    campaign_id: str,
    repo_root: Path,
    runtime_config_path: Path = RUNTIME_CONFIG,
    continuation_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifests = _mapping(runtime.get("manifests"), "manifests")
    main_manifests = {
        name: _manifest_contract_entry(manifests, name)
        for name in ("calibration_64", "validate_512", "full_2048", "full_visual_64")
    }
    clean = _mapping(runtime.get("clean_source"), "clean_source")
    main_manifests["arcface_clean_pool"] = {
        "path": str(clean["index"]),
        "sha256": str(clean["index_sha256"]),
        "sample_count": _positive_int(clean["sample_count"], "clean sample_count"),
        "ordered_sample_id_sha256": str(clean["ordered_sample_id_sha256"]),
    }
    phases = _mapping(runtime.get("phases"), "phases")
    clean_contract = {
        "path": str(clean["index"]),
        "sha256": str(clean["index_sha256"]),
        "sample_count": _positive_int(clean["sample_count"], "clean sample_count"),
        "ordered_sample_id_sha256": str(clean["ordered_sample_id_sha256"]),
        "arcface_exact_one": clean.get("arcface_exact_one"),
    }
    construction = _mapping(
        runtime.get("manifest_construction"), "manifest_construction"
    )
    construction_contract = {
        "r8_calibration_64": dict(
            _mapping(construction.get("r8_calibration_64"), "R8 calibration")
        ),
        "diagnose_18": dict(_mapping(construction.get("diagnose_18"), "diagnose_18")),
    }
    manifest_contract = validate_manifest_contracts(
        main_manifests,
        repo_root,
        clean_source=clean_contract,
        r8_calibration_binding=construction_contract["r8_calibration_64"],
        diagnose_manifest=construction_contract["diagnose_18"],
    )
    resources = _build_effective_resources(
        runtime,
        repo_root=repo_root,
        manifests=manifest_contract["manifests"],
    )
    continuation_contract = None
    continuation_binding = None
    closure_campaign_id = campaign_id
    if continuation_source is not None:
        continuation_contract = build_continuation_contract(
            repo_root=repo_root,
            child_campaign_id=campaign_id,
            source=continuation_source,
        )
        _, _, continuation_binding = continuation_contract_binding(
            continuation_contract, repo_root=repo_root
        )
        closure_campaign_id = str(
            continuation_contract["parent"]["campaign_id"]
        )
    formal_closure = resolve_formal_campaign_semigroup_closure(
        closure_campaign_id, repo_root=repo_root
    )
    if formal_closure is None:
        schedule_binding = {
            "path": str(runtime["schedule_manifest"]),
            "file_sha256": str(runtime["schedule_manifest_sha256"]),
            "contract_sha256": str(runtime["schedule_contract_sha256"]),
        }
        gate_binding = {
            "path": str(runtime["gate_contract"]),
            "file_sha256": str(runtime["gate_contract_sha256"]),
            "contract_sha256": str(runtime["gate_canonical_sha256"]),
        }
    else:
        schedule_binding = dict(formal_closure["schedule"])
        gate_binding = dict(formal_closure["gate"])
    campaign_template = _bound_file(
        repo_root,
        runtime_config_path,
        _sha256_path(repo_root / runtime_config_path),
        "campaign template",
    )
    effective = {
        "schema_version": 1,
        "experiment_contract": str(runtime["campaign_contract"]),
        "generation_experiment_contract": str(runtime["experiment_contract"]),
        "campaign_id": campaign_id,
        "campaign_root": str(Path(str(runtime["campaign_root"])) / campaign_id),
        "campaign_template": campaign_template,
        "base_config": _bound_file(
            repo_root,
            runtime.get("base_config"),
            runtime.get("base_config_sha256"),
            "base config",
        ),
        "checkpoint": dict(_mapping(runtime.get("checkpoint"), "checkpoint")),
        "determinism_policy_sha256": str(runtime["determinism_policy_sha256"]),
        "attention_backend": str(runtime["attention_backend"]),
        "schedule": schedule_binding,
        "semigroup_gate": gate_binding,
        "seeds": {
            "preflight": [_positive_int(phases["preflight"]["seed"], "preflight seed")],
            "diagnose": [_positive_int(phases["diagnose"]["seed"], "diagnose seed")],
            "calibrate": list(phases["calibrate"]["seeds"]),
            "confirm512": [
                _positive_int(phases["confirm512"]["seed"], "confirm512 seed")
            ],
            "full": [_positive_int(phases["full"]["seed"], "full seed")],
        },
        "manifests": manifest_contract["manifests"],
        "clean_source": clean_contract,
        "manifest_construction": construction_contract,
        "resources": resources,
        "bootstrap": dict(_mapping(runtime.get("bootstrap"), "bootstrap")),
        "evaluation": _build_effective_evaluation(
            runtime,
            repo_root=repo_root,
            runtime_config=campaign_template,
        ),
        "phases": dict(phases),
    }
    if continuation_binding is not None:
        effective["continuation"] = continuation_binding
    diagnose_contract = manifest_contract["provenance"]["diagnose_18"]
    if resources.get("resource_smoke", {}).get("result") is None:
        claim = dict(effective)
        claim["campaign_claim_sha256"] = _canonical_json_sha256(claim)
        claim["campaign_runtime_sha256"] = None
        return claim, manifest_contract, diagnose_contract
    validated_runtime = validate_campaign_runtime(
        effective,
        repo_root,
        continuation_contract=continuation_contract,
    )
    return validated_runtime, manifest_contract, diagnose_contract


def _bound_file(
    repo_root: Path,
    path_value: Any,
    sha256_value: Any,
    label: str,
) -> dict[str, str]:
    path = _repo_path(repo_root, path_value, label)
    declared = _require_sha256(sha256_value, f"{label} SHA256")
    if _sha256_path(path) != declared:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": str(path.relative_to(repo_root.resolve())), "sha256": declared}


def _build_effective_resources(
    runtime: Mapping[str, Any],
    *,
    repo_root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw = _mapping(runtime.get("resources"), "resources")
    smoke = dict(_mapping(raw.get("resource_smoke"), "resource_smoke"))
    result_path = (repo_root / str(smoke["output_path"])).resolve()
    result_binding = None
    ram_slot_budget_bytes = None
    if result_path.is_file():
        result_payload = validate_resource_smoke_contract(
            _read_json_mapping(result_path, "resource smoke result")
        )
        expected = {
            "run_id": smoke["run_id"],
            "arm_id": smoke["arm_id"],
            "manifest": smoke["manifest"],
            "manifest_sha256": manifests[str(smoke["manifest"])]["sha256"],
            "checkpoint_sha256": _mapping(runtime.get("checkpoint"), "checkpoint")[
                "sha256"
            ],
        }
        if any(result_payload.get(key) != value for key, value in expected.items()):
            raise ValueError("resource smoke result binding mismatch")
        result_binding = {
            "path": str(result_path.relative_to(repo_root.resolve())),
            "file_sha256": _sha256_path(result_path),
            "contract_sha256": result_payload["resource_smoke_sha256"],
        }
        peak = _positive_int(result_payload["peak_rss_bytes"], "smoke peak RSS")
        ram_slot_budget_bytes = (peak * 110 + 99) // 100
    normalized = {
        "physical_gpus": list(raw["physical_gpus"]),
        "global_slot_lock_root": str(raw["global_slot_lock_root"]),
        "max_slots_per_gpu": raw["max_slots_per_gpu"],
        "gpu_slot_claim_bytes": raw["gpu_slot_claim_bytes"],
        "gpu_headroom_bytes": raw["gpu_headroom_bytes"],
        "ram_admission_percent": raw["ram_admission_percent"],
        "ram_hard_limit_percent": raw["ram_hard_limit_percent"],
        "require_tmux": raw["require_tmux"],
        "retry_count": raw["retry_count"],
        "resource_smoke": {
            "required": smoke["required"],
            "run_id": smoke["run_id"],
            "arm_id": smoke["arm_id"],
            "manifest": smoke["manifest"],
            "output_path": smoke["output_path"],
            "factor": smoke["factor"],
            "result": result_binding,
        },
    }
    if ram_slot_budget_bytes is not None:
        normalized["ram_slot_budget_bytes"] = ram_slot_budget_bytes
    return normalized


def _build_effective_evaluation(
    runtime: Mapping[str, Any],
    *,
    repo_root: Path,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(runtime.get("evaluation"), "evaluation")
    if set(raw) != {"worker", "quality", "arcface", "heldout", "resource_smokes"}:
        raise ValueError("evaluation fields are not canonical")
    worker = _mapping(raw.get("worker"), "evaluation worker")
    if set(worker) != {
        "path",
        "sha256",
        "implementation_path",
        "implementation_sha256",
    }:
        raise ValueError("evaluation worker fields are not canonical")
    worker_wrapper = _bound_file(
        repo_root, worker.get("path"), worker.get("sha256"), "evaluation worker wrapper"
    )
    worker_implementation = _bound_file(
        repo_root,
        worker.get("implementation_path"),
        worker.get("implementation_sha256"),
        "evaluation worker implementation",
    )
    quality = _mapping(raw.get("quality"), "quality evaluation")
    if set(quality) != {"script", "real_index", "metrics", "iqa_method", "device"}:
        raise ValueError("quality evaluation fields are not canonical")
    if quality.get("metrics") != ["fid", "kid", "niqe", "sharpness"]:
        raise ValueError("quality metrics must be fid,kid,niqe,sharpness")
    if quality.get("iqa_method") != "niqe" or quality.get("device") != "cuda:0":
        raise ValueError("quality evaluation must lock NIQE on assigned cuda:0")
    arcface = _mapping(raw.get("arcface"), "ArcFace evaluation")
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
    if set(arcface) != required_arcface:
        raise ValueError("ArcFace evaluation fields are not canonical")
    if (
        arcface.get("model_name") != "buffalo_l"
        or arcface.get("det_size") != [224, 224]
        or arcface.get("provider") != "CUDAExecutionProvider"
        or arcface.get("insightface_version") != "0.7.3"
        or arcface.get("onnxruntime_version") != "1.26.0"
    ):
        raise ValueError("ArcFace must lock buffalo_l CUDA at det_size 224")
    model_root = Path(str(arcface["model_root"])).resolve()
    model_dir = model_root / "models" / "buffalo_l"
    assets = _mapping(arcface.get("assets"), "ArcFace assets")
    if set(assets) != {
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    }:
        raise ValueError("ArcFace buffalo_l assets are incomplete")
    normalized_assets = {}
    for name in sorted(assets):
        declared = _require_sha256(assets[name], f"ArcFace asset {name} SHA256")
        path = model_dir / name
        if _sha256_path(path) != declared:
            raise ValueError(f"ArcFace asset {name} SHA256 mismatch")
        normalized_assets[name] = declared
    execution_probe = dict(
        _mapping(arcface.get("execution_probe"), "ArcFace execution probe provenance")
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
        raise ValueError("ArcFace execution probe provenance fields are not canonical")
    probe_binding = _bound_file(
        repo_root,
        execution_probe.get("path"),
        execution_probe.get("sha256"),
        "ArcFace execution probe",
    )
    probe_payload = _read_json_mapping(
        repo_root / probe_binding["path"], "ArcFace execution probe"
    )
    if not isinstance(probe_payload.get("execution"), Mapping):
        raise ValueError("ArcFace execution probe omitted execution")
    from safa.evaluation.r9_evaluator_worker import _validate_arcface_contract

    normalized_arcface = _validate_arcface_contract(
        {
            "model_name": "buffalo_l",
            "model_root": str(model_root),
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
    campaign_arcface_declaration = {
        "model_name": normalized_arcface["model_name"],
        "model_root": normalized_arcface["model_root"],
        "det_size": normalized_arcface["det_size"],
        "provider": normalized_arcface["provider"],
        "insightface_version": normalized_arcface["insightface_version"],
        "onnxruntime_version": normalized_arcface["onnxruntime_version"],
        "assets": normalized_arcface["assets"],
        "execution_probe": normalized_arcface["execution_probe"],
    }
    heldout = _mapping(raw.get("heldout"), "heldout evaluation")
    if set(heldout) != {
        "batch_size",
        "representation_image_size",
        "facenet",
        "adaface",
    }:
        raise ValueError("heldout evaluation fields are not canonical")
    if (
        heldout.get("batch_size") != 16
        or heldout.get("representation_image_size") != 224
    ):
        raise ValueError("heldout evaluation batch/image size mismatch")
    expected_recognizers = {
        "facenet": {"embedding_dim": 512, "input_size": 160},
        "adaface": {"embedding_dim": 512, "input_size": 112},
    }
    for name, expected in expected_recognizers.items():
        if dict(_mapping(heldout.get(name), name)) != expected:
            raise ValueError(f"heldout {name} settings mismatch")
    normalized_worker = {
        **worker_wrapper,
        "implementation_path": worker_implementation["path"],
        "implementation_sha256": worker_implementation["sha256"],
    }
    normalized_quality = {
        "script": _bound_file(
            repo_root,
            _mapping(quality.get("script"), "quality script").get("path"),
            _mapping(quality.get("script"), "quality script").get("sha256"),
            "quality script",
        ),
        "real_index": _bound_file(
            repo_root,
            _mapping(quality.get("real_index"), "quality real index").get("path"),
            _mapping(quality.get("real_index"), "quality real index").get("sha256"),
            "quality real index",
        ),
        "metrics": list(quality["metrics"]),
        "iqa_method": "niqe",
        "device": "cuda:0",
    }
    resource_smokes = materialize_evaluator_resource_profiles(
        raw.get("resource_smokes"),
        repo_root=repo_root,
        worker_contract=normalized_worker,
        arcface_contract_sha256=_canonical_json_sha256(normalized_arcface),
        quality_script_sha256=normalized_quality["script"]["sha256"],
        runtime_config_path=runtime_config["path"],
        runtime_config_sha256=runtime_config["sha256"],
    )
    return {
        "worker": normalized_worker,
        "quality": normalized_quality,
        "arcface": campaign_arcface_declaration,
        "heldout": {
            "batch_size": 16,
            "representation_image_size": 224,
            **expected_recognizers,
        },
        "resource_smokes": resource_smokes,
    }


def validate_diagnose_manifest(
    runtime: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    manifests = _mapping(runtime.get("manifests"), "manifests")
    declared = _mapping(manifests.get("diagnose_18"), "diagnose_18")
    path = _repo_path(repo_root, declared.get("path"), "diagnose manifest")
    if _sha256_path(path) != str(declared.get("sha256")):
        raise ValueError("diagnose_18 file SHA256 mismatch")
    rows = _read_jsonl(path)
    if len(rows) != 18 or declared.get("sample_count") != 18:
        raise ValueError("diagnose_18 must contain exactly 18 rows")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != 18:
        raise ValueError("diagnose_18 sample IDs must be unique and non-empty")
    ordered_sha256 = hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode()
    ).hexdigest()
    if ordered_sha256 != str(declared.get("ordered_sample_id_sha256")):
        raise ValueError("diagnose_18 ordered sample ID SHA256 mismatch")
    pairs: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pair_index = row.get("pair_index")
        role = row.get("role")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int):
            raise ValueError("diagnose_18 pair_index must be an integer")
        if role not in {"difficult", "control"}:
            raise ValueError("diagnose_18 role must be difficult or control")
        pair = pairs.setdefault(pair_index, {})
        if role in pair:
            raise ValueError("diagnose_18 pair has a duplicate role")
        pair[str(role)] = row
    if set(pairs) != set(range(9)) or any(
        set(pair) != {"difficult", "control"} for pair in pairs.values()
    ):
        raise ValueError("diagnose_18 must contain nine complete one-to-one pairs")
    for pair in pairs.values():
        difficult = pair["difficult"]
        control = pair["control"]
        if difficult.get("matched_control_sample_id") != control.get("sample_id"):
            raise ValueError("diagnose_18 difficult/control binding mismatch")
        if control.get("matched_difficult_sample_id") != difficult.get("sample_id"):
            raise ValueError("diagnose_18 control/difficult binding mismatch")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_diagnose_manifest_v1",
        "path": str(path.relative_to(Path(repo_root).resolve())),
        "sha256": _sha256_path(path),
        "ordered_sample_id_sha256": ordered_sha256,
        "sample_count": 18,
        "difficult_count": 9,
        "control_count": 9,
        "pair_count": 9,
    }
    payload["diagnose_manifest_contract_sha256"] = _canonical_json_sha256(payload)
    return payload


def render_dry_run(
    runtime: Mapping[str, Any],
    plans: Sequence[PhasePlan],
    *,
    effective_runtime: Mapping[str, Any] | None = None,
    manifest_contract: Mapping[str, Any] | None = None,
    diagnose_contract: Mapping[str, Any] | None = None,
) -> str:
    runtime_path = str(RUNTIME_CONFIG)
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_campaign_dry_run_v1",
        "execute": False,
        "runtime_config": runtime_path,
        "runtime_has_cli_algorithm_overrides": False,
        "campaign_id": plans[0].campaign_id,
        "campaign_runtime_sha256": None
        if effective_runtime is None
        else effective_runtime.get("campaign_runtime_sha256"),
        "campaign_claim_sha256": None
        if effective_runtime is None
        else effective_runtime.get("campaign_claim_sha256"),
        "resource_smoke_measured": bool(
            effective_runtime
            and _mapping(effective_runtime.get("resources"), "resources").get(
                "ram_slot_budget_bytes"
            )
        ),
        "manifest_contracts_sha256": None
        if manifest_contract is None
        else manifest_contract["manifest_contracts_sha256"],
        "diagnose_manifest_contract_sha256": None
        if diagnose_contract is None
        else _canonical_json_sha256(diagnose_contract),
        "phases": [
            {
                "phase": plan.phase,
                "logical_run_count": plan.logical_run_count,
                "shard_count": plan.shard_count,
                "sample_run_count": plan.logical_run_count
                * (plan.runs[0].sample_count if plan.runs else 0),
                "runs": [
                    {
                        "logical_run_id": run.logical_run_id,
                        "arm_ref": run.arm_ref,
                        "seed": run.seed,
                        "repeat_index": run.repeat_index,
                        "shard_index": run.shard_index,
                        "num_shards": run.num_shards,
                        "manifest": run.manifest_key,
                        "runtime_config": str(run.runtime_config),
                        "output_dir": str(run.output_dir),
                        "command": list(run.command),
                    }
                    for run in plan.runs
                ],
            }
            for plan in plans
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime, runtime_config_path, continuation_source = load_campaign_configuration(
        str(args.campaign_id)
    )
    effective_runtime, manifest_contract, diagnose_contract = (
        build_effective_campaign_runtime(
            runtime,
            campaign_id=str(args.campaign_id),
            repo_root=REPO_ROOT,
            runtime_config_path=runtime_config_path,
            continuation_source=continuation_source,
        )
    )
    continuation_contract = (
        build_continuation_contract(
            repo_root=REPO_ROOT,
            child_campaign_id=str(args.campaign_id),
            source=continuation_source,
        )
        if continuation_source is not None
        else None
    )
    continuation_selected = (
        [str(row["arm_id"]) for row in continuation_contract["selected_arms"]]
        if continuation_contract is not None
        else None
    )
    plans = build_requested_plans(
        runtime,
        phase=str(args.phase),
        campaign_id=str(args.campaign_id),
        continuation_selected_arm_ids=continuation_selected,
    )
    _validate_requested_campaign_role(
        effective_runtime,
        requested_phase=str(args.phase),
        continuation_contract=continuation_contract,
    )
    if not args.execute:
        print(
            render_dry_run(
                runtime,
                plans,
                effective_runtime=effective_runtime,
                manifest_contract=manifest_contract,
                diagnose_contract=diagnose_contract,
            )
        )
        return 0
    if "TMUX" not in os.environ:
        raise RuntimeError("R9 execution must be launched inside tmux")
    if continuation_source is not None:
        _, binding = materialize_continuation_contract(
            repo_root=REPO_ROOT,
            child_campaign_id=str(args.campaign_id),
            source=continuation_source,
        )
        if effective_runtime.get("continuation") != binding:
            raise RuntimeError("materialized continuation binding changed")
    if effective_runtime.get("campaign_runtime_sha256") is None:
        run_resource_smoke(
            runtime,
            effective_runtime,
            manifest_contract,
        )
        effective_runtime, manifest_contract, diagnose_contract = (
            build_effective_campaign_runtime(
                runtime,
                campaign_id=str(args.campaign_id),
                repo_root=REPO_ROOT,
                runtime_config_path=runtime_config_path,
                continuation_source=continuation_source,
            )
        )
        _validate_requested_campaign_role(
            effective_runtime,
            requested_phase=str(args.phase),
            continuation_contract=continuation_contract,
        )
    if effective_runtime.get("campaign_runtime_sha256") is None:
        raise RuntimeError("resource smoke did not produce a final campaign runtime")
    _write_immutable_bytes(
        REPO_ROOT / plans[0].campaign_root / "campaign_runtime.json",
        (
            json.dumps(
                effective_runtime,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    scheduler, gpu_bindings, peer_status_store = build_resource_scheduler(
        effective_runtime
    )
    evaluators = R9ProductionEvaluatorCallbacks(
        runtime=runtime,
        campaign_runtime=effective_runtime,
        scheduler=scheduler,
        gpu_bindings=gpu_bindings,
        peer_status_store=peer_status_store,
    )
    return execute_dynamic_campaign(
        runtime,
        effective_runtime,
        manifest_contract,
        diagnose_contract,
        requested_phase=str(args.phase),
        campaign_id=str(args.campaign_id),
        scheduler=scheduler,
        gpu_bindings=gpu_bindings,
        peer_status_store=peer_status_store,
        quality_evaluator=evaluators.quality,
        arcface_evaluator=evaluators.arcface,
        heldout_evaluator=evaluators.heldout,
    )


def execute_dynamic_campaign(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    diagnose_contract: Mapping[str, Any],
    *,
    requested_phase: str,
    campaign_id: str,
    scheduler: R9ResourceScheduler,
    gpu_bindings: Mapping[int, str],
    peer_status_store: R9PeerStatusStore,
    quality_evaluator: QualityEvaluator | None = None,
    arcface_evaluator: ArcFaceEvaluator | None = None,
    heldout_evaluator: HeldoutEvaluator | None = None,
) -> int:
    if campaign_runtime:
        _validate_requested_campaign_role(
            campaign_runtime, requested_phase=requested_phase
        )
    is_continuation = campaign_runtime.get("continuation") is not None
    if is_continuation and requested_phase in {"preflight", "diagnose"}:
        raise ValueError("continuation child rejects preflight and diagnose")
    selected_phases = (
        ("calibrate", "confirm512", "full")
        if is_continuation and requested_phase == "all"
        else (PHASES if requested_phase == "all" else (requested_phase,))
    )
    for phase in selected_phases:
        promoted, winner = resolve_phase_promotion(
            runtime,
            campaign_runtime,
            phase=phase,
            campaign_id=campaign_id,
        )
        plan = build_phase_plan(
            runtime,
            phase=phase,
            campaign_id=campaign_id,
            promoted_arm_ids=promoted,
            winner_arm_id=winner,
        )
        materialize_phase_runtime_configs(
            runtime,
            campaign_runtime,
            manifest_contract,
            plan,
        )
        closure_request = None
        if phase in {"diagnose", "calibrate", "confirm512", "full"}:
            closure_request = build_phase_results_request(
                runtime,
                campaign_runtime,
                manifest_contract,
                diagnose_contract,
                plan=plan,
                campaign_id=campaign_id,
            )
            closure = resume_phase_results(closure_request)
            if closure.status == "awaiting_visual_review":
                _print_phase_closure(phase, closure)
                return AWAITING_VISUAL_REVIEW_EXIT_CODE
            if closure.status == "complete":
                gate = finalize_phase_gate(
                    runtime,
                    campaign_runtime,
                    manifest_contract,
                    diagnose_contract,
                    phase=phase,
                    campaign_id=campaign_id,
                )
                if gate["verdict"] == "stop_zero_candidates":
                    return 0
                continue
        execute_campaign(
            (plan,),
            scheduler=scheduler,
            gpu_bindings=gpu_bindings,
            peer_status_store=peer_status_store,
        )
        if closure_request is not None:
            closure = materialize_phase_results(
                closure_request,
                quality_evaluator=quality_evaluator,
                arcface_evaluator=arcface_evaluator,
                heldout_evaluator=heldout_evaluator,
            )
            if closure.status == "awaiting_visual_review":
                _print_phase_closure(phase, closure)
                return AWAITING_VISUAL_REVIEW_EXIT_CODE
            if closure.status != "complete":
                raise RuntimeError(
                    f"{phase} generation completed without materialized phase results"
                )
            gate = finalize_phase_gate(
                runtime,
                campaign_runtime,
                manifest_contract,
                diagnose_contract,
                phase=phase,
                campaign_id=campaign_id,
            )
            if gate["verdict"] == "stop_zero_candidates":
                return 0
    return 0


def build_phase_results_request(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    diagnose_contract: Mapping[str, Any],
    *,
    plan: PhasePlan,
    campaign_id: str,
) -> PhaseResultsRequest:
    if plan.phase not in {"diagnose", "calibrate", "confirm512", "full"}:
        raise ValueError("phase results are not defined for preflight")
    if plan.campaign_id != campaign_id:
        raise ValueError("phase plan campaign ID mismatch")
    grouped: dict[str, list[RunSpec]] = {}
    for run in plan.runs:
        grouped.setdefault(run.logical_run_id, []).append(run)
    run_specs = []
    for logical_run_id in sorted(grouped):
        shards = sorted(grouped[logical_run_id], key=lambda run: run.shard_index)
        first = shards[0]
        expected_indices = list(range(first.num_shards))
        if (
            [run.shard_index for run in shards] != expected_indices
            or any(run.num_shards != first.num_shards for run in shards)
            or any(
                (run.arm_ref, run.seed, run.repeat_index)
                != (first.arm_ref, first.seed, first.repeat_index)
                for run in shards
            )
        ):
            raise ValueError(f"logical run {logical_run_id!r} shard contract mismatch")
        arm = _diagnose_arm(runtime, first.arm_ref)
        run_specs.append(
            RunEvidenceSpec(
                logical_run_id=logical_run_id,
                arm_id=first.arm_ref,
                family=str(arm["family"]),
                seed=first.seed,
                repeat_index=first.repeat_index,
                shard_output_dirs=tuple(REPO_ROOT / run.output_dir for run in shards),
            )
        )
    manifests = _mapping(manifest_contract.get("manifests"), "manifests")
    manifest_key = {
        "diagnose": "diagnose_18",
        "calibrate": "calibration_64",
        "confirm512": "validate_512",
        "full": "full_2048",
    }[plan.phase]
    manifest = (
        diagnose_contract
        if plan.phase == "diagnose"
        else _mapping(manifests.get(manifest_key), manifest_key)
    )
    phase_config = _mapping(
        _mapping(campaign_runtime.get("phases"), "phases").get(plan.phase),
        plan.phase,
    )
    source_index = _mapping(
        _mapping(
            _mapping(campaign_runtime.get("evaluation"), "evaluation").get("quality"),
            "quality evaluation",
        ).get("real_index"),
        "quality source index",
    )
    selection = None
    heldout_seal = None
    upstream_gate = None
    visual_manifest_path = None
    visual_manifest_sha256 = None
    campaign_root = REPO_ROOT / plan.campaign_root
    upstream_phase = {
        "calibrate": "diagnose",
        "confirm512": "calibrate",
        "full": "confirm512",
    }.get(plan.phase)
    if upstream_phase is not None:
        continuation = _continuation_for_runtime(campaign_runtime)
        if plan.phase == "calibrate" and continuation is not None:
            upstream_path = REPO_ROOT / str(
                continuation["parent"]["diagnose_gate"]["path"]
            )
        else:
            upstream_path = campaign_root / upstream_phase / "gate_contract.json"
        upstream_gate = _load_gate(upstream_path, upstream_phase)
        if plan.phase != "calibrate" and continuation is not None:
            _require_gate_continuation(upstream_gate, campaign_runtime)
    if plan.phase == "full":
        assert upstream_gate is not None
        selection = validate_selection_contract(
            _read_json_mapping(campaign_root / "selection.json", "selection"),
            upstream_gate,
        )
        heldout_seal = _read_json_mapping(
            campaign_root / "heldout_seal.json", "heldout seal"
        )
        visual_manifest = _mapping(manifests.get("full_visual_64"), "full_visual_64")
        visual_manifest_path = REPO_ROOT / str(visual_manifest["path"])
        visual_manifest_sha256 = str(visual_manifest["sha256"])
    return PhaseResultsRequest(
        repo_root=REPO_ROOT,
        phase_root=campaign_root / plan.phase,
        phase=plan.phase,
        campaign_id=campaign_id,
        campaign_runtime_sha256=str(campaign_runtime["campaign_runtime_sha256"]),
        manifest_contracts_sha256=str(manifest_contract["manifest_contracts_sha256"]),
        manifest_path=REPO_ROOT / str(manifest["path"]),
        manifest_sha256=str(manifest["sha256"]),
        source_index_path=REPO_ROOT / str(source_index["path"]),
        source_index_sha256=str(source_index["sha256"]),
        checkpoint_sha256=str(
            _mapping(campaign_runtime.get("checkpoint"), "checkpoint")["sha256"]
        ),
        bootstrap_seed=int(
            _mapping(campaign_runtime.get("bootstrap"), "bootstrap")["seed"]
        ),
        runs=tuple(run_specs),
        expected_candidate_arm_ids=tuple(
            dict.fromkeys(run.arm_ref for run in plan.runs if run.arm_ref != "native")
        ),
        expected_seeds=tuple(dict.fromkeys(run.seed for run in plan.runs)),
        upstream_gate=upstream_gate,
        visual_manifest_path=visual_manifest_path,
        visual_manifest_sha256=visual_manifest_sha256,
        confirm_seed=(
            int(phase_config["seed"]) if plan.phase == "confirm512" else None
        ),
        selection=selection,
        heldout_seal=heldout_seal,
    )


def _print_phase_closure(phase: str, closure: PhaseClosureOutcome) -> None:
    print(
        json.dumps(
            {
                "phase": phase,
                "status": closure.status,
                "awaiting_path": (
                    None
                    if closure.awaiting_path is None
                    else str(closure.awaiting_path)
                ),
                "required_review_count": closure.required_review_count,
                "completed_review_count": closure.completed_review_count,
                "exit_code": AWAITING_VISUAL_REVIEW_EXIT_CODE,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


class R9ProductionEvaluatorCallbacks:
    """Run every production evaluator in a globally admitted GPU subprocess."""

    def __init__(
        self,
        *,
        runtime: Mapping[str, Any],
        campaign_runtime: Mapping[str, Any],
        scheduler: R9ResourceScheduler,
        gpu_bindings: Mapping[int, str],
        peer_status_store: R9PeerStatusStore,
        process_factory: Any = subprocess.Popen,
        sleep: Any = time.sleep,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self._python = str(runtime["python"])
        evaluation = _mapping(campaign_runtime.get("evaluation"), "evaluation")
        worker = _mapping(evaluation.get("worker"), "evaluation worker")
        self._worker_script = _repo_path(
            REPO_ROOT, worker.get("path"), "evaluation worker"
        )
        self._worker_implementation = _repo_path(
            REPO_ROOT,
            worker.get("implementation_path"),
            "evaluation worker implementation",
        )
        self._worker_contract = {
            "path": str(self._worker_script.resolve()),
            "sha256": str(worker.get("sha256")),
            "implementation_path": str(self._worker_implementation.resolve()),
            "implementation_sha256": str(worker.get("implementation_sha256")),
        }
        self._validate_current_worker_contract()
        self._evaluation = dict(evaluation)
        quality = _mapping(evaluation.get("quality"), "quality evaluation")
        quality_script = _mapping(quality.get("script"), "quality script")
        self._quality_script_sha256 = _require_sha256(
            quality_script.get("sha256"), "quality script SHA256"
        )
        self._arcface_contract_sha256 = _canonical_json_sha256(
            _mapping(evaluation.get("arcface"), "ArcFace evaluation")
        )
        resource_smokes = _mapping(
            evaluation.get("resource_smokes"), "evaluator resource smokes"
        )
        self._evaluator_ram_slot_budgets = {
            kind: _positive_int(
                _mapping(resource_smokes.get(kind), f"{kind} resource smoke").get(
                    "ram_slot_budget_bytes"
                ),
                f"{kind} evaluator RAM slot budget",
            )
            for kind in ("arcface", "quality")
        }
        heldout_resource = _mapping(
            resource_smokes.get("heldout"), "heldout resource contract"
        )
        if heldout_resource != {
            "mode": "exclusive_single_official_run",
            "smoke_execution": "sealed_until_winner_lock",
            "global_exclusive_slots": 16,
            "ram_admission_percent": 85,
            "ram_hard_limit_percent": 90,
        }:
            raise ValueError("heldout evaluator resource contract mismatch")
        self._campaign_root = REPO_ROOT / str(campaign_runtime["campaign_root"])
        self._scheduler = scheduler
        self._gpu_bindings = _validate_gpu_bindings(gpu_bindings)
        self._peer_status_store = peer_status_store
        self._process_factory = process_factory
        self._sleep = sleep
        self._poll_interval_seconds = float(poll_interval_seconds)
        if self._poll_interval_seconds <= 0:
            raise ValueError("evaluator poll interval must be positive")
        self._launch_counter = 0

    def quality(self, request: QualityEvaluationRequest) -> Mapping[str, Any]:
        payload = {
            "phase": request.phase,
            "logical_run_id": request.logical_run_id,
            "arm_id": request.arm_id,
            "seed": request.seed,
            "image_role": request.image_role,
            "manifest_path": str(request.manifest_path.resolve()),
            "source_index_path": str(request.source_index_path.resolve()),
            "source_index_sha256": request.source_index_sha256,
            "samples": _serialize_evaluator_samples(request.samples),
            "algorithm_config_sha256": request.algorithm_config_sha256,
            "runner_arm_config_sha256": request.runner_arm_config_sha256,
            "semantic_output_sha256": request.semantic_output_sha256,
            "evidence_binding_sha256": request.evidence_binding_sha256,
            "generation_result_set_sha256": request.generation_result_set_sha256,
            "per_sample_set_sha256": request.per_sample_set_sha256,
        }
        return self._run(
            "quality",
            request.phase,
            f"{request.logical_run_id}__{request.image_role}",
            payload,
        )

    def arcface(self, request: ArcFaceEvaluationRequest) -> Sequence[Mapping[str, Any]]:
        payload = {
            "phase": request.phase,
            "logical_run_id": request.logical_run_id,
            "arm_id": request.arm_id,
            "seed": request.seed,
            "source_index_path": str(request.source_index_path.resolve()),
            "source_index_sha256": request.source_index_sha256,
            "samples": _serialize_evaluator_samples(request.samples),
        }
        result = self._run("arcface", request.phase, request.logical_run_id, payload)
        if not isinstance(result, list):
            raise ValueError("ArcFace evaluator result must be a list")
        return result

    def heldout(self, request: HeldoutEvaluationRequest) -> Mapping[str, Any]:
        payload = {
            "phase": request.phase,
            "arm_id": request.arm_id,
            "seed": request.seed,
            "source_index_path": str(request.source_index_path.resolve()),
            "source_index_sha256": request.source_index_sha256,
            "samples": _serialize_evaluator_samples(request.samples),
            "selection": dict(request.selection),
            "heldout_seal": dict(request.heldout_seal),
        }
        return self._run("heldout", request.phase, request.arm_id, payload)

    def _run(
        self, evaluator: str, phase: str, unit_id: str, payload: Mapping[str, Any]
    ) -> Any:
        self._validate_current_worker_contract()
        if evaluator not in {"quality", "arcface", "heldout"}:
            raise ValueError("unknown R9 evaluator")
        if evaluator == "heldout":
            raise RuntimeError(
                "heldout evaluator remains sealed until the winner-locked exclusive runner"
            )
        if CAMPAIGN_ID_PATTERN.fullmatch(unit_id.replace("_", "-")) is None:
            if re.fullmatch(r"[A-Za-z0-9_.-]+", unit_id) is None:
                raise ValueError("evaluator unit ID is not filesystem-safe")
        root = self._campaign_root / phase / "evaluator_runs" / evaluator / unit_id
        request_path = root / "request.json"
        output_path = root / "result.json"
        attempt_path = root / "attempt.json"
        log_path = root / "worker.log"
        contract = {
            "schema_version": 1,
            "contract_type": "safa_r9_phase_evaluator_request_v1",
            "task": evaluator,
            "config": {
                "repo_root": str(REPO_ROOT.resolve()),
                "device": "cuda:0",
                "work_root": str((root / "work").resolve()),
                "batch_size": int(
                    _mapping(self._evaluation.get("heldout"), "heldout evaluation")[
                        "batch_size"
                    ]
                ),
                "arcface": dict(
                    _mapping(self._evaluation.get("arcface"), "ArcFace evaluation")
                ),
                "quality_script": dict(
                    _mapping(
                        _mapping(
                            self._evaluation.get("quality"), "quality evaluation"
                        ).get("script"),
                        "quality script",
                    )
                ),
                "worker_contract": dict(self._worker_contract),
            },
            "payload": dict(payload),
        }
        contract["evaluator_request_sha256"] = _canonical_json_sha256(contract)
        _write_immutable_bytes(
            request_path,
            (
                json.dumps(
                    contract, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            ).encode("utf-8"),
        )
        if output_path.is_file():
            return _load_evaluator_result(
                output_path,
                evaluator=evaluator,
                request_sha256=contract["evaluator_request_sha256"],
                worker_contract=self._worker_contract,
                arcface_contract_sha256=self._arcface_contract_sha256,
                quality_script_sha256=self._quality_script_sha256,
            )
        attempt = {
            "schema_version": 1,
            "contract_type": "safa_r9_evaluator_attempt_v1",
            "evaluator": evaluator,
            "evaluator_request_sha256": contract["evaluator_request_sha256"],
        }
        attempt["attempt_sha256"] = _canonical_json_sha256(attempt)
        if attempt_path.exists():
            raise RuntimeError(
                "evaluator attempt exists without a completed result; automatic retry is forbidden"
            )
        self._launch_counter += 1
        worker_id = f"evaluator:{evaluator}:{phase}:{unit_id}"
        lease = None
        while lease is None:
            lease = _admit_worker(
                self._scheduler,
                worker_id=worker_id,
                launch_ordinal=50_000 + self._launch_counter,
                gpu_bindings=self._gpu_bindings,
                ram_slot_budget_bytes=self._evaluator_ram_slot_budgets[evaluator],
            )
            if lease is None:
                self._scheduler.enforce_actual_ram_limit()
                self._sleep(self._poll_interval_seconds)
        self._peer_status_store.record_admitted(worker_id)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = lease.gpu_uuid
        environment["SAFA_R9_WORKER_ID"] = worker_id
        environment["SAFA_R9_GPU_UUID"] = lease.gpu_uuid
        environment["SAFA_R9_GPU_SLOT"] = str(lease.slot_index)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process = None
        try:
            try:
                _write_exclusive_bytes(
                    attempt_path,
                    (
                        json.dumps(
                            attempt,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            except FileExistsError as error:
                raise RuntimeError(
                    "evaluator attempt was claimed by another controller"
                ) from error
            with log_path.open("xb") as log:
                process = self._process_factory(
                    (
                        self._python,
                        str(self._worker_script),
                        "--request",
                        str(request_path),
                        "--output",
                        str(output_path),
                    ),
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                self._peer_status_store.record_running(
                    worker_id, pid=_positive_int(process.pid, "evaluator PID")
                )
                while process.poll() is None:
                    try:
                        self._scheduler.enforce_actual_ram_limit()
                    except CampaignFailedError:
                        _terminate_process(process)
                        self._peer_status_store.record_terminal(
                            worker_id, state="terminated"
                        )
                        raise
                    self._sleep(self._poll_interval_seconds)
                if process.returncode != 0:
                    self._peer_status_store.record_terminal(worker_id, state="failed")
                    self._scheduler.fail_worker(
                        worker_id, kind=FailureKind.PEER_FAILURE
                    )
            result = _load_evaluator_result(
                output_path,
                evaluator=evaluator,
                request_sha256=contract["evaluator_request_sha256"],
                worker_contract=self._worker_contract,
                arcface_contract_sha256=self._arcface_contract_sha256,
                quality_script_sha256=self._quality_script_sha256,
            )
        except BaseException:
            if process is not None and process.poll() is None:
                _terminate_process(process)
            if worker_id in {
                active.worker_id for active in self._scheduler.active_leases
            }:
                try:
                    self._peer_status_store.record_terminal(worker_id, state="failed")
                    self._scheduler.fail_worker(
                        worker_id, kind=FailureKind.CONTRACT_MISMATCH
                    )
                except CampaignFailedError:
                    pass
            _cleanup_evaluator_work_root(root / "work", evaluator_root=root)
            raise
        _cleanup_evaluator_work_root(root / "work", evaluator_root=root)
        self._peer_status_store.record_terminal(worker_id, state="succeeded")
        self._scheduler.release_worker(worker_id)
        return result

    def _validate_current_worker_contract(self) -> None:
        for path_field, digest_field, label in (
            ("path", "sha256", "evaluation worker wrapper"),
            (
                "implementation_path",
                "implementation_sha256",
                "evaluation worker implementation",
            ),
        ):
            if (
                _sha256_path(Path(self._worker_contract[path_field]))
                != self._worker_contract[digest_field]
            ):
                raise ValueError(f"{label} SHA256 mismatch")


def _serialize_evaluator_samples(samples: Sequence[Any]) -> list[dict[str, str]]:
    return [
        {
            "sample_id": str(sample.sample_id),
            "source": str(Path(sample.source).resolve()),
            "native": str(Path(sample.native).resolve()),
            "candidate": str(Path(sample.candidate).resolve()),
            "source_sha256": str(sample.source_sha256),
            "native_sha256": str(sample.native_sha256),
            "candidate_sha256": str(sample.candidate_sha256),
        }
        for sample in samples
    ]


def _load_evaluator_result(
    path: Path,
    *,
    evaluator: str,
    request_sha256: str,
    worker_contract: Mapping[str, Any],
    arcface_contract_sha256: str,
    quality_script_sha256: str,
) -> Any:
    payload = _read_json_mapping(path, "evaluator result")
    expected = {
        "schema_version",
        "contract_type",
        "task",
        "evaluator_request_sha256",
        "worker_contract",
        "arcface_contract_sha256",
        "quality_script_sha256",
        "result",
        "evaluator_output_sha256",
    }
    if set(payload) != expected:
        raise ValueError("evaluator result fields are not canonical")
    if (
        payload["schema_version"] != 1
        or payload["contract_type"] != "safa_r9_phase_evaluator_output_v1"
        or payload["task"] != evaluator
        or payload["evaluator_request_sha256"] != request_sha256
        or payload["worker_contract"] != dict(worker_contract)
        or payload["arcface_contract_sha256"]
        != _require_sha256(arcface_contract_sha256, "ArcFace contract SHA256")
        or payload["quality_script_sha256"]
        != _require_sha256(quality_script_sha256, "quality script SHA256")
    ):
        raise ValueError("evaluator result request binding mismatch")
    declared = _require_sha256(
        payload["evaluator_output_sha256"], "evaluator output SHA256"
    )
    canonical = dict(payload)
    canonical.pop("evaluator_output_sha256")
    if _canonical_json_sha256(canonical) != declared:
        raise ValueError("evaluator result digest mismatch")
    return payload["result"]


def _cleanup_evaluator_work_root(work_root: Path, *, evaluator_root: Path) -> None:
    resolved_root = evaluator_root.resolve()
    resolved_work = work_root.resolve()
    if resolved_work.parent != resolved_root or resolved_work.name != "work":
        raise ValueError("evaluator work cleanup path escaped its unit root")
    if resolved_work.exists():
        shutil.rmtree(resolved_work)


def resolve_phase_promotion(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    *,
    phase: str,
    campaign_id: str,
) -> tuple[list[str] | None, str | None]:
    del runtime
    root = REPO_ROOT / str(campaign_runtime["campaign_root"])
    if phase in {"preflight", "diagnose"}:
        if campaign_runtime.get("continuation") is not None:
            raise ValueError("continuation child rejects preflight and diagnose")
        return None, None
    if phase == "calibrate":
        continuation = _continuation_for_runtime(campaign_runtime)
        if continuation is not None:
            selected = [str(row["arm_id"]) for row in continuation["selected_arms"]]
        else:
            gate = _load_gate(root / "diagnose" / "gate_contract.json", "diagnose")
            selected = list(gate["selected_arm_ids"])
        if not 1 <= len(selected) <= 3:
            raise RuntimeError("calibrate requires 1..3 A-stage promotions")
        return selected, None
    if phase == "confirm512":
        gate = _load_gate(root / "calibrate" / "gate_contract.json", "calibrate")
        _require_gate_continuation(gate, campaign_runtime)
        selected = list(gate["selected_arm_ids"])
        if not 1 <= len(selected) <= 2:
            raise RuntimeError("confirm512 requires 1..2 B-stage promotions")
        return selected, None
    confirm_gate = _load_gate(root / "confirm512" / "gate_contract.json", "confirm512")
    _require_gate_continuation(confirm_gate, campaign_runtime)
    selection_path = root / "selection.json"
    selection = validate_selection_contract(
        _read_json_mapping(selection_path, "selection"),
        confirm_gate,
    )
    if selection["campaign_id"] != campaign_id:
        raise ValueError("selection campaign ID mismatch")
    _require_selection_continuation(selection, campaign_runtime)
    return None, str(selection["winner"]["arm_id"])


def finalize_phase_gate(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    diagnose_contract: Mapping[str, Any],
    *,
    phase: str,
    campaign_id: str,
) -> dict[str, Any]:
    root = REPO_ROOT / str(campaign_runtime["campaign_root"])
    results = _load_phase_results(root / phase / "phase_results.json", phase)
    _validate_phase_evidence_chain(root, phase, results)
    manifests = _mapping(manifest_contract.get("manifests"), "manifests")
    manifest_key = {
        "diagnose": "diagnose_18",
        "calibrate": "calibration_64",
        "confirm512": "validate_512",
        "full": "full_2048",
    }[phase]
    manifest = (
        diagnose_contract
        if phase == "diagnose"
        else _mapping(manifests.get(manifest_key), manifest_key)
    )
    context = {
        "campaign_id": campaign_id,
        "campaign_runtime_sha256": campaign_runtime["campaign_runtime_sha256"],
        "manifest_contracts_sha256": manifest_contract["manifest_contracts_sha256"],
        "manifest_sha256": manifest["sha256"],
        "checkpoint_sha256": _mapping(campaign_runtime.get("checkpoint"), "checkpoint")[
            "sha256"
        ],
        "phase_results_sha256": results["phase_results_sha256"],
        "automatic_evidence_sha256": results["automatic_evidence_sha256"],
        "run_plan_sha256": results["run_plan_sha256"],
        "evaluator_evidence_sha256": _phase_evaluator_evidence_sha256(results, phase),
    }
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is not None:
        context["continuation_contract_sha256"] = continuation[
            "continuation_contract_sha256"
        ]
    for field in (
        "campaign_runtime_sha256",
        "manifest_contracts_sha256",
        "manifest_sha256",
    ):
        if results.get(field) != context[field]:
            raise ValueError(f"phase results {field} mismatch")
    if phase == "diagnose":
        gate = build_a_gate_contract(
            context,
            results["arms"],
            diagnose_manifest=diagnose_contract,
        )
    elif phase == "calibrate":
        gate = build_b_gate_contract(
            context,
            results["arms"],
            bootstrap_seed=int(
                _mapping(campaign_runtime.get("bootstrap"), "bootstrap")["seed"]
            ),
        )
    elif phase == "confirm512":
        gate = build_c_gate_contract(
            context,
            results["arms"],
            confirm_seed=int(
                _mapping(
                    _mapping(campaign_runtime.get("phases"), "phases")["confirm512"],
                    "confirm512",
                )["seed"]
            ),
            bootstrap_seed=int(
                _mapping(campaign_runtime.get("bootstrap"), "bootstrap")["seed"]
            ),
        )
    else:
        confirm_gate = _load_gate(
            root / "confirm512" / "gate_contract.json", "confirm512"
        )
        selection = validate_selection_contract(
            _read_json_mapping(root / "selection.json", "selection"),
            confirm_gate,
        )
        _require_selection_continuation(selection, campaign_runtime)
        heldout = _read_json_mapping(root / "heldout_seal.json", "heldout seal")
        gate = build_d_gate_contract(
            context,
            selection=selection,
            heldout_seal=heldout,
            result=results["result"],
            bootstrap_seed=int(
                _mapping(campaign_runtime.get("bootstrap"), "bootstrap")["seed"]
            ),
        )
    gate_path = root / phase / "gate_contract.json"
    _require_gate_continuation(gate, campaign_runtime)
    write_immutable_contract(
        gate_path,
        gate,
        digest_field="gate_contract_sha256",
    )
    if phase == "confirm512" and gate["verdict"] == "winner_locked":
        manifest_sha256s = {
            name: str(entry["sha256"]) for name, entry in manifests.items()
        }
        selection = build_selection_contract(
            gate,
            manifest_sha256s=manifest_sha256s,
        )
        _require_selection_continuation(selection, campaign_runtime)
        write_immutable_contract(
            root / "selection.json",
            selection,
            digest_field="selection_sha256",
        )
        heldout_assets = _mapping(runtime.get("heldout_assets"), "heldout assets")
        assets = {
            name: _bound_file(
                REPO_ROOT,
                _mapping(heldout_assets.get(name), name)["path"],
                _mapping(heldout_assets.get(name), name)["sha256"],
                f"heldout {name}",
            )
            for name in ("e1", "e2", "facenet", "adaface")
        }
        seal = build_heldout_seal_contract(selection, assets)
        write_immutable_contract(
            root / "heldout_seal.json",
            seal,
            digest_field="heldout_seal_sha256",
        )
    return gate


def _load_gate(path: Path, expected_phase: str) -> dict[str, Any]:
    gate = validate_gate_contract(_read_json_mapping(path, f"{expected_phase} gate"))
    if gate["phase"] != expected_phase:
        raise ValueError(f"expected {expected_phase} gate, got {gate['phase']}")
    return gate


def _require_gate_continuation(
    gate: Mapping[str, Any], campaign_runtime: Mapping[str, Any]
) -> None:
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is None:
        return
    context = _mapping(gate.get("context"), "gate context")
    if context.get("continuation_contract_sha256") != continuation[
        "continuation_contract_sha256"
    ]:
        raise ValueError("child gate continuation SHA256 mismatch")


def _require_selection_continuation(
    selection: Mapping[str, Any], campaign_runtime: Mapping[str, Any]
) -> None:
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is None:
        return
    if selection.get("continuation_contract_sha256") != continuation[
        "continuation_contract_sha256"
    ]:
        raise ValueError("child selection continuation SHA256 mismatch")


def _load_phase_results(path: Path, phase: str) -> dict[str, Any]:
    payload = _read_json_mapping(path, f"{phase} phase results")
    value_field = "result" if phase == "full" else "arms"
    expected = {
        "schema_version",
        "contract_type",
        "phase",
        "campaign_runtime_sha256",
        "manifest_contracts_sha256",
        "manifest_sha256",
        "automatic_evidence_sha256",
        "run_plan_sha256",
        value_field,
        "phase_results_sha256",
    }
    if set(payload) != expected:
        raise ValueError(f"{phase} phase results fields are not canonical")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_type") != "safa_r9_phase_results_v1"
        or payload.get("phase") != phase
    ):
        raise ValueError(f"{phase} phase results contract type mismatch")
    declared = _require_sha256(
        payload.get("phase_results_sha256"), "phase results SHA256"
    )
    canonical = dict(payload)
    canonical.pop("phase_results_sha256")
    if _canonical_json_sha256(canonical) != declared:
        raise ValueError(f"{phase} phase results digest mismatch")
    if phase != "full" and not isinstance(payload["arms"], list):
        raise ValueError(f"{phase} phase results arms must be a list")
    if phase == "full" and not isinstance(payload["result"], Mapping):
        raise ValueError("full phase result must be a mapping")
    return payload


def _phase_evaluator_evidence_sha256(results: Mapping[str, Any], phase: str) -> str:
    rows = [results["result"]] if phase == "full" else list(results["arms"])
    evidence = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("phase evaluator evidence row must be a mapping")
        evidence.append(
            {
                "arm_id": str(
                    row["winner_arm_id"] if phase == "full" else row["arm_id"]
                ),
                "evaluator_evidence_sha256": _require_sha256(
                    row.get("evaluator_evidence_sha256"),
                    "evaluator evidence SHA256",
                ),
            }
        )
    return _canonical_json_sha256(evidence)


def _validate_phase_evidence_chain(
    campaign_root: Path, phase: str, results: Mapping[str, Any]
) -> dict[str, Any]:
    automatic = _read_json_mapping(
        campaign_root / phase / "automatic_evidence.json", "automatic evidence"
    )
    if (
        automatic.get("schema_version") != 1
        or automatic.get("contract_type") != "safa_r9_automatic_phase_evidence_v1"
        or automatic.get("phase") != phase
    ):
        raise ValueError("automatic evidence contract type mismatch")
    declared = _require_sha256(
        automatic.get("automatic_evidence_sha256"),
        "automatic evidence SHA256",
    )
    canonical = dict(automatic)
    canonical.pop("automatic_evidence_sha256")
    if _canonical_json_sha256(canonical) != declared:
        raise ValueError("automatic evidence digest mismatch")
    if declared != results.get("automatic_evidence_sha256"):
        raise ValueError("phase results do not bind current automatic evidence")
    if automatic.get("run_plan_sha256") != results.get("run_plan_sha256"):
        raise ValueError("phase results do not bind current run plan")
    return automatic


def run_resource_smoke(
    runtime: Mapping[str, Any],
    campaign_claim: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    *,
    probe: Any | None = None,
    process_factory: Any = subprocess.Popen,
    rss_sampler: Any | None = None,
    sleep: Any = time.sleep,
    poll_interval_seconds: float = 0.1,
) -> dict[str, Any]:
    """Run the one-worker RSS bootstrap and exclusively lock its contract."""
    if campaign_claim.get("campaign_runtime_sha256") is not None:
        raise ValueError("resource smoke is only valid before runtime finalization")
    resource_probe = SystemResourceProbe() if probe is None else probe
    ram = resource_probe.ram_snapshot()
    if ram.used_bytes * 100 >= ram.total_bytes * 85:
        raise ResourceContractError("resource smoke cannot start at or above 85% RAM")
    gpu_snapshots = resource_probe.gpu_snapshots()
    gpu_bindings = _validate_gpu_bindings(
        {
            snapshot.index: snapshot.uuid
            for snapshot in gpu_snapshots
            if snapshot.index in {0, 1, 2, 3}
        }
    )
    snapshots = {snapshot.index: snapshot for snapshot in gpu_snapshots}
    eligible = [
        index
        for index in sorted(gpu_bindings)
        if gpu_slot_capacity(snapshots[index]) >= 1
    ]
    if not eligible:
        raise ResourceContractError("resource smoke has no GPU with one exact R9 slot")
    run, declaration = materialize_resource_smoke_runtime(
        runtime,
        campaign_claim,
        manifest_contract,
    )
    lock_path = REPO_ROOT / str(runtime["campaign_root"]) / "resource_smoke.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourceContractError("resource smoke lock is contended") from error
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu_bindings[eligible[0]]
        process = process_factory(run.command, cwd=REPO_ROOT, env=environment)
        sampler = _process_tree_rss_bytes if rss_sampler is None else rss_sampler
        peak_rss_bytes = 0
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            rss_bytes, reaped_returncode = _sample_or_reap_process_tree(
                process, sampler
            )
            if reaped_returncode is not None:
                returncode = reaped_returncode
                break
            if rss_bytes is None:
                raise AssertionError("running process RSS sample is missing")
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            current_ram = resource_probe.ram_snapshot()
            if current_ram.used_bytes * 100 >= current_ram.total_bytes * 90:
                _terminate_process(process)
                raise ResourceContractError(
                    "resource smoke crossed the R9 90% RAM hard limit"
                )
            sleep(poll_interval_seconds)
        if returncode != 0:
            raise ResourceContractError(
                f"resource smoke worker failed once with exit code {returncode}"
            )
        if peak_rss_bytes <= 0:
            raise ResourceContractError(
                "resource smoke measured no positive process RSS"
            )
        validate_worker_completion(run)
        manifests = _mapping(manifest_contract.get("manifests"), "manifests")
        checkpoint = _mapping(runtime.get("checkpoint"), "checkpoint")
        contract = build_resource_smoke_contract(
            run_id=str(declaration["run_id"]),
            arm_id=str(declaration["arm_id"]),
            manifest=str(declaration["manifest"]),
            manifest_sha256=str(manifests[str(declaration["manifest"])]["sha256"]),
            checkpoint_sha256=str(checkpoint["sha256"]),
            peak_rss_bytes=peak_rss_bytes,
        )
        destination = REPO_ROOT / str(declaration["output_path"])
        _write_exclusive_bytes(
            destination,
            (
                json.dumps(
                    contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        return contract
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _sample_or_reap_process_tree(
    process: Any, sampler: Any
) -> tuple[int | None, int | None]:
    """Sample a running tree, or synchronously reap an observed root exit."""
    try:
        return int(sampler(process.pid)), None
    except _ProcessTreeRootExitObserved as observation:
        try:
            returncode = process.wait(timeout=RESOURCE_SMOKE_ROOT_EXIT_WAIT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise ResourceContractError(
                f"resource smoke root process {observation.pid} was observed "
                f"{observation.reason} but did not exit within "
                f"{RESOURCE_SMOKE_ROOT_EXIT_WAIT_SECONDS:.1f} seconds"
            ) from error
        return None, int(returncode)


def materialize_resource_smoke_runtime(
    runtime: Mapping[str, Any],
    campaign_claim: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
) -> tuple[RunSpec, Mapping[str, Any]]:
    resources = _mapping(runtime.get("resources"), "resources")
    declaration = _mapping(resources.get("resource_smoke"), "resource_smoke")
    manifests = _mapping(manifest_contract.get("manifests"), "manifests")
    manifest = _mapping(manifests.get(str(declaration["manifest"])), "smoke manifest")
    base_path = _repo_path(REPO_ROOT, runtime.get("base_config"), "base config")
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("R9 base config must contain a mapping")
    output_dir = Path(str(declaration["output_path"])).parent / str(
        declaration["run_id"]
    )
    campaign_root = Path(str(campaign_claim["campaign_root"]))
    runtime_path = campaign_root / "runtime_configs" / "resource_smoke.yaml"
    seed = _positive_int(
        _mapping(_mapping(runtime.get("phases"), "phases")["preflight"], "preflight")[
            "seed"
        ],
        "resource smoke seed",
    )
    config.update(
        {
            "experiment_name": str(declaration["run_id"]),
            "experiment_contract": str(runtime["experiment_contract"]),
            "out_dir": str(output_dir),
            "mode": "native",
            "phase": "resource_smoke",
            "seed": seed,
            "sampling_seed": seed,
            "device": "cuda:0",
            "max_samples": int(manifest["sample_count"]),
            "sample_id_manifest": str(manifest["path"]),
            "sample_id_manifest_sha256": str(manifest["sha256"]),
            "calibration_sample_id_manifest": str(manifest["path"]),
            "calibration_sample_id_manifest_sha256": str(manifest["sha256"]),
            "asset_digest_cache": str(
                output_dir.parent / f"{declaration['run_id']}.assets.json"
            ),
            "contact_sheets": False,
            "r9_campaign_id": str(campaign_claim["campaign_id"]),
            "r9_campaign_claim_sha256": str(campaign_claim["campaign_claim_sha256"]),
            "r9_manifest_contracts_sha256": str(
                manifest_contract["manifest_contracts_sha256"]
            ),
            "r9_phase_manifest_sha256": str(manifest["sha256"]),
        }
    )
    for field in (
        "active_guidance_intervals",
        "collect_interval_diagnostics",
        "r9_guidance_interval_contract",
    ):
        config.pop(field, None)
    config = resolve_frozen_effective_guidance_config(config)
    _write_immutable_bytes(
        REPO_ROOT / runtime_path,
        yaml.safe_dump(config, sort_keys=False).encode("utf-8"),
    )
    run = RunSpec(
        phase="resource_smoke",
        logical_run_id=str(declaration["run_id"]),
        arm_ref=str(declaration["arm_id"]),
        seed=seed,
        repeat_index=None,
        shard_index=0,
        num_shards=1,
        sample_count=int(manifest["sample_count"]),
        manifest_key=str(declaration["manifest"]),
        runtime_config=runtime_path,
        output_dir=output_dir,
        command=(
            str(runtime["python"]),
            str(runtime["generation_script"]),
            "--config",
            str(runtime_path),
            "--output-dir",
            str(output_dir),
            "--shard-index",
            "0",
            "--num-shards",
            "1",
        ),
    )
    return run, declaration


def execute_campaign(
    plans: Sequence[PhasePlan],
    *,
    scheduler: R9ResourceScheduler,
    gpu_bindings: Mapping[int, str],
    peer_status_store: R9PeerStatusStore,
    process_factory: Any = subprocess.Popen,
    poll_interval_seconds: float = 1.0,
    sleep: Any = time.sleep,
) -> int:
    """Refill all admitted GPU slots and fail the campaign on any peer error."""
    bindings = _validate_gpu_bindings(gpu_bindings)
    pending: list[tuple[RunSpec, int]] = []
    for plan in plans:
        for run_index, run in enumerate(plan.runs):
            runtime_config = REPO_ROOT / run.runtime_config
            if not runtime_config.is_file():
                raise FileNotFoundError(
                    f"immutable R9 runtime config is missing: {runtime_config}"
                )
            completion = REPO_ROOT / run.output_dir / "completion.json"
            if completion.is_file():
                validate_worker_completion(run)
                continue
            pending.append((run, _stable_launch_ordinal(run.phase, run_index)))
    active: dict[str, ActiveWorker] = {}
    next_gpu_index = min(bindings)
    while pending or active:
        launched = False
        pending_index = 0
        while pending_index < len(pending):
            run, launch_ordinal = pending[pending_index]
            worker_id = f"{run.phase}:{run.logical_run_id}:shard-{run.shard_index}"
            lease = _admit_worker(
                scheduler,
                worker_id=worker_id,
                launch_ordinal=launch_ordinal,
                gpu_bindings=bindings,
                ram_slot_budget_bytes=scheduler.ram_slot_budget_bytes,
                start_gpu_index=next_gpu_index,
            )
            if lease is None:
                pending_index += 1
                continue
            next_gpu_index = _next_gpu_index(bindings, lease.gpu_uuid)
            try:
                peer_status_store.record_admitted(worker_id)
            except (OSError, ResourceContractError, ValueError):
                scheduler.fail_worker(worker_id, kind=FailureKind.CONTRACT_MISMATCH)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = lease.gpu_uuid
            environment["SAFA_R9_WORKER_ID"] = worker_id
            environment["SAFA_R9_GPU_UUID"] = lease.gpu_uuid
            environment["SAFA_R9_GPU_SLOT"] = str(lease.slot_index)
            try:
                process = process_factory(
                    run.command,
                    cwd=REPO_ROOT,
                    env=environment,
                )
            except BaseException:
                try:
                    peer_status_store.record_terminal(worker_id, state="failed")
                    scheduler.fail_worker(worker_id, kind=FailureKind.PEER_FAILURE)
                except CampaignFailedError:
                    _cleanup_active_workers(active, scheduler, peer_status_store)
                    raise
            try:
                process_pid = _positive_int(process.pid, "worker process PID")
                peer_status_store.record_running(worker_id, pid=process_pid)
            except (AttributeError, OSError, ResourceContractError, ValueError):
                if process.poll() is None:
                    _terminate_process(process)
                try:
                    peer_status_store.record_terminal(worker_id, state="failed")
                    scheduler.fail_worker(worker_id, kind=FailureKind.CONTRACT_MISMATCH)
                except CampaignFailedError:
                    _cleanup_active_workers(active, scheduler, peer_status_store)
                    raise
            active[worker_id] = ActiveWorker(
                run=run,
                worker_id=worker_id,
                process=process,
                launch_ordinal=launch_ordinal,
            )
            pending.pop(pending_index)
            launched = True
        if not active and pending:
            sleep(poll_interval_seconds)
            continue
        try:
            scheduler.enforce_actual_ram_limit()
        except CampaignFailedError:
            _cleanup_active_workers(active, scheduler, peer_status_store)
            raise
        completed_any = False
        for worker_id, worker in list(active.items()):
            returncode = worker.process.poll()
            if returncode is None:
                continue
            completed_any = True
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                try:
                    peer_status_store.record_terminal(worker_id, state="failed")
                    scheduler.fail_worker(
                        worker_id,
                        kind=FailureKind.CONTRACT_MISMATCH,
                    )
                except CampaignFailedError:
                    del active[worker_id]
                    _cleanup_active_workers(active, scheduler, peer_status_store)
                    raise
            if returncode != 0:
                try:
                    peer_status_store.record_terminal(worker_id, state="failed")
                    scheduler.fail_worker(
                        worker_id,
                        kind=FailureKind.PEER_FAILURE,
                    )
                except CampaignFailedError:
                    del active[worker_id]
                    _cleanup_active_workers(active, scheduler, peer_status_store)
                    raise
            try:
                validate_worker_completion(worker.run)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                try:
                    peer_status_store.record_terminal(worker_id, state="failed")
                    scheduler.fail_worker(
                        worker_id,
                        kind=FailureKind.CONTRACT_MISMATCH,
                    )
                except CampaignFailedError:
                    del active[worker_id]
                    _cleanup_active_workers(active, scheduler, peer_status_store)
                    raise
            peer_status_store.record_terminal(worker_id, state="succeeded")
            scheduler.release_worker(worker_id)
            del active[worker_id]
        if active and not launched and not completed_any:
            sleep(poll_interval_seconds)
    return 0


def _stable_launch_ordinal(phase: str, run_index: int) -> int:
    bases = {
        "resource_smoke": 0,
        "preflight": 1_000,
        "diagnose": 10_000,
        "calibrate": 20_000,
        "confirm512": 30_000,
        "full": 40_000,
    }
    if phase not in bases or run_index < 0 or run_index >= 1_000:
        raise ValueError("R9 stable launch ordinal input is invalid")
    return bases[phase] + run_index


def _cleanup_active_workers(
    active: Mapping[str, ActiveWorker],
    scheduler: R9ResourceScheduler,
    peer_status_store: R9PeerStatusStore,
) -> None:
    for worker in sorted(
        active.values(), key=lambda value: value.launch_ordinal, reverse=True
    ):
        if worker.process.poll() is None:
            _terminate_process(worker.process)
        peer_status_store.record_terminal(worker.worker_id, state="terminated")
        try:
            scheduler.release_worker(worker.worker_id)
        except ResourceContractError:
            if worker.worker_id in {
                lease.worker_id for lease in scheduler.active_leases
            }:
                raise


def _terminate_process(process: Any) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def validate_worker_completion(run: RunSpec) -> dict[str, Any]:
    """Validate a successful shard against its immutable runtime contract."""
    runtime_path = REPO_ROOT / run.runtime_config
    config = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("worker runtime config must contain a mapping")
    contract_output = Path(run.output_dir)
    if contract_output.is_absolute():
        raise ValueError("worker output directory must be repo-relative")
    resolved_root = REPO_ROOT.resolve()
    output = (resolved_root / contract_output).resolve()
    try:
        output.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("worker output directory escapes repo root") from error
    completion_path = output / "completion.json"
    result_path = output / "generation_result.json"
    run_manifest_path = output / "run_manifest.json"
    contract_result_path = contract_output / "generation_result.json"
    contract_run_manifest_path = contract_output / "run_manifest.json"
    completion = _read_json_mapping(completion_path, "worker completion")
    result = _read_json_mapping(result_path, "generation result")
    run_manifest = _read_json_mapping(run_manifest_path, "run manifest")
    if result != run_manifest:
        raise ValueError("generation result and run manifest differ")
    manifest_path = _repo_path(
        REPO_ROOT,
        config.get("sample_id_manifest"),
        "worker sample manifest",
    )
    if _sha256_path(manifest_path) != config.get("sample_id_manifest_sha256"):
        raise ValueError("worker sample manifest digest mismatch")
    all_ids = [str(row["sample_id"]) for row in _read_jsonl(manifest_path)]
    maximum = _positive_int(config.get("max_samples"), "worker max_samples")
    expected_ids = all_ids[:maximum][run.shard_index :: run.num_shards]
    expected_id_sha256 = hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in expected_ids).encode()
    ).hexdigest()
    expected_count = len(expected_ids)
    expected_arm_sha256 = _require_sha256(
        config.get("arm_config_sha256"), "worker arm config SHA256"
    )
    exact_completion = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": expected_count,
        "sample_id_sha256": expected_id_sha256,
        "arm_config_sha256": expected_arm_sha256,
        "generation_result": str(contract_result_path),
        "run_manifest": str(contract_run_manifest_path),
    }
    if completion != exact_completion:
        raise ValueError("worker completion disagrees with immutable run contract")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("sample_count") != expected_count
        or result.get("sample_id_sha256") != expected_id_sha256
        or result.get("arm_config_sha256") != expected_arm_sha256
        or result.get("shard") != {"index": run.shard_index, "count": run.num_shards}
    ):
        raise ValueError("generation result core fields disagree with run contract")
    checkpoint = _mapping(result.get("checkpoint"), "generation checkpoint")
    if checkpoint.get("sha256") != config.get("checkpoint_sha256"):
        raise ValueError("generation checkpoint digest mismatch")
    result_config = _mapping(result.get("config"), "generation config")
    campaign_digest_field = (
        "r9_campaign_claim_sha256"
        if run.phase == "resource_smoke"
        else "r9_campaign_runtime_sha256"
    )
    for field in (
        "r9_campaign_id",
        campaign_digest_field,
        "r9_manifest_contracts_sha256",
        "r9_phase_manifest_sha256",
        "arm_config_sha256",
    ):
        if result_config.get(field) != config.get(field):
            raise ValueError(f"generation config field {field} mismatch")
    closure_fields = (
        "r9_semigroup_closure_seal",
        "r9_semigroup_closure_seal_sha256",
        "r9_semigroup_closure_contract_sha256",
        "r9_semigroup_bootstrap_campaign_id",
    )
    present_closure_fields = {field for field in closure_fields if field in config}
    if present_closure_fields and present_closure_fields != set(closure_fields):
        raise ValueError("worker runtime has a partial semigroup closure binding")
    for field in (
        *closure_fields,
        "schedule_manifest",
        "schedule_contract_sha256",
        "r9_semigroup_gate_contract",
        "r9_semigroup_gate_contract_sha256",
    ):
        if field in config and result_config.get(field) != config[field]:
            raise ValueError(f"generation config field {field} mismatch")
    verified = {
        "schema_version": 1,
        "contract_type": "safa_r9_verified_worker_completion_v1",
        "worker_id": f"{run.phase}:{run.logical_run_id}:shard-{run.shard_index}",
        "runtime_config_sha256": _sha256_path(runtime_path),
        "completion_sha256": _sha256_path(completion_path),
        "generation_result_sha256": _sha256_path(result_path),
        "run_manifest_sha256": _sha256_path(run_manifest_path),
        "sample_count": expected_count,
        "sample_id_sha256": expected_id_sha256,
        "arm_config_sha256": expected_arm_sha256,
        "manifest_contracts_sha256": config["r9_manifest_contracts_sha256"],
        "phase_manifest_sha256": config["r9_phase_manifest_sha256"],
    }
    verified[campaign_digest_field.removeprefix("r9_")] = config[campaign_digest_field]
    verified["verified_completion_sha256"] = _canonical_json_sha256(verified)
    _write_immutable_bytes(
        output / "verified_completion.json",
        (
            json.dumps(
                verified,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return verified


def build_resource_scheduler(
    campaign_runtime: Mapping[str, Any],
    *,
    probe: Any | None = None,
    lock_backend: Any | None = None,
    peer_status_store: R9PeerStatusStore | None = None,
) -> tuple[R9ResourceScheduler, dict[int, str], R9PeerStatusStore]:
    """Build the exact post-smoke scheduler bound to the effective runtime."""
    resources = _mapping(campaign_runtime.get("resources"), "resources")
    smoke = _mapping(resources.get("resource_smoke"), "resource_smoke")
    result = _mapping(smoke.get("result"), "resource_smoke result")
    peak_rss_bytes = _positive_int(
        result.get("peak_rss_bytes"), "resource smoke peak RSS bytes"
    )
    resource_contract_sha256 = _canonical_json_sha256(resources)
    resource_probe = SystemResourceProbe() if probe is None else probe
    physical_gpus = resources.get("physical_gpus")
    if physical_gpus != [0, 1, 2, 3]:
        raise ValueError("R9 effective runtime must bind physical GPUs 0,1,2,3")
    snapshots = resource_probe.gpu_snapshots()
    by_index = {snapshot.index: snapshot.uuid for snapshot in snapshots}
    gpu_bindings = {index: by_index[index] for index in physical_gpus}
    status_store = (
        R9PeerStatusStore(
            (REPO_ROOT / str(campaign_runtime["campaign_root"])).parent,
            campaign_id=str(campaign_runtime["campaign_id"]),
        )
        if peer_status_store is None
        else peer_status_store
    )
    locks = (
        FcntlSlotLockBackend(Path(str(resources["global_slot_lock_root"])))
        if lock_backend is None
        else lock_backend
    )
    scheduler = R9ResourceScheduler(
        campaign_id=str(campaign_runtime["campaign_id"]),
        resource_contract_sha256=resource_contract_sha256,
        smoke_peak_rss_bytes=peak_rss_bytes,
        probe=resource_probe,
        lock_backend=locks,
        peer_status_probe=status_store,
    )
    return scheduler, gpu_bindings, status_store


def _validate_gpu_bindings(bindings: Mapping[int, str]) -> dict[int, str]:
    if set(bindings) != {0, 1, 2, 3}:
        raise ValueError("R9 scheduler requires GPU indices 0,1,2,3")
    normalized: dict[int, str] = {}
    for index in sorted(bindings):
        uuid = bindings[index]
        if not isinstance(uuid, str) or not uuid:
            raise ValueError("R9 GPU UUID bindings must be non-empty strings")
        normalized[index] = uuid
    if len(set(normalized.values())) != 4:
        raise ValueError("R9 GPU UUID bindings must be unique")
    return normalized


def _admit_worker(
    scheduler: R9ResourceScheduler,
    *,
    worker_id: str,
    launch_ordinal: int,
    gpu_bindings: Mapping[int, str],
    ram_slot_budget_bytes: int,
    start_gpu_index: int | None = None,
) -> Any | None:
    bindings = _validate_gpu_bindings(gpu_bindings)
    if start_gpu_index is None:
        start_gpu_index = min(bindings)
    if start_gpu_index not in bindings:
        raise ValueError("R9 round-robin start GPU is not bound")
    indices = tuple(sorted(bindings))
    start_offset = indices.index(start_gpu_index)
    ordered_indices = indices[start_offset:] + indices[:start_offset]
    stale_incumbents = []
    for gpu_index in ordered_indices:
        gpu_uuid = bindings[gpu_index]
        decision = scheduler.admit_worker(
            WorkerRequest(
                worker_id=worker_id,
                gpu_index=gpu_index,
                expected_gpu_uuid=gpu_uuid,
                resource_contract_sha256=scheduler.resource_contract_sha256,
                launch_ordinal=launch_ordinal,
                ram_slot_budget_bytes=ram_slot_budget_bytes,
            )
        )
        if decision.status in {
            AdmissionStatus.ADMITTED,
            AdmissionStatus.RESUMED,
            AdmissionStatus.RECLAIMED,
        }:
            if decision.lease is None:
                raise ResourceContractError(
                    "successful R9 admission omitted its slot lease"
                )
            if decision.lease.gpu_uuid != gpu_uuid:
                raise ResourceContractError(
                    "successful R9 admission returned a GPU UUID outside its request"
                )
            if (
                isinstance(decision.lease.slot_index, bool)
                or not isinstance(decision.lease.slot_index, int)
                or not 0 <= decision.lease.slot_index < 4
            ):
                raise ResourceContractError(
                    "successful R9 admission returned an invalid GPU slot"
                )
            return decision.lease
        if decision.status is AdmissionStatus.STALE_PEER:
            stale_incumbents.append(decision.incumbent)
    if stale_incumbents:
        incumbent_ids = sorted(
            {
                str(incumbent.worker_id)
                for incumbent in stale_incumbents
                if incumbent is not None
            }
        )
        raise StaleSlotLeaseError(
            "R9 admission found non-terminal stale slot lease(s): "
            + ",".join(incumbent_ids)
        )
    return None


def _next_gpu_index(bindings: Mapping[int, str], admitted_gpu_uuid: str) -> int:
    normalized = _validate_gpu_bindings(bindings)
    index_by_uuid = {uuid: index for index, uuid in normalized.items()}
    if admitted_gpu_uuid not in index_by_uuid:
        raise ResourceContractError(
            "successful R9 admission returned an unbound GPU UUID"
        )
    indices = tuple(sorted(normalized))
    admitted_offset = indices.index(index_by_uuid[admitted_gpu_uuid])
    return indices[(admitted_offset + 1) % len(indices)]


def build_run_runtime_config(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    run: RunSpec,
) -> dict[str, Any]:
    """Resolve one immutable generator config from campaign-owned YAML values."""
    base_path = _repo_path(REPO_ROOT, runtime.get("base_config"), "base config")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("R9 base config must contain a mapping")
    manifests = _mapping(manifest_contract.get("manifests"), "manifest contract")
    if run.manifest_key == "diagnose_18":
        provenance = _mapping(
            manifest_contract.get("provenance"), "manifest provenance"
        )
        manifest = _mapping(provenance.get("diagnose_18"), "diagnose manifest")
    else:
        manifest = _mapping(manifests.get(run.manifest_key), run.manifest_key)
    campaign_runtime_sha256 = _require_sha256(
        campaign_runtime.get("campaign_runtime_sha256"), "campaign runtime SHA256"
    )
    manifest_contracts_sha256 = _require_sha256(
        manifest_contract.get("manifest_contracts_sha256"),
        "manifest contracts SHA256",
    )
    checkpoint = _mapping(campaign_runtime.get("checkpoint"), "checkpoint")
    if base.get("checkpoint_sha256") != checkpoint.get("sha256"):
        raise ValueError("base config checkpoint disagrees with campaign runtime")
    formal_closure = _formal_closure_for_runtime(campaign_runtime)
    config = dict(base)
    for field in (
        "sample_mode",
        "optimization_mode",
        "num_optim_iters",
        "step_size",
        "active_guidance_intervals",
        "collect_interval_diagnostics",
        "r9_guidance_interval_contract",
        "arm_config_sha256",
        "locked_schedule",
    ):
        config.pop(field, None)
    config.update(
        {
            "experiment_name": f"{run.phase}__{run.logical_run_id}",
            "experiment_contract": str(runtime["experiment_contract"]),
            "out_dir": str(run.output_dir),
            "phase": run.phase,
            "seed": run.seed,
            "sampling_seed": run.seed,
            "device": "cuda:0",
            "max_samples": run.sample_count,
            "sample_id_manifest": str(manifest["path"]),
            "sample_id_manifest_sha256": str(manifest["sha256"]),
            "calibration_sample_id_manifest": str(
                _mapping(manifests["calibration_64"], "calibration_64")["path"]
            ),
            "calibration_sample_id_manifest_sha256": str(
                _mapping(manifests["calibration_64"], "calibration_64")["sha256"]
            ),
            "asset_digest_cache": str(
                run.output_dir.parent / "shared" / f"{run.logical_run_id}.assets.json"
            ),
            "contact_sheets": False,
            "r9_campaign_id": str(campaign_runtime["campaign_id"]),
            "r9_campaign_runtime_sha256": campaign_runtime_sha256,
            "r9_manifest_contracts_sha256": manifest_contracts_sha256,
            "r9_phase_manifest_sha256": str(manifest["sha256"]),
        }
    )
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is not None:
        config["r9_continuation_contract_sha256"] = continuation[
            "continuation_contract_sha256"
        ]
    if formal_closure is not None:
        closure_binding = _mapping(formal_closure.get("closure"), "semigroup closure")
        config.update(
            {
                "r9_semigroup_closure_seal": str(closure_binding["path"]),
                "r9_semigroup_closure_seal_sha256": str(closure_binding["file_sha256"]),
                "r9_semigroup_closure_contract_sha256": str(
                    closure_binding["contract_sha256"]
                ),
                "r9_semigroup_bootstrap_campaign_id": str(
                    formal_closure["bootstrap_campaign_id"]
                ),
            }
        )
    if run.phase == "preflight":
        config["mode"] = "semigroup"
        config["phase"] = "semigroup"
        if formal_closure is not None:
            _bind_locked_schedule(config, campaign_runtime)
    else:
        arm = _diagnose_arm(runtime, run.arm_ref)
        config["mode"] = str(arm["mode"])
        if config["mode"] in {"official_head_current_xt", "paper_algorithm_split"}:
            for field in (
                "sample_mode",
                "optimization_mode",
                "num_optim_iters",
                "step_size",
                "active_guidance_intervals",
            ):
                if field in arm:
                    config[field] = arm[field]
            config["collect_interval_diagnostics"] = bool(
                arm.get("collect_interval_diagnostics", False)
                if run.phase == "diagnose"
                else False
            )
            _bind_locked_schedule(config, campaign_runtime)
    resolved = resolve_frozen_effective_guidance_config(config)
    if continuation is not None and run.arm_ref != "native":
        selected = {
            str(row["arm_id"]): row for row in continuation["selected_arms"]
        }
        if run.arm_ref not in selected:
            raise ValueError("child run references a candidate outside parent A selection")
        algorithm_config_sha256 = canonical_r9_algorithm_config_digest(
            resolved, str(checkpoint["sha256"])
        )
        if algorithm_config_sha256 != selected[run.arm_ref][
            "config_sha256"
        ]:
            raise ValueError("child candidate arm config drifted from parent A evidence")
    return resolved


def materialize_phase_runtime_configs(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    plan: PhasePlan,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    by_logical_run: dict[str, RunSpec] = {}
    for run in plan.runs:
        by_logical_run.setdefault(run.logical_run_id, run)
    for logical_run_id in sorted(by_logical_run):
        run = by_logical_run[logical_run_id]
        config = build_run_runtime_config(
            runtime, campaign_runtime, manifest_contract, run
        )
        content = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
        destination = REPO_ROOT / run.runtime_config
        _write_immutable_bytes(destination, content)
        records.append(
            {
                "logical_run_id": logical_run_id,
                "arm_id": run.arm_ref,
                "seed": run.seed,
                "manifest_sha256": config["sample_id_manifest_sha256"],
                "path": str(run.runtime_config),
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "campaign_runtime_sha256": config["r9_campaign_runtime_sha256"],
                "manifest_contracts_sha256": config["r9_manifest_contracts_sha256"],
                "checkpoint_sha256": config["checkpoint_sha256"],
                "determinism_policy_sha256": config["determinism_policy_sha256"],
                "schedule_contract_sha256": config.get("schedule_contract_sha256"),
                "semigroup_gate_contract_sha256": config.get(
                    "r9_semigroup_gate_contract_sha256"
                ),
                "semigroup_closure_seal_sha256": config.get(
                    "r9_semigroup_closure_seal_sha256"
                ),
                "semigroup_closure_contract_sha256": config.get(
                    "r9_semigroup_closure_contract_sha256"
                ),
                "continuation_contract_sha256": config.get(
                    "r9_continuation_contract_sha256"
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_runtime_configs_v1",
        "phase": plan.phase,
        "campaign_id": plan.campaign_id,
        "campaign_runtime_sha256": campaign_runtime["campaign_runtime_sha256"],
        "manifest_contracts_sha256": manifest_contract["manifest_contracts_sha256"],
        "runs": records,
    }
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is not None:
        payload["continuation_contract_sha256"] = continuation[
            "continuation_contract_sha256"
        ]
    payload["runtime_configs_sha256"] = _canonical_json_sha256(payload)
    contract_path = REPO_ROOT / plan.campaign_root / plan.phase / "runtime_configs.json"
    _write_immutable_bytes(
        contract_path,
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    return payload


def _diagnose_arm(runtime: Mapping[str, Any], arm_ref: str) -> Mapping[str, Any]:
    if arm_ref in {"native", "semigroup_preflight"}:
        return {"arm_id": "native", "family": "native", "mode": "native"}
    arms = _mapping(
        _mapping(runtime.get("phases"), "phases")["diagnose"], "diagnose"
    ).get("arms")
    if not isinstance(arms, list):
        raise ValueError("diagnose arms must be a list")
    matches = [
        arm for arm in arms if isinstance(arm, Mapping) and arm.get("arm_id") == arm_ref
    ]
    if len(matches) != 1:
        raise ValueError(f"promoted arm {arm_ref!r} is not uniquely registered")
    return matches[0]


def _bind_locked_schedule(
    config: dict[str, Any], campaign_runtime: Mapping[str, Any]
) -> None:
    schedule = _mapping(campaign_runtime.get("schedule"), "schedule")
    gate = _mapping(campaign_runtime.get("semigroup_gate"), "semigroup gate")
    manifest_path = _repo_path(REPO_ROOT, schedule.get("path"), "locked schedule")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("locked schedule must contain a JSON object")
    config["schedule_manifest"] = str(schedule["path"])
    config["schedule_contract_sha256"] = str(schedule["contract_sha256"])
    for field in (
        "semigroup_report",
        "semigroup_sample_id_manifest",
        "semigroup_sample_id_manifest_sha256",
        "semigroup_preflight_contract",
        "semigroup_preflight_contract_sha256",
        "r9_semigroup_gate_contract",
        "r9_semigroup_gate_contract_sha256",
    ):
        if field not in payload:
            raise ValueError(f"locked schedule is missing {field}")
        config[field] = payload[field]
    if config["r9_semigroup_gate_contract"] != gate["path"]:
        raise ValueError("campaign semigroup gate path disagrees with locked schedule")
    if config["r9_semigroup_gate_contract_sha256"] != gate["file_sha256"]:
        raise ValueError(
            "campaign semigroup gate digest disagrees with locked schedule"
        )


def _write_immutable_bytes(path: Path, content: bytes) -> None:
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError(f"immutable path must not be a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise ValueError(
                f"immutable file already has different content: {destination}"
            )
        return
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
                raise ValueError(f"concurrent immutable write disagrees: {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_exclusive_bytes(path: Path, content: bytes) -> None:
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError(f"exclusive path must not be a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o444)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_tree_rss_bytes(root_pid: int, *, proc_root: Path = Path("/proc")) -> int:
    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        raise ValueError("process-tree root PID must be a positive integer")
    pending = [root_pid]
    seen: set[int] = set()
    total_kib = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        status_path = proc_root / str(pid) / "status"
        children_path = proc_root / str(pid) / "task" / str(pid) / "children"
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            if pid == root_pid:
                raise _ProcessTreeRootExitObserved(root_pid, "vanished") from None
            continue
        rss_lines = [line for line in lines if line.startswith("VmRSS:")]
        if len(rss_lines) != 1:
            if pid == root_pid:
                state_lines = [line for line in lines if line.startswith("State:")]
                if len(state_lines) != 1:
                    raise ResourceContractError(
                        f"process {pid} has no unique process state"
                    )
                state_fields = state_lines[0].split()
                if len(state_fields) < 2:
                    raise ResourceContractError(
                        f"process {pid} process state format is invalid"
                    )
                state = state_fields[1]
                if state == "Z":
                    raise _ProcessTreeRootExitObserved(root_pid, "zombie")
                raise ResourceContractError(
                    f"process {pid} has no unique VmRSS while in live state {state}"
                )
            raise ResourceContractError(f"process {pid} has no unique VmRSS")
        fields = rss_lines[0].split()
        if len(fields) != 3 or fields[2] != "kB":
            raise ResourceContractError(f"process {pid} VmRSS format is invalid")
        total_kib += int(fields[1])
        try:
            children = children_path.read_text(encoding="utf-8").split()
        except FileNotFoundError:
            children = []
        pending.extend(int(value) for value in children)
    return total_kib * 1024


def _logical_runs(
    runtime: Mapping[str, Any],
    phase: str,
    phase_config: Mapping[str, Any],
    *,
    promoted_arm_ids: Sequence[str] | None = None,
    winner_arm_id: str | None = None,
) -> list[tuple[str, str, int, int | None]]:
    if phase == "preflight":
        seed = _positive_int(phase_config["seed"], "preflight seed")
        return [("semigroup_preflight", "semigroup_preflight", seed, None)]
    if phase == "diagnose":
        seed = _positive_int(phase_config["seed"], "diagnose seed")
        repeats = _positive_int(phase_config["repeats"], "diagnose repeats")
        arm_ids = _arm_ids(phase_config, expected=13)
        return [
            (f"{arm_id}__repeat_{repeat_index}", arm_id, seed, repeat_index)
            for repeat_index in range(repeats)
            for arm_id in arm_ids
        ]
    if phase == "calibrate":
        seeds = _seed_list(phase_config, expected=3)
        candidate_slots = _positive_int(
            phase_config["candidate_slots"], "calibrate candidate_slots"
        )
        if candidate_slots != 3:
            raise ValueError("R9 calibration requires exactly three candidate slots")
        arm_refs = [
            "native",
            *(
                [f"diagnose_candidate_{index}" for index in range(3)]
                if promoted_arm_ids is None
                else _validate_promoted_arms(runtime, promoted_arm_ids, maximum=3)
            ),
        ]
        return [
            (f"{arm_ref}__seed_{seed}", arm_ref, seed, None)
            for seed in seeds
            for arm_ref in arm_refs
        ]
    if phase == "confirm512":
        seed = _positive_int(phase_config["seed"], "confirm512 seed")
        candidate_slots = _positive_int(
            phase_config["candidate_slots"], "confirm512 candidate_slots"
        )
        if candidate_slots != 2:
            raise ValueError("R9 confirm512 requires exactly two candidate slots")
        arm_refs = [
            "native",
            *(
                ["calibrate_candidate_0", "calibrate_candidate_1"]
                if promoted_arm_ids is None
                else _validate_promoted_arms(runtime, promoted_arm_ids, maximum=2)
            ),
        ]
        return [(arm_ref, arm_ref, seed, None) for arm_ref in arm_refs]
    seed = _positive_int(phase_config["seed"], "full seed")
    winner = "selection_winner" if winner_arm_id is None else winner_arm_id
    if winner_arm_id is not None:
        _validate_promoted_arms(runtime, [winner_arm_id], maximum=1)
    return [("native", "native", seed, None), ("winner", winner, seed, None)]


def _validate_promoted_arms(
    runtime: Mapping[str, Any], values: Sequence[str], *, maximum: int
) -> list[str]:
    arms = [str(value) for value in values]
    if not arms or len(arms) > maximum or len(set(arms)) != len(arms):
        raise ValueError(f"R9 promotion requires 1..{maximum} unique arm IDs")
    registered = set(
        _arm_ids(
            _mapping(_mapping(runtime.get("phases"), "phases")["diagnose"], "diagnose"),
            expected=13,
        )
    )
    if any(arm == "native" or arm not in registered for arm in arms):
        raise ValueError("R9 promotion references an unregistered candidate arm")
    return arms


def _arm_ids(phase_config: Mapping[str, Any], *, expected: int) -> list[str]:
    arms = phase_config.get("arms")
    if not isinstance(arms, list) or len(arms) != expected:
        raise ValueError(f"R9 diagnose requires exactly {expected} YAML arms")
    arm_ids: list[str] = []
    for arm in arms:
        payload = _mapping(arm, "diagnose arm")
        arm_id = str(payload.get("arm_id", ""))
        if not arm_id or arm_id in arm_ids:
            raise ValueError("R9 diagnose arm IDs must be unique non-empty strings")
        arm_ids.append(arm_id)
    return arm_ids


def _manifest_contract_entry(manifests: Mapping[str, Any], name: str) -> dict[str, Any]:
    declared = _mapping(manifests.get(name), f"manifest {name}")
    return {
        "path": str(declared["path"]),
        "sha256": str(declared["sha256"]),
        "sample_count": _positive_int(declared["sample_count"], f"{name} count"),
        "ordered_sample_id_sha256": str(declared["ordered_sample_id_sha256"]),
    }


def _seed_list(phase_config: Mapping[str, Any], *, expected: int) -> list[int]:
    values = phase_config.get("seeds")
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"R9 phase requires exactly {expected} seeds")
    seeds = [_positive_int(value, "sampling seed") for value in values]
    if len(set(seeds)) != len(seeds):
        raise ValueError("R9 sampling seeds must be unique")
    return seeds


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _repo_path(repo_root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    resolved_root = Path(repo_root).resolve()
    resolved = (
        (resolved_root / path).resolve() if not path.is_absolute() else path.resolve()
    )
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repo root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be a mapping")
        rows.append(row)
    return rows


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
