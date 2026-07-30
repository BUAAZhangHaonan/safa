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
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
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
from safa.evaluation.r9_calibration_selection_contracts import (
    build_calibration_report_only_selection_contract,
    materialize_calibration_report_only_selection_contract,
)
from safa.evaluation.r9_confirm_continuation_contracts import (
    build_confirm_continuation_contract,
    materialize_confirm_continuation_contract,
)
from safa.evaluation.r9_full_continuation_contracts import (
    CHILD_CAMPAIGN_ID as FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
    build_full_continuation_contract,
    build_full_continuation_selection_contract,
    expected_source_from_full_continuation,
    materialize_full_continuation_contract,
    validate_full_continuation_selection_contract,
)
from safa.evaluation.r9_full_smoke_supersession import (
    materialize_full_smoke_supersession,
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
    SampleEvidence,
    canonical_r9_algorithm_config_digest,
    materialize_phase_results,
    resume_phase_results,
)
from safa.evaluation.r9_evaluator_resources import (
    materialize_evaluator_resource_profiles,
)
from safa.evaluation.r9_generation_batch_benchmark import (
    BatchRunEvidence,
    BenchmarkGpuSnapshot,
    build_generation_batch_benchmark_contract,
    materialize_generation_batch_benchmark_contract,
    validate_generation_batch_benchmark_contract,
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
CONFIRM_CONTINUATION_RUNTIME_CONFIG = Path(
    "configs/medium_v2/experiments/r9_meanflow_confirm_continuation_campaign_v8.yaml"
)
CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID = "r9-report-only-formal-v8"
FULL_CONTINUATION_RUNTIME_CONFIG = Path(
    "configs/medium_v2/experiments/r9_meanflow_full_continuation_campaign_v9.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("preflight", "diagnose", "calibrate", "confirm512", "full")
CAMPAIGN_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
RESOURCE_SMOKE_ROOT_EXIT_WAIT_SECONDS = 1.0
FULL_GUARDED_MAX_ACTIVE_WORKERS = 2


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
    gpu_index: int


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
    if (
        args.campaign_id == FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        and args.allow_busy_gpus
    ):
        parser.error("v9 Full forbids --allow-busy-gpus")
    if (
        args.campaign_id != FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        and args.execute
        and not args.allow_busy_gpus
    ):
        parser.error("R9 execution requires explicit --allow-busy-gpus")
    rejected = {
        CONTINUATION_CHILD_CAMPAIGN_ID: {"preflight", "diagnose"},
        CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID: {
            "preflight",
            "diagnose",
            "calibrate",
        },
    }
    if (
        args.campaign_id == FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        and args.phase != "full"
    ):
        parser.error("Full continuation only accepts --phase full")
    if args.phase in rejected.get(args.campaign_id, set()):
        parser.error("continuation child rejects the requested upstream phase")
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


def load_confirm_continuation_request(
    path: Path = CONFIRM_CONTINUATION_RUNTIME_CONFIG,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    request_path = _repo_path(repo_root, path, "confirm continuation request")
    request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "contract_type",
        "child_campaign_id",
        "start_phase",
        "base_runtime",
        "source",
        "selection",
        "generation_batch_benchmark",
        "semigroup_closure_campaign_id",
        "evaluator_resources",
    }
    if not isinstance(request, Mapping) or set(request) != expected_fields:
        raise ValueError("confirm continuation request fields are not canonical")
    if (
        request.get("schema_version") != 1
        or request.get("contract_type")
        != "safa_r9_confirm_continuation_request_v1"
        or request.get("child_campaign_id")
        != CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID
        or request.get("start_phase") != "confirm512"
        or request.get("semigroup_closure_campaign_id")
        != "r9-report-only-formal-v2"
    ):
        raise ValueError("confirm continuation request identity mismatch")
    base = _mapping(request.get("base_runtime"), "confirm base runtime")
    if set(base) != {"path", "sha256"}:
        raise ValueError("confirm base runtime fields are not canonical")
    base_path = _repo_path(repo_root, base.get("path"), "confirm base runtime")
    if _sha256_path(base_path) != _require_sha256(
        base.get("sha256"), "confirm base runtime SHA256"
    ):
        raise ValueError("confirm base runtime SHA256 mismatch")
    source = _mapping(request.get("source"), "confirm continuation source")
    expected_source = {
        "campaign_id": "r9-report-only-formal-v6",
        "calibrate_gate_contract_sha256": (
            "84c4aa802965601bfeccc03fa0e9da2baef25d8cc98cb9dbbc536058037520b9"
        ),
        "calibrate_phase_results_sha256": (
            "2be463aaadc7b5cf9f4cfd87b452034bdcecd3bf65d13ecb3bebb4b68844a35c"
        ),
        "automatic_evidence_sha256": (
            "c9840ff3a4c96b64db386e64a543e2c637b69ec0f9cd2453913070267ecaffbe"
        ),
        "evaluation_repair_contract_sha256": (
            "716355ccf9171d3b6d35f51c124139e110b99986393ed7e2b397c02d7c0fb355"
        ),
        "generation_inventory_sha256": (
            "e40516b8dc852c6b6930e38b89b656b28274c91f40003884ab688d8768ab145a"
        ),
    }
    if source != expected_source:
        raise ValueError("confirm continuation source changed")
    selection = _mapping(request.get("selection"), "confirm selection")
    if selection != {
        "path": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v8/calibration_report_only_selection.json"
        ),
        "materialization": "build_from_frozen_source_o_excl",
    }:
        raise ValueError("confirm selection declaration changed")
    runtime = load_runtime_config(base_path)
    evaluation = _mapping(runtime.get("evaluation"), "confirm evaluation")
    worker = _mapping(evaluation.get("worker"), "confirm evaluator worker")
    worker["sha256"] = _sha256_path(
        _repo_path(repo_root, worker.get("path"), "confirm evaluator entrypoint")
    )
    worker["implementation_sha256"] = _sha256_path(
        _repo_path(
            repo_root,
            worker.get("implementation_path"),
            "confirm evaluator implementation",
        )
    )
    evaluation["worker"] = worker
    quality = _mapping(evaluation.get("quality"), "confirm quality")
    quality_script = _mapping(quality.get("script"), "confirm quality script")
    quality_script["sha256"] = _sha256_path(
        _repo_path(repo_root, quality_script.get("path"), "confirm quality script")
    )
    quality["script"] = quality_script
    evaluation["quality"] = quality
    resources = _mapping(
        request.get("evaluator_resources"), "confirm evaluator resources"
    )
    if set(resources) != {"arcface", "quality", "heldout"}:
        raise ValueError("confirm evaluator resources are not canonical")
    evaluation["resource_smokes"] = resources
    runtime["evaluation"] = evaluation
    batch_benchmark = _mapping(
        request.get("generation_batch_benchmark"),
        "confirm generation batch benchmark",
    )
    if batch_benchmark != {
        "contract_path": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v8/generation_batch_benchmark.json"
        ),
        "output_root": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v8/generation_batch_benchmark"
        ),
        "manifest": "calibration_64",
        "sample_count": 8,
        "seed": 4549,
        "required_arms": [
            "native",
            "paper_eta_0p125",
            "flow_map2_normalized_eta_0p125",
        ],
        "batch_sizes": [2, 4],
        "record_final_latent_sha256": True,
    }:
        raise ValueError("confirm generation batch benchmark declaration changed")
    runtime["generation_batch_benchmark"] = batch_benchmark
    source["request_contract_type"] = str(request["contract_type"])
    return runtime, Path(str(request_path.relative_to(repo_root.resolve()))), source


def load_full_continuation_request(
    path: Path = FULL_CONTINUATION_RUNTIME_CONFIG,
    *,
    repo_root: Path = REPO_ROOT,
    validate_chain: bool = True,
    allow_pre_e2e_profiles: bool = False,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    request_path = _repo_path(repo_root, path, "Full continuation request")
    request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "contract_type",
        "child_campaign_id",
        "start_phase",
        "base_runtime",
        "source",
        "generation_batch_benchmark",
        "semigroup_closure_campaign_id",
        "evaluator_resources",
    }
    if not isinstance(request, Mapping) or set(request) != fields:
        raise ValueError("Full continuation request fields are not canonical")
    if (
        request.get("schema_version") != 1
        or request.get("contract_type")
        != "safa_r9_full_continuation_request_v1"
        or request.get("child_campaign_id") != FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        or request.get("start_phase") != "full"
        or request.get("semigroup_closure_campaign_id")
        != "r9-report-only-formal-v2"
    ):
        raise ValueError("Full continuation request identity mismatch")
    base = _mapping(request.get("base_runtime"), "Full base runtime")
    if set(base) != {"path", "sha256"}:
        raise ValueError("Full base runtime fields are not canonical")
    base_path = _repo_path(repo_root, base.get("path"), "Full base runtime")
    if _sha256_path(base_path) != _require_sha256(
        base.get("sha256"), "Full base runtime SHA256"
    ):
        raise ValueError("Full base runtime SHA256 mismatch")
    source = _mapping(request.get("source"), "Full source")
    # The contract builder rehashes and cross-binds every declared source.
    if validate_chain:
        build_full_continuation_contract(
            repo_root=repo_root, expected_source=source
        )
    runtime = load_runtime_config(base_path)
    evaluation = _mapping(runtime.get("evaluation"), "Full evaluation")
    worker = _mapping(evaluation.get("worker"), "Full evaluator worker")
    worker["sha256"] = _sha256_path(
        _repo_path(repo_root, worker.get("path"), "Full evaluator entrypoint")
    )
    worker["implementation_sha256"] = _sha256_path(
        _repo_path(
            repo_root,
            worker.get("implementation_path"),
            "Full evaluator implementation",
        )
    )
    evaluation["worker"] = worker
    quality = _mapping(evaluation.get("quality"), "Full quality")
    quality_script = _mapping(quality.get("script"), "Full quality script")
    quality_script["sha256"] = _sha256_path(
        _repo_path(repo_root, quality_script.get("path"), "Full quality script")
    )
    quality["script"] = quality_script
    evaluation["quality"] = quality
    resources = _mapping(request.get("evaluator_resources"), "Full evaluator resources")
    if set(resources) != {"arcface", "quality", "heldout"}:
        raise ValueError("Full evaluator resources are not canonical")
    for kind, mode in (
        ("arcface", "measured_single_worker"),
        ("quality", "measured_exclusive_bootstrap"),
    ):
        if _mapping(resources.get(kind), kind) != {
            "mode": mode,
            "source": "full_e2e_profile_v1",
        }:
            raise ValueError(f"Full {kind} E2E profile declaration changed")
    profile_path = (
        repo_root.resolve()
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
        f"{FULL_CONTINUATION_CHILD_CAMPAIGN_ID}/full_e2e/resource_profiles.json"
    )
    if not profile_path.is_file():
        if not allow_pre_e2e_profiles:
            raise FileNotFoundError(
                f"Full E2E resource profiles are missing: {profile_path}"
            )
        normalized_resources = {
            "profile_state": "pre_e2e_not_execution_authority",
        }
    else:
        profile_runtime = {
            "campaign_root": (
                "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
                f"{FULL_CONTINUATION_CHILD_CAMPAIGN_ID}"
            ),
            "evaluation": evaluation,
        }
        profiles = build_full_e2e_resource_profiles(profile_runtime)
        observed_profiles = _read_json_mapping(
            profile_path, "Full E2E resource profiles"
        )
        if observed_profiles != profiles:
            raise ValueError("Full E2E resource profile bytes changed")
        normalized_resources = {
            "arcface": dict(profiles["arcface"]),
            "quality": dict(profiles["quality"]),
            "heldout": dict(profiles["heldout"]),
            "resource_profiles_sha256": profiles["resource_profiles_sha256"],
            "resource_profile_binding": {
                "path": str(profile_path.relative_to(repo_root.resolve())),
                "file_sha256": _sha256_path(profile_path),
                "contract_sha256": profiles["resource_profiles_sha256"],
            },
        }
    evaluation["resource_smokes"] = normalized_resources
    runtime["evaluation"] = evaluation
    benchmark = _mapping(
        request.get("generation_batch_benchmark"), "Full batch benchmark"
    )
    expected_benchmark_fields = {
        "contract_path",
        "source_campaign_id",
        "source_continuation_contract_sha256",
        "manifest",
        "sample_count",
        "seed",
        "required_arms",
        "batch_sizes",
        "selected_batch_size",
        "selected_slots_per_gpu",
    }
    if set(benchmark) != expected_benchmark_fields:
        raise ValueError("Full batch benchmark declaration is not canonical")
    if (
        benchmark.get("source_campaign_id") != "r9-report-only-formal-v8"
        or benchmark.get("selected_batch_size") != 2
        or benchmark.get("selected_slots_per_gpu") != 2
        or benchmark.get("batch_sizes") != [2, 4]
    ):
        raise ValueError("Full requires the frozen batch=2/two-worker decision")
    runtime["generation_batch_benchmark"] = benchmark
    return runtime, Path(str(request_path.relative_to(repo_root.resolve()))), source


def prepare_full_continuation_evaluator_smoke_requests(
    *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    return materialize_full_smoke_supersession(repo_root=repo_root)


def load_campaign_configuration(
    campaign_id: str,
) -> tuple[dict[str, Any], Path, dict[str, Any] | None]:
    if campaign_id == CONTINUATION_CHILD_CAMPAIGN_ID:
        return load_continuation_request()
    if campaign_id == CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID:
        return load_confirm_continuation_request()
    if campaign_id == FULL_CONTINUATION_CHILD_CAMPAIGN_ID:
        return load_full_continuation_request()
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
    if _continuation_digest(payload) != row["contract_sha256"]:
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
    closure_campaign_id = campaign_id
    if continuation is not None:
        if "semigroup_closure_campaign_id" in continuation:
            closure_campaign_id = str(
                continuation["semigroup_closure_campaign_id"]
            )
        else:
            closure_campaign_id = str(
                _mapping(continuation.get("parent"), "continuation parent")[
                    "campaign_id"
                ]
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
        continuation = (
            validate_continuation_contract(
                continuation_contract, repo_root=REPO_ROOT
            )
            if continuation_contract is not None
            else _continuation_for_runtime(campaign_runtime)
        )
        start_phase = (
            str(continuation.get("start_phase", "calibrate"))
            if continuation is not None
            else "calibrate"
        )
        rejected = {
            "calibrate": {"preflight", "diagnose"},
            "confirm512": {"preflight", "diagnose", "calibrate"},
            "full": {"preflight", "diagnose", "calibrate", "confirm512"},
        }[start_phase]
        if requested_phase in rejected:
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
    continuation_start_phase: str | None = None,
) -> tuple[PhasePlan, ...]:
    continuation = continuation_selected_arm_ids is not None
    if continuation and phase in {"preflight", "diagnose"}:
        raise ValueError("continuation child rejects preflight and diagnose")
    if continuation_start_phase not in {None, "calibrate", "confirm512", "full"}:
        raise ValueError("continuation start phase is invalid")
    if continuation_start_phase == "confirm512" and phase in {
        "preflight",
        "diagnose",
        "calibrate",
    }:
        raise ValueError("confirm continuation rejects upstream phases")
    if continuation_start_phase == "full" and phase != "full":
        raise ValueError("Full continuation rejects every upstream phase")
    if continuation and phase == "all":
        selected = (
            ("full",)
            if continuation_start_phase == "full"
            else ("confirm512", "full")
            if continuation_start_phase == "confirm512"
            else ("calibrate", "confirm512", "full")
        )
    else:
        selected = PHASES if phase == "all" else (phase,)
    return tuple(
        build_phase_plan(
            runtime,
            phase=item,
            campaign_id=campaign_id,
            promoted_arm_ids=(
                continuation_selected_arm_ids
                if item == (continuation_start_phase or "calibrate")
                and item != "full"
                else None
            ),
            winner_arm_id=(
                str(continuation_selected_arm_ids[0])
                if item == "full"
                and continuation_start_phase == "full"
                and continuation_selected_arm_ids is not None
                else None
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
    continuation_contract_override: Mapping[str, Any] | None = None,
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
    if continuation_source is not None and continuation_contract_override is not None:
        raise ValueError("continuation source and override are mutually exclusive")
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
    elif continuation_contract_override is not None:
        continuation_contract = validate_continuation_contract(
            continuation_contract_override, repo_root=repo_root
        )
        _, _, continuation_binding = continuation_contract_binding(
            continuation_contract, repo_root=repo_root
        )
        closure_campaign_id = str(
            continuation_contract["semigroup_closure_campaign_id"]
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
    benchmark = _generation_batch_benchmark_for_runtime(
        runtime,
        campaign_id=campaign_id,
        repo_root=repo_root,
        continuation_contract=continuation_contract,
        manifests=manifest_contract["manifests"],
    )
    if benchmark is not None:
        effective["generation_batch_benchmark"] = benchmark["binding"]
        decision = benchmark["decision"]
        resources["gpu_slot_claim_bytes"] = decision[
            "selected_gpu_slot_claim_bytes"
        ]
        resources["ram_slot_budget_bytes"] = decision[
            "selected_ram_slot_budget_bytes"
        ]
        resources["generation_batch_size"] = decision["selected_batch_size"]
        resources["generation_slots_per_gpu"] = decision[
            "selected_slots_per_gpu"
        ]
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


def _generation_batch_benchmark_for_runtime(
    runtime: Mapping[str, Any],
    *,
    campaign_id: str,
    repo_root: Path,
    continuation_contract: Mapping[str, Any] | None,
    manifests: Mapping[str, Any],
) -> dict[str, Any] | None:
    declaration_value = runtime.get("generation_batch_benchmark")
    if declaration_value is None:
        return None
    if continuation_contract is None:
        raise ValueError("generation batch benchmark requires a continuation")
    declaration = _mapping(declaration_value, "generation batch benchmark")
    relative = Path(str(declaration.get("contract_path")))
    if relative.is_absolute():
        raise ValueError("generation batch benchmark path must be relative")
    path = (repo_root.resolve() / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError("generation batch benchmark path escapes repo root") from error
    if not path.is_file():
        return None
    source_campaign_id = str(
        declaration.get("source_campaign_id", campaign_id)
    )
    source_continuation_sha256 = declaration.get(
        "source_continuation_contract_sha256",
        _continuation_digest(continuation_contract),
    )
    payload = validate_generation_batch_benchmark_contract(
        _read_json_mapping(path, "generation batch benchmark"),
        repo_root=repo_root,
        expected_campaign_id=source_campaign_id,
        expected_continuation_contract_sha256=_require_sha256(
            source_continuation_sha256,
            "source continuation contract SHA256",
        ),
    )
    manifest = _mapping(payload.get("manifest"), "benchmark manifest")
    expected_manifest = _mapping(
        manifests.get(str(declaration["manifest"])), "declared benchmark manifest"
    )
    if (
        manifest.get("path") != expected_manifest.get("path")
        or manifest.get("sha256") != expected_manifest.get("sha256")
        or manifest.get("sample_count") != declaration.get("sample_count")
        or payload.get("seed") != declaration.get("seed")
        or [row["arm_id"] for row in payload.get("arms", [])]
        != sorted(declaration.get("required_arms", []))
    ):
        raise ValueError("generation batch benchmark disagrees with its request")
    if payload.get("status") != "ready":
        raise ResourceContractError("generation batch benchmark blocked confirm512")
    decision = _mapping(payload.get("decision"), "batch benchmark decision")
    selected_batch_size = declaration.get("selected_batch_size")
    selected_slots_per_gpu = declaration.get("selected_slots_per_gpu")
    if selected_batch_size is not None:
        if (
            selected_batch_size != decision.get("selected_batch_size")
            or selected_batch_size != 2
        ):
            raise ValueError("declared generation batch size changed")
        slots = _positive_int(
            selected_slots_per_gpu, "selected generation slots per GPU"
        )
        if slots > int(decision.get("selected_slots_per_gpu", 0)):
            raise ValueError("declared generation workers exceed measured capacity")
        decision = dict(decision)
        decision["selected_slots_per_gpu"] = slots
        decision["aggregate_batch_per_gpu"] = selected_batch_size * slots
        decision["aggregate_batch_four_gpus"] = (
            selected_batch_size * slots * 4
        )
    return {
        "binding": {
            "path": str(path.relative_to(repo_root.resolve())),
            "file_sha256": _sha256_path(path),
            "contract_sha256": payload["generation_batch_benchmark_sha256"],
        },
        "decision": decision,
    }


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
    raw_resources = _mapping(
        raw.get("resource_smokes"), "evaluator resource profiles"
    )
    if "resource_profile_binding" in raw_resources:
        resource_smokes = _validate_full_e2e_runtime_resource_profiles(
            raw_resources,
            repo_root=repo_root,
            worker_contract=_normalized_worker_contract(
                {"worker": normalized_worker},
                repo_root=repo_root,
            ),
            arcface_contract_sha256=_canonical_json_sha256(normalized_arcface),
            quality_script_sha256=normalized_quality["script"]["sha256"],
        )
    else:
        resource_smokes = materialize_evaluator_resource_profiles(
            raw_resources,
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


def _validate_full_e2e_runtime_resource_profiles(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    worker_contract: Mapping[str, Any],
    arcface_contract_sha256: str,
    quality_script_sha256: str,
) -> dict[str, Any]:
    normalized = _mapping(value, "Full E2E runtime resource profiles")
    if set(normalized) != {
        "arcface",
        "quality",
        "heldout",
        "resource_profiles_sha256",
        "resource_profile_binding",
    }:
        raise ValueError("Full E2E runtime resource profile fields changed")
    binding = _mapping(
        normalized.get("resource_profile_binding"),
        "Full E2E resource profile binding",
    )
    profile_path = _repo_path(
        repo_root, binding.get("path"), "Full E2E resource profile"
    )
    profile = _read_json_mapping(
        profile_path, "Full E2E resource profile contract"
    )
    declared = _require_sha256(
        profile.get("resource_profiles_sha256"),
        "Full E2E resource profiles SHA256",
    )
    canonical = dict(profile)
    canonical.pop("resource_profiles_sha256")
    if (
        binding
        != {
            "path": str(profile_path.relative_to(repo_root.resolve())),
            "file_sha256": _sha256_path(profile_path),
            "contract_sha256": declared,
        }
        or _canonical_json_sha256(canonical) != declared
        or profile.get("contract_type")
        != "safa_r9_full_e2e_resource_profiles_v1"
        or profile.get("worker_contract") != dict(worker_contract)
        or profile.get("arcface_contract_sha256")
        != arcface_contract_sha256
        or profile.get("quality_script_sha256") != quality_script_sha256
        or normalized["resource_profiles_sha256"] != declared
        or normalized["arcface"] != profile.get("arcface")
        or normalized["quality"] != profile.get("quality")
        or normalized["heldout"] != profile.get("heldout")
    ):
        raise ValueError(
            "Full runtime does not bind the current measured E2E resource profile"
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
            raise ValueError(f"Full E2E {kind} resource budget changed")
    return normalized


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
    benchmark = runtime.get("generation_batch_benchmark")
    if benchmark is not None:
        declaration = _mapping(benchmark, "generation batch benchmark")
        payload["generation_batch_benchmark"] = {
            "logical_run_count": len(declaration["required_arms"])
            * len(declaration["batch_sizes"]),
            "sample_run_count": len(declaration["required_arms"])
            * len(declaration["batch_sizes"])
            * int(declaration["sample_count"]),
            "arms": list(declaration["required_arms"]),
            "batch_sizes": list(declaration["batch_sizes"]),
            "contract_materialized": bool(
                effective_runtime
                and effective_runtime.get("generation_batch_benchmark")
            ),
        }
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    is_full_continuation = (
        str(args.campaign_id) == FULL_CONTINUATION_CHILD_CAMPAIGN_ID
    )
    if is_full_continuation and not args.execute:
        gate_path = (
            REPO_ROOT
            / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
            / FULL_CONTINUATION_CHILD_CAMPAIGN_ID
            / "full_e2e/gate_contract.json"
        )
        if not gate_path.is_file():
            runtime, _, continuation_source = load_full_continuation_request(
                allow_pre_e2e_profiles=True
            )
            full_continuation = build_full_continuation_contract(
                repo_root=REPO_ROOT, expected_source=continuation_source
            )
            blocked_plans = build_requested_plans(
                runtime,
                phase="full",
                campaign_id=FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
                continuation_selected_arm_ids=[
                    str(row["arm_id"])
                    for row in full_continuation["selected_arms"]
                ],
                continuation_start_phase="full",
            )
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "contract_type": "safa_r9_full_dry_run_blocked_v1",
                        "status": "blocked_missing_e2e",
                        "campaign_id": FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
                        "phase": "full",
                        "full_continuation_sha256": full_continuation[
                            "full_continuation_sha256"
                        ],
                        "required_gate": str(gate_path.relative_to(REPO_ROOT)),
                        "logical_run_count": blocked_plans[0].logical_run_count,
                        "artifact_write_count": 0,
                        "gpu_execution_count": 0,
                    },
                    sort_keys=True,
                )
            )
            return 0
    runtime, runtime_config_path, continuation_source = load_campaign_configuration(
        str(args.campaign_id)
    )
    is_confirm_continuation = (
        str(args.campaign_id) == CONFIRM_CONTINUATION_CHILD_CAMPAIGN_ID
    )
    calibration_selection = (
        build_calibration_report_only_selection_contract(repo_root=REPO_ROOT)
        if is_confirm_continuation
        else None
    )
    confirm_continuation = (
        build_confirm_continuation_contract(
            repo_root=REPO_ROOT, selection=calibration_selection
        )
        if calibration_selection is not None
        else None
    )
    full_selection = (
        build_full_continuation_selection_contract(
            repo_root=REPO_ROOT, expected_source=continuation_source
        )
        if is_full_continuation and continuation_source is not None
        else None
    )
    full_continuation = (
        build_full_continuation_contract(
            repo_root=REPO_ROOT, expected_source=continuation_source
        )
        if full_selection is not None and continuation_source is not None
        else None
    )
    legacy_continuation_source = (
        None
        if is_confirm_continuation or is_full_continuation
        else continuation_source
    )
    continuation_override = confirm_continuation or full_continuation
    effective_runtime, manifest_contract, diagnose_contract = (
        build_effective_campaign_runtime(
            runtime,
            campaign_id=str(args.campaign_id),
            repo_root=REPO_ROOT,
            runtime_config_path=runtime_config_path,
            continuation_source=legacy_continuation_source,
            continuation_contract_override=continuation_override,
        )
    )
    continuation_contract = (
        confirm_continuation
        if confirm_continuation is not None
        else full_continuation
        if full_continuation is not None
        else build_continuation_contract(
            repo_root=REPO_ROOT,
            child_campaign_id=str(args.campaign_id),
            source=legacy_continuation_source,
        )
        if legacy_continuation_source is not None
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
        continuation_start_phase=(
            None
            if continuation_contract is None
            else str(continuation_contract.get("start_phase", "calibrate"))
        ),
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
    if calibration_selection is not None:
        _, selection_binding = (
            materialize_calibration_report_only_selection_contract(
                repo_root=REPO_ROOT
            )
        )
        if confirm_continuation is None:
            raise RuntimeError("confirm continuation was not built")
        if confirm_continuation["selection"]["contract_sha256"] != selection_binding[
            "contract_sha256"
        ]:
            raise RuntimeError("materialized calibration selection changed")
        _, binding = materialize_confirm_continuation_contract(repo_root=REPO_ROOT)
        if effective_runtime.get("continuation") != binding:
            raise RuntimeError("materialized confirm continuation binding changed")
    elif full_continuation is not None and continuation_source is not None:
        _, binding, materialized_selection = materialize_full_continuation_contract(
            repo_root=REPO_ROOT,
            expected_source=continuation_source,
        )
        if effective_runtime.get("continuation") != binding:
            raise RuntimeError("materialized Full continuation binding changed")
        validated_selection = validate_full_continuation_selection_contract(
            materialized_selection,
            repo_root=REPO_ROOT,
            expected_source=continuation_source,
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
        seal = build_heldout_seal_contract(validated_selection, assets)
        write_immutable_contract(
            REPO_ROOT
            / str(effective_runtime["campaign_root"])
            / "heldout_seal.json",
            seal,
            digest_field="heldout_seal_sha256",
        )
    elif legacy_continuation_source is not None:
        _, binding = materialize_continuation_contract(
            repo_root=REPO_ROOT,
            child_campaign_id=str(args.campaign_id),
            source=legacy_continuation_source,
        )
        if effective_runtime.get("continuation") != binding:
            raise RuntimeError("materialized continuation binding changed")
    if effective_runtime.get("campaign_runtime_sha256") is None:
        if is_full_continuation:
            _full_admission_preflight()
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
                continuation_source=legacy_continuation_source,
                continuation_contract_override=continuation_override,
            )
        )
        _validate_requested_campaign_role(
            effective_runtime,
            requested_phase=str(args.phase),
            continuation_contract=continuation_contract,
        )
    if effective_runtime.get("campaign_runtime_sha256") is None:
        raise RuntimeError("resource smoke did not produce a final campaign runtime")
    if is_confirm_continuation and effective_runtime.get(
        "generation_batch_benchmark"
    ) is None:
        run_generation_batch_benchmark(
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
                continuation_contract_override=confirm_continuation,
            )
        )
        if effective_runtime.get("campaign_runtime_sha256") is None:
            raise RuntimeError(
                "generation batch benchmark did not produce a final campaign runtime"
            )
    _require_generation_batch_benchmark_before_confirm(
        is_confirm_continuation=is_confirm_continuation,
        campaign_runtime=effective_runtime,
    )
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
    if (
        is_full_continuation
        and (
            REPO_ROOT
            / str(effective_runtime["campaign_root"])
            / "formal_execution_claim.json"
        ).is_file()
    ):
        return _resume_formal_full_report_only(
            runtime,
            effective_runtime,
            manifest_contract,
            diagnose_contract,
            plan=plans[0],
            campaign_id=str(args.campaign_id),
        )
    runtime_guard = None
    formal_execution_claim = None
    if is_full_continuation:
        e2e_gate = _require_full_e2e_gate(effective_runtime)
        admission = _full_admission_preflight()
        if full_continuation is None:
            raise RuntimeError("Full continuation is missing")
        resource_policy = _mapping(
            _mapping(
                _mapping(full_continuation["bindings"], "Full bindings").get(
                    "full_e2e_requirement"
                ),
                "Full E2E requirement",
            ).get("policy"),
            "Full E2E policy",
        )["resource_policy"]
        campaign_root = REPO_ROOT / str(effective_runtime["campaign_root"])
        runtime_guard = FullRuntimeGuard(
            resource_policy,
            monitor_path=(
                campaign_root / "formal_monitor/runtime_guard_samples.jsonl"
            ),
            allowed_external_gpu_pids=admission.get(
                "external_compute_pid_baseline"
            ),
        )
        monitor_claim = _materialize_formal_full_monitor_claim(
            effective_runtime
        )
        admission_path = campaign_root / "formal_admission.json"
        _write_exclusive_bytes(
            admission_path,
            (
                json.dumps(
                    admission,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        formal_execution_claim = {
            "schema_version": 1,
            "contract_type": "safa_r9_formal_full_execution_claim_v1",
            "campaign_runtime_sha256": effective_runtime[
                "campaign_runtime_sha256"
            ],
            "full_e2e_gate_sha256": e2e_gate["full_e2e_gate_sha256"],
            "full_admission": {
                "path": str(admission_path.relative_to(REPO_ROOT)),
                "file_sha256": _sha256_path(admission_path),
                "contract_sha256": admission["full_admission_sha256"],
            },
            "monitor_claim_sha256": monitor_claim["monitor_claim_sha256"],
            "retry_allowed": False,
        }
        formal_execution_claim["formal_execution_claim_sha256"] = (
            _canonical_json_sha256(formal_execution_claim)
        )
        _write_exclusive_bytes(
            campaign_root / "formal_execution_claim.json",
            (
                json.dumps(
                    formal_execution_claim,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        try:
            _start_formal_full_monitor(effective_runtime, monitor_claim)
            runtime_guard.bind_monitor(
                session_name=monitor_claim["session_name"],
                claim_path=campaign_root / "formal_monitor/claim.json",
                claim_sha256=monitor_claim["monitor_claim_sha256"],
            )
        except BaseException:
            subprocess.run(
                ["tmux", "kill-session", "-t", monitor_claim["session_name"]],
                check=False,
                capture_output=True,
            )
            _materialize_formal_full_terminal(
                effective_runtime,
                formal_execution_claim,
                status="failed_before_worker",
            )
            raise
    scheduler, gpu_bindings, peer_status_store = build_resource_scheduler(
        effective_runtime
    )
    evaluators = R9ProductionEvaluatorCallbacks(
        runtime=runtime,
        campaign_runtime=effective_runtime,
        scheduler=scheduler,
        gpu_bindings=gpu_bindings,
        peer_status_store=peer_status_store,
        runtime_guard=runtime_guard,
    )
    try:
        exit_code = execute_dynamic_campaign(
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
            runtime_guard=runtime_guard,
        )
    except BaseException:
        if formal_execution_claim is not None:
            _materialize_formal_full_terminal(
                effective_runtime,
                formal_execution_claim,
                status="failed",
            )
            _wait_formal_full_monitor(effective_runtime)
        raise
    if formal_execution_claim is not None:
        _materialize_formal_full_terminal(
            effective_runtime,
            formal_execution_claim,
            status=(
                "awaiting_visual_review"
                if exit_code == AWAITING_VISUAL_REVIEW_EXIT_CODE
                else "succeeded"
            ),
            exit_code=exit_code,
        )
        _wait_formal_full_monitor(effective_runtime)
    return exit_code


def _validate_admission_external_pid_policy(
    admission: Mapping[str, Any],
) -> bool:
    count = admission.get("unknown_compute_pid_count")
    if count == 0:
        return (
            "external_compute_pid_baseline" not in admission
            and "external_compute_pid_policy" not in admission
        )
    if type(count) is not int or count < 0:
        return False
    policy = admission.get("external_compute_pid_policy")
    if policy != {
        "schema_version": 1,
        "mode": "user_authorized_preexisting_gpu_pid_baseline_v1",
        "authorization_env": FULL_ADMISSION_EXTERNAL_PID_BASELINE_ENV,
        "new_unknown_gpu_pids": "forbidden_after_admission",
        "resource_hard_stops": "unchanged",
    }:
        return False
    try:
        baseline = _normalize_external_gpu_pid_baseline(
            admission.get("external_compute_pid_baseline")
        )
    except ResourceContractError:
        return False
    return len(baseline) == count


def _resume_formal_full_report_only(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    diagnose_contract: Mapping[str, Any],
    *,
    plan: PhasePlan,
    campaign_id: str,
) -> int:
    campaign_root = REPO_ROOT / str(campaign_runtime["campaign_root"])
    chain = _validate_formal_full_execution_chain(campaign_runtime)
    claim = chain["claim"]
    terminal = chain["terminal"]
    if terminal.get("status") != "awaiting_visual_review":
        raise RuntimeError(
            "formal Full existing execution is not report-only resumable"
        )
    _require_full_e2e_gate(campaign_runtime)
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
        _print_phase_closure("full", closure)
        return AWAITING_VISUAL_REVIEW_EXIT_CODE
    if closure.status != "complete":
        raise RuntimeError("formal Full report-only resume evidence changed")
    gate = finalize_phase_gate(
        runtime,
        campaign_runtime,
        manifest_contract,
        diagnose_contract,
        phase="full",
        campaign_id=campaign_id,
    )
    report = {
        "schema_version": 1,
        "contract_type": "safa_r9_formal_full_report_only_finalize_v1",
        "formal_execution_claim_sha256": claim[
            "formal_execution_claim_sha256"
        ],
        "formal_execution_terminal_sha256": terminal[
            "formal_execution_terminal_sha256"
        ],
        "gate_contract_sha256": gate["gate_contract_sha256"],
        "generation_execution_count": 0,
        "evaluator_execution_count": 0,
        "heldout_execution_count": 0,
    }
    report["report_only_finalize_sha256"] = _canonical_json_sha256(report)
    _write_immutable_bytes(
        campaign_root / "full/report_only_finalize.json",
        (
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return 0


def _validate_formal_full_execution_chain(
    campaign_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_root = REPO_ROOT / str(campaign_runtime["campaign_root"])
    gate = _require_full_e2e_gate(campaign_runtime)
    claim = _read_json_mapping(
        campaign_root / "formal_execution_claim.json",
        "formal Full execution claim",
    )
    claim_digest = _require_sha256(
        claim.get("formal_execution_claim_sha256"),
        "formal Full execution claim SHA256",
    )
    canonical_claim = dict(claim)
    canonical_claim.pop("formal_execution_claim_sha256")
    admission_binding = _mapping(
        claim.get("full_admission"), "formal Full admission binding"
    )
    admission_path = _repo_path(
        REPO_ROOT, admission_binding.get("path"), "formal Full admission"
    )
    admission = _read_json_mapping(admission_path, "formal Full admission")
    admission_canonical = dict(admission)
    admission_digest = _require_sha256(
        admission_canonical.pop("full_admission_sha256", None),
        "formal Full admission SHA256",
    )
    monitor_claim = _read_json_mapping(
        campaign_root / "formal_monitor/claim.json",
        "formal Full monitor claim",
    )
    monitor_claim_canonical = dict(monitor_claim)
    monitor_claim_digest = _require_sha256(
        monitor_claim_canonical.pop("monitor_claim_sha256", None),
        "formal Full monitor claim SHA256",
    )
    terminal = _read_json_mapping(
        campaign_root / "formal_execution_terminal.json",
        "formal Full execution terminal",
    )
    terminal_canonical = dict(terminal)
    terminal_digest = _require_sha256(
        terminal_canonical.pop("formal_execution_terminal_sha256", None),
        "formal Full execution terminal SHA256",
    )
    summary = _read_json_mapping(
        campaign_root / "formal_monitor/summary.json",
        "formal Full monitor summary",
    )
    summary_canonical = dict(summary)
    summary_digest = _require_sha256(
        summary_canonical.pop("monitor_summary_sha256", None),
        "formal Full monitor summary SHA256",
    )
    admission_fields = {
        "schema_version",
        "contract_type",
        "gpu_indices",
        "gpu_uuids",
        "free_vram_bytes",
        "unknown_compute_pid_count",
        "temperatures_c",
        "ram_used_bytes",
        "ram_total_bytes",
        "disk_used_bytes",
        "disk_total_bytes",
        "swap_in_delta_pages",
        "swap_out_delta_pages",
        "full_admission_sha256",
    }
    if admission.get("unknown_compute_pid_count") != 0:
        admission_fields = admission_fields | {
            "external_compute_pid_baseline",
            "external_compute_pid_policy",
        }
    monitor_claim_fields = {
        "schema_version",
        "contract_type",
        "campaign_id",
        "campaign_runtime_sha256",
        "session_name",
        "command",
        "records",
        "monitor_claim_sha256",
    }
    terminal_fields = {
        "schema_version",
        "contract_type",
        "status",
        "formal_execution_claim_sha256",
        "exit_code",
        "formal_execution_terminal_sha256",
    }
    summary_fields = {
        "schema_version",
        "contract_type",
        "monitor_claim_sha256",
        "formal_execution_terminal_sha256",
        "tmux_session",
        "sample_count",
        "log_progress_count",
        "samples",
        "monitor_summary_sha256",
    }
    sample_binding = _mapping(
        summary.get("samples"), "formal Full monitor samples"
    )
    samples_path = _repo_path(
        REPO_ROOT, sample_binding.get("path"), "formal Full monitor samples"
    )
    if (
        set(claim)
        != {
            "schema_version",
            "contract_type",
            "campaign_runtime_sha256",
            "full_e2e_gate_sha256",
            "full_admission",
            "monitor_claim_sha256",
            "retry_allowed",
            "formal_execution_claim_sha256",
        }
        or claim.get("contract_type")
        != "safa_r9_formal_full_execution_claim_v1"
        or claim.get("campaign_runtime_sha256")
        != campaign_runtime.get("campaign_runtime_sha256")
        or claim.get("full_e2e_gate_sha256")
        != gate["full_e2e_gate_sha256"]
        or claim.get("monitor_claim_sha256") != monitor_claim_digest
        or claim.get("retry_allowed") is not False
        or _canonical_json_sha256(canonical_claim) != claim_digest
        or admission_binding
        != {
            "path": str(admission_path.relative_to(REPO_ROOT)),
            "file_sha256": _sha256_path(admission_path),
            "contract_sha256": admission_digest,
        }
        or _canonical_json_sha256(admission_canonical) != admission_digest
        or set(admission) != admission_fields
        or admission.get("contract_type") != "safa_r9_full_admission_v1"
        or admission.get("gpu_indices") != [0, 1, 2, 3]
        or not isinstance(admission.get("gpu_uuids"), list)
        or len(set(admission["gpu_uuids"])) != 4
        or not _validate_admission_external_pid_policy(admission)
        or any(
            value < 2 * 1024**3
            for value in admission.get("free_vram_bytes", [])
        )
        or len(admission.get("free_vram_bytes", [])) != 4
        or admission["ram_used_bytes"] * 100
        >= admission["ram_total_bytes"] * 85
        or admission["disk_used_bytes"] * 100
        >= admission["disk_total_bytes"] * 85
        or admission.get("swap_in_delta_pages") != 0
        or admission.get("swap_out_delta_pages") != 0
        or any(
            value > 85
            for value in _mapping(
                admission.get("temperatures_c"),
                "formal Full admission temperatures",
            ).values()
        )
        or set(monitor_claim) != monitor_claim_fields
        or _canonical_json_sha256(monitor_claim_canonical)
        != monitor_claim_digest
        or monitor_claim.get("contract_type")
        != "safa_r9_formal_full_monitor_claim_v1"
        or monitor_claim.get("campaign_id")
        != FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        or monitor_claim.get("campaign_runtime_sha256")
        != campaign_runtime.get("campaign_runtime_sha256")
        or monitor_claim.get("session_name")
        != "safa-r9-v9-formal-full-monitor"
        or monitor_claim.get("records")
        != [
            "gpu",
            "cpu",
            "ram",
            "disk",
            "log_byte_progress",
            "png_count",
            "result_count",
        ]
        or not isinstance(monitor_claim.get("command"), list)
        or monitor_claim["command"][-3:]
        != ["--phase", "formal-monitor", "--execute"]
        or set(terminal) != terminal_fields
        or terminal.get("contract_type")
        != "safa_r9_formal_full_execution_terminal_v1"
        or terminal.get("status") != "awaiting_visual_review"
        or terminal.get("formal_execution_claim_sha256") != claim_digest
        or terminal.get("exit_code") != AWAITING_VISUAL_REVIEW_EXIT_CODE
        or _canonical_json_sha256(terminal_canonical) != terminal_digest
        or set(summary) != summary_fields
        or summary.get("contract_type")
        != "safa_r9_formal_full_monitor_summary_v1"
        or summary.get("monitor_claim_sha256") != monitor_claim_digest
        or summary.get("formal_execution_terminal_sha256")
        != terminal_digest
        or not isinstance(summary.get("tmux_session"), str)
        or not summary["tmux_session"]
        or not isinstance(summary.get("sample_count"), int)
        or summary["sample_count"] <= 0
        or not isinstance(summary.get("log_progress_count"), int)
        or summary["log_progress_count"] < 0
        or sample_binding
        != {
            "path": str(samples_path.relative_to(REPO_ROOT)),
            "file_sha256": _sha256_path(samples_path),
        }
        or _canonical_json_sha256(summary_canonical) != summary_digest
    ):
        raise ValueError("formal Full execution/monitor chain changed")
    return {
        "claim": claim,
        "admission": admission,
        "monitor_claim": monitor_claim,
        "terminal": terminal,
        "monitor_summary": summary,
    }


def _materialize_formal_full_monitor_claim(
    campaign_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_root = REPO_ROOT / str(campaign_runtime["campaign_root"])
    session = "safa-r9-v9-formal-full-monitor"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/prepare_r9_full_continuation.py"),
        "--phase",
        "formal-monitor",
        "--execute",
    ]
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_formal_full_monitor_claim_v1",
        "campaign_id": FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        "campaign_runtime_sha256": campaign_runtime["campaign_runtime_sha256"],
        "session_name": session,
        "command": command,
        "records": [
            "gpu",
            "cpu",
            "ram",
            "disk",
            "log_byte_progress",
            "png_count",
            "result_count",
        ],
    }
    claim["monitor_claim_sha256"] = _canonical_json_sha256(claim)
    path = campaign_root / "formal_monitor/claim.json"
    _write_exclusive_bytes(
        path,
        (
            json.dumps(
                claim,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return claim


def _start_formal_full_monitor(
    campaign_runtime: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> None:
    session = str(claim["session_name"])
    campaign_root = REPO_ROOT / str(campaign_runtime["campaign_root"])
    samples_path = campaign_root / "formal_monitor/session_samples.jsonl"
    log_path = campaign_root / "formal_monitor/operator.log"
    if subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        capture_output=True,
    ).returncode == 0:
        raise RuntimeError("formal Full monitor tmux session already exists")
    monitor_command = " ".join(shlex.quote(str(value)) for value in claim["command"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "bash",
            "-lc",
            f"exec >> {shlex.quote(str(log_path))} 2>&1\nexec {monitor_command}",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        raise RuntimeError(
            f"formal Full monitor tmux start failed: {started.stderr.strip()}"
        )
    def monitor_log_tail() -> str:
        if not log_path.is_file():
            return ""
        return log_path.read_text(encoding="utf-8", errors="replace")[-4000:]

    if subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        capture_output=True,
    ).returncode != 0:
        log_tail = monitor_log_tail()
        raise RuntimeError(
            "formal Full monitor exited before first sample"
            + (f": {log_tail}" if log_tail else "")
        )
    deadline = time.monotonic() + 15
    while not samples_path.is_file():
        if subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=False,
            capture_output=True,
        ).returncode != 0:
            log_tail = monitor_log_tail()
            raise RuntimeError(
                "formal Full monitor exited before first sample"
                + (f": {log_tail}" if log_tail else "")
            )
        if time.monotonic() >= deadline:
            raise RuntimeError("formal Full monitor did not publish its first sample")
        time.sleep(0.2)


def _materialize_formal_full_terminal(
    campaign_runtime: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    status: str,
    exit_code: int | None = None,
) -> dict[str, Any]:
    terminal = {
        "schema_version": 1,
        "contract_type": "safa_r9_formal_full_execution_terminal_v1",
        "status": status,
        "formal_execution_claim_sha256": claim[
            "formal_execution_claim_sha256"
        ],
    }
    if exit_code is not None:
        terminal["exit_code"] = int(exit_code)
    terminal["formal_execution_terminal_sha256"] = _canonical_json_sha256(
        terminal
    )
    _write_exclusive_bytes(
        REPO_ROOT
        / str(campaign_runtime["campaign_root"])
        / "formal_execution_terminal.json",
        (
            json.dumps(
                terminal,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return terminal


def _wait_formal_full_monitor(
    campaign_runtime: Mapping[str, Any],
) -> None:
    summary = (
        REPO_ROOT
        / str(campaign_runtime["campaign_root"])
        / "formal_monitor/summary.json"
    )
    deadline = time.monotonic() + 30
    while not summary.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("formal Full monitor did not finalize")
        time.sleep(0.5)


def _require_generation_batch_benchmark_before_confirm(
    *, is_confirm_continuation: bool, campaign_runtime: Mapping[str, Any]
) -> None:
    if is_confirm_continuation and campaign_runtime.get(
        "generation_batch_benchmark"
    ) is None:
        raise RuntimeError(
            "confirm512 is blocked until generation_batch_benchmark.json is materialized"
        )


def _require_full_e2e_gate(
    campaign_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is None or continuation.get("start_phase") != "full":
        raise ValueError("Full E2E gate requires a Full continuation")
    selection = _require_full_selection_binding(continuation, campaign_runtime)
    current_evaluation = _mapping(
        _mapping(continuation.get("bindings"), "Full bindings").get(
            "current_evaluation"
        ),
        "current Full evaluation",
    )
    runtime_evaluation = _mapping(
        campaign_runtime.get("evaluation"), "Full runtime evaluation"
    )
    if (
        current_evaluation.get("classification")
        != "canonical_current_v9_execution_authority"
        or runtime_evaluation.get("worker")
        != current_evaluation.get("worker")
        or _mapping(
            runtime_evaluation.get("quality"), "Full runtime quality"
        ).get("script")
        != current_evaluation.get("quality_script")
    ):
        raise ValueError("Full E2E runtime is not authorized by current evaluator")
    manifests = _mapping(campaign_runtime.get("manifests"), "campaign manifests")
    full_manifest = _mapping(manifests.get("full_2048"), "full_2048")
    gate_path = (
        REPO_ROOT
        / str(campaign_runtime["campaign_root"])
        / "full_e2e"
        / "gate_contract.json"
    )
    gate = _read_json_mapping(gate_path, "Full E2E gate")
    expected_fields = {
        "schema_version",
        "contract_type",
        "campaign_id",
        "continuation_contract_sha256",
        "selection_sha256",
        "manifest",
        "generation_policy",
        "generation_results",
        "evaluator_results",
        "resource_profiles",
        "full_e2e_result_sha256",
        "verdict",
        "full_e2e_gate_sha256",
    }
    if set(gate) != expected_fields:
        raise ValueError("Full E2E gate fields are not canonical")
    declared = _require_sha256(
        gate.get("full_e2e_gate_sha256"), "Full E2E gate SHA256"
    )
    canonical = dict(gate)
    canonical.pop("full_e2e_gate_sha256")
    if _canonical_json_sha256(canonical) != declared:
        raise ValueError("Full E2E gate digest mismatch")
    manifest = _mapping(gate.get("manifest"), "Full E2E manifest")
    policy = _mapping(gate.get("generation_policy"), "Full E2E generation policy")
    results = gate.get("generation_results")
    evaluators = _mapping(gate.get("evaluator_results"), "Full E2E evaluators")
    _require_sha256(
        gate.get("full_e2e_result_sha256"), "Full E2E result SHA256"
    )
    if (
        gate.get("schema_version") != 1
        or gate.get("contract_type") != "safa_r9_full_e2e_gate_v1"
        or gate.get("campaign_id") != FULL_CONTINUATION_CHILD_CAMPAIGN_ID
        or gate.get("continuation_contract_sha256")
        != _continuation_digest(continuation)
        or gate.get("selection_sha256") != selection["selection_sha256"]
        or manifest.get("path")
        != "configs/medium_v2/experiments/r9_manifests/full_smoke_8.jsonl"
        or manifest.get("sha256")
        != "04a7d89db541b065755c965505bb26b1e58aea306cc59c1717f251ec32dfc87f"
        or manifest.get("sample_count") != 8
        or manifest.get("parent_path") != full_manifest.get("path")
        or manifest.get("parent_sha256") != full_manifest.get("sha256")
        or policy
        != {
            "phase": "full",
            "seed": 7919,
            "batch_size": 2,
            "arms": ["native", "paper_eta_0p125"],
            "retry_count": 0,
        }
        or not isinstance(results, list)
        or [row.get("arm_id") for row in results if isinstance(row, Mapping)]
        != ["native", "paper_eta_0p125"]
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "arm_id",
                "runtime_config_sha256",
                "generation_result_sha256",
                "per_sample_sha256",
            }
            for row in results
        )
        or set(evaluators)
        != {"arcface", "quality_native", "quality_candidate"}
        or any(
            set(_mapping(evaluators[name], name))
            != {"request_sha256", "result_sha256"}
            for name in ("arcface", "quality_native", "quality_candidate")
        )
        or gate.get("resource_profiles")
        != _mapping(
            runtime_evaluation.get("resource_smokes"),
            "Full runtime resource profiles",
        ).get("resource_profile_binding")
        or gate.get("verdict") != "pass"
    ):
        raise ValueError("Full E2E gate does not authorize formal Full execution")
    rebuilt = _rebuild_full_e2e_evidence(campaign_runtime)
    if gate != rebuilt["gate"]:
        raise ValueError("Full E2E gate does not match current evidence bytes")
    return gate


def _rebuild_full_e2e_evidence(
    campaign_runtime: Mapping[str, Any],
    *,
    require_materialized_result: bool = True,
) -> dict[str, Any]:
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is None:
        raise ValueError("Full E2E evidence has no continuation")
    selection = _require_full_selection_binding(continuation, campaign_runtime)
    current_evaluation = _mapping(
        _mapping(continuation.get("bindings"), "Full bindings").get(
            "current_evaluation"
        ),
        "current Full evaluation",
    )
    runtime_evaluation = _mapping(
        campaign_runtime.get("evaluation"), "Full runtime evaluation"
    )
    if (
        current_evaluation.get("classification")
        != "canonical_current_v9_execution_authority"
        or runtime_evaluation.get("worker")
        != current_evaluation.get("worker")
        or _mapping(
            runtime_evaluation.get("quality"), "Full runtime quality"
        ).get("script")
        != current_evaluation.get("quality_script")
    ):
        raise ValueError("Full E2E runtime is not authorized by current evaluator")
    root = (
        REPO_ROOT
        / str(campaign_runtime["campaign_root"])
        / "full_e2e"
    )
    plan = _read_json_mapping(root / "plan.json", "Full E2E plan")
    expected_plan_fields = {
        "schema_version",
        "contract_type",
        "campaign_id",
        "continuation_contract_sha256",
        "selection_sha256",
        "request_config",
        "e2e_request",
        "generation_batch_benchmark",
        "provisional_runtime",
        "manifest",
        "generation_policy",
        "runs",
        "full_e2e_plan_sha256",
    }
    if (
        set(plan) != expected_plan_fields
        or plan.get("schema_version") != 1
        or plan.get("contract_type") != "safa_r9_full_e2e_plan_v1"
        or plan.get("campaign_id") != FULL_CONTINUATION_CHILD_CAMPAIGN_ID
    ):
        raise ValueError("Full E2E plan fields changed")
    declared_plan = _require_sha256(
        plan.get("full_e2e_plan_sha256"), "Full E2E plan SHA256"
    )
    canonical_plan = dict(plan)
    canonical_plan.pop("full_e2e_plan_sha256")
    if _canonical_json_sha256(canonical_plan) != declared_plan:
        raise ValueError("Full E2E plan digest mismatch")
    request_config = _mapping(plan.get("request_config"), "E2E request config")
    if (
        request_config
        != {
            "path": str(FULL_CONTINUATION_RUNTIME_CONFIG),
            "sha256": _sha256_path(REPO_ROOT / FULL_CONTINUATION_RUNTIME_CONFIG),
        }
        or plan.get("e2e_request")
        != continuation["bindings"]["full_e2e_requirement"]["request"]
        or plan.get("generation_batch_benchmark")
        != campaign_runtime["generation_batch_benchmark"]
        or plan.get("continuation_contract_sha256")
        != _continuation_digest(continuation)
        or plan.get("selection_sha256") != selection["selection_sha256"]
    ):
        raise ValueError("Full E2E plan source binding changed")
    provisional_binding = _mapping(
        plan.get("provisional_runtime"), "Full E2E provisional runtime"
    )
    provisional_path = _repo_path(
        REPO_ROOT,
        provisional_binding.get("path"),
        "Full E2E provisional runtime",
    )
    observed_provisional = _read_json_mapping(
        provisional_path, "Full E2E provisional runtime"
    )
    if (
        set(provisional_binding) != {"path", "contract_sha256"}
        or provisional_binding.get("contract_sha256")
        != observed_provisional.get("campaign_runtime_sha256")
        or _canonical_json_sha256(
            {
                key: value
                for key, value in observed_provisional.items()
                if key != "campaign_runtime_sha256"
            }
        )
        != observed_provisional.get("campaign_runtime_sha256")
        or observed_provisional.get("continuation")
        != campaign_runtime.get("continuation")
        or observed_provisional.get("checkpoint")
        != campaign_runtime.get("checkpoint")
        or observed_provisional.get("manifests")
        != campaign_runtime.get("manifests")
        or observed_provisional.get("generation_batch_benchmark")
        != campaign_runtime.get("generation_batch_benchmark")
        or _mapping(
            observed_provisional.get("evaluation"),
            "Full E2E provisional evaluation",
        ).get("worker")
        != runtime_evaluation.get("worker")
    ):
        raise ValueError("Full E2E provisional runtime binding changed")
    manifest = _mapping(plan.get("manifest"), "Full E2E manifest")
    parent_manifest = _mapping(
        _mapping(campaign_runtime.get("manifests"), "manifests").get("full_2048"),
        "full_2048",
    )
    manifest_path = _repo_path(
        REPO_ROOT, manifest.get("path"), "Full E2E manifest"
    )
    if (
        set(manifest)
        != {
            "path",
            "sha256",
            "sample_count",
            "parent_path",
            "parent_sha256",
            "sample_ids",
        }
        or manifest.get("sha256") != _sha256_path(manifest_path)
        or manifest.get("sample_count") != 8
        or manifest.get("parent_path") != parent_manifest["path"]
        or manifest.get("parent_sha256") != parent_manifest["sha256"]
        or plan.get("generation_policy")
        != {
            "phase": "full",
            "seed": 7919,
            "batch_size": 2,
            "arms": ["native", "paper_eta_0p125"],
            "retry_count": 0,
        }
    ):
        raise ValueError("Full E2E plan manifest/policy changed")
    sample_ids = [
        _mapping(json.loads(line), "Full E2E manifest row")["sample_id"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    if sample_ids != manifest["sample_ids"] or len(set(sample_ids)) != 8:
        raise ValueError("Full E2E ordered sample IDs changed")
    runs = plan.get("runs")
    if not isinstance(runs, list) or [
        row.get("arm_id") for row in runs if isinstance(row, Mapping)
    ] != ["native", "paper_eta_0p125"]:
        raise ValueError("Full E2E run order changed")
    generation_results = []
    per_sample_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for row in runs:
        run_row = _mapping(row, "Full E2E run")
        arm_id = str(run_row["arm_id"])
        config_path = _repo_path(
            REPO_ROOT, run_row.get("runtime_config"), "Full E2E runtime config"
        )
        if _sha256_path(config_path) != run_row.get("runtime_config_sha256"):
            raise ValueError("Full E2E runtime config SHA256 changed")
        config = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "Full E2E runtime config",
        )
        expected_output = (
            Path(str(campaign_runtime["campaign_root"]))
            / "full_e2e"
            / "generation"
            / arm_id
        )
        if (
            config.get("phase") != "full"
            or config.get("seed") != 7919
            or config.get("sampling_seed") != 7919
            or config.get("batch_size") != 2
            or config.get("max_samples") != 8
            or config.get("sample_id_manifest") != manifest["path"]
            or config.get("sample_id_manifest_sha256") != manifest["sha256"]
            or config.get("r9_full_e2e_role") != "formal_gate_v1"
            or run_row.get("output_dir") != str(expected_output)
        ):
            raise ValueError("Full E2E generation runtime semantics changed")
        output = REPO_ROOT / expected_output
        run = RunSpec(
            phase="full",
            logical_run_id=f"formal_e2e_{arm_id}_8",
            arm_ref=arm_id,
            seed=7919,
            repeat_index=None,
            shard_index=0,
            num_shards=1,
            sample_count=8,
            manifest_key="full_2048",
            runtime_config=Path(str(run_row["runtime_config"])),
            output_dir=expected_output,
            command=tuple(run_row["command"]),
        )
        validate_worker_completion(run)
        rows = _read_ordered_per_sample(output / "per_sample.jsonl")
        if list(rows) != sample_ids:
            raise ValueError("Full E2E per-sample order changed")
        per_sample_by_arm[arm_id] = rows
        generation_results.append(
            {
                "arm_id": arm_id,
                "runtime_config_sha256": run_row["runtime_config_sha256"],
                "generation_result_sha256": _sha256_path(
                    output / "generation_result.json"
                ),
                "per_sample_sha256": _sha256_path(output / "per_sample.jsonl"),
            }
        )
    samples = []
    for sample_id in sample_ids:
        native_row = per_sample_by_arm["native"][sample_id]
        winner_row = per_sample_by_arm["paper_eta_0p125"][sample_id]
        if native_row.get("source") != winner_row.get("source"):
            raise ValueError("Full E2E matched source changed")
        source = Path(str(native_row["source"])).resolve()
        native = _repo_path(
            REPO_ROOT, native_row.get("native"), "Full E2E native image"
        )
        candidate = _repo_path(
            REPO_ROOT, winner_row.get("generated"), "Full E2E winner image"
        )
        samples.append(
            SampleEvidence(
                sample_id=sample_id,
                source=source,
                native=native,
                candidate=candidate,
                source_sha256=_sha256_path(source),
                native_sha256=_sha256_path(native),
                candidate_sha256=_sha256_path(candidate),
            )
        )
    expected_payloads = _expected_full_e2e_payloads(
        campaign_runtime, selection, manifest_path, tuple(samples), generation_results
    )
    evaluator_results = {}
    evaluator_units = {
        "arcface": ("arcface", "formal_e2e_arcface_8"),
        "quality_native": ("quality", "formal_e2e_quality_8__native"),
        "quality_candidate": ("quality", "formal_e2e_quality_8__candidate"),
    }
    for evidence_key, (task, unit) in evaluator_units.items():
        unit_root = root / "evaluator_runs" / task / unit
        request = _read_json_mapping(
            unit_root / "request.json", f"E2E {evidence_key} request"
        )
        output = _read_json_mapping(
            unit_root / "result.json", f"E2E {evidence_key} result"
        )
        request_sha = _require_sha256(
            request.get("evaluator_request_sha256"), f"E2E {task} request SHA256"
        )
        request_canonical = dict(request)
        request_canonical.pop("evaluator_request_sha256")
        output_sha = _require_sha256(
            output.get("evaluator_output_sha256"), f"E2E {task} result SHA256"
        )
        output_canonical = dict(output)
        output_canonical.pop("evaluator_output_sha256")
        evaluation = _mapping(campaign_runtime.get("evaluation"), "evaluation")
        expected_config = {
            "repo_root": str(REPO_ROOT.resolve()),
            "device": "cuda:0",
            "work_root": str((unit_root / "work").resolve()),
            "batch_size": int(
                _mapping(evaluation.get("heldout"), "heldout")["batch_size"]
            ),
            "arcface": _normalized_arcface_evaluation_contract(evaluation),
            "quality_script": dict(
                _mapping(
                    _mapping(evaluation.get("quality"), "quality").get("script"),
                    "quality script",
                )
            ),
            "worker_contract": _normalized_worker_contract(evaluation),
        }
        if (
            request.get("contract_type") != "safa_r9_phase_evaluator_request_v1"
            or request.get("task") != task
            or request.get("config") != expected_config
            or request.get("payload") != expected_payloads[evidence_key]
            or _canonical_json_sha256(request_canonical) != request_sha
            or output.get("contract_type") != "safa_r9_phase_evaluator_output_v1"
            or output.get("task") != task
            or output.get("evaluator_request_sha256") != request_sha
            or _canonical_json_sha256(output_canonical) != output_sha
        ):
            raise ValueError(
                f"Full E2E {evidence_key} request/result binding changed"
            )
        _validate_full_e2e_result_semantics(
            task, output.get("result"), sample_ids
        )
        evaluator_results[evidence_key] = {
            "request_sha256": request_sha,
            "result_sha256": output_sha,
        }
    resource_profiles = build_full_e2e_resource_profiles(campaign_runtime)
    observed_profiles = _read_json_mapping(
        root / "resource_profiles.json", "Full E2E resource profiles"
    )
    if observed_profiles != resource_profiles:
        raise ValueError("Full E2E resource profiles changed")
    resource_profile_binding = {
        "path": str(
            (root / "resource_profiles.json").relative_to(REPO_ROOT)
        ),
        "file_sha256": _sha256_path(root / "resource_profiles.json"),
        "contract_sha256": resource_profiles["resource_profiles_sha256"],
    }
    result = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_result_v1",
        "plan_sha256": declared_plan,
        "generation_results": generation_results,
        "evaluator_results": evaluator_results,
        "resource_profiles": resource_profile_binding,
    }
    result["full_e2e_result_sha256"] = _canonical_json_sha256(result)
    if require_materialized_result:
        observed_result = _read_json_mapping(
            root / "run_result.json", "Full E2E result"
        )
        if observed_result != result:
            raise ValueError("Full E2E run result does not match current evidence")
    gate = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_gate_v1",
        "campaign_id": FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        "continuation_contract_sha256": _continuation_digest(continuation),
        "selection_sha256": selection["selection_sha256"],
        "manifest": {
            key: manifest[key]
            for key in (
                "path",
                "sha256",
                "sample_count",
                "parent_path",
                "parent_sha256",
            )
        },
        "generation_policy": plan["generation_policy"],
        "generation_results": generation_results,
        "evaluator_results": evaluator_results,
        "resource_profiles": resource_profile_binding,
        "full_e2e_result_sha256": result["full_e2e_result_sha256"],
        "verdict": "pass",
    }
    gate["full_e2e_gate_sha256"] = _canonical_json_sha256(gate)
    return {"plan": plan, "result": result, "gate": gate}


def _normalized_worker_contract(
    evaluation: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, str]:
    root = REPO_ROOT if repo_root is None else repo_root
    worker = _mapping(evaluation.get("worker"), "evaluation worker")
    worker_script = _repo_path(root, worker.get("path"), "evaluation worker")
    worker_implementation = _repo_path(
        root,
        worker.get("implementation_path"),
        "evaluation worker implementation",
    )
    return {
        "path": str(worker_script.resolve()),
        "sha256": str(worker.get("sha256")),
        "implementation_path": str(worker_implementation.resolve()),
        "implementation_sha256": str(worker.get("implementation_sha256")),
    }


def _normalized_arcface_evaluation_contract(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    from safa.evaluation.r9_evaluator_worker import _validate_arcface_contract

    return _validate_arcface_contract(
        _mapping(evaluation.get("arcface"), "ArcFace evaluation"),
        repo_root=REPO_ROOT,
    )


def _arcface_evaluation_contract_sha256(
    evaluation: Mapping[str, Any],
) -> str:
    return _canonical_json_sha256(
        _normalized_arcface_evaluation_contract(evaluation)
    )


def build_full_e2e_resource_profiles(
    campaign_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    root = (
        REPO_ROOT
        / str(campaign_runtime["campaign_root"])
        / "full_e2e"
    )
    evaluation = _mapping(campaign_runtime.get("evaluation"), "evaluation")
    expected_worker = _normalized_worker_contract(evaluation)
    expected_arcface = _arcface_evaluation_contract_sha256(evaluation)
    expected_quality = _mapping(
        _mapping(evaluation.get("quality"), "quality evaluation").get("script"),
        "quality script",
    )["sha256"]
    units = {
        "arcface": ("arcface", "formal_e2e_arcface_8"),
        "quality_native": ("quality", "formal_e2e_quality_8__native"),
        "quality_candidate": ("quality", "formal_e2e_quality_8__candidate"),
    }
    observations = {}
    for evidence_key, (task, unit) in units.items():
        unit_root = root / "evaluator_runs" / task / unit
        request_path = unit_root / "request.json"
        result_path = unit_root / "result.json"
        observation_path = unit_root / "resource_observation.json"
        request = _read_json_mapping(request_path, f"{evidence_key} request")
        result = _read_json_mapping(result_path, f"{evidence_key} result")
        observation = _read_json_mapping(
            observation_path, f"{evidence_key} resource observation"
        )
        observation_canonical = dict(observation)
        observation_sha = _require_sha256(
            observation_canonical.pop("resource_observation_sha256", None),
            f"{evidence_key} resource observation SHA256",
        )
        if (
            observation.get("contract_type")
            != "safa_r9_full_e2e_evaluator_resource_observation_v1"
            or observation.get("task") != task
            or observation.get("unit_id") != unit
            or observation.get("evaluator_request_sha256")
            != request.get("evaluator_request_sha256")
            or observation.get("evaluator_output_sha256")
            != result.get("evaluator_output_sha256")
            or observation.get("worker_contract") != expected_worker
            or observation.get("arcface_contract_sha256") != expected_arcface
            or observation.get("quality_script_sha256") != expected_quality
            or observation.get("resource_policy_id")
            != "frozen_conservative_e2e_v1"
            or _canonical_json_sha256(observation_canonical) != observation_sha
        ):
            raise ValueError(
                f"Full E2E {evidence_key} resource observation changed"
            )
        peak_rss = _positive_int(
            observation.get("peak_process_tree_rss_bytes"),
            f"{evidence_key} peak RSS",
        )
        peak_gpu = _positive_int(
            observation.get("peak_gpu_memory_bytes"),
            f"{evidence_key} peak GPU memory",
        )
        observations[evidence_key] = {
            "request": {
                "path": str(request_path.relative_to(REPO_ROOT)),
                "file_sha256": _sha256_path(request_path),
                "contract_sha256": request["evaluator_request_sha256"],
            },
            "result": {
                "path": str(result_path.relative_to(REPO_ROOT)),
                "file_sha256": _sha256_path(result_path),
                "contract_sha256": result["evaluator_output_sha256"],
            },
            "observation": {
                "path": str(observation_path.relative_to(REPO_ROOT)),
                "file_sha256": _sha256_path(observation_path),
                "contract_sha256": observation_sha,
            },
            "peak_process_tree_rss_bytes": peak_rss,
            "peak_gpu_memory_bytes": peak_gpu,
            "gpu_uuid": str(observation["gpu_uuid"]),
        }
    arcface_peak = observations["arcface"]["peak_process_tree_rss_bytes"]
    quality_peak = max(
        observations[key]["peak_process_tree_rss_bytes"]
        for key in ("quality_native", "quality_candidate")
    )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_resource_profiles_v1",
        "campaign_id": FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        "source": "measured_from_successful_formal_e2e_workers",
        "worker_contract": dict(expected_worker),
        "arcface_contract_sha256": expected_arcface,
        "quality_script_sha256": expected_quality,
        "arcface": {
            "mode": "measured_single_worker",
            "evidence": [observations["arcface"]],
            "peak_process_tree_rss_bytes": arcface_peak,
            "peak_gpu_memory_bytes": observations["arcface"][
                "peak_gpu_memory_bytes"
            ],
            "ram_slot_budget_bytes": (arcface_peak * 110 + 99) // 100,
        },
        "quality": {
            "mode": "measured_exclusive_bootstrap",
            "evidence": [
                observations["quality_native"],
                observations["quality_candidate"],
            ],
            "peak_process_tree_rss_bytes": quality_peak,
            "peak_gpu_memory_bytes": max(
                observations[key]["peak_gpu_memory_bytes"]
                for key in ("quality_native", "quality_candidate")
            ),
            "ram_slot_budget_bytes": (quality_peak * 110 + 99) // 100,
        },
        "heldout": {
            "mode": "exclusive_single_official_run",
            "smoke_execution": "sealed_until_winner_lock",
            "global_exclusive_slots": 16,
            "ram_admission_percent": 85,
            "ram_hard_limit_percent": 90,
        },
    }
    payload["resource_profiles_sha256"] = _canonical_json_sha256(payload)
    return payload


def materialize_full_e2e_resource_profiles(
    campaign_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    profiles = build_full_e2e_resource_profiles(campaign_runtime)
    destination = (
        REPO_ROOT
        / str(campaign_runtime["campaign_root"])
        / "full_e2e/resource_profiles.json"
    )
    _write_exclusive_bytes(
        destination,
        (
            json.dumps(
                profiles,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return profiles


def _read_ordered_per_sample(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = _mapping(json.loads(line), "Full E2E per-sample row")
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError("Full E2E duplicate sample ID")
        rows[sample_id] = row
    return rows


def _expected_full_e2e_payloads(
    campaign_runtime: Mapping[str, Any],
    selection: Mapping[str, Any],
    manifest_path: Path,
    samples: tuple[SampleEvidence, ...],
    generation_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    source_index = _mapping(
        _mapping(
            _mapping(campaign_runtime.get("evaluation"), "evaluation").get("quality"),
            "quality",
        ).get("real_index"),
        "real index",
    )
    serialized = _serialize_evaluator_samples(samples)
    common = {
        "phase": "full_e2e",
        "seed": 7919,
        "source_index_path": str(
            _repo_path(REPO_ROOT, source_index["path"], "source index").resolve()
        ),
        "source_index_sha256": source_index["sha256"],
        "samples": serialized,
    }
    winner = _mapping(selection.get("winner"), "Full winner")
    native_config_path = next(
        _repo_path(
            REPO_ROOT,
            row["runtime_config"],
            "Full E2E native runtime config",
        )
        for row in _read_json_mapping(
            REPO_ROOT
            / str(campaign_runtime["campaign_root"])
            / "full_e2e/plan.json",
            "Full E2E plan",
        )["runs"]
        if row["arm_id"] == "native"
    )
    native_config = _mapping(
        yaml.safe_load(native_config_path.read_text(encoding="utf-8")),
        "Full E2E native runtime config",
    )
    evidence_binding = _canonical_json_sha256(
        [
            {
                "sample_id": row.sample_id,
                "source": row.source_sha256,
                "native": row.native_sha256,
                "candidate": row.candidate_sha256,
            }
            for row in samples
        ]
    )
    generation_set = _canonical_json_sha256(
        [row["generation_result_sha256"] for row in generation_results]
    )
    per_sample_set = _canonical_json_sha256(
        [row["per_sample_sha256"] for row in generation_results]
    )
    return {
        "arcface": {
            **common,
            "arm_id": "paper_eta_0p125",
            "logical_run_id": "formal_e2e_arcface_8",
        },
        "quality_native": {
            **common,
            "logical_run_id": "formal_e2e_quality_8",
            "arm_id": "native",
            "image_role": "native",
            "manifest_path": str(manifest_path.resolve()),
            "algorithm_config_sha256": native_config["arm_config_sha256"],
            "runner_arm_config_sha256": native_config["arm_config_sha256"],
            "semantic_output_sha256": _canonical_json_sha256(
                [
                    {
                        "sample_id": row.sample_id,
                        "sha256": row.native_sha256,
                    }
                    for row in samples
                ]
            ),
            "evidence_binding_sha256": evidence_binding,
            "generation_result_set_sha256": generation_set,
            "per_sample_set_sha256": per_sample_set,
        },
        "quality_candidate": {
            **common,
            "logical_run_id": "formal_e2e_quality_8",
            "arm_id": "paper_eta_0p125",
            "image_role": "candidate",
            "manifest_path": str(manifest_path.resolve()),
            "algorithm_config_sha256": winner["config_sha256"],
            "runner_arm_config_sha256": winner["config_sha256"],
            "semantic_output_sha256": _canonical_json_sha256(
                [
                    {
                        "sample_id": row.sample_id,
                        "sha256": row.candidate_sha256,
                    }
                    for row in samples
                ]
            ),
            "evidence_binding_sha256": evidence_binding,
            "generation_result_set_sha256": generation_set,
            "per_sample_set_sha256": per_sample_set,
        },
    }


def _validate_full_e2e_result_semantics(
    task: str, result: Any, sample_ids: Sequence[str]
) -> None:
    if task == "arcface":
        if (
            not isinstance(result, list)
            or [row.get("sample_id") for row in result if isinstance(row, Mapping)]
            != list(sample_ids)
            or any(
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "sample_id",
                    "source_face_count",
                    "native_face_count",
                    "candidate_face_count",
                    "source_native_cosine",
                    "source_candidate_cosine",
                }
                for row in result
            )
        ):
            raise ValueError("Full E2E ArcFace result coverage changed")
        return
    quality = _mapping(result, "Full E2E quality result")
    per_sample = _mapping(
        quality.get("per_sample_metrics"), "Full E2E quality per-sample"
    )
    rows = per_sample.get("rows")
    if (
        quality.get("num_generated") != 8
        or quality.get("metrics") != ["fid", "kid", "niqe", "sharpness"]
        or not isinstance(rows, list)
        or [row.get("sample_id") for row in rows if isinstance(row, Mapping)]
        != list(sample_ids)
    ):
        raise ValueError("Full E2E quality result coverage changed")


FULL_ADMISSION_EXTERNAL_PID_BASELINE_ENV = (
    "SAFA_R9_ALLOW_PREEXISTING_EXTERNAL_GPU_PIDS"
)


def _full_admission_preflight(
    *,
    resource_probe: Any | None = None,
    compute_apps: Sequence[tuple[str, int]] | None = None,
    temperatures: Mapping[str, int] | None = None,
    disk_usage: Any | None = None,
    swap_io_delta: tuple[int, int] | None = None,
) -> dict[str, Any]:
    probe = SystemResourceProbe() if resource_probe is None else resource_probe
    ram = probe.ram_snapshot()
    if ram.used_bytes * 100 >= ram.total_bytes * 85:
        raise ResourceContractError("Full admission requires RAM below 85%")
    snapshots = tuple(
        snapshot for snapshot in probe.gpu_snapshots() if snapshot.index in {0, 1, 2, 3}
    )
    if [snapshot.index for snapshot in snapshots] != [0, 1, 2, 3]:
        raise ResourceContractError("Full admission requires exactly GPU0-3")
    if any(snapshot.free_bytes < 2 * 1024**3 for snapshot in snapshots):
        raise ResourceContractError(
            "Full admission requires at least 2 GiB free on every GPU"
        )
    apps = (
        tuple(compute_apps)
        if compute_apps is not None
        else _query_gpu_compute_apps()
    )
    selected_uuids = {snapshot.uuid for snapshot in snapshots}
    unknown = sorted(
        (uuid, pid) for uuid, pid in apps if uuid in selected_uuids
    )
    external_pid_baseline: list[dict[str, Any]] = []
    if unknown:
        if os.environ.get(FULL_ADMISSION_EXTERNAL_PID_BASELINE_ENV) != "1":
            raise ResourceContractError(
                "Full admission found unknown GPU compute PIDs: "
                + ",".join(f"{uuid}:{pid}" for uuid, pid in unknown)
            )
        external_pid_baseline = [
            _gpu_pid_baseline_row(uuid, pid) for uuid, pid in unknown
        ]
    all_temperatures = (
        dict(temperatures)
        if temperatures is not None
        else _query_gpu_temperatures()
    )
    observed_temperatures = {
        uuid: all_temperatures[uuid]
        for uuid in selected_uuids
        if uuid in all_temperatures
    }
    if set(observed_temperatures) != selected_uuids:
        raise ResourceContractError("Full admission GPU temperature set changed")
    if any(value > 85 for value in observed_temperatures.values()):
        raise ResourceContractError("Full admission found GPU temperature above 85C")
    usage = shutil.disk_usage(REPO_ROOT) if disk_usage is None else disk_usage
    if usage.used * 100 >= usage.total * 85:
        raise ResourceContractError("Full admission requires disk usage below 85%")
    swap_delta = (
        _sample_swap_io_delta()
        if swap_io_delta is None
        else tuple(swap_io_delta)
    )
    if swap_delta != (0, 0):
        raise ResourceContractError("Full admission requires zero swap I/O")
    evidence = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_admission_v1",
        "gpu_indices": [snapshot.index for snapshot in snapshots],
        "gpu_uuids": [snapshot.uuid for snapshot in snapshots],
        "free_vram_bytes": [snapshot.free_bytes for snapshot in snapshots],
        "unknown_compute_pid_count": 0,
        "temperatures_c": {
            uuid: observed_temperatures[uuid] for uuid in sorted(selected_uuids)
        },
        "ram_used_bytes": ram.used_bytes,
        "ram_total_bytes": ram.total_bytes,
        "disk_used_bytes": usage.used,
        "disk_total_bytes": usage.total,
        "swap_in_delta_pages": 0,
        "swap_out_delta_pages": 0,
    }
    if external_pid_baseline:
        evidence["unknown_compute_pid_count"] = len(external_pid_baseline)
        evidence["external_compute_pid_baseline"] = external_pid_baseline
        evidence["external_compute_pid_policy"] = {
            "schema_version": 1,
            "mode": "user_authorized_preexisting_gpu_pid_baseline_v1",
            "authorization_env": FULL_ADMISSION_EXTERNAL_PID_BASELINE_ENV,
            "new_unknown_gpu_pids": "forbidden_after_admission",
            "resource_hard_stops": "unchanged",
        }
    evidence["full_admission_sha256"] = _canonical_json_sha256(evidence)
    return evidence


def _gpu_pid_baseline_row(gpu_uuid: str, pid: int) -> dict[str, Any]:
    start_ticks = _pid_start_time_ticks(pid)
    completed = subprocess.run(
        ["ps", "-o", "user=", "-o", "args=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ResourceContractError("Full admission GPU PID process disappeared")
    fields = completed.stdout.strip().split(maxsplit=1)
    user = fields[0]
    command = fields[1] if len(fields) > 1 else ""
    return {
        "gpu_uuid": gpu_uuid,
        "pid": int(pid),
        "start_time_ticks": start_ticks,
        "user": user,
        "command": command,
    }


def _pid_start_time_ticks(pid: int) -> int:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResourceContractError(
            f"cannot read GPU PID start time: {pid}"
        ) from error
    try:
        after_comm = raw.rsplit(")", 1)[1].split()
        start_ticks = int(after_comm[19])
    except (IndexError, ValueError) as error:
        raise ResourceContractError(
            f"GPU PID stat format changed: {pid}"
        ) from error
    if start_ticks <= 0:
        raise ResourceContractError("GPU PID start time is not positive")
    return start_ticks


def _normalize_external_gpu_pid_baseline(
    rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    if rows is None:
        return ()
    normalized = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        value = _mapping(row, "external GPU PID baseline row")
        if set(value) != {"gpu_uuid", "pid", "start_time_ticks", "user", "command"}:
            raise ResourceContractError("external GPU PID baseline fields changed")
        key = (str(value["gpu_uuid"]), _positive_int(value["pid"], "external GPU PID"))
        if key in seen:
            raise ResourceContractError("external GPU PID baseline duplicated")
        seen.add(key)
        normalized.append(
            {
                "gpu_uuid": key[0],
                "pid": key[1],
                "start_time_ticks": _positive_int(
                    value["start_time_ticks"], "external GPU PID start time"
                ),
                "user": str(value["user"]),
                "command": str(value["command"]),
            }
        )
    return tuple(sorted(normalized, key=lambda item: (item["gpu_uuid"], item["pid"])))


def _query_gpu_compute_apps() -> tuple[tuple[str, int], ...]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ResourceContractError("nvidia-smi compute-app output changed")
        rows.append((fields[0], int(fields[1])))
    return tuple(rows)


def _query_gpu_temperatures() -> dict[str, int]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=uuid,temperature.gpu",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) != 2
            or not fields[0]
            or not fields[1].isdigit()
            or fields[0] in rows
        ):
            raise ResourceContractError("nvidia-smi temperature output changed")
        rows[fields[0]] = int(fields[1])
    return rows


def _sample_swap_io_delta() -> tuple[int, int]:
    before = _read_swap_io()
    time.sleep(0.25)
    after = _read_swap_io()
    delta = (after[0] - before[0], after[1] - before[1])
    if any(value < 0 for value in delta):
        raise ResourceContractError("swap I/O counters moved backwards")
    return delta


def _read_swap_io() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"pswpin", "pswpout"}:
            if not fields[1].isdigit():
                raise ResourceContractError("swap I/O counter is invalid")
            values[fields[0]] = int(fields[1])
    if set(values) != {"pswpin", "pswpout"}:
        raise ResourceContractError("swap I/O counters are missing")
    return values["pswpin"], values["pswpout"]


def _read_cpu_times() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    if len(fields) < 9 or fields[0] != "cpu" or any(
        not value.isdigit() for value in fields[1:]
    ):
        raise ResourceContractError("/proc/stat aggregate CPU row changed")
    values = [int(value) for value in fields[1:]]
    idle = values[3] + values[4]
    return sum(values), idle


class FullRuntimeGuard:
    """Enforce the preregistered Full hard stops and record live evidence."""

    def __init__(
        self,
        policy: Mapping[str, Any],
        *,
        monitor_path: Path,
        probe: Any | None = None,
        temperatures: Any = _query_gpu_temperatures,
        swap_reader: Any = _read_swap_io,
        disk_usage: Any = shutil.disk_usage,
        gpu_process_memory: Any | None = None,
        gpu_compute_apps: Any = _query_gpu_compute_apps,
        cpu_reader: Any = _read_cpu_times,
        allowed_external_gpu_pids: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._policy = dict(_mapping(policy, "Full runtime guard policy"))
        if self._policy.get("policy_id") != "frozen_conservative_e2e_v1":
            raise ValueError("Full runtime guard policy ID changed")
        self._hard_stop = _mapping(
            self._policy.get("hard_stop"), "Full hard-stop policy"
        )
        self._probe = SystemResourceProbe() if probe is None else probe
        self._temperatures = temperatures
        self._swap_reader = swap_reader
        self._disk_usage = disk_usage
        self._gpu_process_memory = (
            _query_gpu_process_memory_bytes
            if gpu_process_memory is None
            else gpu_process_memory
        )
        self._gpu_compute_apps = gpu_compute_apps
        self._allowed_external_gpu_pids = _normalize_external_gpu_pid_baseline(
            allowed_external_gpu_pids
        )
        self._monitor_path = Path(monitor_path)
        self._previous_swap = tuple(self._swap_reader())
        self._cpu_reader = cpu_reader
        self._previous_cpu = tuple(self._cpu_reader())
        self._sustained = {"temperature": 0, "swap": 0, "cpu": 0}
        self._monitor_binding: dict[str, Any] | None = None

    def bind_monitor(
        self,
        *,
        session_name: str,
        claim_path: Path,
        claim_sha256: str,
    ) -> None:
        self._monitor_binding = {
            "session_name": session_name,
            "claim_path": Path(claim_path),
            "claim_sha256": _require_sha256(
                claim_sha256, "Full monitor claim SHA256"
            ),
        }

    def enforce(
        self, processes: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._monitor_binding is not None:
            session = self._monitor_binding["session_name"]
            if subprocess.run(
                ["tmux", "has-session", "-t", session],
                check=False,
                capture_output=True,
            ).returncode != 0:
                raise ResourceContractError(
                    "Full runtime monitor tmux session died"
                )
            monitor_claim = _read_json_mapping(
                self._monitor_binding["claim_path"],
                "Full runtime monitor claim",
            )
            digest_field = (
                "monitor_claim_sha256"
                if "monitor_claim_sha256" in monitor_claim
                else None
            )
            if (
                digest_field is None
                or _require_sha256(
                    monitor_claim[digest_field],
                    "Full runtime monitor claim SHA256",
                )
                != self._monitor_binding["claim_sha256"]
                or _canonical_json_sha256(
                    {
                        key: value
                        for key, value in monitor_claim.items()
                        if key != digest_field
                    }
                )
                != monitor_claim[digest_field]
            ):
                raise ResourceContractError(
                    "Full runtime monitor claim changed"
                )
        ram = self._probe.ram_snapshot()
        snapshots = tuple(
            row
            for row in self._probe.gpu_snapshots()
            if row.index in set(self._policy["gpu_indices"])
        )
        if [row.index for row in snapshots] != self._policy["gpu_indices"]:
            raise ResourceContractError("Full runtime GPU set changed")
        temperatures = dict(self._temperatures())
        current_swap = tuple(self._swap_reader())
        swap_delta = tuple(
            current - previous
            for previous, current in zip(self._previous_swap, current_swap)
        )
        self._previous_swap = current_swap
        if any(value < 0 for value in swap_delta):
            raise ResourceContractError("Full runtime swap counters moved backwards")
        usage = self._disk_usage(REPO_ROOT)
        current_cpu = tuple(self._cpu_reader())
        total_delta = current_cpu[0] - self._previous_cpu[0]
        idle_delta = current_cpu[1] - self._previous_cpu[1]
        self._previous_cpu = current_cpu
        if total_delta <= 0 or not 0 <= idle_delta <= total_delta:
            raise ResourceContractError("Full runtime CPU counters changed")
        cpu_busy_percent = 100.0 * (total_delta - idle_delta) / total_delta
        temperature_hot = any(
            temperatures.get(row.uuid, -1)
            > self._hard_stop["temperature_c_above"]
            for row in snapshots
        )
        swap_active = (
            self._hard_stop.get("swap_io_positive") is True
            and swap_delta != (0, 0)
        )
        self._sustained["temperature"] = (
            self._sustained["temperature"] + 1 if temperature_hot else 0
        )
        self._sustained["swap"] = (
            self._sustained["swap"] + 1 if swap_active else 0
        )
        self._sustained["cpu"] = (
            self._sustained["cpu"] + 1
            if cpu_busy_percent
            >= self._hard_stop["cpu_percent_at_or_above"]
            else 0
        )
        process_rows = {}
        live_worker_pids: set[int] = set()
        for worker_id, process in sorted((processes or {}).items()):
            if process.poll() is None:
                pid = int(process.pid)
                live_worker_pids.add(pid)
                process_rows[worker_id] = {
                    "pid": pid,
                    "process_tree_rss_bytes": _process_tree_rss_bytes(pid),
                }
        selected_uuids = {row.uuid for row in snapshots}
        current_compute_apps = tuple(self._gpu_compute_apps())
        allowed_external = {
            (row["gpu_uuid"], row["pid"], row["start_time_ticks"])
            for row in self._allowed_external_gpu_pids
        }
        new_unknown = []
        for gpu_uuid, pid in current_compute_apps:
            if gpu_uuid not in selected_uuids or int(pid) in live_worker_pids:
                continue
            start_ticks = _pid_start_time_ticks(int(pid))
            if (gpu_uuid, int(pid), start_ticks) not in allowed_external:
                new_unknown.append((gpu_uuid, int(pid)))
        if new_unknown:
            raise ResourceContractError(
                "Full runtime found unknown GPU compute PIDs: "
                + ",".join(f"{uuid}:{pid}" for uuid, pid in sorted(new_unknown))
            )
        gpu_memory = dict(self._gpu_process_memory())
        sample = {
            "schema_version": 1,
            "contract_type": "safa_r9_full_runtime_monitor_sample_v1",
            "monotonic_ns": time.monotonic_ns(),
            "ram_used_bytes": ram.used_bytes,
            "ram_total_bytes": ram.total_bytes,
            "disk_used_bytes": usage.used,
            "disk_total_bytes": usage.total,
            "cpu_busy_percent": cpu_busy_percent,
            "temperatures_c": temperatures,
            "swap_in_delta_pages": swap_delta[0],
            "swap_out_delta_pages": swap_delta[1],
            "gpu": [
                {
                    "index": row.index,
                    "uuid": row.uuid,
                    "used_bytes": row.total_bytes - row.free_bytes,
                    "total_bytes": row.total_bytes,
                    "compute_process_bytes": gpu_memory.get(row.uuid, 0),
                }
                for row in snapshots
            ],
            "processes": process_rows,
            "external_compute_pid_baseline": list(
                self._allowed_external_gpu_pids
            ),
        }
        self._monitor_path.parent.mkdir(parents=True, exist_ok=True)
        with self._monitor_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    sample,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        if (
            ram.used_bytes * 100
            >= ram.total_bytes * self._hard_stop["ram_percent_at_or_above"]
        ):
            raise ResourceContractError("Full runtime crossed the 90% RAM hard stop")
        if (
            usage.used * 100
            >= usage.total * self._hard_stop["disk_percent_at_or_above"]
        ):
            raise ResourceContractError("Full runtime crossed the 90% disk hard stop")
        if any(
            (row.total_bytes - row.free_bytes) * 100
            >= row.total_bytes
            * self._hard_stop["gpu_memory_percent_at_or_above"]
            for row in snapshots
        ):
            raise ResourceContractError(
                "Full runtime crossed the 90% GPU-memory hard stop"
            )
        sustained = self._hard_stop["sustained_sample_count"]
        if self._sustained["temperature"] >= sustained:
            raise ResourceContractError(
                "Full runtime sustained GPU temperature above 85C"
            )
        if self._sustained["swap"] >= sustained:
            raise ResourceContractError("Full runtime sustained swap I/O")
        if self._sustained["cpu"] >= sustained:
            raise ResourceContractError(
                "Full runtime sustained host CPU at or above 90%"
            )
        return sample


def _query_gpu_process_memory_bytes() -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ResourceContractError("nvidia-smi compute-memory query failed")
    totals: dict[str, int] = {}
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ResourceContractError(
                "nvidia-smi compute-memory output changed"
            )
        totals[fields[0]] = totals.get(fields[0], 0) + int(fields[1]) * 1024**2
    return totals


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
    runtime_guard: FullRuntimeGuard | None = None,
) -> int:
    if campaign_runtime:
        _validate_requested_campaign_role(
            campaign_runtime, requested_phase=requested_phase
        )
    is_continuation = campaign_runtime.get("continuation") is not None
    continuation = _continuation_for_runtime(campaign_runtime)
    continuation_start = (
        None
        if continuation is None
        else str(continuation.get("start_phase", "calibrate"))
    )
    rejected = (
        {"preflight", "diagnose", "calibrate"}
        if continuation_start == "confirm512"
        else {"preflight", "diagnose"}
    )
    if is_continuation and requested_phase in rejected:
        raise ValueError("continuation child rejects preflight and diagnose")
    if is_continuation and requested_phase == "all":
        selected_phases = (
            ("confirm512", "full")
            if continuation_start == "confirm512"
            else ("calibrate", "confirm512", "full")
        )
    else:
        selected_phases = (
            PHASES if requested_phase == "all" else (requested_phase,)
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
            runtime_guard=runtime_guard,
        )
        if closure_request is not None:
            closure = materialize_phase_results(
                closure_request,
                quality_evaluator=quality_evaluator,
                arcface_evaluator=arcface_evaluator,
                heldout_evaluator=heldout_evaluator,
                evaluator_parallelism=4,
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
    upstream_calibration_selection = None
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
        elif (
            plan.phase == "confirm512"
            and continuation is not None
            and continuation.get("start_phase") == "confirm512"
        ):
            selection_binding = _mapping(
                continuation.get("selection"), "calibration selection binding"
            )
            selection_path = _repo_path(
                REPO_ROOT,
                selection_binding.get("path"),
                "calibration report-only selection",
            )
            from safa.evaluation.r9_calibration_selection_contracts import (
                validate_calibration_report_only_selection_contract,
            )

            upstream_calibration_selection = (
                validate_calibration_report_only_selection_contract(
                    _read_json_mapping(
                        selection_path, "calibration report-only selection"
                    ),
                    repo_root=REPO_ROOT,
                )
            )
        elif (
            plan.phase == "full"
            and continuation is not None
            and continuation.get("start_phase") == "full"
        ):
            selection = _require_full_selection_binding(
                continuation, campaign_runtime
            )
        else:
            upstream_path = campaign_root / upstream_phase / "gate_contract.json"
        if upstream_calibration_selection is None and selection is None:
            upstream_gate = _load_gate(upstream_path, upstream_phase)
            if plan.phase != "calibrate" and continuation is not None:
                _require_gate_continuation(upstream_gate, campaign_runtime)
    if plan.phase == "full":
        if selection is None:
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
        upstream_calibration_selection=upstream_calibration_selection,
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
        runtime_guard: FullRuntimeGuard | None = None,
        rss_sampler: Any | None = None,
        gpu_process_memory: Any | None = None,
    ) -> None:
        self._python = str(runtime["python"])
        evaluation = _mapping(campaign_runtime.get("evaluation"), "evaluation")
        self._worker_contract = _normalized_worker_contract(evaluation)
        self._worker_script = Path(self._worker_contract["path"])
        self._worker_implementation = Path(self._worker_contract["implementation_path"])
        self._validate_current_worker_contract()
        self._evaluation = dict(evaluation)
        normalized_arcface = _normalized_arcface_evaluation_contract(evaluation)
        self._evaluation["arcface"] = normalized_arcface
        quality = _mapping(evaluation.get("quality"), "quality evaluation")
        quality_script = _mapping(quality.get("script"), "quality script")
        self._quality_script_sha256 = _require_sha256(
            quality_script.get("sha256"), "quality script SHA256"
        )
        self._arcface_contract_sha256 = _canonical_json_sha256(
            normalized_arcface
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
        self._scheduler_lock = threading.RLock()
        self._active_evaluator_processes: dict[str, Any] = {}
        self._quality_execution_lock = threading.Lock()
        self._runtime_guard = runtime_guard
        self._rss_sampler = (
            _process_tree_rss_bytes if rss_sampler is None else rss_sampler
        )
        self._gpu_process_memory = (
            _query_gpu_process_memory_bytes
            if gpu_process_memory is None
            else gpu_process_memory
        )

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
        with self._quality_execution_lock:
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
        if request.phase != "full":
            raise RuntimeError("heldout evaluator is only authorized for formal Full")
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
        with self._scheduler_lock:
            self._launch_counter += 1
            launch_counter = self._launch_counter
        worker_id = f"evaluator:{evaluator}:{phase}:{unit_id}"
        lease = None
        exclusive_lock_fd: int | None = None
        reserved_worker_ids: list[str] = []
        if evaluator == "heldout":
            with self._scheduler_lock:
                if self._scheduler.active_leases:
                    raise ResourceContractError(
                        "heldout requires an empty globally exclusive scheduler"
                    )
            exclusive_lock_path = Path(
                "/tmp/safa-r9-heldout-global-exclusive-v1.lock"
            )
            exclusive_lock_fd = os.open(
                exclusive_lock_path,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            try:
                fcntl.flock(
                    exclusive_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError as error:
                os.close(exclusive_lock_fd)
                raise ResourceContractError(
                    "heldout global exclusive lock is contended"
                ) from error
        reservation_count = 1
        for reservation_index in range(reservation_count):
            reservation_id = (
                worker_id
                if reservation_index == 0
                else f"{worker_id}:exclusive-slot-{reservation_index}"
            )
            reserved = None
            while reserved is None:
                with self._scheduler_lock:
                    reserved = _admit_worker(
                        self._scheduler,
                        worker_id=reservation_id,
                        launch_ordinal=(
                            50_000 + launch_counter + reservation_index
                        ),
                        gpu_bindings=self._gpu_bindings,
                        ram_slot_budget_bytes=(
                            16 * 1024**3
                            if evaluator == "heldout"
                            else self._evaluator_ram_slot_budgets[evaluator]
                        ),
                        start_gpu_index=(
                            (launch_counter - 1 + reservation_index) % 4
                        ),
                    )
                if reserved is None:
                    with self._scheduler_lock:
                        self._scheduler.enforce_actual_ram_limit()
                    self._sleep(self._poll_interval_seconds)
            reserved_worker_ids.append(reservation_id)
            self._peer_status_store.record_admitted(reservation_id)
            if reservation_index == 0:
                lease = reserved
        if lease is None:
            raise AssertionError("evaluator primary lease is missing")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = lease.gpu_uuid
        environment["SAFA_R9_WORKER_ID"] = worker_id
        environment["SAFA_R9_GPU_UUID"] = lease.gpu_uuid
        environment["SAFA_R9_GPU_SLOT"] = str(lease.slot_index)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process = None
        worker_terminal = False
        peak_rss_bytes = 0
        peak_gpu_memory_bytes = 0
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
                with self._scheduler_lock:
                    self._active_evaluator_processes[worker_id] = process
                self._peer_status_store.record_running(
                    worker_id, pid=_positive_int(process.pid, "evaluator PID")
                )
                while process.poll() is None:
                    try:
                        with self._scheduler_lock:
                            self._scheduler.enforce_actual_ram_limit()
                            guarded_processes = dict(self._active_evaluator_processes)
                        guarded_processes[worker_id] = process
                        peak_rss_bytes = max(
                            peak_rss_bytes,
                            int(self._rss_sampler(int(process.pid))),
                        )
                        peak_gpu_memory_bytes = max(
                            peak_gpu_memory_bytes,
                            int(
                                dict(self._gpu_process_memory()).get(
                                    lease.gpu_uuid, 0
                                )
                            ),
                        )
                        if self._runtime_guard is not None:
                            self._runtime_guard.enforce(guarded_processes)
                    except (CampaignFailedError, ResourceContractError):
                        _terminate_process(process)
                        self._peer_status_store.record_terminal(
                            worker_id, state="terminated"
                        )
                        worker_terminal = True
                        raise
                    self._sleep(self._poll_interval_seconds)
                if process.returncode != 0:
                    self._peer_status_store.record_terminal(worker_id, state="failed")
                    worker_terminal = True
                    with self._scheduler_lock:
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
            if phase == "full_e2e":
                if peak_rss_bytes <= 0 or peak_gpu_memory_bytes <= 0:
                    raise ResourceContractError(
                        "Full E2E evaluator recorded no positive resource peak"
                    )
                observation = {
                    "schema_version": 1,
                    "contract_type": (
                        "safa_r9_full_e2e_evaluator_resource_observation_v1"
                    ),
                    "task": evaluator,
                    "unit_id": unit_id,
                    "worker_id": worker_id,
                    "gpu_uuid": lease.gpu_uuid,
                    "peak_process_tree_rss_bytes": peak_rss_bytes,
                    "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                    "evaluator_request_sha256": contract[
                        "evaluator_request_sha256"
                    ],
                    "evaluator_output_sha256": _require_sha256(
                        _read_json_mapping(
                            output_path, "Full E2E evaluator output"
                        ).get("evaluator_output_sha256"),
                        "Full E2E evaluator output SHA256",
                    ),
                    "worker_contract": dict(self._worker_contract),
                    "arcface_contract_sha256": self._arcface_contract_sha256,
                    "quality_script_sha256": self._quality_script_sha256,
                    "resource_policy_id": "frozen_conservative_e2e_v1",
                }
                observation["resource_observation_sha256"] = (
                    _canonical_json_sha256(observation)
                )
                _write_exclusive_bytes(
                    root / "resource_observation.json",
                    (
                        json.dumps(
                            observation,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
        except BaseException:
            self._terminate_evaluator_peers(
                current_worker_id=worker_id,
                current_worker_terminal=worker_terminal,
            )
            _cleanup_evaluator_work_root(root / "work", evaluator_root=root)
            if exclusive_lock_fd is not None:
                fcntl.flock(exclusive_lock_fd, fcntl.LOCK_UN)
                os.close(exclusive_lock_fd)
            raise
        _cleanup_evaluator_work_root(root / "work", evaluator_root=root)
        self._peer_status_store.record_terminal(worker_id, state="succeeded")
        with self._scheduler_lock:
            self._active_evaluator_processes.pop(worker_id, None)
            for reservation_id in reversed(reserved_worker_ids):
                if reservation_id != worker_id:
                    self._peer_status_store.record_terminal(
                        reservation_id, state="succeeded"
                    )
                self._scheduler.release_worker(reservation_id)
        if exclusive_lock_fd is not None:
            fcntl.flock(exclusive_lock_fd, fcntl.LOCK_UN)
            os.close(exclusive_lock_fd)
        return result

    def _terminate_evaluator_peers(
        self, *, current_worker_id: str, current_worker_terminal: bool
    ) -> None:
        with self._scheduler_lock:
            processes = dict(self._active_evaluator_processes)
            active_worker_ids = {
                lease.worker_id for lease in self._scheduler.active_leases
            }
        for worker_id, process in processes.items():
            if process.poll() is None:
                _terminate_process(process)
        terminal_worker_ids = active_worker_ids | set(processes)
        if current_worker_terminal:
            terminal_worker_ids.discard(current_worker_id)
        for worker_id in sorted(terminal_worker_ids):
            self._peer_status_store.record_terminal(worker_id, state="terminated")
        with self._scheduler_lock:
            self._active_evaluator_processes.clear()
            if self._scheduler.failure is None:
                try:
                    self._scheduler.fail_worker(
                        current_worker_id, kind=FailureKind.CONTRACT_MISMATCH
                    )
                except CampaignFailedError:
                    pass
            self._scheduler.release_all_workers_after_failure()

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
        continuation = _continuation_for_runtime(campaign_runtime)
        if continuation is not None and continuation.get("start_phase") == "confirm512":
            selected = [
                str(row["arm_id"]) for row in continuation["selected_arms"]
            ]
        else:
            gate = _load_gate(root / "calibrate" / "gate_contract.json", "calibrate")
            _require_gate_continuation(gate, campaign_runtime)
            selected = list(gate["selected_arm_ids"])
        if not 1 <= len(selected) <= 2:
            raise RuntimeError("confirm512 requires 1..2 B-stage promotions")
        return selected, None
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is not None and continuation.get("start_phase") == "full":
        selection = _require_full_selection_binding(
            continuation, campaign_runtime
        )
        return None, str(_mapping(selection["winner"], "winner")["arm_id"])
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
        context["continuation_contract_sha256"] = _continuation_digest(
            continuation
        )
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
        continuation = _continuation_for_runtime(campaign_runtime)
        if continuation is not None and continuation.get("start_phase") == "full":
            selection = _require_full_selection_binding(
                continuation, campaign_runtime
            )
        else:
            confirm_gate = _load_gate(
                root / "confirm512" / "gate_contract.json", "confirm512"
            )
            selection = validate_selection_contract(
                _read_json_mapping(root / "selection.json", "selection"),
                confirm_gate,
            )
        if not (
            continuation is not None
            and continuation.get("start_phase") == "full"
        ):
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
            formal_hard_requirements=(
                campaign_id == FULL_CONTINUATION_CHILD_CAMPAIGN_ID
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
    if context.get("continuation_contract_sha256") != _continuation_digest(
        continuation
    ):
        raise ValueError("child gate continuation SHA256 mismatch")


def _require_selection_continuation(
    selection: Mapping[str, Any], campaign_runtime: Mapping[str, Any]
) -> None:
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is None:
        return
    if selection.get("continuation_contract_sha256") != _continuation_digest(
        continuation
    ):
        raise ValueError("child selection continuation SHA256 mismatch")


def _require_full_selection_binding(
    continuation: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if continuation.get("start_phase") != "full":
        raise ValueError("Full selection validator requires a Full continuation")
    runtime_binding = _mapping(
        campaign_runtime.get("continuation"), "runtime continuation binding"
    )
    if runtime_binding.get("contract_sha256") != _continuation_digest(continuation):
        raise ValueError("runtime Full continuation digest changed")
    binding = _mapping(
        continuation.get("selection"), "Full continuation selection binding"
    )
    if set(binding) != {
        "path",
        "file_sha256",
        "contract_sha256",
        "prospective_file_sha256",
    }:
        raise ValueError("Full continuation selection binding fields changed")
    selection_path = _repo_path(
        REPO_ROOT, binding.get("path"), "Full continuation selection"
    )
    file_sha256 = _sha256_path(selection_path)
    if (
        file_sha256
        != _require_sha256(binding.get("file_sha256"), "Full selection file SHA256")
        or file_sha256
        != _require_sha256(
            binding.get("prospective_file_sha256"),
            "Full selection prospective SHA256",
        )
    ):
        raise ValueError("Full continuation selection file binding changed")
    selection = validate_full_continuation_selection_contract(
        _read_json_mapping(selection_path, "Full continuation selection"),
        repo_root=REPO_ROOT,
        expected_source=expected_source_from_full_continuation(continuation),
    )
    if selection["selection_sha256"] != _require_sha256(
        binding.get("contract_sha256"), "Full selection contract SHA256"
    ):
        raise ValueError("Full continuation selection internal digest changed")
    if "continuation_contract_sha256" in selection:
        raise ValueError("Full selection must not contain a cyclic continuation digest")
    return selection


def _continuation_digest(continuation: Mapping[str, Any]) -> str:
    value = continuation.get(
        "continuation_contract_sha256",
        continuation.get(
            "confirm_continuation_sha256",
            continuation.get("full_continuation_sha256"),
        ),
    )
    return _require_sha256(value, "continuation contract SHA256")


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


def run_generation_batch_benchmark(
    runtime: Mapping[str, Any],
    campaign_runtime: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    *,
    probe: Any | None = None,
    process_factory: Any = subprocess.Popen,
    rss_sampler: Any | None = None,
    sleep: Any = time.sleep,
    poll_interval_seconds: float = 0.1,
) -> dict[str, Any]:
    """Run the fixed six-run batch equivalence and measured resource benchmark."""
    declaration = _mapping(
        runtime.get("generation_batch_benchmark"), "generation batch benchmark"
    )
    contract_path = REPO_ROOT / str(declaration["contract_path"])
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is None:
        raise ValueError("generation batch benchmark requires a continuation")
    continuation_sha = _continuation_digest(continuation)
    if contract_path.is_file():
        return validate_generation_batch_benchmark_contract(
            _read_json_mapping(contract_path, "generation batch benchmark"),
            repo_root=REPO_ROOT,
            expected_campaign_id=str(campaign_runtime["campaign_id"]),
            expected_continuation_contract_sha256=continuation_sha,
        )
    output_root = REPO_ROOT / str(declaration["output_root"])
    request_path = output_root / "benchmark_request.json"
    if request_path.exists():
        raise RuntimeError(
            "generation benchmark request exists without a final contract; retry is forbidden"
        )
    resource_probe = SystemResourceProbe() if probe is None else probe
    snapshots = tuple(
        row for row in resource_probe.gpu_snapshots() if row.index in {0, 1, 2, 3}
    )
    if tuple(row.index for row in snapshots) != (0, 1, 2, 3):
        raise ResourceContractError("generation benchmark requires GPUs 0-3")
    manifests = _mapping(manifest_contract.get("manifests"), "manifests")
    manifest_key = str(declaration["manifest"])
    manifest = _mapping(manifests.get(manifest_key), "benchmark manifest")
    sample_count = _positive_int(declaration["sample_count"], "benchmark samples")
    seed = _positive_int(declaration["seed"], "benchmark seed")
    arms = tuple(str(value) for value in declaration["required_arms"])
    batch_sizes = tuple(int(value) for value in declaration["batch_sizes"])
    if arms != (
        "native",
        "paper_eta_0p125",
        "flow_map2_normalized_eta_0p125",
    ) or batch_sizes != (2, 4):
        raise ValueError("generation benchmark matrix changed")
    runs: list[tuple[RunSpec, int, int]] = []
    request_runs: list[dict[str, Any]] = []
    ordinal = 0
    for arm_index, arm_id in enumerate(arms):
        for batch_size in batch_sizes:
            logical_id = f"{arm_id}__batch_{batch_size}"
            relative_output = Path(str(declaration["output_root"])) / logical_id
            relative_config = (
                Path(str(campaign_runtime["campaign_root"]))
                / "runtime_configs"
                / "generation_batch_benchmark"
                / f"{logical_id}.yaml"
            )
            run = RunSpec(
                phase="confirm512",
                logical_run_id=logical_id,
                arm_ref=arm_id,
                seed=seed,
                repeat_index=None,
                shard_index=0,
                num_shards=1,
                sample_count=sample_count,
                manifest_key=manifest_key,
                runtime_config=relative_config,
                output_dir=relative_output,
                command=(),
            )
            config = build_run_runtime_config(
                runtime, campaign_runtime, manifest_contract, run
            )
            config.update(
                {
                    "experiment_name": f"generation_batch_benchmark__{logical_id}",
                    "out_dir": str(relative_output),
                    "max_samples": sample_count,
                    "batch_size": batch_size,
                    "record_final_latent_sha256": True,
                }
            )
            config = resolve_frozen_effective_guidance_config(config)
            content = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
            _write_immutable_bytes(REPO_ROOT / relative_config, content)
            gpu_index = arm_index
            command = (
                str(runtime["python"]),
                str(runtime["generation_script"]),
                "--config",
                str(relative_config),
                "--output-dir",
                str(relative_output),
                "--shard-index",
                "0",
                "--num-shards",
                "1",
            )
            bound_run = RunSpec(**{**run.__dict__, "command": command})
            runs.append((bound_run, batch_size, gpu_index))
            request_runs.append(
                {
                    "arm_id": arm_id,
                    "batch_size": batch_size,
                    "gpu_index": gpu_index,
                    "gpu_uuid": snapshots[gpu_index].uuid,
                    "runtime_config": str(relative_config),
                    "runtime_config_sha256": hashlib.sha256(content).hexdigest(),
                    "output_dir": str(relative_output),
                }
            )
            ordinal += 1
    request = {
        "schema_version": 1,
        "contract_type": "safa_r9_generation_batch_benchmark_request_v1",
        "campaign_id": str(campaign_runtime["campaign_id"]),
        "campaign_runtime_sha256": _require_sha256(
            campaign_runtime.get("campaign_runtime_sha256"),
            "benchmark source runtime SHA256",
        ),
        "continuation_contract_sha256": continuation_sha,
        "manifest": {
            "path": str(manifest["path"]),
            "sha256": str(manifest["sha256"]),
            "sample_count": sample_count,
        },
        "seed": seed,
        "runs": request_runs,
        "gpu_snapshots": [
            {
                "index": row.index,
                "uuid": row.uuid,
                "total_bytes": row.total_bytes,
                "free_bytes": row.free_bytes,
            }
            for row in snapshots
        ],
        "retry_count": 0,
    }
    request["benchmark_request_sha256"] = _canonical_json_sha256(request)
    _write_exclusive_bytes(
        request_path,
        (
            json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode(),
    )
    sampler = _process_tree_rss_bytes if rss_sampler is None else rss_sampler
    evidence: list[BatchRunEvidence] = []
    for run, batch_size, gpu_index in runs:
        snapshot = resource_probe.gpu_snapshot(
            gpu_index, expected_uuid=snapshots[gpu_index].uuid
        )
        ram_before = resource_probe.ram_snapshot()
        if ram_before.used_bytes * 100 >= ram_before.total_bytes * 85:
            raise ResourceContractError(
                "generation benchmark cannot launch at or above 85% RAM"
            )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = snapshot.uuid
        log_path = output_root / "controller_logs" / f"{run.logical_run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        peak_rss = 0
        with log_path.open("xb") as log:
            process = process_factory(
                run.command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                rss_bytes, reaped = _sample_or_reap_process_tree(process, sampler)
                if reaped is not None:
                    break
                if rss_bytes is None:
                    raise AssertionError("benchmark RSS sample is missing")
                peak_rss = max(peak_rss, rss_bytes)
                ram = resource_probe.ram_snapshot()
                if ram.used_bytes * 100 >= ram.total_bytes * 90:
                    _terminate_process(process)
                    raise ResourceContractError(
                        "generation benchmark crossed the 90% RAM hard limit"
                    )
                sleep(poll_interval_seconds)
            returncode = process.wait() if process.returncode is None else process.returncode
        if returncode != 0:
            raise ResourceContractError(
                f"generation benchmark worker failed once with exit code {returncode}"
            )
        if peak_rss <= 0:
            raise ResourceContractError("generation benchmark measured no positive RSS")
        validate_worker_completion(run)
        evidence.append(
            BatchRunEvidence(
                arm_id=run.arm_ref,
                batch_size=batch_size,
                output_dir=run.output_dir,
                gpu_uuid=snapshot.uuid,
                free_vram_before_bytes=snapshot.free_bytes,
                peak_process_tree_rss_bytes=peak_rss,
            )
        )
    contract = build_generation_batch_benchmark_contract(
        repo_root=REPO_ROOT,
        campaign_id=str(campaign_runtime["campaign_id"]),
        manifest_path=Path(str(manifest["path"])),
        manifest_sha256=str(manifest["sha256"]),
        sample_count=sample_count,
        seed=seed,
        continuation_contract_sha256=continuation_sha,
        request_sha256=request["benchmark_request_sha256"],
        required_arm_ids=arms,
        gpu_snapshots=[
            BenchmarkGpuSnapshot(
                index=row.index,
                uuid=row.uuid,
                total_bytes=row.total_bytes,
                free_bytes=row.free_bytes,
            )
            for row in snapshots
        ],
        evidence=evidence,
    )
    materialize_generation_batch_benchmark_contract(contract_path, contract)
    return contract


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
    runtime_guard: FullRuntimeGuard | None = None,
) -> int:
    """Refill all admitted GPU slots and fail the campaign on any peer error."""
    bindings = _validate_gpu_bindings(gpu_bindings)
    pending: list[tuple[RunSpec, int, int]] = []
    for plan in plans:
        for run_index, scheduled in enumerate(_generation_launch_schedule(plan)):
            run = scheduled["run"]
            runtime_config = REPO_ROOT / run.runtime_config
            if not runtime_config.is_file():
                raise FileNotFoundError(
                    f"immutable R9 runtime config is missing: {runtime_config}"
                )
            completion = REPO_ROOT / run.output_dir / "completion.json"
            if completion.is_file():
                validate_worker_completion(run)
                continue
            pending.append(
                (
                    run,
                    _stable_launch_ordinal(run.phase, run_index),
                    int(scheduled["preferred_gpu_index"]),
                )
            )
    active: dict[str, ActiveWorker] = {}
    while pending or active:
        launched = False
        pending_index = 0
        while pending_index < len(pending):
            run, launch_ordinal, preferred_gpu_index = pending[pending_index]
            if (
                runtime_guard is not None
                and run.phase == "full"
            ):
                active_gpu_indices = {worker.gpu_index for worker in active.values()}
                if (
                    len(active) >= FULL_GUARDED_MAX_ACTIVE_WORKERS
                    or preferred_gpu_index in active_gpu_indices
                ):
                    pending_index += 1
                    continue
            worker_id = f"{run.phase}:{run.logical_run_id}:shard-{run.shard_index}"
            lease = _admit_worker(
                scheduler,
                worker_id=worker_id,
                launch_ordinal=launch_ordinal,
                gpu_bindings=bindings,
                ram_slot_budget_bytes=scheduler.ram_slot_budget_bytes,
                start_gpu_index=preferred_gpu_index,
            )
            if lease is None:
                pending_index += 1
                continue
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
                gpu_index={uuid: index for index, uuid in bindings.items()}[
                    lease.gpu_uuid
                ],
            )
            pending.pop(pending_index)
            launched = True
        if not active and pending:
            sleep(poll_interval_seconds)
            continue
        try:
            scheduler.enforce_actual_ram_limit()
            if runtime_guard is not None:
                runtime_guard.enforce(
                    {
                        worker_id: worker.process
                        for worker_id, worker in active.items()
                    }
                )
        except (CampaignFailedError, ResourceContractError):
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


def _generation_launch_schedule(plan: PhasePlan) -> tuple[dict[str, Any], ...]:
    """Pre-register deterministic GPU launch preferences for one phase."""
    by_logical: dict[str, dict[int, RunSpec]] = {}
    logical_order: list[str] = []
    for run in plan.runs:
        if run.logical_run_id not in by_logical:
            by_logical[run.logical_run_id] = {}
            logical_order.append(run.logical_run_id)
        shards = by_logical[run.logical_run_id]
        if run.shard_index in shards:
            raise ValueError("launch schedule repeats a logical shard")
        shards[run.shard_index] = run
    if len(logical_order) != plan.logical_run_count:
        raise ValueError("launch schedule logical-run count mismatch")
    logical_count = len(logical_order)
    shard_counts = {run.num_shards for run in plan.runs}
    if len(shard_counts) != 1:
        raise ValueError("launch schedule requires one shard count per phase")
    shard_count = next(iter(shard_counts))
    for logical_index, logical_run_id in enumerate(logical_order):
        if set(by_logical[logical_run_id]) != set(range(shard_count)):
            raise ValueError("launch schedule logical run has incomplete shards")
    schedule = []
    launch_index = 0
    if plan.phase == "full":
        for logical_index, logical_run_id in enumerate(logical_order):
            for shard_index in range(shard_count):
                run = by_logical[logical_run_id][shard_index]
                schedule.append(
                    {
                        "launch_index": launch_index,
                        "logical_run_id": logical_run_id,
                        "arm_ref": run.arm_ref,
                        "shard_index": shard_index,
                        "preferred_gpu_index": shard_index % 4,
                        "run": run,
                    }
                )
                launch_index += 1
    else:
        for shard_index in range(shard_count):
            for offset in range(logical_count):
                logical_index = (shard_index + offset) % logical_count
                logical_run_id = logical_order[logical_index]
                run = by_logical[logical_run_id][shard_index]
                schedule.append(
                    {
                        "launch_index": launch_index,
                        "logical_run_id": logical_run_id,
                        "arm_ref": run.arm_ref,
                        "shard_index": shard_index,
                        "preferred_gpu_index": (shard_index + logical_index) % 4,
                        "run": run,
                    }
                )
                launch_index += 1
    if len(schedule) != len(plan.runs):
        raise ValueError("launch schedule does not cover every shard")
    return tuple(schedule)


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
        "r9_generation_batch_benchmark_sha256",
        "r9_generation_gpu_slot_claim_bytes",
        "r9_generation_ram_slot_budget_bytes",
        "r9_generation_slots_per_gpu",
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
    result_path = _repo_path(
        REPO_ROOT, result.get("path"), "resource smoke result contract"
    )
    result_payload = validate_resource_smoke_contract(
        _read_json_mapping(result_path, "resource smoke result contract")
    )
    peak_rss_bytes = _positive_int(
        result_payload.get("peak_rss_bytes"), "resource smoke peak RSS bytes"
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
    max_gpu_slots = _positive_int(
        resources.get("max_slots_per_gpu"), "max GPU slots"
    )
    generation_slots_per_gpu = resources.get("generation_slots_per_gpu")
    if generation_slots_per_gpu is not None:
        max_gpu_slots = min(
            max_gpu_slots,
            _positive_int(
                generation_slots_per_gpu, "generation slots per GPU"
            ),
        )
    scheduler = R9ResourceScheduler(
        campaign_id=str(campaign_runtime["campaign_id"]),
        resource_contract_sha256=resource_contract_sha256,
        smoke_peak_rss_bytes=peak_rss_bytes,
        probe=resource_probe,
        lock_backend=locks,
        peer_status_probe=status_store,
        gpu_slot_claim_bytes=_positive_int(
            resources.get("gpu_slot_claim_bytes"), "GPU slot claim bytes"
        ),
        gpu_headroom_bytes=_positive_int(
            resources.get("gpu_headroom_bytes"), "GPU headroom bytes"
        ),
        max_gpu_slots=max_gpu_slots,
        ram_slot_budget_bytes_override=_positive_int(
            resources.get("ram_slot_budget_bytes"), "RAM slot budget bytes"
        ),
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
    *,
    continuation_contract: Mapping[str, Any] | None = None,
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
    formal_closure = _formal_closure_for_runtime(
        campaign_runtime, continuation_contract=continuation_contract
    )
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
    continuation = (
        validate_continuation_contract(continuation_contract, repo_root=REPO_ROOT)
        if continuation_contract is not None
        else _continuation_for_runtime(campaign_runtime)
    )
    if continuation is not None:
        config["r9_continuation_contract_sha256"] = _continuation_digest(
            continuation
        )
    batch_benchmark = campaign_runtime.get("generation_batch_benchmark")
    if batch_benchmark is not None:
        binding = _mapping(batch_benchmark, "generation batch benchmark binding")
        resources = _mapping(campaign_runtime.get("resources"), "resources")
        config.update(
            {
                "batch_size": _positive_int(
                    resources.get("generation_batch_size"),
                    "generation batch size",
                ),
                "r9_generation_batch_benchmark_sha256": _require_sha256(
                    binding.get("contract_sha256"),
                    "generation batch benchmark SHA256",
                ),
                "r9_generation_gpu_slot_claim_bytes": _positive_int(
                    resources.get("gpu_slot_claim_bytes"), "GPU slot claim bytes"
                ),
                "r9_generation_ram_slot_budget_bytes": _positive_int(
                    resources.get("ram_slot_budget_bytes"), "RAM slot budget bytes"
                ),
                "r9_generation_slots_per_gpu": _positive_int(
                    resources.get("generation_slots_per_gpu"),
                    "generation slots per GPU",
                ),
            }
        )
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
                "generation_batch_benchmark_sha256": config.get(
                    "r9_generation_batch_benchmark_sha256"
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
        "launch_schedule": [
            {key: value for key, value in row.items() if key != "run"}
            for row in _generation_launch_schedule(plan)
        ],
    }
    continuation = _continuation_for_runtime(campaign_runtime)
    if continuation is not None:
        payload["continuation_contract_sha256"] = _continuation_digest(
            continuation
        )
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


def _read_proc_stat_identity(
    pid: int, *, proc_root: Path
) -> tuple[int, int, str] | None:
    stat_path = proc_root / str(pid) / "stat"
    try:
        raw = stat_path.read_bytes().strip()
    except FileNotFoundError:
        return None
    left = raw.find(b"(")
    right = raw.rfind(b")")
    if left < 0 or right <= left:
        raise ResourceContractError(f"process {pid} stat format is invalid")
    declared_pid = raw[:left].strip()
    fields = raw[right + 1 :].split()
    valid_states = b"RSDZTWtXxKWPIN"
    if (
        len(fields) < 20
        or len(fields[0]) != 1
        or fields[0][0] not in valid_states
    ):
        raise ResourceContractError(f"process {pid} stat format is invalid")
    if not declared_pid.isdigit() or not fields[19].isdigit():
        raise ResourceContractError(f"process {pid} stat numeric field is invalid")
    parsed_pid = int(declared_pid)
    starttime = int(fields[19])
    if parsed_pid != pid or starttime < 0:
        raise ResourceContractError(f"process {pid} stat identity is invalid")
    return parsed_pid, starttime, fields[0].decode("ascii")


def _sample_proc_rss_and_children(
    pid: int,
    *,
    expected_starttime: int | None,
    proc_root: Path,
    page_size_bytes: int,
) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
    first = _read_proc_stat_identity(pid, proc_root=proc_root)
    if first is None:
        return None
    if expected_starttime is not None and first[1] != expected_starttime:
        return None
    if first[2] == "Z":
        return 0, first[1], ()
    statm_path = proc_root / str(pid) / "statm"
    statm_raw: bytes | None
    statm_error: OSError | None = None
    try:
        statm_raw = statm_path.read_bytes()
    except OSError as error:
        statm_raw = None
        statm_error = error
    child_identities: set[tuple[int, int]] = set()
    children_error: Exception | None = None
    task_root = proc_root / str(pid) / "task"
    try:
        task_dirs = tuple(
            path for path in task_root.iterdir() if path.name.isdecimal()
        )
    except OSError as error:
        task_dirs = ()
        children_error = error
    for task_dir in task_dirs:
        try:
            children_raw = (task_dir / "children").read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            children_error = error
            continue
        for token in children_raw.split():
            if not token.isdigit() or int(token) <= 0:
                children_error = ResourceContractError(
                    f"live process {pid} children format is invalid"
                )
                continue
            child_pid = int(token)
            try:
                child = _read_proc_stat_identity(child_pid, proc_root=proc_root)
            except ResourceContractError as error:
                children_error = error
                continue
            if child is not None:
                child_identities.add((child_pid, child[1]))
    second = _read_proc_stat_identity(pid, proc_root=proc_root)
    if second is None or second[:2] != first[:2]:
        return None
    if second[2] == "Z":
        return 0, first[1], ()
    if statm_raw is None:
        detail = "vanished" if isinstance(statm_error, FileNotFoundError) else "unreadable"
        raise ResourceContractError(f"live process {pid} statm {detail}")
    statm_fields = statm_raw.split()
    if len(statm_fields) < 2 or not statm_fields[1].isdigit():
        raise ResourceContractError(f"live process {pid} statm format is invalid")
    if children_error is not None:
        raise ResourceContractError(
            f"live process {pid} children snapshot is invalid"
        ) from children_error
    resident_pages = int(statm_fields[1])
    return resident_pages * page_size_bytes, first[1], tuple(sorted(child_identities))


def _process_tree_rss_bytes(
    root_pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> int:
    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        raise ValueError("process-tree root PID must be a positive integer")
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise ValueError("page size must be positive")
    pending: list[tuple[int, int | None]] = [(root_pid, None)]
    seen: set[tuple[int, int]] = set()
    total_bytes = 0
    while pending:
        pid, expected_starttime = pending.pop()
        sampled = _sample_proc_rss_and_children(
            pid,
            expected_starttime=expected_starttime,
            proc_root=proc_root,
            page_size_bytes=page_size,
        )
        if sampled is None:
            if pid == root_pid:
                raise _ProcessTreeRootExitObserved(root_pid, "vanished") from None
            continue
        rss_bytes, starttime, children = sampled
        identity = (pid, starttime)
        if identity in seen:
            continue
        seen.add(identity)
        total_bytes += rss_bytes
        pending.extend(children)
    return total_bytes


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
