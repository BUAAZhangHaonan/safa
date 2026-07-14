from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Callable, Literal, Mapping, Sequence

from safa.evaluation.r8_arm_contracts import canonical_arm_config_payload
from safa.evaluation.r8_visual_evidence import (
    build_visual_evidence_contract,
    write_contact_sheets,
)
from safa.evaluation.r9_campaign_contracts import (
    derive_visual_arm_pass,
    privacy_delta_cluster_bootstrap,
    validate_gate_contract,
    validate_identity_report,
    write_immutable_contract,
)


AWAITING_VISUAL_REVIEW_EXIT_CODE = 20
PHASE_RESULT_PHASES = ("diagnose", "calibrate", "confirm512", "full")
PHASE_SAMPLE_COUNTS = {
    "diagnose": 18,
    "calibrate": 64,
    "confirm512": 512,
    "full": 2048,
}
EVALUATION_REPAIR_FILENAME = "evaluation_repair_contract.json"
EVALUATION_REPAIR_V2_FILENAME = "evaluation_repair_contract_v2.json"
EVALUATION_REPAIR_V3_FILENAME = "evaluation_repair_contract_v3.json"
FORBIDDEN_DERIVED_INPUT_FIELDS = frozenset(
    {"passed", "severe_count", "severe_failure_count", "failures", "verdict"}
)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_PATH_ONLY_FIXED_ASSET_FIELDS = frozenset(
    {
        "checkpoint",
        "e0_checkpoint",
        "edev_checkpoint",
        "heldout_e1_checkpoint",
        "heldout_e2_checkpoint",
        "vae_path",
        "index",
        "features",
        "sampling_seed",
    }
)
_SEMANTIC_ROW_FIELDS = (
    "sample_id",
    "candidate_cosine",
    "native_cosine",
    "edev_cosine",
    "native_edev_cosine",
    "candidate_nfe",
    "native_nfe",
    "candidate_algorithm_nfe",
    "candidate_diagnostic_nfe",
    "candidate_trace",
    "native_trace",
    "candidate_diagnostic_trace",
    "route_diagnostics",
)
_PAIRED_METRIC_FIELDS = (
    "candidate_e0",
    "native_e0",
    "candidate_edev",
    "native_edev",
    "candidate_niqe",
    "native_niqe",
    "candidate_sharpness",
    "native_sharpness",
)


class PhaseResultsError(ValueError):
    """Raised when R9 phase evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class RunEvidenceSpec:
    logical_run_id: str
    arm_id: str
    family: str
    seed: int
    repeat_index: int | None
    shard_output_dirs: tuple[Path, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.logical_run_id, "logical run ID"),
            (self.arm_id, "arm ID"),
            (self.family, "arm family"),
        ):
            _require_safe_id(value, label)
        _require_nonnegative_int(self.seed, "run seed")
        if self.repeat_index is not None:
            _require_nonnegative_int(self.repeat_index, "repeat index")
        if not self.shard_output_dirs:
            raise PhaseResultsError("run evidence requires at least one shard output")


@dataclass(frozen=True)
class PhaseResultsRequest:
    repo_root: Path
    phase_root: Path
    phase: str
    campaign_id: str
    campaign_runtime_sha256: str
    manifest_contracts_sha256: str
    manifest_path: Path
    manifest_sha256: str
    source_index_path: Path
    source_index_sha256: str
    checkpoint_sha256: str
    bootstrap_seed: int
    runs: tuple[RunEvidenceSpec, ...]
    expected_candidate_arm_ids: tuple[str, ...]
    expected_seeds: tuple[int, ...]
    upstream_gate: Mapping[str, Any] | None = None
    upstream_calibration_selection: Mapping[str, Any] | None = None
    visual_manifest_path: Path | None = None
    visual_manifest_sha256: str | None = None
    confirm_seed: int | None = None
    selection: Mapping[str, Any] | None = None
    heldout_seal: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASE_RESULT_PHASES:
            raise PhaseResultsError(f"phase must be one of {PHASE_RESULT_PHASES!r}")
        _require_safe_id(self.campaign_id, "campaign ID")
        for value, label in (
            (self.campaign_runtime_sha256, "campaign runtime SHA256"),
            (self.manifest_contracts_sha256, "manifest contracts SHA256"),
            (self.manifest_sha256, "manifest SHA256"),
            (self.source_index_sha256, "source index SHA256"),
            (self.checkpoint_sha256, "checkpoint SHA256"),
        ):
            _require_sha256(value, label)
        _require_nonnegative_int(self.bootstrap_seed, "bootstrap seed")
        if not self.runs:
            raise PhaseResultsError("phase results require at least one run")
        if not self.expected_candidate_arm_ids or len(
            set(self.expected_candidate_arm_ids)
        ) != len(self.expected_candidate_arm_ids):
            raise PhaseResultsError(
                "expected candidate arm IDs must be nonempty and unique"
            )
        for arm_id in self.expected_candidate_arm_ids:
            _require_safe_id(arm_id, "expected candidate arm ID")
            if arm_id == "native":
                raise PhaseResultsError("native is not a candidate arm ID")
        if not self.expected_seeds or len(set(self.expected_seeds)) != len(
            self.expected_seeds
        ):
            raise PhaseResultsError("expected seeds must be nonempty and unique")
        for seed in self.expected_seeds:
            _require_nonnegative_int(seed, "expected seed")
        if self.phase == "diagnose" and (
            self.upstream_gate is not None
            or self.upstream_calibration_selection is not None
        ):
            raise PhaseResultsError("diagnose must not bind upstream selection")
        if self.upstream_gate is not None and self.upstream_calibration_selection is not None:
            raise PhaseResultsError("upstream gate and calibration selection are exclusive")
        if self.phase == "confirm512":
            if self.upstream_gate is None and self.upstream_calibration_selection is None:
                raise PhaseResultsError(
                    "confirm512 requires a gate or report-only calibration selection"
                )
        elif self.phase != "diagnose":
            if self.upstream_gate is None or self.upstream_calibration_selection is not None:
                raise PhaseResultsError(f"{self.phase} requires its upstream gate")
        if self.phase == "confirm512":
            _require_nonnegative_int(self.confirm_seed, "confirm512 seed")
        if self.phase == "full":
            if self.visual_manifest_path is None or self.visual_manifest_sha256 is None:
                raise PhaseResultsError(
                    "Full requires the locked full_visual_64 manifest"
                )
            _require_sha256(self.visual_manifest_sha256, "visual manifest SHA256")
            if self.selection is None or self.heldout_seal is None:
                raise PhaseResultsError(
                    "Full requires selection and heldout seal contracts"
                )


@dataclass(frozen=True)
class SampleEvidence:
    sample_id: str
    source: Path
    native: Path
    candidate: Path
    source_sha256: str
    native_sha256: str
    candidate_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or "\0" in self.sample_id
        ):
            raise PhaseResultsError("sample ID must be a non-empty string without NUL")
        for value, label in (
            (self.source_sha256, "source SHA256"),
            (self.native_sha256, "native SHA256"),
            (self.candidate_sha256, "candidate SHA256"),
        ):
            _require_sha256(value, label)


@dataclass(frozen=True)
class QualityEvaluationRequest:
    phase: str
    logical_run_id: str
    arm_id: str
    seed: int
    image_role: Literal["candidate", "native"]
    manifest_path: Path
    source_index_path: Path
    source_index_sha256: str
    samples: tuple[SampleEvidence, ...]
    algorithm_config_sha256: str
    runner_arm_config_sha256: str
    semantic_output_sha256: str
    evidence_binding_sha256: str
    generation_result_set_sha256: str
    per_sample_set_sha256: str


@dataclass(frozen=True)
class ArcFaceEvaluationRequest:
    phase: str
    logical_run_id: str
    arm_id: str
    seed: int
    source_index_path: Path
    source_index_sha256: str
    samples: tuple[SampleEvidence, ...]


@dataclass(frozen=True)
class HeldoutEvaluationRequest:
    phase: str
    arm_id: str
    seed: int
    source_index_path: Path
    source_index_sha256: str
    samples: tuple[SampleEvidence, ...]
    selection: Mapping[str, Any]
    heldout_seal: Mapping[str, Any]


QualityEvaluator = Callable[[QualityEvaluationRequest], Mapping[str, Any]]
ArcFaceEvaluator = Callable[[ArcFaceEvaluationRequest], Sequence[Mapping[str, Any]]]
HeldoutEvaluator = Callable[[HeldoutEvaluationRequest], Mapping[str, Any]]


@dataclass(frozen=True)
class PhaseClosureOutcome:
    status: Literal["needs_generation", "awaiting_visual_review", "complete"]
    phase_results_path: Path | None
    awaiting_path: Path | None
    required_review_count: int
    completed_review_count: int


def resume_phase_results(request: PhaseResultsRequest) -> PhaseClosureOutcome:
    """Resume only from immutable automatic evidence; never rerun an evaluator."""
    validated = _validate_request(request)
    automatic_path = request.phase_root / "automatic_evidence.json"
    if not automatic_path.is_file():
        return PhaseClosureOutcome("needs_generation", None, None, 0, 0)
    automatic = _read_digest_contract(
        automatic_path,
        digest_field="automatic_evidence_sha256",
        contract_type="safa_r9_automatic_phase_evidence_v1",
    )
    _validate_automatic_context(automatic, validated)
    reviews, completed = _load_reviews(automatic)
    awaiting = _awaiting_contract(request, automatic)
    awaiting_path = request.phase_root / "awaiting_visual_review.json"
    write_immutable_contract(
        awaiting_path,
        awaiting,
        digest_field="awaiting_visual_review_sha256",
    )
    if completed != len(automatic["visual_units"]):
        if (request.phase_root / "phase_results.json").exists():
            raise PhaseResultsError(
                "phase_results exists before visual review coverage"
            )
        return PhaseClosureOutcome(
            "awaiting_visual_review",
            None,
            awaiting_path,
            len(automatic["visual_units"]),
            completed,
        )
    phase_results = _build_phase_results(validated, automatic, reviews)
    phase_results_path = request.phase_root / "phase_results.json"
    write_immutable_contract(
        phase_results_path,
        phase_results,
        digest_field="phase_results_sha256",
    )
    return PhaseClosureOutcome(
        "complete",
        phase_results_path,
        awaiting_path,
        len(automatic["visual_units"]),
        completed,
    )


def materialize_phase_results(
    request: PhaseResultsRequest,
    *,
    quality_evaluator: QualityEvaluator | None = None,
    arcface_evaluator: ArcFaceEvaluator | None = None,
    heldout_evaluator: HeldoutEvaluator | None = None,
) -> PhaseClosureOutcome:
    """Aggregate completed generation into immutable evidence, then bound-exit."""
    state = resume_phase_results(request)
    if state.status != "needs_generation":
        return state
    validated = _validate_request(request)
    runs = [_load_run_evidence(validated, spec) for spec in request.runs]
    automatic = _build_automatic_evidence(
        validated,
        runs,
        quality_evaluator=quality_evaluator,
        arcface_evaluator=arcface_evaluator,
        heldout_evaluator=heldout_evaluator,
    )
    write_immutable_contract(
        request.phase_root / "automatic_evidence.json",
        automatic,
        digest_field="automatic_evidence_sha256",
    )
    return resume_phase_results(request)


def submit_visual_review(
    *, evidence_path: Path, decisions_path: Path, output_path: Path
) -> dict[str, Any]:
    """Publish one review with O_EXCL from severe-only ordered decisions."""
    evidence = _load_visual_evidence(Path(evidence_path))
    decisions = _read_json_mapping(Path(decisions_path), "visual review decisions")
    if set(decisions) != {"evidence_contract_sha256", "samples"}:
        raise PhaseResultsError(
            "review input accepts only evidence_contract_sha256 and samples"
        )
    if decisions["evidence_contract_sha256"] != evidence["evidence_contract_sha256"]:
        raise PhaseResultsError("review input does not bind the visual evidence hash")
    _reject_derived_input_fields(decisions)
    rows = decisions.get("samples")
    if not isinstance(rows, list):
        raise PhaseResultsError("review input samples must be a list")
    normalized_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"sample_id", "severe"}:
            raise PhaseResultsError(
                f"review row {index} accepts only sample_id and severe"
            )
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
            raise PhaseResultsError(f"review row {index} sample_id is invalid")
        if not isinstance(row.get("severe"), bool):
            raise PhaseResultsError(f"review row {index} severe must be boolean")
        normalized_rows.append({"sample_id": row["sample_id"], "severe": row["severe"]})
    review_for_derivation = {"samples": normalized_rows}
    derive_visual_arm_pass(
        review_for_derivation,
        evidence,
        severe_limit=len(normalized_rows),
    )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_visual_review_v1",
        "evidence_contract_sha256": evidence["evidence_contract_sha256"],
        "samples": normalized_rows,
    }
    payload["review_sha256"] = _canonical_digest(payload, "review_sha256")
    _write_exclusive_json(Path(output_path), payload)
    return payload


def validate_visual_review(review_path: Path, evidence_path: Path) -> dict[str, Any]:
    evidence = _load_visual_evidence(Path(evidence_path))
    review = _read_digest_contract(
        Path(review_path),
        digest_field="review_sha256",
        contract_type="safa_r9_visual_review_v1",
    )
    if set(review) != {
        "schema_version",
        "contract_type",
        "evidence_contract_sha256",
        "samples",
        "review_sha256",
    }:
        raise PhaseResultsError("visual review fields are not canonical")
    if review["evidence_contract_sha256"] != evidence["evidence_contract_sha256"]:
        raise PhaseResultsError("visual review evidence hash mismatch")
    _reject_derived_input_fields(review)
    derive_visual_arm_pass(
        review,
        evidence,
        severe_limit=int(evidence["sample_count"]),
    )
    return review


def _validate_request(request: PhaseResultsRequest) -> dict[str, Any]:
    repo_root = request.repo_root.resolve()
    phase_root = _contained_path(repo_root, request.phase_root, "phase root")
    manifest_path = _contained_file(repo_root, request.manifest_path, "phase manifest")
    if _sha256_file(manifest_path) != request.manifest_sha256:
        raise PhaseResultsError("phase manifest SHA256 mismatch")
    manifest_rows = _read_jsonl(manifest_path, "phase manifest")
    if len(manifest_rows) != PHASE_SAMPLE_COUNTS[request.phase]:
        raise PhaseResultsError(
            f"{request.phase} manifest must contain exactly "
            f"{PHASE_SAMPLE_COUNTS[request.phase]} samples"
        )
    manifest_ids = _ordered_sample_ids(manifest_rows, "phase manifest")
    source_index_path, source_paths = _load_source_index_contract(
        repo_root, request.source_index_path, request.source_index_sha256
    )
    missing_source_ids = set(manifest_ids) - set(source_paths)
    if missing_source_ids:
        raise PhaseResultsError(
            f"source index is missing phase IDs: {sorted(missing_source_ids)!r}"
        )
    visual_path = None
    visual_rows = None
    if request.phase == "full":
        assert request.visual_manifest_path is not None
        assert request.visual_manifest_sha256 is not None
        visual_path = _contained_file(
            repo_root, request.visual_manifest_path, "visual manifest"
        )
        if _sha256_file(visual_path) != request.visual_manifest_sha256:
            raise PhaseResultsError("visual manifest SHA256 mismatch")
        visual_rows = _read_jsonl(visual_path, "visual manifest")
        visual_ids = _ordered_sample_ids(visual_rows, "visual manifest")
        if len(visual_ids) != 64 or not set(visual_ids) <= set(manifest_ids):
            raise PhaseResultsError(
                "Full visual manifest must contain 64 IDs from full_2048"
            )
    logical_ids = [spec.logical_run_id for spec in request.runs]
    if len(set(logical_ids)) != len(logical_ids):
        raise PhaseResultsError("phase run logical IDs must be unique")
    expected_seeds = tuple(request.expected_seeds)
    if request.phase == "diagnose" and expected_seeds != (1337,):
        raise PhaseResultsError("diagnose seed plan must be exactly (1337,)")
    if request.phase == "calibrate" and expected_seeds != (1337, 2027, 3407):
        raise PhaseResultsError("calibrate seed plan must be exactly (1337,2027,3407)")
    if request.phase == "confirm512" and expected_seeds != (request.confirm_seed,):
        raise PhaseResultsError("confirm512 run plan disagrees with confirm_seed")
    arm_limits = {
        "diagnose": (12, 12),
        "calibrate": (1, 3),
        "confirm512": (1, 2),
        "full": (1, 1),
    }
    minimum, maximum = arm_limits[request.phase]
    if not minimum <= len(request.expected_candidate_arm_ids) <= maximum:
        raise PhaseResultsError(f"{request.phase} candidate arm count is invalid")
    expected_repeats: tuple[int | None, ...] = (
        (0, 1, 2) if request.phase == "diagnose" else (None,)
    )
    expected_native_keys = {
        (seed, repeat) for seed in expected_seeds for repeat in expected_repeats
    }
    native_keys = {
        (spec.seed, spec.repeat_index)
        for spec in request.runs
        if spec.arm_id == "native" and spec.family == "native"
    }
    if native_keys != expected_native_keys or sum(
        spec.arm_id == "native" for spec in request.runs
    ) != len(expected_native_keys):
        raise PhaseResultsError("phase run plan does not contain exact matched natives")
    candidate_specs = [spec for spec in request.runs if spec.arm_id != "native"]
    if not candidate_specs:
        raise PhaseResultsError(f"{request.phase} requires at least one candidate run")
    actual_candidate_keys = set()
    for spec in candidate_specs:
        if (spec.seed, spec.repeat_index) not in native_keys:
            raise PhaseResultsError(
                f"candidate run {spec.logical_run_id!r} has no matched native run"
            )
        key = (spec.arm_id, spec.seed, spec.repeat_index)
        if key in actual_candidate_keys:
            raise PhaseResultsError(
                "phase run plan repeats a candidate arm/seed/repeat"
            )
        actual_candidate_keys.add(key)
    expected_candidate_keys = {
        (arm_id, seed, repeat)
        for arm_id in request.expected_candidate_arm_ids
        for seed in expected_seeds
        for repeat in expected_repeats
    }
    if actual_candidate_keys != expected_candidate_keys:
        raise PhaseResultsError(
            "phase run plan has missing, stale, or extra candidate runs"
        )
    upstream_gate = None
    upstream_calibration_selection = None
    if request.upstream_gate is not None:
        try:
            upstream_gate = validate_gate_contract(request.upstream_gate)
        except ValueError as error:
            raise PhaseResultsError("upstream gate contract is invalid") from error
        expected_upstream_phase = {
            "calibrate": "diagnose",
            "confirm512": "calibrate",
            "full": "confirm512",
        }[request.phase]
        if upstream_gate.get("phase") != expected_upstream_phase or upstream_gate.get(
            "selected_arm_ids"
        ) != list(request.expected_candidate_arm_ids):
            raise PhaseResultsError(
                "phase run plan disagrees with upstream selected arms"
            )
        if request.phase == "full":
            assert request.selection is not None
            if request.selection.get("gate_contract_sha256") != upstream_gate.get(
                "gate_contract_sha256"
            ):
                raise PhaseResultsError("Full selection does not bind upstream gate")
            winner = request.selection.get("winner")
            if (
                not isinstance(winner, Mapping)
                or winner.get("arm_id") != request.expected_candidate_arm_ids[0]
            ):
                raise PhaseResultsError("Full run plan changed the locked winner")
    if request.upstream_calibration_selection is not None:
        if request.phase != "confirm512":
            raise PhaseResultsError(
                "report-only calibration selection is only valid for confirm512"
            )
        from safa.evaluation.r9_calibration_selection_contracts import (
            validate_calibration_report_only_selection_contract,
        )

        try:
            upstream_calibration_selection = (
                validate_calibration_report_only_selection_contract(
                    request.upstream_calibration_selection,
                    repo_root=repo_root,
                )
            )
        except ValueError as error:
            raise PhaseResultsError(
                "upstream calibration selection is invalid"
            ) from error
        selected_ids = [
            str(row["arm_id"])
            for row in upstream_calibration_selection["selected_arms"]
        ]
        if selected_ids != list(request.expected_candidate_arm_ids):
            raise PhaseResultsError(
                "confirm512 run plan disagrees with calibration selection"
            )
    run_plan = _run_plan_payload(request)
    return {
        "request": request,
        "repo_root": repo_root,
        "phase_root": phase_root,
        "manifest_path": manifest_path,
        "manifest_rows": manifest_rows,
        "manifest_ids": manifest_ids,
        "source_index_path": source_index_path,
        "source_paths": source_paths,
        "visual_manifest_path": visual_path,
        "visual_manifest_rows": visual_rows,
        "upstream_gate": upstream_gate,
        "upstream_calibration_selection": upstream_calibration_selection,
        "run_plan": run_plan,
        "run_plan_sha256": _canonical_json_sha256(run_plan),
    }


def _load_run_evidence(
    validated: Mapping[str, Any], spec: RunEvidenceSpec
) -> dict[str, Any]:
    request: PhaseResultsRequest = validated["request"]
    manifest_ids: list[str] = validated["manifest_ids"]
    repo_root: Path = validated["repo_root"]
    manifest_ids: list[str] = validated["manifest_ids"]
    shards: dict[int, dict[str, Any]] = {}
    shard_count: int | None = None
    algorithm_digests: set[str] = set()
    runner_arm_digests: set[str] = set()
    interval_contracts: dict[str, dict[str, Any]] = {}
    for declared_dir in spec.shard_output_dirs:
        output = _contained_path(repo_root, declared_dir, "shard output")
        result_path = _contained_file(
            output, output / "generation_result.json", "generation result"
        )
        run_manifest_path = _contained_file(
            output, output / "run_manifest.json", "run manifest"
        )
        completion_path = _contained_file(
            output, output / "completion.json", "completion"
        )
        per_sample_path = _contained_file(
            output, output / "per_sample.jsonl", "per-sample evidence"
        )
        result = _read_json_mapping(result_path, "generation result")
        if result != _read_json_mapping(run_manifest_path, "run manifest"):
            raise PhaseResultsError("generation result and run manifest differ")
        if result.get("schema_version") != 1 or result.get("status") != "complete":
            raise PhaseResultsError(
                "generation result is not complete schema_version=1"
            )
        checkpoint = result.get("checkpoint")
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("sha256") != request.checkpoint_sha256
        ):
            raise PhaseResultsError("generation result checkpoint mismatch")
        config = result.get("config")
        if not isinstance(config, Mapping):
            raise PhaseResultsError("generation result config must be a mapping")
        for field, expected in (
            ("r9_campaign_runtime_sha256", request.campaign_runtime_sha256),
            ("r9_manifest_contracts_sha256", request.manifest_contracts_sha256),
            ("r9_phase_manifest_sha256", request.manifest_sha256),
            ("sampling_seed", spec.seed),
        ):
            if config.get(field) != expected:
                raise PhaseResultsError(
                    f"generation config {field} disagrees with phase request"
                )
        algorithm_digest = _algorithm_config_digest(config, request.checkpoint_sha256)
        algorithm_digests.add(algorithm_digest)
        interval_contract = config.get("r9_guidance_interval_contract")
        if isinstance(interval_contract, Mapping):
            normalized_interval_contract = _json_roundtrip(
                interval_contract, "R9 guidance interval contract"
            )
            interval_contracts[_canonical_json_sha256(normalized_interval_contract)] = (
                normalized_interval_contract
            )
        runner_arm_digests.add(
            _require_sha256(result.get("arm_config_sha256"), "runner arm SHA256")
        )
        shard = result.get("shard")
        if not isinstance(shard, Mapping):
            raise PhaseResultsError("generation result shard must be a mapping")
        shard_index = _strict_int(shard.get("index"), "shard index")
        current_count = _strict_int(shard.get("count"), "shard count")
        if current_count <= 0 or not 0 <= shard_index < current_count:
            raise PhaseResultsError("generation shard index/count is invalid")
        if shard_count is None:
            shard_count = current_count
        elif shard_count != current_count:
            raise PhaseResultsError("logical run shard counts disagree")
        if shard_index in shards:
            raise PhaseResultsError("logical run repeats a shard index")
        expected_ids = manifest_ids[shard_index::current_count]
        rows = _read_jsonl(per_sample_path, "per-sample evidence")
        if _ordered_sample_ids(rows, "per-sample evidence") != expected_ids:
            raise PhaseResultsError(
                "shard per-sample rows do not match manifest slicing"
            )
        sample_rows = []
        for row in rows:
            sample_id = str(row["sample_id"])
            source_value = Path(str(row.get("source", "")))
            source = (
                source_value.resolve()
                if source_value.is_absolute()
                else (repo_root / source_value).resolve()
            )
            if source != validated["source_paths"][sample_id]:
                raise PhaseResultsError(
                    f"source image for {sample_id!r} disagrees with locked source index"
                )
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"source image is not a regular file: {source}")
            generated = _bound_output_file(
                repo_root,
                output,
                Path(str(row.get("generated", ""))),
                "generated image",
            )
            native = _bound_output_file(
                repo_root,
                output,
                Path(str(row.get("native", ""))),
                "native image",
            )
            _assert_finite_json(row, f"per-sample row {sample_id}")
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "source": str(source),
                    "native": str(native),
                    "candidate": str(generated),
                    "source_sha256": _sha256_file(source),
                    "native_sha256": _sha256_file(native),
                    "candidate_sha256": _sha256_file(generated),
                    "metrics": {
                        field: row[field]
                        for field in _SEMANTIC_ROW_FIELDS
                        if field in row
                    },
                }
            )
        completion = _read_json_mapping(completion_path, "completion")
        if completion.get("status") != "complete":
            raise PhaseResultsError("worker completion is not complete")
        shards[shard_index] = {
            "shard_index": shard_index,
            "shard_count": current_count,
            "output_dir": str(output),
            "generation_result_path": str(result_path),
            "generation_result_sha256": _sha256_file(result_path),
            "run_manifest_sha256": _sha256_file(run_manifest_path),
            "completion_sha256": _sha256_file(completion_path),
            "per_sample_sha256": _sha256_file(per_sample_path),
            "rows": sample_rows,
        }
    if shard_count is None or set(shards) != set(range(shard_count)):
        raise PhaseResultsError("logical run does not cover every registered shard")
    if len(algorithm_digests) != 1 or len(runner_arm_digests) != 1:
        raise PhaseResultsError("logical run shards changed their config digest")
    if len(interval_contracts) > 1:
        raise PhaseResultsError("logical run shards changed their interval contract")
    rows_by_id = {
        row["sample_id"]: row for shard in shards.values() for row in shard["rows"]
    }
    if set(rows_by_id) != set(manifest_ids) or len(rows_by_id) != len(manifest_ids):
        raise PhaseResultsError("logical run does not cover the phase manifest exactly")
    ordered_rows = [rows_by_id[sample_id] for sample_id in manifest_ids]
    semantic_rows = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "native_sha256": row["native_sha256"],
            "candidate_sha256": row["candidate_sha256"],
            "metrics": row["metrics"],
        }
        for row in ordered_rows
    ]
    output_contract = {
        "logical_run_id": spec.logical_run_id,
        "algorithm_config_sha256": next(iter(algorithm_digests)),
        "runner_arm_config_sha256": next(iter(runner_arm_digests)),
        "interval_contract": (
            next(iter(interval_contracts.values())) if interval_contracts else None
        ),
        "shards": [
            {key: value for key, value in shards[index].items() if key != "rows"}
            for index in range(shard_count)
        ],
        "images": [
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "native": row["native"],
                "candidate": row["candidate"],
                "source_sha256": row["source_sha256"],
                "native_sha256": row["native_sha256"],
                "candidate_sha256": row["candidate_sha256"],
            }
            for row in ordered_rows
        ],
    }
    semantic_payload = {
        "algorithm_config_sha256": next(iter(algorithm_digests)),
        "rows": semantic_rows,
    }
    return {
        "logical_run_id": spec.logical_run_id,
        "arm_id": spec.arm_id,
        "family": spec.family,
        "seed": spec.seed,
        "repeat_index": spec.repeat_index,
        "algorithm_config_sha256": next(iter(algorithm_digests)),
        "runner_arm_config_sha256": next(iter(runner_arm_digests)),
        "interval_contract": output_contract["interval_contract"],
        "semantic_run_sha256": _canonical_json_sha256(
            {"contract_type": "safa_r9_semantic_run_v1", **semantic_payload}
        ),
        "output_sha256": _canonical_json_sha256(
            {"contract_type": "safa_r9_semantic_output_v1", **semantic_payload}
        ),
        "evidence_binding_sha256": _canonical_json_sha256(output_contract),
        "output_contract": output_contract,
        "rows": ordered_rows,
    }


def _build_automatic_evidence(
    validated: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    *,
    quality_evaluator: QualityEvaluator | None,
    arcface_evaluator: ArcFaceEvaluator | None,
    heldout_evaluator: HeldoutEvaluator | None,
) -> dict[str, Any]:
    request: PhaseResultsRequest = validated["request"]
    manifest_ids = list(validated["manifest_ids"])
    native_by_key: dict[tuple[int, int | None], Mapping[str, Any]] = {}
    candidates = []
    for run in runs:
        key = (int(run["seed"]), run["repeat_index"])
        if run["arm_id"] == "native":
            if key in native_by_key:
                raise PhaseResultsError("phase repeats a matched native run")
            native_by_key[key] = run
        else:
            candidates.append(run)
    for candidate in candidates:
        key = (int(candidate["seed"]), candidate["repeat_index"])
        _validate_matched_native(candidate, native_by_key[key])
    visual_units = []
    automatic_arms = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["arm_id"]), []).append(candidate)
    for arm_id in sorted(grouped):
        arm_runs = sorted(
            grouped[arm_id],
            key=lambda row: (
                -1 if row["repeat_index"] is None else int(row["repeat_index"]),
                int(row["seed"]),
            ),
        )
        families = {str(row["family"]) for row in arm_runs}
        config_digests = {str(row["algorithm_config_sha256"]) for row in arm_runs}
        if len(families) != 1 or len(config_digests) != 1:
            raise PhaseResultsError("arm runs changed family or algorithm config")
        auto_arm: dict[str, Any] = {
            "arm_id": arm_id,
            "family": next(iter(families)),
            "config_sha256": next(iter(config_digests)),
            "automatic_output_sha256": _canonical_json_sha256(
                {"run_outputs": [row["output_sha256"] for row in arm_runs]}
            ),
        }
        if request.phase == "diagnose":
            repeat_rows = []
            for run in arm_runs:
                diagnostics_contract = validate_interval_diagnostics(
                    run["rows"], run.get("interval_contract")
                )
                unit = _materialize_visual_unit(validated, run)
                visual_units.append(unit)
                repeat_rows.append(
                    {
                        "repeat_index": run["repeat_index"],
                        "run_sha256": run["semantic_run_sha256"],
                        "e0_mean": _mean_metric(run["rows"], "candidate_cosine"),
                        "edev_delta_vs_matched_native": (
                            _mean_metric(run["rows"], "edev_cosine")
                            - _mean_metric(run["rows"], "native_edev_cosine")
                        ),
                        "diagnostics_finite": True,
                        "diagnostics_contract_sha256": diagnostics_contract[
                            "diagnostics_contract_sha256"
                        ],
                        "visual_unit_id": unit["unit_id"],
                    }
                )
            auto_arm["repeat_results"] = repeat_rows
            auto_arm["evaluator_evidence_sha256"] = _canonical_json_sha256(
                {
                    "diagnostics": [
                        row["diagnostics_contract_sha256"] for row in repeat_rows
                    ]
                }
            )
        else:
            if quality_evaluator is None or arcface_evaluator is None:
                raise PhaseResultsError(
                    f"{request.phase} requires explicit quality and ArcFace evaluators"
                )
            seed_rows = []
            privacy_rows = []
            paired_metric_rows = []
            all_exact_one = True
            for run in arm_runs:
                native = native_by_key[(int(run["seed"]), run["repeat_index"])]
                candidate_samples = _sample_evidence(run)
                native_samples = _sample_evidence(native)
                candidate_quality = _evaluate_quality(
                    request,
                    run,
                    candidate_samples,
                    "candidate",
                    quality_evaluator,
                )
                native_quality = _evaluate_quality(
                    request,
                    native,
                    native_samples,
                    "native",
                    quality_evaluator,
                )
                paired_metric_rows.extend(
                    _paired_metric_rows(
                        run,
                        candidate_quality=candidate_quality,
                        native_quality=native_quality,
                        manifest_ids=manifest_ids,
                    )
                )
                arcface = _evaluate_arcface(
                    request, run, candidate_samples, arcface_evaluator
                )
                exact_one = bool(arcface["exact_one"])
                all_exact_one = all_exact_one and exact_one
                if exact_one:
                    privacy_rows.extend(
                        {
                            "sample_id": row["sample_id"],
                            "seed": int(run["seed"]),
                            "source_candidate_cosine": row["source_candidate_cosine"],
                            "source_native_cosine": row["source_native_cosine"],
                        }
                        for row in arcface["rows"]
                    )
                unit = _materialize_visual_unit(validated, run)
                visual_units.append(unit)
                seed_rows.append(
                    {
                        "seed": int(run["seed"]),
                        "fid": candidate_quality["fid"],
                        "native_fid": native_quality["fid"],
                        "kid": candidate_quality["kid"],
                        "native_kid": native_quality["kid"],
                        "niqe": candidate_quality["niqe"],
                        "native_niqe": native_quality["niqe"],
                        "sharpness": candidate_quality["sharpness"],
                        "native_sharpness": native_quality["sharpness"],
                        "e0": _mean_metric(run["rows"], "candidate_cosine"),
                        "delta_e0": (
                            _mean_metric(run["rows"], "candidate_cosine")
                            - _mean_metric(run["rows"], "native_cosine")
                        ),
                        "delta_edev": (
                            _mean_metric(run["rows"], "edev_cosine")
                            - _mean_metric(run["rows"], "native_edev_cosine")
                        ),
                        "arcface_exact_one": exact_one,
                        "arcface_summary": {
                            key: arcface[key]
                            for key in (
                                "source_exact_one_count",
                                "native_exact_one_count",
                                "candidate_exact_one_count",
                                "paired_exact_one_count",
                                "failure_sample_ids",
                            )
                        },
                        "quality_evidence_sha256": candidate_quality[
                            "quality_evidence_sha256"
                        ],
                        "quality_raw_evidence_path": candidate_quality[
                            "raw_evidence_path"
                        ],
                        "quality_raw_evidence_sha256": candidate_quality[
                            "raw_evidence_sha256"
                        ],
                        "native_quality_evidence_sha256": native_quality[
                            "quality_evidence_sha256"
                        ],
                        "native_quality_raw_evidence_path": native_quality[
                            "raw_evidence_path"
                        ],
                        "native_quality_raw_evidence_sha256": native_quality[
                            "raw_evidence_sha256"
                        ],
                        "arcface_evidence_sha256": arcface["arcface_evidence_sha256"],
                        "arcface_raw_evidence_path": arcface["raw_evidence_path"],
                        "arcface_raw_evidence_sha256": arcface["raw_evidence_sha256"],
                        "visual_unit_id": unit["unit_id"],
                    }
                )
            if not all_exact_one:
                privacy_rows = []
                privacy_bootstrap = None
            else:
                privacy_bootstrap = privacy_delta_cluster_bootstrap(
                    privacy_rows,
                    expected_seeds=tuple(row["seed"] for row in seed_rows),
                    bootstrap_seed=request.bootstrap_seed,
                )
            paired_metrics = _build_paired_metric_rows_contract(
                paired_metric_rows,
                manifest_ids=manifest_ids,
                expected_seeds=tuple(int(row["seed"]) for row in seed_rows),
            )
            auto_arm.update(
                {
                    "seed_results": seed_rows,
                    "privacy_rows": privacy_rows,
                    "privacy_bootstrap": privacy_bootstrap,
                    "paired_metric_rows": paired_metrics,
                    "evaluator_evidence_sha256": _canonical_json_sha256(
                        {
                            "seeds": [
                                {
                                    "seed": row["seed"],
                                    "quality": row["quality_evidence_sha256"],
                                    "quality_raw": row["quality_raw_evidence_sha256"],
                                    "native_quality": row[
                                        "native_quality_evidence_sha256"
                                    ],
                                    "native_quality_raw": row[
                                        "native_quality_raw_evidence_sha256"
                                    ],
                                    "arcface": row["arcface_evidence_sha256"],
                                    "arcface_raw": row["arcface_raw_evidence_sha256"],
                                }
                                for row in seed_rows
                            ],
                            "paired_metric_rows": paired_metrics[
                                "paired_metric_rows_sha256"
                            ],
                        }
                    ),
                }
            )
        automatic_arms.append(auto_arm)
    heldout = None
    if request.phase == "full":
        if len(automatic_arms) != 1 or len(candidates) != 1:
            raise PhaseResultsError("Full requires exactly one locked winner run")
        if heldout_evaluator is None:
            raise PhaseResultsError("Full requires an explicit heldout evaluator")
        heldout = _materialize_heldout(
            validated,
            candidates[0],
            heldout_evaluator,
            automatic_arms[0],
        )
        automatic_arms[0]["heldout_evidence_sha256"] = heldout[
            "heldout_raw_evidence_sha256"
        ]
        automatic_arms[0]["evaluator_evidence_sha256"] = _canonical_json_sha256(
            {
                "quality_arcface": automatic_arms[0]["evaluator_evidence_sha256"],
                "heldout": heldout["heldout_raw_evidence_sha256"],
            }
        )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_automatic_phase_evidence_v1",
        "phase": request.phase,
        "campaign_id": request.campaign_id,
        "context": _request_context(request),
        "run_plan": validated["run_plan"],
        "run_plan_sha256": validated["run_plan_sha256"],
        "manifest": {
            "path": str(validated["manifest_path"]),
            "sha256": request.manifest_sha256,
            "sample_count": len(manifest_ids),
            "ordered_sample_id_sha256": _sample_id_digest(manifest_ids),
        },
        "source_index": {
            "path": str(validated["source_index_path"]),
            "sha256": request.source_index_sha256,
        },
        "visual_manifest": (
            None
            if validated["visual_manifest_path"] is None
            else {
                "path": str(validated["visual_manifest_path"]),
                "sha256": request.visual_manifest_sha256,
                "sample_count": 64,
                "ordered_sample_id_sha256": _sample_id_digest(
                    _ordered_sample_ids(
                        validated["visual_manifest_rows"], "visual manifest"
                    )
                ),
            }
        ),
        "runs": [
            {key: value for key, value in run.items() if key not in {"rows"}}
            for run in sorted(runs, key=lambda row: str(row["logical_run_id"]))
        ],
        "arms": automatic_arms,
        "visual_units": sorted(visual_units, key=lambda row: row["unit_id"]),
        "heldout": heldout,
    }
    payload["automatic_evidence_sha256"] = _canonical_digest(
        payload, "automatic_evidence_sha256"
    )
    return payload


def _sample_evidence(run: Mapping[str, Any]) -> tuple[SampleEvidence, ...]:
    return tuple(
        SampleEvidence(
            sample_id=str(row["sample_id"]),
            source=Path(str(row["source"])),
            native=Path(str(row["native"])),
            candidate=Path(str(row["candidate"])),
            source_sha256=str(row["source_sha256"]),
            native_sha256=str(row["native_sha256"]),
            candidate_sha256=str(row["candidate_sha256"]),
        )
        for row in run["rows"]
    )


def _evaluate_quality(
    request: PhaseResultsRequest,
    run: Mapping[str, Any],
    samples: tuple[SampleEvidence, ...],
    role: Literal["candidate", "native"],
    evaluator: QualityEvaluator,
) -> dict[str, Any]:
    output_contract = run.get("output_contract")
    if not isinstance(output_contract, Mapping):
        raise PhaseResultsError("quality run lacks its output evidence contract")
    shards = output_contract.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PhaseResultsError("quality run lacks shard evidence")
    generation_result_set_sha256 = _canonical_json_sha256(
        {"generation_results": [shard["generation_result_sha256"] for shard in shards]}
    )
    per_sample_set_sha256 = _canonical_json_sha256(
        {"per_sample": [shard["per_sample_sha256"] for shard in shards]}
    )
    real_asset_manifest_sha256 = _asset_manifest_digest(samples, "source")
    generated_asset_manifest_sha256 = _asset_manifest_digest(samples, role)
    expected_binding = {
        "schema_version": 1,
        "algorithm_config_sha256": run["algorithm_config_sha256"],
        "runner_arm_config_sha256": run["runner_arm_config_sha256"],
        "semantic_output_sha256": run["output_sha256"],
        "evidence_binding_sha256": run["evidence_binding_sha256"],
        "generation_result_set_sha256": generation_result_set_sha256,
        "per_sample_set_sha256": per_sample_set_sha256,
        "manifest_sha256": request.manifest_sha256,
        "source_index_sha256": request.source_index_sha256,
        "ordered_sample_id_sha256": _sample_id_digest(
            [sample.sample_id for sample in samples]
        ),
        "real_asset_manifest_sha256": real_asset_manifest_sha256,
        "generated_asset_manifest_sha256": generated_asset_manifest_sha256,
    }
    raw = evaluator(
        QualityEvaluationRequest(
            phase=request.phase,
            logical_run_id=str(run["logical_run_id"]),
            arm_id=str(run["arm_id"]),
            seed=int(run["seed"]),
            image_role=role,
            manifest_path=request.manifest_path,
            source_index_path=request.source_index_path,
            source_index_sha256=request.source_index_sha256,
            samples=samples,
            algorithm_config_sha256=str(run["algorithm_config_sha256"]),
            runner_arm_config_sha256=str(run["runner_arm_config_sha256"]),
            semantic_output_sha256=str(run["output_sha256"]),
            evidence_binding_sha256=str(run["evidence_binding_sha256"]),
            generation_result_set_sha256=generation_result_set_sha256,
            per_sample_set_sha256=per_sample_set_sha256,
        )
    )
    if not isinstance(raw, Mapping):
        raise PhaseResultsError("quality evaluator must return a mapping")
    _reject_derived_input_fields(raw)
    metrics = raw.get("metrics")
    if metrics != ["fid", "kid", "niqe", "sharpness"]:
        raise PhaseResultsError("quality evaluator must run fid,kid,niqe,sharpness")
    if raw.get("num_generated") != len(samples):
        raise PhaseResultsError("quality result image count mismatch")
    if raw.get("num_real") != len(samples):
        raise PhaseResultsError("quality result real image count mismatch")
    if raw.get("sample_id_count") != len(samples) or raw.get(
        "sample_id_sha256"
    ) != _sample_id_digest([sample.sample_id for sample in samples]):
        raise PhaseResultsError("quality result sample-ID coverage mismatch")
    if raw.get("r9_evidence_binding") != expected_binding:
        raise PhaseResultsError("quality result R9 evidence binding mismatch")
    contract = raw.get("quality_contract")
    if not isinstance(contract, Mapping):
        raise PhaseResultsError("quality result lacks its quality_contract")
    if contract.get("sample_id_manifest_sha256") != request.manifest_sha256:
        raise PhaseResultsError("quality result manifest binding mismatch")
    if (
        contract.get("schema_version") != 1
        or contract.get("metrics") != ["fid", "kid", "niqe", "sharpness"]
        or contract.get("real_asset_manifest_sha256") != real_asset_manifest_sha256
        or contract.get("generated_asset_manifest_sha256")
        != generated_asset_manifest_sha256
    ):
        raise PhaseResultsError("quality result asset contract mismatch")
    iqa = raw.get("iqa")
    sharpness = raw.get("sharpness")
    if not isinstance(iqa, Mapping) or iqa.get("method") != "niqe":
        raise PhaseResultsError("quality result must contain NIQE")
    if (
        not isinstance(sharpness, Mapping)
        or sharpness.get("definition") != "grayscale_laplacian_variance"
    ):
        raise PhaseResultsError("quality result must contain registered Sharpness")
    niqe_mean = _finite_float(iqa.get("mean"), "NIQE")
    sharpness_mean = _finite_float(sharpness.get("mean"), "Sharpness")
    per_sample_metrics = _validate_quality_per_sample_metrics(
        raw.get("per_sample_metrics"),
        sample_ids=[sample.sample_id for sample in samples],
        niqe_mean=niqe_mean,
        sharpness_mean=sharpness_mean,
    )
    raw_contract = {
        "schema_version": 1,
        "contract_type": "safa_r9_quality_raw_evidence_v1",
        "phase": request.phase,
        "logical_run_id": run["logical_run_id"],
        "image_role": role,
        "r9_evidence_binding": expected_binding,
        "raw": _json_roundtrip(raw, "quality raw evidence"),
    }
    raw_contract["quality_raw_evidence_sha256"] = _canonical_digest(
        raw_contract, "quality_raw_evidence_sha256"
    )
    raw_path = (
        _evaluator_evidence_root(request)
        / "quality"
        / f"{run['logical_run_id']}__{role}.json"
    )
    write_immutable_contract(
        raw_path,
        raw_contract,
        digest_field="quality_raw_evidence_sha256",
    )
    normalized = {
        "fid": _finite_float(raw.get("fid"), "FID"),
        "kid": _finite_float(raw.get("kid_mean"), "KID"),
        "niqe": niqe_mean,
        "sharpness": sharpness_mean,
        "per_sample_metrics": per_sample_metrics,
        "raw_evidence_path": str(raw_path),
        "raw_evidence_sha256": raw_contract["quality_raw_evidence_sha256"],
    }
    normalized["quality_evidence_sha256"] = _canonical_json_sha256(
        {key: value for key, value in normalized.items() if key != "raw_evidence_path"}
    )
    return normalized


def _evaluator_evidence_root(request: PhaseResultsRequest) -> Path:
    repo_root = request.repo_root.resolve()
    repair_candidates = (
        (
            EVALUATION_REPAIR_V3_FILENAME,
            "safa_r9_evaluation_repair_contract_v3",
        ),
        (
            EVALUATION_REPAIR_V2_FILENAME,
            "safa_r9_evaluation_repair_contract_v2",
        ),
        (EVALUATION_REPAIR_FILENAME, "safa_r9_evaluation_repair_contract_v1"),
    )
    for filename, contract_type in repair_candidates:
        path = _contained_path(
            repo_root,
            request.phase_root / filename,
            "evaluation repair contract",
        )
        if not path.exists():
            continue
        repair = _read_digest_contract(
            path,
            digest_field="repair_contract_sha256",
            contract_type=contract_type,
        )
        if repair.get("campaign_id") != request.campaign_id or repair.get(
            "phase"
        ) != request.phase:
            raise PhaseResultsError("evaluation repair campaign or phase mismatch")
        return _contained_path(
            repo_root,
            request.phase_root.parent
            / "evaluation_repairs"
            / str(repair["repair_contract_sha256"])
            / request.phase
            / "evaluator_evidence",
            "repair evaluator evidence root",
        )
    return _contained_path(
        repo_root,
        request.phase_root / "evaluator_evidence",
        "evaluator evidence root",
    )


def _validate_quality_per_sample_metrics(
    value: Any,
    *,
    sample_ids: Sequence[str],
    niqe_mean: float,
    sharpness_mean: float,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseResultsError("quality result lacks per_sample_metrics")
    required = {
        "schema_version",
        "contract_type",
        "sample_count",
        "ordered_sample_id_sha256",
        "metric_fields",
        "rows",
        "per_sample_metrics_sha256",
    }
    if set(value) != required:
        raise PhaseResultsError("per_sample_metrics fields are not canonical")
    _assert_finite_json(value, "per_sample_metrics")
    if (
        _strict_int(value.get("schema_version"), "per_sample_metrics schema") != 1
        or value.get("contract_type")
        != "safa_r9_quality_per_sample_metrics_v1"
    ):
        raise PhaseResultsError("per_sample_metrics contract identity mismatch")
    digest = _require_sha256(
        value.get("per_sample_metrics_sha256"),
        "per_sample_metrics_sha256",
    )
    if digest != _canonical_utf8_digest(value, "per_sample_metrics_sha256"):
        raise PhaseResultsError("per_sample_metrics_sha256 mismatch")
    expected_ids = list(sample_ids)
    if _strict_int(value.get("sample_count"), "per_sample_metrics count") != len(
        expected_ids
    ):
        raise PhaseResultsError("per_sample_metrics count mismatch")
    if value.get("ordered_sample_id_sha256") != _sample_id_digest(expected_ids):
        raise PhaseResultsError("per_sample_metrics ordered sample digest mismatch")
    if value.get("metric_fields") != ["niqe", "sharpness"]:
        raise PhaseResultsError("per_sample_metrics metric fields mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise PhaseResultsError("per_sample_metrics rows do not cover the manifest")
    observed_ids = []
    observed_niqe = []
    observed_sharpness = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "niqe",
            "sharpness",
        }:
            raise PhaseResultsError(
                f"per_sample_metrics row {index} fields are not canonical"
            )
        observed_ids.append(row.get("sample_id"))
        observed_niqe.append(
            _finite_float(row.get("niqe"), "per-sample NIQE")
        )
        observed_sharpness.append(
            _finite_float(row.get("sharpness"), "per-sample Sharpness")
        )
    if observed_ids != expected_ids:
        raise PhaseResultsError("per_sample_metrics rows violate manifest order")
    if statistics.fmean(observed_niqe) != niqe_mean:
        raise PhaseResultsError("per-sample NIQE summary mismatch")
    if statistics.fmean(observed_sharpness) != sharpness_mean:
        raise PhaseResultsError("per-sample Sharpness summary mismatch")
    return _json_roundtrip(value, "per_sample_metrics")


def _paired_metric_rows(
    run: Mapping[str, Any],
    *,
    candidate_quality: Mapping[str, Any],
    native_quality: Mapping[str, Any],
    manifest_ids: Sequence[str],
) -> list[dict[str, Any]]:
    candidate_contract = candidate_quality.get("per_sample_metrics")
    native_contract = native_quality.get("per_sample_metrics")
    if not isinstance(candidate_contract, Mapping) or not isinstance(
        native_contract, Mapping
    ):
        raise PhaseResultsError("quality per-sample contracts are unavailable")
    candidate_rows = candidate_contract.get("rows")
    native_rows = native_contract.get("rows")
    if not isinstance(candidate_rows, list) or not isinstance(native_rows, list):
        raise PhaseResultsError("quality per-sample rows are unavailable for pairing")
    if [row.get("sample_id") for row in candidate_rows] != list(manifest_ids):
        raise PhaseResultsError("candidate quality rows violate manifest order")
    if [row.get("sample_id") for row in native_rows] != list(manifest_ids):
        raise PhaseResultsError("native quality rows violate manifest order")
    run_rows = run.get("rows")
    if not isinstance(run_rows, list) or [
        row.get("sample_id") for row in run_rows
    ] != list(manifest_ids):
        raise PhaseResultsError("generation rows violate manifest order")
    seed = _strict_int(run.get("seed"), "paired metric seed")
    rows = []
    for sample_id, generation, candidate, native in zip(
        manifest_ids, run_rows, candidate_rows, native_rows, strict=True
    ):
        metrics = generation.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PhaseResultsError("generation metrics are unavailable for pairing")
        rows.append(
            {
                "sample_id": sample_id,
                "seed": seed,
                "candidate_e0": _cosine(
                    metrics.get("candidate_cosine"), "candidate E0 cosine"
                ),
                "native_e0": _cosine(
                    metrics.get("native_cosine"), "native E0 cosine"
                ),
                "candidate_edev": _cosine(
                    metrics.get("edev_cosine"), "candidate Edev cosine"
                ),
                "native_edev": _cosine(
                    metrics.get("native_edev_cosine"), "native Edev cosine"
                ),
                "candidate_niqe": _finite_float(
                    candidate.get("niqe"), "candidate NIQE"
                ),
                "native_niqe": _finite_float(native.get("niqe"), "native NIQE"),
                "candidate_sharpness": _finite_float(
                    candidate.get("sharpness"), "candidate Sharpness"
                ),
                "native_sharpness": _finite_float(
                    native.get("sharpness"), "native Sharpness"
                ),
            }
        )
    return rows


def _build_paired_metric_rows_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_ids: Sequence[str],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    seeds = tuple(
        sorted(_strict_int(seed, "paired metric seed") for seed in expected_seeds)
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise PhaseResultsError("paired metric seeds must be unique and non-empty")
    expected_keys = {"sample_id", "seed", *_PAIRED_METRIC_FIELDS}
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise PhaseResultsError(
                f"paired metric row {index} fields are not canonical"
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise PhaseResultsError(f"paired metric row {index} sample_id is invalid")
        normalized.append(
            {
                "sample_id": sample_id,
                "seed": _strict_int(
                    row.get("seed"), f"paired metric row {index} seed"
                ),
                **{
                    field: _finite_float(
                        row.get(field), f"paired metric row {index} {field}"
                    )
                    for field in _PAIRED_METRIC_FIELDS
                },
            }
        )
    expected_order = [
        (sample_id, seed) for seed in seeds for sample_id in manifest_ids
    ]
    if [(row["sample_id"], row["seed"]) for row in normalized] != expected_order:
        raise PhaseResultsError("paired metric rows violate seed-major manifest order")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_paired_metric_rows_v1",
        "direction": "candidate_minus_native",
        "seeds": list(seeds),
        "sample_count": len(manifest_ids),
        "observation_count": len(normalized),
        "ordered_sample_id_sha256": _sample_id_digest(manifest_ids),
        "metric_fields": list(_PAIRED_METRIC_FIELDS),
        "rows": normalized,
    }
    payload["paired_metric_rows_sha256"] = _canonical_digest(
        payload, "paired_metric_rows_sha256"
    )
    return payload


def _evaluate_arcface(
    request: PhaseResultsRequest,
    run: Mapping[str, Any],
    samples: tuple[SampleEvidence, ...],
    evaluator: ArcFaceEvaluator,
) -> dict[str, Any]:
    raw_rows = evaluator(
        ArcFaceEvaluationRequest(
            phase=request.phase,
            logical_run_id=str(run["logical_run_id"]),
            arm_id=str(run["arm_id"]),
            seed=int(run["seed"]),
            source_index_path=request.source_index_path,
            source_index_sha256=request.source_index_sha256,
            samples=samples,
        )
    )
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise PhaseResultsError("ArcFace evaluator must return ordered rows")
    expected_ids = [sample.sample_id for sample in samples]
    normalized = []
    exact_one = True
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise PhaseResultsError(f"ArcFace row {index} must be a mapping")
        _reject_derived_input_fields(row)
        exact_keys = {
            "sample_id",
            "source_face_count",
            "native_face_count",
            "candidate_face_count",
            "source_native_cosine",
            "source_candidate_cosine",
        }
        counts = {
            field: _require_nonnegative_int(row.get(field), field)
            for field in (
                "source_face_count",
                "native_face_count",
                "candidate_face_count",
            )
        }
        row_exact = all(value == 1 for value in counts.values())
        exact_one = exact_one and row_exact
        if row_exact:
            if set(row) != exact_keys:
                raise PhaseResultsError(
                    "exact-one ArcFace row fields are not canonical"
                )
            native_cosine = _cosine(row.get("source_native_cosine"), "ArcFace native")
            candidate_cosine = _cosine(
                row.get("source_candidate_cosine"), "ArcFace candidate"
            )
        else:
            if set(row) != exact_keys - {
                "source_native_cosine",
                "source_candidate_cosine",
            }:
                raise PhaseResultsError(
                    "non-exact-one ArcFace row must omit unavailable cosines"
                )
            native_cosine = None
            candidate_cosine = None
        normalized.append(
            {
                "sample_id": row.get("sample_id"),
                **counts,
                "source_native_cosine": native_cosine,
                "source_candidate_cosine": candidate_cosine,
            }
        )
    if [row["sample_id"] for row in normalized] != expected_ids:
        raise PhaseResultsError("ArcFace rows must exactly follow manifest order")
    failure_sample_ids = [
        row["sample_id"]
        for row in normalized
        if not all(
            row[field] == 1
            for field in (
                "source_face_count",
                "native_face_count",
                "candidate_face_count",
            )
        )
    ]
    payload = {
        "source_index_sha256": request.source_index_sha256,
        "source_asset_manifest_sha256": _asset_manifest_digest(samples, "source"),
        "native_asset_manifest_sha256": _asset_manifest_digest(samples, "native"),
        "candidate_asset_manifest_sha256": _asset_manifest_digest(samples, "candidate"),
        "exact_one": exact_one,
        "source_exact_one_count": sum(
            row["source_face_count"] == 1 for row in normalized
        ),
        "native_exact_one_count": sum(
            row["native_face_count"] == 1 for row in normalized
        ),
        "candidate_exact_one_count": sum(
            row["candidate_face_count"] == 1 for row in normalized
        ),
        "paired_exact_one_count": len(normalized) - len(failure_sample_ids),
        "failure_sample_ids": failure_sample_ids,
        "rows": normalized,
    }
    payload["arcface_evidence_sha256"] = _canonical_json_sha256(payload)
    raw_contract = {
        "schema_version": 1,
        "contract_type": "safa_r9_arcface_raw_evidence_v1",
        "phase": request.phase,
        "logical_run_id": run["logical_run_id"],
        "algorithm_config_sha256": run["algorithm_config_sha256"],
        "semantic_output_sha256": run["output_sha256"],
        "arcface": dict(payload),
    }
    raw_contract["arcface_raw_evidence_sha256"] = _canonical_digest(
        raw_contract, "arcface_raw_evidence_sha256"
    )
    raw_path = (
        _evaluator_evidence_root(request)
        / "arcface"
        / f"{run['logical_run_id']}.json"
    )
    write_immutable_contract(
        raw_path,
        raw_contract,
        digest_field="arcface_raw_evidence_sha256",
    )
    payload["raw_evidence_path"] = str(raw_path)
    payload["raw_evidence_sha256"] = raw_contract["arcface_raw_evidence_sha256"]
    return payload


def _materialize_visual_unit(
    validated: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, Any]:
    request: PhaseResultsRequest = validated["request"]
    unit_id = str(run["logical_run_id"])
    _require_safe_id(unit_id, "visual unit ID")
    evidence_dir = request.phase_root / "visual_evidence" / unit_id
    evidence_path = evidence_dir / "visual_evidence.json"
    review_path = request.phase_root / "visual_reviews" / f"{unit_id}.json"
    rows_by_id = {str(row["sample_id"]): row for row in run["rows"]}
    if request.phase == "full":
        manifest_path: Path = validated["visual_manifest_path"]
        manifest_ids = _ordered_sample_ids(
            validated["visual_manifest_rows"], "visual manifest"
        )
    else:
        manifest_path = validated["manifest_path"]
        manifest_ids = validated["manifest_ids"]
    rows = []
    for sample_id in manifest_ids:
        if sample_id not in rows_by_id:
            raise PhaseResultsError("visual evidence is missing a manifest sample")
        row = rows_by_id[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "source": row["source"],
                "native": row["native"],
                "candidate": row["candidate"],
            }
        )
    pages = write_contact_sheets(
        evidence_dir / "contact_sheets",
        rows,
        columns=("source", "native", "candidate"),
    )
    evidence = build_visual_evidence_contract(
        manifest_path=manifest_path,
        rows=rows,
        pages=pages,
        columns=("source", "native", "candidate"),
        expected_count=len(manifest_ids),
    )
    write_immutable_contract(
        evidence_path,
        evidence,
        digest_field="evidence_contract_sha256",
    )
    return {
        "unit_id": unit_id,
        "arm_id": run["arm_id"],
        "seed": run["seed"],
        "repeat_index": run["repeat_index"],
        "evidence_path": str(evidence_path),
        "evidence_contract_sha256": evidence["evidence_contract_sha256"],
        "review_path": str(review_path),
    }


def _materialize_heldout(
    validated: Mapping[str, Any],
    winner_run: Mapping[str, Any],
    evaluator: HeldoutEvaluator,
    automatic_arm: Mapping[str, Any],
) -> dict[str, Any]:
    request: PhaseResultsRequest = validated["request"]
    assert request.selection is not None and request.heldout_seal is not None
    _, source_paths = _load_source_index_contract(
        request.repo_root.resolve(),
        request.source_index_path,
        request.source_index_sha256,
    )
    samples = _sample_evidence(winner_run)
    if [sample.sample_id for sample in samples] != list(validated["manifest_ids"]):
        raise PhaseResultsError("heldout samples do not follow the full manifest")
    for sample in samples:
        source_declared = Path(sample.source)
        if (
            source_declared.is_symlink()
            or source_declared.resolve() != source_paths.get(sample.sample_id)
        ):
            raise PhaseResultsError(
                "heldout source path disagrees with locked source index"
            )
        for role, expected_sha256 in (
            ("source", sample.source_sha256),
            ("native", sample.native_sha256),
            ("candidate", sample.candidate_sha256),
        ):
            declared = Path(getattr(sample, role))
            if declared.is_symlink():
                raise FileNotFoundError(f"heldout {role} image must not be a symlink")
            path = declared.resolve()
            if role != "source":
                path = _contained_file(
                    request.repo_root.resolve(), declared, f"heldout {role} image"
                )
            elif not path.is_file():
                raise FileNotFoundError(f"heldout source image is missing: {path}")
            if _sha256_file(path) != expected_sha256:
                raise PhaseResultsError(f"heldout {role} image SHA256 mismatch")
    _verify_external_digest(request.selection, "selection_sha256")
    _verify_external_digest(request.heldout_seal, "heldout_seal_sha256")
    if request.heldout_seal.get("selection_sha256") != request.selection.get(
        "selection_sha256"
    ):
        raise PhaseResultsError("heldout seal does not bind selection")
    if (
        request.heldout_seal.get("sealed") is not True
        or request.heldout_seal.get("execution_count") != 0
    ):
        raise PhaseResultsError("heldout evaluator was not sealed-unrun")
    _validate_heldout_assets(request.heldout_seal, request.repo_root)
    winner = request.selection.get("winner")
    if not isinstance(winner, Mapping) or winner.get("arm_id") != winner_run["arm_id"]:
        raise PhaseResultsError("Full run does not match the selected winner")
    if winner.get("config_sha256") != winner_run["algorithm_config_sha256"]:
        raise PhaseResultsError("Full run changed the selected algorithm config")
    claim_path = request.phase_root / "heldout_execution_claim.json"
    started_path = request.phase_root / "heldout_execution_started.json"
    result_path = request.phase_root / "heldout_raw_evidence.json"
    claim = {
        "schema_version": 1,
        "contract_type": "safa_r9_heldout_execution_claim_v1",
        "selection_sha256": request.selection["selection_sha256"],
        "heldout_seal_sha256": request.heldout_seal["heldout_seal_sha256"],
        "winner_output_sha256": winner_run["output_sha256"],
        "source_index_path": str(request.source_index_path.resolve()),
        "source_index_sha256": request.source_index_sha256,
        "source_asset_manifest_sha256": _asset_manifest_digest(samples, "source"),
        "native_asset_manifest_sha256": _asset_manifest_digest(samples, "native"),
        "winner_asset_manifest_sha256": _asset_manifest_digest(samples, "candidate"),
    }
    claim["heldout_execution_claim_sha256"] = _canonical_digest(
        claim, "heldout_execution_claim_sha256"
    )
    write_immutable_contract(
        claim_path,
        claim,
        digest_field="heldout_execution_claim_sha256",
    )
    if result_path.is_file():
        if not started_path.is_file():
            raise PhaseResultsError(
                "heldout raw evidence exists without an execution-started contract"
            )
        started_contract = _read_digest_contract(
            started_path,
            digest_field="heldout_execution_started_sha256",
            contract_type="safa_r9_heldout_execution_started_v1",
        )
        if (
            started_contract.get("heldout_execution_claim_sha256")
            != claim["heldout_execution_claim_sha256"]
        ):
            raise PhaseResultsError(
                "heldout execution-started contract does not bind current claim"
            )
        raw_contract = _read_digest_contract(
            result_path,
            digest_field="heldout_raw_evidence_sha256",
            contract_type="safa_r9_heldout_raw_evidence_v1",
        )
        if raw_contract.get("claim_sha256") != claim["heldout_execution_claim_sha256"]:
            raise PhaseResultsError("heldout raw evidence does not bind current claim")
        raw = raw_contract.get("raw")
    else:
        if started_path.exists():
            raise PhaseResultsError(
                "heldout execution started without a result; rerun is forbidden"
            )
        started = {
            "schema_version": 1,
            "contract_type": "safa_r9_heldout_execution_started_v1",
            "heldout_execution_claim_sha256": claim["heldout_execution_claim_sha256"],
        }
        started["heldout_execution_started_sha256"] = _canonical_digest(
            started, "heldout_execution_started_sha256"
        )
        _write_exclusive_json(started_path, started)
        raw = evaluator(
            HeldoutEvaluationRequest(
                phase="full",
                arm_id=str(winner_run["arm_id"]),
                seed=int(winner_run["seed"]),
                source_index_path=request.source_index_path,
                source_index_sha256=request.source_index_sha256,
                samples=samples,
                selection=request.selection,
                heldout_seal=request.heldout_seal,
            )
        )
        if not isinstance(raw, Mapping):
            raise PhaseResultsError("heldout evaluator must return a mapping")
        _reject_derived_input_fields(raw)
        raw_contract = {
            "schema_version": 1,
            "contract_type": "safa_r9_heldout_raw_evidence_v1",
            "claim_sha256": claim["heldout_execution_claim_sha256"],
            "raw": _json_roundtrip(raw, "heldout raw evidence"),
        }
        raw_contract["heldout_raw_evidence_sha256"] = _canonical_digest(
            raw_contract, "heldout_raw_evidence_sha256"
        )
        write_immutable_contract(
            result_path,
            raw_contract,
            digest_field="heldout_raw_evidence_sha256",
        )
    if not isinstance(raw, Mapping):
        raise PhaseResultsError("heldout raw evidence is invalid")
    return _normalize_heldout_raw(
        raw,
        manifest_ids=validated["manifest_ids"],
        seed=int(winner_run["seed"]),
        bootstrap_seed=request.bootstrap_seed,
        raw_sha256=raw_contract["heldout_raw_evidence_sha256"],
        arcface_summary=automatic_arm["seed_results"][0]["arcface_summary"],
        arcface_bootstrap=automatic_arm.get("privacy_bootstrap"),
    )


def _validate_heldout_assets(heldout_seal: Mapping[str, Any], repo_root: Path) -> None:
    assets = heldout_seal.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != {
        "e1",
        "e2",
        "facenet",
        "adaface",
    }:
        raise PhaseResultsError("heldout seal assets are incomplete")
    for name, asset in assets.items():
        if not isinstance(asset, Mapping) or set(asset) != {
            "path",
            "sha256",
            "state",
        }:
            raise PhaseResultsError(f"heldout asset {name} fields mismatch")
        if asset.get("state") != "sealed_unrun":
            raise PhaseResultsError(f"heldout asset {name} is not sealed-unrun")
        path = _contained_file(
            repo_root, Path(str(asset.get("path", ""))), f"heldout asset {name}"
        )
        if _sha256_file(path) != asset.get("sha256"):
            raise PhaseResultsError(f"heldout asset {name} digest mismatch")


def _normalize_heldout_raw(
    raw: Mapping[str, Any],
    *,
    manifest_ids: Sequence[str],
    seed: int,
    bootstrap_seed: int,
    raw_sha256: str,
    arcface_summary: Any,
    arcface_bootstrap: Any,
) -> dict[str, Any]:
    if set(raw) != {"representations", "recognizers", "identity_report"}:
        raise PhaseResultsError("heldout raw evidence fields are not canonical")
    representations = raw.get("representations")
    recognizers = raw.get("recognizers")
    if not isinstance(representations, Mapping) or set(representations) != {"e1", "e2"}:
        raise PhaseResultsError("heldout representations require e1 and e2")
    if not isinstance(recognizers, Mapping) or set(recognizers) != {
        "facenet",
        "adaface",
    }:
        raise PhaseResultsError("heldout recognizers require facenet and adaface")
    normalized_representations = {}
    for name in ("e1", "e2"):
        rows = _normalize_paired_rows(
            representations[name], manifest_ids, seed=seed, label=name
        )
        bootstrap = privacy_delta_cluster_bootstrap(
            rows,
            expected_seeds=(seed,),
            bootstrap_seed=bootstrap_seed,
        )
        normalized_representations[name] = {
            "winner_mean": statistics.fmean(
                row["source_candidate_cosine"] for row in rows
            ),
            "native_mean": statistics.fmean(
                row["source_native_cosine"] for row in rows
            ),
            "paired_bootstrap_lower_95": bootstrap["lower_95_one_sided"],
            "bootstrap_sha256": bootstrap["bootstrap_sha256"],
        }
    normalized_recognizers = {
        "arcface": _normalize_recognizer_result(
            arcface_summary,
            manifest_ids,
            rows=None,
            seed=seed,
            bootstrap_seed=bootstrap_seed,
            precomputed_bootstrap=arcface_bootstrap,
            label="arcface",
        )
    }
    for name in ("facenet", "adaface"):
        recognizer = recognizers[name]
        if not isinstance(recognizer, Mapping):
            raise PhaseResultsError(f"{name} recognizer evidence must be a mapping")
        normalized_recognizers[name] = _normalize_recognizer_result(
            recognizer,
            manifest_ids,
            rows=recognizer.get("rows"),
            seed=seed,
            bootstrap_seed=bootstrap_seed,
            precomputed_bootstrap=None,
            label=name,
        )
    report = raw.get("identity_report")
    try:
        normalized_report = validate_identity_report(
            report, expected_count=len(manifest_ids)
        )
    except ValueError as error:
        raise PhaseResultsError(
            "heldout identity report contract is invalid"
        ) from error
    for name, recognizer in normalized_recognizers.items():
        if normalized_report["recognizers"][name]["coverage"] != recognizer["coverage"]:
            raise PhaseResultsError(
                f"{name} identity report coverage disagrees with recognizer evidence"
            )
    return {
        "execution_count": 1,
        "representations": normalized_representations,
        "recognizers": normalized_recognizers,
        "identity_report": normalized_report,
        "heldout_raw_evidence_sha256": raw_sha256,
    }


def _normalize_recognizer_result(
    value: Any,
    manifest_ids: Sequence[str],
    *,
    rows: Any,
    seed: int,
    bootstrap_seed: int,
    precomputed_bootstrap: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseResultsError(f"{label} recognizer evidence must be a mapping")
    winner_count_field = (
        "candidate_exact_one_count" if label == "arcface" else "winner_exact_one_count"
    )
    expected_fields = {
        "source_exact_one_count",
        "native_exact_one_count",
        winner_count_field,
        "paired_exact_one_count",
        "failure_sample_ids",
    }
    if label != "arcface":
        expected_fields.add("rows")
    if set(value) != expected_fields:
        raise PhaseResultsError(f"{label} recognizer evidence fields are not canonical")
    total = len(manifest_ids)
    counts = {
        "source_exact_one_count": _require_nonnegative_int(
            value.get("source_exact_one_count"),
            f"{label} source exact-one count",
        ),
        "native_exact_one_count": _require_nonnegative_int(
            value.get("native_exact_one_count"),
            f"{label} native exact-one count",
        ),
        "winner_exact_one_count": _require_nonnegative_int(
            value.get(winner_count_field),
            f"{label} winner exact-one count",
        ),
        "coverage": _require_nonnegative_int(
            value.get("paired_exact_one_count"),
            f"{label} paired exact-one count",
        ),
    }
    if any(count > total for count in counts.values()):
        raise PhaseResultsError(f"{label} exact-one count exceeds manifest size")
    if counts["coverage"] > min(
        counts["source_exact_one_count"],
        counts["native_exact_one_count"],
        counts["winner_exact_one_count"],
    ):
        raise PhaseResultsError(f"{label} paired coverage exceeds a role count")
    failure_ids = value.get("failure_sample_ids")
    if not isinstance(failure_ids, list) or any(
        not isinstance(sample_id, str) for sample_id in failure_ids
    ):
        raise PhaseResultsError(f"{label} failure sample IDs must be a list")
    ordered_failures = [
        sample_id for sample_id in manifest_ids if sample_id in set(failure_ids)
    ]
    if (
        failure_ids != ordered_failures
        or len(set(failure_ids)) != len(failure_ids)
        or len(failure_ids) != total - counts["coverage"]
    ):
        raise PhaseResultsError(
            f"{label} failures must be the ordered complement of paired coverage"
        )
    if counts["coverage"] == total:
        if any(count != total for count in counts.values()) or failure_ids:
            raise PhaseResultsError(
                f"{label} complete coverage requires exact-one for every role"
            )
        if label == "arcface":
            if not isinstance(precomputed_bootstrap, Mapping):
                raise PhaseResultsError(
                    "complete ArcFace coverage requires its paired bootstrap"
                )
            bootstrap = precomputed_bootstrap
        else:
            normalized_rows = _normalize_paired_rows(
                rows, manifest_ids, seed=seed, label=label
            )
            bootstrap = privacy_delta_cluster_bootstrap(
                normalized_rows,
                expected_seeds=(seed,),
                bootstrap_seed=bootstrap_seed,
            )
        upper = _finite_float(
            bootstrap.get("upper_95_one_sided"),
            f"{label} privacy upper bound",
        )
        bootstrap_sha256 = _require_sha256(
            bootstrap.get("bootstrap_sha256"), f"{label} bootstrap SHA256"
        )
    else:
        if rows not in (None, []):
            raise PhaseResultsError(
                f"{label} incomplete coverage forbids partial cosine rows"
            )
        if precomputed_bootstrap is not None:
            raise PhaseResultsError(
                f"{label} incomplete coverage forbids a partial bootstrap"
            )
        upper = None
        bootstrap_sha256 = None
    return {
        **counts,
        "failure_sample_ids": failure_ids,
        "privacy_delta_upper_95": upper,
        "bootstrap_sha256": bootstrap_sha256,
    }


def _normalize_paired_rows(
    value: Any, manifest_ids: Sequence[str], *, seed: int, label: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PhaseResultsError(f"{label} paired evidence must be a list")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "native_cosine",
            "winner_cosine",
        }:
            raise PhaseResultsError(f"{label} paired row {index} fields mismatch")
        rows.append(
            {
                "sample_id": row["sample_id"],
                "seed": seed,
                "source_native_cosine": _cosine(
                    row["native_cosine"], f"{label} native cosine"
                ),
                "source_candidate_cosine": _cosine(
                    row["winner_cosine"], f"{label} winner cosine"
                ),
            }
        )
    if [row["sample_id"] for row in rows] != list(manifest_ids):
        raise PhaseResultsError(f"{label} paired rows must follow manifest order")
    return rows


def _build_phase_results(
    validated: Mapping[str, Any],
    automatic: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    request: PhaseResultsRequest = validated["request"]
    evidence_by_unit = {
        unit["unit_id"]: _load_visual_evidence(Path(unit["evidence_path"]))
        for unit in automatic["visual_units"]
    }
    if request.phase == "diagnose":
        roles = {
            str(row["sample_id"]): row.get("role") for row in validated["manifest_rows"]
        }
        if set(roles.values()) != {"difficult", "control"}:
            raise PhaseResultsError(
                "diagnose manifest must label difficult/control roles"
            )
        arms = []
        for arm in automatic["arms"]:
            repeats = []
            visual_gates = []
            for repeat in arm["repeat_results"]:
                unit_id = repeat["visual_unit_id"]
                evidence = evidence_by_unit[unit_id]
                review = reviews[unit_id]
                difficult_gate = _derive_subset_visual_gate(
                    review,
                    evidence,
                    [
                        sample_id
                        for sample_id, role in roles.items()
                        if role == "difficult"
                    ],
                    severe_limit=3,
                )
                control_gate = _derive_subset_visual_gate(
                    review,
                    evidence,
                    [
                        sample_id
                        for sample_id, role in roles.items()
                        if role == "control"
                    ],
                    severe_limit=1,
                )
                visual_gates.extend(
                    [
                        difficult_gate["visual_gate_sha256"],
                        control_gate["visual_gate_sha256"],
                    ]
                )
                repeats.append(
                    {
                        "repeat_index": repeat["repeat_index"],
                        "run_sha256": repeat["run_sha256"],
                        "difficult_severe_count": difficult_gate["severe_count"],
                        "control_severe_count": control_gate["severe_count"],
                        "e0_mean": repeat["e0_mean"],
                        "edev_delta_vs_matched_native": repeat[
                            "edev_delta_vs_matched_native"
                        ],
                        "diagnostics_finite": repeat["diagnostics_finite"],
                        "diagnostics_contract_sha256": repeat[
                            "diagnostics_contract_sha256"
                        ],
                    }
                )
            arms.append(
                {
                    "arm_id": arm["arm_id"],
                    "family": arm["family"],
                    "config_sha256": arm["config_sha256"],
                    "evaluator_evidence_sha256": arm["evaluator_evidence_sha256"],
                    "output_sha256": _canonical_json_sha256(
                        {
                            "automatic": arm["automatic_output_sha256"],
                            "evaluators": arm["evaluator_evidence_sha256"],
                            "visual_gates": visual_gates,
                        }
                    ),
                    "repeat_results": repeats,
                }
            )
        value_field = "arms"
        value = arms
    elif request.phase in {"calibrate", "confirm512"}:
        arms = []
        for arm in automatic["arms"]:
            seed_results = []
            visual_gates = []
            for seed_row in arm["seed_results"]:
                unit_id = seed_row["visual_unit_id"]
                severe_limit = 3 if request.phase == "calibrate" else 25
                visual = derive_visual_arm_pass(
                    reviews[unit_id],
                    evidence_by_unit[unit_id],
                    severe_limit=severe_limit,
                )
                visual_gates.append(visual["visual_gate_sha256"])
                seed_results.append(
                    {
                        key: value
                        for key, value in seed_row.items()
                        if key
                        not in {
                            "visual_unit_id",
                            "arcface_summary",
                            "quality_raw_evidence_path",
                            "native_quality_raw_evidence_path",
                            "arcface_raw_evidence_path",
                        }
                    }
                    | {
                        "severe_count": visual["severe_count"],
                        "severe_sample_ids": visual["severe_sample_ids"],
                    }
                )
            arms.append(
                {
                    "arm_id": arm["arm_id"],
                    "family": arm["family"],
                    "config_sha256": arm["config_sha256"],
                    "evaluator_evidence_sha256": arm["evaluator_evidence_sha256"],
                    "output_sha256": _canonical_json_sha256(
                        {
                            "automatic": arm["automatic_output_sha256"],
                            "evaluators": arm["evaluator_evidence_sha256"],
                            "visual_gates": visual_gates,
                        }
                    ),
                    "seed_results": seed_results,
                    "privacy_rows": arm["privacy_rows"],
                    "paired_metric_rows": arm["paired_metric_rows"],
                }
            )
        value_field = "arms"
        value = arms
    else:
        if len(automatic["arms"]) != 1 or automatic.get("heldout") is None:
            raise PhaseResultsError("Full automatic evidence is incomplete")
        arm = automatic["arms"][0]
        seed_row = arm["seed_results"][0]
        unit_id = seed_row["visual_unit_id"]
        visual = derive_visual_arm_pass(
            reviews[unit_id], evidence_by_unit[unit_id], severe_limit=3
        )
        heldout = automatic["heldout"]
        assert request.selection is not None
        winner = request.selection["winner"]
        quality = {
            key: value
            for key, value in seed_row.items()
            if key
            not in {
                "visual_unit_id",
                "arcface_summary",
                "quality_raw_evidence_path",
                "native_quality_raw_evidence_path",
                "arcface_raw_evidence_path",
            }
        } | {
            "severe_count": visual["severe_count"],
            "severe_sample_ids": visual["severe_sample_ids"],
            "paired_metric_rows": arm["paired_metric_rows"],
        }
        value_field = "result"
        value = {
            "execution_count": heldout["execution_count"],
            "winner_arm_id": winner["arm_id"],
            "config_sha256": winner["config_sha256"],
            "evaluator_evidence_sha256": arm["evaluator_evidence_sha256"],
            "output_sha256": _canonical_json_sha256(
                {
                    "automatic": arm["automatic_output_sha256"],
                    "evaluators": arm["evaluator_evidence_sha256"],
                    "visual_gate": visual["visual_gate_sha256"],
                    "heldout": heldout["heldout_raw_evidence_sha256"],
                }
            ),
            "seed": seed_row["seed"],
            "full_visual_severe_count": visual["severe_count"],
            "representations": {
                name: {
                    key: row[key]
                    for key in (
                        "winner_mean",
                        "native_mean",
                        "paired_bootstrap_lower_95",
                    )
                }
                for name, row in heldout["representations"].items()
            },
            "recognizers": heldout["recognizers"],
            "quality": quality,
            "identity_report": heldout["identity_report"],
        }
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_results_v1",
        "phase": request.phase,
        "campaign_runtime_sha256": request.campaign_runtime_sha256,
        "manifest_contracts_sha256": request.manifest_contracts_sha256,
        "manifest_sha256": request.manifest_sha256,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "run_plan_sha256": automatic["run_plan_sha256"],
        value_field: value,
    }
    payload["phase_results_sha256"] = _canonical_digest(payload, "phase_results_sha256")
    return payload


def _derive_subset_visual_gate(
    review: Mapping[str, Any],
    evidence: Mapping[str, Any],
    sample_ids: Sequence[str],
    *,
    severe_limit: int,
) -> dict[str, Any]:
    wanted = set(sample_ids)
    evidence_rows = [row for row in evidence["samples"] if row["sample_id"] in wanted]
    review_rows = [row for row in review["samples"] if row["sample_id"] in wanted]
    if [row["sample_id"] for row in evidence_rows] != list(sample_ids):
        raise PhaseResultsError("visual subset order disagrees with diagnose manifest")
    subset_evidence = {"sample_count": len(evidence_rows), "samples": evidence_rows}
    return derive_visual_arm_pass(
        {"samples": review_rows}, subset_evidence, severe_limit=severe_limit
    )


def _load_reviews(
    automatic: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], int]:
    reviews = {}
    for unit in automatic.get("visual_units", ()):
        if not isinstance(unit, Mapping):
            raise PhaseResultsError("automatic visual unit is invalid")
        evidence_path = Path(str(unit.get("evidence_path", "")))
        evidence = _load_visual_evidence(evidence_path)
        if evidence.get("evidence_contract_sha256") != unit.get(
            "evidence_contract_sha256"
        ):
            raise PhaseResultsError("automatic visual evidence digest mismatch")
        review_path = Path(str(unit.get("review_path", "")))
        if review_path.exists():
            reviews[str(unit["unit_id"])] = validate_visual_review(
                review_path, evidence_path
            )
    return reviews, len(reviews)


def _awaiting_contract(
    request: PhaseResultsRequest, automatic: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_phase_status_v1",
        "status": "awaiting_visual_review",
        "phase": request.phase,
        "campaign_id": request.campaign_id,
        "automatic_evidence_sha256": automatic["automatic_evidence_sha256"],
        "bounded_exit_code": AWAITING_VISUAL_REVIEW_EXIT_CODE,
        "required_reviews": [
            {
                "unit_id": unit["unit_id"],
                "evidence_path": unit["evidence_path"],
                "evidence_contract_sha256": unit["evidence_contract_sha256"],
                "review_path": unit["review_path"],
            }
            for unit in automatic["visual_units"]
        ],
    }
    payload["awaiting_visual_review_sha256"] = _canonical_digest(
        payload, "awaiting_visual_review_sha256"
    )
    return payload


def _validate_automatic_context(
    automatic: Mapping[str, Any], validated: Mapping[str, Any]
) -> None:
    request: PhaseResultsRequest = validated["request"]
    if (
        automatic.get("phase") != request.phase
        or automatic.get("campaign_id") != request.campaign_id
    ):
        raise PhaseResultsError("automatic evidence phase/campaign mismatch")
    if automatic.get("context") != _request_context(request):
        raise PhaseResultsError("automatic evidence context mismatch")
    if (
        automatic.get("run_plan") != validated["run_plan"]
        or automatic.get("run_plan_sha256") != validated["run_plan_sha256"]
    ):
        raise PhaseResultsError("automatic evidence run plan mismatch")
    manifest = automatic.get("manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("sha256") != request.manifest_sha256
    ):
        raise PhaseResultsError("automatic evidence manifest mismatch")
    if _sha256_file(Path(str(manifest["path"]))) != request.manifest_sha256:
        raise PhaseResultsError("automatic evidence manifest was replaced")
    source_index = automatic.get("source_index")
    expected_source_index = {
        "path": str(validated["source_index_path"]),
        "sha256": request.source_index_sha256,
    }
    if source_index != expected_source_index:
        raise PhaseResultsError("automatic evidence source index mismatch")
    if _sha256_file(validated["source_index_path"]) != request.source_index_sha256:
        raise PhaseResultsError("automatic evidence source index was replaced")
    _rehash_automatic_runs(automatic, validated)
    _rehash_evaluator_evidence(automatic, validated)


def _rehash_automatic_runs(
    automatic: Mapping[str, Any], validated: Mapping[str, Any]
) -> None:
    runs = automatic.get("runs")
    if not isinstance(runs, list) or not runs:
        raise PhaseResultsError("automatic evidence runs are missing")
    for run in runs:
        if not isinstance(run, Mapping):
            raise PhaseResultsError("automatic evidence run is invalid")
        output = run.get("output_contract")
        if not isinstance(output, Mapping):
            raise PhaseResultsError("automatic run output contract is missing")
        if _canonical_json_sha256(output) != run.get("evidence_binding_sha256"):
            raise PhaseResultsError("automatic run evidence binding digest mismatch")
        for shard in output.get("shards", ()):
            output_dir = Path(str(shard.get("output_dir", "")))
            for filename, digest_field in (
                ("generation_result.json", "generation_result_sha256"),
                ("run_manifest.json", "run_manifest_sha256"),
                ("completion.json", "completion_sha256"),
                ("per_sample.jsonl", "per_sample_sha256"),
            ):
                path = output_dir / filename
                if _sha256_file(path) != shard.get(digest_field):
                    raise PhaseResultsError(
                        f"automatic run shard {digest_field} evidence was replaced"
                    )
        images = output.get("images")
        if not isinstance(images, list) or not images:
            raise PhaseResultsError("automatic run image evidence is missing")
        image_ids: list[str] = []
        for image_row in images:
            sample_id = image_row.get("sample_id")
            if (
                not isinstance(sample_id, str)
                or sample_id not in validated["source_paths"]
            ):
                raise PhaseResultsError(
                    "automatic run image sample ID is not registered"
                )
            image_ids.append(sample_id)
            source = Path(str(image_row.get("source", ""))).resolve()
            if source != validated["source_paths"][sample_id]:
                raise PhaseResultsError(
                    "automatic run source path disagrees with locked source index"
                )
            for path_field, digest_field in (
                ("source", "source_sha256"),
                ("native", "native_sha256"),
                ("candidate", "candidate_sha256"),
            ):
                if _sha256_file(
                    Path(str(image_row.get(path_field, "")))
                ) != image_row.get(digest_field):
                    raise PhaseResultsError(
                        f"automatic run {path_field} image evidence was replaced"
                    )
        if image_ids != validated["manifest_ids"]:
            raise PhaseResultsError("automatic run image order changed")


def _rehash_evaluator_evidence(
    automatic: Mapping[str, Any], validated: Mapping[str, Any]
) -> None:
    for arm in automatic.get("arms", ()):
        for seed_row in arm.get("seed_results", ()):
            for path_field, digest_field, contract_type, contract_digest in (
                (
                    "quality_raw_evidence_path",
                    "quality_raw_evidence_sha256",
                    "safa_r9_quality_raw_evidence_v1",
                    "quality_raw_evidence_sha256",
                ),
                (
                    "native_quality_raw_evidence_path",
                    "native_quality_raw_evidence_sha256",
                    "safa_r9_quality_raw_evidence_v1",
                    "quality_raw_evidence_sha256",
                ),
                (
                    "arcface_raw_evidence_path",
                    "arcface_raw_evidence_sha256",
                    "safa_r9_arcface_raw_evidence_v1",
                    "arcface_raw_evidence_sha256",
                ),
            ):
                evidence = _read_digest_contract(
                    Path(str(seed_row.get(path_field, ""))),
                    digest_field=contract_digest,
                    contract_type=contract_type,
                )
                if evidence[contract_digest] != seed_row.get(digest_field):
                    raise PhaseResultsError(
                        f"automatic evaluator evidence {digest_field} mismatch"
                    )
    if automatic.get("phase") == "full":
        heldout = automatic.get("heldout")
        if not isinstance(heldout, Mapping):
            raise PhaseResultsError("Full automatic evidence lacks heldout results")
        raw = _read_digest_contract(
            validated["request"].phase_root / "heldout_raw_evidence.json",
            digest_field="heldout_raw_evidence_sha256",
            contract_type="safa_r9_heldout_raw_evidence_v1",
        )
        if raw["heldout_raw_evidence_sha256"] != heldout.get(
            "heldout_raw_evidence_sha256"
        ):
            raise PhaseResultsError("Full heldout raw evidence digest mismatch")
        claim = _read_digest_contract(
            validated["request"].phase_root / "heldout_execution_claim.json",
            digest_field="heldout_execution_claim_sha256",
            contract_type="safa_r9_heldout_execution_claim_v1",
        )
        started = _read_digest_contract(
            validated["request"].phase_root / "heldout_execution_started.json",
            digest_field="heldout_execution_started_sha256",
            contract_type="safa_r9_heldout_execution_started_v1",
        )
        if (
            raw.get("claim_sha256") != claim["heldout_execution_claim_sha256"]
            or started.get("heldout_execution_claim_sha256")
            != claim["heldout_execution_claim_sha256"]
        ):
            raise PhaseResultsError("Full heldout claim chain mismatch")
        request: PhaseResultsRequest = validated["request"]
        assert request.heldout_seal is not None
        _validate_heldout_assets(request.heldout_seal, request.repo_root)


def _validate_matched_native(
    candidate: Mapping[str, Any], native: Mapping[str, Any]
) -> None:
    native_by_id = {row["sample_id"]: row for row in native["rows"]}
    for row in candidate["rows"]:
        baseline = native_by_id.get(row["sample_id"])
        if baseline is None:
            raise PhaseResultsError("candidate/native sample membership mismatch")
        if row["native_sha256"] != baseline["candidate_sha256"]:
            raise PhaseResultsError(
                "candidate matched-native PNG differs from native run"
            )
        for candidate_field, native_field in (
            ("native_cosine", "candidate_cosine"),
            ("native_edev_cosine", "edev_cosine"),
        ):
            left = row["metrics"].get(candidate_field)
            right = baseline["metrics"].get(native_field)
            if left != right:
                raise PhaseResultsError(
                    f"candidate matched-native {candidate_field} differs from native run"
                )


def validate_interval_diagnostics(
    rows: Sequence[Mapping[str, Any]], interval_contract: Any
) -> dict[str, Any]:
    """Validate every A diagnostic against the generator-owned interval contract."""
    if not isinstance(interval_contract, Mapping) or set(interval_contract) != {
        "schema_version",
        "mode",
        "active_guidance_intervals",
        "collect_interval_diagnostics",
        "expected_algorithm_nfe",
        "expected_diagnostic_nfe",
        "expected_algorithm_trace",
        "expected_diagnostic_trace",
    }:
        raise PhaseResultsError("A run lacks its canonical interval contract")
    if interval_contract.get("collect_interval_diagnostics") is not True:
        raise PhaseResultsError("A run must collect interval diagnostics")
    active = interval_contract.get("active_guidance_intervals")
    if not isinstance(active, list) or active != [
        name for name in ("I1", "I2", "I3") if name in active
    ]:
        raise PhaseResultsError("A interval mask is not canonical")
    algorithm_nfe = _require_nonnegative_int(
        interval_contract.get("expected_algorithm_nfe"), "expected algorithm NFE"
    )
    diagnostic_nfe = _require_nonnegative_int(
        interval_contract.get("expected_diagnostic_nfe"), "expected diagnostic NFE"
    )
    expected_algorithm_trace = interval_contract.get("expected_algorithm_trace")
    expected_diagnostic_trace = interval_contract.get("expected_diagnostic_trace")
    _assert_finite_json(interval_contract, "A interval contract")
    interval_times = {
        "I1": (1.0, 0.75),
        "I2": (0.75, 0.5),
        "I3": (0.5, 0.25),
    }
    interval_fields = {
        "interval_id",
        "active",
        "t",
        "s",
        "loss_before_correction",
        "loss_after_correction",
        "gradient_norm",
        "velocity_norm",
        "transport_norm",
        "correction_norm",
        "correction_transport_ratio",
        "gradient_velocity_cosine",
        "local_semigroup_residual",
    }
    route_common_fields = {
        "active_guidance_intervals",
        "interval_diagnostics_enabled",
        "interval_diagnostics",
        "algorithm_nfe",
        "diagnostic_nfe",
        "guided_times",
        "unguided_times",
        "loss_history",
        "mode",
        "step_size",
    }
    route_mode = interval_contract.get("mode")
    if route_mode == "paper_algorithm_split":
        route_fields = route_common_fields
        allowed_step_sizes = {0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5}
    elif route_mode == "official_head_current_xt":
        route_fields = route_common_fields | {
            "adam_learning_rates",
            "num_optim_iters",
            "optimization_mode",
            "sample_mode",
            "uses_adam",
        }
        allowed_step_sizes = {0.125, 0.1875, 0.25}
    else:
        raise PhaseResultsError("A interval contract mode is not registered")
    expected_guided_times = [1.0, 0.75, 0.5, 0.25]
    expected_unguided_times = [0.25, 0.125, 0.0]
    diagnostic_rows = []
    for ordinal, row in enumerate(rows):
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PhaseResultsError(f"A diagnostic row {ordinal} lacks metrics")
        actual_algorithm_nfe = _strict_int(
            metrics.get("candidate_algorithm_nfe"), "candidate algorithm NFE"
        )
        actual_diagnostic_nfe = _strict_int(
            metrics.get("candidate_diagnostic_nfe"), "candidate diagnostic NFE"
        )
        if (
            actual_algorithm_nfe != algorithm_nfe
            or _strict_int(metrics.get("candidate_nfe"), "candidate NFE")
            != algorithm_nfe
            or actual_diagnostic_nfe != diagnostic_nfe
            or metrics.get("candidate_trace") != expected_algorithm_trace
            or metrics.get("candidate_diagnostic_trace") != expected_diagnostic_trace
        ):
            raise PhaseResultsError(
                f"A diagnostic row {ordinal} trace/NFE contract mismatch"
            )
        route = metrics.get("route_diagnostics")
        if not isinstance(route, Mapping) or set(route) != route_fields:
            raise PhaseResultsError(
                f"A diagnostic row {ordinal} route fields are not canonical"
            )
        loss_history = route["loss_history"]
        step_size = _finite_float(route["step_size"], "diagnostic route step size")
        if (
            route["active_guidance_intervals"] != active
            or route["interval_diagnostics_enabled"] is not True
            or _strict_int(route["algorithm_nfe"], "route algorithm NFE")
            != algorithm_nfe
            or _strict_int(route["diagnostic_nfe"], "route diagnostic NFE")
            != diagnostic_nfe
            or route["guided_times"] != expected_guided_times
            or route["unguided_times"] != expected_unguided_times
            or route["mode"] != route_mode
            or step_size not in allowed_step_sizes
            or not isinstance(loss_history, list)
            or len(loss_history) != len(active)
        ):
            raise PhaseResultsError(
                f"A diagnostic row {ordinal} route contract mismatch"
            )
        for loss_index, loss in enumerate(loss_history):
            _finite_float(loss, f"diagnostic route loss history {loss_index}")
        if route_mode == "official_head_current_xt" and (
            route["adam_learning_rates"] != []
            or _strict_int(route["num_optim_iters"], "route optimization iterations")
            != 1
            or route["optimization_mode"] != "paper_normalized_direct_autograd"
            or route["sample_mode"] != "flow_map2"
            or route["uses_adam"] is not False
        ):
            raise PhaseResultsError(
                f"A diagnostic row {ordinal} flow-map2 route contract mismatch"
            )
        intervals = route["interval_diagnostics"]
        if not isinstance(intervals, Mapping) or tuple(intervals) != (
            "I1",
            "I2",
            "I3",
        ):
            raise PhaseResultsError(
                f"A diagnostic row {ordinal} interval coverage mismatch"
            )
        for interval_id, expected_times in interval_times.items():
            values = intervals[interval_id]
            if not isinstance(values, Mapping) or set(values) != interval_fields:
                raise PhaseResultsError(
                    f"A diagnostic row {ordinal} {interval_id} fields mismatch"
                )
            if (
                values["interval_id"] != interval_id
                or values["active"] is not (interval_id in active)
                or (values["t"], values["s"]) != expected_times
            ):
                raise PhaseResultsError(
                    f"A diagnostic row {ordinal} {interval_id} binding mismatch"
                )
            for field in (
                "loss_before_correction",
                "loss_after_correction",
                "gradient_norm",
                "velocity_norm",
                "transport_norm",
                "correction_norm",
                "correction_transport_ratio",
                "local_semigroup_residual",
            ):
                if _finite_float(values[field], f"{interval_id} {field}") < 0.0:
                    raise PhaseResultsError(
                        f"{interval_id} {field} must be nonnegative"
                    )
            cosine = _finite_float(
                values["gradient_velocity_cosine"],
                f"{interval_id} gradient-velocity cosine",
            )
            if not -1.0 <= cosine <= 1.0:
                raise PhaseResultsError(
                    f"{interval_id} gradient-velocity cosine is outside [-1,1]"
                )
            if interval_id not in active and (
                values["loss_after_correction"] != values["loss_before_correction"]
                or values["gradient_norm"] != 0.0
                or values["correction_norm"] != 0.0
                or values["correction_transport_ratio"] != 0.0
            ):
                raise PhaseResultsError(f"inactive {interval_id} contains a correction")
            if interval_id in active and not math.isclose(
                float(values["correction_transport_ratio"]),
                step_size,
                rel_tol=1.0e-6,
                abs_tol=1.0e-7,
            ):
                raise PhaseResultsError(
                    f"active {interval_id} correction ratio disagrees with step size"
                )
        _assert_finite_json(route, f"A diagnostic row {ordinal}")
        diagnostic_rows.append(
            {"sample_id": row["sample_id"], "route_diagnostics": route}
        )
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_interval_diagnostics_v1",
        "interval_contract_sha256": _canonical_json_sha256(interval_contract),
        "sample_count": len(diagnostic_rows),
        "rows_sha256": _canonical_json_sha256({"rows": diagnostic_rows}),
    }
    payload["diagnostics_contract_sha256"] = _canonical_digest(
        payload, "diagnostics_contract_sha256"
    )
    return payload


def _mean_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(
        _finite_float(row["metrics"].get(field), field) for row in rows
    )


def _algorithm_config_digest(config: Mapping[str, Any], checkpoint_sha256: str) -> str:
    base = canonical_arm_config_payload(config)
    fixed_assets = dict(base["fixed_assets"])
    for field in _PATH_ONLY_FIXED_ASSET_FIELDS:
        fixed_assets.pop(field, None)
    fixed_assets["checkpoint_sha256"] = checkpoint_sha256
    interval_contract = config.get("r9_guidance_interval_contract")
    algorithm_interval_contract = None
    if isinstance(interval_contract, Mapping):
        algorithm_interval_contract = {
            field: _json_roundtrip(
                interval_contract.get(field), f"algorithm interval contract {field}"
            )
            for field in (
                "schema_version",
                "mode",
                "active_guidance_intervals",
                "expected_algorithm_nfe",
                "expected_algorithm_trace",
            )
        }
    payload = {
        "schema_version": 2,
        "mode": base["mode"],
        "algorithm": base["algorithm"],
        "schedule_contract_sha256": base["schedule_contract_sha256"],
        "fixed_assets": fixed_assets,
        "determinism_policy_sha256": config.get("determinism_policy_sha256"),
        "attention_backend": config.get("attention_backend"),
        "active_guidance_intervals": config.get("active_guidance_intervals"),
        "algorithm_interval_contract": algorithm_interval_contract,
        "semigroup_preflight_contract_sha256": config.get(
            "semigroup_preflight_contract_sha256"
        ),
        "r9_semigroup_gate_contract_sha256": config.get(
            "r9_semigroup_gate_contract_sha256"
        ),
    }
    _assert_finite_json(payload, "algorithm config")
    return _canonical_json_sha256(payload)


def canonical_r9_algorithm_config_digest(
    config: Mapping[str, Any], checkpoint_sha256: str
) -> str:
    return _algorithm_config_digest(config, checkpoint_sha256)


def _request_context(request: PhaseResultsRequest) -> dict[str, Any]:
    context = {
        "campaign_id": request.campaign_id,
        "campaign_runtime_sha256": request.campaign_runtime_sha256,
        "manifest_contracts_sha256": request.manifest_contracts_sha256,
        "manifest_sha256": request.manifest_sha256,
        "source_index_path": str(request.source_index_path.resolve()),
        "source_index_sha256": request.source_index_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "upstream_gate_sha256": (
            None
            if request.upstream_gate is None
            else request.upstream_gate.get("gate_contract_sha256")
        ),
        "upstream_calibration_selection_sha256": (
            None
            if request.upstream_calibration_selection is None
            else request.upstream_calibration_selection.get(
                "calibration_selection_sha256"
            )
        ),
    }
    repair = evaluation_repair_binding(request)
    if repair is not None:
        context["evaluation_repair"] = repair
    return context


def generation_evidence_inventory(request: PhaseResultsRequest) -> dict[str, Any]:
    repo_root = request.repo_root.resolve()
    phase_root = _contained_path(repo_root, request.phase_root, "phase root")
    files: list[dict[str, str]] = []
    shard_roots: set[Path] = set()
    for spec in request.runs:
        for value in spec.shard_output_dirs:
            shard_root = _contained_path(repo_root, value, "generation shard output")
            if shard_root.parent != phase_root or shard_root in shard_roots:
                raise PhaseResultsError(
                    "generation repair inventory requires unique direct phase children"
                )
            if shard_root.is_symlink() or not shard_root.is_dir():
                raise PhaseResultsError("generation shard output is not a directory")
            shard_roots.add(shard_root)
            for path in sorted(shard_root.rglob("*")):
                if path.is_symlink():
                    raise PhaseResultsError("generation repair inventory rejects symlinks")
                if path.is_file():
                    files.append(
                        {
                            "path": str(path.relative_to(phase_root)),
                            "sha256": _sha256_file(path),
                        }
                    )
    files.sort(key=lambda row: row["path"])
    completion_count = sum(row["path"].endswith("/completion.json") for row in files)
    generation_result_count = sum(
        row["path"].endswith("/generation_result.json") for row in files
    )
    png_count = sum(row["path"].lower().endswith(".png") for row in files)
    payload = {
        "phase_root": str(phase_root.relative_to(repo_root)),
        "logical_run_count": len(request.runs),
        "shard_count": len(shard_roots),
        "completion_count": completion_count,
        "generation_result_count": generation_result_count,
        "file_count": len(files),
        "png_count": png_count,
        "inventory_sha256": _canonical_json_sha256({"files": files}),
    }
    if completion_count != len(shard_roots) or generation_result_count != len(
        shard_roots
    ):
        raise PhaseResultsError("generation repair inventory is incomplete")
    return payload


def evaluation_attempt_inventory(
    repo_root: Path, namespace_root: Path
) -> dict[str, Any]:
    root = repo_root.resolve()
    namespace = _contained_path(root, namespace_root, "evaluation repair namespace")
    if (
        namespace.is_symlink()
        or not namespace.is_dir()
        or namespace.parent.name != "evaluation_repairs"
    ):
        raise PhaseResultsError("evaluation repair namespace is invalid")
    files: list[dict[str, str]] = []
    for path in sorted(namespace.rglob("*")):
        if path.is_symlink():
            raise PhaseResultsError("evaluation repair namespace rejects symlinks")
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(namespace)),
                    "sha256": _sha256_file(path),
                }
            )
    result_paths = [
        row["path"] for row in files if row["path"].endswith("/result.json")
    ]
    payload = {
        "namespace_root": str(namespace.relative_to(root)),
        "file_count": len(files),
        "result_count": len(result_paths),
        "quality_result_count": sum(
            "/evaluator_runs/quality/" in f"/{path}" for path in result_paths
        ),
        "arcface_result_count": sum(
            "/evaluator_runs/arcface/" in f"/{path}" for path in result_paths
        ),
        "files": files,
        "inventory_sha256": _canonical_json_sha256({"files": files}),
    }
    return payload


def evaluation_repair_binding(
    request: PhaseResultsRequest,
) -> dict[str, str] | None:
    repo_root = request.repo_root.resolve()
    v1_path = _contained_path(
        repo_root,
        request.phase_root / EVALUATION_REPAIR_FILENAME,
        "evaluation repair contract",
    )
    v2_path = _contained_path(
        repo_root,
        request.phase_root / EVALUATION_REPAIR_V2_FILENAME,
        "superseding evaluation repair contract",
    )
    v3_path = _contained_path(
        repo_root,
        request.phase_root / EVALUATION_REPAIR_V3_FILENAME,
        "second superseding evaluation repair contract",
    )
    if v3_path.exists():
        path = v3_path
        contract_type = "safa_r9_evaluation_repair_contract_v3"
        required_fields = {"supersedes"}
    elif v2_path.exists():
        path = v2_path
        contract_type = "safa_r9_evaluation_repair_contract_v2"
        required_fields = {"supersedes"}
    elif v1_path.exists():
        path = v1_path
        contract_type = "safa_r9_evaluation_repair_contract_v1"
        required_fields = set()
    else:
        return None
    repair = _read_digest_contract(
        path,
        digest_field="repair_contract_sha256",
        contract_type=contract_type,
    )
    if set(repair) != required_fields | {
        "schema_version",
        "contract_type",
        "campaign_id",
        "phase",
        "campaign_runtime",
        "generation_evidence",
        "failed_evaluation",
        "implementations",
        "policy",
        "repair_contract_sha256",
    }:
        raise PhaseResultsError("evaluation repair contract fields are not canonical")
    if repair.get("campaign_id") != request.campaign_id or repair.get(
        "phase"
    ) != request.phase:
        raise PhaseResultsError("evaluation repair campaign or phase mismatch")
    runtime = repair.get("campaign_runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "path",
        "file_sha256",
        "contract_sha256",
    }:
        raise PhaseResultsError("evaluation repair runtime binding is invalid")
    runtime_path = _contained_file(
        repo_root, Path(str(runtime["path"])), "evaluation repair runtime"
    )
    runtime_payload = _read_digest_contract(
        runtime_path,
        digest_field="campaign_runtime_sha256",
        contract_type=None,
    )
    if (
        _sha256_file(runtime_path) != runtime["file_sha256"]
        or runtime["contract_sha256"] != runtime_payload["campaign_runtime_sha256"]
        or runtime["contract_sha256"] != request.campaign_runtime_sha256
    ):
        raise PhaseResultsError("evaluation repair runtime binding mismatch")
    if repair.get("generation_evidence") != generation_evidence_inventory(request):
        raise PhaseResultsError("evaluation repair generation inventory mismatch")
    failed = repair.get("failed_evaluation")
    if not isinstance(failed, Mapping) or set(failed) != {
        "evaluator",
        "unit_id",
        "request",
        "result",
        "mismatch",
    }:
        raise PhaseResultsError("evaluation repair failure evidence is invalid")
    if failed.get("evaluator") != "quality" or not isinstance(
        failed.get("unit_id"), str
    ):
        raise PhaseResultsError("evaluation repair failure identity is invalid")
    request_binding = _validate_repair_file_binding(
        failed.get("request"),
        repo_root=repo_root,
        digest_field="evaluator_request_sha256",
        contract_type="safa_r9_phase_evaluator_request_v1",
        label="failed evaluator request",
    )
    result_binding = _validate_repair_file_binding(
        failed.get("result"),
        repo_root=repo_root,
        digest_field="evaluator_output_sha256",
        contract_type="safa_r9_phase_evaluator_output_v1",
        label="failed evaluator result",
    )
    request_payload = _read_json_mapping(
        repo_root / request_binding["path"], "failed evaluator request"
    )
    result_payload = _read_json_mapping(
        repo_root / result_binding["path"], "failed evaluator result"
    )
    if result_payload.get("evaluator_request_sha256") != request_binding[
        "contract_sha256"
    ]:
        raise PhaseResultsError("failed evaluator result does not bind its request")
    request_evaluation = request_payload.get("payload")
    result_evaluation = result_payload.get("result")
    raw_binding = (
        result_evaluation.get("r9_evidence_binding")
        if isinstance(result_evaluation, Mapping)
        else None
    )
    if (
        not isinstance(request_evaluation, Mapping)
        or not isinstance(request_evaluation.get("source_index_path"), str)
        or not isinstance(raw_binding, Mapping)
        or "source_index_path" in raw_binding
        or raw_binding.get("source_index_sha256")
        != request_evaluation.get("source_index_sha256")
    ):
        raise PhaseResultsError("failed evaluator result is not the registered mismatch")
    if failed.get("mismatch") != {
        "field": "r9_evidence_binding.source_index_path",
        "classification": "request_transport_field_in_raw_content_binding",
        "producer_has_field": False,
        "consumer_required_field": True,
    }:
        raise PhaseResultsError("evaluation repair mismatch classification changed")
    implementations = repair.get("implementations")
    if not isinstance(implementations, Mapping) or set(implementations) != {
        "source_git_commit",
        "prior_phase_results_sha256",
        "repaired_phase_results",
        "driver",
        "evaluator_worker",
        "quality",
        "repair_runner",
    }:
        raise PhaseResultsError("evaluation repair implementation fields are invalid")
    implementation_paths = {}
    for field in (
        "repaired_phase_results",
        "driver",
        "evaluator_worker",
        "quality",
        "repair_runner",
    ):
        implementation_paths[field] = _validate_repair_implementation_binding(
            implementations.get(field), repo_root=repo_root, label=field
        )
    if implementation_paths["repaired_phase_results"] != Path(__file__).resolve():
        raise PhaseResultsError("evaluation repair binds another phase implementation")
    _require_sha256(
        implementations.get("prior_phase_results_sha256"),
        "prior phase-results SHA256",
    )
    source_commit = implementations.get("source_git_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise PhaseResultsError("evaluation repair source commit is invalid")
    if repair.get("policy") != {
        "generation_execution": "forbidden",
        "expected_generation_worker_count": 0,
        "old_failed_result_usage": "input_evidence_only",
        "old_attempt_retry_allowed": False,
        "evaluation_namespace": "evaluation_repairs/{repair_contract_sha256}",
        "request_binding": "full_repair_sha256_in_logical_run_id",
    }:
        raise PhaseResultsError("evaluation repair policy changed")
    if contract_type in {
        "safa_r9_evaluation_repair_contract_v2",
        "safa_r9_evaluation_repair_contract_v3",
    }:
        _validate_superseded_repair(
            repair.get("supersedes"),
            request=request,
            repo_root=repo_root,
            prior_phase_results_sha256=str(
                implementations["prior_phase_results_sha256"]
            ),
        )
    return {
        "path": str(path.relative_to(repo_root)),
        "file_sha256": _sha256_file(path),
        "contract_sha256": repair["repair_contract_sha256"],
    }


def _validate_superseded_repair(
    value: Any,
    *,
    request: PhaseResultsRequest,
    repo_root: Path,
    prior_phase_results_sha256: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "repair",
        "failure",
        "evaluation_attempt",
        "policy",
    }:
        raise PhaseResultsError("superseded evaluation repair fields are invalid")
    prior_value = value.get("repair")
    if not isinstance(prior_value, Mapping):
        raise PhaseResultsError("superseded evaluation repair binding is invalid")
    prior_filename = Path(str(prior_value.get("path", ""))).name
    if prior_filename == EVALUATION_REPAIR_FILENAME:
        prior_contract_type = "safa_r9_evaluation_repair_contract_v1"
        expected_failure = {
            "exception_type": "NameError",
            "symbol": "manifest_ids",
            "message": "NameError: name 'manifest_ids' is not defined",
        }
        expected_counts = {
            "file_count": 8,
            "result_count": 2,
            "quality_result_count": 2,
            "arcface_result_count": 0,
        }
    elif prior_filename == EVALUATION_REPAIR_V2_FILENAME:
        prior_contract_type = "safa_r9_evaluation_repair_contract_v2"
        expected_failure = {
            "exception_type": "CampaignContractError",
            "symbol": "raw_evidence_namespace",
            "message": "immutable contract already exists with other content",
        }
        expected_counts = {
            "file_count": 4,
            "result_count": 1,
            "quality_result_count": 1,
            "arcface_result_count": 0,
        }
    else:
        raise PhaseResultsError("superseded evaluation repair version is invalid")
    prior_binding = _validate_repair_file_binding(
        prior_value,
        repo_root=repo_root,
        digest_field="repair_contract_sha256",
        contract_type=prior_contract_type,
        label="superseded evaluation repair",
    )
    prior = _read_json_mapping(
        repo_root / prior_binding["path"], "superseded evaluation repair"
    )
    if (
        prior.get("campaign_id") != request.campaign_id
        or prior.get("phase") != request.phase
        or prior.get("generation_evidence") != generation_evidence_inventory(request)
    ):
        raise PhaseResultsError("superseded repair does not bind frozen generation")
    prior_implementations = prior.get("implementations")
    if (
        not isinstance(prior_implementations, Mapping)
        or not isinstance(prior_implementations.get("repaired_phase_results"), Mapping)
        or prior_implementations["repaired_phase_results"].get("sha256")
        != prior_phase_results_sha256
    ):
        raise PhaseResultsError("superseded repair phase implementation mismatch")
    if prior_contract_type == "safa_r9_evaluation_repair_contract_v2":
        _validate_superseded_repair(
            prior.get("supersedes"),
            request=request,
            repo_root=repo_root,
            prior_phase_results_sha256=str(
                prior_implementations.get("prior_phase_results_sha256")
            ),
        )
    failure = value.get("failure")
    if not isinstance(failure, Mapping) or set(failure) != {
        "exception_type",
        "symbol",
        "controller_log",
    }:
        raise PhaseResultsError("superseded repair failure fields are invalid")
    if failure.get("exception_type") != expected_failure[
        "exception_type"
    ] or failure.get("symbol") != expected_failure["symbol"]:
        raise PhaseResultsError("superseded repair failure identity changed")
    log_path = _validate_repair_implementation_binding(
        failure.get("controller_log"),
        repo_root=repo_root,
        label="superseded controller log",
    )
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise PhaseResultsError("superseded controller log is not UTF-8") from error
    if expected_failure["message"] not in log_text:
        raise PhaseResultsError("superseded controller log lacks the registered failure")
    attempt = value.get("evaluation_attempt")
    if not isinstance(attempt, Mapping):
        raise PhaseResultsError("superseded evaluation attempt is invalid")
    namespace_root = attempt.get("namespace_root")
    if not isinstance(namespace_root, str):
        raise PhaseResultsError("superseded evaluation namespace is invalid")
    if attempt != evaluation_attempt_inventory(repo_root, Path(namespace_root)):
        raise PhaseResultsError("superseded evaluation inventory mismatch")
    if {field: attempt.get(field) for field in expected_counts} != expected_counts:
        raise PhaseResultsError("superseded evaluation result counts changed")
    if value.get("policy") != {
        "prior_repair_usage": "input_evidence_only",
        "prior_evaluation_results_usage": "input_evidence_only",
        "prior_namespace_reuse": False,
    }:
        raise PhaseResultsError("superseded evaluation repair policy changed")


def _validate_repair_file_binding(
    value: Any,
    *,
    repo_root: Path,
    digest_field: str,
    contract_type: str,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "contract_sha256",
    }:
        raise PhaseResultsError(f"{label} binding fields are invalid")
    path = _contained_file(repo_root, Path(str(value["path"])), label)
    payload = _read_digest_contract(
        path, digest_field=digest_field, contract_type=contract_type
    )
    if (
        _sha256_file(path) != value["file_sha256"]
        or payload[digest_field] != value["contract_sha256"]
    ):
        raise PhaseResultsError(f"{label} binding mismatch")
    return dict(value)


def _validate_repair_implementation_binding(
    value: Any, *, repo_root: Path, label: str
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PhaseResultsError(f"evaluation repair {label} binding is invalid")
    path = _contained_file(
        repo_root, Path(str(value["path"])), f"evaluation repair {label}"
    )
    if _sha256_file(path) != value["sha256"]:
        raise PhaseResultsError(f"evaluation repair {label} SHA256 mismatch")
    return path.resolve()


def _run_plan_payload(request: PhaseResultsRequest) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": request.phase,
        "expected_candidate_arm_ids": list(request.expected_candidate_arm_ids),
        "expected_seeds": list(request.expected_seeds),
        "runs": [
            {
                "logical_run_id": spec.logical_run_id,
                "arm_id": spec.arm_id,
                "family": spec.family,
                "seed": spec.seed,
                "repeat_index": spec.repeat_index,
                "shard_output_dirs": [
                    str(path.resolve()) for path in spec.shard_output_dirs
                ],
            }
            for spec in sorted(request.runs, key=lambda row: row.logical_run_id)
        ],
    }


def _load_visual_evidence(path: Path) -> dict[str, Any]:
    evidence = _read_digest_contract(
        path,
        digest_field="evidence_contract_sha256",
        contract_type=None,
    )
    expected_keys = {
        "schema_version",
        "columns",
        "sample_count",
        "sample_id_manifest",
        "sample_id_manifest_sha256",
        "ordered_sample_id_sha256",
        "pages",
        "samples",
        "evidence_contract_sha256",
    }
    if set(evidence) != expected_keys or evidence.get("schema_version") != 1:
        raise PhaseResultsError("visual evidence fields are not canonical")
    if evidence.get("columns") != ["source", "native", "candidate"]:
        raise PhaseResultsError("visual evidence columns are not canonical")
    manifest = Path(str(evidence["sample_id_manifest"]))
    if _sha256_file(manifest) != evidence["sample_id_manifest_sha256"]:
        raise PhaseResultsError("visual evidence manifest was replaced")
    sample_ids = []
    for page in evidence.get("pages", ()):
        page_path = Path(str(page["path"]))
        if _sha256_file(page_path) != page.get("sha256"):
            raise PhaseResultsError("visual evidence page was replaced")
    for row in evidence.get("samples", ()):
        sample_ids.append(row["sample_id"])
        for asset in row.get("assets", {}).values():
            if _sha256_file(Path(str(asset["path"]))) != asset.get("sha256"):
                raise PhaseResultsError("visual evidence image was replaced")
    if len(sample_ids) != evidence["sample_count"] or len(set(sample_ids)) != len(
        sample_ids
    ):
        raise PhaseResultsError("visual evidence sample coverage is invalid")
    return evidence


def _read_digest_contract(
    path: Path, *, digest_field: str, contract_type: str | None
) -> dict[str, Any]:
    payload = _read_json_mapping(path, "digest contract")
    if contract_type is not None and payload.get("contract_type") != contract_type:
        raise PhaseResultsError(f"contract type must be {contract_type!r}")
    declared = _require_sha256(payload.get(digest_field), digest_field)
    if _canonical_digest(payload, digest_field) != declared:
        raise PhaseResultsError(f"{digest_field} mismatch")
    _assert_finite_json(payload, "digest contract")
    return payload


def _verify_external_digest(payload: Mapping[str, Any], digest_field: str) -> None:
    declared = _require_sha256(payload.get(digest_field), digest_field)
    if _canonical_digest(payload, digest_field) != declared:
        raise PhaseResultsError(f"{digest_field} mismatch")


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.is_symlink():
        raise PhaseResultsError("exclusive review path must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise PhaseResultsError("exclusive review parent must not be a symlink")
    content = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("exclusive review write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _contained_path(root: Path, path: Path, label: str) -> Path:
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PhaseResultsError(f"{label} escapes repo root") from error
    return candidate


def _load_source_index_contract(
    repo_root: Path, index_path: Path, expected_sha256: str
) -> tuple[Path, dict[str, Path]]:
    locked_index = _contained_file(repo_root, index_path, "source index")
    if _sha256_file(locked_index) != expected_sha256:
        raise PhaseResultsError("source index SHA256 mismatch")
    source_paths: dict[str, Path] = {}
    for index, row in enumerate(_read_jsonl(locked_index, "source index")):
        sample_id = row.get("sample_id")
        image_path = row.get("image_path")
        if not isinstance(sample_id, str) or not sample_id or "\0" in sample_id:
            raise PhaseResultsError(
                f"source index row {index} has an invalid sample ID"
            )
        if sample_id in source_paths:
            raise PhaseResultsError("source index repeats a sample ID")
        if not isinstance(image_path, str) or not image_path:
            raise PhaseResultsError(
                f"source index row {index} has an invalid image path"
            )
        declared = Path(image_path)
        source = declared if declared.is_absolute() else repo_root / declared
        if source.is_symlink():
            raise FileNotFoundError(
                f"source index image must not be a symlink: {source}"
            )
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"source index image is not a regular file: {source}"
            )
        source_paths[sample_id] = source
    return locked_index, source_paths


def _contained_file(root: Path, path: Path, label: str) -> Path:
    candidate = _contained_path(root, path, label)
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {candidate}")
    return candidate


def _bound_output_file(
    repo_root: Path, output_root: Path, path: Path, label: str
) -> Path:
    candidate = _contained_file(repo_root, path, label)
    try:
        candidate.relative_to(output_root)
    except ValueError as error:
        raise PhaseResultsError(
            f"{label} escapes its shard output: {candidate}"
        ) from error
    return candidate


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhaseResultsError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise PhaseResultsError(f"{label} must contain a mapping")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PhaseResultsError(
                f"{label} {path}:{line_number} contains invalid JSON"
            ) from error
        if not isinstance(row, dict):
            raise PhaseResultsError(f"{label} row {line_number} must be a mapping")
        rows.append(row)
    if not rows:
        raise PhaseResultsError(f"{label} must not be empty")
    return rows


def _ordered_sample_ids(rows: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    values = []
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise PhaseResultsError(f"{label} row {index} sample_id is invalid")
        values.append(sample_id)
    if len(set(values)) != len(values):
        raise PhaseResultsError(f"{label} contains duplicate sample IDs")
    return values


def _reject_derived_input_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_DERIVED_INPUT_FIELDS & set(value)
        if forbidden:
            raise PhaseResultsError(
                f"input contains derived fields: {sorted(forbidden)!r}"
            )
        for nested in value.values():
            _reject_derived_input_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_derived_input_fields(nested)


def _assert_finite_json(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PhaseResultsError(f"{label} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_finite_json(nested, label)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite_json(nested, label)
        return
    raise PhaseResultsError(f"{label} contains non-JSON data")


def _json_roundtrip(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise PhaseResultsError(f"{label} must contain finite JSON data") from error


def _canonical_json_sha256(value: Any) -> str:
    normalized = _json_roundtrip(value, "canonical payload")
    return hashlib.sha256(
        json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _canonical_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return _canonical_json_sha256(canonical)


def _canonical_utf8_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _asset_manifest_digest(
    samples: Sequence[SampleEvidence], role: Literal["source", "native", "candidate"]
) -> str:
    return hashlib.sha256(
        "".join(
            f"{sample.sample_id}\t{_sha256_file(getattr(sample, role))}\n"
            for sample in samples
        ).encode("utf-8")
    ).hexdigest()


def _require_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PhaseResultsError(f"{label} must be a filesystem-safe non-empty ID")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PhaseResultsError(f"{label} must be a lowercase SHA256 digest")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PhaseResultsError(f"{label} must be an integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    parsed = _strict_int(value, label)
    if parsed < 0:
        raise PhaseResultsError(f"{label} must be non-negative")
    return parsed


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseResultsError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PhaseResultsError(f"{label} must be finite")
    return parsed


def _cosine(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if not -1.0 <= parsed <= 1.0:
        raise PhaseResultsError(f"{label} must be in [-1,1]")
    return parsed
