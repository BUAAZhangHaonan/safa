#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safa.evaluation.r9_evaluator_worker import (
    ProductionEvaluatorConfig,
    build_worker_request,
)
from safa.evaluation.r9_phase_results import ArcFaceEvaluationRequest, SampleEvidence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = (
    REPO_ROOT
    / "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9/full/evaluator_runs/arcface/winner/request.json"
)
ARM_IDS = (
    "eta0p125_baseline",
    "eta0p125_disable_i1",
    "eta0p125_disable_i2",
    "eta0p125_disable_i3",
)


class Fixed32ArcFaceRequestError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise Fixed32ArcFaceRequestError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Fixed32ArcFaceRequestError(f"{label} must be an object")
    return value


def _jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise Fixed32ArcFaceRequestError(f"{label} is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise Fixed32ArcFaceRequestError(
                f"{label} contains a blank row at line {line_number}"
            )
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise Fixed32ArcFaceRequestError(
                f"{label} row {line_number} must be an object"
            )
        rows.append(value)
    return rows


def _resolve_file(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Fixed32ArcFaceRequestError(f"{label} path is invalid")
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    if not path.is_file():
        raise Fixed32ArcFaceRequestError(f"{label} is missing: {path}")
    return path


def _ordered_ids(rows: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    values = [row.get("sample_id") for row in rows]
    if (
        len(values) != 32
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != 32
    ):
        raise Fixed32ArcFaceRequestError(
            f"{label} must contain exactly 32 unique ordered sample IDs"
        )
    return [str(value) for value in values]


def _production_config(
    template: Mapping[str, Any], *, device: str, work_root: Path
) -> ProductionEvaluatorConfig:
    config = template.get("config")
    if not isinstance(config, Mapping) or set(config) != {
        "repo_root",
        "device",
        "work_root",
        "quality_script",
        "arcface",
        "worker_contract",
        "batch_size",
    }:
        raise Fixed32ArcFaceRequestError(
            "authoritative template evaluator config is not canonical"
        )
    if Path(str(config["repo_root"])).resolve() != REPO_ROOT:
        raise Fixed32ArcFaceRequestError(
            "authoritative template repository root disagrees with this checkout"
        )
    return ProductionEvaluatorConfig(
        repo_root=REPO_ROOT,
        device=device,
        work_root=work_root,
        quality_script=config["quality_script"],
        arcface=config["arcface"],
        worker_contract=config["worker_contract"],
        batch_size=config["batch_size"],
    )


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    canonical = dict(value)
    canonical.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _template(path: Path) -> Mapping[str, Any]:
    template = _json(path, "authoritative ArcFace template request")
    if (
        template.get("schema_version") != 1
        or template.get("contract_type") != "safa_r9_phase_evaluator_request_v1"
        or template.get("task") != "arcface"
        or template.get("evaluator_request_sha256")
        != _canonical_digest(template, "evaluator_request_sha256")
    ):
        raise Fixed32ArcFaceRequestError(
            "authoritative ArcFace template request identity/digest mismatch"
        )
    return template


def _source_index(template: Mapping[str, Any]) -> tuple[Path, str]:
    payload = template.get("payload")
    if not isinstance(payload, Mapping):
        raise Fixed32ArcFaceRequestError(
            "authoritative template ArcFace payload is missing"
        )
    path = _resolve_file(payload.get("source_index_path"), "source index")
    declared = payload.get("source_index_sha256")
    if _sha256(path) != declared:
        raise Fixed32ArcFaceRequestError("source index digest mismatch")
    return path, str(declared)


def _samples(
    path: Path, *, expected_ids: Sequence[str], label: str
) -> tuple[SampleEvidence, ...]:
    rows = _jsonl(path, label)
    if _ordered_ids(rows, label) != list(expected_ids):
        raise Fixed32ArcFaceRequestError(
            f"{label} does not match the fixed32 selection order"
        )
    samples = []
    for row in rows:
        sample_id = str(row["sample_id"])
        source = _resolve_file(row.get("source"), f"{label} source {sample_id}")
        native = _resolve_file(row.get("native"), f"{label} native {sample_id}")
        candidate = _resolve_file(
            row.get("generated"), f"{label} candidate {sample_id}"
        )
        samples.append(
            SampleEvidence(
                sample_id=sample_id,
                source=source,
                native=native,
                candidate=candidate,
                source_sha256=_sha256(source),
                native_sha256=_sha256(native),
                candidate_sha256=_sha256(candidate),
            )
        )
    return tuple(samples)


def build_requests(
    *,
    diagnostic_manifest: Path,
    selection_manifest: Path,
    runs_root: Path,
    template_request: Path,
    output_root: Path,
    device: str,
) -> list[Path]:
    diagnostic_path = diagnostic_manifest.resolve()
    diagnostic = _json(diagnostic_path, "fixed32 diagnostic manifest")
    if (
        diagnostic.get("contract_type")
        != "safa_r10_triangle_fixed32_diagnostic_preparation_v1"
        or diagnostic.get("registration") != "new_diagnostic_not_r9_continuation"
    ):
        raise Fixed32ArcFaceRequestError("fixed32 diagnostic manifest is invalid")
    arms = diagnostic.get("arms")
    if not isinstance(arms, list) or [
        arm.get("arm_id") for arm in arms if isinstance(arm, Mapping)
    ] != list(ARM_IDS):
        raise Fixed32ArcFaceRequestError("fixed32 arm IDs/order disagree with the lock")
    shared = diagnostic.get("shared")
    if not isinstance(shared, Mapping):
        raise Fixed32ArcFaceRequestError("fixed32 shared contract is missing")

    selection_path = selection_manifest.resolve()
    expected_manifest = _resolve_file(
        shared.get("sample_id_manifest"), "locked fixed32 selection manifest"
    )
    if selection_path != expected_manifest:
        raise Fixed32ArcFaceRequestError(
            "selection manifest path disagrees with the diagnostic lock"
        )
    if _sha256(selection_path) != shared.get("sample_id_manifest_sha256"):
        raise Fixed32ArcFaceRequestError(
            "selection manifest digest disagrees with the diagnostic lock"
        )
    selection_ids = _ordered_ids(
        _jsonl(selection_path, "fixed32 selection manifest"),
        "fixed32 selection manifest",
    )
    seed = shared.get("sampling_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Fixed32ArcFaceRequestError("fixed32 sampling seed is invalid")

    template_path = template_request.resolve()
    template = _template(template_path)
    source_index_path, source_index_sha256 = _source_index(template)
    resolved_runs_root = runs_root.resolve()
    resolved_output_root = output_root.resolve()
    requests: list[tuple[str, dict[str, Any]]] = []
    for arm_id in ARM_IDS:
        run_root = resolved_runs_root / arm_id
        generation = _json(run_root / "generation_result.json", f"{arm_id} generation")
        if generation.get("status") != "complete" or generation.get("sample_count") != 32:
            raise Fixed32ArcFaceRequestError(f"{arm_id} generation is not complete")
        samples = _samples(
            run_root / "per_sample.jsonl",
            expected_ids=selection_ids,
            label=f"{arm_id} per-sample evidence",
        )
        config = _production_config(
            template,
            device=device,
            work_root=resolved_output_root / arm_id / "work",
        )
        request = ArcFaceEvaluationRequest(
            phase="diagnose",
            logical_run_id=f"r10_triangle_fixed32__{arm_id}",
            arm_id=arm_id,
            seed=seed,
            source_index_path=source_index_path,
            source_index_sha256=source_index_sha256,
            samples=samples,
        )
        requests.append((arm_id, build_worker_request("arcface", request, config=config)))

    output_paths = []
    for arm_id, payload in requests:
        arm_root = resolved_output_root / arm_id
        arm_root.mkdir(parents=True, exist_ok=False)
        request_path = arm_root / "request.json"
        with request_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        output_paths.append(request_path)
    build_manifest = {
        "schema_version": 1,
        "contract_type": "safa_r10_triangle_fixed32_arcface_request_build_v1",
        "diagnostic_manifest": {
            "path": str(diagnostic_path),
            "sha256": _sha256(diagnostic_path),
        },
        "selection_manifest": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
        },
        "authoritative_template_request": {
            "path": str(template_path),
            "sha256": _sha256(template_path),
            "evaluator_request_sha256": template["evaluator_request_sha256"],
        },
        "device": device,
        "requests": [
            {
                "arm_id": arm_id,
                "path": str(request_path),
                "sha256": _sha256(request_path),
                "evaluator_request_sha256": payload["evaluator_request_sha256"],
            }
            for (arm_id, payload), request_path in zip(
                requests, output_paths, strict=True
            )
        ],
    }
    manifest_path = resolved_output_root / "request_build_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(build_manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable official buffalo_l requests for fixed32 arms."
    )
    parser.add_argument("--diagnostic-manifest", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument(
        "--template-request", type=Path, default=DEFAULT_RUNTIME_CONFIG
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = build_requests(
        diagnostic_manifest=args.diagnostic_manifest,
        selection_manifest=args.selection_manifest,
        runs_root=args.runs_root,
        template_request=args.template_request,
        output_root=args.output_root,
        device=args.device,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
