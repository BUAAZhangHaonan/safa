#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    canonical_digest,
    load_json,
    sha256_file,
)
from safa.closeout.canonical_screening_worker import (
    TRIANGLE32_REQUEST_CONTRACT,
    validate_triangle32_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "src/safa/closeout/canonical_screening_worker.py"
RUNNER = REPO_ROOT / "scripts/run_triangle32_screening_worker.py"
PYTHON = Path("/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python")


def _binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise CanonicalScreeningError(f"required file is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def prepare(
    *,
    selected24_path: Path,
    candidate_manifest_path: Path,
    historical_primary_root: Path,
    prefix32_path: Path,
    native_per_sample_path: Path,
    canonical_request_root: Path,
    runs_root: Path,
    logs_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selected24_path = selected24_path.resolve()
    candidate_manifest_path = candidate_manifest_path.resolve()
    historical_primary_root = historical_primary_root.resolve()
    prefix32_path = prefix32_path.resolve()
    native_per_sample_path = native_per_sample_path.resolve()
    canonical_request_root = canonical_request_root.resolve()
    runs_root = runs_root.resolve()
    logs_root = logs_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise CanonicalScreeningError(
            f"triangle32 preparation output already exists: {output_dir}"
        )
    selected = load_json(selected24_path, "selected24")
    selected_rows = selected.get("selected")
    if (
        selected.get("contract_type") != "safa_triangle_historical24_v1"
        or selected.get("candidate_count") != 193
        or selected.get("selected_count") != 24
        or not isinstance(selected_rows, list)
        or len(selected_rows) != 24
        or [row.get("selection_rank") for row in selected_rows]
        != list(range(1, 25))
        or len({row.get("candidate_id") for row in selected_rows}) != 24
    ):
        raise CanonicalScreeningError("selected24 is not the exact locked selection")
    manifest = load_json(candidate_manifest_path, "candidate manifest")
    if (
        manifest.get("contract_type")
        != "safa_canonical_screening_candidate_manifest_v1"
        or manifest.get("candidate_count") != 193
        or manifest.get("candidate_manifest_sha256")
        != canonical_digest(manifest, "candidate_manifest_sha256")
    ):
        raise CanonicalScreeningError(
            "candidate manifest is not the authoritative 193 contract"
        )
    candidates = {
        row["candidate_id"]: row for row in manifest.get("candidates", [])
    }
    if len(candidates) != 193:
        raise CanonicalScreeningError("candidate manifest IDs are not unique")
    primary_results = sorted(historical_primary_root.glob("*/result.json"))
    if (
        not historical_primary_root.is_dir()
        or len(primary_results) != 193
        or {
            path.parent.name for path in primary_results
        } != set(candidates)
    ):
        raise CanonicalScreeningError(
            "historical primary evidence does not map exactly to 193 candidates"
        )
    for path in primary_results:
        result = load_json(path, "historical primary result")
        evidence = result.get("evidence")
        if (
            result.get("status") != "completed"
            or result.get("failure") is not None
            or not isinstance(evidence, Mapping)
            or evidence.get("mode") != "smoke8"
            or evidence.get("replicate") != "primary"
            or evidence.get("sample_count") != 8
        ):
            raise CanonicalScreeningError(
                f"historical primary result is not authoritative: {path}"
            )
    manifest_binding = {
        **_binding(candidate_manifest_path),
        "canonical_sha256": manifest["candidate_manifest_sha256"],
    }
    selected_binding = _binding(selected24_path)
    sample_binding = {**_binding(prefix32_path), "sample_count": 32}
    native_binding = _binding(native_per_sample_path)
    worker_binding = _binding(WORKER)
    runner_binding = _binding(RUNNER)

    requests: list[dict[str, Any]] = []
    queues: list[list[dict[str, Any]]] = [[], [], [], []]
    for row in selected_rows:
        rank = row["selection_rank"]
        candidate_id = row["candidate_id"]
        candidate = candidates.get(candidate_id)
        if not isinstance(candidate, Mapping):
            raise CanonicalScreeningError(
                f"selected candidate is absent from manifest: {candidate_id}"
            )
        template_path = canonical_request_root / f"{candidate_id}.json"
        template = load_json(template_path, f"{candidate_id} canonical request")
        if template.get("candidate") != dict(candidate):
            raise CanonicalScreeningError(
                f"{candidate_id} canonical request candidate semantics differ"
            )
        gpu_index = (rank - 1) % 4
        registry = template.get("authorized_gpu_registry")
        if (
            not isinstance(registry, list)
            or [item.get("physical_gpu_index") for item in registry]
            != [0, 1, 2, 3]
        ):
            raise CanonicalScreeningError(
                f"{candidate_id} authorized GPU registry differs"
            )
        authorized_gpu = registry[gpu_index]
        output_path = runs_root / candidate_id
        log_path = logs_root / f"{candidate_id}.log"
        request = {
            "schema_version": 1,
            "contract_type": TRIANGLE32_REQUEST_CONTRACT,
            "mode": "triangle32",
            "sample_count": 32,
            "seed": 4549,
            "batch_size": 2,
            "retry_count": 0,
            "selection_rank": rank,
            "selected24": selected_binding,
            "candidate_manifest": manifest_binding,
            "candidate": dict(candidate),
            "output_decoder_registry": template["output_decoder_registry"],
            "output_contract": candidate["output_contract"],
            "sample_manifest": sample_binding,
            "native_per_sample": native_binding,
            "source_index": template["source_index"],
            "features": template["features"],
            "e0": template["e0"],
            "edev": template["edev"],
            "pixel_image_size": template["pixel_image_size"],
            "screening_worker": worker_binding,
            "runner": runner_binding,
            "quality_protocol_family": candidate["output_contract"][
                "quality_protocol_family"
            ],
            "native_rgb_size": [
                candidate["output_contract"]["rgb_contract"]["height"],
                candidate["output_contract"]["rgb_contract"]["width"],
            ],
            "nfe": candidate["output_contract"]["capability"]["nfe"],
            "authorized_gpu": authorized_gpu,
            "authorized_gpu_registry": registry,
            "output_dir": str(output_path),
            "log_path": str(log_path),
        }
        request["run_request_sha256"] = canonical_digest(
            request, "run_request_sha256"
        )
        validate_triangle32_request(request)
        request_path = output_dir / "requests" / f"{candidate_id}.json"
        _write_json(request_path, request)
        entry = {
            "queue_position": len(queues[gpu_index]) + 1,
            "selection_rank": rank,
            "candidate_id": candidate_id,
            "gpu_index": gpu_index,
            "gpu_uuid": authorized_gpu["physical_gpu_uuid"],
            "environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": authorized_gpu["physical_gpu_uuid"],
            },
            "command": [
                str(PYTHON),
                str(RUNNER),
                "--request",
                str(request_path),
                "--gpu-index",
                str(gpu_index),
                "--gpu-uuid",
                authorized_gpu["physical_gpu_uuid"],
            ],
            "request_path": str(request_path),
            "run_request_sha256": request["run_request_sha256"],
            "output_dir": str(output_path),
            "log_path": str(log_path),
            "retry_count": 0,
        }
        queues[gpu_index].append(entry)
        requests.append(request)

    if (
        len(requests) != 24
        or any(len(queue) != 6 for queue in queues)
        or len({item["output_dir"] for queue in queues for item in queue}) != 24
        or len({item["log_path"] for queue in queues for item in queue}) != 24
    ):
        raise CanonicalScreeningError("triangle32 queue coverage is not exact")
    queue_paths = []
    for gpu_index, entries in enumerate(queues):
        queue = {
            "schema_version": 1,
            "contract_type": "safa_r10_triangle32_gpu_queue_v1",
            "gpu_index": gpu_index,
            "job_count": 6,
            "retry_count": 0,
            "sequential": True,
            "requires_tmux": True,
            "launchable": True,
            "generation_only": True,
            "jobs": entries,
        }
        path = output_dir / f"queue_gpu{gpu_index}.json"
        _write_json(path, queue)
        queue_paths.append(str(path))
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r10_triangle32_pilot_preparation_v1",
        "status": "prepared_not_launched",
        "candidate_count": 24,
        "queue_count": 4,
        "jobs_per_queue": 6,
        "retry_count": 0,
        "batch_size": 2,
        "seed": 4549,
        "generation_only": True,
        "post_generation_evaluation": (
            "official NIQE/sharpness plus three-role R9 ArcFace; "
            "source-candidate-only evidence is forbidden"
        ),
        "selected24": selected_binding,
        "selection_source": {
            "historical_primary_root": str(historical_primary_root),
            "completed_primary_result_count": 193,
        },
        "candidate_manifest": manifest_binding,
        "sample_manifest": sample_binding,
        "native_per_sample": native_binding,
        "worker": worker_binding,
        "runner": runner_binding,
        "queue_paths": queue_paths,
    }
    _write_json(output_dir / "preparation_manifest.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare direct generation-only triangle32 pilot queues."
    )
    parser.add_argument("--selected24", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--historical-primary-root", required=True, type=Path)
    parser.add_argument("--prefix32", required=True, type=Path)
    parser.add_argument("--native-per-sample", required=True, type=Path)
    parser.add_argument("--canonical-request-root", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--logs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(
        selected24_path=args.selected24,
        candidate_manifest_path=args.candidate_manifest,
        historical_primary_root=args.historical_primary_root,
        prefix32_path=args.prefix32,
        native_per_sample_path=args.native_per_sample,
        canonical_request_root=args.canonical_request_root,
        runs_root=args.runs_root,
        logs_root=args.logs_root,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
