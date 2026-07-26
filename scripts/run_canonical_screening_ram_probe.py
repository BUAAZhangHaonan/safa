#!/usr/bin/env python3
"""One-shot, non-scientific RAM probe for the canonical screening worker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from run_canonical_checkpoint_screening import (
    REPO_ROOT,
    _disk_percent,
    _gpu_compute_processes,
    _gpu_hard_resource_violation,
    _gpu_snapshot,
    _memory_percent,
    _worker_environment,
    assert_cpu_resource_admission,
)
from run_r9_meanflow_campaign import (
    _process_tree_rss_bytes,
    _sample_or_reap_process_tree,
    _terminate_process,
)
from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    canonical_digest,
    load_json,
    sha256_file,
    validate_policy,
    write_exclusive_json,
)
from safa.closeout.canonical_screening_worker import (
    _assert_runtime_cuda_binding,
    _run_arcface,
    _run_generation,
    _run_quality,
)


PROBE_CONTRACT = "safa_canonical_screening_ram_probe_v1"
PROBE_RESULT_CONTRACT = "safa_canonical_screening_ram_probe_result_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--gpu-uuid")
    return parser.parse_args(argv)


def _validate_manifest_envelope(path: Path) -> dict[str, Any]:
    manifest = load_json(path, "candidate manifest")
    digest = manifest.get("candidate_manifest_sha256")
    if (
        not isinstance(digest, str)
        or digest != canonical_digest(manifest, "candidate_manifest_sha256")
    ):
        raise CanonicalScreeningError("probe candidate manifest digest mismatch")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CanonicalScreeningError("probe candidate manifest is empty")
    return manifest


def _select_probe_candidates(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for output_space in ("latent", "pixel"):
        rows = []
        for candidate in manifest["candidates"]:
            capability = candidate.get("output_contract", {}).get("capability", {})
            if capability.get("output_space") != output_space:
                continue
            checkpoint = Path(str(candidate["checkpoint_path"]))
            if not checkpoint.is_absolute():
                checkpoint = REPO_ROOT / checkpoint
            checkpoint = checkpoint.resolve()
            if not checkpoint.is_file():
                raise CanonicalScreeningError(
                    f"probe checkpoint is missing: {checkpoint}"
                )
            rows.append((checkpoint.stat().st_size, str(candidate["candidate_id"]), candidate))
        if not rows:
            raise CanonicalScreeningError(
                f"probe manifest has no {output_space} candidate"
            )
        size, _, candidate = max(rows, key=lambda row: (row[0], row[1]))
        checkpoint = Path(str(candidate["checkpoint_path"]))
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        checkpoint = checkpoint.resolve()
        if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
            raise CanonicalScreeningError(
                f"probe selected checkpoint binding mismatch: {checkpoint}"
            )
        selected.append(
            {
                "candidate_id": candidate["candidate_id"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "checkpoint_model": candidate["checkpoint_model"],
                "checkpoint_size_bytes": size,
                "output_space": output_space,
                "output_contract_sha256": candidate["output_contract"][
                    "output_contract_sha256"
                ],
            }
        )
    return selected


def _build_spec(
    policy: Mapping[str, Any],
    config: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    artifact_root: Path,
    gpu_registry: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    spec = {
        "schema_version": 1,
        "contract_type": PROBE_CONTRACT,
        "purpose": "resource_measurement_only_scientific_reuse_forbidden",
        "policy": {
            "path": str(config.resolve()),
            "sha256": sha256_file(config),
            "canonical_sha256": policy["policy_sha256"],
        },
        "candidate_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "canonical_sha256": manifest["candidate_manifest_sha256"],
        },
        "selected_candidates": _select_probe_candidates(manifest),
        "sample_manifest": dict(policy["protocol"]["manifests"]["smoke8"]),
        "sample_count": 8,
        "seed": 4549,
        "batch_size": 2,
        "authorized_gpu_registry": gpu_registry,
        "artifact_root": str(artifact_root.resolve()),
        "implementations": dict(policy["implementations"]),
        "retry_count": 0,
        "probe_sha256": None,
    }
    spec["probe_sha256"] = canonical_digest(spec, "probe_sha256")
    return spec


def _validate_spec(
    spec: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(spec)
    if (
        value.get("schema_version") != 1
        or value.get("contract_type") != PROBE_CONTRACT
        or value.get("purpose")
        != "resource_measurement_only_scientific_reuse_forbidden"
        or value.get("sample_count") != 8
        or value.get("seed") != 4549
        or value.get("batch_size") != 2
        or value.get("retry_count") != 0
        or value.get("implementations") != policy["implementations"]
        or value.get("probe_sha256")
        != canonical_digest(value, "probe_sha256")
    ):
        raise CanonicalScreeningError("RAM probe spec frozen fields differ")
    for label in ("policy", "candidate_manifest", "sample_manifest"):
        binding = value.get(label)
        if not isinstance(binding, Mapping):
            raise CanonicalScreeningError(f"RAM probe {label} binding is invalid")
        path = Path(str(binding.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise CanonicalScreeningError(f"RAM probe {label} file binding mismatch")
    if value["policy"]["canonical_sha256"] != policy["policy_sha256"]:
        raise CanonicalScreeningError("RAM probe policy binding mismatch")
    return value


def _vmhwm_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise CanonicalScreeningError("worker /proc/self/status omits VmHWM")


def _probe_request(
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    output_contract = candidate["output_contract"]
    return {
        "run_request_sha256": spec["probe_sha256"],
        "screening_worker": policy["implementations"]["screening_worker"],
        "candidate": dict(candidate),
        "output_contract": dict(output_contract),
        "output_decoder_registry": dict(policy["output_decoder_registry"]),
        "sample_manifest": dict(policy["protocol"]["manifests"]["smoke8"]),
        "source_index": dict(policy["protocol"]["source_index"]),
        "features": dict(policy["protocol"]["features"]),
        "e0": dict(policy["protocol"]["e0"]),
        "edev": dict(policy["protocol"]["edev"]),
        "pixel_image_size": policy["protocol"]["pixel_image_size"],
        "pixel_protocol_config": dict(policy["protocol"]["pixel_protocol_config"]),
        "quality_script": dict(policy["protocol"]["quality_script"]),
        "kid_subset_size": policy["protocol"]["kid_subset_sizes"]["smoke8"],
        "arcface": dict(policy["arcface"]),
        "candidate_manifest": {
            "canonical_sha256": manifest["candidate_manifest_sha256"]
        },
        "native_rgb_size": [
            output_contract["rgb_contract"]["height"],
            output_contract["rgb_contract"]["width"],
        ],
        "quality_protocol_family": output_contract["quality_protocol_family"],
        "nfe": output_contract["capability"]["nfe"],
    }


def _run_worker(
    spec_path: Path,
    config: Path,
    gpu_index: int,
    gpu_uuid: str,
) -> dict[str, Any]:
    policy = validate_policy(
        REPO_ROOT, config, verify_historical_output_evidence=False
    )
    spec = _validate_spec(load_json(spec_path, "RAM probe spec"), policy)
    registry = spec["authorized_gpu_registry"]
    if not isinstance(registry, list):
        raise CanonicalScreeningError("RAM probe GPU registry is missing")
    device_binding = _assert_runtime_cuda_binding(
        {"authorized_gpu_registry": registry}, gpu_index, gpu_uuid
    )
    manifest_path = Path(spec["candidate_manifest"]["path"])
    manifest = _validate_manifest_envelope(manifest_path)
    selected = {
        row["candidate_id"]: row for row in spec["selected_candidates"]
    }
    candidates = {
        row["candidate_id"]: row
        for row in manifest["candidates"]
        if row["candidate_id"] in selected
    }
    if set(candidates) != set(selected):
        raise CanonicalScreeningError("RAM probe selected candidates are missing")
    steps = []
    work_root = Path(spec["artifact_root"]) / "work"
    work_root.mkdir(parents=True, exist_ok=False)
    for descriptor in spec["selected_candidates"]:
        candidate = candidates[descriptor["candidate_id"]]
        if (
            candidate["checkpoint_sha256"] != descriptor["checkpoint_sha256"]
            or candidate["checkpoint_model"] != descriptor["checkpoint_model"]
            or candidate["output_contract"]["output_contract_sha256"]
            != descriptor["output_contract_sha256"]
        ):
            raise CanonicalScreeningError("RAM probe candidate binding mismatch")
        output_dir = work_root / descriptor["output_space"]
        output_dir.mkdir()
        request = _probe_request(policy, manifest, candidate, spec)
        rows, source_paths, candidate_paths = _run_generation(
            request, gpu_index, output_dir
        )
        _run_arcface(
            request, gpu_index, rows, source_paths, candidate_paths
        )
        _run_quality(request, gpu_index, output_dir, rows)
        pngs = sorted((output_dir / "generated").glob("*.png"))
        if len(rows) != 8 or len(pngs) != 8:
            raise CanonicalScreeningError("RAM probe workload coverage differs")
        steps.append(
            {
                **dict(descriptor),
                "sample_count": len(rows),
                "generated_png_manifest_sha256": canonical_digest(
                    {
                        "rows": [
                            {
                                "name": path.name,
                                "sha256": sha256_file(path),
                            }
                            for path in pngs
                        ]
                    },
                    "unused",
                ),
            }
        )
    result = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_worker_result_v1",
        "probe_sha256": spec["probe_sha256"],
        "purpose": spec["purpose"],
        "device_binding": device_binding,
        "steps": steps,
        "worker_vmhwm_bytes": _vmhwm_bytes(),
        "completed_at": _utc_now(),
    }
    result["worker_result_sha256"] = canonical_digest(
        result, "worker_result_sha256"
    )
    write_exclusive_json(Path(spec["artifact_root"]) / "worker_result.json", result)
    return result


def _run_controller(
    policy: Mapping[str, Any],
    config: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("RAM probe must run inside tmux")
    if policy["resources"]["ram_budget_status"] != "probe_required":
        raise CanonicalScreeningError("RAM probe requires probe_required policy")
    artifact_root.mkdir(parents=True, exist_ok=False)
    host = assert_cpu_resource_admission(policy, artifact_root)
    gpus = [
        row
        for row in _gpu_snapshot()
        if row["index"] in policy["resources"]["physical_gpus"]
    ]
    registry = [
        {
            "physical_gpu_index": row["index"],
            "physical_gpu_uuid": row["uuid"],
        }
        for row in gpus
    ]
    if [row["physical_gpu_index"] for row in registry] != [0, 1, 2, 3]:
        raise CanonicalScreeningError("RAM probe physical GPU registry differs")
    target_uuids = {row["physical_gpu_uuid"] for row in registry}
    busy = [
        row
        for row in _gpu_compute_processes()
        if row["gpu_uuid"] in target_uuids
    ]
    if busy:
        raise CanonicalScreeningError(f"RAM probe GPU admission is busy: {busy}")
    gpu0 = gpus[0]
    if (
        gpu0["memory_free_mib"]
        < policy["resources"]["gpu_headroom_bytes"] // 1024**2
    ):
        raise CanonicalScreeningError("RAM probe GPU0 headroom admission failed")
    spec = _build_spec(
        policy, config, manifest, manifest_path, artifact_root, registry
    )
    spec_path = artifact_root / "probe_spec.json"
    write_exclusive_json(spec_path, spec)
    admission = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_admission_v1",
        "probe_sha256": spec["probe_sha256"],
        "host": host,
        "gpu_snapshot": gpus,
        "authorized_gpu_registry": registry,
        "observed_at": _utc_now(),
    }
    admission["admission_sha256"] = canonical_digest(
        admission, "admission_sha256"
    )
    write_exclusive_json(artifact_root / "admission.json", admission)
    command = [
        str(policy["python"]),
        str(Path(__file__).resolve()),
        "--config",
        str(config.resolve()),
        "--candidate-manifest",
        str(manifest_path.resolve()),
        "--artifact-root",
        str(artifact_root.resolve()),
        "--execute",
        "--worker",
        "--spec",
        str(spec_path.resolve()),
        "--gpu-index",
        "0",
        "--gpu-uuid",
        registry[0]["physical_gpu_uuid"],
    ]
    log_path = artifact_root / "worker.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_worker_environment(registry[0]["physical_gpu_uuid"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        peak_rss = 0
        failure: str | None = None
        returncode: int | None = None
        while True:
            polled = process.poll()
            if polled is not None:
                returncode = polled
                break
            rss_bytes, reaped_returncode = _sample_or_reap_process_tree(
                process, _process_tree_rss_bytes
            )
            if reaped_returncode is not None:
                returncode = reaped_returncode
                break
            if rss_bytes is None:
                raise CanonicalScreeningError(
                    "running RAM probe process RSS sample is missing"
                )
            peak_rss = max(peak_rss, rss_bytes)
            if _memory_percent() >= 90:
                failure = "RAM runtime hard stop reached 90%"
            elif _disk_percent(artifact_root.parent) >= 90:
                failure = "disk runtime hard stop reached 90%"
            else:
                failure = _gpu_hard_resource_violation(policy)
            if failure is not None:
                _terminate_process(process)
                returncode = process.wait()
                break
            time.sleep(0.1)
    if returncode is None:
        raise CanonicalScreeningError("RAM probe worker was not reaped")
    if failure is None and returncode != 0:
        failure = f"RAM probe worker exited once with status {returncode}"
    worker_result_path = artifact_root / "worker_result.json"
    worker_result = (
        load_json(worker_result_path, "RAM probe worker result")
        if failure is None
        else None
    )
    if worker_result is not None:
        if (
            worker_result.get("probe_sha256") != spec["probe_sha256"]
            or worker_result.get("worker_result_sha256")
            != canonical_digest(worker_result, "worker_result_sha256")
        ):
            raise CanonicalScreeningError("RAM probe worker result mismatch")
        peak_rss = max(peak_rss, int(worker_result["worker_vmhwm_bytes"]))
    if failure is None and peak_rss <= 0:
        failure = "RAM probe measured no positive RSS/VmHWM"
    budget = (peak_rss * 11 + 9) // 10 if peak_rss > 0 else None
    result = {
        "schema_version": 1,
        "contract_type": PROBE_RESULT_CONTRACT,
        "status": "succeeded" if failure is None else "failed",
        "purpose": spec["purpose"],
        "probe_sha256": spec["probe_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "worker_result_sha256": (
            None if worker_result is None else worker_result["worker_result_sha256"]
        ),
        "worker_log_sha256": sha256_file(log_path),
        "peak_process_tree_rss_bytes": peak_rss,
        "worker_vmhwm_bytes": (
            None if worker_result is None else worker_result["worker_vmhwm_bytes"]
        ),
        "ram_slot_budget_bytes": budget,
        "budget_method": "ceil(single_worker_process_tree_peak_rss_bytes*11/10)",
        "measurement_factor_numerator": 11,
        "measurement_factor_denominator": 10,
        "failure": failure,
        "retry_count": 0,
        "completed_at": _utc_now(),
    }
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    write_exclusive_json(artifact_root / "probe_result.json", result)
    if failure is not None:
        raise CanonicalScreeningError(failure)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = args.config.resolve()
    manifest_path = args.candidate_manifest.resolve()
    artifact_root = args.artifact_root.resolve()
    policy = validate_policy(
        REPO_ROOT, config, verify_historical_output_evidence=False
    )
    manifest = _validate_manifest_envelope(manifest_path)
    if args.worker:
        if (
            not args.execute
            or args.spec is None
            or args.gpu_index is None
            or args.gpu_uuid is None
        ):
            raise CanonicalScreeningError("RAM probe worker arguments are incomplete")
        result = _run_worker(
            args.spec.resolve(), config, args.gpu_index, args.gpu_uuid
        )
    elif args.dry_run:
        result = _build_spec(
            policy, config, manifest, manifest_path, artifact_root, None
        )
    else:
        result = _run_controller(
            policy, config, manifest, manifest_path, artifact_root
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
