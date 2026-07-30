from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType

import pytest

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    RUN_MODES,
    canonical_digest,
)
from safa.closeout import canonical_screening_worker as worker


ROOT = Path(__file__).resolve().parents[1]
PREPARER_PATH = ROOT / "scripts/prepare_triangle32_checkpoint_pilot.py"
SELECTED24 = (
    ROOT
    / "artifacts/r10_triangle_exploration/checkpoint_fixed32_pilot/"
    "selected24/selected24.json"
)
CANDIDATE_MANIFEST = (
    ROOT
    / "artifacts/closeout/historical-canonical-512-v1/"
    "candidate_manifest__5dbb82fdb1c89d8f.json"
)
PREFIX32 = ROOT / "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"
ELIGIBLE512 = (
    ROOT / "artifacts/r10_triangle_exploration/preparation_v1/eligible512.jsonl"
)
NATIVE = (
    ROOT
    / "artifacts/r10_triangle_exploration/fixed32_evaluation/"
    "inputs/native_per_sample.jsonl"
)
CANONICAL_REQUESTS = (
    ROOT
    / "artifacts/closeout/historical-canonical-512-v1/by_policy/"
    "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af/"
    "run_requests/smoke8_primary"
)
HISTORICAL_PRIMARY = (
    ROOT
    / "artifacts/closeout/historical-canonical-512-v1/by_policy/"
    "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af/"
    "runs/smoke8_primary"
)


def _preparer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prepare_triangle32_checkpoint_pilot", PREPARER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def preparation() -> dict[str, object]:
    module = _preparer()
    parent = ROOT / "artifacts/r10_triangle_exploration"
    with tempfile.TemporaryDirectory(prefix=".triangle32-test-", dir=parent) as temp:
        temp_root = Path(temp)
        output = temp_root / "preparation"
        summary = module.prepare(
            selected24_path=SELECTED24,
            candidate_manifest_path=CANDIDATE_MANIFEST,
            historical_primary_root=HISTORICAL_PRIMARY,
            prefix32_path=PREFIX32,
            native_per_sample_path=NATIVE,
            canonical_request_root=CANONICAL_REQUESTS,
            runs_root=temp_root / "runs",
            logs_root=temp_root / "logs",
            output_dir=output,
        )
        yield {"root": temp_root, "output": output, "summary": summary}


def test_old_modes_are_unchanged_and_queues_cover_heterogeneous_selected24(
    preparation: dict[str, object],
) -> None:
    assert RUN_MODES == {"smoke8": 8, "screen512": 512}
    output = preparation["output"]
    assert isinstance(output, Path)
    jobs = []
    for gpu_index in range(4):
        queue = json.loads(
            (output / f"queue_gpu{gpu_index}.json").read_text(encoding="utf-8")
        )
        assert queue["gpu_index"] == gpu_index
        assert queue["job_count"] == 6
        assert queue["retry_count"] == 0
        assert queue["launchable"] is True
        assert all(job["gpu_index"] == gpu_index for job in queue["jobs"])
        jobs.extend(queue["jobs"])
    assert sorted(job["selection_rank"] for job in jobs) == list(range(1, 25))
    assert len({job["candidate_id"] for job in jobs}) == 24
    assert len({job["output_dir"] for job in jobs}) == 24
    assert len({job["log_path"] for job in jobs}) == 24
    requests = [
        json.loads(Path(job["request_path"]).read_text(encoding="utf-8"))
        for job in jobs
    ]
    assert {request["candidate"]["checkpoint_model"] for request in requests} == {
        "raw",
        "ema",
    }
    assert {
        request["output_contract"]["capability"]["output_space"]
        for request in requests
    } == {"latent", "pixel"}
    assert all("native_per_sample" in request for request in requests)
    assert all("arcface" not in request for request in requests)
    assert all("quality_script" not in request for request in requests)
    pixel = next(
        request
        for request in requests
        if request["candidate"]["candidate_id"] == "g_10b333a6c4b88e46_raw"
    )
    assert pixel["nfe"] == 63
    assert pixel["native_rgb_size"] == [224, 224]
    lora_raw = next(
        request
        for request in requests
        if "legacy_lora_loader_bug"
        in request["candidate"]["protocol_families"]
    )
    assert lora_raw["candidate"]["checkpoint_model"] == "raw"
    assert (
        lora_raw["candidate"]["preflight_result_sha256"]
        and lora_raw["candidate"]["output_contract"]
    )


def test_triangle32_rejects_wrong_cardinality_prefix_and_missing_native(
    preparation: dict[str, object],
) -> None:
    output = preparation["output"]
    assert isinstance(output, Path)
    request = json.loads(
        next((output / "requests").glob("*.json")).read_text(encoding="utf-8")
    )
    wrong_count = dict(request)
    wrong_count["sample_count"] = 31
    wrong_count["run_request_sha256"] = canonical_digest(
        wrong_count, "run_request_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="frozen fields"):
        worker.validate_triangle32_request(wrong_count)

    wrong_prefix = dict(request)
    wrong_prefix["sample_manifest"] = {
        "path": str(ELIGIBLE512),
        "sha256": _sha(ELIGIBLE512),
        "sample_count": 512,
    }
    wrong_prefix["run_request_sha256"] = canonical_digest(
        wrong_prefix, "run_request_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="registered prefix32"):
        worker.validate_triangle32_request(wrong_prefix)

    missing_native = dict(request)
    del missing_native["native_per_sample"]
    missing_native["run_request_sha256"] = canonical_digest(
        missing_native, "run_request_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="fields are not canonical"):
        worker.validate_triangle32_request(missing_native)


def test_generation_only_result_merges_native_role(
    preparation: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = preparation["output"]
    root = preparation["root"]
    assert isinstance(output, Path)
    assert isinstance(root, Path)
    request_path = next((output / "requests").glob("*.json"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_id = request["candidate"]["candidate_id"]
    run_output = root / "execute-test" / candidate_id
    request["output_dir"] = str(run_output)
    request["log_path"] = str(root / "execute-test.log")
    request["run_request_sha256"] = canonical_digest(
        request, "run_request_sha256"
    )
    test_request = root / "execute-request.json"
    test_request.write_text(json.dumps(request) + "\n", encoding="utf-8")
    native_rows = [
        json.loads(line) for line in NATIVE.read_text(encoding="utf-8").splitlines()
    ]
    generated_rows = [
        {
            "sample_id": row["sample_id"],
            "source_path": row["source"],
            "candidate_path": row["native"],
        }
        for row in native_rows
    ]
    monkeypatch.setenv("TMUX", "triangle32-test")
    monkeypatch.setattr(worker, "_assert_runtime_cuda_binding", lambda *args: {})
    monkeypatch.setattr(
        worker,
        "_run_generation",
        lambda *args: (generated_rows, [], []),
    )
    result = worker.execute_triangle32_request(
        test_request,
        request["authorized_gpu"]["physical_gpu_index"],
        request["authorized_gpu"]["physical_gpu_uuid"],
    )
    assert result["status"] == "completed"
    assert result["generation_only"] is True
    assert result["evaluation_status"] == "not_started"
    rows = [
        json.loads(line)
        for line in (run_output / "per_sample.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 32
    assert all("native" in row and "native_sha256" in row for row in rows)
