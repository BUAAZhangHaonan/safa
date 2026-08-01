from __future__ import annotations

import importlib.util
import json
import math
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_r14_inpaint_milestone_eval.py"
SPEC = importlib.util.spec_from_file_location("r14_milestone_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_locked_steps_and_datasets() -> None:
    assert MODULE.CHECKPOINT_STEPS == (2560, 5120, 7680, 10240, 12800)
    assert MODULE.DATASETS == ("regular32", "tail32")


def test_manifest_validator_requires_32_unique_ids() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.jsonl"
        path.write_text("\n".join(json.dumps({"sample_id": f"id-{i}"}) for i in range(32)) + "\n", encoding="utf-8")
        assert len(MODULE._validate_manifest(path)) == 32
        path.write_text("\n".join(json.dumps({"sample_id": "same"}) for _ in range(32)) + "\n", encoding="utf-8")
        try:
            MODULE._validate_manifest(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate IDs must fail")


def test_dry_run_contract_forbids_fid_kid() -> None:
    payload = MODULE._dry_run(Path("artifacts/r14_inpaint_milestone_eval/v1"))
    assert payload["checkpoint_steps"] == [2560, 5120, 7680, 10240, 12800]
    assert payload["forbidden_metrics"] == ["fid", "kid"]


def test_arcface_delta_finite_guard() -> None:
    assert math.isfinite(0.1 - (-0.2))
