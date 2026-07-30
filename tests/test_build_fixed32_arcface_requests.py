from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_fixed32_arcface_requests.py"
DIAGNOSTIC = (
    ROOT
    / "artifacts/r10_triangle_exploration/preparation_v1/"
    "fixed32_diagnostic_manifest.json"
)
SELECTION = (
    ROOT / "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"
)
RUNS = ROOT / "artifacts/r10_triangle_exploration/fixed32_runs"
TEMPLATE = (
    ROOT
    / "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9/full/evaluator_runs/arcface/winner/request.json"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_fixed32_arcface_requests", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_unmocked_official_envelopes_with_exact_fixed32_bindings() -> None:
    module = _module()
    validation_parent = ROOT / "artifacts/r10_triangle_exploration"
    with tempfile.TemporaryDirectory(
        prefix=".arcface-request-test-", dir=validation_parent
    ) as temporary:
        output = Path(temporary) / "arcface"
        paths = module.build_requests(
            diagnostic_manifest=DIAGNOSTIC,
            selection_manifest=SELECTION,
            runs_root=RUNS,
            template_request=TEMPLATE,
            output_root=output,
            device="cuda:0",
        )
        assert [path.parent.name for path in paths] == list(module.ARM_IDS)
        selection_ids = [
            json.loads(line)["sample_id"]
            for line in SELECTION.read_text(encoding="utf-8").splitlines()
        ]
        for arm_id, path in zip(module.ARM_IDS, paths, strict=True):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            assert envelope["task"] == "arcface"
            assert envelope["payload"]["arm_id"] == arm_id
            assert [row["sample_id"] for row in envelope["payload"]["samples"]] == (
                selection_ids
            )
            assert envelope["evaluator_request_sha256"] == module._canonical_digest(
                envelope, "evaluator_request_sha256"
            )
            source_rows = {
                row["sample_id"]: row
                for row in module._jsonl(
                    RUNS / arm_id / "per_sample.jsonl", f"{arm_id} evidence"
                )
            }
            for sample in envelope["payload"]["samples"]:
                source = source_rows[sample["sample_id"]]
                assert sample["source_sha256"] == _sha(Path(source["source"]))
                assert sample["native_sha256"] == _sha(ROOT / source["native"])
                assert sample["candidate_sha256"] == _sha(ROOT / source["generated"])
        template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        manifest = json.loads(
            (output / "request_build_manifest.json").read_text(encoding="utf-8")
        )
        assert (
            manifest["authoritative_template_request"]["evaluator_request_sha256"]
            == template["evaluator_request_sha256"]
        )
        assert manifest["authoritative_template_request"]["sha256"] == _sha(TEMPLATE)
