from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_r9_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_r9_manifests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_tracked_manifests_reproduce_byte_for_byte() -> None:
    result = builder.verify_or_write(ROOT, write=False)
    assert set(result) == set(builder.EXPECTED_MANIFESTS)
    assert {name: result[name]["file_sha256"] for name in result} == {
        name: contract["file_sha256"]
        for name, contract in builder.EXPECTED_MANIFESTS.items()
    }


def test_builder_writes_once_and_refuses_manifest_replacement(tmp_path: Path) -> None:
    for relative in (builder.CLEAN_INDEX, builder.SOURCE_SNAPSHOT):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    first = builder.verify_or_write(tmp_path, write=True)
    second = builder.verify_or_write(tmp_path, write=True)
    assert first == second
    target = tmp_path / builder.MANIFEST_ROOT / "validate_512.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="refusing to replace immutable manifest"):
        builder.verify_or_write(tmp_path, write=True)
