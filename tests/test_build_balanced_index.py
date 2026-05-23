from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_balanced_index.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_builder(source: Path, output: Path, samples_per_class: int = 2, seed: int = 1337) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-index",
            str(source),
            "--output-index",
            str(output),
            "--samples-per-class",
            str(samples_per_class),
            "--seed",
            str(seed),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_writes_balanced_index_manifest_and_repeats_byte_identically(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"sample_id": f"s-{label}-{idx}", "label": label, "payload": f"{label}:{idx}"}
        for label in [2, 0, 1]
        for idx in range(4)
    ]
    _write_jsonl(source, rows)

    output_a = tmp_path / "balanced.jsonl"
    output_b = tmp_path / "balanced_again.jsonl"
    result_a = _run_builder(source, output_a)
    result_b = _run_builder(source, output_b)

    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert output_a.read_bytes() == output_b.read_bytes()

    balanced_rows = _read_jsonl(output_a)
    assert len(balanced_rows) == 6
    assert Counter(row["label"] for row in balanced_rows) == {0: 2, 1: 2, 2: 2}
    assert len({row["sample_id"] for row in balanced_rows}) == 6

    manifest = json.loads(output_a.with_name("balanced_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_index"] == str(source)
    assert manifest["output_index"] == str(output_a)
    assert manifest["seed"] == 1337
    assert manifest["samples_per_class"] == 2
    assert manifest["class_counts"] == {"0": 2, "1": 2, "2": 2}
    assert manifest["num_samples"] == 6
    assert len(manifest["source_index_sha256"]) == 64
    assert manifest["label_order"] == [0, 1, 2]
    assert manifest["output_order_rule"] == (
        "labels sorted ascending; each label sampled with random.Random(f'{seed}:{label}') "
        "from source-order rows; combined rows shuffled with random.Random(f'{seed}:output')"
    )


def test_missing_sample_id_fails_fast(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"label": 0}])

    result = _run_builder(source, tmp_path / "out.jsonl", samples_per_class=1)

    assert result.returncode != 0
    assert "sample_id" in result.stderr


def test_non_int_label_fails_fast(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"sample_id": "s-0", "label": "0"}])

    result = _run_builder(source, tmp_path / "out.jsonl", samples_per_class=1)

    assert result.returncode != 0
    assert "label" in result.stderr
    assert "int" in result.stderr


def test_duplicate_sample_id_fails_fast(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {"sample_id": "duplicate", "label": 0},
            {"sample_id": "duplicate", "label": 1},
        ],
    )

    result = _run_builder(source, tmp_path / "out.jsonl", samples_per_class=1)

    assert result.returncode != 0
    assert "duplicate" in result.stderr


def test_insufficient_class_count_fails_with_label_and_count(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {"sample_id": "s-0-0", "label": 0},
            {"sample_id": "s-1-0", "label": 1},
            {"sample_id": "s-1-1", "label": 1},
        ],
    )

    result = _run_builder(source, tmp_path / "out.jsonl", samples_per_class=2)

    assert result.returncode != 0
    assert "label 0" in result.stderr
    assert "1 available" in result.stderr
