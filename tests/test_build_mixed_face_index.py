from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from safa.data.index_schema import read_index


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_mixed_face_index.py"


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(17, 29, 43)).save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _record(root: Path, rel_path: str, *, sample_id: str, label: int, split: str) -> dict:
    image_path = root / rel_path
    _make_image(image_path)
    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "label": label,
        "split": split,
        "dataset_root": str(root),
        "dataset_version": "affectnet-unit",
    }


def _run_builder(
    *,
    affectnet_train: Path,
    affectnet_val: Path,
    celebahq_root: Path,
    ffhq_root: Path,
    train_out: Path,
    val_out: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--affectnet-train-index",
            str(affectnet_train),
            "--affectnet-val-index",
            str(affectnet_val),
            "--celebahq-root",
            str(celebahq_root),
            "--ffhq-root",
            str(ffhq_root),
            "--train-out",
            str(train_out),
            "--val-out",
            str(val_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_builds_mixed_train_val_indexes_and_manifest(tmp_path: Path) -> None:
    affectnet_root = tmp_path / "affectnet"
    affectnet_train = tmp_path / "train.jsonl"
    affectnet_val = tmp_path / "val.jsonl"
    _write_jsonl(
        affectnet_train,
        [_record(affectnet_root, "train/a.jpg", sample_id="train:a.jpg", label=3, split="train")],
    )
    _write_jsonl(
        affectnet_val,
        [_record(affectnet_root, "val/b.jpg", sample_id="val:b.jpg", label=4, split="val")],
    )

    celebahq_root = tmp_path / "celebahq"
    _make_image(celebahq_root / "00002.jpg")
    _make_image(celebahq_root / "00001.jpg")
    ffhq_root = tmp_path / "ffhq"
    _make_image(ffhq_root / "FFHQ-1024-2" / "00003.png")
    _make_image(ffhq_root / "FFHQ-1024-1" / "00001.png")
    _make_image(ffhq_root / "FFHQ-1024-1" / "00002.png")

    train_out = tmp_path / "mixed_train.jsonl"
    val_out = tmp_path / "mixed_val.jsonl"
    result = _run_builder(
        affectnet_train=affectnet_train,
        affectnet_val=affectnet_val,
        celebahq_root=celebahq_root,
        ffhq_root=ffhq_root,
        train_out=train_out,
        val_out=val_out,
    )

    assert result.returncode == 0, result.stderr
    train_rows = _read_jsonl(train_out)
    val_rows = _read_jsonl(val_out)
    assert len(train_rows) == 6
    assert len(val_rows) == 1
    assert [row["sample_id"] for row in train_rows] == [
        "train:a.jpg",
        "celebahq_000000",
        "celebahq_000001",
        "ffhq_000000",
        "ffhq_000001",
        "ffhq_000002",
    ]
    assert {row["label"] for row in train_rows if row["sample_id"].startswith(("celebahq_", "ffhq_"))} == {0}
    assert {row["split"] for row in train_rows if row["sample_id"].startswith(("celebahq_", "ffhq_"))} == {"train"}
    assert read_index(train_out)
    assert read_index(val_out)

    manifest = json.loads(train_out.with_name("mixed_train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_train"] == 6
    assert manifest["num_val"] == 1
    assert manifest["train_dataset_counts"] == {"affectnet": 1, "celebahq": 2, "ffhq": 3}
    assert manifest["val_dataset_counts"] == {"affectnet": 1}


def test_cli_fails_when_generic_root_is_missing(tmp_path: Path) -> None:
    affectnet_root = tmp_path / "affectnet"
    affectnet_train = tmp_path / "train.jsonl"
    affectnet_val = tmp_path / "val.jsonl"
    _write_jsonl(
        affectnet_train,
        [_record(affectnet_root, "train/a.jpg", sample_id="train:a.jpg", label=0, split="train")],
    )
    _write_jsonl(
        affectnet_val,
        [_record(affectnet_root, "val/b.jpg", sample_id="val:b.jpg", label=0, split="val")],
    )

    result = _run_builder(
        affectnet_train=affectnet_train,
        affectnet_val=affectnet_val,
        celebahq_root=tmp_path / "missing-celebahq",
        ffhq_root=tmp_path / "missing-ffhq",
        train_out=tmp_path / "mixed_train.jsonl",
        val_out=tmp_path / "mixed_val.jsonl",
    )

    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_cli_fails_on_duplicate_sample_id_across_sources(tmp_path: Path) -> None:
    affectnet_root = tmp_path / "affectnet"
    affectnet_train = tmp_path / "train.jsonl"
    affectnet_val = tmp_path / "val.jsonl"
    _write_jsonl(
        affectnet_train,
        [_record(affectnet_root, "train/a.jpg", sample_id="celebahq_000000", label=1, split="train")],
    )
    _write_jsonl(
        affectnet_val,
        [_record(affectnet_root, "val/b.jpg", sample_id="val:b.jpg", label=1, split="val")],
    )
    celebahq_root = tmp_path / "celebahq"
    ffhq_root = tmp_path / "ffhq"
    _make_image(celebahq_root / "00000.jpg")
    _make_image(ffhq_root / "FFHQ-1024-1" / "00000.png")

    result = _run_builder(
        affectnet_train=affectnet_train,
        affectnet_val=affectnet_val,
        celebahq_root=celebahq_root,
        ffhq_root=ffhq_root,
        train_out=tmp_path / "mixed_train.jsonl",
        val_out=tmp_path / "mixed_val.jsonl",
    )

    assert result.returncode != 0
    assert "duplicate sample_id" in result.stderr
