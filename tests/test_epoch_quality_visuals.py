from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visualize_epoch_quality_pairs_writes_grid_from_per_sample_metadata(tmp_path: Path) -> None:
    module = _load_script("scripts/visualize_epoch_quality_pairs.py", "visualize_epoch_quality_pairs")
    source_a = tmp_path / "source_a.png"
    source_b = tmp_path / "source_b.png"
    generated_a = tmp_path / "generated_a.png"
    generated_b = tmp_path / "generated_b.png"
    _write_png(source_a, (220, 30, 30))
    _write_png(source_b, (30, 220, 30))
    _write_png(generated_a, (30, 30, 220))
    _write_png(generated_b, (220, 220, 30))
    index_path = tmp_path / "index.jsonl"
    _write_jsonl(
        index_path,
        [
            {"sample_id": "sample-a", "image_path": str(source_a), "label": 1},
            {"sample_id": "sample-b", "image_path": str(source_b), "label": 2},
        ],
    )
    epoch_dir = tmp_path / "quality" / "epoch_0001"
    epoch_dir.mkdir(parents=True)
    _write_jsonl(
        epoch_dir / "per_sample.jsonl",
        [
            {"sample_id": "sample-a", "artifacts": {"generated_image_path": str(generated_a)}},
            {"sample_id": "sample-b", "artifacts": {"generated_image_path": str(generated_b)}},
        ],
    )
    out_path = tmp_path / "pairs.png"

    result = module.visualize_epoch_quality_pairs(
        quality_epoch_dir=epoch_dir,
        index=index_path,
        out_path=out_path,
        num_samples=2,
    )

    assert result == out_path
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    with Image.open(out_path) as image:
        assert image.size[0] >= 640
        assert image.size[1] >= 320


def test_visualize_epoch_quality_pairs_fails_fast_without_metadata(tmp_path: Path) -> None:
    module = _load_script("scripts/visualize_epoch_quality_pairs.py", "visualize_epoch_quality_pairs")
    index_path = tmp_path / "index.jsonl"
    source = tmp_path / "source.png"
    generated_dir = tmp_path / "quality" / "epoch_0001" / "generated_images"
    generated = generated_dir / "00000000__sample-a.png"
    _write_png(source, (220, 30, 30))
    generated_dir.mkdir(parents=True)
    _write_png(generated, (30, 30, 220))
    _write_jsonl(index_path, [{"sample_id": "sample-a", "image_path": str(source), "label": 1}])

    with pytest.raises(ValueError, match="metadata"):
        module.visualize_epoch_quality_pairs(
            quality_epoch_dir=generated_dir.parent,
            index=index_path,
            out_path=tmp_path / "pairs.png",
            num_samples=1,
        )


def test_watch_medium_v2_epoch_visuals_processes_new_epoch(tmp_path: Path) -> None:
    module = _load_script("scripts/watch_medium_v2_epoch_visuals.py", "watch_medium_v2_epoch_visuals")
    source = tmp_path / "source.png"
    generated = tmp_path / "generated.png"
    _write_png(source, (220, 30, 30))
    _write_png(generated, (30, 30, 220))
    index_path = tmp_path / "index.jsonl"
    _write_jsonl(index_path, [{"sample_id": "sample-a", "image_path": str(source), "label": 1}])
    epoch_dir = tmp_path / "quality" / "epoch_0001"
    epoch_dir.mkdir(parents=True)
    _write_jsonl(epoch_dir / "per_sample.jsonl", [{"sample_id": "sample-a", "generated_image_path": str(generated)}])
    paths = module.WatcherPaths(
        quality_dir=tmp_path / "quality",
        index=index_path,
        out_dir=tmp_path / "plots",
        events=tmp_path / "events.jsonl",
        log=tmp_path / "events.log",
        state=tmp_path / "state.json",
    )

    exit_code = module.run_once(paths, num_samples=1)

    assert exit_code == 0
    out_path = tmp_path / "plots" / "epoch_0001_pairs.png"
    assert out_path.is_file()
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["epoch_visual_created"]
    assert events[0]["epoch"] == 1
    assert "epoch_visual_created" in paths.log.read_text(encoding="utf-8")


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
