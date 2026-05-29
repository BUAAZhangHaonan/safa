from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "watch_medium_v2_checkpoint_visuals.py"
    spec = importlib.util.spec_from_file_location("watch_medium_v2_checkpoint_visuals", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_stage_epoch_requires_one_based_epoch(tmp_path: Path) -> None:
    module = _load_script()
    metrics = tmp_path / "last_metrics.json"
    metrics.write_text(json.dumps({"stage_epoch_1based": 7}), encoding="utf-8")

    assert module.read_stage_epoch(metrics) == 7

    metrics.write_text(json.dumps({"stage_epoch_0based": 6}), encoding="utf-8")
    with pytest.raises(ValueError, match="stage_epoch_1based"):
        module.read_stage_epoch(metrics)


def test_select_samples_from_index_first_n_and_seeded(tmp_path: Path) -> None:
    module = _load_script()
    index = tmp_path / "val.jsonl"
    _write_jsonl(
        index,
        [
            {"sample_id": "sample-a", "image_path": "/a.jpg", "label": 1},
            {"sample_id": "sample-b", "image_path": "/b.jpg", "label": 2},
            {"sample_id": "sample-c", "image_path": "/c.jpg", "label": 3},
        ],
    )

    first = module.select_samples(index, num_samples=2, sample_seed=None)
    seeded_a = module.select_samples(index, num_samples=2, sample_seed=11)
    seeded_b = module.select_samples(index, num_samples=2, sample_seed=11)

    assert [row["sample_id"] for row in first] == ["sample-a", "sample-b"]
    assert seeded_a == seeded_b
    assert {row["sample_id"] for row in seeded_a}.issubset({"sample-a", "sample-b", "sample-c"})


def test_run_once_detects_new_epoch_and_writes_manifest(tmp_path: Path) -> None:
    module = _load_script()
    index = tmp_path / "val.jsonl"
    features = tmp_path / "features"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "plots"
    config = tmp_path / "config.yaml"
    metrics = checkpoint_dir / "last_metrics.json"
    checkpoint = checkpoint_dir / "last.pt"
    _write_jsonl(index, [{"sample_id": "sample-a", "image_path": "/a.jpg", "label": 1}])
    features.mkdir()
    (features / "features.pt").write_bytes(b"fake")
    (features / "manifest.json").write_text("{}", encoding="utf-8")
    checkpoint_dir.mkdir()
    checkpoint.write_bytes(b"fake")
    config.write_text("sampling_seed: 1337\n", encoding="utf-8")
    metrics.write_text(json.dumps({"stage_epoch_1based": 3}), encoding="utf-8")
    paths = module.WatcherPaths(
        metrics=metrics,
        checkpoint=checkpoint,
        config=config,
        index=index,
        features=features,
        out_dir=output_dir,
        events=output_dir / "events.jsonl",
        log=output_dir / "events.log",
        state=output_dir / "state.json",
    )

    calls = []

    def fake_generate(*, epoch, paths, out_path, manifest_path, num_samples, sample_seed, device, sampling_seed):
        calls.append((epoch, out_path, manifest_path, num_samples, sample_seed, device, sampling_seed))
        out_path.write_bytes(b"png")
        module.write_manifest(
            manifest_path,
            module.build_manifest(
                epoch=epoch,
                paths=paths,
                samples=module.select_samples(paths.index, num_samples=num_samples, sample_seed=sample_seed),
                out_path=out_path,
                device=device,
                sampling_seed=sampling_seed,
                metrics=[{"sample_id": "sample-a", "label": 1}],
                note="fake generation for unit test",
            ),
        )

    assert module.run_once(paths, num_samples=1, generate_func=fake_generate) == 0
    assert calls == [
        (
            3,
            output_dir / "epoch_0003_checkpoint_pairs.png",
            output_dir / "epoch_0003_checkpoint_pairs_manifest.json",
            1,
            None,
            "cuda:0",
            1337,
        )
    ]
    manifest = json.loads((output_dir / "epoch_0003_checkpoint_pairs_manifest.json").read_text(encoding="utf-8"))
    assert manifest["epoch"] == 3
    assert manifest["inputs"]["index"] == str(index)
    assert manifest["inputs"]["features"] == str(features)
    assert manifest["samples"][0]["sample_id"] == "sample-a"
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["checkpoint_visual_created"]


def test_gpu_guard_rejects_disallowed_visible_devices_and_busy_gpu0() -> None:
    module = _load_script()

    with pytest.raises(RuntimeError, match="GPU3-6"):
        module.validate_cuda_visible_devices("3")
    with pytest.raises(RuntimeError, match="GPU3-6"):
        module.validate_cuda_visible_devices("0,3")

    busy = "0, 99, 2048, 24576\n1, 0, 0, 24576\n"
    with pytest.raises(RuntimeError, match="GPU0 is busy"):
        module.guard_gpu0_available(busy, max_memory_mb=1024, max_util_pct=50)

    module.guard_gpu0_available("0, 1, 512, 24576\n", max_memory_mb=1024, max_util_pct=50)


def test_build_tmux_command_pins_gpu0_and_never_mentions_gpu3_6() -> None:
    module = _load_script()

    command = module.build_tmux_command(
        session_name="watch_medium_v2_m2_checkpoint_visuals",
        python_exe="/env/bin/python",
        script="scripts/watch_medium_v2_checkpoint_visuals.py",
        interval=60,
        device="cuda:0",
        cuda_visible_devices="0",
    )
    joined = " ".join(command)

    assert command[:4] == ["tmux", "new-session", "-d", "-s"]
    assert "CUDA_VISIBLE_DEVICES=0" in joined
    assert "--device cuda:0" in joined
    for forbidden in ("CUDA_VISIBLE_DEVICES=3", "CUDA_VISIBLE_DEVICES=4", "CUDA_VISIBLE_DEVICES=5", "CUDA_VISIBLE_DEVICES=6"):
        assert forbidden not in joined


def test_draw_checkpoint_pair_grid_writes_image(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source.png"
    generated = tmp_path / "generated.png"
    _write_png(source, (220, 30, 30))
    _write_png(generated, (30, 30, 220))
    out_path = tmp_path / "pairs.png"

    result = module.draw_checkpoint_pair_grid(
        [
            {
                "sample_id": "sample-a",
                "label": 1,
                "source_path": source,
                "generated_path": generated,
                "latent_cosine": None,
                "source_pred": None,
                "generated_pred": None,
            }
        ],
        out_path,
        epoch=4,
        checkpoint_path=tmp_path / "last.pt",
    )

    assert result == out_path
    assert out_path.is_file()
    with Image.open(out_path) as image:
        assert image.size[0] >= 300
        assert image.size[1] >= 280


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path)
