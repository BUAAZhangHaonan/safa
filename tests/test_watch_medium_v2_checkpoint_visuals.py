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
    output_dir.mkdir()
    completed = {}
    for epoch in (1, 2):
        out_path, manifest_path = module.output_paths(output_dir, epoch)
        out_path.write_bytes(b"png")
        module.write_manifest(
            manifest_path,
            module.build_manifest(
                epoch=epoch,
                paths=paths,
                samples=[],
                out_path=out_path,
                device="cuda:0",
                sampling_seed=1337,
                metrics=[],
                note="pre-existing",
                checkpoint_epoch_1based=epoch,
                visual_epoch_1based=epoch,
                backfilled_from_latest_checkpoint=False,
            ),
        )
        completed[f"{epoch:04d}"] = str(out_path)
    module.write_json_atomic(paths.state, {"completed": completed})

    calls = []

    def fake_generate(
        *,
        epoch,
        paths,
        out_path,
        manifest_path,
        num_samples,
        sample_seed,
        device,
        sampling_seed,
        checkpoint_epoch_1based,
        visual_epoch_1based,
        backfilled_from_latest_checkpoint,
    ):
        calls.append(
            (
                epoch,
                out_path,
                manifest_path,
                num_samples,
                sample_seed,
                device,
                sampling_seed,
                checkpoint_epoch_1based,
                visual_epoch_1based,
                backfilled_from_latest_checkpoint,
            )
        )
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
                checkpoint_epoch_1based=checkpoint_epoch_1based,
                visual_epoch_1based=visual_epoch_1based,
                backfilled_from_latest_checkpoint=backfilled_from_latest_checkpoint,
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
            3,
            3,
            False,
        )
    ]
    manifest = json.loads((output_dir / "epoch_0003_checkpoint_pairs_manifest.json").read_text(encoding="utf-8"))
    assert manifest["epoch"] == 3
    assert manifest["visual_epoch_1based"] == 3
    assert manifest["checkpoint_epoch_1based"] == 3
    assert manifest["backfilled_from_latest_checkpoint"] is False
    assert manifest["inputs"]["index"] == str(index)
    assert manifest["inputs"]["features"] == str(features)
    assert manifest["samples"][0]["sample_id"] == "sample-a"
    events = [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["checkpoint_visual_created"]


def test_run_once_backfills_missing_epochs_between_state_and_metrics(tmp_path: Path) -> None:
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
    metrics.write_text(json.dumps({"stage_epoch_1based": 12}), encoding="utf-8")
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
    output_dir.mkdir()
    for epoch in (1, 2):
        out_path, manifest_path = module.output_paths(output_dir, epoch)
        out_path.write_bytes(b"png")
        module.write_manifest(
            manifest_path,
            module.build_manifest(
                epoch=epoch,
                paths=paths,
                samples=[],
                out_path=out_path,
                device="cuda:0",
                sampling_seed=1337,
                metrics=[],
                note="pre-existing",
                checkpoint_epoch_1based=epoch,
                visual_epoch_1based=epoch,
                backfilled_from_latest_checkpoint=False,
            ),
        )
    module.write_json_atomic(
        paths.state,
        {
            "completed": {
                "0001": str(output_dir / "epoch_0001_checkpoint_pairs.png"),
                "0002": str(output_dir / "epoch_0002_checkpoint_pairs.png"),
            }
        },
    )

    calls = []

    def fake_generate(
        *,
        epoch,
        paths,
        out_path,
        manifest_path,
        num_samples,
        sample_seed,
        device,
        sampling_seed,
        checkpoint_epoch_1based,
        visual_epoch_1based,
        backfilled_from_latest_checkpoint,
    ):
        calls.append((epoch, checkpoint_epoch_1based, visual_epoch_1based, backfilled_from_latest_checkpoint))
        out_path.write_bytes(b"png")
        module.write_manifest(
            manifest_path,
            module.build_manifest(
                epoch=epoch,
                paths=paths,
                samples=[],
                out_path=out_path,
                device=device,
                sampling_seed=sampling_seed,
                metrics=[],
                note="fake generation for unit test",
                checkpoint_epoch_1based=checkpoint_epoch_1based,
                visual_epoch_1based=visual_epoch_1based,
                backfilled_from_latest_checkpoint=backfilled_from_latest_checkpoint,
            ),
        )

    assert module.run_once(paths, num_samples=1, generate_func=fake_generate) == 0

    assert [call[0] for call in calls] == list(range(3, 13))
    assert calls[0] == (3, 12, 3, True)
    assert calls[-1] == (12, 12, 12, False)
    manifest = json.loads((output_dir / "epoch_0003_checkpoint_pairs_manifest.json").read_text(encoding="utf-8"))
    assert manifest["visual_epoch_1based"] == 3
    assert manifest["checkpoint_epoch_1based"] == 12
    assert manifest["backfilled_from_latest_checkpoint"] is True


def test_run_once_can_limit_backfill_to_latest_and_every_n(tmp_path: Path) -> None:
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
    metrics.write_text(json.dumps({"stage_epoch_1based": 45}), encoding="utf-8")
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

    def fake_generate(
        *,
        epoch,
        paths,
        out_path,
        manifest_path,
        num_samples,
        sample_seed,
        device,
        sampling_seed,
        checkpoint_epoch_1based,
        visual_epoch_1based,
        backfilled_from_latest_checkpoint,
    ):
        calls.append((epoch, checkpoint_epoch_1based, visual_epoch_1based, backfilled_from_latest_checkpoint))
        out_path.write_bytes(b"png")
        module.write_manifest(
            manifest_path,
            module.build_manifest(
                epoch=epoch,
                paths=paths,
                samples=[],
                out_path=out_path,
                device=device,
                sampling_seed=sampling_seed,
                metrics=[],
                note="fake generation for unit test",
                checkpoint_epoch_1based=checkpoint_epoch_1based,
                visual_epoch_1based=visual_epoch_1based,
                backfilled_from_latest_checkpoint=backfilled_from_latest_checkpoint,
            ),
        )

    assert module.run_once(paths, num_samples=1, backfill_every=20, generate_func=fake_generate) == 0

    assert calls == [
        (20, 45, 20, True),
        (40, 45, 40, True),
        (45, 45, 45, False),
    ]


def test_resolve_paths_uses_checkpoint_dir_output_dir_and_run_name(tmp_path: Path) -> None:
    module = _load_script()
    checkpoint_dir = tmp_path / "ckpt"
    output_dir = tmp_path / "plots"

    args = module.parse_args(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--config",
            str(tmp_path / "config.yaml"),
            "--index",
            str(tmp_path / "index.jsonl"),
            "--features",
            str(tmp_path / "features"),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "stage1_long1000_checkpoint_visuals",
        ]
    )
    paths = module.resolve_paths(args)

    assert paths.metrics == checkpoint_dir / "last_metrics.json"
    assert paths.checkpoint == checkpoint_dir / "last.pt"
    assert paths.out_dir == output_dir
    assert paths.events == output_dir / "stage1_long1000_checkpoint_visuals_events.jsonl"
    assert paths.log == output_dir / "stage1_long1000_checkpoint_visuals.log"
    assert paths.state == output_dir / "stage1_long1000_checkpoint_visuals_state.json"


def test_completed_epoch_numbers_caps_checkpoint_history_to_metrics_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    metrics = checkpoint_dir / "last_metrics.json"
    checkpoint = checkpoint_dir / "last.pt"
    metrics.write_text(json.dumps({"stage_epoch_1based": 12}), encoding="utf-8")
    checkpoint.write_bytes(b"fake")
    paths = module.WatcherPaths(metrics=metrics, checkpoint=checkpoint)

    monkeypatch.setattr(module, "read_checkpoint_history_epochs", lambda _path: {1, 2, 12, 53, 182})

    assert module.completed_epoch_numbers(paths) == list(range(1, 13))


def test_gpu_guard_rejects_disallowed_visible_devices_and_busy_gpu0() -> None:
    module = _load_script()

    with pytest.raises(RuntimeError, match="GPU1"):
        module.validate_cuda_visible_devices("1")
    with pytest.raises(RuntimeError, match="GPU1"):
        module.validate_cuda_visible_devices("0,1")
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
    assert "--run-name checkpoint_visuals" in joined
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
