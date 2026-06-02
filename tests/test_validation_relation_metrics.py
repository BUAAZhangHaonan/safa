from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from safa.evaluation.metrics import compute_validation_relation_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def _offdiag_values(gram):
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return gram[mask]


def _pearson(x, y) -> float:
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    return float((centered_x * centered_y).sum() / (centered_x.norm() * centered_y.norm()))


def _average_ranks(values):
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    start = 0
    while start < int(values.numel()):
        end = start + 1
        while end < int(values.numel()) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def test_validation_relation_metrics_match_exact_small_embeddings() -> None:
    root_half = math.sqrt(0.5)
    root_third = math.sqrt(1.0 / 3.0)
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [root_half, 0.0, root_half],
            [root_third, root_third, root_third],
        ],
        dtype=torch.float64,
    )
    pred = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [root_half, root_half, 0.0],
        ],
        dtype=torch.float64,
    )

    metrics = compute_validation_relation_metrics(pred, target)

    pred_gram = pred @ pred.T
    target_gram = target @ target.T
    pred_pairs = _offdiag_values(pred_gram)
    target_pairs = _offdiag_values(target_gram)
    diff = pred_pairs - target_pairs
    expected_point = (1.0 - (pred * target).sum(dim=1)).mean()
    expected_mse = diff.pow(2).mean()
    expected_mae = diff.abs().mean()
    expected_pearson = _pearson(pred_pairs, target_pairs)
    expected_spearman = _pearson(_average_ranks(pred_pairs), _average_ranks(target_pairs))

    assert metrics["repr_point_loss"] == pytest.approx(float(expected_point), abs=1e-12)
    assert metrics["repr_relation_loss"] == pytest.approx(float(expected_mse), abs=1e-12)
    assert metrics["offdiag_gram_mse"] == pytest.approx(float(expected_mse), abs=1e-12)
    assert metrics["offdiag_gram_mae"] == pytest.approx(float(expected_mae), abs=1e-12)
    assert metrics["pairwise_pearson"] == pytest.approx(expected_pearson, abs=1e-12)
    assert metrics["pairwise_spearman"] == pytest.approx(expected_spearman, abs=1e-12)


def test_validation_relation_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_validation_relation_metrics(torch.eye(3), torch.eye(4))
    with pytest.raises(ValueError, match="B > 1"):
        compute_validation_relation_metrics(torch.ones(1, 3), torch.ones(1, 3))
    with pytest.raises(FloatingPointError, match="finite"):
        compute_validation_relation_metrics(torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]), torch.eye(2))
    with pytest.raises(ValueError, match="unit-norm"):
        compute_validation_relation_metrics(torch.tensor([[2.0, 0.0], [0.0, 1.0]]), torch.eye(2))
    with pytest.raises(ValueError, match="pairwise_pearson.*zero variance"):
        compute_validation_relation_metrics(torch.eye(3), torch.eye(3))


def _load_relation_watcher():
    path = REPO_ROOT / "scripts" / "watch_medium_v2_validation_relation_metrics.py"
    spec = importlib.util.spec_from_file_location("watch_medium_v2_validation_relation_metrics", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_relation_watcher_cli_parse_and_paths(tmp_path: Path) -> None:
    module = _load_relation_watcher()
    checkpoint_dir = tmp_path / "checkpoints"
    out_dir = tmp_path / "relation"
    summary = tmp_path / "summary.json"

    args = module.parse_args(
        [
            "--once",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--config",
            "config.yaml",
            "--index",
            "val.jsonl",
            "--features",
            "features",
            "--output-dir",
            str(out_dir),
            "--summary",
            str(summary),
            "--run-name",
            "reltest",
            "--model",
            "both",
            "--device",
            "cuda:0",
            "--stable-seconds",
            "0",
            "--load-retries",
            "2",
        ]
    )
    paths = module.resolve_paths(args)

    assert args.once is True
    assert args.model == "both"
    assert args.device == "cuda:0"
    assert args.stable_seconds == 0.0
    assert args.load_retries == 2
    assert paths.checkpoints == (checkpoint_dir / "last.pt",)
    assert paths.summary == summary
    assert paths.events == out_dir / "reltest_events.jsonl"
    assert paths.state == out_dir / "reltest_state.json"
    assert paths.log == out_dir / "reltest.log"


def test_validation_relation_watcher_cli_requires_bounded_loop_max_samples() -> None:
    module = _load_relation_watcher()

    ok = module.parse_args(["--max-samples", "512"])
    module.validate_cli_args(ok)

    missing = module.parse_args([])
    with pytest.raises(ValueError, match=r"--max-samples.*loop"):
        module.validate_cli_args(missing)

    zero = module.parse_args(["--max-samples", "0"])
    with pytest.raises(ValueError, match="positive"):
        module.validate_cli_args(zero)

    too_large = module.parse_args(["--max-samples", "2049"])
    with pytest.raises(ValueError, match="dense Gram cap.*2048"):
        module.validate_cli_args(too_large)

    high_cap = module.parse_args(["--max-samples", "4096", "--dense-gram-cap", "4096"])
    module.validate_cli_args(high_cap)


def _watcher_paths(module, tmp_path: Path):
    checkpoint = tmp_path / "last.pt"
    out_dir = tmp_path / "relation"
    return checkpoint, module.WatcherPaths(
        checkpoints=(checkpoint,),
        config=tmp_path / "config.yaml",
        index=tmp_path / "val.jsonl",
        features=tmp_path / "features",
        out_dir=out_dir,
        summary=out_dir / "summary.json",
        events=out_dir / "events.jsonl",
        state=out_dir / "state.json",
        log=out_dir / "watch.log",
    )


def test_validation_relation_watcher_state_skips_unchanged_checkpoint(tmp_path: Path, monkeypatch) -> None:
    module = _load_relation_watcher()
    checkpoint, paths = _watcher_paths(module, tmp_path)
    checkpoint.write_bytes(b"same-checkpoint")
    stat = module.checkpoint_stat(checkpoint)
    record = module.checkpoint_state_record(checkpoint, stat, model_source="raw", stage_epoch_1based=3)
    key_payload = json.loads(record["key"])
    assert key_payload["checkpoint"] == str(checkpoint)
    assert key_payload["model_source"] == "raw"
    assert key_payload["stage_epoch_1based"] == 3
    module.write_json_atomic(paths.state, {"processed_checkpoints": [record]})

    calls = []

    def fake_compute(**kwargs):
        calls.append(kwargs["checkpoint_path"])
        raise AssertionError("unchanged checkpoint should be skipped")

    monkeypatch.setattr(module, "compute_checkpoint_relation_metrics", fake_compute)

    assert module.run_once(
        paths,
        model="raw",
        device="cpu",
        max_samples=512,
        batch_size=1,
        num_workers=0,
        stable_seconds=0.0,
        stable_check_interval=0.01,
        load_retries=1,
        load_retry_interval=0.0,
        dense_gram_cap=2048,
        skip_unchanged=True,
    ) == 0
    assert calls == []


def test_validation_relation_watcher_state_processes_changed_last_checkpoint(tmp_path: Path, monkeypatch) -> None:
    module = _load_relation_watcher()
    checkpoint, paths = _watcher_paths(module, tmp_path)
    checkpoint.write_bytes(b"old")
    old_record = module.checkpoint_state_record(
        checkpoint,
        module.checkpoint_stat(checkpoint),
        model_source="raw",
        stage_epoch_1based=3,
    )
    module.write_json_atomic(paths.state, {"processed_checkpoints": [old_record]})
    checkpoint.write_bytes(b"new-last-checkpoint")

    calls = []

    def fake_compute(**kwargs):
        calls.append(kwargs["checkpoint_path"])
        return {
            "checkpoint": str(kwargs["checkpoint_path"]),
            "checkpoint_stat": module.checkpoint_stat(kwargs["checkpoint_path"]),
            "stage": "stage2",
            "stage_epoch_1based": 4,
            "model": "raw",
            "variants": {"raw": {"sample_count": 512.0}},
        }

    monkeypatch.setattr(module, "compute_checkpoint_relation_metrics", fake_compute)

    assert module.run_once(
        paths,
        model="raw",
        device="cpu",
        max_samples=512,
        batch_size=1,
        num_workers=0,
        stable_seconds=0.0,
        stable_check_interval=0.01,
        load_retries=1,
        load_retry_interval=0.0,
        dense_gram_cap=2048,
        skip_unchanged=True,
    ) == 0
    assert calls == [checkpoint]
    state = module.read_state(paths.state)
    processed = state["processed_checkpoints"]
    assert any(json.loads(item["key"])["stage_epoch_1based"] == 4 for item in processed)
