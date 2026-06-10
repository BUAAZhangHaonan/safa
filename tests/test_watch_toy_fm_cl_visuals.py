from __future__ import annotations

import importlib
import json

import pytest


def test_toy_visual_watcher_refreshes_progress_and_plots(tmp_path) -> None:
    watcher = importlib.import_module("scripts.watch_toy_fm_cl_visuals")
    run_dir = tmp_path / "toy_run"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "actual_fm_delta_after_repr_step": 0.0,
                        "conflict_fraction": 0.0,
                        "delta_deg": 0.0,
                        "dot_after_mean": 0.0,
                        "dot_before_mean": 0.0,
                        "generation_mse_to_x1": 1.0,
                        "lambda_repr": 0.1,
                        "method": "fm_only",
                        "projected_repr_norm_ratio": 0.0,
                        "repr_cosine_mean": 0.1,
                        "repr_point_loss": 0.9,
                        "repr_relation_loss": 0.1,
                        "soft_margin": 0.0,
                        "step": 0.0,
                        "valid_fm_loss": 2.0,
                    }
                ),
                json.dumps(
                    {
                        "actual_fm_delta_after_repr_step": 0.0,
                        "conflict_fraction": 0.2,
                        "delta_deg": 0.0,
                        "dot_after_mean": 0.1,
                        "dot_before_mean": 0.1,
                        "generation_mse_to_x1": 0.5,
                        "lambda_repr": 0.1,
                        "method": "fm_only",
                        "projected_repr_norm_ratio": 1.0,
                        "repr_cosine_mean": 0.9,
                        "repr_point_loss": 0.1,
                        "repr_relation_loss": 0.01,
                        "soft_margin": 0.0,
                        "step": 10.0,
                        "valid_fm_loss": 0.2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    progress = watcher.refresh_visuals(run_dir, expected_experiments=2, steps_per_experiment=10)

    assert progress["metric_rows"] == 2
    assert progress["completed_experiments_estimate"] == 1
    assert progress["expected_experiments"] == 2
    assert (run_dir / "live_curves.png").is_file()
    assert (run_dir / "live_tradeoff.png").is_file()
    assert (run_dir / "progress.json").is_file()


def test_watch_visuals_waits_for_empty_metrics_file(tmp_path, capsys, monkeypatch) -> None:
    watch = importlib.import_module("scripts.watch_toy_fm_cl_visuals")
    (tmp_path / "metrics.jsonl").write_text("", encoding="utf-8")

    class StopAfterFirstSleep(Exception):
        pass

    def stop_after_first_sleep(_seconds: int) -> None:
        raise StopAfterFirstSleep

    monkeypatch.setattr(watch.time, "sleep", stop_after_first_sleep)
    with pytest.raises(StopAfterFirstSleep):
        watch.watch_visuals(
            run_dir=tmp_path,
            expected_experiments=1,
            steps_per_experiment=10,
            interval_seconds=1,
            stop_when_summary_exists=False,
        )

    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["status"] == "waiting_for_metrics"
    assert "Toy metrics file is empty" in payload["message"]
