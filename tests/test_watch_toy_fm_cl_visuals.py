from __future__ import annotations

import importlib
import json

import pytest


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
