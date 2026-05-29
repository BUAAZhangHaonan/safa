from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "supervise_medium_v2_stages.py"
    spec = importlib.util.spec_from_file_location("supervise_medium_v2_stages", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(module, tmp_path: Path):
    return module.StageSupervisorPaths(
        status=tmp_path / "status.json",
        events=tmp_path / "events.jsonl",
        log=tmp_path / "supervisor.log",
        report_dir=tmp_path / "monitor",
        draft_dir=tmp_path / "drafts",
        plot_dir=tmp_path / "plots",
        marker=tmp_path / "medium_v2_start_m3.approved",
        m2_metrics=tmp_path / "m2" / "last_metrics.json",
        m2_history=tmp_path / "m2" / "history.json",
        m2_quality_dir=tmp_path / "m2_quality",
        m2_train_log=tmp_path / "logs" / "m2.log",
        m2_last_checkpoint=tmp_path / "m2" / "last.pt",
        m2_best_checkpoint=tmp_path / "m2" / "best_raw_utility.pt",
        m3_metrics=tmp_path / "m3" / "last_metrics.json",
        m3_history=tmp_path / "m3" / "history.json",
        m3_quality_dir=tmp_path / "m3_quality",
        m3_train_log=tmp_path / "logs" / "m3.log",
        m3_last_checkpoint=tmp_path / "m3" / "last.pt",
        m3_best_checkpoint=tmp_path / "m3" / "best_raw_utility.pt",
        privacy_skip_json=tmp_path / "monitor" / "privacy_skipped.json",
    )


def _complete_m2(paths, *, epoch: int = 120) -> None:
    paths.m2_metrics.parent.mkdir(parents=True)
    paths.m2_metrics.write_text(
        json.dumps(
            {
                "stage_epoch_1based": epoch,
                "loss": 0.4,
                "validation_raw_latent_cosine_mean": 0.91,
                "validation_raw_single_face_eq1_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    paths.m2_last_checkpoint.write_text("last", encoding="utf-8")
    paths.m2_best_checkpoint.write_text("best", encoding="utf-8")


def _runner(module, calls: list[tuple[str, ...]]):
    def fake_runner(command):
        calls.append(tuple(command))
        if command[:3] == ("tmux", "has-session", "-t"):
            return module.CommandResult(1, "", "")
        if command[0] == "pgrep":
            return module.CommandResult(1, "", "")
        if command[0] == "nvidia-smi":
            return module.CommandResult(0, "3, 0, 0, 24576\n4, 0, 0, 24576\n5, 0, 0, 24576\n6, 0, 0, 24576\n", "")
        if command[:3] == ("tmux", "new-session", "-d"):
            return module.CommandResult(0, "", "")
        return module.CommandResult(0, "", "")

    return fake_runner


def test_missing_marker_does_not_start_m3_when_m2_is_ready(tmp_path: Path) -> None:
    module = _load_script()
    paths = _paths(module, tmp_path)
    _complete_m2(paths)
    (paths.plot_dir / "m2_curves.png").parent.mkdir(parents=True)
    (paths.plot_dir / "m2_curves.png").write_text("png", encoding="utf-8")
    paths.draft_dir.mkdir(parents=True)
    (paths.draft_dir / "MEDIUM_V2_STAGE2_M2_GRAM_WEIGHTED_DRAFT.md").write_text("draft", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    exit_code = module.run_once(paths, module.StageSupervisorConfig(run_completion_actions=False), _runner(module, calls))

    assert exit_code == 0
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["m3_gate"]["marker_present"] is False
    assert status["m3_gate"]["ready"] is True
    assert status["state"] == "waiting_m3_approval"
    assert not any(call[:3] == ("tmux", "new-session", "-d") for call in calls)


def test_marker_and_ready_m2_constructs_m3_launch_command(tmp_path: Path) -> None:
    module = _load_script()
    paths = _paths(module, tmp_path)
    _complete_m2(paths)
    paths.marker.parent.mkdir(parents=True, exist_ok=True)
    paths.marker.write_text("approved\n", encoding="utf-8")
    (paths.plot_dir / "m2_curves.png").parent.mkdir(parents=True)
    (paths.plot_dir / "m2_curves.png").write_text("png", encoding="utf-8")
    paths.draft_dir.mkdir(parents=True)
    (paths.draft_dir / "MEDIUM_V2_STAGE2_M2_GRAM_WEIGHTED_DRAFT.md").write_text("draft", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    exit_code = module.run_once(paths, module.StageSupervisorConfig(run_completion_actions=False), _runner(module, calls))

    assert exit_code == 0
    launch = next(call for call in calls if call[:3] == ("tmux", "new-session", "-d"))
    assert launch[4] == "train_g_medium_v2_stage2_m3_gram_projected_gpu3_6"
    command_text = launch[5]
    assert "CUDA_VISIBLE_DEVICES=3,4,5,6" in command_text
    assert "configs/medium_v2/train_g_medium_v2_stage2_m3_gram_projected.yaml" in command_text
    assert "torch.distributed.run --standalone --nproc_per_node=4" in command_text
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["state"] == "m3_started"
