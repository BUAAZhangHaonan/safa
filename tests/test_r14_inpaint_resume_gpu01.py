from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from safa.training.g_loop import _validate_train_g_config


REPO = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_script("validate_r14_inpaint_resume_gpu01.py")


def final_metrics(*, step: int = 2688, optimizer_resumed: bool = True) -> dict:
    return {
        "global_step": step,
        "required_optimizer_steps": 2688,
        "stage": "stage2",
        "stage_epoch_0based": 19,
        "stage_epoch_1based": 20,
        "world_size": 2,
        "global_batch_size": 4,
        "per_device_batch_size": 2,
        "optimizer_resumed": optimizer_resumed,
    }


def source_metrics(*, step: int = 2432) -> dict:
    return {
        "global_step": step,
        "required_optimizer_steps": 2560,
        "stage": "stage2",
        "stage_epoch_0based": 18,
        "stage_epoch_1based": 19,
        "world_size": 4,
        "global_batch_size": 8,
        "per_device_batch_size": 2,
        "optimizer_resumed": False,
    }


def source_checkpoint_payload(*, optimizer_state: bool = True) -> dict:
    history = [{"global_step": 128 * (index + 1)} for index in range(19)]
    return {
        "metrics": source_metrics(),
        "history": history,
        "training_config": {
            "world_size": 4,
            "global_batch_size": 8,
            "per_device_batch_size": 2,
            "stages": {"stage2": {"epochs": 20}},
        },
        "model_state_dict": {"vector_field.context_embedder.weight": object()},
        "ema_model_state_dict": {"vector_field.context_embedder.weight": object()},
        "optimizer_state_dict": {
            "state": {0: {}} if optimizer_state else {},
            "param_groups": [{"params": [0]}],
        },
    }


def final_checkpoint_payload(*, step: int = 2688, optimizer_state: bool = True) -> dict:
    history = [{"global_step": 128 * (index + 1)} for index in range(19)]
    history.append(final_metrics(step=step))
    return {
        "metrics": final_metrics(step=step),
        "history": history,
        "training_config": {
            "r14_contract": "safa_r14_face_region_inpaint_feasibility_v1",
            "r14_resume_contract": validator.RESUME_CONTRACT,
            "world_size": 2,
            "global_batch_size": 4,
            "per_device_batch_size": 2,
        },
        "model_state_dict": {"weight": object()},
        "ema_model_state_dict": {"weight": object()},
        "optimizer_state_dict": {
            "state": {0: {}} if optimizer_state else {},
            "param_groups": [{"params": [0]}],
        },
    }


def step_checkpoint_payload(*, optimizer_state: bool = True) -> dict:
    history = [{"global_step": 128 * (index + 1)} for index in range(19)]
    return {
        "metrics": {
            "global_step": 2688,
            "required_optimizer_steps": 2688,
            "stage": "stage2",
            "stage_epoch_0based": 19,
            "stage_epoch_1based": 20,
            "world_size": 2,
            "global_batch_size": 4,
            "per_device_batch_size": 2,
            "checkpoint_kind": "optimizer_step",
        },
        "history": history,
        "training_config": {
            "r14_resume_contract": validator.RESUME_CONTRACT,
            "world_size": 2,
            "global_batch_size": 4,
            "per_device_batch_size": 2,
        },
        "model_state_dict": {"weight": object()},
        "ema_model_state_dict": {"weight": object()},
        "optimizer_state_dict": {
            "state": {0: {}} if optimizer_state else {},
            "param_groups": [{"params": [0]}],
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_resume_config_is_exact_and_valid() -> None:
    path = REPO / validator.CONFIG
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validator._validate_config(path)
    _validate_train_g_config(config)
    assert config["r14_resume_contract"] == validator.RESUME_CONTRACT
    assert config["resume_from"] == str(validator.SOURCE_CHECKPOINT)
    assert config["resume_mode"] == "training_state"
    assert config["resume_checkpoint_model"] == "raw"
    assert config["resume_optimizer_state"] is True
    assert config["global_batch_size"] == 4
    assert config["per_device_batch_size"] == 2
    assert config["stages"]["stage2"]["epochs"] == 20
    assert config["optimizer_step_contract"]["required_steps"] == 2688
    assert config["optimizer_checkpoint_contract"]["save_steps"] == [2688]
    assert "resume_from_sha256" not in config


def test_launcher_is_training_only_gpu01_and_independent() -> None:
    path = REPO / "scripts/run_r14_inpaint_resume_gpu01.sh"
    text = path.read_text(encoding="utf-8")
    assert 'GPU_LIST="0,1"' in text
    assert "NPROC=2" in text
    assert 'SESSION="safa-r14-inpaint-resume-gpu01-v1"' in text
    assert 'ARTIFACT_ROOT="artifacts/r14_inpaint_resume_gpu01/v1"' in text
    assert 'CHECKPOINT_ROOT="checkpoints/r14_inpaint_resume_gpu01_step2688"' in text
    assert 'NCCL_IB_DISABLE_VALUE="1"' in text
    assert 'NCCL_P2P_DISABLE_VALUE="0"' in text
    assert 'export NCCL_IB_DISABLE="$NCCL_IB_DISABLE_VALUE"' in text
    assert 'export NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE_VALUE"' in text
    assert '"--nproc_per_node=$NPROC"' in text
    assert "-m safa.cli.train_g" in text
    assert "validate_r14_inpaint_resume_gpu01.py" in text
    assert validator.SOURCE_CHECKPOINT_SHA256 in text
    assert "Git SHA:" in text
    for forbidden in (
        "run_r14_inpaint_smoke.py",
        "export_r14_inpaint_ema.py",
        "run_r14_inpaint_generation.py",
        "evaluate_r14_inpaint_feasibility.py",
        "retry",
        "controller",
        "postclaim",
        "/tmp/safa-node3",
    ):
        assert forbidden not in text.lower()
    result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_nccl_transport_and_gpu_binding_are_locked() -> None:
    assert validator.NCCL_TRANSPORT_ENV == {
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
    }
    assert set(validator.GPU_BINDINGS) == {0, 1}
    assert validator.PROJECTED_PEAK_MIB == 8192
    validator._validate_nccl_transport_environment(validator.NCCL_TRANSPORT_ENV)
    with pytest.raises(validator.R14ResumeError, match="NCCL_IB_DISABLE"):
        validator._validate_nccl_transport_environment({"NCCL_P2P_DISABLE": "0"})
    with pytest.raises(validator.R14ResumeError, match="NCCL_P2P_DISABLE"):
        validator._validate_nccl_transport_environment(
            {"NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "1"}
        )


def test_process_isolation_rejects_resume_and_source_writers() -> None:
    validator._validate_process_isolation("", [])
    with pytest.raises(validator.R14ResumeError, match="conflicting tmux"):
        validator._validate_process_isolation("", [validator.SESSION])
    with pytest.raises(validator.R14ResumeError, match="source-checkpoint writer"):
        validator._validate_process_isolation("", [validator.SOURCE_WRITER_SESSION])
    with pytest.raises(validator.R14ResumeError, match="two-GPU resume"):
        validator._validate_process_isolation(
            f"python -m safa.cli.train_g --config {validator.CONFIG}", []
        )
    with pytest.raises(validator.R14ResumeError, match="source-checkpoint writer"):
        validator._validate_process_isolation(
            "python -m safa.cli.train_g --config "
            "configs/medium_v2/experiments/r14_inpaint_feasibility_2560step.yaml",
            [],
        )
    with pytest.raises(validator.R14ResumeError, match="source-checkpoint writer"):
        validator._validate_process_isolation(
            "bash scripts/run_r14_inpaint_feasibility.sh", []
        )


def test_source_checkpoint_contract_accepts_exact_and_rejects_missing_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validator, "_load_checkpoint", lambda _path: source_checkpoint_payload())
    validator._validate_source_checkpoint_payload(Path("unused.pt"))
    monkeypatch.setattr(
        validator,
        "_load_checkpoint",
        lambda _path: source_checkpoint_payload(optimizer_state=False),
    )
    with pytest.raises(validator.R14ResumeError, match="optimizer state is empty"):
        validator._validate_source_checkpoint_payload(Path("unused.pt"))


def test_source_external_contract_rejects_wrong_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path("source")
    checkpoint = source_root / "last.pt"
    (tmp_path / checkpoint).parent.mkdir(parents=True)
    (tmp_path / checkpoint).write_bytes(b"x")
    write_json(tmp_path / source_root / "last_metrics.json", source_metrics(step=2431))
    write_jsonl(
        tmp_path / source_root / "metrics_history.jsonl",
        [{"global_step": 128 * (index + 1)} for index in range(19)],
    )
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(validator, "SOURCE_CHECKPOINT", checkpoint)
    monkeypatch.setattr(validator, "SOURCE_CHECKPOINT_SIZE", 1)
    monkeypatch.setattr(validator, "_sha256", lambda path: {
        "last.pt": validator.SOURCE_CHECKPOINT_SHA256,
        "last_metrics.json": validator.SOURCE_METRICS_SHA256,
        "metrics_history.jsonl": validator.SOURCE_HISTORY_SHA256,
    }[path.name])
    with pytest.raises(validator.R14ResumeError, match="source.last_metrics.global_step"):
        validator._validate_source_files(load_checkpoint=False)


def prepare_final_artifacts(root: Path, *, step: int = 2688) -> None:
    checkpoint_root = root / validator.CHECKPOINT_ROOT
    checkpoint_root.mkdir(parents=True)
    for name in ("last.pt", "step_00002688.pt"):
        (checkpoint_root / name).write_bytes(b"x")
    history = [{"global_step": 128 * (index + 1)} for index in range(19)]
    history.append(final_metrics(step=step))
    write_json(
        checkpoint_root / "manifest.json",
        {
            "checkpoint": str(validator.CHECKPOINT_ROOT / "last.pt"),
            "metrics": final_metrics(step=step),
            "history": history,
            "distributed": {"enabled": True, "world_size": 2, "backend": "nccl"},
        },
    )
    write_json(
        checkpoint_root / "completion.json",
        {
            "contract_type": "safa_r14_inpaint_exact_optimizer_steps_v1",
            "completed": True,
            "optimizer_steps": 2688,
            "ema_available": True,
            "checkpoint": str(validator.CHECKPOINT_ROOT / "last.pt"),
            "manifest": str(validator.CHECKPOINT_ROOT / "manifest.json"),
        },
    )
    write_json(checkpoint_root / "last_metrics.json", final_metrics(step=step))
    write_jsonl(checkpoint_root / "metrics_history.jsonl", [final_metrics(step=step)])


def test_artifact_requires_step2688_resumed_optimizer_and_full_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_final_artifacts(tmp_path)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "_validate_source_files", lambda **_kwargs: None)
    monkeypatch.setattr(
        validator,
        "_load_checkpoint",
        lambda path: step_checkpoint_payload()
        if path.name == "step_00002688.pt"
        else final_checkpoint_payload(),
    )
    validator.validate_artifact()

    write_json(
        tmp_path / validator.CHECKPOINT_ROOT / "last_metrics.json",
        final_metrics(step=2687),
    )
    with pytest.raises(validator.R14ResumeError, match="last_metrics.global_step"):
        validator.validate_artifact()


def test_artifact_rejects_nonresumed_or_missing_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_final_artifacts(tmp_path)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validator, "_validate_source_files", lambda **_kwargs: None)
    monkeypatch.setattr(
        validator,
        "_load_checkpoint",
        lambda path: step_checkpoint_payload(optimizer_state=False)
        if path.name == "step_00002688.pt"
        else final_checkpoint_payload(optimizer_state=False),
    )
    with pytest.raises(validator.R14ResumeError, match="raw, EMA, and optimizer"):
        validator.validate_artifact()

    monkeypatch.setattr(
        validator,
        "_load_checkpoint",
        lambda path: step_checkpoint_payload()
        if path.name == "step_00002688.pt"
        else final_checkpoint_payload(),
    )
    write_json(
        tmp_path / validator.CHECKPOINT_ROOT / "last_metrics.json",
        final_metrics(optimizer_resumed=False),
    )
    with pytest.raises(validator.R14ResumeError, match="optimizer_resumed"):
        validator.validate_artifact()
