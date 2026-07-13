from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_r9_meanflow_semigroup_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_r9_meanflow_semigroup_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_dry_run_resolves_contract_without_cuda_or_generation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()

    def forbidden(*args, **kwargs):
        pytest.fail("R9 dry-run must not initialize CUDA or execute generation")

    monkeypatch.setattr(module, "run_guidance_from_config", forbidden)
    monkeypatch.setattr(
        module.sys.modules["safa.evaluation.meanflow_guidance_runner"].torch.cuda,
        "is_initialized",
        forbidden,
    )

    assert module.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execute"] is False
    assert payload["shard"] == {"index": 0, "count": 4}
    assert payload["preflight_contract"]["attention_backend"] == "native"
    assert len(payload["arm_config_sha256"]) == 64


def test_execute_requires_allow_busy_authorization(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "run_guidance_from_config",
        lambda *args, **kwargs: pytest.fail("unauthorized execute must stop before generation"),
    )

    assert module.main(["--execute", "--shard-index", "2"]) == 1
    assert "--allow-busy-gpus" in capsys.readouterr().err


def test_execute_passes_only_locked_config_and_shard_coordinates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    def fake_run(config, *, output_dir, shard_index, num_shards):
        captured.update(
            config=config,
            output_dir=str(output_dir),
            shard_index=shard_index,
            num_shards=num_shards,
        )
        return {"status": "complete"}

    monkeypatch.setattr(module, "run_guidance_from_config", fake_run)

    assert (
        module.main(
            ["--execute", "--allow-busy-gpus", "--shard-index", "3"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}
    assert captured["output_dir"].endswith("/shard_3")
    assert captured["shard_index"] == 3
    assert captured["num_shards"] == 4
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["experiment_contract"] == "safa_r9_meanflow_v1"
    assert config["attention_backend"] == "native"


def test_finalize_cli_calls_bound_production_finalizer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_finalize", lambda: {"gate_passed": True})

    assert module.main(["--finalize"]) == 0
    assert json.loads(capsys.readouterr().out) == {"gate_passed": True}
