from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/prepare_r9_full_continuation.py"
spec = importlib.util.spec_from_file_location("prepare_r9_full_continuation", SCRIPT)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def test_e2e_cli_exposes_three_strict_phases() -> None:
    for phase in ("e2e-prepare", "e2e-run", "e2e-finalize"):
        assert cli.parse_args(["--phase", phase]).phase == phase
    with pytest.raises(RuntimeError, match="requires --execute"):
        cli.main(["--phase", "e2e-run"])
    with pytest.raises(RuntimeError, match="requires --execute"):
        cli.main(["--phase", "e2e-finalize"])


def test_e2e_run_requires_tmux_before_any_gpu_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TMUX", raising=False)
    with pytest.raises(RuntimeError, match="inside tmux"):
        cli._run_e2e(object())


def test_e2e_run_uses_bound_runtime_resource_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class StopAfterGuard(RuntimeError):
        pass

    captured: dict[str, dict] = {}

    class Guard:
        def __init__(self, policy: dict, *, monitor_path: Path) -> None:
            captured["policy"] = policy
            captured["monitor_path"] = {"path": str(monitor_path)}
            raise StopAfterGuard

    class Driver:
        REPO_ROOT = tmp_path
        FULL_CONTINUATION_CHILD_CAMPAIGN_ID = "r9-report-only-formal-v9"
        FullRuntimeGuard = Guard

        @staticmethod
        def _full_admission_preflight() -> dict:
            return {}

        @staticmethod
        def build_resource_scheduler(runtime: dict):
            del runtime
            return object(), {}, object()

        @staticmethod
        def _mapping(value, label: str) -> dict:
            if not isinstance(value, dict):
                raise ValueError(label)
            return dict(value)

    policy = {"gpu_indices": [0, 1, 2, 3], "retry_count": 0}
    runtime = {"full_e2e_bootstrap": {"resource_policy": policy}}
    plan = {
        "full_e2e_plan_sha256": "a" * 64,
        "e2e_request": {"path": "frozen.yaml", "file_sha256": "b" * 64},
    }
    monkeypatch.setenv("TMUX", "test")
    monkeypatch.setattr(
        cli,
        "_load_effective_runtime",
        lambda driver, *, allow_provisional: runtime,
    )
    monkeypatch.setattr(cli, "_load_e2e_plan", lambda driver: plan)

    with pytest.raises(StopAfterGuard):
        cli._run_e2e(Driver)

    assert captured["policy"] == policy


def test_e2e_evaluators_allow_pre_e2e_resource_profile_bootstrap(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    source_path = tmp_path / "source.png"
    native_path = tmp_path / "native.png"
    generated_path = tmp_path / "generated.png"
    for path, content in (
        (source_path, b"source"),
        (native_path, b"native"),
        (generated_path, b"generated"),
    ):
        path.write_bytes(content)
    root = (
        tmp_path
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / "r9-report-only-formal-v9/full_e2e"
    )
    for arm in ("native", "paper_eta_0p125"):
        arm_root = root / "generation" / arm
        arm_root.mkdir(parents=True)
        generated = native_path if arm == "native" else generated_path
        row = {
            "sample_id": "sample-000",
            "source": str(source_path),
            "native": str(native_path),
            "generated": str(generated),
        }
        (arm_root / "per_sample.jsonl").write_text(
            json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
        )
        (arm_root / "generation_result.json").write_text(
            json.dumps({"arm_id": arm}), encoding="utf-8"
        )
    native_config = tmp_path / "native.yaml"
    native_config.write_text("arm_config_sha256: " + "1" * 64 + "\n", encoding="utf-8")

    class Request:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeSampleEvidence(Request):
        pass

    class Callbacks:
        def __init__(self, **kwargs) -> None:
            events.append("callbacks")
            assert kwargs["runtime"] == {"python": sys.executable}

        def arcface(self, request) -> list[dict]:
            events.append(f"arcface:{request.phase}:{request.arm_id}")
            return []

        def quality(self, request) -> dict:
            events.append(f"quality:{request.image_role}")
            return {}

    class Driver:
        REPO_ROOT = tmp_path
        FULL_CONTINUATION_CHILD_CAMPAIGN_ID = "r9-report-only-formal-v9"
        ArcFaceEvaluationRequest = Request
        QualityEvaluationRequest = Request
        R9ProductionEvaluatorCallbacks = Callbacks
        SampleEvidence = FakeSampleEvidence

        @staticmethod
        def load_full_continuation_request(*, allow_pre_e2e_profiles: bool = False):
            assert allow_pre_e2e_profiles is True
            events.append("load_pre_e2e")
            return {"python": sys.executable}, Path("request.yaml"), "source"

        @staticmethod
        def _canonical_json_sha256(value) -> str:
            return _digest(value)

        @staticmethod
        def _continuation_for_runtime(runtime):
            return {"runtime": runtime}

        @staticmethod
        def _require_full_selection_binding(continuation, runtime):
            del continuation, runtime
            return {"winner": {"config_sha256": "2" * 64}}

        @staticmethod
        def materialize_full_e2e_resource_profiles(runtime) -> dict:
            del runtime
            events.append("materialize_profiles")
            return {}

        @staticmethod
        def _rebuild_full_e2e_evidence(runtime, *, require_materialized_result=False):
            del runtime, require_materialized_result
            return {"plan": plan, "result": {"full_e2e_result_sha256": "3" * 64}}

        @staticmethod
        def _write_exclusive_bytes(path: Path, content: bytes) -> None:
            events.append(f"write:{path.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    plan = {
        "manifest": {"sample_ids": ["sample-000"], "path": "manifest.json"},
        "runs": [
            {
                "arm_id": "native",
                "runtime_config": str(native_config.relative_to(tmp_path)),
            }
        ],
    }
    runtime = {
        "campaign_root": (
            "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
            "r9-report-only-formal-v9"
        ),
        "evaluation": {
            "quality": {"real_index": {"path": "real.json", "sha256": "4" * 64}}
        },
    }

    result = cli._run_e2e_evaluators(
        Driver,
        runtime,
        plan,
        scheduler=object(),
        gpu_bindings={},
        peer_status_store=object(),
        runtime_guard=object(),
    )

    assert result["full_e2e_result_sha256"] == "3" * 64
    assert events == [
        "load_pre_e2e",
        "callbacks",
        "arcface:full_e2e:paper_eta_0p125",
        "quality:native",
        "quality:candidate",
        "materialize_profiles",
        "write:run_result.json",
    ]


def test_e2e_prepare_dry_run_has_zero_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def prepare(driver, *, materialize: bool):
        calls.append((driver, materialize))
        return {
            "status": "dry_run_validated",
            "artifact_write_count": 0,
            "generation_execution_count": 0,
            "evaluator_execution_count": 0,
        }

    sentinel = object()
    monkeypatch.setattr(cli, "_driver", lambda: sentinel)
    monkeypatch.setattr(cli, "_prepare_e2e", prepare)
    before = list(tmp_path.iterdir())
    assert cli.main(["--phase", "e2e-prepare"]) == 0
    assert calls == [(sentinel, False)]
    assert list(tmp_path.iterdir()) == before
    assert cli.main(["--phase", "e2e-prepare", "--execute"]) == 0
    assert calls[-1] == (sentinel, True)


def test_real_e2e_prepare_subprocess_is_zero_write() -> None:
    campaign_root = (
        REPO_ROOT
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / "r9-report-only-formal-v9"
    )

    def snapshot() -> dict[str, str]:
        if not campaign_root.exists():
            return {}
        return {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(campaign_root.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "e2e-prepare",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "dry_run_validated"
    assert result["artifact_write_count"] == 0
    assert result["generation_execution_count"] == 0
    assert result["evaluator_execution_count"] == 0
    assert snapshot() == before


def test_real_e2e_prepare_execute_capture_has_closed_artifact_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = cli._driver()
    captured: dict[Path, bytes] = {}

    def capture(path: Path, content: bytes) -> None:
        resolved = path.resolve()
        assert resolved not in captured
        captured[resolved] = content

    monkeypatch.setattr(driver, "_write_immutable_bytes", capture)
    result = cli._prepare_e2e(driver, materialize=True)
    assert result["status"] == "prepared"
    assert result["artifact_write_count"] == 6
    assert len(captured) == 6
    assert [path.name for path in captured] == [
        "full_continuation_selection.json",
        "full_continuation_contract.json",
        "provisional_runtime.json",
        "native.yaml",
        "paper_eta_0p125.yaml",
        "plan.json",
    ]

    selection_candidates = [
        (path, content)
        for path, content in captured.items()
        if path.name == "full_continuation_selection.json"
    ]
    assert len(selection_candidates) == 1
    expected_selection = json.loads(selection_candidates[0][1])
    selection_path, selection_content, selection_binding = (
        cli.full_selection_binding(
            expected_selection, repo_root=driver.REPO_ROOT
        )
    )
    continuation_candidates = [
        (path, content)
        for path, content in captured.items()
        if path.name == "full_continuation_contract.json"
    ]
    assert len(continuation_candidates) == 1
    expected_continuation = json.loads(continuation_candidates[0][1])
    continuation_path, continuation_content, continuation_binding = (
        cli.full_continuation_contract_binding(
            expected_continuation, repo_root=driver.REPO_ROOT
        )
    )
    assert captured[selection_path] == selection_content
    assert captured[continuation_path] == continuation_content

    provisional_path = cli._provisional_runtime_path(driver)
    provisional = json.loads(captured[provisional_path])
    assert provisional["continuation"] == continuation_binding
    assert provisional["campaign_runtime_sha256"] == driver._canonical_json_sha256(
        {
            key: value
            for key, value in provisional.items()
            if key != "campaign_runtime_sha256"
        }
    )

    plan_path = cli._e2e_root(driver) / "plan.json"
    plan = json.loads(captured[plan_path])
    assert plan["selection_sha256"] == selection_binding["contract_sha256"]
    assert (
        plan["continuation_contract_sha256"]
        == continuation_binding["contract_sha256"]
    )
    assert (
        plan["provisional_runtime"]["contract_sha256"]
        == provisional["campaign_runtime_sha256"]
    )
    for run in plan["runs"]:
        config_path = driver.REPO_ROOT / run["runtime_config"]
        content = captured[config_path]
        assert hashlib.sha256(content).hexdigest() == run["runtime_config_sha256"]
        config = yaml.safe_load(content)
        assert (
            config["r9_continuation_contract_sha256"]
            == continuation_binding["contract_sha256"]
        )


def test_e2e_plan_digest_tamper_fails_closed(tmp_path: Path) -> None:
    class Driver:
        REPO_ROOT = tmp_path
        FULL_CONTINUATION_CHILD_CAMPAIGN_ID = "r9-report-only-formal-v9"

        @staticmethod
        def _read_json_mapping(path: Path, label: str) -> dict:
            del label
            return json.loads(path.read_text(encoding="utf-8"))

        @staticmethod
        def _canonical_json_sha256(value: dict) -> str:
            return _digest(value)

    root = cli._e2e_root(Driver)
    root.mkdir(parents=True)
    plan = {
        "schema_version": 1,
        "contract_type": "safa_r9_full_e2e_plan_v1",
        "campaign_id": Driver.FULL_CONTINUATION_CHILD_CAMPAIGN_ID,
        "continuation_contract_sha256": "1" * 64,
        "selection_sha256": "2" * 64,
        "request_config": {},
        "e2e_request": {},
        "generation_batch_benchmark": {},
        "provisional_runtime": {},
        "manifest": {},
        "generation_policy": {},
        "runs": [],
    }
    plan["full_e2e_plan_sha256"] = _digest(plan)
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli._load_e2e_plan(Driver) == plan
    plan["selection_sha256"] = "3" * 64
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        cli._load_e2e_plan(Driver)
