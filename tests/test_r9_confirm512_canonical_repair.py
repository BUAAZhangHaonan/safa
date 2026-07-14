from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from safa.evaluation import r9_confirm512_canonical_repair as repair


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_r9_confirm512_canonical_repair.py"
SPEC = importlib.util.spec_from_file_location(
    "run_r9_confirm512_canonical_repair", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_cli_requires_explicit_execute_and_busy_gpu_permission() -> None:
    base = [
        "--campaign-id",
        repair.SOURCE_CAMPAIGN_ID,
        "--repair-id",
        "canonical-v1",
        "--source-failure-sha256",
        "a" * 64,
    ]
    with pytest.raises(SystemExit):
        runner.parse_args(base)
    with pytest.raises(SystemExit):
        runner.parse_args([*base, "--execute"])


def test_exact_root_inventory_hashes_every_png_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    shard = campaign / "confirm512" / "native" / "shards" / "shard_0"
    _write(shard / "generation_result.json", b"{}")
    _write(shard / "completion.json", b"{}")
    for index in range(32):
        _write(shard / "generated_images" / f"{index:08d}.png", bytes([index]))
    (shard.parent / "shared").mkdir()
    monkeypatch.setattr(repair, "EXPECTED_ARMS", ("native",))
    monkeypatch.setattr(repair, "EXPECTED_ROOT_COUNT", 1)
    monkeypatch.setattr(repair, "EXPECTED_SHARD_COUNT", 1)
    monkeypatch.setattr(repair, "EXPECTED_ROOT_FILE_COUNT", 34)
    monkeypatch.setattr(repair, "EXPECTED_SHARED_FILE_COUNT", 0)
    monkeypatch.setattr(repair, "EXPECTED_FILE_COUNT", 34)
    monkeypatch.setattr(repair, "EXPECTED_PNG_COUNT", 32)
    first = repair.validate_canonical_native_inventory(
        campaign_root=campaign,
        expected_roots=(shard,),
    )
    assert first["root_count"] == 1
    assert first["root_file_count"] == 34
    assert first["shared_file_count"] == 0
    assert first["file_count"] == 34
    assert first["png_count"] == 32
    _write(shard / "generated_images" / "00000000.png", b"tampered")
    second = repair.validate_canonical_native_inventory(
        campaign_root=campaign,
        expected_roots=(shard,),
    )
    assert second["inventory_sha256"] != first["inventory_sha256"]


def test_inventory_rejects_symlink_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    shard = campaign / "confirm512" / "native" / "shards" / "shard_0"
    _write(shard / "generation_result.json", b"{}")
    _write(shard / "completion.json", b"{}")
    for index in range(32):
        _write(shard / "generated_images" / f"{index:08d}.png", bytes([index]))
    (shard.parent / "shared").mkdir()
    target = tmp_path / "outside"
    _write(target)
    try:
        (shard / "linked").symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")
    monkeypatch.setattr(repair, "EXPECTED_ARMS", ("native",))
    monkeypatch.setattr(repair, "EXPECTED_ROOT_COUNT", 1)
    monkeypatch.setattr(repair, "EXPECTED_SHARD_COUNT", 1)
    monkeypatch.setattr(repair, "EXPECTED_ROOT_FILE_COUNT", 34)
    monkeypatch.setattr(repair, "EXPECTED_SHARED_FILE_COUNT", 0)
    monkeypatch.setattr(repair, "EXPECTED_FILE_COUNT", 34)
    monkeypatch.setattr(repair, "EXPECTED_PNG_COUNT", 32)
    with pytest.raises(repair.CanonicalNativeRepairError, match="symlinks"):
        repair.validate_canonical_native_inventory(
            campaign_root=campaign,
            expected_roots=(shard,),
        )


def _row(
    sample_id: str,
    *,
    candidate: str,
    native: str,
    candidate_sha: str,
    native_sha: str,
    candidate_cosine: float,
    native_cosine: float,
    edev_cosine: float,
    native_edev_cosine: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source": "/source.png",
        "candidate": candidate,
        "native": native,
        "source_sha256": "0" * 64,
        "candidate_sha256": candidate_sha,
        "native_sha256": native_sha,
        "metrics": {
            "candidate_cosine": candidate_cosine,
            "native_cosine": native_cosine,
            "edev_cosine": edev_cosine,
            "native_edev_cosine": native_edev_cosine,
        },
    }


def test_canonical_view_uses_standalone_native_without_relabeling_candidate() -> None:
    native = {
        "rows": [
            _row(
                "s",
                candidate="/standalone.png",
                native="/standalone.png",
                candidate_sha="1" * 64,
                native_sha="1" * 64,
                candidate_cosine=0.2,
                native_cosine=0.2,
                edev_cosine=0.3,
                native_edev_cosine=0.3,
            )
        ]
    }
    candidate = {
        "rows": [
            _row(
                "s",
                candidate="/candidate.png",
                native="/embedded.png",
                candidate_sha="2" * 64,
                native_sha="3" * 64,
                candidate_cosine=0.8,
                native_cosine=0.21,
                edev_cosine=0.7,
                native_edev_cosine=0.31,
            )
        ],
        "output_contract": {"shards": [], "images": []},
        "output_sha256": "4" * 64,
        "evidence_binding_sha256": "5" * 64,
    }
    canonical = repair._canonicalize_candidate(candidate, native)
    row = canonical["rows"][0]
    assert row["candidate"] == "/candidate.png"
    assert row["candidate_sha256"] == "2" * 64
    assert row["native"] == "/standalone.png"
    assert row["native_sha256"] == "1" * 64
    assert row["metrics"]["native_cosine"] == 0.2
    assert row["metrics"]["native_edev_cosine"] == 0.3
    assert canonical["source_generation_output_sha256"] == "4" * 64
    assert candidate["rows"][0]["native"] == "/embedded.png"


def test_runner_materializes_contract_without_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign = tmp_path / "campaign"
    phase_request = SimpleNamespace(runs=())
    plan = SimpleNamespace(phase="confirm512", runs=tuple(range(48)))
    effective = {"campaign_root": str(campaign)}
    prepared = SimpleNamespace(
        contract={
            "canonical_native_policy": {"generation_execution_count": 0},
            "generation_inventory": {"inventory_sha256": "1" * 64},
        },
        contract_sha256="2" * 64,
        namespace_root=campaign / "canonical",
    )
    monkeypatch.setenv("TMUX", "yes")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "_load_campaign", lambda *_: ({}, effective, plan, phase_request)
    )
    monkeypatch.setattr(runner, "_assert_generation_terminal", lambda *_: None)
    inventory = {"inventory_sha256": "1" * 64}
    monkeypatch.setattr(runner, "_inventory_for", lambda *_: inventory)
    monkeypatch.setattr(runner, "build_canonical_native_repair", lambda *_: prepared)
    contract_path = campaign / "repair_contract.json"
    monkeypatch.setattr(runner, "materialize_repair_contract", lambda *_: contract_path)
    monkeypatch.setattr(
        runner.driver,
        "build_resource_scheduler",
        lambda *_: ("scheduler", "gpu-bindings", "status"),
    )

    class _Callbacks:
        def __init__(self, **kwargs):
            self.quality = "quality"
            self.arcface = "arcface"

    monkeypatch.setattr(runner.driver, "R9ProductionEvaluatorCallbacks", _Callbacks)
    monkeypatch.setattr(
        runner,
        "execute_canonical_evaluations",
        lambda *_, **__: {
            "evaluator_unit_count": 5,
            "winner_arm_id": "paper_eta_0p125",
            "selection_sha256": "3" * 64,
            "verdict": "winner_locked",
        },
    )
    result = runner.main(
        [
            "--campaign-id",
            repair.SOURCE_CAMPAIGN_ID,
            "--repair-id",
            "canonical-v1",
            "--source-failure-sha256",
            "a" * 64,
            "--execute",
            "--allow-busy-gpus",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["generation_execution_count"] == 0
    assert payload["evaluator_unit_count"] == 5
    assert payload["status"] == "winner_locked"


def test_live_v8_inventory_is_exact_when_frozen_artifacts_are_present() -> None:
    campaign = ROOT / (
        "artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v8"
    )
    if not campaign.is_dir():
        pytest.skip("frozen formal-v8 artifacts are not present")
    roots = tuple(
        campaign / "confirm512" / arm / "shards" / f"shard_{index}"
        for arm in repair.EXPECTED_ARMS
        for index in range(repair.EXPECTED_SHARD_COUNT)
    )
    inventory = repair.validate_canonical_native_inventory(
        campaign_root=campaign,
        expected_roots=roots,
    )
    assert inventory["root_count"] == 48
    assert inventory["root_file_count"] == 2944
    assert inventory["shared_file_count"] == 9
    assert inventory["file_count"] == 2953
    assert inventory["png_count"] == 2560


def test_five_evaluators_materialize_automatic_gate_and_repair_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_request = SimpleNamespace(
        phase_root=tmp_path / "source" / "confirm512",
        manifest_sha256="4" * 64,
        bootstrap_seed=7,
    )
    runs = {
        arm_id: {
            "logical_run_id": arm_id,
            "arm_id": arm_id,
            "output_contract": {"logical_run_id": arm_id, "shards": []},
            "evidence_binding_sha256": "5" * 64,
        }
        for arm_id in repair.EXPECTED_ARMS
    }
    prepared = SimpleNamespace(
        phase_request=phase_request,
        namespace_root=tmp_path / "repair",
        contract_sha256="6" * 64,
        canonical_runs=runs,
        validated_phase_request={"manifest_ids": ["s"]},
        request=SimpleNamespace(campaign_id=repair.SOURCE_CAMPAIGN_ID),
        contract={"generation_inventory": {"inventory_sha256": "7" * 64}},
    )
    monkeypatch.setattr(
        repair,
        "replace",
        lambda value, **kwargs: SimpleNamespace(**(vars(value) | kwargs)),
    )
    monkeypatch.setattr(repair, "_sample_evidence", lambda run: (run["arm_id"],))
    calls: list[tuple[str, str]] = []

    def quality(_, run, __, role, ___):
        calls.append(("quality", str(run["arm_id"])))
        return {"quality_evidence_sha256": (str(run["arm_id"])[0] * 64)}

    def arcface(_, run, __, ___):
        calls.append(("arcface", str(run["arm_id"])))
        return {"arcface_evidence_sha256": (str(run["arm_id"])[-1] * 64)}

    def arm_report(*, run, **kwargs):
        better = run["arm_id"] == "paper_eta_0p125"
        return {
            "arm_id": run["arm_id"],
            "config_sha256": "8" * 64,
            "source_generation_output_sha256": "9" * 64,
            "canonical_evidence_binding_sha256": "a" * 64,
            "evaluator_evidence_sha256": "b" * 64,
            "passed_coverage": True,
            "coverage_failures": [],
            "quality": {"kid": 0.01 if better else 0.02, "fid": 100.0},
            "representation": {"delta_edev": 0.1, "e0": 0.8},
        }

    written: list[tuple[Path, str]] = []
    monkeypatch.setattr(repair, "_evaluate_quality", quality)
    monkeypatch.setattr(repair, "_evaluate_arcface", arcface)
    monkeypatch.setattr(repair, "_arm_report", arm_report)
    monkeypatch.setattr(
        repair,
        "write_immutable_contract",
        lambda path, payload, *, digest_field: written.append(
            (Path(path), digest_field)
        ),
    )
    result = repair.execute_canonical_evaluations(
        prepared,
        quality_evaluator=object(),
        arcface_evaluator=object(),
    )
    assert sorted(calls) == sorted(
        [
            ("quality", "native"),
            ("quality", "flow_map2_normalized_eta_0p125"),
            ("quality", "paper_eta_0p125"),
            ("arcface", "flow_map2_normalized_eta_0p125"),
            ("arcface", "paper_eta_0p125"),
        ]
    )
    assert result["evaluator_unit_count"] == 5
    assert result["winner_arm_id"] == "paper_eta_0p125"
    assert result["generation_execution_count"] == 0
    assert {path.name for path, _ in written} == {
        "automatic_evidence.json",
        "gate_contract.json",
        "selection.json",
        "repair_result.json",
    }
