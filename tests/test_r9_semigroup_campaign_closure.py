from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator

from PIL import Image
import pytest
import yaml

from safa.evaluation.meanflow_guidance_runner import (
    resolve_frozen_effective_guidance_config,
)
from safa.evaluation.r9_determinism import (
    canonical_r9_arm_config_digest,
    validate_r9_execution_config,
)
import safa.evaluation.r9_semigroup_campaign_closure as closure_module
from safa.evaluation.r9_semigroup_campaign_closure import (
    CampaignSemigroupClosureError,
    finalize_campaign_semigroup_closure,
    finalize_campaign_semigroup_policy_recovery,
    prepare_campaign_semigroup_visual_review,
    resolve_formal_campaign_semigroup_closure,
)
from safa.evaluation.r9_semigroup_contracts import (
    R9_SEMIGROUP_RECOVERY_AUTHORIZATION_ID,
    R9_SEMIGROUP_RECOVERY_POLICY_SHA256,
    R9_SEMIGROUP_RECOVERY_POLICY_VERSION,
    R9_SEMIGROUP_RECOVERY_SELECTION_RULE,
    canonical_r9_semigroup_preflight_digest,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "configs"
    / "medium_v2"
    / "experiments"
    / "r9_meanflow_semigroup_preflight.yaml"
)
CLI = ROOT / "scripts" / "finalize_r9_meanflow_semigroup_campaign.py"
PREPARE_CLI = ROOT / "scripts" / "prepare_r9_meanflow_semigroup_review.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: dict[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sample_digest(sample_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode()
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (8, 8),
        (value % 256, (value * 3) % 256, (value * 7) % 256),
    ).save(path)


def _repo_relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


@contextmanager
def _cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _fixture(tmp_path: Path, *, latent_residual: float = 0.05) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    bootstrap_id = "bootstrap-r9-v2"
    formal_id = "formal-r9-v2"
    campaign_root = (
        repo_root
        / "artifacts"
        / "r9_meanflow_flow_map_guidance"
        / "campaigns"
        / bootstrap_id
    )
    config_path = (
        campaign_root / "runtime_configs" / "preflight" / "semigroup_preflight.yaml"
    )
    shard_root = campaign_root / "preflight" / "semigroup_preflight" / "shards"
    output_root = (
        repo_root
        / "artifacts"
        / "r9_meanflow_flow_map_guidance"
        / "semigroup_campaign_closures"
        / f"{bootstrap_id}__for__{formal_id}"
    )
    manifest = (
        repo_root
        / "configs"
        / "medium_v2"
        / "experiments"
        / "r9_manifests"
        / "calibration_64.jsonl"
    )
    sample_ids = [f"sample-{index:02d}" for index in range(64)]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(
            json.dumps({"sample_id": sample_id}, sort_keys=True) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(base, dict)
    config = {
        **base,
        "experiment_name": "preflight__semigroup_preflight",
        "out_dir": _repo_relative(repo_root, shard_root / "shard_0"),
        "sample_id_manifest": _repo_relative(repo_root, manifest),
        "sample_id_manifest_sha256": _sha(manifest),
        "calibration_sample_id_manifest": _repo_relative(repo_root, manifest),
        "calibration_sample_id_manifest_sha256": _sha(manifest),
        "asset_digest_cache": _repo_relative(
            repo_root, shard_root / "shared" / "semigroup_preflight.assets.json"
        ),
        "contact_sheets": False,
        "r9_campaign_id": bootstrap_id,
        "r9_campaign_runtime_sha256": "0" * 64,
        "r9_manifest_contracts_sha256": "8" * 64,
        "r9_phase_manifest_sha256": _sha(manifest),
    }
    sources = repo_root / "data" / "sources"
    sources.mkdir(parents=True)
    source_rows = []
    for global_index, sample_id in enumerate(sample_ids):
        source = sources / f"{sample_id}.png"
        _image(source, 10 + global_index)
        source_rows.append(
            {"sample_id": sample_id, "image_path": str(source.resolve())}
        )
    source_index = repo_root / str(config["index"])
    source_index.parent.mkdir(parents=True, exist_ok=True)
    source_index.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in source_rows
        ),
        encoding="utf-8",
    )
    config["index_sha256"] = _sha(source_index)
    checkpoint = repo_root / str(config["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic-meanflow-checkpoint")
    config["checkpoint_sha256"] = _sha(checkpoint)
    campaign_runtime = {
        "schema_version": 1,
        "campaign_id": bootstrap_id,
        "campaign_root": _repo_relative(repo_root, campaign_root),
        "determinism_policy_sha256": config["determinism_policy_sha256"],
        "attention_backend": config["attention_backend"],
        "manifest_contracts_sha256": config["r9_manifest_contracts_sha256"],
        "checkpoint": {
            "path": config["checkpoint"],
            "sha256": config["checkpoint_sha256"],
        },
        "manifests": {
            "calibration_64": {
                "path": config["sample_id_manifest"],
                "sha256": config["sample_id_manifest_sha256"],
                "sample_count": 64,
                "ordered_sample_id_sha256": _sample_digest(sample_ids),
            }
        },
    }
    campaign_runtime["campaign_runtime_sha256"] = _canonical_digest(
        campaign_runtime, "campaign_runtime_sha256"
    )
    config["r9_campaign_runtime_sha256"] = campaign_runtime["campaign_runtime_sha256"]
    _write_json(campaign_root / "campaign_runtime.json", campaign_runtime)
    config["semigroup_preflight_contract_sha256"] = (
        canonical_r9_semigroup_preflight_digest(config)
    )
    with _cwd(repo_root):
        config = resolve_frozen_effective_guidance_config(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    execution = validate_r9_execution_config(config)
    executed_config = {**config, "r9_execution_contract": execution}
    arm_sha256 = canonical_r9_arm_config_digest(executed_config)
    executed_config["arm_config_sha256"] = arm_sha256
    for shard_index in range(4):
        shard_dir = shard_root / f"shard_{shard_index}"
        shard_dir.mkdir(parents=True)
        ids = sample_ids[shard_index::4]
        per_sample = []
        semigroup_rows = []
        sample_manifest_rows = []
        for local_index, sample_id in enumerate(ids):
            source = sources / f"{sample_id}.png"
            native = shard_dir / "native_images" / f"{local_index:08d}.png"
            generated = shard_dir / "generated_images" / f"{local_index:08d}.png"
            _image(native, 80 + local_index + shard_index * 16)
            _image(generated, 150 + local_index + shard_index * 16)
            splits: dict[str, Any] = {}
            for split, directory in (
                ("0.25", "t_cut_0p25"),
                ("0.5", "t_cut_0p5"),
                ("0.75", "t_cut_0p75"),
            ):
                decoded = (
                    shard_dir
                    / "semigroup_split_images"
                    / directory
                    / f"{local_index:08d}.png"
                )
                _image(
                    decoded,
                    30
                    + local_index
                    + shard_index * 16
                    + {"0.25": 0, "0.5": 64, "0.75": 128}[split],
                )
                splits[split] = {
                    "latent_residual": latent_residual,
                    "endpoint_e0_cosine": 0.99,
                    "decoded_pixel_l1": 0.01,
                    "decoded_psnr": 35.0,
                    "decoded_image": _repo_relative(repo_root, decoded),
                }
            row = {
                "sample_id": sample_id,
                "source": str(source.resolve()),
                "native": _repo_relative(repo_root, native),
                "generated": _repo_relative(repo_root, generated),
                "semigroup": splits,
            }
            per_sample.append(row)
            semigroup_rows.append({"sample_id": sample_id, "splits": splits})
            sample_manifest_rows.append(
                {
                    "ordinal": local_index,
                    "sample_id": sample_id,
                    "source": str(source.resolve()),
                }
            )
        (shard_dir / "per_sample.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in per_sample
            ),
            encoding="utf-8",
        )
        (shard_dir / "sample_id_manifest.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in sample_manifest_rows
            ),
            encoding="utf-8",
        )
        _write_json(
            shard_dir / "semigroup.json",
            {
                "mode": "semigroup",
                "split_times": [0.25, 0.5, 0.75],
                "rows": semigroup_rows,
            },
        )
        resume = {
            "mode": "semigroup",
            "seed": 1337,
            "shard": {"index": shard_index, "count": 4},
            "arm_config_sha256": arm_sha256,
            "sample_id_sha256": _sample_digest(ids),
            "checkpoint": {"sha256": config["checkpoint_sha256"]},
            "input_sample_manifest": {
                "path": config["sample_id_manifest"],
                "sha256": config["sample_id_manifest_sha256"],
            },
            "r9_execution_contract": execution,
            "config": executed_config,
        }
        generation = {
            "schema_version": 1,
            "status": "complete",
            "mode": "semigroup",
            "sample_count": 16,
            "sample_id_sha256": _sample_digest(ids),
            "shard": {"index": shard_index, "count": 4},
            "arm_config_sha256": arm_sha256,
            "r9_execution_contract": execution,
            "checkpoint": {
                "sha256": config["checkpoint_sha256"],
                "attention_backend_requested": "native",
                "attention_backend_resolved": "native",
            },
            "config": executed_config,
            "resume_contract": resume,
        }
        generation_bytes = (
            json.dumps(
                generation, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode()
        (shard_dir / "generation_result.json").write_bytes(generation_bytes)
        (shard_dir / "run_manifest.json").write_bytes(generation_bytes)
        _write_json(shard_dir / "resume_contract.json", resume)
        completion = {
            "schema_version": 1,
            "status": "complete",
            "sample_count": 16,
            "sample_id_sha256": _sample_digest(ids),
            "arm_config_sha256": arm_sha256,
            "generation_result": _repo_relative(
                repo_root, shard_dir / "generation_result.json"
            ),
            "run_manifest": _repo_relative(repo_root, shard_dir / "run_manifest.json"),
        }
        _write_json(shard_dir / "completion.json", completion)
        verified = {
            "schema_version": 1,
            "contract_type": "safa_r9_verified_worker_completion_v1",
            "worker_id": f"preflight:semigroup_preflight:shard-{shard_index}",
            "runtime_config_sha256": _sha(config_path),
            "completion_sha256": _sha(shard_dir / "completion.json"),
            "generation_result_sha256": _sha(shard_dir / "generation_result.json"),
            "run_manifest_sha256": _sha(shard_dir / "run_manifest.json"),
            "sample_count": 16,
            "sample_id_sha256": _sample_digest(ids),
            "arm_config_sha256": arm_sha256,
            "manifest_contracts_sha256": config["r9_manifest_contracts_sha256"],
            "phase_manifest_sha256": config["r9_phase_manifest_sha256"],
            "campaign_runtime_sha256": config["r9_campaign_runtime_sha256"],
        }
        verified["verified_completion_sha256"] = _canonical_digest(
            verified, "verified_completion_sha256"
        )
        _write_json(shard_dir / "verified_completion.json", verified)
        (shard_dir / "session_history.jsonl").write_text(
            json.dumps({"session_index": 0, "generated_count": 16}) + "\n",
            encoding="utf-8",
        )

    preparation = prepare_campaign_semigroup_visual_review(
        config_path=config_path,
        shard_root=shard_root,
        formal_campaign_id=formal_id,
        repo_root=repo_root,
    )
    evidence_path = campaign_root / "preflight" / "evidence_manifest.json"
    assignment_path = campaign_root / "preflight" / "visual_review_assignment.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    review_path = campaign_root / "preflight" / "visual_review.json"
    review = {
        "schema_version": 2,
        "contract_type": "safa_r9_semigroup_visual_review_v2",
        "bootstrap_campaign_id": bootstrap_id,
        "formal_campaign_id": formal_id,
        "review_type": "independent_blinded_semigroup_structure_review",
        "decision_rule": "passed_if_and_only_if_severe_count_equals_zero",
        "sample_count": 64,
        "reviewed_sample_ids": sample_ids,
        "evidence_manifest_sha256": evidence["evidence_manifest_sha256"],
        "visual_review_assignment_sha256": assignment[
            "visual_review_assignment_sha256"
        ],
        "conditions": {
            condition["condition_id"]: {
                "passed": True,
                "severe_count": 0,
                "severe_sample_ids": [],
            }
            for condition in assignment["conditions"]
        },
    }
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(review_path, review)
    contact_sheet = repo_root / assignment["conditions"][0]["pages"][0]["path"]
    return {
        "repo_root": repo_root,
        "config_path": config_path,
        "shard_root": shard_root,
        "output_root": output_root,
        "review_path": review_path,
        "contact_sheet": contact_sheet,
        "bootstrap_id": bootstrap_id,
        "formal_id": formal_id,
        "sample_ids": sample_ids,
        "evidence": evidence,
        "assignment": assignment,
        "assignment_path": assignment_path,
        "blinding_map_path": (
            campaign_root / "preflight" / "visual_review_blinding_map.json"
        ),
        "preparation": preparation,
    }


def _finalize(paths: dict[str, Any]) -> dict[str, Any]:
    return finalize_campaign_semigroup_closure(
        config_path=paths["config_path"],
        shard_root=paths["shard_root"],
        output_root=paths["output_root"],
        visual_review_path=paths["review_path"],
        repo_root=paths["repo_root"],
    )


def _rewrite_assignment_and_review(
    paths: dict[str, Any], assignment: dict[str, Any]
) -> None:
    assignment["visual_review_assignment_sha256"] = _canonical_digest(
        assignment, "visual_review_assignment_sha256"
    )
    paths["assignment_path"].chmod(0o644)
    _write_json(paths["assignment_path"], assignment)
    review = json.loads(paths["review_path"].read_text(encoding="utf-8"))
    review["visual_review_assignment_sha256"] = assignment[
        "visual_review_assignment_sha256"
    ]
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(paths["review_path"], review)


def _set_all_review_decisions(paths: dict[str, Any], *, passed: bool) -> None:
    review = json.loads(paths["review_path"].read_text(encoding="utf-8"))
    severe_ids = [] if passed else [paths["sample_ids"][0]]
    for condition_id in review["conditions"]:
        review["conditions"][condition_id] = {
            "passed": passed,
            "severe_count": len(severe_ids),
            "severe_sample_ids": list(severe_ids),
        }
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(paths["review_path"], review)


def test_prepare_review_writes_blinded_assignment_without_decisions(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    assignment = paths["assignment"]
    preparation = paths["preparation"]
    assert preparation["bootstrap_campaign_id"] == paths["bootstrap_id"]
    assert preparation["formal_campaign_id"] == paths["formal_id"]
    assert preparation["review_decisions_materialized"] is False
    assert preparation["condition_count"] == 3
    assert preparation["contact_sheet_count"] == 24
    assert assignment["reviewed_sample_ids"] == paths["sample_ids"]
    assert assignment["registered_splits"] == ["0.25", "0.5", "0.75"]
    assert len(assignment["conditions"]) == 3
    for condition in assignment["conditions"]:
        assert condition["condition_id"].startswith("condition_")
        assert len(condition["column_ids"]) == 3
        assert all(value.startswith("column_") for value in condition["column_ids"])
        assert len(condition["pages"]) == 8
        assert all(
            len(page["file_sha256"]) == 64 and len(page["sheet_contract_sha256"]) == 64
            for page in condition["pages"]
        )
        assert all(
            "t_cut" not in page["path"]
            and "0p25" not in page["path"]
            and "0p5" not in page["path"]
            and "0p75" not in page["path"]
            for page in condition["pages"]
        )
    serialized = json.dumps(assignment, sort_keys=True)
    assert '"passed"' not in serialized
    assert '"severe_count"' not in serialized
    assert '"severe_sample_ids"' not in serialized
    assert '"source"' not in serialized
    assert '"native"' not in serialized
    assert '"candidate"' not in serialized
    assert '"generated_direct"' not in serialized
    assert paths["blinding_map_path"].stat().st_mode & 0o777 == 0
    with Image.open(paths["contact_sheet"]) as sheet:
        assert sheet.size == (384, 1052)


def test_prepare_review_is_exactly_once_and_binds_formal_campaign(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(CampaignSemigroupClosureError, match="already exists"):
        prepare_campaign_semigroup_visual_review(
            config_path=paths["config_path"],
            shard_root=paths["shard_root"],
            formal_campaign_id=paths["formal_id"],
            repo_root=paths["repo_root"],
        )


def test_assignment_rejects_reviewer_decision_fields(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    assignment = json.loads(paths["assignment_path"].read_text(encoding="utf-8"))
    assignment["passed"] = True
    _rewrite_assignment_and_review(paths, assignment)
    with pytest.raises(CampaignSemigroupClosureError, match="decision fields"):
        _finalize(paths)
    assert paths["blinding_map_path"].stat().st_mode & 0o777 == 0


def test_assignment_rejects_split_or_ordered_id_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    assignment = json.loads(paths["assignment_path"].read_text(encoding="utf-8"))
    assignment["registered_splits"] = ["0.25", "0.75", "0.5"]
    assignment["reviewed_sample_ids"][0:2] = assignment["reviewed_sample_ids"][0:2][
        ::-1
    ]
    _rewrite_assignment_and_review(paths, assignment)
    with pytest.raises(CampaignSemigroupClosureError, match="ID/split"):
        _finalize(paths)


def test_incomplete_review_does_not_reveal_blinding_map(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    review = json.loads(paths["review_path"].read_text(encoding="utf-8"))
    review["conditions"].pop(next(iter(review["conditions"])))
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(paths["review_path"], review)
    with pytest.raises(CampaignSemigroupClosureError, match="three blinded"):
        _finalize(paths)
    assert paths["blinding_map_path"].stat().st_mode & 0o777 == 0


def test_completed_review_rejects_blinding_map_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    map_path = paths["blinding_map_path"]
    map_path.chmod(0o600)
    blinding_map = json.loads(map_path.read_text(encoding="utf-8"))
    condition_ids = list(blinding_map["conditions"])
    blinding_map["conditions"][condition_ids[1]]["split"] = blinding_map["conditions"][
        condition_ids[0]
    ]["split"]
    blinding_map["blinding_map_sha256"] = _canonical_digest(
        blinding_map, "blinding_map_sha256"
    )
    _write_json(map_path, blinding_map)
    map_file_sha256 = _sha(map_path)
    map_path.chmod(0)
    assignment = json.loads(paths["assignment_path"].read_text(encoding="utf-8"))
    assignment["blinding_map"] = {
        **assignment["blinding_map"],
        "file_sha256": map_file_sha256,
        "contract_sha256": blinding_map["blinding_map_sha256"],
    }
    _rewrite_assignment_and_review(paths, assignment)
    with pytest.raises(CampaignSemigroupClosureError, match="split/column"):
        _finalize(paths)
    assert map_path.stat().st_mode & 0o777 == 0o400


def test_all_failed_review_writes_terminal_contract_without_loading_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _set_all_review_decisions(paths, passed=False)
    assert paths["blinding_map_path"].stat().st_mode & 0o777 == 0

    def forbidden_map_load(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("all-failed path must not load the map")

    monkeypatch.setattr(
        closure_module, "_load_blinding_map_after_complete_review", forbidden_map_load
    )
    result = _finalize(paths)
    failure_path = paths["output_root"] / "closure_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))

    assert result["gate_passed"] is False
    assert result["terminal_failure"] is True
    assert failure["terminal_path_read_map"] is False
    assert failure["failure_contract_sha256"] == _canonical_digest(
        failure, "failure_contract_sha256"
    )
    assert {path.name for path in paths["output_root"].iterdir()} == {
        "closure_failure.json"
    }
    assert failure_path.stat().st_mode & 0o222 == 0
    assert paths["blinding_map_path"].stat().st_mode & 0o777 == 0
    assert not (paths["output_root"] / "closure_seal.json").exists()
    with pytest.raises(CampaignSemigroupClosureError, match="terminal.*failure"):
        resolve_formal_campaign_semigroup_closure(
            paths["formal_id"], repo_root=paths["repo_root"]
        )


@pytest.mark.parametrize("partial_review", [False, True])
def test_numeric_all_fail_writes_terminal_without_loading_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_review: bool,
) -> None:
    paths = _fixture(tmp_path, latent_residual=0.5)
    if partial_review:
        review = json.loads(paths["review_path"].read_text(encoding="utf-8"))
        condition_id = next(iter(review["conditions"]))
        review["conditions"][condition_id] = {
            "passed": False,
            "severe_count": 1,
            "severe_sample_ids": [paths["sample_ids"][0]],
        }
        review["visual_review_sha256"] = _canonical_digest(
            review, "visual_review_sha256"
        )
        _write_json(paths["review_path"], review)

    def forbidden_map_load(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("numeric-all-fail path must not load the map")

    monkeypatch.setattr(
        closure_module, "_load_blinding_map_after_complete_review", forbidden_map_load
    )
    result = _finalize(paths)
    failure_path = paths["output_root"] / "closure_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))

    assert result["schema_version"] == 2
    assert result["terminal_failure"] is True
    assert result["gate_passed"] is False
    assert failure["failure_reason"] == "no_quantitative_candidate"
    assert failure["terminal_path_read_map"] is False
    assert failure["numeric_precheck"]["gate_passed"] is False
    assert all(
        candidate["visual_pass"] is True and candidate["passed"] is False
        for candidate in failure["numeric_precheck"]["candidates"]
    )
    assert paths["blinding_map_path"].stat().st_mode & 0o777 == 0
    assert {path.name for path in paths["output_root"].iterdir()} == {
        "closure_failure.json"
    }
    assert not (paths["output_root"] / "closure_seal.json").exists()
    with pytest.raises(CampaignSemigroupClosureError, match="terminal.*failure"):
        resolve_formal_campaign_semigroup_closure(
            paths["formal_id"], repo_root=paths["repo_root"]
        )


def test_source_review_tamper_invalidates_published_closure(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _finalize(paths)
    paths["review_path"].write_text(
        paths["review_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CampaignSemigroupClosureError, match="source visual review"):
        resolve_formal_campaign_semigroup_closure(
            paths["formal_id"], repo_root=paths["repo_root"]
        )


def test_resolver_shared_map_validator_rejects_fully_rebound_tamper(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _finalize(paths)
    output = paths["output_root"]
    source_map = paths["blinding_map_path"]
    published_map = output / "visual_review_blinding_map.json"
    source_map.chmod(0o600)
    tampered_map = json.loads(source_map.read_text(encoding="utf-8"))
    first = next(iter(tampered_map["conditions"].values()))
    first["columns"][1]["role"] = first["columns"][0]["role"]
    tampered_map["blinding_context_sha256"] = "0" * 64
    tampered_map["blinding_map_sha256"] = _canonical_digest(
        tampered_map, "blinding_map_sha256"
    )
    for path in (source_map, published_map):
        path.chmod(0o644)
        _write_json(path, tampered_map)

    source_assignment = paths["assignment_path"]
    published_assignment = output / "visual_review_assignment.json"
    source_assignment.chmod(0o644)
    assignment = json.loads(source_assignment.read_text(encoding="utf-8"))
    assignment["blinding_map"]["file_sha256"] = _sha(source_map)
    assignment["blinding_map"]["contract_sha256"] = tampered_map["blinding_map_sha256"]
    assignment["visual_review_assignment_sha256"] = _canonical_digest(
        assignment, "visual_review_assignment_sha256"
    )
    for path in (source_assignment, published_assignment):
        path.chmod(0o644)
        _write_json(path, assignment)

    source_review = paths["review_path"]
    published_review = output / "visual_review.json"
    source_review.chmod(0o644)
    review = json.loads(source_review.read_text(encoding="utf-8"))
    review["visual_review_assignment_sha256"] = assignment[
        "visual_review_assignment_sha256"
    ]
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    for path in (source_review, published_review):
        path.chmod(0o644)
        _write_json(path, review)

    seal_path = output / "closure_seal.json"
    seal_path.chmod(0o644)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for artifact, path in {
        "visual_review_assignment": published_assignment,
        "visual_review_blinding_map": published_map,
        "visual_review": published_review,
    }.items():
        seal["artifacts"][artifact]["sha256"] = _sha(path)
    seal["bindings"]["visual_review_assignment_sha256"] = assignment[
        "visual_review_assignment_sha256"
    ]
    seal["bindings"]["visual_review_blinding_map_sha256"] = tampered_map[
        "blinding_map_sha256"
    ]
    seal["bindings"]["visual_review_sha256"] = review["visual_review_sha256"]
    seal["bindings"]["visual_review_source_sha256"] = _sha(source_review)
    seal["bindings"]["visual_review_published_copy_sha256"] = _sha(published_review)
    seal["closure_seal_sha256"] = _canonical_digest(seal, "closure_seal_sha256")
    _write_json(seal_path, seal)

    with pytest.raises(CampaignSemigroupClosureError, match="blinding map"):
        resolve_formal_campaign_semigroup_closure(
            paths["formal_id"], repo_root=paths["repo_root"]
        )


def test_campaign_closure_writes_new_immutable_chain_and_campaign_relation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = _finalize(paths)

    assert result["gate_passed"] is True
    assert result["selected_t_cut"] == 0.25
    expected = {
        "preflight_contract.json",
        "effective_config.json",
        "executed_config.json",
        "evidence_manifest.json",
        "visual_review_assignment.json",
        "visual_review_blinding_map.json",
        "visual_review.json",
        "semigroup_report.json",
        "gate_contract.json",
        "locked_schedule_manifest.json",
        "closure_seal.json",
    }
    assert {path.name for path in paths["output_root"].iterdir()} == expected
    assert all(
        path.stat().st_mode & 0o222 == 0 for path in paths["output_root"].iterdir()
    )
    seal = json.loads((paths["output_root"] / "closure_seal.json").read_text())
    assert seal["bootstrap_campaign"]["campaign_id"] == paths["bootstrap_id"]
    assert seal["formal_campaign"]["campaign_id"] == paths["formal_id"]
    assert seal["formal_campaign"]["relationship"] == (
        "bootstrap_preflight_for_distinct_formal_campaign"
    )
    assert (
        seal["bindings"]["evidence_manifest_sha256"]
        == paths["evidence"]["evidence_manifest_sha256"]
    )
    assert (
        seal["bindings"]["visual_review_assignment_sha256"]
        == paths["assignment"]["visual_review_assignment_sha256"]
    )
    assert (
        seal["bindings"]["visual_review_sha256"]
        == json.loads(paths["review_path"].read_text())["visual_review_sha256"]
    )
    assert seal["bindings"]["visual_review_source_path"] == _repo_relative(
        paths["repo_root"], paths["review_path"]
    )
    assert seal["bindings"]["visual_review_source_sha256"] == _sha(paths["review_path"])
    assert seal["bindings"]["visual_review_published_copy_sha256"] == _sha(
        paths["output_root"] / "visual_review.json"
    )
    schedule = json.loads(
        (paths["output_root"] / "locked_schedule_manifest.json").read_text()
    )
    assert schedule["gate_passed"] is True
    assert schedule["t_cut"] == 0.25
    resolved = resolve_formal_campaign_semigroup_closure(
        paths["formal_id"], repo_root=paths["repo_root"]
    )
    assert resolved is not None
    assert resolved["bootstrap_campaign_id"] == paths["bootstrap_id"]
    assert resolved["formal_campaign_id"] == paths["formal_id"]
    assert resolved["closure"]["path"].endswith("/closure_seal.json")
    assert resolved["schedule"]["path"].endswith("/locked_schedule_manifest.json")
    assert resolved["gate"]["path"].endswith("/gate_contract.json")


def test_formal_closure_resolver_never_accepts_tampered_seal(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _finalize(paths)
    seal_path = paths["output_root"] / "closure_seal.json"
    seal_path.chmod(0o644)
    seal = json.loads(seal_path.read_text())
    seal["formal_campaign"]["relationship"] = "legacy_global_preflight"
    _write_json(seal_path, seal)

    with pytest.raises(CampaignSemigroupClosureError, match="closure seal"):
        resolve_formal_campaign_semigroup_closure(
            paths["formal_id"], repo_root=paths["repo_root"]
        )


def test_visual_review_must_bind_exact_complete_evidence(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    review = json.loads(paths["review_path"].read_text())
    review["evidence_manifest_sha256"] = "0" * 64
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(paths["review_path"], review)

    with pytest.raises(CampaignSemigroupClosureError, match="evidence"):
        _finalize(paths)
    assert not paths["output_root"].exists()


def test_any_reviewed_image_tamper_invalidates_visual_review(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    split = next(
        (paths["shard_root"] / "shard_0" / "semigroup_split_images").rglob("*.png")
    )
    split.write_bytes(b"tampered-after-review")

    with pytest.raises(CampaignSemigroupClosureError, match="evidence"):
        _finalize(paths)
    assert not paths["output_root"].exists()


def test_contact_sheet_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["contact_sheet"].chmod(0o644)
    paths["contact_sheet"].write_bytes(b"tampered-sheet")

    with pytest.raises(CampaignSemigroupClosureError, match="contact sheet"):
        _finalize(paths)


def test_bootstrap_campaign_runtime_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    runtime_path = (
        paths["repo_root"]
        / "artifacts"
        / "r9_meanflow_flow_map_guidance"
        / "campaigns"
        / paths["bootstrap_id"]
        / "campaign_runtime.json"
    )
    runtime = json.loads(runtime_path.read_text())
    runtime["attention_backend"] = "auto"
    _write_json(runtime_path, runtime)

    with pytest.raises(CampaignSemigroupClosureError, match="runtime digest"):
        _finalize(paths)


def test_checkpoint_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = yaml.safe_load(paths["config_path"].read_text())
    checkpoint = paths["repo_root"] / str(config["checkpoint"])
    checkpoint.write_bytes(b"tampered-checkpoint")

    with pytest.raises(CampaignSemigroupClosureError, match="checkpoint SHA256"):
        _finalize(paths)


def test_driver_verified_completion_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    verified_path = paths["shard_root"] / "shard_1" / "verified_completion.json"
    verified = json.loads(verified_path.read_text())
    verified["generation_result_sha256"] = "0" * 64
    _write_json(verified_path, verified)

    with pytest.raises(CampaignSemigroupClosureError, match="verified completion"):
        _finalize(paths)


def test_unregistered_evidence_file_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    extra = paths["shard_root"] / "shard_0" / "generated_images" / "extra.png"
    extra.write_bytes(b"unregistered")

    with pytest.raises(CampaignSemigroupClosureError, match="inventory"):
        _finalize(paths)


def test_source_evidence_must_match_locked_index(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    shard = paths["shard_root"] / "shard_0"
    per_sample_path = shard / "per_sample.jsonl"
    manifest_path = shard / "sample_id_manifest.jsonl"
    per_sample = [json.loads(line) for line in per_sample_path.read_text().splitlines()]
    shard_manifest = [
        json.loads(line) for line in manifest_path.read_text().splitlines()
    ]
    wrong_source = per_sample[1]["source"]
    per_sample[0]["source"] = wrong_source
    shard_manifest[0]["source"] = wrong_source
    per_sample_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_sample),
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in shard_manifest),
        encoding="utf-8",
    )

    with pytest.raises(CampaignSemigroupClosureError, match="source index binding"):
        _finalize(paths)


def test_all_shards_must_share_exact_executed_config(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    generation_path = paths["shard_root"] / "shard_2" / "generation_result.json"
    generation = json.loads(generation_path.read_text())
    generation["config"]["num_workers"] = 99
    _write_json(generation_path, generation)
    _write_json(generation_path.with_name("run_manifest.json"), generation)
    verified_path = generation_path.with_name("verified_completion.json")
    verified = json.loads(verified_path.read_text())
    verified["generation_result_sha256"] = _sha(generation_path)
    verified["run_manifest_sha256"] = _sha(
        generation_path.with_name("run_manifest.json")
    )
    verified["verified_completion_sha256"] = _canonical_digest(
        verified, "verified_completion_sha256"
    )
    _write_json(verified_path, verified)

    with pytest.raises(CampaignSemigroupClosureError, match="executed config"):
        _finalize(paths)


def test_incomplete_or_noncanonical_review_cannot_claim_pass(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    review = json.loads(paths["review_path"].read_text())
    condition_id = next(iter(review["conditions"]))
    review["conditions"][condition_id] = {
        "passed": True,
        "severe_count": 1,
        "severe_sample_ids": [paths["sample_ids"][0]],
    }
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(paths["review_path"], review)

    with pytest.raises(CampaignSemigroupClosureError, match="passed"):
        _finalize(paths)


def test_output_must_be_new_direct_campaign_closure_root(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    map_path = paths["blinding_map_path"]
    paths["output_root"].mkdir(parents=True)
    (paths["output_root"] / "foreign.txt").write_text("do not replace")

    with pytest.raises(CampaignSemigroupClosureError, match="already exists"):
        _finalize(paths)
    assert (paths["output_root"] / "foreign.txt").read_text() == "do not replace"
    assert map_path.stat().st_mode & 0o777 == 0

    paths = _fixture(tmp_path / "legacy-output-case")
    legacy = (
        paths["repo_root"]
        / "artifacts"
        / "r9_meanflow_flow_map_guidance"
        / "semigroup"
        / "new"
    )
    paths["output_root"] = legacy
    with pytest.raises(CampaignSemigroupClosureError, match="output root"):
        _finalize(paths)
    assert paths["blinding_map_path"].stat().st_mode & 0o777 == 0


def test_shared_map_validator_rejects_role_and_context_tamper(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    map_path = paths["blinding_map_path"]
    map_path.chmod(0o600)
    blinding_map = json.loads(map_path.read_text(encoding="utf-8"))
    first = next(iter(blinding_map["conditions"].values()))
    first["columns"][1]["role"] = first["columns"][0]["role"]
    blinding_map["blinding_context_sha256"] = "0" * 64
    blinding_map["blinding_map_sha256"] = _canonical_digest(
        blinding_map, "blinding_map_sha256"
    )
    _write_json(map_path, blinding_map)
    map_file_sha256 = _sha(map_path)
    map_path.chmod(0)
    assignment = json.loads(paths["assignment_path"].read_text(encoding="utf-8"))
    assignment["blinding_map"]["file_sha256"] = map_file_sha256
    assignment["blinding_map"]["contract_sha256"] = blinding_map["blinding_map_sha256"]
    _rewrite_assignment_and_review(paths, assignment)

    with pytest.raises(CampaignSemigroupClosureError, match="blinding map"):
        _finalize(paths)


def test_bootstrap_and_formal_campaign_ids_must_be_distinct(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    review = json.loads(paths["review_path"].read_text())
    review["formal_campaign_id"] = paths["bootstrap_id"]
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(paths["review_path"], review)

    with pytest.raises(CampaignSemigroupClosureError, match="distinct"):
        _finalize(paths)


def test_formal_campaign_must_not_precede_bootstrap_seal(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    formal_root = (
        paths["repo_root"]
        / "artifacts"
        / "r9_meanflow_flow_map_guidance"
        / "campaigns"
        / paths["formal_id"]
    )
    formal_root.mkdir(parents=True)

    with pytest.raises(CampaignSemigroupClosureError, match="not be materialized"):
        _finalize(paths)


def test_cli_requires_all_four_explicit_paths() -> None:
    spec = importlib.util.spec_from_file_location("r9_semigroup_closure_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for argv in (
        [],
        ["--config", "runtime.yaml"],
        ["--config", "runtime.yaml", "--shard-root", "shards"],
        [
            "--config",
            "runtime.yaml",
            "--shard-root",
            "shards",
            "--output-root",
            "closure",
        ],
    ):
        with pytest.raises(SystemExit):
            module.parse_args(argv)


def test_prepare_cli_requires_config_shards_and_formal_campaign() -> None:
    spec = importlib.util.spec_from_file_location(
        "r9_semigroup_review_prepare_cli", PREPARE_CLI
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for argv in (
        [],
        ["--config", "runtime.yaml"],
        ["--config", "runtime.yaml", "--shard-root", "shards"],
    ):
        with pytest.raises(SystemExit):
            module.parse_args(argv)
    parsed = module.parse_args(
        [
            "--config",
            "runtime.yaml",
            "--shard-root",
            "shards",
            "--formal-campaign-id",
            "formal-r9-v2",
        ]
    )
    assert parsed.formal_campaign_id == "formal-r9-v2"


def _policy_recovery_fixture(
    tmp_path: Path, *, latent_residual: float = 0.5, severe_count: int = 1
) -> dict[str, Any]:
    paths = _fixture(tmp_path, latent_residual=latent_residual)
    review = json.loads(paths["review_path"].read_text(encoding="utf-8"))
    severe_ids = paths["sample_ids"][:severe_count]
    for condition_id in review["conditions"]:
        review["conditions"][condition_id] = {
            "passed": severe_count == 0,
            "severe_count": severe_count,
            "severe_sample_ids": list(severe_ids),
        }
    review["visual_review_sha256"] = _canonical_digest(review, "visual_review_sha256")
    _write_json(paths["review_path"], review)
    terminal = _finalize(paths)
    assert terminal["terminal_failure"] is True
    source_failure = paths["output_root"] / "closure_failure.json"
    paths["blinding_map_path"].chmod(0o400)
    policy_id = "policy-preflight-r9-v2"
    formal_id = "formal-r9-policy-v2"
    policy_output = (
        paths["repo_root"]
        / "artifacts"
        / "r9_meanflow_flow_map_guidance"
        / "semigroup_campaign_closures"
        / f"{policy_id}__for__{formal_id}"
    )
    return {
        **paths,
        "source_failure": source_failure,
        "source_formal_id": paths["formal_id"],
        "policy_id": policy_id,
        "policy_formal_id": formal_id,
        "policy_output": policy_output,
    }


def _recover(paths: dict[str, Any]) -> dict[str, Any]:
    return finalize_campaign_semigroup_policy_recovery(
        config_path=paths["config_path"],
        shard_root=paths["shard_root"],
        policy_campaign_id=paths["policy_id"],
        formal_campaign_id=paths["policy_formal_id"],
        output_root=paths["policy_output"],
        visual_review_path=paths["review_path"],
        source_terminal_failure_path=paths["source_failure"],
        user_recovery_authorization_id=R9_SEMIGROUP_RECOVERY_AUTHORIZATION_ID,
        repo_root=paths["repo_root"],
    )


def test_policy_recovery_treats_numeric_thresholds_as_report_only_and_locks_025(
    tmp_path: Path,
) -> None:
    paths = _policy_recovery_fixture(tmp_path, latent_residual=0.5, severe_count=1)

    result = _recover(paths)

    assert result["gate_passed"] is True
    assert result["selected_t_cut"] == 0.25
    report = json.loads((paths["policy_output"] / "semigroup_report.json").read_text())
    assert report["numerical_metrics_role"] == "report_only"
    assert report["selected_t_cut"] == 0.25
    assert all(
        candidate["numeric_threshold_pass"] is False
        for candidate in report["candidates"]
    )
    assert all(candidate["passed"] is True for candidate in report["candidates"])
    assert all(row["severe_count"] == 1 for row in report["visual_assessment"].values())
    assert report["policy_version"] == R9_SEMIGROUP_RECOVERY_POLICY_VERSION
    assert report["policy_sha256"] == R9_SEMIGROUP_RECOVERY_POLICY_SHA256


def test_policy_recovery_visual_limit_is_one_per_split(tmp_path: Path) -> None:
    passing = _policy_recovery_fixture(tmp_path / "one", severe_count=1)
    assert _recover(passing)["gate_passed"] is True

    failing = _policy_recovery_fixture(tmp_path / "two", severe_count=2)
    with pytest.raises(CampaignSemigroupClosureError, match="visual severe limit"):
        _recover(failing)
    assert not failing["policy_output"].exists()


@pytest.mark.parametrize(
    ("metric", "value"),
    (("latent_residual", float("nan")), ("endpoint_e0_cosine", float("inf"))),
)
def test_policy_recovery_still_hard_fails_nonfinite_metrics(
    tmp_path: Path, metric: str, value: float
) -> None:
    paths = _policy_recovery_fixture(tmp_path)
    semigroup_path = paths["shard_root"] / "shard_0" / "semigroup.json"
    payload = json.loads(semigroup_path.read_text(encoding="utf-8"))
    payload["rows"][0]["splits"]["0.25"][metric] = value
    semigroup_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True),
        encoding="utf-8",
    )

    with pytest.raises(CampaignSemigroupClosureError, match="non-finite"):
        _recover(paths)
    assert not paths["policy_output"].exists()


def test_policy_recovery_still_hard_fails_contract_tamper(tmp_path: Path) -> None:
    paths = _policy_recovery_fixture(tmp_path)
    generation_path = paths["shard_root"] / "shard_0" / "generation_result.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["checkpoint"]["sha256"] = "0" * 64
    _write_json(generation_path, generation)
    _write_json(generation_path.with_name("run_manifest.json"), generation)

    with pytest.raises(CampaignSemigroupClosureError, match="checkpoint"):
        _recover(paths)
    assert not paths["policy_output"].exists()


def test_policy_recovery_seal_and_resolver_bind_exact_policy(tmp_path: Path) -> None:
    paths = _policy_recovery_fixture(tmp_path)
    _recover(paths)
    seal_path = paths["policy_output"] / "closure_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["schema_version"] == 2
    assert seal["policy_campaign"]["campaign_id"] == paths["policy_id"]
    assert seal["source_review"]["state_at_recovery"] == "previously_revealed"
    assert seal["source_review"]["formal_campaign_id"] == paths["source_formal_id"]
    assert seal["policy"]["policy_sha256"] == R9_SEMIGROUP_RECOVERY_POLICY_SHA256
    assert seal["policy"]["numerical_metrics_role"] == "report_only"
    assert seal["policy"]["visual_severe_limit_per_split"] == 1
    assert seal["policy"]["selected_t_cut"] == 0.25
    assert (
        seal["authorization"]["authorization_id"]
        == R9_SEMIGROUP_RECOVERY_AUTHORIZATION_ID
    )
    report = json.loads((paths["policy_output"] / "semigroup_report.json").read_text())
    gate = json.loads((paths["policy_output"] / "gate_contract.json").read_text())
    schedule = json.loads(
        (paths["policy_output"] / "locked_schedule_manifest.json").read_text()
    )
    assert report["selection_rule"] == R9_SEMIGROUP_RECOVERY_SELECTION_RULE
    assert gate["selection_rule"] == R9_SEMIGROUP_RECOVERY_SELECTION_RULE
    assert schedule["selection_rule"] == R9_SEMIGROUP_RECOVERY_SELECTION_RULE
    assert gate["recovery_policy_sha256"] == R9_SEMIGROUP_RECOVERY_POLICY_SHA256
    assert schedule["recovery_policy_sha256"] == R9_SEMIGROUP_RECOVERY_POLICY_SHA256

    resolved = resolve_formal_campaign_semigroup_closure(
        paths["policy_formal_id"], repo_root=paths["repo_root"]
    )
    assert resolved is not None
    assert resolved["policy_campaign_id"] == paths["policy_id"]
    assert resolved["bootstrap_campaign_id"] == paths["bootstrap_id"]
    assert resolved["policy_sha256"] == R9_SEMIGROUP_RECOVERY_POLICY_SHA256

    seal_path.chmod(0o644)
    seal["policy"]["visual_severe_limit_per_split"] = 2
    seal["closure_seal_sha256"] = _canonical_digest(seal, "closure_seal_sha256")
    _write_json(seal_path, seal)
    with pytest.raises(CampaignSemigroupClosureError, match="policy"):
        resolve_formal_campaign_semigroup_closure(
            paths["policy_formal_id"], repo_root=paths["repo_root"]
        )


def test_policy_recovery_resolver_rejects_coherently_rehashed_old_selection_rule(
    tmp_path: Path,
) -> None:
    paths = _policy_recovery_fixture(tmp_path)
    _recover(paths)
    root = paths["policy_output"]
    report_path = root / "semigroup_report.json"
    gate_path = root / "gate_contract.json"
    schedule_path = root / "locked_schedule_manifest.json"
    seal_path = root / "closure_seal.json"
    for path in (report_path, gate_path, schedule_path, seal_path):
        path.chmod(0o644)

    report = json.loads(report_path.read_text())
    report["selection_rule"] = (
        "smallest_numeric_t_cut_passing_all_registered_thresholds"
    )
    _write_json(report_path, report)

    schedule = json.loads(schedule_path.read_text())
    schedule["selection_rule"] = report["selection_rule"]
    schedule["schedule_contract_sha256"] = (
        closure_module.canonical_r9_schedule_contract_sha256(schedule)
    )

    gate = json.loads(gate_path.read_text())
    gate["selection_rule"] = report["selection_rule"]
    gate["semigroup_report_sha256"] = _sha(report_path)
    gate["schedule_contract_sha256"] = schedule["schedule_contract_sha256"]
    gate["gate_contract_sha256"] = _canonical_digest(gate, "gate_contract_sha256")
    _write_json(gate_path, gate)
    schedule["r9_semigroup_gate_contract_sha256"] = _sha(gate_path)
    _write_json(schedule_path, schedule)

    seal = json.loads(seal_path.read_text())
    for name, path in (
        ("semigroup_report", report_path),
        ("gate_contract", gate_path),
        ("locked_schedule_manifest", schedule_path),
    ):
        seal["artifacts"][name]["sha256"] = _sha(path)
    seal["bindings"].update(
        {
            "semigroup_report_sha256": _sha(report_path),
            "gate_contract_sha256": gate["gate_contract_sha256"],
            "gate_contract_file_sha256": _sha(gate_path),
            "schedule_contract_sha256": schedule["schedule_contract_sha256"],
            "schedule_manifest_file_sha256": _sha(schedule_path),
        }
    )
    seal["closure_seal_sha256"] = _canonical_digest(seal, "closure_seal_sha256")
    _write_json(seal_path, seal)

    with pytest.raises(CampaignSemigroupClosureError, match="policy semantics"):
        resolve_formal_campaign_semigroup_closure(
            paths["policy_formal_id"], repo_root=paths["repo_root"]
        )


@pytest.mark.parametrize(
    "tamper_kind",
    ("wrong_severe_ids", "boolean_severe_count", "negative_severe_count"),
)
def test_policy_recovery_resolver_rederives_visual_assessment_from_review(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    paths = _policy_recovery_fixture(tmp_path)
    _recover(paths)
    root = paths["policy_output"]
    report_path = root / "semigroup_report.json"
    gate_path = root / "gate_contract.json"
    schedule_path = root / "locked_schedule_manifest.json"
    seal_path = root / "closure_seal.json"
    for path in (report_path, gate_path, schedule_path, seal_path):
        path.chmod(0o644)

    report = json.loads(report_path.read_text())
    assessment = report["visual_assessment"]["0.25"]
    if tamper_kind == "wrong_severe_ids":
        assessment["severe_count"] = 0
        assessment["severe_sample_ids"] = []
    elif tamper_kind == "boolean_severe_count":
        assessment["severe_count"] = True
    elif tamper_kind == "negative_severe_count":
        assessment["severe_count"] = -1
        assessment["severe_sample_ids"] = []
    else:
        raise AssertionError(f"unregistered tamper kind: {tamper_kind}")
    _write_json(report_path, report)

    report_sha256 = _sha(report_path)
    schedule = json.loads(schedule_path.read_text())
    schedule["semigroup_report_sha256"] = report_sha256
    schedule["schedule_contract_sha256"] = (
        closure_module.canonical_r9_schedule_contract_sha256(schedule)
    )

    gate = json.loads(gate_path.read_text())
    gate["semigroup_report_sha256"] = report_sha256
    gate["schedule_contract_sha256"] = schedule["schedule_contract_sha256"]
    gate["gate_contract_sha256"] = _canonical_digest(gate, "gate_contract_sha256")
    _write_json(gate_path, gate)
    schedule["r9_semigroup_gate_contract_sha256"] = _sha(gate_path)
    _write_json(schedule_path, schedule)

    seal = json.loads(seal_path.read_text())
    for name, path in (
        ("semigroup_report", report_path),
        ("gate_contract", gate_path),
        ("locked_schedule_manifest", schedule_path),
    ):
        seal["artifacts"][name]["sha256"] = _sha(path)
    seal["bindings"].update(
        {
            "semigroup_report_sha256": report_sha256,
            "gate_contract_sha256": gate["gate_contract_sha256"],
            "gate_contract_file_sha256": _sha(gate_path),
            "schedule_contract_sha256": schedule["schedule_contract_sha256"],
            "schedule_manifest_file_sha256": _sha(schedule_path),
        }
    )
    seal["closure_seal_sha256"] = _canonical_digest(seal, "closure_seal_sha256")
    _write_json(seal_path, seal)

    with pytest.raises(CampaignSemigroupClosureError, match="visual assessment"):
        resolve_formal_campaign_semigroup_closure(
            paths["policy_formal_id"], repo_root=paths["repo_root"]
        )
