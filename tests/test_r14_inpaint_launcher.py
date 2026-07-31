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


validator = load_script("validate_r14_inpaint_feasibility.py")
closeout = load_script("close_r14_inpaint_feasibility.py")
smoke = load_script("run_r14_inpaint_smoke.py")


def point68() -> list[list[float]]:
    return [[float(index), float(index + 1)] for index in range(68)]


def eval_row(index: int) -> dict:
    return {
        "contract_version": "safa_r14_spatial_eval_v1",
        "sample_id": f"sample-{index}",
        "image_path": f"/data/{index}.jpg",
        "affect_label": index % 8,
        "bbox_xywh": [1, 2, 30, 40],
        "landmarks68": point68(),
    }


def pair_row(index: int) -> dict:
    return {
        "contract_version": "safa_r14_spatial_pair_v1",
        "pair_id": f"pair-{index}",
        "source_sample_id": f"source-{index}",
        "target_sample_id": f"target-{index}",
        "affect_label": index % 8,
        "source_image_path": f"/data/source-{index}.jpg",
        "target_image_path": f"/data/target-{index}.jpg",
        "source_bbox_xywh": [1, 2, 30, 40],
        "target_bbox_xywh": [3, 4, 31, 41],
        "source_landmarks68": point68(),
        "target_landmarks68": point68(),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_locked_config_validates() -> None:
    path = REPO / "configs/medium_v2/experiments/r14_inpaint_feasibility_256step.yaml"
    validator._validate_config(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_train_g_config(config)
    assert config["amp"] is False
    assert config["global_batch_size"] == 8
    assert config["per_device_batch_size"] == 2
    assert config["stages"]["stage2"]["epochs"] == 2
    assert validator._optimizer_steps_from_geometry(1024, 8, 2) == 256
    assert config["generator"]["inpaint"]["conditioning"] == "cached_source_z_only"
    assert config["eval_index"] == "data/index/val_face_mixed_e14.jsonl"
    assert config["eval_features"] == "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"
    for entrypoint in (
        "run_r14_inpaint_smoke.py",
        "run_r14_inpaint_generation.py",
        "evaluate_r14_inpaint_feasibility.py",
    ):
        source = (REPO / "scripts" / entrypoint).read_text(encoding="utf-8")
        assert 'config["eval_index"]' in source
        assert 'config["eval_features"]' in source
        assert 'config["train_index"]' not in source
        assert 'config["train_features"]' not in source


def test_legacy_e15_loader_allows_only_zero_init_context_projection() -> None:
    import torch

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.existing = torch.nn.Linear(2, 2)
            self.context_embedder = torch.nn.Conv2d(9, 2, kernel_size=1)
            torch.nn.init.zeros_(self.context_embedder.weight)
            torch.nn.init.zeros_(self.context_embedder.bias)

    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vector_field = Backbone()

    generator = Generator()
    legacy = {
        name: value.clone()
        for name, value in generator.state_dict().items()
        if "context_embedder" not in name
    }
    smoke._load_legacy_e15_into_inpaint(generator, legacy)
    with pytest.raises(RuntimeError, match="topology differs"):
        smoke._load_legacy_e15_into_inpaint(generator, {})
    bad = Generator()
    torch.nn.init.ones_(bad.vector_field.context_embedder.weight)
    with pytest.raises(RuntimeError, match="zero-init"):
        smoke._load_legacy_e15_into_inpaint(bad, legacy)


def test_train_and_eval_manifests_are_separate_and_fail_closed(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    pair_path = tmp_path / "pairs.jsonl"
    write_jsonl(eval_path, [eval_row(index) for index in range(8)])
    write_jsonl(pair_path, [pair_row(index) for index in range(1024)])
    validator._validate_eval_manifest(eval_path, 8)
    validator._validate_train_manifest(pair_path)

    bad_pairs = [pair_row(index) for index in range(1024)]
    bad_pairs[0]["target_sample_id"] = bad_pairs[0]["source_sample_id"]
    write_jsonl(tmp_path / "bad-pair.jsonl", bad_pairs)
    with pytest.raises(validator.R14LaunchError, match="source_sample_id != target_sample_id"):
        validator._validate_train_manifest(tmp_path / "bad-pair.jsonl")

    bad_eval = eval_row(0)
    bad_eval["landmarks68"][0][0] = float("nan")
    write_jsonl(tmp_path / "bad-eval.jsonl", [bad_eval])
    with pytest.raises(validator.R14LaunchError, match="non-finite"):
        validator._validate_eval_manifest(tmp_path / "bad-eval.jsonl", 1)


def test_eval_cache_membership_is_strict_and_uses_val_index(tmp_path: Path) -> None:
    index_path = tmp_path / "val.jsonl"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    index_rows = [
        {"sample_id": "val-0"},
        {"sample_id": "val-1"},
    ]
    write_jsonl(index_path, index_rows)
    (feature_dir / "manifest.json").write_text(
        json.dumps(
            {
                "index_path": "data/index/val_face_mixed_e14.jsonl",
                "sample_ids": ["val-0", "val-1"],
            }
        ),
        encoding="utf-8",
    )
    (feature_dir / "features.pt").write_bytes(b"nonempty")
    validator._validate_cache_membership(
        index_path,
        feature_dir,
        ["val-0", "val-1"],
        "data/index/val_face_mixed_e14.jsonl",
        "evaluation",
    )
    with pytest.raises(validator.R14LaunchError, match="lacks 1 required sample IDs"):
        validator._validate_cache_membership(
            index_path,
            feature_dir,
            ["val-0", "train-only"],
            "data/index/val_face_mixed_e14.jsonl",
            "evaluation",
        )
    bad_manifest = {
        "index_path": "data/index/train_face_mixed_e14_4029avail.jsonl",
        "sample_ids": ["val-0", "val-1"],
    }
    (feature_dir / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(validator.R14LaunchError, match="cache index_path"):
        validator._validate_cache_membership(
            index_path,
            feature_dir,
            ["val-0"],
            "data/index/val_face_mixed_e14.jsonl",
            "evaluation",
        )


def test_locked_conclusion_priority_and_roi_inflation() -> None:
    all_pass = {
        "exact_one": True,
        "representation": True,
        "privacy": True,
        "full_quality": True,
        "roi_quality": True,
    }
    assert closeout._classification(all_pass, {"severe_count": None})[0] == "numeric_pass_visual_review_pending"
    assert closeout._classification(all_pass, {"severe_count": 0})[0] == "feasibility_pass_stage128_allowed"
    roi_fail = {**all_pass, "roi_quality": False}
    assert closeout._classification(roi_fail, {"severe_count": 0})[0] == "no_go_copied_background_metric_inflation"
    privacy_fail = {**all_pass, "privacy": False, "roi_quality": False}
    assert closeout._classification(privacy_fail, {"severe_count": 0})[0] == "no_go_privacy"


def test_launcher_is_single_locked_gpu0123_pipeline() -> None:
    path = REPO / "scripts/run_r14_inpaint_feasibility.sh"
    text = path.read_text(encoding="utf-8")
    assert 'GPU_LIST="0,1,2,3"' in text
    assert 'NPROC=4' in text
    assert 'SESSION="safa-r14-inpaint-v1"' in text
    assert "--nproc_per_node=$NPROC" in text
    assert text.count('"--nproc_per_node=$NPROC"') == 3
    assert "batch" not in text.lower() or "batch" in text.lower()
    for forbidden in ("retry", "controller", "postclaim", "/tmp/safa-node3"):
        assert forbidden not in text.lower()
    expected_order = ["SMOKE", "TRAIN", "EXPORT", "GENERATE", "EVALUATE", "RENDER", "CLOSE"]
    positions = [text.index(f'  "${{{name}[@]}}"', text.index("run_pipeline()")) for name in expected_order]
    assert positions == sorted(positions)
    result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_smoke_and_generation_require_four_rank_batch2() -> None:
    smoke_source = (REPO / "scripts/run_r14_inpaint_smoke.py").read_text(encoding="utf-8")
    generation_source = (REPO / "scripts/run_r14_inpaint_generation.py").read_text(encoding="utf-8")
    assert "world_size != 4" in smoke_source
    assert "DistributedDataParallel" in smoke_source
    assert "range(rank * 2, (rank + 1) * 2)" in smoke_source
    assert '"world_size": world_size' in smoke_source
    assert '"batch_size_per_rank": 2' in smoke_source
    assert "world_size != 4" in generation_source
    assert '"world_size": 4' in generation_source
    assert '"batch_size_per_rank": 2' in generation_source
    assert set(validator.GPU_BINDINGS) == {0, 1, 2, 3}


def test_artifact_validator_rejects_511_equivalent_and_nonfinite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    evaluation = tmp_path / validator.ARTIFACT_ROOT / "regular32_evaluation"
    evaluation.mkdir(parents=True)
    payload = {
        "sample_count": 32,
        "arcface": {
            "source": {"exact_one": 32},
            "native": {"exact_one": 32},
            "candidate": {"exact_one": 31},
        },
        "metrics": {
            "e0": 0.8,
            "delta_e0": 0.31,
            "delta_edev": 0.06,
            "arcface_u95": 0.01,
            "full_niqe": 4.0,
            "full_sharpness": 400.0,
            "roi_niqe": 4.0,
            "roi_sharpness": 400.0,
        },
        "gate": {
            "exact_one": False,
            "representation": True,
            "privacy": True,
            "full_quality": True,
            "roi_quality": True,
        },
    }
    (evaluation / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(validator.R14LaunchError, match="candidate.exact_one"):
        validator.validate_artifact("evaluation")
