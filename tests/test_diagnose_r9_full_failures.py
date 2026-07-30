from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/diagnose_r9_full_failures.py"
spec = importlib.util.spec_from_file_location("diagnose_r9_full_failures", SCRIPT)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def _face(offset: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        bbox=np.array([1.0 + offset, 2.0, 11.0 + offset, 22.0]),
        det_score=np.float32(0.875),
        kps=np.array(
            [
                [3.0 + offset, 6.0],
                [8.0 + offset, 6.0],
                [5.5 + offset, 10.0],
                [3.5 + offset, 16.0],
                [7.5 + offset, 16.0],
            ]
        ),
    )


class FakeAnalyzer:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def get(self, image):
        assert image.shape == (40, 50, 3)
        return self.outputs.pop(0)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_fake_analyzer_serializes_zero_one_two_faces(count: int) -> None:
    faces = [_face(float(index)) for index in range(count)]
    analyzer = FakeAnalyzer([faces, faces])
    observations = cli.observe_twice(
        analyzer, np.zeros((40, 50, 3), dtype=np.uint8)
    )
    assert [row["face_count"] for row in observations] == [count, count]
    assert observations[0]["image_shape"] == [40, 50, 3]
    if count:
        assert observations[0]["faces"][0] == {
            "bbox": [1.0, 2.0, 11.0, 22.0],
            "det_score": pytest.approx(0.875),
            "kps": [
                [3.0, 6.0],
                [8.0, 6.0],
                [5.5, 10.0],
                [3.5, 16.0],
                [7.5, 16.0],
            ],
            "bbox_area_ratio": pytest.approx(0.1),
        }


def test_repeated_face_count_nondeterminism_is_rejected() -> None:
    analyzer = FakeAnalyzer([[], [_face()]])
    with pytest.raises(cli.DiagnosticError, match="nondeterministic"):
        cli.observe_twice(analyzer, np.zeros((40, 50, 3), dtype=np.uint8))


def _asset(path: Path, value: bytes) -> tuple[str, str]:
    path.write_bytes(value)
    return str(path.resolve()), hashlib.sha256(value).hexdigest()


def _sample(tmp_path: Path, sample_id: str = "sample-0") -> dict[str, str]:
    sample = {"sample_id": sample_id}
    for role in cli.ROLES:
        path, digest = _asset(tmp_path / f"{role}.png", role.encode())
        sample[role] = path
        sample[f"{role}_sha256"] = digest
    return sample


def test_sample_id_and_path_validation(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    normalized, by_id = cli.validate_sample_paths([sample])
    assert normalized[0]["sample_id"] == "sample-0"
    assert by_id["sample-0"]["candidate"].is_file()
    with pytest.raises(cli.DiagnosticError, match="duplicate sample_id"):
        cli.validate_sample_paths([sample, sample])
    invalid = dict(sample)
    invalid["source"] = "relative.png"
    with pytest.raises(cli.DiagnosticError, match="must be absolute"):
        cli.validate_sample_paths([invalid])
    missing = dict(sample)
    missing["native"] = str((tmp_path / "absent.png").resolve())
    with pytest.raises(FileNotFoundError, match="missing native image"):
        cli.validate_sample_paths([missing])


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    full_root = tmp_path / "full"
    output_dir = tmp_path / "diagnostic"
    full_root.mkdir()
    output_dir.mkdir()
    args = argparse.Namespace(
        full_root=full_root, output_dir=output_dir, device="cuda:0"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.run(args)


def _pair_rows() -> tuple[list[str], list[dict[str, float | int | str]]]:
    ids = [f"sample-{index:04d}" for index in range(2048)]
    rows = [
        {
            "sample_id": sample_id,
            "seed": 7919,
            "native_sharpness": float(index + 1),
            "candidate_sharpness": float(index + 2),
        }
        for index, sample_id in enumerate(ids)
    ]
    return ids, rows


def test_sharpness_row_alignment_and_deciles() -> None:
    ids, rows = _pair_rows()
    normalized = cli.normalize_pair_rows(rows, ids, 2048)
    assert normalized[-1]["delta_sharpness"] == pytest.approx(1.0)
    deciles, summary = cli.summarize_sharpness(normalized)
    assert len(deciles) == 10
    assert sum(row["count"] for row in deciles) == 2048
    assert summary["delta_mean"] == pytest.approx(1.0)
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(cli.DiagnosticError, match="do not align"):
        cli.normalize_pair_rows(rows, ids, 2048)


def test_sharpness_nonfinite_and_duplicate_ids_are_rejected() -> None:
    ids, rows = _pair_rows()
    rows[0]["candidate_sharpness"] = float("nan")
    with pytest.raises(cli.DiagnosticError, match="must be finite"):
        cli.normalize_pair_rows(rows, ids, 2048)
    _, rows = _pair_rows()
    rows[1]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(cli.DiagnosticError, match="duplicate ID"):
        cli.normalize_pair_rows(rows, ids, 2048)


def test_laplacian_variance_matches_formal_quality_implementation(
    tmp_path: Path,
) -> None:
    quality_path = REPO_ROOT / "scripts/eval_generation_quality.py"
    quality_spec = importlib.util.spec_from_file_location(
        "eval_generation_quality_for_diagnostic_test", quality_path
    )
    assert quality_spec is not None and quality_spec.loader is not None
    quality = importlib.util.module_from_spec(quality_spec)
    quality_spec.loader.exec_module(quality)
    image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    path = tmp_path / "gray.png"
    assert cv2.imwrite(str(path), image)
    assert cli.laplacian_variance(image) == pytest.approx(
        quality.laplacian_variance(path), rel=0.0, abs=0.0
    )
