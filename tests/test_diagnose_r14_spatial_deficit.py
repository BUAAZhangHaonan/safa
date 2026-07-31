from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_r14_spatial_deficit.py"
spec = importlib.util.spec_from_file_location("diagnose_r14_spatial_deficit", SCRIPT)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def test_bbox_union_uses_pixel_centers_without_expansion() -> None:
    mask = cli.bbox_union_mask(
        (5, 6, 3),
        [1.0, 1.0, 3.0, 3.0],
        [2.0, 2.0, 5.0, 4.0],
    )
    expected = np.zeros((5, 6), dtype=bool)
    expected[1:3, 1:3] = True
    expected[2:4, 2:5] = True
    assert np.array_equal(mask, expected)
    assert int(mask.sum()) == 9


def test_regional_partition_exactly_reconstructs_total() -> None:
    energy = np.arange(30, dtype=np.float64).reshape(5, 6)
    mask = cli.bbox_union_mask((5, 6, 3), [1, 1, 3, 3], [2, 2, 5, 4])
    result = cli.partition(energy, mask)
    assert result["total"] == result["face"] + result["background"]
    assert result["total"] == pytest.approx(float(energy.sum()))


def test_centered_laplacian_energy_reproduces_variance() -> None:
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, size=(24, 32), dtype=np.uint8)
    energy = cli.centered_laplacian_energy(gray)
    official = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    assert float(energy.mean()) == pytest.approx(official, abs=1e-12)


def test_multiscale_gradient_maps_are_finite_and_partitionable() -> None:
    gray = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
    maps = cli.gradient_energy_maps(gray)
    assert set(maps) == {"sigma_0.0", "sigma_1.0", "sigma_2.0"}
    mask = cli.bbox_union_mask((24, 32, 3), [2, 2, 10, 10], [4, 4, 14, 14])
    for energy in maps.values():
        parts = cli.partition(energy, mask)
        assert np.isfinite(list(parts.values())).all()
        assert parts["total"] == parts["face"] + parts["background"]


def test_bootstrap_share_is_deterministic_and_strict_boundary() -> None:
    face = np.array([1.0, 2.0, -1.0, 1.0])
    background = np.array([3.0, 2.0, 4.0, -1.0])
    indices = np.random.default_rng(91637).integers(0, 4, size=(1000, 4))
    first = cli.bootstrap_share(face, background, indices)
    second = cli.bootstrap_share(face, background, indices)
    assert first == second
    assert first["point"] == pytest.approx(9.0 / 13.0)
    assert (0.5 > cli.BACKGROUND_SHARE_THRESHOLD) is False
    assert (0.5000001 > cli.BACKGROUND_SHARE_THRESHOLD) is True


def test_exact_one_and_nonfinite_fail_closed() -> None:
    with pytest.raises(cli.SpatialDeficitError, match="exact-one"):
        cli.exact_one_bbox({"face_count": 2, "bbox": [1, 1, 2, 2]}, "candidate")
    with pytest.raises(cli.SpatialDeficitError, match="finite"):
        cli.exact_one_bbox(
            {"face_count": 1, "bbox": [1, 1, float("nan"), 2]}, "candidate"
        )
    bad = np.zeros((8, 8), dtype=np.float64)
    bad[0, 0] = np.inf
    with pytest.raises(cli.SpatialDeficitError, match="finite"):
        cli.centered_laplacian_energy(bad)


def test_missing_input_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli.read_jsonl(tmp_path / "missing.jsonl")
