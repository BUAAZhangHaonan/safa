from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/diagnose_r12_face_frequency.py"
spec = importlib.util.spec_from_file_location("diagnose_r12_face_frequency", SCRIPT)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def _face() -> SimpleNamespace:
    return SimpleNamespace(
        bbox=np.array([5.0, 6.0, 45.0, 54.0]),
        det_score=np.float32(0.95),
        kps=np.array(
            [[15.0, 20.0], [35.0, 20.0], [25.0, 30.0], [17.0, 42.0], [33.0, 42.0]],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("faces", [[], [_face(), _face()]])
def test_arcface_requires_exactly_one_face(faces) -> None:
    with pytest.raises(cli.FrequencyDiagnosticError, match="exact-one"):
        cli.serialize_exact_one_face(faces, (64, 64, 3))


def test_arcface_serialization_and_alignment() -> None:
    detection = cli.serialize_exact_one_face([_face()], (64, 64, 3))
    assert detection["face_count"] == 1
    assert detection["bbox_area_ratio"] == pytest.approx(0.46875)

    def aligner(image, landmarks):
        assert landmarks.shape == (5, 2)
        return np.zeros((112, 112, 3), dtype=np.uint8)

    crop = cli.align_face(np.zeros((64, 64, 3), dtype=np.uint8), detection, aligner)
    assert crop.shape == (112, 112, 3)


def test_invalid_landmarks_and_nonfinite_image_fail_explicitly() -> None:
    face = _face()
    face.kps[0, 0] = np.nan
    with pytest.raises(cli.FrequencyDiagnosticError, match="five finite"):
        cli.serialize_exact_one_face([face], (64, 64, 3))
    image = np.zeros((32, 32), dtype=np.float64)
    image[0, 0] = np.inf
    with pytest.raises(cli.FrequencyDiagnosticError, match="non-finite"):
        cli.radial_fft_energy(image)


def test_checkerboard_has_more_high_frequency_energy_than_gradient() -> None:
    y, x = np.indices((112, 112))
    checkerboard = ((x + y) % 2 * 255).astype(np.uint8)
    gradient = np.tile(np.arange(112, dtype=np.uint8), (112, 1))
    checker = cli.radial_fft_energy(checkerboard)[cli.PRIMARY_FREQUENCY_METRIC]
    smooth = cli.radial_fft_energy(gradient)[cli.PRIMARY_FREQUENCY_METRIC]
    assert checker > 100.0 * smooth


def test_multiscale_laplacian_is_finite_and_complete() -> None:
    image = np.tile(np.arange(112, dtype=np.uint8), (112, 1))
    result = cli.multiscale_laplacian(image)
    assert set(result) == {"scale_1.00", "scale_0.75", "scale_0.50"}
    assert all(np.isfinite(value) for value in result.values())


def _decision_row(monotonic: bool, ratio: float) -> dict:
    return {
        "roi_nfe1_gt_nfe2_gt_nfe5": monotonic,
        "transfers": {
            "transport_nfe5": {
                "roi": {
                    "fft_energy_ratio": {cli.PRIMARY_FREQUENCY_METRIC: ratio}
                }
            }
        },
    }


def test_predeclared_decision_boundary_passes_at_24_and_0p8() -> None:
    rows = [_decision_row(index < 24, 0.8) for index in range(32)]
    result = cli.evaluate_decision(
        rows,
        {"enabled": True, "required_monotonic_count": 24, "median_nfe5_ratio_max": 0.8},
    )
    assert result["confirmed"] is True
    assert result["classification"] == "face_roi_sampler_low_pass_confirmed"


@pytest.mark.parametrize(
    ("monotonic_count", "ratio"),
    [(23, 0.8), (24, 0.8000001)],
)
def test_predeclared_decision_fails_either_boundary(monotonic_count: int, ratio: float) -> None:
    rows = [_decision_row(index < monotonic_count, ratio) for index in range(32)]
    result = cli.evaluate_decision(
        rows,
        {"enabled": True, "required_monotonic_count": 24, "median_nfe5_ratio_max": 0.8},
    )
    assert result["confirmed"] is False


def test_asset_loader_preserves_manifest_order_and_rejects_missing(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    assert cv2.imwrite(str(image_a), np.zeros((16, 16, 3), dtype=np.uint8))
    assert cv2.imwrite(str(image_b), np.ones((16, 16, 3), dtype=np.uint8))
    assets = tmp_path / "assets.jsonl"
    assets.write_text(
        "\n".join(
            [
                '{"sample_id":"b","generated":"' + str(image_b).replace("\\", "\\\\") + '"}',
                '{"sample_id":"a","generated":"' + str(image_a).replace("\\", "\\\\") + '"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = cli.method_asset_paths(assets, ["a", "b"], tmp_path, "method")
    assert list(loaded) == ["a", "b"]
    with pytest.raises(cli.FrequencyDiagnosticError, match="missing 1"):
        cli.method_asset_paths(assets, ["a", "missing"], tmp_path, "method")


def test_positive_ratio_rejects_zero_reference() -> None:
    with pytest.raises(cli.FrequencyDiagnosticError, match="must be positive"):
        cli.positive_ratio(1.0, 0.0, "frequency")
