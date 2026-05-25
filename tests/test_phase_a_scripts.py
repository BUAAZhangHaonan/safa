from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), color=color).save(path)


def test_filter_writes_only_single_face_rows_and_complete_manifest(tmp_path: Path) -> None:
    module = _load_script("filter_single_face_index")
    image_paths = []
    for index, color in enumerate([(0, 0, 0), (30, 0, 0), (60, 0, 0), (90, 0, 0)]):
        path = tmp_path / f"image_{index}.png"
        _write_png(path, color)
        image_paths.append(path)

    source = tmp_path / "source.jsonl"
    rows = [
        {"sample_id": "zero", "image_path": str(image_paths[0]), "label": 0},
        {"sample_id": "single-a", "image_path": str(image_paths[1]), "label": 1},
        {"sample_id": "multi", "image_path": str(image_paths[2]), "label": 2},
        {"sample_id": "single-b", "image_path": str(image_paths[3]), "label": 3},
    ]
    _write_jsonl(source, rows)

    class FakeDetector:
        def __init__(self) -> None:
            self.seen: list[Path] = []

        def count_faces(self, image_path: Path) -> int:
            self.seen.append(image_path)
            return {
                image_paths[0]: 0,
                image_paths[1]: 1,
                image_paths[2]: 2,
                image_paths[3]: 1,
            }[image_path]

    fake_detector = FakeDetector()
    output = tmp_path / "val_single_face.jsonl"

    manifest_path = module.filter_single_face_index(
        source_index=source,
        output_index=output,
        detector_name="insightface_buffalo_l",
        device="cuda:0",
        detector_factory=lambda detector_name, device: fake_detector,
    )

    assert [row["sample_id"] for row in _read_jsonl(output)] == ["single-a", "single-b"]
    assert fake_detector.seen == image_paths

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_index"] == str(source)
    assert manifest["output_index"] == str(output)
    assert len(manifest["source_index_sha256"]) == 64
    assert len(manifest["output_index_sha256"]) == 64
    assert manifest["detector"] == "insightface_buffalo_l"
    assert manifest["device"] == "cuda:0"
    assert manifest["num_source"] == 4
    assert manifest["num_single_face"] == 2
    assert manifest["num_zero_face"] == 1
    assert manifest["num_multi_face"] == 1


def test_filter_rejects_unsupported_detector(tmp_path: Path) -> None:
    module = _load_script("filter_single_face_index")

    with pytest.raises(ValueError, match="unsupported detector"):
        module.build_detector("mtcnn", "cuda:0")


def test_quality_eval_uses_unpaired_real_and_generated_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("eval_generation_quality")
    real_dir = tmp_path / "real"
    generated_dir = tmp_path / "generated"
    real_dir.mkdir()
    generated_dir.mkdir()

    real_paths = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        path = real_dir / f"real_{index}.png"
        _write_png(path, color)
        real_paths.append(path)
    for index, color in enumerate([(0, 0, 255), (255, 255, 0), (0, 255, 255)]):
        _write_png(generated_dir / f"generated_{index}.png", color)

    real_index = tmp_path / "real.jsonl"
    _write_jsonl(
        real_index,
        [
            {"sample_id": f"real-{index}", "image_path": str(path), "label": index}
            for index, path in enumerate(real_paths)
        ],
    )

    class FakeFid:
        def __init__(self) -> None:
            self.real = 0
            self.generated = 0

        def update(self, images, real: bool) -> None:
            assert str(images.dtype) == "torch.uint8"
            if real:
                self.real += int(images.shape[0])
            else:
                self.generated += int(images.shape[0])

        def compute(self):
            import torch

            assert self.real == 2
            assert self.generated == 3
            return torch.tensor(12.5)

    class FakeKid(FakeFid):
        def compute(self):
            import torch

            assert self.real == 2
            assert self.generated == 3
            return torch.tensor(0.125), torch.tensor(0.025)

    class FakeIqa:
        def __init__(self) -> None:
            self.values = [1.0, 2.0, 3.0]

        def __call__(self, images):
            import torch

            assert str(images.dtype) == "torch.float32"
            assert images.shape[0] == 1
            return torch.tensor([self.values.pop(0)])

    monkeypatch.setattr(module, "create_fid_metric", FakeFid)
    monkeypatch.setattr(module, "create_kid_metric", FakeKid)
    monkeypatch.setattr(module, "create_iqa_metric", lambda method: FakeIqa())

    output = tmp_path / "quality.json"
    result = module.main(
        [
            "--real-index",
            str(real_index),
            "--generated-dir",
            str(generated_dir),
            "--output",
            str(output),
            "--iqa-method",
            "niqe",
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["num_real"] == 2
    assert payload["num_generated"] == 3
    assert payload["fid"] == pytest.approx(12.5)
    assert payload["kid_mean"] == pytest.approx(0.125)
    assert payload["kid_std"] == pytest.approx(0.025)
    assert payload["iqa"] == {"method": "niqe", "mean": 2.0, "std": pytest.approx(0.816496580927726)}


def test_quality_eval_niqe_only_does_not_create_fid_or_kid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("eval_generation_quality")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    _write_png(generated_dir / "generated.png", (0, 255, 0))

    class FakeIqa:
        def __call__(self, images):
            import torch

            assert str(images.dtype) == "torch.float32"
            assert images.shape[0] == 1
            return torch.tensor([3.5])

    monkeypatch.setattr(module, "create_fid_metric", lambda: pytest.fail("FID metric should not be created"))
    monkeypatch.setattr(module, "create_kid_metric", lambda: pytest.fail("KID metric should not be created"))
    monkeypatch.setattr(module, "create_iqa_metric", lambda method: FakeIqa())

    output = tmp_path / "quality.json"
    payload = module.evaluate_generation_quality(
        real_index=None,
        generated_dir=generated_dir,
        output=output,
        iqa_method="niqe",
        metrics=["niqe"],
    )

    assert payload["metrics"] == ["niqe"]
    assert payload["num_generated"] == 1
    assert "num_real" not in payload
    assert "fid" not in payload
    assert "kid_mean" not in payload
    assert "kid_std" not in payload
    assert payload["iqa"] == {"method": "niqe", "mean": 3.5, "std": 0.0}


def test_quality_eval_rejects_empty_generated_dir_before_creating_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("eval_generation_quality")
    real_image = tmp_path / "real.png"
    _write_png(real_image, (255, 0, 0))
    real_index = tmp_path / "real.jsonl"
    _write_jsonl(real_index, [{"sample_id": "real-0", "image_path": str(real_image), "label": 0}])
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    output = tmp_path / "quality.json"

    monkeypatch.setattr(module, "create_fid_metric", lambda: pytest.fail("FID metric should not be created"))
    monkeypatch.setattr(module, "create_kid_metric", lambda: pytest.fail("KID metric should not be created"))
    monkeypatch.setattr(module, "create_iqa_metric", lambda method: pytest.fail("IQA metric should not be created"))

    result = module.main(
        [
            "--real-index",
            str(real_index),
            "--generated-dir",
            str(generated_dir),
            "--output",
            str(output),
            "--metrics",
            "niqe",
        ]
    )

    assert result == 1
    assert "generated-dir contains no supported images" in capsys.readouterr().err
    assert not output.exists()


def test_quality_eval_reports_missing_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("eval_generation_quality")
    real_image = tmp_path / "real.png"
    generated_image = tmp_path / "generated.png"
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    _write_png(real_image, (255, 0, 0))
    _write_png(generated_image, (0, 255, 0))
    generated_image.rename(generated_dir / generated_image.name)

    real_index = tmp_path / "real.jsonl"
    _write_jsonl(real_index, [{"sample_id": "real-0", "image_path": str(real_image), "label": 0}])
    output = tmp_path / "quality.json"

    monkeypatch.setattr(module, "create_fid_metric", lambda: (_ for _ in ()).throw(RuntimeError("torchmetrics is required")))

    result = module.main(
        [
            "--real-index",
            str(real_index),
            "--generated-dir",
            str(generated_dir),
            "--output",
            str(output),
        ]
    )

    assert result == 1
    assert "torchmetrics is required" in capsys.readouterr().err
    assert not output.exists()
