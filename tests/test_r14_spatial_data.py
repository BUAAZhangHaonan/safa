from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from safa.data.index_schema import IndexRecord, write_index
from safa.data.r14_spatial import (
    AffectNetSpatialRecord,
    R14SpatialEvalDataset,
    R14SpatialEvalRecord,
    R14SpatialPairDataset,
    build_eval_records,
    build_self_free_same_label_pairs,
    build_spatial_index_from_affectnet_csv,
    write_eval_manifest,
    write_pair_manifest,
)
from safa.training.transforms import r14_joint_transform
from safa.utils.hashing import sha256_file


def _landmarks(offset: float = 0.0) -> tuple[tuple[float, float], ...]:
    return tuple((float(index % 10) + offset, float(index // 10) + offset) for index in range(68))


def _spatial_record(
    sample_id: str,
    image_path: Path,
    label: int,
    bbox: tuple[int, int, int, int] = (2, 3, 4, 5),
) -> AffectNetSpatialRecord:
    return AffectNetSpatialRecord(
        sample_id=sample_id,
        image_path=str(image_path),
        affect_label=label,
        split="train",
        dataset_root=str(image_path.parent),
        dataset_version="unit",
        bbox_xywh=bbox,
        landmarks68=_landmarks(),
    )


def _write_feature_fixture(root: Path):
    import torch

    image_a = root / "a.png"
    image_b = root / "b.png"
    Image.new("RGB", (10, 10), (10, 20, 30)).save(image_a)
    Image.new("RGB", (10, 10), (100, 150, 200)).save(image_b)
    index_path = root / "source_index.jsonl"
    records = [
        IndexRecord("a", str(image_a), 1, "train", str(root), "unit"),
        IndexRecord("b", str(image_b), 1, "train", str(root), "unit"),
    ]
    write_index(records, index_path)
    checkpoint = root / "e0.pt"
    checkpoint.write_bytes(b"checkpoint")
    cache_dir = root / "features"
    cache_dir.mkdir()
    features = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    shard = cache_dir / "features.pt"
    torch.save({"features": features, "sample_ids": ["a", "b"], "labels": [1, 1]}, shard)
    manifest = {
        "dataset": "AffectNet",
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "encoder_checkpoint": str(checkpoint),
        "encoder_checkpoint_sha256": sha256_file(checkpoint),
        "num_samples": 2,
        "feature_dim": 4,
        "l2_normalized": True,
        "dtype": "float32",
        "shard": "features.pt",
        "shard_sha256": sha256_file(shard),
        "sample_ids": ["a", "b"],
        "labels": [1, 1],
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return image_a, image_b, index_path, cache_dir, checkpoint, features


def test_csv_sidecar_preserves_exact_bbox_and_68_landmarks(tmp_path: Path) -> None:
    image_dir = tmp_path / "Manually_Annotated_Images" / "folder"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (20, 20)).save(image_path)
    index_path = tmp_path / "index.jsonl"
    write_index(
        [IndexRecord("train:sample", str(image_path), 3, "train", str(tmp_path), "unit")],
        index_path,
    )
    values = [float(index) / 10.0 for index in range(136)]
    csv_path = tmp_path / "training.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subDirectory_filePath",
                "face_x",
                "face_y",
                "face_width",
                "face_height",
                "facial_landmarks",
                "expression",
                "valence",
                "arousal",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subDirectory_filePath": "folder/sample.jpg",
                "face_x": "2",
                "face_y": "3",
                "face_width": "4",
                "face_height": "5",
                "facial_landmarks": ";".join(str(value) for value in values),
                "expression": "3",
                "valence": "0",
                "arousal": "0",
            }
        )

    records = build_spatial_index_from_affectnet_csv(index_path, [csv_path])

    assert len(records) == 1
    assert records[0].bbox_xywh == (2, 3, 4, 5)
    assert records[0].landmarks68[0] == (0.0, 0.1)
    assert records[0].landmarks68[-1] == (13.4, 13.5)


def test_csv_sidecar_rejects_missing_landmark_column(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (10, 10)).save(image)
    index = tmp_path / "index.jsonl"
    write_index([IndexRecord("a", str(image), 0, "train", str(tmp_path), "unit")], index)
    csv_path = tmp_path / "training.csv"
    csv_path.write_text(
        "subDirectory_filePath,face_x,face_y,face_width,face_height,expression\n"
        "sample.jpg,0,0,5,5,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="facial_landmarks"):
        build_spatial_index_from_affectnet_csv(index, [csv_path])


def test_pair_builder_is_same_label_and_self_free(tmp_path: Path) -> None:
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (10, 10)).save(path)
        paths.append(path)
    records = [_spatial_record(name, path, 2) for name, path in zip(("a", "b", "c"), paths)]

    pairs = build_self_free_same_label_pairs(records, records, pairing_seed=7, pairs_per_source=2)

    assert len(pairs) == 6
    assert all(pair.source_sample_id != pair.target_sample_id for pair in pairs)
    assert all(pair.affect_label == 2 for pair in pairs)
    assert len({pair.pair_id for pair in pairs}) == 6


def test_pair_dataset_uses_cached_a_feature_and_only_b_pixels(tmp_path: Path) -> None:
    import torch

    image_a, image_b, index, cache, checkpoint, features = _write_feature_fixture(tmp_path)
    pair = build_self_free_same_label_pairs(
        [_spatial_record("a", image_a, 1)],
        [_spatial_record("b", image_b, 1)],
        pairing_seed=0,
    )[0]
    manifest = tmp_path / "train_pairs.jsonl"
    write_pair_manifest([pair], manifest)
    dataset = R14SpatialPairDataset(
        manifest,
        index,
        cache,
        checkpoint,
        r14_joint_transform(20, horizontal_flip_probability=0.0),
    )
    # The source path is validated as provenance at construction, but source
    # pixels are never decoded for an item.
    image_a.unlink()

    item = dataset[0]

    torch.testing.assert_close(item["source_z"], features[0])
    assert "source_image" not in item
    assert item["source_sample_id"] == "a"
    assert item["target_sample_id"] == "b"
    assert item["source_sample_id"] != item["target_sample_id"]
    assert tuple(item["face_mask"].shape) == (1, 20, 20)
    assert item["face_mask"].sum().item() == 80
    expanded = item["face_mask"].expand_as(item["target_image"])
    assert torch.count_nonzero(item["context_image"].masked_select(expanded)).item() == 0
    torch.testing.assert_close(
        item["context_image"].masked_select(~expanded),
        item["target_image"].masked_select(~expanded),
    )
    torch.testing.assert_close(item["bbox_xywh"], torch.tensor([4.0, 6.0, 8.0, 10.0]))


def test_eval_dataset_is_single_input_s_with_its_cached_feature(tmp_path: Path) -> None:
    import torch

    _, image_b, index, cache, checkpoint, features = _write_feature_fixture(tmp_path)
    eval_record = R14SpatialEvalRecord(
        sample_id="b",
        image_path=str(image_b),
        affect_label=1,
        bbox_xywh=(2, 3, 4, 5),
        landmarks68=_landmarks(),
    )
    manifest = tmp_path / "regular32.jsonl"
    write_eval_manifest([eval_record], manifest)
    dataset = R14SpatialEvalDataset(
        manifest,
        index,
        cache,
        checkpoint,
        r14_joint_transform(10),
    )

    item = dataset[0]

    torch.testing.assert_close(item["source_z"], features[1])
    assert item["sample_id"] == "b"
    assert "target_sample_id" not in item
    assert torch.count_nonzero(
        item["context_image"].masked_select(item["face_mask"].expand_as(item["image"]))
    ).item() == 0


def test_build_eval_records_has_no_donor_fields(tmp_path: Path) -> None:
    image = tmp_path / "s.png"
    Image.new("RGB", (10, 10)).save(image)
    record = build_eval_records([_spatial_record("s", image, 0)])[0]
    mapping = record.to_mapping()
    assert mapping["contract_version"] == "safa_r14_spatial_eval_v1"
    assert not any("target" in key or "donor" in key for key in mapping)


def test_joint_flip_moves_mask_bbox_and_landmarks_together() -> None:
    import torch

    image = Image.new("RGB", (10, 10), (20, 30, 40))
    transform = r14_joint_transform(10, horizontal_flip_probability=1.0)

    result = transform(image, (2, 3, 4, 5), _landmarks())

    torch.testing.assert_close(result["bbox_xywh"], torch.tensor([4.0, 3.0, 4.0, 5.0]))
    assert result["face_mask"][0, 3:8, 4:8].all()
    assert result["landmarks68"][0, 0].item() == 9.0


def test_nonfinite_landmark_and_self_pair_fail_explicitly(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (10, 10)).save(image)
    mapping = R14SpatialEvalRecord(
        "s",
        str(image),
        0,
        (1, 1, 5, 5),
        _landmarks(),
    ).to_mapping()
    mapping["landmarks68"][4][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        R14SpatialEvalRecord.from_mapping(mapping)

    records = [_spatial_record("s", image, 0)]
    with pytest.raises(ValueError, match="No self-free target"):
        build_self_free_same_label_pairs(records, records, pairing_seed=0)
