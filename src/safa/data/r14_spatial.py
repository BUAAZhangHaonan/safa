from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from safa.data.dataset import load_rgb_image_strict
from safa.data.feature_cache import load_feature_cache
from safa.data.index_schema import IndexRecord, read_index


SPATIAL_INDEX_CONTRACT = "safa_r14_affectnet_spatial_v1"
PAIR_MANIFEST_CONTRACT = "safa_r14_spatial_pair_v1"
EVAL_MANIFEST_CONTRACT = "safa_r14_spatial_eval_v1"
AFFECTNET_SPATIAL_FIELDS = (
    "subDirectory_filePath",
    "face_x",
    "face_y",
    "face_width",
    "face_height",
    "facial_landmarks",
    "expression",
)


def _require_exact_keys(data: dict[str, Any], required: set[str], context: str) -> None:
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")
    extra = set(data).difference(required)
    if extra:
        raise ValueError(f"{context} has unexpected fields: {sorted(extra)}")


def _require_string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_label(value: Any, name: str = "affect_label") -> int:
    if type(value) is not int or value not in range(8):
        raise ValueError(f"{name} must be an integer in [0, 7], got {value!r}")
    return value


def _require_bbox(value: Any, name: str) -> tuple[int, int, int, int]:
    if type(value) is not list or len(value) != 4:
        raise ValueError(f"{name} must be a four-element list [x, y, width, height]")
    if any(type(item) is not int for item in value):
        raise ValueError(f"{name} entries must be integers")
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"{name} must have x/y >= 0 and width/height > 0, got {value}")
    return x, y, width, height


def _require_landmarks(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if type(value) is not list or len(value) != 68:
        raise ValueError(f"{name} must contain exactly 68 two-dimensional points")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if type(point) is not list or len(point) != 2:
            raise ValueError(f"{name}[{index}] must be [x, y]")
        if any(type(item) not in {int, float} for item in point):
            raise ValueError(f"{name}[{index}] coordinates must be numeric")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{name}[{index}] contains a non-finite coordinate")
        points.append((x, y))
    return tuple(points)


def _landmarks_to_json(points: Sequence[tuple[float, float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


@dataclass(frozen=True)
class AffectNetSpatialRecord:
    sample_id: str
    image_path: str
    affect_label: int
    split: str
    dataset_root: str
    dataset_version: str
    bbox_xywh: tuple[int, int, int, int]
    landmarks68: tuple[tuple[float, float], ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AffectNetSpatialRecord":
        required = {
            "contract_version",
            "sample_id",
            "image_path",
            "affect_label",
            "split",
            "dataset_root",
            "dataset_version",
            "bbox_xywh",
            "landmarks68",
        }
        if type(data) is not dict:
            raise ValueError("R14 spatial index record must be a mapping")
        _require_exact_keys(data, required, "R14 spatial index record")
        if data["contract_version"] != SPATIAL_INDEX_CONTRACT:
            raise ValueError(
                f"R14 spatial index contract must be {SPATIAL_INDEX_CONTRACT!r}, "
                f"got {data['contract_version']!r}"
            )
        image_path = Path(_require_string(data["image_path"], "image_path"))
        if not image_path.is_file():
            raise FileNotFoundError(f"R14 spatial image does not exist: {image_path}")
        return cls(
            sample_id=_require_string(data["sample_id"], "sample_id"),
            image_path=str(image_path),
            affect_label=_require_label(data["affect_label"]),
            split=_require_string(data["split"], "split"),
            dataset_root=_require_string(data["dataset_root"], "dataset_root"),
            dataset_version=_require_string(data["dataset_version"], "dataset_version"),
            bbox_xywh=_require_bbox(data["bbox_xywh"], "bbox_xywh"),
            landmarks68=_require_landmarks(data["landmarks68"], "landmarks68"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "contract_version": SPATIAL_INDEX_CONTRACT,
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "affect_label": self.affect_label,
            "split": self.split,
            "dataset_root": self.dataset_root,
            "dataset_version": self.dataset_version,
            "bbox_xywh": list(self.bbox_xywh),
            "landmarks68": _landmarks_to_json(self.landmarks68),
        }


@dataclass(frozen=True)
class R14SpatialPairRecord:
    pair_id: str
    source_sample_id: str
    target_sample_id: str
    affect_label: int
    source_image_path: str
    target_image_path: str
    source_bbox_xywh: tuple[int, int, int, int]
    target_bbox_xywh: tuple[int, int, int, int]
    source_landmarks68: tuple[tuple[float, float], ...]
    target_landmarks68: tuple[tuple[float, float], ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "R14SpatialPairRecord":
        required = {
            "contract_version",
            "pair_id",
            "source_sample_id",
            "target_sample_id",
            "affect_label",
            "source_image_path",
            "target_image_path",
            "source_bbox_xywh",
            "target_bbox_xywh",
            "source_landmarks68",
            "target_landmarks68",
        }
        if type(data) is not dict:
            raise ValueError("R14 spatial pair record must be a mapping")
        _require_exact_keys(data, required, "R14 spatial pair record")
        if data["contract_version"] != PAIR_MANIFEST_CONTRACT:
            raise ValueError(
                f"R14 pair contract must be {PAIR_MANIFEST_CONTRACT!r}, "
                f"got {data['contract_version']!r}"
            )
        source_sample_id = _require_string(data["source_sample_id"], "source_sample_id")
        target_sample_id = _require_string(data["target_sample_id"], "target_sample_id")
        if source_sample_id == target_sample_id:
            raise ValueError("R14 training pairs require source_sample_id != target_sample_id")
        source_image_path = Path(_require_string(data["source_image_path"], "source_image_path"))
        target_image_path = Path(_require_string(data["target_image_path"], "target_image_path"))
        if not source_image_path.is_file():
            raise FileNotFoundError(f"R14 pair source image does not exist: {source_image_path}")
        if not target_image_path.is_file():
            raise FileNotFoundError(f"R14 pair target image does not exist: {target_image_path}")
        return cls(
            pair_id=_require_string(data["pair_id"], "pair_id"),
            source_sample_id=source_sample_id,
            target_sample_id=target_sample_id,
            affect_label=_require_label(data["affect_label"]),
            source_image_path=str(source_image_path),
            target_image_path=str(target_image_path),
            source_bbox_xywh=_require_bbox(data["source_bbox_xywh"], "source_bbox_xywh"),
            target_bbox_xywh=_require_bbox(data["target_bbox_xywh"], "target_bbox_xywh"),
            source_landmarks68=_require_landmarks(data["source_landmarks68"], "source_landmarks68"),
            target_landmarks68=_require_landmarks(data["target_landmarks68"], "target_landmarks68"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "contract_version": PAIR_MANIFEST_CONTRACT,
            "pair_id": self.pair_id,
            "source_sample_id": self.source_sample_id,
            "target_sample_id": self.target_sample_id,
            "affect_label": self.affect_label,
            "source_image_path": self.source_image_path,
            "target_image_path": self.target_image_path,
            "source_bbox_xywh": list(self.source_bbox_xywh),
            "target_bbox_xywh": list(self.target_bbox_xywh),
            "source_landmarks68": _landmarks_to_json(self.source_landmarks68),
            "target_landmarks68": _landmarks_to_json(self.target_landmarks68),
        }


@dataclass(frozen=True)
class R14SpatialEvalRecord:
    sample_id: str
    image_path: str
    affect_label: int
    bbox_xywh: tuple[int, int, int, int]
    landmarks68: tuple[tuple[float, float], ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "R14SpatialEvalRecord":
        required = {
            "contract_version",
            "sample_id",
            "image_path",
            "affect_label",
            "bbox_xywh",
            "landmarks68",
        }
        if type(data) is not dict:
            raise ValueError("R14 spatial eval record must be a mapping")
        _require_exact_keys(data, required, "R14 spatial eval record")
        if data["contract_version"] != EVAL_MANIFEST_CONTRACT:
            raise ValueError(
                f"R14 eval contract must be {EVAL_MANIFEST_CONTRACT!r}, "
                f"got {data['contract_version']!r}"
            )
        image_path = Path(_require_string(data["image_path"], "image_path"))
        if not image_path.is_file():
            raise FileNotFoundError(f"R14 eval image does not exist: {image_path}")
        return cls(
            sample_id=_require_string(data["sample_id"], "sample_id"),
            image_path=str(image_path),
            affect_label=_require_label(data["affect_label"]),
            bbox_xywh=_require_bbox(data["bbox_xywh"], "bbox_xywh"),
            landmarks68=_require_landmarks(data["landmarks68"], "landmarks68"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "contract_version": EVAL_MANIFEST_CONTRACT,
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "affect_label": self.affect_label,
            "bbox_xywh": list(self.bbox_xywh),
            "landmarks68": _landmarks_to_json(self.landmarks68),
        }


def _read_jsonl(path: str | Path, parser: Callable[[dict[str, Any]], Any], kind: str) -> list[Any]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"R14 {kind} does not exist: {input_path}")
    records: list[Any] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"R14 {kind} contains an empty line at {line_no}")
            try:
                data = json.loads(line)
                records.append(parser(data))
            except Exception as exc:
                raise ValueError(f"Invalid R14 {kind} line {line_no} in {input_path}: {exc}") from exc
    if not records:
        raise ValueError(f"R14 {kind} contains no records: {input_path}")
    return records


def _write_jsonl(records: Iterable[Any], path: str | Path) -> None:
    materialized = list(records)
    if not materialized:
        raise ValueError("Refusing to write an empty R14 manifest")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialized:
            handle.write(json.dumps(record.to_mapping(), sort_keys=True, allow_nan=False) + "\n")


def read_spatial_index(path: str | Path) -> list[AffectNetSpatialRecord]:
    records = _read_jsonl(path, AffectNetSpatialRecord.from_mapping, "spatial index")
    _require_unique([record.sample_id for record in records], "spatial index sample_id")
    return records


def write_spatial_index(records: Iterable[AffectNetSpatialRecord], path: str | Path) -> None:
    materialized = list(records)
    _require_unique([record.sample_id for record in materialized], "spatial index sample_id")
    _write_jsonl(materialized, path)


def read_pair_manifest(path: str | Path) -> list[R14SpatialPairRecord]:
    records = _read_jsonl(path, R14SpatialPairRecord.from_mapping, "pair manifest")
    _require_unique([record.pair_id for record in records], "pair manifest pair_id")
    return records


def write_pair_manifest(records: Iterable[R14SpatialPairRecord], path: str | Path) -> None:
    materialized = list(records)
    _require_unique([record.pair_id for record in materialized], "pair manifest pair_id")
    _write_jsonl(materialized, path)


def read_eval_manifest(path: str | Path) -> list[R14SpatialEvalRecord]:
    records = _read_jsonl(path, R14SpatialEvalRecord.from_mapping, "eval manifest")
    _require_unique([record.sample_id for record in records], "eval manifest sample_id")
    return records


def write_eval_manifest(records: Iterable[R14SpatialEvalRecord], path: str | Path) -> None:
    materialized = list(records)
    _require_unique([record.sample_id for record in materialized], "eval manifest sample_id")
    _write_jsonl(materialized, path)


def _require_unique(values: Sequence[str], context: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"R14 {context} values must be unique")


def build_spatial_index_from_affectnet_csv(
    index_path: str | Path,
    csv_paths: Sequence[str | Path],
) -> list[AffectNetSpatialRecord]:
    """Bind exact AffectNet bbox/68-point CSV annotations to an existing index.

    The existing index is authoritative for sample selection and labels. Every
    indexed image must match exactly one CSV row; missing or duplicate matches
    fail instead of being inferred from another detector.
    """

    index_records = read_index(Path(index_path))
    expected_by_path: dict[Path, IndexRecord] = {}
    for record in index_records:
        resolved = Path(record.image_path).absolute()
        if resolved in expected_by_path:
            raise ValueError(f"R14 source index contains duplicate image path: {resolved}")
        expected_by_path[resolved] = record

    matched: dict[str, AffectNetSpatialRecord] = {}
    for csv_value in csv_paths:
        csv_path = Path(csv_value)
        if not csv_path.is_file():
            raise FileNotFoundError(f"AffectNet annotation CSV does not exist: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"AffectNet CSV has no header: {csv_path}")
            for name in AFFECTNET_SPATIAL_FIELDS:
                count = reader.fieldnames.count(name)
                if count != 1:
                    raise ValueError(
                        f"AffectNet CSV {csv_path} must contain exactly one {name!r} column; got {count}"
                    )
            for row_no, row in enumerate(reader, start=2):
                candidates = _affectnet_csv_candidate_paths(
                    csv_path.parent,
                    row["subDirectory_filePath"],
                )
                indexed_matches = [expected_by_path[path] for path in candidates if path in expected_by_path]
                if not indexed_matches:
                    continue
                if len(indexed_matches) != 1:
                    raise ValueError(
                        f"AffectNet CSV path is ambiguous at {csv_path}:{row_no}: {candidates}"
                    )
                indexed = indexed_matches[0]
                if indexed.sample_id in matched:
                    raise ValueError(
                        f"AffectNet spatial annotation is ambiguous for {indexed.sample_id}: "
                        f"duplicate row at {csv_path}:{row_no}"
                    )
                label = _parse_csv_int(row["expression"], "expression", csv_path, row_no)
                if label != indexed.label:
                    raise ValueError(
                        f"AffectNet spatial label mismatch for {indexed.sample_id}: "
                        f"index={indexed.label}, csv={label}"
                    )
                bbox = tuple(
                    _parse_csv_int(row[name], name, csv_path, row_no)
                    for name in ("face_x", "face_y", "face_width", "face_height")
                )
                _require_bbox(list(bbox), f"{csv_path}:{row_no} bbox")
                landmarks = _parse_csv_landmarks(row["facial_landmarks"], csv_path, row_no)
                matched[indexed.sample_id] = AffectNetSpatialRecord(
                    sample_id=indexed.sample_id,
                    image_path=indexed.image_path,
                    affect_label=indexed.label,
                    split=indexed.split,
                    dataset_root=indexed.dataset_root,
                    dataset_version=indexed.dataset_version,
                    bbox_xywh=bbox,
                    landmarks68=landmarks,
                )

    missing = [record.sample_id for record in index_records if record.sample_id not in matched]
    if missing:
        raise ValueError(
            f"R14 spatial annotations are missing for {len(missing)} indexed samples; "
            f"first={missing[:8]}"
        )
    return [matched[record.sample_id] for record in index_records]


def _affectnet_csv_candidate_paths(root: Path, raw_path: Any) -> list[Path]:
    relative_text = _require_string(raw_path, "subDirectory_filePath")
    relative = Path(relative_text)
    if relative.is_absolute():
        return [relative.absolute()]
    return [
        (root / relative).absolute(),
        (root / "Manually_Annotated_Images" / relative).absolute(),
    ]


def _parse_csv_int(raw: Any, field: str, csv_path: Path, row_no: int) -> int:
    text = _require_string(raw, f"{csv_path}:{row_no} {field}")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{csv_path}:{row_no} {field} must be numeric, got {text!r}") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{csv_path}:{row_no} {field} must be a finite integer, got {text!r}")
    return int(value)


def _parse_csv_landmarks(raw: Any, csv_path: Path, row_no: int) -> tuple[tuple[float, float], ...]:
    text = _require_string(raw, f"{csv_path}:{row_no} facial_landmarks")
    tokens = text.split(";")
    if len(tokens) != 136:
        raise ValueError(
            f"{csv_path}:{row_no} facial_landmarks must contain exactly 136 semicolon-separated values; "
            f"got {len(tokens)}"
        )
    coordinates: list[float] = []
    for index, token in enumerate(tokens):
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"{csv_path}:{row_no} facial_landmarks[{index}] must be numeric, got {token!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"{csv_path}:{row_no} facial_landmarks[{index}] is non-finite")
        coordinates.append(value)
    return tuple((coordinates[index], coordinates[index + 1]) for index in range(0, 136, 2))


def build_self_free_same_label_pairs(
    source_records: Sequence[AffectNetSpatialRecord],
    target_records: Sequence[AffectNetSpatialRecord],
    *,
    pairing_seed: int,
    pairs_per_source: int = 1,
) -> list[R14SpatialPairRecord]:
    if type(pairing_seed) is not int:
        raise ValueError("pairing_seed must be an integer")
    if type(pairs_per_source) is not int or pairs_per_source <= 0:
        raise ValueError("pairs_per_source must be a positive integer")
    _require_unique([record.sample_id for record in source_records], "source sample_id")
    _require_unique([record.sample_id for record in target_records], "target sample_id")
    target_buckets: dict[int, list[AffectNetSpatialRecord]] = {}
    for record in target_records:
        target_buckets.setdefault(record.affect_label, []).append(record)
    for bucket in target_buckets.values():
        bucket.sort(key=lambda record: record.sample_id)

    pairs: list[R14SpatialPairRecord] = []
    for source_index, source in enumerate(source_records):
        eligible = [
            target
            for target in target_buckets.get(source.affect_label, [])
            if target.sample_id != source.sample_id
        ]
        if not eligible:
            raise ValueError(
                f"No self-free target exists for source {source.sample_id} and label {source.affect_label}"
            )
        for pair_round in range(pairs_per_source):
            target = eligible[(source_index + pairing_seed + pair_round) % len(eligible)]
            if source.sample_id == target.sample_id:
                raise AssertionError("R14 self-free pairing invariant was violated")
            pair_id = f"{source.sample_id}__to__{target.sample_id}__round{pair_round}"
            pairs.append(
                R14SpatialPairRecord(
                    pair_id=pair_id,
                    source_sample_id=source.sample_id,
                    target_sample_id=target.sample_id,
                    affect_label=source.affect_label,
                    source_image_path=source.image_path,
                    target_image_path=target.image_path,
                    source_bbox_xywh=source.bbox_xywh,
                    target_bbox_xywh=target.bbox_xywh,
                    source_landmarks68=source.landmarks68,
                    target_landmarks68=target.landmarks68,
                )
            )
    return pairs


def build_eval_records(records: Sequence[AffectNetSpatialRecord]) -> list[R14SpatialEvalRecord]:
    _require_unique([record.sample_id for record in records], "eval source sample_id")
    return [
        R14SpatialEvalRecord(
            sample_id=record.sample_id,
            image_path=record.image_path,
            affect_label=record.affect_label,
            bbox_xywh=record.bbox_xywh,
            landmarks68=record.landmarks68,
        )
        for record in records
    ]


class _R14FeatureLookup:
    def __init__(
        self,
        source_index_path: str | Path,
        source_feature_dir: str | Path,
        e0_checkpoint: str | Path,
    ):
        import torch

        source_records = read_index(Path(source_index_path))
        payload, self.feature_manifest = load_feature_cache(
            source_feature_dir,
            source_index_path,
            e0_checkpoint,
        )
        sample_ids = list(payload["sample_ids"])
        labels = list(payload["labels"])
        expected_ids = [record.sample_id for record in source_records]
        if sample_ids != expected_ids:
            raise ValueError("R14 feature cache sample_id order does not match source index")
        features = payload["features"]
        if features.dtype != torch.float32:
            raise ValueError(f"R14 source features must be float32, got {features.dtype}")
        if not torch.isfinite(features).all():
            raise ValueError("R14 source features contain a non-finite value")
        self._by_id: dict[str, tuple[int, Any]] = {}
        for index, (record, label) in enumerate(zip(source_records, labels, strict=True)):
            if type(label) is not int or label != record.label:
                raise ValueError(
                    f"R14 feature label mismatch for {record.sample_id}: cache={label!r}, index={record.label}"
                )
            if record.sample_id in self._by_id:
                raise ValueError(f"R14 source index has duplicate sample_id {record.sample_id!r}")
            self._by_id[record.sample_id] = (label, features[index])

    def get(self, sample_id: str, expected_label: int):
        if sample_id not in self._by_id:
            raise ValueError(f"R14 source feature is missing for sample_id {sample_id!r}")
        label, feature = self._by_id[sample_id]
        if label != expected_label:
            raise ValueError(
                f"R14 source feature label mismatch for {sample_id}: cache={label}, manifest={expected_label}"
            )
        return feature


def _validate_original_geometry(
    image: Image.Image,
    bbox_xywh: tuple[int, int, int, int],
    landmarks68: tuple[tuple[float, float], ...],
    sample_id: str,
) -> None:
    width, height = image.size
    x, y, box_width, box_height = bbox_xywh
    if x + box_width > width or y + box_height > height:
        raise ValueError(
            f"R14 bbox for {sample_id} exceeds image bounds: bbox={bbox_xywh}, image={(width, height)}"
        )
    for index, (point_x, point_y) in enumerate(landmarks68):
        if not math.isfinite(point_x) or not math.isfinite(point_y):
            raise ValueError(f"R14 landmark {index} for {sample_id} is non-finite")


def _apply_joint_transform(
    joint_transform: Callable,
    image: Image.Image,
    bbox_xywh: tuple[int, int, int, int],
    landmarks68: tuple[tuple[float, float], ...],
    sample_id: str,
) -> dict[str, Any]:
    import torch

    _validate_original_geometry(image, bbox_xywh, landmarks68, sample_id)
    transformed = joint_transform(image, bbox_xywh, landmarks68)
    if type(transformed) is not dict:
        raise ValueError("R14 joint transform must return a mapping")
    required = {"image", "face_mask", "bbox_xywh", "landmarks68"}
    if set(transformed) != required:
        raise ValueError(
            f"R14 joint transform must return exactly {sorted(required)}, got {sorted(transformed)}"
        )
    image_tensor = transformed["image"]
    face_mask = transformed["face_mask"]
    bbox_tensor = transformed["bbox_xywh"]
    landmarks_tensor = transformed["landmarks68"]
    if not isinstance(image_tensor, torch.Tensor) or image_tensor.dtype != torch.float32:
        raise ValueError("R14 transformed image must be a float32 torch.Tensor")
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3 or not torch.isfinite(image_tensor).all():
        raise ValueError("R14 transformed image must have finite shape [3, H, W]")
    if torch.any(image_tensor < 0.0) or torch.any(image_tensor > 1.0):
        raise ValueError("R14 transformed image must be in [0, 1]")
    expected_mask_shape = (1, image_tensor.shape[1], image_tensor.shape[2])
    if not isinstance(face_mask, torch.Tensor) or face_mask.dtype != torch.bool:
        raise ValueError("R14 face_mask must be a bool torch.Tensor")
    if tuple(face_mask.shape) != expected_mask_shape or not face_mask.any():
        raise ValueError(
            f"R14 face_mask must be a non-empty tensor of shape {expected_mask_shape}, "
            f"got {tuple(face_mask.shape)}"
        )
    if not isinstance(bbox_tensor, torch.Tensor) or bbox_tensor.dtype != torch.float32:
        raise ValueError("R14 transformed bbox_xywh must be a float32 torch.Tensor")
    if tuple(bbox_tensor.shape) != (4,) or not torch.isfinite(bbox_tensor).all():
        raise ValueError("R14 transformed bbox_xywh must have finite shape [4]")
    if not isinstance(landmarks_tensor, torch.Tensor) or landmarks_tensor.dtype != torch.float32:
        raise ValueError("R14 transformed landmarks68 must be a float32 torch.Tensor")
    if tuple(landmarks_tensor.shape) != (68, 2) or not torch.isfinite(landmarks_tensor).all():
        raise ValueError("R14 transformed landmarks68 must have finite shape [68, 2]")
    return transformed


def _context_without_face(image, face_mask):
    context = image.clone()
    context.masked_fill_(face_mask.expand_as(context), 0.0)
    if not bool((context.masked_select(face_mask.expand_as(context)) == 0.0).all()):
        raise AssertionError("R14 context face pixels were not zeroed exactly")
    if not bool((context.masked_select(~face_mask.expand_as(context)) == image.masked_select(~face_mask.expand_as(image))).all()):
        raise AssertionError("R14 context changed pixels outside the exact face mask")
    return context


class R14SpatialPairDataset:
    """Affect-aligned A->B training data with no source identity pixels.

    ``source_z`` comes from identity A's verified feature cache. The image,
    context, exact bbox mask and clean target all come from identity B.
    """

    def __init__(
        self,
        pair_manifest_path: str | Path,
        source_index_path: str | Path,
        source_feature_dir: str | Path,
        e0_checkpoint: str | Path,
        joint_transform: Callable,
    ):
        if joint_transform is None:
            raise ValueError("R14SpatialPairDataset requires an explicit joint_transform")
        self.records = read_pair_manifest(pair_manifest_path)
        self.features = _R14FeatureLookup(source_index_path, source_feature_dir, e0_checkpoint)
        self.joint_transform = joint_transform
        for record in self.records:
            self.features.get(record.source_sample_id, record.affect_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.source_sample_id == record.target_sample_id:
            raise AssertionError("R14 training pair unexpectedly contains the same source and target identity")
        target_image = load_rgb_image_strict(record.target_image_path)
        transformed = _apply_joint_transform(
            self.joint_transform,
            target_image,
            record.target_bbox_xywh,
            record.target_landmarks68,
            record.target_sample_id,
        )
        image = transformed["image"]
        face_mask = transformed["face_mask"]
        return {
            "source_z": self.features.get(record.source_sample_id, record.affect_label),
            "target_image": image,
            "context_image": _context_without_face(image, face_mask),
            "face_mask": face_mask,
            "pair_id": record.pair_id,
            "source_sample_id": record.source_sample_id,
            "target_sample_id": record.target_sample_id,
            "affect_label": record.affect_label,
            "bbox_xywh": transformed["bbox_xywh"],
            "landmarks68": transformed["landmarks68"],
        }


class R14SpatialEvalDataset:
    """Single-input R14 evaluation data; all image roles come from sample S."""

    def __init__(
        self,
        eval_manifest_path: str | Path,
        source_index_path: str | Path,
        source_feature_dir: str | Path,
        e0_checkpoint: str | Path,
        joint_transform: Callable,
    ):
        if joint_transform is None:
            raise ValueError("R14SpatialEvalDataset requires an explicit joint_transform")
        self.records = read_eval_manifest(eval_manifest_path)
        self.features = _R14FeatureLookup(source_index_path, source_feature_dir, e0_checkpoint)
        self.joint_transform = joint_transform
        for record in self.records:
            self.features.get(record.sample_id, record.affect_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = load_rgb_image_strict(record.image_path)
        transformed = _apply_joint_transform(
            self.joint_transform,
            image,
            record.bbox_xywh,
            record.landmarks68,
            record.sample_id,
        )
        image_tensor = transformed["image"]
        face_mask = transformed["face_mask"]
        return {
            "source_z": self.features.get(record.sample_id, record.affect_label),
            "image": image_tensor,
            "context_image": _context_without_face(image_tensor, face_mask),
            "face_mask": face_mask,
            "sample_id": record.sample_id,
            "affect_label": record.affect_label,
            "bbox_xywh": transformed["bbox_xywh"],
            "landmarks68": transformed["landmarks68"],
        }
