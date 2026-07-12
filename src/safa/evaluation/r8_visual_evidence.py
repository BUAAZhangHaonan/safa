from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ALLOWED_DECISIONS = frozenset(
    {
        "pass",
        "repeated_tiled_face_or_image_pattern",
        "near_uniform_or_blank_frame",
        "severe_saturation_or_clipping_destroying_global_structure",
        "broken_global_face_or_image_structure",
        "large_non_image_texture_region",
        "blank_or_near_constant",
        "unstructured_noise",
        "repeated_patch_or_tiled_artifact",
        "severe_color_clipping_or_saturation",
        "broken_global_structure",
    }
)


def write_contact_sheets(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    rows_per_page: int = 8,
    tile_size: int = 128,
) -> list[dict[str, Any]]:
    from PIL import Image

    if len(columns) != 3:
        raise ValueError("R8 visual evidence requires exactly three image columns")
    if rows_per_page <= 0 or tile_size <= 0:
        raise ValueError("contact sheet dimensions must be positive")
    if output_dir.is_symlink():
        raise ValueError(f"contact sheet output must not be a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    page_count = (len(rows) + rows_per_page - 1) // rows_per_page
    expected_names = {f"page_{index:03d}.png" for index in range(page_count)}
    for existing in output_dir.iterdir():
        if existing.is_symlink():
            raise ValueError(f"contact sheet output contains a symlink: {existing}")
        if any(existing.name.startswith(f".{name}.") for name in expected_names) and existing.name.endswith(
            ".tmp"
        ):
            if not existing.is_file():
                raise ValueError(f"contact sheet temporary output has an invalid type: {existing}")
            existing.unlink()
    pages = []
    for page_index, start in enumerate(range(0, len(rows), rows_per_page)):
        page_rows = rows[start : start + rows_per_page]
        sheet = Image.new("RGB", (tile_size * len(columns), tile_size * len(page_rows)))
        for row_index, row in enumerate(page_rows):
            for column_index, column in enumerate(columns):
                path = Path(str(row.get(column, "")))
                if not path.is_file():
                    raise FileNotFoundError(f"visual evidence image is missing: {path}")
                with Image.open(path) as image:
                    tile = image.convert("RGB").resize(
                        (tile_size, tile_size), Image.Resampling.BILINEAR
                    )
                sheet.paste(tile, (column_index * tile_size, row_index * tile_size))
        buffer = BytesIO()
        sheet.save(buffer, format="PNG")
        content = buffer.getvalue()
        path = output_dir / f"page_{page_index:03d}.png"
        if path.resolve(strict=False).parent != output_root:
            raise ValueError(f"contact sheet output escapes its directory: {path}")
        _write_or_validate_bytes(path, content)
        pages.append(
            {
                "page_index": page_index,
                "path": str(path),
                "sample_ids": [str(row["sample_id"]) for row in page_rows],
            }
        )
    extras = sorted(path.name for path in output_dir.iterdir() if path.name not in expected_names)
    if extras:
        raise ValueError(f"contact sheet directory contains unowned entries: {extras!r}")
    return pages


def build_visual_evidence_contract(
    *,
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
    pages: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    expected_count: int,
) -> dict[str, Any]:
    manifest_ids = _read_manifest_ids(manifest_path)
    if len(manifest_ids) != expected_count:
        raise ValueError(f"visual manifest must contain exactly {expected_count} sample IDs")
    row_ids = [str(row.get("sample_id", "")) for row in rows]
    if row_ids != manifest_ids:
        raise ValueError("visual rows must exactly match the locked manifest order")
    page_by_id: dict[str, dict[str, Any]] = {}
    bound_pages = []
    flattened_page_ids = []
    for expected_index, page in enumerate(pages):
        if page.get("page_index") != expected_index:
            raise ValueError("visual pages must use contiguous ordered page indices")
        page_path = Path(str(page.get("path", "")))
        page_ids = [str(value) for value in page.get("sample_ids", ())]
        if not page_ids or any(sample_id in page_by_id for sample_id in page_ids):
            raise ValueError("visual pages contain missing or duplicate sample IDs")
        page_contract = {
            "page_index": expected_index,
            "path": str(page_path),
            "sha256": _sha256_file(page_path),
            "sample_ids": page_ids,
        }
        for sample_id in page_ids:
            page_by_id[sample_id] = {
                key: page_contract[key] for key in ("page_index", "path", "sha256")
            }
        bound_pages.append(page_contract)
        flattened_page_ids.extend(page_ids)
    if flattened_page_ids != manifest_ids:
        raise ValueError("visual page membership must exactly match the locked manifest order")
    samples = []
    for sample_id, row in zip(manifest_ids, rows, strict=True):
        assets = {}
        for column in columns:
            path = Path(str(row.get(column, "")))
            assets[str(column)] = {"path": str(path), "sha256": _sha256_file(path)}
        samples.append(
            {
                "sample_id": sample_id,
                "assets": assets,
                "page": page_by_id[sample_id],
            }
        )
    payload = {
        "schema_version": 1,
        "columns": [str(column) for column in columns],
        "sample_count": expected_count,
        "sample_id_manifest": str(manifest_path),
        "sample_id_manifest_sha256": _sha256_file(manifest_path),
        "ordered_sample_id_sha256": _sample_id_digest(manifest_ids),
        "pages": bound_pages,
        "samples": samples,
    }
    payload["evidence_contract_sha256"] = _canonical_contract_sha256(payload)
    return payload


def validate_visual_review_arm(
    review: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    evidence_sha = str(evidence.get("evidence_contract_sha256", ""))
    if _canonical_contract_sha256(evidence) != evidence_sha:
        raise ValueError("visual evidence canonical digest mismatch")
    if review.get("evidence_contract_sha256") != evidence_sha:
        raise ValueError("visual review does not bind the evidence contract")
    if review.get("pages") != evidence.get("pages"):
        raise ValueError("visual review page paths or digests disagree with the evidence")
    evidence_samples = evidence.get("samples")
    review_samples = review.get("samples")
    if not isinstance(evidence_samples, list) or not isinstance(review_samples, list):
        raise ValueError("visual evidence and review must contain sample lists")
    if len(review_samples) != int(evidence.get("sample_count", -1)):
        raise ValueError("visual review sample count disagrees with the evidence")
    failures = []
    for expected, row in zip(evidence_samples, review_samples, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("visual review contains an invalid sample row")
        for field in ("sample_id", "assets", "page"):
            if row.get(field) != expected.get(field):
                raise ValueError(f"visual review sample {field} disagrees with the evidence")
        decision = str(row.get("decision", ""))
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"visual review contains an unknown decision: {decision!r}")
        if decision != "pass":
            failures.append({"sample_id": row["sample_id"], "category": decision})
    if review.get("severe_failure_count") != len(failures):
        raise ValueError("visual review severe failure count disagrees with its decisions")
    if not isinstance(review.get("passed"), bool):
        raise ValueError("visual review passed field must be boolean")
    _rehash_evidence_files(evidence)
    return {
        "reviewed_sample_count": len(review_samples),
        "severe_failure_count": len(failures),
        "failures": failures,
        "passed": bool(review["passed"]),
        "evidence_contract_sha256": evidence_sha,
    }


def _rehash_evidence_files(evidence: Mapping[str, Any]) -> None:
    for page in evidence.get("pages", ()):
        if _sha256_file(Path(str(page["path"]))) != page.get("sha256"):
            raise ValueError("visual evidence page file was replaced after locking")
    for sample in evidence.get("samples", ()):
        for asset in sample.get("assets", {}).values():
            if _sha256_file(Path(str(asset["path"]))) != asset.get("sha256"):
                raise ValueError("visual evidence image file was replaced after locking")


def _read_manifest_ids(path: Path) -> list[str]:
    ids = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id") if isinstance(row, Mapping) else None
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"{path}:{line_no}: invalid or duplicate sample_id")
        ids.append(sample_id)
        seen.add(sample_id)
    return ids


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required visual evidence file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _canonical_contract_sha256(payload: Mapping[str, Any]) -> str:
    contract = dict(payload)
    contract.pop("evidence_contract_sha256", None)
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_or_validate_bytes(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"contact sheet output must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"existing contact sheet disagrees with regenerated content: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
