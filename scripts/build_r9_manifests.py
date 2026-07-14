#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = Path("configs/medium_v2/experiments/r9_manifests")
SOURCE_SNAPSHOT = MANIFEST_ROOT / "r8_manifest_source_snapshot.json"
CLEAN_INDEX = Path("data/index/val_single_face_privacy_clean_arcface.jsonl")
EXPECTED_CLEAN_SHA256 = (
    "c425eedb7e2939bf89782d9a0ecbb9fa896208a55546f895e673354b16432354"
)
EXPECTED_SOURCE_FILE_SHA256 = (
    "0e70ad691df801752faebdfc9c7b615f0ae27a7d58bb4d64fc38af6f5bb21329"
)
EXPECTED_SOURCE_BINDINGS = {
    "r8_calibration_64_sha256": "b030b23ab5e688f709213f4671c1b12c2f53905a488909882234f0d5688b1a63",
    "r8_visual_review_sha256": "a6dc4bd1a9d6c5b99e7fc65419948be92109859ecb1733619f6fbb12d67ffc14",
    "r8_native_per_sample_sha256": "9aaa7d7eb330c5090b0b797e2355c728fdc17a5eec4c6c1df0a6ec2b3fffcd8c",
}
EXPECTED_MANIFESTS = {
    "diagnose_18": {
        "count": 18,
        "file_sha256": "e41b68999939f3ca53a60cb4d7a2452f75c4cd103a8bd6ac7136a0a5f08a0aa3",
        "ordered_sha256": "c2ba8aed127b392aadabdb65b19253fa7dd7413337304b33d30d9692c56842f8",
    },
    "calibration_64": {
        "count": 64,
        "file_sha256": "ffc1f04f671533ee1498f4b03565826920afcc4e5c6ab244fc6f9b7aa680f964",
        "ordered_sha256": "0e9dc5bd1da3c265efe4d66959cdc6649a6b60b82c29058adf0dab843b7c1df3",
    },
    "validate_512": {
        "count": 512,
        "file_sha256": "b05bc43a5ad03b77567ef1e1053c4d97ae47b016e8ec149d8a5cd49040e0e391",
        "ordered_sha256": "46872147bedc0f796e8ae3ca1a59083d375fea6ada38f653b6eaab7d30156816",
    },
    "full_2048": {
        "count": 2048,
        "file_sha256": "ba760481941505b2d519951b8077026cdc0cdf97bcd44b5eac7e0e9329b36b68",
        "ordered_sha256": "6462edef5fa4b19f970c35038baf206a906c5ae9c3abc7c342354dcec614b1fc",
    },
    "full_visual_64": {
        "count": 64,
        "file_sha256": "16955700ab7630d5029dcbf815fe0e0ca77a03e591297fc54595c9031b23b9b5",
        "ordered_sha256": "527fd0695f7b9578f90c62eb42a0cb6dc23f14be2b886e08a61be61c9bbb63a0",
    },
}
EXPECTED_MATCHED_PAIR_SHA256 = (
    "080f45a7d6f108afa903df4e03d8773a198b83d878e435b8cf128436bcbc5c24"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild or verify the five tracked SAFA R9 sample manifests."
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def ordered_id_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes("".join(f"{row['sample_id']}\n" for row in rows).encode())


def rank_ids(sample_ids: Sequence[str], domain: str) -> list[str]:
    return sorted(
        sample_ids,
        key=lambda sample_id: (
            hashlib.sha256(f"{domain}\0{sample_id}".encode()).digest(),
            sample_id,
        ),
    )


def load_source_snapshot(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / SOURCE_SNAPSHOT
    if sha256_path(path) != EXPECTED_SOURCE_FILE_SHA256:
        raise ValueError("tracked R8 manifest source snapshot SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "contract_type",
        "source_bindings",
        "samples",
        "source_snapshot_sha256",
    }:
        raise ValueError("tracked R8 manifest source snapshot fields are not canonical")
    if payload["schema_version"] != 1 or payload["contract_type"] != (
        "safa_r9_manifest_source_snapshot_v1"
    ):
        raise ValueError("tracked R8 manifest source snapshot contract mismatch")
    declared = str(payload.pop("source_snapshot_sha256"))
    if sha256_bytes(canonical_json(payload)) != declared:
        raise ValueError("tracked R8 manifest source snapshot contract SHA256 mismatch")
    payload["source_snapshot_sha256"] = declared
    if payload["source_bindings"] != EXPECTED_SOURCE_BINDINGS:
        raise ValueError("tracked R8 source evidence bindings changed")
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) != 64:
        raise ValueError("tracked R8 source snapshot must contain 64 samples")
    normalized: list[dict[str, Any]] = []
    for row in samples:
        if not isinstance(row, dict) or set(row) != {
            "sample_id",
            "native_e0_cosine",
            "paper_split_eta0.25_severe",
        }:
            raise ValueError("tracked R8 sample source fields are not canonical")
        sample_id = row["sample_id"]
        cosine = row["native_e0_cosine"]
        severe = row["paper_split_eta0.25_severe"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("tracked R8 sample ID must be non-empty")
        if isinstance(cosine, bool) or not isinstance(cosine, (int, float)):
            raise ValueError("tracked R8 native E0 cosine must be numeric")
        if not math.isfinite(float(cosine)) or not isinstance(severe, bool):
            raise ValueError("tracked R8 sample evidence is invalid")
        normalized.append(
            {
                "sample_id": sample_id,
                "native_e0_cosine": float(cosine),
                "paper_split_eta0.25_severe": severe,
            }
        )
    if len({row["sample_id"] for row in normalized}) != 64:
        raise ValueError("tracked R8 source snapshot repeats sample IDs")
    if sum(bool(row["paper_split_eta0.25_severe"]) for row in normalized) != 9:
        raise ValueError("tracked R8 source snapshot must bind exactly nine severe IDs")
    return normalized


def build_rows(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    clean_path = repo_root / CLEAN_INDEX
    if sha256_path(clean_path) != EXPECTED_CLEAN_SHA256:
        raise ValueError("tracked ArcFace-clean index SHA256 mismatch")
    clean_ids = [str(row["sample_id"]) for row in read_jsonl(clean_path)]
    if len(clean_ids) != 3968 or len(set(clean_ids)) != len(clean_ids):
        raise ValueError("tracked ArcFace-clean index must contain 3968 unique IDs")
    source = load_source_snapshot(repo_root)
    calibration_ids = [str(row["sample_id"]) for row in source]
    if not set(calibration_ids) <= set(clean_ids):
        raise ValueError("R8 calibration IDs are not ArcFace-clean")

    severe = [row for row in source if row["paper_split_eta0.25_severe"]]
    controls = [row for row in source if not row["paper_split_eta0.25_severe"]]
    costs = [
        [
            abs(
                float(difficult["native_e0_cosine"])
                - float(control["native_e0_cosine"])
            )
            for control in controls
        ]
        for difficult in severe
    ]
    severe_indices, control_indices = linear_sum_assignment(costs)
    if list(severe_indices) != list(range(9)):
        raise ValueError("diagnose matching did not preserve difficult-row order")
    diagnose: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for pair_index, (difficult, control_index) in enumerate(
        zip(severe, control_indices, strict=True)
    ):
        control = controls[int(control_index)]
        difficult_id = str(difficult["sample_id"])
        control_id = str(control["sample_id"])
        difficult_e0 = float(difficult["native_e0_cosine"])
        control_e0 = float(control["native_e0_cosine"])
        diagnose.extend(
            [
                {
                    "matched_control_sample_id": control_id,
                    "native_e0_cosine": difficult_e0,
                    "pair_index": pair_index,
                    "role": "difficult",
                    "sample_id": difficult_id,
                },
                {
                    "matched_difficult_sample_id": difficult_id,
                    "native_e0_cosine": control_e0,
                    "pair_index": pair_index,
                    "role": "control",
                    "sample_id": control_id,
                },
            ]
        )
        pairs.append(
            {
                "pair_index": pair_index,
                "difficult_sample_id": difficult_id,
                "control_sample_id": control_id,
                "difficult_native_e0_cosine": difficult_e0,
                "control_native_e0_cosine": control_e0,
            }
        )
    pair_sha = sha256_bytes(canonical_json({"schema_version": 1, "pairs": pairs}))
    if pair_sha != EXPECTED_MATCHED_PAIR_SHA256:
        raise ValueError("diagnose matched-pair SHA256 changed")

    available = [item for item in clean_ids if item not in set(calibration_ids)]
    validate_ids = rank_ids(available, "safa-r9-validate-512-v1")[:512]
    remaining = [item for item in available if item not in set(validate_ids)]
    full_ids = rank_ids(remaining, "safa-r9-full-2048-v1")[:2048]
    visual_ids = rank_ids(full_ids, "safa-r9-full-visual-64-v1")[:64]
    return {
        "diagnose_18": diagnose,
        "calibration_64": [{"sample_id": item} for item in calibration_ids],
        "validate_512": [{"sample_id": item} for item in validate_ids],
        "full_2048": [{"sample_id": item} for item in full_ids],
        "full_visual_64": [
            {"full_index": full_ids.index(item), "sample_id": item}
            for item in visual_ids
        ],
    }


def write_new(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to replace immutable manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_or_write(repo_root: Path, *, write: bool) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for name, rows in build_rows(repo_root).items():
        contract = EXPECTED_MANIFESTS[name]
        content = canonical_jsonl(rows)
        actual = {
            "count": len(rows),
            "file_sha256": sha256_bytes(content),
            "ordered_sha256": ordered_id_sha256(rows),
        }
        if actual != contract:
            raise ValueError(f"generated {name} contract changed: {actual}")
        path = repo_root / MANIFEST_ROOT / f"{name}.jsonl"
        if write:
            write_new(path, content)
        elif not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"tracked manifest does not reproduce exactly: {path}")
        outputs[name] = {**actual, "path": str(path.relative_to(repo_root))}
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    result = verify_or_write(repo_root, write=bool(args.write))
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
