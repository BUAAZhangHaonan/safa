from __future__ import annotations

import json
from pathlib import Path

import pytest

from safa.evaluation.r8_visual_evidence import (
    build_visual_evidence_contract,
    validate_visual_review_arm,
    write_contact_sheets,
)


def _fixture(tmp_path: Path):
    ids = [f"sample-{index:02d}" for index in range(64)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in ids),
        encoding="utf-8",
    )
    rows = []
    for index, sample_id in enumerate(ids):
        assets = {}
        for column in ("source", "native", "candidate"):
            path = tmp_path / f"{column}-{index:02d}.png"
            path.write_bytes(f"{column}-{sample_id}".encode())
            assets[column] = str(path)
        rows.append({"sample_id": sample_id, **assets})
    pages = []
    for page_index in range(8):
        page = tmp_path / f"page-{page_index}.png"
        page.write_bytes(f"page-{page_index}".encode())
        pages.append(
            {
                "page_index": page_index,
                "path": str(page),
                "sample_ids": ids[page_index * 8 : (page_index + 1) * 8],
            }
        )
    evidence = build_visual_evidence_contract(
        manifest_path=manifest,
        rows=rows,
        pages=pages,
        columns=("source", "native", "candidate"),
        expected_count=64,
    )
    review = {
        "evidence_contract_sha256": evidence["evidence_contract_sha256"],
        "pages": evidence["pages"],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "assets": sample["assets"],
                "page": sample["page"],
                "decision": "pass",
            }
            for sample in evidence["samples"]
        ],
        "severe_failure_count": 0,
        "passed": True,
    }
    return evidence, review


def test_visual_review_binds_every_asset_page_and_decision(tmp_path: Path) -> None:
    evidence, review = _fixture(tmp_path)

    normalized = validate_visual_review_arm(review, evidence)

    assert normalized["reviewed_sample_count"] == 64
    assert normalized["severe_failure_count"] == 0
    assert normalized["passed"] is True


@pytest.mark.parametrize("tamper", ["order", "asset", "page", "decision"])
def test_visual_review_rejects_any_unbound_or_replaced_evidence(
    tmp_path: Path, tamper: str
) -> None:
    evidence, review = _fixture(tmp_path)
    if tamper == "order":
        review["samples"][0], review["samples"][1] = (
            review["samples"][1],
            review["samples"][0],
        )
    elif tamper == "asset":
        review["samples"][0]["assets"]["candidate"]["sha256"] = "0" * 64
    elif tamper == "page":
        review["samples"][0]["page"]["sha256"] = "0" * 64
    else:
        review["samples"][0]["decision"] = "looks_ok"

    with pytest.raises(ValueError):
        validate_visual_review_arm(review, evidence)


def test_contact_sheets_reject_symlink_output_escape(tmp_path: Path) -> None:
    from PIL import Image

    assets = []
    for name in ("source", "native", "candidate"):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (4, 4), color="white").save(path)
        assets.append(path)
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "contact"
    output.symlink_to(outside, target_is_directory=True)
    rows = [
        {
            "sample_id": "sample",
            "source": str(assets[0]),
            "native": str(assets[1]),
            "candidate": str(assets[2]),
        }
    ]

    with pytest.raises(ValueError, match="symlink"):
        write_contact_sheets(output, rows, columns=("source", "native", "candidate"))
    assert not list(outside.iterdir())
