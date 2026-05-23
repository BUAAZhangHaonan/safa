#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OUTPUT_ORDER_RULE = (
    "labels sorted ascending; each label sampled with random.Random(f'{seed}:{label}') "
    "from source-order rows; combined rows shuffled with random.Random(f'{seed}:output')"
)
DEFAULT_EXPECTED_LABELS = tuple(range(8))
DEFAULT_EXPECTED_LABELS_ARG = ",".join(str(label) for label in DEFAULT_EXPECTED_LABELS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sample_ids(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_expected_labels(value: str) -> tuple[int, ...]:
    labels: list[int] = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            raise argparse.ArgumentTypeError("--expected-labels must be a comma-separated list of integers")
        try:
            labels.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--expected-labels contains non-integer label {token!r}") from exc
    if not labels:
        raise argparse.ArgumentTypeError("--expected-labels must include at least one label")
    duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicates:
        details = ", ".join(f"label {label}" for label in duplicates)
        raise argparse.ArgumentTypeError(f"--expected-labels contains duplicate label(s): {details}")
    return tuple(sorted(labels))


def read_source_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[Any] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            missing = {"sample_id", "label"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing required field(s): {sorted(missing)}")
            label = row["label"]
            if not isinstance(label, int) or isinstance(label, bool):
                raise ValueError(f"{path}:{line_no}: label must be int for sample {row['sample_id']!r}")
            sample_id = row["sample_id"]
            if sample_id in seen_sample_ids:
                raise ValueError(f"{path}:{line_no}: duplicate sample_id {sample_id!r}")
            seen_sample_ids.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: source index contains no rows")
    return rows


def build_balanced_rows(
    rows: list[dict[str, Any]],
    samples_per_class: int,
    seed: int,
    expected_labels: tuple[int, ...] = DEFAULT_EXPECTED_LABELS,
) -> tuple[list[dict[str, Any]], list[int]]:
    if samples_per_class <= 0:
        raise ValueError("--samples-per-class must be a positive integer")

    rows_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["label"]].append(row)

    label_order = list(expected_labels)
    expected_label_set = set(label_order)
    missing = [label for label in label_order if label not in rows_by_label]
    unexpected = sorted(label for label in rows_by_label if label not in expected_label_set)
    label_errors: list[str] = []
    if missing:
        details = ", ".join(f"label {label}" for label in missing)
        label_errors.append(f"missing expected label(s): {details}")
    if unexpected:
        details = ", ".join(f"label {label}" for label in unexpected)
        label_errors.append(f"unexpected label(s): {details}")
    if label_errors:
        expected = ", ".join(str(label) for label in label_order)
        raise ValueError(f"source labels must match expected labels [{expected}]: {'; '.join(label_errors)}")

    insufficient = [
        (label, len(rows_by_label[label]))
        for label in label_order
        if len(rows_by_label[label]) < samples_per_class
    ]
    if insufficient:
        details = ", ".join(f"label {label}: {count} available" for label, count in insufficient)
        raise ValueError(f"class count below --samples-per-class ({samples_per_class}): {details}")

    selected: list[dict[str, Any]] = []
    for label in label_order:
        rng = random.Random(f"{seed}:{label}")
        selected.extend(rng.sample(rows_by_label[label], samples_per_class))

    random.Random(f"{seed}:output").shuffle(selected)
    return selected, label_order


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_path_for(output_index: Path) -> Path:
    return output_index.with_name(f"{output_index.stem}_manifest.json")


def write_manifest(
    *,
    source_index: Path,
    output_index: Path,
    selected_rows: list[dict[str, Any]],
    label_order: list[int],
    seed: int,
    samples_per_class: int,
) -> Path:
    class_counts = Counter(row["label"] for row in selected_rows)
    manifest_path = manifest_path_for(output_index)
    manifest = {
        "schema_version": 1,
        "source_index": str(source_index),
        "source_index_sha256": sha256_file(source_index),
        "output_index": str(output_index),
        "output_index_sha256": sha256_file(output_index),
        "ordered_sample_id_sha256": sha256_sample_ids(selected_rows),
        "seed": seed,
        "samples_per_class": samples_per_class,
        "class_counts": {str(label): class_counts[label] for label in label_order},
        "num_samples": len(selected_rows),
        "label_order": label_order,
        "output_order_rule": OUTPUT_ORDER_RULE,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_balanced_index(
    source_index: Path,
    output_index: Path,
    samples_per_class: int,
    seed: int,
    expected_labels: tuple[int, ...] = DEFAULT_EXPECTED_LABELS,
) -> Path:
    rows = read_source_index(source_index)
    selected_rows, label_order = build_balanced_rows(rows, samples_per_class, seed, expected_labels)
    write_jsonl(selected_rows, output_index)
    return write_manifest(
        source_index=source_index,
        output_index=output_index,
        selected_rows=selected_rows,
        label_order=label_order,
        seed=seed,
        samples_per_class=samples_per_class,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic balanced JSONL index.")
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--samples-per-class", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--expected-labels",
        default=DEFAULT_EXPECTED_LABELS_ARG,
        type=parse_expected_labels,
        metavar="CSV",
        help=f"Comma-separated expected labels. Default: {DEFAULT_EXPECTED_LABELS_ARG}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_balanced_index(
            source_index=args.source_index,
            output_index=args.output_index,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            expected_labels=args.expected_labels,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
