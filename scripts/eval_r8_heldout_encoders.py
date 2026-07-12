#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from safa.evaluation.encoder_generalization import (
    EncoderBatch,
    multi_encoder_median,
    within_encoder_generalization,
)


CHECKPOINT_SHA256 = "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d"
FULL_MANIFEST_FILE_SHA256 = "7f830ad3f84089bcf83d092fbffaf2b5c3335cf68a4b397f04b65f362f79ae5b"
ENCODERS = {
    "e1_dinov2_large_v2": {
        "path": Path("artifacts/checkpoints/e0_dinov2_large_v2/best.pt"),
        "sha256": "cce0de2f1eab097cb6091886f587a9f334dd84ced1ca4dd5e08c3a765718a14c",
    },
    "e2_convnext_tiny": {
        "path": Path("artifacts/checkpoints/e0_convnext_tiny/best.pt"),
        "sha256": "09c88bd416057222abefeba52ebe88d710715ede791ec34198a23ae5e6e850a8",
    },
}


def validate_heldout_contract(
    selection: Mapping[str, Any],
    native_manifest: Mapping[str, Any],
    winner_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    winner = selection.get("winner")
    if not isinstance(winner, Mapping) or not winner.get("arm_id"):
        raise ValueError("held-out evaluation requires a locked winner")
    if selection.get("winner_locked_before_heldout") is not True:
        raise ValueError("winner must be locked before held-out evaluation")
    if selection.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("held-out selection checkpoint SHA256 mismatch")
    if selection.get("full_sample_count") != 2048:
        raise ValueError("held-out selection must lock exactly 2048 samples")
    manifest_file_sha256 = _sha(
        selection.get("full_sample_id_manifest_sha256"),
        "selection full manifest file SHA256",
    )
    if manifest_file_sha256 != FULL_MANIFEST_FILE_SHA256:
        raise ValueError("selection full manifest file SHA256 is not the registered manifest")
    expected_digest = _sha(
        selection.get("full_sample_id_sha256"), "selection full sample-ID digest"
    )
    for label, payload in (("native", native_manifest), ("winner", winner_manifest)):
        if payload.get("sample_count") != 2048:
            raise ValueError(f"held-out {label} manifest must contain exactly 2048 samples")
        if _sha(payload.get("sample_id_sha256"), f"{label} sample-ID digest") != expected_digest:
            raise ValueError(f"held-out {label} sample-ID digest disagrees with locked selection")
        _sha(payload.get("per_sample_sha256"), f"{label} per-sample SHA256")
        _sha(
            payload.get("ordered_image_manifest_sha256"),
            f"{label} ordered image manifest SHA256",
        )
    contract = {
        "winner_arm_id": str(winner["arm_id"]),
        "winner_config_sha256": _sha(winner.get("config_sha256"), "winner config SHA256"),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_count": 2048,
        "sample_id_sha256": expected_digest,
        "sample_id_manifest_sha256": manifest_file_sha256,
        "native": {
            "per_sample_sha256": native_manifest["per_sample_sha256"],
            "ordered_image_manifest_sha256": native_manifest[
                "ordered_image_manifest_sha256"
            ],
        },
        "winner": {
            "per_sample_sha256": winner_manifest["per_sample_sha256"],
            "ordered_image_manifest_sha256": winner_manifest[
                "ordered_image_manifest_sha256"
            ],
        },
    }
    contract["contract_sha256"] = _contract_sha256(contract)
    return contract


def claim_protocol_marker(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    marker = {
        "schema_version": 1,
        "status": "started",
        "started_at": _timestamp(),
        "contract_sha256": _sha(contract.get("contract_sha256"), "contract SHA256"),
        "contract": dict(contract),
        "encoders": {
            name: {"path": str(metadata["path"]), "sha256": metadata["sha256"]}
            for name, metadata in ENCODERS.items()
        },
    }
    try:
        _write_exclusive_json(path, marker)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing a second E1/E2 prospective evaluation; protocol marker exists: {path}"
        ) from exc
    return marker


def evaluate_heldout(root: Path, *, device: str, batch_size: int) -> dict[str, Any]:
    root = Path(root)
    marker_path = root / "heldout_protocol_marker.json"
    output = root / "heldout_e1_e2.json"
    if marker_path.exists() or output.exists():
        raise FileExistsError("refusing a second or stale held-out evaluation artifact")
    selection = _read_json(root / "selection.json", "locked selection")
    finalization_path = root / "full/finalization_completion.json"
    finalization = _read_json(finalization_path, "full finalization completion")
    if finalization.get("status") != "complete":
        raise ValueError("held-out evaluation requires completed full visual/quality finalization")
    manifest_path = root / "manifests/full_2048.jsonl"
    if _sha256_file(manifest_path) != FULL_MANIFEST_FILE_SHA256:
        raise ValueError("full 2048 manifest file digest mismatch")
    sample_ids = _read_manifest_ids(manifest_path)
    if len(sample_ids) != 2048:
        raise ValueError("held-out evaluation requires exactly 2048 manifest IDs")
    index_rows = _read_jsonl(Path("data/index/val_face_mixed_e14.jsonl"))
    index_by_id = _unique_rows(index_rows, "real index")
    if set(sample_ids) - set(index_by_id):
        raise ValueError("real index does not cover every held-out manifest ID")
    native_path = root / "full/merged/native/per_sample.jsonl"
    winner_path = root / "full/merged/winner/per_sample.jsonl"
    native_rows, native_manifest = read_generated_evidence(native_path, sample_ids)
    winner_rows, winner_manifest = read_generated_evidence(winner_path, sample_ids)
    contract = validate_heldout_contract(selection, native_manifest, winner_manifest)
    winner_config = Path(str(selection["winner"].get("config", "")))
    if _sha256_file(winner_config) != contract["winner_config_sha256"]:
        raise ValueError("locked winner config file SHA256 mismatch")
    contract.update(
        {
            "sample_id_manifest": str(manifest_path),
            "sample_id_manifest_sha256": FULL_MANIFEST_FILE_SHA256,
            "native_per_sample": str(native_path),
            "winner_per_sample": str(winner_path),
            "full_finalization_sha256": _sha256_file(finalization_path),
        }
    )
    contract["contract_sha256"] = _contract_sha256(contract)
    for name, metadata in ENCODERS.items():
        if _sha256_file(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"fixed held-out encoder SHA256 mismatch: {name}")
    marker = claim_protocol_marker(marker_path, contract)

    source_paths = [Path(str(index_by_id[sample_id]["image_path"])) for sample_id in sample_ids]
    labels = torch.tensor([int(index_by_id[sample_id]["label"]) for sample_id in sample_ids])
    native_paths = [Path(str(native_rows[sample_id]["generated"])) for sample_id in sample_ids]
    winner_paths = [Path(str(winner_rows[sample_id]["generated"])) for sample_id in sample_ids]
    results: dict[str, Any] = {}
    try:
        for name, metadata in ENCODERS.items():
            source, native, winner_batch = extract_encoder_batches(
                name,
                metadata["path"],
                (source_paths, native_paths, winner_paths),
                device=device,
                batch_size=batch_size,
            )
            results[name] = {
                "checkpoint": {
                    "path": str(metadata["path"]),
                    "sha256": metadata["sha256"],
                },
                "native": within_encoder_generalization(source, native, labels),
                "winner": within_encoder_generalization(source, winner_batch, labels),
            }
            del source, native, winner_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        payload = {
            "schema_version": 1,
            "protocol": "prospective_once_after_winner_and_2048_manifest_lock",
            "contract": contract,
            "encoders": results,
            "multi_encoder_median": {
                "native": multi_encoder_median(
                    {name: payload["native"] for name, payload in results.items()}
                ),
                "winner": multi_encoder_median(
                    {name: payload["winner"] for name, payload in results.items()}
                ),
            },
        }
        _write_exclusive_json(output, payload)
        marker.update(
            {
                "status": "complete",
                "completed_at": _timestamp(),
                "contract_sha256": contract["contract_sha256"],
                "result": str(output),
                "result_sha256": _sha256_file(output),
            }
        )
        _atomic_write_json(marker_path, marker)
        return payload
    except Exception as exc:
        marker.update(
            {
                "status": "failed_after_claim",
                "failed_at": _timestamp(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(marker_path, marker)
        raise


def extract_encoder_batch(
    encoder_name: str,
    checkpoint: Path,
    image_paths: Sequence[Path],
    *,
    device: str,
    batch_size: int,
) -> EncoderBatch:
    return extract_encoder_batches(
        encoder_name,
        checkpoint,
        (image_paths,),
        device=device,
        batch_size=batch_size,
    )[0]


def extract_encoder_batches(
    encoder_name: str,
    checkpoint: Path,
    image_path_groups: Sequence[Sequence[Path]],
    *,
    device: str,
    batch_size: int,
) -> tuple[EncoderBatch, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not image_path_groups or any(not paths for paths in image_path_groups):
        raise ValueError("each encoder image-path group must be non-empty")
    from safa.models.e0 import freeze_e0, load_e0_checkpoint
    from safa.training.transforms import eval_transform

    model, _ = load_e0_checkpoint(checkpoint, device="cpu")
    freeze_e0(model)
    model = model.to(device)
    transform = eval_transform(224)
    results = []
    for image_paths in image_path_groups:
        embeddings = []
        logits = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                paths = image_paths[start : start + batch_size]
                images = torch.stack(
                    [transform(Image.open(path).convert("RGB")) for path in paths]
                ).to(device)
                output = model(images)
                embeddings.append(output["embedding"].detach().cpu())
                logits.append(output["logits"].detach().cpu())
        results.append(
            EncoderBatch(
                encoder_name=encoder_name,
                embeddings=torch.cat(embeddings),
                logits=torch.cat(logits),
            )
        )
    del model
    return tuple(results)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-shot prospective E1/E2 evaluation for locked R8 outputs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate_heldout(args.root, device=args.device, batch_size=args.batch_size)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def read_generated_evidence(
    path: Path, expected_ids: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ordered_rows = _read_jsonl(path)
    actual_ids = [str(row.get("sample_id", "")) for row in ordered_rows]
    if actual_ids != list(expected_ids):
        raise ValueError(f"generated per-sample order does not exactly match locked manifest: {path}")
    rows = _unique_rows(ordered_rows, str(path))
    image_manifest_lines = []
    for sample_id in expected_ids:
        generated = rows[str(sample_id)].get("generated")
        if not isinstance(generated, str) or not Path(generated).is_file():
            raise FileNotFoundError(f"generated image is missing for {sample_id!r}: {generated!r}")
        image_manifest_lines.append(
            f"{sample_id}\t{generated}\t{_sha256_file(Path(generated))}\n"
        )
    evidence = {
        "sample_count": len(ordered_rows),
        "sample_id_sha256": _sample_id_digest([str(value) for value in expected_ids]),
        "per_sample_sha256": _sha256_file(path),
        "ordered_image_manifest_sha256": hashlib.sha256(
            "".join(image_manifest_lines).encode("utf-8")
        ).hexdigest(),
    }
    return rows, evidence


def _unique_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} contains an invalid sample_id")
        if sample_id in result:
            raise ValueError(f"{label} contains duplicate sample_id {sample_id!r}")
        result[sample_id] = dict(row)
    return result


def _read_manifest_ids(path: Path) -> list[str]:
    rows = _read_jsonl(path)
    by_id = _unique_rows(rows, "full sample manifest")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(by_id):
        raise AssertionError("duplicate manifest IDs escaped validation")
    return ids


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL does not exist: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{sample_id}\n" for sample_id in sample_ids).encode()).hexdigest()


def _contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required locked file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        os.write(
            fd,
            (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
