from __future__ import annotations

from pathlib import Path
from typing import Iterable


FORBIDDEN_TRAINING_TERMS = (
    "identity_loss",
    "id_loss",
    "face_recognition_loss",
    "arcface_loss",
    "facenet_loss",
    "adaface_loss",
    "magface_loss",
    "identity_supervision: true",
)

FORBIDDEN_CONFIG_KEYS = {
    "identity_loss",
    "id_loss",
    "face_recognition_loss",
    "arcface_loss",
    "facenet_loss",
    "adaface_loss",
    "magface_loss",
    "identity_weight",
    "id_weight",
}


def audit_no_identity_supervision(config: dict, source_paths: Iterable[str | Path] = ()) -> None:
    _audit_config(config)
    for path in source_paths:
        _audit_source(Path(path))


def _audit_config(config: dict, prefix: str = "") -> None:
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if str(key).lower() in FORBIDDEN_CONFIG_KEYS:
            raise RuntimeError(f"Forbidden identity supervision config key: {full_key}")
        if isinstance(value, dict):
            _audit_config(value, full_key)
        elif isinstance(value, str) and any(term in value.lower() for term in FORBIDDEN_TRAINING_TERMS):
            raise RuntimeError(f"Forbidden identity supervision config value at {full_key}: {value}")


def _audit_source(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot audit missing source file: {path}")
    text = path.read_text(encoding="utf-8").lower()
    violations = [term for term in FORBIDDEN_TRAINING_TERMS if term in text]
    allowed_comments = ("identity_supervision\": false", '"identity_supervision": false')
    filtered = [term for term in violations if term not in allowed_comments]
    if filtered:
        raise RuntimeError(f"Forbidden identity supervision terms in {path}: {filtered}")

