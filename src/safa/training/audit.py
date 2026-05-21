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
        _audit_source_path(Path(path))


def _audit_config(config: dict, prefix: str = "") -> None:
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if str(key).lower() in FORBIDDEN_CONFIG_KEYS:
            raise RuntimeError(f"Forbidden identity supervision config key: {full_key}")
        if isinstance(value, dict):
            _audit_config(value, full_key)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _audit_config(item, f"{full_key}[{i}]")
                elif isinstance(item, str) and any(term in item.lower() for term in FORBIDDEN_TRAINING_TERMS):
                    raise RuntimeError(f"Forbidden identity supervision config value at {full_key}[{i}]: {item}")
        elif isinstance(value, str) and any(term in value.lower() for term in FORBIDDEN_TRAINING_TERMS):
            raise RuntimeError(f"Forbidden identity supervision config value at {full_key}: {value}")


def _audit_source_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cannot audit missing source path: {path}")
    if path.is_dir():
        for source_file in sorted(path.rglob("*.py")):
            if "__pycache__" in source_file.parts or _is_audit_catalog(source_file):
                continue
            _audit_source_file(source_file)
        return
    if not path.is_file():
        raise FileNotFoundError(f"Cannot audit non-file source path: {path}")
    _audit_source_file(path)


def _is_audit_catalog(path: Path) -> bool:
    return path.name == "audit.py" and len(path.parts) >= 2 and path.parts[-2:] == ("training", "audit.py")


def _audit_source_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    violations = [term for term in FORBIDDEN_TRAINING_TERMS if term in text]
    filtered = []
    for term in violations:
        if term == "identity_supervision: true":
            allowed_comments = ('identity_supervision": false',)
        else:
            allowed_comments = ()
        start = 0
        has_forbidden_occurrence = False
        while True:
            idx = text.find(term, start)
            if idx == -1:
                break
            if allowed_comments:
                context = text[max(0, idx - 60):idx + len(term) + 60]
                is_allowed = any(ac in context for ac in allowed_comments)
            else:
                is_allowed = False
            if not is_allowed:
                has_forbidden_occurrence = True
                break
            start = idx + 1
        if has_forbidden_occurrence:
            filtered.append(term)
    if filtered:
        raise RuntimeError(f"Forbidden identity supervision terms in {path}: {filtered}")
