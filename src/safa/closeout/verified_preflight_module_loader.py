"""Stdlib-only verified loader for the CPU-preflight shared contract.

This module is itself loaded from an exact repository path and a caller-pinned
SHA-256.  It never imports the target through Python's ambient import system.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import types
from typing import Any, Mapping, Sequence


class VerifiedPreflightModuleError(RuntimeError):
    """A policy-bound implementation or its live file identity differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_sealed_regular_file(
    path: Path,
    expected_sha256: str | None,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_CLOEXEC"):
        raise VerifiedPreflightModuleError(
            "verified preflight loading requires no-follow descriptors"
        )
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or path.is_symlink()
    ):
        raise VerifiedPreflightModuleError(f"{label} path is not exact")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise VerifiedPreflightModuleError(
            f"{label} cannot be opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise VerifiedPreflightModuleError(
            f"{label} cannot be read"
        ) from exc
    finally:
        os.close(descriptor)
    source = b"".join(chunks)
    identity = {
        "path": str(path),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mode": int(before.st_mode),
        "size": int(before.st_size),
    }
    after_identity = {
        "path": str(path),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mode": int(after.st_mode),
        "size": int(after.st_size),
    }
    if (
        identity != after_identity
        or not stat.S_ISREG(before.st_mode)
        or before.st_size != len(source)
        or (
            expected_sha256 is not None
            and _sha256_bytes(source) != expected_sha256
        )
    ):
        raise VerifiedPreflightModuleError(
            f"{label} file identity or SHA-256 differs"
        )
    return source, identity


def _parse_policy(
    config_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    source, identity = _read_sealed_regular_file(
        config_path,
        None,
        "preflight policy",
    )
    try:
        value = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedPreflightModuleError(
            "preflight policy is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise VerifiedPreflightModuleError(
            "preflight policy is not a mapping"
        )
    return value, source, identity


def _implementation_binding(
    policy: Mapping[str, Any],
    *,
    name: str,
    repo_root: Path,
    exact_relative_path: str,
) -> tuple[Path, str]:
    implementations = policy.get("implementations")
    raw = (
        implementations.get(name)
        if isinstance(implementations, Mapping)
        else None
    )
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"path", "sha256"}
        or raw.get("path") != exact_relative_path
    ):
        raise VerifiedPreflightModuleError(
            f"preflight implementation path differs: {name}"
        )
    expected_sha256 = raw.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        raise VerifiedPreflightModuleError(
            f"preflight implementation SHA-256 differs: {name}"
        )
    path = repo_root / exact_relative_path
    if path.parent.resolve(strict=True) != path.parent:
        raise VerifiedPreflightModuleError(
            f"preflight implementation parent path differs: {name}"
        )
    return path, expected_sha256


def load_verified_preflight_module(
    *,
    config_path: str,
    repo_root: str,
    caller_name: str,
    caller_relative_path: str,
    target_name: str,
    target_relative_path: str,
    expected_exports: Sequence[str],
) -> dict[str, Any]:
    """Load a target from sealed bytes after verifying policy and caller."""

    root = Path(repo_root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise VerifiedPreflightModuleError(
            "preflight repository root is not exact"
        )
    config = Path(config_path)
    if not config.is_absolute() or config.resolve(strict=True) != config:
        raise VerifiedPreflightModuleError(
            "preflight policy path is not exact"
        )
    policy, config_source, config_identity = _parse_policy(config)
    caller_path, caller_sha256 = _implementation_binding(
        policy,
        name=caller_name,
        repo_root=root,
        exact_relative_path=caller_relative_path,
    )
    _caller_source, caller_identity = _read_sealed_regular_file(
        caller_path,
        caller_sha256,
        f"preflight caller {caller_name}",
    )
    target_path, target_sha256 = _implementation_binding(
        policy,
        name=target_name,
        repo_root=root,
        exact_relative_path=target_relative_path,
    )
    source, target_identity = _read_sealed_regular_file(
        target_path,
        target_sha256,
        f"preflight target {target_name}",
    )
    module = types.ModuleType(
        f"_safa_verified_{target_name}_{target_sha256}"
    )
    module.__file__ = str(target_path)
    module.__package__ = "safa.closeout"
    try:
        exec(
            compile(source, str(target_path), "exec"),
            module.__dict__,
        )
    except BaseException as exc:
        raise VerifiedPreflightModuleError(
            f"verified preflight target import failed: {target_name}"
        ) from exc
    exports: dict[str, Any] = {}
    for name in expected_exports:
        if not isinstance(name, str) or not hasattr(module, name):
            raise VerifiedPreflightModuleError(
                f"verified preflight target API differs: {target_name}"
            )
        exports[name] = getattr(module, name)
    handle = {
        "module": module,
        "exports": exports,
        "binding": {
            "path": str(target_path),
            "sha256": target_sha256,
            "file_identity": dict(target_identity),
        },
        "target_path": str(target_path),
        "target_sha256": target_sha256,
        "target_identity": dict(target_identity),
        "caller_path": str(caller_path),
        "caller_sha256": caller_sha256,
        "caller_identity": dict(caller_identity),
        "config_path": str(config),
        "config_sha256": _sha256_bytes(config_source),
        "config_identity": dict(config_identity),
    }
    reverify_verified_preflight_module(handle)
    return handle


def reverify_verified_preflight_module(
    handle: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck config, caller, target inode/SHA, and bound API objects."""

    target_path = Path(str(handle["target_path"]))
    _target_source, target_identity = _read_sealed_regular_file(
        target_path,
        str(handle["target_sha256"]),
        "verified preflight target",
    )
    caller_path = Path(str(handle["caller_path"]))
    _caller_source, caller_identity = _read_sealed_regular_file(
        caller_path,
        str(handle["caller_sha256"]),
        "verified preflight caller",
    )
    config_path = Path(str(handle["config_path"]))
    config_source, config_identity = _read_sealed_regular_file(
        config_path,
        str(handle["config_sha256"]),
        "verified preflight policy",
    )
    module = handle["module"]
    exports = handle["exports"]
    if (
        target_identity != handle["target_identity"]
        or caller_identity != handle["caller_identity"]
        or config_identity != handle["config_identity"]
        or _sha256_bytes(config_source) != handle["config_sha256"]
        or any(
            getattr(module, name, None) is not value
            for name, value in exports.items()
        )
    ):
        raise VerifiedPreflightModuleError(
            "verified preflight module changed after loading"
        )
    return dict(handle["binding"])
