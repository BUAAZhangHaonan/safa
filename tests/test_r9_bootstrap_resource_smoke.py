from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_r9_bootstrap_resource_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("r9_bootstrap_resource_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _patch_smi(monkeypatch: pytest.MonkeyPatch, rows: str) -> None:
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=rows),
    )


def test_gpu_snapshots_select_only_registered_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_smi(
        monkeypatch,
        "\n".join(
            f"{index}, GPU-{index}, {24000 - index}, 24576" for index in range(7)
        ),
    )

    snapshots = MODULE._gpu_snapshots()

    assert list(snapshots) == [0, 1, 2, 3]
    assert {row[0] for row in snapshots.values()} == {
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    }


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            "0, GPU-0, 24000, 24576\n1, GPU-1, 24000, 24576\n"
            "2, GPU-2, 24000, 24576\n4, GPU-4, 24000, 24576",
            "requires registered physical GPUs",
        ),
        (
            "0, GPU-X, 24000, 24576\n1, GPU-X, 24000, 24576\n"
            "2, GPU-2, 24000, 24576\n3, GPU-3, 24000, 24576",
            "UUIDs must be unique",
        ),
        (
            "0, GPU-0, 24000, 24576\n0, GPU-X, 24000, 24576\n"
            "1, GPU-1, 24000, 24576\n2, GPU-2, 24000, 24576\n"
            "3, GPU-3, 24000, 24576",
            "duplicate GPU index",
        ),
    ],
)
def test_gpu_snapshots_reject_invalid_registered_binding(
    monkeypatch: pytest.MonkeyPatch, rows: str, message: str
) -> None:
    _patch_smi(monkeypatch, rows)

    with pytest.raises(MODULE.BootstrapError, match=message):
        MODULE._gpu_snapshots()
