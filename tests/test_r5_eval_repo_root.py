from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_r5_eval_repo_root_defaults_to_script_and_accepts_env_override(monkeypatch, tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "r5_eval.py"
    monkeypatch.setenv("SAFA_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [str(script), "checkpoint.pt", "test", "0"])

    namespace = runpy.run_path(str(script), run_name="r5_eval_repo_root_test")

    assert namespace["REPO"] == tmp_path.resolve()
    assert namespace["resolve_repo_root"]({}, script_path=script) == REPO_ROOT
