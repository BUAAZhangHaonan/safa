from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from safa.training.audit import audit_no_identity_supervision


class NoIdentityAuditTests(unittest.TestCase):
    def test_rejects_identity_loss_config_key(self) -> None:
        with self.assertRaises(RuntimeError):
            audit_no_identity_supervision({"loss_weights": {"identity_loss": 1.0}})

    def test_rejects_forbidden_source_term(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            path.write_text("loss = arcface_loss(x)\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                audit_no_identity_supervision({}, [path])

    def test_accepts_current_style_config(self) -> None:
        audit_no_identity_supervision(
            {
                "generator": {"model_type": "conditional_flow_matching", "sample_steps": 8},
                "stages": {
                    "stage1": {"epochs": 1},
                    "stage2": {"epochs": 1, "lambda_initial": 0.005, "lambda_max": 0.05, "lambda_growth": 0.005},
                },
            }
        )


if __name__ == "__main__":
    unittest.main()
