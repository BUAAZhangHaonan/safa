from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

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

    def test_rejects_common_identity_supervision_source_spellings(self) -> None:
        cases = {
            "arcface_class.py": "loss = ArcFaceLoss()(x)\n",
            "facenet_class.py": "loss = FaceNetLoss()(x)\n",
            "adaface_class.py": "loss = AdaFaceLoss()(x)\n",
            "loss_id.py": "loss_id = criterion(x)\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, source in cases.items():
                path = root / name
                path.write_text(source, encoding="utf-8")
                with self.subTest(name=name):
                    with self.assertRaises(RuntimeError):
                        audit_no_identity_supervision({}, [path])

    def test_rejects_identity_weight_config_key(self) -> None:
        with self.assertRaises(RuntimeError):
            audit_no_identity_supervision({"loss": {"identity_weight": 0.1}})

    def test_scans_source_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "training"
            source_dir.mkdir()
            (source_dir / "loop.py").write_text("loss = arcface_loss(x)\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "arcface_loss"):
                audit_no_identity_supervision({}, [source_dir])

    def test_train_g_from_config_audits_training_and_model_sources(self) -> None:
        from safa.training import g_loop

        captured = {}

        def fake_audit(config, source_paths=()):
            captured["source_paths"] = [str(path) for path in source_paths]
            raise RuntimeError("stop after audit")

        with patch.object(g_loop, "audit_no_identity_supervision", side_effect=fake_audit):
            with self.assertRaisesRegex(RuntimeError, "stop after audit"):
                g_loop.train_g_from_config({"seed": 1337})

        source_paths = [Path(path) for path in captured["source_paths"]]
        self.assertEqual([path.name for path in source_paths], ["training", "models"])
        self.assertTrue(all(path.is_absolute() for path in source_paths))

    def test_train_g_from_config_source_audit_is_independent_of_cwd(self) -> None:
        import os
        from safa.training import g_loop

        original_cwd = Path.cwd()
        captured = {}

        def fake_audit(config, source_paths=()):
            captured["source_paths"] = [Path(path) for path in source_paths]
            raise RuntimeError("stop after audit")

        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with patch.object(g_loop, "audit_no_identity_supervision", side_effect=fake_audit):
                    with self.assertRaisesRegex(RuntimeError, "stop after audit"):
                        g_loop.train_g_from_config({"seed": 1337})
            finally:
                os.chdir(original_cwd)

        self.assertEqual([path.name for path in captured["source_paths"]], ["training", "models"])
        self.assertTrue(all(path.is_absolute() and path.exists() for path in captured["source_paths"]))

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
