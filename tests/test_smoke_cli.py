from __future__ import annotations

from argparse import Namespace
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from safa.data.index_schema import IndexRecord


class SmokeCliTests(unittest.TestCase):
    def test_smoke_passes_cached_feature_dim_to_generator_training(self) -> None:
        from safa.cli import smoke

        captured: dict = {}
        records = [
            IndexRecord(
                sample_id=f"val:{label}",
                image_path="/tmp/unused.jpg",
                label=label,
                split="val",
                dataset_root="/tmp",
                dataset_version="unit",
            )
            for label in range(8)
        ]

        def fake_train_g_from_config(config: dict) -> dict:
            captured.update(config)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "smoke.json"
            config_path.write_text(
                json.dumps(
                    {
                        "seed": 1,
                        "device": "cpu",
                        "num_workers": 1,
                        "batch_size": 2,
                        "image_size": 64,
                        "limit": 8,
                        "root": tmp,
                        "work_dir": str(Path(tmp) / "work"),
                        "e0_checkpoint": str(Path(tmp) / "e0.pt"),
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(smoke, "parse_args", return_value=Namespace(config=str(config_path))), patch.object(
                smoke, "build_affectnet_index", return_value=records
            ), patch.object(smoke, "cache_e0_from_config", return_value={"feature_dim": 128}), patch.object(
                smoke, "train_g_from_config", side_effect=fake_train_g_from_config
            ):
                smoke.main()

        self.assertEqual(captured["embedding_dim"], 128)
        required_generator_fields = {
            "embedding_dim",
            "image_size",
            "base_channels",
            "channel_multipliers",
            "condition_dim",
            "sample_steps",
            "train_cycle_steps",
            "sampler",
        }
        self.assertFalse(required_generator_fields - set(captured["generator"]))
        self.assertEqual(captured["generator"]["embedding_dim"], 128)
        self.assertEqual(captured["generator"]["image_size"], 64)

    def test_smoke_yaml_uses_positive_num_workers_required_by_training(self) -> None:
        config = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))

        self.assertGreaterEqual(config["num_workers"], 1)

    def test_runbook_referenced_smoke_script_exists(self) -> None:
        runbook = Path("docs/4029_runbook.md").read_text(encoding="utf-8")
        script = Path("scripts/run_smoke_tmux.sh")

        self.assertIn(str(script), runbook)
        self.assertTrue(script.is_file())


if __name__ == "__main__":
    unittest.main()
