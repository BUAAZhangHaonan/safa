from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import yaml


class TrainGV2BestEntrypointTests(unittest.TestCase):
    def test_v2_best_stage1_config_is_complete_prerequisite_for_v2_best(self) -> None:
        path = Path("configs/train_g_v2_best_stage1.yaml")
        self.assertTrue(path.is_file())
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        v2_best = yaml.safe_load(Path("configs/train_g_v2_best.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["out_dir"], "artifacts/checkpoints/g_v2_best_stage1")
        self.assertEqual(v2_best["resume_from"], f"{config['out_dir']}/best_stage1.pt")
        self.assertNotIn("resume_from", config)
        self.assertEqual(config["stages"]["stage1"]["epochs"], 80)
        self.assertEqual(config["stages"]["stage2"]["epochs"], 0)
        self.assertEqual(config["generator"]["base_channels"], 32)
        self.assertEqual(config["sampling_seed"], v2_best["sampling_seed"])
        self.assertEqual(config["generator"]["cycle_steps_schedule"], [4, 8, 16, 32])
        self.assertEqual(config["train_index"], v2_best["train_index"])
        self.assertEqual(config["train_features"], v2_best["train_features"])
        self.assertEqual(config["e0_checkpoint"], v2_best["e0_checkpoint"])

        from safa.cli import train_g

        captured: dict = {}

        def fake_train_g_from_config(runner_config: dict) -> dict:
            captured.update(runner_config)
            return {"ok": True}

        with patch.object(train_g, "parse_args", return_value=Namespace(config=str(path))), patch.object(
            train_g, "train_g_from_config", side_effect=fake_train_g_from_config
        ):
            train_g.main()

        self.assertEqual(captured["out_dir"], "artifacts/checkpoints/g_v2_best_stage1")

    def test_v2_best_config_documents_fixed_low_lambda_recipe(self) -> None:
        path = Path("configs/train_g_v2_best.yaml")
        self.assertTrue(path.is_file())
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        stage1_config = yaml.safe_load(Path("configs/train_g_v2_best_stage1.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["batch_size"], 16)
        self.assertEqual(config["validation"]["batch_size"], 16)
        self.assertEqual(stage1_config["batch_size"], 32)
        self.assertEqual(stage1_config["validation"]["batch_size"], 32)
        self.assertEqual(config["sampling_seed"], 1337)
        self.assertEqual(config["generator"]["cycle_steps_schedule"], [4, 8, 16, 32])
        self.assertEqual(config["stages"]["stage2"]["lambda_initial"], 0.01)
        self.assertEqual(config["stages"]["stage2"]["lambda_max"], 0.01)
        self.assertEqual(config["stages"]["stage2"]["lambda_growth"], 0)
        self.assertEqual(config["stages"]["stage2"]["gradient_conflict"], {"enabled": True, "interval": 50})
        self.assertIn("resume_from", config)
        self.assertNotEqual(config["resume_from"], "artifacts/checkpoints/g_v2/best.pt")
        self.assertNotIn("g_v2/best.pt", str(config["resume_from"]))

    def test_balanced_debug_fixed16_ema_monitor_config_validates(self) -> None:
        from safa.training import g_loop

        path = Path("configs/stability/train_g_balanced_debug_ema_monitor_fixed16.yaml")
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(text)

        self.assertEqual(config["train_index"], "data/index/train_balanced_debug.jsonl")
        self.assertEqual(config["train_features"], "artifacts/e0_features/train_balanced_debug")
        self.assertEqual(config["resume_from"], "artifacts/checkpoints/g_v2_best_stage1/best.pt")
        self.assertIs(config["resume_from_legacy_stage1_metrics"], True)
        self.assertEqual(config["stages"]["stage1"]["epochs"], 0)
        self.assertEqual(config["stages"]["stage2"]["epochs"], 5)
        self.assertEqual(config["stages"]["stage2"]["lambda_initial"], 0.01)
        self.assertEqual(config["stages"]["stage2"]["lambda_max"], 0.01)
        self.assertEqual(config["stages"]["stage2"]["lambda_growth"], 0)
        self.assertEqual(config["stages"]["stage2"]["gradient_conflict"], {"enabled": True, "interval": 50})
        self.assertEqual(config["generator"]["cycle_steps_schedule"], [])
        self.assertEqual(config["generator"]["train_cycle_steps"], 16)
        self.assertEqual(config["ema"]["decay"], 0.999)
        self.assertEqual(config["ema"]["enabled"], True)
        self.assertEqual(config["ema"]["evaluate_raw"], True)
        self.assertEqual(config["ema"]["evaluate_ema"], True)
        self.assertEqual(config["best_model"], "ema")
        self.assertEqual(config["out_dir"], "artifacts/checkpoints/stability_balanced_debug_fixed16")
        self.assertIn("artifacts/eval/stability_balanced_debug_fixed16", text)
        g_loop._validate_train_g_config(config)

    def test_phase_c_balanced_debug_monitor10_rawbest_configs_validate(self) -> None:
        from safa.training import g_loop

        cases = {
            "fixed8": {
                "path": Path("configs/stability/train_g_balanced_debug_monitor10_rawbest_fixed8.yaml"),
                "base": Path("configs/stability/train_g_balanced_debug_ema_monitor_fixed8.yaml"),
                "out_dir": "artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed8",
                "eval_dir": "artifacts/eval/stability_balanced_debug_monitor10_rawbest_fixed8",
                "train_cycle_steps": 8,
                "cycle_steps_schedule": [],
            },
            "fixed16": {
                "path": Path("configs/stability/train_g_balanced_debug_monitor10_rawbest_fixed16.yaml"),
                "base": Path("configs/stability/train_g_balanced_debug_ema_monitor_fixed16.yaml"),
                "out_dir": "artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed16",
                "eval_dir": "artifacts/eval/stability_balanced_debug_monitor10_rawbest_fixed16",
                "train_cycle_steps": 16,
                "cycle_steps_schedule": [],
            },
            "schedule_4_8_16": {
                "path": Path("configs/stability/train_g_balanced_debug_monitor10_rawbest_schedule_4_8_16.yaml"),
                "base": Path("configs/stability/train_g_balanced_debug_ema_monitor_schedule_4_8_16.yaml"),
                "out_dir": "artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16",
                "eval_dir": "artifacts/eval/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16",
                "train_cycle_steps": 16,
                "cycle_steps_schedule": [4, 8, 16],
            },
            "schedule_4_8_16_32": {
                "path": Path("configs/stability/train_g_balanced_debug_monitor10_rawbest_schedule_4_8_16_32.yaml"),
                "base": Path("configs/stability/train_g_balanced_debug_ema_monitor_schedule_4_8_16_32.yaml"),
                "out_dir": "artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16_32",
                "eval_dir": "artifacts/eval/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16_32",
                "train_cycle_steps": 16,
                "cycle_steps_schedule": [4, 8, 16, 32],
            },
        }

        unchanged_fields = (
            "seed",
            "sampling_seed",
            "batch_size",
            "train_index",
            "train_features",
            "e0_checkpoint",
            "resume_from",
            "resume_from_legacy_stage1_metrics",
            "allow_stage2_without_stage1_gate",
            "ema",
            "validation",
        )
        unchanged_generator_fields = (
            "model_type",
            "base_channels",
            "channel_multipliers",
            "time_embedding_dim",
            "condition_dim",
            "sample_steps",
            "sampler",
        )

        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertTrue(expected["path"].is_file())
                text = expected["path"].read_text(encoding="utf-8")
                config = yaml.safe_load(text)
                base_config = yaml.safe_load(expected["base"].read_text(encoding="utf-8"))

                self.assertEqual(config["out_dir"], expected["out_dir"])
                self.assertIn(expected["eval_dir"], text)
                self.assertEqual(config["best_model"], "raw")
                self.assertEqual(config["stages"]["stage2"]["gradient_conflict"], {"enabled": True, "interval": 10})
                self.assertEqual(config["generator"]["train_cycle_steps"], expected["train_cycle_steps"])
                self.assertEqual(config["generator"]["cycle_steps_schedule"], expected["cycle_steps_schedule"])
                self.assertEqual(config["stages"]["stage2"]["lambda_initial"], 0.01)
                self.assertEqual(config["stages"]["stage2"]["lambda_max"], 0.01)
                self.assertEqual(config["stages"]["stage2"]["lambda_growth"], 0)
                self.assertIs(config["ema"]["enabled"], True)
                self.assertIs(config["ema"]["evaluate_raw"], True)
                self.assertIs(config["ema"]["evaluate_ema"], True)
                self.assertIs(config["ema"]["save_ema_checkpoint"], True)
                for field in unchanged_fields:
                    self.assertEqual(config[field], base_config[field], field)
                for field in unchanged_generator_fields:
                    self.assertEqual(config["generator"][field], base_config["generator"][field], field)
                self.assertNotEqual(config["out_dir"], base_config["out_dir"])
                g_loop._validate_train_g_config(config)

    def test_train_g_tmux_defaults_to_v2_best_on_gpu_4_to_7_with_ram_guard(self) -> None:
        script = Path("scripts/run_train_g_tmux.sh").read_text(encoding="utf-8")

        self.assertIn('CONFIG="${CONFIG:-configs/train_g_v2_best.yaml}"', script)
        self.assertIn('SESSION="${SESSION:-train_g_v2_best}"', script)
        self.assertIn('LOG="${LOG:-artifacts/logs/train_g_v2_best.log}"', script)
        self.assertIn('SAFA_CUDA_VISIBLE_DEVICES:-4,5,6,7', script)
        self.assertIn("tmux new-session", script)
        self.assertIn("scripts/guarded_run.py --max-ram-fraction 0.90", script)
        self.assertIn("--nproc_per_node=4", script)

    def test_train_g_tmux_requires_values_for_options(self) -> None:
        for option in ("--config", "--log", "--session"):
            with self.subTest(option=option):
                result = subprocess.run(
                    ["bash", "scripts/run_train_g_tmux.sh", option],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{option} requires a value", result.stderr)
                self.assertIn("Usage:", result.stderr)
                self.assertNotIn("unbound variable", result.stderr)


if __name__ == "__main__":
    unittest.main()
