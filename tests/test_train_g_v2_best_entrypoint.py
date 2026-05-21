from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

import yaml


class TrainGV2BestEntrypointTests(unittest.TestCase):
    def test_v2_best_config_documents_fixed_low_lambda_recipe(self) -> None:
        path = Path("configs/train_g_v2_best.yaml")
        self.assertTrue(path.is_file())
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(config["sampling_seed"], 1337)
        self.assertEqual(config["generator"]["cycle_steps_schedule"], [4, 8, 16, 32])
        self.assertEqual(config["stages"]["stage2"]["lambda_initial"], 0.01)
        self.assertEqual(config["stages"]["stage2"]["lambda_max"], 0.01)
        self.assertEqual(config["stages"]["stage2"]["lambda_growth"], 0)
        self.assertIn("resume_from", config)
        self.assertNotEqual(config["resume_from"], "artifacts/checkpoints/g_v2/best.pt")
        self.assertNotIn("g_v2/best.pt", str(config["resume_from"]))

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
