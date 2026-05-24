from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safa.training.g_loop import DistributedContext, _distributed_manifest, _sync_epoch_control


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class DDPContractTests(unittest.TestCase):
    def test_train_g_tmux_uses_four_gpu_torchrun(self) -> None:
        script = Path("scripts/run_train_g_tmux.sh").read_text(encoding="utf-8")
        self.assertIn('SAFA_CUDA_VISIBLE_DEVICES:-4,5,6,7', script)
        self.assertIn("scripts/guarded_run.py", script)
        self.assertIn("torch.distributed.run", script)
        self.assertIn("--nproc_per_node=4", script)
        self.assertIn("tmux new-session", script)

    def test_sync_epoch_control_single_process_is_noop(self) -> None:
        context = DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, is_main=True, device="cuda:0", backend="single")
        result = _sync_epoch_control(0.02, 0.8, 3, True, "cuda:0", context)
        self.assertEqual(result, (0.02, 0.8, 3, True))

    def test_distributed_manifest_does_not_expose_device_mapping(self) -> None:
        context = DistributedContext(enabled=True, rank=0, local_rank=0, world_size=4, is_main=True, device="cuda:0", backend="gloo")
        self.assertEqual(_distributed_manifest(context), {"enabled": True, "world_size": 4, "backend": "gloo"})

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for DDP init contract")
    def test_ddp_env_requires_local_rank(self) -> None:
        from safa.training.g_loop import _init_distributed

        with patch.dict(os.environ, {"WORLD_SIZE": "4", "RANK": "0"}, clear=False):
            os.environ.pop("LOCAL_RANK", None)
            with self.assertRaises(RuntimeError):
                _init_distributed({"device": "cuda:0"})

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for checkpoint save contract")
    def test_save_generator_unwraps_module_prefix(self) -> None:
        import torch

        from safa.models.generator import ConditionalFlowGenerator
        from safa.training.g_loop import _save_generator

        config = {
            "embedding_dim": 512,
            "image_size": 224,
            "base_channels": 4,
            "channel_multipliers": [1],
            "time_embedding_dim": 16,
            "condition_dim": 32,
            "sample_steps": 1,
            "train_cycle_steps": 1,
            "sampler": "euler",
        }

        class Wrapper:
            def __init__(self, module):
                self.module = module

        generator = ConditionalFlowGenerator(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.pt"
            metrics = {
                "stage": "stage1",
                "loss": 1.0,
                "validation_latent_cosine_mean": 0.9,
                "validation_single_face_eq1_rate": 0.8,
            }
            _save_generator(
                path,
                Wrapper(generator),
                generator.config,
                {
                    "ema": {"enabled": False, "decay": 0.999, "evaluate_raw": True, "evaluate_ema": False, "save_ema_checkpoint": False},
                    "best_model": "raw",
                },
                metrics,
                [],
            )
            payload = torch.load(path, map_location="cpu")
        self.assertTrue(payload["model_state_dict"])
        self.assertFalse(any(key.startswith("module.") for key in payload["model_state_dict"]))
        self.assertNotIn("sampling_seed", payload["training_config"])


if __name__ == "__main__":
    unittest.main()
