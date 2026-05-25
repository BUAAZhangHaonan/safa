from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class MediumV1GSupportTests(unittest.TestCase):
    def _base_stage2_metrics(self) -> dict:
        return {
            "stage": "stage2",
            "loss": 1.0,
            "validation_raw_latent_cosine_mean": 0.90,
            "validation_raw_single_face_eq1_rate": 0.80,
            "validation_ema_latent_cosine_mean": 0.90,
            "validation_ema_single_face_eq1_rate": 0.90,
        }

    def test_stage2_checkpoint_filenames_cover_utility_and_real_quality_metrics(self) -> None:
        from safa.training.g_loop import _stage2_checkpoint_filenames_to_save

        metrics = self._base_stage2_metrics()
        metrics.update(
            {
                "quality_raw_fid": 12.0,
                "quality_raw_kid_mean": 0.02,
                "quality_raw_niqe": 4.5,
            }
        )
        previous = [
            {
                "stage": "stage2",
                "loss": 1.2,
                "validation_raw_latent_cosine_mean": 0.80,
                "validation_raw_single_face_eq1_rate": 0.70,
                "validation_ema_latent_cosine_mean": 0.80,
                "validation_ema_single_face_eq1_rate": 0.80,
                "quality_raw_fid": 20.0,
                "quality_raw_kid_mean": 0.03,
                "quality_raw_niqe": 5.0,
            }
        ]

        names = _stage2_checkpoint_filenames_to_save(metrics, previous)

        self.assertIn("best_raw_utility.pt", names)
        self.assertIn("best_ema_utility.pt", names)
        self.assertIn("best_raw_quality.pt", names)
        self.assertNotIn("best_ema_quality.pt", names)

    def test_quality_checkpoint_is_not_selected_without_complete_quality_metrics(self) -> None:
        from safa.training.g_loop import _stage2_checkpoint_filenames_to_save

        metrics = self._base_stage2_metrics()
        metrics["quality_raw_fid"] = 12.0

        names = _stage2_checkpoint_filenames_to_save(metrics, [])

        self.assertIn("best_raw_utility.pt", names)
        self.assertNotIn("best_raw_quality.pt", names)

    def test_uncertainty_calibration_rejects_non_positive_or_non_finite_scales(self) -> None:
        from safa.training.g_loop import _finalize_uncertainty_calibration

        with self.assertRaisesRegex(RuntimeError, "flow_loss_initial"):
            _finalize_uncertainty_calibration(flow_sum=0.0, cycle_sum=1.0, batches=1)
        with self.assertRaisesRegex(RuntimeError, "cycle_loss_initial"):
            _finalize_uncertainty_calibration(flow_sum=1.0, cycle_sum=float("inf"), batches=1)
        with self.assertRaisesRegex(RuntimeError, "calibration_batches"):
            _finalize_uncertainty_calibration(flow_sum=1.0, cycle_sum=1.0, batches=0)

    def test_quality_eval_hook_generates_epoch_images_instead_of_reusing_generated_dir(self) -> None:
        from safa.training import g_loop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale_generated = root / "old_eval" / "generated_images"
            quality_dir = root / "quality"
            config = {
                "stages": {
                    "stage1": {
                        "quality_eval": {
                            "enabled": True,
                            "niqe_interval_epochs": 1,
                            "niqe_max_samples": 2,
                            "metrics": ["niqe"],
                            "generated_dir": str(stale_generated),
                            "output_dir": str(quality_dir),
                            "model": "raw",
                        }
                    }
                }
            }

            load_calls: list[int] = []
            generation_calls: list[dict] = []
            eval_calls: list[dict] = []

            def fake_loader(runner_config: dict, max_samples: int, **kwargs):
                load_calls.append(max_samples)
                return object()

            def fake_generate(**kwargs):
                generation_calls.append(kwargs)
                Path(kwargs["generated_dir"]).mkdir(parents=True)
                return int(kwargs["max_samples"])

            def fake_quality_eval(**kwargs):
                eval_calls.append(kwargs)
                return {"iqa": {"method": "niqe", "mean": 4.25, "std": 0.5}, "num_generated": 2}

            generator = object()

            with patch.object(g_loop, "_build_quality_eval_loader", side_effect=fake_loader), patch.object(
                g_loop, "_generate_quality_eval_images", side_effect=fake_generate
            ), patch.object(g_loop, "_evaluate_generation_quality", side_effect=fake_quality_eval):
                metrics = g_loop._run_quality_eval_hook(
                    config,
                    "stage1",
                    0,
                    generator=generator,
                    ema=None,
                    device="cpu",
                    generator_config=SimpleNamespace(image_size=4, sample_steps=1),
                    sampling_seed=1337,
                    use_amp=False,
                    ema_config={"enabled": False},
                )

            expected_generated = quality_dir / "epoch_0001" / "generated_images"
            self.assertEqual(load_calls, [2])
            self.assertEqual(len(generation_calls), 1)
            self.assertIs(generation_calls[0]["generator"], generator)
            self.assertEqual(generation_calls[0]["generated_dir"], expected_generated)
            self.assertNotEqual(generation_calls[0]["generated_dir"], stale_generated)
            self.assertEqual(len(eval_calls), 1)
            self.assertEqual(eval_calls[0]["generated_dir"], expected_generated)
            self.assertEqual(eval_calls[0]["metrics"], ("niqe",))
            self.assertEqual(eval_calls[0]["max_generated"], 2)
            self.assertTrue(str(eval_calls[0]["output"]).endswith("stage1_epoch_0001_raw_niqe.json"))
            self.assertEqual(metrics["quality_raw_niqe"], 4.25)
            self.assertNotIn("quality_raw_fid", metrics)

    def test_quality_eval_hook_runs_distribution_metrics_outside_training_process(self) -> None:
        from safa.training import g_loop

        with tempfile.TemporaryDirectory() as tmp:
            quality_dir = Path(tmp) / "quality"
            config = {
                "stages": {
                    "stage1": {
                        "quality_eval": {
                            "enabled": True,
                            "niqe_interval_epochs": 1,
                            "distribution_interval_epochs": 20,
                            "niqe_max_samples": 2,
                            "distribution_max_samples": 5,
                            "distribution_timeout_seconds": 600,
                            "metrics": ["niqe", "fid", "kid"],
                            "real_index": "data/index/val_single_face.jsonl",
                            "output_dir": str(quality_dir),
                            "distribution_cuda_visible_devices": "0",
                            "distribution_device": "cuda:0",
                            "model": "raw",
                        }
                    }
                }
            }

            def run_epoch(stage_epoch: int):
                load_calls: list[int] = []
                generation_calls: list[dict] = []
                eval_calls: list[dict] = []
                external_eval_calls: list[dict] = []

                def fake_loader(runner_config: dict, max_samples: int, **kwargs):
                    load_calls.append(max_samples)
                    return object()

                def fake_generate(**kwargs):
                    generation_calls.append(kwargs)
                    Path(kwargs["generated_dir"]).mkdir(parents=True)
                    return int(kwargs["max_samples"])

                def fake_quality_eval(**kwargs):
                    eval_calls.append(kwargs)
                    names = tuple(kwargs["metrics"])
                    if names == ("niqe",):
                        return {"iqa": {"method": "niqe", "mean": 3.0, "std": 0.0}}
                    raise AssertionError(f"in-process distribution quality eval is forbidden: {names}")

                def fake_external_quality_eval(**kwargs):
                    external_eval_calls.append(kwargs)
                    return {"fid": 12.0, "kid_mean": 0.2, "kid_std": 0.01, "num_real": 5, "num_generated": 5}

                with patch.object(g_loop, "_build_quality_eval_loader", side_effect=fake_loader), patch.object(
                    g_loop, "_generate_quality_eval_images", side_effect=fake_generate
                ), patch.object(g_loop, "_evaluate_generation_quality", side_effect=fake_quality_eval), patch.object(
                    g_loop, "_evaluate_generation_quality_subprocess", side_effect=fake_external_quality_eval
                ):
                    metrics = g_loop._run_quality_eval_hook(
                        config,
                        "stage1",
                        stage_epoch,
                        generator=object(),
                        ema=None,
                        device="cpu",
                        generator_config=SimpleNamespace(image_size=4, sample_steps=1),
                        sampling_seed=1337,
                        use_amp=False,
                        ema_config={"enabled": False},
                    )
                return metrics, load_calls, generation_calls, eval_calls, external_eval_calls

            metrics, load_calls, generation_calls, eval_calls, external_eval_calls = run_epoch(0)
            self.assertEqual(load_calls, [2])
            self.assertEqual(generation_calls[0]["max_samples"], 2)
            self.assertEqual([call["metrics"] for call in eval_calls], [("niqe",)])
            self.assertEqual(external_eval_calls, [])
            self.assertEqual(eval_calls[0]["max_generated"], 2)
            self.assertIsNone(eval_calls[0]["real_index"])
            self.assertEqual(metrics["quality_raw_niqe"], 3.0)
            self.assertEqual(metrics["quality_raw_niqe_mean"], 3.0)
            self.assertEqual(metrics["quality_raw_niqe_std"], 0.0)

            metrics, load_calls, generation_calls, eval_calls, external_eval_calls = run_epoch(19)
            self.assertEqual(load_calls, [5])
            self.assertEqual(len(generation_calls), 1)
            self.assertEqual(generation_calls[0]["max_samples"], 5)
            self.assertEqual([call["metrics"] for call in eval_calls], [("niqe",)])
            self.assertEqual([call["metrics"] for call in external_eval_calls], [("fid", "kid")])
            self.assertEqual(eval_calls[0]["max_generated"], 2)
            self.assertIsNone(eval_calls[0]["real_index"])
            self.assertEqual(external_eval_calls[0]["max_generated"], 5)
            self.assertEqual(external_eval_calls[0]["max_real"], 5)
            self.assertEqual(str(external_eval_calls[0]["real_index"]), "data/index/val_single_face.jsonl")
            self.assertEqual(external_eval_calls[0]["cuda_visible_devices"], "0")
            self.assertEqual(external_eval_calls[0]["device"], "cuda:0")
            self.assertEqual(external_eval_calls[0]["timeout_seconds"], 600)
            self.assertTrue(str(external_eval_calls[0]["generated_dir"]).endswith("quality/epoch_0020/generated_images"))
            self.assertEqual(metrics["quality_raw_niqe"], 3.0)
            self.assertEqual(metrics["quality_raw_fid"], 12.0)
            self.assertEqual(metrics["quality_raw_kid_mean"], 0.2)

    def test_quality_eval_distribution_subprocess_builds_gpu_command_and_reads_json(self) -> None:
        from safa.training import g_loop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "epoch_0020" / "stage1_epoch_0020_raw_distribution.json"
            generated_dir = root / "epoch_0020" / "generated_images"
            real_index = root / "real.jsonl"
            generated_dir.mkdir(parents=True)
            real_index.write_text("{}", encoding="utf-8")
            calls: list[dict] = []

            def fake_run(command, *, cwd, env, text, capture_output, timeout):
                calls.append({"command": command, "cwd": cwd, "env": env, "text": text, "capture_output": capture_output, "timeout": timeout})
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps({"fid": 10.0, "kid_mean": 0.1, "kid_std": 0.01, "num_real": 5, "num_generated": 5}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(g_loop.subprocess, "run", side_effect=fake_run):
                payload = g_loop._evaluate_generation_quality_subprocess(
                    real_index=real_index,
                    generated_dir=generated_dir,
                    output=output,
                    iqa_method="niqe",
                    metrics=("fid", "kid"),
                    max_generated=5,
                    max_real=5,
                    subset_seed=123,
                    device="cuda:0",
                    cuda_visible_devices="0",
                    timeout_seconds=600,
                )

            self.assertEqual(payload["fid"], 10.0)
            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertTrue(call["text"])
            self.assertTrue(call["capture_output"])
            self.assertEqual(call["timeout"], 600)
            command = call["command"]
            self.assertTrue(str(command[1]).endswith("scripts/eval_generation_quality.py"))
            self.assertIn("--real-index", command)
            self.assertIn(str(real_index), command)
            self.assertIn("--generated-dir", command)
            self.assertIn(str(generated_dir), command)
            self.assertIn("--output", command)
            self.assertIn(str(output), command)
            self.assertIn("--metrics", command)
            self.assertIn("fid", command)
            self.assertIn("kid", command)
            self.assertIn("--max-generated", command)
            self.assertIn("5", command)
            self.assertIn("--max-real", command)
            self.assertIn("--seed", command)
            self.assertIn("123", command)
            self.assertIn("--device", command)
            self.assertIn("cuda:0", command)

    def test_quality_eval_distribution_subprocess_fails_without_json(self) -> None:
        from safa.training import g_loop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "missing.json"
            generated_dir = root / "generated_images"
            real_index = root / "real.jsonl"
            generated_dir.mkdir()
            real_index.write_text("{}", encoding="utf-8")
            output.write_text('{"fid": 999.0}', encoding="utf-8")

            def fake_run(command, *, cwd, env, text, capture_output, timeout):
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(g_loop.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(FileNotFoundError, "did not write JSON"):
                    g_loop._evaluate_generation_quality_subprocess(
                        real_index=real_index,
                        generated_dir=generated_dir,
                        output=output,
                        iqa_method="niqe",
                        metrics=("fid", "kid"),
                        max_generated=5,
                        max_real=5,
                        subset_seed=123,
                        device="cuda:0",
                        cuda_visible_devices="0",
                        timeout_seconds=600,
                    )

    def test_quality_eval_distribution_subprocess_fails_on_nonzero_exit(self) -> None:
        from safa.training import g_loop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "quality.json"
            generated_dir = root / "generated_images"
            real_index = root / "real.jsonl"
            generated_dir.mkdir()
            real_index.write_text("{}", encoding="utf-8")

            def fake_run(command, *, cwd, env, text, capture_output, timeout):
                return SimpleNamespace(returncode=7, stdout="stdout text", stderr="stderr text")

            with patch.object(g_loop.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "failed with exit code 7"):
                    g_loop._evaluate_generation_quality_subprocess(
                        real_index=real_index,
                        generated_dir=generated_dir,
                        output=output,
                        iqa_method="niqe",
                        metrics=("fid", "kid"),
                        max_generated=5,
                        max_real=5,
                        subset_seed=123,
                        device="cuda:0",
                        cuda_visible_devices="0",
                        timeout_seconds=600,
                    )

    def test_quality_eval_distribution_subprocess_fails_on_timeout_with_output(self) -> None:
        from safa.training import g_loop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "quality.json"
            generated_dir = root / "generated_images"
            real_index = root / "real.jsonl"
            generated_dir.mkdir()
            real_index.write_text("{}", encoding="utf-8")

            def fake_run(command, *, cwd, env, text, capture_output, timeout):
                raise g_loop.subprocess.TimeoutExpired(command, timeout, output="stdout text", stderr="stderr text")

            with patch.object(g_loop.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "(?s)timed out after 3 seconds.*stderr text.*stdout text"):
                    g_loop._evaluate_generation_quality_subprocess(
                        real_index=real_index,
                        generated_dir=generated_dir,
                        output=output,
                        iqa_method="niqe",
                        metrics=("fid", "kid"),
                        max_generated=5,
                        max_real=5,
                        subset_seed=123,
                        device="cuda:0",
                        cuda_visible_devices="0",
                        timeout_seconds=3,
                    )

    def test_quality_eval_enabled_requires_new_schedule_fields(self) -> None:
        from safa.training import g_loop

        payload = {
            "enabled": True,
            "metrics": ["niqe"],
            "niqe_max_samples": 2,
            "output_dir": "artifacts/eval/example/quality",
            "model": "raw",
        }

        with self.assertRaisesRegex(ValueError, "niqe_interval_epochs"):
            g_loop._quality_eval_due_groups(payload, "stage1", 1)

    def test_quality_eval_block_requires_explicit_enabled(self) -> None:
        from safa.training import g_loop

        config = {
            "stages": {"stage1": {"quality_eval": {"metrics": ["niqe"]}}},
            "validation": {"enabled": True, "index": "x", "features": "y", "batch_size": 1},
        }

        with self.assertRaisesRegex(ValueError, "quality_eval.enabled"):
            g_loop._validate_quality_eval_configs(config, config["stages"])

    def test_quality_eval_distribution_requires_explicit_timeout_and_max_samples(self) -> None:
        from safa.training import g_loop

        payload = {
            "enabled": True,
            "metrics": ["fid", "kid"],
            "distribution_interval_epochs": 20,
            "real_index": "data/index/val_single_face.jsonl",
        }

        with self.assertRaisesRegex(ValueError, "distribution_max_samples"):
            g_loop._quality_eval_due_groups(payload, "stage1", 20)

        payload["distribution_max_samples"] = 3969
        with self.assertRaisesRegex(ValueError, "distribution_timeout_seconds"):
            g_loop._quality_eval_due_groups(payload, "stage1", 20)

    def test_quality_payload_to_metrics_writes_clear_raw_aliases(self) -> None:
        from safa.training import g_loop

        metrics = g_loop._quality_payload_to_metrics(
            {
                "fid": 10.0,
                "kid_mean": 0.1,
                "kid_std": 0.01,
                "iqa": {"method": "niqe", "mean": 4.5, "std": 0.25},
            },
            "raw",
            ("fid", "kid", "niqe"),
        )

        self.assertEqual(metrics["quality_raw_niqe_mean"], 4.5)
        self.assertEqual(metrics["quality_raw_niqe_std"], 0.25)
        self.assertEqual(metrics["quality_raw_niqe"], 4.5)
        self.assertEqual(metrics["quality_raw_fid"], 10.0)
        self.assertEqual(metrics["quality_raw_kid_mean"], 0.1)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for loss weighting integration tests")
    def test_generator_training_step_fixed_loss_weighting_uses_config_weights(self) -> None:
        import torch
        from torch import nn

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _GeneratorTrainingStep, _loss_weighting_runtime_from_config

        class DummyGenerator(nn.Module):
            def flow_matching_loss(self, images, z):
                loss = images.sum() * 0.0 + z.sum() * 0.0 + 2.0
                return loss, {"flow_matching_mse": loss.detach()}

            def sample(self, z, **kwargs):
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        class DummyE0(nn.Module):
            def forward(self, images):
                return {
                    "embedding": torch.tensor([[0.0, 1.0]], device=images.device).repeat(images.shape[0], 1),
                    "logits": torch.zeros(images.shape[0], 2, device=images.device),
                }

        runtime = _loss_weighting_runtime_from_config({"loss_weighting": {"type": "fixed", "flow_weight": 2.0, "cycle_weight": 0.01}})
        module = _GeneratorTrainingStep(DummyGenerator(), DummyE0(), FlowGeneratorConfig(embedding_dim=2, image_size=4), 1337, runtime)

        loss, _, cycle, _, _ = module(torch.zeros(1, 3, 4, 4), torch.tensor([[1.0, 0.0]]), ["sample"], True, 99.0)

        self.assertAlmostEqual(float(cycle), 1.0, places=6)
        self.assertAlmostEqual(float(loss.detach()), 4.01, places=6)
        self.assertEqual(module.last_loss_metrics["loss_weighting_type"], "fixed")
        self.assertEqual(module.last_loss_metrics["effective_cycle_loss_weight"], 0.01)

        module(torch.zeros(1, 3, 4, 4), torch.tensor([[1.0, 0.0]]), ["sample"], False, 99.0)
        self.assertEqual(module.last_loss_metrics["effective_cycle_loss_weight"], 0.0)

    def test_plot_script_missing_required_input_fails_fast(self) -> None:
        from scripts.plot_medium_v1_curves import load_medium_v1_history

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_medium_v1_history(Path(tmp) / "missing.json")

    def test_stage1_long200_plot_timeseries_reads_latest_run_and_quality_jsons(self) -> None:
        from scripts.plot_medium_v1_curves import build_stage1_long200_timeseries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "history.json"
            last_metrics_path = root / "last_metrics.json"
            quality_dir = root / "quality"
            history_path.write_text(
                json.dumps(
                    {
                        "history": [
                            {"stage": "stage1", "stage_epoch": 0, "loss": 9.0, "flow_matching_mse": 9.0},
                            {"stage": "stage1", "stage_epoch": 1, "loss": 8.0, "flow_matching_mse": 8.0},
                            {
                                "stage": "stage1",
                                "stage_epoch": 0,
                                "loss": 0.3,
                                "flow_matching_mse": 0.3,
                                "grad_norm": 0.7,
                                "validation_raw_latent_cosine_mean": 0.1,
                                "validation_raw_source_prediction_preserved": 0.2,
                                "validation_raw_face_detect_ge1_rate": 0.9,
                                "validation_raw_single_face_eq1_rate": 0.8,
                                "validation_raw_zero_face_rate": 0.1,
                                "validation_raw_multi_face_rate": 0.0,
                            },
                            {
                                "stage": "stage1",
                                "stage_epoch": 1,
                                "loss": 0.2,
                                "flow_matching_mse": 0.2,
                                "grad_norm": 0.6,
                                "validation_raw_latent_cosine_mean": 0.3,
                                "validation_raw_source_prediction_preserved": 0.4,
                                "validation_raw_face_detect_ge1_rate": 1.0,
                                "validation_raw_single_face_eq1_rate": 0.9,
                                "validation_raw_zero_face_rate": 0.0,
                                "validation_raw_multi_face_rate": 0.1,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            last_metrics_path.write_text(
                json.dumps(
                    {
                        "stage": "stage1",
                        "stage_epoch": 2,
                        "loss": 0.1,
                        "flow_matching_mse": 0.1,
                        "grad_norm": 0.5,
                        "validation_raw_latent_cosine_mean": 0.5,
                        "validation_raw_source_prediction_preserved": 0.6,
                        "validation_raw_face_detect_ge1_rate": 1.0,
                        "validation_raw_single_face_eq1_rate": 1.0,
                        "validation_raw_zero_face_rate": 0.0,
                        "validation_raw_multi_face_rate": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            (quality_dir / "epoch_0001").mkdir(parents=True)
            (quality_dir / "epoch_0001" / "stage1_epoch_0001_raw_niqe.json").write_text(
                json.dumps({"iqa": {"method": "niqe", "mean": 4.0, "std": 0.1}, "metrics": ["niqe"]}),
                encoding="utf-8",
            )
            (quality_dir / "epoch_0002").mkdir()
            (quality_dir / "epoch_0002" / "stage1_epoch_0002_raw_niqe.json").write_text(
                json.dumps({"iqa": {"method": "niqe", "mean": 3.5, "std": 0.2}, "metrics": ["niqe"]}),
                encoding="utf-8",
            )
            (quality_dir / "epoch_0002" / "stage1_epoch_0002_raw_distribution.json").write_text(
                json.dumps({"fid": 12.0, "kid_mean": 0.02, "kid_std": 0.003, "metrics": ["fid", "kid"]}),
                encoding="utf-8",
            )
            (quality_dir / "epoch_0003").mkdir()
            (quality_dir / "epoch_0003" / "stage1_epoch_0003_raw_niqe.json").write_text(
                json.dumps({"iqa": {"method": "niqe", "mean": 3.0, "std": 0.3}, "metrics": ["niqe"]}),
                encoding="utf-8",
            )

            payload = build_stage1_long200_timeseries(history_path, last_metrics_path, quality_dir, run_name="unit_run")

        rows = payload["epochs"]
        self.assertEqual(payload["run"], "unit_run")
        self.assertEqual([row["epoch"] for row in rows], [1, 2, 3])
        self.assertEqual([row["loss"] for row in rows], [0.3, 0.2, 0.1])
        self.assertEqual(rows[0]["niqe"], 4.0)
        self.assertIsNone(rows[0]["fid"])
        self.assertEqual(rows[1]["fid"], 12.0)
        self.assertEqual(rows[1]["kid_mean"], 0.02)
        self.assertEqual(rows[2]["niqe"], 3.0)

    def test_stage1_long200_plot_strict_mode_rejects_missing_quality_jsons(self) -> None:
        from scripts.plot_medium_v1_curves import build_stage1_long200_timeseries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "history.json"
            quality_dir = root / "quality"
            history_path.write_text(json.dumps({"history": [{"stage": "stage1", "stage_epoch": 0, "loss": 1.0}]}), encoding="utf-8")
            (quality_dir / "epoch_0001").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "missing NIQE quality JSON"):
                build_stage1_long200_timeseries(history_path, None, quality_dir)

            payload = build_stage1_long200_timeseries(history_path, None, quality_dir, allow_missing_quality=True)
            self.assertIsNone(payload["epochs"][0]["niqe"])

    def test_stage1_long200_plot_strict_mode_rejects_missing_quality_dir(self) -> None:
        from scripts.plot_medium_v1_curves import build_stage1_long200_timeseries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "history.json"
            quality_dir = root / "missing_quality"
            history_path.write_text(json.dumps({"history": [{"stage": "stage1", "stage_epoch": 0, "loss": 1.0}]}), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "required Stage1 quality directory is missing"):
                build_stage1_long200_timeseries(history_path, None, quality_dir)

            payload = build_stage1_long200_timeseries(history_path, None, quality_dir, allow_missing_quality=True)
            self.assertIsNone(payload["epochs"][0]["niqe"])

    def test_stage1_long200_plot_strict_mode_rejects_missing_distribution_json_on_interval(self) -> None:
        from scripts.plot_medium_v1_curves import build_stage1_long200_timeseries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "history.json"
            quality_dir = root / "quality"
            history_path.write_text(json.dumps({"history": [{"stage": "stage1", "stage_epoch": 19, "loss": 1.0}]}), encoding="utf-8")
            (quality_dir / "epoch_0020").mkdir(parents=True)
            (quality_dir / "epoch_0020" / "stage1_epoch_0020_raw_niqe.json").write_text(
                json.dumps({"iqa": {"method": "niqe", "mean": 4.0}, "metrics": ["niqe"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing FID/KID distribution quality JSON"):
                build_stage1_long200_timeseries(history_path, None, quality_dir)

    def test_stage1_long200_plot_prefers_clear_quality_history_fields(self) -> None:
        from scripts.plot_medium_v1_curves import build_stage1_long200_timeseries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "history.json"
            history_path.write_text(
                json.dumps(
                    {
                        "history": [
                            {
                                "stage": "stage1",
                                "stage_epoch": 0,
                                "loss": 1.0,
                                "quality_raw_niqe_mean": 3.0,
                                "quality_raw_niqe_std": 0.3,
                                "quality_raw_niqe": 9.0,
                                "quality_raw_fid": 12.0,
                                "quality_raw_kid_mean": 0.02,
                                "quality_raw_kid_std": 0.003,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = build_stage1_long200_timeseries(
                history_path, None, root / "missing_quality", allow_missing_quality=True
            )

        row = payload["epochs"][0]
        self.assertEqual(row["niqe"], 3.0)
        self.assertEqual(row["niqe_std"], 0.3)
        self.assertEqual(row["fid"], 12.0)

    def test_stage1_long200_plot_writes_required_outputs(self) -> None:
        from scripts.plot_medium_v1_curves import plot_stage1_long200_curves

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "history.json"
            quality_dir = root / "quality" / "epoch_0001"
            out_dir = root / "plots"
            history_path.write_text(
                json.dumps(
                    {
                        "history": [
                            {
                                "stage": "stage1",
                                "stage_epoch": 0,
                                "loss": 0.3,
                                "flow_matching_mse": 0.2,
                                "grad_norm": 0.7,
                                "validation_raw_latent_cosine_mean": 0.1,
                                "validation_raw_source_prediction_preserved": 0.2,
                                "validation_raw_face_detect_ge1_rate": 0.9,
                                "validation_raw_single_face_eq1_rate": 0.8,
                                "validation_raw_zero_face_rate": 0.1,
                                "validation_raw_multi_face_rate": 0.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            quality_dir.mkdir(parents=True)
            (quality_dir / "stage1_epoch_0001_raw_niqe.json").write_text(
                json.dumps({"iqa": {"method": "niqe", "mean": 4.0}, "metrics": ["niqe"]}),
                encoding="utf-8",
            )

            outputs = plot_stage1_long200_curves(
                history_path=history_path,
                last_metrics_path=None,
                quality_dir=root / "quality",
                out_dir=out_dir,
                output_prefix="stage1_long200_v4",
            )

            expected = {
                out_dir / "stage1_long200_v4_quality_curves.png",
                out_dir / "stage1_long200_v4_face_curves.png",
                out_dir / "stage1_long200_v4_training_curves.png",
                out_dir / "stage1_long200_v4_metrics_timeseries.json",
            }
            self.assertEqual(set(outputs), expected)
            for path in expected:
                self.assertTrue(path.is_file(), str(path))
            payload = json.loads((out_dir / "stage1_long200_v4_metrics_timeseries.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["epochs"][0]["niqe"], 4.0)

    def test_medium_v1_g_configs_use_e0_medium_cache_and_checkpoint(self) -> None:
        expected_train_features = "artifacts/e0_features/train_balanced_medium_e0_medium_v1"
        expected_validation_features = "artifacts/e0_features/val_single_face_e0_medium_v1"
        expected_e0_checkpoint = "artifacts/checkpoints/e0_medium_v1/best.pt"
        config_paths = (
            Path("configs/medium_v1/train_g_medium_v1_stage1.yaml"),
            Path("configs/medium_v1/train_g_medium_v1_stage2_m0.yaml"),
            Path("configs/medium_v1/train_g_medium_v1_stage2_m1_uw.yaml"),
            Path("configs/medium_v1/train_g_medium_v1_stage1_long200_v3.yaml"),
            Path("configs/medium_v1/train_g_medium_v1_stage1_long200_v4.yaml"),
        )

        for path in config_paths:
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file())
                config = yaml.safe_load(path.read_text(encoding="utf-8"))

                self.assertEqual(config["train_features"], expected_train_features)
                self.assertEqual(config["validation"]["features"], expected_validation_features)
                self.assertEqual(config["e0_checkpoint"], expected_e0_checkpoint)

    def test_medium_v1_stage1_configs_enable_niqe_only_epoch_quality(self) -> None:
        from safa.training import g_loop

        config_paths = (
            Path("configs/medium_v1/train_g_medium_v1_stage1.yaml"),
            Path("configs/medium_v1/train_g_medium_v1_stage1_continue_best_sf.yaml"),
        )

        for path in config_paths:
            with self.subTest(path=str(path)):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["stages"]["stage1"]["epochs"], 200)
                quality_eval = config["stages"]["stage1"]["quality_eval"]

                self.assertIs(quality_eval["enabled"], True)
                self.assertEqual(quality_eval["niqe_interval_epochs"], 1)
                self.assertEqual(quality_eval["niqe_max_samples"], 512)
                self.assertEqual(quality_eval["metrics"], ["niqe"])
                self.assertNotIn("fid", quality_eval["metrics"])
                self.assertNotIn("kid", quality_eval["metrics"])
                g_loop._validate_train_g_config(config)

    def test_medium_v1_stage1_long200_restart_config_uses_current_quality_schedule(self) -> None:
        from safa.training import g_loop

        path = Path("configs/medium_v1/train_g_medium_v1_stage1_long200.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        quality_eval = config["stages"]["stage1"]["quality_eval"]

        self.assertEqual(config["out_dir"], "artifacts/checkpoints/g_medium_v1_stage1_long200_v2")
        self.assertEqual(config["resume_from"], "artifacts/checkpoints/g_medium_v1_stage1_long200/last.pt")
        self.assertEqual(config["stages"]["stage1"]["epochs"], 200)
        self.assertEqual(config["stages"]["stage2"]["epochs"], 0)
        self.assertEqual(quality_eval["metrics"], ["niqe", "fid", "kid"])
        self.assertEqual(quality_eval["niqe_interval_epochs"], 1)
        self.assertEqual(quality_eval["distribution_interval_epochs"], 20)
        self.assertEqual(quality_eval["niqe_max_samples"], 512)
        self.assertEqual(quality_eval["distribution_max_samples"], 3969)
        self.assertEqual(quality_eval["distribution_timeout_seconds"], 3600)
        self.assertNotIn("generated_dir", quality_eval)
        g_loop._validate_train_g_config(config)

    def test_medium_v1_stage1_long200_v3_resumes_v2_with_niqe_only_training_quality(self) -> None:
        from safa.training import g_loop

        path = Path("configs/medium_v1/train_g_medium_v1_stage1_long200_v3.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        quality_eval = config["stages"]["stage1"]["quality_eval"]

        self.assertEqual(config["out_dir"], "artifacts/checkpoints/g_medium_v1_stage1_long200_v3")
        self.assertEqual(config["resume_from"], "artifacts/checkpoints/g_medium_v1_stage1_long200_v2/last.pt")
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["validation"]["batch_size"], 32)
        self.assertEqual(config["train_index"], "data/index/train_balanced_medium.jsonl")
        self.assertEqual(config["train_features"], "artifacts/e0_features/train_balanced_medium_e0_medium_v1")
        self.assertEqual(config["validation"]["features"], "artifacts/e0_features/val_single_face_e0_medium_v1")
        self.assertEqual(config["stages"]["stage1"]["epochs"], 200)
        self.assertEqual(config["stages"]["stage2"]["epochs"], 0)
        self.assertEqual(quality_eval["metrics"], ["niqe"])
        self.assertNotIn("fid", quality_eval["metrics"])
        self.assertNotIn("kid", quality_eval["metrics"])
        self.assertEqual(quality_eval["niqe_interval_epochs"], 1)
        self.assertEqual(quality_eval["distribution_interval_epochs"], 0)
        self.assertEqual(quality_eval["niqe_max_samples"], 512)
        self.assertEqual(quality_eval["distribution_max_samples"], 0)
        self.assertEqual(quality_eval["output_dir"], "artifacts/eval/g_medium_v1_stage1_long200_v3/quality")
        self.assertNotIn("real_index", quality_eval)
        self.assertNotIn("generated_dir", quality_eval)
        g_loop._validate_train_g_config(config)

    def test_medium_v1_stage1_long200_v4_resumes_v3_with_external_distribution_quality(self) -> None:
        from safa.training import g_loop

        path = Path("configs/medium_v1/train_g_medium_v1_stage1_long200_v4.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        quality_eval = config["stages"]["stage1"]["quality_eval"]

        self.assertEqual(config["out_dir"], "artifacts/checkpoints/g_medium_v1_stage1_long200_v4")
        self.assertEqual(config["resume_from"], "artifacts/checkpoints/g_medium_v1_stage1_long200_v4/last.pt")
        self.assertEqual(config["stages"]["stage1"]["epochs"], 200)
        self.assertEqual(config["stages"]["stage2"]["epochs"], 0)
        self.assertEqual(quality_eval["metrics"], ["niqe", "fid", "kid"])
        self.assertEqual(quality_eval["niqe_interval_epochs"], 1)
        self.assertEqual(quality_eval["distribution_interval_epochs"], 20)
        self.assertEqual(quality_eval["niqe_max_samples"], 512)
        self.assertEqual(quality_eval["distribution_max_samples"], 3969)
        self.assertEqual(quality_eval["distribution_timeout_seconds"], 3600)
        self.assertEqual(quality_eval["distribution_cuda_visible_devices"], "0")
        self.assertEqual(quality_eval["distribution_device"], "cuda:0")
        self.assertEqual(quality_eval["quality_num_workers"], 2)
        self.assertEqual(quality_eval["real_index"], "data/index/val_single_face.jsonl")
        self.assertEqual(quality_eval["output_dir"], "artifacts/eval/g_medium_v1_stage1_long200_v4/quality")
        self.assertNotIn("generated_dir", quality_eval)
        g_loop._validate_train_g_config(config)

    def test_medium_v1_stage2_m0_and_m1_configs_only_differ_in_loss_weighting_and_out_dir(self) -> None:
        from safa.training import g_loop

        m0_path = Path("configs/medium_v1/train_g_medium_v1_stage2_m0.yaml")
        m1_path = Path("configs/medium_v1/train_g_medium_v1_stage2_m1_uw.yaml")
        stage1_path = Path("configs/medium_v1/train_g_medium_v1_stage1.yaml")
        self.assertTrue(stage1_path.is_file())
        self.assertTrue(m0_path.is_file())
        self.assertTrue(m1_path.is_file())

        stage1 = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))
        m0 = yaml.safe_load(m0_path.read_text(encoding="utf-8"))
        m1 = yaml.safe_load(m1_path.read_text(encoding="utf-8"))

        self.assertEqual(stage1["out_dir"], "artifacts/checkpoints/g_medium_v1_stage1_m0")
        self.assertEqual(m0["resume_from"], "artifacts/checkpoints/g_medium_v1_stage1_m0/best_single_face.pt")
        self.assertEqual(m1["resume_from"], "artifacts/checkpoints/g_medium_v1_stage1_m0/best_single_face.pt")
        self.assertEqual(m0["loss_weighting"], {"type": "fixed", "flow_weight": 1.0, "cycle_weight": 0.01})
        self.assertEqual(m1["loss_weighting"]["type"], "uncertainty")
        self.assertNotIn("lambda_initial", m1["stages"]["stage2"])
        self.assertNotIn("lambda_max", m1["stages"]["stage2"])
        self.assertNotIn("lambda_growth", m1["stages"]["stage2"])

        comparable_m0 = copy.deepcopy(m0)
        comparable_m1 = copy.deepcopy(m1)
        comparable_m0.pop("out_dir")
        comparable_m1.pop("out_dir")
        comparable_m0.pop("loss_weighting")
        comparable_m1.pop("loss_weighting")
        for field in ("lambda_initial", "lambda_max", "lambda_growth"):
            comparable_m0["stages"]["stage2"].pop(field, None)
            comparable_m1["stages"]["stage2"].pop(field, None)
        self.assertEqual(comparable_m0, comparable_m1)

        g_loop._validate_train_g_config(stage1)
        g_loop._validate_train_g_config(m0)
        g_loop._validate_train_g_config(m1)


if __name__ == "__main__":
    unittest.main()
