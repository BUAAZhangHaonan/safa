from __future__ import annotations

import unittest

from safa.training.g_loop import _assert_stage1_gate_allows_stage2


class StageGateTests(unittest.TestCase):
    def _base_train_config(self) -> dict:
        return {
            "embedding_dim": 2,
            "image_size": 4,
            "allow_stage2_without_stage1_gate": False,
            "ema": {
                "enabled": False,
                "decay": 0.999,
                "evaluate_raw": True,
                "evaluate_ema": False,
                "save_ema_checkpoint": False,
            },
            "best_model": "raw",
            "generator": {
                "model_type": "conditional_flow_matching",
                "base_channels": 4,
                "channel_multipliers": [1],
                "time_embedding_dim": 4,
                "condition_dim": 4,
                "sample_steps": 1,
                "train_cycle_steps": 1,
                "sampler": "heun",
            },
            "stages": {
                "stage1": {
                    "epochs": 1,
                    "require_face_detection_gate": True,
                    "face_detection_threshold": 0.95,
                    "stable_epochs": 1,
                },
                "stage2": {
                    "epochs": 1,
                    "lambda_initial": 0.005,
                    "lambda_max": 0.01,
                    "lambda_growth": 0.005,
                    "gradient_conflict": {"enabled": False},
                },
            },
            "validation": {
                "enabled": True,
                "index": "val.jsonl",
                "features": "features",
                "max_samples": 8,
                "batch_size": 2,
                "face_detection": {"enabled": True, "model_name": "buffalo_l"},
            },
        }

    def test_blocks_stage2_when_detection_rate_missing(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 1}}
        with self.assertRaises(RuntimeError):
            _assert_stage1_gate_allows_stage2(stages, stable_hits=0, detection_rate=None, allow_bypass=False)

    def test_blocks_stage2_when_detection_gate_not_stable(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 2}}
        with self.assertRaises(RuntimeError):
            _assert_stage1_gate_allows_stage2(stages, stable_hits=1, detection_rate=0.99, allow_bypass=False)

    def test_allows_stage2_after_gate(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 2}}
        _assert_stage1_gate_allows_stage2(stages, stable_hits=2, detection_rate=0.99, allow_bypass=False)

    def test_smoke_bypass_is_explicit(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 1}}
        _assert_stage1_gate_allows_stage2(stages, stable_hits=0, detection_rate=0.0, allow_bypass=True)

    def test_stage1_gate_requires_explicit_gate_switch(self) -> None:
        from safa.training import g_loop

        config = self._base_train_config()
        config["stages"]["stage1"].pop("require_face_detection_gate")
        with self.assertRaisesRegex(ValueError, "stages.stage1.require_face_detection_gate"):
            g_loop._validate_train_g_config(config)

        stages = {"stage1": {"face_detection_threshold": 0.95, "stable_epochs": 1}}
        with self.assertRaisesRegex(ValueError, "stages.stage1.require_face_detection_gate"):
            _assert_stage1_gate_allows_stage2(stages, stable_hits=1, detection_rate=1.0, allow_bypass=False)

    def test_stage1_gate_requires_explicit_threshold_and_stable_epochs(self) -> None:
        with self.assertRaisesRegex(ValueError, "face_detection_threshold"):
            _assert_stage1_gate_allows_stage2(
                {"stage1": {"require_face_detection_gate": True, "stable_epochs": 1}},
                stable_hits=1,
                detection_rate=1.0,
                allow_bypass=False,
            )
        with self.assertRaisesRegex(ValueError, "stable_epochs"):
            _assert_stage1_gate_allows_stage2(
                {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95}},
                stable_hits=1,
                detection_rate=1.0,
                allow_bypass=False,
            )

    def test_stage1_gate_fields_are_not_required_when_gate_disabled(self) -> None:
        _assert_stage1_gate_allows_stage2(
            {"stage1": {"require_face_detection_gate": False}},
            stable_hits=0,
            detection_rate=None,
            allow_bypass=False,
        )

    def test_generator_config_requires_explicit_generator_block(self) -> None:
        from safa.training.g_loop import _generator_config_from_train_config

        with self.assertRaisesRegex(ValueError, "generator"):
            _generator_config_from_train_config({"embedding_dim": 2, "image_size": 4})

    def test_stage2_requires_validation_face_detection_config(self) -> None:
        import copy

        from safa.training import g_loop

        cases = [
            ("validation", lambda config: config.pop("validation")),
            ("validation.enabled", lambda config: config["validation"].pop("enabled")),
            ("validation.enabled", lambda config: config["validation"].update({"enabled": False})),
            ("validation.index", lambda config: config["validation"].pop("index")),
            ("validation.features", lambda config: config["validation"].pop("features")),
            ("validation.max_samples", lambda config: config["validation"].pop("max_samples")),
            ("validation.batch_size", lambda config: config["validation"].pop("batch_size")),
            ("validation.face_detection", lambda config: config["validation"].pop("face_detection")),
            ("validation.face_detection.enabled", lambda config: config["validation"]["face_detection"].pop("enabled")),
            ("validation.face_detection.enabled", lambda config: config["validation"]["face_detection"].update({"enabled": False})),
            ("validation.face_detection.model_name", lambda config: config["validation"]["face_detection"].pop("model_name")),
        ]
        for field, mutate in cases:
            config = copy.deepcopy(self._base_train_config())
            mutate(config)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    g_loop._validate_train_g_config(config)

    def test_stage2_accepts_explicit_validation_face_detection_config(self) -> None:
        from safa.training import g_loop

        g_loop._validate_train_g_config(self._base_train_config())

    def test_training_config_requires_explicit_validation_block_even_without_stage2(self) -> None:
        from safa.training import g_loop

        config = self._base_train_config()
        config["stages"]["stage2"]["epochs"] = 0
        config.pop("validation")

        with self.assertRaisesRegex(ValueError, "validation"):
            g_loop._validate_train_g_config(config)


    def test_quality_eval_loader_uses_explicit_non_persistent_workers(self) -> None:
        import torch
        from unittest import mock

        from safa.training import g_loop

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> dict:
                return {"image": torch.zeros(3, 4, 4), "z": torch.zeros(2), "sample_id": f"sample-{index}"}

        config = self._base_train_config()
        config.update({"e0_checkpoint": "e0.pt", "image_size": 4, "num_workers": 8})
        quality_eval = {"quality_num_workers": 2}

        with mock.patch.object(g_loop, "FeatureAlignedAffectNet", return_value=DummyDataset()):
            loader = g_loop._build_quality_eval_loader(
                config,
                3,
                quality_eval_config=quality_eval,
                quality_eval_context="stages.stage1.quality_eval",
            )

        self.assertEqual(loader.num_workers, 2)
        self.assertFalse(loader.persistent_workers)

    def test_resume_stage_progress_starts_after_completed_stage_epoch(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics, _resume_stage_start_epoch

        progress = _resume_stage_progress_from_metrics({"stage": "stage1", "stage_epoch": 51}, "last.pt")
        stages = {"stage1": {"epochs": 200}, "stage2": {"epochs": 10}}

        self.assertEqual(_resume_stage_start_epoch("stage1", stages, progress), 52)
        self.assertEqual(_resume_stage_start_epoch("stage2", stages, progress), 0)

    def test_stage1_checkpoint_initializes_stage2_only_from_epoch_zero(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics, _resume_stage_start_epoch

        progress = _resume_stage_progress_from_metrics({"stage": "stage1", "stage_epoch": 181}, "best_stage1.pt")
        stages = {"stage1": {"epochs": 0}, "stage2": {"epochs": 20}}

        self.assertEqual(_resume_stage_start_epoch("stage1", stages, progress), 0)
        self.assertEqual(_resume_stage_start_epoch("stage2", stages, progress), 0)

    def test_same_stage_resume_progress_exceeding_configured_epochs_fails_fast(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics, _resume_stage_start_epoch

        progress = _resume_stage_progress_from_metrics({"stage": "stage2", "stage_epoch": 7}, "stage2.pt")
        stages = {"stage1": {"epochs": 0}, "stage2": {"epochs": 7}}

        with self.assertRaisesRegex(ValueError, r"stage2.*exceeds configured stage2\.epochs=7"):
            _resume_stage_start_epoch("stage2", stages, progress)

    def test_stage1_checkpoint_is_not_silently_reset_without_stage2_work(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics, _resume_stage_start_epoch

        progress = _resume_stage_progress_from_metrics({"stage": "stage1", "stage_epoch": 181}, "stage1.pt")
        stages = {"stage1": {"epochs": 0}, "stage2": {"epochs": 0}}

        with self.assertRaisesRegex(ValueError, r"stage1.*exceeds configured stage1\.epochs=0"):
            _resume_stage_start_epoch("stage1", stages, progress)

    def test_stage2_checkpoint_cannot_resume_stage1_only_config(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics, _resume_stage_start_epoch

        progress = _resume_stage_progress_from_metrics({"stage": "stage2", "stage_epoch": 0}, "stage2.pt")
        stages = {"stage1": {"epochs": 10}, "stage2": {"epochs": 0}}

        self.assertEqual(_resume_stage_start_epoch("stage1", stages, progress), 10)
        with self.assertRaisesRegex(ValueError, r"stage2.*exceeds configured stage2\.epochs=0"):
            _resume_stage_start_epoch("stage2", stages, progress)

    def test_resume_stage_progress_supports_stage2_zero_based_field(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics, _resume_stage_start_epoch

        progress = _resume_stage_progress_from_metrics({"stage": "stage2", "stage_epoch_0based": 7}, "last.pt")
        stages = {"stage1": {"epochs": 200}, "stage2": {"epochs": 20}}

        self.assertEqual(_resume_stage_start_epoch("stage1", stages, progress), 200)
        self.assertEqual(_resume_stage_start_epoch("stage2", stages, progress), 8)

    def test_resume_stage_progress_requires_checkpoint_progress_field(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics

        with self.assertRaisesRegex(ValueError, "stage_epoch"):
            _resume_stage_progress_from_metrics({"stage": "stage1"}, "last.pt")

    def test_stage2_resume_does_not_recheck_stage1_gate(self) -> None:
        from safa.training.g_loop import _resume_stage_progress_from_metrics, _should_check_stage2_gate

        progress = _resume_stage_progress_from_metrics({"stage": "stage2", "stage_epoch": 3}, "last.pt")

        self.assertFalse(_should_check_stage2_gate("stage2", progress))

    def test_stage2_requires_explicit_ema_block(self) -> None:
        from safa.training import g_loop

        config = self._base_train_config()
        config.pop("ema")

        with self.assertRaisesRegex(ValueError, "ema"):
            g_loop._validate_train_g_config(config)

    def test_ema_config_requires_explicit_fields(self) -> None:
        import copy

        from safa.training import g_loop

        for field in ("enabled", "decay", "evaluate_raw", "evaluate_ema", "save_ema_checkpoint"):
            config = copy.deepcopy(self._base_train_config())
            config["ema"].pop(field)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"ema.{field}"):
                    g_loop._validate_train_g_config(config)

    def test_enabled_ema_requires_ema_evaluation_for_ema_best(self) -> None:
        from safa.training import g_loop

        config = self._base_train_config()
        config["ema"].update({"enabled": True, "evaluate_raw": True, "evaluate_ema": False, "save_ema_checkpoint": True})
        config["best_model"] = "ema"

        with self.assertRaisesRegex(ValueError, "ema.evaluate_ema"):
            g_loop._validate_train_g_config(config)

    def test_stage2_requires_explicit_best_model(self) -> None:
        from safa.training import g_loop

        config = self._base_train_config()
        config.pop("best_model")

        with self.assertRaisesRegex(ValueError, "best_model"):
            g_loop._validate_train_g_config(config)

    def test_best_model_ema_requires_enabled_ema(self) -> None:
        from safa.training import g_loop

        config = self._base_train_config()
        config["best_model"] = "ema"

        with self.assertRaisesRegex(ValueError, "ema.enabled"):
            g_loop._validate_train_g_config(config)

    def test_stage2_cycle_uses_stable_x_init_and_unclamped_sampling(self) -> None:
        import torch
        from torch import nn

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _GeneratorTrainingStep
        from safa.utils.sampling import make_x_init_for_sample_ids

        class DummyGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_calls = []

            def flow_matching_loss(self, images, z):
                loss = images.sum() * 0.0 + z.sum() * 0.0
                return loss, {"flow_matching_mse": loss.detach()}

            def sample(self, z, **kwargs):
                self.sample_calls.append(kwargs)
                return torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)

        class DummyE0(nn.Module):
            def forward(self, images):
                return {"embedding": torch.ones(images.shape[0], 2, device=images.device), "logits": torch.zeros(images.shape[0], 2, device=images.device)}

        generator = DummyGenerator()
        module = _GeneratorTrainingStep(
            generator,
            DummyE0(),
            FlowGeneratorConfig(embedding_dim=2, image_size=4, train_cycle_steps=1),
            sampling_seed=1337,
        )
        z = torch.ones(2, 2)

        module(torch.zeros(2, 3, 4, 4), z, ["sample-b", "sample-a"], True, 1.0)

        self.assertEqual(len(generator.sample_calls), 1)
        sample_kwargs = generator.sample_calls[0]
        self.assertIs(sample_kwargs["clamp_output"], False)
        self.assertIsNotNone(sample_kwargs["x_init"])
        expected = make_x_init_for_sample_ids(["sample-b", "sample-a"], 1337, 4, z.device, z.dtype)
        self.assertTrue(torch.equal(sample_kwargs["x_init"], expected))

    def test_stage2_gradient_conflict_config_requires_explicit_setting(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        with self.assertRaisesRegex(ValueError, "stages.stage2.gradient_conflict"):
            _stage2_gradient_conflict_config({"stage1": {"epochs": 0}, "stage2": {"epochs": 1}})

    def test_stage2_gradient_conflict_config_rejects_invalid_interval(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        stages = {
            "stage1": {"epochs": 0},
            "stage2": {"epochs": 1, "gradient_conflict": {"enabled": True, "interval": 0, "max_samples": 8}},
        }

        with self.assertRaisesRegex(ValueError, "interval"):
            _stage2_gradient_conflict_config(stages)

    def test_stage2_gradient_conflict_config_requires_max_samples_when_enabled(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        stages = {
            "stage1": {"epochs": 0},
            "stage2": {"epochs": 1, "gradient_conflict": {"enabled": True, "interval": 20}},
        }

        with self.assertRaisesRegex(ValueError, "max_samples"):
            _stage2_gradient_conflict_config(stages)

    def test_stage2_gradient_conflict_config_rejects_invalid_max_samples(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        for value in (0, -1, True, 1.5):
            stages = {
                "stage1": {"epochs": 0},
                "stage2": {"epochs": 1, "gradient_conflict": {"enabled": True, "interval": 20, "max_samples": value}},
            }
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_samples"):
                _stage2_gradient_conflict_config(stages)

    def test_stage2_gradient_conflict_config_records_max_samples(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        config = _stage2_gradient_conflict_config(
            {"stage1": {"epochs": 0}, "stage2": {"epochs": 1, "gradient_conflict": {"enabled": True, "interval": 20, "max_samples": 8}}}
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.interval, 20)
        self.assertEqual(config.max_samples, 8)

    def test_stage2_gradient_conflict_config_is_not_required_without_stage2(self) -> None:
        from safa.training.g_loop import _stage2_gradient_conflict_config

        config = _stage2_gradient_conflict_config({"stage1": {"epochs": 1}, "stage2": {"epochs": 0}})

        self.assertFalse(config.enabled)

    def test_stage2_gradient_conflict_monitor_batch_slices_without_changing_training_batch(self) -> None:
        import torch

        from safa.training.g_loop import _slice_gradient_conflict_monitor_batch

        images = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
        z = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        sample_ids = ["a", "b", "c", "d"]

        sampled_images, sampled_z, sampled_ids, sample_size, full_batch_size = _slice_gradient_conflict_monitor_batch(
            images,
            z,
            sample_ids,
            max_samples=2,
        )

        self.assertEqual(sample_size, 2)
        self.assertEqual(full_batch_size, 4)
        self.assertEqual(tuple(sampled_images.shape), (2, 3, 2, 2))
        self.assertEqual(tuple(sampled_z.shape), (2, 2))
        self.assertEqual(sampled_ids, ["a", "b"])
        self.assertEqual(tuple(images.shape), (4, 3, 2, 2))
        self.assertEqual(tuple(z.shape), (4, 2))
        self.assertEqual(sample_ids, ["a", "b", "c", "d"])

    def test_stage2_gradient_conflict_monitor_batch_rejects_max_samples_larger_than_batch(self) -> None:
        import torch

        from safa.training.g_loop import _slice_gradient_conflict_monitor_batch

        with self.assertRaisesRegex(ValueError, "max_samples"):
            _slice_gradient_conflict_monitor_batch(
                torch.zeros(4, 3, 2, 2),
                torch.zeros(4, 2),
                ["a", "b", "c", "d"],
                max_samples=5,
            )

    def test_stage2_gradient_conflict_validation_rejects_max_samples_larger_than_train_batch(self) -> None:
        from safa.training import g_loop

        config = self._base_train_config()
        config["batch_size"] = 4
        config["stages"]["stage2"]["gradient_conflict"] = {"enabled": True, "interval": 20, "max_samples": 5}

        with self.assertRaisesRegex(ValueError, "max_samples"):
            g_loop._validate_train_g_config(config)

    def test_stage2_gradient_conflict_metrics_compute_cosine_and_norms(self) -> None:
        import torch

        from safa.training.g_loop import _compute_gradient_conflict_metrics

        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        flow_loss = parameter[0] * 2.0
        cycle_loss = parameter[1] * 3.0

        metrics = _compute_gradient_conflict_metrics(
            flow_loss,
            cycle_loss,
            [parameter],
            lambda_cycle=0.01,
            sample_size=2,
            full_batch_size=4,
        )

        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle"], 0.0, places=6)
        self.assertAlmostEqual(metrics["gradient_norm_fm"], 2.0, places=6)
        self.assertAlmostEqual(metrics["gradient_norm_cycle"], 3.0, places=6)
        self.assertAlmostEqual(metrics["weighted_gradient_norm_cycle"], 0.03, places=6)
        self.assertAlmostEqual(metrics["weighted_gradient_ratio_cycle_to_fm"], 0.015, places=6)
        self.assertEqual(metrics["gradient_conflict_sample_size"], 2)
        self.assertEqual(metrics["gradient_conflict_full_batch_size"], 4)

    def test_stage2_gradient_conflict_metrics_reject_zero_norm_gradient(self) -> None:
        import torch

        from safa.training.g_loop import _compute_gradient_conflict_metrics

        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        flow_loss = parameter[0] * 0.0
        cycle_loss = parameter[1] * 3.0

        with self.assertRaisesRegex(RuntimeError, "zero norm"):
            _compute_gradient_conflict_metrics(
                flow_loss,
                cycle_loss,
                [parameter],
                lambda_cycle=0.01,
                sample_size=2,
                full_batch_size=4,
            )

    def test_checkpoint_composite_uses_single_face_eq1_rate_not_legacy_ge1(self) -> None:
        from safa.training.g_loop import _composite_score

        ge1_high_single_low = {
            "validation_latent_cosine_mean": 0.90,
            "validation_face_detection_rate": 1.00,
            "validation_single_face_eq1_rate": 0.10,
        }
        ge1_low_single_high = {
            "validation_latent_cosine_mean": 0.80,
            "validation_face_detection_rate": 0.20,
            "validation_single_face_eq1_rate": 0.90,
        }

        self.assertAlmostEqual(_composite_score(ge1_high_single_low), 0.09)
        self.assertAlmostEqual(_composite_score(ge1_low_single_high), 0.72)
        self.assertGreater(_composite_score(ge1_low_single_high), _composite_score(ge1_high_single_low))

    def test_checkpoint_composite_requires_single_face_eq1_rate(self) -> None:
        from safa.training.g_loop import _composite_score

        with self.assertRaisesRegex(KeyError, "validation_single_face_eq1_rate"):
            _composite_score({"validation_latent_cosine_mean": 0.90, "validation_face_detection_rate": 1.00})

    def test_checkpoint_writer_validates_composite_metrics_before_writing(self) -> None:
        import tempfile
        from pathlib import Path

        import torch

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _save_generator

        generator = torch.nn.Linear(2, 2)
        generator_config = FlowGeneratorConfig(
            embedding_dim=2,
            image_size=4,
            sample_steps=1,
            train_cycle_steps=1,
        )
        metrics = {"stage": "stage2", "loss": 1.0, "validation_latent_cosine_mean": 0.90}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "last.pt"
            with self.assertRaisesRegex(ValueError, "validation_single_face_eq1_rate"):
                _save_generator(
                    checkpoint_path,
                    generator,
                    generator_config,
                    {
                        "stages": {},
                        "validation": {},
                        "ema": {"enabled": False, "decay": 0.999, "evaluate_raw": True, "evaluate_ema": False, "save_ema_checkpoint": False},
                        "best_model": "raw",
                    },
                    metrics,
                    [],
                )
            self.assertFalse(checkpoint_path.exists())

    def test_epoch_metrics_include_gradient_conflict_when_recorded(self) -> None:
        import torch

        from safa.training.g_loop import _reduce_epoch_metrics
        from safa.utils.distributed import DistributedContext

        totals = {
            "loss": 8.0,
            "flow_matching_mse": 4.0,
            "cycle": 2.0,
            "grad_norm": 0.0,
            "gradient_conflict_count": 2.0,
            "gradient_cosine_fm_cycle": -0.5,
            "gradient_norm_fm": 4.0,
            "gradient_norm_cycle": 6.0,
            "weighted_gradient_norm_cycle": 0.06,
            "weighted_gradient_ratio_cycle_to_fm": 0.03,
            "gradient_conflict_sample_size": 4.0,
            "gradient_conflict_full_batch_size": 8.0,
            "gradient_conflict_samples": [
                {
                    "gradient_cosine_fm_cycle": -0.5,
                    "gradient_norm_fm": 2.0,
                    "gradient_norm_cycle": 3.0,
                    "weighted_gradient_norm_cycle": 0.03,
                    "weighted_gradient_ratio_cycle_to_fm": 0.015,
                },
                {
                    "gradient_cosine_fm_cycle": 0.0,
                    "gradient_norm_fm": 2.0,
                    "gradient_norm_cycle": 3.0,
                    "weighted_gradient_norm_cycle": 0.03,
                    "weighted_gradient_ratio_cycle_to_fm": 0.015,
                },
            ],
        }

        distributed = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            device=torch.device("cpu"),
            backend="single",
        )

        metrics = _reduce_epoch_metrics(totals, seen=4, device=torch.device("cpu"), distributed=distributed)

        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle"], -0.25)
        self.assertAlmostEqual(metrics["gradient_norm_fm"], 2.0)
        self.assertAlmostEqual(metrics["gradient_norm_cycle"], 3.0)
        self.assertAlmostEqual(metrics["gradient_conflict_sample_size"], 2.0)
        self.assertAlmostEqual(metrics["gradient_conflict_full_batch_size"], 4.0)

    def test_epoch_metrics_include_gradient_quantiles_norm_ratio_and_conflict_fraction(self) -> None:
        import torch

        from safa.training.g_loop import _reduce_epoch_metrics
        from safa.utils.distributed import DistributedContext

        totals = {
            "loss": 8.0,
            "flow_matching_mse": 4.0,
            "cycle": 2.0,
            "grad_norm": 0.0,
            "gradient_conflict_count": 3.0,
            "gradient_cosine_fm_cycle": 0.0,
            "gradient_norm_fm": 6.0,
            "gradient_norm_cycle": 12.0,
            "weighted_gradient_norm_cycle": 0.12,
            "weighted_gradient_ratio_cycle_to_fm": 0.06,
            "gradient_conflict_sample_size": 6.0,
            "gradient_conflict_full_batch_size": 12.0,
            "gradient_conflict_samples": [
                {
                    "gradient_cosine_fm_cycle": -1.0,
                    "gradient_norm_fm": 1.0,
                    "gradient_norm_cycle": 2.0,
                    "weighted_gradient_norm_cycle": 0.02,
                    "weighted_gradient_ratio_cycle_to_fm": 0.02,
                },
                {
                    "gradient_cosine_fm_cycle": 0.0,
                    "gradient_norm_fm": 2.0,
                    "gradient_norm_cycle": 4.0,
                    "weighted_gradient_norm_cycle": 0.04,
                    "weighted_gradient_ratio_cycle_to_fm": 0.02,
                },
                {
                    "gradient_cosine_fm_cycle": 1.0,
                    "gradient_norm_fm": 3.0,
                    "gradient_norm_cycle": 6.0,
                    "weighted_gradient_norm_cycle": 0.06,
                    "weighted_gradient_ratio_cycle_to_fm": 0.02,
                },
            ],
        }
        distributed = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            device=torch.device("cpu"),
            backend="single",
        )

        metrics = _reduce_epoch_metrics(totals, seen=4, device=torch.device("cpu"), distributed=distributed)

        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle_mean"], 0.0)
        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle_p10"], -0.8)
        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle_p50"], 0.0)
        self.assertAlmostEqual(metrics["gradient_cosine_fm_cycle_p90"], 0.8)
        self.assertAlmostEqual(metrics["gradient_norm_ratio_cycle_to_fm_mean"], 2.0)
        self.assertAlmostEqual(metrics["gradient_conflict_fraction"], 1.0 / 3.0)

    def test_epoch_metrics_include_weighted_gradient_norm_and_ratio(self) -> None:
        import torch

        from safa.training.g_loop import _reduce_epoch_metrics
        from safa.utils.distributed import DistributedContext

        totals = {
            "loss": 8.0,
            "flow_matching_mse": 4.0,
            "cycle": 2.0,
            "grad_norm": 0.0,
            "gradient_conflict_count": 2.0,
            "gradient_cosine_fm_cycle": 0.0,
            "gradient_norm_fm": 4.0,
            "gradient_norm_cycle": 6.0,
            "weighted_gradient_norm_cycle": 0.06,
            "weighted_gradient_ratio_cycle_to_fm": 0.03,
            "gradient_conflict_sample_size": 4.0,
            "gradient_conflict_full_batch_size": 8.0,
            "gradient_conflict_samples": [
                {
                    "gradient_cosine_fm_cycle": -0.5,
                    "gradient_norm_fm": 2.0,
                    "gradient_norm_cycle": 3.0,
                    "weighted_gradient_norm_cycle": 0.03,
                    "weighted_gradient_ratio_cycle_to_fm": 0.015,
                },
                {
                    "gradient_cosine_fm_cycle": 0.5,
                    "gradient_norm_fm": 2.0,
                    "gradient_norm_cycle": 3.0,
                    "weighted_gradient_norm_cycle": 0.03,
                    "weighted_gradient_ratio_cycle_to_fm": 0.015,
                },
            ],
        }
        distributed = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            device=torch.device("cpu"),
            backend="single",
        )

        metrics = _reduce_epoch_metrics(totals, seen=4, device=torch.device("cpu"), distributed=distributed)

        self.assertAlmostEqual(metrics["gradient_norm_cycle_mean"], 3.0)
        self.assertAlmostEqual(metrics["gradient_norm_ratio_cycle_to_fm_mean"], 1.5)
        self.assertAlmostEqual(metrics["weighted_gradient_norm_cycle_mean"], 0.03)
        self.assertAlmostEqual(metrics["weighted_gradient_ratio_cycle_to_fm_mean"], 0.015)

    def test_weighted_gradient_ratio_rejects_zero_flow_norm_like_raw_ratio(self) -> None:
        from safa.training.g_loop import _summarize_gradient_conflict_samples

        with self.assertRaisesRegex(RuntimeError, "Gradient norm ratio contains non-finite values"):
            _summarize_gradient_conflict_samples(
                [
                    {
                        "gradient_cosine_fm_cycle": 0.0,
                        "gradient_norm_fm": 0.0,
                        "gradient_norm_cycle": 3.0,
                        "weighted_gradient_norm_cycle": 0.03,
                        "weighted_gradient_ratio_cycle_to_fm": 0.015,
                    }
                ]
            )

    def test_epoch_metrics_reject_zero_seen_samples(self) -> None:
        import torch

        from safa.training.g_loop import _reduce_epoch_metrics
        from safa.utils.distributed import DistributedContext

        distributed = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            device=torch.device("cpu"),
            backend="single",
        )
        totals = {
            "loss": 0.0,
            "flow_matching_mse": 0.0,
            "cycle": 0.0,
            "grad_norm": 0.0,
        }

        with self.assertRaisesRegex(RuntimeError, "zero samples"):
            _reduce_epoch_metrics(totals, seen=0, device=torch.device("cpu"), distributed=distributed)

    def test_training_scalar_metrics_reject_non_finite_values(self) -> None:
        import torch

        from safa.training import g_loop

        cases = [
            ("loss", torch.tensor(float("nan")), torch.tensor(0.0), torch.tensor(0.0)),
            ("flow_matching_mse", torch.tensor(0.0), torch.tensor(float("inf")), torch.tensor(0.0)),
            ("cycle", torch.tensor(0.0), torch.tensor(0.0), torch.tensor(float("nan"))),
        ]
        for metric_name, loss, flow_mse, cycle in cases:
            with self.subTest(metric_name=metric_name):
                with self.assertRaisesRegex(RuntimeError, metric_name):
                    g_loop._assert_finite_training_scalars(loss, flow_mse, cycle)

    def test_first_epoch_checkpoint_comparison_requires_validation_composite_metrics(self) -> None:
        from safa.training.g_loop import _is_better, _is_better_overall

        metrics = {"stage": "stage2", "loss": 1.0}
        with self.assertRaisesRegex(ValueError, "validation_latent_cosine_mean"):
            _is_better(metrics, [])
        with self.assertRaisesRegex(ValueError, "validation_latent_cosine_mean"):
            _is_better_overall(metrics, [])

    def test_checkpoint_comparison_requires_explicit_stage(self) -> None:
        from safa.training.g_loop import _is_better

        metrics = {
            "loss": 1.0,
            "validation_latent_cosine_mean": 0.9,
            "validation_single_face_eq1_rate": 0.8,
        }

        with self.assertRaisesRegex(ValueError, "stage"):
            _is_better(metrics, [])

    def test_checkpoint_selection_uses_ema_metrics_when_configured(self) -> None:
        from safa.training.g_loop import _is_better

        previous = {
            "stage": "stage2",
            "loss": 1.0,
            "validation_raw_latent_cosine_mean": 0.20,
            "validation_raw_single_face_eq1_rate": 0.20,
            "validation_ema_latent_cosine_mean": 0.90,
            "validation_ema_single_face_eq1_rate": 0.90,
            "validation_latent_cosine_mean": 0.20,
            "validation_single_face_eq1_rate": 0.20,
        }
        current = {
            "stage": "stage2",
            "loss": 0.5,
            "validation_raw_latent_cosine_mean": 1.00,
            "validation_raw_single_face_eq1_rate": 1.00,
            "validation_ema_latent_cosine_mean": 0.50,
            "validation_ema_single_face_eq1_rate": 0.50,
            "validation_latent_cosine_mean": 1.00,
            "validation_single_face_eq1_rate": 1.00,
        }

        self.assertFalse(_is_better(current, [previous], best_model="ema"))
        self.assertTrue(_is_better(current, [previous], best_model="raw"))

    def test_stage1_single_face_checkpoint_prefers_zero_multi_face_over_composite(self) -> None:
        from safa.training.g_loop import _is_better_single_face_stage1

        previous = {
            "stage": "stage1",
            "stage_epoch": 0,
            "loss": 1.0,
            "validation_raw_single_face_eq1_rate": 0.80,
            "validation_raw_multi_face_rate": 0.0,
            "validation_raw_zero_face_rate": 0.20,
            "validation_raw_face_detect_ge1_rate": 0.80,
            "validation_raw_latent_cosine_mean": 0.70,
            "validation_single_face_eq1_rate": 0.80,
            "validation_latent_cosine_mean": 0.70,
        }
        current = {
            "stage": "stage1",
            "stage_epoch": 1,
            "loss": 0.1,
            "validation_raw_single_face_eq1_rate": 0.95,
            "validation_raw_multi_face_rate": 0.05,
            "validation_raw_zero_face_rate": 0.0,
            "validation_raw_face_detect_ge1_rate": 1.0,
            "validation_raw_latent_cosine_mean": 0.99,
            "validation_single_face_eq1_rate": 0.95,
            "validation_latent_cosine_mean": 0.99,
        }

        self.assertFalse(_is_better_single_face_stage1(current, [previous]))

    def test_stage1_single_face_checkpoint_uses_pure_quality_tie_breaks(self) -> None:
        from safa.training.g_loop import _is_better_single_face_stage1

        baseline = {
            "stage": "stage1",
            "stage_epoch": 0,
            "loss": 1.0,
            "validation_raw_single_face_eq1_rate": 0.90,
            "validation_raw_multi_face_rate": 0.0,
            "validation_raw_zero_face_rate": 0.10,
            "validation_raw_face_detect_ge1_rate": 0.85,
            "validation_raw_latent_cosine_mean": 0.90,
            "validation_single_face_eq1_rate": 0.90,
            "validation_latent_cosine_mean": 0.90,
        }

        lower_zero = dict(baseline, stage_epoch=1, loss=5.0, validation_raw_zero_face_rate=0.05)
        higher_ge1 = dict(baseline, stage_epoch=1, loss=5.0, validation_raw_face_detect_ge1_rate=0.95)
        lower_loss = dict(baseline, stage_epoch=1, loss=0.5)
        later_epoch = dict(baseline, stage_epoch=1)

        self.assertTrue(_is_better_single_face_stage1(lower_zero, [baseline]))
        self.assertTrue(_is_better_single_face_stage1(higher_ge1, [baseline]))
        self.assertTrue(_is_better_single_face_stage1(lower_loss, [baseline]))
        self.assertFalse(_is_better_single_face_stage1(later_epoch, [baseline]))

    def test_stage1_single_face_checkpoint_saves_epoch_named_new_best_only(self) -> None:
        from safa.training.g_loop import _stage1_single_face_checkpoint_filenames_to_save

        previous = {
            "stage": "stage1",
            "stage_epoch": 5,
            "loss": 1.0,
            "validation_raw_single_face_eq1_rate": 0.80,
            "validation_raw_multi_face_rate": 0.0,
            "validation_raw_zero_face_rate": 0.20,
            "validation_raw_face_detect_ge1_rate": 0.80,
            "validation_raw_latent_cosine_mean": 0.70,
            "validation_single_face_eq1_rate": 0.80,
            "validation_latent_cosine_mean": 0.70,
        }
        current = dict(previous, stage_epoch=6, validation_raw_single_face_eq1_rate=0.90)

        self.assertEqual(
            _stage1_single_face_checkpoint_filenames_to_save(current, [previous]),
            ["best_single_face.pt", "best_single_face_epoch_0007.pt"],
        )
        self.assertEqual(_stage1_single_face_checkpoint_filenames_to_save(previous, [current]), [])

    def test_checkpoint_writer_persists_ema_payload_fields(self) -> None:
        import tempfile
        from pathlib import Path

        import torch

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _save_generator

        generator = torch.nn.Linear(2, 2)
        generator_config = FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1, train_cycle_steps=1)
        metrics_raw = {
            "latent_cosine_mean": 0.50,
            "single_face_eq1_rate": 0.60,
        }
        metrics_ema = {
            "latent_cosine_mean": 0.90,
            "single_face_eq1_rate": 0.80,
        }
        metrics = {
            "stage": "stage2",
            "loss": 1.0,
            "validation_raw_latent_cosine_mean": 0.50,
            "validation_raw_single_face_eq1_rate": 0.60,
            "validation_ema_latent_cosine_mean": 0.90,
            "validation_ema_single_face_eq1_rate": 0.80,
            "validation_latent_cosine_mean": 0.50,
            "validation_single_face_eq1_rate": 0.60,
        }
        ema_config = {
            "enabled": True,
            "decay": 0.999,
            "evaluate_raw": True,
            "evaluate_ema": True,
            "save_ema_checkpoint": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "last.pt"
            _save_generator(
                checkpoint_path,
                generator,
                generator_config,
                {"stages": {}, "validation": {}, "ema": ema_config, "best_model": "ema"},
                metrics,
                [],
                ema_model_state_dict=generator.state_dict(),
                metrics_raw=metrics_raw,
                metrics_ema=metrics_ema,
                ema_config=ema_config,
                best_model="ema",
            )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.assertIn("model_state_dict", payload)
        self.assertIn("ema_model_state_dict", payload)
        self.assertEqual(payload["model_config"], generator_config.to_dict())
        self.assertEqual(payload["metrics_raw"], metrics_raw)
        self.assertEqual(payload["metrics_ema"], metrics_ema)
        self.assertEqual(payload["ema_config"], ema_config)

    def test_checkpoint_writer_rejects_enabled_ema_without_state_dict(self) -> None:
        import tempfile
        from pathlib import Path

        import torch

        from safa.models.generator import FlowGeneratorConfig
        from safa.training.g_loop import _save_generator

        generator = torch.nn.Linear(2, 2)
        generator_config = FlowGeneratorConfig(embedding_dim=2, image_size=4, sample_steps=1, train_cycle_steps=1)
        metrics = {
            "stage": "stage2",
            "loss": 1.0,
            "validation_ema_latent_cosine_mean": 0.90,
            "validation_ema_single_face_eq1_rate": 0.80,
            "validation_latent_cosine_mean": 0.50,
            "validation_single_face_eq1_rate": 0.60,
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "last.pt"
            with self.assertRaisesRegex(ValueError, "ema_model_state_dict"):
                _save_generator(
                    checkpoint_path,
                    generator,
                    generator_config,
                    {"stages": {}, "validation": {}, "best_model": "ema"},
                    metrics,
                    [],
                    ema_config={"enabled": True, "save_ema_checkpoint": True},
                    best_model="ema",
                )

    def test_legacy_stage1_resume_history_requires_explicit_gate(self) -> None:
        from safa.training import g_loop

        history = [{"stage": "stage1", "loss": 1.0, "validation_latent_cosine_mean": 0.8}]
        stages = {"stage1": {"epochs": 0}, "stage2": {"epochs": 5}}
        config = {"resume_from_legacy_stage1_metrics": False}

        with self.assertRaisesRegex(ValueError, "resume_from_legacy_stage1_metrics"):
            g_loop._resume_history_for_checkpoint_selection(history, "legacy.pt", config, stages)

        config["resume_from_legacy_stage1_metrics"] = True
        self.assertEqual(g_loop._resume_history_for_checkpoint_selection(history, "legacy.pt", config, stages), [])


if __name__ == "__main__":
    unittest.main()
