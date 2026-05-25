from __future__ import annotations

from pathlib import Path
import unittest


class VisualizationContractTests(unittest.TestCase):
    def test_visualize_results_requires_checkpoint_image_size(self) -> None:
        from scripts.visualize_results import _checkpoint_image_size

        with self.assertRaisesRegex(ValueError, "model_config.image_size"):
            _checkpoint_image_size({"model_config": {"embedding_dim": 128}})

    def test_visualize_multi_requires_checkpoint_image_size(self) -> None:
        from scripts.visualize_multi import _checkpoint_image_size

        with self.assertRaisesRegex(ValueError, "model_config.image_size"):
            _checkpoint_image_size({"model_config": {"embedding_dim": 128}})

    def test_visualization_sources_do_not_use_224_fallbacks(self) -> None:
        for path in (Path("scripts/visualize_results.py"), Path("scripts/visualize_multi.py")):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path)):
                self.assertNotIn("eval_transform(224)", source)
                self.assertNotIn("getattr(g, \"image_size\", 224)", source)
                self.assertNotIn("getattr(generator, \"image_size\", 224)", source)

    def test_visualize_multi_uses_raw_single_face_metrics_for_raw_checkpoints(self) -> None:
        from scripts.visualize_multi import _checkpoint_metric_summary

        summary = _checkpoint_metric_summary(
            {
                "validation_raw_latent_cosine_mean": 0.9,
                "validation_raw_single_face_eq1_rate": 0.8,
                "validation_raw_face_detect_ge1_rate": 1.0,
                "validation_face_detection_rate": 1.0,
            },
            checkpoint_model="raw",
            checkpoint_label="unit",
        )

        self.assertEqual(summary["cosine"], 0.9)
        self.assertEqual(summary["single_face_eq1"], 0.8)
        self.assertEqual(summary["face_detect_ge1"], 1.0)
        self.assertEqual(summary["single_face_source"], "validation_raw_single_face_eq1_rate")

    def test_visualize_multi_fails_fast_when_ema_checkpoint_only_has_raw_metrics(self) -> None:
        from scripts.visualize_multi import _checkpoint_metric_summary

        with self.assertRaisesRegex(ValueError, "EMA checkpoint metrics require validation_ema_single_face_eq1_rate"):
            _checkpoint_metric_summary(
                {
                    "validation_raw_latent_cosine_mean": 0.9,
                    "validation_raw_single_face_eq1_rate": 0.8,
                },
                checkpoint_model="ema",
                checkpoint_label="unit",
            )


if __name__ == "__main__":
    unittest.main()
