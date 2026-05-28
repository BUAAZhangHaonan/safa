from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


class VisualizationContractTests(unittest.TestCase):
    def test_eval_pair_visualization_writes_two_column_png(self) -> None:
        from scripts.visualize_eval_pairs import visualize_eval_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "source_a.png"
            source_b = root / "source_b.png"
            generated_a = root / "generated_a.png"
            generated_b = root / "generated_b.png"
            _write_png(source_a, (255, 0, 0))
            _write_png(source_b, (0, 255, 0))
            _write_png(generated_a, (0, 0, 255))
            _write_png(generated_b, (255, 255, 0))

            index_path = root / "index.jsonl"
            _write_jsonl(
                index_path,
                [
                    {"sample_id": "val:person/a.png", "image_path": str(source_a), "label": 1},
                    {"sample_id": "val:person/b.png", "image_path": str(source_b), "label": 2},
                ],
            )
            per_sample_path = root / "per_sample.jsonl"
            _write_jsonl(
                per_sample_path,
                [
                    {
                        "sample_id": "val:person/a.png",
                        "label": 1,
                        "affective": {"latent_cosine": 0.91, "source_prediction_preserved": 1.0},
                        "face_detection": {"count": 1, "single_face_eq1_rate": 1.0},
                        "artifacts": {"generated_image_path": str(generated_a)},
                    },
                    {
                        "sample_id": "val:person/b.png",
                        "label": 2,
                        "affective": {"latent_cosine": 0.52, "source_prediction_preserved": 0.0},
                        "face_detection": {"count": 2, "single_face_eq1_rate": 0.0},
                        "artifacts": {"generated_image_path": str(generated_b)},
                    },
                ],
            )
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps({"artifacts": {"per_sample_jsonl": str(per_sample_path)}, "dataset": {"index": str(index_path)}}),
                encoding="utf-8",
            )
            out_path = root / "pairs.png"

            visualize_eval_pairs(
                [
                    "--result-json",
                    str(result_path),
                    "--out-path",
                    str(out_path),
                    "--num-samples",
                    "2",
                    "--sort-by",
                    "sample_id",
                ]
            )

            self.assertTrue(out_path.is_file())
            self.assertGreater(out_path.stat().st_size, 0)
            with Image.open(out_path) as image:
                width, height = image.size
            self.assertEqual(width, 560)
            self.assertEqual(height, 380)

    def test_eval_pair_visualization_fails_when_source_image_is_missing(self) -> None:
        from scripts.visualize_eval_pairs import visualize_eval_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / "generated.png"
            _write_png(generated, (0, 0, 255))
            result_path = _write_minimal_eval_fixture(root, root / "missing_source.png", generated)

            with self.assertRaisesRegex(FileNotFoundError, "source image"):
                visualize_eval_pairs(["--result-json", str(result_path), "--out-path", str(root / "pairs.png"), "--num-samples", "1"])

    def test_eval_pair_visualization_fails_when_generated_image_is_missing(self) -> None:
        from scripts.visualize_eval_pairs import visualize_eval_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            _write_png(source, (255, 0, 0))
            result_path = _write_minimal_eval_fixture(root, source, root / "missing_generated.png")

            with self.assertRaisesRegex(FileNotFoundError, "generated image"):
                visualize_eval_pairs(["--result-json", str(result_path), "--out-path", str(root / "pairs.png"), "--num-samples", "1"])

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


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (24, 24), color).save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_minimal_eval_fixture(root: Path, source_path: Path, generated_path: Path) -> Path:
    index_path = root / "index.jsonl"
    per_sample_path = root / "per_sample.jsonl"
    _write_jsonl(index_path, [{"sample_id": "sample-1", "image_path": str(source_path), "label": 4}])
    _write_jsonl(
        per_sample_path,
        [
            {
                "sample_id": "sample-1",
                "label": 4,
                "affective": {"latent_cosine": 0.88, "source_prediction_preserved": 1.0},
                "face_detection": {"count": 1, "single_face_eq1_rate": 1.0},
                "artifacts": {"generated_image_path": str(generated_path)},
            }
        ],
    )
    result_path = root / "result.json"
    result_path.write_text(
        json.dumps({"artifacts": {"per_sample_jsonl": str(per_sample_path)}, "dataset": {"index": str(index_path)}}),
        encoding="utf-8",
    )
    return result_path


if __name__ == "__main__":
    unittest.main()
