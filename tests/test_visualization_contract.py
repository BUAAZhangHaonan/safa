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


if __name__ == "__main__":
    unittest.main()
