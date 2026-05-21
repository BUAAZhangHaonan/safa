from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
import sys
import types
import unittest


class _TensorStub:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return self._values


@contextmanager
def _torch_module():
    try:
        import torch
    except ModuleNotFoundError:
        torch = types.SimpleNamespace(
            device=lambda name: name,
            float64=object(),
            tensor=lambda values, device=None, dtype=None: _TensorStub(values),
        )
        previous = sys.modules.get("torch")
        sys.modules["torch"] = torch
        try:
            yield torch
        finally:
            if previous is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous
    else:
        yield torch


class TrainMetricsContractTests(unittest.TestCase):
    def test_reduce_train_metrics_returns_loss_and_samples(self) -> None:
        from safa.utils.distributed import DistributedContext, reduce_train_metrics

        with _torch_module() as torch:
            device = torch.device("cpu")
            context = DistributedContext(
                enabled=False,
                rank=0,
                local_rank=0,
                world_size=1,
                is_main=True,
                device=device,
                backend="single",
            )

            metrics = reduce_train_metrics(6.0, 3, device, context)

        self.assertEqual(metrics["loss"], 2.0)
        self.assertEqual(metrics["samples"], 3)
        self.assertNotIn("train_loss", metrics)

    def test_e0_loop_reads_reduced_loss_contract(self) -> None:
        source = Path("src/safa/training/e0_loop.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        keys = {
            node.slice.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "train_metrics"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }

        self.assertIn("loss", keys)
        self.assertNotIn("train_loss", keys)


if __name__ == "__main__":
    unittest.main()
