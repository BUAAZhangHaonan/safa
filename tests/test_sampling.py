from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for sampling tests")
class StableSamplingTests(unittest.TestCase):
    def test_stable_sample_seed_is_stable_and_sample_specific(self) -> None:
        from safa.utils.sampling import stable_sample_seed

        self.assertEqual(stable_sample_seed(1337, "sample-a"), stable_sample_seed(1337, "sample-a"))
        self.assertNotEqual(stable_sample_seed(1337, "sample-a"), stable_sample_seed(1337, "sample-b"))

    def test_make_x_init_is_order_independent(self) -> None:
        import torch

        from safa.utils.sampling import make_x_init_for_sample_ids

        ordered = make_x_init_for_sample_ids(["a", "b", "c"], 1337, 8, torch.device("cpu"), torch.float32)
        shuffled = make_x_init_for_sample_ids(["c", "a", "b"], 1337, 8, torch.device("cpu"), torch.float32)

        self.assertTrue(torch.equal(ordered[0], shuffled[1]))
        self.assertTrue(torch.equal(ordered[1], shuffled[2]))
        self.assertTrue(torch.equal(ordered[2], shuffled[0]))

    def test_make_x_init_differs_for_different_samples(self) -> None:
        import torch

        from safa.utils.sampling import make_x_init_for_sample_ids

        x_init = make_x_init_for_sample_ids(["a", "b"], 1337, 8, torch.device("cpu"), torch.float32)

        self.assertFalse(torch.equal(x_init[0], x_init[1]))

    def test_make_x_init_respects_shape_dtype_and_device(self) -> None:
        import torch

        from safa.utils.sampling import make_x_init_for_sample_ids
        device = torch.device("cpu")
        dtype = torch.float64

        x_init = make_x_init_for_sample_ids(["a", "b"], 1337, 8, device, dtype)

        self.assertEqual(tuple(x_init.shape), (2, 3, 8, 8))
        self.assertEqual(x_init.device, device)
        self.assertEqual(x_init.dtype, dtype)

    def test_sampling_base_seed_requires_sampling_or_global_seed(self) -> None:
        from safa.utils.sampling import sampling_base_seed_from_config

        self.assertEqual(sampling_base_seed_from_config({"sampling_seed": 9, "seed": 1}), 9)
        self.assertEqual(sampling_base_seed_from_config({"seed": 1}), 1)
        with self.assertRaises(KeyError):
            sampling_base_seed_from_config({})


if __name__ == "__main__":
    unittest.main()
