from __future__ import annotations

import argparse

from safa.training.g_loop import train_g_from_config
from safa.utils.config import load_yaml, require_keys


REQUIRED_KEYS = (
    "seed",
    "device",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "image_size",
    "embedding_dim",
    "train_index",
    "train_features",
    "e0_checkpoint",
    "out_dir",
    "generator",
    "stages",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional flow matching generator G.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    require_keys(config, REQUIRED_KEYS)
    if "batch_size" not in config and ("global_batch_size" not in config or "per_device_batch_size" not in config):
        raise KeyError("Config must define either legacy batch_size or both global_batch_size and per_device_batch_size")
    print(train_g_from_config(config))


if __name__ == "__main__":
    main()
