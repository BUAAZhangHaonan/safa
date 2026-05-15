from __future__ import annotations

import argparse

from safa.training.e0_loop import train_e0_from_config
from safa.utils.config import load_yaml, require_keys


REQUIRED_KEYS = (
    "seed",
    "device",
    "num_workers",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "num_classes",
    "embedding_dim",
    "image_size",
    "imagenet_weights",
    "train_index",
    "val_index",
    "out_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AffectNet E0 emotion encoder.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    require_keys(config, REQUIRED_KEYS)
    manifest = train_e0_from_config(config)
    print(manifest)


if __name__ == "__main__":
    main()

