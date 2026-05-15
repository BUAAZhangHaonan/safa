from __future__ import annotations

import argparse

from safa.training.cache_e0 import cache_e0_from_config
from safa.utils.config import load_yaml, require_keys


REQUIRED_KEYS = ("seed", "device", "num_workers", "batch_size", "image_size", "index", "checkpoint", "out_dir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen E0 embeddings.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    require_keys(config, REQUIRED_KEYS)
    print(cache_e0_from_config(config))


if __name__ == "__main__":
    main()

