from __future__ import annotations

import argparse

from safa.training.e0_loop import REQUIRED_E0_TRAIN_KEYS, train_e0_from_config
from safa.utils.config import load_yaml, require_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AffectNet E0 emotion encoder.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    require_keys(config, REQUIRED_E0_TRAIN_KEYS)
    manifest = train_e0_from_config(config)
    print(manifest)


if __name__ == "__main__":
    main()
