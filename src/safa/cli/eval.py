from __future__ import annotations

import argparse

from safa.evaluation.runner import run_eval_from_config
from safa.utils.config import load_yaml, require_keys


REQUIRED_KEYS = (
    "seed",
    "device",
    "num_workers",
    "batch_size",
    "image_size",
    "index",
    "features",
    "e0_checkpoint",
    "g_checkpoint",
    "out_json",
    "sample_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate samplewise affective face anonymization.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    require_keys(config, REQUIRED_KEYS)
    result = run_eval_from_config(config)
    print(result["out_json"] if "out_json" in result else config["out_json"])


if __name__ == "__main__":
    main()

