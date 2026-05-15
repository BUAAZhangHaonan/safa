from __future__ import annotations

import argparse
from pathlib import Path
import json

from safa.data.affectnet_index import build_affectnet_index
from safa.data.index_schema import write_index
from safa.training.cache_e0 import cache_e0_from_config
from safa.training.g_loop import train_g_from_config
from safa.utils.config import load_yaml, require_keys


REQUIRED_KEYS = ("seed", "device", "num_workers", "batch_size", "image_size", "limit", "root", "work_dir", "e0_checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal closed-loop smoke validation.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    require_keys(config, REQUIRED_KEYS)
    work_dir = Path(config["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    index_path = work_dir / "smoke_index.jsonl"
    records = build_affectnet_index(
        root=Path(config["root"]),
        default_split="smoke",
        dataset_version="affectnet-smoke",
        limit=int(config["limit"]),
    )
    write_index(records, index_path)
    feature_dir = work_dir / "features"
    cache_manifest = cache_e0_from_config(
        {
            "seed": config["seed"],
            "device": config["device"],
            "num_workers": config["num_workers"],
            "batch_size": config["batch_size"],
            "image_size": config["image_size"],
            "index": str(index_path),
            "checkpoint": config["e0_checkpoint"],
            "out_dir": str(feature_dir),
        }
    )
    g_manifest = train_g_from_config(
        {
            "seed": config["seed"],
            "device": config["device"],
            "num_workers": config["num_workers"],
            "batch_size": config["batch_size"],
            "epochs": 1,
            "learning_rate": 0.0002,
            "weight_decay": 0.0,
            "image_size": config["image_size"],
            "embedding_dim": 512,
            "train_index": str(index_path),
            "train_features": str(feature_dir),
            "e0_checkpoint": config["e0_checkpoint"],
            "out_dir": str(work_dir / "g"),
            "loss_weights": {"cycle": 1.0, "semantic_ce": 0.25, "image_tv": 0.001},
        }
    )
    result = {
        "index": str(index_path),
        "num_records": len(records),
        "cache": cache_manifest,
        "generator": g_manifest,
    }
    out_json = work_dir / "smoke_result.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()

