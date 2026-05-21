from __future__ import annotations

import argparse
from pathlib import Path

from safa.training.audit import audit_no_identity_supervision
from safa.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit training config/source for forbidden identity supervision.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", nargs="*", default=["src/safa/training", "src/safa/models"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    audit_no_identity_supervision(config, [Path(item) for item in args.source])
    print("static audit passed: no identity supervision terms found in config or source text")


if __name__ == "__main__":
    main()
