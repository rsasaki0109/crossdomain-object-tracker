#!/usr/bin/env python3
"""Download datasets for cross-domain evaluation.

Standalone script to download sample data from all supported datasets.

Usage:
    python scripts/download_datasets.py --dataset covla
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --all --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path so this script works standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossdomain_object_tracker.common.config import load_datasets_config
from crossdomain_object_tracker.common.download import download_dataset


def main() -> None:
    """Download datasets based on CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download datasets for cross-domain evaluation."
    )
    parser.add_argument("--dataset", type=str, help="Dataset name to download")
    parser.add_argument(
        "--all", action="store_true", help="Download all configured datasets"
    )
    parser.add_argument(
        "--dest", type=str, default="data/", help="Destination directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Download only sample images for quick testing",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.dataset and not args.all:
        parser.print_help()
        return

    if args.all:
        config = load_datasets_config()
        dataset_names = list(config.get("datasets", {}).keys())
    else:
        dataset_names = [args.dataset]

    dest = Path(args.dest)
    print(f"Downloading {len(dataset_names)} dataset(s) to {dest}")
    print()

    for name in dataset_names:
        print(f"--- {name} ---")
        try:
            path = download_dataset(
                name,
                output_dir=dest,
                max_samples=args.max_samples,
                demo=args.demo,
            )
            print(f"  Downloaded to: {path}")
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
