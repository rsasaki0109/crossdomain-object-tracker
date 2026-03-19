#!/usr/bin/env python3
"""Run a quick demo of cross-domain object detection.

End-to-end demo that:
1. Downloads sample images from each dataset
2. Runs YOLOv8 on all datasets
3. Generates comparison report
4. Prints summary to console

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --model yolov8s --output outputs/demo/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path so this script works standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Run the demo pipeline."""
    parser = argparse.ArgumentParser(description="Run cross-domain detection demo.")
    parser.add_argument("--model", type=str, default="yolov8n", help="Model to use")
    parser.add_argument("--data-dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--output", type=str, default="outputs/demo/", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=10, help="Max images per dataset")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading sample images",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from crossdomain_object_tracker.common.config import load_datasets_config
    from crossdomain_object_tracker.common.download import download_dataset
    from crossdomain_object_tracker.detector import get_detector
    from crossdomain_object_tracker.evaluate import (
        compare_domains,
        compute_domain_gap,
        evaluate_dataset,
        save_results,
    )
    from crossdomain_object_tracker.report import generate_report

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_datasets_config()
    dataset_names = list(config.get("datasets", {}).keys())

    # Step 1: Download sample images
    if not args.skip_download:
        print("=" * 60)
        print("Step 1: Downloading sample images")
        print("=" * 60)
        for name in dataset_names:
            print(f"\n  Downloading: {name}")
            try:
                download_dataset(name, output_dir=data_dir, demo=True)
                print("    Done.")
            except Exception as exc:
                print(f"    Error: {exc}")
        print()

    # Step 2: Run detection
    print("=" * 60)
    print(f"Step 2: Running {args.model} on all datasets")
    print("=" * 60)

    detector = get_detector(args.model)
    results: dict[str, dict[str, Any]] = {}

    for name in dataset_names:
        dataset_dir = data_dir / name
        if not dataset_dir.exists():
            print(f"\n  Skipping {name}: no data found at {dataset_dir}")
            continue

        print(f"\n  Processing: {name}")
        result = evaluate_dataset(detector, name, dataset_dir, max_samples=args.max_samples)
        results[name] = result
        print(
            f"    {result['total_detections']} detections in "
            f"{result['num_images']} images "
            f"({result['avg_inference_time_ms']:.0f} ms/img)"
        )

    if not results:
        print("\nNo datasets found. Run without --skip-download to get sample images.")
        return

    # Step 3: Save results
    results_path = output_dir / "results.json"
    save_results(results, results_path)

    # Step 4: Generate report
    print()
    print("=" * 60)
    print("Step 3: Generating report")
    print("=" * 60)

    report_path = generate_report(results, output_dir=output_dir / "report")
    print(f"\n  Report: {report_path}")

    # Step 5: Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    comparison = compare_domains(results)
    print()
    print(comparison.to_string())

    gap = compute_domain_gap(results)
    print(f"\nOverall domain gap score: {gap['overall_gap_score']}")

    print()
    print(f"Results JSON: {results_path}")
    print(f"Report HTML:  {report_path}")
    print()
    print("Demo complete.")


if __name__ == "__main__":
    main()
