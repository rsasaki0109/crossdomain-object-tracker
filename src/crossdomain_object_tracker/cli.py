"""Command-line interface for crossdomain-object-tracker.

Provides subcommands for downloading datasets, running detections,
evaluating across domains, generating visualizations, and producing reports.

Usage:
    crossdomain-tracker download --dataset covla --output data/
    crossdomain-tracker detect --model yolov8n --dataset covla --data-dir data/covla
    crossdomain-tracker evaluate --model yolov8n --datasets covla polaris --data-dir data/
    crossdomain-tracker visualize --results outputs/results.json --output outputs/plots/
    crossdomain-tracker report --results outputs/results.json --output outputs/report/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="crossdomain-tracker",
        description="Cross-domain object detection and tracking evaluation tool for robotics datasets.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # download
    dl = subparsers.add_parser("download", help="Download datasets")
    dl.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (covla, polaris, mcd, hm3d-ovon) or 'all'",
    )
    dl.add_argument("--output", type=str, default="data/", help="Output directory")
    dl.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download",
    )
    dl.add_argument(
        "--demo",
        action="store_true",
        help="Download only sample images for quick testing",
    )

    # detect
    det = subparsers.add_parser("detect", help="Run detection on a dataset")
    det.add_argument("--model", type=str, default="yolov8n", help="Detection model name")
    det.add_argument("--dataset", type=str, required=True, help="Dataset name")
    det.add_argument("--data-dir", type=str, required=True, help="Data directory")
    det.add_argument("--output", type=str, default="outputs/", help="Output directory")
    det.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    det.add_argument("--max-samples", type=int, default=50, help="Max images to process")
    det.add_argument(
        "--text-prompt",
        type=str,
        default=None,
        help="Text prompt for Grounding DINO (e.g. 'car . person . tree')",
    )

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Run cross-domain evaluation")
    ev.add_argument("--model", type=str, default="yolov8n", help="Detection model name")
    ev.add_argument("--datasets", nargs="+", default=None, help="Datasets to evaluate on")
    ev.add_argument("--data-dir", type=str, default="data/", help="Data root directory")
    ev.add_argument("--output-dir", type=str, default="outputs/", help="Output directory")
    ev.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    ev.add_argument("--max-samples", type=int, default=50, help="Max images per dataset")

    # visualize
    vis = subparsers.add_parser("visualize", help="Generate visualizations")
    vis.add_argument("--results", type=str, required=True, help="Path to results JSON")
    vis.add_argument("--output", type=str, default="outputs/plots/", help="Output directory")

    # report
    rp = subparsers.add_parser("report", help="Generate evaluation report")
    rp.add_argument("--results", type=str, required=True, help="Path to results JSON")
    rp.add_argument("--output", type=str, default="outputs/report/", help="Output directory")

    # latex
    lx = subparsers.add_parser("latex", help="Generate LaTeX tables from results")
    lx.add_argument("--results", type=str, required=True, help="Path to results JSON")
    lx.add_argument("--output", type=str, default=None, help="Output .tex file (default: print to stdout)")
    lx.add_argument(
        "--type",
        type=str,
        choices=["summary", "class", "both"],
        default="both",
        help="Table type: summary, class, or both (default: both)",
    )

    return parser


def _cmd_download(args: argparse.Namespace) -> None:
    """Handle the 'download' subcommand."""
    from crossdomain_object_tracker.common.config import load_datasets_config
    from crossdomain_object_tracker.common.download import download_dataset

    if args.dataset == "all":
        config = load_datasets_config()
        dataset_names = list(config.get("datasets", {}).keys())
    else:
        dataset_names = [args.dataset]

    for name in dataset_names:
        print(f"Downloading dataset: {name}")
        try:
            path = download_dataset(
                name,
                output_dir=args.output,
                max_samples=args.max_samples,
                demo=args.demo,
            )
            print(f"  -> {path}")
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)


def _cmd_detect(args: argparse.Namespace) -> None:
    """Handle the 'detect' subcommand."""
    from crossdomain_object_tracker.detector import get_detector
    from crossdomain_object_tracker.evaluate import evaluate_dataset, save_results

    print(f"Loading detector: {args.model}")
    detector = get_detector(args.model)

    print(f"Running detection on '{args.dataset}' from {args.data_dir}")
    result = evaluate_dataset(
        detector,
        args.dataset,
        args.data_dir,
        max_samples=args.max_samples,
        conf=args.confidence,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"detect_{args.dataset}_{args.model}.json"
    save_results({args.dataset: result}, results_path)

    print(f"\nResults for {args.dataset}:")
    print(f"  Images: {result['num_images']}")
    print(f"  Detections: {result['total_detections']}")
    print(f"  Avg detections/image: {result['avg_detections_per_image']:.2f}")
    print(f"  Avg confidence: {result['avg_confidence']:.4f}")
    print(f"  Avg inference time: {result['avg_inference_time_ms']:.1f} ms")
    print(f"  Results saved to: {results_path}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Handle the 'evaluate' subcommand."""
    from crossdomain_object_tracker.common.config import load_datasets_config
    from crossdomain_object_tracker.detector import get_detector
    from crossdomain_object_tracker.evaluate import (
        compare_domains,
        compute_domain_gap,
        evaluate_dataset,
        save_results,
    )

    # Determine datasets
    if args.datasets:
        dataset_names = args.datasets
    else:
        config = load_datasets_config()
        dataset_names = list(config.get("datasets", {}).keys())

    print(f"Loading detector: {args.model}")
    detector = get_detector(args.model)

    results: dict[str, dict[str, Any]] = {}
    data_root = Path(args.data_dir)

    for name in dataset_names:
        data_dir = data_root / name
        if not data_dir.exists():
            print(f"Skipping '{name}': data directory not found at {data_dir}")
            continue

        print(f"\nEvaluating: {name}")
        result = evaluate_dataset(
            detector,
            name,
            data_dir,
            max_samples=args.max_samples,
            conf=args.confidence,
        )
        results[name] = result
        print(f"  {result['total_detections']} detections in {result['num_images']} images")

    if not results:
        print("\nNo datasets found. Download data first with: crossdomain-tracker download --dataset all --demo")
        return

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    save_results(results, results_path)

    # Print comparison
    comparison = compare_domains(results)
    print("\n" + "=" * 80)
    print("Cross-Domain Comparison")
    print("=" * 80)
    print(comparison.to_string())

    gap = compute_domain_gap(results)
    print(f"\nOverall domain gap score: {gap['overall_gap_score']}")
    print(f"\nResults saved to: {results_path}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    """Handle the 'visualize' subcommand."""
    from crossdomain_object_tracker.evaluate import load_results
    from crossdomain_object_tracker.visualize import (
        plot_class_distribution,
        plot_confidence_distribution,
        plot_detection_counts,
    )

    results = load_results(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")
    plot_class_distribution(results, output_path=output_dir / "class_distribution.png")
    print(f"  Saved: {output_dir / 'class_distribution.png'}")

    plot_confidence_distribution(results, output_path=output_dir / "confidence_distribution.png")
    print(f"  Saved: {output_dir / 'confidence_distribution.png'}")

    plot_detection_counts(results, output_path=output_dir / "detection_counts.png")
    print(f"  Saved: {output_dir / 'detection_counts.png'}")

    print("Done.")


def _cmd_report(args: argparse.Namespace) -> None:
    """Handle the 'report' subcommand."""
    from crossdomain_object_tracker.evaluate import load_results
    from crossdomain_object_tracker.report import generate_report

    results = load_results(args.results)

    print("Generating report...")
    report_path = generate_report(results, output_dir=args.output)
    print(f"Report generated: {report_path}")


def _cmd_latex(args: argparse.Namespace) -> None:
    """Handle the 'latex' subcommand."""
    from crossdomain_object_tracker.evaluate import load_results
    from crossdomain_object_tracker.report import (
        generate_latex_class_table,
        generate_latex_table,
    )

    results = load_results(args.results)

    parts: list[str] = []
    if args.type in ("summary", "both"):
        parts.append(generate_latex_table(results))
    if args.type in ("class", "both"):
        parts.append(generate_latex_class_table(results))

    latex_output = "\n\n".join(parts)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_output, encoding="utf-8")
        print(f"LaTeX tables saved to: {output_path}")
    else:
        print(latex_output)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "download": _cmd_download,
        "detect": _cmd_detect,
        "evaluate": _cmd_evaluate,
        "visualize": _cmd_visualize,
        "report": _cmd_report,
        "latex": _cmd_latex,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)

    try:
        handler(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
