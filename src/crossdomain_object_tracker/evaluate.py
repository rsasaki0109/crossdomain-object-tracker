"""Cross-domain evaluation pipeline.

Runs a detection model across multiple datasets and computes evaluation
statistics per domain. Produces structured results for downstream
visualization and reporting.

Usage:
    from crossdomain_object_tracker.evaluate import evaluate_dataset, compare_domains
    results = {}
    results["covla"] = evaluate_dataset(detector, "covla", "data/covla")
    results["polaris"] = evaluate_dataset(detector, "polaris", "data/polaris")
    comparison = compare_domains(results)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crossdomain_object_tracker.detector import BaseDetector, Detection

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def find_images(data_dir: Path, max_samples: int | None = None) -> list[Path]:
    """Find image files in a directory (recursively).

    Args:
        data_dir: Directory to search for images.
        max_samples: Maximum number of images to return.

    Returns:
        Sorted list of image file paths.
    """
    images: list[Path] = []
    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return images

    for ext in IMAGE_EXTENSIONS:
        images.extend(data_dir.rglob(f"*{ext}"))
        images.extend(data_dir.rglob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    images = sorted(set(images))

    if max_samples is not None:
        images = images[:max_samples]

    return images


def evaluate_dataset(
    detector: BaseDetector,
    dataset_name: str,
    data_dir: str | Path,
    max_samples: int = 50,
    conf: float = 0.25,
) -> dict[str, Any]:
    """Run detector on a dataset and collect results.

    Args:
        detector: Detector instance implementing BaseDetector.
        dataset_name: Name of the dataset (for labeling).
        data_dir: Directory containing dataset images.
        max_samples: Maximum number of images to process.
        conf: Detection confidence threshold.

    Returns:
        Dictionary with evaluation statistics:
            - dataset: Dataset name
            - num_images: Number of images processed
            - total_detections: Total number of detections
            - avg_detections_per_image: Average detections per image
            - avg_confidence: Mean confidence score
            - class_distribution: Counter of class names
            - confidence_scores: List of all confidence scores
            - inference_times_ms: List of per-image inference times
            - avg_inference_time_ms: Mean inference time
            - per_image_results: List of per-image detection details
    """
    data_dir = Path(data_dir)
    images = find_images(data_dir, max_samples=max_samples)

    if not images:
        logger.warning("No images found in %s for dataset '%s'", data_dir, dataset_name)
        return {
            "dataset": dataset_name,
            "num_images": 0,
            "total_detections": 0,
            "avg_detections_per_image": 0.0,
            "avg_confidence": 0.0,
            "class_distribution": {},
            "confidence_scores": [],
            "inference_times_ms": [],
            "avg_inference_time_ms": 0.0,
            "per_image_results": [],
        }

    logger.info("Evaluating '%s' on %d images from %s", dataset_name, len(images), data_dir)

    all_detections: list[Detection] = []
    inference_times: list[float] = []
    per_image_results: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()

    for img_path in images:
        t0 = time.perf_counter()
        try:
            dets = detector.detect(img_path, conf=conf)
        except Exception as exc:
            logger.warning("Detection failed on %s: %s", img_path, exc)
            dets = []
        elapsed_ms = (time.perf_counter() - t0) * 1000

        inference_times.append(elapsed_ms)
        all_detections.extend(dets)

        for d in dets:
            class_counter[d.class_name] += 1

        per_image_results.append(
            {
                "image_path": str(img_path),
                "num_detections": len(dets),
                "inference_time_ms": elapsed_ms,
                "detections": [d.to_dict() for d in dets],
            }
        )

    confidence_scores = [d.confidence for d in all_detections]

    return {
        "dataset": dataset_name,
        "num_images": len(images),
        "total_detections": len(all_detections),
        "avg_detections_per_image": len(all_detections) / len(images) if images else 0.0,
        "avg_confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
        "class_distribution": dict(class_counter.most_common()),
        "confidence_scores": confidence_scores,
        "inference_times_ms": inference_times,
        "avg_inference_time_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "per_image_results": per_image_results,
    }


def compare_domains(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Compare detection results across domains.

    Args:
        results: Dictionary mapping dataset names to evaluation results
            (output of evaluate_dataset).

    Returns:
        DataFrame with one row per dataset and comparison metrics.
    """
    rows: list[dict[str, Any]] = []

    for name, res in results.items():
        rows.append(
            {
                "dataset": name,
                "num_images": res["num_images"],
                "total_detections": res["total_detections"],
                "avg_detections_per_image": round(res["avg_detections_per_image"], 2),
                "avg_confidence": round(res["avg_confidence"], 4),
                "unique_classes": len(res["class_distribution"]),
                "avg_inference_time_ms": round(res["avg_inference_time_ms"], 2),
                "top_classes": ", ".join(list(res["class_distribution"].keys())[:5]),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("dataset")
    return df


def compute_domain_gap(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Quantify domain gap between datasets.

    Compares class distributions and confidence distributions across
    dataset pairs to measure how different the domains are from a
    detection perspective.

    Args:
        results: Dictionary mapping dataset names to evaluation results.

    Returns:
        Dictionary with domain gap metrics:
            - pairwise_class_overlap: Jaccard similarity of detected classes
            - pairwise_confidence_diff: Mean confidence difference
            - pairwise_detection_rate_diff: Detection rate difference
            - overall_gap_score: Aggregate gap score
    """
    dataset_names = list(results.keys())
    n = len(dataset_names)

    pairwise_class_overlap: dict[str, float] = {}
    pairwise_confidence_diff: dict[str, float] = {}
    pairwise_detection_rate_diff: dict[str, float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            name_a = dataset_names[i]
            name_b = dataset_names[j]
            pair_key = f"{name_a}_vs_{name_b}"

            res_a = results[name_a]
            res_b = results[name_b]

            # Class overlap (Jaccard similarity)
            classes_a = set(res_a["class_distribution"].keys())
            classes_b = set(res_b["class_distribution"].keys())
            if classes_a or classes_b:
                intersection = len(classes_a & classes_b)
                union = len(classes_a | classes_b)
                jaccard = intersection / union if union > 0 else 0.0
            else:
                jaccard = 0.0
            pairwise_class_overlap[pair_key] = round(jaccard, 4)

            # Confidence difference
            conf_diff = abs(res_a["avg_confidence"] - res_b["avg_confidence"])
            pairwise_confidence_diff[pair_key] = round(conf_diff, 4)

            # Detection rate difference
            rate_diff = abs(res_a["avg_detections_per_image"] - res_b["avg_detections_per_image"])
            pairwise_detection_rate_diff[pair_key] = round(rate_diff, 4)

    # Compute overall gap score (average of normalized metrics)
    overlap_vals = list(pairwise_class_overlap.values())
    conf_vals = list(pairwise_confidence_diff.values())
    rate_vals = list(pairwise_detection_rate_diff.values())

    avg_overlap = float(np.mean(overlap_vals)) if overlap_vals else 0.0
    avg_conf_diff = float(np.mean(conf_vals)) if conf_vals else 0.0
    avg_rate_diff = float(np.mean(rate_vals)) if rate_vals else 0.0

    # Gap score: lower overlap and higher differences = larger gap
    overall_gap = (1.0 - avg_overlap) * 0.4 + min(avg_conf_diff, 1.0) * 0.3 + min(avg_rate_diff / 10.0, 1.0) * 0.3

    return {
        "pairwise_class_overlap": pairwise_class_overlap,
        "pairwise_confidence_diff": pairwise_confidence_diff,
        "pairwise_detection_rate_diff": pairwise_detection_rate_diff,
        "overall_gap_score": round(overall_gap, 4),
    }


def save_results(results: dict[str, dict[str, Any]], output_path: str | Path) -> None:
    """Save evaluation results to JSON.

    Args:
        results: Dictionary mapping dataset names to evaluation results.
        output_path: Path to save the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make results JSON-serializable
    serializable = {}
    for name, res in results.items():
        s = dict(res)
        # Remove per-image results from the saved file to keep it smaller
        # (they can be large); keep the summary stats
        s.pop("per_image_results", None)
        serializable[name] = s

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info("Results saved to %s", output_path)


def load_results(input_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load evaluation results from JSON.

    Args:
        input_path: Path to the results JSON file.

    Returns:
        Dictionary mapping dataset names to evaluation results.
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        return json.load(f)
