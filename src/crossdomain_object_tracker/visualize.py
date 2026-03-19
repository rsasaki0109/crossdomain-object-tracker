"""Visualization utilities for cross-domain evaluation results.

Generates comparative plots showing detection performance across domains:
- Per-domain detection count bar charts
- Confidence distribution histograms
- Class distribution charts
- Detection overlay images with bounding boxes

Usage:
    from crossdomain_object_tracker.visualize import draw_detections, plot_class_distribution
    annotated = draw_detections("image.jpg", detections)
    plot_class_distribution(results, output_path="outputs/classes.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from crossdomain_object_tracker.detector import Detection

# Use non-interactive backend by default so plots can be saved without display
matplotlib.use("Agg")

# Consistent color palette for up to 20 classes
_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
]

# BGR colors for OpenCV drawing
_COLORS_BGR = [(b, g, r) for r, g, b in _COLORS]


def _get_color(index: int) -> tuple[int, int, int]:
    """Get a BGR color for a class index."""
    return _COLORS_BGR[index % len(_COLORS_BGR)]


def _get_color_rgb_norm(index: int) -> tuple[float, float, float]:
    """Get an RGB normalized color for matplotlib."""
    r, g, b = _COLORS[index % len(_COLORS)]
    return (r / 255.0, g / 255.0, b / 255.0)


def draw_detections(
    image_path: str | Path,
    detections: list[Detection],
    output_path: str | Path | None = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes on image with class labels and confidence.

    Args:
        image_path: Path to the source image.
        detections: List of Detection objects.
        output_path: Path to save the annotated image. If None, not saved.
        line_thickness: Bounding box line thickness.
        font_scale: Font scale for labels.

    Returns:
        Annotated image as BGR numpy array.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = _get_color(det.class_id)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label background
        label = f"{det.class_name} {det.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(
            image,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w, y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    return image


def plot_class_distribution(
    results: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
    top_n: int = 15,
) -> None:
    """Bar chart comparing class distributions across domains.

    Args:
        results: Dictionary mapping dataset names to evaluation results.
        output_path: Path to save the plot. If None, calls plt.show().
        top_n: Number of top classes to display.
    """
    # Collect all classes and their counts per dataset
    all_classes: set[str] = set()
    for res in results.values():
        all_classes.update(res.get("class_distribution", {}).keys())

    if not all_classes:
        return

    # Get top N classes by total count across all datasets
    total_counts: dict[str, int] = {}
    for cls in all_classes:
        total_counts[cls] = sum(res.get("class_distribution", {}).get(cls, 0) for res in results.values())
    top_classes = sorted(total_counts, key=total_counts.get, reverse=True)[:top_n]  # type: ignore[arg-type]

    dataset_names = list(results.keys())
    x = np.arange(len(top_classes))
    width = 0.8 / max(len(dataset_names), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(dataset_names):
        counts = [results[name].get("class_distribution", {}).get(cls, 0) for cls in top_classes]
        offset = (i - len(dataset_names) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            counts,
            width,
            label=name,
            color=_get_color_rgb_norm(i),
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Detection Count")
    ax.set_title("Class Distribution Across Domains")
    ax.set_xticks(x)
    ax.set_xticklabels(top_classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confidence_distribution(
    results: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> None:
    """Histogram of confidence scores per domain.

    Args:
        results: Dictionary mapping dataset names to evaluation results.
        output_path: Path to save the plot. If None, calls plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, res) in enumerate(results.items()):
        scores = res.get("confidence_scores", [])
        if not scores:
            continue
        ax.hist(
            scores,
            bins=30,
            alpha=0.5,
            label=f"{name} (n={len(scores)})",
            color=_get_color_rgb_norm(i),
            density=True,
        )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Score Distribution Across Domains")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_detection_counts(
    results: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> None:
    """Compare detection counts across domains.

    Creates a grouped bar chart showing total detections, average
    detections per image, and unique class counts.

    Args:
        results: Dictionary mapping dataset names to evaluation results.
        output_path: Path to save the plot. If None, calls plt.show().
    """
    dataset_names = list(results.keys())
    if not dataset_names:
        return

    metrics = {
        "Avg Detections/Image": [results[n].get("avg_detections_per_image", 0) for n in dataset_names],
        "Avg Confidence": [results[n].get("avg_confidence", 0) for n in dataset_names],
        "Unique Classes": [len(results[n].get("class_distribution", {})) for n in dataset_names],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        colors = [_get_color_rgb_norm(i) for i in range(len(dataset_names))]
        bars = ax.bar(dataset_names, values, color=colors)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("Detection Statistics Across Domains", fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def create_detection_grid(
    image_paths: list[str | Path],
    all_detections: list[list[Detection]],
    output_path: str | Path | None = None,
    grid_cols: int = 3,
    cell_size: tuple[int, int] = (400, 300),
) -> np.ndarray:
    """Create a grid of images with detections from different domains.

    Args:
        image_paths: List of image file paths.
        all_detections: List of detection lists, one per image.
        output_path: Path to save the grid image.
        grid_cols: Number of columns in the grid.
        cell_size: (width, height) of each cell in the grid.

    Returns:
        Grid image as BGR numpy array.
    """
    n = len(image_paths)
    if n == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    grid_rows = (n + grid_cols - 1) // grid_cols
    cell_w, cell_h = cell_size
    grid = np.ones((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8) * 240

    for idx, (img_path, dets) in enumerate(zip(image_paths, all_detections)):
        row = idx // grid_cols
        col = idx % grid_cols

        # Draw detections on image
        try:
            annotated = draw_detections(img_path, dets)
            resized = cv2.resize(annotated, (cell_w, cell_h))
        except Exception:
            resized = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 200
            cv2.putText(
                resized,
                "Load Error",
                (10, cell_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        y0 = row * cell_h
        x0 = col * cell_w
        grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = resized

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), grid)

    return grid
