"""Tests for visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from crossdomain_object_tracker.detector import Detection
from crossdomain_object_tracker.visualize import (
    draw_detections,
    plot_class_distribution,
    plot_confidence_distribution,
)


@pytest.fixture()
def small_image(tmp_path: Path) -> Path:
    """Create a small test image and return its path."""
    img = np.full((100, 150, 3), fill_value=128, dtype=np.uint8)
    img_path = tmp_path / "test_img.png"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture()
def fake_detections() -> list[Detection]:
    """Return a list of fake Detection objects."""
    return [
        Detection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.9, class_id=0, class_name="car"),
        Detection(bbox=(60.0, 20.0, 120.0, 80.0), confidence=0.75, class_id=1, class_name="person"),
    ]


@pytest.fixture()
def sample_results() -> dict[str, dict[str, Any]]:
    """Create mock evaluation results for plot functions."""
    return {
        "domain_a": {
            "class_distribution": {"car": 10, "person": 5},
            "confidence_scores": [0.8, 0.7, 0.9, 0.6, 0.85],
            "avg_detections_per_image": 3.0,
            "avg_confidence": 0.77,
        },
        "domain_b": {
            "class_distribution": {"person": 8, "dog": 3},
            "confidence_scores": [0.5, 0.6, 0.55, 0.7],
            "avg_detections_per_image": 2.0,
            "avg_confidence": 0.59,
        },
    }


class TestDrawDetections:
    """Tests for draw_detections."""

    def test_returns_image(self, small_image: Path, fake_detections: list[Detection]) -> None:
        """draw_detections returns a numpy array with the same shape as the input."""
        result = draw_detections(small_image, fake_detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 150, 3)

    def test_no_detections(self, small_image: Path) -> None:
        """Empty detection list returns the original image unchanged."""
        original = cv2.imread(str(small_image))
        result = draw_detections(small_image, [])
        assert isinstance(result, np.ndarray)
        assert result.shape == original.shape
        np.testing.assert_array_equal(result, original)

    def test_saves_to_file(self, small_image: Path, fake_detections: list[Detection], tmp_path: Path) -> None:
        """draw_detections saves to output_path when provided."""
        out = tmp_path / "annotated.png"
        draw_detections(small_image, fake_detections, output_path=out)
        assert out.exists()

    def test_nonexistent_image(self, tmp_path: Path) -> None:
        """draw_detections raises FileNotFoundError for a missing image."""
        with pytest.raises(FileNotFoundError):
            draw_detections(tmp_path / "missing.png", [])


class TestPlotFunctions:
    """Tests for plot_class_distribution and plot_confidence_distribution."""

    def test_plot_class_distribution_saves(self, sample_results: dict, tmp_path: Path) -> None:
        """plot_class_distribution creates an output file."""
        out = tmp_path / "class_dist.png"
        plot_class_distribution(sample_results, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_confidence_distribution_saves(self, sample_results: dict, tmp_path: Path) -> None:
        """plot_confidence_distribution creates an output file."""
        out = tmp_path / "conf_dist.png"
        plot_confidence_distribution(sample_results, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_class_distribution_empty(self, tmp_path: Path) -> None:
        """plot_class_distribution with empty results does not crash."""
        out = tmp_path / "empty_class.png"
        plot_class_distribution({}, output_path=out)
        # File should not be created since there's nothing to plot
        # (function returns early when no classes)

    def test_plot_confidence_distribution_empty(self, tmp_path: Path) -> None:
        """plot_confidence_distribution with empty results does not crash."""
        out = tmp_path / "empty_conf.png"
        plot_confidence_distribution({}, output_path=out)
