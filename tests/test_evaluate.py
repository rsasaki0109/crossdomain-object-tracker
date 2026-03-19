"""Tests for evaluation pipeline functions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from crossdomain_object_tracker.evaluate import (
    compare_domains,
    compute_domain_gap,
    load_results,
    save_results,
)


def _make_result(
    dataset: str,
    num_images: int = 5,
    total_detections: int = 10,
    avg_detections_per_image: float = 2.0,
    avg_confidence: float = 0.7,
    class_distribution: dict[str, int] | None = None,
    confidence_scores: list[float] | None = None,
    avg_inference_time_ms: float = 15.0,
) -> dict[str, Any]:
    """Create a mock evaluation result dict."""
    if class_distribution is None:
        class_distribution = {"car": 6, "person": 4}
    if confidence_scores is None:
        confidence_scores = [0.7] * total_detections
    return {
        "dataset": dataset,
        "num_images": num_images,
        "total_detections": total_detections,
        "avg_detections_per_image": avg_detections_per_image,
        "avg_confidence": avg_confidence,
        "class_distribution": class_distribution,
        "confidence_scores": confidence_scores,
        "inference_times_ms": [avg_inference_time_ms] * num_images,
        "avg_inference_time_ms": avg_inference_time_ms,
        "per_image_results": [],
    }


class TestCompareDomains:
    """Tests for compare_domains."""

    def test_creates_dataframe(self) -> None:
        """compare_domains returns a DataFrame with expected columns and values."""
        results = {
            "domain_a": _make_result("domain_a", num_images=3, total_detections=9, avg_detections_per_image=3.0),
            "domain_b": _make_result("domain_b", num_images=5, total_detections=20, avg_detections_per_image=4.0),
        }
        df = compare_domains(results)

        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "dataset"
        assert set(df.index) == {"domain_a", "domain_b"}
        expected_cols = {
            "num_images",
            "total_detections",
            "avg_detections_per_image",
            "avg_confidence",
            "unique_classes",
            "avg_inference_time_ms",
            "top_classes",
        }
        assert expected_cols.issubset(set(df.columns))
        assert df.loc["domain_a", "num_images"] == 3
        assert df.loc["domain_b", "total_detections"] == 20

    def test_single_domain(self) -> None:
        """compare_domains works with a single domain."""
        results = {"only": _make_result("only")}
        df = compare_domains(results)
        assert len(df) == 1

    def test_empty_results(self) -> None:
        """compare_domains with empty dict returns an empty DataFrame."""
        df = compare_domains({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestComputeDomainGap:
    """Tests for compute_domain_gap."""

    def test_returns_pairwise_metrics(self) -> None:
        """compute_domain_gap returns pairwise overlap, confidence diff, and rate diff."""
        results = {
            "a": _make_result("a", avg_confidence=0.8, class_distribution={"car": 5, "person": 3}),
            "b": _make_result("b", avg_confidence=0.5, class_distribution={"person": 4, "dog": 2}),
        }
        gap = compute_domain_gap(results)

        assert "pairwise_class_overlap" in gap
        assert "pairwise_confidence_diff" in gap
        assert "pairwise_detection_rate_diff" in gap
        assert "overall_gap_score" in gap

        pair_key = "a_vs_b"
        assert pair_key in gap["pairwise_class_overlap"]
        # Jaccard: intersection={person}=1, union={car,person,dog}=3 => 1/3
        assert abs(gap["pairwise_class_overlap"][pair_key] - round(1 / 3, 4)) < 0.001
        # Confidence diff = |0.8 - 0.5| = 0.3
        assert abs(gap["pairwise_confidence_diff"][pair_key] - 0.3) < 0.001

    def test_single_domain(self) -> None:
        """compute_domain_gap with one domain returns empty pairwise dicts."""
        results = {"only": _make_result("only")}
        gap = compute_domain_gap(results)
        assert gap["pairwise_class_overlap"] == {}
        assert gap["pairwise_confidence_diff"] == {}
        assert gap["pairwise_detection_rate_diff"] == {}

    def test_empty_results(self) -> None:
        """compute_domain_gap with empty results does not crash."""
        gap = compute_domain_gap({})
        assert isinstance(gap, dict)
        assert "overall_gap_score" in gap


class TestSaveAndLoadResults:
    """Tests for save_results and load_results roundtrip."""

    def test_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Saving and loading results preserves the data."""
        results = {
            "ds1": _make_result("ds1"),
            "ds2": _make_result("ds2", class_distribution={"tree": 7}),
        }
        out_file = tmp_path / "results.json"  # type: ignore[operator]
        save_results(results, out_file)

        assert out_file.exists()  # type: ignore[union-attr]
        loaded = load_results(out_file)

        assert set(loaded.keys()) == {"ds1", "ds2"}
        assert loaded["ds1"]["num_images"] == results["ds1"]["num_images"]
        assert loaded["ds2"]["class_distribution"] == {"tree": 7}

    def test_save_creates_parent_dirs(self, tmp_path: pytest.TempPathFactory) -> None:
        """save_results creates parent directories if they don't exist."""
        results = {"x": _make_result("x")}
        out_file = tmp_path / "sub" / "dir" / "results.json"  # type: ignore[operator]
        save_results(results, out_file)
        assert out_file.exists()  # type: ignore[union-attr]

    def test_save_strips_per_image_results(self, tmp_path: pytest.TempPathFactory) -> None:
        """save_results removes per_image_results to keep the file smaller."""
        results = {
            "ds": _make_result("ds"),
        }
        results["ds"]["per_image_results"] = [{"image_path": "a.jpg", "num_detections": 1}]
        out_file = tmp_path / "results.json"  # type: ignore[operator]
        save_results(results, out_file)
        loaded = load_results(out_file)
        assert "per_image_results" not in loaded["ds"]
