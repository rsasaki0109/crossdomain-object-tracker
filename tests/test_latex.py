"""Tests for LaTeX table generation."""

from __future__ import annotations

from pathlib import Path

from crossdomain_object_tracker.report import (
    generate_latex_class_table,
    generate_latex_table,
)

# Sample data mimicking evaluate_dataset output
SAMPLE_RESULTS: dict = {
    "polaris": {
        "dataset": "polaris",
        "num_images": 3,
        "total_detections": 1,
        "avg_detections_per_image": 0.33,
        "avg_confidence": 0.843,
        "class_distribution": {"person": 1},
        "confidence_scores": [0.843],
        "inference_times_ms": [10.0, 12.0, 11.0],
        "avg_inference_time_ms": 11.0,
        "per_image_results": [],
    },
    "covla": {
        "dataset": "covla",
        "num_images": 3,
        "total_detections": 22,
        "avg_detections_per_image": 7.33,
        "avg_confidence": 0.483,
        "class_distribution": {"car": 10, "person": 8, "truck": 4},
        "confidence_scores": [0.5] * 22,
        "inference_times_ms": [15.0, 14.0, 16.0],
        "avg_inference_time_ms": 15.0,
        "per_image_results": [],
    },
    "mcd": {
        "dataset": "mcd",
        "num_images": 3,
        "total_detections": 17,
        "avg_detections_per_image": 5.67,
        "avg_confidence": 0.576,
        "class_distribution": {"person": 8, "bicycle": 5, "bench": 4},
        "confidence_scores": [0.6] * 17,
        "inference_times_ms": [13.0, 12.0, 14.0],
        "avg_inference_time_ms": 13.0,
        "per_image_results": [],
    },
}


class TestGenerateLatexTable:
    """Tests for generate_latex_table."""

    def test_contains_booktabs_commands(self) -> None:
        latex = generate_latex_table(SAMPLE_RESULTS)
        assert "\\toprule" in latex
        assert "\\midrule" in latex
        assert "\\bottomrule" in latex

    def test_contains_table_environment(self) -> None:
        latex = generate_latex_table(SAMPLE_RESULTS)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_contains_dataset_names(self) -> None:
        latex = generate_latex_table(SAMPLE_RESULTS)
        assert "polaris" in latex
        assert "covla" in latex
        assert "mcd" in latex

    def test_contains_total_row(self) -> None:
        latex = generate_latex_table(SAMPLE_RESULTS)
        assert "Total/Avg" in latex
        # Total images = 9
        assert "\\textbf{9}" in latex
        # Total detections = 40
        assert "\\textbf{40}" in latex

    def test_contains_header_columns(self) -> None:
        latex = generate_latex_table(SAMPLE_RESULTS)
        assert "Domain" in latex
        assert "Images" in latex
        assert "Detections" in latex
        assert "Avg Conf." in latex

    def test_saves_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "table.tex"
        result = generate_latex_table(SAMPLE_RESULTS, output_path=out)
        assert out.exists()
        assert out.read_text() == result

    def test_empty_results(self) -> None:
        latex = generate_latex_table({})
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex

    def test_escapes_special_characters(self) -> None:
        results = {
            "test_domain": {
                "num_images": 1,
                "total_detections": 1,
                "avg_detections_per_image": 1.0,
                "avg_confidence": 0.9,
                "class_distribution": {"fire_hydrant": 1},
            },
        }
        latex = generate_latex_table(results)
        assert "test\\_domain" in latex
        assert "fire\\_hydrant" in latex


class TestGenerateLatexClassTable:
    """Tests for generate_latex_class_table."""

    def test_contains_booktabs_commands(self) -> None:
        latex = generate_latex_class_table(SAMPLE_RESULTS)
        assert "\\toprule" in latex
        assert "\\midrule" in latex
        assert "\\bottomrule" in latex

    def test_contains_class_names(self) -> None:
        latex = generate_latex_class_table(SAMPLE_RESULTS)
        assert "person" in latex
        assert "car" in latex
        assert "truck" in latex
        assert "bicycle" in latex
        assert "bench" in latex

    def test_classes_sorted_by_total_descending(self) -> None:
        latex = generate_latex_class_table(SAMPLE_RESULTS)
        lines = latex.split("\n")
        data_lines = [
            line
            for line in lines
            if "\\\\" in line and "toprule" not in line and "midrule" not in line and "Class" not in line
        ]
        # person total=9 should appear before car total=10? No: person=1+8+8=17, car=10
        # person(17) first, then car(10)
        assert data_lines[0].startswith("person")
        assert data_lines[1].startswith("car")

    def test_contains_domain_headers(self) -> None:
        latex = generate_latex_class_table(SAMPLE_RESULTS)
        assert "polaris" in latex
        assert "covla" in latex
        assert "mcd" in latex
        assert "Total" in latex

    def test_saves_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "class_table.tex"
        result = generate_latex_class_table(SAMPLE_RESULTS, output_path=out)
        assert out.exists()
        assert out.read_text() == result

    def test_zero_counts_shown(self) -> None:
        latex = generate_latex_class_table(SAMPLE_RESULTS)
        # car only in covla, so polaris and mcd columns should have 0
        car_line = [line for line in latex.split("\n") if line.startswith("car")][0]
        assert "0" in car_line

    def test_empty_results(self) -> None:
        latex = generate_latex_class_table({})
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex
