"""Tests for COCO-style evaluation metrics."""

from __future__ import annotations

import json

import pytest

from crossdomain_object_tracker.detector import Detection
from crossdomain_object_tracker.metrics.coco_eval import (
    COCOResults,
    compute_ap,
    compute_iou,
    evaluate_coco,
    load_coco_annotations,
)


def _det(bbox: tuple, confidence: float, class_name: str, class_id: int = 0) -> Detection:
    """Helper to create a Detection."""
    return Detection(bbox=bbox, confidence=confidence, class_id=class_id, class_name=class_name)


class TestComputeIoU:
    """Tests for compute_iou."""

    def test_perfect_overlap(self) -> None:
        """Identical boxes have IoU = 1.0."""
        box = (10, 20, 50, 60)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes have IoU = 0.0."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes have 0 < IoU < 1."""
        box1 = (0, 0, 10, 10)  # area = 100
        box2 = (5, 5, 15, 15)  # area = 100
        # intersection: (5,5)-(10,10) = 25
        # union: 100 + 100 - 25 = 175
        assert compute_iou(box1, box2) == pytest.approx(25.0 / 175.0)

    def test_one_inside_other(self) -> None:
        """Box fully inside another has IoU = small_area / large_area."""
        box1 = (0, 0, 20, 20)  # area = 400
        box2 = (5, 5, 10, 10)  # area = 25
        # intersection = 25, union = 400 + 25 - 25 = 400
        assert compute_iou(box1, box2) == pytest.approx(25.0 / 400.0)

    def test_zero_area_box(self) -> None:
        """Zero-area box returns IoU = 0."""
        box1 = (5, 5, 5, 5)  # zero area
        box2 = (0, 0, 10, 10)
        assert compute_iou(box1, box2) == pytest.approx(0.0)


class TestComputeAP:
    """Tests for compute_ap."""

    def test_perfect_precision_recall(self) -> None:
        """Perfect detector: precision=1 at all recall levels -> AP=1."""
        precisions = [1.0, 1.0, 1.0, 1.0]
        recalls = [0.25, 0.5, 0.75, 1.0]
        ap = compute_ap(precisions, recalls)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_zero_precision(self) -> None:
        """No correct detections -> AP near 0."""
        precisions = [0.0, 0.0, 0.0]
        recalls = [0.0, 0.0, 0.0]
        ap = compute_ap(precisions, recalls)
        assert ap == pytest.approx(0.0, abs=0.01)

    def test_decreasing_precision(self) -> None:
        """Decreasing precision with increasing recall gives intermediate AP."""
        precisions = [1.0, 0.5, 0.33]
        recalls = [0.33, 0.66, 1.0]
        ap = compute_ap(precisions, recalls)
        assert 0.0 < ap < 1.0


class TestEvaluateCoco:
    """Tests for evaluate_coco with synthetic data."""

    def test_perfect_detection(self) -> None:
        """Predictions matching GT exactly should give high mAP."""
        gt1 = [_det((10, 10, 50, 50), 1.0, "car")]
        gt2 = [_det((20, 20, 60, 60), 1.0, "car")]
        gt3 = [_det((5, 5, 30, 30), 1.0, "person")]

        pred1 = [_det((10, 10, 50, 50), 0.9, "car")]
        pred2 = [_det((20, 20, 60, 60), 0.8, "car")]
        pred3 = [_det((5, 5, 30, 30), 0.95, "person")]

        results = evaluate_coco([pred1, pred2, pred3], [gt1, gt2, gt3])

        assert results.mAP_50 == pytest.approx(1.0, abs=0.02)
        assert results.mAP_75 == pytest.approx(1.0, abs=0.02)
        assert results.num_gt == 3
        assert results.num_pred == 3
        assert "car" in results.per_class_ap
        assert "person" in results.per_class_ap

    def test_no_predictions(self) -> None:
        """No predictions should give mAP = 0."""
        gt1 = [_det((10, 10, 50, 50), 1.0, "car")]
        gt2 = [_det((20, 20, 60, 60), 1.0, "car")]

        results = evaluate_coco([[], []], [gt1, gt2])

        assert results.mAP == pytest.approx(0.0)
        assert results.mAP_50 == pytest.approx(0.0)
        assert results.num_gt == 2
        assert results.num_pred == 0

    def test_no_ground_truths(self) -> None:
        """No ground truths should give mAP = 0."""
        pred1 = [_det((10, 10, 50, 50), 0.9, "car")]

        results = evaluate_coco([pred1], [[]])

        assert results.mAP == pytest.approx(0.0)
        assert results.num_gt == 0
        assert results.num_pred == 1

    def test_empty_inputs(self) -> None:
        """Both empty should give mAP = 0."""
        results = evaluate_coco([], [])

        assert results.mAP == pytest.approx(0.0)
        assert results.num_gt == 0
        assert results.num_pred == 0

    def test_wrong_class_predictions(self) -> None:
        """Predictions with wrong class should not match GT."""
        gt = [_det((10, 10, 50, 50), 1.0, "car")]
        pred = [_det((10, 10, 50, 50), 0.9, "person")]

        results = evaluate_coco([pred], [gt])

        assert results.mAP_50 == pytest.approx(0.0)

    def test_low_iou_predictions(self) -> None:
        """Predictions with low IoU should fail at high thresholds."""
        gt = [_det((0, 0, 100, 100), 1.0, "car")]
        # Overlapping but not much
        pred = [_det((80, 80, 180, 180), 0.9, "car")]

        results = evaluate_coco([pred], [gt])

        # IoU is small: intersection=(80,80)-(100,100)=400,
        # union=10000+10000-400=19600, IoU~0.02 -> should fail even at 0.5
        assert results.mAP_50 == pytest.approx(0.0)

    def test_multiple_classes(self) -> None:
        """Per-class AP should be computed independently."""
        gt = [
            _det((0, 0, 10, 10), 1.0, "car"),
            _det((50, 50, 60, 60), 1.0, "person"),
        ]
        pred = [
            _det((0, 0, 10, 10), 0.9, "car"),  # correct
            _det((90, 90, 100, 100), 0.8, "person"),  # wrong location
        ]

        results = evaluate_coco([pred], [gt])

        assert results.per_class_ap.get("car", 0) > 0.5
        assert results.per_class_ap.get("person", 1) < 0.5


class TestCOCOResults:
    """Tests for COCOResults dataclass methods."""

    def _make_results(self) -> COCOResults:
        return COCOResults(
            mAP=0.4567,
            mAP_50=0.6789,
            mAP_75=0.3456,
            per_class_ap={"car": 0.8, "person": 0.55},
            per_class_ap_75={"car": 0.5, "person": 0.2},
            num_gt=100,
            num_pred=120,
        )

    def test_to_dict(self) -> None:
        """to_dict returns expected keys and rounded values."""
        d = self._make_results().to_dict()

        assert d["mAP"] == 0.4567
        assert d["mAP@50"] == 0.6789
        assert d["mAP@75"] == 0.3456
        assert d["num_gt"] == 100
        assert d["num_pred"] == 120
        assert "car" in d["per_class_AP@50"]
        assert "person" in d["per_class_AP@50"]

    def test_to_latex(self) -> None:
        """to_latex returns a valid LaTeX table string."""
        latex = self._make_results().to_latex()

        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert "mAP@[.5:.95]" in latex
        assert "0.4567" in latex
        assert "car" in latex
        assert "person" in latex

    def test_to_latex_escapes_underscores(self) -> None:
        """to_latex escapes underscores in class names."""
        results = COCOResults(
            mAP=0.5,
            mAP_50=0.7,
            mAP_75=0.3,
            per_class_ap={"fire_hydrant": 0.6},
            per_class_ap_75={"fire_hydrant": 0.4},
            num_gt=10,
            num_pred=10,
        )
        latex = results.to_latex()
        assert r"fire\_hydrant" in latex


class TestLoadCocoAnnotations:
    """Tests for load_coco_annotations."""

    def test_load_synthetic_coco(self, tmp_path: pytest.TempPathFactory) -> None:
        """Load a minimal synthetic COCO annotation file."""
        coco_data = {
            "images": [
                {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img_002.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 200]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 30, 40]},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [0, 0, 50, 50]},
            ],
            "categories": [
                {"id": 1, "name": "car"},
                {"id": 2, "name": "person"},
            ],
        }

        json_path = tmp_path / "annotations.json"  # type: ignore[operator]
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        annotations = load_coco_annotations(str(json_path))

        assert "img_001.jpg" in annotations
        assert "img_002.jpg" in annotations
        assert len(annotations["img_001.jpg"]) == 2
        assert len(annotations["img_002.jpg"]) == 1

        # Check first annotation: bbox should be converted from (x,y,w,h) to (x1,y1,x2,y2)
        det = annotations["img_001.jpg"][0]
        assert det.bbox == (10, 20, 110, 220)  # x, y, x+w, y+h
        assert det.class_name == "car"
        assert det.confidence == 1.0

        det2 = annotations["img_001.jpg"][1]
        assert det2.bbox == (50, 50, 80, 90)
        assert det2.class_name == "person"

    def test_empty_annotations(self, tmp_path: pytest.TempPathFactory) -> None:
        """Load COCO file with no annotations."""
        coco_data = {
            "images": [{"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480}],
            "annotations": [],
            "categories": [{"id": 1, "name": "car"}],
        }

        json_path = tmp_path / "empty_ann.json"  # type: ignore[operator]
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        annotations = load_coco_annotations(str(json_path))

        assert "img_001.jpg" in annotations
        assert len(annotations["img_001.jpg"]) == 0
