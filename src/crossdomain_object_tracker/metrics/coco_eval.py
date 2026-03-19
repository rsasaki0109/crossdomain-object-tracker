"""COCO-style evaluation metrics (pure Python, no pycocotools dependency).

Computes mAP@[0.5:0.95], mAP@0.5, mAP@0.75 and per-class AP using the
101-point interpolation method defined by the COCO evaluation protocol.

Usage:
    from crossdomain_object_tracker.metrics.coco_eval import evaluate_coco
    results = evaluate_coco(predictions, ground_truths)
    print(results.to_dict())
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from crossdomain_object_tracker.detector import Detection


@dataclass
class COCOResults:
    """COCO-style evaluation results."""

    mAP: float  # mAP@[0.5:0.95]
    mAP_50: float  # mAP@0.5
    mAP_75: float  # mAP@0.75
    per_class_ap: dict[str, float]  # per-class AP@0.5
    per_class_ap_75: dict[str, float]
    num_gt: int
    num_pred: int

    def to_dict(self) -> dict:
        return {
            "mAP": round(self.mAP, 4),
            "mAP@50": round(self.mAP_50, 4),
            "mAP@75": round(self.mAP_75, 4),
            "per_class_AP@50": {k: round(v, 4) for k, v in self.per_class_ap.items()},
            "num_gt": self.num_gt,
            "num_pred": self.num_pred,
        }

    def to_latex(self) -> str:
        """Generate LaTeX table of results."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{COCO evaluation metrics.}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
            f"mAP@[.5:.95] & {self.mAP:.4f} \\\\",
            f"mAP@.50 & {self.mAP_50:.4f} \\\\",
            f"mAP@.75 & {self.mAP_75:.4f} \\\\",
            r"\midrule",
        ]
        for cls_name, ap in sorted(self.per_class_ap.items(), key=lambda x: -x[1]):
            escaped = cls_name.replace("_", r"\_")
            lines.append(f"AP@.50 ({escaped}) & {ap:.4f} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute Average Precision using 101-point interpolation (COCO style)."""
    precisions = [0.0] + list(precisions) + [0.0]
    recalls = [0.0] + list(recalls) + [1.0]

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = 0.0
        for r, pr in zip(recalls, precisions):
            if r >= t:
                p = max(p, pr)
        ap += p / 101

    return ap


def evaluate_coco(
    predictions: list[list[Detection]],
    ground_truths: list[list[Detection]],
    iou_thresholds: list[float] | None = None,
) -> COCOResults:
    """Compute COCO-style mAP metrics.

    Args:
        predictions: List of detection lists, one per image.
        ground_truths: List of ground truth lists, one per image.
        iou_thresholds: IoU thresholds (default: [0.5, 0.55, ..., 0.95]).

    Returns:
        COCOResults with mAP, per-class AP, and counts.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    # Collect all classes
    all_classes: set[str] = set()
    for gts in ground_truths:
        for gt in gts:
            all_classes.add(gt.class_name)
    for preds in predictions:
        for pred in preds:
            all_classes.add(pred.class_name)

    num_gt = sum(len(gts) for gts in ground_truths)
    num_pred = sum(len(preds) for preds in predictions)

    if num_gt == 0:
        return COCOResults(
            mAP=0.0,
            mAP_50=0.0,
            mAP_75=0.0,
            per_class_ap={},
            per_class_ap_75={},
            num_gt=0,
            num_pred=num_pred,
        )

    # Compute per-class, per-threshold AP
    class_aps: dict[float, dict[str, float]] = {t: {} for t in iou_thresholds}

    for cls in sorted(all_classes):
        for iou_thresh in iou_thresholds:
            # Gather all predictions and GTs for this class
            all_preds: list[tuple[float, int, Detection]] = []
            all_gts_count = 0

            for img_idx, (preds, gts) in enumerate(zip(predictions, ground_truths)):
                cls_preds = [p for p in preds if p.class_name == cls]
                cls_gts = [g for g in gts if g.class_name == cls]
                all_gts_count += len(cls_gts)

                for pred in cls_preds:
                    all_preds.append((pred.confidence, img_idx, pred))

            if all_gts_count == 0:
                continue

            # Sort predictions by confidence (descending)
            all_preds.sort(key=lambda x: -x[0])

            tp: list[int] = []
            fp: list[int] = []
            gt_matched: dict[tuple[int, int], bool] = {}

            for _conf, img_idx, pred in all_preds:
                cls_gts = [g for g in ground_truths[img_idx] if g.class_name == cls]

                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(cls_gts):
                    iou = compute_iou(pred.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh and (img_idx, best_gt_idx) not in gt_matched:
                    tp.append(1)
                    fp.append(0)
                    gt_matched[(img_idx, best_gt_idx)] = True
                else:
                    tp.append(0)
                    fp.append(1)

            # Compute precision-recall curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls_curve = tp_cumsum / all_gts_count
            precisions_curve = tp_cumsum / (tp_cumsum + fp_cumsum)

            ap = compute_ap(precisions_curve.tolist(), recalls_curve.tolist())
            class_aps[iou_thresh][cls] = ap

    # Aggregate
    def mean_ap(thresh: float) -> float:
        aps = class_aps.get(thresh, {})
        return float(np.mean(list(aps.values()))) if aps else 0.0

    mAP = float(np.mean([mean_ap(t) for t in iou_thresholds]))
    mAP_50 = mean_ap(0.5)
    mAP_75 = mean_ap(0.75)

    return COCOResults(
        mAP=mAP,
        mAP_50=mAP_50,
        mAP_75=mAP_75,
        per_class_ap=class_aps.get(0.5, {}),
        per_class_ap_75=class_aps.get(0.75, {}),
        num_gt=num_gt,
        num_pred=num_pred,
    )


def load_coco_annotations(json_path: str) -> dict[str, list[Detection]]:
    """Load COCO-format annotation file.

    Returns:
        Dict mapping image filename to list of Detection ground truths.
    """
    with open(json_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img["file_name"] for img in coco["images"]}

    annotations: dict[str, list[Detection]] = {}
    for img in coco["images"]:
        annotations[img["file_name"]] = []

    for ann in coco["annotations"]:
        img_name = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
        det = Detection(
            bbox=(x, y, x + w, y + h),
            confidence=1.0,
            class_id=ann["category_id"],
            class_name=categories[ann["category_id"]],
        )
        annotations[img_name].append(det)

    return annotations
