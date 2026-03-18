"""Object detection backends (YOLOv8, Grounding DINO).

Defines the common Detection dataclass and BaseDetector protocol used
by all detector implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class Detection:
    """A single object detection result.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence score in [0, 1].
        class_id: Integer class ID from the model.
        class_name: Human-readable class name.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(
            bbox=tuple(d["bbox"]),  # type: ignore[arg-type]
            confidence=d["confidence"],
            class_id=d["class_id"],
            class_name=d["class_name"],
        )


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def detect(
        self, image: Union[str, Path, np.ndarray], **kwargs: object
    ) -> list[Detection]:
        """Run detection on a single image.

        Args:
            image: Image as file path or numpy array (BGR, H x W x 3).
            **kwargs: Additional detector-specific parameters.

        Returns:
            List of Detection objects.
        """
        ...

    def detect_batch(
        self, images: list[Union[str, Path, np.ndarray]], **kwargs: object
    ) -> list[list[Detection]]:
        """Run detection on a batch of images.

        Default implementation calls detect() in a loop.
        Subclasses can override for more efficient batched inference.

        Args:
            images: List of images (file paths or numpy arrays).
            **kwargs: Additional detector-specific parameters.

        Returns:
            List of lists of Detection objects.
        """
        return [self.detect(img, **kwargs) for img in images]


def get_detector(name: str, **kwargs: object) -> BaseDetector:
    """Factory function to create a detector by name.

    Args:
        name: Detector name. Supported values:
            - 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x': YOLOv8 variants
            - 'grounding-dino', 'grounding_dino': Grounding DINO
        **kwargs: Additional keyword arguments passed to the detector constructor.

    Returns:
        A BaseDetector instance.

    Raises:
        ValueError: If the detector name is not recognized.
    """
    name_lower = name.lower().replace("-", "_")

    if name_lower.startswith("yolov8"):
        from crossdomain_object_tracker.detector.yolo import YOLODetector

        # Extract variant suffix (e.g. 'n', 's', 'm', 'l', 'x')
        model_name = name if name.endswith(".pt") else f"{name}.pt"
        return YOLODetector(model_name=model_name, **kwargs)  # type: ignore[arg-type]

    elif name_lower in ("grounding_dino", "groundingdino"):
        from crossdomain_object_tracker.detector.grounding_dino import (
            GroundingDINODetector,
        )

        return GroundingDINODetector(**kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(
            f"Unknown detector: '{name}'. "
            "Supported: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, grounding-dino"
        )
