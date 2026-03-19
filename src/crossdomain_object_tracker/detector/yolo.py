"""YOLOv8 object detection backend.

Wraps the Ultralytics YOLOv8 model for inference on arbitrary image datasets.
Supports multiple YOLOv8 variants (nano, small, medium, large, xlarge) and
provides a unified detection interface returning Detection dataclass instances.

Usage:
    from crossdomain_object_tracker.detector.yolo import YOLODetector
    detector = YOLODetector(model_name="yolov8n.pt")
    detections = detector.detect("image.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from crossdomain_object_tracker.detector import BaseDetector, Detection


class YOLODetector(BaseDetector):
    """YOLOv8-based object detector.

    Args:
        model_name: YOLOv8 weights file (e.g. 'yolov8n.pt').
            Automatically downloaded if not present locally.
        confidence_threshold: Default minimum confidence for detections.
        device: Torch device string ('cpu', 'cuda', 'cuda:0', etc.).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazily load the YOLO model on first use."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLOv8 detection. Install with: pip install ultralytics"
            ) from e

        self._model = YOLO(self.model_name)
        self._model.to(self.device)

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf: float | None = None,
        **kwargs: object,
    ) -> list[Detection]:
        """Run detection on a single image.

        Args:
            image: Image file path (str/Path) or BGR numpy array.
            conf: Confidence threshold. Uses instance default if None.

        Returns:
            List of Detection objects sorted by confidence (descending).
        """
        self._ensure_loaded()
        conf = conf if conf is not None else self.confidence_threshold

        # Convert Path to string for ultralytics
        if isinstance(image, Path):
            image = str(image)

        results = self._model(image, conf=conf, verbose=False)  # type: ignore[union-attr]
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = result.names.get(class_id, str(class_id))

                detections.append(
                    Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_batch(
        self,
        images: list[Union[str, Path, np.ndarray]],
        conf: float | None = None,
        **kwargs: object,
    ) -> list[list[Detection]]:
        """Run detection on a batch of images.

        Args:
            images: List of image paths or numpy arrays.
            conf: Confidence threshold. Uses instance default if None.

        Returns:
            List of lists of Detection objects, one list per image.
        """
        self._ensure_loaded()
        conf = conf if conf is not None else self.confidence_threshold

        # Convert Paths to strings
        processed = [str(img) if isinstance(img, Path) else img for img in images]

        results = self._model(processed, conf=conf, verbose=False)  # type: ignore[union-attr]
        all_detections: list[list[Detection]] = []

        for result in results:
            detections: list[Detection] = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names.get(class_id, str(class_id))

                    detections.append(
                        Detection(
                            bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )

            detections.sort(key=lambda d: d.confidence, reverse=True)
            all_detections.append(detections)

        return all_detections
