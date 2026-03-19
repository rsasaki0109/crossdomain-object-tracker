"""Grounding DINO open-vocabulary object detection backend.

Wraps the Grounding DINO model via HuggingFace transformers for
text-prompted object detection. Allows detecting arbitrary object
categories specified via natural language prompts.

Usage:
    from crossdomain_object_tracker.detector.grounding_dino import GroundingDINODetector
    detector = GroundingDINODetector()
    detections = detector.detect("image.jpg", text_prompt="car . person . tree")
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from crossdomain_object_tracker.detector import BaseDetector, Detection

# Domain-specific default text prompts for zero-shot detection.
# Each prompt uses dot-separated categories as required by Grounding DINO.
DEFAULT_DOMAIN_PROMPTS: dict[str, str] = {
    "maritime": "boat . ship . buoy . person . kayak . dock . bird . water",
    "autonomous_driving": "car . truck . bus . person . bicycle . motorcycle . traffic light . stop sign",
    "driving": "car . truck . bus . person . bicycle . motorcycle . traffic light . stop sign",
    "campus": "person . bicycle . bench . backpack . dog . skateboard . tree . building",
    "indoor": "chair . table . couch . tv . book . bottle . plant . lamp . bed",
    "indoor/construction": "wall . door . window . pipe . column . person . sign . equipment",
    "outdoor/extreme": "car . truck . bus . person . bicycle . building . sign . pole",
    "global/outdoor": "car . person . building . tree . sign . bicycle . animal . furniture",
}


class GroundingDINODetector(BaseDetector):
    """Open-vocabulary object detector using Grounding DINO via HuggingFace transformers.

    Uses AutoProcessor and AutoModelForZeroShotObjectDetection for inference.
    Supports domain-specific default prompts for zero-shot detection across
    different robotics domains.

    Args:
        model_id: HuggingFace model ID for Grounding DINO.
        box_threshold: Minimum confidence threshold for box predictions.
        text_threshold: Minimum confidence threshold for text-box association.
        device: Torch device string. If None, auto-selects CUDA when available.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Grounding DINO requires transformers and torch. Install with: pip install transformers torch"
            ) from e

        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """Lazily load the Grounding DINO model on first use."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for Grounding DINO. Install with: pip install transformers"
            ) from e

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    def _resolve_text_prompt(
        self,
        text_prompt: str | None = None,
        domain: str | None = None,
    ) -> str:
        """Resolve the text prompt, falling back to domain defaults.

        Args:
            text_prompt: Explicit text prompt. Used as-is if provided.
            domain: Domain name for looking up default prompts.

        Returns:
            A dot-separated text prompt string.
        """
        if text_prompt is not None:
            return text_prompt
        if domain is not None:
            return DEFAULT_DOMAIN_PROMPTS.get(domain, "object . thing")
        return "object . thing"

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        text_prompt: str | None = None,
        conf: float | None = None,
        domain: str | None = None,
        **kwargs: object,
    ) -> list[Detection]:
        """Run open-vocabulary detection on a single image.

        Args:
            image: Image file path or BGR numpy array (H, W, 3).
            text_prompt: Dot-separated object categories
                (e.g. "car . person . tree"). If None, uses the domain default.
            conf: Box threshold override. Uses instance default if None.
            domain: Domain name for default prompt lookup (e.g. "maritime",
                "autonomous_driving"). Ignored when text_prompt is provided.

        Returns:
            List of Detection objects sorted by confidence (descending).
        """
        self._ensure_loaded()

        import torch
        from PIL import Image as PILImage

        box_threshold = conf if conf is not None else self.box_threshold
        resolved_prompt = self._resolve_text_prompt(text_prompt, domain)

        # Load image
        if isinstance(image, (str, Path)):
            pil_image = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB
            rgb = image[:, :, ::-1] if image.ndim == 3 else image
            pil_image = PILImage.fromarray(rgb)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Parse text prompt - split by dots and clean up
        categories = [c.strip() for c in resolved_prompt.split(".") if c.strip()]
        # Grounding DINO expects a single text string with dots separating categories
        text = ". ".join(categories) + "."

        # Process inputs
        inputs = self._processor(images=pil_image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process
        w, h = pil_image.size
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        detections: list[Detection] = []
        if results:
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result.get("labels", result.get("text", []))

            # Handle labels - they may be text strings
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()

            for i in range(len(boxes)):
                bbox = boxes[i]
                score = float(scores[i])
                label = labels[i] if i < len(labels) else "object"

                # Map label text to a category index
                if isinstance(label, str):
                    label_text = label.strip().lower()
                    try:
                        class_id = categories.index(label_text)
                    except ValueError:
                        # Try partial match
                        class_id = -1
                        for idx, cat in enumerate(categories):
                            if cat.lower() in label_text or label_text in cat.lower():
                                class_id = idx
                                break
                        if class_id == -1:
                            class_id = 0
                    class_name = label_text
                else:
                    class_id = int(label)
                    class_name = categories[class_id] if class_id < len(categories) else str(class_id)

                detections.append(
                    Detection(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=score,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_batch(
        self,
        images: list[Union[str, Path, np.ndarray]],
        text_prompt: str | None = None,
        conf: float | None = None,
        domain: str | None = None,
        **kwargs: object,
    ) -> list[list[Detection]]:
        """Run detection on a batch of images.

        Args:
            images: List of image file paths or BGR numpy arrays.
            text_prompt: Dot-separated object categories.
            conf: Box threshold override.
            domain: Domain name for default prompt lookup.

        Returns:
            List of lists of Detection objects.
        """
        return [self.detect(img, text_prompt=text_prompt, conf=conf, domain=domain) for img in images]
