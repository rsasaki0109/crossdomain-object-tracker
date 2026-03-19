"""ByteTrack-based multi-object tracker using ultralytics.

Wraps the ultralytics YOLO tracker to provide consistent Track objects
for video files and image sequences.

Usage:
    from crossdomain_object_tracker.tracker.byte_tracker import ByteTracker
    tracker = ByteTracker(model_name="yolov8n.pt")
    tracks = tracker.track_video("video.mp4")
"""

from __future__ import annotations

import logging
from pathlib import Path

from crossdomain_object_tracker.tracker import Track

logger = logging.getLogger(__name__)


class ByteTracker:
    """Multi-object tracker using ByteTrack via ultralytics."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.5) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ByteTracker requires ultralytics. Install: pip install ultralytics")
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def track_video(self, video_path: str, output_path: str | None = None) -> list[Track]:
        """Run tracking on a video file.

        Args:
            video_path: Path to input video.
            output_path: If provided, save annotated video.

        Returns:
            List of Track objects.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path_obj}")

        results = self.model.track(
            source=str(video_path_obj),
            conf=self.conf,
            iou=self.iou,
            tracker="bytetrack.yaml",
            stream=True,
            save=output_path is not None,
            project=str(Path(output_path).parent) if output_path else None,
            name=Path(output_path).stem if output_path else None,
        )

        tracks_dict: dict[int, Track] = {}
        frame_idx = 0

        for frame_idx, result in enumerate(results):
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                bbox = tuple(boxes.xyxy[i].cpu().numpy().tolist())
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id]

                if track_id not in tracks_dict:
                    tracks_dict[track_id] = Track(
                        track_id=track_id,
                        class_name=cls_name,
                        frames=[],
                        bboxes=[],
                        confidences=[],
                    )

                tracks_dict[track_id].frames.append(frame_idx)
                tracks_dict[track_id].bboxes.append(bbox)
                tracks_dict[track_id].confidences.append(conf)

        tracks = sorted(tracks_dict.values(), key=lambda t: t.track_id)
        logger.info("Found %d tracks across %d frames", len(tracks), frame_idx + 1)
        return tracks

    def track_image_sequence(self, image_dir: str, output_dir: str | None = None) -> list[Track]:
        """Run tracking on a sequence of images.

        Args:
            image_dir: Path to directory containing images.
            output_dir: If provided, save annotated images.

        Returns:
            List of Track objects.
        """
        image_dir_path = Path(image_dir)
        images = sorted(list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png")))
        if not images:
            raise ValueError(f"No images found in {image_dir_path}")

        results = self.model.track(
            source=str(image_dir_path),
            conf=self.conf,
            iou=self.iou,
            tracker="bytetrack.yaml",
            stream=True,
            save=output_dir is not None,
            project=str(Path(output_dir).parent) if output_dir else None,
            name=Path(output_dir).stem if output_dir else None,
        )

        tracks_dict: dict[int, Track] = {}
        frame_idx = 0

        for frame_idx, result in enumerate(results):
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                bbox = tuple(boxes.xyxy[i].cpu().numpy().tolist())
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id]

                if track_id not in tracks_dict:
                    tracks_dict[track_id] = Track(
                        track_id=track_id,
                        class_name=cls_name,
                        frames=[],
                        bboxes=[],
                        confidences=[],
                    )

                tracks_dict[track_id].frames.append(frame_idx)
                tracks_dict[track_id].bboxes.append(bbox)
                tracks_dict[track_id].confidences.append(conf)

        return sorted(tracks_dict.values(), key=lambda t: t.track_id)


def compute_tracking_metrics(tracks: list[Track]) -> dict:
    """Compute tracking statistics.

    Args:
        tracks: List of Track objects from a tracker run.

    Returns:
        Dictionary with tracking summary metrics.
    """
    if not tracks:
        return {"num_tracks": 0, "avg_duration": 0, "avg_confidence": 0, "classes": {}}

    class_counts: dict[str, int] = {}
    for t in tracks:
        class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1

    return {
        "num_tracks": len(tracks),
        "avg_duration": sum(t.duration for t in tracks) / len(tracks),
        "max_duration": max(t.duration for t in tracks),
        "avg_confidence": sum(t.avg_confidence for t in tracks) / len(tracks),
        "classes": class_counts,
        "total_detections": sum(t.duration for t in tracks),
    }
