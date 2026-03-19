"""Multi-object tracking module.

Provides the Track dataclass and ByteTrack-based tracker for
video and image sequence tracking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    """A single object track across frames."""

    track_id: int
    class_name: str
    frames: list[int]  # frame indices where this track appears
    bboxes: list[tuple[float, float, float, float]]  # bbox per frame
    confidences: list[float]

    @property
    def duration(self) -> int:
        return len(self.frames)

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
