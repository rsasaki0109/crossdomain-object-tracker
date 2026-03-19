"""Tests for the tracker module."""

from __future__ import annotations

from crossdomain_object_tracker.tracker import Track
from crossdomain_object_tracker.tracker.byte_tracker import compute_tracking_metrics


def _make_track(
    track_id: int = 1,
    class_name: str = "car",
    num_frames: int = 10,
    conf: float = 0.8,
) -> Track:
    """Create a mock Track for testing."""
    return Track(
        track_id=track_id,
        class_name=class_name,
        frames=list(range(num_frames)),
        bboxes=[(0.0, 0.0, 100.0, 100.0)] * num_frames,
        confidences=[conf] * num_frames,
    )


class TestTrack:
    """Tests for the Track dataclass."""

    def test_duration(self) -> None:
        """duration returns the number of frames."""
        track = _make_track(num_frames=5)
        assert track.duration == 5

    def test_duration_single_frame(self) -> None:
        """duration for a single-frame track is 1."""
        track = _make_track(num_frames=1)
        assert track.duration == 1

    def test_avg_confidence(self) -> None:
        """avg_confidence computes the mean of confidence values."""
        track = Track(
            track_id=1,
            class_name="person",
            frames=[0, 1, 2],
            bboxes=[(0, 0, 10, 10)] * 3,
            confidences=[0.6, 0.8, 1.0],
        )
        assert abs(track.avg_confidence - 0.8) < 1e-9

    def test_avg_confidence_empty(self) -> None:
        """avg_confidence returns 0.0 for empty confidences list."""
        track = Track(
            track_id=1,
            class_name="car",
            frames=[],
            bboxes=[],
            confidences=[],
        )
        assert track.avg_confidence == 0.0

    def test_avg_confidence_uniform(self) -> None:
        """avg_confidence equals the uniform value."""
        track = _make_track(num_frames=10, conf=0.75)
        assert abs(track.avg_confidence - 0.75) < 1e-9


class TestComputeTrackingMetrics:
    """Tests for compute_tracking_metrics."""

    def test_basic_metrics(self) -> None:
        """Computes correct metrics for a list of tracks."""
        tracks = [
            _make_track(track_id=1, class_name="car", num_frames=10, conf=0.8),
            _make_track(track_id=2, class_name="person", num_frames=5, conf=0.6),
            _make_track(track_id=3, class_name="car", num_frames=20, conf=0.9),
        ]
        metrics = compute_tracking_metrics(tracks)

        assert metrics["num_tracks"] == 3
        assert abs(metrics["avg_duration"] - (10 + 5 + 20) / 3) < 1e-9
        assert metrics["max_duration"] == 20
        assert metrics["total_detections"] == 35
        assert metrics["classes"] == {"car": 2, "person": 1}

    def test_avg_confidence_metric(self) -> None:
        """avg_confidence in metrics is the mean of per-track avg confidences."""
        tracks = [
            _make_track(track_id=1, conf=0.8),
            _make_track(track_id=2, conf=0.6),
        ]
        metrics = compute_tracking_metrics(tracks)
        expected = (0.8 + 0.6) / 2
        assert abs(metrics["avg_confidence"] - expected) < 1e-9

    def test_empty_tracks(self) -> None:
        """Returns zero-value metrics for empty track list."""
        metrics = compute_tracking_metrics([])
        assert metrics["num_tracks"] == 0
        assert metrics["avg_duration"] == 0
        assert metrics["avg_confidence"] == 0
        assert metrics["classes"] == {}

    def test_single_track(self) -> None:
        """Works correctly with a single track."""
        tracks = [_make_track(track_id=1, class_name="dog", num_frames=7, conf=0.95)]
        metrics = compute_tracking_metrics(tracks)

        assert metrics["num_tracks"] == 1
        assert metrics["avg_duration"] == 7
        assert metrics["max_duration"] == 7
        assert abs(metrics["avg_confidence"] - 0.95) < 1e-9
        assert metrics["classes"] == {"dog": 1}
        assert metrics["total_detections"] == 7
