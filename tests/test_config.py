"""Tests for configuration loading and core data structures."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossdomain_object_tracker.common.config import (
    get_dataset_config,
    get_model_config,
    get_project_root,
    load_datasets_config,
    load_models_config,
)
from crossdomain_object_tracker.detector import Detection, get_detector

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


# --- Config tests ---


def test_get_project_root() -> None:
    """Project root contains expected directories."""
    root = get_project_root()
    assert (root / "configs").is_dir()
    assert (root / "src").is_dir()


def test_load_datasets_config() -> None:
    """Dataset config loads and contains expected datasets."""
    config = load_datasets_config(CONFIGS_DIR / "datasets.yaml")
    assert "datasets" in config
    datasets = config["datasets"]
    for name in ("covla", "polaris", "mcd", "hm3d-ovon"):
        assert name in datasets, f"Missing dataset: {name}"
        assert "domain" in datasets[name]
        assert "modalities" in datasets[name]


def test_load_datasets_config_has_required_fields() -> None:
    """Each dataset entry has all required fields."""
    config = load_datasets_config(CONFIGS_DIR / "datasets.yaml")
    required_fields = {"name", "description", "domain", "modalities", "license"}
    for name, ds in config["datasets"].items():
        for field in required_fields:
            assert field in ds, f"Dataset '{name}' missing field: {field}"


def test_load_models_config() -> None:
    """Model config loads and contains expected models."""
    config = load_models_config(CONFIGS_DIR / "models.yaml")
    assert "models" in config
    models = config["models"]
    assert "yolov8n" in models
    assert "grounding-dino" in models


def test_get_dataset_config() -> None:
    """get_dataset_config returns correct dataset."""
    ds = get_dataset_config("covla")
    assert ds["name"] == "CoVLA"
    assert ds["domain"] == "autonomous_driving"


def test_get_dataset_config_not_found() -> None:
    """get_dataset_config raises KeyError for unknown dataset."""
    with pytest.raises(KeyError, match="not found"):
        get_dataset_config("nonexistent_dataset")


def test_get_model_config() -> None:
    """get_model_config returns correct model."""
    model = get_model_config("yolov8n")
    assert model["name"] == "YOLOv8 Nano"
    assert model["weights"] == "yolov8n.pt"


def test_get_model_config_not_found() -> None:
    """get_model_config raises KeyError for unknown model."""
    with pytest.raises(KeyError, match="not found"):
        get_model_config("nonexistent_model")


# --- Detection dataclass tests ---


def test_detection_creation() -> None:
    """Detection dataclass can be created with expected fields."""
    det = Detection(
        bbox=(10.0, 20.0, 100.0, 200.0),
        confidence=0.95,
        class_id=0,
        class_name="car",
    )
    assert det.bbox == (10.0, 20.0, 100.0, 200.0)
    assert det.confidence == 0.95
    assert det.class_id == 0
    assert det.class_name == "car"


def test_detection_properties() -> None:
    """Detection computed properties are correct."""
    det = Detection(
        bbox=(10.0, 20.0, 110.0, 220.0),
        confidence=0.8,
        class_id=1,
        class_name="person",
    )
    assert det.width == 100.0
    assert det.height == 200.0
    assert det.area == 20000.0
    assert det.center == (60.0, 120.0)


def test_detection_to_dict() -> None:
    """Detection can be serialized to dict."""
    det = Detection(
        bbox=(10.0, 20.0, 100.0, 200.0),
        confidence=0.95,
        class_id=0,
        class_name="car",
    )
    d = det.to_dict()
    assert d["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert d["confidence"] == 0.95
    assert d["class_id"] == 0
    assert d["class_name"] == "car"


def test_detection_from_dict() -> None:
    """Detection can be deserialized from dict."""
    d = {
        "bbox": [10.0, 20.0, 100.0, 200.0],
        "confidence": 0.95,
        "class_id": 0,
        "class_name": "car",
    }
    det = Detection.from_dict(d)
    assert det.bbox == (10.0, 20.0, 100.0, 200.0)
    assert det.confidence == 0.95


def test_detection_roundtrip() -> None:
    """Detection survives dict serialization roundtrip."""
    original = Detection(
        bbox=(1.5, 2.5, 30.5, 40.5),
        confidence=0.75,
        class_id=3,
        class_name="bicycle",
    )
    reconstructed = Detection.from_dict(original.to_dict())
    assert original.bbox == reconstructed.bbox
    assert original.confidence == reconstructed.confidence
    assert original.class_id == reconstructed.class_id
    assert original.class_name == reconstructed.class_name


# --- get_detector factory tests ---


def test_get_detector_unknown() -> None:
    """get_detector raises ValueError for unknown detector."""
    with pytest.raises(ValueError, match="Unknown detector"):
        get_detector("unknown_model")


def test_get_detector_yolo_import() -> None:
    """get_detector with yolov8n returns YOLODetector (import check only)."""
    try:
        detector = get_detector("yolov8n")
        from crossdomain_object_tracker.detector.yolo import YOLODetector

        assert isinstance(detector, YOLODetector)
    except ImportError:
        pytest.skip("ultralytics not installed")


def test_get_detector_grounding_dino_import() -> None:
    """get_detector with grounding-dino returns GroundingDINODetector."""
    try:
        detector = get_detector("grounding-dino")
        from crossdomain_object_tracker.detector.grounding_dino import (
            GroundingDINODetector,
        )

        assert isinstance(detector, GroundingDINODetector)
    except ImportError:
        pytest.skip("transformers not installed")
