"""Configuration loading and management.

Loads dataset and model configurations from YAML files under configs/.

Usage:
    from crossdomain_object_tracker.common.config import load_datasets_config, load_models_config
    datasets = load_datasets_config()
    models = load_models_config()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root directory (contains configs/, src/, etc.)."""
    return Path(__file__).resolve().parents[3]


def _configs_dir() -> Path:
    """Return the path to the configs/ directory."""
    return get_project_root() / "configs"


def load_datasets_config(path: Path | None = None) -> dict[str, Any]:
    """Load dataset configuration from YAML.

    Args:
        path: Path to datasets.yaml. Defaults to configs/datasets.yaml.

    Returns:
        Full configuration dictionary with a top-level 'datasets' key.
    """
    path = path or _configs_dir() / "datasets.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_models_config(path: Path | None = None) -> dict[str, Any]:
    """Load model configuration from YAML.

    Args:
        path: Path to models.yaml. Defaults to configs/models.yaml.

    Returns:
        Full configuration dictionary with a top-level 'models' key.
    """
    path = path or _configs_dir() / "models.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataset_config(name: str) -> dict[str, Any]:
    """Get configuration for a specific dataset by name.

    Args:
        name: Dataset key (e.g. 'covla', 'polaris', 'mcd', 'hm3d-ovon').

    Returns:
        Dataset configuration dictionary.

    Raises:
        KeyError: If the dataset name is not found in the configuration.
    """
    config = load_datasets_config()
    datasets = config.get("datasets", {})
    if name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")
    return datasets[name]


def get_model_config(name: str) -> dict[str, Any]:
    """Get configuration for a specific model by name.

    Args:
        name: Model key (e.g. 'yolov8n', 'grounding-dino').

    Returns:
        Model configuration dictionary.

    Raises:
        KeyError: If the model name is not found in the configuration.
    """
    config = load_models_config()
    models = config.get("models", {})
    if name not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return models[name]


# Keep backward compatibility aliases
load_dataset_config = load_datasets_config
load_model_config = load_models_config
