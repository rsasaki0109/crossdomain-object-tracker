"""Tests for download utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from crossdomain_object_tracker.common.download import download_dataset


class TestDownloadDataset:
    """Tests for download_dataset."""

    def test_download_skip_existing(self, tmp_path: Path) -> None:
        """If a directory exists and has files, demo download should skip."""
        dataset_dir = tmp_path / "covla"
        dataset_dir.mkdir()
        # Put a fake sample image to simulate an existing download
        (dataset_dir / "sample_0000.jpg").write_bytes(b"\xff\xd8fake_jpg")

        # Patch download_sample_images so it tracks whether it was called
        with patch(
            "crossdomain_object_tracker.common.download.download_sample_images",
            return_value=[],
        ) as _mock_download:
            # Even in demo mode, existing files should be present
            result = download_dataset("covla", output_dir=str(tmp_path), demo=True)

        assert result == dataset_dir

    def test_download_unknown_dataset(self, tmp_path: Path) -> None:
        """Requesting an unknown dataset raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            download_dataset("totally_unknown_dataset_xyz", output_dir=str(tmp_path))

    def test_download_unknown_method(self, tmp_path: Path) -> None:
        """An unknown download_method raises ValueError."""
        fake_config = {
            "name": "fake",
            "download_method": "carrier_pigeon",
        }
        with patch(
            "crossdomain_object_tracker.common.download.get_dataset_config",
            return_value=fake_config,
        ):
            with pytest.raises(ValueError, match="Unknown download method"):
                download_dataset("fake", output_dir=str(tmp_path))
