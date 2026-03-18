"""Dataset download utilities.

Provides functions to download and extract datasets from various sources
(HuggingFace Hub, Google Drive, direct URLs). Supports resumable downloads
and progress display via tqdm.

Usage:
    from crossdomain_object_tracker.common.download import download_dataset
    download_dataset("covla", output_dir="data/")
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

from crossdomain_object_tracker.common.config import get_dataset_config

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_file_requests(url: str, dest: Path, desc: str | None = None) -> Path:
    """Download a file from a URL using requests with tqdm progress."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(
            f"Required package not installed: {e.name}. "
            "Install with: pip install requests tqdm"
        ) from e

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=desc or dest.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def _download_file_urllib(url: str, dest: Path, desc: str | None = None) -> Path:
    """Fallback download using urllib (no extra dependencies)."""
    logger.info("Downloading %s -> %s", url, dest)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            return
        if not hasattr(_reporthook, "_pbar"):
            _reporthook._pbar = tqdm(  # type: ignore[attr-defined]
                total=total_size, unit="B", unit_scale=True, desc=desc or dest.name
            )
        _reporthook._pbar.update(block_size)  # type: ignore[attr-defined]

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)

    if hasattr(_reporthook, "_pbar"):
        _reporthook._pbar.close()  # type: ignore[attr-defined]
        del _reporthook._pbar  # type: ignore[attr-defined]

    return dest


def download_from_url(url: str, dest: Path, desc: str | None = None) -> Path:
    """Download a file from a direct URL.

    Tries requests first, falls back to urllib.
    Skips download if file already exists.
    """
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("File already exists, skipping: %s", dest)
        return dest

    try:
        return _download_file_requests(url, dest, desc)
    except ImportError:
        return _download_file_urllib(url, dest, desc)


def download_from_huggingface(
    repo_id: str,
    output_dir: Path,
    max_samples: int | None = None,
) -> Path:
    """Download dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g. 'tier4/CoVLA_Dataset').
        output_dir: Directory to save downloaded files.
        max_samples: Maximum number of files to download (None = all).

    Returns:
        Path to the downloaded dataset directory.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for HuggingFace downloads. "
            "Install with: pip install huggingface-hub"
        ) from e

    from tqdm import tqdm

    _ensure_dir(output_dir)
    api = HfApi()

    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning(
            "Could not list files from HuggingFace repo '%s': %s. "
            "The dataset may require authentication or may not be publicly available.",
            repo_id,
            exc,
        )
        raise

    # Filter for image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]

    if not image_files:
        # If no image files, take any files
        image_files = list(files)

    if max_samples is not None:
        image_files = image_files[:max_samples]

    logger.info("Downloading %d files from %s", len(image_files), repo_id)

    for fname in tqdm(image_files, desc=f"Downloading from {repo_id}"):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
        except Exception as exc:
            logger.warning("Failed to download %s: %s", fname, exc)

    return output_dir


def download_from_gdrive(file_id: str, output_dir: Path, filename: str) -> Path:
    """Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID.
        output_dir: Directory to save downloaded file.
        filename: Output filename.

    Returns:
        Path to the downloaded file.
    """
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required for Google Drive downloads. "
            "Install with: pip install gdown"
        ) from e

    _ensure_dir(output_dir)
    dest = output_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("File already exists, skipping: %s", dest)
        return dest

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False)
    return dest


def download_from_git(repo_url: str, output_dir: Path) -> Path:
    """Clone a git repository.

    Args:
        repo_url: Git repository URL.
        output_dir: Directory to clone into.

    Returns:
        Path to the cloned repository.
    """
    import subprocess

    if output_dir.exists() and (output_dir / ".git").exists():
        logger.info("Repository already cloned: %s", output_dir)
        return output_dir

    _ensure_dir(output_dir.parent)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(output_dir)],
        check=True,
    )
    return output_dir


def download_sample_images(
    dataset_name: str,
    output_dir: Path,
    dataset_config: dict[str, Any] | None = None,
) -> list[Path]:
    """Download sample images for a dataset from public URLs.

    This is the quick demo mode - downloads a few sample images that can
    be used for testing without needing full dataset access.

    Args:
        dataset_name: Name of the dataset.
        output_dir: Directory to save sample images.
        dataset_config: Dataset configuration dict. If None, loaded from config.

    Returns:
        List of paths to downloaded images.
    """
    if dataset_config is None:
        dataset_config = get_dataset_config(dataset_name)

    sample_urls = dataset_config.get("sample_images_url", [])
    if not sample_urls:
        logger.warning(
            "No sample image URLs configured for dataset '%s'. "
            "You may need to download the full dataset manually.",
            dataset_name,
        )
        return []

    img_dir = _ensure_dir(output_dir / dataset_name)
    downloaded: list[Path] = []

    for i, url in enumerate(sample_urls):
        ext = ".jpg"
        dest = img_dir / f"sample_{i:04d}{ext}"
        try:
            download_from_url(url, dest, desc=f"{dataset_name} sample {i}")
            downloaded.append(dest)
        except Exception as exc:
            logger.warning("Failed to download sample image %s: %s", url, exc)

    return downloaded


def download_dataset(
    name: str,
    output_dir: str | Path = "data/",
    max_samples: int | None = None,
    demo: bool = False,
) -> Path:
    """Download a dataset by name.

    Main entry point for dataset downloading. Reads configuration from
    configs/datasets.yaml and dispatches to the appropriate download method.

    Args:
        name: Dataset name (e.g. 'covla', 'polaris', 'mcd', 'hm3d-ovon').
        output_dir: Root directory for downloaded data.
        max_samples: Maximum number of samples to download (None = all).
        demo: If True, download only sample images for quick testing.

    Returns:
        Path to the downloaded dataset directory.
    """
    output_dir = Path(output_dir)
    dataset_config = get_dataset_config(name)
    dataset_dir = output_dir / name

    # Demo mode: just download sample images
    if demo:
        logger.info("Demo mode: downloading sample images for '%s'", name)
        downloaded = download_sample_images(name, output_dir, dataset_config)
        if downloaded:
            logger.info(
                "Downloaded %d sample images to %s", len(downloaded), dataset_dir
            )
        else:
            logger.warning("No sample images available for '%s'", name)
        return dataset_dir

    download_method = dataset_config.get("download_method", "url")

    if download_method == "huggingface":
        repo_id = dataset_config.get("huggingface_repo")
        if not repo_id:
            raise ValueError(
                f"Dataset '{name}' is configured for HuggingFace download "
                "but no 'huggingface_repo' is specified."
            )
        try:
            return download_from_huggingface(
                repo_id, dataset_dir, max_samples=max_samples
            )
        except Exception as exc:
            logger.warning(
                "HuggingFace download failed for '%s': %s. "
                "Falling back to sample images.",
                name,
                exc,
            )
            download_sample_images(name, output_dir, dataset_config)
            return dataset_dir

    elif download_method == "gdrive":
        gdrive_id = dataset_config.get("gdrive_id", "")
        filename = dataset_config.get("gdrive_filename", f"{name}.zip")
        return download_from_gdrive(gdrive_id, dataset_dir, filename)

    elif download_method == "git":
        repo_url = dataset_config.get("git_url", dataset_config.get("url", ""))
        return download_from_git(repo_url, dataset_dir)

    elif download_method == "url":
        # For datasets with direct URL download, fall back to sample images
        # since full datasets usually require manual download or auth
        logger.info(
            "Dataset '%s' requires manual download from: %s. "
            "Downloading sample images instead.",
            name,
            dataset_config.get("url", "N/A"),
        )
        download_sample_images(name, output_dir, dataset_config)
        return dataset_dir

    else:
        raise ValueError(f"Unknown download method: {download_method}")
