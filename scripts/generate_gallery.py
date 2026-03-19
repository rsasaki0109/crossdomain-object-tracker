#!/usr/bin/env python3
"""Download sample images from 4 domains and run YOLOv8 detection, saving annotated images and results."""

import json
import ssl
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
GALLERY_DIR = BASE_DIR / "docs" / "gallery"
RESULTS_PATH = BASE_DIR / "docs" / "results.json"

SAMPLE_IMAGES = {
    "maritime": [
        ("https://images.unsplash.com/photo-1500514966906-fe245eea9344?w=640", "maritime_01.jpg"),
        ("https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=640", "maritime_02.jpg"),
        ("https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=640", "maritime_03.jpg"),
    ],
    "driving": [
        ("https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640", "driving_01.jpg"),
        ("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=640", "driving_02.jpg"),
        ("https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=640", "driving_03.jpg"),
    ],
    "campus": [
        ("https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=640", "campus_01.jpg"),
        ("https://images.unsplash.com/photo-1562774053-701939374585?w=640", "campus_02.jpg"),
        ("https://images.unsplash.com/photo-1498243691581-b145c3f54a5a?w=640", "campus_03.jpg"),
    ],
    "indoor": [
        ("https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=640", "indoor_01.jpg"),
        ("https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=640", "indoor_02.jpg"),
        ("https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?w=640", "indoor_03.jpg"),
    ],
}

# Domain-specific colours (BGR for OpenCV)
DOMAIN_COLORS = {
    "maritime": (0xE9, 0xA5, 0x0E),  # #0ea5e9
    "driving": (0x0B, 0x9E, 0xF5),  # #f59e0b
    "campus": (0x81, 0xB9, 0x10),  # #10b981
    "indoor": (0xF6, 0x5C, 0x8B),  # #8b5cf6
}

MODEL_NAME = "yolov8n"
CONF_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_image(url: str, dest: Path) -> bool:
    """Download an image from *url* to *dest*. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  Downloaded {dest.name} ({len(data) / 1024:.0f} KB)")
        return True
    except Exception as exc:
        print(f"  SKIP {dest.name}: {exc}")
        return False


def draw_detections(img: np.ndarray, detections: list, color: tuple) -> np.ndarray:
    """Draw bounding boxes with labels on *img* (in-place) and return it."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Ensure output dirs exist
    for domain in SAMPLE_IMAGES:
        (GALLERY_DIR / domain).mkdir(parents=True, exist_ok=True)

    # Load YOLOv8n
    print(f"Loading {MODEL_NAME} model ...")
    model = YOLO(f"{MODEL_NAME}.pt")

    all_results: dict = {
        "model": MODEL_NAME,
        "domains": {},
        "comparison": {},
    }

    grand_total_detections = 0
    grand_total_images = 0
    grand_all_confs: list[float] = []
    grand_classes: dict[str, int] = {}

    for domain, entries in SAMPLE_IMAGES.items():
        print(f"\n=== Domain: {domain} ===")
        domain_dir = GALLERY_DIR / domain
        color = DOMAIN_COLORS[domain]

        domain_images: list[dict] = []
        domain_total_dets = 0
        domain_confs: list[float] = []
        domain_classes: dict[str, int] = {}

        for url, filename in entries:
            img_path = domain_dir / filename
            if not download_image(url, img_path):
                continue

            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Could not read {img_path}")
                continue

            # Run detection
            results = model(img, conf=CONF_THRESHOLD, verbose=False)
            detections: list[dict] = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 1) for v in bbox],
                    }
                )
                domain_confs.append(conf)
                grand_all_confs.append(conf)
                domain_classes[cls_name] = domain_classes.get(cls_name, 0) + 1
                grand_classes[cls_name] = grand_classes.get(cls_name, 0) + 1

            # Draw and save annotated image
            annotated = draw_detections(img, detections, color)
            cv2.imwrite(str(img_path), annotated)
            print(f"  {filename}: {len(detections)} detections")

            domain_images.append({"filename": filename, "detections": detections})
            domain_total_dets += len(detections)
            grand_total_detections += len(detections)
            grand_total_images += 1

        all_results["domains"][domain] = {
            "images": domain_images,
            "stats": {
                "total_detections": domain_total_dets,
                "avg_confidence": round(sum(domain_confs) / len(domain_confs), 4) if domain_confs else 0,
                "classes": dict(sorted(domain_classes.items(), key=lambda x: -x[1])),
            },
        }

    # Comparison summary
    all_results["comparison"] = {
        "total_images": grand_total_images,
        "total_detections": grand_total_detections,
        "avg_confidence": round(sum(grand_all_confs) / len(grand_all_confs), 4) if grand_all_confs else 0,
        "classes": dict(sorted(grand_classes.items(), key=lambda x: -x[1])),
    }

    # Write JSON
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Gallery saved to {GALLERY_DIR}")
    print(f"Total: {grand_total_images} images, {grand_total_detections} detections")


if __name__ == "__main__":
    main()
