"""Generate an animated GIF demo for crossdomain-object-tracker."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "docs" / "demo.gif"
FRAMES = []
TARGET_SIZE = (1200, 600)  # Consistent frame size for GIF


def save_frame(fig, duration_ms=1500):
    """Save current matplotlib figure as a PIL Image frame."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)
    img = img.convert('RGB').resize(TARGET_SIZE, Image.LANCZOS)
    FRAMES.append((img, duration_ms))
    plt.close(fig)


# Load real results
with open(PROJECT_ROOT / "docs" / "results.json") as f:
    data = json.load(f)

domains = ["maritime", "driving", "campus", "indoor"]
colors = {
    "maritime": "#0ea5e9",
    "driving": "#f59e0b",
    "campus": "#10b981",
    "indoor": "#8b5cf6",
}
labels = {
    "maritime": "Maritime (PoLaRIS)",
    "driving": "Driving (CoVLA)",
    "campus": "Campus (MCD)",
    "indoor": "Indoor (HM3D-OVON)",
}

# --- Frame 1-2: Title card (2 seconds) ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.set_facecolor('#1a1a2e')
ax.text(0.5, 0.6, 'crossdomain-object-tracker',
        fontsize=32, ha='center', va='center', color='white', fontweight='bold')
ax.text(0.5, 0.4, 'Cross-domain object detection evaluation for robotics',
        fontsize=14, ha='center', va='center', color='#8b949e')
ax.text(0.5, 0.25, 'YOLOv8 · Grounding DINO · ByteTrack · COCO mAP',
        fontsize=11, ha='center', va='center', color='#58a6ff')
save_frame(fig, 2000)

# --- Frame 3-6: Detection images grid (4 seconds, shown as 2 frames) ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.set_facecolor('white')
fig.suptitle('Detection Results Across Domains', fontsize=16, fontweight='bold', y=1.02)
for i, domain in enumerate(domains):
    img_path = PROJECT_ROOT / "docs" / "gallery" / domain / f"{domain}_01.jpg"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        axes[i].imshow(img)
    else:
        # Fallback: generate a placeholder
        placeholder = np.random.randint(40, 80, (240, 320, 3), dtype=np.uint8)
        axes[i].imshow(placeholder)
    axes[i].set_title(labels[domain], fontsize=10, color=colors[domain], fontweight='bold')
    axes[i].axis('off')
    stats = data["domains"][domain]["stats"]
    axes[i].text(
        0.5, -0.12,
        f'{stats["total_detections"]} detections | avg conf: {stats["avg_confidence"]:.2f}',
        transform=axes[i].transAxes, ha='center', fontsize=8, color='#555',
    )
plt.tight_layout()
save_frame(fig, 2000)

# Second detection frame: show second images from each domain
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.set_facecolor('white')
fig.suptitle('Detection Results Across Domains (continued)', fontsize=16, fontweight='bold', y=1.02)
for i, domain in enumerate(domains):
    img_path = PROJECT_ROOT / "docs" / "gallery" / domain / f"{domain}_02.jpg"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        axes[i].imshow(img)
    else:
        img_path_fallback = PROJECT_ROOT / "docs" / "gallery" / domain / f"{domain}_01.jpg"
        if img_path_fallback.exists():
            img = mpimg.imread(str(img_path_fallback))
            axes[i].imshow(img)
        else:
            placeholder = np.random.randint(40, 80, (240, 320, 3), dtype=np.uint8)
            axes[i].imshow(placeholder)
    axes[i].set_title(labels[domain], fontsize=10, color=colors[domain], fontweight='bold')
    axes[i].axis('off')
plt.tight_layout()
save_frame(fig, 2000)

# --- Frame 7-8: Bar chart (2 seconds) ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor('white')
counts = [data["domains"][d]["stats"]["total_detections"] for d in domains]
bars = ax.bar(
    range(len(domains)), counts,
    color=[colors[d] for d in domains],
    edgecolor='white', linewidth=1.5,
)
ax.set_xticks(range(len(domains)))
ax.set_xticklabels([labels[d] for d in domains], fontsize=10)
ax.set_ylabel("Total Detections", fontsize=12)
ax.set_title("Detection Count Comparison", fontsize=16, fontweight='bold')
for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
        str(count), ha='center', fontweight='bold', fontsize=14,
    )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_frame(fig, 2000)

# --- Frame 9-10: Confidence distribution (2 seconds) ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor('white')
for domain in domains:
    confs = []
    for img_data in data["domains"][domain]["images"]:
        for det in img_data["detections"]:
            confs.append(det["confidence"])
    if confs:
        ax.hist(
            confs, bins=15, alpha=0.6,
            label=labels[domain], color=colors[domain], edgecolor='white',
        )
ax.set_xlabel("Confidence Score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Confidence Distribution by Domain", fontsize=16, fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_frame(fig, 2000)

# --- Frame 11-12: Class heatmap (2 seconds) ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor('white')
all_classes = sorted(
    data["comparison"]["classes"].keys(),
    key=lambda c: -data["comparison"]["classes"][c],
)[:8]
matrix = []
for domain in domains:
    row = [data["domains"][domain]["stats"]["classes"].get(c, 0) for c in all_classes]
    matrix.append(row)
matrix = np.array(matrix)
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(all_classes)))
ax.set_xticklabels(all_classes, rotation=45, ha='right')
ax.set_yticks(range(len(domains)))
ax.set_yticklabels([labels[d] for d in domains])
ax.set_title("Class Distribution Heatmap", fontsize=16, fontweight='bold')
for i in range(len(domains)):
    for j in range(len(all_classes)):
        ax.text(
            j, i, str(matrix[i, j]),
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white' if matrix[i, j] > matrix.max() / 2 else 'black',
        )
plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
save_frame(fig, 2000)

# --- Frame 13: Final card (1 second) ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.set_facecolor('#1a1a2e')
ax.text(0.5, 0.6, 'github.com/rsasaki0109/crossdomain-object-tracker',
        fontsize=18, ha='center', color='#58a6ff')
ax.text(0.5, 0.4, 'pip install -e . && crossdomain-tracker evaluate',
        fontsize=13, ha='center', color='#8b949e', family='monospace')
save_frame(fig, 2000)

# --- Save GIF ---
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
images = [f[0] for f in FRAMES]
durations = [f[1] for f in FRAMES]
images[0].save(
    str(OUTPUT),
    save_all=True,
    append_images=images[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB, {len(FRAMES)} frames)")
