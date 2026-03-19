#!/usr/bin/env python3
"""Generate demo comparison image for crossdomain-object-tracker README."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
import numpy as np
import os

os.chdir(Path(__file__).resolve().parent.parent)

# Load results
with open("docs/results.json") as f:
    data = json.load(f)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Cross-Domain Object Detection Comparison (YOLOv8n)", fontsize=16, fontweight='bold', y=0.98)

domains = ["maritime", "driving", "campus", "indoor"]
domain_colors = {"maritime": "#0ea5e9", "driving": "#f59e0b", "campus": "#10b981", "indoor": "#8b5cf6"}
domain_labels = {"maritime": "Maritime\n(PoLaRIS)", "driving": "Driving\n(CoVLA)", "campus": "Campus\n(MCD)", "indoor": "Indoor\n(HM3D-OVON)"}

# Top row: sample images or placeholder cards
for i, domain in enumerate(domains):
    ax = fig.add_subplot(2, 4, i + 1)
    img_path = Path(f"docs/gallery/{domain}/{domain}_01.jpg")
    if img_path.exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
    else:
        # Create a stylish placeholder card
        color = domain_colors[domain]
        ax.set_facecolor('#f8f9fa')
        rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                                        facecolor=color, alpha=0.15, edgecolor=color, linewidth=2,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        stats = data["domains"][domain]["stats"]
        classes_str = ", ".join(list(stats["classes"].keys())[:3])
        ax.text(0.5, 0.55, domain.upper(), transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
        ax.text(0.5, 0.38, f"{stats['total_detections']} detections", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='#555555')
        ax.text(0.5, 0.25, f"avg conf: {stats['avg_confidence']:.3f}", transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='#777777')
        ax.text(0.5, 0.12, classes_str, transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='#999999', style='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title(domain_labels[domain], fontsize=11, color=domain_colors[domain], fontweight='bold')
    ax.axis('off')

# Bottom left: detection count bar chart
ax_bar = fig.add_subplot(2, 2, 3)
counts = [data["domains"][d]["stats"]["total_detections"] for d in domains]
bars = ax_bar.bar(range(len(domains)), counts, color=[domain_colors[d] for d in domains], edgecolor='white', linewidth=1.5)
ax_bar.set_xticks(range(len(domains)))
ax_bar.set_xticklabels([d.capitalize() for d in domains])
ax_bar.set_ylabel("Total Detections")
ax_bar.set_title("Detection Count by Domain", fontweight='bold')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
for bar, count in zip(bars, counts):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha='center', fontweight='bold', fontsize=11)

# Bottom right: class distribution (stacked bar by domain)
ax_class = fig.add_subplot(2, 2, 4)
all_classes = set()
for d in domains:
    all_classes.update(data["domains"][d]["stats"]["classes"].keys())
top_classes = sorted(all_classes,
                     key=lambda c: sum(data["domains"][d]["stats"]["classes"].get(c, 0) for d in domains),
                     reverse=True)[:6]

x = np.arange(len(top_classes))
width = 0.2
for i, domain in enumerate(domains):
    vals = [data["domains"][domain]["stats"]["classes"].get(c, 0) for c in top_classes]
    ax_class.bar(x + i * width, vals, width, label=domain.capitalize(),
                 color=domain_colors[domain], alpha=0.85, edgecolor='white', linewidth=0.5)

ax_class.set_xticks(x + width * 1.5)
ax_class.set_xticklabels(top_classes, rotation=30, ha='right')
ax_class.set_ylabel("Count")
ax_class.set_title("Top Classes by Domain", fontweight='bold')
ax_class.legend(fontsize=8, framealpha=0.9)
ax_class.spines['top'].set_visible(False)
ax_class.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("docs/demo_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
print("Saved docs/demo_comparison.png")
