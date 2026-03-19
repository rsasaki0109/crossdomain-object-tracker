"""Streamlit demo app for Cross-Domain Object Tracker.

Provides an interactive UI to run YOLOv8 detection on uploaded or demo
images, compare results across domains, and export LaTeX tables.

Usage:
    pip install -e ".[app]"
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from crossdomain_object_tracker.detector import Detection, get_detector
    from crossdomain_object_tracker.report import (
        generate_latex_class_table,
        generate_latex_table,
    )
    from crossdomain_object_tracker.visualize import draw_detections

    HAS_TRACKER = True
except ImportError:
    HAS_TRACKER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
GALLERY_DIR = PROJECT_ROOT / "docs" / "gallery"
MODELS = ["yolov8n", "yolov8s", "yolov8m"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector(model_name: str, conf: float):
    """Load and cache a YOLOv8 detector."""
    return get_detector(model_name, confidence_threshold=conf)


def run_detection_on_image(
    detector,
    image_path: str | Path,
    conf: float,
) -> list[Detection]:
    """Run detection on a single image file."""
    return detector.detect(image_path, conf=conf)


def detections_to_dataframe(detections: list[Detection]) -> pd.DataFrame:
    """Convert a list of Detection objects to a DataFrame."""
    rows = []
    for d in detections:
        rows.append(
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "x1": int(d.bbox[0]),
                "y1": int(d.bbox[1]),
                "x2": int(d.bbox[2]),
                "y2": int(d.bbox[3]),
            }
        )
    return pd.DataFrame(rows)


def build_domain_results(
    all_results: dict[str, list[tuple[str, list[Detection]]]],
) -> dict[str, dict[str, Any]]:
    """Build evaluation-style result dicts from per-domain detection lists."""
    domain_results: dict[str, dict[str, Any]] = {}
    for domain, image_dets in all_results.items():
        all_dets: list[Detection] = []
        class_counter: Counter[str] = Counter()
        for _path, dets in image_dets:
            all_dets.extend(dets)
            for d in dets:
                class_counter[d.class_name] += 1

        confidence_scores = [d.confidence for d in all_dets]
        num_images = len(image_dets)
        domain_results[domain] = {
            "dataset": domain,
            "num_images": num_images,
            "total_detections": len(all_dets),
            "avg_detections_per_image": len(all_dets) / num_images if num_images else 0.0,
            "avg_confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            "class_distribution": dict(class_counter.most_common()),
            "confidence_scores": confidence_scores,
        }
    return domain_results


def load_demo_images() -> dict[str, list[Path]]:
    """Load demo images from docs/gallery/, grouped by domain."""
    domains: dict[str, list[Path]] = {}
    if not GALLERY_DIR.exists():
        return domains
    for domain_dir in sorted(GALLERY_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        images = sorted(p for p in domain_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if images:
            domains[domain_dir.name] = images
    return domains


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cross-Domain Object Tracker", layout="wide")
st.title("Cross-Domain Object Tracker")

if not HAS_TRACKER:
    st.error(
        "Could not import `crossdomain_object_tracker`. Make sure the package is installed: `pip install -e '.[app]'`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

model_name = st.sidebar.selectbox("Model", MODELS, index=0)
conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Image source")

uploaded_files = st.sidebar.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

use_demo = st.sidebar.button("Use demo images")

# ---------------------------------------------------------------------------
# Resolve images
# ---------------------------------------------------------------------------
# domain_name -> list of file paths
image_groups: dict[str, list[Path]] = {}

if use_demo or ("use_demo" in st.session_state and st.session_state["use_demo"]):
    st.session_state["use_demo"] = True
    image_groups = load_demo_images()
    if not image_groups:
        st.warning("No demo images found in `docs/gallery/`.")

if uploaded_files:
    # Reset demo flag when user uploads
    st.session_state["use_demo"] = False
    tmp_paths: list[Path] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_paths.append(Path(tmp.name))
    image_groups = {"uploaded": tmp_paths}

if not image_groups:
    st.info("Upload images or click **Use demo images** to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
try:
    detector = load_detector(model_name, conf_threshold)
except ImportError:
    st.error("ultralytics is not installed. Install it with: `pip install ultralytics`")
    st.stop()

all_results: dict[str, list[tuple[str, list[Detection]]]] = {}

with st.spinner("Running detection..."):
    for domain, paths in image_groups.items():
        domain_dets: list[tuple[str, list[Detection]]] = []
        for img_path in paths:
            dets = run_detection_on_image(detector, str(img_path), conf_threshold)
            domain_dets.append((str(img_path), dets))
        all_results[domain] = domain_dets

domain_results = build_domain_results(all_results)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_detect, tab_compare, tab_latex = st.tabs(["Detection", "Domain Comparison", "LaTeX Export"])

# ---- Tab 1: Detection -----------------------------------------------------
with tab_detect:
    for domain, image_dets in all_results.items():
        st.subheader(f"Domain: {domain}")
        for img_path, dets in image_dets:
            col_img, col_table = st.columns([1, 1])
            with col_img:
                annotated_bgr = draw_detections(img_path, dets)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(
                    annotated_rgb,
                    caption=f"{Path(img_path).name} ({len(dets)} detections)",
                    use_container_width=True,
                )
            with col_table:
                if dets:
                    st.dataframe(
                        detections_to_dataframe(dets),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("No detections.")

# ---- Tab 2: Domain Comparison ---------------------------------------------
with tab_compare:
    if len(domain_results) < 2:
        st.info("Load images from multiple domains to see a comparison.")
    else:
        if not HAS_PLOTLY:
            st.warning("Plotly is not installed. Install it with `pip install plotly` for interactive charts.")

        # Bar chart: detections per domain
        st.subheader("Detections per domain")
        det_df = pd.DataFrame(
            [{"domain": name, "detections": res["total_detections"]} for name, res in domain_results.items()]
        )
        if HAS_PLOTLY:
            fig_bar = px.bar(
                det_df,
                x="domain",
                y="detections",
                color="domain",
                title="Total detections per domain",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.bar_chart(det_df.set_index("domain"))

        # Box plot: confidence distribution
        st.subheader("Confidence distribution per domain")
        conf_rows = []
        for name, res in domain_results.items():
            for score in res["confidence_scores"]:
                conf_rows.append({"domain": name, "confidence": score})
        if conf_rows:
            conf_df = pd.DataFrame(conf_rows)
            if HAS_PLOTLY:
                fig_box = px.box(
                    conf_df,
                    x="domain",
                    y="confidence",
                    color="domain",
                    title="Confidence distribution per domain",
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.dataframe(conf_df.groupby("domain")["confidence"].describe())

        # Class distribution heatmap
        st.subheader("Class distribution heatmap")
        all_classes: set[str] = set()
        for res in domain_results.values():
            all_classes.update(res["class_distribution"].keys())
        if all_classes:
            sorted_classes = sorted(all_classes)
            domains_list = list(domain_results.keys())
            heat_data = np.zeros((len(domains_list), len(sorted_classes)))
            for i, dname in enumerate(domains_list):
                for j, cls in enumerate(sorted_classes):
                    heat_data[i, j] = domain_results[dname]["class_distribution"].get(cls, 0)
            if HAS_PLOTLY:
                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=heat_data,
                        x=sorted_classes,
                        y=domains_list,
                        colorscale="YlOrRd",
                    )
                )
                fig_heat.update_layout(
                    title="Class distribution across domains",
                    xaxis_title="Class",
                    yaxis_title="Domain",
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.dataframe(pd.DataFrame(heat_data, index=domains_list, columns=sorted_classes))

# ---- Tab 3: LaTeX Export ---------------------------------------------------
with tab_latex:
    st.subheader("Summary table")
    latex_summary = generate_latex_table(domain_results)
    st.code(latex_summary, language="latex")
    st.download_button(
        label="Download summary .tex",
        data=latex_summary,
        file_name="crossdomain_summary.tex",
        mime="text/plain",
    )

    st.subheader("Per-class table")
    latex_class = generate_latex_class_table(domain_results)
    st.code(latex_class, language="latex")
    st.download_button(
        label="Download class .tex",
        data=latex_class,
        file_name="crossdomain_classes.tex",
        mime="text/plain",
    )
