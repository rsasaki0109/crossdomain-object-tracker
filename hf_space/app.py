import streamlit as st
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Cross-Domain Object Tracker",
    layout="wide",
    page_icon="🔍",
)

st.title("Cross-Domain Object Tracker")
st.markdown("Run YOLOv8 object detection across different robotics domains and compare results.")

# Sidebar
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", ["yolov8n", "yolov8s"], index=0)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

# File upload
uploaded_files = st.sidebar.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

domain = st.sidebar.selectbox("Domain Label", ["driving", "maritime", "campus", "indoor", "other"])


@st.cache_resource
def load_model(name):
    from ultralytics import YOLO

    return YOLO(f"{name}.pt")


if uploaded_files:
    model = load_model(model_name)

    cols = st.columns(min(len(uploaded_files), 3))
    all_detections = []

    for idx, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")

        # Save to temp file for YOLO
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            results = model(tmp.name, conf=conf_threshold)

        result = results[0]

        # Draw detections
        annotated = result.plot()

        with cols[idx % len(cols)]:
            st.image(annotated[:, :, ::-1], caption=file.name, use_container_width=True)

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_name = result.names[int(box.cls)]
                    conf = float(box.conf)
                    detections.append({"class": cls_name, "confidence": f"{conf:.3f}"})
                    all_detections.append({"class": cls_name, "confidence": conf, "domain": domain})

            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.info("No detections")

    # Summary
    if all_detections:
        st.subheader("Detection Summary")
        import pandas as pd

        df = pd.DataFrame(all_detections)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Detections", len(all_detections))
        col2.metric("Unique Classes", df["class"].nunique())
        col3.metric("Avg Confidence", f"{df['confidence'].mean():.3f}")

        st.bar_chart(df["class"].value_counts())
else:
    st.info("Upload images in the sidebar to get started.")
    st.markdown("""
    ### How to use
    1. Upload one or more images from any domain (driving, maritime, campus, indoor)
    2. Select the detection model and confidence threshold
    3. View detection results and statistics

    ### Links
    - [GitHub Repository](https://github.com/rsasaki0109/crossdomain-object-tracker)
    - [Interactive Dashboard](https://rsasaki0109.github.io/crossdomain-object-tracker/)
    """)
