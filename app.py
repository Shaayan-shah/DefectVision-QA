"""
Interactive Streamlit Web Dashboard for DefectVision-QA.
Provides real-time parameter tuning, visual inspection, and compliance report generation.
"""

import os
import io
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from defect_vision.preprocessing import SurfacePreprocessor
from defect_vision.defect_engine import DefectDetector
from defect_vision.golden_comparator import GoldenTemplateComparator
from defect_vision.visualizer import QualityVisualizer
from defect_vision.reporter import QualityAuditReporter

# Page configuration
st.set_page_config(
    page_title="DefectVision-QA | Quality Inspection Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for industrial clean UI
st.markdown("""
<style>
    .metric-card {
        background-color: #161F30;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-pass { color: #2ECC71; }
    .status-fail { color: #E74C3C; }
    .status-warn { color: #F39C12; }
</style>
""", unsafe_allow_html=True)


def load_image_from_file_or_upload(selected_sample, uploaded_file):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img, uploaded_file.name
    
    samples_dir = "samples"
    sample_map = {
        "Pristine Machined Surface [PASS]": "sample_pass_pristine.png",
        "Hairline Fracture / Crack [FAIL]": "sample_fail_crack.png",
        "Linear Handling Scratches [FAIL]": "sample_fail_scratches.png",
        "Porosity & Pitting Voids [FAIL]": "sample_fail_voids.png",
        "Multi-Defect Composite [FAIL]": "sample_fail_multidefect.png",
    }
    rel_path = os.path.join(samples_dir, sample_map.get(selected_sample, "sample_pass_pristine.png"))
    if os.path.exists(rel_path):
        img = cv2.imread(rel_path)
        return img, os.path.basename(rel_path)
    return None, "none"


def main():
    st.title("DefectVision-QA: Industrial Surface Inspection Platform")
    st.caption("Deterministic real-time flaw detection, geometric contour classification, and QA compliance gating.")

    # Sidebar parameters
    st.sidebar.header("Inspection Configuration")
    
    mode = st.sidebar.radio(
        "Inspection Mode",
        ["Reference-Free (Morphological Gradient)", "Golden-Template Comparative"],
        index=0
    )

    st.sidebar.subheader("Detection Sensitivity")
    sensitivity = st.sidebar.slider(
        "Gradient Sensitivity",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Higher values detect fainter micro-abrasions but increase sensitivity to rough background texture."
    )
    
    min_area = st.sidebar.number_input(
        "Minimum Defect Area (px)",
        min_value=5.0,
        max_value=1000.0,
        value=25.0,
        step=5.0,
        help="Filters out high-frequency camera noise granules smaller than this threshold."
    )

    st.sidebar.subheader("Quality Tolerances")
    max_allowed_defects = st.sidebar.number_input(
        "Max Allowable Defects",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Maximum flaw count before component is rejected."
    )
    
    max_flaw_area_pct = st.sidebar.number_input(
        "Max Flaw Area Limit (%)",
        min_value=0.001,
        max_value=5.0,
        value=0.05,
        step=0.01,
        format="%.3f"
    )

    critical_crack = st.sidebar.checkbox("Classify CRACK as Critical Reject", value=True)
    critical_classes = ["CRACK"] if critical_crack else []

    # Main sample selection
    st.subheader("Component Ingestion")
    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        sample_choice = st.selectbox(
            "Select Benchmark Sample Target",
            [
                "Pristine Machined Surface [PASS]",
                "Hairline Fracture / Crack [FAIL]",
                "Linear Handling Scratches [FAIL]",
                "Porosity & Pitting Voids [FAIL]",
                "Multi-Defect Composite [FAIL]",
            ]
        )

    with col_input2:
        uploaded_image = st.file_uploader(
            "Or Upload Custom Surface Image",
            type=["png", "jpg", "jpeg", "bmp", "tif"]
        )

    image, image_name = load_image_from_file_or_upload(sample_choice, uploaded_image)

    if image is None:
        st.warning("No image target loaded. Generate samples or upload a file.")
        return

    # Golden template handling if comparative mode
    golden_image = None
    if mode == "Golden-Template Comparative":
        st.sidebar.subheader("Golden Reference Template")
        golden_path = "samples/golden_reference.png"
        if os.path.exists(golden_path):
            golden_image = cv2.imread(golden_path)
            st.sidebar.success("Default Golden Reference Loaded")
        else:
            st.sidebar.error("Golden reference not found in samples/ directory.")

    # Execution
    preprocessor = SurfacePreprocessor()
    detector = DefectDetector(
        min_defect_area=min_area,
        sensitivity=sensitivity,
        tolerance_max_defects=max_allowed_defects,
        tolerance_max_area_pct=max_flaw_area_pct,
        critical_classes=critical_classes,
        preprocessor=preprocessor
    )
    visualizer = QualityVisualizer()
    reporter = QualityAuditReporter()

    if mode == "Golden-Template Comparative" and golden_image is not None:
        comparator = GoldenTemplateComparator(
            golden_template=golden_image,
            min_defect_area=min_area,
            tolerance_max_defects=max_allowed_defects,
            preprocessor=preprocessor
        )
        result, processed_ref, binary_mask = comparator.compare(image)
    else:
        result, processed_ref, binary_mask = detector.inspect(image)

    annotated = visualizer.annotate_frame(image, result, show_hud=True)
    annotated_no_hud = visualizer.annotate_frame(image, result, show_hud=False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quad_panel = visualizer.create_diagnostic_quad(image, gray, binary_mask, annotated_no_hud)

    # Metrics Bar
    st.markdown("---")
    st.subheader("Inspection Determination & Telemetry")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    status_class = "status-pass" if result.status == "PASS" else ("status-fail" if result.status == "FAIL" else "status-warn")
    
    with m_col1:
        st.markdown(
            f"""<div class="metric-card">
                <div>Quality Gate Status</div>
                <div class="metric-value {status_class}">[{result.status}]</div>
            </div>""",
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            f"""<div class="metric-card">
                <div>Detected Flaws</div>
                <div class="metric-value">{result.defect_count}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with m_col3:
        st.markdown(
            f"""<div class="metric-card">
                <div>Flaw Coverage</div>
                <div class="metric-value">{result.defect_area_ratio_pct:.3f}%</div>
            </div>""",
            unsafe_allow_html=True
        )

    with m_col4:
        fps = 1000.0 / max(0.1, result.processing_time_ms)
        st.markdown(
            f"""<div class="metric-card">
                <div>Inference Latency</div>
                <div class="metric-value">{result.processing_time_ms:.1f} ms <span style="font-size: 14px; color: #94A3B8;">({fps:.0f} FPS)</span></div>
            </div>""",
            unsafe_allow_html=True
        )

    if result.rejection_reasons:
        st.error("Rejection Criteria Triggered:\n" + "\n".join([f"* {r}" for r in result.rejection_reasons]))

    # Visualization Tabs
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Annotated Inspection Frame", "2x2 Diagnostic Quad", "Binary Segmentation Mask"])

    with tab1:
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption=f"Annotated Inspection Output: {image_name}", use_container_width=True)

    with tab2:
        quad_rgb = cv2.cvtColor(quad_panel, cv2.COLOR_BGR2RGB)
        st.image(quad_rgb, caption="Four-Panel Diagnostic Pipeline View", use_container_width=True)

    with tab3:
        st.image(binary_mask, caption="Segmented Morphological Gradient Defect Mask", use_container_width=True)

    # Defect Manifest Table
    st.markdown("---")
    st.subheader("Defect Manifest & Geometry")
    if result.defects:
        defect_records = []
        for d in result.defects:
            defect_records.append({
                "ID": d.defect_id,
                "Classification": d.defect_type,
                "Severity (%)": d.severity_score,
                "Area (px)": d.area,
                "Aspect Ratio": d.aspect_ratio,
                "Solidity": d.solidity,
                "Centroid (X, Y)": f"({d.centroid[0]}, {d.centroid[1]})",
                "Bounding Box (X, Y, W, H)": f"({d.bbox[0]}, {d.bbox[1]}, {d.bbox[2]}, {d.bbox[3]})"
            })
        df = pd.DataFrame(defect_records)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No surface defects detected within specified tolerance parameters. Substrate conforms to nominal specifications.")

    # Export & Compliance
    st.markdown("---")
    st.subheader("Compliance Export & Audit Trail")

    serial_input = st.text_input("Component Serial ID for Audit Record", value=f"PART_{os.path.splitext(image_name)[0].upper()}")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # Prepare JSON record
    record = {
        "serial_id": serial_input,
        "summary": result.to_dict()
    }
    json_bytes = json.dumps(record, indent=2).encode('utf-8')

    with exp_col1:
        st.download_button(
            label="Download JSON Audit Report",
            data=json_bytes,
            file_name=f"{serial_input}_audit.json",
            mime="application/json"
        )

    with exp_col2:
        # Prepare annotated image bytes
        is_success, buffer = cv2.imencode(".png", annotated)
        if is_success:
            st.download_button(
                label="Download Annotated Frame (PNG)",
                data=buffer.tobytes(),
                file_name=f"{serial_input}_annotated.png",
                mime="image/png"
            )

    with exp_col3:
        if result.defects:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download Flaw Manifest (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"{serial_input}_manifest.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
