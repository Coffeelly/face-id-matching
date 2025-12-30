# app/main.py
import streamlit as st
import torch
import sys
import os
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.face_engine import FaceVerifier

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Face ID - KYC",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .success-badge { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .error-badge { color: #dc3545; font-weight: bold; font-size: 1.2em; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD BOTH MODELS ---
@st.cache_resource
def load_models():
    std_engine = FaceVerifier(use_quantized=False)
    
    q8_engine = FaceVerifier(use_quantized=True)
    
    return std_engine, q8_engine

try:
    with st.spinner("Initializing AI Engines"):
        std_engine, q8_engine = load_models()
except Exception as e:
    st.error(f"Failed to load models.Error: {e}")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Console")

# Model Selector
model_choice = st.sidebar.radio(
    "Inference Model",
    ("Standard (FP32)", "Quantized (INT8)"),
    index=0
)
active_engine = q8_engine if "INT8" in model_choice else std_engine

# Threshold
threshold = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.60, 0.01)

# Debug Info
st.sidebar.markdown("---")
st.sidebar.caption(f"Current Model: **{model_choice}**")
st.sidebar.caption(f"Device: **{active_engine.device}**")
if "INT8" in model_choice:
    st.sidebar.success("Optimized Mode Active")
else:
    st.sidebar.warning("Standard Mode Active")

# --- MAIN INTERFACE ---
st.title("FaceID Verification Suite")
st.markdown("### Digital Identity & Liveness Detection Prototype")

col1, col2 = st.columns(2)
with col1:
    st.info("**Reference ID (Database)**")
    id_file = st.file_uploader("Upload ID Card", type=['jpg', 'png', 'jpeg'], key="id")
    if id_file:
        st.image(id_file, use_container_width=True)

with col2:
    st.info("**Live Selfie (User)**")
    selfie_file = st.file_uploader("Upload Selfie", type=['jpg', 'png', 'jpeg'], key="selfie")
    if selfie_file:
        st.image(selfie_file, use_container_width=True)

# --- EXECUTION LOGIC ---
if id_file and selfie_file:
    st.markdown("---")
    
    if st.button("Run Verification", type="primary", use_container_width=True):
        
        # Detect & Align
        with st.status("Running Computer Vision Pipeline...", expanded=True) as status:
            st.write("Detecting faces with MTCNN...")
            start_det = time.time()
            id_face = active_engine.process_image(id_file)
            selfie_face = active_engine.process_image(selfie_file)
            det_time = (time.time() - start_det) * 1000

            if id_face is None or selfie_face is None:
                status.update(label="Face Detection Failed", state="error")
                st.error("Could not detect faces. Please try clearer photos.")
            else:
                st.write(f"✅ Faces Detected in {det_time:.1f}ms")
                
                # Extract & Compare
                st.write(f"🧠 Generating Embeddings using {model_choice}...")
                score, latency = active_engine.verify(id_face, selfie_face)
                status.update(label="Verification Complete", state="complete")

                # --- RESULTS DASHBOARD ---
                st.markdown("### 📊 Verification Results")

                res_col1, res_col2, res_col3 = st.columns(3)
                
                res_col1.metric("Similarity Score", f"{score:.4f}", delta_color="off")
                
                lat_color = "normal" if latency < 100 else "off"
                res_col2.metric("Inference Latency", f"{latency:.2f} ms", f"- {(100-latency):.1f}ms target", delta_color=lat_color)
                
                # Decision
                is_match = score > threshold
                if is_match:
                    res_col3.markdown('<p class="success-badge">MATCH VERIFIED</p>', unsafe_allow_html=True)
                else:
                    res_col3.markdown('<p class="error-badge">MATCH REJECTED</p>', unsafe_allow_html=True)

                # Technical Details
                with st.expander("Show Technical Details (Unit Test Data)"):
                    st.json({
                        "model_type": "INT8 Quantized" if "INT8" in model_choice else "FP32 Standard",
                        "threshold_set": threshold,
                        "raw_similarity": score,
                        "inference_time_ms": latency,
                        "input_tensor_shape": str(list(id_face.shape))
                    })