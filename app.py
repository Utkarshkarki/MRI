import streamlit as st
from PIL import Image
import numpy as np

from MODEL.inference import InferenceEngine

st.set_page_config(page_title="Brain Tumor AI Analysis v2.0", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stApp {max-width: 1200px; margin: 0 auto;}
    h1 {color: #2c3e50;}
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Clinical-Grade Brain Tumor AI Analyzer")
st.markdown("Powered by Hybrid CNN-SVM Architecture with Native Uncertainty Quantification")

@st.cache_resource
def get_engine():
    return InferenceEngine(weights_path="best_model.pth")

engine = get_engine()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Secure MRI Upload")
    uploaded_file = st.file_uploader("Insert Patient MRI...", type=["png", "jpg", "jpeg", "dcm"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    with col1:
        st.image(image, use_container_width=True, caption="Source Evaluation")
        
    with st.spinner("Executing Parallel AI Diagnostics..."):
        outputs = engine.predict(image_np)
        
    with col2:
        st.subheader("📊 Model Diagnostics")
        
        pred_class = outputs['diagnosis']
        conf = outputs['confidence'] * 100
        unc = outputs['uncertainty']
        is_anomaly = outputs['is_anomaly']
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Class: <span style="color:{'#e74c3c' if pred_class != 'notumor' else '#27ae60'}">{pred_class.upper()}</span></h3>
            <p><strong>Baseline Confidence:</strong> {conf:.1f}%</p>
            <p><strong>Entropy Bounds (Uncertainty):</strong> {unc:.4f} <span style="color:red; font-size: 13px;">{' 🚨 CLINICAL ANOMALY TRIGGERED' if is_anomaly else ' (Stable Limits)'}</span></p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.subheader("🔥 Interpretability Heatmap (Grad-CAM)")
        st.image(outputs['heatmap_overlay'], use_container_width=True, caption="XAI Spatial Activation Visualized")
else:
    st.info("Awaiting structural image input to initiate diagnostic routing constraints.")
