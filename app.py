import streamlit as st
import cv2
import numpy as np
from PIL import Image
from MODEL.inference import InferenceEngine

# Page Config
st.set_page_config(page_title="Brain Tumor AI Analysis", page_icon="🧠", layout="wide")

# Theme Setup & Styling
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

st.title("🧠 Robust Brain Tumor AI Analysis")
st.markdown("Upload a brain MRI scan to perform classification with uncertainty estimation and Grad-CAM interpretability.")

import torch

# Initialize Inference Engine
@st.cache_resource
def load_engine():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return InferenceEngine(model_path='best_model.pth', device=device)

engine = load_engine()

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    mc_passes = st.slider("MC Dropout Passes", min_value=10, max_value=50, value=30, 
                          help="Number of stochastic forward passes for uncertainty estimation.")
    st.info("Higher passes yield better uncertainty estimates but take longer.")
    
# Main UI
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Input")
        st.image(image, use_container_width=True, caption="Uploaded Scan")
        
        with st.spinner("Analyzing scan..."):
            # Run inference
            results = engine.predict_with_uncertainty(image_np, num_passes=mc_passes)
            
    with col2:
        st.subheader("Model Insights")
        
        # Display Metrics
        pred_class = results['predicted_class']
        conf = float(results['confidence']) * 100
        unc = float(results['uncertainty']) * 100
        
        # Anomaly detection via high uncertainty
        is_anomaly = unc > 15.0 # threshold can be tuned
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Prediction: <span style="color:{'#e74c3c' if pred_class != 'notumor' else '#27ae60'}">{pred_class.upper()}</span></h3>
            <p><strong>Confidence:</strong> {conf:.1f}%</p>
            <p><strong>Uncertainty:</strong> {unc:.1f}% <span style="color:red; font-size: 12px;">{' (High Uncertainty: Possible Anomaly)' if is_anomaly else ''}</span></p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.subheader("Grad-CAM Interpretability")
        st.image(results['cam_overlay'], use_container_width=True, caption="Regions driving the prediction")

else:
    st.info("Please upload an image to begin analysis.")
