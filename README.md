# Robust Brain Tumor AI Analysis System 🧠

An end-to-end medical imaging classification and visualization framework to detect brain tumors securely, predictably, and interpretably using PyTorch and Streamlit.

## Overview

Brain tumor detection is a pivotal area in medical diagnostics. Early and accurate detection substantially improves survival rates. However, real-world clinical implementation faces barriers such as data scarcity, imbalance, and the "black-box" nature of advanced AI models. 

This repository provides an end-to-end Artificial Intelligence system designed to address these challenges, inspired by systematic literature reviews emphasizing the need for **Explainable AI (XAI)**, **Robust Data Augmentations**, and **Confidence/Uncertainty Evaluation** (Alam et al., 2024). 

Our system leverages a robust deep learning backbone combined with **Grad-CAM (Gradient-weighted Class Activation Mapping)** for visual interpretability and **Monte Carlo (MC) Dropout** for out-of-distribution anomaly and uncertainty detection, packaged into a highly accessible Streamlit application.

## Key Features

1. **State-of-the-Art Deep Learning**: Utilizes a Convolutional Neural Network (ResNet-50) backbone modified for diagnostic robustness, achieving high-accuracy feature extraction.
2. **Explainable AI (XAI) Integration**: Deploys Grad-CAM to highlight precise regions of the brain that influenced the model's prediction. This transparency aims to foster clinician trust and improve diagnostic usability.
3. **Monte Carlo Uncertainty Estimation**: Employs MC Dropout to output a prediction confidence band rather than a standard single-point probability. This flags highly uncertain scans, effectively acting as an anomaly detector for unseen or out-of-distribution images.
4. **Resilient Data Augmentation**: Addresses dataset heterogeneity and limited annotations using the `Albumentations` library, applying contrast enhancements (CLAHE) and spatial variance mimicking MRI scanner disparities.
5. **Real-time Diagnostic UI**: A fast, interactive Streamlit frontend built for immediate clinical upload and review, reducing diagnostic turnaround time.

## Architecture & Scientific Context

The architecture choices made in this repository map strictly to advancing trends in AI-assisted Neuro-Oncology:
- **CNN Backbones**: Studies show that Convolutional architecture consistency achieves >90% precision for brain tumor types and effectively maps intricate patterns (Alam et al., 2024).
- **Focal Loss**: To combat class imbalance natively during training without synthetic dataset explosions.
- **XAI (Explainability)**: By overlaying heatmaps (Grad-CAM), we convert a complex mathematical matrix into a biologically actionable localization metric, aiding surgical planning and reducing diagnosis time. 

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Utkarshkarki/MRI.git
   cd MRI
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   Place your MRI scans inside the `Dataset` folder split by subclass (e.g., `Dataset/glioma`, `Dataset/meningioma`, `Dataset/notumor`, `Dataset/pituitary`).

## Usage

### 1. Training the Model
To train the CNN model with Focal Loss on your dataset and generate the optimal weights (`best_model.pth`):
```bash
python -m MODEL.train
```

### 2. Launching the Clinical Interface
To start the real-time Streamlit diagnostic system:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`) empowering you to upload unseen MRI scans and view the resulting predictions, confidence levels, and Grad-CAM insights interactively.

---

### References
* Alam, M. A., Sohel, A., Hasan, K. M., & Ahmad, I. (2024). *Advancing Brain Tumor Detection Using Machine Learning and Artificial Intelligence: A Systematic Literature Review of Predictive Models and Diagnostic Accuracy.* Strategic Data Management and Innovation, 1(1), 37-55.
