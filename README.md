<div align="center">
  <h1>🧠 Robust Brain Tumor AI Analysis System</h1>
  <p><strong>A production-grade, end-to-end medical imaging classification and visualization framework.</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://reactjs.org/)
  [![Vite](https://img.shields.io/badge/Vite-5.0+-646cff.svg)](https://vitejs.dev/)
</div>

<br />

## 📖 Overview

Brain tumor detection is a pivotal area in medical diagnostics where early and accurate detection substantially improves survival rates. However, real-world clinical implementation faces barriers such as dataset class imbalance, the "black-box" nature of advanced AI models, and a lack of uncertainty metrics in point-prediction models.

This system addresses these challenges directly. It provides an enterprise-ready **React / FastAPI** stack built on top of a highly robust **PyTorch** backbone.

Inspired by systematic literature reviews (Alam et al., 2024), this application features **Explainable AI (XAI)** localization via Grad-CAM, and **Confidence/Uncertainty Evaluation** by utilizing Monte Carlo (MC) Dropout.

---

## ✨ Key Technical Features

### 🧩 1. Deep Learning Architecture
* **MCDropoutResNet50:** A modified ResNet-50 backbone specifically optimized for medical imaging feature extraction.
* **Ensemble Heads:** Implements 3 parallel classification heads, averaging logits to improve stability across batches.
* **Focal Loss:** Natively combats class imbalance (common in rare tumor phenotypes) without relying exclusively on synthetic dataset explosions.

### 🔍 2. Explainable AI (XAI)
* **Grad-CAM (Gradient-weighted Class Activation Mapping):** Converts complex latent matrices into biologically actionable heatmaps. By overlaying these spatial activations onto the original MRI, clinicians can visually verify which anatomical regions drove the model's pathology prediction.

### 🧪 3. Uncertainty Quantification (Anomaly Detection)
* **Monte Carlo Dropout:** Runs $N=15$ stochastic forward passes at inference time, utilizing dropout as a Bayesian approximation.
* **Shannon Entropy:** Calculates variance across the predictions. If the entropy exceeds a predefined threshold, the system triggers a **Clinical Anomaly Alert**, aggressively flagging out-of-distribution, noisy, or edge-case images that require human radiologist intervention.

### 💻 4. Production Web Stack
* **FastAPI Backend:** A heavily-typed, asynchronous, RESTful Python API exposing the PyTorch inference models (`/api/predict`).
* **React + Vite Frontend:** An ultra-fast, premium medical UI featuring drag-and-drop intake, scan-line animations, live probability charts, and clinical urgency routing.

---

## 🛠 Project Structure

```text
MRI/
├── MODEL/                    # Core Deep Learning logic
│   ├── dataset.py            # Custom PyTorch Dataset & Stratified Loader
│   ├── explainability.py     # Grad-CAM heatmap generation mechanics
│   ├── inference.py          # Unified InferenceEngine pipeline
│   ├── loss.py               # Imbalanced FocalLoss formulations
│   ├── models.py             # MCDropoutResNet50 Architecture
│   ├── train.py              # Automated GPU-accelerated training orchestrator
│   ├── transforms.py         # Albumentations MRI augmentations
│   └── uncertainty.py        # Entropy and bounds estimations
│
├── frontend/                 # React/Vite SPA (Single Page Application)
│   ├── src/
│   │   ├── components/       # UI Components (Hero, Uploader, GradCAMViewer, etc.)
│   │   ├── App.jsx           # Main React coordinator
│   │   └── index.css         # Clinical dark-mode design system tokens
│   └── index.html            # Vite entrypoint with SEO tracking
│
├── api.py                    # FastAPI application layer
├── app.py                    # Legacy Streamlit prototype UI
├── config.yaml               # Engine thresholds, epochs, and hyperparams
└── requirements.txt          # Shared Python dependencies
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Utkarshkarki/MRI.git
cd MRI
```

### 2. Configure Backend Engine (Python / Conda / Venv)
Ensure you have Python 3.9+ and CUDA capability (if processing on GPU).
```bash
pip install -r requirements.txt
```

### 3. Data Preparation
Place your structured MRI scans inside the `Dataset` directory split by subclass:
```text
Dataset/
 ├── glioma/
 ├── meningioma/
 ├── notumor/
 └── pituitary/
```

### 4. Configure Frontend (Node.js)
Ensure you have Node 18+ installed.
```bash
cd frontend
npm install
```

---

## 💻 Usage & Deployment

### Phase A: Model Training
If you need to retrain or fine-tune the pipeline on a new dataset, run the training orchestrator. *Note: Training automatically utilizes Automatic Mixed Precision (AMP) and creates checkpoint resumption targets.*
```bash
python -m MODEL.train
```

> **⚠️ CRITICAL: Weights in RAM**
> If your FastAPI backend is currently running while `train.py` finishes a new epoch and overwrites `best_model.pth`, **the backend will not automatically know.** The API loads weights into memory strictly at startup for performance. You must manually stop the API (`Ctrl+C`) and restart it to load your newly trained model.

### Phase B: Detailed Model Evaluation
If you want to view the raw mathematical performance (Sensitivity, Specificity, F1-Score) of your trained weights against the unseen validation set, run the evaluation script:
```bash
python evaluate.py
```
*(Note: You must have a complete `best_model.pth` saved in the root directory for this to work).*

### Phase C: Local Web Server Sequence
You need to launch **both** the backend API and the frontend UI concurrently in separate terminal windows.

**Terminal 1 — Launch the FastAPI Backend:**
```bash
uvicorn api:app --port 8000 --reload
```
*(Backend documentation and interactive Swagger UI available at http://127.0.0.1:8000/docs)*

**Terminal 2 — Launch the React Frontend:**
```bash
cd frontend
npm run dev
```
*(Navigate to http://localhost:5173 to access the diagnostic dashboard)*

---

## ⚕️ Regulatory & Medical Disclaimer
**This project is intended strictly for academic, research, and non-commercial decision-support purposes.** The output interpretations, Grad-CAM overlays, and Uncertainty thresholds do not constitute official medical diagnoses. Clinical treatment plans must always be corroborated by qualified human neurological and radiological staff.

---

### References
* Alam, M. A., Sohel, A., Hasan, K. M., & Ahmad, I. (2024). *Advancing Brain Tumor Detection Using Machine Learning and Artificial Intelligence: A Systematic Literature Review of Predictive Models and Diagnostic Accuracy.* Strategic Data Management and Innovation, 1(1), 37-55.
