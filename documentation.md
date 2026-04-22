# NeuroScan AI: Advanced Brain Tumor Diagnostic System

## Team Members
* Utkarsh 
* Sanu Bashera 

*(Note: Please update team member names if there are any additional contributors.)*

## Abstract
Brain tumors are complex, potentially life-threatening conditions that require fast and highly accurate diagnoses. Traditional analysis of Magnetic Resonance Imaging (MRI) scans can be time-consuming and subjective, underscoring the need for automated computer-aided diagnostic tools. **NeuroScan AI** provides a robust, GPU-optimized machine learning pipeline that classifies brain MRI scans into four categories: Glioma, Meningioma, Pituitary, and No Tumor. To tackle the inherent "black-box" nature of deep learning, this system incorporates Monte Carlo (MC) Dropout for uncertainty quantification and Grad-CAM for visual interpretability. The result is a clinical-grade web application that supports medical professionals by providing highly reliable, transparent AI assessments.

## Approach and Technologies Used
The project adopts a multi-phased approach transitioning from an initial prototype to a robust, production-grade system.
* **Core Machine Learning:** PyTorch, Torchvision
* **Model Backbone:** ResNet50 with custom multi-head classifiers
* **Data Processing & Augmentation:** OpenCV, Albumentations (for robust spatial/color manipulations)
* **Backend Framework:** FastAPI (REST API), Python-Multipart
* **Frontend:** Streamlit (Prototype / Rapid Testing) and React / Vite (Production)
* **Interpretability Tools:** Grad-CAM (Heatmap generation), Shannon Entropy (for uncertainty measurements)
* **Hardware Utilization:** Automatic Mixed Precision (AMP) and CUDA for efficient GPU training

## Data Collection and Preprocessing
The dataset consists of patient brain MRI scans structured into four distinct classes: `glioma`, `meningioma`, `notumor`, and `pituitary`.
* **Data Loading & Stratification:** Images are aggressively stratified using a `WeightedRandomSampler` to address any class imbalances within the dataset.
* **Preprocessing:** Images are read via OpenCV, converted from BGR to RGB, and resized to match the required input dimensions (224x224). 
* **Augmentation:** Using `Albumentations`, rigorous training transforms are applied (e.g., rotations, flips) to ensure the model does not overfit to specific MRI scanner biases. 
* **Validation Isolation:** A deep-copy isolation protocol ensures the validation set solely uses deterministic validation transformations.

## Model Architecture
The underlying model uses an **MCDropoutResNet** architecture:
* **Feature Extraction:** Relies on the proven, pre-trained `ResNet50` backbone (derived from ImageNet).
* **Latent Feature Projection:** The extracted features pass through a specialized multi-layer perceptron (flatten, robust dense layers with 1024 and 512 units, Batch Normalization, and ReLUs).
* **Monte Carlo Dropout:** Dropout layers (default rate 0.5) remain active during the *inference* phase (`enable_mc_dropout()`). By taking several stochastic forward passes, the model calculates variance and produces an uncertainty score.
* **Multi-Head Ensemble:** To ensure stable logits, the network employs a multi-head classifier configuration (default: 3 heads). The outputs from these independent classifiers are averaged to yield the final prediction.

## Training Process
Training is orchestrated by a `GPUOptimizedTrainer` class to streamline performance.
* **Optimizer & Scheduler:** Employs the `AdamW` optimizer paired with a `CosineAnnealingLR` scheduler for smooth gradient traversal.
* **Loss Function:** Utilizes a `CombinedLoss` function.
* **Mixed Precision (AMP):** Leverages `torch.amp.GradScaler` and `autocast` on CUDA devices to drastically reduce GPU memory usage (e.g., running `float16` backward passes) and increase batch throughput.
* **Fault Tolerance & Checkpointing:** The trainer natively supports "hot-resumes." Absolute structural layouts (model state, optimizer, scaler, scheduler, and epoch) are saved at `.pth` intervals to protect against spontaneous interruptions.

## Evaluation Metrics
During training and validation rounds, the system logs:
* **Epoch-wise Loss:** A running tally of average batch loss.
* **Accuracy:** Straightforward batch correctness against true labels.
* **Anomaly Detection Threshold:** Checks the probabilistic Shannon entropy boundary to flag uncertain predictions, signaling cases where human review is critical.

## Results and Discussion
By utilizing deep-feature transfer learning (ResNet) alongside rigorous augmentation protocols, the model achieves high categorical accuracy across all four tumor types. The critical inclusion of Grad-CAM enables heatmaps overlaying raw MRIs. These visual cues consistently highlight active tumor regions, fostering trust between the AI system and medical specialists. Cases with excessive MC Dropout entropy (uncertainty) are appropriately flagged, fulfilling a primary safety requirement for clinical applications.

## Deployment Information
The model inference pipeline is isolated inside an `InferenceEngine` class and bundled into a **FastAPI** REST backend. 
* A REST endpoint (`/api/predict`) receives raw MRI images via multiform-data POST requests.
* The API returns precise JSON telemetry containing: Predicted Diagnosis, Softmax Confidence, Uncertainty Entropy, Anomaly Flags, an array of Probabilities, and a Base64-encoded Grad-CAM Heatmap.
* The frontend (React / Streamlit) seamlessly parses this API JSON telemetry to provide a sleek, dark-mode user interface.

## Instructions for Running the Project

**1. Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. To Train the Model:**
Ensure your dataset is stored inside the `Dataset` folder. If checking out from a crash, the checkpoint will automatically load.
```bash
python -m MODEL.train
```

**3. To Run the FastAPI Backend:**
```bash
uvicorn api:app --reload --port 8000
```
*Tip: Visit `http://localhost:8000/docs` to view the interactive Swagger API interface.*

**4. To Run the Streamlit Interface:**
```bash
streamlit run app.py
```

## Relevant Code Snippets

**Activating Monte Carlo Dropout during Inference**
```python
def enable_mc_dropout(self):
    for m in self.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train() # Force dropout to remain active even during evaluation
```

**Graceful GPU-to-CPU Checkpoint Resumption (`train.py`)**
```python
# Handles empty scaler dict when resuming from a CPU save state
if os.path.exists('latest_checkpoint.pth'):
    checkpoint = torch.load('latest_checkpoint.pth', map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    if checkpoint.get('scaler_state_dict'):
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    self.start_epoch = checkpoint['epoch'] + 1
```

## Conclusion
The **NeuroScan AI** project goes beyond standard ML classification by implementing crucial failure detection protocols (Uncertainty Entropy) and interpretability functionality (Grad-CAM heatmaps). Through a tiered architecture spanning an optimized PyTorch training harness, a robust FastAPI backend, and dynamic frontend components, the system successfully bridges the gap between deep learning research and practical clinical deployment.
