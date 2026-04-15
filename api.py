"""
Phase 1 — FastAPI Backend
Wraps the existing InferenceEngine in a REST API.
Run with: uvicorn api:app --reload --port 8000
"""
import io
import base64

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from MODEL.inference import InferenceEngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Brain Tumor AI — REST API",
    description="Clinical-grade brain tumor analysis powered by MCDropoutResNet + Grad-CAM",
    version="1.0.0",
)

# Allow the Vite dev server (Phase 2) and any localhost origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA / fallback
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load the model once at startup (safe — training uses a separate process)
# ---------------------------------------------------------------------------
print("[API] Loading InferenceEngine… (this may take a few seconds on first load)")
engine = InferenceEngine(weights_path="best_model.pth", config_path="config.yaml")
print("[API] Model ready. Server is live.")

# ---------------------------------------------------------------------------
# Helper — convert NumPy BGR image → base64 PNG string
# ---------------------------------------------------------------------------
def _ndarray_to_base64(img_bgr: np.ndarray) -> str:
    """
    The heatmap overlay from build_cam_overlay() is already a NumPy uint8 array
    in BGR color order (OpenCV convention). We encode it to PNG bytes and then
    base64-encode for safe JSON transport.
    """
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode heatmap image to PNG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Quick liveness probe — the React app will call this to verify the backend is up."""
    return {"status": "ok", "model": "MCDropoutResNet", "classes": engine.classes}


@app.post("/api/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Accept a single MRI image (PNG / JPG / JPEG) and return:
    - diagnosis        : predicted class name (str)
    - confidence       : softmax confidence 0–1 (float)
    - uncertainty      : Shannon entropy score (float)
    - is_anomaly       : True if entropy > threshold (bool)
    - probabilities    : per-class probability dict (dict[str, float])
    - heatmap_b64      : Grad-CAM overlay encoded as base64 PNG (str)
    """
    # --- Validate file type ---------------------------------------------------
    allowed = {"image/png", "image/jpeg", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload PNG or JPG.",
        )

    # --- Read and decode the image -------------------------------------------
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)          # HxWx3, uint8, RGB
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    # --- Run inference (uses existing InferenceEngine — no changes there) -----
    try:
        outputs = engine.predict(image_np)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # --- Build per-class probability dict ------------------------------------
    probs_array = outputs["probabilities"]          # numpy float32 array, length 4
    probabilities = {
        cls: round(float(prob), 6)
        for cls, prob in zip(engine.classes, probs_array)
    }

    # --- Encode heatmap overlay to base64 ------------------------------------
    # heatmap_overlay is a uint8 NumPy array in BGR (built by build_cam_overlay)
    heatmap_b64 = _ndarray_to_base64(outputs["heatmap_overlay"])

    return JSONResponse(content={
        "diagnosis":    outputs["diagnosis"],
        "confidence":   round(float(outputs["confidence"]), 6),
        "uncertainty":  round(float(outputs["uncertainty"]), 6),
        "is_anomaly":   bool(outputs["is_anomaly"]),
        "probabilities": probabilities,
        "heatmap_b64":  heatmap_b64,
    })
