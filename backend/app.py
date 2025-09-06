# app.py
import os
import io
import json
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMARY_PATH = os.path.join(MODELS_DIR, "pipeline_summary.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_emotion_model.joblib")

# Validate
for p in [MODELS_DIR, SUMMARY_PATH, SCALER_PATH, MODEL_PATH]:
    if not os.path.exists(p):
        logging.error(f"Missing required file or folder: {p}")

# Load pipeline summary (to get target shape and label map)
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)
else:
    summary = {}

TARGET_ROWS = int(summary.get("target_rows", 62))
TARGET_COLS = int(summary.get("target_cols", 208))
LABEL_MAP = {int(k): v for k, v in summary.get("label_map", {"0":"neutral","1":"sadness","2":"fear","3":"happiness"}).items()}

# Load scaler and classifier
scaler = None
clf = None
try:
    scaler = joblib.load(SCALER_PATH)
    logging.info("Scaler loaded.")
except Exception as e:
    logging.warning(f"Could not load scaler: {e}")

try:
    clf = joblib.load(MODEL_PATH)
    logging.info("Classifier loaded.")
except Exception as e:
    logging.warning(f"Could not load classifier: {e}")

# Flask app
app = Flask(__name__)
CORS(app)  # allow all origins for dev

def label_name(idx):
    return LABEL_MAP.get(int(idx), str(idx))

def fix_shape_and_flatten(arr: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    rows, cols = arr.shape
    # trim/pad rows
    if rows >= target_rows:
        arr = arr[:target_rows, :]
    else:
        pad = np.zeros((target_rows - rows, cols), dtype=np.float32)
        arr = np.vstack([arr, pad])
    # trim/pad cols
    if arr.shape[1] >= target_cols:
        arr = arr[:, :target_cols]
    else:
        pad = np.zeros((target_rows, target_cols - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, pad])
    return arr.reshape(-1)

def load_csv_bytes_to_matrix(b: bytes) -> np.ndarray:
    bio = io.BytesIO(b)
    # Try header=None (raw numeric)
    try:
        df = pd.read_csv(bio, header=None)
    except Exception:
        bio.seek(0)
        df = pd.read_csv(bio)
        # reduce to numeric columns (if file contains metadata)
        df = df.select_dtypes(include=[np.number])
    if df is None or df.size == 0:
        raise ValueError("CSV contains no numeric data.")
    arr = df.values
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

@app.get("/health")
def health():
    return {
        "status": "ok",
        "target_rows": TARGET_ROWS,
        "target_cols": TARGET_COLS,
        "classes": LABEL_MAP,
        "model_loaded": bool(clf is not None),
        "scaler_loaded": bool(scaler is not None)
    }

@app.post("/predict")
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    filename = file.filename
    if not filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are accepted"}), 400
    try:
        content = file.read()
        mat = load_csv_bytes_to_matrix(content)
        vec = fix_shape_and_flatten(mat, TARGET_ROWS, TARGET_COLS).astype(np.float32).reshape(1, -1)
        if scaler is not None:
            vec_scaled = scaler.transform(vec)
        else:
            vec_scaled = vec
        if clf is None:
            return jsonify({"error": "Model not loaded on server"}), 500
        pred = int(clf.predict(vec_scaled)[0])
        probs = clf.predict_proba(vec_scaled)[0]
        # Build probability map according to clf.classes_ ordering
        prob_map = {}
        for lbl, p in zip(clf.classes_, probs):
            prob_map[label_name(int(lbl))] = float(p)
        return jsonify({
            "pred_label": int(pred),
            "pred_label_name": label_name(pred),
            "probabilities": prob_map
        })
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    # Run locally
    app.run(host="0.0.0.0", port=5000, debug=True)
