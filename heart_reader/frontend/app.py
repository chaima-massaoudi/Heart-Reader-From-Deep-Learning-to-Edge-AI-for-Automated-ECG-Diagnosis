"""
Heart Reader Challenge — FastAPI Web Application.

Serves a web dashboard for interactive ECG classification:
  - Upload a 12-lead ECG (CSV) or pick a random PTB-XL sample
  - Visualize all 12 leads
  - Get multi-label diagnosis predictions with confidence scores
  - View model architecture info & performance metrics

Run:
    cd heart_reader
    uvicorn frontend.app:app --reload --port 8000
"""

import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Ensure heart_reader is importable ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # heart_reader/
sys.path.insert(0, str(ROOT))

from data.preprocessing import SUPERCLASSES, load_ptbxl_database, compute_superdiagnostic_labels
from models.fusion_model import FusionModel

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Heart Reader", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML/CSS/JS)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Globals (loaded once) ────────────────────────────────────────────────────
MODEL = None
SCALER_SIGNAL = None
SCALER_FEATURES = None
DEVICE = "cpu"  # always CPU for the web app
CONFIG = None
TEST_SIGNALS = None       # (N, 1000, 12)
TEST_FEATURES = None      # (N, F)
TEST_LABELS = None        # (N, 5)
TEST_DF = None
NUM_FEATURES = 0
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

CLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG — no major abnormalities detected.",
    "MI":   "Myocardial Infarction — signs of heart muscle damage from blocked blood supply.",
    "STTC": "ST/T Change — abnormalities in the ST segment or T wave, may indicate ischemia.",
    "CD":   "Conduction Disturbance — abnormal electrical conduction (e.g., bundle branch block).",
    "HYP":  "Hypertrophy — enlarged heart muscle, often from chronic high blood pressure.",
}


def _load_config():
    """Load the YAML config."""
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict) -> FusionModel:
    """Load the best checkpoint."""
    model_cfg = cfg["model"]

    # Try inception1d first (our best single model), else xresnet
    for backbone in ["inception1d", "xresnet1d101", "se_resnet1d"]:
        ckpt_path = ROOT / "checkpoints" / f"fusion_{backbone}_best.pt"
        if ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

            # Detect num_features from checkpoint
            nf = 0
            for k in ckpt.get("model_state_dict", ckpt).keys():
                if "feature_branch" in k:
                    nf = 1313  # PTB-XL+ default
                    break

            model = FusionModel(
                backbone_name=backbone,
                model_cfg=model_cfg,
                num_classes=model_cfg.get("num_classes", 5),
                input_channels=model_cfg.get("input_channels", 12),
                num_features=nf,
            )
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()
            return model, nf, backbone

    raise FileNotFoundError("No checkpoint found in checkpoints/")


def _load_scalers():
    """Load signal and feature scalers if they exist."""
    sig_scaler = None
    feat_scaler = None

    sig_path = ROOT / "cache" / "signal_scaler.pkl"
    feat_path = ROOT / "cache" / "feature_scaler.pkl"

    if sig_path.exists():
        with open(sig_path, "rb") as f:
            sig_scaler = pickle.load(f)
        print("Loaded signal scaler")

    if feat_path.exists():
        with open(feat_path, "rb") as f:
            feat_scaler = pickle.load(f)
        print("Loaded feature scaler")

    return sig_scaler, feat_scaler


def _load_test_data(cfg: dict):
    """Pre-load test fold for quick random-sample demos."""
    data_cfg = cfg["data"]
    ptbxl_path = str(ROOT / data_cfg["ptbxl_path"])

    cache_test = ROOT / "cache" / "test_cache.npz"
    if cache_test.exists():
        print("Loading cached test data...")
        d = np.load(str(cache_test), allow_pickle=True)
        return d["signals"], d.get("features", None), d["labels"], None

    try:
        df, signals = load_ptbxl_database(ptbxl_path, data_cfg.get("sampling_rate", 100))
        df = compute_superdiagnostic_labels(df, os.path.join(ptbxl_path, "scp_statements.csv"))

        # Filter to test fold
        test_mask = df["strat_fold"] == data_cfg.get("test_fold", 10)
        test_df = df[test_mask]
        test_signals = signals[test_mask.values]

        # Encode labels
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
        test_labels = mlb.fit_transform(test_df["superdiagnostic"].values).astype(np.float32)

        # Try loading features
        test_features = None
        if data_cfg.get("use_ptbxl_plus_features", False):
            try:
                from data.preprocessing import load_ptbxl_plus_features
                ptbxl_plus_path = str(ROOT / data_cfg["ptbxl_plus_path"])
                test_features, _ = load_ptbxl_plus_features(ptbxl_plus_path, test_df.index.values)
            except Exception as e:
                print(f"Could not load PTB-XL+ features: {e}")

        # Cache for next startup
        save_dict = {"signals": test_signals, "labels": test_labels}
        if test_features is not None:
            save_dict["features"] = test_features
        os.makedirs(str(ROOT / "cache"), exist_ok=True)
        np.savez_compressed(str(cache_test), **save_dict)

        return test_signals, test_features, test_labels, test_df

    except Exception as e:
        print(f"Warning: Could not load test data: {e}")
        return None, None, None, None


@app.on_event("startup")
def startup():
    """Initialize model and data on startup."""
    global MODEL, SCALER_SIGNAL, SCALER_FEATURES, CONFIG
    global TEST_SIGNALS, TEST_FEATURES, TEST_LABELS, TEST_DF, NUM_FEATURES

    print("=" * 60)
    print("Heart Reader — Starting Web Application")
    print("=" * 60)

    CONFIG = _load_config()

    # Load model
    MODEL, NUM_FEATURES, backbone_name = _load_model(CONFIG)
    print(f"Model loaded: FusionModel({backbone_name}), features={NUM_FEATURES}")

    # Load scalers
    SCALER_SIGNAL, SCALER_FEATURES = _load_scalers()

    # Load test data for random sample feature
    TEST_SIGNALS, TEST_FEATURES, TEST_LABELS, TEST_DF = _load_test_data(CONFIG)
    if TEST_SIGNALS is not None:
        print(f"Test data loaded: {TEST_SIGNALS.shape[0]} samples")
    else:
        print("No test data available (upload-only mode)")

    print("=" * 60)
    print("Ready!  Open http://localhost:8000 in your browser")
    print("=" * 60)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the main dashboard page."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "test_samples_available": TEST_SIGNALS.shape[0] if TEST_SIGNALS is not None else 0,
        "classes": SUPERCLASSES,
    }


@app.get("/api/model-info")
def model_info():
    """Return model architecture info and performance summary."""
    info = {
        "classes": SUPERCLASSES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "num_parameters": sum(p.numel() for p in MODEL.parameters()),
        "num_features": NUM_FEATURES,
        "input_shape": {"leads": 12, "samples": 1000, "duration_sec": 10, "sampling_rate": 100},
        "lead_names": LEAD_NAMES,
    }

    # Load evaluation results if available
    report_path = ROOT / "results" / "evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        info["performance"] = report

    return info


@app.get("/api/random-sample")
def random_sample():
    """Return a random ECG sample from the test set."""
    if TEST_SIGNALS is None:
        raise HTTPException(status_code=404, detail="No test data loaded")

    idx = np.random.randint(0, len(TEST_SIGNALS))
    signal = TEST_SIGNALS[idx]          # (1000, 12)
    labels = TEST_LABELS[idx]            # (5,)

    # Get features if available
    features = None
    if TEST_FEATURES is not None and idx < len(TEST_FEATURES):
        features = TEST_FEATURES[idx]

    # Run prediction
    pred = _predict(signal, features)

    # Ground truth
    gt_classes = [SUPERCLASSES[i] for i, v in enumerate(labels) if v > 0.5]

    return {
        "sample_index": int(idx),
        "signal": signal.tolist(),         # (1000, 12)
        "ground_truth": gt_classes,
        "predictions": pred,
        "lead_names": LEAD_NAMES,
    }


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    """Predict from an uploaded ECG CSV file.

    Expected CSV format:
      - 12 columns (one per lead), 1000 rows (10s at 100Hz) → signal-only mode
      - OR >12 columns: first 12 = leads, remaining = PTB-XL+ features (row 0 only)
    """
    try:
        contents = await file.read()
        text = contents.decode("utf-8").strip()

        # ── Robust CSV parsing: detect header vs. headerless ────────
        first_line = text.split("\n")[0].strip()
        has_header = False
        try:
            [float(x.strip()) for x in first_line.split(",")]
        except ValueError:
            has_header = True          # first row contains non-numeric values

        if has_header:
            df = pd.read_csv(io.StringIO(text))       # use row 0 as header
        else:
            df = pd.read_csv(io.StringIO(text), header=None)

        # Coerce everything to numeric; drop fully-empty columns
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")

        ncols = df.shape[1]

        # ── Extract signal and optional features ────────────────────
        features = None
        if ncols > 12:
            # First 12 columns = signal leads, rest = structured features
            signal = df.iloc[:, :12].values.astype(np.float32)
            feat_cols = df.iloc[:, 12:]
            # Features: take first row (they're repeated or only in row 0)
            features = feat_cols.iloc[0].values.astype(np.float32)
            # Replace any NaN with 0
            features = np.nan_to_num(features, nan=0.0)
        else:
            signal = df.values.astype(np.float32)

        # Validate shape
        if signal.shape[1] != 12:
            # Try transpose
            if signal.shape[0] == 12:
                signal = signal.T
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected 12 columns (leads), got shape {signal.shape}"
                )

        # Pad or truncate to 1000 samples
        if signal.shape[0] < 1000:
            pad = np.zeros((1000 - signal.shape[0], 12), dtype=np.float32)
            signal = np.vstack([signal, pad])
        elif signal.shape[0] > 1000:
            signal = signal[:1000]

        pred = _predict(signal, features=features)

        return {
            "filename": file.filename,
            "signal_shape": list(signal.shape),
            "signal": signal.tolist(),
            "predictions": pred,
            "lead_names": LEAD_NAMES,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


def _predict(signal: np.ndarray, features: np.ndarray = None) -> dict:
    """Run inference on a single ECG sample.

    Args:
        signal: (1000, 12) numpy array.
        features: (F,) structured features or None.

    Returns:
        Dict with per-class probabilities and predicted labels.
    """
    with torch.no_grad():
        # Prepare signal: (1, 12, 1000)
        sig = signal.copy()

        # Apply scaler if available
        if SCALER_SIGNAL is not None:
            sig = SCALER_SIGNAL.transform(sig.reshape(-1, 12)).reshape(sig.shape)

        sig_tensor = torch.tensor(sig.T[np.newaxis], dtype=torch.float32)  # (1, 12, 1000)

        # Prepare features
        feat_tensor = None
        if NUM_FEATURES > 0:
            if features is not None:
                feat = features.copy()
                if SCALER_FEATURES is not None:
                    feat = SCALER_FEATURES.transform(feat.reshape(1, -1)).flatten()
                feat_tensor = torch.tensor(feat[np.newaxis], dtype=torch.float32)
            else:
                # No features provided (file upload mode).
                # Use the scaler mean so after scaling -> zeros (= "average patient").
                # This is much more natural than raw zeros which map to extreme values.
                if SCALER_FEATURES is not None and hasattr(SCALER_FEATURES, "mean_"):
                    mean_feat = SCALER_FEATURES.mean_.copy().astype(np.float32)
                    scaled = SCALER_FEATURES.transform(mean_feat.reshape(1, -1)).flatten()
                    feat_tensor = torch.tensor(scaled[np.newaxis], dtype=torch.float32)
                else:
                    feat_tensor = torch.zeros(1, NUM_FEATURES, dtype=torch.float32)

        # Forward pass
        if feat_tensor is not None:
            logits = MODEL(sig_tensor, feat_tensor)
        else:
            logits = MODEL(sig_tensor)

        probs = torch.sigmoid(logits).squeeze().numpy()

    # Build result
    result = {}
    for i, cls_name in enumerate(SUPERCLASSES):
        p = float(probs[i])
        result[cls_name] = {
            "probability": round(p, 4),
            "predicted": p > 0.5,
            "description": CLASS_DESCRIPTIONS[cls_name],
        }

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("frontend.app:app", host="0.0.0.0", port=8000, reload=True)
