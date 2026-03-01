<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.134-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Research-blue" />
</p>

<h1 align="center">Heart Reader</h1>
<h3 align="center">From Deep Learning to Edge AI for Automated 12-Lead ECG Diagnosis</h3>

<p align="center">
  <strong>Chaima Massaoudi</strong> &mdash; March 2026
</p>

<p align="center">
  A multimodal deep learning system for multi-label ECG classification targeting five diagnostic superclasses from 12-lead electrocardiogram recordings, with a complete edge deployment pipeline and interactive web dashboard.
</p>

---

## Highlights

| Metric | Score |
|--------|-------|
| **Macro-AUC (Test)** | **0.9268** |
| **Val Macro-AUC** | **0.9315** |
| **Macro F1 (Test)** | **0.7239** |
| **F-max** | **0.7574** |
| **Model Params** | 977K |
| **Model Size** | 3.79 MB (FP32) / 2.37 MB (INT8) |

<details>
<summary><strong>Per-Class AUC (Test Set — Fold 10)</strong></summary>

| Class | AUC | Description |
|-------|-----|-------------|
| NORM | 0.957 | Normal ECG |
| MI | 0.927 | Myocardial Infarction |
| STTC | 0.930 | ST/T Change |
| CD | 0.913 | Conduction Disturbance |
| HYP | 0.908 | Hypertrophy |

</details>

---

## Demo

<p align="center">
  <img src="heart_reader/demo.gif" alt="Heart Reader Dashboard Demo" width="720" />
</p>

<p align="center"><em>Interactive dashboard — 12-lead ECG visualization, real-time diagnosis, and confidence scores (5× speed)</em></p>

> Full-length video: [`heart_reader/demo.mp4`](heart_reader/demo.mp4)

---

## Architecture Overview

```
 ┌──────────────────────────┐     ┌──────────────────────────┐
 │    12-Lead ECG Signal    │     │   PTB-XL+ Features       │
 │    (1000 × 12)           │     │   (1313-dim vector)      │
 └────────────┬─────────────┘     └────────────┬─────────────┘
              │                                 │
     InceptionTime1D                     Feature MLP
     + SE Channel Attention         [1313 → 256 → 128 → 64]
     (6 blocks, multi-scale)                    │
              │                            64-dim embed
     AdaptiveConcatPool1d                       │
     + Linear → 256-dim                         │
              │                                 │
              └──────────┬──────────────────────┘
                         │
                   Concat [256 ‖ 64] = 320
                         │
                   Fusion MLP Head
                   [320 → 128 → 5]
                         │
                   5-class sigmoid
```

**Training Enhancements:** Focal loss, label smoothing (0.05), mixup augmentation (α = 0.3), Stochastic Weight Averaging (SWA from epoch 40).

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/chaima-massaoudi/Heart-Reader-From-Deep-Learning-to-Edge-AI-for-Automated-ECG-Diagnosis.git
cd Heart-Reader-From-Deep-Learning-to-Edge-AI-for-Automated-ECG-Diagnosis

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate         # Linux / macOS

pip install -r heart_reader/requirements.txt
```

### 2. Dataset Setup

Download the following from [PhysioNet](https://physionet.org):

| Dataset | Version | Path (relative to repo root) |
|---------|---------|------------------------------|
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 1.0.3 | `data/ptbxl/` |
| [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/) | 1.0.1 | `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/` |

### 3. Train

```bash
cd heart_reader

# Full training with improved config (focal loss, mixup, SWA)
python train.py --config configs/improved.yaml --backbone inception1d

# Default config training
python train.py
```

### 4. Evaluate

```bash
python run_evaluation.py
```

### 5. Launch Web Dashboard

```bash
python run_server.py
# Open http://localhost:8000
```

### 6. Edge Export

```bash
python export.py --backbone inception1d
```

---

## Web Dashboard

An interactive FastAPI-powered dashboard for real-time ECG diagnosis:

- **Random Sample**: Load a random test set ECG with full-model predictions
- **File Upload**: Upload your own 12-lead CSV for instant analysis
- **12-Lead ECG Viewer**: Full waveform visualization across all leads
- **Probability Chart**: Interactive bar chart with confidence scores
- **Diagnosis Panel**: Clear Normal/Abnormal classification with detailed class breakdown
- **Graph Export**: Save any visualization as PNG with one click

### Test Files

Pre-selected test CSVs (one per class) are included in `heart_reader/test_files/`:

| File | Expected Diagnosis | Confidence |
|------|--------------------|------------|
| `sample_norm.csv` | NORM | 1.000 |
| `sample_mi.csv` | MI | 0.872 |
| `sample_sttc.csv` | STTC | 0.674 |
| `sample_cd.csv` | CD | 0.962 |
| `sample_hyp.csv` | HYP | 0.710 |

---

## Project Structure

```
heart_reader/
├── configs/
│   ├── default.yaml              # Base configuration
│   └── improved.yaml             # Enhanced training (focal + mixup + SWA)
├── data/
│   ├── preprocessing.py          # PTB-XL loading, label mapping, PTB-XL+ features
│   ├── dataset.py                # PyTorch Dataset & DataLoader
│   └── augmentation.py           # ECG-specific augmentations
├── models/
│   ├── inception1d.py            # InceptionTime1D + SE attention
│   ├── xresnet1d.py              # XResNet1d family (18–152 layers)
│   ├── se_resnet1d.py            # SE-ResNet1d (Wang-style)
│   ├── fusion_model.py           # Multimodal signal + feature fusion
│   ├── feature_branch.py         # MLP for structured features
│   ├── ensemble.py               # Weighted ensemble (Nelder-Mead)
│   └── heads.py                  # Pooling, SE blocks, weight init
├── training/
│   ├── trainer.py                # Training loop (AMP, SWA, early stopping)
│   ├── losses.py                 # Focal loss, label smoothing BCE, mixup
│   ├── metrics.py                # AUC, Fmax, G_beta, AUPRC
│   └── callbacks.py              # LR scheduling callbacks
├── evaluation/
│   ├── evaluate.py               # Bootstrap evaluation, CI, CSV export
│   └── visualization.py          # ROC curves, confusion matrix, plots
├── edge/
│   ├── prune.py                  # L1 structured pruning
│   ├── quantize.py               # INT8 dynamic quantization
│   ├── export_tflite.py          # ONNX export
│   └── benchmark.py              # Inference benchmarking
├── frontend/
│   ├── app.py                    # FastAPI backend (REST API)
│   └── static/                   # Dashboard (HTML, CSS, JS)
├── results/
│   ├── evaluation_report.json    # Full metrics with bootstrap CIs
│   └── edge/edge_stats.json      # Compression & latency stats
├── test_files/                   # Pre-selected test CSVs per class
├── train.py                      # Main entry point
├── evaluate.py                   # Standalone evaluation
├── export.py                     # Edge deployment pipeline
├── run_server.py                 # Web dashboard launcher
├── demo.mp4                      # Video demonstration
├── REPORT.md                     # Detailed technical report
└── requirements.txt              # Python dependencies
```

---

## Edge Deployment

| Variant | Size | Compression | Format |
|---------|------|-------------|--------|
| Full Fusion (FP32) | 3.79 MB | 1.0× | PyTorch |
| Quantized (INT8) | 2.37 MB | 1.6× | PyTorch |
| Signal-Only ONNX | 2.16 MB | 1.75× | ONNX |

Pipeline: **Structured Pruning → INT8 Quantization → ONNX Export**

---

## Training Techniques

| Technique | Configuration | Purpose |
|-----------|---------------|---------|
| Focal Loss | γ = 2, α = 0.25 | Class imbalance handling |
| Label Smoothing | ε = 0.05 | Calibration improvement |
| Mixup | α = 0.3 | Regularization |
| SWA | From epoch 40 | Flat minima, generalization |
| OneCycleLR | max_lr = 0.003 | Fast convergence |
| AdamW | wd = 0.01 | Weight regularization |
| AMP (FP16) | Automatic | Memory/speed optimization |
| Gradient Clipping | max_norm = 1.0 | Training stability |

---

## Data Protocol

Following the official PTB-XL benchmark (Strodthoff et al., 2021):

| Split | Folds | Samples | Purpose |
|-------|-------|---------|---------|
| Train | 1–8 | 16,289 | Model training |
| Val | 9 | 2,034 | Hyperparameter tuning, thresholds |
| Test | 10 | 2,050 | Final evaluation only |

---

## References

1. Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7 (2020).
2. Strodthoff, N., et al. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE JBHI* 25.5 (2021).
3. Strodthoff, N., et al. "PTB-XL+, a comprehensive electrocardiographic feature dataset." *Scientific Data* 10 (2023).
4. Fawaz, H.I., et al. "InceptionTime: Finding AlexNet for time series classification." *DMKD* 34.6 (2020).
5. Hu, J., et al. "Squeeze-and-Excitation Networks." *CVPR* (2018).

---

## License

This project is for research and educational purposes. Please refer to the individual dataset licenses ([PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/)) for data usage terms.

---

<p align="center"><em>Heart Reader — Automated ECG Diagnosis from Signal to Edge</em></p>
