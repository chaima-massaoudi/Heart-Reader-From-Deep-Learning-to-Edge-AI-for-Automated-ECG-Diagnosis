<!-- ============================================================ -->
<!--                      HEADER & BADGES                         -->
<!-- ============================================================ -->

<p align="center">
  <img src="https://img.shields.io/badge/Macro--AUC-0.927-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/F--max-0.757-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-2.37%20MB%20INT8-orange?style=for-the-badge" />
</p>

<h1 align="center">
  <br>
  <img src="https://em-content.zobj.net/source/twitter/376/anatomical-heart_1fac0.png" width="80" />
  <br>
  Heart Reader
  <br>
</h1>

<h3 align="center">From Deep Learning to Edge AI for Automated 12-Lead ECG Diagnosis</h3>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.134-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Research-lightgrey?style=flat-square" />
</p>

<p align="center">
  <strong>Chaima Massaoudi</strong> &mdash; March 2026
</p>

<p align="center">
  A production-ready, multimodal deep learning pipeline for <strong>multi-label ECG classification</strong> of five diagnostic superclasses from 12-lead electrocardiogram recordings &mdash; featuring an <strong>InceptionTime1D + Squeeze-and-Excitation</strong> backbone fused with <strong>1,313 PTB-XL+ structured features</strong>, an interactive web dashboard, and a complete edge deployment pipeline.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-live-demo">Demo</a> &bull;
  <a href="#-architecture">Architecture</a> &bull;
  <a href="#-results">Results</a> &bull;
  <a href="#-web-dashboard">Dashboard</a> &bull;
  <a href="#-edge-deployment">Edge AI</a> &bull;
  <a href="heart_reader/REPORT.md">Full Report</a>
</p>

---

## Key Results

<table>
  <tr>
    <td align="center"><strong>Macro-AUC</strong><br/><code>0.9268</code></td>
    <td align="center"><strong>Macro F1</strong><br/><code>0.7239</code></td>
    <td align="center"><strong>F-max</strong><br/><code>0.7574</code></td>
    <td align="center"><strong>AUPRC</strong><br/><code>0.8097</code></td>
    <td align="center"><strong>Parameters</strong><br/><code>977K</code></td>
    <td align="center"><strong>Model Size</strong><br/><code>2.37 MB</code></td>
  </tr>
</table>

<details>
<summary><strong>Per-Class AUC Breakdown (Test Set &mdash; Fold 10)</strong></summary>
<br/>

| Class | AUC | 90% CI | Description |
|:------|:---:|:------:|:------------|
| **NORM** | 0.957 | [0.949, 0.964] | Normal ECG |
| **MI** | 0.927 | [0.915, 0.940] | Myocardial Infarction |
| **STTC** | 0.930 | [0.918, 0.941] | ST/T Change |
| **CD** | 0.913 | [0.902, 0.928] | Conduction Disturbance |
| **HYP** | 0.908 | [0.893, 0.924] | Hypertrophy |

All classes exceed **AUC 0.90**, demonstrating robust classification across the full diagnostic spectrum.

</details>

---

## Live Demo

<p align="center">
  <a href="heart_reader/demo.mp4">
    <img src="heart_reader/demo.gif" alt="Heart Reader Dashboard Demo" width="780" />
  </a>
</p>

<p align="center">
  <em>Interactive dashboard &mdash; 12-lead ECG visualization, real-time diagnosis, and confidence scores (5&times; speed)</em>
  <br/>
  <a href="heart_reader/demo.mp4"><strong>Watch full demo video &rarr;</strong></a>
</p>

---

## Why Heart Reader?

| Challenge | Our Solution |
|:----------|:-------------|
| Manual ECG interpretation requires years of specialist training | **Automated AI-powered diagnosis** in milliseconds |
| Single-modality models miss complementary clinical features | **Multimodal fusion** of raw signals + 1,313 structured PTB-XL+ features |
| Large models can't run on portable medical devices | **Edge-optimized** pipeline: pruning + INT8 quantization + ONNX (2.37 MB) |
| Research models lack practical interfaces | **Interactive web dashboard** with real-time analysis & export |
| ECG diagnosis is multi-label (patients have multiple conditions) | **Multi-label classification** handling co-occurring pathologies |

---

## Architecture

### Multimodal Late Fusion Pipeline

```
 ┌─────────────────────────────┐       ┌─────────────────────────────┐
 │     12-Lead ECG Signal      │       │     PTB-XL+ Features        │
 │     (1000 × 12 matrix)      │       │     (1313-dim vector)       │
 └──────────────┬──────────────┘       └──────────────┬──────────────┘
                │                                      │
   ┌────────────▼────────────┐            ┌────────────▼────────────┐
   │   InceptionTime1D       │            │     Feature MLP         │
   │   + SE Channel Attention│            │  [1313 → 256 → 128 → 64]│
   │   (6 blocks, multi-     │            │  BN + ReLU + Dropout    │
   │    scale: k=39,19,9)    │            └────────────┬────────────┘
   └────────────┬────────────┘                         │
                │                                 64-dim embed
   AdaptiveConcatPool1d                                │
   + Linear → 256-dim                                  │
                │                                      │
                └────────────┬─────────────────────────┘
                             │
                    Concat [256 ‖ 64] = 320-dim
                             │
                    ┌────────▼────────┐
                    │  Fusion MLP Head │
                    │ [320 → 128 → 5] │
                    │ ReLU + BN + Drop │
                    └────────┬────────┘
                             │
                      5-class sigmoid
                    (multi-label output)
```

### Signal Backbone: InceptionTime1D + SE Attention

| Component | Configuration |
|:----------|:-------------|
| Inception Blocks | 6 deep, residual every 3 |
| Multi-scale Kernels | k=39, k=19, k=9 (captures patterns at multiple temporal resolutions) |
| SE Attention | Reduction ratio 16 (adaptive channel recalibration) |
| Pooling | AdaptiveConcatPool1d (avg + max) |
| Output | 256-dim signal embedding |

### Training Enhancements

| Technique | Configuration | Purpose |
|:----------|:-------------|:--------|
| **Focal Loss** | gamma=2, alpha=0.25 | Handle class imbalance (HYP is 4x rarer) |
| **Label Smoothing** | epsilon=0.05 | Better calibrated confidence scores |
| **Mixup** | alpha=0.3 | Data augmentation + regularization |
| **SWA** | From epoch 40 | Converge to flatter minima |
| **OneCycleLR** | max_lr=0.003 | Super-convergence scheduling |
| **AdamW** | wd=0.01 | Decoupled weight regularization |
| **AMP (FP16)** | Automatic | 2× memory reduction & speed |
| **Gradient Clipping** | max_norm=1.0 | Training stability |

### ECG-Specific Data Augmentation

- **Gaussian noise** (sigma=0.05) — simulates sensor noise
- **Random amplitude scaling** (0.8–1.2×) — accounts for gain variation
- **Lead dropout** (up to 2 leads, p=0.1) — robustness to disconnection
- **Baseline wander** (sinusoidal drift, p=0.3) — common ambulatory artifact
- **Temporal warping** (interpolation, p=0.2) — heart rate variability robustness

---

## Results

### Test Set Performance (Fold 10, N=2,198)

All confidence intervals via **100 bootstrap resamples** (5th–95th percentile).

| Metric | Score | 90% CI |
|:-------|:-----:|:------:|
| **Macro-AUC** | **0.9268** | [0.9204, 0.9332] |
| **F-max** | 0.7574 | [0.7441, 0.7696] |
| **Macro F1** | 0.7239 | [0.7048, 0.7468] |
| **Macro AUPRC** | 0.8097 | [0.7946, 0.8256] |
| **F-beta (beta=2)** | 0.7902 | [0.7809, 0.8066] |

### Per-Class Classification Report

| Class | Precision | Recall | F1 | Support |
|:------|:---------:|:------:|:--:|:-------:|
| **NORM** | 0.82 | 0.95 | 0.88 | 954 |
| **MI** | 0.59 | 0.87 | 0.70 | 415 |
| **STTC** | 0.63 | 0.90 | 0.74 | 506 |
| **CD** | 0.67 | 0.82 | 0.74 | 496 |
| **HYP** | 0.38 | 0.74 | 0.50 | 222 |
| **Macro Avg** | 0.62 | 0.86 | **0.71** | 2,593 |

> High recall across all classes (≥0.74) means the model reliably detects pathological patterns — clinically preferred over high precision alone.

### Comparison with Published PTB-XL Benchmarks

| Model | Macro-AUC | Input | Source |
|:------|:---------:|:------|:-------|
| inception1d (benchmark) | 0.925 | Signal only | Strodthoff et al., 2021 |
| xresnet1d101 (benchmark) | 0.925 | Signal only | Strodthoff et al., 2021 |
| resnet1d_wang (benchmark) | 0.930 | Signal only | Strodthoff et al., 2021 |
| **Heart Reader (Ours)** | **0.927** | **Signal + PTB-XL+** | **This work** |

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

### 2. Download Datasets

Download from [PhysioNet](https://physionet.org) (free, requires credentialed access):

| Dataset | Version | Path (relative to repo root) |
|:--------|:--------|:-----------------------------|
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 1.0.3 | `data/ptbxl/` |
| [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/) | 1.0.1 | `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/` |

<details>
<summary><strong>Expected directory layout</strong></summary>

```
repo-root/
├── data/
│   └── ptbxl/
│       ├── ptbxl_database.csv
│       ├── scp_statements.csv
│       └── records100/...
├── ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/
│   └── features/
│       ├── 12sl_features.csv
│       └── ecgdeli_features.csv
└── heart_reader/
    └── (this project)
```
</details>

### 3. Train

```bash
cd heart_reader

# Improved config (focal loss, mixup, SWA) — recommended
python train.py --config configs/improved.yaml --backbone inception1d

# Default baseline training
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

### 6. Export for Edge Devices

```bash
python export.py --backbone inception1d
```

---

## Web Dashboard

A full-featured, interactive **FastAPI-powered** medical dashboard for real-time ECG analysis.

### Features

| Feature | Description |
|:--------|:------------|
| **Random Sample** | Load a random test-set ECG with full-model predictions |
| **File Upload** | Upload your own 12-lead CSV for instant analysis |
| **12-Lead ECG Viewer** | Full waveform visualization across all leads (Chart.js) |
| **Probability Chart** | Interactive bar chart with per-class confidence scores |
| **Diagnosis Panel** | Clear Normal/Abnormal banner with detailed class breakdown |
| **Graph Export** | One-click PNG download of any visualization |
| **Dark Medical Theme** | Professional dark interface with ECG-green (#00df80) accents |

### Upload Formats

| Format | Description |
|:-------|:------------|
| **Signal-only** | 12 columns × 1000 rows (one column per ECG lead) |
| **Signal + Features** | >12 columns × 1000 rows (first 12 = leads, rest = PTB-XL+ features) |

### Pre-built Test Files

One sample CSV per diagnostic class is included in `heart_reader/test_files/`:

| File | Expected Diagnosis | Confidence |
|:-----|:-------------------|:----------:|
| `sample_norm.csv` | NORM | 1.000 |
| `sample_mi.csv` | MI | 0.872 |
| `sample_sttc.csv` | STTC | 0.674 |
| `sample_cd.csv` | CD | 0.962 |
| `sample_hyp.csv` | HYP | 0.710 |

---

## Edge Deployment

A complete compression pipeline for deploying on resource-constrained devices (Raspberry Pi, mobile, embedded medical hardware).

### Pipeline

```
Full Model (3.79 MB)
    │
    ├── L1 Structured Pruning ──→ 50% filter removal
    │
    ├── INT8 Dynamic Quantization ──→ FP32 → INT8 weights
    │
    └── ONNX Export ──→ Cross-platform inference
```

### Compression Results

| Variant | Size | Compression | Format |
|:--------|:----:|:-----------:|:------:|
| Full Fusion (FP32) | 3.79 MB | 1.0× | PyTorch |
| **Quantized (INT8)** | **2.37 MB** | **1.6×** | PyTorch |
| Signal-Only ONNX | 2.16 MB | 1.75× | ONNX |

### Inference Latency (CPU)

| Configuration | Mean | Std Dev |
|:-------------|:----:|:-------:|
| Full Fusion (FP32) | 717 ms | ±398 ms |
| Quantized (INT8) | 827 ms | — |

---

## Data Protocol

Following the **official PTB-XL benchmark** (Strodthoff et al., 2021) — strict fold-based stratification with no patient overlap.

| Split | Folds | Samples | Purpose |
|:------|:-----:|:-------:|:--------|
| **Train** | 1–8 | 16,289 | Model training |
| **Validation** | 9 | 2,034 | Hyperparameter tuning, threshold optimization |
| **Test** | 10 | 2,050 | Final evaluation only (never used during development) |

### Diagnostic Superclasses

| Class | Abbreviation | Test Samples | Multi-label? |
|:------|:------------|:------------:|:------------:|
| Normal ECG | NORM | 954 | — |
| Myocardial Infarction | MI | 415 | Often + STTC |
| ST/T Change | STTC | 506 | Often + MI, HYP |
| Conduction Disturbance | CD | 496 | Can co-occur |
| Hypertrophy | HYP | 222 | Often + STTC |

> 64% of records have exactly one label, 28% have two, 8% have three or more.

---

## Project Structure

```
heart_reader/
├── configs/
│   ├── default.yaml              # Base configuration
│   └── improved.yaml             # Enhanced: focal loss + mixup + SWA
│
├── data/
│   ├── preprocessing.py          # PTB-XL loading, label mapping, PTB-XL+ features
│   ├── dataset.py                # PyTorch Dataset & DataLoader
│   └── augmentation.py           # ECG-specific augmentations
│
├── models/
│   ├── inception1d.py            # InceptionTime1D + SE attention
│   ├── xresnet1d.py              # XResNet1d family (18–152 layers)
│   ├── se_resnet1d.py            # SE-ResNet1d (Wang-style)
│   ├── fusion_model.py           # Multimodal signal + feature fusion
│   ├── feature_branch.py         # MLP for structured features
│   ├── ensemble.py               # Weighted ensemble (Nelder-Mead)
│   └── heads.py                  # Pooling, SE blocks, weight init
│
├── training/
│   ├── trainer.py                # Training loop (AMP, SWA, early stopping)
│   ├── losses.py                 # Focal loss, label smoothing BCE, mixup
│   ├── metrics.py                # AUC, Fmax, G_beta, AUPRC
│   └── callbacks.py              # LR scheduling callbacks
│
├── evaluation/
│   ├── evaluate.py               # Bootstrap evaluation, CI, CSV export
│   └── visualization.py          # ROC curves, confusion matrix, plots
│
├── edge/
│   ├── prune.py                  # L1 structured pruning
│   ├── quantize.py               # INT8 dynamic quantization
│   ├── export_tflite.py          # ONNX export
│   └── benchmark.py              # Inference benchmarking
│
├── frontend/
│   ├── app.py                    # FastAPI backend (REST API)
│   └── static/                   # Dashboard (HTML, CSS, JS)
│
├── results/
│   ├── evaluation_report.json    # Full metrics with bootstrap CIs
│   └── edge/edge_stats.json      # Compression & latency stats
│
├── test_files/                   # Pre-selected test CSVs per class
├── train.py                      # Main entry point
├── evaluate.py                   # Standalone evaluation
├── export.py                     # Edge deployment pipeline
├── run_server.py                 # Web dashboard launcher
├── demo.mp4 / demo.gif          # Video demonstration
├── REPORT.md                     # Detailed technical report (658 lines)
└── requirements.txt              # Python dependencies
```

---

## Hardware & Environment

| Component | Specification |
|:----------|:-------------|
| **GPU** | NVIDIA GeForce RTX 3050 6GB |
| **CUDA** | 12.6 |
| **PyTorch** | 2.10.0+cu126 |
| **Python** | 3.13 |
| **Training Time** | ~27 minutes (41 epochs, early stop at 26) |
| **Seed** | 42 (full deterministic reproducibility) |

---

## References

1. Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7 (2020). [DOI](https://doi.org/10.1038/s41597-020-0495-6)
2. Strodthoff, N., et al. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE JBHI* 25.5 (2021). [DOI](https://doi.org/10.1109/JBHI.2020.3022989)
3. Strodthoff, N., et al. "PTB-XL+, a comprehensive electrocardiographic feature dataset." *Scientific Data* 10 (2023). [DOI](https://doi.org/10.1038/s41597-023-02153-8)
4. Fawaz, H.I., et al. "InceptionTime: Finding AlexNet for time series classification." *DMKD* 34.6 (2020). [DOI](https://doi.org/10.1007/s10618-020-00710-y)
5. Hu, J., et al. "Squeeze-and-Excitation Networks." *CVPR* (2018).
6. Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." *IEEE TPAMI* 42.2 (2020).
7. Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." *ICLR* (2018).
8. Izmailov, P., et al. "Averaging Weights Leads to Wider Optima and Better Generalization." *UAI* (2018).

---

## License

This project is for **research and educational purposes**. Please refer to the individual dataset licenses for data usage terms:
- [PTB-XL License](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL+ License](https://physionet.org/content/ptb-xl-plus/1.0.1/)

---

<p align="center">
  <strong>Heart Reader</strong> &mdash; Automated ECG Diagnosis from Signal to Edge
  <br/>
  <sub>Built with PyTorch, FastAPI, and clinical rigor</sub>
</p>
