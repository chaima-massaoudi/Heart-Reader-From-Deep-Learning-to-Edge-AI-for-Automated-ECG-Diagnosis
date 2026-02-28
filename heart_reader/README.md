# Heart Reader Challenge — Multi-Label ECG Classification

A complete, competition-ready multi-label ECG classification system for the **Heart Reader Challenge**, built on the PTB-XL dataset with modern PyTorch.

## Approach Summary

We detect **5 diagnostic superclasses** from 12-lead ECG recordings:

| Class | Description |
|-------|-------------|
| NORM  | Normal ECG |
| MI    | Myocardial Infarction |
| STTC  | ST/T Change |
| CD    | Conduction Disturbance |
| HYP   | Hypertrophy |

### Architecture

**Three backbone ensemble with multimodal fusion:**

1. **InceptionTime1D** — Multi-scale temporal convolutions (kernel sizes k, k/2, k/4) with Squeeze-and-Excitation channel attention at every block. Depth-6 architecture with residual connections every 3 blocks.

2. **XResNet1d101** — Deep residual network (101 layers) with a 3-convolution stem, bottleneck blocks, and batch-norm-zero initialization for stable deep training.

3. **SE-ResNet1d** — Wang-style ResNet (no pooling in stem, kernel sizes [5,3]) augmented with SE blocks for adaptive channel recalibration.

Each backbone is fused with a **Feature Branch MLP** that encodes ~198 structured features from the PTB-XL+ dataset (P/QRS/T amplitudes, durations, axes from 12SL and ECGdeli). Signal embeddings (256-dim) are concatenated with feature embeddings (64-dim) and fed through a fusion head.

A **learned weighted ensemble** combines the 3 models via scipy Nelder-Mead optimization on validation AUC.

### Key Design Choices

- **Loss**: Focal Loss (α=0.25, γ=2) to handle class imbalance
- **Optimizer**: AdamW with OneCycleLR (max_lr=0.01, pct_start=0.3)
- **Mixed precision**: PyTorch AMP (float16) for 2x speedup
- **Data augmentation**: Gaussian noise, random scaling, lead dropout, baseline wander, time warping
- **Thresholding**: Per-class optimal thresholds via Youden's J statistic on validation fold
- **Evaluation**: Bootstrap confidence intervals (n=100), macro AUC, F_max, G_beta (challenge metrics)

## Project Structure

```
heart_reader/
├── train.py                # Main training entry point
├── evaluate.py             # Standalone evaluation / inference
├── export.py               # Edge deployment pipeline
├── requirements.txt        # Python dependencies
│
├── configs/
│   └── default.yaml        # Full configuration (hyperparams, paths, etc.)
│
├── data/
│   ├── augmentation.py     # ECG-specific data augmentation
│   ├── preprocessing.py    # Data loading, label aggregation, splits
│   └── dataset.py          # PyTorch Dataset & DataLoader construction
│
├── models/
│   ├── heads.py            # Pooling, SE blocks, classifier heads
│   ├── inception1d.py      # InceptionTime1D backbone
│   ├── xresnet1d.py        # XResNet1d family (18/34/50/101/152)
│   ├── se_resnet1d.py      # SE-ResNet1d (Wang-style + SE)
│   ├── feature_branch.py   # MLP for PTB-XL+ structured features
│   ├── fusion_model.py     # Multimodal fusion & standalone wrappers
│   └── ensemble.py         # Weighted ensemble with scipy optimization
│
├── training/
│   ├── losses.py           # Focal Loss, BCE
│   ├── metrics.py          # AUC, F_max, G_beta, threshold optimization
│   ├── callbacks.py        # Early stopping, model checkpointing
│   └── trainer.py          # Training loop (AMP, OneCycleLR, logging)
│
├── evaluation/
│   ├── evaluate.py         # Bootstrap CI, summary tables
│   └── visualization.py    # ROC curves, confusion matrices, plots
│
└── edge/
    ├── prune.py            # L1 structured pruning
    ├── quantize.py         # Dynamic INT8 quantization
    ├── export_tflite.py    # ONNX → TFLite conversion
    └── benchmark.py        # Size/latency benchmarking
```

## Quick Start

### 1. Install Dependencies

```bash
cd heart_reader
pip install -r requirements.txt
```

### 2. Dataset Setup

Download **PTB-XL v1.0.3** and place it at `../data/ptbxl/` relative to this directory (or update `configs/default.yaml`).

The PTB-XL+ feature dataset should be at the path specified in `configs/default.yaml` under `data.ptbxl_plus_path`.

Expected layout:
```
ainc/
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

### 3. Train All Models

```bash
# Train all 3 backbones + ensemble
python train.py

# Train a single backbone
python train.py --backbone inception1d

# Use a custom config
python train.py --config configs/my_config.yaml
```

### 4. Evaluate

```bash
# Evaluate all trained models + ensemble
python evaluate.py

# Evaluate ensemble only
python evaluate.py --ensemble

# Evaluate a specific model
python evaluate.py --backbone inception1d
python evaluate.py --model checkpoints/fusion_inception1d_best.pt
```

### 5. Edge Deployment

```bash
# Run full compression pipeline (prune → quantize → ONNX → TFLite)
python export.py --backbone inception1d

# Skip post-pruning fine-tuning (faster)
python export.py --backbone inception1d --skip-finetune
```

## Data Splits (Challenge Rules)

Following the official PTB-XL benchmark protocol:

| Split      | Folds | Purpose |
|------------|-------|---------|
| Training   | 1–8   | Model training |
| Validation | 9     | Hyperparameter tuning, early stopping, threshold optimization |
| Test       | 10    | Final evaluation (untouched during development) |

## Training Details

| Parameter | Value |
|-----------|-------|
| Input     | 12-lead ECG, 1000 samples (10s @ 100Hz) |
| Batch size | 128 |
| Optimizer | AdamW (lr=0.01, wd=1e-4) |
| Scheduler | OneCycleLR (50 epochs, pct_start=0.3) |
| Loss      | Focal Loss (α=0.25, γ=2) |
| AMP       | float16 mixed precision |
| Early stop | Patience 10 (monitor: val macro AUC) |

## Edge Optimization Pipeline

1. **Structured Pruning**: L1-norm based, 50% of Conv1d channels removed
2. **Post-Pruning Fine-tune**: 10 epochs at 1/10th learning rate
3. **Dynamic Quantization**: INT8 weights for Linear and Conv layers
4. **ONNX Export**: Opset 13, with shape inference
5. **TFLite Conversion**: DEFAULT optimization level

Expected compression: **3-5x smaller**, **2-4x faster** on CPU.

## Evaluation Metrics

- **Macro AUC** (primary): Area under ROC curve, averaged across classes
- **F_max**: Maximum F1 score across all thresholds
- **G_beta** (β=2): Challenge-specific metric (recall-weighted)
- **Per-class AUC**: Individual class performance
- **Bootstrap 95% CI**: 100-sample bootstrap on test set

## Novel Contributions

1. **SE-augmented InceptionTime**: Added Squeeze-and-Excitation attention to every Inception block (not in original benchmark)
2. **PTB-XL+ Multimodal Fusion**: Late-fusion of raw signal features with expert-engineered ECG morphology features from the PTB-XL+ dataset
3. **Learned Ensemble Weights**: Scipy-optimized weighting (vs. simple averaging) for combining model predictions
4. **Full Edge Pipeline**: End-to-end pruning → quantization → TFLite export with benchmarking

## References

- Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset", Scientific Data 2020
- Strodthoff et al., "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL", IEEE JBHI 2021
- PTB-XL+ feature dataset: Strodthoff et al., "PTB-XL+, a comprehensive electrocardiographic feature dataset", Scientific Data 2023

## License

Research use. See individual dataset licenses for data usage terms.
