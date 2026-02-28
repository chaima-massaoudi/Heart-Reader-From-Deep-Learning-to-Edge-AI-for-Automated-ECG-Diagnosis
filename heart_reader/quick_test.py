"""
Quick end-to-end pipeline test using synthetic ECG-like data.

Validates the entire pipeline (data → model → train → evaluate → export)
without requiring the actual PTB-XL download. Useful for:
- Verifying code correctness before real training
- CI/CD smoke testing

Usage:
    python quick_test.py                     # Quick 2-epoch test
    python quick_test.py --backbone inception1d  # Test single backbone
    python quick_test.py --epochs 5          # Custom epochs
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import ECGDataset
from data.augmentation import build_augmentation
from models.fusion_model import FusionModel, StandaloneModel, build_backbone
from models.ensemble import WeightedEnsemble
from training.trainer import Trainer
from training.losses import build_loss
from training.metrics import evaluate_all, macro_auc, find_optimal_thresholds, apply_thresholds
from evaluation.evaluate import bootstrap_evaluate, print_results_table


def generate_synthetic_data(
    n_train: int = 400,
    n_val: int = 100,
    n_test: int = 100,
    seq_len: int = 1000,
    n_leads: int = 12,
    n_classes: int = 5,
    n_features: int = 20,
):
    """Generate synthetic ECG-like data for pipeline testing.

    Creates realistic-looking multi-label data where:
    - Class 0 (NORM) correlates with low-amplitude smooth signals
    - Other classes correlate with specific frequency/amplitude patterns
    """
    np.random.seed(42)

    def make_signals(n):
        t = np.linspace(0, 10, seq_len)
        signals = np.zeros((n, seq_len, n_leads), dtype=np.float32)
        for i in range(n):
            for lead in range(n_leads):
                # Base sinus rhythm
                freq = 1.0 + np.random.uniform(-0.2, 0.2)
                signals[i, :, lead] = (
                    0.5 * np.sin(2 * np.pi * freq * t)
                    + 0.1 * np.sin(2 * np.pi * 3 * freq * t)
                    + np.random.randn(seq_len) * 0.05
                )
        return signals

    def make_labels(n):
        # Create correlated multi-label assignments
        labels = np.zeros((n, n_classes), dtype=np.float32)
        for i in range(n):
            # Each sample has 1-3 labels
            n_labels = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            chosen = np.random.choice(n_classes, size=n_labels, replace=False)
            labels[i, chosen] = 1.0
        return labels

    def make_features(n):
        return np.random.randn(n, n_features).astype(np.float32)

    return {
        "X_train": make_signals(n_train),
        "X_val": make_signals(n_val),
        "X_test": make_signals(n_test),
        "y_train": make_labels(n_train),
        "y_val": make_labels(n_val),
        "y_test": make_labels(n_test),
        "feat_train": make_features(n_train),
        "feat_val": make_features(n_val),
        "feat_test": make_features(n_test),
    }


def quick_test(backbone_name: str = "inception1d", epochs: int = 2, use_features: bool = True):
    """Run a quick end-to-end test of the training pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"QUICK TEST — {backbone_name} on synthetic data ({device})")
    print(f"{'='*70}")

    # 1) Generate synthetic data
    print("\n[1/7] Generating synthetic data...")
    data = generate_synthetic_data(
        n_train=200, n_val=50, n_test=50,
        n_features=20 if use_features else 0,
    )
    print(f"  Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")

    # 2) Build datasets and loaders
    print("[2/7] Building DataLoaders...")
    aug_cfg = {"enabled": True, "gaussian_noise_std": 0.02, "random_scale_range": [0.95, 1.05]}
    aug = build_augmentation(aug_cfg)

    num_features = data["feat_train"].shape[1] if use_features else 0

    train_ds = ECGDataset(
        data["X_train"], data["y_train"],
        features=data["feat_train"] if use_features else None,
        transform=aug,
    )
    val_ds = ECGDataset(
        data["X_val"], data["y_val"],
        features=data["feat_val"] if use_features else None,
    )
    test_ds = ECGDataset(
        data["X_test"], data["y_test"],
        features=data["feat_test"] if use_features else None,
    )

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3) Build model
    print(f"[3/7] Building model: {backbone_name} (features={use_features})...")
    model_cfg = {
        "num_classes": 5,
        "input_channels": 12,
        "input_length": 1000,
        "inception1d": {"depth": 3, "kernel_size": 20, "nb_filters": 16, "bottleneck_size": 16, "use_se": True},
        "xresnet1d101": {"layers": [1, 1, 1, 1], "expansion": 1, "block_szs": [32, 64, 128, 256]},
        "se_resnet1d": {"inplanes": 64, "layers": [1, 1, 1]},
        "feature_branch": {"hidden_dims": [64, 32], "dropout": 0.2},
        "fusion": {"signal_embed_dim": 64, "feature_embed_dim": 32, "fusion_hidden": 32, "dropout": 0.3},
    }

    if use_features and num_features > 0:
        model = FusionModel(
            backbone_name=backbone_name,
            model_cfg=model_cfg,
            num_classes=5,
            input_channels=12,
            num_features=num_features,
        )
    else:
        model = StandaloneModel(
            backbone_name=backbone_name,
            model_cfg=model_cfg,
            num_classes=5,
            input_channels=12,
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # 4) Train
    print(f"[4/7] Training for {epochs} epochs...")
    cfg = {
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.001,
            "weight_decay": 0.01,
            "scheduler": "one_cycle",
            "loss": "bce",
            "grad_clip": 1.0,
            "use_amp": False,
            "early_stopping": {"enabled": False},
            "checkpoint_dir": "./test_checkpoints/",
            "log_dir": "./test_logs/",
            "num_workers": 0,
        },
        "model": model_cfg,
        "evaluation": {"bootstrap": False},
    }

    trainer = Trainer(model, cfg, device, model_name=f"test_{backbone_name}")
    history = trainer.fit(train_loader, val_loader, y_val=data["y_val"])

    # 5) Predict
    print("[5/7] Running inference...")
    val_preds = trainer.predict(val_loader)
    test_preds = trainer.predict(test_loader)

    val_auc = macro_auc(data["y_val"], val_preds)
    test_auc = macro_auc(data["y_test"], test_preds)
    print(f"  Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")

    # 6) Evaluate with metrics
    print("[6/7] Computing all metrics...")
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    results = evaluate_all(data["y_test"], test_preds, class_names)
    print(f"  macro_auc: {results['macro_auc']:.4f}")
    print(f"  fmax: {results['fmax']:.4f}")

    # 7) Test thresholds
    print("[7/7] Testing threshold optimization...")
    thresholds = find_optimal_thresholds(data["y_val"], val_preds)
    binary_preds = apply_thresholds(test_preds, thresholds)
    print(f"  Thresholds: {[f'{t:.3f}' for t in thresholds]}")
    print(f"  Binary predictions shape: {binary_preds.shape}")

    print(f"\n{'='*70}")
    print(f"QUICK TEST PASSED — {backbone_name}")
    print(f"{'='*70}\n")

    return {
        "model": trainer.model,
        "history": history,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "val_auc": val_auc,
        "test_auc": test_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick pipeline test")
    parser.add_argument("--backbone", type=str, default="inception1d",
                        choices=["inception1d", "xresnet1d101", "se_resnet1d", "all"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--no-features", action="store_true")
    args = parser.parse_args()

    start = time.time()

    if args.backbone == "all":
        backbones = ["inception1d", "xresnet1d101", "se_resnet1d"]
        all_results = {}

        for bb in backbones:
            result = quick_test(bb, args.epochs, not args.no_features)
            all_results[bb] = result

        # Test ensemble
        print(f"\n{'='*70}")
        print("TESTING ENSEMBLE")
        print(f"{'='*70}")

        val_preds_list = [r["val_preds"] for r in all_results.values()]
        test_preds_list = [r["test_preds"] for r in all_results.values()]

        data = generate_synthetic_data(n_train=200, n_val=50, n_test=50)

        ens = WeightedEnsemble(method="weighted")
        ens.fit_weights(val_preds_list, data["y_val"])
        ens_preds = ens.predict(test_preds_list)
        ens_auc = macro_auc(data["y_test"], ens_preds)
        print(f"  Ensemble weights: {[f'{w:.3f}' for w in ens.weights]}")
        print(f"  Ensemble AUC: {ens_auc:.4f}")

        print(f"\n{'='*70}")
        print(f"ALL QUICK TESTS PASSED!")
        print(f"{'='*70}")

    else:
        quick_test(args.backbone, args.epochs, not args.no_features)

    elapsed = time.time() - start
    print(f"\nTotal test time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
