"""
PyTorch Dataset for PTB-XL ECG data with optional PTB-XL+ features.

Returns (signal, features, label) tuples for training/evaluation.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentation import Compose, build_augmentation
from .preprocessing import (
    SUPERCLASSES,
    compute_superdiagnostic_labels,
    load_ptbxl_database,
    load_ptbxl_plus_features,
    select_data,
    split_by_folds,
    standardize_features,
    standardize_signals,
)


class ECGDataset(Dataset):
    """PyTorch Dataset for 12-lead ECG signals with optional structured features.

    Attributes:
        signals: np.ndarray of shape (N, seq_len, 12)
        labels: np.ndarray of shape (N, num_classes) — multi-hot
        features: Optional np.ndarray of shape (N, F) — structured features
        transform: Optional augmentation pipeline (applied to signals only)
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ):
        self.signals = signals
        self.labels = labels
        self.features = features
        self.transform = transform

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        signal = self.signals[idx].copy()  # (seq_len, 12)
        label = self.labels[idx]           # (num_classes,)

        # Apply augmentation (on numpy, before converting to tensor)
        if self.transform is not None:
            signal = self.transform(signal)

        # Convert to tensors — signal: (12, seq_len) for Conv1d
        signal_tensor = torch.from_numpy(signal).float().permute(1, 0)  # (12, seq_len)
        label_tensor = torch.from_numpy(label).float()

        item = {"signal": signal_tensor, "label": label_tensor}

        if self.features is not None:
            feat = self.features[idx]
            item["features"] = torch.from_numpy(feat).float()

        return item


def prepare_data(cfg: dict) -> Dict:
    """Full data preparation pipeline.

    Loads PTB-XL signals, computes superdiagnostic labels, splits by fold,
    standardizes, optionally loads PTB-XL+ features, builds DataLoaders.

    Args:
        cfg: Full YAML config dict.

    Returns:
        Dict with keys:
        - 'train_loader', 'val_loader', 'test_loader': DataLoaders
        - 'num_features': int — number of structured features (0 if disabled)
        - 'num_classes': int — 5 (superclasses)
        - 'class_names': list of class name strings
        - 'mlb': fitted MultiLabelBinarizer
        - 'y_val', 'y_test': numpy arrays for evaluation
    """
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg["training"]

    ptbxl_path = data_cfg["ptbxl_path"]
    sampling_rate = data_cfg["sampling_rate"]  # 100
    train_folds = data_cfg["train_folds"]      # [1..8]
    val_fold = data_cfg["val_fold"]            # 9
    test_fold = data_cfg["test_fold"]          # 10

    # 1) Load PTB-XL database + signals
    print("=" * 60)
    print("STEP 1: Loading PTB-XL database and signals...")
    print("=" * 60)
    df, signals = load_ptbxl_database(ptbxl_path, sampling_rate)
    print(f"  Loaded {len(df)} records, signal shape: {signals.shape}")

    # 2) Compute superdiagnostic labels
    print("\nSTEP 2: Computing superdiagnostic labels...")
    scp_path = os.path.join(ptbxl_path, "scp_statements.csv")
    df = compute_superdiagnostic_labels(df, scp_path)

    # 3) Filter and encode labels
    print("\nSTEP 3: Encoding multi-hot labels for 5 superclasses...")
    X, y, mlb, df_filtered = select_data(signals, df, data_cfg.get("min_samples", 0))
    print(f"  After filtering: {X.shape[0]} samples, {y.shape[1]} classes")
    print(f"  Class distribution: {dict(zip(SUPERCLASSES, y.sum(axis=0).astype(int)))}")

    # 4) Split by stratified folds
    print("\nSTEP 4: Splitting by stratified folds...")
    splits = split_by_folds(X, y, df_filtered, train_folds, val_fold, test_fold)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 5) Standardize signals
    print("\nSTEP 5: Standardizing signals (fit on train)...")
    cache_dir = data_cfg.get("cache_dir", "./cache/")
    os.makedirs(cache_dir, exist_ok=True)
    X_train, X_val, X_test, sig_scaler = standardize_signals(
        X_train, X_val, X_test,
        save_path=os.path.join(cache_dir, "signal_scaler.pkl"),
    )

    # 6) Optionally load PTB-XL+ features
    feat_train = feat_val = feat_test = None
    num_features = 0

    if data_cfg.get("use_ptbxl_plus_features", False):
        ptbxl_plus_path = data_cfg.get("ptbxl_plus_path", "")
        if ptbxl_plus_path and Path(ptbxl_plus_path).exists():
            print("\nSTEP 6: Loading PTB-XL+ structured features...")

            train_ids = splits["train_df"].index.values
            val_ids = splits["val_df"].index.values
            test_ids = splits["test_df"].index.values

            feat_train_raw, feature_names = load_ptbxl_plus_features(
                ptbxl_plus_path, train_ids
            )
            feat_val_raw, _ = load_ptbxl_plus_features(ptbxl_plus_path, val_ids)
            feat_test_raw, _ = load_ptbxl_plus_features(ptbxl_plus_path, test_ids)

            if feat_train_raw.shape[1] > 0:
                feat_train, feat_val, feat_test, feat_scaler = standardize_features(
                    feat_train_raw, feat_val_raw, feat_test_raw,
                    save_path=os.path.join(cache_dir, "feature_scaler.pkl"),
                )
                num_features = feat_train.shape[1]
                print(f"  Loaded {num_features} structured features")
            else:
                print("  No numeric features found in PTB-XL+")
        else:
            print("\nSTEP 6: PTB-XL+ path not found, skipping features.")

    # 7) Build augmentation (training only)
    train_transform = build_augmentation(aug_cfg)

    # 8) Create datasets
    print(f"\nSTEP 7: Creating PyTorch datasets (batch_size={train_cfg['batch_size']})...")
    train_dataset = ECGDataset(X_train, y_train, feat_train, transform=train_transform)
    val_dataset = ECGDataset(X_val, y_val, feat_val, transform=None)
    test_dataset = ECGDataset(X_test, y_test, feat_test, transform=None)

    # 9) Create dataloaders
    has_cuda = torch.cuda.is_available()
    common_kwargs = dict(
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=has_cuda,
        persistent_workers=True if train_cfg.get("num_workers", 4) > 0 else False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        **common_kwargs,
    )

    print(f"\n{'='*60}")
    print(f"Data preparation complete!")
    print(f"  Classes: {SUPERCLASSES}")
    print(f"  Signal shape: ({X_train.shape[1]}, {X_train.shape[2]}) = ({data_cfg['duration']}s × {data_cfg['num_leads']} leads)")
    print(f"  Structured features: {num_features}")
    print(f"{'='*60}\n")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_features": num_features,
        "num_classes": y.shape[1],
        "class_names": SUPERCLASSES,
        "mlb": mlb,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
