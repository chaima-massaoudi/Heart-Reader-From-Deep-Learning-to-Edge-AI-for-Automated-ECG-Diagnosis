"""
Data preprocessing utilities for PTB-XL ECG signals.

Handles:
- Signal standardization (StandardScaler fit on training data)
- PTB-XL+ feature loading and imputation
- Label aggregation to 5 superclasses
"""

import ast
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tqdm import tqdm


# ── PTB-XL Superclass mapping ───────────────────────────────────────────────
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_ptbxl_database(path: str, sampling_rate: int = 100) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load PTB-XL database CSV and raw ECG signals.

    Args:
        path: Path to PTB-XL dataset root (containing ptbxl_database.csv).
        sampling_rate: 100 or 500 Hz.

    Returns:
        (df, signals) where df is the metadata DataFrame and signals is
        shape (N, seq_len, 12) numpy array.
    """
    path = Path(path)

    # Load metadata
    df = pd.read_csv(path / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    # Check for cached signals
    cache_file = path / f"raw{sampling_rate}.npy"
    if cache_file.exists():
        print(f"Loading cached signals from {cache_file}")
        signals = np.load(cache_file, allow_pickle=True)
    else:
        print(f"Loading {len(df)} ECG signals at {sampling_rate}Hz...")
        # Determine which filename column to use
        if sampling_rate == 100:
            fname_col = "filename_lr"
        else:
            fname_col = "filename_hr"

        signals = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading signals"):
            fname = str(path / row[fname_col])
            sig, _ = wfdb.rdsamp(fname)
            signals.append(sig)

        signals = np.array(signals, dtype=np.float32)
        np.save(cache_file, signals)
        print(f"Cached signals to {cache_file}")

    return df, signals


def compute_superdiagnostic_labels(
    df: pd.DataFrame, scp_statements_path: str
) -> pd.DataFrame:
    """Map SCP codes to 5 diagnostic superclasses.

    Args:
        df: PTB-XL metadata DataFrame with scp_codes column (dict).
        scp_statements_path: Path to scp_statements.csv.

    Returns:
        DataFrame with added 'superdiagnostic' column containing list of
        superclass labels per record, and 'superdiagnostic_len' for filtering.
    """
    # Load SCP statement definitions
    agg_df = pd.read_csv(scp_statements_path, index_col=0)

    # Filter to diagnostic statements only
    diag_agg = agg_df[agg_df.diagnostic == 1.0]

    # Build mapping: SCP code → diagnostic_class (superclass)
    code_to_superclass: Dict[str, str] = {}
    for code, row in diag_agg.iterrows():
        if isinstance(row.diagnostic_class, str) and row.diagnostic_class in SUPERCLASSES:
            code_to_superclass[code] = row.diagnostic_class

    def _aggregate(scp_dict):
        """Map a sample's SCP code dict to list of superclasses."""
        classes = set()
        for code, likelihood in scp_dict.items():
            if likelihood >= 50.0 and code in code_to_superclass:
                classes.add(code_to_superclass[code])
        return list(classes)

    df = df.copy()
    df["superdiagnostic"] = df.scp_codes.apply(_aggregate)
    df["superdiagnostic_len"] = df.superdiagnostic.apply(len)

    return df


def select_data(
    signals: np.ndarray,
    df: pd.DataFrame,
    min_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray, MultiLabelBinarizer, pd.DataFrame]:
    """Filter and encode labels as multi-hot vectors.

    Args:
        signals: Raw ECG signals array, shape (N, seq_len, 12).
        df: DataFrame with 'superdiagnostic' column.
        min_samples: Minimum samples per class to include.

    Returns:
        (X, y, mlb, df_filtered) where:
        - X: filtered signals, shape (M, seq_len, 12)
        - y: multi-hot labels, shape (M, 5)
        - mlb: fitted MultiLabelBinarizer
        - df_filtered: filtered DataFrame
    """
    # Remove samples with no diagnostic labels
    mask = df["superdiagnostic_len"] > 0
    df_filtered = df[mask].copy()
    X = signals[mask.values]

    # Fit multi-label binarizer on the 5 superclasses
    mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
    y = mlb.fit_transform(df_filtered["superdiagnostic"].values)

    # Optional: filter classes with too few samples
    if min_samples > 0:
        class_counts = y.sum(axis=0)
        valid_classes = class_counts >= min_samples
        if not valid_classes.all():
            print(f"Warning: Some classes have < {min_samples} samples. Keeping all 5 for challenge.")

    return X, y.astype(np.float32), mlb, df_filtered


def split_by_folds(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    train_folds: List[int],
    val_fold: int,
    test_fold: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split data using the predefined stratified folds.

    Args:
        X, y: Signal and label arrays.
        df: DataFrame with 'strat_fold' column.
        train_folds: List of fold numbers for training (e.g., [1..8]).
        val_fold: Fold number for validation (e.g., 9).
        test_fold: Fold number for test (e.g., 10).

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to (X_split, y_split).
    """
    folds = df["strat_fold"].values

    train_mask = np.isin(folds, train_folds)
    val_mask = folds == val_fold
    test_mask = folds == test_fold

    return {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
        "train_df": df[train_mask],
        "val_df": df[val_mask],
        "test_df": df[test_mask],
    }


def standardize_signals(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Standardize signals using a StandardScaler fitted on training data.

    Fits on flattened training signals (across all timesteps and leads),
    then transforms all splits.

    Args:
        X_train, X_val, X_test: Signal arrays, shape (N, seq_len, 12).
        save_path: Optional path to save the fitted scaler.

    Returns:
        (X_train_s, X_val_s, X_test_s, scaler)
    """
    N_train, seq_len, n_leads = X_train.shape
    scaler = StandardScaler()

    # Fit on training data (flatten to 2D: samples × features)
    scaler.fit(X_train.reshape(-1, n_leads))

    X_train_s = scaler.transform(X_train.reshape(-1, n_leads)).reshape(N_train, seq_len, n_leads)
    X_val_s = scaler.transform(X_val.reshape(-1, n_leads)).reshape(X_val.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_leads)).reshape(X_test.shape)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Saved StandardScaler to {save_path}")

    return (
        X_train_s.astype(np.float32),
        X_val_s.astype(np.float32),
        X_test_s.astype(np.float32),
        scaler,
    )


def load_ptbxl_plus_features(
    ptbxl_plus_path: str,
    ecg_ids: np.ndarray,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load and preprocess PTB-XL+ structured features.

    Combines features from 12SL and ECGdeli feature sets.
    Handles missing values via median imputation.

    Args:
        ptbxl_plus_path: Path to PTB-XL+ dataset root.
        ecg_ids: Array of ecg_id values to load features for.
        save_path: Optional path to save the fitted feature scaler.

    Returns:
        (features, scaler, feature_names) where features is shape (N, F).
    """
    ptbxl_plus_path = Path(ptbxl_plus_path)

    dfs = []
    feature_files = [
        ("12sl_features.csv", "12sl"),
        ("ecgdeli_features.csv", "ecgdeli"),
    ]

    for fname, prefix in feature_files:
        fpath = ptbxl_plus_path / "features" / fname
        if fpath.exists():
            feat_df = pd.read_csv(fpath)
            # Rename columns to avoid collision (except ecg_id)
            if "ecg_id" in feat_df.columns:
                rename_cols = {
                    c: f"{prefix}_{c}" for c in feat_df.columns if c != "ecg_id"
                }
                feat_df = feat_df.rename(columns=rename_cols)
                dfs.append(feat_df)
            else:
                print(f"Warning: {fname} has no ecg_id column, skipping.")
        else:
            print(f"Warning: Feature file {fpath} not found, skipping.")

    if not dfs:
        print("No PTB-XL+ features found. Returning empty array.")
        return np.zeros((len(ecg_ids), 0), dtype=np.float32), None, []

    # Merge all feature sets on ecg_id
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="ecg_id", how="outer")

    # Filter to requested ecg_ids and align order
    merged = merged.set_index("ecg_id")
    merged = merged.reindex(ecg_ids)

    # Drop columns that are entirely NaN
    merged = merged.dropna(axis=1, how="all")

    # Keep only numeric columns
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    merged = merged[numeric_cols]

    feature_names = list(merged.columns)

    # Median imputation
    medians = merged.median()
    merged = merged.fillna(medians)

    # Replace any remaining NaN/inf with 0
    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0)

    features = merged.values.astype(np.float32)

    return features, feature_names


def standardize_features(
    feat_train: np.ndarray,
    feat_val: np.ndarray,
    feat_test: np.ndarray,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Standardize structured features with a scaler fit on training data."""
    scaler = StandardScaler()
    feat_train_s = scaler.fit_transform(feat_train)
    feat_val_s = scaler.transform(feat_val)
    feat_test_s = scaler.transform(feat_test)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)

    return (
        feat_train_s.astype(np.float32),
        feat_val_s.astype(np.float32),
        feat_test_s.astype(np.float32),
        scaler,
    )
