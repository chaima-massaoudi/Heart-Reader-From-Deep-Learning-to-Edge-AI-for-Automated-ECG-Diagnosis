"""
Evaluation metrics for multi-label ECG classification.

Primary metric: macro_auc (macro-averaged ROC AUC)
Secondary: per-class AUC, Fmax, F_beta_macro, G_beta_macro

Compatible with the PTB-XL benchmarking evaluation protocol.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict, Optional, Tuple


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged ROC AUC across all classes.

    This is the primary metric for the PTB-XL benchmark.

    Args:
        y_true: (N, C) ground truth multi-hot labels.
        y_pred: (N, C) predicted probabilities.

    Returns:
        Macro AUC score.
    """
    try:
        return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    except ValueError:
        # Handle edge case where a class has no positive samples
        aucs = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:
                try:
                    aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
                except ValueError:
                    pass
        return np.mean(aucs) if aucs else 0.0


def per_class_auc(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None) -> Dict[str, float]:
    """Compute per-class ROC AUC.

    Args:
        y_true: (N, C) ground truth.
        y_pred: (N, C) predictions.
        class_names: Optional list of class names.

    Returns:
        Dict mapping class name → AUC.
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    result = {}
    for i, name in enumerate(class_names):
        try:
            result[name] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            result[name] = float("nan")
    return result


def find_optimal_thresholds(
    y_true: np.ndarray, y_pred: np.ndarray, method: str = "youden"
) -> np.ndarray:
    """Find optimal per-class thresholds for binarizing predictions.

    Args:
        y_true: (N, C) ground truth.
        y_pred: (N, C) predictions.
        method: "youden" (maximize TPR - FPR) or "f1" (maximize F1).

    Returns:
        Array of C optimal thresholds.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    n_classes = y_true.shape[1]
    thresholds = np.zeros(n_classes)

    for i in range(n_classes):
        if method == "youden":
            fpr, tpr, thresh = roc_curve(y_true[:, i], y_pred[:, i])
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            thresholds[i] = thresh[best_idx]
        elif method == "f1":
            # Search over thresholds to maximize F1
            best_f1 = 0.0
            best_t = 0.5
            for t in np.linspace(0.01, 0.99, 100):
                pred_binary = (y_pred[:, i] >= t).astype(int)
                if pred_binary.sum() == 0:
                    continue
                f1 = f1_score(y_true[:, i], pred_binary, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            thresholds[i] = best_t
        else:
            thresholds[i] = 0.5

    return thresholds


def apply_thresholds(y_pred: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binarize predictions using per-class thresholds.

    If no class exceeds its threshold for a sample, the argmax class is set.

    Args:
        y_pred: (N, C) predicted probabilities.
        thresholds: (C,) per-class thresholds.

    Returns:
        (N, C) binary predictions.
    """
    y_binary = (y_pred >= thresholds[np.newaxis, :]).astype(int)

    # For samples with no predicted class, assign the most confident one
    no_pred_mask = y_binary.sum(axis=1) == 0
    if no_pred_mask.any():
        y_binary[no_pred_mask, np.argmax(y_pred[no_pred_mask], axis=1)] = 1

    return y_binary


def fmax_score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute Fmax: maximum macro F1 across all threshold combinations.

    Searches per-class thresholds to maximize the macro-averaged F1 score.

    Args:
        y_true: (N, C) ground truth.
        y_pred: (N, C) predictions.

    Returns:
        (best_f1, best_thresholds)
    """
    thresholds = find_optimal_thresholds(y_true, y_pred, method="f1")
    y_binary = apply_thresholds(y_pred, thresholds)
    f1 = f1_score(y_true, y_binary, average="macro", zero_division=0)
    return f1, thresholds


def challenge_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta1: float = 2.0,
    beta2: float = 2.0,
) -> Dict[str, float]:
    """Compute PhysioNet/CinC challenge-style metrics.

    Returns F_beta and G_beta (macro-averaged).

    Args:
        y_true: (N, C) binary ground truth.
        y_pred: (N, C) binary predictions.
        beta1: Beta for F_beta_macro.
        beta2: Beta for G_beta_macro.

    Returns:
        Dict with 'F_beta_macro' and 'G_beta_macro'.
    """
    n_classes = y_true.shape[1]

    f_betas = []
    g_betas = []

    for i in range(n_classes):
        tp = ((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        fn = ((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum()
        tn = ((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum()

        # F_beta
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall > 0:
                f_beta = (1 + beta1 ** 2) * precision * recall / (beta1 ** 2 * precision + recall)
            else:
                f_beta = 0.0
        else:
            f_beta = 1.0 if (tp + fn) == 0 else 0.0
        f_betas.append(f_beta)

        # G_beta (PhysioNet/CinC style)
        if tp + fp + fn > 0:
            se = tp / (tp + fn) if (tp + fn) > 0 else 0
            p_plus = tp / (tp + fp) if (tp + fp) > 0 else 0
            if se + p_plus > 0:
                g_beta = (2 * se * p_plus) / (se + p_plus)
            else:
                g_beta = 0.0
        else:
            g_beta = 1.0
        g_betas.append(g_beta)

    return {
        "F_beta_macro": np.mean(f_betas),
        "G_beta_macro": np.mean(g_betas),
    }


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    threshold_method: str = "youden",
) -> Dict[str, any]:
    """Comprehensive evaluation of multi-label predictions.

    Args:
        y_true: (N, C) ground truth multi-hot labels.
        y_pred: (N, C) predicted probabilities.
        class_names: List of class names.
        threshold_method: Method for threshold optimization.

    Returns:
        Dict with all metrics.
    """
    results = {}

    # Threshold-free metrics
    results["macro_auc"] = macro_auc(y_true, y_pred)
    results["per_class_auc"] = per_class_auc(y_true, y_pred, class_names)

    # Threshold-based metrics
    thresholds = find_optimal_thresholds(y_true, y_pred, method=threshold_method)
    y_binary = apply_thresholds(y_pred, thresholds)

    results["thresholds"] = thresholds
    results["fmax"], _ = fmax_score(y_true, y_pred)
    results["f1_macro"] = f1_score(y_true, y_binary, average="macro", zero_division=0)

    # Challenge metrics
    cm = challenge_metrics(y_true, y_binary)
    results.update(cm)

    # Macro AUPRC
    try:
        results["macro_auprc"] = average_precision_score(y_true, y_pred, average="macro")
    except Exception:
        results["macro_auprc"] = float("nan")

    return results
