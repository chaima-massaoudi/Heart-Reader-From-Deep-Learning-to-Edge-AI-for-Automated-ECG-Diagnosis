"""
Visualization utilities for ECG classification results.

Produces:
- ROC curves (per-class + micro/macro average)
- Confusion matrix heatmap
- Training/validation loss curves
- Per-class performance bar chart
- Grad-CAM attention maps on ECG signals
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from data.preprocessing import SUPERCLASSES


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    title: str = "ROC Curves",
):
    """Plot ROC curves for each class plus macro-average.

    Args:
        y_true: (N, C) ground truth.
        y_pred: (N, C) predictions.
        class_names: List of class names.
        save_path: Path to save the figure.
        title: Plot title.
    """
    if class_names is None:
        class_names = SUPERCLASSES
    n_classes = y_true.shape[1]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    # Per-class ROC
    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_true[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc_val = roc_auc_score(y_true[:, i], y_pred[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")

    # Macro-average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_true[:, i], y_pred[:, i])[0]
        for i in range(n_classes)
        if y_true[:, i].sum() > 0
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    count = 0
    for i in range(n_classes):
        if y_true[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        count += 1
    mean_tpr /= max(count, 1)
    macro_auc_val = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    ax.plot(all_fpr, mean_tpr, "k--", lw=3, label=f"Macro-avg (AUC={macro_auc_val:.3f})")

    # Diagonal
    ax.plot([0, 1], [0, 1], "gray", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    title: str = "Confusion Matrix",
):
    """Plot per-class confusion matrix for multi-label classification.

    Shows a separate 2×2 confusion matrix for each class.

    Args:
        y_true: (N, C) binary ground truth.
        y_pred_binary: (N, C) binary predictions.
        class_names: List of class names.
        save_path: Path to save the figure.
    """
    if class_names is None:
        class_names = SUPERCLASSES
    n_classes = y_true.shape[1]

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for i, (name, ax) in enumerate(zip(class_names, axes)):
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_curves(
    history: Dict[str, list],
    save_path: str = None,
    title: str = "Training Curves",
):
    """Plot training and validation loss + AUC curves.

    Args:
        history: Dict with 'epoch', 'train_loss', 'val_loss', 'val_macro_auc' lists.
        save_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history["epoch"]

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", lw=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # AUC
    ax2.plot(epochs, history["val_macro_auc"], "g-", label="Val Macro AUC", lw=2)
    best_epoch = epochs[np.argmax(history["val_macro_auc"])]
    best_auc = max(history["val_macro_auc"])
    ax2.axvline(best_epoch, color="gray", ls="--", alpha=0.5, label=f"Best: {best_auc:.4f} @ E{best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro AUC")
    ax2.set_title("Validation AUC")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    plt.close()


def plot_class_performance(
    per_class_metrics: Dict[str, float],
    metric_name: str = "AUC",
    save_path: str = None,
    title: str = "Per-Class Performance",
):
    """Bar chart of per-class metric values.

    Args:
        per_class_metrics: Dict mapping class_name → value.
        metric_name: Name of the metric (for y-axis label).
        save_path: Path to save.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = list(per_class_metrics.keys())
    values = list(per_class_metrics.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    bars = ax.bar(classes, values, color=colors, edgecolor="black", lw=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=10)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Class performance chart saved to {save_path}")
    plt.close()


def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_binary: np.ndarray,
    history: Dict[str, list],
    per_class_auc_dict: Dict[str, float],
    output_dir: str,
    model_name: str = "model",
):
    """Generate all visualization plots.

    Args:
        y_true: Ground truth labels.
        y_pred: Prediction probabilities.
        y_pred_binary: Binarized predictions.
        history: Training history dict.
        per_class_auc_dict: Per-class AUC values.
        output_dir: Base output directory.
        model_name: Model name for titles.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_roc_curves(
        y_true, y_pred,
        save_path=os.path.join(output_dir, f"{model_name}_roc_curves.png"),
        title=f"ROC Curves — {model_name}",
    )

    plot_confusion_matrix(
        y_true, y_pred_binary,
        save_path=os.path.join(output_dir, f"{model_name}_confusion_matrix.png"),
        title=f"Confusion Matrix — {model_name}",
    )

    if history:
        plot_training_curves(
            history,
            save_path=os.path.join(output_dir, f"{model_name}_training_curves.png"),
            title=f"Training Curves — {model_name}",
        )

    plot_class_performance(
        per_class_auc_dict,
        metric_name="AUC",
        save_path=os.path.join(output_dir, f"{model_name}_class_performance.png"),
        title=f"Per-Class AUC — {model_name}",
    )
