"""
Evaluation module for the Heart Reader Challenge.

Provides:
- Bootstrap evaluation with confidence intervals
- Comprehensive metric computation
- Results summary table generation
"""

import os
import csv
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from training.metrics import (
    evaluate_all,
    macro_auc,
    per_class_auc,
    find_optimal_thresholds,
    apply_thresholds,
    fmax_score,
    challenge_metrics,
)
from data.preprocessing import SUPERCLASSES


def bootstrap_evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 100,
    class_names: List[str] = None,
    threshold_method: str = "youden",
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate with bootstrap confidence intervals.

    Follows the approach from the PTB-XL benchmarking code:
    - Draw n_bootstrap samples (with replacement)
    - Compute metrics on each
    - Report point estimate, mean, lower (5th), upper (95th) percentiles

    Args:
        y_true: (N, C) ground truth.
        y_pred: (N, C) predictions.
        n_bootstrap: Number of bootstrap samples.
        class_names: List of class names.
        threshold_method: Threshold optimization method.
        seed: Random seed.

    Returns:
        Dict with 'point', 'mean', 'lower', 'upper' for each metric.
    """
    if class_names is None:
        class_names = SUPERCLASSES

    # Point estimate
    point_results = evaluate_all(y_true, y_pred, class_names, threshold_method)

    # Bootstrap samples
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    def _single_bootstrap(i):
        """Evaluate on one bootstrap sample."""
        indices = rng.choice(n_samples, n_samples, replace=True)
        # Ensure every class has at least one positive sample
        for attempt in range(100):
            yt = y_true[indices]
            if yt.sum(axis=0).min() > 0:
                break
            indices = rng.choice(n_samples, n_samples, replace=True)
        yp = y_pred[indices]
        return evaluate_all(yt, yp, class_names, threshold_method)

    print(f"Running {n_bootstrap} bootstrap evaluations...")
    bootstrap_results = []
    for i in range(n_bootstrap):
        bootstrap_results.append(_single_bootstrap(i))

    # Aggregate
    metric_keys = ["macro_auc", "fmax", "f1_macro", "F_beta_macro", "G_beta_macro", "macro_auprc"]
    summary = {}

    for key in metric_keys:
        values = [r[key] for r in bootstrap_results if key in r and not np.isnan(r.get(key, float("nan")))]
        if values:
            summary[key] = {
                "point": point_results.get(key, float("nan")),
                "mean": np.mean(values),
                "lower": np.percentile(values, 5),
                "upper": np.percentile(values, 95),
            }
        else:
            summary[key] = {
                "point": point_results.get(key, float("nan")),
                "mean": float("nan"),
                "lower": float("nan"),
                "upper": float("nan"),
            }

    # Per-class AUC
    for cls_name in class_names:
        cls_key = f"auc_{cls_name}"
        values = [r["per_class_auc"].get(cls_name, float("nan"))
                  for r in bootstrap_results
                  if "per_class_auc" in r]
        values = [v for v in values if not np.isnan(v)]
        summary[cls_key] = {
            "point": point_results["per_class_auc"].get(cls_name, float("nan")),
            "mean": np.mean(values) if values else float("nan"),
            "lower": np.percentile(values, 5) if values else float("nan"),
            "upper": np.percentile(values, 95) if values else float("nan"),
        }

    return summary


def print_results_table(
    summary: Dict[str, Dict[str, float]],
    model_name: str = "Model",
) -> str:
    """Print a formatted results table.

    Args:
        summary: Output from bootstrap_evaluate.
        model_name: Name of the model.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Results for: {model_name}")
    lines.append(f"{'='*80}")
    lines.append(f"{'Metric':<20} {'Point':>8} {'Mean':>8} {'[5%':>8} {'95%]':>8}")
    lines.append("-" * 56)

    for key, vals in summary.items():
        point = vals["point"]
        mean = vals["mean"]
        lower = vals["lower"]
        upper = vals["upper"]
        lines.append(f"{key:<20} {point:>8.4f} {mean:>8.4f} {lower:>8.4f} {upper:>8.4f}")

    lines.append("=" * 80)
    table = "\n".join(lines)
    print(table)
    return table


def save_results(
    summary: Dict[str, Dict[str, float]],
    output_dir: str,
    model_name: str = "model",
):
    """Save evaluation results to CSV.

    Args:
        summary: Output from bootstrap_evaluate.
        output_dir: Directory to save results.
        model_name: Name prefix.
    """
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{model_name}_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "point", "mean", "lower_5pct", "upper_95pct"])
        for key, vals in summary.items():
            writer.writerow([key, vals["point"], vals["mean"], vals["lower"], vals["upper"]])

    print(f"Results saved to {path}")


def generate_summary_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
):
    """Generate a summary comparison table across all models.

    Args:
        all_results: Dict mapping model_name → bootstrap summary.
        output_dir: Directory to save the summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build DataFrame
    rows = []
    for model_name, summary in all_results.items():
        row = {"model": model_name}
        for metric, vals in summary.items():
            row[f"{metric}_point"] = vals["point"]
            row[f"{metric}_ci"] = f"{vals['lower']:.4f}-{vals['upper']:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = os.path.join(output_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)

    # Print markdown table
    print("\n## Summary Table (Test Set — Fold 10)")
    print()

    # Header
    metrics = ["macro_auc", "fmax"]
    header = "| Model |"
    for m in metrics:
        header += f" {m} |"
    print(header)

    sep = "|" + "---|" * (len(metrics) + 1)
    print(sep)

    for model_name, summary in all_results.items():
        row = f"| {model_name} |"
        for m in metrics:
            if m in summary:
                val = summary[m]["point"]
                row += f" {val:.4f} |"
            else:
                row += " N/A |"
        print(row)

    print(f"\nFull results saved to {csv_path}")
