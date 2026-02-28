"""
Heart Reader Challenge — Main Training Entry Point

Orchestrates the full pipeline:
1. Load and prepare PTB-XL data (100Hz, 5 superclasses)
2. Train 3 backbone models (InceptionTime, XResNet1d101, SE-ResNet1d)
   each with optional PTB-XL+ feature fusion
3. Optimize ensemble weights on validation set
4. Evaluate on test fold (fold 10) with bootstrap CIs
5. Generate all plots and save results

Usage:
    python train.py                         # Train with default config
    python train.py --config configs/custom.yaml
    python train.py --backbone inception1d  # Train single model
"""

import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import prepare_data
from models.fusion_model import FusionModel, StandaloneModel
from models.ensemble import WeightedEnsemble, collect_predictions
from training.trainer import Trainer
from training.metrics import evaluate_all, macro_auc
from evaluation.evaluate import (
    bootstrap_evaluate,
    print_results_table,
    save_results,
    generate_summary_table,
)
from evaluation.visualization import (
    generate_all_plots,
    plot_roc_curves,
    plot_training_curves,
)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_single_model(
    backbone_name: str,
    cfg: dict,
    data: dict,
    device: torch.device,
) -> dict:
    """Train a single model (with optional fusion).

    Args:
        backbone_name: One of "inception1d", "xresnet1d101", "se_resnet1d".
        cfg: Full config dict.
        data: Output from prepare_data().
        device: Compute device.

    Returns:
        Dict with model, trainer, history, predictions.
    """
    model_cfg = cfg["model"]
    num_classes = data["num_classes"]
    num_features = data["num_features"]

    print(f"\n{'#'*70}")
    print(f"# Training: {backbone_name}")
    print(f"# Classes: {num_classes}, Features: {num_features}")
    print(f"{'#'*70}")

    # Build model
    if num_features > 0:
        model = FusionModel(
            backbone_name=backbone_name,
            model_cfg=model_cfg,
            num_classes=num_classes,
            input_channels=model_cfg.get("input_channels", 12),
            num_features=num_features,
        )
        model_name = f"fusion_{backbone_name}"
    else:
        model = StandaloneModel(
            backbone_name=backbone_name,
            model_cfg=model_cfg,
            num_classes=num_classes,
            input_channels=model_cfg.get("input_channels", 12),
        )
        model_name = backbone_name

    # Train
    trainer = Trainer(model, cfg, device, model_name=model_name)
    history = trainer.fit(
        data["train_loader"],
        data["val_loader"],
        y_val=data["y_val"],
    )

    # Collect predictions on all splits
    val_preds = trainer.predict(data["val_loader"])
    test_preds = trainer.predict(data["test_loader"])

    # Quick evaluation
    val_auc = macro_auc(data["y_val"], val_preds)
    test_auc = macro_auc(data["y_test"], test_preds)
    print(f"\n{model_name} — Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")

    return {
        "model": trainer.model,
        "trainer": trainer,
        "history": history,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "model_name": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Heart Reader Challenge — Train")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        choices=["inception1d", "xresnet1d101", "se_resnet1d", "ensemble"],
        help="Train a specific backbone (default: from config)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu). Default: auto-detect.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Set seed
    set_seed(cfg.get("seed", 42))

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Backbone choice
    backbone = args.backbone or cfg["model"].get("backbone", "ensemble")

    # Prepare data
    data = prepare_data(cfg)

    # Create output dirs
    results_dir = cfg["evaluation"].get("results_dir", "./results/")
    os.makedirs(results_dir, exist_ok=True)

    start_time = time.time()

    if backbone == "ensemble":
        # Train all three backbone models
        backbones = ["inception1d", "xresnet1d101", "se_resnet1d"]
        all_results = {}
        all_val_preds = []
        all_test_preds = []

        for bb_name in backbones:
            result = train_single_model(bb_name, cfg, data, device)
            all_results[result["model_name"]] = result
            all_val_preds.append(result["val_preds"])
            all_test_preds.append(result["test_preds"])

            # Save individual predictions
            np.save(os.path.join(results_dir, f"{result['model_name']}_val_preds.npy"), result["val_preds"])
            np.save(os.path.join(results_dir, f"{result['model_name']}_test_preds.npy"), result["test_preds"])

            # Plot training curves
            plot_training_curves(
                result["history"],
                save_path=os.path.join(results_dir, f"{result['model_name']}_training_curves.png"),
                title=f"Training Curves — {result['model_name']}",
            )

        # Optimize ensemble weights
        print(f"\n{'='*70}")
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print(f"{'='*70}")

        ensemble_cfg = cfg["model"].get("ensemble", {})
        ensemble = WeightedEnsemble(
            method=ensemble_cfg.get("method", "weighted")
        )
        ensemble.fit_weights(all_val_preds, data["y_val"])

        # Ensemble predictions
        ensemble_test_preds = ensemble.predict(all_test_preds)
        ensemble_val_preds = ensemble.predict(all_val_preds)

        # Save ensemble predictions
        np.save(os.path.join(results_dir, "ensemble_val_preds.npy"), ensemble_val_preds)
        np.save(os.path.join(results_dir, "ensemble_test_preds.npy"), ensemble_test_preds)

        # Save ensemble weights
        with open(os.path.join(results_dir, "ensemble_weights.pkl"), "wb") as f:
            pickle.dump({"weights": ensemble.weights, "backbones": backbones}, f)

        # Evaluate ensemble
        print(f"\n{'='*70}")
        print("FINAL EVALUATION — ENSEMBLE")
        print(f"{'='*70}")

        eval_cfg = cfg["evaluation"]
        if eval_cfg.get("bootstrap", True):
            summary = bootstrap_evaluate(
                data["y_test"], ensemble_test_preds,
                n_bootstrap=eval_cfg.get("n_bootstrap", 100),
                class_names=data["class_names"],
                threshold_method=eval_cfg.get("threshold_method", "youden"),
            )
        else:
            point_results = evaluate_all(data["y_test"], ensemble_test_preds, data["class_names"])
            summary = {k: {"point": v, "mean": v, "lower": v, "upper": v}
                      for k, v in point_results.items()
                      if isinstance(v, (int, float))}

        print_results_table(summary, "Weighted Ensemble")
        save_results(summary, results_dir, "ensemble")

        # Visualizations
        from training.metrics import find_optimal_thresholds, apply_thresholds
        thresholds = find_optimal_thresholds(data["y_val"], ensemble_val_preds)
        test_binary = apply_thresholds(ensemble_test_preds, thresholds)
        per_class_auc_dict = {
            name: float(summary.get(f"auc_{name}", {}).get("point", 0))
            for name in data["class_names"]
        }

        generate_all_plots(
            data["y_test"], ensemble_test_preds, test_binary,
            history={},  # No single training history for ensemble
            per_class_auc_dict=per_class_auc_dict,
            output_dir=results_dir,
            model_name="ensemble",
        )

        # Also evaluate and save individual model results
        all_summaries = {"ensemble": summary}
        for name, result in all_results.items():
            ind_summary = bootstrap_evaluate(
                data["y_test"], result["test_preds"],
                n_bootstrap=eval_cfg.get("n_bootstrap", 100),
                class_names=data["class_names"],
            )
            print_results_table(ind_summary, name)
            save_results(ind_summary, results_dir, name)
            all_summaries[name] = ind_summary

        generate_summary_table(all_summaries, results_dir)

    else:
        # Train single model
        result = train_single_model(backbone, cfg, data, device)

        # Evaluate
        eval_cfg = cfg["evaluation"]
        if eval_cfg.get("bootstrap", True):
            summary = bootstrap_evaluate(
                data["y_test"], result["test_preds"],
                n_bootstrap=eval_cfg.get("n_bootstrap", 100),
                class_names=data["class_names"],
                threshold_method=eval_cfg.get("threshold_method", "youden"),
            )
        else:
            point_results = evaluate_all(data["y_test"], result["test_preds"], data["class_names"])
            summary = {k: {"point": v, "mean": v, "lower": v, "upper": v}
                      for k, v in point_results.items()
                      if isinstance(v, (int, float))}

        print_results_table(summary, result["model_name"])
        save_results(summary, results_dir, result["model_name"])

        # Save predictions
        np.save(os.path.join(results_dir, f"{result['model_name']}_test_preds.npy"), result["test_preds"])

        # Visualizations
        from training.metrics import find_optimal_thresholds, apply_thresholds, per_class_auc
        thresholds = find_optimal_thresholds(data["y_val"], result["val_preds"])
        test_binary = apply_thresholds(result["test_preds"], thresholds)
        pca = per_class_auc(data["y_test"], result["test_preds"], data["class_names"])

        generate_all_plots(
            data["y_test"], result["test_preds"], test_binary,
            history=result["history"],
            per_class_auc_dict=pca,
            output_dir=results_dir,
            model_name=result["model_name"],
        )

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")
    print("Done!")


if __name__ == "__main__":
    main()
