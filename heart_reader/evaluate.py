"""
Heart Reader Challenge — Standalone Evaluation

Loads trained model checkpoints and runs full evaluation on test fold (10).
Produces bootstrap confidence intervals, AUC, F_max, G_beta metrics, 
ROC curves, confusion matrices, and per-class performance tables.

Usage:
    python evaluate.py
    python evaluate.py --model checkpoints/fusion_inception1d_best.pt
    python evaluate.py --backbone xresnet1d101 --device cuda
"""

import argparse
import os
import sys
import json

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import prepare_data
from models.fusion_model import FusionModel, StandaloneModel
from models.ensemble import WeightedEnsemble, collect_predictions
from training.metrics import evaluate_all, find_optimal_thresholds, apply_thresholds
from evaluation.evaluate import (
    bootstrap_evaluate,
    print_results_table,
    save_results,
    generate_summary_table,
)
from evaluation.visualization import generate_all_plots


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(backbone_name: str, cfg: dict, num_features: int = 0):
    """Build model architecture."""
    model_cfg = cfg["model"]
    num_classes = model_cfg.get("num_classes", 5)
    input_channels = model_cfg.get("input_channels", 12)

    if num_features > 0:
        model = FusionModel(
            backbone_name=backbone_name,
            model_cfg=model_cfg,
            num_classes=num_classes,
            input_channels=input_channels,
            num_features=num_features,
        )
    else:
        model = StandaloneModel(
            backbone_name=backbone_name,
            model_cfg=model_cfg,
            num_classes=num_classes,
            input_channels=input_channels,
        )
    return model


def load_model_from_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def evaluate_single_model(model, data, device, output_dir, model_name):
    """Full evaluation pipeline for a single model."""
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device).eval()
    class_names = data.get("class_names", ["NORM", "MI", "STTC", "CD", "HYP"])
    y_test = data["y_test"]

    # Get predictions
    print(f"\nRunning inference on test set ({len(data['test_loader'].dataset)} samples)...")
    test_preds = collect_predictions(model, data["test_loader"], device)

    # Find optimal thresholds on validation set
    print("Finding optimal thresholds on validation set...")
    val_preds = collect_predictions(model, data["val_loader"], device)
    y_val = data["y_val"]
    thresholds = find_optimal_thresholds(y_val, val_preds, method="youden")
    test_preds_binary = apply_thresholds(test_preds, thresholds)

    # Core metrics
    print("Computing metrics...")
    metrics = evaluate_all(y_test, test_preds, test_preds_binary)

    # Bootstrap CIs
    print("Running bootstrap evaluation (this may take a minute)...")
    bs_results = bootstrap_evaluate(
        y_test, test_preds, thresholds=thresholds,
        n_bootstrap=100, class_names=class_names,
    )

    # Print results
    print_results_table(bs_results, class_names=class_names, title=model_name)

    # Save
    save_results(
        metrics=metrics,
        bootstrap_results=bs_results,
        thresholds=thresholds,
        output_dir=output_dir,
        model_name=model_name,
    )

    # Visualizations
    print("Generating plots...")
    generate_all_plots(
        y_true=y_test,
        y_prob=test_preds,
        y_pred=test_preds_binary,
        class_names=class_names,
        output_dir=output_dir,
        model_name=model_name,
        history=None,  # No training history in eval-only mode
    )

    return metrics, test_preds


def evaluate_ensemble(backbone_names, cfg, data, device, output_dir):
    """Evaluate the weighted ensemble of all trained backbones."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = cfg["training"].get("checkpoint_dir", "./checkpoints/")
    num_features = data["num_features"]
    class_names = data.get("class_names", ["NORM", "MI", "STTC", "CD", "HYP"])

    # Collect val predictions from each model
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION")
    print("=" * 70)

    val_preds_list = []
    test_preds_list = []
    available_names = []

    for backbone_name in backbone_names:
        model = build_model(backbone_name, cfg, num_features)
        ckpt_path = None
        for pattern in [f"fusion_{backbone_name}_best.pt", f"{backbone_name}_best.pt"]:
            p = os.path.join(checkpoint_dir, pattern)
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            print(f"  Skipping {backbone_name}: no checkpoint found")
            continue

        model = load_model_from_checkpoint(model, ckpt_path, device)
        model = model.to(device).eval()

        val_preds = collect_predictions(model, data["val_loader"], device)
        test_preds = collect_predictions(model, data["test_loader"], device)
        val_preds_list.append(val_preds)
        test_preds_list.append(test_preds)
        available_names.append(backbone_name)
        print(f"  Loaded {backbone_name}")

    if len(available_names) < 2:
        print("Not enough models for ensemble. Need at least 2.")
        return None, None

    # Optimize ensemble weights
    y_val = data["y_val"]
    y_test = data["y_test"]
    ensemble = WeightedEnsemble(n_models=len(available_names))
    ensemble.optimize_weights(val_preds_list, y_val)

    print(f"\nEnsemble weights:")
    for name, w in zip(available_names, ensemble.weights):
        print(f"  {name}: {w:.4f}")

    # Ensemble predictions
    ens_val_preds = ensemble.predict(val_preds_list)
    ens_test_preds = ensemble.predict(test_preds_list)

    # Thresholds & metrics
    thresholds = find_optimal_thresholds(y_val, ens_val_preds, method="youden")
    ens_test_binary = apply_thresholds(ens_test_preds, thresholds)
    metrics = evaluate_all(y_test, ens_test_preds, ens_test_binary)

    # Bootstrap
    bs_results = bootstrap_evaluate(
        y_test, ens_test_preds, thresholds=thresholds,
        n_bootstrap=100, class_names=class_names,
    )
    print_results_table(bs_results, class_names=class_names, title="Weighted Ensemble")

    save_results(
        metrics=metrics,
        bootstrap_results=bs_results,
        thresholds=thresholds,
        output_dir=output_dir,
        model_name="ensemble",
    )

    generate_all_plots(
        y_true=y_test,
        y_prob=ens_test_preds,
        y_pred=ens_test_binary,
        class_names=class_names,
        output_dir=output_dir,
        model_name="ensemble",
        history=None,
    )

    # Summary table
    generate_summary_table(output_dir, model_names=available_names + ["ensemble"])

    return metrics, ens_test_preds


def main():
    parser = argparse.ArgumentParser(description="Heart Reader — Evaluation")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit checkpoint path (single model eval)")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["inception1d", "xresnet1d101", "se_resnet1d"],
                        help="Backbone for single-model eval")
    parser.add_argument("--ensemble", action="store_true",
                        help="Evaluate weighted ensemble of all available models")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load data
    print("Loading data...")
    data = prepare_data(cfg)

    eval_dir = cfg.get("evaluation", {}).get("output_dir", "./results/")
    os.makedirs(eval_dir, exist_ok=True)

    if args.ensemble:
        all_backbones = ["inception1d", "xresnet1d101", "se_resnet1d"]
        evaluate_ensemble(all_backbones, cfg, data, device, eval_dir)
    elif args.backbone or args.model:
        backbone_name = args.backbone or "inception1d"
        model = build_model(backbone_name, cfg, data["num_features"])
        if args.model:
            model = load_model_from_checkpoint(model, args.model, device)
        else:
            ckpt_dir = cfg["training"].get("checkpoint_dir", "./checkpoints/")
            for pattern in [f"fusion_{backbone_name}_best.pt", f"{backbone_name}_best.pt"]:
                p = os.path.join(ckpt_dir, pattern)
                if os.path.exists(p):
                    model = load_model_from_checkpoint(model, p, device)
                    break
        evaluate_single_model(model, data, device,
                             os.path.join(eval_dir, backbone_name), backbone_name)
    else:
        # Evaluate all available individual models + ensemble
        all_backbones = ["inception1d", "xresnet1d101", "se_resnet1d"]
        ckpt_dir = cfg["training"].get("checkpoint_dir", "./checkpoints/")

        for backbone_name in all_backbones:
            model = build_model(backbone_name, cfg, data["num_features"])
            loaded = False
            for pattern in [f"fusion_{backbone_name}_best.pt", f"{backbone_name}_best.pt"]:
                p = os.path.join(ckpt_dir, pattern)
                if os.path.exists(p):
                    model = load_model_from_checkpoint(model, p, device)
                    loaded = True
                    break
            if loaded:
                evaluate_single_model(
                    model, data, device,
                    os.path.join(eval_dir, backbone_name), backbone_name,
                )

        # Ensemble
        evaluate_ensemble(all_backbones, cfg, data, device, eval_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
