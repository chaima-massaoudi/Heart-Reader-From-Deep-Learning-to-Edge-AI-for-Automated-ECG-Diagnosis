"""
Heart Reader Challenge — Edge Deployment Export

Runs the full model compression and export pipeline:
1. Load best trained model
2. Structured pruning → fine-tune → save
3. Dynamic quantization → save
4. ONNX export → TFLite conversion
5. Benchmark all variants (size, speed)

Usage:
    python export.py                               # Export with default config
    python export.py --model checkpoints/fusion_inception1d_best.pt
    python export.py --backbone inception1d
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import prepare_data
from models.fusion_model import FusionModel, StandaloneModel
from training.trainer import Trainer
from training.metrics import macro_auc
from edge.prune import (
    apply_structured_pruning,
    make_pruning_permanent,
    compute_sparsity,
    save_pruned_model,
)
from edge.quantize import (
    dynamic_quantize,
    measure_model_size,
    measure_inference_time,
    save_quantized_model,
)
from edge.export_tflite import export_to_onnx, onnx_to_tflite, verify_tflite_model
from edge.benchmark import (
    benchmark_pytorch_model,
    benchmark_tflite_model,
    generate_benchmark_report,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(backbone_name: str, cfg: dict, num_features: int = 0):
    """Build model architecture (without trained weights)."""
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


def main():
    parser = argparse.ArgumentParser(description="Heart Reader — Edge Export")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--backbone", type=str, default="inception1d",
                        choices=["inception1d", "xresnet1d101", "se_resnet1d"],
                        help="Backbone architecture")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip post-pruning fine-tuning")
    args = parser.parse_args()

    cfg = load_config(args.config)
    edge_cfg = cfg["edge"]
    output_dir = edge_cfg.get("output_dir", "./edge_models/")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Step 0: Load trained model ──
    print("=" * 70)
    print("EDGE DEPLOYMENT PIPELINE")
    print("=" * 70)

    # Determine model architecture
    backbone_name = args.backbone

    # Try to load data for num_features
    try:
        data = prepare_data(cfg)
        num_features = data["num_features"]
    except Exception:
        print("Could not load data, assuming 0 structured features.")
        data = None
        num_features = 0

    model = build_model(backbone_name, cfg, num_features)

    # Load weights if checkpoint provided
    if args.model and os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from {args.model}")
    else:
        # Try to find the best checkpoint automatically
        checkpoint_dir = cfg["training"].get("checkpoint_dir", "./checkpoints/")
        for name_pattern in [f"fusion_{backbone_name}_best.pt", f"{backbone_name}_best.pt"]:
            ckpt_path = os.path.join(checkpoint_dir, name_pattern)
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Auto-loaded weights from {ckpt_path}")
                break
        else:
            print("WARNING: No trained weights found. Using random initialization.")

    model = model.to(device).eval()

    # Benchmark list
    benchmarks = []

    # ── Baseline: Original model ──
    print(f"\n{'─'*50}")
    print("BASELINE: Original Model")
    print(f"{'─'*50}")
    orig_size = measure_model_size(model, os.path.join(output_dir, "original.pt"))
    orig_latency = measure_inference_time(model.cpu())
    model = model.to(device)
    print(f"  Size: {orig_size:.2f} MB")
    print(f"  CPU latency: {orig_latency['mean_ms']:.2f} ± {orig_latency['std_ms']:.2f} ms")
    benchmarks.append({"name": "Original", "size_mb": orig_size, **{f"cpu_{k}": v for k, v in orig_latency.items()}})

    # ── Step 1: Pruning ──
    if edge_cfg.get("pruning", {}).get("enabled", True):
        print(f"\n{'─'*50}")
        print("STEP 1: Structured Pruning")
        print(f"{'─'*50}")

        prune_amount = edge_cfg["pruning"].get("amount", 0.5)
        model = apply_structured_pruning(model, amount=prune_amount)

        sparsity = compute_sparsity(model)
        print(f"  Sparsity: {sparsity['sparsity_pct']:.1f}%")

        # Fine-tune after pruning
        if not args.skip_finetune and data is not None:
            print("  Fine-tuning post-pruning...")
            finetune_epochs = edge_cfg["pruning"].get("finetune_epochs", 10)
            # Create a temporary config with fewer epochs
            ft_cfg = dict(cfg)
            ft_cfg["training"] = dict(cfg["training"])
            ft_cfg["training"]["epochs"] = finetune_epochs
            ft_cfg["training"]["lr"] = cfg["training"]["lr"] / 10  # Lower LR

            ft_trainer = Trainer(model, ft_cfg, device, model_name=f"{backbone_name}_pruned")
            ft_trainer.fit(data["train_loader"], data["val_loader"])

        make_pruning_permanent(model)
        pruned_path, pruned_size = save_pruned_model(model, output_dir, f"{backbone_name}_pruned")

        pruned_latency = measure_inference_time(model.cpu())
        model = model.to(device)
        print(f"  Pruned size: {pruned_size:.2f} MB ({orig_size/pruned_size:.1f}x compression)")
        print(f"  Pruned latency: {pruned_latency['mean_ms']:.2f} ms")
        benchmarks.append({"name": "Pruned", "size_mb": pruned_size,
                          **{f"cpu_{k}": v for k, v in pruned_latency.items()}})

    # ── Step 2: Quantization ──
    if edge_cfg.get("quantization", {}).get("enabled", True):
        print(f"\n{'─'*50}")
        print("STEP 2: Dynamic Quantization")
        print(f"{'─'*50}")

        model_cpu = model.cpu().eval()
        quantized_model = dynamic_quantize(model_cpu)
        quant_path, quant_size = save_quantized_model(
            quantized_model, output_dir, f"{backbone_name}_quantized"
        )

        quant_latency = measure_inference_time(quantized_model)
        print(f"  Quantized size: {quant_size:.2f} MB ({orig_size/quant_size:.1f}x compression)")
        print(f"  Quantized latency: {quant_latency['mean_ms']:.2f} ms")
        benchmarks.append({"name": "Quantized", "size_mb": quant_size,
                          **{f"cpu_{k}": v for k, v in quant_latency.items()}})

    # ── Step 3: ONNX + TFLite Export ──
    if edge_cfg.get("tflite", {}).get("enabled", True):
        print(f"\n{'─'*50}")
        print("STEP 3: ONNX → TFLite Export")
        print(f"{'─'*50}")

        # For ONNX export we need a clean model (non-quantized)
        # Reload from pruned or original
        export_model = model.cpu().eval()

        input_shape = (1, cfg["model"]["input_channels"], cfg["model"]["input_length"])

        # For fusion models, we need to handle the dual-input
        # For export, we'll use the signal-only path
        try:
            onnx_path = export_to_onnx(
                export_model.model if hasattr(export_model, 'model') else export_model,
                input_shape=input_shape,
                save_path=os.path.join(output_dir, f"{backbone_name}.onnx"),
            )

            tflite_path = os.path.join(output_dir, f"{backbone_name}.tflite")
            tflite_result = onnx_to_tflite(
                onnx_path, tflite_path,
                optimize=edge_cfg["tflite"].get("optimize", True),
                full_integer=edge_cfg["tflite"].get("full_integer", False),
            )

            if tflite_result:
                verify_tflite_model(tflite_path, input_shape=input_shape)
                tflite_bm = benchmark_tflite_model(tflite_path, input_shape=input_shape)
                benchmarks.append(tflite_bm)
        except Exception as e:
            print(f"  TFLite export failed: {e}")
            print("  ONNX model is still available for deployment.")

    # ── Benchmark Report ──
    print(f"\n{'─'*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'─'*50}")
    generate_benchmark_report(benchmarks, output_dir)

    # Evaluate accuracy at each stage if data is available
    if data is not None:
        print(f"\n{'─'*50}")
        print("ACCURACY AT EACH STAGE")
        print(f"{'─'*50}")

        model = model.to(device).eval()
        from models.ensemble import collect_predictions
        test_preds = collect_predictions(model, data["test_loader"], device)
        test_auc = macro_auc(data["y_test"], test_preds)
        print(f"  Final model test AUC: {test_auc:.4f}")

    print(f"\nAll edge models saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
