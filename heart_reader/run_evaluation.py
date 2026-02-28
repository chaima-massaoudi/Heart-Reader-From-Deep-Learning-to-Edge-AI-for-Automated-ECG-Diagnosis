"""
Quick standalone evaluation — load best checkpoint, evaluate on test fold (10).
Produces test-set metrics, per-class AUC, bootstrap CIs, and exports ONNX.
"""

import os
import sys
import json
import pickle
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import prepare_data
from models.fusion_model import FusionModel
from training.trainer import Trainer
from training.metrics import (
    evaluate_all,
    macro_auc,
    per_class_auc,
    find_optimal_thresholds,
    apply_thresholds,
    fmax_score,
)
from evaluation.evaluate import bootstrap_evaluate, print_results_table, save_results
from edge.quantize import dynamic_quantize, measure_model_size, measure_inference_time
from edge.prune import apply_structured_pruning, make_pruning_permanent, compute_sparsity
from edge.export_tflite import export_to_onnx


def main():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Use fewer workers to avoid conflicts with running training
    cfg["training"]["num_workers"] = 0
    cfg["training"]["batch_size"] = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Prepare data
    print("Loading data...")
    data = prepare_data(cfg)
    num_classes = data["num_classes"]
    num_features = data["num_features"]
    print(f"Data ready: {num_classes} classes, {num_features} features")
    print(f"Test samples: {len(data['y_test'])}, Val samples: {len(data['y_val'])}")

    # 2) Build model & load checkpoint
    model_cfg = cfg["model"]
    backbone_name = "inception1d"
    model = FusionModel(
        backbone_name=backbone_name,
        model_cfg=model_cfg,
        num_classes=num_classes,
        input_channels=model_cfg.get("input_channels", 12),
        num_features=num_features,
    )

    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "fusion_inception1d_best.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', '?')}, val_auc={checkpoint.get('val_macro_auc', '?')})")

    # 3) Predict on val and test
    print("\nRunning inference...")
    trainer = Trainer(model, cfg, device, model_name="fusion_inception1d")

    val_preds = trainer.predict(data["val_loader"])
    test_preds = trainer.predict(data["test_loader"])

    val_auc = macro_auc(data["y_val"], val_preds)
    test_auc = macro_auc(data["y_test"], test_preds)
    print(f"Val AUC:  {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # 4) Per-class AUC
    pca = per_class_auc(data["y_test"], test_preds, data["class_names"])
    print("\nPer-class Test AUC:")
    for cls, auc_val in pca.items():
        print(f"  {cls}: {auc_val:.4f}")

    # 5) Optimal thresholds & F1
    thresholds = find_optimal_thresholds(data["y_val"], val_preds)
    test_binary = apply_thresholds(test_preds, thresholds)
    from sklearn.metrics import f1_score, classification_report
    f1_macro = f1_score(data["y_test"], test_binary, average="macro", zero_division=0)
    print(f"\nMacro F1-Score (Youden thresholds): {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        data["y_test"], test_binary,
        target_names=data["class_names"], zero_division=0
    ))

    # 6) Bootstrap evaluation
    print("\nBootstrap evaluation (100 samples)...")
    summary = bootstrap_evaluate(
        data["y_test"], test_preds,
        n_bootstrap=100,
        class_names=data["class_names"],
        threshold_method="youden",
    )
    table_str = print_results_table(summary, "Fusion InceptionTime1D")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_results(summary, results_dir, "fusion_inception1d")

    np.save(os.path.join(results_dir, "fusion_inception1d_test_preds.npy"), test_preds)
    np.save(os.path.join(results_dir, "fusion_inception1d_val_preds.npy"), val_preds)

    # 7) Model size & param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    original_size = measure_model_size(model)
    print(f"\nModel stats:")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Model size:       {original_size:.2f} MB")

    # 8) Inference latency
    print("\nBenchmarking inference latency (CPU)...")
    model_cpu = model.cpu().eval()
    latency = measure_inference_time(model_cpu, input_shape=(1, 12, 1000), n_runs=50, warmup=5, device="cpu")
    print(f"  CPU latency: {latency['mean_ms']:.1f} +/- {latency['std_ms']:.1f} ms")

    if torch.cuda.is_available():
        print("Benchmarking inference latency (GPU)...")
        model_gpu = model.to("cuda").eval()
        lat_gpu = measure_inference_time(model_gpu, input_shape=(1, 12, 1000), n_runs=50, warmup=10, device="cuda")
        print(f"  GPU latency: {lat_gpu['mean_ms']:.1f} +/- {lat_gpu['std_ms']:.1f} ms")
        model = model.cpu()

    # 9) Edge: Pruning
    print("\n--- Edge Deployment: Pruning ---")
    import copy
    pruned_model = copy.deepcopy(model).cpu().eval()
    pruned_model = apply_structured_pruning(pruned_model, amount=0.5)
    sparsity = compute_sparsity(pruned_model)
    print(f"  Sparsity: {sparsity['sparsity_pct']:.1f}%")
    make_pruning_permanent(pruned_model)
    pruned_size = measure_model_size(pruned_model)
    print(f"  Pruned model size: {pruned_size:.2f} MB")

    # 10) Edge: Dynamic Quantization
    print("\n--- Edge Deployment: Dynamic Quantization ---")
    quant_model = dynamic_quantize(copy.deepcopy(model).cpu().eval())
    quant_size = measure_model_size(quant_model)
    print(f"  Quantized model size: {quant_size:.2f} MB")

    quant_latency = measure_inference_time(quant_model, input_shape=(1, 12, 1000), n_runs=50, warmup=5, device="cpu")
    print(f"  Quantized CPU latency: {quant_latency['mean_ms']:.1f} +/- {quant_latency['std_ms']:.1f} ms")

    # 11) Export ONNX
    print("\n--- Edge Deployment: ONNX Export ---")
    onnx_dir = os.path.join(results_dir, "edge")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = export_to_onnx(
        copy.deepcopy(model).cpu().eval(),
        save_path=os.path.join(onnx_dir, "fusion_inception1d.onnx"),
        input_shape=(1, 12, 1000),
    )

    # 12) Save comprehensive report data as JSON
    report_data = {
        "model": "Fusion InceptionTime1D + PTB-XL+ Features",
        "backbone": "InceptionTime1D with SE attention",
        "num_features": num_features,
        "num_classes": num_classes,
        "val_macro_auc": float(val_auc),
        "test_macro_auc": float(test_auc),
        "test_macro_f1": float(f1_macro),
        "per_class_auc": {k: float(v) for k, v in pca.items()},
        "bootstrap_summary": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in summary.items()
        },
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": float(original_size),
        "pruned_size_mb": float(pruned_size),
        "quantized_size_mb": float(quant_size),
        "sparsity_pct": float(sparsity["sparsity_pct"]),
        "cpu_latency_ms": float(latency["mean_ms"]),
        "quantized_cpu_latency_ms": float(quant_latency["mean_ms"]),
        "compression_ratio": float(original_size / quant_size) if quant_size > 0 else 0,
    }

    with open(os.path.join(results_dir, "evaluation_report.json"), "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport data saved to {os.path.join(results_dir, 'evaluation_report.json')}")
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
