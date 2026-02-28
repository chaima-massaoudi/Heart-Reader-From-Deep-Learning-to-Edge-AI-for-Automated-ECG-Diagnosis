"""Quick evaluation on CPU — avoids GPU conflict with running training."""
import os, sys, json, copy, time, traceback
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import prepare_data
from models.fusion_model import FusionModel
from training.trainer import Trainer
from training.metrics import (evaluate_all, macro_auc, per_class_auc,
                               find_optimal_thresholds, apply_thresholds, fmax_score)
from evaluation.evaluate import bootstrap_evaluate, print_results_table, save_results
from edge.quantize import dynamic_quantize, measure_model_size, measure_inference_time
from edge.prune import apply_structured_pruning, make_pruning_permanent, compute_sparsity
from edge.export_tflite import export_to_onnx
from sklearn.metrics import f1_score, classification_report

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["num_workers"] = 0
    cfg["training"]["batch_size"] = 64

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    data = prepare_data(cfg)
    nc = data["num_classes"]
    nf = data["num_features"]
    y_test = data["y_test"]
    y_val = data["y_val"]
    cls_names = data["class_names"]
    print(f"Data: {nc} classes, {nf} features, Test={len(y_test)}, Val={len(y_val)}")

    # Model
    model_cfg = cfg["model"]
    model = FusionModel("inception1d", model_cfg, nc, 12, nf)
    ckpt = torch.load("checkpoints/fusion_inception1d_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()
    print(f"Checkpoint epoch: {ckpt.get('epoch','?')}")

    # Predict
    print("Running inference (CPU)...")
    trainer = Trainer(model, cfg, device, model_name="eval")
    val_preds = trainer.predict(data["val_loader"])
    test_preds = trainer.predict(data["test_loader"])

    val_auc = macro_auc(y_val, val_preds)
    test_auc = macro_auc(y_test, test_preds)
    print(f"Val AUC:  {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    pca = per_class_auc(y_test, test_preds, cls_names)
    for c, v in pca.items():
        print(f"  {c}: {v:.4f}")

    thresholds = find_optimal_thresholds(y_val, val_preds)
    test_binary = apply_thresholds(test_preds, thresholds)
    f1m = f1_score(y_test, test_binary, average="macro", zero_division=0)
    print(f"Macro F1: {f1m:.4f}")
    print(classification_report(y_test, test_binary, target_names=cls_names, zero_division=0))

    # Bootstrap
    print("Bootstrap (100)...")
    summary = bootstrap_evaluate(y_test, test_preds, n_bootstrap=100, class_names=cls_names)
    table = print_results_table(summary, "Fusion InceptionTime1D")

    # Save
    rd = "./results"
    os.makedirs(rd, exist_ok=True)
    save_results(summary, rd, "fusion_inception1d")
    np.save(os.path.join(rd, "fusion_inception1d_test_preds.npy"), test_preds)
    np.save(os.path.join(rd, "fusion_inception1d_val_preds.npy"), val_preds)

    # Model stats
    tp = sum(p.numel() for p in model.parameters())
    orig_sz = measure_model_size(model)
    print(f"Params: {tp:,}, Size: {orig_sz:.2f} MB")

    lat = measure_inference_time(model, (1,12,1000), n_runs=30, warmup=3, device="cpu")
    print(f"CPU latency: {lat['mean_ms']:.1f} ms")

    # Pruning
    pm = copy.deepcopy(model).eval()
    pm = apply_structured_pruning(pm, amount=0.5)
    sp = compute_sparsity(pm)
    make_pruning_permanent(pm)
    psz = measure_model_size(pm)
    print(f"Pruned: sparsity={sp['sparsity_pct']:.1f}%, size={psz:.2f} MB")

    # Quantization
    qm = dynamic_quantize(copy.deepcopy(model).eval())
    qsz = measure_model_size(qm)
    qlat = measure_inference_time(qm, (1,12,1000), n_runs=30, warmup=3, device="cpu")
    print(f"Quantized: size={qsz:.2f} MB, latency={qlat['mean_ms']:.1f} ms")

    # ONNX
    onnx_dir = os.path.join(rd, "edge")
    os.makedirs(onnx_dir, exist_ok=True)
    export_to_onnx(copy.deepcopy(model).eval(), os.path.join(onnx_dir, "fusion_inception1d.onnx"))

    # JSON report
    report = {
        "model": "Fusion InceptionTime1D + PTB-XL+ Features",
        "val_macro_auc": round(float(val_auc), 4),
        "test_macro_auc": round(float(test_auc), 4),
        "test_macro_f1": round(float(f1m), 4),
        "per_class_auc": {k: round(float(v), 4) for k, v in pca.items()},
        "bootstrap": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in summary.items()},
        "total_params": tp,
        "model_size_mb": round(float(orig_sz), 2),
        "pruned_size_mb": round(float(psz), 2),
        "quantized_size_mb": round(float(qsz), 2),
        "sparsity_pct": round(float(sp["sparsity_pct"]), 1),
        "cpu_latency_ms": round(float(lat["mean_ms"]), 1),
        "quant_cpu_latency_ms": round(float(qlat["mean_ms"]), 1),
        "compression_ratio": round(float(orig_sz / qsz), 2) if qsz > 0 else 0,
    }
    with open(os.path.join(rd, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON saved to {rd}/evaluation_report.json")
    print("=== EVALUATION COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
