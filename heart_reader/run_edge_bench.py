"""Edge deployment benchmarks: pruning, quantization, ONNX export, latency."""
import os, sys, json, copy, time
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import FusionModel
from edge.quantize import measure_model_size, dynamic_quantize
from edge.prune import apply_structured_pruning, make_pruning_permanent, compute_sparsity
from edge.export_tflite import export_to_onnx


class SignalOnlyModel(nn.Module):
    """Extract signal-only path from fusion model for edge deployment."""
    def __init__(self, fusion_model):
        super().__init__()
        self.signal_backbone = fusion_model.signal_backbone
        self.signal_pool = fusion_model.signal_pool
        self.signal_proj = fusion_model.signal_proj
        self.head = nn.Linear(256, 5)

    def forward(self, x):
        x = self.signal_backbone(x)
        x = self.signal_pool(x)
        x = self.signal_proj(x)
        return self.head(x)


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = FusionModel("inception1d", model_cfg, 5, 12, 1313)
    ckpt = torch.load("checkpoints/fusion_inception1d_best.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()

    # Stats
    tp = sum(p.numel() for p in model.parameters())
    orig_sz = measure_model_size(model)
    print(f"Params: {tp:,}")
    print(f"Size: {orig_sz:.2f} MB")

    # Latency (fusion model with both inputs)
    sig = torch.randn(1, 12, 1000)
    feat = torch.randn(1, 1313)
    with torch.no_grad():
        for _ in range(5):
            model(sig, feat)
    lats = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            model(sig, feat)
            lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    print(f"CPU latency: {lats.mean():.1f} +/- {lats.std():.1f} ms")

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
    print(f"Quantized: size={qsz:.2f} MB")

    try:
        with torch.no_grad():
            for _ in range(5):
                qm(sig, feat)
            qlats = []
            for _ in range(50):
                t0 = time.perf_counter()
                qm(sig, feat)
                qlats.append((time.perf_counter() - t0) * 1000)
        qlats = np.array(qlats)
        qlat_mean = float(qlats.mean())
        print(f"Quantized latency: {qlat_mean:.1f} ms")
    except Exception as e:
        print(f"Quantized latency error: {e}")
        qlat_mean = None

    # Signal-only model for ONNX
    sig_model = SignalOnlyModel(model).eval()
    sig_params = sum(p.numel() for p in sig_model.parameters())
    sig_sz = measure_model_size(sig_model)
    print(f"Signal-only: {sig_params:,} params, {sig_sz:.2f} MB")

    os.makedirs("results/edge", exist_ok=True)
    export_to_onnx(sig_model, "results/edge/heart_reader_signal_only.onnx", (1, 12, 1000))

    # Edge report JSON
    edge_stats = {
        "full_model_params": tp,
        "full_model_size_mb": round(orig_sz, 2),
        "cpu_latency_mean_ms": round(float(lats.mean()), 1),
        "cpu_latency_std_ms": round(float(lats.std()), 1),
        "pruned_sparsity_pct": round(sp["sparsity_pct"], 1),
        "pruned_size_mb": round(psz, 2),
        "quantized_size_mb": round(qsz, 2),
        "compression_ratio_quant": round(orig_sz / qsz, 2) if qsz > 0 else 0,
        "compression_ratio_pruned": round(orig_sz / psz, 2) if psz > 0 else 0,
        "signal_only_params": sig_params,
        "signal_only_size_mb": round(sig_sz, 2),
    }
    if qlat_mean is not None:
        edge_stats["quantized_latency_ms"] = round(qlat_mean, 1)

    with open("results/edge/edge_stats.json", "w") as f:
        json.dump(edge_stats, f, indent=2)
    print("\nEdge stats:")
    print(json.dumps(edge_stats, indent=2))
    print("=== DONE ===")


if __name__ == "__main__":
    main()
