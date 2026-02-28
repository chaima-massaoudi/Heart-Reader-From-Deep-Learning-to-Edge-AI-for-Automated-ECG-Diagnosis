"""
Benchmarking utilities for edge-optimized models.

Compares:
- Model file size (original vs pruned vs quantized vs TFLite)
- Inference latency (CPU)
- Accuracy (macro_auc) at each compression stage
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .quantize import measure_inference_time, measure_model_size


def benchmark_pytorch_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    model_name: str = "model",
    save_path: Optional[str] = None,
) -> dict:
    """Benchmark a PyTorch model's size and speed.

    Args:
        model: Model to benchmark.
        dataloader: DataLoader for accuracy measurement.
        device: Compute device.
        model_name: Name for reporting.
        save_path: Optional path to save model for size measurement.

    Returns:
        Dict with size_mb, latency stats.
    """
    results = {"name": model_name}

    # Model size
    if save_path:
        torch.save(model.state_dict(), save_path)
        results["size_mb"] = os.path.getsize(save_path) / (1024 * 1024)
    else:
        results["size_mb"] = measure_model_size(model)

    # Inference speed on CPU
    latency = measure_inference_time(model.cpu(), device="cpu")
    results.update({f"cpu_{k}": v for k, v in latency.items()})

    return results


def benchmark_tflite_model(
    tflite_path: str,
    input_shape: tuple = (1, 12, 1000),
    n_runs: int = 100,
) -> dict:
    """Benchmark a TFLite model's size and speed.

    Args:
        tflite_path: Path to .tflite file.
        input_shape: Input shape.
        n_runs: Number of inference runs.

    Returns:
        Dict with size_mb, latency stats.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available for TFLite benchmarking.")
        return {}

    results = {"name": "TFLite"}
    results["size_mb"] = os.path.getsize(tflite_path) / (1024 * 1024)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.random.randn(*input_shape).astype(
        input_details[0]["dtype"] if input_details[0]["dtype"] != np.int8 else np.float32
    )

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()

    # Timing
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    results["cpu_mean_ms"] = latencies.mean()
    results["cpu_std_ms"] = latencies.std()
    results["cpu_median_ms"] = np.median(latencies)

    return results


def generate_benchmark_report(
    benchmarks: List[dict],
    output_dir: str,
) -> str:
    """Generate a comparison report across all model versions.

    Args:
        benchmarks: List of benchmark result dicts.
        output_dir: Directory to save report.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("EDGE DEPLOYMENT BENCHMARK REPORT")
    lines.append("=" * 80)

    # Table header
    lines.append(f"\n{'Model':<25} {'Size (MB)':>10} {'Latency (ms)':>14} {'Compression':>12}")
    lines.append("-" * 65)

    baseline_size = benchmarks[0]["size_mb"] if benchmarks else 1.0

    for bm in benchmarks:
        name = bm.get("name", "Unknown")
        size = bm.get("size_mb", 0)
        latency = bm.get("cpu_mean_ms", 0)
        compression = baseline_size / max(size, 0.001)

        lines.append(f"{name:<25} {size:>10.2f} {latency:>14.2f} {compression:>11.1f}x")

    lines.append("-" * 65)
    lines.append("")

    # Summary
    if len(benchmarks) >= 2:
        orig = benchmarks[0]
        final = benchmarks[-1]
        lines.append(f"Size reduction: {orig['size_mb']:.2f} MB → {final['size_mb']:.2f} MB "
                      f"({orig['size_mb']/max(final['size_mb'], 0.001):.1f}x compression)")
        if "cpu_mean_ms" in orig and "cpu_mean_ms" in final:
            lines.append(f"Speedup: {orig['cpu_mean_ms']:.2f} ms → {final['cpu_mean_ms']:.2f} ms "
                          f"({orig['cpu_mean_ms']/max(final['cpu_mean_ms'], 0.001):.1f}x faster)")

    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    return report
