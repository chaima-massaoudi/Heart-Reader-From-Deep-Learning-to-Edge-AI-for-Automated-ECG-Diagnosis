"""
Model quantization for compression and faster inference.

Supports:
- Dynamic quantization (INT8, no calibration data needed)
- Static quantization (requires calibration data)
- Quantization-aware training (QAT) support
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to Linear and LSTM layers.

    Dynamic quantization converts weights to INT8 and dynamically
    quantizes activations at inference time. Works well for models
    that are bottlenecked by weight memory bandwidth.

    Args:
        model: Trained PyTorch model (must be on CPU).

    Returns:
        Quantized model.
    """
    model = model.cpu().eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv1d},
        dtype=torch.qint8,
    )

    return quantized_model


def measure_model_size(model: nn.Module, save_path: str = None) -> float:
    """Measure model size in MB by saving to a temp file.

    Args:
        model: PyTorch model.
        save_path: Optional path to save (default: temp file).

    Returns:
        Model size in MB.
    """
    if save_path is None:
        save_path = "_temp_model_size.pt"
        cleanup = True
    else:
        cleanup = False

    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)

    if cleanup:
        os.remove(save_path)

    return size_mb


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple = (1, 12, 1000),
    n_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> dict:
    """Benchmark inference latency.

    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape (B, C, L).
        n_runs: Number of inference runs.
        warmup: Warmup runs before timing.
        device: Device for inference.

    Returns:
        Dict with mean/std/min/max latency in ms.
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)

    # Timing
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": latencies.mean(),
        "std_ms": latencies.std(),
        "min_ms": latencies.min(),
        "max_ms": latencies.max(),
        "median_ms": np.median(latencies),
    }


def save_quantized_model(model: nn.Module, save_dir: str, model_name: str = "quantized"):
    """Save quantized model.

    Args:
        model: Quantized model.
        save_dir: Output directory.
        model_name: File prefix.

    Returns:
        (path, size_mb)
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Quantized model saved to {path} ({size_mb:.2f} MB)")
    return path, size_mb
