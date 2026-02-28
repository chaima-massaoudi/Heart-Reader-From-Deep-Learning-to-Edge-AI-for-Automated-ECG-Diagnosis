"""
Export PyTorch model to TFLite format for edge deployment.

Pipeline: PyTorch → ONNX → TensorFlow SavedModel → TFLite

Supports:
- Default optimization (dynamic range quantization)
- Full integer quantization (INT8) with representative dataset
"""

import os
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    save_path: str = "model.onnx",
    input_shape: tuple = (1, 12, 1000),
    opset_version: int = 13,
) -> str:
    """Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model.
        save_path: Output .onnx file path.
        input_shape: (batch, channels, seq_len).
        opset_version: ONNX opset version.

    Returns:
        Path to saved ONNX model.
    """
    model = model.cpu().eval()
    dummy_input = torch.randn(*input_shape)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Use dynamo=False to force legacy TorchScript-based exporter
    # (more compatible with Conv1d/BatchNorm1d/SE blocks)
    export_kwargs = dict(
        input_names=["ecg_signal"],
        output_names=["predictions"],
        dynamic_axes={
            "ecg_signal": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    try:
        # PyTorch >= 2.6: explicit dynamo=False for legacy exporter
        torch.onnx.export(model, dummy_input, save_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Older PyTorch without dynamo kwarg
        torch.onnx.export(model, dummy_input, save_path, **export_kwargs)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"ONNX model saved to {save_path} ({size_mb:.2f} MB)")
    return save_path


def onnx_to_tflite(
    onnx_path: str,
    tflite_path: str = "model.tflite",
    optimize: bool = True,
    full_integer: bool = False,
    representative_data: Optional[np.ndarray] = None,
) -> str:
    """Convert ONNX model to TFLite via TensorFlow.

    Args:
        onnx_path: Path to .onnx model.
        tflite_path: Output .tflite path.
        optimize: Apply DEFAULT optimizations (dynamic range quantization).
        full_integer: Apply full INT8 quantization (requires representative_data).
        representative_data: Calibration data array, shape (N, C, L).

    Returns:
        Path to saved .tflite model.
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        print(f"Missing dependency for TFLite conversion: {e}")
        print("Install with: pip install onnx onnx-tf tensorflow")
        return None

    # ONNX → TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    # Save as TF SavedModel
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, "saved_model")
        tf_rep.export_graph(saved_model_dir)

        # TF SavedModel → TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if full_integer and representative_data is not None:
            def representative_dataset():
                for i in range(min(100, len(representative_data))):
                    sample = representative_data[i:i+1].astype(np.float32)
                    yield [sample]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

    # Save TFLite model
    os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved to {tflite_path} ({size_mb:.2f} MB)")
    return tflite_path


def verify_tflite_model(
    tflite_path: str,
    test_input: np.ndarray = None,
    input_shape: tuple = (1, 12, 1000),
) -> np.ndarray:
    """Verify TFLite model produces valid outputs.

    Args:
        tflite_path: Path to .tflite model.
        test_input: Test input array. If None, uses random data.
        input_shape: Shape for random test input.

    Returns:
        Model output array.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available for verification.")
        return None

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"TFLite model info:")
    print(f"  Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")

    if test_input is None:
        test_input = np.random.randn(*input_shape).astype(np.float32)

    # Handle quantized models
    if input_details[0]["dtype"] == np.int8:
        input_scale = input_details[0]["quantization"][0]
        input_zero = input_details[0]["quantization"][1]
        test_input = (test_input / input_scale + input_zero).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    print(f"  Output values: min={output.min():.4f}, max={output.max():.4f}, "
          f"mean={output.mean():.4f}")

    return output


def full_export_pipeline(
    model: nn.Module,
    output_dir: str,
    model_name: str = "heart_reader",
    input_shape: tuple = (1, 12, 1000),
    optimize: bool = True,
    full_integer: bool = False,
    representative_data: Optional[np.ndarray] = None,
) -> dict:
    """Run the full PyTorch → ONNX → TFLite export pipeline.

    Args:
        model: Trained PyTorch model.
        output_dir: Output directory.
        model_name: Name prefix for output files.
        input_shape: Model input shape.
        optimize: Apply TFLite DEFAULT optimization.
        full_integer: Apply full INT8 quantization.
        representative_data: Calibration data for INT8 quantization.

    Returns:
        Dict with paths and sizes of all exported models.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Step 1: ONNX export
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    print("\n[1/3] Exporting to ONNX...")
    export_to_onnx(model, onnx_path, input_shape)
    results["onnx_path"] = onnx_path
    results["onnx_size_mb"] = os.path.getsize(onnx_path) / (1024 * 1024)

    # Step 2: TFLite conversion
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
    print("\n[2/3] Converting to TFLite...")
    tflite_result = onnx_to_tflite(
        onnx_path, tflite_path,
        optimize=optimize,
        full_integer=full_integer,
        representative_data=representative_data,
    )

    if tflite_result:
        results["tflite_path"] = tflite_path
        results["tflite_size_mb"] = os.path.getsize(tflite_path) / (1024 * 1024)

        # Step 3: Verify
        print("\n[3/3] Verifying TFLite model...")
        verify_tflite_model(tflite_path, input_shape=input_shape)
    else:
        print("TFLite conversion failed — ONNX model still available.")

    return results
