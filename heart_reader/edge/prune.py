"""
Structured pruning for model compression.

Uses PyTorch's built-in pruning utilities to remove conv/linear filters
based on L1-norm magnitude. Includes post-pruning fine-tuning support.
"""

import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def get_prunable_layers(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """Find all Conv1d and Linear layers eligible for pruning.

    Args:
        model: PyTorch model.

    Returns:
        List of (module, 'weight') tuples for pruning.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            layers.append((module, "weight"))
    return layers


def apply_structured_pruning(
    model: nn.Module,
    amount: float = 0.5,
    n: int = 1,
) -> nn.Module:
    """Apply L1-norm structured pruning to all conv/linear layers.

    Removes `amount` fraction of filters based on L1 magnitude.

    Args:
        model: Model to prune.
        amount: Fraction of filters to prune (0.0 to 1.0).
        n: Norm order for importance scoring (1 = L1).

    Returns:
        Pruned model (with pruning masks applied).
    """
    layers = get_prunable_layers(model)

    for module, param_name in layers:
        if isinstance(module, nn.Conv1d):
            # Structured pruning: remove entire output channels
            prune.ln_structured(module, name=param_name, amount=amount, n=n, dim=0)
        elif isinstance(module, nn.Linear):
            # Unstructured pruning for linear layers
            prune.l1_unstructured(module, name=param_name, amount=amount)

    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """Remove pruning reparameterization — make pruned weights permanent.

    After this, the pruning masks are applied and the `_orig` / `_mask`
    buffers are removed.

    Args:
        model: Pruned model.

    Returns:
        Model with permanent pruning.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # Not pruned
    return model


def compute_sparsity(model: nn.Module) -> dict:
    """Compute sparsity statistics for the model.

    Returns:
        Dict with total/nonzero/sparsity_pct.
    """
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += param.nonzero().size(0)

    sparsity = 1.0 - (nonzero / total) if total > 0 else 0.0
    return {
        "total_params": total,
        "nonzero_params": nonzero,
        "sparsity_pct": sparsity * 100,
    }


def save_pruned_model(model: nn.Module, save_path: str, model_name: str = "pruned"):
    """Save pruned model state dict.

    Args:
        model: Pruned model.
        save_path: Directory to save.
        model_name: File name prefix.
    """
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{model_name}.pt")
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Pruned model saved to {path} ({size_mb:.2f} MB)")
    return path, size_mb
