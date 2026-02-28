"""
Training callbacks: EarlyStopping and ModelCheckpoint.

Modeled after the PTB-XL benchmarking fastai callbacks,
re-implemented for a clean PyTorch training loop.
"""

import os
import numpy as np
import torch
from typing import Optional


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Args:
        monitor: Metric name to monitor (e.g., 'macro_auc', 'val_loss').
        patience: Number of epochs with no improvement before stopping.
        mode: 'max' (higher is better) or 'min' (lower is better).
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        monitor: str = "macro_auc",
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, score: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoint when a monitored metric improves.

    Args:
        save_dir: Directory to save checkpoints.
        model_name: Name prefix for the checkpoint file.
        monitor: Metric to monitor.
        mode: 'max' or 'min'.
    """

    def __init__(
        self,
        save_dir: str,
        model_name: str,
        monitor: str = "macro_auc",
        mode: str = "max",
    ):
        self.save_dir = save_dir
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode

        self.best_score = -np.inf if mode == "max" else np.inf
        self.best_path: Optional[str] = None

        os.makedirs(save_dir, exist_ok=True)

    def step(self, model: torch.nn.Module, score: float, epoch: int) -> bool:
        """Save model if metric improved.

        Args:
            model: The model to save.
            score: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if model was saved.
        """
        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            path = os.path.join(
                self.save_dir, f"{self.model_name}_best.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "score": score,
            }, path)
            self.best_path = path
            return True

        return False

    def load_best(self, model: torch.nn.Module, device: torch.device = None) -> torch.nn.Module:
        """Load the best saved model checkpoint.

        Args:
            model: Model with matching architecture.
            device: Device to load onto.

        Returns:
            Model with loaded weights.
        """
        if self.best_path is None:
            print("Warning: No checkpoint saved yet.")
            return model

        checkpoint = torch.load(self.best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']} "
              f"({self.monitor}={checkpoint['score']:.4f})")
        return model
