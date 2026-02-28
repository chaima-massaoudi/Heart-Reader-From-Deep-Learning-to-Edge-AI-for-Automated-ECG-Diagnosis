"""
Training loop for ECG multi-label classification.

Features:
- OneCycleLR scheduling (matching fastai fit_one_cycle)
- Mixed precision training (AMP)
- Early stopping on macro_auc
- Model checkpointing
- Gradient clipping
- CSV-based metric logging
"""

import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint
from .losses import build_loss
from .metrics import evaluate_all, macro_auc


class Trainer:
    """Training loop for ECG multi-label classification models.

    Handles the full training lifecycle:
    1. Setup optimizer, scheduler, loss, callbacks
    2. Training loop with mixed precision
    3. Validation and metric computation
    4. Early stopping and checkpointing
    5. Metric logging to CSV

    Args:
        model: The PyTorch model to train.
        cfg: Full YAML config dict.
        device: Compute device.
        model_name: Name for logging and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.model_name = model_name

        train_cfg = cfg["training"]

        # Loss
        self.criterion = build_loss(train_cfg)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Gradient clipping
        self.grad_clip = train_cfg.get("grad_clip", 1.0)

        # Mixed precision
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.amp_device_type = device.type  # "cuda" or "cpu"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Training config
        self.epochs = train_cfg["epochs"]
        self.batch_size = train_cfg["batch_size"]

        # Callbacks
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = None
        if es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                monitor=es_cfg.get("monitor", "macro_auc"),
                patience=es_cfg.get("patience", 10),
                mode=es_cfg.get("mode", "max"),
            )

        checkpoint_dir = train_cfg.get("checkpoint_dir", "./checkpoints/")
        self.checkpoint = ModelCheckpoint(
            save_dir=checkpoint_dir,
            model_name=model_name,
            monitor=es_cfg.get("monitor", "macro_auc"),
            mode=es_cfg.get("mode", "max"),
        )

        # Logging
        self.log_dir = train_cfg.get("log_dir", "./logs/")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{model_name}_training_log.csv")

        # Scheduler (set up per-fit since it needs train_loader length)
        self.scheduler = None
        self.scheduler_type = train_cfg.get("scheduler", "one_cycle")

    def _setup_scheduler(self, steps_per_epoch: int):
        """Initialize LR scheduler.

        Args:
            steps_per_epoch: Number of batches per epoch.
        """
        train_cfg = self.cfg["training"]

        if self.scheduler_type == "one_cycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=train_cfg["lr"],
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=train_cfg["lr"] / 100,
            )
        else:
            self.scheduler = None

    def _train_epoch(self, train_loader) -> Tuple[float, float]:
        """Run one training epoch.

        Returns:
            (average_loss, epoch_time_seconds)
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc="  Train", leave=False)
        for batch in pbar:
            signal = batch["signal"].to(self.device)
            label = batch["label"].to(self.device)
            features = batch.get("features")
            if features is not None:
                features = features.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                logits = self.model(signal, features)
                loss = self.criterion(logits, label)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step scheduler per batch (for OneCycleLR)
            if self.scheduler is not None and self.scheduler_type == "one_cycle":
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Step scheduler per epoch (for cosine)
        if self.scheduler is not None and self.scheduler_type == "cosine":
            self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - start_time

        return avg_loss, elapsed

    @torch.no_grad()
    def _validate(self, val_loader) -> Tuple[float, float, np.ndarray]:
        """Run validation and compute metrics.

        Returns:
            (val_loss, val_macro_auc, predictions)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(val_loader, desc="  Val  ", leave=False):
            signal = batch["signal"].to(self.device)
            label = batch["label"].to(self.device)
            features = batch.get("features")
            if features is not None:
                features = features.to(self.device)

            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                logits = self.model(signal, features)
                loss = self.criterion(logits, label)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)

        val_auc = macro_auc(y_true, y_pred)

        return avg_loss, val_auc, y_pred

    def fit(
        self,
        train_loader,
        val_loader,
        y_val: np.ndarray = None,
    ) -> Dict[str, list]:
        """Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            y_val: Ground truth validation labels (for reference).

        Returns:
            History dict with per-epoch metrics.
        """
        self._setup_scheduler(len(train_loader))

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_macro_auc": [],
            "lr": [],
            "time": [],
        }

        # CSV logger
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_macro_auc", "lr", "time_s"])

        print(f"\n{'='*70}")
        print(f"Training {self.model_name}")
        print(f"  Epochs: {self.epochs}, Batch size: {self.batch_size}")
        print(f"  LR: {self.cfg['training']['lr']}, Scheduler: {self.scheduler_type}")
        print(f"  Device: {self.device}, AMP: {self.use_amp}")
        total_params = sum(p.numel() for p in self.model.parameters())
        train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} total, {train_params:,} trainable")
        print(f"{'='*70}\n")

        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss, train_time = self._train_epoch(train_loader)

            # Validate
            val_loss, val_auc, val_preds = self._validate(val_loader)

            # Current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_auc"].append(val_auc)
            history["lr"].append(current_lr)
            history["time"].append(train_time)

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                 f"{val_auc:.4f}", f"{current_lr:.6f}", f"{train_time:.1f}"])

            # Print epoch summary
            print(f"Epoch {epoch:3d}/{self.epochs} │ "
                  f"Train Loss: {train_loss:.4f} │ "
                  f"Val Loss: {val_loss:.4f} │ "
                  f"Val AUC: {val_auc:.4f} │ "
                  f"LR: {current_lr:.6f} │ "
                  f"Time: {train_time:.1f}s")

            # Checkpoint
            saved = self.checkpoint.step(self.model, val_auc, epoch)
            if saved:
                print(f"  ✓ Saved best model (AUC={val_auc:.4f})")

            # Early stopping
            if self.early_stopping:
                stop = self.early_stopping.step(val_auc, epoch)
                if stop:
                    print(f"\n  ⚡ Early stopping at epoch {epoch} "
                          f"(best epoch: {self.early_stopping.best_epoch}, "
                          f"best AUC: {self.early_stopping.best_score:.4f})")
                    break

        # Load best model
        self.model = self.checkpoint.load_best(self.model, self.device)

        print(f"\nTraining complete. Best AUC: {self.checkpoint.best_score:.4f}")
        return history

    def predict(self, dataloader) -> np.ndarray:
        """Run inference with the trained model.

        Args:
            dataloader: DataLoader for inference.

        Returns:
            Predictions array, shape (N, C), values in [0, 1].
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predict", leave=False):
                signal = batch["signal"].to(self.device)
                features = batch.get("features")
                if features is not None:
                    features = features.to(self.device)

                with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                    logits = self.model(signal, features)

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds, axis=0)
