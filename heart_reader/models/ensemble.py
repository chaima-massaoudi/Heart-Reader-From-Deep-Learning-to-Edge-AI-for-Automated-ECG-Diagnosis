"""
Weighted Ensemble of multiple ECG classification models.

Trains 3 backbone models independently, then combines their predictions
with optimized weights (found via scipy.optimize on the validation set).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from typing import List, Optional, Tuple


class WeightedEnsemble:
    """Weighted ensemble that combines predictions from multiple models.

    Supports two modes:
    - "average": Equal-weight averaging of predictions.
    - "weighted": Optimize weights on validation set to maximize macro_auc.

    Usage:
        ensemble = WeightedEnsemble(method="weighted")
        ensemble.fit_weights(val_predictions_list, y_val)
        final_pred = ensemble.predict(test_predictions_list)
    """

    def __init__(self, method: str = "weighted"):
        self.method = method
        self.weights: Optional[np.ndarray] = None

    def fit_weights(
        self,
        predictions: List[np.ndarray],
        y_true: np.ndarray,
    ) -> np.ndarray:
        """Optimize ensemble weights on validation data.

        Args:
            predictions: List of K prediction arrays, each shape (N, C).
            y_true: Ground truth labels, shape (N, C).

        Returns:
            Optimized weight vector of length K.
        """
        n_models = len(predictions)
        preds_stack = np.stack(predictions, axis=0)  # (K, N, C)

        if self.method == "average":
            self.weights = np.ones(n_models) / n_models
            return self.weights

        # Optimize weights to maximize macro AUC
        def neg_auc(w):
            w = np.abs(w)
            w = w / w.sum()  # normalize
            combined = np.tensordot(w, preds_stack, axes=([0], [0]))  # (N, C)
            try:
                auc = roc_auc_score(y_true, combined, average="macro", multi_class="ovr")
                return -auc
            except ValueError:
                return 0.0

        # Initial: equal weights
        w0 = np.ones(n_models) / n_models

        # Constrain weights to sum to 1, all non-negative
        constraints = {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}
        bounds = [(0.0, 1.0)] * n_models

        result = minimize(
            neg_auc, w0,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6},
        )

        self.weights = np.abs(result.x)
        self.weights = self.weights / self.weights.sum()

        print(f"Ensemble weights: {self.weights}")
        print(f"Ensemble validation AUC: {-result.fun:.4f}")

        return self.weights

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Combine predictions using learned weights.

        Args:
            predictions: List of K prediction arrays, each shape (N, C).

        Returns:
            Weighted combination, shape (N, C).
        """
        if self.weights is None:
            self.weights = np.ones(len(predictions)) / len(predictions)

        preds_stack = np.stack(predictions, axis=0)  # (K, N, C)
        return np.tensordot(self.weights, preds_stack, axes=([0], [0]))


def collect_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference and collect sigmoid predictions.

    Args:
        model: Trained model.
        dataloader: Data loader.
        device: Compute device.

    Returns:
        Predictions array, shape (N, C), values in [0, 1].
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            signal = batch["signal"].to(device)
            features = batch.get("features")
            if features is not None:
                features = features.to(device)

            logits = model(signal, features)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
