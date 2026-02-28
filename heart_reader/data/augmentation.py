"""
ECG-specific data augmentation transforms for 12-lead signals.

All transforms operate on numpy arrays of shape (sequence_length, num_leads).
"""

import numpy as np


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal


class GaussianNoise:
    """Add Gaussian noise to the signal."""

    def __init__(self, std: float = 0.05, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            noise = np.random.normal(0, self.std, signal.shape).astype(signal.dtype)
            signal = signal + noise
        return signal


class RandomScale:
    """Randomly scale signal amplitude."""

    def __init__(self, scale_range=(0.8, 1.2), p: float = 0.5):
        self.lo, self.hi = scale_range
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            scale = np.random.uniform(self.lo, self.hi)
            signal = signal * scale
        return signal


class LeadDropout:
    """Randomly zero-out entire leads to simulate missing/noisy leads."""

    def __init__(self, max_leads: int = 2, p: float = 0.1):
        self.max_leads = max_leads
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            n_drop = np.random.randint(1, self.max_leads + 1)
            leads = np.random.choice(signal.shape[1], n_drop, replace=False)
            signal = signal.copy()
            signal[:, leads] = 0.0
        return signal


class BaselineWander:
    """Simulate baseline wander by adding low-frequency sinusoidal drift."""

    def __init__(self, max_amplitude: float = 0.1, max_freq: float = 0.5, p: float = 0.3):
        self.max_amplitude = max_amplitude
        self.max_freq = max_freq
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            seq_len = signal.shape[0]
            t = np.linspace(0, 1, seq_len)
            freq = np.random.uniform(0.1, self.max_freq)
            amplitude = np.random.uniform(0, self.max_amplitude)
            phase = np.random.uniform(0, 2 * np.pi)
            wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
            # Apply same wander to all leads (or per-lead)
            signal = signal + wander[:, np.newaxis].astype(signal.dtype)
        return signal


class TimeWarp:
    """Apply slight temporal stretching/compression via interpolation."""

    def __init__(self, max_ratio: float = 0.1, p: float = 0.2):
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            seq_len, n_leads = signal.shape
            ratio = 1.0 + np.random.uniform(-self.max_ratio, self.max_ratio)
            new_len = int(seq_len * ratio)
            old_indices = np.linspace(0, seq_len - 1, new_len)
            orig_indices = np.arange(seq_len)
            warped = np.zeros_like(signal)
            # Interpolate each lead
            for lead in range(n_leads):
                interp_signal = np.interp(
                    np.linspace(0, new_len - 1, seq_len),
                    np.arange(new_len),
                    np.interp(old_indices, orig_indices, signal[:, lead]),
                )
                warped[:, lead] = interp_signal
            signal = warped
        return signal


class RandomCrop:
    """Randomly crop a fixed-length window from the signal."""

    def __init__(self, crop_length: int = 250):
        self.crop_length = crop_length

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        seq_len = signal.shape[0]
        if seq_len <= self.crop_length:
            return signal
        start = np.random.randint(0, seq_len - self.crop_length)
        return signal[start : start + self.crop_length]


class RandomShift:
    """Circular shift the signal in time."""

    def __init__(self, max_shift: int = 50, p: float = 0.3):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            shift = np.random.randint(-self.max_shift, self.max_shift)
            signal = np.roll(signal, shift, axis=0)
        return signal


def build_augmentation(cfg: dict) -> Compose:
    """Build augmentation pipeline from config dict.

    Args:
        cfg: augmentation section of the YAML config.

    Returns:
        Compose transform that applies all enabled augmentations.
    """
    transforms = []

    if not cfg.get("enabled", False):
        return Compose(transforms)

    transforms.append(GaussianNoise(
        std=cfg.get("gaussian_noise_std", 0.05), p=0.5
    ))
    transforms.append(RandomScale(
        scale_range=cfg.get("random_scale_range", [0.8, 1.2]), p=0.5
    ))
    transforms.append(LeadDropout(
        max_leads=cfg.get("lead_dropout_max", 2),
        p=cfg.get("lead_dropout_prob", 0.1),
    ))
    transforms.append(BaselineWander(
        p=cfg.get("baseline_wander_prob", 0.3)
    ))
    transforms.append(TimeWarp(
        max_ratio=cfg.get("time_warp_max_ratio", 0.1),
        p=cfg.get("time_warp_prob", 0.2),
    ))
    transforms.append(RandomShift(max_shift=50, p=0.3))

    if cfg.get("random_crop", False):
        transforms.append(RandomCrop(
            crop_length=cfg.get("crop_length", 250)
        ))

    return Compose(transforms)
