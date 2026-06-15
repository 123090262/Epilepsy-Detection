"""Classical EEG features for small patient-specific training sets."""

from __future__ import annotations

import numpy as np


EEG_BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0))


def extract_eeg_features(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return per-channel temporal statistics and log band powers."""
    x = np.asarray(segment, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("EEG segment must have shape (channels, samples)")

    features = [
        np.mean(x, axis=1),
        np.std(x, axis=1),
        np.sqrt(np.mean(x * x, axis=1)),
        np.mean(np.abs(np.diff(x, axis=1)), axis=1),
    ]
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0 / sample_rate)
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    for low, high in EEG_BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        features.append(np.log1p(np.mean(power[:, mask], axis=1)))
    return np.concatenate(features).astype(np.float32)
