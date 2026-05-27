"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from epilepsy.config import DataConfig


def prepare_csv_dataset(config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build EEG segments from CHB-MIT style patient CSV folders.

    Expected folder layout:

    ```text
    data_dir/
      chb06/
        0.csv
        1.csv
      chb08/
        0.csv
        seizure_x.csv
    ```

    Label rule kept from the notebook:
    - `0.csv` -> non-seizure class 0
    - every other `.csv` -> seizure class 1

    Returns:
        X: shape `(N, C, L)`, usually `(N, 22, 512)`
        y: shape `(N,)`
        patient_ids: shape `(N,)`
    """

    root_dir = Path(config.data_dir).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {root_dir}. "
            "Update data.data_dir in the YAML config."
        )
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {root_dir}")

    all_segments: list[np.ndarray] = []
    all_labels: list[int] = []
    all_patient_ids: list[str] = []

    for patient_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        patient = patient_dir.name
        for csv_path in sorted(patient_dir.glob("*.csv")):
            label = 0 if csv_path.name == "0.csv" else 1

            try:
                df = pd.read_csv(csv_path, header=None)
                df = df.apply(pd.to_numeric, errors="coerce").dropna()

                if df.shape[1] < config.num_channels:
                    print(
                        f"Skip {csv_path}: expected at least "
                        f"{config.num_channels} columns, got {df.shape[1]}"
                    )
                    continue

                eeg = df.iloc[:, : config.num_channels].values.astype(np.float32)
                total_samples = len(eeg)
                if total_samples < config.segment_length:
                    print(
                        f"Skip {csv_path}: {total_samples} samples "
                        f"< segment length {config.segment_length}"
                    )
                    continue

                num_segments = total_samples // config.segment_length
                for i in range(num_segments):
                    start = i * config.segment_length
                    end = start + config.segment_length
                    segment = eeg[start:end].T
                    all_segments.append(segment)
                    all_labels.append(label)
                    all_patient_ids.append(patient)
            except Exception as exc:
                print(f"Failed to read {csv_path}: {exc}")

    if not all_segments:
        raise RuntimeError(f"No valid EEG segments were generated from {root_dir}")

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    patient_ids = np.array(all_patient_ids)
    return X, y, patient_ids


def make_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
