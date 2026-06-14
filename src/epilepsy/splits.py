"""Leakage-resistant grouped dataset splitting utilities."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_train_val_split(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("grouped validation requires at least two groups")
    requested_splits = max(2, int(round(1.0 / val_size)))
    n_splits = min(requested_splits, len(unique_groups))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_relative, val_relative = next(
        splitter.split(indices, labels, groups=groups)
    )
    return indices[train_relative], indices[val_relative]


def grouped_kfold_splits(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
):
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups, got {len(np.unique(groups))}"
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for train_relative, test_relative in splitter.split(
        indices, labels, groups=groups
    ):
        yield indices[train_relative], indices[test_relative]
