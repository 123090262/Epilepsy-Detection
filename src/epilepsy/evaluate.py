"""Evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_summary(path: str | Path, fold_metrics: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = [
        name
        for name in (
            "acc",
            "auc",
            "pr_auc",
            "prec",
            "recall",
            "specificity",
            "balanced_acc",
            "f1",
        )
        if all(name in metrics for metrics in fold_metrics)
    ]
    values = np.array(
        [[m[name] for name in metric_names] for m in fold_metrics],
        dtype=np.float64,
    )
    summary = {
        "folds": fold_metrics,
        "mean": dict(zip(metric_names, np.nanmean(values, axis=0).tolist())),
        "std": dict(zip(metric_names, np.nanstd(values, axis=0).tolist())),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
