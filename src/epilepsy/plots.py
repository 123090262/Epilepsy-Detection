"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Seizure (1)"],
        yticklabels=["Normal (0)", "Seizure (1)"],
    )
    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_summary(metrics: dict[str, list[float]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(metrics["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].plot(epochs, metrics["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, metrics["val_loss"], "r-", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, metrics["val_accuracy"], "m-", linewidth=2)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    bars = ["Precision", "Recall", "F1"]
    means = [
        np.nanmean(metrics["val_precision"]) * 100,
        np.nanmean(metrics["val_recall"]) * 100,
        np.nanmean(metrics["val_f1"]) * 100,
    ]
    axes[2].bar(bars, means, color=["#2196F3", "#FFC107", "#4CAF50"], alpha=0.85)
    axes[2].set_title("Validation Metrics")
    axes[2].set_ylabel("Score (%)")
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
