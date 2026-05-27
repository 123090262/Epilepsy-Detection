"""Training loop utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@dataclass
class Metrics:
    loss: float
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "auc": self.auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_items += data.size(0)

    return total_loss / max(total_items, 1)


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[Metrics, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs[:, 1].cpu().numpy().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_score_np = np.array(y_score)

    auc = math.nan
    if len(np.unique(y_true_np)) == 2:
        auc = float(roc_auc_score(y_true_np, y_score_np))

    metrics = Metrics(
        loss=total_loss / max(total_items, 1),
        accuracy=float(accuracy_score(y_true_np, y_pred_np)),
        auc=auc,
        precision=float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        recall=float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        f1=float(f1_score(y_true_np, y_pred_np, zero_division=0)),
    )
    return metrics, y_true_np, y_pred_np, y_score_np


# Notebook-compatible aliases.
def train(model, device, train_loader, optimizer, criterion, epoch=None):
    return train_one_epoch(model, device, train_loader, optimizer, criterion)


def test(model, device, loader, criterion):
    metrics, _, _, _ = evaluate(model, device, loader, criterion)
    return (
        metrics.loss,
        metrics.accuracy,
        metrics.auc,
        metrics.precision,
        metrics.recall,
        metrics.f1,
    )
