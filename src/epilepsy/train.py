"""Training, loss, threshold calibration, and metric utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
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
    pr_auc: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    f1: float
    composite: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "auc": self.auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "composite": self.composite,
            "threshold": self.threshold,
        }


class BinaryFocalLoss(nn.Module):
    """Two-class focal loss with explicit positive-class weighting."""

    def __init__(self, gamma: float = 2.0, positive_alpha: float = 0.65) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative")
        if not 0 < positive_alpha < 1:
            raise ValueError("focal positive alpha must be in (0, 1)")
        self.gamma = gamma
        self.positive_alpha = positive_alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target[:, None]).squeeze(1)
        target_probs = probs.gather(1, target[:, None]).squeeze(1)
        alpha = torch.where(
            target == 1,
            torch.as_tensor(self.positive_alpha, device=logits.device),
            torch.as_tensor(1.0 - self.positive_alpha, device=logits.device),
        )
        return (-alpha * (1.0 - target_probs).pow(self.gamma) * target_log_probs).mean()


def build_criterion(train_config) -> nn.Module:
    loss_name = train_config.loss.lower()
    if loss_name == "focal":
        return BinaryFocalLoss(train_config.focal_gamma, train_config.focal_alpha)
    if loss_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {train_config.loss}")


def build_optimizer(model: nn.Module, train_config) -> torch.optim.Optimizer:
    name = train_config.optimizer.lower()
    kwargs = {
        "lr": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {train_config.optimizer}")


def build_scheduler(optimizer, train_config, steps_per_epoch: int):
    name = train_config.scheduler.lower()
    if name in {"none", "off"}:
        return None, False
    if name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_config.learning_rate,
            epochs=train_config.num_epochs,
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
        return scheduler, True
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config.num_epochs, eta_min=train_config.learning_rate / 100
        )
        return scheduler, False
    raise ValueError(f"Unsupported scheduler: {train_config.scheduler}")


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler=None,
    scheduler_per_batch: bool = False,
    max_grad_norm: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None and scheduler_per_batch:
            scheduler.step()

        total_loss += loss.item() * data.size(0)
        total_items += data.size(0)

    return total_loss / max(total_items, 1)


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    loss: float,
    threshold: float,
) -> tuple[Metrics, np.ndarray]:
    auc = math.nan
    pr_auc = math.nan
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    # F1 drives seizure detection while balanced accuracy and raw accuracy prevent
    # thresholds that trade away either class or flood the output with positives.
    composite = 0.60 * f1 + 0.25 * balanced_accuracy + 0.15 * accuracy
    metrics = Metrics(
        loss=float(loss),
        accuracy=accuracy,
        auc=auc,
        pr_auc=pr_auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(tn / max(tn + fp, 1)),
        balanced_accuracy=balanced_accuracy,
        f1=f1,
        composite=composite,
        threshold=float(threshold),
    )
    return metrics, y_pred


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> tuple[Metrics, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
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

            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_score.extend(probs[:, 1].cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    metrics, y_pred_np = calculate_binary_metrics(
        y_true_np,
        y_score_np,
        loss=total_loss / max(total_items, 1),
        threshold=threshold,
    )
    return metrics, y_true_np, y_pred_np, y_score_np


def select_decision_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_name: str,
    loss: float,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
) -> tuple[float, Metrics]:
    supported = {"f1", "balanced_accuracy", "accuracy", "composite"}
    if metric_name not in supported:
        raise ValueError(f"Unsupported threshold metric: {metric_name}")
    if len(np.unique(y_true)) != 2:
        raise ValueError("threshold selection requires both validation classes")
    if not 0 <= min_threshold < max_threshold <= 1:
        raise ValueError("threshold bounds must satisfy 0 <= min < max <= 1")

    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(min_threshold, max_threshold, 181),
                y_score[(y_score >= min_threshold) & (y_score <= max_threshold)],
                np.asarray([0.5]),
            )
        )
    )
    best_threshold = 0.5
    best_metrics, _ = calculate_binary_metrics(y_true, y_score, loss, best_threshold)
    best_value = getattr(best_metrics, metric_name)
    for threshold in candidates:
        metrics, _ = calculate_binary_metrics(y_true, y_score, loss, float(threshold))
        value = getattr(metrics, metric_name)
        if value > best_value or (
            np.isclose(value, best_value)
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value = value
    return best_threshold, best_metrics


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
