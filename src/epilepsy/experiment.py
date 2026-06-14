"""Shared fold training used by cross-validation, LOSO, and parameter search."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from epilepsy.data import (
    ChannelPreprocessor,
    ChbmitPoolDataset,
    EpochBalancedSampler,
    PoolSample,
    fit_channel_preprocessor,
)
from epilepsy.models import EpilepsyGATNet
from epilepsy.train import (
    Metrics,
    build_criterion,
    build_optimizer,
    build_scheduler,
    evaluate,
    select_decision_threshold,
    train_one_epoch,
)


@dataclass
class FoldTrainingResult:
    model: torch.nn.Module
    criterion: torch.nn.Module
    preprocessor: ChannelPreprocessor
    threshold: float
    best_epoch: int
    best_metrics: Metrics
    history: dict[str, list[float]]
    train_pool_size: int
    train_epoch_size: int


def build_model(config, device: torch.device) -> EpilepsyGATNet:
    return EpilepsyGATNet(
        fs=config.data.sample_rate,
        num_classes=config.model.num_classes,
        feature_dim=config.model.feature_dim,
        hid_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        graph_dropout=config.model.graph_dropout,
        spectral_fusion=config.model.spectral_fusion,
        auxiliary_weight=config.model.auxiliary_weight,
    ).to(device)


def make_eval_loader(
    samples: Sequence[PoolSample],
    indices: np.ndarray,
    preprocessor: ChannelPreprocessor,
    config,
) -> DataLoader:
    return DataLoader(
        ChbmitPoolDataset(samples, indices, preprocessor),
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )


def fit_fold(
    samples: Sequence[PoolSample],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config,
    device: torch.device,
    seed: int,
    max_epochs: int | None = None,
    epoch_callback: Callable[[int, float, Metrics], None] | None = None,
) -> FoldTrainingResult:
    train_config = config.train
    if max_epochs is not None:
        train_config = replace(train_config, num_epochs=max_epochs)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    statistics_dataset = ChbmitPoolDataset(samples, train_indices)
    preprocessor = fit_channel_preprocessor(
        statistics_dataset,
        lower_quantile=train_config.clip_lower_quantile,
        upper_quantile=train_config.clip_upper_quantile,
        max_segments=train_config.statistics_max_segments,
        random_state=seed,
    )
    train_dataset = ChbmitPoolDataset(samples, train_indices, preprocessor)
    val_dataset = ChbmitPoolDataset(samples, val_indices, preprocessor)
    sampler = EpochBalancedSampler(
        train_dataset.labels,
        seed=seed,
        negative_ratio=train_config.negative_ratio,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    model = build_model(config, device)
    criterion = build_criterion(train_config)
    optimizer = build_optimizer(model, train_config)
    scheduler, scheduler_per_batch = build_scheduler(
        optimizer, train_config, len(train_loader)
    )

    best_score = -float("inf")
    best_loss = float("inf")
    best_epoch = -1
    best_threshold = 0.5
    best_metrics = None
    best_state = None
    stale_epochs = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(1, train_config.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            scheduler=scheduler,
            scheduler_per_batch=scheduler_per_batch,
            max_grad_norm=train_config.max_grad_norm,
        )
        if scheduler is not None and not scheduler_per_batch:
            scheduler.step()

        raw_metrics, y_true, _, y_score = evaluate(
            model, device, val_loader, criterion
        )
        threshold, val_metrics = select_decision_threshold(
            y_true,
            y_score,
            metric_name=train_config.threshold_metric,
            loss=raw_metrics.loss,
            min_threshold=train_config.min_threshold,
            max_threshold=train_config.max_threshold,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics.loss)
        history["val_accuracy"].append(val_metrics.accuracy)
        history["val_precision"].append(val_metrics.precision)
        history["val_recall"].append(val_metrics.recall)
        history["val_f1"].append(val_metrics.f1)

        if epoch_callback is not None:
            epoch_callback(epoch, train_loss, val_metrics)

        score = getattr(val_metrics, train_config.checkpoint_metric)
        improved = score > best_score or (
            np.isclose(score, best_score) and val_metrics.loss < best_loss
        )
        if improved:
            best_score = score
            best_loss = val_metrics.loss
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_config.early_stopping_patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return FoldTrainingResult(
        model=model,
        criterion=criterion,
        preprocessor=preprocessor,
        threshold=best_threshold,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        history=history,
        train_pool_size=len(train_dataset),
        train_epoch_size=len(sampler),
    )
