"""Shared fold training used by cross-validation, LOSO, and parameter search."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from epilepsy.chbmit import TARGET_CHANNELS
from epilepsy.data import (
    ChannelPreprocessor,
    ChbmitPoolDataset,
    EpochBalancedSampler,
    Sample,
    RawEdfSegmentReader,
    build_raw_reader,
    fit_channel_preprocessor,
)
from epilepsy.models import EpilepsyGATNet, LightSeizureNet
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


def build_model(config, device: torch.device) -> torch.nn.Module:
    architecture = config.model.architecture.lower()
    if architecture in {"light_seizure_net", "lightseizurenet", "light"}:
        return LightSeizureNet(
            num_channels=config.data.num_channels,
            num_classes=config.model.num_classes,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            sample_rate=config.data.sample_rate,
        ).to(device)
    if architecture not in {"epilepsy_gat", "gat"}:
        raise ValueError(f"Unsupported model architecture: {config.model.architecture}")

    model_kwargs = {}
    if config.data.source.lower() == "raw_edf":
        if config.data.num_channels != len(TARGET_CHANNELS):
            raise ValueError(
                f"Raw CHB-MIT uses {len(TARGET_CHANNELS)} common channels, "
                f"but data.num_channels={config.data.num_channels}"
            )
        model_kwargs["channel_names"] = TARGET_CHANNELS
    return EpilepsyGATNet(
        fs=config.data.sample_rate,
        num_classes=config.model.num_classes,
        feature_dim=config.model.feature_dim,
        hid_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        graph_dropout=config.model.graph_dropout,
        spectral_fusion=config.model.spectral_fusion,
        classical_fusion=config.model.classical_fusion,
        classical_hidden_dim=config.model.classical_hidden_dim,
        auxiliary_weight=config.model.auxiliary_weight,
        **model_kwargs,
    ).to(device)


def make_eval_loader(
    samples: Sequence[Sample],
    indices: np.ndarray,
    preprocessor: ChannelPreprocessor,
    config,
    raw_reader: RawEdfSegmentReader | None = None,
) -> DataLoader:
    if config.data.source.lower() == "raw_edf" and raw_reader is None:
        raw_reader = build_raw_reader(config.data)
    return DataLoader(
        ChbmitPoolDataset(samples, indices, preprocessor, raw_reader=raw_reader),
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )


def fit_fold(
    samples: Sequence[Sample],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config,
    device: torch.device,
    seed: int,
    max_epochs: int | None = None,
    epoch_callback: Callable[[int, float, Metrics], None] | None = None,
    raw_reader: RawEdfSegmentReader | None = None,
) -> FoldTrainingResult:
    train_config = config.train
    if max_epochs is not None:
        train_config = replace(train_config, num_epochs=max_epochs)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if config.data.source.lower() == "raw_edf" and raw_reader is None:
        raw_reader = build_raw_reader(config.data)
    statistics_dataset = ChbmitPoolDataset(
        samples, train_indices, raw_reader=raw_reader
    )
    preprocessor = fit_channel_preprocessor(
        statistics_dataset,
        lower_quantile=train_config.clip_lower_quantile,
        upper_quantile=train_config.clip_upper_quantile,
        max_segments=train_config.statistics_max_segments,
        random_state=seed,
    )
    train_dataset = ChbmitPoolDataset(
        samples, train_indices, preprocessor, raw_reader=raw_reader
    )
    val_dataset = ChbmitPoolDataset(
        samples, val_indices, preprocessor, raw_reader=raw_reader
    )
    sampling_strategy = train_config.sampling_strategy.lower()
    sampler = None
    if sampling_strategy == "balanced":
        sampler = EpochBalancedSampler(
            train_dataset.labels,
            seed=seed,
            negative_ratio=train_config.negative_ratio,
        )
    elif sampling_strategy != "all":
        raise ValueError(
            f"Unsupported training sampling strategy: {train_config.sampling_strategy}"
        )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator if sampler is None else None,
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
            model,
            device,
            val_loader,
            criterion,
            segment_duration=config.data.segment_duration,
        )
        threshold, val_metrics = select_decision_threshold(
            y_true,
            y_score,
            metric_name=train_config.threshold_metric,
            loss=raw_metrics.loss,
            min_threshold=train_config.min_threshold,
            max_threshold=train_config.max_threshold,
            segment_duration=config.data.segment_duration,
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
        score_improved = score > best_score and not np.isclose(score, best_score)
        checkpoint_improved = score_improved or (
            np.isclose(score, best_score) and val_metrics.loss < best_loss
        )
        if checkpoint_improved:
            best_score = score
            best_loss = val_metrics.loss
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
        if score_improved:
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
        train_epoch_size=len(sampler) if sampler is not None else len(train_dataset),
    )
