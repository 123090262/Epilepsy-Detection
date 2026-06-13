"""Train epilepsy classification experiments with stratified cross-validation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train epilepsy classifier with stratified K-fold cross-validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional suffix for the output run directory.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=10,
        help="Number of stratified cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load config and dataset summary; do not train.",
    )
    return parser.parse_args()


def split_train_val(indices, labels, val_size: float, random_state: int):
    import numpy as np
    from sklearn.model_selection import train_test_split

    stratify = labels if len(np.unique(labels)) == 2 else None
    return train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )


@dataclass
class BinaryMetrics:
    loss: float
    accuracy: float
    auc: float
    pr_auc: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    f1: float
    threshold: float


def calculate_binary_metrics(y_true, y_score, loss: float, threshold: float):
    import numpy as np
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

    auc = math.nan
    pr_auc = math.nan
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = BinaryMetrics(
        loss=float(loss),
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc=auc,
        pr_auc=pr_auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(tn / max(tn + fp, 1)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
    )
    return metrics, y_pred


def evaluate_binary(model, device, loader, criterion, threshold: float = 0.5):
    import numpy as np
    import torch

    model.eval()
    y_true = []
    y_score = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            probabilities = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * data.size(0)
            total_items += data.size(0)
            y_true.extend(target.cpu().numpy().tolist())
            y_score.extend(probabilities[:, 1].cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    metrics, y_pred = calculate_binary_metrics(
        y_true,
        y_score,
        loss=total_loss / max(total_items, 1),
        threshold=threshold,
    )
    return metrics, y_true, y_pred, y_score


def select_decision_threshold(y_true, y_score, metric_name: str, loss: float):
    import numpy as np

    supported_metrics = {"f1", "balanced_accuracy"}
    if metric_name not in supported_metrics:
        raise ValueError(
            f"Unsupported threshold metric: {metric_name}. "
            f"Choose one of {sorted(supported_metrics)}"
        )
    if len(np.unique(y_true)) != 2:
        raise ValueError("threshold selection requires both validation classes")

    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score)))
    best_threshold = 0.5
    best_metrics, _ = calculate_binary_metrics(
        y_true, y_score, loss=loss, threshold=best_threshold
    )
    best_value = getattr(best_metrics, metric_name)

    for threshold in candidates:
        metrics, _ = calculate_binary_metrics(
            y_true, y_score, loss=loss, threshold=float(threshold)
        )
        value = getattr(metrics, metric_name)
        if value > best_value or (
            value == best_value
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value = value
    return best_threshold, best_metrics


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader

    from epilepsy.config import dump_config_copy, load_config
    from epilepsy.data import (
        ChbmitPoolDataset,
        EpochBalancedSampler,
        fit_channel_preprocessor,
        load_pool_samples,
    )
    from epilepsy.evaluate import save_summary
    from epilepsy.models import EpilepsyGATNet
    from epilepsy.utils import append_csv_row, make_run_dir, setup_logger

    config = load_config(args.config)
    if args.num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    run_dir = make_run_dir(config.output.run_dir, args.run_name)
    checkpoint_root = Path(config.output.checkpoint_dir) / run_dir.name
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "training.log")
    dump_config_copy(args.config, run_dir / "config.yaml")

    logger.info("Loading NPY pool index...")
    samples = load_pool_samples(config.data)
    y = np.asarray([sample.label for sample in samples], dtype=np.int64)
    patient_ids = np.asarray([sample.patient for sample in samples])
    class_counts = np.bincount(y, minlength=2)
    if np.any(class_counts < args.num_folds):
        raise ValueError(
            f"Each class needs at least {args.num_folds} samples for stratified "
            f"cross-validation; got normal={class_counts[0]}, seizure={class_counts[1]}"
        )

    logger.info(
        "Indexed %d segments with shape (%d, %d)",
        len(samples),
        config.data.num_channels,
        config.data.segment_length,
    )
    logger.info(
        "Label counts: normal=%d seizure=%d", class_counts[0], class_counts[1]
    )
    logger.info("Patients: %s", ", ".join(np.unique(patient_ids).tolist()))
    logger.info(
        "Cross-validation: StratifiedKFold(n_splits=%d, shuffle=True, random_state=%d)",
        args.num_folds,
        config.train.random_state,
    )

    if args.dry_run:
        logger.info("Dry run finished. No training was started.")
        return

    from epilepsy.plots import plot_confusion_matrix, plot_training_summary
    from epilepsy.train import train_one_epoch

    threshold_metric = getattr(config.train, "threshold_metric", "f1")
    checkpoint_metric = getattr(config.train, "checkpoint_metric", "f1")
    logger.info(
        "Threshold metric=%s | Checkpoint metric=%s",
        threshold_metric,
        checkpoint_metric,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    splitter = StratifiedKFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=config.train.random_state,
    )
    all_fold_metrics: list[dict[str, float | int]] = []
    all_indices = np.arange(len(samples))

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(all_indices, y), start=1
    ):
        fold_name = f"fold_{fold:02d}"
        logger.info("===== Fold %d/%d =====", fold, args.num_folds)

        train_sub_indices, val_indices = split_train_val(
            train_indices,
            y[train_indices],
            val_size=config.train.val_size,
            random_state=config.train.random_state + fold,
        )
        train_stats_dataset = ChbmitPoolDataset(samples, train_sub_indices)
        preprocessor = fit_channel_preprocessor(
            train_stats_dataset,
            lower_quantile=config.train.clip_lower_quantile,
            upper_quantile=config.train.clip_upper_quantile,
            max_segments=config.train.statistics_max_segments,
            random_state=config.train.random_state + fold,
        )
        train_dataset = ChbmitPoolDataset(samples, train_sub_indices, preprocessor)
        val_dataset = ChbmitPoolDataset(samples, val_indices, preprocessor)
        test_dataset = ChbmitPoolDataset(samples, test_indices, preprocessor)
        train_sampler = EpochBalancedSampler(
            train_dataset.labels, seed=config.train.random_state + fold
        )

        logger.info(
            "Train pool=%d | Train epoch=%d (dynamic 1:1) | Val=%d | Test=%d | "
            "Test seizure ratio=%.4f",
            len(train_dataset),
            len(train_sampler),
            len(val_dataset),
            len(test_dataset),
            float(np.mean(y[test_indices])),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            sampler=train_sampler,
            drop_last=False,
            num_workers=config.train.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
        )

        model = EpilepsyGATNet(
            fs=config.data.sample_rate,
            num_classes=config.model.num_classes,
            feature_dim=config.model.feature_dim,
            hid_dim=config.model.hidden_dim,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        best_val_score = -1.0
        best_val_acc = -1.0
        best_threshold = 0.5
        best_epoch = -1
        best_model_path = checkpoint_root / f"best_model_{fold_name}.pth"
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        for epoch in range(1, config.train.num_epochs + 1):
            train_loss = train_one_epoch(
                model, device, train_loader, optimizer, criterion
            )
            val_metrics, val_y_true, _, val_y_score = evaluate_binary(
                model, device, val_loader, criterion
            )
            val_threshold, val_metrics = select_decision_threshold(
                val_y_true,
                val_y_score,
                metric_name=threshold_metric,
                loss=val_metrics.loss,
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics.loss)
            history["val_accuracy"].append(val_metrics.accuracy)
            history["val_precision"].append(val_metrics.precision)
            history["val_recall"].append(val_metrics.recall)
            history["val_f1"].append(val_metrics.f1)

            append_csv_row(
                run_dir / "metrics.csv",
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics.loss,
                    "val_acc": val_metrics.accuracy,
                    "val_auc": val_metrics.auc,
                    "val_pr_auc": val_metrics.pr_auc,
                    "val_specificity": val_metrics.specificity,
                    "val_balanced_acc": val_metrics.balanced_accuracy,
                    "val_threshold": val_threshold,
                    "val_precision": val_metrics.precision,
                    "val_recall": val_metrics.recall,
                    "val_f1": val_metrics.f1,
                },
            )

            val_score = getattr(val_metrics, checkpoint_metric)
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_acc = val_metrics.accuracy
                best_threshold = val_threshold
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "preprocessor": preprocessor.as_dict(),
                        "fold": fold,
                        "num_folds": args.num_folds,
                        "decision_threshold": best_threshold,
                        "threshold_metric": threshold_metric,
                        "checkpoint_metric": checkpoint_metric,
                        "best_val_score": best_val_score,
                    },
                    best_model_path,
                )
                logger.info(
                    "[BEST] fold=%d epoch=%d val_%s=%.4f val_acc=%.4f threshold=%.4f",
                    fold,
                    epoch,
                    checkpoint_metric,
                    best_val_score,
                    best_val_acc,
                    best_threshold,
                )

            logger.info(
                "[Fold %d | Epoch %03d] train_loss=%.4f val_loss=%.4f "
                "val_acc=%.4f val_f1=%.4f threshold=%.4f",
                fold,
                epoch,
                train_loss,
                val_metrics.loss,
                val_metrics.accuracy,
                val_metrics.f1,
                val_threshold,
            )

        plot_training_summary(history, run_dir / f"training_summary_{fold_name}.png")

        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
        test_metrics, y_true, y_pred, _ = evaluate_binary(
            model, device, test_loader, criterion, threshold=decision_threshold
        )
        plot_confusion_matrix(
            y_true, y_pred, run_dir / f"confusion_matrix_{fold_name}.png"
        )

        logger.info(
            "RESULT fold=%d best_epoch=%d best_val_%s=%.4f threshold=%.4f "
            "test_acc=%.4f auc=%.4f pr_auc=%.4f precision=%.4f recall=%.4f "
            "specificity=%.4f balanced_acc=%.4f f1=%.4f",
            fold,
            best_epoch,
            checkpoint_metric,
            best_val_score,
            decision_threshold,
            test_metrics.accuracy,
            test_metrics.auc,
            test_metrics.pr_auc,
            test_metrics.precision,
            test_metrics.recall,
            test_metrics.specificity,
            test_metrics.balanced_accuracy,
            test_metrics.f1,
        )
        all_fold_metrics.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_score": best_val_score,
                "threshold": decision_threshold,
                "acc": test_metrics.accuracy,
                "auc": test_metrics.auc,
                "pr_auc": test_metrics.pr_auc,
                "prec": test_metrics.precision,
                "recall": test_metrics.recall,
                "specificity": test_metrics.specificity,
                "balanced_acc": test_metrics.balanced_accuracy,
                "f1": test_metrics.f1,
            }
        )

    save_summary(run_dir / "summary.json", all_fold_metrics)
    logger.info("Training finished. Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
