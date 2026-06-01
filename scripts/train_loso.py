"""Train leave-one-subject-out epilepsy classification experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train epilepsy classifier with leave-one-subject-out splits."
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


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
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

    run_dir = make_run_dir(config.output.run_dir, args.run_name)
    checkpoint_root = Path(config.output.checkpoint_dir) / run_dir.name
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "training.log")
    dump_config_copy(args.config, run_dir / "config.yaml")

    logger.info("Loading NPY pool index...")
    samples = load_pool_samples(config.data)
    y = np.asarray([sample.label for sample in samples], dtype=np.int64)
    patient_ids = np.asarray([sample.patient for sample in samples])
    unique_patients = np.unique(patient_ids)
    logger.info(
        "Indexed %d segments with shape (%d, %d)",
        len(samples),
        config.data.num_channels,
        config.data.segment_length,
    )
    logger.info("Label counts: normal=%d seizure=%d", int(np.sum(y == 0)), int(np.sum(y == 1)))
    logger.info("Patients: %s", ", ".join(unique_patients.tolist()))

    invalid = [p for p in config.data.test_patients if p not in unique_patients]
    if invalid:
        raise ValueError(
            f"Test patients not found in dataset: {invalid}. "
            f"Available patients: {unique_patients.tolist()}"
        )

    if args.dry_run:
        logger.info("Dry run finished. No training was started.")
        return

    from epilepsy.plots import plot_confusion_matrix, plot_training_summary
    from epilepsy.train import evaluate, train_one_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    all_fold_metrics: list[dict[str, float | str]] = []

    for fold, test_pid in enumerate(config.data.test_patients, start=1):
        logger.info("===== Fold %d/%d | Test Patient = %s =====", fold, len(config.data.test_patients), test_pid)

        train_indices = np.flatnonzero(patient_ids != test_pid)
        test_indices = np.flatnonzero(patient_ids == test_pid)
        train_sub_indices, val_indices = split_train_val(
            train_indices,
            y[train_indices],
            val_size=config.train.val_size,
            random_state=config.train.random_state,
        )
        train_stats_dataset = ChbmitPoolDataset(samples, train_sub_indices)
        preprocessor = fit_channel_preprocessor(
            train_stats_dataset,
            lower_quantile=config.train.clip_lower_quantile,
            upper_quantile=config.train.clip_upper_quantile,
            max_segments=config.train.statistics_max_segments,
            random_state=config.train.random_state,
        )
        train_dataset = ChbmitPoolDataset(samples, train_sub_indices, preprocessor)
        val_dataset = ChbmitPoolDataset(samples, val_indices, preprocessor)
        test_dataset = ChbmitPoolDataset(samples, test_indices, preprocessor)
        train_sampler = EpochBalancedSampler(
            train_dataset.labels, seed=config.train.random_state
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
            drop_last=True,
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

        best_val_acc = -1.0
        best_epoch = -1
        best_model_path = checkpoint_root / f"best_model_test_{test_pid}.pth"

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        for epoch in range(1, config.train.num_epochs + 1):
            train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
            val_metrics, _, _, _ = evaluate(model, device, val_loader, criterion)

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
                    "patient": test_pid,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics.loss,
                    "val_acc": val_metrics.accuracy,
                    "val_auc": val_metrics.auc,
                    "val_precision": val_metrics.precision,
                    "val_recall": val_metrics.recall,
                    "val_f1": val_metrics.f1,
                },
            )

            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "preprocessor": preprocessor.as_dict(),
                        "test_patient": test_pid,
                    },
                    best_model_path,
                )
                logger.info(
                    "[BEST] fold=%d epoch=%d val_acc=%.4f",
                    fold,
                    epoch,
                    best_val_acc,
                )

            logger.info(
                "[Fold %d | Epoch %03d] train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                fold,
                epoch,
                train_loss,
                val_metrics.loss,
                val_metrics.accuracy,
            )

        plot_training_summary(history, run_dir / f"training_summary_{test_pid}.png")

        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, y_true, y_pred, _ = evaluate(model, device, test_loader, criterion)
        plot_confusion_matrix(y_true, y_pred, run_dir / f"confusion_matrix_{test_pid}.png")

        logger.info(
            "RESULT patient=%s best_epoch=%d best_val_acc=%.4f "
            "test_acc=%.4f auc=%.4f precision=%.4f recall=%.4f f1=%.4f",
            test_pid,
            best_epoch,
            best_val_acc,
            test_metrics.accuracy,
            test_metrics.auc,
            test_metrics.precision,
            test_metrics.recall,
            test_metrics.f1,
        )

        all_fold_metrics.append(
            {
                "patient": test_pid,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "acc": test_metrics.accuracy,
                "auc": test_metrics.auc,
                "prec": test_metrics.precision,
                "recall": test_metrics.recall,
                "f1": test_metrics.f1,
            }
        )

    save_summary(run_dir / "summary.json", all_fold_metrics)
    logger.info("Training finished. Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
