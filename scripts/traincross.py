"""Train epilepsy classifier with stratified or grouped cross-validation."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-fold training for EEG seizure classification."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument(
        "--group-level",
        choices=("segment", "record", "patient"),
        default="segment",
        help=(
            "Split individual segments by default, or keep all segments from "
            "an EDF record/patient in one fold."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from epilepsy.config import dump_config_copy, load_config
    from epilepsy.data import load_pool_samples
    from epilepsy.evaluate import save_summary
    from epilepsy.experiment import fit_fold, make_eval_loader
    from epilepsy.plots import plot_confusion_matrix, plot_training_summary
    from epilepsy.splits import grouped_kfold_splits, grouped_train_val_split
    from epilepsy.train import evaluate
    from epilepsy.utils import append_csv_row, make_run_dir, setup_logger

    config = load_config(args.config)
    if args.num_folds < 2:
        raise ValueError("num-folds must be at least 2")

    run_dir = make_run_dir(config.output.run_dir, args.run_name)
    checkpoint_root = Path(config.output.checkpoint_dir) / run_dir.name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "training.log")
    dump_config_copy(args.config, run_dir / "config.yaml")

    samples = load_pool_samples(config.data)
    y = np.asarray([sample.label for sample in samples], dtype=np.int64)
    patient_ids = np.asarray([sample.patient for sample in samples])
    record_ids = np.asarray([sample.record for sample in samples])
    groups = record_ids if args.group_level == "record" else patient_ids
    indices = np.arange(len(samples))
    logger.info(
        "Indexed %d segments: normal=%d seizure=%d patients=%d records=%d",
        len(samples),
        int(np.sum(y == 0)),
        int(np.sum(y == 1)),
        len(np.unique(patient_ids)),
        len(np.unique(record_ids)),
    )
    if args.group_level == "segment":
        splitter = StratifiedKFold(
            n_splits=args.num_folds,
            shuffle=True,
            random_state=config.train.random_state,
        )
        splits = splitter.split(indices, y)
        logger.info(
            "Stratified random-segment %d-fold CV (shuffle=True, random_state=%d)",
            args.num_folds,
            config.train.random_state,
        )
    else:
        splits = grouped_kfold_splits(
            indices, y, groups, args.num_folds, config.train.random_state
        )
        logger.info(
            "Grouped %d-fold CV by %s; no group can appear in both train and test",
            args.num_folds,
            args.group_level,
        )
    if args.dry_run:
        list(splits)
        logger.info("Dry run finished. No training was started.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    all_fold_metrics = []

    for fold, (outer_train, test_indices) in enumerate(splits, start=1):
        fold_name = f"fold_{fold:02d}"
        if args.group_level == "segment":
            train_indices, val_indices = train_test_split(
                outer_train,
                test_size=config.train.val_size,
                random_state=config.train.random_state + fold,
                stratify=y[outer_train],
            )
        else:
            train_indices, val_indices = grouped_train_val_split(
                outer_train,
                y[outer_train],
                groups[outer_train],
                config.train.val_size,
                config.train.random_state + fold,
            )
        logger.info(
            "===== Fold %d/%d | train=%d val=%d test=%d test_pos=%.4f =====",
            fold,
            args.num_folds,
            len(train_indices),
            len(val_indices),
            len(test_indices),
            float(np.mean(y[test_indices])),
        )

        def on_epoch(epoch, train_loss, metrics):
            append_csv_row(
                run_dir / "metrics.csv",
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in metrics.as_dict().items()},
                },
            )
            logger.info(
                "[Fold %d | Epoch %03d] loss=%.4f val_acc=%.4f "
                "val_f1=%.4f val_bal_acc=%.4f threshold=%.3f",
                fold,
                epoch,
                train_loss,
                metrics.accuracy,
                metrics.f1,
                metrics.balanced_accuracy,
                metrics.threshold,
            )

        result = fit_fold(
            samples,
            train_indices,
            val_indices,
            config,
            device,
            seed=config.train.random_state + fold,
            epoch_callback=on_epoch,
        )
        plot_training_summary(
            result.history, run_dir / f"training_summary_{fold_name}.png"
        )
        checkpoint_path = checkpoint_root / f"best_model_{fold_name}.pth"
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "preprocessor": result.preprocessor.as_dict(),
                "decision_threshold": result.threshold,
                "fold": fold,
                "group_level": args.group_level,
                "best_epoch": result.best_epoch,
                "best_val_metrics": result.best_metrics.as_dict(),
            },
            checkpoint_path,
        )

        test_loader = make_eval_loader(
            samples, test_indices, result.preprocessor, config
        )
        test_metrics, y_true, y_pred, _ = evaluate(
            result.model,
            device,
            test_loader,
            result.criterion,
            threshold=result.threshold,
        )
        plot_confusion_matrix(
            y_true, y_pred, run_dir / f"confusion_matrix_{fold_name}.png"
        )
        logger.info(
            "RESULT fold=%d epoch=%d threshold=%.3f acc=%.4f auc=%.4f "
            "pr_auc=%.4f precision=%.4f recall=%.4f specificity=%.4f "
            "balanced_acc=%.4f f1=%.4f",
            fold,
            result.best_epoch,
            result.threshold,
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
                "best_epoch": result.best_epoch,
                "threshold": result.threshold,
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
