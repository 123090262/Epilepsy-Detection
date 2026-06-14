"""Run leave-one-patient-out cross-validation on CHB-MIT."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JNE-aligned CHB-MIT LOPOCV training.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--test-patients", nargs="+", default=None)
    parser.add_argument(
        "--validation-level",
        choices=("segment", "patient"),
        default="segment",
        help=(
            "Use a stratified 10%% split of the development samples as in the "
            "paper, or reserve complete development patients for validation."
        ),
    )
    parser.add_argument(
        "--skip-continuous-eval",
        action="store_true",
        help="Skip the complete held-out-patient timeline audit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split

    from epilepsy.config import dump_config_copy, load_config
    from epilepsy.data import (
        ChbmitPoolDataset,
        build_raw_reader,
        load_continuous_patient_samples,
        load_samples,
        write_sample_manifest,
    )
    from epilepsy.evaluate import save_summary
    from epilepsy.experiment import fit_fold, make_eval_loader
    from epilepsy.plots import plot_confusion_matrix, plot_training_summary
    from epilepsy.splits import grouped_train_val_split
    from epilepsy.train import evaluate
    from epilepsy.utils import append_csv_row, make_run_dir, setup_logger

    config = load_config(args.config)
    run_dir = make_run_dir(config.output.run_dir, args.run_name)
    checkpoint_root = Path(config.output.checkpoint_dir) / run_dir.name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "training.log")
    dump_config_copy(args.config, run_dir / "config.yaml")

    samples = load_samples(config.data)
    write_sample_manifest(samples, run_dir / "sample_manifest.csv")
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    patient_ids = np.asarray([sample.patient for sample in samples])
    available_patients = np.unique(patient_ids)
    test_patients = (
        available_patients if args.test_patients is None else np.asarray(args.test_patients)
    )
    missing = sorted(set(test_patients) - set(available_patients))
    if missing:
        raise ValueError(f"Test patients not found in indexed data: {missing}")

    logger.info(
        "LOPOCV: each test patient is absent from training and validation; "
        "development validation level=%s",
        args.validation_level,
    )
    logger.info(
        "Primary paper-comparable evaluation uses the JNE 2-3x sampled data; "
        "continuous evaluation=%s",
        "off" if args.skip_continuous_eval else "on",
    )
    if args.dry_run:
        for test_patient in test_patients:
            logger.info(
                "%s development=%d sampled_test=%d",
                test_patient,
                int(np.sum(patient_ids != test_patient)),
                int(np.sum(patient_ids == test_patient)),
            )
        logger.info("Dry run finished. No training was started.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_reader = (
        build_raw_reader(config.data) if config.data.source.lower() == "raw_edf" else None
    )
    logger.info("Using device: %s", device)
    sampled_rows: list[dict[str, float | int | str]] = []
    continuous_rows: list[dict[str, float | int | str]] = []

    for fold, test_patient in enumerate(test_patients, start=1):
        outer_train = np.flatnonzero(patient_ids != test_patient)
        sampled_test_indices = np.flatnonzero(patient_ids == test_patient)
        if args.validation_level == "patient":
            train_indices, val_indices = grouped_train_val_split(
                outer_train,
                labels[outer_train],
                patient_ids[outer_train],
                config.train.val_size,
                config.train.random_state + fold,
            )
        else:
            train_indices, val_indices = train_test_split(
                outer_train,
                test_size=config.train.val_size,
                random_state=config.train.random_state + fold,
                stratify=labels[outer_train],
            )
        logger.info(
            "===== Fold %d/%d | test=%s | train=%d val=%d sampled_test=%d =====",
            fold,
            len(test_patients),
            test_patient,
            len(train_indices),
            len(val_indices),
            len(sampled_test_indices),
        )

        def on_epoch(epoch, train_loss, metrics):
            append_csv_row(
                run_dir / "metrics.csv",
                {
                    "fold": fold,
                    "patient": test_patient,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in metrics.as_dict().items()},
                },
            )
            logger.info(
                "[%s | Epoch %03d] loss=%.4f val_acc=%.4f val_f1=%.4f",
                test_patient,
                epoch,
                train_loss,
                metrics.accuracy,
                metrics.f1,
            )

        result = fit_fold(
            samples,
            train_indices,
            val_indices,
            config,
            device,
            seed=config.train.random_state + fold,
            epoch_callback=on_epoch,
            raw_reader=raw_reader,
        )
        plot_training_summary(
            result.history, run_dir / f"training_summary_{test_patient}.png"
        )
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "preprocessor": result.preprocessor.as_dict(),
                "decision_threshold": result.threshold,
                "protocol": "leave_one_patient_out",
                "test_patient": test_patient,
                "validation_level": args.validation_level,
                "validation_patients": np.unique(patient_ids[val_indices]).tolist(),
                "best_epoch": result.best_epoch,
                "best_val_metrics": result.best_metrics.as_dict(),
            },
            checkpoint_root / f"best_model_test_{test_patient}.pth",
        )

        sampled_loader = make_eval_loader(
            samples,
            sampled_test_indices,
            result.preprocessor,
            config,
            raw_reader=raw_reader,
        )
        sampled_metrics, y_true, y_pred, _ = evaluate(
            result.model,
            device,
            sampled_loader,
            result.criterion,
            threshold=result.threshold,
            segment_duration=config.data.segment_duration,
        )
        plot_confusion_matrix(
            y_true, y_pred, run_dir / f"confusion_sampled_{test_patient}.png"
        )
        sampled_row = metric_row(test_patient, result.best_epoch, sampled_metrics)
        sampled_rows.append(sampled_row)
        logger.info(
            "SAMPLED patient=%s acc=%.4f f1=%.4f sensitivity=%.4f "
            "precision=%.4f FPR/h=%.3f",
            test_patient,
            sampled_metrics.accuracy,
            sampled_metrics.f1,
            sampled_metrics.recall,
            sampled_metrics.precision,
            sampled_metrics.fpr_per_hour,
        )

        if not args.skip_continuous_eval:
            if config.data.source.lower() != "raw_edf":
                raise ValueError("Continuous LOPO evaluation requires data.source=raw_edf")
            continuous_samples = load_continuous_patient_samples(config.data, test_patient)
            continuous_reader = build_raw_reader(
                replace(config.data, raw_cache_segments=0)
            )
            continuous_loader = torch.utils.data.DataLoader(
                ChbmitPoolDataset(
                    continuous_samples,
                    preprocessor=result.preprocessor,
                    raw_reader=continuous_reader,
                ),
                batch_size=config.train.batch_size,
                shuffle=False,
                num_workers=config.train.num_workers,
            )
            continuous_metrics, y_true, y_pred, _ = evaluate(
                result.model,
                device,
                continuous_loader,
                result.criterion,
                threshold=result.threshold,
                segment_duration=config.data.segment_duration,
            )
            plot_confusion_matrix(
                y_true, y_pred, run_dir / f"confusion_continuous_{test_patient}.png"
            )
            continuous_rows.append(
                metric_row(test_patient, result.best_epoch, continuous_metrics)
            )
            logger.info(
                "CONTINUOUS patient=%s hours=%.2f acc=%.4f f1=%.4f "
                "sensitivity=%.4f FPR/h=%.3f",
                test_patient,
                len(continuous_samples) * config.data.segment_duration / 3600.0,
                continuous_metrics.accuracy,
                continuous_metrics.f1,
                continuous_metrics.recall,
                continuous_metrics.fpr_per_hour,
            )

    save_summary(run_dir / "summary_sampled.json", sampled_rows)
    if continuous_rows:
        save_summary(run_dir / "summary_continuous.json", continuous_rows)
    logger.info("Training finished. Outputs saved to %s", run_dir)


def metric_row(patient: str, best_epoch: int, metrics) -> dict[str, float | int | str]:
    return {
        "patient": patient,
        "best_epoch": best_epoch,
        "threshold": metrics.threshold,
        "acc": metrics.accuracy,
        "auc": metrics.auc,
        "pr_auc": metrics.pr_auc,
        "prec": metrics.precision,
        "recall": metrics.recall,
        "specificity": metrics.specificity,
        "fpr_per_hour": metrics.fpr_per_hour,
        "balanced_acc": metrics.balanced_accuracy,
        "f1": metrics.f1,
    }


if __name__ == "__main__":
    main()
