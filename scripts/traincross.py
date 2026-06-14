"""Run JNE-aligned patient-specific ten-fold cross-validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-specific ten-fold seizure detection training."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument(
        "--split-level",
        choices=("segment", "event"),
        default="segment",
        help=(
            "Paper-comparable random segment folds are the default. Event folds "
            "keep overlapping windows from one seizure in the same fold."
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
    from epilepsy.data import build_raw_reader, load_samples, write_sample_manifest
    from epilepsy.evaluate import save_summary
    from epilepsy.experiment import fit_fold, make_eval_loader
    from epilepsy.plots import plot_confusion_matrix, plot_training_summary
    from epilepsy.splits import grouped_kfold_splits
    from epilepsy.train import calculate_binary_metrics, evaluate
    from epilepsy.utils import append_csv_row, make_run_dir, setup_logger

    if args.num_folds < 2:
        raise ValueError("num-folds must be at least 2")
    config = load_config(args.config)
    if args.patients is not None:
        config = replace(
            config,
            data=replace(config.data, test_patients=tuple(args.patients)),
        )
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
    patients = available_patients if args.patients is None else np.asarray(args.patients)
    missing = sorted(set(patients) - set(available_patients))
    if missing:
        raise ValueError(f"Patients not found in indexed data: {missing}")

    logger.info(
        "JNE patient-specific protocol: 1 s windows, %.1f s seizure overlap, "
        "%.1f-%.1fx non-seizure duration, %d-fold CV",
        config.data.seizure_overlap,
        config.data.non_seizure_ratio_min,
        config.data.non_seizure_ratio_max,
        args.num_folds,
    )
    logger.info(
        "Split level=%s; segment mode matches the paper but overlapping seizure "
        "windows can occur in different folds",
        args.split_level,
    )

    planned_splits: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for patient in patients:
        patient_indices = np.flatnonzero(patient_ids == patient)
        patient_labels = labels[patient_indices]
        if min(np.bincount(patient_labels, minlength=2)) < args.num_folds:
            raise ValueError(f"{patient} has too few samples for {args.num_folds} folds")
        if args.split_level == "segment":
            splitter = StratifiedKFold(
                n_splits=args.num_folds,
                shuffle=True,
                random_state=config.train.random_state,
            )
            iterator = (
                (patient_indices[train_rel], patient_indices[test_rel])
                for train_rel, test_rel in splitter.split(patient_indices, patient_labels)
            )
        else:
            event_ids = np.asarray(
                [
                    getattr(samples[index], "event_id", samples[index].record)
                    for index in patient_indices
                ]
            )
            iterator = grouped_kfold_splits(
                patient_indices,
                patient_labels,
                event_ids,
                args.num_folds,
                config.train.random_state,
            )
        for fold, (outer_train, test_indices) in enumerate(iterator, start=1):
            planned_splits.append((str(patient), fold, outer_train, test_indices))
        logger.info(
            "%s samples=%d seizure=%d non_seizure=%d",
            patient,
            len(patient_indices),
            int(np.sum(patient_labels == 1)),
            int(np.sum(patient_labels == 0)),
        )

    if args.dry_run:
        logger.info("Validated %d patient-folds. No training was started.", len(planned_splits))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_reader = (
        build_raw_reader(config.data) if config.data.source.lower() == "raw_edf" else None
    )
    logger.info("Using device: %s", device)
    all_fold_metrics: list[dict[str, float | int | str]] = []
    patient_metrics: dict[str, list[dict[str, float | int | str]]] = {}
    patient_predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for global_fold, (patient, fold, outer_train, test_indices) in enumerate(
        planned_splits, start=1
    ):
        train_indices, val_indices = train_test_split(
            outer_train,
            test_size=config.train.val_size,
            random_state=config.train.random_state + global_fold,
            stratify=labels[outer_train],
        )
        fold_name = f"{patient}_fold_{fold:02d}"
        logger.info(
            "===== %s | train=%d val=%d test=%d test_pos=%.4f =====",
            fold_name,
            len(train_indices),
            len(val_indices),
            len(test_indices),
            float(np.mean(labels[test_indices])),
        )

        def on_epoch(epoch, train_loss, metrics):
            append_csv_row(
                run_dir / "metrics.csv",
                {
                    "patient": patient,
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in metrics.as_dict().items()},
                },
            )
            logger.info(
                "[%s | Epoch %03d] loss=%.4f val_acc=%.4f val_f1=%.4f",
                fold_name,
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
            seed=config.train.random_state + global_fold,
            epoch_callback=on_epoch,
            raw_reader=raw_reader,
        )
        plot_training_summary(result.history, run_dir / f"training_{fold_name}.png")
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "preprocessor": result.preprocessor.as_dict(),
                "decision_threshold": result.threshold,
                "protocol": "patient_specific_kfold",
                "patient": patient,
                "fold": fold,
                "split_level": args.split_level,
                "best_epoch": result.best_epoch,
                "best_val_metrics": result.best_metrics.as_dict(),
            },
            checkpoint_root / f"best_model_{fold_name}.pth",
        )

        test_loader = make_eval_loader(
            samples,
            test_indices,
            result.preprocessor,
            config,
            raw_reader=raw_reader,
        )
        test_metrics, y_true, y_pred, y_score = evaluate(
            result.model,
            device,
            test_loader,
            result.criterion,
            threshold=result.threshold,
            segment_duration=config.data.segment_duration,
        )
        plot_confusion_matrix(y_true, y_pred, run_dir / f"confusion_{fold_name}.png")
        row = {
            "patient": patient,
            "fold": fold,
            "best_epoch": result.best_epoch,
            "threshold": result.threshold,
            "acc": test_metrics.accuracy,
            "auc": test_metrics.auc,
            "pr_auc": test_metrics.pr_auc,
            "prec": test_metrics.precision,
            "recall": test_metrics.recall,
            "specificity": test_metrics.specificity,
            "fpr_per_hour": test_metrics.fpr_per_hour,
            "balanced_acc": test_metrics.balanced_accuracy,
            "f1": test_metrics.f1,
        }
        all_fold_metrics.append(row)
        patient_metrics.setdefault(patient, []).append(row)
        patient_predictions.setdefault(patient, []).append((y_true, y_score))
        logger.info(
            "RESULT %s acc=%.4f f1=%.4f sensitivity=%.4f precision=%.4f FPR/h=%.3f",
            fold_name,
            test_metrics.accuracy,
            test_metrics.f1,
            test_metrics.recall,
            test_metrics.precision,
            test_metrics.fpr_per_hour,
        )

    for patient, rows in patient_metrics.items():
        save_summary(run_dir / f"summary_{patient}.json", rows)
    pooled_by_patient = {}
    all_y_true = []
    all_y_score = []
    for patient, predictions in patient_predictions.items():
        patient_y_true = np.concatenate([values[0] for values in predictions])
        patient_y_score = np.concatenate([values[1] for values in predictions])
        pooled_metrics, _ = calculate_binary_metrics(
            patient_y_true,
            patient_y_score,
            loss=0.0,
            threshold=0.5,
            segment_duration=config.data.segment_duration,
        )
        pooled_by_patient[patient] = pooled_metrics.as_dict()
        all_y_true.append(patient_y_true)
        all_y_score.append(patient_y_score)
    overall_metrics, _ = calculate_binary_metrics(
        np.concatenate(all_y_true),
        np.concatenate(all_y_score),
        loss=0.0,
        threshold=0.5,
        segment_duration=config.data.segment_duration,
    )
    (run_dir / "pooled_metrics.json").write_text(
        json.dumps(
            {"overall": overall_metrics.as_dict(), "patients": pooled_by_patient},
            indent=2,
        ),
        encoding="utf-8",
    )
    save_summary(run_dir / "summary.json", all_fold_metrics)
    logger.info("Training finished. Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
