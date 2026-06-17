"""Run all-selected-patient mixed ten-fold cross-validation.

This protocol pools windows from all selected patients before making folds. It is
intended as a capacity/generalization diagnostic and is intentionally separate
from scripts/traincross.py, which performs patient-specific folds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed-patient ten-fold seizure detection training."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--run-name", type=str, default="mixed_kfold")
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional patient subset to pool before mixed k-fold splitting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from epilepsy.config import dump_config_copy, load_config
    from epilepsy.data import build_raw_reader, load_samples, write_sample_manifest
    from epilepsy.evaluate import save_summary
    from epilepsy.experiment import fit_fold, make_eval_loader
    from epilepsy.plots import plot_confusion_matrix, plot_training_summary
    from epilepsy.train import calculate_binary_metrics_from_predictions, evaluate
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

    selected_indices = np.flatnonzero(np.isin(patient_ids, patients))
    selected_labels = labels[selected_indices]
    if min(np.bincount(selected_labels, minlength=2)) < args.num_folds:
        raise ValueError(
            "The pooled selected data has too few samples in at least one class "
            f"for {args.num_folds} folds"
        )

    logger.info(
        "Mixed-patient protocol: pooled %d patients into %d stratified folds",
        len(patients),
        args.num_folds,
    )
    logger.info(
        "This is not patient-specific CV: samples from the same patient may appear "
        "in train, validation, and test within different folds."
    )
    for patient in patients:
        patient_indices = np.flatnonzero(patient_ids == patient)
        logger.info(
            "%s samples=%d seizure=%d non_seizure=%d",
            patient,
            len(patient_indices),
            int(np.sum(labels[patient_indices] == 1)),
            int(np.sum(labels[patient_indices] == 0)),
        )

    splitter = StratifiedKFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=config.train.random_state,
    )
    planned_splits = [
        (fold, selected_indices[train_rel], selected_indices[test_rel])
        for fold, (train_rel, test_rel) in enumerate(
            splitter.split(selected_indices, selected_labels), start=1
        )
    ]

    if args.dry_run:
        logger.info("Validated %d mixed folds. No training was started.", len(planned_splits))
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_reader = (
        build_raw_reader(config.data) if config.data.source.lower() == "raw_edf" else None
    )
    logger.info("Using device: %s", device)

    all_fold_metrics: list[dict[str, float | int | str]] = []
    patient_predictions: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for fold, outer_train, test_indices in planned_splits:
        train_indices, val_indices = train_test_split(
            outer_train,
            test_size=config.train.val_size,
            random_state=config.train.random_state + fold,
            stratify=labels[outer_train],
        )
        fold_name = f"mixed_fold_{fold:02d}"
        test_patient_count = len(np.unique(patient_ids[test_indices]))
        logger.info(
            "===== %s | train=%d val=%d test=%d test_pos=%.4f test_patients=%d =====",
            fold_name,
            len(train_indices),
            len(val_indices),
            len(test_indices),
            float(np.mean(labels[test_indices])),
            test_patient_count,
        )

        def on_epoch(epoch, train_loss, metrics):
            append_csv_row(
                run_dir / "metrics.csv",
                {
                    "protocol": "mixed_patient_kfold",
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
            seed=config.train.random_state + fold,
            epoch_callback=on_epoch,
            raw_reader=raw_reader,
        )
        plot_training_summary(result.history, run_dir / f"training_{fold_name}.png")
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "preprocessor": result.preprocessor.as_dict(),
                "decision_threshold": result.threshold,
                "protocol": "mixed_patient_kfold",
                "fold": fold,
                "patients": list(map(str, patients)),
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
            "protocol": "mixed_patient_kfold",
            "fold": fold,
            "best_epoch": result.best_epoch,
            "threshold": result.threshold,
            "test_patients": test_patient_count,
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
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_y_score.append(y_score)

        for patient in np.unique(patient_ids[test_indices]):
            mask = patient_ids[test_indices] == patient
            patient_predictions.setdefault(str(patient), []).append(
                (y_true[mask], y_pred[mask], y_score[mask])
            )

        logger.info(
            "RESULT %s acc=%.4f f1=%.4f sensitivity=%.4f precision=%.4f FPR/h=%.3f",
            fold_name,
            test_metrics.accuracy,
            test_metrics.f1,
            test_metrics.recall,
            test_metrics.precision,
            test_metrics.fpr_per_hour,
        )

    pooled_by_patient = {}
    for patient, predictions in patient_predictions.items():
        patient_y_true = np.concatenate([values[0] for values in predictions])
        patient_y_pred = np.concatenate([values[1] for values in predictions])
        patient_y_score = np.concatenate([values[2] for values in predictions])
        pooled_metrics = calculate_binary_metrics_from_predictions(
            patient_y_true,
            patient_y_pred,
            patient_y_score,
            loss=0.0,
            segment_duration=config.data.segment_duration,
        )
        pooled_by_patient[patient] = pooled_metrics.as_dict()

    overall_metrics = calculate_binary_metrics_from_predictions(
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred),
        np.concatenate(all_y_score),
        loss=0.0,
        segment_duration=config.data.segment_duration,
    )
    (run_dir / "pooled_metrics.json").write_text(
        json.dumps(
            {
                "protocol": "mixed_patient_kfold",
                "overall": overall_metrics.as_dict(),
                "patients": pooled_by_patient,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_summary(run_dir / "summary_mixed.json", all_fold_metrics)
    logger.info("Mixed-patient training finished. Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
