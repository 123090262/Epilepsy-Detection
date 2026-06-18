"""Run 24-patient patient-specific ten-fold CV with a lightweight SVM baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "patient_specific_svm_10fold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional patient subset. Omit to run all patients in the config.",
    )
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--kernel", choices=("rbf", "linear"), default="rbf")
    parser.add_argument("--gamma", default="scale")
    parser.add_argument(
        "--threshold-metric",
        choices=("accuracy", "balanced_accuracy", "f1", "composite"),
        default=None,
        help="Validation metric for selecting the SVM decision threshold.",
    )
    parser.add_argument("--run-name", default="patient_specific_svm_10fold")
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not write per-fold SVM models under checkpoints/.",
    )
    return parser.parse_args()


def select_decision_threshold_from_scores(
    y_true,
    y_score,
    metric_name: str,
    segment_duration: float,
) -> tuple[float, object]:
    import numpy as np

    from epilepsy.train import calculate_binary_metrics

    if len(np.unique(y_true)) != 2:
        raise ValueError("threshold selection requires both validation classes")

    candidates = np.unique(
        np.concatenate(
            (
                y_score,
                np.percentile(y_score, np.linspace(1.0, 99.0, 99)),
                np.asarray([0.0]),
            )
        )
    )
    best_threshold = 0.0
    best_metrics, _ = calculate_binary_metrics(
        y_true,
        y_score,
        loss=0.0,
        threshold=best_threshold,
        segment_duration=segment_duration,
    )
    best_value = getattr(best_metrics, metric_name)
    for threshold in candidates:
        metrics, _ = calculate_binary_metrics(
            y_true,
            y_score,
            loss=0.0,
            threshold=float(threshold),
            segment_duration=segment_duration,
        )
        value = getattr(metrics, metric_name)
        if value > best_value or (
            np.isclose(value, best_value)
            and abs(float(threshold)) < abs(best_threshold)
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value = value
    return best_threshold, best_metrics


def main() -> None:
    args = parse_args()

    import joblib
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from epilepsy.classical import extract_eeg_features
    from epilepsy.config import dump_config_copy, load_config
    from epilepsy.data import build_raw_reader, load_samples, write_sample_manifest
    from epilepsy.evaluate import save_summary
    from epilepsy.train import (
        calculate_binary_metrics,
        calculate_binary_metrics_from_predictions,
    )
    from epilepsy.utils import append_csv_row, make_run_dir, setup_logger

    if args.num_folds < 2:
        raise ValueError("num-folds must be at least 2")

    config = load_config(args.config)
    if args.patients is not None:
        config = replace(
            config,
            data=replace(config.data, test_patients=tuple(args.patients)),
        )
    if config.data.source.lower() != "raw_edf":
        raise ValueError("traincross_svm.py currently requires data.source=raw_edf")

    threshold_metric = args.threshold_metric or config.train.threshold_metric
    if threshold_metric == "fixed":
        threshold_metric = "composite"
    if threshold_metric not in {"accuracy", "balanced_accuracy", "f1", "composite"}:
        raise ValueError(f"Unsupported SVM threshold metric: {threshold_metric}")

    run_dir = make_run_dir(config.output.run_dir, args.run_name)
    checkpoint_root = Path(config.output.checkpoint_dir) / run_dir.name
    if not args.no_save_models:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    dump_config_copy(args.config, run_dir / "config.yaml")
    logger = setup_logger(run_dir / "training.log")
    logger.warning(
        "%s: patient-specific random-window CV. This is not cross-patient "
        "validation, and overlapping seizure windows may occur in different folds.",
        PROTOCOL,
    )
    logger.info(
        "SVM settings: kernel=%s C=%.4g gamma=%s threshold_metric=%s",
        args.kernel,
        args.c,
        args.gamma,
        threshold_metric,
    )

    samples = load_samples(config.data)
    write_sample_manifest(samples, run_dir / "sample_manifest.csv")
    patient_ids = np.asarray([sample.patient for sample in samples])
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    available_patients = set(np.unique(patient_ids))
    patients = tuple(config.data.test_patients)
    missing = sorted(set(patients) - available_patients)
    if missing:
        raise ValueError(f"Patients not found in indexed data: {missing}")

    reader = build_raw_reader(config.data)
    feature_dir = run_dir / "classical_features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    fold_rows: list[dict[str, float | int | str]] = []
    patient_rows: dict[str, list[dict[str, float | int | str]]] = {}
    patient_predictions: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for patient in patients:
        indices = np.flatnonzero(patient_ids == patient)
        patient_labels = labels[indices]
        class_counts = np.bincount(patient_labels, minlength=2)
        if np.min(class_counts) < args.num_folds:
            raise ValueError(
                f"{patient} has too few samples for {args.num_folds} folds: "
                f"non_seizure={class_counts[0]} seizure={class_counts[1]}"
            )

        logger.info(
            "Extracting SVM features for %s: samples=%d seizure=%d non_seizure=%d",
            patient,
            len(indices),
            int(class_counts[1]),
            int(class_counts[0]),
        )
        features = np.stack(
            [
                extract_eeg_features(reader.read(samples[index]), config.data.sample_rate)
                for index in indices
            ]
        )
        np.savez_compressed(
            feature_dir / f"{patient}_features.npz",
            features=features,
            labels=patient_labels,
            sample_indices=indices,
        )

        splitter = StratifiedKFold(
            n_splits=args.num_folds,
            shuffle=True,
            random_state=config.train.random_state,
        )
        for fold, (outer_train, test_indices) in enumerate(
            splitter.split(features, patient_labels), start=1
        ):
            train_indices, val_indices = train_test_split(
                outer_train,
                test_size=config.train.val_size,
                random_state=config.train.random_state + fold,
                stratify=patient_labels[outer_train],
            )
            model = make_pipeline(
                StandardScaler(),
                SVC(
                    C=args.c,
                    gamma=args.gamma,
                    kernel=args.kernel,
                    class_weight="balanced",
                ),
            )
            model.fit(features[train_indices], patient_labels[train_indices])
            val_scores = model.decision_function(features[val_indices])
            threshold, val_metrics = select_decision_threshold_from_scores(
                patient_labels[val_indices],
                val_scores,
                threshold_metric,
                config.data.segment_duration,
            )

            test_scores = model.decision_function(features[test_indices])
            test_true = patient_labels[test_indices]
            test_metrics, test_pred = calculate_binary_metrics(
                test_true,
                test_scores,
                loss=0.0,
                threshold=threshold,
                segment_duration=config.data.segment_duration,
            )
            row = {
                "protocol": PROTOCOL,
                "patient": str(patient),
                "fold": fold,
                "threshold": float(threshold),
                "val_metric": threshold_metric,
                "val_acc": val_metrics.accuracy,
                "val_balanced_acc": val_metrics.balanced_accuracy,
                "val_f1": val_metrics.f1,
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
            fold_rows.append(row)
            patient_rows.setdefault(str(patient), []).append(row)
            append_csv_row(run_dir / "fold_metrics.csv", row)

            all_true.append(test_true)
            all_pred.append(test_pred)
            all_score.append(test_scores)
            patient_predictions.setdefault(str(patient), []).append(
                (test_true, test_pred, test_scores)
            )

            if not args.no_save_models:
                joblib.dump(
                    {
                        "model": model,
                        "decision_threshold": threshold,
                        "protocol": PROTOCOL,
                        "patient": str(patient),
                        "fold": fold,
                        "feature": "temporal_stats_log_bandpower",
                        "sample_rate": config.data.sample_rate,
                        "channels": config.data.num_channels,
                    },
                    checkpoint_root / f"svm_{patient}_fold_{fold:02d}.joblib",
                )

            logger.info(
                "%s fold=%02d acc=%.4f f1=%.4f recall=%.4f specificity=%.4f",
                patient,
                fold,
                test_metrics.accuracy,
                test_metrics.f1,
                test_metrics.recall,
                test_metrics.specificity,
            )

    for patient, rows in patient_rows.items():
        save_summary(run_dir / f"summary_{patient}.json", rows)

    pooled_by_patient = {}
    for patient, predictions in patient_predictions.items():
        patient_true = np.concatenate([values[0] for values in predictions])
        patient_pred = np.concatenate([values[1] for values in predictions])
        patient_score = np.concatenate([values[2] for values in predictions])
        pooled_by_patient[patient] = calculate_binary_metrics_from_predictions(
            patient_true,
            patient_pred,
            patient_score,
            loss=0.0,
            segment_duration=config.data.segment_duration,
        ).as_dict()

    pooled = calculate_binary_metrics_from_predictions(
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_score),
        loss=0.0,
        segment_duration=config.data.segment_duration,
    )
    (run_dir / "pooled_metrics.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "overall": pooled.as_dict(),
                "patients": pooled_by_patient,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_summary(run_dir / "summary.json", fold_rows)
    logger.info(
        "POOLED acc=%.4f f1=%.4f recall=%.4f specificity=%.4f",
        pooled.accuracy,
        pooled.f1,
        pooled.recall,
        pooled.specificity,
    )
    logger.info("Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
