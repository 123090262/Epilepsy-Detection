"""Patient-specific random-window ten-fold CV with spectral-statistical SVM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments" / "patient_specific_10fold_light.yaml",
    )
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--run-name", default="patient_specific_svm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from epilepsy.classical import extract_eeg_features
    from epilepsy.config import dump_config_copy, load_config
    from epilepsy.data import build_raw_reader, load_samples
    from epilepsy.train import (
        calculate_binary_metrics,
        calculate_binary_metrics_from_predictions,
    )
    from epilepsy.utils import append_csv_row, make_run_dir, setup_logger

    config = load_config(args.config)
    if config.data.source.lower() != "raw_edf":
        raise ValueError("traincross_svm.py currently requires data.source=raw_edf")
    run_dir = make_run_dir(config.output.run_dir, args.run_name)
    dump_config_copy(args.config, run_dir / "config.yaml")
    logger = setup_logger(run_dir / "training.log")
    logger.warning(
        "Paper-comparable patient-specific random-window CV; overlapping windows "
        "may occur in different folds and this is not cross-patient validation."
    )

    samples = load_samples(config.data)
    patient_ids = np.asarray([sample.patient for sample in samples])
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    patients = np.unique(patient_ids) if args.patients is None else args.patients
    reader = build_raw_reader(config.data)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    fold_rows = []

    for patient in patients:
        indices = np.flatnonzero(patient_ids == patient)
        if not len(indices):
            raise ValueError(f"Patient not found: {patient}")
        features = np.stack(
            [
                extract_eeg_features(reader.read(samples[index]), config.data.sample_rate)
                for index in indices
            ]
        )
        patient_labels = labels[indices]
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
                SVC(C=args.c, gamma="scale", kernel="rbf"),
            )
            model.fit(features[train_indices], patient_labels[train_indices])
            val_scores = model.decision_function(features[val_indices])
            low, high = np.percentile(val_scores, [1.0, 99.0])
            candidates = np.linspace(low, high, 301)
            threshold = max(
                candidates,
                key=lambda value: np.mean(
                    (val_scores >= value) == patient_labels[val_indices]
                ),
            )
            test_scores = model.decision_function(features[test_indices])
            test_true = patient_labels[test_indices]
            test_metrics, test_pred = calculate_binary_metrics(
                test_true,
                test_scores,
                loss=0.0,
                threshold=float(threshold),
                segment_duration=config.data.segment_duration,
            )
            row = {
                "patient": str(patient),
                "fold": fold,
                "threshold": float(threshold),
                **test_metrics.as_dict(),
            }
            fold_rows.append(row)
            append_csv_row(run_dir / "fold_metrics.csv", row)
            all_true.append(test_true)
            all_pred.append(test_pred)
            all_score.append(test_scores)
            logger.info(
                "%s fold=%02d accuracy=%.4f sensitivity=%.4f specificity=%.4f",
                patient,
                fold,
                test_metrics.accuracy,
                test_metrics.recall,
                test_metrics.specificity,
            )

    pooled = calculate_binary_metrics_from_predictions(
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_score),
        loss=0.0,
        segment_duration=config.data.segment_duration,
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"pooled": pooled.as_dict(), "folds": fold_rows}, indent=2),
        encoding="utf-8",
    )
    logger.info("POOLED accuracy=%.4f", pooled.accuracy)


if __name__ == "__main__":
    main()
