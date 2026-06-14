"""Nested, patient-grouped random search without touching the outer test patient."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested random search for LOSO.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--test-patient", required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log_uniform(rng, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_config(base_config, rng, trial: int):
    if trial == 0:
        return replace(
            base_config,
            data=replace(base_config.data, test_patients=(base_config.data.test_patients[0],)),
        )
    train = replace(
        base_config.train,
        learning_rate=log_uniform(rng, 2e-4, 5e-3),
        weight_decay=log_uniform(rng, 1e-6, 1e-3),
        batch_size=int(rng.choice([16, 32, 64])),
        focal_gamma=float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])),
        focal_alpha=float(rng.choice([0.55, 0.60, 0.65, 0.70, 0.75])),
        negative_ratio=float(rng.choice([2.0, 3.0, 4.0, 5.0])),
    )
    model = replace(
        base_config.model,
        feature_dim=int(rng.choice([64, 96, 128])),
        hidden_dim=int(rng.choice([64, 128, 192, 256])),
        dropout=float(rng.choice([0.30, 0.40, 0.50])),
        graph_dropout=float(rng.choice([0.15, 0.25, 0.35])),
        auxiliary_weight=float(rng.choice([0.15, 0.25, 0.35])),
    )
    return replace(base_config, train=train, model=model)


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    import yaml

    from epilepsy.config import load_config
    from epilepsy.data import load_pool_samples
    from epilepsy.experiment import fit_fold
    from epilepsy.splits import grouped_kfold_splits
    from epilepsy.utils import append_csv_row, setup_logger

    if args.trials < 1 or args.inner_folds < 2 or args.epochs < 1:
        raise ValueError("trials >= 1, inner-folds >= 2, and epochs >= 1 are required")
    base_config = load_config(args.config)
    samples = load_pool_samples(base_config.data)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    patients = np.asarray([sample.patient for sample in samples])
    if args.test_patient not in np.unique(patients):
        raise ValueError(f"Unknown outer test patient: {args.test_patient}")

    development_indices = np.flatnonzero(patients != args.test_patient)
    inner_splits = list(
        grouped_kfold_splits(
            development_indices,
            labels[development_indices],
            patients[development_indices],
            args.inner_folds,
            args.seed,
        )
    )
    output_dir = Path(base_config.output.run_dir) / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_search_outer_{args.test_patient}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(output_dir / "search.log")
    logger.info(
        "Outer test patient %s is excluded from all %d search trials",
        args.test_patient,
        args.trials,
    )
    if args.dry_run:
        for fold, (train_indices, val_indices) in enumerate(inner_splits, start=1):
            logger.info(
                "Inner fold %d train_patients=%d val_patients=%s",
                fold,
                len(np.unique(patients[train_indices])),
                ",".join(np.unique(patients[val_indices]).tolist()),
            )
        logger.info("Dry run finished. No training was started.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    ranked = []
    for trial in range(args.trials):
        candidate = sample_config(base_config, rng, trial)
        candidate = replace(
            candidate,
            data=replace(candidate.data, test_patients=(args.test_patient,)),
        )
        fold_metrics = []
        logger.info(
            "Trial %d/%d lr=%.6g wd=%.3g batch=%d neg_ratio=%.1f gamma=%.1f "
            "alpha=%.2f feat=%d hidden=%d dropout=%.2f graph_dropout=%.2f",
            trial + 1,
            args.trials,
            candidate.train.learning_rate,
            candidate.train.weight_decay,
            candidate.train.batch_size,
            candidate.train.negative_ratio,
            candidate.train.focal_gamma,
            candidate.train.focal_alpha,
            candidate.model.feature_dim,
            candidate.model.hidden_dim,
            candidate.model.dropout,
            candidate.model.graph_dropout,
        )
        for fold, (train_indices, val_indices) in enumerate(inner_splits, start=1):
            result = fit_fold(
                samples,
                train_indices,
                val_indices,
                candidate,
                device,
                seed=args.seed + trial * 100 + fold,
                max_epochs=args.epochs,
            )
            fold_metrics.append(result.best_metrics.as_dict())
            logger.info(
                "Trial %d fold %d best_epoch=%d val_f1=%.4f val_acc=%.4f composite=%.4f",
                trial + 1,
                fold,
                result.best_epoch,
                result.best_metrics.f1,
                result.best_metrics.accuracy,
                result.best_metrics.composite,
            )
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregate = {
            key: float(np.nanmean([metrics[key] for metrics in fold_metrics]))
            for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "composite")
        }
        row = {
            "trial": trial + 1,
            **aggregate,
            "learning_rate": candidate.train.learning_rate,
            "weight_decay": candidate.train.weight_decay,
            "batch_size": candidate.train.batch_size,
            "negative_ratio": candidate.train.negative_ratio,
            "focal_gamma": candidate.train.focal_gamma,
            "focal_alpha": candidate.train.focal_alpha,
            "feature_dim": candidate.model.feature_dim,
            "hidden_dim": candidate.model.hidden_dim,
            "dropout": candidate.model.dropout,
            "graph_dropout": candidate.model.graph_dropout,
            "auxiliary_weight": candidate.model.auxiliary_weight,
        }
        append_csv_row(output_dir / "search_results.csv", row)
        ranked.append((aggregate["composite"], candidate, row, fold_metrics))

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, best_config, best_row, _ = ranked[0]
    serializable = json.loads(json.dumps(asdict(best_config)))
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "ranking.json").write_text(
        json.dumps([item[2] for item in ranked], indent=2), encoding="utf-8"
    )
    logger.info(
        "Best trial=%d composite=%.4f f1=%.4f accuracy=%.4f; config=%s",
        best_row["trial"],
        best_row["composite"],
        best_row["f1"],
        best_row["accuracy"],
        output_dir / "best_config.yaml",
    )


if __name__ == "__main__":
    main()
