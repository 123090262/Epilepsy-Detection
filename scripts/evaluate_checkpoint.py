"""Evaluate a saved epilepsy model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from epilepsy.config import load_config
    from epilepsy.data import make_tensor_dataset, prepare_csv_dataset
    from epilepsy.models import EpilepsyGATNet
    from epilepsy.plots import plot_confusion_matrix
    from epilepsy.train import evaluate

    config = load_config(args.config)
    X, y, patient_ids = prepare_csv_dataset(config.data)

    if args.patient not in np.unique(patient_ids):
        raise ValueError(f"Patient not found in dataset: {args.patient}")

    test_idx = patient_ids == args.patient
    loader = DataLoader(
        make_tensor_dataset(X[test_idx], y[test_idx]),
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EpilepsyGATNet(
        fs=config.data.sample_rate,
        num_classes=config.model.num_classes,
        feature_dim=config.model.feature_dim,
        hid_dim=config.model.hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred, _ = evaluate(model, device, loader, criterion)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true,
        y_pred,
        args.output_dir / f"confusion_matrix_{args.patient}.png",
    )

    print(metrics.as_dict())


if __name__ == "__main__":
    main()
