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
    from epilepsy.data import ChannelPreprocessor, ChbmitPoolDataset, load_pool_samples
    from epilepsy.models import EpilepsyGATNet
    from epilepsy.plots import plot_confusion_matrix
    from epilepsy.train import evaluate

    config = load_config(args.config)
    samples = load_pool_samples(config.data)
    patient_ids = np.asarray([sample.patient for sample in samples])

    if args.patient not in np.unique(patient_ids):
        raise ValueError(f"Patient not found in dataset: {args.patient}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" not in checkpoint or "preprocessor" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain fold preprocessing parameters. "
            "Use a checkpoint created by the updated LOSO training script."
        )
    checkpoint_patient = checkpoint.get("test_patient")
    if checkpoint_patient is not None and checkpoint_patient != args.patient:
        raise ValueError(
            f"Checkpoint was trained for test patient {checkpoint_patient}, "
            f"not {args.patient}"
        )
    preprocessor = ChannelPreprocessor.from_dict(checkpoint["preprocessor"])
    test_indices = np.flatnonzero(patient_ids == args.patient)
    loader = DataLoader(
        ChbmitPoolDataset(samples, test_indices, preprocessor),
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
    model.load_state_dict(checkpoint["model_state_dict"])

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
