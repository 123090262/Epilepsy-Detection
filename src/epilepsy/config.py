"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str
    sample_rate: int = 256
    segment_duration: float = 2.0
    num_channels: int = 22
    test_patients: tuple[str, ...] = ("chb06", "chb08", "chb10")

    @property
    def segment_length(self) -> int:
        return int(self.sample_rate * self.segment_duration)


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 1e-6
    val_size: float = 0.1
    random_state: int = 42
    num_workers: int = 0


@dataclass
class ModelConfig:
    feature_dim: int = 128
    hidden_dim: int = 256
    num_classes: int = 2


@dataclass
class OutputConfig:
    run_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class ExperimentConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    output: OutputConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")
    return data


def load_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    data_raw = raw.get("data", {})
    train_raw = raw.get("train", {})
    model_raw = raw.get("model", {})
    output_raw = raw.get("output", {})

    if "data_dir" not in data_raw:
        raise ValueError("Missing required config field: data.data_dir")

    data_raw = dict(data_raw)
    data_raw["test_patients"] = tuple(data_raw.get("test_patients", ()))

    return ExperimentConfig(
        data=DataConfig(**data_raw),
        train=TrainConfig(**train_raw),
        model=ModelConfig(**model_raw),
        output=OutputConfig(**output_raw),
    )


def dump_config_copy(source_config_path: str | Path, destination: str | Path) -> None:
    source = Path(source_config_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
