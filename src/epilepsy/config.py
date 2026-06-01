"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str
    sample_rate: int
    segment_duration: float
    num_channels: int
    test_patients: tuple[str, ...]

    @property
    def segment_length(self) -> int:
        return int(self.sample_rate * self.segment_duration)


@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    val_size: float
    random_state: int
    num_workers: int
    clip_lower_quantile: float
    clip_upper_quantile: float
    statistics_max_segments: int


@dataclass
class ModelConfig:
    feature_dim: int
    hidden_dim: int
    num_classes: int


@dataclass
class OutputConfig:
    run_dir: str
    checkpoint_dir: str


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


def build_section(section_name: str, section_data: Any, config_type: type) -> Any:
    if not isinstance(section_data, dict):
        raise ValueError(f"Missing or invalid config section: {section_name}")

    expected = {field.name for field in fields(config_type)}
    actual = set(section_data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)

    if missing:
        raise ValueError(
            f"Missing required config field(s) in {section_name}: {', '.join(missing)}"
        )
    if unknown:
        raise ValueError(
            f"Unknown config field(s) in {section_name}: {', '.join(unknown)}"
        )

    return config_type(**section_data)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    expected_sections = {"data", "train", "model", "output"}
    unknown_sections = sorted(set(raw) - expected_sections)
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(unknown_sections)}")

    data_raw = dict(raw.get("data", {}))
    if "test_patients" in data_raw:
        data_raw["test_patients"] = tuple(data_raw["test_patients"])

    return ExperimentConfig(
        data=build_section("data", data_raw, DataConfig),
        train=build_section("train", raw.get("train"), TrainConfig),
        model=build_section("model", raw.get("model"), ModelConfig),
        output=build_section("output", raw.get("output"), OutputConfig),
    )


def dump_config_copy(source_config_path: str | Path, destination: str | Path) -> None:
    source = Path(source_config_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
