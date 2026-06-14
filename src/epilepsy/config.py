"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
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
    source: str = "raw_edf"
    seizure_overlap: float = 0.5
    non_seizure_ratio_min: float = 2.0
    non_seizure_ratio_max: float = 3.0
    filter_low_hz: float = 0.5
    filter_high_hz: float = 50.0
    filter_order: int = 4
    filter_context_seconds: float = 4.0
    raw_cache_segments: int = 65536
    sampling_seed: int = 42
    use_jne_selected_durations: bool = True

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
    optimizer: str = "adamw"
    scheduler: str = "onecycle"
    loss: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.65
    negative_ratio: float = 3.0
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    threshold_metric: str = "composite"
    checkpoint_metric: str = "composite"
    min_threshold: float = 0.05
    max_threshold: float = 0.95
    sampling_strategy: str = "all"


@dataclass
class ModelConfig:
    feature_dim: int
    hidden_dim: int
    num_classes: int
    dropout: float = 0.35
    graph_dropout: float = 0.25
    spectral_fusion: bool = True
    auxiliary_weight: float = 0.25


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
    required = {
        field.name
        for field in fields(config_type)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(required - actual)
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
