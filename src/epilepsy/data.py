"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, TensorDataset

from epilepsy.config import DataConfig
from epilepsy.chbmit import (
    JNE_NON_SEIZURE_MINUTES,
    RawEdfSample,
    RawEdfSegmentReader,
    index_patient_article_samples,
    index_patient_continuous_samples,
)


def prepare_csv_dataset(config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build EEG segments from CHB-MIT style patient CSV folders.

    Expected folder layout:

    ```text
    data_dir/
      chb06/
        0.csv
        1.csv
      chb08/
        0.csv
        seizure_x.csv
    ```

    Label rule kept from the notebook:
    - `0.csv` -> non-seizure class 0
    - every other `.csv` -> seizure class 1

    Returns:
        X: shape `(N, C, L)`, usually `(N, 22, 512)`
        y: shape `(N,)`
        patient_ids: shape `(N,)`
    """

    import pandas as pd

    root_dir = Path(config.data_dir).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {root_dir}. "
            "Update data.data_dir in the YAML config."
        )
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {root_dir}")

    all_segments: list[np.ndarray] = []
    all_labels: list[int] = []
    all_patient_ids: list[str] = []

    for patient_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        patient = patient_dir.name
        for csv_path in sorted(patient_dir.glob("*.csv")):
            label = 0 if csv_path.name == "0.csv" else 1

            try:
                df = pd.read_csv(csv_path, header=None)
                df = df.apply(pd.to_numeric, errors="coerce").dropna()

                if df.shape[1] < config.num_channels:
                    print(
                        f"Skip {csv_path}: expected at least "
                        f"{config.num_channels} columns, got {df.shape[1]}"
                    )
                    continue

                eeg = df.iloc[:, : config.num_channels].values.astype(np.float32)
                total_samples = len(eeg)
                if total_samples < config.segment_length:
                    print(
                        f"Skip {csv_path}: {total_samples} samples "
                        f"< segment length {config.segment_length}"
                    )
                    continue

                num_segments = total_samples // config.segment_length
                for i in range(num_segments):
                    start = i * config.segment_length
                    end = start + config.segment_length
                    segment = eeg[start:end].T
                    all_segments.append(segment)
                    all_labels.append(label)
                    all_patient_ids.append(patient)
            except Exception as exc:
                print(f"Failed to read {csv_path}: {exc}")

    if not all_segments:
        raise RuntimeError(f"No valid EEG segments were generated from {root_dir}")

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    patient_ids = np.array(all_patient_ids)
    return X, y, patient_ids


def make_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


@dataclass(frozen=True)
class PoolSample:
    patient: str
    label: int
    array_path: Path
    array_index: int
    record: str


@dataclass(frozen=True)
class ChannelPreprocessor:
    """Fold-specific clipping and Z-score parameters with shape `(C, 1)`."""

    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, segment: np.ndarray) -> np.ndarray:
        clipped = np.clip(segment, self.lower, self.upper)
        return ((clipped - self.mean) / self.std).astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "lower": torch.from_numpy(self.lower.copy()),
            "upper": torch.from_numpy(self.upper.copy()),
            "mean": torch.from_numpy(self.mean.copy()),
            "std": torch.from_numpy(self.std.copy()),
        }

    @classmethod
    def from_dict(cls, values: dict[str, np.ndarray | torch.Tensor]) -> "ChannelPreprocessor":
        def to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        return cls(
            lower=to_numpy(values["lower"]),
            upper=to_numpy(values["upper"]),
            mean=to_numpy(values["mean"]),
            std=to_numpy(values["std"]),
        )


Sample = Union[PoolSample, RawEdfSample]


class ChbmitPoolDataset(Dataset):
    """Access either legacy NPY pool samples or raw EDF-backed samples."""

    def __init__(
        self,
        samples: Sequence[Sample],
        indices: Sequence[int] | None = None,
        preprocessor: ChannelPreprocessor | None = None,
        raw_reader: RawEdfSegmentReader | None = None,
    ) -> None:
        self.samples = samples
        self.indices = np.asarray(
            np.arange(len(samples)) if indices is None else indices, dtype=np.int64
        )
        self.preprocessor = preprocessor
        self.raw_reader = raw_reader
        self._arrays: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _array(self, path: Path) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r")
        return self._arrays[path]

    def raw_segment(self, dataset_index: int) -> np.ndarray:
        sample = self.samples[int(self.indices[dataset_index])]
        if isinstance(sample, PoolSample):
            return np.asarray(
                self._array(sample.array_path)[sample.array_index], dtype=np.float32
            )
        if self.raw_reader is None:
            raise RuntimeError("Raw EDF samples require a RawEdfSegmentReader")
        return self.raw_reader.read(sample)

    def __getitem__(self, dataset_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[int(self.indices[dataset_index])]
        segment = self.raw_segment(dataset_index)
        if self.preprocessor is not None:
            segment = self.preprocessor.transform(segment)
        return torch.from_numpy(segment), torch.tensor(sample.label, dtype=torch.long)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([self.samples[int(index)].label for index in self.indices], dtype=np.int64)


class EpochBalancedSampler(Sampler[int]):
    """Sample both classes toward a configurable negative-to-positive ratio."""

    def __init__(
        self, labels: Sequence[int], seed: int, negative_ratio: float = 1.0
    ) -> None:
        labels = np.asarray(labels)
        self.positive_indices = np.flatnonzero(labels == 1)
        self.negative_indices = np.flatnonzero(labels == 0)
        if not len(self.positive_indices):
            raise ValueError("training split has no seizure samples")
        if not len(self.negative_indices):
            raise ValueError("training split has no non-seizure samples")
        if negative_ratio <= 0:
            raise ValueError("negative_ratio must be positive")
        self.seed = seed
        desired_negatives = int(round(len(self.positive_indices) * negative_ratio))
        if desired_negatives <= len(self.negative_indices):
            self.positive_count = len(self.positive_indices)
            self.negative_count = max(1, desired_negatives)
        else:
            self.negative_count = len(self.negative_indices)
            self.positive_count = min(
                len(self.positive_indices),
                max(1, int(round(self.negative_count / negative_ratio))),
            )
        self.epoch = 0

    def __len__(self) -> int:
        return self.positive_count + self.negative_count

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        positives = rng.choice(
            self.positive_indices, size=self.positive_count, replace=False
        )
        negatives = rng.choice(
            self.negative_indices, size=self.negative_count, replace=False
        )
        indices = np.concatenate((positives, negatives))
        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices.tolist())


def load_pool_samples(config: DataConfig) -> list[PoolSample]:
    root_dir = Path(config.data_dir).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {root_dir}. "
            "Update data.data_dir in the YAML config."
        )

    samples: list[PoolSample] = []
    for patient_dir in sorted(path for path in root_dir.glob("chb*") if path.is_dir()):
        for label, array_name in ((1, "seizure.npy"), (0, "non_seizure.npy")):
            array_path = patient_dir / array_name
            if not array_path.exists():
                raise FileNotFoundError(f"Missing pool array: {array_path}")
            array = np.load(array_path, mmap_mode="r")
            if array.ndim != 3 or array.shape[1:] != (
                config.num_channels,
                config.segment_length,
            ):
                raise ValueError(
                    f"Unexpected array shape in {array_path}: {array.shape}; expected "
                    f"(N, {config.num_channels}, {config.segment_length})"
                )
            manifest_path = patient_dir / array_name.replace(".npy", "_manifest.csv")
            records = [array_name] * len(array)
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != len(array):
                    raise ValueError(
                        f"Manifest length mismatch for {array_path}: "
                        f"array={len(array)}, manifest={len(rows)}"
                    )
                records = [
                    row.get("edf_name") or records[index]
                    for index, row in enumerate(rows)
                ]

            samples.extend(
                PoolSample(
                    patient_dir.name,
                    label,
                    array_path,
                    index,
                    f"{patient_dir.name}/{records[index]}",
                )
                for index in range(len(array))
            )
    if not samples:
        raise RuntimeError(f"No NPY pool samples found in {root_dir}")
    return samples


def build_raw_reader(config: DataConfig) -> RawEdfSegmentReader:
    return RawEdfSegmentReader(
        sample_rate=config.sample_rate,
        low_hz=config.filter_low_hz,
        high_hz=config.filter_high_hz,
        filter_order=config.filter_order,
        context_seconds=config.filter_context_seconds,
        max_cache_segments=config.raw_cache_segments,
    )


def load_raw_samples(config: DataConfig) -> list[RawEdfSample]:
    root_dir = Path(config.data_dir).expanduser()
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Raw CHB-MIT directory does not exist: {root_dir}")
    samples: list[RawEdfSample] = []
    requested = set(config.test_patients)
    patient_dirs = sorted(path for path in root_dir.glob("chb*") if path.is_dir())
    if requested:
        patient_dirs = [path for path in patient_dirs if path.name in requested]
        missing = sorted(requested - {path.name for path in patient_dirs})
        if missing:
            raise ValueError(f"Configured patients not found under {root_dir}: {missing}")
    for patient_dir in patient_dirs:
        patient_digits = "".join(character for character in patient_dir.name if character.isdigit())
        patient_seed = int(patient_digits) if patient_digits else 0
        target_minutes = JNE_NON_SEIZURE_MINUTES.get(patient_dir.name)
        samples.extend(
            index_patient_article_samples(
                patient_dir=patient_dir,
                sample_rate=config.sample_rate,
                segment_duration=config.segment_duration,
                seizure_overlap=config.seizure_overlap,
                ratio_min=config.non_seizure_ratio_min,
                ratio_max=config.non_seizure_ratio_max,
                random_state=config.sampling_seed + patient_seed,
                target_non_seizure_seconds=(
                    target_minutes * 60.0
                    if config.use_jne_selected_durations and target_minutes is not None
                    else None
                ),
            )
        )
    if not samples:
        raise RuntimeError(f"No raw EDF samples were indexed under {root_dir}")
    return samples


def load_samples(config: DataConfig) -> list[Sample]:
    source = config.source.lower()
    if source == "raw_edf":
        return load_raw_samples(config)
    if source == "pool":
        return load_pool_samples(config)
    raise ValueError(f"Unsupported data source: {config.source}")


def load_continuous_patient_samples(
    config: DataConfig, patient: str
) -> list[RawEdfSample]:
    patient_dir = Path(config.data_dir).expanduser() / patient
    if not patient_dir.is_dir():
        raise NotADirectoryError(f"Patient directory does not exist: {patient_dir}")
    return index_patient_continuous_samples(
        patient_dir,
        sample_rate=config.sample_rate,
        segment_duration=config.segment_duration,
    )


def write_sample_manifest(samples: Sequence[Sample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient",
        "label",
        "record",
        "start_s",
        "end_s",
        "event_id",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "patient": sample.patient,
                    "label": sample.label,
                    "record": sample.record,
                    "start_s": getattr(sample, "start_s", ""),
                    "end_s": getattr(sample, "end_s", ""),
                    "event_id": getattr(sample, "event_id", sample.record),
                    "source": "raw_edf" if isinstance(sample, RawEdfSample) else "pool",
                }
            )


def fit_channel_preprocessor(
    dataset: ChbmitPoolDataset,
    lower_quantile: float,
    upper_quantile: float,
    max_segments: int,
    random_state: int,
) -> ChannelPreprocessor:
    """Estimate clipping and normalization parameters using training samples only."""

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("clip quantiles must satisfy 0 <= lower < upper <= 1")
    if max_segments < 1:
        raise ValueError("statistics_max_segments must be positive")

    rng = np.random.default_rng(random_state)
    sample_count = min(len(dataset), max_segments)
    selected = rng.choice(len(dataset), size=sample_count, replace=False)
    values = np.stack([dataset.raw_segment(int(index)) for index in selected])
    lower = np.quantile(values, lower_quantile, axis=(0, 2)).astype(np.float32)[:, None]
    upper = np.quantile(values, upper_quantile, axis=(0, 2)).astype(np.float32)[:, None]
    clipped = np.clip(values, lower[None, ...], upper[None, ...])
    mean = clipped.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
    std = clipped.std(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
    std = np.maximum(std, np.float32(1e-6))
    return ChannelPreprocessor(lower=lower, upper=upper, mean=mean, std=std)
