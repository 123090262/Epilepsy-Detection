"""Raw CHB-MIT EDF indexing and segment reading."""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt


TARGET_CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
)

JNE_NON_SEIZURE_MINUTES = {
    "chb01": 21.88,
    "chb02": 7.60,
    "chb03": 14.28,
    "chb04": 13.98,
    "chb05": 25.90,
    "chb06": 7.27,
    "chb07": 14.95,
    "chb08": 39.07,
    "chb09": 9.97,
    "chb10": 20.02,
    "chb11": 30.90,
    "chb12": 67.43,
    "chb13": 20.73,
    "chb14": 8.03,
    "chb15": 99.47,
    "chb16": 3.85,
    "chb17": 11.57,
    "chb18": 14.73,
    "chb19": 10.73,
    "chb20": 9.90,
    "chb21": 8.02,
    "chb22": 7.15,
    "chb23": 18.93,
    "chb24": 18.55,
}


@dataclass(frozen=True)
class RawEdfSample:
    patient: str
    label: int
    edf_path: Path
    start_sample: int
    end_sample: int
    record: str
    start_s: float
    end_s: float
    event_id: str


@dataclass(frozen=True)
class EdfHeader:
    header_bytes: int
    record_count: int
    record_duration_s: float
    channel_names: tuple[str, ...]
    samples_per_record: tuple[int, ...]
    digital_min: tuple[float, ...]
    digital_max: tuple[float, ...]
    physical_min: tuple[float, ...]
    physical_max: tuple[float, ...]

    @property
    def sample_rate(self) -> float:
        return self.samples_per_record[0] / self.record_duration_s

    @property
    def n_times(self) -> int:
        return self.record_count * self.samples_per_record[0]


def parse_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    current_file: str | None = None
    pending_start: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        file_match = re.match(r"\s*File Name:\s*(.+?)\s*$", line)
        if file_match:
            current_file = file_match.group(1)
            pending_start = None
            continue
        start_match = re.match(
            r"\s*Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds", line
        )
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = re.match(
            r"\s*Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds", line
        )
        if end_match and current_file is not None and pending_start is not None:
            end_s = float(end_match.group(1))
            if end_s <= pending_start:
                raise ValueError(f"Invalid seizure interval in {path}: {pending_start}, {end_s}")
            intervals[current_file].append((pending_start, end_s))
            pending_start = None
    return dict(intervals)


def _parse_ascii_fields(data: bytes, width: int, count: int) -> tuple[str, ...]:
    return tuple(
        data[index * width : (index + 1) * width].decode("ascii").strip()
        for index in range(count)
    )


def read_edf_header(path: Path) -> EdfHeader:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Incomplete EDF header: {path}")
        header_bytes = int(fixed[184:192].decode("ascii").strip())
        record_count = int(fixed[236:244].decode("ascii").strip())
        record_duration_s = float(fixed[244:252].decode("ascii").strip())
        channel_count = int(fixed[252:256].decode("ascii").strip())
        signal_header = handle.read(header_bytes - 256)
    if len(signal_header) != channel_count * 256:
        raise ValueError(f"Incomplete EDF signal header: {path}")

    offset = 0
    fields: dict[str, tuple[str, ...]] = {}
    for name, width in (
        ("label", 16),
        ("transducer", 80),
        ("physical_dimension", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefilter", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    ):
        size = width * channel_count
        fields[name] = _parse_ascii_fields(
            signal_header[offset : offset + size], width, channel_count
        )
        offset += size
    return EdfHeader(
        header_bytes=header_bytes,
        record_count=record_count,
        record_duration_s=record_duration_s,
        channel_names=fields["label"],
        samples_per_record=tuple(int(value) for value in fields["samples_per_record"]),
        digital_min=tuple(float(value) for value in fields["digital_min"]),
        digital_max=tuple(float(value) for value in fields["digital_max"]),
        physical_min=tuple(float(value) for value in fields["physical_min"]),
        physical_max=tuple(float(value) for value in fields["physical_max"]),
    )


def _normalize_channel(name: str) -> str:
    normalized = name.strip().upper().replace(" ", "")
    return {"01": "O1", "02": "O2"}.get(normalized, normalized)


Montage = Tuple[Tuple[Tuple[int, float], ...], ...]


def resolve_montage(channel_names: Iterable[str]) -> Montage:
    normalized = [_normalize_channel(name) for name in channel_names]
    montage: list[tuple[tuple[int, float], ...]] = []
    for target in TARGET_CHANNELS:
        exact = [index for index, name in enumerate(normalized) if name == target]
        suffixed = [
            index
            for index, name in enumerate(normalized)
            if re.fullmatch(re.escape(target) + r"-\d+", name)
        ]
        candidates = exact or suffixed
        if candidates:
            montage.append(((candidates[0], 1.0),))
            continue

        left, right = target.split("-", maxsplit=1)
        inverse = f"{right}-{left}"
        inverse_candidates = [
            index for index, name in enumerate(normalized) if name == inverse
        ]
        if inverse_candidates:
            montage.append(((inverse_candidates[0], -1.0),))
            continue

        left_direct = [
            index for index, name in enumerate(normalized) if name == left
        ]
        right_direct = [
            index for index, name in enumerate(normalized) if name == right
        ]
        if left_direct and right_direct:
            montage.append(((left_direct[0], 1.0), (right_direct[0], -1.0)))
            continue

        left_reference = [
            index
            for index, name in enumerate(normalized)
            if name.startswith(f"{left}-")
        ]
        right_reference = [
            index
            for index, name in enumerate(normalized)
            if name.startswith(f"{right}-")
        ]
        reconstructed = None
        for left_index in left_reference:
            left_suffix = normalized[left_index].split("-", maxsplit=1)[1]
            for right_index in right_reference:
                right_suffix = normalized[right_index].split("-", maxsplit=1)[1]
                if left_suffix == right_suffix:
                    reconstructed = ((left_index, 1.0), (right_index, -1.0))
                    break
            if reconstructed is not None:
                break
        if reconstructed is None:
            raise ValueError(f"Missing required channel {target}")
        montage.append(reconstructed)
    return tuple(montage)


def _overlaps(
    start_s: float, end_s: float, intervals: list[tuple[float, float]]
) -> bool:
    return any(start_s < seizure_end and end_s > seizure_start for seizure_start, seizure_end in intervals)


def _patient_files(patient_dir: Path):
    summaries = sorted(patient_dir.glob("*-summary.txt"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one summary file in {patient_dir}, found {len(summaries)}")
    seizure_map = parse_summary(summaries[0])
    for edf_path in sorted(patient_dir.glob("*.edf")):
        try:
            header = read_edf_header(edf_path)
            resolve_montage(header.channel_names)
        except (OSError, ValueError) as exc:
            print(f"[SKIP] {edf_path}: {exc}", flush=True)
            continue
        yield edf_path, header, seizure_map.get(edf_path.name, [])


def _sample(
    patient: str,
    edf_path: Path,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    label: int,
    event_id: str,
) -> RawEdfSample:
    return RawEdfSample(
        patient=patient,
        label=label,
        edf_path=edf_path,
        start_sample=start_sample,
        end_sample=end_sample,
        record=f"{patient}/{edf_path.name}",
        start_s=start_sample / sample_rate,
        end_s=end_sample / sample_rate,
        event_id=event_id,
    )


def index_patient_article_samples(
    patient_dir: Path,
    sample_rate: int,
    segment_duration: float,
    seizure_overlap: float,
    ratio_min: float,
    ratio_max: float,
    random_state: int,
    target_non_seizure_seconds: float | None = None,
) -> list[RawEdfSample]:
    """Index the JNE sampled protocol without creating waveform pools.

    Seizure intervals use one-second windows with 0.5-second overlap. Negative
    candidates are non-overlapping windows from seizure-free EDF files, and a
    deterministic 2--3x seizure-duration subset is selected per patient.
    """

    if not 0 <= seizure_overlap < segment_duration:
        raise ValueError("seizure_overlap must satisfy 0 <= overlap < segment_duration")
    if not 0 < ratio_min <= ratio_max:
        raise ValueError("non-seizure ratios must satisfy 0 < min <= max")

    segment_samples = int(round(segment_duration * sample_rate))
    seizure_step = int(round((segment_duration - seizure_overlap) * sample_rate))
    positives: list[RawEdfSample] = []
    negative_candidates: list[RawEdfSample] = []
    seizure_duration_s = 0.0

    for edf_path, header, intervals in _patient_files(patient_dir):
        if not np.isclose(header.sample_rate, sample_rate):
            raise ValueError(f"Expected {sample_rate} Hz in {edf_path}, got {header.sample_rate}")
        if intervals:
            for event_index, (start_s, end_s) in enumerate(intervals, start=1):
                seizure_duration_s += end_s - start_s
                first = int(round(start_s * sample_rate))
                stop = int(round(end_s * sample_rate))
                event_id = f"{patient_dir.name}/{edf_path.name}#seizure-{event_index}"
                for start_sample in range(first, stop - segment_samples + 1, seizure_step):
                    positives.append(
                        _sample(
                            patient_dir.name,
                            edf_path,
                            start_sample,
                            start_sample + segment_samples,
                            sample_rate,
                            1,
                            event_id,
                        )
                    )
            continue

        for start_sample in range(0, header.n_times - segment_samples + 1, segment_samples):
            negative_candidates.append(
                _sample(
                    patient_dir.name,
                    edf_path,
                    start_sample,
                    start_sample + segment_samples,
                    sample_rate,
                    0,
                    f"{patient_dir.name}/{edf_path.name}#non-seizure",
                )
            )

    if not positives:
        raise ValueError(f"No seizure windows found for {patient_dir.name}")
    rng = np.random.default_rng(random_state)
    ratio = float(rng.uniform(ratio_min, ratio_max))
    selected_duration_s = (
        seizure_duration_s * ratio
        if target_non_seizure_seconds is None
        else target_non_seizure_seconds
    )
    wanted = min(
        len(negative_candidates),
        int(round(selected_duration_s / segment_duration)),
    )
    selected = rng.choice(len(negative_candidates), size=wanted, replace=False)
    negatives = [negative_candidates[int(index)] for index in np.sort(selected)]
    return positives + negatives


def index_patient_continuous_samples(
    patient_dir: Path,
    sample_rate: int,
    segment_duration: float,
) -> list[RawEdfSample]:
    """Index every non-overlapping window for complete held-out-patient testing."""

    segment_samples = int(round(segment_duration * sample_rate))
    samples: list[RawEdfSample] = []
    for edf_path, header, intervals in _patient_files(patient_dir):
        if not np.isclose(header.sample_rate, sample_rate):
            raise ValueError(f"Expected {sample_rate} Hz in {edf_path}, got {header.sample_rate}")
        for start_sample in range(0, header.n_times - segment_samples + 1, segment_samples):
            end_sample = start_sample + segment_samples
            start_s = start_sample / sample_rate
            end_s = end_sample / sample_rate
            label = int(_overlaps(start_s, end_s, intervals))
            event_id = f"{patient_dir.name}/{edf_path.name}#continuous"
            samples.append(
                _sample(
                    patient_dir.name,
                    edf_path,
                    start_sample,
                    end_sample,
                    sample_rate,
                    label,
                    event_id,
                )
            )
    return samples


class RawEdfSegmentReader:
    """Read, bandpass-filter, and cache EDF windows on first use."""

    def __init__(
        self,
        sample_rate: int,
        low_hz: float,
        high_hz: float,
        filter_order: int,
        context_seconds: float,
        max_cache_segments: int,
    ) -> None:
        if not 0 < low_hz < high_hz < sample_rate / 2:
            raise ValueError("Bandpass frequencies must lie inside (0, Nyquist)")
        self.sample_rate = sample_rate
        self.context_samples = int(round(context_seconds * sample_rate))
        self.max_cache_segments = max(0, max_cache_segments)
        self.sos = butter(
            filter_order,
            [low_hz, high_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        self.headers: dict[Path, EdfHeader] = {}
        self.montages: dict[Path, Montage] = {}
        self.cache: OrderedDict[tuple[Path, int, int], np.ndarray] = OrderedDict()

    def read(self, sample: RawEdfSample) -> np.ndarray:
        key = (sample.edf_path, sample.start_sample, sample.end_sample)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached

        header = self.headers.get(sample.edf_path)
        if header is None:
            header = read_edf_header(sample.edf_path)
            self.headers[sample.edf_path] = header
            self.montages[sample.edf_path] = resolve_montage(header.channel_names)
        montage = self.montages[sample.edf_path]
        picks = tuple(sorted({index for sources in montage for index, _ in sources}))
        pick_positions = {channel_index: position for position, channel_index in enumerate(picks)}
        read_start = max(0, sample.start_sample - self.context_samples)
        read_end = min(header.n_times, sample.end_sample + self.context_samples)
        data = read_edf_range(
            sample.edf_path,
            header,
            picks,
            read_start,
            read_end,
        )
        bipolar = np.stack(
            [
                sum(
                    coefficient * data[pick_positions[index]]
                    for index, coefficient in sources
                )
                for sources in montage
            ]
        )
        filtered = sosfiltfilt(self.sos, bipolar, axis=-1)
        crop_start = sample.start_sample - read_start
        crop_length = sample.end_sample - sample.start_sample
        segment = filtered[:, crop_start : crop_start + crop_length].astype(np.float32)

        if self.max_cache_segments:
            self.cache[key] = segment
            self.cache.move_to_end(key)
            while len(self.cache) > self.max_cache_segments:
                self.cache.popitem(last=False)
        return segment


def read_edf_range(
    path: Path,
    header: EdfHeader,
    picks: tuple[int, ...],
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    samples_per_record = header.samples_per_record[0]
    first_record = start_sample // samples_per_record
    last_record = (end_sample - 1) // samples_per_record
    record_bytes = sum(header.samples_per_record) * 2
    channel_offsets = np.cumsum((0, *header.samples_per_record[:-1])) * 2
    chunks = np.empty(
        (len(picks), (last_record - first_record + 1) * samples_per_record),
        dtype=np.float64,
    )
    with path.open("rb") as handle:
        for record_offset, record_index in enumerate(range(first_record, last_record + 1)):
            record_start = header.header_bytes + record_index * record_bytes
            for output_channel, channel_index in enumerate(picks):
                handle.seek(record_start + int(channel_offsets[channel_index]))
                raw_bytes = handle.read(header.samples_per_record[channel_index] * 2)
                values = np.frombuffer(raw_bytes, dtype="<i2")
                digital_range = header.digital_max[channel_index] - header.digital_min[channel_index]
                physical_range = header.physical_max[channel_index] - header.physical_min[channel_index]
                chunks[
                    output_channel,
                    record_offset * samples_per_record : (record_offset + 1) * samples_per_record,
                ] = (
                    (values - header.digital_min[channel_index])
                    * physical_range
                    / digital_range
                    + header.physical_min[channel_index]
                )
    crop_start = start_sample - first_record * samples_per_record
    return chunks[:, crop_start : crop_start + end_sample - start_sample]
