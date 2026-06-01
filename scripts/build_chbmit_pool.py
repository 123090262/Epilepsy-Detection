"""Build a subject-balanced CHB-MIT seizure/non-seizure segment pool from EDF files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
)


@dataclass(frozen=True)
class Segment:
    patient: str
    edf_name: str
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    label: int


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
        rates = {
            samples / self.record_duration_s for samples in self.samples_per_record
        }
        if len(rates) != 1:
            raise ValueError(f"channels have inconsistent sample rates: {sorted(rates)}")
        return rates.pop()

    @property
    def n_times(self) -> int:
        return self.record_count * self.samples_per_record[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CHB-MIT segment pool with all seizure windows and sampled negatives."
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "CHBMIT")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "chbmit_pool_1to10_22ch"
    )
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--segment-duration", type=float, default=2.0)
    parser.add_argument("--exclusion-margin", type=float, default=30.0)
    parser.add_argument("--filter-low", type=float, default=0.5)
    parser.add_argument("--filter-high", type=float, default=40.0)
    parser.add_argument("--filter-context", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patients",
        nargs="+",
        help="Optional subset such as --patients chb01 chb02. The default is all patients.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate EDF headers and write manifests without extracting waveforms.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip().upper().replace(" ", "")


def resolve_montage(ch_names: list[str]) -> list[tuple[tuple[int, float], ...]]:
    """Select the first direct occurrence of every required bipolar channel."""

    normalized = [normalize_name(name) for name in ch_names]

    montage: list[tuple[tuple[int, float], ...]] = []
    for target in TARGET_CHANNELS:
        exact = [i for i, name in enumerate(normalized) if name == target]
        suffixed = [
            i
            for i, name in enumerate(normalized)
            if re.fullmatch(re.escape(target) + r"-\d+", name)
        ]
        candidates = exact or suffixed
        if not candidates:
            raise ValueError(f"missing required channel {target}; available={ch_names}")
        montage.append(((candidates[0], 1.0),))
    return montage


def parse_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Parse seizure intervals from one CHB-MIT patient summary file."""

    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    current_file: str | None = None
    pending_start: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        file_match = re.match(r"\s*File Name:\s*(.+?)\s*$", line)
        if file_match:
            current_file = file_match.group(1)
            pending_start = None
            continue
        start_match = re.match(r"\s*Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds", line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = re.match(r"\s*Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds", line)
        if end_match and current_file is not None and pending_start is not None:
            end_s = float(end_match.group(1))
            if end_s <= pending_start:
                raise ValueError(f"invalid seizure interval in {path}: {pending_start}, {end_s}")
            intervals[current_file].append((pending_start, end_s))
            pending_start = None
    return dict(intervals)


def overlaps(start_s: float, end_s: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start_s < interval_end and end_s > interval_start for interval_start, interval_end in intervals)


def fully_inside(start_s: float, end_s: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start_s >= interval_start and end_s <= interval_end for interval_start, interval_end in intervals)


def parse_ascii_fields(data: bytes, width: int, count: int) -> tuple[str, ...]:
    return tuple(
        data[index * width : (index + 1) * width].decode("ascii").strip()
        for index in range(count)
    )


def read_edf_header(path: Path) -> EdfHeader:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError("EDF fixed header is incomplete")
        header_bytes = int(fixed[184:192].decode("ascii").strip())
        record_count = int(fixed[236:244].decode("ascii").strip())
        record_duration_s = float(fixed[244:252].decode("ascii").strip())
        channel_count = int(fixed[252:256].decode("ascii").strip())
        signal_header = handle.read(header_bytes - 256)
    if len(signal_header) != channel_count * 256:
        raise ValueError("EDF signal header is incomplete")

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
        fields[name] = parse_ascii_fields(signal_header[offset : offset + size], width, channel_count)
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


def inspect_edf(path: Path) -> tuple[EdfHeader, float, int]:
    header = read_edf_header(path)
    sfreq = header.sample_rate
    if not np.isclose(sfreq, 256.0):
        raise ValueError(f"expected 256 Hz, got {sfreq}")
    resolve_montage(list(header.channel_names))
    return header, sfreq, header.n_times


def read_edf_range(
    path: Path, header: EdfHeader, picks: list[int], start_sample: int, end_sample: int
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
                chunks[output_channel, record_offset * samples_per_record : (record_offset + 1) * samples_per_record] = (
                    (values - header.digital_min[channel_index]) * physical_range / digital_range
                    + header.physical_min[channel_index]
                )
    crop_start = start_sample - first_record * samples_per_record
    return chunks[:, crop_start : crop_start + end_sample - start_sample]


def enumerate_patient(
    patient_dir: Path,
    segment_duration: float,
    exclusion_margin: float,
) -> tuple[list[Segment], list[Segment], list[dict[str, object]]]:
    summary_paths = sorted(patient_dir.glob("*-summary.txt"))
    if len(summary_paths) != 1:
        raise ValueError(f"expected one summary in {patient_dir}, found {len(summary_paths)}")
    seizure_map = parse_summary(summary_paths[0])

    positives: list[Segment] = []
    negative_candidates: list[Segment] = []
    files: list[dict[str, object]] = []
    for edf_path in sorted(patient_dir.glob("*.edf")):
        seizure_intervals = seizure_map.get(edf_path.name, [])
        try:
            _, sfreq, n_times = inspect_edf(edf_path)
        except ValueError as exc:
            reason = str(exc)
            print(f"[SKIP] {edf_path}: {reason}", flush=True)
            files.append(
                {
                    "patient": patient_dir.name,
                    "edf_name": edf_path.name,
                    "skipped": True,
                    "skip_reason": reason,
                    "seizure_intervals": seizure_intervals,
                }
            )
            continue
        duration_s = n_times / sfreq
        segment_samples = int(round(segment_duration * sfreq))
        excluded_intervals = [
            (max(0.0, start_s - exclusion_margin), min(duration_s, end_s + exclusion_margin))
            for start_s, end_s in seizure_intervals
        ]
        files.append(
            {
                "patient": patient_dir.name,
                "edf_name": edf_path.name,
                "duration_s": duration_s,
                "skipped": False,
                "seizure_intervals": seizure_intervals,
            }
        )
        for start_sample in range(0, n_times - segment_samples + 1, segment_samples):
            end_sample = start_sample + segment_samples
            start_s = start_sample / sfreq
            end_s = end_sample / sfreq
            segment = Segment(
                patient=patient_dir.name,
                edf_name=edf_path.name,
                start_sample=start_sample,
                end_sample=end_sample,
                start_s=start_s,
                end_s=end_s,
                label=0,
            )
            if fully_inside(start_s, end_s, seizure_intervals):
                positives.append(Segment(**{**asdict(segment), "label": 1}))
            elif overlaps(start_s, end_s, seizure_intervals):
                continue
            elif not overlaps(start_s, end_s, excluded_intervals):
                negative_candidates.append(segment)
    return positives, negative_candidates, files


def sample_negatives(
    candidates: list[Segment], positive_count: int, ratio: int, rng: np.random.Generator
) -> list[Segment]:
    wanted = positive_count * ratio
    if wanted > len(candidates):
        raise ValueError(f"need {wanted} negatives but only {len(candidates)} candidates are available")
    indices = rng.choice(len(candidates), size=wanted, replace=False)
    return [candidates[int(index)] for index in sorted(indices)]


def extract_segments(
    input_dir: Path,
    segments: list[Segment],
    low_hz: float,
    high_hz: float,
    context_s: float,
) -> np.ndarray:
    if not segments:
        return np.empty((0, len(TARGET_CHANNELS), 0), dtype=np.float32)
    grouped: dict[tuple[str, str], list[tuple[int, Segment]]] = defaultdict(list)
    for output_index, segment in enumerate(segments):
        grouped[(segment.patient, segment.edf_name)].append((output_index, segment))

    segment_length = segments[0].end_sample - segments[0].start_sample
    output = np.empty((len(segments), len(TARGET_CHANNELS), segment_length), dtype=np.float32)
    for (patient, edf_name), indexed_segments in sorted(grouped.items()):
        edf_path = input_dir / patient / edf_name
        header, sfreq, n_times = inspect_edf(edf_path)
        montage = resolve_montage(list(header.channel_names))
        picks = sorted({index for sources in montage for index, _ in sources})
        pick_positions = {channel_index: position for position, channel_index in enumerate(picks)}
        context_samples = int(round(context_s * sfreq))
        sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
        for output_index, segment in indexed_segments:
            read_start = max(0, segment.start_sample - context_samples)
            read_end = min(n_times, segment.end_sample + context_samples)
            source_data = read_edf_range(edf_path, header, picks, read_start, read_end)
            data = np.stack(
                [
                    sum(coefficient * source_data[pick_positions[index]] for index, coefficient in sources)
                    for sources in montage
                ]
            )
            filtered = sosfiltfilt(sos, data, axis=-1)
            crop_start = segment.start_sample - read_start
            crop_end = crop_start + segment_length
            output[output_index] = filtered[:, crop_start:crop_end].astype(np.float32)
    return output


def write_manifest(path: Path, segments: list[Segment], array_name: str) -> None:
    fieldnames = ["patient", "label", "array_name", "array_index", "edf_name", "start_s", "end_s"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for array_index, segment in enumerate(segments):
            writer.writerow(
                {
                    "patient": segment.patient,
                    "label": segment.label,
                    "array_name": array_name,
                    "array_index": array_index,
                    "edf_name": segment.edf_name,
                    "start_s": f"{segment.start_s:.3f}",
                    "end_s": f"{segment.end_s:.3f}",
                }
            )


def main() -> None:
    args = parse_args()
    if args.negative_ratio < 1:
        raise ValueError("--negative-ratio must be positive")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    patient_dirs = sorted(path for path in args.input_dir.glob("chb*") if path.is_dir())
    if args.patients:
        selected = set(args.patients)
        patient_dirs = [path for path in patient_dirs if path.name in selected]
        missing = sorted(selected - {path.name for path in patient_dirs})
        if missing:
            raise ValueError(f"patients not found: {missing}")
    if not patient_dirs:
        raise ValueError(f"no patient directories found in {args.input_dir}")

    rng = np.random.default_rng(args.seed)
    dataset_manifest: list[dict[str, object]] = []
    file_metadata: list[dict[str, object]] = []
    for patient_dir in patient_dirs:
        print(f"[SCAN] {patient_dir.name}", flush=True)
        positives, negative_candidates, files = enumerate_patient(
            patient_dir, args.segment_duration, args.exclusion_margin
        )
        negatives = sample_negatives(negative_candidates, len(positives), args.negative_ratio, rng)
        patient_output = args.output_dir / patient_dir.name
        patient_output.mkdir()
        write_manifest(patient_output / "seizure_manifest.csv", positives, "seizure.npy")
        write_manifest(patient_output / "non_seizure_manifest.csv", negatives, "non_seizure.npy")
        if not args.metadata_only:
            print(
                f"[EXTRACT] {patient_dir.name}: seizure={len(positives)} non_seizure={len(negatives)}",
                flush=True,
            )
            np.save(
                patient_output / "seizure.npy",
                extract_segments(
                    args.input_dir,
                    positives,
                    args.filter_low,
                    args.filter_high,
                    args.filter_context,
                ),
            )
            np.save(
                patient_output / "non_seizure.npy",
                extract_segments(
                    args.input_dir,
                    negatives,
                    args.filter_low,
                    args.filter_high,
                    args.filter_context,
                ),
            )
        dataset_manifest.append(
            {
                "patient": patient_dir.name,
                "seizure_segments": len(positives),
                "non_seizure_segments": len(negatives),
                "negative_candidates": len(negative_candidates),
            }
        )
        file_metadata.extend(files)

    metadata = {
        "channels": TARGET_CHANNELS,
        "sample_rate_hz": 256,
        "segment_duration_s": args.segment_duration,
        "segment_samples": int(round(256 * args.segment_duration)),
        "filter": {"type": "bandpass", "low_hz": args.filter_low, "high_hz": args.filter_high},
        "filter_context_s": args.filter_context,
        "non_seizure_exclusion_margin_s": args.exclusion_margin,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "unit": "microvolt",
        "metadata_only": args.metadata_only,
        "patients": dataset_manifest,
        "files": file_metadata,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset_manifest[0]))
        writer.writeheader()
        writer.writerows(dataset_manifest)
    print(f"[DONE] Wrote pool to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
