# Epilepsy Detection

End-to-end CHB-MIT seizure detection with the project's TCN, spectral fusion,
and dual graph-attention model. The default protocol follows Li et al., JNE
22 (2025) 056016 where the paper is sufficiently specified.

## Data

Point `data.data_dir` in `configs/default.yaml` at the raw CHB-MIT folder:

```text
CHBMIT/
  chb01/
    chb01-summary.txt
    chb01_01.edf
    ...
  chb24/
    ...
```

No 1:10 seizure/non-seizure waveform pool is built. EDF windows are indexed
deterministically and read, filtered, and standardized during training.

## Protocol

- 18 bipolar channels shared across CHB-MIT cases; referential `*-CS2` files
  are converted to the same bipolar montage.
- Fourth-order 0.5-50 Hz zero-phase bandpass filter.
- One-second windows at 256 Hz.
- Seizure windows overlap by 0.5 seconds.
- Non-seizure windows come from seizure-free EDF files. Their per-case
  durations reproduce table 1 of the JNE paper (approximately 2x-3x each
  case's seizure duration).
- Fold-specific clipping and channel Z-score statistics are fitted on the
  training split only.
- Maximum 50 epochs, early-stopping patience 5, fixed 0.5 decision threshold,
  and checkpoint selection by validation F1.

## Run

Install the project in the remote environment:

```bash
pip install -e .
```

Validate indexing without training:

```bash
python scripts/traincross.py --config configs/default.yaml --patients chb01 --dry-run
python scripts/train_loso.py --config configs/default.yaml --test-patients chb01 --dry-run
```

Run patient-specific ten-fold CV for all configured cases:

```bash
python scripts/traincross.py --config configs/default.yaml --num-folds 10
```

The default random segment split is the paper-comparable setting. A stricter
analysis keeps windows from one seizure event together:

```bash
python scripts/traincross.py --config configs/default.yaml --num-folds 10 --split-level event
```

Run LOPOCV. Each held-out patient is absent from training and validation:

```bash
python scripts/train_loso.py --config configs/default.yaml
```

LOPOCV writes `summary_sampled.json` for comparison with the paper's sampled
classification table and `summary_continuous.json` for the complete held-out
patient timeline. Use `--skip-continuous-eval` only when the full audit is not
required.

Training outputs are written under `runs/`; checkpoints are written under
`checkpoints/`.
