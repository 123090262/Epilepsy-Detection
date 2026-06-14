# Epilepsy

EEG seizure detection on CHB-MIT with stratified and grouped validation modes.

This repository is being organized from the original notebook into a standard Python project layout. The current goal is to make data loading, model definition, training, evaluation, and experiment tracking easier to maintain.

## Project Structure

```text
configs/              YAML configuration files for training and experiments
src/epilepsy/         Python package source code
src/epilepsy/models/  Model components
scripts/              Command line entry scripts
notebooks/            Exploratory notebooks
data/                 Local data placeholder, ignored by Git
runs/                 Training outputs, ignored by Git
checkpoints/          Model checkpoints, ignored by Git
docs/                 Experiment notes and project documents
```

## Next Steps

1. Move reusable code from `notebooks/留一2.ipynb` into `src/epilepsy/`.
2. Read training parameters from `configs/default.yaml`.
3. Run training through `scripts/train_loso.py`.
4. Save experiment outputs under `runs/` and checkpoints under `checkpoints/`.

## Usage

Install dependencies:

```bash
pip install -e .
```

Check that the config and dataset can be loaded:

```bash
python scripts/train_loso.py --config configs/default.yaml --dry-run
```

Start strict LOSO training (outer patient and validation patients are disjoint):

```bash
python scripts/train_loso.py --config configs/default.yaml
```

Run stratified random-segment 10-fold CV (the default):

```bash
python scripts/traincross.py --config configs/default.yaml --num-folds 10
```

Run the stricter EDF-record-grouped or patient-grouped variants:

```bash
python scripts/traincross.py --config configs/default.yaml --num-folds 10 --group-level record
python scripts/traincross.py --config configs/default.yaml --num-folds 10 --group-level patient
```

Tune one LOSO outer fold without exposing its test patient to the search:

```bash
python scripts/search_hyperparams.py \
  --config configs/default.yaml \
  --test-patient chb06 \
  --trials 12 \
  --inner-folds 3 \
  --epochs 20
```

The search writes a ranked table and `best_config.yaml`. Use that file with
`train_loso.py`; it contains only the requested outer test patient.

Evaluate a checkpoint:

```bash
python scripts/evaluate_checkpoint.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/<run-name>/best_model_test_chb06.pth \
  --patient chb06
```

Large files such as raw data, training outputs, and model checkpoints are intentionally ignored by Git.
