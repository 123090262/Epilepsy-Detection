# JNE Experiment Protocol

## Paper Target

Li et al., *Feature fusion based on global-local weighted attention model for
automatic epileptic seizure detection*, Journal of Neural Engineering 22
(2025) 056016, reports the following CHB-MIT ten-fold results:

| Accuracy | F1 | Sensitivity | Precision | FPR/h |
|---:|---:|---:|---:|---:|
| 98.82% | 98.15% | 98.96% | 98.88% | 0.108 |

Its average reported LOPOCV results are 89.84% accuracy, 86.76% F1, 89.52%
sensitivity, 88.94% precision, and 0.127 FPR/h.

## Implemented Design

The architecture remains this repository's temporal/spectral TCN plus dynamic
PLV/geometric graph-attention model. Data handling and evaluation follow the
paper:

1. Read the raw CHB-MIT EDF files directly.
2. Use 18 common bipolar channels, reconstructing bipolar channels from a
   shared reference when required.
3. Apply a 0.5-50 Hz zero-phase bandpass filter.
4. Create one-second seizure windows with 0.5-second overlap.
5. Draw non-seizure windows only from seizure-free EDF files; selected duration
   reproduces the per-case values in the paper's table 1 (approximately 2x-3x
   seizure duration).
6. Reserve 10% of each development split for validation.
7. Train for at most 50 epochs with patience 5 and select the highest
   validation-F1 checkpoint.
8. Keep the classification threshold fixed at 0.5; the test fold never tunes
   a threshold or preprocessing statistic.

Patient-specific ten-fold CV is run separately for each CHB-MIT case. LOPOCV
trains on all sampled data except the held-out patient. The default LOPO
validation is a stratified sample split to match the paper;
`--validation-level patient` provides a stricter development-patient split.

## Reporting

Every summary reports accuracy, F1, precision, sensitivity/recall, specificity,
balanced accuracy, ROC-AUC, PR-AUC, and FPR/h. FPR/h is computed as false
positive windows divided by the evaluated negative duration in hours.

For LOPOCV, two result sets are intentionally kept separate:

- `summary_sampled.json`: the paper-comparable 2-3x sampled test set.
- `summary_continuous.json`: every non-overlapping one-second window from the
  complete held-out patient recordings.

## Reproducibility Notes

Random segment ten-fold CV can place adjacent 50%-overlapping windows from the
same seizure in training and testing. This matches the paper-facing protocol
but is optimistic. Report `--split-level event` results alongside it when
making a generalization claim.

The paper states that ICA was used but does not specify fitting duration,
component selection, rejection criteria, or whether ICA was fitted before or
after cross-validation. This implementation does not guess those choices. It
uses the specified bandpass filter plus training-only robust standardization,
which avoids whole-dataset preprocessing leakage.
