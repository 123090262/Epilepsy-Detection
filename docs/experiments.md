# Experiment Protocol

## Why the previous F1 was low

- The data pool is 10:1 non-seizure to seizure. Training every epoch at 1:1
  changes the class prior and can produce too many positive predictions.
- LOSO selected checkpoints by validation accuracy at a fixed 0.5 threshold.
  Accuracy is dominated by the negative class and is a poor early-stopping
  objective for seizure detection.
- Validation segments were randomly split, so segments from the same patient
  and EDF recording could appear in training and validation.
- The graph prior averaged PLV over the batch, making a sample's prediction
  depend on unrelated samples in the same batch.

## Updated method

- Temporal TCN features are fused with differentiable delta/theta/alpha/beta/
  gamma log-bandpower features before dual graph attention.
- PLV is computed from the analytic signal and a separate dynamic graph is
  retained for every sample.
- Residual graph-attention layers and the spatial auxiliary classifier improve
  gradient flow and branch supervision.
- Focal loss, a 3:1 dynamic negative sampler, AdamW, OneCycle scheduling,
  gradient clipping, and early stopping target imbalance and optimization.
- The validation threshold maximizes a composite of F1 (60%), balanced
  accuracy (25%), and accuracy (15%). The test threshold is never tuned on the
  test set.
- Ten-fold CV is grouped by EDF recording by default. LOSO uses unseen patients
  for both outer testing and inner validation.

## Literature basis

- Li et al., *Journal of Neural Engineering* 22 (2025) 056016: temporal,
  spatial, and spectral fusion; validation-F1 model selection; grid search; and
  patient-independent evaluation.
- Lin et al., ICCV 2017: focal loss for severe class imbalance.
- Cui et al., CVPR 2019: class-aware treatment of long-tailed data.
- Song et al., IEEE TNSRE 2023: convolution plus attention for robust EEG
  representations (EEG Conformer).

## Reporting

Always report accuracy, F1, precision, recall/sensitivity, specificity,
balanced accuracy, ROC-AUC, and PR-AUC. Do not compare record-grouped CV and
LOSO numbers as if they measured the same generalization setting.
