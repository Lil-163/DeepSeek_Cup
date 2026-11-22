# Q3 calibrated rerun review — 2026-11-22

## Scope and reproducibility

- Source change: `code/q3_scheduler.py`; Q90 calibration was fitted using only the 16 training environments.
- Calibration method: leave-one-training-environment-out Q90 residuals over 738 out-of-fold rows; additive nonnegative margin = 3.858971 seconds.
- Validation data were used for parameter selection and reporting. No external training data were added.
- Frozen parameters and full search output: `outputs/pycharm_q3_calibrated_tune_20261122/`.
- Full validation run: `outputs/pycharm_q3_calibrated_final_20261122/`; aggregate analysis: `outputs/pycharm_q3_calibrated_analysis_20261122/`.

## Verification

- 1,350 trials: 3 cases × 5 mappings × 30 seeds × 3 policies.
- 1,350 unique trial keys; all split labels are validation.
- Invalid actions: 0/1,350; valid protocol: 1,350/1,350.
- Process log: 11,580 action rows covering all 1,350 trial keys.
- Independent replay of seed 2026111700, case S1, adaptive, alpha=1.0, beta=0.5 matched all 11 recorded metrics.

## Main validation result

Weighted on-time rate, mean across seeds and five mappings:

| Case | Adaptive | No review | EDD |
|---|---:|---:|---:|
| S1 | 0.7917 | 0.8271 | 0.3813 |
| S2 | 0.6097 | 0.6236 | 0.2722 |
| S3 | 0.5073 | 0.5250 | 0.2135 |

Adaptive and no-review pass the 0.50 mean gate in all three cases. EDD does not. For the adaptive policy, mean cost was 26.55 / 29.26 / 36.34 for S1 / S2 / S3; mean observation counts were 4 / 6 / 8.

## Calibration caveat

Training OOF Q90 coverage improved from 0.8157 to 0.8997. On held-out validation environments, raw Q90 coverage was 0.8000 and calibrated Q90 coverage was 0.9524 (20/21 rows). The calibration addresses undercoverage but is conservative on this small validation set and increases Q90 MAE from 9.2395 to 12.2143 seconds. Report this trade-off rather than describing Q90 as perfectly calibrated.

These are validation results, not test-set performance. No official Q3 test seeds were available.

## Reproduction

From the repository root, with the official data directory set as `DATA_ROOT`:

```powershell
$PYTHON = "<PyCharm project's Python interpreter>"
$DATA_ROOT = "<official C problem data root>"
& $PYTHON code/q3_scheduler.py --data-root $DATA_ROOT --output outputs/pycharm_q3_calibrated_final_20261122 --params-json outputs/pycharm_q3_calibrated_tune_20261122/q3_tuned_params_flat.json
& $PYTHON code/q3_analyze.py --input outputs/pycharm_q3_calibrated_final_20261122/q3_trial_metrics.csv --output outputs/pycharm_q3_calibrated_analysis_20261122
```
