# Intervention-Validated Video Quality Rewards

Research software for evaluating video-quality metrics against controlled,
source-grouped degradation interventions.

## Evidence status

The controlled-intervention confirmatory study is **complete**. Its generated
report and hashed evidence manifest are in
[`results/confirmatory/`](results/confirmatory/). The former
200-pair preference file and its reported correlations are retained only as a
legacy fixture: their raw video and annotation provenance is insufficient for
confirmatory use. No human-preference, Prolific, correlation, or metric-
superiority claim from that file is endorsed here.

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) and
[configs/study_v1.json](configs/study_v1.json).
See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for execution order and
artifact gates.

## Study design

Clean UCF-101 videos are paired with deterministic descendants produced by six
frozen degradation families. Metrics must rank clean over degraded clips and
recover severity ordering on unseen source videos. The primary robustness test
holds out an entire degradation family while fitting the composite reward.

All partitions are source-grouped and inherit the three official UCF-101
splits. Results report `N_trials=3`, pair counts, and paired hierarchical-
bootstrap 95% confidence intervals. Composite weights are training-side only.
The report retains a severe unseen-temporal-family failure and constrains the
supported claim accordingly; aggregate accuracy is not presented as general
perceptual validity.

<!-- BEGIN GENERATED CONFIRMATORY RESULTS -->
## Confirmatory results

The study scored **16,200 deterministic clean/degraded pairs** across six
families and **`N_trials=3` official UCF-101 splits**, using **2,000 bootstrap
replicates**. These labels identify controlled interventions, not human
preference.

### Primary leave-one-degradation-family-out result

- Composite accuracy: **67.41% [66.13%, 68.70%]** on **5,400 held-out pairs**.
- Composite ROC-AUC: **0.803 [0.792, 0.815]**.
- Severity association: Spearman rho **0.228 [0.197, 0.258]**.

| Held-out family | Accuracy (95% CI) | ROC-AUC (95% CI) | Severity rho (95% CI) | Pairs |
|---|---:|---:|---:|---:|
| `brightness_multiplier` | 100.00% [100.00%, 100.00%] | 1.000 [1.000, 1.000] | 0.302 [0.252, 0.355] | 900 |
| `downsample_short_side` | 100.00% [100.00%, 100.00%] | 1.000 [1.000, 1.000] | 0.372 [0.286, 0.452] | 900 |
| `frame_drop_fraction` | 2.22% [1.11%, 3.56%] | 0.004 [0.001, 0.008] | 0.429 [0.349, 0.505] | 900 |
| `gaussian_blur_sigma` | 100.00% [100.00%, 100.00%] | 1.000 [1.000, 1.000] | 0.792 [0.762, 0.817] | 900 |
| `h264_crf` | 99.22% [98.56%, 99.78%] | 1.000 [1.000, 1.000] | 0.667 [0.625, 0.706] | 900 |
| `temporal_permutation_window` | 3.00% [1.78%, 4.44%] | 0.005 [0.001, 0.011] | 0.457 [0.375, 0.536] | 900 |

The composite generalized almost perfectly to the four spatial/luminance or
codec families but reversed direction on unseen frame dropping and temporal
permutation. This temporal failure is part of the primary result.

### Fixed metric results under family shift

| Metric | Accuracy (95% CI) | ROC-AUC (95% CI) | Severity rho (95% CI) |
|---|---:|---:|---:|
| `block_boundary_energy` | 81.37% [80.30%, 82.41%] | 0.933 [0.927, 0.939] | 0.180 [0.152, 0.209] |
| `brightness_mean` | 65.69% [63.50%, 67.74%] | 0.767 [0.756, 0.784] | 0.020 [-0.007, 0.047] |
| `composite` | 67.41% [66.13%, 68.70%] | 0.803 [0.792, 0.815] | 0.228 [0.197, 0.258] |
| `spatial_detail_log` | 82.31% [81.06%, 83.57%] | 0.937 [0.930, 0.945] | 0.232 [0.205, 0.259] |
| `temporal_delta_mean` | 83.59% [82.59%, 84.61%] | 0.729 [0.712, 0.745] | 0.400 [0.367, 0.430] |
| `temporal_jerk_mean` | 83.19% [82.17%, 84.20%] | 0.715 [0.698, 0.731] | 0.361 [0.326, 0.394] |

### Paired composite-versus-component accuracy differences

| Comparison | Difference (95% CI) | Holm-adjusted p | Pairs |
|---|---:|---:|---:|
| `composite_minus_block_boundary_energy` | -13.963 pp [-14.907, -12.963] | <0.0005 | 5,400 |
| `composite_minus_brightness_mean` | 1.722 pp [-0.444, 4.130] | 0.156 | 5,400 |
| `composite_minus_spatial_detail_log` | -14.907 pp [-16.222, -13.722] | <0.0005 | 5,400 |
| `composite_minus_temporal_delta_mean` | -16.185 pp [-17.223, -15.148] | <0.0005 | 5,400 |
| `composite_minus_temporal_jerk_mean` | -15.778 pp [-16.871, -14.667] | <0.0005 | 5,400 |

### Operational all-family models for the downstream study

| Official split | Accuracy (source-clustered 95% CI) | ROC-AUC (95% CI) | Train/test pairs |
|---:|---:|---:|---:|
| 1 | 97.83% [97.11%, 98.49%] | 0.997 [0.995, 0.998] | 3,600/1,800 |
| 2 | 97.00% [96.13%, 97.84%] | 0.996 [0.994, 0.998] | 3,600/1,800 |
| 3 | 97.94% [97.24%, 98.56%] | 0.998 [0.996, 0.999] | 3,600/1,800 |

These operational scores apply only to source-disjoint pairs from the six
trained intervention families. They do not establish human-preference or
open-ended perceptual validity.
<!-- END GENERATED CONFIRMATORY RESULTS -->

## Metric terminology

Per-video reference-feature distance is named literally. “FVD” is reserved for
Fréchet distance between two video distributions. Confirmatory execution fails
if pretrained weights or reference features are missing; random-backbone
fallbacks are prohibited.

## Layout

- `src/metrics/`: component metric implementations.
- `src/benchmark/`: source-grouped evaluation and statistical analysis.
- `configs/study_v1.json`: frozen interventions and inference settings.
- `data/human_preferences/`: legacy fixture excluded from the new study.
- `results/confirmatory/`: raw pair lineage and scores, predictions, fitted
  checkpoints, generated report, and SHA-256 artifact manifest.

Human external-validity work requires a separate frozen protocol with raw
anonymized annotations, recruitment and exclusion rules, attention checks, and
sample-size justification.
