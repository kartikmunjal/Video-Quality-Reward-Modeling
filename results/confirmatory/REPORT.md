# Confirmatory report: controlled-intervention video rewards

This report is generated from raw scored pairs by `scripts/publish_confirmatory_report.py`; displayed values are not hand edited.

## Design and evidence boundary

The study scored 16,200 deterministic clean/degraded pairs across six degradation families and the three official UCF-101 splits. All final test sources are disjoint from fitting sources. Ground truth is intervention identity, not human aesthetic preference.

The primary robustness test fits on five degradation families and evaluates the sixth. Reported intervals use 2,000 deterministic hierarchical-bootstrap replicates over `N_trials=3` official splits and pairs.

## Primary result: unseen degradation-family transfer

The five-feature composite achieved **67.41% [66.13%, 68.70%]** accuracy and ROC-AUC **0.803 [0.792, 0.815]** on 5,400 held-out pairs (`N_trials=3`). Severity association was Spearman **0.228 [0.197, 0.258]**.

This aggregate is not evidence of uniform robustness. The family-level results show a sharp mechanical failure on unseen temporal corruptions:

| Held-out family | Accuracy (95% CI) | Severity rho (95% CI) | Pairs |
|---|---:|---:|---:|
| `brightness_multiplier` | 100.00% [100.00%, 100.00%] | 0.302 [0.252, 0.355] | 900 |
| `downsample_short_side` | 100.00% [100.00%, 100.00%] | 0.372 [0.286, 0.452] | 900 |
| `frame_drop_fraction` | 2.22% [1.11%, 3.56%] | 0.429 [0.349, 0.505] | 900 |
| `gaussian_blur_sigma` | 100.00% [100.00%, 100.00%] | 0.792 [0.762, 0.817] | 900 |
| `h264_crf` | 99.22% [98.56%, 99.78%] | 0.667 [0.625, 0.706] | 900 |
| `temporal_permutation_window` | 3.00% [1.78%, 4.44%] | 0.457 [0.375, 0.536] | 900 |

Brightness, downsampling, blur, and H.264 artifacts change direct spatial or luminance diagnostics in a stable direction. Frame dropping and local temporal permutation can instead reduce measured frame-to-frame change, so a model trained without that family assigns the wrong sign. This explains the near-zero temporal holdout accuracy and is a retained negative result, not an excluded slice.

## Fixed-metric comparison

| Metric | Accuracy (95% CI) | ROC-AUC (95% CI) | Severity rho (95% CI) |
|---|---:|---:|---:|
| `block_boundary_energy` | 81.37% [80.30%, 82.41%] | 0.933 [0.927, 0.939] | 0.180 [0.152, 0.209] |
| `brightness_mean` | 65.69% [63.50%, 67.74%] | 0.767 [0.756, 0.784] | 0.020 [-0.007, 0.047] |
| `composite` | 67.41% [66.13%, 68.70%] | 0.803 [0.792, 0.815] | 0.228 [0.197, 0.258] |
| `spatial_detail_log` | 82.31% [81.06%, 83.57%] | 0.937 [0.930, 0.945] | 0.232 [0.205, 0.259] |
| `temporal_delta_mean` | 83.59% [82.59%, 84.61%] | 0.729 [0.712, 0.745] | 0.400 [0.367, 0.430] |
| `temporal_jerk_mean` | 83.19% [82.17%, 84.20%] | 0.715 [0.698, 0.731] | 0.361 [0.326, 0.394] |

Paired composite-versus-component comparisons and Holm-adjusted p-values are stored in `analysis/metrics.json`. They show that the learned composite does not dominate simple components under family shift.

## Operational in-family models

These split-specific models fit all six intervention families on official training sources. They are intended only for the downstream augmentation study, whose candidates are covered by the training interventions; they are not evidence of open-ended perceptual generalization.

| Split | Accuracy (source-clustered 95% CI) | ROC-AUC (95% CI) | Train/test pairs |
|---:|---:|---:|---:|
| 1 | 97.83% [97.11%, 98.49%] | 0.997 [0.995, 0.998] | 3,600/1,800 |
| 2 | 97.00% [96.13%, 97.84%] | 0.996 [0.994, 0.998] | 3,600/1,800 |
| 3 | 97.94% [97.24%, 98.56%] | 0.998 [0.996, 0.999] | 3,600/1,800 |

## Deviations and limitations

- The frozen plan mentioned a calibration subset, calibration error, nonzero-severity pair ranking, and class/resolution/motion slices. The frozen pair builder did not allocate those samples or fields. They are therefore **not reported**; no post-outcome amendment or surrogate analysis was used.
- This benchmark tests deterministic synthetic corruptions of UCF-101, not human preference, generative-video quality, or cross-dataset validity.
- Perfect results on several held-out families reflect easily detected controlled interventions and must not be described as general video-quality accuracy.

## Reproduction

The artifact manifest beside this report records SHA-256 hashes and byte counts for the raw scores, analyses, preregistration, and model checkpoints.
