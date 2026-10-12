# Intervention-validated video quality reward study — frozen protocol

**Study ID:** `ucf101_quality_reward_v1`  
**Status:** preregistered; no confirmatory results have been run  
**Parent protocol:** `Video-Curation/RESEARCH_PLAN.md`

## Research question

Which automated video-quality signals reliably rank a clean UCF-101 clip above
a controlled degraded descendant, including degradation families not seen while
fitting a composite reward?

This study does not claim to measure general human aesthetic preference. Its
ground truth is an explicit randomized intervention with known severity.

## Pair construction

- Start only from training/test assignments in the official UCF-101 splits.
- Create descendants with deterministic seeds and retain the source-video ID.
- Confirmatory degradations: Gaussian blur, H.264 compression, frame dropping,
  temporal permutation, brightness loss, and spatial downsample/upsample.
- Severity levels are frozen in `configs/study_v1.json` before pilot scoring.
- Each pair compares a source clip with one degraded descendant. A secondary
  dose-response task ranks two nonzero severities.
- Train, calibration, and test partitions are source-grouped. No descendant of
  a source may cross partitions.

## Validation design

- Fit normalization and composite weights on split-training sources only.
- Tune on a source-disjoint calibration subset of training data.
- Report final results on the untouched official test side.
- Evaluate both random pair holdout and leave-one-degradation-family-out
  generalization. The latter is the primary robustness result.
- Composite weights must be learned or frozen; they may never be selected using
  confirmatory test labels.

## Outcomes

Primary:

- paired ranking accuracy on unseen source videos;
- leave-one-degradation-family-out ranking accuracy;
- severity-delta Spearman correlation.

Secondary:

- ROC-AUC, Kendall tau, calibration error, and accuracy by degradation family;
- failure slices by action class, source resolution, and motion quartile.

All outcomes include `N_trials=3` official splits, per-trial pair counts, and
paired hierarchical-bootstrap 95% CIs. Differences between metrics use the same
resampled pairs. Holm correction applies to the confirmatory metric family.

## Metric naming and failure policy

- CLIP alignment, temporal LPIPS, optical-flow stability, and reference-video
  feature distance are named literally.
- `FVD` is reserved for Fréchet distance between two sufficiently large video
  distributions. A per-video score may not be called FVD.
- Missing pretrained weights, unavailable reference features, decode failures,
  and nonfinite scores are hard failures in confirmatory mode. No randomly
  initialized proxy is permitted.

## Optional human external-validity study

Human preference is outside the confirmatory scope until a separately frozen
protocol specifies recruitment, consent, rater exclusions, attention checks,
raw anonymized exports, agreement statistics, and a sample-size justification.
No Prolific or human-agreement claim may appear before those artifacts exist.

## Claim policy

The existing 200-pair JSON is treated as legacy illustrative data and excluded
from confirmatory analysis. README claims must come only from generated reports
whose manifests identify raw pair and metric hashes.
