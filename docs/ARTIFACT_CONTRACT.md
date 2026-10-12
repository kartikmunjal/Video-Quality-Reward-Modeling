# Artifact contract

Confirmatory metric claims require hashed pair specifications, source hashes,
raw component scores, source-group assignments, raw predictions, fitted-model
hashes, per-split run manifests, and generated uncertainty-aware metrics.
Legacy preference data without its source videos and platform export is never a
confirmatory input.

Every result reports `N_trials`, pair counts by trial and degradation family,
95% confidence intervals, bootstrap configuration, and held-out-family status.
Failed metric loads and nonfinite observations abort the run.
