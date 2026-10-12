"""Deterministic paired, split-clustered uncertainty for confirmatory reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    value_a: str,
    value_b: str,
    cluster: str = "official_split",
    replicates: int = 2000,
    seed: int = 2025,
) -> dict:
    """Bootstrap a paired mean difference, resampling clusters then rows.

    Both values must occur on the same row, making pairing structural. Empty,
    nonfinite, or single-cluster inputs fail rather than producing weak claims.
    """
    required = {value_a, value_b, cluster}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    work = frame[[cluster, value_a, value_b]].copy()
    if work.empty or not np.isfinite(work[[value_a, value_b]].to_numpy()).all():
        raise ValueError("bootstrap input must be nonempty and finite")
    clusters = np.asarray(sorted(work[cluster].unique()))
    if len(clusters) < 2:
        raise ValueError("at least two independent clusters are required")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for draw_index in range(replicates):
        selected_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for selected in selected_clusters:
            rows = work.loc[work[cluster] == selected]
            take = rng.integers(0, len(rows), size=len(rows))
            parts.append(rows.iloc[take])
        sample = pd.concat(parts, ignore_index=True)
        draws[draw_index] = (sample[value_a] - sample[value_b]).mean()
    delta = work[value_a] - work[value_b]
    return {
        "estimate": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_trials": int(len(clusters)),
        "n_observations": int(len(work)),
        "bootstrap_replicates": int(replicates),
        "seed": int(seed),
    }
