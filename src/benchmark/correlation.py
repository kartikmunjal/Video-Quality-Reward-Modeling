"""
Correlation analysis: how well does each metric predict human preference?

Two complementary measures:
  1. Spearman ρ between metric score delta and human preference label.
  2. Pairwise ranking accuracy (did the metric pick the same winner as humans?).

Additionally computes ROC-AUC treating each metric as a binary classifier
(metric picks A → positive class if A was preferred).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


METRICS = ["clip_score", "lpips_temporal", "motion_smoothness", "fvd_score", "composite"]
METRIC_LABELS = {
    "clip_score": "CLIP Score",
    "lpips_temporal": "LPIPS Temporal",
    "motion_smoothness": "Motion Smoothness",
    "fvd_score": "FVD (pairwise)",
    "composite": "Composite",
}


@dataclass
class MetricCorrelation:
    metric: str
    label: str
    spearman_rho: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    pairwise_accuracy: float
    roc_auc: float
    n_pairs: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.spearman_pvalue < alpha

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "label": self.label,
            "spearman_rho": round(self.spearman_rho, 4),
            "spearman_p": round(self.spearman_pvalue, 4),
            "kendall_tau": round(self.kendall_tau, 4),
            "kendall_p": round(self.kendall_pvalue, 4),
            "pairwise_accuracy": round(self.pairwise_accuracy, 4),
            "roc_auc": round(self.roc_auc, 4),
            "n_pairs": self.n_pairs,
            "significant": self.is_significant(),
        }


class CorrelationAnalysis:
    """
    Runs correlation analysis between automated metric scores and human labels.

    Expects a tidy DataFrame from PreferenceDataset.to_dataframe() with columns:
      label         — 1 if A preferred, 0 if B preferred
      delta_clip    — clip_score_a - clip_score_b
      delta_lpips   — lpips_temporal_a - lpips_temporal_b
      delta_motion  — motion_smoothness_a - motion_smoothness_b
      delta_fvd     — fvd_score_a - fvd_score_b
      delta_composite — composite_a - composite_b
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._results: list[MetricCorrelation] | None = None

    def run(self) -> list[MetricCorrelation]:
        """Compute all correlation statistics. Results cached after first call."""
        if self._results is not None:
            return self._results

        results = []
        label = self.df["label"].values.astype(float)

        for metric in METRICS:
            delta_col = f"delta_{metric.replace('_score', '').replace('_temporal', '_lpips' if 'lpips' in metric else '').replace('motion_smoothness', 'motion').replace('fvd_score', 'fvd')}"
            # Build correct column name
            col = self._delta_col(metric)
            if col not in self.df.columns:
                continue

            delta = self.df[col].values.astype(float)
            mask = np.isfinite(delta) & np.isfinite(label)
            d, l = delta[mask], label[mask]
            n = int(mask.sum())

            rho, rho_p = stats.spearmanr(d, l)
            tau, tau_p = stats.kendalltau(d, l)
            accuracy = float(np.mean((d > 0) == (l == 1)))
            auc = roc_auc_score(l.astype(int), d)

            results.append(MetricCorrelation(
                metric=metric,
                label=METRIC_LABELS.get(metric, metric),
                spearman_rho=float(rho),
                spearman_pvalue=float(rho_p),
                kendall_tau=float(tau),
                kendall_pvalue=float(tau_p),
                pairwise_accuracy=accuracy,
                roc_auc=auc,
                n_pairs=n,
            ))

        self._results = sorted(results, key=lambda r: r.spearman_rho, reverse=True)
        return self._results

    def _delta_col(self, metric: str) -> str:
        mapping = {
            "clip_score": "delta_clip",
            "lpips_temporal": "delta_lpips",
            "motion_smoothness": "delta_motion",
            "fvd_score": "delta_fvd",
            "composite": "delta_composite",
        }
        return mapping[metric]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.run()])

    def best_metric(self) -> MetricCorrelation:
        return self.run()[0]

    def print_summary(self):
        df = self.to_dataframe()
        print("\n=== Metric Correlation with Human Preference ===\n")
        print(df[["label", "spearman_rho", "pairwise_accuracy", "roc_auc", "n_pairs"]].to_string(index=False))
        best = self.best_metric()
        print(f"\nBest: {best.label} (ρ={best.spearman_rho:.3f}, accuracy={best.pairwise_accuracy:.1%})")
