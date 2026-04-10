"""
BenchmarkReport: generate tables and figures from correlation analysis results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .correlation import CorrelationAnalysis, MetricCorrelation

# Consistent color palette across all figures
PALETTE = {
    "CLIP Score": "#4C72B0",
    "LPIPS Temporal": "#DD8452",
    "Motion Smoothness": "#55A868",
    "FVD (pairwise)": "#C44E52",
    "Composite": "#8172B3",
}


class BenchmarkReport:
    """
    Given a CorrelationAnalysis, produce:
      - CSV correlation table
      - Bar chart: Spearman ρ per metric
      - Bar chart: pairwise accuracy per metric
      - Scatter plots: metric delta vs. human preference probability
      - Combined figure for paper
    """

    def __init__(self, analysis: CorrelationAnalysis, output_dir: str | Path = "results"):
        self.analysis = analysis
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(exist_ok=True)
        (self.out / "tables").mkdir(exist_ok=True)

    def save_correlation_table(self) -> Path:
        df = self.analysis.to_dataframe()
        path = self.out / "tables" / "correlation_table.csv"
        df.to_csv(path, index=False)
        print(f"Saved correlation table → {path}")
        return path

    def plot_spearman_bars(self, save: bool = True) -> plt.Figure:
        results = self.analysis.run()
        labels = [r.label for r in results]
        rhos   = [r.spearman_rho for r in results]
        colors = [PALETTE.get(l, "#888") for l in labels]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(labels[::-1], rhos[::-1], color=colors[::-1], height=0.55)
        ax.set_xlabel("Spearman ρ (correlation with human preference)", fontsize=11)
        ax.set_title("Automated Metric → Human Preference Correlation", fontsize=13, pad=12)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-0.1, 1.0)

        for bar, rho in zip(bars, rhos[::-1]):
            ax.text(
                rho + 0.01, bar.get_y() + bar.get_height() / 2,
                f"ρ={rho:.3f}",
                va="center", fontsize=9,
            )

        plt.tight_layout()
        if save:
            path = self.out / "figures" / "spearman_bars.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
        return fig

    def plot_accuracy_bars(self, save: bool = True) -> plt.Figure:
        results = self.analysis.run()
        labels   = [r.label for r in results]
        accs     = [r.pairwise_accuracy for r in results]
        aucs     = [r.roc_auc for r in results]
        colors   = [PALETTE.get(l, "#888") for l in labels]

        x = np.arange(len(labels))
        width = 0.38

        fig, ax = plt.subplots(figsize=(8, 4.5))
        b1 = ax.bar(x - width / 2, accs, width, label="Pairwise accuracy", color=colors, alpha=0.85)
        b2 = ax.bar(x + width / 2, aucs, width, label="ROC-AUC", color=colors, alpha=0.45, edgecolor="black", linewidth=0.6)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, label="Random baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0.4, 1.0)
        ax.set_title("Pairwise Ranking Accuracy & ROC-AUC vs. Human Preference", fontsize=12, pad=10)
        ax.legend(fontsize=9)

        plt.tight_layout()
        if save:
            path = self.out / "figures" / "accuracy_bars.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
        return fig

    def plot_scatter_grid(self, save: bool = True) -> plt.Figure:
        """4-panel scatter: metric delta vs. human label (jittered)."""
        df = self.analysis.df
        delta_cols = {
            "CLIP Score":       "delta_clip",
            "LPIPS Temporal":   "delta_lpips",
            "Motion Smoothness": "delta_motion",
            "FVD (pairwise)":   "delta_fvd",
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
        axes = axes.ravel()

        for ax, (label, col) in zip(axes, delta_cols.items()):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            x = df[col].values
            y = df["label"].values + np.random.uniform(-0.05, 0.05, len(df))  # jitter
            color = PALETTE.get(label, "#888")
            ax.scatter(x, y, alpha=0.35, s=18, color=color)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
            ax.set_xlabel(f"Δ {label} (A − B)", fontsize=9)
            ax.set_ylabel("Human prefers A (1) or B (0)", fontsize=9)
            ax.set_title(label, fontsize=10, color=color)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Prefers B", "Prefers A"])

            # Fit a logistic regression line
            try:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression().fit(x.reshape(-1, 1), df["label"].values.astype(int))
                xs = np.linspace(x.min(), x.max(), 100)
                ys = lr.predict_proba(xs.reshape(-1, 1))[:, 1]
                ax.plot(xs, ys, color=color, linewidth=1.8)
            except Exception:
                pass

        plt.suptitle("Metric Score Delta vs. Human Preference", fontsize=13, y=1.01)
        plt.tight_layout()
        if save:
            path = self.out / "figures" / "scatter_grid.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
        return fig

    def plot_composite_breakdown(self, save: bool = True) -> plt.Figure:
        """Pie chart showing composite weight allocation."""
        weights = {"CLIP\n(0.45)": 0.45, "LPIPS\n(0.25)": 0.25,
                   "Motion\n(0.20)": 0.20, "FVD\n(0.10)": 0.10}
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            list(weights.values()),
            labels=list(weights.keys()),
            colors=colors,
            autopct="%1.0f%%",
            startangle=140,
            pctdistance=0.75,
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.set_title("Composite Reward Weight Allocation", fontsize=12, pad=16)

        plt.tight_layout()
        if save:
            path = self.out / "figures" / "composite_weights.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
        return fig

    def generate_all(self):
        """Run the full report pipeline: tables + all figures."""
        self.save_correlation_table()
        self.plot_spearman_bars()
        self.plot_accuracy_bars()
        self.plot_scatter_grid()
        self.plot_composite_breakdown()
        self.analysis.print_summary()
        print("\nReport complete.")
