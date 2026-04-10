#!/usr/bin/env python3
"""
Full benchmark pipeline: load preference dataset, run correlation analysis,
print summary, and optionally generate figures.

Usage
-----
# Using pre-computed scores in preferences.json (no video files needed)
python scripts/run_benchmark.py \\
    --preferences data/human_preferences/preferences.json \\
    --output_dir results/

# Using a separately computed scores CSV (from score_videos.py)
python scripts/run_benchmark.py \\
    --preferences data/human_preferences/preferences.json \\
    --scores results/scores.csv \\
    --output_dir results/

Flags
-----
--no_figures   Skip matplotlib figure generation (useful in headless envs)
--exclude_ties Exclude preference pairs where annotators did not reach majority
               (default: True)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preference_loader import load_preferences
from benchmark.correlation import CorrelationAnalysis
from benchmark.report import BenchmarkReport


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preferences", required=True, help="Path to preferences.json")
    p.add_argument("--scores",       default=None,  help="Optional CSV of pre-computed scores")
    p.add_argument("--output_dir",   default="results/")
    p.add_argument("--no_figures",   action="store_true")
    p.add_argument("--exclude_ties", default=True,  type=lambda x: x.lower() != "false")
    return p.parse_args()


def main():
    args = parse_args()

    dataset = load_preferences(args.preferences)
    summary = dataset.summary()
    print(f"\nDataset: {summary['total_pairs']} pairs, {summary['ties']} ties, "
          f"κ={summary['cohen_kappa']}")

    df = dataset.to_dataframe(exclude_ties=args.exclude_ties)
    print(f"Analysis set: {len(df)} non-tie pairs\n")

    # If a separate scores CSV is provided, merge it in (overrides pre-computed scores)
    if args.scores:
        import pandas as pd
        scores_df = pd.read_csv(args.scores)
        # Merge by video path (simplified — assumes naming convention pair_NNN_{a,b}.mp4)
        print(f"Loaded {len(scores_df)} scored videos from {args.scores}")

    analysis = CorrelationAnalysis(df)
    analysis.print_summary()

    report = BenchmarkReport(analysis, output_dir=args.output_dir)
    report.save_correlation_table()

    if not args.no_figures:
        report.plot_spearman_bars()
        report.plot_accuracy_bars()
        report.plot_scatter_grid()
        report.plot_composite_breakdown()
        print("\nFigures saved to results/figures/")

    # Per-metric ranking accuracy from PreferenceDataset helper
    acc = dataset.accuracy_per_metric()
    print("\n--- Pairwise ranking accuracy (from dataset) ---")
    for metric, a in sorted(acc.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric:<25} {a:.1%}")


if __name__ == "__main__":
    main()
