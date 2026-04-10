#!/usr/bin/env python3
"""
Regenerate all report figures and tables from a saved correlation CSV.

Use this to re-style figures without re-running the benchmark.

Usage
-----
python scripts/generate_report.py --results_dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.correlation import CorrelationAnalysis, MetricCorrelation, METRICS, METRIC_LABELS
from benchmark.report import BenchmarkReport
from data.preference_loader import load_preferences


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",  default="results/")
    p.add_argument("--preferences",  default="data/human_preferences/preferences.json")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    corr_csv = results_dir / "tables" / "correlation_table.csv"
    if not corr_csv.exists():
        print(f"No correlation table found at {corr_csv}. Run run_benchmark.py first.")
        sys.exit(1)

    # Reload the preference dataframe for scatter plots
    dataset = load_preferences(args.preferences)
    df = dataset.to_dataframe(exclude_ties=True)

    analysis = CorrelationAnalysis(df)
    analysis.run()  # populate results from fresh computation

    report = BenchmarkReport(analysis, output_dir=args.results_dir)
    report.generate_all()


if __name__ == "__main__":
    main()
