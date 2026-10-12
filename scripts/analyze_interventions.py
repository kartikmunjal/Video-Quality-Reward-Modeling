#!/usr/bin/env python3
"""Fit train-side rewards and evaluate source-disjoint intervention rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "spatial_detail_log",
    "brightness_mean",
    "temporal_delta_mean",
    "temporal_jerk_mean",
    "block_boundary_energy",
]


def flatten(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        clean_a = int(hashlib.sha256(record["pair_id"].encode()).hexdigest(), 16) % 2 == 0
        clean = record["clean_metrics"]
        degraded = record["degraded_metrics"]
        row = {key: value for key, value in record.items()
               if key not in {"clean_metrics", "degraded_metrics"}}
        row["label"] = int(clean_a)
        for feature in FEATURES:
            delta = clean[feature] - degraded[feature]
            row[f"delta_{feature}"] = delta if clean_a else -delta
        rows.append(row)
    return pd.DataFrame(rows)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    columns = [f"delta_{feature}" for feature in features]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", random_state=2025),
    )
    model.fit(train[columns], train.label)
    return model.predict_proba(test[columns])[:, 1]


def _statistics(rows: pd.DataFrame) -> dict[str, float]:
    correct = (rows.prediction == rows.label).astype(float)
    confidence = np.abs(rows.probability_a_clean - 0.5)
    return {
        "accuracy": float(correct.mean()),
        "roc_auc": float(roc_auc_score(rows.label, rows.probability_a_clean)),
        "severity_spearman_rho": float(
            spearmanr(rows.severity_index, confidence).statistic
        ),
    }


def cluster_bootstrap_metrics(predictions: pd.DataFrame, replicates: int = 2000) -> dict:
    rng = np.random.default_rng(2025)
    splits = np.asarray(sorted(predictions.official_split.unique()))
    draws = []
    for _ in range(replicates):
        parts = []
        for split in rng.choice(splits, len(splits), replace=True):
            trial = predictions[predictions.official_split == split]
            take = rng.integers(0, len(trial), len(trial))
            parts.append(trial.iloc[take])
        sampled = pd.concat(parts, ignore_index=True)
        draws.append(_statistics(sampled))
    estimates = _statistics(predictions)
    result = {
        "n_trials": int(len(splits)),
        "n_pairs": int(len(predictions)),
        "bootstrap_replicates": replicates,
    }
    for name, estimate in estimates.items():
        values = np.asarray([draw[name] for draw in draws], dtype=float)
        result[name] = {
            "estimate": estimate,
            "ci95": [float(x) for x in np.nanquantile(values, [0.025, 0.975])],
        }
    return result


def paired_metric_differences(predictions: pd.DataFrame, replicates: int) -> list[dict]:
    composite = predictions[predictions.metric == "composite"].copy()
    rng = np.random.default_rng(2026)
    results = []
    feature_names = sorted(set(predictions.metric) - {"composite"})
    for feature in feature_names:
        feature_rows = predictions[predictions.metric == feature][
            ["pair_id", "official_split", "prediction", "label"]
        ]
        paired = composite[["pair_id", "official_split", "prediction", "label"]].merge(
            feature_rows, on=["pair_id", "official_split"], suffixes=("_composite", "_feature"),
            validate="one_to_one",
        )
        paired["delta"] = (
            (paired.prediction_composite == paired.label_composite).astype(float)
            - (paired.prediction_feature == paired.label_feature).astype(float)
        )
        draws = []
        splits = np.asarray(sorted(paired.official_split.unique()))
        for _ in range(replicates):
            parts = []
            for split in rng.choice(splits, len(splits), replace=True):
                trial = paired[paired.official_split == split]
                parts.append(trial.iloc[rng.integers(0, len(trial), len(trial))])
            draws.append(float(pd.concat(parts).delta.mean()))
        draws = np.asarray(draws)
        # Two-sided bootstrap sign test; Holm adjustment is applied below.
        p_raw = min(1.0, 2 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
        results.append({
            "comparison": f"composite_minus_{feature}",
            "estimate": float(paired.delta.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "p_raw": p_raw,
            "n_trials": int(len(splits)),
            "n_pairs": int(len(paired)),
            "bootstrap_replicates": replicates,
        })
    ordered = sorted(range(len(results)), key=lambda i: results[i]["p_raw"])
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (len(results) - rank) * results[index]["p_raw"])
        running = max(running, adjusted)
        results[index]["p_holm"] = running
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    frame = flatten(records)
    if frame.empty or not np.isfinite(frame.filter(like="delta_").to_numpy()).all():
        raise SystemExit("score table is empty or nonfinite")

    prediction_rows = []
    for split in sorted(frame.official_split.unique()):
        split_frame = frame[frame.official_split == split]
        train_all = split_frame[split_frame.partition == "train"]
        test_all = split_frame[split_frame.partition == "test"]
        train_groups = set(train_all.source_group)
        test_groups = set(test_all.source_group)
        if overlap := train_groups & test_groups:
            raise SystemExit(f"source leakage in split {split}: {sorted(overlap)[:5]}")
        for held_family in sorted(frame.degradation_family.unique()):
            train = train_all[train_all.degradation_family != held_family]
            test = test_all[test_all.degradation_family == held_family].copy()
            for metric_name, feature_set in [("composite", FEATURES)] + [
                (feature, [feature]) for feature in FEATURES
            ]:
                probabilities = fit_predict(train, test, feature_set)
                for row, probability in zip(test.itertuples(index=False), probabilities):
                    prediction_rows.append({
                        "pair_id": row.pair_id,
                        "official_split": int(split),
                        "heldout_family": held_family,
                        "metric": metric_name,
                        "label": int(row.label),
                        "probability_a_clean": float(probability),
                        "prediction": int(probability >= 0.5),
                        "severity_index": int(row.severity_index),
                    })
    predictions = pd.DataFrame(prediction_rows)
    metrics = {"study_id": "ucf101_quality_reward_v1", "metrics": {}}
    for metric_name, rows in predictions.groupby("metric"):
        metrics["metrics"][metric_name] = cluster_bootstrap_metrics(
            rows, args.bootstrap_replicates
        )
    metrics["by_degradation_family"] = {}
    for family, rows in predictions[predictions.metric == "composite"].groupby("heldout_family"):
        metrics["by_degradation_family"][family] = cluster_bootstrap_metrics(
            rows, args.bootstrap_replicates
        )
    metrics["paired_comparisons"] = paired_metric_differences(
        predictions, args.bootstrap_replicates
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_json(args.output_dir / "predictions.jsonl", orient="records", lines=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
