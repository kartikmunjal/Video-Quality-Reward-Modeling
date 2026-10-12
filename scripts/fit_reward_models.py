#!/usr/bin/env python3
"""Fit deployable split-specific reward models using training-side pairs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_interventions import FEATURES, flatten


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    frame = flatten(records)
    columns = [f"delta_{feature}" for feature in FEATURES]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"study_id": "ucf101_quality_reward_v1", "models": []}
    prediction_rows = []
    for split in sorted(frame.official_split.unique()):
        train = frame[(frame.official_split == split) & (frame.partition == "train")]
        test = frame[(frame.official_split == split) & (frame.partition == "test")]
        if set(train.source_group) & set(test.source_group):
            raise SystemExit(f"source leakage in split {split}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", random_state=2025),
        )
        model.fit(train[columns], train.label)
        probability = model.predict_proba(test[columns])[:, 1]
        rng = np.random.default_rng(3100 + int(split))
        accuracy_draws = []
        auc_draws = []
        groups = np.asarray(sorted(test.source_group.unique()))
        group_to_index = {group: index for index, group in enumerate(groups)}
        row_group_index = test.source_group.map(group_to_index).to_numpy()
        correct = (test.label.to_numpy() == (probability >= 0.5)).astype(float)
        for _ in range(args.bootstrap_replicates):
            counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
            weights = counts[row_group_index]
            accuracy_draws.append(float(np.average(correct, weights=weights)))
            auc_draws.append(float(roc_auc_score(test.label, probability, sample_weight=weights)))
        accuracy = float(accuracy_score(test.label, probability >= 0.5))
        auc = float(roc_auc_score(test.label, probability))
        result = {
            "official_split": int(split),
            "n_train_pairs": int(len(train)),
            "n_test_pairs": int(len(test)),
            "n_train_sources": int(train.source_group.nunique()),
            "n_test_sources": int(test.source_group.nunique()),
            "test_accuracy": {"estimate": accuracy, "ci95": [float(x) for x in np.quantile(accuracy_draws, [0.025, 0.975])]},
            "test_roc_auc": {"estimate": auc, "ci95": [float(x) for x in np.quantile(auc_draws, [0.025, 0.975])]},
            "bootstrap_replicates": args.bootstrap_replicates,
            "features": FEATURES,
        }
        path = args.output_dir / f"reward_split{split}.joblib"
        joblib.dump({"model": model, "features": FEATURES, "result": result}, path)
        result["checkpoint"] = path.name
        metrics["models"].append(result)
        for row, score in zip(test.itertuples(index=False), probability):
            prediction_rows.append({
                "pair_id": row.pair_id,
                "source_group": row.source_group,
                "official_split": int(split),
                "degradation_family": row.degradation_family,
                "severity_index": int(row.severity_index),
                "label": int(row.label),
                "probability_a_clean": float(score),
                "prediction": int(score >= 0.5),
            })
    pd.DataFrame(prediction_rows).to_json(
        args.output_dir / "predictions.jsonl", orient="records", lines=True
    )
    (args.output_dir / "fit_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(args.output_dir / "fit_metrics.json")


if __name__ == "__main__":
    main()
