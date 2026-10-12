#!/usr/bin/env python3
"""Build deterministic, source-grouped controlled-intervention pair specs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def order_key(study_id: str, split: int, family: str, path: str) -> str:
    return hashlib.sha256(f"{study_id}:{split}:{family}:{path}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/study_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frame = pd.read_json(args.assignments, lines=True)
    required = {"official_split", "partition", "relative_path", "source_group", "sha256"}
    if missing := required - set(frame):
        raise SystemExit(f"missing assignment columns: {sorted(missing)}")
    rows = []
    for split in config["official_splits"]:
        for partition in ("train", "test"):
            candidates = frame[(frame.official_split == split) & (frame.partition == partition)]
            for family, severities in config["degradations"].items():
                n = config["pairs_per_family_per_split"][partition]
                if args.pilot:
                    n = max(3, round(n * config["pilot_max_fraction"]))
                ordered = candidates.assign(
                    _key=[order_key(config["study_id"], split, family, x)
                          for x in candidates.relative_path]
                ).sort_values("_key")
                if len(ordered) < n:
                    raise SystemExit(f"not enough candidates for split={split}, {partition}, {family}")
                selected = ordered.head(n)
                for index, (_, source) in enumerate(selected.iterrows()):
                    severity_index = index % len(severities)
                    pair_id = hashlib.sha256(
                        f"{split}:{partition}:{family}:{source.relative_path}".encode()
                    ).hexdigest()[:20]
                    rows.append({
                        "schema_version": 1,
                        "study_id": config["study_id"],
                        "pair_id": pair_id,
                        "official_split": int(split),
                        "partition": partition,
                        "relative_path": source.relative_path,
                        "source_group": source.source_group,
                        "source_sha256": source.sha256,
                        "degradation_family": family,
                        "severity_index": severity_index + 1,
                        "severity_value": severities[severity_index],
                        "preferred": "clean",
                    })
    output = pd.DataFrame(rows)
    if output.pair_id.duplicated().any():
        raise SystemExit("pair ID collision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_json(args.output, orient="records", lines=True)
    print(json.dumps({"pairs": len(output), "pilot": args.pilot}))


if __name__ == "__main__":
    main()
