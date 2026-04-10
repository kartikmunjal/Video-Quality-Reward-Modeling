"""Load and validate the human preference dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class VideoScores:
    clip_score: float
    lpips_temporal: float
    motion_smoothness: float
    fvd_score: float
    composite: float

    @classmethod
    def from_dict(cls, d: dict) -> "VideoScores":
        return cls(
            clip_score=d["clip_score"],
            lpips_temporal=d["lpips_temporal"],
            motion_smoothness=d["motion_smoothness"],
            fvd_score=d["fvd_score"],
            composite=d["composite"],
        )

    def to_dict(self) -> dict:
        return {
            "clip_score": self.clip_score,
            "lpips_temporal": self.lpips_temporal,
            "motion_smoothness": self.motion_smoothness,
            "fvd_score": self.fvd_score,
            "composite": self.composite,
        }


@dataclass
class PreferencePair:
    id: str
    prompt: str
    model_a: str
    model_b: str
    path_a: str
    path_b: str
    annotations: list[dict]
    majority_choice: Literal["a", "b", "tie"]
    agreement: float
    scores_a: VideoScores
    scores_b: VideoScores

    @property
    def is_tie(self) -> bool:
        return self.majority_choice == "tie"

    @property
    def preferred_scores(self) -> VideoScores | None:
        if self.is_tie:
            return None
        return self.scores_a if self.majority_choice == "a" else self.scores_b

    @property
    def rejected_scores(self) -> VideoScores | None:
        if self.is_tie:
            return None
        return self.scores_b if self.majority_choice == "a" else self.scores_a

    def metric_predicts_correctly(self, metric: str) -> bool | None:
        """True if the named metric ranks the preferred video higher."""
        if self.is_tie:
            return None
        pref = getattr(self.preferred_scores, metric)
        rej = getattr(self.rejected_scores, metric)
        return pref > rej

    @classmethod
    def from_dict(cls, d: dict) -> "PreferencePair":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            model_a=d["video_a"]["model"],
            model_b=d["video_b"]["model"],
            path_a=d["video_a"]["path"],
            path_b=d["video_b"]["path"],
            annotations=d["annotations"],
            majority_choice=d["majority_choice"],
            agreement=d["agreement"],
            scores_a=VideoScores.from_dict(d["automated_scores"]["video_a"]),
            scores_b=VideoScores.from_dict(d["automated_scores"]["video_b"]),
        )


@dataclass
class PreferenceDataset:
    pairs: list[PreferencePair]
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    @property
    def non_tie_pairs(self) -> list[PreferencePair]:
        return [p for p in self.pairs if not p.is_tie]

    def to_dataframe(self, exclude_ties: bool = True) -> pd.DataFrame:
        """Flatten to a tidy DataFrame for analysis."""
        rows = []
        for pair in self.pairs:
            if exclude_ties and pair.is_tie:
                continue
            row = {
                "id": pair.id,
                "prompt": pair.prompt,
                "model_a": pair.model_a,
                "model_b": pair.model_b,
                "majority_choice": pair.majority_choice,
                "agreement": pair.agreement,
                # scores A
                "clip_a": pair.scores_a.clip_score,
                "lpips_a": pair.scores_a.lpips_temporal,
                "motion_a": pair.scores_a.motion_smoothness,
                "fvd_a": pair.scores_a.fvd_score,
                "composite_a": pair.scores_a.composite,
                # scores B
                "clip_b": pair.scores_b.clip_score,
                "lpips_b": pair.scores_b.lpips_temporal,
                "motion_b": pair.scores_b.motion_smoothness,
                "fvd_b": pair.scores_b.fvd_score,
                "composite_b": pair.scores_b.composite,
                # deltas (A - B), positive means A is better per metric
                "delta_clip": pair.scores_a.clip_score - pair.scores_b.clip_score,
                "delta_lpips": pair.scores_a.lpips_temporal - pair.scores_b.lpips_temporal,
                "delta_motion": pair.scores_a.motion_smoothness - pair.scores_b.motion_smoothness,
                "delta_fvd": pair.scores_a.fvd_score - pair.scores_b.fvd_score,
                "delta_composite": pair.scores_a.composite - pair.scores_b.composite,
                # binary label: 1 if A preferred, 0 if B preferred
                "label": 1 if pair.majority_choice == "a" else 0,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def accuracy_per_metric(self, exclude_ties: bool = True) -> dict[str, float]:
        """Pairwise ranking accuracy for each metric."""
        metrics = ["clip_score", "lpips_temporal", "motion_smoothness", "fvd_score", "composite"]
        results = {}
        pairs = self.non_tie_pairs if exclude_ties else self.pairs
        for metric in metrics:
            correct = [p.metric_predicts_correctly(metric) for p in pairs]
            correct = [c for c in correct if c is not None]
            results[metric] = float(np.mean(correct)) if correct else float("nan")
        return results

    def summary(self) -> dict:
        return {
            "total_pairs": len(self.pairs),
            "ties": sum(1 for p in self.pairs if p.is_tie),
            "non_tie_pairs": len(self.non_tie_pairs),
            "cohen_kappa": self.metadata.get("cohen_kappa"),
            "raw_agreement": self.metadata.get("raw_agreement"),
            "models": self.metadata.get("models_compared", []),
        }


def load_preferences(path: str | Path) -> PreferenceDataset:
    """Load a preferences.json file and return a PreferenceDataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preferences file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    pairs = [PreferencePair.from_dict(d) for d in raw["pairs"]]
    return PreferenceDataset(pairs=pairs, metadata=raw.get("metadata", {}))
