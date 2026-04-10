"""
Composite reward: weighted combination of all four metrics.

Default weights mirror what was found optimal in the Video-Generation
reward training ablation: CLIP carries the most weight (prompt alignment
is the primary quality axis), temporal consistency second, motion third.
FVD is included but weighted low because as a pairwise score it has
higher variance than the other three signals.

score = w_clip * clip_score + w_lpips * lpips_temporal
      + w_motion * motion_smoothness + w_fvd * fvd_score
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import MetricResult, VideoMetric
from .clip_score import CLIPScore
from .fvd import FVDScore
from .lpips_temporal import LPIPSTemporal
from .motion_smoothness import MotionSmoothness

DEFAULT_WEIGHTS = {
    "clip_score": 0.45,
    "lpips_temporal": 0.25,
    "motion_smoothness": 0.20,
    "fvd_score": 0.10,
}


@dataclass
class CompositeResult:
    """Scores from all constituent metrics plus the weighted composite."""
    clip: MetricResult
    lpips: MetricResult
    motion: MetricResult
    fvd: MetricResult
    composite: float
    weights: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "clip_score": self.clip.score,
            "lpips_temporal": self.lpips.score,
            "motion_smoothness": self.motion.score,
            "fvd_score": self.fvd.score,
            "composite": self.composite,
        }


class CompositeMetric(VideoMetric):
    """
    Weighted combination of CLIP, LPIPS, motion smoothness, and FVD.

    This is the primary signal used by the Video-Generation reward model.
    Returning all constituent scores alongside the composite makes it easy
    to run the full benchmarking suite in a single forward pass.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        device: str | None = None,
        reference_features: np.ndarray | None = None,
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        _validate_weights(self.weights)

        self._clip   = CLIPScore(device=device)
        self._lpips  = LPIPSTemporal(device=device)
        self._motion = MotionSmoothness()
        self._fvd    = FVDScore(reference_features=reference_features, device=device)

    @property
    def name(self) -> str:
        return "composite"

    def compute(self, frames: np.ndarray, prompt: str | None = None) -> MetricResult:
        result = self.compute_full(frames, prompt)
        return MetricResult(
            name=self.name,
            score=result.composite,
            metadata=result.to_dict(),
        )

    def compute_full(self, frames: np.ndarray, prompt: str | None = None) -> CompositeResult:
        """Run all four metrics and return a CompositeResult with all scores."""
        clip_r   = self._clip.compute(frames, prompt)
        lpips_r  = self._lpips.compute(frames, prompt)
        motion_r = self._motion.compute(frames, prompt)
        fvd_r    = self._fvd.compute(frames, prompt)

        composite = (
            self.weights["clip_score"]       * clip_r.score
            + self.weights["lpips_temporal"] * lpips_r.score
            + self.weights["motion_smoothness"] * motion_r.score
            + self.weights["fvd_score"]      * fvd_r.score
        )

        return CompositeResult(
            clip=clip_r,
            lpips=lpips_r,
            motion=motion_r,
            fvd=fvd_r,
            composite=round(float(composite), 4),
            weights=self.weights,
        )


def _validate_weights(weights: dict[str, float]):
    expected = {"clip_score", "lpips_temporal", "motion_smoothness", "fvd_score"}
    missing = expected - set(weights)
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
