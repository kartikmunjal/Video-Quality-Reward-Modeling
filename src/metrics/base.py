"""Abstract base class for video quality metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricResult:
    name: str
    score: float          # higher is better, always in [0, 1]
    raw: float | None = None  # raw value before normalization (e.g. raw LPIPS)
    metadata: dict | None = None


class VideoMetric(ABC):
    """
    Base class for a video quality metric.

    Subclasses must implement `compute()`, which takes a video as a NumPy array
    of shape (T, H, W, C) with float32 values in [0, 1], and an optional text
    prompt string, and returns a MetricResult.

    All scores are normalized to [0, 1] where higher is better.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute(
        self,
        frames: np.ndarray,
        prompt: str | None = None,
    ) -> MetricResult:
        """
        Parameters
        ----------
        frames : np.ndarray
            Video frames, shape (T, H, W, C), float32 in [0, 1].
        prompt : str, optional
            Text prompt used to generate the video.

        Returns
        -------
        MetricResult with score in [0, 1] (higher = better).
        """
        ...

    def compute_batch(
        self,
        frames_list: list[np.ndarray],
        prompts: list[str] | None = None,
    ) -> list[MetricResult]:
        if prompts is None:
            prompts = [None] * len(frames_list)
        return [self.compute(f, p) for f, p in zip(frames_list, prompts)]
