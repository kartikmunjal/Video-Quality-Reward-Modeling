"""
Motion smoothness metric via optical flow coefficient of variation.

Uses OpenCV's Farneback dense optical flow. Smooth, intentional motion has
low variability in flow magnitude; jitter or flickering has high variability.

score = clip(1 - CV / 2, 0, 1)   where CV = std(flow_mag) / mean(flow_mag)
"""

from __future__ import annotations

import numpy as np

from .base import MetricResult, VideoMetric


class MotionSmoothness(VideoMetric):
    """
    Optical flow coefficient of variation (CV) as a smoothness proxy.

    A video with smooth, coherent motion has low CV; a jittery or flickering
    video has high CV. The raw CV is mapped to [0, 1] so that higher is better.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        self.flow_kwargs = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )

    @property
    def name(self) -> str:
        return "motion_smoothness"

    def compute(self, frames: np.ndarray, prompt: str | None = None) -> MetricResult:
        """
        Parameters
        ----------
        frames : np.ndarray (T, H, W, C) float32 in [0, 1]
        """
        import cv2

        T = frames.shape[0]
        if T < 2:
            return MetricResult(name=self.name, score=1.0, raw=0.0)

        # Convert frames to uint8 grayscale
        def to_gray(f: np.ndarray) -> np.ndarray:
            uint8 = (f * 255).astype(np.uint8)
            return cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY)

        all_magnitudes = []
        for i in range(T - 1):
            gray1 = to_gray(frames[i])
            gray2 = to_gray(frames[i + 1])
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.flow_kwargs)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            all_magnitudes.append(mag.ravel())

        magnitudes = np.concatenate(all_magnitudes)
        mean_mag = float(magnitudes.mean())
        std_mag = float(magnitudes.std())

        if mean_mag < 1e-6:
            # Static video — perfectly smooth by definition
            cv = 0.0
        else:
            cv = std_mag / mean_mag

        raw = cv
        score = float(np.clip(1.0 - cv / 2.0, 0.0, 1.0))

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            raw=round(raw, 4),
            metadata={"mean_flow_magnitude": round(mean_mag, 4), "cv": round(cv, 4)},
        )
