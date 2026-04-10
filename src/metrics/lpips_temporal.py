"""
LPIPS temporal consistency metric.

Measures frame-to-frame perceptual similarity using LPIPS (Learned Perceptual
Image Patch Similarity) with an AlexNet backbone. Low LPIPS between consecutive
frames means the video doesn't flicker or jitter.

Score = 1 - mean_lpips  (so higher = more temporally consistent)
"""

from __future__ import annotations

import numpy as np
import torch

from .base import MetricResult, VideoMetric


class LPIPSTemporal(VideoMetric):
    """
    Frame-to-frame perceptual consistency via LPIPS.

    Computes LPIPS between every pair of consecutive frames and averages.
    Lower raw LPIPS = smoother video. Score is inverted so higher is better.

    score = clip(1 - mean_lpips, 0, 1)
    """

    def __init__(self, net: str = "alex", device: str | None = None):
        self.net = net
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_fn = None

    def _load_model(self):
        if self._lpips_fn is None:
            import lpips
            self._lpips_fn = lpips.LPIPS(net=self.net).to(self.device)
            self._lpips_fn.eval()

    @property
    def name(self) -> str:
        return "lpips_temporal"

    def compute(self, frames: np.ndarray, prompt: str | None = None) -> MetricResult:
        """
        Parameters
        ----------
        frames : np.ndarray (T, H, W, C) float32 in [0, 1]
        """
        self._load_model()

        T = frames.shape[0]
        if T < 2:
            return MetricResult(name=self.name, score=1.0, raw=0.0)

        # Convert to torch tensors in [-1, 1], shape (1, 3, H, W)
        def to_tensor(f: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
            return (t * 2.0 - 1.0).to(self.device)

        distances = []
        with torch.no_grad():
            for i in range(T - 1):
                d = self._lpips_fn(to_tensor(frames[i]), to_tensor(frames[i + 1]))
                distances.append(d.item())

        raw = float(np.mean(distances))
        score = float(np.clip(1.0 - raw, 0.0, 1.0))

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            raw=round(raw, 4),
            metadata={"n_frame_pairs": len(distances), "net": self.net},
        )
