"""
VideoEvaluator: score a directory of videos with all metrics in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.video_utils import load_video_frames
from ..metrics import (
    CLIPScore,
    FVDScore,
    LPIPSTemporal,
    MotionSmoothness,
    MetricResult,
)


@dataclass
class EvaluationResult:
    video_path: str
    prompt: str | None
    clip: MetricResult
    lpips: MetricResult
    motion: MetricResult
    fvd: MetricResult

    @property
    def composite(self) -> float:
        return round(
            0.45 * self.clip.score
            + 0.25 * self.lpips.score
            + 0.20 * self.motion.score
            + 0.10 * self.fvd.score,
            4,
        )

    def to_dict(self) -> dict:
        return {
            "video": self.video_path,
            "prompt": self.prompt or "",
            "clip_score": self.clip.score,
            "lpips_temporal": self.lpips.score,
            "motion_smoothness": self.motion.score,
            "fvd_score": self.fvd.score,
            "composite": self.composite,
        }


class VideoEvaluator:
    """
    Scores videos with all four metrics in a single pass.

    Parameters
    ----------
    device : str, optional
    reference_features : np.ndarray, optional
        Precomputed I3D features from real reference videos, shape (N, D).
        Used by FVDScore to compute feature-space distance to real distribution.
    num_frames : int
        Number of frames to decode per video.
    """

    def __init__(
        self,
        device: str | None = None,
        reference_features: np.ndarray | None = None,
        num_frames: int = 16,
    ):
        self.num_frames = num_frames
        self._clip   = CLIPScore(device=device)
        self._lpips  = LPIPSTemporal(device=device)
        self._motion = MotionSmoothness()
        self._fvd    = FVDScore(reference_features=reference_features, device=device)

    def score(self, video_path: str | Path, prompt: str | None = None) -> EvaluationResult:
        frames = load_video_frames(str(video_path), num_frames=self.num_frames)
        return EvaluationResult(
            video_path=str(video_path),
            prompt=prompt,
            clip=self._clip.compute(frames, prompt) if prompt else MetricResult("clip_score", float("nan")),
            lpips=self._lpips.compute(frames),
            motion=self._motion.compute(frames),
            fvd=self._fvd.compute(frames),
        )

    def score_directory(
        self,
        video_dir: str | Path,
        prompts: dict[str, str] | None = None,
        glob: str = "*.mp4",
    ) -> pd.DataFrame:
        """
        Score all videos in a directory.

        Parameters
        ----------
        prompts : dict mapping filename stem → prompt string, optional.
        glob : file pattern to match.

        Returns
        -------
        pd.DataFrame with one row per video and columns for all metric scores.
        """
        video_dir = Path(video_dir)
        video_files = sorted(video_dir.glob(glob))
        if not video_files:
            raise FileNotFoundError(f"No files matching '{glob}' in {video_dir}")

        rows = []
        for vf in tqdm(video_files, desc="Scoring videos"):
            prompt = (prompts or {}).get(vf.stem)
            try:
                result = self.score(vf, prompt)
                rows.append(result.to_dict())
            except Exception as e:
                rows.append({"video": str(vf), "prompt": prompt or "", "error": str(e)})

        return pd.DataFrame(rows)

    def score_pairs(
        self,
        pairs: list[tuple[str, str, str | None]],
    ) -> Iterator[tuple[EvaluationResult, EvaluationResult]]:
        """
        Yield (result_a, result_b) for a list of (path_a, path_b, prompt) tuples.
        Useful for directly scoring preference pairs when video files are available.
        """
        for path_a, path_b, prompt in tqdm(pairs, desc="Scoring pairs"):
            yield self.score(path_a, prompt), self.score(path_b, prompt)
