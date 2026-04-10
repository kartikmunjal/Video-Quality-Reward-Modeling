"""
CLIP score: prompt–video semantic alignment.

Encodes `n_frames` uniformly sampled frames and the text prompt with CLIP
ViT-B/32, then returns the average cosine similarity across frames, normalized
to [0, 1].
"""

from __future__ import annotations

import numpy as np
import torch

from .base import MetricResult, VideoMetric


class CLIPScore(VideoMetric):
    """
    Measures how well a video's visual content matches the text prompt.

    Score = mean over sampled frames of cos_sim(frame_emb, text_emb),
    normalized from [-1, 1] to [0, 1].

    Higher is better: score=1 means perfect semantic alignment.
    """

    def __init__(
        self,
        n_frames: int = 4,
        model_name: str = "ViT-B/32",
        device: str | None = None,
    ):
        self.n_frames = n_frames
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None

    def _load_model(self):
        if self._model is None:
            import clip
            self._model, self._preprocess = clip.load(self.model_name, device=self.device)
            self._model.eval()

    @property
    def name(self) -> str:
        return "clip_score"

    def compute(self, frames: np.ndarray, prompt: str | None = None) -> MetricResult:
        """
        Parameters
        ----------
        frames : np.ndarray (T, H, W, C) float32 in [0, 1]
        prompt : str  — required for CLIP score
        """
        if prompt is None:
            raise ValueError("CLIPScore requires a text prompt.")

        self._load_model()
        import clip
        from PIL import Image

        T = frames.shape[0]
        indices = np.linspace(0, T - 1, min(self.n_frames, T), dtype=int)
        sampled = frames[indices]  # (n, H, W, C)

        # Encode frames
        frame_imgs = [
            self._preprocess(Image.fromarray((f * 255).astype(np.uint8))).unsqueeze(0)
            for f in sampled
        ]
        frame_tensor = torch.cat(frame_imgs, dim=0).to(self.device)  # (n, 3, 224, 224)

        with torch.no_grad():
            image_features = self._model.encode_image(frame_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_tokens = clip.tokenize([prompt]).to(self.device)
            text_features = self._model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity per frame, then average
        sims = (image_features @ text_features.T).squeeze(-1).cpu().numpy()  # (n,)
        raw = float(sims.mean())
        score = float((raw + 1.0) / 2.0)  # [-1, 1] → [0, 1]

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            raw=round(raw, 4),
            metadata={"n_frames_sampled": len(indices), "model": self.model_name},
        )
