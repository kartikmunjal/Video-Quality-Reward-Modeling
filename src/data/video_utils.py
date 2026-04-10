"""Video loading utilities with decord/opencv fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_video_frames(
    path: str | Path,
    num_frames: int = 16,
    height: int = 480,
    width: int = 720,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load a video file and return uniformly sampled frames.

    Returns
    -------
    np.ndarray of shape (T, H, W, C), dtype float32 in [0, 1] if normalize=True,
    else uint8 in [0, 255].
    """
    path = str(path)
    try:
        frames = _load_decord(path, num_frames, height, width)
    except Exception:
        frames = _load_opencv(path, num_frames, height, width)

    if normalize:
        frames = frames.astype(np.float32) / 255.0
    return frames


def _load_decord(path: str, num_frames: int, height: int, width: int) -> np.ndarray:
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path, width=width, height=height)
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    return frames


def _load_opencv(path: str, num_frames: int, height: int, width: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")
    return np.stack(frames)  # (T, H, W, C)


def sample_frames(
    frames: np.ndarray,
    n: int,
    strategy: str = "uniform",
) -> np.ndarray:
    """
    Subsample frames from a (T, H, W, C) array.

    Parameters
    ----------
    strategy : "uniform" | "random" | "first_last"
    """
    T = frames.shape[0]
    if n >= T:
        return frames
    if strategy == "uniform":
        idx = np.linspace(0, T - 1, n, dtype=int)
    elif strategy == "random":
        idx = np.sort(np.random.choice(T, n, replace=False))
    elif strategy == "first_last":
        mid = np.linspace(1, T - 2, n - 2, dtype=int)
        idx = np.concatenate([[0], mid, [T - 1]])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return frames[idx]
