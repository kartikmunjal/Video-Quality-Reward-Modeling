#!/usr/bin/env python3
"""Apply frozen degradations and emit raw, auditable metric observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_frames(path: Path, count: int = 16) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < 2 or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid video {path}")
    indices = np.unique(np.linspace(0, len(frames) - 1, count).astype(int))
    return np.stack([frames[i] for i in indices]), fps


def h264_roundtrip(frames: np.ndarray, fps: float, crf: int) -> np.ndarray:
    import imageio_ffmpeg

    height, width = frames.shape[1:3]
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory) / "degraded.mp4")
        writer = imageio_ffmpeg.write_frames(
            output, (width, height), fps=fps, codec="libx264",
            output_params=["-crf", str(crf)],
        )
        writer.send(None)
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
        writer.close()
        decoded, _ = load_frames(Path(output), count=len(frames))
    return decoded


def degrade(frames: np.ndarray, fps: float, family: str, value, seed: int) -> np.ndarray:
    if family == "gaussian_blur_sigma":
        return np.stack([cv2.GaussianBlur(frame, (0, 0), float(value)) for frame in frames])
    if family == "brightness_multiplier":
        return np.clip(frames.astype(np.float32) * float(value), 0, 255).astype(np.uint8)
    if family == "downsample_short_side":
        target = int(value)
        height, width = frames.shape[1:3]
        scale = target / min(height, width)
        small = (max(1, round(width * scale)), max(1, round(height * scale)))
        return np.stack([
            cv2.resize(cv2.resize(frame, small, interpolation=cv2.INTER_AREA),
                       (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ])
    if family == "frame_drop_fraction":
        keep_n = max(2, round(len(frames) * (1 - float(value))))
        keep = np.linspace(0, len(frames) - 1, keep_n).astype(int)
        mapping = np.asarray([keep[np.argmin(abs(keep - i))] for i in range(len(frames))])
        return frames[mapping]
    if family == "temporal_permutation_window":
        rng = np.random.default_rng(seed)
        output = frames.copy()
        window = int(value)
        for start in range(0, len(output), window):
            stop = min(len(output), start + window)
            output[start:stop] = output[start:stop][rng.permutation(stop - start)]
        return output
    if family == "h264_crf":
        return h264_roundtrip(frames, fps, int(value))
    raise ValueError(f"unknown degradation: {family}")


def metrics(frames: np.ndarray) -> dict:
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames])
    sharpness = np.median([cv2.Laplacian(frame, cv2.CV_64F).var() for frame in gray])
    temporal_delta = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0)))
    temporal_jerk = np.mean(np.abs(np.diff(gray.astype(np.float32), n=2, axis=0)))
    # JPEG-style block-boundary energy is also sensitive to H.264 blocking and
    # is treated as a literal diagnostic, not a calibrated perceptual score.
    x_boundaries = np.arange(8, gray.shape[2], 8)
    y_boundaries = np.arange(8, gray.shape[1], 8)
    vertical = np.abs(
        gray[:, :, x_boundaries].astype(float) - gray[:, :, x_boundaries - 1].astype(float)
    ).mean()
    horizontal = np.abs(
        gray[:, y_boundaries, :].astype(float) - gray[:, y_boundaries - 1, :].astype(float)
    ).mean()
    return {
        "spatial_detail_log": float(np.log1p(sharpness)),
        "brightness_mean": float(gray.mean()),
        "temporal_delta_mean": float(temporal_delta),
        "temporal_jerk_mean": float(temporal_jerk),
        "block_boundary_energy": float((vertical + horizontal) / 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = pd.read_json(args.pairs, lines=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in pairs.itertuples(index=False):
            path = args.video_root / Path(row.relative_path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row.source_sha256:
                raise SystemExit(f"source hash mismatch: {path}")
            frames, fps = load_frames(path)
            seed = int(hashlib.sha256(row.pair_id.encode()).hexdigest()[:8], 16)
            altered = degrade(frames, fps, row.degradation_family, row.severity_value, seed)
            record = row._asdict()
            record["clean_metrics"] = metrics(frames)
            record["degraded_metrics"] = metrics(altered)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"scored_pairs": len(pairs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
