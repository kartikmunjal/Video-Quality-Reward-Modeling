#!/usr/bin/env python3
"""
Score a directory of videos with all quality metrics.

Usage
-----
python scripts/score_videos.py \\
    --video_dir data/videos/ \\
    --prompts data/sample_prompts.txt \\
    --output results/scores.csv

The prompts file should have one prompt per line. Videos are matched to
prompts by sort order (video[0] ↔ prompt[0], etc.).  Pass --prompts_json
instead if you have a filename→prompt mapping as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.evaluator import VideoEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Score videos with all quality metrics.")
    p.add_argument("--video_dir",    required=True,  help="Directory containing .mp4 files")
    p.add_argument("--output",       default="results/scores.csv", help="Output CSV path")
    p.add_argument("--prompts",      default=None,   help="Text file, one prompt per line")
    p.add_argument("--prompts_json", default=None,   help="JSON mapping filename_stem → prompt")
    p.add_argument("--glob",         default="*.mp4", help="File glob pattern")
    p.add_argument("--num_frames",   type=int, default=16)
    p.add_argument("--device",       default=None)
    return p.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)

    # Build prompt mapping
    prompts: dict[str, str] | None = None
    if args.prompts_json:
        with open(args.prompts_json) as f:
            prompts = json.load(f)
    elif args.prompts:
        with open(args.prompts) as f:
            lines = [l.strip() for l in f if l.strip()]
        video_files = sorted(video_dir.glob(args.glob))
        prompts = {vf.stem: lines[i] for i, vf in enumerate(video_files) if i < len(lines)}

    evaluator = VideoEvaluator(device=args.device, num_frames=args.num_frames)
    df = evaluator.score_directory(video_dir, prompts=prompts, glob=args.glob)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nScored {len(df)} videos → {out_path}")
    print(df[["video", "clip_score", "lpips_temporal", "motion_smoothness", "fvd_score", "composite"]].to_string(index=False))


if __name__ == "__main__":
    main()
