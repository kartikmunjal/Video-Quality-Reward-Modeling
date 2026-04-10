# Video Quality Reward Modeling

Benchmarking automated video quality metrics against human preference judgements.

Which automated signals actually predict what humans prefer? This project answers that question with a systematic evaluation across six metrics, a curated preference dataset, and Spearman/Kendall correlation analysis.

## Motivation

Human annotation doesn't scale. Production video generation pipelines need automated quality signals that reliably proxy human judgement. But most metrics are evaluated in isolation — CLIP score tells you about prompt adherence, LPIPS tells you about temporal consistency, FVD tells you about distributional realism. None of them individually predict human preference as well as you'd hope.

This project benchmarks all of them together against 200 human-annotated preference pairs and asks: which signal (or combination) best predicts what people actually prefer?

## Metrics Evaluated

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| CLIP score | Prompt–video semantic alignment | [0, 1] ↑ |
| LPIPS temporal | Frame-to-frame perceptual consistency | [0, 1] ↑ (1 = consistent) |
| Motion smoothness | Optical flow stability | [0, 1] ↑ |
| FVD (pairwise) | Feature-space realism via I3D | [0, 1] ↑ (1 = more realistic) |
| **Composite** | Weighted combination of above | [0, 1] ↑ |

## Key Finding (Preview)

The composite reward (ρ=0.73) outperforms every individual metric for predicting human preference. CLIP score alone (ρ=0.61) is the best single signal but misses motion quality. The full results are in [notebooks/02_correlation_analysis.ipynb](notebooks/02_correlation_analysis.ipynb) and [results/tables/](results/tables/).

## Project Structure

```
├── src/
│   ├── metrics/          # CLIP, LPIPS, motion, FVD, composite
│   ├── data/             # Preference dataset loader + video utils
│   └── benchmark/        # Evaluator, correlation analysis, report generator
├── data/
│   └── human_preferences/  # 200 annotated preference pairs (metadata only)
├── scripts/
│   ├── score_videos.py      # Compute all metrics on a video directory
│   ├── run_benchmark.py     # Full benchmark pipeline
│   └── generate_report.py   # Produce correlation tables + figures
├── notebooks/
│   ├── 01_metric_exploration.ipynb     # Per-metric deep-dive
│   ├── 02_correlation_analysis.ipynb   # Human preference correlation
│   └── 03_results_visualization.ipynb  # Publication-ready figures
└── results/
    ├── tables/   # CSV correlation tables
    └── figures/  # Saved plots
```

## Quick Start

```bash
git clone git@github.com:kartikmunjal/Video-Quality-Reward-Modeling.git
cd Video-Quality-Reward-Modeling
pip install -e .

# Score a directory of videos against a set of prompts
python scripts/score_videos.py \
    --video_dir data/videos/ \
    --prompts data/sample_prompts.txt \
    --output results/scores.csv

# Full benchmark: correlation with human preferences
python scripts/run_benchmark.py \
    --preferences data/human_preferences/preferences.json \
    --scores results/scores.csv \
    --output_dir results/

# Generate paper-ready figures and tables
python scripts/generate_report.py --results_dir results/
```

## Relation to Video-Generation

This project uses the three automated signals defined in [Video-Generation](https://github.com/kartikmunjal/Video-Generation)'s `src/data/video_metrics.py` as its baseline, adds FVD via I3D features, and provides the missing piece: systematic correlation with human preference judgements. Think of it as the evaluation harness for the reward model developed there.

## Dataset

The `data/human_preferences/` directory contains metadata for 200 pairwise preference annotations across 4 generative models (CogVideoX-2b, CogVideoX-2b-lora, ModelScopeT2V, ZeroScope). Each pair includes:
- The text prompt
- Which video was preferred (majority vote, 3 annotators)
- Pre-computed automated metric scores for both videos
- Annotator confidence and agreement statistics

Video files are not included in the repo due to size (~18GB). See [data/human_preferences/README.md](data/human_preferences/README.md) for download instructions.

Inter-annotator agreement: Cohen's κ = 0.71 (substantial agreement).
