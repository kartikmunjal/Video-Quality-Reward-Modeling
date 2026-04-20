# Video Quality Reward Modeling

Benchmarking automated video quality metrics against human preference judgements.

Which automated signals actually predict what humans prefer? This project answers
that question with a systematic evaluation across five metrics, a 200-pair
human-annotated preference dataset, and Spearman/Kendall correlation analysis.

## Motivation

Human annotation doesn't scale. Production video generation pipelines need automated
quality signals that reliably proxy human judgement. Most metrics are evaluated in
isolation — CLIP score tells you about prompt adherence, LPIPS tells you about
temporal consistency, FVD tells you about distributional realism. None of them
individually predicts human preference as well as you'd hope.

This project benchmarks all of them together against 200 human-annotated preference
pairs (3 annotators each, Cohen's κ = 0.71) and asks: which signal — or combination
— best predicts what people actually prefer?

## Results

**188 non-tie pairs, 4 models (CogVideoX-2b base/LoRA, ZeroScope, ModelScopeT2V).**

| Metric | Spearman ρ | Pairwise Acc. | ROC-AUC |
|--------|:----------:|:-------------:|:-------:|
| **Composite** | **0.858** | **96.3%** | **0.996** |
| CLIP Score | 0.854 | 95.7% | 0.994 |
| Motion Smoothness | 0.737 | 84.0% | 0.926 |
| LPIPS Temporal | 0.735 | 82.5% | 0.925 |
| FVD (pairwise) | 0.554 | 73.9% | 0.821 |

**Key takeaways:**
- The composite reward (ρ=0.858, acc=96.3%) is the strongest single predictor.
- CLIP score alone (ρ=0.854) is nearly as good for semantic alignment but misses motion quality.
- Motion smoothness and LPIPS carry distinct, complementary signal; together they push composite above CLIP.
- FVD as a pairwise metric is noisier than the other signals — useful for distributional eval but not pairwise ranking.
- Composite wins most clearly on pairs where motion quality and prompt adherence trade off against each other.

Full analysis: [notebooks/02_correlation_analysis.ipynb](notebooks/02_correlation_analysis.ipynb)

## Research Lineage

This repo is the evaluation layer for the video preference-alignment stack:

| Stage | Repo | Role |
|-------|------|------|
| Data | [`Video-Curation`](https://github.com/kartikmunjal/Video-Curation) | Produces curated and synthetic video mixtures for downstream experiments. |
| Training | [`Video-Generation`](https://github.com/kartikmunjal/Video-Generation) | Trains LoRA and DiffusionDPO models whose outputs are scored here. |
| Reward validation | `Video-Quality-Reward-Modeling` | Tests whether automated quality signals actually agree with human preferences. |

The reward-design framing follows
[`rlhf-and-reward-modelling-alt`](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt):
define candidate reward signals, validate them against held-out preference data,
then only report training improvements whose metrics have human-correlation support.

## Composite Reward Weights

```
Composite = 0.45 × CLIP + 0.25 × LPIPS_temporal + 0.20 × motion_smoothness + 0.10 × FVD
```

Weights were chosen based on the ablation study in the companion
[Video-Generation](https://github.com/kartikmunjal/Video-Generation) repo
(reward training ablation, Table 3). CLIP carries the largest weight because
prompt adherence is the primary failure mode in text-to-video generation.

## Metrics

| Metric | What It Measures | Implementation |
|--------|-----------------|---------------|
| CLIP Score | Prompt–video semantic alignment | CLIP ViT-B/32, avg over 4 frames |
| LPIPS Temporal | Frame-to-frame perceptual consistency | LPIPS AlexNet, consecutive frames |
| Motion Smoothness | Optical flow stability | Farneback dense flow, CV of magnitude |
| FVD (pairwise) | Feature-space realism | R3D-18 I3D features, cosine similarity |
| **Composite** | Weighted combination | Weighted sum, all normalized to [0, 1] |

## Project Structure

```
├── src/
│   ├── metrics/          # CLIP, LPIPS, motion, FVD, composite implementations
│   ├── data/             # Preference dataset loader + video utils
│   └── benchmark/        # Evaluator, correlation analysis, report generator
├── data/
│   └── human_preferences/  # 200 annotated preference pairs + sample prompts
├── scripts/
│   ├── score_videos.py      # Compute all metrics on a video directory
│   ├── run_benchmark.py     # Full benchmark pipeline
│   └── generate_report.py   # Regenerate figures from saved results
├── notebooks/
│   ├── 01_metric_exploration.ipynb     # Score distributions, inter-metric correlations
│   ├── 02_correlation_analysis.ipynb   # Human preference correlation (main result)
│   └── 03_results_visualization.ipynb  # Publication-ready figures
└── results/
    ├── tables/   # correlation_table.csv, model_metric_means.csv
    └── figures/  # Saved plots
```

## Quick Start

```bash
git clone git@github.com:kartikmunjal/Video-Quality-Reward-Modeling.git
cd Video-Quality-Reward-Modeling
pip install -e .

# Reproduce the benchmark from pre-computed scores in preferences.json
# (no video files needed — scores are embedded in the dataset)
PYTHONPATH=src python scripts/run_benchmark.py \
    --preferences data/human_preferences/preferences.json \
    --output_dir results/

# Score your own videos
python scripts/score_videos.py \
    --video_dir path/to/videos/ \
    --prompts data/sample_prompts.txt \
    --output results/scores.csv

# Regenerate all figures
python scripts/generate_report.py --results_dir results/
```

## Dataset

200 pairwise preference pairs across 4 text-to-video models, 3 annotators per pair.

- **Prompts:** 70 diverse natural-language descriptions (EvalCrafter + custom)
- **Models:** CogVideoX-2b (base), CogVideoX-2b + LoRA fine-tune, ZeroScope v2, ModelScopeT2V
- **Annotation:** Majority vote (2-of-3), confidence self-reported
- **Agreement:** Cohen's κ = 0.71 (substantial), raw agreement 83%
- **Ties:** 12 pairs (6%) excluded from correlation analysis

Each pair includes pre-computed metric scores for both videos, so the full
correlation analysis runs without video files. Video files (~18 GB) are
available separately — see [data/human_preferences/README.md](data/human_preferences/README.md).

## Relation to Video-Generation

This project benchmarks the automated signals first defined in
[Video-Generation](https://github.com/kartikmunjal/Video-Generation)'s
`src/data/video_metrics.py`, adds FVD via I3D features, and provides
the missing validation layer: systematic Spearman correlation with human
preference data. It is the evaluation harness for the reward model
trained there.

## Requirements

```
torch>=2.1.0  torchvision>=0.16.0  transformers>=4.40.0
decord>=0.6.0  opencv-python>=4.9.0  lpips>=0.1.4  einops>=0.7.0
numpy>=1.26.0  pandas>=2.1.0  scipy>=1.12.0  scikit-learn>=1.3.0
matplotlib>=3.8.0  seaborn>=0.13.0  tqdm>=4.66.0
```

See [requirements.txt](requirements.txt) for the full pinned list.
