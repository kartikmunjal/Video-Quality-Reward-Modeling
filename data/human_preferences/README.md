# Human Preference Dataset

200 pairwise preference annotations across 4 text-to-video models.

## Schema (`preferences.json`)

```json
{
  "metadata": { ... },
  "pairs": [
    {
      "id": "pair_001",
      "prompt": "...",
      "video_a": {"model": "...", "path": "videos/pair_001_a.mp4"},
      "video_b": {"model": "...", "path": "videos/pair_001_b.mp4"},
      "annotations": [
        {"annotator": 1, "choice": "a", "confidence": "high"},
        ...
      ],
      "majority_choice": "a",
      "agreement": 1.0,
      "automated_scores": {
        "video_a": {"clip_score": ..., "lpips_temporal": ..., "motion_smoothness": ..., "fvd_score": ..., "composite": ...},
        "video_b": { ... }
      }
    }
  ]
}
```

## Models Compared

| Model | Short name |
|-------|-----------|
| CogVideoX-2b (base) | `cogvideox_base` |
| CogVideoX-2b + LoRA fine-tune | `cogvideox_lora` |
| ModelScopeT2V | `modelscope` |
| ZeroScope v2 | `zeroscope` |

Each pair compares two different models on the same prompt.

## Annotation Protocol

- 3 annotators per pair, recruited via Prolific
- Task: "Which video better matches the text prompt and looks more natural?"
- Confidence: high / medium / low (self-reported)
- Majority vote (2-of-3) determines `majority_choice`
- Ties (1-1-1 for A/tie/B) are rare (6 of 200 pairs) and excluded from correlation analysis
- Inter-annotator agreement: Cohen's κ = 0.71

## Video Files

Videos are not stored in this repo (~18 GB). To reproduce:

1. Download CogVideoX-2b from HuggingFace: `THUDM/CogVideoX-2b`
2. Generate videos using prompts in `../sample_prompts.txt`
3. Place outputs at `videos/pair_{NNN}_{a,b}.mp4` (16 frames, 480×720)

Or use the pre-generated set (link TBD after paper submission).
