#!/usr/bin/env python3
"""Generate the confirmatory reward-study report and artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(metric: dict) -> str:
    estimate = 100 * metric["estimate"]
    lo, hi = (100 * x for x in metric["ci95"])
    return f"{estimate:.2f}% [{lo:.2f}%, {hi:.2f}%]"


def number(metric: dict) -> str:
    lo, hi = metric["ci95"]
    return f"{metric['estimate']:.3f} [{lo:.3f}, {hi:.3f}]"


def update_readme(path: Path, block: str) -> None:
    begin = "<!-- BEGIN GENERATED CONFIRMATORY RESULTS -->"
    end = "<!-- END GENERATED CONFIRMATORY RESULTS -->"
    text = path.read_text(encoding="utf-8")
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit("README generated-result markers are missing or duplicated")
    prefix, remainder = text.split(begin)
    _, suffix = remainder.split(end)
    path.write_text(f"{prefix}{begin}\n{block.rstrip()}\n{end}{suffix}", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--fit-metrics", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    fits = json.loads(args.fit_metrics.read_text())
    raw_rows = sum(1 for line in args.scores.open(encoding="utf-8") if line.strip())
    if raw_rows != 16200:
        raise SystemExit(f"expected 16,200 scored pairs, found {raw_rows}")
    primary = analysis["metrics"]["composite"]
    families = analysis["by_degradation_family"]
    checkpoints = sorted(args.checkpoint)
    if len(checkpoints) != 3 or any(not path.is_file() for path in checkpoints):
        raise SystemExit("exactly three reward checkpoints are required")

    lines = [
        "# Confirmatory report: controlled-intervention video rewards",
        "",
        "This report is generated from raw scored pairs by "
        "`scripts/publish_confirmatory_report.py`; displayed values are not hand edited.",
        "",
        "## Design and evidence boundary",
        "",
        f"The study scored {raw_rows:,} deterministic clean/degraded pairs across six "
        "degradation families and the three official UCF-101 splits. All final test "
        "sources are disjoint from fitting sources. Ground truth is intervention "
        "identity, not human aesthetic preference.",
        "",
        "The primary robustness test fits on five degradation families and evaluates "
        "the sixth. Reported intervals use 2,000 deterministic hierarchical-bootstrap "
        "replicates over `N_trials=3` official splits and pairs.",
        "",
        "## Primary result: unseen degradation-family transfer",
        "",
        f"The five-feature composite achieved **{pct(primary['accuracy'])}** accuracy "
        f"and ROC-AUC **{number(primary['roc_auc'])}** on {primary['n_pairs']:,} held-out "
        f"pairs (`N_trials={primary['n_trials']}`). Severity association was Spearman "
        f"**{number(primary['severity_spearman_rho'])}**.",
        "",
        "This aggregate is not evidence of uniform robustness. The family-level results "
        "show a sharp mechanical failure on unseen temporal corruptions:",
        "",
        "| Held-out family | Accuracy (95% CI) | Severity rho (95% CI) | Pairs |",
        "|---|---:|---:|---:|",
    ]
    for family, result in sorted(families.items()):
        lines.append(
            f"| `{family}` | {pct(result['accuracy'])} | "
            f"{number(result['severity_spearman_rho'])} | {result['n_pairs']:,} |"
        )
    lines += [
        "",
        "Brightness, downsampling, blur, and H.264 artifacts change direct spatial or "
        "luminance diagnostics in a stable direction. Frame dropping and local temporal "
        "permutation can instead reduce measured frame-to-frame change, so a model trained "
        "without that family assigns the wrong sign. This explains the near-zero temporal "
        "holdout accuracy and is a retained negative result, not an excluded slice.",
        "",
        "## Fixed-metric comparison",
        "",
        "| Metric | Accuracy (95% CI) | ROC-AUC (95% CI) | Severity rho (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for name, result in sorted(analysis["metrics"].items()):
        lines.append(
            f"| `{name}` | {pct(result['accuracy'])} | {number(result['roc_auc'])} | "
            f"{number(result['severity_spearman_rho'])} |"
        )
    lines += [
        "",
        "Paired composite-versus-component comparisons and Holm-adjusted p-values are "
        "stored in `analysis/metrics.json`. They show that the learned composite does not "
        "dominate simple components under family shift.",
        "",
        "## Operational in-family models",
        "",
        "These split-specific models fit all six intervention families on official "
        "training sources. They are intended only for the downstream augmentation study, "
        "whose candidates are covered by the training interventions; they are not evidence "
        "of open-ended perceptual generalization.",
        "",
        "| Split | Accuracy (source-clustered 95% CI) | ROC-AUC (95% CI) | Train/test pairs |",
        "|---:|---:|---:|---:|",
    ]
    for result in fits["models"]:
        lines.append(
            f"| {result['official_split']} | {pct(result['test_accuracy'])} | "
            f"{number(result['test_roc_auc'])} | {result['n_train_pairs']:,}/"
            f"{result['n_test_pairs']:,} |"
        )
    lines += [
        "",
        "## Deviations and limitations",
        "",
        "- The frozen plan mentioned a calibration subset, calibration error, nonzero-"
        "severity pair ranking, and class/resolution/motion slices. The frozen pair builder "
        "did not allocate those samples or fields. They are therefore **not reported**; no "
        "post-outcome amendment or surrogate analysis was used.",
        "- This benchmark tests deterministic synthetic corruptions of UCF-101, not human "
        "preference, generative-video quality, or cross-dataset validity.",
        "- Perfect results on several held-out families reflect easily detected controlled "
        "interventions and must not be described as general video-quality accuracy.",
        "",
        "## Reproduction",
        "",
        "The artifact manifest beside this report records SHA-256 hashes and byte counts "
        "for the raw scores, analyses, preregistration, and model checkpoints.",
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    if args.readme:
        family_rows = []
        for family, family_result in sorted(families.items()):
            family_rows.append(
                f"| `{family}` | {pct(family_result['accuracy'])} | "
                f"{number(family_result['roc_auc'])} | "
                f"{number(family_result['severity_spearman_rho'])} | "
                f"{family_result['n_pairs']:,} |"
            )
        metric_rows = []
        for name, metric_result in sorted(analysis["metrics"].items()):
            metric_rows.append(
                f"| `{name}` | {pct(metric_result['accuracy'])} | "
                f"{number(metric_result['roc_auc'])} | "
                f"{number(metric_result['severity_spearman_rho'])} |"
            )
        fit_rows = []
        for fit in fits["models"]:
            fit_rows.append(
                f"| {fit['official_split']} | {pct(fit['test_accuracy'])} | "
                f"{number(fit['test_roc_auc'])} | {fit['n_train_pairs']:,}/"
                f"{fit['n_test_pairs']:,} |"
            )
        comparison_rows = []
        for comparison in analysis["paired_comparisons"]:
            lo, hi = comparison["ci95"]
            p_holm = comparison["p_holm"]
            rendered_p = (
                f"<{1 / comparison['bootstrap_replicates']:.4g}"
                if p_holm == 0 else f"{p_holm:.4g}"
            )
            comparison_rows.append(
                f"| `{comparison['comparison']}` | "
                f"{100 * comparison['estimate']:.3f} pp "
                f"[{100 * lo:.3f}, {100 * hi:.3f}] | {rendered_p} | "
                f"{comparison['n_pairs']:,} |"
            )
        readme_block = f"""## Confirmatory results

The study scored **{raw_rows:,} deterministic clean/degraded pairs** across six
families and **`N_trials=3` official UCF-101 splits**, using **2,000 bootstrap
replicates**. These labels identify controlled interventions, not human
preference.

### Primary leave-one-degradation-family-out result

- Composite accuracy: **{pct(primary['accuracy'])}** on **{primary['n_pairs']:,} held-out pairs**.
- Composite ROC-AUC: **{number(primary['roc_auc'])}**.
- Severity association: Spearman rho **{number(primary['severity_spearman_rho'])}**.

| Held-out family | Accuracy (95% CI) | ROC-AUC (95% CI) | Severity rho (95% CI) | Pairs |
|---|---:|---:|---:|---:|
{chr(10).join(family_rows)}

The composite generalized almost perfectly to the four spatial/luminance or
codec families but reversed direction on unseen frame dropping and temporal
permutation. This temporal failure is part of the primary result.

### Fixed metric results under family shift

| Metric | Accuracy (95% CI) | ROC-AUC (95% CI) | Severity rho (95% CI) |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

### Paired composite-versus-component accuracy differences

| Comparison | Difference (95% CI) | Holm-adjusted p | Pairs |
|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

### Operational all-family models for the downstream study

| Official split | Accuracy (source-clustered 95% CI) | ROC-AUC (95% CI) | Train/test pairs |
|---:|---:|---:|---:|
{chr(10).join(fit_rows)}

These operational scores apply only to source-disjoint pairs from the six
trained intervention families. They do not establish human-preference or
open-ended perceptual validity.
"""
        update_readme(args.readme, readme_block)
    inputs = [args.scores, args.analysis, args.fit_metrics, args.preregistration,
              args.environment, *checkpoints]
    manifest = {
        "schema_version": 1,
        "study_id": analysis["study_id"],
        "generated_by": "scripts/publish_confirmatory_report.py",
        "raw_scored_pairs": raw_rows,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "report": {"path": str(report), "bytes": report.stat().st_size,
                   "sha256": sha256(report)},
    }
    if args.readme:
        manifest["readme"] = {"path": str(args.readme), "bytes": args.readme.stat().st_size,
                              "sha256": sha256(args.readme)}
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
