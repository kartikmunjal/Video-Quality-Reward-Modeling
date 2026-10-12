# Reproduction guide

This study consumes the parent curation study's hashed official assignments and
locally acquired UCF-101 videos. The executed environment is in
`results/confirmatory/environment.json`.

Run `PYTHONPATH=src pytest -q`, then `build_intervention_pairs.py`,
`score_intervention_pairs.py`, `analyze_interventions.py`, and
`fit_reward_models.py`. The executed Windows command is retained in
`scripts/run_confirmatory_scoring_norgate.cmd`.

Regenerate the report with `publish_confirmatory_report.py`, passing the raw
scores, analyses, frozen preregistration, environment, and three checkpoints.
It derives every displayed number and writes an SHA-256 manifest. The legacy
human-preference fixture is not a confirmatory input.
