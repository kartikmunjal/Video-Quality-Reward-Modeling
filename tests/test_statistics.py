import pandas as pd
import pytest

from benchmark.statistics import paired_cluster_bootstrap


def test_paired_cluster_bootstrap_is_deterministic_and_reports_trials():
    frame = pd.DataFrame({
        "official_split": [1, 1, 2, 2, 3, 3],
        "guided": [1, 1, 1, 0, 1, 1],
        "random": [0, 1, 0, 0, 1, 0],
    })
    first = paired_cluster_bootstrap(frame, "guided", "random", replicates=100)
    second = paired_cluster_bootstrap(frame, "guided", "random", replicates=100)
    assert first == second
    assert first["n_trials"] == 3
    assert first["n_observations"] == 6


def test_bootstrap_rejects_single_trial():
    frame = pd.DataFrame({"official_split": [1], "a": [1.0], "b": [0.0]})
    with pytest.raises(ValueError, match="two independent"):
        paired_cluster_bootstrap(frame, "a", "b")
