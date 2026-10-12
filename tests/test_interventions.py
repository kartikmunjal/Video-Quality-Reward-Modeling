import numpy as np

from scripts.score_intervention_pairs import degrade, metrics


def _frames():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)


def test_blur_reduces_spatial_detail():
    clean = _frames()
    blurred = degrade(clean, 8.0, "gaussian_blur_sigma", 3.0, seed=1)
    assert metrics(blurred)["spatial_detail_log"] < metrics(clean)["spatial_detail_log"]


def test_temporal_permutation_is_seeded():
    clean = _frames()
    first = degrade(clean, 8.0, "temporal_permutation_window", 4, seed=9)
    second = degrade(clean, 8.0, "temporal_permutation_window", 4, seed=9)
    assert np.array_equal(first, second)
