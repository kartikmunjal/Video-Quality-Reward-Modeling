from scripts.build_intervention_pairs import order_key


def test_pair_order_is_deterministic_and_family_specific():
    first = order_key("study", 1, "blur", "a.avi")
    assert first == order_key("study", 1, "blur", "a.avi")
    assert first != order_key("study", 1, "brightness", "a.avi")
