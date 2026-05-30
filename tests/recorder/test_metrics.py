from circuitry.recorder._metrics import group, stats


def test_group_sorts_by_step():
    rows = [
        {"tag": "a/b", "value": 2.0, "step": 1, "kind": "scalar"},
        {"tag": "a/b", "value": 1.0, "step": 0, "kind": "scalar"},
    ]
    g = group(rows)
    assert g["a/b"] == [(0, 1.0), (1, 2.0)]


def test_stats_single_point():
    first, last, vmin, vmax, delta = stats([(0, 5.0)])
    assert first == last == vmin == vmax == 5.0
    assert delta == 0.0
