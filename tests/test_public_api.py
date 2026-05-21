"""Pin the v0.1.0 public surface. Anything not in this set is internal."""

from __future__ import annotations


def test_public_surface():
    import circuitry

    expected = {
        "HookPoint",
        "MetricWriter",
        "Recipe",
        "Recorder",
        "StepContext",
        "TensorSource",
        "__version__",
        "build_report",
        "register_recipe",
        "scan_run",
    }
    assert set(circuitry.__all__) == expected
    for name in expected:
        assert hasattr(circuitry, name), f"circuitry.{name} not re-exported"


def test_version_is_a_string():
    import circuitry
    assert isinstance(circuitry.__version__, str)
    assert circuitry.__version__
