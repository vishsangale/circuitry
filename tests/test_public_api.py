"""Pin the v0.1.0 public surface. Anything not in this set is internal."""

from __future__ import annotations


def test_public_surface():
    import circuitry

    expected = {
        "HookPoint",
        "MetricWriter",
        "ModelInventory",
        "ParameterRecord",
        "Recipe",
        "Recorder",
        "StepContext",
        "TensorSource",
        "__version__",
        "build_report",
        "direction_cosine",
        "discover",
        "register_recipe",
        "scan_run",
        "token_similarity",
        "update_delta",
    }
    assert set(circuitry.__all__) == expected
    for name in expected:
        assert hasattr(circuitry, name), f"circuitry.{name} not re-exported"


def test_version_is_a_string():
    import circuitry
    assert isinstance(circuitry.__version__, str)
    assert circuitry.__version__


def test_v02_surface_exports():
    import circuitry
    assert circuitry.__version__ == "0.6.0"
    assert hasattr(circuitry, "token_similarity")
    assert hasattr(circuitry, "update_delta")
    assert hasattr(circuitry, "direction_cosine")
    assert hasattr(circuitry, "discover")
