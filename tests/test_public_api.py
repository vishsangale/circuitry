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
        "repr_drift",
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
    # The exact version is pinned by tests/test_version_consistency.py against the
    # single source of truth (circuitry.__version__); don't hard-code it here.
    assert hasattr(circuitry, "token_similarity")
    assert hasattr(circuitry, "update_delta")
    assert hasattr(circuitry, "direction_cosine")
    assert hasattr(circuitry, "discover")
    assert hasattr(circuitry, "repr_drift")


def test_sae_feature_runner_accessible():
    """SAEFeatureRunner lives under circuitry.patching (not top-level circuitry).

    Two checks:
      (a) Accessible via ``circuitry.patching.SAEFeatureRunner`` and in __all__.
      (b) Lazy-import guarantee: importing ``circuitry.patching`` does NOT pull in
          ``transformer_lens`` (verified in a clean subprocess so import state is fresh).
    """
    import subprocess
    import sys

    import circuitry.patching

    # (a) Accessible and in __all__
    from circuitry.patching import SAEFeatureRunner
    assert SAEFeatureRunner is not None
    assert hasattr(circuitry.patching, "SAEFeatureRunner"), (
        "SAEFeatureRunner not accessible as circuitry.patching.SAEFeatureRunner"
    )
    assert "SAEFeatureRunner" in circuitry.patching.__all__, (
        "SAEFeatureRunner missing from circuitry.patching.__all__"
    )

    # (b) Lazy-import: importing circuitry.patching must NOT eagerly load transformer_lens.
    # Run in a subprocess so sys.modules is completely clean.
    code = (
        "import sys; "
        "import circuitry.patching; "
        "assert 'transformer_lens' not in sys.modules, "
        "  f'transformer_lens was eagerly imported by circuitry.patching: {list(sys.modules)}'; "
        "runner = circuitry.patching.SAEFeatureRunner; "
        "assert runner is not None, 'SAEFeatureRunner is None after lazy import'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        # If transformer_lens isn't installed at all, skip gracefully
        if "ModuleNotFoundError" in stderr and "transformer_lens" in stderr:
            import pytest
            pytest.skip("transformer_lens not installed — skipping lazy-import check")
        raise AssertionError(
            f"Subprocess lazy-import check failed (returncode={result.returncode}):\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )
    assert result.stdout.strip() == "OK", (
        f"Unexpected subprocess output: {result.stdout!r}"
    )


def test_sae_feature_edge_types_accessible():
    """v1.6 feature-circuit types live under circuitry.patching (lazy-exported)."""
    import circuitry.patching

    for name in (
        "SAEFeatureEdge",
        "SAEFeatureEdgeGraph",
        "SAEFeatureCircuit",
        "SAEFeatureEdgeRunner",
        "FeatureACDCRunner",
    ):
        assert hasattr(circuitry.patching, name), (
            f"{name} not accessible as circuitry.patching.{name}"
        )
        assert name in circuitry.patching.__all__, (
            f"{name} missing from circuitry.patching.__all__"
        )
