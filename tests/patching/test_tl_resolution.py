"""Tests for TLSiteResolver + lazy transformer_lens import."""
from __future__ import annotations

import subprocess
import sys

import pytest

from circuitry.patching.sites import Site, TLSiteResolver


def test_tl_hook_name_resid_pre():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="resid_pre", layer=3)) == "blocks.3.hook_resid_pre"


def test_tl_hook_name_resid_post():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="resid_post", layer=5)) == "blocks.5.hook_resid_post"


def test_tl_hook_name_attn_head_out():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="attn_head_out", layer=2, head=3)) == "blocks.2.attn.hook_z"


def test_tl_hook_name_mlp_out():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="mlp_out", layer=1)) == "blocks.1.mlp.hook_post"


def test_tl_hook_name_mlp_neuron():
    r = TLSiteResolver()
    assert r.hook_name(Site(component="mlp_neuron", layer=0, neuron=42)) == "blocks.0.mlp.hook_post"


def test_lazy_import_does_not_import_transformer_lens():
    """Importing circuitry (incl. patching) must NOT import transformer_lens.

    Run in a fresh subprocess so the invariant is verified deterministically
    regardless of whether another test already imported transformer_lens into
    this process — and regardless of whether transformer_lens is installed.
    """
    code = (
        "import circuitry, circuitry.patching, circuitry.patching.sites, sys; "
        "assert 'transformer_lens' not in sys.modules, 'transformer_lens leaked'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_tl_resolver_resolve_requires_transformer_lens():
    r = TLSiteResolver()
    site = Site(component="resid_post", layer=0)

    class FakeModel:
        pass

    try:
        import transformer_lens  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="transformer_lens"):
            r.resolve(FakeModel(), site)  # type: ignore[arg-type]
