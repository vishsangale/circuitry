"""Tests for load_gemma_scope and load_llama_scope convenience loaders.

All tests mock load_sae so they run fully offline without hitting HuggingFace.
"""
from __future__ import annotations

import unittest.mock as mock

from circuitry.sae.loader import load_gemma_scope, load_llama_scope


def test_load_gemma_scope_builds_correct_release_id():
    """load_gemma_scope("2b", layer=12, width=16) must call load_sae with
    release='gemma-scope-2b-pt-res' and sae_id containing 'layer_12/width_16k'.
    """
    with mock.patch("circuitry.sae.loader.load_sae") as m:
        m.return_value = object()
        load_gemma_scope("2b", layer=12, width=16)
        call_args = m.call_args
        release_arg = call_args[0][0]
        sae_id_arg = call_args[0][1]
        assert "gemma-scope-2b-pt-res" in release_arg, (
            f"Expected release 'gemma-scope-2b-pt-res', got {release_arg!r}"
        )
        assert "layer_12" in sae_id_arg, f"Expected 'layer_12' in sae_id, got {sae_id_arg!r}"
        assert "width_16k" in sae_id_arg, f"Expected 'width_16k' in sae_id, got {sae_id_arg!r}"


def test_load_gemma_scope_site_mlp():
    """site='mlp' must produce release 'gemma-scope-9b-pt-mlp'."""
    with mock.patch("circuitry.sae.loader.load_sae") as m:
        m.return_value = object()
        load_gemma_scope("9b", layer=4, width=32, site="mlp")
        release_arg = m.call_args[0][0]
        assert "gemma-scope-9b-pt-mlp" in release_arg, (
            f"Expected 'gemma-scope-9b-pt-mlp' in release, got {release_arg!r}"
        )


def test_load_gemma_scope_average_l0_explicit():
    """Explicit average_l0 must appear in the sae_id."""
    with mock.patch("circuitry.sae.loader.load_sae") as m:
        m.return_value = object()
        load_gemma_scope("2b", layer=0, width=16, average_l0=45)
        sae_id_arg = m.call_args[0][1]
        assert "average_l0_45" in sae_id_arg, (
            f"Expected 'average_l0_45' in sae_id, got {sae_id_arg!r}"
        )


def test_load_llama_scope_builds_correct_release_id():
    """load_llama_scope(layer=8, width=16) must produce a llama_scope release
    and sae_id containing 'l8' and 'width_16k'.
    """
    with mock.patch("circuitry.sae.loader.load_sae") as m:
        m.return_value = object()
        load_llama_scope(layer=8, width=16)
        call_args = m.call_args
        release_arg = call_args[0][0]
        sae_id_arg = call_args[0][1]
        assert "llama_scope" in release_arg.lower(), (
            f"Expected 'llama_scope' in release, got {release_arg!r}"
        )
        assert "l8" in sae_id_arg, f"Expected 'l8' in sae_id, got {sae_id_arg!r}"
        assert "width_16k" in sae_id_arg, f"Expected 'width_16k' in sae_id, got {sae_id_arg!r}"


def test_load_gemma_scope_passes_device():
    """device kwarg must be forwarded to load_sae."""
    with mock.patch("circuitry.sae.loader.load_sae") as m:
        m.return_value = object()
        load_gemma_scope("2b", layer=0, width=16, device="cuda")
        # load_sae is called as load_sae(release, sae_id, device=device)
        kwargs = m.call_args[1]
        assert kwargs.get("device") == "cuda", (
            f"Expected device='cuda' kwarg, got kwargs={kwargs}"
        )


def test_load_llama_scope_passes_device():
    """device kwarg must be forwarded to load_sae for llama_scope."""
    with mock.patch("circuitry.sae.loader.load_sae") as m:
        m.return_value = object()
        load_llama_scope(layer=3, width=8, device="cuda")
        kwargs = m.call_args[1]
        assert kwargs.get("device") == "cuda", (
            f"Expected device='cuda' kwarg, got kwargs={kwargs}"
        )
