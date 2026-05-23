"""Tests for circuitry.sae.load_sae. Spec §4.3."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from circuitry.sae import load_sae


@patch("circuitry.sae.loader.sae_lens")
def test_load_sae_forwards_args_and_returns_sae(mock_sae_lens):
    mock_sae = MagicMock()
    mock_sae_lens.SAE.from_pretrained.return_value = (mock_sae, {}, None)
    out = load_sae("release-x", "id-y", device="cpu")
    mock_sae_lens.SAE.from_pretrained.assert_called_once_with(
        release="release-x", sae_id="id-y", device="cpu"
    )
    assert out is mock_sae


@patch("circuitry.sae.loader.sae_lens")
def test_load_sae_handles_non_tuple_return(mock_sae_lens):
    """Some SAELens versions return the SAE directly, not a tuple."""
    mock_sae = MagicMock()
    mock_sae_lens.SAE.from_pretrained.return_value = mock_sae
    out = load_sae("r", "i")
    assert out is mock_sae
