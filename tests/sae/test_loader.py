"""Tests for circuitry.sae.load_sae. Spec §4.3."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from circuitry.sae import load_sae


# load_sae imports sae_lens lazily (optional extra), so patch the real
# sae_lens.SAE the function reaches rather than a loader module attribute.
@patch("sae_lens.SAE")
def test_load_sae_forwards_args_and_returns_sae(mock_SAE):
    mock_sae = MagicMock()
    mock_SAE.from_pretrained.return_value = (mock_sae, {}, None)
    out = load_sae("release-x", "id-y", device="cpu")
    mock_SAE.from_pretrained.assert_called_once_with(
        release="release-x", sae_id="id-y", device="cpu"
    )
    assert out is mock_sae


@patch("sae_lens.SAE")
def test_load_sae_handles_non_tuple_return(mock_SAE):
    """Some SAELens versions return the SAE directly, not a tuple."""
    mock_sae = MagicMock()
    mock_SAE.from_pretrained.return_value = mock_sae
    out = load_sae("r", "i")
    assert out is mock_sae
