"""Thin wrapper over sae_lens.SAE.from_pretrained. See v0.9 spec §4.3."""

from __future__ import annotations


def load_sae(release: str, sae_id: str, device: str = "cpu"):
    """Load a SAELens-format SAE from the HF Hub and move it to `device`.

    Returns the raw `sae_lens.SAE` object so users can also call SAELens
    utilities directly. Some SAELens versions return a tuple
    `(sae, cfg, sparsity)` from `from_pretrained`; we unwrap the first
    element transparently.
    """
    try:
        import sae_lens
    except ImportError as _e:
        raise ImportError(
            "circuitry: SAE features require the 'sae-lens' package, which is "
            "an optional extra. Install it with `pip install \"circuitry[sae]\"`."
        ) from _e
    result = sae_lens.SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device
    )
    if isinstance(result, tuple):
        return result[0]
    return result
