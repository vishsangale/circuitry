"""Thin wrapper over sae_lens.SAE.from_pretrained. See v0.9 spec §4.3."""

from __future__ import annotations

from typing import Any


def load_sae(release: str, sae_id: str, device: str = "cpu") -> Any:
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


def load_gemma_scope(
    model_size: str,
    layer: int,
    width: int,
    *,
    site: str = "res",
    average_l0: int | None = None,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> Any:
    """Load a pre-trained Gemma Scope JumpReLU SAE.

    Convenience wrapper around load_sae() for Gemma Scope weight suites
    (Lieberum et al. 2024, https://arxiv.org/abs/2408.05147).

    Args:
        model_size: Gemma 2 model size — "2b", "9b", or "27b".
        layer:      transformer layer index (0-based).
        width:      SAE width in thousands of features (e.g. 16 for 16k).
        site:       activation site — "res" (residual post), "mlp", or "att".
        average_l0: target average L0 sparsity; if None uses the first
                    listed checkpoint for that width (implementation detail
                    — pass through to sae_lens, which picks a default).
        device:     device string for load_sae().
        cache_dir:  optional local cache directory.

    Returns:
        SAE object (same type as load_sae()).

    Example::
        sae = load_gemma_scope("2b", layer=12, width=16)
    """
    release = f"gemma-scope-{model_size}-pt-{site}"
    if average_l0 is not None:
        sae_id = f"layer_{layer}/width_{width}k/average_l0_{average_l0}"
    else:
        # 71 is a common default L0 across Gemma Scope checkpoints; callers
        # who need a specific sparsity should pass average_l0 explicitly.
        sae_id = f"layer_{layer}/width_{width}k/average_l0_71"
    return load_sae(release, sae_id, device=device)


def load_llama_scope(
    layer: int,
    width: int,
    *,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> Any:
    """Load a pre-trained Llama Scope JumpReLU SAE for Llama 3.1 8B.

    Convenience wrapper around load_sae() for Llama Scope weight suites
    (Tu et al. 2024).

    Args:
        layer:     transformer layer index (0-based).
        width:     SAE width in thousands of features (e.g. 16 for 16k).
        device:    device string for load_sae().
        cache_dir: optional local cache directory.

    Returns:
        SAE object (same type as load_sae()).
    """
    release = "llama_scope_lx_bf_8b"
    sae_id = f"l{layer}/width_{width}k"
    return load_sae(release, sae_id, device=device)
