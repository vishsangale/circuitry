"""Bridge a loaded HuggingFace model into TransformerLens for the TL patching backend.

The HF-eager patching backend targets Llama-family layouts. For GPT-2 and the other
architectures TransformerLens supports, wrap the loaded HF model as a
``HookedTransformer`` and use it with ``TLSiteResolver``.
"""

from __future__ import annotations

from typing import Any


def to_hooked_transformer(
    hf_model: Any,
    model_name: str,
    *,
    device: str | None = None,
    dtype: Any = None,
    **tl_kwargs: Any,
):
    """Wrap a loaded HF causal-LM as a TransformerLens ``HookedTransformer``.

    Args:
        hf_model: an already-loaded HF ``*ForCausalLM`` model (its weights are reused).
        model_name: the TransformerLens architecture name, e.g. ``"gpt2"``.
        device / dtype: forwarded to ``from_pretrained``.
        **tl_kwargs: forwarded to ``HookedTransformer.from_pretrained`` (e.g.
            ``fold_ln=False``). Defaults apply TL's standard processing.

    Returns:
        A ``HookedTransformer`` usable with ``TLSiteResolver`` and the patching runners.

    Note:
        TransformerLens folds LayerNorm and centers writing/unembed weights, so the
        wrapped model's *activations* differ from the raw HF model's (logits are
        equivalent). Patching runs on the TL-processed model.

    Raises:
        ImportError: if ``transformer_lens`` is not installed.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_hooked_transformer requires transformer_lens. "
            "Install it with: pip install transformer_lens"
        ) from e

    return HookedTransformer.from_pretrained(
        model_name, hf_model=hf_model, device=device, dtype=dtype, **tl_kwargs
    )
