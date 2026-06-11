"""Transformer circuit primitives — QK/OV weight-space analysis.

Implements the core building blocks from the Transformer Circuits framework
(Elhage et al. 2021, https://transformer-circuits.pub/2021/framework/index.html):

  ov_matrix        — W_V @ W_O: what does each head write to the residual stream?
  qk_matrix        — W_Q @ W_K.T: what does each head attend to?
  head_composition_score  — how strongly does head A's output feed into head B's Q/K/V?
  composition_scores      — batched composition score matrix for all head pairs
  top_logit_tokens        — which tokens does a residual-space direction promote?
  top_embedding_tokens    — which token embeddings are most aligned with a direction?

v1.42 adds weight-based (input-independent) transcoder analysis (Circuit Insights,
arXiv:2510.14936; "Transcoders Beat Sparse Autoencoders", arXiv:2501.18823):

  transcoder_virtual_weights — W_dec_up @ W_enc_down: global feature→feature connectivity
  top_virtual_connections   — strongest entries of a virtual-weight matrix
  feature_token_alignment   — per-feature top-k promoted logit tokens (decoder @ W_U)

All functions are pure; no forward passes, no hooks.  Inputs are weight matrices
extracted from the model (e.g. via `model.named_parameters()`).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def ov_matrix(W_V: Any, W_O: Any) -> Tensor:
    """Compute the OV circuit matrix W_V @ W_O per head.

    The OV circuit captures what each attention head writes to the residual
    stream: for any attended-to token with residual representation ``x``,
    the head's contribution is ``attn_weight * x @ W_OV``.

    Args:
        W_V: Value projection, shape ``(..., d_model, d_head)``.
        W_O: Output projection, shape ``(..., d_head, d_model)``.

    Returns:
        ``(..., d_model, d_model)`` — the OV matrix ``W_V @ W_O`` per head.

    Reference:
        Elhage et al. 2021 §3.1 "A Mathematical Framework for Transformer Circuits".
    """
    v = torch.as_tensor(W_V).detach().to(torch.float32)
    o = torch.as_tensor(W_O).detach().to(torch.float32)
    return torch.matmul(v, o)


def qk_matrix(W_Q: Any, W_K: Any) -> Tensor:
    """Compute the QK circuit matrix W_Q @ W_K.T per head.

    The QK circuit captures which pairs of residual-stream positions a head
    attends between: ``score(q, k) = x_q @ W_QK @ x_k.T``.

    Args:
        W_Q: Query projection, shape ``(..., d_model, d_head)``.
        W_K: Key projection, shape ``(..., d_model, d_head)``.

    Returns:
        ``(..., d_model, d_model)`` — the QK matrix ``W_Q @ W_K.T`` per head.
    """
    q = torch.as_tensor(W_Q).detach().to(torch.float32)
    k = torch.as_tensor(W_K).detach().to(torch.float32)
    return torch.matmul(q, k.transpose(-2, -1))


def head_composition_score(W_OV: Any, W_dest: Any) -> float:
    """Frobenius-norm composition score between a source head's OV and a destination Q/K/V.

    Measures how strongly head A's output feeds into head B's query, key, or value
    computation.  The score is the normalised Frobenius norm of the composed matrix:

    .. code-block:: text

        score = ‖W_OV @ W_dest‖_F / (‖W_OV‖_F · ‖W_dest‖_F)

    A score near 1 indicates strong composition; near 0 indicates the heads operate
    in nearly orthogonal subspaces.  The score is always in ``[0, 1]`` (follows from
    Frobenius sub-multiplicativity).

    Args:
        W_OV:   ``(d_model, d_model)`` OV matrix of the source head (from
                :func:`ov_matrix`).
        W_dest: ``(d_model, d_head)`` weight matrix (W_Q / W_K / W_V) of the
                destination head.

    Returns:
        Scalar float in ``[0, 1]``.

    Reference:
        Elhage et al. 2021 §3.3 "Composition".
    """
    ov = torch.as_tensor(W_OV).detach().to(torch.float32)
    d = torch.as_tensor(W_dest).detach().to(torch.float32)
    composed = ov @ d
    num = composed.norm(p="fro")
    denom = ov.norm(p="fro") * d.norm(p="fro")
    return (num / denom.clamp_min(1e-8)).item()


def composition_scores(W_OV_src: Any, W_dest: Any) -> Tensor:
    """Batched composition scores for all source→destination head pairs.

    Args:
        W_OV_src: ``(n_heads_src, d_model, d_model)`` OV matrices for each
                  source head (from :func:`ov_matrix`).
        W_dest:   ``(n_heads_dst, d_model, d_head)`` Q/K/V matrices for each
                  destination head.

    Returns:
        ``(n_heads_src, n_heads_dst)`` float tensor of composition scores.
    """
    src = torch.as_tensor(W_OV_src).detach().to(torch.float32)   # (S, d, d)
    dst = torch.as_tensor(W_dest).detach().to(torch.float32)      # (D, d, h)

    # composed[s, d_idx] = src[s] @ dst[d_idx] → (S, D, d_model, d_head)
    composed = torch.einsum("sij,djk->sdik", src, dst)            # (S, D, d, h)

    num = composed.flatten(start_dim=2).norm(dim=-1)              # (S, D)
    src_norms = src.flatten(start_dim=1).norm(dim=-1)             # (S,)
    dst_norms = dst.flatten(start_dim=1).norm(dim=-1)             # (D,)
    denom = torch.outer(src_norms, dst_norms).clamp_min(1e-8)    # (S, D)
    return num / denom


def top_logit_tokens(
    direction: Any,
    W_U: Any,
    *,
    k: int = 10,
) -> tuple[list[int], list[float]]:
    """Top-k tokens promoted by a residual-space direction via the unembedding.

    Computes ``direction @ W_U`` and returns the tokens with the highest scores.
    Useful for interpreting what a feature, steering direction, or head output
    promotes in the model's output distribution.

    Args:
        direction: ``(d_model,)`` direction vector in residual space.
        W_U:       ``(d_model, vocab_size)`` unembedding matrix.
        k:         Number of top tokens to return (default 10).

    Returns:
        ``(token_ids, scores)`` — two parallel lists of length ``k``.
    """
    d = torch.as_tensor(direction).detach().to(torch.float32).flatten()
    wu = torch.as_tensor(W_U).detach().to(torch.float32)
    logits = d @ wu                           # (vocab_size,)
    topk = logits.topk(k)
    return topk.indices.tolist(), topk.values.tolist()


def top_embedding_tokens(
    direction: Any,
    W_E: Any,
    *,
    k: int = 10,
) -> tuple[list[int], list[float]]:
    """Top-k tokens whose embeddings are most aligned with a residual-space direction.

    Computes ``W_E @ direction`` (dot product of each token embedding with
    *direction*) and returns the tokens with the highest scores.

    Args:
        direction: ``(d_model,)`` direction vector in residual space.
        W_E:       ``(vocab_size, d_model)`` token embedding matrix.
        k:         Number of top tokens to return (default 10).

    Returns:
        ``(token_ids, scores)`` — two parallel lists of length ``k``.
    """
    d = torch.as_tensor(direction).detach().to(torch.float32).flatten()
    we = torch.as_tensor(W_E).detach().to(torch.float32)
    scores = we @ d                           # (vocab_size,)
    topk = scores.topk(k)
    return topk.indices.tolist(), topk.values.tolist()


def transcoder_virtual_weights(W_dec_up: Any, W_enc_down: Any) -> Tensor:
    """Input-independent feature→feature connectivity between two transcoders.

    Composes the upstream transcoder's decoder with the downstream transcoder's
    encoder: ``V = W_dec_up @ W_enc_down``.  ``V[i, j]`` is the weight-derived
    ("virtual") connection strength from upstream feature ``i`` to downstream
    feature ``j`` — the contribution feature ``i`` makes to feature ``j``'s
    pre-activation, before any input-dependent gating.  This is the global
    connectivity map that complements per-prompt attribution graphs: it depends
    only on weights, so it covers all inputs at once and costs one matmul.

    Args:
        W_dec_up:   ``(n_features_up, d_model)`` upstream decoder matrix.
        W_enc_down: ``(d_model, n_features_down)`` downstream encoder matrix.

    Returns:
        ``(n_features_up, n_features_down)`` float32 virtual-weight matrix.

    Reference:
        Circuit Insights (arXiv:2510.14936); Paulo et al.,
        "Transcoders Beat Sparse Autoencoders for Interpretability"
        (arXiv:2501.18823).
    """
    dec = torch.as_tensor(W_dec_up).detach().to(torch.float32)
    enc = torch.as_tensor(W_enc_down).detach().to(torch.float32)
    if dec.shape[-1] != enc.shape[0]:
        raise ValueError(
            f"d_model mismatch: W_dec_up has {dec.shape[-1]} columns but "
            f"W_enc_down has {enc.shape[0]} rows"
        )
    return dec @ enc


def top_virtual_connections(
    V: Any,
    *,
    k: int = 20,
) -> list[tuple[int, int, float]]:
    """Strongest entries of a virtual-weight matrix by ``|weight|``.

    Args:
        V: ``(n_features_up, n_features_down)`` virtual-weight matrix (from
           :func:`transcoder_virtual_weights`).
        k: Number of connections to return (default 20; capped at ``V.numel()``).

    Returns:
        List of ``(upstream_feature, downstream_feature, weight)`` triples
        sorted by ``|weight|`` descending.
    """
    v = torch.as_tensor(V).detach().to(torch.float32)
    if v.ndim != 2:
        raise ValueError(f"V must be 2-D, got ndim={v.ndim}")
    k = min(k, v.numel())
    flat_idx = v.abs().flatten().topk(k).indices
    rows = (flat_idx // v.shape[1]).tolist()
    cols = (flat_idx % v.shape[1]).tolist()
    return [(i, j, float(v[i, j])) for i, j in zip(rows, cols, strict=True)]


def feature_token_alignment(
    W_dec: Any,
    W_U: Any,
    *,
    k: int = 10,
) -> tuple[Tensor, Tensor]:
    """Per-feature top-k promoted logit tokens via decoder → unembedding.

    Weight-only "what does this feature do": projects every decoder row through
    the unembedding (``W_dec @ W_U``) and returns each feature's top-k promoted
    tokens.  The batched, all-features analogue of :func:`top_logit_tokens`.

    Args:
        W_dec: ``(n_features, d_model)`` decoder matrix (SAE or transcoder).
        W_U:   ``(d_model, vocab_size)`` unembedding matrix.
        k:     Number of top tokens per feature (default 10).

    Returns:
        ``(token_ids, scores)`` — int64 / float32 tensors, each of shape
        ``(n_features, k)``, row ``f`` sorted by score descending.
    """
    dec = torch.as_tensor(W_dec).detach().to(torch.float32)
    wu = torch.as_tensor(W_U).detach().to(torch.float32)
    logits = dec @ wu                         # (n_features, vocab_size)
    topk = logits.topk(min(k, logits.shape[-1]), dim=-1)
    return topk.indices, topk.values
