"""Stock recsys recipe — sequential recommenders (SASRec, BERT4Rec, GRU4Rec).

Scope and relationship to ``two_tower``
---------------------------------------
This recipe covers **sequential** recommenders — transformer sequence encoders
(SASRec, BERT4Rec) and RNN sequence encoders (GRU4Rec). The reliable anchor for
the family is the item/position **embedding table**, which every sequential
recommender has; that weight pattern is required, and the
architecture-specific patterns (attention vs. GRU, FFN, norms) are marked
``optional`` so a model from one sub-family attaches cleanly under the default
``strict=True`` without matching the others.

Two-tower retrieval and DLRM-style feature-interaction models are covered by the
dedicated :mod:`circuitry.recipes.two_tower` recipe, which also ships the
``embedding_alignment`` custom diagnostic (query-tower vs. item-tower cosine).
Use ``two_tower`` for those; this recipe does **not** duplicate its patterns.

Architecture coverage
----------------------
**Transformer-recsys** (SASRec / BERT4Rec):
  ``encoder.item_emb``, ``encoder.pos_emb``  — nn.Embedding (required anchor)
  ``encoder.blocks.N.self_attn``              — nn.MultiheadAttention (see NOTE-A)
  ``encoder.blocks.N.ffn``                   — nn.Sequential
  ``encoder.blocks.N.ffn.0``, ``.2``         — nn.Linear (matched individually)
  ``encoder.blocks.N.norm1``, ``.norm2``     — nn.LayerNorm
  ``encoder.layer_norm``                     — nn.LayerNorm

**GRU4Rec**:
  ``item_emb`` — nn.Embedding (required anchor)
  ``gru``      — nn.GRU: inventory extracts ``weight_ih_l0`` / ``weight_hh_l0``
                 from the module's subtree (each is 2-D → resolves as primary weight).

NOTE-A — WEIGHT resolution for nn.MultiheadAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``nn.MultiheadAttention`` owns two 2-D+ weights in its subtree:
``in_proj_weight`` (shape (3*D, D)) and ``out_proj.weight`` (shape (D, D)).
``ModelInventory.find_primary_weight`` returns ``None`` when there is more
than one candidate — so a WEIGHT hookpoint on the fused
``encoder.blocks.N.self_attn`` module is logged as UNRESOLVED.

To get weight diagnostics on attention, hook the projection module directly:
  - ``.*\\.self_attn\\.out_proj$`` — resolves to ``out_proj.weight`` (unique 2-D param)
    (``in_proj_weight`` is not reachable by the recipe DSL; see open follow-up #3
    in docs/observations/2026-06-01-recsys-sasrec-evaluation.md.)

NOTE-B — attention_pattern_entropy + need_weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SASRec (and most custom transformer-recsys models) call
``self_attn(…, need_weights=False)`` by default.  circuitry captures attention
weights via a forward hook that reads ``out[1]``; when ``need_weights=False``
PyTorch returns ``out[1] = None`` and no capture happens.

Before attaching the Recorder, apply a per-instance forward monkeypatch::

    def _patch_mha(model) -> None:
        def force_weights(mha):
            orig = mha.forward
            def wrapped(*a, **kw):
                kw["need_weights"]        = True
                kw["average_attn_weights"] = False   # keeps per-head shape (B, H, T, T)
                return orig(*a, **kw)
            mha.forward = wrapped
        for blk in model.encoder.blocks:
            force_weights(blk.self_attn)

NOTE-C — PAD-row NaN in attention_pattern_entropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SASRec uses LEFT-padding.  PAD query rows attend to an all-(-inf) key set;
``softmax([-inf, …, -inf])`` returns NaN (not uniform) in PyTorch, which
propagates through Shannon-entropy computation.  As of the recsys-followups
cycle ``attention_pattern_entropy`` is NaN-aware: it drops fully-masked PAD
rows from the per-head mean automatically, and accepts an optional
``valid_mask`` (``True`` marks a valid query row, broadcastable to
``(B, H, T_query)``) when you want to restrict the average to specific rows.
So a left-padded pattern no longer emits NaN; pass ``valid_mask`` only for
explicit control over which query rows count.

NOTE-D — HF-only diagnostics disabled by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``induction_score``, ``logit_lens_kl``, and ``drift_probe`` call
``model(probe, output_attentions=True)`` or ``model.get_output_embeddings()``.
These break on models whose forward entry-point is not an HF-style ``forward()``
(e.g. SASRecModel uses ``predict_scores``).  They are disabled by default via
``enabled`` and must be opted in explicitly after verifying HF compatibility::

    from circuitry.recipes.recsys import RECIPE
    recipe = dataclasses.replace(RECIPE,
        enabled={**RECIPE.enabled, "induction_score": True, "logit_lens_kl": True})
"""

from __future__ import annotations

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource

# ---------------------------------------------------------------------------
# WEIGHT hook patterns
# ---------------------------------------------------------------------------

# Embedding tables: item/pos/user/token embeddings (transformer-recsys + GRU4Rec)
# and any module whose final segment ends in '_emb', 'embedding', or 'embeddings'.
# Every sequential recommender has an item embedding, so this is the REQUIRED
# anchor (non-optional): a 0-match means the recipe was pointed at the wrong model.
_W_EMBS = (
    r".*\.(item_emb|pos_emb|user_emb|token_emb)$"
    r"|.*_emb$"
    r"|.*embedding$"
    r"|.*embeddings$"
)

# Attention out-projection — the one 2-D-unique weight on nn.MultiheadAttention
# that the inventory CAN resolve (see NOTE-A). Transformer-recsys only.
_W_ATTN_OUT_PROJ = r".*\.self_attn\.out_proj$"

# FFN Linear leaves — numbered children of nn.Sequential FFNs (ffn.0, ffn.2) plus
# named MLP projections used by other recsys models. Transformer-recsys only.
_W_FFN_LINEARS = (
    r".*\.ffn\.\d+$"
    r"|.*(gate_proj|up_proj|down_proj|w1|w2|w3)$"
)

# GRU weights (GRU4Rec): inventory extracts weight_ih_l0 / weight_hh_l0 from the
# subtree of an nn.GRU module (each is a unique 2-D param in its owning module).
_W_GRU = r".*\.gru$|.*\bgru\b$"

# ---------------------------------------------------------------------------
# OUTPUT (activation) hook patterns
# ---------------------------------------------------------------------------

# Attention submodule output — covers SASRec / BERT4Rec / HF-LLM naming.
# NOTE: requires the need_weights monkeypatch for attention_pattern_entropy
# to emit non-NaN values (see NOTE-B).
_O_ATTN = r".*\.(self_attn|attn|attention)$"

# FFN submodule output — SASRec: 'ffn'; LLM-style: 'mlp'.
_O_FFN = r".*\.(ffn|mlp)$"

# LayerNorm outputs — SASRec (norm1/norm2/layer_norm), HF (input_layernorm /
# post_attention_layernorm / attention_norm / ffn_norm), GPT-2 (ln_1/ln_2).
_O_NORM = (
    r".*\.(norm1|norm2|layer_norm"
    r"|input_layernorm|post_attention_layernorm"
    r"|attention_norm|ffn_norm"
    r"|ln_[12])$"
)

# Block-level output — SASRec 'encoder.blocks.N'; HF 'model.layers.N'.
_O_BLOCK = r".*\.(blocks|layers)\.\d+$"

# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------
#
# Only the embedding anchor is required; every other pattern is optional because
# it is present in one sub-family and absent in another (attention/FFN/norm/block
# on a transformer encoder; GRU on an RNN encoder). optional=True makes a 0-match
# a soft skip under the default strict=True (see HookPoint.optional).

RECIPE = Recipe(
    name="recsys",
    hook_points=[
        # ---- WEIGHT: embeddings (required anchor) ----
        HookPoint(source=TensorSource.WEIGHT, pattern=_W_EMBS),
        # ---- WEIGHT: attention out-projection (transformer-recsys, see NOTE-A) ----
        HookPoint(source=TensorSource.WEIGHT, pattern=_W_ATTN_OUT_PROJ, optional=True),
        # ---- WEIGHT: FFN linear leaves (transformer-recsys) ----
        HookPoint(source=TensorSource.WEIGHT, pattern=_W_FFN_LINEARS, optional=True),
        # ---- WEIGHT: GRU (GRU4Rec) ----
        HookPoint(source=TensorSource.WEIGHT, pattern=_W_GRU, optional=True),
        # ---- OUTPUT: attention (see NOTE-B/C for need_weights and PAD NaN) ----
        HookPoint(source=TensorSource.OUTPUT, pattern=_O_ATTN, optional=True),
        # ---- OUTPUT: FFN / MLP ----
        HookPoint(source=TensorSource.OUTPUT, pattern=_O_FFN, optional=True),
        # ---- OUTPUT: layer-norms ----
        HookPoint(source=TensorSource.OUTPUT, pattern=_O_NORM, optional=True),
        # ---- OUTPUT: block-level (for layer-wise drift / activation tracking) ----
        HookPoint(source=TensorSource.OUTPUT, pattern=_O_BLOCK, optional=True),
    ],
    weight_diagnostics=[
        "effective_rank",
        "attention_head_rank",
        "stable_rank",
        "condition_number",
        "heavy_tail_alpha",
        "sv_histogram",
        # training-dynamics (v1.3):
        "update_delta",
        "rank_trajectory",
        "direction_cosine",
    ],
    activation_diagnostics=[
        "gate_stats",
        "dead_fraction",
        "kurtosis",
        "participation_ratio",
        # attention entropy (see NOTE-C: emits NaN for left-padded models without patch).
        "attention_pattern_entropy",
        # HF-only diagnostics — disabled by default (see NOTE-D).
        "induction_score",
        "logit_lens_kl",
        "drift_probe",
    ],
    gradient_diagnostics=["norms_per_param"],
    # NOTE-D: disable HF-only diagnostics by default. They call
    # model(probe, output_attentions=True) / model.get_output_embeddings(),
    # which break on custom recsys models without an HF-style forward().
    enabled={
        "induction_score": False,
        "logit_lens_kl": False,
        "drift_probe": False,
    },
)


def register() -> None:
    """Register the recsys recipe.  Idempotent under test fixtures via
    ``_clear_registry_for_tests``."""
    register_recipe(RECIPE)
