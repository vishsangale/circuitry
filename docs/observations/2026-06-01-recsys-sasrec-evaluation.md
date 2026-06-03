# Recsys evaluation: SASRec D=64 against circuitry v1.8.0

**Date:** 2026-06-01
**Branch:** main @ v1.8.0
**Goal:** Characterise circuitry's coverage on a trained SASRec recommender model,
identify the gaps, propose and validate a new `recsys` recipe.  This is a
**contribution proposal** — the draft recipe lives at
`src/circuitry/recipes/recsys.py` and the draft test at
`tests/recipes/test_recsys_recipe_draft.py`.

**Model:** SASRec D=64 — 2 transformer blocks, n_heads=4, n_items=1349, max_seq_len=200.
Module names: `encoder.item_emb` / `encoder.pos_emb` (nn.Embedding),
`encoder.blocks.N.self_attn` (nn.MultiheadAttention with fused `in_proj_weight` +
`out_proj`), `encoder.blocks.N.ffn` (nn.Sequential: `ffn.0` Linear / `ffn.1` ReLU /
`ffn.2` Linear), `encoder.blocks.N.norm1/norm2` (LayerNorm), `encoder.layer_norm`,
`encoder.blocks` (nn.ModuleList of blocks).

**Environment:**
- rtx (RTX 5080), CPU only (hard constraint — GPU sweep running).
- circuitry v1.8.0, PYTHONPATH=$HOME/workspace/circuitry/src.
- Checkpoint: `results/checkpoints/salvage/sasrec_d64_seed.pt`.
- Synthetic sequences: B=64, T=50, random lengths [2, 50], seed=0.
- Validation driver: `llm-recsys/scripts/circuitry_recsys_validate.py`.
- Full scan output: `results/circuitry/sasrec_d64_recsys_scan.json`.

---

## 0. Executive summary

The llm recipe fired on **1 of 12** hook patterns on SASRec, emitting 20 unique tags.
The new `recsys` recipe fires on **7 of 10** patterns and emits **106 unique tags** —
a **5.3× increase**.  Three zero-match patterns (DLRM/two-tower MLPs, GRU, per-tower
output hooks) are expected-zero on SASRec and document the recipe's multi-architecture
breadth.  Four findings (A–D) required fixes or workarounds; each has a clear fix
direction for the circuitry maintainers.

---

## 1. Findings register

### Finding A — Recipe gap: no recsys recipe exists  🟠

**Evidence:**
- `two_tower` recipe: patterns locked to `query_tower|item_tower|interaction`.
  Matched **0 modules** on SASRec (no two-tower naming).
- `llm` recipe: 11 of 12 patterns matched 0 modules on SASRec.  Only
  `.*\.(self_attn|attn|attention)$` fired — 2 modules matched — producing 20 tags,
  all from activation diagnostics on the attention OUTPUT.

**Pattern family coverage (llm recipe on SASRec):**

| Pattern | Source | Matched modules |
|---|---|---|
| `.*\.(q\|k\|v\|o)_proj$` | WEIGHT | 0 |
| `.*\.(w1\|w2\|w3\|gate_proj\|up_proj\|down_proj)$` | WEIGHT | 0 |
| `.*\.mlp\.gate$` | WEIGHT | 0 |
| `.*\.mlp\.experts$` | WEIGHT | 0 |
| `.*\.(self_attn\|attn\|attention)$` | **OUTPUT** | **2** (blocks.0/1.self_attn) |
| `.*\.mlp$` | OUTPUT | 0 |
| `.*\.down_proj$` | INPUT | 0 |
| `.*\.(input_layernorm\|…\|ln_[12])$` | OUTPUT | 0 |
| `embed.*` | WEIGHT | 0 |
| `lm_head$` | WEIGHT | 0 |
| `.*\.(q\|k\|v\|o)_proj$` | GRAD | 0 |
| `.*\.layers\.\d+$` | OUTPUT | 0 |

**Fix direction:** Author and ship the `recsys` recipe (see §3).

---

### Finding B — PAD-row NaN in attention_pattern_entropy  🟠

**Evidence:**
SASRec uses **left-padding**.  PAD query rows attend to an all-`−∞` key set;
`softmax([−∞, …, −∞])` returns NaN in PyTorch (not 0 or uniform).  This NaN
propagates through Shannon-entropy computation (`−p log p`), causing
`attention_pattern_entropy` to emit NaN for any model with padding.

circuitry's `attention_pattern_entropy` takes no `pad_mask` argument.

Custom masked-entropy (PAD query rows excluded, confirmed non-NaN):

| Module | Head 0 | Head 1 | Head 2 | Head 3 |
|---|---|---|---|---|
| encoder.blocks.0.self_attn | 2.071 nats | 1.873 nats | 1.967 nats | 1.801 nats |
| encoder.blocks.1.self_attn | 2.944 nats | 2.954 nats | 2.924 nats | 2.946 nats |

Block 0 shows varied per-head entropy (~1.8–2.1 nats); block 1 is near-uniform
(~2.94 nats ≈ log(4 heads × T/…)).  The values are mechanistically meaningful
once PAD rows are excluded.

**Fix direction:** Add an optional `pad_mask` (or `valid_mask`) argument to
`attention_pattern_entropy` in `core/attention.py`.  The mask should be
boolean-broadcastable to `(B, H, T)`, and entropy should average only over
valid (unmasked) query rows.

---

### Finding C — Attention capture fails without need_weights patch  🟠

**Evidence:**
circuitry captures attention weights in `recorder/live.py ~526-539` via a
forward hook that reads `out[1]` and requires `isinstance(out[1], torch.Tensor)`.

SASRec's `TransformerBlock.forward` calls:
```python
attn_out, _ = self.self_attn(q, k, v, …, need_weights=False)
```

`need_weights=False` causes PyTorch's `nn.MultiheadAttention` to return
`out[1] = None` → the capture hook silently skips.

`_set_output_attentions_true()` attempts to set `model.config.output_attentions = True`
(HuggingFace style), but `SASRecModel` has no `config` attribute → silent no-op with
no warning, no `skip` flag.

**Workaround (validated):** Instance-level forward monkeypatch before `rec.attach()`:

```python
def _patch_mha(model) -> None:
    for blk in model.encoder.blocks:
        orig = blk.self_attn.forward
        def wrapped(*a, mha_orig=orig, **kw):
            kw["need_weights"]         = True
            kw["average_attn_weights"] = False  # keeps (B, H, T, T) shape
            return mha_orig(*a, **kw)
        blk.self_attn.forward = wrapped

_patch_mha(model)
rec.attach()
```

Captured shape: `(64, 4, 50, 50)` — `(B, n_heads, T, T)` as expected.

**Fix direction (two independent improvements):**
1. When `_set_output_attentions_true()` finds no `model.config`, emit a
   `WARNING` so users know attention capture is disabled rather than silently
   degrading.
2. Consider exposing a recipe-level `attn_kwargs` dict that the recorder
   injects into the forward call of matched attention modules, as an
   alternative to relying on `model.config`.

---

### Finding D — HF-interface diagnostics break on non-HF models  🟠

**Evidence:**
Three diagnostics call `model(probe[, output_attentions=True])` or
`model.get_output_embeddings()`:
- `induction_score`: calls `model(probe, output_attentions=True)` → `TypeError`
  (SASRecModel's entry point is `predict_scores`, not `forward`).
- `logit_lens_kl`: requires `model.get_output_embeddings()` or `model.lm_head` →
  `WARNING: no resolvable output embedding — skipping`.
- `drift_probe`: calls a second `model(probe_batch)` forward → same `TypeError`.

These are correctly disabled by default in the `recsys` recipe via
`enabled={"induction_score": False, "logit_lens_kl": False, "drift_probe": False}`.

**Fix direction:** Consider adding a `recipe.forward_fn` field — a callable
`(model, batch) -> output` that replaces the bare `model(probe)` call.  This
would allow custom entry points (`predict_scores`, `encode_query`, etc.) without
requiring an HF-style `forward`.  Until then, disable these three diagnostics
via `recipe.disable([...])` for any model without an HF-compatible interface.

---

## 2. Coverage delta: recsys recipe vs llm recipe

**Model:** SASRec D=64 (n_items=1349, D=64, 2 blocks, 4 heads).
**Patch applied:** MHA need_weights monkeypatch (Finding C).
**Disabled for both recipes:** `induction_score`, `logit_lens_kl`, `drift_probe`
(Finding D).

### 2.1 Hookpoint coverage

| HP# | Source | Pattern | llm | recsys |
|---|---|---|---|---|
| — | OUTPUT | `.*\.(self_attn\|attn\|attention)$` | **2 modules** | **2 modules** |
| 0 | WEIGHT | embedding patterns (`item_emb\|pos_emb\|…\|embed_tables`) | — | **2 modules** |
| 1 | WEIGHT | `.*\.self_attn\.out_proj$` | — | **2 modules** |
| 2 | WEIGHT | `.*\.ffn\.\d+$\|…` (FFN leaves) | — | **6 modules** (4 Linear + 2 ReLU — ReLU UNRESOLVED, expected) |
| 3 | WEIGHT | DLRM/two-tower MLPs | — | 0 (expected zero on SASRec) |
| 4 | WEIGHT | `.*\.gru$` | — | 0 (expected zero) |
| 5 | OUTPUT | `.*\.(self_attn\|attn\|attention)$` | — | **2 modules** |
| 6 | OUTPUT | `.*\.(ffn\|mlp)$` | — | **2 modules** |
| 7 | OUTPUT | norm patterns | — | **5 modules** (norm1×2, norm2×2, layer_norm) |
| 8 | OUTPUT | `.*\.(blocks\|layers)\.\d+$` | — | **2 modules** |
| 9 | OUTPUT | tower output patterns | — | 0 (expected zero) |

### 2.2 Tag family comparison

| Family | llm recipe | recsys recipe |
|---|---|---|
| `activation/*` | 20 tags (attn entropy only, all NaN) | 68 tags (dead_fraction, gate_stats, kurtosis, participation_ratio on attn+ffn+norm+block) |
| `weight/*` | 0 tags | 38 tags (effective_rank, stable_rank, condition_number, heavy_tail_alpha on emb/out_proj/ffn) |
| `gradient/*` | 0 tags | 0 tags (grad diagnostics require named matching on q/k/v/o_proj) |

**Net delta:** +86 new unique tags; 0 lost.  Total: 20 → 106 (5.3×).

### 2.3 Inventory gaps (zero-match patterns on SASRec)

The following proposed patterns matched 0 modules on SASRec — they are not
bugs in the recipe but reflect architecture scope:

| Pattern | Source | Gap reason |
|---|---|---|
| `(bottom_mlp\|top_mlp\|query_tower\|item_tower\|interaction).*` | WEIGHT | SASRec is not a two-tower or DLRM model |
| `.*\.gru$` | WEIGHT | SASRec uses transformer blocks, not GRU |
| `(query_tower\|item_tower)(\.\d+)?$` | OUTPUT | Same — two-tower patterns |

These patterns provide coverage for GRU4Rec and DLRM models respectively;
they are correctly silent on SASRec.

**Note on `attention_head_rank`:** This diagnostic requires `model.config.num_attention_heads`
(HF-style config). SASRec has none → circuitry logs a WARNING and skips. This is a Finding D
variant: `attention_head_rank` should accept an explicit `n_heads` override.

---

## 3. Proposed recsys recipe — `src/circuitry/recipes/recsys.py`

> **Maintainer note (shipped form, 2026-06-02).** The recipe was landed *scoped to
> sequential recommenders* (SASRec / BERT4Rec / GRU4Rec). The DLRM and two-tower
> patterns (`bottom_mlp` / `top_mlp` / `query_tower` / `item_tower` / `interaction`)
> were **removed** from this recipe to avoid duplicating the dedicated `two_tower`
> recipe, which already covers two-tower + DLRM *and* ships the `embedding_alignment`
> custom diagnostic. `recsys` and `two_tower` are therefore **complementary**, not
> overlapping. The item/position embedding is the required anchor; every
> architecture-variant pattern (attention/FFN/norm/block vs. GRU) is marked
> `HookPoint(optional=True)` so a model from one sub-family attaches cleanly under
> the default `strict=True` (this relies on the `optional` flag added in the same
> change — see `tests/recorder/test_optional_hookpoints.py`). The §3 description
> below is the original broad proposal; the DLRM/GRU/two-tower bullets in it are
> superseded by this note.

The draft recipe covers:

**WEIGHT hooks:**
- Embeddings: `item_emb|pos_emb|user_emb|token_emb` + `.*_emb$` + `.*embedding$` +
  `embed_tables(\.\d+)?$` — catches both SASRec-style and DLRM-style embedding tables.
- Attention out-projection: `.*\.self_attn\.out_proj$` — the one 2-D-unique weight
  that `ModelInventory.find_primary_weight` can resolve on an `nn.MultiheadAttention`
  subtree (the fused `in_proj_weight` + `out_proj` means the parent module has 2 candidates
  → the inventory returns `None` for the parent, but `out_proj` has exactly one 2-D weight).
- FFN leaves: `.*\.ffn\.\d+$` — catches numbered nn.Sequential children (SASRec ffn.0/ffn.2).
  ReLU children (ffn.1) match but are UNRESOLVED (expected: no weight), logged with a WARNING.
- DLRM/two-tower: `(bottom_mlp|top_mlp|query_tower|item_tower|interaction).*`.
- GRU (GRU4Rec): `.*\.gru$` — inventory extracts `weight_ih_l0`/`weight_hh_l0` from subtree.

**OUTPUT (activation) hooks:**
- Attention: `.*\.(self_attn|attn|attention)$`
- FFN/MLP: `.*\.(ffn|mlp)$`
- LayerNorms: `norm1|norm2|layer_norm|input_layernorm|post_attention_layernorm|attention_norm|ffn_norm|ln_[12]`
- Block-level: `.*\.(blocks|layers)\.\d+$`
- Tower outputs: `(query_tower|item_tower)(\.\d+)?$` (per-table child hooks)

**Diagnostics:**
- Weight: `effective_rank`, `attention_head_rank`, `stable_rank`, `condition_number`,
  `heavy_tail_alpha`, `sv_histogram`, `update_delta`, `rank_trajectory`, `direction_cosine`.
- Activation: `gate_stats`, `dead_fraction`, `kurtosis`, `participation_ratio`,
  `attention_pattern_entropy` (Note B: NaN for left-padded models without patch).
- HF-only disabled: `induction_score`, `logit_lens_kl`, `drift_probe` — `enabled=False` by default.

---

## 4. Open issues and follow-ups

1. **`attention_pattern_entropy` + PAD mask (Finding B):** Until a `pad_mask` argument is
   added, recsys users must compute masked entropy manually (see the validation driver for
   a reference implementation).
2. **`attention_head_rank` + custom n_heads (Finding D variant):** Add an explicit `n_heads`
   override to `attention_head_rank` in `core/weight.py`, or allow the recipe to carry a
   `n_attention_heads` hint that the recorder reads.
3. **MHA WEIGHT resolution (Finding A + inventory):** The fused `in_proj_weight` + `out_proj`
   in `nn.MultiheadAttention` means the parent module is UNRESOLVED as a WEIGHT hookpoint.
   Hooking `out_proj` directly recovers `out_proj.weight`; `in_proj_weight` can only be
   reached by hooking `self_attn.in_proj_weight` as a named-parameter target (not currently
   supported by the recipe DSL).  Consider a `TensorSource.NAMED_PARAM` source or an explicit
   parameter-name hookpoint for `in_proj_weight`.
4. **`_set_output_attentions_true()` silent no-op (Finding C):** Log a WARNING when no
   `model.config` is found so users know attention capture is degraded.
5. **Gradient diagnostics on SASRec:** `norms_per_param` needs patterns matching recsys
   parameter names (e.g. `encoder\.blocks\.\d+\.(self_attn|ffn).*`). The current recipe
   includes `norms_per_param` in `gradient_diagnostics` but emitted 0 gradient tags in
   this run (the GRAD hookpoints resolve only if there is a WEIGHT hookpoint for the same
   module — the FFN linears do have WEIGHT hooks, so this may be a step()-ordering issue
   to investigate).
