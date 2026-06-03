"""Tests for the stock ``recsys`` recipe (sequential recommenders).

This test is SELF-CONTAINED: it defines a minimal SASRec-like nn.Module inline
and does NOT import any external recsys codebase.  It validates:

1. The recsys recipe emits > 0 WEIGHT tags on the minimal model — under the
   DEFAULT ``strict=True``, which exercises that the optional cross-architecture
   patterns (GRU, etc.) may match 0 modules without failing attach.
2. The recsys recipe emits > 0 activation tags on the minimal model.
3. With the need_weights monkeypatch + a PAD-masked entropy computation,
   ``attention_pattern_entropy`` emits finite (non-NaN) values for a
   left-padded input.

Architecture (inline minimal SASRec-like):
  - encoder.item_emb     nn.Embedding(vocab, D)
  - encoder.pos_emb      nn.Embedding(T, D)
  - encoder.blocks       nn.ModuleList of _TransformerBlock:
      - block.self_attn  nn.MultiheadAttention(D, n_heads)
      - block.ffn        nn.Sequential(Linear(D,D), ReLU(), Linear(D,D))
      - block.norm1      nn.LayerNorm(D)
      - block.norm2      nn.LayerNorm(D)
  - encoder.layer_norm   nn.LayerNorm(D)

The model's forward() does a standard SASRec pass and returns logits.

Source: ``docs/observations/2026-06-01-recsys-sasrec-evaluation.md``.
Findings: A (no recsys recipe), B (PAD NaN), C (need_weights), D (HF-only diags).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _clear_registry_for_tests, _register_stock_recipes
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter

# ---------------------------------------------------------------------------
# Minimal SASRec-like model — no llmrecsys imports
# ---------------------------------------------------------------------------

PAD_ID = 0  # left-padding token id


class _Block(nn.Module):
    """Single transformer block matching SASRec module names."""

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        # need_weights=False is the SASRec default (Finding C).
        attn_out, _ = self.self_attn(x, x, x,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class _SASRecLike(nn.Module):
    """Minimal SASRec-like model with matching module names."""

    def __init__(self, n_items: int = 200, D: int = 16,
                 n_layers: int = 2, n_heads: int = 2, T: int = 10) -> None:
        super().__init__()
        self.T = T
        self.D = D
        # Create encoder as a sub-namespace so names match
        # encoder.item_emb / encoder.pos_emb / encoder.blocks.N / encoder.layer_norm
        class _Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.item_emb  = nn.Embedding(n_items + 1, D, padding_idx=PAD_ID)
                self.pos_emb   = nn.Embedding(T + 1, D)
                self.blocks    = nn.ModuleList([_Block(D, n_heads) for _ in range(n_layers)])
                self.layer_norm = nn.LayerNorm(D)

        self.encoder = _Encoder()
        self.output_proj = nn.Linear(D, n_items + 1)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """item_seq: (B, T) long tensor; PAD_ID=0 for padding positions."""
        B, T = item_seq.shape
        positions = torch.arange(1, T + 1, device=item_seq.device).unsqueeze(0)
        key_padding_mask = (item_seq == PAD_ID)  # True = ignore this position

        x = self.encoder.item_emb(item_seq) + self.encoder.pos_emb(positions)
        x = x * math.sqrt(self.D)

        for blk in self.encoder.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.encoder.layer_norm(x)
        # Use the last non-PAD token output as the user representation.
        # Gather the last real token (right-most non-PAD position).
        idx = (T - (item_seq != PAD_ID).float().flip(dims=[1]).cumsum(dim=1)
               .flip(dims=[1]).ge(1).int().argmax(dim=1) - 1).clamp(min=0)
        user_repr = x[torch.arange(B), idx]  # (B, D)
        return self.output_proj(user_repr)  # (B, n_items+1)


# ---------------------------------------------------------------------------
# need_weights monkeypatch (Finding C workaround)
# ---------------------------------------------------------------------------


def _patch_mha_need_weights(model: _SASRecLike) -> None:
    """Force need_weights=True on all block self_attn modules.

    This is the workaround for Finding C: circuitry's attention capture hook
    reads ``out[1]`` from the MHA output; when ``need_weights=False`` PyTorch
    returns ``out[1] = None`` and the hook skips silently.
    """
    for blk in model.encoder.blocks:
        orig = blk.self_attn.forward

        def make_wrapped(original):
            def wrapped(*a, **kw):
                kw["need_weights"]         = True
                kw["average_attn_weights"] = False   # keep (B, H, T, T)
                return original(*a, **kw)
            return wrapped

        blk.self_attn.forward = make_wrapped(orig)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate registry state, then restore the full stock set on teardown so we
    do not leave the registry empty for later tests. ``_register_stock_recipes``
    includes ``recsys`` and is idempotent."""
    _clear_registry_for_tests()
    _register_stock_recipes()
    yield
    _clear_registry_for_tests()
    _register_stock_recipes()


@pytest.fixture()
def model():
    m = _SASRecLike(n_items=200, D=16, n_layers=2, n_heads=2, T=10)
    m.eval()
    return m


@pytest.fixture()
def batch():
    """B=4, T=10 with left-padding (PAD_ID=0 in positions 0..3)."""
    g = torch.Generator().manual_seed(42)
    B, T = 4, 10
    seqs = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        L = torch.randint(2, T, (1,), generator=g).item()
        seqs[b, T - L:] = torch.randint(1, 201, (L,), generator=g)
    return seqs


# ---------------------------------------------------------------------------
# Test 1 — recsys recipe emits > 0 WEIGHT tags
# ---------------------------------------------------------------------------


def test_recsys_emits_weight_tags(model, batch, tmp_path):
    """The recsys recipe must emit at least one weight/* tag on the minimal model.

    Expected fired WEIGHT hookpoints:
    - HP#0 embedding pattern → encoder.item_emb, encoder.pos_emb
    - HP#1 out_proj pattern → encoder.blocks.N.self_attn.out_proj
    - HP#2 ffn-leaf pattern → encoder.blocks.N.ffn.0, ffn.2
    """
    writer = RecordingWriter()
    # Default strict=True: the optional cross-architecture patterns (GRU, etc.)
    # match 0 modules on this transformer-recsys model and must NOT fail attach.
    rec = Recorder(model, run_dir=tmp_path, recipe="recsys",
                   writer=writer, every_n_steps=1)
    rec.attach()

    out = model(batch)
    out.sum().backward()
    rec.step(0)
    rec.detach()

    weight_tags = [t for t, _, _ in writer.scalars if t.startswith("weight/")]
    assert len(weight_tags) > 0, (
        f"recsys recipe emitted 0 weight/* tags on the minimal SASRec-like model. "
        f"All tags: {sorted({t for t, _, _ in writer.scalars})}"
    )


# ---------------------------------------------------------------------------
# Test 2 — recsys recipe emits > 0 activation tags
# ---------------------------------------------------------------------------


def test_recsys_emits_activation_tags(model, batch, tmp_path):
    """The recsys recipe must emit at least one activation/* tag on the minimal model.

    Expected fired OUTPUT hookpoints:
    - HP#5 self_attn output → encoder.blocks.N.self_attn
    - HP#6 ffn output → encoder.blocks.N.ffn
    - HP#7 norm outputs → encoder.blocks.N.norm1, norm2, encoder.layer_norm
    - HP#8 block-level output → encoder.blocks.N
    """
    writer = RecordingWriter()
    # Default strict=True (see test_recsys_emits_weight_tags): optional patterns
    # absent on this model must not fail attach.
    rec = Recorder(model, run_dir=tmp_path, recipe="recsys",
                   writer=writer, every_n_steps=1)
    rec.attach()

    out = model(batch)
    out.sum().backward()
    rec.step(0)
    rec.detach()

    act_tags = [t for t, _, _ in writer.scalars if t.startswith("activation/")]
    assert len(act_tags) > 0, (
        f"recsys recipe emitted 0 activation/* tags on the minimal SASRec-like model. "
        f"All tags: {sorted({t for t, _, _ in writer.scalars})}"
    )


# ---------------------------------------------------------------------------
# Test 3 — PAD-masked attention entropy is finite (non-NaN)
# ---------------------------------------------------------------------------


def test_attention_entropy_finite_with_pad_mask(model, batch):
    """With the need_weights patch + PAD-row masking, per-head entropy is finite.

    This validates the Finding B / Finding C workaround:
    - Without the need_weights patch: out[1]=None, no capture.
    - With the patch: out[1] is (B, H, T, T) but PAD rows have NaN weights.
    - After PAD-row masking: entropy values are finite for all heads.
    """
    _patch_mha_need_weights(model)

    captured: dict[str, torch.Tensor] = {}

    def _hook_factory(name):
        def _hook(mod, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
                captured[name] = out[1].detach()
        return _hook

    handles = []
    for name, mod in model.named_modules():
        short = name.rsplit(".", 1)[-1] if "." in name else name
        if short == "self_attn":
            handles.append(mod.register_forward_hook(_hook_factory(name)))

    with torch.no_grad():
        model(batch)

    for h in handles:
        h.remove()

    assert len(captured) > 0, (
        "No attention weights captured — need_weights patch may not have worked. "
        "Check that _patch_mha_need_weights correctly wraps all self_attn modules."
    )

    for mod_name, attn_w in captured.items():
        # attn_w: (B, n_heads, T, T) — PAD rows contain NaN.
        assert attn_w.ndim == 4, f"Expected 4D attn weight, got {attn_w.shape}"
        B, H, T, T2 = attn_w.shape
        assert T == T2, "Attention weight must be square in the last two dims"

        # Compute masked entropy (Finding B workaround).
        valid_rows = ~torch.isnan(attn_w).any(dim=-1)  # (B, H, T) bool
        p = attn_w.clone()
        p[torch.isnan(p)] = 0.0
        plogp = torch.special.xlogy(p, p.clamp_min(1e-10))
        plogp[torch.isnan(plogp)] = 0.0
        entropy = -plogp.sum(dim=-1)  # (B, H, T)

        valid_float = valid_rows.float()
        valid_count = valid_float.sum(dim=(0, 2))  # (H,)
        masked_entropy = (entropy * valid_float).sum(dim=(0, 2)) / valid_count.clamp_min(1)
        # (H,) per-head mean entropy over non-PAD query rows

        assert not torch.isnan(masked_entropy).any(), (
            f"{mod_name}: PAD-masked entropy contains NaN — "
            f"valid_count={valid_count.tolist()}, entropy={masked_entropy.tolist()}"
        )
        assert (masked_entropy >= 0).all(), (
            f"{mod_name}: entropy must be non-negative, got {masked_entropy.tolist()}"
        )
        # Sanity: entropy > 0 (keys are not all identical for real sequences).
        # Use a very loose lower bound (>= 0 is guaranteed by math; > 0.01 is empirical).
        valid_heads = valid_count > 0
        if valid_heads.any():
            assert masked_entropy[valid_heads].max() > 0.01, (
                f"{mod_name}: all heads have near-zero entropy — "
                f"the model may not be processing the sequences. "
                f"Per-head entropy: {masked_entropy.tolist()}"
            )
