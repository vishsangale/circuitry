"""Regression test for the v1.8.0 dense-model strict-attach bug (F37 root cause).

v1.8.0 added MoE-only weight HookPoints to the stock ``llm`` recipe
(``.*\\.mlp\\.gate$`` and ``.*\\.mlp\\.experts$``). On any *dense* (non-MoE)
model these match 0 modules, and ``Recorder.attach()`` with the default
``strict=True`` RAISED — so live capture on a plain Llama/GPT-2 model failed
out of the box with the stock recipe.

Surfaced by an external evaluation of 67 custom dense 1M-param LMs
(``FEEDBACK-2026-06-01-leaderboard-fingerprint.md`` #7).

Fix: HookPoints carry an ``optional`` flag; a 0-match on an optional HookPoint
is a soft skip even under ``strict=True``. The MoE patterns are marked optional,
so a dense model attaches cleanly while genuinely-missing *required* patterns
still raise under strict.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import _register_stock_recipes, get_recipe
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _ensure_stock_recipes():
    """Other tests clear the recipe registry without restoring it; make these
    tests independent of ambient state. ``_register_stock_recipes`` is idempotent
    (skips already-registered names), so this is safe regardless of prior order."""
    _register_stock_recipes()

# ---------------------------------------------------------------------------
# Minimal DENSE Llama-shaped model: matches every *required* llm-recipe pattern
# (q/k/v/o_proj, gate/up/down_proj, self_attn, mlp, layernorms, embed, lm_head,
# layers.N) but has NO mlp.gate / mlp.experts (it is not MoE).
# ---------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.v_proj(x))


class _MLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d)
        self.up_proj = nn.Linear(d, d)
        self.down_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d)
        self.input_layernorm = nn.LayerNorm(d)
        self.post_attention_layernorm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _Inner(nn.Module):
    """Decoder stack — named ``model.embed_tokens`` / ``model.layers.N`` /
    ``model.norm`` so the recipe's ``.*\\.layers\\.\\d+$`` pattern matches (as it
    does on a real ``LlamaForCausalLM``, where the stack lives under ``.model``)."""

    def __init__(self, vocab: int, d: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Layer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(ids)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class _DenseLM(nn.Module):
    """Dense (non-MoE) decoder with HF-Llama module naming (``model.*`` + ``lm_head``)."""

    def __init__(self, vocab: int = 16, d: int = 16, n_layers: int = 1) -> None:
        super().__init__()
        self.model = _Inner(vocab, d, n_layers)
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(ids))


def test_dense_model_attaches_under_strict_default(tmp_path):
    """A dense model + stock ``llm`` recipe must attach under the DEFAULT strict=True.

    Pre-fix this raised: ``RuntimeError: HookPoint 2 (.*\\.mlp\\.gate$) matched 0
    modules — refusing to attach``.
    """
    model = _DenseLM()
    writer = RecordingWriter()
    # No strict= kwarg => uses the default (strict=True). That is the regression.
    rec = Recorder(model, run_dir=tmp_path, recipe="llm", writer=writer, every_n_steps=1)

    rec.attach()  # must NOT raise
    ids = torch.randint(0, 16, (2, 8))
    model(ids).sum().backward()
    rec.step(0)
    rec.detach()

    # The dense weight diagnostics should have produced output (proves we didn't
    # just skip everything).
    weight_tags = [t for t, _, _ in writer.scalars if t.startswith("weight/")]
    assert weight_tags, "dense model attached but emitted no weight diagnostics"


def test_required_pattern_zero_match_still_raises_under_strict(tmp_path):
    """``optional`` must not neuter strict: a *required* 0-match still raises.

    A bare model that matches none of the llm recipe's required patterns must
    still fail strict attach — otherwise the optional flag would silently mask
    genuinely-misconfigured recipes.
    """
    bare = nn.Sequential(nn.Linear(4, 4))  # no q/k/v/o_proj, no embed, etc.
    rec = Recorder(bare, run_dir=tmp_path, recipe="llm", writer=RecordingWriter())
    with pytest.raises(RuntimeError, match="matched 0 modules"):
        rec.attach()


def test_moe_hookpoints_are_marked_optional():
    """The MoE-only weight patterns in the stock llm recipe must be optional."""
    recipe = get_recipe("llm")
    moe_patterns = {r".*\.mlp\.gate$", r".*\.mlp\.experts$"}
    by_pattern = {hp.pattern: hp for hp in recipe.hook_points if hp.pattern in moe_patterns}
    assert moe_patterns <= set(by_pattern), (
        f"expected MoE patterns {moe_patterns} in the llm recipe, got {set(by_pattern)}"
    )
    for pat, hp in by_pattern.items():
        assert hp.optional, f"MoE pattern {pat!r} must be marked optional=True"
