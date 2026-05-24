"""Shared toy models for patching tests."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class ToyPatchModel(nn.Module):
    """Two-layer identity model. Patching layer 0 output → output changes to
    layer1(patched_value). With identity weights: output == input normally,
    output == patched_value when layer 0 output is patched."""

    def __init__(self, d: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=False),
            nn.Linear(d, d, bias=False),
        ])
        for layer in self.layers:
            nn.init.eye_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FakeAttention(nn.Module):
    """Identity attention with o_proj for head-slicing tests."""

    def __init__(self, d_model: int):
        super().__init__()
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.o_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class FakeMLP(nn.Module):
    """Identity MLP with gate_proj + down_proj for neuron-slicing tests."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(x))


class FakeTransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.self_attn = FakeAttention(d_model)
        self.mlp = FakeMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class FakeTransformerModel(nn.Module):
    """Llama-like module structure: model.layers.{L}.self_attn.o_proj,
    model.layers.{L}.mlp.down_proj. NOT an HF model — just has matching names."""

    def __init__(self, n_layers: int = 2, d_model: int = 8, n_heads: int = 2,
                 d_mlp: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            FakeTransformerLayer(d_model, d_mlp) for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def toy_model():
    return ToyPatchModel(d=4)


@pytest.fixture
def transformer_model():
    return FakeTransformerModel(n_layers=2, d_model=8, n_heads=2, d_mlp=16)


class LinearMLPToy(nn.Module):
    """Attention-free linear residual stack with HF-like names. Fully linear:
    EAP's first-order score is EXACT here, so it must equal brute-force patching."""

    def __init__(self, n_layers=2, d=4, vocab=5):
        super().__init__()
        self.n_layers, self.d, self.vocab = n_layers, d, vocab
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.Module()
            block.mlp = nn.Module()
            block.mlp.up_proj = nn.Linear(d, d, bias=False)
            block.mlp.down_proj = nn.Linear(d, d, bias=False)
            self.layers.append(block)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)            # (b, s, d) — embed writer output
        for block in self.layers:
            mlp_out = block.mlp.down_proj(block.mlp.up_proj(x))   # linear, no activation
            x = x + mlp_out                          # residual write
        return self.lm_head(x)                       # (b, s, vocab)


@pytest.fixture
def linear_mlp_toy():
    torch.manual_seed(0)
    return LinearMLPToy(n_layers=2, d=4, vocab=5)


class LinearAttnToy(nn.Module):
    """Linear transformer with a FIXED, input-independent attention pattern, so
    the whole model is linear and the EAP per-edge exact gate holds. q/k are dead
    (fixed pattern) — their reader gradients are ~0; the v path carries signal."""
    def __init__(self, n_layers=2, n_heads=2, d=4, vocab=5, seq=4):
        super().__init__()
        self.n_layers, self.n_heads, self.d, self.vocab = n_layers, n_heads, d, vocab
        self.head_dim = d // n_heads
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            attn = nn.Module()
            attn.q_proj = nn.Linear(d, d, bias=False)
            attn.k_proj = nn.Linear(d, d, bias=False)
            attn.v_proj = nn.Linear(d, d, bias=False)
            attn.o_proj = nn.Linear(d, d, bias=False)
            layer.self_attn = attn
            mlp = nn.Module()
            mlp.up_proj = nn.Linear(d, d, bias=False)
            mlp.down_proj = nn.Linear(d, d, bias=False)
            layer.mlp = mlp
            self.layers.append(layer)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        pat = torch.tril(torch.ones(seq, seq))
        self.register_buffer("attn_pattern", pat / pat.sum(-1, keepdim=True))
        torch.manual_seed(0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.4)

    def forward(self, input_ids):
        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            v = layer.self_attn.v_proj(x).view(b, s, self.n_heads, self.head_dim)
            attended = torch.einsum("st,bthd->bshd", self.attn_pattern[:s, :s], v)
            attn_out = layer.self_attn.o_proj(attended.reshape(b, s, self.d))
            x = x + attn_out
            x = x + layer.mlp.down_proj(layer.mlp.up_proj(x))
        return self.lm_head(x)


@pytest.fixture
def linear_attn_toy():
    return LinearAttnToy(n_layers=2, n_heads=2, d=4, vocab=5, seq=4)
