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
