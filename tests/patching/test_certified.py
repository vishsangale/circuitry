"""Tests for CertifiedCircuitRunner and CertifiedCircuitResult. v1.32."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.certified import (
    CertifiedCircuitResult,
    CertifiedCircuitRunner,
    _batch_size,
    _index_inputs,
)
from circuitry.patching.eap import EAPRunner
from circuitry.patching.graph import Edge, Node, build_graph


# ---------------------------------------------------------------------------
# Minimal flat toy model
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    class _Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.mlp = nn.ModuleDict(
                {"up_proj": nn.Linear(d, d * 2, bias=False),
                 "down_proj": nn.Linear(d * 2, d, bias=False)}
            )

        def forward(self, x: Tensor) -> Tensor:
            return x + self.mlp["down_proj"](torch.relu(self.mlp["up_proj"](x)))

    def __init__(self, d: int = 8, vocab: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([self._Block(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


def _metric(logits: Tensor) -> Tensor:
    return logits[:, -1, 0].mean()


def _make_eap_runner(n_layers=2, d=8, vocab=32):
    torch.manual_seed(0)
    model = _TinyMLP(d=d, vocab=vocab, n_layers=n_layers)
    return model, EAPRunner(model, resolver=None)


# ---------------------------------------------------------------------------
# _batch_size / _index_inputs helpers
# ---------------------------------------------------------------------------

def test_batch_size_tensor():
    x = torch.randn(5, 4, 3)
    assert _batch_size(x) == 5


def test_batch_size_dict():
    d = {"input_ids": torch.randint(0, 10, (7, 4)), "attention_mask": torch.ones(7, 4)}
    assert _batch_size(d) == 7


def test_index_inputs_tensor():
    x = torch.arange(20).reshape(4, 5)
    idx = torch.tensor([0, 2])
    out = _index_inputs(x, idx)
    assert out.shape == (2, 5)
    assert torch.equal(out[0], x[0])
    assert torch.equal(out[1], x[2])


def test_index_inputs_dict():
    d = {"input_ids": torch.arange(20).reshape(4, 5)}
    idx = torch.tensor([1, 3])
    out = _index_inputs(d, idx)
    assert isinstance(out, dict)
    assert torch.equal(out["input_ids"][0], d["input_ids"][1])


# ---------------------------------------------------------------------------
# CertifiedCircuitRunner
# ---------------------------------------------------------------------------

def test_certified_runner_returns_result():
    _, eap = _make_eap_runner()
    runner = CertifiedCircuitRunner(eap, n_subsamples=5, seed=0)
    clean = torch.randint(0, 32, (8, 4))
    corr = torch.randint(0, 32, (8, 4))
    result = runner.run(clean, corr, _metric, top_k=5)
    assert isinstance(result, CertifiedCircuitResult)


def test_certified_result_fields():
    _, eap = _make_eap_runner()
    runner = CertifiedCircuitRunner(eap, n_subsamples=4, confidence=0.5, seed=1)
    clean = torch.randint(0, 32, (8, 4))
    corr = torch.randint(0, 32, (8, 4))
    result = runner.run(clean, corr, _metric, top_k=4)
    assert result.n_subsamples == 4
    assert result.confidence == 0.5
    assert result.top_k == 4
    assert isinstance(result.vote_counts, dict)


def test_certified_edges_plus_abstained_equals_all_voted():
    _, eap = _make_eap_runner()
    runner = CertifiedCircuitRunner(eap, n_subsamples=6, confidence=0.5, seed=2)
    clean = torch.randint(0, 32, (8, 4))
    corr = torch.randint(0, 32, (8, 4))
    result = runner.run(clean, corr, _metric, top_k=3)
    total = len(result.certified_edges) + len(result.abstained_edges)
    assert total == len(result.vote_counts)


def test_certified_confidence_1_requires_all_votes():
    """confidence=1.0 means all n_subsamples must vote for an edge to be certified."""
    torch.manual_seed(3)
    _, eap = _make_eap_runner()
    n_sub = 5
    runner = CertifiedCircuitRunner(eap, n_subsamples=n_sub, confidence=1.0, seed=3)
    clean = torch.randint(0, 32, (8, 4))
    corr = torch.randint(0, 32, (8, 4))
    result = runner.run(clean, corr, _metric, top_k=2)
    for e in result.certified_edges:
        assert result.vote_counts[e] >= n_sub


def test_certified_vote_counts_bounded_by_n_subsamples():
    _, eap = _make_eap_runner()
    n_sub = 8
    runner = CertifiedCircuitRunner(eap, n_subsamples=n_sub, seed=4)
    clean = torch.randint(0, 32, (10, 4))
    corr = torch.randint(0, 32, (10, 4))
    result = runner.run(clean, corr, _metric, top_k=3)
    for e, v in result.vote_counts.items():
        assert 0 <= v <= n_sub


def test_certified_set_is_set_of_certified_edges():
    _, eap = _make_eap_runner()
    runner = CertifiedCircuitRunner(eap, n_subsamples=4, seed=5)
    clean = torch.randint(0, 32, (8, 4))
    corr = torch.randint(0, 32, (8, 4))
    result = runner.run(clean, corr, _metric, top_k=3)
    assert result.certified_set() == set(result.certified_edges)


def test_certified_runner_invalid_confidence_raises():
    with pytest.raises(ValueError, match="confidence"):
        CertifiedCircuitRunner(None, confidence=1.5)


def test_certified_runner_invalid_subsample_frac_raises():
    with pytest.raises(ValueError, match="subsample_frac"):
        CertifiedCircuitRunner(None, subsample_frac=0.0)


def test_certified_runner_invalid_n_subsamples_raises():
    with pytest.raises(ValueError, match="n_subsamples"):
        CertifiedCircuitRunner(None, n_subsamples=0)


def test_certified_n_certified_and_n_abstained():
    _, eap = _make_eap_runner()
    runner = CertifiedCircuitRunner(eap, n_subsamples=4, seed=6)
    clean = torch.randint(0, 32, (8, 4))
    corr = torch.randint(0, 32, (8, 4))
    result = runner.run(clean, corr, _metric, top_k=3)
    assert result.n_certified() == len(result.certified_edges)
    assert result.n_abstained() == len(result.abstained_edges)
