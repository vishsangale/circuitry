"""Tests for ReLPRunner. v1.32."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.eap import EAPResult, EAPRunner
from circuitry.patching.graph import Edge, Node, build_graph
from circuitry.patching.relp import ReLPRunner, _score_edges_relp


# ---------------------------------------------------------------------------
# Minimal 2-layer flat toy model (same shape as test_eap_mlp_exact.py)
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    """Embed → 2 MLP-only layers → lm_head. No attention."""

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


def _make_runner(n_layers=2, d=8, vocab=32):
    torch.manual_seed(0)
    model = _TinyMLP(d=d, vocab=vocab, n_layers=n_layers)
    # Pass resolver=None for MLP-only toy (no attention heads)
    return model, ReLPRunner(model, resolver=None)


# ---------------------------------------------------------------------------
# _score_edges_relp unit tests
# ---------------------------------------------------------------------------

def test_score_edges_relp_returns_eap_result():
    torch.manual_seed(0)
    g = build_graph(n_layers=1, n_heads=0)
    b, p, W, R, d = 2, 3, len(g.writers), len(g.readers), 4
    act_clean = torch.randn(b, p, W, d)
    act_corrupted = torch.randn(b, p, W, d)
    grad = torch.randn(b, p, R, d)
    result = _score_edges_relp(g, act_clean, act_corrupted, grad)
    assert isinstance(result, EAPResult)
    assert set(result.scores.keys()) == set(g.edges)


def test_score_edges_relp_zero_delta_gives_zero():
    """When clean == corrupted, delta == 0, so all scores must be zero."""
    torch.manual_seed(1)
    g = build_graph(n_layers=1, n_heads=0)
    b, p, W, R, d = 2, 3, len(g.writers), len(g.readers), 4
    act = torch.randn(b, p, W, d)
    grad = torch.randn(b, p, R, d)
    result = _score_edges_relp(g, act, act.clone(), grad)
    for score in result.scores.values():
        assert score == pytest.approx(0.0, abs=1e-6)


def test_score_edges_relp_differs_from_eap():
    """RelP scores should differ from vanilla EAP on a non-degenerate graph."""
    from circuitry.patching.eap import score_edges
    torch.manual_seed(2)
    g = build_graph(n_layers=2, n_heads=0)
    b, p, W, R, d = 2, 4, len(g.writers), len(g.readers), 8
    act_clean = torch.randn(b, p, W, d)
    act_corr = torch.randn(b, p, W, d)
    grad = torch.randn(b, p, R, d)
    eap_result = score_edges(g, act_clean, act_corr, grad)
    relp_result = _score_edges_relp(g, act_clean, act_corr, grad)
    # At least one edge should differ
    diffs = [abs(eap_result.scores[e] - relp_result.scores[e]) for e in g.edges]
    assert max(diffs) > 1e-4, "RelP and EAP should differ on a non-degenerate input"


def test_score_edges_relp_uniform_positive_writers():
    """When all writers contribute equal POSITIVE activations, lrp_coeff = 1/W
    and RelP score == EAP score / W."""
    torch.manual_seed(3)
    g = build_graph(n_layers=1, n_heads=0)
    b, p, W, R, d = 2, 3, len(g.writers), len(g.readers), 4
    # All writers contribute equally and positively: act_clean[w] = +ones
    act_clean = torch.ones(b, p, W, d)     # all +1 → x_clean = W, lrp_coeff = 1/W
    act_corr = torch.randn(b, p, W, d)
    grad = torch.randn(b, p, R, d)

    from circuitry.patching.eap import score_edges
    eap_result = score_edges(g, act_clean, act_corr, grad)
    relp_result = _score_edges_relp(g, act_clean, act_corr, grad, eps=0.0)

    # lrp_coeff = 1 / W exactly (no eps needed since x_clean = W > 0)
    for e in g.edges:
        assert relp_result.scores[e] == pytest.approx(
            eap_result.scores[e] / W, rel=1e-3
        )


# ---------------------------------------------------------------------------
# ReLPRunner integration tests
# ---------------------------------------------------------------------------

def test_relp_runner_run_returns_eap_result():
    torch.manual_seed(0)
    model, runner = _make_runner()
    clean = torch.randint(0, 32, (2, 4))
    corr = torch.randint(0, 32, (2, 4))
    result = runner.run(clean, corr, _metric)
    assert isinstance(result, EAPResult)


def test_relp_runner_scores_all_edges():
    torch.manual_seed(1)
    model, runner = _make_runner()
    clean = torch.randint(0, 32, (2, 4))
    corr = torch.randint(0, 32, (2, 4))
    result = runner.run(clean, corr, _metric)
    assert len(result.scores) == len(runner.graph.edges)


def test_relp_runner_ranked_and_top_k():
    torch.manual_seed(2)
    model, runner = _make_runner()
    clean = torch.randint(0, 32, (2, 4))
    corr = torch.randint(0, 32, (2, 4))
    result = runner.run(clean, corr, _metric)
    k = 3
    top = result.top_k(k)
    assert len(top) == k
    # Should be sorted by |score| descending
    scores = [abs(s) for _, s in top]
    assert scores == sorted(scores, reverse=True)


def test_relp_runner_differs_from_eap():
    """ReLPRunner scores should differ from EAPRunner on the same inputs."""
    torch.manual_seed(3)
    model = _TinyMLP(d=8, vocab=32, n_layers=2)
    eap_runner = EAPRunner(model, resolver=None)
    relp_runner = ReLPRunner(model, resolver=None)
    clean = torch.randint(0, 32, (4, 6))
    corr = torch.randint(0, 32, (4, 6))
    eap_result = eap_runner.run(clean, corr, _metric)
    relp_result = relp_runner.run(clean, corr, _metric)
    diffs = [abs(eap_result.scores[e] - relp_result.scores[e]) for e in eap_runner.graph.edges]
    assert max(diffs) > 1e-4


def test_relp_runner_same_graph_as_eap():
    torch.manual_seed(4)
    model = _TinyMLP(d=8, vocab=32, n_layers=2)
    relp_runner = ReLPRunner(model, resolver=None)
    eap_runner = EAPRunner(model, resolver=None)
    assert set(relp_runner.graph.edges) == set(eap_runner.graph.edges)


def test_relp_runner_model_back_to_eval():
    """Model must remain in eval mode (not training) after run()."""
    torch.manual_seed(5)
    model, runner = _make_runner()
    model.eval()
    clean = torch.randint(0, 32, (2, 4))
    corr = torch.randint(0, 32, (2, 4))
    runner.run(clean, corr, _metric)
    assert not model.training
