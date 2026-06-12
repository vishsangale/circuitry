"""Tests for CLTGraphRunner and CLTGraphResult. v1.34."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.clt import CLTGraphResult, CLTGraphRunner

# ---------------------------------------------------------------------------
# Toy model and helpers
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

    def __init__(self, d: int = 8, vocab: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([self._Block(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


class _LinearTC:
    """Minimal transcoder: encode=ReLU(x @ W_enc), decode=f @ W_dec."""
    def __init__(self, d_model: int, d_feat: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W_enc = torch.randn(d_model, d_feat, generator=g) * 0.1
        self.W_dec = torch.randn(d_feat, d_model, generator=g) * 0.1
    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc)
    def decode(self, f: Tensor) -> Tensor:
        return f @ self.W_dec


def _metric(logits: Tensor) -> Tensor:
    return logits[:, -1, 0].mean()


def _make_runner(n_layers=3, d=8, vocab=32, d_feat=4):
    torch.manual_seed(0)
    model = _TinyMLP(d=d, vocab=vocab, n_layers=n_layers)
    tcs = {i: _LinearTC(d, d_feat, seed=i) for i in range(n_layers)}
    return model, CLTGraphRunner(model, tcs)


def _make_inputs(batch=2, seq=4, vocab=32):
    clean = torch.randint(0, vocab, (batch, seq))
    corr = torch.randint(0, vocab, (batch, seq))
    return clean, corr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clt_returns_result():
    model, runner = _make_runner()
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert isinstance(result, CLTGraphResult)


def test_clt_result_has_scores_dict():
    model, runner = _make_runner()
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert isinstance(result.scores, dict)


def test_clt_result_has_node_scores():
    model, runner = _make_runner()
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert isinstance(result.node_scores, dict)


def test_clt_n_layers_correct():
    n_layers = 3
    model, runner = _make_runner(n_layers=n_layers)
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert result.n_layers == n_layers


def test_clt_n_features_matches_transcoder():
    n_layers = 3
    d_feat = 4
    model, runner = _make_runner(n_layers=n_layers, d_feat=d_feat)
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert result.n_features == [d_feat] * n_layers


def test_clt_layer_order_is_sorted():
    model, runner = _make_runner()
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert result.layer_order == sorted(result.layer_order)


def test_clt_edges_connect_consecutive_layers():
    model, runner = _make_runner()
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    for edge in result.scores:
        assert edge.dst.layer == edge.src.layer + 1


def test_clt_top_k():
    model, runner = _make_runner()
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    top3 = result.top_k(3)
    assert len(top3) == 3
    scores = [abs(s) for _, s in top3]
    assert scores == sorted(scores, reverse=True)


def test_clt_threshold():
    model, runner = _make_runner()
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert len(result.threshold(0.0)) == len(result.scores)
    assert result.threshold(1e9) == []


def test_clt_zero_delta_gives_zero_scores():
    model, runner = _make_runner()
    clean = corr = torch.randint(0, 32, (2, 4))
    result = runner.run(clean, corr, _metric)
    for edge, score in result.scores.items():
        assert abs(score) < 1e-5, f"Expected ~0 score for {edge}, got {score}"


def test_clt_nonzero_scores_on_different_inputs():
    model, runner = _make_runner()
    torch.manual_seed(42)
    clean, corr = _make_inputs()
    result = runner.run(clean, corr, _metric)
    assert any(abs(s) > 1e-8 for s in result.scores.values()), (
        "Expected at least one nonzero edge score for different clean/corrupted inputs"
    )


def test_clt_model_back_to_eval():
    model, runner = _make_runner()
    model.eval()
    clean, corr = _make_inputs()
    runner.run(clean, corr, _metric)
    assert not model.training
