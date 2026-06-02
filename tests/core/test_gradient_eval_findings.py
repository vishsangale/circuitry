"""Regression test for the v1.7 real-model evaluation finding in ``core/gradient``.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md`` (F31).

RED under current code; flips GREEN once fixed. Marked ``xfail(strict=True)``.
Reproduce with ``--runxfail``.
"""

from __future__ import annotations

import math

import pytest
import torch

from circuitry.core import gradient

# ---------------------------------------------------------------------------
# F31 — grad_norm_per_module crashes (NotImplementedError, SparseCPU backend)
#       on sparse nn.Embedding gradients. This is the standard recsys embedding
#       setup (sparse=True). torch.linalg.vector_norm has no sparse kernel.
#       Fix: g = g.to_dense() if g.is_sparse else g  before the norm.
# ---------------------------------------------------------------------------


def test_F31_grad_norm_per_module_handles_sparse_embedding_grad():
    emb = torch.nn.Embedding(1000, 16, sparse=True)
    idx = torch.randint(0, 1000, (32,))
    emb(idx).sum().backward()
    g = emb.weight.grad
    assert g.is_sparse, "precondition: sparse=True embedding must produce a sparse grad"

    out = gradient.grad_norm_per_module({"emb.weight": g})

    assert "emb.weight" in out
    assert math.isfinite(out["emb.weight"])
    # The L2 norm must match the dense equivalent.
    expected = float(torch.linalg.vector_norm(g.to_dense()).item())
    assert out["emb.weight"] == pytest.approx(expected, rel=1e-5)
