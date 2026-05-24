"""Tests for gradient-space diagnostics."""

from __future__ import annotations

import pytest
import torch

from circuitry.core import gradient


def test_grad_norm_per_module_returns_dict_of_floats():
    grads = {
        "layer0.weight": torch.ones(3, 3),
        "layer1.weight": torch.zeros(3, 3),
    }
    out = gradient.grad_norm_per_module(grads)
    assert set(out) == {"layer0.weight", "layer1.weight"}
    assert all(isinstance(v, float) for v in out.values())
    assert out["layer0.weight"] == pytest.approx(3.0)  # frobenius norm of ones(3,3)
    assert out["layer1.weight"] == pytest.approx(0.0)


def test_grad_norm_per_module_empty_dict():
    assert gradient.grad_norm_per_module({}) == {}


def test_grad_norm_per_module_skips_none():
    out = gradient.grad_norm_per_module({"a": torch.ones(2, 2), "b": None})
    assert set(out) == {"a"}
    assert out["a"] == pytest.approx(2.0)


def test_grad_norm_per_module_computes_in_float32():
    # bf16 gradient: float32 reduction is more accurate than a bf16 reduction.
    g = torch.ones(256, 256, dtype=torch.bfloat16)
    out = gradient.grad_norm_per_module({"w": g})
    assert out["w"] == pytest.approx(256.0, rel=1e-3)  # sqrt(256*256) = 256


def test_total_grad_norm_sqrt_sum_of_squares():
    assert gradient.total_grad_norm({"a": 3.0, "b": 4.0}) == pytest.approx(5.0)


def test_total_grad_norm_empty():
    assert gradient.total_grad_norm({}) == 0.0


def test_total_grad_norm_composes_with_per_module():
    grads = {"x.weight": torch.ones(3, 3), "y.weight": torch.full((4,), 2.0)}
    per = gradient.grad_norm_per_module(grads)
    # ||ones(3,3)|| = 3, ||[2,2,2,2]|| = 4 → total = 5
    assert gradient.total_grad_norm(per) == pytest.approx(5.0)


def test_signal_propagation_depth_all_alive():
    # Norms decreasing but all above eps → reaches max depth.
    grads = [torch.full((4,), v) for v in (1.0, 0.5, 0.25, 0.1)]
    assert gradient.signal_propagation_depth(grads) == 4


def test_signal_propagation_depth_vanishing():
    grads = [torch.ones(4), torch.full((4,), 1e-2), torch.zeros(4), torch.zeros(4)]
    # eps_ratio default 1e-3 of first layer norm → 2.0 cutoff; layer 1 = 2e-2 > 2e-3 → alive; layer 2 = 0 → dead.
    assert gradient.signal_propagation_depth(grads) == 2
