"""Tests for gradient-space diagnostics."""

from __future__ import annotations

import pytest
import torch

from circuitry.core import gradient


def test_layer_norm_returns_dict_of_floats():
    grads = {
        "layer0.weight": torch.ones(3, 3),
        "layer1.weight": torch.zeros(3, 3),
    }
    out = gradient.layer_norm(grads)
    assert set(out) == {"layer0.weight", "layer1.weight"}
    assert all(isinstance(v, float) for v in out.values())
    assert out["layer0.weight"] == pytest.approx(3.0)  # frobenius norm of ones(3,3)
    assert out["layer1.weight"] == pytest.approx(0.0)


def test_layer_norm_empty_dict():
    assert gradient.layer_norm({}) == {}


def test_signal_propagation_depth_all_alive():
    # Norms decreasing but all above eps → reaches max depth.
    grads = [torch.full((4,), v) for v in (1.0, 0.5, 0.25, 0.1)]
    assert gradient.signal_propagation_depth(grads) == 4


def test_signal_propagation_depth_vanishing():
    grads = [torch.ones(4), torch.full((4,), 1e-2), torch.zeros(4), torch.zeros(4)]
    # eps_ratio default 1e-3 of first layer norm → 2.0 cutoff; layer 1 = 2e-2 > 2e-3 → alive; layer 2 = 0 → dead.
    assert gradient.signal_propagation_depth(grads) == 2
