"""Tests for gate_stats — post-gate MLP activation statistics."""
from __future__ import annotations

import torch

from circuitry.core.activation import gate_stats


def test_gate_stats_all_active():
    x = torch.full((4, 8), 0.5)
    out = gate_stats(x)
    assert out["frac_active"] == 1.0
    assert abs(out["mean_abs"] - 0.5) < 1e-6
    assert abs(out["std"]) < 1e-6


def test_gate_stats_all_inactive():
    x = torch.zeros(4, 8)
    out = gate_stats(x)
    assert out["frac_active"] == 0.0
    assert out["mean_abs"] == 0.0


def test_gate_stats_half_active():
    x = torch.zeros(2, 4)
    x[:, :2] = 1.0  # half the channels active at value 1
    out = gate_stats(x)
    assert out["frac_active"] == 0.5
    assert abs(out["mean_abs"] - 0.5) < 1e-6


def test_gate_stats_respects_eps():
    x = torch.full((4,), 1e-7)
    out_default = gate_stats(x)
    out_loose = gate_stats(x, eps=1e-8)
    assert out_default["frac_active"] == 0.0
    assert out_loose["frac_active"] == 1.0


def test_gate_stats_handles_negative_values():
    # frac_active counts |x| > eps, sign agnostic.
    x = torch.tensor([-1.0, 1.0, 0.0, 0.0])
    out = gate_stats(x)
    assert out["frac_active"] == 0.5
