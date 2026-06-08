"""Tests for circuitry.core.probe — LinearProbe and train_linear_probe."""
from __future__ import annotations

import torch
import pytest

from circuitry.core.probe import LinearProbe, train_linear_probe


def _binary_separable_data(n: int = 100, d: int = 16, seed: int = 0):
    """Class 0 at mean=-3, class 1 at mean=+3 along dim 0."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    acts0 = torch.randn(n // 2, d, generator=rng) - 3.0
    acts1 = torch.randn(n // 2, d, generator=rng) + 3.0
    acts = torch.cat([acts0, acts1], dim=0)
    labels = torch.cat([torch.zeros(n // 2, dtype=torch.long),
                        torch.ones(n // 2, dtype=torch.long)])
    return acts, labels


def test_train_linear_probe_binary_separable():
    acts, labels = _binary_separable_data()
    probe = train_linear_probe(acts, labels, max_iter=500)
    acc = probe.accuracy(acts, labels)
    assert acc >= 0.95, f"Expected near-perfect accuracy, got {acc:.3f}"


def test_train_linear_probe_random_labels():
    rng = torch.Generator()
    rng.manual_seed(42)
    acts = torch.randn(200, 32, generator=rng)
    labels = torch.randint(0, 2, (200,), generator=rng)
    probe = train_linear_probe(acts, labels, max_iter=200)
    acc = probe.accuracy(acts, labels)
    assert acc < 0.75, f"Expected near-chance accuracy on random labels, got {acc:.3f}"


def test_train_linear_probe_multiclass():
    """3 classes, linearly separable; accuracy > 0.9."""
    rng = torch.Generator()
    rng.manual_seed(7)
    n, d = 60, 16
    # Each class separated by large offset on a different dimension
    acts0 = torch.randn(n, d, generator=rng); acts0[:, 0] += 5.0
    acts1 = torch.randn(n, d, generator=rng); acts1[:, 1] += 5.0
    acts2 = torch.randn(n, d, generator=rng); acts2[:, 2] += 5.0
    acts = torch.cat([acts0, acts1, acts2], dim=0)
    labels = torch.cat([
        torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.long),
        torch.full((n,), 2, dtype=torch.long),
    ])
    probe = train_linear_probe(acts, labels, max_iter=1000)
    acc = probe.accuracy(acts, labels)
    assert acc > 0.9, f"Expected >90% accuracy on separable 3-class data, got {acc:.3f}"


def test_linear_probe_accuracy():
    acts, labels = _binary_separable_data()
    probe = train_linear_probe(acts, labels, max_iter=300)
    acc = probe.accuracy(acts, labels)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_linear_probe_predict_shape():
    acts, labels = _binary_separable_data(n=40, d=8)
    probe = train_linear_probe(acts, labels, max_iter=200)
    preds = probe.predict(acts)
    assert preds.shape == (40,), f"predict() shape mismatch: {preds.shape}"
    proba = probe.predict_proba(acts)
    assert proba.shape == (40, 2), f"predict_proba() shape mismatch: {proba.shape}"


def test_linear_probe_direction_unit():
    # Binary probe
    acts, labels = _binary_separable_data()
    probe_bin = train_linear_probe(acts, labels, max_iter=300)
    d = probe_bin.direction()
    assert abs(d.norm().item() - 1.0) < 1e-5, f"Binary direction norm: {d.norm().item()}"

    # Multiclass probe
    rng = torch.Generator()
    rng.manual_seed(3)
    n, dim = 60, 8
    a0 = torch.randn(n, dim, generator=rng); a0[:, 0] += 5.0
    a1 = torch.randn(n, dim, generator=rng); a1[:, 1] += 5.0
    a2 = torch.randn(n, dim, generator=rng); a2[:, 2] += 5.0
    acts3 = torch.cat([a0, a1, a2])
    labels3 = torch.cat([torch.zeros(n, dtype=torch.long),
                         torch.ones(n, dtype=torch.long),
                         torch.full((n,), 2, dtype=torch.long)])
    probe_mc = train_linear_probe(acts3, labels3, max_iter=500)
    d_mc = probe_mc.direction()
    assert abs(d_mc.norm().item() - 1.0) < 1e-5, f"Multiclass direction norm: {d_mc.norm().item()}"


def test_train_linear_probe_returns_linearprobe():
    acts, labels = _binary_separable_data(n=20, d=4)
    probe = train_linear_probe(acts, labels, max_iter=50)
    assert isinstance(probe, LinearProbe)


def test_train_linear_probe_cpu_output():
    acts, labels = _binary_separable_data(n=20, d=4)
    probe = train_linear_probe(acts, labels, max_iter=50)
    assert probe.weight.device.type == "cpu", f"weight on {probe.weight.device}"
    assert probe.bias.device.type == "cpu", f"bias on {probe.bias.device}"
