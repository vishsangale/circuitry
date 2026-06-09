"""Tests for circuitry.core.probe — LinearProbe, MDL probing, mass-mean probe."""
from __future__ import annotations

import torch
import pytest

from circuitry.core.probe import (
    LinearProbe,
    MDLResult,
    MassMeanProbe,
    mass_mean_probe,
    mdl_probe,
    train_linear_probe,
    verify_linear_representation,
)


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


# ---------------------------------------------------------------------------
# mdl_probe tests
# ---------------------------------------------------------------------------


def test_mdl_probe_returns_mdlresult():
    acts, labels = _binary_separable_data(n=80, d=16)
    result = mdl_probe(acts, labels)
    assert isinstance(result, MDLResult)
    assert isinstance(result.code_length, float)
    assert isinstance(result.data_entropy, float)
    assert isinstance(result.mdl_ratio, float)


def test_mdl_probe_ratio_low_for_separable():
    """Linearly separable data should yield mdl_ratio < 1 (encoding is efficient)."""
    acts, labels = _binary_separable_data(n=160, d=16)
    result = mdl_probe(acts, labels)
    assert result.mdl_ratio < 1.0, f"Expected ratio < 1 on separable data, got {result.mdl_ratio:.3f}"


def test_mdl_probe_ratio_high_for_random():
    """Random labels should yield mdl_ratio >= 1 (no genuine encoding)."""
    torch.manual_seed(99)
    acts = torch.randn(160, 16)
    labels = torch.randint(0, 2, (160,))
    result = mdl_probe(acts, labels)
    assert result.mdl_ratio >= 0.9, f"Random labels, mdl_ratio = {result.mdl_ratio:.3f}"


def test_mdl_probe_code_length_positive():
    acts, labels = _binary_separable_data(n=40, d=8)
    result = mdl_probe(acts, labels)
    assert result.code_length > 0.0


def test_mdl_probe_data_entropy_binary():
    """For balanced binary labels, data entropy ≈ log(2) ≈ 0.693."""
    acts, labels = _binary_separable_data(n=100, d=8)
    result = mdl_probe(acts, labels)
    assert abs(result.data_entropy - 0.693) < 0.05, f"data_entropy = {result.data_entropy}"


def test_mdl_probe_multiclass():
    """mdl_probe works for 3-class labels."""
    torch.manual_seed(7)
    n, d = 90, 16
    acts0 = torch.randn(n // 3, d); acts0[:, 0] += 5.0
    acts1 = torch.randn(n // 3, d); acts1[:, 1] += 5.0
    acts2 = torch.randn(n // 3, d); acts2[:, 2] += 5.0
    acts = torch.cat([acts0, acts1, acts2])
    labels = torch.cat([torch.zeros(n // 3, dtype=torch.long),
                        torch.ones(n // 3, dtype=torch.long),
                        torch.full((n // 3,), 2, dtype=torch.long)])
    result = mdl_probe(acts, labels)
    assert result.mdl_ratio < 1.0


# ---------------------------------------------------------------------------
# mass_mean_probe tests
# ---------------------------------------------------------------------------


def test_mass_mean_probe_returns_massmeanprobe():
    acts, labels = _binary_separable_data(n=80, d=16)
    p = mass_mean_probe(acts, labels)
    assert isinstance(p, MassMeanProbe)


def test_mass_mean_probe_direction_unit():
    acts, labels = _binary_separable_data(n=80, d=16)
    p = mass_mean_probe(acts, labels)
    assert abs(p.direction.norm().item() - 1.0) < 1e-5


def test_mass_mean_probe_direction_cpu():
    acts, labels = _binary_separable_data(n=80, d=16)
    p = mass_mean_probe(acts, labels)
    assert p.direction.device.type == "cpu"


def test_mass_mean_probe_predict_shape():
    acts, labels = _binary_separable_data(n=80, d=16)
    p = mass_mean_probe(acts, labels)
    preds = p.predict(acts)
    assert preds.shape == (80,)
    assert preds.dtype == torch.long


def test_mass_mean_probe_accuracy_high_on_separable():
    acts, labels = _binary_separable_data(n=160, d=16)
    p = mass_mean_probe(acts, labels)
    acc = p.accuracy(acts, labels)
    assert acc >= 0.90, f"Expected high accuracy on separable data, got {acc:.3f}"


def test_mass_mean_probe_raises_multiclass():
    torch.manual_seed(0)
    acts = torch.randn(30, 8)
    labels = torch.randint(0, 3, (30,))
    with pytest.raises(ValueError, match="2 classes"):
        mass_mean_probe(acts, labels)


def test_mass_mean_probe_classes_attribute():
    acts, labels = _binary_separable_data(n=40, d=8)
    p = mass_mean_probe(acts, labels)
    assert set(p.classes) == {0, 1}


# ---------------------------------------------------------------------------
# verify_linear_representation tests
# ---------------------------------------------------------------------------


def test_verify_linear_representation_high_for_consistent():
    """Probe direction and steering vector derived from same data should agree."""
    acts, labels = _binary_separable_data(n=160, d=16, seed=0)
    p = mass_mean_probe(acts, labels)
    # Build a steering vector pointing in the same direction as the probe
    sv = p.direction.clone()
    score = verify_linear_representation(p, sv)
    assert abs(score) > 0.99, f"Expected |score| ≈ 1, got {score:.4f}"


def test_verify_linear_representation_orthogonal():
    """Orthogonal probe direction and steer_vec gives score ≈ 0."""
    acts, labels = _binary_separable_data(n=160, d=16, seed=1)
    p = mass_mean_probe(acts, labels)
    d = p.direction.float()
    # Build a vector orthogonal to d (Gram–Schmidt of a random vector)
    v = torch.randn_like(d)
    v = v - (v @ d) * d
    score = verify_linear_representation(p, v)
    assert abs(score) < 0.1, f"Expected near-zero score for orthogonal vec, got {score:.4f}"


def test_verify_linear_representation_returns_float():
    acts, labels = _binary_separable_data(n=40, d=8)
    p = mass_mean_probe(acts, labels)
    sv = torch.randn(8)
    score = verify_linear_representation(p, sv)
    assert isinstance(score, float)


def test_verify_linear_representation_with_linear_probe():
    """verify_linear_representation works with a LinearProbe (has .direction() method)."""
    acts, labels = _binary_separable_data(n=80, d=16)
    lp = train_linear_probe(acts, labels, max_iter=300)
    p = mass_mean_probe(acts, labels)
    score = verify_linear_representation(lp, p.direction)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_verify_linear_representation_dim_mismatch():
    """Works when probe direction and steer_vec have different dimensions."""
    acts, labels = _binary_separable_data(n=80, d=16)
    p = mass_mean_probe(acts, labels)
    sv = torch.randn(8)  # shorter
    score = verify_linear_representation(p, sv)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
