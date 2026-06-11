"""Tests for v1.32 MIB additions: load_ravel, load_arithmetic, load_mcqa,
mib_circuit_f1, mib_iia_score."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from circuitry.benchmarks.mib import (
    MIBTask,
    load_ravel,
    load_arithmetic,
    load_mcqa,
    mib_circuit_f1,
    mib_iia_score,
)


# ---------------------------------------------------------------------------
# load_ravel
# ---------------------------------------------------------------------------

def test_load_ravel_returns_mib_task():
    task = load_ravel(n_examples=10)
    assert isinstance(task, MIBTask)
    assert task.name == "ravel"


def test_load_ravel_input_shapes():
    n, seq = 15, 12
    task = load_ravel(n_examples=n, seq_len=seq)
    assert task.clean_inputs["input_ids"].shape == (n, seq)
    assert task.corrupted_inputs["input_ids"].shape == (n, seq)


def test_load_ravel_clean_and_corrupted_differ():
    task = load_ravel(n_examples=20)
    clean = task.clean_inputs["input_ids"]
    corrupt = task.corrupted_inputs["input_ids"]
    # Entity tokens (second-to-last column) should differ on most rows
    entity_diff = (clean[:, -2] != corrupt[:, -2]).sum().item()
    assert entity_diff > 5


def test_load_ravel_attribute_unchanged():
    """Last token (attribute) must be identical in clean and corrupted."""
    task = load_ravel(n_examples=20)
    clean = task.clean_inputs["input_ids"]
    corrupt = task.corrupted_inputs["input_ids"]
    assert (clean[:, -1] == corrupt[:, -1]).all()


def test_load_ravel_metric_callable():
    task = load_ravel(n_examples=4)
    logits = torch.randn(4, task.clean_inputs["input_ids"].shape[1], 1000)
    val = task.metric(logits)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0


def test_load_ravel_seed_reproducible():
    t1 = load_ravel(n_examples=10, seed=7)
    t2 = load_ravel(n_examples=10, seed=7)
    assert torch.equal(t1.clean_inputs["input_ids"], t2.clean_inputs["input_ids"])


# ---------------------------------------------------------------------------
# load_arithmetic
# ---------------------------------------------------------------------------

def test_load_arithmetic_add_returns_mib_task():
    task = load_arithmetic(n_examples=10, op="add")
    assert isinstance(task, MIBTask)
    assert task.name == "arithmetic_add"


def test_load_arithmetic_mod_add_name():
    task = load_arithmetic(n_examples=10, op="mod_add")
    assert task.name == "arithmetic_mod_add"


def test_load_arithmetic_input_shapes():
    n, seq = 12, 10
    task = load_arithmetic(n_examples=n, seq_len=seq)
    assert task.clean_inputs["input_ids"].shape == (n, seq)
    assert task.corrupted_inputs["input_ids"].shape == (n, seq)


def test_load_arithmetic_clean_corrupted_differ():
    task = load_arithmetic(n_examples=20)
    clean = task.clean_inputs["input_ids"]
    corrupt = task.corrupted_inputs["input_ids"]
    assert not torch.equal(clean, corrupt)


def test_load_arithmetic_metric_callable():
    task = load_arithmetic(n_examples=4)
    logits = torch.randn(4, task.clean_inputs["input_ids"].shape[1], 1000)
    val = task.metric(logits)
    assert isinstance(val, torch.Tensor) and val.ndim == 0


def test_load_arithmetic_invalid_op_raises():
    with pytest.raises(ValueError, match="op"):
        load_arithmetic(op="multiply")


def test_load_arithmetic_mod_add_tokens_in_range():
    """Operand and sum tokens must be in [200, 200+modulus)."""
    modulus = 113
    task = load_arithmetic(n_examples=20, op="mod_add", modulus=modulus)
    for ids in (task.clean_inputs["input_ids"], task.corrupted_inputs["input_ids"]):
        # Last two tokens are the operands
        for col in (-2, -1):
            col_vals = ids[:, col]
            assert (col_vals >= 200).all() and (col_vals < 200 + modulus).all()


# ---------------------------------------------------------------------------
# load_mcqa
# ---------------------------------------------------------------------------

def test_load_mcqa_returns_mib_task():
    task = load_mcqa(n_examples=10)
    assert isinstance(task, MIBTask)
    assert task.name == "mcqa"


def test_load_mcqa_input_shapes():
    n, seq = 12, 14
    task = load_mcqa(n_examples=n, seq_len=seq)
    assert task.clean_inputs["input_ids"].shape == (n, seq)
    assert task.corrupted_inputs["input_ids"].shape == (n, seq)


def test_load_mcqa_clean_corrupted_differ():
    task = load_mcqa(n_examples=20)
    clean = task.clean_inputs["input_ids"]
    corrupt = task.corrupted_inputs["input_ids"]
    assert not torch.equal(clean, corrupt)


def test_load_mcqa_metric_callable():
    task = load_mcqa(n_examples=4)
    logits = torch.randn(4, task.clean_inputs["input_ids"].shape[1], 1000)
    val = task.metric(logits)
    assert isinstance(val, torch.Tensor) and val.ndim == 0


def test_load_mcqa_seq_len_too_short_raises():
    with pytest.raises(ValueError, match="seq_len"):
        load_mcqa(n_choices=4, seq_len=3)


def test_load_mcqa_seed_reproducible():
    t1 = load_mcqa(n_examples=8, seed=99)
    t2 = load_mcqa(n_examples=8, seed=99)
    assert torch.equal(t1.clean_inputs["input_ids"], t2.clean_inputs["input_ids"])


# ---------------------------------------------------------------------------
# mib_circuit_f1
# ---------------------------------------------------------------------------

def test_mib_circuit_f1_perfect_overlap():
    edges = {"a", "b", "c"}
    assert mib_circuit_f1(edges, edges) == pytest.approx(1.0)


def test_mib_circuit_f1_no_overlap():
    assert mib_circuit_f1({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)


def test_mib_circuit_f1_partial():
    circuit = {"a", "b", "c"}
    gt = {"b", "c", "d"}
    # tp=2, fp=1, fn=1 → prec=2/3, recall=2/3, f1=2/3
    f1 = mib_circuit_f1(circuit, gt)
    assert f1 == pytest.approx(2 / 3, rel=1e-4)


def test_mib_circuit_f1_empty_circuit():
    assert mib_circuit_f1(set(), {"a", "b"}) == pytest.approx(0.0)


def test_mib_circuit_f1_empty_ground_truth():
    assert mib_circuit_f1({"a"}, set()) == pytest.approx(0.0)


def test_mib_circuit_f1_accepts_lists():
    f1 = mib_circuit_f1(["a", "b"], ["b", "c"])
    assert 0.0 < f1 < 1.0


# ---------------------------------------------------------------------------
# mib_iia_score
# ---------------------------------------------------------------------------

def test_mib_iia_score_above_threshold():
    das = MagicMock()
    das.iia_score = 0.8
    score = mib_iia_score(das, threshold=0.5)
    assert score == pytest.approx(0.8)


def test_mib_iia_score_below_threshold_returns_zero():
    das = MagicMock()
    das.iia_score = 0.3
    score = mib_iia_score(das, threshold=0.5)
    assert score == pytest.approx(0.0)


def test_mib_iia_score_at_threshold():
    das = MagicMock()
    das.iia_score = 0.5
    score = mib_iia_score(das, threshold=0.5)
    assert score == pytest.approx(0.5)


def test_mib_iia_score_task_arg_optional():
    das = MagicMock()
    das.iia_score = 0.9
    # task arg is optional (defaults to None)
    score = mib_iia_score(das)
    assert score == pytest.approx(0.9)
