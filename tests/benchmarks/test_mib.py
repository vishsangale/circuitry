"""Tests for circuitry.benchmarks.mib — MIB task loaders."""
from __future__ import annotations

import torch

from circuitry.benchmarks.mib import MIBTask, load_greater_than, load_ioi

# ---------------------------------------------------------------------------
# load_ioi
# ---------------------------------------------------------------------------

def test_load_ioi_returns_mib_task():
    task = load_ioi(n_examples=10)
    assert isinstance(task, MIBTask)
    assert task.name == "ioi"


def test_load_ioi_input_shape():
    n, seq = 20, 14
    task = load_ioi(n_examples=n, seq_len=seq)
    assert task.clean_inputs["input_ids"].shape == (n, seq)
    assert task.corrupted_inputs["input_ids"].shape == (n, seq)


def test_load_ioi_corrupted_differs():
    task = load_ioi(n_examples=20)
    clean    = task.clean_inputs["input_ids"]
    corrupt  = task.corrupted_inputs["input_ids"]
    # At minimum the last two positions should differ for most examples
    assert not torch.equal(clean, corrupt), "clean and corrupted inputs should differ"


def test_load_ioi_metric_differentiable():
    task = load_ioi(n_examples=8)
    n = len(task)
    vocab = 1000
    seq = task.clean_inputs["input_ids"].shape[1]
    # Fake logits: (batch, seq, vocab) — requires grad
    logits = torch.randn(n, seq, vocab, requires_grad=True)
    score = task.metric(logits)
    assert score.dim() == 0, "metric should return a scalar tensor"
    assert score.grad_fn is not None, "metric should be differentiable (has grad_fn)"


def test_load_ioi_correct_token_ids_shape():
    task = load_ioi(n_examples=15)
    assert task.correct_token_ids.shape == (15,)
    assert task.incorrect_token_ids.shape == (15,)


def test_load_ioi_metric_2d_logits():
    """metric should also accept (batch, vocab) shaped logits."""
    task = load_ioi(n_examples=8)
    n = len(task)
    vocab = 1000
    logits = torch.randn(n, vocab, requires_grad=True)
    score = task.metric(logits)
    assert score.dim() == 0


def test_load_ioi_seed_reproducible():
    t1 = load_ioi(n_examples=10, seed=7)
    t2 = load_ioi(n_examples=10, seed=7)
    assert torch.equal(t1.clean_inputs["input_ids"], t2.clean_inputs["input_ids"])


# ---------------------------------------------------------------------------
# load_greater_than
# ---------------------------------------------------------------------------

def test_load_greater_than_returns_mib_task():
    task = load_greater_than(n_examples=10)
    assert isinstance(task, MIBTask)
    assert task.name == "greater_than"


def test_load_greater_than_correct_gt_incorrect():
    task = load_greater_than(n_examples=50)
    # correct (year2) should be >= incorrect (year_corrupt) element-wise
    # (year2 = year1 + delta >= year1 > year_corrupt for most examples)
    correct   = task.correct_token_ids.float()
    incorrect = task.incorrect_token_ids.float()
    assert (correct >= incorrect).all(), (
        "correct_token_ids should be >= incorrect_token_ids for greater-than task"
    )


def test_load_greater_than_input_shape():
    n, seq = 12, 10
    task = load_greater_than(n_examples=n, seq_len=seq)
    assert task.clean_inputs["input_ids"].shape == (n, seq)
    assert task.corrupted_inputs["input_ids"].shape == (n, seq)


def test_load_greater_than_corrupted_differs():
    task = load_greater_than(n_examples=20)
    clean   = task.clean_inputs["input_ids"]
    corrupt = task.corrupted_inputs["input_ids"]
    assert not torch.equal(clean, corrupt)


def test_load_greater_than_metric_differentiable():
    task = load_greater_than(n_examples=8)
    n    = len(task)
    seq  = task.clean_inputs["input_ids"].shape[1]
    logits = torch.randn(n, seq, 1000, requires_grad=True)
    score  = task.metric(logits)
    assert score.dim() == 0
    assert score.grad_fn is not None


# ---------------------------------------------------------------------------
# MIBTask.__len__
# ---------------------------------------------------------------------------

def test_mib_task_len():
    for n in (1, 10, 50):
        assert len(load_ioi(n_examples=n)) == n
        assert len(load_greater_than(n_examples=n)) == n
