"""Tests for core/moe.py — MoE routing diagnostics (v1.44)."""
from __future__ import annotations

import math

import pytest
import torch

from circuitry.core.moe import (
    expert_load_balance,
    pathway_complexity,
    routing_entropy,
)

# ---------------------------------------------------------------------------
# routing_entropy
# ---------------------------------------------------------------------------


class TestRoutingEntropy:
    def test_uniform_logits_give_log_n(self):
        logits = torch.zeros(10, 8)
        assert routing_entropy(logits) == pytest.approx(math.log(8), abs=1e-5)

    def test_confident_routing_near_zero(self):
        logits = torch.zeros(10, 8)
        logits[:, 3] = 100.0
        assert routing_entropy(logits) == pytest.approx(0.0, abs=1e-3)

    def test_leading_dims_flattened(self):
        logits = torch.randn(2, 5, 4)
        flat = logits.reshape(-1, 4)
        assert routing_entropy(logits) == pytest.approx(routing_entropy(flat))

    def test_from_probs(self):
        probs = torch.full((6, 4), 0.25)
        assert routing_entropy(probs, from_logits=False) == pytest.approx(
            math.log(4), abs=1e-5,
        )

    def test_too_few_experts_raises(self):
        with pytest.raises(ValueError, match="n_experts dim"):
            routing_entropy(torch.ones(5, 1))


# ---------------------------------------------------------------------------
# expert_load_balance
# ---------------------------------------------------------------------------


class TestExpertLoadBalance:
    def test_uniform_load_is_one(self):
        ids = torch.arange(8).repeat(10)
        assert expert_load_balance(ids, 8) == pytest.approx(1.0)

    def test_collapsed_load_is_one_over_n(self):
        ids = torch.zeros(100, dtype=torch.long)
        assert expert_load_balance(ids, 8) == pytest.approx(1 / 8)

    def test_topk_2d_input(self):
        ids = torch.tensor([[0, 1], [2, 3]])
        assert expert_load_balance(ids, 4) == pytest.approx(1.0)

    def test_unused_experts_lower_score(self):
        ids = torch.tensor([0, 1, 0, 1])
        # 2 of 4 experts used uniformly -> exp(log 2)/4 = 0.5
        assert expert_load_balance(ids, 4) == pytest.approx(0.5)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            expert_load_balance(torch.tensor([0, 5]), 4)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            expert_load_balance(torch.tensor([], dtype=torch.long), 4)


# ---------------------------------------------------------------------------
# pathway_complexity
# ---------------------------------------------------------------------------


class TestPathwayComplexity:
    def test_single_shared_path(self):
        per_layer = [torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)]
        assert pathway_complexity(per_layer) == pytest.approx(1.0)

    def test_all_distinct_paths(self):
        per_layer = [torch.arange(6), torch.arange(6)]
        assert pathway_complexity(per_layer) == pytest.approx(6.0)

    def test_topk_order_normalised(self):
        # [0,1] and [1,0] are the same expert SET -> one path
        layer = torch.tensor([[0, 1], [1, 0]])
        assert pathway_complexity([layer]) == pytest.approx(1.0)

    def test_two_equally_likely_paths(self):
        per_layer = [torch.tensor([0, 0, 1, 1]), torch.tensor([2, 2, 3, 3])]
        assert pathway_complexity(per_layer) == pytest.approx(2.0)

    def test_sample_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="must route the same samples"):
            pathway_complexity([torch.zeros(3, dtype=torch.long),
                                torch.zeros(4, dtype=torch.long)])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pathway_complexity([])

    def test_3d_raises(self):
        with pytest.raises(ValueError, match=r"\(n,\) or \(n, k\)"):
            pathway_complexity([torch.zeros(2, 2, 2, dtype=torch.long)])


def test_top_level_exports():
    import circuitry

    for name in ("routing_entropy", "expert_load_balance", "pathway_complexity"):
        assert hasattr(circuitry, name)
        assert name in circuitry.__all__
