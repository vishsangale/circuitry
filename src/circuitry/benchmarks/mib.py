"""MIB benchmark task loaders.

Mueller et al. ICML 2025. https://arxiv.org/abs/2504.13151
Each loader returns a MIBTask with clean/corrupted token ID tensors
and a differentiable metric function compatible with Runner.run().

All instances are generated synthetically — no tokenizer or dataset
download required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


@dataclass
class MIBTask:
    """Container for a single MIB benchmark task.

    Attributes:
        name:                Human-readable task identifier.
        clean_inputs:        Dict mapping ``"input_ids"`` to a ``(n, seq_len)``
                             integer tensor for the clean (unperturbed) inputs.
        corrupted_inputs:    Same structure as ``clean_inputs`` but with the
                             causal intervention applied.
        metric:              Callable ``(logits) -> scalar Tensor`` that is
                             differentiable w.r.t. the logits.  Compatible with
                             ``Runner.run()``.
        correct_token_ids:   ``(n,)`` ground-truth next-token ids.
        incorrect_token_ids: ``(n,)`` foil token ids.
    """

    name: str
    clean_inputs: dict[str, Tensor]
    corrupted_inputs: dict[str, Tensor]
    metric: Callable[[Any], Tensor]
    correct_token_ids: Tensor
    incorrect_token_ids: Tensor

    def __len__(self) -> int:
        return self.clean_inputs["input_ids"].shape[0]


def load_ioi(
    n_examples: int = 50,
    *,
    seed: int = 42,
    vocab_size: int = 1000,
    seq_len: int = 12,
) -> MIBTask:
    """Generate synthetic IOI (Indirect Object Identification) task instances.

    Mimics the IOI circuit task structure (Wang et al. 2022): each sequence
    contains a "subject" (S) token and an "indirect object" (IO) token; the
    clean target is the IO token and the corrupted version swaps them.

    Since real tokenisation requires a specific model vocabulary, this
    generates synthetic integer token IDs in ``[10, vocab_size // 2)``.  The
    metric is logit_diff(IO, S) at the last sequence position.

    Args:
        n_examples: Number of task instances to generate.
        seed:       RNG seed for reproducibility.
        vocab_size: Upper bound on token IDs used.
        seq_len:    Length of each generated sequence.

    Returns:
        :class:`MIBTask` with ``name="ioi"``.
    """
    rng = torch.Generator().manual_seed(seed)

    # Sample n_examples pairs of distinct name tokens in the lower half of vocab
    name_ids = torch.randint(10, vocab_size // 2, (n_examples, 2), generator=rng)
    io_ids = name_ids[:, 0]   # indirect object token
    s_ids  = name_ids[:, 1]   # subject token

    # Build fixed-length sequences.
    # Structure: [prefix tokens..., io_id, s_id]
    # seq_len total: (seq_len - 2) prefix tokens + io slot + s slot
    prefix = torch.randint(vocab_size // 2, vocab_size, (n_examples, seq_len - 2), generator=rng)

    # Clean: ends with [..., io_id, s_id]
    clean_ids = torch.cat([prefix, io_ids.unsqueeze(1), s_ids.unsqueeze(1)], dim=1)

    # Corrupted: swap — ends with [..., s_id, io_id]
    corrupt_ids = torch.cat([prefix, s_ids.unsqueeze(1), io_ids.unsqueeze(1)], dim=1)

    def metric(logits: Any) -> Tensor:
        """Logit difference: logit(IO) - logit(S) at last position."""
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        # logits: (batch, seq, vocab) or (batch, vocab)
        last = logits[:, -1, :] if logits.dim() == 3 else logits
        return (
            last.gather(1, io_ids.to(last.device).unsqueeze(1))
            - last.gather(1, s_ids.to(last.device).unsqueeze(1))
        ).mean()

    return MIBTask(
        name="ioi",
        clean_inputs={"input_ids": clean_ids},
        corrupted_inputs={"input_ids": corrupt_ids},
        metric=metric,
        correct_token_ids=io_ids,
        incorrect_token_ids=s_ids,
    )


def load_greater_than(
    n_examples: int = 50,
    *,
    seed: int = 42,
    vocab_size: int = 1000,
    seq_len: int = 8,
) -> MIBTask:
    """Generate synthetic Greater-Than task instances.

    Simulates the structure of the greater-than circuit task (Hanna et al.
    2023): a sequence ends with a "year" token; the model should predict
    tokens numerically greater than a prior year token in the sequence.

    Token IDs ``[100, 199]`` represent decade tokens.  For each example:

    - ``year1`` is a randomly sampled decade token.
    - ``year2 = year1 + delta`` where ``delta ∈ [1, 20]`` (valid completion).
    - ``year_corrupt ≤ year1`` (invalid / corrupted completion).

    The metric is ``logit(year2) - logit(year_corrupt)`` at the last position.

    Args:
        n_examples: Number of task instances to generate.
        seed:       RNG seed for reproducibility.
        vocab_size: Upper bound on non-year token IDs used for prefix context.
        seq_len:    Length of each generated sequence (must be ≥ 2).

    Returns:
        :class:`MIBTask` with ``name="greater_than"``.
    """
    rng = torch.Generator().manual_seed(seed)

    year_start = 100   # token IDs 100–199 represent "decades"
    year_range = 100

    year1 = torch.randint(year_start, year_start + year_range - 1, (n_examples,), generator=rng)
    # year2 > year1 (valid completion)
    delta = torch.randint(1, 20, (n_examples,), generator=rng)
    year2 = (year1 + delta).clamp(max=year_start + year_range - 1)
    # corrupted: year where year_corrupt <= year1
    year_corrupt = (year1 - torch.randint(1, 10, (n_examples,), generator=rng)).clamp(min=year_start)

    prefix = torch.randint(vocab_size // 2, vocab_size, (n_examples, seq_len - 2), generator=rng)
    clean_ids   = torch.cat([prefix, year1.unsqueeze(1), year2.unsqueeze(1)], dim=1)
    corrupt_ids = torch.cat([prefix, year_corrupt.unsqueeze(1), year2.unsqueeze(1)], dim=1)

    def metric(logits: Any) -> Tensor:
        """Logit difference: logit(year2) - logit(year_corrupt) at last position."""
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        last = logits[:, -1, :] if logits.dim() == 3 else logits
        return (
            last.gather(1, year2.to(last.device).unsqueeze(1))
            - last.gather(1, year_corrupt.to(last.device).unsqueeze(1))
        ).mean()

    return MIBTask(
        name="greater_than",
        clean_inputs={"input_ids": clean_ids},
        corrupted_inputs={"input_ids": corrupt_ids},
        metric=metric,
        correct_token_ids=year2,
        incorrect_token_ids=year_corrupt,
    )
