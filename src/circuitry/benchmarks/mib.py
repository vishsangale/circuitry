"""MIB benchmark task loaders.

Mueller et al. ICML 2025. https://arxiv.org/abs/2504.13151
Each loader returns a MIBTask with clean/corrupted token ID tensors
and a differentiable metric function compatible with Runner.run().

v1.32 adds: load_ravel, load_arithmetic, load_mcqa, mib_circuit_f1, mib_iia_score.

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


# ---------------------------------------------------------------------------
# v1.32 — RAVEL, Arithmetic, MCQA tasks + MIB evaluation helpers
# ---------------------------------------------------------------------------


def load_ravel(
    n_examples: int = 50,
    *,
    seed: int = 42,
    entity_type: str = "country",
    attribute: str = "capital",
    vocab_size: int = 1000,
    seq_len: int = 10,
) -> MIBTask:
    """Generate synthetic RAVEL entity-attribute disentanglement task instances.

    RAVEL (Resolving Attribute-Value Entanglement in Language models) tests
    whether a model can separate entity identity from entity attributes.

    Each sequence has the structure ``[context..., entity_token, attribute_token]``.
    Clean: entity A with attribute X (true pairing).
    Corrupted: entity B with attribute X (entity replaced; attribute unchanged).
    Metric: logit_diff(correct_attr_token, foil_attr_token) at the last position.

    Args:
        n_examples:  Number of task instances.
        seed:        RNG seed for reproducibility.
        entity_type: Semantic label (ignored in synthetic generation; metadata only).
        attribute:   Semantic label (ignored in synthetic generation; metadata only).
        vocab_size:  Upper bound on token IDs used.
        seq_len:     Total sequence length (must be ≥ 3).

    Returns:
        :class:`MIBTask` with ``name="ravel"``.

    Reference: arXiv:2402.17700; MIB arXiv:2504.13151.
    """
    rng = torch.Generator().manual_seed(seed)

    # Entity tokens in the lower quarter; attribute tokens in the next quarter
    entity_range_lo = 10
    entity_range_hi = vocab_size // 4
    attr_range_lo = vocab_size // 4
    attr_range_hi = vocab_size // 2

    # n_examples distinct entity pairs (A, B), same attribute X
    entity_a = torch.randint(entity_range_lo, entity_range_hi, (n_examples,), generator=rng)
    entity_b = torch.randint(entity_range_lo, entity_range_hi, (n_examples,), generator=rng)
    attr_correct = torch.randint(attr_range_lo, attr_range_hi, (n_examples,), generator=rng)
    attr_foil = torch.randint(attr_range_lo, attr_range_hi, (n_examples,), generator=rng)

    # Prefix context tokens
    prefix = torch.randint(vocab_size // 2, vocab_size, (n_examples, seq_len - 2), generator=rng)

    # Clean:     [context..., entity_a, attr_correct]
    # Corrupted: [context..., entity_b, attr_correct]  (entity changed, attr unchanged)
    clean_ids = torch.cat([prefix, entity_a.unsqueeze(1), attr_correct.unsqueeze(1)], dim=1)
    corrupt_ids = torch.cat([prefix, entity_b.unsqueeze(1), attr_correct.unsqueeze(1)], dim=1)

    def metric(logits: Any) -> Tensor:
        """Logit diff: logit(attr_correct) − logit(attr_foil) at last position."""
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        last = logits[:, -1, :] if logits.dim() == 3 else logits
        return (
            last.gather(1, attr_correct.to(last.device).unsqueeze(1))
            - last.gather(1, attr_foil.to(last.device).unsqueeze(1))
        ).mean()

    return MIBTask(
        name="ravel",
        clean_inputs={"input_ids": clean_ids},
        corrupted_inputs={"input_ids": corrupt_ids},
        metric=metric,
        correct_token_ids=attr_correct,
        incorrect_token_ids=attr_foil,
    )


def load_arithmetic(
    n_examples: int = 50,
    *,
    seed: int = 42,
    op: str = "add",
    modulus: int = 113,
    vocab_size: int = 1000,
    seq_len: int = 8,
) -> MIBTask:
    """Generate synthetic arithmetic circuit task instances.

    Supports addition (``op="add"``) and modular addition (``op="mod_add"``).
    Each sequence has the structure ``[context..., a, b]`` where the correct
    next-token is ``a + b`` (or ``(a + b) % modulus``).  Token IDs ``[200, 312]``
    represent operands (0–112) and sums.

    Clean: pair ``(a, b)`` → correct sum token.
    Corrupted: pair ``(a', b')`` → different correct sum token.
    Metric: logit_diff(correct_sum, corrupted_sum) at last position.

    Args:
        n_examples: Number of task instances.
        seed:       RNG seed.
        op:         ``"add"`` (clamped integer addition) or ``"mod_add"``
                    (modular addition ``% modulus``).
        modulus:    Prime modulus for modular arithmetic (default 113; a common
                    choice in the grokking / algorithmic task literature).
        vocab_size: Upper bound on non-operand token IDs.
        seq_len:    Total sequence length (must be ≥ 3).

    Returns:
        :class:`MIBTask` with ``name="arithmetic_add"`` or ``"arithmetic_mod_add"``.

    Reference: MIB arXiv:2504.13151.
    """
    if op not in ("add", "mod_add"):
        raise ValueError(f"op must be 'add' or 'mod_add', got {op!r}")

    rng = torch.Generator().manual_seed(seed)

    # Operands in [0, modulus-1]; token IDs offset by 200 so they don't overlap
    # with the vocab used for context tokens.
    offset = 200
    a = torch.randint(0, modulus, (n_examples,), generator=rng)
    b = torch.randint(0, modulus, (n_examples,), generator=rng)
    a2 = torch.randint(0, modulus, (n_examples,), generator=rng)
    b2 = torch.randint(0, modulus, (n_examples,), generator=rng)

    if op == "mod_add":
        correct_sum = (a + b) % modulus
        corrupted_sum = (a2 + b2) % modulus
    else:
        correct_sum = (a + b).clamp(max=modulus - 1)
        corrupted_sum = (a2 + b2).clamp(max=modulus - 1)

    correct_tok = correct_sum + offset
    corrupted_tok = corrupted_sum + offset

    prefix = torch.randint(vocab_size // 2, vocab_size, (n_examples, seq_len - 2), generator=rng)
    clean_ids = torch.cat([prefix, (a + offset).unsqueeze(1), (b + offset).unsqueeze(1)], dim=1)
    corrupt_ids = torch.cat([prefix, (a2 + offset).unsqueeze(1), (b2 + offset).unsqueeze(1)], dim=1)

    def metric(logits: Any) -> Tensor:
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        last = logits[:, -1, :] if logits.dim() == 3 else logits
        return (
            last.gather(1, correct_tok.to(last.device).unsqueeze(1))
            - last.gather(1, corrupted_tok.to(last.device).unsqueeze(1))
        ).mean()

    task_name = "arithmetic_mod_add" if op == "mod_add" else "arithmetic_add"
    return MIBTask(
        name=task_name,
        clean_inputs={"input_ids": clean_ids},
        corrupted_inputs={"input_ids": corrupt_ids},
        metric=metric,
        correct_token_ids=correct_tok,
        incorrect_token_ids=corrupted_tok,
    )


def load_mcqa(
    n_examples: int = 50,
    *,
    seed: int = 42,
    n_choices: int = 4,
    vocab_size: int = 1000,
    seq_len: int = 12,
) -> MIBTask:
    """Generate synthetic multiple-choice Q&A task instances.

    Each sequence has the structure ``[question_tokens..., choice_0, ..., choice_{n-1}]``.
    Clean: choice 0 is the correct answer.
    Corrupted: a randomly selected different choice is "correct" (swapped).
    Metric: logit_diff(correct_choice, corrupted_choice) at the last position.

    Args:
        n_examples: Number of task instances.
        seed:       RNG seed.
        n_choices:  Number of answer choices per question (default 4).
        vocab_size: Upper bound on token IDs.
        seq_len:    Total sequence length (must be ≥ n_choices + 1).

    Returns:
        :class:`MIBTask` with ``name="mcqa"``.

    Reference: MIB arXiv:2504.13151.
    """
    if seq_len < n_choices + 1:
        raise ValueError(
            f"seq_len ({seq_len}) must be >= n_choices + 1 = {n_choices + 1}"
        )
    rng = torch.Generator().manual_seed(seed)

    # Choice tokens in [10, vocab_size//4); question context in [vocab_size//2, vocab_size)
    choice_ids = torch.randint(10, vocab_size // 4, (n_examples, n_choices), generator=rng)
    n_ctx = seq_len - n_choices
    context = torch.randint(vocab_size // 2, vocab_size, (n_examples, n_ctx), generator=rng)

    # Clean: [context..., choice_0, choice_1, ...] with choice_0 correct
    clean_ids = torch.cat([context, choice_ids], dim=1)

    # Corrupted: swap choice_0 with a random other choice
    foil_idx = torch.randint(1, n_choices, (n_examples,), generator=rng)  # always != 0
    corrupt_choice_ids = choice_ids.clone()
    for i in range(n_examples):
        fi = int(foil_idx[i].item())
        corrupt_choice_ids[i, 0], corrupt_choice_ids[i, fi] = (
            corrupt_choice_ids[i, fi].clone(),
            corrupt_choice_ids[i, 0].clone(),
        )
    corrupt_ids = torch.cat([context, corrupt_choice_ids], dim=1)

    correct_tok = choice_ids[:, 0]           # original correct choice
    foil_tok = choice_ids[torch.arange(n_examples), foil_idx]  # foil

    def metric(logits: Any) -> Tensor:
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        last = logits[:, -1, :] if logits.dim() == 3 else logits
        return (
            last.gather(1, correct_tok.to(last.device).unsqueeze(1))
            - last.gather(1, foil_tok.to(last.device).unsqueeze(1))
        ).mean()

    return MIBTask(
        name="mcqa",
        clean_inputs={"input_ids": clean_ids},
        corrupted_inputs={"input_ids": corrupt_ids},
        metric=metric,
        correct_token_ids=correct_tok,
        incorrect_token_ids=foil_tok,
    )


def mib_circuit_f1(
    circuit_edges: "set | list",
    ground_truth_edges: "set | list",
) -> float:
    """Edge-set F1 for the MIB circuit localisation track.

    Computes precision, recall, and harmonic mean (F1) between the predicted
    circuit edges and the reference ground-truth edge set.  Returns 0.0 when
    there are no true positives.

    Args:
        circuit_edges:      Predicted set of :class:`~circuitry.patching.graph.Edge`
                            objects (or any hashable edge representation).
        ground_truth_edges: Reference ground-truth edge set.

    Returns:
        F1 score in [0, 1].

    Reference: MIB arXiv:2504.13151.
    """
    circuit_set = set(circuit_edges)
    gt_set = set(ground_truth_edges)
    tp = len(circuit_set & gt_set)
    if tp == 0:
        return 0.0
    precision = tp / len(circuit_set)
    recall = tp / len(gt_set)
    return 2.0 * precision * recall / (precision + recall)


def mib_iia_score(
    das_result: "Any",
    task: "MIBTask | None" = None,
    *,
    threshold: float = 0.5,
) -> float:
    """IIA-at-threshold score for MIB causal variable localisation.

    Returns the IIA score from a :class:`~circuitry.patching.das.DASResult`
    if it meets the ``threshold``; otherwise returns ``0.0``.  The threshold
    determines whether the located subspace qualifies as having "found" the
    causal variable.

    Args:
        das_result: A :class:`~circuitry.patching.das.DASResult` from
                    :class:`~circuitry.patching.das.DASRunner`.
        task:       Optional :class:`MIBTask` for metadata (unused in this
                    implementation; reserved for future validation).
        threshold:  Minimum IIA to count as a successful localisation
                    (default 0.5, matching the MIB benchmark convention).

    Returns:
        ``das_result.iia_score`` if ≥ ``threshold``, else ``0.0``.

    Reference: MIB arXiv:2504.13151.
    """
    iia = float(das_result.iia_score)
    return iia if iia >= threshold else 0.0
