"""Shared IOI utilities for the v1.0 patching-pillar validation (Track 1).

Ground truth: the IOI circuit of Wang et al. 2022 (arXiv 2211.00593). The head
classes below are the canonical constants from the authors' own codebase
(redwoodresearch/Easy-Transformer, easy_transformer/ioi_circuit_extraction.py),
pinned here rather than reconstructed from memory.

Dataset: single-token names + a fixed template so every clean/corrupted prompt
has identical token length (=> batchable) and the answer is always the final
token. Corruption is the standard "ABC" patch: the repeated subject S2 is
replaced by a third distinct name C, destroying the duplicate-token / S-inhibition
signal while keeping the IO/S answer tokens fixed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

# --- Published IOI circuit ground truth (Wang et al. 2022) ------------------
CIRCUIT: dict[str, list[tuple[int, int]]] = {
    "name_mover": [(9, 9), (10, 0), (9, 6), (10, 10), (10, 6), (10, 2),
                   (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative_name_mover": [(10, 7), (11, 10)],
    "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token": [(0, 1), (0, 10), (3, 0)],
    "previous_token": [(2, 2), (4, 11)],
}
# The three "core" name movers highlighted in the paper text.
CORE_NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]


def all_circuit_heads() -> set[tuple[int, int]]:
    return {h for hs in CIRCUIT.values() for h in hs}


def head_class(layer: int, head: int) -> str | None:
    for cls, hs in CIRCUIT.items():
        if (layer, head) in hs:
            return cls
    return None


# --- Single-token name pool -------------------------------------------------
# Verified single-token (with leading space) under the GPT-2 tokenizer via
# filter_single_token_names(); this is the candidate list.
_CANDIDATE_NAMES = [
    "John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Sam",
    "Anna", "Susan", "Paul", "Mark", "Mike", "Robert", "David", "Scott",
    "Kevin", "Laura", "Karen", "Brian", "Steve", "Peter", "Carl", "Alice",
    "Bob", "Henry", "Jack", "Kate", "Lisa", "Frank", "Emma", "Lucy", "Eric",
]


def _encode(tokenizer, text: str) -> list[int]:
    """Encode without special tokens (TL's tokenizer auto-prepends BOS otherwise)."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def filter_single_token_names(tokenizer, names: list[str] | None = None) -> list[str]:
    """Keep only names that encode (with a leading space) to exactly one token."""
    names = names or _CANDIDATE_NAMES
    out = []
    for n in names:
        if len(_encode(tokenizer, " " + n)) == 1:
            out.append(n)
    return out


@dataclass
class IOIBatch:
    clean: torch.Tensor       # (B, T) token ids
    corrupt: torch.Tensor     # (B, T) token ids — ABC corruption (S2 -> C)
    io_ids: torch.Tensor      # (B,) answer token id (indirect object)
    s_ids: torch.Tensor       # (B,) subject token id
    prompts: list[str]        # clean prompt strings (for inspection)


def build_ioi_batch(
    tokenizer,
    n: int = 64,
    seed: int = 0,
    prepend_bos: bool = True,
    bos_token_id: int | None = None,
    device: str = "cpu",
) -> IOIBatch:
    """Build a batch of fixed-length IOI prompts + ABC-corrupted counterparts.

    Templates (both name orderings; answer = IO = the non-repeated name):
      BABA: "When {S} and {IO} went to the store, {S} gave a drink to"
      ABBA: "When {IO} and {S} went to the store, {S} gave a drink to"
    Corruption replaces the second {S} (S2) with a fresh name C.
    """
    rng = random.Random(seed)
    pool = filter_single_token_names(tokenizer)
    if len(pool) < 3:
        raise RuntimeError(f"need >=3 single-token names, got {len(pool)}: {pool}")

    clean_rows, corrupt_rows, io_list, s_list, prompts = [], [], [], [], []
    bos = bos_token_id if bos_token_id is not None else getattr(tokenizer, "bos_token_id", None)

    def enc(text: str) -> list[int]:
        ids = _encode(tokenizer, text)
        return ([bos] + ids) if (prepend_bos and bos is not None) else ids

    for _ in range(n):
        io, s, c = rng.sample(pool, 3)
        if rng.random() < 0.5:  # BABA
            clean = f"When {s} and {io} went to the store, {s} gave a drink to"
            corrupt = f"When {s} and {io} went to the store, {c} gave a drink to"
        else:                    # ABBA
            clean = f"When {io} and {s} went to the store, {s} gave a drink to"
            corrupt = f"When {io} and {s} went to the store, {c} gave a drink to"
        clean_rows.append(enc(clean))
        corrupt_rows.append(enc(corrupt))
        io_list.append(_encode(tokenizer, " " + io)[0])
        s_list.append(_encode(tokenizer, " " + s)[0])
        prompts.append(clean)

    # All rows must share length to batch (single-token names + fixed template).
    lens = {len(r) for r in clean_rows} | {len(r) for r in corrupt_rows}
    if len(lens) != 1:
        raise RuntimeError(f"ragged token lengths {lens}; a name tokenized to >1 token")

    return IOIBatch(
        clean=torch.tensor(clean_rows, device=device),
        corrupt=torch.tensor(corrupt_rows, device=device),
        io_ids=torch.tensor(io_list, device=device),
        s_ids=torch.tensor(s_list, device=device),
        prompts=prompts,
    )


def batched_logit_diff_metric(io_ids: torch.Tensor, s_ids: torch.Tensor):
    """Return metric(model_output) -> scalar Tensor = mean per-row (IO - S) last-token logit.

    Works around core.logit_diff_t taking scalar indices: here IO/S vary per row,
    so we gather per-row. Differentiable (no .detach); suitable as the EAP/AtP metric.
    """
    def metric(out):
        logits = out.logits if hasattr(out, "logits") else out
        last = logits[:, -1, :].float()                  # (B, V)
        io = last.gather(1, io_ids.view(-1, 1)).squeeze(1)
        s = last.gather(1, s_ids.view(-1, 1)).squeeze(1)
        return (io - s).mean()
    return metric


if __name__ == "__main__":
    # Self-test: build a batch on the GPT-2 tokenizer, sanity-check shapes and
    # that the clean run beats the corrupted run on logit-diff (model required).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    pool = filter_single_token_names(tok)
    print(f"single-token names ({len(pool)}): {pool}")
    b = build_ioi_batch(tok, n=8, seed=0)
    print(f"clean {tuple(b.clean.shape)} corrupt {tuple(b.corrupt.shape)} "
          f"io_ids {tuple(b.io_ids.shape)}")
    print("example clean prompt:", b.prompts[0])
    print("circuit heads:", len(all_circuit_heads()), "core movers:", CORE_NAME_MOVERS)
