# v0.9.2 — correctness patch (design)

- **Date:** 2026-05-23
- **Status:** approved (design); pending implementation plan
- **Predecessor:** v0.9.1 (HEAD `e1bfbd5`)
- **Source:** external field feedback — a user comparing softmax vs sigmoid attention
  (HF-Llama, 1M & 10M params, 16 GB GPU, live `Recorder` + jsonl). Captured in
  `TODO.md` → "External feedback — softmax-vs-sigmoid attention study (2026-05-23)".

## Motivation

Three confirmed correctness/robustness bugs surfaced in real use, plus two near-zero-risk
wins. All are fixes to existing behavior or trivial additions — no new diagnostic, no
ergonomics/report rework (those defer to the v1.0 surface). The CI-enforced invariants in
`docs/design.md` (core ↮ recorder/recipes/writers/cli; no `.cuda()` in `core/`) are
preserved throughout.

## Scope

**In:** #1 attention-entropy normalization, #2 `logit_lens_kl` OOM, #3 `output_attentions`
on wrapped models, #7 `circuitry --version`, #8 scan-vs-report docs.

**Out (deferred to v1.0):** #5 per-diagnostic `disable`/`only`, #6 report summary +
`circuitry compare`, `lens_layers` layer-stride knob.

---

## #1 — `attention_pattern_entropy` is not normalization-invariant

**Problem.** `core/attention.py:56-62` computes raw `-Σ xlogy(p, p)` over the key axis.
Valid for softmax (rows sum to 1), but for attention whose weights don't sum to 1 (sigmoid,
some linear attention) it conflates *concentration* with *total attention mass* — the value
isn't a true entropy, and cross-architecture comparison is confounded. This blocked the
reporter's core experiment.

**Fix (decided: normalize + warn).**

- **`core/attention.py`** — in `attention_pattern_entropy`, divide each query row by its
  key-axis sum (`eps`-clamped denominator, e.g. `1e-12`) before `xlogy`. Stays pure and
  device-deterministic; no logging in core.
  - Softmax rows already sum to 1, so the result is **unchanged within fp tolerance** for the
    common case (the divisor is `1 ± fp-noise`, so tiny deltas are possible — not bit-for-bit).
  - For non-softmax attention the result is now a true normalization-invariant entropy over
    the attention *shape*; total mass is intentionally discarded (that's what makes it
    comparable across architectures).
- **`recorder/live.py`** — when dispatching `attention_pattern_entropy`, compute the captured
  pattern's key-axis row-sums and, if `max |row_sum − 1| > tol` (`tol ≈ 1e-3`), emit a
  **one-time WARN** guarded by a `self._warned_unnormalized_attn` flag (mirrors the v0.9.1
  SDPA-warn pattern). Message: attention rows are not a probability distribution; entropy is
  computed over the normalized shape and total mass is discarded. The row-sum check is a cheap
  reduction over the already-captured tensor.

**Why the warn lives in the Recorder, not core:** `docs/design.md` holds primitives to "pure
functions with no I/O." The Recorder already owns runtime user-facing warnings (SDPA) and has
the raw pre-normalization pattern in hand.

**Tests.**
- `tests/core/test_attention.py`: a softmax pattern (rows sum to 1) yields the same entropy as
  before within `atol`; a non-normalized pattern (rows sum to ≠1) yields the entropy of its
  normalized form (assert against a hand-computed value).
- `tests/recorder/`: the row-sum-deviation WARN fires once for a sigmoid-like pattern and never
  for softmax.

---

## #2 — `logit_lens_kl` OOMs at modest scale and takes down the run

**Problem.** `core/lens.py:72` materializes a full `(batch, seq, vocab)` lens-logits tensor,
upcast to float32 (lines 66/69 — doubles footprint vs bf16); `log_softmax` adds more
same-shape temporaries. The Recorder calls it once per residual-stream layer, so per-call
transients (~1.5 GB at seq 512 / vocab 47k) OOM a 16 GB GPU on a 10M-param model and crash
the whole run. (The model's own `final_logits` is passed in once and shared across layers, so
it is not the per-layer cost; the lens-logits transient is.)

**Fix (decided: chunk + OOM-guard + `lens_max_tokens`; drop `lens_layers`).**

- **`core/lens.py`** — chunk the matmul + KL over the flattened token axis. New keyword arg
  `chunk_size: int = 256`. Flatten `(batch, seq, ·)` to `(N, ·)`, iterate token-chunks,
  accumulate `Σ KL` and a token count, return `Σ / count`. **Exact up to float accumulation
  order** — chunked equals unchunked within `torch.allclose` tolerance, not bit-for-bit (sum
  reduction is reordered); bounds the transient to `(chunk, vocab)`.
  At vocab 47k × fp32 × ~3 softmax temporaries, `chunk_size=256 ≈ 144 MB`; tunable upward.
  Loops on the input's device — pure, no `.cuda()`.
- **`recipes/__init__.py` (`Recipe` dataclass)** — add `lens_max_tokens: int | None = None`.
  When set, the Recorder caps **each sequence to its first `lens_max_tokens` positions** —
  i.e. slices the sequence dimension per-sequence (`residual[:, :lens_max_tokens, :]`, and the
  matching slice of `final_logits`), *not* a total-tokens-across-batch budget — before calling
  the primitive (opt-in cost lever for the §10 budget; `None` = all tokens = exact). It is a
  `Recipe` field because the recipe already owns `activation_diagnostics`.
- **`recorder/live.py`** — wrap the `logit_lens_kl` dispatch in an OOM guard catching
  `torch.cuda.OutOfMemoryError` and `RuntimeError` whose message contains `"out of memory"`.
  On OOM: `torch.cuda.empty_cache()`, emit a WARN, **skip that emission and keep the run
  alive** (rather than propagating and killing training). Pass `lens_max_tokens` (from the
  recipe) into the call. `chunk_size` is **not** plumbed or exposed for this patch — the
  Recorder relies on the `core` default (`256`); it stays an internal memory-bounding detail
  with a safe default, tunable only by direct callers of the primitive.

**Dropped:** `lens_layers`. Which layers the lens runs on is governed by the recipe's
`.*\.layers\.\d+$` hook pattern + the v0.9.1 dispatcher filter; a separate stride knob is a
v1.0 surface concern, not a correctness fix.

**Tests.**
- `tests/core/test_lens.py`: chunked output equals unchunked (single-shot) output within
  `atol` across several `chunk_size` values, including `chunk_size=1` and `chunk_size > N`.
- `tests/recorder/`: a monkeypatched lens that raises `torch.cuda.OutOfMemoryError` →
  the run survives, a WARN is logged, and no lens tag is emitted for that step.
- `lens_max_tokens` honored: with it set below seq length, the lens sees only that many tokens.

---

## #3 — `output_attentions=True` injection breaks wrapped models

**Problem.** `recorder/live.py:427-435` installs a forward-pre-hook on the passed model that
injects `output_attentions=True` into `kwargs`. A thin wrapper whose
`forward(input_ids, labels)` lacks `**kwargs` raises `TypeError` (the reporter had to hand the
Recorder the inner `LlamaForCausalLM`).

**Fix (decided: set on config, not forward kwargs).**

- **`recorder/live.py`** — replace the kwarg-injecting forward-pre-hook with setting
  `config.output_attentions = True` on the resolved HF config at attach, reusing the same
  config resolution the v0.9.1 SDPA warn already performs near `live.py:256` (handles
  `config` / `text_config`). **Verified honored at forward time in transformers 5.9.0**: with
  `config.output_attentions=True` and no forward kwarg, an eager Llama populates
  `outputs.attentions` (see design discussion — control with the kwarg and the default both
  behave as expected).
  - **Detach idempotency:** capture the *original* `config.output_attentions` value at attach
    (it may already be `True`) into an instance attribute and restore exactly that value on
    detach — never blindly `False`.
  - **Restore on partial attach (load-bearing):** `attach()` is long and has fallible steps
    (strict module-resolution, SAE checkpoint loading). To guarantee a failed attach never
    leaves the user's `config` mutated, set `config.output_attentions` as the **final statement
    of `attach()`**, after all fallible work — `attach()` runs no forward pass, so nothing
    downstream needs the flag set earlier. Any failure therefore occurs *before* the mutation
    and cannot pollute the config. `detach()` restores the captured original via a shared
    `_restore_output_attentions()` helper.
  - If no HF config is reachable (truly non-HF model), **skip silently** rather than inject —
    a model without an HF config won't honor the attribute or the kwarg meaningfully, and
    skipping avoids the `TypeError`.
- **Eager:** unchanged from v0.9.1 — we **warn**, we do not **force** eager. SDPA/flash drop
  per-head weights even with `output_attentions=True`; forcing eager would change training
  numerics and speed without consent. The existing attach-time WARN already signals this.

**Tests.**
- `tests/recorder/`: a wrapper module whose `forward(input_ids, labels)` lacks `**kwargs` —
  previously raised on attach — now attaches cleanly and captures attention under eager.
- The config's `output_attentions` is restored to its original value after `detach()`
  (test both original-`False` and original-`True`).
- Restore-on-partial-attach: force `attach()` to raise (e.g. a failing SAE checkpoint load on a
  recipe that also requests `attention_pattern_entropy`) and assert `config.output_attentions`
  was **never** set to `True` (stays at its original value) — the mutation-last ordering means a
  failed attach never touches the config.

---

## #7 — `circuitry --version`

**Problem.** `cli/main.py` exposes no `--version`, forcing an `importlib.metadata` workaround.

**Fix.** On the top-level parser (`cli/main.py:39`), add
`parser.add_argument("--version", action="version", version=f"circuitry {__version__}")`,
importing `__version__` from `circuitry`. The `version` action short-circuits and exits before
the required-subcommand check, so `circuitry --version` works without a subcommand.

**Test.** `tests/`: invoking the CLI with `--version` prints `circuitry <version>` and exits 0.

---

## #8 — clarify `scan` vs. live `report`

**Problem.** `report --run` works directly on a live run's `metrics.jsonl` (no `scan`, no
`findings.json`), but that path isn't documented alongside the retrospective
`scan → findings.json → report` flow (README:34-38, `docs/design.md`).

**Fix (docs only).** Add a short clarifying note near README:34-38 (and the corresponding
`docs/design.md` workflow section) stating the two entry points:
1. **Live:** `Recorder` writes `metrics.jsonl`; `circuitry report --run <dir>` renders it
   directly — no `scan` step.
2. **Retrospective:** `circuitry scan` reads saved checkpoints → `findings.json` →
   `circuitry report`.

No code change.

---

## Release mechanics

- Version bump `0.9.1 → 0.9.2` in: `pyproject.toml`, `src/circuitry/__init__.py`,
  `tests/test_public_api.py`.
- `CHANGELOG.md`: new `[0.9.2]` section — **Fixed** (#1, #2, #3), **Added** (#7), **Docs** (#8).
  No breaking changes (the #1 softmax path is unchanged within fp tolerance; #3 is
  internal-mechanism only).
- Full test suite green (currently 207); CI layering + no-`.cuda()`-in-core checks pass.
- Tag + GitHub Release after merge, consistent with prior releases.

## Invariants & risk

- **Layering:** all new logic respects `core/ ↮ recorder/recipes/writers/cli`. Chunking and
  normalization stay in `core/`; warnings, OOM-guarding, config mutation, and `lens_max_tokens`
  plumbing stay in `recorder/`. `Recipe` field addition is in `recipes/`.
- **No `.cuda()` in core:** chunk loop and normalization operate on the input's device.
- **Backward compatibility:** #1 leaves softmax results unchanged within fp tolerance; #2 is
  numerically exact (chunking) with new opt-in knobs defaulting to prior behavior; #3 changes
  only the internal capture mechanism; #7/#8 are additive. No public-signature breaks.

## Out of scope

#5 (`disable`/`only`), #6 (report summary + `circuitry compare`), `lens_layers`. These inform
the v1.0 surface and get their own spec.
