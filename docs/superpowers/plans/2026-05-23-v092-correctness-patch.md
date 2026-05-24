# v0.9.2 Correctness Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v0.9.2 fixing three confirmed correctness bugs (#1 attention-entropy normalization, #2 logit_lens_kl OOM, #3 output_attentions on wrapped models) plus two quick wins (#7 `--version`, #8 docs), from the approved spec.

**Architecture:** Pure fixes in `core/` (normalization, token-chunking) stay device-deterministic with no I/O; runtime concerns (one-time warns, OOM survival, config mutation, `lens_max_tokens` plumbing) live in `recorder/`; one additive `Recipe` field. No public-signature breaks; softmax entropy unchanged within fp tolerance; chunked KL exact within `allclose`.

**Tech Stack:** Python 3.12, PyTorch, pytest. Spec: `docs/superpowers/specs/2026-05-23-v092-correctness-patch-design.md`.

**Invariants (CI-enforced — do not violate):**
- `core/` MUST NOT import from `recorder/`, `recipes/`, `writers/`, `cli/`.
- No `.cuda()` in `core/` — operate on the input tensor's device.
- Use full venv paths: `venv/bin/pytest`, `venv/bin/python`.
- Branch is `fix/v0.9.2-correctness-patch`. Commit after each task.

---

## Task 1: #7 — `circuitry --version`

**Files:**
- Modify: `src/circuitry/cli/main.py:38-40`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_cli_version_flag_prints_version_and_exits_zero():
    from circuitry import __version__
    out = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "--version"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert __version__ in out
    assert out.strip().startswith("circuitry ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_cli.py::test_cli_version_flag_prints_version_and_exits_zero -v`
Expected: FAIL — argparse exits non-zero ("invalid choice" / "required: cmd") because `--version` isn't defined.

- [ ] **Step 3: Implement**

In `src/circuitry/cli/main.py`, add the import near the top (after line 8 `from circuitry.recipes import list_recipes`):

```python
from circuitry import __version__
```

Then in `main()`, immediately after `parser = argparse.ArgumentParser(prog="circuitry")` (line 39), add:

```python
    parser.add_argument(
        "--version", action="version", version=f"circuitry {__version__}",
    )
```

(Verified: the `version` action short-circuits and exits 0 before the `required=True` subparser check.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (all CLI tests).

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/cli/main.py tests/test_cli.py
git commit -m "feat(cli): add circuitry --version flag

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: #8 — document `scan` vs. live `report`

**Files:**
- Modify: `README.md` (after the scan/report code block, around line 38)
- Modify: `docs/design.md` (CLI/workflow section mentioning `report`)

Docs-only; no test. Verify with a grep.

- [ ] **Step 1: Read the README anchor**

Run: `venv/bin/python -c "print(open('README.md').read()[:2000])"` and locate the block:

```
Retrospective scan + report from saved checkpoints:

circuitry scan   --run runs/my_run --recipe llm
circuitry report --run runs/my_run
```

- [ ] **Step 2: Add the clarifying note in README.md**

Immediately after that fenced code block (before the next section), insert:

```markdown
`circuitry report` has two entry points:

- **Live run** — the `Recorder` writes `metrics.jsonl` during training; run
  `circuitry report --run <dir>` directly on it. No `scan`, no `findings.json`.
- **Retrospective** — `circuitry scan` reads saved checkpoints and writes
  `findings.json`, which `circuitry report` then renders.
```

- [ ] **Step 3: Add a one-line equivalent to docs/design.md**

Locate the `report`/`scan` workflow section: `grep -n "circuitry report\|findings.json\|scan" docs/design.md`. In that section add the sentence:

```markdown
`report` accepts either a live `metrics.jsonl` (written by the Recorder, no
`scan` step) or a retrospective `findings.json` produced by `scan`.
```

- [ ] **Step 4: Verify**

Run: `grep -n "No \`scan\`, no \`findings.json\`" README.md && grep -n "live \`metrics.jsonl\`" docs/design.md`
Expected: both match.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/design.md
git commit -m "docs: clarify report works on live metrics.jsonl vs scan findings.json

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: #1 (core) — normalize attention rows before entropy

**Files:**
- Modify: `src/circuitry/core/attention.py:56-62`
- Test: `tests/core/test_attention.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_attention.py` (`math` and `attention_pattern_entropy` are already imported):

```python
def test_entropy_normalizes_unnormalized_rows():
    """Rows that don't sum to 1 (sigmoid / linear attention) must be normalized
    before entropy, so uniform-but-unnormalized weights give ln(seq), not a
    value inflated by the total mass."""
    seq, n_heads = 8, 2
    # Every key weighted 0.5 → each row sums to 4.0, not 1.0.
    attn = torch.full((1, n_heads, seq, seq), 0.5)
    ents = attention_pattern_entropy(attn)
    for e in ents:
        assert e == pytest.approx(math.log(seq), abs=1e-5)


def test_entropy_handles_fully_masked_zero_rows_without_nan():
    """A fully-masked query row (all zeros, sums to 0) must not produce NaN;
    the eps-clamped divide leaves it all-zero → entropy 0."""
    seq, n_heads = 5, 1
    attn = torch.zeros(1, n_heads, seq, seq)
    attn[0, 0, 0, 0] = 1.0  # row 0 valid (one-hot), rows 1..4 fully masked
    ents = attention_pattern_entropy(attn)
    assert not math.isnan(ents[0])
    assert ents[0] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/core/test_attention.py::test_entropy_normalizes_unnormalized_rows -v`
Expected: FAIL — raw entropy of 0.5-weighted rows is `seq*0.5*ln2 ≈ 2.77`, not `ln(8) ≈ 2.079`.

- [ ] **Step 3: Implement normalization**

In `src/circuitry/core/attention.py`, replace the body of `attention_pattern_entropy` (lines 58-62) with:

```python
    t = _ensure_4d(_as_tensor(attn_pattern)).detach().to(torch.float32)
    # Normalize each query row to a probability distribution before computing
    # entropy. Softmax rows already sum to 1 (no-op within fp tolerance); for
    # sigmoid / linear attention whose weights don't sum to 1 this makes the
    # entropy normalization-invariant (a pure concentration measure) so it is
    # comparable across attention variants. A fully-masked row (sums to 0) stays
    # all-zero after the eps-clamped divide, and xlogy(0, 0) = 0 → entropy 0.
    row_sum = t.sum(dim=-1, keepdim=True)
    p = t / row_sum.clamp_min(torch.finfo(t.dtype).eps)
    # xlogy(p, p) = p * log(p), and xlogy(0, 0) = 0 (no NaN).
    plogp = torch.special.xlogy(p, p)
    entropy = -plogp.sum(dim=-1)  # (batch, n_heads, seq)
    return entropy.mean(dim=(0, 2)).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/core/test_attention.py -v`
Expected: PASS — including the pre-existing `test_entropy_of_uniform_attention_is_log_seq` and `test_entropy_of_one_hot_attention_is_zero` (softmax inputs unchanged within tolerance).

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/attention.py tests/core/test_attention.py
git commit -m "fix(core): normalize attention rows before entropy

attention_pattern_entropy now divides each query row by its key-axis sum
(eps-clamped) before computing entropy, making it normalization-invariant for
sigmoid / linear attention. Softmax rows sum to 1, so the result is unchanged
within fp tolerance. Fully-masked zero rows yield entropy 0, no NaN.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: #1 (recorder) — warn once when attention rows aren't normalized

**Files:**
- Modify: `src/circuitry/recorder/live.py:149` (init flag) and `:778-789` (dispatch)
- Test: `tests/recorder/test_recorder_step.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/recorder/test_recorder_step.py` (use the file's existing imports; if `logging`, `types`, `torch`, `nn` aren't imported there, add them at top):

```python
def test_attention_entropy_warns_once_on_unnormalized_rows(tmp_path, caplog):
    """When captured attention rows don't sum to 1 (sigmoid-like), the Recorder
    warns once that entropy is over the normalized shape."""
    import logging
    import types
    import torch
    import torch.nn as nn

    from circuitry import HookPoint, Recipe, Recorder, TensorSource

    d_model, n_heads = 8, 2

    class _Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, output_attentions: bool = False):
            B, T, D = x.shape
            H = n_heads
            q = self.q_proj(x).view(B, T, H, D // H).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, D // H).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (D // H) ** 0.5
            attn = torch.sigmoid(scores)  # rows do NOT sum to 1
            out = (attn @ q)  # shape only; value irrelevant
            out = out.transpose(1, 2).reshape(B, T, D)
            if output_attentions:
                return out, attn
            return out

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attn()

        def forward(self, x, output_attentions: bool = False):
            return self.self_attn(x, output_attentions=output_attentions)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(
                output_attentions=False, _attn_implementation="eager")
            self.layers = nn.ModuleList([_Block()])

        def forward(self, x, output_attentions=None):
            # Dual-mode so this test passes regardless of Task ordering: under
            # the (pre-Task-7) kwarg injection it receives output_attentions as
            # a kwarg; under the (Task-7) config mechanism the kwarg is absent
            # and it falls back to config.output_attentions.
            if output_attentions is None:
                output_attentions = self.config.output_attentions
            for b in self.layers:
                x = b(x, output_attentions=output_attentions)
            return x

    model = _M()
    recipe = Recipe(
        name="entropy_warn",
        hook_points=[HookPoint(source=TensorSource.OUTPUT,
                               pattern=r"layers\.\d+\.self_attn$")],
        activation_diagnostics=["attention_pattern_entropy"],
    )
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec.attach()
    model(torch.randn(1, 4, d_model))
    rec.step(0)
    rec.detach()

    warns = [r for r in caplog.records if "do not sum to 1" in r.getMessage()]
    assert len(warns) == 1, [r.getMessage() for r in caplog.records]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/recorder/test_recorder_step.py::test_attention_entropy_warns_once_on_unnormalized_rows -v`
Expected: FAIL — no `do not sum to 1` warning emitted yet. (The dual-mode model captures attention under both the current kwarg-injection and the Task-7 config mechanism, so this task is self-contained and order-independent.)

- [ ] **Step 3: Implement the one-time warn**

In `src/circuitry/recorder/live.py`, in `__init__` next to `self._main_pass_attn` (line 149), add:

```python
        self._warned_unnormalized_attn = False
```

Then in the `attention_pattern_entropy` dispatch (lines 778-789), change the loop body to check row-sums once:

```python
            if name == "attention_pattern_entropy":
                from circuitry.core.attention import (
                    attention_pattern_entropy as _ape,
                )
                for mn, attn in self._main_pass_attn.items():
                    if not self._warned_unnormalized_attn:
                        rs = attn.detach().to(torch.float32).sum(dim=-1)
                        dev = (rs - 1.0).abs().max().item()
                        if dev > 1e-3:
                            logger.warning(
                                "circuitry: attention_pattern_entropy rows do "
                                "not sum to 1 (max deviation %.3g) — entropy is "
                                "computed over the normalized attention shape; "
                                "total attention mass is discarded. Values are "
                                "comparable across attention variants but are "
                                "not raw softmax entropy.", dev,
                            )
                            self._warned_unnormalized_attn = True
                    ents = _ape(attn)
                    for i, e in enumerate(ents):
                        self._writer.add_scalar(
                            f"activation/attention_pattern_entropy/{mn}/head_{i}",
                            e, ctx.step,
                        )
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/recorder/test_recorder_step.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/live.py tests/recorder/test_recorder_step.py
git commit -m "feat(recorder): warn once when attention_pattern_entropy rows aren't normalized

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: #2 (core) — token-chunk `logit_lens_kl`

**Files:**
- Modify: `src/circuitry/core/lens.py:26-78`
- Test: `tests/core/test_lens.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_lens.py`:

```python
def test_chunking_matches_single_shot():
    """Token-chunked KL equals the single-shot result within allclose (sum
    reduction is reordered, not bit-for-bit)."""
    torch.manual_seed(5)
    d_model, vocab = 8, 32
    residual = torch.randn(2, 9, d_model)
    W = torch.randn(d_model, vocab)
    final_logits = torch.randn(2, 9, vocab)
    ref = logit_lens_kl(residual, W, final_logits, chunk_size=100_000)
    for cs in (1, 3, 7, 18, 1000):
        got = logit_lens_kl(residual, W, final_logits, chunk_size=cs)
        assert got == pytest.approx(ref, abs=1e-5), f"chunk_size={cs}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/core/test_lens.py::test_chunking_matches_single_shot -v`
Expected: FAIL — `logit_lens_kl() got an unexpected keyword argument 'chunk_size'`.

- [ ] **Step 3: Implement chunking**

In `src/circuitry/core/lens.py`, change the signature (line 26-32) to add `chunk_size`:

```python
def logit_lens_kl(
    residual: Any,
    unembed: Any,
    final_logits: Any,
    *,
    layer_norm: Callable[[Tensor], Tensor] | None = None,
    chunk_size: int = 256,
) -> float:
```

Then replace the tail of the function (current lines 66-78, from `res_f32 = ...` onward) with:

```python
    res_f32 = res.detach().to(torch.float32)
    if layer_norm is not None:
        res_f32 = layer_norm(res_f32)
    W_f32 = proj_W.detach().to(torch.float32)
    fl_f32 = fl.detach().to(torch.float32)

    # Chunk over the flattened token axis so the (tokens, vocab) lens-logits
    # transient never materializes for the whole batch at once. Exact up to
    # float accumulation order. Stays on the input's device (no .cuda()).
    res_flat = res_f32.reshape(-1, res_f32.shape[-1])  # (N, d_model)
    fl_flat = fl_f32.reshape(-1, fl_f32.shape[-1])      # (N, vocab)
    n = res_flat.shape[0]
    if n == 0:
        return 0.0
    kl_sum = res_flat.new_zeros(())
    for start in range(0, n, max(1, chunk_size)):
        r = res_flat[start:start + chunk_size]
        f = fl_flat[start:start + chunk_size]
        lens_logits = r @ W_f32
        log_q = torch.log_softmax(lens_logits, dim=-1)  # lens distribution
        log_p = torch.log_softmax(f, dim=-1)             # final distribution
        q = log_q.exp()
        kl_sum = kl_sum + (q * (log_q - log_p)).sum(dim=-1).sum()
    return float((kl_sum / n).item())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/core/test_lens.py -v`
Expected: PASS — including all pre-existing tests (default `chunk_size=256` is exact for their small inputs).

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/core/lens.py tests/core/test_lens.py
git commit -m "fix(core): token-chunk logit_lens_kl to bound peak memory

Iterate the flattened token axis in chunk_size (default 256) blocks,
accumulating KL, so the (tokens, vocab) lens-logits transient never
materializes for the whole batch. Exact up to float accumulation order;
device-deterministic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: #2 (recorder) — `lens_max_tokens` field + OOM survival

**Files:**
- Modify: `src/circuitry/recipes/__init__.py:26` (add field)
- Modify: `src/circuitry/recorder/live.py:694-712` (slice + OOM guard)
- Test: `tests/recorder/test_recorder_step.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/recorder/test_recorder_step.py`:

```python
def _lens_model_and_recipe(lens_max_tokens=None):
    import torch.nn as nn
    from circuitry import HookPoint, Recipe, TensorSource

    d_model, vocab = 8, 16

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            return x + self.lin(x)

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block(), _Block()])
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, x):
            for b in self.layers:
                x = b(x)
            return self.lm_head(self.ln_f(x))

    recipe = Recipe(
        name=f"lens_{lens_max_tokens}",
        hook_points=[HookPoint(source=TensorSource.OUTPUT,
                               pattern=r"layers\.\d+$")],
        activation_diagnostics=["logit_lens_kl"],
        lens_max_tokens=lens_max_tokens,
    )
    return _Tiny(), recipe, d_model


def test_lens_max_tokens_caps_sequence_dim(tmp_path, monkeypatch):
    import torch
    import circuitry.core.lens as lens_mod
    from circuitry import Recorder

    seen = {}
    real = lens_mod.logit_lens_kl

    def spy(residual, *a, **k):
        seen["seq"] = residual.shape[1]
        return real(residual, *a, **k)

    monkeypatch.setattr(lens_mod, "logit_lens_kl", spy)
    model, recipe, d_model = _lens_model_and_recipe(lens_max_tokens=2)
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 6, d_model))
    rec.step(0)
    rec.detach()
    assert seen["seq"] == 2


def test_logit_lens_kl_oom_is_survived(tmp_path, monkeypatch, caplog):
    import logging
    import torch
    import circuitry.core.lens as lens_mod
    from circuitry import Recorder

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate ...")

    monkeypatch.setattr(lens_mod, "logit_lens_kl", boom)
    model, recipe, d_model = _lens_model_and_recipe()
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec.attach()
    model(torch.randn(1, 4, d_model))
    rec.step(0)        # must NOT raise
    rec.detach()
    assert any("out of memory" in r.getMessage().lower()
               for r in caplog.records)
    out = (tmp_path / "metrics.jsonl").read_text()
    assert "activation/logit_lens_kl/layers.0" not in out  # skipped


def test_non_oom_runtimeerror_still_propagates(tmp_path, monkeypatch):
    import torch
    import pytest
    import circuitry.core.lens as lens_mod
    from circuitry import Recorder

    def boom(*a, **k):
        raise RuntimeError("some unrelated bug")

    monkeypatch.setattr(lens_mod, "logit_lens_kl", boom)
    model, recipe, d_model = _lens_model_and_recipe()
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 4, d_model))
    with pytest.raises(RuntimeError, match="some unrelated bug"):
        rec.step(0)
    rec.detach()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/recorder/test_recorder_step.py -k "lens_max_tokens or oom or non_oom" -v`
Expected: FAIL — `Recipe.__init__() got an unexpected keyword argument 'lens_max_tokens'`.

- [ ] **Step 3a: Add the Recipe field**

In `src/circuitry/recipes/__init__.py`, after line 26 (`induction_probe_seq_len: int = 25`), add:

```python
    lens_max_tokens: int | None = None
```

- [ ] **Step 3b: Slice + OOM-guard the lens dispatch**

In `src/circuitry/recorder/live.py`, replace the block at lines 694-712 (from `_last_name, last_x = block_outputs[-1]` through the `add_scalar` for `logit_lens_kl`) with:

```python
                _last_name, last_x = block_outputs[-1]
                max_tok = self.recipe.lens_max_tokens
                with torch.inference_mode():
                    last_f32 = last_x.detach().to(torch.float32)
                    if max_tok is not None:
                        last_f32 = last_f32[:, :max_tok, :]
                    ln = self._lens_meta.layer_norm
                    last_normed = ln(last_f32) if ln is not None else last_f32
                    W = _W_raw.to(torch.float32)
                    # unembed for HF is (vocab, d_model); transpose if needed.
                    if W.shape[-1] == last_normed.shape[-1]:
                        final_logits = last_normed @ W.t()
                    else:
                        final_logits = last_normed @ W
                for mod_name, x in block_outputs:
                    if max_tok is not None:
                        x = x[:, :max_tok, :]
                    try:
                        kl = _llk(
                            x, self._lens_meta.unembed, final_logits,
                            layer_norm=self._lens_meta.layer_norm,
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if (not isinstance(e, torch.cuda.OutOfMemoryError)
                                and "out of memory" not in str(e).lower()):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(
                            "circuitry: logit_lens_kl ran out of memory on %s — "
                            "skipping this layer's emission for this step. Set "
                            "recipe.lens_max_tokens to cap the lens cost. (%s)",
                            mod_name, e,
                        )
                        continue
                    self._writer.add_scalar(
                        f"activation/logit_lens_kl/{mod_name}", kl, ctx.step,
                    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/recorder/test_recorder_step.py tests/recorder/test_recorder_attach.py -v`
Expected: PASS (including the pre-existing `test_logit_lens_kl_is_dispatched_per_block` and `..._skips_same_dmodel_submodules`).

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recipes/__init__.py src/circuitry/recorder/live.py tests/recorder/test_recorder_step.py
git commit -m "feat(recorder): survive logit_lens_kl OOM + add lens_max_tokens cap

A per-layer lens OOM now empties the cache, warns, skips that emission, and
keeps the run alive instead of crashing training. Non-OOM RuntimeErrors still
propagate. New Recipe.lens_max_tokens caps each sequence to its first N
positions as a cost lever (None = all tokens = exact).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: #3 — enable `output_attentions` via config, not forward kwargs

**Files:**
- Modify: `src/circuitry/recorder/live.py` — `__init__` (~149), the injection block (422-435), end of `attach()`, and `detach()` (503-510)
- Test: `tests/recorder/test_recorder_attach.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/recorder/test_recorder_attach.py`:

```python
def _attn_entropy_model(output_attentions_default=False):
    """HF-like model whose forward() has NO **kwargs and reads
    config.output_attentions (not a forward kwarg) — exactly the wrapper shape
    that broke the old kwarg-injection."""
    import torch
    import types

    d_model, n_heads = 8, 2

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, output_attentions: bool = False):
            B, T, D = x.shape
            q = self.q_proj(x).view(B, T, n_heads, D // n_heads).transpose(1, 2)
            attn = (q @ q.transpose(-2, -1)).softmax(dim=-1)
            out = (attn @ q).transpose(1, 2).reshape(B, T, D)
            if output_attentions:
                return out, attn
            return out

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attn()

        def forward(self, x, output_attentions: bool = False):
            return self.self_attn(x, output_attentions=output_attentions)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                output_attentions=output_attentions_default,
                _attn_implementation="eager")
            self.layers = nn.ModuleList([_Block()])

        def forward(self, x):  # NO **kwargs, NO output_attentions param
            oa = self.config.output_attentions
            for b in self.layers:
                x = b(x, output_attentions=oa)
            return x

    return _M(), d_model


def _entropy_recipe(name):
    return Recipe(
        name=name,
        hook_points=[HookPoint(source=TensorSource.OUTPUT,
                               pattern=r"layers\.\d+\.self_attn$")],
        activation_diagnostics=["attention_pattern_entropy"],
    )


def test_attach_does_not_break_kwargless_wrapper_forward(tmp_path):
    """Old behavior injected output_attentions=True into the forward kwargs,
    raising TypeError on a forward() without **kwargs. The config approach must
    let a normal forward run cleanly and still capture attention."""
    import torch
    model, d_model = _attn_entropy_model()
    rec = Recorder(model, run_dir=tmp_path, recipe=_entropy_recipe("w1"),
                   writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 4, d_model))  # must NOT raise TypeError
    rec.step(0)
    rec.detach()
    out = (tmp_path / "metrics.jsonl").read_text()
    assert "activation/attention_pattern_entropy/layers.0.self_attn/head_0" in out


def test_output_attentions_restored_on_detach_from_false(tmp_path):
    model, _ = _attn_entropy_model(output_attentions_default=False)
    rec = Recorder(model, run_dir=tmp_path, recipe=_entropy_recipe("w2"),
                   writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    assert model.config.output_attentions is True
    rec.detach()
    assert model.config.output_attentions is False


def test_output_attentions_restored_on_detach_from_true(tmp_path):
    model, _ = _attn_entropy_model(output_attentions_default=True)
    rec = Recorder(model, run_dir=tmp_path, recipe=_entropy_recipe("w3"),
                   writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    assert model.config.output_attentions is True
    rec.detach()
    assert model.config.output_attentions is True  # original preserved


def test_failed_attach_never_mutates_config(tmp_path, monkeypatch):
    """Config is set as the final attach() step, so a failure earlier (here, a
    failing SAE load) must leave config.output_attentions untouched."""
    import circuitry.sae.loader as sae_loader

    def boom(*a, **k):
        raise RuntimeError("sae load failed")

    monkeypatch.setattr(sae_loader, "load_sae", boom)
    model, _ = _attn_entropy_model(output_attentions_default=False)
    recipe = Recipe(
        name="w4",
        hook_points=[HookPoint(source=TensorSource.OUTPUT,
                               pattern=r"layers\.\d+\.self_attn$")],
        activation_diagnostics=["attention_pattern_entropy"],
        sae_checkpoints={r"layers\.\d+\.self_attn$": ("rel", "id")},
    )
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe,
                   writer="jsonl", every_n_steps=1, strict=False)
    with pytest.raises(RuntimeError, match="sae load failed"):
        rec.attach()
    assert model.config.output_attentions is False  # never mutated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/recorder/test_recorder_attach.py -k "kwargless or restored or never_mutates" -v`
Expected: FAIL — `test_attach_does_not_break_kwargless_wrapper_forward` raises `TypeError` (old kwarg injection); restore tests fail (config never set).

- [ ] **Step 3a: Add restore state + helper**

In `src/circuitry/recorder/live.py` `__init__`, next to `self._main_pass_attn` (line 149), add:

```python
        self._output_attentions_restore: list[tuple[Any, Any]] = []
```

Add a helper method (place it just before `def detach`, around line 502):

```python
    def _set_output_attentions_true(self) -> None:
        """Enable per-head attention output via the HF config (not a forward
        kwarg, which breaks wrappers whose forward lacks **kwargs). Records the
        original value(s) so detach() can restore exactly. Call LAST in
        attach() so a failed attach never leaves the config mutated."""
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        for source in (cfg, text_cfg):
            if source is None:
                continue
            self._output_attentions_restore.append(
                (source, getattr(source, "output_attentions", False))
            )
            source.output_attentions = True

    def _restore_output_attentions(self) -> None:
        for source, original in self._output_attentions_restore:
            source.output_attentions = original
        self._output_attentions_restore.clear()
```

- [ ] **Step 3b: Remove the kwarg-injection pre-hook**

In `attach()`, delete the `_inject_kwargs` forward-pre-hook (lines 428-435) — keep the comment updated and keep the capture-hook loop (437-459 unchanged). Replace lines 422-435 with:

```python
        # If attention_pattern_entropy is requested, capture attn_weights from
        # matched self_attn modules during the main forward. Per-head weights
        # are enabled via config.output_attentions (set at the END of attach;
        # see _set_output_attentions_true) rather than by injecting an
        # output_attentions=True forward kwarg — the kwarg path raises TypeError
        # on wrapper models whose forward() lacks **kwargs.
        if "attention_pattern_entropy" in self.recipe.activation_diagnostics:
```

Concretely: delete the 8 lines `def _inject_kwargs ... self._hook_handles.append(handle)` (the pre-hook), and update the comment. Do NOT touch the existing `attn_modules: list[str] = []` line (currently 437) or the `for idx, hp in enumerate(...)` / `_mk_attn_capture` registration loop (438-459) — they stay exactly as-is, directly below the `if`.

- [ ] **Step 3c: Enable the config flag as the final attach() step**

At the very END of `attach()` (after the SAE-loading block that ends near line 486), add:

```python
        # Set last: attach() runs no forward, so nothing above needs the flag;
        # putting it last guarantees a failed attach never mutates user config.
        if "attention_pattern_entropy" in self.recipe.activation_diagnostics:
            self._set_output_attentions_true()
```

- [ ] **Step 3d: Restore on detach**

In `detach()` (lines 503-510), add a restore call (before or after hook removal):

```python
    def detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._restore_output_attentions()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/recorder/test_recorder_attach.py -v`
Expected: PASS (including the pre-existing SDPA-warn tests, which are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/circuitry/recorder/live.py tests/recorder/test_recorder_attach.py
git commit -m "fix(recorder): enable output_attentions via config, not forward kwargs

Setting config.output_attentions=True (verified honored in transformers 5.9.0)
replaces the forward-pre-hook kwarg injection that raised TypeError on wrapper
models whose forward() lacks **kwargs. The flag is set as the final attach()
step (so a failed attach never mutates user config) and the original value is
restored on detach.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: version bump + CHANGELOG

**Files:**
- Modify: `pyproject.toml`, `src/circuitry/__init__.py`, `tests/test_public_api.py`, `CHANGELOG.md`

- [ ] **Step 1: Locate the version strings**

Run: `grep -rn "0\.9\.1" pyproject.toml src/circuitry/__init__.py tests/test_public_api.py`
Expected: one `0.9.1` literal in each.

- [ ] **Step 2: Bump to 0.9.2**

Edit each occurrence `0.9.1` → `0.9.2` in `pyproject.toml`, `src/circuitry/__init__.py`, `tests/test_public_api.py`.

- [ ] **Step 3: Add the CHANGELOG section**

In `CHANGELOG.md`, add a new section immediately above the existing `## [0.9.1]` heading (preserving the file's existing format):

```markdown
## [0.9.2] — 2026-05-23

### Fixed
- **`attention_pattern_entropy` is now normalization-invariant.** Each query row
  is divided by its key-axis sum before the entropy, so the metric is comparable
  across attention variants (softmax / sigmoid / linear). Softmax rows sum to 1,
  so existing values are unchanged within fp tolerance; fully-masked rows yield 0.
  The Recorder warns once when captured rows don't sum to 1.
- **`logit_lens_kl` no longer OOMs the run.** The KL is computed in token chunks
  (`chunk_size`, default 256) so the `(tokens, vocab)` lens-logits transient stays
  bounded; a per-layer OOM now empties the cache, warns, skips that emission, and
  keeps training alive instead of crashing. New `Recipe.lens_max_tokens` caps each
  sequence to its first N positions as a cost lever (`None` = all tokens = exact).
- **`output_attentions` capture no longer breaks wrapped models.** Per-head
  attention weights are enabled via `config.output_attentions` instead of a
  forward-kwarg injection that raised `TypeError` on wrappers whose `forward()`
  lacks `**kwargs`. Set as the final `attach()` step (a failed attach never mutates
  the config) and restored on `detach()`.

### Added
- **`circuitry --version`** prints the installed version and exits.

### Docs
- Clarified that `circuitry report` runs on a live `metrics.jsonl` (no `scan`)
  as well as on a retrospective `findings.json`.
```

- [ ] **Step 4: Run the full suite**

Run: `venv/bin/pytest -q`
Expected: PASS (was 207; new tests added across Tasks 1, 3, 4, 5, 6, 7).
Also run the layering guard: `venv/bin/pytest tests/test_layering.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/circuitry/__init__.py tests/test_public_api.py CHANGELOG.md
git commit -m "chore(release): v0.9.2 — entropy normalization, lens OOM survival, wrapper attn fix

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification (run before declaring done)

- [ ] `venv/bin/pytest -q` — all green.
- [ ] `venv/bin/pytest tests/test_layering.py tests/test_public_api.py -v` — invariants + version.
- [ ] `grep -rn "_inject_kwargs" src/` — returns nothing (kwarg injection fully removed).
- [ ] `venv/bin/python -m circuitry.cli.main --version` — prints `circuitry 0.9.2`.
- [ ] `git log --oneline e1bfbd5..HEAD` — one commit per task, clean messages.

## Out of scope (deferred to v1.0)

#5 (`disable`/`only`), #6 (report summary + `circuitry compare`), `lens_layers`.
